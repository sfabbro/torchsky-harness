"""Catalogue-free coarse registration of exposures that start far from the sky (--coarse-reg xcorr), before the registration pre-pass.

WHY. The pre-pass (train.register_prepass) is gradient steps on per-exposure translations; it cannot capture a pointing error of several
detector pixels (HST exposures added to a JWST field start 0.4 to 0.5 arcsec off: the pre-pass moved them by 5 mas). Here each selected exposure
is registered against the starting sky by one masked normalised cross-correlation of high-passed images, by FFT, within a search radius.
No source detection and no catalogue: only pixels.

METHOD (per exposure cut)
 1. The starting sky (built from the reference instrument's training exposures only, by the inverse-variance coadd data.splat_coadd on the sky grid, header
    positions) of the reference band nearest in wavelength (pivot, ln scale) to the exposure's own band is resampled bilinearly at the exposure's pixel
    positions (translation zero). No kernel rendering and no colour model. Data d(p): pixels of the file, background not removed.
 2. High-pass of both (high_pass): subtract a masked block median (blocks of --coarse-reg-block arcsec, smoothed and bilinearly interpolated),
    asinh stretch (soft scale 3 sigma, sigma = MAD of the high-passed pixels), masked Gaussian smoothing (--coarse-reg-smooth arcsec), zero mean.
 3. Masked normalised cross-correlation (Padfield; six FFT correlation terms, so the denominators use exactly the pixels in the overlap) for all
    integer lags (dx, dy) in pixels (x = column, y = row) with |lag| <= radius; overlap at least min(30 arcsec^2, a quarter of the valid pixels).
 4. Peak = highest correlation inside the radius; second peak = highest other local maximum (5 x 5 pixel neighbourhood) farther than 0.1 arcsec
    (at least 3 pixels) from the peak; ratio = peak / second. Sub-pixel position: parabola through the three correlation values along each axis
    around the peak (vertex clipped to +-0.5 pixel).
 5. An exposure is REFUSED (translation left at its start value, a line in the log) if the peak is below --coarse-reg-min-peak (0.05), the
    ratio is below --coarse-reg-min-ratio (1.5), or the peak lies within one pixel of the search boundary. Both thresholds are those of the older
    stand-alone pairwise tool (ref_xcorr/register_xcorr.py), whose masking, high-pass, overlap rule and peak finding this follows (re-implemented
    here in torch; the sub-pixel step there is an upsampled Fourier transform, here the parabola asked for).

CONVENTIONS (the same as the pre-pass translations, model.affine[:, 4:6] x 0.01 arcsec)
 * Positions are tangent-plane arcsec of the field frame: X toward the WEST, Y toward the NORTH (sky.py / data.py convention); pixel x = column,
   y = row. J = d(X, Y) / d(column, row) is the exposure's mean header Jacobian (arcsec per pixel).
 * The shift t = (tX, tY) [arcsec] is the amount to ADD to the header position (+ shift0) of the exposure's pixels to get the position on the sky
   the starting sky is on: true position = header position + t. It is what model.affine[e, 4:6] x 0.01 holds (model.positions adds it), so the
   pre-pass starts from it and refines it.
 * Derivation: data d(p) = M(x_p + t) with M the sky, m(p) = M(x_p), so d(p) = m(p + delta) with delta = J^-1 t; the correlation
   sum_p d(p) m(p + lag) peaks at lag = delta; t = J delta. Sign check on the synthetic field: injected +0.5 arcsec in X is recovered as +0.5.
 * Units: shifts in the log and records are in mas; peak = normalised correlation (1 = identical), ratio = peak / second peak (99 = no other
   local maximum).
 * The common-translation gauge (project_shift, applied by the pre-pass and the fit) removes the mean translation of the training exposures; it
   acts on the sum, so with the translations of a quarter of the exposures set the whole field is re-centred by that mean in the first pre-pass
   step (relative positions are what the fit uses).

SCATTER WITHIN A VISIT. Exposures of one visit share the guide-star error to first order, so their shifts should agree; the spread of the shifts
of the accepted exposures of each visit (exposure extra["visit_key"], else the dither group) is logged and returned.

APPROXIMATIONS: translation only (no rotation or scale); the mean Jacobian converts pixels to arcsec (scale and rotation vary by < 1e-3 over a
detector); the reference image is that of the nearest reference band and is not blurred to the exposure's own response, so structure that changes with wavelength lowers the peak; the block median is a
block estimate; blocks without enough valid pixels take the median of the other blocks.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F


def _gauss_kernel(s, dev, dt):
    r = int(math.ceil(3 * s))
    g = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=dev, dtype=dt) / s) ** 2)
    return g / g.sum(), r


def masked_smooth(img, valid, s_px):
    """Normalised Gaussian smoothing over the valid pixels (s_px <= 0.05: unchanged)."""
    if s_px <= 0.05:
        return img
    g, r = _gauss_kernel(s_px, img.device, img.dtype)
    w = valid.to(img.dtype)

    def conv(x):
        x = F.conv2d(x[None, None], g.view(1, 1, 1, -1), padding=(0, r))
        return F.conv2d(x, g.view(1, 1, -1, 1), padding=(r, 0))[0, 0]
    return conv(img * w) / conv(w).clamp(min=1e-3)


def block_background(img, valid, blk, min_frac=0.25):
    """Masked block median on blk x blk blocks (blocks with fewer than min_frac valid pixels: median of the other blocks), smoothed by a
    one-block Gaussian, bilinearly interpolated to the pixels (cell-centred)."""
    ny, nx = img.shape
    by, bx = int(math.ceil(ny / blk)), int(math.ceil(nx / blk))
    a = torch.full((by * blk, bx * blk), float("nan"), dtype=img.dtype, device=img.device)
    a[:ny, :nx] = torch.where(valid, img, torch.full_like(img, float("nan")))
    b = a.view(by, blk, bx, blk).permute(0, 2, 1, 3).reshape(by, bx, blk * blk)
    cnt = torch.isfinite(b).sum(-1)
    med = torch.nanmedian(b, dim=-1).values
    ok = (cnt >= min_frac * blk * blk) & torch.isfinite(med)
    if not bool(ok.any()):
        return torch.zeros_like(img)
    med = torch.where(ok, med, med[ok].median())
    med = masked_smooth(med, torch.ones_like(ok), 1.0)
    # cell-centred bilinear interpolation of the block grid to pixel centres
    gy = ((torch.arange(ny, device=img.device, dtype=img.dtype) + 0.5) / blk - 0.5).clamp(0, by - 1)
    gx = ((torch.arange(nx, device=img.device, dtype=img.dtype) + 0.5) / blk - 0.5).clamp(0, bx - 1)
    y0, x0 = gy.floor().long().clamp(max=max(by - 2, 0)), gx.floor().long().clamp(max=max(bx - 2, 0))
    y1, x1 = (y0 + 1).clamp(max=by - 1), (x0 + 1).clamp(max=bx - 1)
    fy, fx = (gy - y0)[:, None], (gx - x0)[None]
    return (med[y0][:, x0] * (1 - fy) * (1 - fx) + med[y0][:, x1] * (1 - fy) * fx + med[y1][:, x0] * fy * (1 - fx) + med[y1][:, x1] * fy * fx)


def high_pass(img, valid, scale, block_arcsec=0.6, smooth_arcsec=0.1, asinh_soft=3.0):
    """High-passed, stretched, smoothed image (zero outside valid, zero mean inside) or None when the image is empty or flat.
    scale = pixel size [arcsec]; the block is at least 6 pixels."""
    img = img.double()
    valid = valid & torch.isfinite(img)
    if int(valid.sum()) < 100:
        return None
    blk = max(int(round(block_arcsec / scale)), 6)
    x = torch.where(valid, img - block_background(img, valid, blk), torch.zeros_like(img))
    xv = x[valid]
    sig = 1.4826 * float((xv - xv.median()).abs().median())
    if not sig > 0:
        sig = float(xv.std())
    if not sig > 0:
        return None
    y = torch.where(valid, torch.asinh(x / (asinh_soft * sig)), torch.zeros_like(x))
    y = torch.where(valid, masked_smooth(y, valid, smooth_arcsec / scale), torch.zeros_like(y))
    y = torch.where(valid, y - y[valid].mean(), torch.zeros_like(y))
    return y, valid


def xcorr_peak(a, va, b, vb, radius_px, min_overlap_px, excl_px=3.0, min_peak=0.05, min_ratio=1.5):
    """Masked normalised cross-correlation of two high-passed images on the same pixel grid (ny, nx), lags (dx, dy) in pixels with
    dx^2 + dy^2 <= radius_px^2. Returns dict: ok, reason, dx_px, dy_px (the lag at which sum_p a(p) b(p + lag) peaks: feature position in b minus in a),
    peak, second, ratio, n_overlap. See the module docstring for the conventions."""
    ny, nx = a.shape
    R = int(math.ceil(radius_px))
    L = R + 2
    fy, fx = ny + L, nx + L
    s = (fy, fx)
    Ma, Mb = va.to(a.dtype), vb.to(a.dtype)
    ya, yb = a * Ma, b * Mb
    rf = lambda x: torch.fft.rfft2(x, s=s)  # noqa: E731
    A0, A1, A2 = rf(Ma), rf(ya), rf(ya * a)
    B0, B1, B2 = rf(Mb), rf(yb), rf(yb * b)
    idy = torch.arange(-L, L + 1) % fy
    idx = torch.arange(-L, L + 1) % fx

    def cc(X, Y):
        c = torch.fft.irfft2(torch.conj(X) * Y, s=s)
        return c[idy][:, idx]
    N, Sa, Sb, Saa, Sbb, Sab = cc(A0, B0), cc(A1, B0), cc(A0, B1), cc(A2, B0), cc(A0, B2), cc(A1, B1)
    Nc = N.clamp(min=1.0)
    va_, vb_ = Saa - Sa * Sa / Nc, Sbb - Sb * Sb / Nc
    r = (Sab - Sa * Sb / Nc) / torch.sqrt(va_.clamp(min=1e-30) * vb_.clamp(min=1e-30))
    ok = (N >= min_overlap_px) & (va_ > 1e-9 * Nc) & (vb_ > 1e-9 * Nc) & torch.isfinite(r)
    ncc = torch.where(ok, r, torch.full_like(r, float("nan")))
    yy, xx = torch.meshgrid(torch.arange(-L, L + 1, dtype=a.dtype), torch.arange(-L, L + 1, dtype=a.dtype), indexing="ij")
    search = ((yy ** 2 + xx ** 2) <= radius_px ** 2) & ok
    bad = dict(ok=False, dx_px=float("nan"), dy_px=float("nan"), peak=float("nan"), second=float("nan"), ratio=0.0, n_overlap=0.0)
    if not bool(search.any()):
        return dict(bad, reason="no overlap")
    nz = torch.where(search, ncc, torch.full_like(ncc, -float("inf")))
    k = int(torch.argmax(nz))
    iy, ix = divmod(k, nz.shape[1])
    peak = float(nz[iy, ix])
    filled = torch.where(torch.isfinite(ncc), ncc, torch.full_like(ncc, -float("inf")))
    locmax = (F.max_pool2d(filled[None, None], 5, 1, 2)[0, 0] == filled) & search
    far = torch.sqrt((yy - yy[iy, ix]) ** 2 + (xx - xx[iy, ix]) ** 2) > max(excl_px, 3.0)
    cand = ncc[locmax & far]
    second = float(cand.max()) if cand.numel() else 0.0
    ratio = 99.0 if second <= 0 else min(peak / second, 99.0)

    def vertex(m, c, p):         # parabola through the three values around the integer peak (m, c, p)
        if not (math.isfinite(m) and math.isfinite(p)):
            return 0.0
        den = m - 2 * c + p
        return float(np.clip(0.5 * (m - p) / den, -0.5, 0.5)) if den < 0 else 0.0
    ddx = vertex(float(ncc[iy, ix - 1]), peak, float(ncc[iy, ix + 1])) if 0 < ix < ncc.shape[1] - 1 else 0.0
    ddy = vertex(float(ncc[iy - 1, ix]), peak, float(ncc[iy + 1, ix])) if 0 < iy < ncc.shape[0] - 1 else 0.0
    ly, lx = iy - L, ix - L
    reason = None
    if peak < min_peak:
        reason = "low peak"
    elif ratio < min_ratio:
        reason = "ambiguous (second-peak ratio)"
    elif math.hypot(ly, lx) >= radius_px - 1.0:
        reason = "peak at the search boundary"
    return dict(ok=reason is None, reason=reason, dx_px=float(lx + ddx), dy_px=float(ly + ddy), peak=peak, second=second, ratio=float(ratio),
                n_overlap=float(N[iy, ix]))


def coarse_shift(d, vd, m, vm, J, radius_arcsec=1.5, block_arcsec=0.6, smooth_arcsec=0.1, min_peak=0.05, min_ratio=1.5, min_overlap_arcsec2=30.0):
    """Shift t [arcsec, tangent plane X west, Y north] to add to the header positions of the pixels of data d (ny, nx) so that they sit on the
    sky whose rendering on the same pixels is m (same shape; vd, vm the valid masks); J (2, 2) = d(X, Y) / d(column, row). Returns the dict of
    xcorr_peak plus tX, tY (arcsec; nan when refused is NOT applied: they are still given for the log) and the pixel scale."""
    J = np.asarray(J, np.float64)
    scale = float(abs(np.linalg.det(J)) ** 0.5)
    pd, pm = high_pass(d, vd, scale, block_arcsec, smooth_arcsec), high_pass(m, vm, scale, block_arcsec, smooth_arcsec)
    if pd is None or pm is None:
        return dict(ok=False, reason="empty or flat image", tX=float("nan"), tY=float("nan"), peak=float("nan"), second=float("nan"), ratio=0.0,
                    n_overlap=0.0, scale=scale)
    nv = int((pd[1] & pm[1]).sum())
    n_min = max(25.0, min(min_overlap_arcsec2 / scale ** 2, 0.25 * nv))
    r = xcorr_peak(pd[0], pd[1], pm[0], pm[1], radius_arcsec / scale, n_min, excl_px=0.1 / scale, min_peak=min_peak, min_ratio=min_ratio)
    t = J @ np.array([r["dx_px"], r["dy_px"]])
    r.update(tX=float(t[0]), tY=float(t[1]), scale=scale)
    return r


# ----------------------------------------------------------------------------- in the fit
def _select(exps, cfg, inst_names):
    """Instrument indices to register: --coarse-reg-instruments ('' = every instrument except the reference one of --gauge-ref-inst; 'all'; or a
    comma list of instrument names or indices)."""
    n_inst = len({e.inst for e in exps})
    ref = int(getattr(cfg, "gauge_ref_inst", -1))
    spec = str(getattr(cfg, "coarse_reg_instruments", "") or "").strip()
    allidx = sorted({e.inst for e in exps})
    if spec.lower() == "all":
        return allidx, ref
    if spec:
        out = []
        for tok in [s.strip() for s in spec.split(",") if s.strip()]:
            if tok.lstrip("-").isdigit():
                out.append(int(tok))
            elif inst_names and tok.upper() in [str(n).upper() for n in inst_names]:
                out.append([str(n).upper() for n in inst_names].index(tok.upper()))
            else:
                raise SystemExit(f"--coarse-reg-instruments: unknown instrument {tok!r} (known {inst_names})")
        return sorted(set(out)), ref
    if n_inst < 2 or ref < 0:
        return [], ref
    return [i for i in allidx if i != ref], ref


@torch.no_grad()
def coarse_register(model, exps, cfg, log=print, ts=None, inst_names=None):
    """--coarse-reg xcorr: set the starting translation of the selected exposures (model.affine[eid, 4:6]) from the cross-correlation of their
    pixels with the starting sky rendered through their warm-start kernel, then rebuild the sky warm start from every training exposure at the
    new translations (as the pre-pass does). See the module docstring. Returns the record {per exposure: shift, peak, ratio, ok; visits; ...}."""
    from .data import splat_coadd
    from .sky import spectral_init
    from .tiles import TileSet
    if getattr(cfg, "coarse_reg", "off") == "off":
        return {}
    t_start = time.perf_counter()
    sel, ref = _select(exps, cfg, inst_names)
    if not sel:
        log("coarse registration: no instrument to register (one instrument, or the reference instrument only); nothing done")
        return {"selected": []}
    ts = ts or TileSet.single(model, exps, cfg, log)
    refs = [i for i in sorted({e.inst for e in exps}) if ref < 0 or i == ref]
    if not any(e.inst in refs and e.split == "train" for e in exps):
        raise SystemExit("--coarse-reg: no training exposure of the reference instrument to build the starting sky from")
    nf = len(model.fset)
    have = sorted({e.filt for e in exps if e.inst in refs and e.split == "train"})
    log(f"coarse registration (xcorr): instruments {sel} against a sky built from instrument(s) {refs} (filters {have}), radius {cfg.coarse_reg_radius} arcsec, "
        f"min peak {cfg.coarse_reg_min_peak}, min second-peak ratio {cfg.coarse_reg_min_ratio}")
    sc = model.aff_scale[4:6]
    piv = [float(f.pivot) for f in model.fset.filters]
    out = {}
    best = {}
    for k in ts.keys:
        ts.activate(k)
        ref_cuts = [e for e in exps if e.tile == k and e.split == "train" and e.inst in refs]
        if not ref_cuts:
            continue
        sk = ts.skies[k]
        img, wt = splat_coadd(ref_cuts, nf, sk.N, sk.pitch, model.dev, per_area=model.input_level == "rate")      # starting sky: reference instrument only
        x0 = -(sk.N - 1) / 2 * sk.pitch
        for e in exps:
            if e.tile != k or e.inst not in sel:
                continue
            # CHOICE: no rendering through a kernel and no colour model. The starting image of the nearest reference band (by pivot wavelength, in
            # ln wavelength) is resampled (bilinear) at the header positions of the exposure's pixels (translation zero); valid where that band's
            # coadd has weight at all four neighbouring nodes. Correlation is blind to a scale factor, and the high-pass removes the background.
            bnd = min(have, key=lambda f: abs(math.log(piv[f] / piv[e.filt])))
            gx, gy = (e.X + e.shift0[0] - x0) / sk.pitch, (e.Y + e.shift0[1] - x0) / sk.pitch
            grid = torch.stack([2 * gx / (sk.N - 1) - 1, 2 * gy / (sk.N - 1) - 1], -1)[None].to(img.dtype)
            S = F.grid_sample(img[bnd][None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
            wv = F.grid_sample((wt[bnd] > 0).to(img.dtype)[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
            vm = wv > 0.999
            r = coarse_shift(e.d.cpu(), (e.valid & torch.isfinite(e.d)).cpu(), S.cpu(), vm.cpu(), e.extra["J"], cfg.coarse_reg_radius,
                             cfg.coarse_reg_block, cfg.coarse_reg_smooth, cfg.coarse_reg_min_peak, cfg.coarse_reg_min_ratio)
            r["name"], r["eid"], r["visit"], r["band"] = e.name, e.eid, str(e.extra.get("visit_key", e.group)), int(bnd)
            if e.eid not in best or (r["ok"] and (not best[e.eid]["ok"] or r["peak"] > best[e.eid]["peak"])) or \
                    (not r["ok"] and not best[e.eid]["ok"] and r["peak"] > best[e.eid]["peak"]):
                best[e.eid] = r
    for eid, r in sorted(best.items()):
        if r["ok"]:
            model.affine.data[eid, 4:6] = torch.tensor([r["tX"], r["tY"]], dtype=model.affine.dtype, device=model.dev) / sc
            log(f"coarse registration {r['name']}: shift ({r['tX'] * 1e3:+.1f}, {r['tY'] * 1e3:+.1f}) mas, peak {r['peak']:.3f}, ratio {r['ratio']:.2f}, "
                f"overlap {r['n_overlap']:.0f} px")
        else:
            log(f"coarse registration {r['name']}: REFUSED ({r['reason']}); peak {r['peak']:.3f}, ratio {r['ratio']:.2f}; translation left at 0")
        out[r["name"]] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items() if k not in ("name",)}
    # scatter within a visit
    visits = {}
    for r in best.values():
        if r["ok"]:
            visits.setdefault(r["visit"], []).append((r["tX"], r["tY"]))
    vrec = {}
    for v, lst in sorted(visits.items()):
        a = np.asarray(lst) * 1e3
        vrec[v] = {"n": len(lst), "mean_mas": a.mean(0).tolist(), "std_mas": (a.std(0, ddof=1).tolist() if len(lst) > 1 else [float("nan")] * 2)}
        log(f"coarse registration visit {v}: {len(lst)} exposures, mean shift ({a.mean(0)[0]:+.1f}, {a.mean(0)[1]:+.1f}) mas, scatter "
            f"{'(%.1f, %.1f)' % tuple(a.std(0, ddof=1)) if len(lst) > 1 else 'n/a'} mas")
    nref = sum(1 for r in best.values() if not r["ok"])
    # common-translation gauge, applied here to EVERY exposure (training, validation, held out): the pre-pass and the fit remove the mean translation of the
    # training exposures from the training rows only, which would leave a held-out exposure at the old frame (off by the mean, 0.1 arcsec or more).
    if len(ts.shift_groups) == 1 and not model.fs_on:
        g = ts.shift_groups[0]
        mean = model.affine[g, 4:6].mean(0)
        model.affine.data[:, 4:6] -= mean
        log(f"coarse registration: mean translation of the training exposures ({float(mean[0] * sc[0]) * 1e3:+.1f}, {float(mean[1] * sc[1]) * 1e3:+.1f}) mas "
            f"removed from every exposure (common-translation gauge)")
    # rebuild the sky warm start from every training exposure at the new translations
    shifts = {}
    for k in ts.keys:
        ts.activate(k)
        tr = [e for e in exps if e.tile == k and e.split == "train"]
        if not tr:
            continue
        gids = sorted({e.eid for e in exps})
        t = model.affine[torch.as_tensor(gids, device=model.dev), 4:6] * sc
        shifts = {g: (float(t[i, 0]), float(t[i, 1])) for i, g in enumerate(gids)}
        img, wt = splat_coadd(tr, nf, ts.skies[k].N, ts.skies[k].pitch, model.dev, extra_shift=shifts, per_area=model.input_level == "rate")
        ts.skies[k].reset_from_coeffs(spectral_init(img, model.Phi_band))
    if nref:
        log("WARNING: coarse registration refused " + str(nref) + " exposure(s), translation left at its starting value (weak or ambiguous correlation peak: peak < "
            f"{cfg.coarse_reg_min_peak} or peak / second peak < {cfg.coarse_reg_min_ratio}, or peak at the search boundary): "
            + ", ".join(f"{r['name']} (peak {r['peak']:.3f}, ratio {r['ratio']:.2f})" for r in best.values() if not r["ok"]))
    log(f"coarse registration: {len(best) - nref} of {len(best)} exposures moved, {nref} refused ({time.perf_counter() - t_start:.1f} s); sky warm start rebuilt")
    return {"selected": sel, "reference": refs, "exposures": out, "visits": vrec, "n_refused": nref, "seconds": time.perf_counter() - t_start}
