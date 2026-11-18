"""Checks of the instrument interface on the real sample files (no fit): units, geometry, valid pixels and noise (INSTRUMENT_INTERFACE.md, 4a-4c).

Every file is read by its instrument's reader (fieldop/instruments.py) and cut by the package's own data.load_exposure_file around one virtual
box of side --box (140) arcsec centred on the JWST NRCB1 F090W exposure (it also holds the mosaic cut-outs), so that the positions are header positions in one tangent plane (X west,
Y north, arcsec) exactly as a fit would see them, and the science planes are in MJy/sr after the instrument's conversion.

(a) units: matched circular apertures (radius 0.4 and 0.8 arcsec, background = median of the annulus 1.5-2.5 arcsec) on bright compact sources
    found in the JWST exposure; per source the ratio of the HST to the JWST aperture flux; median and quartiles over sources.
(b) geometry: windowed centroids (Gaussian window sigma 0.12 arcsec, 8 iterations) of the same sources in each HST exposure and in the JWST
    exposure through each file's own header solution; offsets in milliarcseconds, and a similarity fit (translation, scale, rotation).
(c) valid fraction under each instrument's policy (whole chip and inside the box), median pipeline sigma against the robust scatter of
    blank sky (1.4826 median absolute deviation of source-masked pixels, and of first differences / sqrt 2).

    PYTHONNOUSERSITE=1 PYTHONPATH=/home/claude/torchsky/src python tests/check_instruments_real.py --out tests/instrument_results/real_checks.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy import ndimage

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fieldop import data as D  # noqa: E402
from fieldop import instruments as I  # noqa: E402

JW = "/home/claude/work/dl_full"
HST = "/home/claude/work/dl_hst"
MOS = "/home/claude/work/kernel_check/dl"
JWST_FILES = {"F090W": f"{JW}/jw02736001001_02101_00009_nrcb1_cal.fits", "F277W": f"{JW}/jw02736001001_02101_00004_nrcblong_cal.fits"}
HST_FILES = ["iczgz2nsq_flt.fits", "iczgz4kcq_flt.fits", "iczgz2nxq_flt.fits", "iczgzbmgq_flt.fits", "iczgz6j9q_flt.fits", "jczgz3k8q_flc.fits"]
PAIR = {"F105W": "F090W", "F140W": "F150W", "F160W": "F150W", "F814W": "F090W"}   # nearest JWST filter


def cut(raw, box, want_jac=True):
    r = D.load_exposure_file("", box, box["size_arcsec"] / 2, want_jac=want_jac, raw=raw)
    return r


def mosaic_cut(filt, box):
    """Official calibrated mosaic cut-out (MJy/sr) as a cut: header positions in the same tangent plane."""
    from astropy.io import fits
    from torchsky.wcs import WCS
    with fits.open(f"{MOS}/cut_{filt}.fits") as h:
        d = h[0].data.astype(np.float32)
        hd = {k: h[0].header[k] for k in h[0].header if k and k not in ("COMMENT", "HISTORY")}
        err = h["ERR"].data.astype(np.float32)
    size = int(math.ceil(box["size_arcsec"] / 0.01)) + 64
    wt = WCS(header=D.target_header(box["ra"], box["dec"], size, 0.01))
    rr, cc = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    X, Y = D.tangent_positions(WCS(header=hd), wt, cc.astype(np.float64), rr.astype(np.float64), 0.01, size)
    valid = np.isfinite(d) & (d != 0)
    return dict(d=np.nan_to_num(d), sig=err, valid=valid, X=X.astype(np.float32), Y=Y.astype(np.float32),
                Jpix=D.pixel_jacobian(X, Y), cut0=(0, 0))


def window(c, x, y, rad):
    """The cut restricted to the pixels within about rad (+ 4 arcsec of slack for distortion) of (x, y), via the cut's mean linear map."""
    if "_lin" not in c:
        ny, nx = c["X"].shape
        rr, cc = np.mgrid[0:ny:16, 0:nx:16]
        A = np.stack([np.ones(rr.size), cc.ravel(), rr.ravel()], 1)
        coef = np.linalg.lstsq(A, np.stack([c["X"][::16, ::16].ravel(), c["Y"][::16, ::16].ravel()], 1), rcond=None)[0]
        c["_lin"] = (coef[0], np.linalg.inv(coef[1:].T))
        c["_area"] = np.abs(c["Jpix"][..., 0, 0] * c["Jpix"][..., 1, 1] - c["Jpix"][..., 0, 1] * c["Jpix"][..., 1, 0])
    o, Jinv = c["_lin"]
    col, row = Jinv @ (np.array([x, y]) - o)
    pix = math.sqrt(abs(np.linalg.det(np.linalg.inv(Jinv))))
    m = int((rad + 4.0) / pix) + 2
    ny, nx = c["X"].shape
    r0, r1, c0, c1 = max(int(row) - m, 0), min(int(row) + m + 1, ny), max(int(col) - m, 0), min(int(col) + m + 1, nx)
    if r0 >= r1 or c0 >= c1:
        return None
    sl = (slice(r0, r1), slice(c0, c1))
    return {"X": c["X"][sl], "Y": c["Y"][sl], "d": c["d"][sl], "valid": c["valid"][sl], "a": c["_area"][sl]}


def area(c):
    J = c["Jpix"]
    return np.abs(J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0])


def aperture_flux(c, x, y, r, ann=(1.5, 2.5), need=0.9):
    """Background-subtracted flux [MJy/sr x arcsec^2] in a circle of radius r: sum over valid pixels of (d - b) a / (valid area fraction)."""
    c = window(c, x, y, max(r, ann[1]))
    if c is None:
        return np.nan
    R = np.hypot(c["X"] - x, c["Y"] - y)
    a = c["a"]
    inn = R < r
    an = (R >= ann[0]) & (R < ann[1]) & c["valid"]
    if inn.sum() == 0 or an.sum() < 20:
        return np.nan
    v = inn & c["valid"]
    frac = a[v].sum() / max(a[inn].sum(), 1e-30)
    if frac < need:
        return np.nan
    b = np.median(c["d"][an])
    return float(((c["d"][v] - b) * a[v]).sum() / frac)


def centroid(c, x, y, sig=0.12, it=8, box=0.6):
    c = window(c, x, y, 2.5)
    if c is None:
        return None
    R2box = (np.abs(c["X"] - x) < box) & (np.abs(c["Y"] - y) < box)
    if R2box.sum() < 9:
        return None
    X, Y, d, v = c["X"][R2box].astype(np.float64), c["Y"][R2box].astype(np.float64), c["d"][R2box].astype(np.float64), c["valid"][R2box]
    R = np.hypot(c["X"] - x, c["Y"] - y)
    an = (R >= 1.5) & (R < 2.5) & c["valid"]
    b = np.median(c["d"][an]) if an.sum() > 20 else np.median(d[v])
    w0 = np.clip(d - b, 0, None) * v
    cx, cy = x, y
    for _ in range(it):
        w = w0 * np.exp(-0.5 * ((X - cx) ** 2 + (Y - cy) ** 2) / sig ** 2)
        if w.sum() <= 0:
            return None
        cx, cy = float((w * X).sum() / w.sum()), float((w * Y).sum() / w.sum())
        if abs(cx - x) > box or abs(cy - y) > box:
            return None
    # valid coverage near the centre
    near = ((X - cx) ** 2 + (Y - cy) ** 2) < (2 * sig) ** 2
    if near.sum() == 0 or v[near].mean() < 0.9:
        return None
    return cx, cy


def find_sources(c, n=60, pix=0.031, iso=1.5, edge=3.0, nsig=30.0):
    """Bright compact sources of a JWST cut: peaks of the image smoothed by a Gaussian of 2 pixels, isolated (no brighter peak within `iso`
    arcsec), at least `edge` arcsec inside the valid region, with the peak / 0.4-arcsec-aperture ratio of a compact source."""
    d = np.where(c["valid"], c["d"], np.median(c["d"][c["valid"]]))
    s = ndimage.gaussian_filter(d, 2.0)
    mx = ndimage.maximum_filter(s, size=int(iso / pix) * 2 + 1)
    foot = ndimage.uniform_filter(c["valid"].astype(np.float32), 15) > 0.5      # footprint (isolated flagged pixels do not cut holes)
    vin = ndimage.binary_erosion(foot, iterations=int(edge / pix))
    pk = (s == mx) & vin
    bk = np.median(s[c["valid"]])
    rs = 1.4826 * np.median(np.abs(s[c["valid"]] - bk))
    pk &= s > bk + nsig * rs
    ii = np.argwhere(pk)
    order = np.argsort(-s[pk])
    out = []
    for k in order[: 4 * n]:
        r_, c_ = ii[k]
        x, y = float(c["X"][r_, c_]), float(c["Y"][r_, c_])
        f4 = aperture_flux(c, x, y, 0.4)
        if not np.isfinite(f4) or f4 <= 0:
            continue
        f8 = aperture_flux(c, x, y, 0.8)
        if not np.isfinite(f8) or f8 <= 0 or f4 / f8 < 0.55:           # compact: most of the 0.8" flux inside 0.4"
            continue
        out.append((x, y, f4))
        if len(out) >= n:
            break
    return out


def match(ref, c, src, coarse=1.2):
    """Centroids of the sources in the reference cut and in cut c (after a coarse translation from the 15 brightest: brightest 0.25-arcsec
    aperture within +-coarse arcsec); similarity fit of c's positions against the reference's, with one clipping pass."""
    offs = []
    for x, y, _ in src[:15]:
        cj = centroid(ref, x, y)
        if cj is None:
            continue
        best = None
        for dx in np.arange(-coarse, coarse + 0.01, 0.1):
            for dy in np.arange(-coarse, coarse + 0.01, 0.1):
                f_ = aperture_flux(c, cj[0] + dx, cj[1] + dy, 0.25, need=0.5)
                if np.isfinite(f_) and (best is None or f_ > best[0]):
                    best = (f_, dx, dy)
        if best is not None:
            ch = centroid(c, cj[0] + best[1], cj[1] + best[2], box=0.4)
            if ch is not None:
                offs.append((ch[0] - cj[0], ch[1] - cj[1]))
    if len(offs) < 3:
        return None
    t0 = np.median(np.asarray(offs), 0)
    pj, ph = [], []
    for x, y, _ in src:
        cj = centroid(ref, x, y)
        if cj is None:
            continue
        ch = centroid(c, cj[0] + t0[0], cj[1] + t0[1], sig=0.15, box=0.5)
        if ch is not None:
            pj.append(cj)
            ph.append(ch)
    if len(pj) < 4:
        return None
    P, Q = np.asarray(pj), np.asarray(ph)
    off = (Q - P) * 1e3
    fit = similarity_fit(P, Q)
    resid = np.hypot(*(off - np.median(off, 0)).T)
    keep = resid < max(4 * 1.4826 * np.median(resid), 30.0)
    fit_c = similarity_fit(P[keep], Q[keep]) if keep.sum() >= 4 else None
    t = np.median(Q - P, 0)
    return {"t": (float(t[0]), float(t[1])), "summary": {"n": len(pj), "median_offset_mas": np.median(off, 0).tolist(),
            "offset_rms_about_median_mas": float(np.sqrt(np.mean(np.sum((off - np.median(off, 0)) ** 2, 1)))), "similarity": fit, "similarity_clipped": fit_c,
            "n_clipped": int((~keep).sum())}}


def similarity_fit(P, Q):
    """Q = (1 + s) R(theta) P + t, least squares on complex numbers: returns t (mas), s (ppm), theta (arcsec of angle), residual rms (mas)."""
    p = P[:, 0] + 1j * P[:, 1]
    q = Q[:, 0] + 1j * Q[:, 1]
    pm, qm = p.mean(), q.mean()
    a = ((q - qm) * np.conj(p - pm)).sum() / (np.abs(p - pm) ** 2).sum()
    t = qm - a * pm
    res = q - (a * p + t)
    rms = float(np.sqrt(np.mean(np.abs(res) ** 2)))
    n = len(P)
    sig1 = rms / math.sqrt(2) * math.sqrt(n / max(n - 4, 1))          # per-coordinate residual, corrected for the 4 fitted numbers
    err_a = sig1 / math.sqrt((np.abs(p - pm) ** 2).sum())             # standard error of Re a and Im a
    return {"scale_err_ppm": float(err_a * 1e6), "rotation_err_arcsec": float(err_a * 180 / math.pi * 3600),"translation_mas": [float(t.real * 1e3), float(t.imag * 1e3)], "scale_minus_1_ppm": float((abs(a) - 1) * 1e6),
            "rotation_arcsec_of_angle": float(np.angle(a) * 180 / math.pi * 3600), "residual_rms_mas": float(np.sqrt(np.mean(np.abs(res) ** 2)) * 1e3),
            "rotation_displacement_at_30arcsec_mas": float(abs(np.angle(a)) * 30e3), "n": int(len(P)),
            "radius_span_arcsec": float(np.abs(p - pm).max())}


def blank_noise(c, pixarea_pix=None):
    """Median pipeline sigma and robust scatter of source-masked pixels inside the cut's valid region (MJy/sr)."""
    d, v = c["d"], c["valid"].copy()
    m = v.copy()
    for _ in range(3):
        med = np.median(d[m])
        rs = 1.4826 * np.median(np.abs(d[m] - med))
        src = ndimage.binary_dilation((d > med + 3 * rs) & v, iterations=3)
        m = v & ~src & (np.abs(d - med) < 5 * rs)
    med = np.median(d[m])
    rs = 1.4826 * np.median(np.abs(d[m] - med))
    dd = (d[:, 1:] - d[:, :-1])[m[:, 1:] & m[:, :-1]]
    rd = 1.4826 * np.median(np.abs(dd - np.median(dd))) / math.sqrt(2)
    sig = float(np.median(c["sig"][m]))
    return {"median_sigma": sig, "robust_scatter": float(rs), "robust_scatter_first_differences": float(rd), "ratio_scatter_over_sigma": float(rs / sig),
            "ratio_diff_scatter_over_sigma": float(rd / sig), "n_blank": int(m.sum()), "blank_median_MJy_sr": float(med)}


def in_box_valid(c, half):
    """Fraction of the pixels whose header position is inside the box (and on the chip) that the policy keeps."""
    ins = (np.abs(c["X"]) < half) & (np.abs(c["Y"]) < half)
    return float(c["valid"][ins].mean()) if ins.any() else float("nan")


def whole_chip_valid(raw):
    sci, err = raw["sci"], raw["err"]
    fin = np.isfinite(sci) & np.isfinite(err) & (err > 0)
    dq_ok = raw["dq_ok"](raw["dq"])
    return {"dq_policy_only": float(dq_ok.mean()), "dq_and_finite": float((dq_ok & fin).mean()), "dq_zero_and_finite": float(((raw["dq"].astype(np.int64) & 0xFFFFFFFF) == 0).mean() if raw["dq"] is not None else np.nan),
            "ignore_bits": int(raw["dq_ok"].ignore_bits),
            **({"jwst_ignoring_jump_bit_4": float((((raw["dq"].astype(np.int64) & ~4) == 0) & fin).mean())} if raw["instrument"] == "NIRCAM" else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", type=float, default=140.0)
    ap.add_argument("--hst-wcs", default="apriori")
    ap.add_argument("--n-sources", type=int, default=80)
    ap.add_argument("--nsig", type=float, default=10.0)
    a = ap.parse_args()
    T0 = time.time()
    I.HST_WCS = a.hst_wcs
    nc = I.INSTRUMENTS["NIRCAM"]
    # virtual box centred on the NRCB1 F090W exposure
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(JWST_FILES["F090W"]) as h:
        w = WCS(h["SCI"].header)
        ra0, de0 = w.all_pix2world([[1023.5, 1023.5]], 0)[0]
    box = {"id": 0, "ra": float(ra0), "dec": float(de0), "size_arcsec": a.box}
    res = {"box": box, "hst_wcs": a.hst_wcs, "valid": {}, "noise": {}, "units": {}, "geometry": {}, "conversion": {}}
    jw = {}
    for f, p in JWST_FILES.items():
        raw = nc.read(p)[0]
        res["valid"][f"JWST {f} {raw['detector']}"] = {"whole_chip": whole_chip_valid(raw)}
        c = cut(raw, box)
        jw[f] = c
        res["valid"][f"JWST {f} {raw['detector']}"]["in_box"] = in_box_valid(c, a.box / 2)
        res["noise"][f"JWST {f} {raw['detector']}"] = blank_noise(c)
        print(f"JWST {f}: cut {c['d'].shape} {time.time() - T0:.0f}s", flush=True)
    mos = {f: mosaic_cut(f, {"ra": box["ra"], "dec": box["dec"], "size_arcsec": a.box}) for f in ("F090W", "F150W")}
    src = find_sources(jw["F090W"], n=a.n_sources, nsig=a.nsig)
    res["n_sources_found"] = len(src)
    print(f"{len(src)} compact sources in NRCB1 F090W", flush=True)
    hst_cuts = {}
    mos_t = {}
    for f, mc in mos.items():                       # the mosaic is aligned to an absolute frame; the exposure is not: register it
        g = match(jw["F090W"], mc, src)
        mos_t[f] = None if g is None else g["t"]
        res["geometry"][f"official mosaic {f} (aligned) against the NRCB1 exposure"] = None if g is None else g["summary"]
    for fn in HST_FILES:
        ins = I.resolve("WFC3_IR" if fn.startswith("i") else "ACS_WFC")
        for raw in ins.read(f"{HST}/{fn}"):
            lab = f"{raw['instrument']} {raw['filter']} {fn.split('_')[0]}{raw['name_suffix']}"
            res["conversion"][lab] = {"to_sky_MJy_sr_per_unit": raw["to_sky"], "unit": raw["sci_unit"], **raw["to_sky_how"], "wcs": raw["wcs_name"],
                                      "lookup_tables": raw["wcs_lookup_tables"], "pivot_um": raw["photplam_A"] / 1e4}
            res["valid"][lab] = {"whole_chip": whole_chip_valid(raw)}
            c = cut(raw, box)
            if c is None:
                print(f"{lab}: no overlap", flush=True)
                continue
            res["valid"][lab]["in_box"] = in_box_valid(c, a.box / 2)
            hst_cuts[lab] = (raw["filter"], c)
            res["noise"][lab] = blank_noise(c)
            # (b) geometry against the JWST F090W exposure (also gives the translation used to place the HST apertures in (a))
            geo = match(jw["F090W"], c, src)
            if geo is None:
                print(f"{lab}: no matched sources", flush=True)
                continue
            res["geometry"][lab] = geo["summary"]
            # (a) units against the nearest JWST filter: the exposure when it is local, and the official mosaic (registered to the JWST exposure)
            jf = PAIR[raw["filter"]]
            refs = {}
            if jf in jw:
                refs[f"exposure {jf}"] = (jw[jf], (0.0, 0.0))
            if jf in mos and mos_t.get(jf) is not None:
                refs[f"mosaic {jf}"] = (mos[jf], mos_t[jf])
            ua = {}
            th = geo["t"]
            for rn, (rc, tr) in refs.items():
                for r_ap in (0.4, 0.8, 1.2):
                    rat = []
                    for x, y, _ in src:
                        fj, fh = aperture_flux(rc, x + tr[0], y + tr[1], r_ap), aperture_flux(c, x + th[0], y + th[1], r_ap)
                        if np.isfinite(fj) and np.isfinite(fh) and fj > 0:
                            rat.append(fh / fj)
                    if len(rat) >= 3:
                        q = np.percentile(rat, [25, 50, 75])
                        ua[f"{rn} r={r_ap}"] = {"median_ratio_hst_over_jwst": float(q[1]), "quartiles": [float(q[0]), float(q[2])], "n": len(rat)}
            res["units"][lab] = ua
            g = geo["summary"]["similarity_clipped"] or geo["summary"]["similarity"]
            print(f"{lab}: units {json.dumps({k: round(v['median_ratio_hst_over_jwst'], 3) for k, v in ua.items()})}; geometry t "
                  f"{np.round(g['translation_mas'], 1).tolist()} mas, scale {g['scale_minus_1_ppm']:.0f} +- {g['scale_err_ppm']:.0f} ppm, rotation "
                  f"{g['rotation_arcsec_of_angle']:.0f} +- {g['rotation_err_arcsec']:.0f} arcsec, residual {g['residual_rms_mas']:.1f} mas, n {g['n']} "
                  f"{time.time() - T0:.0f}s", flush=True)
    # (a) again against the official mosaics, with sources found in the mosaic itself (the F150W exposures are not on this machine)
    res["units_mosaic"] = {}
    for jf, mc in mos.items():
        src_m = find_sources(mc, n=a.n_sources, nsig=a.nsig)
        for lab, (hf, c) in hst_cuts.items():
            if PAIR.get(hf) != jf:
                continue
            g = match(mc, c, src_m)
            if g is None:
                continue
            ua = {}
            for r_ap in (0.4, 0.8, 1.2):
                rat = []
                for x, y, _ in src_m:
                    fj, fh = aperture_flux(mc, x, y, r_ap), aperture_flux(c, x + g["t"][0], y + g["t"][1], r_ap)
                    if np.isfinite(fj) and np.isfinite(fh) and fj > 0:
                        rat.append(fh / fj)
                if len(rat) >= 3:
                    q = np.percentile(rat, [25, 50, 75])
                    ua[f"mosaic {jf} r={r_ap}"] = {"median_ratio_hst_over_jwst": float(q[1]), "quartiles": [float(q[0]), float(q[2])], "n": len(rat)}
            res["units_mosaic"][lab] = {"apertures": ua, "n_sources_in_mosaic": len(src_m), "registration": g["summary"]}
            print(f"{lab} against mosaic {jf}: {json.dumps({k: round(v['median_ratio_hst_over_jwst'], 3) for k, v in ua.items()})} "
                  f"(t {np.round(g['t'], 3).tolist()} arcsec)", flush=True)
    # JWST F277W against F090W (same telescope, two detectors): the geometry check's own floor
    pj, pl = [], []
    for x, y, _ in src:
        cj = centroid(jw["F090W"], x, y)
        cl = centroid(jw["F277W"], x, y, sig=0.12) if cj is not None else None
        if cj is not None and cl is not None:
            pj.append(cj)
            pl.append(cl)
    if len(pj) >= 4:
        res["geometry"]["JWST F277W NRCBLONG (same visit, other detector)"] = {"n": len(pj), "similarity": similarity_fit(np.asarray(pj), np.asarray(pl))}
    res["seconds"] = time.time() - T0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"done {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
