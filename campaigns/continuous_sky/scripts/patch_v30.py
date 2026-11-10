"""v29 -> v30 (5 Nov 2026): HST starting WCS policy and a coarse shift search inside the model.
* --hst-wcs apriori|archive (default apriori): MAST's primary HST WCS is an a-posteriori fit to Gaia made by registering
  these images ('-FIT_REL_GAIA*'), i.e. a pre-alignment. 'apriori' starts from the guide-star solution (with the
  catalogue-level GSC correction when present). Offsets to the archive fit: ~0.4" (WFC3/IR sample), 70-90 mas (ACS).
* --coarse-shift R (arcsec, default 0 = off): before the coadds and the fit, each HST exposure (all chips together) gets a
  translation from a likelihood scan against the JWST drizzle-like coadd of the nearest-wavelength NIRCam band, blurred to
  the HST resolution: d ~ g * ref(tx + sx, ty + sy) + c, weighted least squares per shift, grid +-R at 1 target px, then
  0.25 px, then a parabola. JWST is the anchor frame (its exposures are not shifted). The transform is then fitted jointly
  with PSF and sky as before. Translation only (rotation and scale are left to the fit)."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


rep('''    hexps = hst_io.load_hst_exposures(files, thead, dev, pad=pad,''', '''    hst_io.WCS_POLICY = getattr(a, "hst_wcs", "apriori")
    LOG["hst_wcs_policy"] = hst_io.WCS_POLICY
    hexps = hst_io.load_hst_exposures(files, thead, dev, pad=pad,''')
rep('''def band_median_sigma(c, cover=None):''', '''def coarse_register_hst(exps, thead, dev, a):
    """v30: translation per HST exposure from a likelihood scan against a JWST coadd (see patch_v30.py)."""
    nir = [b for b in REG.bands if not REG.is_hst[b]]
    if not nir:
        tick("coarse shift: no NIRCam band to anchor on, skipped")
        return
    scale = abs(float(thead["CD2_2"])) * 3600.0
    R = int(round(a.coarse_shift / scale))
    refs = {}
    gen = torch.Generator(device="cpu").manual_seed(0)
    groups = {}
    for i, e in enumerate(exps):
        if REG.is_hst[e["band"]]:
            groups.setdefault(dither_key(e["name"]), []).append(i)
    LOG["coarse_shift"] = {}
    for key, idx in groups.items():
        b = exps[idx[0]]["band"]
        bref = min(nir, key=lambda x: abs(math.log(REG.lam_um[x] / REG.lam_um[b])))
        if bref not in refs:
            refs[bref] = classical([e for e in exps if e["band"] == bref], thead, "drizzle_square", 0.8, dev, bands=[bref])[0]
        ref = refs[bref]
        sig = math.sqrt(max(REG.fwhm_arcsec[b] ** 2 - REG.fwhm_arcsec[bref] ** 2, 0.0)) / scale / 2.3548
        refb = gauss_blur(ref[None], sig)[0] if sig > 0.3 else ref
        pts = []
        for i in idx:
            e = exps[i]
            n = e["d"].numel()
            sel = torch.randperm(n, generator=gen)[: min(n, 40000)].to(e["d"].device)
            pts.append((e["tx"][sel], e["ty"][sel], e["d"][sel], e["ivar"][sel]))

        def chi2_at(sx, sy):
            tot = 0.0
            for tx, ty, d, iv in pts:
                v = sample(refb, tx + sx, ty + sy)
                A = torch.stack([v, torch.ones_like(v)], 1).double()
                w = iv.double()
                coef = torch.linalg.lstsq((A * w.sqrt()[:, None]).cpu(), (d.double() * w.sqrt()).cpu()[:, None]).solution.to(A.device)[:, 0]
                r2 = w * (d.double() - A @ coef) ** 2
                tot += float(torch.minimum(r2, torch.full_like(r2, 25.0)).sum())  # clipped at 5 sigma (cosmic rays, artefacts)
            return tot

        grid = [(chi2_at(float(sx), float(sy)), sx, sy) for sy in range(-R, R + 1) for sx in range(-R, R + 1)]
        c0 = chi2_at(0.0, 0.0)
        best = min(grid)
        fine = [(chi2_at(best[1] + fx, best[2] + fy), best[1] + fx, best[2] + fy)
                for fy in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0) for fx in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)]
        cb, bx, by = min(fine)

        def parab(f_m, f_0, f_p, h=0.25):
            den = f_m - 2 * f_0 + f_p
            return 0.0 if den <= 0 else float(max(-h, min(h, 0.5 * h * (f_m - f_p) / den)))
        bx += parab(chi2_at(bx - 0.25, by), cb, chi2_at(bx + 0.25, by))
        by += parab(chi2_at(bx, by - 0.25), cb, chi2_at(bx, by + 0.25))
        edge = abs(best[1]) == R or abs(best[2]) == R
        for i in idx:
            exps[i]["tx"] = exps[i]["tx"] + bx
            exps[i]["ty"] = exps[i]["ty"] + by
        LOG["coarse_shift"][str(key)] = {"band": b, "ref": bref, "shift_mas": [bx * scale * 1000, by * scale * 1000],
                                         "chi2_ratio": cb / max(c0, 1e-30), "at_grid_edge": edge, "chips": len(idx)}
        tick(f"coarse shift {key} {b} (ref {bref}): ({bx * scale * 1000:.0f}, {by * scale * 1000:.0f}) mas, chi2 x{cb / max(c0, 1e-30):.3f}"
             + (" AT GRID EDGE" if edge else ""))


def band_median_sigma(c, cover=None):''')
rep('''def classical(exps, thead, kernel, pixfrac, device):''', '''def classical(exps, thead, kernel, pixfrac, device, bands=None):''')
rep('''    for b in REG.bands:
        if REG.is_hst[b]:  # v29''', '''    for b in (bands or REG.bands):
        if REG.is_hst[b]:  # v29''')
rep('''    return torch.stack([out[b] for b in REG.bands])''', '''    return torch.stack([out[b] for b in (bands or REG.bands)])''')
rep('''    # ---------------- classical (torchsky)
    cfn = classical_fn or classical''', '''    if REG.multi and getattr(a, "coarse_shift", 0.0) > 0:  # v30
        t = time.perf_counter()
        coarse_register_hst(exps, thead, dev, a)
        LOG["timings"]["coarse_shift"] = round(time.perf_counter() - t, 1)

    # ---------------- classical (torchsky)
    cfn = classical_fn or classical''')
rep('''    ap.add_argument("--hst-dq-policy",''', '''    ap.add_argument("--hst-wcs", choices=("apriori", "archive"), default="apriori", help="v30: HST starting WCS (archive = MAST's a-posteriori Gaia fit, a pre-alignment)")
    ap.add_argument("--coarse-shift", type=float, default=0.0, help="v30: HST translation search radius (arcsec) against the JWST coadd; 0 = off")
    ap.add_argument("--hst-dq-policy",''')
open(dst, "w").write(s)
print("wrote", dst)
