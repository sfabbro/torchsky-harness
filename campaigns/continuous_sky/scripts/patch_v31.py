"""v30 -> v31 (5 Nov 2026): coarse translation for JWST exposures too, inside the model.
Step 1 found that the 2D learned grid fitted on unregistered JWST cal files scores S/N>5 2.87 [2.71, 3.02] against
2.42 [2.33, 2.51] on the tweakreg-aligned crf files (same test dithers): the gradient-only transform fit does not fully
absorb 25-55 mas starting offsets. v31 adds --coarse-shift-jwst R (arcsec): every exposure group (one dither: all its
detectors) gets a translation from a likelihood scan against a reference coadd of ANOTHER band not observed at the same
time (NIRCam SW and LW bands of one visit are simultaneous, so they are excluded), built by inverse-variance splatting of the
training pixels at their current positions (no file-header resampling). Two passes: the second rebuilds the references with
the first-pass shifts (+-3 px). HST groups use the nearest NIRCam band as before (--coarse-shift). Held-out and validation
exposures are shifted the same way (the references contain training exposures only). Translation only."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


rep('''def band_median_sigma(c, cover=None):''', '''def coarse_register_all(exps, thead, dev, a):
    """v31: translation per exposure group (dither) for every instrument; see patch_v31.py."""
    scale = abs(float(thead["CD2_2"])) * 3600.0
    gen = torch.Generator(device="cpu").manual_seed(0)
    groups = {}
    for i, e in enumerate(exps):
        groups.setdefault((e["band"], dither_key(e["name"])), []).append(i)
    nir = [b for b in REG.bands if not REG.is_hst[b]]
    vis = {b: {visit_key(e["name"]) for e in exps if e["band"] == b} for b in REG.bands}

    def ref_band(b):
        if REG.is_hst[b]:
            cand = nir
        else:
            cand = [x for x in nir if x != b and not (vis[x] & vis[b])]
        if not cand:
            return None
        return min(cand, key=lambda x: abs(math.log(REG.lam_um[x] / REG.lam_um[b])))

    LOG["coarse_shift"] = {}
    for it in range(2):
        refs = {}
        for (b, key), idx in groups.items():
            jw = not REG.is_hst[b]
            R0 = (a.coarse_shift_jwst if jw else a.coarse_shift)
            if R0 <= 0:
                continue
            bref = ref_band(b)
            if bref is None:
                continue
            if bref not in refs:
                img, w = splat_coadd([e for e in exps if e["band"] == bref and not e["holdout"] and not e.get("val")], thead, dev)
                refs[bref] = (torch.nan_to_num(img).to(dev), (w > 0).float().to(dev))
            ref, cov = refs[bref]
            sig = math.sqrt(max(REG.fwhm_arcsec[b] ** 2 - REG.fwhm_arcsec[bref] ** 2, 0.0)) / scale / 2.3548
            refb = gauss_blur(ref[None], sig)[0] if sig > 0.3 else ref
            R = int(round(R0 / scale)) if it == 0 else 3
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
                    m_ = sample(cov, tx + sx, ty + sy) > 0.999  # reference fully covered there
                    if int(m_.sum()) < 100:
                        return float("inf")
                    A = torch.stack([v[m_], torch.ones_like(v[m_])], 1).double()
                    w = iv[m_].double()
                    coef = torch.linalg.lstsq((A * w.sqrt()[:, None]).cpu(), (d[m_].double() * w.sqrt()).cpu()[:, None]).solution.to(A.device)[:, 0]
                    r2 = w * (d[m_].double() - A @ coef) ** 2
                    tot += float(torch.minimum(r2, torch.full_like(r2, 25.0)).sum()) / max(int(m_.sum()), 1) * 40000.0
                return tot

            grid = [(chi2_at(float(sx), float(sy)), sx, sy) for sy in range(-R, R + 1) for sx in range(-R, R + 1)]
            c0 = chi2_at(0.0, 0.0)
            best = min(grid)
            fine = [(chi2_at(best[1] + fx, best[2] + fy), best[1] + fx, best[2] + fy)
                    for fy in (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75) for fx in (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75)]
            cb, bx, by = min(fine)

            def parab(f_m, f_0, f_p, h=0.25):
                den = f_m - 2 * f_0 + f_p
                return 0.0 if den <= 0 or not math.isfinite(den) else float(max(-h, min(h, 0.5 * h * (f_m - f_p) / den)))
            bx += parab(chi2_at(bx - 0.25, by), cb, chi2_at(bx + 0.25, by))
            by += parab(chi2_at(bx, by - 0.25), cb, chi2_at(bx, by + 0.25))
            if not math.isfinite(cb):
                continue
            for i in idx:
                exps[i]["tx"] = exps[i]["tx"] + bx
                exps[i]["ty"] = exps[i]["ty"] + by
            rec = LOG["coarse_shift"].setdefault(f"{b}|{key}", {"band": b, "ref": bref, "shift_mas": [0.0, 0.0], "chips": len(idx)})
            rec["shift_mas"] = [rec["shift_mas"][0] + bx * scale * 1000, rec["shift_mas"][1] + by * scale * 1000]
            rec[f"chi2_ratio_pass{it}"] = cb / max(c0, 1e-30)
            rec["at_grid_edge"] = rec.get("at_grid_edge", False) or (it == 0 and (abs(best[1]) == R or abs(best[2]) == R))
        shifts = [math.hypot(*r["shift_mas"]) for r in LOG["coarse_shift"].values()]
        tick(f"coarse shift pass {it}: {len(shifts)} groups, |shift| median {np.median(shifts) if shifts else 0:.0f} mas, max {max(shifts) if shifts else 0:.0f} mas")


def band_median_sigma(c, cover=None):''')
rep('''    if REG.multi and getattr(a, "coarse_shift", 0.0) > 0:  # v30
        t = time.perf_counter()
        coarse_register_hst(exps, thead, dev, a)''', '''    if getattr(a, "coarse_shift_jwst", 0.0) > 0:  # v31: every instrument
        t = time.perf_counter()
        coarse_register_all(exps, thead, dev, a)
        LOG["timings"]["coarse_shift"] = round(time.perf_counter() - t, 1)
    elif REG.multi and getattr(a, "coarse_shift", 0.0) > 0:  # v30
        t = time.perf_counter()
        coarse_register_hst(exps, thead, dev, a)''')
rep('''    ap.add_argument("--coarse-shift", type=float, default=0.0,''', '''    ap.add_argument("--coarse-shift-jwst", type=float, default=0.0, help="v31: translation search radius (arcsec) for JWST exposure groups too (two passes, cross-band references)")
    ap.add_argument("--coarse-shift", type=float, default=0.0,''')
open(dst, "w").write(s)
print("wrote", dst)
