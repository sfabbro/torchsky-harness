"""Is the residual on bright pixels a position error? Per held-out exposure and method: the shift (dx, dy) [pixels] that best explains
d - m as -grad(m) . shift (weighted least squares on pixels with snr_ref >= 20), for the whole cut and for its four quadrants, and chi2/N
on those pixels before and after removing that linear term. A common shift = registration of the exposure; shifts that differ between
quadrants = distortion or item positions. Finite-difference gradients on the pixel grid: crude where the kernel is undersampled."""
import glob
import json
import os
import numpy as np
S = os.environ["ONESKY_SCRATCH"]  # folder holding canfar/ (predictions, masks, xpred) and summary/ (outputs)
C, OUT = S + "/canfar", S + "/summary"
METHODS = {"pipeline": "pred_stage3_blot", "base0": "pred_lib_base_s0", "direct0": "pred_lib_direct_s0", "ks_direct0": "pred_ks_direct_s0", "ks_direct0_400": "pred_ks_direct_s0_400m", "ssf4_200": "pred_ssf4_200m", "ks_aff": "pred_ks_direct_s0_aff"}
PIX = {"F090W": 31.0, "F150W": 31.0, "F200W": 31.0, "F277W": 63.0, "F356W": 63.0, "F444W": 63.0}
res = {k: [] for k in METHODS}
for f in sorted(glob.glob(f"{C}/xpred_lat_base_s0/*.npz")):
    root = os.path.basename(f)[:-4]
    x, m = np.load(f), np.load(f"{C}/heldout/{root}.npz")
    y0, x0 = int(x["y0"]), int(x["x0"]); ny, nx = x["data"].shape; sl = (slice(y0, y0 + ny), slice(x0, x0 + nx))
    d, sig, snr = x["data"].astype(np.float64), m["sigma"][sl].astype(np.float64), m["snr_ref"][sl].astype(np.float64)
    ok0 = m["valid"][sl] & x["valid"].astype(bool) & (sig > 0) & np.isfinite(snr)
    yy, xx = np.mgrid[0:ny, 0:nx]
    for k, folder in METHODS.items():
        mod = np.load(f"{C}/{folder}/{root}.npz")["model"][sl].astype(np.float64)
        gy, gx = np.gradient(np.nan_to_num(mod))
        ok = ok0 & np.isfinite(mod) & (snr >= 20)
        ok[0, :] = ok[-1, :] = False; ok[:, 0] = ok[:, -1] = False
        def solve(sel):
            if sel.sum() < 200: return None
            A = np.stack([-gx[sel] / sig[sel], -gy[sel] / sig[sel]], 1); b = ((d - mod) / sig)[sel]
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            return sol, float((b**2).mean()), float(((b - A @ sol)**2).mean()), int(sel.sum())
        whole = solve(ok)
        if whole is None: continue
        quads = [solve(ok & ((yy >= np.median(yy[ok])) == a) & ((xx >= np.median(xx[ok])) == b)) for a in (False, True) for b in (False, True)]
        qs = np.array([q[0] for q in quads if q is not None])
        res[k].append(dict(root=root, band=str(x["band"]), shift=whole[0].tolist(), chi_before=whole[1], chi_after=whole[2], n=whole[3],
                           quad_scatter=(qs.std(0).tolist() if len(qs) >= 3 else None)))
json.dump(res, open(OUT + "/diag_shift.json", "w"), indent=1)
print("metric: best-fitting shift of the prediction against the data on pixels with snr_ref >= 20 [mas], and chi2/N there before -> after removing it")
for k, rows in res.items():
    sh = np.array([np.array(r["shift"]) * PIX[r["band"]] for r in rows]); mag = np.hypot(sh[:, 0], sh[:, 1])
    qs = np.array([np.hypot(*(np.array(r["quad_scatter"]) * PIX[r["band"]])) for r in rows if r["quad_scatter"]])
    n = np.array([r["n"] for r in rows]); cb = np.array([r["chi_before"] for r in rows]); ca = np.array([r["chi_after"] for r in rows])
    print(f"{k:15s} exposures {len(rows):2d}  |shift| median {np.median(mag):5.2f} max {mag.max():5.2f} mas ({np.median(mag / np.array([PIX[r['band']] for r in rows])):.3f} px median)  "
          f"scatter between quadrants (median) {np.median(qs):5.2f} mas   chi2/N {np.sum(cb * n) / n.sum():6.2f} -> {np.sum(ca * n) / n.sum():6.2f}")
print("\nper exposure, ks_direct0 and ssf4_200: band, shift [mas], chi2/N before -> after")
for a, b in zip(res["ks_direct0"], res["ssf4_200"]):
    sa, sb = np.array(a["shift"]) * PIX[a["band"]], np.array(b["shift"]) * PIX[b["band"]]
    print(f"  {a['root'][-22:]:22s} {a['band']}  ours ({sa[0]:+5.1f}, {sa[1]:+5.1f}) {a['chi_before']:6.2f} -> {a['chi_after']:6.2f}   Survey Fusion ({sb[0]:+5.1f}, {sb[1]:+5.1f}) {b['chi_before']:6.2f} -> {b['chi_after']:6.2f}")
