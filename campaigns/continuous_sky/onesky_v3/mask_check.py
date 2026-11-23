"""How much of a library fit's protocol score comes from pixels its loader masked (so the fit never saw them) but the protocol scores?

    python mask_check.py --heldout DIR_WITH_PROTOCOL_MASKS --xpred RUN/xpred

The protocol's held-out masks (valid, sigma, snr_ref on the full detector) were built on an older calibration of the files; the newer files
flag the brightest cores as saturated, and torchsky's loader drops flagged pixels. For each class of the official mosaic's signal-to-noise:
pixels scored by the protocol, the share the loader masked, and chi2/N on all protocol pixels, on those the loader kept, and on those it
masked. No offset is fitted (the protocol's "without offset" score). Reads the runner's side files (data, valid, model on the cut, y0, x0).
"""
import argparse
import glob
import os

import numpy as np

BINS = [("<1", -np.inf, 1), ("1-5", 1, 5), ("5-20", 5, 20), ("20-50", 20, 50), ("50-200", 50, 200), (">200", 200, np.inf), ("headline >=5", 5, np.inf)]
ap = argparse.ArgumentParser()
ap.add_argument("--heldout", required=True)
ap.add_argument("--xpred", required=True)
a = ap.parse_args()
acc = {b[0]: np.zeros(6) for b in BINS}  # n_all, chi_all, n_kept, chi_kept, n_masked, chi_masked
n_exp = 0
for f in sorted(glob.glob(os.path.join(a.xpred, "*.npz"))):
    root = os.path.basename(f)[:-4]
    mp = os.path.join(a.heldout, root + ".npz")
    if not os.path.exists(mp):
        continue
    x, m = np.load(f), np.load(mp)
    y0, x0 = int(x["y0"]), int(x["x0"])
    ny, nx = x["model"].shape
    sl = (slice(y0, y0 + ny), slice(x0, x0 + nx))
    pv, sig, snr = m["valid"][sl], m["sigma"][sl].astype(np.float64), m["snr_ref"][sl]
    ok = pv & np.isfinite(x["model"]) & np.isfinite(x["data"]) & (sig > 0)
    r2 = ((x["data"].astype(np.float64) - x["model"]) / np.where(sig > 0, sig, 1)) ** 2
    kept = x["valid"].astype(bool)
    n_exp += 1
    for name, lo, hi in BINS:
        s = ok & (snr >= lo) & (snr < hi)
        acc[name] += [s.sum(), r2[s].sum(), (s & kept).sum(), r2[s & kept].sum(), (s & ~kept).sum(), r2[s & ~kept].sum()]
print(f"{n_exp} held-out exposures\n")
print("| Class | protocol pixels | masked by the loader | chi2/N all | chi2/N kept by the loader | chi2/N masked by the loader |")
print("|---|---|---|---|---|---|")
for name, _, _ in BINS:
    n, c, nk, ck, nm, cm = acc[name]
    if n:
        print(f"| {name} | {int(n)} | {100 * nm / n:.2f}% | {c / n:.3f} | {ck / max(nk, 1):.3f} | {(cm / nm if nm else float('nan')):.1f} |")
