#!/usr/bin/env python
"""Post-hoc star diagnostics of stage3_blot predictions: for isolated bright stars of the held-out _cal, aperture-sum ratio model/data
vs radius, centroid offset model-data (pixels) and width ratio. Usage: star_diag.py CAL_DIR PRED_DIR [rootname ...]"""
import glob, json, os, sys
import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi

def diag(cal, model, valid, nstars=40, H=20, rads=(2, 3, 4, 6, 8, 12)):
    with fits.open(cal) as h:
        sci = np.array(h["SCI"].data, float); dq = np.array(h["DQ"].data)
    ok = np.isfinite(sci) & ((dq & 1) == 0) & valid
    d = np.where(ok, sci, np.nan); m = np.where(ok, model, np.nan)
    sm = ndi.gaussian_filter(np.where(ok, sci, 0.0), 1.5)
    med = np.nanmedian(d); mad = 1.4826 * np.nanmedian(np.abs(d - med))
    pk = (sm == ndi.maximum_filter(sm, size=9)) & (sm - med > 15 * mad / 3.0) & ok
    ys, xs = np.nonzero(pk); o = np.argsort(-sm[ys, xs]); ys, xs = ys[o], xs[o]
    yy, xx = np.mgrid[-H:H + 1, -H:H + 1]; rr = np.hypot(yy, xx)
    rows = []
    for y, x in zip(ys, xs):
        if len(rows) >= nstars: break
        if y < H or x < H or y >= sci.shape[0] - H or x >= sci.shape[1] - H: continue
        sl = (slice(y - H, y + H + 1), slice(x - H, x + H + 1))
        if not ok[sl].all() or ((dq[sl] & 2) != 0).any(): continue
        if (pk[sl] & (rr > 0) & (rr < 16)).any(): continue
        an = (rr >= 16) & (rr <= 20)
        dd = d[sl] - np.nanmedian(d[sl][an]); mm = m[sl] - np.nanmedian(m[sl][an])
        ratios = []
        for r in rads:
            a = rr <= r
            ratios.append(mm[a].sum() / dd[a].sum() if dd[a].sum() > 0 else np.nan)
        a = rr <= 4
        w = np.clip(dd, 0, None) * a; wm = np.clip(mm, 0, None) * a
        if w.sum() <= 0 or wm.sum() <= 0: continue
        cd = ((w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()); cm = ((wm * xx).sum() / wm.sum(), (wm * yy).sum() / wm.sum())
        sd = np.sqrt((w * rr ** 2).sum() / w.sum()); smod = np.sqrt((wm * rr ** 2).sum() / wm.sum())
        rows.append([int(x), int(y), float(dd[rr <= 4].sum())] + ratios + [cm[0] - cd[0], cm[1] - cd[1], smod / sd])
    return np.array(rows), rads

if __name__ == "__main__":
    cal_dir, pred_dir = sys.argv[1:3]
    for p in sorted(glob.glob(os.path.join(pred_dir, "*.npz"))):
        rn = os.path.basename(p)[:-4]
        if len(sys.argv) > 3 and rn not in sys.argv[3:]: continue
        z = np.load(p, allow_pickle=False)
        rows, rads = diag(os.path.join(cal_dir, rn + "_cal.fits"), z["model"], z["model_valid"])
        if len(rows) == 0: print(rn, "no stars"); continue
        n = len(rads)
        print(rn[-16:], "N=%d" % len(rows), "ratio(r=%s)=" % ",".join(map(str, rads)), np.round(np.nanmedian(rows[:, 3:3 + n], 0), 3),
              "dcentroid px (x,y) med %.2f %.2f  rms %.2f" % (np.median(rows[:, 3 + n]), np.median(rows[:, 4 + n]), np.sqrt(np.mean(rows[:, 3 + n:5 + n] ** 2))),
              "width ratio %.2f" % np.median(rows[:, 5 + n]))
