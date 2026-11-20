#!/usr/bin/env python3
"""Sanity check of the independent renderer against the package's own prediction of a held-out exposure (pred/*.npz written by the
fit, model incl. background, detector pixels): the brightest isolated peaks of the package prediction (above an annulus level) are
matched by their mutual separations to peaks of my full-kernel render on the 0.04" sky grid (q4_wing_<F>.npz, 'full'), and the peak
heights above the local level are compared. Held-out exposures: jw02736001001_02101_00009_nrcb3 (F090W, from the dither group / filter
pairing in metrics.json val_groups: 02101 = F090W + F277W, 02105 = F200W + F444W) and jw02736001001_02105_00002_nrcblong (F444W)."""
import itertools
import json
import math
import os
import sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kc_common import HERE, SKY_PITCH


def peaks(img, valid, pitch, n=6, sep=1.0, ann=(0.6, 1.0)):
    d = np.where(valid, img, np.nan)
    mx = ndimage.maximum_filter(np.nan_to_num(d, nan=-1e9), size=7)
    ii, jj = np.nonzero((np.nan_to_num(d, nan=-1e9) == mx) & valid)
    out = []
    for i, j in zip(ii, jj):
        _r1, r2 = int(ann[0] / pitch), int(ann[1] / pitch) + 1
        i0, i1, j0, j1 = max(i - r2, 0), min(i + r2 + 1, d.shape[0]), max(j - r2, 0), min(j + r2 + 1, d.shape[1])
        sl = d[i0:i1, j0:j1]
        yy, xx = np.mgrid[i0:i1, j0:j1]
        rr = np.hypot(yy - i, xx - j) * pitch
        a = sl[(rr >= ann[0]) & (rr <= ann[1]) & np.isfinite(sl)]
        if a.size < 10:
            continue
        out.append((float(d[i, j] - np.median(a)), i, j))
    out.sort(reverse=True)
    keep = []
    for o in out:
        if all(math.hypot((o[1] - k[1]) * pitch, (o[2] - k[2]) * pitch) > sep for k in keep):
            keep.append(o)
        if len(keep) >= n:
            break
    return keep


res = {}
for fn, filt, pix in (("pred_b3.npz", "F090W", 0.031), ("pred_long.npz", "F444W", 0.063)):
    p = np.load(os.path.join(HERE, "dl", fn), allow_pickle=True)
    P = peaks(p["model"], p["model_valid"], pix, n=4)
    w = np.load(os.path.join(HERE, "results", f"q4_wing_{filt}.npz"))
    full = w["full"]
    N = full.shape[0]
    g = (np.arange(N) - (N - 1) / 2) * SKY_PITCH
    inner = (np.abs(g)[:, None] <= 10.24) & (np.abs(g)[None, :] <= 10.24)
    S = peaks(full, inner, SKY_PITCH, n=8)
    # match: the 3 brightest prediction peaks against all ordered triples of render peaks by their pairwise distances
    def dists(pts, s):
        return np.array([math.hypot((a[1] - b[1]) * s, (a[2] - b[2]) * s) for a, b in itertools.combinations(pts, 2)])
    dP = dists(P[:3], pix)
    best = min(itertools.permutations(range(len(S)), 3), key=lambda t: np.abs(dists([S[k] for k in t], SKY_PITCH) - dP).sum())
    err = float(np.abs(dists([S[k] for k in best], SKY_PITCH) - dP).max())
    pairs = [dict(package_peak=P[m][0], my_render_peak=S[k][0], ratio=S[k][0] / P[m][0]) for m, k in enumerate(best)]
    res[filt] = dict(max_separation_mismatch_arcsec=err, pairs=pairs)
    print(filt, "separation mismatch %.3f\"" % err, [(round(q["package_peak"], 3), round(q["my_render_peak"], 3), round(q["ratio"], 3)) for q in pairs])
json.dump(res, open(os.path.join(HERE, "results", "crosscheck_package_prediction.json"), "w"), indent=1)
