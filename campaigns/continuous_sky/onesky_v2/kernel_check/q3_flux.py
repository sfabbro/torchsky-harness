#!/usr/bin/env python3
"""Question 3: flux scale of the fitted sky against the official pipeline mosaic (calwebb_image3 i2d, MJy/sr), per filter.

Units of the fitted sky (from the code): input level 'cal' (cli default, metrics.json config input_level = cal), so the model is
  m = sum_q w_q exp(gamma_q) [k * s](lambda_q) + background, in the units of the _cal SCI, MJy/sr (bandpass.py: "a flat f_nu sky gives
  the cal surface brightness when gamma = 0"; the throughput gauge F17 makes the mean gamma of the training exposures zero at every node).
  The kernel has unit integral (F7). Hence the band image sum_k Phi_band[f, k] A_k is a surface brightness in MJy/sr at gamma = 0, the
  same unit as the mosaic. No conversion is applied besides multiplying surface brightness by area (both images: arcsec^2; the
  mosaic pixel area from PIXAR_A2, the sky node cell = 0.04"^2). Fluxes are given in microjansky: MJy/sr x arcsec^2 x 1e12 / 4.2545e10.

Geometry: the mosaic is a north-up gnomonic projection about the box centre (CRVAL = box centre), x to the west, exactly the frame of
the sky grid (sky.py), so sky node (i, j) at X = (j - 306) 0.04", Y = (i - 306) 0.04" maps linearly onto mosaic pixels. A residual
offset between the two (the fit's frame is the mean header WCS of its training exposures, the mosaic is tweakreg-aligned) is measured
from the centroids of the bright compact sources and removed.

Aperture photometry (identical code for both images): each image cell is split into 7 x 7 sub-cells, a sub-cell counts when its centre
is inside the circle; local background = 3-sigma-clipped median of the cells in an annulus.  The sky band image is treated as an image
of cells of 0.04" holding the node values (the integral of the tent interpolant equals the node sum x pitch^2 away from edges).
Run: python3 q3_flux.py   (about 1 min)
"""
import json
import math
import os

import numpy as np
from astropy.io import fits
from scipy import ndimage

from kc_common import FILTERS, HERE, SKY_PITCH, Fit

UJY = 1e12 / 4.2545170296e10     # MJy/sr x arcsec^2 -> microjansky
BOXH = 10.24


def load_mosaic(f):
    with fits.open(os.path.join(HERE, "dl", f"cut_{f}.fits")) as hl:
        h = hl[0].header
        d = hl[0].data.astype(np.float64)
    scale = abs(h["CDELT1"]) * 3600 if "CDELT1" in h else abs(h["CD1_1"]) * 3600
    assert abs(h["CRVAL1"] - 110.8268) < 1e-9 and abs(h["CRVAL2"] + 73.4544) < 1e-9
    assert h.get("PC1_1", -1) < 0 or h.get("CD1_1", -1) < 0          # x to the west
    # image coordinates: X (west) = (col - (CRPIX1 - 1)) * scale, Y (north) = (row - (CRPIX2 - 1)) * scale
    return d, scale, h["CRPIX1"] - 1, h["CRPIX2"] - 1, float(h["PIXAR_A2"])


class Img:
    """An image of square cells: value d[row, col] at X = (col - c0) p, Y = (row - r0) p (arcsec); cell area a (arcsec^2)."""

    def __init__(self, d, p, c0, r0, a):
        self.d, self.p, self.c0, self.r0, self.a = d, p, c0, r0, a

    def to_px(self, X, Y):
        return X / self.p + self.c0, Y / self.p + self.r0

    def weights(self, X, Y, r, sub=7):
        """fractional coverage of cells by the circle of radius r about (X, Y): (rows, cols, w)."""
        cx, cy = self.to_px(X, Y)
        rp = r / self.p
        j0, j1 = int(math.floor(cx - rp - 1)), int(math.ceil(cx + rp + 1))
        i0, i1 = int(math.floor(cy - rp - 1)), int(math.ceil(cy + rp + 1))
        jj, ii = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1))
        s = (np.arange(sub) + 0.5) / sub - 0.5
        sx, sy = np.meshgrid(s, s)
        dx = jj[..., None, None] + sx - cx
        dy = ii[..., None, None] + sy - cy
        w = ((dx ** 2 + dy ** 2) <= rp ** 2).mean((-1, -2))
        ok = (ii >= 0) & (jj >= 0) & (ii < self.d.shape[0]) & (jj < self.d.shape[1])
        return ii[ok], jj[ok], w[ok]

    def annulus_level(self, X, Y, r1, r2):
        cx, cy = self.to_px(X, Y)
        j0, j1 = int(cx - r2 / self.p - 1), int(cx + r2 / self.p + 2)
        i0, i1 = int(cy - r2 / self.p - 1), int(cy + r2 / self.p + 2)
        jj, ii = np.meshgrid(np.arange(max(j0, 0), min(j1, self.d.shape[1])), np.arange(max(i0, 0), min(i1, self.d.shape[0])))
        rr = np.hypot((jj - cx) * self.p, (ii - cy) * self.p)
        v = self.d[ii, jj][(rr >= r1) & (rr <= r2)]
        v = v[np.isfinite(v)]
        for _ in range(5):
            m, s = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))
            v = v[np.abs(v - m) <= 3 * s + 1e-30]
        return float(np.median(v))

    def phot(self, X, Y, r, ann):
        ii, jj, w = self.weights(X, Y, r)
        b = self.annulus_level(X, Y, *ann)
        v = self.d[ii, jj]
        good = np.isfinite(v)
        return float(((v[good] - b) * w[good]).sum() * self.a * UJY), b

    def centroid(self, X, Y, r=0.12):
        ii, jj, w = self.weights(X, Y, r, sub=3)
        v = np.clip(self.d[ii, jj] - self.annulus_level(X, Y, 0.5, 0.8), 0, None) * w
        Xc = ((jj - self.c0) * self.p * v).sum() / v.sum()
        Yc = ((ii - self.r0) * self.p * v).sum() / v.sum()
        return Xc, Yc


def find_peaks(img, n=80, lim=8.5, sep=0.8):
    """Local maxima of an image (5x5 window) inside |X|,|Y| <= lim, brightest first, with no brighter peak within `sep` arcsec."""
    d = np.nan_to_num(img.d, nan=-1e30)
    mx = ndimage.maximum_filter(d, size=5)
    ii, jj = np.nonzero((d == mx) & (d > np.nanpercentile(img.d, 90)))
    X, Y = (jj - img.c0) * img.p, (ii - img.r0) * img.p
    ok = (np.abs(X) <= lim) & (np.abs(Y) <= lim)
    ii, jj, X, Y = ii[ok], jj[ok], X[ok], Y[ok]
    o = np.argsort(-d[ii, jj])
    keep = []
    for k in o:
        if all(math.hypot(X[k] - X[m], Y[k] - Y[m]) > sep for m in keep):
            keep.append(k)
        if len(keep) >= n:
            break
    return [(float(X[k]), float(Y[k]), float(d[ii[k], jj[k]])) for k in keep]


def isolated_compact(mos, peaks, n_want=20):
    """Isolated: no other peak brighter than 30 % of it within 0.8". Compact: mosaic flux within 0.15" / within 0.5" (local background
    removed); among peaks with mosaic flux > 0.3 microjansky within 0.5", the n_want most compact of the isolated peaks with positive fluxes are kept (a point source gives about 0.75 at 2 um in the
    mosaic; few are present in this cluster field, so compact galaxies enter the sample: the ratio is listed per source)."""
    out = []
    for k, (X, Y, v) in enumerate(peaks):
        if any(math.hypot(X - X2, Y - Y2) < 0.8 and v2 > 0.3 * v for m, (X2, Y2, v2) in enumerate(peaks) if m != k):
            continue
        f1, _ = mos.phot(X, Y, 0.15, (0.6, 1.0))
        f2, _ = mos.phot(X, Y, 0.5, (0.6, 1.0))
        if f2 > 0.3 and f1 > 0:      # 0.3 microjansky within 0.5" in the mosaic: bright enough for a ratio to 10 per cent
            out.append((X, Y, v, f1 / f2))
    out.sort(key=lambda t: -t[3])
    return out[:n_want]


def source_level(img, inside):
    """Blank level of an image region: 3-sigma-clipped median of the cells inside the region after masking cells above median + 3 sigma
    (dilated by 0.3")."""
    v = img.d[inside]
    m, s = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))
    src = np.zeros_like(img.d, bool)
    src[inside] = img.d[inside] > m + 3 * s
    src = ndimage.binary_dilation(src, iterations=max(1, int(0.3 / img.p)))
    v = img.d[inside & ~src]
    for _ in range(5):
        m, s = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))
        v = v[np.abs(v - m) <= 3 * s]
    return float(np.median(v))


def main():
    fit = Fit()
    N = fit.N
    res = {}
    for f in range(6):
        name = FILTERS[f]
        B = fit.band_image(f).double().numpy()
        sky = Img(B, SKY_PITCH, (N - 1) / 2, (N - 1) / 2, SKY_PITCH ** 2)
        d, p, c0, r0, a = load_mosaic(name)
        mos = Img(d, p, c0, r0, a)
        # sources: peaks of the fitted sky band image, compactness / isolation judged on the mosaic
        peaks = find_peaks(sky, n=200)
        comp = isolated_compact(mos, peaks)
        # astrometric offset sky - mosaic from centroids of the 15 brightest compact sources
        offs = np.array([np.subtract(sky.centroid(X, Y), mos.centroid(X, Y)) for X, Y, _, _ in comp[:15]])
        dX, dY = np.median(offs, 0)
        mos.c0 -= dX / p     # shift the mosaic frame so that its sources sit at the sky positions
        mos.r0 -= dY / p
        rows = []
        for X, Y, v, cmp in comp:
            fs, bs = sky.phot(X, Y, 0.3, (0.6, 1.0))
            fm, bm = mos.phot(X, Y, 0.3, (0.6, 1.0))
            rows.append(dict(X=X, Y=Y, compactness=cmp, sky_uJy=fs, mosaic_uJy=fm, ratio=fs / fm if fm > 0 else None, sky_bg=bs, mos_bg=bm))
        rat = np.array([r["ratio"] for r in rows if r["ratio"] is not None and r["mosaic_uJy"] > 0])
        # (b) bright galaxies: brightest peaks of the MOSAIC-at-sky-positions by 1" flux, centres within 7" of the box centre
        gal = []
        for X, Y, v in find_peaks(sky, n=200, lim=7.0, sep=2.0):
            fm1, _ = mos.phot(X, Y, 1.0, (2.5, 3.5))
            gal.append((fm1, X, Y))
        gal = sorted(gal, reverse=True)[:8]
        grows = []
        for _, X, Y in gal:
            o = dict(X=X, Y=Y)
            for r, ann in ((0.3, (0.6, 1.0)), (1.0, (2.5, 3.5)), (2.0, (3.0, 4.0))):
                fs, _ = sky.phot(X, Y, r, ann)
                fm, _ = mos.phot(X, Y, r, ann)
                o[f"r{r}"] = dict(sky_uJy=fs, mosaic_uJy=fm, ratio=fs / fm if fm > 0 else None)
            grows.append(o)
        # (c) totals over the box |X|, |Y| <= 10.24"
        def box_mask(img):
            jj, ii = np.meshgrid(np.arange(img.d.shape[1]), np.arange(img.d.shape[0]))
            X, Y = (jj - img.c0) * img.p, (ii - img.r0) * img.p
            return (np.abs(X) <= BOXH) & (np.abs(Y) <= BOXH) & np.isfinite(img.d)
        ms, mm = box_mask(sky), box_mask(mos)
        ls, lm = source_level(sky, ms), source_level(mos, mm)
        area = (2 * BOXH) ** 2
        tot = dict(sky_raw_uJy=float(sky.d[ms].sum() * sky.a * UJY), mosaic_raw_uJy=float(mos.d[mm].sum() * mos.a * UJY),
                   sky_blank_level_MJysr=ls, mosaic_blank_level_MJysr=lm,
                   sky_minus_level_uJy=float((sky.d[ms] - ls).sum() * sky.a * UJY), mosaic_minus_level_uJy=float((mos.d[mm] - lm).sum() * mos.a * UJY),
                   box_area_arcsec2=area)
        tot["ratio_raw"] = tot["sky_raw_uJy"] / tot["mosaic_raw_uJy"]
        tot["ratio_level_subtracted"] = tot["sky_minus_level_uJy"] / tot["mosaic_minus_level_uJy"]
        tot["ratio_sky_raw_vs_mosaic_minus_level"] = tot["sky_raw_uJy"] / tot["mosaic_minus_level_uJy"]
        g = {r: np.array([o[f"r{r}"]["ratio"] for o in grows if o[f"r{r}"]["ratio"]]) for r in (0.3, 1.0, 2.0)}
        res[name] = dict(offset_sky_minus_mosaic_arcsec=[float(dX), float(dY)], offset_scatter_arcsec=float(np.std(offs, 0).max()),
                         compact_n=len(rat), compact_ratio_median=float(np.median(rat)),
                         compact_ratio_scatter_1p4826MAD=float(1.4826 * np.median(np.abs(rat - np.median(rat)))),
                         compact_ratio_p16_p84=[float(np.percentile(rat, 16)), float(np.percentile(rat, 84))], compact=rows,
                         galaxies=grows, galaxy_ratio_median={str(r): float(np.median(v)) for r, v in g.items()},
                         galaxy_ratio_minmax={str(r): [float(v.min()), float(v.max())] for r, v in g.items()}, totals=tot,
                         mosaic_pixel_arcsec=p, mosaic_pixar_a2=a)
        print(f"{name}: offset {dX * 1e3:+.0f},{dY * 1e3:+.0f} mas; compact r=0.3\" n={len(rat)} ratio {np.median(rat):.3f} "
              f"(scatter {res[name]['compact_ratio_scatter_1p4826MAD']:.3f}); galaxies median r0.3/1/2: "
              f"{[round(float(np.median(v)), 3) for v in g.values()]}; total raw {tot['ratio_raw']:.3f} level-sub {tot['ratio_level_subtracted']:.3f} "
              f"(levels sky {ls:.4f} mos {lm:.4f} MJy/sr)", flush=True)
    with open(os.path.join(HERE, "results", "q3_flux.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
