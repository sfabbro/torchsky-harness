"""Build B, test 2 (stand-alone check, no fit): coarse shift of real HST WFC3/IR exposures of the SMACS field against a reference image built
from two JWST calibrated exposures, compared with the offset between the file's two header solutions.

Reference: the SCI planes of the JWST exposures (DQ == 0, finite) are splatted (bilinear, inverse-variance not used: plain mean) at their header
(world coordinate) positions onto a tangent-plane grid of 0.05 arcsec (X west, Y north, field frame at the mean JWST pointing); each is high-passed
on its own by the same function the fit uses, divided by its rms, and the two are added where both exist. For each HST exposure the reference
(before high-pass) is smoothed by the filter's point-spread function (Gaussian of the diffraction FWHM, plus the pixel) and sampled at the
a priori header positions of the HST pixels; then fieldop.coarse_reg.coarse_shift is used exactly as in the fit (data against this image, on
the exposure's own pixels).
Independent estimate: archive solution minus a priori solution (both from the file's headers, through the package's own reader class) evaluated
at the pixel that the a priori solution puts at the field centre, as tangent-plane arcsec (X west, Y north): the same convention as the shift.

    PYTHONNOUSERSITE=1 python tests/check_coarse_reg_hst.py OUT.json [--per-filter 7] [--skip n --take m]
"""
import argparse
import glob
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
warnings.filterwarnings("ignore")
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from fieldop.coarse_reg import coarse_shift, high_pass, masked_smooth  # noqa: E402
from fieldop.instruments import AstropyWCS  # noqa: E402

FWHM = {"F105W": 0.151, "F125W": 0.160, "F140W": 0.172, "F160W": 0.183}      # WFC3/IR diffraction-limited FWHM [arcsec]
PITCH = 0.05


def tangent(ra, dec, ra0, dec0):
    """Gnomonic tangent-plane arcsec, X toward the west, Y toward the north."""
    ra, dec, ra0, dec0 = map(np.radians, (np.asarray(ra, float), np.asarray(dec, float), ra0, dec0))
    c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / c
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / c
    k = 180 / math.pi * 3600
    return -xi * k, eta * k


def jwst_points(path):
    h = fits.open(path)
    w = WCS(h[("SCI", 1)].header, fobj=h)
    sci, dq = h[("SCI", 1)].data.astype(np.float64), h[("DQ", 1)].data
    ny, nx = sci.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = w.all_pix2world(xx.ravel(), yy.ravel(), 0)
    good = ((dq == 0) & np.isfinite(sci)).ravel()
    return ra, dec, sci.ravel(), good


def splat(X, Y, val, good, x0, N):
    gx, gy = (X[good] - x0) / PITCH, (Y[good] - x0) / PITCH
    v = val[good]
    num, den = np.zeros(N * N), np.zeros(N * N)
    ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - ix, gy - iy
    for dy in (0, 1):
        for dx in (0, 1):
            jx, jy = ix + dx, iy + dy
            ww = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
            ok = (jx >= 0) & (jx < N) & (jy >= 0) & (jy < N)
            idx = jy[ok] * N + jx[ok]
            np.add.at(num, idx, (v * ww)[ok])
            np.add.at(den, idx, ww[ok])
    img = np.where(den > 0.05, num / np.maximum(den, 1e-9), 0.0).reshape(N, N)
    return torch.tensor(img), torch.tensor((den > 0.05).reshape(N, N))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-filter", type=int, default=7)
    ap.add_argument("--filters", default="F105W,F125W,F140W,F160W")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--take", type=int, default=1000)
    ap.add_argument("--radius", type=float, default=1.5)
    a = ap.parse_args()
    t0 = time.perf_counter()
    jfiles = sorted(glob.glob("/home/claude/work/dl_full/*_cal.fits"))
    pts = [jwst_points(p) for p in jfiles]
    ra0 = float(np.mean([p[0][p[3]].mean() for p in pts]))
    dec0 = float(np.mean([p[1][p[3]].mean() for p in pts]))
    P = [(tangent(p[0], p[1], ra0, dec0), p[2], p[3]) for p in pts]
    half = max(float(np.abs(np.concatenate([q[0][0][q[2]], q[0][1][q[2]]])).max()) for q in P) + 5.0
    N = int(math.ceil(2 * half / PITCH)) + 1
    x0 = -(N - 1) / 2 * PITCH
    ref_hp, ref_valid = torch.zeros(N, N, dtype=torch.float64), torch.zeros(N, N, dtype=torch.bool)
    for (X, Y), val, good in P:
        img, v = splat(X, Y, val, good, x0, N)
        hp = high_pass(img, v, PITCH, 0.6, 0.0)
        y, vv = hp
        y = y / float(y[vv].std())
        ref_hp += torch.where(vv, y, torch.zeros_like(y))
        ref_valid |= vv
    # the same sum, not high-passed (for the HST point-spread function smoothing, applied after the high-pass: a linear operation)
    print(f"reference grid {N} x {N} at {PITCH} arcsec, field centre {ra0:.5f} {dec0:.5f}, built in {time.perf_counter() - t0:.1f} s", flush=True)
    files = sorted(glob.glob("/home/claude/work/dl_hst/*_flt.fits"))
    by = {}
    for f in files:
        flt = fits.getheader(f)["FILTER"]
        by.setdefault(flt, []).append(f)
    sel = []
    for flt in a.filters.split(","):
        sel += [(flt, f) for f in by.get(flt, [])]
    print("HST files by filter:", {k: len(v) for k, v in by.items()}, flush=True)
    rows = []
    cnt = {}
    done = 0
    for flt, f in sel:
        cnt[flt] = cnt.get(flt, 0)
        if cnt[flt] >= a.per_filter:
            continue
        if done < a.skip:
            cnt[flt] += 1
            done += 1
            continue
        if len(rows) >= a.take:
            break
        h = fits.open(f)
        ap_w, ar_w = AstropyWCS(h, 1, "apriori"), AstropyWCS(h, 1, "archive")
        sci, err, dq = h[("SCI", 1)].data.astype(np.float64), h[("ERR", 1)].data, h[("DQ", 1)].data
        ny, nx = sci.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        pix = torch.tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float64)
        rd = ap_w.pixel_to_world_differentiable(pix, 0).numpy()
        Xh, Yh = tangent(rd[:, 0], rd[:, 1], ra0, dec0)
        Xh, Yh = Xh.reshape(ny, nx), Yh.reshape(ny, nx)
        # footprint: fraction of the HST pixels inside the reference grid
        gx, gy = (Xh - x0) / PITCH, (Yh - x0) / PITCH
        inside = (gx > 1) & (gx < N - 2) & (gy > 1) & (gy < N - 2)
        cnt[flt] += 1
        if inside.mean() < 0.15:
            print(f"{os.path.basename(f)} {flt}: footprint inside reference only {inside.mean():.2f}; skipped", flush=True)
            continue
        # reference smoothed by the point-spread function, sampled at the HST pixels
        s_px = math.sqrt((FWHM[flt] / 2.3548) ** 2 + 0.128 ** 2 / 12) / PITCH
        sm = masked_smooth(ref_hp, ref_valid, s_px)
        vsm = masked_smooth(ref_valid.to(torch.float64), torch.ones_like(ref_valid), s_px) > 0.5
        gxt, gyt = torch.tensor(gx), torch.tensor(gy)
        grid = torch.stack([gxt / (N - 1) * 2 - 1, gyt / (N - 1) * 2 - 1], -1)[None]
        m = torch.nn.functional.grid_sample(sm[None, None], grid, mode="bilinear", align_corners=True, padding_mode="zeros")[0, 0]
        vm = torch.nn.functional.grid_sample(vsm.to(torch.float64)[None, None], grid, mode="nearest", align_corners=True, padding_mode="zeros")[0, 0] > 0.5
        # the sampled image is already high-passed on the reference grid; coarse_shift high-passes both again (harmless, block 0.6 arcsec)
        J = np.array([[np.gradient(Xh, axis=1).mean(), np.gradient(Xh, axis=0).mean()], [np.gradient(Yh, axis=1).mean(), np.gradient(Yh, axis=0).mean()]])
        vd = torch.tensor(((dq & ~528) == 0) & np.isfinite(sci) & np.isfinite(err) & (err > 0))
        r = coarse_shift(torch.tensor(sci), vd, m, vm & vd, J, a.radius, 0.6, 0.1)
        # independent: archive minus a priori at the pixel the a priori solution puts at the field centre
        cpix = ap_w.w.all_world2pix([ra0], [dec0], 0)
        c = torch.tensor([[float(cpix[0][0]), float(cpix[1][0])]], dtype=torch.float64)
        cx, cy = float(c[0, 0]), float(c[0, 1])
        pc = torch.tensor([[min(max(cx, 0), nx - 1), min(max(cy, 0), ny - 1)]], dtype=torch.float64)
        ra_p, de_p = ap_w.pixel_to_world_differentiable(pc, 0).numpy()[0]
        ra_a, de_a = ar_w.pixel_to_world_differentiable(pc, 0).numpy()[0]
        tp = np.array(tangent(ra_p, de_p, ra0, dec0))
        ta = np.array(tangent(ra_a, de_a, ra0, dec0))
        ind = ta - tp
        row = dict(file=os.path.basename(f), filter=flt, ok=bool(r["ok"]), reason=r.get("reason"), tX_mas=r["tX"] * 1e3, tY_mas=r["tY"] * 1e3,
                   peak=r["peak"], ratio=r["ratio"], n_overlap=r["n_overlap"], archive_minus_apriori_X_mas=float(ind[0] * 1e3),
                   archive_minus_apriori_Y_mas=float(ind[1] * 1e3), field_centre_pixel=[cx, cy], inside_fraction=float(inside.mean()),
                   diff_X_mas=r["tX"] * 1e3 - float(ind[0] * 1e3), diff_Y_mas=r["tY"] * 1e3 - float(ind[1] * 1e3),
                   visit=os.path.basename(f)[:6])
        rows.append(row)
        print(f"{row['file']} {flt} ok={row['ok']} shift ({row['tX_mas']:+.1f},{row['tY_mas']:+.1f}) mas peak {row['peak']:.3f} ratio {row['ratio']:.2f} "
              f"| archive-apriori ({row['archive_minus_apriori_X_mas']:+.1f},{row['archive_minus_apriori_Y_mas']:+.1f}) | diff ({row['diff_X_mas']:+.1f},{row['diff_Y_mas']:+.1f}) "
              f"[{time.perf_counter() - t0:.0f} s]", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)
        if time.perf_counter() - t0 > 200:
            print("time limit of the call reached; rerun with --skip", flush=True)
            break


if __name__ == "__main__":
    main()
