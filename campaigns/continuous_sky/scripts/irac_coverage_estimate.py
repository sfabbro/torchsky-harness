#!/usr/bin/env python3
"""Estimate, for ALL IRAC frames of the union manifest, how many detector pixels have their whole patch inside JWST NIRCam LW coverage.

Why an estimate: only a small sample of the 3486 IRAC frames and a few NIRCam exposures are on the local disk. This script uses only the
manifests: (i) IRAC frame footprint = the 4 corners of datasearch/manifest_spitzer_union.csv (footprint_region), pixel centres by bilinear
interpolation of the corners (no SIP distortion; error << 1 arcsec for the 5.2' frames); (ii) JWST coverage per band = union of the 4-corner
S_REGION polygons of the NIRCam LW exposures (NRCALONG / NRCBLONG) of datasearch/manifest_jwst_cal_union.csv, rasterised on a 1 arcsec
tangent grid (polygon edge accuracy ~ 1 arcsec; detector gaps, DQ holes and the splat-coadd edge are NOT modelled);
(iii) 'patch inside coverage' = the pixel lies inside the band mask eroded by a disk of radius 5.1 arcsec (= half-diagonal of the
25 x 25 x 0.3 arcsec patch: conservative, the true patch is a rotated square of half-width 3.6 arcsec).
Required bands (default F356W and F444W, as learned_instrument.py): coverage = intersection. bimsk (fatal bits) is not applied
(it removes ~0.5-2 percent of pixels in the local sample).

Use: python3 irac_coverage_estimate.py --out usable_estimate.json [--programs 2736,4043,6882] [--erode 5.1]
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import binary_erosion

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "datasearch")
RA0, DEC0 = 110.8020, -73.4586


def parse_poly(s):
    v = s.replace("POLYGON", "").replace("ICRS", "").split()
    v = np.array([float(x) for x in v]).reshape(-1, 2)
    return v[:4] if len(v) > 4 else v


def tan_offsets(ra, dec):
    """arcsec west (x), north (y) about (RA0, DEC0), gnomonic."""
    ra, dec = np.radians(ra), np.radians(dec)
    r0, d0 = math.radians(RA0), math.radians(DEC0)
    c = np.sin(d0) * np.sin(dec) + np.cos(d0) * np.cos(dec) * np.cos(ra - r0)
    xi = np.cos(dec) * np.sin(ra - r0) / c
    eta = (np.cos(d0) * np.sin(dec) - np.sin(d0) * np.cos(dec) * np.cos(ra - r0)) / c
    return -np.degrees(xi) * 3600.0, np.degrees(eta) * 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--programs", default="2736,4043,6882")
    ap.add_argument("--bands", default="F356W,F444W")
    ap.add_argument("--erode", type=float, default=5.1)
    ap.add_argument("--grid", type=float, default=1.0, help="mask pixel [arcsec]")
    ap.add_argument("--half", type=float, default=700.0, help="half-size of the mask grid [arcsec]")
    a = ap.parse_args()
    progs, bands = set(a.programs.split(",")), a.bands.split(",")
    n = int(2 * a.half / a.grid)
    ax = (np.arange(n) + 0.5) * a.grid - a.half
    gx, gy = np.meshgrid(ax, ax)  # gx: arcsec west, gy: north
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    masks = {b: np.zeros(n * n, bool) for b in bands}
    npoly = defaultdict(int)
    for r in csv.DictReader(open(os.path.join(DATA, "manifest_jwst_cal_union.csv"))):
        if r["program"] not in progs or r["instrument"] != "NIRCAM" or "LONG" not in r["detector"] or r["filter"] not in bands:
            continue
        if r["pupil"] not in ("CLEAR", "") or r.get("exp_type", "NRC_IMAGE") not in ("NRC_IMAGE", ""):
            continue
        poly = parse_poly(r["s_region"])
        x, y = tan_offsets(poly[:, 0], poly[:, 1])
        p = Path(np.stack([x, y], 1))
        lo, hi = np.array([x.min(), y.min()]), np.array([x.max(), y.max()])
        sel = np.nonzero(np.all((pts >= lo - 2) & (pts <= hi + 2), 1))[0]
        masks[r["filter"]][sel[p.contains_points(pts[sel])]] = True
        npoly[r["filter"]] += 1
    cov = np.ones(n * n, bool)
    for b in bands:
        cov &= masks[b]
    cov = cov.reshape(n, n)
    rad = int(round(a.erode / a.grid))
    yy, xx = np.mgrid[-rad: rad + 1, -rad: rad + 1]
    safe = binary_erosion(cov, structure=(xx ** 2 + yy ** 2) <= rad ** 2, border_value=0)
    print("polygons per band", dict(npoly), "coverage arcsec^2:", cov.sum() * a.grid ** 2, "patch-safe:", safe.sum() * a.grid ** 2)

    frames = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(os.path.join(DATA, "manifest_spitzer_union.csv"))):
        if r["file_type"] != "cbcd":
            continue
        frames[r["aor"]][r["detector"]].append((r["filename"], r["program"], parse_poly(r["footprint_region"])))
    s = (np.arange(256) + 0.5) / 256.0
    S, T = np.meshgrid(s, s)
    S, T = S.ravel(), T.ravel()
    out, tot = {}, 0
    for aor in sorted(frames):
        rec = dict(program=None, n_frames=0, n_pixels=0, n_usable=0, per_channel={})
        for det, fl in sorted(frames[aor].items()):
            ch = rec["per_channel"].setdefault(det, dict(n_frames=0, n_frames_with_usable=0, n_usable=0))
            for fn, prog, poly in fl:
                rec["program"] = prog
                C = poly
                ra = (1 - S) * (1 - T) * C[0, 0] + S * (1 - T) * C[1, 0] + S * T * C[2, 0] + (1 - S) * T * C[3, 0]
                dec = (1 - S) * (1 - T) * C[0, 1] + S * (1 - T) * C[1, 1] + S * T * C[2, 1] + (1 - S) * T * C[3, 1]
                x, y = tan_offsets(ra, dec)
                ix, iy = np.floor((x + a.half) / a.grid).astype(int), np.floor((y + a.half) / a.grid).astype(int)
                ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
                k = int(safe[iy[ok], ix[ok]].sum())
                rec["n_frames"] += 1
                rec["n_pixels"] += 65536
                rec["n_usable"] += k
                ch["n_frames"] += 1
                ch["n_frames_with_usable"] += int(k > 0)
                ch["n_usable"] += k
        tot += rec["n_usable"]
        out[aor] = rec
    json.dump(dict(method=__doc__.strip().split("\n")[0], programs=sorted(progs), bands=bands, erode_arcsec=a.erode,
                   jwst_polygons=dict(npoly), coverage_arcsec2=float(cov.sum() * a.grid ** 2), patch_safe_arcsec2=float(safe.sum() * a.grid ** 2),
                   aors=out, total_usable=tot, total_frames=sum(v["n_frames"] for v in out.values())), open(a.out, "w"), indent=1)
    for aor, r in out.items():
        print(f"AOR {aor} program {r['program']}: {r['n_frames']} frames, usable {r['n_usable']} "
              f"({', '.join(f'{d}: {c['n_usable']} in {c['n_frames_with_usable']}/{c['n_frames']} frames' for d, c in r['per_channel'].items())})")
    print("total usable", tot)


if __name__ == "__main__":
    main()
