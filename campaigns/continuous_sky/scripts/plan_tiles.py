#!/usr/bin/env python3
"""plan_tiles.py: choose T disjoint sky tiles of a multi-pointing NIRCam `_cal` data set for multi-tile instrument training (experiment P1).

Reads HEADERS ONLY (primary: DETECTOR, FILTER, PUPIL, EXP_TYPE, EXPSTART; SCI: FITS WCS incl. SIP, NAXIS1/2, MJD-AVG). No pixel is read.

WHAT IT DOES
  1. Every usable `_cal` file (one detector image = one "exposure") gets a footprint polygon: the array boundary, inset by --edge-px
     (reference pixels / detector edge), sampled every ~64 pixels and transformed with the full SCI WCS (astropy, SIP included).
  2. All footprints are rasterised on ONE north-up gnomonic grid (cell = size/--cells-per-tile, default 20.48/32 = 0.64 arcsec) about the
     mean pointing. For a candidate tile (a square of --size-arcsec) the box sums of the rasters give, per exposure, the covered area fraction and the
     fraction of the tile falling in each of the 3 x 3 cells of that detector.
  3. Candidates are tile lower-left corners on the lattice (stride --cand-stride cells, default 4 = 2.56 arcsec). A candidate is ELIGIBLE if in each
     requested band at least --n-min exposures cover >= --cov-frac of the tile (default 4 exposures, 0.9; with 2 held-out dithers per band
     at least 2 remain for training).
  4. Greedy selection of --n-tiles tiles (greedy maximisation of a monotone submodular coverage function, so >= 1 - 1/e of the best possible set for
     the objective as defined below): at each step choose the eligible candidate that adds the most NEW (detector, 3x3-cell) positions (a position is
     "reached" by a covering exposure if >= --cell-frac (0.15) of the tile area falls in that cell); ties: more new (detector, band, cell) positions,
     then the deeper min-band coverage, then closeness to the centroid of the eligible region. Selected tiles must not overlap: the
     Chebyshev distance between tile boxes is >= --gap-arcsec (default 2.0 arcsec = twice the 24-pixel data pad of smacs_four_v38 at 0.04 arcsec/px, so
     the LOADED pixel sets of two tiles are disjoint; the sky-grid halo of 64 px is not loaded data and is ignored).
  5. For each chosen tile the exposure lists (per band: covering ones and partial ones with cover >= --partial-frac), roll, MJD, cells reached,
     and coverage statistics are written to JSON; a PNG shows the footprints, the minimum-over-bands depth map and the tiles.

APPROXIMATIONS (also written to the JSON under "approximations")
  * Footprints are polygons through the sampled array boundary (full WCS); no DQ / bad-pixel / gap information (headers only): the usable pixel count of a tile
    in the real run is smaller than the geometric cover fraction says (tilefit reports the real counts).
  * The 3 x 3 cell of a grid point is found with an AFFINE fit of the pixel <-> sky map of the detector (error of a few pixels of a 683-pixel cell).
  * The tile is a square of the planner's gnomonic grid (north-up, about the mean pointing); the tile actually fitted by tilefit.py is the north-up TAN
    square about the tile centre: differences are projection terms of a few mas over the field (neglected). Tile centres are snapped to the
    candidate lattice and the cover fractions are evaluated on cells of size/cells-per-tile (default 0.64 arcsec): cover fractions are good to ~1/32 of the tile side.
  * "Exposure" = one detector image (a dither of both modules is several exposures). N_min counts detector images, not dithers; the number of distinct
    dithers is reported per tile.
  * A (detector, cell) position is counted once whichever band reaches it ((detector, band, cell) is the secondary objective).
  * Greedy, not optimal; deterministic.

CLI:  plan_tiles.py --inputs DIR_OR_GLOB [DIR_OR_GLOB ...] --out plan.json [--png plan.png] [--bands F090W,...] [--n-tiles 8] [--size-arcsec 20.48]
      [--n-min 4] [--cov-frac 0.9] [--gap-arcsec 2.0] [--edge-px 8] [--cell-frac 0.15] [--cand-stride 4] [--cells-per-tile 32]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

warnings.filterwarnings("ignore", category=FITSFixedWarning)
import logging
logging.getLogger("astropy").setLevel(logging.ERROR)

DEFAULT_BANDS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
CLEAR = ("", "CLEAR", "CLEARP", "OPEN", "N/A", "NONE")
NAME_RE = re.compile(r"^(?:(?P<idx>\d+)_)?jw(?P<prog>\d{5})(?P<obs>\d{3})(?P<vis>\d{3})_(?P<grp>[0-9a-z]{5})_(?P<exp>\d{5})_(?P<det>[a-z0-9]+)(?:_|\.|$)", re.I)


# ----------------------------------------------------------------------------- tangent plane
def tan_forward(ra, dec, ra0, dec0):
    """RA, Dec [deg] -> (east, north) offsets [arcsec] on the gnomonic plane about (ra0, dec0)."""
    a, d, a0, d0 = map(np.radians, (np.asarray(ra, float), np.asarray(dec, float), ra0, dec0))
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    x = np.cos(d) * np.sin(a - a0) / cosc
    y = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return np.degrees(x) * 3600.0, np.degrees(y) * 3600.0


def tan_inverse(xe, yn, ra0, dec0):
    """(east, north) [arcsec] -> RA, Dec [deg]."""
    x, y = np.radians(np.asarray(xe, float) / 3600.0), np.radians(np.asarray(yn, float) / 3600.0)
    a0, d0 = np.radians(ra0), np.radians(dec0)
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.arcsin(np.cos(c) * np.sin(d0) + np.where(rho > 0, y * np.sin(c) * np.cos(d0) / rho, 0.0))
    a = a0 + np.arctan2(x * np.sin(c), rho * np.cos(d0) * np.cos(c) - y * np.sin(d0) * np.sin(c))
    return np.degrees(a) % 360.0, np.degrees(d)


# ----------------------------------------------------------------------------- headers
def expand_inputs(items):
    files = []
    for it in items:
        if os.path.isdir(it):
            files += glob.glob(os.path.join(it, "**", "*_cal.fits"), recursive=True)
        else:
            files += glob.glob(it, recursive=True)
    return sorted(set(os.path.abspath(f) for f in files))


def read_info(path, bands):
    """Header-only record of one file, or {"skip": reason}."""
    try:
        h0 = fits.getheader(path, 0)
        hs = fits.getheader(path, "SCI")
    except Exception as exc:  # noqa: BLE001
        return {"skip": f"unreadable headers: {exc!r}"}
    g = lambda k: str(h0.get(k, "") or "").strip().upper()
    ins, det, f, p, et = g("INSTRUME"), g("DETECTOR"), g("FILTER"), g("PUPIL"), g("EXP_TYPE")
    if ins != "NIRCAM":
        return {"skip": f"INSTRUME {ins!r} is not NIRCAM"}
    if et and not et.endswith("_IMAGE"):
        return {"skip": f"EXP_TYPE {et} is not imaging"}
    cands = [x for x in (f, p) if x in bands]
    others = [x for x in (f, p) if x not in bands]
    if len(cands) != 1 or any(x not in CLEAR for x in others):
        return {"skip": f"FILTER/PUPIL {f!r}/{p!r} is not exactly one of the requested bands"}
    if "CTYPE1" not in hs or "NAXIS1" not in hs:
        return {"skip": "no FITS WCS (CTYPE1) / NAXIS in the SCI header"}
    try:
        w = WCS(hs)
    except Exception as exc:  # noqa: BLE001
        return {"skip": f"WCS failed: {exc!r}"}
    nx, ny = int(hs["NAXIS1"]), int(hs["NAXIS2"])
    mjd = hs.get("MJD-AVG", hs.get("MJD-BEG", h0.get("EXPSTART", float("nan"))))
    cd = w.pixel_scale_matrix if hasattr(w, "pixel_scale_matrix") else None
    pa_y = float("nan")
    if cd is not None:
        cosd = math.cos(math.radians(float(hs.get("CRVAL2", 0.0))))
        pa_y = math.degrees(math.atan2(cd[0, 1] * cosd, cd[1, 1]))  # PA of the detector +y axis east of north
    return {"path": path, "name": os.path.basename(path), "detector": det, "band": cands[0], "mjd": float(mjd), "nx": nx, "ny": ny, "wcs": w, "pa_y_deg": pa_y,
            "expstart": float(h0.get("EXPSTART", float("nan")))}


def dither_key(name):
    m = NAME_RE.match(name)
    return None if not m else f"{m['obs']}{m['vis']}_{m['grp']}_{m['exp']}"


def footprint(info, edge, ra0, dec0, nper=64):
    """Polygon (n,2) [arcsec east, north] of the array boundary inset by `edge` px, and the affine map sky(arcsec) -> pixel (x, y)."""
    nx, ny = info["nx"], info["ny"]
    t = np.linspace(edge, nx - 1 - edge, nper)
    u = np.linspace(edge, ny - 1 - edge, nper)
    xs = np.concatenate([t, np.full(nper, nx - 1 - edge), t[::-1], np.full(nper, edge)])
    ys = np.concatenate([np.full(nper, edge), u, np.full(nper, ny - 1 - edge), u[::-1]])
    ra, dec = info["wcs"].all_pix2world(xs, ys, 0)
    e, n = tan_forward(ra, dec, ra0, dec0)
    poly = np.stack([e, n], 1)
    # affine pixel = A [e, n, 1] from a 5x5 lattice of the whole array
    gx, gy = np.meshgrid(np.linspace(0, nx - 1, 5), np.linspace(0, ny - 1, 5))
    r2, d2 = info["wcs"].all_pix2world(gx.ravel(), gy.ravel(), 0)
    e2, n2 = tan_forward(r2, d2, ra0, dec0)
    X = np.stack([e2, n2, np.ones_like(e2)], 1)
    coef, *_ = np.linalg.lstsq(X, np.stack([gx.ravel(), gy.ravel()], 1), rcond=None)  # (3,2)
    return poly, coef


# ----------------------------------------------------------------------------- planning
def integral(a):
    I = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.float64)
    I[1:, 1:] = a.cumsum(0).cumsum(1)
    return I


def box_sum(I, v0, v1, u0, u1):
    """sum of the rasters over rows [v0, v1) and columns [u0, u1) (arrays, clipped to the raster)."""
    h, w = I.shape[0] - 1, I.shape[1] - 1
    v0, v1 = np.clip(v0, 0, h), np.clip(v1, 0, h)
    u0, u1 = np.clip(u0, 0, w), np.clip(u1, 0, w)
    return I[v1, u1] - I[v0, u1] - I[v1, u0] + I[v0, u0]


def rasterise(info, edge, ra0, dec0, g):
    """(v_origin, u_origin, inside[bool], cellcode[int8, -1 outside]) on the global grid (cell index = floor(offset / g))."""
    from matplotlib.path import Path

    poly, coef = footprint(info, edge, ra0, dec0)
    pad = g
    u_lo, u_hi = poly[:, 0].min() - pad, poly[:, 0].max() + pad
    v_lo, v_hi = poly[:, 1].min() - pad, poly[:, 1].max() + pad
    iu0, iu1 = int(math.floor(u_lo / g)), int(math.ceil(u_hi / g))
    iv0, iv1 = int(math.floor(v_lo / g)), int(math.ceil(v_hi / g))
    cu, cv = (np.arange(iu0, iu1) + 0.5) * g, (np.arange(iv0, iv1) + 0.5) * g
    U, V = np.meshgrid(cu, cv)
    pts = np.stack([U.ravel(), V.ravel()], 1)
    inside = Path(poly).contains_points(pts).reshape(U.shape)
    xy = np.stack([pts[:, 0], pts[:, 1], np.ones(len(pts))], 1) @ coef
    cx = np.clip(np.floor(3 * xy[:, 0] / info["nx"]), 0, 2).astype(int)
    cy = np.clip(np.floor(3 * xy[:, 1] / info["ny"]), 0, 2).astype(int)
    code = np.where(inside.ravel(), cy * 3 + cx, -1).astype(np.int8).reshape(U.shape)
    return iv0, iu0, inside, code


def plan(files, a):
    t0 = time.time()
    bands = [b.strip().upper() for b in a.bands.split(",")]
    infos, skipped = [], []
    for f in files:
        r = read_info(f, bands)
        (skipped.append([os.path.basename(f), r["skip"]]) if "skip" in r else infos.append(r))
    if not infos:
        raise SystemExit("no usable files")
    # reference point: mean of the array centres
    cen = np.array([i["wcs"].all_pix2world(i["nx"] / 2, i["ny"] / 2, 0) for i in infos], float)
    ra0 = float(np.degrees(np.arctan2(np.sin(np.radians(cen[:, 0])).mean(), np.cos(np.radians(cen[:, 0])).mean())) % 360.0)
    dec0 = float(cen[:, 1].mean())
    g = a.size_arcsec / a.cells_per_tile
    Ks = int(a.cells_per_tile)
    st = int(a.cand_stride)
    dets = sorted({i["detector"] for i in infos})
    bidx = {b: k for k, b in enumerate(bands)}
    didx = {d: k for k, d in enumerate(dets)}
    rasters = []
    for i in infos:
        rasters.append(rasterise(i, a.edge_px, ra0, dec0, g))
    ivmin = min(r[0] for r in rasters) - Ks
    iumin = min(r[1] for r in rasters) - Ks
    ivmax = max(r[0] + r[2].shape[0] for r in rasters)
    iumax = max(r[1] + r[2].shape[1] for r in rasters)
    # candidate lower-left corners (global cell indices), aligned to the stride
    cvs = np.arange(math.ceil(ivmin / st) * st, ivmax + 1, st)
    cus = np.arange(math.ceil(iumin / st) * st, iumax + 1, st)
    while len(cvs) * len(cus) > 250000:
        st *= 2
        cvs = np.arange(math.ceil(ivmin / st) * st, ivmax + 1, st)
        cus = np.arange(math.ceil(iumin / st) * st, iumax + 1, st)
    CU, CV = np.meshgrid(cus, cvs)
    shp = CU.shape
    ncand = CU.size
    cover_cnt = np.zeros((len(bands), ncand), np.int16)
    pos = np.zeros((ncand, len(dets) * 9), bool)
    posb = np.zeros((ncand, len(dets) * len(bands) * 9), bool)
    area = float(Ks * Ks)

    def evaluate(info, ras, cu, cv):
        v0, u0, inside, code = ras
        sh = (cv - v0, cv - v0 + Ks, cu - u0, cu - u0 + Ks)
        cov = box_sum(integral(inside.astype(np.float64)), *sh) / area
        cells = []
        for k in range(9):
            cells.append(box_sum(integral((code == k).astype(np.float64)), *sh) / area)
        return cov, np.stack(cells, 0)

    for info, ras in zip(infos, rasters):
        v0, u0, inside, code = ras
        sel = ((CV + Ks > v0) & (CV < v0 + inside.shape[0]) & (CU + Ks > u0) & (CU < u0 + inside.shape[1])).ravel()
        if not sel.any():
            continue
        cov, cells = evaluate(info, ras, CU.ravel()[sel], CV.ravel()[sel])
        ok = cov >= a.cov_frac
        ids = np.nonzero(sel)[0][ok]
        cover_cnt[bidx[info["band"]], ids] += 1
        for k in range(9):
            hit = ids[cells[k][ok] >= a.cell_frac]
            pos[hit, didx[info["detector"]] * 9 + k] = True
            posb[hit, (didx[info["detector"]] * len(bands) + bidx[info["band"]]) * 9 + k] = True
    minb = cover_cnt.min(0)
    eligible = minb >= a.n_min
    n_el = int(eligible.sum())
    print(f"[plan] {len(infos)} usable files ({len(skipped)} skipped), {len(dets)} detectors, grid cell {g:.3f}\", {ncand} candidates, {n_el} eligible "
          f"(>= {a.n_min} covering exposures in every band) [{time.time() - t0:.0f} s]", flush=True)
    # greedy
    chosen, covered, coveredb = [], np.zeros(pos.shape[1], bool), np.zeros(posb.shape[1], bool)
    gap = int(math.ceil(a.gap_arcsec / g))
    if n_el:
        cu_e, cv_e = CU.ravel()[eligible], CV.ravel()[eligible]
        cen_u, cen_v = cu_e.mean(), cv_e.mean()
    alive = eligible.copy()
    for t in range(a.n_tiles):
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            print(f"[plan] only {len(chosen)} tiles fit (no eligible non-overlapping candidate left)", flush=True)
            break
        new = (pos[idx] & ~covered).sum(1)
        newb = (posb[idx] & ~coveredb).sum(1)
        dist = np.hypot(CU.ravel()[idx] - cen_u, CV.ravel()[idx] - cen_v)
        dist = dist / max(dist.max(), 1.0)
        score = new * 1e7 + newb * 1e4 + np.minimum(minb[idx], 999) * 1.0 + (1.0 - dist) * 0.5
        j = int(idx[int(np.argmax(score))])
        chosen.append({"cand": j, "new_positions": int((pos[j] & ~covered).sum()), "new_positions_band": int((posb[j] & ~coveredb).sum())})
        covered |= pos[j]
        coveredb |= posb[j]
        cu_j, cv_j = int(CU.ravel()[j]), int(CV.ravel()[j])
        alive &= ~((np.abs(CU.ravel() - cu_j) < Ks + gap) & (np.abs(CV.ravel() - cv_j) < Ks + gap))
    # per-tile details
    tiles = []
    for t, c in enumerate(chosen):
        j = c["cand"]
        cu_j, cv_j = int(CU.ravel()[j]), int(CV.ravel()[j])
        e0, n0 = cu_j * g, cv_j * g
        ec, nc = e0 + 0.5 * Ks * g, n0 + 0.5 * Ks * g
        ra_c, dec_c = tan_inverse(ec, nc, ra0, dec0)
        per_band, dith, cells_set = {b: [] for b in bands}, set(), set()
        pa, mjd = [], []
        for info, ras in zip(infos, rasters):
            v0, u0, inside, code = ras
            if not (cv_j + Ks > v0 and cv_j < v0 + inside.shape[0] and cu_j + Ks > u0 and cu_j < u0 + inside.shape[1]):
                continue
            cov, cells = evaluate(info, ras, np.array([cu_j]), np.array([cv_j]))
            cov = float(cov[0])
            if cov < a.partial_frac:
                continue
            cl = {int(k): round(float(cells[k][0]), 3) for k in range(9) if cells[k][0] > 0.0}
            rec = {"file": info["name"], "path": info["path"], "detector": info["detector"], "cover_frac": round(cov, 3), "covering": bool(cov >= a.cov_frac),
                   "cells": cl, "mjd": info["mjd"], "pa_y_deg": round(info["pa_y_deg"], 3), "dither": dither_key(info["name"])}
            per_band[info["band"]].append(rec)
            if rec["covering"]:
                dith.add((info["band"], rec["dither"] or info["name"]))
                pa.append(info["pa_y_deg"])
                mjd.append(info["mjd"])
                for k, fr in cl.items():
                    if fr >= a.cell_frac:
                        cells_set.add((info["detector"], k))
        for b in bands:
            per_band[b].sort(key=lambda r: (-r["cover_frac"], r["file"]))
        ncov = {b: sum(r["covering"] for r in per_band[b]) for b in bands}
        tiles.append({
            "id": t, "ra": float(ra_c), "dec": float(dec_c), "size_arcsec": a.size_arcsec, "offset_east_north_arcsec": [round(ec, 3), round(nc, 3)],
            "n_covering": ncov, "n_partial": {b: len(per_band[b]) - ncov[b] for b in bands},
            "n_dithers_covering": {b: len({d for bb, d in dith if bb == b}) for b in bands},
            "n_distinct_det_cell": len(cells_set), "det_cells": sorted(f"{d}:{k}" for d, k in cells_set),
            "new_det_cell_when_chosen": c["new_positions"], "new_det_band_cell_when_chosen": c["new_positions_band"],
            "pa_y_deg_range": [round(float(min(pa)), 2), round(float(max(pa)), 2)] if pa else None, "n_distinct_pa_0.5deg": len({round(p / 0.5) for p in pa}),
            "mjd_range": [round(float(min(mjd)), 3), round(float(max(mjd)), 3)] if mjd else None,
            "exposures": per_band, "files": sorted({r["path"] for b in bands for r in per_band[b]})})
    avail = len(dets) * 9
    out = {
        "version": 1, "tiles": tiles, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bands": bands, "detectors": dets,
        "params": {k: getattr(a, k) for k in ("n_tiles", "size_arcsec", "n_min", "cov_frac", "partial_frac", "gap_arcsec", "edge_px", "cell_frac", "cand_stride", "cells_per_tile")},
        "grid": {"ra0": ra0, "dec0": dec0, "cell_arcsec": g, "candidate_stride_cells_used": st, "n_candidates": int(ncand), "n_eligible": n_el},
        "summary": {"n_tiles": len(tiles), "n_files_used": len(infos), "n_files_skipped": len(skipped),
                    "distinct_det_cell_positions": int(covered.sum()), "available_det_cell_positions": avail,
                    "distinct_det_band_cell_positions": int(coveredb.sum()), "available_det_band_cell_positions": len(dets) * len(bands) * 9,
                    "cumulative_det_cell": list(np.cumsum([c["new_positions"] for c in chosen]).astype(int).tolist()),
                    "detectors_with_cells_reached": sorted({p.split(":")[0] for t_ in tiles for p in t_["det_cells"]})},
        "skipped_files": skipped[:200],
        "approximations": ["footprints from the sampled array boundary with the full WCS (no DQ/bad-pixel/gap map: real usable pixel counts are lower)",
                           "3x3 cell of a grid point from an affine fit (few-pixel error against 683-px cells)",
                           "tile = square of the planner's north-up gnomonic grid; tilefit builds a TAN square about the tile centre (mas differences)",
                           "centres snapped to the candidate lattice; cover fractions good to ~1/cells_per_tile of the side",
                           "exposure = detector image; N_min is not a count of dithers",
                           "(detector, cell) position counted once whichever band reaches it; greedy, not optimal"],
    }
    return out, infos, rasters, (ra0, dec0, g, CU, CV, minb, eligible)


def make_png(out, infos, rasters, aux, path, size):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ra0, dec0, g, CU, CV, minb, eligible = aux
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2), constrained_layout=True)
    mod_col = {"A": "#4c78a8", "B": "#e08a2c"}
    for info in infos:
        poly, _ = footprint(info, 8, ra0, dec0, 8)
        c = mod_col.get(info["detector"][3:4], "#888888")
        ax[0].plot(*np.vstack([poly, poly[:1]]).T, color=c, lw=0.4, alpha=0.35)
    mappable = ax[1].pcolormesh(CU[0] * g, CV[:, 0] * g, minb.reshape(CU.shape)[:-1, :-1].astype(float), shading="flat", cmap="Blues", vmin=0)
    cb = fig.colorbar(mappable, ax=ax[1], shrink=0.8)
    cb.set_label("minimum over bands of covering exposures (tile lower-left corner)")
    for t in out["tiles"]:
        e, n = t["offset_east_north_arcsec"]
        s = t["size_arcsec"]
        for a_ in ax:
            a_.add_patch(Rectangle((e - s / 2, n - s / 2), s, s, fill=False, ec="#c0392b", lw=1.8))
            a_.text(e, n, str(t["id"]), color="#c0392b", ha="center", va="center", fontsize=9, fontweight="bold")
    for a_ in ax:
        a_.set_aspect("equal")
        a_.invert_xaxis()
        a_.set_xlabel("East offset [arcsec]")
        a_.set_ylabel("North offset [arcsec]")
    ax[0].set_title("Detector footprints (blue module A, orange module B) and tiles")
    ax[1].set_title(f"Coverage depth; red = {len(out['tiles'])} tiles of {size:g} arcsec")
    s = out["summary"]
    fig.suptitle(f"{s['distinct_det_cell_positions']}/{s['available_det_cell_positions']} (detector, 3x3 cell) positions reached; "
                 f"{s['n_files_used']} files", fontsize=10)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="directories (searched recursively for *_cal.fits) and/or globs")
    ap.add_argument("--out", required=True, help="plan JSON")
    ap.add_argument("--png", default=None)
    ap.add_argument("--bands", default=",".join(DEFAULT_BANDS))
    ap.add_argument("--n-tiles", type=int, default=8)
    ap.add_argument("--size-arcsec", type=float, default=20.48)
    ap.add_argument("--n-min", type=int, default=4, help="minimum number of covering exposures (detector images) in EVERY band")
    ap.add_argument("--cov-frac", type=float, default=0.9, help="area fraction of the tile an exposure must cover to count as covering")
    ap.add_argument("--partial-frac", type=float, default=0.05, help="exposures covering less (but at least this) are listed as partial")
    ap.add_argument("--gap-arcsec", type=float, default=2.0)
    ap.add_argument("--edge-px", type=float, default=8.0, help="detector edge / reference-pixel inset [detector px]")
    ap.add_argument("--cell-frac", type=float, default=0.15, help="tile area fraction in a 3x3 cell for the position to count as reached")
    ap.add_argument("--cand-stride", type=int, default=4, help="candidate lattice stride in grid cells")
    ap.add_argument("--cells-per-tile", type=int, default=32)
    a = ap.parse_args(argv)
    files = expand_inputs(a.inputs)
    if not files:
        raise SystemExit("no *_cal.fits files found")
    out, infos, rasters, aux = plan(files, a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for t in out["tiles"]:
        print(f"[plan] tile {t['id']}: RA {t['ra']:.5f} Dec {t['dec']:.5f}  covering {t['n_covering']}  det-cells {t['n_distinct_det_cell']} "
              f"(+{t['new_det_cell_when_chosen']} new)  dithers {t['n_dithers_covering']}", flush=True)
    s = out["summary"]
    print(f"[plan] {s['n_tiles']} tiles; distinct (detector, cell) positions {s['distinct_det_cell_positions']}/{s['available_det_cell_positions']}; "
          f"(detector, band, cell) {s['distinct_det_band_cell_positions']}/{s['available_det_band_cell_positions']}", flush=True)
    if a.png:
        make_png(out, infos, rasters, aux, a.png, a.size_arcsec)
    return out


if __name__ == "__main__":
    main()
