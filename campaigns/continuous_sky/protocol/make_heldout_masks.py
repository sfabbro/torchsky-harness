#!/usr/bin/env python3
"""make_heldout_masks.py -- noise factors and held-out pixel masks of evaluation protocol v1 (see INTERFACE.md).

    python make_heldout_masks.py --field SMACS|LMC [--stage noise|masks|all] [--root /arc/projects/ots/adass2026/protocol/v1]

Writes  noise_factors_<FIELD>.json  and  heldout/<FIELD>/<rootname>.npz  for every exposure in the union of the held-out lists of
splits_<FIELD>.json.   Needs the output of official_align.py (aligned WCS asdf files and reference mosaics).

NOISE FACTOR (model independent)   f[detector/filter] = sqrt( <(A - B)^2 / (ERR_A^2 + ERR_B^2)> ) on blank pixels
  Method: pairs of TRAINING exposures of the same detector and filter that overlap on the sky (after the official alignment).  B is
  sampled on A's pixel grid by NEAREST-PIXEL lookup through the aligned WCSs (A pixel -> sky -> B pixel).  Nearest-pixel sampling keeps
  the noise of B independent and unreduced (interpolation would correlate it and lower its variance), at the price of up to half a
  pixel of misregistration; that only matters at sources, so sources are masked.  Blank pixels: both pixels have DQ == 0, finite SCI and
  ERR; the local 5x5 mean of (A+B)/2 is not above the block sky by more than 1.5 sigma_smooth (this and a 3 px dilation mask stars and
  galaxies); a block-wise (128 px) median of A-B is removed (sky/offset differences between the two exposures); then the ratio is taken from
  an iteratively 4-sigma-clipped RMS of (A-B)/sqrt(ERR_A^2+ERR_B^2).  The key's factor is the median over pairs; the spread over pairs, the number
  of pairs and pixels, and the MAD-based value are recorded.  The factor multiplies ERR (so it is a ratio of standard deviations; the earlier
  work found about 1.29 for NIRCam).
  N1  The factors are calibrated on the training exposures of the 'dither' split only (held-out dither exposures never enter) and then used
      for every split of the field (a detector-level property of the pipeline ERR; exposures held out by the visit/epoch/filter splits can
      therefore have contributed).  N2  Pixel noise is assumed independent between A and B and between pixels (correlated read noise,
      1/f stripes and inter-pixel capacitance are not modelled; they would bias the ratio by a few percent).  N3  Blank pixels are
      selected from the data themselves; faint undetected sources that are shifted by the nearest-pixel lookup add a small positive bias.
      N4  The ratio is measured on blank (sky + read-noise dominated) pixels and applied to all pixels, including bright ones, where the
      pipeline Poisson term may be mis-scaled differently.  N5  A key (detector/filter) without pairs gets the median factor of the
      same channel (SW or LW) and is flagged.

MASKS  (heldout/<FIELD>/<rootname>.npz, detector-shaped arrays; outside the scoring boxes: valid False, snr_ref/sigma/noise_factor NaN, box_id -1)
  valid       bool    DQ == 0, finite SCI and ERR, ERR > 0, pixel CENTRE inside a scoring box (tangent-plane square, north up) using the ALIGNED WCS
  snr_ref     float32 SCI/ERR of the reference mosaic (ref_*_i2d.fits), bilinear at the pixel centre; NaN where the mosaic has no data
  sigma       float32 ERR * noise_factor
  noise_factor float32, box_id int16
  For LMC the aligned WCS and reference mosaic of the tile whose box is being tested are used (approximation P1 of official_align.py).
  M1  The box test is done on the full-resolution pixel centres of a window found from a coarse (16 px) sky grid, inflated by 16 px.
  M2  snr_ref uses the mosaic ERR as pipeline-propagated (no noise factor), so it is a pipeline S/N; the mosaic is a cut-out around the box.
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

ARC = "/arc/projects/ots/adass2026"
ROOT = f"{ARC}/protocol/v1"
BLOCK = 128


# ----------------------------------------------------------------------------- WCS helpers
def wcs_file(root_dir, field, tile, rootname):
    base = os.path.join(root_dir, f"align_{field}")
    return os.path.join(base, f"tile{tile}" if field == "LMC" else "", rootname + ".asdf")


def load_wcs(path):
    import asdf
    with asdf.open(path, lazy_load=False, memmap=False) as af:
        return af["wcs"]


def tangent_offsets(ra, dec, ra0, dec0):
    """Standard (tangent-plane) coordinates in arcsec: xi east, eta north, of (ra, dec) deg about (ra0, dec0)."""
    a, d, a0, d0 = np.radians(ra), np.radians(dec), math.radians(ra0), math.radians(dec0)
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return np.degrees(xi) * 3600.0, np.degrees(eta) * 3600.0


def coarse_nodes(n=2048, step=16):
    g = np.unique(np.r_[np.arange(0, n, step), n - 1]).astype(float)
    return g, np.meshgrid(g, g)


def upsample_nodes(g, arr, n=2048):
    """Cubic interpolation of a node grid arr[iy, ix] (nodes at pixel coordinates g) to every pixel."""
    from scipy.ndimage import map_coordinates
    idx = np.interp(np.arange(n), g, np.arange(len(g)))
    yy, xx = np.meshgrid(idx, idx, indexing="ij")
    return map_coordinates(arr, [yy, xx], order=3, mode="nearest")


# ----------------------------------------------------------------------------- noise factor
def read_cal(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as hl:
        return (np.array(hl["SCI"].data, np.float32), np.array(hl["ERR"].data, np.float32), np.array(hl["DQ"].data, np.int32))


def block_median_subtract(x, ok, block=BLOCK):
    out = x.copy()
    n = x.shape[0] // block
    for i in range(n):
        for j in range(n):
            s = (slice(i * block, (i + 1) * block), slice(j * block, (j + 1) * block))
            v = x[s][ok[s]]
            out[s] = x[s] - (np.median(v) if v.size > 50 else 0.0)
    return out


def pair_factor(args):
    """args: (wA, wB, pathA, pathB). Returns dict(f, f_mad, n_pix, overlap) or dict(error)."""
    from scipy.ndimage import uniform_filter, binary_dilation
    wa_path, wb_path, pa, pb = args
    try:
        wa, wb = load_wcs(wa_path), load_wcs(wb_path)
        g, (X, Y) = coarse_nodes()
        ra, dec = wa(X, Y)
        xb, yb = wb.invert(ra, dec, with_bounding_box=False)
        xbf, ybf = upsample_nodes(g, xb), upsample_nodes(g, yb)
        ix, iy = np.rint(xbf).astype(int), np.rint(ybf).astype(int)
        inside = np.isfinite(xbf) & np.isfinite(ybf) & (ix >= 0) & (ix < 2048) & (iy >= 0) & (iy < 2048)
        overlap = float(inside.mean())
        if overlap < 0.2:
            return dict(error="overlap", overlap=overlap)
        sa, ea, da = read_cal(pa)
        sb, eb, db = read_cal(pb)
        ix, iy = np.clip(ix, 0, 2047), np.clip(iy, 0, 2047)
        sbn, ebn, dbn = sb[iy, ix], eb[iy, ix], db[iy, ix]
        ok = inside & (da == 0) & (dbn == 0) & np.isfinite(sa) & np.isfinite(sbn) & np.isfinite(ea) & np.isfinite(ebn) & (ea > 0) & (ebn > 0)
        # source mask from the mean image
        mean = np.where(ok, 0.5 * (sa + sbn), 0.0)
        sm = uniform_filter(mean, 5)
        var_mean = np.where(ok, 0.25 * (ea ** 2 + ebn ** 2), 0.0)
        sig_sm = np.sqrt(uniform_filter(var_mean, 5) / 25.0)  # sigma of the 5x5 mean of (A+B)/2 for independent pixels
        sky = np.zeros_like(sm)
        n = 2048 // BLOCK
        for i in range(n):
            for j in range(n):
                s = (slice(i * BLOCK, (i + 1) * BLOCK), slice(j * BLOCK, (j + 1) * BLOCK))
                v = sm[s][ok[s]]
                sky[s] = np.median(v) if v.size > 50 else np.nan
        src = (sm - sky) > 1.5 * sig_sm
        src = binary_dilation(src | ~np.isfinite(sky), iterations=3)
        blank = ok & ~src
        diff = np.where(blank, sa - sbn, 0.0)
        diff = block_median_subtract(diff, blank)
        zimg = np.where(blank, diff / np.sqrt(ea ** 2 + ebn ** 2), np.nan)
        z = zimg[blank]
        z = z[np.isfinite(z)]
        if z.size < 5000:
            return dict(error="few blank pixels", n_pix=int(z.size), overlap=overlap)
        keep = np.ones(z.size, bool)
        s = z.std()
        for _ in range(8):
            keep = np.abs(z) < 4.0 * s
            s2 = z[keep].std()
            if abs(s2 - s) < 1e-4:
                s = s2
                break
            s = s2
        mad = 1.4826 * float(np.median(np.abs(z - np.median(z))))
        zc = np.where(np.abs(zimg) < 4.0 * s, zimg, np.nan)         # lag-1 correlation of the clipped normalised differences (diagnostic for N2)
        def lag1(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            return float(np.mean(a[m] * b[m]) / np.mean(zc[np.isfinite(zc)] ** 2)) if m.sum() > 1000 else None
        return dict(f=float(s), f_mad=mad, n_pix=int(keep.sum()), overlap=overlap, blank_fraction=float(blank.mean()),
                    rho_x=lag1(zc[:, :-1], zc[:, 1:]), rho_y=lag1(zc[:-1, :], zc[1:, :]))
    except Exception as e:                                   # noqa
        return dict(error=repr(e))


def noise_stage(a, sp):
    rng = np.random.default_rng(a.seed)
    ex = {e["rootname"]: e for e in sp["exposures"]}
    train = [ex[r] for r in sp["splits"]["dither"]["train"]]
    keys = {}
    for e in train:
        keys.setdefault(f'{e["detector"]}/{e["filter"]}', []).append(e)
    jobs, meta = [], []
    for k, es in sorted(keys.items()):
        cand = []
        for i, e1 in enumerate(es):
            for e2 in es[i + 1:]:
                tiles = sorted(set(e1["boxes"]) & set(e2["boxes"])) if a.field == "LMC" else ["all"]
                if not tiles:
                    continue
                if abs(e1["mjd"] - e2["mjd"]) > 5 and a.field == "LMC":
                    continue                                  # same epoch: same orientation, comparable sky
                cand.append((e1, e2, tiles[0]))
        rng.shuffle(cand)
        for e1, e2, t in cand[: a.pairs_per_key * 3]:        # extra candidates; the first usable ones are kept
            wa = wcs_file(a.root, a.field, t, e1["rootname"]); wb = wcs_file(a.root, a.field, t, e2["rootname"])
            if os.path.exists(wa) and os.path.exists(wb):
                jobs.append((wa, wb, e1["path"], e2["path"])); meta.append((k, e1["rootname"], e2["rootname"]))
    print(f"noise: {len(keys)} detector/filter keys, {len(jobs)} candidate pairs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex_:
        res = list(ex_.map(pair_factor, jobs, chunksize=1))
    print(f"noise: pairs done in {time.time() - t0:.0f}s", flush=True)
    per = {}
    for (k, r1, r2), r in zip(meta, res):
        per.setdefault(k, []).append(dict(a=r1, b=r2, **r))
    out_keys, chan = {}, {"SW": [], "LW": []}
    for k in sorted(keys):
        good = [r for r in per.get(k, []) if "f" in r][: a.pairs_per_key]
        if good:
            f = float(np.median([r["f"] for r in good]))
            out_keys[k] = dict(factor=f, n_pairs=len(good), spread_pairs=float(np.std([r["f"] for r in good])), f_mad_median=float(np.median([r["f_mad"] for r in good])),
                               n_pixels=int(sum(r["n_pix"] for r in good)), flag=None)
            chan["LW" if "LONG" in k else "SW"].append(f)
        else:
            out_keys[k] = dict(factor=None, n_pairs=0, flag="no usable pairs", errors=[r.get("error") for r in per.get(k, [])][:3])
    for k, v in out_keys.items():
        if v["factor"] is None:
            c = chan["LW" if "LONG" in k else "SW"]
            v["factor"], v["flag"] = (float(np.median(c)) if c else 1.0), "fallback: median of the same channel (N5)" if c else "fallback 1.0, no pairs anywhere"
    allf = [v["factor"] for v in out_keys.values() if v["flag"] is None]
    doc = dict(field=a.field, protocol="v1", method=__doc__.split("NOISE FACTOR (model independent)")[1].split("MASKS")[0].strip(),
               factors={k: v["factor"] for k, v in out_keys.items()}, details=out_keys, median_factor=float(np.median(allf)) if allf else None,
               pairs=[dict(key=k, **r) for k, rs in sorted(per.items()) for r in rs], seed=a.seed, pairs_per_key=a.pairs_per_key)
    json.dump(doc, open(os.path.join(a.root, f"noise_factors_{a.field}.json"), "w"), indent=1)
    print("noise factors (median over keys): ", doc["median_factor"], flush=True)
    for k, v in sorted(out_keys.items()):
        print(f"  {k}: {v['factor']:.3f} (pairs {v['n_pairs']}, spread {v.get('spread_pairs', float('nan')):.3f}) {v['flag'] or ''}", flush=True)
    return doc


# ----------------------------------------------------------------------------- masks
def ref_path(root, field, tile, filt):
    return os.path.join(root, f"ref_SMACS_{filt}_i2d.fits" if field == "SMACS" else f"ref_LMC_tile{tile}_{filt}_i2d.fits")


def sample_snr(ref, ra, dec):
    """Bilinear S/N of the reference mosaic at (ra, dec); NaN where any of the four neighbours has no data."""
    from scipy.ndimage import map_coordinates
    snr, w = ref
    x, y = w.all_world2pix(ra, dec, 0)
    good = np.isfinite(snr)
    filled = np.where(good, snr, 0.0)
    v = map_coordinates(filled, [y, x], order=1, mode="constant", cval=0.0)
    gm = map_coordinates(good.astype(np.float32), [y, x], order=1, mode="constant", cval=0.0)
    return np.where(gm > 0.999, v, np.nan).astype(np.float32)


def load_ref(path):
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(path) as hl:
        sci, err = np.array(hl["SCI"].data, np.float64), np.array(hl["ERR"].data, np.float64)
        w = WCS(hl["SCI"].header)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.where((err > 0) & np.isfinite(err) & np.isfinite(sci), sci / err, np.nan)
    return snr.astype(np.float32), w


def mask_one(job):
    root_dir, field, e, boxes, factors, chan_fb = job
    try:
        from astropy.io import fits
        sci, err, dq = read_cal(e["path"])
        key = f'{e["detector"]}/{e["filter"]}'
        nf = factors.get(key, chan_fb["LW" if "LONG" in key else "SW"])
        shape = sci.shape
        valid = np.zeros(shape, bool)
        snr_ref = np.full(shape, np.nan, np.float32)
        sigma = np.full(shape, np.nan, np.float32)
        nfac = np.full(shape, np.nan, np.float32)
        box_id = np.full(shape, -1, np.int16)
        g, (X, Y) = coarse_nodes(shape[0])
        used = {}
        for t in sorted(e["boxes"], key=int):
            b = boxes[t]
            wp = wcs_file(root_dir, field, t if field == "LMC" else "all", e["rootname"])
            if not os.path.exists(wp):
                used[t] = "no aligned WCS file"
                continue
            w = load_wcs(wp)
            ra, dec = w(X, Y)
            xi, eta = tangent_offsets(ra, dec, b["ra"], b["dec"])
            h = b["size_arcsec"] / 2.0
            near = np.isfinite(xi) & (np.abs(xi) < h + 2.0) & (np.abs(eta) < h + 2.0)
            if not near.any():
                used[t] = "box not on detector"
                continue
            ys, xs = np.where(near)
            x0, x1 = max(int(X[ys, xs].min()) - 16, 0), min(int(X[ys, xs].max()) + 17, shape[1])
            y0, y1 = max(int(Y[ys, xs].min()) - 16, 0), min(int(Y[ys, xs].max()) + 17, shape[0])
            yy, xx = np.mgrid[y0:y1, x0:x1]
            r, d = w(xx.astype(float), yy.astype(float))
            xi, eta = tangent_offsets(r, d, b["ra"], b["dec"])
            inb = np.isfinite(xi) & (np.abs(xi) <= h) & (np.abs(eta) <= h)
            if not inb.any():
                used[t] = "box not on detector"
                continue
            rp = ref_path(root_dir, field, t, e["filter"])
            if not os.path.exists(rp):
                used[t] = "no reference mosaic"
                continue
            ref = load_ref(rp)
            s = sample_snr(ref, r[inb], d[inb])
            sl = (slice(y0, y1), slice(x0, x1))
            S = np.full(inb.shape, np.nan, np.float32)
            S[inb] = s
            free = inb & (box_id[sl] < 0)                    # a pixel in two boxes keeps the first
            ok = (dq[sl] == 0) & np.isfinite(sci[sl]) & np.isfinite(err[sl]) & (err[sl] > 0)
            box_id[sl][free] = int(t)                        # basic slicing -> views, so boolean assignment writes through
            snr_ref[sl][free] = S[free]
            sigma[sl][free] = (err[sl] * nf)[free]
            nfac[sl][free] = nf
            valid[sl][free] = ok[free]
            used[t] = int(free.sum())
        p = os.path.join(root_dir, "heldout", field, e["rootname"] + ".npz")
        np.savez_compressed(p, valid=valid, snr_ref=snr_ref, sigma=sigma, noise_factor=nfac, box_id=box_id)
        return dict(rootname=e["rootname"], ok=True, n_box_pixels=int((box_id >= 0).sum()), n_valid=int(valid.sum()), boxes=used,
                    n_snr_nan=int((valid & ~np.isfinite(snr_ref)).sum()), noise_factor=float(nf))
    except Exception as ex:                                   # noqa
        import traceback
        return dict(rootname=e["rootname"], ok=False, error=repr(ex), tb=traceback.format_exc())


def mask_stage(a, sp, doc):
    ex = {e["rootname"]: e for e in sp["exposures"]}
    held = sorted({r for s in sp["splits"].values() for r in s["heldout"]})
    if a.only:
        held = [r for r in held if r in set(a.only.split(","))]
    boxes = {str(b["id"]): b for b in sp["boxes"]}
    os.makedirs(os.path.join(a.root, "heldout", a.field), exist_ok=True)
    fac = doc["factors"]
    chan = {"SW": float(np.median([v for k, v in fac.items() if "LONG" not in k] or [1.0])), "LW": float(np.median([v for k, v in fac.items() if "LONG" in k] or [1.0]))}
    jobs = [(a.root, a.field, ex[r], boxes, fac, chan) for r in held]
    print(f"masks: {len(jobs)} held-out exposures", flush=True)
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex_:
        for i, r in enumerate(ex_.map(mask_one, jobs, chunksize=1)):
            res.append(r)
            if not r["ok"]:
                print("FAILED", r["rootname"], r["error"], flush=True)
            elif i % 25 == 0:
                print(f"  {i}/{len(jobs)} {r['rootname']} valid {r['n_valid']} boxpix {r['n_box_pixels']} {r['boxes']} ({time.time() - t0:.0f}s)", flush=True)
    summ = dict(field=a.field, n=len(res), n_failed=sum(not r["ok"] for r in res), n_no_box_pixels=sum(1 for r in res if r["ok"] and r["n_box_pixels"] == 0),
                n_valid_total=int(sum(r.get("n_valid", 0) for r in res if r["ok"])), exposures=res)
    json.dump(summ, open(os.path.join(a.root, f"heldout_{a.field}_summary.json"), "w"), indent=1)
    print(f"masks done: {summ['n'] - summ['n_failed']} ok, {summ['n_failed']} failed, {summ['n_no_box_pixels']} with no box pixels", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--field", required=True, choices=["SMACS", "LMC"])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--stage", default="all", choices=["noise", "masks", "all"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pairs-per-key", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--only", default="", help="debug: comma list of rootnames")
    a = ap.parse_args()
    os.environ.setdefault("CRDS_SERVER_URL", "https://jwst-crds.stsci.edu")
    os.environ.setdefault("CRDS_PATH", f"{ARC}/crds_cache")
    sp = json.load(open(os.path.join(a.root, f"splits_{a.field}.json")))
    doc = None
    if a.stage in ("noise", "all"):
        doc = noise_stage(a, sp)
    if a.stage in ("masks", "all"):
        if doc is None:
            doc = json.load(open(os.path.join(a.root, f"noise_factors_{a.field}.json")))
        mask_stage(a, sp, doc)


if __name__ == "__main__":
    main()
