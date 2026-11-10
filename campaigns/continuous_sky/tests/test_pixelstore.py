"""Tests of scripts/pixelstore.py on 1-2 sample files per mission (testdata/union_samples/).

Run:  python3 -m pytest -s tests/test_pixelstore.py      (the -s shows the printed unit/size tables)
      python3 tests/test_pixelstore.py                    (same)
The sample store is built once into testdata/union_samples/_store (idempotent: a re-run only re-checks).
(a) round trip  (b) halo duplication + exactly one in_core  (c) positions vs astropy all_pix2world < 1 mas
(d) units sanity per mission  (e) SPHEREx lambda in the detector range  (f) reader -> smacs_four_v31 structures
(g) size estimate of the full union from the samples.
"""
import csv
import re
import glob
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pixelstore as ps  # noqa: E402

warnings.filterwarnings("ignore", category=FITSFixedWarning)
pytestmark = pytest.mark.filterwarnings("ignore")
SAMPLES = ROOT / "testdata" / "union_samples"
STORE = SAMPLES / "_store_parts"        # part-file layout (built with --local-dir)
FRAG_STORE = SAMPLES / "_store"        # fragment layout (older store, kept for the layout-equivalence test)
LOCAL = SAMPLES / "_local"
ASEC = 1.0 / 3600.0

# (mission, file) samples, 1-2 per mission
SAMPLE_FILES = {
    "jwst": ["jwst/jw02736001001_02101_00002_nrcb3_cal.fits", "jwst/jw02736001001_02101_00002_nrcblong_cal.fits",
             "jwst/jw04043001001_14101_00002_nrca1_cal.fits", "jwst/jw02736003001_02103_00002_nis_cal.fits",
             "jwst/jw02736002001_02103_00008_mirimage_cal.fits"],
    "hst": ["hst/iczgzbmaq_flt.fits", "hst/jczgz3k6q_flc.fits", "hst/iepl84jtq_flc.fits"],
    "spitzer": ["spitzer/SPITZER_I1_48186112_0000_0000_2_cbcd.fits", "spitzer/SPITZER_I2_48186112_0005_0000_2_cbcd.fits"],
    "spherex": ["spherex/level2_2025W30_1B_0429_3D2_spx_l2b-v20-2025-262_cutout10arcmin.fits",
                "spherex/level2_2025W28_1B_0606_2D6_spx_l2b-v20-2025-260_cutout10arcmin.fits",
                "spherex/level2_2026W38_2A_0141_1D4_spx_l2b-v27-2026-264_cutout10arcmin.fits"],
    "galex": ["galex/AIS_427_0001_sg47/AIS_427_0001_sg47-nd-int.fits.gz", "galex/AIS_427_0001_sg47/AIS_427_0001_sg47-fd-int.fits.gz"],
}
TIMES = {}


@pytest.fixture(scope="session")
def store():
    for mission, files in SAMPLE_FILES.items():
        have = [str(SAMPLES / r) for r in files if (SAMPLES / r).exists()]
        assert len(have) >= 1, f"no sample files for {mission} in {SAMPLES}"
        t0 = time.perf_counter()
        ps.main(["extract", "--store", str(STORE), "--local-dir", str(LOCAL), "--mission", mission] + sum([["--inputs", h] for h in have], []))
        TIMES[mission] = time.perf_counter() - t0
    ps.main(["finalize", "--store", str(STORE)])
    return STORE


@pytest.fixture(scope="session")
def exposures(store):
    return {r["exposure_id"]: r for r in ps.read_exposures(store)}


@pytest.fixture(scope="session")
def grid(store):
    cfg = ps.load_store_config(store)
    return ps.Grid(cfg["tile_arcsec"], cfg["box_arcsec"], cfg["center_ra"], cfg["center_dec"], cfg["halo_arcsec"], cfg["pos_quant_bits"])


_PIECES = {}


def read_exposure_rows(store, eid, columns=None):
    """All stored rows of one exposure over all tiles, as a pyarrow table (fragment order)."""
    key = str(store)
    if key not in _PIECES:
        _PIECES[key] = ps.scan_pieces(store)
    t, files = ps.exposure_table(store, eid, columns, pieces=_PIECES[key])
    return t, files


def _strip_sip(ih):
    import re
    ih = ih.copy()
    for k in [k for k in ih.keys() if re.match(r"^(A|B|AP|BP)_(\d+_\d+|ORDER)$", k)]:
        del ih[k]
    return ih


def inv_gnomonic(x, y, ra0, dec0):
    """Independent inverse of the plane convention: x west, y north, arcsec -> (ra, dec) deg (closed form, no astropy)."""
    xi, eta = -np.asarray(x) * ASEC * np.pi / 180, np.asarray(y) * ASEC * np.pi / 180
    a0, d0 = np.deg2rad(ra0), np.deg2rad(dec0)
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(np.cos(c) * np.sin(d0) + np.where(rho > 0, eta * np.sin(c) * np.cos(d0) / rho, 0.0))
    ra = a0 + np.arctan2(xi * np.sin(c), rho * np.cos(d0) * np.cos(c) - eta * np.sin(d0) * np.sin(c))
    return np.rad2deg(ra) % 360.0, np.rad2deg(dec)


def sep_mas(ra1, dec1, ra2, dec2):
    d1, d2 = np.deg2rad(dec1), np.deg2rad(dec2)
    dra = np.deg2rad(((ra1 - ra2) + 180) % 360 - 180)
    return np.rad2deg(np.hypot(dra * np.cos(0.5 * (d1 + d2)), d1 - d2)) * 3.6e6


# --------------------------------------------------------------------------------------------- (a) round trip
def independent_pixels(rec):
    """Re-read the source file with plain astropy (+ hst_io for HST) and return dict(row, col -> value arrays) of the pixels that
    the default policy keeps, as (row, col, data, ivar, dq) arrays over the WHOLE image, plus a function (row, col) -> (ra, dec)."""
    path = rec["path"]
    m = rec["mission"]
    if m == "jwst":
        with fits.open(path, memmap=False) as h:
            sci, err, dq = h["SCI"].data.astype(np.float32), h["ERR"].data.astype(np.float32), h["DQ"].data
            w = WCS(h["SCI"].header, fobj=h)
        good = np.isfinite(sci) & np.isfinite(err) & (err > 0) & (dq == 0)
        yy, xx = np.nonzero(good)
        return yy, xx, sci[good], (1.0 / (err[good] * err[good])).astype(np.float32), dq[good].astype(np.uint32).view(np.int32), \
            lambda r, c: w.all_pix2world(c.astype(float), r.astype(float), 0)
    if m == "hst":
        import hst_io
        ext = int(json.loads(rec["extra_json"])["ext"])
        with fits.open(path, memmap=False) as h:
            h0, sh = h[0].header, h[("SCI", ext)].header
            inst = hst_io._identify(h0)
            w = hst_io.hst_wcs(h, ext)
            conv = hst_io.units_conversion(sh, h0, inst)
            good_bits = hst_io._good_bits(inst, float(h0["EXPSTART"]), "default")
            raw, err = h[("SCI", ext)].data.astype(np.float64), h[("ERR", ext)].data.astype(np.float64)
            dq = h[("DQ", ext)].data
        good = np.isfinite(raw) & np.isfinite(err) & (err > 0) & hst_io.dq_pass(dq, good_bits)
        yy, xx = np.nonzero(good)
        d = (raw[good] * conv["to_mjysr"]).astype(np.float32)
        e = (err[good] * conv["to_mjysr"]).astype(np.float32)
        return yy, xx, d, (1.0 / (e * e)).astype(np.float32), (dq[good].astype(np.int64) & 0xFFFF).astype(np.int32), \
            lambda r, c: w.all_pix2world(c.astype(float), r.astype(float), 0)
    if m == "spitzer":
        with fits.open(path, memmap=False) as h:
            sci, w = h[0].data.astype(np.float32), WCS(h[0].header)
        err = fits.getdata(path.replace("_cbcd", "_cbunc")).astype(np.float32)
        bim = fits.getdata(path.replace("_cbcd", "_bimsk")).astype(np.int32) & 0xFFFF
        good = np.isfinite(sci) & np.isfinite(err) & (err > 0) & ((bim & 32520) == 0)
        yy, xx = np.nonzero(good)
        return yy, xx, sci[good], (1.0 / (err[good] * err[good])).astype(np.float32), bim[good].astype(np.int32), \
            lambda r, c: w.all_pix2world(c.astype(float), r.astype(float), 0)
    if m == "spherex":
        with fits.open(path, memmap=False) as h:
            ih = h["IMAGE"].header
            sci, var, fl = h["IMAGE"].data.astype(np.float32), h["VARIANCE"].data.astype(np.float32), h["FLAGS"].data.astype(np.int32)
            w, wa = WCS(ih, fobj=h), WCS(_strip_sip(ih), fobj=h, key="A")
        mask = ps.SPHEREX_FATAL_MASK
        good = np.isfinite(sci) & np.isfinite(var) & (var > 0) & ((fl & mask) == 0)
        yy, xx = np.nonzero(good)
        dc, dr = wa.all_pix2world(xx.astype(float), yy.astype(float), 0)
        # return detector row/col as the "row/col" keys (what the store holds); pixel->world through the cutout coordinates
        lookup = {(int(np.rint(r)), int(np.rint(c))): (int(y), int(x)) for r, c, y, x in zip(dr, dc, yy, xx)}

        def radec(r, c):
            yx = np.array([lookup[(int(a), int(b))] for a, b in zip(r, c)])
            return w.all_pix2world(yx[:, 1].astype(float), yx[:, 0].astype(float), 0)
        return np.rint(dr).astype(int), np.rint(dc).astype(int), sci[good], (1.0 / var[good]).astype(np.float32), fl[good], radec
    if m == "galex":
        base = path.replace("-int.fits.gz", "")
        cps = fits.getdata(path).astype(np.float32)
        rr = fits.getdata(base + "-rrhr.fits.gz").astype(np.float32)
        sky = fits.getdata(base + "-skybg.fits.gz").astype(np.float32)
        fl = np.kron(fits.getdata(base + "-flags.fits.gz").astype(np.int32), np.ones((8, 8), np.int32))
        w = WCS(fits.getheader(path))
        K = json.loads(rec["extra_json"])["k_mjysr_per_cps_pix"]
        rate = np.maximum(cps, sky)
        with np.errstate(divide="ignore", invalid="ignore"):
            ivar = (1.0 / ((rate / rr) * (K ** 2))).astype(np.float32)
        good = np.isfinite(cps) & (rr > 0) & (sky > 0) & (fl == 0) & np.isfinite(ivar) & (ivar > 0)
        yy, xx = np.nonzero(good)
        return yy, xx, (cps[good].astype(np.float64) * K).astype(np.float32), ivar[good], fl[good], \
            lambda r, c: w.all_pix2world(c.astype(float), r.astype(float), 0)
    raise ValueError(m)


def test_a_round_trip(store, exposures, grid):
    """Reader rows (all tiles, in_core only) == independent re-read of the file: same pixel set, values, dq; positions == x0,y0."""
    print()
    for eid, rec in sorted(exposures.items(), key=lambda kv: kv[1]["name"]):
        t, _ = read_exposure_rows(store, eid)
        core = t.filter(t["in_core"])
        yy, xx, d, iv, dq, radec = independent_pixels(rec)
        # restrict the independent set to pixels with an in-grid position (the stored core set)
        ra, dec = radec(yy, xx)
        px, py = grid.world_to_plane(ra, dec)
        px, py = grid.quantize(px), grid.quantize(py)
        del ra, dec
        cx, cy = np.floor((px + grid.h) / grid.T), np.floor((py + grid.h) / grid.T)
        inc = (cx >= 0) & (cx < grid.n) & (cy >= 0) & (cy < grid.n)
        key_ref = yy[inc].astype(np.int64) * 100000 + xx[inc]
        order = np.argsort(key_ref)
        sk = key_ref[order]
        r_, c_ = core["row"].to_numpy().astype(np.int64), core["col"].to_numpy().astype(np.int64)
        assert len(r_) == len(key_ref), (rec["name"], len(r_), len(key_ref))      # same pixel count (one core row per pixel)
        kc = r_ * 100000 + c_
        pos = np.searchsorted(sk, kc)
        assert np.array_equal(sk[np.minimum(pos, len(sk) - 1)], kc), "pixel sets differ"
        assert len(np.unique(kc)) == len(kc)
        k = order[pos]
        assert np.array_equal(core["data"].to_numpy(), d[inc][k]), rec["name"]
        assert np.array_equal(core["ivar"].to_numpy(), iv[inc][k]), rec["name"]
        assert np.array_equal(core["dq"].to_numpy(), dq[inc][k]), rec["name"]
        # x0, y0 equal the independently recomputed plane position within the quantisation step
        step = 2.0 ** -grid.q if grid.q else 0.0
        assert np.abs(core["x0"].to_numpy() - px[inc][k]).max() <= step / 2 + 1e-9
        assert np.abs(core["y0"].to_numpy() - py[inc][k]).max() <= step / 2 + 1e-9
        print(f"(a) {rec['mission']:8s} {rec['name'][:48]:48s} core rows {len(r_):>9,d} identical to independent read (data, ivar, dq exact)")
    # the reader returns the same numbers as the fragments
    import torch  # noqa: F401
    e = ps.load_tile(store, 7, 7, device="cpu")
    for ex in e:
        t = ps.read_piece(next(p for p in ps.pieces_of(Path(store) / "rows" / "tile_ix=7" / "tile_iy=7") if p["eid"] == ex["exposure_id"]))
        assert np.array_equal(ex["d"].numpy(), t["data"].to_numpy())
        assert np.array_equal(ex["x0"].numpy(), t["x0"].to_numpy())
        assert np.array_equal(ex["row"].numpy(), t["row"].to_numpy().astype(np.int64))
        tx = (t["x0"].to_numpy() - grid.tile_extent(7, 7)[0]) / 0.04 - 0.5
        assert np.abs(ex["tx"].numpy() - tx).max() < 1e-2   # float32 at ~750 px
    print(f"(a) reader tile (7,7): {len(e)} exposures, fragment values/positions reproduced")


# --------------------------------------------------------------------------------------------- (b) halo / in_core
def brute_membership(grid, x, y, halo):
    out = []
    for ix in range(grid.n):
        for iy in range(grid.n):
            x0, x1, y0, y1 = grid.tile_extent(ix, iy)
            if x0 - halo <= x <= x1 + halo and y0 - halo <= y <= y1 + halo:
                out.append((ix, iy))
    return out


def test_b_halo_and_core(store, exposures, grid):
    print()
    rng = np.random.default_rng(1)
    for eid, rec in sorted(exposures.items(), key=lambda kv: kv[1]["name"]):
        t, files = read_exposure_rows(store, eid, ["row", "col", "x0", "y0", "tile_ix", "tile_iy", "in_core"])
        r, c = t["row"].to_numpy().astype(np.int64), t["col"].to_numpy().astype(np.int64)
        key = r * 100000 + c
        uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        core = t["in_core"].to_numpy()
        ncore = np.bincount(inv, weights=core.astype(float), minlength=len(uniq))
        x, y = t["x0"].to_numpy(), t["y0"].to_numpy()
        first = np.zeros(len(uniq), int)
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        ingrid = (x[first] >= -grid.h) & (x[first] < grid.h) & (y[first] >= -grid.h) & (y[first] < grid.h)
        assert np.all(ncore[ingrid] == 1), "pixel inside the box must be in_core in exactly one tile"
        assert np.all(ncore[~ingrid] == 0), "pixel outside the box must never be in_core"
        # in_core tile is the tile whose core holds the pixel
        cx, cy = np.floor((x + grid.h) / grid.T), np.floor((y + grid.h) / grid.T)
        tix, tiy = t["tile_ix"].to_numpy(), t["tile_iy"].to_numpy()
        assert np.all((cx[core] == tix[core]) & (cy[core] == tiy[core]))
        # copies per pixel against a brute-force loop over all tiles, on a random subset of pixels
        halo = grid.halo_for(ps.family_of(rec["instrument"]))
        pick = rng.choice(len(uniq), size=min(300, len(uniq)), replace=False)
        for u in pick:
            i0 = first[u]
            bm = sorted(brute_membership(grid, x[i0], y[i0], halo))
            have = sorted(zip(tix[inv == u].tolist(), tiy[inv == u].tolist()))
            assert have == bm, (rec["name"], u, have, bm)
        ncopies = np.bincount(cnt)
        print(f"(b) {rec['instrument']:8s} halo {halo:4.1f}\" {rec['name'][:40]:40s} px {len(uniq):>9,d} rows {len(r):>10,d} dup x{len(r)/len(uniq):4.2f} "
              f"copies hist {dict((i, int(n)) for i, n in enumerate(ncopies) if n)} in_core==1 for {int((ncore[ingrid]==1).sum()):,d} box px")


# --------------------------------------------------------------------------------------------- (c) positions
def test_c_positions(store, exposures, grid):
    print()
    rng = np.random.default_rng(2)
    for eid, rec in sorted(exposures.items(), key=lambda kv: kv[1]["name"]):
        t, _ = read_exposure_rows(store, eid, ["row", "col", "x0", "y0", "in_core"])
        t = t.filter(t["in_core"])
        n = len(t)
        pick = np.sort(rng.choice(n, size=min(40000, n), replace=False))
        r, c = t["row"].to_numpy()[pick].astype(int), t["col"].to_numpy()[pick].astype(int)
        x, y = t["x0"].to_numpy()[pick], t["y0"].to_numpy()[pick]
        ra_s, dec_s = inv_gnomonic(x, y, ps.CENTER_RA, ps.CENTER_DEC)          # stored position -> sky (independent closed form)
        m = rec["mission"]
        if m == "hst":
            import hst_io
            ext = int(json.loads(rec["extra_json"])["ext"])
            with fits.open(rec["path"], memmap=False) as h:
                w = hst_io.hst_wcs(h, ext)
                ra_a, dec_a = w.all_pix2world(c.astype(float), r.astype(float), 0)
            src = "hst_io.hst_wcs (a priori, SIP+D2IM+CPDIS)"
        elif m == "spherex":
            with fits.open(rec["path"], memmap=False) as h:
                ih = h["IMAGE"].header
                w = WCS(ih, fobj=h)
                # detector (0-based active) coordinates = cutout pixel (0-based) + 1 - CRPIXA   (CDELT = 1, CRVAL = 0, PC = identity)
                xc, yc = c - (1.0 - ih["CRPIX1A"]), r - (1.0 - ih["CRPIX2A"])
                ra_a, dec_a = w.all_pix2world(xc, yc, 0)
            src = "IMAGE WCS, cutout pixel = detector - (1 - CRPIXA)"
        else:
            hdr = json.loads(rec["header_json"])                                 # rebuild from the stored JSON header subset
            w = WCS(fits.Header({k: v for k, v in hdr.items()}))
            ra_a, dec_a = w.all_pix2world(c.astype(float), r.astype(float), 0)
            src = "WCS rebuilt from header_json"
        s = sep_mas(ra_s, dec_s, ra_a, dec_a)
        print(f"(c) {rec['instrument']:8s} {rec['name'][:40]:40s} {len(pick):>6d} px  max {s.max():.5f} mas  median {np.median(s):.5f} mas   [{src}]")
        assert s.max() < 1.0, (rec["name"], s.max())


# --------------------------------------------------------------------------------------------- (d) units
def test_d_units(store, exposures):
    print("\n(d) units sanity: median of `data` over kept rows, MJy/sr per nominal pixel (surface brightness)")
    print(f"    {'mission':8s} {'band':14s} {'name':40s} {'median':>11s} {'p16':>10s} {'p84':>10s} {'mean':>10s}  median err (1/sqrt ivar)   kept/region")
    bands = {}
    for eid, rec in sorted(exposures.items(), key=lambda kv: (kv[1]["mission"], kv[1]["band"])):
        t, _ = read_exposure_rows(store, eid, ["data", "ivar", "in_core"])
        t = t.filter(t["in_core"])
        d, iv = t["data"].to_numpy(), t["ivar"].to_numpy()
        assert np.all(np.isfinite(d)) and np.all(iv > 0) and np.all(np.isfinite(iv))
        med = float(np.median(d))
        err = float(np.median(1 / np.sqrt(iv)))
        print(f"    {rec['mission']:8s} {rec['band']:14s} {rec['name'][:40]:40s} {med:11.5g} {np.percentile(d, 16):10.4g} {np.percentile(d, 84):10.4g} {d.mean():10.4g}  {err:10.4g}"
              f"               {rec['n_pix_kept']/rec['n_pix_region']:.3f}")
        bands[eid] = med
        # plausibility windows (MJy/sr); SMACS0723 is a high-ecliptic-latitude field (zodi-dominated at 1-5 um)
        if rec["mission"] == "jwst" and rec["instrument"] == "NIRCam":
            assert 0.02 < med < 1.0
        if rec["mission"] == "hst":
            assert -0.5 < med < 2.0
        if rec["mission"] == "galex":
            assert 1e-4 < float(d.mean()) < 0.05          # most 96 s pixels hold zero counts: use the mean
        if rec["mission"] == "spherex":
            assert 0.01 < med < 3.0
        if rec["mission"] == "spitzer":
            assert abs(med) < 20.0
    for eid, rec in exposures.items():
        if rec["mission"] == "galex":
            print(f"    GALEX {rec['band']}: {rec['unit_conversion']}")
        if rec["mission"] == "hst":
            print(f"    HST {rec['name']}: {rec['unit_conversion']}")


# --------------------------------------------------------------------------------------------- (e) SPHEREx lambda
def test_e_spherex_lambda(store, exposures):
    print()
    n = 0
    for eid, rec in exposures.items():
        t, _ = read_exposure_rows(store, eid, ["lambda_um"])
        lam = t["lambda_um"].to_numpy(zero_copy_only=False)
        if rec["mission"] != "spherex":
            assert t["lambda_um"].null_count == len(t)           # null elsewhere
            continue
        det = int(rec["detector"][-1])
        lo, hi = ps.SPHEREX_RANGE_UM[det]
        assert t["lambda_um"].null_count == 0
        assert lam.min() >= lo - 1e-4 and lam.max() <= hi + 1e-4, (lam.min(), lam.max(), lo, hi)
        assert hi - lo > lam.max() - lam.min() > 0
        print(f"(e) {rec['detector']} {rec['name'][:50]}: lambda {lam.min():.4f}-{lam.max():.4f} um within detector range {lo}-{hi} um")
        n += 1
    assert n >= 2


# --------------------------------------------------------------------------------------------- (f) v31 structures
V31_KEYS = {"name", "band", "tx", "ty", "d", "ivar", "row", "col", "dx", "dy", "bkg0", "path", "header", "cut", "noise_scale", "detname"}
HST_KEYS = {"instrument", "filter", "mjd", "exptime", "pixar_sr_nominal", "units_note", "dq", "photplam", "photflam", "photfnu_jy",
            "to_mjysr", "svo_id", "ext", "dq_good_bits"}


def test_f_reader_feeds_v31(store, exposures, grid, tmp_path):
    import torch
    import smacs_four_v31 as v31
    import hst_io
    ix, iy = 7, 7
    exps = ps.load_tile(store, ix, iy, missions=["jwst", "hst"], device="cpu")
    nir = [e for e in exps if e["band"] in v31.NIRCAM_BANDS]
    hst = [e for e in exps if e.get("instrument") in hst_io.DQ_GOOD_BITS]
    print(f"\n(f) tile ({ix},{iy}): {len(exps)} exposures ({len(nir)} NIRCam, {len(hst)} HST chips)")
    assert nir and hst
    for e in exps:
        assert V31_KEYS <= set(e), V31_KEYS - set(e)
        n = e["d"].numel()
        for k in ("tx", "ty", "ivar", "dx", "dy", "row", "col"):
            assert e[k].shape == (n,), k
        assert e["d"].dtype == e["tx"].dtype == e["ivar"].dtype == torch.float32 and e["row"].dtype == torch.int64
        assert torch.isfinite(e["d"]).all() and (e["ivar"] > 0).all()
        assert e["dx"].min() >= -1 and e["dx"].max() <= 1
    for e in hst:
        assert HST_KEYS <= set(e), HST_KEYS - set(e)
    # v31 machinery accepts them: registry, target header, splat coadd, holdout selection
    reg = v31.registry_from_exposures(nir + hst)
    thead = ps.tile_header(store, ix, iy, 0.04)
    img, den = v31.splat_coadd(hst, thead, "cpu")
    assert img.shape == (thead["NAXIS2"], thead["NAXIS1"]) and torch.isfinite(img).any()
    v31.set_registry(reg)
    sel = v31.select_exposures(nir + hst, n_test=1)
    print(f"    registry bands {reg.bands}; splat_coadd of the HST chips -> {tuple(img.shape)}, "
          f"{int(torch.isfinite(img).sum())} filled px; select_exposures holdout {sel['holdout']}")
    # compare against the legacy loaders on the same files (values exact, positions within 0.01 px)
    jw = next(e for e in nir if "nrcb3" in e["name"])
    d = tmp_path / "jw"
    d.mkdir()
    os.symlink(jw["path"], d / os.path.basename(jw["path"]))
    try:
        ref = v31.load_exposures(str(d), thead, "cpu", pad=24)
    except ImportError as exc:        # torchsky.wcs -> torchfits built for another torch than the one installed (environment issue)
        ref = None
        print(f"    SKIPPED comparison with smacs_four_v31.load_exposures: {str(exc).splitlines()[0][:110]}")
    if ref is not None:
        assert len(ref) == 1
        ref = ref[0]
        mine = {(int(r), int(c)): k for k, (r, c) in enumerate(zip(jw["row"].numpy(), jw["col"].numpy()))}
        kk = np.array([mine.get((int(r), int(c)), -1) for r, c in zip(ref["row"].numpy(), ref["col"].numpy())])
        common = kk >= 0
        assert common.mean() > 0.99, common.mean()                   # v31's crop (target grid + 24 px) lies inside the stored tile+halo
        assert np.array_equal(ref["d"].numpy()[common], jw["d"].numpy()[kk[common]])
        assert np.array_equal(ref["ivar"].numpy()[common], jw["ivar"].numpy()[kk[common]])
        dtx = np.abs(ref["tx"].numpy()[common] - jw["tx"].numpy()[kk[common]]).max()
        dty = np.abs(ref["ty"].numpy()[common] - jw["ty"].numpy()[kk[common]]).max()
        print(f"    vs smacs_four_v31.load_exposures ({os.path.basename(jw['path'])}): {int(common.sum()):,d} common px, d and ivar identical, "
              f"max |dtx|,|dty| = {dtx:.4f}, {dty:.4f} px (tile TAN header with plane-Jacobian CD vs global plane)")
        assert dtx < 0.01 and dty < 0.01
    ch = next(e for e in hst if e["instrument"] == "WFC3/IR")
    href = hst_io.load_hst_exposures([ch["path"]], thead, "cpu", pad=24)
    assert len(href) == 1
    href = href[0]
    mine = {(int(r), int(c)): k for k, (r, c) in enumerate(zip(ch["row"].numpy(), ch["col"].numpy()))}
    kk = np.array([mine.get((int(r), int(c)), -1) for r, c in zip(href["row"].numpy(), href["col"].numpy())])
    common = kk >= 0
    assert common.mean() > 0.99
    assert np.array_equal(href["d"].numpy()[common], ch["d"].numpy()[kk[common]])
    assert np.array_equal(href["ivar"].numpy()[common], ch["ivar"].numpy()[kk[common]])
    assert np.array_equal(href["dq"].numpy()[common], ch["dq"].numpy()[kk[common]])
    dtx = np.abs(href["tx"].numpy()[common] - ch["tx"].numpy()[kk[common]]).max()
    print(f"    vs hst_io.load_hst_exposures ({os.path.basename(ch['path'])}): {int(common.sum()):,d} common px, d, ivar, dq identical, max |dtx| = {dtx:.4f} px")
    assert dtx < 0.01


# --------------------------------------------------------------------------------------------- (g) size estimate
def _poly(s):
    nums = [float(v) for v in s.replace("POLYGON ICRS", "").replace("POLYGON", "").split()]
    return list(zip(nums[0::2], nums[1::2]))


def _cls(rec):
    if rec["mission"] == "jwst":
        if rec["instrument"] == "NIRCam":
            return "NIRCam-SW" if rec["detector"] in ("NRCA1", "NRCA2", "NRCA3", "NRCA4", "NRCB1", "NRCB2", "NRCB3", "NRCB4") else "NIRCam-LW"
        return rec["instrument"]
    return {"spitzer": "IRAC", "spherex": "SPHEREx", "galex": "GALEX"}.get(rec["mission"], rec["instrument"])


def test_g_size_estimate(store, exposures, grid):
    """Bytes per row from the sample fragments x expected rows for the full union. Expected rows per exposure =
    sum over tiles of footprint-polygon area inside (tile core + halo) / pixel area x kept fraction (measured on samples of the
    same instrument/detector class); HST from the number of overlapping chips x rows per chip measured on the samples."""
    import shapely
    from shapely.geometry import Polygon, box as sbox
    dat = ROOT / "datasearch"
    # ---- measured per class on the samples
    meas = {}
    for eid, rec in exposures.items():
        cls = {"jwst": lambda r: ("NIRCam-SW" if r["detector"] in ("NRCA1", "NRCA2", "NRCA3", "NRCA4", "NRCB1", "NRCB2", "NRCB3", "NRCB4")
                                   else "NIRCam-LW") if r["instrument"] == "NIRCam" else r["instrument"],
               "hst": lambda r: r["instrument"], "spitzer": lambda r: "IRAC", "spherex": lambda r: "SPHEREx",
               "galex": lambda r: "GALEX"}[rec["mission"]](rec)
        read_exposure_rows(store, eid, ["exposure_id"])            # fills the piece cache
        nbytes = sum(p["bytes"] for p in _PIECES[str(store)] if p["eid"] == eid)
        m = meas.setdefault(cls, {"rows": 0, "bytes": 0, "kept": [], "pixsr": [], "t": [], "n": 0, "rows_list": []})
        m["rows"] += rec["n_rows"]
        m["bytes"] += nbytes
        m["kept"].append(rec["n_pix_kept"] / rec["n_pix_region"])
        m["pixsr"].append(rec["pixar_sr_nominal"])
        m["t"].append(rec["t_extract_s"])
        m["n"] += 1
        m["rows_list"].append(rec["n_rows"])
    print("\n(g) measured on the samples:")
    for k, m in sorted(meas.items()):
        print(f"    {k:10s} n={m['n']} rows/exposure(sample) {np.mean(m['rows_list']):>12,.0f}  bytes/row {m['bytes']/m['rows']:6.2f}  "
              f"kept(mask) {np.mean(m['kept']):.3f}  pixel {np.sqrt(np.mean(m['pixsr'])/ps.ARCSEC2_TO_SR):.4f}\"  t/exposure {np.mean(m['t']):.2f}s")
    T = grid.T
    tiles = {fam: [sbox(*grid.tile_extent(i, j)[0:1], *grid.tile_extent(i, j)[2:3], grid.tile_extent(i, j)[1], grid.tile_extent(i, j)[3]).buffer(h, join_style=2)
                   for i in range(grid.n) for j in range(grid.n)] for fam, h in grid.halos.items()}

    def poly_plane(pts):
        ra, dec = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        x, y = grid.world_to_plane(ra, dec)
        return Polygon(np.c_[x, y]).buffer(0)

    def rows_in(poly, fam):
        return float(sum(poly.intersection(t).area for t in tiles[fam]))   # arcsec^2 summed over tiles = pixels x pixel area

    result = {}
    # JWST (manifest_jwst_cal_union.csv, s_region)
    jw = list(csv.DictReader(open(dat / "manifest_jwst_cal_union.csv")))
    est = {"NIRCam-SW": 0.0, "NIRCam-LW": 0.0, "NIRISS": 0.0, "MIRI": 0.0}
    cnt = dict.fromkeys(est, 0)
    for r in jw:
        fam = r["instrument"]                                            # NIRCAM | NIRISS | MIRI (halo family)
        if r["instrument"] == "NIRCAM":
            cls = "NIRCam-LW" if "LONG" in r["detector"].upper() else "NIRCam-SW"
        else:
            cls = r["instrument"]
        poly = poly_plane(_poly(r["s_region"]))
        area = 0.0 if poly.is_empty else rows_in(poly, fam)
        if area <= 0 or cls not in meas:
            continue
        pix = float(np.mean(meas[cls]["pixsr"])) / ps.ARCSEC2_TO_SR
        est[cls] += area / pix * float(np.mean(meas[cls]["kept"]))
        cnt[cls] += 1
    for k in est:
        result[k] = (cnt[k], est[k])
    # Spitzer
    sp = [r for r in csv.DictReader(open(dat / "manifest_spitzer_union.csv")) if r["file_type"] == "cbcd"]
    e_sp, n_sp = 0.0, 0
    pix = float(np.mean(meas["IRAC"]["pixsr"])) / ps.ARCSEC2_TO_SR
    for r in sp:
        poly = poly_plane(_poly(r["footprint_region"]))
        a = rows_in(poly, "IRAC") if not poly.is_empty else 0.0
        if a > 0:
            e_sp += a / pix * float(np.mean(meas["IRAC"]["kept"]))
            n_sp += 1
    result["IRAC"] = (n_sp, e_sp)
    # SPHEREx: cutouts are 10' squares around the union centre intersected with the detector footprint
    cut = sbox(-300, -300, 300, 300)
    spx = list(csv.DictReader(open(dat / "manifest_spherex_union.csv")))
    e_spx, n_spx = 0.0, 0
    pix = float(np.mean(meas["SPHEREx"]["pixsr"])) / ps.ARCSEC2_TO_SR
    for r in spx:
        poly = poly_plane(_poly(r["footprint_region"])).intersection(cut)
        a = rows_in(poly, "SPHEREX") if not poly.is_empty else 0.0
        if a > 0:
            e_spx += a / pix * float(np.mean(meas["SPHEREx"]["kept"]))
            n_spx += 1
    result["SPHEREx"] = (n_spx, e_spx)
    # GALEX: int maps cover the whole box (box8_fully_inside_image)
    gx = [r for r in csv.DictReader(open(dat / "manifest_galex_union.csv")) if r["product"] == "int"]
    n_gx = len(gx)
    result["GALEX"] = (n_gx, n_gx * float(np.mean(meas["GALEX"]["rows_list"])))
    # HST: chips overlapping x measured rows per chip (instrument class)
    hm = list(csv.DictReader(open(ROOT / "hst" / "hst_imaging_unique.csv")))
    best = {}
    for r in hm:
        if r["hap"] == "True" or f"{r['instrument']}/{r['detector']}" not in ("WFC3/IR", "ACS/WFC", "WFC3/UVIS"):
            continue
        root = r["root"] if r.get("root") else r["obs_id"]
        if root not in best or r["product_type"].upper() == "FLC":
            best[root] = r
    h_est = {"WFC3/IR": [0, 0.0], "ACS/WFC": [0, 0.0], "WFC3/UVIS": [0, 0.0]}
    for r in best.values():
        inst = f"{r['instrument']}/{r['detector']}"
        if inst not in meas:
            continue
        nch = 1 if inst == "WFC3/IR" else 2
        per_chip = float(np.mean(meas[inst]["rows_list"]))               # each sample record is one chip
        h_est[inst][0] += 1
        h_est[inst][1] += per_chip * nch
    for k, v in h_est.items():
        result[k] = (v[0], v[1])
    # ---- estimator check: footprint-based prediction vs the actual rows of the sample files themselves
    print("\n    estimator check (manifest footprint -> predicted rows, vs actual rows of the sample file):")
    sample_rows = {rec["name"]: rec for rec in exposures.values()}
    for r in jw:
        stem = r["filename"]
        if stem in sample_rows:
            rec = sample_rows[stem]
            cls = ("NIRCam-LW" if "LONG" in r["detector"].upper() else "NIRCam-SW") if r["instrument"] == "NIRCAM" else r["instrument"]
            poly = poly_plane(_poly(r["s_region"]))
            pred = rows_in(poly, r["instrument"]) / (rec["pixar_sr_nominal"] / ps.ARCSEC2_TO_SR) * (rec["n_pix_kept"] / rec["n_pix_region"])
            print(f"      {stem[:48]:48s} predicted {pred:>12,.0f}  actual {rec['n_rows']:>12,d}  ratio {pred / rec['n_rows']:.3f}")
    for r in sp:
        stem = r["filename"].replace("_cbcd.fits", "")
        if stem in sample_rows:
            rec = sample_rows[stem]
            pred = rows_in(poly_plane(_poly(r["footprint_region"])), "IRAC") / (rec["pixar_sr_nominal"] / ps.ARCSEC2_TO_SR) * (rec["n_pix_kept"] / rec["n_pix_region"])
            print(f"      {stem[:48]:48s} predicted {pred:>12,.0f}  actual {rec['n_rows']:>12,d}  ratio {pred / rec['n_rows']:.3f}")
    for r in spx:
        stem = r["filename"].replace(".fits", "")
        if stem in sample_rows:
            rec = sample_rows[stem]
            pred = rows_in(poly_plane(_poly(r["footprint_region"])).intersection(cut), "SPHEREX") / (rec["pixar_sr_nominal"] / ps.ARCSEC2_TO_SR) * (rec["n_pix_kept"] / rec["n_pix_region"])
            print(f"      {stem[:48]:48s} predicted {pred:>12,.0f}  actual {rec['n_rows']:>12,d}  ratio {pred / rec['n_rows']:.3f}")
    # ---- fragments (files) per class: exposures x mean tiles per exposure on the samples
    tiles_per = {k: float(np.mean([r["n_tiles"] for r in exposures.values() if _cls(r) == k])) for k in meas}
    # ---- totals
    bpr = {"NIRCam-SW": "NIRCam-SW", "NIRCam-LW": "NIRCam-LW", "NIRISS": "NIRISS", "MIRI": "MIRI", "IRAC": "IRAC", "SPHEREx": "SPHEREx",
           "GALEX": "GALEX", "WFC3/IR": "WFC3/IR", "ACS/WFC": "ACS/WFC", "WFC3/UVIS": "WFC3/UVIS"}
    print("\n    expected union (footprints from the manifests):")
    print(f"    {'class':10s} {'#exposures':>10s} {'rows':>16s} {'B/row':>7s} {'GB':>8s} {'fragments':>10s}")
    tot_rows = tot_b = tot_frag = 0.0
    for k, (n, rows) in result.items():
        b = meas.get(bpr[k], meas.get("NIRCam-SW"))
        bytes_per_row = b["bytes"] / b["rows"]
        gb = rows * bytes_per_row / 1e9
        tot_rows += rows
        tot_b += rows * bytes_per_row
        nfrag = n * tiles_per.get(bpr[k], 0.0)
        tot_frag += nfrag
        print(f"    {k:10s} {n:>10d} {rows:>16,.0f} {bytes_per_row:7.2f} {gb:8.1f} {nfrag:>10,.0f}")
    print(f"    TOTAL rows {tot_rows:,.0f}  ~ {tot_b/1e9:.0f} GB parquet (zstd), ~{tot_rows*37/1e9:.0f} GB uncompressed in-memory at 37 B/row; "
          f"~{tot_frag:,.0f} fragment files")
    ts = sum(np.sum(m["t"]) / m["n"] * result.get(k, (0, 0))[0] for k, m in meas.items() if k in result)
    print(f"    extraction CPU time ~ {ts/3600:.1f} core-hours for {sum(v[0] for v in result.values())} exposures (sample mean time per class)")
    assert tot_rows > 1e8


# --------------------------------------------------------------------------------------------- CLI behaviour
def test_h_cli_idempotence_shards_manifest(tmp_path):
    """Idempotent re-run, missing-fragment repair, shards, manifest discovery, mask-policy none, summary."""
    st = tmp_path / "s"
    sp = sorted(glob.glob(str(SAMPLES / "spitzer" / "*_cbcd.fits")))
    gx = sorted(glob.glob(str(SAMPLES / "galex" / "*" / "*-int.fits.gz")))
    base = ["extract", "--store", str(st), "--no-finalize"]
    ps.main(base + ["--mission", "spitzer", "--inputs", str(SAMPLES / "spitzer" / "*_cbcd.fits")])
    frs = sorted(st.glob("rows/*/*/exp_*.parquet"))
    mt = {f: f.stat().st_mtime_ns for f in frs}
    ps.main(base + ["--mission", "spitzer", "--inputs", str(SAMPLES / "spitzer" / "*_cbcd.fits")])           # all skipped by the markers
    assert {f: f.stat().st_mtime_ns for f in sorted(st.glob("rows/*/*/exp_*.parquet"))} == mt
    # repair: drop one fragment and one marker -> only that fragment is rewritten, the others keep their mtime
    victim = frs[3]
    victim.unlink()
    for m in (st / "done").glob("*.json"):
        m.unlink()
    ps.main(base + ["--mission", "spitzer", "--inputs", str(SAMPLES / "spitzer" / "*_cbcd.fits")])
    after = {f: f.stat().st_mtime_ns for f in sorted(st.glob("rows/*/*/exp_*.parquet"))}
    assert victim in after and all(after[f] == mt[f] for f in frs if f != victim)
    # shards partition the inputs
    files = ps.discover("spitzer", [str(SAMPLES / "spitzer" / "*_cbcd.fits")], [])
    assert files[0::2] + files[1::2] == files and len(files) == 2
    st2 = tmp_path / "s2"
    for k in range(2):
        ps.main(["extract", "--store", str(st2), "--mission", "spitzer", "--inputs", str(SAMPLES / "spitzer" / "*_cbcd.fits"), "--shard", f"{k}/2"])
    assert len(list((st2 / "exposure_meta").glob("*.json"))) == 2
    # manifest discovery (JWST manifest + data dir; HST manifest)
    found = ps.discover("jwst", [str(ROOT / "datasearch" / "manifest_jwst_cal_union.csv")], [str(SAMPLES / "jwst")])
    assert any("nrcb3" in f for f in found) and len(found) >= 4, found
    hfound = ps.discover("hst", [str(ROOT / "hst" / "hst_imaging_unique.csv")], [str(SAMPLES / "hst")])
    assert any("iczgzbmaq" in f for f in hfound), hfound
    # mask-policy none keeps masked pixels with ivar = 0
    st3 = tmp_path / "s3"
    ps.main(["extract", "--store", str(st3), "--mission", "spitzer", "--inputs", sp[0], "--mask-policy", "none"])
    ex = ps.read_exposures(st3)[0]
    t = pa.concat_tables([pq.read_table(f) for f in st3.glob("rows/*/*/exp_*.parquet")])
    t = t.filter(t["in_core"])
    dflt = [e for e in ps.read_exposures(st) if e["name"] == ex["name"]][0]
    td = pa.concat_tables([pq.read_table(f) for f in st.glob(f"rows/*/*/exp_{dflt['exposure_id']}.parquet")])
    ncore_default = int(td["in_core"].to_numpy().sum())
    assert len(t) >= ncore_default and int((t["ivar"].to_numpy() == 0).sum()) == len(t) - ncore_default
    assert int((td["ivar"].to_numpy() > 0).all()) == 1
    ps.main(["summary", "--store", str(st2)])


# --------------------------------------------------------------------------------------------- part-file layout
def test_i_part_layout_equals_fragment_layout(store):
    """The sample store is in the part layout; the reader gives the same numbers as from the fragment store."""
    parts = list(Path(store).glob("rows/*/*/part-*.parquet"))
    assert parts and not list(Path(store).glob("rows/*/*/exp_*.parquet"))
    assert len({(p.parent, p.name.split("-")[1]) for p in parts}) == len(parts)    # at most one part file per tile and mission (one job each here)
    if not FRAG_STORE.exists():
        pytest.skip("fragment store not present")
    n = 0
    for ix, iy in [(7, 7), (3, 12)]:
        a = {e["name"]: e for e in ps.load_tile(store, ix, iy)}
        b = {e["name"]: e for e in ps.load_tile(FRAG_STORE, ix, iy)}
        for name in set(a) & set(b):
            for k in ("d", "ivar", "tx", "ty", "row", "col", "dq", "in_core"):
                assert np.array_equal(a[name][k].numpy(), b[name][k].numpy()), (name, k)
            n += 1
    assert n >= 6
    print(f"\n(i) part layout == fragment layout for {n} exposure-tiles; {len(parts)} part files hold the whole sample store")


def test_j_compact_job_semantics(tmp_path, capsys):
    st, loc = tmp_path / "arc", tmp_path / "scratch"
    inp = str(SAMPLES / "spitzer" / "*_cbcd.fits")
    base = ["extract", "--store", str(st), "--local-dir", str(loc), "--mission", "spitzer", "--inputs", inp]
    for k in range(2):
        ps.main(base + ["--shard", f"{k}/2"])
    parts = sorted(st.glob("rows/*/*/part-*.parquet"))
    names = {p.name for p in parts}
    assert names == {"part-spitzer-0of2.parquet", "part-spitzer-1of2.parquet"}
    for p in parts:                                                      # one row group per exposure, one exposure per group here
        md = pq.ParquetFile(p).metadata
        assert md.num_row_groups == 1
    assert (st / "done" / "job-spitzer-0of2.json").exists() and (st / "done" / "job-spitzer-1of2.json").exists()
    assert len(list((st / "exposure_meta").glob("exp_*.json"))) == 2
    assert not list(loc.glob("*")), "scratch must be cleaned"
    assert not list(st.glob("rows/*/*/exp_*.parquet")), "no fragments on the arc store"
    # job-level idempotence: marker present -> skipped; marker removed -> part files rewritten
    mt = {p: p.stat().st_mtime_ns for p in parts}
    ps.main(base + ["--shard", "0/2"])
    assert {p: p.stat().st_mtime_ns for p in parts} == mt
    (st / "done" / "job-spitzer-0of2.json").unlink()
    ps.main(base + ["--shard", "0/2"])
    after = {p: p.stat().st_mtime_ns for p in parts}
    for p in parts:
        assert (after[p] != mt[p]) == (p.name == "part-spitzer-0of2.parquet"), p
    assert (st / "done" / "job-spitzer-0of2.json").exists()
    # a part file with several exposures: one row group each (a single job over both files)
    st2 = tmp_path / "arc2"
    ps.main(["extract", "--store", str(st2), "--local-dir", str(loc), "--mission", "spitzer", "--inputs", inp])
    multi = [p for p in st2.glob("rows/*/*/part-*.parquet") if pq.ParquetFile(p).metadata.num_row_groups == 2]
    assert multi, "tiles covered by both frames must hold two row groups"
    # finalize / summary / reader on the part layout
    ps.main(["finalize", "--store", str(st)])
    assert len(ps.read_exposures(st)) == 2
    ps.main(["summary", "--store", str(st2)])
    out = capsys.readouterr().out
    assert "2 exposures" in out and "IRAC" in out
    ix, iy = [int(x.split("=")[1]) for x in multi[0].parent.parts[-2:]]
    ps.main(["finalize", "--store", str(st2)])
    es = ps.load_tile(st2, ix, iy)
    assert len(es) == 2 and all(e["d"].numel() > 0 for e in es)
    print(f"\n(j) compaction: {len(parts)} part files for 2 shard-jobs, {len(multi)} tiles with 2 row groups, job markers/meta/scratch cleanup/idempotence OK")


def test_k_hst_skips_non_science(tmp_path, capsys):
    import shutil
    src = SAMPLES / "hst" / "iczgzbmaq_flt.fits"
    cases = {"dark": ("TARGNAME", "DARK", "calibration frame"), "counts": ("BUNIT", "COUNTS", "BUNIT=COUNTS")}
    for tag, (key, val, why) in cases.items():
        f = tmp_path / f"{tag}_flt.fits"
        shutil.copy(src, f)
        with fits.open(f, mode="update") as h:
            (h[0].header if key == "TARGNAME" else h[("SCI", 1)].header)[key] = val
        st = tmp_path / f"s_{tag}"
        ps.main(["extract", "--store", str(st), "--mission", "hst", "--inputs", str(f)])
        assert not list((st / "exposure_meta").glob("*.json"))
        assert why in capsys.readouterr().out
    f = tmp_path / "nophot_flt.fits"
    shutil.copy(src, f)
    with fits.open(f, mode="update") as h:
        for k in ("PHOTFNU", "PHOTFLAM"):
            for hh in (h[0].header, h[("SCI", 1)].header):
                if k in hh:
                    hh[k] = 0.0
    ps.main(["extract", "--store", str(tmp_path / "s_ph"), "--mission", "hst", "--inputs", str(f)])
    assert "PHOTFNU=0" in capsys.readouterr().out
    print("\n(k) HST dark / COUNTS / PHOTFNU=0 files skipped with a logged reason")


if __name__ == "__main__":
    sys.exit(pytest.main(["-s", "-q", __file__] + sys.argv[1:]))
