#!/usr/bin/env python3
"""Tiled pixel store of all space exposures around SMACS J0723 (step U0 of union-fit-design.md).

One stored row = one detector pixel: its value, inverse variance, native quality word, detector row/col and its *starting* sky
position (header WCS, all distortion terms) on one global tangent plane. No resampling, no registration.

Layout
    <store>/store.json                      grid and halo configuration (checked by every extract run)
    <store>/exposures.parquet, exposures.json   one record per detector image (built by `finalize`)
    <store>/exposure_meta/exp_<id>.json     per-exposure record written by `extract` (merged by `finalize`)
    <store>/done/<file-id>.json             per-input-file marker (idempotent resume)
    <store>/rows/tile_ix=<i>/tile_iy=<j>/exp_<id>.parquet     one fragment per exposure per tile

CLI
    python3 pixelstore.py extract  --store DIR --inputs GLOB_OR_MANIFEST [--inputs ...] --mission jwst|hst|spitzer|spherex|galex
                                   [--shard k/N] [--data-dir DIR ...] [--tile-arcsec 30] [--force] ...
    python3 pixelstore.py finalize --store DIR      (merge exposure_meta -> exposures.parquet/json; run after all shards)
    python3 pixelstore.py summary  --store DIR      (rows per tile per instrument, exposures per tile, bytes)

Global plane: gnomonic (TAN) projection about (RA, Dec) = (110.8020, -73.4586); x0 = arcsec WEST of the centre (RA decreasing,
the same sense as CD1_1 < 0 of smacs_four_v31.target_header), y0 = arcsec NORTH. The union box is x0, y0 in [-B/2, B/2), B = 480".
Tile (ix, iy) has core x0 in [ix*T - B/2, (ix+1)*T - B/2), same for y0; a pixel belongs to every tile whose core expanded by the
instrument halo (square, Chebyshev) contains it; in_core is True in exactly the tile whose core contains it (False everywhere for
pixels outside the grid, which only occur as halo of edge tiles).

Reader: load_tile(store, ix, iy, missions=None, bands=None, device="cpu", scale_arcsec=0.04) -> list of dicts shaped like the
exposure dicts of smacs_four_v31.load_exposures / hst_io.load_hst_exposures (see its docstring).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy import log as _astropy_log
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

warnings.filterwarnings("ignore", category=FITSFixedWarning)
_astropy_log.setLevel("WARNING")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STORE_VERSION = 1
CENTER_RA, CENTER_DEC = 110.8020, -73.4586
BOX_ARCSEC = 480.0
TILE_ARCSEC = 30.0
ARCSEC2_TO_SR = (math.pi / 648000.0) ** 2
# x0, y0 are float64 but rounded to a multiple of 2**-POS_QUANT_BITS arcsec (default 2**-20 = 0.95 microarcsec; max error 0.0005 mas):
# the low mantissa bits are noise and cost ~4.5 bytes per coordinate in the file. 0 = exact float64.
POS_QUANT_BITS = 20

# Halo [arcsec] per instrument family (task defaults).
HALO_ARCSEC = {"NIRCAM": 2.5, "NIRISS": 2.5, "HST": 2.5, "MIRI": 5.0, "IRAC": 10.0, "GALEX": 15.0, "SPHEREX": 30.0}
TELESCOPE_DIAMETER_M = {"JWST": 6.5, "HST": 2.4, "SPITZER": 0.85, "GALEX": 0.5, "SPHEREX": 0.20}

# Fixed detector registry (det_id = index). Names follow v31 / hst_io detname.
DETECTORS = ["NRCA1", "NRCA2", "NRCA3", "NRCA4", "NRCALONG", "NRCB1", "NRCB2", "NRCB3", "NRCB4", "NRCBLONG", "NIS", "MIRIMAGE",
             "WFC3_IR", "ACS_WFC1", "ACS_WFC2", "WFC3_UVIS1", "WFC3_UVIS2", "IRAC1", "IRAC2", "IRAC3", "IRAC4",
             "GALEX_FUV", "GALEX_NUV"] + [f"SPHEREX_D{i}" for i in range(1, 7)]
DET_ID = {n: i for i, n in enumerate(DETECTORS)}

# Pivot wavelengths [um] from the SVO Filter Profile Service (WavelengthPivot), queried 2026-11-05.
PIVOT_UM = {
    "NIRCam": {"F070W": 0.70391, "F090W": 0.90215, "F115W": 1.15426, "F140M": 1.40532, "F150W": 1.50074, "F162M": 1.62725,
               "F164N": 1.64454, "F150W2": 1.65921, "F182M": 1.84517, "F187N": 1.87390, "F200W": 1.98865, "F210M": 2.09545,
               "F212N": 2.12132, "F250M": 2.50323, "F277W": 2.76174, "F300M": 2.98912, "F322W2": 3.23198, "F323N": 3.23685,
               "F335M": 3.36207, "F356W": 3.56836, "F360M": 3.62418, "F405N": 4.05165, "F410M": 4.08224, "F430M": 4.28126,
               "F444W": 4.40432, "F460M": 4.62993, "F466N": 4.65443, "F470N": 4.70779, "F480M": 4.81819},
    "NIRISS": {"F090W": 0.90039, "F115W": 1.15010, "F140M": 1.40513, "F150W": 1.49829, "F158M": 1.58665, "F200W": 1.98368,
               "F277W": 2.77579, "F356W": 3.59482, "F380M": 3.82836, "F430M": 4.28551, "F444W": 4.43460, "F480M": 4.81682},
    "MIRI": {"F560W": 5.63526, "F770W": 7.63933, "F1000W": 9.95312, "F1130W": 11.30850, "F1280W": 12.81014, "F1500W": 15.06351,
             "F1800W": 17.98372, "F2100W": 20.79501, "F2550W": 25.36400},
    "IRAC": {"I1": 3.53784, "I2": 4.47805, "I3": 5.69618, "I4": 7.79784},
    "GALEX": {"FUV": 0.15351, "NUV": 0.23008},
}
# SPHEREx detector wavelength ranges [um] (manifest_spherex_union.csv wave_min_um/wave_max_um).
SPHEREX_RANGE_UM = {1: (0.7455, 1.1167), 2: (1.1033, 1.6499), 3: (1.6301, 2.4346), 4: (2.4027, 3.8473), 5: (3.8113, 4.4300),
                    6: (4.4115, 5.0096)}

# ---- quality policies (see mission readers for sources)
JWST_DQ_GOOD = 0                       # DQ == 0, as smacs_four_v31.load_exposures / v28
IRAC_FATAL_BITS = 32520                # IRAC Instrument Handbook sect. 7.1.1: "DCE Status Mask Fatal Bit Pattern = 32520 (bits 3, 8-14)"
# SPHEREx Explanatory Supplement v2.0 (2026-08-25) sect. 3.4.3 (Mosaic Tool bad-pixel masking) bit numbers, Table 16.
SPHEREX_FATAL_NAMES = {"TRANSIENT": 0, "SUR_ERROR": 2, "NONFUNC": 6, "DICHROIC": 7, "MISSING_DATA": 9, "HOT": 10, "COLD": 11,
                       "PHANMISS": 14, "NONLINEAR": 15, "PERSIST": 17, "CROSSTALK": 20, "GHOST": 22, "GHOST_FPA": 23,
                       "GHOST_EXT": 24, "STREAK": 25, "BLOOM": 26, "SNOWBALL": 27, "HALO": 28, "SATELLITE_HALO": 29}
SPHEREX_FATAL_MASK = sum(1 << b for b in SPHEREX_FATAL_NAMES.values())
# GALEX standard AB zero points (Morrissey et al. 2007, ApJS 173, 682, Table 2): m_AB = ZP - 2.5 log10(count rate [cps]).
GALEX_ZP_AB = {"FUV": 18.82, "NUV": 20.08}
GALEX_PIXEL_ARCSEC = 1.5               # intensity-map pixel (CDELT = 0.000416667 deg)
GALEX_FLAG_BIN = 8                     # -flags map is binned 8 x 8 relative to -int/-rrhr/-skybg


def family_of(instrument):
    """Halo family of an exposure-table ``instrument`` value."""
    u = str(instrument).upper()
    return u if u in HALO_ARCSEC else ("HST" if "/" in u else u)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================================== grid
class Grid:
    """Global tangent plane and tiling (see module docstring)."""

    def __init__(self, tile_arcsec=TILE_ARCSEC, box_arcsec=BOX_ARCSEC, ra=CENTER_RA, dec=CENTER_DEC, halos=None, pos_quant_bits=POS_QUANT_BITS):
        self.T, self.B, self.ra, self.dec = float(tile_arcsec), float(box_arcsec), float(ra), float(dec)
        self.q = int(pos_quant_bits)
        self.n = int(math.ceil(self.B / self.T - 1e-9))
        self.h = self.B / 2.0
        self.halos = dict(HALO_ARCSEC)
        self.halos.update(halos or {})
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [self.ra, self.dec]
        w.wcs.crpix = [1.0, 1.0]
        w.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
        w.wcs.cunit = ["deg", "deg"]
        w.wcs.set()
        self.wcs = w

    def config(self):
        return {"version": STORE_VERSION, "center_ra": self.ra, "center_dec": self.dec, "box_arcsec": self.B,
                "tile_arcsec": self.T, "n_tiles_per_axis": self.n, "halo_arcsec": self.halos, "pos_quant_bits": self.q,
                "plane": "TAN about the centre; x0 = arcsec west (RA decreasing), y0 = arcsec north"}

    def halo_for(self, family):
        return float(self.halos[family])

    def world_to_plane(self, ra, dec):
        x, y = self.wcs.all_world2pix(np.asarray(ra, float), np.asarray(dec, float), 0)
        return np.asarray(x), np.asarray(y)

    def quantize(self, v):
        if self.q <= 0:
            return v
        f = float(2 ** self.q)
        return np.round(v * f) / f

    def plane_to_world(self, x, y):
        ra, dec = self.wcs.all_pix2world(np.asarray(x, float), np.asarray(y, float), 0)
        return np.asarray(ra), np.asarray(dec)

    def tile_extent(self, ix, iy):
        """Core extent (xmin, xmax, ymin, ymax) of tile (ix, iy) on the plane [arcsec]."""
        return (ix * self.T - self.h, (ix + 1) * self.T - self.h, iy * self.T - self.h, (iy + 1) * self.T - self.h)

    def membership(self, x, y, halo):
        """Per pixel: (ix_lo, ix_hi, iy_lo, iy_hi, core_ix, core_iy, core_valid), tile ranges clipped to the grid; a pixel is in the
        store iff ix_lo <= ix_hi and iy_lo <= iy_hi."""
        T, n = self.T, self.n
        fx, fy = (x + self.h) / T, (y + self.h) / T
        hh = halo / T
        with np.errstate(invalid="ignore"):
            ixl = np.clip(np.ceil(fx - hh - 1.0), 0, n - 1)
            ixh = np.clip(np.floor(fx + hh), 0, n - 1)
            iyl = np.clip(np.ceil(fy - hh - 1.0), 0, n - 1)
            iyh = np.clip(np.floor(fy + hh), 0, n - 1)
            cx, cy = np.floor(fx), np.floor(fy)
            cv = (cx >= 0) & (cx < n) & (cy >= 0) & (cy < n)
            # a pixel entirely beyond the halo of every tile has clipped bounds that no longer satisfy the closed-interval test
            inx = (fx - hh <= n) & (fx + hh >= 0)
            iny = (fy - hh <= n) & (fy + hh >= 0)
        ok = np.isfinite(x) & np.isfinite(y) & inx & iny
        ixl = np.where(ok, ixl, 1).astype(np.int32)
        ixh = np.where(ok, ixh, 0).astype(np.int32)
        iyl = np.where(ok, iyl, 1).astype(np.int32)
        iyh = np.where(ok, iyh, 0).astype(np.int32)
        return ixl, ixh, iyl, iyh, cx.astype(np.int32), cy.astype(np.int32), cv & ok

    def expand(self, x, y, halo):
        """All (pixel index, tile ix, tile iy, in_core) memberships, sorted by (tile code, pixel index)."""
        ixl, ixh, iyl, iyh, cx, cy, cv = self.membership(x, y, halo)
        valid = (ixl <= ixh) & (iyl <= iyh)
        if not valid.any():
            e = np.zeros(0, np.int32)
            return e, e, e, np.zeros(0, bool)
        sx, sy = int((ixh - ixl)[valid].max()) + 1, int((iyh - iyl)[valid].max()) + 1
        idx_l, tx_l, ty_l = [], [], []
        for dx in range(sx):
            for dy in range(sy):
                m = valid & (ixl + dx <= ixh) & (iyl + dy <= iyh)
                if m.any():
                    ii = np.nonzero(m)[0].astype(np.int32)
                    idx_l.append(ii)
                    tx_l.append(ixl[ii] + dx)
                    ty_l.append(iyl[ii] + dy)
        idx, tix, tiy = np.concatenate(idx_l), np.concatenate(tx_l), np.concatenate(ty_l)
        code = tix.astype(np.int64) * self.n + tiy
        order = np.lexsort((idx, code))
        idx, tix, tiy = idx[order], tix[order], tiy[order]
        core = cv[idx] & (cx[idx] == tix) & (cy[idx] == tiy)
        return idx, tix, tiy, core


# =============================================================================================== helpers
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        return _jsonable(o.tolist())
    return o


_WCS_PREFIXES = ("CTYPE", "CUNIT", "CRPIX", "CRVAL", "CD1_", "CD2_", "PC1_", "PC2_", "CDELT", "LONPOLE", "LATPOLE", "RADESYS",
                 "EQUINOX", "WCSAXES", "WCSNAME", "A_", "B_", "AP_", "BP_", "PV", "PS", "DP1", "DP2", "CPDIS", "D2IM", "CPERR",
                 "CROTA", "NAXIS", "MJDREF", "IDCSCALE", "IDCTAB", "ORIENTAT", "VAFACTOR", "CCDCHIP")


def wcs_header_subset(hdr):
    """WCS-relevant keywords of a header as a plain dict (SIP/distortion terms included; lookup-table data are NOT)."""
    out = {}
    for k in hdr.keys():
        if k and k not in ("COMMENT", "HISTORY") and k.startswith(_WCS_PREFIXES):
            v = hdr[k]
            if isinstance(v, (int, float, str, bool)):
                out[k] = v
    return _jsonable(out)


def exposure_id_for(key):
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), "big") & 0x7FFFFFFF


def pixel_scale_arcsec(w):
    return float(np.mean(proj_plane_pixel_scales(w.celestial))) * 3600.0


def crop_bbox(w, ny, nx, grid, halo, probe_arcsec=10.0):
    """Pixel bounding box (y0, y1, x0, x1) of the part of an ny x nx image that may fall in the store, from a lattice of ~probe_arcsec
    spacing (one extra lattice step of margin). None if no lattice point falls inside. A footprint sliver narrower than the
    lattice spacing that holds no lattice point is missed (documented approximation)."""
    scale = pixel_scale_arcsec(w)
    step = int(np.clip(round(probe_arcsec / scale), 2, 256))
    ys = np.unique(np.append(np.arange(0, ny, step), ny - 1))
    xs = np.unique(np.append(np.arange(0, nx, step), nx - 1))
    X, Y = np.meshgrid(xs, ys)
    ra, dec = w.all_pix2world(X.ravel().astype(float), Y.ravel().astype(float), 0)
    px, py = grid.world_to_plane(ra, dec)
    ixl, ixh, iyl, iyh, _, _, _ = grid.membership(px, py, halo + 0.0)
    inside = (ixl <= ixh) & (iyl <= iyh)
    if not inside.any():
        return None
    xi, yi = X.ravel()[inside], Y.ravel()[inside]
    return (max(int(yi.min()) - step, 0), min(int(yi.max()) + step + 1, ny), max(int(xi.min()) - step, 0), min(int(xi.max()) + step + 1, nx))


def positions(w, xw, yw, grid, chunk=2_000_000):
    """x0, y0 [arcsec] on the global plane of pixels (xw, yw) (0-based pixel coordinates of WCS ``w``), float64."""
    n = xw.size
    x0, y0 = np.empty(n), np.empty(n)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        ra, dec = w.all_pix2world(xw[a:b].astype(float), yw[a:b].astype(float), 0)
        x0[a:b], y0[a:b] = grid.world_to_plane(ra, dec)
    return x0, y0


@dataclass
class Unit:
    """One detector image ready to write: ``meta`` is the exposure-table record, ``arr`` the kept rows (raster order)."""
    key: str
    meta: dict
    arr: dict
    n_region: int          # pixels of the image that fall in at least one tile (before quality mask)
    n_kept: int            # ... after the default quality mask
    family: str = ""
    notes: dict = field(default_factory=dict)


def assemble(grid, family, w, xw, yw, row, col, data, ivar, dq, good, policy, lam=None):
    """Positions + tile membership + quality mask for a flattened crop. Returns (arrays dict, n_region, n_kept)."""
    x0, y0 = positions(w, xw, yw, grid)
    x0, y0 = grid.quantize(x0), grid.quantize(y0)
    halo = grid.halo_for(family)
    ixl, ixh, iyl, iyh, _, _, _ = grid.membership(x0, y0, halo)
    in_store = (ixl <= ixh) & (iyl <= iyh)
    n_region = int(in_store.sum())
    if policy == "default":
        sel = in_store & good
        ivar_out = ivar
    else:  # "none": keep every pixel with a position, bad ones get ivar = 0
        sel = in_store
        ivar_out = np.where(good, ivar, 0.0).astype(np.float32)
    n_kept = int((in_store & good).sum())
    d = np.where(np.isfinite(data), data, 0.0).astype(np.float32) if policy != "default" else data
    out = {"row": row[sel], "col": col[sel], "data": d[sel], "ivar": ivar_out[sel], "dq": dq[sel], "x0": x0[sel], "y0": y0[sel]}
    out["lam"] = None if lam is None else lam[sel]
    for k in ("row", "col"):
        if out[k].size and (out[k].max() > 32767 or out[k].min() < -32768):
            raise ValueError(f"{k} does not fit int16")
    return out, n_region, n_kept


def raster(y0, y1, x0, x1):
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return yy.ravel().astype(np.int32), xx.ravel().astype(np.int32)


# =============================================================================================== units
BUNIT_TO_MJYSR = {"mjy/sr": 1.0, "jy/sr": 1e-6, "kjy/sr": 1e-3, "gjy/sr": 1e3}


def bunit_factor(bunit):
    k = re.sub(r"\s+", "", str(bunit)).lower()
    if k not in BUNIT_TO_MJYSR:
        raise ValueError(f"unsupported BUNIT {bunit!r}")
    return BUNIT_TO_MJYSR[k]


def _fits_str(h, key, default=""):
    v = h.get(key, default)
    return default if v is None else str(v).strip()


# ------------------------------------------------------------------------------------- JWST
_JWST_INST = {"NIRCAM": "NIRCam", "NIRISS": "NIRISS", "MIRI": "MIRI"}


def jwst_band(h0):
    """Band name (no namespace): the pupil-wheel filter when one is used (NIRISS; NIRCam F162M/F164N/F323N/F405N/F466N/F470N with a
    blocking wide filter), else FILTER. The v31 rule (FILTER or PUPIL when it is one of the six wide bands) is a special case."""
    f, p = _fits_str(h0, "FILTER").upper(), _fits_str(h0, "PUPIL").upper()
    inst = _fits_str(h0, "INSTRUME").upper()
    if inst in ("NIRCAM", "NIRISS") and re.fullmatch(r"F\d{3}[WMN]2?", p) and (f in ("CLEAR", "") or p != f):
        return p
    return f


def read_jwst(path, grid, opts):
    """JWST level-2 ``_cal`` image (NIRCam, NIRISS, MIRI). data = SCI (MJy/sr as delivered), ivar = 1/ERR^2, dq = DQ (uint32 bits,
    stored int32), default mask = DQ == 0 & finite SCI & finite ERR > 0 (smacs_four_v31.load_exposures / v28)."""
    with fits.open(path, memmap=False) as h:
        h0, sh = h[0].header, h["SCI"].header
        inst = _JWST_INST[_fits_str(h0, "INSTRUME").upper()]
        family = inst.upper()
        det = _fits_str(h0, "DETECTOR").upper()
        band = jwst_band(h0)
        bunit = _fits_str(sh, "BUNIT")
        fac = bunit_factor(bunit)
        if fac != 1.0:
            raise ValueError(f"{path}: SCI BUNIT {bunit!r} is not MJy/sr")
        w = WCS(sh, fobj=h)
        ny, nx = h["SCI"].data.shape
        halo = grid.halo_for(family)
        bb = crop_bbox(w, ny, nx, grid, halo)
        if bb is None:
            return []
        y0, y1, x0, x1 = bb
        sci = h["SCI"].data[y0:y1, x0:x1].astype(np.float32).ravel()
        err = h["ERR"].data[y0:y1, x0:x1].astype(np.float32).ravel()
        dq = h["DQ"].data[y0:y1, x0:x1].ravel().astype(np.uint32).view(np.int32)
        pixar = float(sh.get("PIXAR_SR", np.nan))
        hd = wcs_header_subset(sh)
        extra = {k: (_jsonable(h0.get(k)) if k in h0 else None) for k in
                 ("PROGRAM", "OBSERVTN", "VISIT", "EXPOSURE", "EXP_TYPE", "SUBARRAY", "READPATT", "NGROUPS", "NINTS", "TARGNAME",
                  "FILTER", "PUPIL", "CAL_VER", "CRDS_CTX", "VISITGRP", "SEQ_ID", "ACT_ID")}
        extra.update({"PIXAR_A2": _jsonable(sh.get("PIXAR_A2")), "S_REGION": sh.get("S_REGION"), "BUNIT": bunit,
                      "EFFEXPTM": _jsonable(h0.get("EFFEXPTM")), "DURATION": _jsonable(h0.get("DURATION"))})
        mjd = float(h0.get("EXPMID", h0.get("EXPSTART", np.nan)))
        mjd0 = float(h0.get("EXPSTART", np.nan))
        exptime = float(h0.get("EFFEXPTM", h0.get("XPOSURE", np.nan)))
    row, col = raster(y0, y1, x0, x1)
    good = np.isfinite(sci) & np.isfinite(err) & (err > 0) & (dq == JWST_DQ_GOOD)
    with np.errstate(divide="ignore", invalid="ignore"):
        ivar = (1.0 / (err * err)).astype(np.float32)
    arr, n_reg, n_kept = assemble(grid, family, w, col.astype(float), row.astype(float), row, col, sci, ivar, dq, good, opts["mask_policy"])
    if n_reg == 0:
        return []
    lam = PIVOT_UM[inst].get(band)
    detname = det
    meta = dict(
        name=os.path.basename(path), mission="jwst", path=str(path), instrument=inst, detector=detname, det_id=DET_ID[detname],
        band=f"{family}_{band}", filter=band, mjd=mjd, mjd_start=mjd0, exptime=exptime, units="MJy/sr", unit_factor=1.0,
        unit_conversion=f"none: SCI BUNIT = {bunit!r} verified == MJy/sr (surface brightness); ERR in the same units",
        pixar_sr_nominal=pixar, wcs_policy="header", wcs_name="SCI-header WCS (TAN-SIP fit of the gwcs, all A/B/AP/BP terms)",
        telescope_diameter_m=TELESCOPE_DIAMETER_M["JWST"], lam_pivot_um=lam, lam_min_um=lam, lam_max_um=lam,
        dq_policy="DQ == 0 & finite(SCI, ERR) & ERR > 0", nx=nx, ny=ny, header=hd, extra=extra)
    return [Unit(f"jwst:{os.path.basename(path)}", meta, arr, n_reg, n_kept, family)]


# ------------------------------------------------------------------------------------- HST (hst_io)
def hst_not_science(hdul, h0):
    """Reason string if an HST file is a calibration frame / uncalibrated (TARGNAME DARK|BIAS, IMAGETYP DARK|BIAS, BUNIT COUNTS,
    PHOTFNU and PHOTFLAM both 0 in the first SCI header or the primary header), else None."""
    targ, typ = _fits_str(h0, "TARGNAME").upper(), _fits_str(h0, "IMAGETYP").upper()
    if targ in ("DARK", "BIAS") or typ in ("DARK", "BIAS"):
        return f"calibration frame (TARGNAME={targ!r}, IMAGETYP={typ!r})"
    try:
        sh = hdul[("SCI", 1)].header
    except KeyError:
        return "no SCI extension"
    bunit = (_fits_str(sh, "BUNIT") or _fits_str(h0, "BUNIT")).upper()
    if bunit == "COUNTS":
        return "BUNIT=COUNTS (uncalibrated units)"
    def num(k):
        v = sh.get(k, h0.get(k, None))
        return None if v in (None, "") else float(v)
    if num("PHOTFNU") == 0.0 and not num("PHOTFLAM"):
        return "PHOTFNU=0 and PHOTFLAM=0 (no photometric calibration)"
    return None


def read_hst(path, grid, opts):
    """HST ACS/WFC, WFC3/IR, WFC3/UVIS flt/flc through hst_io: a priori WCS (policy ``--hst-wcs``, default apriori; SIP + D2IM +
    NPOL/CPDIS tables), MJy/sr per nominal pixel (hst_io.units_conversion), DQ good-bit policy of hst_io (default = drizzlepac HAP
    final_bits per instrument). One unit per SCI extension (chip)."""
    import hst_io
    hst_io.WCS_POLICY = opts.get("hst_wcs", "apriori")
    dq_policy = opts.get("hst_dq_policy", "default")
    units = []
    with fits.open(str(path), memmap=False) as hdul:
        h0 = hdul[0].header
        inst = hst_io._identify(h0)
        if inst not in hst_io.DQ_GOOD_BITS:
            log(f"skip {os.path.basename(path)}: instrument/detector {inst} not supported")
            return []
        reason = hst_not_science(hdul, h0)
        if reason:
            log(f"skip {os.path.basename(path)}: {reason}")
            return []
        root = str(h0.get("ROOTNAME", "") or os.path.basename(path).split("_")[0]).strip().lower()
        filt = hst_io._filter_name(h0)
        mjd0 = float(h0.get("EXPSTART", np.nan))
        mjd1 = float(h0.get("EXPEND", np.nan))
        good_bits = hst_io._good_bits(inst, mjd0, dq_policy)
        halo = grid.halo_for("HST")
        for ext in [x.ver for x in hdul if x.name == "SCI"]:
            sh = hdul[("SCI", ext)].header
            try:
                w = hst_io.hst_wcs(hdul, ext)
                conv = hst_io.units_conversion(sh, h0, inst)
            except Exception as exc:  # noqa: BLE001
                log(f"skip {os.path.basename(path)} sci{ext}: {exc}")
                continue
            sci_hdu = hdul[("SCI", ext)]
            ny, nx = sci_hdu.shape
            bb = crop_bbox(w, ny, nx, grid, halo)
            if bb is None:
                continue
            y0, y1, x0, x1 = bb
            raw = sci_hdu.data[y0:y1, x0:x1].astype(np.float64).ravel()
            err = hdul[("ERR", ext)].data[y0:y1, x0:x1].astype(np.float64).ravel()
            dq16 = hdul[("DQ", ext)].data[y0:y1, x0:x1].ravel()
            dqu = (dq16.astype(np.int64) & 0xFFFF).astype(np.int32)
            row, col = raster(y0, y1, x0, x1)
            d32 = (raw * conv["to_mjysr"]).astype(np.float32)
            e32 = (err * conv["to_mjysr"]).astype(np.float32)
            good = np.isfinite(raw) & np.isfinite(err) & (err > 0) & hst_io.dq_pass(dq16, good_bits)
            with np.errstate(divide="ignore", invalid="ignore"):
                ivar = (1.0 / (e32 * e32)).astype(np.float32)
            arr, n_reg, n_kept = assemble(grid, "HST", w, col.astype(float), row.astype(float), row, col, d32, ivar, dqu, good,
                                          opts["mask_policy"])
            if n_reg == 0:
                continue
            chip = sh.get("CCDCHIP", None)
            detname = "WFC3_IR" if inst == "WFC3/IR" else f"{inst.replace('/', '_')}{int(chip) if chip is not None else ext}"
            short = hst_io._SHORT[inst]
            hd = wcs_header_subset(sh)
            hd.update({"CRVAL1": float(w.wcs.crval[0]), "CRVAL2": float(w.wcs.crval[1]), "CRPIX1": float(w.wcs.crpix[0]),
                       "CRPIX2": float(w.wcs.crpix[1])})
            cd = w.wcs.cd if w.wcs.has_cd() else w.wcs.get_pc() * w.wcs.cdelt[:, None]
            hd.update({"CD1_1": float(cd[0, 0]), "CD1_2": float(cd[0, 1]), "CD2_1": float(cd[1, 0]), "CD2_2": float(cd[1, 1])})
            for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2"):
                hd.pop(k, None)
            pl = conv["photplam"]
            lam = None if pl is None else pl / 1e4
            bi = hst_io.hst_band_info(inst, filt) if filt != "CLEAR" else {"svo_id": None, "diameter_m": 2.4}
            meta = dict(
                name=f"{root}_sci{ext}", mission="hst", path=str(path), instrument=inst, detector=detname, det_id=DET_ID[detname],
                band=f"{short}_{filt}", filter=filt, mjd=(0.5 * (mjd0 + mjd1) if np.isfinite(mjd1) else mjd0), mjd_start=mjd0,
                exptime=conv["exptime"], units="MJy/sr per nominal pixel (surface brightness)", unit_factor=conv["to_mjysr"],
                unit_conversion=(f"hst_io.units_conversion: BUNIT={conv['bunit']}; d = raw{'/EXPTIME' if conv['divided_by_exptime'] else ''}"
                                 f" * PHOTFNU({conv['photfnu_jy']:.6g} Jy s/e-, from {conv['photfnu_source']}) * 1e-6 / Omega_nom, "
                                 f"Omega_nom = ({conv['nominal_scale']} arcsec [{conv['nominal_scale_source']}])^2; true/nominal "
                                 f"area ratio NOT applied"),
                pixar_sr_nominal=conv["omega_nom_sr"], wcs_policy=hst_io.WCS_POLICY, wcs_name=getattr(w, "wcs_name_used", ""),
                telescope_diameter_m=TELESCOPE_DIAMETER_M["HST"], lam_pivot_um=lam, lam_min_um=lam, lam_max_um=lam,
                dq_policy=f"hst_io dq_policy={dq_policy!r}: good bits {good_bits} (any DQ bit outside is fatal) & finite(SCI,ERR) & ERR>0",
                nx=nx, ny=ny, header=hd,
                extra={"photplam_A": pl, "photflam": conv["photflam"], "photfnu_jy": conv["photfnu_jy"], "svo_id": bi["svo_id"],
                       "ext": ext, "dq_good_bits": good_bits, "ccdchip": None if chip is None else int(chip),
                       "proposid": _jsonable(h0.get("PROPOSID")), "has_lookup_tables": bool(w.cpdis1 is not None or w.det2im1 is not None),
                       "nominal_scale_arcsec": conv["nominal_scale"], "to_mjysr": conv["to_mjysr"]})
            units.append(Unit(f"hst:{root}_sci{ext}", meta, arr, n_reg, n_kept, "HST"))
    return units


# ------------------------------------------------------------------------------------- Spitzer IRAC
def read_spitzer(path, grid, opts):
    """IRAC cbcd (+ cbunc, bimsk). data = cbcd, ivar = 1/cbunc^2 (both BUNIT MJy/sr, verified), dq = bimsk, default mask = no bit of
    the IRAC Instrument Handbook fatal pattern 32520 (bits 3, 8-14) set & finite & cbunc > 0."""
    path = str(path)
    unc, msk = path.replace("_cbcd.fits", "_cbunc.fits"), path.replace("_cbcd.fits", "_bimsk.fits")
    with fits.open(path, memmap=False) as h:
        hdr = h[0].header
        w = WCS(hdr)
        sci_all = h[0].data
        ny, nx = sci_all.shape
        halo = grid.halo_for("IRAC")
        bb = crop_bbox(w, ny, nx, grid, halo)
        if bb is None:
            return []
        y0, y1, x0, x1 = bb
        sci = sci_all[y0:y1, x0:x1].astype(np.float32).ravel()
        bunit = _fits_str(hdr, "BUNIT")
        if bunit_factor(bunit) != 1.0:
            raise ValueError(f"{path}: BUNIT {bunit!r} is not MJy/sr")
        hd = wcs_header_subset(hdr)
        ch = int(hdr["CHNLNUM"])
        mjd0 = float(hdr["MJD_OBS"])
        frame = float(hdr.get("FRAMTIME", 0.0))
        exptime = float(hdr.get("EXPTIME", np.nan))
        extra = {k: _jsonable(hdr.get(k)) for k in ("FLUXCONV", "GAIN", "FRAMTIME", "AORKEY", "AOT_TYPE", "PROGID", "ZODY_EST",
                                                    "SKYDRKZB", "SKYDKMED", "CREATOR", "BUNIT", "PXSCAL1", "PXSCAL2", "AREADMOD")}
        pxa = abs(float(hdr["PXSCAL1"]) * float(hdr["PXSCAL2"]))
    with fits.open(unc, memmap=False) as h:
        if bunit_factor(_fits_str(h[0].header, "BUNIT")) != 1.0:
            raise ValueError(f"{unc}: BUNIT is not MJy/sr")
        err = h[0].data[y0:y1, x0:x1].astype(np.float32).ravel()
    with fits.open(msk, memmap=False) as h:
        bim = h[0].data[y0:y1, x0:x1].ravel().astype(np.int32) & 0xFFFF
    row, col = raster(y0, y1, x0, x1)
    good = np.isfinite(sci) & np.isfinite(err) & (err > 0) & ((bim & IRAC_FATAL_BITS) == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ivar = (1.0 / (err * err)).astype(np.float32)
    arr, n_reg, n_kept = assemble(grid, "IRAC", w, col.astype(float), row.astype(float), row, col, sci, ivar, bim.astype(np.int32), good,
                                  opts["mask_policy"])
    if n_reg == 0:
        return []
    lam = PIVOT_UM["IRAC"][f"I{ch}"]
    detname = f"IRAC{ch}"
    stem = os.path.basename(path).replace("_cbcd.fits", "")
    meta = dict(
        name=stem, mission="spitzer", path=path, instrument="IRAC", detector=detname, det_id=DET_ID[detname], band=f"IRAC_I{ch}",
        filter=f"I{ch}", mjd=mjd0 + 0.5 * frame / 86400.0, mjd_start=mjd0, exptime=exptime, units="MJy/sr", unit_factor=1.0,
        unit_conversion=f"none: cbcd and cbunc BUNIT = {bunit!r} == MJy/sr (surface brightness); pixel-area (distortion) variation not applied",
        pixar_sr_nominal=pxa * ARCSEC2_TO_SR, wcs_policy="header", wcs_name="cbcd header TAN-SIP (pointing-reconstructed CD)",
        telescope_diameter_m=TELESCOPE_DIAMETER_M["SPITZER"], lam_pivot_um=lam, lam_min_um=lam, lam_max_um=lam,
        dq_policy=f"bimsk & {IRAC_FATAL_BITS} == 0 (fatal bits 3, 8-14; IRAC Instrument Handbook 7.1.1) & finite & cbunc>0",
        nx=nx, ny=ny, header=hd, extra=extra)
    return [Unit(f"spitzer:{stem}", meta, arr, n_reg, n_kept, "IRAC")]


# ------------------------------------------------------------------------------------- SPHEREx
def read_spherex(path, grid, opts):
    """SPHEREx L2 cutout. data = IMAGE (zodiacal light NOT subtracted; ZODI ext. not stored per pixel), ivar = 1/VARIANCE,
    dq = FLAGS, default mask = no bit of SPHEREX_FATAL_MASK (Explanatory Supplement v2.0 sect. 3.4.3 list) & finite & VARIANCE>0.
    lambda_um from the per-pixel wavelength WCS (alternate 'W', WAVE-TAB); detector row/col from alternate 'A' (0-based active pixels)."""
    path = str(path)
    with fits.open(path, memmap=False) as h:
        ih = h["IMAGE"].header
        w = WCS(ih, fobj=h)
        ny, nx = h["IMAGE"].data.shape
        bunit, vunit = _fits_str(ih, "BUNIT"), _fits_str(h["VARIANCE"].header, "BUNIT")
        fac = bunit_factor(bunit)
        vfac = {"mjy2/sr2": 1.0}.get(re.sub(r"\s+", "", vunit).lower())
        if vfac is None:
            raise ValueError(f"{path}: VARIANCE BUNIT {vunit!r} unsupported")
        halo = grid.halo_for("SPHEREX")
        bb = crop_bbox(w, ny, nx, grid, halo)
        if bb is None:
            return []
        y0, y1, x0, x1 = bb
        sci = (h["IMAGE"].data[y0:y1, x0:x1].astype(np.float64) * fac).astype(np.float32).ravel()
        var = (h["VARIANCE"].data[y0:y1, x0:x1].astype(np.float64) * fac * fac).astype(np.float32).ravel()
        fl = h["FLAGS"].data[y0:y1, x0:x1].ravel().astype(np.int32)
        # The SIP polynomial keywords (A_*, B_*, AP_*, BP_*) belong to the celestial WCS only, but astropy would also apply them to the
        # LINEAR ('A') and WAVE-TAB ('W') alternates (shifting detector coordinates by up to ~1.5 px): strip them for those.
        ih_ns = ih.copy()
        for k in [k for k in ih_ns.keys() if re.match(r"^(A|B|AP|BP)_(\d+_\d+|ORDER)$", k)]:
            del ih_ns[k]
        wa = WCS(ih_ns, fobj=h, key="A")
        wl = WCS(ih_ns, fobj=h, key="W")
        det = int(ih["DETECTOR"])
        row, col = raster(y0, y1, x0, x1)
        # detector (active-pixel) coordinates and per-pixel wavelength
        xc, yc = wa.all_pix2world(col.astype(float), row.astype(float), 0)
        drow, dcol = np.rint(yc).astype(np.int32), np.rint(xc).astype(np.int32)
        lam = np.asarray(wl.all_pix2world(col.astype(float), row.astype(float), 0)[0], dtype=np.float32)
        zodi_med = float(np.nanmedian(h["ZODI"].data)) if "ZODI" in h else None
        hd = wcs_header_subset(ih)
        extra = {"ZODI_median_MJy_sr": zodi_med, "BUNIT": bunit, "VARIANCE_BUNIT": vunit, "OBSID": _fits_str(ih, "OBSID"),
                 "EXPIDN": _jsonable(ih.get("EXPIDN")), "OMEGA_MEDIAN": _jsonable(ih.get("OMEGA_MEDIAN")),
                 "PXSRCMSK": _jsonable(ih.get("PXSRCMSK")), "FINAST": _jsonable(ih.get("FINAST")),
                 "TSAMP": _jsonable(ih.get("TSAMP")), "cutout_offset_cols_rows": [x0, y0], "fatal_mask": SPHEREX_FATAL_MASK,
                 "wcs_alt_A": wa.wcs.name if hasattr(wa.wcs, "name") else "A", "alt_W_ctype": list(map(str, wl.wcs.ctype))}
        mjd = float(ih.get("MJD-AVG", ih.get("MJD-OBS", np.nan)))
        mjd0 = float(ih.get("MJD-BEG", mjd))
        exptime = float(ih.get("XPOSURE", np.nan))
        try:
            wcs_name = _fits_str(ih, "WCSNAME") or "IMAGE header TAN-SIP"
        except Exception:  # noqa: BLE001
            wcs_name = "IMAGE header TAN-SIP"
    good = np.isfinite(sci) & np.isfinite(var) & (var > 0) & ((fl & SPHEREX_FATAL_MASK) == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ivar = (1.0 / var).astype(np.float32)
    arr, n_reg, n_kept = assemble(grid, "SPHEREX", w, col.astype(float), row.astype(float), drow, dcol, sci, ivar, fl, good,
                                  opts["mask_policy"], lam=lam)
    if n_reg == 0:
        return []
    rng = SPHEREX_RANGE_UM[det]
    lam_k = arr["lam"]
    detname = f"SPHEREX_D{det}"
    stem = os.path.basename(path).replace(".fits", "")
    meta = dict(
        name=stem, mission="spherex", path=path, instrument="SPHEREx", detector=detname, det_id=DET_ID[detname], band=f"SPHEREX_D{det}",
        filter=f"D{det}", mjd=mjd, mjd_start=mjd0, exptime=exptime, units="MJy/sr", unit_factor=fac,
        unit_conversion=f"IMAGE BUNIT={bunit!r} (== MJy/sr, factor {fac}); VARIANCE BUNIT={vunit!r} used as (MJy/sr)^2; zodi not subtracted",
        pixar_sr_nominal=float(ih.get("OMEGA_MEDIAN", np.nan)) * ARCSEC2_TO_SR,   # OMEGA_MEDIAN = median pixel solid angle [arcsec^2]
        wcs_policy="header", wcs_name=wcs_name, telescope_diameter_m=TELESCOPE_DIAMETER_M["SPHEREX"],
        lam_pivot_um=0.5 * (rng[0] + rng[1]), lam_min_um=float(lam_k.min()) if lam_k.size else rng[0],
        lam_max_um=float(lam_k.max()) if lam_k.size else rng[1],
        dq_policy=f"FLAGS & {SPHEREX_FATAL_MASK:#x} == 0 (bits {sorted(SPHEREX_FATAL_NAMES.values())}) & finite & VARIANCE>0",
        nx=2040, ny=2040, header=hd, extra=extra)
    return [Unit(f"spherex:{stem}", meta, arr, n_reg, n_kept, "SPHEREX")]


# ------------------------------------------------------------------------------------- GALEX
def read_galex(path, grid, opts):
    """GALEX intensity map ``*-int.fits[.gz]`` (+ -rrhr, -skybg, -flags). data = int [counts/s/pixel] * K (MJy/sr per cps/pix from
    the AB zero point), ivar from Poisson statistics: var(rate) = max(int, skybg)/rrhr (counts = rate*rrhr), dq = -flags (8x8-binned map,
    upsampled), default mask = flags == 0 & rrhr > 0 & finite & skybg > 0."""
    path = str(path)
    gz = ".gz" if path.endswith(".gz") else ""
    m = re.match(r"(.*)-int\.fits(\.gz)?$", path)
    base = m.group(1)
    with fits.open(path, memmap=False) as h:
        ih = h[0].header
        w = WCS(ih)
        ny, nx = h[0].data.shape
        halo = grid.halo_for("GALEX")
        bb = crop_bbox(w, ny, nx, grid, halo)
        if bb is None:
            return []
        y0, y1, x0, x1 = bb
        cps = h[0].data[y0:y1, x0:x1].astype(np.float32).ravel()
        band_code = int(ih["BAND"])
        band = {1: "NUV", 2: "FUV"}[band_code]
        hd = wcs_header_subset(ih)
        exptime = float(ih.get("EXPTIME", np.nan))
        tstart, tend = float(ih["EXPSTART"]), float(ih.get("EXPEND", ih["EXPSTART"]))
        extra = {k: _jsonable(ih.get(k)) for k in ("OBJECT", "TILE", "VISIT", "ECLIPSE", "GRELEASE", "PSFFWHM", "RRMED", "NDEAD",
                                                   "OBS-DATE", "EXPTIME", "BAND", "FLATFL")}
    with fits.open(f"{base}-rrhr.fits{gz}", memmap=False) as h:
        rr = h[0].data[y0:y1, x0:x1].astype(np.float32).ravel()
    with fits.open(f"{base}-skybg.fits{gz}", memmap=False) as h:
        sky = h[0].data[y0:y1, x0:x1].astype(np.float32).ravel()
    with fits.open(f"{base}-flags.fits{gz}", memmap=False) as h:
        fgrid = h[0].data.astype(np.int32)
    row, col = raster(y0, y1, x0, x1)
    fl = fgrid[row // GALEX_FLAG_BIN, col // GALEX_FLAG_BIN]
    zp = GALEX_ZP_AB[band]
    omega = (GALEX_PIXEL_ARCSEC ** 2) * ARCSEC2_TO_SR
    k_mjysr = 3631.0 * 10.0 ** (-0.4 * zp) * 1e-6 / omega     # MJy/sr per (count/s/pixel)
    rate_model = np.maximum(cps, sky)
    good = (np.isfinite(cps) & np.isfinite(rr) & (rr > 0) & np.isfinite(sky) & (sky > 0) & (fl == 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        var_rate = rate_model / rr
        ivar = (1.0 / (var_rate * (k_mjysr ** 2))).astype(np.float32)
    good &= np.isfinite(ivar) & (ivar > 0)
    data = (cps.astype(np.float64) * k_mjysr).astype(np.float32)
    arr, n_reg, n_kept = assemble(grid, "GALEX", w, col.astype(float), row.astype(float), row, col, data, ivar, fl.astype(np.int32), good,
                                  opts["mask_policy"])
    if n_reg == 0:
        return []
    lam = PIVOT_UM["GALEX"][band]
    detname = f"GALEX_{band}"
    stem = os.path.basename(base) + f"-int"
    extra.update({"ab_zero_point": zp, "k_mjysr_per_cps_pix": k_mjysr, "variance_model": "max(int, skybg)/rrhr [cps^2], x K^2",
                  "flags_bin": GALEX_FLAG_BIN, "ivar_floor_note": "pixels with skybg<=0 are masked"})
    meta = dict(
        name=stem, mission="galex", path=path, instrument="GALEX", detector=detname, det_id=DET_ID[detname], band=f"GALEX_{band}",
        filter=band, mjd=40587.0 + 0.5 * (tstart + tend) / 86400.0, mjd_start=40587.0 + tstart / 86400.0, exptime=exptime,
        units="MJy/sr (per 1.5 arcsec map pixel)", unit_factor=k_mjysr,
        unit_conversion=(f"int [count/s/pixel] * 3631 Jy * 10^(-0.4*{zp}) * 1e-6 / Omega_pix, Omega_pix = (1.5 arcsec)^2 "
                         f"(AB ZP {band} = {zp}, Morrissey et al. 2007 ApJS 173, 682); K = {k_mjysr:.6g} MJy/sr per cps/pix"),
        pixar_sr_nominal=omega, wcs_policy="header", wcs_name="int-map header TAN (pipeline sky grid)",
        telescope_diameter_m=TELESCOPE_DIAMETER_M["GALEX"], lam_pivot_um=lam, lam_min_um=lam, lam_max_um=lam,
        dq_policy="flags(8x8 binned, upsampled) == 0 & rrhr > 0 & finite(int, rrhr, skybg) & skybg > 0",
        nx=nx, ny=ny, header=hd, extra=extra)
    return [Unit(f"galex:{stem}", meta, arr, n_reg, n_kept, "GALEX")]


READERS = {"jwst": read_jwst, "hst": read_hst, "spitzer": read_spitzer, "spherex": read_spherex, "galex": read_galex}


# =============================================================================================== input discovery
def _csv_rows(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def find_file(name, dirs):
    for d in dirs:
        for pat in (os.path.join(d, name), os.path.join(d, "*", name), os.path.join(d, "*", "*", name)):
            g = glob.glob(pat)
            if g:
                return g[0]
    return None


def discover(mission, inputs, data_dirs):
    """Sorted list of primary input files (one per extraction unit group)."""
    files = []
    missing = []
    for spec in inputs:
        if spec.lower().endswith(".csv"):
            rows = _csv_rows(spec)
            cols = rows[0].keys() if rows else []
            fcol = "filename" if "filename" in cols else "product_filename"
            for r in rows:
                if mission == "spitzer" and r.get("file_type", "cbcd") != "cbcd":
                    continue
                if mission == "galex" and r.get("product", "int") != "int":
                    continue
                if r.get("hap", "False") == "True":
                    continue
                name = r[fcol]
                dirs = list(data_dirs)
                extra_sub = r.get("aor") or r.get("visit") or r.get("collection")
                f = None
                if extra_sub:
                    f = find_file(os.path.join(extra_sub, name), dirs) or find_file(name, [os.path.join(d, extra_sub) for d in dirs])
                f = f or find_file(name, dirs)
                if f:
                    files.append(f)
                else:
                    missing.append(name)
        else:
            files += glob.glob(spec)
    if missing:
        log(f"{len(missing)} manifest files not found under --data-dir (first: {missing[:3]})")
    suffix = {"jwst": "_cal.fits", "hst": ("_flt.fits", "_flc.fits"), "spitzer": "_cbcd.fits", "spherex": ".fits",
              "galex": ("-int.fits", "-int.fits.gz")}[mission]
    files = sorted({f for f in files if f.endswith(suffix)})
    if mission == "hst":
        files = [f for f in files if not os.path.basename(f).startswith("hst_")]       # HAP-named copies
        files = _select_hst(files)
    return files


def _select_hst(files):
    """flc preferred over flt for one exposure, supported instruments only (smacs_four_v31.select_hst_files rule, via that function
    for each directory)."""
    import smacs_four_v31 as v31
    keep = []
    for d in sorted({os.path.dirname(f) for f in files}):
        sel, _ = v31.select_hst_files(d, None)
        mine = {os.path.abspath(f) for f in files}
        keep += [f for f in sel if os.path.abspath(f) in mine]
    return sorted(keep)


# =============================================================================================== writing
def _atomic_write(path, writer):
    tmp = f"{path}.tmp{os.getpid()}"
    writer(tmp)
    os.replace(tmp, path)


def write_fragment(path, a, sel, tix, tiy, core, eid, det_id):
    n = int(sel.size)
    lam = a["lam"]
    t = pa.table({
        "exposure_id": pa.array(np.full(n, eid, np.int32)), "det_id": pa.array(np.full(n, det_id, np.int16)),
        "row": pa.array(a["row"][sel].astype(np.int16)), "col": pa.array(a["col"][sel].astype(np.int16)),
        "data": pa.array(a["data"][sel]), "ivar": pa.array(a["ivar"][sel]), "dq": pa.array(a["dq"][sel].astype(np.int32)),
        "x0": pa.array(a["x0"][sel]), "y0": pa.array(a["y0"][sel]),
        "tile_ix": pa.array(np.full(n, tix, np.int16)), "tile_iy": pa.array(np.full(n, tiy, np.int16)),
        "in_core": pa.array(core),
        "lambda_um": pa.array(lam[sel].astype(np.float32)) if lam is not None else pa.nulls(n, pa.float32()),
    }, schema=ROW_SCHEMA)
    plain = ["exposure_id", "det_id", "tile_ix", "tile_iy", "in_core"]

    def w(p):
        pq.write_table(t, p, compression="zstd", compression_level=6, use_dictionary=plain,
                       use_byte_stream_split=["data", "ivar", "x0", "y0", "lambda_um"], row_group_size=1 << 20)
    _atomic_write(path, w)


ROW_SCHEMA = pa.schema([
    ("exposure_id", pa.int32()), ("det_id", pa.int16()), ("row", pa.int16()), ("col", pa.int16()), ("data", pa.float32()),
    ("ivar", pa.float32()), ("dq", pa.int32()), ("x0", pa.float64()), ("y0", pa.float64()), ("tile_ix", pa.int16()),
    ("tile_iy", pa.int16()), ("in_core", pa.bool_()), ("lambda_um", pa.float32())])

EXPOSURE_COLUMNS = [
    ("exposure_id", pa.int32()), ("name", pa.string()), ("mission", pa.string()), ("path", pa.string()), ("instrument", pa.string()),
    ("detector", pa.string()), ("det_id", pa.int16()), ("band", pa.string()), ("filter", pa.string()), ("mjd", pa.float64()),
    ("mjd_start", pa.float64()), ("exptime", pa.float64()), ("units", pa.string()), ("unit_factor", pa.float64()),
    ("unit_conversion", pa.string()), ("pixar_sr_nominal", pa.float64()), ("wcs_policy", pa.string()), ("wcs_name", pa.string()),
    ("telescope_diameter_m", pa.float64()), ("lam_pivot_um", pa.float64()), ("lam_min_um", pa.float64()), ("lam_max_um", pa.float64()),
    ("dq_policy", pa.string()), ("nx", pa.int32()), ("ny", pa.int32()), ("n_pix_region", pa.int64()), ("n_pix_kept", pa.int64()),
    ("n_rows", pa.int64()), ("n_tiles", pa.int32()), ("t_extract_s", pa.float64()), ("header_json", pa.string()),
    ("extra_json", pa.string()), ("key", pa.string())]


def write_unit(store, grid, unit, force=False):
    """Write all tile fragments of one unit (skipping existing ones) and its meta record. Returns (n_rows, n_tiles, n_written)."""
    t0 = time.perf_counter()
    eid = exposure_id_for(unit.key)
    meta_path = Path(store) / "exposure_meta" / f"exp_{eid}.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["key"] != unit.key:
            raise RuntimeError(f"exposure_id collision {eid}: {old['key']} vs {unit.key}")
    a = unit.arr
    idx, tix, tiy, core = grid.expand(a["x0"], a["y0"], grid.halo_for(unit.family))
    n_rows = int(idx.size)
    n_written = 0
    n_tiles = 0
    if n_rows:
        code = tix.astype(np.int64) * grid.n + tiy
        cuts = np.nonzero(np.diff(code))[0] + 1
        starts = np.concatenate([[0], cuts])
        ends = np.concatenate([cuts, [n_rows]])
        for s, e in zip(starts, ends):
            ix, iy = int(tix[s]), int(tiy[s])
            n_tiles += 1
            d = Path(store) / "rows" / f"tile_ix={ix}" / f"tile_iy={iy}"
            d.mkdir(parents=True, exist_ok=True)
            fp = d / f"exp_{eid}.parquet"
            if fp.exists() and not force:
                continue
            write_fragment(str(fp), a, idx[s:e], ix, iy, core[s:e], eid, unit.meta["det_id"])
            n_written += 1
    rec = dict(unit.meta)
    rec.update(exposure_id=eid, n_pix_region=unit.n_region, n_pix_kept=unit.n_kept, n_rows=n_rows, n_tiles=n_tiles,
               t_extract_s=None, key=unit.key)
    rec["header_json"] = json.dumps(_jsonable(rec.pop("header")))
    rec["extra_json"] = json.dumps(_jsonable(rec.pop("extra")))
    return rec, n_rows, n_tiles, n_written, time.perf_counter() - t0


def check_store_config(store, grid):
    p = Path(store) / "store.json"
    cfg = grid.config()
    if p.exists():
        old = json.loads(p.read_text())
        if old != json.loads(json.dumps(cfg)):
            raise SystemExit(f"store.json differs from the requested grid configuration:\n  store {old}\n  now   {cfg}")
    else:
        Path(store).mkdir(parents=True, exist_ok=True)
        _atomic_write(str(p), lambda t: Path(t).write_text(json.dumps(cfg, indent=1)))


def file_id(path):
    return hashlib.blake2b(os.path.basename(path).encode(), digest_size=8).hexdigest()


def parse_shard(shard):
    if not shard:
        return 0, 1
    k, n = (int(x) for x in shard.split("/"))
    if not (0 <= k < n):
        raise SystemExit("--shard k/N needs 0 <= k < N")
    return k, n


def pieces_of(tdir):
    """Stored pieces of one tile directory, both layouts: fragments ``exp_<id>.parquet`` and part files ``part-*.parquet`` (one row
    group per exposure). Returns dicts {eid, path, rg (None for a fragment), rows, bytes}; one piece per exposure (first seen)."""
    out, seen = [], set()
    for fp in sorted(Path(tdir).glob("exp_*.parquet")):
        md = pq.ParquetFile(fp).metadata
        eid = int(fp.stem.split("_")[1])
        if eid not in seen:
            seen.add(eid)
            out.append({"eid": eid, "path": str(fp), "rg": None, "rows": md.num_rows, "bytes": fp.stat().st_size})
    for fp in sorted(Path(tdir).glob("part-*.parquet")):
        pf = pq.ParquetFile(fp)
        md = pf.metadata
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            st = rg.column(0).statistics                     # column 0 = exposure_id (constant in a row group)
            eid = int(st.min) if st is not None and st.has_min_max else int(pf.read_row_group(i, columns=["exposure_id"])["exposure_id"][0].as_py())
            if eid not in seen:
                seen.add(eid)
                out.append({"eid": eid, "path": str(fp), "rg": i, "rows": rg.num_rows, "bytes": sum(rg.column(c).total_compressed_size for c in range(rg.num_columns))})
    return out


def read_piece(piece, columns=None):
    if piece["rg"] is None:
        return pq.read_table(piece["path"], columns=columns)
    return pq.ParquetFile(piece["path"]).read_row_group(piece["rg"], columns=columns)


def scan_pieces(store):
    """All pieces of a store (both layouts) with their tile: list of dicts (+ ix, iy)."""
    out = []
    for tdir in sorted(Path(store).glob("rows/tile_ix=*/tile_iy=*")):
        ix, iy = int(tdir.parent.name.split("=")[1]), int(tdir.name.split("=")[1])
        for pc in pieces_of(tdir):
            pc["ix"], pc["iy"] = ix, iy
            out.append(pc)
    return out


def exposure_table(store, eid, columns=None, pieces=None):
    """All stored rows of one exposure over all tiles (both layouts) as one table, and the number of pieces."""
    pcs = [p for p in (pieces if pieces is not None else scan_pieces(store)) if p["eid"] == eid]
    return pa.concat_tables([read_piece(p, columns) for p in pcs]), pcs


def compact_job(work, final, job, grid):
    """Merge the fragments of the local store ``work`` into ONE part file per tile in ``final`` (one row group per exposure), copy
    the exposure_meta records, and write the job done-marker last. A part file left by an interrupted attempt is simply rewritten."""
    t0 = time.perf_counter()
    plain = ["exposure_id", "det_id", "tile_ix", "tile_iy", "in_core"]
    n_parts = n_rows = 0
    exps = set()
    for tdir in sorted(Path(work).glob("rows/tile_ix=*/tile_iy=*")):
        frs = sorted(tdir.glob("exp_*.parquet"), key=lambda f: int(f.stem.split("_")[1]))
        if not frs:
            continue
        dst = Path(final) / "rows" / tdir.parent.name / tdir.name
        dst.mkdir(parents=True, exist_ok=True)
        out = dst / f"part-{job}.parquet"

        def w(tmp, frs=frs):
            with pq.ParquetWriter(tmp, ROW_SCHEMA, compression="zstd", compression_level=6, use_dictionary=plain,
                                  use_byte_stream_split=["data", "ivar", "x0", "y0", "lambda_um"]) as pw:
                for f in frs:
                    t = pq.read_table(f, schema=ROW_SCHEMA)
                    pw.write_table(t, row_group_size=max(t.num_rows, 1))
        _atomic_write(str(out), w)
        n_parts += 1
        for f in frs:
            exps.add(int(f.stem.split("_")[1]))
        n_rows += sum(pq.ParquetFile(f).metadata.num_rows for f in frs)
    (Path(final) / "exposure_meta").mkdir(parents=True, exist_ok=True)
    for m in sorted((Path(work) / "exposure_meta").glob("exp_*.json")):
        txt = m.read_text()
        _atomic_write(str(Path(final) / "exposure_meta" / m.name), lambda t, txt=txt: Path(t).write_text(txt))
    (Path(final) / "done").mkdir(parents=True, exist_ok=True)
    info = {"job": job, "part_files": n_parts, "exposures": len(exps), "rows": n_rows, "seconds": round(time.perf_counter() - t0, 2)}
    _atomic_write(str(Path(final) / "done" / f"job-{job}.json"), lambda t: Path(t).write_text(json.dumps(info)))
    return info


def cmd_extract(a):
    halos = {k.upper(): float(v) for k, v in (h.split("=") for h in (a.halo or []))}
    grid = Grid(a.tile_arcsec, a.box_arcsec, halos=halos, pos_quant_bits=a.pos_quant_bits)
    final = Path(a.store)
    compact = bool(a.compact or a.local_dir)
    if compact and not a.local_dir:
        raise SystemExit("--compact needs --local-dir (scratch)")
    k, n = parse_shard(a.shard)
    job = f"{a.mission}-{k}of{n}"
    check_store_config(final, grid)
    if compact:
        if (final / "done" / f"job-{job}.json").exists() and not a.force:
            log(f"job {job}: done-marker exists, skipping")
            return
        store = Path(a.local_dir) / job
        check_store_config(store, grid)
    else:
        store = final
    for sub in ("rows", "exposure_meta", "done"):
        (store / sub).mkdir(parents=True, exist_ok=True)
    files = discover(a.mission, a.inputs, a.data_dir or [])
    if a.shard:
        files = files[k::n]
    if a.limit:
        files = files[: a.limit]
    log(f"{a.mission}: {len(files)} input files (shard {a.shard or '-'}){'; compacting to ' + str(final) if compact else ''}")
    opts = {"mask_policy": a.mask_policy, "hst_wcs": a.hst_wcs, "hst_dq_policy": a.hst_dq_policy}
    reader = READERS[a.mission]
    tot = {"files": 0, "skipped_done": 0, "units": 0, "rows": 0, "fragments": 0}
    for f in files:
        fid = file_id(f)
        marker = store / "done" / f"{fid}.json"
        if marker.exists() and not a.force:
            tot["skipped_done"] += 1
            continue
        t0 = time.perf_counter()
        try:
            units = reader(f, grid, opts)
        except Exception as exc:  # noqa: BLE001
            log(f"FAILED {os.path.basename(f)}: {type(exc).__name__}: {exc}")
            if a.strict:
                raise
            continue
        t_read = time.perf_counter() - t0
        recs = []
        for u in units:
            rec, n_rows, n_tiles, n_w, t_w = write_unit(store, grid, u, force=a.force)
            rec["t_extract_s"] = round(time.perf_counter() - t0, 3) if len(units) == 1 else round(t_read / len(units) + t_w, 3)
            _atomic_write(str(store / "exposure_meta" / f"exp_{rec['exposure_id']}.json"), lambda t, r=rec: Path(t).write_text(json.dumps(r)))
            recs.append({"exposure_id": rec["exposure_id"], "name": rec["name"], "n_rows": n_rows, "n_tiles": n_tiles})
            tot["units"] += 1
            tot["rows"] += n_rows
            tot["fragments"] += n_w
        dt = time.perf_counter() - t0
        _atomic_write(str(marker), lambda t: Path(t).write_text(json.dumps({"file": f, "units": recs, "seconds": round(dt, 3)})))
        tot["files"] += 1
        log(f"{os.path.basename(f)}: {len(units)} unit(s), {sum(r['n_rows'] for r in recs)} rows, {dt:.1f}s")
    log(f"done {tot}")
    if compact:
        info = compact_job(store, final, job, grid)
        log(f"compacted job {job}: {info}")
        if not a.keep_local:
            shutil.rmtree(store, ignore_errors=True)
    elif not a.no_finalize:
        cmd_finalize(argparse.Namespace(store=a.store))


# =============================================================================================== finalize / summary
def cmd_finalize(a):
    store = Path(a.store)
    recs = [json.loads(p.read_text()) for p in sorted((store / "exposure_meta").glob("exp_*.json"))]
    if not recs:
        log("no exposure records")
        return
    cols = {n: [r.get(n) for r in recs] for n, _ in EXPOSURE_COLUMNS}
    table = pa.table({n: pa.array(cols[n], type=t) for n, t in EXPOSURE_COLUMNS})
    _atomic_write(str(store / "exposures.parquet"), lambda t: pq.write_table(table, t, compression="zstd"))
    cfg = json.loads((store / "store.json").read_text())
    out = {"store": cfg, "detectors": {str(i): n for i, n in enumerate(DETECTORS)}, "n_exposures": len(recs),
           "exposures": [{**{k: v for k, v in r.items() if k not in ("header_json", "extra_json")},
                          "header": json.loads(r["header_json"]), "extra": json.loads(r["extra_json"])} for r in recs]}
    _atomic_write(str(store / "exposures.json"), lambda t: Path(t).write_text(json.dumps(out)))
    log(f"exposures.parquet/json: {len(recs)} exposures")


def read_exposures(store):
    p = Path(store) / "exposures.parquet"
    if not p.exists():
        cmd_finalize(argparse.Namespace(store=str(store)))
    return pq.read_table(p).to_pylist()


def cmd_summary(a):
    store = Path(a.store)
    ex = {r["exposure_id"]: r for r in read_exposures(store)}
    per_tile = {}
    inst_tot, mission_tot = {}, {}
    nbytes_tot = 0
    nfiles = len(list((store / "rows").glob("tile_ix=*/tile_iy=*/*.parquet")))
    for pc in scan_pieces(store):                       # both layouts: fragments and part files (one row group per exposure)
        ix, iy, eid, n, sz = pc["ix"], pc["iy"], pc["eid"], pc["rows"], pc["bytes"]
        r = ex.get(eid)
        inst = r["instrument"] if r else "?"
        d = per_tile.setdefault((ix, iy), {"rows": {}, "exposures": set(), "bytes": 0})
        d["rows"][inst] = d["rows"].get(inst, 0) + n
        d["exposures"].add(eid)
        d["bytes"] += sz
        t = inst_tot.setdefault(inst, {"rows": 0, "bytes": 0, "fragments": 0})
        t["rows"] += n
        t["bytes"] += sz
        t["fragments"] += 1
        nbytes_tot += sz
    insts = sorted(inst_tot)
    print(f"store {store}: {len(ex)} exposures, {len(per_tile)} tiles, {sum(t['rows'] for t in inst_tot.values())} rows, "
          f"{nbytes_tot / 1e6:.1f} MB in {nfiles} files")
    print("per instrument:")
    for i in insts:
        t = inst_tot[i]
        print(f"  {i:8s} rows {t['rows']:>13,d}  pieces {t['fragments']:>6d}  {t['bytes'] / 1e6:10.2f} MB  {t['bytes'] / max(t['rows'], 1):6.2f} B/row")
    print("per tile  (rows by instrument | #exposures | MB):")
    print("  tile      " + " ".join(f"{i:>11s}" for i in insts) + "   n_exp       MB")
    for (ix, iy), d in sorted(per_tile.items()):
        print(f"  ({ix:2d},{iy:2d})  " + " ".join(f"{d['rows'].get(i, 0):>11,d}" for i in insts) + f"  {len(d['exposures']):6d}  {d['bytes'] / 1e6:7.2f}")
    if a.json:
        out = {"per_instrument": inst_tot, "per_tile": {f"{k[0]},{k[1]}": {"rows": v["rows"], "n_exposures": len(v["exposures"]), "bytes": v["bytes"]}
                                                          for k, v in per_tile.items()}, "bytes_total": nbytes_tot}
        Path(a.json).write_text(json.dumps(out, indent=1))
    return per_tile, inst_tot


# =============================================================================================== reader
def load_store_config(store):
    return json.loads((Path(store) / "store.json").read_text())


def tile_header(store, ix, iy, scale_arcsec=0.04):
    """TAN header (v31 ``target_header`` keys) of tile (ix, iy) at ``scale_arcsec`` whose pixel grid IS the tile grid of ``load_tile``
    (tx, ty): tangent point = tile centre, CD = Jacobian of the global plane at the tile centre. The CD matrix is rotated by up to
    ~0.2 deg with respect to the north-up v31 target_header (meridian convergence between the tile and the global tangent point,
    4 arcmin away at Dec -73.5); the residual between this header and the global plane is second order (< 0.01 mas within the tile and 3 arcsec around it, checked on tiles (0,0), (7,7), (15,3))."""
    cfg = load_store_config(store)
    g = Grid(cfg["tile_arcsec"], cfg["box_arcsec"], cfg["center_ra"], cfg["center_dec"], cfg["halo_arcsec"], cfg.get("pos_quant_bits", 0))
    x0, x1, y0, y1 = g.tile_extent(ix, iy)
    xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    ra, dec = g.plane_to_world(xc, yc)
    loc = WCS(naxis=2)
    loc.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    loc.wcs.crval = [float(ra), float(dec)]
    loc.wcs.crpix = [1.0, 1.0]
    loc.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    loc.wcs.set()
    h = 1.0
    pts = [(xc + h, yc), (xc, yc + h)]
    ra_p, dec_p = g.plane_to_world([p[0] for p in pts], [p[1] for p in pts])
    xl, yl = loc.all_world2pix(ra_p, dec_p, 0)           # local west/north arcsec of +1 arcsec along global x, y
    xl, yl = np.asarray(xl) / h, np.asarray(yl) / h
    size = int(round(g.T / scale_arcsec))
    s = scale_arcsec / 3600.0
    return {"NAXIS": 2, "NAXIS1": size, "NAXIS2": size, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CUNIT1": "deg", "CUNIT2": "deg",
            "CRVAL1": float(ra), "CRVAL2": float(dec), "CRPIX1": size / 2 + 0.5, "CRPIX2": size / 2 + 0.5,
            "CD1_1": float(-xl[0] * s), "CD1_2": float(-xl[1] * s), "CD2_1": float(yl[0] * s), "CD2_2": float(yl[1] * s)}


def load_tile(store, ix, iy, missions=None, bands=None, device="cpu", scale_arcsec=0.04, core_only=False, v31_bands=True, size=None):
    """Exposure dicts for tile (ix, iy) (core + stored halo), compatible with smacs_four_v31 exposure dicts.

    tx, ty : pixel coordinates (float32, pixel centres at integers) in the TILE's target grid of ``scale_arcsec`` (default 0.04"):
             tx = (x0 - xmin)/scale - 0.5 with xmin the tile core edge on the global plane; tx grows westward as in v31
             target_header (CD1_1 < 0), ty northward. Pixels in the halo have tx/ty outside [0, size).
    d, ivar, row, col (int64), dx, dy ((col/nx)*2-1 over the detector array), bkg0 (median d), noise_scale = 1.0 (not calibrated here),
    path, header (WCS subset dict), cut (y0, y1, x0, x1 of the stored rows), detname, name, band, plus instrument/filter/mjd/exptime/
    pixar_sr_nominal/units_note, dq (int32), in_core (bool), x0, y0 (float64 global plane), lam_um (SPHEREx), exposure_id, mission, and
    for HST photplam/photflam/photfnu_jy/to_mjysr/svo_id/dq_good_bits.
    missions : subset of {"jwst","hst","spitzer","spherex","galex"}; bands : namespaced store bands (e.g. "NIRCAM_F150W") or, with
    v31_bands, the v31 spelling ("F150W"). v31_bands=True gives NIRCam exposures the plain band name of v31 (F150W) and no
    ``instrument`` key (what smacs_four_v31.registry_from_exposures expects); everything else keeps the store band name.
    Not provided (not stored): area_ratio, svo/PSF registries, holdout flags (set later by select_exposures)."""
    import torch
    store = Path(store)
    cfg = load_store_config(store)
    T = cfg["tile_arcsec"]
    xmin, _, ymin, _ = Grid(T, cfg["box_arcsec"]).tile_extent(ix, iy)
    size = size or int(round(T / scale_arcsec))
    ex = {r["exposure_id"]: r for r in read_exposures(store)}
    tdir = store / "rows" / f"tile_ix={ix}" / f"tile_iy={iy}"
    out = []
    want_b = None
    if bands:
        want_b = set(bands) | {f"NIRCAM_{b}" for b in bands}
    for pc in pieces_of(tdir):                           # fragments and/or part files
        eid = pc["eid"]
        r = ex[eid]
        if missions and r["mission"] not in missions:
            continue
        if want_b is not None and r["band"] not in want_b:
            continue
        t = read_piece(pc)
        if core_only:
            t = t.filter(t["in_core"])
        if t.num_rows == 0:
            continue
        x0, y0 = t["x0"].to_numpy(), t["y0"].to_numpy()
        row, col = t["row"].to_numpy().astype(np.int64), t["col"].to_numpy().astype(np.int64)
        d = t["data"].to_numpy()
        hdr = json.loads(r["header_json"])
        extra = json.loads(r["extra_json"])
        nx, ny = r["nx"], r["ny"]
        is_nircam = r["instrument"] == "NIRCam"
        band = r["band"][len("NIRCAM_"):] if (v31_bands and is_nircam) else r["band"]
        tx = ((x0 - xmin) / scale_arcsec - 0.5).astype(np.float32)
        ty = ((y0 - ymin) / scale_arcsec - 0.5).astype(np.float32)
        tt = lambda a, dt=None: torch.from_numpy(np.ascontiguousarray(a if dt is None else a.astype(dt))).to(device)  # noqa: E731
        e = {
            "name": r["name"], "band": band, "tx": tt(tx), "ty": tt(ty), "d": tt(d), "ivar": tt(t["ivar"].to_numpy()),
            "row": tt(row), "col": tt(col),
            "dx": tt(((col / nx) * 2 - 1), np.float32), "dy": tt(((row / ny) * 2 - 1), np.float32),
            "bkg0": float(np.median(d)), "path": r["path"], "header": hdr,
            "cut": (int(row.min()), int(row.max()) + 1, int(col.min()), int(col.max()) + 1), "noise_scale": 1.0, "detname": r["detector"],
            "filter": r["filter"], "mjd": r["mjd"], "exptime": r["exptime"], "pixar_sr_nominal": r["pixar_sr_nominal"],
            "units_note": f"{r['units']}; {r['unit_conversion']}", "dq": tt(t["dq"].to_numpy()), "in_core": tt(t["in_core"].to_numpy()),
            "x0": tt(x0), "y0": tt(y0), "lam_um": (tt(t["lambda_um"].to_numpy(zero_copy_only=False)) if r["mission"] == "spherex" else None),
            "exposure_id": eid, "mission": r["mission"], "det_id": r["det_id"], "store_band": r["band"], "instrument_name": r["instrument"],
            "tile": (ix, iy), "tile_size_px": size,
        }
        if not (v31_bands and is_nircam):
            e["instrument"] = r["instrument"]
        if r["mission"] == "hst":
            e.update({"photplam": extra.get("photplam_A"), "photflam": extra.get("photflam"), "photfnu_jy": extra.get("photfnu_jy"),
                      "to_mjysr": extra.get("to_mjysr"), "svo_id": extra.get("svo_id"), "ext": extra.get("ext"),
                      "dq_good_bits": extra.get("dq_good_bits")})
        out.append(e)
    out.sort(key=lambda e: (e.get("photplam") or 0.0, e["band"], e["name"]))
    return out


# =============================================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--store", required=True)
    e.add_argument("--inputs", action="append", required=True, help="glob or manifest CSV (repeatable)")
    e.add_argument("--mission", required=True, choices=sorted(READERS))
    e.add_argument("--data-dir", action="append", help="where manifest file names are searched (repeatable)")
    e.add_argument("--shard", help="k/N, 0 <= k < N: every N-th input file starting at k")
    e.add_argument("--limit", type=int)
    e.add_argument("--tile-arcsec", type=float, default=TILE_ARCSEC)
    e.add_argument("--box-arcsec", type=float, default=BOX_ARCSEC)
    e.add_argument("--pos-quant-bits", type=int, default=POS_QUANT_BITS, help="round x0,y0 to 2**-bits arcsec (0 = exact float64)")
    e.add_argument("--halo", action="append", help="FAMILY=arcsec override, e.g. IRAC=12 (family: NIRCAM NIRISS HST MIRI IRAC GALEX SPHEREX)")
    e.add_argument("--mask-policy", choices=["default", "none"], default="default", help="none: keep masked pixels with ivar = 0")
    e.add_argument("--hst-wcs", choices=["apriori", "archive"], default="apriori")
    e.add_argument("--hst-dq-policy", default="default", help="default | strict | integer good-bit sum")
    e.add_argument("--force", action="store_true", help="re-extract and overwrite existing fragments")
    e.add_argument("--strict", action="store_true", help="stop at the first failing file")
    e.add_argument("--no-finalize", action="store_true")
    e.add_argument("--local-dir", help="scratch dir: extract fragments there, then compact into one part file per tile in --store")
    e.add_argument("--compact", action="store_true", help="compaction mode (implied by --local-dir)")
    e.add_argument("--keep-local", action="store_true", help="keep the scratch fragments after compaction")
    f = sp.add_parser("finalize")
    f.add_argument("--store", required=True)
    s = sp.add_parser("summary")
    s.add_argument("--store", required=True)
    s.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        if a.hst_dq_policy.lstrip("-").isdigit():
            a.hst_dq_policy = int(a.hst_dq_policy)
        cmd_extract(a)
    elif a.cmd == "finalize":
        cmd_finalize(a)
    else:
        cmd_summary(a)


if __name__ == "__main__":
    main()
