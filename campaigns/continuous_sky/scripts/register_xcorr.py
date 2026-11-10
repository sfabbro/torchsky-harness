#!/usr/bin/env python3
"""register_xcorr.py - catalogue-free relative registration of JWST NIRCam `_cal` exposures by image cross-correlation, joint over all tiles of a tilefit plan.

WHY: the A12 coarse registration of tilefit.py (likelihood grid scan against a coadd made of the same, misregistered, exposures) is nearly flat in a crowded stellar field. Here
the exposures are registered against EACH OTHER: pairwise image offsets (masked normalised cross-correlation of high-passed, resampled images) feed one robust least-squares
solve for one offset per exposure. No source detection, no catalogue matching, no tweakreg, no Gaia: only pixels.

CONVENTION (identical to tilefit.py A12 / coarse_shifts.json): shift_mas = [dx, dy] in mas along the axes of the tile pixel grid (north-up TAN, x increases to the WEST because
CD1_1 < 0, y increases to the north; the same orientation in every tile), the shift to ADD to the header-WCS pixel position of the exposure: true_position = header_position +
shift. A pair (i, j) measured by correlation gives d_ij = (position of a feature in image j) - (position in image i) = s_i - s_j, which is the model of every edge.
Only RELATIVE shifts are measurable: the common offset of all exposures (or of the gauge set) is fixed by --gauge (default: mean over all detector images = 0), the absolute
frame is NOT recovered (no catalogue). tilefit's A12 has its own (cross-band reference) gauge, so compare with --compare after removing the mean difference.

PIPELINE (one task = tile x band, run in parallel with multiprocessing)
 1 plan: tiles (centre, size), per-band exposure records (path, detector, covering); same parsing as tilefit.py (tile files = every overlapping file, or --tile-use covering).
 2 resample each exposure's SCI onto the tile's north-up TAN grid (tile box + radius + 0.5 arcsec margin) at the DETECTOR scale of its channel (0.031 SW, 0.063 LW arcsec/px),
   with the header WCS (SCI-header FITS WCS incl. SIP = what tilefit/the engine uses; --wcs gwcs reads the gwcs of the ASDF extension with asdf+gwcs). The tile->detector
   coordinate map is evaluated on a 16-px lattice, cubic-spline interpolated to the full grid; the image is sampled bilinearly. DQ != 0 and non-finite pixels are masked
   (a resampled pixel is valid only if all 4 neighbours are valid). Only the footprint crop of each file is read.
 3 high-pass: subtract a masked block-median background (0.6 arcsec blocks, smoothed, bilinearly interpolated), asinh stretch (or clip / rank), masked Gaussian smoothing.
 4 pairwise offsets: masked normalised cross-correlation (Padfield) by FFT within +-radius (default 1.5 arcsec); sub-pixel peak by an upsampled (matrix) DFT of the
   cross-power spectrum (0.05 px grid, +-1 px) + parabolic vertex; per pair: peak height, second-peak ratio (peak / highest other local maximum outside a small exclusion
   disk), overlap pixels, peak curvature -> sigma.
 5 pair graph per tile and band: each exposure is paired with up to N others of maximum overlap: quota of other-programme, other-visit and within-visit partners; bridging edges
   if the graph is disconnected. Bands are linked by the SW/LW simultaneity: the default unknown is one translation per DITHER GROUP (programme, observation, visit, activity,
   exposure number = all detectors, both modules, SW and LW), so a SW image and its LW partner share the unknown (exact pointing constraint, no cross-band correlation).
   With --unit detector every detector image is its own unknown and SW/LW (same module) are tied by a soft constraint (--tie-sigma-mas). Optional cross-band correlation
   of adjacent bands (--xband; F150W:F200W, F277W:F356W, plus F356W:F444W because the F444W-only dither groups of the LMC plan have no SW partner in it) adds extra edges.
   WITHOUT them the graph of the LMC plan has 3 components (F090W+F200W+F356W, F150W+F277W, F444W): the simultaneity links only the band pairs observed together, and the
   relative offset between components is not measured (each has its own gauge; flagged in the summary). With the two user-listed pairs: 2 components; with F356W:F444W: 1.
 6 solve: one joint weighted least squares over all tiles (an exposure in several tiles is one unknown), IRLS with Huber weights and hard rejection, global sigma inflation from
   the median edge residual, gauge by projection. Reports uncertainties, closure residuals, rejected fractions, per-programme / per-visit / per-band summaries and the
   agreement of the detectors inside a dither group.
 7 optional --similarity (needs --unit detector): per dither group, fit translation + scale + rotation to the per-detector offsets at the detectors' sky positions.

OUTPUT (--out): register_xcorr.json (exposures{name: {shift_mas, sigma_mas, n_edges, ...}}, summary, config, approximations), register_xcorr_edges.json (every measured edge with
its residual), register_xcorr_summary.txt, work/edges_t<tile>_<band>.json (per-task cache for --resume).

APPROXIMATIONS (all repeated in register_xcorr.json["approximations"])
 R1 shifts are translations in mas along the tile axes; the same vector is used in every tile (the tile planes differ by < ~2 mrad in orientation over the LMC plan; same as A5/A12).
 R2 a detector image has ONE shift for all tiles it appears in (default and --unit detector); a scale / rotation error inside the image shows up as tile-to-tile residuals
    (by_tile, spread_mas), only --similarity models it, per dither group.
 R3 bilinear resampling of the (undersampled) SW images; bilinear interpolation preserves centroids but smooths by a phase dependent amount.
 R4 background = block median + interpolation (not a sliding median filter).
 R5 edge sigma = sqrt(4 pi (1 - rho^2)) / (rho sqrt(N) h) from the correlation peak (rho, N overlap pixels, curvature h in px^-2) with a floor (--sigma-floor-mas); the
    absolute level is calibrated by the median edge residual (global inflation factor, reported). Edge errors are treated as independent (they are not: shared images, shared
    systematic structures).
 R6 gauge: the mean over the gauge set is zero; no absolute astrometry. Uncertainties are relative to that gauge.
 R7 a spurious peak from detector-frame artefacts (hot pixels, persistence) that survived DQ masking would sit at the dither offset; only the second-peak ratio and the robust
    solve guard against it.
 R8 the correlation measures the mean offset over the overlap (24 arcsec box); a proper-motion field or an exposure-time dependent offset (jitter) is not modelled.
 R9 offsets larger than the search radius (default 1.5 arcsec) between a pair are not measurable; with offsets of the order of 1 arcsec, graph neighbours across programmes
    can fall outside the window and the edge is then flagged (peak at the window boundary).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import collections
import glob
import io
import json
import math
import multiprocessing as mp
import re
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp
from scipy import ndimage as ndi
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.csgraph import connected_components
from scipy.linalg import pinvh
from scipy.special import ndtri

from astropy.wcs import FITSFixedWarning  # noqa: E402

warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*node failed to convert.*")

SCALE_SW, SCALE_LW = 0.031, 0.063    # nominal NIRCam detector pixel scales [arcsec]
T0 = time.perf_counter()

DEFAULTS = dict(
    tile_use="all", data_root=None, wcs="fits", workers=8, radius_arcsec=1.5, margin_arcsec=0.5,
    pairs_per_exposure=8, n_other_programme=3, n_other_visit=3, min_overlap_arcsec2=30.0,
    bg_block_arcsec=0.6, smooth_arcsec=0.03, stretch="asinh", asinh_soft=3.0, clip_sigma=8.0,
    min_peak=0.05, min_ratio=1.5, excl_arcsec=0.1, sigma_floor_mas=1.0,
    unit="group", similarity=False, tie_sigma_mas=3.0, tie_cross_module_sigma_mas=0.0,
    xband=False, xband_pairs="F150W:F200W,F277W:F356W,F356W:F444W", n_xband=2,
    gauge="all", huber=1.5, reject=4.0, irls_iters=25, resume=False, max_exposures_per_task=0, lattice_px=16,
)


def log(msg):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- plan, names
_NAME_RE = re.compile(r"^(?:(?P<idx>\d+)_)?jw(?P<prog>\d{5})(?P<obs>\d{3})(?P<vis>\d{3})_(?P<grp>[0-9a-z]{5})_(?P<exp>\d{5})_(?P<det>[a-z0-9]+)(?:_|\.|$)", re.IGNORECASE)


def parse_name(name):
    """File name -> dict (programme, observation+visit, activity group, exposure, detector, module, channel and the keys used here), or None."""
    m = _NAME_RE.match(os.path.basename(str(name)))
    if m is None:
        return None
    g = m.groupdict()
    det = g["det"].lower()
    return dict(prog=g["prog"], obs=g["obs"], vis=g["vis"], grp=g["grp"], exp=g["exp"], det=det,
                module=det[3].upper() if det.startswith("nrc") and len(det) > 3 else "?", channel="LW" if det.endswith("long") else "SW",
                dither=f"{g['prog']}{g['obs']}{g['vis']}_{g['grp']}_{g['exp']}", visit_id=f"{g['prog']}{g['obs']}{g['vis']}",
                visit=f"{g['obs']}{g['vis']}-{g['grp']}", engine_dither=str(((g["obs"] + g["vis"], g["grp"]), g["exp"])))


def det_scale(detector):
    return SCALE_LW if str(detector).lower().endswith("long") else SCALE_SW


def resolve_path(path, data_root=None):
    if os.path.exists(path) or not data_root:
        return path
    b = os.path.basename(path)
    for c in [os.path.join(data_root, b)] + sorted(glob.glob(os.path.join(data_root, "*", b))):
        if os.path.exists(c):
            return c
    return path


def exposure_record(name, path, band, detector):
    p = parse_name(name) or {}
    prog = p.get("prog", "?")
    return dict(name=name, path=path, band=band, detector=str(detector).upper(), prog=prog, module=p.get("module", "?"), channel=p.get("channel", "LW" if det_scale(detector) == SCALE_LW else "SW"),
                dither=p.get("dither", name), visit_id=p.get("visit_id", name), visit=p.get("visit", "?"), engine_dither=p.get("engine_dither", str(name)))


def load_plan(path, tile_use="all", data_root=None, tiles=None, bands=None):
    """Same parsing as tilefit.py main(): plan["tiles"][k] with ra, dec, exposures{band: [records with file, path, detector, covering]}, files; plan["params"]["size_arcsec"].
    Returns (tiles, exps): tiles = list of dict(id, ra, dec, size_arcsec, offset_en, exposures{band: [names]}), exps = {name: record}."""
    plan = json.load(open(path))
    size = float(plan["params"]["size_arcsec"])
    out, exps = [], {}
    for rec in plan["tiles"]:
        if tiles and rec["id"] not in tiles:
            continue
        keep = None if tile_use != "all" else set(rec.get("files", []))
        t = dict(id=int(rec["id"]), ra=float(rec["ra"]), dec=float(rec["dec"]), size_arcsec=float(rec.get("size_arcsec", size)), offset_en=list(rec.get("offset_east_north_arcsec", [0.0, 0.0])), exposures={})
        for b in plan["bands"]:
            if bands and b not in bands:
                continue
            names = []
            for e in rec["exposures"][b]:
                if tile_use == "covering" and not e["covering"]:
                    continue
                if keep and e["path"] not in keep:
                    continue
                nm = os.path.basename(e["path"])
                if nm not in exps:
                    exps[nm] = exposure_record(nm, resolve_path(e["path"], data_root), b, e["detector"])
                names.append(nm)
            if names:
                t["exposures"][b] = sorted(set(names))
        out.append(t)
    return out, exps


# ----------------------------------------------------------------------------- WCS and resampling
class TileGrid:
    """North-up TAN grid of one tile at a given scale: x increases to the west, y to the north (CD = [[-s, 0], [0, s]], as the engine's target_header)."""

    def __init__(self, ra, dec, size_arcsec, scale, extra_arcsec):
        from astropy.wcs import WCS
        self.scale = float(scale)
        self.n = int(math.ceil((size_arcsec + 2 * extra_arcsec) / scale))
        s = scale / 3600.0
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [ra, dec]
        w.wcs.crpix = [(self.n + 1) / 2.0, (self.n + 1) / 2.0]
        w.wcs.cd = [[-s, 0.0], [0.0, s]]
        self.wcs = w

    def world(self, x, y):
        return self.wcs.all_pix2world(x, y, 0)


class ExpWCS:
    """World -> detector pixel (0-based) of one `_cal` file: FITS WCS of the SCI header (astropy, SIP included) or the gwcs of the ASDF extension."""

    def __init__(self, hdul, mode="fits"):
        h = hdul["SCI"].header
        self.nx, self.ny = int(h["NAXIS1"]), int(h["NAXIS2"])
        self.mode = mode
        if mode == "gwcs":
            import asdf
            raw = hdul["ASDF"].data["ASDF_METADATA"].tobytes()
            with asdf.open(io.BytesIO(raw), lazy_load=False) as af:
                self.g = af.tree["meta"]["wcs"]
        else:
            from astropy.wcs import WCS
            self.w = WCS(h, relax=True)

    def world2pix(self, ra, dec):
        if self.mode == "gwcs":
            x, y = self.g.invert(ra, dec, with_bounding_box=False)
        else:
            x, y = self.w.all_world2pix(ra, dec, 0, quiet=True)
        x, y = np.asarray(x, float), np.asarray(y, float)
        bad = ~(np.isfinite(x) & np.isfinite(y))
        x[bad], y[bad] = -1e6, -1e6
        return np.clip(x, -1e6, 1e6), np.clip(y, -1e6, 1e6)


def _crop(hdul, ext, y0, y1, x0, x1):
    hdu = hdul[ext]
    raw = np.asarray(hdu.data[y0:y1, x0:x1])
    bz, bs = hdu.header.get("BZERO", 0), hdu.header.get("BSCALE", 1)
    return raw if (bz == 0 and bs == 1) else raw.astype(np.float64) * bs + bz


def resample_exposure(path, grid, wcs_mode="fits", lattice_px=16):
    """SCI of one `_cal` file on the tile grid. Returns (img float32, valid bool) both (n, n), or None if the footprint does not cover >= 1000 grid pixels."""
    from astropy.io import fits
    n = grid.n
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as h:
        ew = ExpWCS(h, wcs_mode)
        idx = np.unique(np.r_[np.arange(0, n, lattice_px), n - 1])
        gx, gy = np.meshgrid(idx, idx)
        ra, dec = grid.world(gx.ravel().astype(float), gy.ravel().astype(float))
        px, py = ew.world2pix(ra, dec)
        px, py = px.reshape(gx.shape), py.reshape(gx.shape)
        inside = (px > -lattice_px) & (px < ew.nx + lattice_px) & (py > -lattice_px) & (py < ew.ny + lattice_px)
        if not inside.any():
            return None
        full = np.arange(n, dtype=float)
        X = RectBivariateSpline(idx.astype(float), idx.astype(float), px, kx=3, ky=3)(full, full)
        Y = RectBivariateSpline(idx.astype(float), idx.astype(float), py, kx=3, ky=3)(full, full)
        sel = (X > -1) & (X < ew.nx) & (Y > -1) & (Y < ew.ny)
        if int(sel.sum()) < 1000:
            return None
        x0, x1 = max(int(math.floor(X[sel].min())) - 1, 0), min(int(math.ceil(X[sel].max())) + 2, ew.nx)
        y0, y1 = max(int(math.floor(Y[sel].min())) - 1, 0), min(int(math.ceil(Y[sel].max())) + 2, ew.ny)
        sci = _crop(h, "SCI", y0, y1, x0, x1).astype(np.float32)
        dq = _crop(h, "DQ", y0, y1, x0, x1)
    bad = (dq != 0) | ~np.isfinite(sci)
    S = np.where(bad, 0.0, sci).astype(np.float32)
    v = (~bad).astype(np.float32)
    co = [Y - y0, X - x0]
    img = ndi.map_coordinates(S, co, order=1, mode="constant", cval=0.0)
    wgt = ndi.map_coordinates(v, co, order=1, mode="constant", cval=0.0)
    valid = (wgt > 0.999) & sel
    img = np.where(valid, img / np.maximum(wgt, 1e-6), 0.0).astype(np.float32)
    return img, valid


# ----------------------------------------------------------------------------- high-pass
def block_median_background(img, valid, blk, min_frac=0.25):
    n = img.shape[0]
    nb = int(math.ceil(n / blk))
    a = np.full((nb * blk, nb * blk), np.nan, np.float32)
    a[:n, :n] = np.where(valid, img, np.nan)
    b = a.reshape(nb, blk, nb, blk).transpose(0, 2, 1, 3).reshape(nb, nb, blk * blk)
    cnt = np.isfinite(b).sum(-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(b, axis=-1)
    med[cnt < min_frac * blk * blk] = np.nan
    if np.isnan(med).all():
        return np.zeros_like(img)
    ii = ndi.distance_transform_edt(np.isnan(med), return_distances=False, return_indices=True)
    med = ndi.gaussian_filter(med[tuple(ii)], 1.0, mode="nearest")
    c = (np.arange(n) + 0.5) / blk - 0.5
    return ndi.map_coordinates(med, np.meshgrid(c, c, indexing="ij"), order=1, mode="nearest").astype(np.float32)


def preprocess(img, valid, scale, cfg):
    """High-pass + stretch + masked smoothing. Returns (y float32 zero outside the mask, valid) or None if fewer than 2000 valid pixels."""
    if int(valid.sum()) < 2000:
        return None
    blk = max(int(round(cfg["bg_block_arcsec"] / scale)), 6)
    x = np.where(valid, img - block_median_background(img, valid, blk), 0.0)
    xv = x[valid]
    sig = 1.4826 * float(np.median(np.abs(xv - np.median(xv))))
    if not sig > 0:
        sig = float(xv.std())
    if not sig > 0:
        return None
    st = cfg["stretch"]
    if st == "asinh":
        y = np.arcsinh(x / (cfg["asinh_soft"] * sig))
    elif st == "clip":
        y = np.clip(x / sig, -cfg["clip_sigma"], cfg["clip_sigma"])
    elif st == "rank":
        r = np.empty(xv.size)
        r[np.argsort(xv, kind="stable")] = np.arange(xv.size)
        y = np.zeros_like(x)
        y[valid] = ndtri((r + 0.5) / xv.size)
    else:
        raise ValueError(f"unknown stretch {st}")
    y = np.where(valid, y, 0.0)
    s = cfg["smooth_arcsec"] / scale
    if s > 0.05:
        num = ndi.gaussian_filter(y, s)
        den = ndi.gaussian_filter(valid.astype(np.float64), s)
        y = np.where(valid, num / np.maximum(den, 1e-3), 0.0)
    y = np.where(valid, y - y[valid].mean(), 0.0)
    return y.astype(np.float32), valid


# ----------------------------------------------------------------------------- masked normalised cross-correlation
class Correlator:
    """FFT machinery for grids of n x n pixels and a search radius of R pixels (lags up to L = R + 2 are computed, circular wrap-around cannot reach them: nf >= n + L)."""

    def __init__(self, n, scale, cfg):
        self.n, self.scale, self.cfg = n, scale, cfg
        self.R = int(math.ceil(cfg["radius_arcsec"] / scale))
        self.L = self.R + 2
        self.nf = sfft.next_fast_len(n + self.L, real=True)
        nf = self.nf
        self.idx = np.arange(-self.L, self.L + 1) % nf
        yy, xx = np.mgrid[-self.L:self.L + 1, -self.L:self.L + 1]
        self.yy, self.xx, self.rr = yy, xx, np.hypot(yy, xx)
        self.fy, self.fx = np.fft.fftfreq(nf), np.fft.rfftfreq(nf)
        self.wx = np.full(self.fx.size, 2.0)
        self.wx[0] = 1.0
        if nf % 2 == 0:
            self.wx[-1] = 1.0
        self.d = np.arange(-1.0, 1.0 + 1e-9, 0.05)
        self.n_min = cfg["min_overlap_arcsec2"] / scale ** 2
        self.r_ex = max(3.0, cfg["excl_arcsec"] / scale)

    def prep(self, y, valid):
        s = (self.nf, self.nf)
        M = valid.astype(np.float64)
        yM = y.astype(np.float64) * M
        return (sfft.rfft2(yM, s), sfft.rfft2(yM * y, s), sfft.rfft2(M, s))

    def _cc(self, X, Y):
        c = sfft.irfft2(np.conj(X) * Y, s=(self.nf, self.nf))
        return c[np.ix_(self.idx, self.idx)]

    @staticmethod
    def _ncc(Sab, Sa, Sb, Saa, Sbb, N, n_min):
        with np.errstate(all="ignore"):
            va, vb = Saa - Sa * Sa / N, Sbb - Sb * Sb / N
            r = (Sab - Sa * Sb / N) / np.sqrt(va * vb)
        ok = (N >= n_min) & (va > 1e-9 * np.maximum(N, 1)) & (vb > 1e-9 * np.maximum(N, 1)) & np.isfinite(r)
        return np.where(ok, r, np.nan), ok

    def measure(self, fa, fb):
        """Offset d = (position in b) - (position in a) in pixels (x = columns, y = rows) of the images prepared with prep(), and the reliability metrics."""
        A1, A2, A0 = fa
        B1, B2, B0 = fb
        w = dict(N=self._cc(A0, B0), Sa=self._cc(A1, B0), Sb=self._cc(A0, B1), Saa=self._cc(A2, B0), Sbb=self._cc(A0, B2), Sab=self._cc(A1, B1))
        ncc, ok = self._ncc(w["Sab"], w["Sa"], w["Sb"], w["Saa"], w["Sbb"], w["N"], self.n_min)
        search = (self.rr <= self.R) & ok
        if not search.any():
            return dict(ok=False, reason="no overlap", dx_px=float("nan"), dy_px=float("nan"), peak=float("nan"), second=float("nan"), ratio=0.0, n_overlap=0.0, sigma_px=float("inf"))
        nz = np.where(search, ncc, -np.inf)
        iy, ix = np.unravel_index(int(np.argmax(nz)), nz.shape)
        peak = float(nz[iy, ix])
        locmax = (ndi.maximum_filter(np.where(np.isfinite(ncc), ncc, -np.inf), size=5) == ncc) & search
        far = np.hypot(self.yy - self.yy[iy, ix], self.xx - self.xx[iy, ix]) > self.r_ex
        cand = ncc[locmax & far]
        second = float(cand.max()) if cand.size else 0.0
        ratio = 99.0 if second <= 0 else min(peak / second, 99.0)
        # sub-pixel: upsampled DFT of the numerator on a 0.05 px grid, denominators bilinear
        d, nf, L = self.d, self.nf, self.L
        ly, lx = iy - L, ix - L
        Ey = np.exp(2j * np.pi * np.outer(ly + d, self.fy))
        Ex = np.exp(2j * np.pi * np.outer(self.fx, lx + d)) * self.wx[:, None]

        def upsampled(X, Y):      # band-limited (DFT) interpolation of one correlation term on the 0.05 px grid; ALL six terms use the same interpolation, so the
            return np.real(Ey @ (np.conj(X) * Y) @ Ex) / (nf * nf)      # normalised correlation stays consistent (bilinear denominators gave NCC > 1 and biased peaks)
        up, _ = self._ncc(upsampled(A1, B1), upsampled(A1, B0), upsampled(A0, B1), upsampled(A2, B0), upsampled(A0, B2), upsampled(A0, B0), 0.5 * self.n_min)
        if not np.isfinite(up).any():
            return dict(ok=False, reason="no overlap", dx_px=float("nan"), dy_px=float("nan"), peak=peak, second=second, ratio=float(ratio), n_overlap=float(w["N"][iy, ix]), sigma_px=float("inf"))
        my, mx = np.unravel_index(int(np.nanargmax(up)), up.shape)

        def vertex(vals, k):
            k0 = int(min(max(k, 2), len(vals) - 3))
            xs = np.arange(-2, 3) * 0.05
            v = vals[k0 - 2:k0 + 3]
            if not np.isfinite(v).all():
                return d[k]
            c = np.polyfit(xs, v, 2)
            off = -c[1] / (2 * c[0]) if c[0] < 0 else 0.0
            return d[k0] + float(np.clip(off, -0.1, 0.1))
        ddx, ddy = vertex(up[my, :], mx), vertex(up[:, mx], my)
        rho = float(up[my, mx]) if np.isfinite(up[my, mx]) else peak
        h = []
        for line, k in ((up[my, :], mx), (up[:, mx], my)):
            kp, km = min(k + 10, len(line) - 1), max(k - 10, 0)
            step = 0.5 * (kp - km) * 0.05
            h.append(max((2 * rho - line[kp] - line[km]) / (max(rho, 1e-6) * step ** 2) if step > 0 else 0.0, 1e-3))
        hm = float(np.sqrt(h[0] * h[1])) if np.all(np.isfinite(h)) else 1e-3
        n_ov = float(w["N"][iy, ix])
        sig_px = math.sqrt(4 * math.pi * max(1 - rho ** 2, 1e-4)) / (max(rho, 1e-3) * math.sqrt(max(n_ov, 1.0)) * hm)
        edge = bool(np.hypot(ly, lx) >= self.R - 1)
        reason = None
        if peak < self.cfg["min_peak"]:
            reason = "low peak"
        elif ratio < self.cfg["min_ratio"]:
            reason = "ambiguous (second-peak ratio)"
        elif edge:
            reason = "peak at search boundary"
        return dict(ok=reason is None, reason=reason, dx_px=float(lx + ddx), dy_px=float(ly + ddy), peak=rho, second=second, ratio=float(ratio), n_overlap=n_ov, sigma_px=float(sig_px), curvature=hm)


def measure_images(img_a, valid_a, img_b, valid_b, scale, cfg=None):
    """Convenience (tests): preprocess two images already on a common grid and measure d = (feature position in b) - (in a) in mas."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    pa, pb = preprocess(img_a, valid_a, scale, cfg), preprocess(img_b, valid_b, scale, cfg)
    if pa is None or pb is None:
        return dict(ok=False, reason="empty")
    cor = Correlator(img_a.shape[0], scale, cfg)
    r = cor.measure(cor.prep(*pa), cor.prep(*pb))
    r["dx_mas"], r["dy_mas"] = r["dx_px"] * scale * 1000, r["dy_px"] * scale * 1000
    r["sigma_mas"] = max(r["sigma_px"] * scale * 1000, cfg["sigma_floor_mas"])
    return r


# ----------------------------------------------------------------------------- pair graph
def select_pairs(meta, ov, n_pairs, n_prog, n_visit, min_ov, allow_same_unit=False):
    """meta: list of dict(unit, prog, visit_id) aligned with ov (n, n) overlap pixels. Each exposure takes up to n_prog partners of another programme, n_visit of another visit of
    its programme, then within-visit dithers (other exposures of its visit) up to n_pairs, leftovers of any kind fill the rest; best overlap first.
    Returns {(i, j): category} with i < j, category in prog / visit / within as seen from the lower index owner."""
    n = len(meta)
    sel = {}
    for i in range(n):
        cands = sorted(((ov[i, j], j) for j in range(n) if j != i and ov[i, j] >= min_ov and (allow_same_unit or meta[j]["unit"] != meta[i]["unit"])), reverse=True)
        def cat(j):
            return "prog" if meta[j]["prog"] != meta[i]["prog"] else ("visit" if meta[j]["visit_id"] != meta[i]["visit_id"] else "within")
        buckets = {"prog": [], "visit": [], "within": []}
        for _, j in cands:
            buckets[cat(j)].append(j)
        pick = buckets["prog"][:n_prog] + buckets["visit"][:n_visit]
        pick += buckets["within"][:max(n_pairs - len(pick), 0)]
        if len(pick) < n_pairs:
            pick += [j for _, j in cands if j not in pick][:n_pairs - len(pick)]
        for j in pick:
            sel.setdefault((min(i, j), max(i, j)), cat(j))
    return sel


def bridge_components(meta, ov, sel, min_ov):
    """Add the best-overlap edge between disconnected unit components until the graph is connected (or no candidate remains)."""
    units = sorted({m["unit"] for m in meta})
    parent = {u: u for u in units}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    for (i, j) in sel:
        parent[find(meta[i]["unit"])] = find(meta[j]["unit"])
    n = len(meta)
    added = 0
    while len({find(u) for u in units}) > 1:
        best = None
        for i in range(n):
            for j in range(i + 1, n):
                if ov[i, j] >= min_ov and find(meta[i]["unit"]) != find(meta[j]["unit"]) and (best is None or ov[i, j] > best[0]):
                    best = (ov[i, j], i, j)
        if best is None:
            break
        _, i, j = best
        sel[(i, j)] = "bridge"
        parent[find(meta[i]["unit"])] = find(meta[j]["unit"])
        added += 1
    return added


def coverage_overlap(valids, ds=8):
    """Pairwise overlap in grid pixels from masks downsampled by ds (cheap)."""
    n = valids[0].shape[0]
    m = (n // ds) * ds
    A = np.stack([v[:m, :m].reshape(m // ds, ds, m // ds, ds).mean((1, 3)).ravel() for v in valids]).astype(np.float32)
    return (A @ A.T) * ds * ds


# ----------------------------------------------------------------------------- one task: tile x band
def unit_key(rec, cfg):
    return rec["dither"] if cfg["unit"] == "group" else rec["name"]


def run_task(task):
    """tile x band: resample + high-pass the exposures, choose the pairs, correlate. Returns dict(edges, stats, seconds, ...)."""
    tile, band, recs, xrecs, cfg = task
    t0 = time.perf_counter()
    cache = Path(cfg["out"]) / "work" / f"edges_t{tile['id']}_{band}.json"
    if cfg.get("resume") and cache.exists():
        try:
            res = json.load(open(cache))
            if res.get("sig") == cfg["sig"]:
                res["cached"] = True
                return res
        except Exception:  # noqa: BLE001
            pass
    scale = det_scale(recs[0]["detector"])
    grid = TileGrid(tile["ra"], tile["dec"], tile["size_arcsec"], scale, cfg["radius_arcsec"] + cfg["margin_arcsec"])
    cor = Correlator(grid.n, scale, cfg)
    if cfg["max_exposures_per_task"] > 0:
        recs = recs[:cfg["max_exposures_per_task"]]
        xrecs = xrecs[:cfg["max_exposures_per_task"]]
    stats, skipped, items = [], [], []
    for kind, lst in (("own", recs), ("x", xrecs)):
        for r in lst:
            try:
                rs = resample_exposure(r["path"], grid, cfg["wcs"], cfg["lattice_px"])
                pp = None if rs is None else preprocess(rs[0], rs[1], scale, cfg)
            except Exception as exc:  # noqa: BLE001
                skipped.append([r["name"], f"{type(exc).__name__}: {exc}"])
                continue
            if pp is None:
                skipped.append([r["name"], "no usable overlap with the tile"])
                continue
            y, v = pp
            if int(v.sum()) < cor.n_min:
                skipped.append([r["name"], f"only {int(v.sum())} valid px"])
                continue
            jj, ii = np.nonzero(v)
            cx, cy = float(ii.mean() - (grid.n - 1) / 2) * scale, float(jj.mean() - (grid.n - 1) / 2) * scale
            items.append(dict(rec=r, kind=kind, y=y, v=v, unit=unit_key(r, cfg)))
            if kind == "own":
                stats.append(dict(name=r["name"], tile=tile["id"], band=band, n_valid=int(v.sum()), cx_arcsec=cx, cy_arcsec=cy))
    own = [k for k, it in enumerate(items) if it["kind"] == "own"]
    edges = []
    if len(own) >= 2:
        meta = [dict(unit=items[k]["unit"], prog=items[k]["rec"]["prog"], visit_id=items[k]["rec"]["visit_id"]) for k in own]
        ov = coverage_overlap([items[k]["v"] for k in own])
        sel = select_pairs(meta, ov, cfg["pairs_per_exposure"], cfg["n_other_programme"], cfg["n_other_visit"], cor.n_min)
        nb = bridge_components(meta, ov, sel, cor.n_min)
        pairs = [(own[i], own[j], c) for (i, j), c in sorted(sel.items())]
    else:
        pairs, nb = [], 0
    if cfg["xband"] and xrecs:
        xi = [k for k, it in enumerate(items) if it["kind"] == "x"]
        if xi and own:
            ovx = coverage_overlap([items[k]["v"] for k in own + xi])
            no = len(own)
            for a_, k in enumerate(own):
                cands = sorted(((ovx[a_, no + b_], xi[b_]) for b_ in range(len(xi)) if ovx[a_, no + b_] >= cor.n_min and items[xi[b_]]["unit"] != items[k]["unit"]), reverse=True)
                dif = [c for c in cands if items[c[1]]["rec"]["visit_id"] != items[k]["rec"]["visit_id"]]
                for _, j in (dif + [c for c in cands if c not in dif])[:cfg["n_xband"]]:
                    pairs.append((k, j, "xband"))
    pairs.sort(key=lambda p: (p[0], p[1]))
    ffts = collections.OrderedDict()

    def get(k):
        if k in ffts:
            ffts.move_to_end(k)
            return ffts[k]
        ffts[k] = cor.prep(items[k]["y"], items[k]["v"])
        if len(ffts) > 24:
            ffts.popitem(last=False)
        return ffts[k]
    for i, j, cat in pairs:
        m = cor.measure(get(i), get(j))
        ri, rj = items[i]["rec"], items[j]["rec"]
        sig = max(m["sigma_px"] * scale * 1000.0, cfg["sigma_floor_mas"]) if math.isfinite(m["sigma_px"]) else float("inf")
        edges.append(dict(tile=tile["id"], band_i=ri["band"], band_j=rj["band"], i=ri["name"], j=rj["name"], cat=cat, dx_mas=m["dx_px"] * scale * 1000.0, dy_mas=m["dy_px"] * scale * 1000.0,
                          sigma_mas=sig, peak=m["peak"], second=m["second"], ratio=m["ratio"], n_overlap=m["n_overlap"], quality_ok=bool(m["ok"]), reason=m["reason"], scale=scale))
    res = dict(sig=cfg["sig"], tile=tile["id"], band=band, edges=edges, stats=stats, skipped=skipped, n_bridge=nb, n_images=len(own), seconds=round(time.perf_counter() - t0, 1), cached=False)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(cache, "w"))
    except OSError:
        pass
    return res


# ----------------------------------------------------------------------------- robust joint solve
def _design(iu, ju, U):
    E = len(iu)
    return sp.csr_matrix((np.r_[np.ones(E), -np.ones(E)], (np.r_[np.arange(E), np.arange(E)], np.r_[iu, ju])), shape=(E, U))


def solve_offsets(U, iu, ju, d, sig, tie_a=(), tie_b=(), tie_sig=(), unit_weight=None, gauge_mask=None, huber=1.5, reject=4.0, iters=25):
    """Robust weighted least squares for s (U, 2) [mas] with d_e = s[iu_e] - s[ju_e] (+ noise sigma_e), soft ties s[a] - s[b] = 0 (+ sigma). IRLS: Huber weight above `huber` x
    scale, hard rejection above `reject` x scale, scale = max(1, median(|r|/sigma)/0.8326) from the non-tie edges (so the sigmas are inflated, never deflated). Gauge per connected
    component: weighted mean of s over the gauge set (unit_weight x gauge_mask; the whole component if the set is empty there) is 0.
    Returns dict(s, sigma (U, 2), scale, active (E bool), resid (E, 2), weight (E), comp (U), n_comp, degree)."""
    iu, ju = np.asarray(iu, int), np.asarray(ju, int)
    d, sig = np.asarray(d, float).reshape(-1, 2), np.asarray(sig, float)
    E, Tn = len(iu), len(tie_a)
    uw = np.ones(U) if unit_weight is None else np.asarray(unit_weight, float)
    gm = np.ones(U, bool) if gauge_mask is None else np.asarray(gauge_mask, bool)
    A = _design(np.r_[iu, np.asarray(tie_a, int)], np.r_[ju, np.asarray(tie_b, int)], U) if E + Tn else sp.csr_matrix((0, U))
    dd = np.vstack([d, np.zeros((Tn, 2))])
    sg = np.r_[sig, np.asarray(tie_sig, float)]
    ok_e = np.isfinite(sg) & (sg > 0)
    comp_n, comp = connected_components(A.T @ A, directed=False) if E + Tn else (U, np.arange(U))
    scale, hw = 1.0, np.ones(E + Tn)
    active = ok_e.copy()
    s = np.zeros((U, 2))
    for it in range(iters):
        w = np.where(active, hw / np.where(ok_e, sg, 1.0) ** 2 / scale ** 2, 0.0)
        Nm = (A.T @ sp.diags(w) @ A).toarray()
        Ninv = pinvh(Nm, rtol=1e-10)
        s_new = Ninv @ (A.T @ (w[:, None] * dd))
        r = dd - A @ s_new
        rn = np.hypot(r[:, 0], r[:, 1]) / np.sqrt(2) / np.where(ok_e, sg, 1.0)
        free = ok_e.copy()
        free[E:] = False
        if free.any():
            scale = max(1.0, float(np.median(rn[free])) / 0.8326)
        u = rn / scale
        hw = np.where(u <= huber, 1.0, huber / np.maximum(u, 1e-12))
        hw[E:] = 1.0
        active = ok_e & ((u <= reject) | (np.arange(E + Tn) >= E))
        conv = np.abs(s_new - s).max() < 1e-3 and it > 0
        s = s_new
        if conv:
            break
    w = np.where(active, hw / np.where(ok_e, sg, 1.0) ** 2 / scale ** 2, 0.0)
    Nm = (A.T @ sp.diags(w) @ A).toarray()
    Ninv = pinvh(Nm, rtol=1e-10)
    s = Ninv @ (A.T @ (w[:, None] * dd))
    P = np.eye(U)
    for c in range(comp_n):
        mem = np.where(comp == c)[0]
        g = uw[mem] * gm[mem]
        if g.sum() <= 0:
            g = uw[mem]
        g = g / g.sum()
        P[np.ix_(mem, mem)] -= np.outer(np.ones(len(mem)), g)
    s = P @ s
    cov = P @ Ninv @ P.T
    sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))[:, None] * np.ones((1, 2))
    r = dd - A @ s
    deg = np.zeros(U, int)
    for k in np.where(active[:E])[0]:
        deg[iu[k]] += 1
        deg[ju[k]] += 1
    sigma[deg == 0] = np.inf
    return dict(s=s, sigma=sigma, scale=scale, active=active[:E], resid=r[:E], weight=w[:E], comp=comp, n_comp=int(comp_n), degree=deg, tie_resid=r[E:], tie_active=active[E:])


def fit_similarity(pos, s, sig):
    """s_k = t + eps (X_k - X0) + theta J (X_k - X0), J(x, y) = (-y, x), X0 = weighted mean position. pos (K, 2) arcsec (x west, y north), s (K, 2) mas, sig (K,) mas.
    eps (dimensionless scale error) and theta (rotation, rad) follow: a position r arcsec from the pivot is displaced by 1000 (eps r + theta J r) mas.
    Returns dict(t, eps, theta, sigma_eps, sigma_theta, pivot, rms, dof, model (K, 2)) or None for K < 3."""
    K = len(pos)
    if K < 3:
        return None
    w = 1.0 / np.maximum(np.asarray(sig, float), 1e-3) ** 2
    X0 = (w[:, None] * pos).sum(0) / w.sum()
    r = pos - X0
    A = np.zeros((2 * K, 4))
    A[0::2, 0], A[1::2, 1] = 1.0, 1.0
    A[0::2, 2], A[1::2, 2] = r[:, 0], r[:, 1]      # eps
    A[0::2, 3], A[1::2, 3] = -r[:, 1], r[:, 0]     # theta
    W = np.repeat(w, 2)
    Nm = A.T @ (W[:, None] * A)
    if np.linalg.cond(Nm) > 1e12:
        return None
    C = np.linalg.inv(Nm)
    b = np.linalg.solve(Nm, A.T @ (W * s.reshape(-1)))
    res = s.reshape(-1) - A @ b
    dof = 2 * K - 4
    chi2 = float((W * res ** 2).sum())
    inflate = math.sqrt(max(chi2 / dof, 1.0)) if dof > 0 else 1.0
    return dict(t=b[:2].tolist(), eps=float(b[2]) * 1e-3, theta=float(b[3]) * 1e-3, sigma_eps=float(math.sqrt(C[2, 2])) * inflate * 1e-3, sigma_theta=float(math.sqrt(C[3, 3])) * inflate * 1e-3,
                pivot=X0.tolist(), rms=float(math.sqrt(np.mean(res ** 2))), dof=dof, chi2=chi2, model=(A @ b).reshape(K, 2))


# ----------------------------------------------------------------------------- assembly, summaries
def _pctl(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "p50": float(np.percentile(v, 50)), "p68": float(np.percentile(v, 68)), "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def _wmean(values, weights):
    w = np.asarray(weights, float)
    return (np.asarray(values, float) * w[..., None]).sum(0) / w.sum() if w.sum() > 0 else np.zeros(2)


def assemble_and_solve(results, exps, tiles, cfg):
    """Merge the task results of all tiles and bands, solve jointly, build the output dictionary."""
    t0 = time.perf_counter()
    tile_by_id = {t["id"]: t for t in tiles}
    edges = [dict(e) for r in results for e in r["edges"]]
    stats = [s for r in results for s in r["stats"]]
    names = sorted({s["name"] for s in stats} | {e[k] for e in edges for k in ("i", "j")})
    unit_of = {n: unit_key(exps[n], cfg) for n in names}
    units = sorted(set(unit_of.values()))
    uidx = {u: k for k, u in enumerate(units)}
    U = len(units)
    members = collections.defaultdict(list)
    for n in names:
        members[unit_of[n]].append(n)
    unit_weight = np.array([len(members[u]) for u in units], float)
    gmask = np.ones(U, bool)
    gauge = cfg["gauge"]
    if gauge.startswith("prog:"):
        gp = gauge.split(":", 1)[1]
        gmask = np.array([any(exps[n]["prog"] == gp for n in members[u]) for u in units])
        if not gmask.any():
            log(f"gauge programme {gp} not present: gauge = all")
            gmask[:] = True
    for e in edges:
        e["iu"], e["ju"] = uidx[unit_of[e["i"]]], uidx[unit_of[e["j"]]]
        e["intra_unit"] = e["iu"] == e["ju"]
    use = [k for k, e in enumerate(edges) if e["quality_ok"] and not e["intra_unit"]]
    # pointing ties (detector unit): SW <-> LW of the same dither group and module (soft), optional cross-module
    ta, tb, ts = [], [], []
    if cfg["unit"] == "detector":
        bygrp = collections.defaultdict(list)
        for n in names:
            bygrp[exps[n]["dither"]].append(n)
        for g, lst in bygrp.items():
            for mod in sorted({exps[n]["module"] for n in lst}):
                lw = [n for n in lst if exps[n]["module"] == mod and exps[n]["channel"] == "LW"]
                sw = [n for n in lst if exps[n]["module"] == mod and exps[n]["channel"] == "SW"]
                if lw and cfg["tie_sigma_mas"] > 0:
                    for n in sw:
                        ta.append(uidx[unit_of[n]]); tb.append(uidx[unit_of[lw[0]]]); ts.append(cfg["tie_sigma_mas"])
            if cfg["tie_cross_module_sigma_mas"] > 0:
                rep = {}
                for n in lst:
                    m = exps[n]["module"]
                    if m not in rep or (exps[n]["channel"] == "LW"):
                        rep[m] = n
                ms = sorted(rep)
                for a_, b_ in zip(ms[:-1], ms[1:]):
                    ta.append(uidx[unit_of[rep[a_]]]); tb.append(uidx[unit_of[rep[b_]]]); ts.append(cfg["tie_cross_module_sigma_mas"])
    sol = solve_offsets(U, [edges[k]["iu"] for k in use], [edges[k]["ju"] for k in use], [[edges[k]["dx_mas"], edges[k]["dy_mas"]] for k in use], [edges[k]["sigma_mas"] for k in use],
                        ta, tb, ts, unit_weight=unit_weight, gauge_mask=gmask, huber=cfg["huber"], reject=cfg["reject"], iters=cfg["irls_iters"])
    s, sigma = sol["s"], sol["sigma"]
    active = np.zeros(len(edges), bool)
    active[use] = sol["active"]
    wts = np.zeros(len(edges))
    wts[use] = sol["weight"]
    # residuals of every edge with the final solution (also the ones not used)
    for k, e in enumerate(edges):
        e["resid_mas"] = [e["dx_mas"] - (s[e["iu"], 0] - s[e["ju"], 0]), e["dy_mas"] - (s[e["iu"], 1] - s[e["ju"], 1])]
        e["used"] = bool(active[k])
        e["robust_rejected"] = bool(e["quality_ok"] and not e["intra_unit"] and not active[k])
        e["weight"] = float(wts[k])
    # per image: edge residual bookkeeping (positive residual on i means image i should move further by +r)
    img_edges = collections.defaultdict(list)
    for k, e in enumerate(edges):
        if not e["used"]:
            continue
        for who, sgn in ((e["i"], 1.0), (e["j"], -1.0)):
            img_edges[who].append((e["tile"], sgn * np.array(e["resid_mas"]), wts[k], e["sigma_mas"] * sol["scale"], e["peak"], e["ratio"]))
    pos_by_img = collections.defaultdict(list)
    for st in stats:
        t = tile_by_id[st["tile"]]
        east, north = t["offset_en"]
        pos_by_img[st["name"]].append((-east + st["cx_arcsec"], north + st["cy_arcsec"], st["n_valid"], st["tile"]))
    # similarity per dither group (detector unit)
    sim = {}
    if cfg["similarity"]:
        bygrp = collections.defaultdict(list)
        for n in names:
            if n in pos_by_img and sol["degree"][uidx[unit_of[n]]] > 0:
                bygrp[exps[n]["dither"]].append(n)
        for g, lst in bygrp.items():
            P = np.array([_wmean([[p[0], p[1]] for p in pos_by_img[n]], [p[2] for p in pos_by_img[n]]) for n in lst])
            S = np.array([s[uidx[unit_of[n]]] for n in lst])
            sg = np.array([sigma[uidx[unit_of[n]], 0] for n in lst])
            f = fit_similarity(P, S, sg)
            if f is not None:
                f["names"] = lst
                sim[g] = f
    csize = collections.Counter(int(c) for c, d_ in zip(sol["comp"], sol["degree"]) if d_ > 0)
    comp_rank = {c: k for k, (c, _) in enumerate(sorted(csize.items(), key=lambda kv: (-kv[1], kv[0])))}
    comp_rank = collections.defaultdict(lambda: -1, comp_rank)
    out_exp = {}
    for n in names:
        r = exps[n]
        u = uidx[unit_of[n]]
        ie = img_edges.get(n, [])
        registered = bool(sol["degree"][u] > 0)
        by_tile = {}
        for tid in sorted({x[0] for x in ie}):
            sub = [x for x in ie if x[0] == tid]
            w_ = np.array([x[2] for x in sub])
            res = _wmean([x[1] for x in sub], w_)
            by_tile[str(tid)] = dict(n_edges=len(sub), resid_mas=[float(res[0]), float(res[1])], shift_est_mas=[float(s[u, 0] + res[0]), float(s[u, 1] + res[1])],
                                     sigma_est_mas=float(1.0 / math.sqrt(max(w_.sum(), 1e-12))),
                                     n_valid_px=int(sum(p[2] for p in pos_by_img[n] if p[3] == tid)))
        if ie:
            w_all = np.array([x[2] for x in ie])
            gres = _wmean([x[1] for x in ie], w_all)
        else:
            gres = np.zeros(2)
        est = [v["shift_est_mas"] for v in by_tile.values() if v["n_edges"] >= 3]
        spread = max((float(np.hypot(*(np.array(est[a_]) - np.array(est[b_])))) for a_ in range(len(est)) for b_ in range(a_ + 1, len(est))), default=0.0)
        rr = [np.hypot(*x[1]) for x in ie]
        sh = s[u] if registered else np.zeros(2)
        rec = dict(band=r["band"], detector=r["detector"], programme=r["prog"], visit=r["visit"], visit_key=r["prog"] + "/" + r["visit"], dither_group=r["engine_dither"], unit=unit_of[n], registered=registered, component=comp_rank[int(sol["comp"][u])],
                   tiles=sorted(int(t) for t in by_tile), shift_mas=[float(sh[0]), float(sh[1])], abs_shift_mas=float(np.hypot(*sh)),
                   sigma_mas=[float(sigma[u, 0]), float(sigma[u, 1])] if registered else [None, None], spread_mas=spread, n_edges=len(ie), edge_resid_rms_mas=float(np.sqrt(np.mean(np.square(rr)))) if rr else None,
                   group_resid_mas=[float(gres[0]), float(gres[1])], peak_median=float(np.median([x[4] for x in ie])) if ie else None, ratio_median=float(np.median([x[5] for x in ie])) if ie else None,
                   by_tile=by_tile)
        if r["dither"] in sim and n in sim[r["dither"]]["names"]:
            f = sim[r["dither"]]
            k = f["names"].index(n)
            rec["shift_mas_similarity"] = [float(f["model"][k, 0]), float(f["model"][k, 1])]
            rec["similarity"] = dict(theta_mrad=f["theta"] * 1e3, scale_ppm=f["eps"] * 1e6, sigma_theta_mrad=f["sigma_theta"] * 1e3, sigma_scale_ppm=f["sigma_eps"] * 1e6, rms_mas=f["rms"], n_images=len(f["names"]),
                                     pivot_arcsec_west_north=f["pivot"], translation_at_pivot_mas=f["t"])
            X0 = np.array(f["pivot"])
            for tid, v in by_tile.items():
                t = tile_by_id[int(tid)]
                xt = np.array([-t["offset_en"][0], t["offset_en"][1]]) - X0
                v["shift_mas_model"] = [float(f["t"][0] + 1e3 * (f["eps"] * xt[0] - f["theta"] * xt[1])), float(f["t"][1] + 1e3 * (f["eps"] * xt[1] + f["theta"] * xt[0]))]
        out_exp[n] = rec
    summary = summarize(out_exp, edges, sol, sim, cfg, units, members, exps, results, s, unit_of)
    summary["seconds_solve"] = round(time.perf_counter() - t0, 1)
    return out_exp, edges, summary, sol


def summarize(out_exp, edges, sol, sim, cfg, units, members, exps, results, s, unit_of):
    reg = {n: x for n, x in out_exp.items() if x["registered"]}
    q = [e for e in edges if not e["intra_unit"]]
    qok = [e for e in q if e["quality_ok"]]
    n_rej = sum(1 for e in qok if not e["used"])
    resid = lambda sub: _pctl([math.hypot(*e["resid_mas"]) for e in sub if e["used"]])  # noqa: E731
    rms = lambda sub: float(np.sqrt(np.mean([np.sum(np.square(e["resid_mas"])) / 2 for e in sub if e["used"]]))) if any(e["used"] for e in sub) else None  # noqa: E731
    prog_of = lambda e: exps[e["i"]]["prog"]  # noqa: E731
    reasons = collections.Counter(e["reason"] for e in q if not e["quality_ok"])

    def group(key):
        keys = sorted({x[key] for x in reg.values() if x[key] is not None}, key=str)
        out = {}
        for k in keys:
            sub = [x for x in reg.values() if x[key] == k]
            mv = np.mean([x["shift_mas"] for x in sub], 0)
            out[str(k)] = dict(_pctl([x["abs_shift_mas"] for x in sub]), mean_shift_mas=[float(mv[0]), float(mv[1])], sigma_mas_median=float(np.median([x["sigma_mas"][0] for x in sub])),
                               edge_resid_rms_mas=float(np.sqrt(np.mean([x["edge_resid_rms_mas"] ** 2 for x in sub if x["edge_resid_rms_mas"] is not None]))) if any(x["edge_resid_rms_mas"] is not None for x in sub) else None)
        return out
    # agreement inside a dither group
    agree = []
    bygrp = collections.defaultdict(list)
    for n, x in reg.items():
        bygrp[exps[n]["dither"]].append(n)
    for g, lst in bygrp.items():
        if len(lst) < 2:
            continue
        if cfg["unit"] == "detector":
            vec = [np.array(out_exp[n]["shift_mas"]) for n in lst]
        else:
            vec = [np.array(out_exp[n]["group_resid_mas"]) for n in lst if out_exp[n]["n_edges"] >= 3]
        if len(vec) >= 2:
            agree.append((max(float(np.hypot(*(vec[a] - vec[b]))) for a in range(len(vec)) for b in range(a + 1, len(vec))), g))
    spreads = [x["spread_mas"] for x in reg.values() if len(x["tiles"]) > 1]
    comps = []
    for c in sorted({x["component"] for x in reg.values()}):
        sub = [n for n, x in reg.items() if x["component"] == c]
        comps.append(dict(component=c, n_exposures=len(sub), bands=sorted({reg[n]["band"] for n in sub}), programmes=sorted({reg[n]["programme"] for n in sub}),
                          mean_shift_mas=[float(np.mean([reg[n]["shift_mas"][0] for n in sub])), float(np.mean([reg[n]["shift_mas"][1] for n in sub]))]))
    summ = dict(
        n_exposures=len(out_exp), n_registered=len(reg), n_unregistered=len(out_exp) - len(reg), n_units=len(units), n_components=len(comps), inflation_factor=sol["scale"], components=comps,
        components_warning=("the exposures form %d disconnected groups (no edge, no shared unknown): their RELATIVE offsets are not measured (each has its own gauge). Typical cause: the SW/LW "
                            "simultaneity only links the band pairs observed together (e.g. F150W+F277W and F200W+F356W are separate families); use --xband (F150W:F200W, F277W:F356W edges) or "
                            "--unit detector with ties" % len(comps)) if len(comps) > 1 else None,
        edges=dict(n_measured=len(edges), n_intra_unit=sum(1 for e in edges if e["intra_unit"]), n_quality_ok=len(qok), n_quality_rejected=len(q) - len(qok), quality_reject_reasons=dict(reasons),
                   n_robust_rejected=n_rej, frac_rejected_total=float((len(q) - len(qok) + n_rej) / max(len(q), 1)), frac_robust_rejected_of_ok=float(n_rej / max(len(qok), 1)),
                   n_by_category=dict(collections.Counter(e["cat"] for e in qok)), peak=_pctl([e["peak"] for e in q]), second_peak_ratio=_pctl([e["ratio"] for e in q]),
                   overlap_px=_pctl([e["n_overlap"] for e in q]), sigma_mas_input=_pctl([e["sigma_mas"] for e in qok])),
        closure=dict(all_used_mas=resid(qok), rms_per_axis_mas=rms(qok), by_band={b: resid([e for e in qok if e["band_i"] == b]) for b in sorted({e["band_i"] for e in qok})},
                     by_tile={str(t): resid([e for e in qok if e["tile"] == t]) for t in sorted({e["tile"] for e in qok})},
                     by_programme={p: resid([e for e in qok if prog_of(e) == p]) for p in sorted({prog_of(e) for e in qok})},
                     by_category={c: resid([e for e in qok if e["cat"] == c]) for c in sorted({e["cat"] for e in qok})},
                     cross_programme=resid([e for e in qok if exps[e["i"]]["prog"] != exps[e["j"]]["prog"]]), same_programme=resid([e for e in qok if exps[e["i"]]["prog"] == exps[e["j"]]["prog"]]),
                     note="edge residual |d_ij - (s_i - s_j)| in mas of the edges used in the solve (final solution)"),
        abs_shift_mas_all=_pctl([x["abs_shift_mas"] for x in reg.values()]), sigma_mas=_pctl([x["sigma_mas"][0] for x in reg.values()]),
        by_band=group("band"), by_programme=group("programme"), by_visit=group("visit_key"),
        n_multi_tile=len(spreads), spread_mas_multi_tile=_pctl(spreads), n_spread_gt_10mas=sum(1 for v in spreads if v > 10.0),
        worst_spread=[dict(exposure=n, spread_mas=x["spread_mas"]) for n, x in sorted(reg.items(), key=lambda kv: -kv[1]["spread_mas"])[:10]],
        dither_group_agreement=dict(method=("max pairwise |s_a - s_b| of the detector images of a group (solved shifts)" if cfg["unit"] == "detector" else
                                           "max pairwise difference of the mean edge residuals of the detector images of a group (they share one unknown, so this is the measured disagreement)"),
                                   n_groups=len(agree), spread_mas=_pctl([a for a, _ in agree]), worst=[dict(group=g, spread_mas=a) for a, g in sorted(agree, reverse=True)[:10]]),
        tasks=dict(n=len(results), seconds_sum=round(sum(r["seconds"] for r in results), 1), n_cached=sum(1 for r in results if r.get("cached")), n_skipped_images=sum(len(r["skipped"]) for r in results),
                   skipped_examples=[x for r in results for x in r["skipped"]][:10], n_bridge_edges=sum(r["n_bridge"] for r in results)),
    )
    if cfg["similarity"]:
        summ["similarity"] = dict(n_groups=len(sim), abs_theta_mrad=_pctl([abs(f["theta"]) * 1e3 for f in sim.values()]), abs_scale_ppm=_pctl([abs(f["eps"]) * 1e6 for f in sim.values()]),
                                  median_sigma_theta_mrad=float(np.median([f["sigma_theta"] * 1e3 for f in sim.values()])) if sim else None,
                                  note="per dither group from the per-detector offsets at the detectors' sky positions (pivot = weighted mean position)")
    return summ


def compare_with(res_exp, coarse_path):
    """Differences (new - A12 coarse_shifts.json shift_mas) per exposure, after removing the mean difference (gauges differ)."""
    c = json.load(open(coarse_path))["exposures"]
    names = [n for n, x in res_exp.items() if x["registered"] and n in c and c[n].get("registered")]
    if not names:
        return dict(n=0)
    D = np.array([np.array(res_exp[n]["shift_mas"]) - np.array(c[n]["shift_mas"]) for n in names])
    m = D.mean(0)
    Dm = D - m
    out = dict(n=len(names), mean_difference_mas=[float(m[0]), float(m[1])], abs_difference_after_mean_removal_mas=_pctl(np.hypot(Dm[:, 0], Dm[:, 1])),
               abs_shift_new=_pctl([res_exp[n]["abs_shift_mas"] for n in names]), abs_shift_coarse=_pctl([c[n]["abs_shift_mas"] for n in names]))
    prog = {}
    for p in sorted({res_exp[n]["programme"] for n in names}):
        ii = [k for k, n in enumerate(names) if res_exp[n]["programme"] == p]
        prog[p] = dict(n=len(ii), abs_difference=_pctl(np.hypot(Dm[ii, 0], Dm[ii, 1])), new_abs_shift=_pctl([res_exp[names[k]]["abs_shift_mas"] for k in ii]), coarse_abs_shift=_pctl([c[names[k]]["abs_shift_mas"] for k in ii]))
    out["by_programme"] = prog
    return out


APPROXIMATIONS = [
    "R1 shifts are translations in mas along the tile axes, the same vector in every tile (tile planes differ by < ~2 mrad in orientation over the plan; as A5/A12 of tilefit.py)",
    "R2 a detector image has ONE shift for all tiles it appears in; scale / rotation errors inside an image appear as tile-to-tile residuals (by_tile, spread_mas); --similarity models them per dither group only",
    "R3 bilinear resampling of the undersampled SW images (centroid preserving, phase dependent smoothing); tile->detector map on a 16 px lattice, cubic spline",
    "R4 background = masked block median (0.6 arcsec), smoothed and bilinearly interpolated, not a sliding median filter; asinh stretch, masked Gaussian smoothing",
    "R5 edge sigma = sqrt(4 pi (1 - rho^2)) / (rho sqrt(N) h) from the correlation peak with a floor, calibrated by one global inflation factor from the median edge residual; edge errors treated as independent",
    "R6 only relative shifts: gauge = weighted mean over the gauge set is zero per connected component; no absolute astrometry; uncertainties are relative to the gauge",
    "R7 detector-frame artefacts surviving the DQ mask would give a spurious peak at the dither offset: guarded only by the second-peak ratio and the robust solve",
    "R8 the correlation measures the mean offset over the overlap (the tile box plus margin); offsets larger than the search radius (default 1.5 arcsec) between two exposures are not measurable",
    "R9 the unknown is one translation per dither group (all detectors, both modules, SW and LW share it: exact pointing constraint, assumes no rotation/scale/distortion error between detectors); --unit detector relaxes this with soft SW-LW ties",
    "R10 reliability thresholds (min peak 0.05, second-peak ratio 1.5, overlap 30 arcsec^2, exclusion 0.1 arcsec) are heuristics tuned on synthetic data only",
]


def format_summary(res):
    s = res["summary"]
    L = [f"register_xcorr summary  ({res['seconds']} s, unit={res['config']['unit']}, gauge={res['config']['gauge']})",
         f"exposures {s['n_exposures']}: registered {s['n_registered']}, unregistered {s['n_unregistered']}; unknowns {s['n_units']}; graph components {s['n_components']}; sigma inflation {s['inflation_factor']:.2f}",
         f"edges: measured {s['edges']['n_measured']}, quality ok {s['edges']['n_quality_ok']}, quality rejected {s['edges']['n_quality_rejected']} {s['edges']['quality_reject_reasons']}, "
         f"robust rejected {s['edges']['n_robust_rejected']} (fraction rejected overall {s['edges']['frac_rejected_total']:.3f})"]
    c = s["closure"]["all_used_mas"]
    if s.get("components_warning"):
        L.append("WARNING: " + s["components_warning"])
        for c_ in s["components"]:
            L.append(f"  component {c_['component']}: {c_['n_exposures']} exposures, bands {c_['bands']}, programmes {c_['programmes']}, mean shift ({c_['mean_shift_mas'][0]:.1f}, {c_['mean_shift_mas'][1]:.1f}) mas")
    if c.get("n"):
        L.append(f"closure |edge residual| mas: p50 {c['p50']:.2f} p90 {c['p90']:.2f} p99 {c['p99']:.2f} max {c['max']:.2f}; rms per axis {s['closure']['rms_per_axis_mas']:.2f}")
    a = s["abs_shift_mas_all"]
    if a.get("n"):
        L.append(f"|shift| mas (relative to the gauge): p50 {a['p50']:.1f} p90 {a['p90']:.1f} p99 {a['p99']:.1f} max {a['max']:.1f}; sigma mas p50 {s['sigma_mas'].get('p50', float('nan')):.2f} p90 {s['sigma_mas'].get('p90', float('nan')):.2f}")
    for p, q in s["by_programme"].items():
        L.append(f"  programme {p}: n {q['n']}, |shift| p50 {q['p50']:.1f} p90 {q['p90']:.1f} max {q['max']:.1f}, mean ({q['mean_shift_mas'][0]:.1f}, {q['mean_shift_mas'][1]:.1f}) mas, sigma {q['sigma_mas_median']:.2f}")
    for v, q in s["by_visit"].items():
        L.append(f"  visit {v}: n {q['n']}, |shift| p50 {q['p50']:.1f} max {q['max']:.1f}, mean ({q['mean_shift_mas'][0]:.1f}, {q['mean_shift_mas'][1]:.1f})")
    for b, q in s["by_band"].items():
        L.append(f"  band {b}: n {q['n']}, |shift| p50 {q['p50']:.1f} p90 {q['p90']:.1f} max {q['max']:.1f}")
    g = s["dither_group_agreement"]["spread_mas"]
    if g.get("n"):
        L.append(f"dither-group agreement (mas, {s['dither_group_agreement']['n_groups']} groups): p50 {g['p50']:.1f} p90 {g['p90']:.1f} max {g['max']:.1f}")
    sp_ = s["spread_mas_multi_tile"]
    if sp_.get("n"):
        L.append(f"spread between per-tile estimates ({s['n_multi_tile']} multi-tile exposures): p50 {sp_['p50']:.1f} p90 {sp_['p90']:.1f} max {sp_['max']:.1f} mas; > 10 mas: {s['n_spread_gt_10mas']}")
    if "compare_coarse" in res:
        cc = res["compare_coarse"]
        if cc.get("n"):
            d = cc["abs_difference_after_mean_removal_mas"]
            L.append(f"vs coarse_shifts.json: n {cc['n']}, |new - coarse| after mean removal p50 {d['p50']:.1f} p90 {d['p90']:.1f} max {d['max']:.1f} mas (|coarse| p50 {cc['abs_shift_coarse']['p50']:.1f}, |new| p50 {cc['abs_shift_new']['p50']:.1f})")
    return "\n".join(L)


# ----------------------------------------------------------------------------- driver
def _clean(o, nd=None):
    """JSON-safe copy: numpy scalars -> python, non-finite floats -> None, optional rounding."""
    if isinstance(o, dict):
        return {str(k): _clean(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v, nd) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist(), nd)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not math.isfinite(v) else (round(v, nd) if nd is not None else v)
    return o


def make_cfg(**kw):
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in kw.items() if v is not None or k in ("data_root",)})
    if cfg["similarity"] and cfg["unit"] != "detector":
        log("--similarity needs per-detector offsets: unit = detector")
        cfg["unit"] = "detector"
    return cfg


def run(plan, out, tiles=None, bands=None, compare=None, **kw):
    """Whole pipeline. Writes OUT/register_xcorr.json, register_xcorr_edges.json, register_xcorr_summary.txt; returns the result dict."""
    t_start = time.perf_counter()
    cfg = make_cfg(**kw)
    out = Path(out)
    (out / "work").mkdir(parents=True, exist_ok=True)
    cfg["out"] = str(out)
    tl, exps = load_plan(plan, cfg["tile_use"], cfg["data_root"], set(tiles) if tiles else None, set(bands) if bands else None)
    keys = ("radius_arcsec", "margin_arcsec", "pairs_per_exposure", "n_other_programme", "n_other_visit", "min_overlap_arcsec2", "bg_block_arcsec", "smooth_arcsec", "stretch", "asinh_soft",
            "clip_sigma", "min_peak", "min_ratio", "excl_arcsec", "sigma_floor_mas", "unit", "xband", "xband_pairs", "n_xband", "wcs", "lattice_px", "max_exposures_per_task", "tile_use")
    cfg["sig"] = json.dumps({k: cfg[k] for k in keys}, sort_keys=True) + "|" + str(plan)
    xp = {}
    for pr in (cfg["xband_pairs"].split(",") if cfg["xband"] and cfg["xband_pairs"] else []):
        a_, b_ = pr.split(":")
        xp[a_] = b_
    tasks = []
    for t in tl:
        for b, names in t["exposures"].items():
            xr = [exps[n] for n in t["exposures"].get(xp.get(b), [])] if b in xp else []
            tasks.append((t, b, [exps[n] for n in names], xr, cfg))
    tasks.sort(key=lambda k: -(len(k[2]) ** 2))
    log(f"plan {plan}: {len(tl)} tiles, {len(exps)} exposures, {len(tasks)} tile x band tasks, {cfg['workers']} workers, unit {cfg['unit']}, wcs {cfg['wcs']}")
    results = []
    if cfg["workers"] > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(cfg["workers"]) as pool:
            for r in pool.imap_unordered(run_task, tasks):
                results.append(r)
                log(f"tile {r['tile']} {r['band']}: {r['n_images']} images, {len(r['edges'])} edges, {sum(1 for e in r['edges'] if e['quality_ok'])} ok, {r['seconds']} s{' (cached)' if r.get('cached') else ''}"
                    f"{' skipped ' + str(len(r['skipped'])) if r['skipped'] else ''} [{len(results)}/{len(tasks)}]")
    else:
        for tk in tasks:
            r = run_task(tk)
            results.append(r)
            log(f"tile {r['tile']} {r['band']}: {r['n_images']} images, {len(r['edges'])} edges, {sum(1 for e in r['edges'] if e['quality_ok'])} ok, {r['seconds']} s [{len(results)}/{len(tasks)}]")
    results.sort(key=lambda r: (r["tile"], r["band"]))
    out_exp, edges, summary, sol = assemble_and_solve(results, exps, tl, cfg)
    res = dict(description="register_xcorr: catalogue-free relative registration by pairwise image cross-correlation + one robust joint solve. shift_mas = [dx, dy] in mas along the tile "
               "pixel axes (north-up TAN, x to the west, y to the north), the shift to ADD to the header-WCS pixel position of the exposure (same convention as coarse_shifts.json of "
               "tilefit A12). Relative to the gauge only. sigma_mas = formal uncertainty (inflated by the global factor), relative to the gauge.",
               radius_arcsec=cfg["radius_arcsec"], applied=False, seconds=round(time.perf_counter() - t_start, 1), summary=summary, exposures=out_exp,
               config={k: v for k, v in cfg.items() if k not in ("sig", "out")}, approximations=APPROXIMATIONS, plan=str(plan))
    if compare:
        res["compare_coarse"] = compare_with(out_exp, compare)
    json.dump(_clean(res), open(out / "register_xcorr.json", "w"), indent=1)
    slim = [{k: v for k, v in e.items() if k not in ("iu", "ju", "scale")} for e in edges]
    json.dump(_clean(slim, 4), open(out / "register_xcorr_edges.json", "w"))
    txt = format_summary(res)
    open(out / "register_xcorr_summary.txt", "w").write(txt + "\n")
    log("\n" + txt)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--plan", required=True, help="plan_tiles.py JSON (plan_lmc.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiles", default=None, help="comma list of tile ids (default all)")
    ap.add_argument("--bands", default=None, help="comma list of bands (default all)")
    ap.add_argument("--tile-use", choices=("all", "covering"), default=DEFAULTS["tile_use"])
    ap.add_argument("--data-root", default=None, help="directory (or parent of directories) holding the _cal files when the plan paths do not exist")
    ap.add_argument("--wcs", choices=("fits", "gwcs"), default="fits", help="fits: SCI-header WCS with SIP (what tilefit / the engine use); gwcs: gwcs of the ASDF extension (asdf + gwcs)")
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--radius-arcsec", type=float, default=DEFAULTS["radius_arcsec"])
    ap.add_argument("--margin-arcsec", type=float, default=DEFAULTS["margin_arcsec"], help="grid margin beyond the search radius")
    ap.add_argument("--pairs-per-exposure", type=int, default=DEFAULTS["pairs_per_exposure"])
    ap.add_argument("--n-other-programme", type=int, default=DEFAULTS["n_other_programme"])
    ap.add_argument("--n-other-visit", type=int, default=DEFAULTS["n_other_visit"])
    ap.add_argument("--min-overlap-arcsec2", type=float, default=DEFAULTS["min_overlap_arcsec2"])
    ap.add_argument("--bg-block-arcsec", type=float, default=DEFAULTS["bg_block_arcsec"])
    ap.add_argument("--smooth-arcsec", type=float, default=DEFAULTS["smooth_arcsec"])
    ap.add_argument("--stretch", choices=("asinh", "clip", "rank"), default=DEFAULTS["stretch"])
    ap.add_argument("--asinh-soft", type=float, default=DEFAULTS["asinh_soft"])
    ap.add_argument("--min-peak", type=float, default=DEFAULTS["min_peak"])
    ap.add_argument("--min-ratio", type=float, default=DEFAULTS["min_ratio"], help="minimum peak / second-peak ratio")
    ap.add_argument("--excl-arcsec", type=float, default=DEFAULTS["excl_arcsec"])
    ap.add_argument("--sigma-floor-mas", type=float, default=DEFAULTS["sigma_floor_mas"])
    ap.add_argument("--unit", choices=("group", "detector"), default=DEFAULTS["unit"], help="unknown per dither group (default) or per detector image")
    ap.add_argument("--similarity", action="store_true", help="per dither group scale + rotation from the per-detector offsets (implies --unit detector)")
    ap.add_argument("--tie-sigma-mas", type=float, default=DEFAULTS["tie_sigma_mas"], help="--unit detector: soft SW-LW (same module, same dither) pointing tie")
    ap.add_argument("--tie-cross-module-sigma-mas", type=float, default=DEFAULTS["tie_cross_module_sigma_mas"], help="--unit detector: soft tie between modules A and B of a dither (0 = none)")
    ap.add_argument("--xband", action="store_true", help="extra edges from cross-band correlation of adjacent bands (--xband-pairs)")
    ap.add_argument("--xband-pairs", default=DEFAULTS["xband_pairs"])
    ap.add_argument("--n-xband", type=int, default=DEFAULTS["n_xband"])
    ap.add_argument("--gauge", default=DEFAULTS["gauge"], help="all: weighted mean offset of every detector image is 0; prog:<id>: of that programme only")
    ap.add_argument("--huber", type=float, default=DEFAULTS["huber"])
    ap.add_argument("--reject", type=float, default=DEFAULTS["reject"])
    ap.add_argument("--resume", action="store_true", help="reuse OUT/work/edges_t*_*.json of an earlier run with the same settings")
    ap.add_argument("--max-exposures-per-task", type=int, default=0, help="debug: cap on images per tile x band")
    ap.add_argument("--compare", default=None, help="coarse_shifts.json of tilefit A12 to compare with")
    a = ap.parse_args(argv)
    kw = {k: v for k, v in vars(a).items() if k not in ("plan", "out", "tiles", "bands", "compare")}
    run(a.plan, a.out, tiles=[int(x) for x in a.tiles.split(",")] if a.tiles else None, bands=a.bands.split(",") if a.bands else None, compare=a.compare, **kw)


if __name__ == "__main__":
    main()
