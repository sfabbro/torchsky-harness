"""Exposures, patches and the real-data loader (protocol v1 splits; JWST _cal / _rate, HST _flt via the same header-WCS path).

Every exposure keeps a rectangular CUT of its detector: all pixels whose header-WCS position falls in the data region (scoring box +
--data-margin).  Header-WCS positions X, Y [arcsec] in the tangent plane at the box centre (sky.py convention) are precomputed per pixel;
the learned geometry acts on top of them.  Exposures are never resampled or pre-aligned.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from . import instruments as _ins

# instrument facts live in instruments.py (INSTRUMENT_INTERFACE.md); these names are kept for older callers
JWST_DET_CHAN = dict(_ins._NIRCAM_CHAN)
CHAN_PIX = {c: p for i in _ins.INSTRUMENTS.values() for c, p in i.pixel_scale.items()}
INST_DIAM = _ins.diameters()


@dataclass
class Exposure:
    name: str
    filt: int
    inst: int
    chan: int
    det: int
    t: float
    mjd: float
    det_shape: tuple
    cut0: tuple            # (y0, x0) of the cut in detector pixels
    d: torch.Tensor        # (ny, nx)
    sig: torch.Tensor      # pipeline sigma
    valid: torch.Tensor    # bool
    X: torch.Tensor        # header-WCS tangent-plane positions [arcsec]
    Y: torch.Tensor
    R: np.ndarray          # 2x2 rotation (parity) of the mean Jacobian
    split: str = "train"
    group: str = ""
    shift0: tuple = (0.0, 0.0)
    stripes: bool = False
    bkg0: float = 0.0
    Jpix: torch.Tensor = None   # (ny, nx, 2, 2) per-pixel header Jacobian d(X,Y)/d(col,row) (rate input, area tests)
    eid: int = -1               # GLOBAL exposure id: the row of every per-exposure parameter (shared by all cuts of the exposure)
    patches: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    tile: int = 0               # scoring box (sky patch) this cut belongs to
    c0: tuple = (0.0, 0.0)      # tangent-plane offset of the tile centre from the field reference [arcsec] (X west, Y north; 0 for one tile)
    cid: int = -1               # index of this cut in the list of cuts (== eid with one tile)
    host: dict = None           # multi-tile: CPU copies of the pixel tensors; the attributes point to them unless the tile is resident

    DATA = ("d", "sig", "valid", "X", "Y", "Jpix")

    @property
    def shape(self):
        return tuple((self.d if self.d is not None else self.host["d"]).shape)

    def nbytes(self):
        src = self.host if self.host is not None else {k: getattr(self, k) for k in self.DATA}
        return int(sum(v.numel() * v.element_size() for v in src.values() if v is not None))

    def hostv(self, k):
        """The tensor k without touching the device (host copy when swapping is on)."""
        return self.host[k] if self.host is not None else getattr(self, k)

    def to_dev(self, dev):
        """Make the pixel tensors resident on dev (a fresh copy; the host copy is kept). No-op without host storage."""
        if self.host is None:
            return
        for k in self.DATA:
            v = self.host.get(k)
            setattr(self, k, None if v is None else v.to(dev, non_blocking=True, copy=True))

    def release(self):
        """Drop the device copies (the attributes point back to the host tensors)."""
        if self.host is None:
            return
        for k in self.DATA:
            setattr(self, k, self.host.get(k))

    def pix(self, sl, only_valid=False):
        y0, y1, x0, x1 = sl
        dev = self.d.device
        rows = torch.arange(y0, y1, device=dev) + self.cut0[0]
        cols = torch.arange(x0, x1, device=dev) + self.cut0[1]
        RR, CC = torch.meshgrid(rows, cols, indexing="ij")
        ny, nx = self.det_shape
        out = {"X": self.X[y0:y1, x0:x1].reshape(-1), "Y": self.Y[y0:y1, x0:x1].reshape(-1),
               "row": RR.reshape(-1), "col": CC.reshape(-1),
               "xi_px": torch.stack([CC.reshape(-1).float(), RR.reshape(-1).float()], -1),
               "d": self.d[y0:y1, x0:x1].reshape(-1), "sig": self.sig[y0:y1, x0:x1].reshape(-1),
               "valid": self.valid[y0:y1, x0:x1].reshape(-1)}
        out["xi"] = torch.stack([out["xi_px"][:, 0] / (nx - 1) * 2 - 1, out["xi_px"][:, 1] / (ny - 1) * 2 - 1], -1)
        out["bounds"] = (x0 + self.cut0[1], x1 + self.cut0[1], y0 + self.cut0[0], y1 + self.cut0[0])
        out["c0"] = self.c0
        if self.Jpix is not None:
            out["Jpix"] = self.Jpix[y0:y1, x0:x1].reshape(-1, 2, 2)
        if only_valid:
            v = out["valid"]
            out = {k: (x[v] if torch.is_tensor(x) and x.shape[0] == v.shape[0] else x) for k, x in out.items()}
        return out


def make_patches(e: Exposure, patch_px, min_valid=0.15):
    ny, nx = e.shape
    out = []
    for y0 in range(0, ny, patch_px):
        for x0 in range(0, nx, patch_px):
            sl = (y0, min(ny, y0 + patch_px), x0, min(nx, x0 + patch_px))
            v = e.valid[sl[0]:sl[1], sl[2]:sl[3]]
            if float(v.float().mean()) >= min_valid and int(v.sum()) >= 16:
                out.append(sl)
    e.patches = out
    return out


def rotation_of(J):
    """Polar factor of a 2x2 Jacobian (rotation, with the parity flip when det J < 0)."""
    U, _, Vt = np.linalg.svd(np.asarray(J, np.float64))
    return U @ Vt


def mean_jacobian(X, Y):
    """Least-squares linear map d(X,Y)/d(col,row) of a cut (numpy arrays)."""
    ny, nx = X.shape
    rr, cc = np.mgrid[0:ny, 0:nx]
    s = (slice(None, None, max(1, ny // 64)), slice(None, None, max(1, nx // 64)))
    A = np.stack([np.ones(rr[s].size), cc[s].ravel() - cc.mean(), rr[s].ravel() - rr.mean()], 1)
    coef = np.linalg.lstsq(A, np.stack([X[s].ravel(), Y[s].ravel()], 1), rcond=None)[0]
    return coef[1:].T  # J[i, j] = d(X_i)/d(xi_j), xi = (col, row)


def pixel_jacobian(X, Y):
    """Per-pixel header Jacobian by central differences (one-sided at the edges): (ny, nx, 2, 2)."""
    dXc, dXr = np.gradient(X, axis=1), np.gradient(X, axis=0)
    dYc, dYr = np.gradient(Y, axis=1), np.gradient(Y, axis=0)
    return np.stack([np.stack([dXc, dXr], -1), np.stack([dYc, dYr], -1)], -2).astype(np.float32)


# ----------------------------------------------------------------------------- real data
def target_header(ra, dec, size, scale_arcsec):
    s = scale_arcsec / 3600.0
    return {"NAXIS": 2, "NAXIS1": size, "NAXIS2": size, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CUNIT1": "deg", "CUNIT2": "deg",
            "CRVAL1": ra, "CRVAL2": dec, "CRPIX1": size / 2 + 0.5, "CRPIX2": size / 2 + 0.5, "CD1_1": -s, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": s}


def tangent_positions(wcs_det, wcs_t, cols, rows, scale, size):
    pc = torch.tensor(np.stack([cols.ravel(), rows.ravel()], 1), dtype=torch.float64)
    tp = wcs_t.world_to_pixel_differentiable(wcs_det.pixel_to_world_differentiable(pc, origin=0), origin=0).numpy()
    c = (size - 1) / 2
    return ((tp[:, 0] - c) * scale).reshape(cols.shape), ((tp[:, 1] - c) * scale).reshape(cols.shape)


RATE_WCS = "cal"     # --rate-wcs: with _rate input, the header WCS comes from the matching _cal file ("cal"), the _rate file ("rate"), or the telescope frame
LOADER = "astropy"   # --loader: "astropy" (the prototype's reader) or "torchfits" (torchsky's field_observation_from_exposure_file; falls back to astropy with a message)
_LOADER_NOTE = []


def read_exposure_file(path, rate_path=None):
    """Dispatcher: the prototype's astropy reader (default), or torchsky's reader (--loader torchfits), and with --rate-wcs telescope the
    count-rate positions from torchsky's telescope-frame transform. Every default value leaves the astropy path below unchanged."""
    if LOADER == "astropy" and RATE_WCS != "telescope":
        return _read_exposure_file_astropy(path, rate_path)
    from . import tsky_backends as tb
    raw = None
    if LOADER == "torchfits":
        if tb.torchfits_reader_available():
            raw = tb.read_exposure_file_torchfits(path, rate_path, DQ_IGNORE_BITS, RATE_WCS)
        elif not _LOADER_NOTE:
            _LOADER_NOTE.append(tb.LOADER_FALLBACK_MESSAGE.format(err=tb._LOADER_STATE["err"]))
            print(_LOADER_NOTE[0], flush=True)
    if raw is None:
        raw = _read_exposure_file_astropy(path, rate_path)
    if rate_path and RATE_WCS == "cal" and "photmjsr" not in raw:       # torchfits reader: PHOTMJSR from the _cal SCI header
        from astropy.io import fits
        raw["photmjsr"] = float(fits.getheader(path, "SCI").get("PHOTMJSR", float("nan")))
    if rate_path and RATE_WCS == "telescope":
        raw["w"] = tb.TelescopeRateWCS(rate_path)
        raw["pixar_a2"] = raw["w"].pixar_a2
    return raw


def _read_exposure_file_astropy(path, rate_path=None):
    """SCI / ERR / DQ / AREA and the header WCS of one _cal (or _rate) file. With rate_path, SCI / ERR / DQ come from the _rate file and,
    unless RATE_WCS == "rate", the header WCS (and PIXAR_A2) from the SCI header of the _cal file `path`: the _rate header carries only the
    linear TAN WCS of the pointing (no distortion; 0.07 arcsec median, 0.24 arcsec maximum from the _cal SIP WCS over NRCB1), and the WCS is a
    geometric, not a photometric, calibration (F46)."""
    from astropy.io import fits
    from torchsky.wcs import WCS
    src = rate_path or path
    with fits.open(src, memmap=False) as h:
        sh = h["SCI"].header
        sci = h["SCI"].data.astype(np.float32)
        err = h["ERR"].data.astype(np.float32)
        dq = h["DQ"].data
        h0 = h[0].header
        area = h["AREA"].data.astype(np.float32) if "AREA" in h else None
    if rate_path and RATE_WCS == "cal":
        sh = fits.getheader(path, "SCI")
    hd = {k: sh[k] for k in sh.keys() if k and k not in ("COMMENT", "HISTORY", "")}
    # PHOTMJSR (MJy/sr per DN/s) of the calibrated file whose header is read (--rate-units sky); absent from _rate headers
    return dict(sci=sci, err=err, dq=dq, area=area, w=WCS(header=hd), pixar_a2=float(sh.get("PIXAR_A2", float("nan"))),
                photmjsr=float(sh.get("PHOTMJSR", float("nan"))) if (rate_path and RATE_WCS == "cal") else float("nan"),
                detector=str(h0.get("DETECTOR", "")).strip().upper(), instrument=str(h0.get("INSTRUME", "")).strip().upper())


DQ_IGNORE_BITS = 0   # --dq-ignore-bits: NIRCam's data-quality bits that do NOT exclude a pixel (0 = strict DQ == 0, F27; set by cli.main, F42);
#                      other instruments: their own policy (instruments.py, --dq-ignore)
FILTER_CURVE = None  # filter curve reader (instrument, filter, Q, cache dir) -> Filter; None = instruments.filter_curve (tests may set a stand-in)


def load_exposure_file(path, box, half_data, scale=0.01, want_jac=False, rate_path=None, shift0=(0.0, 0.0), raw=None):
    """Read one _cal (or _rate) file (or use `raw` from read_exposure_file); return a dict of numpy arrays for the cut that falls inside the
    data region of `box`, or None. The data region is tested at the REGISTERED positions (header + shift0), so a header error does not move
    the cut off the sky domain; the returned X, Y are the header positions (shift0 is applied by the model)."""
    from torchsky.wcs import WCS
    ra, dec = box["ra"], box["dec"]
    size = int(math.ceil(2 * half_data / scale)) + 64
    thead = target_header(ra, dec, size, scale)
    wt = WCS(header=thead)
    raw = raw or read_exposure_file(path, rate_path)
    sci, err, dq, area, w = raw["sci"], raw["err"], raw["dq"], raw["area"], raw["w"]
    sx, sy = float(shift0[0]), float(shift0[1])
    ny, nx = sci.shape
    yy, xx = np.mgrid[0:ny:32, 0:nx:32]
    X, Y = tangent_positions(w, wt, xx.astype(np.float64), yy.astype(np.float64), scale, size)
    lim = half_data + 1.5
    ins = (np.abs(X + sx) < lim) & (np.abs(Y + sy) < lim)
    if not ins.any():
        return None
    x0, x1 = max(int(xx[ins].min()) - 40, 0), min(int(xx[ins].max()) + 41, nx)
    y0, y1 = max(int(yy[ins].min()) - 40, 0), min(int(yy[ins].max()) + 41, ny)
    rr, cc = np.mgrid[y0:y1, x0:x1]
    X, Y = tangent_positions(w, wt, cc.astype(np.float64), rr.astype(np.float64), scale, size)
    inside = (np.abs(X + sx) < half_data) & (np.abs(Y + sy) < half_data)
    if not inside.any():
        return None
    r_any, c_any = np.nonzero(inside)
    a0, a1, b0, b1 = r_any.min(), r_any.max() + 1, c_any.min(), c_any.max() + 1
    sl = (slice(a0, a1), slice(b0, b1))
    X, Y, inside = X[sl], Y[sl], inside[sl]
    y0, x0 = y0 + a0, x0 + b0
    y1, x1 = y0 + (a1 - a0), x0 + (b1 - b0)
    d, e, q = sci[y0:y1, x0:x1], err[y0:y1, x0:x1], None if dq is None else dq[y0:y1, x0:x1]
    if raw.get("usable") is not None:   # --loader torchfits: the reader returns the usability mask (finite, error > 0, DQ policy), not the DQ plane
        valid = inside & raw["usable"][y0:y1, x0:x1]
    elif raw.get("dq_ok") is not None:  # the instrument's own data-quality policy (instruments.py; NIRCam: the expression below)
        valid = inside & np.isfinite(d) & np.isfinite(e) & (e > 0) & raw["dq_ok"](q)
    else:
        valid = inside & np.isfinite(d) & np.isfinite(e) & (e > 0) & ((q.astype(np.int64) & ~int(DQ_IGNORE_BITS)) == 0)
    if valid.sum() < int(raw.get("min_valid", 2000)):
        return None
    f_sky = float(raw.get("to_sky", 1.0)) if raw.get("level", "cal") == "cal" else 1.0
    if f_sky != 1.0:                    # calibrated level, file not in MJy/sr (HST): science and error to surface brightness (instruments.to_sky_factor)
        d, e = (d * np.float32(f_sky)).astype(np.float32), (e * np.float32(f_sky)).astype(np.float32)
    out = dict(d=np.nan_to_num(d), sig=np.where(np.isfinite(e) & (e > 0), e, 1.0).astype(np.float32), valid=valid, X=X.astype(np.float32),
               Y=Y.astype(np.float32), cut0=(int(y0), int(x0)), det_shape=(ny, nx), J=mean_jacobian(X, Y),
               detector=raw["detector"], instrument=raw["instrument"],
               bkg0=float(np.median(d[valid])), area_ext=None if area is None else area[y0:y1, x0:x1],
               pixar_a2=raw["pixar_a2"], photmjsr=raw.get("photmjsr", float("nan")))
    if want_jac:
        out["Jpix"] = pixel_jacobian(X.astype(np.float64), Y.astype(np.float64))
    return out


def svo_id(instrument, filt):
    """Filter-curve identifier of the instrument's description (instruments.Instrument.bandpass_id)."""
    return _ins.resolve(instrument).bandpass_id(filt)


def filter_instruments(ex):
    """{filter name: instrument} of the split's exposures. Filters are keyed by NAME everywhere (FilterSet.index), so a name used by two
    instruments (e.g. F814W on ACS and WFC3) is refused rather than given one instrument's curve."""
    out = {}
    for e in ex:
        f, i = e["filter"], _ins.resolve(e).key
        if out.setdefault(f, i) != i:
            raise SystemExit(f"filter {f} is used by two instruments ({out[f]}, {i}); filters are keyed by name: rename one of them in the splits file")
    return out


def parse_tiles(cfg, sp):
    """Selected scoring boxes (sky patches): --tiles '' = the single box --box-id; 'all'; or a comma list of box ids."""
    t = getattr(cfg, "tiles", "") or ""
    if not t:
        return [int(cfg.box_id)]
    if t == "all":
        return sorted(int(b["id"]) for b in sp["boxes"])
    return [int(x) for x in t.split(",") if x.strip() != ""]


def tile_offsets(boxes):
    """Tangent-plane offsets [arcsec] of the box centres from their centroid (gnomonic at the centroid; X to the WEST, Y north, the sky.py
    convention). One box: (0, 0)."""
    ids = [int(b["id"]) for b in boxes]
    if len(boxes) == 1:
        return {ids[0]: (0.0, 0.0)}
    ra = np.radians([b["ra"] for b in boxes])
    de = np.radians([b["dec"] for b in boxes])
    v = np.stack([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)], 1).mean(0)
    a0, d0 = math.atan2(v[1], v[0]), math.atan2(v[2], math.hypot(v[0], v[1]))
    cosc = math.sin(d0) * np.sin(de) + math.cos(d0) * np.cos(de) * np.cos(ra - a0)
    xi = np.cos(de) * np.sin(ra - a0) / cosc                                     # east
    eta = (math.cos(d0) * np.sin(de) - math.sin(d0) * np.cos(de) * np.cos(ra - a0)) / cosc
    r2a = 180.0 / math.pi * 3600.0
    return {k: (float(-xi[i] * r2a), float(eta[i] * r2a)) for i, k in enumerate(ids)}


def _read_cuts(path, boxes, half_data, shift0, want_jac, rpath, inst=None):
    """One file read (by its instrument's reader), one cut per chip and box (thread worker). Returns ([(box_id, cut dict)], error or None); each
    cut dict carries its chip's record fields (chip, name_suffix, to_sky, ...)."""
    try:
        chips = (inst or _ins.INSTRUMENTS["NIRCAM"]).read(path, rpath)
        out = []
        for raw in chips:
            for b in boxes:
                r = load_exposure_file(path, b, half_data, want_jac=want_jac, shift0=shift0, raw=raw)
                if r is not None:
                    r.update({k: raw[k] for k in ("chip", "name_suffix", "to_sky", "level", "sci_unit", "to_sky_how", "wcs_name", "exptime", "filter")
                              if k in raw})
                    out.append((int(b["id"]), r))
        return out, None
    except Exception as exc:  # noqa: BLE001
        return [], exc


def epoch_coordinates(mjd, inst, min_days, per_instrument=True, ref_mjd=None, ref_inst=None):
    """Normalised epoch t in [-1, 1] of every exposure (audit finding 12). per_instrument: each instrument's own first and last exposure map
    to -1 and +1, and an instrument whose own span is shorter than min_days gets t = 0 for all its exposures (its epoch axes are off). Not
    per_instrument (--legacy-defaults): the old rule, one span over the whole field, so adding an instrument observed years earlier switched
    the epoch axes on for every instrument. ref_mjd / ref_inst (default: the same arrays) are the exposures that define the spans."""
    mjd = np.asarray(mjd, float)
    inst = np.asarray(inst)
    rm = mjd if ref_mjd is None else np.asarray(ref_mjd, float)
    ri = inst if ref_inst is None else np.asarray(ref_inst)
    t = np.zeros(len(mjd))
    for k in ([None] if not per_instrument else list(np.unique(inst))):
        g = np.ones(len(mjd), bool) if k is None else (inst == k)
        if not g.any():
            continue
        r = np.ones(len(rm), bool) if k is None else (ri == k)
        lo, hi = rm[r].min(), rm[r].max()
        span = max(float(hi - lo), 1e-9)
        if span >= min_days:
            t[g] = (mjd[g] - lo) / span * 2 - 1
    return [float(x) for x in t]


def load_field(cfg, dev, log=print):
    """Protocol v1 field: exposures of the split that overlap the data region of the selected box(es) (--tiles, default the one box --box-id).
    Each file is read once and cut once per selected box it overlaps (a CUT is an Exposure object: its own pixels, header positions in that
    box's tangent plane, tile id and tile offset c0, and the GLOBAL exposure id eid shared by all cuts of the file). With one box this is
    the single-box loader (cuts == exposures, cid == eid, tensors on `dev`). With several boxes the pixel tensors stay on the host (pinned
    when dev is CUDA) and are moved to the device per tile by tiles.TileSet. Returns (cuts, fset, meta, info); meta is per global exposure."""
    from concurrent.futures import ThreadPoolExecutor
    from .bandpass import FilterSet
    sp = json.load(open(cfg.splits))
    tiles = parse_tiles(cfg, sp)
    boxes = {int(b["id"]): b for b in sp["boxes"] if int(b["id"]) in tiles}
    missing = [k for k in tiles if k not in boxes]
    if missing:
        raise SystemExit(f"--tiles: unknown box ids {missing}")
    sizes = {float(b["size_arcsec"]) for b in boxes.values()}
    if len(sizes) != 1:
        raise SystemExit(f"selected boxes have different sizes {sizes}")
    box = boxes[tiles[0]]
    half_data = box["size_arcsec"] / 2 + cfg.data_margin
    multi = len(tiles) > 1
    c0s = tile_offsets([boxes[k] for k in tiles])
    split = sp["splits"][cfg.split]
    held, train = set(split["heldout"]), set(split["train"])
    tset = {str(k) for k in tiles}
    ex = [e for e in sp["exposures"] if e["rootname"] in held | train and (bool(tset & {str(k) for k in e.get("boxes", {})}) or cfg.all_box_files)]
    if cfg.filters:
        ex = [e for e in ex if e["filter"] in cfg.filters.split(",")]
    if cfg.max_exposures > 0:
        ex = ex[: cfg.max_exposures]
    shifts = {}
    if cfg.shifts_from:
        rj = json.load(open(cfg.shifts_from))
        for fn, r in rj.get("exposures", {}).items():
            if r.get("registered", True):
                shifts[fn.replace("_cal.fits", "")] = (r["shift_mas"][0] / 1000.0, r["shift_mas"][1] / 1000.0)
    ins_of = {id(e): _ins.resolve(e) for e in ex}                   # instrument description of every entry (instruments.py)
    insts = sorted({ins_of[id(e)].key for e in ex})
    f_inst = filter_instruments(ex)       # audit 17 Nov 2026: each curve from ITS instrument (was insts[0] for every filter)
    for e in ex:
        if cfg.input_level not in ins_of[id(e)].levels:
            raise SystemExit(f"{e['rootname']}: instrument {ins_of[id(e)].key} has no --input-level {cfg.input_level} (it supports {ins_of[id(e)].levels})")
    curve = FILTER_CURVE or _ins.filter_curve
    if len(insts) <= 1:                   # one instrument: the old order (digits of the name), names without digits after them by name
        filters = sorted({e["filter"] for e in ex}, key=lambda f: (0, int("".join(c for c in f if c.isdigit()))) if any(c.isdigit() for c in f) else (1, f))
        fl = [curve(_ins.INSTRUMENTS[f_inst[f]], f, cfg.n_lambda, cfg.svo_cache or None) for f in filters]
    else:                                 # several instruments: digits of names are not comparable (F435W is 0.43 um, F444W 4.4 um): by pivot wavelength
        fl = sorted([curve(_ins.INSTRUMENTS[f_inst[f]], f, cfg.n_lambda, cfg.svo_cache or None) for f in sorted({e["filter"] for e in ex})],
                    key=lambda c: (c.pivot, c.name))
        filters = [c.name for c in fl]
    fset = FilterSet(fl)
    # validation carve-out: per filter one TRAINING dither group (deterministic; global over the selected boxes)
    rng = np.random.default_rng(cfg.seed)
    val_groups = set()
    if cfg.n_val_groups > 0:
        for f in filters:
            gs = sorted({e["dither_group"] for e in ex if e["filter"] == f and e["rootname"] in train})
            if len(gs) > 3:
                val_groups.update(rng.choice(gs, size=min(cfg.n_val_groups, len(gs) - 3), replace=False).tolist())
    mjds = np.array([e["mjd"] for e in ex])
    jobs = []
    for e in ex:
        path = e["path"] if not cfg.data_root else os.path.join(cfg.data_root, e["file"])
        rpath = None
        if cfg.input_level == "rate":     # count-rate file: JWST naming (the only instrument with a count-rate level)
            rpath = _ins.nircam_rate_path(path, e["file"], cfg.rate_dir)
        jobs.append((path, [boxes[k] for k in tiles], half_data, shifts.get(e["rootname"], (0.0, 0.0)), cfg.input_level == "rate", rpath, ins_of[id(e)]))
    nw = max(1, int(getattr(cfg, "load_workers", 1))) if multi else 1
    pin = multi and torch.device(dev).type == "cuda"

    def to_t(a):
        t = torch.from_numpy(a)
        if not multi:
            return t.to(dev)
        return t.pin_memory() if pin else t
    cuts, dets, chans, gid = [], [], [], {}
    with ThreadPoolExecutor(max_workers=nw) as pool:
        results = pool.map(lambda j: _read_cuts(*j), jobs)
        for k, (e, (rs, exc)) in enumerate(zip(ex, results)):
            if exc is not None:
                log(f"skip {e['rootname']}: {exc!r}")
                continue
            if not rs:
                log(f"skip {e['rootname']}: no usable pixels in the data region")
                continue
            path = jobs[k][0]
            ins = ins_of[id(e)]
            hf = {r.get("filter") for _, r in rs} - {None}
            if hf and hf != {e["filter"]}:
                raise SystemExit(f"{e['rootname']}: the list of exposures says filter {e['filter']}, the file header says {sorted(hf)}")
            for tile, r in rs:
                det = r["detector"] or e["detector"]
                if det not in dets:
                    dets.append(det)
                chan = ins.channel_of(det)
                if chan not in chans:
                    chans.append(chan)
                split_lab = "held" if e["rootname"] in held else ("val" if e["dither_group"] in val_groups else "train")
                host = {"d": to_t(r["d"]), "sig": to_t(r["sig"]), "valid": to_t(r["valid"]), "X": to_t(r["X"]), "Y": to_t(r["Y"]),
                        "Jpix": None if "Jpix" not in r else to_t(r["Jpix"])}
                ee = Exposure(name=e["rootname"] + r.get("name_suffix", ""), filt=fset.index[e["filter"]], inst=insts.index(ins.key), chan=chans.index(chan),
                              det=dets.index(det), t=0.0, mjd=float(e["mjd"]), det_shape=r["det_shape"], cut0=r["cut0"],
                              d=host["d"], sig=host["sig"], valid=host["valid"], X=host["X"], Y=host["Y"], R=rotation_of(r["J"]), split=split_lab,
                              group=e["dither_group"], shift0=shifts.get(e["rootname"], (0.0, 0.0)), stripes=ins.stripes, bkg0=r["bkg0"],
                              Jpix=host["Jpix"],
                              extra={"file": e["file"], "path": path, "J": r["J"].tolist(), "detector": det, "filter": e["filter"], "pixar_a2": r["pixar_a2"],
                                     "photmjsr": r.get("photmjsr", float("nan")), "diameter_m": ins.diameter_m, "instrument": ins.key,
                                     "rootname": e["rootname"], "visit_key": str(e.get("visit_key", "")) or str(e["dither_group"]), "chip": r.get("chip", 1), "to_sky": r.get("to_sky", 1.0), "sci_unit": r.get("sci_unit", "")},
                              tile=tile, c0=c0s[tile], host=host if multi else None)
                ee.eid = gid.setdefault(ee.name, len(gid))       # one exposure per chip (rootname + _sci<chip> for a file of several chips)
                cuts.append(ee)
            if (k + 1) % 10 == 0:
                log(f"loaded {k + 1}/{len(ex)} files ({len(cuts)} cuts)")
    if cfg.input_level == "rate" and getattr(cfg, "rate_units", "dn") == "sky":
        set_rate_to_sky(cuts, cfg, log)
    span = max(float(mjds.max() - mjds.min()), 1e-9) if len(mjds) else 1.0      # whole-field span (reported; the epoch coordinate may use per-instrument spans)
    for i, e in enumerate(cuts):
        e.cid = i
    tt = epoch_coordinates([e.mjd for e in cuts], [insts[e.inst] for e in cuts], cfg.epoch_min_days, bool(getattr(cfg, "epoch_per_instrument", 1)),
                           ref_mjd=mjds, ref_inst=[ins_of[id(e)].key for e in ex])
    for e, t_ in zip(cuts, tt):
        e.t = t_
    # one representative cut per global exposure (most valid pixels): source of the per-exposure metadata (R, sigbar, bkg0, J)
    reps = {}
    for e in cuts:
        nv = int(e.valid.sum())
        if e.eid not in reps or nv > reps[e.eid][0]:
            reps[e.eid] = (nv, e)
    rep_list = [reps[i][1] for i in range(len(gid))]
    cuts_by_eid = [[] for _ in range(len(gid))]
    for e in cuts:
        cuts_by_eid[e.eid].append(e)
    info = {"box": box, "boxes": [boxes[k] for k in tiles], "tiles": tiles, "tile_c0": {k: list(c0s[k]) for k in tiles}, "half_data": half_data,
            "dets": dets, "chans": chans, "insts": insts, "filters": filters,
            "n_by_split": {s: sum(1 for e in rep_list if e.split == s) for s in ("train", "val", "held")}, "val_groups": sorted(val_groups),
            "epoch_span_days": span, "shifts_applied": sum(1 for e in rep_list if e.shift0 != (0.0, 0.0)), "n_exposures": len(rep_list),
            "n_cuts": len(cuts), "cuts_per_tile": {k: {s: sum(1 for e in cuts if e.tile == k and e.split == s) for s in ("train", "val", "held")} for k in tiles},
            "host_gb": sum(e.nbytes() for e in cuts) / 2 ** 30, "multi_tile": multi}
    meta = build_meta(rep_list, dets, chans, insts, cuts_by_eid)
    meta["det_xi_box"] = det_footprints([e for e in cuts if e.split == "train"])
    if multi:
        meta["aff_ref"] = affine_refs(cuts_by_eid)
    return cuts, fset, meta, info


def set_rate_to_sky(cuts, cfg, log=print):
    """--input-level rate --rate-units sky: per exposure (cut), the conversion C (MJy/sr per DN/s) and the nominal pixel area a_nom [arcsec^2],
    stored as extra["rate_kappa"] = 1 / (a_nom C), so that count rate per pixel = sky [MJy/sr] x a_pix x rate_kappa (x learned terms).
    C: PHOTMJSR of the matching _cal file when --rate-wcs cal (its header is read for the positions anyway), else the --rate-photom table
    {"DETECTOR/FILTER": C}; a_nom: PIXAR_A2 of the header read, else the table's "DETECTOR/FILTER/area", else the median header pixel area of
    the cut (|det J|). Refuses to run without a conversion for every exposure."""
    tab = {}
    if getattr(cfg, "rate_photom", ""):
        tab = {str(k).upper(): float(v) for k, v in json.load(open(cfg.rate_photom)).items()}
    use_hdr = getattr(cfg, "rate_wcs", "cal") == "cal"
    if not use_hdr and not tab:
        raise SystemExit("--rate-units sky needs a conversion: --rate-wcs cal (PHOTMJSR of the _cal files) or --rate-photom FILE")
    seen = {}
    for e in cuts:
        key = f"{e.extra['detector']}/{e.extra['filter']}".upper()
        C = float(e.extra.get("photmjsr", float("nan"))) if use_hdr else float("nan")
        src = "header PHOTMJSR"
        if not (math.isfinite(C) and C > 0):
            C, src = tab.get(key, float("nan")), "--rate-photom"
        if not (math.isfinite(C) and C > 0):
            raise SystemExit(f"--rate-units sky: no conversion for {e.name} ({key}): neither a PHOTMJSR in its _cal header nor an entry in --rate-photom")
        a = float(e.extra.get("pixar_a2", float("nan")))
        if not (math.isfinite(a) and a > 0):
            a = tab.get(key + "/AREA", float("nan"))
        if not (math.isfinite(a) and a > 0):
            a = float(abs(np.linalg.det(np.asarray(e.extra["J"], np.float64))))
        e.extra["rate_to_sky"], e.extra["nominal_area"], e.extra["rate_kappa"] = C, a, 1.0 / (a * C)
        seen.setdefault(key, (C, a, src))
    log("count rate in sky units: conversion (MJy/sr per DN/s), nominal pixel area (arcsec^2), source per detector/filter: "
        + ", ".join(f"{k} {v[0]:.4g} {v[1]:.5g} ({v[2]})" for k, v in sorted(seen.items())))


def det_footprints(cuts):
    """Per detector index: [x0, x1, y0, y1] of the valid pixels of the given cuts in normalised detector coordinates (xi in [-1, 1])."""
    out = {}
    for c in cuts:
        v = c.hostv("valid")
        rows = torch.nonzero(v.any(1))[:, 0]
        cols = torch.nonzero(v.any(0))[:, 0]
        if rows.numel() == 0:
            continue
        ny, nx = c.det_shape
        x0 = (int(cols.min()) + c.cut0[1]) / (nx - 1) * 2 - 1
        x1 = (int(cols.max()) + c.cut0[1]) / (nx - 1) * 2 - 1
        y0 = (int(rows.min()) + c.cut0[0]) / (ny - 1) * 2 - 1
        y1 = (int(rows.max()) + c.cut0[0]) / (ny - 1) * 2 - 1
        b = out.get(c.det)
        out[c.det] = [x0, x1, y0, y1] if b is None else [min(b[0], x0), max(b[1], x1), min(b[2], y0), max(b[3], y1)]
    return out


def affine_refs(cuts_by_eid):
    """Per exposure: centroid of its valid pixels over all its cuts in the FIELD frame (header X, Y + tile offset c0) [arcsec] (F32)."""
    out = []
    for cs in cuts_by_eid:
        sx, sy, n = 0.0, 0.0, 0
        for c in cs:
            v = c.hostv("valid")
            k = int(v.sum())
            sx += float(c.hostv("X")[v].sum()) + k * c.c0[0]
            sy += float(c.hostv("Y")[v].sum()) + k * c.c0[1]
            n += k
        out.append([sx / max(n, 1), sy / max(n, 1)])
    return out


def build_meta(exps, dets, chans, insts, cuts_by_eid=None):
    """Per GLOBAL exposure metadata from one representative cut per exposure (exps[i].eid == i); `cuts_by_eid` (optional) lists every cut of
    each exposure (detector rows / columns used, for the stripe gauge)."""
    cb = cuts_by_eid if cuts_by_eid is not None else [[e] for e in exps]
    # telescope diameter per exposure, by the same rule as the kernel warm start (cli.filt_info_of), so that the blur gauge (model.py) and
    # the warm start use one value (audit 17 Nov 2026: the blur gauge read a key that was never set and always used 6.5 m)
    diam = [float(e.extra.get("diameter_m", INST_DIAM.get(insts[e.inst] if e.inst < len(insts) else "", 6.5))) for e in exps]
    return {"diameter_m_e": diam, "filt": [e.filt for e in exps], "inst": [e.inst for e in exps], "chan": [e.chan for e in exps], "det": [e.det for e in exps],
            "t": [e.t for e in exps], "train": [e.split == "train" for e in exps], "R": np.stack([e.R for e in exps]),
            "sigbar": [float(e.sig[e.valid].median()) for e in exps], "det_shape": [list(e.det_shape) for e in exps],
            "shift0": [list(e.shift0) for e in exps], "stripes": [e.stripes for e in exps], "bkg0": [e.bkg0 for e in exps],
            "n_det": len(dets), "n_inst": len(insts), "n_chan": len(chans), "det_names": list(dets),
            "cut": [(int(e.cut0[0]), int(e.cut0[1]), int(e.shape[0]), int(e.shape[1])) for e in exps],
            "cuts": [[(int(c.cut0[0]), int(c.cut0[1]), int(c.shape[0]), int(c.shape[1])) for c in cs] for cs in cb],
            "chan_pix": [float(np.median([abs(np.linalg.det(np.asarray(e.extra["J"]))) ** 0.5 for e in exps if e.chan == c])) for c in range(len(chans))],
            "rate_kappa": [float(e.extra["rate_kappa"]) for e in exps] if all("rate_kappa" in e.extra for e in exps) else None}


# ----------------------------------------------------------------------------- sky warm start
def pixel_area_of(e):
    """Header pixel area [arcsec^2] per pixel of a cut, (ny, nx): |det Jpix| when the per-pixel Jacobian is loaded, else |det J| of the mean
    header Jacobian (the same factor the rate-level model applies at zero learned geometry, F15)."""
    if e.Jpix is not None:
        J = e.Jpix
        return (J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]).abs()
    return torch.full(e.d.shape, float(abs(np.linalg.det(np.asarray(e.extra["J"], np.float64)))), device=e.d.device)


def splat_coadd(exps, nf, N, pitch, dev, extra_shift=None, per_area=False):
    """Inverse-variance bilinear splat of (d - bkg0) of the given exposures at their header positions (+ shift0) onto the sky nodes,
    per filter: (nf, N, N) and the weight maps. Used only for the warm start of the sky and the bright-patch strata.
    per_area (--input-level rate): data and sigma are divided by the header pixel area first (and, with --rate-units sky, multiplied by the
    nominal area x the conversion, i.e. divided by rate_kappa too), so that the warm start is in the units the rate-level model gives the sky
    (count rate per arcsec^2, or MJy/sr with --rate-units sky; the model multiplies by the pixel area and rate_kappa, F15). Without it the warm start of a
    rate-level fit is low by the pixel area (0.001 to 0.004 arcsec^2 on NIRCam), see RATE_HELDOUT_DIAGNOSIS.md."""
    num = torch.zeros(nf, N * N, device=dev)
    den = torch.zeros(nf, N * N, device=dev)
    x0 = -(N - 1) / 2 * pitch
    for e in exps:
        v = e.valid
        es = (0.0, 0.0) if extra_shift is None else extra_shift.get(e.eid, (0.0, 0.0))
        X = (e.X[v] + e.shift0[0] + es[0] - x0) / pitch
        Y = (e.Y[v] + e.shift0[1] + es[1] - x0) / pitch
        if per_area:
            a = pixel_area_of(e)[v] * float(e.extra.get("rate_kappa", 1.0))    # --rate-units sky: x 1 / (nominal area x conversion)
            w = (a / e.sig[v]) ** 2
            val = (e.d[v] - e.bkg0) / a * w
        else:
            w = 1.0 / e.sig[v] ** 2
            val = (e.d[v] - e.bkg0) * w
        ix, iy = X.floor().long(), Y.floor().long()
        fx, fy = X - ix, Y - iy
        for dy in (0, 1):
            for dx in (0, 1):
                jx, jy = ix + dx, iy + dy
                ww = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                ok = (jx >= 0) & (jx < N) & (jy >= 0) & (jy < N)
                idx = (jy * N + jx)[ok]
                num[e.filt].index_add_(0, idx, (val * ww)[ok])
                den[e.filt].index_add_(0, idx, (w * ww)[ok])
    img = torch.where(den > 0, num / den.clamp(min=1e-30), torch.zeros_like(num))
    return img.view(nf, N, N), den.view(nf, N, N)
