"""Instrument descriptions: everything the package assumes about one instrument, so that a new instrument is a data description plus a reader.

One `Instrument` per camera (key used in the "instrument" field of the list of exposures): today NIRCAM (JWST NIRCam), WFC3_IR (HST WFC3
infrared channel) and ACS_WFC (HST ACS wide-field channel, two chips). The rest of the package reads instrument facts only from here (the
reader, the data-quality policy, the telescope diameter, the detector and channel names, the stripes switch, the minimum number of valid
pixels, the filter-curve identifier, the units) and the spectral domain of the sky and the filter-curve input grid are derived from the
loaded filter curves (`spectral_setup`).

The common exposure record (one dict per chip = science extension; a file with two chips gives two records and two exposures):
    sci, err            float32 (ny, nx) science plane and its 1-sigma error, in the file's unit `sci_unit`
    dq                  integer (ny, nx) data-quality plane, or None when `usable` is given (torchfits reader)
    dq_ok               function dq -> bool mask: True where the instrument's policy keeps the pixel (its own bit meanings)
    w                   the header solution: an object with pixel_to_world_differentiable((n, 2) pixels, origin=0) -> (n, 2) degrees
    detector            detector name, unique over all instruments of a run (NRCB1, WFC3_IR, ACS_WFC1, ACS_WFC2)
    instrument          instrument key (NIRCAM, WFC3_IR, ACS_WFC)
    chip                science extension version (1, 2), name_suffix "" for a one-chip file, "_sci<chip>" otherwise
    to_sky              factor from `sci_unit` to MJy/sr (surface brightness per nominal pixel area): S = to_sky * sci
    exptime, filter, wcs_name, saturation (text)

The conversion to surface brightness, one formula per instrument (see `to_sky_factor`):
    JWST calibrated (_cal, MJy/sr):   to_sky = 1
    HST (_flt / _flc):                to_sky = PHOTFNU * 1e-6 / (Omega_nom * t_unit)       [MJy/sr per (e- or e-/s)]
        PHOTFNU [Jy s / e-] = 1e23 PHOTFLAM lambda_p^2 / c  when the header has no PHOTFNU (PHOTFLAM [erg cm^-2 A^-1 / e-], lambda_p = PHOTPLAM [A],
        c = 2.99792458e18 A/s); Omega_nom = IDCSCALE^2 [arcsec^2] in sr; t_unit = EXPTIME [s] when BUNIT = ELECTRONS, else 1.
HST _flt / _flc and JWST _cal are flat-fielded so that a uniform sky gives a uniform image: both are surface brightness per NOMINAL pixel area
(the pixel-area map is not applied), which is what the model's calibrated level assumes. HST files therefore enter at the calibrated level only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

C_ANG_PER_S = 2.99792458e18
ARCSEC2_TO_SR = (math.pi / 648000.0) ** 2

# reference spectral domain of the sky and filter-curve input grid (the constants used before 18 Nov 2026; sky.LAM_LO / LAM_HI, bandpass.CURVE_*)
REF_LAM = (0.70, 5.30)
REF_CURVE = (0.55, 5.6, 48)
NODE_MARGIN = 0.02        # the derived spectral domain covers every quadrature node x (1 +- 2 %)


# ----------------------------------------------------------------------------- data-quality policies
def dq_ok_jwst(ignore_bits):
    """JWST: every bit rejects except `ignore_bits` (--dq-ignore-bits; 0 = strict DQ == 0, the default). 32-bit DQ."""
    ig = int(ignore_bits)

    def ok(q):
        return (q.astype(np.int64) & ~ig) == 0
    ok.ignore_bits = ig
    return ok


def dq_ok_16bit(ignore_bits):
    """HST: 16-bit DQ (int16 in the file; bit 32768 read as unsigned); every bit rejects except `ignore_bits`."""
    ig = int(ignore_bits) & 0xFFFF

    def ok(q):
        return ((q.astype(np.int64) & 0xFFFF) & ~ig & 0xFFFF) == 0
    ok.ignore_bits = ig
    return ok


# bit meanings (value: meaning) as documented by each pipeline; `ignore` = bits that do NOT reject the pixel by default
JWST_DQ_BITS = {1: "do not use", 2: "saturated", 4: "jump (cosmic ray) detected", 8: "dropout", 16: "outlier", 32: "persistence", 64: "AD floor",
                128: "charge loss", 256: "unreliable error", 512: "non-science", 1024: "dead", 2048: "hot", 4096: "warm", 8192: "low QE",
                16384: "RC", 32768: "telegraph", 65536: "non-linear", 131072: "bad reference pixel", 262144: "no flat field", 524288: "no gain value",
                1048576: "no linearity correction", 2097152: "no saturation check", 4194304: "unreliable bias", 8388608: "unreliable dark",
                16777216: "unreliable slope", 33554432: "unreliable flat", 67108864: "open", 134217728: "adjacent to open", 268435456: "unreliable reset",
                536870912: "MSA failed open", 1073741824: "other bad pixel", 2147483648: "reference pixel"}
WFC3_IR_DQ_BITS = {1: "Reed-Solomon decoding error", 2: "data replaced by fill value", 4: "bad detector pixel", 8: "deviant zero read (bias)",
                   16: "hot pixel", 32: "unstable response", 64: "warm pixel", 128: "bad reference pixel", 256: "full-well saturation",
                   512: "bad or uncertain flat value", 1024: "reserved", 2048: "signal in zero read", 4096: "cosmic ray flagged by AstroDrizzle",
                   8192: "cosmic ray detected in up-the-ramp fit", 16384: "ghost or crosstalk", 32768: "reserved"}
ACS_WFC_DQ_BITS = {1: "Reed-Solomon decoding error", 2: "data replaced by fill value", 4: "bad detector pixel", 8: "masked by aperture feature",
                   16: "hot pixel", 32: "charge-transfer tail", 64: "warm pixel", 128: "bad column", 256: "full-well saturation",
                   512: "bad or uncertain flat value", 1024: "charge trap (sink pixel)", 2048: "analogue-to-digital saturation",
                   4096: "cosmic ray flagged by AstroDrizzle", 8192: "cosmic ray flagged by the pipeline's rejection", 16384: "reserved", 32768: "reserved"}
# HST defaults follow drizzlepac's final_bits for these detectors (WFC3/IR 528 = 16 + 512) without the saturated bit (ACS 80 = 16 + 64, HAP
# uses 336 = 16 + 64 + 256): a saturated pixel is not linear and the model has no non-linear detector term.
WFC3_IR_IGNORE = 16 + 512
ACS_WFC_IGNORE = 16 + 64

DQ_IGNORE_OVERRIDE = {}   # --dq-ignore INST:BITS,...: replaces an instrument's default ignore bits (set by cli.main)


# ----------------------------------------------------------------------------- the description
@dataclass
class Instrument:
    key: str                          # NIRCAM, WFC3_IR, ACS_WFC: the "instrument" value of the list of exposures (aliases in ALIASES)
    telescope: str
    diameter_m: float                 # aperture diameter: kernel warm start and blur gauge (0.42 lambda / D)
    pixel_scale: dict                 # nominal pixel scale [arcsec] per channel
    detectors: tuple                  # detector names this instrument produces
    channel_of: Callable              # detector name -> channel name (kernel render pitch and level grids are per channel)
    sci_units: dict                   # level -> unit of the science plane (only "cal")
    dq_bits: dict                     # bit value -> meaning
    dq_ignore_default: int            # bits that do not reject a pixel unless --dq-ignore changes them
    dq_kind: str                      # "jwst32" or "hst16"
    stripes: bool                     # per-exposure row / column stripes (detector terms)
    min_valid_pixels: int             # a cut with fewer valid pixels is dropped
    bandpass_prefix: str              # filter-curve identifier = prefix + filter name (SVO Filter Profile Service id)
    reader: Callable                  # reader(path, inst) -> list of exposure records (one per chip)
    saturation: str = ""              # what the files carry about saturation and non-linearity
    notes: tuple = field(default_factory=tuple)

    def bandpass_id(self, filt):
        return f"{self.bandpass_prefix}{filt}"

    def dq_ignore_bits(self):
        return int(DQ_IGNORE_OVERRIDE.get(self.key, self.dq_ignore_default if self.key != "NIRCAM" else _jwst_ignore()))

    def dq_policy(self):
        ig = self.dq_ignore_bits()
        return dq_ok_jwst(ig) if self.dq_kind == "jwst32" else dq_ok_16bit(ig)

    def dq_reject_names(self):
        ig = self.dq_ignore_bits()
        return [v for b, v in sorted(self.dq_bits.items()) if not (b & ig)]

    def read(self, path):
        return self.reader(path, self)


def _jwst_ignore():
    from . import data
    return int(data.DQ_IGNORE_BITS)          # --dq-ignore-bits (kept as NIRCam's own ignore bits; was global before)


# ----------------------------------------------------------------------------- readers
def read_nircam(path, inst):
    """JWST NIRCam: the package's existing reader (data.read_exposure_file: astropy or torchfits, header WCS of the _cal file, or the telescope
    frame), unchanged, plus the common-record fields. One chip per file."""
    from . import data
    raw = data.read_exposure_file(path)
    raw.update(chip=1, name_suffix="", instrument="NIRCAM",
               sci_unit="MJy/sr", dq_ok=inst.dq_policy(), min_valid=inst.min_valid_pixels)
    raw["to_sky"] = 1.0                      # the file is already MJy/sr
    if not raw.get("detector"):
        raw["detector"] = ""
    raw.setdefault("saturation", inst.saturation)
    return [raw]


def _hget(sh, h0, key, default=None):
    """HST keyword: the SCI header first, then the primary (the PHOT* keywords sit in either, depending on instrument and pipeline version)."""
    for h in (sh, h0):
        if key in h and h[key] not in (None, ""):
            return h[key]
    return default


def hst_filter_name(h0):
    for k in ("FILTER", "FILTER1", "FILTER2"):
        v = str(h0.get(k, "") or "").strip().upper()
        if v and not v.startswith("CLEAR") and v != "N/A":
            return v
    return "CLEAR"


def to_sky_factor(bunit, exptime, photfnu=None, photflam=None, photplam=None, nominal_scale=None):
    """HST: factor from the science plane to MJy/sr per nominal pixel area, and how it was obtained.

    to_sky = PHOTFNU * 1e-6 / (Omega_nom * t_unit), PHOTFNU = 1e23 PHOTFLAM PHOTPLAM^2 / c if absent, Omega_nom = nominal_scale^2 [sr],
    t_unit = EXPTIME for BUNIT ELECTRONS, 1 for ELECTRONS/S."""
    b = str(bunit).strip().upper()
    if b == "ELECTRONS/S":
        t_unit = 1.0
    elif b == "ELECTRONS":
        if not exptime or exptime <= 0:
            raise ValueError("BUNIT ELECTRONS without a positive EXPTIME")
        t_unit = float(exptime)
    else:
        raise ValueError(f"unsupported BUNIT {bunit!r} (HST reader: ELECTRONS or ELECTRONS/S)")
    src = "PHOTFNU"
    if photfnu is None:
        if photflam is None or photplam is None:
            raise ValueError("neither PHOTFNU nor PHOTFLAM and PHOTPLAM in the header")
        photfnu = 1e23 * float(photflam) * float(photplam) ** 2 / C_ANG_PER_S
        src = "1e23 PHOTFLAM PHOTPLAM^2 / c"
    omega = float(nominal_scale) ** 2 * ARCSEC2_TO_SR
    return float(photfnu) * 1e-6 / (omega * t_unit), {"photfnu_jy_s_per_e": float(photfnu), "photfnu_source": src, "omega_nom_sr": omega,
                                                      "t_unit_s": t_unit, "bunit": b}


HST_WCS = "apriori"   # --hst-wcs: "apriori" (guide-star solution, as the JWST _cal files) or "archive" (the primary WCS, usually aligned to Gaia)


def hst_apriori_key(hdr):
    """Alternate-WCS key of the a priori solution: the guide-star catalogue correction ('IDC_x-GSC...') if present, else the plain 'IDC_x'. Never
    an a posteriori '-FIT_*' or '-HSC*' solution (those were derived by registering the images: a pre-alignment the fit must not use)."""
    alts = {k[-1]: str(hdr[k]) for k in hdr if k.startswith("WCSNAME") and len(k) == 8}
    gsc = [k for k, v in sorted(alts.items()) if "-GSC" in v and "FIT" not in v]
    plain = [k for k, v in sorted(alts.items()) if v.startswith("IDC_") and "-" not in v]
    return (gsc or plain or [None])[0]


class AstropyWCS:
    """Header solution of an HST chip through astropy with every distortion term (SIP polynomial, detector-to-image and non-polynomial look-up
    tables: the D2IMARR / WCSDVARR extensions of ACS). Offers the one method the package calls, pixel_to_world_differentiable. Not differentiable
    (numpy); the package only evaluates it once per pixel at load time."""

    def __init__(self, hdul, ver, policy="apriori"):
        import warnings

        from astropy.wcs import WCS, FITSFixedWarning
        hdr = hdul[("SCI", ver)].header
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(hdr, fobj=hdul)
            self.name = str(hdr.get("WCSNAME", ""))
            if policy == "apriori":
                key = hst_apriori_key(hdr)
                if key is None:
                    raise ValueError(f"no a priori alternate WCS (WCSNAME {self.name}); use --hst-wcs archive to accept the primary solution")
                if str(hdr.get("WCSNAME", "")).split("-")[0] != str(hdr["WCSNAME" + key]).split("-")[0]:
                    raise ValueError("the a priori and primary solutions use different distortion models")
                wa = WCS(hdr, key=key, relax=True, fobj=hdul)
                w.wcs.crval = wa.wcs.crval
                w.wcs.crpix = wa.wcs.crpix
                w.wcs.cd = wa.wcs.cd if wa.wcs.has_cd() else wa.wcs.get_pc() * wa.wcs.cdelt[:, None]
                w.wcs.set()
                self.name = str(hdr["WCSNAME" + key])
        self.w = w
        self.has_lookup = (w.cpdis1 is not None) or (w.det2im1 is not None)

    def pixel_to_world_differentiable(self, pixels, origin=0):
        import torch
        p = np.asarray(pixels.detach().cpu().numpy() if hasattr(pixels, "detach") else pixels, np.float64)
        ra, dec = self.w.all_pix2world(p[:, 0], p[:, 1], int(origin))
        return torch.tensor(np.stack([ra, dec], -1), dtype=torch.float64)


def read_hst(path, inst):
    """HST WFC3/IR _flt and ACS/WFC _flc (and _flt): one record per SCI extension (chip). Science and error stay in the file unit; to_sky is applied
    by the loader (calibrated level)."""
    from astropy.io import fits
    out = []
    with fits.open(path, memmap=False) as h:
        h0 = h[0].header
        filt = hst_filter_name(h0)
        vers = [x.ver for x in h if x.name == "SCI"]
        for ver in vers:
            sh = h[("SCI", ver)].header
            scale = sh.get("IDCSCALE", None)
            scale_src = "IDCSCALE"
            if scale is None:
                scale, scale_src = inst.pixel_scale[inst.channel_of("")], "nominal table"
            exptime = float(_hget(sh, h0, "EXPTIME", 0.0) or 0.0)
            f, how = to_sky_factor(_hget(sh, h0, "BUNIT", ""), exptime, _hget(sh, h0, "PHOTFNU"), _hget(sh, h0, "PHOTFLAM"), _hget(sh, h0, "PHOTPLAM"), scale)
            w = AstropyWCS(h, ver, HST_WCS)
            chip = sh.get("CCDCHIP", None)
            det = inst.detectors[0] if len(inst.detectors) == 1 else f"{inst.key}{int(chip) if chip is not None else ver}"
            out.append(dict(sci=h[("SCI", ver)].data.astype(np.float32), err=h[("ERR", ver)].data.astype(np.float32), dq=h[("DQ", ver)].data,
                            w=w, detector=det, instrument=inst.key,
                            chip=int(ver), name_suffix="" if len(vers) == 1 else f"_sci{ver}", sci_unit=how["bunit"], to_sky=f,
                            to_sky_how={**how, "nominal_scale_arcsec": float(scale), "nominal_scale_source": scale_src}, dq_ok=inst.dq_policy(),
                            min_valid=inst.min_valid_pixels, exptime=exptime, filter=filt, wcs_name=w.name, wcs_lookup_tables=w.has_lookup,
                            photplam_A=float(_hget(sh, h0, "PHOTPLAM", float("nan"))), photbw_A=float(_hget(sh, h0, "PHOTBW", float("nan"))),
                            saturation=inst.saturation))
    return out


# ----------------------------------------------------------------------------- the registry
_NIRCAM_CHAN = {**{f"NRC{m}{i}": "NIRCAM_SW" for m in "AB" for i in "1234"}, **{f"NRC{m}LONG": "NIRCAM_LW" for m in "AB"}}

INSTRUMENTS = {
    "NIRCAM": Instrument(
        key="NIRCAM", telescope="JWST", diameter_m=6.5, pixel_scale={"NIRCAM_SW": 0.031, "NIRCAM_LW": 0.063}, detectors=tuple(_NIRCAM_CHAN),
        channel_of=lambda d: _NIRCAM_CHAN.get(d, d), sci_units={"cal": "MJy/sr"}, dq_bits=JWST_DQ_BITS,
        dq_ignore_default=0, dq_kind="jwst32", stripes=True, min_valid_pixels=2000, bandpass_prefix="JWST/NIRCam.", reader=read_nircam,
        saturation="DQ bit 2 (saturated); the pipeline's linearity step corrects non-linearity below saturation; no SATURATE keyword used",
        notes=("--dq-ignore-bits is NIRCam's own ignore mask (it was global before 18 Nov 2026)",)),
    "WFC3_IR": Instrument(
        key="WFC3_IR", telescope="HST", diameter_m=2.4, pixel_scale={"WFC3_IR": 0.12825}, detectors=("WFC3_IR",), channel_of=lambda d: "WFC3_IR",
        sci_units={"cal": "ELECTRONS/S"}, dq_bits=WFC3_IR_DQ_BITS, dq_ignore_default=WFC3_IR_IGNORE, dq_kind="hst16", stripes=False,
        min_valid_pixels=2000, bandpass_prefix="HST/WFC3_IR.", reader=read_hst,
        saturation="DQ 256 (full well) and 2048 (signal in zero read); calwf3 corrects non-linearity (NLINCORR) and fits the ramp from unsaturated reads",
        notes=("a priori WCS (guide-star solution) by default; SIP distortion only (no look-up tables on WFC3/IR)",)),
    "ACS_WFC": Instrument(
        key="ACS_WFC", telescope="HST", diameter_m=2.4, pixel_scale={"ACS_WFC": 0.05}, detectors=("ACS_WFC1", "ACS_WFC2"), channel_of=lambda d: "ACS_WFC",
        sci_units={"cal": "ELECTRONS"}, dq_bits=ACS_WFC_DQ_BITS, dq_ignore_default=ACS_WFC_IGNORE, dq_kind="hst16", stripes=False,
        min_valid_pixels=2000, bandpass_prefix="HST/ACS_WFC.", reader=read_hst,
        saturation="DQ 256 (full well) and 2048 (analogue-to-digital); no non-linearity correction below full well (linear to about 1 % there)",
        notes=("two chips = two exposures (own detector, own per-exposure terms)", "single exposures carry cosmic rays: DQ 4096 / 8192 reject them only "
               "where the pipeline flagged them",)),
}
ALIASES = {"WFC3": "WFC3_IR", "WFC3/IR": "WFC3_IR", "ACS": "ACS_WFC", "ACS/WFC": "ACS_WFC", "NIRCAM": "NIRCAM"}


def resolve(entry_or_name, detector=None):
    """Instrument description of an entry of the list of exposures ("instrument", default NIRCAM; WFC3 with detector IR, ACS with WFC)."""
    if isinstance(entry_or_name, dict):
        name, detector = str(entry_or_name.get("instrument", "NIRCAM")).strip().upper(), entry_or_name.get("detector", None)
    else:
        name = str(entry_or_name).strip().upper()
    if name in INSTRUMENTS:
        return INSTRUMENTS[name]
    if detector and f"{name}_{str(detector).upper()}" in INSTRUMENTS:
        return INSTRUMENTS[f"{name}_{str(detector).upper()}"]
    if name in ALIASES and (not detector or str(detector).upper() in ("IR", "WFC", "") or name == "NIRCAM"):
        return INSTRUMENTS[ALIASES[name]]
    raise SystemExit(f"unknown instrument {name!r}: known {sorted(INSTRUMENTS)} (aliases {sorted(ALIASES)}); add a description to fieldop/instruments.py")


def parse_dq_ignore(spec):
    """--dq-ignore 'WFC3_IR:528,ACS_WFC:80' -> {key: bits}."""
    out = {}
    for item in (spec or "").split(","):
        if item.strip():
            k, v = item.split(":")
            out[resolve(k.strip()).key] = int(v, 0)
    return out


def diameters():
    """{instrument key and alias: diameter [m]} (data.INST_DIAM)."""
    d = {k: v.diameter_m for k, v in INSTRUMENTS.items()}
    d.update({a: INSTRUMENTS[k].diameter_m for a, k in ALIASES.items()})
    return d


# ----------------------------------------------------------------------------- filter curves
def filter_curve(inst, filt, Q, cache_dir=None):
    """The published curve of `filt` of instrument `inst` (SVO id inst.bandpass_id(filt)) through torchsky's cached reader. No network on a run
    without the cache file: the cache must hold filter_<id with / replaced by _>.json (torchsky's naming) in --svo-cache, $TORCHSKY_SVO_CACHE or
    ~/.cache/torchsky/svo."""
    from .bandpass import svo_filter
    return svo_filter(filt, inst.key, inst.bandpass_id(filt), Q, cache_dir)


def curve_cache_file(svo_id):
    import re
    return "filter_" + re.sub(r"[^A-Za-z0-9_.-]", "_", svo_id.strip()) + ".json"


def stand_in_filter(filt, inst_key, pivot_um, rms_width_um, Q):
    """Stand-in curve for tests without the curve cache: a flat-topped box of full width sqrt(12) x the rms width (PHOTBW), centred on the pivot
    (PHOTPLAM). Not used by any real run."""
    from .bandpass import box_filter
    half = 0.5 * math.sqrt(12.0) * rms_width_um
    return box_filter(filt, inst_key, pivot_um - half, pivot_um + half, Q)


# ----------------------------------------------------------------------------- spectral domain from the loaded bandpasses (audit finding 4)
def spectral_setup(fset, cfg, log=print):
    """Spectral domain of the sky basis and of the kernel / throughput wavelength coordinate, filter-curve input grid and number of spectral
    coefficients K, from the loaded filters.

    --spectral-domain union (default): domain = hull(REF_LAM, [min node / (1 + m), max node x (1 + m)]) over every filter's quadrature nodes,
    m = 2 %; curve grid = hull(REF_CURVE range, union of the curves' supports T >= 1e-3 max), its point count grown at the reference log density
    (48 points over ln(5.6 / 0.55)). With every node inside 0.70-5.30 um and every support inside 0.55-5.6 um (all NIRCam wide filters of the
    SMACS field; the synthetic fields) both are exactly the old constants. 'fixed': the old constants whatever is loaded.
    --spec-k-rule fixed (default): K = --spec-k. 'span': K = 1 + round((--spec-k - 1) ln(hi / lo) / ln(5.30 / 0.70)), the same bump spacing in
    ln(lambda) as 12 bumps over the reference domain (0.184 in ln lambda, 0.080 dex; bump width = spacing), so K = 12 for the reference domain.
    Sets sky.SPECTRAL_DOMAIN, the filters' curve vectors, and cfg.spec_k; returns a record for the log and metrics."""
    from . import sky as _sky
    mode = getattr(cfg, "spectral_domain", "union")
    nodes = np.concatenate([np.asarray(f.nodes, np.float64) for f in fset.filters])
    sup = []
    for f in fset.filters:
        keep = f.T >= 1e-3 * f.T.max()
        sup.append((float(f.lam_um[keep].min()), float(f.lam_um[keep].max())))
    s_lo, s_hi = min(s[0] for s in sup), max(s[1] for s in sup)
    if mode == "fixed":
        lo, hi = REF_LAM
        c_lo, c_hi, c_n = REF_CURVE
    else:
        lo = min(REF_LAM[0], float(nodes.min()) / (1 + NODE_MARGIN))
        hi = max(REF_LAM[1], float(nodes.max()) * (1 + NODE_MARGIN))
        c_lo, c_hi = min(REF_CURVE[0], s_lo), max(REF_CURVE[1], s_hi)
        if (c_lo, c_hi) == REF_CURVE[:2]:
            c_n = REF_CURVE[2]
        else:
            c_n = 1 + int(math.ceil((REF_CURVE[2] - 1) * math.log(c_hi / c_lo) / math.log(REF_CURVE[1] / REF_CURVE[0])))
    k_req = int(cfg.spec_k)
    rule = getattr(cfg, "spec_k_rule", "fixed")
    if rule == "span":
        k = 1 + int(round((k_req - 1) * math.log(hi / lo) / math.log(REF_LAM[1] / REF_LAM[0])))
    else:
        k = k_req
    _sky.set_spectral_domain(lo, hi)
    changed_grid = fset.set_curve_grid(c_lo, c_hi, c_n)
    cfg.spec_k = k
    rec = {"mode": mode, "domain_um": [lo, hi], "reference_domain_um": list(REF_LAM), "node_range_um": [float(nodes.min()), float(nodes.max())],
           "curve_grid_um": [c_lo, c_hi, c_n], "curve_support_um": [s_lo, s_hi], "curve_grid_changed": changed_grid, "spec_k_rule": rule,
           "spec_k_requested": k_req, "spec_k": k, "ln_spacing": math.log(hi / lo) / max(k - 1, 1),
           "outside_old_domain": [f.name for f in fset.filters if f.nodes.min() < REF_LAM[0] or f.nodes.max() > REF_LAM[1]]}
    log(f"spectral domain {lo:.4f}-{hi:.4f} um ({mode}; nodes {nodes.min():.4f}-{nodes.max():.4f} um), K = {k} ({rule}, bump spacing "
        f"{rec['ln_spacing']:.3f} in ln lambda), filter-curve grid {c_lo:.4f}-{c_hi:.4f} um x {c_n}"
        + (f"; filters with nodes outside the old 0.70-5.30 um domain: {rec['outside_old_domain']}" if rec["outside_old_domain"] else ""))
    return rec


def describe():
    """Plain table of every instrument (for the report and the log)."""
    rows = []
    for k, i in INSTRUMENTS.items():
        rows.append({"key": k, "telescope": i.telescope, "diameter_m": i.diameter_m, "pixel_scale": i.pixel_scale, "detectors": list(i.detectors),
                     "units": i.sci_units, "dq_ignore": i.dq_ignore_bits(), "dq_reject": i.dq_reject_names(),
                     "stripes": i.stripes, "min_valid_pixels": i.min_valid_pixels, "bandpass_example": i.bandpass_id("F150W"), "saturation": i.saturation})
    return rows
