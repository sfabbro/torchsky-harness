"""SMACS J0723: four ways to make a multiband sky image from the same JWST/NIRCam exposures, all torchsky.

  SWarp-like : torchsky Lanczos3 resample + inverse-variance weighted mean
  Drizzle    : torchsky drizzle (square kernel, pixfrac 0.8)
  Explicit   : pixel-grid sky s_b + parametric PSF (Gaussian + learned residual kernel)
               + per-exposure plane background + per-exposure affine geometry, fitted to detector pixels
  Implicit   : sky = multi-resolution feature-grid neural field with 6 band heads, PSF = MLP field,
               background = MLP field, geometry learned; same detector likelihood

Forward model for exposure i in band b at detector pixel p:
    m_i(p) = B_i(p) + [K_b * s_b]( c_i + (I + M_i)(T_i(p) - c_i) + t_i )
T_i = torchsky WCS (detector pixel -> sky -> target grid), M_i, t_i = learned affine (or fixed = 0).
Loss = sum ivar (d - m)^2 / N  (+ weak TV on the explicit grid). One exposure per band is held out.

Outputs (OUT): cubes/*.fits, rgb/*.png, figure PNGs, metrics.json

v29: HST (WFC3/IR, ACS/WFC, WFC3/UVIS flt/flc) joins JWST/NIRCam in the same fit. Exposures are NOT registered: each stays in its
detector frame, the header WCS is the starting map only, the learned Geometry/PSF/sky do the rest (see patch_v29.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from astropy.io import fits

# ----------------------------------------------------------------------------- v29: band registry
# v28 kept four NIRCam-only module tables (BANDS, FWHM_ARCSEC, LAM_UM, CHANNEL). They are now the NIRCam rows of a registry built from
# the data that were loaded (NIRCam bands as before, HST bands from hst_io), read through REG everywhere.
NIRCAM_BANDS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
NIRCAM_FWHM_ARCSEC = {"F090W": 0.030, "F150W": 0.049, "F200W": 0.064, "F277W": 0.091, "F356W": 0.115, "F444W": 0.145}
NIRCAM_LAM_UM = {"F090W": 0.902, "F150W": 1.501, "F200W": 1.990, "F277W": 2.786, "F356W": 3.563, "F444W": 4.421}  # pivot
NIRCAM_CHANNEL = {"F090W": "NIRCAM_SW", "F150W": "NIRCAM_SW", "F200W": "NIRCAM_SW",
                  "F277W": "NIRCAM_LW", "F356W": "NIRCAM_LW", "F444W": "NIRCAM_LW"}
CHANNELS = ["NIRCAM_SW", "NIRCAM_LW", "WFC3_IR", "ACS_WFC", "WFC3_UVIS"]  # canonical order = one-hot order (NIRCam first)
CHANNEL_DIAMETER_M = {"NIRCAM_SW": 6.5, "NIRCAM_LW": 6.5, "WFC3_IR": 2.4, "ACS_WFC": 2.4, "WFC3_UVIS": 2.4}
HST_CHANNEL = {"WFC3/IR": "WFC3_IR", "ACS/WFC": "ACS_WFC", "WFC3/UVIS": "WFC3_UVIS"}
LOD_COEF = 0.0317  # lambda/D [arcsec per um] for D = 6.5 m (v28 value); scaled by 6.5 / D for other telescopes
D_REF_M = 6.5


def nominal_fwhm_arcsec(lam_um, diameter_m, pixel_arcsec):
    """Starting FWHM for an HST band (NIRCam keeps its table): Airy FWHM 1.028 lambda/D added in quadrature to half the detector
    pixel. A starting value for the Gaussian PSF core / rigid model only (WFC3/IR F160W: 0.156 arcsec vs 0.15 published)."""
    lod = 0.206265 * lam_um / diameter_m
    return round(math.hypot(1.028 * lod, 0.5 * pixel_arcsec), 4)


class BandRegistry:
    """Per band: pivot lambda [um], telescope diameter [m], nominal FWHM [arcsec], channel (str) and its id in self.channels, SVO
    filter id (for load_bandpasses), target-grid PSF kernel size [px], instrument label (roll-up of the scores) and facility.
    Band order: NIRCam bands (canonical order) first, then HST bands sorted by wavelength. self.channels is the one-hot basis of the
    learned PSF: the channels in use, in canonical order; both NIRCam channels whenever any NIRCam band is present (the v28 shapes)."""

    def __init__(self, entries):
        self.entries = [dict(e) for e in entries]
        self.bands = [e["band"] for e in self.entries]
        assert len(set(self.bands)) == len(self.bands), self.bands
        self.index = {b: k for k, b in enumerate(self.bands)}
        col = lambda key: {e["band"]: e[key] for e in self.entries}
        self.lam_um, self.fwhm_arcsec, self.channel = col("lam_um"), col("fwhm_arcsec"), col("channel")
        self.diameter_m, self.svo_id, self.kernel_px = col("diameter_m"), col("svo_id"), col("kernel_px")
        self.instrument, self.facility = col("instrument"), col("facility")
        used = set(self.channel.values())
        if any(c.startswith("NIRCAM") for c in used):
            used |= {"NIRCAM_SW", "NIRCAM_LW"}
        self.channels = [c for c in CHANNELS if c in used]
        self.chan_id = {b: self.channels.index(c) for b, c in self.channel.items()}
        self.is_nircam = {b: c.startswith("NIRCAM") for b, c in self.channel.items()}
        self.is_hst = {b: not v for b, v in self.is_nircam.items()}
        self.instruments = list(dict.fromkeys(self.instrument[b] for b in self.bands))
        self.multi = len(self.instruments) > 1
        self.psf_n = max(self.kernel_px.values()) if self.kernel_px else 31

    def summary(self):
        return [{"band": b, "instrument": self.instrument[b], "channel": self.channel[b], "chan_id": self.chan_id[b],
                 "lam_um": self.lam_um[b], "diameter_m": self.diameter_m[b], "fwhm_arcsec": self.fwhm_arcsec[b],
                 "svo_id": self.svo_id[b], "kernel_px": self.kernel_px[b]} for b in self.bands]


def _nircam_entry(b, psf_n):
    ch = NIRCAM_CHANNEL[b]
    return {"band": b, "lam_um": NIRCAM_LAM_UM[b], "diameter_m": CHANNEL_DIAMETER_M[ch], "fwhm_arcsec": NIRCAM_FWHM_ARCSEC[b],
            "channel": ch, "svo_id": f"JWST/NIRCam.{b}", "kernel_px": psf_n, "instrument": "NIRCam", "facility": "JWST"}


def _hst_entry(info, psf_n):
    """info: band, instrument ("WFC3/IR", "ACS/WFC", "WFC3/UVIS"), filter, photplam [Angstrom], svo_id (None for a clear filter)."""
    import hst_io  # the same directory as this script
    bi = hst_io.hst_band_info(info["instrument"], info["filter"])
    if info.get("photplam") in (None, 0):
        raise ValueError(f"HST band {info['band']}: no PHOTPLAM in the headers, cannot place it in wavelength")
    lam = float(info["photplam"]) / 1e4
    return {"band": info["band"], "lam_um": lam, "diameter_m": bi["diameter_m"],
            "fwhm_arcsec": nominal_fwhm_arcsec(lam, bi["diameter_m"], bi["pixel_scale_arcsec"]),
            "channel": HST_CHANNEL[bi["instrument"]], "svo_id": info.get("svo_id") or bi["svo_id"], "kernel_px": psf_n,
            "instrument": bi["instrument"], "facility": "HST"}


def build_registry(nircam_bands, hst_infos=(), psf_n=31):
    """nircam_bands: names (any subset of NIRCAM_BANDS, put in canonical order); hst_infos: list of dicts for _hst_entry."""
    nir = [b for b in NIRCAM_BANDS if b in set(nircam_bands)]
    hst = sorted((_hst_entry(i, psf_n) for i in hst_infos), key=lambda e: (e["lam_um"], e["band"]))
    return BandRegistry([_nircam_entry(b, psf_n) for b in nir] + hst)


def registry_from_exposures(exps, psf_n=31):
    """Registry of the bands present in a loaded exposure list (HST exposures carry instrument/filter/photplam/svo_id)."""
    nir = {e["band"] for e in exps if not e.get("instrument") and e["band"] in NIRCAM_BANDS}
    by = {}
    for e in exps:
        if e.get("instrument"):
            by.setdefault(e["band"], []).append(e)
    infos = []
    for b, es in by.items():
        pl = [float(e["photplam"]) for e in es if e.get("photplam")]
        infos.append({"band": b, "instrument": es[0]["instrument"], "filter": es[0]["filter"],
                      "photplam": float(np.median(pl)) if pl else None, "svo_id": es[0].get("svo_id")})
    return build_registry(sorted(nir), infos, psf_n)


REG = build_registry(NIRCAM_BANDS)  # default = the v28 NIRCam registry; run() replaces it by the registry of the loaded data


def set_registry(reg):
    global REG
    REG = reg
    return reg


def uses_stripes(band):
    """1/f row/column stripes are a NIRCam readout effect: NIRCam exposures only."""
    return REG.is_nircam.get(band, True)


def detection_bands(reg=None):
    """Indices of the bands of the detection image (aperture catalogue, splat seeds): the NIRCam SW bands, by name; if there are
    none (HST-only), the three shortest-wavelength bands."""
    reg = reg or REG
    sw = [k for k, b in enumerate(reg.bands) if reg.channel[b] == "NIRCAM_SW"]
    if sw:
        return sw
    return sorted(sorted(range(len(reg.bands)), key=lambda k: reg.lam_um[reg.bands[k]])[:3])


def require_nircam_only(what):
    hst = [b for b in REG.bands if REG.is_hst[b]]
    if hst:
        raise SystemExit(f"{what} is NIRCam-only (6.5 m pupil model) and cannot be used with HST bands present: {hst}")
LOG: dict = {"timings": {}, "metrics": {}}
ARCSEC2_TO_SR = (math.pi / 648000.0) ** 2
T0 = time.perf_counter()
ZIP_PSF_CONVENTION = "explicit"  # v24; "v20" reproduces the old (180-degree rotated) zip PSF
PSF_CONVENTION = "correlation"  # v25: correlation (v19/v24 explicit) | convolution (pupil orientation fit, v20 zip)


CALIBRATE_NOISE = False
LAM_NEG = 0.0
LAM_TNV = 0.0
APERTURES = None  # (label image, centres, radius px) of catalogue sources for aperture photometry
LAM_BAND = 0.0
HUBER_DELTA = None  # set from args: None = Gaussian chi2, else robust scale in sigma
LOSS_KIND = "gauss"  # gauss | huber (torch.nn.functional.huber_loss) | pseudo_huber


def rho(z2):
    """Per-pixel loss on squared normalised residuals z^2 = ivar (d - m)^2, on the chi2 scale (= z^2 in the core).
    huber: 2 * F.huber_loss(z, 0, delta) (quadratic core, linear tails); pseudo_huber: smooth version."""
    if HUBER_DELTA is None or LOSS_KIND == "gauss":
        return z2
    d = HUBER_DELTA
    if LOSS_KIND == "huber":
        z = torch.sqrt(z2.clamp(min=d * d))  # only used where |z| >= delta: no sqrt gradient at zero
        return torch.where(z2 < d * d, z2, 2.0 * d * z - d * d)
    d2 = d * d
    return 2.0 * d2 * (torch.sqrt(1.0 + z2 / d2) - 1.0)


def noise_scale(d, ivar, row, col):
    """Robust ratio of the empirical to the ERR-predicted noise, from first differences along detector rows
    (smooth sky, ICL and 1/f row offsets cancel; sources are rejected by the MAD)."""
    order = torch.argsort(row * 100000 + col)
    d, iv, r, c = d[order], ivar[order], row[order], col[order]
    nb = (r[1:] == r[:-1]) & (c[1:] == c[:-1] + 1)
    z = (d[1:] - d[:-1])[nb] / torch.sqrt(1.0 / iv[1:][nb] + 1.0 / iv[:-1][nb])
    if z.numel() < 1000:
        return 1.0
    z = z[:: max(1, z.numel() // 2_000_000)]
    return float(1.4826 * (z - z.median()).abs().median())


def tick(msg):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- data
def target_header(ra, dec, size, scale_arcsec):
    s = scale_arcsec / 3600.0
    return {"NAXIS": 2, "NAXIS1": size, "NAXIS2": size, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
            "CUNIT1": "deg", "CUNIT2": "deg", "CRVAL1": ra, "CRVAL2": dec,
            "CRPIX1": size / 2 + 0.5, "CRPIX2": size / 2 + 0.5,
            "CD1_1": -s, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": s}


def exposure_band(path):
    h0 = fits.getheader(path, 0)
    f = str(h0.get("FILTER", "")).strip().upper()
    p = str(h0.get("PUPIL", "")).strip().upper()
    if f in NIRCAM_BANDS:
        return f
    if p in NIRCAM_BANDS:
        return p
    return None


def load_exposures(data_dir, thead, device, pad=24):
    from torchsky.wcs import WCS

    tw = WCS(header=thead)
    size = thead["NAXIS1"]
    files = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
    exps = []
    for f in files:
        b = exposure_band(f)
        if b is None:
            continue
        with fits.open(f, memmap=False) as h:
            sh = h["SCI"].header
            sci = h["SCI"].data.astype(np.float32)
            err = h["ERR"].data.astype(np.float32)
            dq = h["DQ"].data
        hd = {k: sh[k] for k in sh.keys() if k and k not in ("COMMENT", "HISTORY", "")}
        try:
            w = WCS(header=hd)
        except Exception as exc:  # noqa: BLE001
            print("skip (wcs)", os.path.basename(f), exc)
            continue
        ny, nx = sci.shape
        # coarse footprint test on a 32-px lattice first
        yy, xx = np.mgrid[0:ny:32, 0:nx:32]
        pc = torch.tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float64)
        tp = tw.world_to_pixel_differentiable(w.pixel_to_world_differentiable(pc, origin=0), origin=0)
        inside = ((tp[:, 0] > -pad - 64) & (tp[:, 0] < size + pad + 64) & (tp[:, 1] > -pad - 64) & (tp[:, 1] < size + pad + 64)).numpy()
        if not inside.any():
            continue
        x0, x1 = max(int(xx.ravel()[inside].min()) - 32, 0), min(int(xx.ravel()[inside].max()) + 33, nx)
        y0, y1 = max(int(yy.ravel()[inside].min()) - 32, 0), min(int(yy.ravel()[inside].max()) + 33, ny)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        pc = torch.tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float64)
        tp = tw.world_to_pixel_differentiable(w.pixel_to_world_differentiable(pc, origin=0), origin=0).float()
        d = torch.from_numpy(sci[y0:y1, x0:x1].ravel())
        e = torch.from_numpy(err[y0:y1, x0:x1].ravel())
        q = torch.from_numpy(dq[y0:y1, x0:x1].ravel().astype(np.int64))
        keep = ((tp[:, 0] > -pad) & (tp[:, 0] < size + pad) & (tp[:, 1] > -pad) & (tp[:, 1] < size + pad)
                & torch.isfinite(d) & torch.isfinite(e) & (e > 0) & (q == 0))
        if keep.sum() < 2000:
            continue
        dk, ek = d[keep], e[keep]
        med = float(dk.median())
        nsc = 1.0
        if CALIBRATE_NOISE:
            rk = torch.from_numpy(yy.ravel()[keep.numpy()].astype(np.int64))
            ck = torch.from_numpy(xx.ravel()[keep.numpy()].astype(np.int64))
            nsc = noise_scale(dk, 1.0 / ek ** 2, rk, ck)
            ek = ek * nsc
        exps.append({
            "name": os.path.basename(f), "band": b, "tx": tp[keep, 0].to(device), "ty": tp[keep, 1].to(device),
            "d": dk.to(device), "ivar": (1.0 / ek ** 2).to(device),
            "row": torch.from_numpy(yy.ravel()[keep.numpy()].astype(np.int64)).to(device),
            "col": torch.from_numpy(xx.ravel()[keep.numpy()].astype(np.int64)).to(device),
            "dx": torch.from_numpy(((xx.ravel()[keep.numpy()] / nx) * 2 - 1).astype(np.float32)).to(device),
            "dy": torch.from_numpy(((yy.ravel()[keep.numpy()] / ny) * 2 - 1).astype(np.float32)).to(device),
            "bkg0": med, "path": f, "header": hd, "cut": (y0, y1, x0, x1), "noise_scale": nsc,
            "detname": str(fits.getheader(f, 0).get("DETECTOR", "")).strip().upper() or os.path.basename(f).split("_")[4],
        })
    return exps


# ----------------------------------------------------------------------------- v29: HST selection / loading
def _csv_list(v):
    return [x.strip() for x in str(v).split(",") if x.strip()] if v else []


def _norm_inst(x):
    return x.strip().upper().replace("_", "/").replace(" ", "")


def parse_dq_policy(v):
    v = str(v).strip().lower()
    return int(v) if v.lstrip("-").isdigit() else v


def select_hst_files(hst_dir, manifest=None, programs=None, instruments=None, filters=None, mjd_min=None, mjd_max=None):
    """HST flt/flc files to load. With a manifest (CSV like hst_imaging_unique.csv: proposal_id, instrument, detector, filter,
    obs_id, product_filename, product_type, t_start_mjd) the selection is made on the table; without, on the primary headers of
    *_flt.fits / *_flc.fits in hst_dir. programs/instruments/filters: comma lists ("WFC3/IR,ACS/WFC"); mjd_min/max: EXPSTART bounds.
    Only WFC3/IR, ACS/WFC, WFC3/UVIS are kept (the loader supports no others); when both flt and flc exist for one exposure the flc
    (CTE-corrected) is used. Returns (files, info)."""
    import csv

    import hst_io
    progs = {int(x) for x in _csv_list(programs)} or None
    insts = {_norm_inst(x) for x in _csv_list(instruments)} or None
    filts = {x.upper() for x in _csv_list(filters)} or None
    rows = []
    if manifest:
        with open(manifest, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({"path": os.path.join(hst_dir, r["product_filename"]), "inst": f"{r['instrument']}/{r['detector']}".upper(),
                             "filt": r["filter"].strip().upper(), "prog": int(r["proposal_id"]), "mjd": float(r["t_start_mjd"])})
    else:
        for f in sorted(glob.glob(os.path.join(hst_dir, "*_fl[tc].fits"))):
            h0 = fits.getheader(f, 0)
            rows.append({"path": f, "inst": hst_io._identify(h0), "filt": hst_io._filter_name(h0),
                         "prog": int(h0.get("PROPOSID", -1) or -1), "mjd": float(h0.get("EXPSTART", float("nan")))})
    n0 = len(rows)
    ok = [r for r in rows if r["inst"] in hst_io.DQ_GOOD_BITS and r["path"].endswith(("_flt.fits", "_flc.fits"))]
    n_unsupported = len(rows) - len(ok)
    ok = [r for r in ok if (progs is None or r["prog"] in progs) and (insts is None or r["inst"] in insts)
          and (filts is None or r["filt"] in filts) and (mjd_min is None or r["mjd"] >= mjd_min) and (mjd_max is None or r["mjd"] <= mjd_max)]
    best = {}
    for r in ok:  # one file per exposure (root), flc preferred over flt
        root = os.path.basename(r["path"]).split("_")[0]
        if root not in best or r["path"].endswith("_flc.fits"):
            best[root] = r
    sel = sorted(best.values(), key=lambda r: (r["filt"], os.path.basename(r["path"])))
    missing = [r["path"] for r in sel if not os.path.exists(r["path"])]
    files = [r["path"] for r in sel if os.path.exists(r["path"])]
    return files, {"n_rows": n0, "n_unsupported_or_other_product": n_unsupported, "n_selected": len(sel), "n_missing_files": len(missing),
                   "missing_files": [os.path.basename(p) for p in missing][:50], "n_files": len(files)}


def load_all_exposures(a, thead, dev, pad):
    """JWST exactly as v28, plus HST through hst_io when --hst-dir is given; one merged list (JWST first, HST by wavelength, name)."""
    exps = load_exposures(a.data_dir, thead, dev, pad=pad)
    if not getattr(a, "hst_dir", None):
        if any(getattr(a, k, None) for k in ("hst_manifest", "hst_programs", "hst_instruments", "hst_filters", "hst_mjd_min", "hst_mjd_max")):
            raise SystemExit("--hst-manifest/--hst-programs/--hst-instruments/--hst-filters/--hst-mjd-* need --hst-dir")
        return exps
    import hst_io
    files, info = select_hst_files(a.hst_dir, a.hst_manifest, a.hst_programs, a.hst_instruments, a.hst_filters, a.hst_mjd_min, a.hst_mjd_max)
    LOG["hst_selection"] = info
    if info["n_missing_files"]:
        tick(f"HST: {info['n_missing_files']} selected files not found in {a.hst_dir} (skipped): {info['missing_files'][:5]}...")
    hexps = hst_io.load_hst_exposures(files, thead, dev, pad=pad, dq_policy=parse_dq_policy(a.hst_dq_policy), calibrate_noise=CALIBRATE_NOISE)
    hexps.sort(key=lambda e: (e["photplam"] or 0.0, e["band"], e["name"]))
    perb = {}
    for e in hexps:
        perb[e["band"]] = perb.get(e["band"], 0) + 1
    LOG["hst_selection"]["loaded_chips_per_band"] = perb
    tick(f"HST: {len(files)} files selected, {len(hexps)} chips overlap the target grid; per band {perb}")
    return exps + hexps


def splat_coadd(exps, thead, device, radius=None):
    """APPROXIMATE classical coadd for HST bands (torchsky resampling from file headers does not know the HST lookup-table
    distortions): inverse-variance weighted splat of the loaded pixels (tx, ty, d - bkg0, ivar), which already carry the full header
    WCS, onto the target grid. Each pixel is spread with a tent (bilinear) kernel of radius h = max(1, input pixel / target pixel)
    target pixels: h = 1 is plain bilinear splatting (2 x 2 target pixels); for an undersampled detector (WFC3/IR 0.128 arcsec on a
    0.04 arcsec grid, h = 3.2) a plain bilinear splat leaves most target pixels unfilled, so the tent follows the input footprint.
    No flux/area conservation (surface brightness in, surface brightness out), no PSF matching, no outlier rejection.
    Returns (image with NaN where nothing landed, summed weight)."""
    H, W = int(thead["NAXIS2"]), int(thead["NAXIS1"])
    s_out = abs(float(thead["CD2_2"])) * 3600.0
    num = torch.zeros(H * W, dtype=torch.float64, device=device)
    den = torch.zeros_like(num)
    for e in exps:
        s_in = math.sqrt(float(e["pixar_sr_nominal"]) / ARCSEC2_TO_SR) if e.get("pixar_sr_nominal") else s_out
        h = max(1.0, s_in / s_out) if radius is None else float(radius)  # radius: test override (1.0 = plain bilinear)
        tx, ty = e["tx"].double().to(device), e["ty"].double().to(device)
        iv, val = e["ivar"].double().to(device), (e["d"] - e["bkg0"]).double().to(device)
        ok = torch.isfinite(val) & torch.isfinite(iv) & (iv > 0) & torch.isfinite(tx) & torch.isfinite(ty)
        tx, ty, iv, val = tx[ok], ty[ok], iv[ok], val[ok]
        x0, y0 = torch.floor(tx), torch.floor(ty)
        m = int(math.ceil(h)) - 1
        for oy in range(-m, m + 2):
            iy = y0 + oy
            wy = (1.0 - (iy - ty).abs() / h).clamp(min=0.0)
            for ox in range(-m, m + 2):
                ix = x0 + ox
                w = wy * (1.0 - (ix - tx).abs() / h).clamp(min=0.0)
                inside = (w > 0) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
                if not bool(inside.any()):
                    continue
                lin = (iy[inside] * W + ix[inside]).long()
                wi = w[inside] * iv[inside]
                num.index_add_(0, lin, wi * val[inside])
                den.index_add_(0, lin, wi)
    return (num / den).reshape(H, W).float(), den.reshape(H, W).float()


def band_median_sigma(c, cover=None):
    """(median, 1.4826 MAD) of a band image: all pixels (NIRCam, as v28) or the covered pixels only (HST splat coadds)."""
    v = c if cover is None or not bool(cover.any()) else c[cover]
    m = v.median()
    return m, 1.4826 * (v - m).abs().median()


# ----------------------------------------------------------------------------- classical
def classical(exps, thead, kernel, pixfrac, device):
    from torchsky.map.image_plane_coadd import coadd_image_plane
    from torchsky.map.observation import ImageObservation

    out, cover = {}, {}
    for b in REG.bands:
        if REG.is_hst[b]:  # v29: approximate splat coadd from the loaded pixels (SWarp-like and Drizzle-like are the same image)
            img, wsum = splat_coadd([e for e in exps if e["band"] == b and not e["holdout"] and not e.get("val")], thead, device)
            cover[b] = (wsum > 0).to(device)
            out[b] = torch.nan_to_num(img).to(device)
            continue
        cover[b] = None
        obs = []
        for e in exps:
            if e["band"] != b or e["holdout"] or e.get("val"):
                continue
            with fits.open(e["path"], memmap=False) as h:
                sci = h["SCI"].data.astype(np.float32) - e["bkg0"]
                err = h["ERR"].data.astype(np.float32)
                dq = h["DQ"].data
            y0, y1, x0, x1 = e["cut"]
            hd = dict(e["header"])
            hd["CRPIX1"] = hd["CRPIX1"] - x0
            hd["CRPIX2"] = hd["CRPIX2"] - y0
            hd["NAXIS1"], hd["NAXIS2"] = x1 - x0, y1 - y0
            s = sci[y0:y1, x0:x1]
            ee = err[y0:y1, x0:x1]
            ok = np.isfinite(s) & np.isfinite(ee) & (ee > 0) & (dq[y0:y1, x0:x1] == 0)
            wmap = np.where(ok, 1.0 / np.where(ok, ee, 1) ** 2, 0).astype(np.float32)
            obs.append(ImageObservation(data=torch.from_numpy(np.where(ok, s, 0).astype(np.float32)), header=hd,
                                        weight_map=torch.from_numpy(wmap)))
        res = coadd_image_plane(obs, target_header=thead, resampling_kernel=kernel, drizzle_pixfrac=pixfrac,
                                stacking_method="weighted_mean", device=device)
        out[b] = torch.nan_to_num(res.data.float()).to(device)
    classical.last_cover = cover
    return torch.stack([out[b] for b in REG.bands])


# ----------------------------------------------------------------------------- forward model pieces
# Spatially varying PSF: K(p) = sum_k w_k(detector position, band, detector) B_k^b ;  B_0 = normalized core,
# B_{k>0} zero-sum residual eigen-kernels.  m = sum_k w_k(p) [B_k^b * s_b](x(p)) -> n_k convolutions + one
# multi-channel bilinear sample per pixel (exact for PSFs that vary smoothly over the detector).
# Geometry: torchsky WCS (fixed) o per-detector static distortion field o per-exposure affine.
# Background: per-exposure smooth field + per-detector pattern.  Photometry: per-exposure gain.

_LCACHE: dict = {}


def legendre2d(x, y, order):
    """Cached for large persistent tensors (training chunks); see _legendre2d."""
    if x.numel() >= 100_000:
        key = (x.data_ptr(), x.numel(), order)
        v = _LCACHE.get(key)
        if v is None:
            with torch.no_grad():
                v = _legendre2d(x, y, order).contiguous()
            _LCACHE[key] = v
        return v
    return _legendre2d(x, y, order)


def _legendre2d(x, y, order):
    """Legendre P_i(x)P_j(y), i+j<=order, x,y in [-1,1] -> (N, nterms)."""
    def leg(t, n):
        P = [torch.ones_like(t), t]
        for k in range(2, n + 1):
            P.append(((2 * k - 1) * t * P[k - 1] - (k - 1) * P[k - 2]) / k)
        return P[: n + 1]
    Px, Py = leg(x, order), leg(y, order)
    return torch.stack([Px[i] * Py[j] for i in range(order + 1) for j in range(order + 1 - i)], -1)


def vec(idx, like):
    """int or 0-d index -> per-sample LongTensor matching like's length."""
    if isinstance(idx, torch.Tensor) and idx.dim() == 1:
        return idx
    return torch.full((like.shape[0],), int(idx), dtype=torch.long, device=like.device)


def nterms(order):
    return (order + 1) * (order + 2) // 2


def gauss_kernel(sigma_px, n=31, device="cpu"):
    r = torch.arange(n, device=device, dtype=torch.float32) - n // 2
    g = torch.exp(-0.5 * (r[None, :] ** 2 + r[:, None] ** 2) / sigma_px[:, None, None] ** 2)
    return g / g.sum((-2, -1), keepdim=True)


def fourier(x, nf=6):
    f = (2.0 ** torch.arange(nf, device=x.device, dtype=x.dtype)) * math.pi
    xf = x[..., None] * f
    return torch.cat([torch.sin(xf), torch.cos(xf)], -1).flatten(-2)


class ExplicitVarPSF(nn.Module):
    """Gaussian core (learned width) + nk-1 learned zero-sum residual kernels per band;
    per-band, per-detector quadratic Legendre maps of the mixing weights over the detector."""

    def __init__(self, sig0, ndet, nk=4, n=31, order=2):
        super().__init__()
        nb = len(sig0)
        self.n, self.nk, self.order = n, nk, order
        self.log_sig = nn.Parameter(torch.log(sig0.clone()))
        self.res = nn.Parameter(torch.zeros(nb, nk, n, n, device=sig0.device))  # res[:,0] refines the core
        self.wc = nn.Parameter(torch.zeros(nb, ndet, nk, nterms(order), device=sig0.device))

    def basis(self):
        core = gauss_kernel(self.log_sig.exp(), self.n, self.res.device) + self.res[:, 0]
        core = core.clamp(min=0)
        core = core / core.sum((-2, -1), keepdim=True)
        rest = self.res[:, 1:] - self.res[:, 1:].mean((-2, -1), keepdim=True)
        return torch.cat([core[:, None], rest], 1)  # nb nk n n

    def weights(self, b, det, dx, dy):
        L = legendre2d(dx, dy, self.order)  # N T
        det = vec(det, dx)
        w = torch.einsum("nt,nkt->nk", L, self.wc[b][det])
        w = w + torch.nn.functional.one_hot(torch.zeros(1, dtype=torch.long, device=dx.device), self.nk).float()
        return w

    def penalty(self):
        return 10.0 * self.res.pow(2).sum() + 1e-2 * self.wc.pow(2).sum()


class ImplicitVarPSF(nn.Module):
    """Basis kernels from an MLP field over (dx_kernel, dy_kernel, band); mixing weights from an MLP over
    (detector position Fourier features, band, detector embedding)."""

    def __init__(self, nb, ndet, nk=4, n=31, hidden=64):
        super().__init__()
        self.n, self.nk, self.nb = n, nk, nb
        self.kern = nn.Sequential(nn.Linear(2 * 2 * 6 + nb, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, nk))
        self.det_emb = nn.Embedding(ndet, 4)
        self.wnet = nn.Sequential(nn.Linear(2 * 2 * 3 + nb + 4, 32), nn.SiLU(), nn.Linear(32, nk - 1))
        nn.init.zeros_(self.wnet[-1].weight)
        nn.init.zeros_(self.wnet[-1].bias)

    def basis(self):
        dev = self.det_emb.weight.device
        r = (torch.arange(self.n, device=dev, dtype=torch.float32) - self.n // 2) / (self.n // 2)
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        xy = torch.stack([xx, yy], -1).reshape(-1, 2)
        enc = fourier(xy * 0.5, 6)
        out = []
        for b in range(self.nb):
            oh = torch.zeros(xy.shape[0], self.nb, device=dev)
            oh[:, b] = 1
            k = self.kern(torch.cat([enc, oh], 1)).T.reshape(self.nk, self.n, self.n)
            core = torch.nn.functional.softplus(k[0])
            core = core / core.sum()
            rest = 0.01 * (k[1:] - k[1:].mean((-2, -1), keepdim=True))
            out.append(torch.cat([core[None], rest], 0))
        return torch.stack(out)

    def weights(self, b, det, dx, dy):
        N = dx.shape[0]
        oh = torch.zeros(N, self.nb, device=dx.device)
        oh[:, b] = 1
        inp = torch.cat([fourier(torch.stack([dx, dy], -1) * 0.5, 3), oh, self.det_emb.weight[vec(det, dx)]], 1)
        return torch.cat([torch.ones(N, 1, device=dx.device), self.wnet(inp)], 1)

    def penalty(self):
        return torch.zeros((), device=self.det_emb.weight.device)


class ChromaticPSF(nn.Module):
    """One PSF field for all bands, smooth in wavelength and detector position.
    Basis kernels K_k(dx; lambda) from an MLP over (i) the kernel plane in diffraction units rho = dx * pix / (lambda/D),
    so the Airy-like structure scales with lambda by construction, (ii) the kernel plane in pixels (pixel response,
    jitter), (iii) log lambda and the SW/LW channel. Mixing weights w_k(u; lambda, detector) vary smoothly over the
    detector. No per-band free kernels: bands constrain each other through lambda.
    v29: one network for all instruments. lambda/D uses the band's telescope diameter (6.5 m NIRCam, 2.4 m HST; the per-channel diameter
    is a non-persistent buffer so the state dict keeps the v28 layout), the channel one-hot runs over the registry's channels
    (NIRCam SW/LW, WFC3/IR, ACS/WFC, WFC3/UVIS), and the conditioning is (log lambda, channel one-hot)."""

    def __init__(self, bands, ndet, scale_arcsec, nk=4, n=31, hidden=96, reg=None):
        super().__init__()
        reg = reg or REG
        self.n, self.nk, self.nb = n, nk, len(bands)
        self.nchan = len(reg.channels)
        lam = torch.tensor([reg.lam_um[b] for b in bands])
        cid = torch.tensor([reg.chan_id[b] for b in bands])
        self.register_buffer("loglam", (torch.log(lam) - math.log(2.0)) / 0.8)
        self.register_buffer("chan", torch.nn.functional.one_hot(cid, self.nchan).float())
        self.register_buffer("chan_D", torch.tensor([CHANNEL_DIAMETER_M[c] for c in reg.channels]), persistent=False)
        lod = LOD_COEF * lam * (D_REF_M / self.chan_D[cid])  # lambda/D in arcsec (exactly 0.0317 lambda for D = 6.5 m)
        self.scale_arcsec = scale_arcsec
        self.register_buffer("chan_idx", cid.clone())
        self.register_buffer("pix_per_lod", scale_arcsec / lod)
        self.kern = nn.Sequential(nn.Linear(2 * 2 * 6 + 2 * 2 * 4 + 1 + self.nchan, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, nk))
        self.det_emb = nn.Embedding(ndet, 4)
        self.wnet = nn.Sequential(nn.Linear(2 * 2 * 3 + 1 + self.nchan + 4, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, nk - 1))
        nn.init.zeros_(self.wnet[-1].weight)
        nn.init.zeros_(self.wnet[-1].bias)

    def basis(self):
        return self._basis(self.pix_per_lod, self.loglam, self.chan)

    def basis_lam(self, lam_um, chan_idx):
        """Basis kernels at arbitrary wavelengths (um) and registry channel ids: (M, nk, n, n)."""
        lod = LOD_COEF * lam_um * (D_REF_M / self.chan_D[chan_idx.long()])
        loglam = (torch.log(lam_um) - math.log(2.0)) / 0.8
        chan = torch.nn.functional.one_hot(chan_idx.long(), self.nchan).float()
        return self._basis(self.scale_arcsec / lod, loglam, chan)

    def _basis(self, pix_per_lod, loglam, chan):
        dev = self.loglam.device
        r = torch.arange(self.n, device=dev, dtype=torch.float32) - self.n // 2
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        xy = torch.stack([xx, yy], -1).reshape(-1, 2)  # pixels
        P = xy.shape[0]
        M = loglam.shape[0]
        rows = []
        for b in range(M):
            rho = xy * pix_per_lod[b] / 24.0  # diffraction units, |rho| <~ 1 over the kernel
            cond = torch.cat([loglam[b].view(1, 1), chan[b].view(1, -1)], 1).expand(P, 1 + self.nchan)
            rows.append(torch.cat([fourier(rho, 6), fourier(xy / (self.n), 4), cond], 1))
        k = self.kern(torch.cat(rows, 0)).reshape(M, P, self.nk).permute(0, 2, 1).reshape(M, self.nk, self.n, self.n)
        core = torch.nn.functional.softplus(k[:, 0])
        core = core / core.sum((-2, -1), keepdim=True)
        rest = 0.01 * (k[:, 1:] - k[:, 1:].mean((-2, -1), keepdim=True))
        return torch.cat([core[:, None], rest], 1)

    def weights(self, b, det, dx, dy):
        N = dx.shape[0]
        cond = torch.cat([self.loglam[b].view(1, 1), self.chan[b].view(1, -1)], 1).expand(N, 1 + self.nchan)
        inp = torch.cat([fourier(torch.stack([dx, dy], -1) * 0.5, 3), cond, self.det_emb.weight[vec(det, dx)]], 1)
        return torch.cat([torch.ones(N, 1, device=dx.device), self.wnet(inp)], 1)

    def penalty(self):
        if not getattr(self, "centre_weight", 0.0):
            return torch.zeros((), device=self.loglam.device)
        B = self.basis()  # nb nk n n
        r = torch.arange(self.n, device=B.device, dtype=B.dtype) - self.n // 2
        mx = (B.sum(-2) * r).sum(-1)  # first moments, nb nk
        my = (B.sum(-1) * r).sum(-1)
        return self.centre_weight * (mx.pow(2) + my.pow(2)).sum()


def warm_psf(psf, target_k, steps):
    opt = torch.optim.Adam(psf.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        l = (psf.basis()[:, 0] - target_k).pow(2).sum()
        l.backward()
        opt.step()
    return float(l.detach()) if steps else float("nan")


class Geometry(nn.Module):
    """x' = c_i + (I+M_i)(T_i(p) + D_det(p) - c_i) + t_i.  D_det: static per-detector distortion residual,
    cubic Legendre (explicit) or MLP field (implicit); M_i, t_i per-exposure affine."""

    def __init__(self, exps, ndet, mode="fixed"):
        super().__init__()
        n = len(exps)
        dev = exps[0]["tx"].device
        self.mode = mode
        learn = mode != "fixed"
        self.M = nn.Parameter(torch.zeros(n, 2, 2, device=dev), requires_grad=learn)
        self.t = nn.Parameter(torch.zeros(n, 2, device=dev), requires_grad=learn)
        self.c = torch.stack([torch.stack([e["tx"].mean(), e["ty"].mean()]) for e in exps])
        self.order = 3
        self.D = nn.Parameter(torch.zeros(ndet, 2, nterms(3), device=dev), requires_grad=(mode == "explicit"))
        self.net = None
        if mode == "implicit":
            self.det_emb = nn.Embedding(ndet, 4).to(dev)
            self.net = nn.Sequential(nn.Linear(2 * 2 * 4 + 4, 32), nn.SiLU(), nn.Linear(32, 2)).to(dev)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, i, det, tx, ty, dx, dy):
        if self.mode != "fixed":
            d = self.distortion(det, dx, dy)
            tx, ty = tx + d[:, 0], ty + d[:, 1]
        i = vec(i, dx)
        c = self.c[i]
        p = torch.stack([tx, ty], -1) - c
        q = p + torch.einsum("nij,nj->ni", self.M[i], p) + self.t[i]
        return q[:, 0] + c[:, 0], q[:, 1] + c[:, 1]

    def distortion(self, det, dx, dy):
        det = vec(det, dx)
        if self.mode == "explicit":  # no constant term: a global offset is the per-exposure affine's job (gauge)
            return torch.einsum("nt,ndt->nd", legendre2d(dx, dy, self.order)[:, 1:], self.D[det][:, :, 1:])
        if self.mode == "implicit":
            return self.net(torch.cat([fourier(torch.stack([dx, dy], -1) * 0.5, 4), self.det_emb.weight[det]], 1))
        return torch.zeros(dx.shape[0], 2, device=dx.device)

    def penalty(self):
        p = self.M.pow(2).sum() * 1e4 + self.t.pow(2).sum() * 1e-2 + self.D.pow(2).sum() * 1e-1
        if self.t.requires_grad:  # gauge: the sky frame must not drift with the mean exposure transform
            p = p + 1e4 * self.t.mean(0).pow(2).sum() + 1e6 * self.M.mean(0).pow(2).sum()
        if self.mode == "implicit":  # zero-mean distortion per detector over a fixed lattice
            dev = self.t.device
            lat = torch.linspace(-0.9, 0.9, 7, device=dev)
            yy, xx = torch.meshgrid(lat, lat, indexing="ij")
            dx, dy = xx.flatten(), yy.flatten()
            for d in range(self.det_emb.num_embeddings):
                p = p + 1e3 * self.distortion(d, dx, dy).mean(0).pow(2).sum()
        return p


class Stripes(nn.Module):
    """NIRCam 1/f noise: per-exposure row offsets (fast-read direction) and weaker column offsets."""

    def __init__(self, nexp, dev, n=2048):
        super().__init__()
        self.r = nn.Parameter(torch.zeros(nexp, n, device=dev))
        self.c = nn.Parameter(torch.zeros(nexp, n, device=dev))

    def forward(self, i, row, col):
        return self.r[i, row] + self.c[i, col]

    def penalty(self, scale):
        return (self.r.pow(2).sum() + 4 * self.c.pow(2).sum()) / scale ** 2 * 1e-3


class ExplicitBackground(nn.Module):
    """Per-exposure quadratic Legendre field + per-detector quartic pattern (shared across exposures)
    + per-exposure gain."""

    def __init__(self, exps, ndet):
        super().__init__()
        dev = exps[0]["tx"].device
        self.b0 = torch.tensor([e["bkg0"] for e in exps], device=dev)
        self.pe = nn.Parameter(torch.zeros(len(exps), nterms(2), device=dev))
        self.pd = nn.Parameter(torch.zeros(ndet, nterms(4), device=dev))
        self.logg = nn.Parameter(torch.zeros(len(exps), device=dev))

    def forward(self, i, det, dx, dy):
        i, det = vec(i, dx), vec(det, dx)
        return (self.b0[i] + (legendre2d(dx, dy, 2) * self.pe[i]).sum(1) + (legendre2d(dx, dy, 4) * self.pd[det]).sum(1))

    def gain(self, i):
        return self.logg[i].exp()

    def penalty(self):
        return 1.0 * self.logg.pow(2).sum()


class ImplicitBackground(nn.Module):
    """MLP field over detector-position Fourier features + exposure and detector embeddings; per-exposure gain."""

    def __init__(self, exps, ndet, emb=8, hidden=48):
        super().__init__()
        dev = exps[0]["tx"].device
        self.b0 = torch.tensor([e["bkg0"] for e in exps], device=dev)
        self.eemb = nn.Embedding(len(exps), emb).to(dev)
        self.demb = nn.Embedding(ndet, 4).to(dev)
        self.net = nn.Sequential(nn.Linear(2 * 2 * 4 + emb + 4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)).to(dev)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.logg = nn.Parameter(torch.zeros(len(exps), device=dev))
        self.scale = nn.Parameter(torch.tensor(1.0, device=dev), requires_grad=False)

    def forward(self, i, det, dx, dy):
        i, det = vec(i, dx), vec(det, dx)
        inp = torch.cat([fourier(torch.stack([dx, dy], -1) * 0.5, 4), self.eemb.weight[i], self.demb.weight[det]], 1)
        return self.b0[i] + self.scale * self.net(inp)[:, 0]

    def gain(self, i):
        return self.logg[i].exp()

    def penalty(self):
        return 1.0 * self.logg.pow(2).sum()


def load_updec(updec_dir, thead, dev):
    """UPdec-Webb module-b i2d products (iter0 = coadd, iter10 = deconvolved) read with torchfits and resampled
    with torchsky (Lanczos-3, surface brightness) onto the target grid: {iteration: (nb, H, W)}."""
    import torchfits
    from torchsky.map.image_plane_coadd import coadd_image_plane
    from torchsky.map.observation import ImageObservation

    out = {}
    for it in (0, 10):
        planes = []
        for b in REG.bands:
            f = Path(updec_dir) / f"{b}b_UPDC-v2_iter{it}_i2d.fits.fz"
            if not f.exists():
                break
            d, hdr = torchfits.read(str(f), hdu="SCI", return_header=True)
            hd = {k: hdr[k] for k in hdr.keys() if k and k not in ("COMMENT", "HISTORY", "")}
            for k in ("ZNAXIS", "ZNAXIS1", "ZNAXIS2", "ZIMAGE", "ZBITPIX", "ZCMPTYPE", "ZTILE1", "ZTILE2", "ZQUANTIZ", "XTENSION", "PCOUNT", "GCOUNT"):
                hd.pop(k, None)
            d = d.float()
            w = torch.isfinite(d).float()
            obs = ImageObservation(data=torch.nan_to_num(d), header=hd, weight_map=w)
            res = coadd_image_plane([obs], target_header=thead, resampling_kernel="lanczos3", value_semantics="surface_brightness",
                                    stacking_method="weighted_mean", device=str(dev))
            planes.append(torch.nan_to_num(res.data.float()).to(dev))
        if len(planes) == len(REG.bands):
            out[it] = torch.stack(planes)
            tick(f"UPdec iter{it} resampled onto the target grid")
    return out


def load_bandpasses(bands, q_nodes, cache_dir=None, reg=None):
    """Per band: Q equal-weight quadrature nodes of the photon-counting f_nu average
    <f_nu>_b = int f_nu T(l) dl/l / int T(l) dl/l, from SVO transmission curves (torchsky.filter_profiles,
    cached on disk). Returns lam_um (nb, Q), weights (nb, Q) summing to 1, and the curves."""
    from torchsky.filter_profiles import bandpass_from_svo

    reg = reg or REG
    lam, wts, curves = [], [], {}
    for b in bands:
        sid = reg.svo_id[b]  # NIRCam: JWST/NIRCam.<filter>; HST: the loader's id (HST/ACS_WFC.F814W, HST/WFC3_IR.F160W, ...)
        if not sid:
            raise ValueError(f"band {b} has no SVO filter id (clear filter?): cannot build its bandpass")
        fac, rest = sid.split("/", 1)
        ins, filt = rest.rsplit(".", 1)
        spec = bandpass_from_svo(band_name=filt, facility=fac, instrument=ins, filter_id=sid, prefer="any", cache_dir=cache_dir)
        wl = torch.as_tensor(spec["transmission_wavelength"], dtype=torch.float64) / 1e4  # um
        tr = torch.as_tensor(spec["transmission_throughput"], dtype=torch.float64).clamp(min=0)
        o = torch.argsort(wl)
        wl, tr = wl[o], tr[o]
        dens = tr / wl
        cdf = torch.cat([torch.zeros(1, dtype=wl.dtype), torch.cumsum(0.5 * (dens[1:] + dens[:-1]) * (wl[1:] - wl[:-1]), 0)])
        cdf = cdf / cdf[-1]
        qs = (torch.arange(q_nodes, dtype=wl.dtype) + 0.5) / q_nodes
        idx = torch.searchsorted(cdf, qs).clamp(1, len(cdf) - 1)
        f = (qs - cdf[idx - 1]) / (cdf[idx] - cdf[idx - 1]).clamp(min=1e-12)
        lam.append(wl[idx - 1] + f * (wl[idx] - wl[idx - 1]))
        wts.append(torch.full((q_nodes,), 1.0 / q_nodes, dtype=wl.dtype))
        curves[b] = {"svo_filter_id": spec.get("svo_filter_id"), "lam_um": wl[::4].tolist(), "T": tr[::4].tolist()}
    return torch.stack(lam).float(), torch.stack(wts).float(), curves


LAM_LO, LAM_HI = 0.70, 5.30  # um, spectral domain of the sky field


def lam_unit(lam_um):
    return (torch.log(lam_um) - math.log(LAM_LO)) / (math.log(LAM_HI) - math.log(LAM_LO))


class RBFSpectralBasis(nn.Module):
    """phi_k(lambda): K normalised Gaussian bumps uniform in log lambda (partition of unity), optionally plus a
    learned smooth correction (zero-initialised MLP on Fourier features of log lambda)."""

    def __init__(self, k=8, learned=False):
        super().__init__()
        self.k = k
        self.register_buffer("c", torch.linspace(0.0, 1.0, k))
        self.width = 1.0 / (k - 1)
        self.net = None
        if learned:
            self.net = nn.Sequential(nn.Linear(2 * 6, 32), nn.SiLU(), nn.Linear(32, k))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, lam_um):  # (M,) -> (M, K)
        u = lam_unit(lam_um)
        g = torch.exp(-0.5 * ((u[:, None] - self.c[None]) / self.width) ** 2)
        phi = g / g.sum(1, keepdim=True)
        if self.net is not None:
            phi = phi + 0.1 * self.net(fourier(u[:, None], 6))
        return phi


class SpectralSky(nn.Module):
    """Continuous sky s(x, y, lambda) = sum_k a_k(x, y) phi_k(lambda): a_k from a pixel grid (explicit) or a neural
    field (implicit). Bands see it only through their transmission curves:
    s_b(x) = sum_q w_bq s(x, lambda_bq)  (photon-weighted quadrature of the SVO curve)."""

    def __init__(self, coeffs, basis, lam_nodes, w_nodes, lam_smooth=0.0, sig_ref=1.0, relative=False, floor_sigma=3.0):
        super().__init__()
        self.coeffs, self.basis = coeffs, basis
        self.register_buffer("lam_nodes", lam_nodes)  # nb Q
        self.register_buffer("w_nodes", w_nodes)
        self.lam_smooth, self.sig_ref = lam_smooth, sig_ref
        self.relative, self.floor_sigma = relative, floor_sigma

    def nodes(self):
        A = self.coeffs()  # K H W
        nb, q = self.lam_nodes.shape
        phi = self.basis(self.lam_nodes.flatten())  # nbQ K
        Sn = torch.einsum("mk,khw->mhw", phi, A).view(nb, q, *A.shape[-2:])
        S = (Sn * self.w_nodes[:, :, None, None]).sum(1)
        return Sn, S

    def forward(self):
        return self.nodes()[1]

    def spectral_cube(self, lam_um):
        return torch.einsum("mk,khw->mhw", self.basis(lam_um), self.coeffs())

    def penalty(self):
        if self.lam_smooth <= 0:
            return torch.zeros((), device=self.lam_nodes.device)
        A = self.coeffs()
        d2 = A[2:] - 2 * A[1:-1] + A[:-2]
        if self.relative:  # brightness-independent: curvature relative to the pixel's own spectral level
            scale = (A.abs().mean(0, keepdim=True) + self.floor_sigma * self.sig_ref).detach()
            return self.lam_smooth * (d2 / scale).pow(2).mean()
        return self.lam_smooth * (d2 / self.sig_ref).pow(2).mean()

    def param_groups(self, base):
        if isinstance(self.coeffs, ConvDecoderCoeffs):
            return [{"params": list(self.coeffs.parameters()), "lr": DIP_LR}]
        if isinstance(self.coeffs, HybridCoeffs):
            sp = self.coeffs.splats
            return [{"params": list(self.coeffs.grid.parameters()) + list(self.basis.parameters()), "lr": base},
                    {"params": [sp.amp], "lr": base * 20},
                    {"params": [sp.mu], "lr": 1e-2},
                    {"params": [sp.log_s, sp.theta], "lr": 5e-3}]
        return [{"params": list(self.coeffs.parameters()) + list(self.basis.parameters()), "lr": base}]


DIP_LR = 1e-3


def spectral_init(S_band, lam_nodes, w_nodes, basis, mu=1e-5):
    """a_k per pixel from band images: ridge least squares with a second-difference prior across k."""
    with torch.no_grad():
        nb, q = lam_nodes.shape
        phi = basis(lam_nodes.flatten()).view(nb, q, -1)
        Phi = (phi * w_nodes[:, :, None]).sum(1)  # nb K
        k = Phi.shape[1]
        D = torch.zeros(k - 2, k, device=Phi.device)
        for i in range(k - 2):
            D[i, i:i + 3] = torch.tensor([1.0, -2.0, 1.0], device=Phi.device)
        M = Phi.T @ Phi + mu * D.T @ D + 1e-6 * torch.eye(k, device=Phi.device)
        P = torch.linalg.solve(M.cpu(), Phi.T.cpu()).to(Phi.device)  # K nb
        return torch.einsum("kb,bhw->khw", P, S_band)


class GridSky(nn.Module):
    """Pixel-grid sky; optionally s = G_sigma * latent (a narrow Gaussian band limit that damps grid-scale
    structure no exposure constrains, for super-resolved grids)."""

    def __init__(self, init, blur_px=0.0):
        super().__init__()
        self.s = nn.Parameter(init.clone())
        self.blur_px = blur_px
        if blur_px > 0:
            k = gauss_kernel(torch.full((init.shape[0],), blur_px, device=init.device), 2 * int(math.ceil(3 * blur_px)) + 1, init.device)
            self.register_buffer("k", k[:, None])

    def forward(self):
        if self.blur_px <= 0:
            return self.s
        p = self.k.shape[-1] // 2
        return F.conv2d(F.pad(self.s[None], (p,) * 4, mode="reflect"), self.k, groups=self.s.shape[0])[0]


class SineLayer(nn.Module):
    def __init__(self, nin, nout, w0, first=False):
        super().__init__()
        self.w0 = w0
        self.lin = nn.Linear(nin, nout)
        with torch.no_grad():
            bound = 1.0 / nin if first else math.sqrt(6.0 / nin) / w0
            self.lin.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.lin(x))


class SirenSky(nn.Module):
    """SIREN (Sitzmann et al. 2020): s_b(x) = W_out sin(w0 ... sin(w0 W_1 x + b_1) ...) with one output per band."""

    def __init__(self, size, nb, hidden=384, layers=5, w0=30.0, scale=None, chunk=65536):
        super().__init__()
        self.size, self.chunk = size, chunk
        net = [SineLayer(2, hidden, w0, first=True)] + [SineLayer(hidden, hidden, w0) for _ in range(layers - 1)]
        self.body = nn.Sequential(*net)
        self.head = nn.Linear(hidden, nb)
        with torch.no_grad():
            bnd = math.sqrt(6.0 / hidden) / w0
            self.head.weight.uniform_(-bnd, bnd)
            self.head.bias.zero_()
        self.register_buffer("scale", scale if scale is not None else torch.ones(nb))
        r = (torch.arange(size, dtype=torch.float32) + 0.5) / size * 2 - 1
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        self.register_buffer("xy", torch.stack([xx, yy], -1).reshape(-1, 2))

    def forward(self):
        out = self.head(self.body(self.xy))  # (H*W, nb)
        # sinh output: linear near the noise level, logarithmic for bright cores (dynamic range ~1e4 sigma)
        return torch.sinh(out.T.reshape(-1, self.size, self.size)) * self.scale[:, None, None]


class FieldSky(nn.Module):
    """Multi-resolution dense feature grids (Instant-NGP without hashing) + MLP with one head per band."""

    def __init__(self, size, nb, levels=(16, 32, 64, 128, 256, 512, 1024), feat=4, hidden=96, scale=None):
        super().__init__()
        self.size = size
        self.grids = nn.ParameterList([nn.Parameter(1e-2 * torch.randn(1, feat, r, r)) for r in levels if r <= size])
        nin = feat * len(self.grids)
        self.mlp = nn.Sequential(nn.Linear(nin, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, nb))
        self.register_buffer("scale", scale if scale is not None else torch.ones(nb))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy], -1)[None])

    def forward(self):
        fs = [F.grid_sample(g, self.grid, mode="bilinear", align_corners=True) for g in self.grids]
        f = torch.cat(fs, 1)[0].permute(1, 2, 0)  # H W C
        out = self.mlp(f).permute(2, 0, 1)  # nb H W
        return out * self.scale[:, None, None]


class GaussSky(nn.Module):
    """2D Gaussian splatting sky: N anisotropic Gaussians (position, two scales, angle, one flux per band),
    splatted onto the grid inside a small window, + a low-resolution diffuse component (ICL, faint wings).
    Fluxes are integrated fluxes in units of the per-band sky noise."""

    def __init__(self, init, sig_b, n=80000, win=13, lowres=96, seed=0):
        super().__init__()
        nb, H, W = init.shape
        self.size, self.win, self.nb = H, win, nb
        g = torch.Generator(device="cpu").manual_seed(seed)
        I = (init / sig_b[:, None, None]).clamp(min=0).sum(0).cpu()
        p = (I + 1.0).pow(0.6).flatten()
        idx = torch.multinomial(p / p.sum(), n, replacement=True, generator=g)
        y, x = (idx // W).float(), (idx % W).float()
        jit = torch.rand(n, 2, generator=g) - 0.5
        dev = init.device
        self.mu = nn.Parameter(torch.stack([x, y], 1).add(jit).to(dev))
        self.log_s = nn.Parameter(torch.full((n, 2), math.log(0.7), device=dev))
        self.theta = nn.Parameter(torch.zeros(n, device=dev))
        cnt = torch.bincount(idx, minlength=H * W).float().clamp(min=1)[idx]
        a0 = (init.cpu() / sig_b.cpu()[:, None, None]).flatten(1)[:, idx] / cnt  # share pixel flux among its splats
        self.flux = nn.Parameter((a0.T * 0.5).to(dev))
        self.register_buffer("sig_b", sig_b.clone())
        lo = F.adaptive_avg_pool2d(init[None], lowres)[0] / sig_b[:, None, None] * 0.5
        self.low = nn.Parameter(lo)
        r = torch.arange(win, device=dev) - win // 2
        self.register_buffer("r", r)

    def param_groups(self, base):
        return [{"params": [self.flux, self.low], "lr": 2e-2},
                {"params": [self.mu], "lr": 2e-2},
                {"params": [self.log_s, self.theta], "lr": 1e-2}]

    def forward(self):
        H = W = self.size
        c = self.mu.detach().round().long()
        ox = c[:, 0, None, None] + self.r[None, None, :]
        oy = c[:, 1, None, None] + self.r[None, :, None]
        dx = ox.float() - self.mu[:, 0, None, None]
        dy = oy.float() - self.mu[:, 1, None, None]
        s = self.log_s.exp().clamp(0.2, self.win / 5.0)
        ct, st = torch.cos(self.theta)[:, None, None], torch.sin(self.theta)[:, None, None]
        u, v = ct * dx + st * dy, -st * dx + ct * dy
        g = torch.exp(-0.5 * ((u / s[:, 0, None, None]) ** 2 + (v / s[:, 1, None, None]) ** 2))
        g = g / (2 * math.pi * s[:, 0, None, None] * s[:, 1, None, None])  # unit integrated flux
        ok = ((ox >= 0) & (ox < W) & (oy >= 0) & (oy < H)).float()
        lin = (oy.clamp(0, H - 1) * W + ox.clamp(0, W - 1)).flatten()
        vals = (g * ok).flatten(1)  # n, w*w
        out = torch.zeros(self.nb, H * W, device=g.device, dtype=g.dtype)
        contrib = (self.flux.T[:, :, None] * vals[None]).reshape(self.nb, -1)
        out = out.index_add(1, lin, contrib).reshape(self.nb, H, W)
        low = F.interpolate(self.low[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        return (out + low) * self.sig_b[:, None, None]


def sample(img, tx, ty):
    """Bilinear sample a (C,H,W) or (H,W) image at target pixel coords (0-based centres) -> (C,N) or (N,)."""
    squeeze = img.dim() == 2
    if squeeze:
        img = img[None]
    C, H, W = img.shape
    gx = (tx + 0.5) / W * 2 - 1
    gy = (ty + 0.5) / H * 2 - 1
    g = torch.stack([gx, gy], -1)[None, None]
    out = F.grid_sample(img[None], g, mode="bilinear", padding_mode="zeros", align_corners=False)[0, :, 0]
    return out[0] if squeeze else out


def convolve_basis(S, B):
    """S (nb,H,W), B (nb,nk,n,n) -> (nb,nk,H,W); FFT convolution with reflect padding (same result as conv2d)."""
    if PSF_CONVENTION == "convolution":  # v25: true convolution m(x) = sum_d B(d) S(x - d)
        B = B.flip(-2, -1)
    nb, nk, n, _ = B.shape
    H, W = S.shape[-2:]
    p = n // 2
    Sp = F.pad(S[None], (p,) * 4, mode="reflect")[0]  # nb, H+2p, W+2p
    Hp, Wp = Sp.shape[-2:]
    Kp = torch.zeros(nb, nk, Hp, Wp, device=S.device, dtype=S.dtype)
    Kp[..., :n, :n] = B.flip(-2, -1)  # conv2d is correlation; flip for true convolution equivalence below
    Kp = torch.roll(Kp, shifts=(-(n // 2), -(n // 2)), dims=(-2, -1))
    out = torch.fft.irfft2(torch.fft.rfft2(Sp)[:, None] * torch.fft.rfft2(Kp), s=(Hp, Wp))
    return out[..., p:p + H, p:p + W]


def effective_psf(B, w):
    """B (nk,n,n), w (nk,) -> (n,n)."""
    return (w[:, None, None] * B).sum(0)


TARGET_SIG_PX = 0.0  # Gaussian target resolution r (sigma, target px); 0 = off
WIENER_EPS = 1e-3


def partial_kernels(B, sig_px, eps=WIENER_EPS):
    """Deconvolution with correct sampling (Magain, Courbin & Sohy 1998): the sky grid holds r*s with r a Gaussian of
    sigma sig_px, so the instrument sees it through t = K / r. B (..., n, n) -> t (..., n, n) by Wiener division in
    Fourier space (K and r both centred). Zero-sum kernels stay zero-sum, unit-sum stay unit-sum."""
    if sig_px <= 0:
        return B
    n = B.shape[-1]
    N = 2 * n
    f = torch.fft.fftfreq(N, device=B.device)
    f2 = f[:, None] ** 2 + f[None, :] ** 2
    G = torch.exp(-2 * math.pi ** 2 * sig_px ** 2 * f2)
    Kp = torch.zeros(*B.shape[:-2], N, N, device=B.device, dtype=B.dtype)
    Kp[..., :n, :n] = B
    Kp = torch.roll(Kp, shifts=(-(n // 2), -(n // 2)), dims=(-2, -1))  # centre at (0, 0)
    T = torch.fft.fft2(Kp) * G / (G * G + eps)
    t = torch.fft.ifft2(T).real
    t = torch.roll(t, shifts=(n // 2, n // 2), dims=(-2, -1))[..., :n, :n]
    return t


def gauss_blur(S, sig_px):
    k = 2 * int(math.ceil(4 * sig_px)) + 1
    g = gauss_kernel(torch.full((S.shape[0],), sig_px, device=S.device), k, S.device)
    return F.conv2d(F.pad(S[None], (k // 2,) * 4, mode="reflect"), g[:, None], groups=S.shape[0])[0]


def tnv(S, sig):
    """Total nuclear variation (Holt 2014; Rodriguez & Wohlberg): sum over pixels of the nuclear norm of the
    (channels x 2) Jacobian. Favours edges shared by all bands (structure transfer from sharp to blurred bands)."""
    Z = S / sig[:, None, None]
    gx = Z[:, :-1, 1:] - Z[:, :-1, :-1]
    gy = Z[:, 1:, :-1] - Z[:, :-1, :-1]
    a, b, c = (gx * gx).sum(0), (gx * gy).sum(0), (gy * gy).sum(0)
    tr, det = a + c, (a * c - b * b).clamp(min=0)
    # sigma1 + sigma2 = sqrt(tr + 2 sqrt(det))
    return torch.sqrt(tr + 2 * torch.sqrt(det + 1e-12) + 1e-8).mean() / S.shape[0] ** 0.5


def band_excess(S, sig, sig_px):
    """MCS-style smoothness: energy of the sky outside the band of the target resolution r."""
    return ((S - gauss_blur(S, sig_px)) / sig[:, None, None]).pow(2).mean()


class SplatCoeffs(nn.Module):
    """Compact sources as Gaussians with spectral coefficients, rendered at the target resolution
    (mip filter: rendered variance = own variance + r^2, as in Mip-Splatting, Yu et al. 2024)."""

    def __init__(self, xy, amp, H, W, sig_r, win=13):
        super().__init__()
        self.H, self.W, self.win, self.sig_r = H, W, win, sig_r
        self.mu = nn.Parameter(xy.clone())
        self.log_s = nn.Parameter(torch.full((xy.shape[0], 2), math.log(0.3), device=xy.device))
        self.theta = nn.Parameter(torch.zeros(xy.shape[0], device=xy.device))
        self.amp = nn.Parameter(amp.clone())  # N K integrated coefficient flux
        self.register_buffer("r", torch.arange(win, device=xy.device) - win // 2)

    def forward(self):
        H, W = self.H, self.W
        c = self.mu.detach().round().long()
        ox = c[:, 0, None, None] + self.r[None, None, :]
        oy = c[:, 1, None, None] + self.r[None, :, None]
        dx = ox.float() - self.mu[:, 0, None, None]
        dy = oy.float() - self.mu[:, 1, None, None]
        s2 = self.log_s.exp().clamp(0.05, self.win / 5.0) ** 2 + self.sig_r ** 2
        ct, st = torch.cos(self.theta)[:, None, None], torch.sin(self.theta)[:, None, None]
        u, v = ct * dx + st * dy, -st * dx + ct * dy
        g = torch.exp(-0.5 * (u * u / s2[:, 0, None, None] + v * v / s2[:, 1, None, None]))
        g = g / (2 * math.pi * torch.sqrt(s2[:, 0] * s2[:, 1]))[:, None, None]
        ok = ((ox >= 0) & (ox < W) & (oy >= 0) & (oy < H)).float()
        lin = (oy.clamp(0, H - 1) * W + ox.clamp(0, W - 1)).flatten()
        vals = (g * ok).flatten(1)
        K = self.amp.shape[1]
        out = torch.zeros(K, H * W, device=g.device, dtype=g.dtype)
        out = out.index_add(1, lin, (self.amp.T[:, :, None] * vals[None]).reshape(K, -1))
        return out.reshape(K, H, W)


class HybridCoeffs(nn.Module):
    """Spectral coefficient maps = smooth pixel grid + compact Gaussian sources."""

    def __init__(self, grid, splats):
        super().__init__()
        self.grid, self.splats = grid, splats

    def forward(self):
        return self.grid() + self.splats()


class ConvDecoderCoeffs(nn.Module):
    """Untrained-network prior (deep image prior, Ulyanov et al. 2018) in the ConvDecoder form (Darestani & Heckel
    2021): a fixed random input upsampled through conv + ReLU + BatchNorm layers to the K coefficient maps;
    a = s0 * sinh(net) to cover the dynamic range. Early stopping by the validation exposures."""

    def __init__(self, K, H, s0, base=32, chans=(128, 128, 128, 96, 64, 48), seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        nup = len(chans) - 1
        h0 = H // 2 ** nup
        self.register_buffer("z", torch.rand(1, chans[0], h0, h0, generator=g) * 0.1)
        layers = []
        cin = chans[0]
        for i, c in enumerate(chans[1:]):
            layers += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                       nn.Conv2d(cin, c, 3, padding=1, padding_mode="reflect"), nn.ReLU(), nn.BatchNorm2d(c)]
            cin = c
        layers += [nn.Conv2d(cin, cin, 3, padding=1, padding_mode="reflect"), nn.ReLU(), nn.BatchNorm2d(cin), nn.Conv2d(cin, K, 1)]
        self.net = nn.Sequential(*layers)
        self.s0, self.H = s0, H

    def raw(self):
        return self.net(self.z)[0]

    def forward(self):
        return self.s0 * torch.sinh(self.raw())


def tv(s, sig):
    return (((s[:, 1:, :] - s[:, :-1, :]).abs().mean((1, 2)) + (s[:, :, 1:] - s[:, :, :-1]).abs().mean((1, 2))) / sig).mean()


class SkyModel(nn.Module):
    def __init__(self, sky, psf, bkg, geom, band_index, stripes=None, sig=1.0):
        super().__init__()
        self.sky, self.psf, self.bkg, self.geom = sky, psf, bkg, geom
        self.stripes = stripes
        self.sig = sig
        self.band_index = band_index

    def prepare(self):
        if hasattr(self.sky, "nodes"):
            Sn, S = self.sky.nodes()  # nb Q H W ; nb H W
            nb, q = Sn.shape[:2]
            lam = self.sky.lam_nodes.flatten()
            if hasattr(self.psf, "basis_lam"):  # PSF evaluated at every quadrature wavelength
                B = self.psf.basis_lam(lam, self.psf.chan_idx.repeat_interleave(q))
            else:  # per-band PSF for all nodes of the band
                B = self.psf.basis().repeat_interleave(q, 0)
            B = partial_kernels(B, TARGET_SIG_PX)
            CBn = convolve_basis(Sn.flatten(0, 1), B)  # nbQ nk H W
            CB = (CBn.view(nb, q, *CBn.shape[1:]) * self.sky.w_nodes[:, :, None, None, None]).sum(1)
            return S, CB
        S = self.sky()
        return S, convolve_basis(S, partial_kernels(self.psf.basis(), TARGET_SIG_PX))

    def predict(self, CB, i, e):
        b = self.band_index[e["band"]]
        i = e.get("eidx", i)
        det = e.get("detv", e["det"])
        x, y = self.geom(i, det, e["tx"], e["ty"], e["dx"], e["dy"])
        v = sample(CB[b], x, y)  # nk N
        w = self.psf.weights(b, det, e["dx"], e["dy"])  # N nk
        m = self.bkg(i, det, e["dx"], e["dy"]) + self.bkg.gain(vec(i, e["dx"])) * (w * v.T).sum(1)
        if self.stripes is not None and uses_stripes(e["band"]):  # v29: 1/f stripes are a NIRCam effect
            m = m + self.stripes(vec(i, e["dx"]), e["row"], e["col"])
        return m

    def penalty(self):
        p = self.psf.penalty() + self.geom.penalty() + self.bkg.penalty()
        if hasattr(self.sky, "penalty"):
            p = p + self.sky.penalty()
        if self.stripes is not None:
            p = p + self.stripes.penalty(self.sig)
        return p


def chi2(exps, idx, pred_fn):
    num, n = 0.0, 0
    for i in idx:
        e = exps[i]
        r = e["d"] - pred_fn(i, e)
        num += float((e["ivar"] * r * r).sum())
        n += e["d"].numel()
    return num / max(n, 1)


def band_batches(exps, idx, nchunks=1, seed=0):
    out = []
    g = torch.Generator().manual_seed(seed)
    for b in REG.bands:
        sel = [i for i in idx if exps[i]["band"] == b]
        if not sel:
            continue
        dev = exps[sel[0]]["d"].device
        cat = lambda k: torch.cat([exps[i][k] for i in sel])
        full = {"tx": cat("tx"), "ty": cat("ty"), "d": cat("d"), "ivar": cat("ivar"), "dx": cat("dx"), "dy": cat("dy"),
                "row": cat("row"), "col": cat("col"),
                "eidx": torch.cat([torch.full((exps[i]["d"].numel(),), i, dtype=torch.long, device=dev) for i in sel]),
                "detv": torch.cat([torch.full((exps[i]["d"].numel(),), exps[i]["det"], dtype=torch.long, device=dev) for i in sel])}
        n = full["d"].numel()
        perm = torch.randperm(n, generator=g).to(dev)
        for c in range(nchunks):
            ix = perm[c::nchunks]
            ch = {k: v[ix].contiguous() for k, v in full.items()}
            ch.update(band=b, det=exps[sel[0]]["det"], chunk=c)
            out.append(ch)
        del full
    return out


def fit(name, model, exps, steps, lrs, lam_tv, sig_b, train_idx, log_every=100, budget_s=None, nchunks=4,
        callback=None, cb_every_s=None, ckpt_every_s=1800.0, patience=0):
    if getattr(args_global, "optimizer", "adam") == "lbfgs":
        return fit_lbfgs(name, model, exps, lam_tv, sig_b, train_idx, budget_s=budget_s, callback=callback,
                         cb_every_s=cb_every_s, patience=patience)
    sky_groups = model.sky.param_groups(lrs["sky"]) if hasattr(model.sky, "param_groups") else [{"params": list(model.sky.parameters()), "lr": lrs["sky"]}]
    groups = sky_groups + [
              {"params": [p for p in model.psf.parameters() if p.requires_grad], "lr": lrs["psf"]},
              {"params": [p for p in model.bkg.parameters() if p.requires_grad]
               + (list(model.stripes.parameters()) if model.stripes is not None else []), "lr": lrs["bkg"]}]
    gp = [p for p in model.geom.parameters() if p.requires_grad]
    if gp:
        groups.append({"params": gp, "lr": lrs["geom"]})
    opt = torch.optim.Adam(groups)
    base_lr = [g["lr"] for g in groups]

    def set_lr(frac):  # warm-up then cosine, on progress = max(step fraction, time fraction)
        wu = max(0.03, 20.0 / steps)
        f = max(frac, 1e-3) / wu if frac < wu else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - wu) / (1 - wu))))
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * f
    ntot = sum(exps[i]["d"].numel() for i in train_idx)
    batches = band_batches(exps, train_idx, nchunks)
    _LCACHE.clear()
    hist = []
    it0, el0 = 0, 0.0
    ck = SAVE_DIR / f"{name}_ckpt.pt" if SAVE_DIR is not None else None
    if ck is not None and ck.exists():  # resume a long run
        st = torch.load(ck, map_location=next(model.parameters()).device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        it0, el0, hist = st["it"] + 1, st["elapsed"], st["hist"]
        LOG.update(st.get("log", {}))
        sc = getattr(args_global, "resume_lr_scale", 1.0)
        if sc != 1.0:  # continuing a finished schedule: keep the step size near where it ended
            base_lr = [lr * sc for lr in base_lr]
            tick(f"{name}: learning rates scaled by {sc} after resume")
        tick(f"{name}: resumed at step {it0}, {el0 / 3600:.2f} h done")
    t_start = time.perf_counter() - el0
    t_ck = t_cb = time.perf_counter()
    best, bad = None, 0
    if SAVE_DIR is not None and (SAVE_DIR / f"{name}_best.pt").exists() and it0 > 0:
        best = (float(LOG.get(f"{name}_best_score", float("inf"))), -1, torch.load(SAVE_DIR / f"{name}_best.pt", map_location="cpu"))
    set_lr(max(it0 / steps, el0 / budget_s if budget_s else 0.0))
    it = it0
    for it in range(it0, steps):
        opt.zero_grad(set_to_none=True)
        S, CB = model.prepare()
        # memory-bounded: per-exposure backward into a detached copy of the convolved basis images,
        # then one backward through sky/PSF
        CBl = CB.detach().requires_grad_(True)
        loss = 0.0
        for e in batches:
            if e["chunk"] != it % nchunks:
                continue
            m = model.predict(CBl, None, e)
            li = rho(e["ivar"] * (e["d"] - m) ** 2).sum() * nchunks / ntot
            li.backward()
            loss += float(li.detach())
        reg = model.penalty()
        if lam_tv > 0:
            reg = reg + lam_tv * tv(S, sig_b)
        if LAM_TNV > 0:
            reg = reg + LAM_TNV * tnv(S, sig_b)
        if LAM_BAND > 0 and TARGET_SIG_PX > 0:
            reg = reg + LAM_BAND * band_excess(S, sig_b, TARGET_SIG_PX)
        if LAM_NEG > 0:  # the sky (background is modelled separately) should not go far below zero
            reg = reg + LAM_NEG * F.relu(-S / sig_b[:, None, None] - 2.0).pow(2).mean()
        torch.autograd.backward([CB, reg], [CBl.grad, torch.ones_like(reg)])
        opt.step()
        el = time.perf_counter() - t_start
        set_lr(max((it + 1) / steps, el / budget_s if budget_s else 0.0))
        stop = budget_s is not None and time.perf_counter() - t_start > budget_s
        if it % log_every == 0 or it == steps - 1 or stop:
            hist.append([it, loss])
            tick(f"{name} step {it} chi2/pix={loss:.4f}")
        now = time.perf_counter()
        if ck is not None and (now - t_ck > ckpt_every_s or stop):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "elapsed": now - t_start, "hist": hist,
                        "log": {k: v for k, v in LOG.items() if k.startswith(name)}}, ck)
            t_ck = now
        if callback is not None and cb_every_s and now - t_cb > cb_every_s:
            score = callback(model, it, now - t_start)
            t_cb = time.perf_counter()
            if score is not None:
                if best is None or score < best[0]:
                    best = (score, it, {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
                    bad = 0
                    if SAVE_DIR is not None:
                        torch.save(best[2], SAVE_DIR / f"{name}_best.pt")
                    LOG[f"{name}_best_score"] = score
                    tick(f"{name}: new best validation score {score:.4f} at step {it}")
                else:
                    bad += 1
                    if patience and bad >= patience:
                        tick(f"{name}: no validation improvement in {bad} evaluations, stopping at step {it}")
                        stop = True
        if stop:
            tick(f"{name}: time budget reached at step {it}")
            break
    LOG[f"{name}_history"] = hist
    LOG[f"{name}_steps_done"] = it + 1
    if best is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in best[2].items()})
        LOG[f"{name}_best_score"] = best[0]
        LOG[f"{name}_best_step"] = best[1]
        tick(f"{name}: restored best validation checkpoint (score {best[0]:.4f}, step {best[1]})")
    del batches
    _LCACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if SAVE_DIR is not None:
        torch.save(model.state_dict(), SAVE_DIR / f"{name}_state.pt")
        json.dump(LOG, open(SAVE_DIR / "metrics_partial.json", "w"), indent=1, default=float)
    return model


def fit_lbfgs(name, model, exps, lam_tv, sig_b, train_idx, budget_s=None, callback=None, cb_every_s=0.0, patience=0, inner=20):
    """Full-batch L-BFGS (strong-Wolfe line search) over the free parameters; same loss, priors, validation and
    best-checkpoint logic as fit()."""
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=inner, history_size=30, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-14)
    ntot = sum(exps[i]["d"].numel() for i in train_idx)
    batches = band_batches(exps, train_idx, 1)
    nev = [0]

    def closure():
        opt.zero_grad(set_to_none=True)
        S, CB = model.prepare()
        CBl = CB.detach().requires_grad_(True)
        loss = 0.0
        for e in batches:
            m = model.predict(CBl, None, e)
            li = rho(e["ivar"] * (e["d"] - m) ** 2).sum() / ntot
            li.backward()
            loss += float(li.detach())
        reg = model.penalty()
        if lam_tv > 0:
            reg = reg + lam_tv * tv(S, sig_b)
        if LAM_NEG > 0:
            reg = reg + LAM_NEG * F.relu(-S / sig_b[:, None, None] - 2.0).pow(2).mean()
        torch.autograd.backward([CB, reg], [CBl.grad, torch.ones_like(reg)])
        nev[0] += 1
        closure.last = loss
        return torch.tensor(loss + float(reg.detach()), device=S.device)

    t0 = time.perf_counter(); t_cb = t0
    best, bad, outer, hist = None, 0, 0, []
    while True:
        opt.step(closure)
        outer += 1
        el = time.perf_counter() - t0
        hist.append([nev[0], round(el, 1), closure.last])
        tick(f"{name} L-BFGS outer {outer}, {nev[0]} evaluations, {el / 60:.1f} min: chi2/pix={closure.last:.4f}")
        stop = budget_s is not None and el > budget_s
        if callback is not None and (time.perf_counter() - t_cb > cb_every_s or stop):
            score = callback(model, nev[0], el)
            t_cb = time.perf_counter()
            if score is not None:
                if best is None or score < best[0]:
                    best = (score, nev[0], {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
                    bad = 0
                    LOG[f"{name}_best_score"] = score
                    tick(f"{name}: new best validation score {score:.4f} after {nev[0]} evaluations")
                else:
                    bad += 1
                    if patience and bad >= patience:
                        stop = True
        if stop:
            break
    LOG[f"{name}_history"] = hist
    if best is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in best[2].items()})
        tick(f"{name}: restored best validation checkpoint (score {best[0]:.4f})")
    if SAVE_DIR is not None:
        torch.save(model.state_dict(), SAVE_DIR / f"{name}_state.pt")
        json.dump(LOG, open(SAVE_DIR / "metrics_partial.json", "w"), indent=1, default=float)
    return model


args_global = None
SAVE_DIR = None
UQ = None  # uncertainty run: {"name", "var_s", "nu", "rows"}
RIGID = False  # rigid models were trained without stripes: scoring may add the held-out stripe refit


def heldout_chi2(model_or_image, exps, hold_idx, band_index, ndet, steps=150, src_masks=None):
    """Held-out prediction, identical protocol for every method: sky (and learned PSF / static distortion /
    detector background pattern) frozen; for the held-out exposure only, an affine registration is refitted
    (Adam) and then gain + quadratic background are solved exactly by weighted least squares.
    Returns chi2/N over all pixels, chi2/N on source pixels, residual maps."""
    tot, n, tot_s, n_s = 0.0, 0, 0.0, 0
    rob, rob_s = 0.0, 0.0
    phot_acc = {}
    per_band = {}
    per_inst = {}  # v29: roll-up by instrument
    resid = {}
    per_exposure = []  # v22: one record per scored exposure
    heldout_chi2.per_exposure = per_exposure
    for i in hold_idx:
        e = exps[i]
        sub = [e]
        b = band_index[e["band"]]
        geom = Geometry(sub, ndet, "fixed")
        geom.M.requires_grad_(True)
        geom.t.requires_grad_(True)
        pattern = torch.zeros_like(e["d"])
        zipm = isinstance(model_or_image, ZipSkyModel)
        if isinstance(model_or_image, (SkyModel, ZipSkyModel)):
            m0 = model_or_image
            with torch.no_grad():
                _, CB = (None, None) if zipm else m0.prepare()
                if m0.geom.mode != "fixed":
                    dist = m0.geom.distortion(e["det"], e["dx"], e["dy"])
                else:
                    dist = torch.zeros(e["d"].numel(), 2, device=e["d"].device)
                w = m0.psf.weights(b, e["det"], e["dx"], e["dy"])
                if isinstance(m0.bkg, ExplicitBackground):
                    pattern = (legendre2d(e["dx"], e["dy"], 4) * m0.bkg.pd[e["det"]]).sum(1)
                else:  # implicit: detector pattern = field at the mean exposure embedding
                    N = e["dx"].numel()
                    inp = torch.cat([fourier(torch.stack([e["dx"], e["dy"]], -1) * 0.5, 4),
                                     m0.bkg.eemb.weight.mean(0, keepdim=True).expand(N, -1),
                                     m0.bkg.demb.weight[vec(e["det"], e["dx"])]], 1)
                    pattern = m0.bkg.scale * m0.bkg.net(inp)[:, 0]
            CB = CB[b].detach() if CB is not None else None
        else:
            zipm = False
            CB = model_or_image[b][None]
            dist = torch.zeros(e["d"].numel(), 2, device=e["d"].device)
            w = torch.ones(e["d"].numel(), 1, device=e["d"].device)
        tx0, ty0 = e["tx"] + dist[:, 0], e["ty"] + dist[:, 1]
        L = legendre2d(e["dx"], e["dy"], 2)

        zgen = torch.Generator(device=e["d"].device)

        def core(final=False):
            x, y = geom(0, e["det"], tx0, ty0, e["dx"], e["dy"])
            if zipm:
                zgen.manual_seed(1234)  # same samples at every call: a smooth objective for the registration refit
                return m0.core(b, e["det"], x, y, 32 if final else 8, zgen)
            return (w * sample(CB, x, y).T).sum(1)

        def solve(c):
            return solve_with(c, torch.zeros_like(c))

        def solve_with(c, st):
            A = torch.cat([c[:, None], L], 1)  # gain, background terms
            y = e["d"] - e["bkg0"] - pattern - st
            sw = e["ivar"].sqrt()
            Aw = (A * sw[:, None]).double()
            yw = (y * sw).double()
            AtA, Aty = (Aw.T @ Aw).cpu(), (Aw.T @ yw).cpu()  # tiny normal equations, solved on CPU (no cusolver on MIG)
            AtA = AtA + 1e-9 * torch.eye(AtA.shape[0], dtype=AtA.dtype) * AtA.diagonal().mean()
            coef = torch.linalg.solve(AtA, Aty).to(A.device).float()
            solve_with.gain = coef[0]
            return e["bkg0"] + pattern + A @ coef

        opt = torch.optim.Adam([geom.M, geom.t], lr=0.01)
        for _ in range(steps):
            opt.zero_grad()
            c = core()
            with torch.no_grad():
                pass
            m = solve(c)
            r = e["d"] - m
            l = rho(e["ivar"] * r * r).mean() + 1e4 * geom.M.pow(2).sum()
            l.backward()
            opt.step()
        with torch.no_grad():
            c = core(final=True)
            m = solve(c)
            if getattr(args_global, "stripes", 0) and uses_stripes(e["band"]):  # same 1/f treatment for every method: row/col weighted means (NIRCam)
                st = torch.zeros_like(m)
                for _ in range(3):
                    for key in ("row", "col"):
                        r = e["d"] - m - st
                        idx = e[key]
                        num = torch.zeros(2048, device=r.device).index_add_(0, idx, e["ivar"] * r)
                        den = torch.zeros(2048, device=r.device).index_add_(0, idx, e["ivar"]).clamp(min=1e-12)
                        st = st + (num / den)[idx]
                    m = solve_with(c, st)
                m = m + st
            r = e["d"] - m
            if UQ is not None and isinstance(model_or_image, SkyModel) and model_or_image is UQ["model"]:
                xg_, yg_ = geom(0, e["det"], tx0, ty0, e["dx"], e["dy"])
                sd = UQ["var_s"].sqrt()
                Kb = partial_kernels(model_or_image.psf.basis(), TARGET_SIG_PX)
                vm = torch.zeros_like(r)
                for _ in range(UQ["nu"]):
                    u = torch.randint(0, 2, sd.shape, device=sd.device, dtype=sd.dtype) * 2 - 1
                    CBp = convolve_basis(sd * u, Kb)[b]
                    cp = (w * sample(CBp, xg_, yg_).T).sum(1)
                    vm += (solve_with.gain * cp) ** 2
                vm /= UQ["nu"]
                gsel = torch.Generator(device="cpu").manual_seed(int(i))
                keep = (torch.rand(r.numel(), generator=gsel) < 0.03).to(r.device)
                srcm = sample(src_masks[b][None].float(), e["tx"], e["ty"])[0] > 0.5 if src_masks is not None else torch.zeros_like(keep)
                keep = (keep | srcm) & (e["ivar"] > 0)
                UQ["rows"].append({"band": e["band"], "split": "test" if e.get("holdout") else ("val" if e.get("val") else "train"),
                                   "r": r[keep].cpu().numpy(), "ivar": e["ivar"][keep].cpu().numpy(), "vm": vm[keep].cpu().numpy(),
                                   "src": srcm[keep].cpu().numpy(), "pred": (solve_with.gain * c)[keep].cpu().numpy()})
            chi = e["ivar"] * r * r
            d3 = 9.0  # robust score: pseudo-Huber at 3 sigma, independent of the training loss
            rh = 2.0 * d3 * (torch.sqrt(1.0 + chi / d3) - 1.0)
            tot += float(chi.sum())
            rob += float(rh.sum())
            n += r.numel()
            acc = per_band.setdefault(e["band"], {"chi2": 0.0, "robust": 0.0, "n": 0, "chi2_src": 0.0, "robust_src": 0.0, "n_src": 0, "n_exp": 0})
            acc["chi2"] += float(chi.sum()); acc["robust"] += float(rh.sum()); acc["n"] += r.numel(); acc["n_exp"] += 1
            ia = per_inst.setdefault(REG.instrument.get(e["band"], "?"), {"chi2": 0.0, "robust": 0.0, "n": 0, "chi2_src": 0.0, "robust_src": 0.0, "n_src": 0, "n_exp": 0})
            ia["chi2"] += float(chi.sum()); ia["robust"] += float(rh.sum()); ia["n"] += r.numel(); ia["n_exp"] += 1
            pe = {"name": e["name"], "band": e["band"], "det": int(e["det"]), "n": int(r.numel()), "n_src": 0, "chi2_sum": float(chi.sum()),
                  "chi2_src_sum": 0.0, "robust_sum": float(rh.sum()), "robust_src_sum": 0.0, "pulls": []}
            if REG.multi:
                pe["instrument"] = REG.instrument.get(e["band"], "?")
            per_exposure.append(pe)
            if src_masks is not None:
                sm = sample(src_masks[b][None].float(), e["tx"], e["ty"])[0] > 0.5
                tot_s += float(chi[sm].sum())
                rob_s += float(rh[sm].sum())
                n_s += int(sm.sum())
                acc["chi2_src"] += float(chi[sm].sum()); acc["robust_src"] += float(rh[sm].sum()); acc["n_src"] += int(sm.sum())
                ia["chi2_src"] += float(chi[sm].sum()); ia["robust_src"] += float(rh[sm].sum()); ia["n_src"] += int(sm.sum())
                pe["n_src"], pe["chi2_src_sum"], pe["robust_src_sum"] = int(sm.sum()), float(chi[sm].sum()), float(rh[sm].sum())
            if APERTURES is not None:  # aperture photometry and centroids of catalogue sources on this exposure
                with torch.no_grad():
                    xg, yg = geom(0, e["det"], tx0, ty0, e["dx"], e["dy"])
                    sky_pred = solve_with.gain * c  # gain x model sky, as seen by this exposure
                    data_sky = e["d"] - (m - sky_pred)  # data minus the same fitted background/stripes
                    lab, cxy, Rpx = APERTURES
                    Hh, Ww = lab.shape
                    ix = xg.round().long().clamp(0, Ww - 1)
                    iy = yg.round().long().clamp(0, Hh - 1)
                    sid = lab[iy, ix]
                    ok = sid >= 0
                    ns = int(cxy.shape[0])
                    idx_ = sid[ok]
                    def bsum(v):
                        return torch.zeros(ns, device=v.device, dtype=torch.float64).index_add_(0, idx_, v[ok].double())
                    npx = bsum(torch.ones_like(data_sky))
                    fd, fm = bsum(data_sky), bsum(sky_pred)
                    var = bsum(1.0 / e["ivar"].clamp(min=1e-20))
                    xd, xm = bsum(data_sky * xg), bsum(sky_pred * xg)
                    yd, ym = bsum(data_sky * yg), bsum(sky_pred * yg)
                    full = npx >= 0.9 * npx[npx > 0].median() if (npx > 0).any() else npx > 0
                    good = full & (fm > 0) & (fd > 0) & (fm / var.clamp(min=1e-30).sqrt() > 20)
                    if good.any():
                        dm = -2.5 * torch.log10(fd[good] / fm[good])
                        pull = (fd[good] - fm[good]) / var[good].sqrt()
                        dx_ = (xd[good] / fd[good] - xm[good] / fm[good]) * args_global.scale * 1000
                        dy_ = (yd[good] / fd[good] - ym[good] / fm[good]) * args_global.scale * 1000
                        P = phot_acc.setdefault(e["band"], {"dm": [], "pull": [], "dx": [], "dy": []})
                        P["dm"].append(dm.cpu()); P["pull"].append(pull.cpu()); P["dx"].append(dx_.cpu()); P["dy"].append(dy_.cpu())
                        pe["pulls"] = [float(x) for x in pull.cpu().tolist()]
            if e["band"] not in resid:  # first scored exposure of each band
                resid[e["band"]] = (r * e["ivar"].sqrt()).cpu().numpy()
    for bnd, acc in list(per_band.items()) + list(per_inst.items()):  # per-band (and per-instrument) means over every scored exposure
        nn, ns = max(acc.pop("n"), 1), acc.pop("n_src")
        acc["chi2"] /= nn; acc["robust"] /= nn
        acc["chi2_src"] = acc["chi2_src"] / ns if ns else None
        acc["robust_src"] = acc["robust_src"] / ns if ns else None
    def nmad_(v):
        return float(1.4826 * (v - v.median()).abs().median()) if v.numel() else float("nan")
    phot = {}
    alls = {"dm": [], "pull": [], "dx": [], "dy": []}
    for bnd, P in phot_acc.items():
        c_ = {k: torch.cat(v) for k, v in P.items()}
        for k in alls:
            alls[k].append(c_[k])
        phot[bnd] = {"n": int(c_["dm"].numel()), "dm_median_mmag": float(c_["dm"].median() * 1000), "dm_nmad_mmag": nmad_(c_["dm"]) * 1000,
                     "pull_nmad": nmad_(c_["pull"]), "centroid_nmad_mas": float(math.hypot(nmad_(c_["dx"]), nmad_(c_["dy"])) / math.sqrt(2))}
    if alls["dm"]:
        c_ = {k: torch.cat(v) for k, v in alls.items()}
        phot["all"] = {"n": int(c_["dm"].numel()), "dm_median_mmag": float(c_["dm"].median() * 1000), "dm_nmad_mmag": nmad_(c_["dm"]) * 1000,
                       "pull_nmad": nmad_(c_["pull"]), "centroid_nmad_mas": float(math.hypot(nmad_(c_["dx"]), nmad_(c_["dy"])) / math.sqrt(2))}
    heldout_chi2.photometry = phot
    heldout_chi2.last = {"robust": rob / max(n, 1), "robust_src": (rob_s / max(n_s, 1) if src_masks is not None else None),
                         "per_band": per_band, "per_instrument": per_inst}
    return tot / max(n, 1), (tot_s / max(n_s, 1) if src_masks is not None else None), resid


# ----------------------------------------------------------------------------- display
# ----------------------------------------------------------------------------- v22: names, selection, bootstrap
import re as _re

# [NNN_]jwPPPPPOOOVVV_GGSAA_EEEEE_<detector>_...   (PPPPP programme, OOO observation, VVV visit, GGSAA visit group /
# parallel seq / activity, EEEEE exposure number = dither position, detector e.g. nrcb1 or nrcblong).  The optional NNN_
# is the sorted-file index prefix present on the files of this data set.
_NAME_RE = _re.compile(r"^(?:(?P<idx>\d+)_)?jw(?P<prog>\d{5})(?P<obs>\d{3})(?P<vis>\d{3})_(?P<grp>[0-9a-z]{5})_(?P<exp>\d{5})_(?P<det>[a-z0-9]+)(?:_|\.|$)",
                       _re.IGNORECASE)


_HST_NAME_RE = _re.compile(r"^(?P<root>[a-z][a-z0-9]{8})_sci(?P<ext>\d+)$", _re.IGNORECASE)  # hst_io chip names: <rootname>_sci<ext>


def _parse_hst_name(name):
    """HST chip name <ipppssoot>_sci<N> (i instrument, ppp programme, ss visit, oo observation, t transmission). visit = instrument +
    programme + visit (root[:6]); dither = one exposure = all chips (sci1, sci2) of one rootname."""
    m = _HST_NAME_RE.match(os.path.basename(str(name)))
    if m is None:
        return None
    root = m.group("root").lower()
    visit = (root[:6], "hst")
    return {"prog": root[1:4], "obs": root[4:6], "vis": root[4:6], "grp": "hst", "exp": root[6:], "det": "sci" + m.group("ext"), "idx": None,
            "visit": visit, "dither": (visit, root[6:]), "root": root}


def hst_root(name):
    p = _parse_hst_name(name)
    return p["root"] if p else None


def read_holdout_names(path):
    """Names to hold out from a file: a metrics.json / holdout_selection.json ({"holdout": [...]}), a JSON list, or one name per line."""
    txt = open(path).read()
    try:
        obj = json.loads(txt)
    except ValueError:
        return [ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return list(obj["holdout"]) if isinstance(obj, dict) else list(obj)


def parse_exposure_name(name):
    """Parse a JWST NIRCam file name. Returns a dict (prog, obs, vis, grp, exp, det, idx, visit, dither) or None.
    visit  = (obs+vis, grp): the observation/visit number plus the visit-group/activity code, so that the key
             separates both different JWST visits and different activity groups inside one visit;
    dither = (visit, exp): all detectors that share an exposure number of one visit (one dither / time stamp)."""
    m = _NAME_RE.match(os.path.basename(str(name)))
    if m is None:
        return _parse_hst_name(name)
    g = m.groupdict()
    out = {"prog": g["prog"], "obs": g["obs"], "vis": g["vis"], "grp": g["grp"], "exp": g["exp"], "det": g["det"].lower(),
           "idx": int(g["idx"]) if g["idx"] is not None else None}
    out["visit"] = (g["obs"] + g["vis"], g["grp"])
    out["dither"] = (out["visit"], g["exp"])
    return out


def dither_key(name):
    p = parse_exposure_name(name)
    return p["dither"] if p else (("?", "?"), str(name))  # unparsable: every file is its own dither


def visit_key(name):
    p = parse_exposure_name(name)
    return p["visit"] if p else ("?", "?")


def _auto_holdout(exps, bands, n_test, warnings):
    """The v21 (n_test <= 1) / v22 (n_test > 1, dithers over visits) holdout rules on the given bands (unchanged from v28)."""
    if n_test <= 1:
        for b in bands:
            bl = [e for e in exps if e["band"] == b]
            if len(bl) >= 3:  # hold out the exposure covering the most of the field
                max(bl, key=lambda e: e["d"].numel())["holdout"] = True
        return
    for b in bands:
        bl = [e for e in exps if e["band"] == b]
        dith = {}
        for e in bl:
            dith.setdefault(dither_key(e["name"]), []).append(e)
        n_eff = min(n_test, len(dith) - 2)
        if n_eff < n_test:
            warnings.append(f"{b}: {len(dith)} dithers, holding out {max(n_eff, 0)} instead of {n_test}")
        if n_eff < 1:
            continue
        px = {k: sum(e["d"].numel() for e in v) for k, v in dith.items()}
        byvis = {}
        for k in dith:
            byvis.setdefault(k[0], []).append(k)
        for v in byvis.values():
            v.sort(key=lambda k: (-px[k], k))
        order = sorted(byvis, key=lambda v: (-px[byvis[v][0]], v))
        chosen = []
        while len(chosen) < n_eff:
            for v in order:
                if byvis[v] and len(chosen) < n_eff:
                    chosen.append(byvis[v].pop(0))
        for k in chosen:
            for e in dith[k]:
                e["holdout"] = True


def select_exposures(exps, n_test=1, n_val=0, holdout_names=None, no_holdout=False, bands=None, fill_missing_bands=False):
    """Sets e["holdout"] / e["val"] on every exposure dict (keys needed: name, band, d) and returns
    {"holdout": [names], "validation": [names], "warnings": [...]}.
    * no_holdout: nothing held out (v21).  holdout_names: reuse a list (v21; --holdout-from).
    * n_test == 1 and no holdout_names: v21 rule, the single detector image with most pixels per band (bands with >= 3 images).
    * n_test > 1: per band, n_test *dithers* (all detector images sharing visit + exposure number), chosen round-robin over the
      visits of that band (visits ordered by their best dither), inside a visit by most pixels, ties by key. Deterministic.
      At least 2 dithers are always left to training/validation; if the band has fewer, fewer are held out (warning).
    * validation (n_val rounds): v21 rule on what is left, one detector image per band and round, only while >= 4 images remain.
    * v29: holdout_names + fill_missing_bands: the names are held out (exact chip name, or an HST rootname = all its chips) and every
      band with no held-out exposure after that gets the n_test rule above (n_test == 1: v21 rule; > 1: that many dithers)."""
    bands = REG.bands if bands is None else bands
    warnings = []
    for e in exps:
        e["holdout"] = False
        e["val"] = False
    if no_holdout:
        pass
    else:
        todo = bands
        if holdout_names is not None:
            names = set(holdout_names)
            for e in exps:
                e["holdout"] = e["name"] in names or hst_root(e["name"]) in names
            todo = [b for b in bands if fill_missing_bands and not any(e["holdout"] for e in exps if e["band"] == b)]
            for b in todo:
                warnings.append(f"{b}: no held-out exposure in the holdout file, selected by the n_test={n_test} rule")
        _auto_holdout(exps, todo, n_test, warnings)
    if not no_holdout:
        for _ in range(n_val):  # validation: early stopping / selection
            for b in bands:
                bl = [e for e in exps if e["band"] == b and not e["holdout"] and not e["val"]]
                if len(bl) >= 4:
                    max(bl, key=lambda e: e["d"].numel())["val"] = True
    return {"holdout": [e["name"] for e in exps if e["holdout"]], "validation": [e["name"] for e in exps if e["val"]],
            "warnings": warnings}


_BOOT_STATS = ("chi2_all", "chi2_src", "robust_all", "robust_src", "flux_err")


def _pooled_scores(recs, pulls):
    """Pooled statistics exactly as heldout_chi2 pools them: sums over exposures / total pixels; flux error = NMAD of pooled pulls
    (torch median convention, as nmad_ in heldout_chi2)."""
    n = sum(r["n"] for r in recs)
    ns = sum(r["n_src"] for r in recs)
    out = {"chi2_all": sum(r["chi2_sum"] for r in recs) / max(n, 1),
           "chi2_src": (sum(r["chi2_src_sum"] for r in recs) / ns) if ns else float("nan"),
           "robust_all": sum(r["robust_sum"] for r in recs) / max(n, 1),
           "robust_src": (sum(r["robust_src_sum"] for r in recs) / ns) if ns else float("nan")}
    v = torch.cat(pulls) if pulls else torch.zeros(0)
    out["flux_err"] = float(1.4826 * (v - v.median()).abs().median()) if v.numel() else float("nan")
    return out


def _record_instrument(r):
    return r.get("instrument") or REG.instrument.get(r["band"], "?")


def bootstrap_scores(per_exposure, B=1000, seed=0, cluster="dither"):
    """Stratified (by band) cluster bootstrap over scored exposures (the records of heldout_chi2.per_exposure; one record = one
    detector image). cluster="dither" (default): the resampling unit is a dither, i.e. all records of a band sharing dither_key(name)
    (detectors of one dither share pointing and background); cluster="image": the unit is one detector image. Each replicate
    redraws, within every band, as many units as the band has, with replacement, and recomputes the pooled statistics. Returns {stat: {"est", "p16", "p84", "p2.5", "p97.5"}, "n_exposures": {band: n},
    "single_exposure_bands": [...], "single_dither_bands": [...], "B", "seed"}."""
    assert cluster in ("dither", "image")
    by = {}  # band -> list of units; a unit is a list of records
    for r in per_exposure:
        by.setdefault(r["band"], [])
    if cluster == "dither":
        dd = {}
        for r in per_exposure:
            dd.setdefault((r["band"], dither_key(r["name"])), []).append(r)
        for (b, _), v in sorted(dd.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            by[b].append(v)
    else:
        for r in per_exposure:
            by[r["band"]].append([r])
    pul = {id(r): torch.as_tensor(r.get("pulls", []), dtype=torch.float32).reshape(-1) for r in per_exposure}
    est = _pooled_scores(list(per_exposure), [pul[id(r)] for r in per_exposure])
    rng = np.random.default_rng(seed)
    bands = sorted(by)
    samp = np.full((B, len(_BOOT_STATS)), np.nan)
    insts = sorted({_record_instrument(r) for r in per_exposure})
    samp_i = np.full((B, len(insts), len(_BOOT_STATS)), np.nan) if len(insts) > 1 else None  # v29: per-instrument replicates
    for k in range(B):
        recs, pulls = [], []
        for b in bands:
            idx = rng.integers(0, len(by[b]), len(by[b]))
            for j in idx:
                for r in by[b][j]:
                    recs.append(r)
                    pulls.append(pul[id(r)])
        sc = _pooled_scores(recs, pulls)
        samp[k] = [sc[t] for t in _BOOT_STATS]
        if samp_i is not None:
            for j_, ins in enumerate(insts):
                sel_ = [q_ for q_, r_ in enumerate(recs) if _record_instrument(r_) == ins]
                if sel_:
                    sci_ = _pooled_scores([recs[q_] for q_ in sel_], [pulls[q_] for q_ in sel_])
                    samp_i[k, j_] = [sci_[t] for t in _BOOT_STATS]
    res = {}
    for j, t in enumerate(_BOOT_STATS):
        col = samp[:, j]
        col = col[np.isfinite(col)]
        q = np.percentile(col, [2.5, 16, 84, 97.5]) if col.size else [float("nan")] * 4
        res[t] = {"est": est[t], "p16": float(q[1]), "p84": float(q[2]), "p2.5": float(q[0]), "p97.5": float(q[3])}
    if samp_i is not None:
        res["per_instrument"] = {}
        for j_, ins in enumerate(insts):
            recs_i = [r for r in per_exposure if _record_instrument(r) == ins]
            est_i = _pooled_scores(recs_i, [pul[id(r)] for r in recs_i])
            d_ = {}
            for q_, t in enumerate(_BOOT_STATS):
                col = samp_i[:, j_, q_]
                col = col[np.isfinite(col)]
                p_ = np.percentile(col, [16, 84]) if col.size else [float("nan")] * 2
                d_[t] = {"est": est_i[t], "p16": float(p_[0]), "p84": float(p_[1])}
            d_["n_images"] = len(recs_i)
            res["per_instrument"][ins] = d_
    nimg = {b: sum(len(u) for u in by[b]) for b in bands}
    res["n_exposures"] = nimg
    res["n_dithers"] = {b: len({dither_key(r["name"]) for u in by[b] for r in u}) for b in bands}
    res["single_exposure_bands"] = [b for b in bands if nimg[b] == 1]
    res["single_dither_bands"] = [b for b in bands if res["n_dithers"][b] == 1]
    res["cluster"], res["B"], res["seed"] = cluster, B, seed
    res["note"] = ("resampling unit = " + ("one dither (all its scored detector images)" if cluster == "dither" else "one detector image")
                   + "; dithers of one visit are treated as independent. Bands in single_dither_bands (cluster mode) or "
                   "single_exposure_bands contribute no resampling variance (their variance is not seen).")
    return res


def boot_line(name, bs):
    def f(t):
        d = bs[t]
        return f"{d['est']:.4f} [{d['p16']:.4f}, {d['p84']:.4f}]"
    msg = (f"BOOT {name}: src a {f('chi2_src')} | all {f('chi2_all')} | robust src {f('robust_src')} | robust all {f('robust_all')} "
           f"| flux err {f('flux_err')}  (68% interval, B={bs['B']})")
    msg += " | n_dithers " + ",".join(f"{b}:{n}" for b, n in bs["n_dithers"].items())
    for ins, d_ in bs.get("per_instrument", {}).items():  # v29
        msg += f" | {ins} (n={d_['n_images']}): src {d_['chi2_src']['est']:.4f} [{d_['chi2_src']['p16']:.4f}, {d_['chi2_src']['p84']:.4f}] all {d_['chi2_all']['est']:.4f} flux err {d_['flux_err']['est']:.4f}"
    one = bs["single_dither_bands"] if bs["cluster"] == "dither" else bs["single_exposure_bands"]
    if one:
        msg += f" | SINGLE-{'DITHER' if bs['cluster'] == 'dither' else 'EXPOSURE'} bands (variance not seen): {','.join(one)}"
    return msg


def rgb_groups(n):
    """Three channel groups (blue, green, red) of band indices: the bands in wavelength order split in three contiguous parts
    ([0,1], [2,3], [4,5] for the six NIRCam bands, as v28)."""
    order = sorted(range(n), key=lambda k: REG.lam_um[REG.bands[k]]) if n == len(REG.bands) else list(range(n))
    if n < 3:
        return [[order[0]], [order[min(1, n - 1)]], [order[-1]]]
    return [list(map(int, ch)) for ch in np.array_split(np.array(order), 3)]


def _group_sum(cube, idx):  # sum of the group, rescaled to a two-band sum (factor 1.0 for two bands: v28 values)
    acc = cube[idx[0]]
    for j in idx[1:]:
        acc = acc + cube[j]
    return acc if len(idx) == 2 else acc * (2.0 / len(idx))


def rgb_from_cube(cube, stretch, Q=8.0, norm=None):
    if norm is not None:
        cube = cube / norm[:, None, None]
    gb, gg, gr = rgb_groups(cube.shape[0])
    b = _group_sum(cube, gb)
    g = _group_sum(cube, gg)
    r = _group_sum(cube, gr)
    I = (r + g + b) / 3
    I = torch.clamp(I, min=1e-12)
    fac = torch.asinh(Q * I / stretch) / (Q * I)
    rgb = torch.stack([r * fac, g * fac, b * fac], -1).clamp(min=0)
    mx = rgb.max(-1, keepdim=True).values.clamp(min=1.0)
    rgb = (rgb / mx).clamp(0, 1)
    return rgb.flip(0).cpu().numpy()  # north up


def freeze_instrument(m, path, dev, name):
    """Copy PSF / background / geometry / stripes from a saved model and freeze them (only the sky stays free)."""
    st = torch.load(path, map_location=dev)
    inst = {k: v for k, v in st.items() if k.split(".")[0] in ("psf", "bkg", "geom", "stripes")}
    res = m.load_state_dict(inst, strict=False)
    miss = [k for k in res.missing_keys if k.split(".")[0] in ("psf", "bkg", "geom", "stripes") and k != "psf.chan_idx"]
    if miss or res.unexpected_keys:
        raise RuntimeError(f"instrument mismatch: missing {miss}, unexpected {res.unexpected_keys}")
    for part in (m.psf, m.bkg, m.geom) + ((m.stripes,) if m.stripes is not None else ()):
        for p_ in part.parameters():
            p_.requires_grad_(False)
    nfree = sum(p_.numel() for p_ in m.parameters() if p_.requires_grad)
    tick(f"{name}: instrument frozen from {path} ({len(inst)} tensors); {nfree / 1e6:.3f} M free sky parameters")


def make_pupil_psf(a, dets, dev, driz, band_index, tick_=None):
    from pupil_psf import PupilPSF
    require_nircam_only("the pupil PSF")
    det_chan = [1 if "LONG" in d.upper() else 0 for d in dets]
    psf = PupilPSF(REG.bands, [REG.lam_um[b] for b in REG.bands], [REG.chan_id[b] for b in REG.bands], det_chan, a.scale, n=a.psf_n).to(dev)
    b0 = band_index[REG.bands[0]]
    sw_det = det_chan.index(0)
    best, scores, stars = psf.fit_orientation(driz[b0].to(dev), b0, sw_det, step=2.0)
    LOG["pupil_orientation"] = {"angle_deg": best[1], "flip": best[2], "stars_yx": stars, "band": REG.bands[b0]}
    tick(f"pupil PSF: orientation {best[1]:.0f} deg (flip {best[2]}) from {len(stars)} stars in {REG.bands[b0]}; "
         f"{sum(p_.numel() for p_ in psf.parameters() if p_.requires_grad)} free PSF parameters")
    return psf


class ZipField(nn.Module):
    """Continuous field a_k(x): multi-resolution dense feature grids read at any position (bilinear) + MLP -> K
    spectral coefficients. Coordinates in target pixels (0-based centres); every level spans the same field."""

    def __init__(self, size, k, levels=(16, 32, 64, 128, 256, 512, 1024), feat=4, hidden=96, scale=1.0):
        super().__init__()
        self.size = size
        self.grids = nn.ParameterList([nn.Parameter(1e-2 * torch.randn(1, feat, r, r)) for r in levels])
        self.mlp = nn.Sequential(nn.Linear(feat * len(levels), hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, k))
        self.scale = scale

    def forward(self, tx, ty):  # (N,), (N,) -> (N, K)
        g = torch.stack([(tx + 0.5) / self.size * 2 - 1, (ty + 0.5) / self.size * 2 - 1], -1)[None, :, None]
        f = torch.cat([F.grid_sample(G, g, mode="bilinear", padding_mode="border", align_corners=False)[0, :, :, 0] for G in self.grids], 0)
        return self.mlp(f.T) * self.scale


class BSplineField(nn.Module):
    """Linear multiresolution field a_k(x, y) = scale * sum_l sum_ij c_l[k, i, j] B3((x - x_i)/h_l) B3((y - y_j)/h_l),
    B3 the uniform cubic B-spline (support 4 knots, C2). No MLP, no nonlinearity; coefficients start at zero.
    Same interface as ZipField: forward(tx, ty) -> (N, K), attribute size; coordinates in target pixels (0-based
    centres), the field spans [-0.5, size-0.5].
    Level l has knot spacing h_l (target px). With u = (x + 0.5)/h_l the knots sit at u = i, i = -2 .. ceil(size/h_l)+2
    (a margin of 2 knots on every side of the field, n_l = ceil(size/h_l) + 5 coefficients per axis). Index i is the x
    (column) axis, j the y (row) axis: c_l has shape (K, n_l, n_l) = (K, i, j).
    Outside the field: u is clamped to [-1, ceil(size/h_l)+1) per axis, i.e. positions further out than the margin
    knots read the value at the clamp (constant extension, zero gradient w.r.t. position, the margin coefficients are
    used). Inside the field the levels form a partition of unity (sum of weights = 1). Evaluation gathers the 4x4
    support of each point, in chunks of `chunk` points (activation-checkpointed when gradients are needed)."""

    def __init__(self, size, k, spacings=(8, 4, 2, 1, 0.5), scale=1.0, chunk=262144):
        super().__init__()
        self.size, self.k, self.scale, self.chunk = size, k, scale, chunk
        self.spacings = tuple(float(h) for h in spacings)
        self.ncs = [int(math.ceil(size / h - 1e-9)) + 5 for h in self.spacings]
        self.coefs = nn.ParameterList([nn.Parameter(torch.zeros(k, n, n)) for n in self.ncs])

    @staticmethod
    def _taps(u, n):
        """u (N,) knot units -> flat index offsets/weights along one axis: idx (N, 4) in [0, n), w (N, 4)."""
        u = u.clamp(-1.0, float(n - 4))  # n - 4 = ceil(size/h) + 1, the last margin knot but one
        f = torch.floor(u).clamp(-1.0, float(n - 5))  # taps f-1..f+2 -> storage 0..n-1; at u = n-4: f = n-5, t = 1
        t = u - f
        t2, t3 = t * t, t * t * t
        w = torch.stack([(1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6], -1)
        idx = (f.long() + 2)[:, None] + torch.arange(-1, 3, device=u.device)[None]  # knot i -> storage i + 2
        return idx, w

    def _eval(self, tx, ty, *coefs):
        out = 0.0
        for c, h, n in zip(coefs, self.spacings, self.ncs):
            ix, wx = self._taps((tx + 0.5) / h, n)
            iy, wy = self._taps((ty + 0.5) / h, n)
            cf = c.reshape(self.k, -1)
            for a in range(4):
                for b in range(4):
                    out = out + cf[:, ix[:, a] * n + iy[:, b]] * (wx[:, a] * wy[:, b])[None]
        return out  # (K, N)

    def forward(self, tx, ty):  # (N,), (N,) -> (N, K)
        coefs = tuple(self.coefs)
        N = tx.shape[0]
        if N <= self.chunk:
            return self._eval(tx, ty, *coefs).T * self.scale
        need = torch.is_grad_enabled() and any(c.requires_grad for c in coefs)
        parts = []
        for i in range(0, N, self.chunk):
            a, b = tx[i:i + self.chunk], ty[i:i + self.chunk]
            if need:
                parts.append(torch.utils.checkpoint.checkpoint(self._eval, a, b, *coefs, use_reentrant=False))
            else:
                parts.append(self._eval(a, b, *coefs))
        return torch.cat(parts, 1).T * self.scale


class ZipSkyModel(nn.Module):
    """m_i(p) = g_i E_{q, delta}[ s(T_i(p) - delta, lambda_q) ] + B_i(p) + stripes: the continuous field is read at
    each detector pixel's own sky position, displaced by Monte Carlo samples delta of the physical PSF (which
    includes the detector pixel response) at quadrature wavelengths lambda_q of the band. Unbiased for the
    prediction; the squared loss of an S-sample average adds Var/S, a weak smoothing (reported)."""

    def __init__(self, field, basis, lam_nodes, w_nodes, psf, bkg, geom, band_index, stripes=None, sig=1.0, lam_smooth=0.0,
                 sig_ref=1.0, floor_sigma=3.0):
        super().__init__()
        self.sky, self.basis, self.psf, self.bkg, self.geom, self.stripes = field, basis, psf, bkg, geom, stripes
        self.register_buffer("lam_nodes", lam_nodes)
        self.register_buffer("w_nodes", w_nodes)
        self.band_index, self.sig = band_index, sig
        self.lam_smooth, self.sig_ref, self.floor_sigma = lam_smooth, sig_ref, floor_sigma
        self.tables = None

    @torch.no_grad()
    def build_samplers(self):
        """Inverse-CDF tables of the oversampled physical PSF (detector response included), per band, node, detector."""
        nb, q = self.lam_nodes.shape
        P = self.psf
        lam = self.lam_nodes.flatten()
        chan = P.chan_idx.repeat_interleave(q)
        I = P._detector(P._mono(lam, chan, torch.arange(P.ndet, device=lam.device)), chan)  # nbQ D no no
        I = I.clamp(min=0)
        no = I.shape[-1]
        if ZIP_PSF_CONVENTION == "explicit" and PSF_CONVENTION == "correlation":  # v24/v25: match convolve_basis
            I = I.flip(-2, -1)
        self.I = I  # (nbQ, D, no, no), kept for the deterministic renderer (v21)
        cdf = torch.cumsum(I.flatten(2), -1)
        self.ee = cdf[..., -1].clone()  # energy inside the window
        self.cdf = cdf / cdf[..., -1:]
        self.no, self.os = no, P.os
        self.phi = self.basis(lam).view(nb, q, -1)  # nb Q K

    render_mode, render_os, _R, _Kf = "mc", 2, None, None

    @torch.no_grad()
    def build_kernels(self, os_r, band_dets):
        """Deterministic renderer (v21). The same physical PSF as the sampler (detector response included, energy in
        the window kept), binned from the sampler's oversampling P.os to os_r points per target pixel, bins centred on
        offsets k/os_r. The outermost ring of the fine grid that does not fill a whole bin is dropped (reported).
        Kernel FFTs are cached on the padded render grid for the (band, detector) pairs that occur."""
        f = self.os // os_r
        assert f * os_r == self.os and f % 2 == 0 and self.no % 2 == 0,             "bins centred on k/os_r need an even number of fine pixels per bin (even PSF grid)"
        no = self.no
        d = (torch.arange(no, device=self.I.device, dtype=torch.float64) - (no - 1) / 2) / self.os  # fine offsets, target px
        k = torch.floor(d * os_r + 0.5).long()
        kmax = (no // f - 1) // 2
        M = torch.zeros(2 * kmax + 1, no, device=self.I.device, dtype=self.I.dtype)
        ok = k.abs() <= kmax
        M[k[ok] + kmax, torch.nonzero(ok)[:, 0]] = 1.0
        Kr = torch.einsum("ki,ndij,lj->ndkl", M, self.I, M)  # nbQ D nk nk
        self.render_os, self.kmax = os_r, kmax
        self.pad_t = int(math.ceil(kmax / os_r))  # render margin in target px (no circular wrap in the inner region)
        H = self.sky.size
        self.Ne = (H + 2 * self.pad_t) * os_r
        nk = 2 * kmax + 1
        Q = self.lam_nodes.shape[1]
        self._Kf, lost = {}, []
        for (b, det) in sorted(band_dets):
            for q in range(Q):
                K = torch.zeros(self.Ne, self.Ne, device=self.I.device)
                K[:nk, :nk] = Kr[b * Q + q, det]
                K = torch.roll(K, shifts=(-kmax, -kmax), dims=(0, 1))  # offset 0 at index 0
                self._Kf[(b, det, q)] = torch.fft.rfft2(K)
                lost.append(float(1 - Kr[b * Q + q, det].sum() / self.I[b * Q + q, det].sum().clamp(min=1e-30)))
        self.band_dets = sorted(band_dets)
        LOG["zip_render"] = {"os": os_r, "kernel_px": nk, "margin_target_px": self.pad_t, "grid": self.Ne,
                             "energy_dropped_edge_ring_max": max(lost),
                             "note": "field evaluated on the oversampled grid, FFT-convolved, then bilinear-sampled at detector positions"}
        tick(f"deterministic renderer: {nk}^2 kernels at {os_r}x, render grid {self.Ne}^2, "
             f"edge-ring energy dropped <= {max(lost):.2e}")

    def render(self):
        """Band images through the PSF, per (band, detector), on the os_r-oversampled target grid (inner region)."""
        os_r, H, pad = self.render_os, self.sky.size, self.pad_t
        u = (torch.arange(self.Ne, device=self.lam_nodes.device, dtype=torch.float32) + 0.5) / os_r - 0.5 - pad
        yy, xx = torch.meshgrid(u, u, indexing="ij")
        A = self.sky(xx.flatten(), yy.flatten()).T.reshape(-1, self.Ne, self.Ne)  # K Ne Ne
        Q = self.lam_nodes.shape[1]
        out = {}
        for b in sorted({b for b, _ in self.band_dets}):
            Fq = torch.fft.rfft2(torch.einsum("qk,khw->qhw", self.phi[b], A))
            for (bb, det) in self.band_dets:
                if bb != b:
                    continue
                acc = sum(self.w_nodes[b, q] * Fq[q] * self._Kf[(b, det, q)] for q in range(Q))
                R = torch.fft.irfft2(acc, s=(self.Ne, self.Ne))
                out[(b, det)] = R[pad * os_r:(pad + H) * os_r, pad * os_r:(pad + H) * os_r]
        return out

    def offsets(self, b, det, N, n, gen=None):
        """n samples for each of N pixels of one detector: node index (N,n) and offsets dx, dy (N,n) in target pixels."""
        Q = self.lam_nodes.shape[1]
        dev = self.lam_nodes.device
        qi = torch.randint(0, Q, (N, n), device=dev, generator=gen)
        u = torch.rand(N, n, device=dev, generator=gen)
        idx = torch.empty(N, n, dtype=torch.long, device=dev)
        for q in range(Q):
            sel = qi == q
            idx[sel] = torch.searchsorted(self.cdf[b * Q + q, det].contiguous(), u[sel].contiguous()).clamp(max=self.no * self.no - 1)
        iy, ix = idx // self.no, idx % self.no
        j = torch.rand(N, n, 2, device=dev, generator=gen) - 0.5
        c = (self.no - 1) / 2
        return qi, (ix.float() - c + j[..., 0]) / self.os, (iy.float() - c + j[..., 1]) / self.os

    def core(self, b, det, x, y, n=8, gen=None, chunk=200000):
        det = int(det)
        if self.render_mode == "det":
            if self._R is None:  # evaluation: render once without gradients, cached until the next training step
                with torch.no_grad():
                    self._R = self.render()
            os_r = self.render_os
            return sample(self._R[(b, det)], (x + 0.5) * os_r - 0.5, (y + 0.5) * os_r - 0.5)
        Q = self.lam_nodes.shape[1]
        out = []
        for s0 in range(0, x.numel(), chunk):
            xs, ys = x[s0:s0 + chunk], y[s0:s0 + chunk]
            qi, dx, dy = self.offsets(b, det, xs.numel(), n, gen)
            A = self.sky((xs[:, None] - dx).flatten(), (ys[:, None] - dy).flatten())  # (Nn, K)
            v = (A * self.phi[b][qi.flatten()]).sum(1).view(-1, n)
            out.append((v * self.ee[b * Q + qi, det]).mean(1))
        return torch.cat(out)

    def predict_zip(self, i, e, idx, n=8, gen=None):
        b = self.band_index[e["band"]]
        tx, ty, dx_, dy_ = e["tx"][idx], e["ty"][idx], e["dx"][idx], e["dy"][idx]
        det = e["det"]
        x, y = self.geom(i, det, tx, ty, dx_, dy_)
        c = self.core(b, det, x, y, n, gen)
        m = self.bkg(i, det, dx_, dy_) + self.bkg.gain(vec(i, dx_)) * c
        if self.stripes is not None and uses_stripes(e["band"]):
            m = m + self.stripes(vec(i, dx_), e["row"][idx], e["col"][idx])
        return m

    def grid_coeffs_os(self, os_):  # v28: K (H os) (H os) on the os-times oversampled grid
        H = self.sky.size
        u = (torch.arange(H * os_, device=self.lam_nodes.device, dtype=torch.float32) + 0.5) / os_ - 0.5
        yy, xx = torch.meshgrid(u, u, indexing="ij")
        return self.sky(xx.flatten(), yy.flatten()).T.reshape(-1, H * os_, H * os_)

    def grid_coeffs(self):  # K H W at the target pixel centres (outputs, priors)
        H = self.sky.size
        yy, xx = torch.meshgrid(torch.arange(H, device=self.lam_nodes.device, dtype=torch.float32),
                                torch.arange(H, device=self.lam_nodes.device, dtype=torch.float32), indexing="ij")
        return self.sky(xx.flatten(), yy.flatten()).T.reshape(-1, H, H)

    def prepare(self):  # band images at the grid nodes and their PSF convolution, for outputs only
        A = self.grid_coeffs()
        nb, q = self.lam_nodes.shape
        Sn = torch.einsum("bqk,khw->bqhw", self.phi, A)
        S = (Sn * self.w_nodes[:, :, None, None]).sum(1)
        return S, convolve_basis(S, self.psf.basis())

    def spectral_cube(self, lam_um):
        return torch.einsum("mk,khw->mhw", self.basis(lam_um), self.grid_coeffs())

    def penalty_at(self, n=65536):
        """Spectral smoothness (relative curvature in log lambda) and negativity, at random positions."""
        H = self.sky.size
        dev = self.lam_nodes.device
        x = torch.rand(n, device=dev) * H - 0.5
        y = torch.rand(n, device=dev) * H - 0.5
        A = self.sky(x, y).T  # K n
        d2 = A[2:] - 2 * A[1:-1] + A[:-2]
        scale = (A.abs().mean(0, keepdim=True) + self.floor_sigma * self.sig_ref).detach()
        p = self.lam_smooth * (d2 / scale).pow(2).mean()
        if LAM_NEG > 0:
            s = (self.phi.mean((0, 1)) @ A)  # broadband value
            p = p + LAM_NEG * F.relu(-s / self.sig_ref - 2.0).pow(2).mean()
        return p

    def penalty(self):
        return self.penalty_at()


def fit_zip(name, model, exps, train_idx, lr, budget_s, callback, cb_every_s, patience, frac=0.125, nsamp=8):
    params = [p_ for p_ in model.sky.parameters() if p_.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    ntot = sum(exps[i]["d"].numel() for i in train_idx)
    t0 = time.perf_counter(); t_cb = t0
    best, bad, it, hist = None, 0, 0, []
    gen = torch.Generator(device=exps[0]["d"].device).manual_seed(0)
    det_mode = model.render_mode == "det"
    while True:
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        if det_mode:  # render once with gradients; per-exposure losses back-propagate into detached leaves
            R_full = model.render()
            model._R = {k: v.detach().requires_grad_(True) for k, v in R_full.items()}
        for i in train_idx:
            e = exps[i]
            N = e["d"].numel()
            idx = torch.randperm(N, device=e["d"].device, generator=gen)[: max(1, int(frac * N))]
            m = model.predict_zip(i, e, idx, nsamp, gen)
            li = rho(e["ivar"][idx] * (e["d"][idx] - m) ** 2).sum() / (frac * ntot)
            li.backward()
            loss += float(li.detach())
        if det_mode:
            keys = [k for k in R_full if model._R[k].grad is not None]
            torch.autograd.backward([R_full[k] for k in keys], [model._R[k].grad for k in keys])
            model._R, R_full = None, None
        reg = model.penalty_at()
        reg.backward()
        opt.step()
        it += 1
        el = time.perf_counter() - t0
        frac_t = el / budget_s
        for g in opt.param_groups:  # warm-up 3 %, then cosine to 2 % over the time budget
            g["lr"] = lr * (frac_t / 0.03 if frac_t < 0.03 else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac_t - 0.03) / 0.97))))
        if it % 50 == 0:
            hist.append([it, round(el, 1), loss])
            tick(f"{name} step {it} ({el / 60:.1f} min) chi2/pix={loss:.4f}")
        stop = el > budget_s
        if callback is not None and (time.perf_counter() - t_cb > cb_every_s or stop):
            score = callback(model, it, el)
            t_cb = time.perf_counter()
            if score is not None:
                if best is None or score < best[0]:
                    best = (score, it, {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
                    bad = 0
                    LOG[f"{name}_best_score"] = score
                    tick(f"{name}: new best validation score {score:.4f} at step {it}")
                else:
                    bad += 1
                    stop = stop or (patience and bad >= patience)
        if stop:
            break
    LOG[f"{name}_history"] = hist
    model._R = None
    if best is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in best[2].items()})
        tick(f"{name}: restored best validation checkpoint (score {best[0]:.4f}, step {best[1]})")
    if SAVE_DIR is not None:
        torch.save(model.state_dict(), SAVE_DIR / f"{name}_state.pt")
    return model


def cg_warm_start(m, A0, sig, iters):
    """v27: conjugate gradients on G^T G c = G^T A0 for a linear field (G c = m.grid_coeffs()). Returns mse/sig^2."""
    params = [p_ for p_ in m.sky.parameters() if p_.requires_grad]

    def G(c):
        with torch.no_grad():
            for p_, v in zip(params, c):
                p_.copy_(v)
        return m.grid_coeffs_os(getattr(m, "warm_os", 1))

    def GT(r):  # vector-Jacobian product at any point (linear field)
        out = m.grid_coeffs_os(getattr(m, "warm_os", 1))
        g = torch.autograd.grad(out, params, grad_outputs=r)
        return [x.detach() for x in g]

    def dot(u, v):
        return sum((x * y).sum() for x, y in zip(u, v))

    # Jacobi-like preconditioner: the Gram diagonal of a level grows as h_l^2 (each coefficient covers h_l^2 nodes)
    hs = getattr(m.sky, "spacings", None) or [1.0] * len(params)
    Minv = [1.0 / max(h, 1.0) ** 2 for h in hs]
    x = [torch.zeros_like(p_) for p_ in params]
    b = GT(A0)
    r = [bi.clone() for bi in b]
    z = [mi * ri for mi, ri in zip(Minv, r)]
    p = [zi.clone() for zi in z]
    rs = dot(r, z)
    for k in range(iters):
        Gp = G(p)
        Ap = GT(Gp)
        alpha = rs / dot(p, Ap).clamp(min=1e-30)
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * api for ri, api in zip(r, Ap)]
        z = [mi * ri for mi, ri in zip(Minv, r)]
        rs_new = dot(r, z)
        if k % 100 == 0 or k == iters - 1:
            with torch.no_grad():
                mse = float(((G(x) - A0) / sig).pow(2).mean())
            tick(f"CG warm start iteration {k}: mse/sig2 {mse:.4f}")
        if rs_new.abs().sqrt() < 1e-9 * b[0].abs().max():
            break
        p = [zi + (rs_new / rs) * pi for zi, pi in zip(z, p)]
        rs = rs_new
    with torch.no_grad():
        mse = (G(x) - A0) / sig
    return mse.pow(2).mean()


def zip_capacity(a, init, sig_b, src_masks, lam_nodes, w_nodes, dev, out):
    """Q1 capacity control (v21): fit the zip-field directly to target spectral-coefficient grids, with no
    instrument, no PSF and no exposures. It measures whether the representation can hold each target at the grid
    nodes; it says nothing about deconvolution or about values between nodes. Objective = the v20 warm start's
    (coefficient mse / mean(sig_b)^2). Scored in band space against each target's band images: mse/sig_b^2 on all
    pixels and on the S/N > 5 pixels of the SWarp-like coadd (the mask of the held-out 'src' score)."""
    basis = RBFSpectralBasis(a.spec_k, learned=False).to(dev)
    targets = {"coadd_" + a.init: spectral_init(init, lam_nodes, w_nodes, basis).detach()}
    if a.capacity_target_state:
        st = torch.load(a.capacity_target_state, map_location=dev)
        cand = [(k, v) for k, v in st.items() if k.startswith("sky.") and v.dim() == 3 and v.shape[0] == a.spec_k
                and v.shape[-1] == a.size]
        assert len(cand) == 1, [k for k, _ in cand]
        targets["grid3d"] = cand[0][1].float().detach()
        LOG["capacity_target_grid3d"] = {"file": a.capacity_target_state, "key": cand[0][0]}
    phi = basis(lam_nodes.flatten()).view(*lam_nodes.shape, -1)  # nb Q K

    def bands(A):
        return torch.einsum("bqk,khw->bhw", phi * w_nodes[..., None], A)

    sig = sig_b[:, None, None]
    H = a.size
    ar = torch.arange(H, device=dev, dtype=torch.float32)
    yy, xx = torch.meshgrid(ar, ar, indexing="ij")
    res = {}
    for cfg in a.capacity_configs.split(";"):
        cname, feat, hidden, levels = cfg.split(":")
        for tname, A0 in targets.items():
            torch.manual_seed(0)
            field = ZipField(H, a.spec_k, levels=tuple(int(v) for v in levels.split(",")), feat=int(feat),
                             hidden=int(hidden), scale=float(sig_b.mean()) * 20).to(dev)
            npar = sum(p_.numel() for p_ in field.parameters())
            opt = torch.optim.Adam(field.parameters(), lr=a.capacity_lr)
            with torch.no_grad():
                S0 = bands(A0)
            t0, it, hist, budget = time.perf_counter(), 0, [], a.capacity_min * 60
            while True:
                el = time.perf_counter() - t0
                ft = el / budget
                for g in opt.param_groups:  # warm-up 3 %, cosine to 1 %
                    g["lr"] = a.capacity_lr * (ft / 0.03 if ft < 0.03 else
                                               0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (ft - 0.03) / 0.97))))
                opt.zero_grad(set_to_none=True)
                A = field(xx.flatten(), yy.flatten()).T.reshape(-1, H, H)
                l = ((A - A0) / sig_b.mean()).pow(2).mean()
                l.backward()
                opt.step()
                it += 1
                done = el > budget
                if it % 500 == 0 or done or it == 800:
                    with torch.no_grad():
                        r2 = ((bands(A) - S0) / sig).pow(2)
                        row = [it, round(el, 1), float(l), float(r2.mean()), float(r2[src_masks].mean())]
                    hist.append(row)
                    tick(f"CAPACITY {cname}/{tname} step {it} ({el / 60:.1f} min): coeff mse/sig2 {row[2]:.4f} | "
                         f"band mse/sig2 all {row[3]:.4f} bright {row[4]:.4f}")
                if done:
                    break
            with torch.no_grad():
                A = field(xx.flatten(), yy.flatten()).T.reshape(-1, H, H)
                rb = (bands(A) - S0) / sig
                per_band = {REG.bands[j]: {"all": float(rb[j].pow(2).mean()), "bright": float(rb[j][src_masks[j]].pow(2).mean()),
                                           "bright_max_abs": float(rb[j][src_masks[j]].abs().max())} for j in range(len(REG.bands))}
                np.save(out / f"capacity_{cname}_{tname}_resid.npy", rb.cpu().numpy())
            res[f"{cname}/{tname}"] = {"params": npar, "steps": it, "minutes": a.capacity_min, "config": cfg,
                                       "coeff_mse_sig2": hist[-1][2], "band_mse_sig2_all": hist[-1][3],
                                       "band_mse_sig2_bright": hist[-1][4], "per_band": per_band, "history": hist}
            LOG["capacity"] = res
            json.dump(LOG, open(out / "metrics_partial.json", "w"), indent=1, default=float)
            del field, opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return res


def sky_fisher(m, exps, train_idx, nz=24):
    """Diagonal Gauss-Newton (Fisher) of the Gaussian likelihood with respect to the pixel-grid sky, by Hutchinson:
    diag(J^T W J) = E_z[(J^T W^(1/2) z)^2], z Rademacher on the detector pixels (exact in expectation: the model is
    linear in the sky). One backward pass over the training exposures per probe."""
    for p_ in m.parameters():
        p_.requires_grad_(False)
    m.sky.s.requires_grad_(True)
    Fd = torch.zeros_like(m.sky.s)
    for k in range(nz):
        S = m.sky()
        CB = convolve_basis(S, partial_kernels(m.psf.basis(), TARGET_SIG_PX))
        CBd = CB.detach().requires_grad_(True)
        for i in train_idx:
            e = exps[i]
            pred = m.predict(CBd, i, e)
            z = torch.randint(0, 2, pred.shape, device=pred.device, dtype=pred.dtype) * 2 - 1
            (z * e["ivar"].sqrt() * pred).sum().backward()
        g, = torch.autograd.grad(CB, m.sky.s, grad_outputs=CBd.grad)
        Fd += g.detach() ** 2
        tick(f"sky Fisher probe {k + 1}/{nz}")
    Fd /= nz
    m.sky.s.requires_grad_(False)
    return Fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/arc/projects/ots/torchsky-data/jwst_smacs0723/aligned")
    ap.add_argument("--hst-dir", default=None, help="v29: directory with HST *_flt.fits / *_flc.fits; enables the HST exposures")
    ap.add_argument("--hst-manifest", default=None, help="CSV (like hst_imaging_unique.csv) selecting the HST files in --hst-dir; without it every flt/flc in the directory is filtered on its headers")
    ap.add_argument("--hst-programs", default=None, help="comma list of HST proposal ids")
    ap.add_argument("--hst-instruments", default=None, help='comma list, e.g. "WFC3/IR,ACS/WFC" (also WFC3/UVIS)')
    ap.add_argument("--hst-filters", default=None, help="comma list of HST filter names, e.g. F160W,F814W")
    ap.add_argument("--hst-mjd-min", type=float, default=None, help="earliest EXPSTART (MJD) of the HST exposures")
    ap.add_argument("--hst-mjd-max", type=float, default=None, help="latest EXPSTART (MJD) of the HST exposures")
    ap.add_argument("--hst-dq-policy", default="default", help='HST DQ: "default" (drizzlepac good bits per instrument), "strict" (DQ == 0) or an integer good-bit sum')
    ap.add_argument("--lam-lo", type=float, default=0.70, help="lower end (um) of the spectral domain of the sky field (HST optical needs ~0.35)")
    ap.add_argument("--lam-hi", type=float, default=5.30, help="upper end (um) of the spectral domain of the sky field")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ra", type=float, default=110.8268)
    ap.add_argument("--dec", type=float, default=-73.4544)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--scale", type=float, default=0.04)
    ap.add_argument("--steps-explicit", type=int, default=1500)
    ap.add_argument("--steps-implicit", type=int, default=4000)
    ap.add_argument("--lam-tv", type=float, default=0.02)
    ap.add_argument("--nk", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--stripes", type=int, default=1, help="model NIRCam 1/f row/column stripes per exposure")
    ap.add_argument("--load-dir", default=None, help="evaluate saved *_state.pt models instead of fitting")
    ap.add_argument("--sky-model", choices=("grid", "siren", "gs"), default="grid", help="implicit sky: feature-grid MLP, SIREN or Gaussian splats")
    ap.add_argument("--psf-model", choices=("band", "chrom", "pupil"), default="band", help="implicit PSF: per-band MLP or one chromatic field")
    ap.add_argument("--explicit-psf", choices=("band", "chrom", "pupil"), default="band")
    ap.add_argument("--explicit-configs", default="fixed,explicit", help="comma list of explicit geometry modes to fit")
    ap.add_argument("--psf-n", type=int, default=31, help="PSF kernel size in target pixels")
    ap.add_argument("--gs-n", type=int, default=80000)
    ap.add_argument("--tag", default="", help="suffix for model names")
    ap.add_argument("--init", choices=("swarp", "drizzle"), default="swarp", help="coadd used to initialise / warm-start the sky")
    ap.add_argument("--holdout-from", default=None, help="metrics.json / holdout_selection.json ('holdout' names), JSON list or text file (one name per line; an HST rootname holds out all its chips) to reuse")
    ap.add_argument("--eval-every-min", type=float, default=0.0, help="cross-validation during long fits (0 = off)")
    ap.add_argument("--n-val", type=int, default=0, help="validation exposures per band (early stopping), kept out of training")
    ap.add_argument("--n-test", type=int, default=None, help="test exposures per band: 1 (default) = v21 (one detector image); N>1 = N dithers (all their detector images; HST: one exposure = all its chips), spread over visits. With --holdout-from: given explicitly, bands without a held-out exposure in the file get this rule")
    ap.add_argument("--boot-b", type=int, default=1000, help="bootstrap replicates for the held-out scores")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many validation checks without improvement")
    ap.add_argument("--calibrate-noise", action="store_true", help="rescale ERR per exposure from first-difference noise")
    ap.add_argument("--loss", choices=("gauss", "huber", "pseudo_huber"), default="gauss")
    ap.add_argument("--spectral", choices=("none", "rbf"), default="none", help="continuous s(x,y,lambda) sky with SVO bandpass integration")
    ap.add_argument("--spec-k", type=int, default=8, help="spectral basis functions (log-lambda RBF)")
    ap.add_argument("--spec-q", type=int, default=6, help="quadrature nodes per bandpass")
    ap.add_argument("--spec-smooth", type=float, default=1e-3, help="second-difference penalty on spectral coefficients")
    ap.add_argument("--svo-cache", default=None, help="SVO filter cache directory")
    ap.add_argument("--spec-relative", action="store_true", help="spectral smoothness relative to each pixel's level (brightness-independent)")
    ap.add_argument("--spec-learned-basis", type=int, default=1, help="neural sky: learned correction to the spectral basis (1) or fixed RBF (0)")
    ap.add_argument("--no-holdout", action="store_true", help="train on every exposure (refit)")
    ap.add_argument("--score-all", action="store_true", help="score every method in-sample on all exposures")
    ap.add_argument("--updec-dir", default=None, help="UPdec-Webb module-b products to score alongside")
    ap.add_argument("--huber-delta", type=float, default=3.0)
    ap.add_argument("--target-fwhm", type=float, default=0.0, help="target resolution r (arcsec FWHM): the sky grid holds r*s, the instrument sees it through K/r")
    ap.add_argument("--lam-tnv", type=float, default=0.0, help="total nuclear variation across bands (shared edges)")
    ap.add_argument("--lam-band", type=float, default=0.0, help="penalty on sky structure narrower than the target resolution")
    ap.add_argument("--sky-coeffs", choices=("grid", "hybrid", "convdecoder"), default="grid", help="explicit spectral sky coefficients")
    ap.add_argument("--n-splats", type=int, default=3000, help="compact Gaussian sources for --sky-coeffs hybrid")
    ap.add_argument("--splat-snr", type=float, default=15.0, help="detection threshold (sigma) for splat seeds")
    ap.add_argument("--dip-warm", type=int, default=1500, help="ConvDecoder warm-up steps on the spectral initialisation")
    ap.add_argument("--dip-lr", type=float, default=1e-3)
    ap.add_argument("--resume-lr-scale", type=float, default=1.0, help="scale all learning rates when resuming a checkpoint")
    ap.add_argument("--lam-neg", type=float, default=0.0, help="penalty on sky below -2 sigma")
    ap.add_argument("--sky-blur", type=float, default=0.0, help="Gaussian band limit (target px) on the explicit grid sky")
    ap.add_argument("--psf-centre", type=float, default=0.0, help="penalty on PSF basis first moments (chromatic field)")
    ap.add_argument("--bkg-model", choices=("implicit", "explicit"), default="implicit", help="background for the implicit model")
    ap.add_argument("--geom-model", choices=("implicit", "explicit"), default="implicit", help="distortion for the implicit model")
    ap.add_argument("--lam-tv-implicit", type=float, default=0.0)
    ap.add_argument("--siren-hidden", type=int, default=384)
    ap.add_argument("--siren-layers", type=int, default=5)
    ap.add_argument("--siren-w0", type=float, default=30.0)
    ap.add_argument("--skip-explicit", action="store_true")
    ap.add_argument("--freeze-instrument-from", default=None, help="load PSF, background, geometry and stripes from this run's explicit_chrom_state.pt and freeze them: only the sky is fitted")
    ap.add_argument("--sky-lr", type=float, default=0.0, help="override the sky learning rate (0 = default)")
    ap.add_argument("--psf-lr", type=float, default=0.0, help="override the PSF learning rate (0 = default)")
    ap.add_argument("--zip", action="store_true", help="continuous field read at detector-pixel positions with Monte Carlo samples of the physical PSF (needs --instrument-file)")
    ap.add_argument("--instrument-file", default=None, help="state file to take (and freeze) PSF, background, geometry and stripes from")
    ap.add_argument("--zip-samples", type=int, default=8)
    ap.add_argument("--zip-frac", type=float, default=0.125)
    ap.add_argument("--zip-render", choices=("mc", "det"), default="mc", help="mc: Monte Carlo PSF samples per pixel (v20); det: field on an oversampled grid, FFT-convolved with the same PSF, bilinear at detector positions")
    ap.add_argument("--psf-convention", choices=("correlation", "convolution"), default="correlation", help="v25: how every model applies its PSF; convolution matches the pupil orientation fit")
    ap.add_argument("--zip-psf-convention", choices=("explicit", "v20"), default="explicit", help="v24: zip PSF orientation; v20 = the old 180-degree rotated one")
    ap.add_argument("--zip-render-os", type=int, default=2, help="render points per target pixel for --zip-render det")
    ap.add_argument("--zip-init-state", default=None, help="load the field weights (sky.*) from this zip state file instead of the warm start")
    ap.add_argument("--zip-eval-only", action="store_true", help="score the zip model without fitting")
    ap.add_argument("--zip-warm-steps", type=int, default=800)
    ap.add_argument("--zip-warm-lr", type=float, default=5e-3)
    ap.add_argument("--zip-warm-os", type=int, default=1, help="v28: warm start on an N-times oversampled grid (bilinear target)")
    ap.add_argument("--zip-warm-solver", choices=("adam", "cg"), default="adam", help="v27: cg = conjugate gradients (linear fields only)")
    ap.add_argument("--zip-levels", default="16,32,64,128,256,512,1024")
    ap.add_argument("--zip-feat", type=int, default=4)
    ap.add_argument("--zip-hidden", type=int, default=96)
    ap.add_argument("--zip-field", choices=("mlp", "bspline"), default="mlp", help="mlp: multiresolution feature grids + MLP (ZipField); bspline: linear multiresolution cubic B-splines (BSplineField)")
    ap.add_argument("--bspline-spacings", default="8,4,2,1,0.5", help="knot spacings in target pixels, coarse to fine, for --zip-field bspline")
    ap.add_argument("--zip-capacity", action="store_true", help="Q1: fit the field directly to target coefficient grids (no instrument), report, exit")
    ap.add_argument("--capacity-target-state", default=None, help="explicit 3D state file whose grid sky is a second capacity target")
    ap.add_argument("--capacity-configs", default="base:4:96:16,32,64,128,256,512,1024;big:8:256:16,32,64,128,256,512,1024,2048",
                    help="name:feat:hidden:levels, separated by ;")
    ap.add_argument("--capacity-min", type=float, default=12.0, help="minutes per (config, target)")
    ap.add_argument("--capacity-lr", type=float, default=5e-3)
    ap.add_argument("--optimizer", choices=("adam", "lbfgs"), default="adam", help="lbfgs: full-batch quasi-Newton with strong-Wolfe line search")
    ap.add_argument("--uq", action="store_true", help="sky variance (Laplace, Hutchinson Fisher) and held-out predictive variances for calibration")
    ap.add_argument("--uq-probes", type=int, default=24)
    ap.add_argument("--rigid", action="store_true", help="explicit model with a rigid instrument: only the sky is learned (Gaussian PSF of the nominal FWHM, pipeline WCS, constant background, unit gain)")
    ap.add_argument("--skip-implicit", action="store_true")
    ap.add_argument("--budget-explicit", type=float, default=12.0, help="minutes per explicit fit")
    ap.add_argument("--budget-implicit", type=float, default=35.0, help="minutes for the implicit fit")
    a = ap.parse_args()
    global args_global, SAVE_DIR, CALIBRATE_NOISE, HUBER_DELTA, LAM_NEG, LAM_LO, LAM_HI
    LAM_LO, LAM_HI = a.lam_lo, a.lam_hi
    global ZIP_PSF_CONVENTION, PSF_CONVENTION
    ZIP_PSF_CONVENTION = a.zip_psf_convention
    PSF_CONVENTION = a.psf_convention
    LOG["psf_convention"] = PSF_CONVENTION
    args_global = a
    CALIBRATE_NOISE = a.calibrate_noise
    global LOSS_KIND
    LOSS_KIND = a.loss
    HUBER_DELTA = a.huber_delta if a.loss != "gauss" else None
    LAM_NEG = a.lam_neg
    global LAM_TNV, LAM_BAND, TARGET_SIG_PX, DIP_LR
    LAM_TNV, LAM_BAND, DIP_LR = a.lam_tnv, a.lam_band, a.dip_lr
    TARGET_SIG_PX = a.target_fwhm / 2.3548 / a.scale if a.target_fwhm > 0 else 0.0
    LOG["args"] = vars(a)
    out = Path(a.out)
    SAVE_DIR = out
    (out / "cubes").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG["device"] = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    thead = target_header(a.ra, a.dec, a.size, a.scale)
    LOG["target"] = thead

    pad_ = int(round(24 * 0.04 / a.scale))
    exps = load_all_exposures(a, thead, dev, pad_)
    counts = {b: sum(1 for e in exps if e["band"] == b and e["d"].numel() > 0.5 * (a.size * a.scale / 0.031) ** 2 * (0.031 / 0.063) ** 2) for b in NIRCAM_BANDS}
    tick(f"loaded {len(exps)} exposures at ({a.ra},{a.dec}); well-covering per band {counts}")
    if min(counts.values()) < 4 and not a.no_fallback:
        a.ra, a.dec = 110.7724, -73.4528  # centre of the harness' common-support crop
        thead = target_header(a.ra, a.dec, a.size, a.scale)
        LOG["target"] = thead
        LOG["centre_fallback"] = True
        exps = load_all_exposures(a, thead, dev, pad_)
    tick(f"loaded {len(exps)} exposures, {sum(e['d'].numel() for e in exps)/1e6:.1f} M pixels")
    run(exps, thead, dev, a, out)


def load_state_checked(m, path, dev):
    """Load a saved model; only buffers added after the run was made (recomputed in __init__) may be missing."""
    res = m.load_state_dict(torch.load(path, map_location=dev), strict=False)
    allowed = {"psf.chan_idx"}
    bad = [k for k in res.missing_keys if k not in allowed and not (RIGID and k.startswith("stripes."))]
    if bad or res.unexpected_keys:
        raise RuntimeError(f"state mismatch for {path}: missing {bad}, unexpected {res.unexpected_keys}")
    if res.missing_keys:
        tick(f"loaded {path.name}; recomputed buffers {res.missing_keys}")


def splat_seeds(init, sig_b, A0, nmax, snr, sig_px):
    """Local maxima of the noise-weighted short-wavelength detection image above snr; each seed takes half of the
    coefficient flux of a Gaussian of sigma sig_px at its peak."""
    with torch.no_grad():
        sw = detection_bands()  # NIRCam SW bands by name
        D = (init[sw] / sig_b[sw][:, None, None]).sum(0) / math.sqrt(len(sw))
        D = D - D.median()
        mx = F.max_pool2d(D[None, None], 5, stride=1, padding=2)[0, 0]
        pk = (D == mx) & (D > snr)
        pk[:8], pk[-8:], pk[:, :8], pk[:, -8:] = False, False, False, False
        y, x = torch.nonzero(pk, as_tuple=True)
        order = torch.argsort(D[y, x], descending=True)[:nmax]
        y, x = y[order], x[order]
        amp = 0.5 * A0[:, y, x].T * (2 * math.pi * sig_px ** 2)
        return torch.stack([x, y], 1).float(), amp


def run(exps, thead, dev, a, out, classical_fn=None):
    global LAM_LO, LAM_HI
    LAM_LO, LAM_HI = getattr(a, "lam_lo", LAM_LO), getattr(a, "lam_hi", LAM_HI)
    set_registry(registry_from_exposures(exps, psf_n=getattr(a, "psf_n", 31)))
    LOG["band_registry"] = REG.summary()
    LOG["lam_domain_um"] = [LAM_LO, LAM_HI]
    tick("band registry: " + ", ".join(f"{b} [{REG.instrument[b]} {REG.channel[b]} {REG.lam_um[b]:.3f}um D={REG.diameter_m[b]}m fwhm={REG.fwhm_arcsec[b]}\"]" for b in REG.bands))
    if REG.multi and (getattr(a, "explicit_psf", "") == "pupil" or getattr(a, "psf_model", "") == "pupil" or getattr(a, "zip", False)):
        require_nircam_only("--explicit-psf/--psf-model pupil and --zip")
    if getattr(a, "spectral", "none") != "none":
        lo_b = min(REG.lam_um.values())
        if lo_b < LAM_LO or max(REG.lam_um.values()) > LAM_HI:
            raise SystemExit(f"band pivots span {lo_b:.3f}-{max(REG.lam_um.values()):.3f} um, outside the spectral domain "
                             f"[{LAM_LO}, {LAM_HI}] um: set --lam-lo/--lam-hi")
    dets = sorted({e["detname"] for e in exps})
    for e in exps:
        e["det"] = dets.index(e["detname"])
    ndet = len(dets)
    LOG["detectors"] = dets
    counts = {b: sum(1 for e in exps if e["band"] == b) for b in REG.bands}
    LOG["exposures_per_band"] = counts
    LOG["exposures_per_instrument"] = {ins: sum(1 for e in exps if REG.instrument[e["band"]] == ins) for ins in REG.instruments}
    print("per band:", counts, "detectors:", dets, flush=True)
    n_test_arg = getattr(a, "n_test", None)
    sel = select_exposures(exps, n_test=1 if n_test_arg is None else n_test_arg, n_val=getattr(a, "n_val", 0),
                           holdout_names=(read_holdout_names(a.holdout_from) if getattr(a, "holdout_from", None) else None),
                           no_holdout=getattr(a, "no_holdout", False), fill_missing_bands=n_test_arg is not None)
    for w_ in sel["warnings"]:
        print("selection warning:", w_, flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    json.dump({"holdout": sel["holdout"], "validation": sel["validation"], "warnings": sel["warnings"]},
              open(Path(out) / "holdout_selection.json", "w"), indent=1)
    train_idx = [i for i, e in enumerate(exps) if not e["holdout"] and not e["val"]]
    hold_idx = [i for i, e in enumerate(exps) if e["holdout"]]
    val_idx = [i for i, e in enumerate(exps) if e["val"]]
    LOG["holdout"] = [exps[i]["name"] for i in hold_idx]
    LOG["validation"] = [exps[i]["name"] for i in val_idx]
    LOG["noise_scale"] = {e["name"]: e.get("noise_scale", 1.0) for e in exps}
    band_index = dict(REG.index)

    # ---------------- classical (torchsky)
    cfn = classical_fn or classical
    t = time.perf_counter()
    swarp = cfn(exps, thead, "lanczos3", 1.0, dev)
    LOG["timings"]["swarp_lanczos3"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    driz = cfn(exps, thead, "drizzle_square", 0.8, dev)
    LOG["timings"]["drizzle_square"] = round(time.perf_counter() - t, 1)
    tick("classical coadds done")
    cover = dict(getattr(cfn, "last_cover", {}))  # HST splat coadds: bool coverage per band (None / absent = all pixels, as v28)
    if any(v is not None for v in cover.values()):
        hb = [b for b in REG.bands if REG.is_hst[b]]
        LOG["hst_coadd_approximation"] = ("HST bands " + ",".join(hb) + ": SWarp-like and Drizzle-like are the SAME inverse-variance weighted "
                                          "tent-splat of the loaded pixels (tx, ty from the header WCS incl. lookup tables); approximation, "
                                          "no flux conservation / PSF matching / outlier rejection; statistics on covered pixels only")
        tick("APPROXIMATION: " + LOG["hst_coadd_approximation"])
    stats = [band_median_sigma(c, cover.get(b)) for b, c in zip(REG.bands, swarp)]
    sig_b = torch.stack([s_ for _, s_ in stats]).clamp(min=1e-6)
    results = {"swarp": swarp, "drizzle": driz}
    init = driz if getattr(a, "init", "swarp") == "drizzle" else swarp
    src_masks = []
    for (m_, _), b, c, sb in zip(stats, REG.bands, swarp, sig_b):
        sm_ = (c - m_) > 5 * sb
        if cover.get(b) is not None:
            sm_ = sm_ & cover[b]
        src_masks.append(sm_)
    src_masks = torch.stack(src_masks)
    global APERTURES
    with torch.no_grad():  # catalogue: isolated peaks of the SWarp-like short-wavelength detection image
        Rpx = 0.2 / a.scale
        sw = detection_bands()  # v29: the NIRCam SW bands by name (F090W, F150W, F200W)
        Dd = ((swarp[sw] - swarp[sw].flatten(1).median(1).values[:, None, None]) / sig_b[sw][:, None, None]).sum(0) / math.sqrt(len(sw))
        k = 2 * int(round(Rpx)) + 1
        mx = F.max_pool2d(Dd[None, None], k, stride=1, padding=k // 2)[0, 0]
        pk = (Dd == mx) & (Dd > 5)
        Hh, Ww = Dd.shape
        mrg = int(math.ceil(2 * Rpx))
        pk[:mrg], pk[-mrg:], pk[:, :mrg], pk[:, -mrg:] = False, False, False, False
        yy_, xx_ = torch.nonzero(pk, as_tuple=True)
        cxy = torch.stack([xx_, yy_], 1).float()
        if cxy.shape[0] > 1:  # drop sources whose apertures would overlap (keep isolated ones)
            d2 = torch.cdist(cxy, cxy)
            d2.fill_diagonal_(1e9)
            cxy = cxy[d2.min(1).values > 2.0 * Rpx]
        lab = torch.full((Hh, Ww), -1, dtype=torch.long, device=Dd.device)
        r = torch.arange(-int(math.ceil(Rpx)), int(math.ceil(Rpx)) + 1, device=Dd.device)
        oy, ox = torch.meshgrid(r, r, indexing="ij")
        disk = (ox ** 2 + oy ** 2) <= Rpx ** 2
        oy, ox = oy[disk], ox[disk]
        for j, (x0, y0) in enumerate(cxy.long().tolist()):
            lab[y0 + oy, x0 + ox] = j
        APERTURES = (lab, cxy, Rpx)
        LOG["aperture_catalogue"] = {"n": int(cxy.shape[0]), "radius_arcsec": 0.2, "detection": f"SWarp-like {'+'.join(REG.bands[k] for k in sw)} peak > 5 sigma, no neighbour within 0.4 arcsec; scored if S/N > 20 in the exposure"}
        tick(f"aperture catalogue: {cxy.shape[0]} isolated sources, r = 0.2 arcsec")

    def make_cb(name):
        ev_idx = val_idx or hold_idx  # validation exposures if any, else (v6 behaviour) the test exposures

        def cb(model, it, el):
            try:
                c2, c2s, _ = heldout_chi2(model, exps, ev_idx, band_index, ndet, steps=150, src_masks=src_masks)
            except Exception as exc:  # noqa: BLE001
                tick(f"{name} periodic CV failed: {exc!r}")
                return None
            L = heldout_chi2.last
            LOG.setdefault(f"{name}_cv_history", []).append([it, round(el / 3600, 3), c2, c2s, L["robust"], L["robust_src"]])
            tick(f"{name} {'VAL' if val_idx else 'CV'} at step {it} ({el / 3600:.2f} h): chi2 all {c2:.4f} src {c2s:.4f} | "
                 f"robust all {L['robust']:.4f} src {L['robust_src']:.4f}")
            if SAVE_DIR is not None:
                json.dump(LOG, open(SAVE_DIR / "metrics_partial.json", "w"), indent=1, default=float)
            return L["robust"] + L["robust_src"] if val_idx else None
        return cb if getattr(a, "eval_every_min", 0) > 0 else None

    sig0 = torch.tensor([REG.fwhm_arcsec[b] / a.scale / 2.3548 for b in REG.bands], device=dev)
    steps_e = 30 if a.quick else a.steps_explicit
    steps_i = 40 if a.quick else a.steps_implicit
    lrs_e = {"sky": float(sig_b.mean()) * 0.3, "psf": 2e-3, "bkg": 1e-3, "geom": 2e-3}
    spec = getattr(a, "spectral", "none") != "none"
    if spec:
        lam_nodes, w_nodes, curves = load_bandpasses(REG.bands, a.spec_q, a.svo_cache)
        if float(lam_nodes.min()) < LAM_LO or float(lam_nodes.max()) > LAM_HI:
            tick(f"WARNING: bandpass nodes span {float(lam_nodes.min()):.3f}-{float(lam_nodes.max()):.3f} um, beyond the sky domain [{LAM_LO}, {LAM_HI}]")
        lam_nodes, w_nodes = lam_nodes.to(dev), w_nodes.to(dev)
        LOG["bandpass_nodes_um"] = {b: [round(float(x), 4) for x in lam_nodes[k]] for k, b in enumerate(REG.bands)}
        LOG["bandpass_curves"] = curves
        tick(f"SVO bandpasses: {LOG['bandpass_nodes_um']}")
    models = {}
    esfx = ("_chrom" if a.explicit_psf == "chrom" else "") + ("_3d" if spec else "") + a.tag
    gnames = {"fixed": "explicit_fixedgeom", "explicit": "explicit"}
    configs = [] if a.skip_explicit else [(gnames[g] + esfx, g) for g in a.explicit_configs.split(",") if g]
    for name, gmode in configs:
        t = time.perf_counter()
        if a.explicit_psf == "pupil":
            epsf = make_pupil_psf(a, dets, dev, driz, band_index)
        elif a.explicit_psf == "chrom":
            epsf = ChromaticPSF(REG.bands, ndet, a.scale, a.nk, n=a.psf_n).to(dev)
            if not getattr(a, "load_dir", None):
                tick(f"explicit chromatic PSF warm start l={warm_psf(epsf, gauss_kernel(sig0, a.psf_n, dev), 30 if a.quick else 2000):.2e}")
        else:
            epsf = ExplicitVarPSF(sig0, ndet, a.nk, n=a.psf_n).to(dev)
        if a.psf_centre > 0 and hasattr(epsf, "loglam"):
            epsf.centre_weight = a.psf_centre
        if spec:
            basis = RBFSpectralBasis(a.spec_k, learned=False).to(dev)
            A0 = spectral_init(init, lam_nodes, w_nodes, basis)
            if TARGET_SIG_PX > 0:  # start from the coadd, already broader than r: nothing to remove
                pass
            kind = getattr(a, "sky_coeffs", "grid")
            if kind == "hybrid":
                xy, amp = splat_seeds(init, sig_b, A0, a.n_splats, a.splat_snr, max(TARGET_SIG_PX, 0.7))
                splats = SplatCoeffs(xy, amp, A0.shape[-2], A0.shape[-1], TARGET_SIG_PX).to(dev)
                with torch.no_grad():
                    A_grid = A0 - splats()
                coeffs = HybridCoeffs(GridSky(A_grid, blur_px=a.sky_blur).to(dev), splats)
                LOG[f"{name}_n_splats"] = int(xy.shape[0])
                tick(f"{name}: {xy.shape[0]} compact-source splats seeded above {a.splat_snr} sigma")
            elif kind == "convdecoder":
                s0 = float(sig_b.mean())
                coeffs = ConvDecoderCoeffs(a.spec_k, A0.shape[-1], s0).to(dev)
                target_raw = torch.asinh(A0 / s0)
                opt_w = torch.optim.Adam(coeffs.parameters(), lr=3e-3)
                lw = torch.full((), float("nan"))
                for it_w in range(30 if a.quick else a.dip_warm):
                    opt_w.zero_grad()
                    lw = (coeffs.raw() - target_raw).pow(2).mean()
                    lw.backward()
                    opt_w.step()
                tick(f"{name}: ConvDecoder warm start on the spectral initialisation, mse(asinh) {float(lw):.4f}, "
                     f"{sum(p.numel() for p in coeffs.parameters()) / 1e6:.2f} M parameters")
            else:
                coeffs = GridSky(A0, blur_px=a.sky_blur).to(dev)
            esky = SpectralSky(coeffs, basis, lam_nodes, w_nodes, lam_smooth=a.spec_smooth, sig_ref=float(sig_b.mean()),
                               relative=a.spec_relative).to(dev)
        else:
            esky = GridSky(init, blur_px=a.sky_blur).to(dev)
        m = SkyModel(esky, epsf, ExplicitBackground(exps, ndet),
                     Geometry(exps, ndet, gmode), band_index,
                     stripes=Stripes(len(exps), dev) if a.stripes else None, sig=float(sig_b.mean()))
        if getattr(a, "rigid", False):
            global RIGID
            RIGID = True
            for p_ in list(m.psf.parameters()) + [m.bkg.pe, m.bkg.pd, m.bkg.logg]:
                p_.requires_grad_(False)
            tick(f"{name}: rigid instrument, only the sky is learned ({sum(p_.numel() for p_ in m.parameters() if p_.requires_grad) / 1e6:.2f} M free parameters)")
        if getattr(a, "freeze_instrument_from", None):
            freeze_instrument(m, Path(a.freeze_instrument_from) / "explicit_chrom_state.pt", dev, name)
        if getattr(a, "load_dir", None):
            load_state_checked(m, Path(a.load_dir) / f"{name}_state.pt", dev)
        else:
            fit(name, m, exps, steps_e, dict(dict(lrs_e, psf=5e-4) if a.explicit_psf == "chrom" else lrs_e, **({"sky": a.sky_lr} if a.sky_lr > 0 else {}), **({"psf": a.psf_lr} if a.psf_lr > 0 else {})), a.lam_tv, sig_b, train_idx,
                budget_s=a.budget_explicit * 60, callback=make_cb(name), cb_every_s=a.eval_every_min * 60, patience=a.patience)
        LOG["timings"][name] = round(time.perf_counter() - t, 1)
        models[name] = m

    if getattr(a, "zip_capacity", False):
        zip_capacity(a, init, sig_b, src_masks, lam_nodes, w_nodes, dev, out)
        json.dump(LOG, open(out / "metrics.json", "w"), indent=1, default=float)
        tick("done (capacity control only)")
        return

    if getattr(a, "zip", False):
        from pupil_psf import PupilPSF
        require_nircam_only("--zip (pupil PSF)")
        assert spec, "--zip needs --spectral rbf"
        zname = "zip_pupil_3d" + a.tag
        t = time.perf_counter()
        det_chan = [1 if "LONG" in d.upper() else 0 for d in dets]
        zpsf = PupilPSF(REG.bands, [REG.lam_um[b] for b in REG.bands], [REG.chan_id[b] for b in REG.bands], det_chan, a.scale, n=a.psf_n).to(dev)
        zbasis = RBFSpectralBasis(a.spec_k, learned=False).to(dev)
        if a.zip_field == "bspline":
            zfield = BSplineField(a.size, a.spec_k, spacings=tuple(float(v) for v in a.bspline_spacings.split(",")),
                                  scale=float(sig_b.mean()) * 20).to(dev)
        else:
            zfield = ZipField(a.size, a.spec_k, levels=tuple(int(v) for v in a.zip_levels.split(",")), feat=a.zip_feat,
                              hidden=a.zip_hidden, scale=float(sig_b.mean()) * 20).to(dev)
        n_field = sum(p_.numel() for p_ in zfield.parameters())
        LOG["zip_field"] = {"kind": a.zip_field, "spacings": (a.bspline_spacings if a.zip_field == "bspline" else a.zip_levels),
                            "params": n_field}
        tick(f"zip field {a.zip_field}: {n_field} parameters")
        m = ZipSkyModel(zfield, zbasis, lam_nodes, w_nodes, zpsf, ExplicitBackground(exps, ndet), Geometry(exps, ndet, "explicit"),
                        band_index, stripes=Stripes(len(exps), dev) if a.stripes else None, sig=float(sig_b.mean()),
                        lam_smooth=a.spec_smooth, sig_ref=float(sig_b.mean())).to(dev)
        st = torch.load(a.instrument_file, map_location=dev)
        inst = {k: v for k, v in st.items() if k.split(".")[0] in ("psf", "bkg", "geom", "stripes")}
        res = m.load_state_dict(inst, strict=False)
        bad_ = [k for k in res.missing_keys if k.split(".")[0] in ("psf", "bkg", "geom", "stripes")]
        if bad_ or res.unexpected_keys:
            raise RuntimeError(f"instrument mismatch: missing {bad_}, unexpected {res.unexpected_keys}")
        for part in (m.psf, m.bkg, m.geom) + ((m.stripes,) if m.stripes is not None else ()):
            for p_ in part.parameters():
                p_.requires_grad_(False)
        m.build_samplers()
        LOG[f"{zname}_psf_energy_in_window"] = m.ee.mean().item()
        tick(f"{zname}: instrument frozen from {a.instrument_file}; PSF samplers built ({m.no}^2 per node and detector); "
             f"{sum(p_.numel() for p_ in m.sky.parameters()) / 1e6:.2f} M field parameters; {a.zip_samples} samples per pixel")
        LOG["zip_psf_convention"] = ZIP_PSF_CONVENTION
        if a.zip_render == "det":
            m.build_kernels(a.zip_render_os, {(band_index[e["band"]], int(e["det"])) for e in exps})
            m.render_mode = "det"
        LOG[f"{zname}_render"] = m.render_mode
        if a.zip_init_state:
            st0 = torch.load(a.zip_init_state, map_location=dev)
            m.sky.load_state_dict({k[4:]: v for k, v in st0.items() if k.startswith("sky.")})  # strict
            LOG[f"{zname}_init_state"] = a.zip_init_state
            tick(f"{zname}: field weights loaded from {a.zip_init_state} (no warm start)")
        else:
            A0 = spectral_init(init, lam_nodes, w_nodes, zbasis)
            if a.zip_warm_os > 1:  # v28: target on the oversampled grid (bilinear between coadd nodes)
                A0 = F.interpolate(A0[None], scale_factor=a.zip_warm_os, mode="bilinear", align_corners=False)[0]
                m.warm_os = a.zip_warm_os
            if a.zip_warm_solver == "cg":
                assert a.zip_field == "bspline", "CG warm start needs a field linear in its parameters"
                lw = cg_warm_start(m, A0, float(sig_b.mean()), 20 if a.quick else a.zip_warm_steps)
            else:
                optw = torch.optim.Adam(m.sky.parameters(), lr=a.zip_warm_lr)
                for it_w in range(20 if a.quick else a.zip_warm_steps):
                    optw.zero_grad()
                    lw = ((m.grid_coeffs_os(a.zip_warm_os) - A0) / sig_b.mean()).pow(2).mean()
                    lw.backward()
                    optw.step()
            LOG[f"{zname}_warm_mse_sig2"] = float(lw)
            tick(f"{zname}: field warm start on the {a.init} coadd's spectral coefficients ({a.zip_warm_steps} steps), mse/sig2 {float(lw):.3f}")
        if a.zip_eval_only:
            tick(f"{zname}: evaluation only, no fit")
        else:
            fit_zip(zname, m, exps, train_idx, a.sky_lr if a.sky_lr > 0 else 5e-3, a.budget_implicit * 60, make_cb(zname),
                    a.eval_every_min * 60, a.patience, frac=a.zip_frac, nsamp=a.zip_samples)
        LOG["timings"][zname] = round(time.perf_counter() - t, 1)
        models[zname] = m

    if not a.skip_implicit:
        # implicit: warm-start sky field on the swarp image and PSF field on the nominal Gaussians
        iname = (("implicit" if a.sky_model == "grid" else f"implicit_{a.sky_model}") + ("_chrom" if a.psf_model == "chrom" else "")
                 + ("_3d" if spec else "") + a.tag)
        t = time.perf_counter()
        if a.sky_model == "siren":
            fsky = SirenSky(a.size, len(REG.bands), hidden=a.siren_hidden, layers=a.siren_layers, w0=a.siren_w0, scale=sig_b * 20).to(dev)
            opt = torch.optim.Adam(fsky.parameters(), lr=1e-4)
            n_warm = 20 if a.quick else 3000
        elif a.sky_model == "gs":
            fsky = GaussSky(swarp, sig_b, n=a.gs_n).to(dev)
            opt = torch.optim.Adam(fsky.param_groups(0))
            n_warm = 20 if a.quick else 1500
        else:
            if spec:
                fsky = SpectralSky(FieldSky(a.size, a.spec_k, scale=torch.full((a.spec_k,), float(sig_b.mean()) * 20, device=dev)),
                                   RBFSpectralBasis(a.spec_k, learned=a.spec_learned_basis), lam_nodes, w_nodes,
                                   lam_smooth=a.spec_smooth, sig_ref=float(sig_b.mean()), relative=a.spec_relative).to(dev)
            else:
                fsky = FieldSky(a.size, len(REG.bands), scale=sig_b * 20).to(dev)
            opt = torch.optim.Adam(fsky.parameters(), lr=5e-3)
            n_warm = 20 if a.quick else 600
        l = torch.full((), float("nan"))
        for it in range(0 if getattr(a, "load_dir", None) else n_warm):
            opt.zero_grad()
            l = ((fsky() - init) / sig_b[:, None, None]).pow(2).mean()
            l.backward()
            opt.step()
        tick(f"implicit sky warm start done ({a.sky_model}) mse/sig2={float(l.detach()) if n_warm else float('nan'):.3f}")
        ipsf = ImplicitVarPSF(len(REG.bands), ndet, a.nk, n=a.psf_n).to(dev)
        target_k = gauss_kernel(sig0, a.psf_n, dev)
        if a.psf_model == "chrom":
            ipsf = ChromaticPSF(REG.bands, ndet, a.scale, a.nk, n=a.psf_n).to(dev)
        if a.psf_model == "pupil":
            ipsf = make_pupil_psf(a, dets, dev, driz, band_index)
            l = float("nan")
        else:
            l = warm_psf(ipsf, target_k, 0 if getattr(a, "load_dir", None) else (20 if a.quick else (800 if a.psf_model == "band" else 2000)))
        tick(f"implicit PSF warm start done ({a.psf_model}) l={l:.2e}")
        if a.psf_centre > 0 and hasattr(ipsf, "loglam"):
            ipsf.centre_weight = a.psf_centre
        ibkg = ExplicitBackground(exps, ndet) if a.bkg_model == "explicit" else ImplicitBackground(exps, ndet)
        m = SkyModel(fsky, ipsf, ibkg, Geometry(exps, ndet, a.geom_model), band_index,
                     stripes=Stripes(len(exps), dev) if a.stripes else None, sig=float(sig_b.mean()))
        if a.bkg_model != "explicit":
            m.bkg.scale.data = sig_b.mean() * 5
        if getattr(a, "freeze_instrument_from", None):
            freeze_instrument(m, Path(a.freeze_instrument_from) / "explicit_chrom_state.pt", dev, iname)
        if getattr(a, "load_dir", None):
            load_state_checked(m, Path(a.load_dir) / f"{iname}_state.pt", dev)
        else:
            fit(iname, m, exps, steps_i, {"sky": a.sky_lr if a.sky_lr > 0 else (1e-4 if a.sky_model == "siren" else 2e-3), "psf": a.psf_lr if a.psf_lr > 0 else 5e-4, "bkg": 1e-3, "geom": 2e-3},
                a.lam_tv_implicit, sig_b, train_idx, budget_s=a.budget_implicit * 60, callback=make_cb(iname), cb_every_s=a.eval_every_min * 60,
                patience=a.patience)
        LOG["timings"][iname] = round(time.perf_counter() - t, 1)
        models[iname] = m

    # ---------------- evaluation: identical held-out protocol for all five
    LOG["metrics"]["heldout_chi2"] = {}
    resid_all, resid_val = {}, {}
    # source mask: swarp pixels > 5 sigma above sky in that band
    LOG["metrics"]["heldout_chi2_sources"] = {}
    eval_items = [("swarp", swarp), ("drizzle", driz)] + list(models.items())
    if getattr(a, "updec_dir", None) and REG.multi:
        tick("WARNING: --updec-dir products exist for NIRCam only; skipped because HST bands are present")
    elif getattr(a, "updec_dir", None):
        updec = load_updec(a.updec_dir, thead, dev)
        for it, cube in updec.items():
            results[f"updec_iter{it}"] = cube
            eval_items.append((f"updec_iter{it}", cube))
        if models and 10 in updec:  # UPdec's sky observed through our learned instrument (PSF, distortion, background pattern)
            m0 = next(iter(models.values()))
            fixed = GridSky(updec[10].clone()).to(dev)
            fixed.s.requires_grad_(False)
            eval_items.append(("updec_iter10_ourinstrument", SkyModel(fixed, m0.psf, m0.bkg, m0.geom, band_index, stripes=None)))
    if getattr(a, "uq", False) and models:
        global UQ
        uname, um = next(iter(models.items()))
        t_uq = time.perf_counter()
        Fis = sky_fisher(um, exps, train_idx, a.uq_probes)
        eps = 1e-6 * float(Fis[Fis > 0].median())
        var_s = 1.0 / (Fis + eps)
        UQ = {"model": um, "var_s": var_s, "nu": 32, "rows": []}
        (out / "cubes").mkdir(exist_ok=True)
        fits.writeto(out / "cubes" / f"{uname}_sky_sigma.fits", var_s.sqrt().cpu().numpy().astype(np.float32), overwrite=True)
        fits.writeto(out / "cubes" / f"{uname}_sky_fisher.fits", Fis.cpu().numpy().astype(np.float32), overwrite=True)
        LOG["timings"]["uq_fisher"] = round(time.perf_counter() - t_uq, 1)
        tick(f"UQ: sky Fisher done for {uname} in {LOG['timings']['uq_fisher']} s")
    score_idx = train_idx if getattr(a, "score_all", False) else hold_idx
    LOG["scored_exposures"] = "all (in-sample)" if getattr(a, "score_all", False) else "test (left out)"
    for name, obj in eval_items:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            c2, c2s, rs = heldout_chi2(obj, exps, score_idx, band_index, ndet, steps=20 if a.quick else 150, src_masks=src_masks)
        except Exception as exc:  # noqa: BLE001
            tick(f"held-out eval failed for {name}: {exc!r}")
            continue
        LOG["metrics"]["heldout_chi2"][name] = c2
        LOG["metrics"]["heldout_chi2_sources"][name] = c2s
        L = heldout_chi2.last
        LOG["metrics"].setdefault("heldout_robust", {})[name] = L["robust"]
        LOG["metrics"].setdefault("heldout_robust_sources", {})[name] = L["robust_src"]
        LOG["metrics"].setdefault("heldout_per_band", {})[name] = L["per_band"]
        if REG.multi:
            LOG["metrics"].setdefault("heldout_per_instrument", {})[name] = L["per_instrument"]
        LOG["metrics"].setdefault("heldout_photometry", {})[name] = heldout_chi2.photometry
        ph = heldout_chi2.photometry.get("all", {})
        if ph:
            tick(f"PHOT {name}: {ph['n']} source-measurements, dm median {ph['dm_median_mmag']:.1f} mmag, scatter {ph['dm_nmad_mmag']:.1f} mmag, "
                 f"pull nmad {ph['pull_nmad']:.2f}, centroid {ph['centroid_nmad_mas']:.2f} mas")
        resid_all[name] = rs
        per_exp = list(heldout_chi2.per_exposure)
        LOG["metrics"].setdefault("heldout_per_exposure", {})[name] = [{k: v for k, v in r_.items() if k != "pulls"} for r_ in per_exp]
        try:
            bs = bootstrap_scores(per_exp, B=getattr(a, "boot_b", 1000), seed=0)
            LOG["metrics"].setdefault("heldout_bootstrap", {})[name] = bs
            tick(boot_line(name, bs))
        except Exception as exc:  # noqa: BLE001
            tick(f"bootstrap failed for {name}: {exc!r}")
        inst_txt = ""
        if REG.multi:  # v29: instrument roll-up (and per-band lines) of the held-out scores
            f_ = lambda v: "nan" if v is None else f"{v:.4f}"
            inst_txt = "".join(f" | {ins} (n_exp {v['n_exp']}): chi2 all {f_(v['chi2'])} src {f_(v['chi2_src'])} robust all {f_(v['robust'])} src {f_(v['robust_src'])}"
                               for ins, v in L["per_instrument"].items())
            for b_, v in L["per_band"].items():
                tick(f"TEST-BAND {name} {b_} [{REG.instrument[b_]}] (n_exp {v['n_exp']}): chi2 all {f_(v['chi2'])} src {f_(v['chi2_src'])} "
                     f"robust all {f_(v['robust'])} src {f_(v['robust_src'])}")
        tick(f"TEST {name}: chi2 all {c2:.4f} src {c2s:.4f} | robust all {L['robust']:.4f} src {L['robust_src']:.4f}" + inst_txt)
        if val_idx:
            try:
                v2, v2s, rs_v = heldout_chi2(obj, exps, val_idx, band_index, ndet, steps=20 if a.quick else 150, src_masks=src_masks)
                resid_val[name] = rs_v
                LV = heldout_chi2.last
                LOG["metrics"].setdefault("val", {})[name] = [v2, v2s, LV["robust"], LV["robust_src"]]
                tick(f"VAL  {name}: chi2 all {v2:.4f} src {v2s:.4f} | robust all {LV['robust']:.4f} src {LV['robust_src']:.4f}")
            except Exception as exc:  # noqa: BLE001
                tick(f"val eval failed for {name}: {exc!r}")
    if UQ is not None and UQ["rows"]:
        np.save(out / "uq_rows.npy", UQ["rows"], allow_pickle=True)
        tick(f"UQ: saved {sum(len(x['r']) for x in UQ['rows'])} held-out pixels from {len(UQ['rows'])} exposures")
    hold_geo = {exps[i]["band"]: (exps[i]["tx"].cpu().numpy(), exps[i]["ty"].cpu().numpy()) for i in score_idx}
    np.save(out / "heldout_resid.npy", {"resid": resid_all, "coords": hold_geo}, allow_pickle=True)
    if resid_val:
        val_geo = {exps[i]["band"]: (exps[i]["tx"].cpu().numpy(), exps[i]["ty"].cpu().numpy()) for i in val_idx}
        np.save(out / "val_resid.npy", {"resid": resid_val, "coords": val_geo}, allow_pickle=True)

    disp_sig = torch.full((len(REG.bands),), 0.06 / a.scale / 2.3548, device=dev)
    Kdisp = gauss_kernel(disp_sig, a.psf_n, dev)
    with torch.no_grad():
        for name, m in models.items():
            S, CB = m.prepare()
            B = m.psf.basis()
            results[name + "_ownpsf"] = CB[:, 0]
            results[name + "_0p06"] = F.conv2d(F.pad(S[None], (a.psf_n // 2,) * 4, mode="reflect"), Kdisp[:, None], groups=len(REG.bands))[0]
            results[name + "_sky"] = S
            if TARGET_SIG_PX > 0:
                LOG.setdefault("target_resolution_fwhm_arcsec", a.target_fwhm)
            if hasattr(m.sky, "spectral_cube"):  # the continuous sky on a log-lambda grid
                lo_out = 0.8 if LAM_LO >= 0.70 else LAM_LO * 8.0 / 7.0  # inside the domain, as v28's 0.8-5.0 inside 0.70-5.30
                hi_out = 5.0 if LAM_HI <= 5.30 else LAM_HI * 5.0 / 5.3
                lam_out = torch.exp(torch.linspace(math.log(lo_out), math.log(hi_out), 25, device=dev))
                results[name + "_spectral"] = m.sky.spectral_cube(lam_out)
                LOG.setdefault("spectral_cube_lam_um", [round(float(x), 4) for x in lam_out])
            # effective PSF at detector centre and corner for each band (first detector of the band)
            psfs = {}
            for b in REG.bands:
                k = band_index[b]
                es = [e for e in exps if e["band"] == b]
                det = es[0]["det"]
                z = torch.zeros(1, device=dev)
                o = torch.ones(1, device=dev) * 0.9
                wc = m.psf.weights(k, det, z, z)[0]
                wk = m.psf.weights(k, det, o, o)[0]
                psfs[b] = {"centre": effective_psf(B[k], wc).cpu().numpy(), "corner": effective_psf(B[k], wk).cpu().numpy()}
            np.save(out / f"{name}_psf.npy", psfs, allow_pickle=True)
            g = m.geom
            if g.t.requires_grad:
                shifts = (g.t.detach() * a.scale * 1000).cpu().numpy()
                LOG["metrics"][f"{name}_shift_rms_mas"] = float(np.sqrt((shifts ** 2).sum(1)[train_idx].mean()))
                np.save(out / f"{name}_shifts_mas.npy", shifts)
                np.save(out / f"{name}_centres.npy", g.c.cpu().numpy())
            # static distortion field per detector on a 9x9 lattice (mas)
            if g.mode != "fixed":
                lat = torch.linspace(-0.9, 0.9, 9, device=dev)
                yy, xx = torch.meshgrid(lat, lat, indexing="ij")
                dxx, dyy = xx.flatten(), yy.flatten()
                dist = {}
                for di, dn in enumerate(dets):
                    dist[dn] = (g.distortion(di, dxx, dyy) * a.scale * 1000).cpu().numpy()
                np.save(out / f"{name}_distortion_mas.npy", {"lattice": (dxx.cpu().numpy(), dyy.cpu().numpy()), "d": dist}, allow_pickle=True)
            if hasattr(m.psf, "log_sig"):
                LOG["metrics"][f"{name}_psf_core_fwhm_arcsec"] = dict(zip(REG.bands, [float(x) for x in (m.psf.log_sig.exp() * 2.3548 * a.scale).cpu()]))
            LOG["metrics"][f"{name}_gain_rms"] = float(m.bkg.logg.detach()[train_idx].std())

    for name, m in models.items():
        torch.save(m.state_dict(), out / f"{name}_state.pt")
    hdr = fits.Header(thead)
    for k, v in results.items():
        fits.writeto(out / "cubes" / f"{k}.fits", v.detach().cpu().numpy().astype(np.float32), hdr, overwrite=True)
    # per-band display normalisation from the swarp image (99.5th percentile), shared by every method
    norm = torch.stack([torch.quantile(c.flatten()[:: max(1, c.numel() // 1_000_000)], 0.995) for c in swarp]).clamp(min=1e-9)
    ref = (swarp / norm[:, None, None]).sum(0) / 3 * (6.0 / len(REG.bands))  # factor 1.0 for the six NIRCam bands
    stretch = float(torch.quantile(ref.flatten()[:: max(1, ref.numel() // 2_000_000)], 0.995)) / 2
    LOG["display_norm"] = [float(x) for x in norm]
    LOG["display_stretch"] = stretch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (out / "rgb").mkdir(exist_ok=True)
    for k, v in results.items():
        if not k.endswith("_sky") and not k.endswith("_spectral"):
            plt.imsave(out / "rgb" / f"{k}.png", rgb_from_cube(v.detach(), stretch, norm=norm))
    json.dump(LOG, open(out / "metrics.json", "w"), indent=1, default=float)
    tick("done")


if __name__ == "__main__":
    main()
