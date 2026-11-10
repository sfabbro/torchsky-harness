"""Loader for un-registered HST single exposures (ACS/WFC flc, WFC3/IR flt, WFC3/UVIS flc) for a forward-model sky fit.

Principle: exposures are NOT pre-registered. Each chip is read in its own detector frame. The header WCS (SIP + D2IM +
NPOL/CPDIS lookup tables, all applied through astropy's ``all_pix2world``) is only the *starting* map from detector
pixels to the sky; the fit learns a continuous per-exposure transform afterwards.

Returned dicts follow ``load_exposures()`` of smacs_four_v28.py (keys name, band, tx, ty, d, ivar, row, col, dx, dy, bkg0,
path, header, cut, noise_scale, detname) plus: instrument, filter, mjd, exptime, pixar_sr_nominal, area_ratio, units_note,
and a few extras (ext, dq, dq_good_bits, svo_id, photplam, photflam, photfnu_jy, to_mjysr, area_ratio_ref).

UNIT CONVENTION of ``d`` and ``ivar`` (``ivar`` = 1/err^2, err in the same units as ``d``):
    d = surface brightness in MJy/sr *per nominal pixel area*, i.e.
        d = (raw / EXPTIME if BUNIT == 'ELECTRONS' else raw) [e-/s]  *  PHOTFNU [Jy s/e-]  *  1e-6  /  Omega_nom [sr]
    with Omega_nom = (nominal pixel scale)^2, nominal scale = header IDCSCALE (fallback: table below).
    Omega_nom is a CONSTANT per chip; the true per-pixel solid angle is NOT applied. ``area_ratio`` = true/nominal
    (WCS Jacobian / Omega_nom) is returned separately. A flux (MJy) in pixel i is d_i * Omega_nom * area_ratio_i if
    the caller wants flux conservation; for surface-brightness comparisons use d as is.
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

log = logging.getLogger("hst_io")
warnings.filterwarnings("ignore", category=FITSFixedWarning)

C_ANG_PER_S = 2.99792458e18
ARCSEC2_TO_SR = (np.pi / 648000.0) ** 2

# --------------------------------------------------------------------------------------------- DQ policy
# Good-bit sums ("final_bits") from drizzlepac's HAP parameter files (drizzlepac 3.11.0,
# drizzlepac/pars/hap_pars/svm_parameters/<inst>/<det>/*_astrodrizzle_any_n2.json, n6, total):
#   acs/wfc   final_bits "336" = 16 (hot) + 64 (warm) + 256 (saturated)  (n2, n6, total)
#   wfc3/uvis final_bits "336" (post-2012 data); "352" = 32+64+256 for the *_pre2012 files
#   wfc3/ir   final_bits "528" = 16 (hot) + 512 (bad flat/reference pixel)   (n2, n4, total)
# Bit meanings follow the comment block in drizzlepac/runastrodriz.py (1024 sink/charge trap, 256 saturated/full well,
# 64 warm, 16 hot, 512 bad reference-file pixel).
# NOTE: the stand-alone AstroDrizzle task default is final_bits=0 (any non-zero DQ is bad) = the "strict" policy here.
# The pre-2012 UVIS cut at MJD 55927 (2012-01-01) is my choice; drizzlepac's own code does not define it.
# 5 Nov 2026: saturated (256) removed from the good bits: until a nonlinear detector model exists, saturated pixels must
# not be fitted as linear. drizzlepac HAP n2/n6 use 336 (16+64+256); we use 80 (16 hot + 64 warm), 96 before 2012 for UVIS.
DQ_GOOD_BITS = {"ACS/WFC": 80, "WFC3/UVIS": 80, "WFC3/IR": 528}
DQ_GOOD_BITS_UVIS_PRE2012 = 96
UVIS_PRE2012_MJD = 55927.0

_NOMINAL_SCALE = {"WFC3/IR": 0.12825, "ACS/WFC": 0.05, "WFC3/UVIS": 0.04}  # arcsec/pix; IR value = IDCSCALE of the sample
_SVO_DET = {"WFC3/IR": "WFC3_IR", "ACS/WFC": "ACS_WFC", "WFC3/UVIS": "WFC3_UVIS2"}  # UVIS id NOT verified (SVO offline)
_SHORT = {"WFC3/IR": "WFC3IR", "ACS/WFC": "ACS", "WFC3/UVIS": "UVIS"}
HST_DIAMETER_M = 2.4


def hst_band_info(instrument: str, filt: str) -> dict:
    """Static info for an HST instrument/filter: SVO filter id, telescope diameter [m], nominal pixel scale [arcsec].
    Pivot wavelength is deliberately not hard-coded: the loader records PHOTPLAM from the header ("photplam", Angstrom)."""
    inst = instrument.strip().upper()
    if inst not in _SVO_DET:
        raise ValueError(f"unsupported instrument {instrument!r}")
    f = filt.strip().upper()
    return {"svo_id": f"HST/{_SVO_DET[inst]}.{f}", "diameter_m": HST_DIAMETER_M,
            "pixel_scale_arcsec": _NOMINAL_SCALE[inst], "instrument": inst, "filter": f}


def _good_bits(inst, mjd, dq_policy):
    if isinstance(dq_policy, (int, np.integer)):
        return int(dq_policy)
    if dq_policy == "strict":
        return 0
    if dq_policy == "default":
        if inst == "WFC3/UVIS" and mjd < UVIS_PRE2012_MJD:
            return DQ_GOOD_BITS_UVIS_PRE2012
        return DQ_GOOD_BITS[inst]
    raise ValueError(f"dq_policy must be 'default', 'strict' or an int good-bit sum, got {dq_policy!r}")


def dq_pass(dq, good_bits):
    """True where the DQ word contains no bit outside ``good_bits`` (int16 -> uint16 so bit 32768 is handled)."""
    d = np.asarray(dq).astype(np.int64) & 0xFFFF
    return (d & ~int(good_bits) & 0xFFFF) == 0


# --------------------------------------------------------------------------------------------- header helpers
def _identify(h0):
    ins = str(h0.get("INSTRUME", "")).strip().upper()
    det = str(h0.get("DETECTOR", "")).strip().upper()
    return f"{ins}/{det}"


def _filter_name(h0):
    for k in ("FILTER", "FILTER1", "FILTER2"):
        v = str(h0.get(k, "") or "").strip().upper()
        if v and not v.startswith("CLEAR") and v != "N/A":
            return v
    return "CLEAR"


def _get(sh, h0, key, default=None):
    """Keyword lookup: SCI header first, then primary header (HST puts PHOT* in different places per instrument)."""
    for h in (sh, h0):
        if key in h and h[key] not in (None, ""):
            return h[key]
    return default


WCS_POLICY = "apriori"  # apriori | archive (set by the caller)


def apriori_key(hdr):
    """Alternate-WCS key of the a priori solution: guide-star pointing with the catalogue-level GSC correction
    ('IDC_xxx-GSC...') if present, else plain 'IDC_xxx'. Never an a posteriori '-FIT_*' / HSC solution, which MAST
    derived by registering these images (a pre-alignment the model must not use). None if neither exists."""
    alts = {k[-1]: str(hdr[k]) for k in hdr if k.startswith("WCSNAME") and len(k) == 8}
    gsc = [k for k, v in alts.items() if "-GSC" in v and "FIT" not in v]
    plain = [k for k, v in alts.items() if v.startswith("IDC_") and "-" not in v]
    return (gsc or plain or [None])[0]


def hst_wcs(hdul, ext, policy=None):
    """Starting WCS of one SCI extension with ALL distortions (SIP, D2IM, CPDIS/NPOL lookup tables).
    policy 'apriori' (default): the linear part (CRVAL, CRPIX, CD) from the a priori alternate WCS, the distortion
    (same IDC solution) from the primary; 'archive': the primary WCS as MAST serves it (usually '-FIT_REL_GAIA*')."""
    hdr = hdul[("SCI", ext)].header
    w = WCS(hdr, fobj=hdul)
    policy = policy or WCS_POLICY
    if policy == "apriori":
        key = apriori_key(hdr)
        if key is None:
            raise ValueError(f"no a priori alternate WCS in {hdr.get('WCSNAME')}")
        idc_p, idc_a = str(hdr.get("WCSNAME", "")).split("-")[0], str(hdr["WCSNAME" + key]).split("-")[0]
        if idc_p != idc_a:
            raise ValueError(f"distortion solutions differ: {idc_p} vs {idc_a}")
        wa = WCS(hdr, key=key, relax=True, fobj=hdul)
        w.wcs.crval = wa.wcs.crval
        w.wcs.crpix = wa.wcs.crpix
        w.wcs.cd = wa.wcs.cd if wa.wcs.has_cd() else wa.wcs.get_pc() * wa.wcs.cdelt[:, None]
        w.wcs.set()
        w.wcs_name_used = str(hdr["WCSNAME" + key])
    else:
        w.wcs_name_used = str(hdr.get("WCSNAME", ""))
    return w


def _strip_lookups(w):
    w2 = w.deepcopy()
    w2.cpdis1 = w2.cpdis2 = None
    w2.det2im1 = w2.det2im2 = None
    return w2


def lookup_table_contribution(w, shape, step=16, pixels=None):
    """Displacement (mas) on the sky between the full WCS and the same WCS with D2IM/CPDIS lookup tables removed.
    ``pixels`` (N,2) 0-based (x, y), or a lattice with ``step`` over ``shape=(ny, nx)``. Returns dict of max/median/rms."""
    if pixels is None:
        ny, nx = shape
        yy, xx = np.mgrid[0:ny:step, 0:nx:step]
        x, y = xx.ravel().astype(float), yy.ravel().astype(float)
    else:
        x, y = np.asarray(pixels, float).T
    has = w.cpdis1 is not None or w.det2im1 is not None
    if not has:
        return {"has_lookup": False, "max_mas": 0.0, "median_mas": 0.0, "rms_mas": 0.0}
    r1, d1 = w.all_pix2world(x, y, 0)
    r0, d0 = _strip_lookups(w).all_pix2world(x, y, 0)
    sep = np.hypot((r1 - r0) * np.cos(np.deg2rad(d1)), d1 - d0) * 3.6e6
    return {"has_lookup": True, "max_mas": float(sep.max()), "median_mas": float(np.median(sep)),
            "rms_mas": float(np.sqrt(np.mean(sep ** 2)))}


def pixel_area_arcsec2(w, x, y, eps=0.5):
    """True pixel solid angle [arcsec^2] at 0-based detector pixel (x, y) from the full-WCS Jacobian
    (central differences of ``all_pix2world``, tangent-plane at each pixel)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ra0, de0 = w.all_pix2world(x, y, 0)
    c = np.cos(np.deg2rad(de0))
    rxp, dxp = w.all_pix2world(x + eps, y, 0)
    rxm, dxm = w.all_pix2world(x - eps, y, 0)
    ryp, dyp = w.all_pix2world(x, y + eps, 0)
    rym, dym = w.all_pix2world(x, y - eps, 0)

    def dra(a, b):  # wrap-safe RA difference
        return (a - b + 180.0) % 360.0 - 180.0
    ax0, ax1 = dra(rxp, rxm) * c / (2 * eps) * 3600, (dxp - dxm) / (2 * eps) * 3600
    ay0, ay1 = dra(ryp, rym) * c / (2 * eps) * 3600, (dyp - dym) / (2 * eps) * 3600
    return np.abs(ax0 * ay1 - ax1 * ay0)


def units_conversion(sh, h0, inst):
    """Factors to convert raw SCI/ERR to MJy/sr per nominal pixel. Returns a dict (see module docstring)."""
    bunit = str(_get(sh, h0, "BUNIT", "")).strip().upper()
    exptime = float(_get(sh, h0, "EXPTIME", 0.0) or 0.0)
    if bunit == "ELECTRONS/S":
        divide = False
    elif bunit == "ELECTRONS":
        divide = True
        if exptime <= 0:
            raise ValueError("BUNIT=ELECTRONS but no EXPTIME")
    else:
        raise ValueError(f"unsupported BUNIT {bunit!r} (need ELECTRONS or ELECTRONS/S)")
    photplam = _get(sh, h0, "PHOTPLAM")
    photflam = _get(sh, h0, "PHOTFLAM")
    photfnu = _get(sh, h0, "PHOTFNU")
    src = "PHOTFNU"
    if photfnu is None:
        if photflam is None or photplam is None:
            raise ValueError("neither PHOTFNU nor PHOTFLAM+PHOTPLAM found")
        photfnu = 1e23 * float(photflam) * float(photplam) ** 2 / C_ANG_PER_S  # Jy s / e-
        src = "PHOTFLAM*PHOTPLAM^2/c"
    elif photflam is not None and photplam is not None:
        chk = 1e23 * float(photflam) * float(photplam) ** 2 / C_ANG_PER_S
        if abs(chk / float(photfnu) - 1) > 1e-3:
            log.warning("PHOTFNU and PHOTFLAM/PHOTPLAM disagree by %.2e", chk / float(photfnu) - 1)
    scale = sh.get("IDCSCALE", None)
    scale_src = "IDCSCALE"
    if scale is None:
        scale, scale_src = _NOMINAL_SCALE[inst], "table"
    omega = float(scale) ** 2 * ARCSEC2_TO_SR
    to_mjysr = float(photfnu) * 1e-6 / omega / (exptime if divide else 1.0)
    return {"bunit": bunit, "exptime": exptime, "divided_by_exptime": divide, "photfnu_jy": float(photfnu),
            "photfnu_source": src, "photflam": None if photflam is None else float(photflam),
            "photplam": None if photplam is None else float(photplam), "nominal_scale": float(scale),
            "nominal_scale_source": scale_src, "omega_nom_sr": omega, "to_mjysr": to_mjysr}


def noise_scale(d, ivar, row, col):
    """Same estimator as smacs_four_v28.noise_scale (first differences along detector rows)."""
    order = torch.argsort(row * 100000 + col)
    d, iv, r, c = d[order], ivar[order], row[order], col[order]
    nb = (r[1:] == r[:-1]) & (c[1:] == c[:-1] + 1)
    z = (d[1:] - d[:-1])[nb] / torch.sqrt(1.0 / iv[1:][nb] + 1.0 / iv[:-1][nb])
    if z.numel() < 1000:
        return 1.0
    z = z[:: max(1, z.numel() // 2_000_000)]
    return float(1.4826 * (z - z.median()).abs().median())


# --------------------------------------------------------------------------------------------- main loader
def load_hst_exposures(files, thead, device, pad=24, dq_policy="default", calibrate_noise=False, min_pixels=2000):
    """One dict per chip (SCI extension) of every HST file that overlaps the target grid ``thead`` (a plain TAN header dict).

    dq_policy: "default" (drizzlepac good bits per instrument, see DQ_GOOD_BITS), "strict" (DQ == 0) or an int good-bit sum.
    WFPC2 (and anything not ACS/WFC, WFC3/IR, WFC3/UVIS) is skipped with a logged reason.
    """
    from astropy.io.fits import Header
    tw = WCS(Header(thead))
    size = thead["NAXIS1"]
    exps = []
    for f in files:
        f = str(f)
        try:
            hdul = fits.open(f, memmap=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: cannot open (%s)", os.path.basename(f), exc)
            continue
        with hdul:
            h0 = hdul[0].header
            inst = _identify(h0)
            if inst not in DQ_GOOD_BITS:
                log.warning("skip %s: instrument/detector %s not supported (WFPC2 etc.)", os.path.basename(f), inst)
                continue
            root = str(h0.get("ROOTNAME", "") or os.path.basename(f).split("_")[0]).strip().lower()
            filt = _filter_name(h0)
            mjd = float(h0.get("EXPSTART", np.nan))
            good = _good_bits(inst, mjd, dq_policy)
            exts = [h.ver for h in hdul if h.name == "SCI"]
            for ext in exts:
                sh = hdul[("SCI", ext)].header
                try:
                    w = hst_wcs(hdul, ext)
                    conv = units_conversion(sh, h0, inst)
                except Exception as exc:  # noqa: BLE001
                    log.warning("skip %s sci%d: %s", os.path.basename(f), ext, exc)
                    continue
                sci = hdul[("SCI", ext)].data
                ny, nx = sci.shape
                # coarse footprint test on a 32-px lattice first
                yy, xx = np.mgrid[0:ny:32, 0:nx:32]
                ra, de = w.all_pix2world(xx.ravel().astype(float), yy.ravel().astype(float), 0)
                tpx, tpy = tw.all_world2pix(ra, de, 0)
                inside = ((tpx > -pad - 64) & (tpx < size + pad + 64) & (tpy > -pad - 64) & (tpy < size + pad + 64))
                if not inside.any():
                    log.info("skip %s sci%d: outside target grid", os.path.basename(f), ext)
                    continue
                x0, x1 = max(int(xx.ravel()[inside].min()) - 32, 0), min(int(xx.ravel()[inside].max()) + 33, nx)
                y0, y1 = max(int(yy.ravel()[inside].min()) - 32, 0), min(int(yy.ravel()[inside].max()) + 33, ny)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                xf, yf = xx.ravel(), yy.ravel()
                ra, de = w.all_pix2world(xf.astype(float), yf.astype(float), 0)
                tpx, tpy = tw.all_world2pix(ra, de, 0)
                raw = sci[y0:y1, x0:x1].astype(np.float64).ravel()
                err = hdul[("ERR", ext)].data[y0:y1, x0:x1].astype(np.float64).ravel()
                dq = hdul[("DQ", ext)].data[y0:y1, x0:x1].ravel()
                dqu = dq.astype(np.int64) & 0xFFFF
                keep = ((tpx > -pad) & (tpx < size + pad) & (tpy > -pad) & (tpy < size + pad)
                        & np.isfinite(raw) & np.isfinite(err) & (err > 0) & dq_pass(dq, good))
                if keep.sum() < min_pixels:
                    log.info("skip %s sci%d: only %d kept pixels", os.path.basename(f), ext, int(keep.sum()))
                    continue
                xk, yk = xf[keep], yf[keep]
                nom2 = conv["nominal_scale"] ** 2
                area = pixel_area_arcsec2(w, xk.astype(float), yk.astype(float)) / nom2
                ref = pixel_area_arcsec2(w, np.array([w.wcs.crpix[0] - 1]), np.array([w.wcs.crpix[1] - 1]))[0] / nom2
                dk = torch.from_numpy((raw[keep] * conv["to_mjysr"]).astype(np.float32))
                ek = torch.from_numpy((err[keep] * conv["to_mjysr"]).astype(np.float32))
                rk = torch.from_numpy(yk.astype(np.int64))
                ck = torch.from_numpy(xk.astype(np.int64))
                nsc = 1.0
                if calibrate_noise:
                    nsc = noise_scale(dk, 1.0 / ek ** 2, rk, ck)
                    ek = ek * nsc
                chip = sh.get("CCDCHIP", None)
                if inst == "WFC3/IR":
                    detname = "WFC3_IR"
                else:
                    detname = f"{inst.replace('/', '_')}{int(chip) if chip is not None else ext}"
                hd = {k: sh[k] for k in sh.keys() if k and k not in ("COMMENT", "HISTORY", "")}
                svo = hst_band_info(inst, filt)["svo_id"] if filt != "CLEAR" else None
                exps.append({
                    "name": f"{root}_sci{ext}", "band": f"{_SHORT[inst]}_{filt}",
                    "tx": torch.from_numpy(tpx[keep].astype(np.float32)).to(device),
                    "ty": torch.from_numpy(tpy[keep].astype(np.float32)).to(device),
                    "d": dk.to(device), "ivar": (1.0 / ek ** 2).to(device),
                    "row": rk.to(device), "col": ck.to(device),
                    "dx": torch.from_numpy(((xk / nx) * 2 - 1).astype(np.float32)).to(device),
                    "dy": torch.from_numpy(((yk / ny) * 2 - 1).astype(np.float32)).to(device),
                    "bkg0": float(dk.median()), "path": f, "header": hd, "cut": (y0, y1, x0, x1),
                    "noise_scale": nsc, "detname": detname,
                    # --- additions
                    "instrument": inst, "filter": filt, "mjd": mjd, "exptime": conv["exptime"],
                    "pixar_sr_nominal": conv["omega_nom_sr"],
                    "area_ratio": torch.from_numpy(area.astype(np.float32)).to(device),
                    "units_note": ("MJy/sr per NOMINAL pixel area (Omega_nom=%.6g arcsec^2, constant); "
                                   "d=raw%s*PHOTFNU(%.6g Jy s/e-, from %s)*1e-6/Omega_nom; area_ratio=true/nominal NOT applied"
                                   % (nom2, "/EXPTIME" if conv["divided_by_exptime"] else "", conv["photfnu_jy"],
                                      conv["photfnu_source"])),
                    "ext": ext, "dq": torch.from_numpy(dqu[keep].astype(np.int32)).to(device), "dq_good_bits": good,
                    "svo_id": svo, "photplam": conv["photplam"], "photflam": conv["photflam"],
                    "photfnu_jy": conv["photfnu_jy"], "to_mjysr": conv["to_mjysr"], "area_ratio_ref": float(ref),
                })
    return exps


def dq_keep_fractions(path, ext, policies=("default", "strict")):
    """Fraction of ALL chip pixels (not just the target crop) passing DQ alone and DQ + finite SCI/ERR>0, per policy."""
    out = {}
    with fits.open(path, memmap=False) as h:
        h0 = h[0].header
        inst = _identify(h0)
        mjd = float(h0.get("EXPSTART", np.nan))
        dq = h[("DQ", ext)].data
        e = h[("ERR", ext)].data
        ok = np.isfinite(h[("SCI", ext)].data) & np.isfinite(e) & (e > 0)
        for p in policies:
            g = _good_bits(inst, mjd, p)
            m = dq_pass(dq, g)
            out[p if not isinstance(p, int) else f"bits{p}"] = {
                "good_bits": g, "dq_only": float(m.mean()), "dq_and_finite": float((m & ok).mean())}
    return out


# --------------------------------------------------------------------------------------------- JWST NIRCam cal helper
def jwst_area_ratio(header_or_path, pixels, method="auto"):
    """True/nominal pixel area for 0-based detector pixels ``pixels`` (N,2) = (x=col, y=row) of a NIRCam cal image.

    method "jacobian": WCS Jacobian of the SCI-header WCS [arcsec^2] / PIXAR_A2 (works with a Header or a path).
    method "area": bilinear sample of the AREA extension (path only). The AREA map is the pixel-area map relative to
                   the average pixel (PIXAR_SR/PIXAR_A2 are its mean), so it is directly true/nominal.
    method "auto": AREA if available, else Jacobian.
    """
    px = np.asarray(pixels, float).reshape(-1, 2)
    x, y = px[:, 0], px[:, 1]
    hdul = None
    if isinstance(header_or_path, (str, Path)):
        hdul = fits.open(header_or_path, memmap=False)
        hdr = hdul[("SCI", 1)].header
    else:
        hdr = header_or_path
    try:
        has_area = hdul is not None and "AREA" in hdul
        if method == "auto":
            method = "area" if has_area else "jacobian"
        if method == "area":
            if not has_area:
                raise ValueError("no AREA extension available")
            a = hdul["AREA"].data
            ny, nx = a.shape
            xc, yc = np.clip(x, 0, nx - 1), np.clip(y, 0, ny - 1)
            x0, y0 = np.minimum(np.floor(xc).astype(int), nx - 2), np.minimum(np.floor(yc).astype(int), ny - 2)
            fx, fy = xc - x0, yc - y0
            return ((1 - fx) * (1 - fy) * a[y0, x0] + fx * (1 - fy) * a[y0, x0 + 1]
                    + (1 - fx) * fy * a[y0 + 1, x0] + fx * fy * a[y0 + 1, x0 + 1]).astype(np.float64)
        if method == "jacobian":
            w = WCS(hdr, fobj=hdul) if hdul is not None else WCS(hdr)
            return pixel_area_arcsec2(w, x, y) / float(hdr["PIXAR_A2"])
        raise ValueError(method)
    finally:
        if hdul is not None:
            hdul.close()
