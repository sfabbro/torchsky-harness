#!/usr/bin/env python3
"""Frozen JWST NIRCam long-wavelength sky for the "fully learned instrument" track (forward-operator-spec.md, section
"Fully learned instrument (separate track)").

What it makes
    One coadd per band (F277W, F356W, F444W and F410M when present) of all NIRCam LW (NRCALONG / NRCBLONG, both modules) imaging
    `_cal` exposures found under the input directories, on ONE tangent-plane grid (TAN, north up, 0.12 arcsec/px, same tangent point
    as pixelstore.py: RA, Dec = 110.8020, -73.4586) that covers the union of the LW footprints of all selected exposures.

Coadd method (--method splat, the default)
    Inverse-variance weighted *bilinear (tent) splat* of every good input pixel: the pixel position comes from the SCI header WCS
    (TAN-SIP, all distortion terms, exactly what pixelstore.read_jwst uses), the value is SCI minus a per-exposure background, the
    weight is 1/ERR^2 times the tent weight. This is the splat_coadd logic of smacs_four_v33.py (h = 1: the 0.063 arcsec input pixel
    is smaller than the 0.12 arcsec output pixel, so every output pixel receives ~3-4 input pixels and none stays empty).
    --method drizzle uses torchsky.map.image_plane_coadd.coadd_image_plane (the classical() path of smacs_four_v33.py, kernel
    "drizzle_square", pixfrac 0.8, weighted mean). Both give surface brightness in MJy/sr (no flux conservation, no AREA/pixel-area
    correction, no outlier rejection, no PSF matching).

Per-exposure background: iterative 3-sigma clipped median of the good SCI pixels (6 iterations, robust sigma from the MAD, every
4th pixel). It is subtracted before coadding, so the frozen sky has ~0 background; the IRAC track absorbs the absolute level in
its per-frame offset b_i.

Good pixels: DQ == 0, finite SCI and ERR, ERR > 0 (pixelstore.JWST_DQ_GOOD policy). Coadd value is NaN outside coverage
(summed weight == 0); the weight map (sum of w/ERR^2, units (MJy/sr)^-2) and a count map (number of exposures with weight > 0)
are written with it.

Approximation used by the IRAC track (stated here and in the report)
    The NIRCam LW PSF (0.12-0.15 arcsec FWHM) is negligible next to IRAC's (~1.7 arcsec, 1.2 arcsec pixels), so this coadd stands in
    for the *sky* s(alpha, delta, band) seen by IRAC. Not deconvolved, not registered beyond the header WCS (the header WCS errors,
    ~0.05-0.1 arcsec between exposures, are small compared with IRAC's pixel and are absorbed by the learned operator), no
    colour-dependent PSF correction. The NIRCam bands only approximate the IRAC bands (IRAC1 3.54 um vs F356W 3.57 um, IRAC2 4.48
    um vs F444W 4.40 um; the learned operator mixes the bands).

Output (out dir): jwst_lw_sky.fits = PRIMARY (grid WCS, no data) + for each band extensions <BAND> (sky, NaN outside),
<BAND>_WGT, <BAND>_N; jwst_lw_sky.json (grid, exposure list, background per exposure); jwst_lw_sky.png (quicklook).

CLI
    python3 jwst_lw_sky.py --out DIR --input DIR [--input DIR ...] [--bands F277W,F356W,F444W,F410M] [--scale 0.12]
                           [--method splat|drizzle] [--max-files N]
Reader: load_sky(path, device="cpu") -> FrozenSky (names, data [nb,H,W] with NaN outside, weight [nb,H,W], wcs, scale_arcsec).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

warnings.filterwarnings("ignore", category=FITSFixedWarning)

CENTER_RA, CENTER_DEC = 110.8020, -73.4586  # same tangent point as pixelstore.Grid
DEFAULT_BANDS = ("F277W", "F356W", "F444W", "F410M")
PIVOT_UM = {"F277W": 2.76174, "F356W": 3.56836, "F410M": 4.08224, "F444W": 4.40432}  # SVO, as pixelstore.PIVOT_UM["NIRCam"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sky_header(ra, dec, nx, ny, scale_arcsec, cx, cy):
    """TAN grid header: north up, east left; CRPIX (1-based FITS) = (cx, cy) the pixel holding (ra, dec)."""
    s = scale_arcsec / 3600.0
    return {"NAXIS": 2, "NAXIS1": int(nx), "NAXIS2": int(ny), "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CUNIT1": "deg",
            "CUNIT2": "deg", "CRVAL1": float(ra), "CRVAL2": float(dec), "CRPIX1": float(cx), "CRPIX2": float(cy), "CD1_1": -s,
            "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": s}


def _str(h, k):
    v = h.get(k, "")
    return str(v).strip().upper() if v is not None else ""


def discover(inputs, bands, max_files=None):
    """LW imaging _cal files (NRC[AB]LONG, EXP_TYPE NRC_IMAGE, PUPIL CLEAR, FILTER in bands). Returns list of dicts."""
    files = []
    for d in inputs:
        files += sorted(glob.glob(os.path.join(d, "**", "*_cal.fits"), recursive=True)) if os.path.isdir(d) else [d]
    out, seen = [], set()
    for f in files:
        base = os.path.basename(f)
        if base in seen:
            continue
        try:
            h0 = fits.getheader(f, 0)
        except Exception as exc:  # noqa: BLE001
            log(f"skip (unreadable) {base}: {exc}")
            continue
        det, filt, pup = _str(h0, "DETECTOR"), _str(h0, "FILTER"), _str(h0, "PUPIL")
        if _str(h0, "INSTRUME") != "NIRCAM" or "LONG" not in det or _str(h0, "EXP_TYPE") not in ("NRC_IMAGE", ""):
            continue
        if filt not in bands or pup not in ("CLEAR", ""):
            continue
        seen.add(base)
        out.append({"path": f, "name": base, "band": filt, "detector": det, "program": str(h0.get("PROGRAM", "")).strip()})
        if max_files and len(out) >= max_files:
            break
    return out


def exposure_corners(path, n=9):
    """Sky positions (ra, dec) of an n x n lattice of pixel centres spanning the SCI array."""
    with fits.open(path, memmap=False) as h:
        sh = h["SCI"].header
        w = WCS(sh, fobj=h)
        ny, nx = h["SCI"].data.shape
    xs, ys = np.meshgrid(np.linspace(0, nx - 1, n), np.linspace(0, ny - 1, n))
    ra, dec = w.all_pix2world(xs.ravel(), ys.ravel(), 0)
    return ra, dec


def plane_offsets(ra, dec, ra0=CENTER_RA, dec0=CENTER_DEC):
    """Gnomonic offsets (arcsec west, arcsec north) of (ra, dec) about (ra0, dec0)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [1.0, 1.0]
    w.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    w.wcs.set()
    x, y = w.all_world2pix(np.asarray(ra, float), np.asarray(dec, float), 0)
    return np.asarray(x), np.asarray(y)


def union_grid(files, scale_arcsec, margin_px=4):
    """Grid header covering the union of the exposures' footprints (bounding box in the tangent plane)."""
    xs, ys = [], []
    for f in files:
        ra, dec = exposure_corners(f["path"])
        x, y = plane_offsets(ra, dec)
        xs.append(x), ys.append(y)
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    # x = arcsec west -> array x index increases to the WEST? target grid has CD1_1 < 0 (RA decreasing with +x = west). Yes: +x = west.
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    nx = int(math.ceil((x1 - x0) / scale_arcsec)) + 2 * margin_px
    ny = int(math.ceil((y1 - y0) / scale_arcsec)) + 2 * margin_px
    # tangent point (offset 0, 0 arcsec) sits at pixel index (-x0/scale + margin, -y0/scale + margin) (0-based) -> FITS +1
    cx = (0.0 - x0) / scale_arcsec + margin_px + 1.0
    cy = (0.0 - y0) / scale_arcsec + margin_px + 1.0
    return sky_header(CENTER_RA, CENTER_DEC, nx, ny, scale_arcsec, cx, cy)


def clipped_median(a, iters=6, nsig=3.0, stride=4):
    v = a.ravel()[::stride]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    med = float(np.median(v))
    for _ in range(iters):
        sig = 1.4826 * float(np.median(np.abs(v - med)))
        if not sig > 0:
            break
        keep = np.abs(v - med) < nsig * sig
        if keep.all():
            break
        v = v[keep]
        med = float(np.median(v))
    return med


def read_exposure(path):
    with fits.open(path, memmap=False) as h:
        sh = h["SCI"].header
        w = WCS(sh, fobj=h)
        sci = h["SCI"].data.astype(np.float32)
        err = h["ERR"].data.astype(np.float32)
        dq = h["DQ"].data.astype(np.uint32)
        bunit = str(sh.get("BUNIT", "")).strip()
        hd = {k: sh[k] for k in sh.keys() if k and k not in ("COMMENT", "HISTORY", "")}
    if bunit != "MJy/sr":
        raise ValueError(f"{path}: BUNIT {bunit!r} is not MJy/sr")
    good = np.isfinite(sci) & np.isfinite(err) & (err > 0) & (dq == 0)
    return w, sci, err, good, hd


def splat_accumulate(num, den, cnt, w_in, sci, err, good, grid_wcs, shape):
    """Add one exposure to the (num, den) accumulators by bilinear splatting of its good pixels. In place."""
    H, W = shape
    yy, xx = np.nonzero(good)
    ra, dec = w_in.all_pix2world(xx.astype(np.float64), yy.astype(np.float64), 0)
    tx, ty = grid_wcs.all_world2pix(ra, dec, 0)
    iv = 1.0 / err[yy, xx].astype(np.float64) ** 2
    val = sci[yy, xx].astype(np.float64)
    x0, y0 = np.floor(tx), np.floor(ty)
    touched = np.zeros(H * W, bool)
    for oy in (0, 1):
        for ox in (0, 1):
            ix, iy = (x0 + ox).astype(np.int64), (y0 + oy).astype(np.int64)
            wt = (1.0 - np.abs(ix - tx)) * (1.0 - np.abs(iy - ty))
            ok = (wt > 0) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            lin = iy[ok] * W + ix[ok]
            wi = wt[ok] * iv[ok]
            num += np.bincount(lin, weights=wi * val[ok], minlength=H * W)
            den += np.bincount(lin, weights=wi, minlength=H * W)
            touched[lin] = True
    cnt += touched


def drizzle_band(exps_data, grid_head, device="cpu"):
    """torchsky drizzle-like coadd (classical() of smacs_four_v33.py). exps_data: list of (header dict, sci-bkg, err, good)."""
    import torch
    from torchsky.map.image_plane_coadd import coadd_image_plane
    from torchsky.map.observation import ImageObservation

    obs = []
    for hd, sci, err, good in exps_data:
        wmap = np.where(good, 1.0 / np.where(good, err, 1.0) ** 2, 0.0).astype(np.float32)
        obs.append(ImageObservation(data=torch.from_numpy(np.where(good, sci, 0.0).astype(np.float32)), header=hd,
                                    weight_map=torch.from_numpy(wmap)))
    res = coadd_image_plane(obs, target_header=grid_head, resampling_kernel="drizzle_square", drizzle_pixfrac=0.8,
                            stacking_method="weighted_mean", device=device)
    img = res.data.float().cpu().numpy()
    wsum = getattr(res, "weight", None)
    wsum = wsum.float().cpu().numpy() if wsum is not None else None
    return img, wsum


def build(inputs, out_dir, bands=DEFAULT_BANDS, scale_arcsec=0.12, method="splat", max_files=None):
    os.makedirs(out_dir, exist_ok=True)
    files = discover(inputs, set(bands), max_files)
    if not files:
        raise SystemExit("no NIRCam LW imaging _cal files found")
    present = [b for b in bands if any(f["band"] == b for f in files)]
    log(f"{len(files)} LW exposures; bands present: {present}; per band " + str({b: sum(f['band'] == b for f in files) for b in present}))
    gh = union_grid(files, scale_arcsec)
    H, W = gh["NAXIS2"], gh["NAXIS1"]
    gw = WCS({k: v for k, v in gh.items() if k not in ("NAXIS",)})
    log(f"grid {W} x {H} px at {scale_arcsec} arcsec ({W * scale_arcsec / 60:.2f}' x {H * scale_arcsec / 60:.2f}')")
    sky, wgt, cnt, rec = {}, {}, {}, []
    for b in present:
        num = np.zeros(H * W)
        den = np.zeros(H * W)
        nct = np.zeros(H * W)
        drz = []
        for f in [f for f in files if f["band"] == b]:
            w_in, sci, err, good, hd = read_exposure(f["path"])
            bkg = clipped_median(np.where(good, sci, np.nan))
            sci = sci - np.float32(bkg)
            if method == "splat":
                splat_accumulate(num, den, nct, w_in, sci, err, good, gw, (H, W))
            else:
                drz.append((hd, sci, err, good))
            rec.append({"name": f["name"], "band": b, "detector": f["detector"], "program": f["program"], "bkg_MJy_sr": bkg,
                        "n_good": int(good.sum())})
            log(f"  {f['name']}  {b}  bkg {bkg:.4f} MJy/sr  good {good.mean():.3f}")
        if method == "splat":
            with np.errstate(divide="ignore", invalid="ignore"):
                img = np.where(den > 0, num / den, np.nan).reshape(H, W).astype(np.float32)
            sky[b], wgt[b], cnt[b] = img, den.reshape(H, W).astype(np.float32), nct.reshape(H, W).astype(np.float32)
        else:
            img, ws = drizzle_band(drz, gh)
            ws = ws if ws is not None else np.isfinite(img).astype(np.float32)
            img = np.where(ws > 0, img, np.nan)
            sky[b], wgt[b], cnt[b] = img, ws, (ws > 0).astype(np.float32)
    # fits
    hdr = fits.Header()
    for k, v in gh.items():
        if not k.startswith("NAXIS"):
            hdr[k] = v
    hdr["BANDS"] = ",".join(present)
    hdr["METHOD"] = method
    hdr["HISTORY"] = "NIRCam LW coadd, per-exposure clipped-median background subtracted, MJy/sr, NaN outside coverage"
    hl = [fits.PrimaryHDU(header=fits.Header({"BANDS": ",".join(present), "METHOD": method}))]
    for b in present:
        for nm, arr in ((b, sky[b]), (b + "_WGT", wgt[b]), (b + "_N", cnt[b])):
            hh = fits.ImageHDU(arr.astype(np.float32), header=hdr.copy(), name=nm)
            hl.append(hh)
    fp = os.path.join(out_dir, "jwst_lw_sky.fits")
    fits.HDUList(hl).writeto(fp, overwrite=True)
    meta = {"grid": gh, "scale_arcsec": scale_arcsec, "bands": present, "method": method, "exposures": rec,
            "pivot_um": {b: PIVOT_UM[b] for b in present},
            "coverage_frac": {b: float(np.isfinite(sky[b]).mean()) for b in present},
            "note": "NIRCam LW coadd stands in for the sky seen by IRAC (PSF 0.12-0.15 arcsec << IRAC 1.7 arcsec)"}
    json.dump(meta, open(os.path.join(out_dir, "jwst_lw_sky.json"), "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, len(present), figsize=(4.5 * len(present), 4.5), squeeze=False)
        for a, b in zip(ax[0], present):
            im = sky[b]
            v = im[np.isfinite(im)]
            lo, hi = (np.percentile(v, [1, 99.5]) if v.size else (0, 1))
            a.imshow(im, origin="lower", cmap="magma", vmin=lo, vmax=hi)
            a.set_title(f"{b} (cov {meta['coverage_frac'][b]:.2f})")
            a.set_xticks([]), a.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "jwst_lw_sky.png"), dpi=70)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log(f"quicklook failed: {exc}")
    log(f"wrote {fp}")
    return fp


# ----------------------------------------------------------------------------------------------- reader
@dataclass
class FrozenSky:
    names: list
    data: "np.ndarray"      # [nb, H, W] float32, NaN outside coverage
    weight: "np.ndarray"    # [nb, H, W] float32
    wcs: WCS
    scale_arcsec: float
    pivot_um: list

    @property
    def shape(self):
        return self.data.shape[1:]

    def world_to_px(self, ra, dec):
        """0-based (x = column, y = row) pixel coordinates of sky positions."""
        x, y = self.wcs.all_world2pix(np.asarray(ra, float), np.asarray(dec, float), 0)
        return np.asarray(x), np.asarray(y)


def load_sky(path, device="cpu"):  # device kept for API symmetry; tensors are made by the consumer
    with fits.open(path, memmap=False) as h:
        names = [b for b in h[0].header["BANDS"].split(",") if b]
        data = np.stack([h[b].data.astype(np.float32) for b in names])
        weight = np.stack([h[b + "_WGT"].data.astype(np.float32) for b in names])
        hd = h[names[0]].header
        w = WCS({k: hd[k] for k in ("CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1",
                                    "CD1_2", "CD2_1", "CD2_2")})
        scale = abs(float(hd["CD2_2"])) * 3600.0
    return FrozenSky(names, data, weight, w, scale, [PIVOT_UM[b] for b in names])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--input", action="append", required=True, help="directory (searched recursively) or file; repeatable")
    ap.add_argument("--bands", default=",".join(DEFAULT_BANDS))
    ap.add_argument("--scale", type=float, default=0.12)
    ap.add_argument("--method", choices=["splat", "drizzle"], default="splat")
    ap.add_argument("--max-files", type=int, default=None)
    a = ap.parse_args(argv)
    build(a.input, a.out, tuple(a.bands.split(",")), a.scale, a.method, a.max_files)


if __name__ == "__main__":
    main()
