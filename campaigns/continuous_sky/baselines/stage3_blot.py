#!/usr/bin/env python
"""stage3_blot: official JWST stage-3 imaging pipeline (calwebb_image3) as a baseline predictor of held-out exposures.

Per (field[, tile], filter) task:
  1. tweakreg (jwst TweakRegStep) on TRAINING + HELD-OUT _cal exposures together  -> every exposure gets the standard aligned WCS.
  2. Image3Pipeline (skymatch, outlier_detection, resample; tweakreg already done, source_catalog skipped) on the TRAINING
     _tweakreg exposures only -> training i2d mosaic (MJy/sr).
  3. blot (stcal.outlier_detection.utils.gwcs_blot, the function outlier_detection itself calls, with outlier_detection's own
     arguments: bilinear interpolation, iscale=1, pixmap_stepsize=1) of the training i2d into each held-out exposure's detector frame
     with its aligned WCS.
  4. write pred/<FIELD>/stage3_blot/<SPLIT>/<rootname>.npz {model, model_valid, meta}; unit check on stars; preliminary chi2/N.

Every non-default pipeline parameter is recorded in the task's summary json (key "nondefault_params") and in the log.
Run `python stage3_blot.py --help`.  See run_stage3_blot.sh for the CANFAR launcher.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict

import numpy as np

SMACS_CAL = "/arc/projects/ots/torchsky-data/jwst_smacs0723/cal"
A = "/arc/projects/ots/adass2026"
PROTO = f"{A}/protocol/v1"
SMACS_BOX = dict(ra=110.8268, dec=-73.4544, size=20.48)
METHOD = "stage3_blot"
# escalation used only if exposure groups fail to align with the pipeline defaults (+ explicit CLI overrides)
TWEAKREG_LADDER = [dict(tolerance=0.3, separation=0.5),
                   dict(searchrad=8.0, brightest=400, snr_threshold=5.0, tolerance=0.3, separation=0.5, minobj=8)]
RMSE_MAX_ARCSEC = 0.25  # a group whose tweakreg fit rms exceeds this (ca. 8 SW / 4 LW pixels) counts as not aligned
BANDS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]

log = logging.getLogger("stage3_blot")


# ------------------------------------------------------------------------------------------------ helpers
def rootname(path_or_name: str) -> str:
    b = os.path.basename(path_or_name)
    return re.sub(r"(_cal)?\.fits$", "", b)


def read_hdr(path):
    from astropy.io import fits
    with fits.open(path) as h:
        p, s = h[0].header, h["SCI"].header
        return dict(file=os.path.basename(path), path=path, rootname=rootname(path), filter=p.get("FILTER"), pupil=p.get("PUPIL"),
                    detector=p.get("DETECTOR"), program=str(p.get("PROGRAM")), observtn=p.get("OBSERVTN"), visit=p.get("VISIT"),
                    exposure=p.get("EXPOSURE"), mjd=p.get("EXPSTART"), pixar_sr=s.get("PIXAR_SR"), pixar_a2=s.get("PIXAR_A2"),
                    bunit=s.get("BUNIT"), crds_ctx=p.get("CRDS_CTX"), cal_ver=p.get("CAL_VER"))


def tangent_offsets_arcsec(ra, dec, ra0, dec0):
    """North-up tangent-plane offsets (east, north) in arcsec; x_east = (ra-ra0) cos(dec0) is accurate to << mas at <1 arcmin."""
    d2r = np.pi / 180.0
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    dra = ((ra - ra0 + 180.0) % 360.0) - 180.0
    # exact gnomonic projection
    a, d, a0, d0 = ra0 * d2r, dec0 * d2r, ra0 * d2r, dec0 * d2r
    a = (ra0 + dra) * d2r
    d = dec * d2r
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return xi / d2r * 3600.0, eta / d2r * 3600.0


def pixel_grid_sky(wcs, shape, step=1):
    """RA, Dec (deg) at every `step`-th pixel centre of a gwcs (y, x ordering of the returned arrays)."""
    yy, xx = np.mgrid[0:shape[0]:step, 0:shape[1]:step]
    ra, dec = wcs(xx.astype(float), yy.astype(float))
    return ra, dec


# ------------------------------------------------------------------------------------------------ splits
def load_split_from_protocol(field, split):
    """Held-out rootnames from protocol splits_<FIELD>.json if it exists, else None."""
    p = f"{PROTO}/splits_{field}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    lst = d["splits"][split]["heldout"]
    return {rootname(x) for x in lst}


def smacs_provisional_heldout(path=f"{A}/scripts/holdout_q5_cal.json"):
    d = json.load(open(path))
    return {rootname(x) for x in d["holdout"]}


def lmc_provisional_heldout(plan, seed=20261110, n_groups=2):
    """PROVISIONAL dither split for LMC (used only until protocol splits_LMC.json exists).

    Per (tile, band): choose n_groups dither groups (programme + dither label = all detectors of one exposure) at random among
    those with at least one exposure covering the tile; hold out all that group's files listed for the tile/band.
    The union over tiles is the global held-out set (an exposure is held out wherever it is used)."""
    rng = np.random.default_rng(seed)
    held = set()
    for t in plan["tiles"]:
        for band, exps in t["exposures"].items():
            groups = defaultdict(list)
            for e in exps:
                groups[(e["file"][:7], e["dither"])].append(e)
            cand = sorted(k for k, v in groups.items() if any(x["covering"] for x in v))
            if len(cand) <= n_groups + 2:
                continue
            for i in rng.choice(len(cand), n_groups, replace=False):
                for e in groups[cand[i]]:
                    held.add(rootname(e["file"]))
    return held


# ------------------------------------------------------------------------------------------------ pipeline pieces
def nondefault_params(args):
    """Only the tweakreg parameters explicitly set on the command line are passed to the step (so that the CRDS parameter reference
    pars-tweakregstep, which is part of the pipeline's own defaults, still applies); they are the recorded deviations."""
    names = ["searchrad", "expand_refcat", "use2dhist", "separation", "tolerance", "minobj", "fitgeometry", "brightest", "snr_threshold",
             "kernel_fwhm"]
    tw = {k: getattr(args, k) for k in names if getattr(args, k) is not None}
    nd = {f"tweakreg.{k}": v for k, v in tw.items()}
    nd["image3.tweakreg run separately"] = ("tweakreg run on training+held-out via TweakRegStep; Image3Pipeline then run on the training "
                                            "_tweakreg exposures with tweakreg and source_catalog skipped")
    return tw, nd


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, rec):
        if rec.name.startswith(("tweakwcs", "jwst.tweakreg", "stpipe")):
            self.records.append((rec.levelname, rec.name, rec.getMessage()))


def parse_tweakreg_log(records):
    """Per tweakreg group id: matches, fit RMSE/MAE (arcsec, tweakwcs units), initial-offset guess, warnings. This jwst version does not
    keep wcs_fit_results in the saved models, so the quality numbers come from the tweakwcs log."""
    info, cur = {}, None
    for lvl, name, msg in records:
        m = re.search(r"GROUP ID: (\S+?)'", msg)
        if m:
            cur = m.group(1)
            info.setdefault(cur, dict(warnings=[]))
        if cur is None:
            continue
        d = info[cur]
        if msg.startswith("Aligning image catalog"):
            d["aligned"] = True
        if "as reference image" in msg:
            d["reference"] = True
        m2 = re.search(r"initial X and Y shifts of (\S+), (\S+) \(arcsec\) with significance of (\S+) and (\d+) matches", msg)
        if m2:
            d["init_dx_arcsec"], d["init_dy_arcsec"] = float(m2.group(1)), float(m2.group(2))
            d["init_significance"], d["init_matches"] = float(m2.group(3)), int(m2.group(4))
        m3 = re.search(r"Found (\d+) matches for", msg)
        if m3:
            d["nmatches"] = int(m3.group(1))
        if msg.startswith("Computed"):
            d["fit_done"] = True
        m4 = re.search(r"FIT RMSE: (\S+)\s+FIT MAE: (\S+)", msg)
        if m4:
            d["rmse_arcsec"], d["mae_arcsec"] = float(m4.group(1)), float(m4.group(2))
        if lvl in ("WARNING", "ERROR"):
            d["warnings"].append(msg[:300])
    return info


def run_tweakreg(files, outdir, tw, tag):
    from jwst.associations.asn_from_list import asn_from_list
    from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
    from jwst.tweakreg import TweakRegStep
    indir = os.path.join(outdir, "in")
    os.makedirs(indir, exist_ok=True)
    local = []
    for f in files:  # symlink inputs next to the association so expnames resolve; inputs are never modified
        dst = os.path.join(indir, os.path.basename(f))
        if not os.path.exists(dst):
            os.symlink(f, dst)
        local.append(os.path.basename(f))
    asn = asn_from_list(local, rule=DMS_Level3_Base, product_name=f"{tag}_all")
    name, ser = asn.dump()
    asn_path = os.path.join(indir, f"{tag}_all_asn.json")
    open(asn_path, "w").write(ser)
    twdir = os.path.join(outdir, "tweakreg")
    os.makedirs(twdir, exist_ok=True)
    for f_ in glob.glob(os.path.join(twdir, "*.fits")):
        os.remove(f_)
    t0 = time.time()
    for attempt in range(3):
        cap = _Capture()
        logging.getLogger().addHandler(cap)
        try:
            TweakRegStep.call(asn_path, output_dir=twdir, save_results=True, **tw)
        finally:
            logging.getLogger().removeHandler(cap)
        pars = [m for _, _, m in cap.records if m.startswith("PARS-")]
        if any("TWEAKREG" in m for m in pars):
            break
        # CRDS parameter reference not retrieved (cache race): the step would silently run with spec defaults -> redo
        log.warning("tweakreg ran without its CRDS parameter reference (attempt %d); retrying", attempt + 1)
        for f_ in glob.glob(os.path.join(twdir, "*.fits")):
            os.remove(f_)
        time.sleep(30)
    else:
        raise RuntimeError("tweakreg could not retrieve its CRDS parameter reference")
    log.info("tweakreg done in %.0fs", time.time() - t0)
    return twdir, parse_tweakreg_log(cap.records), pars


def _tw_path(twdir, rn):
    c = sorted(glob.glob(os.path.join(twdir, f"{rn}_tweakreg*.fits")))
    return c[0] if c else None


def tweakreg_table(cal_files, twdir, ginfo, tile_ra=None, tile_dec=None):
    """Per-exposure alignment quality + the WCS shift tweakreg applied (aligned minus header WCS) at a reference pixel."""
    from stdatamodels.jwst import datamodels
    rows = []
    for f in cal_files:
        rn = rootname(f)
        p = _tw_path(twdir, rn)
        row = dict(rootname=rn, file=os.path.basename(f))
        if p is None:
            row["status"] = "MISSING_OUTPUT"
            rows.append(row)
            continue
        with datamodels.open(f) as c, datamodels.open(p) as t:
            gid = str(t.meta.group_id)
            g = ginfo.get(gid, {})
            row["group_id"] = gid
            if g.get("reference"):
                row["status"] = "REFERENCE"
            elif g.get("fit_done"):
                row["status"] = "SUCCESS"
            elif g.get("aligned"):
                row["status"] = "FAILED"
            else:
                row["status"] = "NOT_ALIGNED"
            for k in ("nmatches", "rmse_arcsec", "mae_arcsec", "init_dx_arcsec", "init_dy_arcsec", "init_significance", "init_matches"):
                if k in g:
                    row[k] = g[k]
            if g.get("warnings"):
                row["warnings"] = g["warnings"]
            ny, nx = c.data.shape
            if tile_ra is not None:
                try:
                    xs, ys = c.meta.wcs.invert(tile_ra, tile_dec)
                    xs, ys = float(np.clip(xs, 0, nx - 1)), float(np.clip(ys, 0, ny - 1))
                    if not (np.isfinite(xs) and np.isfinite(ys)):
                        xs, ys = nx / 2.0, ny / 2.0
                except Exception:
                    xs, ys = nx / 2.0, ny / 2.0
            else:
                xs, ys = nx / 2.0, ny / 2.0
            r0, d0 = c.meta.wcs(xs, ys)
            r1, d1 = t.meta.wcs(xs, ys)
            dra = ((r1 - r0 + 180) % 360 - 180) * np.cos(np.deg2rad(d0)) * 3.6e6  # mas, east
            ddec = (d1 - d0) * 3.6e6
            row.update(ref_x=xs, ref_y=ys, d_east_mas=float(dra), d_north_mas=float(ddec),
                       tile_dx_west_mas=float(-dra), tile_dy_north_mas=float(ddec))
        rows.append(row)
    return rows


def effective_config(sample_tweak_file, tw, resample_kw):
    """Parameters the pipeline applies on top of the spec defaults for this data (CRDS pars-* references), then our overrides."""
    from jwst.pipeline import Image3Pipeline
    from stdatamodels.jwst import datamodels
    out = {}
    with datamodels.open(sample_tweak_file) as m:
        cfg = Image3Pipeline.get_config_from_reference(m)
    st = cfg.get("steps", {}) if cfg else {}
    for k, v in st.items():
        out[k] = {kk: (vv if isinstance(vv, (int, float, str, bool, type(None))) else str(vv)) for kk, vv in dict(v).items()}
    out.setdefault("tweakreg", {}).update({f"OVERRIDE {k}": v for k, v in tw.items()})
    out.setdefault("resample", {}).update({f"OVERRIDE {k}": v for k, v in resample_kw.items()})
    return out


def warm_crds(sample_file):
    """Serially fetch the CRDS context and parameter references into the shared cache before any parallel task starts (concurrent
    first-time cache population races: a step then silently runs without its pars-* reference)."""
    from jwst.outlier_detection import OutlierDetectionStep
    from jwst.pipeline import Image3Pipeline
    from jwst.resample import ResampleStep
    from jwst.skymatch import SkyMatchStep
    from jwst.tweakreg import TweakRegStep
    from stdatamodels.jwst import datamodels
    with datamodels.open(sample_file) as m:
        for c in (Image3Pipeline, TweakRegStep, SkyMatchStep, OutlierDetectionStep, ResampleStep):
            try:
                c.get_config_from_reference(m)
            except Exception as ex:
                log.warning("warm_crds %s: %r", c.__name__, ex)


def run_image3_training(train_tweak_files, outdir, tag, resample_kw, nd, pars_log, extra_steps=None):
    from jwst.associations.asn_from_list import asn_from_list
    from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
    from jwst.pipeline import Image3Pipeline
    d = os.path.join(outdir, "stage3")
    os.makedirs(d, exist_ok=True)
    local = []
    for f in train_tweak_files:
        dst = os.path.join(d, os.path.basename(f))
        if not os.path.exists(dst):
            os.symlink(f, dst)
        local.append(os.path.basename(f))
    asn = asn_from_list(local, rule=DMS_Level3_Base, product_name=f"{tag}_train")
    name, ser = asn.dump()
    asn_path = os.path.join(d, f"{tag}_train_asn.json")
    open(asn_path, "w").write(ser)
    steps = {"tweakreg": {"skip": True}, "source_catalog": {"skip": True}}
    if resample_kw:
        steps["resample"] = dict(resample_kw)
    if extra_steps:
        for k, v in extra_steps.items():
            steps.setdefault(k, {}).update(v)
    t0 = time.time()
    cap = _Capture()
    logging.getLogger().addHandler(cap)
    try:
        Image3Pipeline.call(asn_path, steps=steps, output_dir=d, save_results=True)
    finally:
        logging.getLogger().removeHandler(cap)
    pars_log.extend(m for _, _, m in cap.records if m.startswith("PARS-"))
    log.info("image3 (skymatch, outlier_detection, resample) on %d training exposures done in %.0fs", len(local), time.time() - t0)
    i2d = os.path.join(d, f"{tag}_train_i2d.fits")
    if not os.path.exists(i2d):
        cands = glob.glob(os.path.join(d, "*_i2d.fits"))
        if not cands:
            raise RuntimeError("no i2d produced in %s" % d)
        i2d = cands[0]
    return i2d, d


# ------------------------------------------------------------------------------------------------ blot + checks
def blot_heldout(i2d_path, tweak_path):
    """Blot the training mosaic into a held-out exposure's frame with the pipeline's own function and arguments."""
    from stcal.outlier_detection.utils import gwcs_blot
    from stdatamodels.jwst import datamodels
    with datamodels.open(i2d_path) as m, datamodels.open(tweak_path) as e:
        blot = gwcs_blot(median_data=np.asarray(m.data, dtype=np.float32), median_wcs=m.meta.wcs, blot_shape=e.data.shape,
                         blot_wcs=e.meta.wcs, fillval=np.nan, pixmap_stepsize=1, pixmap_order=1)
        # (identical call to jwst.outlier_detection.utils.flag_model_crs; no extra pixel-area factor, see unit check)
        blot = np.asarray(blot, dtype=np.float32)
        ra, dec = pixel_grid_sky(e.meta.wcs, e.data.shape)
    return blot, ra, dec


def star_unit_check(cal_path, blot, nstars=40, rads=(4, 8), nsig=15.0, badmask=1, satmask=2, H=20, iso=16, strict=True):
    """Aperture-sum ratio blot/data on isolated bright stars of the held-out _cal (sums of MJy/sr pixels, local sky removed).
    Returns rows (x, y, flux_data_r4, ratio_r4, ratio_r8, peak_data, peak_blot)."""
    from astropy.io import fits
    from scipy import ndimage as ndi
    with fits.open(cal_path) as h:
        sci = np.array(h["SCI"].data, dtype=np.float64)
        dq = np.array(h["DQ"].data).astype(np.int64)
    bad = (dq & badmask) != 0
    ok = np.isfinite(sci) & ~bad & np.isfinite(blot)
    d = np.where(ok, sci, np.nan)
    m = np.where(ok, blot, np.nan)
    sm = ndi.gaussian_filter(np.where(ok, sci, 0.0), 1.5)
    med = np.nanmedian(d)
    mad = 1.4826 * np.nanmedian(np.abs(d - med))
    mx = ndi.maximum_filter(sm, size=9)
    pk = (sm == mx) & (sm - med > nsig * mad / 3.0) & ok
    ys, xs = np.nonzero(pk)
    order = np.argsort(-(sm[ys, xs]))
    ys, xs = ys[order], xs[order]
    ny, nx = sci.shape
    yy, xx = np.mgrid[-H:H + 1, -H:H + 1]
    rr = np.hypot(yy, xx)
    sat = (dq & satmask) != 0
    out = []
    for y, x in zip(ys, xs):
        if len(out) >= nstars:
            break
        if y < H or x < H or y > ny - H - 1 or x > nx - H - 1:
            continue
        sl = (slice(y - H, y + H + 1), slice(x - H, x + H + 1))
        core = (rr <= max(rads)) if not strict else np.ones_like(rr, bool)  # strict: whole cutout must be clean; else only the aperture
        if not np.all(ok[sl][core]) or sat[sl][core].any():
            continue
        if (pk[sl] & (rr > 0) & (rr < iso)).sum() > 0:  # isolated: no other detected peak within `iso` px
            continue
        an = (rr >= H - 4) & (rr <= H)
        sd, sb = np.nanmedian(d[sl][an]), np.nanmedian(m[sl][an])
        f = []
        for r in rads:
            ap = rr <= r
            f.append((np.sum(d[sl][ap] - sd), np.sum(m[sl][ap] - sb)))
        if f[0][0] <= 0 or f[1][0] <= 0:
            continue
        out.append((int(x), int(y), float(f[0][0]), float(f[0][1] / f[0][0]), float(f[1][1] / f[1][0]), float(d[y, x] - sd), float(m[y, x] - sb)))
    return out


def prelim_score(cal_path, blot, model_valid, ra, dec, box):
    """Preliminary chi2/N (NOT the protocol scorer): sigma = ERR (no noise factor), S/N proxy = model/ERR, one median offset
    fitted on in-box pixels with model S/N<1, box = north-up square about (ra0, dec0) in the aligned WCS."""
    from astropy.io import fits
    with fits.open(cal_path) as h:
        sci = np.array(h["SCI"].data, dtype=np.float64)
        err = np.array(h["ERR"].data, dtype=np.float64)
        dq = np.array(h["DQ"].data)
    xe, yn = tangent_offsets_arcsec(ra, dec, box["ra"], box["dec"])
    half = box["size"] / 2.0
    inbox = (np.abs(xe) <= half) & (np.abs(yn) <= half)
    valid = (dq == 0) & np.isfinite(sci) & np.isfinite(err) & (err > 0) & inbox & model_valid & np.isfinite(blot)
    res = dict(n_valid=int(valid.sum()))
    if valid.sum() < 100:
        return res
    d, m, s = sci[valid], blot[valid].astype(np.float64), err[valid]
    sky = float(np.median(m))  # S/N proxy: (model - sky)/ERR (the protocol uses S/N of its reference mosaic)
    snr = (m - sky) / s
    lo = snr < 1
    off = float(np.median(d[lo] - m[lo])) if lo.sum() > 50 else 0.0
    r = (d - (m + off)) / s
    edges = [-np.inf, 1, 5, 20, 50, 200, np.inf]
    names = ["<1", "1-5", "5-20", "20-50", "50-200", ">200"]
    snr_o = (m + off - (sky + off)) / s
    bins = {}
    for i, n in enumerate(names):
        k = (snr_o >= edges[i]) & (snr_o < edges[i + 1])
        bins[n] = dict(n=int(k.sum()), chi2_per_n=float(np.mean(r[k] ** 2)) if k.sum() else None)
    hi = snr_o > 5
    res.update(offset=off, model_sky=sky, chi2_per_n_snr_gt5=float(np.mean(r[hi] ** 2)) if hi.sum() else None, n_snr_gt5=int(hi.sum()), bins=bins)
    hb = snr_o > 50
    if hb.sum() > 20:
        res["frac_resid_rms_snr_gt50"] = float(np.sqrt(np.mean(((d[hb] - m[hb] - off) / (m[hb] + off)) ** 2)))
        res["frac_resid_median_snr_gt50"] = float(np.median((d[hb] - m[hb] - off) / (m[hb] + off)))
    # same without the offset
    r0 = (d - m) / s
    hi0 = snr > 5
    res["chi2_per_n_snr_gt5_no_offset"] = float(np.mean(r0[hi0] ** 2)) if hi0.sum() else None
    return res


# ------------------------------------------------------------------------------------------------ task
def run_task(cfg):
    """One (field, [tile], band) task. cfg is a plain dict (picklable)."""
    field, band, tag = cfg["field"], cfg["band"], cfg["tag"]
    work = os.path.join(cfg["scratch"], tag)
    os.makedirs(work, exist_ok=True)
    out_summary = os.path.join(cfg["outroot"], field, cfg["split"], f"{tag}_summary.json")
    os.makedirs(os.path.dirname(out_summary), exist_ok=True)
    fh = logging.FileHandler(os.path.join(cfg["outroot"], field, cfg["split"], f"{tag}.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    t00 = time.time()
    summ = dict(tag=tag, field=field, band=band, split=cfg["split"], tile=cfg.get("tile"), split_source=cfg["split_source"],
                nondefault_params=cfg["nondefault"], tweakreg_params=cfg["tw"], resample_params=cfg["resample_kw"],
                train=[rootname(f) for f in cfg["train"]], heldout=[rootname(f) for f in cfg["heldout"]])
    try:
        import jwst
        import stcal
        summ["versions"] = dict(jwst=jwst.__version__, stcal=stcal.__version__)
        all_files = list(cfg["train"]) + list(cfg["heldout"])
        log.info("%s: %d training + %d held-out exposures", tag, len(cfg["train"]), len(cfg["heldout"]))
        # ---- 1. tweakreg, training + held-out together; escalate the search/detection settings only if some exposure group did not align
        heldset = {rootname(f) for f in cfg["heldout"]}
        ladder = [dict()] + ([] if cfg.get("no_ladder") else TWEAKREG_LADDER)
        summ["tweakreg_attempts"] = []
        for rung, extra in enumerate(ladder):
            tw_try = {**cfg["tw"], **extra}
            twdir, ginfo, pars_tw = run_tweakreg(all_files, work, tw_try, tag)
            rows = tweakreg_table(all_files, twdir, ginfo, cfg.get("tile_ra"), cfg.get("tile_dec"))
            bad = [r for r in rows if r.get("status") not in ("SUCCESS", "REFERENCE") or r.get("rmse_arcsec", 0.0) > RMSE_MAX_ARCSEC]
            summ["tweakreg_attempts"].append(dict(rung=rung, params=tw_try, n_bad=len(bad), n=len(rows)))
            log.info("tweakreg rung %d params %s: %d/%d exposures not aligned", rung, tw_try, len(bad), len(rows))
            if not bad:
                break
        cfg["tw"] = tw_try
        summ["tweakreg_params"] = tw_try
        for k, v in extra.items():
            summ["nondefault_params"][f"tweakreg.{k}"] = f"{v} (escalation rung {rung}, used because some groups did not align with the previous settings)"
        summ["crds_pars_tweakreg"] = pars_tw
        for r in rows:
            r["heldout"] = r["rootname"] in heldset
        summ["tweakreg_table"] = rows
        summ["tweakreg_n_failed"] = len(bad)
        summ["tweakreg_n_success"] = sum(1 for r in rows if r.get("status") == "SUCCESS")
        summ["tweakreg_n_reference"] = sum(1 for r in rows if r.get("status") == "REFERENCE")
        log.info("tweakreg final: %d success, %d reference, %d not aligned of %d exposures", summ["tweakreg_n_success"],
                 summ["tweakreg_n_reference"], len(bad), len(rows))
        bad_roots = {r["rootname"] for r in bad}
        summ["excluded_unaligned_training"] = sorted(bad_roots - heldset)
        json.dump(summ, open(out_summary, "w"), indent=1, default=str)
        # ---- 2. image3 on training only
        train_tw = [_tw_path(twdir, rootname(f)) for f in cfg["train"] if rootname(f) not in bad_roots]
        train_tw = [p for p in train_tw if p]
        summ["crds_pars_image3"] = []
        i2d, d3 = run_image3_training(train_tw, work, tag, cfg["resample_kw"], cfg["nondefault"], summ["crds_pars_image3"])
        summ["effective_step_config"] = effective_config(train_tw[0], cfg["tw"], cfg["resample_kw"])
        # keep the mosaic
        keep = os.path.join(cfg["outroot"], field, cfg["split"], f"{tag}_train_i2d.fits")
        shutil.copy(i2d, keep)
        summ["i2d"] = keep
        from astropy.io import fits
        with fits.open(i2d) as h:
            summ["i2d_header"] = dict(shape=list(h["SCI"].data.shape), bunit=h["SCI"].header.get("BUNIT"), pixar_sr=h["SCI"].header.get("PIXAR_SR"),
                                      pixar_a2=h["SCI"].header.get("PIXAR_A2"), xpix=h["SCI"].header.get("CDELT1"))
        # ---- 3. blot into the held-out frames
        predroot = os.path.join(cfg["pred_root"], field, METHOD, cfg["split"])
        partroot = os.path.join(cfg["outroot"], field, cfg["split"], "partial", tag)
        os.makedirs(predroot, exist_ok=True)
        os.makedirs(partroot, exist_ok=True)
        hdr = {rootname(f): read_hdr(f) for f in cfg["heldout"]}
        per_exp = []
        for f in cfg["heldout"]:
            rn = rootname(f)
            tp = _tw_path(twdir, rn)
            if tp is None:
                per_exp.append(dict(rootname=rn, error="no tweakreg output"))
                continue
            t0 = time.time()
            blot, ra, dec = blot_heldout(i2d, tp)
            valid = np.isfinite(blot)
            if cfg.get("claim_box"):  # claim only pixels near the scoring box (tile): box half-size + margin, in the aligned frame
                b = cfg["claim_box"]
                xe, yn = tangent_offsets_arcsec(ra, dec, b["ra"], b["dec"])
                lim = b["size"] / 2.0 + b["margin"]
                valid &= (np.abs(xe) <= lim) & (np.abs(yn) <= lim)
                dist = np.hypot(xe, yn).astype(np.float32)
            else:
                dist = np.zeros(blot.shape, np.float32)
            model = np.where(valid, blot, 0.0).astype(np.float32)
            meta = dict(method=METHOD, tag=tag, note="blot of training-only calwebb_image3 i2d with tweakreg-aligned WCS; no per-source fit; "
                        "sky = i2d sky (skymatch match-down), no per-exposure background applied (scorer fits it)", nondefault=cfg["nondefault"],
                        i2d=os.path.basename(i2d), alignment_status=[r for r in rows if r["rootname"] == rn][0].get("status"),
                        tweakreg_rmse_arcsec=[r for r in rows if r["rootname"] == rn][0].get("rmse_arcsec"), excluded_unaligned_training=summ["excluded_unaligned_training"])
            np.savez_compressed(os.path.join(partroot, f"{rn}.npz"), model=model, model_valid=valid, dist=dist, meta=json.dumps(meta, default=str))
            rec = dict(rootname=rn, n_model_valid=int(valid.sum()), seconds=round(time.time() - t0, 1))
            if cfg.get("box") is not None:
                rec["prelim_score"] = prelim_score(f, blot, valid, ra, dec, cfg["box"])
            try:
                st = star_unit_check(f, np.where(valid, blot, np.nan))
                rec["stars"] = st
                if st:
                    rat = np.array([s[3] for s in st])
                    rat8 = np.array([s[4] for s in st])
                    rec["star_flux_ratio_r4_median"] = float(np.median(rat))
                    rec["star_flux_ratio_r4_mad_sigma"] = float(1.4826 * np.median(np.abs(rat - np.median(rat))))
                    rec["star_flux_ratio_r8_median"] = float(np.median(rat8))
                    rec["star_flux_ratio_r8_mad_sigma"] = float(1.4826 * np.median(np.abs(rat8 - np.median(rat8))))
                    rec["n_stars"] = len(st)
                    pk = np.array([s[6] / s[5] for s in st])
                    rec["star_peak_ratio_median"] = float(np.median(pk))
            except Exception as ex:
                rec["star_check_error"] = repr(ex)
            per_exp.append(rec)
            log.info("%s blot done: valid %d px, stars %s", rn, rec["n_model_valid"], rec.get("star_flux_ratio_r8_median"))
        summ["heldout_results"] = per_exp
        summ["pixar_ratio_heldout_cal_over_i2d"] = {rn: (h["pixar_sr"] / summ["i2d_header"]["pixar_sr"]) if summ["i2d_header"]["pixar_sr"] else None
                                                    for rn, h in hdr.items()}
        summ["ok"] = True
    except Exception as ex:
        summ["ok"] = False
        summ["error"] = repr(ex)
        summ["traceback"] = traceback.format_exc()
        log.error("TASK FAILED %s\n%s", tag, summ["traceback"])
    summ["seconds"] = round(time.time() - t00, 1)
    json.dump(summ, open(out_summary, "w"), indent=1, default=str)
    if not cfg.get("keep_work"):
        shutil.rmtree(work, ignore_errors=True)
    logging.getLogger().removeHandler(fh)
    return tag, summ.get("ok", False)


# ------------------------------------------------------------------------------------------------ merge (LMC partial tiles -> one npz per exposure)
def merge_partials(outroot, pred_root, field, split):
    parts = defaultdict(list)
    for p in glob.glob(os.path.join(outroot, field, split, "partial", "*", "*.npz")):
        parts[os.path.basename(p)].append(p)
    n = 0
    for fn, ps in parts.items():
        model = None
        for p in sorted(ps):
            z = np.load(p, allow_pickle=False)
            if model is None:
                model, valid, dist = z["model"].copy(), z["model_valid"].copy(), z["dist"].copy()
                metas = [json.loads(str(z["meta"]))]
                continue
            take = z["model_valid"] & (~valid | (z["dist"] < dist))
            model[take] = z["model"][take]
            dist[take] = z["dist"][take]
            valid |= z["model_valid"]
            metas.append(json.loads(str(z["meta"])))
        d = os.path.join(pred_root, field, METHOD, split)
        os.makedirs(d, exist_ok=True)
        meta = dict(method=METHOD, tiles=[m.get("tag") for m in metas], per_tile=metas)
        np.savez_compressed(os.path.join(d, fn), model=model.astype(np.float32), model_valid=valid, meta=json.dumps(meta, default=str))
        n += 1
    return n


# ------------------------------------------------------------------------------------------------ main
def build_tasks(args):
    tw, nd = nondefault_params(args)
    resample_kw_base = {}
    tasks = []
    pred_root = args.pred_root
    if args.field == "SMACS":
        files = sorted(glob.glob(os.path.join(args.smacs_cal, "*_cal.fits")))
        hs = {f: read_hdr(f) for f in files}
        held = load_split_from_protocol("SMACS", args.split)
        src = "protocol splits_SMACS.json"
        if held is None:
            held = smacs_provisional_heldout(args.holdout_json)
            src = "PROVISIONAL: /arc/projects/ots/adass2026/scripts/holdout_q5_cal.json (2 dithers per band)"
        for band in args.bands:
            fb = [f for f in files if hs[f]["filter"] == band]
            if not fb:
                continue
            h = [f for f in fb if rootname(f) in held]
            t = [f for f in fb if rootname(f) not in held]
            tasks.append(dict(field="SMACS", band=band, tag=f"SMACS_{band}", train=t, heldout=h, resample_kw=dict(resample_kw_base), box=SMACS_BOX,
                              claim_box=None, split_source=src))
    else:
        plan = json.load(open(args.plan))
        held = load_split_from_protocol("LMC", args.split)
        src = "protocol splits_LMC.json"
        if held is None:
            held = lmc_provisional_heldout(plan)
            src = "PROVISIONAL per-tile random 2 dither groups per band (seed 20261110), union over tiles = global held-out"
        for t in plan["tiles"]:
            if args.tiles and t["id"] not in args.tiles:
                continue
            for band in args.bands:
                exps = t["exposures"].get(band, [])
                if not exps:
                    continue
                paths = [e["path"] for e in exps]
                hh = [p for p in paths if rootname(p) in held]
                tt = [p for p in paths if rootname(p) not in held]
                if not hh:
                    continue
                pa = np.median([read_pixscale(p) for p in tt[:3]]) if tt else 0.031
                half = (t["size_arcsec"] / 2.0 + args.mosaic_margin) / pa
                n = int(np.ceil(2 * half))
                rk = dict(crval=[t["ra"], t["dec"]], crpix=[n / 2.0, n / 2.0], rotation=0.0, output_shape=[n, n])
                box = dict(ra=t["ra"], dec=t["dec"], size=t["size_arcsec"])
                tasks.append(dict(field="LMC", band=band, tile=t["id"], tag=f"LMC_t{t['id']}_{band}", train=tt, heldout=hh, resample_kw=rk, box=box,
                                  claim_box=dict(ra=t["ra"], dec=t["dec"], size=t["size_arcsec"], margin=args.claim_margin), split_source=src,
                                  tile_ra=t["ra"], tile_dec=t["dec"]))

    nd_all = dict(nd)
    for tk in tasks:
        tk.update(split=args.split, scratch=args.scratch, outroot=args.outroot, pred_root=pred_root, tw=tw, nondefault=dict(nd_all),
                  keep_work=args.keep_work, no_ladder=args.no_ladder)
        if tk["field"] == "LMC":
            tk["nondefault"]["resample.output grid"] = "custom tile grid: crval/crpix/rotation=0/output_shape (tile + margin), pixel scale default"
        # the blot targets sit inside the tile; mosaic margin must exceed the frame offset between our aligned WCS and the protocol's
    return tasks


def read_pixscale(path):
    from astropy.io import fits
    with fits.open(path) as h:
        a2 = h["SCI"].header.get("PIXAR_A2")
    return float(np.sqrt(a2)) if a2 else 0.031


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", choices=["SMACS", "LMC"], required=True)
    ap.add_argument("--split", default="dither")
    ap.add_argument("--bands", default=",".join(BANDS))
    ap.add_argument("--tiles", default="", help="LMC tile ids, comma list (default all)")
    ap.add_argument("--holdout-json", default=f"{A}/scripts/holdout_q5_cal.json", help="SMACS provisional held-out list")
    ap.add_argument("--plan", default=f"{A}/p1/plan_lmc.json")
    ap.add_argument("--smacs-cal", default=SMACS_CAL)
    ap.add_argument("--scratch", default="/scratch/stage3_blot")
    ap.add_argument("--outroot", default=f"{A}/baselines/stage3_blot")
    ap.add_argument("--pred-root", default=f"{PROTO}/pred")
    ap.add_argument("--procs", type=int, default=1, help="tasks run in parallel (separate processes)")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--mosaic-margin", type=float, default=6.0, help="LMC: margin (arcsec) around the tile in the training mosaic")
    ap.add_argument("--claim-margin", type=float, default=4.0, help="LMC: model_valid extends this far (arcsec) beyond the tile box")
    # tweakreg (defaults = jwst defaults; deviations are recorded as non-default)
    ap.add_argument("--searchrad", type=float, default=None)
    ap.add_argument("--expand-refcat", type=int, default=None)
    ap.add_argument("--use2dhist", type=int, default=None)
    ap.add_argument("--separation", type=float, default=None)
    ap.add_argument("--tolerance", type=float, default=None)
    ap.add_argument("--minobj", type=int, default=None)
    ap.add_argument("--fitgeometry", default=None)
    ap.add_argument("--brightest", type=int, default=None)
    ap.add_argument("--snr-threshold", type=float, default=None)
    ap.add_argument("--kernel-fwhm", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N tasks")
    args = ap.parse_args()
    args.expand_refcat = None if args.expand_refcat is None else bool(args.expand_refcat)
    args.use2dhist = None if args.use2dhist is None else bool(args.use2dhist)
    args.bands = [b for b in args.bands.split(",") if b]
    args.tiles = {int(x) for x in args.tiles.split(",") if x}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    os.makedirs(args.scratch, exist_ok=True)
    if args.merge_only:
        print("merged", merge_partials(args.outroot, args.pred_root, args.field, args.split))
        return
    tasks = build_tasks(args)
    if tasks:
        warm_crds(tasks[0]["train"][0])
    if args.limit:
        tasks = tasks[:args.limit]
    log.info("%d tasks: %s", len(tasks), [t["tag"] for t in tasks])
    if args.procs <= 1:
        res = [run_task(t) for t in tasks]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.procs, maxtasksperchild=1) as pool:
            res = pool.map(run_task, tasks, chunksize=1)
    log.info("results: %s", res)
    if args.field == "LMC":
        log.info("merged %d exposures", merge_partials(args.outroot, args.pred_root, "LMC", args.split))
    else:
        # SMACS: one task per band writes partial/<tag>/<rootname>.npz -> merge to the pred directory (single tile, trivial merge)
        log.info("merged %d exposures", merge_partials(args.outroot, args.pred_root, "SMACS", args.split))


if __name__ == "__main__":
    main()
