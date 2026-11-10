#!/usr/bin/env python
"""stage3_blot for the SMACS HST WFC3/IR exposures: AstroDrizzle (the standard HST stage-3 drizzle) + blot (drizzlepac.ablot).

Per filter:
  1. drizzlepac TweakReg on TRAINING + HELD-OUT _flt together (relative alignment, default parameters; headers of working COPIES updated).
  2. AstroDrizzle on the TRAINING exposures only (default parameters, build=True).
  3. blot of the training _drz SCI into each held-out exposure's frame (drizzlepac.ablot.do_blot, the core of ablot.blot, default poly5
     interpolation, exptime=1 so the output stays in electrons/s).
  4. pred/SMACS_HST/stage3_blot/<SPLIT>/<rootname>.npz {model, model_valid, meta} in the units of the _flt SCI (ELECTRONS/S), unit check
     on stars, preliminary chi2/N.
There is no protocol split for HST yet (v1.1): PROVISIONAL split = 2 exposures per filter (seeded random), training keeps all others.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage3_blot import prelim_score, star_unit_check  # noqa: E402

log = logging.getLogger("stage3_blot_hst")
A = "/arc/projects/ots/adass2026"
SMACS_BOX = dict(ra=110.8268, dec=-73.4544, size=20.48)


def select(raw_dir, manifest, programs):
    rows = list(csv.DictReader(open(manifest)))
    sel = [r for r in rows if r["proposal_id"] in programs and r["instrument"] == "WFC3" and r["detector"] == "IR" and r["product_type"] == "FLT"]
    return sorted((os.path.join(raw_dir, r["product_filename"]), r["filter"].strip().upper(), r["obs_id"]) for r in sel)


def provisional_split(items, seed=20261110, n_held=2):
    rng = np.random.default_rng(seed)
    held = set()
    for f in sorted({it[1] for it in items}):
        roots = sorted(it[2] for it in items if it[1] == f)
        if len(roots) >= n_held + 3:
            held.update(roots[i] for i in rng.choice(len(roots), n_held, replace=False))
    return held


def run_band(band, items, held, work, outroot, pred_root, split, tw_kw, driz_kw, keep):
    from astropy.io import fits
    from drizzlepac import astrodrizzle, tweakreg
    from drizzlepac.ablot import do_blot
    from stwcs.wcsutil import HSTWCS
    tag = f"SMACS_HST_{band}"
    wd = os.path.join(work, tag)
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd)
    os.makedirs(os.path.join(outroot, "SMACS_HST", split), exist_ok=True)
    summ = dict(tag=tag, band=band, split=split, tweakreg_kw=tw_kw, astrodrizzle_kw=driz_kw)
    cwd = os.getcwd()
    try:
        import drizzlepac
        summ["drizzlepac"] = drizzlepac.__version__
        files = []
        for p, f, root in items:
            if f != band:
                continue
            dst = os.path.join(wd, os.path.basename(p))
            shutil.copy(p, dst)
            files.append((dst, root))
        train = [d for d, r in files if r not in held]
        held_f = [d for d, r in files if r in held]
        summ["train"] = [os.path.basename(x) for x in train]
        summ["heldout"] = [os.path.basename(x) for x in held_f]
        os.chdir(wd)
        allf = [os.path.basename(d) for d, _ in files]
        t0 = time.time()
        tweakreg.TweakReg(allf, updatehdr=True, wcsname="STAGE3_TWEAK", shiftfile=True, outshifts="shifts_all.txt", clean=True, **tw_kw)
        summ["tweakreg_seconds"] = round(time.time() - t0, 1)
        rows = []
        if os.path.exists("shifts_all.txt"):
            for line in open("shifts_all.txt"):
                if line.startswith("#") or not line.strip():
                    continue
                t = line.split()
                if len(t) >= 6:
                    try:
                        rows.append(dict(file=t[0], dx_pix=float(t[1]), dy_pix=float(t[2]), rot=float(t[3]), scale=float(t[4]),
                                         rms_x=float(t[5]), rms_y=float(t[6]) if len(t) > 6 else None))
                    except ValueError:
                        pass
        summ["tweakreg_shifts"] = rows
        summ["shifts_text"] = open("shifts_all.txt").read() if os.path.exists("shifts_all.txt") else None
        t0 = time.time()
        out = f"{tag}_train"
        astrodrizzle.AstroDrizzle([os.path.basename(x) for x in train], output=out, build=True, **driz_kw)
        summ["astrodrizzle_seconds"] = round(time.time() - t0, 1)
        drz = f"{out}_drz.fits"
        shutil.copy(drz, os.path.join(outroot, "SMACS_HST", split, drz))
        with fits.open(drz) as h:
            summ["drz_header"] = dict(shape=list(h["SCI"].data.shape), bunit=h["SCI"].header.get("BUNIT"), exptime=h[0].header.get("EXPTIME"),
                                      scale=h["SCI"].header.get("D001SCAL"), orient=h["SCI"].header.get("ORIENTAT"))
        res = []
        src = np.array(fits.getdata(drz, ("SCI", 1)), dtype=np.float32)
        sw = HSTWCS(drz, ("SCI", 1))
        os.makedirs(os.path.join(pred_root, "SMACS_HST", "stage3_blot", split), exist_ok=True)
        for hf in [os.path.basename(x) for x in held_f]:
            rn = hf.replace("_flt.fits", "")
            # drizzlepac's blot core (what ablot.blot / AstroDrizzle's blot step call); exptime=1.0 keeps the output in ELECTRONS/S (cps)
            bw = HSTWCS(hf, ("SCI", 1))
            model = np.array(do_blot(src, sw, bw, 1.0, coeffs=True, interp="poly5", stepsize=10), dtype=np.float32)
            w = bw
            yy, xx = np.mgrid[0:model.shape[0], 0:model.shape[1]]
            ra, dec = w.all_pix2world(xx, yy, 0)
            valid = np.isfinite(model) & (model != 0)
            rec = dict(rootname=rn, n_model_valid=int(valid.sum()), seconds=0)
            rec["prelim_score"] = prelim_score(hf, np.where(valid, model, np.nan), valid, ra, dec, SMACS_BOX)
            try:
                st = star_unit_check(hf, np.where(valid, model, np.nan), rads=(3, 6), badmask=0xFFFF & ~4096, satmask=256, nsig=20.0, nstars=40, H=12, iso=4, strict=False)
                rec["stars"] = st
                if st:
                    r3 = np.array([s[3] for s in st]); r6 = np.array([s[4] for s in st])
                    rec.update(n_stars=len(st), star_flux_ratio_r3_median=float(np.median(r3)), star_flux_ratio_r3_mad_sigma=float(1.4826 * np.median(np.abs(r3 - np.median(r3)))),
                               star_flux_ratio_r6_median=float(np.median(r6)), star_flux_ratio_r6_mad_sigma=float(1.4826 * np.median(np.abs(r6 - np.median(r6)))))
            except Exception as ex:
                rec["star_check_error"] = repr(ex)
            meta = dict(method="stage3_blot", tag=tag, note="AstroDrizzle (training only) + drizzlepac.ablot.blot; tweakreg-aligned header WCS; units ELECTRONS/S; no sky added")
            np.savez_compressed(os.path.join(pred_root, "SMACS_HST", "stage3_blot", split, f"{rn}.npz"), model=model, model_valid=valid, meta=json.dumps(meta))
            res.append(rec)
            log.info("%s blotted: valid %d stars r6 %s", rn, rec["n_model_valid"], rec.get("star_flux_ratio_r6_median"))
        summ["heldout_results"] = res
        summ["ok"] = True
    except Exception as ex:
        summ["ok"] = False
        summ["error"] = repr(ex)
        summ["traceback"] = traceback.format_exc()
        log.error("FAILED %s\n%s", tag, summ["traceback"])
    finally:
        os.chdir(cwd)
    json.dump(summ, open(os.path.join(outroot, "SMACS_HST", split, f"{tag}_summary.json"), "w"), indent=1, default=str)
    if not keep:
        shutil.rmtree(wd, ignore_errors=True)
    return summ.get("ok", False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/arc/projects/ots/torchsky-data/hst_smacs0723/raw")
    ap.add_argument("--manifest", default=f"{A}/scripts/hst_imaging_unique.csv")
    ap.add_argument("--programs", default="14096,16729")
    ap.add_argument("--bands", default="")
    ap.add_argument("--split", default="dither")
    ap.add_argument("--scratch", default="/scratch/stage3_blot_hst")
    ap.add_argument("--outroot", default=f"{A}/baselines/stage3_blot")
    ap.add_argument("--pred-root", default=f"{A}/baselines/stage3_blot_hst_pred")
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    items = select(a.raw, a.manifest, set(a.programs.split(",")))
    held = provisional_split(items)
    bands = [b for b in (a.bands.split(",") if a.bands else sorted({i[1] for i in items})) if b]
    log.info("%d exposures, held-out roots %s", len(items), sorted(held))
    os.makedirs(a.scratch, exist_ok=True)
    ok = [run_band(b, items, held, a.scratch, a.outroot, a.pred_root, a.split, dict(interactive=False, residplot="No plot", see2dplot=False), {}, a.keep_work) for b in bands]
    log.info("results %s", dict(zip(bands, ok)))


if __name__ == "__main__":
    main()
