#!/usr/bin/env python3
"""official_align.py -- official-pipeline alignment + reference mosaics for the evaluation protocol v1 (see INTERFACE.md).

Runs the steps of the official JWST pipeline stage 3 (calwebb_image3) on ALL exposures of a field, through the pipeline's own
step classes with the pipeline's CRDS parameter files (jwst version is recorded in the output):
    tweakreg (relative, then relative + absolute to Gaia DR3 if the catalogue service answers) -> skymatch -> outlier_detection
    -> resample (one mosaic per filter).
Outputs, under <out>/ :
    align_<FIELD>/[tile<ID>/]<rootname>.asdf   aligned WCS (gwcs) of every exposure + its tweakreg group and status
    align_<FIELD>/[tile<ID>/]summary.json      mode actually used (gaia_absolute | relative_only | none), per-exposure sky offset of the
                                               detector centre (aligned - header WCS), tweakreg status, timings
    align_<FIELD>/[tile<ID>/]tweakreg.log      the pipeline log (matches, rms, warnings)
    ref_SMACS_<F>_i2d.fits  /  ref_LMC_tile<ID>_<F>_i2d.fits   reference mosaics
    align_LMC/xcorr_crosscheck.json            (LMC, --xcorr) tweakreg offsets against the cross-correlation shifts

APPROXIMATIONS (stated, not hidden)
  P1  LMC is processed PER TILE: each scoring box (tile) is aligned and mosaicked from the exposures that overlap it in the plan
      (cover_frac > 0), not from the whole 1032-file field at once.  An exposure used by several tiles therefore has one aligned
      WCS per tile (stored under align_LMC/tile<ID>/), and the mosaics are per tile (ref_LMC_tile<ID>_<F>_i2d.fits), not one LMC mosaic
      per filter.  Tile solutions are tied to each other only through Gaia (when absolute alignment worked).
  P2  All filters of a field/tile are fitted by tweakreg TOGETHER (cross-filter relative alignment, one correction per exposure
      "group" = file name without the detector = all detector images of one pointing).  The stock pipeline aligns one filter per
      association; here sources of different filters are matched with each other.  Skymatch / outlier_detection / resample are per filter.
  P3  Two tweakreg passes: (1) relative, wide search ('shift', searchrad {sr1}"), tolerant matching, to absorb header WCS errors of up to
      ~1.8" (LMC programme 1476); (2) relative with tight matching + absolute Gaia DR3 (abs search radius {asr}") using the proper-motion
      corrected epoch of the first image.  If the absolute step fails (service down / too few matches) pass 2 is repeated without Gaia and
      the mode is recorded as relative_only.  A relative-only solution is tied to the first image's header WCS (arbitrary common offset).
  P4  Reference mosaics are cut out on the scoring box (box + {margin}" margin each side), north up, native pixel scale
      (0.031"/px SW, 0.063"/px LW) -- not the full field extent (--grid full gives the pipeline default).  The resampling inputs are
      still ALL aligned exposures of the field/tile.  Outlier detection runs on the pipeline's default (full) frame.
  P5  Source finding is the pipeline default (iraf starfinder, kernel_fwhm 2.5 px, snr 10) with brightest={brightest} (pipeline default 200).
  P7  The sky is SUBTRACTED (skymatch skymethod='global+match', subtract=True; the stock image3 default is 'match' without subtraction, which
      keeps the absolute sky in the i2d).  The reference mosaic is only used for a source S/N, so it must be sky free; the global sky is the
      pipeline's clipped 'mode' of the pixel values and is biased in crowded or diffuse regions (LMC star fields, cluster light in SMACS).
  P6  Pipeline parameters not listed here are the CRDS 'pars-*' reference values of the installed jwst.
"""
import argparse
import datetime
import glob
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ARC = "/arc/projects/ots/adass2026"
PROTO = f"{ARC}/protocol/v1"
SCALE = {"SW": 0.031, "LW": 0.063}
CFG = dict(sr1=4.0, asr=6.0, margin=8.0, brightest=500)
__doc__ = __doc__.format(sr1=CFG["sr1"], asr=CFG["asr"], margin=CFG["margin"], brightest=CFG["brightest"])


def is_lw(detector):
    return detector.upper().endswith("LONG")


def group_of(rec):
    return f'jw{rec["program"]}{rec["observation"]}{rec["visit"]}_{rec["activity"]}_{rec["exposure"]}'


def centre_radec(wcs, shape=(2048, 2048)):
    ra, dec = wcs(shape[1] / 2 - 0.5, shape[0] / 2 - 0.5)
    return float(ra), float(dec)


def write_asn(path, name, recs, key=lambda r: r["path"]):
    members = [dict(expname=key(r), exptype="science", group_id=group_of(r)) for r in recs]
    json.dump(dict(asn_type="image3", asn_id="o001", asn_pool="none", version_id=None, target="x",
                   products=[dict(name=name, members=members)]), open(path, "w"), indent=1)
    return path


# ----------------------------------------------------------------------------- one unit (SMACS field, or one LMC tile)
def tweak_kwargs(preset, pass_no, absolute):
    if preset == "default":                                   # stock pipeline defaults, relative only (cross-check)
        return dict(in_memory=False)
    base = dict(in_memory=False, brightest=CFG["brightest"], minobj=10, expand_refcat=True)
    if pass_no == 1:
        return dict(base, fitgeometry="shift", searchrad=CFG["sr1"], use2dhist=True, separation=0.6, tolerance=0.4, nclip=3, sigma=3.0,
                    abs_refcat="")
    kw = dict(base, fitgeometry="rshift", searchrad=0.6, use2dhist=True, separation=0.25, tolerance=0.15, nclip=3, sigma=3.0)
    if absolute:
        kw.update(abs_refcat="GAIADR3", abs_minobj=8, abs_fitgeometry="rshift", abs_searchrad=CFG["asr"], abs_use2dhist=True,
                  abs_separation=0.5, abs_tolerance=0.3, abs_nclip=3, abs_sigma=3.0)
    else:
        kw.update(abs_refcat="")
    return kw


class MemHandler(logging.Handler):
    """Keeps the formatted pipeline log records in memory so the absolute-alignment outcome can be read back."""
    def __init__(self):
        super().__init__(logging.INFO)
        self.msgs = []

    def emit(self, rec):
        self.msgs.append(rec.getMessage())


def parse_abs_fit(msgs):
    """Outcome of the Gaia (absolute) fit from the tweakwcs log lines logged after 'Aligning to absolute reference catalog'.
    The pipeline step reports COMPLETE even when the absolute fit found too few matches and only the relative part was applied."""
    i0 = max([i for i, m in enumerate(msgs) if "Aligning to absolute reference catalog" in m], default=None)
    if i0 is None:
        return dict(applied=False, reason="absolute alignment not attempted")
    t = msgs[i0:]
    notenough = [m for m in t if "Not enough matches" in m]
    sol = [m for m in t if m.startswith("Final solution based on")]
    out = dict(applied=bool(sol) and not notenough)
    if notenough:
        out["reason"] = notenough[-1]
    if sol:
        out["n_objects"] = int(sol[-1].split()[4])
        for m in t:
            if m.startswith("XSH:"):
                out["xsh_ysh_rot_scale"] = m
            if m.startswith("FIT RMSE"):
                out["fit_rmse_mae_arcsec"] = m
    return out


def lib_status(lib, field="tweakreg"):
    out = []
    with lib:
        for i, m in enumerate(lib):
            out.append(str(getattr(m.meta.cal_step, field, None)))
            lib.shelve(m, i, modify=False)
    return out


def process_unit(job):
    """Runs in a worker process.  job: dict(field, unit, recs, out, work, box, preset, stage, grid)."""
    t0 = time.time()
    field, unit, recs, out, work = job["field"], job["unit"], job["recs"], job["out"], job["work"]
    os.makedirs(out, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    logf = logging.FileHandler(os.path.join(out, "tweakreg.log"), mode="w")
    logf.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(logf)
    mem = MemHandler()
    logging.getLogger().addHandler(mem)
    logging.getLogger().setLevel(logging.INFO)
    for nm in ("stpipe", "stcal", "tweakwcs", "jwst"):         # these loggers may not propagate to the root logger
        lg = logging.getLogger(nm)
        lg.propagate = True
        lg.setLevel(logging.INFO)
    log = logging.getLogger("official_align")
    res = dict(field=field, unit=unit, n_exposures=len(recs), preset=job["preset"], ok=False)
    try:
        from jwst.datamodels import ModelLibrary
        from jwst.tweakreg import TweakRegStep
        from jwst.skymatch import SkyMatchStep
        from jwst.outlier_detection import OutlierDetectionStep
        from jwst.resample import ResampleStep
        from stdatamodels.jwst import datamodels
        import asdf
        import jwst
        res["jwst_version"] = jwst.__version__
        by_root = {r["rootname"]: r for r in recs}

        # ---- original (header) WCS centres
        orig = {}
        for r in recs:
            with datamodels.ImageModel(r["path"]) as m:
                orig[r["rootname"]] = centre_radec(m.meta.wcs, m.data.shape)
        log.info("original WCS read for %d exposures", len(orig))

        # ---- tweakreg
        asn_all = write_asn(os.path.join(work, "asn_all.json"), f"{field}_{unit}", recs)
        mode, lib = "none", None
        if job["preset"] == "default":
            lib = TweakRegStep.call(asn_all, **tweak_kwargs("default", 1, False))
            st = lib_status(lib)
            mode = "relative_only_pipeline_defaults" if "COMPLETE" in st else "none"
            res["tweakreg_status_counts"] = {s: st.count(s) for s in set(st)}
        else:
            lib1 = TweakRegStep.call(asn_all, **tweak_kwargs("ours", 1, False))
            st1 = lib_status(lib1)
            res["pass1_status_counts"] = {s: st1.count(s) for s in set(st1)}
            lib = lib1
            mode = "relative_only" if "COMPLETE" in st1 else "none"
            pass2_done = False
            try:
                n_before = len(mem.msgs)
                lib2 = TweakRegStep.call(lib1, **tweak_kwargs("ours", 2, True))
                st2 = lib_status(lib2)
                res["pass2_gaia_status_counts"] = {s: st2.count(s) for s in set(st2)}
                res["gaia_fit"] = parse_abs_fit(mem.msgs[n_before:])
                if "COMPLETE" in st2:
                    lib, pass2_done = lib2, True
                    mode = "gaia_absolute" if res["gaia_fit"]["applied"] else "relative_only"
            except Exception as e:                           # e.g. catalogue service unreachable
                res["gaia_error"] = repr(e)
                log.warning("Gaia absolute alignment raised: %r", e)
            if not pass2_done:
                try:
                    lib2 = TweakRegStep.call(lib1, **tweak_kwargs("ours", 2, False))
                    st2 = lib_status(lib2)
                    res["pass2_relative_status_counts"] = {s: st2.count(s) for s in set(st2)}
                    if "COMPLETE" in st2:
                        lib, mode = lib2, "relative_only"
                except Exception as e:
                    res["pass2_relative_error"] = repr(e)
                    log.warning("pass 2 relative raised: %r", e)
        res["mode"] = mode
        res["t_tweakreg_s"] = round(time.time() - t0, 1)
        log.info("tweakreg mode: %s", mode)

        # ---- save aligned WCS (asdf) + aligned models
        aligned_dir = os.path.join(work, "aligned")
        os.makedirs(aligned_dir, exist_ok=True)
        per_exp = {}
        with lib:
            for i, m in enumerate(lib):
                fn = os.path.basename(m.meta.filename)
                root = fn.split("_cal")[0].split("_tweakreg")[0]
                r = by_root[root]
                ra0, dec0 = orig[root]
                ra1, dec1 = centre_radec(m.meta.wcs, m.data.shape)
                dxi = (ra1 - ra0) * math.cos(math.radians(dec0)) * 3600.0
                deta = (dec1 - dec0) * 3600.0
                per_exp[root] = dict(group_id=str(m.meta.group_id), filter=r["filter"], detector=r["detector"],
                                     tweakreg=str(getattr(m.meta.cal_step, "tweakreg", None)),
                                     ra_header=ra0, dec_header=dec0, ra_aligned=ra1, dec_aligned=dec1,
                                     d_east_arcsec=dxi, d_north_arcsec=deta, d_arcsec=math.hypot(dxi, deta))
                asdf.AsdfFile({"wcs": m.meta.wcs, "rootname": root, "group_id": str(m.meta.group_id), "mode": mode,
                               "tweakreg_status": per_exp[root]["tweakreg"], "jwst_version": jwst.__version__,
                               "d_east_arcsec": dxi, "d_north_arcsec": deta}).write_to(os.path.join(out, f"{root}.asdf"))
                if job["stage"] == "all":
                    m.save(os.path.join(aligned_dir, f"{root}_tweakreg.fits"))
                lib.shelve(m, i, modify=False)
        res["exposures"] = per_exp
        res["t_wcs_saved_s"] = round(time.time() - t0, 1)
        del lib
        if job["stage"] != "all":
            res["ok"] = True
            return finish(res, out, t0)

        # ---- per filter: skymatch, outlier detection, resample
        box = job["box"]
        res["mosaics"] = {}
        for filt in sorted({r["filter"] for r in recs}):
            tf = time.time()
            frecs = [r for r in recs if r["filter"] == filt]
            asn_f = write_asn(os.path.join(work, f"asn_{filt}.json"), f"{field}_{unit}_{filt}", frecs,
                              key=lambda r: os.path.join(aligned_dir, f'{r["rootname"]}_tweakreg.fits'))
            lw = is_lw(frecs[0]["detector"])
            scale = SCALE["LW" if lw else "SW"]
            n = int(math.ceil((box["size_arcsec"] + 2 * CFG["margin"]) / scale))
            rs = dict(output_shape=[n, n], crpix=[(n - 1) / 2.0, (n - 1) / 2.0], crval=[box["ra"], box["dec"]], rotation=0.0,
                      pixel_scale=scale) if job["grid"] == "cutout" else {}
            lib = SkyMatchStep.call(asn_f, skymethod="global+match", subtract=True, in_memory=False)
            lib = OutlierDetectionStep.call(lib, in_memory=False)
            i2d = ResampleStep.call(lib, in_memory=False, **rs)
            pth = job["ref_path"](filt)
            i2d.save(pth)
            sh = i2d.data.shape
            res["mosaics"][filt] = dict(path=pth, shape=list(sh), pixel_scale_arcsec=scale, n_exposures=len(frecs),
                                        finite_frac=float(np.isfinite(i2d.data).mean()), seconds=round(time.time() - tf, 1))
            log.info("mosaic %s done %s", filt, res["mosaics"][filt])
            del lib, i2d
        res["ok"] = True
    except Exception as e:
        res["error"] = repr(e)
        res["traceback"] = traceback.format_exc()
        log.error("FAILED: %s", res["traceback"])
    return finish(res, out, t0)


def finish(res, out, t0):
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return dict(field=res["field"], unit=res["unit"], ok=res["ok"], mode=res.get("mode"), seconds=res["seconds"], error=res.get("error"))


# ----------------------------------------------------------------------------- cross-check against the cross-correlation shifts
def crosscheck(out_dir, xcorr_path, tiles, label="tweakreg"):
    """Tweakreg offset of every exposure (aligned - header WCS at the detector centre) expressed as the registration shift of
    register_xcorr.py: (dx, dy) in mas to ADD to the header-WCS pixel position in the tile frame (north-up TAN, x to the WEST,
    y to the NORTH)  =>  dx = -d_east, dy = +d_north.  Both are relative to an arbitrary common offset, which is removed per
    (tile, xcorr connected component) with the median difference; the global version is also given."""
    xc = json.load(open(xcorr_path))["exposures"]
    rows = []
    for t in tiles:
        p = os.path.join(out_dir, f"tile{t}", "summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        for root, e in s.get("exposures", {}).items():
            x = xc.get(root + "_cal.fits")
            if not x or not x.get("registered") or e["tweakreg"] != "COMPLETE":
                continue
            rows.append(dict(tile=t, root=root, comp=x.get("component", 0), prog=x["programme"], band=x["band"], group=x["unit"],
                             tw=(-1000 * e["d_east_arcsec"], 1000 * e["d_north_arcsec"]), xc=tuple(x["shift_mas"]), mode=s.get("mode")))
    if not rows:
        return {"n": 0}
    tw = np.array([r["tw"] for r in rows])
    xs = np.array([r["xc"] for r in rows])
    out = {"description": crosscheck.__doc__, "n_pairs": len(rows)}
    for lab, sgn in (("convention_as_documented", 1.0), ("tweakreg_sign_flipped", -1.0)):
        d = sgn * tw - xs
        out[lab] = dict(raw_diff_rms_mas=float(np.sqrt((d ** 2).mean())))
    # slope of tweakreg against xcorr per axis after removing per-(tile, comp) medians
    def demedian(a, keys):
        a = a.copy()
        for k in set(keys):
            ix = np.array([kk == k for kk in keys])
            a[ix] -= np.median(a[ix], axis=0)
        return a
    keys = [(r["tile"], r["comp"]) for r in rows]
    twc, xsc = demedian(tw, keys), demedian(xs, keys)
    res = twc - xsc
    mag = np.hypot(*res.T)
    out["slope_tweakreg_vs_xcorr"] = [float(np.polyfit(xsc[:, k], twc[:, k], 1)[0]) for k in (0, 1)]
    out["after_common_offset_per_tile_component"] = dict(
        rms_mas=[float(res[:, k].std()) for k in (0, 1)], median_abs_mas=float(np.median(mag)), p90_mas=float(np.percentile(mag, 90)),
        max_mas=float(mag.max()), n_over_50mas=int((mag > 50).sum()), n_over_100mas=int((mag > 100).sum()), n_over_500mas=int((mag > 500).sum()))
    resg = tw - np.median(tw - xs, axis=0) - xs
    out["after_single_global_offset"] = dict(median_offset_mas=[float(v) for v in np.median(tw - xs, axis=0)],
                                             rms_mas=[float(resg[:, k].std()) for k in (0, 1)], median_abs_mas=float(np.median(np.hypot(*resg.T))))
    out["xcorr_shift_range_mas"] = dict(max_abs=float(np.abs(xs).max()), p90_abs=float(np.percentile(np.hypot(*xs.T), 90)))
    bt = {}
    for t in sorted({r["tile"] for r in rows}):
        ix = np.array([r["tile"] == t for r in rows])
        bt[str(t)] = dict(n=int(ix.sum()), mode=rows[int(np.argmax(ix))]["mode"], rms_mas=[float(res[ix][:, k].std()) for k in (0, 1)],
                          n_over_100mas=int((mag[ix] > 100).sum()),
                          raw_median_offset_mas=[float(v) for v in np.median(tw[ix] - xs[ix], axis=0)])
    out["by_tile"] = bt
    worst = np.argsort(-mag)[:15]
    out["worst"] = [dict(root=rows[i]["root"], tile=rows[i]["tile"], resid_mas=[float(v) for v in res[i]], tw_mas=[float(v) for v in tw[i]],
                         xcorr_mas=[float(v) for v in xs[i]]) for i in worst]
    return out


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--field", required=True, choices=["SMACS", "LMC"])
    ap.add_argument("--out", default=PROTO)
    ap.add_argument("--work", default="/scratch/align_work")
    ap.add_argument("--units", default="", help="LMC tile ids (comma list); default all")
    ap.add_argument("--workers", type=int, default=4, help="parallel units (LMC tiles)")
    ap.add_argument("--filters", default="", help="restrict filters (comma list); default all in the splits file")
    ap.add_argument("--stage", choices=["all", "tweakreg"], default="all")
    ap.add_argument("--preset", choices=["ours", "default"], default="ours", help="'default' = stock tweakreg parameters, relative only (cross-check)")
    ap.add_argument("--grid", choices=["cutout", "full"], default="cutout")
    ap.add_argument("--tag", default="", help="suffix of the output directory name (cross-check runs)")
    ap.add_argument("--xcorr", default=f"{ARC}/p1/register_xcorr_v2_r3/register_xcorr.json")
    ap.add_argument("--max-files", type=int, default=0, help="debug: only the first N exposures of each unit")
    a = ap.parse_args()
    os.environ.setdefault("CRDS_SERVER_URL", "https://jwst-crds.stsci.edu")
    os.environ.setdefault("CRDS_PATH", f"{ARC}/crds_cache")
    sp = json.load(open(os.path.join(a.out, f"splits_{a.field}.json")))
    exps = sp["exposures"]
    if a.filters:
        keep = set(a.filters.split(","))
        exps = [e for e in exps if e["filter"] in keep]
    boxes = {str(b["id"]): b for b in sp["boxes"]}
    adir = os.path.join(a.out, f"align_{a.field}" + (f"_{a.tag}" if a.tag else ""))
    jobs = []
    if a.field == "SMACS":
        units = [("all", exps, boxes["0"])]
    else:
        want = set(a.units.split(",")) if a.units else set(boxes)
        units = [(f"tile{t}", [e for e in exps if t in e["boxes"]], boxes[t]) for t in sorted(boxes, key=int) if t in want]
    for name, recs, box in units:
        if a.max_files:
            recs = recs[:a.max_files]
        t = name.replace("tile", "")
        jobs.append(dict(field=a.field, unit=name, recs=recs, box=box, out=adir if a.field == "SMACS" else os.path.join(adir, name),
                         work=os.path.join(a.work, a.field + (a.tag or ""), name), preset=a.preset, stage=a.stage, grid=a.grid, ref_path=None,
                         _ref=(a.out, a.field)))
    for j in jobs:                                           # picklable path builder
        out_root, F = j.pop("_ref")
        j["ref_path"] = RefPath(out_root, F, j["unit"])
    print(f"{a.field}: {len(jobs)} unit(s), {sum(len(j['recs']) for j in jobs)} exposure slots, workers={a.workers}", flush=True)
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs)), mp_context=ctx) as ex:
        for r in ex.map(process_unit, jobs):
            print("unit done:", json.dumps(r), flush=True)
            results.append(r)
    json.dump(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), args=vars(a), results=results,
                   crds=dict(server=os.environ["CRDS_SERVER_URL"], path=os.environ["CRDS_PATH"])),
              open(os.path.join(adir, "run_summary.json"), "w"), indent=1)
    if a.field == "LMC" and a.xcorr and os.path.exists(a.xcorr):
        cc = crosscheck(adir, a.xcorr, [j["unit"].replace("tile", "") for j in jobs])
        json.dump(cc, open(os.path.join(adir, "xcorr_crosscheck.json"), "w"), indent=1)
        print("xcorr cross-check:", json.dumps({k: v for k, v in cc.items() if k not in ("description", "worst", "by_tile")}), flush=True)
    bad = [r for r in results if not r["ok"]]
    sys.exit(1 if bad else 0)


class RefPath:
    def __init__(self, root, field, unit):
        self.root, self.field, self.unit = root, field, unit

    def __call__(self, filt):
        if self.field == "SMACS":
            return os.path.join(self.root, f"ref_SMACS_{filt}_i2d.fits")
        return os.path.join(self.root, f"ref_LMC_{self.unit}_{filt}_i2d.fits")


if __name__ == "__main__":
    main()
