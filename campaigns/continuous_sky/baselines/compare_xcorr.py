#!/usr/bin/env python
"""Compare the WCS shifts applied by tweakreg (stage3_blot LMC task summaries) with our cross-correlation shifts
(register_xcorr.json), per task up to a common offset (median of the difference), in the tile frame (x west, y north, mas).
Usage: compare_xcorr.py RESULTS_DIR XCORR_JSON [OUT_JSON]"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
res_dir, xj = sys.argv[1:3]
x = json.load(open(xj))["exposures"]
out, per_band, all_res = [], defaultdict(list), []
for p in sorted(glob.glob(os.path.join(res_dir, "LMC_t*_summary.json"))):
    s = json.load(open(p))
    if "tweakreg_table" not in s:
        continue
    A, B, names = [], [], []
    for r in s["tweakreg_table"]:
        e = x.get(r["file"])
        if e is None or r.get("status") not in ("SUCCESS", "REFERENCE") or not np.isfinite(r.get("tile_dx_west_mas", np.nan)):
            continue
        A.append([r["tile_dx_west_mas"], r["tile_dy_north_mas"]]); B.append(e["shift_mas"]); names.append(r["file"])
    if len(A) < 3:
        continue
    A, B = np.array(A), np.array(B)
    d = A - B
    c = np.median(d, 0)
    rr = np.hypot(*(d - c).T)
    rms = [r["rmse_arcsec"] for r in s["tweakreg_table"] if "rmse_arcsec" in r]
    rec = dict(tag=s["tag"], band=s["band"], tile=s.get("tile"), n=len(A), common_offset_mas=[float(c[0]), float(c[1])],
               resid_median=float(np.median(rr)), resid_p90=float(np.percentile(rr, 90)), resid_max=float(rr.max()),
               worst=names[int(rr.argmax())], n_not_aligned=len(s["tweakreg_table"]) - len(A),
               rmse_arcsec_median=float(np.median(rms)) if rms else None,
               attempts=len(s.get("tweakreg_attempts", [])), params=s.get("tweakreg_params"))
    out.append(rec)
    per_band[s["band"]].extend(rr.tolist()); all_res.extend(rr.tolist())
    print("%-14s n=%3d resid med/p90/max = %5.1f %5.1f %6.1f mas  offset=(%7.0f,%7.0f)  rmse_med=%s  tries=%d  not_aligned=%d" % (
        s["tag"], len(A), rec["resid_median"], rec["resid_p90"], rec["resid_max"], c[0], c[1],
        None if rec["rmse_arcsec_median"] is None else round(rec["rmse_arcsec_median"] * 1000), rec["attempts"], rec["n_not_aligned"]))
print()
for b in sorted(per_band):
    v = np.array(per_band[b]); print("%s  n=%d  resid median %.1f  p90 %.1f  max %.1f mas" % (b, len(v), np.median(v), np.percentile(v, 90), v.max()))
v = np.array(all_res)
if len(v):
    print("ALL  n=%d  resid median %.1f  p90 %.1f  max %.1f mas" % (len(v), np.median(v), np.percentile(v, 90), v.max()))
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=1)
