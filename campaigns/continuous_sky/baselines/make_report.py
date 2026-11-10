#!/usr/bin/env python
"""Aggregate stage3_blot task summaries (results/<FIELD>/*_summary.json) per filter. Usage: make_report.py FIELD [DIR]"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
field = sys.argv[1]
d = sys.argv[2] if len(sys.argv) > 2 else f"/home/claude/work/baselines/results/{field}"
agg = defaultdict(lambda: defaultdict(list))
for p in sorted(glob.glob(os.path.join(d, f"{field}_*_summary.json"))):
    s = json.load(open(p))
    if not s.get("ok"):
        agg[s["band"]]["failed_tasks"].append(s["tag"]); continue
    b = s["band"]
    agg[b]["tasks"].append(s["tag"])
    agg[b]["ntrain"].append(len(s["train"] if not s.get("excluded_unaligned_training") else s["train"]))
    agg[b]["nheld"].append(len(s["heldout"]))
    for r in s["tweakreg_table"]:
        if r.get("status") == "SUCCESS" and "rmse_arcsec" in r:
            agg[b]["rmse_mas"].append(1000 * r["rmse_arcsec"]); agg[b]["nmatch"].append(r["nmatches"])
        if r.get("status") not in ("SUCCESS", "REFERENCE") or r.get("rmse_arcsec", 0) > 0.25:
            agg[b]["bad_alignment"].append((s["tag"], r["rootname"], r.get("rmse_arcsec"), r["heldout"]))
    agg[b]["attempts"].append(len(s.get("tweakreg_attempts", [])))
    for h in s["heldout_results"]:
        if "star_flux_ratio_r4_median" in h:
            agg[b]["r4"].append(h["star_flux_ratio_r4_median"]); agg[b]["r8"].append(h["star_flux_ratio_r8_median"]); agg[b]["nstars"].append(h["n_stars"])
        ps = h.get("prelim_score", {})
        if ps.get("chi2_per_n_snr_gt5") is not None:
            agg[b]["chi2"].append((ps["chi2_per_n_snr_gt5"], ps["n_snr_gt5"]))
            if ps.get("frac_resid_rms_snr_gt50") is not None: agg[b]["frac"].append(ps["frac_resid_rms_snr_gt50"])
print("%-6s %5s %5s %6s %9s %9s %8s %8s %9s %9s %s" % ("band", "tasks", "held", "tries>1", "rmse_mas", "nmatch", "r4", "r8", "chi2>5pool", "chi2>5med", "fracrms>50"))
for b in sorted(agg):
    a = agg[b]
    ch = a["chi2"]
    pool = sum(c * n for c, n in ch) / sum(n for c, n in ch) if ch else float("nan")
    print("%-6s %5d %5d %6d %9.1f %9.0f %8.3f %8.3f %9.2f %9.2f %s  bad_align=%d failed=%s" % (
        b, len(a["tasks"]), sum(a["nheld"]), sum(1 for x in a["attempts"] if x > 1), np.median(a["rmse_mas"]) if a["rmse_mas"] else float("nan"),
        np.median(a["nmatch"]) if a["nmatch"] else float("nan"), np.median(a["r4"]) if a["r4"] else float("nan"), np.median(a["r8"]) if a["r8"] else float("nan"),
        pool, np.median([c for c, _ in ch]) if ch else float("nan"), "%.3f" % np.median(a["frac"]) if a["frac"] else "-", len(a["bad_alignment"]), a.get("failed_tasks", [])))
for b in sorted(agg):
    for x in agg[b]["bad_alignment"]:
        print("BAD", b, x)
