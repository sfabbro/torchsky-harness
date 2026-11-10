#!/usr/bin/env python
"""Download stage3_blot task summaries from CANFAR (results/<FIELD>/) and print one line per task. Usage: fetch_summaries.py FIELD [root]"""
import json, os, re, subprocess, sys
import numpy as np
field = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else "baselines/stage3_blot"
base = "https://ws-uv.canfar.net/arc"
cert = os.path.expanduser("~/.ssl/cadcproxy.pem")
def get(url, out=None):
    r = subprocess.run(["curl", "-s", "--cert", cert, url], capture_output=True)
    return r.stdout
xml = get(f"{base}/nodes/projects/ots/adass2026/{root}/{field}/dither").decode()
names = sorted(set(re.findall(r'uri="[^"]*/([^"/]+_summary\.json)"', xml)))
out = f"/home/claude/work/baselines/results/{field}"
os.makedirs(out, exist_ok=True)
for n in names:
    b = get(f"{base}/files/projects/ots/adass2026/{root}/{field}/dither/{n}")
    open(f"{out}/{n}", "wb").write(b)
    try:
        s = json.loads(b)
    except Exception:
        print(n, "unreadable"); continue
    hr = s.get("heldout_results", [])
    r4 = [h["star_flux_ratio_r4_median"] for h in hr if "star_flux_ratio_r4_median" in h]
    c2 = [h["prelim_score"]["chi2_per_n_snr_gt5"] for h in hr if h.get("prelim_score", {}).get("chi2_per_n_snr_gt5")]
    print(f"{n[:-13]:14s} ok={s.get('ok')} t={s.get('seconds')} ntrain={len(s['train'])} nheld={len(s['heldout'])} "
          f"tw: succ={s.get('tweakreg_n_success')} bad={s.get('tweakreg_n_failed')} attempts={len(s.get('tweakreg_attempts', []))} "
          f"r4med={np.median(r4) if r4 else None} chi2med={np.median(c2) if c2 else None} {('ERR ' + s['error'][:80]) if s.get('error') else ''}")
