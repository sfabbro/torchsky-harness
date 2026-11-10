"""Ladder on the time synthetic (synth_time.py): flat z vs z-hier vs restarts.  Appends JSON lines to runs/opfield_synth/time_results.jsonl.
Usage: python tests/ladder_time.py --tag a_flat --extra="--xfit-reg 2 --mu-skip --mu-skip-scale 8 --lr-mu-skip 3e-2" [--wide]"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth_time as ST  # noqa: E402

S, O = ST.S, ST.O
BASE = "--xfit-reg 2 --mu-skip --mu-skip-scale 8 --lr-mu-skip 3e-2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--extra", default=BASE)
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--out", default=os.path.join(S.ROOT, "runs", "opfield_synth"))
    a = ap.parse_args()
    if a.wide:
        S.NPIX = 160
        S.G = S.NPIX + 2 * S.HALO
    world = ST.build_world_time(0, a.wide)
    trace = []
    orig = O.log

    def cap(msg):
        m = re.search(r"step (\d+) .*? 2NLL ([\d.]+)", msg)
        if m:
            trace.append((int(m.group(1)), float(m.group(2))))
        orig(msg)
    O.log = cap
    out = os.path.join(a.out, f"time_{a.tag}")
    t0 = time.time()
    try:
        ctx, args = S.run_variant(world, out, a.steps, 0.3, a.tag, False, 40.0, ["--nuis-steps", "200"] + a.extra.split())
    finally:
        O.log = orig
    R = S.analyse(ctx, world, a.tag)
    agg, rows = R["heldout_agg"], R["registration"]["rows"]
    res = dict(tag=a.tag, extra=a.extra, wide=a.wide, seconds=time.time() - t0, steps=a.steps)
    for sp in ("T1", "T3", "val"):
        if f"{sp}/all" in agg:
            g = agg[f"{sp}/all"]
            res[sp] = dict(chi2_all=g["chi2_all"], oracle_all=g["oracle_all"], chi2_bright=g["chi2_bright"], oracle_bright=g["oracle_bright"],
                           reg={k.split("/")[0]: round(v["rms_resid_mas"], 1) for k, v in rows.items() if k.endswith("/@" + sp) and not k.startswith("all")},
                           reg_true={k.split("/")[0]: round(v["rms_true_mas"], 1) for k, v in rows.items() if k.endswith("/@" + sp) and not k.startswith("all")})
    res["reg_train"] = {k.split("/")[0]: round(v["rms_resid_mas"], 1) for k, v in rows.items() if k.endswith("/@train") and not k.startswith("all")}
    res["relk"] = R["kernels"]["relative_vs_sharp"]
    res["stars"] = ST.moving_star_stats(ctx, world)
    if len(trace) >= 4:
        last = trace[-1]
        prev = [v for s_, v in trace if s_ <= 0.9 * last[0]][-1]
        res["loss_last"], res["loss_improve_last10pct"] = last[1], (prev - last[1]) / abs(last[1]) * 100
    with open(os.path.join(a.out, "time_results.jsonl"), "a") as f:
        f.write(json.dumps(res, default=float) + "\n")
    print("TIME", json.dumps(res, default=float)[:1500], flush=True)


if __name__ == "__main__":
    main()
