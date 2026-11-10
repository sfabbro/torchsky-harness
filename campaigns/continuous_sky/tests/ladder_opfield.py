"""Difficulty ladder L0..L6 on the synthetic generator of synth_opfield.py, field vs factorised operator (diagnostic; see docs in the report).

L0 one instrument (sharp), one roll, dithers only, no shifts / distortion / saturation / CRs;  L1 + 3 instruments;  L2 + 3 rolls;
L3 + shifts (coarse 0.6 arcsec);  L4 + distortion;  L5 + soft saturation;  L6 + CRs (= the full case of synth_opfield.py).
Usage: venvT/bin/python tests/ladder_opfield.py --levels 0,1,2 --operators field,factorised --steps 1800 --tag name [--extra "--mu-skip"]
Appends one JSON line per run to <out>/ladder_results.jsonl.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth_opfield as S  # noqa: E402

O = S.O
LEVELS = {
    0: dict(insts=["sharp"], rolls=[0.0], shifts=False, dist=False, sat=False, cr=False),
    1: dict(insts=["sharp", "coarse", "mid"], rolls=[0.0], shifts=False, dist=False, sat=False, cr=False),
    2: dict(insts=["sharp", "coarse", "mid"], rolls=None, shifts=False, dist=False, sat=False, cr=False),
    3: dict(insts=["sharp", "coarse", "mid"], rolls=None, shifts=True, dist=False, sat=False, cr=False),
    4: dict(insts=["sharp", "coarse", "mid"], rolls=None, shifts=True, dist=True, sat=False, cr=False),
    5: dict(insts=["sharp", "coarse", "mid"], rolls=None, shifts=True, dist=True, sat=True, cr=False),
    6: dict(insts=["sharp", "coarse", "mid"], rolls=None, shifts=True, dist=True, sat=True, cr=True),
}
FULL_INSTR = dict(S.INSTR)


def build(level, seed=0):
    cfg = dict(LEVELS[level])
    S.INSTR.clear()
    S.INSTR.update({k: FULL_INSTR[k] for k in cfg.pop("insts")})
    return S.build_world(seed, False, cfg)


def one(world, level, op, steps, c2f_frac, out, extra, c2f_start=None):
    trace = []
    orig = O.log

    def cap(msg):
        m = re.search(r"step (\d+) .*? 2NLL ([\d.]+)", msg)
        if m:
            trace.append((int(m.group(1)), float(m.group(2))))
        orig(msg)
    O.log = cap
    ex = ["--operator", op, "--nuis-steps", "200"] + list(extra)
    if c2f_start is not None:
        ex += ["--c2f-start-arcsec", str(c2f_start)]
    try:
        ctx, args = S.run_variant(world, out, steps, c2f_frac, f"L{level}{op}", False, 30.0, ex)
    finally:
        O.log = orig
    R = S.analyse(ctx, world, op)
    agg = R["heldout_agg"]
    rows = R["registration"]["rows"]
    t = [agg[k] for k in ("T1/all", "T2/all") if k in agg]
    res = dict(level=level, operator=op, extra=list(extra), c2f_frac=c2f_frac, c2f_start=c2f_start, steps=steps, seconds=ctx["seconds"],
               chi2_all=float(np.mean([x["chi2_all"] for x in t])), oracle_all=float(np.mean([x["oracle_all"] for x in t])),
               chi2_bright=float(np.nanmean([x["chi2_bright"] for x in t])), oracle_bright=float(np.nanmean([x["oracle_bright"] for x in t])),
               chi2_noCR=float(np.mean([x["chi2_noCR"] for x in t])),
               reg_train={k.split("/")[0]: round(v["rms_resid_mas"], 1) for k, v in rows.items() if k.endswith("/train") and not k.startswith("all")},
               reg_heldout={k.split("/")[0]: round(v["rms_resid_mas"], 1) for k, v in rows.items() if k.endswith("/heldout") and not k.startswith("all")},
               reg_true={k.split("/")[0]: round(v["rms_true_mas"], 1) for k, v in rows.items() if k.endswith("/train") and not k.startswith("all")},
               relk=R["kernels"]["relative_vs_sharp"], common_mode=R["kernels"]["common_mode"],
               amp={k: round(v["relative_to_sharp"], 3) for k, v in R["amplitude"].items()})
    if len(trace) >= 4:
        last = trace[-1]
        prev = [v for s_, v in trace if s_ <= 0.9 * last[0]][-1]
        res["loss_last"] = last[1]
        res["loss_improve_last10pct"] = (prev - last[1]) / abs(last[1]) * 100
    res["trace"] = trace
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="0,1,2,3,4,5,6")
    ap.add_argument("--operators", default="field,factorised")
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--c2f-frac", type=float, default=0.3)
    ap.add_argument("--c2f-start", type=float, default=None)
    ap.add_argument("--extra", default="")
    ap.add_argument("--tag", default="ladder")
    ap.add_argument("--out", default=os.path.join(S.ROOT, "runs", "opfield_synth"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for lv in [int(x) for x in a.levels.split(",")]:
        world = build(lv)
        for op in a.operators.split(","):
            t0 = time.time()
            res = one(world, lv, op, a.steps, a.c2f_frac, os.path.join(a.out, f"{a.tag}_L{lv}_{op}"), a.extra.split(), a.c2f_start)
            res["tag"] = a.tag
            with open(os.path.join(a.out, "ladder_results.jsonl"), "a") as f:
                f.write(json.dumps(res, default=float) + "\n")
            print(f"LADDER {a.tag} L{lv} {op} {time.time() - t0:.0f}s chi2 all {res['chi2_all']:.1f}/{res['oracle_all']:.1f} bright {res['chi2_bright']:.1f}/{res['oracle_bright']:.1f} "
                  f"reg_ho {res['reg_heldout']} (true {res['reg_true']}) relk {res['relk']} d10% {res.get('loss_improve_last10pct', float('nan')):.2f}", flush=True)


if __name__ == "__main__":
    main()
