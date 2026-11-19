"""Re-run of one grid realisation of tests/truth/uncertainty_calibration.py with the current fieldop/uncertainty.py, writing into a COPY of
the results directory (argument 1), so the saved results are untouched. Usage (one thread each; the two parts can run in parallel):
    python tests/check_uncertainty_rerun.py DIR analyse --noise-seed 0 --samples 2 --part 0            # as the original part 0
    python tests/check_uncertainty_rerun.py DIR analyse --noise-seed 0 --samples 1 --part 1 --polish 1  # as the original part 1
    python tests/check_uncertainty_rerun.py DIR summarise [--form items]                                               # pull statistics from DIR
    python tests/check_uncertainty_rerun.py DIR compare [--form items] --noise-seed 0 --part 0 [--ref tests/truth/results/uncertainty]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "truth"))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402

import uncertainty_calibration as cal  # noqa: E402

d, cmd, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
cal.OUT = os.path.abspath(d)
cal.KERNEL = os.path.join(cal.OUT, "kernel_truth.pt")
if cmd == "compare":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default="grid")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--ref", default=os.path.join(HERE, "truth", "results", "uncertainty"))
    a = ap.parse_args(rest)
    f = f"post_{a.form}_n8_s{a.noise_seed}_part{a.part}.pt"
    new, old = torch.load(os.path.join(cal.OUT, f), weights_only=False), torch.load(os.path.join(a.ref, f), weights_only=False)
    out = {"file": f, "n_pix": [new["n_pix"], old["n_pix"]], "n_huber": [new["n_huber"], old["n_huber"]], "mu": [new["mu"], old["mu"]],
           "n_apply": [new["n_apply"], old["n_apply"]], "cg_iterations": [[r["iterations"] for r in new["cg"]], [r["iterations"] for r in old["cg"]]],
           "samples_max_abs_diff": float((new["samples"] - old["samples"]).abs().max()), "samples_max_abs": float(old["samples"].abs().max())}
    for k in ("A_pol",):
        if k in old:
            out[k + "_max_abs_diff"] = float((new[k] - old[k]).abs().max())
            out[k + "_max_abs"] = float(old[k].abs().max())
    print(json.dumps(out, indent=1))
else:
    sys.argv = ["uncertainty_calibration.py", cmd] + rest
    cal.main() if hasattr(cal, "main") else None
