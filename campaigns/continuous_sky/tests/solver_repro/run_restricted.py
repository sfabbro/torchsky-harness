"""Restricted solve (RestrictedOperator) on the reproducer: checks, Gauss-Newton step residual curve, one draw (19 Nov, second part).

    python tests/solver_repro/run_restricted.py --name NAME [--factor 2] [--base filter] [--precision mixed] [run_repro.py feature options]
"""
import argparse
import json
import os
import sys
import time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from repro_field import build_model  # noqa: E402
from fieldop import uncertainty as U  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--factor", type=int, default=2)
ap.add_argument("--base", default="filter")
ap.add_argument("--precision", default="mixed")
ap.add_argument("--gap-centre-filter", default="")
ap.add_argument("--maxiter", type=int, default=100)
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)
t0 = time.time()
model, exps, cfg, truth, o = build_model(gap_centre_filter=a.gap_centre_filter)
logs = []
op = U.SkyNormalOperator(model, exps, cfg, rho=10.0, precond_mode=a.base, precision=a.precision, log=logs.append, tile=64, batch_px=16000)
rec = {"name": a.name, "args": vars(a), "mu": op.mu}
rec["checks_full"] = {k: v for k, v in U.operator_checks(op, log=logs.append).items()}
ro = U.RestrictedOperator(op, factor=a.factor, log=logs.append) if a.factor > 1 else op
if a.factor > 1:
    g = torch.Generator().manual_seed(3)
    y = torch.randn(ro.nA, generator=g, dtype=torch.float64)
    xr = torch.randn(op.nA, generator=g, dtype=torch.float64)
    rec["lift_adjoint_error"] = abs(float(ro.lift_A(y).reshape(-1) @ xr - y @ ro.restrict_A(xr))) / float(y.norm() * xr.norm())
    rec["checks_restricted"] = U.operator_checks(ro, log=logs.append)
n0 = op.n_apply
x, info = U.pcg(ro, ro.residual_rhs(), tol=1e-8, maxiter=a.maxiter)
hist = info["history"]
rec["polish"] = {"history_every10": hist, "final": info["rel_residual"], "iterations": info["iterations"], "ritz": [info["ritz_min"], info["ritz_max"]],
                 "breakdown": info["breakdown"], "applications": op.n_apply - n0}
xd, infod = U.pcg(ro, ro.sample_rhs(torch.Generator().manual_seed(11)), tol=1e-8, maxiter=a.maxiter)
rec["draw"] = {"history_every10": infod["history"], "final": infod["rel_residual"], "ritz": [infod["ritz_min"], infod["ritz_max"]]}
rec["log"] = logs
rec["seconds"] = time.time() - t0
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(rec, open(os.path.join(HERE, "results", f"restricted_{a.name}.json"), "w"), indent=1, default=str)
print(a.name, "lift adjoint", rec.get("lift_adjoint_error"), "polish every 10:", [f"{h:.1e}" for h in hist], "final", f"{info['rel_residual']:.1e}",
      "ritz", rec["polish"]["ritz"], "breakdown", info["breakdown"] or "-", "| draw every 10:", [f"{h:.1e}" for h in infod["history"]], f"{rec['seconds']:.0f} s", flush=True)
