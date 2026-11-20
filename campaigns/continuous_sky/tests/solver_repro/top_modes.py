"""Where do the extreme eigenvectors of the preconditioned operator live? Power iteration on P^{-1} F (largest) for one reproducer
configuration and one preconditioner; reports the eigenvalue estimate, the share of the vector's F-energy per spectral coefficient, per
region (bright-source neighbourhoods, short-wavelength chip-gap cross, uncovered margin, the rest) and saves the vector.

    python tests/solver_repro/top_modes.py --precond fourier --iters 25 [feature options as run_repro.py]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from repro_field import build_model  # noqa: E402
from fieldop import uncertainty as U  # noqa: E402


def regions(model, exps, truth, o):
    sky = model.sky
    N, p = sky.N, sky.pitch
    x = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * p
    Y, X = torch.meshgrid(x, x, indexing="ij")
    bright = torch.zeros(N, N, dtype=torch.bool)
    if o["bright"] > 0:
        for (bx, by, amp, beta) in truth.pts[:4]:
            bright |= ((X - bx) ** 2 + (Y - by) ** 2) < 0.15 ** 2
    gap = torch.zeros(N, N, dtype=torch.bool)
    if o["gap"]:
        half_gap = o["gap_px"] * 0.031 / 2 + 0.04
        for e in exps:
            if e.chan == 0:
                cx, cy = e.extra["centre"]
                gap |= ((X - cx).abs() < half_gap) | ((Y - cy).abs() < half_gap)
    hd = sky.half_extent - 0.3 - 0.05
    edge = (X.abs() > hd) | (Y.abs() > hd)
    return {"bright": bright, "gap": gap & ~edge & ~bright, "edge": edge, "rest": ~(bright | gap | edge)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precond", default="fourier")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--gap-centre-filter", default="")
    ap.add_argument("--bright", type=float, default=3000.0)
    ap.add_argument("--huber", type=int, default=1)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, exps, cfg, truth, o = build_model(gap_centre_filter=a.gap_centre_filter, bright=a.bright, huber=bool(a.huber), gap=bool(a.gap))
    op = U.SkyNormalOperator(model, exps, cfg, rho=10.0, precond_mode=a.precond, log=lambda s: None, tile=64, batch_px=16000)
    g = torch.Generator().manual_seed(1)
    v = torch.randn(op.nA + op.nb, generator=g)
    v = torch.cat([v[:op.nA], op.gauge(v[op.nA:])])
    lam = []
    for it in range(a.iters):
        w = op.precond(op.apply(v))
        lam.append(float(w.norm() / v.norm()))
        v = w / w.norm()
    Fv = op.apply(v)
    e = (v[:op.nA] * Fv[:op.nA]).view(op.K, op.N, op.N)          # F-energy density of the mode (sky part)
    tot = float(e.sum())
    reg = regions(model, exps, truth, o)
    rep = {"precond": a.precond, "lambda_history": lam, "energy_sky_fraction": tot / float(v @ Fv),
           "per_coefficient": (e.sum((1, 2)) / tot).tolist(), "per_region": {k: float(e[:, m].sum() / tot) for k, m in reg.items()},
           "region_node_fraction": {k: float(m.float().mean()) for k, m in reg.items()}}
    print(json.dumps(rep, indent=1))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rep, open(os.path.join(HERE, "results", f"topmode_{a.precond}{a.tag}.json"), "w"), indent=1)
    torch.save({"v": v, "energy": e}, os.path.join(HERE, "results", f"topmode_{a.precond}{a.tag}.pt"))


if __name__ == "__main__":
    main()
