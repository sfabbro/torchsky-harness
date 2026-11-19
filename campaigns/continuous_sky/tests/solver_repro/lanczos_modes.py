"""Ritz vectors of the preconditioned operator P^{-1} F from a conjugate-gradient run (the Lanczos vectors are the preconditioned residuals
z_j / sqrt(r_j . z_j) with alternating signs): where the extreme modes live (region, spectral coefficient, background share).

    python tests/solver_repro/lanczos_modes.py --precond filter --iters 80 [--gap-centre-filter F090W] [--tag x]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from repro_field import build_model  # noqa: E402
from top_modes import regions  # noqa: E402
from fieldop import uncertainty as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precond", default="filter")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--gap-centre-filter", default="")
    ap.add_argument("--regions", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--small", type=int, default=0, help="the small field of tests/test_uncertainty_solver.py")
    ap.add_argument("--kw", default="", help="extra build_model options, JSON")
    a = ap.parse_args()
    torch.set_num_threads(1)
    kw = dict(sw_npix=36, lw_npix=18, sky_half=0.65, n_sw=2, n_lw=1, kernel_steps=40, gap_px=6, dither=0.25, Q=2) if a.small else {}
    kw.update(json.loads(a.kw) if a.kw else {})
    model, exps, cfg, truth, o = build_model(gap_centre_filter=a.gap_centre_filter, **kw)
    op = U.SkyNormalOperator(model, exps, cfg, rho=10.0, precond_mode=a.precond, log=lambda s: None, tile=64, batch_px=16000, n_regions=a.regions)
    b = op.residual_rhs()
    x = torch.zeros_like(b)
    r = b.clone()
    z = op.precond(r)
    p = z.clone()
    rz = torch.dot(r, z)
    V, al, be = [], [], []
    for j in range(a.iters):
        V.append(((-1) ** j) * z / float(rz.sqrt()))
        Ap = op.apply(p)
        alpha = rz / torch.dot(p, Ap)
        al.append(float(alpha))
        x += alpha * p
        r -= alpha * Ap
        z = op.precond(r)
        rz_new = torch.dot(r, z)
        be.append(float(rz_new / rz))
        p = z + (rz_new / rz) * p
        rz = rz_new
    m = len(al)
    T = np.zeros((m, m))
    for j in range(m):
        T[j, j] = 1.0 / al[j] + (be[j - 1] / al[j - 1] if j > 0 else 0.0)
        if j + 1 < m:
            T[j, j + 1] = T[j + 1, j] = np.sqrt(be[j]) / al[j]
    ev, S = np.linalg.eigh(T)
    Vm = torch.stack(V)                                    # (m, n)
    reg = regions(model, exps, truth, o)
    out = {"ritz": ev.tolist(), "modes": []}
    picks = list(range(4)) + list(range(m - 4, m))
    for i in picks:
        y = (torch.tensor(S[:, i], dtype=torch.float32)[:, None] * Vm).sum(0)
        Fy = op.apply(y)
        lam = float(y @ op.precond(Fy)) / float(y @ op.precond(op.apply(y)) ) if False else None
        e = (y * Fy)
        eA = e[:op.nA].view(op.K, op.N, op.N)
        tot = float(e.sum())
        sky_share = float(eA.sum()) / tot
        q = (y[:op.nA] ** 2).view(op.K, op.N, op.N)       # where the vector itself sits
        qt = float(q.sum())
        rec = {"ritz": float(ev[i]), "F_energy_sky_share": sky_share, "background_vector_share": float((y[op.nA:] ** 2).sum() / (y ** 2).sum()),
               "per_coefficient": [round(float(v), 4) for v in (q.sum((1, 2)) / qt)],
               "per_region": {k: round(float(q[:, mm].sum() / qt), 4) for k, mm in reg.items()},
               "region_label_share": {int(r_): round(float(q[:, op.region_label.to(q.device) == r_].sum() / qt), 4) for r_ in op.region_label.unique().tolist()} if hasattr(op, "region_label") else {}}
        # spatial scale: fraction of the vector's power below a quarter of the Nyquist frequency
        Fq = torch.fft.fft2(y[:op.nA].view(op.K, op.N, op.N)).abs() ** 2
        fy = torch.fft.fftfreq(op.N).abs()
        low = (fy[:, None] < 0.125) & (fy[None, :] < 0.125)
        rec["low_frequency_power_share"] = round(float(Fq[:, low].sum() / Fq.sum()), 4)
        out["modes"].append(rec)
    out["region_dbar"] = getattr(op, "region_dbar", None)
    out["region_nodes"] = {int(r_): int((op.region_label == r_).sum()) for r_ in op.region_label.unique().tolist()} if hasattr(op, "region_label") else {}
    print(json.dumps(out["modes"], indent=0))
    print("ritz lowest", ev[:6], "highest", ev[-6:])
    json.dump(out, open(os.path.join(HERE, "results", f"lanczos_{a.precond}{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
