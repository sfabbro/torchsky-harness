"""Run one configuration of the reproducer (repro_field.py) with one preconditioner and record what the diagnosis needs.

    PYTHONNOUSERSITE=1 PYTHONPATH=<torchsky/src> python tests/solver_repro/run_repro.py --name all_old --precond fourier [feature options]

Records (JSON in tests/solver_repro/results/<name>.json):
  * set-up: training pixels, Huber-downweighted pixels, mu, kappa per coefficient, applications;
  * checks of the operator as applied with the real-data settings (CUDA tiling: 64-pixel tiles, 16000-pixel batches): symmetry
    |u.Fv - v.Fu| / (|u||Fv|), repeatability (F u twice), positivity, and the same for the preconditioner;
  * the Gauss-Newton step's solve (right-hand side of the polish) and one posterior-draw solve: relative residual per iteration;
  * Lanczos estimates (Ritz values) of the preconditioned operator's spectrum, from the conjugate-gradient coefficients.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from repro_field import build_model  # noqa: E402
from fieldop import uncertainty as U  # noqa: E402


def lanczos_ritz(alphas, betas):
    """Ritz values of the preconditioned operator from conjugate-gradient coefficients (alpha_j step lengths, beta_j direction updates)."""
    m = len(alphas)
    if m == 0:
        return []
    T = np.zeros((m, m))
    for j in range(m):
        T[j, j] = 1.0 / alphas[j] + (betas[j - 1] / alphas[j - 1] if j > 0 else 0.0)
        if j + 1 < m:
            T[j, j + 1] = T[j + 1, j] = np.sqrt(max(betas[j], 0.0)) / alphas[j]
    return np.sort(np.linalg.eigvalsh(T)).tolist()


def pcg_record(op, b, maxiter, tol=1e-8, snaps=()):
    """Same algorithm as uncertainty.pcg, keeping alpha and beta (for the Ritz values) and the residual of every iteration."""
    with torch.no_grad():
        x = torch.zeros_like(b)
        r = b.clone()
        z = op.precond(r)
        p = z.clone()
        rz = torch.dot(r, z)
        bn = float(b.norm())
        hist, al, be = [], [], []
        saved = {}
        for it in range(1, maxiter + 1):
            Ap = op.apply(p)
            pAp = torch.dot(p, Ap)
            a = rz / pAp
            al.append(float(a))
            x += a * p
            r -= a * Ap
            hist.append(float(r.norm()) / bn)
            if it in snaps:
                saved[it] = x.clone()
            if hist[-1] < tol or float(pAp) <= 0:
                break
            z = op.precond(r)
            rz_new = torch.dot(r, z)
            be.append(float(rz_new / rz))
            p = z + (rz_new / rz) * p
            rz = rz_new
    if snaps:
        return x, hist, al, be, saved
    return x, hist, al, be


def product_error(op, model, snaps, x_ref, hist, s_pitch=2.0):
    """Error of the step at earlier iterations against the final iterate, per band, at resolution s_pitch sky pitches, in units of the
    tool's own sigma at that resolution (circulant_sigma: the local Fourier sigma in 'filter' mode)."""
    out = {}
    s = s_pitch * model.sky.pitch
    sm = U.Smoother(op.N, model.sky.pitch, s, dtype=torch.float64)
    cov = op.fmask.cpu()
    for f in range(model.Phi_band.shape[0]):
        phi = U.band_vector(model, f).cpu()
        sig = op.circulant_sigma(phi, s)
        rec = {}
        for it, x in snaps.items():
            dA = (x - x_ref)[:op.nA].view(op.K, op.N, op.N).cpu()
            dimg = sm(U.band_of(dA.double(), phi.double()))
            z = (dimg / sig)[cov]
            rec[it] = {"residual": hist[it - 1], "rms_over_sigma": float(z.pow(2).mean().sqrt()), "max_over_sigma": float(z.abs().max())}
        out[model.fset.names[f]] = rec
    return out


def checks(op, seed=3):
    g = torch.Generator().manual_seed(seed)
    out = {}

    def rv():
        u = torch.randn(op.nA + op.nb, generator=g)
        return torch.cat([u[:op.nA], op.gauge(u[op.nA:])])
    u, v = rv(), rv()
    Fu, Fv = op.apply(u), op.apply(v)
    Fu2 = op.apply(u)
    out["F_symmetry"] = abs(float(u @ Fv - v @ Fu)) / float(u.norm() * Fv.norm())
    out["F_repeat"] = float((Fu2 - Fu).norm() / Fu.norm())
    out["F_positive_min_uFu_over_uu"] = min(float(u @ Fu) / float(u @ u), float(v @ Fv) / float(v @ v))
    Pu, Pv = op.precond(u), op.precond(v)
    out["P_symmetry"] = abs(float(u @ Pv - v @ Pu)) / float(u.norm() * Pv.norm())
    pos = []
    for _ in range(4):
        w = rv()
        pos.append(float(w @ op.precond(w)) / float(w @ w))
    out["P_positive_min"] = min(pos)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--precond", default="fourier")
    ap.add_argument("--regions", type=int, default=8)
    ap.add_argument("--coverage", default="quadratic")
    ap.add_argument("--region-smooth", type=float, default=1.5)
    ap.add_argument("--filters", default="six")
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--gap-centre-filter", default="")
    ap.add_argument("--bright", type=float, default=3000.0)
    ap.add_argument("--floor", type=float, nargs=2, default=(0.09, 0.04))
    ap.add_argument("--huber", type=int, default=1)
    ap.add_argument("--sky-form", default="grid")
    ap.add_argument("--two-channels", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--draw", type=int, default=1)
    ap.add_argument("--checks", type=int, default=1)
    ap.add_argument("--real-tiling", type=int, default=1, help="tiles of 64 pixels, 16000-pixel batches (the CUDA defaults of the real runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu", type=float, default=0.0, help="prior weight forced (0 = from the data); to compare preconditioners on the same F")
    ap.add_argument("--mu-from", default="", help="take mu from this result JSON (name in results/)")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.perf_counter()
    floor = tuple(a.floor) if max(a.floor) > 0 else None
    model, exps, cfg, truth, o = build_model(filters=a.filters, gap=bool(a.gap), gap_centre_filter=a.gap_centre_filter, bright=a.bright, floor=floor,
                                             huber=bool(a.huber), sky_form=a.sky_form, two_channels=bool(a.two_channels), seed=a.seed)
    logs = []
    kw = dict(tile=64, batch_px=16000) if a.real_tiling else {}
    if a.mu_from:
        a.mu = json.load(open(os.path.join(HERE, "results", a.mu_from + ".json")))["mu"]
    if a.mu > 0:
        kw["mu"] = a.mu
    op = U.SkyNormalOperator(model, exps, cfg, rho=10.0, precond_mode=a.precond, log=logs.append, n_regions=a.regions, region_smooth=a.region_smooth, coverage=a.coverage, **kw)
    rec = {"name": a.name, "args": vars(a), "setup_log": logs, "n_pix": op.n_pix, "n_huber": op.n_huber, "nA": op.nA, "mu": op.mu,
           "kappa": op.kappa, "setup_applications": op.n_apply, "seconds_build": time.perf_counter() - t0}
    if a.checks:
        rec["checks"] = checks(op)
    n0 = op.n_apply
    b = op.residual_rhs()
    x, hist, al, be, snaps = pcg_record(op, b, a.maxiter, snaps=(20, 30, 40, 60, 80))
    rec["polish"] = {"history": hist, "ritz": lanczos_ritz(al, be), "applications": op.n_apply - n0}
    if a.precond == "filter" and hist[-1] < 1e-3:
        rec["polish"]["product_error_vs_final"] = product_error(op, model, snaps, x, hist)
    if a.draw:
        b = op.sample_rhs(torch.Generator().manual_seed(11))
        x, hist, al, be = pcg_record(op, b, a.maxiter)
        rec["draw"] = {"history": hist, "ritz": lanczos_ritz(al, be)}
    rec["seconds_per_apply"] = op.t_apply / op.n_apply
    rec["seconds_total"] = time.perf_counter() - t0
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rec, open(os.path.join(HERE, "results", f"{a.name}.json"), "w"), indent=1)
    h = rec["polish"]["history"]
    print(a.name, "kappa", np.round(op.kappa, 5).tolist(), "checks", rec.get("checks"),
          "polish residual at 10/30/60/100:", [f"{h[i - 1]:.3g}" for i in (10, 30, 60, 100) if i <= len(h)],
          "ritz min/max", f"{min(rec['polish']['ritz']):.3g} {max(rec['polish']['ritz']):.3g}",
          ("draw at 60/100: " + str([f"{rec['draw']['history'][i - 1]:.3g}" for i in (60, 100) if i <= len(rec['draw']['history'])])) if a.draw else "",
          f"{rec['seconds_total']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
