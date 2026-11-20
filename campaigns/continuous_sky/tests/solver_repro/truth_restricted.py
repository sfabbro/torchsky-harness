"""The restricted formulation (RestrictedOperator) on the known-truth problem of UNCERTAINTY_2026-11-18.md (grid sky, 8 exposures, saved fits).

Per noise realisation: the restricted Gauss-Newton step (band image on a grid of twice the sky pitch, 0.10 arcsec) and restricted posterior draws;
pulls of the product against the truth at resolution s, as in tests/truth/uncertainty_calibration.py ("total" = with the fitted background's
offset from the true one, which the gauge holds; inner region: |x|, |y| < 1.5 arcsec, also with the best quadratic removed; sources by S/N).
sigma = the restricted posterior's, from the pooled draws, locally averaged (0.25 arcsec), as in the calibration.

    python tests/solver_repro/truth_restricted.py run --noise-seed 0 [--draws 3] [--factor 2]      # one realisation -> results/truth_restricted_s0.pt
    python tests/solver_repro/truth_restricted.py summarise                                        # pulls from every saved realisation
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tests", "truth"))
sys.path.insert(0, ROOT)
import uncertainty_calibration as cal  # noqa: E402
import truth_problem as tp  # noqa: E402
from fieldop import uncertainty as U  # noqa: E402

torch.set_num_threads(1)
RES = os.path.join(HERE, "results")


def run(a):
    model, cfg, exps, st = cal.load_model("grid", 8, a.noise_seed, log=lambda s: None)
    logs = []
    op = U.SkyNormalOperator(model, exps, cfg, rho=10.0, precond_mode="fourier", precision="mixed", log=logs.append)
    ro = U.RestrictedOperator(op, factor=a.factor, log=logs.append)
    chk = U.operator_checks(ro, log=logs.append, label="restricted")
    A_pol, db, info = U.gauss_newton_polish(ro, tol=1e-5, maxiter=150)
    print(f"step: {info['iterations']} iterations, relative residual {info['rel_residual']:.2e}, Lanczos {info['ritz_min']:.3g} to {info['ritz_max']:.3g}, "
          f"breakdown {info['breakdown'] or '-'}; {op.t_apply / op.n_apply:.2f} s per application", flush=True)
    samp, recs = U.posterior_samples(ro, a.draws, seed=500 + a.noise_seed, tol=1e-5, maxiter=150, log=lambda s_: print(s_, flush=True))
    circ = {f"{sc:.3f}": ro.circulant_sigma(U.band_vector(model, 0).cpu(), sc).float() for sc in (0.111, 0.15, 0.2)}
    out = {"circulant": circ, "A_pol": A_pol.cpu(), "samples": samp, "polish_iterations": info["iterations"], "polish_residual": info["rel_residual"],
           "draw_iterations": [r["iterations"] for r in recs], "draw_residuals": [r["rel_residual"] for r in recs], "checks": chk,
           "bkg_train_mean": st["bkg_train_mean"], "phi": U.band_vector(model, 0).cpu(), "pitch": model.sky.pitch, "mu": op.mu, "log": logs,
           "factor": a.factor, "seconds_per_apply": op.t_apply / op.n_apply, "n_apply": op.n_apply}
    torch.save(out, os.path.join(RES, f"truth_restricted_s{a.noise_seed}.pt"))
    print(json.dumps({k: v for k, v in out.items() if k in ("polish_iterations", "polish_residual", "draw_iterations", "draw_residuals", "n_apply")}))


def summarise(a):
    D = [torch.load(p, weights_only=False) for p in sorted(glob.glob(os.path.join(RES, "truth_restricted_s*.pt")))]
    pop = tp.make_population()
    fl = tp.flux_for_snr(pop, n_ref=8)
    pf = [(p[0], p[1], p[2], float(F)) for p, F in zip(pop, fl)]
    emp = tp.empty_positions(pop)
    snr = np.array([p[3] for p in pop])
    sx, sy = np.array([p[0] for p in pop]), np.array([p[1] for p in pop])
    pitch = D[0]["pitch"]
    N = D[0]["A_pol"].shape[-1]
    phi = D[0]["phi"].double()
    xn = (np.arange(N) - (N - 1) / 2) * pitch
    inner = torch.tensor((np.abs(xn)[None, :] < cal.INNER) & (np.abs(xn)[:, None] < cal.INNER))
    pooled = torch.cat([d["samples"].double() for d in D])
    out = {"realisations": len(D), "pooled_draws": int(pooled.shape[0]), "polish_iterations": [d["polish_iterations"] for d in D],
           "polish_residual": [d["polish_residual"] for d in D], "draw_iterations": [d["draw_iterations"] for d in D], "scales": {}}
    for s in (0.111, 0.15, 0.2):
        sm = U.Smoother(N, pitch, s, dtype=torch.float64)
        var = sm(U.band_of(pooled, phi)).pow(2).mean(0)
        varS = cal.smooth_var(var, inner, pitch)

        def sig_at(xs, ys):
            ix = np.clip(np.round(xs / pitch + (N - 1) / 2).astype(int), 0, N - 1)
            iy = np.clip(np.round(ys / pitch + (N - 1) / 2).astype(int), 0, N - 1)
            return varS[iy, ix].sqrt().numpy()
        Mx_e, My_e = U.tent_gauss_1d(torch.tensor(emp[:, 0]), torch.tensor(xn), pitch, s), U.tent_gauss_1d(torch.tensor(emp[:, 1]), torch.tensor(xn), pitch, s)
        Mx_s, My_s = U.tent_gauss_1d(torch.tensor(sx), torch.tensor(xn), pitch, s), U.tent_gauss_1d(torch.tensor(sy), torch.tensor(xn), pitch, s)
        pv = lambda img, Mx, My: torch.einsum("pi,ij,pj->p", My, img, Mx).numpy()  # noqa: E731
        se, ss = sig_at(emp[:, 0], emp[:, 1]), sig_at(sx, sy)
        te, ts = cal.truth_product(pf, emp[:, 0], emp[:, 1], s), cal.truth_product(pf, sx, sy, s)
        tg = torch.tensor(cal.truth_product(pf, *[g.reshape(-1) for g in np.meshgrid(xn, xn)], s).reshape(N, N))
        sgS = varS.sqrt()
        Ze, Zs, Zi, Zq = [], [], [], []
        X, Y = np.meshgrid(xn, xn)
        Bq = np.stack([np.ones_like(X), X, Y, X * X, X * Y, Y * Y], -1)[inner.numpy()]
        for d in D:
            band = U.band_of(d["A_pol"].double(), phi)
            db = d["bkg_train_mean"] - tp.BKG
            Ze.append((pv(band, Mx_e, My_e) + db - te) / se)
            Zs.append((pv(band, Mx_s, My_s) + db - ts) / ss)
            err = sm(band) + db - tg
            Zi.append((err / sgS)[inner].numpy())
            e_in = err[inner].numpy()
            Zq.append((e_in - Bq @ np.linalg.lstsq(Bq, e_in, rcond=None)[0]) / sgS[inner].numpy())
        circ = [d["circulant"][f"{s:.3f}"] for d in D if "circulant" in d]
        rec = {"sigma_inner_median": float(sgS[inner].median()),
               "circulant_over_draws_inner_median": [float((c_.double() / sgS)[inner].median()) for c_ in circ], "empty_total": cal.stats(Ze), "inner_total": cal.stats(Zi),
               "inner_quadratic_removed": cal.stats(Zq), "sources": []}
        Zs = np.array(Zs)
        for lo, hi in tp.SN_BINS:
            sel = (snr >= lo) & (snr < hi) if hi < 100 else (snr >= lo)
            if sel.any():
                rec["sources"].append({"bin": [lo, hi], **cal.stats(Zs[:, sel])})
        out["scales"][f"{s:.3f}"] = rec
        f = lambda r: f"{r['mean']:+.2f} rms {r['rms']:.2f} +- {r['rms_err']:.2f}"  # noqa: E731
        print(f"s = {s}: sigma inner median {rec['sigma_inner_median']:.4g} (cheap / draws {rec['circulant_over_draws_inner_median']}); empty total {f(rec['empty_total'])}; inner total {f(rec['inner_total'])}; "
              f"inner, quadratic removed {f(rec['inner_quadratic_removed'])}; sources " + "; ".join(f"S/N {r['bin']}: {r['mean']:+.2f} rms {r['rms']:.2f}" for r in rec["sources"]))
    json.dump(out, open(os.path.join(RES, "truth_restricted_summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("realisations", "pooled_draws", "polish_iterations", "polish_residual", "draw_iterations")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarise"])
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--factor", type=int, default=2)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summarise(a)
