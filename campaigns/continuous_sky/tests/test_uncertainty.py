"""Tests of fieldop/uncertainty.py (UNCERTAINTY_2026-11-18.md). CPU, about 30 s. No convergence test of a fit.

1. product: smoothing weights (partition of unity, tent at s = 0, Gaussian width), supported resolution on a known sigma(s)
2. estimators on a tiny dense linear problem (sky coefficients + per-exposure backgrounds with the pedestal gauge): conjugate-gradient
   exact variance = dense covariance; posterior draws reproduce it; the Gauss-Newton step = the dense least-squares solution
3. the model's own operator on a tiny synthetic field: symmetric, matches sum w (J v)^2 rendered directly, background block exact
4. half-difference and seed-difference tool on two synthetic states with known noise
5. --train-half option and alternate_groups; FITS product round trip
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fieldop import uncertainty as U  # noqa: E402
from fieldop.cli import default_config, make_sky, parse_config  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402


# ----------------------------------------------------------------------------- 1. product
def test_smoother_weights():
    N, p = 41, 0.05
    for s in (0.0, 0.03, 0.1):
        M = U.Smoother(N, p, s, dtype=torch.float64).M
        inner = M[15:26]
        assert torch.allclose(inner.sum(1), torch.ones(11, dtype=torch.float64), atol=1e-6), s       # constant sky unchanged (away from the edges)
    assert torch.allclose(U.Smoother(N, p, 0.0, dtype=torch.float64).M, torch.eye(N, dtype=torch.float64))   # s = 0: the sky at the nodes
    # second moment of (tent * g_s) = s^2 + p^2 / 6
    x = torch.linspace(-1, 1, 4001, dtype=torch.float64)
    k = U.tent_gauss_1d(x, torch.zeros(1, dtype=torch.float64), p, 0.1)[:, 0]
    dx = float(x[1] - x[0])
    assert abs(float((k * dx).sum()) - p) < 1e-6 * p * 1e3            # integral over x of the node weight = p (one node's share)
    var = float((k * x * x).sum() / k.sum())
    assert abs(var / (0.1 ** 2 + p * p / 6) - 1) < 1e-4


def test_supported_resolution_gaussian():
    p = 0.08
    scales = [f * 0.025 for f in (0, 4, 5, 6, 7, 8, 10, 12, 16)]
    sig = torch.tensor([1e3] + [1.0 / math.sqrt(max(s * s - p * p, 1e-6)) for s in scales[1:]], dtype=torch.float64)[:, None, None].expand(-1, 2, 2)
    s_star, alpha = U.supported_resolution(sig, scales, alpha_max=2.0)
    assert abs(float(s_star[0, 0]) / (math.sqrt(2) * p) - 1) < 0.05, float(s_star[0, 0])


# ----------------------------------------------------------------------------- 2. dense linear problem
class DenseOp:
    """Same interface and algebra as SkyNormalOperator, with an explicit Jacobian: x = [A (K N N), b (n_exp x 6)], gauge on b_e0."""

    def __init__(self, K=2, N=6, n_exp=3, n_pix=60, mu=0.05, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.K, self.N, self.nA, self.nb, self.mu = K, N, K * N * N, 6 * n_exp, mu
        self.JA = [torch.randn(n_pix, self.nA, generator=g, dtype=torch.float64) * 0.3 for _ in range(n_exp)]
        xi = torch.rand(n_pix, 2, generator=g, dtype=torch.float64) * 2 - 1
        self.L = U.legendre6(xi)
        self.w = [torch.rand(n_pix, generator=g, dtype=torch.float64) + 0.5 for _ in range(n_exp)]
        self.n_exp = n_exp
        self.A_fit = torch.randn(K, N, N, generator=g, dtype=torch.float64)
        self.r = [torch.randn(n_pix, generator=g, dtype=torch.float64) for _ in range(n_exp)]
        self.bkg_gauge = "constant"
        self.gauge_groups = [torch.arange(n_exp)]
        self.dense = self._dense()

    gauge = U.SkyNormalOperator.gauge

    def _J(self):
        rows = []
        for e in range(self.n_exp):
            B = torch.zeros(self.JA[e].shape[0], self.nb, dtype=torch.float64)
            B[:, 6 * e:6 * e + 6] = self.L
            rows.append(torch.cat([self.JA[e], B], 1))
        return torch.cat(rows)

    def _dense(self):
        J = self._J()
        W = torch.cat(self.w)
        F = J.T @ (W[:, None] * J)
        F[:self.nA, :self.nA] += self.mu * torch.eye(self.nA, dtype=torch.float64)
        return F

    def apply(self, x):
        x = torch.cat([x[:self.nA], self.gauge(x[self.nA:])])
        y = self.dense @ x
        return torch.cat([y[:self.nA], self.gauge(y[self.nA:])])

    def precond(self, r):
        d = torch.diag(self.dense)
        return torch.cat([r[:self.nA] / d[:self.nA], self.gauge(r[self.nA:] / d[self.nA:])])

    def sample_rhs(self, gen):
        J = self._J()
        z = torch.randn(J.shape[0], generator=gen).double()
        y = J.T @ (torch.cat(self.w).sqrt() * z)
        zA = torch.randn(self.nA, generator=gen).double()
        return torch.cat([y[:self.nA] + math.sqrt(self.mu) * zA, self.gauge(y[self.nA:])])

    def unit_rhs(self, kA):
        return torch.cat([kA.reshape(-1).double(), torch.zeros(self.nb, dtype=torch.float64)])

    def residual_rhs(self):
        y = self._J().T @ (torch.cat(self.w) * torch.cat(self.r))
        return torch.cat([y[:self.nA] - self.mu * self.A_fit.reshape(-1), self.gauge(y[self.nA:])])

    def covariance(self):
        """Dense posterior covariance in the gauge subspace: P (P F P)^+ P, P the projector of the gauge."""
        n = self.nA + self.nb
        P = torch.stack([torch.cat([e[:self.nA], self.gauge(e[self.nA:])]) for e in torch.eye(n, dtype=torch.float64)], 1)
        return P @ torch.linalg.pinv(P @ self.dense @ P, hermitian=True, rtol=1e-10) @ P


def test_dense_exact_variance_and_draws():
    op = DenseOp()
    C = op.covariance()
    phi = torch.tensor([0.7, 0.3], dtype=torch.float64)
    sm = U.Smoother(op.N, 0.05, 0.04, dtype=torch.float64)
    pts = [(2, 3), (4, 1)]
    ex = U.exact_variance(op, phi, sm, pts, tol=1e-12, maxiter=400)
    for (ix, iy), (sg, _) in zip(pts, ex):
        k = (phi[:, None, None] * sm.weights_at(ix, iy)[None]).reshape(-1)
        assert abs(sg ** 2 / float(k @ C[:op.nA, :op.nA] @ k) - 1) < 1e-6
    samp, recs = U.posterior_samples(op, 400, seed=3, tol=1e-12, maxiter=400, log=lambda s: None)
    S = samp.reshape(400, -1).double()
    emp = S.T @ S / 400
    d_true, d_emp = torch.diag(C[:op.nA, :op.nA]), torch.diag(emp)
    assert float(((d_emp / d_true) - 1).abs().max()) < 0.35            # each of 72 variances within about 4.9 sigma of a 400-draw estimate
    assert abs(float(d_emp.mean() / d_true.mean()) - 1) < 0.05
    # the band product variance at one pixel from the draws
    k = (phi[:, None, None] * sm.weights_at(2, 3)[None]).reshape(-1)
    v_emp = float(((S @ k) ** 2).mean())
    assert abs(v_emp / float(k @ C[:op.nA, :op.nA] @ k) - 1) < 0.25


def test_dense_gauss_newton_step():
    op = DenseOp(seed=1)
    A_pol, db, info = U.gauss_newton_polish(op, tol=1e-12, maxiter=400)
    x_dense = op.covariance() @ op.residual_rhs()
    assert torch.allclose((A_pol - op.A_fit).reshape(-1), x_dense[:op.nA], atol=1e-6)
    assert abs(float(db[:, 0].sum())) < 1e-8                            # the step stays in the gauge


# ----------------------------------------------------------------------------- 3. the model's operator on a tiny field
@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    insts = [SynthInstrument("A", 0.10, 16, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3)])]
    exps, fset, meta, truth, _ = make_field(seed=1, n_per_filter=3, n_val=0, n_held=1, insts=insts, centre_range=0.2, rot_choices=(0,), patch_px=8,
                                            truth_half=0.6)
    cfg = default_config(legacy_defaults=False, sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.6, region_px=8, n_lambda=3, k_hidden=16,
                         spec_k=3, pattern_grid=0, stripes=0, steps=0, budget_min=1e6, tau_mode="fixed", seed=0, held_steps=0)
    sky, _ = make_sky(exps, fset, cfg, 1.0, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    op = U.SkyNormalOperator(model, exps, cfg, rho=10.0, log=lambda s: None)
    return model, exps, cfg, op


def test_operator_symmetric_and_matches_render(tiny):
    model, exps, cfg, op = tiny
    g = torch.Generator().manual_seed(5)
    u = torch.randn(op.nA + op.nb, generator=g)
    v = torch.randn(op.nA + op.nb, generator=g)
    u = torch.cat([u[:op.nA], op.gauge(u[op.nA:])])
    v = torch.cat([v[:op.nA], op.gauge(v[op.nA:])])
    Fu, Fv = op.apply(u), op.apply(v)
    assert abs(float(u @ Fv - v @ Fu)) < 1e-3 * float(u.abs() @ Fv.abs())
    # v^T F v = sum w (J_A vA + L vb)^2 + mu |vA|^2, the model rendered directly with the sky coefficients replaced by vA
    vA = v[:op.nA].view(op.K, op.N, op.N)
    vb = v[op.nA:].view(-1, 6)
    q = 0.0
    with torch.no_grad(), U.sky_override(model.sky, vA):
        outs = model.render_batch([(t["eid"], t["pix"]) for t in op.tiles], op.P)
    for t, o in zip(op.tiles, outs):
        q += float((t["w"] * (o["S"] + t["Lb"] @ vb[t["row"]]) ** 2).sum())
    q += op.mu * float(vA.pow(2).sum())
    assert abs(float(v @ Fv) / q - 1) < 1e-3
    # circulant approximation runs and is finite where covered
    sc = op.circulant_sigma(U.band_vector(model, 0).cpu(), 0.12)
    assert torch.isfinite(sc[op.fmask.cpu()]).all()


def test_operator_draw_and_product(tiny):
    model, exps, cfg, op = tiny
    res = U.uncertainty_map(model, exps, cfg, 0, n_samples=2, maxiter=15, tol=1e-6, log=lambda s: None, op=op)
    assert res["image_fit"].shape == res["image"].shape and res["polish"]["iterations"] >= 1
    res_c = U.uncertainty_map(model, exps, cfg, 0, method="circulant", polish=False, log=lambda s: None, op=op)
    assert torch.isfinite(res_c["sigma"]).all()
    assert res["sigma"].shape == res["image"].shape
    assert torch.isfinite(res["sigma"][res["covered"]]).all() and float(res["sigma"][res["covered"]].min()) > 0


# ----------------------------------------------------------------------------- 4. difference tool
def test_half_and_seed_difference():
    N, p, s = 81, 0.05, 0.1
    g = torch.Generator().manual_seed(2)
    truth = U.Smoother(N, p, 0.2, dtype=torch.float64)(torch.randn(1, N, N, generator=g, dtype=torch.float64))
    sig_half = 0.4                                         # node noise of a half fit; the full fit has sig_half / sqrt(2)
    A = truth + sig_half * torch.randn(1, N, N, generator=g, dtype=torch.float64)
    B = truth + sig_half * torch.randn(1, N, N, generator=g, dtype=torch.float64)
    phi = torch.ones(1, dtype=torch.float64)
    sm = U.Smoother(N, p, s, dtype=torch.float64)
    c = N // 2
    knorm = float(sm.weights_at(c, c).pow(2).sum().sqrt())
    inner = torch.zeros(N, N, dtype=torch.bool)
    inner[10:-10, 10:-10] = True
    d = U.difference_map(A, B, phi, p, s, mode="halves", covered=inner)
    expect = sig_half / math.sqrt(2) * knorm
    assert abs(d["sigma_global"] / expect - 1) < 0.08, (d["sigma_global"], expect)
    assert abs(float(d["sigma_local"][inner].median()) / expect - 1) < 0.15
    d2 = U.difference_map(A, A + 0.1 * torch.randn(1, N, N, generator=g, dtype=torch.float64), phi, p, s, mode="seeds", covered=inner)
    assert abs(d2["sigma_global"] / (0.1 / math.sqrt(2) * knorm) - 1) < 0.08     # var(a - b) = 2 sigma_opt^2


# ----------------------------------------------------------------------------- 5. option and file
def test_train_half_option_and_groups():
    assert parse_config([]).train_half == -1
    assert parse_config(["--train-half", "1"]).train_half == 1
    gs = ["g3", "g1", "g2", "g4", "g5"]
    a, b = U.alternate_groups(gs, 0), U.alternate_groups(gs, 1)
    assert a == {"g1", "g3", "g5"} and b == {"g2", "g4"} and not (a & b)


def test_write_product_fits(tmp_path):
    pytest.importorskip("astropy")
    from astropy.io import fits
    N = 9
    res = {"image": torch.rand(N, N, dtype=torch.float64), "sigma": torch.rand(N, N, dtype=torch.float64), "s_star": torch.rand(N, N, dtype=torch.float64),
           "covered": torch.ones(N, N, dtype=torch.bool), "sigma_scales": torch.rand(3, N, N, dtype=torch.float64), "resolution": 0.11,
           "resolution_median": 0.11, "corr_length": 0.09, "rho": 10.0, "cg_iterations": [60, 60], "scales": [0.05, 0.1, 0.2]}
    p = U.write_product(str(tmp_path / "u.fits"), res, "F200W", 0.04)
    with fits.open(p) as h:
        assert np.allclose(h["SIGMA"].data, res["sigma"].numpy().astype(np.float32))
        assert h[0].header["BAND"] == "F200W" and abs(h[0].header["PRODRES"] - 0.11) < 1e-9


def test_command_line_uses_stored_config(tmp_path, monkeypatch):
    """--unc-config: the fit's configuration is used, options written on the command line win (even when equal to a default)."""
    import json
    from fieldop.cli import build_parser
    d = vars(U.add_args(build_parser()).parse_args([]))
    d.update(sky_form="items", sky_pitch=0.02, field="X")
    p = tmp_path / "metrics.json"
    json.dump({"config": d}, open(p, "w"), default=str)
    seen = {}

    def fake_build(cfg, log=print):
        seen.update(sky_form=cfg.sky_form, sky_pitch=cfg.sky_pitch, field=cfg.field, state=cfg.state)
        raise SystemExit(0)
    monkeypatch.setattr(U, "build_from_args", fake_build)
    with pytest.raises(SystemExit):
        U.main(["--unc-config", str(p), "--sky-pitch", "0.04"])
    assert seen == {"sky_form": "items", "sky_pitch": 0.04, "field": "X", "state": str(tmp_path / "fieldop_state.pt")}


def test_diff_products(tmp_path):
    pytest.importorskip("astropy")
    N, g = 41, torch.Generator().manual_seed(7)
    paths = []
    for k in range(2):
        img = 0.3 * torch.randn(N, N, generator=g, dtype=torch.float64)
        res = {"image": img, "sigma": torch.ones(N, N, dtype=torch.float64), "s_star": torch.zeros(N, N, dtype=torch.float64),
               "covered": torch.ones(N, N, dtype=torch.bool), "sigma_scales": torch.ones(1, N, N, dtype=torch.float64), "resolution": 0.1,
               "resolution_median": 0.1, "corr_length": 0.1, "rho": 10.0, "cg_iterations": [], "scales": [0.1]}
        paths.append(U.write_product(str(tmp_path / f"p{k}.fits"), res, "A1", 0.05))
    glob = U.diff_products(paths[0], paths[1], str(tmp_path / "d.fits"), mode="halves")
    assert abs(glob / (0.3 / math.sqrt(2)) - 1) < 0.08
