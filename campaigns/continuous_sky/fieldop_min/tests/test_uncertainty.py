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
    exps, fset, meta, truth, _ = make_field(seed=1, n_per_filter=3, n_held=1, insts=insts, centre_range=0.2, rot_choices=(0,), patch_px=8,
                                            truth_half=0.6)
    cfg = default_config(sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.6, region_px=8, n_lambda=3, k_hidden=16,
                         spec_k=3, stripes=0, steps=0, budget_min=1e6, tau_mode="fixed", seed=0, held_steps=0)
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
    d.update(spec_k=7, sky_pitch=0.02, field="X")
    p = tmp_path / "metrics.json"
    json.dump({"config": d}, open(p, "w"), default=str)
    seen = {}

    def fake_build(cfg, log=print):
        seen.update(spec_k=cfg.spec_k, sky_pitch=cfg.sky_pitch, field=cfg.field, state=cfg.state)
        raise SystemExit(0)
    monkeypatch.setattr(U, "build_from_args", fake_build)
    with pytest.raises(SystemExit):
        U.main(["--unc-config", str(p), "--sky-pitch", "0.04"])
    assert seen == {"spec_k": 7, "sky_pitch": 0.04, "field": "X", "state": str(tmp_path / "fieldop_state.pt")}


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


# ----------------------------------------------------------------------------- 6. real-data adaptation (18 Nov): batching, bands, progress, budget
def _rand_x(op, seed=1):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(op.nA + op.nb, generator=g)
    return torch.cat([u[:op.nA], op.gauge(u[op.nA:])])


def test_pack_batches_and_plan():
    assert U.pack_batches([5, 5, 5, 5, 5], chunk=2) == [[0, 1], [2, 3], [4]]
    assert U.pack_batches([4, 4, 4, 9, 1], batch_px=8) == [[0, 1], [2], [3], [4]]        # a tile above the budget goes alone
    assert U.pack_batches([], batch_px=8) == []
    p = U.plan_passes(12, "draws", True, 8, 60)
    assert p["setup"] == 13 and p["forward_only"] == 1 and p["polish"] == 61 and p["draws"] == 8 * 61 and p["total"] == 13 + 61 + 8 * 61
    c = U.plan_passes(12, "circulant", False, 8, 60)
    assert c["total"] == 13 and c["polish"] == 0 and c["draws"] == 0           # cheap mode: the preconditioner's K + 1 applications (+ 1 forward-only)


def test_batching_policy():
    r = U.resolve_batching
    assert r("cuda", 256) == (64, 16000, None)                       # CUDA default: 64-pixel tiles, 16000 pixels per batched render
    assert r("cpu", 256) == (256, None, 32)                          # CPU default: the validated one (tile = region, 32 tiles per render)
    assert r("cuda", 256, tile=128) == (128, 16000, None)            # explicit tile wins
    assert r("cuda", 256, chunk=8) == (64, None, 8)                  # explicit chunk wins over the budget
    assert r("cuda", 256, tile=64, chunk=8, batch_px=1000) == (64, None, 8)
    assert r("cpu", 256, batch_px=5000) == (256, 5000, None)         # on CPU a budget only when asked for
    assert U.pack_batches([4096] * 7, batch_px=16000) == [[0, 1, 2], [3, 4, 5], [6]]      # 3 tiles of 64 x 64 = 12288 pixels per render


def test_tiling_and_batching_equivalence(tiny):
    """Other tilings / pixel budgets give the same F u to float32 round-off (the render of a pixel does not depend on its tile; the sums differ
    in order). The default tiling is the validated one (tile = --region-px, 32 tiles per render)."""
    model, exps, cfg, op = tiny
    u = _rand_x(op)
    ref = op.apply(u)
    for kw in (dict(tile=4), dict(tile=64), dict(tile=4, batch_px=40), dict(tile=8, chunk=1), dict(batch_px=100)):
        o2 = U.SkyNormalOperator(model, exps, cfg, rho=10.0, log=lambda s: None, **kw)
        assert o2.n_pix == op.n_pix
        assert float((o2.apply(u) - ref).norm() / ref.norm()) < 1e-5, kw
        assert abs(o2.mu / op.mu - 1) < 1e-5, kw
    o3 = U.SkyNormalOperator(model, exps, cfg, rho=10.0, log=lambda s: None, tile=4, batch_px=40)
    for b in o3.batches:                                          # budget respected (a tile of 16 pixels fits 2 per batch of 40)
        assert len(b) == 1 or sum(o3.tiles[i].n for i in b) <= 40
    assert len(o3.batches) > len(op.batches)


def test_tiles_hold_little_and_graphs_are_freed(tiny):
    """A tile keeps only the mask (bool) and w, w r (float32): no pixel dictionary, no background basis; nothing grows with the number of
    applications; the render of a batch leaves no live graph behind."""
    model, exps, cfg, op = tiny
    t = op.tiles[0]
    assert set(t.__slots__) == {"op", "eid", "row", "e", "sl", "n", "mask", "w", "wr"}
    assert t.w.dtype == torch.float32 and t.wr.dtype == torch.float32 and t.mask.dtype == torch.bool and t.w.numel() == t.n
    assert set(t["pix"]).issuperset({"X", "Y", "d", "xi"}) and t["Lb"].shape == (t.n, 6)
    nbytes = lambda: sum(x.numel() * x.element_size() for tt in op.tiles for x in (tt.mask, tt.w, tt.wr))
    before = nbytes()
    u = _rand_x(op)
    for _ in range(3):
        op.apply(u)
    assert nbytes() == before
    flags = [p.requires_grad for p in model.parameters()]
    op.apply(u)
    assert flags == [p.requires_grad for p in model.parameters()]            # frozen() restored the flags
    with torch.no_grad():
        x = op.apply(u)                                           # also fine under a caller's no_grad: the operator enables grad itself
    assert torch.allclose(x, op.apply(u))


def test_cheap_mode_costs_the_preconditioner_only(tiny):
    """--unc-method circulant --unc-polish 0: one forward-only pass (the weights) + K + 1 applications (coverage and the K delta responses of the
    Fourier preconditioner), and nothing more however many bands are read out."""
    model, exps, cfg, _ = tiny
    op = U.SkyNormalOperator(model, exps, cfg, rho=10.0, log=lambda s: None)
    assert (op.n_forward, op.n_apply) == (1, op.K + 1)
    sh = U.solve_shared(op, method="circulant", polish=False, log=lambda s: None)
    for b in (0, "coef:1", "coef:2"):
        res = U.band_readout(model, op, sh, b, log=lambda s: None)
        assert torch.isfinite(res["sigma"]).all() and res["sigma_source"] == "circulant"
    assert (op.n_forward, op.n_apply) == (1, op.K + 1)
    full = U.plan_passes(op.K, "circulant", False, 8, 60)
    assert full["total"] == op.n_apply


def _fake_main(monkeypatch, tiny, tmp_path):
    model, exps, cfg, _ = tiny
    monkeypatch.setattr(U, "build_from_args", lambda c, log=print: (model, exps, model.fset, {}))
    monkeypatch.setattr(U, "read_coeffs", lambda m, s, st, log=print, cfg=None: m.sky.coeffs().detach().clone())
    st = tmp_path / "state.pt"
    torch.save({}, st)
    return ["--state", str(st), "--region-px", "8", "--pred-margin", str(cfg.pred_margin), "--unc-samples", "2", "--unc-maxiter", "12",
            "--unc-tol", "1e-9"]


def test_bands_from_one_solve_equal_separate_runs(tiny, monkeypatch, tmp_path, capsys):
    pytest.importorskip("astropy")
    from astropy.io import fits
    base = _fake_main(monkeypatch, tiny, tmp_path)
    U.main(base + ["--unc-band", "A1,coef:1,coef:2", "--unc-out", str(tmp_path / "multi_{band}.fits")])
    out = capsys.readouterr().out
    assert "tiles" in out and "batches per pass" in out and "planned passes" in out and "first pass" in out and "posterior draw 2/2" in out
    assert out.count("operator applications") == 3 and "done: 3 band(s)" in out
    n_multi = int(out.split("done: 3 band(s), ")[1].split(" applications")[0])
    for b in ("A1", "coef:1", "coef:2"):
        U.main(base + ["--unc-band", b, "--unc-out", str(tmp_path / f"one_{b.replace(':', '')}.fits")])
        with fits.open(tmp_path / f"multi_{b.replace(':', '')}.fits") as hm, fits.open(tmp_path / f"one_{b.replace(':', '')}.fits") as h1:
            for ext in ("SKY", "SKY_FIT", "SIGMA", "SIGMA_CIRC", "SIGMA_SCALES", "RESOLUTION"):
                x, y = np.nan_to_num(hm[ext].data.astype(np.float64)), np.nan_to_num(h1[ext].data.astype(np.float64))
                assert np.abs(x - y).max() <= 1e-6 * max(np.abs(y).max(), 1e-30), (b, ext)
            assert hm[0].header["TRUNCATD"] is False and hm[0].header["NSAMPLE"] == 2
    out = capsys.readouterr().out
    n_one = int(out.split("done: 1 band(s), ")[1].split(" applications")[0])
    assert n_multi == n_one                                       # three bands cost the applications of one
    with pytest.raises(SystemExit):                               # several bands need a {band} pattern: refused before any work
        U.main(base + ["--unc-band", "A1,coef:1", "--unc-out", str(tmp_path / "x.fits")])


def test_time_budget_truncates_cleanly(tiny, tmp_path):
    pytest.importorskip("astropy")
    from astropy.io import fits
    model, exps, cfg, _ = tiny
    # a clock that counts applications: 1 second per application, budget 12 s, no reserve
    op = U.SkyNormalOperator(model, exps, cfg, rho=10.0, log=lambda s: None)
    mon = U.Monitor(op, log=lambda s: None, max_minutes=12 / 60, t0=0.0, reserve=0.0, clock=lambda: float(op.n_apply))
    res = U.uncertainty_map(model, exps, cfg, 0, n_samples=2, maxiter=15, tol=1e-12, log=lambda s: None, op=op, monitor=mon)
    assert res["truncated"] and res["truncated_in"] == "Gauss-Newton step" and op.n_apply <= 12
    assert 1 <= res["polish"]["iterations"] < 15 and res["polish"]["truncated"] and res["residual"] == res["polish"]["rel_residual"] > 1e-12
    assert res["samples"] is None and res["sigma_source"] == "circulant" and torch.isfinite(res["sigma"]).all()
    p = U.write_product(str(tmp_path / "t.fits"), res, 0, model.sky.pitch)
    with fits.open(p) as h:
        assert h[0].header["TRUNCATD"] is True and abs(h[0].header["RESIDUAL"] / res["residual"] - 1) < 1e-6 and h[0].header["TRUNCIN"] == "Gauss-Newton step"
        assert h[0].header["SIGMETH"] == "circulant" and h[0].header["NSAMPLE"] == 0 and h[0].header["NSAMPREQ"] == 2
    # budget enough for the step (4 set-up + 16) but not for the first draw: the cut draw is discarded
    op2 = U.SkyNormalOperator(model, exps, cfg, rho=10.0, log=lambda s: None)
    mon2 = U.Monitor(op2, log=lambda s: None, max_minutes=25 / 60, t0=0.0, reserve=0.0, clock=lambda: float(op2.n_apply))
    res2 = U.uncertainty_map(model, exps, cfg, 0, n_samples=2, maxiter=15, tol=1e-12, log=lambda s: None, op=op2, monitor=mon2)
    assert res2["truncated"] and res2["truncated_in"].startswith("draw") and not res2["polish"]["truncated"]
    assert res2["samples"] is None and res2["cg_iterations"] == [] and op2.n_apply <= 25
    # no budget: nothing truncated
    op3 = U.SkyNormalOperator(model, exps, cfg, rho=10.0, log=lambda s: None)
    res3 = U.uncertainty_map(model, exps, cfg, 0, n_samples=1, maxiter=6, tol=1e-12, log=lambda s: None, op=op3, monitor=U.Monitor(op3, log=lambda s: None))
    assert not res3["truncated"] and len(res3["cg_iterations"]) == 1


def test_command_line_budget_exits_cleanly_and_reports_progress(tiny, monkeypatch, tmp_path, capsys):
    pytest.importorskip("astropy")
    from astropy.io import fits
    base = _fake_main(monkeypatch, tiny, tmp_path)
    U.main(base + ["--unc-band", "A1", "--unc-out", str(tmp_path / "b.fits"), "--unc-max-minutes", "1e-9"])      # returns normally = exit status 0
    out = capsys.readouterr().out
    assert "time budget" in out and "TRUNCATED" in out and "written" in out
    with fits.open(tmp_path / "b.fits") as h:
        assert h[0].header["TRUNCATD"] is True and h[0].header["POLITER"] == 0 and h[0].header["POLISHED"] is False
    # progress lines every 5 iterations, with passes so far, elapsed and time left
    U.main(base + ["--unc-band", "A1", "--unc-out", str(tmp_path / "c.fits"), "--unc-maxiter", "11", "--unc-samples", "1"])
    out = capsys.readouterr().out
    assert "iteration 5, relative residual" in out and "iteration 10, relative residual" in out and "passes so far" in out and "estimated time left" in out
    assert "iteration 4," not in out


def test_blocked_eigendecomposition_equals_unblocked(tiny):
    model, exps, cfg, op = tiny
    assert op.N > 8
    c = (op.N - 1) // 2
    fh = torch.fft.fft2(torch.roll(op.fc.double(), (-c, -c), (-2, -1))).permute(2, 3, 0, 1)
    ph = U.band_vector(model, 0).double()

    def q(b):
        b = 0.5 * (b + b.conj().transpose(-1, -2))
        ev, V = torch.linalg.eigh(b)
        return ((V.conj().transpose(-1, -2) @ ph.to(V.dtype)).abs() ** 2 / (ev.clamp(min=0) + op.mu)).sum(-1)
    full, blocks = q(fh), U._eigh_rows(fh, q, rows=5)
    assert torch.allclose(full, blocks, rtol=1e-10, atol=0)
    assert torch.allclose(op._circ_q(U.band_vector(model, 0)), full, rtol=1e-10, atol=0)
    assert len(op._qcache) >= 1
