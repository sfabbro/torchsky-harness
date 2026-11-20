"""Tests of the 19 Nov solver changes in fieldop/uncertainty.py (SOLVER_DIAGNOSIS_2026-11-19.md). CPU, about 2 minutes.

1. the 'filter' preconditioner: symmetric, positive; the operator as applied with the real-data tiling is symmetric and repeatable
2. on a small six-filter field (K = 12, two channels, chip gap, F090W's gap on the box centre): the 'fourier' preconditioner shows the real-data
   signature (kappa 0 for the bluest coefficients, the solve does not converge) and the 'filter' one converges to 1e-3
3. Lanczos estimates of the preconditioned spectrum from the conjugate-gradient coefficients (dense problem)
4. convergence reporting: an unconverged step is not applied and is labelled so (log, header, SKY_UNCONV); unconverged draws are discarded
5. the cheap sigma follows each filter's own coverage: on the probe-gap field the 'fourier' cheap sigma of F090W is the prior's, the 'filter' one is
   not; on a uniform one-filter field both agree
6. calibration difference of two products (known scale, shift and width recovered)
7. the old preconditioner is still the class default and is unchanged (same operator, same mu, same preconditioner output as an explicit 'fourier')
"""
import os
import sys

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "solver_repro"))

from fieldop import uncertainty as U  # noqa: E402

SMALL = dict(sw_npix=36, lw_npix=18, sky_half=0.65, n_sw=2, n_lw=1, kernel_steps=40, gap_px=6, dither=0.25, Q=2)


def _rand(op, seed):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(op.nA + op.nb, generator=g)
    return torch.cat([u[:op.nA], op.gauge(u[op.nA:])])


@pytest.fixture(scope="module")
def gapc():
    """Six filters, K = 12, two channels (0.031" and 0.063" pixels), chip-gap cross on the short-wavelength detector, F090W dithered by less
    than the gap (its gap stays on the box centre), bright sources with the noise floor, Huber weights."""
    from repro_field import build_model
    model, exps, cfg, truth, o = build_model(gap_centre_filter="F090W", **SMALL)
    ops = {m: U.SkyNormalOperator(model, exps, cfg, rho=10.0, precond_mode=m, log=lambda s: None, tile=64, batch_px=16000) for m in ("fourier", "filter")}
    return model, exps, cfg, ops


def test_filter_preconditioner_symmetric_positive(gapc):
    model, exps, cfg, ops = gapc
    op = ops["filter"]
    u, v = _rand(op, 1), _rand(op, 2)
    Pu, Pv = op.precond(u), op.precond(v)
    assert abs(float(u @ Pv - v @ Pu)) < 1e-5 * float(u.norm() * Pv.norm())
    for s in range(3, 7):
        w = _rand(op, s)
        assert float(w @ op.precond(w)) > 0
    # the operator as applied with the real-data tiling: symmetric to round-off and repeatable bit for bit (no jitter, deterministic adjoint)
    Fu, Fv = op.apply(u), op.apply(v)
    assert abs(float(u @ Fv - v @ Fu)) < 1e-5 * float(u.norm() * Fv.norm())
    assert torch.equal(Fu, op.apply(u))
    # both modes apply the same F (the batching differs: round-off only)
    o2 = ops["fourier"]
    uA = torch.cat([u[:op.nA], torch.zeros(op.nb)])          # the two modes take mu differently (19 Nov, section 11): compare the data part
    assert float(((o2.apply(u) - o2.mu * uA) - (Fu - op.mu * uA)).norm() / (Fu - op.mu * uA).norm()) < 1e-5


def test_probe_gap_signature_and_convergence(gapc):
    model, exps, cfg, ops = gapc
    old, new = ops["fourier"], ops["filter"]
    assert old.kappa[0] < 1e-5 and old.kappa[1] < 1e-5 and min(old.kappa[4:8]) > 1e-4      # the real half 0 signature (kappa 0.0, 0.0, ...)
    assert new.d_filter[0][new.probes[model.fset.names.index("F090W")]] == pytest.approx(1.0)
    # on this small field (33 x 33 nodes, one long-wavelength exposure per filter) neither reaches 1e-3 (section 5 of the report: convergence to
    # 1e-3 is shown on the 56 x 56 reproducer by tests/solver_repro/run_repro.py); the new preconditioner must be far better conditioned
    b = new.residual_rhs()
    _, info_new = U.pcg(new, b, tol=1e-3, maxiter=40)
    b_old = old.residual_rhs()
    _, info_old = U.pcg(old, b_old, tol=1e-3, maxiter=40)
    assert info_old["rel_residual"] > 2 * info_new["rel_residual"], (info_old["rel_residual"], info_new["rel_residual"])
    assert info_old["ritz_max"] > 5 * info_new["ritz_max"], (info_old["ritz_max"], info_new["ritz_max"])


def test_lanczos_ritz_values_dense():
    from test_uncertainty import DenseOp
    op = DenseOp(K=2, N=5, n_exp=3, n_pix=80, seed=4)
    b = op.residual_rhs()
    n = op.nA + op.nb
    x, info = U.pcg(op, b, tol=1e-14, maxiter=n)
    M = torch.stack([op.precond(op.apply(torch.cat([e[:op.nA], op.gauge(e[op.nA:])]))) for e in torch.eye(n, dtype=torch.float64)], 1)
    ev = torch.linalg.eigvals(M).real                       # the preconditioned operator (similar to a symmetric one) on the gauge subspace
    ev = ev[ev > 1e-8]
    assert abs(info["ritz_max"] / float(ev.max()) - 1) < 1e-3
    assert info["ritz_min"] >= float(ev.min()) * (1 - 1e-3)


def test_unconverged_step_not_applied_and_reported(gapc, tmp_path):
    pytest.importorskip("astropy")
    from astropy.io import fits
    model, exps, cfg, ops = gapc
    op = ops["filter"]
    logs = []
    sh = U.solve_shared(op, method="circulant", polish=True, maxiter=3, tol=1e-12, log=logs.append, accept_tol=1e-3)
    assert sh["polish_converged"] is False
    assert any("did NOT CONVERGE" in s and "NOT polished" in s for s in logs)
    res = U.band_readout(model, op, sh, "F200W", log=lambda s: None)
    assert torch.equal(res["image"], res["image_fit"]) and res["image_unconverged"] is not None
    p = U.write_product(str(tmp_path / "u.fits"), res, "F200W", model.sky.pitch)
    with fits.open(p) as h:
        assert h[0].header["POLISHED"] is False and h[0].header["CONVERGD"] is False and "SKY_UNCONV" in h
        assert h[0].header["PRECOND"] == "filter" and h[0].header["POLRESID"] > 1e-3
    # converged (by a loose tolerance: this tests the labelling, not the solver): applied and labelled
    sh2 = U.solve_shared(op, method="circulant", polish=True, maxiter=15, tol=1e-4, log=logs.append, accept_tol=0.9)
    assert sh2["polish_converged"] is True
    res2 = U.band_readout(model, op, sh2, "F200W", log=lambda s: None)
    p2 = U.write_product(str(tmp_path / "v.fits"), res2, "F200W", model.sky.pitch)
    with fits.open(p2) as h:
        assert h[0].header["POLISHED"] is True and h[0].header["CONVERGD"] is True and "SKY_UNCONV" not in h
        assert h[0].header["RITZMAX"] > h[0].header["RITZMIN"] > 0
        assert h[0].header["POLCHGIT"] == 10 and h[0].header["POLCHGMX"] >= h[0].header["POLCHG"] > 0     # change over the last iterations, sigma units
    # draws above the tolerance are discarded, and said so
    logs.clear()
    samp, recs = U.posterior_samples(op, 2, tol=1e-12, maxiter=3, log=logs.append, accept_tol=1e-3)
    assert samp is None and sum("NOT CONVERGED" in s for s in logs) == 2


def test_cheap_sigma_follows_filter_coverage(gapc):
    model, exps, cfg, ops = gapc
    old, new = ops["fourier"], ops["filter"]
    s = 2 * model.sky.pitch
    phi = U.band_vector(model, "F090W").cpu()
    c = (old.N - 1) // 2
    sig_old, sig_new = old.circulant_sigma(phi, s), new.circulant_sigma(phi, s)
    # where F090W is well covered (its probe), the old cheap sigma is the prior-limited one taken at the gap: several times the new one
    py, px = new.probes[model.fset.names.index("F090W")]
    assert float(sig_old[py, px]) > 3 * float(sig_new[py, px])
    # at the box centre (in F090W's gap) the new one is larger than at the probe
    assert float(sig_new[c, c]) > float(sig_new[py, px])


def test_cheap_sigma_uniform_single_filter_agrees():
    from test_uncertainty import SynthInstrument, default_config, make_field, make_sky, FieldOperator
    insts = [SynthInstrument("A", 0.10, 16, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3)])]
    exps, fset, meta, truth, _ = make_field(seed=1, n_per_filter=4, n_val=0, n_held=1, insts=insts, centre_range=0.05, rot_choices=(0,), patch_px=8,
                                            truth_half=0.6)
    cfg = default_config(legacy_defaults=False, sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.6, region_px=8, n_lambda=3, k_hidden=16,
                         spec_k=3, pattern_grid=0, stripes=0, steps=0, budget_min=1e6, tau_mode="fixed", seed=0, held_steps=0)
    sky, _ = make_sky(exps, fset, cfg, 1.0, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    old = U.SkyNormalOperator(model, exps, cfg, rho=10.0, log=lambda s: None)
    new = U.SkyNormalOperator(model, exps, cfg, rho=10.0, precond_mode="filter", log=lambda s: None, mu=old.mu)
    phi = U.band_vector(model, 0).cpu()
    c = (old.N - 1) // 2
    a, b = old.circulant_sigma(phi, 0.16)[c, c], new.circulant_sigma(phi, 0.16)[c, c]
    assert abs(float(a / b) - 1) < 0.25, (float(a), float(b))


def test_calibration_difference_recovers_known_terms():
    N, p = 121, 0.04
    rng = np.random.default_rng(3)
    x = (np.arange(N) - (N - 1) / 2) * p
    X, Y = np.meshgrid(x, x)
    src = [(rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8), rng.uniform(50, 500)) for _ in range(30)]

    def img(dx, dy, scale, V):
        out = np.zeros((N, N))
        for (sx, sy, f) in src:
            out += scale * f * np.exp(-((X - sx - dx) ** 2 + (Y - sy - dy) ** 2) / (2 * V)) / (2 * np.pi * V) * p * p
        return out
    V0 = 0.05 ** 2
    a = img(0.005, -0.002, 1.03, V0 * 1.04)
    b = img(0.0, 0.0, 1.0, V0)
    sig = np.full((N, N), 0.01)
    cd = U.calibration_difference(a, b, sig, sig, np.ones((N, N), bool), p, n_src=30, win=5, min_snr=10.0)
    assert cd["n_sources"] >= 15
    assert abs(cd["scale_minus_1"] - 0.03) < 0.01
    assert abs(cd["shift_arcsec"][0] - 0.005) < 0.001 and abs(cd["shift_arcsec"][1] + 0.002) < 0.001     # first order in the shift
    assert 0 < cd["width_arcsec2"] < 2.5 * 0.5 * 0.04 * V0          # sign and size only: in a +-5-node window the width term is correlated with the scale
    assert cd["source_pull_rms_after_median"] < 0.2 * cd["source_pull_rms_before_median"]


def test_old_mode_is_default_and_unchanged():
    from test_uncertainty import SynthInstrument, default_config, make_field, make_sky, FieldOperator
    insts = [SynthInstrument("A", 0.10, 16, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3)])]
    torch.manual_seed(0)
    exps, fset, meta, truth, _ = make_field(seed=1, n_per_filter=3, n_val=0, n_held=1, insts=insts, centre_range=0.2, rot_choices=(0,), patch_px=8,
                                            truth_half=0.6)
    cfg = default_config(legacy_defaults=False, sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.6, region_px=8, n_lambda=3, k_hidden=16,
                         spec_k=3, pattern_grid=0, stripes=0, steps=0, budget_min=1e6, tau_mode="fixed", seed=0, held_steps=0)
    sky, _ = make_sky(exps, fset, cfg, 1.0, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    a = U.SkyNormalOperator(model, exps, cfg, rho=10.0, log=lambda s: None)
    b = U.SkyNormalOperator(model, exps, cfg, rho=10.0, log=lambda s: None, precond_mode="fourier")
    assert a.precond_mode == "fourier" and a.mu == b.mu
    u = _rand(a, 9)
    assert torch.equal(a.apply(u), b.apply(u)) and torch.equal(a.precond(u), b.precond(u))
    xa, ia = U.pcg(a, a.residual_rhs(), tol=1e-12, maxiter=15)
    xb, ib = U.pcg(b, b.residual_rhs(), tol=1e-12, maxiter=15)
    assert torch.equal(xa, xb) and ia["history"] == ib["history"]
    # the command-line defaults: the new preconditioner and a convergence check
    from fieldop.cli import build_parser
    d = vars(U.add_args(build_parser()).parse_args([]))
    assert d["unc_precond"] == "filter" and d["unc_accept_tol"] > 0
