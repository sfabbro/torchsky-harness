"""Unit tests of the three wing / shoulder options of scripts/opfield.py (8 Nov 2026), CPU:

 1. --psi-param log: start kernel == softplus start kernel and forward L; stable (finite, unit sum) for log-densities spanning +-300; init-from-E2 path
 2. --importance-sampling: peak -> annulus lookup equals a brute-force computation; the weights give an UNBIASED estimate of the all-pixel loss on a toy
    problem (mean over many draws vs the full mean; the unweighted stratified mean is shown to be biased); effective sample size
 3. --two-scale: (a) wing fraction -> 0 reproduces the core-only prediction exactly; (b) wing fraction -> 1 with a known table: the FFT path (cache, bicubic
    read) equals a direct stencil sum to < 1e-4; (c) table orientation / Jacobian convention (a spike at detector +x under a 90 degree roll lands at sky +y);
    (d) gradients reach table, fraction, modulation and sky; (e) untrained (detector, band) cells are aliased; (f) default model has no wing parameters
 4. smoke runs on testdata (slow): python tests/test_opfield_wings.py --smoke  (each flag alone and all three; exit 0, finite scores, dumps)

Run: venvT/bin/python tests/test_opfield_wings.py   (add --smoke for 4)
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
import opfield as O  # noqa: E402
import test_opfield as T  # noqa: E402

torch.set_num_threads(2)


# ------------------------------------------------------------------------------------------------ 1  --psi-param log
def test_psi_log_start_equals_softplus_start():
    ms, a, st = T.make_multi()
    ml, a2, st2 = T.make_multi(flags=["--psi-param", "log"])
    assert ml.psi_log and not ms.psi_log
    bt = T.batch_on_grid(torch.linspace(15, 45, 5), torch.linspace(15, 45, 5), eid=2)
    with torch.no_grad():
        lv_s, lv_l = ms.make_levels(0.3), ml.make_levels(0.3)
        os_, ol = ms(bt, ms.nuis, lv_s), ml(bt, ml.nuis, lv_l)
        h = ms.trunk(ms.ctx_input(bt["eid"], bt["dx"], bt["dy"], bt["nid"], ms.nuis))
        ws, cs, _ = ms.shape(0, h, ms.m_band[bt["eid"]], ms.m_brow[bt["eid"]])
        wl, cl, _ = ml.shape(0, h, ml.m_band[bt["eid"]], ml.m_brow[bt["eid"]])
    err_w = float(((ws - wl).abs() / ws.clamp(min=1e-12)).max())
    assert err_w < 2e-4, err_w                                   # node masses (the floor nodes 1e-7 x dv included) agree to float32 round-off of exp(log)
    assert torch.allclose(os_["L"], ol["L"], rtol=1e-5, atol=1e-9)
    assert torch.allclose(cs, cl, atol=1e-5)
    print(f"  log vs softplus start: max relative node-mass difference {err_w:.1e}; max |L difference| {float((os_['L'] - ol['L']).abs().max()):.1e}")


def test_psi_log_is_stable_over_a_huge_dynamic_range():
    ml, a, st = T.make_multi(flags=["--psi-param", "log"])
    with torch.no_grad():
        ml.base.copy_(torch.linspace(100.0, -300.0, ml.base.shape[1])[None].expand_as(ml.base))     # relative mass 1e-174 at the far nodes
    bt = T.batch_on_grid(torch.linspace(15, 45, 4), torch.linspace(15, 45, 4), eid=0)
    h = ml.trunk(ml.ctx_input(bt["eid"], bt["dx"], bt["dy"], bt["nid"], ml.nuis))
    w, cbar, _ = ml.shape(0, h, ml.m_band[bt["eid"]], ml.m_brow[bt["eid"]])
    assert torch.isfinite(w).all() and torch.isfinite(cbar).all()
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
    out = ml(bt, ml.nuis, ml.make_levels(0.0))
    out["L"].sum().backward()
    assert torch.isfinite(ml.base.grad).all()


# ------------------------------------------------------------------------------------------------ 2  --importance-sampling
class MockPool:
    """The attributes PeakSampler needs: N, device, eid, x0, bright, bright_idx, get()."""

    def __init__(self, N=30000, nexp=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.device, self.N = torch.device("cpu"), N
        self.eid = torch.randint(0, nexp, (N,), generator=g)
        self.x0 = torch.rand(N, 2, generator=g) * 63.0
        self.bright = torch.rand(N, generator=g) < 0.1
        self.bright_idx = torch.nonzero(self.bright).squeeze(1)

    def get(self, sel):
        return dict(eid=self.eid[sel], x0=self.x0[sel])


def make_sampler(radii="2,5,10,20", frac=0.5, flags=()):
    m, a, st = T.make_multi(flags=["--importance-sampling", radii, "--is-frac", str(frac), "--is-snr", "50"] + list(flags))
    pool = MockPool()
    sig_b = torch.full((2,), 0.01)
    S = O.PeakSampler(pool, m, O.parse_is_radii(radii), a, sig_b)
    S.refresh(0)
    return S, pool, m


def test_importance_sampling_lookup_matches_brute_force():
    S, pool, m = make_sampler()
    r = (pool.x0 - torch.tensor([32.0, 30.0])).norm(dim=1)                    # the single peak of the model band image (J = identity)
    exp_lab = torch.bucketize(r, torch.tensor([2.0, 5.0, 10.0, 20.0]), right=True)
    exp_lab = torch.where(exp_lab >= 4, torch.full_like(exp_lab, -1), exp_lab)
    assert S.history[0]["peaks_per_band"] == [1, 1]
    assert torch.equal(S.lab.long(), exp_lab), int((S.lab.long() != exp_lab).sum())
    assert sum(S.counts) == int((exp_lab >= 0).sum())
    print(f"  lookup: annulus counts {S.counts} == brute force; peaks per band {S.history[0]['peaks_per_band']}")


def test_importance_sampling_is_unbiased_on_a_toy_problem():
    S, pool, m = make_sampler(frac=0.5)
    r = (pool.x0 - torch.tensor([32.0, 30.0])).norm(dim=1)
    f = 1.0 / (1.0 + (r / 3.0) ** 2) * (1.0 + 0.5 * torch.sin(pool.x0[:, 0])) + 0.3 * pool.bright.float() + 0.05 * pool.x0[:, 1] / 63.0   # strongly peaked toy 'nll'
    full = float(f.mean())
    gen = torch.Generator().manual_seed(5)
    B, nrep = 256, 4000
    sel_log = []
    orig_get = pool.get

    def get_rec(sel):                       # record the pool index of every draw (draw() returns only the batch dict)
        sel_log.append(sel.clone())
        return orig_get(sel)

    pool.get = get_rec
    wm, um, ess = [], [], []
    for _ in range(nrep):
        bt = S.draw(B, 0.25, gen)
        sel = sel_log.pop()
        wm.append(float((bt["w"] * f[sel]).mean()))
        um.append(float(f[sel].mean()))
        ess.append(S.last["ess"])
    wm, um = np.array(wm), np.array(um)
    se = wm.std() / math.sqrt(nrep)
    mean_w = float(wm.mean())
    assert abs(mean_w - full) < 4.5 * se, (mean_w, full, se)
    assert abs(mean_w - full) / full < 0.01
    se_u = um.std() / math.sqrt(nrep)
    assert abs(float(um.mean()) - full) > 10 * se_u           # the unweighted stratified mean is clearly biased (the test has power)
    ess_m = float(np.mean(ess))
    assert 0.3 * B < ess_m < B
    print(f"  full-data mean {full:.5f}; weighted batch mean over {nrep} draws {mean_w:.5f} +- {se:.5f} (unbiased); unweighted {float(um.mean()):.5f} (biased, "
          f"{abs(float(um.mean()) - full) / se_u:.0f} sigma); mean ESS {ess_m:.0f} of {B}; weights in [{S.last['w_min']:.3f}, {S.last['w_max']:.2f}]")


def test_importance_sampling_without_peaks_reduces_to_the_default_pmf():
    S, pool, m = make_sampler(radii="2,5")
    with torch.no_grad():
        m.sky.zero_()                                          # no peak above S/N 50: no stratum, sampling = uniform + bright
    S.refresh(1)
    assert sum(S.counts) == 0
    B = 1000
    bt = S.draw(B, 0.25, torch.Generator().manual_seed(1))
    nb, nu = 250, 750
    assert S.last["n_is"] == 0
    # q = (nu / N + nb 1_bright / N_b) / B  ->  w = 1 / (N q)
    w_expected_nonbright = B / nu
    assert torch.isfinite(bt["w"]).all() and float(bt["w"].max()) <= w_expected_nonbright + 1e-6
    assert abs(float(bt["w"].max()) - w_expected_nonbright) < 1e-4          # a non-bright pixel drawn from the uniform part only


# ------------------------------------------------------------------------------------------------ 3  --two-scale
TS = ["--two-scale", "0.15", "--wing-mod", "2"]


def test_two_scale_wing_fraction_zero_is_the_core_only_prediction():
    m, a, st = T.make_multi(flags=TS)
    T.randomise_heads(m, 0.3)
    with torch.no_grad():
        m.bank.add_(torch.randn_like(m.bank) * 0.3)
        m.wing_logit.fill_(-40.0)
        m.wing_tab.add_(torch.randn_like(m.wing_tab))
    bt = T.batch_on_grid(torch.linspace(18, 44, 5), torch.linspace(18, 44, 5), eid=1)
    lv = m.make_levels(0.4)
    with torch.no_grad():
        out = m(bt, m.nuis, lv)
        eid = bt["eid"]
        h = m.trunk(m.ctx_input(eid, bt["dx"], bt["dy"], bt["nid"], m.nuis))
        mu, Lc, M, a_ = m.heads(h, eid, (bt["dx"], bt["dy"], bt["nid"], m.nuis))
        w, cbar, wq = m.shape(0, h, m.m_band[eid], m.m_brow[eid])
        pts = O.stencil_positions(bt["x0"], mu, M, m.sv[: m.nk_ch[0]], cbar)
        st_ = torch.einsum("kbn,bqk->bqn", O.sample_levels(lv, pts, m.slices_ch[0]), m.Phi[m.m_band[eid]])
        ref = torch.exp(a_) * (((w * st_).sum(-1)) * wq).sum(-1)
    assert torch.allclose(out["L"], ref, rtol=1e-6, atol=1e-12), float((out["L"] - ref).abs().max())
    assert m.nk_ch[0] < m.stencil.nc                         # the stencil really is truncated inside the dense core
    print(f"  f_w = 4e-18: max |L - core-only| = {float((out['L'] - ref).abs().max()):.1e}; per-pixel stencil {m.nk_ch[0]} nodes (dense core {m.stencil.nc})")


def direct_correlation(S, K, nh):
    """out[y, x] = sum_D K[y_D, x_D] S_padreplicate[y + D_y, x + D_x] (float64 loops over the offsets)."""
    S = S.double().numpy()
    H, W = S.shape
    P = np.pad(S, nh, mode="edge")
    out = np.zeros((H, W))
    Kn = K.double().numpy()
    for j in range(2 * nh + 1):
        for i in range(2 * nh + 1):
            if Kn[j, i] != 0.0:
                out[:, :] += Kn[j, i] * P[j: j + H, i: i + W]               # offset D = (i - nh, j - nh): P index = y + nh + D_y = y + j
    return out


def test_two_scale_fft_equals_direct_stencil_sum():
    m, a, st = T.make_multi(flags=["--two-scale", "0.15"])
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        m.wing_tab.copy_(torch.randn(m.wing_tab.shape, generator=g) * 1.5)       # a rough known table
        m.wing_logit.fill_(40.0)                                                 # f_w = 1: the pixel is the wing term alone
        th = math.radians(23.0)
        Rm = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        m.wg_Jinv.copy_(torch.linalg.inv(1.1 * Rm)[None].expand_as(m.wg_Jinv))   # a rotated, slightly scaled group Jacobian
        m.sky.copy_(torch.randn(m.sky.shape, generator=g) * 0.1 + m.sky)         # rough image
    bt = T.batch_on_grid(torch.arange(6.0, 58.0, 4.0), torch.arange(6.0, 58.0, 4.0), eid=0)
    lv = m.make_levels(0.0)
    with torch.no_grad():
        out = m(bt, m.nuis, lv)
        gi = m.m_wgrp[0][None]
        K, mask = m.wing_kernels(gi)
        assert abs(float(K.sum()) - 1.0) < 1e-5
        S = m.band_image()[int(m.m_band[0])]
        D = direct_correlation(S, K[0], m.wing_nh)
        ix, iy = bt["x0"][:, 0].long(), bt["x0"][:, 1].long()
        ref = torch.tensor(D)[iy, ix]
        _, modv = m.wing_amp(bt["eid"], bt["dx"], bt["dy"])
        ref = ref * modv.double() * torch.exp(out["a"]).double()
    err = float((out["L"].double() - ref).abs().max() / ref.abs().max())
    assert err < 1e-4, err
    print(f"  FFT wing path vs direct stencil sum (rotated, scaled group, rough table and image): max relative error {err:.1e}")


def test_two_scale_table_orientation_follows_the_jacobian():
    nh = 8
    tab = torch.full((1, 2 * nh + 1, 2 * nh + 1), -30.0)
    tab[0, nh, nh + 4] = 0.0                                              # spike at detector offset u = (+4, 0) (column +4, row 0)
    one = torch.ones(1)
    for deg, expect_xy in ((0.0, (4, 0)), (90.0, (0, 4)), (180.0, (-4, 0))):
        th = math.radians(deg)
        Jm = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])[None]
        K, mask = O.render_wing_kernels(tab, torch.linalg.inv(Jm), one, 0.5 * one, nh * one, nh)
        j, i = np.unravel_index(int(K[0].argmax()), K[0].shape)
        assert (i - nh, j - nh) == expect_xy, (deg, (i - nh, j - nh))
        # a pixel at x reads the sky at x + D: the delta image appears at x0 - D
        img = torch.zeros(40, 40)
        img[20, 20] = 1.0
        Hp = O.next_fast_len(40 + 2 * nh)
        out = O.correlate_fft(O.pad_fft_image(img, nh, Hp, Hp)[None], K, nh, Hp, Hp, 40, 40)[0]
        y, x = np.unravel_index(int(out.argmax()), out.shape)
        assert (x - 20, y - 20) == (-expect_xy[0], -expect_xy[1]), (deg, x, y)
    print("  table spike at detector +x lands at sky (+4,0), (0,+4), (-4,0) for rolls 0, 90, 180 degrees; the pixel reads the sky at x + D")


def test_two_scale_gradients_and_prior():
    m, a, st = T.make_multi(flags=TS)
    T.randomise_heads(m, 0.2)
    bt = T.batch_on_grid(torch.linspace(15, 45, 6), torch.linspace(15, 45, 6), eid=3)
    bt["dx"] = torch.linspace(-0.8, 0.8, 36)
    bt["dy"] = torch.linspace(0.8, -0.8, 36)
    out = m(bt, m.nuis, m.make_levels(0.7))
    out["L"].sum().backward()
    for n_, p_ in (("tab", m.wing_tab), ("logit", m.wing_logit), ("mod", m.wing_mod), ("sky", m.sky), ("base", m.base)):
        assert p_.grad is not None and float(p_.grad.abs().sum()) > 0, n_
    assert float(m.wing_tab.grad[m.wing_alias[m.m_wrow[3]]].abs().sum()) > 0
    pr = m.prior_terms(torch.arange(6))
    assert float(pr["wing"]) < 1e-12                                          # at the start
    print(f"  gradients reach tab {float(m.wing_tab.grad.abs().sum()):.1e}, logit {float(m.wing_logit.grad.abs().sum()):.1e}, mod {float(m.wing_mod.grad.abs().sum()):.1e}")


def test_two_scale_untrained_cells_are_aliased_and_default_has_no_wing_parameters():
    m, a, st = T.make_multi(flags=TS)
    moves = m.set_train_rows([0, 1, 3, 4, 5])                                 # no training exposure in (detector 1, band 0) = exposure 2
    assert any("-> " in x for x in m.untrained_wing_cells), m.untrained_wing_cells
    r2 = int(m.m_wrow0[2])
    assert int(m.wing_alias[r2]) != r2 and int(m.m_wrow[2]) == int(m.wing_alias[r2]) and m.wing_row_band[int(m.wing_alias[r2])] == m.wing_row_band[r2]
    d, a0, st0 = T.make_multi()
    assert not d.two_scale and not any(n_.startswith("wing") for n_ in d.state_dict())
    assert not any(n_.startswith("wing") for n_, _ in d.named_parameters())
    lv = d.make_levels(0.2)
    assert type(lv) is list                                                   # plain list: the default path is untouched
    print(f"  aliased: {m.untrained_wing_cells}; default model: no wing parameters / buffers in the state dict")


def test_two_scale_kernel_points_include_the_wings():
    m, a, st = T.make_multi(flags=TS)
    with torch.no_grad():
        m.wing_logit.fill_(math.log(0.1 / 0.9))
    kp = O.kernel_points(m, 0)
    nk = m.nk_ch[0]
    assert kp["pos"].shape[0] == kp["mass"].shape[0] == kp["vdet"].shape[0] > nk
    assert abs(kp["mass"][nk:].sum() - 0.1 * float(m.wing_amp(torch.tensor([0]), torch.zeros(1), torch.zeros(1))[1])) < 1e-4
    assert abs(kp["mass"][:nk].sum() - 0.9) < 1e-4 and kp["nc"] <= nk
    print(f"  kernel_points: {nk} core + {kp['pos'].shape[0] - nk} wing nodes, wing mass {kp['mass'][nk:].sum():.3f}")


# ------------------------------------------------------------------------------------------------ 4  smoke runs (slow)
def smoke_args(out, extra):
    d = os.path.join(ROOT, "runs", "flag_tests")
    sys.path.insert(0, HERE)
    import run_flag_tests as R
    R.OUT = d
    data = R.make_data_dir()
    return R.common(data, 60) + list(extra), d


@pytest.mark.slow
@pytest.mark.parametrize("name,extra", [("psi_log", ["--psi-param", "log"]),
                                        ("is", ["--importance-sampling", "1,3,8,15,30", "--is-refresh", "20", "--is-snr", "20"]),
                                        ("twoscale", ["--two-scale", "0.3", "--wing-mod", "1"]),
                                        ("all_three", ["--psi-param", "log", "--importance-sampling", "1,3,8,15,30", "--is-refresh", "20", "--is-snr", "20", "--two-scale", "0.3", "--wing-mod", "1"])])
def test_smoke(name, extra):
    import run_flag_tests as R
    args, d = smoke_args(None, extra)
    rec = R.run(name, args)
    assert rec["exit"] == 0, open(os.path.join(d, name + ".log")).read()[-3000:]
    assert rec["finite"] and rec["n_dump"] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and k != "test_smoke"]
    for f in fns:
        print(f.__name__)
        f()
    if "--smoke" in sys.argv:
        for nm, ex in (("psi_log", ["--psi-param", "log"]), ("is", ["--importance-sampling", "1,3,8,15,30", "--is-refresh", "20", "--is-snr", "20"]),
                       ("twoscale", ["--two-scale", "0.3", "--wing-mod", "1"]),
                       ("all_three", ["--psi-param", "log", "--importance-sampling", "1,3,8,15,30", "--is-refresh", "20", "--is-snr", "20", "--two-scale", "0.3", "--wing-mod", "1"])):
            print("smoke", nm)
            test_smoke(nm, ex)
    print("ALL OK")
