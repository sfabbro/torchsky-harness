"""Unit tests and smoke run of scripts/opfield.py (CPU).

 1. stencil construction (sizes, order, prefixes, levels, quadrature areas)
 2. Psi: unit sum and zero centroid for random weights
 3. a mu shift moves the predicted image by the right amount
 4. Lambda rotation by 90 degrees rotates the predicted image
 5. gradients reach the sky, theta (trunk, heads, bank, base) and z
 6. the shift and flux gauge penalties are zero at their gauge (and not elsewhere); the blur gauge vanishes for a delta kernel
 7. bicubic interpolation accuracy (measured, stated in the module docstring)
 8. no data value enters pi or sigma (forward takes no 'd'); mixture NLL and outlier posterior behave
 9. smoke run on testdata/nircam: exit 0, metrics.json with finite scores and the bootstrap block (slow: pytest -m slow or python tests/test_opfield.py --smoke)

Run: /tmp/claude-0/-home-claude/e41a3836-de0d-574f-a748-03c2fea2b582/scratchpad/venvT/bin/python tests/test_opfield.py   (add --smoke for 9)
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import opfield as O  # noqa: E402
import smacs_four_v36 as V  # noqa: E402

torch.set_num_threads(2)
DEV = torch.device("cpu")


# ------------------------------------------------------------------------------------------------ fixture
def make_model(n_exp=3, H=64, sky_fn=None, J=None, core_radius=2.0, core_pitch=0.5, wing_ratio=1.5, nang=12, wing_arcsec=1.2, seed=0, Q=3):
    """A tiny OpField on a hand-made world: one band, one channel, n_exp exposures (pixel scale 1 target px per detector px)."""
    torch.manual_seed(seed)
    a = O.parse_args(["--out", "/tmp/unused", "--core-radius", str(core_radius), "--core-pitch", str(core_pitch), "--wing-ratio", str(wing_ratio),
                      "--wing-nang", str(nang), "--wing-arcsec", str(wing_arcsec), "--hidden", "16", "--n-bank", "4", "--shape-hidden", "16",
                      "--scale", "0.1", "--spec-q", str(Q)])
    reg = O.OpReg(["bA"], [1.5], ["A"], ["CH"], [False])
    Jm = torch.eye(2).repeat(n_exp, 1, 1) if J is None else J
    meta = dict(band=torch.zeros(n_exp, dtype=torch.long), inst=torch.zeros(n_exp, dtype=torch.long), chan=torch.zeros(n_exp, dtype=torch.long),
                det=torch.zeros(n_exp, dtype=torch.long), epoch=torch.zeros(n_exp), ltexp=torch.zeros(n_exp),
                roll=torch.tensor([[1.0, 0.0, 1.0]]).repeat(n_exp, 1), lscale=torch.zeros(n_exp), ro=torch.zeros(n_exp, 2), J=Jm,
                unit=torch.full((n_exp,), 0.1), b0=torch.zeros(n_exp), stripe_exp=torch.zeros(n_exp))
    Hs = Ws = 64
    yy, xx = torch.meshgrid(torch.arange(Hs).float(), torch.arange(Ws).float(), indexing="ij")
    img = torch.exp(-0.5 * ((xx - 32) ** 2 + (yy - 30) ** 2) / 3.0 ** 2) if sky_fn is None else sky_fn(xx, yy)
    A0 = img[None].repeat(12, 1, 1)  # partition of unity: flat spectrum = the image
    basis = V.RBFSpectralBasis(12)
    lam_nodes = torch.linspace(1.45, 1.55, Q)[None]
    w_nodes = torch.full((1, Q), 1.0 / Q)
    Phi = basis(lam_nodes.flatten()).view(1, Q, 12).detach()
    scale_ch = [{"sky_per_det_px": 1.0, "arcsec_per_det_px": 0.1}]
    st = O.Stencil(a.core_radius, a.core_pitch, a.wing_ratio, a.wing_nang, a.wing_arcsec / 0.1)
    m = O.OpField(a, reg, meta, st, A0, Phi, w_nodes, lam_nodes, 1, scale_ch, DEV)
    return m, a, st


def batch_on_grid(xs, ys, eid=0):
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    x0 = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
    n = x0.shape[0]
    z = torch.zeros(n)
    return dict(eid=torch.full((n,), eid, dtype=torch.long), nid=torch.full((n,), eid, dtype=torch.long), x0=x0, dx=z, dy=z, sp=torch.ones(n),
                row=torch.zeros(n, dtype=torch.long), col=torch.zeros(n, dtype=torch.long))


def delta_psi(m, sigma_core=None):
    """Set base so that Psi is (almost) a delta at the stencil origin."""
    with torch.no_grad():
        m.base.fill_(-16.0)
        m.base[:, 0] = 20.0


def set_head_bias(m, head, vals):
    with torch.no_grad():
        getattr(m, head)[-1].bias.copy_(torch.tensor(vals, dtype=torch.float32))


def randomise_heads(m, std=0.2):
    with torch.no_grad():
        for mod in (m.head_mu, m.head_lam, m.head_a, m.shape_mlp, m.g_net, m.var_net, m.pi_net):
            for p in mod.parameters():
                p.add_(torch.randn_like(p) * std)


# ------------------------------------------------------------------------------------------------ 1
def test_stencil():
    st = O.Stencil(3.0, 0.25, 1.25, 24, 97.0)
    assert st.nc == 441 + 0 or st.nc > 400
    assert np.allclose(st.v[0], 0.0)                       # origin first
    assert np.all(np.diff(st.r[: st.nc]) >= -1e-12)        # core sorted by radius
    assert st.r[st.nc:].min() > 3.0                        # wings outside the core
    assert st.ring_r[-1] > 80 and st.n_rings * 24 + st.nc == len(st)
    # cell areas: core ~ pi r^2, wings = annulus area
    assert abs(st.dv[: st.nc].sum() - math.pi * 9.0) / (math.pi * 9.0) < 0.05
    edges = [3.0 * 1.25 ** j for j in range(st.n_rings + 1)]
    assert abs(st.dv[st.nc:].sum() - math.pi * (edges[-1] ** 2 - 9.0)) / (math.pi * edges[-1] ** 2) < 1e-9
    # prefixes
    assert st.nk_for(23) < st.nk_for(48) < st.nk_for(97) <= len(st)
    for s in (0.775, 1.5, 3.2):
        lev = st.levels_for(st.nk_for(97), s)
        assert np.all(np.diff(lev) >= 0) and lev[0] == 0
        sl = O.level_slices(lev)
        assert sl[0][1] == 0 and sl[-1][2] == len(lev) and all(sl[i][2] == sl[i + 1][1] for i in range(len(sl) - 1))
    print(f"  stencil: {st.nc} core + {st.n_rings} rings x 24 = {len(st)}; prefixes {st.nk_for(23)}, {st.nk_for(48)}, {st.nk_for(97)}")


# ------------------------------------------------------------------------------------------------ 2
def test_kernel_unit_sum_zero_centroid():
    m, a, st = make_model()
    randomise_heads(m, 0.5)
    with torch.no_grad():
        m.bank.add_(torch.randn_like(m.bank) * 0.5)
        bt = batch_on_grid(torch.linspace(10, 50, 4), torch.linspace(10, 50, 4))
        h = m.trunk(m.ctx_input(bt["eid"], bt["dx"], bt["dy"], bt["nid"], m.nuis))
        w, cbar, wq = m.shape(0, h, m.m_band[bt["eid"]])
        assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)       # unit sum for every (pixel, lambda_q)
        nc = st.nc
        v = m.sv[:nc]
        wk = (w * wq[:, :, None]).sum(1)[:, :nc]                                      # band-integrated, core window
        cen = (wk[:, :, None] * (v[None] - cbar[:, None])).sum(1) / wk.sum(1, keepdim=True)
        assert cen.abs().max() < 1e-4, cen.abs().max()                                # zero centroid of the band-integrated CORE
        print(f"  unit sum err {float((w.sum(-1) - 1).abs().max()):.1e}, max |band core centroid| {float(cen.abs().max()):.1e} detector px (cbar up to {float(cbar.abs().max()):.2f})")


# ------------------------------------------------------------------------------------------------ 3
def centroid(L, X, Y):
    w = L.clamp(min=0)
    return torch.stack([(w * X).sum() / w.sum(), (w * Y).sum() / w.sum()])


def test_mu_shift():
    m, a, st = make_model()
    delta_psi(m)
    xs = torch.arange(14.0, 51.0)
    bt = batch_on_grid(xs, xs)
    X, Y = bt["x0"][:, 0], bt["x0"][:, 1]
    with torch.no_grad():
        lv = m.make_levels(0.0)
        L0 = m(bt, m.nuis, lv)["L"]
        mu = torch.tensor([0.7, -0.3])
        set_head_bias(m, "head_mu", (mu / m.mu_scale).tolist())
        out = m(bt, m.nuis, lv)
        L1 = out["L"]
        assert torch.allclose(out["mu"], mu[None].expand_as(out["mu"]), atol=1e-6)
        c0, c1 = centroid(L0, X, Y), centroid(L1, X, Y)
        d = (c1 - c0)
        # the data pixel at x0 sees the sky at x0 + mu: the image centroid moves by -mu
        assert torch.allclose(d, -mu, atol=0.06), d      # bicubic (Keys a=-0.75) phase error of a sigma = 3 px blob is ~0.04 px (see test_bicubic_accuracy)
        # and the value equals the sky read at x0 + mu (plumbing check): same bicubic sampler, delta kernel
        pts = (bt["x0"] + mu)[:, None, :]
        S = O.sample_levels(lv, pts, [(0, 0, 1)])[:, :, 0]                     # K,B
        ref = (S * m.Phi_band[0][:, None]).sum(0)
        assert torch.allclose(L1, ref, rtol=2e-3, atol=1e-5), float((L1 - ref).abs().max())
        print(f"  mu = {mu.tolist()}: image centroid moved by {d.tolist()} (expected {(-mu).tolist()})")


# ------------------------------------------------------------------------------------------------ 4
def second_moments(L, X, Y):
    w = L.clamp(min=0)
    cx, cy = (w * X).sum() / w.sum(), (w * Y).sum() / w.sum()
    return float((w * (X - cx) ** 2).sum() / w.sum()), float((w * (Y - cy) ** 2).sum() / w.sum())


def test_lambda_rotation():
    sky = lambda xx, yy: torch.exp(-0.5 * ((xx - 32) ** 2 + (yy - 32) ** 2) / 1.0 ** 2)  # noqa: E731
    m, a, st = make_model(sky_fn=sky, core_radius=3.0, core_pitch=0.25)
    # elongated Psi along detector x: Gaussian sigma (2.0, 0.6) in detector px
    with torch.no_grad():
        v = m.sv
        G = torch.exp(-0.5 * ((v[:, 0] / 2.0) ** 2 + (v[:, 1] / 0.6) ** 2)) / (2 * math.pi * 2.0 * 0.6)
        m.base.copy_(torch.log(torch.expm1(G.clamp(min=1e-9)))[None, : m.nk_max])
    xs = torch.arange(14.0, 51.0)
    bt = batch_on_grid(xs, xs)
    X, Y = bt["x0"][:, 0], bt["x0"][:, 1]
    with torch.no_grad():
        lv = m.make_levels(0.0)
        L0 = m(bt, m.nuis, lv)["L"]
        set_head_bias(m, "head_lam", [0.0, 0.0, math.atanh(0.5), 0.0])  # angle = pi/2
        L1 = m(bt, m.nuis, lv)["L"]
        vx0, vy0 = second_moments(L0, X, Y)
        vx1, vy1 = second_moments(L1, X, Y)
        assert vx0 > 3 * vy0 and vy1 > 3 * vx1
        assert abs(vx0 - vy1) / vx0 < 0.03 and abs(vy0 - vx1) / vy0 < 0.03
        print(f"  image variance (x, y): Lambda=I ({vx0:.2f}, {vy0:.2f}); Lambda=R(90deg) ({vx1:.2f}, {vy1:.2f})")


# ------------------------------------------------------------------------------------------------ 5
def test_gradients_reach_everything():
    m, a, st = make_model()
    randomise_heads(m, 0.2)
    with torch.no_grad():
        m.bank.add_(torch.randn_like(m.bank) * 0.2)
        m.nuis.z.add_(torch.randn_like(m.nuis.z) * 0.3)
    bt = batch_on_grid(torch.linspace(15, 45, 6), torch.linspace(15, 45, 6), eid=1)
    bt["d"] = torch.randn(36)
    lv = m.make_levels(0.7)
    out = m(bt, m.nuis, lv)
    nll, _ = O.nll_terms(bt["d"], out["pred"], bt["sp"], out["h"], out["pi"])
    nll.mean().backward()
    g = lambda p: float(p.grad.abs().sum())  # noqa: E731
    for name, p in [("sky", m.sky), ("trunk", m.trunk[0].weight), ("head_mu", m.head_mu[-1].weight), ("head_lam", m.head_lam[-1].weight),
                    ("head_a", m.head_a[-1].weight), ("shape_mlp", m.shape_mlp[-1].weight), ("bank", m.bank), ("base", m.base),
                    ("g_net", m.g_net[-1].weight), ("var_net", m.var_net[-1].weight), ("pi_net", m.pi_net[-1].weight), ("pe", m.nuis.pe)]:
        assert p.grad is not None and g(p) > 0, name
    assert float(m.nuis.z.grad[1].abs().sum()) > 0 and float(m.nuis.z.grad[0].abs().sum()) == 0   # z of the used exposure only
    print(f"  grad |sky| {g(m.sky):.2e}, |trunk| {g(m.trunk[0].weight):.2e}, |bank| {g(m.bank):.2e}, |z[1]| {float(m.nuis.z.grad[1].abs().sum()):.2e}")


# ------------------------------------------------------------------------------------------------ 6
def test_gauges():
    m, a, st = make_model(n_exp=3)
    eids = torch.arange(3)
    npix = torch.tensor([100.0, 200.0, 300.0])
    ref = torch.tensor([True, True, False])
    with torch.no_grad():
        mu_bar, a_ref, m2 = m.gauge_terms(eids, npix, ref)
        assert float(mu_bar.abs().max()) == 0.0 and float(a_ref) == 0.0           # at the gauge (zero-initialised heads)
        set_head_bias(m, "head_mu", [0.3, -0.2])
        set_head_bias(m, "head_a", [0.4])
        mu_bar, a_ref, _ = m.gauge_terms(eids, npix, ref)
        assert float(mu_bar.norm()) > 0.5 and abs(float(a_ref)) > 0.3             # away from the gauge: nonzero penalty
        # restore the gauge by subtracting the weighted mean: penalty zero again
        set_head_bias(m, "head_mu", (torch.tensor([0.3, -0.2]) - mu_bar / m.mu_scale).tolist())
        set_head_bias(m, "head_a", [0.0])
        mu_bar2, a_ref2, _ = m.gauge_terms(eids, npix, ref)
        assert float(mu_bar2.abs().max()) < 1e-6 and abs(float(a_ref2)) < 1e-6
        # blur gauge: monotone tie-breaker, vanishes for a delta kernel
        m2_init = float(m.gauge_terms(eids, npix, ref)[2].mean())
        delta_psi(m)
        m2_delta = float(m.gauge_terms(eids, npix, ref)[2].mean())
        assert m2_init > 0.5 and m2_delta < 1e-4
        print(f"  shift/flux penalties 0 at gauge ({float(mu_bar2.norm()):.1e}, {float(a_ref2):.1e}); core m2 {m2_init:.3f} px^2 -> {m2_delta:.1e} for a delta kernel")


# ------------------------------------------------------------------------------------------------ 7
def test_bicubic_accuracy():
    res = {}
    rng = np.random.default_rng(0)
    yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    for sig in (0.7, 1.0, 1.5, 2.0, 3.0):
        img = torch.exp(-0.5 * ((xx - 32) ** 2 + (yy - 32) ** 2) / sig ** 2)[None]
        p = torch.tensor(rng.uniform(26, 38, (4000, 2)), dtype=torch.float32)
        got = O.sample_levels([img], p[:, None, :], [(0, 0, 1)])[0, :, 0]
        ref = torch.exp(-0.5 * ((p[:, 0] - 32) ** 2 + (p[:, 1] - 32) ** 2) / sig ** 2)
        res[sig] = float((got - ref).abs().max())                                  # relative to the peak value 1
    print("  bicubic max |error| / peak for a Gaussian of sigma 0.7, 1.0, 1.5, 2.0, 3.0 sky px: " + ", ".join(f"{res[s]:.3f}" for s in res))
    assert res[1.0] < 0.15 and res[2.0] < 0.04 and res[3.0] < 0.02
    return res


# ------------------------------------------------------------------------------------------------ 8
def test_noise_and_no_data_leak():
    m, a, st = make_model()
    randomise_heads(m, 0.3)
    bt = batch_on_grid(torch.linspace(15, 45, 5), torch.linspace(15, 45, 5))
    assert "d" not in bt
    lv = m.make_levels(0.0)
    out = m(bt, m.nuis, lv)                                           # works without any data value
    assert float(out["pi"].detach().max()) <= O.PI_MAX and float(out["h"].detach().abs().max()) <= 3.0
    d = out["pred"] + torch.tensor([0.0, 0.0, 1.0, 5.0, 100.0] * 5)
    out = {k: v.detach() for k, v in out.items()}
    nll, pout = O.nll_terms(d, out["pred"], bt["sp"], torch.zeros(25), torch.full((25,), 0.01))
    k0, k1, k5, k100 = [torch.arange(25)[torch.arange(25) % 5 == j] for j in (0, 2, 3, 4)]
    assert torch.all(pout[k0] < 1e-3) and torch.all(pout[k1] < 1e-3)      # 0 and 1 sigma: inliers
    assert torch.all(pout[k5] > 0.5) and torch.all(pout[k100] > 0.99)     # 5 sigma at pi = 0.01: outlier already (U density 1/(200 sigma)); 100 sigma: surely
    # mixture NLL equals the direct formula
    sig = bt["sp"]
    ref = -torch.log(0.99 * torch.exp(-0.5 * ((d - out["pred"]) / sig) ** 2) / (sig * math.sqrt(2 * math.pi)) + 0.01 / (O.U_WIDTH * sig))
    assert torch.allclose(nll, ref, rtol=1e-4, atol=1e-5)
    print(f"  pi <= {float(out['pi'].detach().max()):.3f}; outlier posterior (pi = 0.01) at 0, 1, 5, 100 sigma: " + ", ".join(f"{float(pout[i]):.3g}" for i in (0, 2, 3, 4)))


# ------------------------------------------------------------------------------------------------ 9
def test_visit_codes():
    import pandas as pd
    for d, nv in (("nircam", 1), ("hst_all", 2), ("nircam_all", 1)):
        fs = sorted(f for f in os.listdir(os.path.join(ROOT, "testdata", d)) if f.endswith(".fits"))
        ep, vis = O.assign_codes(fs, [0.0] * len(fs))
        assert len(set(vis)) == nv, (d, len(set(vis)))
    man = os.path.join(ROOT, "hst", "hst_imaging_manifest.csv")
    if os.path.exists(man):
        m = pd.read_csv(man)
        ep, vis = O.assign_codes(m.product_filename.tolist(), m.t_start_mjd.tolist())
        m["ep"], m["vis"] = ep, vis
        assert int((m.groupby("vis").ep.nunique() > 1).sum()) == 0
        assert int((m.groupby("vis").proposal_id.nunique() > 1).sum()) == 0
        print(f"  visit codes: HST manifest {len(m)} exposures -> {len(set(ep))} epochs, {len(set(vis))} visits; real test names consistent")
    assert O.visit_key("hst_16729_79_wfc3_ir_f105w_iepl79pm_flt.fits") == "hst16729_79"
    assert O.visit_key("jw02736001001_02101_00002_nrcb3_cal.fits") == "jw02736001001" and O.visit_key("iczgzbmaq_flt.fits") == "iczgzb"


# ------------------------------------------------------------------------------------------------ 10: NEW flags (default off)
def make_multi(flags=(), dets=(0, 0, 1, 1, 0, 1), bands=(0, 0, 0, 1, 1, 1), chans=None, core_radius=2.0, core_pitch=0.5, nang=12, wing_arcsec=1.2, Q=3, seed=0):
    """OpField with several detectors / bands (and optionally channels): exposure i has det dets[i], band bands[i]; band b is in channel chans[b]."""
    torch.manual_seed(seed)
    n_exp = len(dets)
    nb = max(bands) + 1
    chans = list(chans) if chans is not None else [0] * nb
    a = O.parse_args(["--out", "/tmp/unused", "--core-radius", str(core_radius), "--core-pitch", str(core_pitch), "--wing-ratio", "1.5", "--wing-nang", str(nang),
                      "--wing-arcsec", str(wing_arcsec), "--hidden", "16", "--n-bank", "4", "--shape-hidden", "16", "--scale", "0.1", "--spec-q", str(Q)] + list(flags))
    names = [f"c{c}" for c in range(max(chans) + 1)]
    reg = O.OpReg([f"b{k}" for k in range(nb)], [1.5 + 0.5 * k for k in range(nb)], ["A"] * nb, [names[c] for c in chans], [False] * nb)
    band = torch.tensor(bands)
    meta = dict(band=band, inst=torch.zeros(n_exp, dtype=torch.long), chan=torch.tensor([reg.chan_of_band[b] for b in bands]), det=torch.tensor(dets),
                epoch=torch.zeros(n_exp), ltexp=torch.zeros(n_exp), roll=torch.tensor([[1.0, 0.0, 1.0]]).repeat(n_exp, 1), lscale=torch.zeros(n_exp),
                ro=torch.zeros(n_exp, 2), J=torch.eye(2).repeat(n_exp, 1, 1), unit=torch.full((n_exp,), 0.1), b0=torch.zeros(n_exp), stripe_exp=torch.zeros(n_exp))
    yy, xx = torch.meshgrid(torch.arange(64).float(), torch.arange(64).float(), indexing="ij")
    A0 = torch.exp(-0.5 * ((xx - 32) ** 2 + (yy - 30) ** 2) / 9.0)[None].repeat(12, 1, 1)
    basis = V.RBFSpectralBasis(12)
    lam_nodes = torch.stack([torch.linspace(1.45 + 0.5 * k, 1.55 + 0.5 * k, Q) for k in range(nb)])
    w_nodes = torch.full((nb, Q), 1.0 / Q)
    Phi = basis(lam_nodes.flatten()).view(nb, Q, 12).detach()
    scale_ch = [{"sky_per_det_px": 1.0, "arcsec_per_det_px": 0.1 * (1 + c)} for c in range(len(reg.channels))]
    st = O.make_stencil(a, max(a.wing_arcsec / s_["arcsec_per_det_px"] for s_ in scale_ch))
    m = O.OpField(a, reg, meta, st, A0, Phi, w_nodes, lam_nodes, max(dets) + 1, scale_ch, DEV, det_names=[f"DET{d}" for d in range(max(dets) + 1)])
    return m, a, st


def multi_batch(m, eids, n_each=6):
    pts = []
    for e in eids:
        xs = torch.linspace(20, 44, n_each)
        X, Y = torch.meshgrid(xs, xs, indexing="ij")
        pts.append((e, torch.stack([X.reshape(-1), Y.reshape(-1)], 1)))
    x0 = torch.cat([p for _, p in pts])
    eid = torch.cat([torch.full((p.shape[0],), e, dtype=torch.long) for e, p in pts])
    n = x0.shape[0]
    z = torch.zeros(n)
    return dict(eid=eid, nid=eid.clone(), x0=x0, dx=z, dy=z, sp=torch.ones(n), row=torch.zeros(n, dtype=torch.long), col=torch.zeros(n, dtype=torch.long))


def test_new_flags_default_off():
    a = O.parse_args(["--out", "/tmp/unused"])
    assert (a.base_per, a.shape_head_per, a.wings, a.base_table, a.base_l2, a.eval_only, a.dump_heldout_pixels, a.wing_nang) == ("channel", "shared", "logpolar", "none", 1e-4, False, False, 24)
    st0 = O.Stencil(3.0, 0.25, 1.25, 24, 97.0)
    st1 = O.make_stencil(O.parse_args(["--out", "/tmp/unused"]), 97.0)
    assert np.array_equal(st0.v, st1.v) and np.array_equal(st0.dv, st1.dv) and st0.wings == "logpolar"
    m, a, st = make_multi()
    assert m.base.shape[0] == 1 and isinstance(m.bank, torch.nn.Parameter) and m.n_rows == 1
    assert not any(k.startswith("m_brow") for k in m.state_dict()), "state dict keys must stay those of the original model"
    print("  defaults of the new flags = original model; state_dict has no new keys")


def test_cartesian_stencil():
    st = O.Stencil(10.0, 0.5, 1.25, 24, 97.0, wings="cartesian", cart_steps=O.parse_cart_steps("24:1,48:2,inf:4"))
    assert st.wings == "cartesian" and st.n_rings == 0 and np.all(np.diff(st.r) >= -1e-9)
    assert abs(st.dv.sum() - math.pi * 97.0 ** 2) / (math.pi * 97.0 ** 2) < 0.02, st.dv.sum()
    nk = {R: st.nk_for(R) for R in (23.4, 48.0, 97.0)}
    assert nk[23.4] < nk[48.0] < nk[97.0] <= len(st) and st.nc == int((st.r[: st.nc] <= 10 + 1e-9).sum())
    for pit, lo, hi in ((0.5, 0, 10), (1.0, 10, 24), (2.0, 24, 48), (4.0, 48, 98)):
        sel = np.isclose(st.pitch, pit)
        assert st.r[sel].min() >= lo - 1e-6 and st.r[sel].max() <= hi + 1e-6, (pit, st.r[sel].min(), st.r[sel].max())
    lp = O.Stencil(10.0, 0.5, 1.25, 24, 97.0)
    lp72 = O.Stencil(10.0, 0.5, 1.25, 72, 97.0)
    for s_, R in ((0.775, 97.0), (1.575, 48.0), (3.2, 23.4)):
        lev = st.levels_for(st.nk_for(R), s_)
        assert np.all(np.diff(lev) >= 0)
    print(f"  nodes (R = 97 / 48 / 23.4 px, i.e. NIRCam SW / LW / WFC3-IR at 3 arcsec): cartesian {nk[97.0]} / {nk[48.0]} / {nk[23.4]}; log-polar 24 az: "
          f"{lp.nk_for(97.0)} / {lp.nk_for(48.0)} / {lp.nk_for(23.4)}; log-polar 72 az: {lp72.nk_for(97.0)} / {lp72.nk_for(48.0)} / {lp72.nk_for(23.4)}")


def test_cartesian_model_unit_sum_and_zero_centroid():
    m, a, st = make_multi(["--wings", "cartesian", "--cart-steps", "4:1,inf:2"], core_radius=2.0, core_pitch=0.5, wing_arcsec=1.6)
    randomise_heads(m, 0.4)
    with torch.no_grad():
        m.bank.add_(torch.randn_like(m.bank) * 0.4)
        bt = multi_batch(m, [0, 3])
        h = m.trunk(m.ctx_input(bt["eid"], bt["dx"], bt["dy"], bt["nid"], m.nuis))
        w, cbar, wq = m.shape(0, h, m.m_band[bt["eid"]], m.m_brow[bt["eid"]])
        assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)
        nc = st.nc
        wk = (w * wq[:, :, None]).sum(1)[:, :nc]
        cen = (wk[:, :, None] * (m.sv[:nc][None] - cbar[:, None])).sum(1) / wk.sum(1, keepdim=True)
        assert cen.abs().max() < 1e-4
        out = m(bt, m.nuis, m.make_levels(0.0))
        assert torch.isfinite(out["pred"]).all()


def test_base_per_rows_and_grads():
    for per, n_expect in (("channel", 1), ("instrument", 1), ("detector", 2), ("detector_band", 4)):
        m, a, st = make_multi(["--base-per", per])
        assert m.n_rows == n_expect and m.base.shape[0] == n_expect, (per, m.n_rows)
        randomise_heads(m, 0.2)
        bt = multi_batch(m, [0, 3])          # exposure 0: det 0 band 0; exposure 3: det 1 band 1
        bt["d"] = torch.randn(bt["eid"].numel())
        out = m(bt, m.nuis, m.make_levels(0.5))
        nll, _ = O.nll_terms(bt["d"], out["pred"], bt["sp"], out["h"], out["pi"])
        nll.mean().backward()
        gr = m.base.grad.abs().sum(1)
        used = sorted(set(m.m_brow[[0, 3]].tolist()))
        assert all(float(gr[r]) > 0 for r in used) and all(float(gr[r]) == 0 for r in range(m.n_rows) if r not in used), (per, gr)
        pc = m.param_counts()
        assert pc["base"] == m.base.numel() and pc["total"] == sum(p.numel() for p in m.parameters())
        print(f"  --base-per {per}: {m.n_rows} rows ({[m.row_label(r) for r in range(m.n_rows)]}), base params {pc['base']}, rows used by the batch {used}")


def test_untrained_cells_are_aliased():
    m, a, st = make_multi(["--base-per", "detector_band"])
    # cells: (0,0) e0,e1,e4? bands=(0,0,0,1,1,1), dets=(0,0,1,1,0,1): cells (0,0):e0,e1  (1,0):e2  (1,1):e3,e5  (0,1):e4
    lab = {r: m.row_label(r) for r in range(m.n_rows)}
    moves = m.set_train_rows([0, 1, 3, 4, 5])      # exposure 2 = the only one of cell DET1/b0 is held out
    assert len(moves) == 1 and moves[0] == "DET1/b0 -> DET0/b0", moves
    assert int(m.m_brow[2]) == int(m.m_brow[0])
    # idempotent and deterministic
    assert m.set_train_rows([0, 1, 3, 4, 5]) == moves
    print("  untrained (detector, band) cell aliased:", moves)


def test_shape_head_per_channel():
    m, a, st = make_multi(["--shape-head-per", "channel"], bands=(0, 0, 0, 1, 1, 1), chans=(0, 1), wing_arcsec=2.4)
    assert isinstance(m.bank, torch.nn.ParameterList) and len(m.bank) == 2 and isinstance(m.shape_mlp, torch.nn.ModuleList) and m.base.shape[0] == 2
    assert m.bank[0].shape[1] == m.nk_ch[0] and m.bank[1].shape[1] == m.nk_ch[1]
    randomise_heads(m, 0.0)
    with torch.no_grad():
        for mod in m.shape_mlp:
            for p in mod.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        for b in m.bank:
            b.add_(torch.randn_like(b) * 0.3)
    bt = multi_batch(m, [0])                # channel 0 only
    bt["d"] = torch.randn(bt["eid"].numel())
    out = m(bt, m.nuis, m.make_levels(0.5))
    nll, _ = O.nll_terms(bt["d"], out["pred"], bt["sp"], out["h"], out["pi"])
    nll.mean().backward()
    g = lambda p: float(p.grad.abs().sum()) if p.grad is not None else 0.0  # noqa: E731
    assert g(m.bank[0]) > 0 and g(m.bank[1]) == 0 and g(m.shape_mlp[0][-1].weight) > 0 and g(m.shape_mlp[1][-1].weight) == 0
    pc = m.param_counts()
    m0, _, _ = make_multi(bands=(0, 0, 0, 1, 1, 1), chans=(0, 1), wing_arcsec=2.4)
    print(f"  --shape-head-per channel: bank + shape MLP params {pc['bank'] + pc['shape_mlp']} (shared: {m0.param_counts()['bank'] + m0.param_counts()['shape_mlp']})")


def test_base_init_from_e2():
    import tempfile
    m, a, st = make_multi(["--base-per", "detector_band", "--n-bank", "4"], core_radius=3.0, core_pitch=0.5, wing_arcsec=1.5)
    os_, N = 4, 2 * 24 * 4 + 1
    ax = (np.arange(N) - N // 2) / os_
    xx, yy = np.meshgrid(ax, ax)
    K = np.exp(-0.5 * ((xx - 0.3) ** 2 / 1.4 ** 2 + yy ** 2 / 0.9 ** 2))
    K /= K.sum()
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "explicit_lib_3d_psf_groups.npy"), {"detector_frame": {"DET0/b0": {"oversample": 4, "learned": K, "table": K, "library": K}}}, allow_pickle=True)
        rep = m.init_base_from_e2(tmp, "learned")
    assert rep["matched"] == ["DET0/b0"] and len(rep["unmatched"]) == 3
    r0 = [r for r in range(m.n_rows) if m.row_label(r) == "DET0/b0"][0]
    assert torch.equal(m.base0, m.base.detach())
    # the base density x cell area is the table's mass inside each node cell: brute force on a 8x upsampled table (sub-sample centres inside the box)
    with torch.no_grad():
        raw = (torch.nn.functional.softplus(m.base[r0]) * m.sdv).numpy()
    v, side = m.sv.numpy(), np.sqrt(m.sdv.numpy())
    up = 8
    Kf = np.kron(K, np.ones((up, up))) / up ** 2
    axf = ((np.arange(N * up) + 0.5) / (up * os_)) - (N // 2 + 0.5) / os_      # sub-sample centres in detector px
    XF, YF = np.meshgrid(axf, axf)
    err = []
    for k in range(0, min(len(v), 150), 7):
        sel = (np.abs(XF - v[k, 0]) < side[k] / 2) & (np.abs(YF - v[k, 1]) < side[k] / 2)
        err.append(abs(Kf[sel].sum() - raw[k]) / max(Kf[sel].sum(), 1e-4))
    assert max(err) < 0.03, err
    mass = raw / raw.sum()
    cx = float((mass[: st.nc] * v[: st.nc, 0]).sum() / mass[: st.nc].sum())
    assert abs(cx - 0.3) < 0.1, cx
    print(f"  --base-init-from: matched DET0/b0, 3 cells kept at the Gaussian start; node masses = table cell masses (max rel err {max(err):.4f}); core centroid x {cx:.3f} (table 0.3); table mass inside the stencil {rep['mass_in_stencil']['DET0/b0']:.3f}")


def test_base_l2_zero_allowed():
    a = O.parse_args(["--out", "/tmp/unused", "--base-l2", "0"])
    assert a.base_l2 == 0.0
    m, _, _ = make_multi(["--base-l2", "0"])
    with torch.no_grad():
        m.base.add_(1.0)
    assert float(m.prior_terms(torch.tensor([0]))["base"]) > 0.99 and a.base_l2 * float(m.prior_terms(torch.tensor([0]))["base"]) == 0.0


def test_lambda_gauge_jacobian():
    """--lambda-gauge jacobian: Lambda == I, M == J EXACTLY; head_lam unused (no gradient); default (learned) = the original formula bit for bit."""
    torch.manual_seed(3)
    Jr = torch.tensor([[0.31, -0.74], [0.80, 0.28]])
    ml, a_l, _ = make_multi()
    mj, a_j, _ = make_multi(["--lambda-gauge", "jacobian"])
    assert a_l.lambda_gauge == "learned" and ml.lam_gauge == "learned" and mj.lam_gauge == "jacobian"
    for m_ in (ml, mj):
        randomise_heads(m_, 0.5)                          # head_lam non-zero: a learned Lambda != I
        m_.m_J.copy_(Jr[None] * (1 + 0.1 * torch.arange(m_.m_J.shape[0]))[:, None, None])
    mj.load_state_dict(ml.state_dict(), strict=True)       # same keys: a learned-gauge state dict loads into the jacobian model
    bt = multi_batch(ml, [0, 3, 5])
    h = ml.trunk(ml.ctx_input(bt["eid"], bt["dx"], bt["dy"], bt["nid"], ml.nuis))
    cx = (bt["dx"], bt["dy"], bt["nid"], ml.nuis)
    mu_l, Lc_l, M_l, a_l_ = ml.heads(h, bt["eid"], cx)
    ref = torch.einsum("bij,bjk->bik", ml.m_J[bt["eid"]], O.lambda_matrix(ml.head_lam(h)))
    assert torch.equal(M_l, ref) and not torch.allclose(Lc_l, torch.eye(2).expand_as(Lc_l), atol=1e-3)   # learned: original formula, Lambda != I
    h2 = mj.trunk(mj.ctx_input(bt["eid"], bt["dx"], bt["dy"], bt["nid"], mj.nuis))
    mu_j, Lc_j, M_j, a_j_ = mj.heads(h2, bt["eid"], (bt["dx"], bt["dy"], bt["nid"], mj.nuis))
    assert torch.equal(M_j, mj.m_J[bt["eid"]]) and torch.equal(Lc_j, torch.eye(2).expand_as(Lc_j))           # M == J exactly
    assert torch.equal(mu_j, mu_l) and torch.equal(a_j_, a_l_)                                              # mu and a untouched
    # gradients: nothing reaches head_lam under the jacobian gauge (also through the lc-l2 term); everything else still does
    bt = multi_batch(mj, [0, 3, 5], n_each=4)
    bt["d"] = torch.randn(bt["eid"].numel())
    out = mj(bt, mj.nuis, mj.make_levels(0.7))
    assert torch.count_nonzero(out["lraw"]) == 0
    nll, _ = O.nll_terms(bt["d"], out["pred"], bt["sp"], out["h"], out["pi"])
    (nll.mean() + 1.0 * (torch.tanh(out["lraw"]) ** 2).mean()).backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in mj.head_lam.parameters()), "gradient reached head_lam"
    for name, p in (("sky", mj.sky), ("trunk", mj.trunk[0].weight), ("head_mu", mj.head_mu[-1].weight), ("head_a", mj.head_a[-1].weight), ("shape_mlp", mj.shape_mlp[-1].weight)):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, name
    # the learned gauge DOES train head_lam
    bt = multi_batch(ml, [0, 3, 5], n_each=4)
    bt["d"] = torch.randn(bt["eid"].numel())
    out = ml(bt, ml.nuis, ml.make_levels(0.7))
    nll, _ = O.nll_terms(bt["d"], out["pred"], bt["sp"], out["h"], out["pi"])
    nll.mean().backward()
    assert float(ml.head_lam[-1].weight.grad.abs().sum()) > 0
    print("  lambda-gauge jacobian: M == J exactly, Lambda = I, no gradient to head_lam; learned gauge = original formula bitwise")


def test_lam_in_band_features():
    m, a, st = make_multi(["--lam-in-band"], Q=5)
    m0, _, _ = make_multi(Q=5)
    assert m.lamb.shape == (2, 5, 4) and "lamb" not in m.state_dict() and "lamb" not in m0.state_dict()
    t = m.lamb[..., 0]
    for b in range(2):
        assert abs(float(t[b].min()) + 0.5) < 1e-6 and abs(float(t[b].max()) - 0.5) < 1e-6 and abs(float(t[b, 2])) < 1e-6   # [-0.5, 0.5] per band, 0 at its midpoint
    assert float(t.min()) >= -0.5 - 1e-6 and float(t.max()) <= 0.5 + 1e-6
    assert torch.allclose(m.lamb[..., 1], t ** 2) and torch.allclose(m.lamb[..., 2], torch.sin(math.pi * t)) and torch.allclose(m.lamb[..., 3], torch.cos(math.pi * t))
    assert m.shape_mlp[0].in_features == m.H + 9 and m0.shape_mlp[0].in_features == m0.H + 5       # default input untouched
    # the band-wise t is the same for two bands whose nodes are offset copies (position inside the band, not the global lambda)
    assert torch.allclose(m.lamb[0], m.lamb[1], atol=1e-5)
    # the feature reaches the shape: with a non-zero shape MLP the kernel differs between quadrature nodes more than without it? -> it enters the input
    randomise_heads(m, 0.4)
    with torch.no_grad():
        m.shape_mlp[-1].weight.add_(torch.randn_like(m.shape_mlp[-1].weight) * 0.3)
    bt = multi_batch(m, [0], n_each=2)
    h = m.trunk(m.ctx_input(bt["eid"], bt["dx"], bt["dy"], bt["nid"], m.nuis))
    w1, _, _ = m.shape(0, h, m.m_band[bt["eid"]], m.m_brow[bt["eid"]])
    m.lamb.zero_()
    w2, _, _ = m.shape(0, h, m.m_band[bt["eid"]], m.m_brow[bt["eid"]])
    assert not torch.allclose(w1, w2) and torch.allclose(w1.sum(-1), torch.ones_like(w1.sum(-1)), atol=1e-5)
    print("  lam-in-band: t in [-0.5, 0.5] per band (0 at the band midpoint), 4 extra shape-MLP columns, no new state-dict keys")


def run_smoke(tmp=None, extra=()):
    out = tmp or os.path.join(ROOT, "runs", "opfield_smoke_test")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "opfield.py"), "--out", out, "--data-dir", os.path.join(ROOT, "testdata", "nircam"),
           "--size", "128", "--quick", "--no-fallback", "--no-holdout", "--score-all", "--boot-b", "50", "--stripes", "1", "--svo-cache",
           os.path.join(ROOT, "testdata", "svo_cache"), "--budget-min", "10", "--steps", "300", "--nuis-steps", "20", "--batch", "512", "--log-every", "50"] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    M = json.load(open(os.path.join(out, "metrics.json")))
    m = M["metrics"]
    assert np.isfinite(m["heldout_chi2"]["opfield"]) and np.isfinite(m["heldout_robust"]["opfield"]) and np.isfinite(m["heldout_chi2_sources"]["opfield"])
    bs = m["heldout_bootstrap"]["opfield"]
    for k in ("chi2_all", "chi2_src", "robust_all", "robust_src", "flux_err"):
        assert np.isfinite(bs[k]["est"]) and bs[k]["p16"] <= bs[k]["p84"]
    kd = M["kernel_diagnostics"]["cells"]
    assert kd and all(np.isfinite(v["exp_a_mean"]) and np.isfinite(v["exp_a_std"]) and all(np.isfinite(x) for x in v["ee_core_arcsec"].values()) for v in kd.values())
    assert len(M["approximations"]) >= 10 and M["gauges"] and os.path.exists(os.path.join(out, "cubes", "opfield_sky.fits"))
    print(f"  smoke: exit 0; chi2 all {m['heldout_chi2']['opfield']:.3f} src {m['heldout_chi2_sources']['opfield']:.3f} robust {m['heldout_robust']['opfield']:.3f} "
          f"src {m['heldout_robust_sources']['opfield']:.3f}; bootstrap flux_err {bs['flux_err']['est']:.3f}; {M['train_steps']} steps")
    return M


@pytest.mark.slow
def test_smoke_run():
    run_smoke()


@pytest.mark.slow
def test_smoke_run_lambda_jacobian():
    run_smoke(os.path.join(ROOT, "runs", "opfield_smoke_jac"), ["--lambda-gauge", "jacobian"])


@pytest.mark.slow
def test_smoke_run_lambda_jacobian_lam_in_band():
    run_smoke(os.path.join(ROOT, "runs", "opfield_smoke_jac_band"), ["--lambda-gauge", "jacobian", "--lam-in-band"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and not name.startswith("test_smoke_run"):
            print(name)
            fn()
    if "--smoke" in sys.argv:
        print("test_smoke_run")
        run_smoke()
        print("test_smoke_run_lambda_jacobian(+lam_in_band)")
        run_smoke(os.path.join(ROOT, "runs", "opfield_smoke_jac"), ["--lambda-gauge", "jacobian"])
        run_smoke(os.path.join(ROOT, "runs", "opfield_smoke_jac_band"), ["--lambda-gauge", "jacobian", "--lam-in-band"])
    print("ALL OK")
