"""Unit tests of the learned position shift per (detector, filter) pair (--filter-shift learned).  CPU, well under a minute.
python tests/test_filter_shift.py  (or pytest)."""
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fieldop.cli import default_config, make_sky  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402


def small(mode="off", **kw):
    torch.manual_seed(0)
    insts = [SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", 0.12, 20, 6.5, 0.10, 2.4, filters=[("B1", 2.0, 2.8)])]
    exps, fset, meta, truth, _ = make_field(seed=1, n_per_filter=5, n_val=1, n_held=1, insts=insts, centre_range=0.4, rot_choices=(0, 1, 2, 3),
                                            patch_px=12, truth_half=1.0)
    c = dict(sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.8, region_px=12, n_lambda=3, k_hidden=16, pattern_grid=0, stripes=0,
             filter_shift=mode)
    c.update(kw)
    cfg = default_config(**c)
    sky, _ = make_sky(exps, fset, cfg, 1.6, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    model.attach_filter_shift(exps, cfg, fset_names=fset.names, det_names=["A", "B"])
    return model, exps, cfg


def render_m(model, e):
    P = {k: v.detach() for k, v in model.prepare().items()}
    pix = e.pix((0, 12, 0, 12))
    return model.render_batch([(e.eid, pix)], P)[0]["m"]


def test_off_is_noop_on_the_forward_model():
    m0, exps, _ = small("off")
    assert not m0.fs_on and m0.fshift is None
    assert not any(k in m0.state_dict() for k in ("fshift", "e_pair", "e_J"))       # no new state: the default state hash cannot change
    m1, _, _ = small("learned")                                                       # same seed, same weights; shifts start at zero
    assert m1.fs_on and torch.count_nonzero(m1.fshift) == 0 and m1.fshift.shape == (3, 2)
    e = exps[0]
    with torch.no_grad():
        assert torch.equal(render_m(m0, e), render_m(m1, e))                          # zero shift: identical forward model, bit for bit
        m1.fshift.data[:] = 1.0
        assert not torch.equal(render_m(m0, e), render_m(m1, e))
    # the operator group is unchanged by the switch (the shift has its own optimiser)
    assert len(m0.param_groups()[0]) == len(m1.param_groups()[0])


def test_position_convention_and_gradient():
    m, exps, _ = small("learned")
    e = exps[0]
    ei = torch.as_tensor(e.eid)
    P = {k: v.detach() for k, v in m.prepare().items()}
    pix = e.pix((0, 12, 0, 12))
    p0 = m.positions(ei, pix["X"], pix["Y"], pix["xi"], P).detach()
    s = torch.tensor([1.5, -0.5])
    with torch.no_grad():
        m.fshift.data[m.e_pair[ei]] = s
    p1 = m.positions(ei, pix["X"], pix["Y"], pix["xi"], P).detach()
    J = torch.as_tensor(np.asarray(e.extra["J"]), dtype=torch.float32)
    assert torch.allclose(p1 - p0, (J @ s).expand_as(p0), atol=1e-6)               # sky displacement = J s: pixel (col, row) is read at (col + sx, row + sy)
    # the pixel at column c + 1 with zero shift sits where the pixel at column c sits with shift (+1, 0)
    # gradient: the loss reaches the shift through the sky read, nonzero and finite, and only in the pair of the exposure
    m.fshift.data.zero_()
    m.fshift.requires_grad_(True)
    out = m.render_batch([(e.eid, pix)], P)[0]["m"]
    (out * torch.linspace(0.5, 1.5, out.numel())).sum().backward()
    g = m.fshift.grad
    assert torch.isfinite(g).all() and g[m.e_pair[ei]].abs().sum() > 0 and (g.abs().sum(1) > 0).sum() == 1


def test_projection_removes_exactly_the_common_sky_translation():
    m, exps, _ = small("learned")
    tr = m.ref_eids
    g = torch.Generator().manual_seed(3)
    m.fshift.data = torch.randn(3, 2, generator=g) * 1.5
    m.affine.data[:, 4:6] = torch.randn(m.E, 2, generator=g) * 3.0
    sc = m.aff_scale[4:6]

    def disp(ids):
        return torch.einsum("nij,nj->ni", m.e_J[ids], m.fshift[m.e_pair[ids]]) + m.affine[ids, 4:6] * sc
    snap = (m.fshift.detach().clone(), m.affine.detach().clone())
    d0 = disp(tr)
    m.project_filter_shift(tr)
    d1 = disp(tr)
    assert torch.allclose(d1, d0 - d0.mean(0), atol=2e-6)                              # exactly the common translation removed, nothing else moved
    assert d1.mean(0).abs().max() < 2e-6
    for p in range(3):                                                                  # translations are J-orthogonal within each pair
        ids = tr[m.e_pair[tr] == p]
        r = torch.einsum("nij,ni->j", m.e_J[ids], m.affine[ids, 4:6] * sc)
        assert r.abs().max() < 2e-5, (p, r)
    a, b = m.fshift.detach().clone(), m.affine.detach().clone()
    m.project_filter_shift(tr)                                                          # idempotent
    assert torch.allclose(a, m.fshift, atol=1e-5) and torch.allclose(b, m.affine, atol=1e-4)
    # gauge invariance: add one common sky translation c to every training exposure (in the sky tangent plane); the projection gives the same state
    m.fshift.data, m.affine.data = snap[0].clone(), snap[1].clone()
    c = torch.tensor([0.37, -0.21])
    m.affine.data[tr, 4:6] += c / sc
    m.project_filter_shift(tr)
    assert torch.allclose(m.fshift, a, atol=1e-5) and torch.allclose(m.affine[tr], b[tr], atol=1e-4)
    # pairs differ in pixel scale and orientation: the shifts of the pairs agree in the sky plane only through J (A: 0.10, B: 0.12 arcsec per pixel)
    assert abs(float(torch.det(m.e_J[tr[0]]).abs().sqrt()) - float(torch.det(m.e_J[tr[-1]]).abs().sqrt())) > 1e-3


def test_held_out_use_the_fitted_shift_of_their_pair_fixed():
    from fieldop.train import fit_heldout
    m, exps, cfg = small("learned", held_steps=3, held_reg_steps=2, batch_patches=2, held_rounds=1)
    held = [e for e in exps if e.split == "held"]
    assert len(held) == 3
    m.fshift.data = torch.tensor([[0.4, -0.2], [1.1, 0.7], [-0.6, 0.3]])
    for e in held:
        ei = torch.as_tensor(e.eid)
        P = {k: v.detach() for k, v in m.prepare().items()}
        pix = e.pix((0, 12, 0, 12))
        got = m.positions(ei, pix["X"], pix["Y"], pix["xi"], P).detach()
        keep = m.fshift.detach().clone()
        m.fshift.data.zero_()
        base = m.positions(ei, pix["X"], pix["Y"], pix["xi"], P).detach()              # same exposure, same (zero) affine and distortion, no shift
        m.fshift.data = keep
        want = base + m.e_J[ei] @ m.fshift[m.e_pair[ei]].detach()
        assert torch.allclose(got, want, atol=1e-6)
    before = m.fshift.detach().clone()
    fit_heldout(m, exps, [e.eid for e in held], cfg, log=lambda s: None)
    assert torch.equal(before, m.fshift.detach())                                      # fixed during the held-out nuisance fit
    assert m.fshift.requires_grad


if __name__ == "__main__":
    import time
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            t = time.perf_counter()
            fn()
            print(f"{name}: ok ({time.perf_counter() - t:.1f} s)")
