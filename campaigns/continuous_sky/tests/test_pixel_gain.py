"""Unit tests of the per-pixel gain map (--pixel-gain, F43-F45). CPU, a few seconds.  python tests/test_pixel_gain.py  (or pytest)."""
import math
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fieldop.cli import default_config, make_sky  # noqa: E402
from fieldop.model import FieldOperator, PixelGain, poly_basis  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402


def small(pixel_gain="off", inject=0.0, **kw):
    torch.manual_seed(0)
    insts = [SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", 0.12, 20, 6.5, 0.10, 2.4, filters=[("B1", 2.0, 2.8)])]
    exps, fset, meta, truth, _ = make_field(seed=1, n_per_filter=4, n_val=1, n_held=1, insts=insts, centre_range=0.4, rot_choices=(0, 1, 2, 3),
                                            patch_px=12, truth_half=1.0, pix_gain_rms=inject)
    c = dict(sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.8, region_px=12, n_lambda=3, k_hidden=16, pattern_grid=4, stripes=1,
             pixel_gain=pixel_gain)
    c.update(kw)
    cfg = default_config(**c)
    sky, _ = make_sky(exps, fset, cfg, 1.6, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    model.attach_pixel_gain(exps, cfg, fset_names=fset.names, det_names=["A", "B"])
    return model, exps, cfg, truth


def test_off_is_inert():
    m0, exps, cfg, _ = small("off")
    assert m0.pixgain is None and m0.gain_key is None
    names = [n for n, _ in m0.named_parameters()]
    assert not any("pixgain" in n for n in names)
    P = m0.prepare()
    e = exps[0]
    pix = e.pix((0, 12, 0, 12))
    out = m0.render_batch([(e.eid, pix)], P)[0]
    assert torch.isfinite(out["m"]).all()


def test_projection_gauge():
    torch.manual_seed(1)
    obs = np.zeros((30, 40), bool)
    obs[3:27, 5:38] = True
    obs[10:14, 10:30] = False                     # a hole of unobserved pixels inside the box
    pg = PixelGain(["X"], [(100, 200, 30, 40)], [obs], [np.ones((30, 40))], [np.ones((30, 40))], 2, "cpu")
    pg.logg.data.normal_()
    pg.project()
    r = pg.gauge_residual()
    assert r["max_projected_component_rms"] < 1e-6, r
    assert r["max_abs_unobserved"] == 0.0
    before = pg.logg.detach().clone()
    pg.project()                                  # idempotent
    assert torch.allclose(before, pg.logg.detach(), atol=1e-6)
    # a pure degree-2 polynomial over the observed pixels is removed entirely
    idx = torch.nonzero(torch.as_tensor(obs).flatten())[:, 0]
    rr, cc = idx // 40 + 100, idx % 40 + 200
    V = poly_basis(rr, cc, (100, 200, 30, 40), 2).float()
    pg.logg.data.zero_()
    pg.logg.data[idx] = V @ torch.tensor([0.3, -0.2, 0.1, 0.05, -0.07, 0.02])
    pg.project()
    assert float(pg.logg.abs().max()) < 1e-6
    # order 0: only the mean is removed
    pg0 = PixelGain(["X"], [(0, 0, 30, 40)], [obs], [np.ones((30, 40))], [np.ones((30, 40))], 0, "cpu")
    pg0.logg.data[idx] = V[:, 1] + 1.0
    pg0.project()
    v = pg0.logg[idx]
    assert abs(float(v.mean())) < 1e-6 and float(v.std()) > 0.1


def test_lookup_and_box():
    obs = np.ones((5, 6), bool)
    pg = PixelGain(["X", "Y"], [(10, 20, 5, 6), (0, 0, 2, 2)], [obs, np.ones((2, 2), bool)], [obs * 1.0, np.ones((2, 2))], [obs * 1.0, np.ones((2, 2))], 0, "cpu")
    pg.logg.data = torch.arange(34, dtype=torch.float32)
    row = torch.tensor([10, 14, 12, 9, 15, 12])
    col = torch.tensor([20, 25, 22, 22, 22, 26])
    v = pg.log_at(0, row, col)
    assert v.tolist() == [0.0, 29.0, 14.0, 0.0, 0.0, 0.0]
    assert pg.log_at(1, torch.tensor([1]), torch.tensor([1])).tolist() == [33.0]


def test_gain_enters_signal_and_sky_background_only():
    model, exps, cfg, _ = small("detector")
    pg = model.pixgain
    assert pg is not None and len(pg.labels) == 2
    e = [x for x in exps if x.split == "train"][0]
    pix = e.pix((0, 12, 0, 12))
    with torch.no_grad():
        model.bkg[e.eid] = torch.tensor([3.0, 0.2, -0.1, 0.05, 0.02, 0.01])
        model.pattern.normal_(0, 0.5)
        model.stripe_rows[e.eid].normal_(0, 0.3)
        model.stripe_cols[e.eid].normal_(0, 0.3)
        model.e_stripes[e.eid] = True
        P = model.prepare()
        out1 = model.render_batch([(e.eid, pix)], P)[0]
        pg.logg.normal_(0, 0.05)
        pg.project()
        out2 = model.render_batch([(e.eid, pix)], P)[0]
        G = torch.exp(pg.log_at(model.gain_key[e.eid], pix["row"], pix["col"]))
        from fieldop.model import legendre6
        L = (legendre6(pix["xi"]) * model.bkg[e.eid]).sum(-1)
        # m = G S + G L + (pattern + stripes), with S the gain-free signal of out1
        add1 = out1["m"] - out1["S"] - L                       # pattern + stripes (no nonlinearity in this config)
        expect = G * out1["S"] + G * L + add1
        assert torch.allclose(out2["m"], expect, atol=1e-4), float((out2["m"] - expect).abs().max())
        assert torch.allclose(out2["S"], G * out1["S"], atol=1e-5)
        assert float(add1.abs().max()) > 0.1                   # pattern and stripes were really present and are not multiplied by G
    cfg.batch_render = 0                                        # the non-batched renderer applies the same gain
    with torch.no_grad():
        out3 = model.render(e.eid, pix, P, method="fft")
    assert torch.allclose(out3["m"], out2["m"], atol=1e-3)


def test_maps_counts_and_groups():
    model, exps, cfg, _ = small("detector")
    pg = model.pixgain
    op, nuis, taus, skyp = model.param_groups()
    assert all(p is not pg.logg for p in op)
    for k, lab in enumerate(pg.labels):
        _, ob, cnt, inf_ = pg.map(k)
        r0, c0, nr, nc = pg.boxes[k]
        ref = np.zeros((nr, nc))
        for e in exps:
            if e.split == "train" and model.gain_key[e.eid] == k:
                a, b = e.valid.shape
                ref[e.cut0[0] - r0:e.cut0[0] - r0 + a, e.cut0[1] - c0:e.cut0[1] - c0 + b] += e.valid.numpy()
        assert np.array_equal(ref, cnt) and np.array_equal(ref > 0, ob)
        assert (inf_[ob] > 0).all()
    m2, _, _, _ = small("detector-filter")
    assert len(m2.pixgain.labels) == 3 and m2.pixgain.labels[0] == "A/A1"


def test_gradient_reaches_only_batch_pixels():
    model, exps, cfg, _ = small("detector")
    pg = model.pixgain
    e = [x for x in exps if x.split == "train"][0]
    pix = e.pix((0, 6, 0, 6))
    P = {k: v.detach() for k, v in model.prepare().items()}
    out = model.render_batch([(e.eid, pix)], P)[0]
    (out["m"] ** 2).sum().backward()
    g = pg.logg.grad
    k = model.gain_key[e.eid]
    r0, c0, nr, nc = pg.boxes[k]
    nz = torch.nonzero(g)[:, 0]
    assert nz.numel() == 36
    rr, cc = (nz - pg.off[k]) // nc + r0, (nz - pg.off[k]) % nc + c0
    assert int(rr.min()) == 0 and int(rr.max()) == 5 and int(cc.min()) == 0 and int(cc.max()) == 5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_cut_with_invalid_margins():
    """Regression (16 Nov 2026, first real-data smoke run): the gain box is the bounding box of the VALID pixels, so a cut whose array
    has invalid margins sticks out of it; build_pixel_gain must keep the overlap only."""
    from types import SimpleNamespace
    from fieldop.model import build_pixel_gain
    g = torch.Generator().manual_seed(3)
    cuts = []
    for e, (y0, x0) in enumerate([(100, 200), (110, 190)]):
        v = torch.zeros(40, 30, dtype=torch.bool)
        v[7:33, 5:22] = True                                     # invalid margins on every side
        d = torch.rand(40, 30, generator=g) + 5.0
        s = torch.ones(40, 30)
        cuts.append(SimpleNamespace(split="train", eid=e, cut0=(y0, x0), hostv=lambda k, v=v, d=d, s=s: {"valid": v, "d": d, "sig": s}[k]))
    meta = {"det": [0, 0], "filt": [0, 0]}
    pg, ekey = build_pixel_gain(cuts, meta, "detector", 2, torch.device("cpu"))
    r0, c0, nr, nc = pg.boxes[0]
    assert (r0, c0, nr, nc) == (107, 195, 36, 27), pg.boxes[0]
    lg, ob, cnt, inf = pg.map(0)
    assert int(ob.sum()) == int(cnt.astype(bool).sum()) and cnt.max() == 2 and int((cnt > 0).sum()) > 26 * 17
    assert int(ob.sum()) == 2 * 26 * 17 - int((cnt == 2).sum())
