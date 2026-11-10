"""CPU tests of the field-operator renderers (fieldop/render.py): the overlap-add FFT renderer against the per-pixel quadrature reference,
and how the difference falls with render pitch, region size (number of region nodes) and lambda-node count.

Run:  python -m pytest -q -s tests/test_fieldop_render.py      (about 5 min on 2 CPU cores; numbers written to tests/fieldop_results/)
"""
import json
import math
import os
import sys
import time

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fieldop.cli import default_config, filt_info_of, make_sky  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.render import bspline_prefilter, bspline_read, correlate_same, quad_points, render_quad  # noqa: E402
from fieldop.synth import make_field  # noqa: E402
from fieldop.train import warm_kernel  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "fieldop_results")
os.makedirs(OUT, exist_ok=True)
BASE = dict(sky_pitch=0.04, render_pitch=0.01, level_factor=2, level_nh=24, kernel_extent=1.0, region_px=5, n_lambda=3, k_hidden=32,
            warm_kernel_steps=300, pattern_grid=0, stripes=0, xfit=1, enc_r0=0.005, k_space="on", tau0_code=0.15, amplitude="unit",
            distortion=0, k_exp="off", k_time="off", sky_precond=0.0)


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    exps, fset, meta, truth, insts = make_field(seed=1)

    def build(**kw):
        b = dict(BASE)
        b.update(kw)
        cfg = default_config(**b)
        torch.manual_seed(0)
        sky, _ = make_sky(exps, fset, cfg, 2.0, "cpu")
        return cfg, FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    cfg, m0 = build()
    warm_kernel(m0, cfg, filt_info_of(exps, fset, ["A", "B"]), log=lambda s: None)
    sd = {k: v.clone() for k, v in m0.state_dict().items()}
    e = exps[5]                                # filter A1: the sharpest kernel of the synthetic field (sigma ~ 0.022")
    iy, ix = divmod(int(e.d.argmax()), e.d.shape[1])
    n = 12
    y0, x0 = min(max(0, iy - 6), e.d.shape[0] - n), min(max(0, ix - 6), e.d.shape[1] - n)
    pix = e.pix((y0, y0 + n, x0, x0 + n))
    return dict(exps=exps, fset=fset, meta=meta, build=build, m0=m0, sd=sd, e=e, pix=pix)


def rel(a, b):
    d = a - b
    return float(d.abs().max() / b.abs().max()), float(d.pow(2).mean().sqrt() / b.pow(2).mean().sqrt())


def test_correlate_matches_conv2d():
    torch.manual_seed(0)
    img, ker = torch.randn(3, 40, 50), torch.randn(3, 7, 7)
    a = correlate_same(img, ker)
    b = torch.stack([F.conv2d(img[c][None, None], ker[c][None, None], padding=3)[0, 0] for c in range(3)])
    assert float((a - b).abs().max()) < 1e-4


def test_bspline_read_exact_on_cubics():
    H = W = 40
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    f = lambda x, y: 0.3 + 0.1 * x - 0.02 * y + 1e-3 * x * x * y - 2e-4 * y ** 3  # noqa: E731
    img = f(xx, yy)[None, None]
    c = bspline_prefilter(img)
    X = torch.rand(1, 200, dtype=torch.float64) * 10 + 15
    Y = torch.rand(1, 200, dtype=torch.float64) * 10 + 15
    v = bspline_read(c, (0.0, 0.0), 1.0, X, Y)[0, 0]
    assert float((v - f(X[0], Y[0])).abs().max()) < 1e-6 * float(img.abs().max())


def kernel_width_variation(m, e, P):
    """Relative change of the kernel core second moment between detector corners (how strong the injected spatial variation is)."""
    from fieldop.render import kernel_level_values
    xi = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    h = m.codes_at(torch.tensor(e.eid), xi, P)[0]
    Ks, _ = kernel_level_values(m.net, m.levels, m.e_R[e.eid], h)
    K0, u = Ks[0], m.levels.u[0]
    m2 = (K0 * (u ** 2).sum(-1)).sum((-1, -2)) / K0.sum((-1, -2))
    return float((m2.max() - m2.min()) / m2.mean())


def test_fft_vs_quadrature_convergence(setup):
    S = setup
    e, pix = S["e"], S["pix"]
    res = {"exposure": e.name, "pixels": int(pix["X"].numel())}
    with torch.no_grad():
        refs = {}
        for ks in ("off", "on"):
            cfg, m = S["build"](k_space=ks)
            m.load_state_dict(S["sd"], strict=False)
            P = m.prepare()
            t = time.time()
            q1 = quad_points(m.levels, 0.004, n_rad=48, n_az=192, gen=torch.Generator().manual_seed(0))
            R1 = m.render(e.eid, pix, P, method="quad", quad=q1)["S"]
            q2 = quad_points(m.levels, 0.0025, n_rad=64, n_az=256, gen=torch.Generator().manual_seed(1))
            R2 = m.render(e.eid, pix, P, method="quad", quad=q2)["S"]
            refs[ks] = (R2, P)
            res[f"reference_self_consistency_max_rel_k_space_{ks}"] = rel(R1, R2)[0]
            res[f"reference_seconds_k_space_{ks}"] = time.time() - t
            if ks == "on":
                res["kernel_core_m2_variation_across_detector"] = kernel_width_variation(m, e, P)
        rows = []
        sweep = [dict(k_space="off", render_pitch=0.02), dict(k_space="off", render_pitch=0.01), dict(k_space="off", render_pitch=0.005),
                 dict(k_space="off", render_pitch=0.01, interp="keys"), dict(k_space="off", render_pitch=0.005, interp="keys")]
        sweep += [dict(k_space="on", render_pitch=0.005, region_px=r) for r in (40, 20, 10, 5, 2)]
        for kw in sweep:
            cfg, m = S["build"](**kw)
            m.load_state_dict(S["sd"], strict=False)
            R, P = refs[kw["k_space"]]
            t = time.time()
            Sf = m.render(e.eid, pix, P)["S"]
            mx, rms = rel(Sf, R)
            rows.append(dict(kw, max_rel=mx, rms_rel=rms, seconds=time.time() - t))
            print(kw, f"max rel {mx:.2e} rms rel {rms:.2e}")
        res["fft_vs_reference"] = rows
    json.dump(res, open(os.path.join(OUT, "render_convergence.json"), "w"), indent=1)
    print("reference self-consistency", {k: v for k, v in res.items() if "consistency" in k}, "kernel m2 variation", res["kernel_core_m2_variation_across_detector"])
    by = {(r["k_space"], r.get("render_pitch"), r.get("region_px", 5), r.get("interp", "bspline")): r["max_rel"] for r in rows}
    assert max(v for k, v in res.items() if "consistency" in k) < 3e-3
    # render pitch (no spatial variation): monotone decrease, finest below 0.5 %
    assert by[("off", 0.02, 5, "bspline")] > by[("off", 0.01, 5, "bspline")] > by[("off", 0.005, 5, "bspline")]
    assert by[("off", 0.005, 5, "bspline")] < 5e-3
    # B-spline reads beat Keys bicubic at equal pitch
    assert by[("off", 0.01, 5, "keys")] > by[("off", 0.01, 5, "bspline")]
    # region size (spatially varying kernel): more region nodes -> smaller error
    errs = [by[("on", 0.005, r, "bspline")] for r in (40, 20, 10, 5, 2)]
    assert all(a >= b - 1e-4 for a, b in zip(errs[:-1], errs[1:])) and errs[-1] < 0.5 * errs[0]


def test_lambda_node_convergence():
    """Q quadrature nodes vs a dense 32-node rule (same renderer, same chromatic kernel and sky)."""
    torch.manual_seed(0)
    out = {}
    ref = None
    for Q in (32, 1, 2, 3, 5, 8):
        exps, fset, meta, truth, insts = make_field(seed=1, Q=Q)
        b = dict(BASE, n_lambda=Q, region_px=40, render_pitch=0.01, k_space="off")
        cfg = default_config(**b)
        torch.manual_seed(0)
        sky, _ = make_sky(exps, fset, cfg, 2.0, "cpu")
        if ref is None:
            A_ref = sky.A.data.clone()
        sky.A.data[:] = A_ref
        m = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
        torch.manual_seed(0)
        warm_kernel(m, cfg, filt_info_of(exps, fset, ["A", "B"]), log=lambda s: None)
        e = exps[5]
        with torch.no_grad():
            S = m.render(e.eid, e.pix(e.patches[0]), m.prepare())["S"]
        if ref is None:
            ref = S
            continue
        out[Q] = rel(S, ref)
        print(f"Q = {Q}: max rel {out[Q][0]:.2e} rms rel {out[Q][1]:.2e} (vs Q = 32)")
    json.dump({str(k): v for k, v in out.items()}, open(os.path.join(OUT, "lambda_convergence.json"), "w"), indent=1)
    assert out[8][0] < out[2][0] and out[5][0] < out[1][0]


def test_detector_aligned_fine_levels(setup):
    """Arbitrary detector rotation (35 deg): fine levels on a detector-aligned grid (default) vs a sky-aligned grid, both against the per-pixel
    quadrature with the same kernel orientation; the detector-aligned window is smaller (fewer FFT nodes)."""
    S = setup
    e, pix = S["e"], S["pix"]
    th = math.radians(35.0)
    R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    out = {}
    with torch.no_grad():
        cfg, m = S["build"](k_space="off", render_pitch=0.005)
        m.load_state_dict(S["sd"], strict=False)
        m.e_R[e.eid] = R
        P = m.prepare()
        q = quad_points(m.levels, 0.0025, n_rad=64, n_az=256, gen=torch.Generator().manual_seed(1))
        Rq = m.render(e.eid, pix, P, method="quad", quad=q)["S"]
        for frame in ("sky", "detector"):
            cfg, mf = S["build"](k_space="off", render_pitch=0.005, render_frame=frame)
            mf.load_state_dict(S["sd"], strict=False)
            mf.e_R[e.eid] = R
            t = time.time()
            Sf = mf.render(e.eid, pix, mf.prepare())["S"]
            out[frame] = dict(zip(("max_rel", "rms_rel"), rel(Sf, Rq)), seconds=time.time() - t)
            print(frame, out[frame])
    json.dump(out, open(os.path.join(OUT, "render_frame_rotated35.json"), "w"), indent=1)
    assert out["detector"]["max_rel"] < 2e-3 and out["sky"]["max_rel"] < 2e-3
