"""Regression test: the sky warm start of a count-rate fit (--input-level rate) is in the units the rate-level model gives the sky.

The rate-level model multiplies the rendered sky by the pixel area alpha (arcsec^2, F15), so its sky is a count rate per arcsec^2. The warm
start (splat of the training pixels, data.splat_coadd) used to be built from the count rate per PIXEL, so a rate-level fit started with every
source too faint by the pixel area (0.001 to 0.004 on NIRCam, 0.01 here). On SMACS 0723 the fit never recovered: the noise model (fractional
floor 10, i.e. 1000 per cent) made the bright pixels weightless and the frozen sky predicted held-out sources at 0.08 to 0.15 of the data
(RATE_HELDOUT_DIAGNOSIS.md).

Test: the same synthetic field, once in surface brightness (input level cal) and once multiplied by each exposure's pixel area (input level
rate, which is exactly what the rate-level model computes). At the warm start, and after the registration pre-pass that rebuilds it, the
rate-level prediction of every training exposure must equal alpha x the cal-level prediction. Before the fix the ratio was alpha (0.01).
CPU, about 15 s.   python tests/test_rate_warm_start.py   (or pytest)
"""
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fieldop.cli import default_config, make_sky  # noqa: E402
from fieldop.data import build_meta, splat_coadd  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402
from fieldop.train import register_prepass  # noqa: E402


def field(level):
    torch.manual_seed(0)
    insts = [SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", 0.12, 20, 6.5, 0.10, 2.4, filters=[("B1", 2.0, 2.8)])]
    exps, fset, meta, truth, _ = make_field(seed=1, n_per_filter=4, n_val=1, n_held=1, insts=insts, centre_range=0.4, rot_choices=(0, 1, 2, 3),
                                            patch_px=12, truth_half=1.0)
    if level == "rate":       # count rate per pixel = surface brightness x pixel area (noise scaled alike: same S/N)
        for e in exps:
            a = float(abs(np.linalg.det(np.asarray(e.extra["J"]))))
            e.d, e.sig = e.d * a, e.sig * a
            e.bkg0 = float(e.d.median())
        meta = build_meta(exps, ["A", "B"], ["A", "B"], ["A", "B"])
    cfg = default_config(sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.8, region_px=12, n_lambda=3, k_hidden=16, pattern_grid=0,
                         stripes=0, batch_patches=3, reg_prepass_steps=2, reg_prepass_iters=1, input_level=level, batch_render=0,
                         distortion=0)   # no learned geometry: the model's pixel area is then exactly the header area
    sky, _ = make_sky(exps, fset, cfg, 1.6, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    return model, exps, cfg


@torch.no_grad()
def signals(model, exps):
    P = model.prepare()
    out = []
    for e in exps:
        if e.split != "train":
            continue
        n0, n1 = e.shape
        out.append(model.render(e.eid, e.pix((0, n0, 0, n1)), P)["S"])
    return out


def areas(exps):
    return [float(abs(np.linalg.det(np.asarray(e.extra["J"])))) for e in exps if e.split == "train"]


def check_ratio(sr, sc, al, tol):
    for s_r, s_c, a in zip(sr, sc, al):
        bright = s_c.abs() > 0.2 * s_c.abs().max()
        r = float((s_r[bright] / a).sum() / s_c[bright].sum())
        assert abs(r - 1.0) < tol, f"rate-level signal / (pixel area x cal-level signal) = {r:.4g} (pixel area {a:.4g}): the warm start is not per arcsec^2"


def test_splat_per_area():
    """splat_coadd(per_area=True) of rate-level exposures equals splat_coadd of the cal-level exposures (pixel area constant per detector)."""
    mc, ec, _ = field("cal")
    mr, er, _ = field("rate")
    N, pitch = 41, 0.08
    ic, wc = splat_coadd([e for e in ec if e.split == "train"], len(mc.fset), N, pitch, "cpu")
    ir, _ = splat_coadd([e for e in er if e.split == "train"], len(mr.fset), N, pitch, "cpu", per_area=True)
    cov = wc > 0
    assert torch.allclose(ir[cov], ic[cov], rtol=1e-4, atol=1e-5 * float(ic.abs().max()))
    ir0, _ = splat_coadd([e for e in er if e.split == "train"], len(mr.fset), N, pitch, "cpu")      # default: unchanged (per pixel)
    assert float(ir0.abs().max()) < 0.02 * float(ic.abs().max())


def test_rate_warm_start_units():
    mc, ec, cc = field("cal")
    mr, er, cr = field("rate")
    al = areas(er)
    check_ratio(signals(mr, er), signals(mc, ec), al, 2e-3)
    register_prepass(mc, ec, cc, log=lambda s: None)       # rebuilds the warm start at the registered positions
    register_prepass(mr, er, cr, log=lambda s: None)
    check_ratio(signals(mr, er), signals(mc, ec), al, 2e-2)


if __name__ == "__main__":
    import time
    t = time.perf_counter()
    test_splat_per_area()
    test_rate_warm_start_units()
    print(f"ok ({time.perf_counter() - t:.1f} s)")
