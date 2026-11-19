"""Kernel-centroid gauge (F43, --kernel-centroid-gauge, KERNEL_CENTROID_GAUGE_2026-11-19.md): fast CPU checks on a small synthetic field.

1. Off (the default): nothing of the gauge is made (no buffer in the state, no attribute that the render reads), so every existing oracle is
   unchanged (the golden steps and state hashes are checked by test_oracle_golden.py and tests/oracle/).
2. The projection leaves every prediction unchanged: its translation part to float32 round-off; with a step of the size seen in a fit (0.05 mas
   per class) the item-sky part (a shift of a fraction of that) changes predictions by less than 1e-5 of the signal.
3. After a projection the held moment of every class is zero to round-off (below 1e-7 arcsec) and the training translations keep zero mean.
4. A short fit with the gauge on runs, projects once per step and reports the record in the fit's result.
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
torch.set_num_threads(1)

from fieldop.cli import apply_sky_module_options, default_config, make_sky  # noqa: E402
from fieldop.instruments import spectral_setup  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402
from fieldop.train import fit, register_prepass, warm_kernel  # noqa: E402

INSTS = [SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, space_grad=0.15, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
         SynthInstrument("B", 0.12, 20, 6.5, 0.10, 2.4, space_grad=0.15, filters=[("B1", 2.0, 2.8)])]
KW = dict(sky_pitch=0.08, item_count=300, spec_k=6, kernel_extent=1.0, render_pitch=0.04, level_nh=8, region_px=13, n_lambda=3, k_hidden=16,
          warm_kernel_steps=30, batch_patches=3, inner_sky=1, budget_min=1e6, val_every=2, held_steps=10, held_reg_steps=6, log_every=5,
          is_snr=10.0, reg_prepass_steps=8, reg_prepass_iters=1, gauge_ref_inst=0, device="cpu", seed=0)


def build(gauge, steps=0, **over):
    torch.manual_seed(0)
    exps, fset, meta, _, _ = make_field(seed=3, n_per_filter=4, n_held=1, insts=INSTS, shift_rms=0.03, centre_range=0.4,
                                        rot_choices=(0, 1, 2, 3), patch_px=12, truth_half=1.0)
    cfg = default_config(**dict(KW, steps=max(steps, 1), kernel_centroid_gauge=gauge, **over))
    apply_sky_module_options(cfg)
    spectral_setup(fset, cfg, log=lambda s: None)
    sky, _ = make_sky(exps, fset, cfg, 1.8, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    finfo = {f: {"pixel_arcsec": INSTS[0 if f < 2 else 1].pixel, "diameter_m": 6.5, "inst": 0 if f < 2 else 1} for f in range(len(fset))}
    warm_kernel(model, cfg, finfo, log=lambda s: None)
    register_prepass(model, exps, cfg, log=lambda s: None)
    rec = fit(model, exps, cfg, log=lambda s: None) if steps else None
    return model, exps, rec


@torch.no_grad()
def predict(model, exps):
    P = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in model.prepare().items()}
    out = []
    for e in exps:
        for sl in e.patches:
            out.append(model.render_batch([(e.eid, e.pix(sl))], P, jitter=False)[0]["m"].clone())
    return torch.cat([o.flatten() for o in out])


def rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.fixture(scope="module")
def on():
    return build(1)


def test_off_makes_nothing():
    cfg = default_config()
    assert cfg.kernel_centroid_gauge == 0
    model, _, _ = build(0)
    assert not model.kc_on and not hasattr(model, "kc_off")
    assert not any(k.startswith("kc_") for k in model.state_dict())


def test_projection_preserves_predictions_and_zeroes_the_moment(on):
    model, exps, _ = on
    st = {k: v.clone() for k, v in model.state_dict().items()}
    try:
        R, cls, tr = model.e_R, model.kc_class, model.ref_eids
        C = len(model.kc_pairs)
        with torch.no_grad():
            # (a) a large perturbation (+-20 mas per class) made function-preservingly through the translations, then the translation part of the
            #     projection alone: both exact to round-off
            base = predict(model, exps)
            dk = torch.stack([torch.tensor([0.02, -0.015]) * (-1) ** c for c in range(C)])
            model.kc_off += dk
            model.affine.data[:, 4:6] -= torch.einsum("eij,ej->ei", R, dk[cls]) / model.aff_scale[4:6]
            assert rel(predict(model, exps), base) < 2e-6
            dl = model.dilation()
            dil = None if dl is None else dl.detach()
            old = model.kc_off.clone()
            new = torch.stack([model.kernel_centroid_solve(c, old[c], dil)[0] for c in range(C)])
            model.kc_off.copy_(new)
            model.affine.data[:, 4:6] -= torch.einsum("eij,ej->ei", R, (new - old)[cls]) / model.aff_scale[4:6]
            assert rel(predict(model, exps), base) < 2e-6
            for c in range(C):
                assert float(model.kernel_centroid(c, model.kc_off[c], dil).norm()) < 1e-7
            # (b) the full projection after a step-sized perturbation (0.05 mas per class): sky part included
            dk = torch.stack([torch.tensor([5e-5, -3e-5]) * (-1) ** c for c in range(C)])
            model.kc_off += dk
            model.affine.data[:, 4:6] -= torch.einsum("eij,ej->ei", R, dk[cls]) / model.aff_scale[4:6]
            pre = predict(model, exps)
            t_mean = (model.affine[tr, 4:6] * model.aff_scale[4:6]).mean(0).clone()
            delta, a = model.project_kernel_centroid()
            assert float(delta.abs().max()) > 1e-5             # the projection did move the offsets back
            assert rel(predict(model, exps), pre) < 1e-5
            assert float(((model.affine[tr, 4:6] * model.aff_scale[4:6]).mean(0) - t_mean).abs().max()) < 1e-9
            dl = model.dilation()
            dil = None if dl is None else dl.detach()
            for c in range(C):
                assert float(model.kernel_centroid(c, model.kc_off[c], dil).norm()) < 1e-7
            # (c) an uncompensated kernel shift of 1 mas is NOT invisible (the check above can fail)
            model.kc_off[1] += torch.tensor([1e-3, 0.0])
            assert rel(predict(model, exps), pre) > 1e-4
    finally:
        model.load_state_dict(st)
        model.sky._mut += 1
        model.sky._cache = None


def test_short_fit_with_the_gauge_reports_it():
    model, exps, rec = build(1, steps=6)
    kr = rec["kernel_centroid_gauge"]
    assert kr["projections"] == 6 and len(kr["classes"]) == 3
    assert kr["held_moment_after_last_projection_mas"] < 1e-4
    assert all(c["window_sigma_arcsec"] > 0 for c in kr["classes"])
    assert "kc_off" in model.state_dict()
    g = model.gauge_report()
    assert max(abs(x) for v in g["kernel_centroid_held_mas"].values() for x in v) < 1e-4
