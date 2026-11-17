"""Sky zero level (--sky-zero) and positivity margin (--sky-positivity-margin), AUDIT_2026-11-17.md section F, change 1.

On a small version of the known-truth field (tests/truth/truth_problem.py: background 5.0, noise 1.0, 36 injected sources whose light raises
every exposure's median by about 0.15):
1. default (median): sky_zero_refresh does nothing and the gauge reference stays the exposure median;
2. masked: the reference levels (median of the data on the pixels where the smoothed rendered sky |S| < 0.3 sigma, with the warm-start sky)
   are at least 2 times closer to the true background than the medians (with the warm-start sky; the fitted sky does better), and the pedestal gauge moves the mean fitted background to them;
3. the masked level is a property of the DATA: adding a constant layer of 0.05 to the sky (the direction the data cannot fix) moves it by
   on average by less than 0.4 of it (only through the mask), whereas a level computed on the residual d - S would move by the full layer;
4. positivity margin: grid penalty zero for a sky at -p/2 and positive at -2p; item sky = items - p, the items carry the layer.

    python tests/test_sky_zero.py     (or pytest; about 30 s)
"""
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "truth"))

import truth_problem as tp  # noqa: E402
from fieldop.cli import make_sky  # noqa: E402
from fieldop.tiles import TileSet  # noqa: E402
from fieldop.train import sky_regs, sky_zero_refresh  # noqa: E402

QUIET = lambda s: None  # noqa: E731


def small(form="grid", **over):
    pop = tp.make_population()
    fl = tp.flux_for_snr(pop, n_ref=8)
    exps, fset, meta, truth, pf, _ = tp.build_field(4, seed=0, pop=pop, fluxes=fl)
    cfg = tp.base_config(form, steps=10, item_count=300, **over)
    tp.KERNEL_STEPS = 300
    model, _ = tp.build_model(exps, fset, meta, cfg, freeze_kernel=True, log=QUIET)
    return model, exps, cfg


def train_ids(exps):
    return [e.eid for e in exps if e.split == "train"]


def test_median_default_is_untouched():
    model, exps, cfg = small()
    ref = model.bkg0.clone()
    assert cfg.sky_zero == "median"
    assert sky_zero_refresh(model, exps, cfg, TileSet.single(model, exps, cfg, QUIET), QUIET) is None
    assert torch.equal(model.bkg0, ref)


def test_masked_level_closer_to_truth_and_not_moved_by_a_sky_layer():
    model, exps, cfg = small(sky_zero="masked")
    ids = train_ids(exps)
    med = model.bkg0[ids].clone()
    ts = TileSet.single(model, exps, cfg, QUIET)
    rec = sky_zero_refresh(model, exps, cfg, ts, QUIET)
    lv = model.bkg0[ids].clone()
    err_med, err_mask = float((med - tp.BKG).mean()), float((lv - tp.BKG).mean())
    assert err_med > 0.1, err_med                      # the medians contain source light
    assert abs(err_mask) < err_med / 2, (err_mask, err_med)   # 0.063 against 0.187 with the warm-start sky; 0.03-0.05 after the fits
    assert abs(float(model.bkg[ids, 0].mean() - lv.mean())) < 1e-5     # pedestal gauge applied at once
    assert all(0.2 < v[1] <= 1.0 for v in rec.values())
    with torch.no_grad():                               # constant sky layer of 0.05 (all spectral coefficients, partition of unity)
        model.sky.A.data += 0.05 / model.sky.pre / float(model.Phi_band[0].sum())
    sky_zero_refresh(model, exps, cfg, ts, QUIET)
    shift = float((model.bkg0[ids] - lv).mean())         # through the mask only (pixels move across the |S| threshold)
    assert abs(shift) < 0.4 * 0.05, shift


def test_positivity_margin_grid_and_items():
    model, exps, cfg = small(sky_positivity_margin=3.0, sky_neg=1.0)
    sky = model.sky
    p = float(model.Phi_band[0] @ sky.ped)
    assert p > 0
    sig_b = sky.sig_b
    with torch.no_grad():
        sky.A.data[:] = (-0.5 * p / float(model.Phi_band[0].sum())) / sky.pre
        assert float(sky_regs(model, sig_b, cfg)) == 0.0
        sky.A.data[:] = (-2.0 * p / float(model.Phi_band[0].sum())) / sky.pre
        assert float(sky_regs(model, sig_b, cfg)) > 0.0
    # items: built with the margin, the item layer is p above the returned sky
    mi, ei, ci = small("items", sky_positivity_margin=3.0)
    si = mi.sky
    with torch.no_grad():
        A = si.coeffs()
        raw = si.raster(0)
    np.testing.assert_allclose((raw - A).numpy(), np.broadcast_to(si.ped.numpy()[:, None, None], A.shape), rtol=1e-5, atol=1e-5)
    # no margin (default): no pedestal attribute value
    sk, _ = make_sky(ei, mi.fset, tp.base_config("items", item_count=300), 2.5, "cpu")
    assert sk.ped is None


if __name__ == "__main__":
    import time
    t = time.perf_counter()
    test_median_default_is_untouched()
    test_masked_level_closer_to_truth_and_not_moved_by_a_sky_layer()
    test_positivity_margin_grid_and_items()
    print(f"ok ({time.perf_counter() - t:.1f} s)")
