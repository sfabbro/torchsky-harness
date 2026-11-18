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
    model, exps, cfg = small(sky_zero="median")      # (the default became "masked" on 18 Nov 2026; the option still works)
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


def test_two_tiles_one_level_from_all_cuts_and_subsample():
    """Build A: the synthetic two-tile field. One level per exposure (global id) from the source-free pixels of all its cuts, equal to the
    pooled median computed by hand with the full-exposure prediction; the subsampled refresh stays within a few per cent of a pixel sigma."""
    from fieldop.cli import default_config
    from fieldop.model import FieldOperator
    from fieldop.synth import make_field_tiles
    from fieldop.train import predict_exposure, smooth_image, warm_kernel
    torch.manual_seed(0)
    cuts, fset, meta, truth, insts, tiles, half_data = make_field_tiles(seed=0, n_wide=4, n_side=2)
    cfg = default_config(sky_pitch=0.06, render_pitch=0.02, level_nh=10, kernel_extent=1.0, region_px=13, n_lambda=3, k_hidden=16,
                         warm_kernel_steps=20, pattern_grid=0, stripes=0, batch_patches=3, inner_sky=1, budget_min=1e6, steps=4,
                         tau_mode="fixed", data_margin=0.3, sky_margin=0.6, sky_form="grid", sky_zero="masked", sky_zero_snr=3.0, gauge_ref_inst=0)
    half = half_data - cfg.data_margin + cfg.sky_margin
    skies = {}
    for k in sorted(tiles):
        cs = [e for e in cuts if e.tile == k]
        for e in cs:
            e.to_dev("cpu")
        skies[k], _ = make_sky(cs, fset, cfg, half, "cpu")
    model = FieldOperator(cfg, fset, meta, skies[0], torch.device("cpu"))
    ts = TileSet(model, cuts, skies, cfg, log=lambda s: None)
    finfo = {f: {"pixel_arcsec": 0.10 if f < 2 else 0.12, "diameter_m": 6.5, "inst": 0 if f < 2 else 1} for f in range(len(fset))}
    warm_kernel(model, cfg, finfo, log=lambda s: None)
    cfg.sky_zero_pixels = 0
    lines = []
    rec = sky_zero_refresh(model, cuts, cfg, ts, lines.append)
    assert rec is not None and len(lines) == 1 and "re-referenced" in lines[0], lines      # one line per refresh
    eids_two = [i for i in {c.eid for c in cuts if c.split == "train"} if sum(1 for c in cuts if c.eid == i) > 1]
    assert eids_two, "the field has no exposure with cuts in both tiles"
    lvl = model.bkg0.clone()
    P = {kk: v.detach() for kk, v in model.prepare().items()}
    for eid in eids_two:
        vs = []
        for c in cuts:
            if c.eid != eid or c.split != "train":
                continue
            ts.activate(c.tile)
            _, pv, S = predict_exposure(model, c, P, cfg)
            v0 = c.valid & pv & torch.isfinite(S)
            Ss = smooth_image(torch.where(v0, S, torch.zeros_like(S)), v0, float(cfg.sky_zero_smooth))
            vs.append(c.d[v0 & (Ss.abs() < cfg.sky_zero_snr * c.sig)])
        assert len(vs) == 2
        ref = float(torch.cat(vs).median())
        assert abs(float(lvl[eid]) - ref) < 1e-5, (eid, float(lvl[eid]), ref)
    # subsample: about 60 per cent of the pixels of each cut (cuts are 30 pixels wide)
    cfg.sky_zero_pixels = int(0.6 * np.mean([c.shape[0] * c.shape[1] for c in cuts])), cfg.sky_zero_patch
    cfg.sky_zero_pixels, cfg.sky_zero_patch = cfg.sky_zero_pixels[0], 8
    sky_zero_refresh(model, cuts, cfg, ts, QUIET)
    sm = sky_zero_refresh.last_summary
    tr = sorted({c.eid for c in cuts if c.split == "train"})
    n_tot = sum(c.shape[0] * c.shape[1] for c in cuts if c.split == "train")
    assert sm["pixels_rendered"] < 0.85 * n_tot, (sm, n_tot)
    assert float((model.bkg0[tr] - lvl[tr]).abs().max()) < 0.6, float((model.bkg0[tr] - lvl[tr]).abs().max())      # pixel sigma is 1


if __name__ == "__main__":
    import time
    t = time.perf_counter()
    test_two_tiles_one_level_from_all_cuts_and_subsample()
    test_median_default_is_untouched()
    test_masked_level_closer_to_truth_and_not_moved_by_a_sky_layer()
    test_positivity_margin_grid_and_items()
    print(f"ok ({time.perf_counter() - t:.1f} s)")
