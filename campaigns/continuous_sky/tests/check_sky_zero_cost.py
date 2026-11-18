"""Cost and agreement of the zero-level refresh on a subsample of pixels (BUILDS_2026-11-18.md, build A).

Truth problem (tests/truth), n_train training exposures, sky and response at their warm start (kernel fitted to the truth, 300 steps): the levels of
sky_zero_refresh with --sky-zero-px 0 (every pixel) against several subsamples; differences are given in units of the pixel noise (1.0).
    PYTHONNOUSERSITE=1 PYTHONPATH=/home/claude/torchsky/src python tests/check_sky_zero_cost.py [npix] [n_train]
"""
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "truth"))
import truth_problem as tp  # noqa: E402
from fieldop.tiles import TileSet  # noqa: E402
from fieldop.train import sky_zero_refresh  # noqa: E402

npix = int(sys.argv[1]) if len(sys.argv) > 1 else 44
ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 12
pop = tp.make_population()
fl = tp.flux_for_snr(pop, n_ref=8)
exps, fset, meta, truth, pf, _ = tp.build_field(ntr, seed=0, pop=pop, fluxes=fl, npix=npix)
cfg = tp.base_config("grid", steps=10, sky_zero="masked")
tp.KERNEL_STEPS = 300
model, _ = tp.build_model(exps, fset, meta, cfg, freeze_kernel=True, log=lambda s: None, sky_half=max(2.5, 0.05 * npix + 0.5))
ts = TileSet.single(model, exps, cfg, lambda s: None)
tr = [e for e in exps if e.split == "train"]
res = {}
PA = 8 if npix < 100 else 32
for label, px, patch in [("full", 0, 0), (f"random {PA}x{PA} patches ~1/10", int(0.1 * npix * npix), PA), (f"random {PA}x{PA} patches ~1/4", int(0.25 * npix * npix), PA),
                         ("default 20000", 20000, 0)]:
    cfg.sky_zero_pixels, cfg.sky_zero_patch = px, patch
    model.bkg0.zero_()
    t0 = time.perf_counter()
    rec = sky_zero_refresh(model, exps, cfg, ts, lambda s: None)
    dt = time.perf_counter() - t0
    lv = np.array([rec[e.name][0] if e.name in rec else np.nan for e in tr])
    res[label] = (lv, sky_zero_refresh.last_summary, dt)
lv0 = res["full"][0]
print(f"npix {npix}, {ntr} training exposures, true background {tp.BKG}, pixel noise {tp.SIGMA_NOISE}")
print(f"{'variant':22s} {'px/exposure':>12s} {'seconds':>8s} {'mean(level-full)':>17s} {'rms(level-full)':>16s} {'rms(level-truth)':>17s}  expected rms diff")
n_full = res["full"][1]["pixels_source_free"] / ntr
for k, (lv, sm, dt) in res.items():
    ok = np.isfinite(lv) & np.isfinite(lv0)
    d, lv = (lv - lv0)[ok], lv[ok]
    nbad = int((~ok).sum())
    nsf = max(sm["pixels_source_free"] / ntr, 1)
    exp_sd = 1.2533 * tp.SIGMA_NOISE * np.sqrt(max(1.0 / nsf - 1.0 / n_full, 0.0))
    print(f"{k:22s} {sm['pixels_per_exposure']:12.0f} {dt:8.2f} {d.mean():17.4f} {np.sqrt((d ** 2).mean()):16.4f} {np.sqrt(((lv - tp.BKG) ** 2).mean()):17.4f}  {exp_sd:.4f}  (refused: {nbad})")
