"""Smoke test of the known-truth problem (under 60 s on one core): the problem builds, one very short fit runs for the item sky, the measurement
functions return finite numbers. NOT a convergence test (3 steps)."""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import truth_problem as tp  # noqa: E402

torch.set_num_threads(1)


def _finite(x):
    return bool(np.all(np.isfinite(np.asarray(x, dtype=float))))


def _build():
    pop = tp.make_population()
    fluxes = tp.flux_for_snr(pop)
    assert len(pop) == 36 and _finite(fluxes) and (fluxes > 0).all()
    empties = tp.empty_positions(pop)
    assert len(empties) >= 15
    sel = list(range(0, len(pop), 3))                      # a third of the sources keeps it quick
    pop_s, fl_s = [pop[i] for i in sel], fluxes[sel]
    exps, fset, meta, truth, pf, _ = tp.build_field(2, seed=0, pop=pop_s, fluxes=fl_s)
    assert [e.split for e in exps] == ["held", "train", "train"]
    return pop_s, fl_s, empties[:6], exps, fset, meta, pf


def _one(form):
    pop, fl, em, exps, fset, meta, pf = _build()
    tp.KERNEL_STEPS = 60
    cfg = tp.base_config(form, steps=3, held_steps=3, held_reg_steps=3, reg_prepass_steps=3, item_count=200, log_every=1)
    model, res = tp.run_fit(exps, fset, meta, cfg, freeze_kernel=True, log=lambda s: None)
    assert res["steps"] == 3 and _finite(res["heldout_chi2"]) and _finite(res["noise_only_chi2"]) and res["noise_only_chi2"] > 0.8
    tr = [e for e in exps if e.split == "train"]
    cs = tp.IdealCache([(p[0], p[1], p[2]) for p in pop], tr, fset)
    ce = tp.IdealCache([(x, y, tp.SIG_RES) for x, y in em], tr, fset)
    assert _finite(cs.sigma) and _finite(ce.sigma)
    m = tp.measure(model, exps, fset, pop, fl, em, cs, ce)
    for k in ("rec", "rec_matched", "ideal", "snr_true"):
        assert _finite(m["src"][k]), k
    for k in ("rec", "rec_raw", "ideal", "ref_sky", "ideal_ref"):
        assert _finite(m["emp"][k]), k
    s = tp.summarise_sources([m, m])
    e = tp.summarise_empty([m, m])
    assert len(s) == 5 and _finite(e["rec"]["rms_sigma"]) and _finite(e["ideal"]["rms_sigma"]) and _finite(e["rec"]["mean_sigma"])
    # truth aperture check: the true sky in its own aperture returns the true flux
    N = 101
    img = tp.true_sky_nodes(pf, N, 0.05)
    for (x, y, sg, F) in pf[:3]:
        idx, P, W = tp.aperture_weights(N, 0.05, x, y, sg)
        assert math.isclose(tp.sky_aperture_flux(img, idx, P, W), F, rel_tol=0.05)


def test_smoke_items():
    t = time.time()
    _one("items")
    assert time.time() - t < 60


if __name__ == "__main__":
    for f in (test_smoke_grid, test_smoke_items):
        t = time.time()
        f()
        print(f.__name__, "ok %.1f s" % (time.time() - t))
