"""A filter with no training exposure (--train-exclude-filters, 19 Nov 2026): what the package does with it, without a fit (about 20 s).
    PYTHONNOUSERSITE=1 python tests/test_unseen_band.py   (or pytest)"""
import math
import os
import sys

import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fieldop.cli import default_config, make_sky, parse_config  # noqa: E402
from fieldop.instruments import spectral_setup  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.sky import spectral_init  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402
from fieldop.train import bound_report  # noqa: E402


def build(exclude=()):
    fa = [("A1", 1.0, 1.3), ("A2", 1.6, 2.1), ("A3", 2.4, 3.0)]
    insts = [SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, filters=fa)]
    exps, fset, meta, _, _ = make_field(seed=3, n_per_filter=5, n_val=1, n_held=1, insts=insts, patch_px=12, truth_half=1.2, train_exclude=exclude)
    cfg = default_config(sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=1.0, region_px=11, n_lambda=3, k_hidden=16, pattern_grid=0, stripes=0,
                         noise_scale_max=2.0)
    spectral_setup(fset, cfg, log=lambda s: None)
    sky, _ = make_sky(exps, fset, cfg, 2.0, "cpu")
    return exps, fset, FieldOperator(cfg, fset, meta, sky, torch.device("cpu")), cfg


def test_option_default_and_parse():
    assert parse_config([]).train_exclude_filters == ""
    assert parse_config(["--train-exclude-filters", "A2,B1"]).train_exclude_filters == "A2,B1"


def test_unseen_filter_bookkeeping_and_neutral_terms():
    exps, fset, model, cfg = build(("A2",))
    mid = fset.index["A2"]
    assert model.unseen == [mid]
    assert [e.split for e in exps if e.filt == mid] == ["held"]             # only its held-out exposure is left
    assert float(model.hs[:, mid].abs().max()) == 0.0                       # noise scale exp(0) = 1
    assert all(not s.endswith("A2") for s in [n for n, _ in bound_report(model, cfg)["noise_scale"]])
    assert model.blur_ref != mid and mid not in model.blur_sig2
    P = model.prepare()
    m = P["m"]
    assert torch.isfinite(m).all()
    # the throughput convention at the unseen filter's nodes is the nearest covered node's (continuous in wavelength), not 0 or noise
    lam = model.lam.flatten()
    seen = [n for n in range(lam.numel()) if n // fset.Q != mid]
    for q in range(fset.Q):
        n = mid * fset.Q + q
        j = min(seen, key=lambda s: abs(float(lam[s] - lam[n])))
        assert float(m.flatten()[n]) == float(m.flatten()[j])
    g = model.gauge_report(exps, P)
    assert g["throughput_gauge_max_abs_mean_log_g"] < 1e-5 and g["filters_without_training_exposures"] == ["A2"]
    assert "A2" not in g["background_mean_per_filter"]


def test_ordinary_field_has_nothing_unseen():
    _, _, model, _ = build(())
    assert model.unseen == []


def test_starting_sky_uses_only_bands_with_data():
    exps, fset, model, cfg = build(("A2",))
    mid = fset.index["A2"]
    sky = model.sky
    nb = len(fset)
    phi = model.Phi_band
    # a sky with a smooth spectrum, seen in A1 and A3 only; the empty band holds zeros as the coadd does
    k = torch.linspace(0, 1, phi.shape[1])
    A_true = (1.0 + 0.5 * k)[:, None, None].expand(-1, 3, 3)
    img = torch.einsum("bk,khw->bhw", phi, A_true)
    expect = float(img[mid, 0, 0])
    img0 = img.clone()
    img0[mid] = 0.0
    naive = torch.einsum("bk,khw->bhw", phi, spectral_init(img0, phi))[mid, 0, 0]
    bands = [b for b in range(nb) if b != mid]
    ok = torch.einsum("bk,khw->bhw", phi, spectral_init(img0, phi, bands=bands))[mid, 0, 0]
    assert float(naive) < 0.6 * expect                                       # the empty band pulls the start towards zero
    assert abs(float(ok) - expect) < 0.25 * expect                           # other bands only: the empty band is interpolated
    assert torch.isfinite(sky.sig_b).all() and float(sky.sig_b[mid]) > 0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
