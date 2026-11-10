"""CPU tests of the gauges of the field operator (fieldop F17, F18).

* Throughput gauge (enforced): for a reference set of exposures, the T-weighted mean of the learned log throughput gamma is zero at every lambda
  node, i.e. the mean g of the reference set equals the published curves. Checked with random (perturbed) networks, random per-exposure gains,
  overlapping filters, after optimisation steps that try to move the mean, and with a reference set restricted to one instrument.
* Declared, logged gauges: common shift (and its penalty), common blur, background vs smooth sky: the report is finite and responds to the
  quantity it measures.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fieldop.bandpass import FilterSet, box_filter  # noqa: E402
from fieldop.cli import default_config, make_sky  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402


def build(insts=None, **kw):
    exps, fset, meta, truth, ins = make_field(seed=2, n_per_filter=4, insts=insts)
    b = dict(sky_pitch=0.04, render_pitch=0.02, level_nh=16, kernel_extent=0.6, n_lambda=4, k_hidden=16, pattern_grid=0, stripes=0,
             warm_kernel_steps=0, amplitude="structured", tau0_g=0.3)
    b.update(kw)
    cfg = default_config(**b)
    torch.manual_seed(0)
    sky, _ = make_sky(exps, fset, cfg, 2.0, "cpu")
    return exps, fset, cfg, FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))


def overlapping_insts():
    return [SynthInstrument("A", 0.06, 20, 6.5, 0.05, 1.5, filters=[("A1", 1.0, 1.5), ("A2", 1.3, 1.9)]),
            SynthInstrument("B", 0.11, 12, 6.5, 0.09, 3.5, filters=[("B1", 1.6, 2.4)])]


def perturb(m):
    torch.manual_seed(1)
    with torch.no_grad():
        for net in (m.gbar, m.gdet, m.gep):
            for p in net.parameters():
                p.add_(0.5 * torch.randn_like(p))
        m.u.normal_(0, 1.0)


def mean_logg_ref(m):
    with torch.no_grad():
        return m.gauge_report()["throughput_gauge_max_abs_mean_log_g"]


def test_throughput_gauge_exact_random_networks():
    exps, fset, cfg, m = build(insts=overlapping_insts())
    perturb(m)
    v = mean_logg_ref(m)
    # the raw (unprojected) mean is far from zero, the projected one is zero to float precision
    with torch.no_grad():
        P = m.prepare()
        raw = m.gamma_raw_ref(P)
        Tw = m.T_ref[m.e_filt[m.gref]]
        raw_mean = ((Tw * raw).sum(0) / Tw.sum(0).clamp(min=1e-12)).abs().max()
    print(f"throughput gauge: raw max |mean log g| {float(raw_mean):.3f} -> projected {v:.2e}")
    assert float(raw_mean) > 0.05 and v < 1e-5


def test_throughput_gauge_survives_optimisation():
    exps, fset, cfg, m = build(insts=overlapping_insts())
    perturb(m)
    params = list(m.gbar.parameters()) + list(m.gdet.parameters()) + [m.u]
    opt = torch.optim.Adam(params, lr=0.05)
    for _ in range(20):
        P = m.prepare()
        e = exps[3]
        g = m.gamma_at(torch.tensor(e.eid), torch.zeros(5, 2), P)
        loss = -(g.mean())              # tries to raise the throughput of one exposure (and with it the reference mean)
        opt.zero_grad()
        loss.backward()
        opt.step()
    v = mean_logg_ref(m)
    print(f"after 20 adversarial steps: max |mean log g| over the reference set {v:.1e} (float32)")
    assert v < 2e-4


def test_throughput_gauge_reference_instrument_only():
    exps, fset, cfg, m = build(insts=overlapping_insts(), gauge_ref_inst=0)
    perturb(m)
    with torch.no_grad():
        P = m.prepare()
        g = m.gamma_raw_ref(P) - P["m"].flatten()[None]
        Tw = m.T_ref[m.e_filt[m.gref]]
        mean = (Tw * g).sum(0) / Tw.sum(0).clamp(min=1e-12)
        covered = Tw.sum(0) > 0
    assert set(int(i) for i in m.e_inst[m.gref]) == {0}
    assert float(mean[covered].abs().max()) < 1e-5
    # lambda nodes of B1 outside instrument A's coverage carry no gauge condition (declared): m = 0 there
    assert bool((~covered).any())


def test_logged_gauges_respond():
    exps, fset, cfg, m = build()
    rep0 = m.gauge_report()
    for k in ("common_shift_mas", "common_blur_core_m2_arcsec2", "background_mean_per_filter", "sky_band_p05", "throughput_gauge_max_abs_mean_log_g"):
        assert k in rep0
    assert np.all(np.isfinite(rep0["common_shift_mas"])) and np.isfinite(rep0["common_blur_core_m2_arcsec2"])
    with torch.no_grad():
        m.affine.data[:, 4] += 0.5     # +5 mas common shift in X on every exposure
        m.bkg.data[:, 0] += 2.0
    rep1 = m.gauge_report()
    assert abs(rep1["common_shift_mas"][0] - rep0["common_shift_mas"][0] - 5.0) < 1e-3
    f0 = list(rep0["background_mean_per_filter"])[0]
    assert abs(rep1["background_mean_per_filter"][f0] - rep0["background_mean_per_filter"][f0] - 2.0) < 1e-4
    # the shift penalty used in training pulls the common shift back
    opt = torch.optim.Adam([m.affine], lr=0.05)
    for _ in range(50):
        P = m.prepare()
        loss = (m.shift_gauge(P) / 0.001).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert abs(m.gauge_report()["common_shift_mas"][0]) < 0.5
