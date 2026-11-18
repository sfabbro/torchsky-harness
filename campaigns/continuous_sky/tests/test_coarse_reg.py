"""Build B: catalogue-free coarse registration (--coarse-reg xcorr), tests 1 (synthetic) and the numerical core. CPU, about 1.5 minutes.
The real-file test (HST against JWST) is the stand-alone script tests/check_coarse_reg_hst.py (not a unit test).

1. coarse_shift: a known sub-pixel translation of an image with sources and noise is recovered with the right sign; a periodic pattern and pure
   noise are refused (second-peak ratio, peak height).
2. option defaults: --coarse-reg off, radius 1.0 arcsec, instruments default empty (every instrument except the reference one).
3. synthetic two-instrument field with 0.1 to 0.6 arcsec injected on the second instrument: recovered to better than a quarter of B's pixel (against
   the full target, see synth_coarse_reg.recovery), the visit scatter and per-exposure lines are logged; after a short fit the held-out score is
   close to the one without injected translations.
"""
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import synth_coarse_reg as sc  # noqa: E402
from fieldop.cli import parse_config  # noqa: E402
from fieldop.coarse_reg import coarse_register, coarse_shift  # noqa: E402
from fieldop.tiles import TileSet  # noqa: E402


def _sky(n=160, seed=0, n_src=60):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij")
    return yy, xx, torch.rand(n_src, 2, generator=g, dtype=torch.float64) * (n - 40) + 20, torch.rand(n_src, generator=g, dtype=torch.float64) * 8 + 1


def _render(yy, xx, pos, amp, dx, dy, width=1.6):
    img = torch.zeros_like(yy)
    for (px, py), a in zip(pos.tolist(), amp.tolist()):
        img += a * torch.exp(-0.5 * (((xx - px - dx) ** 2 + (yy - py - dy) ** 2) / width ** 2))
    return img


def test_known_shift_sign_and_refusals():
    yy, xx, pos, amp = _sky()
    pix = 0.1
    J = np.array([[-pix, 0.0], [0.0, pix]])                      # X toward the west: columns run east (X decreases with column)
    m = _render(yy, xx, pos, amp, 0.0, 0.0)
    g = torch.Generator().manual_seed(5)
    d = _render(yy, xx, pos, amp, -3.4, 2.3) + 0.3 * torch.randn(m.shape, generator=g, dtype=torch.float64)    # data = model sampled at p + delta, delta = (3.4, -2.3) px
    # d(p) = m(p + delta): a feature of m at pixel q sits in d at q - delta, so delta = (+3.4, -2.3) columns / rows, t = J delta
    ok = torch.ones_like(m, dtype=torch.bool)
    r = coarse_shift(d, ok, m, ok, J, radius_arcsec=1.2)
    assert r["ok"], r
    t_true = J @ np.array([3.4, -2.3])
    assert abs(r["tX"] - t_true[0]) < 0.02 and abs(r["tY"] - t_true[1]) < 0.02, (r["tX"], r["tY"], t_true)
    assert r["peak"] > 0.5 and r["ratio"] > 1.5
    # a periodic pattern has many equal peaks: refused as ambiguous; pure noise: low peak
    per = torch.sin(xx * 0.9) * torch.sin(yy * 0.9) + 0.05 * torch.randn(m.shape, generator=g, dtype=torch.float64)
    rp = coarse_shift(per, ok, torch.sin(xx * 0.9) * torch.sin(yy * 0.9), ok, J, radius_arcsec=1.2)
    assert not rp["ok"] and "ambiguous" in str(rp["reason"]), rp
    rn = coarse_shift(torch.randn(m.shape, generator=g, dtype=torch.float64), ok, m, ok, J, radius_arcsec=1.2)
    assert not rn["ok"], rn


def test_option_defaults():
    a = parse_config([])
    assert (a.coarse_reg, a.coarse_reg_radius, a.coarse_reg_instruments, a.coarse_reg_min_ratio, a.coarse_reg_min_peak) == ("off", 1.0, "", 1.5, 0.05)
    assert parse_config(["--coarse-reg", "xcorr", "--coarse-reg-inst", "all"]).coarse_reg_instruments == "all"


def test_synthetic_two_instruments_recovery_and_heldout():
    logs = []
    model, exps, cfg = sc.build(inject=1, coarse=1)
    ts = TileSet.single(model, exps, cfg, logs.append)
    coarse_register(model, exps, cfg, log=logs.append, ts=ts, inst_names=["A", "B"])
    r = sc.recovery(model, exps, gauge_free=True)
    assert r["injected_min"] >= 0.1 and r["injected_max"] <= 0.6 + 1e-9
    assert r["max_err_vs_full_target_pix"] < 0.25, r                         # better than a quarter of B's pixel
    assert sum("coarse registration B1_" in s and "peak" in s and "ratio" in s for s in logs) == r["n"]     # one line per exposure
    assert sum("coarse registration visit" in s for s in logs) >= 1             # the visit scatter is reported
    a_rows = model.affine[[e.eid for e in exps if e.inst == 0], 4:6]
    assert (a_rows - a_rows[0]).abs().max() == 0                                # instrument A (reference): only the common-translation gauge applied, equal rows
    # held-out score after a short fit
    a = sc.run(1, 1, 60)
    b = sc.run(0, 0, 60)
    assert a["heldout_chi2"]["B"] < 0.5 * b["heldout_chi2"]["B"], (a["heldout_chi2"], b["heldout_chi2"])   # the held-out B exposure is far better with the coarse step


if __name__ == "__main__":
    import time
    t = time.perf_counter()
    test_known_shift_sign_and_refusals()
    test_option_defaults()
    test_synthetic_two_instruments_recovery_and_heldout()
    print(f"ok ({time.perf_counter() - t:.1f} s)")
