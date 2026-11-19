"""Regression tests of the bugs fixed by the independent audit of 17 Nov 2026 (AUDIT_2026-11-17.md, section E), and of the read-only size
report of the learned terms (section B). CPU, under a minute in all.

1. Filter curves of a second instrument: load_field fetched EVERY filter curve as the first instrument's (sorted name), so an HST filter in a
   NIRCam field was looked up as "JWST/NIRCam.F160W" (failure) or, for a name shared by two instruments, silently got the other's curve.
2. Telescope diameter of the blur gauge: the gauge read meta["diameter_m"], which no loader sets, so it always used 6.5 m while the kernel
   warm start used each instrument's diameter; the sharpest-kernel class (the blur reference) could then be the wrong filter.
3. nuisance_sizes: reports the end-of-fit sizes and leaves the fitted state bit-identical.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fieldop import bandpass, data, train  # noqa: E402
from fieldop.cli import default_config, filt_info_of, make_sky  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402


# ----------------------------------------------------------------------------- 1. filter curve of each filter's own instrument
def _fake_splits(tmp_path, exps):
    sp = {"boxes": [{"id": 0, "ra": 10.0, "dec": -20.0, "size_arcsec": 2.0}],
          "splits": {"dither": {"heldout": [], "train": [e["rootname"] for e in exps]}},
          "exposures": exps}
    p = tmp_path / "splits_TEST.json"
    p.write_text(json.dumps(sp))
    return str(p)


def _fake_read_cuts(path, boxes, half_data, shift0, inst=None):
    n = 48
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    X, Y = (xx - n / 2) * 0.05, (yy - n / 2) * 0.05
    r = dict(d=np.ones((n, n), np.float32), sig=np.ones((n, n), np.float32), valid=np.ones((n, n), bool), X=X, Y=Y, cut0=(0, 0),
             det_shape=(n, n), J=np.array([[0.05, 0.0], [0.0, 0.05]]), detector="", instrument="", bkg0=1.0)
    return [(0, r)], None


def _load(tmp_path, monkeypatch, exps):
    calls = []

    def fake_svo(name, instrument, svo_id, Q, cache_dir=None):
        calls.append((name, instrument, svo_id))
        return bandpass.box_filter(name, instrument, 1.0, 1.3, Q)
    monkeypatch.setattr(bandpass, "svo_filter", fake_svo)
    monkeypatch.setattr(data, "_read_cuts", _fake_read_cuts)
    cfg = default_config(splits=_fake_splits(tmp_path, exps), n_lambda=3)
    data.load_field(cfg, "cpu", log=lambda s: None)
    return calls


def _exp(root, filt, inst, det):
    return {"rootname": root, "filter": filt, "instrument": inst, "dither_group": root, "mjd": 60000.0, "boxes": {"0": 1},
            "path": f"/nonexistent/{root}_cal.fits", "file": f"{root}_cal.fits", "detector": det}


def test_filter_curve_uses_each_filters_instrument(tmp_path, monkeypatch):
    calls = _load(tmp_path, monkeypatch, [_exp("j1", "F150W", "NIRCAM", "NRCB1"), _exp("h1", "F160W", "WFC3", "IR")])
    got = {c[0]: (c[1], c[2]) for c in calls}
    assert got["F150W"] == ("NIRCAM", "JWST/NIRCam.F150W")
    assert got["F160W"] == ("WFC3_IR", "HST/WFC3_IR.F160W")   # before the fix: ("NIRCAM", "JWST/NIRCam.F160W"); label = instrument key since 18 Nov


def test_filter_name_shared_by_two_instruments_is_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _load(tmp_path, monkeypatch, [_exp("a1", "F814W", "ACS", "WFC"), _exp("w1", "F814W", "WFC3", "UVIS")])


# ----------------------------------------------------------------------------- 2. blur gauge uses the same diameter as the warm start
def _small_field(insts):
    return make_field(seed=1, n_per_filter=3, n_held=1, insts=insts, centre_range=0.3, rot_choices=(0,), patch_px=12, truth_half=0.8)


def test_blur_gauge_uses_instrument_diameter():
    # instrument B: small pixels but a 0.85 m aperture. With the assumed 6.5 m B's filter looks sharpest; with its true aperture A's is.
    insts = [SynthInstrument("A", 0.10, 20, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3)]),
             SynthInstrument("B", 0.03, 40, 0.85, 0.15, 1.15, filters=[("B1", 1.0, 1.3)])]
    exps, fset, meta, _, _ = _small_field(insts)
    cfg = default_config(sky_pitch=0.08, render_pitch=0.01, level_nh=4, kernel_extent=0.3, n_lambda=3, k_hidden=8, stripes=0)
    sky, _ = make_sky(exps, fset, cfg, 1.2, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    fi = filt_info_of(exps, fset, [i.name for i in insts])          # the kernel warm start's inputs
    want = {f: (0.42 * float(model.piv[f]) * 1e-6 / fi[f]["diameter_m"] * 206264.8) ** 2 + fi[f]["pixel_arcsec"] ** 2 / 12 for f in fi}
    assert fset.names[model.blur_ref] == "A1"                        # before the fix: "B1"
    assert min(want, key=want.get) == model.blur_ref
    for f in want:
        assert abs(model.blur_sig2[f] - want[f]) < 1e-12


# ----------------------------------------------------------------------------- 3. size report: present, finite, read-only
def _hash(model):
    h = hashlib.sha256()
    st = model.state_dict()
    for k in sorted(st):
        h.update(k.encode())
        h.update(st[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_nuisance_sizes_reported_and_read_only():
    torch.manual_seed(0)
    insts = [SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3)]),
             SynthInstrument("B", 0.12, 20, 6.5, 0.10, 2.4, filters=[("B1", 2.0, 2.8)])]
    exps, fset, meta, _, _ = make_field(seed=1, n_per_filter=4, n_held=1, insts=insts, centre_range=0.4, rot_choices=(0, 1),
                                        patch_px=12, truth_half=1.0)
    cfg = default_config(sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.8, region_px=12, n_lambda=3, k_hidden=16,
                         stripes=0, batch_patches=3, steps=3, budget_min=1e6, tau_mode="fixed", log_every=2, seed=0, held_steps=2, held_reg_steps=1)
    sky, _ = make_sky(exps, fset, cfg, 1.6, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    train.fit(model, exps, cfg, log=lambda s: None)
    held = [e.eid for e in exps if e.split == "held"]
    train.fit_heldout(model, exps, held, cfg, log=lambda s: None)
    h0 = _hash(model)
    rng0 = torch.get_rng_state().clone()
    out = train.nuisance_sizes(model, exps)
    assert _hash(model) == h0 and torch.equal(torch.get_rng_state(), rng0)
    for k in ("noise_floor_fraction", "noise_scale_exp_h", "train", "held", "kernel_wing_fraction"):
        assert k in out, k
    assert set(out["noise_scale_exp_h"]) == {"A/A1", "B/B1"}
    for s in ("train", "held"):
        for k in ("bkg_const_minus_median_over_sigma", "translation_mas", "gain_u_raw"):
            assert np.isfinite(out[s][k]["rms"]), (s, k)
    for f, w in out["kernel_wing_fraction"].items():
        assert 0.0 <= w["beyond_0.5_arcsec"] <= 1.0 and abs(w["integral_mean_over_lambda_nodes"] - 1.0) < 1e-3, (f, w)
    json.dumps(out)
