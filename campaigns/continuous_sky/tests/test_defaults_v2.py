"""Tests of the defaults and fixes of 18 Nov 2026 (DEFAULTS_V2_2026-11-18.md). CPU, tiny synthetic two-instrument field, about 2 minutes in all.
The proof that --legacy-defaults reproduces the recorded state hashes is not here (it takes minutes): see tests/synth_small.py and the report.

1. defaults, removed options, --legacy-defaults
2. noise-scale clamp and WARNING lines (audit finding 3)
3. several instruments: refusal, kernel normalisation radius, mean log-throughput of the other instruments (findings 5, 9)
4. held-out cross-fitting (finding 8): the other half's data never reach a prediction
5. --heldout-only state loading: complete state, a key removed, a key added
6. epoch axes per instrument (finding 12)
7. held-out optimiser steps only the rows present (finding 14)
8. zero-initialised throughput and distortion fields (finding 15)
"""
import copy
import hashlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fieldop import data, train  # noqa: E402
from fieldop.cli import default_config, load_saved_state, make_sky, parse_config, state_for_saving  # noqa: E402
from fieldop.model import FieldOperator, resolve_instrument_options  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402


def build(steps=0, n_held=1, **over):
    """The synthetic two-instrument field (A: two filters, B: one), small; new defaults (legacy_defaults False), reference instrument 0."""
    torch.manual_seed(over.pop("init_seed", 0))
    insts = [SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", 0.12, 20, 6.5, 0.10, 2.4, filters=[("B1", 2.0, 2.8)])]
    exps, fset, meta, truth, _ = make_field(seed=1, n_per_filter=4, n_val=1, n_held=n_held, insts=insts, centre_range=0.4, rot_choices=(0, 1, 2, 3),
                                            patch_px=12, truth_half=1.0)
    kw = dict(legacy_defaults=False, gauge_ref_inst=0, sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.8, region_px=12, n_lambda=3,
              k_hidden=16, pattern_grid=0, stripes=0, batch_patches=3, steps=steps, budget_min=1e6, tau_mode="fixed", log_every=2, seed=0,
              held_steps=6, held_reg_steps=4, sky_zero="median")
    kw.update(over)
    cfg = default_config(**kw)
    sky, _ = make_sky(exps, fset, cfg, 1.6, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    logs = []
    if steps:
        train.fit(model, exps, cfg, log=logs.append)
    return model, exps, cfg, sky, logs


def state_hash(model):
    h = hashlib.sha256()
    st = model.state_dict()
    for k in sorted(st):
        h.update(k.encode())
        h.update(st[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- 1. defaults
def test_new_defaults_and_legacy_flag():
    a = parse_config([])
    assert (a.sky_zero, a.sky_positivity_margin, a.noise_floor_max, a.noise_scale_max, a.nuisance_steps) == ("median", 0.0, 0.2, 2.0, "exposure")
    assert (a.held_xfit, a.held_step_rows, a.epoch_per_instrument, a.field_init, a.gauge_ref_inst, a.kernel_norm_radius) == (1, 1, 1, "random", None, None)
    b = parse_config(["--legacy-defaults"])
    assert (b.sky_zero, b.noise_floor_max, b.noise_scale_max, b.nuisance_steps, b.held_xfit, b.held_step_rows, b.epoch_per_instrument,
            b.field_init) == ("median", 0.0, 0.0, "mean", 0, 0, 0, "random")
    assert (b.gauge_ref_inst, b.kernel_norm_radius) == (None, None)        # findings 5 and 9 are not undone by the flag
    c = parse_config(["--legacy-defaults", "--noise-floor-max", "0.3", "--held-xfit", "1"])
    assert c.noise_floor_max == 0.3 and c.held_xfit == 1 and c.sky_zero == "median"       # explicit options win


@pytest.mark.parametrize("opt", ["--tau-prior", "--sky-tv", "--xfit", "--xfit-frac", "--gauge-shift"])
def test_removed_options_are_gone(opt):
    with pytest.raises(SystemExit):
        parse_config([opt, "1"])
    a = parse_config([])
    assert not any(hasattr(a, k) for k in ("tau_prior", "sky_tv", "xfit", "xfit_frac", "gauge_shift"))


# ----------------------------------------------------------------------------- 2. noise-scale clamp
def test_noise_scale_clamp_and_warnings():
    model, exps, cfg, sky, logs = build(steps=8, floor0=0.3, noise_floor_max=0.05, noise_scale_max=1.002)
    h = 3 * torch.tanh(model.hs / 3)
    assert float(torch.exp(h).max()) <= 1.002 + 1e-6 and float(torch.exp(h).min()) >= 1 / 1.002 - 1e-6
    assert max(train.noise_floor_values(model)) <= 0.05 + 1e-6
    w = [s for s in logs if s.startswith("WARNING") and "sits at its bound" in s]
    assert len(w) == 2 and "at step" in w[0] and "end of the fit" in w[1] and "noise scale" in w[0] and "noise scale" in w[1], w
    assert train.bound_report(model, cfg)["noise_scale"]
    assert train.noise_scale_raw_max(0) is None and train.noise_scale_raw_max(100.0) is None
    assert abs(3 * np.tanh(train.noise_scale_raw_max(2.0) / 3) - np.log(2.0)) < 1e-9


def test_noise_scale_bound_off_is_old_behaviour():
    model, _, cfg, _, logs = build(steps=4, floor0=0.3, noise_floor_max=0.0, noise_scale_max=0.0)
    assert max(train.noise_floor_values(model)) > 0.2
    assert not any("sits at its bound" in s for s in logs)


# ----------------------------------------------------------------------------- 3. several instruments
def test_several_instruments_need_explicit_reference_and_default_radius():
    c = SimpleNamespace(gauge_ref_inst=None, kernel_norm_radius=None)
    with pytest.raises(SystemExit):
        resolve_instrument_options(c, 2)
    c = SimpleNamespace(gauge_ref_inst=1, kernel_norm_radius=None)
    resolve_instrument_options(c, 2)
    assert (c.gauge_ref_inst, c.kernel_norm_radius) == (1, 2.0)
    c = SimpleNamespace(gauge_ref_inst=None, kernel_norm_radius=None)
    resolve_instrument_options(c, 1)
    assert (c.gauge_ref_inst, c.kernel_norm_radius) == (-1, 0.0)
    c = SimpleNamespace(gauge_ref_inst=0, kernel_norm_radius=0.7)
    resolve_instrument_options(c, 2)
    assert c.kernel_norm_radius == 0.7                                       # explicit value kept
    with pytest.raises(SystemExit):                                          # the synthetic two-instrument field, reference not given
        build(gauge_ref_inst=None, kernel_norm_radius=None)


def test_two_instrument_field_radius_and_throughput_log():
    model, exps, cfg, sky, logs = build(steps=6, kernel_norm_radius=None)
    assert cfg.kernel_norm_radius == 2.0 and model.n_inst == 2
    r = model.instrument_mean_log_throughput(names=["A", "B"])
    assert set(r) == {"A", "B"} and abs(r["A"]["mean_log_g"]) < 1e-4 and np.isfinite(r["B"]["mean_log_g"])
    assert r["A"]["n_train_exposures"] + r["B"]["n_train_exposures"] == int(model.train_mask.sum())


# ----------------------------------------------------------------------------- 4. held-out cross-fitting
def _run_stage(model, exps, cfg, rows0, snap0, perturb_half=None):
    held = [e.eid for e in exps if e.split == "held"]
    saved = {}
    if perturb_half is not None:
        for c in held:
            e = exps[c]
            saved[c] = e.d.clone()
            e.d = torch.where(train.half_map(e) == perturb_half, e.d + 50.0 * e.sig, e.d)
    train.restore_rows(model, rows0, snap0)
    torch.manual_seed(7)                                                      # the render jitter of training steps draws from the global stream
    stage = train.heldout_stage(model, exps, held, cfg, log=lambda s: None)
    out = {c: stage["pred"](exps[c]) for c in held}
    for c, d in saved.items():
        exps[c].d = d
    return out


def test_cross_fit_never_uses_the_other_half():
    model, exps, cfg, sky, _ = build(steps=4, held_xfit=1)
    held = [e.eid for e in exps if e.split == "held"]
    rows0 = torch.as_tensor(sorted({exps[c].eid for c in held}))
    snap0 = train.snapshot_rows(model, rows0)
    base = _run_stage(model, exps, cfg, rows0, snap0)
    pert = _run_stage(model, exps, cfg, rows0, snap0, perturb_half=1)        # the data of half 1 are changed by 50 sigma
    for c in held:
        e = exps[c]
        m0, _, _, r0 = base[c]
        m1, _, _, r1 = pert[c]
        h = train.half_map(e)
        ok = torch.isfinite(m0) & torch.isfinite(m1)
        # pixels of half 1 are predicted by terms fitted on half 0: the change of half 1's data does not reach them
        assert torch.equal(m0[(h == 1) & ok], m1[(h == 1) & ok]), "half-1 predictions changed with half-1 data"
        # pixels of half 0 come from the fit on half 1: they do change
        assert float((m0[(h == 0) & ok] - m1[(h == 0) & ok]).abs().max()) > 1e-4
        # the old all-pixel fit saw both halves and is reported next to it
        assert "all_pixel_old" in r0 and np.isfinite(r0["chi2_model_noise"]) and np.isfinite(r0["all_pixel_old"]["chi2_model_noise"])
        assert r0["n"] == r0["all_pixel_old"]["n"]
        assert r1["all_pixel_old"]["chi2"] != r0["all_pixel_old"]["chi2"]
    s = train.summarise_scores({exps[c].name: base[c][3] for c in held})
    assert set(s) == {"written", "all_pixel_old"} and s["written"]["n_px"] == s["all_pixel_old"]["n_px"]


def test_held_xfit_zero_is_the_old_single_fit():
    model, exps, cfg, sky, _ = build(steps=3, held_xfit=0)
    held = [e.eid for e in exps if e.split == "held"]
    stage = train.heldout_stage(model, exps, held, cfg, log=lambda s: None)
    assert set(stage["snaps"]) == {"all"}
    m, pv, S, rec = stage["pred"](exps[held[0]])
    assert "all_pixel_old" not in rec and "chi2_model_noise" in rec
    assert set(train.summarise_scores({"x": rec})) == {"written"}


# ----------------------------------------------------------------------------- 5. --heldout-only: loading saved states
def test_state_loading_complete_missing_and_extra(tmp_path):
    model, exps, cfg, sky, _ = build(steps=3)
    p = tmp_path / "fieldop_state.pt"
    torch.save(state_for_saving(model, {0: sky}), p)
    ref = state_hash(model)

    def fresh():
        m2, _, _, sky2, _ = build(init_seed=5)                              # another random start
        assert state_hash(m2) != ref
        return m2, sky2

    logs = []
    m2, sky2 = fresh()
    rep = load_saved_state(m2, {0: sky2}, torch.load(p, weights_only=False), log=logs.append)
    assert state_hash(m2) == ref
    assert not rep["model"]["missing"] and not rep["model"]["extra_ignored"] and not rep["skies"][0]["extra_ignored"]
    # a key removed (the noise scales) and keys added (the removed all-zero sky.resid of the old item sky; a stray model tensor)
    st = torch.load(p, weights_only=False)
    del st["hs"]
    st["not_a_parameter"] = torch.zeros(3)
    st["skies"][0]["resid"] = torch.zeros(4, 4)
    logs = []
    m3, sky3 = fresh()
    rep = load_saved_state(m3, {0: sky3}, st, log=logs.append)
    assert rep["model"]["missing"] == ["hs"] and rep["model"]["set_to_neutral"] == ["hs"] and float(m3.hs.abs().max()) == 0.0
    assert rep["model"]["extra_ignored"] == ["not_a_parameter"] and rep["skies"][0]["extra_ignored"] == ["resid"]
    txt = "\n".join(logs)
    assert "missing ['hs']" in txt and "extra keys ignored ['not_a_parameter']" in txt and "extra keys ignored ['resid']" in txt
    a, b = m3.state_dict(), model.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a if k != "hs")               # everything else loaded exactly
    # the held-out stage runs on the loaded state and gives the same prediction as on the original model
    held = [e.eid for e in exps if e.split == "held"]
    m4, sky4 = fresh()
    load_saved_state(m4, {0: sky4}, torch.load(p, weights_only=False), log=lambda s: None)
    s_a = train.heldout_stage(model, exps, held, cfg, log=lambda s: None)
    s_b = train.heldout_stage(m4, exps, held, cfg, log=lambda s: None)
    for c in held:
        ra, rb = s_a["pred"](exps[c])[3], s_b["pred"](exps[c])[3]
        assert abs(ra["chi2"] - rb["chi2"]) < 1e-5 * max(1.0, ra["chi2"]), (ra["chi2"], rb["chi2"])


# ----------------------------------------------------------------------------- 6. epoch axes per instrument
def test_epoch_coordinates_per_instrument():
    mjd = [0.0, 0.5, 2.0, 1000.0, 1000.0]               # A: three exposures over 2 days; B: one epoch, years later
    inst = ["A", "A", "A", "B", "B"]
    new = data.epoch_coordinates(mjd, inst, 1.0, True)
    old = data.epoch_coordinates(mjd, inst, 1.0, False)
    assert new[:3] == [-1.0, -0.5, 1.0] and new[3:] == [0.0, 0.0]            # A spans its own range; B has no span of its own: axes off (t = 0)
    assert old[0] == -1.0 and abs(old[1] + 0.999) < 1e-9 and abs(old[2] + 0.996) < 1e-9 and old[3] == 1.0   # old: A squeezed into the first 0.4 per cent of the range
    # B alone observed years after a one-day A field: the old rule switched the epoch axes on for A, the new rule does not
    mjd2, inst2 = [0.0, 0.2, 0.4, 900.0], ["A", "A", "A", "B"]
    assert data.epoch_coordinates(mjd2, inst2, 1.0, True) == [0.0, 0.0, 0.0, 0.0]
    assert len(set(np.round(data.epoch_coordinates(mjd2, inst2, 1.0, False), 6))) > 1


def _with(cfg, v):
    c = copy.copy(cfg)
    c.epoch_per_instrument = v
    return c


def test_epoch_axes_decision_per_instrument():
    base = default_config(legacy_defaults=False, tau_mode="fixed")
    meta = {"t": [-1.0, -1.0, 1.0, 1.0], "inst": [0, 0, 1, 1], "train": [True] * 4}          # each instrument has ONE epoch, at a different t
    on_old = FieldOperator.tau_spec(_with(base, 0), meta)
    on_new = FieldOperator.tau_spec(_with(base, 1), meta)
    assert on_old["k_time"][0] != "off" and on_new["k_time"][0] == "off"
    meta2 = {"t": [-1.0, 1.0, 0.0, 0.0], "inst": [0, 0, 1, 1], "train": [True] * 4}          # instrument 0 has two epochs of its own
    assert FieldOperator.tau_spec(_with(base, 1), meta2)["k_time"][0] != "off"


# ----------------------------------------------------------------------------- 7. held-out steps only the rows present
def _held_run(rows_policy, monkeypatch=None):
    model, exps, cfg, sky, _ = build(steps=2, held_step_rows=rows_policy, batch_patches=1, held_steps=8, held_reg_steps=4, n_held=1)
    held = [e.eid for e in exps if e.split == "held"]
    calls = []
    orig = train.step_rows

    def spy(opt, tensors, eids, rows=None):
        before = [t.detach().clone() for t in tensors]
        orig(opt, tensors, eids, rows=rows)
        absent = torch.ones(tensors[0].shape[0], dtype=torch.bool)
        absent[eids] = False
        calls.append(all(torch.equal(b[absent], t.detach()[absent]) for b, t in zip(before, tensors)) and bool(absent.any()))
        return None
    train.step_rows = spy
    try:
        train.fit_heldout(model, exps, held, cfg, log=lambda s: None, half=None)
    finally:
        train.step_rows = orig
    rows = torch.as_tensor(sorted({exps[c].eid for c in held}))
    return calls, train.snapshot_rows(model, rows)


def test_heldout_steps_only_rows_present():
    calls1, snap1 = _held_run(1)
    assert len(calls1) >= 8 and all(calls1), "a held-out step moved a row absent from its batch (or no row was absent)"
    calls0, snap0 = _held_run(0)
    assert calls0 == []                                                      # old policy does not go through step_rows
    assert any(not torch.equal(snap0[k], snap1[k]) for k in snap0)           # and the two policies give different held-out terms


# ----------------------------------------------------------------------------- 8. zero-initialised fields
def test_fields_start_at_exactly_zero():
    model, exps, cfg, sky, _ = build(field_init="zero")
    x = torch.randn(7, 27)
    assert torch.equal(model.gdet(x), torch.zeros(7, 1)) and torch.equal(model.gep(torch.randn(7, 18)), torch.zeros(7, 1))
    assert torch.equal(model.Dst(torch.randn(7, 27)), torch.zeros(7, 2)) and torch.equal(model.Ddr(torch.randn(7, 23)), torch.zeros(7, 2))
    P = model.prepare()
    xi = torch.zeros(1, 2)
    e = torch.as_tensor(0)
    assert float(model.gamma_at(e, xi, P).abs().max()) == 0.0
    D = model.distortion(e, xi, P)
    assert D is None or float(D.abs().max()) == 0.0
    old, *_ = build(field_init="random")
    assert float(old.gdet(x).abs().max()) > 0 and float(old.Dst(torch.randn(7, 27)).abs().max()) > 0
    # the other parameters are not shifted by the change (same random stream): the kernel net is identical
    assert all(torch.equal(a, b) for (ka, a), (kb, b) in zip(model.net.state_dict().items(), old.net.state_dict().items()))
    # and a few steps from the zero start run to finite values
    m2, _, _, _, logs = build(steps=4, field_init="zero")
    assert all(np.isfinite(v).all() for v in [p.detach().numpy() for p in m2.parameters()])


# ----------------------------------------------------------------------------- 5b. --heldout-only through the command line (loader replaced by the synthetic field)
def _cli_run(tmp_path, monkeypatch, extra, out, method):
    from fieldop.cli import main
    insts = [SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", 0.12, 20, 6.5, 0.10, 2.4, filters=[("B1", 2.0, 2.8)])]

    def fake_load(cfg, dev, log=print):
        exps, fset, meta, truth, _ = make_field(seed=1, n_per_filter=4, n_val=1, n_held=1, insts=insts, centre_range=0.4, rot_choices=(0, 1, 2, 3),
                                                patch_px=12, truth_half=1.0)
        for i, e in enumerate(exps):
            e.cid = i
        info = dict(box={}, boxes=[{}], tiles=[0], tile_c0={0: [0, 0]}, half_data=1.9, dets=list(range(meta["n_det"])), chans=[f"c{i}" for i in range(meta["n_chan"])],
                    insts=["A", "B"], filters=list(fset.names), n_by_split={s: sum(1 for e in exps if e.split == s) for s in ("train", "val", "held")},
                    epoch_span_days=0.0, shifts_applied=0, n_exposures=len(exps), n_cuts=len(exps), cuts_per_tile={0: {}}, host_gb=0.0, multi_tile=False, val_groups=[])
        return exps, fset, meta, info
    monkeypatch.setattr(data, "load_field", fake_load)
    common = ["--sky-pitch", "0.08", "--render-pitch", "0.04", "--level-nh", "8", "--kernel-extent", "0.8", "--region-px", "12", "--n-lambda", "3",
              "--k-hidden", "16", "--pattern-grid", "0", "--stripes", "0", "--batch-patches", "3", "--tau-mode", "fixed", "--log-every", "2",
              "--held-steps", "6", "--held-reg-steps", "4", "--protocol-root", str(tmp_path / "proto"), "--warm-kernel-steps", "5", "--reg-prepass-steps", "3",
              "--reg-prepass-iters", "1", "--budget-min", "100000", "--sky-zero", "median", "--device", "cpu", "--out", str(out), "--method-name", method]
    return main(extra + common)


def test_heldout_only_command_line(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):                                          # two instruments, no --gauge-ref-inst: refuses to start
        _cli_run(tmp_path, monkeypatch, ["--steps", "2"], tmp_path / "r0", "m0")
    m1 = _cli_run(tmp_path, monkeypatch, ["--steps", "3", "--gauge-ref-inst", "0"], tmp_path / "r1", "m1")
    assert m1["config"]["kernel_norm_radius"] == 2.0 and set(m1["instrument_mean_log_throughput"]) == {"A", "B"}
    state = tmp_path / "r1" / "fieldop_state.pt"
    assert state.exists()
    m2 = _cli_run(tmp_path, monkeypatch, ["--heldout-only", "--state", str(state), "--gauge-ref-inst", "0"], tmp_path / "r2", "m2")
    assert m2["fit"].get("heldout_only") and not (tmp_path / "r2" / "fieldop_state.pt").exists()
    assert m2["state_load_report"]["model"]["missing"] == [] and (tmp_path / "r2" / "metrics.json").exists()
    files = sorted((tmp_path / "proto" / "pred" / "SMACS" / "m2" / "dither").glob("*.npz"))
    assert len(files) == 3
    z = np.load(files[0], allow_pickle=True)
    assert set(z.files) == {"model", "model_valid", "meta"} and np.isfinite(z["model"][z["model_valid"]]).all()      # file format unchanged
    st0 = torch.load(state, weights_only=False)
    assert st0["launch_config"]["kernel_norm_radius"] == 2.0 and st0["launch_config"]["sky_form"] == "grid" and m2["state_load_report"]["config_differences"] == {}
    m2b = _cli_run(tmp_path, monkeypatch, ["--heldout-only", "--state", str(state), "--gauge-ref-inst", "0", "--kernel-norm-radius", "1.0"], tmp_path / "r2b", "m2b")
    assert m2b["state_load_report"]["config_differences"] == {"kernel_norm_radius": [2.0, 1.0]} or m2b["state_load_report"]["config_differences"] == {"kernel_norm_radius": (2.0, 1.0)}
    assert "WARNING: options that change the model's structure differ" in (tmp_path / "r2b" / "fieldop.log").read_text()
    # an older-style state: no stored configuration, one key missing, one added to the model and one to the sky
    st = torch.load(state, weights_only=False)
    del st["launch_config"]
    del st["hs"]
    st["old_buffer"] = torch.zeros(2)
    st["skies"][0]["resid"] = torch.zeros(3, 3)
    torch.save(st, tmp_path / "old_state.pt")
    m3 = _cli_run(tmp_path, monkeypatch, ["--heldout-only", "--state", str(tmp_path / "old_state.pt"), "--gauge-ref-inst", "0"], tmp_path / "r3", "m3")
    rep = m3["state_load_report"]
    assert rep["model"]["missing"] == ["hs"] and rep["model"]["extra_ignored"] == ["old_buffer"] and rep["skies"][0]["extra_ignored"] == ["resid"]
    log = (tmp_path / "r3" / "fieldop.log").read_text()
    assert "missing ['hs']" in log and "extra keys ignored ['old_buffer']" in log and "extra keys ignored ['resid']" in log
    assert "no stored launch configuration" in log and rep["config_differences"] is None
    assert m3["heldout_summary"]["written"]["chi2_input_noise"] > 0 and "all_pixel_old" in m3["heldout_summary"]


# ----------------------------------------------------------------------------- 9. size of the additive pattern over the observed footprint
def test_pattern_size_observed_and_beyond_degree_two():
    model, exps, cfg, sky, _ = build(pattern_grid=5)
    n = 5
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    with torch.no_grad():
        model.pattern[:] = 0.0
        model.pattern[0] = 0.3 * xx + 0.2 * yy                                  # a plane: exactly degree 1
        model.pattern[1] = torch.randn(n, n, generator=torch.Generator().manual_seed(1))
    nz = train.nuisance_sizes(model, exps)
    assert "detector_pattern_rms_over_sigma" in nz                              # old key kept
    obs, bey = nz["detector_pattern_observed_rms_over_sigma"], nz["detector_pattern_beyond_deg2_observed_rms_over_sigma"]
    assert set(obs) == set(bey) and len(obs) == 2
    k0, k1 = sorted(obs)[:2]
    assert obs[k0] > 0 and bey[k0] < 1e-3 * obs[k0] + 1e-9                      # plane: nothing beyond degree 2
    assert bey[k1] > 0.3 * obs[k1]                                              # random nodes: most of it is beyond degree 2
