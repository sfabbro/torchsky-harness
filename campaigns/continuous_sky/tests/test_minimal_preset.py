"""Tests of the --minimal preset (MINIMAL_PRESET_2026-11-18.md). CPU, small synthetic field, about 2 minutes in all.

1. the preset parses, sets what the table in the report says, explicit options win, it does not combine with --legacy-defaults, the default parse is unchanged
2. no optimiser group holds a parameter of a switched-off term, the kept terms are there, and the switched-off tensors do not move in a short fit
3. no validation exposures are held back (the loader's carve-out is off), and the control (default options) still holds one back
4. the damped EM step of the random-effect scales runs without validation exposures; no evidence machinery runs
5. an explicit option brings a switched-off term back (the preset only changes defaults)
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fieldop import bandpass, cli, data, train  # noqa: E402
from fieldop import sky as sky_mod  # noqa: E402
from fieldop.cli import MINIMAL_DEFAULTS, default_config, make_sky, parse_config  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_item_module_state():
    old = (sky_mod.ITEM_HALF, sky_mod.PYR_HALF, sky_mod.ITEM_CHUNK)
    yield
    sky_mod.ITEM_HALF, sky_mod.PYR_HALF, sky_mod.ITEM_CHUNK = old


# ----------------------------------------------------------------------------- 1. the preset
def test_minimal_parses_and_sets_the_table():
    a = parse_config(["--minimal"])
    for k, v in MINIMAL_DEFAULTS.items():
        assert getattr(a, k) == v, k
    # the sky of the reference fits, as the task states it
    assert (a.sky_form, a.item_count, a.item_levels, a.item_raster, a.item_patch, a.item_buckets, a.item_max_width) == ("items", 60000, 3, "pyramid", "sheared", "coarse", 2.56)
    assert a.item_chunk == 0                                                  # left at its default
    assert (a.k_space, a.k_time, a.k_lambda, a.k_exp, a.g_fields, a.distortion, a.pattern_grid, a.nonlin) == ("off", "off", "off", "learned", "off", 0, 0, 0)
    assert (a.sky_zero, a.sky_positivity_margin, a.pixel_gain, a.filter_shift, a.input_level, a.render, a.interp) == ("median", 0.0, "off", "off", "cal", "fft", "bspline")
    assert (a.noise_scale_max, a.noise_floor_max, a.loss, a.huber_delta, a.kernel_norm_radius, a.blur_floor) == (2.0, 0.2, "huber", 3.0, 2.0, 1.0)
    assert (a.batch_patches, a.inner_sky, a.variability_evidence, a.held_xfit) == (5, 2, 0, 1)
    assert a.gauge_ref_inst is None and a.coarse_reg == "off"                  # several instruments still need --gauge-ref-inst; coarse registration stays opt-in


def test_explicit_options_win_and_flags_do_not_combine():
    a = parse_config(["--minimal", "--item-count", "123", "--k-space", "learned", "--batch-patches", "8", "--variability-evidence", "1"])
    assert (a.item_count, a.k_space, a.batch_patches, a.variability_evidence) == (123, "learned", 8, 1)
    assert a.sky_form == "items" and a.k_time == "off"                         # the rest of the preset is kept
    with pytest.raises(SystemExit):
        parse_config(["--minimal", "--legacy-defaults"])


def test_default_parse_is_unchanged_by_the_new_options():
    a = parse_config([])
    assert (a.minimal, a.g_fields, a.variability_evidence, a.n_val_groups) == (False, "default", 1, 1)
    assert (a.sky_form, a.k_space, a.k_time, a.k_lambda, a.distortion, a.pattern_grid, a.batch_patches) == ("grid", "learned", "learned", "learned", 1, 8, 8)


# ----------------------------------------------------------------------------- 2. optimiser groups
def build(steps=0, minimal=True, n_val=0, **over):
    torch.manual_seed(0)
    insts = [SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)])]
    exps, fset, meta, _, _ = make_field(seed=1, n_per_filter=5, n_val=n_val, n_held=1, insts=insts, centre_range=0.4, rot_choices=(0, 1, 2, 3), patch_px=12, truth_half=1.0)
    kw = dict(sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.8, region_px=12, n_lambda=3, k_hidden=16, item_count=200, batch_patches=3, steps=steps,
              budget_min=1e6, log_every=10, seed=0, held_steps=4, held_reg_steps=3, reg_prepass_steps=4, reg_prepass_iters=1, val_every=2, tau_accum=2, tau_start=0.2)
    if minimal:
        kw.update(minimal=True, legacy_defaults=False, kernel_norm_radius=2.0, gauge_ref_inst=None)       # conftest.py would otherwise set the legacy values
    kw.update(over)
    cfg = default_config(**kw)
    cli.apply_sky_module_options(cfg)
    sky, _ = make_sky(exps, fset, cfg, 1.6, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    logs = []
    rec = None
    if steps:
        rec = train.fit(model, exps, cfg, log=logs.append)
    return model, exps, cfg, logs, rec


def _ids(opts):
    return {id(p) for o in opts if o is not None for g in o.param_groups for p in g["params"]}


def _switched_off(model):
    """Every tensor that belongs to a switched-off term, by name."""
    pre = ("gdet.", "gep.", "g_det_emb.", "Dst.", "Ddr.", "D_det.", "codes.hdet.", "codes.det.", "codes.hep.", "nl_chan.")
    return {n: p for n, p in model.named_parameters() if n.startswith(pre) or n == "zD"}


def test_optimiser_groups_hold_no_switched_off_parameter():
    model, exps, cfg, _, _ = build()
    opts = train.make_optimizers(model, cfg, 1.0)
    held = _ids(opts)
    off = _switched_off(model)
    assert len(off) == 41, sorted(off)
    assert not [n for n, p in off.items() if id(p) in held], "a switched-off tensor is in an optimiser group"
    assert model.pattern is None and model.nonlin is None and model.pixgain is None and not model.fs_on        # absent, not zero
    assert {k: model.tau.mode[k] for k in ("k_space", "k_time", "k_lambda", "g_space", "g_time", "D_static", "D_drift", "D_exp") if k in model.tau.mode} \
        == {"k_space": "off", "k_time": "off", "k_lambda": "off", "g_space": "off", "g_time": "off"}              # D_*: no entry at all (--distortion 0)
    assert set(model.tau.raw.keys()) == {"k_exp", "g_exp"}                                                        # the kept random effects
    # the kept terms are in a group
    for n, p in model.named_parameters():
        if n in off or n.startswith("tau.") or n.startswith("sky."):
            continue
        assert id(p) in held, f"kept tensor {n} is in no optimiser group"
    # the operator group is exactly the kept operator tensors
    op, nuis, taus, skyp = model.param_groups()
    assert not {id(p) for p in op} & {id(p) for p in off.values()}
    assert len(taus) == 2 and len(skyp) > 0


def test_switched_off_tensors_do_not_move_and_get_no_gradient():
    model, exps, cfg, logs, rec = build(steps=6)
    ref, _, _, _, _ = build()                                                  # same start (seeded): compare the untouched tensors
    for n, p in _switched_off(model).items():
        assert torch.equal(p.detach(), _switched_off(ref)[n].detach()), n
    # one loss, one backward over every parameter: the switched-off ones are not reached
    pool = train.PatchPool(exps, [i for i, e in enumerate(exps) if e.split == "train"], cfg)
    items = pool.draw(3, np.random.default_rng(0))
    train.set_prior_pixels(cfg, exps, log=lambda s: None)
    P = model.prepare()
    loss, _, _ = train.batch_loss(model, exps, items, P, cfg, pool.mean_valid)
    loss = loss + model.priors(train.batch_eids(exps, items, model.dev)) / cfg.prior_pixels
    loss.backward()
    for n, p in _switched_off(model).items():
        assert p.grad is None, n
    assert model.hs.grad is not None and model.bkg.grad is not None and model.codes.z.grad is not None and model.net.inp.weight.grad is not None


# ----------------------------------------------------------------------------- 3. no validation exposures
def _fake_splits(tmp_path, exps):
    sp = {"boxes": [{"id": 0, "ra": 10.0, "dec": -20.0, "size_arcsec": 2.0}],
          "splits": {"dither": {"heldout": [], "train": [e["rootname"] for e in exps]}}, "exposures": exps}
    p = tmp_path / "splits_TEST.json"
    p.write_text(json.dumps(sp))
    return str(p)


def _fake_read_cuts(path, boxes, half_data, shift0, want_jac, rpath, inst=None):
    n = 48
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = dict(d=np.ones((n, n), np.float32), sig=np.ones((n, n), np.float32), valid=np.ones((n, n), bool), X=(xx - n / 2) * 0.05, Y=(yy - n / 2) * 0.05, cut0=(0, 0),
             det_shape=(n, n), J=np.array([[0.05, 0.0], [0.0, 0.05]]), detector="", instrument="", bkg0=1.0, pixar_a2=float("nan"))
    return [(0, r)], None


def _exp(root):
    return {"rootname": root, "filter": "F150W", "instrument": "NIRCAM", "dither_group": root, "mjd": 60000.0, "boxes": {"0": 1},
            "path": f"/nonexistent/{root}_cal.fits", "file": f"{root}_cal.fits", "detector": "NRCB1"}


def _load(tmp_path, monkeypatch, **over):
    monkeypatch.setattr(bandpass, "svo_filter", lambda name, instrument, svo_id, Q, cache_dir=None: bandpass.box_filter(name, instrument, 1.0, 1.3, Q))
    monkeypatch.setattr(data, "_read_cuts", _fake_read_cuts)
    exps = [_exp(f"j{i}") for i in range(6)]                                    # six dither groups of one filter: more than three, so a group can be carved out
    cfg = default_config(splits=_fake_splits(tmp_path, exps), n_lambda=3, **over)
    return data.load_field(cfg, "cpu", log=lambda s: None)[3]


def test_no_validation_exposures_are_held_back(tmp_path, monkeypatch):
    control = _load(tmp_path, monkeypatch)                                      # default options: one dither group per filter is set aside
    assert control["n_by_split"]["val"] == 1 and len(control["val_groups"]) == 1
    mini = _load(tmp_path, monkeypatch, minimal=True, legacy_defaults=False, kernel_norm_radius=2.0)
    assert mini["n_by_split"]["val"] == 0 and mini["val_groups"] == [] and mini["n_by_split"]["train"] == 6
    # the carve-out stays off even if --n-val-groups is given again: the evidence machinery is what needs it
    mini2 = _load(tmp_path, monkeypatch, minimal=True, legacy_defaults=False, kernel_norm_radius=2.0, n_val_groups=2)
    assert mini2["n_by_split"]["val"] == 0
    back = _load(tmp_path, monkeypatch, minimal=True, legacy_defaults=False, kernel_norm_radius=2.0, variability_evidence=1, n_val_groups=1)
    assert back["n_by_split"]["val"] == 1                                       # the preset only changes defaults


# ----------------------------------------------------------------------------- 4. EM of the random-effect scales without validation
def test_em_of_random_effect_scales_runs_without_validation():
    model, exps, cfg, logs, rec = build(steps=60)
    assert not any(e.split == "val" for e in exps)
    assert rec["val"] == [] and "ablation_z" not in rec                           # no validation evaluations, no likelihood-ratio checks
    t = model.tau
    assert t.mode["k_exp"] == "learned" and t.mode["g_exp"] == "learned"
    assert abs(t.value("k_exp") - cfg.tau0_code) > 1e-6 and abs(t.value("g_exp") - cfg.tau0_g) > 1e-6, "the EM step never ran"
    ref, _, rcfg, _, _ = build(steps=60, variability_evidence=1)                  # same, with the evidence rule on but no validation exposures: no EM at all
    assert abs(ref.tau.value("k_exp") - rcfg.tau0_code) < 1e-6 and abs(ref.tau.value("g_exp") - rcfg.tau0_g) < 1e-6


def test_tau_at_cap_report_is_empty_without_the_evidence_rule():
    model, exps, cfg, _, _ = build(steps=0)
    assert train.nuisance_sizes(model, exps)["tau_at_cap"] == {}
    model2, exps2, cfg2, _, _ = build(steps=0, variability_evidence=1)
    assert "k_exp" in train.nuisance_sizes(model2, exps2)["tau_at_cap"]


# ----------------------------------------------------------------------------- 5. explicit options bring the terms back
@pytest.mark.parametrize("opt,over,names", [
    ("k_space", dict(k_space="learned"), ("codes.hdet.0.weight", "codes.det.weight")),
    ("g_fields", dict(g_fields="default", amplitude="structured"), ()),            # g_time needs several epochs; g_space is what returns
    ("distortion", dict(distortion=1), ("Dst.0.weight", "D_det.weight", "zD")),
])
def test_explicit_option_returns_the_term(opt, over, names):
    model, exps, cfg, _, _ = build(**over)
    held = _ids(train.make_optimizers(model, cfg, 1.0))
    params = dict(model.named_parameters())
    for n in names:
        assert id(params[n]) in held, n
    if opt == "g_fields":
        assert model.tau.mode["g_space"] == "learned" and id(params["gdet.0.weight"]) in held


# ----------------------------------------------------------------------------- 6. the command line end to end (loader replaced by the synthetic field)
def test_main_runs_under_minimal(tmp_path, monkeypatch):
    insts = [SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)])]

    def fake_load(cfg, dev, log=print):
        exps, fset, meta, truth, _ = make_field(seed=1, n_per_filter=5, n_val=0 if cfg.variability_evidence == 0 else 1, n_held=1, insts=insts, centre_range=0.4,
                                                rot_choices=(0, 1, 2, 3), patch_px=12, truth_half=1.0)
        for i, e in enumerate(exps):
            e.cid = i
        info = dict(box={}, boxes=[{}], tiles=[0], tile_c0={0: [0, 0]}, half_data=1.9, dets=list(range(meta["n_det"])), chans=[f"c{i}" for i in range(meta["n_chan"])],
                    insts=["A"], filters=list(fset.names), n_by_split={s: sum(1 for e in exps if e.split == s) for s in ("train", "val", "held")},
                    epoch_span_days=0.0, shifts_applied=0, n_exposures=len(exps), n_cuts=len(exps), cuts_per_tile={0: {}}, host_gb=0.0, multi_tile=False, val_groups=[])
        return exps, fset, meta, info
    monkeypatch.setattr(data, "load_field", fake_load)
    args = ["--minimal", "--item-count", "150", "--sky-pitch", "0.08", "--render-pitch", "0.04", "--level-nh", "8", "--kernel-extent", "0.8", "--region-px", "12",
            "--n-lambda", "3", "--k-hidden", "16", "--batch-patches", "3", "--log-every", "10", "--held-steps", "6", "--held-reg-steps", "4",
            "--protocol-root", str(tmp_path / "proto"), "--warm-kernel-steps", "5", "--reg-prepass-steps", "3", "--reg-prepass-iters", "1", "--budget-min", "100000",
            "--device", "cpu", "--out", str(tmp_path / "r"), "--method-name", "mini", "--steps", "40", "--tau-start", "0.2", "--val-every", "2", "--tau-accum", "2"]
    m = cli.main(args)
    cfgd = m["config"]
    assert cfgd["minimal"] is True and cfgd["sky_form"] == "items" and cfgd["item_raster"] == "pyramid" and cfgd["kernel_norm_radius"] == 2.0 and cfgd["k_space"] == "off"
    assert m["info"]["n_by_split"]["val"] == 0 and m["fit"]["n_steps"] == 40 and m["fit"]["val"] == []
    assert m["tau"]["g_space"]["mode"] == "off" and m["tau"]["k_exp"]["value"] != m["tau"]["k_exp"]["init"]            # EM ran
    assert m["nuisance_sizes"]["tau_at_cap"] == {}
    assert sky_mod.ITEM_HALF == (2, 4, 8, 16, 32, 64, 128)                                                              # --item-buckets coarse was applied
    log = (tmp_path / "r" / "fieldop.log").read_text()
    assert "minimal preset True" in log and "tau learned: False" in log
    st = torch.load(tmp_path / "r" / "fieldop_state.pt", weights_only=False)
    assert st["launch_config"]["minimal"] is True and st["launch_config"]["g_fields"] == "off"
