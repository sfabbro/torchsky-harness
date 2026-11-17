"""Tests of the noise-floor guard: input-noise chi-squared log, --noise-floor-max, WARNING lines. CPU, tiny synthetic fit, well under 60 s.
The recorded 150-step default state hash (tests/tsky_results/synth_2_tsky_defaults.json) takes about 140 s and is checked by running
    python tests/synth_small.py --pkg . --sky-form items --seed 0 --steps 150 --out r.json
(see the final report); here the extra log quantities are checked by switching them off and comparing the fitted state of a tiny fit."""
import hashlib
import os
import sys

import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fieldop import train  # noqa: E402
from fieldop.cli import default_config, make_sky  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402


def tiny_fit(steps=6, floor0=0.01, **kw):
    torch.manual_seed(0)
    insts = [SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", 0.12, 20, 6.5, 0.10, 2.4, filters=[("B1", 2.0, 2.8)])]
    exps, fset, meta, truth, _ = make_field(seed=1, n_per_filter=4, n_val=1, n_held=1, insts=insts, centre_range=0.4, rot_choices=(0, 1, 2, 3),
                                            patch_px=12, truth_half=1.0)
    cfg = default_config(sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.8, region_px=12, n_lambda=3, k_hidden=16, pattern_grid=0,
                         stripes=0, batch_patches=3, steps=steps, budget_min=1e6, tau_mode="fixed", log_every=2, floor0=floor0, seed=0, **kw)
    sky, _ = make_sky(exps, fset, cfg, 1.6, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    logs = []
    rec = train.fit(model, exps, cfg, log=logs.append)
    h = hashlib.sha256()
    st = model.state_dict()
    for k in sorted(st):
        h.update(k.encode())
        h.update(st[k].detach().cpu().contiguous().numpy().tobytes())
    return model, rec, logs, h.hexdigest()


def test_extra_log_quantities_do_not_change_state(monkeypatch):
    _, _, logs_on, h_on = tiny_fit()
    step_lines = [s for s in logs_on if s.startswith("step ")]
    assert step_lines and all("chi2/px(input noise)" in s and "noise-floor [" in s and "chi2/px(batch)" in s for s in step_lines)
    monkeypatch.setattr(train, "acc_input_chi", lambda *a, **k: None)       # the only extra computation in the loss path
    monkeypatch.setattr(train, "noise_floor_values", lambda m: [0.0])
    _, _, _, h_off = tiny_fit()
    assert h_on == h_off


def test_recorded_default_hash_is_recorded():
    # the recorded default hash exists; its full reproduction (150 steps, ~140 s) is run by tests/synth_small.py, see header
    import json
    d = json.load(open(os.path.join(HERE, "tsky_results", "synth_2_tsky_defaults.json")))
    assert len(d["state_sha256"]) == 64
    if os.environ.get("FIELDOP_FULL_HASH") == "1":       # opt-in, slow
        import subprocess
        out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "nf_guard_hash.json")
        subprocess.run([sys.executable, os.path.join(HERE, "synth_small.py"), "--pkg", os.path.dirname(HERE), "--sky-form", "items", "--seed", "0",
                        "--steps", "150", "--out", out], check=True)
        assert json.load(open(out))["state_sha256"] == d["state_sha256"]


def test_bound_holds():
    model, rec, logs, _ = tiny_fit(steps=8, floor0=0.3, noise_floor_max=0.05)
    assert max(train.noise_floor_values(model)) <= 0.05 + 1e-6
    assert max(rec["noise_floor_final"]) <= 0.05 + 1e-6
    assert not any(s.startswith("WARNING") for s in logs)


def test_default_is_unbounded():
    model, _, _, _ = tiny_fit(steps=2, floor0=0.3)
    assert max(train.noise_floor_values(model)) > 0.2


def test_warning_when_floor_above_limit():
    model, rec, logs, _ = tiny_fit(steps=4, floor0=0.5)
    w = [s for s in logs if s.startswith("WARNING")]
    assert len(w) == 2, w                                  # first exceedance, and final
    assert "first at step 0" in w[0] and "final" in w[1]
    assert rec["noise_floor_first_above"] == 0
