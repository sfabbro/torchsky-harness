"""Build C: WARNING when the blur dilation exceeds 2, and the maximum stored in the metrics (no change to the optimisation). CPU, about 20 s.
A fit whose dilation is forced to 3.0 (the model's dilation method replaced) logs the WARNING once at the first logged step and once at the end and
stores dilation_max = 3; a normal fit (dilation 1) logs no such warning and stores dilation_max = 1."""
import os
import sys

import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import synth_coarse_reg as sc  # noqa: E402
from fieldop import train  # noqa: E402
from fieldop.tiles import TileSet  # noqa: E402


def _fit(force=None):
    model, exps, cfg = sc.build(0, 0, steps=4)
    cfg.log_every = 2
    if force is None:
        cfg.blur_floor = 0.0          # (this synthetic sky is coarser than its response, so the floor would legitimately dilate by about 3)
    else:
        model.dilation = lambda: torch.tensor(force)
    logs = []
    rec = train.fit(model, exps, cfg, log=logs.append, ts=TileSet.single(model, exps, cfg, logs.append))
    return rec, [s for s in logs if "WARNING: blur dilation" in s]


def test_dilation_warning_and_maximum():
    rec, w = _fit(3.0)
    assert len(w) == 2 and "exceeds 2" in w[0] and "reached 3" in w[1], w
    assert rec["dilation_max"] == 3.0 and rec["dilation_final"] == 3.0 and "dilation_first_above" in rec
    rec, w = _fit(None)
    assert w == [] and rec["dilation_max"] == 1.0 and "dilation_first_above" not in rec


def test_dilation_stop_ends_the_fit_cleanly():
    model, exps, cfg = sc.build(0, 0, steps=6)
    cfg.log_every = 2
    cfg.dilation_stop = 2.5
    model.dilation = lambda: torch.tensor(3.0)
    logs = []
    rec = train.fit(model, exps, cfg, log=logs.append, ts=TileSet.single(model, exps, cfg, logs.append))
    assert rec["dilation_stopped_at"] == 0 and rec["n_steps"] == 0 and any(s.startswith("STOP: blur dilation") for s in logs), logs[-3:]
    assert rec["dilation_max"] == 3.0
    model, exps, cfg = sc.build(0, 0, steps=3)         # 10 is a second warning, once
    model.dilation = lambda: torch.tensor(12.0)
    logs = []
    rec = train.fit(model, exps, cfg, log=logs.append, ts=TileSet.single(model, exps, cfg, logs.append))
    assert sum("exceeds 10" in s for s in logs) == 1 and sum("exceeds 2" in s for s in logs) == 1 and "dilation_stopped_at" not in rec



if __name__ == "__main__":
    test_dilation_warning_and_maximum()
    test_dilation_stop_ends_the_fit_cleanly()
    print("ok")
