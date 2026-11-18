"""Golden steps of the minimal model (tests/oracle/golden_step.py, MINIMAL_PRESET_2026-11-18.md section 6): reload each file, rebuild the small synthetic problem,
load the saved state, recompute the loss and the gradient of every parameter on the saved batch and compare to 1e-6 relative. float64, CPU, about 30 s in all."""
import os
import sys

import pytest
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "oracle"))

import golden_step as gs  # noqa: E402


@pytest.mark.parametrize("name", gs.NAMES)
def test_golden_step_reproduces(name):
    path = os.path.join(gs.RESULTS, f"golden_{name}.pt")
    assert os.path.exists(path), f"{path} missing: run tests/oracle/golden_step.py"
    rec = torch.load(path, weights_only=False)
    assert rec["dtype"] == "float64" and rec["jitter"] is False and torch.get_default_dtype() == torch.float32
    assert all(v.dtype in (torch.float64, torch.int64, torch.bool) for v in rec["state"].values()), "the state must be float64"
    loss, grads = gs.recompute(rec)
    assert torch.get_default_dtype() == torch.float32                          # float64_mode() restored the process
    worst, bad = gs.compare(rec, loss, grads, tol=1e-6)
    assert not bad, bad
    assert worst < 1e-6
    # the switched-off tensors get no gradient, the kept ones do
    for n in rec["switched_off_parameters"]:
        assert rec["grads"][n] is None, n
    assert rec["grads"]["hs"] is not None and rec["grads"]["net.inp.weight"] is not None and rec["grads"]["sky.pos"] is not None
    # the finite-difference record: wherever the gradient is not tiny the autograd value matches the numeric one
    for k, v in rec["fd_check"].items():
        if len(v) == 2 and abs(v[0]) > 1e-3:
            assert abs(v[0] - v[1]) / abs(v[0]) < 1e-6, (k, v)
