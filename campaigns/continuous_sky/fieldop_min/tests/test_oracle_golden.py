"""Golden steps (tests/oracle/golden_step.py; the files are those of the reference package before the pruning): reload each file, rebuild the small synthetic problem,
load the saved state (the tensors of deleted terms are skipped; a surviving tensor missing from the file fails), recompute the loss and the gradient of every
surviving parameter on the saved batch and compare to 1e-12 relative. float64, CPU, about 30 s in all."""
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
    if not os.path.exists(path):   # in the repository the files are kept once, with the reference package next to this folder
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "oracle", "results", f"golden_{name}.pt")
        path = alt if os.path.exists(alt) else path
    assert os.path.exists(path), f"{path} missing: run tests/oracle/golden_step.py"
    rec = torch.load(path, weights_only=False)
    assert rec["dtype"] == "float64" and rec["jitter"] is False and torch.get_default_dtype() == torch.float32
    assert all(v.dtype in (torch.float64, torch.int64, torch.bool) for v in rec["state"].values()), "the state must be float64"
    loss, grads = gs.recompute(rec)
    assert torch.get_default_dtype() == torch.float32                          # float64_mode() restored the process
    det = {}
    worst, bad = gs.compare(rec, loss, grads, tol=1e-12, detail=det)
    assert not bad, bad
    assert worst < 1e-12
    assert det["grads_compared"] > 0
    # every parameter of the model is in the file; the parameters only in the file belong to deleted terms and had no gradient in the reference (checked below)
    assert not any(k in grads for k in det["parameters_only_in_file"])
    # a parameter that had no gradient in the file has none now
    for n in rec["switched_off_parameters"]:
        if n in grads:
            assert grads[n] is None, n
    assert rec["grads"]["hs"] is not None and rec["grads"]["net.inp.weight"] is not None and rec["grads"]["sky.pos"] is not None
    # the finite-difference record: wherever the gradient is not tiny the autograd value matches the numeric one
    for k, v in rec["fd_check"].items():
        if len(v) == 2 and abs(v[0]) > 1e-3:
            assert abs(v[0] - v[1]) / abs(v[0]) < 1e-6, (k, v)
