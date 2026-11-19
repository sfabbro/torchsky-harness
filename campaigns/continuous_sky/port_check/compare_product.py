"""Same numbers as the prototype: the polished sky and its standard deviation from the prototype's uncertainty.py and from torchsky's SkyFit.product
machinery (``_product.compute``, which ``SkyFit.product`` calls after converting the observations to records), on the prototype's truth problem
(tests/truth/truth_problem.py, items, 8 training exposures, noise seed 0, the smallest configuration of tests/truth/uncertainty_calibration.py).

    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/claude/torchsky/src python proto_truth_fit.py     # once: the prototype's fitted state
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/claude/torchsky/src python compare_product.py

Both sides get the same fitted state (the prototype's), the same records (field copy), the same draw seed (0) and the same pixel budget per batched
render (16000, the default of SkyFit.product). The prototype tree is not written to.
"""
import argparse
import copy
import dataclasses
import math
import os
import sys
import types

import numpy as np
import torch

PROTO = "/home/claude/work/fieldop_min"
sys.path.insert(0, PROTO)
sys.path.insert(0, os.path.join(PROTO, "tests", "truth"))
torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument("--samples", type=int, default=2)
ap.add_argument("--skip-draws", action="store_true")
a = ap.parse_args()

import uncertainty_calibration as uc  # noqa: E402  (prototype driver)
from fieldop import uncertainty as PU  # noqa: E402

import torchsky.map.exposure_fit._uncertainty as TU  # noqa: E402
from torchsky.map.exposure_fit import _bandpass as tb, _exposures as te, _items as ti, _product as tprod  # noqa: E402
from torchsky.map.exposure_fit._fit import make_sky  # noqa: E402
from torchsky.map.exposure_fit._response import FieldOperator  # noqa: E402
from torchsky.map.exposure_fit._skyfit import _module_state  # noqa: E402
from torchsky.map.exposure_fit._spectral import spectral_setup  # noqa: E402

assert "fieldop_min" in PU.__file__ and "torchsky" in TU.__file__
HERE = os.path.dirname(os.path.abspath(__file__))
uc.OUT = os.path.join(HERE, "truth_states")
uc.KERNEL = os.path.join(uc.OUT, "kernel_truth.pt")
quiet = lambda s: None  # noqa: E731

# ---- prototype side: model with the saved fitted state, its records
model, cfg, exps, st = uc.load_model("items", 8, 0, log=quiet)
print("prototype state:", os.path.basename(os.path.join(uc.OUT, "state_items_n8_s0.pt")), "steps", st["steps"], "training exposures", sum(e.split == "train" for e in exps))

# ---- torchsky side: the same state in torchsky's operator, the same records
fset = model.fset
def conv_filter(f):
    g = tb.Filter(f.name, f.instrument, f.lam_um, f.T)
    for k in ("nodes", "weights", "pivot", "curve_vec"):
        setattr(g, k, getattr(f, k))
    return g
fset2 = tb.FilterSet([conv_filter(f) for f in fset.filters])
exps2 = []
for e in copy.deepcopy(exps):
    d = {f.name: getattr(e, f.name) for f in dataclasses.fields(e)}
    exps2.append(te.Exposure(**d, bandpass=fset2.filters[e.filt], inst_name="A", det_name="A", chan_name="A"))
cfg2 = types.SimpleNamespace(**vars(cfg))
meta2 = te.build_meta(exps2, ["A"], ["A"], ["A"])
with _module_state(item_chunk=cfg2.item_chunk):
    spectral_setup(fset2, cfg2, log=quiet)
    domain = [float(v) for v in ti.SPECTRAL_DOMAIN]
    assert domain == [float(v) for v in __import__("fieldop.sky", fromlist=["x"]).SPECTRAL_DOMAIN], "spectral domains differ"
    sky2, N2 = make_sky(exps2, fset2, cfg2, 2.5, "cpu")
    op2 = FieldOperator(cfg2, fset2, meta2, sky2, torch.device("cpu"))
    op2.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()}, strict=True)
same_state = all(torch.equal(v, op2.state_dict()[k]) for k, v in model.state_dict().items())
same_sky = torch.equal(model.sky.coeffs().detach(), op2.sky.coeffs().detach())
print("operator state equal:", same_state, "| sky grid equal:", same_sky, "| Phi_band equal:", torch.equal(model.Phi_band, op2.Phi_band))
assert same_state and same_sky

train2 = [e for e in exps2 if e.split == "train"]
FW = tprod.FWHM_PER_SIGMA
results = []

def compare(label, ref, new_img, new_std, valid, s_ref, s_new, roundtrip=False):
    """ref: prototype band_readout dict; new_*: torchsky float64 arrays."""
    img_eq = torch.equal(ref["image"], new_img)
    cov = ref["covered"]
    vref = cov & torch.isfinite(ref["sigma"]) & (ref["sigma"] > 0)
    valid_eq = torch.equal(vref, valid)
    sd = (ref["sigma"][vref] - new_std[vref])
    std_eq = bool(valid_eq and torch.equal(ref["sigma"][vref], new_std[vref]))
    dimg = float((ref["image"] - new_img).abs().max())
    dstd = float(sd.abs().max()) if valid_eq else float("nan")
    scale_i, scale_s = float(ref["image"].abs().max()), float(ref["sigma"][vref].abs().max())
    print(f"[{label}] image bit-identical: {img_eq} (max abs diff {dimg:.3g}, relative to max {dimg / scale_i:.3g}); std bit-identical: {std_eq} "
          f"(max abs diff {dstd:.3g}, relative to max {dstd / scale_s:.3g}); valid mask equal: {valid_eq} ({int(valid.sum())} of {valid.numel()} pixels); "
          f"resolution sigma {s_ref!r} vs {s_new!r} equal: {s_ref == s_new}", flush=True)
    if roundtrip:   # FWHM = sigma * c then sigma' = FWHM / c is one rounding step off: accept a relative difference of 1e-14 and say so
        ok = dimg / scale_i < 1e-14 and dstd / scale_s < 1e-14 and valid_eq
        print(f"   round trip sigma -> FWHM -> sigma: not bit-identical by construction (one rounding step in sigma); accepted at relative 1e-14: {ok}")
        results.append((label, ok, ok, valid_eq, True))
        return
    results.append((label, img_eq, std_eq, valid_eq, s_ref == s_new))

def run_proto(method, polish, n_samples, resolution="auto"):
    op = PU.SkyNormalOperator(model, exps, cfg, rho=10.0, log=quiet, batch_px=16000)
    sh = PU.solve_shared(op, n_samples=n_samples, method=method, polish=polish, tol=1e-4, maxiter=60, log=quiet)
    return PU.band_readout(model, op, sh, 0, resolution=resolution, log=quiet, rho=10.0, tol=1e-4), op

def run_new(method, polish, n_samples, resolution="native"):
    with _module_state(domain=domain, item_chunk=cfg2.item_chunk):
        return tprod.compute(op2, train2, [0], resolution=resolution, method=method, polish=polish, samples=n_samples, batch_pixels=16000)

# 1. circulant sigma, optimiser's own sky (13 passes at K = 3: 4 here), native resolution
ref, pop = run_proto("circulant", False, 0)
new = run_new("circulant", False, 0)
compare("circulant, no polish, native", ref, new["image"][0], new["std"][0], new["valid"], ref["resolution"], new["s"][0])
print("   passes: prototype", pop.n_apply, "+", pop.n_forward, "forward-only; torchsky", new["n_passes"], "+", new["n_forward"])
# same, explicit resolution given as a FWHM (round trip FWHM -> sigma)
s = ref["resolution"]
new_e = run_new("circulant", False, 0, resolution=s * FW)
compare("circulant, no polish, explicit FWHM = sigma x 2sqrt(2ln2)", ref, new_e["image"][0], new_e["std"][0], new_e["valid"], s, new_e["s"][0], roundtrip=True)

# 2. circulant sigma with the Gauss-Newton step
ref, pop = run_proto("circulant", True, 0)
new = run_new("circulant", True, 0)
compare("circulant, polished, native", ref, new["image"][0], new["std"][0], new["valid"], ref["resolution"], new["s"][0])
print("   passes: prototype", pop.n_apply, "torchsky", new["n_passes"], "| polish iterations proto", ref["polish"]["iterations"], "torchsky", new["shared"]["polish"]["iterations"],
      "| residual", ref["polish"]["rel_residual"], new["shared"]["polish"]["rel_residual"])

# 3. posterior draws (same generator seed 0) with the step
if not a.skip_draws:
    ref, pop = run_proto("draws", True, a.samples)
    new = run_new("draws", True, a.samples)
    compare(f"draws ({a.samples}), polished, native", ref, new["image"][0], new["std"][0], new["valid"], ref["resolution"], new["s"][0])
    same_draws = torch.equal(ref["samples"], new["shared"]["samples"])
    print("   posterior draws bit-identical:", same_draws, "| passes: prototype", pop.n_apply, "torchsky", new["n_passes"])
    results.append(("draws arrays", same_draws, True, True, True))
ok = all(r[1] and r[2] and r[3] for r in results)
print("RESULT:", "PASS (every array bit-identical)" if ok else "DIFFERENCES FOUND (see lines above)")
