"""Bit-identity check of the port: the prototype (fieldop_min) against torchsky.map.exposure_fit._fit_exposures on the same synthetic field.

    PYTHONNOUSERSITE=1 PYTHONPATH=/home/claude/torchsky/src python compare_port.py --steps 30
    PYTHONNOUSERSITE=1 PYTHONPATH=/home/claude/torchsky/src python compare_port.py --steps 150 [--reference-state REF.pt]
    ... compare_port.py --steps 30 --kernel-centroid-gauge 1      (both sides with the kernel-centroid gauge on)

The field is the one of fieldop_min/tests/synth_small.py (item sky, seed 0, minimal protocol). It is built once with the prototype's generator;
each side gets its own copy. The prototype side repeats synth_small.py's sequence of calls (which is why, at 150 steps, its final state is also
compared with the recorded reference state of PRUNE_2026-11-19.md). Every tensor of the two final operator states is compared with the logic of
fieldop_min/tests/oracle/compare_states.py (equal dtype, shape and bytes). Nothing here lives in the torchsky tree.
"""
import argparse
import copy
import dataclasses
import os
import sys

import torch

PROTO = "/home/claude/work/fieldop_min"
sys.path.insert(0, PROTO)
sys.path.insert(0, os.path.join(PROTO, "tests", "oracle"))
torch.set_num_threads(1)

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--reference-state", default="")
ap.add_argument("--kernel-centroid-gauge", type=int, choices=[0, 1], default=0,
                help="1: run both sides with the kernel-centroid gauge on (prototype --kernel-centroid-gauge 1, torchsky internal kernel_centroid_gauge 1)")
a = ap.parse_args()

import fieldop  # noqa: E402  (prototype)
from fieldop.cli import apply_sky_module_options, default_config, make_sky  # noqa: E402
from fieldop.instruments import spectral_setup  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402
from fieldop.train import fit, heldout_stage, register_prepass, warm_kernel  # noqa: E402
import compare_states  # noqa: E402

import torchsky.map.exposure_fit as ef  # noqa: E402
from torchsky.map.exposure_fit import _bandpass as tb, _exposures as te  # noqa: E402
from torchsky.map.exposure_fit import _skyfit as tsf, _predict as tpr, _config as tcf  # noqa: E402

assert "fieldop_min" in fieldop.__file__ and "torchsky" in ef.__file__

# ---- the field, built once
torch.manual_seed(a.seed)
insts = [SynthInstrument("A", 0.10, 40, 6.5, 0.07, 1.5, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
         SynthInstrument("B", 0.12, 34, 6.5, 0.10, 2.4, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("B1", 2.0, 2.8)])]
exps, fset, meta, truth, _ = make_field(seed=3 + a.seed, n_per_filter=8, n_held=1, insts=insts, shift_rms=0.03, centre_range=0.6,
                                        rot_choices=(0, 1, 2, 3), patch_px=20, truth_half=1.6)
exps_proto, fset_proto = copy.deepcopy(exps), copy.deepcopy(fset)
exps_port = copy.deepcopy(exps)

kw = dict(sky_pitch=0.06, render_pitch=0.02, level_nh=12, kernel_extent=1.2, region_px=13, n_lambda=3, k_hidden=32, warm_kernel_steps=150,
          batch_patches=4, inner_sky=1, budget_min=1e6, steps=a.steps, val_every=2, held_steps=50, held_reg_steps=30, log_every=10, is_snr=10.0,
          reg_prepass_steps=40, reg_prepass_iters=1, seed=a.seed, item_count=1500, gauge_ref_inst=0, device="cpu")
if a.kernel_centroid_gauge:
    kw["kernel_centroid_gauge"] = 1

# ---- prototype side: the sequence of synth_small.py
cfg = default_config(**kw)
apply_sky_module_options(cfg)
spectral_setup(fset_proto, cfg, log=lambda s: None)
sky, N = make_sky(exps_proto, fset_proto, cfg, 2.8, "cpu")
model = FieldOperator(cfg, fset_proto, meta, sky, torch.device("cpu"))
finfo = {}
for f in range(len(fset_proto)):
    ii = [i for i, ins in enumerate(insts) if any(fn == fset_proto.names[f] for fn, _, _ in ins.filters)][0]
    finfo[f] = {"pixel_arcsec": insts[ii].pixel, "diameter_m": 6.5, "inst": ii}
lg = lambda s: None  # noqa: E731
warm_kernel(model, cfg, finfo, log=lg)
register_prepass(model, exps_proto, cfg, log=lg)
rec = fit(model, exps_proto, cfg, log=lg)
held = [e.eid for e in exps_proto if e.split == "held"]
stage = heldout_stage(model, exps_proto, held, cfg, log=lg)
proto_chi2 = {e.name: stage["pred"](e)[3] for e in exps_proto if e.split == "held"}
proto_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

# ---- torchsky side: fit_sky on converted copies
def conv_filter(f):
    g = tb.Filter(f.name, f.instrument, f.lam_um, f.T)
    for k in ("nodes", "weights", "pivot", "curve_vec"):
        setattr(g, k, getattr(f, k))
    return g

conv_cache = {}
port = []
for e in exps_port:
    d = {f.name: getattr(e, f.name) for f in dataclasses.fields(e)}
    if e.filt not in conv_cache:
        conv_cache[e.filt] = conv_filter(fset.filters[e.filt])
    ins = insts[e.inst].name
    port.append(te.Exposure(**d, bandpass=conv_cache[e.filt], inst_name=ins, det_name=ins, chan_name=ins))
config = ef.SkyFitConfig(sky_pitch=0.06, item_count=1500, kernel_extent=1.2, reference_instrument=0, steps=a.steps, budget_minutes=1e6, patches_per_step=4, seed=a.seed,
                         device="cpu")
internal = dict(render_pitch=0.02, level_nh=12, region_px=13, n_lambda=3, k_hidden=32, warm_kernel_steps=150, inner_sky=1, val_every=2,
                held_steps=50, held_reg_steps=30, log_every=10, is_snr=10.0, reg_prepass_steps=40, reg_prepass_iters=1)
if a.kernel_centroid_gauge:
    internal["kernel_centroid_gauge"] = 1
# every setting that is neither a public field nor overridden above must equal the prototype's parser default
ref_ns, new_ns = vars(default_config(**kw)), vars(tcf._namespace(config, internal))
bad = {k: (ref_ns.get(k, "<absent>"), new_ns[k]) for k in new_ns if k not in ref_ns or ref_ns[k] != new_ns[k]}
only_proto = sorted(k for k in ref_ns if k not in new_ns)
print("settings that differ between the prototype's namespace and torchsky's:", bad or "none", "| prototype-only attributes (deleted/CLI-only):", only_proto)
sf = tsf._fit_exposures(port, config, internal=internal, sky_half_extent=2.8)
port_state = {k: v.detach().cpu().clone() for k, v in sf.operator.state_dict().items()}
port_chi2 = {e.name: tpr._predict_record(sf, e).chi2 for e in port if e.split == "held"}


def report(title, ref, new, shared_only=False):
    rows, n_diff = compare_states.compare(ref, new)
    for k, s, d in rows:
        if s != "identical" and not (shared_only and s == "only in reference"):
            print(f"{s:18s} {k}  {d}")
    n_id = sum(r[1] == "identical" for r in rows)
    only_new = [k for k in new if k not in ref]
    good = n_diff == 0 and not only_new and (shared_only or len(ref) == len(new) == n_id)
    print(f"{title}: {n_id} identical, {n_diff} different, {len(ref)} reference and {len(new)} new tensors"
          + (" (the 49 reference tensors of deleted terms are not in the pruned state: not a difference)" if shared_only else "")
          + " -> " + ("PASS (every shared tensor bit-identical)" if good else "FAIL"))
    return good


ok = report(f"prototype vs torchsky, {a.steps} steps", proto_state, port_state)
if a.reference_state:
    ref = compare_states.load(a.reference_state)
    ok &= report(f"recorded reference ({os.path.basename(a.reference_state)}) vs prototype side of this script", ref, proto_state, shared_only=True)
same_chi2 = all(proto_chi2[k]["chi2"] == port_chi2[k]["chi2"] and proto_chi2[k]["n"] == port_chi2[k]["n"] and
                proto_chi2[k]["chi2_model_noise"] == port_chi2[k]["chi2_model_noise"] for k in proto_chi2)
print("held-out chi2 (prototype, torchsky):", {k: (proto_chi2[k]["chi2"], port_chi2[k]["chi2"]) for k in proto_chi2}, "equal:", same_chi2)
strip = lambda L: [[r[0]] + list(r[2:]) for r in L]          # column 1 is the elapsed time
print("loss log equal (step, loss, chi2 columns):", strip(rec["steps"]) == strip(sf.report["loss_log"]))
if a.kernel_centroid_gauge:
    kp, kt = rec.get("kernel_centroid_gauge"), sf.report.get("kernel_centroid_gauge")
    print("kernel-centroid gauge record (prototype):", kp)
    print("kernel-centroid gauge records equal:", kp == kt)
    ok &= kp is not None and kp == kt
print("RESULT:", "PASS" if ok and same_chi2 else "FAIL")
sys.exit(0 if ok and same_chi2 else 1)
