"""Unseen-band test on the synthetic field (19 Nov 2026): the middle filter has no training or validation exposure; its held-out exposures are
predicted from the other filters. Three numbers per field (held-out chi-squared per pixel against the input noise, middle filter only):
  trained   : the same fit with the middle filter trained (control);
  unseen    : the middle filter excluded from training (make_field train_exclude, the synthetic version of --train-exclude-filters);
  zero sky  : for the unseen fit, the prediction replaced by background only (zero sky; the best constant per exposure, a generous baseline).
    PYTHONNOUSERSITE=1 python tests/synth_unseen_band.py --pkg . --field one|two --mode trained|unseen --steps 60 --out result.json
UNCONVERGED by design (short fits): read the ordering of the three numbers, not their size.
"""
import argparse
import json
import os
import sys
import time

import torch

torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=".")
    ap.add_argument("--field", choices=["one", "two"], default="one")
    ap.add_argument("--mode", choices=["trained", "unseen"], default="unseen")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.pkg))
    from fieldop.cli import default_config, make_sky
    from fieldop.model import FieldOperator
    from fieldop.synth import SynthInstrument, make_field
    from fieldop.train import fit, heldout_stage, register_prepass, warm_kernel, chi2_exposure
    from fieldop.instruments import spectral_setup
    T0 = time.perf_counter()
    torch.manual_seed(a.seed)
    fa = [("A1", 1.0, 1.3), ("A2", 1.6, 2.1), ("A3", 2.4, 3.0)]
    insts = [SynthInstrument("A", 0.10, 40, 6.5, 0.07, 1.5, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=fa)]
    if a.field == "two":
        insts.append(SynthInstrument("B", 0.12, 34, 6.5, 0.10, 2.4, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("B1", 1.75, 2.35)]))
    mid = "A2"
    exps, fset, meta, truth, _ = make_field(seed=3 + a.seed, n_per_filter=6, n_val=1, n_held=2, insts=insts, shift_rms=0.03, centre_range=0.6,
                                            rot_choices=(0, 1, 2, 3), patch_px=20, truth_half=1.6, train_exclude=(mid,) if a.mode == "unseen" else ())
    kw = dict(sky_pitch=0.06, render_pitch=0.02, level_nh=12, kernel_extent=1.2, region_px=13, n_lambda=3, k_hidden=32, warm_kernel_steps=150,
              pattern_grid=0, stripes=0, batch_patches=4, inner_sky=1, budget_min=1e6, steps=a.steps, tau_mode="fixed", val_every=2, held_steps=50,
              held_reg_steps=30, log_every=10, is_snr=10.0, reg_prepass_steps=40, reg_prepass_iters=1, seed=a.seed, sky_form="grid")
    if a.field == "two":
        kw.update(gauge_ref_inst=0, kernel_norm_radius=2.0)
    cfg = default_config(**kw)
    spectral_setup(fset, cfg, log=lambda s: None)
    sky, N = make_sky(exps, fset, cfg, 2.8, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    finfo = {}
    for f in range(len(fset)):
        ii = [i for i, ins in enumerate(insts) if any(fn == fset.names[f] for fn, _, _ in ins.filters)][0]
        finfo[f] = {"pixel_arcsec": insts[ii].pixel, "diameter_m": 6.5, "inst": ii}
    logs = []
    lg = logs.append
    warm_kernel(model, cfg, finfo, log=lg)
    register_prepass(model, exps, cfg, log=lg)
    rec = fit(model, exps, cfg, log=lg)
    held = [e.eid for e in exps if e.split == "held"]
    stage = heldout_stage(model, exps, held, cfg, log=lg, tile=20)
    midi = fset.index[mid]
    res = {"field": a.field, "mode": a.mode, "steps": rec["n_steps"], "unseen_filters_in_model": [fset.names[f] for f in model.unseen],
           "loss_first_last": [rec["steps"][0][2], rec["steps"][-1][2]], "per_filter": {}, "noise_scale_unseen": None}
    acc = {}
    for e in exps:
        if e.split != "held":
            continue
        m, pv, S, sc = stage["pred"](e)
        v = e.valid & pv & torch.isfinite(m)
        w = 1.0 / e.sig[v] ** 2
        c = float((e.d[v] * w).sum() / w.sum())
        zs = chi2_exposure(e, torch.full_like(e.d, c), pv, S=S, snr_min=5.0)
        d = acc.setdefault(fset.names[e.filt], {"n": 0, "n_bright": 0, "chi2": 0.0, "chi2_bright": 0.0, "zero_chi2": 0.0, "zero_chi2_bright": 0.0})
        d["n"] += sc["n"]; d["n_bright"] += sc["n_bright"]
        d["chi2"] += sc["chi2"] * sc["n"]; d["zero_chi2"] += zs["chi2"] * zs["n"]
        if sc["n_bright"]:
            d["chi2_bright"] += sc["chi2_bright"] * sc["n_bright"]; d["zero_chi2_bright"] += zs["chi2_bright"] * zs["n_bright"]
    for f, d in acc.items():
        res["per_filter"][f] = {"chi2": d["chi2"] / d["n"], "zero_sky_chi2": d["zero_chi2"] / d["n"], "n_px": d["n"],
                                "chi2_bright": d["chi2_bright"] / max(d["n_bright"], 1), "zero_sky_chi2_bright": d["zero_chi2_bright"] / max(d["n_bright"], 1)}
    res["middle_filter"] = mid
    res["held_chi2_middle"] = res["per_filter"][mid]["chi2"]
    res["zero_sky_chi2_middle"] = res["per_filter"][mid]["zero_sky_chi2"]
    h = 3 * torch.tanh(model.hs / 3)
    res["noise_scale_exp_h_middle_filter"] = [float(torch.exp(h[d, midi])) for d in range(h.shape[0])]
    res["warnings"] = [x for x in logs if "WARN" in x or "STOP" in x][:6]
    res["total_seconds"] = time.perf_counter() - T0
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("field", "mode", "steps", "unseen_filters_in_model", "loss_first_last", "per_filter", "noise_scale_exp_h_middle_filter", "warnings", "total_seconds")}))


if __name__ == "__main__":
    main()
