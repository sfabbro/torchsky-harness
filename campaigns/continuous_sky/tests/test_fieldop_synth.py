"""Synthetic recovery test of the field operator against an INDEPENDENT analytic renderer (fieldop/synth.py: erf pixel integrals of
Gaussian mixtures, dense lambda rule; no FFT, no neural field).

Field: two instruments (pixels 0.10" and 0.12", 4" detectors), three filters (A1 1.0-1.3, A2 1.6-2.1, B1 2.0-2.8 um: B1 overlaps A2, so the
sky spectrum couples the instruments), 16 exposures per filter (12 train, 2 validation, 2 held out), dithers +-0.6", rotations by multiples of
90 deg, two epochs. Sky pitch 0.06".
Injected: PSF width x (1 + 0.15 xi_x) (space), PSF width proportional to lambda (chromatic), throughput 1 + 0.08 xi_y (space), per-exposure
offsets of ~40 mas rms. NOT injected: epoch dependence, per-exposure kernel / gain / distortion, static distortion.

The common blur is a true degeneracy (sky x g, kernels x g^-1), so the checks use GAUGE-INVARIANT quantities:
  * held-out prediction chi2/N with the true sigma, at checkpoints along the fit (must not get worse with fit length);
  * relative kernels as Fourier ratios K_a(f) / K_b(f) (the common blur cancels) between detector positions, lambda nodes, epochs and
    exposures, against the same ratios of the true kernels;
  * relative throughput across the detector and between epochs; relative offsets (common mean removed).
Absolute kernel widths are reported too, labelled gauge-dependent.
Run:  FIELDOP_SYNTH_MIN=30 python -m pytest -q -s tests/test_fieldop_synth.py      (numbers in tests/fieldop_results/synth_recovery.json)
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fieldop.cli import default_config, make_sky  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402
from fieldop.train import chi2_exposure, fit, fit_heldout, predict_exposure, register_prepass, warm_kernel  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "fieldop_results")
os.makedirs(OUT, exist_ok=True)
BUDGET_MIN = float(os.environ.get("FIELDOP_SYNTH_MIN", "30"))
FREQS = [(f, 0.0) for f in (1.5, 3.0, 4.5)] + [(0.0, f) for f in (1.5, 3.0, 4.5)]   # cycles / arcsec


def instruments():
    return [SynthInstrument("A", 0.10, 40, 6.5, 0.07, 1.5, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
            SynthInstrument("B", 0.12, 34, 6.5, 0.10, 2.4, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("B1", 2.0, 2.8)])]


def build(budget_min):
    torch.manual_seed(0)
    insts = instruments()
    exps, fset, meta, truth, _ = make_field(seed=3, n_per_filter=16, n_val=2, n_held=2, insts=insts, shift_rms=0.03, centre_range=0.6,
                                            rot_choices=(0, 1, 2, 3), patch_px=20, truth_half=1.6)
    cfg = default_config(sky_pitch=0.06, render_pitch=0.02, level_nh=16, kernel_extent=1.5, region_px=13, n_lambda=3, k_hidden=32,
                         warm_kernel_steps=300, pattern_grid=0, stripes=0, batch_patches=6, inner_sky=1, budget_min=budget_min,
                         val_every=2, held_steps=100, log_every=50, is_snr=10.0, reg_prepass_steps=100)
    sky, N = make_sky(exps, fset, cfg, 2.8, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    finfo = {}
    for f in range(len(fset)):
        ii = [i for i, ins in enumerate(insts) if any(fn == fset.names[f] for fn, _, _ in ins.filters)][0]
        finfo[f] = {"pixel_arcsec": insts[ii].pixel, "diameter_m": 6.5, "inst": ii}
    return exps, fset, insts, cfg, model, finfo


def heldout_chi2(model, exps, cfg):
    fit_heldout(model, exps, [e.eid for e in exps if e.split == "held"], cfg, log=lambda s: None)
    P = {k: v.detach() for k, v in model.prepare().items()}
    sc = {}
    for e in exps:
        if e.split == "held":
            m, pv, S = predict_exposure(model, e, P, cfg, tile=20)
            sc[e.name] = chi2_exposure(e, m, pv, S=S, snr_min=5.0)
    n = sum(v["n"] for v in sc.values())
    nb = sum(v["n_bright"] for v in sc.values())
    return {"chi2": sum(v["chi2"] * v["n"] for v in sc.values()) / n,
            "chi2_snr5": sum(v["chi2_bright"] * v["n_bright"] for v in sc.values() if v["n_bright"]) / max(nb, 1), "per_exposure": sc}


def offset_residual(model, exps, P, split="train"):
    lt, tt = [], []
    for e in exps:
        if e.split != split:
            continue
        with torch.no_grad():
            pos = model.positions(torch.tensor(e.eid), torch.zeros(1), torch.zeros(1), torch.zeros(1, 2), P)[:, 0].mean(0)
        lt.append(pos.numpy())
        tt.append(np.array(e.extra["true_shift"]))
    lt, tt = np.array(lt), np.array(tt)
    d = (lt - lt.mean(0)) - (tt - tt.mean(0))
    return float(np.sqrt((d ** 2).sum(1).mean()) * 1e3), float(np.sqrt(((tt - tt.mean(0)) ** 2).sum(1).mean()) * 1e3)


def learned_khat(model, e, xi, P, q):
    """|K(f)| at FREQS of the learned (normalised) kernel of exposure e at normalised position xi (2,), lambda node q, sky frame."""
    with torch.no_grad():
        Ks, _ = model.node_kernels(torch.tensor(e.eid), xi[None], P)
        lv = model.levels_of(e.eid)
        out = []
        for fx, fy in FREQS:
            v = 0.0
            for l, K in enumerate(Ks):
                ph = 2 * math.pi * (fx * lv.u[l][..., 0] + fy * lv.u[l][..., 1])
                v = v + torch.complex((K[0, q] * torch.cos(ph)).sum(), -(K[0, q] * torch.sin(ph)).sum())
            out.append(float(v.abs()))
    return np.array(out)


def true_khat(inst, lam, xi):
    sc = inst.sigma0 * (lam / inst.lam0) ** inst.chromatic * (1 + inst.space_grad * float(xi[0]))
    sw = inst.wing_ratio * sc
    out = []
    for fx, fy in FREQS:
        f2 = fx * fx + fy * fy
        g = (1 - inst.f_wing) * math.exp(-2 * math.pi ** 2 * sc ** 2 * f2) + inst.f_wing * math.exp(-2 * math.pi ** 2 * sw ** 2 * f2)
        out.append(abs(g * np.sinc(inst.pixel * fx) * np.sinc(inst.pixel * fy)))
    return np.array(out)


def rel_kernel_checks(model, exps, fset, insts, P):
    """Gauge-invariant relative kernels: D = log|K_a(f)| - log|K_b(f)| for pairs differing along ONE axis; learned vs truth (rms over f)."""
    rows = {}
    for fname in fset.names:
        f = fset.index[fname]
        inst = [ins for ins in insts if any(fn == fname for fn, _, _ in ins.filters)][0]
        tr = [e for e in exps if e.split == "train" and e.filt == f]
        e0 = tr[0]
        lamq = fset.filters[f].nodes
        Q = len(lamq)
        pairs = {
            "space_x (injected)": ((e0, (-0.8, 0.0), Q // 2), (e0, (0.8, 0.0), Q // 2)),
            "space_y (not injected)": ((e0, (0.0, -0.8), Q // 2), (e0, (0.0, 0.8), Q // 2)),
            "lambda (injected)": ((e0, (0.0, 0.0), 0), (e0, (0.0, 0.0), Q - 1)),
        }
        ea = [e for e in tr if e.t == tr[0].t]
        eb = [e for e in tr if e.t != tr[0].t]
        if eb:
            pairs["epoch (not injected)"] = ((ea[0], (0.0, 0.0), Q // 2), (eb[0], (0.0, 0.0), Q // 2))
        if len(ea) > 1:
            pairs["exposure (not injected)"] = ((ea[0], (0.0, 0.0), Q // 2), (ea[1], (0.0, 0.0), Q // 2))
        for name, ((ea_, xa, qa), (eb_, xb, qb)) in pairs.items():
            Dl = np.log(learned_khat(model, ea_, torch.tensor(xa), P, qa)) - np.log(learned_khat(model, eb_, torch.tensor(xb), P, qb))
            Dt = np.log(true_khat(inst, lamq[qa], xa)) - np.log(true_khat(inst, lamq[qb], xb))
            rows.setdefault(name, []).append({"filter": fname, "true_rms": float(np.sqrt((Dt ** 2).mean())), "learned_rms": float(np.sqrt((Dl ** 2).mean())),
                                              "error_rms": float(np.sqrt(((Dl - Dt) ** 2).mean()))})
    return {k: {"true_rms": float(np.mean([r["true_rms"] for r in v])), "learned_rms": float(np.mean([r["learned_rms"] for r in v])),
                "error_rms": float(np.mean([r["error_rms"] for r in v])), "per_filter": v} for k, v in rows.items()}


def rel_throughput_checks(model, exps, fset, P):
    out = {}
    with torch.no_grad():
        for f in range(len(fset)):
            e = [x for x in exps if x.split == "train" and x.filt == f][0]
            g = model.gamma_at(torch.tensor(e.eid), torch.tensor([[0.0, -0.8], [0.0, 0.8], [-0.8, 0.0], [0.8, 0.0]]), P).mean(0)
            out.setdefault("y (injected): learned", []).append(float(g[1] - g[0]))
            out.setdefault("x (not injected): learned", []).append(float(g[3] - g[2]))
            ep = [x for x in exps if x.split == "train" and x.filt == f and x.t != e.t]
            if ep:
                g2 = model.gamma_at(torch.tensor(ep[0].eid), torch.zeros(1, 2), P).mean()
                g1 = model.gamma_at(torch.tensor(e.eid), torch.zeros(1, 2), P).mean()
                out.setdefault("epoch (not injected): learned", []).append(float(g2 - g1))
    truth_y = math.log(1.064 / 0.936)
    return {k: float(np.mean(v)) for k, v in out.items()} | {"y (injected): truth": truth_y, "x: truth": 0.0, "epoch: truth": 0.0}


def test_synthetic_recovery():
    exps, fset, insts, cfg, model, finfo = build(BUDGET_MIN)
    warm_kernel(model, cfg, finfo)
    res = {"n": {s: sum(e.split == s for e in exps) for s in ("train", "val", "held")}, "budget_min": BUDGET_MIN}
    res["heldout_truth_chi2"] = float(np.mean([float((((e.d - e.extra["clean"] - e.extra["bkg_true"]) / e.sig) ** 2).mean()) for e in exps if e.split == "held"]))
    snap0 = copy.deepcopy(model.state_dict())
    res["registration_prepass"] = register_prepass(model, exps, cfg)
    P = {k: v.detach() for k, v in model.prepare().items()}
    res["offset_residual_after_prepass_mas"] = offset_residual(model, exps, P)[0]
    snaps, t0 = [], time.perf_counter()
    marks = [0.15, 0.3, 0.5, 0.75]

    def hist(st, m):
        el = (time.perf_counter() - t0) / (BUDGET_MIN * 60)
        if marks and el >= marks[0]:
            snaps.append((marks.pop(0), st, copy.deepcopy(m.state_dict())))
    rec = fit(model, exps, cfg, history=hist)
    snaps.append((1.0, rec["n_steps"], copy.deepcopy(model.state_dict())))
    final = copy.deepcopy(model.state_dict())
    # held-out chi2 along the fit (each checkpoint: held-out nuisances refitted from scratch)
    traj = []
    model.load_state_dict(snap0)
    h0 = heldout_chi2(model, exps, cfg)
    traj.append({"fraction": 0.0, "step": 0, "chi2": h0["chi2"], "chi2_snr5": h0["chi2_snr5"]})
    for frac, st, sd in snaps:
        model.load_state_dict(sd)
        h = heldout_chi2(model, exps, cfg)
        traj.append({"fraction": frac, "step": st, "chi2": h["chi2"], "chi2_snr5": h["chi2_snr5"]})
        print(f"  held-out after {frac:.2f} of the budget (step {st}): chi2/N {h['chi2']:.3f}, S/N>5 {h['chi2_snr5']:.3f}", flush=True)
    res["heldout_trajectory"] = traj
    res["heldout_final_per_exposure"] = h["per_exposure"]
    model.load_state_dict(final)
    fit_heldout(model, exps, [e.eid for e in exps if e.split == "held"], cfg, log=lambda s: None)
    P = {k: v.detach() for k, v in model.prepare().items()}
    res["fit"] = {"steps": rec["n_steps"], "seconds": rec["seconds"], "s_per_step": rec["s_per_step"]}
    res["tau"] = model.tau.table()
    res["effective_variability_model_internal"] = model.effective_variability(P)
    res["relative_kernels"] = rel_kernel_checks(model, exps, fset, insts, P)
    res["relative_throughput"] = rel_throughput_checks(model, exps, fset, P)
    r_, t_ = offset_residual(model, exps, P)
    rh_, th_ = offset_residual(model, exps, P, split="held")
    res["relative_offsets"] = {"train_true_rms_mas": t_, "train_residual_rms_mas": r_, "held_true_rms_mas": th_, "held_residual_rms_mas": rh_}
    # gauge-dependent (absolute) quantities, reported only
    with torch.no_grad():
        res["gauge_dependent"] = {"blur_floor_m2": model.blur_floor, "dilation": float(P["dil"]) if "dil" in P else 1.0,
                                  "reference_filter": fset.names[model.blur_ref], "reference_core_m2": float(model.blur_moment(float(P.get("dil", 1.0)))),
                                  "gauges": model.gauge_report(P=P)}
    json.dump(res, open(os.path.join(OUT, "synth_recovery.json"), "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("heldout_truth_chi2", "heldout_trajectory", "fit", "relative_kernels", "relative_throughput",
                                          "relative_offsets", "gauge_dependent")}, indent=1, default=str)[:6000])
    print("tau", {k: round(v["value"], 5) for k, v in res["tau"].items()})
    # ---- checks
    chis = [t["chi2"] for t in traj]
    assert all(b <= a * 1.03 for a, b in zip(chis[1:], chis[2:])), f"held-out chi2 must not get worse with fit length: {chis}"
    assert chis[-1] < 1.5, chis
    rk = res["relative_kernels"]
    for k in ("space_x (injected)", "lambda (injected)"):
        assert rk[k]["error_rms"] < 0.3 * rk[k]["true_rms"], (k, rk[k])
    inj = min(rk["space_x (injected)"]["true_rms"], rk["lambda (injected)"]["true_rms"])
    for k in ("space_y (not injected)", "epoch (not injected)", "exposure (not injected)"):
        assert rk[k]["learned_rms"] < 0.3 * inj, (k, rk[k])
    th = res["relative_throughput"]
    assert abs(th["y (injected): learned"] - th["y (injected): truth"]) < 0.3 * th["y (injected): truth"]
    assert abs(th["x (not injected): learned"]) < 0.3 * th["y (injected): truth"] and abs(th["epoch (not injected): learned"]) < 0.3 * th["y (injected): truth"]
    assert res["relative_offsets"]["train_residual_rms_mas"] < 0.3 * res["relative_offsets"]["train_true_rms_mas"]
