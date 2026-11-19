"""Demonstration of the kernel-centroid gauge (F43, KERNEL_CENTROID_GAUGE_2026-11-19.md section 5): two fits of one synthetic field on
DISJOINT halves of the training exposures, with the gauge off and on; the common shift between the two fitted skies, off against on.

    PYTHONNOUSERSITE=1 python tests/demo_kernel_centroid_halves.py run --seed 0 --half 0 --gauge 1 --steps 300 --out R.json
    PYTHONNOUSERSITE=1 python tests/demo_kernel_centroid_halves.py summary R1.json R2.json ...

Field: the synthetic field of compare_port.py / synth_small.py (two instruments, three filters, exposures at four orientations or, with
--rot 0, at one orientation as in SMACS 0723, 8 exposures
per filter: 1 held out and shared by both halves, the other 7 split alternately 4 / 3). Not collected by pytest (several minutes per fit).

Per fit (all in arcsec, sky frame, then reported in mas):
  F      = common position offset of the fitted sky against the truth: windowed centroids (Gaussian window, sigma 0.08 arcsec, iterated
           re-centring) of the fitted band image at each true point source, minus the same centroid on the TRUE sky rasterised in the same way
           (so the estimator's own bias cancels), averaged over sources and bands; error = scatter / sqrt(number).
  a_reg  = mean over the training exposures of (shift0_e + t_e - true_shift_e): the frame set by the registration and the zero-mean
           translation convention.
  kappa  = mean over the training exposures of R_e mu_c(e): the sky-frame windowed centroid of the RENDERED mean kernels (the held moment
           of the gauge, with the class offset d_c when the gauge is on; it is zero when the gauge is on).
The model predicts F = a_reg + kappa (the true kernel is centred). Between the halves: dF = da_reg + dkappa; with the gauge on dkappa = 0 by
construction, so the free part of the frame is gone and dF - da_reg should be zero within its error.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
torch.set_num_threads(1)


def run(a):
    from fieldop.cli import apply_sky_module_options, default_config, make_sky
    from fieldop.data import build_meta
    from fieldop.instruments import spectral_setup
    from fieldop.model import FieldOperator
    from fieldop.synth import SynthInstrument, make_field
    from fieldop.train import fit, register_prepass, warm_kernel
    insts = [SynthInstrument("A", 0.10, 40, 6.5, 0.07, 1.5, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", 0.12, 34, 6.5, 0.10, 2.4, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("B1", 2.0, 2.8)])]
    torch.manual_seed(a.seed)
    exps, fset, meta, truth, _ = make_field(seed=3 + a.seed, n_per_filter=8, n_held=1, insts=insts, shift_rms=0.03, centre_range=0.6,
                                            rot_choices=tuple(int(c) for c in a.rot), patch_px=20, truth_half=1.6)
    # disjoint halves: training exposures of each filter alternately to half 0 and half 1; the other half is not used (split 'unused')
    k_in_f = {}
    for e in exps:
        if e.split == "train":
            j = k_in_f.setdefault(e.filt, 0)
            k_in_f[e.filt] += 1
            if j % 2 != a.half:
                e.split = "unused"
    meta = build_meta(exps, ["A", "B"], ["A", "B"], ["A", "B"])
    kw = dict(sky_pitch=0.06, render_pitch=0.02, level_nh=12, kernel_extent=1.2, region_px=13, n_lambda=3, k_hidden=32, warm_kernel_steps=150,
              batch_patches=4, inner_sky=1, budget_min=1e6, steps=a.steps, val_every=2, held_steps=50, held_reg_steps=30, log_every=50,
              is_snr=10.0, reg_prepass_steps=40, reg_prepass_iters=1, seed=a.seed, item_count=1500, gauge_ref_inst=0, device="cpu",
              kernel_centroid_gauge=a.gauge)
    cfg = default_config(**kw)
    apply_sky_module_options(cfg)
    spectral_setup(fset, cfg, log=lambda s: None)
    sky, N = make_sky(exps, fset, cfg, 2.8, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    finfo = {f: {"pixel_arcsec": insts[0 if f < 2 else 1].pixel, "diameter_m": 6.5, "inst": 0 if f < 2 else 1} for f in range(len(fset))}
    lines = []
    warm_kernel(model, cfg, finfo, log=lines.append)
    register_prepass(model, exps, cfg, log=lines.append)
    t0 = time.perf_counter()
    rec = fit(model, exps, cfg, log=lines.append)
    out = {"seed": a.seed, "half": a.half, "gauge": a.gauge, "rot": a.rot, "steps": a.steps, "s_per_step": rec["s_per_step"], "seconds": time.perf_counter() - t0,
           "train": [e.name for e in exps if e.split == "train"], "dilation_final": rec["dilation_final"], "log_tail": lines[-3:]}
    out.update(measure(model, exps, fset, truth, insts, cfg))
    if a.gauge:
        out["gauge_record"] = rec["kernel_centroid_gauge"]
    json.dump(out, open(a.out, "w"), indent=1)
    if a.save_state:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, a.save_state)
    print(json.dumps({k: v for k, v in out.items() if k not in ("train", "per_source")}, indent=1))


def windowed_centroid(img, x0, pitch, px, py, sig=0.08, iters=4):
    """Iterated Gaussian-windowed centroid of img (N, N; node (i, j) at (x0 + j p, x0 + i p)) starting at (px, py)."""
    N = img.shape[0]
    g = x0 + np.arange(N) * pitch
    X, Y = np.meshgrid(g, g)
    cx, cy = px, py
    for _ in range(iters):
        w = np.exp(-0.5 * ((X - cx) ** 2 + (Y - cy) ** 2) / sig ** 2) * img
        s = w.sum()
        cx, cy = float((w * X).sum() / s), float((w * Y).sum() / s)
    return cx, cy


@torch.no_grad()
def measure(model, exps, fset, truth, insts, cfg):
    sky = model.sky
    img = sky.band_images(model.Phi_band).cpu().numpy()             # (nb, N, N)
    # the TRUE sky rasterised on the same grid: point sources and galaxies as narrow Gaussians (sigma 0.03 arcsec) of the band flux
    from fieldop.synth import quantile_nodes
    N, p, x0 = sky.N, sky.pitch, sky.x0
    g = x0 + np.arange(N) * p
    X, Y = np.meshgrid(g, g)
    tru = np.zeros_like(img)
    for b, flt in enumerate(fset.filters):
        lam, w = quantile_nodes(flt.lam_um, flt.T, 12)
        lam, w = np.asarray(lam, float), np.asarray(w, float)
        for x, y, amp, beta in truth.pts:
            F = float((w * amp * (lam / 2.0) ** beta).sum() / w.sum())
            tru[b] += F * np.exp(-0.5 * ((X - x) ** 2 + (Y - y) ** 2) / 0.03 ** 2)
        for x, y, sg, amp, beta, _ in truth.gals:
            F = float((w * amp * (lam / 2.0) ** beta).sum() / w.sum())
            s2 = 0.03 ** 2 + sg ** 2
            tru[b] += F * 0.03 ** 2 / s2 * np.exp(-0.5 * ((X - x) ** 2 + (Y - y) ** 2) / s2)
    lim = 0.75 * sky.half_extent
    per = []
    for b in range(img.shape[0]):
        for k, (x, y, amp, beta) in enumerate(truth.pts):
            if abs(x) > min(lim, 1.4) or abs(y) > min(lim, 1.4):
                continue
            # isolation: no other point source or galaxy within 0.3 arcsec
            others = [(xx, yy) for kk, (xx, yy, *_r) in enumerate(truth.pts) if kk != k] + [(xx, yy) for xx, yy, *_r in truth.gals]
            if any(math.hypot(x - xx, y - yy) < 0.3 for xx, yy in others):
                continue
            fx, fy = windowed_centroid(img[b], x0, p, x, y)
            tx, ty = windowed_centroid(tru[b], x0, p, x, y)
            per.append([b, k, (fx - tx) * 1e3, (fy - ty) * 1e3])
    arr = np.array(per)
    tr = model.ref_eids
    e_by = {e.eid: e for e in exps}
    t = (model.affine[tr, 4:6] * model.aff_scale[4:6]).cpu().numpy()
    s0 = model.shift0[tr].cpu().numpy()
    ts = np.array([e_by[int(i)].extra["true_shift"] for i in tr.tolist()])
    a_reg = (s0 + t - ts).mean(0) * 1e3
    # kappa: windowed centroid of the rendered mean kernel per class, rotated to the sky, averaged over the training exposures
    if not model.kc_on:          # measurement only: the gauge's helpers, offsets zero (the render does not read them)
        meta = model.meta
        fpix, fdiam = {}, {}
        for i in range(model.E):
            fpix.setdefault(int(meta["filt"][i]), meta["chan_pix"][int(meta["chan"][i])])
            fdiam.setdefault(int(meta["filt"][i]), 6.5)
        model._init_kernel_centroid(meta, fpix, fdiam)
    dl = model.dilation()
    dil = None if dl is None else dl.detach()
    mu = torch.stack([model.kernel_centroid(c, model.kc_off[c], dil) for c in range(len(model.kc_pairs))])
    kap = torch.einsum("eij,ej->ei", model.e_R[tr], mu[model.kc_class[tr]]).mean(0).cpu().numpy() * 1e3
    F = arr[:, 2:4].mean(0)
    Fe = arr[:, 2:4].std(0, ddof=1) / math.sqrt(len(arr))
    return {"F_mas": F.tolist(), "F_err_mas": Fe.tolist(), "n_source_bands": len(arr), "a_reg_mas": a_reg.tolist(), "kappa_mas": kap.tolist(),
            "class_centroid_det_mas": (mu.cpu().numpy() * 1e3).tolist(), "per_source": arr.tolist()}


def summary(files):
    rs = [json.load(open(f)) for f in files]
    by = {}
    for r in rs:
        by[(r.get("rot", "0123"), r["seed"], r["gauge"], r["half"])] = r
    rows = []
    for (rot, seed, gauge) in sorted({(o, s, g) for o, s, g, _ in by}):
        if (rot, seed, gauge, 0) not in by or (rot, seed, gauge, 1) not in by:
            continue
        A, B = by[(rot, seed, gauge, 0)], by[(rot, seed, gauge, 1)]
        pa, pb = np.array(A["per_source"]), np.array(B["per_source"])
        ka = {(int(b), int(k)): r for b, k, *r in pa}
        kb = {(int(b), int(k)): r for b, k, *r in pb}
        common = sorted(set(ka) & set(kb))
        d = np.array([np.array(ka[c]) - np.array(kb[c]) for c in common])          # per source and band, half 0 - half 1 [mas]
        dF, dFe = d.mean(0), d.std(0, ddof=1) / math.sqrt(len(d))
        da = np.array(A["a_reg_mas"]) - np.array(B["a_reg_mas"])
        dk = np.array(A["kappa_mas"]) - np.array(B["kappa_mas"])
        rows.append({"orientations": rot, "seed": seed, "gauge": gauge, "steps": A["steps"], "n": len(d), "dF_mas": dF.round(2).tolist(), "dF_err_mas": dFe.round(2).tolist(),
                     "da_reg_mas": da.round(2).tolist(), "dkappa_mas": dk.round(2).tolist(),
                     "dF_minus_da_reg_mas": (dF - da).round(2).tolist(), "kappa_half0_mas": np.round(A["kappa_mas"], 2).tolist(),
                     "kappa_half1_mas": np.round(B["kappa_mas"], 2).tolist(), "F_minus_areg_minus_kappa_half0": np.round(np.array(A["F_mas"]) - A["a_reg_mas"] - np.array(A["kappa_mas"]), 2).tolist(),
                     "F_minus_areg_minus_kappa_half1": np.round(np.array(B["F_mas"]) - B["a_reg_mas"] - np.array(B["kappa_mas"]), 2).tolist()})
    print(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--half", type=int, default=0)
    ap.add_argument("--gauge", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="demo.json")
    ap.add_argument("--save-state", default="")
    ap.add_argument("--rot", default="0123", help="orientations of the exposures (multiples of 90 deg): '0123' (default) or '0' (one orientation, as SMACS)")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summary(a.files)
