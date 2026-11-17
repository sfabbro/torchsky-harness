"""Synthetic test of the learned shift per (detector, filter) pair (--filter-shift learned): the small synthetic field of synth_small.py
(two detectors, three filters, 8 exposures per filter: 6 train, 1 validation, 1 held out), with a known position shift in DETECTOR PIXELS injected
per pair into the data (not into the header positions the model is given), fitted with the switch off or learned, equal steps.
    python tests/synth_filter_shift.py --pkg . --inject 1 --filter-shift learned --out result.json
All exposures have the same roll here (rot_choices = (0,)), so that a common sky translation (the unrecoverable origin) is a pixel shift per pair
and the recovered minus injected shifts can be compared after removing it.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)


INJECTED = {("A", "A1"): (1.2, -0.8), ("A", "A2"): (-0.6, 1.6), ("B", "B1"): (0.5, 0.7)}      # detector pixels; 1.4, 1.7, 0.9 pixels


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--sky-form", default="grid")
    ap.add_argument("--legacy", type=int, default=0, help="untouched package: no --sky-form flag at all")
    ap.add_argument("--seed", type=int, default=0, help="synthetic field (truth, noise) AND fit seed")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--item-count", type=int, default=1500)
    ap.add_argument("--item-relocate", type=int, default=0)
    ap.add_argument("--item-smooth-pitch", type=float, default=0.0)
    ap.add_argument("--item-raster", default="", help="direct / pyramid (empty = not passed, for packages without the flag)")
    ap.add_argument("--item-patch", default="square")
    ap.add_argument("--item-impl", default="", help="prototype / torchsky (empty = not passed, for packages without the flag)")
    ap.add_argument("--sky-pitch", type=float, default=0.06)
    ap.add_argument("--inject", type=int, default=1, help="1: inject the known pair shifts into the data; 0: none (reference)")
    ap.add_argument("--filter-shift", default="off")
    ap.add_argument("--filter-shift-init", default="")
    ap.add_argument("--lr-filter-shift", type=float, default=0.03)
    ap.add_argument("--rolls", default="0", help="comma list of rotation choices (multiples of 90 degrees)")
    ap.add_argument("--truth-eval", type=int, default=1)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse()
    sys.path.insert(0, os.path.abspath(a.pkg))
    import fieldop
    from fieldop.cli import default_config, make_sky
    from fieldop.model import FieldOperator
    from fieldop.synth import SynthInstrument, make_field
    from fieldop.train import chi2_exposure, fit, fit_heldout, predict_exposure, register_prepass, warm_kernel
    T0 = time.perf_counter()
    torch.manual_seed(a.seed)
    insts = [SynthInstrument("A", 0.10, 40, 6.5, 0.07, 1.5, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", 0.12, 34, 6.5, 0.10, 2.4, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("B1", 2.0, 2.8)])]
    exps, fset, meta, truth, _ = make_field(seed=3 + a.seed, n_per_filter=8, n_val=1, n_held=1, insts=insts, shift_rms=0.03, centre_range=0.6,
                                            rot_choices=tuple(int(x) for x in a.rolls.split(",")), patch_px=20, truth_half=1.6,
                                            pair_shift_px=INJECTED if a.inject else None)
    # smaller than tests_ref/test_fieldop_synth.py to fit the 4-minute rule on a shared CPU: kernel extent 1.2" (was 1.5"), 12 level nodes
    # (was 16), 4 patches per step (was 6), variability scales FIXED at their starting values (no validation steps), short held-out fit
    kw = dict(sky_pitch=a.sky_pitch, render_pitch=0.02, level_nh=12, kernel_extent=1.2, region_px=13, n_lambda=3, k_hidden=32,
              warm_kernel_steps=150, pattern_grid=0, stripes=0, batch_patches=4, inner_sky=1, budget_min=1e6, steps=a.steps, tau_mode="fixed",
              val_every=2, held_steps=50, held_reg_steps=30, log_every=10, is_snr=10.0, reg_prepass_steps=40, reg_prepass_iters=1, seed=a.seed)
    kw.update(filter_shift=a.filter_shift, lr_filter_shift=a.lr_filter_shift, filter_shift_init=a.filter_shift_init)
    if not a.legacy:
        kw.update(sky_form=a.sky_form, item_count=a.item_count, item_relocate=a.item_relocate, item_smooth_pitch=a.item_smooth_pitch)
        if a.item_raster:
            kw.update(item_raster=a.item_raster, item_patch=a.item_patch)
        if a.item_impl:
            kw.update(item_impl=a.item_impl)
    cfg = default_config(**kw)
    sky, N = make_sky(exps, fset, cfg, 2.8, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    model.attach_filter_shift(exps, cfg, fset_names=fset.names, det_names=["A", "B"])
    finfo = {}
    for f in range(len(fset)):
        ii = [i for i, ins in enumerate(insts) if any(fn == fset.names[f] for fn, _, _ in ins.filters)][0]
        finfo[f] = {"pixel_arcsec": insts[ii].pixel, "diameter_m": 6.5, "inst": ii}
    logs = []
    lg = lambda s: logs.append(s)  # noqa: E731
    t = time.perf_counter()
    warm_kernel(model, cfg, finfo, log=lg)
    reg = register_prepass(model, exps, cfg, log=lg)
    t_setup = time.perf_counter() - t
    rec = fit(model, exps, cfg, log=lg)
    held = [e.eid for e in exps if e.split == "held"]
    t = time.perf_counter()
    fit_heldout(model, exps, held, cfg, log=lg)
    P = {k: v.detach() for k, v in model.prepare().items()}
    sc = {}
    for e in exps:
        if e.split == "held":
            m, pv, S = predict_exposure(model, e, P, cfg, tile=20)
            sc[e.name] = chi2_exposure(e, m, pv, S=S, snr_min=5.0)
    t_held = time.perf_counter() - t
    n = sum(v["n"] for v in sc.values())
    nb = sum(v["n_bright"] for v in sc.values())
    chi2 = sum(v["chi2"] * v["n"] for v in sc.values()) / n
    chi2b = sum(v["chi2_bright"] * v["n_bright"] for v in sc.values() if v["n_bright"]) / max(nb, 1)
    noise_floor = float(np.mean([float((((e.d - e.extra["clean"] - e.extra["bkg_true"]) / e.sig) ** 2).mean()) for e in exps if e.split == "held"]))
    st = model.state_dict()
    h = hashlib.sha256()
    for k in sorted(st):
        h.update(k.encode())
        h.update(st[k].detach().cpu().contiguous().numpy().tobytes())
    fs = None
    if model.fs_on:
        fs = shift_report(model, exps)
    n_fit = sky.n_fitted() if hasattr(sky, "n_fitted") else sky.A[0].numel()
    res = {"pkg": a.pkg, "sky_form": a.sky_form, "seed": a.seed, "steps": rec["n_steps"], "fit_seconds": rec["seconds"], "s_per_step": rec["s_per_step"],
           "setup_seconds": t_setup, "heldout_seconds": t_held, "train_loss_log": rec["steps"], "heldout_chi2": chi2, "heldout_chi2_snr5": chi2b,
           "heldout_n_px": n, "heldout_n_px_snr5": nb, "heldout_per_exposure": sc, "heldout_truth_noise_chi2": noise_floor,
           "sky_fitted_numbers": int(n_fit), "sky_grid_N": N, "state_sha256": h.hexdigest(), "registration": reg,
           "relocated": rec.get("relocated", []), "filter_shift": fs, "inject": a.inject, "filter_shift_mode": a.filter_shift,
           "injected_px": {f"{k[0]}/{k[1]}": list(v) for k, v in INJECTED.items()} if a.inject else None, "total_seconds": time.perf_counter() - T0}
    if a.truth_eval:
        res["sky_error"] = sky_error(model, exps, fset, truth)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    short = {k: res[k] for k in ("sky_form", "seed", "steps", "s_per_step", "heldout_chi2", "heldout_chi2_snr5", "heldout_truth_noise_chi2",
                                 "sky_fitted_numbers", "state_sha256", "total_seconds")}
    print(json.dumps(short))
    if "sky_error" in res:
        print(json.dumps(res["sky_error"]["summary"]))


def shift_report(model, exps):
    """Recovered minus injected shift per pair (detector pixels) and the same after removing the best common SKY translation c* (the origin the
    data cannot fix): c* = argmin sum_p n_p |J_p delta_p - c|^2, residual r_p = delta_p - J_p^-1 c*, J_p = the pair's Jacobian (arcsec per pixel)."""
    labels = model.fs_labels
    rec = model.fshift.detach().double().numpy()
    inj = np.array([INJECTED[tuple(l.split("/"))] for l in labels], np.float64)
    J = []
    for l in labels:
        e = next(x for x in exps if f"{['A', 'B'][x.det]}/{model.fset.names[x.filt]}" == l)
        J.append(np.asarray(e.extra["J"], np.float64))
    n = np.array(model.fs_ntrain, np.float64)
    delta = rec - inj
    sky = np.stack([J[i] @ delta[i] for i in range(len(labels))])
    c = (n[:, None] * sky).sum(0) / n.sum()
    resid = np.stack([np.linalg.solve(J[i], sky[i] - c) for i in range(len(labels))])
    return {"pairs": labels, "recovered_px": rec.tolist(), "recovered_minus_injected_px": delta.tolist(), "common_sky_translation_arcsec": c.tolist(),
            "residual_after_common_translation_px": resid.tolist(), "max_residual_px": float(np.abs(resid).max()),
            "max_residual_norm_px": float(np.linalg.norm(resid, axis=1).max())}


def sky_error(model, exps, fset, truth, sig_c=0.06, eval_pitch=0.02, half_eval=1.4, r_near=0.15):
    """Error of the fitted sky against the true sky, per filter band, both smoothed by a Gaussian of sigma sig_c [arcsec] (about the sharpest
    true PSF core, so that neither side is asked for detail the data cannot hold):
      fitted: band image of the continuous sky (tent interpolant of coeffs(), read at eval_pitch) convolved with the sig_c Gaussian (FFT);
      truth : analytic, point sources -> Gaussians of sig_c, galaxies -> Gaussians of sqrt(sig_g^2 + sig_c^2), band flux = quadrature of
              amp (lam/2)^beta with the package's own band weights (the same lambda nodes and weights the model's Phi_band uses);
      truth positions shifted by minus the mean true registration offset of the training exposures (the common-shift gauge zeroes the
      mean learned translation, so the fitted sky sits there).
    Over |X|, |Y| < half_eval (inside the region covered by the data): relative rms error sqrt(sum (fit - true)^2 / sum true^2), split into
    nodes within r_near of a true point source ("near point sources") and the rest; also with the best single relative blur (gauge-adjusted:
    the common blur is a degeneracy between sky and kernels above the floor), the extra Gaussian of width d in {0, 0.01, ..., 0.12} applied
    to whichever side is sharper (chosen by the smallest total error), and with the best per-band flux scale."""
    with torch.no_grad():
        sky = model.sky
        A = sky.coeffs()
        g = torch.arange(-half_eval - 3 * 0.1, half_eval + 3 * 0.1 + 1e-9, eval_pitch)
        YY, XX = torch.meshgrid(g, g, indexing="ij")
        Ak = sky.sample_coeffs(A, XX, YY)                                  # (K, n, n)
        fitb = torch.einsum("bk,khw->bhw", model.Phi_band, Ak)             # (nb, n, n)
        tr = [e for e in exps if e.split == "train"]
        c = np.mean([e.extra["true_shift"] for e in tr], 0)
        lam, w, _, _ = fset.tensors("cpu")                                  # (nb, Q)
        n = g.numel()

        def gauss_img(x, y, s):
            return torch.exp(-0.5 * ((XX - x) ** 2 + (YY - y) ** 2) / s ** 2) / (2 * math.pi * s * s)

        def truth_band(extra):
            out = torch.zeros(len(fset), n, n)
            for b in range(len(fset)):
                for x, y, amp, beta in truth.pts:
                    Fb = float((w[b] * amp * (lam[b] / 2.0) ** beta).sum())
                    out[b] += Fb * gauss_img(x - c[0], y - c[1], math.sqrt(sig_c ** 2 + extra ** 2))
                for x, y, sg, amp, beta, _ in truth.gals:
                    Fb = float((w[b] * amp * (lam[b] / 2.0) ** beta).sum())
                    out[b] += Fb * gauss_img(x - c[0], y - c[1], math.sqrt(sg ** 2 + sig_c ** 2 + extra ** 2))
            return out

        def smooth(img, s):
            fx = torch.fft.fftfreq(n, d=eval_pitch)
            G = torch.exp(-2 * math.pi ** 2 * s ** 2 * (fx[:, None] ** 2 + fx[None, :] ** 2))
            return torch.fft.ifft2(torch.fft.fft2(img) * G).real          # periodic: the 0.3" padding absorbs the wrap

        inner = (XX.abs() < half_eval) & (YY.abs() < half_eval)
        near = torch.zeros_like(inner)
        for x, y, _, _ in truth.pts:
            near |= ((XX - (x - c[0])) ** 2 + (YY - (y - c[1])) ** 2) < r_near ** 2
        masks = {"all": inner, "near_point_sources": inner & near, "elsewhere": inner & ~near}

        def errs(F_, T_, scale=False):
            out = {}
            for mk, m in masks.items():
                num, den = 0.0, 0.0
                for b in range(len(fset)):
                    f_, t_ = F_[b][m], T_[b][m]
                    if scale:
                        f_ = f_ * float((f_ * t_).sum() / (f_ * f_).sum().clamp(min=1e-30))
                    num += float(((f_ - t_) ** 2).sum())
                    den += float((t_ ** 2).sum())
                out[mk] = math.sqrt(num / max(den, 1e-30))
            return out
        Fc = smooth(fitb, sig_c)
        Tc = truth_band(0.0)
        raw = errs(Fc, Tc)
        best = (raw["all"], "none", 0.0, raw)
        for d in [0.01 * i for i in range(1, 13)]:
            e1 = errs(smooth(fitb, math.sqrt(sig_c ** 2 + d ** 2)), Tc)        # fitted sky sharper than truth: blur the fit
            if e1["all"] < best[0]:
                best = (e1["all"], "fit", d, e1)
            e2 = errs(Fc, truth_band(d))                                       # fitted sky blurrier: blur the truth
            if e2["all"] < best[0]:
                best = (e2["all"], "truth", d, e2)
        Fb_ = smooth(fitb, math.sqrt(sig_c ** 2 + best[2] ** 2)) if best[1] == "fit" else Fc
        Tb_ = truth_band(best[2]) if best[1] == "truth" else Tc
        scaled = errs(Fb_, Tb_, scale=True)
        summ = {"raw": raw, "gauge_adjusted": best[3], "gauge_adjusted_blur": [best[1], best[2]], "gauge_adjusted_and_flux_scaled": scaled,
                "n_nodes": {k: int(v.sum()) for k, v in masks.items()}, "sig_c": sig_c, "r_near": r_near, "mean_true_shift_mas": (c * 1e3).tolist()}
        return {"summary": summ}


if __name__ == "__main__":
    main()
