"""Driver for the known-truth problem (see truth_problem.py and TRUTH_REPORT.md).

Fit one realisation (noise seed = fit seed = --seed) of the injected-source field and measure it:

    python tests/truth/run_truth.py fit --form grid --n-train 8 --seed 0 --steps 400 --out tests/truth/results/grid_n8_s0.json
    python tests/truth/run_truth.py summarise tests/truth/results/grid_n8_s?.json --out tests/truth/results/summary_grid_n8.json

Defaults are the converging configuration of the report: response (kernel) started at the truth and frozen, noise scale frozen, per-exposure
background and registration learned, sky learned. `--freeze-kernel 0` learns the response from the package's own diffraction warm start.
Run with PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; one core.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import truth_problem as tp  # noqa: E402

torch.set_num_threads(1)


def cmd_fit(a):
    pop = tp.make_population()
    fluxes = tp.flux_for_snr(pop, n_ref=8)
    empties = tp.empty_positions(pop)
    if a.max_sources:
        pop, fluxes = pop[:a.max_sources], fluxes[:a.max_sources]
    T0 = time.perf_counter()
    exps, fset, meta, truth, pf, _ = tp.build_field(a.n_train, seed=a.seed, pop=pop, fluxes=fluxes, npix=a.npix)
    cfg_over = {}
    if a.sky_neg is not None:
        cfg_over["sky_neg"] = a.sky_neg
    if a.blur_floor is not None:
        cfg_over["blur_floor"] = a.blur_floor
    if a.item_count:
        cfg_over["item_count"] = a.item_count
    for kv in a.set or []:                    # generic configuration override, e.g. --set sky_zero=minimal (value parsed as a number when it is one)
        k, v = kv.split("=", 1)
        try:
            v = float(v) if any(c in v for c in ".e") else int(v)
        except ValueError:
            pass
        cfg_over[k] = v
    cfg = tp.base_config(a.form, steps=a.steps, seed=a.seed, **cfg_over)
    tp.KERNEL_STEPS = a.kernel_steps
    logs = []
    model, res = tp.run_fit(exps, fset, meta, cfg, freeze_kernel=bool(a.freeze_kernel), sky_half=a.sky_half, log=lambda s: logs.append(s))
    res["log_tail"] = [l for l in logs if l.startswith("step")][-1:]
    res["config"] = {k: getattr(a, k) for k in ("form", "n_train", "seed", "steps", "freeze_kernel", "sky_neg", "blur_floor", "kernel_steps", "npix", "item_count", "set")}
    res["loss_curve"] = [[int(r[0]), round(float(r[1]), 1), round(float(r[2]), 4), round(float(r[3]), 4)] for r in res["loss_curve"]]
    if a.measure:
        tr = [e for e in exps if e.split == "train"]
        t = time.perf_counter()
        cs = tp.IdealCache([(p[0], p[1], p[2]) for p in pop], tr, fset)
        ce = tp.IdealCache([(x, y, tp.SIG_RES) for x, y in empties], tr, fset)
        res["measure"] = tp.measure(model, exps, fset, pop, fluxes, empties, cs, ce)
        res["measure_seconds"] = time.perf_counter() - t
        # sky error against the truth at the nodes, in the region covered by every exposure (|x|, |y| < 1.5)
        img = tp.sky_band_image(model)
        tt = tp.true_sky_nodes(pf, img.shape[0], 0.05)
        c = (img.shape[0] - 1) // 2
        w = int(1.5 / 0.05)
        sl = (slice(c - w, c + w + 1), slice(c - w, c + w + 1))
        res["sky_nodes"] = {"rms_error_inner": float(((img - tt)[sl] ** 2).mean().sqrt()), "mean_error_inner": float((img - tt)[sl].mean()),
                            "min_value": float(img.min()), "frac_negative_inner": float((img[sl] < 0).double().mean())}
    if a.save_sky:                            # fitted sky band image at the nodes and the training backgrounds, for later re-measurement
        tr_ids = [e.eid for e in exps if e.split == "train"]
        np.savez_compressed(os.path.splitext(a.out)[0] + "_sky.npz", sky=tp.sky_band_image(model).numpy(),
                            bkg=model.bkg[tr_ids].detach().numpy(), bkg_ref=model.bkg0[tr_ids].detach().numpy())
    res["sky_zero_record"] = res.get("sky_zero_record")
    res["wall_seconds"] = time.perf_counter() - T0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("sky_form", "n_train", "steps", "fit_seconds", "total_seconds", "wall_seconds", "heldout_chi2", "noise_only_chi2", "ratio_to_floor")}))


def cmd_summarise(a):
    runs = [json.load(open(p)) for p in a.inputs]
    ms = [r["measure"] for r in runs]
    out = {"inputs": a.inputs, "n_realisations": len(runs), "form": runs[0]["sky_form"], "n_train": runs[0]["n_train"], "config": runs[0]["config"],
           "heldout_ratio_to_floor": [r["ratio_to_floor"] for r in runs], "fit_seconds": [r["fit_seconds"] for r in runs],
           "sources": tp.summarise_sources(ms), "empty": tp.summarise_empty(ms)}
    out["empty"]["sigma_ideal_mean_flux_units"] = float(np.mean(ms[0]["emp"]["sigma_ideal"]))
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"{out['form']} N={out['n_train']} R={len(runs)} held-out/floor {np.round(out['heldout_ratio_to_floor'], 4).tolist()}")
    for row in out["sources"]:
        print(row["bin"], row["n_sources"], "ratio rec %.3f+-%.3f  matched %.3f+-%.3f  ideal %.3f+-%.3f | bias(noise units) rec %.2f ideal %.2f" % (
            row["rec"]["mean_ratio"], row["rec"]["sem_pooled"], row["rec_matched"]["mean_ratio"], row["rec_matched"]["sem_pooled"], row["ideal"]["mean_ratio"],
            row["ideal"]["sem_pooled"], row["rec"]["mean_bias_noise_units"], row["ideal"]["mean_bias_noise_units"]))
    e = out["empty"]
    print("empty: mean %.3f+-%.3f (positions %.3f) raw-gauge %.3f rms %.3f sd-about-mean %.3f | ideal mean %.3f rms %.3f | rms ratio %.3f" % (
        e["rec"]["mean_sigma"], e["rec"]["sem_realisations"], e["rec"]["sem_positions"], e["rec"]["mean_sigma_raw_gauge"], e["rec"]["rms_sigma"], e["rec"]["sd_about_mean"],
        e["ideal"]["mean_sigma"], e["ideal"]["rms_sigma"], e["ratio_rec_over_ideal_rms"]))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--form", default="grid", choices=["grid", "items"])
    f.add_argument("--n-train", type=int, default=8)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--steps", type=int, default=400)
    f.add_argument("--freeze-kernel", type=int, default=1)
    f.add_argument("--kernel-steps", type=int, default=2500)
    f.add_argument("--sky-neg", type=float, default=None)
    f.add_argument("--blur-floor", type=float, default=None)
    f.add_argument("--item-count", type=int, default=0)
    f.add_argument("--npix", type=int, default=44)
    f.add_argument("--sky-half", type=float, default=2.5)
    f.add_argument("--max-sources", type=int, default=0)
    f.add_argument("--set", action="append", help="configuration override key=value (repeatable)")
    f.add_argument("--measure", type=int, default=1)
    f.add_argument("--save-sky", type=int, default=1)
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_fit)
    s = sp.add_parser("summarise")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_summarise)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
