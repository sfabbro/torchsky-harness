"""Synthetic test of the per-pixel gain map (--pixel-gain, F43-F45), CPU, under 4 minutes per run.

Field = tests/synth_small.py's (two instruments A: 40 x 40 pixels of 0.10", B: 34 x 34 of 0.12"; three filters; 8 exposures per filter:
6 train, 1 validation, 1 held out; injected PSF and smooth throughput variation; ~30 mas registration offsets), with two changes:
  * background raised from 5 to --bkg (default 40, noise 1), so that a per-pixel gain is measurable from the background alone
    (per-exposure S/N of a background pixel ~ 40, as for a NIRCam short-wavelength pixel at ~ 0.3 DN/s in 800 s);
  * optional injected per-pixel gain (--inject-rms, white, per detector): d = G (signal + background) + noise.

    python tests/synth_gain.py --inject-rms 0.03 --pixel-gain detector --steps 200 --out r.json

Reports: held-out chi2 (all pixels, and background-dominated pixels: true signal < 0.5 noise), the noise floor (truth), the recovered gain
map against the injected one after the same gauge projection (correlation, rms of the difference, expected noise limit from the
information map), and the gauge residuals (projected components, unobserved pixels) tracked every 10 steps.
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
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject-rms", type=float, default=0.0)
    ap.add_argument("--pixel-gain", default="off")
    ap.add_argument("--pixel-gain-poly", type=int, default=2)
    ap.add_argument("--lr-gain", type=float, default=1e-3)
    ap.add_argument("--pixel-gain-start", type=float, default=0.1)
    ap.add_argument("--bkg", type=float, default=40.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def projected(g, obs, box, order):
    """g (nr, nc) with the monomials of degree <= order removed over obs (as PixelGain.project); 0 elsewhere."""
    from fieldop.model import poly_basis
    r0, c0, nr, nc = box
    idx = np.nonzero(obs.ravel())[0]
    rr, cc = torch.as_tensor(idx // nc + r0), torch.as_tensor(idx % nc + c0)
    V = poly_basis(rr, cc, box, order).numpy()
    Q = np.linalg.qr(V)[0]
    v = g.ravel()[idx].astype(np.float64)
    out = np.zeros(nr * nc)
    out[idx] = v - Q @ (Q.T @ v)
    return out.reshape(nr, nc)


def main():
    a = parse()
    from fieldop.cli import default_config, make_sky
    from fieldop.model import FieldOperator
    from fieldop.synth import SynthInstrument, make_field
    from fieldop.train import fit, fit_heldout, predict_exposure, register_prepass, warm_kernel
    T0 = time.perf_counter()
    torch.manual_seed(a.seed)
    insts = [SynthInstrument("A", 0.10, 40, 6.5, 0.07, 1.5, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", 0.12, 34, 6.5, 0.10, 2.4, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("B1", 2.0, 2.8)])]
    exps, fset, meta, truth, _ = make_field(seed=3 + a.seed, n_per_filter=8, n_val=1, n_held=1, insts=insts, shift_rms=0.03, centre_range=0.6,
                                            rot_choices=(0, 1, 2, 3), patch_px=20, truth_half=1.6, pix_gain_rms=a.inject_rms, bkg_mean=a.bkg)
    kw = dict(sky_pitch=0.06, render_pitch=0.02, level_nh=12, kernel_extent=1.2, region_px=13, n_lambda=3, k_hidden=32,
              warm_kernel_steps=150, pattern_grid=0, stripes=0, batch_patches=4, inner_sky=1, budget_min=1e6, steps=a.steps, tau_mode="fixed",
              val_every=2, held_steps=50, held_reg_steps=30, log_every=10, is_snr=10.0, reg_prepass_steps=40, reg_prepass_iters=1, seed=a.seed,
              pixel_gain=a.pixel_gain, pixel_gain_poly=a.pixel_gain_poly, lr_gain=a.lr_gain, pixel_gain_start=a.pixel_gain_start)
    cfg = default_config(**kw)
    sky, N = make_sky(exps, fset, cfg, 2.8, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    pg = model.attach_pixel_gain(exps, cfg, fset_names=fset.names, det_names=["A", "B"])
    finfo = {}
    for f in range(len(fset)):
        ii = [i for i, ins in enumerate(insts) if any(fn == fset.names[f] for fn, _, _ in ins.filters)][0]
        finfo[f] = {"pixel_arcsec": insts[ii].pixel, "diameter_m": 6.5, "inst": ii}
    logs = []
    lg = lambda s: logs.append(s)  # noqa: E731
    warm_kernel(model, cfg, finfo, log=lg)
    register_prepass(model, exps, cfg, log=lg)
    gtrack = []

    def hist(step, m):
        if m.pixgain is not None and step % 10 == 0:
            r = m.pixgain.gauge_residual()
            means = [float(m.pixgain.logg[getattr(m.pixgain, f"pidx{k}")].mean()) for k in range(len(m.pixgain.boxes))]
            gtrack.append([step, r["max_projected_component_rms"], r["max_abs_unobserved"], max(abs(x) for x in means),
                           float(m.pixgain.logg.abs().max())])
    rec = fit(model, exps, cfg, log=lg, history=hist)
    loss_tail = [r[2] for r in rec["steps"][-5:]]
    held = [e.eid for e in exps if e.split == "held"]
    fit_heldout(model, exps, held, cfg, log=lg)
    P = {k: v.detach() for k, v in model.prepare().items()}
    sc = {}
    tot = {"all": [0.0, 0], "faint": [0.0, 0], "truth_all": [0.0, 0], "truth_faint": [0.0, 0]}
    with torch.no_grad():
        for e in exps:
            if e.split != "held":
                continue
            m, pv, S = predict_exposure(model, e, P, cfg, tile=20)
            v = e.valid & pv & torch.isfinite(m)
            r2 = ((e.d - m) / e.sig) ** 2
            G = e.extra.get("gain_true", torch.ones_like(e.d))
            t2 = ((e.d - G * (e.extra["clean"] + e.extra["bkg_true"])) / e.sig) ** 2
            faint = v & (e.extra["clean"] < 0.5 * e.sig)
            for key, mask, arr in (("all", v, r2), ("faint", faint, r2), ("truth_all", v, t2), ("truth_faint", faint, t2)):
                tot[key][0] += float(arr[mask].sum())
                tot[key][1] += int(mask.sum())
            sc[e.name] = {"chi2": float(r2[v].mean()), "chi2_faint": float(r2[faint].mean()), "n": int(v.sum()), "n_faint": int(faint.sum())}
    res = {"args": vars(a), "steps": rec["n_steps"], "fit_seconds": rec["seconds"], "loss_last5_logged": loss_tail,
           "heldout": {k: v[0] / max(v[1], 1) for k, v in tot.items()}, "heldout_n": {k: v[1] for k, v in tot.items()}, "per_exposure": sc}
    if pg is not None:
        rows = []
        for k, lab in enumerate(pg.labels):
            lgm, ob, cnt, inf_ = pg.map(k)
            r0, c0, nr, nc = pg.boxes[k]
            out = {"label": lab, "box": pg.boxes[k], "n_observed": int(ob.sum()), "cuts_per_pixel_median": float(np.median(cnt[ob]))}
            if a.inject_rms > 0:
                gt = np.log(truth.gain[k].numpy())[r0:r0 + nr, c0:c0 + nc]
                tp = projected(gt, ob, pg.boxes[k], pg.order)
                x, y = lgm[ob], tp[ob]
                nl2 = 1.0 / np.maximum(inf_[ob], 1e-30)
                out.update({"injected_rms_after_gauge": float(y.std()), "learned_rms": float(x.std()), "corr": float(np.corrcoef(x, y)[0, 1]),
                            "rms_diff": float(np.sqrt(((x - y) ** 2).mean())), "noise_limit_rms": float(np.sqrt(nl2.mean())),
                            "expected_corr_at_noise_limit": float(y.std() / math.sqrt(y.var() + nl2.mean())),
                            "slope_learned_on_injected": float((x * y).sum() / (y * y).sum())})
            else:
                nl2 = 1.0 / np.maximum(inf_[ob], 1e-30)
                out.update({"learned_rms": float(lgm[ob].std()), "noise_limit_rms": float(np.sqrt(nl2.mean()))})
            rows.append(out)
        res["gain_maps"] = rows
        res["gauge_track"] = {"columns": ["step", "max projected component (rms)", "max |logg| unobserved", "max |mean logg| observed", "max |logg|"],
                              "rows": gtrack, "max_projected": max(r[1] for r in gtrack), "max_mean": max(r[3] for r in gtrack)}
    st = model.state_dict()
    h = hashlib.sha256()
    for k in sorted(st):
        h.update(k.encode())
        h.update(st[k].detach().cpu().contiguous().numpy().tobytes())
    res["state_sha256"] = h.hexdigest()
    res["total_seconds"] = time.perf_counter() - T0
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("steps", "fit_seconds", "heldout", "loss_last5_logged", "total_seconds")}))
    for r in res.get("gain_maps", []):
        print(json.dumps(r))
    if "gauge_track" in res:
        print("gauge: max projected", res["gauge_track"]["max_projected"], "max |mean|", res["gauge_track"]["max_mean"])


if __name__ == "__main__":
    main()
