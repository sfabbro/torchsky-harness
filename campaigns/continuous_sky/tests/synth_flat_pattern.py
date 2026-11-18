"""Synthetic test of the count-rate detector terms (diagnosis of 17 Nov 2026): learned pixel gain with and without its learned shrinkage
prior (--pixel-gain-prior) and with the gain's gauge extended to the additive pattern's grid (--pixel-gain-gauge poly+pattern). CPU, one variant per call,
under 4 minutes.

Field = tests/synth_gain.py's (two instruments, three filters; here 9 exposures per filter: 6 train, 1 validation, 2 held out), with, injected
after tests' make_field (same noise realisation for every variant):
  * a per-pixel flat G = exp(g) per detector with a HEAVY TAIL like the official NIRCam flats: g = core N(0, --core) for every pixel plus, for
    a fraction --tail-frac of pixels, an extra N(0, --tail);
  * filter-dependent backgrounds (A1 x 0.5, A2 x 1.5, B1 x 1.0 of --bkg), so that G x background and an additive term can be told apart;
  * an additive detector pattern A(xi), the same for every filter of a detector (physical: residual dark): random node values of rms
    --add x sigma on the model's own --pattern-grid bilinear grid (so it is representable), with structure beyond degree 2;
  d = G (clean + background) + A + noise. --bkg is chosen LOW (default 8 sigma) so that the per-pixel noise limit of log G (about 0.04 with
6 training exposures per filter) is about twice the core width: the regime of the SMACS count-rate fits (noise limit 0.019 to 0.042 against
an official robust rms of 0.015 to 0.019).

    python tests/synth_flat_pattern.py --pixel-gain detector --pixel-gain-prior mix2 --pixel-gain-gauge poly+pattern --out r.json

Reports: held-out chi2 by bins of true signal-to-noise (signal = clean x G, without background), the recovered flat against the injected one
after the same gauge (correlation, rms of the difference, slope), the recovered additive pattern beyond degree 2 against the injected one, and
the learned prior widths.
"""
import argparse
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
sys.path.insert(0, HERE)
from synth_gain import projected  # noqa: E402

BINS = [(-1e9, 1), (1, 5), (5, 20), (20, 1e9)]
LAB = ["<1", "1-5", "5-20", ">20"]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixel-gain", default="detector")
    ap.add_argument("--pixel-gain-prior", default="off")
    ap.add_argument("--pixel-gain-gauge", default="poly")
    ap.add_argument("--pattern-grid", type=int, default=5)
    ap.add_argument("--core", type=float, default=0.02)
    ap.add_argument("--tail", type=float, default=0.15)
    ap.add_argument("--tail-frac", type=float, default=0.05)
    ap.add_argument("--add", type=float, default=0.5)
    ap.add_argument("--bkg", type=float, default=8.0)
    ap.add_argument("--lr-gain", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def poly_hp(v, order=2):
    n = v.shape[0]
    y, x = np.mgrid[0:n, 0:n] / (n - 1) * 2 - 1
    P = np.stack([x ** i * y ** (d - i) for d in range(order + 1) for i in range(d + 1)], -1).reshape(-1, (order + 1) * (order + 2) // 2)
    return (v.ravel() - P @ np.linalg.lstsq(P, v.ravel(), rcond=None)[0]).reshape(v.shape)


def main():
    a = parse()
    from fieldop.cli import default_config, make_sky
    from fieldop.data import build_meta
    from fieldop.model import FieldOperator
    from fieldop.synth import SynthInstrument, make_field
    from fieldop.train import fit, fit_heldout, predict_exposure, register_prepass, warm_kernel
    T0 = time.perf_counter()
    torch.manual_seed(a.seed)
    insts = [SynthInstrument("A", 0.10, 40, 6.5, 0.07, 1.5, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", 0.12, 34, 6.5, 0.10, 2.4, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("B1", 2.0, 2.8)])]
    exps, fset, _, truth, _ = make_field(seed=3 + a.seed, n_per_filter=9, n_val=1, n_held=2, insts=insts, shift_rms=0.03, centre_range=0.6,
                                         rot_choices=(0, 1, 2, 3), patch_px=20, truth_half=1.6, pix_gain_rms=0.0, bkg_mean=a.bkg)
    # ---- injection (own random stream; identical for every variant)
    grng = np.random.default_rng(777 + a.seed)
    fac = {"A1": 0.5, "A2": 1.5, "B1": 1.0}
    G, A = {}, {}
    for ii, ins in enumerate(insts):
        n = ins.npix
        g = grng.normal(0, a.core, (n, n)) + (grng.random((n, n)) < a.tail_frac) * grng.normal(0, a.tail, (n, n))
        G[ii] = torch.tensor(np.exp(g), dtype=torch.float32)
        # additive pattern inside the model's span: random node values on the --pattern-grid bilinear grid (noise = 1)
        from scipy.ndimage import map_coordinates
        Gp = max(a.pattern_grid, 2)
        nodes = grng.normal(0, a.add, (Gp, Gp))
        rr, cc = np.mgrid[0:n, 0:n] / (n - 1) * (Gp - 1)
        A[ii] = torch.tensor(map_coordinates(nodes, [rr, cc], order=1), dtype=torch.float32)
        truth.gain[ii] = G[ii]
    for e in exps:
        noise = e.d - (e.extra["clean"] + e.extra["bkg_true"])
        bk = e.extra["bkg_true"] * fac[fset.names[e.filt]]
        e.d = G[e.det] * (e.extra["clean"] + bk) + A[e.det] + noise
        e.bkg0 = float(e.d.median())
        e.extra.update(bkg_true=bk, gain_true=G[e.det], add_true=A[e.det])
    meta = build_meta(exps, ["A", "B"], ["A", "B"], ["A", "B"])
    kw = dict(sky_pitch=0.06, render_pitch=0.02, level_nh=12, kernel_extent=1.2, region_px=13, n_lambda=3, k_hidden=32,
              warm_kernel_steps=150, pattern_grid=a.pattern_grid, stripes=0, batch_patches=4, inner_sky=1, budget_min=1e6, steps=a.steps,
              tau_mode="fixed", val_every=2, held_steps=50, held_reg_steps=30, log_every=10, is_snr=10.0, reg_prepass_steps=40,
              reg_prepass_iters=1, seed=a.seed, pixel_gain=a.pixel_gain, lr_gain=a.lr_gain, pixel_gain_start=0.1,
              pixel_gain_prior=a.pixel_gain_prior, pixel_gain_gauge=a.pixel_gain_gauge,
              legacy_defaults=True, gauge_ref_inst=-1, kernel_norm_radius=0.0)      # old defaults: the diagnosis predates the defaults of 18 Nov 2026
    cfg = default_config(**kw)
    sky, N = make_sky(exps, fset, cfg, 2.8, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    pg = model.attach_pixel_gain(exps, cfg, fset_names=fset.names, det_names=["A", "B"])
    if pg is not None and a.pixel_gain_prior != "off":
        pg.set_prior(a.pixel_gain_prior)
    finfo = {}
    for f in range(len(fset)):
        ii = [i for i, ins in enumerate(insts) if any(fn == fset.names[f] for fn, _, _ in ins.filters)][0]
        finfo[f] = {"pixel_arcsec": insts[ii].pixel, "diameter_m": 6.5, "inst": ii}
    logs = []
    lg = lambda s: logs.append(s)  # noqa: E731
    warm_kernel(model, cfg, finfo, log=lg)
    register_prepass(model, exps, cfg, log=lg)
    rec = fit(model, exps, cfg, log=lg)
    held = [e.eid for e in exps if e.split == "held"]
    fit_heldout(model, exps, held, cfg, log=lg)
    P = {k: v.detach() for k, v in model.prepare().items()}
    acc = {k: [0.0, 0] for k in LAB + ["all", "truth_all", "faint", "truth_faint"]}
    with torch.no_grad():
        for e in exps:
            if e.split != "held":
                continue
            m, pv, S = predict_exposure(model, e, P, cfg, tile=20)
            v = e.valid & pv & torch.isfinite(m)
            r2 = ((e.d - m) / e.sig) ** 2
            t2 = ((e.d - (e.extra["gain_true"] * (e.extra["clean"] + e.extra["bkg_true"]) + e.extra["add_true"])) / e.sig) ** 2
            snr = e.extra["gain_true"] * e.extra["clean"] / e.sig
            for lab, (lo, hi) in zip(LAB, BINS):
                mk = v & (snr >= lo) & (snr < hi)
                acc[lab][0] += float(r2[mk].sum()); acc[lab][1] += int(mk.sum())
            faint = v & (snr < 0.5)
            for key, mk, arr in (("all", v, r2), ("truth_all", v, t2), ("faint", faint, r2), ("truth_faint", faint, t2)):
                acc[key][0] += float(arr[mk].sum()); acc[key][1] += int(mk.sum())
    res = {"args": vars(a), "steps": rec["n_steps"], "fit_seconds": rec["seconds"], "loss_last5_logged": [r[2] for r in rec["steps"][-5:]],
           "heldout_chi2": {k: v[0] / max(v[1], 1) for k, v in acc.items()}, "heldout_n": {k: v[1] for k, v in acc.items()},
           "pixel_gain_prior": rec.get("pixel_gain_prior")}
    if pg is not None:
        rows = []
        for k, lab in enumerate(pg.labels):
            lgm, ob, cnt, inf_ = pg.map(k)
            r0, c0, nr, nc = pg.boxes[k]
            gt = np.log(truth.gain[k].numpy())[r0:r0 + nr, c0:c0 + nc]
            tp = projected(gt, ob, pg.boxes[k], pg.order)
            x, y = lgm[ob], tp[ob]
            nl2 = 1.0 / np.maximum(inf_[ob], 1e-30)
            rows.append({"label": lab, "injected_rms_after_gauge": float(y.std()), "learned_rms": float(x.std()), "corr": float(np.corrcoef(x, y)[0, 1]),
                         "rms_diff": float(np.sqrt(((x - y) ** 2).mean())), "rms_injected_alone": float(np.sqrt((y ** 2).mean())),
                         "noise_limit_rms": float(np.sqrt(nl2.mean())), "slope_learned_on_injected": float((x * y).sum() / (y * y).sum()),
                         "tail_pixels_rms_diff": float(np.sqrt(((x - y)[np.abs(y) > 4 * a.core] ** 2).mean()))})
        res["gain_maps"] = rows
    if model.pattern is not None:
        from scipy.ndimage import map_coordinates
        pr = {}
        for ii, ins in enumerate(insts):
            n = ins.npix
            p = model.pattern[ii].detach().numpy().astype(np.float64)
            p = p - p.mean()
            Gp = p.shape[0]
            rr, cc = np.mgrid[0:n, 0:n] / (n - 1) * (Gp - 1)
            pm = map_coordinates(p, [rr, cc], order=1)
            ph, th = poly_hp(pm), poly_hp(A[ii].numpy().astype(np.float64))
            pr[ins.name] = {"corr_beyond_deg2": float(np.corrcoef(ph.ravel(), th.ravel())[0, 1]), "learned_rms_beyond_deg2": float(ph.std()),
                            "injected_rms_beyond_deg2": float(th.std()), "learned_rms_total": float(pm.std())}
        res["pattern"] = pr
    res["total_seconds"] = time.perf_counter() - T0
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("steps", "fit_seconds", "heldout_chi2", "total_seconds", "pixel_gain_prior")}))
    for r in res.get("gain_maps", []):
        print(json.dumps(r))
    print(json.dumps(res.get("pattern")))


if __name__ == "__main__":
    main()
