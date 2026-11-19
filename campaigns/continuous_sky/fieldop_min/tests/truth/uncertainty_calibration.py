"""Calibration of the sky uncertainty (fieldop/uncertainty.py) on the known-truth problem (truth_problem.py). Report: UNCERTAINTY_2026-11-18.md.

    python tests/truth/uncertainty_calibration.py kernel                         # the frozen true response, once (shared by every fit)
    python tests/truth/uncertainty_calibration.py fit --form items --noise-seed 0 [--fit-seed 0] [--half -1|0|1] [--n-train 8] [--steps 400]
    python tests/truth/uncertainty_calibration.py analyse --form items --noise-seed 0 [--n-train 8] [--samples 24]
    python tests/truth/uncertainty_calibration.py summarise --form items [--n-train 8]

States and analyses go to tests/truth/results/uncertainty/. Same configuration as the truth report (C2 grid 400 steps, C4 items 300 steps; response
frozen at the truth; noise model frozen; default zero-level rule --sky-zero median), except that the response is fitted to the truth ONCE and
shared (the report fitted it per run with the run's seed; the fitted kernels differ by about 0.7 per cent relative L1 from the truth either way).
Run with PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; one thread.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import truth_problem as tp  # noqa: E402
from fieldop.data import build_meta  # noqa: E402
from fieldop import uncertainty as U  # noqa: E402
from fieldop.train import fit, register_prepass  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(HERE, "results", "uncertainty")
KERNEL = os.path.join(OUT, "kernel_truth.pt")
STEPS = {"items": 300}
INNER = 1.5          # arcsec: region seen by every exposure


def tag(form, n, noise, fit_seed=0, half=-1):
    return f"{form}_n{n}_s{noise}" + (f"_f{fit_seed}" if fit_seed else "") + (f"_h{half}" if half >= 0 else "")


def field(n_train, noise_seed, half=-1):
    pop = tp.make_population()
    fl = tp.flux_for_snr(pop, n_ref=8)
    exps, fset, meta, truth, pf, _ = tp.build_field(n_train, seed=noise_seed, pop=pop, fluxes=fl)
    if half in (0, 1):                       # same rule as --train-half: training groups sorted by name, keep rank parity == half
        keep = U.alternate_groups([e.group for e in exps if e.split == "train"], half)
        exps = [e for e in exps if e.split != "train" or e.group in keep]
        for k, e in enumerate(exps):
            e.eid = k
        meta = build_meta(exps, ["A"], ["A"], ["A"])
    return exps, fset, meta, pop, fl, pf


def model_for(exps, fset, meta, form, steps, fit_seed, log):
    cfg = tp.base_config(form, steps=steps, seed=fit_seed)
    tp.KERNEL_STEPS = 0
    model, info = tp.build_model(exps, fset, meta, cfg, freeze_kernel=True, log=log)
    ks = torch.load(KERNEL, weights_only=False)
    model.net.load_state_dict(ks["net"])
    own = model.codes.state_dict()                      # per-exposure codes (z) have one row per exposure: keep the model's own when the count differs
    model.codes.load_state_dict({k: (v if v.shape == own[k].shape else own[k]) for k, v in ks["codes"].items()})
    return model, cfg


def cmd_kernel(a):
    os.makedirs(OUT, exist_ok=True)
    exps, fset, meta, *_ = field(8, 0)
    cfg = tp.base_config("items", steps=1, seed=0)
    tp.KERNEL_STEPS = 2500
    model, info = tp.build_model(exps, fset, meta, cfg, freeze_kernel=True)
    torch.save({"net": model.net.state_dict(), "codes": model.codes.state_dict(), "check": info["kernel_start"]}, KERNEL)


def cmd_fit(a):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    exps, fset, meta, pop, fl, pf = field(a.n_train, a.noise_seed, a.half)
    logs = []
    steps = a.steps or STEPS[a.form]
    model, cfg = model_for(exps, fset, meta, a.form, steps, a.fit_seed, logs.append)
    torch.manual_seed(a.fit_seed)
    register_prepass(model, exps, cfg, log=logs.append)
    rec = fit(model, exps, cfg, log=logs.append)
    t = tag(a.form, a.n_train, a.noise_seed, a.fit_seed, a.half)
    tr = [e.eid for e in exps if e.split == "train"]
    torch.save({"state": model.state_dict(), "form": a.form, "n_train": a.n_train, "noise_seed": a.noise_seed, "fit_seed": a.fit_seed, "half": a.half,
                "steps": steps, "fit_seconds": rec["seconds"], "bkg_train_mean": float(model.bkg[tr, 0].mean()),
                "train_groups": [e.group for e in exps if e.split == "train"], "loss_tail": [l for l in logs if l.startswith("step")][-3:]},
               os.path.join(OUT, f"state_{t}.pt"))
    print(json.dumps({"tag": t, "fit_seconds": rec["seconds"], "wall": time.perf_counter() - t0, "bkg_mean": float(model.bkg[tr, 0].mean()),
                      "tail": [l for l in logs if l.startswith("step")][-1:]}))


def load_model(form, n, noise, fit_seed=0, half=-1, log=print):
    st = torch.load(os.path.join(OUT, f"state_{tag(form, n, noise, fit_seed, half)}.pt"), weights_only=False)
    exps, fset, meta, pop, fl, pf = field(n, noise, half)
    model, cfg = model_for(exps, fset, meta, form, st["steps"], fit_seed, log)
    model.load_state_dict(st["state"])
    return model, cfg, exps, st


def cmd_analyse(a):
    """Posterior draws and the fitted coefficients for one full fit; saved for the summary (all scales are computed there)."""
    t0 = time.perf_counter()
    logs = []
    model, cfg, exps, st = load_model(a.form, a.n_train, a.noise_seed, fit_seed=a.fit_seed, half=a.half, log=logs.append)
    op = U.SkyNormalOperator(model, exps, cfg, rho=a.rho, log=print)
    pol = {}
    if a.polish:
        A_pol, db, info = U.gauss_newton_polish(op, tol=a.tol, maxiter=a.maxiter)
        pol = {"A_pol": A_pol.cpu(), "db_pol": db.cpu(), "polish_info": info}
        print(f"Gauss-Newton polish: {info['iterations']} iterations, relative residual {info['rel_residual']:.2e}", flush=True)
        if a.half >= 0 or a.fit_seed:
            torch.save(pol, os.path.join(OUT, f"pol_{tag(a.form, a.n_train, a.noise_seed, a.fit_seed, a.half)}.pt"))
            print(json.dumps({"polished": tag(a.form, a.n_train, a.noise_seed, a.fit_seed, a.half), "seconds": time.perf_counter() - t0}))
            return
    samp, recs = U.posterior_samples(op, a.samples, seed=1000 + 100 * a.noise_seed + a.part, tol=a.tol, maxiter=a.maxiter, log=print)
    phi = U.band_vector(model, 0).cpu()
    circ = {f"{s:.4f}": op.circulant_sigma(phi, s).float() for s in [f * model.sky.pitch for f in U.SCALES_DEFAULT if f > 0]}
    exact = []
    if a.exact:
        sm = U.Smoother(op.N, model.sky.pitch, a.exact_scale)
        c = (op.N - 1) // 2
        pts = [(c, c), (c + 7, c - 11), (c - 13, c + 5), (c + 20, c + 18)][:a.exact]
        ex = U.exact_variance(op, phi, sm, pts, tol=1e-4, maxiter=800)
        B = U.band_of(samp, phi).double()
        sg = U.Smoother(op.N, model.sky.pitch, a.exact_scale, dtype=torch.float64)(B).pow(2).mean(0).sqrt()
        exact = [{"ix": ix, "iy": iy, "sigma_exact": v, "iterations": it, "sigma_samples": float(sg[iy, ix])} for (ix, iy), (v, it) in zip(pts, ex)]
        print("exact vs samples:", exact)
    t = tag(a.form, a.n_train, a.noise_seed) + (f"_rho{a.rho:g}" if a.rho != 10 else "") + f"_part{a.part}"
    torch.save({"samples": samp.half() if a.half_precision else samp, "A": op.A_fit.cpu(), "phi": phi, "covered": op.covered.cpu(), "mu": op.mu, "rho": a.rho,
                "node_noise": op.node_noise, "cg": recs, "n_apply": op.n_apply, "seconds_per_apply": op.t_apply / op.n_apply, "bkg_train_mean": st["bkg_train_mean"],
                "n_pix": op.n_pix, "n_huber": op.n_huber, "exact": exact, "exact_scale": a.exact_scale, "pitch": model.sky.pitch, "circulant": circ,
                "cov_rel": (op.cov / op.cov[(op.N - 1) // 2, (op.N - 1) // 2]).cpu(),
                "seconds": time.perf_counter() - t0, **pol}, os.path.join(OUT, f"post_{t}.pt"))
    print(json.dumps({"tag": t, "seconds": time.perf_counter() - t0, "iterations": [r["iterations"] for r in recs], "apply_s": op.t_apply / op.n_apply,
                      "n_apply": op.n_apply}))


# ----------------------------------------------------------------------------- summary
def truth_product(pf, xs, ys, s):
    """True sky convolved with g_s at points (xs, ys) [arcsec]."""
    v = np.zeros(len(xs))
    for (x, y, sg, F) in pf:
        w2 = sg * sg + s * s
        v += F * np.exp(-0.5 * ((xs - x) ** 2 + (ys - y) ** 2) / w2) / (2 * math.pi * w2)
    return v


def point_values(img, xs, ys, pitch, s):
    """Product at arbitrary points: img (..., N, N) -> (..., n_points)."""
    N = img.shape[-1]
    xn = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * pitch
    Mx = U.tent_gauss_1d(torch.tensor(xs), xn, pitch, s)
    My = U.tent_gauss_1d(torch.tensor(ys), xn, pitch, s)
    return torch.einsum("pi,...ij,pj->...p", My, img.double(), Mx)


def stats(Z):
    """Z (n_real, n_points): mean and rms with errors from the realisation-to-realisation spread."""
    Z = np.asarray(Z, dtype=np.float64)
    R = Z.shape[0]
    m_r = Z.mean(1)
    rms_r = np.sqrt((Z ** 2).mean(1))
    sd_r = Z.std(1)
    e = (lambda v: float(v.std(ddof=1) / math.sqrt(R)) if R > 1 else float("nan"))
    return {"mean": float(Z.mean()), "mean_err": e(m_r), "rms": float(np.sqrt((Z ** 2).mean())), "rms_err": e(rms_r), "sd": float(Z.std()), "sd_err": e(sd_r),
            "n_points": int(Z.shape[1]), "n_real": R}


def smooth_var(var, inner, pitch, w=0.25):
    """Local average of a variance map over the covered region (Gaussian window w arcsec): reduces the sampling noise of sigma from few draws."""
    sm = U.Smoother(var.shape[-1], pitch, w, dtype=torch.float64)
    m = inner.double()
    return sm(var * m) / sm(m).clamp(min=1e-6)


def cmd_summarise(a):
    posts = sorted(glob.glob(os.path.join(OUT, f"post_{a.form}_n{a.n_train}_s[0-9]*_part*.pt")))
    posts = [p for p in posts if "_rho" not in p]
    pop = tp.make_population()
    fl = tp.flux_for_snr(pop, n_ref=8)
    pf = [(p[0], p[1], p[2], float(F)) for p, F in zip(pop, fl)]
    emp = tp.empty_positions(pop)
    snr = np.array([p[3] for p in pop]) * math.sqrt(a.n_train / 8)
    sx, sy = np.array([p[0] for p in pop]), np.array([p[1] for p in pop])
    by_seed = {}
    for p in posts:
        by_seed.setdefault(int(os.path.basename(p).split("_s")[1].split("_")[0]), []).append(torch.load(p, weights_only=False))
    seeds = sorted(by_seed)
    D = []
    for sd in seeds:
        parts = by_seed[sd]
        d = dict(parts[0])
        d["samples"] = torch.cat([q["samples"].float() for q in parts if q["samples"].numel()])
        d["cg"] = sum((q["cg"] for q in parts), [])
        d["exact"] = sum((q["exact"] for q in parts), [])
        d["A_pol"] = next((q["A_pol"] for q in parts if "A_pol" in q), None)
        d["polish_info"] = next((q["polish_info"] for q in parts if "polish_info" in q), None)
        D.append(d)
    pitch = D[0]["pitch"]
    N = D[0]["A"].shape[-1]
    phi = D[0]["phi"].double()
    xn = (np.arange(N) - (N - 1) / 2) * pitch
    inner = torch.tensor((np.abs(xn)[None, :] < INNER) & (np.abs(xn)[:, None] < INNER))
    pooled = torch.cat([d["samples"] for d in D])
    out = {"form": a.form, "n_train": a.n_train, "realisations": seeds, "draws_per_realisation": [int(d["samples"].shape[0]) for d in D],
           "pooled_draws": int(pooled.shape[0]), "cg_iterations": [[r["iterations"] for r in d["cg"]] for d in D],
           "cg_final_residual": [[round(r["rel_residual"], 5) for r in d["cg"]] for d in D],
           "seconds_per_apply": float(np.mean([d["seconds_per_apply"] for d in D])), "exact": [d["exact"] for d in D if d["exact"]],
           "polish": [d["polish_info"]["iterations"] if d["polish_info"] else None for d in D], "scales": {}}
    # supported resolution (pooled draws), sigma against scale, and its consistency between realisations
    scales = [f * pitch for f in U.SCALES_DEFAULT]
    sg = U.sigma_maps(pooled, phi.float(), pitch, scales)
    s_st, alpha = U.supported_resolution(sg, scales)
    v = s_st[inner].numpy()
    out["supported_resolution"] = {"median": float(np.nanmedian(v)), "p10": float(np.nanpercentile(v, 10)), "p90": float(np.nanpercentile(v, 90)),
                                   "frac_nan": float(np.isnan(v).mean())}
    sig_med = np.array([float(sg[i][inner].median()) for i in range(len(scales))])
    circ_med = [float(D[0]["circulant"][f"{sc:.4f}"][inner].median()) if f"{sc:.4f}" in D[0]["circulant"] else float("nan") for sc in scales]
    out["sigma_vs_scale_inner_median"] = {"scales_arcsec": scales, "sigma_draws": sig_med.tolist(), "sigma_circulant": circ_med,
                                          "alpha": (-(np.diff(np.log(sig_med[1:])) / np.diff(np.log(scales[1:])))).tolist()}
    s_prod = round(float(np.nanmedian(v)), 3)
    out["product_resolution"] = s_prod
    cons = []
    for d in D:
        a_own = U.sigma_maps(d["samples"], phi.float(), pitch, [s_prod])[0]
        cons.append(float(a_own[inner].pow(2).mean().sqrt() / sg_at(pooled, phi, pitch, s_prod)[inner].pow(2).mean().sqrt()))
    out["sigma_own_over_pooled"] = cons
    tn = tp.true_sky_nodes(pf, N, pitch)
    for s in sorted({s_prod, 0.1, 0.15, 0.2}):
        sm = U.Smoother(N, pitch, s, dtype=torch.float64)
        var = sm(U.band_of(pooled.double(), phi)).pow(2).mean(0)
        varS = smooth_var(var, inner, pitch)
        Mx_e = U.tent_gauss_1d(torch.tensor(emp[:, 0]), torch.tensor(xn), pitch, s)
        My_e = U.tent_gauss_1d(torch.tensor(emp[:, 1]), torch.tensor(xn), pitch, s)
        Mx_s = U.tent_gauss_1d(torch.tensor(sx), torch.tensor(xn), pitch, s)
        My_s = U.tent_gauss_1d(torch.tensor(sy), torch.tensor(xn), pitch, s)
        pv = lambda img, Mx, My: torch.einsum("pi,ij,pj->p", My, img, Mx).numpy()
        # sigma at a point: Var of the point value = sum over draws of (point value)^2 / n, smoothed like varS (approximation: read varS at the nearest node)
        def sig_at(xs, ys):
            ix = np.clip(np.round(xs / pitch + (N - 1) / 2).astype(int), 0, N - 1)
            iy = np.clip(np.round(ys / pitch + (N - 1) / 2).astype(int), 0, N - 1)
            return varS[iy, ix].sqrt().numpy()
        se, ss = sig_at(emp[:, 0], emp[:, 1]), sig_at(sx, sy)
        te, ts = truth_product(pf, emp[:, 0], emp[:, 1], s), truth_product(pf, sx, sy, s)
        tg = torch.tensor(truth_product(pf, *[g.reshape(-1) for g in np.meshgrid(xn, xn)], s).reshape(N, N))
        sgS = varS.sqrt()
        rec = {"sigma_inner_median": float(sgS[inner].median()), "representation_sources": None}
        rep = (pv(tn, Mx_s, My_s) - ts) / ss
        rec["representation_sources"] = [float(rep[(snr >= lo) & (snr < hi if hi < 100 else snr > 0)].mean()) for lo, hi in tp.SN_BINS]
        for which in ("fit", "polished"):
            if which == "polished" and any(d["A_pol"] is None for d in D):
                continue
            Ze, Zr, Zs, Zi, Zi_hp, Zh, ratios, Zemp_hd = [], [], [], [], [], [], [], []
            for d, seed in zip(D, seeds):
                A = (d["A"] if which == "fit" else d["A_pol"]).double()
                band = U.band_of(A, phi)
                db = d["bkg_train_mean"] - tp.BKG
                pe, ps = pv(band, Mx_e, My_e), pv(band, Mx_s, My_s)
                Ze.append((pe + db - te) / se)
                Zr.append((pe - te) / se)
                Zs.append((ps + db - ts) / ss)
                err = (sm(band) + db - tg)
                Zi.append((err / sgS)[inner].numpy())
                # the same with the best quadratic polynomial over the inner region removed (sky modes degenerate with the backgrounds)
                X, Y = np.meshgrid(xn, xn)
                B = np.stack([np.ones_like(X), X, Y, X * X, X * Y, Y * Y], -1)[inner.numpy()]
                e_in = err[inner].numpy()
                coef = np.linalg.lstsq(B, e_in, rcond=None)[0]
                Zi_hp.append((e_in - B @ coef) / sgS[inner].numpy())
                # half-difference estimator
                Ah = []
                for h in (0, 1):
                    fs = os.path.join(OUT, f"state_{tag(a.form, a.n_train, seed, 0, h)}.pt")
                    fp = os.path.join(OUT, f"pol_{tag(a.form, a.n_train, seed, 0, h)}.pt")
                    if which == "fit" and os.path.exists(fs):
                        Ah.append(coeffs_of_state(fs, a.form, a.n_train, seed, h))
                    elif which == "polished" and os.path.exists(fp):
                        Ah.append(torch.load(fp, weights_only=False)["A_pol"].double())
                if len(Ah) == 2:
                    hd = U.difference_map(Ah[0], Ah[1], phi, pitch, s, mode="halves", covered=inner)
                    hb = U.band_of(Ah[0], phi) - U.band_of(Ah[1], phi)
                    Zh.append(np.concatenate([pv(hb, Mx_e, My_e) / 2 / se, pv(hb, Mx_s, My_s) / 2 / ss]))
                    ratios.append(hd["sigma_global"] / float(sgS[inner].pow(2).mean().sqrt()))
                    Zemp_hd.append((pe + db - te) / hd["sigma_global"])
            r = {"empty_total": stats(Ze), "empty_raw": stats(Zr), "inner_total": stats(Zi), "inner_total_quadratic_removed": stats(Zi_hp), "sources": []}
            Zs = np.array(Zs)
            for lo, hi in tp.SN_BINS:
                sel = (snr >= lo) & (snr < hi) if hi < 100 else (snr >= lo)
                if sel.any():
                    r["sources"].append({"bin": [lo, hi], "n_sources": int(sel.sum()), **stats(Zs[:, sel])})
            if Zh:
                r["half_difference_pull"] = stats(Zh)
                r["sigma_halfdiff_over_analytic"] = {"values": ratios, "mean": float(np.mean(ratios)),
                                                     "err": float(np.std(ratios, ddof=1) / math.sqrt(len(ratios))) if len(ratios) > 1 else float("nan")}
                r["empty_total_with_halfdiff_sigma"] = stats(Zemp_hd)
            sc = []
            for d, seed in zip(D, seeds):
                fs = os.path.join(OUT, f"state_{tag(a.form, a.n_train, seed, 1)}.pt")
                fp = os.path.join(OUT, f"pol_{tag(a.form, a.n_train, seed, 1)}.pt")
                Ab = None
                if which == "fit" and os.path.exists(fs):
                    Ab = coeffs_of_state(fs, a.form, a.n_train, seed, -1, fit_seed=1)
                elif which == "polished" and os.path.exists(fp):
                    Ab = torch.load(fp, weights_only=False)["A_pol"].double()
                if Ab is not None:
                    A = (d["A"] if which == "fit" else d["A_pol"]).double()
                    dd = U.difference_map(A, Ab, phi, pitch, s, mode="seeds", covered=inner)
                    sc.append({"noise_seed": seed, "sigma_opt_over_analytic": dd["sigma_global"] / float(sgS[inner].pow(2).mean().sqrt())})
            r["optimisation_scatter"] = sc
            rec[which] = r
        out["scales"][f"{s:.3f}"] = rec
    p = os.path.join(OUT, f"summary_{a.form}_n{a.n_train}.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print_summary(out)


def sg_at(samples, phi, pitch, s):
    return U.Smoother(samples.shape[-1], pitch, s, dtype=torch.float64)(U.band_of(samples.double(), phi.double())).pow(2).mean(0).sqrt()


_CACHE = {}


def coeffs_of_state(path, form, n, noise, half, fit_seed=0):
    if path in _CACHE:
        return _CACHE[path]
    model, cfg, exps, st = load_model(form, n, noise, fit_seed=fit_seed, half=half, log=lambda s: None)
    with torch.no_grad():
        A = model.sky.coeffs().detach().double().cpu()
    _CACHE[path] = A
    return A


def print_summary(o):
    print(f"{o['form']} N={o['n_train']} realisations {o['realisations']} draws {o['draws_per_realisation']}; supported resolution {o['supported_resolution']}; "
          f"product {o['product_resolution']}; own/pooled sigma {np.round(o['sigma_own_over_pooled'], 2).tolist()}; polish iterations {o['polish']}")
    sv = o["sigma_vs_scale_inner_median"]
    print("scales", np.round(sv["scales_arcsec"], 4).tolist())
    print("sigma (draws)", np.round(sv["sigma_draws"], 4).tolist())
    print("sigma (circulant)", np.round(sv["sigma_circulant"], 4).tolist())
    print("alpha", np.round(sv["alpha"], 2).tolist())
    f = lambda d: f"{d['mean']:+.2f}+-{d['mean_err']:.2f} rms {d['rms']:.2f}+-{d['rms_err']:.2f}"
    for s, rec in o["scales"].items():
        print(f"== s = {s} arcsec: sigma inner median {rec['sigma_inner_median']:.4f}; representation (sources, by bin) {np.round(rec['representation_sources'], 2).tolist()}")
        for which in ("fit", "polished"):
            if which not in rec:
                continue
            r = rec[which]
            print(f"  [{which}] empty total {f(r['empty_total'])} | empty raw {f(r['empty_raw'])} | inner {f(r['inner_total'])} | inner, quadratic removed {f(r['inner_total_quadratic_removed'])}")
            print("     sources: " + "; ".join(f"{row['bin']} {f(row)}" for row in r["sources"]))
            if "half_difference_pull" in r:
                print(f"     half-difference: h/sigma {f(r['half_difference_pull'])}; sigma_hd/sigma {r['sigma_halfdiff_over_analytic']['mean']:.2f}+-{r['sigma_halfdiff_over_analytic']['err']:.2f}"
                      f" ({len(r['sigma_halfdiff_over_analytic']['values'])}); empty pulls with sigma_hd {f(r['empty_total_with_halfdiff_sigma'])}")
            if r["optimisation_scatter"]:
                print("     optimisation scatter / sigma:", [round(x["sigma_opt_over_analytic"], 3) for x in r["optimisation_scatter"]])


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("kernel").set_defaults(fn=cmd_kernel)
    f = sp.add_parser("fit")
    f.add_argument("--form", default="items", choices=["items"])
    f.add_argument("--n-train", type=int, default=8)
    f.add_argument("--noise-seed", type=int, default=0)
    f.add_argument("--fit-seed", type=int, default=0)
    f.add_argument("--half", type=int, default=-1)
    f.add_argument("--steps", type=int, default=0)
    f.set_defaults(fn=cmd_fit)
    g = sp.add_parser("analyse")
    g.add_argument("--form", default="items", choices=["items"])
    g.add_argument("--n-train", type=int, default=8)
    g.add_argument("--noise-seed", type=int, default=0)
    g.add_argument("--samples", type=int, default=2)
    g.add_argument("--part", type=int, default=0)
    g.add_argument("--polish", type=int, default=0)
    g.add_argument("--half", type=int, default=-1)
    g.add_argument("--fit-seed", type=int, default=0)
    g.add_argument("--rho", type=float, default=10.0)
    g.add_argument("--tol", type=float, default=1e-4)
    g.add_argument("--maxiter", type=int, default=60)
    g.add_argument("--exact", type=int, default=0)
    g.add_argument("--exact-scale", type=float, default=0.1)
    g.add_argument("--half-precision", type=int, default=0)
    g.set_defaults(fn=cmd_analyse)
    s = sp.add_parser("summarise")
    s.add_argument("--form", default="items", choices=["items"])
    s.add_argument("--n-train", type=int, default=8)
    s.add_argument("--pool", type=int, default=1)
    s.set_defaults(fn=cmd_summarise)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
