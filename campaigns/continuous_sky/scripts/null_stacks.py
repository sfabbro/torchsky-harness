#!/usr/bin/env python
"""Residual null stacks (T7) from the pixels dumped by smacs_four_v38.py --dump-heldout-pixels.

    python scripts/null_stacks.py RUN_DIR [--method explicit_lib_3d] [--out DIR] [--splits test,val] [--clip 6] [--min-n 30]

Reads RUN_DIR/heldout_pixels/<method>/*.npz and writes PNGs and null_summary.json into RUN_DIR/null_stacks/<method>/ (or --out).
Whitened residual z = (data - model) / sigma, sigma = the pipeline ERR as loaded (times noise_scale if --calibrate-noise was used).

(a) mean and rms of z on a 16 x 16 grid of detector position (dx, dy in [-1, 1]), per detector                    null_a_position.png
(b) mean z versus detector row and versus column (--rowcol-bin px bins), per detector                              null_b_rowcol.png
(c) per-exposure mean and rms z versus roll (Jacobian rotation of the exposure)                                    null_c_roll.png
(d) per-exposure mean and rms z versus MJD                                                                         null_d_mjd.png
(e) fractional residual (data - model) / model_sky versus model S/N = model_sky / sigma (log bins): mean, rms, and the
    model-error floor f from rms^2 = a / SN^2 + f^2 (weighted least squares; a ~ 1 if sigma is right)               null_e_fractional.png
(f) residual stacked around local maxima of the MODEL image (model_sky / sigma > --peak-snr, 5 x 5 maximum, full stamp inside
    the loaded pixels; catalogue-free): radial profile (detector px from the model centroid) and the detector-frame stamp
    split in 4 x 4 bins of the sub-pixel phase of the model centroid relative to the detector pixel grid            null_f_peaks_radial.png, null_f_peaks_phase.png

Null statistic of each binned map: chi2_unit = sum_b n_b mean_b^2 over bins with n_b >= min_n (assumes unit-variance independent z; expectation =
number of bins used = dof); chi2_scatter uses each bin's own variance instead of 1 (robust to a wrong sigma; same expectation); perm_mean / perm_p95 /
perm_p = the chi2_unit distribution when z is shuffled over pixels (n_perm shuffles of at most 3e6 pixels, bin membership fixed): the null "z
does not depend on this coordinate" that keeps the real z distribution. APPROXIMATIONS: pixels are treated as independent (resampling, correlated noise
and model errors correlated over many pixels inflate chi2 above its expectation even for a perfect operator); z is clipped to |z| <= clip for (a)-(d)
(cosmic rays, bad pixels; (e) and (f) are not clipped); the exposure-level tests (c), (d) have as many points as exposures, so their expectation is small.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

INK, MUTED, ACC = "#2b2b2b", "#8a8a8a", "#2a6f97"


def load_run(run_dir, method=None, splits=None):
    base = os.path.join(run_dir, "heldout_pixels")
    methods = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    if not methods:
        raise SystemExit(f"{base}: no model directory")
    method = method or methods[0]
    exps = []
    for f in sorted(glob.glob(os.path.join(base, method, "*.npz"))):
        z = np.load(f, allow_pickle=False)
        e = {k: z[k] for k in z.files}
        for k in ("name", "band", "instrument", "det", "split", "method"):
            if k in e:
                e[k] = str(e[k])
        if splits and e["split"] not in splits:
            continue
        exps.append(e)
    return method, exps


def whiten(e):
    return (e["data"] - e["model"]) / e["sigma"]


def bin_stats(idx, z, nb):
    n = np.bincount(idx, minlength=nb).astype(float)
    s = np.bincount(idx, weights=z, minlength=nb)
    s2 = np.bincount(idx, weights=z * z, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = s2 / n - mean ** 2
    return n, mean, var


def null_stat(idx, z, nb, min_n=30, n_perm=200, seed=0):
    """chi2 of the binned means against zero, its expectation and a shuffle null (see module docstring)."""
    n, mean, var = bin_stats(idx, z, nb)
    use = n >= min_n
    k = int(use.sum())
    out = {"n_pixels": int(len(z)), "n_bins_used": k, "dof": k}
    if k == 0:
        return dict(out, chi2_unit=None, chi2_over_dof=None, chi2_scatter=None, perm_mean=None, perm_p95=None, perm_p=None)
    chi = float((n[use] * mean[use] ** 2).sum())
    chi_s = float((n[use] * mean[use] ** 2 / np.maximum(var[use], 1e-12)).sum())
    rng = np.random.default_rng(seed)
    if len(z) > 3_000_000:
        sel = rng.choice(len(z), 3_000_000, replace=False)
        idx_p, z_p = idx[sel], z[sel]
    else:
        idx_p, z_p = idx, z
    nc, _, _ = bin_stats(idx_p, z_p, nb)
    usep = nc >= min_n
    perm = []
    for _ in range(n_perm):
        zs = rng.permutation(z_p)
        s = np.bincount(idx_p, weights=zs, minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            perm.append(float(((s[usep] ** 2) / nc[usep]).sum()))
    perm = np.asarray(perm)
    return dict(out, chi2_unit=chi, chi2_over_dof=chi / k, chi2_scatter=chi_s, perm_mean=float(perm.mean()), perm_p95=float(np.quantile(perm, 0.95)),
                perm_p=float((1 + (perm >= chi * len(z_p) / max(len(z), 1)).sum()) / (1 + len(perm))), expectation=k,
                note="perm_p compares chi2_unit rescaled to the shuffled sample size")


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "font.size": 8, "figure.dpi": 110,
                         "savefig.dpi": 110})
    return plt


def by_detector(exps, clip):
    out = {}
    for e in exps:
        z = whiten(e)
        ok = np.isfinite(z) & (np.abs(z) <= clip)
        d = out.setdefault(e["det"], {k: [] for k in ("z", "dx", "dy", "row", "col")})
        d["z"].append(z[ok])
        for k in ("dx", "dy", "row", "col"):
            d[k].append(e[k][ok])
    return {k: {kk: np.concatenate(v) for kk, v in d.items()} for k, d in out.items()}


def test_a(exps, outdir, plt, clip, min_n, summ):
    dets = by_detector(exps, clip)
    names = sorted(dets)
    fig, ax = plt.subplots(len(names), 2, figsize=(5.6, 2.5 * len(names)), squeeze=False)
    for i, dn in enumerate(names):
        d = dets[dn]
        ix = np.clip(((d["dx"] + 1) / 2 * 16).astype(int), 0, 15)
        iy = np.clip(((d["dy"] + 1) / 2 * 16).astype(int), 0, 15)
        idx = iy * 16 + ix
        n, mean, var = bin_stats(idx, d["z"], 256)
        summ["a_position"][dn] = null_stat(idx, d["z"], 256, min_n)
        summ["a_position"][dn].update(rms_z_all=float(np.sqrt(np.mean(d["z"] ** 2))), mean_z_all=float(np.mean(d["z"])))
        mm = np.where(n >= min_n, mean, np.nan).reshape(16, 16)
        rr = np.where(n >= min_n, np.sqrt(np.maximum(var + mean ** 2, 0)), np.nan).reshape(16, 16)
        v = max(0.05, float(np.nanmax(np.abs(mm))) if np.isfinite(mm).any() else 0.05)
        im = ax[i, 0].imshow(mm, origin="lower", extent=[-1, 1, -1, 1], cmap="RdBu_r", vmin=-v, vmax=v)
        ax[i, 0].set_title(f"{dn}: mean z", fontsize=8)
        fig.colorbar(im, ax=ax[i, 0], shrink=0.8)
        im = ax[i, 1].imshow(rr, origin="lower", extent=[-1, 1, -1, 1], cmap="Blues", vmin=0.0)
        ax[i, 1].set_title(f"{dn}: rms z", fontsize=8)
        fig.colorbar(im, ax=ax[i, 1], shrink=0.8)
        for a_ in ax[i]:
            a_.set_xlabel("dx"); a_.set_ylabel("dy"); a_.grid(False)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "null_a_position.png")); plt.close(fig)


def test_b(exps, outdir, plt, clip, min_n, binpx, summ):
    dets = by_detector(exps, clip)
    names = sorted(dets)
    fig, ax = plt.subplots(len(names), 2, figsize=(7, 1.9 * len(names)), squeeze=False)
    for i, dn in enumerate(names):
        d = dets[dn]
        for j, key in enumerate(("row", "col")):
            idx = (d[key] // binpx).astype(int)
            nb = int(idx.max()) + 1
            n, mean, var = bin_stats(idx, d["z"], nb)
            summ["b_rowcol"].setdefault(dn, {})[key] = null_stat(idx, d["z"], nb, min_n)
            x = (np.arange(nb) + 0.5) * binpx
            ok = n >= min_n
            err = 1 / np.sqrt(np.maximum(n, 1))
            ax[i, j].axhline(0, color=MUTED, lw=0.8)
            ax[i, j].fill_between(x[ok], -err[ok], err[ok], color=ACC, alpha=0.15, lw=0)
            ax[i, j].plot(x[ok], mean[ok], color=ACC, lw=1.0)
            ax[i, j].set_title(f"{dn}: mean z vs {key} (band +-1 sigma of the mean)", fontsize=7)
            ax[i, j].set_xlabel(f"{key} [detector px]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "null_b_rowcol.png")); plt.close(fig)


def per_exposure(exps, clip):
    rows = []
    for e in exps:
        z = whiten(e)
        z = z[np.isfinite(z) & (np.abs(z) <= clip)]
        rows.append({"name": e["name"], "band": e["band"], "instrument": e["instrument"], "roll": float(e["roll_deg"]), "mjd": float(e["mjd"]),
                     "n": int(len(z)), "mean": float(z.mean()) if len(z) else float("nan"), "rms": float(np.sqrt(np.mean(z ** 2))) if len(z) else float("nan")})
    return rows


def exposure_level(rows, key):
    ok = [r for r in rows if np.isfinite(r[key]) and r["n"] > 0]
    if not ok:
        return {"n_exposures": 0}
    chi = float(sum(r["n"] * r["mean"] ** 2 for r in ok))
    return {"n_exposures": len(ok), "dof": len(ok), "chi2_unit": chi, "chi2_over_dof": chi / len(ok), "expectation": len(ok),
            "mean_of_means": float(np.mean([r["mean"] for r in ok])), "rms_of_rms": float(np.sqrt(np.mean([r["rms"] ** 2 for r in ok])))}


def test_cd(exps, outdir, plt, clip, summ):
    rows = per_exposure(exps, clip)
    insts = sorted({r["instrument"] for r in rows})
    for key, tag, xl in (("roll", "c_roll", "roll [deg]"), ("mjd", "d_mjd", "MJD")):
        summ[tag] = exposure_level(rows, key)
        fig, ax = plt.subplots(2, 1, figsize=(5.2, 4.2), sharex=True)
        mk = dict(zip(insts, "osD^v<>"))
        for ins in insts:
            rs = [r for r in rows if r["instrument"] == ins and np.isfinite(r[key])]
            x = [r[key] for r in rs]
            ax[0].scatter(x, [r["mean"] for r in rs], s=14, marker=mk[ins], color=ACC, alpha=0.8, label=ins)
            ax[1].scatter(x, [r["rms"] for r in rs], s=14, marker=mk[ins], color=ACC, alpha=0.8)
        ax[0].axhline(0, color=MUTED, lw=0.8); ax[1].axhline(1, color=MUTED, lw=0.8)
        ax[0].set_ylabel("mean z"); ax[1].set_ylabel("rms z"); ax[1].set_xlabel(xl)
        if len(insts) > 1:
            ax[0].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"null_{tag}.png")); plt.close(fig)
    summ["per_exposure"] = rows


def test_e(exps, outdir, plt, min_n, summ):
    sn_all, fr_all, ins_all = [], [], []
    for e in exps:
        ms = e["model_sky"]
        ok = np.isfinite(ms) & (ms > 0) & np.isfinite(e["data"])
        sn_all.append((ms / e["sigma"])[ok])
        with np.errstate(divide="ignore", invalid="ignore"):
            fr_all.append(((e["data"] - e["model"]) / ms)[ok])
        ins_all.append(np.full(int(ok.sum()), e["instrument"]))
    sn, fr, ins = np.concatenate(sn_all), np.concatenate(fr_all), np.concatenate(ins_all)
    edges = np.logspace(0, 3, 13)
    fig, ax = plt.subplots(2, 1, figsize=(5.2, 4.6), sharex=True)
    for tag, sel in [("all", np.ones(len(sn), bool))] + [(i, ins == i) for i in sorted(set(ins))]:
        s, f = sn[sel], fr[sel]
        idx = np.digitize(s, edges) - 1
        inr = (idx >= 0) & (idx < len(edges) - 1)
        s, f, idx = s[inr], f[inr], idx[inr]
        nb = len(edges) - 1
        n, mean, var = bin_stats(idx, f, nb)
        rms2 = var + mean ** 2
        x = np.bincount(idx, weights=1 / s ** 2, minlength=nb) / np.maximum(n, 1)
        ok = n >= min_n
        res = {"edges": edges.tolist(), "n": n.astype(int).tolist(), "mean_frac": [None if not o else float(m) for o, m in zip(ok, mean)],
               "rms_frac": [None if not o else float(math.sqrt(v)) for o, v in zip(ok, rms2)]}
        if ok.sum() >= 3:
            w = n[ok]
            A = np.stack([x[ok], np.ones(ok.sum())], 1)
            coef = np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * rms2[ok]))
            res.update(noise_factor_a=float(coef[0]), floor_f2=float(coef[1]), floor_f=float(math.sqrt(max(coef[1], 0.0))))
        err = np.sqrt(np.maximum(rms2, 1e-30) / np.maximum(n, 1))
        use = ok
        res["bias_chi2"] = float((mean[use] ** 2 / err[use] ** 2).sum()) if use.any() else None
        res["bias_dof"] = int(use.sum())
        summ["e_fractional"][tag] = res
        xc = np.sqrt(edges[:-1] * edges[1:])
        if tag == "all":
            ax[0].errorbar(xc[ok], mean[ok], err[ok], color=ACC, lw=1.0, marker="o", ms=3)
            ax[1].plot(xc[ok], np.sqrt(rms2[ok]), color=ACC, marker="o", ms=3, lw=1.0, label="rms of (d-m)/m")
            if "floor_f" in res:
                ax[1].plot(xc, np.sqrt(np.maximum(res["noise_factor_a"], 0) / xc ** 2 + max(res["floor_f2"], 0.0)), color=MUTED, ls="--", lw=1.0,
                           label=f"fit: a/SN^2 + f^2, f = {res['floor_f']:.3f}, a = {res['noise_factor_a']:.2f}")
        else:
            ax[1].plot(xc[ok], np.sqrt(rms2[ok]), lw=0.7, alpha=0.6, label=tag)
    ax[0].axhline(0, color=MUTED, lw=0.8)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("model S/N (model_sky / sigma)")
    ax[0].set_ylabel("mean (d-m)/m"); ax[1].set_ylabel("rms (d-m)/m"); ax[1].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "null_e_fractional.png")); plt.close(fig)


def find_peaks(e, snr, R):
    """(row, col, centroid dy, dx) of local maxima of model_sky / sigma > snr whose (2R+1)^2 stamp lies inside the loaded pixels."""
    from scipy.ndimage import maximum_filter, minimum_filter
    r0, c0 = int(e["row"].min()), int(e["col"].min())
    H, W = int(e["row"].max()) - r0 + 1, int(e["col"].max()) - c0 + 1
    rr, cc = e["row"] - r0, e["col"] - c0
    ms = np.full((H, W), np.nan, np.float32); ms[rr, cc] = e["model_sky"]
    sg = np.full((H, W), np.nan, np.float32); sg[rr, cc] = e["sigma"]
    res = np.full((H, W), np.nan, np.float32); res[rr, cc] = e["data"] - e["model"]
    valid = np.isfinite(ms) & np.isfinite(res)
    sn = np.where(valid, ms / sg, -np.inf)
    mx = maximum_filter(sn, size=5, mode="constant", cval=-np.inf)
    inner = minimum_filter(valid, size=2 * R + 1, mode="constant", cval=False)
    pk = (sn == mx) & (sn > snr) & inner
    py, px = np.nonzero(pk)
    out = []
    for y, x in zip(py, px):
        cut = np.clip(ms[y - 1:y + 2, x - 1:x + 2], 0, None)
        t = cut.sum()
        if t <= 0:
            continue
        gy, gx = np.mgrid[-1:2, -1:2]
        out.append((y, x, float((cut * gy).sum() / t), float((cut * gx).sum() / t)))
    return ms, sg, res, out


def test_f(exps, outdir, plt, snr, R, summ):
    nr = 2 * R
    redges = np.arange(0, R + 0.01, 0.5)
    rad = {"n": np.zeros(len(redges) - 1), "sf": np.zeros(len(redges) - 1), "sz": np.zeros(len(redges) - 1), "sz2": np.zeros(len(redges) - 1)}
    ph_z = np.zeros((4, 4, nr + 1, nr + 1)); ph_f = np.zeros_like(ph_z); ph_n = np.zeros((4, 4))
    npk = 0
    gy, gx = np.mgrid[-R:R + 1, -R:R + 1]
    for e in exps:
        ms, sg, res, pks = find_peaks(e, snr, R)
        for y, x, cy, cx in pks:
            sl = (slice(y - R, y + R + 1), slice(x - R, x + R + 1))
            peak = ms[y, x]
            z = res[sl] / sg[sl]
            f = res[sl] / peak
            r = np.hypot(gx - cx, gy - cy)
            k = np.digitize(r.ravel(), redges) - 1
            ok = (k >= 0) & (k < len(redges) - 1)
            np.add.at(rad["n"], k[ok], 1)
            np.add.at(rad["sf"], k[ok], f.ravel()[ok])
            np.add.at(rad["sz"], k[ok], z.ravel()[ok])
            np.add.at(rad["sz2"], k[ok], z.ravel()[ok] ** 2)
            px_, py_ = min(int((cx + 0.5) * 4), 3), min(int((cy + 0.5) * 4), 3)
            ph_z[py_, px_] += z; ph_f[py_, px_] += f; ph_n[py_, px_] += 1
            npk += 1
    res_ = {"n_peaks": npk, "peak_snr": snr, "stamp_half_width_px": R}
    summ["f_peaks"] = res_
    if npk == 0:
        res_["note"] = "no model peak above the S/N threshold with a full stamp"
        return
    n = np.maximum(rad["n"], 1)
    mz, mf = rad["sz"] / n, rad["sf"] / n
    chi = float((rad["n"] * mz ** 2).sum()); k_ = int((rad["n"] > 0).sum())
    res_["radial"] = {"edges_px": redges.tolist(), "mean_z": mz.tolist(), "mean_frac": mf.tolist(), "rms_z": np.sqrt(rad["sz2"] / n).tolist(),
                      "n_pixels": rad["n"].astype(int).tolist(), "chi2_unit": chi, "dof": k_, "chi2_over_dof": chi / max(k_, 1), "expectation": k_}
    fig, ax = plt.subplots(2, 1, figsize=(4.8, 4.2), sharex=True)
    xc = 0.5 * (redges[1:] + redges[:-1])
    ax[0].axhline(0, color=MUTED, lw=0.8); ax[0].plot(xc, mf, color=ACC, marker="o", ms=3, lw=1.0); ax[0].set_ylabel("mean (d-m)/peak model")
    ax[1].axhline(0, color=MUTED, lw=0.8); ax[1].plot(xc, mz, color=ACC, marker="o", ms=3, lw=1.0); ax[1].set_ylabel("mean z"); ax[1].set_xlabel("radius from model centroid [detector px]")
    ax[0].set_title(f"{npk} model peaks, S/N > {snr:g}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "null_f_peaks_radial.png")); plt.close(fig)
    # detector-frame stamps by sub-pixel phase
    phase = {}
    fig, ax = plt.subplots(4, 4, figsize=(6.4, 6.4), squeeze=False)
    v = 0.0
    maps = {}
    for j in range(4):
        for i in range(4):
            if ph_n[j, i] >= 1:
                maps[(j, i)] = ph_z[j, i] / ph_n[j, i]
                v = max(v, float(np.abs(maps[(j, i)]).max()))
    v = max(v, 0.1)
    for j in range(4):
        for i in range(4):
            a_ = ax[3 - j, i]
            a_.grid(False); a_.set_xticks([]); a_.set_yticks([])
            if (j, i) in maps:
                a_.imshow(maps[(j, i)], origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
                chi = float(ph_n[j, i] * (maps[(j, i)] ** 2).sum())
                phase[f"y{j}_x{i}"] = {"n_peaks": int(ph_n[j, i]), "chi2_unit": chi, "dof": (nr + 1) ** 2, "chi2_over_dof": chi / (nr + 1) ** 2,
                                       "mean_frac_centre": float((ph_f[j, i] / ph_n[j, i])[R, R])}
                a_.set_title(f"n={int(ph_n[j, i])}", fontsize=6)
    fig.suptitle("mean z, detector frame, by phase of the model centroid (rows: y phase, columns: x phase)", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "null_f_peaks_phase.png")); plt.close(fig)
    ok = [v_ for v_ in phase.values()]
    res_["phase"] = phase
    res_["phase_total"] = {"chi2_unit": float(sum(p["chi2_unit"] for p in ok)), "dof": int(sum(p["dof"] for p in ok)), "expectation": int(sum(p["dof"] for p in ok)),
                           "note": "pixels of one stamp and neighbouring pixels are correlated; chi2 above dof is expected to be inflated by that"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--method")
    ap.add_argument("--out")
    ap.add_argument("--splits", default="test,val", help="comma list of splits to use (test, val, train); 'all' = every dumped exposure")
    ap.add_argument("--clip", type=float, default=6.0)
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--rowcol-bin", type=int, default=16)
    ap.add_argument("--peak-snr", type=float, default=50.0)
    ap.add_argument("--peak-radius", type=int, default=5)
    a = ap.parse_args(argv)
    splits = None if a.splits == "all" else set(a.splits.split(","))
    method, exps = load_run(a.run_dir, a.method, splits)
    if not exps:
        raise SystemExit(f"no dumped exposure in splits {a.splits} (with --score-all the dump is of the training exposures: use --splits all or train)")
    outdir = a.out or os.path.join(a.run_dir, "null_stacks", method)
    os.makedirs(outdir, exist_ok=True)
    plt = setup_mpl()
    summ = {"run_dir": a.run_dir, "method": method, "splits": sorted(splits) if splits else "all", "n_exposures": len(exps),
            "n_pixels": int(sum(len(e["data"]) for e in exps)), "clip": a.clip, "min_n": a.min_n,
            "a_position": {}, "b_rowcol": {}, "e_fractional": {}, "caveat": "pixels treated as independent; see the module docstring"}
    test_a(exps, outdir, plt, a.clip, a.min_n, summ)
    test_b(exps, outdir, plt, a.clip, a.min_n, a.rowcol_bin, summ)
    test_cd(exps, outdir, plt, a.clip, summ)
    test_e(exps, outdir, plt, a.min_n, summ)
    test_f(exps, outdir, plt, a.peak_snr, a.peak_radius, summ)
    json.dump(summ, open(os.path.join(outdir, "null_summary.json"), "w"), indent=1, default=float)
    print(f"null stacks: {len(exps)} exposures, {summ['n_pixels']} pixels -> {outdir}")
    for d, v in summ["a_position"].items():
        print(f"  (a) {d}: chi2/dof {v['chi2_over_dof']}  (dof {v['dof']}; shuffle p95 {v['perm_p95']})")
    return summ


if __name__ == "__main__":
    main()
