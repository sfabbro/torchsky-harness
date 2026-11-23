"""Diagnostics on the held-out JWST exposures of SMACS, on the common pixels: metrics other than chi2/N, and residual images.

    python diag.py            -> diag.json, diag_*.png in this folder

Per held-out pixel: data d, prediction m of a method, protocol noise sigma (which includes the Poisson noise of the source: checked below),
signal-to-noise of the official mosaic snr_ref. Metrics, each named where it is reported:
  chi2/N                      mean of ((d - m) / sigma)^2
  robust pull width           1.4826 x median absolute deviation of (d - m) / sigma; 1 for a perfect model with a correct sigma
  tail fraction               share of pixels with |d - m| > 5 sigma; 5.7e-7 for a unit Gaussian
  fractional residual         robust width of (d - m) / S, S = snr_ref x sigma the signal above the background as the official mosaic gives it
  noise-only fractional       what the fractional residual of a perfect model would be: robust width of sigma / S ~ 1 / snr_ref
  flux ratio on sources       sum(m - b_m) / sum(d - b_d) over pixels with snr_ref >= 20, b = the exposure's median over the faintest fifth of its pixels
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = os.environ["ONESKY_SCRATCH"]  # folder holding canfar/ (predictions, masks, xpred) and summary/ (outputs)
C, OUT = S + "/canfar", S + "/summary"
METHODS = {"pipeline": "pred_stage3_blot", "base0": "pred_lib_base_s0", "direct0": "pred_lib_direct_s0", "ks_direct0": "pred_ks_direct_s0",
           "ks_direct0_400": "pred_ks_direct_s0_400m", "ssf4_200": "pred_ssf4_200m", "ks_nofloor": "pred_ks_direct_nofloor", "ks_flat": "pred_ks_direct_flatnoise"}
METHODS = {k: v for k, v in METHODS.items() if os.path.isdir(f"{C}/{v}")}
BINS = [("1-5", 1, 5), ("5-20", 5, 20), ("20-50", 20, 50), ("50-200", 50, 200), (">200", 200, np.inf)]
mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else float("nan")

rows = []
for f in sorted(glob.glob(f"{C}/xpred_lat_base_s0/*.npz")):
    root = os.path.basename(f)[:-4]
    x, m = np.load(f), np.load(f"{C}/heldout/{root}.npz")
    y0, x0 = int(x["y0"]), int(x["x0"])
    ny, nx = x["data"].shape
    sl = (slice(y0, y0 + ny), slice(x0, x0 + nx))
    d, err = x["data"].astype(np.float64), x["err"].astype(np.float64)
    sig, snr, nf = m["sigma"][sl].astype(np.float64), m["snr_ref"][sl].astype(np.float64), m["noise_factor"][sl].astype(np.float64)
    ok = m["valid"][sl] & x["valid"].astype(bool) & (sig > 0) & np.isfinite(snr) & np.isfinite(d)
    mod = {}
    for k, folder in METHODS.items():
        z = np.load(f"{C}/{folder}/{root}.npz")
        mod[k] = z["model"][sl].astype(np.float64)
        ok &= np.isfinite(mod[k]) & (z["model_valid"][sl] if "model_valid" in z.files else True)
    rows.append(dict(root=root, band=str(x["band"]), d=d, err=err, sig=sig, snr=snr, nf=nf, ok=ok, mod=mod))
print(len(rows), "held-out exposures; methods", list(METHODS))

# ---- what is sigma? ratio to the pipeline error, and whether it carries the source's Poisson noise
out = {"sigma": {}, "classes": {}, "flux": {}, "bands": sorted({r["band"] for r in rows})}
rat = np.concatenate([(r["sig"] / r["err"])[r["ok"]] for r in rows])
out["sigma"]["sigma_over_pipeline_error"] = [float(np.percentile(rat, q)) for q in (5, 50, 95)]
for lab, lo, hi in [("<1", -np.inf, 1)] + BINS:
    v = np.concatenate([(r["sig"])[r["ok"] & (r["snr"] >= lo) & (r["snr"] < hi)] / np.median(r["sig"][r["ok"] & (r["snr"] < 1)]) for r in rows if (r["ok"] & (r["snr"] < 1)).sum() > 50])
    out["sigma"][f"sigma_over_blank_sky_sigma[{lab}]"] = float(np.median(v)) if v.size else None
print("sigma / pipeline error (5, 50, 95%):", np.round(out["sigma"]["sigma_over_pipeline_error"], 3))
print("median sigma relative to blank-sky sigma, per class:", {k.split("[")[1][:-1]: round(v, 2) for k, v in out["sigma"].items() if "[" in k and v})

# ---- metrics per class and method
for lab, lo, hi in BINS + [("headline >=5", 5, np.inf)]:
    rec = {}
    sels = [r["ok"] & (r["snr"] >= lo) & (r["snr"] < hi) for r in rows]
    rec["n"] = int(sum(s.sum() for s in sels))
    S_ = np.concatenate([(r["snr"] * r["sig"])[s] for r, s in zip(rows, sels)])
    rec["noise_only_fractional"] = float(np.median(np.concatenate([r["sig"][s] for r, s in zip(rows, sels)]) / S_))
    for k in METHODS:
        pull = np.concatenate([((r["d"] - r["mod"][k]) / r["sig"])[s] for r, s in zip(rows, sels)])
        frac = np.concatenate([(r["d"] - r["mod"][k])[s] for r, s in zip(rows, sels)]) / S_
        rec[k] = {"chi2": float((pull**2).mean()), "pull_width": float(mad(pull)), "tail5": float((np.abs(pull) > 5).mean()), "frac_resid": float(mad(frac)),
                  "frac_resid_rms": float(np.sqrt((frac**2).mean())), "pull_mean": float(pull.mean())}
    out["classes"][lab] = rec

# ---- flux ratio on sources per band and method
for b in out["bands"]:
    rec = {}
    for k in METHODS:
        num = den = 0.0
        for r in rows:
            if r["band"] != b:
                continue
            sky, src = r["ok"] & (r["snr"] <= np.percentile(r["snr"][r["ok"]], 20)), r["ok"] & (r["snr"] >= 20)  # sky: the faintest fifth of the cut
            if sky.sum() < 50 or src.sum() < 50:
                continue
            num += (r["mod"][k][src] - np.median(r["mod"][k][sky])).sum()
            den += (r["d"][src] - np.median(r["d"][sky])).sum()
        rec[k] = float(num / den) if den else None
    out["flux"][b] = rec
json.dump(out, open(OUT + "/diag.json", "w"), indent=1)

NAMES = {"pipeline": "Official pipeline", "base0": "Ours, baseline", "direct0": "Ours, new renderer", "ks_direct0": "Ours, official kernel start",
         "ks_direct0_400": "Ours, official start, 400 min", "ssf4_200": "Survey Fusion"}
print("\nmetric: chi2/N | robust pull width | share beyond 5 sigma | fractional residual (robust)")
for lab, rec in out["classes"].items():
    print(f"{lab:13s} n {rec['n']:8d} noise-only fractional {rec['noise_only_fractional']:.3f} | " + " | ".join(
        f"{k}: {rec[k]['chi2']:.2f} {rec[k]['pull_width']:.3f} {rec[k]['tail5']:.4f} {rec[k]['frac_resid']:.3f}" for k in METHODS))
print("\nmetric: flux ratio on sources (model / data, snr_ref >= 20)")
for b, rec in out["flux"].items():
    print(b, " ".join(f"{k} {v:.3f}" if v else f"{k} -" for k, v in rec.items()))

# ---- figure 1: pull distributions against a unit Gaussian
SHOW = [k for k in ("pipeline", "direct0", "ks_direct0", "ssf4_200") if k in METHODS]
COL = {"pipeline": "#8f949c", "base0": "#2a78d6", "direct0": "#1baf7a", "ks_direct0": "#2a78d6", "ssf4_200": "#eb6834"}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
edges = np.linspace(-10, 10, 161)
cen = 0.5 * (edges[1:] + edges[:-1])
for ax, (lab, lo, hi) in zip(axes, BINS[1:]):
    for k in SHOW:
        pull = np.concatenate([((r["d"] - r["mod"][k]) / r["sig"])[r["ok"] & (r["snr"] >= lo) & (r["snr"] < hi)] for r in rows])
        hgram = np.histogram(np.clip(pull, -10, 10), edges, density=True)[0]
        ax.step(cen, hgram, where="mid", color=COL[k], lw=1.6, label=NAMES[k])
    ax.plot(cen, np.exp(-0.5 * cen**2) / np.sqrt(2 * np.pi), color="k", lw=1, ls=":", label="unit Gaussian")
    ax.set_yscale("log"); ax.set_ylim(1e-5, 1); ax.set_xlim(-10, 10)
    ax.set_title(f"signal-to-noise {lab.replace('-', ' to ').replace('>200', 'above 200')}", fontsize=10)
    ax.set_xlabel("pull = (data − prediction) / noise")
axes[0].set_ylabel("density (log scale)")
axes[0].legend(fontsize=7.5, frameon=False, loc="lower center")
fig.tight_layout(); fig.savefig(OUT + "/diag_pulls.png", dpi=120); plt.close(fig)

# ---- figure 2: residual maps of three held-out exposures, full cut
def pick(band):
    c = [r for r in rows if r["band"] == band]
    return max(c, key=lambda r: (r["ok"] & (r["snr"] > 50)).sum()) if c else None


picks = [p for p in (pick("F090W"), pick("F200W"), pick("F444W")) if p]
fig, axes = plt.subplots(len(picks), len(SHOW) + 1, figsize=(3.0 * (len(SHOW) + 1), 3.25 * len(picks)))
for i, r in enumerate(picks):
    sky = np.median(r["d"][r["ok"] & (r["snr"] <= np.percentile(r["snr"][r["ok"]], 20))])
    s0 = np.median(r["sig"][r["ok"]])
    img = np.where(r["ok"], np.arcsinh((r["d"] - sky) / (3 * s0)), np.nan)
    axes[i, 0].imshow(img, origin="lower", cmap="gray_r", vmin=-1, vmax=6, interpolation="nearest")
    axes[i, 0].set_title(f"{r['band']} held-out exposure\n(data, asinh stretch)", fontsize=9)
    for j, k in enumerate(SHOW, 1):
        pull = np.where(r["ok"], (r["d"] - r["mod"][k]) / r["sig"], np.nan)
        im = axes[i, j].imshow(pull, origin="lower", cmap="RdBu_r", vmin=-6, vmax=6, interpolation="nearest")
        c2 = float(np.nanmean(pull[r["ok"] & (r["snr"] >= 5)] ** 2))
        axes[i, j].set_title(f"{NAMES[k]}\nchi2/N at signal-to-noise ≥ 5: {c2:.2f}", fontsize=9)
    for a in axes[i]:
        a.set_xticks([]); a.set_yticks([])
cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.01)
cb.set_label("residual / noise  (data − prediction)")
fig.savefig(OUT + "/diag_residual_maps.png", dpi=105, bbox_inches="tight"); plt.close(fig)

# ---- figure 3: the brightest compact source of each of those exposures, 41 x 41 pixels
fig, axes = plt.subplots(len(picks), len(SHOW) + 1, figsize=(2.5 * (len(SHOW) + 1), 2.75 * len(picks)))
for i, r in enumerate(picks):
    h = 20
    okc = r["ok"].copy()  # a peak whose 5 x 5 core is fully scored
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            okc &= np.roll(np.roll(r["ok"], dy, 0), dx, 1)
    d = np.where(okc, r["d"], -np.inf)
    inner = d[h:-h, h:-h]
    py, px = np.unravel_index(np.argmax(inner), inner.shape)
    py, px = py + h, px + h
    sl = (slice(py - h, py + h + 1), slice(px - h, px + h + 1))
    sky = np.median(r["d"][r["ok"] & (r["snr"] <= np.percentile(r["snr"][r["ok"]], 20))])
    peak = r["d"][py, px] - sky
    axes[i, 0].imshow(np.arcsinh((r["d"][sl] - sky) / (0.01 * peak)), origin="lower", cmap="gray_r", interpolation="nearest")
    axes[i, 0].set_title(f"{r['band']}: brightest source\npeak signal-to-noise {r['snr'][py, px]:.0f}", fontsize=9)
    for j, k in enumerate(SHOW, 1):
        frac = np.where(r["ok"][sl], (r["d"][sl] - r["mod"][k][sl]) / peak, np.nan)
        im = axes[i, j].imshow(frac, origin="lower", cmap="RdBu_r", vmin=-0.05, vmax=0.05, interpolation="nearest")
        core = frac[h - 2:h + 3, h - 2:h + 3]
        axes[i, j].set_title(f"{NAMES[k]}\nworst core pixel: {100 * np.nanmax(np.abs(core)):.1f}% of the peak", fontsize=9)
    for a in axes[i]:
        a.set_xticks([]); a.set_yticks([])
cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.01)
cb.set_label("residual as a fraction of the source's peak")
fig.savefig(OUT + "/diag_stamps.png", dpi=105, bbox_inches="tight"); plt.close(fig)
print("figures written")
