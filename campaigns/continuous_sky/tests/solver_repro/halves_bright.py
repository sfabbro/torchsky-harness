"""What is the bright-source discrepancy between the two half fits (SMACS, 19 Nov GPU runs)? Reads the products of both halves for each band
and, per bright source, fits the half-difference with the local model

    I_0(x) - I_1(x) = (s - 1) I(x) - delta . grad I(x) + w lap I(x) + c,      I = (I_0 + I_1) / 2,

i.e. a flux-scale difference s, a position difference delta (arcsec) and a width difference w (arcsec^2: a Gaussian of variance V has
dI/dV = lap I / 2, so w = (V_0 - V_1) / 2), plus a constant. Reports the pull of the difference against the stated sigma before and after
the fit, the spread of s, delta and w over sources, and whether they are common to all sources (a calibration or registration difference
between the halves) or source to source (fit scatter).

    python tests/solver_repro/halves_bright.py [--dir /home/claude/work/checkin/unc] [--out tests/solver_repro/results/halves_bright.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load(d, h, b):
    from astropy.io import fits
    with fits.open(os.path.join(d, f"h{h}_{b}.fits")) as f:
        return {k: f[k].data.astype(np.float64) for k in ("SKY", "SIGMA", "SKY_FIT", "COVERED")}, dict(f[0].header)


def peaks(img, mask, n, min_sep):
    """Local maxima (3 x 3) inside mask, brightest first, at least min_sep nodes apart."""
    from scipy.ndimage import maximum_filter
    mx = maximum_filter(img, size=3)
    cand = np.argwhere((img == mx) & mask)
    vals = img[cand[:, 0], cand[:, 1]]
    order = np.argsort(-vals)
    out = []
    for i in order:
        y, x = cand[i]
        if all((y - yy) ** 2 + (x - xx) ** 2 >= min_sep ** 2 for yy, xx in out):
            out.append((int(y), int(x)))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/claude/work/checkin/unc")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "halves_bright.json"))
    ap.add_argument("--n-src", type=int, default=40)
    ap.add_argument("--win", type=int, default=6, help="half width of the fitting window [nodes]")
    a = ap.parse_args()
    res = {}
    for b in ("F090W", "F200W", "F444W"):
        p0, h0 = load(a.dir, 0, b)
        p1, h1 = load(a.dir, 1, b)
        pitch = float(h0["PITCH"])
        cov = (p0["COVERED"] > 0) & (p1["COVERED"] > 0) & (p0["SIGMA"] > 0) & (p1["SIGMA"] > 0)
        I0, I1 = p0["SKY"], p1["SKY"]
        I = 0.5 * (I0 + I1)
        sig = np.sqrt(p0["SIGMA"] ** 2 + p1["SIGMA"] ** 2)            # sd of I0 - I1 if the two halves' errors are independent
        pull = np.where(cov, (I0 - I1) / np.where(sig > 0, sig, 1), np.nan)
        snr = np.where(cov, I / np.where(sig > 0, sig, 1), np.nan)
        q = np.nanpercentile(I[cov], [30, 90])
        faint, bright = cov & (I <= q[0]), cov & (I >= q[1])
        rob = lambda v: 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v)))  # noqa: E731
        gy, gx = np.gradient(I, pitch)
        lap = np.gradient(np.gradient(I, pitch, axis=0), pitch, axis=0) + np.gradient(np.gradient(I, pitch, axis=1), pitch, axis=1)
        inner = cov.copy()
        inner[:a.win + 2], inner[-a.win - 2:], inner[:, :a.win + 2], inner[:, -a.win - 2:] = False, False, False, False
        pk = peaks(np.where(inner, I, -np.inf), inner, a.n_src, 2 * a.win)
        srcs = []
        for (y, x) in pk:
            sl = (slice(y - a.win, y + a.win + 1), slice(x - a.win, x + a.win + 1))
            m = cov[sl].ravel()
            D = (I0 - I1)[sl].ravel()[m]
            W = 1.0 / sig[sl].ravel()[m]
            X = np.stack([I[sl].ravel()[m], -gx[sl].ravel()[m], -gy[sl].ravel()[m], lap[sl].ravel()[m], np.ones(m.sum())], 1)
            coef, *_ = np.linalg.lstsq(X * W[:, None], D * W, rcond=None)
            r0 = D * W
            r1 = (D - X @ coef) * W
            rs = (D - X[:, [0, 4]] @ np.linalg.lstsq(X[:, [0, 4]] * W[:, None], D * W, rcond=None)[0]) * W      # scale only
            srcs.append({"row": y, "col": x, "peak_snr": float(snr[y, x]), "scale_minus_1": float(coef[0]), "shift_mas": [float(coef[1] * 1e3), float(coef[2] * 1e3)],
                         "width_var_diff_arcsec2_half": float(coef[3]), "pull_rms_before": float(np.sqrt(np.mean(r0 ** 2))),
                         "pull_rms_after_scale": float(np.sqrt(np.mean(rs ** 2))), "pull_rms_after_full": float(np.sqrt(np.mean(r1 ** 2)))})
        arr = lambda k: np.array([s[k] for s in srcs])  # noqa: E731
        sh = np.array([s["shift_mas"] for s in srcs])
        res[b] = {"sigma_median_half0": float(np.median(p0["SIGMA"][cov])), "sigma_median_half1": float(np.median(p1["SIGMA"][cov])),
                  "pull_rms_faint30": float(np.sqrt(np.nanmean(pull[faint] ** 2))), "pull_robust_faint30": float(rob(pull[faint])),
                  "pull_rms_bright10": float(np.sqrt(np.nanmean(pull[bright] ** 2))), "pull_robust_bright10": float(rob(pull[bright])),
                  "sky_minus_skyfit_max_abs": float(max(np.abs(p0["SKY"] - p0["SKY_FIT"])[cov].max(), np.abs(p1["SKY"] - p1["SKY_FIT"])[cov].max())),
                  "n_sources": len(srcs), "median_peak_snr": float(np.median(arr("peak_snr"))),
                  "scale_minus_1_median": float(np.median(arr("scale_minus_1"))), "scale_minus_1_spread_rob": float(rob(arr("scale_minus_1"))),
                  "shift_mas_median": np.median(sh, 0).tolist(), "shift_mas_spread_rob": [float(rob(sh[:, 0])), float(rob(sh[:, 1]))],
                  "width_median": float(np.median(arr("width_var_diff_arcsec2_half"))), "width_spread_rob": float(rob(arr("width_var_diff_arcsec2_half"))),
                  "pull_rms_sources_before_median": float(np.median(arr("pull_rms_before"))),
                  "pull_rms_sources_after_scale_median": float(np.median(arr("pull_rms_after_scale"))),
                  "pull_rms_sources_after_full_median": float(np.median(arr("pull_rms_after_full"))), "sources": srcs[:12]}
        r = res[b]
        print(f"{b}: sigma median h0 {r['sigma_median_half0']:.3g} h1 {r['sigma_median_half1']:.3g}; pull faint30 rms {r['pull_rms_faint30']:.2f} "
              f"(robust {r['pull_robust_faint30']:.2f}), bright10 rms {r['pull_rms_bright10']:.1f} (robust {r['pull_robust_bright10']:.1f}); "
              f"{r['n_sources']} sources, median peak S/N {r['median_peak_snr']:.0f}: scale-1 median {r['scale_minus_1_median']:+.4f} spread {r['scale_minus_1_spread_rob']:.4f}; "
              f"shift median {np.round(r['shift_mas_median'], 2)} mas spread {np.round(r['shift_mas_spread_rob'], 2)}; width {r['width_median']:+.2e} spread {r['width_spread_rob']:.2e}; "
              f"source pull rms before {r['pull_rms_sources_before_median']:.1f}, after scale {r['pull_rms_sources_after_scale_median']:.1f}, after scale+shift+width {r['pull_rms_sources_after_full_median']:.1f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
