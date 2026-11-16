"""Check of the rate-level scorer (score_paired.py --level rate) on the 12-exposure sample (no offset, pooled per S/N bin of snr_ref).

For each method with calibrated-unit predictions m (MJy/sr), on the protocol pixels:
  cal          r = (cal SCI - m) / sigma_protocol                       (score.py, cal level)
  rate         r = (rate SCI - C m) / (rate ERR x noise_factor)         (score_paired.py --level rate)
  rate_calsig  r = (rate SCI - C m) / (C sigma_protocol)                (rate level with the calibrated sigma carried over: must equal cal,
                                                                         since rate SCI = C cal SCI exactly; tests C and the pixel handling)
and the decomposition of rate / cal: (sigma_protocol / (rate ERR x nf / C))^2 = (sigma_protocol / (cal ERR x nf))^2 x (cal ERR^2 / (cal ERR^2
- VAR_FLAT)); the first factor needs the full _cal files (ERR, VAR_FLAT), which are fetched for one exposure only (--full-cal-dir).

    python tests/check_rate_scoring.py --data /home/claude/work/score_paired/tests/data --rate-dir dl/rate [--full-cal-dir dl/cal]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from score_paired import BIN_NAMES, EDGES, conversion_map, prediction_units  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rate-dir", required=True)
    ap.add_argument("--full-cal-dir", default="")
    ap.add_argument("--methods", nargs="+", default=["stage3_blot", "fieldop_fix_200"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from astropy.io import fits
    roots = sorted(f[:-4] for f in os.listdir(os.path.join(a.data, "heldout", "SMACS")))
    acc = {m: {k: np.zeros((2, 6)) for k in ("cal", "rate", "rate_calsig")} for m in a.methods}
    sig_ratio = np.zeros((2, 6))
    decomp = {}
    for r in roots:
        mk = np.load(os.path.join(a.data, "heldout", "SMACS", r + ".npz"))
        cal = np.asarray(fits.open(os.path.join(a.data, "cal", r + "_cal.fits"))["SCI"].data, np.float64)
        with fits.open(os.path.join(a.rate_dir, r + "_rate.fits")) as h:
            rate, rerr = np.asarray(h["SCI"].data, np.float64), np.asarray(h["ERR"].data, np.float64)
        v0 = mk["valid"].astype(bool)
        sp = mk["sigma"].astype(np.float64)
        nf = mk["noise_factor"].astype(np.float64)
        C, _ = conversion_map(os.path.join(a.data, "cal", r + "_cal.fits"), os.path.join(a.rate_dir, r + "_rate.fits"), need=v0)
        snr = mk["snr_ref"].astype(np.float64)
        ib = np.clip(np.digitize(snr, EDGES) - 1, 0, 5)
        sr = rerr * nf
        for m in a.methods:
            pr = np.load(os.path.join(a.data, "pred", "SMACS", m, "dither", r + ".npz"))
            assert prediction_units(pr) == "cal"
            mod = pr["model"].astype(np.float64)
            v = v0 & np.isfinite(cal) & np.isfinite(sp) & (sp > 0) & np.isfinite(snr) & pr["model_valid"].astype(bool) & np.isfinite(mod) \
                & np.isfinite(rate) & np.isfinite(sr) & (sr > 0) & np.isfinite(C)
            for k, res in (("cal", (cal - mod) / sp), ("rate", (rate - C * mod) / sr), ("rate_calsig", (rate - C * mod) / (C * sp))):
                acc[m][k][0] += np.bincount(ib[v], minlength=6)
                acc[m][k][1] += np.bincount(ib[v], weights=res[v] ** 2, minlength=6)
        v = v0 & np.isfinite(sp) & (sp > 0) & np.isfinite(sr) & (sr > 0) & np.isfinite(C) & np.isfinite(snr)
        sig_ratio[0] += np.bincount(ib[v], minlength=6)
        sig_ratio[1] += np.bincount(ib[v], weights=(sp[v] * C[v] / sr[v]) ** 2, minlength=6)
        fc = os.path.join(a.full_cal_dir, r + "_cal.fits") if a.full_cal_dir else ""
        if fc and os.path.exists(fc):
            with fits.open(fc) as h:
                ce, vf = np.asarray(h["ERR"].data, np.float64), np.asarray(h["VAR_FLAT"].data, np.float64)
            w = v & np.isfinite(ce) & (ce > 0)
            decomp[r] = {"protocol_sigma_over_current_cal_err_x_nf_median": float(np.median(sp[w] / (ce[w] * nf[w]))),
                         "VAR_FLAT_share_by_bin": [float(np.mean((vf / ce ** 2)[w & (ib == b)])) if (w & (ib == b)).any() else None for b in range(6)],
                         "identity_cal_err2_over_rate_err_over_C_2_plus_VAR_FLAT_median": float(np.median(ce[w] ** 2 / ((rerr[w] / C[w]) ** 2 + vf[w])))}
    out = {"bins": BIN_NAMES, "methods": {}, "sigma_variance_ratio_protocol_over_rate_by_bin": (sig_ratio[1] / np.maximum(sig_ratio[0], 1)).tolist(),
           "n_by_bin": sig_ratio[0].astype(int).tolist(), "decomposition_one_exposure": decomp}
    for m in a.methods:
        row = {}
        for k in ("cal", "rate", "rate_calsig"):
            n, c = acc[m][k]
            row[k] = {"by_bin": (c / np.maximum(n, 1)).tolist(), "snr_5_50": float(c[2:4].sum() / n[2:4].sum()), "snr_ge5": float(c[2:].sum() / n[2:].sum()),
                      "n": n.astype(int).tolist()}
        row["rate_minus_cal_snr_5_50"] = row["rate"]["snr_5_50"] - row["cal"]["snr_5_50"]
        row["rate_calsig_minus_cal_max_abs_by_bin"] = float(np.max(np.abs(np.array(row["rate_calsig"]["by_bin"]) - np.array(row["cal"]["by_bin"]))))
        out["methods"][m] = row
    s = json.dumps(out, indent=1)
    print(s)
    if a.out:
        open(a.out, "w").write(s)


if __name__ == "__main__":
    main()
