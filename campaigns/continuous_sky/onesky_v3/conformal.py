"""Calibrated prediction intervals for held-out exposures, after the fit (split conformal within classes), for any protocol-v1 method.

    python conformal.py --root /arc/projects/ots/adass2026/protocol/v1 --field SMACS --split dither --methods lib_base_s0 lib_direct_s0 ssf_11ep

For a held-out pixel i a method gives a prediction m_i and a noise s_i (its own `sigma` when the prediction file has one, else the
protocol's sigma: pipeline error x noise factor). Score: r_i = |d_i - m_i| / s_i. Within a class (bin of the official mosaic's
signal-to-noise) and for a coverage 1 - alpha, q is the ceil((n + 1)(1 - alpha)) / n quantile of the scores of the CALIBRATION exposures,
and the interval of a new pixel is m +- q s.

Exchangeability is assumed between dither groups, not between pixels (pixels of one exposure share its nuisance terms): calibration and
test are whole dither groups, leave one group out. What is reported per method and class:

  raw      coverage of m +- z s with z the Gaussian value (0.9945 for 68.27%, 1.96 for 95%): how honest the method's own noise is;
  q / z    the factor by which its interval must be stretched;
  cover    coverage on the left-out group after calibration, mean and spread over groups (should be 1 - alpha);
  width    median of q s / (protocol sigma): the size of the calibrated interval in units of the data's noise. At equal coverage the
           method with the smaller width knows the sky better: this is the number to compare between methods.

Shortcuts: the quantile is taken over pixels pooled across the calibration groups (a class with fewer than 200 pixels is skipped); one
alpha at a time; spatial correlation inside an exposure is not modelled, which the spread over groups exposes.
"""

import argparse
import json
import math
import os

import numpy as np

BINS = [("<1", -np.inf, 1), ("1-5", 1, 5), ("5-20", 5, 20), ("20-50", 20, 50), ("50-200", 50, 200), (">200", 200, np.inf)]


def load_sci(path):
    from astropy.io import fits

    with fits.open(path, memmap=False) as hl:
        return np.array(hl["SCI"].data, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--field", default="SMACS")
    ap.add_argument("--split", default="dither")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.3173, 0.05])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sp = json.load(open(os.path.join(a.root, f"splits_{a.field}.json")))
    by_root = {e["rootname"]: e for e in sp["exposures"]}
    held = [r for r in sp["splits"][a.split]["heldout"]]
    res = {}
    for meth in a.methods:
        rows = []  # per exposure: group, per class arrays (score, own sigma / protocol sigma)
        for root in held:
            pp = os.path.join(a.root, "pred", a.field, meth, a.split, root + ".npz")
            mp = os.path.join(a.root, "heldout", a.field, root + ".npz")
            if not (os.path.exists(pp) and os.path.exists(mp)):
                continue
            pred, mask = np.load(pp), np.load(mp)
            d = load_sci(by_root[root]["path"])
            ok = mask["valid"] & np.isfinite(pred["model"])
            if "model_valid" in pred.files:
                ok &= pred["model_valid"]
            psig = mask["sigma"].astype(np.float64)
            own = pred["sigma"].astype(np.float64) if "sigma" in pred.files else psig
            ok &= np.isfinite(own) & (own > 0) & (psig > 0)
            r = np.abs(d - pred["model"])[ok] / own[ok]
            rows.append(dict(group=by_root[root]["dither_group"], snr=mask["snr_ref"][ok], r=r, rel=(own[ok] / psig[ok]), own_sigma="sigma" in pred.files))
        groups = sorted({x["group"] for x in rows})
        out = {"n_exposures": len(rows), "n_groups": len(groups), "own_sigma": bool(rows and rows[0]["own_sigma"]), "classes": {}}
        for name, lo, hi in BINS:
            per = {}
            for g in groups:
                sel = [x for x in rows if x["group"] == g]
                rr = np.concatenate([x["r"][(x["snr"] >= lo) & (x["snr"] < hi)] for x in sel]) if sel else np.zeros(0)
                rl = np.concatenate([x["rel"][(x["snr"] >= lo) & (x["snr"] < hi)] for x in sel]) if sel else np.zeros(0)
                per[g] = (rr, rl)
            n_all = sum(v[0].size for v in per.values())
            if n_all < 200 or len(groups) < 3:
                continue
            cls = {"n_pixels": int(n_all)}
            for al in a.alphas:
                z = {0.3173: 0.99998, 0.05: 1.95996}.get(round(al, 4)) or math.sqrt(2) * _erfinv(1 - al)
                allr = np.concatenate([v[0] for v in per.values()])
                raw = float((allr <= z).mean())
                cov, qs, wid = [], [], []
                for g in groups:
                    cal = np.concatenate([per[h][0] for h in groups if h != g])
                    test_r, test_rel = per[g]
                    if cal.size < 200 or test_r.size < 20:
                        continue
                    k = min(int(math.ceil((cal.size + 1) * (1 - al))), cal.size)
                    q = float(np.partition(cal, k - 1)[k - 1])
                    cov.append(float((test_r <= q).mean()))
                    qs.append(q)
                    wid.append(float(np.median(q * test_rel)))
                if not cov:
                    continue
                cls[f"alpha_{al:g}"] = {"raw_coverage": raw, "stretch_q_over_z": float(np.mean(qs) / z), "coverage_mean": float(np.mean(cov)),
                                        "coverage_min": float(np.min(cov)), "coverage_max": float(np.max(cov)),
                                        "width_in_data_sigma": float(np.mean(wid)), "width_spread": float(np.std(wid))}
            out["classes"][name] = cls
        res[meth] = out
    for al in a.alphas:
        key = f"alpha_{al:g}"
        print(f"\n### target coverage {100 * (1 - al):.1f}%\n")
        print("| Method | own noise | class | pixels | raw coverage | stretch | coverage after calibration (min to max over groups) | interval half-width / data sigma |")
        print("|---|---|---|---|---|---|---|---|")
        for meth, out in res.items():
            for name, cls in out["classes"].items():
                c = cls.get(key)
                if c:
                    print(f"| {meth} | {'yes' if out['own_sigma'] else 'no'} | {name} | {cls['n_pixels']} | {c['raw_coverage']:.3f} | {c['stretch_q_over_z']:.2f} | "
                          f"{c['coverage_mean']:.3f} ({c['coverage_min']:.3f} to {c['coverage_max']:.3f}) | {c['width_in_data_sigma']:.2f} ± {c['width_spread']:.2f} |")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def _erfinv(x):
    from scipy.special import erfinv

    return float(erfinv(x))


if __name__ == "__main__":
    main()
