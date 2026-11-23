"""Paired scores of held-out exposures from the runner's side files (<run>/xpred/<rootname>.npz), for exposures that have no protocol mask
(HST) or for a quick paired look at any instrument.

    python score_cross.py --reference hst=RUNS/x_hst_direct --runs "jwst sky=RUNS/x_j2h_direct" joint=RUNS/x_joint_direct --instrument WFC3_IR

For a held-out pixel: data d, pipeline error e, each method's prediction m. Everything is computed on the pixels that are valid and
predicted by EVERY method (identical pixels).

Brightness classes come from the REFERENCE method's own prediction, not from the data (selecting on noisy data would bias every class):
    s = (m_ref - median of m_ref over the exposure) / e,   classes <1, 1-5, 5-20, 20-50, 50-200, >200; headline: s >= 5.
The reference plays the part the official mosaic plays in protocol v1. Stated consequence: the classes favour the reference slightly.

Noise: sigma = f x e, with one factor f per exposure = 1.4826 x MAD of (d - m_ref) / e over the faintest class (s < 1), so that blank sky
has unit scatter (the pipeline error of WFC3/IR is 10 to 25% too large, that of ACS too small). The same f for every method.

Reported: chi2 / N per class and method, the paired difference from the reference (method - reference; negative = better than the
reference), each with a 68% interval from a bootstrap over dither groups, and the headline per band.
"""

import argparse
import glob
import json
import os

import numpy as np

BINS = [("<1", -np.inf, 1), ("1-5", 1, 5), ("5-20", 5, 20), ("20-50", 20, 50), ("50-200", 50, 200), (">200", 200, np.inf), ("headline >=5", 5, np.inf)]


def load(run):
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "*.npz"))):
        z = np.load(f)
        out[os.path.basename(f)[:-4]] = {k: z[k] for k in z.files}
    return out


def boot(groups, num, den, n=2000, seed=0):
    """68% interval of sum(num) / sum(den) over groups resampled with replacement; num and den are per-group sums."""
    rng = np.random.default_rng(seed)
    g = len(groups)
    if g < 3:
        return None, None
    idx = rng.integers(0, g, (n, g))
    v = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(np.percentile(v, 16)), float(np.percentile(v, 84))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="label=run directory (its xpred folder is read)")
    ap.add_argument("--runs", nargs="*", default=[], help="label=run directory, or label=run directory:xpred_<name> for a variant")
    ap.add_argument("--instrument", default="WFC3_IR")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    def parse(tok):
        lab, path = tok.split("=", 1)
        run, sub = (path.rsplit(":", 1) + ["xpred"])[:2] if ":" in path else (path, "xpred")
        return lab, os.path.join(run, sub)

    labs = [parse(a.reference)] + [parse(t) for t in a.runs]
    data = {lab: load(path) for lab, path in labs}
    ref = labs[0][0]
    roots = sorted(r for r in data[ref] if str(data[ref][r]["instrument"]) == a.instrument and all(r in data[lab] for lab, _ in labs))
    if not roots:
        raise SystemExit(f"no held-out exposure of {a.instrument} common to {[lab for lab, _ in labs]}")
    rows = []  # per exposure: group, band, class index per pixel, squared normalised residual per method
    for r in roots:
        z = data[ref][r]
        d, e, mref = z["data"].astype(np.float64), z["err"].astype(np.float64), z["model"].astype(np.float64)
        ok = z["valid"] & np.isfinite(d) & (e > 0)
        for lab, _ in labs:
            ok &= np.isfinite(data[lab][r]["model"])
        if ok.sum() < 100:
            continue
        s = (mref - np.median(mref[ok])) / e
        faint = ok & (s < 1)
        q = ((d - mref) / e)[faint]
        f = 1.4826 * np.median(np.abs(q - np.median(q))) if q.size > 100 else 1.0
        res = {lab: (((d - data[lab][r]["model"].astype(np.float64)) / (f * e)) ** 2)[ok] for lab, _ in labs}
        rows.append({"root": r, "group": str(z["group"]), "band": str(z["band"]), "s": s[ok], "r2": res, "f": float(f)})
    groups = sorted({x["group"] for x in rows})
    bands = sorted({x["band"] for x in rows})
    print(f"{a.instrument}: {len(rows)} held-out exposures in {len(groups)} groups, bands {bands}; noise factor per exposure "
          f"{min(x['f'] for x in rows):.2f} to {max(x['f'] for x in rows):.2f} (median {np.median([x['f'] for x in rows]):.2f})\n")

    def table(sel_rows, title):
        out = {}
        print(f"**{title}**\n")
        print("| Class | pixels | " + " | ".join(lab for lab, _ in labs) + " | " + " | ".join(f"{lab} minus {ref}" for lab, _ in labs[1:]) + " |")
        print("|---|---|" + "---|" * (2 * len(labs) - 1))
        gl = sorted({x["group"] for x in sel_rows})
        for name, lo, hi in BINS:
            num = {lab: np.zeros(len(gl)) for lab, _ in labs}
            den = np.zeros(len(gl))
            for x in sel_rows:
                m = (x["s"] >= lo) & (x["s"] < hi)
                gi = gl.index(x["group"])
                den[gi] += m.sum()
                for lab, _ in labs:
                    num[lab][gi] += x["r2"][lab][m].sum()
            if den.sum() < 50:
                continue
            cells, rec = [], {"n": int(den.sum())}
            for lab, _ in labs:
                v = num[lab].sum() / den.sum()
                lo_, hi_ = boot(gl, num[lab], den)
                cells.append(f"{v:.3f}" + (f" [{lo_:.2f}, {hi_:.2f}]" if lo_ is not None else ""))
                rec[lab] = {"value": float(v), "lo": lo_, "hi": hi_}
            for lab, _ in labs[1:]:
                dn = num[lab] - num[ref]
                v = dn.sum() / den.sum()
                lo_, hi_ = boot(gl, dn, den)
                cells.append(f"{v:+.3f}" + (f" [{lo_:+.2f}, {hi_:+.2f}]" if lo_ is not None else ""))
                rec[f"{lab}-{ref}"] = {"value": float(v), "lo": lo_, "hi": hi_}
            print(f"| {name} | {int(den.sum())} | " + " | ".join(cells) + " |")
            out[name] = rec
        print()
        return out

    result = {"instrument": a.instrument, "reference": ref, "exposures": [x["root"] for x in rows], "all": table(rows, "All bands"), "bands": {}}
    for b in bands:
        result["bands"][b] = table([x for x in rows if x["band"] == b], f"Band {b}")
    if a.out:
        json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
