"""Protocol-v1 scores of several methods on identical pixels that every library fit saw: the protocol's scored pixels, minus those the
torchsky loader dropped, minus those any listed method did not predict.

    python score_common.py --heldout MASKS --kept RUN/xpred --methods pipeline=PRED/stage3_blot base=PRED/lib_base_s0 direct=PRED/lib_direct_s0 \
        --reference pipeline

Why: the protocol's held-out masks (valid, sigma, snr_ref on the full detector) predate newer calibration flags. The newer files flag the
brightest cores as saturated and torchsky's loader drops flagged pixels, so a library fit is scored by the protocol on pixels it never saw
(0.33% of the headline pixels, 5% above signal-to-noise 200), and those pixels dominate the bright classes. Methods that read the older
files (the official mosaic, the harness prototype) saw them. Here every method is scored on the same kept pixels.

Inputs: MASKS/<rootname>.npz (protocol: valid, sigma, snr_ref); RUN/xpred/<rootname>.npz from any library run of the same split (data,
valid = kept by the loader, y0, x0, group, band: the loader and the cut are the same for every library run); PRED/<method>/<rootname>.npz
(protocol predictions: model, model_valid on the full detector).

Reported: chi2 / N per class of the official mosaic's signal-to-noise (no offset fitted: the protocol's "without offset" score), the paired
difference from the reference (method - reference; negative = better), each with a 68% interval from a bootstrap over dither groups, then
the headline (signal-to-noise 5 and above) per filter.

Stated limits: the data values are the loader's (same files as the protocol's for kept pixels); sigma is the protocol's.
"""

import argparse
import json
import os

import numpy as np

BINS = [("<1", -np.inf, 1), ("1-5", 1, 5), ("5-20", 5, 20), ("20-50", 20, 50), ("50-200", 50, 200), (">200", 200, np.inf), ("headline >=5", 5, np.inf)]


def boot(num, den, n=2000, seed=0):
    """68% interval of sum(num) / sum(den) over groups resampled with replacement; num and den are per-group sums."""
    g = len(den)
    if g < 3:
        return None, None
    idx = np.random.default_rng(seed).integers(0, g, (n, g))
    v = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(np.percentile(v, 16)), float(np.percentile(v, 84))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--kept", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="label=folder of protocol predictions")
    ap.add_argument("--reference", default="")
    ap.add_argument("--bands", default="", help="comma-separated filters: only their exposures are scored (default all)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    labs = [t.split("=", 1) for t in a.methods]
    ref = a.reference or labs[0][0]
    if ref not in [lab for lab, _ in labs]:
        raise SystemExit(f"reference {ref} is not one of the methods")
    roots = sorted(f[:-4] for f in os.listdir(a.kept) if f.endswith(".npz") and os.path.exists(os.path.join(a.heldout, f)))
    roots = [r for r in roots if all(os.path.exists(os.path.join(p, r + ".npz")) for _, p in labs)]
    if not roots:
        raise SystemExit("no held-out exposure common to the masks, the kept folder and every method")
    rows = []
    n_protocol = n_kept = 0
    for r in roots:
        x, m = np.load(os.path.join(a.kept, r + ".npz")), np.load(os.path.join(a.heldout, r + ".npz"))
        if a.bands and str(x["band"]) not in a.bands.split(","):
            continue
        y0, x0 = int(x["y0"]), int(x["x0"])
        ny, nx = x["data"].shape
        sl = (slice(y0, y0 + ny), slice(x0, x0 + nx))
        d, sig, snr = x["data"].astype(np.float64), m["sigma"][sl].astype(np.float64), m["snr_ref"][sl]
        prot = m["valid"][sl] & (sig > 0) & np.isfinite(snr)
        ok = prot & x["valid"].astype(bool) & np.isfinite(d)
        models = {}
        for lab, p in labs:
            z = np.load(os.path.join(p, r + ".npz"))
            models[lab] = z["model"][sl].astype(np.float64)
            ok &= np.isfinite(models[lab])
            if "model_valid" in z.files:
                ok &= z["model_valid"][sl]
        n_protocol += int((prot & (snr >= 5)).sum())
        n_kept += int((ok & (snr >= 5)).sum())
        rows.append({"group": str(x["group"]), "band": str(x["band"]), "s": snr[ok], "r2": {lab: (((d - models[lab]) / np.where(sig > 0, sig, 1)) ** 2)[ok] for lab, _ in labs}})
    print(f"{len(rows)} held-out exposures in {len({x['group'] for x in rows})} dither groups; headline pixels: {n_protocol} scored by the protocol on the cut, "
          f"{n_kept} common here ({100 * (1 - n_kept / max(n_protocol, 1)):.2f}% dropped)\n")

    def table(sel, title):
        out = {}
        gl = sorted({x["group"] for x in sel})
        print(f"**{title}**\n")
        print("| Class | pixels | " + " | ".join(lab for lab, _ in labs) + " | " + " | ".join(f"{lab} minus {ref}" for lab, _ in labs if lab != ref) + " |")
        print("|---|---|" + "---|" * (2 * len(labs) - 1))
        for name, lo, hi in BINS:
            num = {lab: np.zeros(len(gl)) for lab, _ in labs}
            den = np.zeros(len(gl))
            for x in sel:
                msk = (x["s"] >= lo) & (x["s"] < hi)
                gi = gl.index(x["group"])
                den[gi] += msk.sum()
                for lab, _ in labs:
                    num[lab][gi] += x["r2"][lab][msk].sum()
            if den.sum() < 50:
                continue
            cells, rec = [], {"n": int(den.sum())}
            for lab, _ in labs:
                v = num[lab].sum() / den.sum()
                lo_, hi_ = boot(num[lab], den)
                cells.append(f"{v:.3f}" + (f" [{lo_:.2f}, {hi_:.2f}]" if lo_ is not None else ""))
                rec[lab] = {"value": float(v), "lo": lo_, "hi": hi_}
            for lab, _ in labs:
                if lab == ref:
                    continue
                dn = num[lab] - num[ref]
                v = dn.sum() / den.sum()
                lo_, hi_ = boot(dn, den)
                cells.append(f"{v:+.3f}" + (f" [{lo_:+.2f}, {hi_:+.2f}]" if lo_ is not None else ""))
                rec[f"{lab}-{ref}"] = {"value": float(v), "lo": lo_, "hi": hi_}
            print(f"| {name} | {int(den.sum())} | " + " | ".join(cells) + " |")
            out[name] = rec
        print()
        return out

    result = {"reference": ref, "exposures": roots, "headline_pixels_protocol": n_protocol, "headline_pixels_common": n_kept, "all": table(rows, "All filters"), "bands": {}}
    bands = sorted({x["band"] for x in rows})
    print("**Headline (signal-to-noise 5 and above) per filter**\n")
    print("| Filter | pixels | " + " | ".join(lab for lab, _ in labs) + " |")
    print("|---|---|" + "---|" * len(labs))
    for b in bands:
        sel = [x for x in rows if x["band"] == b]
        gl = sorted({x["group"] for x in sel})
        cells, rec, n = [], {}, 0
        for lab, _ in labs:
            num, den = np.zeros(len(gl)), np.zeros(len(gl))
            for x in sel:
                msk = x["s"] >= 5
                num[gl.index(x["group"])] += x["r2"][lab][msk].sum()
                den[gl.index(x["group"])] += msk.sum()
            v, (lo_, hi_), n = num.sum() / max(den.sum(), 1), boot(num, den), int(den.sum())
            cells.append(f"{v:.3f}" + (f" [{lo_:.2f}, {hi_:.2f}]" if lo_ is not None else ""))
            rec[lab] = {"value": float(v), "lo": lo_, "hi": hi_}
        print(f"| {b} | {n} | " + " | ".join(cells) + " |")
        result["bands"][b] = rec
    if a.out:
        json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
