#!/usr/bin/env python3
"""score.py -- the scorer of evaluation protocol v1 (see INTERFACE.md).

    python score.py score   --field SMACS --method zero --split dither [--root /arc/projects/ots/adass2026/protocol/v1]
    python score.py compare --field SMACS --split dither            # table across all methods that have a scores file

Inputs (all under --root):  splits_<FIELD>.json,  heldout/<FIELD>/<rootname>.npz (mask files),  pred/<FIELD>/<METHOD>/<SPLIT>/<rootname>.npz,
and the SCI array of the original _cal file (path from the splits file).
Outputs:  scores/<FIELD>/<METHOD>/<SPLIT>.json,  scores/<FIELD>/comparison_<SPLIT>.json and .md.

What is computed, per held-out exposure d (SCI of the _cal), prediction m ('model'), mask (valid, snr_ref, sigma, box_id):
  * pixel set  V = mask.valid & finite(d) & finite(sigma) & sigma > 0 & finite(snr_ref) & model_valid & finite(m).
    Pixels of mask.valid that the method does not claim (model_valid False or a non-finite model) are MISSING: counted and reported, not scored.
    A missing prediction file makes the whole exposure missing.  If the prediction has no 'model_valid' key this is reported
    ("model_valid_provided": false) and only non-finite model pixels count as missing.
  * ONE robust constant background offset per exposure, identical for every method: b = clipped median of (d - m) over V with snr_ref < 1
    (3 sigma clipping with the pixel sigma, 5 iterations).  Scores are reported WITH it (r = (d - m - b)/sigma) and WITHOUT it (b = 0).
    Fewer than {minbg} pixels for the offset: b = 0 and the exposure is flagged.
  * chi2/N = mean r^2 in bins of snr_ref  [<1, 1-5, 5-20, 20-50, 50-200, >200];  headline = chi2/N pooled over snr_ref >= 5;
    fractional residual rms((d - m - b)/m) over snr_ref > 50 and m > 0.
  * Aggregation levels: field, filter, detector, box, (filter x box).  Intervals: 68% (16-84%) bootstrap over DITHER GROUPS (exposure number
    + visit + activity in one filter; all detector images of one pointing are drawn together), {nboot} draws, seed fixed.
APPROXIMATIONS
  S1  pixels are pooled over the aggregate (pixel-weighted chi2/N), not averaged over exposures.
  S2  pixel noise is treated as independent; chi2 intervals from the group bootstrap therefore reflect group-to-group variation only
      (correlated noise in the _cal pixels, charge-sharing or drizzle-like correlations are not modelled).
  S3  an exposure that lies in several boxes contributes one row per box; the offset is shared.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = "/arc/projects/ots/adass2026/protocol/v1"
EDGES = np.array([-np.inf, 1.0, 5.0, 20.0, 50.0, 200.0, np.inf])
BIN_NAMES = ["<1", "1-5", "5-20", "20-50", "50-200", ">200"]
HEAD_FROM = 2           # bins >= index 2 (snr_ref >= 5)
FRAC_FROM = 4           # bins >= index 4 (snr_ref > 50)
MIN_BG, N_BOOT, SEED = 100, 500, 12345
__doc__ = __doc__.format(minbg=MIN_BG, nboot=N_BOOT)


# ----------------------------------------------------------------------------- one exposure
def background_offset(resid, sigma, sel, nclip=5, clip=3.0):
    """Clipped median of resid over sel; returns (offset, n_used)."""
    x, s = resid[sel], sigma[sel]
    if x.size == 0:
        return 0.0, 0
    keep = np.ones(x.size, bool)
    b = float(np.median(x))
    for _ in range(nclip):
        k = np.abs(x - b) < clip * s
        if k.sum() < 10:
            break
        keep = k
        b2 = float(np.median(x[keep]))
        if abs(b2 - b) < 1e-9 * max(1.0, abs(b)):
            b = b2
            break
        b = b2
    return b, int(keep.sum())


def exposure_rows(d, model, mask, model_valid=None):
    """Returns (rows, info).  rows: list of dict(box, n[var,bin], chi2[var,bin], fn[var], f2[var]) for var 0 = without, 1 = with offset."""
    valid = mask["valid"].astype(bool)
    sigma = mask["sigma"].astype(np.float64)
    snr = mask["snr_ref"].astype(np.float64)
    box = mask["box_id"]
    d = np.asarray(d, np.float64)
    m = np.asarray(model, np.float64)
    base = valid & np.isfinite(d) & np.isfinite(sigma) & (sigma > 0) & np.isfinite(snr)
    claimed = np.isfinite(m) if model_valid is None else (np.asarray(model_valid, bool) & np.isfinite(m))
    sel = base & claimed
    info = dict(n_mask_valid=int(valid.sum()), n_scored=int(sel.sum()), n_missing=int((base & ~claimed).sum()),
                n_no_snr_ref=int((valid & ~np.isfinite(snr)).sum()))
    resid = d - m
    bsel = sel & (snr < 1.0)
    b, nb = background_offset(np.where(sel, resid, 0.0), sigma, bsel)
    if nb < MIN_BG:
        info["offset_flag"] = f"only {nb} pixels with snr_ref<1: offset set to 0"
        b = 0.0
    info["offset"], info["n_offset_pixels"] = b, nb
    ibin = np.clip(np.digitize(snr, EDGES) - 1, 0, 5)
    rows = []
    for bx in np.unique(box[sel]):
        if bx < 0:
            continue
        s = sel & (box == bx)
        row = dict(box=int(bx), n=np.zeros((2, 6)), chi2=np.zeros((2, 6)), fn=np.zeros(2), f2=np.zeros(2))
        for v, off in enumerate((0.0, b)):
            r = (resid[s] - off) / sigma[s]
            row["n"][v] = np.bincount(ibin[s], minlength=6)
            row["chi2"][v] = np.bincount(ibin[s], weights=r * r, minlength=6)
            fs = s & (ibin >= FRAC_FROM) & (m > 0)
            fr = (resid[fs] - off) / m[fs]
            row["fn"][v], row["f2"][v] = fr.size, float((fr ** 2).sum())
        rows.append(row)
    return rows, info


# ----------------------------------------------------------------------------- aggregation + bootstrap
class Table:
    """Rows of sufficient statistics: one per (exposure, box)."""

    def __init__(self):
        self.meta, self.n, self.chi2, self.fn, self.f2 = [], [], [], [], []

    def add(self, meta, row):
        self.meta.append(dict(meta, box=row["box"]))
        self.n.append(row["n"]); self.chi2.append(row["chi2"]); self.fn.append(row["fn"]); self.f2.append(row["f2"])

    def arrays(self):
        return (np.array(self.n), np.array(self.chi2), np.array(self.fn), np.array(self.f2))


def _stats_from_sums(n, c2, fn, f2):
    """n, c2: (...,6); fn, f2: (...) -> dict of arrays: chi2N per bin, headline, frac rms."""
    with np.errstate(invalid="ignore", divide="ignore"):
        per = c2 / n
        head = c2[..., HEAD_FROM:].sum(-1) / n[..., HEAD_FROM:].sum(-1)
        frac = np.sqrt(f2 / fn)
    return per, head, frac


def aggregate(tab, idx, nboot=N_BOOT, seed=SEED):
    """Pooled statistics over table rows idx with a 68% bootstrap over dither groups.  Returns dict per variant."""
    N, C, FN, F2 = tab.arrays()
    groups = np.array([tab.meta[i]["dither_group"] for i in idx])
    ug, inv = np.unique(groups, return_inverse=True)
    G = len(ug)
    rng = np.random.default_rng(seed)
    out = {"n_rows": int(len(idx)), "n_dither_groups": int(G)}
    counts = rng.multinomial(G, np.ones(G) / G, size=nboot) if G > 1 else None
    for v, name in enumerate(("without_offset", "with_offset")):
        n, c, fn, f2 = N[idx][:, v], C[idx][:, v], FN[idx][:, v], F2[idx][:, v]
        gn = np.zeros((G, 6)); gc = np.zeros((G, 6)); gfn = np.zeros(G); gf2 = np.zeros(G)
        np.add.at(gn, inv, n); np.add.at(gc, inv, c); np.add.at(gfn, inv, fn); np.add.at(gf2, inv, f2)
        per, head, frac = _stats_from_sums(gn.sum(0), gc.sum(0), gfn.sum(), gf2.sum())
        if counts is not None:
            bn, bc, bfn, bf2 = counts @ gn, counts @ gc, counts @ gfn, counts @ gf2
            bper, bhead, bfrac = _stats_from_sums(bn, bc, bfn, bf2)
            q = lambda a: [None if not np.isfinite(x) else float(x) for x in np.nanpercentile(a, [16, 84], axis=0)] if np.isfinite(a).any() else [None, None]
            lo_hi_per = [q(bper[:, k]) for k in range(6)]
            lo_hi_head, lo_hi_frac = q(bhead), q(bfrac)
        else:
            lo_hi_per, lo_hi_head, lo_hi_frac = [[None, None]] * 6, [None, None], [None, None]
        fl = lambda x: None if not np.isfinite(x) else float(x)
        out[name] = dict(
            bins=[dict(snr_ref=BIN_NAMES[k], n_pixels=int(gn.sum(0)[k]), chi2_per_pixel=fl(per[k]), lo=lo_hi_per[k][0], hi=lo_hi_per[k][1]) for k in range(6)],
            headline_chi2_snr_ref_ge5=dict(value=fl(head), lo=lo_hi_head[0], hi=lo_hi_head[1], n_pixels=int(gn.sum(0)[HEAD_FROM:].sum())),
            frac_resid_rms_snr_ref_gt50=dict(value=fl(frac), lo=lo_hi_frac[0], hi=lo_hi_frac[1], n_pixels=int(gfn.sum())))
    return out


def aggregate_levels(tab):
    levels = {"field": {"all": aggregate(tab, np.arange(len(tab.meta)))}}
    for key, label in (("filter", "filter"), ("detector", "detector"), ("box", "box")):
        d = defaultdict(list)
        for i, m in enumerate(tab.meta):
            d[str(m[key])].append(i)
        levels[label] = {k: aggregate(tab, np.array(v)) for k, v in sorted(d.items())}
    d = defaultdict(list)
    for i, m in enumerate(tab.meta):
        d[f'{m["filter"]}/box{m["box"]}'].append(i)
    levels["filter_box"] = {k: aggregate(tab, np.array(v)) for k, v in sorted(d.items())}
    return levels


# ----------------------------------------------------------------------------- driver
def load_sci(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as hl:
        return np.array(hl["SCI"].data, dtype=np.float64)


def score_run(splits, method, split, mask_for, pred_for, data_for, nboot=N_BOOT):
    """Core: splits dict; mask_for(root)->dict of arrays; pred_for(root)->dict or None; data_for(root)->SCI array.  Returns the result dict."""
    ex = {e["rootname"]: e for e in splits["exposures"]}
    held = splits["splits"][split]["heldout"]
    tab, per_exp, missing_files, no_mask = Table(), {}, [], []
    metas = {}
    for root in held:
        e = ex[root]
        mk = mask_for(root)
        if mk is None:
            no_mask.append(root)
            continue
        pr = pred_for(root)
        if pr is None:
            missing_files.append(root)
            continue
        d = data_for(root)
        mv_given = "model_valid" in pr
        rows, info = exposure_rows(d, pr["model"], mk, pr["model_valid"] if mv_given else None)
        info["model_valid_provided"] = bool(mv_given)
        info.update(filter=e["filter"], detector=e["detector"], dither_group=e["dither_group"])
        if "meta" in pr:
            try:
                metas[root] = json.loads(str(pr["meta"]))
            except Exception:
                metas[root] = str(pr["meta"])
        per_exp[root] = info
        for r in rows:
            tab.add(dict(root=root, filter=e["filter"], detector=e["detector"], dither_group=e["dither_group"]), r)
    res = dict(field=splits["field"], method=method, split=split, protocol="v1", n_heldout_listed=len(held), n_scored_exposures=len(per_exp),
               missing_prediction_files=missing_files, missing_mask_files=no_mask, n_bootstrap=nboot,
               bins=BIN_NAMES, per_exposure=per_exp, method_meta=metas)
    ns = sum(i["n_scored"] for i in per_exp.values())
    nm = sum(i["n_missing"] for i in per_exp.values())
    res["pixels"] = dict(scored=int(ns), missing_pixels_in_scored_exposures=int(nm),
                         missing_fraction=float(nm / (ns + nm)) if ns + nm else None,
                         exposures_without_model_valid=sorted(r for r, i in per_exp.items() if not i["model_valid_provided"]),
                         exposures_with_unflagged_offset=sorted(r for r, i in per_exp.items() if "offset_flag" in i))
    res["levels"] = aggregate_levels(tab) if tab.meta else {}
    return res


def cmd_score(a):
    sp = json.load(open(os.path.join(a.root, f"splits_{a.field}.json")))
    ex = {e["rootname"]: e for e in sp["exposures"]}

    def mask_for(r):
        p = os.path.join(a.root, "heldout", a.field, r + ".npz")
        return dict(np.load(p)) if os.path.exists(p) else None

    def pred_for(r):
        p = os.path.join(a.root, "pred", a.field, a.method, a.split, r + ".npz")
        return dict(np.load(p, allow_pickle=False)) if os.path.exists(p) else None

    def data_for(r):
        p = ex[r]["path"]
        if a.data_root:
            p = os.path.join(a.data_root, ex[r]["file"])
        return load_sci(p)

    res = score_run(sp, a.method, a.split, mask_for, pred_for, data_for, a.nboot)
    out = os.path.join(a.root, "scores", a.field, a.method)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"{a.split}.json"), "w"), indent=1)
    print_summary(res)
    cmd_compare(argparse.Namespace(root=a.root, field=a.field, split=a.split))


def fmt(x, lo=None, hi=None, nd=3):
    if x is None:
        return "n/a"
    s = f"{x:.{nd}g}"
    if lo is not None and hi is not None:
        s += f" [{lo:.{nd}g}, {hi:.{nd}g}]"
    return s


def print_summary(res):
    f = res["levels"].get("field", {}).get("all")
    print(f'{res["field"]}/{res["method"]}/{res["split"]}: {res["n_scored_exposures"]}/{res["n_heldout_listed"]} exposures scored; '
          f'missing prediction files {len(res["missing_prediction_files"])}; missing pixel fraction {res["pixels"]["missing_fraction"]}')
    if f:
        for v in ("with_offset", "without_offset"):
            h = f[v]["headline_chi2_snr_ref_ge5"]
            print(f"  {v}: chi2/N (snr_ref>=5) {fmt(h['value'], h['lo'], h['hi'])}; per bin "
                  + ", ".join(f"{b['snr_ref']}: {fmt(b['chi2_per_pixel'])}" for b in f[v]["bins"]))


def cmd_compare(a):
    d = os.path.join(a.root, "scores", a.field)
    files = sorted(glob.glob(os.path.join(d, "*", f"{a.split}.json")))
    rows = [json.load(open(p)) for p in files]
    if not rows:
        print("no scores files for", a.field, a.split)
        return
    md = [f"# {a.field}: held-out split `{a.split}`, comparison across methods", "",
          "chi2/N = mean ((d - m - b)/sigma)^2 per pixel, ideal 1.0. Intervals: 68% bootstrap over dither groups. `b` = scorer background offset.", ""]
    table = []
    for v, title in (("with_offset", "with the scorer's background offset"), ("without_offset", "without the offset")):
        md += [f"## {title}", "", "| method | exposures | missing pixels | chi2/N S/N>=5 (headline) | " + " | ".join(f"chi2/N {b}" for b in BIN_NAMES) + " | frac. resid. rms (S/N>50) |",
               "|" + "---|" * (5 + len(BIN_NAMES))]
        for r in rows:
            f = r["levels"].get("field", {}).get("all", {}).get(v)
            if not f:
                md.append(f'| {r["method"]} | {r["n_scored_exposures"]}/{r["n_heldout_listed"]} | - | no pixels scored |' + " |" * (len(BIN_NAMES) + 2))
                continue
            h, fr = f["headline_chi2_snr_ref_ge5"], f["frac_resid_rms_snr_ref_gt50"]
            mf = r["pixels"]["missing_fraction"]
            md.append(f'| {r["method"]} | {r["n_scored_exposures"]}/{r["n_heldout_listed"]} | {0 if mf is None else mf:.2%} | {fmt(h["value"], h["lo"], h["hi"])} | '
                      + " | ".join(fmt(b["chi2_per_pixel"], b["lo"], b["hi"]) for b in f["bins"]) + f' | {fmt(fr["value"], fr["lo"], fr["hi"])} |')
            table.append(dict(method=r["method"], variant=v, headline=h, frac_rms=fr, bins=f["bins"], missing_fraction=mf,
                              n_scored_exposures=r["n_scored_exposures"], n_heldout=r["n_heldout_listed"]))
        md.append("")
    md += ["## headline chi2/N (S/N>=5, with offset) per filter", ""]
    filters = sorted({k for r in rows for k in r["levels"].get("filter", {})})
    md += ["| method | " + " | ".join(filters) + " |", "|" + "---|" * (1 + len(filters))]
    for r in rows:
        cells = []
        for fl in filters:
            f = r["levels"].get("filter", {}).get(fl, {}).get("with_offset")
            h = f["headline_chi2_snr_ref_ge5"] if f else None
            cells.append(fmt(h["value"], h["lo"], h["hi"]) if h else "n/a")
        md.append(f'| {r["method"]} | ' + " | ".join(cells) + " |")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, f"comparison_{a.split}.md"), "w").write("\n".join(md) + "\n")
    json.dump(table, open(os.path.join(d, f"comparison_{a.split}.json"), "w"), indent=1)
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--field", required=True)
    s.add_argument("--method", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--root", default=ROOT)
    s.add_argument("--data-root", default="", help="directory holding the _cal files (default: path in the splits file)")
    s.add_argument("--nboot", type=int, default=N_BOOT)
    s.set_defaults(fn=cmd_score)
    c = sub.add_parser("compare")
    c.add_argument("--field", required=True)
    c.add_argument("--split", required=True)
    c.add_argument("--root", default=ROOT)
    c.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
