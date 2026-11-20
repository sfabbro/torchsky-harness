#!/usr/bin/env python3
"""score_paired.py -- companion to score.py (evaluation protocol v1). Same inputs, same conventions, two new commands.

    python score_paired.py paired  --field SMACS --split dither --methods fieldop_fix_200 fieldop --reference stage3_blot
    python score_paired.py byscale --field SMACS --split dither --methods fieldop_fix_200 fieldop stage3_blot
Common options: --root (protocol folder, default as score.py), --data-root (folder with the _cal files, default: path in the splits file),
--out (default <root>/scores_paired), --nboot, --seed.  Needs numpy, scipy (distance transform for the edge taper), astropy (reading the _cal files) and matplotlib (figures) only.
Put this file next to score.py (it imports it, so the residuals are identical to the ones score.py uses).

A. paired.  Two methods are scored on the SAME pixels of the SAME exposures, so the difference of their scores has a much narrower
   interval than the two separate intervals suggest.  For each method M against the reference R:
     diff = chi2/N(M) - chi2/N(R), both on identical pixels, per exposure, per filter, per S/N bin of the official mosaic (same bins as score.py)
   and pooled (headline = S/N >= 5).  NEGATIVE means M is better (lower chi2/N) than R.  Intervals: bootstrap over dither groups (the same
   groups, the same resamples as score.py uses for its separate intervals); for each resample the difference of the pooled scores is computed.
   Reported: 68% and 95% intervals and the fraction of resamples in which M is better (diff < 0).  The pixel-wise form is the mean over
   pixels of ((d-m_M-b_M)^2 - (d-m_R-b_R)^2)/sigma^2, accumulated pixel by pixel (equal to the pooled difference when the pixel sets are
   identical; kept as a check and for the share of pixels where M wins).
B. byscale.  Residual power by angular scale.  r = (d - m - b)/sigma on the scored cut-out of each (exposure, box), masked pixels set to 0.
   Band power  P_band = mean over the Fourier modes of the band of |FFT(w r)|^2 / sum(w^2),  w = scored-pixel mask x cosine taper inside the box edge.
   Dividing by sum(w^2) is the correction for the mask's own power: white noise of unit variance gives P = 1 for any mask.
   Bands: octaves of wavelength in pixels, [2,4), [4,8), ... up to the cut-out size.  Three quantities (see SCORE_PAIRED_REPORT.md):
     auto   P_i of each method's residual;  cross  Re(F_i F_j*)/sum(w^2) between the residuals of two methods on the same exposure;
     diff   power of the difference of the two models, (m_j + b_j - m_i - b_i)/sigma  (contains no data noise).
   Exact identity: diff = auto_i + auto_j - 2 cross.
APPROXIMATIONS (also in the report)
   P1  mask correction exact only for a flat spectrum; poor where the fill factor is low or holes are large, and for steep red spectra (leakage).
   P2  masked pixels are assumed to have the same power as scored pixels; the edge taper (10% of the box side) discards power near the box edge.
       The cut-out is the bounding rectangle of the box; the box is a square rotated on the detector, so about half of the rectangle is outside it (weight 0).
   P3  paired: identical pixels = intersection of the pixels claimed by both methods; each method keeps its own scorer offset b (as in score.py).
   P4  bootstrap: dither groups are resampled, pixels within a group are treated as independent samples of the same sums (as in score.py, S2).
   P5  byscale: pixels of all S/N are included (option --snr-min); the common pixel set of all requested methods is used for every method.
   P6  byscale pools exposures by weighting each with sum(w^2) (effective number of pixels); bands with fewer than 4 modes in an exposure are dropped.
   P7  --level rate (16 Nov 2026): data = _rate SCI, sigma = _rate ERR x protocol noise_factor, pixels = protocol mask (--dq-zero on the _rate
       flags); calibrated-unit predictions x C = rate SCI / cal SCI (conversion_map: exact where both are finite and non-zero; rejected
       pixels that are scored get the 5 x 5 local median of C); predictions whose meta says units = 'rate' are used as they are; the
       scorer's offset b is a constant in rate units. The rate-level sigma has no flat-field variance and comes from the current files,
       while the protocol sigma was built from the earlier _cal files (1.0125 x the current cal ERR x noise_factor on NRCB1): a
       calibrated-unit method scores ~2-3 per cent higher at rate level (12-exposure sample, S/N 5-50: +0.029 stage3_blot, +0.034
       fieldop_fix_200); with the protocol sigma carried over (C x sigma) the two levels agree to 4e-16 (tests/check_rate_scoring.py).
       Compare rate-trained and calibrated-trained fits at the SAME level (rate, paired), not each at its own level.
   P8  --offset-rank METHOD (16 Nov 2026): the scorer's background offset is fitted on the 2000 scored pixels with the lowest value of
       METHOD's prediction (a prediction built from the training exposures only), not on the lowest snr_ref of the official mosaic. The
       official mosaic contains the held-out exposure, so its faintest pixels are those where that exposure's own noise is low: a model
       that does not follow the noise then receives a spurious negative offset. The ranking prediction is box-averaged over
       --offset-rank-smooth arcsec (default 1) so that it carries almost none of the training exposures' pixel noise either: ranking by
       an unsmoothed training mosaic would bias noise-following models the other way. Approximation: nominal pixel scales.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import score as S
except ImportError:                                   # development layout
    sys.path.insert(0, os.path.join(HERE, "protocol_ref"))
    import score as S

ROOT = S.ROOT
EDGES, BIN_NAMES = S.EDGES, S.BIN_NAMES
HEAD_FROM = S.HEAD_FROM
NB = len(BIN_NAMES)            # 6 bins; column NB of the sums is the headline (bins >= HEAD_FROM)
VARIANTS = ("without_offset", "with_offset")


# =============================================================================== one exposure, one method
def pixel_state(d, model, mask, model_valid=None):
    """Pixel set and residual with exactly the conventions of score.exposure_rows (offset taken from it, not recomputed)."""
    _, info = S.exposure_rows(d, model, mask, model_valid)       # offset b and flags: score.py's own code
    valid = mask["valid"].astype(bool)
    sigma = mask["sigma"].astype(np.float64)
    snr = mask["snr_ref"].astype(np.float64)
    d = np.asarray(d, np.float64)
    m = np.asarray(model, np.float64)
    base = valid & np.isfinite(d) & np.isfinite(sigma) & (sigma > 0) & np.isfinite(snr)
    claimed = np.isfinite(m) if model_valid is None else (np.asarray(model_valid, bool) & np.isfinite(m))
    sel = base & claimed
    b = float(info["offset"])
    if mask.get("offset_rank") is not None:
        # P8 (16 Nov 2026): the protocol chooses the offset pixels by the lowest snr_ref of the official mosaic, which is built from ALL
        # exposures, the held-out one included; that selects pixels where the held-out exposure's own noise is low, so a model that does
        # not follow the noise is given a negative offset (measured on SMACS: -0.45 sigma on the faintest 2 per cent for a positive item
        # sky against -0.13 for a grid sky, and no trend at all when the pixels are ranked by a prediction that shares no noise with the
        # held-out exposure). Here the offset pixels are the FALLBACK_N scored pixels with the lowest value of a prediction made from the
        # training exposures only (--offset-rank METHOD); binning by snr_ref is unchanged.
        rank = np.asarray(mask["offset_rank"], np.float64)
        ok = sel & np.isfinite(rank)
        if ok.sum() >= S.FALLBACK_N:
            thr = np.partition(rank[ok], S.FALLBACK_N - 1)[S.FALLBACK_N - 1]
            b, nb = S.background_offset(np.where(sel, d - m, 0.0), sigma, ok & (rank <= thr))
            b = float(b)
            info = dict(info, offset=b, n_offset_pixels=int(nb), offset_rank="independent")
    return dict(sel=sel, resid=d - m, b=b, info=info, model=m)


def channel_of(detector):
    return "long" if str(detector).upper().endswith("LONG") else "short"


# =============================================================================== A. paired
class PairTable:
    """One row per (exposure, box, method): sums per S/N bin plus headline column, for both offset variants."""

    def __init__(self):
        self.meta, self.n, self.a, self.b, self.d, self.w, self.t = [], [], [], [], [], [], []

    def add(self, meta, row):
        self.meta.append(meta)
        for k in ("n", "a", "b", "d", "w", "t"):
            getattr(self, k).append(row[k])

    def arr(self, k):
        return np.array(getattr(self, k))


def _with_headline(x):
    """(..., 6) -> (..., 7): append the headline column (bins >= HEAD_FROM)."""
    return np.concatenate([x, x[..., HEAD_FROM:].sum(-1, keepdims=True)], axis=-1)


def paired_rows(sm, sr, mask):
    """Rows (one per box) of paired sums for method state sm against reference state sr, on the pixels claimed by both."""
    sigma = mask["sigma"].astype(np.float64)
    snr = mask["snr_ref"].astype(np.float64)
    box = mask["box_id"]
    common = sm["sel"] & sr["sel"]
    ibin = np.clip(np.digitize(snr, EDGES) - 1, 0, NB - 1)
    rows = []
    for bx in np.unique(box[common]):
        if bx < 0:
            continue
        s = common & (box == bx)
        row = dict(box=int(bx), n=np.zeros((2, NB + 1)), a=np.zeros((2, NB + 1)), b=np.zeros((2, NB + 1)), d=np.zeros((2, NB + 1)),
                   w=np.zeros((2, NB + 1)), t=np.zeros((2, NB + 1)),
                   n_dropped_m=int((sm["sel"] & (box == bx)).sum() - s.sum()), n_dropped_r=int((sr["sel"] & (box == bx)).sum() - s.sum()))
        for v in range(2):
            om, orf = (0.0, 0.0) if v == 0 else (sm["b"], sr["b"])
            x1, x2, sg = sm["resid"][s] - om, sr["resid"][s] - orf, sigma[s]
            r1, r2 = x1 / sg, x2 / sg                                  # same expressions as score.exposure_rows
            ib = ibin[s]

            def bc(wts):
                return _with_headline(np.bincount(ib, weights=wts, minlength=NB).astype(np.float64))
            row["n"][v] = _with_headline(np.bincount(ib, minlength=NB).astype(np.float64))
            row["a"][v], row["b"][v] = bc(r1 * r1), bc(r2 * r2)
            row["d"][v] = bc((x1 * x1 - x2 * x2) / (sg * sg))           # pixel-wise form
            row["w"][v] = bc((r1 * r1 < r2 * r2).astype(np.float64))
            row["t"][v] = bc((r1 * r1 == r2 * r2).astype(np.float64))
        rows.append(row)
    return rows


def _pct(x, q):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return [None, None] if x.size == 0 else [float(v) for v in np.percentile(x, q)]


def _summ(point, boots):
    f = lambda v: None if v is None or not np.isfinite(v) else float(v)
    out = dict(value=f(point))
    if boots is None:
        out.update(lo68=None, hi68=None, lo95=None, hi95=None, frac_better=None, frac_equal=None)
        return out
    bb = boots[np.isfinite(boots)]
    out["lo68"], out["hi68"] = _pct(bb, [16, 84])
    out["lo95"], out["hi95"] = _pct(bb, [2.5, 97.5])
    out["frac_better"] = float((bb < 0).mean()) if bb.size else None
    out["frac_equal"] = float((bb == 0).mean()) if bb.size else None
    return out


def aggregate_pair(tab, idx, nboot, seed):
    """Pooled paired statistics over rows idx, bootstrap over dither groups (same draws as score.aggregate for the same rows)."""
    idx = np.asarray(idx)
    N, A, B, D, W, T = (tab.arr(k)[idx] for k in ("n", "a", "b", "d", "w", "t"))
    groups = np.array([tab.meta[i]["dither_group"] for i in idx])
    ug, inv = np.unique(groups, return_inverse=True)
    G = len(ug)
    counts = np.random.default_rng(seed).multinomial(G, np.ones(G) / G, size=nboot) if G > 1 else None
    out = dict(n_rows=int(len(idx)), n_dither_groups=int(G))
    names = BIN_NAMES + ["headline S/N>=5"]
    for v, vn in enumerate(VARIANTS):
        gs = {}
        for k, X in (("n", N), ("a", A), ("b", B), ("d", D), ("w", W), ("t", T)):
            g = np.zeros((G, NB + 1))
            np.add.at(g, inv, X[:, v])
            gs[k] = g

        def ratios(sums):
            with np.errstate(invalid="ignore", divide="ignore"):
                n = sums["n"]
                return (sums["a"] / n, sums["b"] / n, (sums["a"] - sums["b"]) / n, sums["d"] / n, sums["w"] / n)
        pt = ratios({k: g.sum(0) for k, g in gs.items()})
        bt = ratios({k: counts @ g for k, g in gs.items()}) if counts is not None else [None] * 5
        entries = []
        for k in range(NB + 1):
            e = dict(snr_ref=names[k], n_pixels=int(gs["n"].sum(0)[k]))
            e["score_method"] = _summ(pt[0][k], bt[0][:, k] if counts is not None else None)
            e["score_reference"] = _summ(pt[1][k], bt[1][:, k] if counts is not None else None)
            e["diff"] = _summ(pt[2][k], bt[2][:, k] if counts is not None else None)
            e["diff_pixelwise"] = _summ(pt[3][k], bt[3][:, k] if counts is not None else None)
            e["fraction_of_pixels_where_method_has_smaller_r2"] = None if not np.isfinite(pt[4][k]) else float(pt[4][k])
            e["fraction_of_pixels_tied"] = float(gs["t"].sum(0)[k] / gs["n"].sum(0)[k]) if gs["n"].sum(0)[k] > 0 else None
            # what the unpaired comparison would give: both intervals from the same resamples, treated as independent
            if counts is not None:
                wm = e["score_method"]["hi68"] - e["score_method"]["lo68"] if e["score_method"]["hi68"] is not None else None
                wr = e["score_reference"]["hi68"] - e["score_reference"]["lo68"] if e["score_reference"]["hi68"] is not None else None
                wp = (e["diff"]["hi68"] - e["diff"]["lo68"]) if e["diff"]["hi68"] is not None else None
                e["width68_method_alone"], e["width68_reference_alone"], e["width68_paired_diff"] = wm, wr, wp
                e["width68_if_independent"] = None if wm is None or wr is None else float(math.hypot(wm, wr))
            entries.append(e)
        out[vn] = dict(bins=entries[:NB], headline=entries[NB])
    return out


def paired_levels(tab, nboot, seed):
    lv = {"field": {"all": aggregate_pair(tab, np.arange(len(tab.meta)), nboot, seed)}}
    for key in ("filter", "detector", "channel", "box"):
        d = defaultdict(list)
        for i, m in enumerate(tab.meta):
            d[str(m[key])].append(i)
        lv[key] = {k: aggregate_pair(tab, v, nboot, seed) for k, v in sorted(d.items())}
    d = defaultdict(list)
    for i, m in enumerate(tab.meta):
        d["%s/box%s" % (m["filter"], m["box"])].append(i)
    lv["filter_box"] = {k: aggregate_pair(tab, v, nboot, seed) for k, v in sorted(d.items())}
    return lv


def per_exposure_table(tab):
    out = []
    fl = lambda arr: [None if not np.isfinite(x) else float(x) for x in arr]
    for i, m in enumerate(tab.meta):
        r = dict(m)
        for v, vn in enumerate(VARIANTS):
            n, a, b, d = (tab.arr(k)[i, v] for k in ("n", "a", "b", "d"))
            with np.errstate(invalid="ignore", divide="ignore"):
                r[vn] = dict(n_pixels=[int(x) for x in n], score_method=fl(a / n), score_reference=fl(b / n), diff=fl((a - b) / n),
                             diff_pixelwise=fl(d / n))
        out.append(r)
    return out


def paired_run(splits, split, methods, reference, mask_for, pred_for, data_for, nboot=S.N_BOOT, seed=S.SEED, log=print):
    """mask_for(root)->dict; pred_for(method, root)->dict or None; data_for(root)->SCI array."""
    ex = {e["rootname"]: e for e in splits["exposures"]}
    held = splits["splits"][split]["heldout"]
    tabs = {m: PairTable() for m in methods}
    missing = {m: [] for m in methods}
    n_exp, no_mask, no_ref = 0, [], []
    for root in held:
        e = ex[root]
        mk = mask_for(root)
        if mk is None:
            no_mask.append(root)
            continue
        pr_ref = pred_for(reference, root)
        if pr_ref is None:
            no_ref.append(root)
            continue
        d = data_for(root)
        sr = pixel_state(d, pr_ref["model"], mk, pr_ref["model_valid"] if "model_valid" in pr_ref else None)
        n_exp += 1
        for m in methods:
            pm = pred_for(m, root)
            if pm is None:
                missing[m].append(root)
                continue
            sm = pixel_state(d, pm["model"], mk, pm["model_valid"] if "model_valid" in pm else None)
            for row in paired_rows(sm, sr, mk):
                tabs[m].add(dict(root=root, filter=e["filter"], detector=e["detector"], channel=channel_of(e["detector"]),
                                 dither_group=e["dither_group"], box=row["box"], n_dropped_method=row["n_dropped_m"],
                                 n_dropped_reference=row["n_dropped_r"]), row)
        log("  %s done" % root)
    res = dict(field=splits["field"], split=split, reference=reference, protocol="v1", n_heldout_listed=len(held), n_exposures_with_reference=n_exp,
               missing_mask_files=no_mask, missing_reference_files=no_ref, n_bootstrap=nboot, seed=seed, bins=BIN_NAMES,
               meaning="diff = chi2/N(method) - chi2/N(reference) on identical pixels; negative = the method is better", methods={})
    for m in methods:
        t = tabs[m]
        res["methods"][m] = dict(missing_prediction_files=missing[m], n_rows=len(t.meta),
                                 n_pixels_dropped_not_claimed_by_both=int(sum(x["n_dropped_method"] + x["n_dropped_reference"] for x in t.meta)),
                                 levels=paired_levels(t, nboot, seed) if t.meta else {}, per_exposure=per_exposure_table(t))
    return res


def _fmt(x, nd=3):
    return "n/a" if x is None else ("%.*g" % (nd, x))


def _ci(e, nd=3):
    d = e["diff"]
    if d["lo68"] is None:
        return _fmt(d["value"], nd)
    return "%s [%s, %s]" % (_fmt(d["value"], nd), _fmt(d["lo68"], nd), _fmt(d["hi68"], nd))


def paired_markdown(res):
    L = ["A negative number means the method has a LOWER chi2/N than the reference `%s` on the same pixels, i.e. it predicts the held-out exposures better; "
         "a positive number means worse." % res["reference"], "",
         "# %s, split `%s`: paired difference of chi2/N against `%s`" % (res["field"], res["split"], res["reference"]), "",
         "diff = chi2/N(method) - chi2/N(reference), computed on identical pixels; chi2/N = mean ((d - m - b)/sigma)^2 as in score.py. Headline = pixels with S/N of the "
         "official mosaic >= 5. Intervals: bootstrap over dither groups (%d resamples, each resample gives the difference of the pooled scores); "
         "'68%%' and '95%%' are central intervals, 'P(better)' is the fraction of resamples with diff < 0. Offset variant: with the scorer's background offset "
         "(without it: see the JSON)." % res["n_bootstrap"], ""]
    v = "with_offset"
    L += ["## pooled over the field, headline (S/N >= 5)", "",
          "| method | exposure rows | dither groups | chi2/N method | chi2/N reference | diff, 68% interval | 95% interval | P(better) | pixel-wise form (68%) | width 68%: paired vs independent |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        f = r["levels"].get("field", {}).get("all")
        if not f:
            L.append("| %s | no pixels |" % m)
            continue
        h = f[v]["headline"]
        pw = h["diff_pixelwise"]
        L.append("| %s | %d | %d | %s | %s | %s | [%s, %s] | %s | %s [%s, %s] | %s vs %s |" % (
            m, f["n_rows"], f["n_dither_groups"], _fmt(h["score_method"]["value"]), _fmt(h["score_reference"]["value"]), _ci(h),
            _fmt(h["diff"]["lo95"]), _fmt(h["diff"]["hi95"]), _fmt(h["diff"]["frac_better"], 2), _fmt(pw["value"]), _fmt(pw["lo68"]), _fmt(pw["hi68"]),
            _fmt(h.get("width68_paired_diff")), _fmt(h.get("width68_if_independent"))))
    L += ["", "## per filter, headline (diff with 68% interval; P(better) and number of dither groups in brackets; with 1 group the interval is not defined)", ""]
    filters = sorted({k for r in res["methods"].values() for k in r["levels"].get("filter", {})})
    L += ["| method | " + " | ".join(filters) + " |", "|" + "---|" * (1 + len(filters))]
    for m, r in res["methods"].items():
        cells = []
        for fl in filters:
            f = r["levels"].get("filter", {}).get(fl)
            cells.append("n/a" if not f else "%s (%s; %d groups)" % (_ci(f[v]["headline"]), _fmt(f[v]["headline"]["diff"]["frac_better"], 2), f["n_dither_groups"]))
        L.append("| %s | %s |" % (m, " | ".join(cells)))
    L += ["", "## per S/N bin of the official mosaic, pooled over the field (diff with 68% interval)", ""]
    L += ["| method | " + " | ".join(BIN_NAMES) + " |", "|" + "---|" * (1 + NB)]
    for m, r in res["methods"].items():
        f = r["levels"].get("field", {}).get("all")
        if f:
            L.append("| %s | %s |" % (m, " | ".join(_ci(b) for b in f[v]["bins"])))
    L += ["", "## per detector channel, headline", ""]
    chans = sorted({k for r in res["methods"].values() for k in r["levels"].get("channel", {})})
    L += ["| method | " + " | ".join(chans) + " |", "|" + "---|" * (1 + len(chans))]
    for m, r in res["methods"].items():
        L.append("| %s | %s |" % (m, " | ".join(_ci(r["levels"]["channel"][c][v]["headline"]) if c in r["levels"].get("channel", {}) else "n/a" for c in chans)))
    L += ["", "## per exposure, headline diff (point values, no interval: one exposure is one dither group)", ""]
    for m, r in res["methods"].items():
        L += ["### %s" % m, "", "| exposure | box | filter | detector | pixels S/N>=5 | chi2/N method | chi2/N reference | diff |", "|---|---|---|---|---|---|---|---|"]
        for x in r["per_exposure"]:
            q = x[v]
            L.append("| %s | %s | %s | %s | %d | %s | %s | %s |" % (x["root"], x["box"], x["filter"], x["detector"], q["n_pixels"][NB],
                                                                  _fmt(q["score_method"][NB]), _fmt(q["score_reference"][NB]), _fmt(q["diff"][NB])))
        L.append("")
    return "\n".join(L) + "\n"


# =============================================================================== B. by scale
def box_taper(inbox, frac):
    """Edge taper: cosine ramp over frac*sqrt(box area) pixels from the box edge inwards (distance to the nearest pixel outside the box).
    Only the box edge is tapered, holes inside the box (bad pixels, unclaimed pixels) stay sharp.  frac = 0: no taper."""
    if frac <= 0:
        return inbox.astype(np.float64)
    from scipy.ndimage import distance_transform_edt
    pad = np.pad(inbox, 1)
    dist = distance_transform_edt(pad)[1:-1, 1:-1] - 0.5
    d0 = max(frac * math.sqrt(float(inbox.sum())), 1.0)
    t = 0.5 * (1 - np.cos(np.pi * np.clip(dist / d0, 0.0, 1.0)))
    return np.where(inbox, t, 0.0)


NBANDS = 11                                  # wavelengths 2 ... 4096 pixels
BAND_LO = np.array([2.0 ** (j + 1) for j in range(NBANDS)])
BAND_HI = BAND_LO * 2
MIN_MODES = 4
_BAND_CACHE = {}


def band_map(ny, nx):
    """Band index of every FFT mode (-1 = excluded: DC, or wavelength < 2 px).  Band j: wavelength in [2^(j+1), 2^(j+2)) pixels."""
    key = (ny, nx)
    if key not in _BAND_CACHE:
        k = np.hypot(*np.meshgrid(np.fft.fftfreq(nx), np.fft.fftfreq(ny)))
        with np.errstate(divide="ignore"):
            lam = np.where(k > 0, 1.0 / np.where(k > 0, k, 1.0), np.inf)
        j = np.floor(np.log2(np.where(np.isfinite(lam), lam, 1.0)) + 1e-9).astype(int) - 1
        j[(~np.isfinite(lam)) | (lam < 2.0 - 1e-9) | (j < 0) | (j >= NBANDS)] = -1
        _BAND_CACHE[key] = j
    return _BAND_CACHE[key]


def band_average(p2d, bm):
    sel = bm >= 0
    num = np.bincount(bm[sel], weights=p2d[sel], minlength=NBANDS)[:NBANDS]
    cnt = np.bincount(bm[sel], minlength=NBANDS)[:NBANDS].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / cnt
    out[cnt < MIN_MODES] = np.nan
    return out, cnt


def cutout_records(states, mask, methods, snr_min, taper, min_size=16, min_fill=0.05):
    """Per box: band powers for each method (auto) and each method pair (cross, diff).  states: {method: pixel_state}."""
    sigma = mask["sigma"].astype(np.float64)
    snr = mask["snr_ref"].astype(np.float64)
    box = mask["box_id"]
    mask["valid"].astype(bool)
    common = np.ones(sigma.shape, bool)
    for m in methods:
        common &= states[m]["sel"]
    common &= snr >= snr_min
    recs = []
    for bx in np.unique(box):
        if bx < 0:
            continue
        inbox = box == bx                      # the whole box, bad pixels included (they are holes, scored pixels are in `common`)
        ys, xs = np.nonzero(inbox)
        if ys.size == 0:
            continue
        sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        ny, nx = inbox[sl].shape
        if min(ny, nx) < min_size:
            continue
        msk = (common & inbox)[sl]
        w = msk * box_taper(inbox[sl], taper)
        W = float((w * w).sum())
        fill = float(msk.sum() / inbox.sum())                  # fraction of the box area that is scored (the rest is holes)
        if W <= 0 or fill < min_fill:
            continue
        sg = np.where(msk, sigma[sl], 1.0)
        bm = band_map(ny, nx)
        F, Fm = {}, {}
        for m in methods:
            st = states[m]
            F[m] = np.fft.fft2(np.where(msk, (st["resid"][sl] - st["b"]) / sg, 0.0) * w)
            Fm[m] = np.fft.fft2(np.where(msk, (st["model"][sl] + st["b"]) / sg, 0.0) * w)
        rec = dict(box=int(bx), shape=[int(ny), int(nx)], W=W, fill=fill, bbox_fill=float(msk.mean()), auto={}, cross={}, diff={})
        for m in methods:
            rec["auto"][m], _ = band_average(np.abs(F[m]) ** 2 / W, bm)
        for a, b in combinations(methods, 2):
            rec["cross"]["%s|%s" % (a, b)], _ = band_average((F[a] * np.conj(F[b])).real / W, bm)
            rec["diff"]["%s|%s" % (a, b)], _ = band_average(np.abs(Fm[b] - Fm[a]) ** 2 / W, bm)
        recs.append(rec)
    return recs


def aggregate_bands(recs, idx, methods, nboot, seed):
    """Mean of band powers over the cut-outs idx, each weighted by sum(w^2); bootstrap over dither groups."""
    groups = np.array([recs[i]["dither_group"] for i in idx])
    ug, inv = np.unique(groups, return_inverse=True)
    G = len(ug)
    counts = np.random.default_rng(seed).multinomial(G, np.ones(G) / G, size=nboot) if G > 1 else None
    pairs = ["%s|%s" % p for p in combinations(methods, 2)]
    items = [("auto", m) for m in methods] + [("cross", p) for p in pairs] + [("diff", p) for p in pairs]
    out = dict(n_cutouts=int(len(idx)), n_dither_groups=int(G),
               bands=[dict(lam_lo=float(BAND_LO[j]), lam_hi=float(BAND_HI[j]), lam_mid=float(BAND_LO[j] * math.sqrt(2))) for j in range(NBANDS)],
               auto={}, cross={}, diff={})
    W = np.array([recs[i]["W"] for i in idx])
    valid_any = np.zeros(NBANDS, int)
    cl = lambda a: [None if not np.isfinite(x) else float(x) for x in a]
    for kind, key in items:
        P = np.array([recs[i][kind][key] for i in idx])                     # (n, NBANDS), NaN where band not usable
        ok = np.isfinite(P)
        gnum, gden = np.zeros((G, NBANDS)), np.zeros((G, NBANDS))
        np.add.at(gnum, inv, np.where(ok, P, 0.0) * W[:, None])
        np.add.at(gden, inv, ok * W[:, None])
        with np.errstate(invalid="ignore", divide="ignore"):
            pt = gnum.sum(0) / gden.sum(0)
            if counts is not None:
                bt = (counts @ gnum) / (counts @ gden)
                lo, hi = np.nanpercentile(bt, 16, axis=0), np.nanpercentile(bt, 84, axis=0)
            else:
                lo = hi = np.full(NBANDS, np.nan)
        out[kind][key] = dict(value=cl(pt), lo68=cl(lo), hi68=cl(hi))
        valid_any = np.maximum(valid_any, ok.sum(0))
    out["n_cutouts_per_band"] = [int(x) for x in valid_any]
    out["mean_fill_factor"] = float(np.mean([recs[i]["fill"] for i in idx]))
    return out


def byscale_run(splits, split, methods, mask_for, pred_for, data_for, nboot=S.N_BOOT, seed=S.SEED, snr_min=0.0, taper=0.1, log=print):
    ex = {e["rootname"]: e for e in splits["exposures"]}
    held = splits["splits"][split]["heldout"]
    recs, skipped = [], []
    for root in held:
        e = ex[root]
        mk = mask_for(root)
        if mk is None:
            skipped.append((root, "no mask"))
            continue
        prs = {m: pred_for(m, root) for m in methods}
        lack = [m for m in methods if prs[m] is None]
        if lack:
            skipped.append((root, "missing prediction for " + ",".join(lack)))
            continue
        d = data_for(root)
        states = {m: pixel_state(d, prs[m]["model"], mk, prs[m]["model_valid"] if "model_valid" in prs[m] else None) for m in methods}
        for r in cutout_records(states, mk, methods, snr_min, taper):
            r.update(root=root, filter=e["filter"], detector=e["detector"], channel=channel_of(e["detector"]), dither_group=e["dither_group"])
            recs.append(r)
        log("  %s done" % root)
    res = dict(field=splits["field"], split=split, methods=methods, n_bootstrap=nboot, seed=seed, snr_min=snr_min, taper_fraction_per_side=taper,
               units="band power of r=(d-m-b)/sigma; white noise of unit variance = 1.0 in every band; wavelength in detector pixels",
               skipped=skipped, n_cutouts=len(recs), levels={})
    if recs:
        groups = defaultdict(list)
        for i, r in enumerate(recs):
            groups["all"].append(i)
            groups["filter:" + r["filter"]].append(i)
            groups["channel:" + r["channel"]].append(i)
            groups["filter:%s,channel:%s" % (r["filter"], r["channel"])].append(i)
        for k in sorted(groups):
            res["levels"][k] = aggregate_bands(recs, groups[k], methods, nboot, seed)
        cl = lambda d: {m: [None if not np.isfinite(x) else float(x) for x in v] for m, v in d.items()}
        res["per_cutout"] = [dict(root=r["root"], box=r["box"], filter=r["filter"], detector=r["detector"], shape=r["shape"], fill_factor=r["fill"],
                                  sum_w2=r["W"], bbox_fill_factor=r["bbox_fill"], auto=cl(r["auto"]), cross=cl(r["cross"]), diff=cl(r["diff"])) for r in recs]
    return res


def byscale_markdown(res):
    L = ["# %s, split `%s`: residual power by angular scale" % (res["field"], res["split"]), "",
         "Band power of r = (d - m - b)/sigma on the scored cut-out (masked pixels = 0, divided by the mean-square mask). White noise of unit variance = 1.0 "
         "in every band. Wavelength bands in detector pixels (short channel 0.031 arcsec, long 0.063 arcsec). Intervals: 68%% bootstrap over dither groups "
         "(%d resamples). `auto` = power of each method's residual; `cross` = real part of the cross-power of two methods' residuals on the same exposure "
         "(still contains the data noise, which is common to both); `diff` = power of the difference of the two models (no data noise); "
         "exact identity diff = auto_1 + auto_2 - 2 cross. Pixels: S/N >= %g, common to all methods. Bands with fewer than 4 Fourier modes in a cut-out are dropped "
         "for that cut-out." % (res["n_bootstrap"], res["snr_min"]), ""]
    for key, lv in res["levels"].items():
        if key.startswith("filter:") and "channel:" in key:
            continue
        L += ["## %s (%d cut-outs, %d dither groups, mean scored fraction of the box area %.2f)" % (key, lv["n_cutouts"], lv["n_dither_groups"], lv["mean_fill_factor"]), ""]
        cols = [("auto", k) for k in lv["auto"]] + [("diff", k) for k in lv["diff"]] + [("cross", k) for k in lv["cross"]]
        L += ["| wavelength (px) | cut-outs | " + " | ".join("%s %s" % c for c in cols) + " |", "|" + "---|" * (2 + len(cols))]
        for j, b in enumerate(lv["bands"]):
            if lv["n_cutouts_per_band"][j] == 0:
                continue
            cells = []
            for kind, k in cols:
                q = lv[kind][k]
                cells.append("n/a" if q["value"][j] is None else "%s [%s, %s]" % (_fmt(q["value"][j]), _fmt(q["lo68"][j]), _fmt(q["hi68"][j])))
            L.append("| %g-%g | %d | %s |" % (b["lam_lo"], b["lam_hi"], lv["n_cutouts_per_band"][j], " | ".join(cells)))
        L.append("")
    return "\n".join(L) + "\n"


def byscale_figures(res, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
    methods = res["methods"]
    col = {m: pal[i % len(pal)] for i, m in enumerate(methods)}
    filters = sorted(k.split(":", 1)[1] for k in res["levels"] if k.startswith("filter:") and "," not in k)

    def series(q):
        return np.array([np.nan if v is None else v for v in q["value"]])

    def panel(ax, lv, kinds):
        lam = np.array([b["lam_mid"] for b in lv["bands"]])
        ok = np.array(lv["n_cutouts_per_band"]) > 0
        if "auto" in kinds:
            for m in methods:
                ax.plot(lam[ok], series(lv["auto"][m])[ok], "-o", ms=3, color=col[m], label=m)
        for kind, ls in (("diff", "--"), ("cross", ":")):
            if kind in kinds:
                for i, (k, q) in enumerate(lv[kind].items()):
                    y = series(q)
                    y = np.where(y > 0, y, np.nan)
                    ax.plot(lam[ok], y[ok], ls, marker="s", ms=3, color=pal[(len(methods) + i) % len(pal)],
                            label="%s %s" % (kind, k.replace("|", " vs ")))
        ax.axhline(1.0, color="0.5", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")

    def figure(name, keys, kinds, titles, nc, ylabel):
        n = len(keys)
        nr = int(math.ceil(n / float(nc)))
        fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.4 * nr + 0.6), squeeze=False, sharex=True)
        for ax, key, ttl in zip(axs.ravel(), keys, titles):
            panel(ax, res["levels"][key], kinds)
            ax.set_title(ttl, fontsize=10)
        for ax in axs.ravel()[n:]:
            ax.axis("off")
        for ax in axs[:, 0]:
            ax.set_ylabel(ylabel)
        for ax in axs[-1, :]:
            ax.set_xlabel("wavelength (detector pixels, band centre)")
        h, l = axs.ravel()[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=min(4, max(1, len(l))), fontsize=8, frameon=False)
        fig.suptitle("%s %s (grey line: white noise = 1)" % (res["field"], res["split"]), fontsize=11)
        fig.tight_layout(rect=(0, 0.1, 1, 0.95))
        fig.savefig(os.path.join(outdir, name), dpi=130)
        plt.close(fig)

    if filters:
        figure("byscale_auto.png", ["filter:" + f for f in filters], ("auto", "diff"), filters, min(3, len(filters)), "residual power")
        if len(methods) > 1:
            figure("byscale_cross.png", ["filter:" + f for f in filters], ("cross",), filters, min(3, len(filters)), "cross-power")
    chans = [k for k in ("channel:short", "channel:long") if k in res["levels"]]
    if chans:
        figure("byscale_channel.png", chans, ("auto", "diff"), [c.split(":")[1] + "-wavelength detectors" for c in chans], len(chans), "residual power")


# =============================================================================== rate level (--level rate)
def conversion_map(cal_path, rate_path, need=None):
    """C = rate SCI / cal SCI of the same exposure: per-pixel ratio map from calibrated units (MJy/sr) to count rate (DN/s); it contains the
    official flat field and the photometric conversion, C = flat / PHOTMJSR (calwebb_image2: cal = rate x PHOTMJSR / flat).
    Both files come from the same ramp fit (checked: CAL_VER, CRDS context and the stage-1 reference files are compared), so the ratio is
    exact to float32 rounding wherever both are finite and cal != 0, at any signal level (noise cancels); measured on NRCB1 F090W: the
    pixel-to-pixel scatter of C is 1.6 per cent at |rate| > 10 ERR and at |rate| < 0.5 ERR alike.
    Rejected pixels: non-finite or zero cal or rate, or C x PHOTMJSR outside [0.2, 5] (an implausible flat value; also catches a cal file from
    a different ramp fit, where the ratio would be noisy). At rejected pixels that are needed (`need`, the scored pixels) C is filled with
    the median of the accepted C in the 5 x 5 neighbourhood (the smooth part of the flat only; that pixel's own flat value is lost), else
    1 / PHOTMJSR. Returns (C float64 (ny, nx) with NaN where rejected and not needed, info dict)."""
    from astropy.io import fits
    with fits.open(cal_path, memmap=False) as h:
        cal = np.asarray(h["SCI"].data, np.float64)
        hc, h0c = h["SCI"].header, h[0].header
    with fits.open(rate_path, memmap=False) as h:
        rate = np.asarray(h["SCI"].data, np.float64)
        h0r = h[0].header
    phot = hc.get("PHOTMJSR")
    with np.errstate(invalid="ignore", divide="ignore"):
        C = rate / cal
    ok = np.isfinite(C) & (cal != 0) & np.isfinite(rate) & (rate != 0)
    if phot:
        ok &= (C * phot > 0.2) & (C * phot < 5.0)
    keys = ("CAL_VER", "CRDS_CTX", "R_DARK", "R_LINEAR", "R_SUPERB", "R_MASK", "R_GAIN", "R_READNO", "R_SATURA")
    mism = {k: [str(h0c.get(k)), str(h0r.get(k))] for k in keys if str(h0c.get(k)) != str(h0r.get(k))}
    C = np.where(ok, C, np.nan)
    nfill = nfill_phot = 0
    if need is not None:
        bad = np.argwhere(np.asarray(need, bool) & ~ok)
        ny, nx = C.shape
        for y, x in bad:
            w = C[max(0, y - 2):min(ny, y + 3), max(0, x - 2):min(nx, x + 3)]
            w = w[np.isfinite(w)]
            if w.size:
                C[y, x] = float(np.median(w))
                nfill += 1
            else:
                C[y, x] = 1.0 / phot if phot else np.nan
                nfill_phot += 1
    info = dict(photmjsr=phot, n_accepted=int(ok.sum()), frac_accepted=float(ok.mean()), n_filled_local_median=nfill, n_filled_photmjsr=nfill_phot,
                median_C_x_photmjsr=float(np.nanmedian(C[ok] * phot)) if phot and ok.any() else None, header_mismatch=mism)
    return C, info


def rate_path_of(rate_dir, root):
    return os.path.join(rate_dir, root + "_rate.fits")


def prediction_units(pr):
    """'rate' if the prediction file's meta says so (rate-trained fieldop writes units = 'rate'), else 'cal'."""
    if "meta" not in pr:
        return "cal"
    try:
        m = json.loads(str(pr["meta"]))
    except Exception:
        return "cal"
    return "rate" if isinstance(m, dict) and str(m.get("units", "cal")).lower() == "rate" else "cal"


# =============================================================================== driver
def load_sci(path):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return S.load_sci(path)


def make_loaders(a):
    sp = json.load(open(os.path.join(a.root, "splits_%s.json" % a.field)))
    ex = {e["rootname"]: e for e in sp["exposures"]}

    def cal_path(r):
        p = ex[r]["path"]
        if a.data_root:
            p = os.path.join(a.data_root, ex[r]["file"])
        return p

    level = getattr(a, "level", "cal")
    cache = {}

    def rate_file(r):
        """SCI, ERR, DQ of the _rate file of r (one exposure cached)."""
        if cache.get("root") != r:
            from astropy.io import fits
            cache.clear()
            with fits.open(rate_path_of(a.rate_dir, r), memmap=False) as hl:
                cache.update(root=r, sci=np.asarray(hl["SCI"].data, np.float64), err=np.asarray(hl["ERR"].data, np.float64),
                             dq=np.asarray(hl["DQ"].data))
        return cache

    def conv(r, mk):
        if cache.get("C_root") != r:
            need = mk["valid"].astype(bool)
            C, info = conversion_map(cal_path(r), rate_path_of(a.rate_dir, r), need=need)
            cache.update(C_root=r, C=C, C_info=info)
            print("conversion %s: C accepted on %.4f of the detector; scored pixels filled with the local median %d, with 1/PHOTMJSR %d; "
                  "median C x PHOTMJSR %.4f; header mismatches %s" % (r, info["frac_accepted"], info["n_filled_local_median"],
                                                                     info["n_filled_photmjsr"], info["median_C_x_photmjsr"] or float("nan"),
                                                                     info["header_mismatch"] or "none"))
            CONV_LOG[r] = info
        return cache["C"]

    def mask_for(r):
        p = os.path.join(a.root, "heldout", a.field, r + ".npz")
        if not os.path.exists(p):
            return None
        m = dict(np.load(p))
        if level == "rate":
            # rate level: sigma = _rate ERR x the protocol's noise factor; pixels = the protocol mask (and --dq-zero on the _rate flags)
            rf = rate_file(r)
            m["sigma"] = (rf["err"] * m["noise_factor"].astype(np.float64)).astype(np.float64)
            if getattr(a, "dq_zero", False):
                n0 = int(m["valid"].sum())
                m["valid"] = m["valid"].astype(bool) & (rf["dq"] == 0)
                print("dq-zero (rate flags) %s: %d of %d protocol-valid pixels dropped" % (r, n0 - int(m["valid"].sum()), n0))
            cache["mask_valid"] = m["valid"].astype(bool)
            return m
        if getattr(a, "dq_zero", False):
            # the protocol mask was built from the files as they were then; a re-fetched file can flag pixels the mask calls valid
            # (14 Nov 2026: SATURATED on the brightest cores). Keep only pixels whose data-quality flag is zero in the file being scored.
            from astropy.io import fits
            with fits.open(cal_path(r), memmap=False) as hl:
                dq = np.asarray(hl["DQ"].data)
            n0 = int(m["valid"].sum())
            m["valid"] = m["valid"].astype(bool) & (dq == 0)
            print("dq-zero %s: %d of %d protocol-valid pixels dropped" % (r, n0 - int(m["valid"].sum()), n0))
        return m

    def pred_for(m, r):
        p = os.path.join(a.root, "pred", a.field, m, a.split, r + ".npz")
        if not os.path.exists(p):
            return None
        pr = dict(np.load(p, allow_pickle=False))
        u = prediction_units(pr)
        UNITS_LOG.setdefault(m, set()).add(u)
        if level == "rate" and u == "cal":
            mk = mask_for(r) if cache.get("root") != r or "mask_valid" not in cache else {"valid": cache["mask_valid"]}
            pr["model"] = np.asarray(pr["model"], np.float64) * conv(r, mk)
        elif level == "cal" and u == "rate":
            raise SystemExit("method %s: prediction %s is in rate units; score it with --level rate --rate-dir DIR" % (m, r))
        return pr

    def data_for(r):
        if level == "rate":
            return rate_file(r)["sci"]
        return load_sci(cal_path(r))

    if getattr(a, "offset_rank", ""):
        mask_plain = mask_for

        def mask_for(r):                                           # noqa: F811
            m = mask_plain(r)
            p = os.path.join(a.root, "pred", a.field, a.offset_rank, a.split, r + ".npz")
            if m is not None and os.path.exists(p):
                rk = np.asarray(np.load(p, allow_pickle=False)["model"], np.float64)
                if a.offset_rank_smooth > 0:
                    # box average over about offset_rank_smooth arcsec (nominal 0.031 / 0.063 arcsec pixels): the ranking then carries
                    # almost none of the training exposures' pixel noise either, so it favours neither noise-following nor smooth models
                    from scipy.ndimage import uniform_filter
                    n = max(3, int(round(a.offset_rank_smooth / (0.063 if "long" in r else 0.031))) | 1)
                    fin = np.isfinite(rk)
                    w = uniform_filter(fin.astype(np.float64), n, mode="constant")
                    rk = np.where(fin & (w > 0.5), uniform_filter(np.where(fin, rk, 0.0), n, mode="constant") / np.maximum(w, 1e-9), np.nan)
                m["offset_rank"] = rk
            return m

    return sp, mask_for, pred_for, data_for


CONV_LOG, UNITS_LOG = {}, {}


def _outdir(a):
    sub = (("dq0",) if a.dq_zero else ()) + (("rate",) if getattr(a, "level", "cal") == "rate" else ()) + \
          (("offrank",) if getattr(a, "offset_rank", "") else ())
    return os.path.join(a.out or os.path.join(a.root, "scores_paired"), a.field, *sub)


def pooled_bins(res_levels_field, variant, lo, hi):
    """chi2/N pooled over the S/N bins [lo, hi) of score.py's field-level entry (pixel-weighted)."""
    bins = res_levels_field[variant]["bins"]
    n = sum(b["n_pixels"] for b in bins[lo:hi])
    c = sum(b["chi2_per_pixel"] * b["n_pixels"] for b in bins[lo:hi] if b["n_pixels"])
    return c / n if n else float("nan"), n


def cmd_score(a):
    """Single-method scores with score.py's own score_run (same bins, offset and bootstrap), at the chosen level."""
    sp, mask_for, pred_for, data_for = make_loaders(a)
    d = _outdir(a)
    os.makedirs(d, exist_ok=True)
    lines = ["| method | level | units of the prediction | variant | " + " | ".join(BIN_NAMES) + " | S/N>=5 | S/N 5-50 |", "|" + "---|" * (NB + 6)]
    for m in a.methods:
        res = S.score_run(sp, m, a.split, mask_for, lambda r, m=m: pred_for(m, r), data_for, a.nboot)
        res["level"] = getattr(a, "level", "cal")
        res["prediction_units"] = sorted(UNITS_LOG.get(m, []))
        res["conversion"] = {r: CONV_LOG[r] for r in res["per_exposure"] if r in CONV_LOG} if res["level"] == "rate" else None
        lv = res["levels"]["field"]["all"] if res["levels"] else None
        if lv:
            for var in VARIANTS:
                b = lv[var]["bins"]
                p550, n550 = pooled_bins(lv, var, 2, 4)
                res.setdefault("pooled_snr_5_50", {})[var] = {"chi2": p550, "n_pixels": n550}
                lines.append("| %s | %s | %s | %s | " % (m, res["level"], "+".join(res["prediction_units"]), var)
                             + " | ".join("%.4f" % x["chi2_per_pixel"] if x["n_pixels"] else "n/a" for x in b)
                             + " | %.4f | %.4f |" % (lv[var]["headline_chi2_snr_ref_ge5"]["value"], p550))
        json.dump(res, open(os.path.join(d, "score_%s_%s.json" % (m, a.split)), "w"), indent=1, default=str)
    md = "\n".join(lines)
    open(os.path.join(d, "score_%s_%s.md" % ("+".join(a.methods), a.split)), "w").write(md + "\n")
    print(md)
    print("wrote", d)


def cmd_paired(a):
    sp, mask_for, pred_for, data_for = make_loaders(a)
    methods = [m for m in a.methods if m != a.reference]
    res = paired_run(sp, a.split, methods, a.reference, mask_for, pred_for, data_for, a.nboot, a.seed)
    res["level"], res["prediction_units"] = getattr(a, "level", "cal"), {k: sorted(v) for k, v in UNITS_LOG.items()}
    d = _outdir(a)
    os.makedirs(d, exist_ok=True)
    stem = "paired_%s_vs_%s_%s" % ("+".join(methods), a.reference, a.split)
    json.dump(res, open(os.path.join(d, stem + ".json"), "w"), indent=1)
    md = paired_markdown(res)
    open(os.path.join(d, stem + ".md"), "w").write(md)
    print(md)
    print("wrote", os.path.join(d, stem + ".json"), "and .md")


def cmd_byscale(a):
    sp, mask_for, pred_for, data_for = make_loaders(a)
    res = byscale_run(sp, a.split, list(a.methods), mask_for, pred_for, data_for, a.nboot, a.seed, a.snr_min, a.taper)
    res["level"] = getattr(a, "level", "cal")
    d = _outdir(a)
    os.makedirs(d, exist_ok=True)
    stem = "byscale_%s_%s" % ("+".join(a.methods), a.split)
    json.dump(res, open(os.path.join(d, stem + ".json"), "w"), indent=1)
    md = byscale_markdown(res)
    open(os.path.join(d, stem + ".md"), "w").write(md)
    byscale_figures(res, d)
    print(md)
    print("wrote", os.path.join(d, stem + ".json"), ".md and byscale_*.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("paired", cmd_paired), ("byscale", cmd_byscale), ("score", cmd_score)):
        s = sub.add_parser(name)
        s.add_argument("--field", required=True)
        s.add_argument("--split", required=True)
        s.add_argument("--methods", nargs="+", required=True)
        s.add_argument("--root", default=ROOT)
        s.add_argument("--data-root", default="", help="directory holding the _cal files (default: path in the splits file)")
        s.add_argument("--out", default="", help="output folder (default <root>/scores_paired)")
        s.add_argument("--nboot", type=int, default=S.N_BOOT)
        s.add_argument("--seed", type=int, default=S.SEED)
        s.add_argument("--offset-rank", default="", help="paired/byscale: choose the scorer's offset pixels by the lowest values of this "
                       "method's prediction (one made from the training exposures only, e.g. stage3_blot) instead of the lowest snr_ref of "
                       "the official mosaic, which shares noise with the held-out exposure (P8); output goes to an offrank/ subfolder")
        s.add_argument("--offset-rank-smooth", type=float, default=1.0,
                       help="arcsec; box-average the --offset-rank prediction over this width before ranking (0 = no smoothing)")
        s.add_argument("--dq-zero", action="store_true",
                       help="also require the data-quality flag of the file being scored to be zero (output goes to a dq0/ subfolder)")
        s.add_argument("--level", choices=["cal", "rate"], default="cal",
                       help="cal: data = _cal SCI, sigma = protocol sigma (default, as score.py). rate: data = _rate SCI, sigma = _rate ERR x the "
                       "protocol noise factor, pixels = protocol mask; calibrated-unit predictions are multiplied by C = rate SCI / cal SCI of "
                       "the same exposure, rate-unit predictions (meta units = 'rate') are used as they are (output in a rate/ subfolder)")
        s.add_argument("--rate-dir", default="", help="folder holding <rootname>_rate.fits (needed with --level rate)")
        if name == "paired":
            s.add_argument("--reference", required=True)
        else:
            s.add_argument("--snr-min", type=float, default=0.0, help="only pixels with snr_ref >= this (default 0: all scored pixels)")
            s.add_argument("--taper", type=float, default=0.1, help="edge taper: cosine ramp of this fraction of sqrt(box area) inside the box edge (default 0.1)")
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
