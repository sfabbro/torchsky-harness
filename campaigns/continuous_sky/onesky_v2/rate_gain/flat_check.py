#!/usr/bin/env python3
"""flat_check.py -- the official flat field as the truth for the learned per-pixel gain (fieldop --pixel-gain, F43-F45).

From a few _cal / _rate pairs of ONE detector (any of its filters) the official flat is C = rate SCI / cal SCI = flat / PHOTMJSR, exact
pixel by pixel (both files from the same ramp fit; score_paired.conversion_map). Its log, normalised, with the SAME gauge as the learned
map removed (monomials of total degree <= order over the map's observed pixels), is the "official pixel-to-pixel map". It is compared
with the learned log-gain map of a saved fit:

    python flat_check.py --state OUT/fieldop_state.pt --cal-dir CAL --rate-dir RATE --roots R1 R2 ... [--label NRCB1] [--png f.png] [--out f.json]

and, for a fit WITHOUT a gain map (or before any gain fit), the expected footprint, coverage and noise limit are predicted from the state's
geometry (the data region of every training exposure, a 22.48" square = box + 2 x data margin, placed with the cut rectangle and the mean
rotation the state stores) and its per-exposure background S/N (median data / median pipeline sigma):

    python flat_check.py --ref-state STATE --metrics metrics.json --cal-dir CAL --rate-dir RATE --roots ...

Reported: rms of the official map (after the gauge; and its high-pass part: minus the 9 x 9 median), correlation, rms of the difference and
slope between learned and official, the noise limit of the learned map (per pixel 1 / sqrt(I_p), I_p = sum over training cuts of
max(d^2/sig^2 - 1, 0) as stored by the fit; F44), the correlation expected at that limit, s / sqrt(s^2 + <sigma_p^2>), all overall, by
bins of sigma_p and on background-dominated pixels (I_p / n_p <= 2 x its median). Drop rule of the experiment: the gain test fails if the
correlation is below 0.3 where the expected correlation is above 0.5 (the noise allows clearly more than 0.3).
With --label a detector (detector mode) and pairs of several filters, the truth is the mean of the filters' maps weighted by the
background information each filter contributes in training (sum over its training exposures of (bkg0 / sigbar)^2); a per-filter map
(detector-filter mode) is compared with that filter's flat. Every approximation is printed with the result (keys 'approximations').
Needs numpy, scipy, astropy, torch (to read the state); matplotlib for --png.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

APPROX = [
    "official map = log(C x PHOTMJSR) with C = rate SCI / cal SCI; it includes everything calwebb_image2 divides by besides PHOTMJSR, i.e. the "
    "flat reference file (pixel-to-pixel QE and the large-scale response); its smooth part is removed by the gauge projection anyway",
    "pixels where C is rejected (non-finite, zero, or C x PHOTMJSR outside [0.2, 5]) are left out of every statistic",
    "noise limit = 1 / sqrt(I_p): the Cramer-Rao bound of log G_p if the sky and all other terms were known exactly and the model mean equals "
    "the data; the sky is fitted from the same data (each sky position is seen by about as many exposures as each pixel), which inflates the "
    "variance by roughly N / (N - 1) (reported as 'noise_limit_inflated', N = median number of training exposures per filter of the detector, "
    "since each sky position is seen by about that many); kernel and sky-model errors near "
    "bright sources are not in it",
    "with --ref-state (no gain map yet) the footprint is a 22.48-arcsec square per training exposure (data region) centred on its cut rectangle "
    "and rotated by the stored mean rotation (pixel scale 0.0311 / 0.0630 arcsec), ignoring distortion and the learned shifts (1-2 pixel "
    "errors at the edges); I_p = sum over training exposures covering p of (bkg0 / sigbar)^2 (background only: a lower bound where sources are)",
    "detector mode with several filters: truth = information-weighted mean of the per-filter maps (the learned map is one map for all filters; "
    "its best constant-in-filter approximation of filter-dependent flats is this weighted mean only if the information is spread evenly)",
]


def load_conv(cal_dir, rate_dir, roots):
    """Per filter: mean over its pairs of log(C x PHOTMJSR) (NaN where rejected) and the detector name."""
    from astropy.io import fits
    from score_paired import conversion_map
    out, det, n = {}, None, {}
    for r in roots:
        cp, rp = os.path.join(cal_dir, r + "_cal.fits"), os.path.join(rate_dir, r + "_rate.fits")
        C, info = conversion_map(cp, rp)
        h0 = fits.getheader(cp, 0)
        f, d = str(h0.get("FILTER")).strip(), str(h0.get("DETECTOR")).strip().upper()
        if str(h0.get("PUPIL", "CLEAR")).strip() not in ("CLEAR", "None"):
            f = f + "/" + str(h0.get("PUPIL")).strip()
        if det is None:
            det = d
        if d != det:
            raise SystemExit(f"pairs of one detector only: {r} is {d}, not {det}")
        lf = np.log(C * info["photmjsr"])
        if f in out:
            # same flat reference for the same detector and filter: check, then average (identical up to rounding)
            both = np.isfinite(out[f]) & np.isfinite(lf)
            dmax = float(np.abs(out[f][both] / n[f] - lf[both]).max()) if both.any() else 0.0
            print(f"{r}: {f} again, max |log flat difference| vs the earlier pair(s) {dmax:.2e}")
            out[f] = np.where(np.isfinite(lf), out[f] + lf, np.nan)
            n[f] += 1
        else:
            out[f], n[f] = lf, 1
    return {f: v / n[f] for f, v in out.items()}, det


def project(v, obs, box, order):
    """v (nr, nc) minus its least-squares monomials of degree <= order over obs (as model.PixelGain.project); NaN outside obs."""
    import torch
    from fieldop.model import poly_basis
    r0, c0, nr, nc = box
    idx = np.nonzero(obs.ravel())[0]
    V = poly_basis(torch.as_tensor(idx // nc + r0), torch.as_tensor(idx % nc + c0), box, order).numpy()
    Q = np.linalg.qr(V)[0]
    x = v.ravel()[idx].astype(np.float64)
    out = np.full(nr * nc, np.nan)
    out[idx] = x - Q @ (Q.T @ x)
    return out.reshape(nr, nc)


def highpass(v, size=9):
    from scipy.ndimage import median_filter
    f = np.where(np.isfinite(v), v, 0.0)
    return v - median_filter(f, size=size, mode="nearest")


def corr(x, y):
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def footprint_from_ref(st, metrics, det_name):
    """Count and background-information maps of the training exposures of one detector, from a state without a gain map (see APPROX)."""
    info = metrics["info"]
    dets = info["dets"]
    k = dets.index(det_name)
    half = float(info["half_data"])
    pix = 0.0630 if det_name.endswith("LONG") else 0.0311
    tr = st["gref"].numpy()
    e_det, e_filt = st["e_det"].numpy(), st["e_filt"].numpy()
    ids = [int(e) for e in tr if int(e_det[e]) == k]
    cnt = np.zeros((2048, 2048), np.float32)
    inf_ = np.zeros((2048, 2048), np.float64)
    perf = {}
    yy, xx = np.mgrid[0:2048, 0:2048]
    for e in ids:
        rows = np.nonzero(st["row_mask"][e].numpy())[0]
        cols = np.nonzero(st["col_mask"][e].numpy())[0]
        R = st["e_R"][e].numpy().astype(np.float64)
        s = 2 * half / pix
        ext = s * (np.abs(R).sum(1))            # full extent of the rotated square along col (x) and row (y), in pixels
        def centre(lo, hi, full):
            if lo <= 0 and hi >= 2047:
                return (lo + hi) / 2
            if lo <= 0:
                return hi - full / 2
            if hi >= 2047:
                return lo + full / 2
            return (lo + hi) / 2
        cx = centre(cols.min(), cols.max(), ext[0])
        cy = centre(rows.min(), rows.max(), ext[1])
        X = (R[0, 0] * (xx - cx) + R[0, 1] * (yy - cy)) * pix
        Y = (R[1, 0] * (xx - cx) + R[1, 1] * (yy - cy)) * pix
        inside = (np.abs(X) < half) & (np.abs(Y) < half)
        snr = float(st["bkg0"][e]) / float(st["e_sigbar"][e])
        cnt += inside
        inf_ += inside * snr ** 2
        f = info["filters"][int(e_filt[e])]
        perf.setdefault(f, []).append(snr)
    ob = cnt > 0
    rr, cc = np.nonzero(ob)
    box = (int(rr.min()), int(cc.min()), int(rr.max() - rr.min() + 1), int(cc.max() - cc.min() + 1))
    sl = (slice(box[0], box[0] + box[2]), slice(box[1], box[1] + box[3]))
    return box, ob[sl], cnt[sl], inf_[sl].astype(np.float32), perf


def filter_weights(st, metrics_or_meta, key_det, key_filt=None):
    """Background information per filter of the training exposures of the detector: sum of (bkg0 / sigbar)^2."""
    dets, filters = metrics_or_meta["dets"], metrics_or_meta["filters"]
    k = dets.index(key_det)
    w = {}
    for e in st["gref"].numpy():
        e = int(e)
        if int(st["e_det"][e]) != k:
            continue
        f = filters[int(st["e_filt"][e])]
        w[f] = w.get(f, 0.0) + (float(st["bkg0"][e]) / float(st["e_sigbar"][e])) ** 2
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state", default="", help="fieldop_state.pt of a fit with --pixel-gain")
    ap.add_argument("--ref-state", default="", help="fieldop_state.pt of a fit WITHOUT a gain map: predicted footprint and noise limit only")
    ap.add_argument("--metrics", default="", help="metrics.json of --ref-state (detector and filter names)")
    ap.add_argument("--label", default="", help="gain map label (NRCB1 or NRCB1/F090W); default: the detector of the pairs")
    ap.add_argument("--cal-dir", required=True)
    ap.add_argument("--rate-dir", required=True)
    ap.add_argument("--roots", nargs="+", required=True, help="rootnames of _cal/_rate pairs of one detector")
    ap.add_argument("--order", type=int, default=-1, help="gauge degree (default: the state's)")
    ap.add_argument("--png", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import torch
    flats, det = load_conv(a.cal_dir, a.rate_dir, a.roots)
    res = {"detector": det, "filters_with_pairs": sorted(flats), "pairs": a.roots, "approximations": APPROX}
    if a.state:
        st = torch.load(a.state, map_location="cpu", weights_only=False)
        pm = st.get("pixel_gain_meta")
        if pm is None:
            raise SystemExit("the state has no gain map (fit without --pixel-gain); use --ref-state for the prediction")
        label = a.label or det
        k = pm["labels"].index(label)
        box = tuple(pm["boxes"][k])
        o, n = pm["offsets"][k], box[2] * box[3]
        f = lambda t: st[t][o:o + n].numpy().reshape(box[2], box[3])  # noqa: E731
        learned, ob, cnt, inf_ = f("pixgain.logg"), f("pixgain.obs").astype(bool), f("pixgain.count"), f("pixgain.info")
        order = pm["poly_order"] if a.order < 0 else a.order
        meta = {"dets": pm["det_names"], "filters": pm["filters"]}
        res.update(mode="learned map", label=label, box=box, gauge_order=order, state_gauge=pm.get("gauge"))
        wdet = label.split("/")[0]
        wf = label.split("/")[1] if "/" in label else None
    elif a.ref_state:
        st = torch.load(a.ref_state, map_location="cpu", weights_only=False)
        metrics = json.load(open(a.metrics))
        box, ob, cnt, inf_, perf = footprint_from_ref(st, metrics, det)
        learned = None
        order = 2 if a.order < 0 else a.order
        meta = {"dets": metrics["info"]["dets"], "filters": metrics["info"]["filters"]}
        res.update(mode="prediction from a state without a gain map", box=box, gauge_order=order,
                   background_snr_per_exposure_by_filter={k: {"n": len(v), "median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
                                                          for k, v in perf.items()})
        wdet, wf = det, None
    else:
        raise SystemExit("give --state or --ref-state")
    # truth: per-filter maps cut to the box; detector mode: information-weighted mean over the filters with pairs
    r0, c0, nr, nc = box
    sl = (slice(r0, r0 + nr), slice(c0, c0 + nc))
    w = filter_weights(st, meta, wdet)
    res["training_background_information_by_filter"] = w
    if wf is not None:
        if wf not in flats:
            raise SystemExit(f"no pair for filter {wf}")
        truth = flats[wf][sl]
        res["truth"] = f"flat of {wf}"
    else:
        have = [f_ for f_ in flats if f_ in w] or list(flats)
        ws = np.array([w.get(f_, 1.0) for f_ in have])
        truth = sum(wi * flats[f_][sl] for wi, f_ in zip(ws / ws.sum(), have))
        res["truth"] = {"weighted mean of": dict(zip(have, (ws / ws.sum()).round(4).tolist())),
                        "training filters without a pair": sorted(set(w) - set(flats))}
    good = ob & np.isfinite(truth)
    tp = project(np.where(good, truth, 0.0), good, box, order)
    res["n_observed"], res["n_used"] = int(ob.sum()), int(good.sum())
    res["cuts_per_pixel"] = {"median": float(np.median(cnt[good])), "p10": float(np.percentile(cnt[good], 10)), "p90": float(np.percentile(cnt[good], 90)),
                             "histogram": {str(int(v)): int((cnt[good] == v).sum()) for v in np.unique(cnt[good])}}
    s_off = float(np.nanstd(tp[good]))
    hp_t = highpass(tp)
    res["official_rms_after_gauge"] = s_off
    res["official_robust_rms_after_gauge"] = float(1.4826 * np.median(np.abs(tp[good] - np.median(tp[good]))))
    res["official_fraction_beyond_5_robust_rms"] = float(np.mean(np.abs(tp[good] - np.median(tp[good])) > 5 * res["official_robust_rms_after_gauge"]))
    res["official_rms_highpass_9x9"] = float(np.nanstd(hp_t[good]))
    # per-filter official maps: rms and correlation between filters (is one map per detector enough?)
    pf = {}
    for f_ in sorted(flats):
        gf = good & np.isfinite(flats[f_][sl])
        pf[f_] = project(np.where(gf, flats[f_][sl], 0.0), gf, box, order)
    res["official_rms_by_filter"] = {f_: float(np.nanstd(v[good])) for f_, v in pf.items()}
    fl = sorted(pf)
    res["official_corr_between_filters"] = {f"{x}~{y}": corr(pf[x][good & np.isfinite(pf[x]) & np.isfinite(pf[y])],
                                                            pf[y][good & np.isfinite(pf[x]) & np.isfinite(pf[y])]) for i, x in enumerate(fl) for y in fl[i + 1:]}
    sig2 = 1.0 / np.maximum(inf_, 1e-30)
    # sky fitted from the same data: each sky position of a filter is seen by about N_f training exposures (all cover most of the box),
    # one of which is the pixel's own; variance inflation ~ N / (N - 1) with N = median training exposures per filter of this detector
    nf_ = {}
    for e in st["gref"].numpy():
        if int(st["e_det"][int(e)]) == meta["dets"].index(wdet):
            nf_[int(st["e_filt"][int(e)])] = nf_.get(int(st["e_filt"][int(e)]), 0) + 1
    Nf = float(np.median(list(nf_.values()))) if nf_ else float("nan")
    res["training_exposures_per_filter"] = {meta["filters"][k_]: v for k_, v in sorted(nf_.items())}
    infl = Nf / (Nf - 1) if Nf > 1 else np.inf
    sig = np.sqrt(sig2)
    res["noise_limit"] = {"median": float(np.median(sig[good])), "p10": float(np.percentile(sig[good], 10)), "p90": float(np.percentile(sig[good], 90)),
                          "rms": float(np.sqrt(sig2[good].mean()))}
    res["noise_limit_inflated"] = {"factor_on_variance": infl, "median": float(np.median(np.sqrt(sig2 * infl)[good]))}
    res["expected_corr_at_noise_limit"] = s_off / math.sqrt(s_off ** 2 + float(sig2[good].mean()))
    res["expected_corr_at_inflated_noise_limit"] = s_off / math.sqrt(s_off ** 2 + infl * float(sig2[good].mean()))
    bgd = good & ((inf_ / np.maximum(cnt, 1)) <= 2 * np.median((inf_ / np.maximum(cnt, 1))[good]))
    res["n_background_dominated"] = int(bgd.sum())
    edges = [0, 0.005, 0.01, 0.02, 0.05, np.inf]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = good & (sig >= lo) & (sig < hi)
        row = {"sigma_p": f"{lo}-{hi}", "n": int(m.sum())}
        if m.sum() > 2:
            row["expected_corr"] = s_off / math.sqrt(s_off ** 2 + float(sig2[m].mean()))
            if learned is not None:
                row["corr"] = corr(learned[m], tp[m])
        rows.append(row)
    res["by_noise_limit"] = rows
    if learned is not None:
        x, y = learned[good], tp[good]
        hp_l = highpass(np.where(good, learned, np.nan))
        res["learned_rms"] = float(x.std())
        res["corr"] = corr(x, y)
        res["corr_highpass_9x9"] = corr(hp_l[good], hp_t[good])
        res["corr_background_dominated"] = corr(learned[bgd], tp[bgd])
        res["rms_diff"] = float(np.sqrt(((x - y) ** 2).mean()))
        res["slope_learned_on_official"] = float((x * y).sum() / (y * y).sum())
        ok = [r for r in rows if r.get("expected_corr", 0) > 0.5 and r["n"] > 1000]
        test = good & np.isin(np.digitize(sig, edges) - 1, [i for i, r in enumerate(rows) if r.get("expected_corr", 0) > 0.5 and r["n"] > 1000])
        res["drop_rule"] = {"pixels where the expected correlation > 0.5": int(test.sum()),
                            "corr there": corr(learned[test], tp[test]) if test.sum() > 2 else None,
                            "verdict": ("no pixels where the noise allows a correlation above 0.5: the test is not decisive" if not ok else
                                        ("FAIL: correlation below 0.3 where the noise allows more" if corr(learned[test], tp[test]) < 0.3 else
                                         "pass: correlation >= 0.3 where the noise allows more"))}
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        npan = 3 if learned is not None else 2
        fig, ax = plt.subplots(1, npan, figsize=(5 * npan, 4.6))
        v = 3 * s_off
        im = ax[0].imshow(np.where(good, tp, np.nan), origin="lower", vmin=-v, vmax=v, cmap="RdBu_r")
        ax[0].set_title(f"official log flat, gauge removed ({det})")
        plt.colorbar(im, ax=ax[0], fraction=0.046)
        if learned is not None:
            im = ax[1].imshow(np.where(good, learned, np.nan), origin="lower", vmin=-v, vmax=v, cmap="RdBu_r")
            ax[1].set_title("learned log gain")
            plt.colorbar(im, ax=ax[1], fraction=0.046)
            ax[2].hexbin(tp[good], learned[good], gridsize=60, extent=(-v, v, -v, v), bins="log", cmap="Greys")
            ax[2].plot([-v, v], [-v, v], "r-", lw=0.8)
            ax[2].set_xlabel("official")
            ax[2].set_ylabel("learned")
            ax[2].set_title(f"corr {res['corr']:.3f}, expected {res['expected_corr_at_noise_limit']:.3f}")
        else:
            im = ax[1].imshow(np.where(good, sig, np.nan), origin="lower", cmap="viridis", vmax=np.percentile(sig[good], 99))
            ax[1].set_title("noise limit of log G per pixel")
            plt.colorbar(im, ax=ax[1], fraction=0.046)
        fig.tight_layout()
        fig.savefig(a.png, dpi=110)
    s = json.dumps(res, indent=1, default=str)
    print(s)
    if a.out:
        open(a.out, "w").write(s)


if __name__ == "__main__":
    main()
