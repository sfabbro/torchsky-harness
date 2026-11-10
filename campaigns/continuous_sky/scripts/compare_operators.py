#!/usr/bin/env python
"""Where exactly does operator A lose against operator B?  Compare two runs on the SAME held-out exposures from their pixel dumps.

    python scripts/compare_operators.py RUN_A RUN_B [--method-a opfield] [--method-b explicit_lib_3d] [--splits test] [--out DIR]
                                         [--peak-snr 50] [--stamp 15] [--isolation-px 8] [--max-exposures N]

RUN_x/heldout_pixels/<method>/*.npz are written by `opfield.py --dump-heldout-pixels` (method opfield) or `smacs_four_v38.py --dump-heldout-pixels`
(method explicit_lib_3d, ...); the format is the one read by null_stacks.py. Exposures are matched by `name`, pixels by (row, col); only the common
pixels are used. z = (data - model) / sigma with the pipeline sigma, exactly as the scores (chi2 = mean z^2). "A - B > 0" means A is worse.

Outputs (OUT, default ./compare_operators_<A>_vs_<B>): compare_operators.json and
  1_chi2_snr.png        mean z^2 and excess chi2 sum(z^2 - 1) versus S/N bin
  2_peaks_radial.png    radial profile of z^2, mean z and the mean fractional residual (data - model) / peak flux around MODEL peaks
  3_peaks_azimuth.png   16-sector azimuthal profile (DETECTOR frame, angle of (col, row) offset) of mean z in three annuli, with the m = 2 and 6 amplitudes
  4_position.png        mean z^2 on an 8 x 8 grid of detector position (dx, dy) per detector: A, B, A - B
  5_readout.png         mean z and mean z^2 versus detector row and column (bins of --rowcol-bin px) per detector
  6_chi2_fraction.png   share of the total chi2 in core / shoulders / wings / beyond / non-peak pixels, A and B, and the A - B difference per category

DEFINITIONS AND APPROXIMATIONS (all catalogue-free):
 * S/N used for the bins and for peak finding is the MEAN of the two models' model_sky / sigma, so both operators are binned identically and neither is
   favoured by its own prediction. sigma is the pipeline ERR of run A (the dumps of both runs hold the same ERR; a mismatch is reported).
 * Peaks = pixels that are the maximum of their 5 x 5 neighbourhood on the common mean-model image with S/N > --peak-snr (default 50), with a stamp
   of +-`--stamp` px entirely inside the loaded pixel footprint (>= 95 per cent of the stamp pixels present) and no other such peak within
   --isolation-px (default 8; 0 = off) so that wings are not dominated by a neighbour. Peak centre = flux-weighted centroid of the positive part of the
   mean model in the 5 x 5 box (sub-pixel), offsets in detector px (x = column, y = row; azimuth = atan2(y, x) in the DETECTOR frame, not rotated to
   the sky). Saturated or cosmic-ray-hit cores are not removed (the pipeline DQ already removed flagged pixels); bright pixels of a blended pair are kept.
 * Distance categories for the chi2 budget use the distance of EVERY loaded pixel to the nearest peak (S/N > --peak-snr, no stamp or isolation cut, so a
   bright blend counts as one source): core d < 1 px, shoulders 1 <= d < 3, wings 3 <= d < 15, beyond 15 <= d < 30, non-peak d >= 30 px (or no peak in the
   exposure). Pixel-wise independent noise is assumed nowhere here: only sums and means are reported, with the standard error of the mean z^2
   (sqrt(var / n), treating pixels as independent; resampling noise and PSF errors correlate neighbouring pixels, so it is a lower bound).
 * The fractional residual divides data - model by the model peak flux of each stamp (the PSF-shape error as a fraction of the source); its mean is
   the average over peaks and pixels weighted by pixel count, not by flux.
 * Position grids use the dumped (dx, dy) in [-1, 1] of each detector. Row / column profiles use the readout rows / columns of the detector array.
 * `robust` = pseudo-Huber 2 d^2 (sqrt(1 + z^2 / d^2) - 1) with d = 3 (the score of the held-out protocol), reported next to the plain chi2.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

SNR_EDGES = np.array([-np.inf, 0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, np.inf])
RAD_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 15.0001])
AZ_ANN = ((1.0, 3.0), (3.0, 8.0), (8.0, 15.0))
N_AZ = 16
CAT_NAMES = ("core r<1", "shoulders 1-3", "wings 3-15", "beyond 15-30", "non-peak >30")
CAT_EDGES = (1.0, 3.0, 15.0, 30.0)
NPOS = 8


def load_run(run_dir, method=None, splits=None):
    base = os.path.join(run_dir, "heldout_pixels")
    methods = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    if not methods:
        raise SystemExit(f"{base}: no model directory")
    method = method or methods[0]
    out = {}
    for f in sorted(glob.glob(os.path.join(base, method, "*.npz"))):
        z = np.load(f, allow_pickle=False)
        e = {k: z[k] for k in z.files}
        for k in ("name", "band", "instrument", "det", "split", "method"):
            if k in e:
                e[k] = str(e[k])
        if splits and e["split"] not in splits:
            continue
        out[e["name"]] = e
    return method, out


def align(a, b):
    """Common pixels of two exposure dumps (by row, col). Returns index arrays ia, ib."""
    ka = a["row"].astype(np.int64) * 8192 + a["col"].astype(np.int64)
    kb = b["row"].astype(np.int64) * 8192 + b["col"].astype(np.int64)
    _, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    return ia, ib


class Acc:
    """Sums per bin of named value arrays (bincount); means / rms are formed at the end."""

    def __init__(self):
        self.d = {}

    def add(self, name, idx, nb, **vals):
        ok = idx >= 0
        idx = idx[ok]
        ent = self.d.setdefault(name, {"n": np.zeros(nb)})
        ent["n"] += np.bincount(idx, minlength=nb)
        for k, v in vals.items():
            ent[k] = ent.get(k, 0.0) + np.bincount(idx, weights=np.asarray(v, float)[ok], minlength=nb)

    def get(self, name):
        return self.d.get(name)


def summarise(ent, keys=("zA", "zB", "z2A", "z2B", "rA", "rB", "fA", "fB")):
    """Per-bin means from the sums; absent keys are skipped."""
    if ent is None:
        return None
    n = ent["n"]
    out = {"n": n.tolist()}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in keys:
            if k in ent:
                out["mean_" + k] = np.where(n > 0, ent[k] / n, np.nan).tolist()
        if "z2A" in ent and "z2B" in ent:
            out["excess_chi2_A"] = (ent["z2A"] - n).tolist()
            out["excess_chi2_B"] = (ent["z2B"] - n).tolist()
            out["sum_chi2_A"], out["sum_chi2_B"] = ent["z2A"].tolist(), ent["z2B"].tolist()
            out["diff_mean_z2_A_minus_B"] = np.where(n > 0, (ent["z2A"] - ent["z2B"]) / n, np.nan).tolist()
    return out


def dense(shape, rows, cols, val, fill=np.nan):
    im = np.full(shape, fill, np.float64)
    im[rows, cols] = val
    return im


def find_peaks(snr_img, thr):
    mx = ndi.maximum_filter(np.where(np.isfinite(snr_img), snr_img, -1e30), size=5, mode="constant", cval=-1e30)
    return (snr_img == mx) & (snr_img > thr)


def azimuth_amplitudes(prof, centres):
    """|c_m| / mean of the sector profile for m = 2, 6 (prof: N_AZ sector means; the mean over sectors is subtracted from the amplitude's meaning:
    amplitude of the cos / sin component of the profile, in units of the profile itself, NOT divided by the mean z (which is ~0))."""
    p = np.asarray(prof, float)
    ok = np.isfinite(p)
    out = {}
    for m in (2, 6):
        if ok.sum() < 8:
            out[f"m{m}"] = None
            continue
        c = (p[ok] * np.cos(m * centres[ok])).sum() * 2 / ok.sum()
        s = (p[ok] * np.sin(m * centres[ok])).sum() * 2 / ok.sum()
        out[f"m{m}"] = float(math.hypot(c, s))
    return out


def analyse_exposure(eA, eB, cfg, accs, rec):
    ia, ib = align(eA, eB)
    if len(ia) < 100:
        rec["skipped"] = f"only {len(ia)} common pixels"
        return
    row, col = eA["row"][ia].astype(np.int64), eA["col"][ia].astype(np.int64)
    data, sig = eA["data"][ia].astype(np.float64), eA["sigma"][ia].astype(np.float64)
    mA, mB = eA["model"][ia].astype(np.float64), eB["model"][ib].astype(np.float64)
    sA, sB = eA["model_sky"][ia].astype(np.float64), eB["model_sky"][ib].astype(np.float64)
    rec["n_pixels"], rec["n_pixels_A"], rec["n_pixels_B"] = int(len(ia)), int(len(eA["row"])), int(len(eB["row"]))
    rec["max_abs_data_diff_over_sigma"] = float(np.max(np.abs(data - eB["data"][ib]) / sig))
    rec["max_rel_sigma_diff"] = float(np.max(np.abs(sig - eB["sigma"][ib]) / sig))
    zA, zB = (data - mA) / sig, (data - mB) / sig
    ok = np.isfinite(zA) & np.isfinite(zB) & np.isfinite(sig) & (sig > 0)
    if not ok.all():
        row, col, data, sig, mA, mB, sA, sB, zA, zB = (x[ok] for x in (row, col, data, sig, mA, mB, sA, sB, zA, zB))
        ia, dxv, dyv = ia[ok], eA["dx"][ia][ok], eA["dy"][ia][ok]
    else:
        dxv, dyv = eA["dx"][ia], eA["dy"][ia]
    d3 = 9.0
    rA = 2 * d3 * (np.sqrt(1 + zA ** 2 / d3) - 1)
    rB = 2 * d3 * (np.sqrt(1 + zB ** 2 / d3) - 1)
    snr = 0.5 * (sA + sB) / sig
    rec.update(chi2_A=float((zA ** 2).mean()), chi2_B=float((zB ** 2).mean()), robust_A=float(rA.mean()), robust_B=float(rB.mean()))
    shape = (int(row.max()) + 1, int(col.max()) + 1)
    groups = ["all", f"band:{rec['band']}", f"det:{rec['instrument']}/{rec['det']}", f"cell:{rec['instrument']}/{rec['det']}/{rec['band']}"]
    for g in groups:
        accs.setdefault(g, Acc())

    # ---- 1. S/N bins
    isn = np.clip(np.digitize(snr, SNR_EDGES) - 1, 0, len(SNR_EDGES) - 2)
    for g in groups:
        accs[g].add("snr", isn, len(SNR_EDGES) - 1, zA=zA, zB=zB, z2A=zA ** 2, z2B=zB ** 2, rA=rA, rB=rB)

    # ---- peaks (common, catalogue-free)
    snr_img = dense(shape, row, col, snr)
    pk = find_peaks(snr_img, cfg["peak_snr"])
    py, px = np.nonzero(pk)
    rec["n_peaks_all"] = int(len(py))

    # ---- 6. chi2 budget by distance to the nearest peak (every loaded pixel; no stamp or isolation cut)
    if len(py):
        dist_img = ndi.distance_transform_edt(~pk)
        dist = dist_img[row, col]
    else:
        dist = np.full(len(row), np.inf)
    icat = np.digitize(dist, CAT_EDGES)          # 0 core, 1 shoulders, 2 wings, 3 beyond, 4 non-peak
    for g in groups:
        accs[g].add("cat", icat, 5, zA=zA, zB=zB, z2A=zA ** 2, z2B=zB ** 2, rA=rA, rB=rB)
    n_c = np.bincount(icat, minlength=5)
    rec["chi2_sum_by_category_A"] = np.bincount(icat, weights=zA ** 2, minlength=5).tolist()
    rec["chi2_sum_by_category_B"] = np.bincount(icat, weights=zB ** 2, minlength=5).tolist()
    rec["n_by_category"] = n_c.tolist()

    # ---- 4. position, 5. readout
    ipos = np.clip(((dxv + 1) / 2 * NPOS).astype(int), 0, NPOS - 1) * NPOS + np.clip(((dyv + 1) / 2 * NPOS).astype(int), 0, NPOS - 1)
    nrb = int(math.ceil(shape[0] / cfg["rowcol_bin"]))
    ncb = int(math.ceil(shape[1] / cfg["rowcol_bin"]))
    gd = [f"det:{rec['instrument']}/{rec['det']}", "all"]
    for g in gd:
        accs[g].add("pos", ipos, NPOS * NPOS, zA=zA, zB=zB, z2A=zA ** 2, z2B=zB ** 2)
        accs[g].add("row", row // cfg["rowcol_bin"], max(nrb, 64), zA=zA, zB=zB, z2A=zA ** 2, z2B=zB ** 2)
        accs[g].add("col", col // cfg["rowcol_bin"], max(ncb, 64), zA=zA, zB=zB, z2A=zA ** 2, z2B=zB ** 2)

    # ---- 2, 3. stacks around isolated model peaks
    H = cfg["stamp"]
    keep = np.ones(len(py), bool)
    if len(py):
        keep &= (py >= H) & (px >= H) & (py < shape[0] - H) & (px < shape[1] - H)
        if cfg["isolation_px"] > 0 and len(py) > 1:
            tr = cKDTree(np.stack([px, py], 1))
            pairs = tr.query_pairs(cfg["isolation_px"], output_type="ndarray")
            iso = np.ones(len(py), bool)
            if len(pairs):
                iso[pairs[:, 0]] = False
                iso[pairs[:, 1]] = False
            keep &= iso
    n_used = 0
    mimg = dense(shape, row, col, 0.5 * (sA + sB))
    zAi, zBi = dense(shape, row, col, zA), dense(shape, row, col, zB)
    fAi, fBi = dense(shape, row, col, (data - mA)), dense(shape, row, col, (data - mB))
    yy, xx = np.mgrid[-H:H + 1, -H:H + 1]
    cy, cx = np.mgrid[-2:3, -2:3]
    for y0, x0 in zip(py[keep], px[keep]):
        sub = mimg[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1]
        if np.isfinite(sub).mean() < 0.95:
            continue
        box = np.clip(np.nan_to_num(mimg[y0 - 2:y0 + 3, x0 - 2:x0 + 3], nan=0.0), 0, None)
        if box.sum() <= 0:
            continue
        ccx, ccy = (box * cx).sum() / box.sum(), (box * cy).sum() / box.sum()
        ox, oy = xx - ccx, yy - ccy
        r = np.hypot(ox, oy).ravel()
        ph = np.arctan2(oy, ox).ravel()
        zs_A = zAi[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].ravel()
        zs_B = zBi[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].ravel()
        fl = max(float(mimg[y0, x0]), 1e-12)
        fs_A = fAi[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].ravel() / fl
        fs_B = fBi[y0 - H:y0 + H + 1, x0 - H:x0 + H + 1].ravel() / fl
        v = np.isfinite(zs_A) & np.isfinite(zs_B) & (r < RAD_EDGES[-1])
        r, ph, zs_A, zs_B, fs_A, fs_B = r[v], ph[v], zs_A[v], zs_B[v], fs_A[v], fs_B[v]
        ir = np.digitize(r, RAD_EDGES) - 1
        for g in groups:
            accs[g].add("radial", ir, len(RAD_EDGES) - 1, zA=zs_A, zB=zs_B, z2A=zs_A ** 2, z2B=zs_B ** 2, fA=fs_A, fB=fs_B)
        isec = np.floor((ph + np.pi) / (2 * np.pi) * N_AZ).astype(int) % N_AZ
        for k, (r1, r2) in enumerate(AZ_ANN):
            m_ = (r >= r1) & (r < r2)
            for g in groups:
                accs[g].add(f"az{k}", isec[m_], N_AZ, zA=zs_A[m_], zB=zs_B[m_], z2A=zs_A[m_] ** 2, z2B=zs_B[m_] ** 2, fA=fs_A[m_], fB=fs_B[m_])
        n_used += 1
    rec["n_peaks_used"] = n_used


def finalize(accs):
    res = {}
    centres = (np.arange(N_AZ) + 0.5) / N_AZ * 2 * np.pi - np.pi
    for g, acc in accs.items():
        ent = {}
        for name in ("snr", "cat", "pos", "row", "col", "radial"):
            s = summarise(acc.get(name))
            if s is not None:
                ent[name] = s
        if "radial" in ent:
            ent["radial"]["r_edges_px"] = RAD_EDGES.tolist()
        if "snr" in ent:
            ent["snr"]["snr_edges"] = [None if not np.isfinite(x) else float(x) for x in SNR_EDGES]
        if "cat" in ent:
            tot_a, tot_b = float(np.sum(acc.get("cat")["z2A"])), float(np.sum(acc.get("cat")["z2B"]))
            ent["cat"]["names"] = list(CAT_NAMES)
            ent["cat"]["fraction_of_total_chi2_A"] = (acc.get("cat")["z2A"] / tot_a).tolist()
            ent["cat"]["fraction_of_total_chi2_B"] = (acc.get("cat")["z2B"] / tot_b).tolist()
            ent["cat"]["delta_chi2_over_total_B"] = ((acc.get("cat")["z2A"] - acc.get("cat")["z2B"]) / tot_b).tolist()
            ent["cat"]["total_chi2_per_pixel_A"] = tot_a / float(np.sum(acc.get("cat")["n"]))
            ent["cat"]["total_chi2_per_pixel_B"] = tot_b / float(np.sum(acc.get("cat")["n"]))
            ent["cat"]["n_pixels_total"] = int(np.sum(acc.get("cat")["n"]))
        for k, (r1, r2) in enumerate(AZ_ANN):
            a_ = acc.get(f"az{k}")
            if a_ is None:
                continue
            s = summarise(a_)
            s["annulus_px"] = [r1, r2]
            s["sector_centres_rad"] = centres.tolist()
            s["amp_mean_z_A"] = azimuth_amplitudes(s["mean_zA"], centres)
            s["amp_mean_z_B"] = azimuth_amplitudes(s["mean_zB"], centres)
            s["amp_frac_resid_A"] = azimuth_amplitudes(s["mean_fA"], centres)
            s["amp_frac_resid_B"] = azimuth_amplitudes(s["mean_fB"], centres)
            ent[f"azimuth_{k}"] = s
        res[g] = ent
    return res


# ------------------------------------------------------------------------------------------------ plots
def make_plots(res, outdir, nameA, nameB):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cA, cB = "#c0392b", "#2a6f97"
    bands = [g for g in res if g.startswith("band:")]
    cells = ["all"] + bands
    ncol = min(4, len(cells))
    nrow = int(math.ceil(len(cells) / ncol))

    def grid(title, fname, fn, rows=1, w=3.6, h=3.0):
        fig, axs = plt.subplots(rows * nrow, ncol, figsize=(w * ncol, h * nrow * rows), squeeze=False)
        for k, g in enumerate(cells):
            r0, c0 = divmod(k, ncol)
            fn(axs, r0 * rows, c0, res[g], g)
        for k in range(len(cells), nrow * ncol):
            r0, c0 = divmod(k, ncol)
            for rr in range(rows):
                axs[r0 * rows + rr, c0].axis("off")
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=100)
        plt.close(fig)

    xs = np.arange(len(SNR_EDGES) - 1)
    xl = [f"{'' if not np.isfinite(SNR_EDGES[i]) else SNR_EDGES[i]:g}" if np.isfinite(SNR_EDGES[i]) else "-inf" for i in range(len(SNR_EDGES) - 1)]

    def f_snr(axs, r, c, ent, g):
        ax = axs[r, c]
        s = ent.get("snr")
        if s is None:
            return
        ax.plot(xs, s["mean_z2A"], "o-", color=cA, label=nameA)
        ax.plot(xs, s["mean_z2B"], "s-", color=cB, label=nameB)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(xl, rotation=60, fontsize=6)
        ax.set_title(g + "  mean z$^2$ vs S/N", fontsize=8)
        ax.axhline(1, color="grey", lw=0.5)
        if g == "all":
            ax.legend(fontsize=6)
    grid("chi2 per pixel in S/N bins (S/N = mean of the two models / sigma)", "1_chi2_snr.png", f_snr)

    rc = 0.5 * (RAD_EDGES[1:] + RAD_EDGES[:-1])

    def f_rad(axs, r, c, ent, g):
        s = ent.get("radial")
        if s is None:
            axs[r, c].axis("off")
            return
        ax = axs[r, c]
        ax.plot(rc, s["mean_z2A"], "o-", color=cA, label=nameA)
        ax.plot(rc, s["mean_z2B"], "s-", color=cB, label=nameB)
        ax.set_yscale("log")
        ax.axhline(1, color="grey", lw=0.5)
        ax.set_title(g + " <z$^2$> around peaks", fontsize=8)
        ax.set_xlabel("r [detector px]", fontsize=7)
        if g == "all":
            ax.legend(fontsize=6)
        ax = axs[r + 1, c]
        ax.plot(rc, s["mean_fA"], "o-", color=cA)
        ax.plot(rc, s["mean_fB"], "s-", color=cB)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_title("mean (data - model) / peak flux", fontsize=8)
    grid("Stacks around model peaks (S/N > threshold, isolated)", "2_peaks_radial.png", f_rad, rows=2)

    cen = (np.arange(N_AZ) + 0.5) / N_AZ * 360 - 180

    def f_az(axs, r, c, ent, g):
        for k in range(len(AZ_ANN)):
            s = ent.get(f"azimuth_{k}")
            if s is None:
                continue
            ax = axs[r, c]
            ax.plot(cen, np.array(s["mean_zA"]) + 0.0, "-", color=cA, alpha=0.4 + 0.3 * k, label=f"{nameA} {AZ_ANN[k]}")
            ax.plot(cen, np.array(s["mean_zB"]), "--", color=cB, alpha=0.4 + 0.3 * k, label=f"{nameB} {AZ_ANN[k]}")
        ax = axs[r, c]
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(g + " mean z vs azimuth (detector frame)", fontsize=8)
        ax.set_xlabel("azimuth [deg]", fontsize=7)
        if g == "all":
            ax.legend(fontsize=5)
    grid("Azimuthal profiles of the whitened residual in annuli 1-3, 3-8, 8-15 px", "3_peaks_azimuth.png", f_az)

    dets = [g for g in res if g.startswith("det:")]
    if dets:
        fig, axs = plt.subplots(len(dets), 3, figsize=(9, 2.8 * len(dets)), squeeze=False)
        for i, g in enumerate(dets):
            s = res[g].get("pos")
            if s is None:
                continue
            A = np.array(s["mean_z2A"]).reshape(NPOS, NPOS).T
            B = np.array(s["mean_z2B"]).reshape(NPOS, NPOS).T
            for j, (im, t, cm, vm) in enumerate(((A, nameA, "viridis", (0.8, 3)), (B, nameB, "viridis", (0.8, 3)), (A - B, "A - B", "RdBu_r", (-1, 1)))):
                p = axs[i, j].imshow(im, origin="lower", extent=[-1, 1, -1, 1], cmap=cm, vmin=vm[0], vmax=vm[1])
                axs[i, j].set_title(f"{g} {t}" if j == 0 else t, fontsize=7)
                fig.colorbar(p, ax=axs[i, j], fraction=0.046)
        fig.suptitle("mean z$^2$ vs detector position (x = dx, y = dy)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "4_position.png"), dpi=100)
        plt.close(fig)

        fig, axs = plt.subplots(len(dets), 4, figsize=(14, 2.6 * len(dets)), squeeze=False)
        for i, g in enumerate(dets):
            for j, (key, lab) in enumerate((("row", "row"), ("col", "col"))):
                s = res[g].get(key)
                if s is None:
                    continue
                n = np.array(s["n"])
                u = n > 0
                b = np.arange(len(n))[u]
                axs[i, 2 * j].plot(b, np.array(s["mean_zA"])[u], "-", color=cA, lw=0.8)
                axs[i, 2 * j].plot(b, np.array(s["mean_zB"])[u], "-", color=cB, lw=0.8)
                axs[i, 2 * j].set_title(f"{g}: mean z vs {lab} bin", fontsize=7)
                axs[i, 2 * j + 1].plot(b, np.array(s["mean_z2A"])[u], "-", color=cA, lw=0.8)
                axs[i, 2 * j + 1].plot(b, np.array(s["mean_z2B"])[u], "-", color=cB, lw=0.8)
                axs[i, 2 * j + 1].set_title("mean z$^2$", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "5_readout.png"), dpi=100)
        plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    lab = [g.replace("band:", "") for g in cells]
    xsb = np.arange(len(cells))
    bot = np.zeros((2, len(cells)))
    cols = ["#7b2d26", "#c0392b", "#e59866", "#f5cba7", "#aab7b8"]
    for ci in range(5):
        for k, key in enumerate(("fraction_of_total_chi2_A", "fraction_of_total_chi2_B")):
            vals = np.array([res[g]["cat"][key][ci] if "cat" in res[g] else 0 for g in cells])
            axs[0].bar(xsb + (k - 0.5) * 0.4, vals, 0.38, bottom=bot[k], color=cols[ci], label=CAT_NAMES[ci] if k == 0 else None, edgecolor="white", lw=0.3)
            bot[k] += vals
    axs[0].set_xticks(xsb)
    axs[0].set_xticklabels([f"{x}\nA | B" for x in lab], fontsize=6)
    axs[0].set_title(f"share of the total chi2 (left bar A = {nameA}, right B = {nameB})", fontsize=8)
    axs[0].legend(fontsize=6)
    for ci in range(5):
        axs[1].bar(xsb + (ci - 2) * 0.15, [res[g]["cat"]["delta_chi2_over_total_B"][ci] if "cat" in res[g] else 0 for g in cells], 0.14, color=cols[ci], label=CAT_NAMES[ci])
    axs[1].axhline(0, color="grey", lw=0.5)
    axs[1].set_xticks(xsb)
    axs[1].set_xticklabels(lab, fontsize=6)
    axs[1].set_title("(sum z$^2$_A - sum z$^2$_B) / total chi2_B per category  (> 0: A worse)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "6_chi2_fraction.png"), dpi=100)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--method-a", default=None)
    ap.add_argument("--method-b", default=None)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--peak-snr", type=float, default=50.0)
    ap.add_argument("--stamp", type=int, default=15)
    ap.add_argument("--isolation-px", type=float, default=8.0)
    ap.add_argument("--rowcol-bin", type=int, default=64)
    ap.add_argument("--max-exposures", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    splits = set(a.splits.split(",")) if a.splits else None
    ma, A = load_run(a.run_a, a.method_a, splits)
    mb, B = load_run(a.run_b, a.method_b, splits)
    common = [n for n in A if n in B]
    if a.max_exposures:
        common = common[: a.max_exposures]
    if not common:
        raise SystemExit(f"no common exposures: A has {len(A)}, B has {len(B)} ({a.splits})")
    nameA, nameB = f"A:{os.path.basename(os.path.normpath(a.run_a))}/{ma}", f"B:{os.path.basename(os.path.normpath(a.run_b))}/{mb}"
    out = a.out or f"compare_operators_{os.path.basename(os.path.normpath(a.run_a))}_vs_{os.path.basename(os.path.normpath(a.run_b))}"
    os.makedirs(out, exist_ok=True)
    cfg = {"peak_snr": a.peak_snr, "stamp": a.stamp, "isolation_px": a.isolation_px, "rowcol_bin": a.rowcol_bin}
    accs, recs = {}, []
    for n in common:
        eA = A[n]
        rec = {"name": n, "band": eA["band"], "instrument": eA["instrument"], "det": eA["det"], "split": eA["split"]}
        analyse_exposure(eA, B[n], cfg, accs, rec)
        recs.append(rec)
        print(f"{n}: chi2 A {rec.get('chi2_A', float('nan')):.3f} B {rec.get('chi2_B', float('nan')):.3f} peaks {rec.get('n_peaks_used', 0)}/{rec.get('n_peaks_all', 0)}", flush=True)
    res = finalize(accs)
    summary = {"A": nameA, "B": nameB, "config": cfg, "n_common_exposures": len(common), "only_in_A": [n for n in A if n not in B], "only_in_B": [n for n in B if n not in A],
               "per_exposure": recs, "groups": res, "convention": "z = (data - model) / sigma; A - B > 0 means A is worse"}
    json.dump(summary, open(os.path.join(out, "compare_operators.json"), "w"), indent=1, default=float)
    if not a.no_plots:
        make_plots(res, out, nameA, nameB)
    print("per band: chi2/pixel A, B | category with the largest |A-B| (share of total chi2_B; + = A worse) | m=6 amplitude of mean z in 3-8 px A, B")
    for g in sorted(k for k in res if k.startswith("band:")):
        c = res[g].get("cat")
        if c is None:
            continue
        d = np.array(c["delta_chi2_over_total_B"])
        k = int(np.argmax(np.abs(d)))
        az = res[g].get("azimuth_1", {})
        a6 = (az.get("amp_mean_z_A", {}) or {}).get("m6")
        b6 = (az.get("amp_mean_z_B", {}) or {}).get("m6")
        print(f"  {g[5:]:>8s}: {c['total_chi2_per_pixel_A']:8.3f} {c['total_chi2_per_pixel_B']:8.3f} | {CAT_NAMES[k]} ({d[k]:+.3f}) | "
              + ("n/a" if a6 is None or b6 is None else f"{a6:.3f} {b6:.3f}"))
    allc = res["all"]["cat"]
    print(f"ALL: chi2 per pixel A {allc['total_chi2_per_pixel_A']:.4f}  B {allc['total_chi2_per_pixel_B']:.4f}; share of total chi2 by category A "
          + ", ".join(f"{n} {f:.3f}" for n, f in zip(CAT_NAMES, allc["fraction_of_total_chi2_A"])) + " | B "
          + ", ".join(f"{f:.3f}" for f in allc["fraction_of_total_chi2_B"]))
    print(f"wrote {out}/compare_operators.json and PNGs")


if __name__ == "__main__":
    main()
