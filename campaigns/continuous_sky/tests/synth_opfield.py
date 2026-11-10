"""Synthetic identifiability test of scripts/opfield.py (CPU, about 10 minutes).

The DATA are rendered with a different renderer from the one in the model (no inverse crime):
  * truth sky on a grid 4x finer than the model sky (0.01 arcsec): colourful point sources (4 bright, 26 faint), an extended elliptical blob and
    a faint diffuse gradient; the band image is a 9-node integral over a +-8 percent boxcar bandpass (the model uses 3 quantile nodes);
  * three instruments: SHARP (pixel 0.032", FWHM 0.07", fine sampled), COARSE (pixel 0.12", FWHM 0.20", 3x undersampled relative to the
    0.04" sky grid, a-priori WCS errors up to 0.6" like HST), MID (pixel 0.06", FWHM 0.13"), all with an analytic PSF = elliptical Moffat
    (beta 3) + an Airy-like ring (3 percent of the flux) + faint spikes along the detector axes (2 percent), fixed in the DETECTOR frame
    (so it rotates with the roll), width growing 12 percent across the detector (a different parametrisation from the model's);
  * each detector pixel integrates the PSF over its box (5x5 supersampling, detector frame, rotated with the roll) by an FFT convolution of the
    fine sky with the sky-frame kernel, read at the true pixel position with a cubic spline (scipy map_coordinates);
  * rolls 0, 35, 70 degrees, sub-pixel dithers, per-exposure header-WCS errors, a mild static distortion (up to ~0.1 target px, defined in the
    detector frame), per-instrument gains 1.0 / 0.7 / 1.4 with 3 percent per-exposure scatter, a soft saturation tanh(x / x_sat) x_sat on the
    brightest pixels (peak 4000+ counts vs x_sat 5000), a per-exposure quadratic background, heteroscedastic Gaussian noise (the pipeline
    ivar is the TRUE noise variance), 1 percent cosmic rays (exponential, mean 40 counts, left in the data).
The three instruments observe overlapping bands (1.50, 1.60, 1.55 micron, +-8 percent boxcars): the sky s(x, lambda) is shared and the
instruments are tied to each other through it (with well separated bands one band image per instrument fits any registration: the
cross-instrument registration is then unidentifiable, as found in a first run of this test).
Splits: training (27 exposures), 1 validation exposure per instrument (roll 35), held-out T1: 2 new dithers per instrument (rolls 0 and 70),
held-out T2: 1 exposure per instrument at a NEW roll (105 degrees).
Reports: held-out chi2 (all, bright S/N>5) vs the oracle (truth expectation on the same pixels), registration recovery (common-shift gauge
removed) with / without coarse-to-fine, relative kernels vs truth and the common-mode g fit (test_v36_lib part h analysis), amplitude (flux)
ratios, outlier-head precision / recall on the cosmic rays.
Run:  venvT/bin/python tests/synth_opfield.py [--steps 1800] [--quick] [--out runs/opfield_synth]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
import opfield as O  # noqa: E402
import smacs_four_v36 as V  # noqa: E402
from test_v36_lib import common_mode, rel_kernel_err, ee as ee_curve  # noqa: E402  (the part h analysis, reused)

torch.set_num_threads(2)
DEV = torch.device("cpu")
SCALE = 0.04        # arcsec per target px (model sky grid)
FINE = 4            # truth sky is FINE times finer
NPIX = 80           # data box [HALO, HALO+NPIX) in grid px
HALO = 24
G = NPIX + 2 * HALO  # model sky grid size
SAT = 5000.0
INSTR = {
    "sharp": dict(band="bA", lam=1.50, pix=0.032, fwhm=0.070, ellip=0.06, gain=1.0, shift_max=0.04, n_dith=2),
    "coarse": dict(band="bB", lam=1.60, pix=0.120, fwhm=0.200, ellip=0.10, gain=0.7, shift_max=0.60, n_dith=4),
    "mid": dict(band="bC", lam=1.55, pix=0.060, fwhm=0.130, ellip=0.04, gain=1.4, shift_max=0.08, n_dith=3),
}
ROLLS = [0.0, 35.0, 70.0]
HW_KERNEL = 1.2     # arcsec, truth kernel half width


# ====================================================================================================== truth
def psf_det(ux, uy, fwhm, ellip, wscale):
    """Analytic detector-frame PSF density [1/arcsec^2] at detector offsets (arcsec): Moffat + ring + spikes; unnormalised."""
    f = fwhm * wscale
    beta = 3.0
    al = f / (2 * math.sqrt(2 ** (1 / beta) - 1))
    xs, ys = ux / (1 + ellip), uy / (1 - ellip)
    r2 = xs ** 2 + ys ** 2
    core = (beta - 1) / (math.pi * al ** 2) * (1 + r2 / al ** 2) ** (-beta)
    r = np.sqrt(r2)
    rr, wr = 2.2 * f, 0.35 * f
    ring = 0.03 * np.exp(-0.5 * ((r - rr) / wr) ** 2) / (2 * math.pi * rr * wr * math.sqrt(2 * math.pi))
    ws = 0.25 * f
    sp = np.exp(-0.5 * (uy / ws) ** 2) / (1 + (np.abs(ux) / (1.5 * f)) ** 1.5) + np.exp(-0.5 * (ux / ws) ** 2) / (1 + (np.abs(uy) / (1.5 * f)) ** 1.5)
    spikes = 0.02 * sp / (2 * ws * math.sqrt(2 * math.pi) * 2 * 1.5 * f * 1.3)
    return core * 0.95 + ring + spikes


def sky_kernel(inst, roll_deg, wscale, hw=HW_KERNEL, step=SCALE / FINE, nsub=5):
    """Sky-frame kernel (per fine cell, unit sum) of a detector pixel at roll `roll_deg`: PSF in the detector frame integrated over the pixel box."""
    n = int(round(hw / step))
    d = (np.arange(-n, n + 1)) * step
    dx, dy = np.meshgrid(d, d)
    th = math.radians(roll_deg)
    c, s = math.cos(th), math.sin(th)
    # displacement in the sky frame -> detector frame: R^-1 d
    ux, uy = c * dx + s * dy, -s * dx + c * dy
    pix = inst["pix"]
    sub = (np.arange(nsub) + 0.5) / nsub * pix - pix / 2
    K = np.zeros_like(dx)
    for bx in sub:
        for by in sub:
            K += psf_det(bx - ux, by - uy, inst["fwhm"], inst["ellip"], wscale)
    K /= nsub * nsub
    return K / K.sum()


def make_truth_sky(rng, lam_grid, x_min, n_fine):
    """band-independent truth: list of (sky_fine(lam)) callables via a stack of components with power-law colours.
    Returns a function lam -> fine image."""
    yy, xx = None, None
    comps = []
    # point sources: positions in grid px inside [HALO-10, HALO+NPIX+10]
    npt = 30
    pos = rng.uniform(HALO - 8, HALO + NPIX + 8, (npt, 2))
    flux = np.concatenate([rng.uniform(1.2e4, 3.0e4, 4), np.exp(rng.uniform(math.log(150), math.log(1500), npt - 4))])
    alpha = rng.uniform(-2.0, 1.0, npt)
    return dict(pos=pos, flux=flux, alpha=alpha, x_min=x_min, n_fine=n_fine)


def render_sky_fine(T, lam):
    n, xm = T["n_fine"], T["x_min"]
    img = np.zeros((n, n))
    for (x, y), f, al in zip(T["pos"], T["flux"], T["alpha"]):
        fx, fy = (x - xm) * FINE - 0.5, (y - xm) * FINE - 0.5
        i0, j0 = int(math.floor(fx)), int(math.floor(fy))
        wx, wy = fx - i0, fy - j0
        fl = f * (lam / 1.6) ** al
        for di, wi in ((0, 1 - wx), (1, wx)):
            for dj, wj in ((0, 1 - wy), (1, wy)):
                if 0 <= i0 + di < n and 0 <= j0 + dj < n:
                    img[j0 + dj, i0 + di] += fl * wi * wj  # integrated flux deposited in the fine cell
    # surface brightness per target px^2: flux F in one fine cell of area (1/FINE)^2 target px^2 -> F / area
    img = img / (1.0 / FINE) ** 2
    # extended elliptical blob (colour alpha = -0.5) and diffuse gradient (flat colour)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    X, Y = xm + (xx + 0.5) / FINE, xm + (yy + 0.5) / FINE
    cx, cy = HALO + 0.45 * NPIX, HALO + 0.55 * NPIX
    th = math.radians(30)
    u = (X - cx) * math.cos(th) + (Y - cy) * math.sin(th)
    v = -(X - cx) * math.sin(th) + (Y - cy) * math.cos(th)
    blob = 22.0 * np.exp(-0.5 * ((u / 7.0) ** 2 + (v / 4.0) ** 2)) * (lam / 1.6) ** -0.5
    grad = 0.6 * (1 + (X - HALO) / NPIX + 0.5 * (Y - HALO) / NPIX)
    return img + blob + grad


def band_image_fine(T, lam0, nq=9, half=0.08):
    """9-node boxcar bandpass integral of the truth sky (different quadrature from the model's)."""
    lams = lam0 * (1 + 2 * half * ((np.arange(nq) + 0.5) / nq - 0.5))
    return sum(render_sky_fine(T, l) for l in lams) / nq


FULL = dict(shifts=True, dist=True, sat=True, cr=True, rolls=None)


def build_world(seed=0, quick=False, cfg=None):
    """cfg (ladder): shifts / dist / sat / cr switch the corresponding truth effects off; rolls = list (default ROLLS). The instrument set is the
    module-level INSTR (the ladder driver substitutes a subset). With a single roll every instrument gets 3x the dithers and the held-out roll is that roll."""
    cfg = dict(FULL, **(cfg or {}))
    rolls_ = cfg["rolls"] or ROLLS
    rng = np.random.default_rng(seed)
    wing = int(HW_KERNEL / SCALE)
    x_min = -(wing + 20 + HALO)               # truth sky extent (grid px)
    x_max = G + wing + 20 + HALO
    n_fine = int((x_max - x_min) * FINE)
    T = make_truth_sky(rng, None, x_min, n_fine)
    reg = O.OpReg([v["band"] for v in INSTR.values()], [v["lam"] for v in INSTR.values()], list(INSTR.keys()), ["CH_" + k for k in INSTR],
                  [False] * len(INSTR))
    # model bandpass: 3 equal-weight nodes of the same +-8 percent boxcar (quantile nodes)
    Q = 3
    lam_nodes = torch.tensor([[v["lam"] * (1 + 0.16 * ((q + 0.5) / Q - 0.5)) for q in range(Q)] for v in INSTR.values()], dtype=torch.float32)
    w_nodes = torch.full((len(INSTR), Q), 1.0 / Q)
    exps, info = [], []
    seq = 0
    splits = []
    for name, inst in INSTR.items():
        band_fine = band_image_fine(T, inst["lam"])
        pix_t = inst["pix"] / SCALE
        ncol = int(math.ceil(1.5 * NPIX / pix_t)) + 6
        # exposure plan
        nd = inst["n_dith"] * (3 if len(rolls_) == 1 else 1)
        if len(rolls_) == 1:
            plan = [(rolls_[0], "train") for _ in range(nd)] + [(rolls_[0], "val"), (rolls_[0], "T1"), (rolls_[0], "T1"), (rolls_[0], "T2")]
        else:
            plan = [(r, "train") for r in rolls_ for _ in range(nd)] + [(35.0, "val"), (0.0, "T1"), (70.0, "T1"), (105.0, "T2")]
        # distortion coefficients (detector frame, target px), static per instrument
        dist_c = rng.normal(0, 0.05, 4) * (1.0 if cfg["dist"] else 0.0)
        kern = {}
        for roll in sorted({r for r, _ in plan}):
            kern[roll] = [fftconvolve(band_fine, sky_kernel(inst, roll, ws), mode="same") for ws in (1.0, 1.12)]
        for (roll, split) in plan:
            seq += 1
            r = np.random.default_rng(1000 * seed + seq)
            th = math.radians(roll)
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            dith = r.uniform(-2.5, 2.5, 2) * pix_t * (0.3 if pix_t > 2 else 1.0)       # header-WCS dither (known)
            if inst["shift_max"] >= 0.5:
                rad = inst["shift_max"] / SCALE * math.sqrt(r.uniform(0.2, 1.0))
                ang = r.uniform(0, 2 * math.pi)
                delta = np.array([rad * math.cos(ang), rad * math.sin(ang)])
            else:
                delta = r.uniform(-1, 1, 2) * inst["shift_max"] / SCALE                  # true minus header position [target px]
            if not cfg["shifts"]:
                delta = delta * 0.0
            gain = inst["gain"] * (1 + 0.03 * r.standard_normal())
            bk = r.uniform(20, 40)
            bcoef = bk * np.array([1.0, 0.05 * r.standard_normal(), 0.05 * r.standard_normal(), 0.04 * r.standard_normal(), 0.02, 0.02])
            cc, rr = np.meshgrid(np.arange(ncol), np.arange(ncol))
            U = np.stack([(cc.ravel() - (ncol - 1) / 2) * pix_t, (rr.ravel() - (ncol - 1) / 2) * pix_t])
            xy0 = (R @ U).T + (G - 1) / 2 + dith                                            # header positions (grid px)
            keep = (xy0[:, 0] >= HALO) & (xy0[:, 0] < HALO + NPIX) & (xy0[:, 1] >= HALO) & (xy0[:, 1] < HALO + NPIX)
            col, row = cc.ravel()[keep], rr.ravel()[keep]
            dxn, dyn = (col / ncol) * 2 - 1, (row / ncol) * 2 - 1
            dist_det = np.stack([dist_c[0] * dxn * dyn + dist_c[1] * (dxn ** 2 - 0.33), dist_c[2] * dyn * dxn + dist_c[3] * (dyn ** 2 - 0.33)], 1)
            true_shift = delta[None] + (R @ dist_det.T).T                                     # total (true - header) in target px
            xt = xy0[keep] + true_shift
            idx = np.stack([(xt[:, 1] - x_min) * FINE - 0.5, (xt[:, 0] - x_min) * FINE - 0.5])
            w = (dxn + 1) / 2
            I0 = map_coordinates(kern[roll][0], idx, order=3, mode="nearest")
            I1 = map_coordinates(kern[roll][1], idx, order=3, mode="nearest")
            flux = gain * ((1 - w) * I0 + w * I1)
            sat = SAT * np.tanh(flux / SAT) if cfg["sat"] else flux
            leg = V._legendre2d(torch.tensor(dxn, dtype=torch.float32), torch.tensor(dyn, dtype=torch.float32), 2).numpy()
            bkg = leg @ bcoef
            expect = sat + bkg
            sig = np.sqrt(1.5 ** 2 + np.maximum(sat, 0) / 50.0)
            d = expect + sig * r.standard_normal(expect.shape)
            cr = (r.random(expect.shape) < 0.01) & cfg["cr"]
            d = d + cr * r.exponential(40.0, expect.shape)
            n = d.size
            t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt)  # noqa: E731
            e = dict(name=f"{name}_r{int(roll):03d}_{seq:02d}", band=inst["band"], tx=t(xy0[keep, 0]), ty=t(xy0[keep, 1]), d=t(d), ivar=t(1.0 / sig ** 2),
                     row=t(row, torch.long), col=t(col, torch.long), dx=t(dxn), dy=t(dyn), bkg0=float(np.median(d)), detname=name.upper() + "1",
                     mjd=59000.0 + 100 * seq / 10, exptime=300.0, pixar_sr_nominal=inst['pix'] ** 2 * V.ARCSEC2_TO_SR, holdout=split in ("T1", "T2"), val=split == "val")
            exps.append(e)
            info.append(dict(split=split, inst=name, roll=roll, delta=delta, dist_c=dist_c, gain=gain, ncol=ncol, pix_t=pix_t, R=R, dith=dith,
                             cr=cr, oracle=expect, sat=sat, true_shift_pix=true_shift, bcoef=bcoef))
    return dict(exps=exps, info=info, reg=reg, bandpass=(lam_nodes, w_nodes), T=T, x_min=x_min, n_fine=n_fine)


def classical_splat(thead):
    def fn(exps, th, kernel, pixfrac, dev, bands=None):
        out, cover = [], {}
        for b in (bands or fn.bands):
            es = [e for e in exps if e["band"] == b and not e["holdout"] and not e.get("val")]
            img, wsum = V.splat_coadd(es, th, dev)
            cover[b] = wsum > 0
            out.append(torch.nan_to_num(img))
        fn.last_cover = cover
        return torch.stack(out)
    return fn


# ====================================================================================================== analysis
def true_kernel_grid(inst_name, roll, half=31.0, pitch=1.0, f=1.0):
    """Truth sky-frame kernel at the detector centre (width scale 1.06) rasterised on the common target-px grid."""
    inst = INSTR[inst_name]
    K = 0.5 * (sky_kernel(inst, roll, f * 1.0) + sky_kernel(inst, roll, f * 1.12))
    n = (K.shape[0] - 1) // 2
    step = 1.0 / FINE
    d = np.arange(-n, n + 1) * step
    X, Y = np.meshgrid(d, d)
    return O.rasterize(np.stack([X.ravel(), Y.ravel()], 1), K.ravel() / K.sum(), half, pitch) * pitch ** 2, K


def learned_kernel_grid(model, eid, nuis=None, half=31.0, pitch=1.0):
    kp = O.kernel_points(model, eid, 0.0, 0.0, nuis)
    return O.rasterize(kp["pos"], kp["mass"], half, pitch) * pitch ** 2, kp



def registration_stats(model, exps, info, train_idx, keep=None):
    """Recovered mu on a 9x9 lattice (inside the data window) minus the true registration (offset + rotated static distortion), common shift
    (mean over training exposures) removed. rms in mas per instrument for training and held-out exposures."""
    names = list(INSTR)
    keep = keep or {}
    lat = torch.linspace(-0.9, 0.9, 9)
    yy, xx = torch.meshgrid(lat, lat, indexing="ij")
    err, tru, ids = [], [], []
    with torch.no_grad():
        for i, inf in enumerate(info):
            if not keep and i not in train_idx:
                continue
            nz, nid = (keep[i][0], 0) if i in keep else (model.nuis, i)
            eid = torch.full((81,), i, dtype=torch.long)
            h = model.trunk(model.ctx_input(eid, xx.reshape(-1), yy.reshape(-1), torch.full((81,), nid, dtype=torch.long), nz))
            mu = model.heads(h, eid, (xx.reshape(-1), yy.reshape(-1), torch.full((81,), nid, dtype=torch.long), nz))[0].numpy()
            dxn, dyn = xx.reshape(-1).numpy(), yy.reshape(-1).numpy()
            dist_c = inf["dist_c"]
            dd = np.stack([dist_c[0] * dxn * dyn + dist_c[1] * (dxn ** 2 - 0.33), dist_c[2] * dyn * dxn + dist_c[3] * (dyn ** 2 - 0.33)], 1)
            ts = inf["delta"][None] + (inf["R"] @ dd.T).T
            col, row = (dxn + 1) / 2 * inf["ncol"], (dyn + 1) / 2 * inf["ncol"]
            U = np.stack([(col - (inf["ncol"] - 1) / 2) * inf["pix_t"], (row - (inf["ncol"] - 1) / 2) * inf["pix_t"]])
            x0 = (inf["R"] @ U).T + (G - 1) / 2 + inf["dith"]
            ok = (x0[:, 0] >= HALO) & (x0[:, 0] < HALO + NPIX) & (x0[:, 1] >= HALO) & (x0[:, 1] < HALO + NPIX)
            err.append((mu - ts)[ok]); tru.append(ts[ok]); ids.append(np.full(ok.sum(), i))
    err_all, tru_all, ids_all = np.concatenate(err), np.concatenate(tru), np.concatenate(ids)
    trs = set(train_idx)
    tr_mask = np.array([i in trs for i in ids_all])
    gauge = err_all[tr_mask].mean(0)                                  # common shift: the one thing the data cannot fix
    res = err_all - gauge
    rows = {}
    for inst in names + ["all"]:
        sel = np.array([(inst == "all") or info[i]["inst"] == inst for i in ids_all])
        for sp in ("train", "heldout"):
            ssel = sel & (tr_mask if sp == "train" else ~tr_mask)
            if ssel.any():
                rows[f"{inst}/{sp}"] = dict(rms_resid_mas=float(np.sqrt((res[ssel] ** 2).sum(1).mean()) * SCALE * 1000),
                                            rms_true_mas=float(np.sqrt(((tru_all[ssel] - tru_all[tr_mask].mean(0)) ** 2).sum(1).mean()) * SCALE * 1000))
    splits = np.array([info[i]["split"] for i in ids_all])
    for inst in names + ["all"]:
        sel = np.array([(inst == "all") or info[i]["inst"] == inst for i in ids_all])
        for sp in sorted(set(splits)):
            ssel = sel & (splits == sp)
            if ssel.any():
                rows[f"{inst}/@{sp}"] = dict(rms_resid_mas=float(np.sqrt((res[ssel] ** 2).sum(1).mean()) * SCALE * 1000),
                                             rms_true_mas=float(np.sqrt(((tru_all[ssel] - tru_all[tr_mask].mean(0)) ** 2).sum(1).mean()) * SCALE * 1000))
    return dict(common_shift_gauge_mas=(gauge * SCALE * 1000).tolist(), rows=rows)


def analyse(ctx, world, label=""):
    model, exps, info = ctx["model"], ctx["exps"], world["info"]
    R = {}
    names = list(INSTR)
    # ---- held-out scores vs oracle
    M = ctx["log"]["metrics"]
    keep = ctx["keep"]
    sm = ctx["src_masks"]
    rows = {}
    for i, inf in enumerate(info):
        if i not in keep:
            continue
        e = exps[i]
        b = ctx["reg"].index[e["band"]]
        m = keep[i][1]["m"]
        r = e["d"] - m
        chi = float((e["ivar"] * r * r).mean())
        orc = torch.tensor(inf["oracle"], dtype=torch.float32)
        ro = e["d"] - orc
        chi_o = float((e["ivar"] * ro * ro).mean())
        srcm = V.sample(sm[b][None].float(), e["tx"], e["ty"])[0] > 0.5
        chi_s = float((e["ivar"] * r * r)[srcm].mean()) if srcm.any() else float("nan")
        chi_os = float((e["ivar"] * ro * ro)[srcm].mean()) if srcm.any() else float("nan")
        cr = torch.tensor(inf["cr"])
        chi_ncr = float((e["ivar"] * r * r)[~cr].mean())
        chi_o_ncr = float((e["ivar"] * ro * ro)[~cr].mean())
        rows[e["name"]] = dict(split=inf["split"], inst=inf["inst"], chi2_all=chi, oracle_all=chi_o, chi2_bright=chi_s, oracle_bright=chi_os, n_bright=int(srcm.sum()),
                               chi2_noCR=chi_ncr, oracle_noCR=chi_o_ncr)
    R["heldout"] = rows
    agg = {}
    for split in ("T1", "T2", "T3", "val", "train"):
        for inst in names + ["all"]:
            sel = [v for v in rows.values() if v["split"] == split and (inst == "all" or v["inst"] == inst)]
            if sel:
                agg[f"{split}/{inst}"] = {k: float(np.mean([v[k] for v in sel if np.isfinite(v[k])])) for k in ("chi2_all", "oracle_all", "chi2_bright", "oracle_bright", "chi2_noCR", "oracle_noCR")}
    R["heldout_agg"] = agg
    # ---- registration
    R["registration"] = registration_stats(model, exps, info, ctx["train_idx"], keep)
    # ---- amplitude (flux) ratios: exp(a) at the detector centre, relative to the reference instrument
    amp = {}
    for inst in names:
        ids_i = [i for i in ctx["train_idx"] if info[i]["inst"] == inst]
        vals = []
        for i in ids_i:
            kp = O.kernel_points(model, i)
            vals.append(kp["amp"] / info[i]["gain"] * INSTR[inst]["gain"])
        amp[inst] = dict(exp_a_over_true_gain_mean=float(np.mean(vals)), std=float(np.std(vals)))
    ref = amp["sharp"]["exp_a_over_true_gain_mean"]
    R["amplitude"] = {k: dict(v, relative_to_sharp=v["exp_a_over_true_gain_mean"] / ref) for k, v in amp.items()}
    # ---- kernels
    Kt, Kl, nm = [], [], []
    for inst in names:
        i = next(i for i in ctx["train_idx"] if info[i]["inst"] == inst and info[i]["roll"] == 0.0)
        kt, _ = true_kernel_grid(inst, 0.0, f=info[i].get("wsc", 1.0))
        kl, kp = learned_kernel_grid(model, i)
        Kt.append(kt); Kl.append(kl); nm.append(inst)
    sharp = 0
    try:
        cm_, rk_ = common_mode(Kt, Kl), rel_kernel_err(Kt, Kl, sharp)
    except Exception as ex_:  # a single instrument has no relative kernel
        cm_, rk_ = dict(raw=float("nan"), free_g=float("nan"), gauss_g=float("nan"), gauss_s2=float("nan")), []
    R["kernels"] = dict(common_mode=cm_, relative_vs_sharp=rk_,
                        ee_truth=[ee_curve(k, 1.0, [1.0, 2.0, 4.0, 8.0]) for k in Kt], ee_learned=[ee_curve(k, 1.0, [1.0, 2.0, 4.0, 8.0]) for k in Kl],
                        ee_radii_target_px=[1.0, 2.0, 4.0, 8.0])
    # all rolls: error of the learned kernel after removing the single g fitted on roll 0, per instrument (relative L2 in Fourier space inside the support)
    allr = {}
    for roll in ROLLS + [105.0]:
        Kt_r, Kl_r = [], []
        for inst in names:
            cand = [i for i in range(len(exps)) if info[i]["inst"] == inst and info[i]["roll"] == roll]
            if not cand:
                break
            i = cand[0]
            nz = keep[i][0] if i in keep else None
            kt, _ = true_kernel_grid(inst, roll, f=info[i].get("wsc", 1.0))
            kl, _ = learned_kernel_grid(model, i, nz)
            Kt_r.append(kt); Kl_r.append(kl)
        if len(Kt_r) == len(names):
            allr[f"roll{int(roll)}"] = dict(common_mode=common_mode(Kt_r, Kl_r), relative_vs_sharp=rel_kernel_err(Kt_r, Kl_r, 0))
    R["kernels_by_roll"] = allr
    # ---- outliers (held-out and validation exposures): posterior > 0.5 vs cosmic rays
    tp = fp = fn = 0
    tp5 = fn5 = 0
    for i in keep:
        e = exps[i]
        P = keep[i][1]
        sp = 1.0 / e["ivar"].sqrt()
        nll, pout = O.nll_terms(e["d"], P["m"], sp, P["h"], P["pi"])
        flag = (pout > 0.5).numpy()
        cr = info[i]["cr"]
        amp_cr = (e["d"] - torch.tensor(info[i]["oracle"], dtype=torch.float32)).numpy() / sp.numpy()   # CR excess in sigma (incl. noise)
        tp += int((flag & cr).sum()); fp += int((flag & ~cr).sum()); fn += int((~flag & cr).sum())
        big = cr & (amp_cr > 5)
        tp5 += int((flag & big).sum()); fn5 += int((~flag & big).sum())
    R["outliers"] = dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), recall_cr_gt_5sigma=tp5 / max(tp5 + fn5, 1), tp=tp, fp=fp, fn=fn,
                         n_cr_gt_5sigma=tp5 + fn5, mean_pi=float(np.mean([float(keep[i][1]["pi"].mean()) for i in keep])))
    return R


def run_variant(world, out, steps, c2f_frac, label, quick=False, budget_min=20.0, extra=(), hook=None):
    exps = [dict(e) for e in world["exps"]]
    args = ["--out", out, "--size", str(NPIX), "--halo-px", str(HALO), "--scale", str(SCALE), "--steps", str(steps), "--budget-min", str(budget_min),
            "--batch", "1024", "--hidden", "48", "--n-bank", "12", "--shape-hidden", "32", "--spec-k", "6", "--spec-q", "3",
            "--core-radius", "2.0", "--core-pitch", "0.5", "--wing-ratio", "1.5", "--wing-nang", "12", "--wing-arcsec", str(HW_KERNEL),
            "--c2f-frac", str(c2f_frac), "--c2f-start-arcsec", "0.4", "--nuis-steps", "150" if not quick else "40", "--val-nuis-steps", "60",
            "--val-every-min", "1000", "--ref-instrument", "sharp", "--boot-b", "50", "--log-every", "100", "--lr-op", "3e-3",
            "--lr-kernel", "4e-3", "--lam-neg", "1.0", "--seed", "0", "--n-val", "1"] + list(extra)
    a = O.parse_args(args)
    a.hook = hook
    a.size_data = NPIX
    thead = V.target_header(110.0, -70.0, G, SCALE)
    cfn = classical_splat(thead)
    cfn.bands = world["reg"].bands
    t0 = time.time()
    ctx = O.run(exps, thead, DEV, a, out, classical_fn=cfn, reg=world["reg"], bandpass=world["bandpass"], preselected=True)
    ctx["seconds"] = time.time() - t0
    return ctx, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "opfield_synth"))
    ap.add_argument("--variants", default="c2f,noc2f")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    world = build_world(0, a.quick)
    print(f"world built in {time.time() - t0:.0f} s: {len(world['exps'])} exposures, {sum(e['d'].numel() for e in world['exps'])} pixels", flush=True)
    results = {}
    for var in a.variants.split(","):
        out = os.path.join(a.out, var)
        ctx, args = run_variant(world, out, a.steps if not a.quick else 60, 0.3 if var == "c2f" else 0.0, var, a.quick)
        R = analyse(ctx, world, var)
        R["seconds"] = ctx["seconds"]
        R["train_steps"] = ctx["log"].get("train_steps")
        results[var] = R
        json.dump(results, open(os.path.join(a.out, "synth_results.json"), "w"), indent=1, default=float)
        print(f"=== {var}: {ctx['seconds']:.0f} s, {R['train_steps']} steps")
        for k, v in R["heldout_agg"].items():
            print(f"  chi2 {k}: all {v['chi2_all']:.3f} (oracle {v['oracle_all']:.3f}); bright {v['chi2_bright']:.3f} (oracle {v['oracle_bright']:.3f}); no-CR {v['chi2_noCR']:.3f} (oracle {v['oracle_noCR']:.3f})")
        for k, v in R["registration"]["rows"].items():
            print(f"  registration {k}: residual rms {v['rms_resid_mas']:.1f} mas (true rms {v['rms_true_mas']:.1f} mas)")
        print("  amplitude ratios:", {k: round(v["relative_to_sharp"], 3) for k, v in R["amplitude"].items()}, "(truth 1.0 / 0.7 / 1.4)")
        c = R["kernels"]["common_mode"]
        print(f"  common mode (roll 0): raw {c['raw']:.3f} free g {c['free_g']:.3f} gauss g {c['gauss_g']:.3f} (s2 {c['gauss_s2']:+.2f}); rel kernels vs sharp {R['kernels']['relative_vs_sharp']}")
        print("  outliers:", R["outliers"])
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
