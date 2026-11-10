"""Synthetic generator with a TIME axis (independent renderer, extends synth_opfield).
Epochs at years 0, 2, 5 (MJD 58000 + 365.25 yr); 2 visits per epoch per instrument, several dithers per visit.
Truth operator drift: PSF width scale = (1 + epoch drift (|.| <= 10 %)) x (1 + visit drift (|.| <= 5 %)); the distortion coefficients drift per epoch (sd 0.02 target px
on top of the static sd 0.05); pointing offset per VISIT (coarse up to 0.6 arcsec, sharp 0.04, mid 0.08) shared by all dithers of the visit + 10 mas jitter per exposure;
the roll is per visit. Truth sky: static except 3 stars with proper motion (<= 50 mas/yr) and 2 variable sources (+-30 % per epoch).
Splits: T0 train = epochs 0 and 5 (all dithers of the training visits except the extra ones), T1 = one extra dither of two training visits per instrument,
val = one extra dither, T3 = epoch at year 2 (both visits, one dither each), never seen in training.  `wide=True`: 2x linear field (4x area) with 4x the sources."""
import math
import os
import sys

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth_opfield as S  # noqa: E402

V, O = S.V, S.O
YEARS = [0.0, 2.0, 5.0]
TRAIN_EPOCHS = [0, 2]
NV = 2
DITH = {"sharp": 2, "coarse": 3, "mid": 2}
ROLLS_V = [0.0, 35.0, 70.0, 0.0, 35.0, 70.0]   # visit k of the whole sequence -> roll


def make_truth_time(rng, x_min, n_fine, wide=False):
    npt = int(30 * (S.NPIX / 80.0) ** 2) if wide else 30
    pos = rng.uniform(S.HALO - 8, S.HALO + S.NPIX + 8, (npt, 2))
    flux = np.concatenate([rng.uniform(1.2e4, 3.0e4, 4), np.exp(rng.uniform(math.log(150), math.log(1500), npt - 4))])
    alpha = rng.uniform(-2.0, 1.0, npt)
    order = 4 + np.argsort(-flux[4:])
    mov, var = order[:3], order[3:5]                              # the 3 brightest faint stars move, the next 2 vary
    pm = np.zeros((npt, 2))
    ang = rng.uniform(0, 2 * math.pi, 3)
    pm[mov] = np.stack([np.cos(ang), np.sin(ang)], 1) * (0.05 / S.SCALE) * rng.uniform(0.7, 1.0, 3)[:, None]   # target px / yr (<= 50 mas/yr)
    vmul = np.ones((npt, len(YEARS)))
    vmul[var] = 1.0 + 0.3 * np.array([[1.0, -1.0, 0.5], [-1.0, 0.8, -0.6]])
    return dict(pos=pos, flux=flux, alpha=alpha, x_min=x_min, n_fine=n_fine, pm=pm, vmul=vmul, mov=mov, var=var)


def render_t(T, lam, ep):
    T2 = dict(T)
    T2["pos"] = T["pos"] + T["pm"] * YEARS[ep]
    T2["flux"] = T["flux"] * T["vmul"][:, ep]
    return S.render_sky_fine(T2, lam)


def band_t(T, lam0, ep, nq=9, half=0.08):
    lams = lam0 * (1 + 2 * half * ((np.arange(nq) + 0.5) / nq - 0.5))
    return sum(render_t(T, l, ep) for l in lams) / nq


def build_world_time(seed=0, wide=False):
    rng = np.random.default_rng(seed)
    wing = int(S.HW_KERNEL / S.SCALE)
    x_min = -(wing + 20 + S.HALO)
    x_max = S.G + wing + 20 + S.HALO
    n_fine = int((x_max - x_min) * S.FINE)
    T = make_truth_time(rng, x_min, n_fine, wide)
    INSTR = S.INSTR
    reg = O.OpReg([v["band"] for v in INSTR.values()], [v["lam"] for v in INSTR.values()], list(INSTR.keys()), ["CH_" + k for k in INSTR], [False] * len(INSTR))
    Q = 3
    lam_nodes = torch.tensor([[v["lam"] * (1 + 0.16 * ((q + 0.5) / Q - 0.5)) for q in range(Q)] for v in INSTR.values()], dtype=torch.float32)
    w_nodes = torch.full((len(INSTR), Q), 1.0 / Q)
    ep_w = rng.uniform(-0.10, 0.10, len(YEARS))
    exps, info = [], []
    seq = 0
    for ii, (name, inst) in enumerate(INSTR.items()):
        pix_t = inst["pix"] / S.SCALE
        ncol = int(math.ceil(1.5 * S.NPIX / pix_t)) + 6
        dist_c0 = rng.normal(0, 0.05, 4)
        dist_ep = [dist_c0 + rng.normal(0, 0.02, 4) for _ in YEARS]
        band_ep = [band_t(T, inst["lam"], k) for k in range(len(YEARS))]
        plan = []   # (epoch, visit, dither, split)
        for k, ep in enumerate(range(len(YEARS))):
            for v in range(NV):
                if ep in TRAIN_EPOCHS:
                    plan += [(ep, v, d, "train") for d in range(DITH[name])]
                else:
                    plan += [(ep, v, 0, "T3")]
        plan += [(0, 0, 90, "T1"), (2, 1, 90, "T1"), (2, 0, 91, "val")]
        kcache = {}
        vstate = {}
        for (ep, v, dith_i, split) in plan:
            vk = (ep, v)
            if vk not in vstate:
                r = np.random.default_rng(777 * seed + 31 * ii + 7 * ep + v + 1)
                rad = inst["shift_max"] / S.SCALE * math.sqrt(r.uniform(0.2, 1.0))
                if inst["shift_max"] < 0.5:
                    rad = inst["shift_max"] / S.SCALE * r.uniform(0.0, 1.0)
                an = r.uniform(0, 2 * math.pi)
                vstate[vk] = dict(delta=np.array([rad * math.cos(an), rad * math.sin(an)]), wsc=(1 + ep_w[ep]) * (1 + r.uniform(-0.05, 0.05)),
                                  roll=ROLLS_V[ep * NV + v], dith0=r.uniform(-2.5, 2.5, 2) * pix_t * (0.3 if pix_t > 2 else 1.0))
            vs = vstate[vk]
            roll, wsc = vs["roll"], vs["wsc"]
            if vk not in kcache:
                kcache[vk] = [fftconvolve(band_ep[ep], S.sky_kernel(inst, roll, ws * wsc), mode="same") for ws in (1.0, 1.12)]
            seq += 1
            r = np.random.default_rng(1000 * seed + seq)
            th = math.radians(roll)
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            dith = vs["dith0"] + r.uniform(-1.0, 1.0, 2) * pix_t * (0.3 if pix_t > 2 else 1.0) * (1 + dith_i % 5) * 0.5   # header-WCS dither within the visit
            delta = vs["delta"] + r.standard_normal(2) * 0.010 / S.SCALE                                                 # visit pointing + 10 mas jitter
            gain = inst["gain"] * (1 + 0.03 * r.standard_normal())
            bk = r.uniform(20, 40)
            bcoef = bk * np.array([1.0, 0.05 * r.standard_normal(), 0.05 * r.standard_normal(), 0.04 * r.standard_normal(), 0.02, 0.02])
            cc, rr = np.meshgrid(np.arange(ncol), np.arange(ncol))
            U = np.stack([(cc.ravel() - (ncol - 1) / 2) * pix_t, (rr.ravel() - (ncol - 1) / 2) * pix_t])
            xy0 = (R @ U).T + (S.G - 1) / 2 + dith
            keep = (xy0[:, 0] >= S.HALO) & (xy0[:, 0] < S.HALO + S.NPIX) & (xy0[:, 1] >= S.HALO) & (xy0[:, 1] < S.HALO + S.NPIX)
            col, row = cc.ravel()[keep], rr.ravel()[keep]
            dxn, dyn = (col / ncol) * 2 - 1, (row / ncol) * 2 - 1
            dc = dist_ep[ep]
            dist_det = np.stack([dc[0] * dxn * dyn + dc[1] * (dxn ** 2 - 0.33), dc[2] * dyn * dxn + dc[3] * (dyn ** 2 - 0.33)], 1)
            true_shift = delta[None] + (R @ dist_det.T).T
            xt = xy0[keep] + true_shift
            idx = np.stack([(xt[:, 1] - x_min) * S.FINE - 0.5, (xt[:, 0] - x_min) * S.FINE - 0.5])
            w = (dxn + 1) / 2
            I0 = map_coordinates(kcache[vk][0], idx, order=3, mode="nearest")
            I1 = map_coordinates(kcache[vk][1], idx, order=3, mode="nearest")
            flux = gain * ((1 - w) * I0 + w * I1)
            sat = S.SAT * np.tanh(flux / S.SAT)
            leg = V._legendre2d(torch.tensor(dxn, dtype=torch.float32), torch.tensor(dyn, dtype=torch.float32), 2).numpy()
            expect = sat + leg @ bcoef
            sig = np.sqrt(1.5 ** 2 + np.maximum(sat, 0) / 50.0)
            d = expect + sig * r.standard_normal(expect.shape)
            cr = r.random(expect.shape) < 0.01
            d = d + cr * r.exponential(40.0, expect.shape)
            t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt)  # noqa: E731
            mjd = 58000.0 + 365.25 * YEARS[ep] + 20.0 * v + 0.01 * seq
            nm = f"jw90000{ii + 1:03d}{ep * 10 + v + 1:03d}_00001_{seq:05d}_{name}_cal"
            exps.append(dict(name=nm, band=inst["band"], tx=t(xy0[keep, 0]), ty=t(xy0[keep, 1]), d=t(d), ivar=t(1.0 / sig ** 2), row=t(row, torch.long), col=t(col, torch.long),
                             dx=t(dxn), dy=t(dyn), bkg0=float(np.median(d)), detname=name.upper() + "1", mjd=mjd, exptime=300.0,
                             pixar_sr_nominal=inst["pix"] ** 2 * V.ARCSEC2_TO_SR, holdout=split in ("T1", "T3"), val=split == "val"))
            info.append(dict(split=split, inst=name, roll=roll, delta=delta, dist_c=dc, gain=gain, ncol=ncol, pix_t=pix_t, R=R, dith=dith, cr=cr, oracle=expect, sat=sat,
                             true_shift_pix=true_shift, bcoef=bcoef, wsc=wsc, epoch=ep, visit=v, year=YEARS[ep], xt=xt))
    return dict(exps=exps, info=info, reg=reg, bandpass=(lam_nodes, w_nodes), T=T, x_min=x_min, n_fine=n_fine)


def moving_star_stats(ctx, world, keep_splits=("T1", "T3")):
    """At the moving / variable stars (true position at the exposure epoch), for held-out exposures: aperture (r < 3 target px) chi2 of the model and of the oracle,
    and the local deviation of the fitted registration mu from the smooth field (mean mu in the aperture minus mean mu in the annulus 8-16 px), compared with the
    true displacement of the star relative to epoch 0 (pm x year). Returns per-class means."""
    T, info, exps, keep = world["T"], world["info"], ctx["exps"], ctx["keep"]
    out = {}
    for cls, ids in (("moving", T["mov"]), ("variable", T["var"]), ("static_bright", np.arange(4))):
        rows = []
        for i in keep:
            inf = info[i]
            if inf["split"] not in keep_splits:
                continue
            e, P = exps[i], keep[i][1]
            xt = inf["xt"]
            for j in ids:
                sp = T["pos"][j] + T["pm"][j] * inf["year"]
                dd = np.hypot(xt[:, 0] - sp[0], xt[:, 1] - sp[1])
                ap, an = dd < 3.0, (dd > 8) & (dd < 16)
                if ap.sum() < 3 or an.sum() < 5:
                    continue
                m = P["m"].numpy()
                d = e["d"].numpy()
                iv = e["ivar"].numpy()
                orc = inf["oracle"]
                chi = float((iv * (d - m) ** 2)[ap].mean())
                chi_o = float((iv * (d - orc) ** 2)[ap].mean())
                mu = P["mu"].numpy() if "mu" in P else None
                dev = float(np.hypot(*(mu[ap].mean(0) - mu[an].mean(0)))) * S.SCALE * 1000 if mu is not None else float("nan")
                disp = float(np.hypot(*(T["pm"][j] * inf["year"]))) * S.SCALE * 1000
                rows.append((chi, chi_o, dev, disp))
        if rows:
            a = np.array(rows)
            out[cls] = dict(n=len(rows), chi2_aperture=float(a[:, 0].mean()), chi2_oracle=float(a[:, 1].mean()), local_mu_dev_mas=float(np.nanmean(a[:, 2])),
                            true_displacement_mas=float(a[:, 3].mean()))
    return out
