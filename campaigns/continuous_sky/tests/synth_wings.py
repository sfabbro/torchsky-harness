"""Synthetic check of the wing / shoulder options of scripts/opfield.py (CPU, ~15-40 min): can the operator fit a PSF with STRONG WINGS?

TRUTH (a small dedicated generator with an independent renderer: no FFT, no stencil, no table; each pixel is the analytic PSF evaluated at the detector
offsets of every star, 3 x 3 supersampled over the pixel box):
  PSF in the DETECTOR frame (rotates with the roll through J), unit sum:
     (1 - eps_h - eps_s) Gaussian(sigma 2 px) + eps_h power-law halo  (1 + (r / 3 px)^2)^-1.5 / (2 pi 3^2),  eps_h = 0.05  (3e-3 of the peak at 5 px, 5e-4 at 10, 2e-4 at 15)
                                                     + eps_s four diffraction spikes (along +-x, +-y rotated by 45 degrees; Gaussian cross-width 0.8 px, profile 1/(1 + r/4 px)), eps_s = 0.02
  5 bright stars (4e4 - 2e5 counts) + 12 faint ones (300 - 3000), pixel = sky grid (0.04 arcsec), no registration error, no distortion, flat gain 1,
  background 30 + noise sqrt(1.5^2 + model / 100) (the pipeline ivar is the true variance), rolls 0 / 40 / 80 degrees (3 dithers each) for training,
  held out: 2 new dithers at rolls 0 and 80 and one at a NEW roll 120.
SCORE: held-out chi2 per pixel (pipeline ivar) in annuli of the distance [detector px] to the nearest BRIGHT star: 0-1, 1-3 (shoulder), 3-8, 8-15 (wing), 15-30,
  against the oracle (truth expectation on the same pixels), for the variants
  default | log (--psi-param log) | is (--importance-sampling) | ts (--two-scale) | ts_is (two-scale + importance sampling) | all (the three).
ASSERTION (--assert): the combination ts_is has a lower chi2 than default in the 3-15 px annulus (sum of the 3-8 and 8-15 annuli, pixel weighted).

Run: venvT/bin/python tests/synth_wings.py [--steps 1200] [--variants default,ts_is] [--out runs/synth_wings]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import opfield as O  # noqa: E402
import smacs_four_v36 as V  # noqa: E402

torch.set_num_threads(2)
DEV = torch.device("cpu")
SCALE = 0.04
NPIX, HALO = 72, 30
G = NPIX + 2 * HALO
EPS_H, EPS_S, SIG_CORE = 0.05, 0.02, 2.0
ROLLS_TRAIN = [0.0, 40.0, 80.0]
ANN = [0.0, 1.0, 3.0, 8.0, 15.0, 30.0]


def psf_det(ux, uy):
    """Analytic detector-frame PSF density [1/px^2] at detector offsets (px)."""
    r2 = ux ** 2 + uy ** 2
    core = np.exp(-0.5 * r2 / SIG_CORE ** 2) / (2 * math.pi * SIG_CORE ** 2)
    r0 = 3.0
    halo = (1 + r2 / r0 ** 2) ** -1.5 / (2 * math.pi * r0 ** 2)
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    vx, vy = c * ux + s * uy, -s * ux + c * uy                              # spikes along the rotated axes
    w = 0.8
    sp = (np.exp(-0.5 * (vy / w) ** 2) / (1 + np.abs(vx) / 4.0) + np.exp(-0.5 * (vx / w) ** 2) / (1 + np.abs(vy) / 4.0))
    # normalise the spike pattern numerically on a big grid (cached)
    return (1 - EPS_H - EPS_S) * core + EPS_H * halo + EPS_S * sp / psf_det.spike_norm


def _spike_norm():
    ax = np.arange(-400, 401) * 0.25
    X, Y = np.meshgrid(ax, ax)
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    vx, vy = c * X + s * Y, -s * X + c * Y
    w = 0.8
    sp = (np.exp(-0.5 * (vy / w) ** 2) / (1 + np.abs(vx) / 4.0) + np.exp(-0.5 * (vx / w) ** 2) / (1 + np.abs(vy) / 4.0))
    return sp.sum() * 0.25 ** 2


psf_det.spike_norm = _spike_norm()


def build_world(seed=0):
    rng = np.random.default_rng(seed)
    nb, nf = 5, 12
    pos = rng.uniform(HALO + 4, HALO + NPIX - 4, (nb + nf, 2))
    # keep the bright stars apart (> 24 px) so the annuli are clean
    for k in range(nb):
        for _ in range(200):
            if all(np.hypot(*(pos[k] - pos[j])) > 24 for j in range(k)):
                break
            pos[k] = rng.uniform(HALO + 4, HALO + NPIX - 4, 2)
    flux = np.concatenate([rng.uniform(4e4, 2e5, nb), np.exp(rng.uniform(math.log(300), math.log(3000), nf))])
    reg = O.OpReg(["bA"], [1.5], ["sharp"], ["CH_sharp"], [False])
    Q = 3
    lam_nodes = torch.tensor([[1.5 * (1 + 0.16 * ((q + 0.5) / Q - 0.5)) for q in range(Q)]], dtype=torch.float32)
    w_nodes = torch.full((1, Q), 1.0 / Q)
    plan = [(r, "train") for r in ROLLS_TRAIN for _ in range(3)] + [(0.0, "T1"), (80.0, "T1"), (120.0, "T2")]
    exps, info = [], []
    ss = (np.arange(3) + 0.5) / 3 - 0.5
    for seq, (roll, split) in enumerate(plan, 1):
        r = np.random.default_rng(1000 * seed + seq)
        th = math.radians(roll)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        dith = r.uniform(-2.5, 2.5, 2)
        n = int(math.ceil(1.5 * NPIX)) + 6
        cc, rr = np.meshgrid(np.arange(n), np.arange(n))
        U = np.stack([(cc.ravel() - (n - 1) / 2), (rr.ravel() - (n - 1) / 2)])
        xy0 = (R @ U).T + (G - 1) / 2 + dith
        keep = (xy0[:, 0] >= HALO) & (xy0[:, 0] < HALO + NPIX) & (xy0[:, 1] >= HALO) & (xy0[:, 1] < HALO + NPIX)
        col, row = cc.ravel()[keep], rr.ravel()[keep]
        xy = xy0[keep]
        dxn, dyn = (col / n) * 2 - 1, (row / n) * 2 - 1
        expect = np.zeros(len(xy))
        Rinv = R.T
        for (sx, sy), f in zip(pos, flux):
            for bx in ss:                                                    # 3 x 3 supersampling over the pixel box (detector frame)
                for by in ss:
                    du = Rinv @ np.stack([sx - xy[:, 0], sy - xy[:, 1]]) - np.array([[bx], [by]])
                    expect += f * psf_det(du[0], du[1]) / 9.0                           # the PSF is even in u
        bkg = 30.0
        sig = np.sqrt(1.5 ** 2 + np.maximum(expect, 0) / 100.0)
        d = expect + bkg + sig * r.standard_normal(expect.shape)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt)  # noqa: E731
        e = dict(name=f"wing_r{int(roll):03d}_{seq:02d}", band="bA", tx=t(xy[:, 0]), ty=t(xy[:, 1]), d=t(d), ivar=t(1.0 / sig ** 2), row=t(row, torch.long),
                 col=t(col, torch.long), dx=t(dxn), dy=t(dyn), bkg0=float(np.median(d)), detname="SHARP1", mjd=59000.0 + 10 * seq, exptime=300.0,
                 pixar_sr_nominal=SCALE ** 2 * V.ARCSEC2_TO_SR, holdout=split in ("T1", "T2"), val=False)
        exps.append(e)
        info.append(dict(split=split, roll=roll, expect=expect, xy=xy))
    return dict(exps=exps, info=info, reg=reg, bandpass=(lam_nodes, w_nodes), pos=pos, flux=flux, nbright=nb)


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


VARIANTS = {
    "default": [],
    "log": ["--psi-param", "log"],
    "is": ["--importance-sampling", "1,3,8,15,30", "--is-refresh", "100", "--is-snr", "50"],
    "ts": ["--two-scale", "0.24"],
    "ts_is": ["--two-scale", "0.24", "--importance-sampling", "1,3,8,15,30", "--is-refresh", "100", "--is-snr", "50"],
    "ts12": ["--two-scale", "0.48"],                                          # per-pixel stencil to 12 px (dense core 6 px + 3 rings), wing table beyond (the geometry of the real runs: R_core beyond the dense core)
    "ts12_is": ["--two-scale", "0.48", "--importance-sampling", "1,3,8,15,30", "--is-refresh", "100", "--is-snr", "50"],
    "core_only": ["--wing-arcsec", "0.24"],                                   # diagnostics: the default operator truncated at 6 px (no wings at all)
    "ts_frozen": ["--two-scale", "0.24", "--wing-lr-mult", "0"],              # diagnostics: two-scale with the wing table / fraction frozen at the start
    "all": ["--psi-param", "log", "--two-scale", "0.24", "--importance-sampling", "1,3,8,15,30", "--is-refresh", "100", "--is-snr", "50"],
}


def run_variant(world, out, steps, extra):
    exps = [dict(e) for e in world["exps"]]
    args = ["--out", out, "--size", str(NPIX), "--halo-px", str(HALO), "--scale", str(SCALE), "--steps", str(steps), "--budget-min", "90", "--batch", "1024",
            "--hidden", "48", "--n-bank", "12", "--shape-hidden", "32", "--spec-k", "6", "--spec-q", "3", "--core-radius", "6.0", "--core-pitch", "0.5",
            "--wing-ratio", "1.25", "--wing-nang", "24", "--wing-arcsec", "1.2", "--c2f-frac", "0.3", "--c2f-start-arcsec", "0.3", "--nuis-steps", "120",
            "--val-nuis-steps", "60", "--val-every-min", "1000", "--ref-instrument", "sharp", "--boot-b", "50", "--log-every", "100", "--lr-op", "3e-3",
            "--lr-kernel", "4e-3", "--lam-neg", "1.0", "--seed", "0", "--n-val", "0", "--dump-heldout-pixels", "--loss", "huber", "--noise-start", "2.0"] + list(extra)
    a = O.parse_args(args)
    a.size_data = NPIX
    thead = V.target_header(110.0, -70.0, G, SCALE)
    cfn = classical_splat(thead)
    cfn.bands = world["reg"].bands
    t0 = time.time()
    ctx = O.run(exps, thead, DEV, a, out, classical_fn=cfn, reg=world["reg"], bandpass=world["bandpass"], preselected=True)
    return ctx, time.time() - t0


def annulus_scores(world, out):
    """Held-out chi2 per pixel per annulus (model from the dumps, oracle from the generator)."""
    bright = world["pos"][: world["nbright"]]
    acc = {k: [0.0, 0.0, 0] for k in range(len(ANN) - 1)}
    for i, e in enumerate(world["exps"]):
        if not e["holdout"]:
            continue
        z = np.load(os.path.join(out, "heldout_pixels", "opfield", f"{e['name']}.npz"))
        inf = world["info"][i]
        R = np.array([[math.cos(math.radians(inf["roll"])), -math.sin(math.radians(inf["roll"]))], [math.sin(math.radians(inf["roll"])), math.cos(math.radians(inf["roll"]))]])
        xy = np.stack([z["tx_hdr"], z["ty_hdr"]], 1)
        dmin = np.min([np.hypot(*(xy - b).T) for b in bright], axis=0)               # J = rotation: detector px = sky px
        chi_m = ((z["data"] - z["model"]) / z["sigma"]) ** 2
        chi_o = ((z["data"] - inf["expect"] - 30.0) / z["sigma"]) ** 2
        for k in range(len(ANN) - 1):
            s = (dmin >= ANN[k]) & (dmin < ANN[k + 1])
            acc[k][0] += float(chi_m[s].sum()); acc[k][1] += float(chi_o[s].sum()); acc[k][2] += int(s.sum())
    return {f"{ANN[k]:g}-{ANN[k + 1]:g}": {"chi2": v[0] / max(v[2], 1), "oracle": v[1] / max(v[2], 1), "n": v[2]} for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--variants", default="default,ts_is")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "synth_wings"))
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--assert-variant", default="ts_is", help="the combination that must beat 'default' in the 3-15 px annulus")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    world = build_world(0)
    print(f"world: {len(world['exps'])} exposures, {sum(e['d'].numel() for e in world['exps'])} pixels; bright fluxes {np.round(world['flux'][:world['nbright']]).tolist()}", flush=True)
    res = {}
    rp = os.path.join(a.out, "synth_wings_results.json")
    if os.path.exists(rp):
        res = json.load(open(rp))
    for v in a.variants.split(","):
        out = os.path.join(a.out, v)
        ctx, sec = run_variant(world, out, a.steps, VARIANTS[v])
        sc = annulus_scores(world, out)
        res[v] = {"annuli": sc, "seconds": sec, "steps": ctx["log"].get("train_steps"), "heldout_chi2": ctx["log"]["metrics"]["heldout_chi2"]["opfield"]}
        json.dump(res, open(rp, "w"), indent=1, default=float)
        print(f"=== {v}: {sec:.0f} s; held-out chi2 all {res[v]['heldout_chi2']:.3f}")
        for k, x in sc.items():
            print(f"   annulus {k:>6} px: chi2 {x['chi2']:.3f} (oracle {x['oracle']:.3f}), n {x['n']}")
    if a.do_assert and "default" in res and a.assert_variant in res:
        def wing(r):
            a_ = [r["annuli"][k] for k in ("3-8", "8-15")]
            return sum(x["chi2"] * x["n"] for x in a_) / sum(x["n"] for x in a_)
        w0, w1 = wing(res["default"]), wing(res[a.assert_variant])
        print(f"3-15 px annulus chi2: default {w0:.3f} vs {a.assert_variant} {w1:.3f}")
        assert w1 < w0, (w0, w1)


if __name__ == "__main__":
    main()
