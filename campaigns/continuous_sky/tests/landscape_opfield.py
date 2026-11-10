"""Registration basin of a coarse (0.12 arcsec pixel) exposure: clipped chi2 (per-pixel chi2 capped at 25) against a GLOBAL shift s of the exposure
position along x or y, s in [-1, +1] arcsec.  (a) truth sky + truth kernel through the independent data renderer, centred on the true position;
(b) the learned model (sky, kernels, background fitted) centred on its learned registration, with no blur and with the 0.4 arcsec c2f blur.
Trains the L6 case itself (extra args as for ladder_opfield.py).  Usage: python tests/landscape_opfield.py --tag name [--extra "..."] [--steps 1800]
Saves runs/opfield_synth/landscape_<tag>.png / .json."""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_opfield as L  # noqa: E402

S, O, V = L.S, L.O, L.S.V
CAP = 25.0


def truth_landscape(world, i, shifts_px, axis):
    e, inf = world["exps"][i], world["info"][i]
    inst = S.INSTR[inf["inst"]]
    band = S.band_image_fine(world["T"], inst["lam"])
    kern = [fftconvolve(band, S.sky_kernel(inst, inf["roll"], ws), mode="same") for ws in (1.0, 1.12)]
    xt0 = np.stack([e["tx"].numpy(), e["ty"].numpy()], 1) + inf["true_shift_pix"]
    w = ((e["dx"].numpy() + 1) / 2)
    leg = V._legendre2d(e["dx"], e["dy"], 2).numpy()
    bkg = leg @ inf["bcoef"]
    out = []
    for s in shifts_px:
        xt = xt0.copy()
        xt[:, axis] += s
        idx = np.stack([(xt[:, 1] - world["x_min"]) * S.FINE - 0.5, (xt[:, 0] - world["x_min"]) * S.FINE - 0.5])
        I0 = map_coordinates(kern[0], idx, order=3, mode="nearest")
        I1 = map_coordinates(kern[1], idx, order=3, mode="nearest")
        flux = inf["gain"] * ((1 - w) * I0 + w * I1)
        sat = S.SAT * np.tanh(flux / S.SAT)
        r = (e["d"].numpy() - (sat + bkg)) ** 2 * e["ivar"].numpy()
        out.append(float(np.minimum(r, CAP).mean()))
    return out


def learned_landscape(ctx, i, shifts_px, axis, blur_px):
    model, e = ctx["model"], ctx["exps"][i]
    nid = i
    bt = O.exp_batch(e, i, None, nid=nid)
    res = []
    with torch.no_grad():
        lv = model.make_levels(blur_px)
        for s in shifts_px:
            b2 = dict(bt)
            x0 = bt["x0"].clone()
            x0[:, axis] += s
            b2["x0"] = x0
            out = model(b2, model.nuis, lv)
            r = (e["d"] - out["pred"]) ** 2 * e["ivar"]
            res.append(float(torch.clamp(r, max=CAP).mean()))
    return res


def width(xs, ys):
    """full width (arcsec) of the central basin at half the rise from its minimum to the local maximum within +-1 arcsec."""
    ys = np.asarray(ys); k = int(np.argmin(ys)); lo, hi = ys[k], ys.max()
    half = lo + 0.5 * (hi - lo)
    l = k
    while l > 0 and ys[l] < half:
        l -= 1
    r = k
    while r < len(ys) - 1 and ys[r] < half:
        r += 1
    return float(xs[r] - xs[l]), float(xs[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="field")
    ap.add_argument("--extra", default="")
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--c2f-frac", type=float, default=0.3)
    ap.add_argument("--c2f-start", type=float, default=None)
    ap.add_argument("--operator", default="field")
    ap.add_argument("--out", default=os.path.join(S.ROOT, "runs", "opfield_synth"))
    a = ap.parse_args()
    world = L.build(6)
    ex = ["--operator", a.operator, "--nuis-steps", "20"] + a.extra.split() + ["--skip-eval"]
    if a.c2f_start:
        ex += ["--c2f-start-arcsec", str(a.c2f_start)]
    ctx, args = S.run_variant(world, os.path.join(a.out, f"landscape_{a.tag}"), a.steps, a.c2f_frac, "ls", False, 30.0, ex)
    info = world["info"]
    i = next(k for k in ctx["train_idx"] if info[k]["inst"] == "coarse" and info[k]["roll"] == 0.0)
    sh_as = np.linspace(-1.0, 1.0, 41)
    sh_px = sh_as / S.SCALE
    res = dict(true_delta_mas=(np.asarray(info[i]["delta"]) * S.SCALE * 1000).tolist())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax_i, axis in enumerate((0, 1)):
        t = truth_landscape(world, i, sh_px, axis)
        l0 = learned_landscape(ctx, i, sh_px, axis, 0.0)
        l1 = learned_landscape(ctx, i, sh_px, axis, 0.4 / S.SCALE)
        for nm, y in (("truth", t), ("learned, no blur", l0), ("learned, 0.4\" blur", l1)):
            res[f"{'xy'[axis]}/{nm}"] = dict(width_arcsec=width(sh_as, y)[0], argmin_arcsec=width(sh_as, y)[1], values=y)
            axs[ax_i].plot(sh_as, y, label=nm)
        axs[ax_i].set_xlabel(f"global shift along {'xy'[axis]} [arcsec]")
        axs[ax_i].set_title(f"coarse exposure {i} (true offset {res['true_delta_mas'][0]:.0f},{res['true_delta_mas'][1]:.0f} mas)", fontsize=8)
    axs[0].set_ylabel("clipped chi2 per pixel")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    png = os.path.join(a.out, f"landscape_{a.tag}.png")
    fig.savefig(png, dpi=130)
    json.dump(res, open(os.path.join(a.out, f"landscape_{a.tag}.json"), "w"), indent=1, default=float)
    for k, v in res.items():
        if isinstance(v, dict):
            print("LANDSCAPE", a.tag, k, "basin width %.2f arcsec, argmin %.2f arcsec" % (v["width_arcsec"], v["argmin_arcsec"]), flush=True)


if __name__ == "__main__":
    main()
