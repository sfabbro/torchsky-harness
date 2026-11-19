"""CPU reproducer of the real-data stall of the uncertainty solver (SOLVER_DIAGNOSIS_2026-11-19.md).

A small synthetic field built from the package's own pieces (synth.render_truth, data.Exposure, model.FieldOperator, train.warm_kernel) with
the features of the SMACS NIRCam fit switched on one at a time or together:

  filters   'one'  : one short-wavelength filter, K = 3 (the truth-problem regime)
            'six'  : F090W, F150W, F200W (short-wavelength channel, 0.031" pixels) and F277W, F356W, F444W (long-wavelength channel, 0.063"
                     pixels), K = 12 spectral coefficients on the reference domain 0.70-5.30 um (the real Phi_band rows)
  gap       a cross-shaped chip gap (invalid rows and columns, width gap_px) through the middle of every short-wavelength detector (the 2 x 2
            NIRCam short-wavelength mosaic seen from a box at the module centre)
  gap_centre_filter
            name of a filter whose exposures are dithered by less than the gap, so the gap stays on the box centre (the probe node of the
            Fourier preconditioner) for that filter only: what the real half 0 shows for F090W (kappa 0 at the centre)
  bright    peak signal-to-noise of the brightest point sources (0: faint field only); with floor > 0 the weights fall as 1 / (f S)^2
  floor     fractional noise floor per channel (real: 0.04 to 0.09)
  huber     Huber weights at the fit's residual (the fit's loss)
  sky_form  'grid' or 'items' (the item sky rasterised onto the grid)

The fitted sky is the package's warm start (inverse-variance splat mapped to coefficients, PSF-blurred), not an optimiser's result: the
right-hand side of the Gauss-Newton step is therefore larger and smoother than on real data. Documented as an approximation.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fieldop.bandpass import FilterSet, box_filter  # noqa: E402
from fieldop.data import Exposure, build_meta, make_patches  # noqa: E402
from fieldop.synth import SynthInstrument, Truth, render_truth  # noqa: E402

SIX = [("SW", "F090W", 0.80, 1.00), ("SW", "F150W", 1.33, 1.67), ("SW", "F200W", 1.75, 2.23),
       ("LW", "F277W", 2.42, 3.13), ("LW", "F356W", 3.14, 3.98), ("LW", "F444W", 3.88, 4.98)]

DEFAULT = dict(filters="six", gap=True, gap_px=10, gap_centre_filter="", bright=3000.0, floor=(0.09, 0.04), huber=True, sky_form="grid",
               n_sw=3, n_lw=2, sw_npix=60, lw_npix=30, Q=3, sky_half=1.1, pitch=0.04, seed=0, dither=0.45, two_channels=True,
               kernel_steps=120, item_count=1500)


def _truth(rng, bright, half=1.0):
    """Faint sources everywhere plus (if bright > 0) a few bright point sources (peak S/N up to `bright` in one short-wavelength pixel)."""
    n = 24
    pts = np.stack([rng.uniform(-half, half, n), rng.uniform(-half, half, n), np.exp(rng.uniform(math.log(0.02), math.log(1.0), n)),
                    rng.uniform(-1.5, 1.0, n)], 1)
    if bright > 0:
        # a 0.03" (sigma) point source puts about 0.031^2 / (2 pi 0.02^2) = 0.38 of its flux in the peak pixel; flux in units of noise x pixel area
        amp = bright * 0.031 ** 2 / 0.38
        bp = np.array([[0.35, 0.25, amp, -0.5], [-0.4, 0.3, amp / 3, 0.5], [0.1, -0.45, amp / 10, 0.0], [-0.3, -0.3, amp / 30, -1.0]])
        pts = np.concatenate([bp, pts])
    pts[:, 2] *= 1.0
    gals = np.stack([rng.uniform(-0.6, 0.6, 3), rng.uniform(-0.6, 0.6, 3), rng.uniform(0.08, 0.2, 3), rng.uniform(0.5, 3.0, 3) * 0.031 ** 2 * 20,
                     rng.uniform(-0.5, 1.5, 3), np.zeros(3)], 1)
    return Truth(pts, gals)


def build_field(**kw):
    o = dict(DEFAULT, **kw)
    rng = np.random.default_rng(o["seed"])
    if o["filters"] == "one":
        flist = [("SW", "F150W", 1.33, 1.67)]
    else:
        flist = SIX if o["two_channels"] else [("SW",) + f[1:] for f in SIX]
    insts = {"SW": SynthInstrument("SW", 0.031, o["sw_npix"], 6.5, 0.020, 1.5, chromatic=1.0, f_wing=0.1, wing_ratio=3.0),
             "LW": SynthInstrument("LW", 0.063, o["lw_npix"], 6.5, 0.047, 3.5, chromatic=1.0, f_wing=0.1, wing_ratio=3.0)}
    names_inst = [n for n in ("SW", "LW") if any(f[0] == n for f in flist)]
    filters = [box_filter(name, "NIRCAM", lo, hi, o["Q"]) for ins, name, lo, hi in flist]
    chan_of = {name: ins for ins, name, lo, hi in flist}
    fset = FilterSet(filters)
    truth = _truth(rng, o["bright"])
    exps = []
    k = 0
    for ins_name, name, lo, hi in flist:
        ins = insts[ins_name]
        ii = names_inst.index(ins_name)
        f = fset.index[name]
        n_exp = o["n_sw"] if ins_name == "SW" else o["n_lw"]
        for j in range(n_exp):
            Rm = np.eye(2)
            amp = 0.06 if name == o["gap_centre_filter"] else o["dither"]
            c = rng.uniform(-amp, amp, 2)
            n = ins.npix
            rr, cc = np.mgrid[0:n, 0:n].astype(np.float64)
            off = np.stack([cc - (n - 1) / 2, rr - (n - 1) / 2], -1) * ins.pixel
            pos = off @ Rm.T + c
            Xh, Yh = torch.tensor(pos[..., 0], dtype=torch.float32), torch.tensor(pos[..., 1], dtype=torch.float32)
            xi = torch.stack([torch.tensor(cc.ravel() / (n - 1) * 2 - 1), torch.tensor(rr.ravel() / (n - 1) * 2 - 1)], -1).float()
            clean = render_truth(truth, ins, fset.filters[f], Xh.reshape(-1), Yh.reshape(-1), xi, n_dense=12).view(n, n)
            noise = 1.0
            fl = o["floor"][0 if ins_name == "SW" else 1] if o["floor"] else 0.0
            sig_true = torch.sqrt(noise ** 2 + (fl * clean.clamp(min=0)) ** 2)
            d = clean + sig_true * torch.randn(n, n, generator=torch.Generator().manual_seed(1000 + k))
            valid = torch.ones(n, n, dtype=torch.bool)
            if o["gap"] and ins_name == "SW":
                g0, g1 = n // 2 - o["gap_px"] // 2, n // 2 + (o["gap_px"] + 1) // 2
                valid[g0:g1, :] = False
                valid[:, g0:g1] = False
            e = Exposure(name=f"{name}_{j}", filt=f, inst=0, chan=ii, det=ii, t=0.0, mjd=0.0, det_shape=(n, n), cut0=(0, 0),
                         d=d, sig=torch.full((n, n), noise), valid=valid, X=Xh, Y=Yh, R=Rm, split="train", group=f"{name}_{j}", stripes=False,
                         bkg0=0.0, Jpix=torch.tensor(np.broadcast_to(Rm * ins.pixel, (n, n, 2, 2)).copy(), dtype=torch.float32),
                         extra={"J": (Rm * ins.pixel).tolist(), "clean": clean, "bkg_true": 0.0, "diameter_m": ins.diameter_m, "centre": c.tolist()})
            e.eid = len(exps)
            exps.append(e)
            k += 1
    for e in exps:
        make_patches(e, 32)
    meta = build_meta(exps, names_inst, names_inst, ["NIRCAM"])
    return exps, fset, meta, truth, insts, names_inst, o


def build_model(**kw):
    """Field, configuration, model with a warm-started kernel and the requested noise floor; the sky is the package's warm start."""
    from fieldop.cli import default_config, make_sky
    from fieldop.instruments import spectral_setup
    from fieldop.kernel import inv_softplus
    from fieldop.model import FieldOperator
    from fieldop.train import warm_kernel
    exps, fset, meta, truth, insts, names_inst, o = build_field(**kw)
    K = 3 if o["filters"] == "one" else 12
    cfg = default_config(legacy_defaults=False, sky_pitch=o["pitch"], render_pitch=0.01, level_nh=8, kernel_extent=0.6, region_px=32,
                         n_lambda=o["Q"], k_hidden=16, spec_k=K, pattern_grid=0, stripes=0, steps=0, budget_min=1e6, tau_mode="fixed", seed=o["seed"],
                         held_steps=0, warm_kernel_steps=o["kernel_steps"], sky_form=o["sky_form"], item_count=o["item_count"],
                         loss="huber" if o["huber"] else "gauss", gauge_ref_inst=-1, kernel_norm_radius=0.0)
    spectral_setup(fset, cfg, log=lambda s: None)
    torch.manual_seed(o["seed"])
    sky, N = make_sky(exps, fset, cfg, o["sky_half"], "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    chan_of = {name: ins for ins, name, lo, hi in (SIX if o["filters"] == "six" else [("SW", "F150W", 0, 0)])}
    finfo = {f: {"pixel_arcsec": insts[chan_of[fset.names[f]]].pixel, "diameter_m": 6.5, "inst": 0} for f in range(len(fset))}
    warm_kernel(model, cfg, finfo, log=lambda s: None)
    with torch.no_grad():
        fl = o["floor"] or (0.0, 0.0)
        for c in range(model.floor_raw.shape[0]):
            model.floor_raw[c] = float(inv_softplus(torch.tensor(max(fl[min(c, 1)], 1e-6))))
    return model, exps, cfg, truth, o
