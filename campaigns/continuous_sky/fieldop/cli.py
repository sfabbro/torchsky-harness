"""Command line of the field operator (python -m fieldop ...). Every ablation axis of the plan is a flag:

  kernel variability     --k-space / --k-time / --k-lambda / --k-exp  {on, off, learned}
  kernel extent          --kernel-extent ARCSEC
  rendering resolution   --render-pitch ARCSEC (+ --level-factor, --level-nh, --region-px, --n-lambda, --render {fft, quad})
  sky pitch              --sky-pitch ARCSEC (0.04, 0.02)
  amplitude              --amplitude {unit, structured, free}     (unit: integral K = 1 and A = alpha T; structured: A = alpha g; free: kernel integral free)
  filter input           --filter-input {token, curve}
  input level            --input-level {cal, rate}
  loss                   --loss {gauss, huber}
  variability scales     --tau-mode {fixed, learned}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def build_parser():
    ap = argparse.ArgumentParser(prog="fieldop", description=__doc__.split("\n")[0])
    g = ap.add_argument_group("data / protocol")
    g.add_argument("--field", default="SMACS")
    g.add_argument("--split", default="dither")
    g.add_argument("--protocol-root", default="/arc/projects/ots/adass2026/protocol/v1")
    g.add_argument("--splits", default="", help="default: <protocol-root>/splits_<FIELD>.json")
    g.add_argument("--box-id", type=int, default=0)
    g.add_argument("--tiles", default="", help="several scoring boxes (sky patches) under one operator: 'all' or a comma list of box ids ('' = --box-id only)")
    g.add_argument("--tile-weight-power", type=float, default=0.5, help="tile k is visited with probability proportional to (training pixels)^power")
    g.add_argument("--visit-steps", type=int, default=10, help="optimisation steps per tile visit")
    g.add_argument("--gpu-resident-gb", type=float, default=4.0, help="pixel data of tiles kept on the device (LRU); tiles beyond it are swapped")
    g.add_argument("--load-workers", type=int, default=8, help="threads reading files (several tiles)")
    g.add_argument("--sky-swap", type=int, default=0, help="several tiles: inactive skies and their Adam state are kept on the host")
    g.add_argument("--held-rounds", type=int, default=4, help="several tiles: rounds over the tiles in the held-out nuisance fit")
    g.add_argument("--held-ref-cuts", type=float, default=24.0, help="several tiles: per-tile held-out steps = --held-steps x (held-out cuts of the tile) / this")
    g.add_argument("--all-box-files", type=int, default=0, help="also load split files not listed as overlapping the box")
    g.add_argument("--method-name", default="fieldop")
    g.add_argument("--out", default="out_fieldop")
    g.add_argument("--data-root", default="", help="directory of the _cal files (default: paths in the splits file)")
    g.add_argument("--rate-dir", default="")
    g.add_argument("--svo-cache", default="")
    g.add_argument("--shifts-from", default="", help="register_xcorr.json (LMC): initial per-exposure registration shifts")
    g.add_argument("--filters", default="")
    g.add_argument("--max-exposures", type=int, default=0)
    g.add_argument("--n-val-groups", type=int, default=1, help="dither groups per filter carved from training for the tau validation likelihood")
    g.add_argument("--epoch-min-days", type=float, default=1.0, help="epoch axes need a training time span at least this long")
    g.add_argument("--epoch-axes", type=int, default=1)
    g.add_argument("--epoch-per-instrument", type=int, choices=[0, 1], default=1,
                   help="1: epoch coordinate normalised, and epoch axes decided, per instrument from its own time span (audit finding 12); 0: one span for the whole field (old)")
    g.add_argument("--field-init", choices=["random", "zero"], default="random",
                   help="start of the throughput and distortion fields: 'random' (old) or 'zero' = last layers at zero output (audit finding 15; the rms "
                   "normalisation is singular there: gradients about 1e4 times larger on step 0; not tested on real data)")
    g.add_argument("--legacy-defaults", action="store_true",
                   help="set every default changed on 18 Nov 2026 back to its old value: --sky-zero median, --noise-floor-max 0, --noise-scale-max 0, "
                   "--nuisance-steps mean, --held-xfit 0, --held-step-rows 0, --epoch-per-instrument 0. Options given explicitly win. "
                   "(The several-instrument rules of audit findings 5 and 9 are not touched: give --gauge-ref-inst -1 --kernel-norm-radius 0 to get the old values.)")
    g.add_argument("--write-pred", type=int, default=1)
    g.add_argument("--pix-default", type=float, default=0.031)
    g = ap.add_argument_group("sky")
    g.add_argument("--sky-pitch", type=float, default=0.04)
    g.add_argument("--sky-margin", type=float, default=2.0, help="sky domain = box + 2 x margin [arcsec]")
    g.add_argument("--data-margin", type=float, default=1.0, help="data region = box + 2 x margin [arcsec] (< sky margin)")
    g.add_argument("--spec-k", type=int, default=12)
    g.add_argument("--sky-neg", type=float, default=1.0)
    g.add_argument("--sky-precond", type=float, default=0.02, help="diagonal preconditioner of the sky: 1 + f x (local warm-start S/N), 0 = off")
    g.add_argument("--sky-precond-cap", type=float, default=2000.0)
    g.add_argument("--sky-form", choices=["grid", "items"], default="grid", help="sky parametrisation: coefficient grid, or Gaussian items "
                   "rasterised onto that grid (sky.ItemSky)")
    g.add_argument("--item-count", type=int, default=20000, help="items per sky (tile); split over --item-levels lattices in proportion 4^-l")
    g.add_argument("--item-levels", type=int, default=3, help="coarse-to-fine seeding lattices")
    g.add_argument("--item-floor", type=float, default=0.0, help="minimum item width [arcsec] (0 = 0.4 x --sky-pitch)")
    g.add_argument("--item-max-width", type=float, default=0.0, help="maximum item width [arcsec] (0 = 16 x --sky-pitch); with "
                   "--item-raster direct, patches are capped at +-128 nodes (wider items are truncated); with pyramid, up to the field size")
    g.add_argument("--item-chunk", type=int, default=0, help="elements (items x patch nodes x K) per rasterisation chunk; 0 = 4 M. On a "
                   "GPU the item scatter is bound by the number of small kernel launches, so larger chunks (32 M) are faster")
    g.add_argument("--item-buckets", choices=["fine", "coarse"], default="fine", help="patch-size ladder: fine (18 to 25 sizes) or coarse "
                   "(powers of two: fewer distinct patch shapes, so fewer launches on a GPU, at up to 4x more patch nodes; a larger patch "
                   "only lowers the truncation error)")
    g.add_argument("--item-raster", choices=["direct", "pyramid"], default="direct", help="item rasteriser: direct (patch of +-4 widths at "
                   "the sky pitch, cost per item ~ (width / pitch)^2) or pyramid (Gaussian-semigroup pyramid, cost per item independent of "
                   "its size; F40)")
    g.add_argument("--item-impl", choices=["prototype", "torchsky"], default="prototype", help="item rasteriser code: the prototype's own (ItemSky, "
                   "default) or torchsky's GaussianItemScene (same parameters, floor and positivity; --item-raster maps to its mode; "
                   "--item-patch sheared is not available with it, --item-buckets is ignored)")
    g.add_argument("--item-patch", choices=["square", "sheared"], default="square", help="pyramid only: patch of each item, square "
                   "(+-4 larger widths) or sheared (+-4 widths in the item's decorrelated coordinates, used where it has 1.5x fewer nodes; "
                   "cost ~ item area instead of larger width squared, for elongated items; also applied on the fine level; F40)")
    g.add_argument("--item-lr-pos", type=float, default=0.05, help="Adam step of item positions [sky pitches]")
    g.add_argument("--item-lr-width", type=float, default=0.03, help="Adam step of item log widths")
    g.add_argument("--item-lr-angle", type=float, default=0.05, help="Adam step of item angles [rad]")
    g.add_argument("--item-lr-coef", type=float, default=0.05, help="Adam step of item log spectral coefficients")
    g = ap.add_argument_group("kernel")
    g.add_argument("--kernel-extent", type=float, default=10.0)
    g.add_argument("--kernel-norm-radius", type=float, default=None,
                   help="arcsec; > 0: the kernel's unit integral is taken inside this radius only (flux convention F41); 0 = whole support. Unset: 0 with "
                   "one instrument, 2 arcsec with several (audit finding 9)")
    g.add_argument("--render-pitch", type=float, default=0.0, help="0 = per channel: largest sky_pitch/k <= pixel / --render-os-pix")
    g.add_argument("--render-os-pix", type=float, default=6.0)
    g.add_argument("--interp", choices=["bspline", "keys"], default="bspline")
    g.add_argument("--level-factor", type=int, default=2)
    g.add_argument("--level-nh", type=int, default=24)
    g.add_argument("--level-overlap", type=float, default=0.5, help="level l hands over to l+1 between a R_l and R_l")
    g.add_argument("--region-px", type=int, default=256)
    g.add_argument("--n-lambda", type=int, default=5)
    g.add_argument("--render", choices=["fft", "quad"], default="fft")
    g.add_argument("--quad-nrad", type=int, default=12)
    g.add_argument("--quad-naz", type=int, default=48)
    g.add_argument("--k-hidden", type=int, default=64)
    g.add_argument("--k-layers", type=int, default=3)
    g.add_argument("--dh", type=int, default=16)
    g.add_argument("--enc-r0", type=float, default=0.0, help="core radius of the log-radius encoding (0 = half the render pitch)")
    g.add_argument("--render-jitter", type=int, default=1, help="random sub-pitch phase of the level-0 kernel grid in training (F11)")
    g.add_argument("--enc-nrad", type=int, default=8)
    g.add_argument("--enc-mmax", type=int, default=12)
    g.add_argument("--filter-input", choices=["token", "curve"], default="curve")
    for ax in ("space", "time", "lambda", "exp"):
        g.add_argument(f"--k-{ax}", choices=["on", "off", "learned"], default="learned")
    g.add_argument("--warm-kernel-steps", type=int, default=400)
    g = ap.add_argument_group("variability scales")
    g.add_argument("--tau-mode", choices=["fixed", "learned"], default="learned")
    g.add_argument("--tau0-code", type=float, default=0.3)
    g.add_argument("--tau0-g", type=float, default=0.02)
    g.add_argument("--tau0-D", type=float, default=0.005)
    g.add_argument("--tau-start", type=float, default=0.35)
    g.add_argument("--val-every", type=int, default=4)
    g.add_argument("--val-nuis-steps", type=int, default=2)
    g.add_argument("--lr-tau", type=float, default=0.05, help="step of log tau per evidence update")
    g.add_argument("--tau-accum", type=int, default=4, help="validation evaluations accumulated per tau step")
    g.add_argument("--tau-beta", type=float, default=0.95, help="EMA factor of the validation-gradient evidence")
    g.add_argument("--tau-occam-z", type=float, default=2.0, help="evidence (in sd) an axis must show to grow")
    g.add_argument("--tau-warm-evals", type=int, default=3)
    g.add_argument("--tau-code-max", type=float, default=1.5, help="cap of the code-space kernel variability scales")
    g.add_argument("--tau-g-max", type=float, default=0.3, help="cap of the throughput function-axis scales (rms log g on the footprint lattice)")
    g.add_argument("--tau-d-max", type=float, default=0.03, help="cap of the static / drift distortion scales [arcsec]")
    g.add_argument("--ref-footprint", type=int, default=1, help="detector fields normalised (and the throughput gauge anchored) on a 3x3 lattice "
                   "spanning each detector's training footprint (0 = fixed lattice at +-0.8)")
    g.add_argument("--ab-every", type=int, default=2, help="likelihood-ratio (axis off vs on) evidence every N validation evaluations (0 = off)")
    g.add_argument("--ab-beta", type=float, default=0.95, help="forgetting factor of the likelihood-ratio evidence")
    g.add_argument("--em-damp", type=float, default=0.5, help="exponent of the damped EM step of the random-effect scales")
    g = ap.add_argument_group("amplitude / geometry / detector")
    g.add_argument("--amplitude", choices=["unit", "structured", "free"], default="structured")
    g.add_argument("--gauge-ref-inst", type=int, default=None,
                   help="instrument index whose training exposures define the throughput gauge (the absolute flux scale); -1 = all training exposures. "
                   "Unset: -1 with one instrument; with several instruments the run refuses to start unless it is given (audit finding 5)")
    g.add_argument("--blur-floor", type=float, default=1.0, help="blur gauge: core m2 of the sharpest kernel class >= this x (sky pitch)^2 / 3 (0 = off)")
    g.add_argument("--pedestal-gauge", type=int, default=1, help="per filter, training background constants (relative to the exposure medians) sum to zero")
    g.add_argument("--sky-zero", choices=["median", "masked"], default="masked",
                   help="reference level of each training exposure for the pedestal gauge (the split between a constant sky and the backgrounds, "
                   "which the data cannot fix): 'median' = median of the exposure's data (sources included; the sky in empty regions then sits "
                   "below zero by the median source light, and positivity turns that into a positive total offset); 'masked' = median of the "
                   "DATA (not the residual) over the pixels where the current model's sky signal S satisfies |S| < --sky-zero-snr x the pixel sigma, "
                   "re-evaluated at --sky-zero-at fractions of the fit: the sky is zero, by convention, where the model sees no source")
    g.add_argument("--sky-zero-snr", type=float, default=0.3, help="--sky-zero masked: a pixel counts as source-free when |S| < this x its sigma")
    g.add_argument("--sky-zero-smooth", type=float, default=1.5, help="--sky-zero masked: Gaussian width [detector pixels] of the smoothing of S before the threshold")
    g.add_argument("--sky-zero-at", default="0,0.1,0.25,0.5,0.75", help="--sky-zero masked: fit progress fractions at which the levels are re-evaluated")
    g.add_argument("--sky-positivity-margin", type=float, default=0.0,
                   help="positivity (grid penalty, item exponentials) applies to sky + p instead of the sky, p = this x the noise of the warm-start "
                   "coadd at one sky node (per band, median over covered nodes): the sky may go down to -p, so that a noise estimate of an empty "
                   "region is not rectified (0 = today)")
    g.add_argument("--distortion", type=int, default=1)
    g.add_argument("--affine-prior", type=float, default=0.01)
    g.add_argument("--input-level", choices=["cal", "rate"], default="cal")
    g.add_argument("--rate-wcs", choices=["cal", "rate", "telescope"], default="cal",
                   help="--input-level rate: header WCS (pixel positions) from the matching _cal file (distortion-corrected SIP, default), "
                   "from the _rate file itself (linear TAN from the pointing, no distortion: up to 0.24 arcsec off at the detector corners), or "
                   "'telescope': the _rate file's own pointing keywords plus the SIAF polynomial of its aperture through torchsky's "
                   "TelescopeFrameDetectorToSky (needs pysiaf; 3 to 60 mas rms from the _cal solution on the 12-exposure sample: see TORCHSKY_INTEGRATION_REPORT.md)")
    g.add_argument("--rate-units", choices=["dn", "sky"], default="dn",
                   help="--input-level rate: units of the sky. 'dn' = count rate per arcsec^2 (today; differs per filter and detector). 'sky' = "
                   "surface brightness (MJy/sr, as the calibrated fit): count rate per pixel = sky x (pixel area / nominal pixel area) / "
                   "conversion, the conversion (MJy/sr per DN/s) a per-exposure constant read at load time: PHOTMJSR of the matching _cal "
                   "file with --rate-wcs cal, else from --rate-photom; the learned throughput terms stay free to correct it")
    g.add_argument("--rate-photom", default="", help="--rate-units sky: JSON table {\"DETECTOR/FILTER\": MJy/sr per DN/s} (and optionally "
                   "{\"DETECTOR/FILTER/area\": nominal pixel area in arcsec^2}); used when the _cal header is not read (--rate-wcs rate or telescope)")
    g.add_argument("--nuisance-steps", choices=["mean", "exposure"], default="exposure",
                   help="Adam step of the data-unit nuisances (background, stripes): lr x mean pipeline sigma over all exposures ('mean', the old "
                   "default) or x each exposure's own median sigma ('exposure', the default since 18 Nov 2026; makes the fit independent of a per-exposure unit constant, e.g. --rate-units sky)")
    g.add_argument("--loader", choices=["astropy", "torchfits"], default="astropy",
                   help="exposure files read by the prototype's astropy code (default) or by torchsky's field_observation_from_exposure_file "
                   "(torchfits); if that cannot be imported the run falls back to astropy and says so")
    g.add_argument("--pixel-gain", choices=["off", "detector", "detector-filter"], default="off",
                   help="per-pixel log-gain map (flat field) learned from the exposures: one per detector, or per detector and filter (F43)")
    g.add_argument("--pixel-gain-poly", type=int, default=2, help="gauge of the gain map: monomials of total degree <= this are projected out "
                   "over the observed pixels of each map (0 = mean only)")
    g.add_argument("--lr-gain", type=float, default=3e-3, help="Adam step of the log-gain map (F45; chosen on the synthetic field among "
                   "3e-4, 1e-3, 3e-3, 1e-2)")
    g.add_argument("--pixel-gain-start", type=float, default=0.1, help="the gain map is stepped from this fraction of the fit on")
    g.add_argument("--pixel-gain-prior", choices=["off", "gauss", "mix2"], default="off",
                   help="zero-mean prior on log G per pixel whose width(s) are learned by EM from the map and its per-pixel information "
                   "(gauss: one Gaussian; mix2: narrow core + wide tail); off = no prior (today's behaviour)")
    g.add_argument("--pixel-gain-gauge", choices=["poly", "poly+pattern"], default="poly",
                   help="gauge of the gain map: 'poly' = monomials of degree <= --pixel-gain-poly projected out (today); 'poly+pattern' = also "
                   "the bilinear functions of the additive pattern's grid (--pattern-grid), so that gain and pattern share no component")
    g.add_argument("--pixel-gain-em-start", type=float, default=0.3, help="--pixel-gain-prior: EM updates of the prior from this fraction of the fit on")
    g.add_argument("--pixel-gain-em-every", type=int, default=25, help="--pixel-gain-prior: steps between EM updates")
    g.add_argument("--filter-shift", choices=["off", "learned"], default="off",
                   help="learned position shift (2 numbers, detector pixels, applied to the pixel coordinates before the pixel-to-sky map) shared by all "
                   "exposures of one (detector, filter) pair; held-out exposures use their pair's fitted shift, fixed. Not available with several "
                   "disconnected shift components (tiles)")
    g.add_argument("--filter-shift-init", default="", help="JSON file {\"DETECTOR/FILTER\": [dx, dy]} (detector pixels) giving the starting shifts")
    g.add_argument("--lr-filter-shift", type=float, default=0.03, help="Adam step of the pair shift, detector pixels (same warm-up and cosine factor as the others)")
    g.add_argument("--dq-ignore-bits", type=int, default=0,
                   help="data-quality bits that do not exclude a pixel (F42); 0 = strict DQ == 0; 2 = keep pixels flagged SATURATED only")
    g.add_argument("--nonlin", type=int, default=0)
    g.add_argument("--pattern-grid", type=int, default=8)
    g.add_argument("--stripes", type=int, default=1)
    g.add_argument("--stripe-l2", type=float, default=1e-3)
    g.add_argument("--floor0", type=float, default=0.01)
    g.add_argument("--noise-floor-max", type=float, default=0.2,
                   help="upper bound of the learned fractional noise floor (softplus(floor_raw)), enforced by clamping floor_raw after every operator "
                        "step; 0 = no bound (the old default)")
    g.add_argument("--noise-scale-max", type=float, default=2.0,
                   help="the learned noise scale exp(h) of every (detector, filter) (audit finding 3) is kept within [1/value, value] by clamping its raw "
                        "parameter after every operator step; 0 = no bound (the old default). A WARNING is logged when a scale or the floor sits at its bound")
    g.add_argument("--loss", choices=["gauss", "huber"], default="huber")
    g.add_argument("--huber-delta", type=float, default=3.0)
    g = ap.add_argument_group("training")
    g.add_argument("--budget-min", type=float, default=30.0)
    g.add_argument("--steps", type=int, default=1000000)
    g.add_argument("--batch-patches", type=int, default=8)
    g.add_argument("--patch-arcsec", type=float, default=2.5)
    g.add_argument("--inner-sky", type=int, default=2)
    g.add_argument("--lr-op", type=float, default=1e-3)
    g.add_argument("--lr-sky", type=float, default=0.3)
    g.add_argument("--lr-bkg", type=float, default=0.1)
    g.add_argument("--lr-affine", type=float, default=0.1, help="Adam lr of the raw affine (x 1e-3 matrix, x 10 mas shift)")
    g.add_argument("--lr-codes", type=float, default=0.02)
    g.add_argument("--clip", type=float, default=10.0)
    g.add_argument("--prior-pixels", type=float, default=0.0, help="random-effect priors are divided by this pixel count (0 = mean valid pixels per training exposure)")
    g.add_argument("--is-frac", type=float, default=0.5)
    g.add_argument("--is-snr", type=float, default=20.0)
    g.add_argument("--is-refresh", type=int, default=200)
    g.add_argument("--held-steps", type=int, default=150)
    g.add_argument("--held-xfit", type=int, choices=[0, 1], default=1,
                   help="1: held-out exposures are scored cross-fitted (audit finding 8): the per-exposure terms (background, affine, gain, codes, stripes) "
                   "are fitted on one half of the interleaved 4 x 4 pixel blocks and predict the other, then the reverse; the written prediction of "
                   "every pixel comes from terms that never saw it. The old all-pixel score is also reported. 0: the old all-pixel fit only")
    g.add_argument("--held-step-rows", type=int, choices=[0, 1], default=1,
                   help="1: the held-out optimiser steps only the exposures present in the batch (as training does; audit finding 14); 0: all rows every step (old)")
    g.add_argument("--heldout-only", action="store_true",
                   help="skip the fit: load --state, run the held-out stage as configured and write predictions and metrics under --method-name and --out")
    g.add_argument("--state", default="", help="--heldout-only: saved state (fieldop_state.pt) to load; older states load with missing / extra keys reported")
    g.add_argument("--held-reg-steps", type=int, default=100, help="held-out exposures: translation + background pre-fit before the nuisance fit")
    g.add_argument("--reg-prepass-iters", type=int, default=2, help="registration pre-pass: translation-only fits against the warm-start sky")
    g.add_argument("--reg-prepass-steps", type=int, default=150)
    g.add_argument("--pred-margin", type=float, default=0.3)
    g.add_argument("--log-every", type=int, default=25)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="auto")
    g.add_argument("--ckpt-kernel", type=int, default=0, help="activation checkpointing of the kernel field (GPU memory)")
    g.add_argument("--batch-render", type=int, default=1, help="batched FFT rendering of the patches of a step")
    g.add_argument("--pred-batch", type=int, default=8)
    g.add_argument("--render-frame", choices=["detector", "sky"], default="detector", help="frame of the fine (tent-sampled) level grids")
    return ap


LEGACY_DEFAULTS = dict(sky_zero="median", noise_floor_max=0.0, noise_scale_max=0.0, nuisance_steps="mean", held_xfit=0, held_step_rows=0,
                       epoch_per_instrument=0)


def parse_config(argv=None):
    """Command line -> config; --legacy-defaults replaces the defaults listed in LEGACY_DEFAULTS (explicit options still win)."""
    ap = build_parser()
    pre, _ = ap.parse_known_args(argv)
    if pre.legacy_defaults:
        ap.set_defaults(**LEGACY_DEFAULTS)
    return ap.parse_args(argv)


def default_config(**kw):
    a = parse_config(["--legacy-defaults"] if kw.get("legacy_defaults") else [])
    for k, v in kw.items():
        if not hasattr(a, k):
            raise KeyError(k)
        setattr(a, k, v)
    return a


def make_sky(exps, fset, cfg, half, dev, basis=None):
    from .data import splat_coadd
    from .sky import RBFBasis, SkyGrid, spectral_init
    N = int(math.ceil(2 * half / cfg.sky_pitch)) + 1
    basis = basis or RBFBasis(cfg.spec_k).to(dev)
    lam, w, _, _ = fset.tensors(dev)
    Phi_band = (basis(lam.flatten()).view(len(fset), fset.Q, -1) * w[..., None]).sum(1)
    img, wt = splat_coadd([e for e in exps if e.split == "train"], len(fset), N, cfg.sky_pitch, dev,
                          per_area=getattr(cfg, "input_level", "cal") == "rate")
    A0 = spectral_init(img, Phi_band)
    sb = []
    for b in range(img.shape[0]):   # band noise: MAD over the covered nodes only, floored
        v = img[b][wt[b] > 0]
        v = v if v.numel() > 100 else img[b].flatten()
        s_ = 1.4826 * (v - v.median()).abs().median()
        sb.append(torch.maximum(s_, 1e-4 * v.abs().max().clamp(min=1e-12)))
    sb = torch.stack(sb)
    pre = None
    if cfg.sky_precond > 0:
        snr = (img.abs() / sb[:, None, None]).amax(0).clamp(max=cfg.sky_precond_cap)
        pre = 1.0 + cfg.sky_precond * torch.nn.functional.max_pool2d(snr[None, None], 5, 1, 2)[0, 0]
    ped = None
    if getattr(cfg, "sky_positivity_margin", 0.0) > 0:     # positivity margin p per band (sky units) -> constant spectral coefficients
        pb = []
        for b in range(img.shape[0]):
            w_ = wt[b][wt[b] > 0]
            pb.append(cfg.sky_positivity_margin * (1.0 / w_.sqrt()).median() if w_.numel() else torch.zeros((), device=img.device))
        ped = spectral_init(torch.stack(pb).to(img.dtype)[:, None, None], Phi_band)[:, 0, 0]
    if getattr(cfg, "sky_form", "grid") == "items":
        from .sky import ItemSky
        sky = ItemSky(A0, cfg.sky_pitch, basis, cfg.item_count, floor=cfg.item_floor, max_width=cfg.item_max_width, n_levels=cfg.item_levels,
                      n_views=1, seed=cfg.seed,
                      raster=getattr(cfg, "item_raster", "direct"), impl=getattr(cfg, "item_impl", "prototype"),
                      patch=getattr(cfg, "item_patch", "square") if getattr(cfg, "item_raster", "direct") == "pyramid" else "square")
    else:
        sky = SkyGrid(A0, cfg.sky_pitch, basis, n_views=1, precond=pre)
    if ped is not None:
        sky.ped = ped                     # plain attribute (not a buffer: the default state is unchanged)
        if getattr(cfg, "sky_form", "grid") == "items":
            sky.reset_from_coeffs(A0)     # re-seed with the items carrying the layer p
    sky.register_buffer("sig_b", sb)
    return sky, N


def filt_info_of(exps, fset, insts):
    from .data import CHAN_PIX, INST_DIAM
    out = {}
    for e in exps:
        if e.filt in out:
            continue
        J = np.asarray(e.extra.get("J"))
        pix = float(abs(np.linalg.det(J)) ** 0.5)
        out[e.filt] = {"pixel_arcsec": pix, "diameter_m": e.extra.get("diameter_m", INST_DIAM.get(insts[e.inst], 6.5)), "inst": e.inst}
    return out


STRUCTURE_OPTIONS = ("sky_form sky_pitch item_count item_levels spec_k n_lambda level_nh level_factor level_overlap kernel_extent render_pitch render_os_pix "
                     "k_hidden k_layers dh enc_r0 enc_nrad enc_mmax filter_input amplitude distortion pattern_grid stripes pixel_gain pixel_gain_poly filter_shift "
                     "nonlin tau_mode k_space k_time k_lambda k_exp epoch_axes epoch_per_instrument input_level kernel_norm_radius gauge_ref_inst blur_floor "
                     "field_init item_raster item_patch item_impl rate_units loader").split()


def plain_config(cfg):
    """The launch configuration as a plain dict of numbers, strings, booleans, None and lists (anything else as its string)."""
    def p(v):
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (list, tuple)):
            return [p(x) for x in v]
        return str(v)
    return {k: p(v) for k, v in vars(cfg).items()}


def config_differences(stored, cfg, names=STRUCTURE_OPTIONS):
    """{option: (stored, now)} over the options that change the model's structure and differ between a state's stored launch configuration and cfg."""
    now = plain_config(cfg)
    return {k: (stored[k], now[k]) for k in names if k in stored and k in now and stored[k] != now[k]}


def state_for_saving(model, skies, cfg=None):
    """The saved state: every model tensor except the sky's, "skies" {tile: sky state}, and with cfg the launch configuration as a plain dict under
    "launch_config" (older loaders, and load_saved_state without it, ignore the key)."""
    st = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("sky.")}
    if cfg is not None:
        st["launch_config"] = plain_config(cfg)
    st["skies"] = {k: {kk: vv.detach().cpu() for kk, vv in s_.state_dict().items()} for k, s_ in skies.items()}
    return st


def _neutral_value(model, key, like):
    """Neutral value of a parameter a saved state does not contain: the per-exposure terms and the noise scales at zero, the noise floor at its
    starting value; None = no rule (the parameter keeps the starting value of the new model, reported as such)."""
    zero_keys = {"hs", "u", "zD", "affine", "stripe_rows", "stripe_cols", "codes.z", "pattern", "pixgain.logg", "fshift"}
    if key in zero_keys:
        return torch.zeros_like(like)
    if key == "floor_raw":
        return torch.full_like(like, float(torch.log(torch.expm1(torch.tensor(float(model.cfg.floor0))))))
    return None


def _load_tolerant(module, sd, label, log, neutral=None, skip_prefix=()):
    own = module.state_dict()
    extra = sorted(k for k in sd if k not in own)
    bad = sorted(k for k in sd if k in own and tuple(own[k].shape) != tuple(sd[k].shape))
    missing = sorted(k for k in own if k not in sd and not k.startswith(tuple(skip_prefix)))
    merged = {k: v.clone() for k, v in own.items()}
    for k, v in sd.items():
        if k in own and k not in bad:
            merged[k] = v.to(own[k].dtype)
    neutral_set, kept = [], []
    for k in missing:
        nv = neutral(k, own[k]) if neutral is not None else None
        if nv is None:
            kept.append(k)
        else:
            merged[k] = nv.to(own[k].dtype)
            neutral_set.append(k)
    module.load_state_dict(merged)
    rep = {"missing": missing, "set_to_neutral": neutral_set, "kept_starting_value": kept, "extra_ignored": extra, "shape_mismatch_ignored": bad}
    log(f"state load [{label}]: {len(sd) - len(extra) - len(bad)} of {len(own)} tensors loaded; missing {missing} (set to neutral values: {neutral_set}; "
        f"kept at the new model's starting value: {kept}); extra keys ignored {extra}; shape mismatches ignored {bad}")
    return rep


def load_saved_state(model, skies, st, log=print, cfg=None):
    """--heldout-only: load a state written by state_for_saving (this or an older version of the package). Keys the state lacks are set to their
    neutral value (or kept at the starting value of the new model when there is no neutral rule), keys the model lacks (for instance the removed
    all-zero sky.resid) and tensors of another shape are ignored; every such key is logged and returned."""
    st = dict(st)
    sk = st.pop("skies", {})
    st.pop("pixel_gain_meta", None)
    stored = st.pop("launch_config", None)
    diff = None
    if stored is None:
        log("state load: no stored launch configuration in this state (older version); the options of this run are used as given")
    elif cfg is not None:
        diff = config_differences(stored, cfg)
        if diff:
            log("WARNING: options that change the model's structure differ from the stored launch configuration of the state (stored, now): " + "; ".join(
                f"--{k.replace('_', '-')} {a!r} -> {b!r}" for k, (a, b) in diff.items()))
        else:
            log("state load: structure options equal the stored launch configuration")
    rep = {"config_differences": diff, "model": _load_tolerant(model, st, "model", log, neutral=lambda k, like: _neutral_value(model, k, like), skip_prefix=("sky.",)), "skies": {}}
    for k, s_ in skies.items():
        if k not in sk:
            log(f"state load [sky tile {k}]: the state has no sky for this tile; the warm-start sky is kept")
            rep["skies"][k] = {"missing_tile": True}
            continue
        rep["skies"][k] = _load_tolerant(s_, sk[k], f"sky tile {k}", log)
    return rep


def main(argv=None):
    from . import APPROXIMATIONS, __version__
    from . import data as _data
    from .data import load_field, make_patches
    from .model import FieldOperator, resolve_instrument_options
    from .train import (heldout_stage, noise_floor_values, nuisance_sizes, fit, predict_merged, register_prepass, summarise_scores, warm_kernel,
                        write_prediction)
    cfg = parse_config(argv)
    _data.DQ_IGNORE_BITS = int(cfg.dq_ignore_bits)
    _data.RATE_WCS = cfg.rate_wcs
    _data.LOADER = getattr(cfg, "loader", "astropy")
    from . import sky as _sky
    if cfg.item_chunk > 0:
        _sky.ITEM_CHUNK = int(cfg.item_chunk)
    if cfg.item_buckets == "coarse":
        _sky.ITEM_HALF = (2, 4, 8, 16, 32, 64, 128)
        _sky.PYR_HALF = _sky.ITEM_HALF + (256, 512)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "fieldop.log", "a")
    T0 = time.perf_counter()

    def log(msg):
        s = f"[{time.perf_counter() - T0:8.1f}s] {msg}"
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    dev = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else ("cpu" if cfg.device == "auto" else cfg.device))
    torch.manual_seed(cfg.seed)
    src = Path(__file__).parent
    shas = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(src.glob("*.py"))}
    log(f"fieldop {__version__}; torch {torch.__version__}; device {dev}; sources {shas}")
    log("APPROXIMATIONS:")
    for s in APPROXIMATIONS:
        log("  " + s)
    if not cfg.splits:
        cfg.splits = os.path.join(cfg.protocol_root, f"splits_{cfg.field}.json")
    if cfg.heldout_only and not (cfg.state and os.path.exists(cfg.state)):
        raise SystemExit("--heldout-only needs --state FILE (a fieldop_state.pt written by an earlier run)")
    exps, fset, meta, info = load_field(cfg, dev, log=log)
    resolve_instrument_options(cfg, len(info["insts"]), log=log)         # audit findings 5 and 9: refuses several instruments without --gauge-ref-inst
    log(f"instruments {info['insts']}; --gauge-ref-inst {cfg.gauge_ref_inst}; --kernel-norm-radius {cfg.kernel_norm_radius} arcsec; legacy defaults {cfg.legacy_defaults}")
    log(f"loaded {info['n_exposures']} exposures ({info['n_cuts']} cuts in {len(info['tiles'])} tile(s) {info['tiles']}, {info['host_gb']:.2f} GB of pixels): "
        f"{info['n_by_split']}; filters {info['filters']}; detectors {info['dets']}; epoch span {info['epoch_span_days']:.3f} d; "
        f"registration shifts applied to {info['shifts_applied']}")
    if info["multi_tile"]:
        log(f"cuts per tile {info['cuts_per_tile']}; tile offsets [arcsec, west / north] {info['tile_c0']}")
    half = info["half_data"] - cfg.data_margin + cfg.sky_margin
    from .sky import RBFBasis
    from .tiles import TileSet
    basis = RBFBasis(cfg.spec_k).to(dev)
    skies = {}
    for k in info["tiles"]:
        cs = [e for e in exps if e.tile == k]
        for e in cs:
            e.to_dev(dev)
        skies[k], N = make_sky(cs, fset, cfg, half, dev, basis=basis)
        for e in cs:
            e.release()
        if dev.type == "cuda" and info["multi_tile"]:
            torch.cuda.empty_cache()
    sky = skies[info["tiles"][0]]
    if cfg.sky_form == "items":
        log(f"item sky: {sky.n_items} items ({sky.n_levels} seeding lattices), floor {sky.floor:.4f}\", max width {sky.max_width:.3f}\"; "
            f"rasterised on {N} x {N} at {cfg.sky_pitch}\" ({sky.raster_mode}"
            f"{', %d coarse levels, %s patches' % (sky.plan().L, sky.patch_shape) if sky.raster_mode == 'pyramid' else ''}); "
            f"{sky.n_fitted() / 1e6:.3f} M fitted numbers per tile and view, {sky.n_views} view(s)) x {len(skies)} tile(s)")
    else:
        log(f"sky grid {N} x {N} at {cfg.sky_pitch}\" ({sky.A.numel() / 1e6:.2f} M coefficients per tile, {sky.n_views} view(s)) x {len(skies)} tile(s)")
    model = FieldOperator(cfg, fset, meta, sky, dev).to(dev)
    pg = model.attach_pixel_gain(exps, cfg, fset_names=fset.names, det_names=info["dets"])
    if pg is not None:
        for k, lab in enumerate(pg.labels):
            lg_, ob_, cn_, inf_ = pg.map(k)
            nl = 1.0 / np.sqrt(np.maximum(inf_[ob_], 1e-30))
            log(f"pixel gain map {lab}: box {pg.boxes[k]} (r0, c0, rows, cols), {int(ob_.sum())} observed pixels, training cuts per pixel "
                f"median {np.median(cn_[ob_]):.0f} (min {cn_[ob_].min():.0f}, max {cn_[ob_].max():.0f}), information limit of log G per pixel "
                f"(1 / sqrt(sum (d / sig)^2 - 1), F44) median {np.median(nl):.4f}, 10-90 % {np.percentile(nl, 10):.4f}-{np.percentile(nl, 90):.4f}")
    if pg is not None and getattr(cfg, "pixel_gain_prior", "off") != "off":
        pg.set_prior(cfg.pixel_gain_prior)
    fs_labels = model.attach_filter_shift(exps, cfg, fset_names=fset.names, det_names=info["dets"])
    ts = TileSet(model, exps, skies, cfg, log=log)
    if fs_labels is not None:
        if len(ts.shift_groups) != 1:
            raise SystemExit(f"--filter-shift learned needs one common-shift component; this run has {len(ts.shift_groups)}")
        log(f"filter shift: {len(fs_labels)} (detector, filter) pairs {fs_labels}, training exposures per pair {model.fs_ntrain}, "
            f"start {model.filter_shift_table()}, lr {cfg.lr_filter_shift} px")
    if ts.multi:
        log(f"tiles: p {dict(zip(ts.keys, np.round(ts.p, 3).tolist()))}; data GB {({k: round(v / 2 ** 30, 2) for k, v in ts.nbytes.items()})}; "
            f"resident budget {cfg.gpu_resident_gb} GB; {len(ts.shift_groups)} common-shift component(s), {len(ts.pedestal_groups)} pedestal group(s)")
    for c, lv in enumerate(model.levels_ch):
        log(f"channel {info['chans'][c]}: render pitch {model.render_pitch_ch[c]}\", kernel levels {lv.L + 1}, pitches {lv.pitch}, radii {lv.R}, "
            f"extent {cfg.kernel_extent}\", {lv.n_points} kernel nodes per (region node, lambda node)")
    for e in exps:
        pix = float(abs(np.linalg.det(np.asarray(e.extra["J"]))) ** 0.5)
        make_patches(e, max(16, int(round(cfg.patch_arcsec / pix))))
    log(f"patches: {sum(len(e.patches) for e in exps)}; parameters: operator {sum(p.numel() for p in model.param_groups()[0]) / 1e3:.1f} k, "
        f"sky {sky.n_fitted() * sky.n_views / 1e6:.2f} M x {len(skies)}")
    load_report = None
    if cfg.heldout_only:
        load_report = load_saved_state(model, skies, torch.load(cfg.state, map_location="cpu", weights_only=False), log=log, cfg=cfg)
        winfo, reg, rec = {}, {}, {"heldout_only": True, "state": cfg.state}
    else:
        winfo = warm_kernel(model, cfg, filt_info_of(exps, fset, info["insts"]), log=log)
        reg = register_prepass(model, exps, cfg, log=log, ts=ts)
        rec = fit(model, exps, cfg, log=log, ts=ts)
    held = [e.cid for e in exps if e.split == "held"]
    stage = heldout_stage(model, exps, held, cfg, log=log, ts=ts)
    P = stage["P"] if stage["P"] is not None else {k: v.detach() for k, v in model.prepare().items()}
    pred_dir = Path(cfg.protocol_root) / "pred" / cfg.field / cfg.method_name / cfg.split
    if cfg.write_pred:
        pred_dir.mkdir(parents=True, exist_ok=True)

    def fold_terms(e):
        if 0 not in stage["snaps"]:
            return None
        r = int((stage["rows"] == e.eid).nonzero()[0, 0])
        return {f"fit_half{h}": {"affine": (stage["snaps"][h]["affine"][r] * model.aff_scale).tolist(), "u": float(stage["snaps"][h]["u"][r]),
                                 "bkg": stage["snaps"][h]["bkg"][r].tolist()} for h in (0, 1)}

    def meta_of(e):
        i = e.eid
        return {"heldout_xfit": cfg.held_xfit, "heldout_fold_terms": fold_terms(e),
                "heldout_note": ("cross-fitted: the model of each pixel comes from the per-exposure terms (and closed-form stripes) fitted on the other half "
                                 "of the interleaved 4 x 4 pixel blocks; affine / u / bkg below are the all-pixel fit's" if cfg.held_xfit else "all-pixel fit"),
                "fitted_on_heldout": ["background quadratic", "affine (4 matrix + 2 shift)", "gain u_e", "code z_e (kernel) and zD_e (distortion)",
                                      "NIRCam row/column stripes"], "method": cfg.method_name, "affine": (model.affine[i] * model.aff_scale).tolist(),
                "u": float(model.u[i]), "bkg": model.bkg[i].tolist(),
                "units": "rate" if cfg.input_level == "rate" else "cal",
                "sky_units": "MJy/sr" if cfg.input_level != "rate" or getattr(cfg, "rate_units", "dn") == "sky" else "DN/s per arcsec^2 (per filter and detector)",
                "units_note": "rate: DN/s of the _rate file (count rate, no flat, no photometric conversion)" if cfg.input_level == "rate"
                else "cal: MJy/sr, as SCI of the _cal file", "pixel_gain": cfg.pixel_gain}
    scores = {}
    if ts.multi:
        scores = predict_merged(model, exps, held, P, cfg, ts, pred_dir=pred_dir if cfg.write_pred else None, meta_fn=meta_of, log=log, pred=stage["pred"])
    else:
        for i in held:
            e = exps[i]
            m, pv, S, scores[e.name] = stage["pred"](e)
            if cfg.write_pred:
                write_prediction(pred_dir / f"{e.name}.npz", e, m, pv, meta_of(e))
            r = scores[e.name]
            old = r.get("all_pixel_old")
            log(f"held-out {e.name}: chi2/N (pipeline ERR, own mask, no offset) {r['chi2']:.3f} on {r['n']} px; model S/N>5 {r.get('chi2_bright', float('nan')):.3f} "
                f"({r.get('n_bright', 0)} px); model noise {r['chi2_model_noise']:.3f}"
                + (f" [{'cross-fitted' if old else 'all-pixel'}]" + (f"; old all-pixel fit: {old['chi2']:.3f} (model noise {old['chi2_model_noise']:.3f})" if old else "")))
    with torch.no_grad():
        effv = model.effective_variability(P)
        gauges = model.gauge_report(P=P)
        nsz = nuisance_sizes(model, exps, P=P)
    log(f"effective variability (output space): {({k: round(v, 5) for k, v in effv.items()})}")
    log(f"sizes of the learned terms that can absorb a modelling error (audit 17 Nov 2026): {json.dumps(nsz, default=str)}")
    log(f"gauges: {gauges}")
    summ = summarise_scores(scores)
    if summ["written"] is not None:
        log("held-out summary (pixel-weighted): " + ("cross-fitted (written) " if "all_pixel_old" in summ else "all-pixel (written) ")
            + f"chi2/px input noise {summ['written']['chi2_input_noise']:.4f}, model noise {summ['written']['chi2_model_noise']:.4f}"
            + (f"; old all-pixel input noise {summ['all_pixel_old']['chi2_input_noise']:.4f}, model noise {summ['all_pixel_old']['chi2_model_noise']:.4f}"
               if "all_pixel_old" in summ else ""))
    with torch.no_grad():
        ithr = model.instrument_mean_log_throughput(P, info["insts"])
    if len(info["insts"]) > 1 and ithr:
        log(f"mean log-throughput per instrument (reference --gauge-ref-inst {cfg.gauge_ref_inst}; the gauge sets the reference set's to 0): "
            + "; ".join(f"{k}: {v['mean_log_g']:+.4f} ({v['n_train_exposures']} train exposures)" for k, v in ithr.items()))
    nf = noise_floor_values(model)
    ho_n = sum(v["n"] for v in scores.values())
    ho_in = {"chi2_input_noise": (sum(v["chi2"] * v["n"] for v in scores.values()) / ho_n) if ho_n else None, "n_px": ho_n,
             "note": "mean squared residual / input sigma^2 (file sigma x load-time factors; no learned floor or noise scale), own mask, held-out exposures"}
    log(f"final learned noise floor {nf} (bound {getattr(cfg, 'noise_floor_max', 0.0)}); held-out chi2/px (input noise) {ho_in['chi2_input_noise']}")
    metrics = {"config": vars(cfg), "info": info, "noise_floor_final": nf, "noise_floor_first_above_0.2_step": rec.get("noise_floor_first_above"),
               "heldout_input_noise": ho_in, "heldout_summary": summ, "instrument_mean_log_throughput": ithr, "state_load_report": load_report, "fit": rec, "warm_kernel": winfo, "registration_prepass": reg, "tau": model.tau.table(), "heldout_own": scores,
               "effective_variability": effv, "gauges": gauges, "nuisance_sizes": nsz, "tiles": ts.summary() if ts.multi else None,
               "approximations": APPROXIMATIONS, "sources_sha256_16": shas, "pred_dir": str(pred_dir), "seconds": time.perf_counter() - T0}
    if model.fs_on:
        metrics["filter_shift"] = {"unit": "detector pixels (x = column, y = row); position of pixel p is W(p + s)", "pairs": model.filter_shift_table(),
                                   "training_exposures_per_pair": dict(zip(model.fs_labels, model.fs_ntrain))}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1, default=str)
    st = state_for_saving(model, skies, cfg)
    if model.pixgain is not None:
        pg = model.pixgain
        st["pixel_gain_meta"] = {"mode": cfg.pixel_gain, "labels": pg.labels, "boxes": pg.boxes, "offsets": [int(o) for o in pg.off],
                                 "poly_order": pg.order, "input_level": cfg.input_level, "det_names": info["dets"], "filters": info["filters"],
                                 **({"prior": pg.prior_table()} if pg.prior_mode != "off" else {}),
                                 "gauge": pg.gauge_residual(), "note": "pixgain.logg: flat log-gain values, map k = logg[off_k : off_k + rows x cols] "
                                 "reshaped (rows, cols), detector rows r0.., cols c0.. (box = (r0, c0, rows, cols)); pixgain.obs / count / info: "
                                 "observed in training, training cuts per pixel, sum of max(d^2/sig^2 - 1, 0)"}
        metrics["pixel_gain"] = {k: v for k, v in st["pixel_gain_meta"].items() if k != "note"}
        json.dump(metrics, open(out / "metrics.json", "w"), indent=1, default=str)
    if not cfg.heldout_only:
        torch.save(st, out / "fieldop_state.pt")
    log(f"done; predictions in {pred_dir}")
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
