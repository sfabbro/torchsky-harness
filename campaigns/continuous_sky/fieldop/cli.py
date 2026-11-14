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
    g.add_argument("--sky-tv", type=float, default=0.0)
    g.add_argument("--xfit", type=int, default=1, help="cross-fitting sky views (1 = off, 2 = two folds)")
    g.add_argument("--xfit-frac", type=float, default=0.6)
    g.add_argument("--sky-form", choices=["grid", "items"], default="grid", help="sky parametrisation: coefficient grid, or Gaussian items "
                   "rasterised onto that grid (sky.ItemSky)")
    g.add_argument("--item-count", type=int, default=20000, help="items per sky (tile); split over --item-levels lattices in proportion 4^-l")
    g.add_argument("--item-levels", type=int, default=3, help="coarse-to-fine seeding lattices")
    g.add_argument("--item-floor", type=float, default=0.0, help="minimum item width [arcsec] (0 = 0.4 x --sky-pitch)")
    g.add_argument("--item-max-width", type=float, default=0.0, help="maximum item width [arcsec] (0 = 16 x --sky-pitch); with "
                   "--item-raster direct, patches are capped at +-128 nodes (wider items are truncated); with pyramid, up to the field size")
    g.add_argument("--item-raster", choices=["direct", "pyramid"], default="direct", help="item rasteriser: direct (patch of +-4 widths at "
                   "the sky pitch, cost per item ~ (width / pitch)^2) or pyramid (Gaussian-semigroup pyramid, cost per item independent of "
                   "its size; F40)")
    g.add_argument("--item-patch", choices=["square", "sheared"], default="square", help="pyramid only: patch of each item, square "
                   "(+-4 larger widths) or sheared (+-4 widths in the item's decorrelated coordinates, used where it has 1.5x fewer nodes; "
                   "cost ~ item area instead of larger width squared, for elongated items; also applied on the fine level; F40)")
    g.add_argument("--item-smooth-pitch", type=float, default=0.0, help="coarse smooth grid for diffuse light at this pitch [arcsec] (0 = none)")
    g.add_argument("--item-relocate", type=int, default=0, help="move faint items to the positive back-projected residual at 15/30/45/60 per cent")
    g.add_argument("--item-lr-pos", type=float, default=0.05, help="Adam step of item positions [sky pitches]")
    g.add_argument("--item-lr-width", type=float, default=0.03, help="Adam step of item log widths")
    g.add_argument("--item-lr-angle", type=float, default=0.05, help="Adam step of item angles [rad]")
    g.add_argument("--item-lr-coef", type=float, default=0.05, help="Adam step of item log spectral coefficients")
    g = ap.add_argument_group("kernel")
    g.add_argument("--kernel-extent", type=float, default=10.0)
    g.add_argument("--kernel-norm-radius", type=float, default=0.0,
                   help="arcsec; > 0: the kernel's unit integral is taken inside this radius only (flux convention F41); 0 = whole support")
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
    g.add_argument("--tau-prior", type=float, default=1e-3)
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
    g.add_argument("--gauge-ref-inst", type=int, default=-1)
    g.add_argument("--gauge-shift", type=float, default=0.0, help="extra penalty on the common learned shift (the exact projection is always on)")
    g.add_argument("--blur-floor", type=float, default=1.0, help="blur gauge: core m2 of the sharpest kernel class >= this x (sky pitch)^2 / 3 (0 = off)")
    g.add_argument("--pedestal-gauge", type=int, default=1, help="per filter, training background constants (relative to the exposure medians) sum to zero")
    g.add_argument("--distortion", type=int, default=1)
    g.add_argument("--affine-prior", type=float, default=0.01)
    g.add_argument("--input-level", choices=["cal", "rate"], default="cal")
    g.add_argument("--dq-ignore-bits", type=int, default=0,
                   help="data-quality bits that do not exclude a pixel (F42); 0 = strict DQ == 0; 2 = keep pixels flagged SATURATED only")
    g.add_argument("--nonlin", type=int, default=0)
    g.add_argument("--pattern-grid", type=int, default=8)
    g.add_argument("--stripes", type=int, default=1)
    g.add_argument("--stripe-l2", type=float, default=1e-3)
    g.add_argument("--floor0", type=float, default=0.01)
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


def default_config(**kw):
    a = build_parser().parse_args([])
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
    img, wt = splat_coadd([e for e in exps if e.split == "train"], len(fset), N, cfg.sky_pitch, dev)
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
    if getattr(cfg, "sky_form", "grid") == "items":
        from .sky import ItemSky
        sky = ItemSky(A0, cfg.sky_pitch, basis, cfg.item_count, floor=cfg.item_floor, max_width=cfg.item_max_width, n_levels=cfg.item_levels,
                      smooth_pitch=cfg.item_smooth_pitch, n_views=max(1, cfg.xfit), relocate=cfg.item_relocate, seed=cfg.seed,
                      raster=getattr(cfg, "item_raster", "direct"),
                      patch=getattr(cfg, "item_patch", "square") if getattr(cfg, "item_raster", "direct") == "pyramid" else "square")
    else:
        sky = SkyGrid(A0, cfg.sky_pitch, basis, n_views=max(1, cfg.xfit), precond=pre)
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


def main(argv=None):
    from . import APPROXIMATIONS, __version__
    from . import data as _data
    from .data import load_field, make_patches
    from .model import FieldOperator
    from .train import chi2_exposure, fit, fit_heldout, predict_exposure, predict_merged, register_prepass, warm_kernel, write_prediction
    cfg = build_parser().parse_args(argv)
    _data.DQ_IGNORE_BITS = int(cfg.dq_ignore_bits)
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
    exps, fset, meta, info = load_field(cfg, dev, log=log)
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
        log(f"item sky: {sky.n_items} items ({sky.n_levels} seeding lattices), floor {sky.floor:.4f}\", max width {sky.max_width:.3f}\", smooth "
            f"grid {sky.Ns} x {sky.Ns} at {cfg.item_smooth_pitch}\"; rasterised on {N} x {N} at {cfg.sky_pitch}\" ({sky.raster_mode}"
            f"{', %d coarse levels, %s patches' % (sky.plan().L, sky.patch_shape) if sky.raster_mode == 'pyramid' else ''}); "
            f"{sky.n_fitted() / 1e6:.3f} M fitted numbers per tile and view, {sky.n_views} view(s)) x {len(skies)} tile(s)")
    else:
        log(f"sky grid {N} x {N} at {cfg.sky_pitch}\" ({sky.A.numel() / 1e6:.2f} M coefficients per tile, {sky.n_views} view(s)) x {len(skies)} tile(s)")
    model = FieldOperator(cfg, fset, meta, sky, dev).to(dev)
    ts = TileSet(model, exps, skies, cfg, log=log)
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
    winfo = warm_kernel(model, cfg, filt_info_of(exps, fset, info["insts"]), log=log)
    reg = register_prepass(model, exps, cfg, log=log, ts=ts)
    rec = fit(model, exps, cfg, log=log, ts=ts)
    held = [e.cid for e in exps if e.split == "held"]
    fit_heldout(model, exps, held, cfg, log=log, ts=ts)
    P = {k: v.detach() for k, v in model.prepare().items()}
    pred_dir = Path(cfg.protocol_root) / "pred" / cfg.field / cfg.method_name / cfg.split
    if cfg.write_pred:
        pred_dir.mkdir(parents=True, exist_ok=True)

    def meta_of(e):
        i = e.eid
        return {"fitted_on_heldout": ["background quadratic", "affine (4 matrix + 2 shift)", "gain u_e", "code z_e (kernel) and zD_e (distortion)",
                                      "NIRCam row/column stripes"], "method": cfg.method_name, "affine": (model.affine[i] * model.aff_scale).tolist(),
                "u": float(model.u[i]), "bkg": model.bkg[i].tolist()}
    scores = {}
    if ts.multi:
        scores = predict_merged(model, exps, held, P, cfg, ts, pred_dir=pred_dir if cfg.write_pred else None, meta_fn=meta_of, log=log)
    else:
        for i in held:
            e = exps[i]
            m, pv, S = predict_exposure(model, e, P, cfg)
            scores[e.name] = chi2_exposure(e, m, pv, S=S, snr_min=5.0)
            if cfg.write_pred:
                write_prediction(pred_dir / f"{e.name}.npz", e, m, pv, meta_of(e))
            log(f"held-out {e.name}: chi2/N (pipeline ERR, own mask, no offset) {scores[e.name]['chi2']:.3f} on {scores[e.name]['n']} px; "
                f"model S/N>5 {scores[e.name].get('chi2_bright', float('nan')):.3f} ({scores[e.name].get('n_bright', 0)} px)")
    with torch.no_grad():
        effv = model.effective_variability(P)
        gauges = model.gauge_report(P=P)
    log(f"effective variability (output space): {({k: round(v, 5) for k, v in effv.items()})}")
    log(f"gauges: {gauges}")
    metrics = {"config": vars(cfg), "info": info, "fit": rec, "warm_kernel": winfo, "registration_prepass": reg, "tau": model.tau.table(), "heldout_own": scores,
               "effective_variability": effv, "gauges": gauges, "tiles": ts.summary() if ts.multi else None,
               "approximations": APPROXIMATIONS, "sources_sha256_16": shas, "pred_dir": str(pred_dir), "seconds": time.perf_counter() - T0}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1, default=str)
    st = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("sky.")}
    st["skies"] = {k: {kk: vv.detach().cpu() for kk, vv in s_.state_dict().items()} for k, s_ in skies.items()}
    torch.save(st, out / "fieldop_state.pt")
    log(f"done; predictions in {pred_dir}")
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
