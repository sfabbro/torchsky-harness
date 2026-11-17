"""Synthetic problem with known truth for the field operator: one instrument, one filter, an injected population of Gaussian sources of
known total flux (point-like and extended) on a flat background, and a learned-response model that may start at (and be frozen to) the truth.

Everything here is built from the package's public pieces (synth.render_truth, data.Exposure, model.FieldOperator, train.*); nothing in
fieldop/ is edited. Approximations are listed in TRUTH_REPORT.md.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fieldop.bandpass import FilterSet, box_filter  # noqa: E402
from fieldop.data import Exposure, build_meta, make_patches  # noqa: E402
from fieldop.synth import SynthInstrument, Truth, box_gauss, render_truth  # noqa: E402

PIX = 0.10            # detector pixel [arcsec]
SIG_PSF = 0.07        # core width of the true response [arcsec]
F_WING, WING_RATIO = 0.1, 3.0
SIGMA_NOISE = 1.0     # per-pixel noise, every exposure
BKG = 5.0             # flat background in every exposure (known to the truth only)
KERNEL_STEPS = 2500
SN_BINS = [(0.3, 1), (1, 3), (3, 10), (10, 30), (30, 100)]


def instrument(npix=44):
    return SynthInstrument("A", PIX, npix, 6.5, SIG_PSF, 1.5, space_grad=0.0, chromatic=0.0, flat_grad=0.0, f_wing=F_WING, wing_ratio=WING_RATIO,
                           filters=[("A1", 1.0, 1.3)])


def make_population(seed=12345, per_bin=(12, 10, 6, 4, 4), half=1.8, sig_point=0.03, sig_ext=(0.12, 0.2), pitch=0.05):
    """Source list (x, y, sigma_int, target_snr, kind) with the brightest placed first and a separation that grows with brightness."""
    rng = np.random.default_rng(seed)
    out = []
    for (lo, hi), n in sorted(zip(SN_BINS, per_bin), key=lambda t: -t[0][0]):
        for i in range(n):
            snr = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
            ext = (i == n - 1) and n >= 3 and snr > 1.0     # one extended source per bin where the bin has three or more sources
            sg = float(rng.choice(sig_ext)) if ext else sig_point
            for _ in range(2000):
                x, y = rng.uniform(-half, half, 2)
                ok = True
                for (x2, y2, sg2, s2, k2) in out:
                    dmin = 0.25 + 0.25 * (snr >= 30 or s2 >= 30) + 0.10 * (snr >= 10 or s2 >= 10) + 1.0 * (sg + sg2)
                    if math.hypot(x - x2, y - y2) < dmin:
                        ok = False
                        break
                if ok:
                    out.append((float(x), float(y), sg, snr, "extended" if ext else "point"))
                    break
            else:
                raise RuntimeError("could not place source")
    return out


def exclusion_radius(snr, sg):
    return (0.25 if snr < 10 else (0.40 if snr < 30 else 0.60)) + 2.0 * sg


def empty_positions(pop, half=1.8, spacing=0.2):
    g = np.arange(-half, half + 1e-9, spacing)
    pts = [(x, y) for x in g for y in g
           if all(math.hypot(x - sx, y - sy) > exclusion_radius(sn, sg) for sx, sy, sg, sn, _ in pop)]
    return np.array(pts)


def geometry(n_total, geom_seed=3, npix=44, centre_range=0.3):
    """Dither centres of exposures 0..n_total-1 (the first n are the same whatever n_total is)."""
    rng = np.random.default_rng(geom_seed)
    return rng.uniform(-centre_range, centre_range, (n_total, 2))


def pixel_positions(centre, npix):
    r, c = np.mgrid[0:npix, 0:npix].astype(np.float64)
    off = np.stack([c - (npix - 1) / 2, r - (npix - 1) / 2], -1) * PIX
    pos = off + centre
    return torch.tensor(pos[..., 0], dtype=torch.float32), torch.tensor(pos[..., 1], dtype=torch.float32)


def make_truth(pop_fluxes):
    """Truth object: every source is a round Gaussian of width sigma_int and total flux F (flat spectrum, beta = 0)."""
    gals = np.array([[x, y, sg, F, 0.0, 0.0] for (x, y, sg, F) in pop_fluxes]).reshape(-1, 6)
    return Truth(np.zeros((0, 4)), gals)


def true_sky_nodes(pop_fluxes, N, pitch):
    """Surface brightness of the injected sources at the sky nodes (flux per arcsec^2), (N, N), node (row j, col i) at ((i - c) pitch, (j - c) pitch)."""
    c = (N - 1) / 2
    g = (torch.arange(N, dtype=torch.float64) - c) * pitch
    Y, X = torch.meshgrid(g, g, indexing="ij")
    s = torch.zeros(N, N, dtype=torch.float64)
    for (x, y, sg, F) in pop_fluxes:
        s += F * torch.exp(-0.5 * ((X - x) ** 2 + (Y - y) ** 2) / sg ** 2) / (2 * math.pi * sg * sg)
    return s


def unit_model(x, y, sg, e_X, e_Y, filt):
    """Noiseless data of a unit-flux source (true response, pixel box included) at the pixels of one exposure: the matched profile m."""
    inst = instrument()
    xi = torch.zeros(e_X.numel(), 2)
    t = Truth(np.zeros((0, 4)), np.array([[x, y, sg, 1.0, 0.0, 0.0]]))
    return render_truth(t, inst, filt, e_X.reshape(-1), e_Y.reshape(-1), xi).float()


def build_field(n_train, seed=0, n_held=1, n_val=0, npix=44, pop=None, fluxes=None, geom_seed=3, noise_seed_base=1000, patch_px=15, sky_half=2.5):
    """Exposures 0..n_held-1 held out, then n_train training exposures. `fluxes` (per source) are set by the caller (see flux_for_snr).
    Returns exps, fset, meta, truth, pop_fluxes, insts. The noise of exposure k is a function of (seed, k) only."""
    inst = instrument(npix)
    fset = FilterSet([box_filter("A1", "A", 1.0, 1.3, 3)])
    pop = pop if pop is not None else make_population()
    pf = [(p[0], p[1], p[2], float(f)) for p, f in zip(pop, fluxes)]
    truth = make_truth(pf)
    n_tot = n_held + n_val + n_train
    cen = geometry(n_tot, geom_seed, npix)
    exps = []
    for k in range(n_tot):
        Xh, Yh = pixel_positions(cen[k], npix)
        xi = torch.zeros(npix * npix, 2)
        clean = render_truth(truth, inst, fset.filters[0], Xh.reshape(-1), Yh.reshape(-1), xi).view(npix, npix)
        g = torch.Generator().manual_seed(noise_seed_base * (seed + 1) + k)
        d = clean + BKG + SIGMA_NOISE * torch.randn(npix, npix, generator=g)
        split = "held" if k < n_held else ("val" if k < n_held + n_val else "train")
        Rm = np.eye(2)
        e = Exposure(name=f"A1_{k}", filt=0, inst=0, chan=0, det=0, t=0.0, mjd=0.0, det_shape=(npix, npix), cut0=(0, 0), d=d, sig=torch.full((npix, npix), SIGMA_NOISE),
                     valid=torch.ones(npix, npix, dtype=torch.bool), X=Xh, Y=Yh, R=Rm, split=split, group=f"A1_{k}", stripes=False, bkg0=float(d.median()),
                     Jpix=torch.tensor(np.broadcast_to(Rm * PIX, (npix, npix, 2, 2)).copy(), dtype=torch.float32),
                     extra={"J": (Rm * PIX).tolist(), "true_shift": [0.0, 0.0], "clean": clean, "bkg_true": BKG, "diameter_m": 6.5})
        e.eid = k
        make_patches(e, patch_px)
        exps.append(e)
    meta = build_meta(exps, ["A"], ["A"], ["A"])
    return exps, fset, meta, truth, pf, [inst]


def info_per_source(pop, exps, fset):
    """Sum over the given exposures of m^2 / sigma^2 for a unit-flux source of each population member (the inverse variance of its optimal flux)."""
    out = []
    for (x, y, sg, _, _) in pop:
        tot = 0.0
        for e in exps:
            m = unit_model(x, y, sg, e.X, e.Y, fset.filters[0])
            tot += float((m ** 2).sum()) / SIGMA_NOISE ** 2
        out.append(tot)
    return np.array(out)


def flux_for_snr(pop, n_ref=8, geom_seed=3, npix=44, held=1):
    """Fluxes giving each source its target total S/N with the first n_ref training exposures (the held-out exposure is exposure 0 of the geometry)."""
    fset = FilterSet([box_filter("A1", "A", 1.0, 1.3, 3)])
    cen = geometry(held + n_ref, geom_seed, npix)

    class _E:  # minimal stand-in carrying X, Y
        pass
    es = []
    for k in range(held, held + n_ref):
        e = _E()
        e.X, e.Y = pixel_positions(cen[k], npix)
        es.append(e)
    info = info_per_source(pop, es, fset)
    return np.array([p[3] for p in pop]) / np.sqrt(info)


# ----------------------------------------------------------------------------- the model: operator frozen to the truth
def true_log_kernel(u):
    """log of the true kernel (pixel-box-averaged two-Gaussian response, per unit area) at offsets u (M, 2) [arcsec]."""
    dx, dy = u[:, 0].double(), u[:, 1].double()
    k = (1 - F_WING) * box_gauss(dx, dy, SIG_PSF, PIX) + F_WING * box_gauss(dx, dy, WING_RATIO * SIG_PSF, PIX)
    return torch.log(k.clamp(min=1e-30)).float()


def fit_kernel_to_truth(model, steps=2500, log=print, seed=0):
    """Supervised start of the learned response at the truth: the kernel network and its code are trained on the log of the true kernel
    (log-uniform radii as in the package's own warm start). Returns the final rms error of the log kernel and the relative integral error."""
    lv = model.levels
    gen = torch.Generator().manual_seed(seed + 99)
    params = list(model.net.parameters()) + list(model.codes.f_in.parameters()) + list(model.codes.inst.parameters()) + list(model.codes.hbar.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    f = 0
    lam = model.lam[f]
    fq = torch.full((lam.numel(),), f, dtype=torch.long)
    iq = torch.zeros_like(fq)
    t0 = time.perf_counter()
    for it in range(steps):
        nr = 4096
        rr = torch.exp(torch.empty(nr).uniform_(math.log(0.1 * lv.p_r), math.log(lv.r_ext), generator=gen))
        th = torch.rand(nr, generator=gen) * 2 * math.pi
        us = torch.stack([rr * torch.cos(th), rr * torch.sin(th)], -1)
        feats = model.net.enc(us)
        hp = model.codes.hbar_at(model.piv[f].expand(lam.numel()), fq, iq)
        lg = model.net(feats, hp)
        tgt = true_log_kernel(us)
        loss = ((lg - tgt[None] if lg.dim() == 2 and lg.shape[0] == lam.numel() else lg - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        # check on a fresh set: rms log error inside 3 core widths and integral of |k - k_true| over a 1 arcsec square
        g = torch.arange(-1.0, 1.0 + 1e-9, 0.01)
        YY, XX = torch.meshgrid(g, g, indexing="ij")
        us = torch.stack([XX.reshape(-1), YY.reshape(-1)], -1)
        hp = model.codes.hbar_at(model.piv[f].expand(1), fq[:1], iq[:1])
        lg = model.net(model.net.enc(us), hp).reshape(-1)
        kt = true_log_kernel(us)
        k_fit, k_true = torch.exp(lg), torch.exp(kt)
        rel_l1 = float((k_fit - k_true).abs().sum() / k_true.sum())
        sel = k_true > 1e-3 * k_true.max()
        rms_log = float(((lg - kt)[sel] ** 2).mean().sqrt())
        mass = float(k_fit.sum() / k_true.sum())
    log(f"kernel started at the truth: {steps} steps, rms log-kernel error {rms_log:.4f}, relative L1 error {rel_l1:.4f}, integral ratio {mass:.4f} "
        f"({time.perf_counter() - t0:.1f} s)")
    return {"steps": steps, "rms_log_error": rms_log, "rel_l1_error": rel_l1, "integral_ratio": mass}


# ----------------------------------------------------------------------------- model construction and fit
from fieldop.cli import default_config, make_sky  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.train import chi2_exposure, fit, fit_heldout, predict_exposure, register_prepass, warm_kernel  # noqa: E402


class TruthOperator(FieldOperator):
    """FieldOperator whose optimiser groups leave out the parameters named in `frozen` (a set of ids, filled by `freeze`). This is the only way
    found without editing the package: fit() and fit_heldout() build their optimisers from param_groups()."""
    frozen_ids: set = set()

    def param_groups(self):
        op, nuis, taus, skyp = super().param_groups()
        return [p for p in op if id(p) not in self.frozen_ids], nuis, taus, skyp


def base_config(form="grid", steps=300, **over):
    kw = dict(sky_pitch=0.05, render_pitch=0.025, level_nh=12, kernel_extent=1.0, region_px=11, n_lambda=3, k_hidden=32, spec_k=3,
              warm_kernel_steps=150, pattern_grid=0, stripes=0, distortion=0, amplitude="unit", k_space="off", k_time="off", k_lambda="off", k_exp="off",
              tau_mode="fixed", blur_floor=0.0, floor0=1e-6, batch_patches=4, inner_sky=1, budget_min=1e6, steps=steps, val_every=1000000,
              held_steps=50, held_reg_steps=30, log_every=25, is_snr=10.0, reg_prepass_steps=40, reg_prepass_iters=1, seed=0, patch_arcsec=2.2,
              sky_form=form, item_count=1200, item_levels=3, sky_precond=0.02)
    kw.update(over)
    return default_config(**kw)


def build_model(exps, fset, meta, cfg, freeze_kernel=True, sky_half=2.5, log=print):
    sky, N = make_sky(exps, fset, cfg, sky_half, "cpu")
    model = TruthOperator(cfg, fset, meta, sky, torch.device("cpu"))
    info = {}
    if freeze_kernel:
        info["kernel_start"] = fit_kernel_to_truth(model, steps=KERNEL_STEPS, log=log, seed=cfg.seed)
        fz = set(id(p) for p in list(model.net.parameters()) + list(model.codes.parameters()))
    else:
        warm_kernel(model, cfg, {0: {"pixel_arcsec": PIX, "diameter_m": 6.5, "inst": 0}}, log=log)
        fz = set()
    # frozen in every configuration: the noise model (scale and fractional floor stay at the truth: sigma as given)
    fz |= {id(model.hs), id(model.floor_raw)}
    model.frozen_ids = fz
    info["N"] = N
    return model, info


def noise_floor_heldout(model, exps, cfg, P):
    """Mean squared (data - truth) / sigma over the held-out pixels the score uses: the value the score would have for a perfect model."""
    out = []
    for e in exps:
        if e.split == "held":
            m, pv, S = predict_exposure(model, e, P, cfg, tile=22)
            v = e.valid & pv & torch.isfinite(m)
            r2 = ((e.d - e.extra["clean"] - e.extra["bkg_true"]) / e.sig) ** 2
            out.append((float(r2[v].mean()), int(v.sum())))
    n = sum(o[1] for o in out)
    return sum(o[0] * o[1] for o in out) / n


def run_fit(exps, fset, meta, cfg, freeze_kernel=True, sky_half=2.5, log=print, sky_state=None):
    """Full pipeline: model, start of the response, registration pre-pass, joint fit, held-out nuisance fit and score. Returns (model, result dict)."""
    torch.manual_seed(cfg.seed)
    t0 = time.perf_counter()
    model, info = build_model(exps, fset, meta, cfg, freeze_kernel, sky_half, log)
    t_setup = time.perf_counter()
    register_prepass(model, exps, cfg, log=log)
    t_reg = time.perf_counter()
    rec = fit(model, exps, cfg, log=log)
    t_fit = time.perf_counter()
    held = [e.eid for e in exps if e.split == "held"]
    fit_heldout(model, exps, held, cfg, log=log)
    P = {k: v.detach() for k, v in model.prepare().items()}
    sc = {}
    for e in exps:
        if e.split == "held":
            m, pv, S = predict_exposure(model, e, P, cfg, tile=22)
            sc[e.name] = chi2_exposure(e, m, pv, S=S, snr_min=5.0)
    n = sum(v["n"] for v in sc.values())
    chi2 = sum(v["chi2"] * v["n"] for v in sc.values()) / n
    floor = noise_floor_heldout(model, exps, cfg, P)
    t_end = time.perf_counter()
    res = {"sky_form": cfg.sky_form, "steps": rec["n_steps"], "fit_seconds": rec["seconds"], "setup_seconds": t_setup - t0, "prepass_seconds": t_reg - t_setup,
           "heldout_seconds": t_end - t_fit, "total_seconds": t_end - t0, "heldout_chi2": chi2, "noise_only_chi2": floor, "ratio_to_floor": chi2 / floor,
           "loss_curve": rec["steps"], "kernel_start": info.get("kernel_start"), "n_train": sum(e.split == "train" for e in exps),
           "sky_zero_record": rec.get("sky_zero")}
    return model, res


# ----------------------------------------------------------------------------- measurements
SIG_RES = 0.07          # width of the "resolution element" profile used in the empty-region tests [arcsec]
N_SKY = 101


def sky_band_image(model):
    """Recovered sky as surface brightness at the nodes, (N, N), row = Y, column = X."""
    with torch.no_grad():
        return torch.einsum("k,khw->hw", model.Phi_band[0], model.sky.coeffs()).double()


def node_axis(N, pitch):
    return (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * pitch


def aperture_weights(N, pitch, x, y, sg, sg_w=None, rmax_sig=4.0, rmin=0.11):
    """Sky-node indices, true profile P (density for unit flux, width sg) and aperture weights W (Gaussian of width sg_w; default the true profile)
    around (x, y), out to max(rmax_sig sg_w, rmin) arcsec."""
    sg_w = sg if sg_w is None else sg_w
    ax = node_axis(N, pitch)
    R = max(rmax_sig * sg_w, rmin)
    ix = torch.nonzero((ax - x).abs() <= R)[:, 0]
    iy = torch.nonzero((ax - y).abs() <= R)[:, 0]
    YY, XX = torch.meshgrid(ax[iy], ax[ix], indexing="ij")
    r2 = (XX - x) ** 2 + (YY - y) ** 2
    keep = r2 <= R * R
    P = torch.exp(-0.5 * r2 / sg ** 2) / (2 * math.pi * sg * sg)
    W = torch.exp(-0.5 * r2 / sg_w ** 2) / (2 * math.pi * sg_w * sg_w)
    idx = (iy[:, None] * N + ix[None, :])
    return idx[keep], P[keep], W[keep]


def sky_aperture_flux(img, idx, P, W):
    """Flux of the source profile P in the sky image img (N, N) with aperture weights W: sum(W s) / sum(W P); equals F when img = F P."""
    return float((W * img.reshape(-1)[idx]).sum() / (W * P).sum())


def const_factor(P, W):
    """Aperture flux of a constant sky of unit value: sum(W) / sum(W P)."""
    return float(W.sum() / (W * P).sum())


class IdealCache:
    """Matched profiles m (unit-flux source through the true response and pixel box) at the pixels of the training exposures, for a list of
    (x, y, sigma) targets. The ideal estimator is F = sum m (d - background) / sum m^2 with known background, position, profile and response, which is
    the inverse-variance weighted coadd of the noiseless-kernel-matched data (the optimal linear estimator; its noise is 1 / sqrt(sum m^2))."""

    def __init__(self, targets, exps, fset):
        self.targets = targets
        self.M = []          # list over targets of (n_exp, npix*npix)
        self.info = np.zeros(len(targets))
        for t, (x, y, sg) in enumerate(targets):
            ms = torch.stack([unit_model(x, y, sg, e.X, e.Y, fset.filters[0]) for e in exps])
            self.M.append(ms)
            self.info[t] = float((ms ** 2).sum()) / SIGMA_NOISE ** 2
        self.sigma = 1.0 / np.sqrt(self.info)

    def estimate(self, data, bkg=BKG):
        """data: (n_exp, npix*npix) -> flux estimates (n_targets,)."""
        return np.array([float((m * (data - bkg)).sum() / (m * m).sum()) for m in self.M])


def measure(model, exps, fset, pop, fluxes, empties, cache_src, cache_emp, bkg_fit=None, sky_img=None):
    """Everything measured on one fitted sky. Returns a dict of per-source and per-empty-position arrays (JSON-ready lists).
    Gauge: a constant sky and the exposure backgrounds are degenerate; the fit fixes their split by its pedestal gauge. The reported recovered flux of
    every aperture therefore includes the offset (mean fitted background - true background) acting as a constant sky ('gauge-invariant'); the raw
    values without it are also stored."""
    img = sky_band_image(model) if sky_img is None else sky_img
    N = img.shape[0]
    pitch = 0.05
    tr = [e for e in exps if e.split == "train"]
    d_tr = torch.stack([e.d.reshape(-1) for e in tr]).double().numpy()
    c_tr = torch.stack([e.extra["clean"].reshape(-1) for e in tr]).double().numpy()
    b_fit = float(model.bkg[[e.eid for e in tr], 0].mean()) if bkg_fit is None else bkg_fit
    db = b_fit - BKG
    truth_img = true_sky_nodes([(p[0], p[1], p[2], F) for p, F in zip(pop, fluxes)], N, pitch)
    out = {"bkg_offset": db, "bkg_fit_mean": b_fit}
    # sources: own true profile
    rows = []
    for i, (p, F) in enumerate(zip(pop, fluxes)):
        idx, P, W = aperture_weights(N, pitch, p[0], p[1], p[2])
        idx2, P2, W2 = aperture_weights(N, pitch, p[0], p[1], p[2], sg_w=math.hypot(p[2], SIG_RES))
        r1, r2 = sky_aperture_flux(img, idx, P, W), sky_aperture_flux(img, idx2, P2, W2)
        rows.append((r1, r1 + db * const_factor(P, W), r2 + db * const_factor(P2, W2)))
    rec_raw, rec_gi, rec_m = (np.array([r[j] for r in rows]) for j in range(3))
    ideal = cache_src.estimate(d_tr)
    out["src"] = {"F_true": list(map(float, fluxes)), "snr_true": [float(F / s) for F, s in zip(fluxes, cache_src.sigma)], "sigma_ideal": cache_src.sigma.tolist(),
                  "rec_raw": rec_raw.tolist(), "rec": rec_gi.tolist(), "rec_matched": rec_m.tolist(), "ideal": ideal.tolist(), "kind": [p[4] for p in pop]}
    # empty regions: resolution-element profile
    rows = []
    ref = []
    for (x, y) in empties:
        idx, P, W = aperture_weights(N, pitch, x, y, SIG_RES)
        raw = sky_aperture_flux(img, idx, P, W)
        gi = raw + db * const_factor(P, W)
        rows.append((raw, gi))
        ref.append(sky_aperture_flux(truth_img, idx, P, W))     # what the true sky gives in the same aperture (faint-source tails)
    ide = cache_emp.estimate(d_tr)
    ide_ref = cache_emp.estimate(c_tr + BKG)
    out["emp"] = {"xy": np.asarray(empties).tolist(), "sigma_ideal": cache_emp.sigma.tolist(), "rec_raw": [r[0] for r in rows], "rec": [r[1] for r in rows],
                  "ref_sky": ref, "ideal": ide.tolist(), "ideal_ref": ide_ref.tolist()}
    return out


def summarise_sources(runs):
    """runs: list of measure() outputs (one per noise realisation) with the same population. Per S/N bin (S/N of the first run's N): mean ratio,
    scatter over sources, error of the mean from the realisation-to-realisation spread, for the fit and for the ideal estimator."""
    snr = np.array(runs[0]["src"]["snr_true"])
    F = np.array(runs[0]["src"]["F_true"])
    sig = np.array(runs[0]["src"]["sigma_ideal"])
    res = []
    for lo, hi in SN_BINS:
        sel = (snr >= lo) & (snr < hi) if hi < 100 else (snr >= lo) & (snr <= hi * 1.0001)
        row = {"bin": [lo, hi], "n_sources": int(sel.sum())}
        for key in ("rec", "rec_matched", "ideal"):
            R = np.array([np.array(r["src"][key])[sel] / F[sel] for r in runs])          # (n_real, n_src)
            D = np.array([(np.array(r["src"][key])[sel] - F[sel]) / sig[sel] for r in runs])
            per_real = R.mean(1)
            row[key] = {"mean_ratio": float(R.mean()), "sd_over_sources": float(R.mean(0).std(ddof=1)) if R.shape[1] > 1 else float("nan"),
                        "sd_all_draws": float(R.std(ddof=1)), "sem_realisations": float(per_real.std(ddof=1) / np.sqrt(len(runs))) if len(runs) > 1 else float("nan"),
                        "sem_pooled": float(R.std(ddof=1) / np.sqrt(R.size)), "mean_bias_noise_units": float(D.mean()), "n_draws": int(R.size)}
        res.append(row)
    return res


def summarise_empty(runs):
    """Mean (bias) and rms of (recovered - true-sky reference) in units of the ideal noise of each aperture, for the fit and the ideal estimator."""
    sig = np.array(runs[0]["emp"]["sigma_ideal"])
    ref = np.array(runs[0]["emp"]["ref_sky"])
    ide_ref = np.array(runs[0]["emp"]["ideal_ref"])
    out = {"n_positions": int(sig.size), "n_realisations": len(runs)}
    for key, refv in (("rec", ref), ("ideal", ide_ref)):
        Z = np.array([(np.array(r["emp"][key]) - refv) / sig for r in runs])               # (n_real, n_pos)
        mean_real = Z.mean(1)
        out[key] = {"mean_sigma": float(Z.mean()), "sem_realisations": float(mean_real.std(ddof=1) / np.sqrt(len(runs))) if len(runs) > 1 else float("nan"),
                    "sem_positions": float(Z.mean(0).std(ddof=1) / np.sqrt(Z.shape[1])), "rms_sigma": float(np.sqrt((Z ** 2).mean())),
                    "rms_sigma_per_real": np.sqrt((Z ** 2).mean(1)).tolist(), "sd_about_mean": float(Z.std())}
        if key == "rec":
            Zr = np.array([(np.array(r["emp"]["rec_raw"]) - refv) / sig for r in runs])
            out[key]["mean_sigma_raw_gauge"] = float(Zr.mean())
            out[key]["bkg_offset_per_real"] = [r["bkg_offset"] for r in runs]
    out["ratio_rec_over_ideal_rms"] = out["rec"]["rms_sigma"] / out["ideal"]["rms_sigma"]
    return out
