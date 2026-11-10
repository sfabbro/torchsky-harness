"""Learning: warm start of the kernel field, joint fit of sky + operator (sky inner steps), variability scales by validation likelihood,
held-out per-exposure nuisance fit and full-exposure prediction (F19-F26)."""
from __future__ import annotations

import contextlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import LATTICE
from .render import quad_points


def huber_rho(z2, delta):
    z = torch.sqrt(z2.clamp(min=delta * delta))
    return torch.where(z2 < delta * delta, z2, 2.0 * delta * z - delta * delta)


@contextlib.contextmanager
def only_grad(model, params):
    """Temporarily restrict requires_grad to the given parameters."""
    keep = {id(p) for p in params}
    old = {}
    for p in model.parameters():
        old[p] = p.requires_grad
        p.requires_grad_(id(p) in keep)
    try:
        yield
    finally:
        for p, r in old.items():
            p.requires_grad_(r)


# ----------------------------------------------------------------------------- patch pool with importance sampling around bright peaks
class PatchPool:
    """Patches of a set of exposures. Draws: uniform over all patches plus a fraction --is-frac uniform over BRIGHT patches (patches whose
    footprint contains a local maximum of the current model band image above --is-snr x the band's robust sigma; no catalogue).
    Exact reweighting w = 1 / (N pi(p)) so the weighted batch mean is an unbiased estimate of the mean over all patches (F20)."""

    def __init__(self, exps, eids, cfg):
        self.items = [(i, sl) for i in eids for sl in exps[i].patches]
        self.N = len(self.items)
        self.nvalid = np.array([int(exps[i].valid[sl[0]:sl[1], sl[2]:sl[3]].sum()) for i, sl in self.items], float)
        self.mean_valid = float(self.nvalid.mean()) if self.N else 1.0
        self.bright = np.zeros(self.N, bool)
        self.cfg = cfg
        self.fp = []
        for i, sl in self.items:
            e = exps[i]
            v = e.valid[sl[0]:sl[1], sl[2]:sl[3]]
            X, Y = e.X[sl[0]:sl[1], sl[2]:sl[3]][v], e.Y[sl[0]:sl[1], sl[2]:sl[3]][v]
            self.fp.append((e.filt, float(X.min()) + e.shift0[0], float(X.max()) + e.shift0[0], float(Y.min()) + e.shift0[1], float(Y.max()) + e.shift0[1]))

    @torch.no_grad()
    def refresh(self, model):
        img = model.sky.band_images(model.Phi_band)
        mx = F.max_pool2d(img[None], 3, 1, 1)[0]
        peaks = {}
        for b in range(img.shape[0]):
            v = img[b]
            s = model.sky.sig_b[b] if hasattr(model.sky, "sig_b") else 1.4826 * (v - v.median()).abs().median()
            ys, xs = torch.nonzero((v >= mx[b]) & (v > v.median() + self.cfg.is_snr * s), as_tuple=True)
            peaks[b] = ((xs.float() * model.sky.pitch + model.sky.x0).cpu().numpy(), (ys.float() * model.sky.pitch + model.sky.x0).cpu().numpy())
        for k, (f, x0, x1, y0, y1) in enumerate(self.fp):
            px, py = peaks.get(f, (np.zeros(0), np.zeros(0)))
            self.bright[k] = bool(np.any((px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)))
        return int(self.bright.sum())

    def draw(self, B, rng):
        nbright = int(self.bright.sum())
        nb = int(round(self.cfg.is_frac * B)) if nbright > 0 else 0
        nu = B - nb
        sel = list(rng.integers(0, self.N, nu))
        if nb:
            bi = np.nonzero(self.bright)[0]
            sel += list(bi[rng.integers(0, len(bi), nb)])
        sel = np.array(sel)
        pi = nu / B / self.N + (nb / B) * self.bright[sel] / max(nbright, 1)
        w = 1.0 / (self.N * pi)
        return [(self.items[k][0], self.items[k][1], float(w[j])) for j, k in enumerate(sel)]


# ----------------------------------------------------------------------------- losses
def patch_loss(model, exps, eid, sl, P, cfg, view=None, detach_sky=False, method="fft", quad=None, jitter=None, half=None):
    """Summed loss over the valid pixels of one patch; half = 0 / 1 keeps one of two interleaved 4 x 4 pixel-block halves (validation)."""
    e = exps[eid]
    pix = e.pix(sl, only_valid=(method == "quad"))
    jit = bool(cfg.render_jitter) if jitter is None else jitter
    if isinstance(quad, dict):
        quad = quad[e.chan]
    out = model.render(eid, pix, P, method=method, view=view, quad=quad, detach_sky=detach_sky, jitter=jit)
    v = pix["valid"]
    if half is not None:
        v = v & ((((pix["row"] // 4) + (pix["col"] // 4)) % 2) == half)
    et = torch.as_tensor(eid, device=model.dev)
    sig = model.sigma(et, out["S"], pix["sig"])
    z2 = ((pix["d"] - out["m"]) / sig) ** 2
    rho = huber_rho(z2, cfg.huber_delta) if cfg.loss == "huber" else z2
    nll = 0.5 * rho + torch.log(sig / pix["sig"])
    vf = v.float()
    return (nll * vf).sum(), vf.sum().detach(), (z2.detach() * vf).sum()


def pix_loss(model, e, pix, out, cfg, half=None):
    v = pix["valid"]
    if half is not None:
        v = v & ((((pix["row"] // 4) + (pix["col"] // 4)) % 2) == half)
    et = torch.as_tensor(e.eid, device=model.dev)
    sig = model.sigma(et, out["S"], pix["sig"])
    z2 = ((pix["d"] - out["m"]) / sig) ** 2
    rho = huber_rho(z2, cfg.huber_delta) if cfg.loss == "huber" else z2
    nll = 0.5 * rho + torch.log(sig / pix["sig"])
    vf = v.float()
    return (nll * vf).sum(), vf.sum().detach(), (z2.detach() * vf).sum()


def batch_loss(model, exps, items, P, cfg, pool_mean_valid, view_of=None, detach_sky=False, method="fft", quad=None, half=None, jitter=None):
    """Importance-weighted mean loss per pixel of a batch of patches; FFT renders are batched (model.render_batch)."""
    tot, n, chi = 0.0, 0, 0.0
    if method == "quad" or not cfg.batch_render:
        for eid, sl, w in items:
            view = None if view_of is None else view_of(eid)
            l, k, c = patch_loss(model, exps, eid, sl, P, cfg, view=view, detach_sky=detach_sky, method=method, quad=quad, half=half, jitter=jitter)
            tot = tot + w * l
            n += k
            chi += c
        return tot / (len(items) * pool_mean_valid), n, chi
    jit = bool(cfg.render_jitter) if jitter is None else jitter
    by_view = {}
    for k, (eid, sl, w) in enumerate(items):
        by_view.setdefault(None if view_of is None else view_of(eid), []).append(k)
    for view, ks in by_view.items():
        pixs = [exps[items[k][0]].pix(items[k][1]) for k in ks]
        outs = model.render_batch([(items[k][0], px) for k, px in zip(ks, pixs)], P, view=view, detach_sky=detach_sky, jitter=jit)
        for k, px, out in zip(ks, pixs, outs):
            l, kk, c = pix_loss(model, exps[items[k][0]], px, out, cfg, half)
            tot = tot + items[k][2] * l
            n += kk
            chi += c
    return tot / (len(items) * pool_mean_valid), n, chi


@torch.no_grad()
def item_losses(model, exps, items, P, cfg, half=None, view_of=None):
    """Mean loss per valid pixel of each patch (no importance weights): (B,) tensor, for paired comparisons."""
    out = []
    pixs = [exps[eid].pix(sl) for eid, sl, _ in items]
    outs = model.render_batch([(eid, px) for (eid, _, _), px in zip(items, pixs)], P, jitter=False)
    for (eid, _, _), px, o in zip(items, pixs, outs):
        l, n, _ = pix_loss(model, exps[eid], px, o, cfg, half)
        out.append(l / n.clamp(min=1))
    return torch.stack(out)


@torch.no_grad()
def ablation_evidence(model, exps, items, cfg, stats, view_of=None):
    """Likelihood-ratio evidence per FUNCTION axis (F21): on the same validation patches (half 1), the held-out loss with the axis switched
    off minus with it on; the paired differences are accumulated with exponential forgetting and give z_ab = mean / se. An axis is 'kept'
    while z_ab > --tau-occam-z (switching it off significantly worsens held-out prediction)."""
    P_on = {k: v.detach() for k, v in model.prepare().items()}
    L_on = item_losses(model, exps, items, P_on, cfg, half=1)
    for k in list(model.tau.raw.keys()):
        if k in EM_AXES or float(F.softplus(model.tau.raw[k])) < 1e-5:
            continue
        old = model.tau.mode[k]
        model.tau.mode[k] = "off"
        try:
            P_off = {kk: v.detach() for kk, v in model.prepare().items()}
            d = item_losses(model, exps, items, P_off, cfg, half=1) - L_on
        finally:
            model.tau.mode[k] = old
        st = stats.setdefault(k, {"s": 0.0, "s2": 0.0, "n": 0.0})
        b = cfg.ab_beta
        st["s"] = b * st["s"] + float(d.sum())
        st["s2"] = b * st["s2"] + float((d * d).sum())
        st["n"] = b * st["n"] + d.numel()
        mean = st["s"] / st["n"]
        var = max(st["s2"] / st["n"] - mean * mean, 1e-30)
        st["z"] = mean / math.sqrt(var / st["n"])
        st["delta"] = mean


def sky_regs(model, sig_b, cfg, view=None):
    if cfg.sky_neg <= 0 and cfg.sky_tv <= 0:
        return 0.0
    S = torch.einsum("bk,khw->bhw", model.Phi_band, model.sky.coeffs(view)) / sig_b[:, None, None]
    r = cfg.sky_neg * F.relu(-S).pow(2).mean()
    if cfg.sky_tv > 0:
        r = r + cfg.sky_tv * ((S[:, 1:] - S[:, :-1]).abs().mean() + (S[:, :, 1:] - S[:, :, :-1]).abs().mean())
    return r


# ----------------------------------------------------------------------------- warm start of the kernel field (physics only as a start)
def warm_kernel(model, cfg, filt_info, log=print):
    """Fit the kernel field (and hbar) so that, with every variability term at zero, l(v; h) matches a generic start profile per filter node:
    log[(1 - f_w) N(r; sigma) + f_w Moffat(r; 2 sigma, beta = 1.5)], sigma^2 = (0.42 lambda / D)^2 + p^2 / 12 (diffraction width of the
    aperture and the pixel box: inputs only). No PSF model, no library. The target is radial; rotations play no role."""
    if cfg.warm_kernel_steps <= 0:
        return {}
    lv = model.levels
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 11)
    params = list(model.net.parameters()) + list(model.codes.f_in.parameters() if cfg.filter_input == "curve" else model.codes.f_tok.parameters()) \
        + list(model.codes.inst.parameters()) + list(model.codes.hbar.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    fl = sorted(filt_info)
    t0 = time.perf_counter()
    for it in range(cfg.warm_kernel_steps):
        # continuous offsets: log-uniform radius from 0.1 render pitch to the extent, uniform angle (the field is a continuous function)
        nr = 4096
        rr = torch.exp(torch.empty(nr).uniform_(math.log(0.1 * lv.p_r), math.log(lv.r_ext), generator=gen))
        th = torch.rand(nr, generator=gen) * 2 * math.pi
        us = torch.stack([rr * torch.cos(th), rr * torch.sin(th)], -1).to(model.dev)
        r = us.norm(dim=-1)
        feats = model.net.enc(us)
        loss = 0.0
        for f in fl:
            inf = filt_info[f]
            lam = model.lam[f]
            sig = torch.sqrt((0.42 * lam * 1e-6 / inf["diameter_m"] * 206264.8) ** 2 + inf["pixel_arcsec"] ** 2 / 12)   # (Q,)
            rw = 2 * sig[:, None]
            g = torch.exp(-0.5 * (r[None] / sig[:, None]) ** 2) / (2 * math.pi * sig[:, None] ** 2)
            mo = 0.5 / (math.pi * rw ** 2) * (1 + (r[None] / rw) ** 2) ** -1.5
            tgt = torch.log(0.85 * g + 0.15 * mo + 1e-12)
            fq = torch.full((lam.numel(),), f, device=model.dev, dtype=torch.long)
            iq = torch.full_like(fq, int(inf["inst"]))
            hp = model.codes.hbar_at(model.piv[f].expand(lam.numel()), fq, iq)
            tl = model.tau("k_lambda")
            h = hp if tl is None else hp + tl * (model.codes.hbar_at(lam, fq, iq) - hp)
            lg = model.net(feats, h)
            loss = loss + ((lg - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    log(f"kernel warm start: {cfg.warm_kernel_steps} steps, final log-density mse {float(loss.detach()) / len(fl):.4f} ({time.perf_counter() - t0:.1f} s)")
    return {"steps": cfg.warm_kernel_steps, "mse": float(loss.detach()) / len(fl)}


# ----------------------------------------------------------------------------- registration pre-pass (catalogue-free coarse registration)
def register_prepass(model, exps, cfg, log=print):
    """--reg-prepass-iters x (--reg-prepass-steps Adam steps on the per-exposure translation + background ONLY, against the current (warm-start)
    sky and kernel; then the sky warm start is rebuilt from the splat coadd at the registered positions). Training and validation exposures;
    the translations are re-centred on their mean (common-shift gauge). Plays the role of the v31 coarse-shift code (F19)."""
    from .data import splat_coadd
    from .sky import spectral_init
    if cfg.reg_prepass_iters <= 0 or cfg.reg_prepass_steps <= 0:
        return {}
    rng = np.random.default_rng(cfg.seed + 3)
    ids = [e.eid for e in exps if e.split in ("train", "val")]
    pool = PatchPool(exps, ids, cfg)
    pool.refresh(model)
    nuis = model.param_groups()[1]
    out = []
    for it in range(cfg.reg_prepass_iters):
        sb = float(model.e_sigbar.mean())
        opt = torch.optim.Adam([{"params": [nuis["affine"]], "lr": cfg.lr_affine * 3}, {"params": [nuis["bkg"]], "lr": cfg.lr_bkg * sb}])
        with torch.no_grad():
            P = {k: v.detach() for k, v in model.prepare().items()}
        t0 = time.perf_counter()
        with only_grad(model, [nuis["affine"], nuis["bkg"]]):
            for st in range(cfg.reg_prepass_steps):
                items = pool.draw(cfg.batch_patches, rng)
                loss, _, _ = batch_loss(model, exps, items, P, cfg, pool.mean_valid)
                ev = torch.as_tensor(sorted({i for i, _, _ in items}), device=model.dev)
                opt.zero_grad()
                loss.backward()
                model.affine.grad[:, :4] = 0          # translation only
                step_rows(opt, [nuis["affine"], nuis["bkg"]], ev)
                if cfg.pedestal_gauge:
                    model.project_gauges({f: v.to(model.dev) for f, v in train_ids_by_filter(exps).items()})
        with torch.no_grad():
            idt = torch.as_tensor(ids, device=model.dev)
            model.project_shift(torch.as_tensor([e.eid for e in exps if e.split == "train"], device=model.dev))
            t = model.affine[idt, 4:6] * model.aff_scale[4:6]
            shifts = {i: (float(t[k, 0]), float(t[k, 1])) for k, i in enumerate(ids)}
            tr = [e for e in exps if e.split == "train"]
            img, wt = splat_coadd(tr, len(model.fset), model.sky.N, model.sky.pitch, model.dev, extra_shift=shifts)
            A0 = spectral_init(img, model.Phi_band)
            model.sky.A.data[:] = (A0 / model.sky.pre)[None]
        rms = float(t.pow(2).sum(1).mean().sqrt() * 1e3)
        out.append({"iter": it, "rms_translation_mas": rms, "seconds": time.perf_counter() - t0})
        log(f"registration pre-pass {it}: {cfg.reg_prepass_steps} steps, translation rms {rms:.1f} mas ({time.perf_counter() - t0:.1f} s); sky warm start rebuilt")
    return {"iters": out}


# ----------------------------------------------------------------------------- main fit
EM_AXES = ("k_exp", "g_exp", "D_exp")


def train_ids_by_filter(exps):
    out = {}
    for e in exps:
        if e.split == "train":
            out.setdefault(e.filt, []).append(e.eid)
    return {f: torch.tensor(v) for f, v in out.items()}


def set_prior_pixels(cfg, exps, log=print):
    """--prior-pixels 0 = auto: the per-exposure prior weighs as much as one exposure's pixels against the per-pixel mean loss."""
    if cfg.prior_pixels <= 0:
        cfg.prior_pixels = float(np.mean([int(e.valid.sum()) for e in exps if e.split == "train"]))
        log(f"random-effect prior weight: 1 / {cfg.prior_pixels:.0f} per pixel (mean valid pixels per training exposure)")


@torch.no_grad()
def em_update(model, tr, cfg):
    """Type-II (parameter-expanded EM) update of the scales of the per-exposure RANDOM EFFECTS (kernel z_e, gain u_e, distortion zD_e; F21):
    tau <- tau x rms(effect over the training exposures), then the effects are divided by the same rms so tau x effect is unchanged.
    MAP effects without the posterior-variance term: unneeded effects shrink to the prior mean and their tau decays geometrically."""
    tr = torch.as_tensor(tr, device=model.dev)
    for ax, tens in (("k_exp", model.codes.z), ("g_exp", model.u), ("D_exp", model.zD)):
        if ax not in model.tau.raw:
            continue
        rms = float(tens[tr].pow(2).mean().sqrt())
        if not math.isfinite(rms) or rms <= 0:
            continue
        r = max(min(rms, 2.0), 0.5) ** cfg.em_damp          # damped, bounded step
        tau = float(F.softplus(model.tau.raw[ax]))
        new = max(tau * r, 1e-6)
        model.tau.raw[ax].data.fill_(float(torch.log(torch.expm1(torch.tensor(new)))))
        tens.data.mul_(tau / new)


@torch.no_grad()
def evidence_step(model, stats, cfg):
    """Variability scales of the FUNCTION axes by the held-out (validation) likelihood, as a sequential evidence rule (F21): per axis, the
    accumulated validation gradient g = dL_val / dlog tau is tracked by an exponential moving mean and variance; its significance
    z = mean / sd(mean) moves log tau by -lr_tau x clip(z + z_occam, -3, 3). An axis grows only while the held-out data significantly ask
    for it (z < -z_occam); otherwise it shrinks geometrically (Occam). Gradients of the random-effect axes are ignored here (EM step)."""
    beta = cfg.tau_beta
    for k, raw in model.tau.raw.items():
        if k in EM_AXES or raw.grad is None or k == "ab":
            continue
        tau = float(F.softplus(raw))
        g = float(raw.grad) / max(float(torch.sigmoid(raw)), 1e-12) * tau     # dL / dlog tau
        st = stats.setdefault(k, {"m": 0.0, "v": 0.0, "n": 0})
        st["n"] += 1
        st["m"] = beta * st["m"] + (1 - beta) * g
        st["v"] = beta * st["v"] + (1 - beta) * g * g
        if st["n"] < cfg.tau_warm_evals:
            continue
        nb = 1 - beta ** st["n"]
        mean, var = st["m"] / nb, max(st["v"] / nb - (st["m"] / nb) ** 2, 1e-30)
        neff = min(st["n"], (1 + beta) / (1 - beta))
        z = mean / math.sqrt(var / neff)
        st["z"] = z
        ab = stats.get("ab", {}).get(k)
        if ab is not None and "z" in ab:
            # kept (switching the axis off significantly worsens held-out prediction): amplitude follows the validation gradient, no Occam
            # pressure; not kept: geometric shrinkage
            step = max(-3.0, min(3.0, z)) if ab["z"] > cfg.tau_occam_z else cfg.tau_occam_z
        else:
            step = max(-3.0, min(3.0, z + cfg.tau_occam_z))
        new = max(tau * math.exp(-cfg.lr_tau * step), 1e-6)
        if k.startswith("k_"):        # code-space kernel scales are capped (not identifiable on their own, F8 / F21)
            new = min(new, cfg.tau_code_max)
        raw.data.fill_(float(torch.log(torch.expm1(torch.tensor(new)))))


def step_rows(opt, tensors, eids):
    """Adam step that leaves the rows (exposures) absent from the batch untouched (no momentum-driven 'phantom' updates of per-exposure
    nuisances, F24)."""
    keep = torch.ones(tensors[0].shape[0], dtype=torch.bool, device=tensors[0].device)
    keep[eids] = False
    saved = [t.data.clone() for t in tensors]
    opt.step()
    for t, s in zip(tensors, saved):                 # restore absent rows with a masked copy (no boolean indexing: no device sync)
        t.data.copy_(torch.where(keep.view(-1, *([1] * (t.dim() - 1))), s, t.data))


def make_optimizers(model, cfg, sig_sky):
    op, nuis, taus, skyp = model.param_groups()
    sb = float(model.e_sigbar.mean())
    pat = [model.pattern] if model.pattern is not None else []
    pat_ids = {id(p) for p in pat} | {id(model.hs), id(model.floor_raw)}
    op_main = [p for p in op if id(p) not in pat_ids]
    opt_op = torch.optim.Adam([{"params": op_main, "lr": cfg.lr_op, "base": cfg.lr_op},
                               {"params": pat, "lr": 0.02 * sb, "base": 0.02 * sb},
                               {"params": [model.hs, model.floor_raw], "lr": 1e-2, "base": 1e-2}])
    opt_nuis = torch.optim.Adam([{"params": [nuis["bkg"]], "lr": cfg.lr_bkg * sb, "base": cfg.lr_bkg * sb},
                                 {"params": [nuis["affine"]], "lr": cfg.lr_affine, "base": cfg.lr_affine},
                                 {"params": [nuis["u"], nuis["z"], nuis["zD"]], "lr": cfg.lr_codes, "base": cfg.lr_codes},
                                 {"params": [nuis["stripe_rows"], nuis["stripe_cols"]], "lr": 0.05 * sb, "base": 0.05 * sb}])
    opt_sky = torch.optim.Adam([{"params": skyp, "lr": cfg.lr_sky * sig_sky, "base": cfg.lr_sky * sig_sky}])
    opt_tau = torch.optim.Adam([{"params": taus, "lr": cfg.lr_tau, "base": cfg.lr_tau}]) if taus else None
    return opt_op, opt_nuis, opt_sky, opt_tau


def set_lr(opts, f):
    for o in opts:
        if o is None:
            continue
        for g in o.param_groups:
            g["lr"] = g["base"] * f


def lr_factor(p, warm=0.03):
    if p < warm:
        return 0.1 + 0.9 * p / warm
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (p - warm) / (1 - warm))))


def fit(model, exps, cfg, log=print, history=None):
    """Joint fit. Returns a record dict (losses, tau trajectory, gauges, timings)."""
    dev = model.dev
    rng = np.random.default_rng(cfg.seed)
    tr = [e.eid for e in exps if e.split == "train"]
    va = [e.eid for e in exps if e.split == "val"]
    pool = PatchPool(exps, tr, cfg)
    vpool = PatchPool(exps, va, cfg) if va else None   # validation exposures: pixel half 0 fits their nuisances, half 1 scores tau (F21)
    with torch.no_grad():
        if hasattr(model.sky, "sig_b"):
            sig_b = model.sky.sig_b.clone()
        else:
            S = model.sky.band_images(model.Phi_band)
            sig_b = torch.stack([1.4826 * (s - s.median()).abs().median() for s in S]).clamp(min=1e-6)
    sig_sky = float(sig_b.mean())
    opt_op, opt_nuis, opt_sky, opt_tau = make_optimizers(model, cfg, sig_sky)
    op, nuis, taus, skyp = model.param_groups()
    learn_tau = opt_tau is not None and len(va) > 0
    nb = pool.refresh(model)
    log(f"fit: {len(tr)} training / {len(va)} validation exposures, {pool.N} patches ({nb} bright), mean valid px/patch {pool.mean_valid:.0f}; "
        f"tau learned: {learn_tau} ({list(model.tau.raw.keys())})")
    folds = {}
    if cfg.xfit > 1:
        groups = sorted({exps[i].group for i in tr})
        folds = {i: groups.index(exps[i].group) % cfg.xfit for i in tr + va}
    xfit_on = cfg.xfit > 1
    method = cfg.render
    quad = None
    if method == "quad":
        quad = {c: quad_points(model.levels_ch[c], model.render_pitch_ch[c], n_rad=cfg.quad_nrad, n_az=cfg.quad_naz, device=dev) for c in range(model.n_chan)}
    rec = {"steps": [], "tau": [], "val": []}
    n_val_eval = 0
    tr_by_f = {f: v.to(dev) for f, v in train_ids_by_filter(exps).items()}
    tr_t = torch.as_tensor(tr, device=dev)
    ev_stats = {}
    set_prior_pixels(cfg, exps, log)
    t0 = time.perf_counter()
    budget = cfg.budget_min * 60.0
    step = 0
    while True:
        prog = max(step / max(cfg.steps, 1), (time.perf_counter() - t0) / max(budget, 1e-9))
        if prog >= 1.0:
            break
        set_lr([opt_op, opt_nuis, opt_sky], lr_factor(prog))
        if xfit_on and prog >= cfg.xfit_frac:
            with torch.no_grad():
                model.sky.A.data[:] = model.sky.A.data.mean(0, keepdim=True)
            xfit_on = False
            log(f"step {step}: cross-fitting ends, sky views averaged")
        view_of = (lambda i: folds.get(i, 0)) if xfit_on else None
        # (a) sky inner steps: sky only, operator fixed
        for _ in range(cfg.inner_sky):
            with only_grad(model, skyp):
                with torch.no_grad():
                    P = model.prepare()
                items = pool.draw(cfg.batch_patches, rng)
                loss, _, _ = batch_loss(model, exps, items, P, cfg, pool.mean_valid, view_of=view_of, method=method, quad=quad)
                loss = loss + sum(sky_regs(model, sig_b, cfg, view=v) for v in (range(cfg.xfit) if xfit_on else [None]))
                opt_sky.zero_grad()
                loss.backward()
                opt_sky.step()
        # (b) joint step: operator + nuisances (+ sky unless cross-fitting, where the operator reads the other fold's sky, detached)
        params = op + list(nuis.values()) + ([] if xfit_on else skyp)
        with only_grad(model, params):
            P = model.prepare()
            items = pool.draw(cfg.batch_patches, rng)
            vo = (lambda i: 1 - folds.get(i, 0)) if xfit_on and cfg.xfit == 2 else view_of
            loss, npx, chi = batch_loss(model, exps, items, P, cfg, pool.mean_valid, view_of=vo, detach_sky=xfit_on, method=method, quad=quad)
            eids = torch.as_tensor(sorted({i for i, _, _ in items}), device=dev)
            pri = model.priors(eids) / cfg.prior_pixels
            gsh = model.shift_gauge(P) if cfg.gauge_shift > 0 else torch.zeros(2, device=dev)
            tot = loss + pri + cfg.gauge_shift * (gsh / 0.001).pow(2).sum() + (0.0 if xfit_on else sky_regs(model, sig_b, cfg))
            opt_op.zero_grad()
            opt_nuis.zero_grad()
            opt_sky.zero_grad()
            tot.backward()
            torch.nn.utils.clip_grad_norm_(op, cfg.clip)
            opt_op.step()
            step_rows(opt_nuis, list(nuis.values()), eids)
            model.project_gauges(tr_by_f if cfg.pedestal_gauge else None, eids)
            model.project_shift(tr_t)
            if not xfit_on:
                opt_sky.step()
        # (c) variability scales by validation likelihood (half A: nuisances of the validation exposures; half B: tau)
        if learn_tau and prog >= cfg.tau_start and step % cfg.val_every == 0 and vpool.N:
            with only_grad(model, [nuis[k] for k in ("bkg", "affine", "u", "z", "zD", "stripe_rows", "stripe_cols")]):
                with torch.no_grad():
                    P = model.prepare()
                for _ in range(cfg.val_nuis_steps):
                    items = vpool.draw(cfg.batch_patches, rng)
                    la, _, _ = batch_loss(model, exps, items, P, cfg, vpool.mean_valid / 2, view_of=view_of, method=method, quad=quad, half=0)
                    ev = torch.as_tensor(sorted({i for i, _, _ in items}), device=dev)
                    opt_nuis.zero_grad()
                    (la + model.priors(ev) / cfg.prior_pixels).backward()
                    step_rows(opt_nuis, list(nuis.values()), ev)
                    model.project_gauges(None, ev)
            with only_grad(model, taus):
                P = model.prepare()
                items = vpool.draw(cfg.batch_patches, rng)
                lb, _, _ = batch_loss(model, exps, items, P, cfg, vpool.mean_valid / 2, view_of=view_of, method=method, quad=quad, half=1)
                ltau = lb / cfg.tau_accum
                ltau.backward()                                  # gradients accumulate over --tau-accum validation evaluations
                n_val_eval += 1
                if cfg.ab_every > 0 and n_val_eval % cfg.ab_every == 0:
                    ablation_evidence(model, exps, items, cfg, ev_stats.setdefault("ab", {}))
                if n_val_eval % cfg.tau_accum == 0:
                    evidence_step(model, ev_stats, cfg)
                    opt_tau.zero_grad()
                    em_update(model, tr, cfg)
            rec["val"].append([step, float(lb)])
        if step % cfg.is_refresh == 0 and step > 0:
            pool.refresh(model)
        if step % cfg.log_every == 0:
            el = time.perf_counter() - t0
            tt = {k: round(v["value"], 5) for k, v in model.tau.table().items()}
            npx, chi = float(npx), float(chi)
            rec["steps"].append([step, el, float(loss), chi / max(npx, 1)])
            rec["tau"].append([step, tt])
            with torch.no_grad():
                gsh = model.shift_gauge(P)
            dl = P.get("dil")
            mem = (f" dil {float(dl):.3f}" if dl is not None else " dil 1") + f" Mref {float(model.blur_moment(1.0 if dl is None else float(dl))) * 1e6:.0f}e-6 floor {model.blur_floor * 1e6:.0f}e-6"
            mem += (f" cuda peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB" if dev.type == "cuda" else "")
            abz = {k: round(v.get("z", float("nan")), 1) for k, v in ev_stats.get("ab", {}).items()}
            if abz:
                rec.setdefault("ablation_z", []).append([step, abz])
            log(f"step {step:6d} t {el:7.1f}s prog {prog:.3f} ablation-z {abz} loss {float(loss):.4f} chi2/px(batch) {chi / max(npx, 1):.3f} "
                f"gauge-shift [{float(gsh[0]) * 1e3:.2f}, {float(gsh[1]) * 1e3:.2f}] mas{mem} tau {tt}")
        if history is not None:
            history(step, model)
        step += 1
    if cfg.xfit > 1 and xfit_on:
        with torch.no_grad():
            model.sky.A.data[:] = model.sky.A.data.mean(0, keepdim=True)
    rec["n_steps"] = step
    rec["seconds"] = time.perf_counter() - t0
    rec["s_per_step"] = rec["seconds"] / max(step, 1)
    if dev.type == "cuda":
        rec["cuda_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    log(f"fit done: {step} steps in {rec['seconds']:.1f} s ({rec['s_per_step']:.3f} s/step); tau {model.tau.table()}")
    return rec


# ----------------------------------------------------------------------------- held-out exposures: per-exposure nuisances only, then predict
def fit_heldout(model, exps, eids, cfg, log=print):
    """Only per-exposure nuisances (background, affine, gain u, code z (kernel z_e and distortion zD), stripes) of the held-out exposures are
    fitted (protocol v1). Operator, sky and tau are frozen."""
    if not eids:
        return {}
    set_prior_pixels(cfg, exps, log)
    dev = model.dev
    rng = np.random.default_rng(cfg.seed + 7)
    nuis = model.param_groups()[1]
    sb = float(model.e_sigbar.mean())
    opt = torch.optim.Adam([{"params": [nuis["bkg"]], "lr": cfg.lr_bkg * sb}, {"params": [nuis["affine"]], "lr": cfg.lr_affine},
                            {"params": [nuis["u"], nuis["z"], nuis["zD"]], "lr": cfg.lr_codes},
                            {"params": [nuis["stripe_rows"], nuis["stripe_cols"]], "lr": 0.05 * sb}])
    pool = PatchPool(exps, eids, cfg)
    pool.refresh(model)
    with torch.no_grad():
        P = {k: v.detach() for k, v in model.prepare().items()}
    t0 = time.perf_counter()
    if cfg.held_reg_steps > 0:   # registration first: translation + background only, larger steps (as the training pre-pass)
        opt_r = torch.optim.Adam([{"params": [nuis["affine"]], "lr": cfg.lr_affine * 3}, {"params": [nuis["bkg"]], "lr": cfg.lr_bkg * sb}])
        with only_grad(model, [nuis["affine"], nuis["bkg"]]):
            for it in range(cfg.held_reg_steps):
                items = pool.draw(cfg.batch_patches, rng)
                loss, _, _ = batch_loss(model, exps, items, P, cfg, pool.mean_valid)
                opt_r.zero_grad()
                loss.backward()
                model.affine.grad[:, :4] = 0
                opt_r.step()
    with only_grad(model, list(nuis.values())):
        for it in range(cfg.held_steps):
            for g in opt.param_groups:
                g.setdefault("base", g["lr"])
                g["lr"] = g["base"] * (1.0 if it < 0.7 * cfg.held_steps else 0.3)
            items = pool.draw(cfg.batch_patches, rng)
            loss, _, _ = batch_loss(model, exps, items, P, cfg, pool.mean_valid)
            ev = torch.as_tensor(sorted({i for i, _, _ in items}), device=dev)
            opt.zero_grad()
            (loss + model.priors(ev) / cfg.prior_pixels).backward()
            opt.step()
    log(f"held-out nuisance fit: {len(eids)} exposures, {cfg.held_steps} steps ({time.perf_counter() - t0:.1f} s)")
    return P


@torch.no_grad()
def predict_exposure(model, e, P, cfg, tile=None):
    """Full prediction over the exposure's cut (all pixels, valid or not) with the FFT renderer; NIRCam stripes re-solved in closed form on
    the residual of valid pixels (3 alternations of ivar-weighted row / column means, F25). Returns model (ny, nx), pred_valid (ny, nx)."""
    ny, nx = e.shape
    tile = tile or max(32, int(round(cfg.patch_arcsec / max(abs(np.linalg.det(np.asarray(e.extra.get("J", [[cfg.pix_default, 0], [0, cfg.pix_default]])))) ** 0.5, 1e-6))))
    m = torch.full((ny, nx), float("nan"), device=e.d.device)
    S = torch.full((ny, nx), float("nan"), device=e.d.device)
    sls = [(y0, min(ny, y0 + tile), x0, min(nx, x0 + tile)) for y0 in range(0, ny, tile) for x0 in range(0, nx, tile)]
    for b0 in range(0, len(sls), max(1, cfg.pred_batch)):
        chunk = sls[b0:b0 + max(1, cfg.pred_batch)]
        pixs = [e.pix(sl) for sl in chunk]
        if cfg.batch_render:
            outs = model.render_batch([(e.eid, px) for px in pixs], P)
        else:
            outs = [model.render(e.eid, px, P) for px in pixs]
        for sl, out in zip(chunk, outs):
            m[sl[0]:sl[1], sl[2]:sl[3]] = out["m"].view(sl[1] - sl[0], sl[3] - sl[2])
            S[sl[0]:sl[1], sl[2]:sl[3]] = out["S"].view(sl[1] - sl[0], sl[3] - sl[2])
    h = model.sky.half_extent - cfg.pred_margin
    pv = (e.X.abs() < h) & (e.Y.abs() < h) & torch.isfinite(m)
    if e.stripes and cfg.stripes:
        v = e.valid & pv
        iv = torch.where(v, 1.0 / e.sig ** 2, torch.zeros_like(e.sig))
        r = torch.where(v, e.d - m, torch.zeros_like(m))
        rows = torch.zeros(ny, device=m.device)
        cols = torch.zeros(nx, device=m.device)
        for _ in range(3):
            rows = ((r - cols[None]) * iv).sum(1) / iv.sum(1).clamp(min=1e-30)
            cols = ((r - rows[:, None]) * iv).sum(0) / iv.sum(0).clamp(min=1e-30)
        m = m + rows[:, None] + cols[None]
    return m, pv, S


def chi2_exposure(e, m, pv, sigma=None, snr_min=None, S=None):
    sig = e.sig if sigma is None else sigma
    v = e.valid & pv & torch.isfinite(m)
    r2 = ((e.d - m) / sig) ** 2
    out = {"chi2": float(r2[v].mean()), "n": int(v.sum())}
    if S is not None and snr_min is not None:
        vb = v & (S / sig > snr_min)
        out["chi2_bright"] = float(r2[vb].mean()) if vb.any() else float("nan")
        out["n_bright"] = int(vb.sum())
    return out


def write_prediction(path, e, m, pv, meta):
    ny, nx = e.det_shape
    full = np.full((ny, nx), np.nan, np.float32)
    val = np.zeros((ny, nx), bool)
    y0, x0 = e.cut0
    a, b = e.shape
    full[y0:y0 + a, x0:x0 + b] = m.cpu().numpy()
    val[y0:y0 + a, x0:x0 + b] = pv.cpu().numpy()
    full[~val] = np.nan
    np.savez_compressed(path, model=full, model_valid=val, meta=json.dumps(meta))
