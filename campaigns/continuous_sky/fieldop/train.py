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

from .kernel import inv_softplus
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
        self.nvalid = np.array([int(exps[i].hostv("valid")[sl[0]:sl[1], sl[2]:sl[3]].sum()) for i, sl in self.items], float)
        self.mean_valid = float(self.nvalid.mean()) if self.N else 1.0
        self.bright = np.zeros(self.N, bool)
        self.cfg = cfg
        self.fp = []
        for i, sl in self.items:
            e = exps[i]
            v = e.hostv("valid")[sl[0]:sl[1], sl[2]:sl[3]]
            X, Y = e.hostv("X")[sl[0]:sl[1], sl[2]:sl[3]][v], e.hostv("Y")[sl[0]:sl[1], sl[2]:sl[3]][v]
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
def patch_loss(model, exps, eid, sl, P, cfg, detach_sky=False, method="fft", quad=None, jitter=None, half=None):
    """Summed loss over the valid pixels of one patch; half = 0 / 1 keeps one of two interleaved 4 x 4 pixel-block halves (validation).
    `eid` is the index of the CUT in exps (== the global exposure id with one tile); the model is addressed by exps[eid].eid."""
    e = exps[eid]
    pix = e.pix(sl, only_valid=(method == "quad"))
    jit = bool(cfg.render_jitter) if jitter is None else jitter
    if isinstance(quad, dict):
        quad = quad[e.chan]
    out = model.render(e.eid, pix, P, method=method, quad=quad, detach_sky=detach_sky, jitter=jit)
    v = pix["valid"]
    if half is not None:
        v = v & ((((pix["row"] // 4) + (pix["col"] // 4)) % 2) == half)
    et = model.et(e.eid)
    sig = model.sigma(et, out["S"], pix["sig"])
    z2 = ((pix["d"] - out["m"]) / sig) ** 2
    rho = huber_rho(z2, cfg.huber_delta) if cfg.loss == "huber" else z2
    nll = 0.5 * rho + torch.log(sig / pix["sig"])
    vf = v.float()
    acc_input_chi(model, pix, out, v)
    return (nll * vf).sum(), vf.sum().detach(), (z2.detach() * vf).sum()


NOISE_FLOOR_WARN = 0.2


def noise_floor_values(model):
    """Learned fractional noise floor(s), softplus(floor_raw), one per channel."""
    with torch.no_grad():
        return [round(float(x), 6) for x in F.softplus(model.floor_raw).detach().cpu()]


def noise_scale_raw_max(value):
    """Largest |hs| (model.hs; noise scale exp(h), h = 3 tanh(hs / 3)) that keeps the scale within [1/value, value]; None = no bound
    (value <= 0, or value so large that the intrinsic bound exp(+-3) is tighter)."""
    if not value or value <= 0:
        return None
    value = max(value, 1.0 / value)
    L = math.log(value)
    return None if L >= 3.0 else 3.0 * math.atanh(L / 3.0)


def bound_report(model, cfg, rel=1e-4):
    """Learned noise terms sitting at their bound (--noise-floor-max, --noise-scale-max): {"floor": [(channel, value)], "noise_scale":
    [("detector/filter", exp(h))]}. Empty lists when nothing is at a bound. Reads the device (call at logged steps only)."""
    out = {"floor": [], "noise_scale": []}
    nf_max = float(getattr(cfg, "noise_floor_max", 0.0) or 0.0)
    if nf_max > 0:
        out["floor"] = [(c, v) for c, v in enumerate(noise_floor_values(model)) if v >= nf_max * (1 - rel)]
    hmax = noise_scale_raw_max(float(getattr(cfg, "noise_scale_max", 0.0) or 0.0))
    if hmax is not None:
        det_names, fn = model.meta.get("det_names", []), model.fset.names
        used = sorted({(int(model.e_det[i]), int(model.e_filt[i])) for i in range(model.E)})
        with torch.no_grad():
            hs = model.hs.detach().cpu()
        for d, f in used:
            if abs(float(hs[d, f])) >= hmax * (1 - rel):
                out["noise_scale"].append((f"{det_names[d] if d < len(det_names) else d}/{fn[f]}", round(math.exp(3 * math.tanh(float(hs[d, f]) / 3)), 4)))
    return out


def acc_input_chi(model, pix, out, v):
    """Add the batch's sum of (data - model)^2 / input sigma^2 over valid pixels to model._chi_in. Diagnostics only: no_grad, nothing read back
    by the optimisation. pix['sig'] is the file sigma (times any load-time factor); no learned floor or noise scale enters."""
    with torch.no_grad():
        r2 = ((pix["d"] - out["m"]).detach() / pix["sig"]) ** 2
        model._chi_in = getattr(model, "_chi_in", 0.0) + (r2 * v.float()).sum()


def pix_loss(model, e, pix, out, cfg, half=None):
    v = pix["valid"]
    if half is not None:
        v = v & ((((pix["row"] // 4) + (pix["col"] // 4)) % 2) == half)
    et = model.et(e.eid)
    sig = model.sigma(et, out["S"], pix["sig"])
    z2 = ((pix["d"] - out["m"]) / sig) ** 2
    rho = huber_rho(z2, cfg.huber_delta) if cfg.loss == "huber" else z2
    nll = 0.5 * rho + torch.log(sig / pix["sig"])
    vf = v.float()
    acc_input_chi(model, pix, out, v)
    return (nll * vf).sum(), vf.sum().detach(), (z2.detach() * vf).sum()


def batch_loss(model, exps, items, P, cfg, pool_mean_valid, detach_sky=False, method="fft", quad=None, half=None, jitter=None):
    """Importance-weighted mean loss per pixel of a batch of patches; FFT renders are batched (model.render_batch)."""
    tot, n, chi = 0.0, 0, 0.0
    if method == "quad" or not cfg.batch_render:
        for eid, sl, w in items:
            l, k, c = patch_loss(model, exps, eid, sl, P, cfg, detach_sky=detach_sky, method=method, quad=quad, half=half, jitter=jitter)
            tot = tot + w * l
            n += k
            chi += c
        return tot / (len(items) * pool_mean_valid), n, chi
    jit = bool(cfg.render_jitter) if jitter is None else jitter
    ks = list(range(len(items)))
    pixs = [exps[items[k][0]].pix(items[k][1]) for k in ks]
    outs = model.render_batch([(exps[items[k][0]].eid, px) for k, px in zip(ks, pixs)], P, detach_sky=detach_sky, jitter=jit)
    for k, px, out in zip(ks, pixs, outs):
        l, kk, c = pix_loss(model, exps[items[k][0]], px, out, cfg, half)
        tot = tot + items[k][2] * l
        n += kk
        chi += c
    return tot / (len(items) * pool_mean_valid), n, chi


@torch.no_grad()
def item_losses(model, exps, items, P, cfg, half=None):
    """Mean loss per valid pixel of each patch (no importance weights): (B,) tensor, for paired comparisons."""
    out = []
    pixs = [exps[eid].pix(sl) for eid, sl, _ in items]
    outs = model.render_batch([(exps[eid].eid, px) for (eid, _, _), px in zip(items, pixs)], P, jitter=False)
    for (eid, _, _), px, o in zip(items, pixs, outs):
        l, n, _ = pix_loss(model, exps[eid], px, o, cfg, half)
        out.append(l / n.clamp(min=1))
    return torch.stack(out)


@torch.no_grad()
def ablation_evidence(model, exps, items, cfg, stats):
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


def sky_regs(model, sig_b, cfg):
    if hasattr(model.sky, "opt_groups"):         # item sky (--sky-form items): non-negative by construction, no regulariser
        return 0.0
    if cfg.sky_neg <= 0:
        return 0.0
    S = torch.einsum("bk,khw->bhw", model.Phi_band, model.sky.coeffs()) / sig_b[:, None, None]
    if getattr(model.sky, "ped", None) is not None:         # --sky-positivity-margin: the penalty acts on sky + p
        S = S + (model.Phi_band @ model.sky.ped.to(S))[:, None, None] / sig_b[:, None, None]
    return cfg.sky_neg * F.relu(-S).pow(2).mean()


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
def batch_eids(exps, items, dev):
    """Global exposure ids (sorted, unique) of the cuts of a batch of patch items."""
    return torch.as_tensor(sorted({exps[i].eid for i, _, _ in items}), device=dev)


def register_prepass(model, exps, cfg, log=print, ts=None):
    """--reg-prepass-iters x (--reg-prepass-steps Adam steps on the per-exposure translation + background ONLY, against the current (warm-start)
    sky and kernel; then the sky warm start is rebuilt from the splat coadd at the registered positions). Training and validation exposures;
    the translations are re-centred on their mean (common-shift gauge, per connected component of tiles). Plays the role of the v31
    coarse-shift code (F19). Several tiles: per iteration, --reg-prepass-steps on each tile in turn (translations are shared), then every
    tile's sky is rebuilt."""
    from .data import splat_coadd
    from .sky import spectral_init
    from .tiles import TileSet
    if cfg.reg_prepass_iters <= 0 or cfg.reg_prepass_steps <= 0:
        return {}
    ts = ts or TileSet.single(model, exps, cfg, log)
    rng = np.random.default_rng(cfg.seed + 3)
    pools = {k: PatchPool(exps, ts.cut_ids(k, ("train", "val")), cfg) for k in ts.keys}
    gids = sorted({e.eid for e in exps if e.split in ("train", "val")})
    if not ts.multi:
        ts.activate(ts.keys[0])
        pools[ts.keys[0]].refresh(model)
    nuis = model.param_groups()[1]
    ped = {f: v.to(model.dev) for f, v in ts.pedestal_groups.items()}
    out = []
    for it in range(cfg.reg_prepass_iters):
        sb = float(model.e_sigbar.mean())
        opt = torch.optim.Adam([{"params": [nuis["affine"]], "lr": cfg.lr_affine * 3}, {"params": [nuis["bkg"]], "lr": cfg.lr_bkg * sb}])
        with torch.no_grad():
            P = {k: v.detach() for k, v in model.prepare().items()}
        t0 = time.perf_counter()
        with only_grad(model, [nuis["affine"], nuis["bkg"]]):
            for k in ts.keys:
                pool = pools[k]
                if pool.N == 0:
                    continue
                if ts.multi:
                    ts.activate(k)
                    pool.refresh(model)
                for st in range(cfg.reg_prepass_steps):
                    items = pool.draw(cfg.batch_patches, rng)
                    loss, _, _ = batch_loss(model, exps, items, P, cfg, pool.mean_valid)
                    ev = batch_eids(exps, items, model.dev)
                    opt.zero_grad()
                    loss.backward()
                    model.affine.grad[:, :4] = 0          # translation only
                    step_rows(opt, [nuis["affine"], nuis["bkg"]], ev, rows=row_steps(model, cfg, [nuis["bkg"]]))
                    if cfg.pedestal_gauge:
                        model.project_gauges(ped)
        with torch.no_grad():
            idt = torch.as_tensor(gids, device=model.dev)
            for g in ts.shift_groups:
                (model.project_filter_shift if model.fs_on else model.project_shift)(g)
            t = model.affine[idt, 4:6] * model.aff_scale[4:6]
            if model.fs_on:                   # the warm-start coadd sees the whole learned displacement J_e s + t_e
                t = t + torch.einsum("nij,nj->ni", model.e_J[idt], model.fshift[model.e_pair[idt]])
            shifts = {i: (float(t[k, 0]), float(t[k, 1])) for k, i in enumerate(gids)}
            for k in ts.keys:
                tr = [exps[c] for c in ts.cut_ids(k, ("train",))]
                if not tr:
                    continue
                if ts.multi:
                    ts.activate(k)
                img, wt = splat_coadd(tr, len(model.fset), ts.skies[k].N, ts.skies[k].pitch, model.dev, extra_shift=shifts,
                                      per_area=model.input_level == "rate")
                A0 = spectral_init(img, model.Phi_band)
                ts.skies[k].reset_from_coeffs(A0)
        rms = float(t.pow(2).sum(1).mean().sqrt() * 1e3)
        out.append({"iter": it, "rms_translation_mas": rms, "seconds": time.perf_counter() - t0})
        log(f"registration pre-pass {it}: {cfg.reg_prepass_steps} steps x {sum(1 for k in ts.keys if pools[k].N)} tile(s), translation rms {rms:.1f} mas "
            f"({time.perf_counter() - t0:.1f} s); sky warm start rebuilt")
    return {"iters": out}


# ----------------------------------------------------------------------------- main fit
EM_AXES = ("k_exp", "g_exp", "D_exp")


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
        elif k in ("g_space", "g_time"):   # throughput fields [log g], capped (F37: a kept axis has no Occam pressure)
            new = min(new, getattr(cfg, "tau_g_max", float("inf")))
        elif k in ("D_static", "D_drift"):  # distortion fields [arcsec], capped (F37)
            new = min(new, getattr(cfg, "tau_d_max", float("inf")))
        raw.data.fill_(float(torch.log(torch.expm1(torch.tensor(new)))))


def step_rows(opt, tensors, eids, rows=None):
    """Adam step that leaves the rows (exposures) absent from the batch untouched (no momentum-driven 'phantom' updates of per-exposure
    nuisances, F24). rows = (scale (E,), set of tensor ids): the update of those tensors is multiplied per row by scale (per-exposure step
    size, --nuisance-steps exposure; None = today)."""
    keep = torch.ones(tensors[0].shape[0], dtype=torch.bool, device=tensors[0].device)
    keep[eids] = False
    saved = [t.data.clone() for t in tensors]
    opt.step()
    for t, s in zip(tensors, saved):                 # restore absent rows with a masked copy (no boolean indexing: no device sync)
        if rows is not None and id(t) in rows[1]:
            t.data.copy_(s + (t.data - s) * rows[0].view(-1, *([1] * (t.dim() - 1))))
        t.data.copy_(torch.where(keep.view(-1, *([1] * (t.dim() - 1))), s, t.data))


def row_steps(model, cfg, tensors):
    """--nuisance-steps exposure: per-exposure step factor sigbar_e / mean(sigbar) for the data-unit nuisances (background, stripes), whose
    Adam step is lr x mean(sigbar) (audit A6); returns (scale, ids) for step_rows, or None (--nuisance-steps mean, today). With it, a fit
    is unchanged when the data and sigma of each exposure are multiplied by a constant (e.g. count rate in place of surface brightness)."""
    if getattr(cfg, "nuisance_steps", "mean") != "exposure":
        return None
    return (model.e_sigbar / model.e_sigbar.mean(), {id(t) for t in tensors})


def step_scaled(opt, tensors, rows):
    """opt.step() with the per-row step factor of row_steps (all rows stepped, as the held-out fit does)."""
    if rows is None:
        opt.step()
        return
    saved = [t.data.clone() if id(t) in rows[1] else None for t in tensors]
    opt.step()
    for t, s in zip(tensors, saved):
        if s is not None:
            t.data.copy_(s + (t.data - s) * rows[0].view(-1, *([1] * (t.dim() - 1))))


def sky_optimizer(sky, cfg, sig_sky):
    """Adam of one sky: one group (grid; lr = --lr-sky x band noise) or the item sky's own groups (positions, log widths, angles, log
    coefficients)."""
    if hasattr(sky, "opt_groups"):
        return torch.optim.Adam(sky.opt_groups(cfg))
    skyp = list(sky.parameters())
    return torch.optim.Adam([{"params": skyp, "lr": cfg.lr_sky * sig_sky, "base": cfg.lr_sky * sig_sky}])


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
    opt_sky = sky_optimizer(model.sky, cfg, sig_sky)
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


def sky_noise(sky, Phi_band):
    """Band noise of a sky (its registered sig_b, else the MAD of its band images)."""
    with torch.no_grad():
        if hasattr(sky, "sig_b"):
            return sky.sig_b.clone()
        S = sky.band_images(Phi_band)
        return torch.stack([1.4826 * (s - s.median()).abs().median() for s in S]).clamp(min=1e-6)


def smooth_image(img, w, s_px):
    """Normalised Gaussian smoothing (width s_px detector pixels) of img over the pixels where w is true; s_px <= 0 returns img."""
    if s_px <= 0:
        return img
    r = int(math.ceil(3 * s_px))
    g = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=img.device, dtype=img.dtype) / s_px) ** 2)
    g = g / g.sum()
    wf = w.to(img.dtype)

    def conv(x):
        x = F.conv2d(x[None, None], g.view(1, 1, 1, -1), padding=(0, r))
        return F.conv2d(x, g.view(1, 1, -1, 1), padding=(r, 0))[0, 0]
    return conv(img * wf) / conv(wf).clamp(min=1e-6)


@torch.no_grad()
def sky_zero_refresh(model, exps, cfg, ts, log=print):
    """--sky-zero masked: reference level of every training exposure = median of its DATA over the valid pixels where the current model's sky
    signal S (rendered sky through the response, before the background), smoothed by a Gaussian of --sky-zero-smooth pixels to lower its
    noise and widen the source masks, satisfies |S| < --sky-zero-snr x the pixel sigma (symmetric, so
    that noise in S, which is correlated with the pixel's own noise, does not bias the median; sources are positive). These levels replace the
    exposure medians in the pedestal gauge (model.bkg0), and the gauge is applied at once. The data, not the residual d - S, are used: a tie to
    the residual is satisfied by any constant sky layer (it moves with the layer) and so does not fix the split; a tie to the data is not.
    Convention: the sky is zero on the pixels the model calls source-free (its faint, unmodelled sources stay in the median). Single tile only.
    Returns a record (per exposure: level, fraction of valid pixels used)."""
    if getattr(cfg, "sky_zero", "median") == "median":
        return None
    if ts is not None and ts.multi:
        raise SystemExit("--sky-zero masked is implemented for a single tile only")
    P = {k: v.detach() for k, v in model.prepare().items()}
    t = float(cfg.sky_zero_snr)
    out = {}
    for e in exps:
        if e.split != "train":
            continue
        _, pv, S = predict_exposure(model, e, P, cfg)
        v0 = e.valid & pv & torch.isfinite(S)
        Ss = smooth_image(torch.where(v0, S, torch.zeros_like(S)), v0, float(cfg.sky_zero_smooth))
        v = v0 & (Ss.abs() < t * e.sig)     # symmetric: a pixel whose S is pulled up OR down by its own noise is dropped alike (no selection bias)
        n0, n = int(v0.sum()), int(v.sum())
        if n < max(100, 0.05 * n0):
            log(f"sky zero: exposure {e.name}: only {n} of {n0} pixels source-free; reference level kept")
            continue
        lvl = float(e.d[v].median())
        model.bkg0[e.eid] = lvl
        out[e.name] = [lvl, n / max(n0, 1)]
    tr_by_f = {f: v.to(model.dev) for f, v in (ts.pedestal_groups.items() if ts is not None else {})}
    if cfg.pedestal_gauge and tr_by_f:
        model.project_gauges(tr_by_f)
    fr = [v[1] for v in out.values()]
    log(f"sky zero (masked): {len(out)} exposures re-referenced, source-free fraction {np.min(fr) if fr else float('nan'):.3f} to "
        f"{np.max(fr) if fr else float('nan'):.3f}, mean level {np.mean([v[0] for v in out.values()]) if out else float('nan'):.5f}")
    return out


def fit(model, exps, cfg, log=print, history=None, ts=None):
    """Joint fit. Returns a record dict (losses, tau trajectory, gauges, timings). `exps` is the list of cuts; `ts` the TileSet (default: one
    tile, the model's sky). Several tiles: a visit of --visit-steps steps on one tile (drawn with probability p_k), the operator, the
    nuisances and that tile's sky are updated; validation (tau) steps use the visited tile's validation cuts when it has some."""
    from .tiles import TileSet
    dev = model.dev
    ts = ts or TileSet.single(model, exps, cfg, log)
    rng = np.random.default_rng(cfg.seed)
    trng = np.random.default_rng(cfg.seed + 101)          # tile draws: a separate stream, so the single-tile fit is unchanged
    tr = sorted({e.eid for e in exps if e.split == "train"})      # global exposure ids
    va = sorted({e.eid for e in exps if e.split == "val"})
    pools = {k: PatchPool(exps, ts.cut_ids(k, ("train",)), cfg) for k in ts.keys}
    vpools = {k: (PatchPool(exps, ts.cut_ids(k, ("val",)), cfg) if ts.cut_ids(k, ("val",)) else None) for k in ts.keys}
    opt_op, opt_nuis, opt_sky0, opt_tau = make_optimizers(model, cfg, float(sky_noise(model.sky, model.Phi_band).mean()))
    ts.make_sky_opts(cfg, lambda sky: sky_noise(sky, model.Phi_band))
    if not ts.multi:
        ts.opt_sky[ts.keys[0]] = opt_sky0          # identical to the single-box optimiser
    op, nuis, taus, _ = model.param_groups()
    rows_nuis = row_steps(model, cfg, [nuis["bkg"], nuis["stripe_rows"], nuis["stripe_cols"]])
    # per-pixel gain (--pixel-gain, F43-F45): its own Adam (step --lr-gain in log G, same warm-up + cosine factor), not clipped, stepped from
    # progress --pixel-gain-start on, projected onto the gauge after every step
    gp = [model.pixgain.logg] if getattr(model, "pixgain", None) is not None else []
    opt_gain = torch.optim.Adam([{"params": gp, "lr": cfg.lr_gain, "base": cfg.lr_gain}]) if gp else None
    if gp:
        model.pixgain.project()
        log(f"pixel gain: {len(model.pixgain.boxes)} maps {model.pixgain.labels}, {gp[0].numel() / 1e6:.3f} M log-gain values "
            f"({int(model.pixgain.obs.sum()) / 1e6:.3f} M observed in training), gauge: {model.pixgain.n_proj} projected monomials per map "
            f"(degree <= {model.pixgain.order}), lr {cfg.lr_gain}, from progress {cfg.pixel_gain_start}")
    opt_fs = (torch.optim.Adam([{"params": [model.fshift], "lr": cfg.lr_filter_shift, "base": cfg.lr_filter_shift}]) if model.fs_on else None)
    gain_prior = bool(gp) and model.pixgain.prior_mode != "off"
    n_train_px = float(sum(int(e.hostv("valid").sum()) for e in exps if e.split == "train")) if gain_prior else 1.0
    if gain_prior:
        log(f"pixel gain prior: {model.pixgain.prior_mode}, start {model.pixgain.prior_table()}, EM from progress {cfg.pixel_gain_em_start} "
            f"every {cfg.pixel_gain_em_every} steps, prior weight 1 / {n_train_px:.0f} per pixel (training pixels)")
    learn_tau = opt_tau is not None and len(va) > 0
    k = ts.keys[0] if not ts.multi else ts.draw(trng)
    ts.activate(k)
    nb = pools[k].refresh(model)
    log(f"fit: {len(tr)} training / {len(va)} validation exposures, {sum(p.N for p in pools.values())} patches ({nb} bright in tile {k}), "
        f"mean valid px/patch {np.mean([p.mean_valid for p in pools.values() if p.N]):.0f}; tau learned: {learn_tau} ({list(model.tau.raw.keys())})"
        + (f"; {len(ts.keys)} tiles, p = {dict(zip(ts.keys, np.round(ts.p, 3).tolist()))}, visit {cfg.visit_steps} steps, "
           f"{len(ts.shift_groups)} shift component(s), {len(ts.pedestal_groups)} pedestal group(s)" if ts.multi else ""))
    method = cfg.render
    quad = None
    if method == "quad":
        quad = {c: quad_points(model.levels_ch[c], model.render_pitch_ch[c], n_rad=cfg.quad_nrad, n_az=cfg.quad_naz, device=dev) for c in range(model.n_chan)}
    rec = {"steps": [], "tau": [], "val": []}
    n_val_eval = 0
    tr_by_f = {f: v.to(dev) for f, v in ts.pedestal_groups.items()}
    ev_stats = {}
    set_prior_pixels(cfg, exps, log)
    t0 = time.perf_counter()
    budget = cfg.budget_min * 60.0
    step = 0
    visit_left = cfg.visit_steps
    nf_max = float(getattr(cfg, "noise_floor_max", 0.0) or 0.0)
    nf_raw_max = float(inv_softplus(torch.tensor(nf_max))) if nf_max > 0 else 0.0
    nf_warn_step = None
    nf_first = torch.full((), -1, dtype=torch.long, device=dev)
    hs_max = noise_scale_raw_max(float(getattr(cfg, "noise_scale_max", 0.0) or 0.0))
    bound_first = None
    if nf_max > 0:
        with torch.no_grad():
            model.floor_raw.clamp_(max=nf_raw_max)
    if hs_max is not None:
        with torch.no_grad():
            model.hs.clamp_(-hs_max, hs_max)
    sz_marks = sorted(float(x) for x in str(getattr(cfg, "sky_zero_at", "")).split(",") if x.strip()) \
        if getattr(cfg, "sky_zero", "median") != "median" else []
    while True:
        prog = max(step / max(cfg.steps, 1), (time.perf_counter() - t0) / max(budget, 1e-9))
        if prog >= 1.0:
            break
        if sz_marks and prog >= sz_marks[0]:            # --sky-zero masked: re-reference the pedestal gauge on the current model's source-free pixels
            while sz_marks and prog >= sz_marks[0]:
                sz_marks.pop(0)
            rec.setdefault("sky_zero", []).append([step, sky_zero_refresh(model, exps, cfg, ts, log)])
        if ts.multi and visit_left <= 0:
            k = ts.draw(trng)
            if ts.activate(k):
                pools[k].refresh(model)
            ts.stats[k]["visits"] += 1
            visit_left = cfg.visit_steps
        visit_left -= 1
        ts.stats[k]["steps"] += 1
        pool, vpool, opt_sky, sig_b = pools[k], vpools[k], ts.opt_sky[k], ts.sig_b[k]
        set_lr([opt_op, opt_nuis, opt_sky, opt_gain, opt_fs], lr_factor(prog))
        skyp = list(model.sky.parameters())
        gain_on = bool(gp) and prog >= cfg.pixel_gain_start
        params = op + list(nuis.values()) + skyp + (gp if gain_on else []) + ([model.fshift] if model.fs_on else [])
        # the operator tables P do not depend on the sky: one evaluation per step serves the sky inner steps (detached copy) and the joint step
        with only_grad(model, params):
            P = model.prepare()
        P_sky = {kk: (v.detach() if torch.is_tensor(v) else v) for kk, v in P.items()}
        # (a) sky inner steps: sky only, operator fixed
        for _ in range(cfg.inner_sky):
            with only_grad(model, skyp):
                items = pool.draw(cfg.batch_patches, rng)
                loss, _, _ = batch_loss(model, exps, items, P_sky, cfg, pool.mean_valid, method=method, quad=quad)
                loss = loss + sky_regs(model, sig_b, cfg)
                opt_sky.zero_grad()
                loss.backward()
                opt_sky.step()
        # (b) joint step: operator + nuisances (+ sky unless cross-fitting, where the operator reads the other fold's sky, detached)
        with only_grad(model, params):
            items = pool.draw(cfg.batch_patches, rng)
            model._chi_in = 0.0
            loss, npx, chi = batch_loss(model, exps, items, P, cfg, pool.mean_valid, method=method, quad=quad)
            eids = batch_eids(exps, items, dev)
            pri = model.priors(eids) / cfg.prior_pixels
            tot = loss + pri + sky_regs(model, sig_b, cfg)
            if gain_on and gain_prior:          # prior on log G per pixel, on the scale of the per-pixel mean loss (full objective / N_train)
                tot = tot + model.pixgain.prior_nll() / n_train_px
            opt_op.zero_grad()
            opt_nuis.zero_grad()
            opt_sky.zero_grad()
            if opt_gain is not None:
                opt_gain.zero_grad()
            if opt_fs is not None:
                opt_fs.zero_grad()
            tot.backward()
            torch.nn.utils.clip_grad_norm_(op, cfg.clip)
            opt_op.step()
            if nf_max > 0:                      # projection onto softplus(floor_raw) <= nf_max; Adam moments untouched (as the gauge projections)
                with torch.no_grad():
                    model.floor_raw.clamp_(max=nf_raw_max)
            if hs_max is not None:              # same for the learned noise scale exp(h) of every (detector, filter): within [1/value, value]
                with torch.no_grad():
                    model.hs.clamp_(-hs_max, hs_max)
            if gain_on:
                opt_gain.step()
                model.pixgain.project()
                if gain_prior and prog >= cfg.pixel_gain_em_start and step % max(1, cfg.pixel_gain_em_every) == 0:
                    model.pixgain.em_update()           # learned prior widths (--pixel-gain-prior); off by default
            step_rows(opt_nuis, list(nuis.values()), eids, rows=rows_nuis)
            if opt_fs is not None:
                step_rows(opt_fs, [model.fshift], model.e_pair[eids])      # only the pairs seen in this batch move (no momentum drift of the others)
            model.project_gauges(tr_by_f if cfg.pedestal_gauge else None, eids)
            for g in ts.shift_groups:
                (model.project_filter_shift if model.fs_on else model.project_shift)(g)
            opt_sky.step()
        # (c) variability scales by validation likelihood (half A: nuisances of the validation exposures; half B: tau)
        if learn_tau and prog >= cfg.tau_start and step % cfg.val_every == 0 and vpool is not None and vpool.N:
            with only_grad(model, [nuis[kk] for kk in ("bkg", "affine", "u", "z", "zD", "stripe_rows", "stripe_cols")]):
                with torch.no_grad():
                    P = model.prepare()
                for _ in range(cfg.val_nuis_steps):
                    items = vpool.draw(cfg.batch_patches, rng)
                    la, _, _ = batch_loss(model, exps, items, P, cfg, vpool.mean_valid / 2, method=method, quad=quad, half=0)
                    ev = batch_eids(exps, items, dev)
                    opt_nuis.zero_grad()
                    (la + model.priors(ev) / cfg.prior_pixels).backward()
                    step_rows(opt_nuis, list(nuis.values()), ev, rows=rows_nuis)
                    model.project_gauges(None, ev)
            with only_grad(model, taus):
                P = model.prepare()
                items = vpool.draw(cfg.batch_patches, rng)
                lb, _, _ = batch_loss(model, exps, items, P, cfg, vpool.mean_valid / 2, method=method, quad=quad, half=1)
                ltau = lb / cfg.tau_accum
                ltau.backward()                                  # gradients accumulate over --tau-accum validation evaluations
                n_val_eval += 1
                if cfg.ab_every > 0 and n_val_eval % cfg.ab_every == 0:
                    ablation_evidence(model, exps, items, cfg, ev_stats.setdefault("ab", {}))
                if n_val_eval % cfg.tau_accum == 0:
                    evidence_step(model, ev_stats, cfg)
                    opt_tau.zero_grad()
                    em_update(model, tr, cfg)
            rec["val"].append([step, float(lb)] + ([k] if ts.multi else []))
        if step % cfg.is_refresh == 0 and step > 0:
            pool.refresh(model)
        if nf_warn_step is None:             # first step above the warning level, tracked on the device and read at logged steps only
            with torch.no_grad():
                above = F.softplus(model.floor_raw).max() > NOISE_FLOOR_WARN + 5e-7      # as round(x, 6) > level
                nf_first = torch.where((nf_first < 0) & above, torch.full_like(nf_first, step), nf_first)
        if step % cfg.log_every == 0:
            if nf_warn_step is None and int(nf_first) >= 0:
                nf_warn_step = rec["noise_floor_first_above"] = int(nf_first)
                log(f"WARNING: learned fractional noise floor {noise_floor_values(model)} exceeds {NOISE_FLOOR_WARN} first at step {nf_warn_step}; bright "
                    f"pixels lose weight and the model-noise chi2 hides misfit (compare chi2/px(input noise))")
            if bound_first is None and (nf_max > 0 or hs_max is not None):
                br = bound_report(model, cfg)
                if br["floor"] or br["noise_scale"]:
                    bound_first = rec["noise_bound_first_step"] = step
                    log(f"WARNING: at step {step} a learned noise term sits at its bound: noise floor {br['floor']} (bound {nf_max}), noise scale "
                        f"exp(h) {br['noise_scale']} (bound x{getattr(cfg, 'noise_scale_max', 0.0)} and 1/x); the model-noise chi2 is then not a "
                        f"measure of fit (compare chi2/px(input noise))")
            el = time.perf_counter() - t0
            tt = {kk: round(v["value"], 5) for kk, v in model.tau.table().items()}
            npx, chi = float(npx), float(chi)
            rec["steps"].append([step, el, float(loss), chi / max(npx, 1)] + ([k] if ts.multi else []))
            rec["tau"].append([step, tt])
            with torch.no_grad():
                gsh = model.shift_gauge(P)
            dl = P.get("dil")
            mem = (f" dil {float(dl):.3f}" if dl is not None else " dil 1") + f" Mref {float(model.blur_moment(1.0 if dl is None else float(dl))) * 1e6:.0f}e-6 floor {model.blur_floor * 1e6:.0f}e-6"
            mem += (f" cuda peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB" if dev.type == "cuda" else "")
            if ts.multi:
                mem += f" tile {k} resident {ts.resident_gb():.2f} GB"
            abz = {kk: round(v.get("z", float("nan")), 1) for kk, v in ev_stats.get("ab", {}).items()}
            if abz:
                rec.setdefault("ablation_z", []).append([step, abz])
            chi_in = float(model._chi_in) / max(npx, 1)
            log(f"step {step:6d} t {el:7.1f}s prog {prog:.3f} ablation-z {abz} loss {float(loss):.4f} chi2/px(batch) {chi / max(npx, 1):.3f} "
                f"chi2/px(input noise) {chi_in:.3f} noise-floor {noise_floor_values(model)} "
                f"gauge-shift [{float(gsh[0]) * 1e3:.2f}, {float(gsh[1]) * 1e3:.2f}] mas{mem} tau {tt}")
        if history is not None:
            history(step, model)
        step += 1
    rec["n_steps"] = step
    if nf_warn_step is None and int(nf_first) >= 0:
        nf_warn_step = rec["noise_floor_first_above"] = int(nf_first)
        log(f"WARNING: learned fractional noise floor exceeds {NOISE_FLOOR_WARN} first at step {nf_warn_step}")
    rec["noise_floor_final"] = noise_floor_values(model)
    if nf_max > 0 or hs_max is not None:
        br = bound_report(model, cfg)
        rec["noise_at_bound_final"] = br
        if br["floor"] or br["noise_scale"]:
            log(f"WARNING: at the end of the fit a learned noise term sits at its bound: noise floor {br['floor']} (bound {nf_max}), noise scale "
                f"exp(h) {br['noise_scale']} (bound x{getattr(cfg, 'noise_scale_max', 0.0)} and 1/x)")
    if max(rec["noise_floor_final"]) > NOISE_FLOOR_WARN:
        log(f"WARNING: final learned fractional noise floor {rec['noise_floor_final']} exceeds {NOISE_FLOOR_WARN}")
    if gp:
        rec["pixel_gain_gauge"] = model.pixgain.gauge_residual()
        if gain_prior:
            rec["pixel_gain_prior"] = model.pixgain.prior_table()
            log(f"pixel gain prior at the end: {rec['pixel_gain_prior']}")
    rec["seconds"] = time.perf_counter() - t0
    rec["s_per_step"] = rec["seconds"] / max(step, 1)
    if ts.multi:
        rec["tiles"] = ts.summary()
    if dev.type == "cuda":
        rec["cuda_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    log(f"fit done: {step} steps in {rec['seconds']:.1f} s ({rec['s_per_step']:.3f} s/step); tau {model.tau.table()}"
        + (f"; per tile {ts.stats}" if ts.multi else ""))
    return rec


# ----------------------------------------------------------------------------- held-out exposures: per-exposure nuisances only, then predict
def fit_heldout(model, exps, eids, cfg, log=print, ts=None, half=None):
    """Only per-exposure nuisances (background, affine, gain u, code z (kernel z_e and distortion zD), stripes) of the held-out exposures are
    fitted (protocol v1). Operator, sky and tau are frozen. `eids` are CUT indices (== exposure ids with one tile). Several tiles: the rows
    are shared by an exposure's cuts, so the fit cycles through the tiles (--held-rounds rounds); tile k gets
    --held-steps x (its held-out cuts / --held-ref-cuts) steps in total (and the same rule for --held-reg-steps); one Adam over all rows.
    half = 0 / 1 (--held-xfit): only the pixels of that half of the interleaved 4 x 4 pixel blocks enter the loss, so the fitted terms never see
    the other half. Rows absent from a step's batch are not moved (--held-step-rows 1, as in training); 0 = every row moves every step (old)."""
    from .tiles import TileSet
    if not eids:
        return {}
    ts = ts or TileSet.single(model, exps, cfg, log)
    set_prior_pixels(cfg, exps, log)
    dev = model.dev
    rng = np.random.default_rng(cfg.seed + 7)
    nuis = model.param_groups()[1]
    sb = float(model.e_sigbar.mean())
    opt = torch.optim.Adam([{"params": [nuis["bkg"]], "lr": cfg.lr_bkg * sb}, {"params": [nuis["affine"]], "lr": cfg.lr_affine},
                            {"params": [nuis["u"], nuis["z"], nuis["zD"]], "lr": cfg.lr_codes},
                            {"params": [nuis["stripe_rows"], nuis["stripe_cols"]], "lr": 0.05 * sb}])
    eset = set(eids)
    held_rows = bool(getattr(cfg, "held_step_rows", 1))
    by_tile = {k: [c for c in ts.cuts[k] if c in eset] for k in ts.keys}
    by_tile = {k: v for k, v in by_tile.items() if v}
    pools = {k: PatchPool(exps, v, cfg) for k, v in by_tile.items()}
    rounds = max(1, int(cfg.held_rounds)) if ts.multi else 1

    def n_steps(S, k):
        if not ts.multi:
            return S
        return max(1, int(round(S * len(by_tile[k]) / max(cfg.held_ref_cuts, 1))))
    if not ts.multi:
        k0 = list(by_tile)[0]
        ts.activate(k0)
        pools[k0].refresh(model)
    with torch.no_grad():
        P = {k: v.detach() for k, v in model.prepare().items()}
    t0 = time.perf_counter()
    if cfg.held_reg_steps > 0:   # registration first: translation + background only, larger steps (as the training pre-pass)
        opt_r = torch.optim.Adam([{"params": [nuis["affine"]], "lr": cfg.lr_affine * 3}, {"params": [nuis["bkg"]], "lr": cfg.lr_bkg * sb}])
        with only_grad(model, [nuis["affine"], nuis["bkg"]]):
            for r in range(rounds):
                for k, pool in pools.items():
                    if ts.multi:
                        ts.activate(k)
                        pool.refresh(model)
                    S = n_steps(cfg.held_reg_steps, k)
                    for it in range(S * r // rounds, S * (r + 1) // rounds):
                        items = pool.draw(cfg.batch_patches, rng)
                        loss, _, _ = batch_loss(model, exps, items, P, cfg, pool.mean_valid, half=half)
                        opt_r.zero_grad()
                        loss.backward()
                        model.affine.grad[:, :4] = 0
                        if held_rows:
                            step_rows(opt_r, [nuis["affine"], nuis["bkg"]], batch_eids(exps, items, dev), rows=row_steps(model, cfg, [nuis["bkg"]]))
                        else:
                            step_scaled(opt_r, [nuis["bkg"]], row_steps(model, cfg, [nuis["bkg"]]))
    with only_grad(model, list(nuis.values())):
        for g in opt.param_groups:
            g.setdefault("base", g["lr"])
        for r in range(rounds):
            for k, pool in pools.items():
                if ts.multi:
                    ts.activate(k)
                    pool.refresh(model)
                S = n_steps(cfg.held_steps, k)
                for it in range(S * r // rounds, S * (r + 1) // rounds):
                    for g in opt.param_groups:
                        g["lr"] = g["base"] * (1.0 if it < 0.7 * S else 0.3)
                    items = pool.draw(cfg.batch_patches, rng)
                    loss, _, _ = batch_loss(model, exps, items, P, cfg, pool.mean_valid, half=half)
                    ev = batch_eids(exps, items, dev)
                    opt.zero_grad()
                    (loss + model.priors(ev) / cfg.prior_pixels).backward()
                    rws = row_steps(model, cfg, [nuis["bkg"], nuis["stripe_rows"], nuis["stripe_cols"]])
                    if held_rows:
                        step_rows(opt, list(nuis.values()), ev, rows=rws)
                    else:
                        step_scaled(opt, [nuis["bkg"], nuis["stripe_rows"], nuis["stripe_cols"]], rws)
    log(f"held-out nuisance fit{'' if half is None else f' on pixel half {half} only'}: {len(eids)} cuts of {len({exps[c].eid for c in eids})} exposures, {cfg.held_steps} steps"
        + (f" (x cuts / {cfg.held_ref_cuts} per tile, {rounds} rounds over {len(pools)} tiles)" if ts.multi else "") + f" ({time.perf_counter() - t0:.1f} s)")
    return P


def half_map(e):
    """(ny, nx) long tensor, 0 / 1: which of the two interleaved 4 x 4 pixel-block halves each pixel of the cut belongs to (detector row and column
    indices, the rule of the validation split in patch_loss / pix_loss)."""
    ny, nx = e.shape
    rr = torch.arange(ny, device=e.d.device) + e.cut0[0]
    cc = torch.arange(nx, device=e.d.device) + e.cut0[1]
    return ((rr // 4)[:, None] + (cc // 4)[None, :]) % 2


NUIS_KEYS = ("bkg", "affine", "u", "z", "zD", "stripe_rows", "stripe_cols")


def snapshot_rows(model, rows):
    nuis = model.param_groups()[1]
    return {k: nuis[k].detach()[rows].clone() for k in NUIS_KEYS}


@torch.no_grad()
def restore_rows(model, rows, snap):
    nuis = model.param_groups()[1]
    for k in NUIS_KEYS:
        nuis[k].data[rows] = snap[k]


def make_heldout_predictor(model, cfg, P, rows, snaps, tile=None):
    """pred(e) -> (m, pv, S, record) for one held-out cut. snaps = {"all": terms fitted on all pixels (the old fit), 0: fitted on half 0, 1: on
    half 1} (0 and 1 only with --held-xfit 1). With them the returned prediction is cross-fitted: the pixels of half q come from the fit on half
    1 - q. The record carries the score of the returned prediction against the input noise (chi2, chi2_bright, n, n_bright) and the model noise
    (chi2_model_noise), and, with cross-fitting, the old all-pixel score under "all_pixel_old". The terms of the all-pixel fit are left in the model."""
    xf = 0 in snaps and 1 in snaps

    @torch.no_grad()
    def one(e, key, fit_half=None):
        restore_rows(model, rows, snaps[key])
        return predict_exposure(model, e, P, cfg, tile=tile, fit_half=fit_half)

    def score(e, m, pv, S):
        rec = chi2_exposure(e, m, pv, S=S, snr_min=5.0)
        with torch.no_grad():
            rec["chi2_model_noise"] = chi2_exposure(e, m, pv, sigma=model.sigma(model.et(e.eid), S, e.sig))["chi2"]
        return rec

    def pred(e):
        res = {}
        if xf:
            for h in (0, 1):
                res[h] = one(e, h, fit_half=h) + (dict(e.extra.get("closed_form_stripes", {})),)
        mA, pvA, SA = one(e, "all")
        stA = e.extra.get("closed_form_stripes")
        recA = score(e, mA, pvA, SA)
        if stA is not None:
            recA["closed_form_stripes"] = stA
        if not xf:
            return mA, pvA, SA, recA
        sel = half_map(e) == 1                       # pixels of half 1 were predicted by the fit on half 0, and vice versa
        m = torch.where(sel, res[0][0], res[1][0])
        S = torch.where(sel, res[0][2], res[1][2])
        pv = res[0][1] & res[1][1]
        rec = score(e, m, pv, S)
        rec["all_pixel_old"] = {k: v for k, v in recA.items() if k != "closed_form_stripes"}
        if stA is not None:
            rec["closed_form_stripes"] = stA
        rec["closed_form_stripes_half_fits"] = {"fit_half0": res[0][3], "fit_half1": res[1][3]}
        return m, pv, S, rec
    return pred


def heldout_stage(model, exps, held, cfg, log=print, ts=None, tile=None):
    """Held-out nuisance fit(s) and the predictor of the held-out cuts. --held-xfit 1 (default): the per-exposure terms are fitted twice, on half 0
    and on half 1 of the interleaved 4 x 4 pixel blocks, and once on all pixels (the old fit, whose terms stay in the model); predictions come from
    the fits that did not see the pixel. --held-xfit 0: the old fit only. Returns {"P", "pred", "snaps", "rows"}; pred is None without held-out cuts."""
    out = {"P": None, "pred": None, "snaps": {}, "rows": None}
    if not held:
        return out
    rows = torch.as_tensor(sorted({exps[c].eid for c in held}), device=model.dev)
    snaps = {}
    if getattr(cfg, "held_xfit", 1):
        snap0 = snapshot_rows(model, rows)
        for h in (0, 1):
            restore_rows(model, rows, snap0)
            fit_heldout(model, exps, held, cfg, log=log, ts=ts, half=h)
            snaps[h] = snapshot_rows(model, rows)
        restore_rows(model, rows, snap0)
    fit_heldout(model, exps, held, cfg, log=log, ts=ts)
    snaps["all"] = snapshot_rows(model, rows)
    P = {k: v.detach() for k, v in model.prepare().items()}
    out.update(P=P, snaps=snaps, rows=rows, pred=make_heldout_predictor(model, cfg, P, rows, snaps, tile=tile))
    return out


def summarise_scores(scores):
    """Pixel-weighted held-out means over the per-exposure records of heldout_stage: {"written": ..., "all_pixel_old": ...}, each with the input-noise
    and model-noise chi2 per pixel."""
    def agg(recs):
        n = sum(r["n"] for r in recs)
        if not n:
            return None
        return {"chi2_input_noise": sum(r["chi2"] * r["n"] for r in recs) / n, "chi2_model_noise": sum(r["chi2_model_noise"] * r["n"] for r in recs) / n, "n_px": n}
    out = {"written": agg(list(scores.values()))}
    if all("all_pixel_old" in r for r in scores.values()) and scores:
        out["all_pixel_old"] = agg([r["all_pixel_old"] for r in scores.values()])
    return out


@torch.no_grad()
def predict_exposure(model, e, P, cfg, tile=None, fit_half=None):
    """Full prediction over the exposure's cut (all pixels, valid or not) with the FFT renderer; NIRCam stripes re-solved in closed form on
    the residual of valid pixels (3 alternations of ivar-weighted row / column means, F25). Returns model (ny, nx), pred_valid (ny, nx).
    fit_half = 0 / 1 (--held-xfit): the closed-form stripes use only the pixels of that half of the interleaved 4 x 4 blocks (a row or column
    then has about half its pixels in the solution; rows or columns with none get 0), so the other half is predicted by terms that never saw it."""
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
    pv = ((e.X + e.shift0[0]).abs() < h) & ((e.Y + e.shift0[1]).abs() < h) & torch.isfinite(m)
    if e.stripes and cfg.stripes:
        v = e.valid & pv
        if fit_half is not None:
            v = v & (half_map(e) == fit_half)
        iv = torch.where(v, 1.0 / e.sig ** 2, torch.zeros_like(e.sig))
        r = torch.where(v, e.d - m, torch.zeros_like(m))
        rows = torch.zeros(ny, device=m.device)
        cols = torch.zeros(nx, device=m.device)
        for _ in range(3):
            rows = ((r - cols[None]) * iv).sum(1) / iv.sum(1).clamp(min=1e-30)
            cols = ((r - rows[:, None]) * iv).sum(0) / iv.sum(0).clamp(min=1e-30)
        m = m + rows[:, None] + cols[None]
        # size of the closed-form stripes (they absorb any row / column structure of the residual, source misfit included): reported only
        sbar = float(e.sig[v].median()) if bool(v.any()) else float("nan")
        ur, uc = iv.sum(1) > 0, iv.sum(0) > 0
        e.extra["closed_form_stripes"] = {"rows_rms_over_sigma": float(rows[ur].pow(2).mean().sqrt()) / sbar if bool(ur.any()) else 0.0,
                                          "cols_rms_over_sigma": float(cols[uc].pow(2).mean().sqrt()) / sbar if bool(uc.any()) else 0.0}
    return m, pv, S


@torch.no_grad()
def nuisance_sizes(model, exps, P=None, radii=(0.5, 1.0, 2.5)):
    """Size at the end of a fit of every learned term that can absorb a modelling error (audit 17 Nov 2026). Read-only: no parameter, buffer,
    optimiser state or random stream is touched. Per-exposure terms are given separately for the training, validation and held-out
    exposures, in units of each exposure's median pipeline sigma (sigbar) where they are additive."""
    P = model.prepare() if P is None else P
    det_names, fn = model.meta.get("det_names", []), model.fset.names
    sb = model.e_sigbar.double()

    def st(x):
        x = x.double().flatten()
        return None if x.numel() == 0 else {"rms": float(x.pow(2).mean().sqrt()), "max_abs": float(x.abs().max())}
    out = {"noise_floor_fraction": noise_floor_values(model)}
    h = 3 * torch.tanh(model.hs / 3)
    used = sorted({(int(model.e_det[i]), int(model.e_filt[i])) for i in range(model.E)})
    out["noise_scale_exp_h"] = {f"{det_names[d] if d < len(det_names) else d}/{fn[f]}": float(torch.exp(h[d, f])) for d, f in used}
    for s in ("train", "val", "held"):
        ids = sorted({e.eid for e in exps if e.split == s})
        if not ids:
            continue
        i = torch.as_tensor(ids, device=model.dev)
        r = {"n": len(ids),
             "bkg_const_minus_median_over_sigma": st((model.bkg[i, 0] - model.bkg0[i]).double() / sb[i]),
             "bkg_shape_terms_over_sigma": st(model.bkg[i, 1:].double() / sb[i, None]),
             "affine_matrix": st(model.affine[i, :4] * model.aff_scale[:4]),
             "translation_mas": st(model.affine[i, 4:6] * model.aff_scale[4:6] * 1e3),
             "gain_u_raw": st(model.u[i]),
             "gain_log_g_exp": st(model.u[i] * model.tau.value("g_exp")) if model.amp == "structured" else None,
             "code_z": st(model.codes.z[i]), "distortion_zD": st(model.zD[i])}
        if model.cfg.stripes and bool(model.e_stripes[i].any()):
            rw = (model.stripe_rows[i] * model.row_mask[i]).double() / sb[i, None]
            cl = (model.stripe_cols[i] * model.col_mask[i]).double() / sb[i, None]
            r["stripes_rows_over_sigma"] = {"rms": float((rw.pow(2).sum() / model.row_mask[i].sum().clamp(min=1)).sqrt()), "max_abs": float(rw.abs().max())}
            r["stripes_cols_over_sigma"] = {"rms": float((cl.pow(2).sum() / model.col_mask[i].sum().clamp(min=1)).sqrt()), "max_abs": float(cl.abs().max())}
        out[s] = r
    if model.pattern is not None:
        pat = {}
        for d in range(model.pattern.shape[0]):
            on = (model.e_det == d)
            if not bool(on.any()):
                continue
            p = model.pattern[d] - model.pattern[d].mean()
            pat[det_names[d] if d < len(det_names) else str(d)] = float(p.pow(2).mean().sqrt()) / float(sb[on].median())
        out["detector_pattern_rms_over_sigma"] = pat
        # audit follow-up (18 Nov 2026): the same pattern read only at the pixels training exposures observe (not at grid nodes no pixel touches),
        # and the part left after the best degree-2 polynomial (which trades with the per-exposure background shapes), both in sigma units
        obs, beyond = {}, {}
        for d in range(model.pattern.shape[0]):
            on = (model.e_det == d)
            if not bool(on.any()):
                continue
            xs = []
            for e in exps:
                if e.split != "train" or int(model.e_det[e.eid]) != d:
                    continue
                ny, nx = e.det_shape
                rr = (torch.arange(e.shape[0], device=e.d.device) + e.cut0[0]).float()[:, None].expand(e.shape)
                cc = (torch.arange(e.shape[1], device=e.d.device) + e.cut0[1]).float()[None, :].expand(e.shape)
                v = e.valid
                xs.append(torch.stack([cc[v] / (nx - 1) * 2 - 1, rr[v] / (ny - 1) * 2 - 1], -1)[:: 7])
            if not xs:
                continue
            xi = torch.cat(xs).to(model.pattern.device)
            p0 = model.pattern[d] - model.pattern[d].mean()
            val = F.grid_sample(p0[None, None], xi[None, None], mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0].double()
            X, Y = xi[:, 0].double(), xi[:, 1].double()
            B = torch.stack([torch.ones_like(X), X, Y, X * X, X * Y, Y * Y], -1)
            res = val - B @ torch.linalg.lstsq(B, val[:, None]).solution[:, 0]
            name = det_names[d] if d < len(det_names) else str(d)
            sbd = float(sb[on].median())
            obs[name] = float(val.pow(2).mean().sqrt()) / sbd
            beyond[name] = float(res.pow(2).mean().sqrt()) / sbd
        out["detector_pattern_observed_rms_over_sigma"] = obs
        out["detector_pattern_beyond_deg2_observed_rms_over_sigma"] = beyond
    if model.pixgain is not None:
        pg = model.pixgain
        out["pixel_gain_log"] = {lab: st(pg.logg[getattr(pg, f"pidx{k}")]) for k, lab in enumerate(pg.labels)}
    if model.nonlin is not None:
        nl = {}
        for c in range(model.n_chan):
            es = (model.e_chan == c).nonzero()
            if es.numel() == 0:
                continue
            e = es[0, 0]
            S = model.e_sigbar[e] * torch.logspace(0, 4, 9, device=model.dev)
            nl[str(c)] = float(((model.apply_nonlin(e, S) - S) / S).abs().max())
        out["nonlinearity_max_fraction"] = nl
    caps = {"g_space": model.cfg.tau_g_max, "g_time": model.cfg.tau_g_max, "D_static": model.cfg.tau_d_max, "D_drift": model.cfg.tau_d_max}
    caps.update({k: model.cfg.tau_code_max for k in ("k_space", "k_time", "k_lambda", "k_exp")})
    out["tau_at_cap"] = {k: bool(model.tau.mode.get(k) == "learned" and model.tau.value(k) >= 0.999 * c) for k, c in caps.items() if k in model.tau.mode}
    # far wings of the kernel: fraction of its integral beyond each radius, at the detector centre of one training exposure per filter (a flat
    # far pedestal is degenerate with the background terms, F41)
    wings = {}
    for f in range(len(model.fset)):
        es = [int(x) for x in model.ref_eids.tolist() if int(model.e_filt[x]) == f]
        if not es:
            continue
        e = torch.as_tensor(es[0], device=model.dev)
        Ks, _ = model.node_kernels(e, torch.zeros(1, 2, device=model.dev), P)
        lv = model.levels_of(e)
        tot = sum(float(K.sum()) for K in Ks)
        wings[fn[f]] = {f"beyond_{r}_arcsec": sum(float((K[0] * (lv.u[l].norm(dim=-1) > r)).sum()) for l, K in enumerate(Ks)) / max(tot, 1e-30)
                        for r in radii}
        wings[fn[f]]["integral_mean_over_lambda_nodes"] = tot / model.fset.Q
    out["kernel_wing_fraction"] = wings
    return out


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


def predict_merged(model, exps, held, P, cfg, ts, pred_dir=None, meta_fn=None, log=print, pred=None):
    """Predictions of the held-out exposures with several tiles: each held-out CUT is rendered with its tile's sky (predict_exposure), placed
    in the full detector frame, and the cuts of one exposure are merged as stage3_blot merges its tiles: a pixel claimed by several cuts takes
    the cut whose tile centre is nearest (registered position). One file per exposure. Returns per-cut own-mask records."""
    acc, recs = {}, {}
    for k in ts.keys:
        cs = [c for c in held if exps[c].tile == k]
        if not cs:
            continue
        ts.activate(k)
        for c in cs:
            e = exps[c]
            if pred is None:
                m, pv, S = predict_exposure(model, e, P, cfg)
                rec0 = chi2_exposure(e, m, pv, S=S, snr_min=5.0) | ({"closed_form_stripes": e.extra["closed_form_stripes"]} if "closed_form_stripes" in e.extra else {})
            else:
                m, pv, S, rec0 = pred(e)
            recs[f"{e.name}@tile{k}"] = rec0 | {"tile": int(k), "exposure": e.name}
            ny, nx = e.det_shape
            y0, x0 = e.cut0
            a, b = e.shape
            dist = torch.hypot(e.X + e.shift0[0], e.Y + e.shift0[1]).cpu().numpy()
            mm, vv = m.cpu().numpy(), pv.cpu().numpy()
            if e.eid not in acc:
                acc[e.eid] = {"model": np.full((ny, nx), np.nan, np.float32), "valid": np.zeros((ny, nx), bool),
                              "dist": np.full((ny, nx), np.inf, np.float32), "cut": e, "tiles": []}
            A = acc[e.eid]
            sub_v, sub_d = A["valid"][y0:y0 + a, x0:x0 + b], A["dist"][y0:y0 + a, x0:x0 + b]
            take = vv & (~sub_v | (dist < sub_d))
            A["model"][y0:y0 + a, x0:x0 + b][take] = mm[take]
            sub_d[take] = dist[take]
            sub_v |= vv
            A["tiles"].append(int(k))
            log(f"held-out {e.name} tile {k}: chi2/N (pipeline ERR, own mask, no offset) {recs[f'{e.name}@tile{k}']['chi2']:.3f} on "
                f"{recs[f'{e.name}@tile{k}']['n']} px; model S/N>5 {recs[f'{e.name}@tile{k}'].get('chi2_bright', float('nan')):.3f}")
    if pred_dir is not None:
        for eid, A in acc.items():
            e = A["cut"]
            full = A["model"]
            full[~A["valid"]] = np.nan
            meta = meta_fn(e) if meta_fn else {}
            meta["tiles"] = A["tiles"]
            meta["merge"] = "nearest tile centre (registered position), as stage3_blot merge_partials"
            np.savez_compressed(f"{pred_dir}/{e.name}.npz", model=full, model_valid=A["valid"], meta=json.dumps(meta))
    return recs
