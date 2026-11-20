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
            s = model.sky.sig_b[b]
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
        used = sorted({(int(model.e_det[i]), int(model.e_filt[i])) for i in range(model.E) if int(model.e_filt[i]) not in model.unseen})   # a pair with no training exposure is not fitted: neutral 1, never "at bound"
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


def batch_loss(model, exps, items, P, cfg, pool_mean_valid, detach_sky=False, half=None, jitter=None):
    """Importance-weighted mean loss per pixel of a batch of patches; the renders are batched (model.render_batch)."""
    tot, n, chi = 0.0, 0, 0.0
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


# ----------------------------------------------------------------------------- warm start of the kernel field (physics only as a start)
def warm_kernel(model, cfg, filt_info, log=print):
    """Fit the kernel field (and hbar) so that, with the per-exposure code at zero, l(v; h) matches a generic start profile per filter node:
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
            h = model.codes.hbar_at(model.piv[f].expand(lam.numel()), fq, iq)
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
    pools = {k: PatchPool(exps, ts.cut_ids(k, ("train",)), cfg) for k in ts.keys}
    gids = sorted({e.eid for e in exps if e.split == "train"})
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
                model.project_shift(g)
            t = model.affine[idt, 4:6] * model.aff_scale[4:6]
            shifts = {i: (float(t[k, 0]), float(t[k, 1])) for k, i in enumerate(gids)}
            for k in ts.keys:
                tr = [exps[c] for c in ts.cut_ids(k, ("train",))]
                if not tr:
                    continue
                if ts.multi:
                    ts.activate(k)
                img, wt = splat_coadd(tr, len(model.fset), ts.skies[k].N, ts.skies[k].pitch, model.dev, extra_shift=shifts)
                A0 = spectral_init(img, model.Phi_band, bands=[b for b in range(len(model.fset)) if b not in model.unseen] if model.unseen else None)
                ts.skies[k].reset_from_coeffs(A0)
        rms = float(t.pow(2).sum(1).mean().sqrt() * 1e3)
        out.append({"iter": it, "rms_translation_mas": rms, "seconds": time.perf_counter() - t0})
        log(f"registration pre-pass {it}: {cfg.reg_prepass_steps} steps x {sum(1 for k in ts.keys if pools[k].N)} tile(s), translation rms {rms:.1f} mas "
            f"({time.perf_counter() - t0:.1f} s); sky warm start rebuilt")
    return {"iters": out}


# ----------------------------------------------------------------------------- main fit
DILATION_WARN2 = 10.0        # second, louder warning level
DILATION_WARN = 2.0          # blur dilation above which a WARNING is logged (real fits: 1.0 to 1.4)
EM_AXES = ("k_exp", "g_exp")


def set_prior_pixels(cfg, exps, log=print):
    """--prior-pixels 0 = auto: the per-exposure prior weighs as much as one exposure's pixels against the per-pixel mean loss."""
    if cfg.prior_pixels <= 0:
        cfg.prior_pixels = float(np.mean([int(e.valid.sum()) for e in exps if e.split == "train"]))
        log(f"random-effect prior weight: 1 / {cfg.prior_pixels:.0f} per pixel (mean valid pixels per training exposure)")


@torch.no_grad()
def em_update(model, tr, cfg):
    """Type-II (parameter-expanded EM) update of the scales of the per-exposure RANDOM EFFECTS (kernel z_e, gain u_e; F21):
    tau <- tau x rms(effect over the training exposures), then the effects are divided by the same rms so tau x effect is unchanged.
    MAP effects without the posterior-variance term: unneeded effects shrink to the prior mean and their tau decays geometrically."""
    tr = torch.as_tensor(tr, device=model.dev)
    for ax, tens in (("k_exp", model.codes.z), ("g_exp", model.u)):
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


def sky_optimizer(sky, cfg):
    """Adam of one sky: the item sky's own groups (positions, log widths, angles, log coefficients)."""
    return torch.optim.Adam(sky.opt_groups(cfg))


def make_optimizers(model, cfg):
    op, nuis, taus, skyp = model.param_groups()
    sb = float(model.e_sigbar.mean())
    noise_ids = {id(model.hs), id(model.floor_raw)}
    op_main = [p for p in op if id(p) not in noise_ids]
    opt_op = torch.optim.Adam([{"params": op_main, "lr": cfg.lr_op, "base": cfg.lr_op},
                               {"params": [model.hs, model.floor_raw], "lr": 1e-2, "base": 1e-2}])
    opt_nuis = torch.optim.Adam([{"params": [nuis["bkg"]], "lr": cfg.lr_bkg * sb, "base": cfg.lr_bkg * sb},
                                 {"params": [nuis["affine"]], "lr": cfg.lr_affine, "base": cfg.lr_affine},
                                 {"params": model.exposure_effects(nuis), "lr": cfg.lr_codes, "base": cfg.lr_codes},
                                 {"params": [nuis["stripe_rows"], nuis["stripe_cols"]], "lr": 0.05 * sb, "base": 0.05 * sb}])
    opt_sky = sky_optimizer(model.sky, cfg)
    return opt_op, opt_nuis, opt_sky


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


def fit(model, exps, cfg, log=print, history=None, ts=None):
    """Joint fit. Returns a record dict (losses, tau trajectory, gauges, timings). `exps` is the list of cuts; `ts` the TileSet (default: one
    tile, the model's sky). Several tiles: a visit of --visit-steps steps on one tile (drawn with probability p_k), the operator, the
    nuisances and that tile's sky are updated; validation (tau) steps use the visited tile's validation cuts when it has some."""
    from .tiles import TileSet
    dev = model.dev
    ts = ts or TileSet.single(model, exps, cfg, log)
    kc_on = bool(getattr(model, "kc_on", False))
    if kc_on and ts.multi:
        raise SystemExit("--kernel-centroid-gauge 1 is implemented for one tile only (the sky shift that keeps the projection function-preserving "
                         "would have to move every tile's sky and respect each shift component)")
    rng = np.random.default_rng(cfg.seed)
    trng = np.random.default_rng(cfg.seed + 101)          # tile draws: a separate stream, so the single-tile fit is unchanged
    tr = sorted({e.eid for e in exps if e.split == "train"})      # global exposure ids
    pools = {k: PatchPool(exps, ts.cut_ids(k, ("train",)), cfg) for k in ts.keys}
    opt_op, opt_nuis, opt_sky0 = make_optimizers(model, cfg)
    ts.make_sky_opts(cfg)
    if not ts.multi:
        ts.opt_sky[ts.keys[0]] = opt_sky0          # identical to the single-box optimiser
    op, nuis, taus, _ = model.param_groups()
    rows_nuis = row_steps(model, cfg, [nuis["bkg"], nuis["stripe_rows"], nuis["stripe_cols"]])
    em_on = bool(taus)                      # the damped EM step of the scales of the two random effects (kernel code, gain)
    n_em_eval = 0
    k = ts.keys[0] if not ts.multi else ts.draw(trng)
    ts.activate(k)
    nb = pools[k].refresh(model)
    log(f"fit: {len(tr)} training exposures, {sum(p.N for p in pools.values())} patches ({nb} bright in tile {k}), "
        f"mean valid px/patch {np.mean([p.mean_valid for p in pools.values() if p.N]):.0f}; tau scales {list(model.tau.raw.keys())}"
        + (f"; {len(ts.keys)} tiles, p = {dict(zip(ts.keys, np.round(ts.p, 3).tolist()))}, visit {cfg.visit_steps} steps, "
           f"{len(ts.shift_groups)} shift component(s), {len(ts.pedestal_groups)} pedestal group(s)" if ts.multi else ""))
    rec = {"steps": [], "tau": []}
    tr_by_f = {f: v.to(dev) for f, v in ts.pedestal_groups.items()}
    set_prior_pixels(cfg, exps, log)
    t0 = time.perf_counter()
    budget = cfg.budget_min * 60.0
    step = 0
    visit_left = cfg.visit_steps
    nf_max = float(getattr(cfg, "noise_floor_max", 0.0) or 0.0)
    nf_raw_max = float(inv_softplus(torch.tensor(nf_max))) if nf_max > 0 else 0.0
    nf_warn_step = None
    dil_max = 1.0
    nf_first = torch.full((), -1, dtype=torch.long, device=dev)
    hs_max = noise_scale_raw_max(float(getattr(cfg, "noise_scale_max", 0.0) or 0.0))
    bound_first = None
    if nf_max > 0:
        with torch.no_grad():
            model.floor_raw.clamp_(max=nf_raw_max)
    if hs_max is not None:
        with torch.no_grad():
            model.hs.clamp_(-hs_max, hs_max)
    while True:
        prog = max(step / max(cfg.steps, 1), (time.perf_counter() - t0) / max(budget, 1e-9))
        if prog >= 1.0:
            break
        if ts.multi and visit_left <= 0:
            k = ts.draw(trng)
            if ts.activate(k):
                pools[k].refresh(model)
            ts.stats[k]["visits"] += 1
            visit_left = cfg.visit_steps
        visit_left -= 1
        ts.stats[k]["steps"] += 1
        pool, opt_sky = pools[k], ts.opt_sky[k]
        set_lr([opt_op, opt_nuis, opt_sky], lr_factor(prog))
        skyp = list(model.sky.parameters())
        params = op + list(nuis.values()) + skyp
        # the operator tables P do not depend on the sky: one evaluation per step serves the sky inner steps (detached copy) and the joint step
        with only_grad(model, params):
            P = model.prepare()
        P_sky = {kk: (v.detach() if torch.is_tensor(v) else v) for kk, v in P.items()}
        if P.get("dil") is not None:       # blur dilation guard (build C): observation, and with --dilation-stop a clean end of the fit; the rule itself is unchanged
            dn = float(P["dil"])
            dil_max = max(dil_max, dn)
            if dn > DILATION_WARN and "dilation_first_above" not in rec:
                rec["dilation_first_above"] = step
                log(f"WARNING: blur dilation {dn:.3g} at step {step} exceeds {DILATION_WARN:g} (real fits stay at 1.0 to 1.4): the kernel is "
                    f"being learned sharper than one sky node allows and the dilation rule is compensating; the fit may be diverging")
            if dn > DILATION_WARN2 and "dilation_first_above_10" not in rec:
                rec["dilation_first_above_10"] = step
                log(f"WARNING: blur dilation {dn:.3g} at step {step} exceeds {DILATION_WARN2:g}: the common-blur dilation has run away (the kernel "
                    f"network's gradient is then zero); results from this fit are not trustworthy")
            dstop = float(getattr(cfg, "dilation_stop", 0.0) or 0.0)
            if dstop > 0 and dn > dstop:
                rec["dilation_stopped_at"] = step
                log(f"STOP: blur dilation {dn:.3g} at step {step} exceeds --dilation-stop {dstop:g}; the fit ends here (steps 0 to {step - 1} done) and the "
                    f"held-out stage, if any, uses the state as it is. Rerun with a lower kernel learning rate or a larger --blur-floor, or check the data.")
                break
        # (a) sky inner steps: sky only, operator fixed
        for _ in range(cfg.inner_sky):
            with only_grad(model, skyp):
                items = pool.draw(cfg.batch_patches, rng)
                loss, _, _ = batch_loss(model, exps, items, P_sky, cfg, pool.mean_valid)
                opt_sky.zero_grad()
                loss.backward()
                opt_sky.step()
        # (b) joint step: operator + nuisances (+ sky unless cross-fitting, where the operator reads the other fold's sky, detached)
        with only_grad(model, params):
            items = pool.draw(cfg.batch_patches, rng)
            model._chi_in = 0.0
            loss, npx, chi = batch_loss(model, exps, items, P, cfg, pool.mean_valid)
            eids = batch_eids(exps, items, dev)
            pri = model.priors(eids) / cfg.prior_pixels
            tot = loss + pri
            opt_op.zero_grad()
            opt_nuis.zero_grad()
            opt_sky.zero_grad()
            tot.backward()
            torch.nn.utils.clip_grad_norm_(op, cfg.clip)
            opt_op.step()
            if nf_max > 0:                      # projection onto softplus(floor_raw) <= nf_max; Adam moments untouched (as the gauge projections)
                with torch.no_grad():
                    model.floor_raw.clamp_(max=nf_raw_max)
            if hs_max is not None:              # same for the learned noise scale exp(h) of every (detector, filter): within [1/value, value]
                with torch.no_grad():
                    model.hs.clamp_(-hs_max, hs_max)
            step_rows(opt_nuis, list(nuis.values()), eids, rows=rows_nuis)
            model.project_gauges(tr_by_f if cfg.pedestal_gauge else None, eids)
            for g in ts.shift_groups:
                model.project_shift(g)
            opt_sky.step()
        if kc_on:                               # kernel-centroid gauge (F43): after the whole joint step, function-preserving
            model.project_kernel_centroid()
        if em_on and prog >= cfg.tau_start and step % cfg.val_every == 0:      # no data, no gradient: the cadence is --val-every x --tau-accum steps from --tau-start
            n_em_eval += 1
            if n_em_eval % cfg.tau_accum == 0:
                em_update(model, tr, cfg)
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
            dil_now = 1.0 if dl is None else float(dl)       # blur dilation guard (build C): observation only, nothing is changed
            dil_max = max(dil_max, dil_now)
            mem = (f" dil {float(dl):.3f}" if dl is not None else " dil 1") + f" Mref {float(model.blur_moment(1.0 if dl is None else float(dl))) * 1e6:.0f}e-6 floor {model.blur_floor * 1e6:.0f}e-6"
            mem += (f" cuda peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB" if dev.type == "cuda" else "")
            if ts.multi:
                mem += f" tile {k} resident {ts.resident_gb():.2f} GB"
            if kc_on:
                kr = model.kernel_centroid_report()
                mem += (f" kcen held {kr['held_moment_after_last_projection_mas']:.1e} mas step {max(c['last_step_mas'] for c in kr['classes']):.2e} "
                        f"(max {kr['step_max_mas']:.2e}) mas sky-shift total [{kr['sky_shift_total_mas'][0]:.2f}, {kr['sky_shift_total_mas'][1]:.2f}] mas")
            chi_in = float(model._chi_in) / max(npx, 1)
            log(f"step {step:6d} t {el:7.1f}s prog {prog:.3f} loss {float(loss):.4f} chi2/px(batch) {chi / max(npx, 1):.3f} "
                f"chi2/px(input noise) {chi_in:.3f} noise-floor {noise_floor_values(model)} "
                f"gauge-shift [{float(gsh[0]) * 1e3:.2f}, {float(gsh[1]) * 1e3:.2f}] mas{mem} tau {tt}")
        if history is not None:
            history(step, model)
        step += 1
    rec["n_steps"] = step
    with torch.no_grad():
        dl_end = model.dilation()
    dil_end = 1.0 if dl_end is None else float(dl_end)
    dil_max = max(dil_max, dil_end)
    rec["dilation_max"], rec["dilation_final"] = dil_max, dil_end       # maximum over the logged steps and the end of the fit
    if dil_max > DILATION_WARN:
        log(f"WARNING: blur dilation reached {dil_max:.3g} (above {DILATION_WARN:g}; final {dil_end:.3g}); real fits stay at 1.0 to 1.4")
    if nf_warn_step is None and int(nf_first) >= 0:
        nf_warn_step = rec["noise_floor_first_above"] = int(nf_first)
        log(f"WARNING: learned fractional noise floor exceeds {NOISE_FLOOR_WARN} first at step {nf_warn_step}")
    rec["noise_floor_final"] = noise_floor_values(model)
    if kc_on:
        rec["kernel_centroid_gauge"] = model.kernel_centroid_report()
        kr = rec["kernel_centroid_gauge"]
        log("kernel-centroid gauge: " + "; ".join(f"{c['filter']}/{c['instrument']} offset [{c['offset_mas'][0]:.2f}, {c['offset_mas'][1]:.2f}] mas "
                                                    f"(window {c['window_sigma_arcsec']:.3f} arcsec)" for c in kr["classes"])
            + f"; {kr['projections']} projections, class step max {kr['step_max_mas']:.3e} rms {kr['step_rms_mas']:.3e} mas, held moment after the "
              f"last projection {kr['held_moment_after_last_projection_mas']:.1e} mas, total sky shift [{kr['sky_shift_total_mas'][0]:.3f}, "
              f"{kr['sky_shift_total_mas'][1]:.3f}] mas")
    if nf_max > 0 or hs_max is not None:
        br = bound_report(model, cfg)
        rec["noise_at_bound_final"] = br
        if br["floor"] or br["noise_scale"]:
            log(f"WARNING: at the end of the fit a learned noise term sits at its bound: noise floor {br['floor']} (bound {nf_max}), noise scale "
                f"exp(h) {br['noise_scale']} (bound x{getattr(cfg, 'noise_scale_max', 0.0)} and 1/x)")
    if max(rec["noise_floor_final"]) > NOISE_FLOOR_WARN:
        log(f"WARNING: final learned fractional noise floor {rec['noise_floor_final']} exceeds {NOISE_FLOOR_WARN}")
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
    """Only per-exposure nuisances (background, affine, gain u, kernel code z, stripes) of the held-out exposures are
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
                            {"params": model.exposure_effects(nuis), "lr": cfg.lr_codes},
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
    indices, the rule of the half split in pix_loss)."""
    ny, nx = e.shape
    rr = torch.arange(ny, device=e.d.device) + e.cut0[0]
    cc = torch.arange(nx, device=e.d.device) + e.cut0[1]
    return ((rr // 4)[:, None] + (cc // 4)[None, :]) % 2


NUIS_KEYS = ("bkg", "affine", "u", "z", "stripe_rows", "stripe_cols")


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
        outs = model.render_batch([(e.eid, px) for px in pixs], P)
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
    if model.unseen:     # these pairs were never fitted: exp(h) = 1 by construction (neutral), not a measurement
        out["noise_scale_not_fitted_no_training_exposure"] = [k for k, (d, f) in zip(out["noise_scale_exp_h"], used) if f in model.unseen]
        out["filters_without_training_exposures"] = [fn[f] for f in model.unseen]
    for s in ("train", "held"):
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
             "code_z": st(model.codes.z[i])}
        if model.cfg.stripes and bool(model.e_stripes[i].any()):
            rw = (model.stripe_rows[i] * model.row_mask[i]).double() / sb[i, None]
            cl = (model.stripe_cols[i] * model.col_mask[i]).double() / sb[i, None]
            r["stripes_rows_over_sigma"] = {"rms": float((rw.pow(2).sum() / model.row_mask[i].sum().clamp(min=1)).sqrt()), "max_abs": float(rw.abs().max())}
            r["stripes_cols_over_sigma"] = {"rms": float((cl.pow(2).sum() / model.col_mask[i].sum().clamp(min=1)).sqrt()), "max_abs": float(cl.abs().max())}
        out[s] = r
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
