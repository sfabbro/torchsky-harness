"""The field operator: everything between the sky s and the detector pixel d (equations in DESIGN.md, approximations F1... in __init__).

  d_e(xi) = (1/Q) sum_q exp(gamma_e(lambda_q)) c_q(xi) + B_e(xi) + n_e(xi)
  c_q(xi) = int d^2u k_e(u; lambda_q) s(Omega_e(xi) + u, lambda_q)          (render.py)
  Omega_e(xi) = x'(xi) + M_e x'(xi) + t_e,    x' = X_wcs(xi) + shift0_e
  gamma_e(lambda) = gbar(lambda, filter input, instrument) + tau_ge u_e  -  m(lambda)                    (m = gauge projection, F17)
  k_e: kernel code h = hbar(lambda_pivot, filter input, instrument) + tau_ke z_e   (kernel.py)
  B_e(xi) = sum_6 b_ej P_j(xi) + rows_e[row] + cols_e[col] (stripes, where the instrument declares them)
  sigma^2 = (sigma_pipe exp(h_sigma[det, filter]))^2 + (f_floor[channel] |S|)^2
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import Codes, KernelNet, LogPolarEncoding, Tau, burn_reference_init, inv_softplus, lam_feats, mlp
from .render import Levels, kernel_level_values, kernel_level_values_grouped, pooled_level, render_fft_batch

LATTICE = torch.tensor([[x, y] for y in (-0.8, 0.0, 0.8) for x in (-0.8, 0.0, 0.8)], dtype=torch.float32)


def legendre6(xi):
    x, y = xi[..., 0], xi[..., 1]
    return torch.stack([torch.ones_like(x), x, y, 1.5 * x * x - 0.5, x * y, 1.5 * y * y - 0.5], -1)


def resolve_instrument_options(cfg, n_inst, log=None):
    """Audit finding 5. --gauge-ref-inst defaults to None (unset): with one instrument it resolves to -1 (every training exposure); with more than one
    instrument the reference instrument of the throughput gauge must be given explicitly (SystemExit otherwise). Idempotent."""
    if getattr(cfg, "gauge_ref_inst", None) is None:
        if n_inst > 1:
            raise SystemExit(f"{n_inst} instruments are loaded: --gauge-ref-inst INDEX must be given explicitly (the absolute flux scale of the "
                             f"fit is the reference instrument's; -1 = all training exposures, weighted by their number, is a compromise)")
        cfg.gauge_ref_inst = -1


class FieldOperator(nn.Module):
    def __init__(self, cfg, fset, meta, sky, dev):
        """meta: dict of per-exposure lists/arrays: filt, inst, chan, det, train(bool), R (E,2,2), sigbar, det_shape (E,2), shift0 (E,2),
        stripes (bool), plus inst_info [{name, diameter_m}], chan_pixel_arcsec, n_det, n_inst, n_chan, det_names."""
        super().__init__()
        self.cfg, self.fset, self.dev = cfg, fset, dev
        E = len(meta["filt"])
        self.E = E
        t = lambda x, dt=torch.long: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
        self.register_buffer("e_filt", t(meta["filt"]))
        self.register_buffer("e_inst", t(meta["inst"]))
        self.register_buffer("e_chan", t(meta["chan"]))
        self.register_buffer("e_det", t(meta["det"]))
        self.register_buffer("e_R", t(meta["R"], torch.float32))
        self.register_buffer("e_sigbar", t(meta["sigbar"], torch.float32))
        self.register_buffer("e_shape", t(meta["det_shape"]))
        self.register_buffer("shift0", t(meta["shift0"], torch.float32))
        # several tiles: the affine of exposure e acts about its own reference point r_e (field frame; centroid of its pixels over all its
        # cuts), so the lever arm of M_e is the extent of the exposure, not the distance to the field reference (F32)
        self.has_aff_ref = meta.get("aff_ref") is not None
        # a plain device tensor, not a registered buffer: with one tile nothing is allocated, which keeps the single-box path bit-identical
        # (an extra allocation alone shifts float32 rounding by ~1e-6, which Adam then amplifies)
        self.aff_ref = t(meta["aff_ref"], torch.float32) if self.has_aff_ref else None
        self.train_mask = np.asarray(meta["train"], bool)
        self.l_filt, self.l_chan = [int(x) for x in meta["filt"]], [int(x) for x in meta["chan"]]   # host copies: no device syncs in hot paths
        self.l_shape = [tuple(int(v) for v in x) for x in meta["det_shape"]]
        self.meta = meta
        lam, w, curves, piv = fset.tensors(dev)
        self.register_buffer("lam", lam)          # (nf, Q)
        self.register_buffer("wq", w)
        self.register_buffer("piv", piv)
        self.sky = sky
        Phi = sky.basis(lam.flatten()).view(len(fset), fset.Q, -1)
        self.register_buffer("Phi", Phi)          # (nf, Q, K)
        self.register_buffer("Phi_band", (Phi * w[..., None]).sum(1))
        n_det, n_inst, n_chan = meta["n_det"], meta["n_inst"], meta["n_chan"]
        resolve_instrument_options(cfg, n_inst)           # no-op when the command line (or a test) already did it
        self.n_det, self.n_inst, self.n_chan = n_det, n_inst, n_chan
        # render pitch per channel: --render-pitch, or (0) the largest sky_pitch / k at or below (channel pixel) / --render-os-pix
        self.levels_ch, self.os_ch, self.render_pitch_ch = [], [], []
        for c in range(n_chan):
            if cfg.render_pitch > 0:
                pr = cfg.render_pitch
            else:
                k = max(1, int(math.ceil(sky.pitch / (meta["chan_pix"][c] / cfg.render_os_pix) - 1e-9)))
                pr = sky.pitch / k
            os_ = int(round(sky.pitch / pr))
            if abs(sky.pitch / pr - os_) > 1e-6:
                raise ValueError("--render-pitch must divide the sky pitch")
            self.levels_ch.append(Levels(pr, cfg.level_factor, cfg.level_nh, cfg.kernel_extent, a=cfg.level_overlap, device=dev,
                                         norm_r=float(getattr(cfg, "kernel_norm_radius", 0.0) or 0.0)))
            self.os_ch.append(os_)
            self.render_pitch_ch.append(pr)
        self.levels = self.levels_ch[int(np.argmin(self.render_pitch_ch))]   # finest (warm start, diagnostics)
        # ---- kernel field + codes
        r0 = cfg.enc_r0 if cfg.enc_r0 > 0 else 0.5 * min(self.render_pitch_ch)
        enc = LogPolarEncoding(r0=r0, r_max=max(cfg.kernel_extent, 1.0), n_rad=cfg.enc_nrad, m_max=cfg.enc_mmax)
        self.net = KernelNet(enc, dh=cfg.dh, hidden=cfg.k_hidden, layers=cfg.k_layers)
        self.codes = Codes(len(fset), n_inst, n_det, E, curves, cfg.filter_input, dh=cfg.dh)
        # ---- throughput (log) relative to T/(lambda N)
        self.gbar = mlp(9 + 16 + 4, 32, 1, 2, zero_last=True)
        self.g_fin = nn.Linear(curves.shape[1], 16) if cfg.filter_input == "curve" else nn.Embedding(len(fset), 16)
        self.g_inst = nn.Embedding(n_inst, 4)
        burn_reference_init(nn.Embedding(n_det, 8), mlp(8 + 10 + 9, 32, 1, 2), mlp(5 + 9 + 4, 32, 1, 2))     # throughput across the detector and with time (deleted)
        self.u = nn.Parameter(torch.zeros(E, device=dev))
        # ---- geometry
        self.affine = nn.Parameter(torch.zeros(E, 6, device=dev))   # raw; physical = raw * aff_scale (matrix 1e-3, shift 0.01 arcsec per unit)
        self.register_buffer("aff_scale", torch.tensor([1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-2], device=dev))
        burn_reference_init(nn.Embedding(n_det, 8), mlp(8 + 10 + 9, 32, 2, 2), mlp(8 + 10 + 5, 32, 2, 2))   # static and drifting distortion (deleted)
        # ---- detector
        self.bkg = nn.Parameter(torch.zeros(E, 6, device=dev))
        with torch.no_grad():
            self.bkg[:, 0] = torch.as_tensor(np.asarray(meta["bkg0"]), dtype=torch.float32, device=dev)
        self.stripe_rows = nn.Parameter(torch.zeros(E, 2048, device=dev))
        self.stripe_cols = nn.Parameter(torch.zeros(E, 2048, device=dev))
        self.register_buffer("e_stripes", t(meta["stripes"], torch.bool))
        self.e_stripes_host = [bool(x) for x in np.asarray(meta["stripes"]).reshape(-1)]     # host copy: the branch below needs no device read
        self._et = {}
        self.hs = nn.Parameter(torch.zeros(n_det, len(fset), device=dev))
        self.floor_raw = nn.Parameter(inv_softplus(torch.full((n_chan,), cfg.floor0)).to(dev))
        burn_reference_init(nn.Embedding(n_chan, 4))                                  # channel code of the non-linearity (deleted)
        # ---- scales of the per-exposure random effects
        self.tau = Tau(self.tau_spec(cfg))
        # ---- reference sets (training exposures)
        tr = np.nonzero(self.train_mask)[0]
        self.ref_eids = torch.as_tensor(tr, device=dev, dtype=torch.long)
        # filters of the filter set with no training exposure (--train-exclude-filters, unseen-band test): plain host list, [] in every ordinary field
        self.unseen = sorted(set(range(len(fset))) - {int(meta["filt"][i]) for i in tr})
        gref = tr if cfg.gauge_ref_inst < 0 else np.array([i for i in tr if int(meta["inst"][i]) == cfg.gauge_ref_inst])
        self.register_buffer("gref", torch.as_tensor(gref, device=dev, dtype=torch.long))
        Tt = fset.T_table(lam.flatten().cpu().numpy())        # (nf, nf*Q)
        self.register_buffer("T_ref", torch.as_tensor(Tt, dtype=torch.float32, device=dev))
        self.amp = cfg.amplitude
        # ---- pedestal / stripe gauges (F18): per-exposure reference levels and the detector rows / columns each exposure uses
        self.register_buffer("bkg0", t(meta["bkg0"], torch.float32))
        rm = torch.zeros(E, 2048, dtype=torch.bool, device=dev)
        cm = torch.zeros(E, 2048, dtype=torch.bool, device=dev)
        cuts = meta.get("cuts") or [[c] for c in meta.get("cut", [(0, 0, 2048, 2048)] * E)]
        for i, cs in enumerate(cuts):          # every cut of the exposure (several tiles): the union of the rows / columns used
            for (r0, c0, nr, nc) in cs:
                rm[i, r0:r0 + nr] = True
                cm[i, c0:c0 + nc] = True
        self.register_buffer("row_mask", rm)
        self.register_buffer("col_mask", cm)
        # ---- blur gauge (F18): the sharpest kernel class (smallest generic width) may not be sharper than one sky tent
        fpix, fdiam = {}, {}
        dm = meta.get("diameter_m_e")
        for i in range(E):
            fpix.setdefault(int(meta["filt"][i]), meta["chan_pix"][int(meta["chan"][i])])
            fdiam.setdefault(int(meta["filt"][i]), float(dm[i]) if dm is not None else meta.get("diameter_m", 6.5))
        # the sharpest class is chosen among the filters that have training exposures: a filter without any has a kernel set by the warm start only
        sig2 = {f: (0.42 * float(self.piv[f]) * 1e-6 / fdiam[f] * 206264.8) ** 2 + p_ ** 2 / 12 for f, p_ in fpix.items() if f not in self.unseen}
        self.blur_sig2 = sig2
        self.blur_ref = min(sig2, key=sig2.get)
        self.blur_floor = float(cfg.blur_floor) * sky.pitch ** 2 / 3.0          # m2 of one sky tent (2 x pitch^2 / 6)
        Rw = 2.0 * math.sqrt(max(self.blur_floor, 1e-8))
        pm = min(self.render_pitch_ch)
        nh = int(math.ceil(4 * Rw / pm))
        g = torch.arange(-nh, nh + 1, device=dev, dtype=torch.float32) * pm
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        self.register_buffer("blur_u", torch.stack([xx.flatten(), yy.flatten()], -1), persistent=False)
        r2 = (self.blur_u ** 2).sum(-1)
        self.register_buffer("blur_r2", r2, persistent=False)
        self.register_buffer("blur_logw", -0.5 * r2 / Rw ** 2, persistent=False)

    # ------------------------------------------------------------------ setup helpers
    @staticmethod
    def tau_spec(cfg):
        glob = "learned" if cfg.tau_mode == "learned" else "on"

        def km(mode):
            return "on" if (mode == "learned" and cfg.tau_mode == "fixed") else mode
        spec = {"k_exp": (km(cfg.k_exp), cfg.tau0_code)}
        if cfg.amplitude == "structured":
            spec["g_exp"] = (glob, cfg.tau0_g)
        return spec

    def g_filt(self, f_idx):
        if self.cfg.filter_input == "curve":
            return self.g_fin(self.codes.curves[f_idx])
        return self.g_fin(f_idx)

    # ------------------------------------------------------------------ blur gauge
    def mean_codes(self, f):
        """Mean kernel code of filter f (the code at its pivot wavelength, z = 0) at every lambda node: (Q, dh)."""
        f = torch.as_tensor(f, device=self.dev)
        Q = self.fset.Q
        ins = self.e_inst[(self.e_filt == f).nonzero()[0, 0]]
        fq, iq = f.expand(Q), ins.expand(Q)
        return self.codes.hbar_at(self.piv[f].expand(Q), fq, iq)

    def blur_moment(self, dil=1.0, f=None):
        """Core second moment (Gaussian window of width 2 sqrt(floor)) of the mean kernel of filter f (default: the blur reference), averaged
        over its lambda nodes, after the common dilation dil [arcsec^2]."""
        h = self.mean_codes(self.blur_ref if f is None else f)
        lg = self.net(self.net.enc(self.blur_u / dil), h) + self.blur_logw[None]
        w = torch.softmax(lg, -1)
        return (w * self.blur_r2[None]).sum(-1).mean()

    def dilation(self):
        """Hard reparametrisation of the blur gauge: every kernel is evaluated at v / s with s = sqrt(max(1, floor / M_ref)), two fixed-point
        passes, so the mean core second moment of the sharpest kernel class is >= one sky tent (F18). s = 1 when the floor is inactive."""
        if self.cfg.blur_floor <= 0:
            return None
        with torch.no_grad():
            s1 = math.sqrt(max(1.0, self.blur_floor / float(self.blur_moment(1.0))))
        if s1 == 1.0:
            return None
        M = self.blur_moment(s1)
        return s1 * torch.sqrt(torch.clamp(self.blur_floor / M, min=1.0))

    # ------------------------------------------------------------------ per-step preparation (normalisations, gauge)
    def prepare(self):
        P = {}
        d = self.dilation()
        if d is not None:
            P["dil"] = d
        if self.amp == "structured":
            P["m"] = self.gauge_mean(P)                                         # (nf, Q)
        return P

    def gbar_at(self, lam, f_idx, inst_idx):
        return self.gbar(torch.cat([lam_feats(lam), self.g_filt(f_idx), self.g_inst(inst_idx)], -1))[..., 0]

    def gamma_raw_ref(self, P):
        """gamma_raw of every gauge-reference exposure, averaged over the 3x3 lattice, at every Lambda node: (n_ref, nL)."""
        lamf = self.lam.flatten()
        e = self.gref
        nL = lamf.numel()
        f, ins = self.e_filt[e], self.e_inst[e]
        g = self.gbar_at(lamf[None].expand(e.numel(), -1), f[:, None].expand(-1, nL), ins[:, None].expand(-1, nL))
        if self.tau("g_exp") is not None:
            g = g + self.tau("g_exp") * self.u[e][:, None]
        return g

    def gauge_mean(self, P):
        """m(lambda) = T-weighted mean of gamma_raw over the reference set at each Lambda node (F17): subtracting it makes the mean
        throughput of the reference set equal to the published curves at every node."""
        g = self.gamma_raw_ref(P)                                    # (n_ref, nL)
        Tw = self.T_ref[self.e_filt[self.gref]]                      # (n_ref, nL)
        den = Tw.sum(0)
        m = torch.where(den > 0, (Tw * g).sum(0) / den.clamp(min=1e-12), torch.zeros_like(den))
        m = m.view(len(self.fset), self.fset.Q)
        if self.unseen:
            # Nodes of a filter with no training exposure: the reference set barely covers them (den small or zero, m then 0 or noise), but the
            # held-out exposures of that filter are predicted there. Use the m of the nearest node (in wavelength) of the filters that have training
            # data, so that the convention stays continuous in wavelength. Nodes of the other filters are untouched.
            lamf = self.lam.flatten()
            seen = torch.as_tensor([f for f in range(len(self.fset)) if f not in self.unseen], device=m.device)
            seen_nodes = (seen[:, None] * self.fset.Q + torch.arange(self.fset.Q, device=m.device)[None]).flatten()
            mf = m.flatten().clone()
            for f in self.unseen:
                for q in range(self.fset.Q):
                    n = f * self.fset.Q + q
                    j = seen_nodes[(lamf[seen_nodes] - lamf[n]).abs().argmin()]
                    mf[n] = m.flatten()[j]
            m = mf.view(len(self.fset), self.fset.Q)
        return m

    # ------------------------------------------------------------------ components at points of one exposure
    def codes_at(self, e, xi, P):
        """Kernel codes for exposure e at normalised positions xi (M, 2): (Q, M, dh)."""
        f, ins = self.e_filt[e], self.e_inst[e]
        Q, M = self.fset.Q, xi.shape[0]
        fq = f.expand(Q)
        iq = ins.expand(Q)
        h = self.codes.hbar_at(self.piv[f].expand(Q), fq, iq)                # (Q, dh) pivot code
        h = h[:, None].expand(Q, M, -1)
        if self.tau("k_exp") is not None:
            h = h + self.tau("k_exp") * self.codes.z[e][None, None]
        return h

    def gamma_at(self, e, xi, P):
        """log throughput relative to T/(lambda N): (Q, M)."""
        f, ins = self.e_filt[e], self.e_inst[e]
        Q, M = self.fset.Q, xi.shape[0]
        lam = self.lam[f]
        if self.amp != "structured":
            return torch.zeros(Q, M, device=xi.device)
        g = self.gbar_at(lam, f.expand(Q), ins.expand(Q))[:, None].expand(Q, M)
        if self.tau("g_exp") is not None:
            g = g + self.tau("g_exp") * self.u[e]
        return g - P["m"][f][:, None]

    def positions(self, e, X, Y, xi, P, c0=None):
        """Omega_e(xi, lambda_q): (Q, M, 2) [arcsec], in the frame of the tile whose header positions X, Y are given. With several tiles the
        per-exposure affine acts about the exposure's reference point r_e in the FIELD frame: Omega = x + M_e (x + c0 - r_e) + t_e, c0 = offset
        of the tile centre from the field reference (F32), so one (M_e, t_e) per exposure is consistent across tiles and the lever arm of M_e
        is the exposure's own extent. One tile (no r_e, c0 = None or (0, 0)): about the box centre."""
        Q = self.fset.Q
        x = torch.stack([X, Y], -1)[None].expand(Q, -1, -1) + self.shift0[e]
        a = self.affine[e] * self.aff_scale
        xm = x
        if c0 is not None and (c0[0] != 0.0 or c0[1] != 0.0):
            xm = x + torch.tensor([float(c0[0]), float(c0[1])], device=x.device, dtype=x.dtype)
        if self.has_aff_ref:
            xm = xm - self.aff_ref[e]
        Mx = torch.stack([a[0] * xm[..., 0] + a[1] * xm[..., 1], a[2] * xm[..., 0] + a[3] * xm[..., 1]], -1)
        return x + Mx + a[4:6]

    def et(self, e):
        """Device scalar index tensor of exposure e (an int), made once: no host-to-device copy per patch and step."""
        if not isinstance(e, (int, np.integer)):
            return torch.as_tensor(e, device=self.dev)
        t = self._et.get(int(e))
        if t is None:
            t = self._et[int(e)] = torch.as_tensor(int(e), device=self.dev)
        return t

    def background(self, e, xi, row, col, e_host=None):
        """Quadratic per-exposure background (sky / zodiacal light) + the row and column stripes of the instruments that declare them."""
        B = (legendre6(xi) * self.bkg[e]).sum(-1)
        if self.cfg.stripes and (self.e_stripes_host[e_host] if e_host is not None else bool(self.e_stripes[e])):
            B = B + self.stripe_rows[e, row] + self.stripe_cols[e, col]
        return B

    def sigma(self, e, S, sig_pipe):
        h = 3 * torch.tanh(self.hs[self.e_det[e], self.e_filt[e]] / 3)
        fl = F.softplus(self.floor_raw[self.e_chan[e]])
        return torch.sqrt((sig_pipe * torch.exp(h)) ** 2 + (fl * S) ** 2)

    def levels_of(self, e):
        return self.levels_ch[self.l_chan[int(e)] if isinstance(e, int) else int(self.e_chan[e])]

    # ------------------------------------------------------------------ kernels at region nodes
    def node_kernels(self, e, xi_nodes, P):
        """Discrete level kernels for the region nodes of exposure e: list over levels of (n, Q, kh, kw)."""
        h = self.codes_at(e, xi_nodes, P)                        # (Q, n, dh)
        Q, n, dh = h.shape
        Ks, logZ = kernel_level_values(self.net, self.levels_of(e), self.e_R[e], h.permute(1, 0, 2).reshape(n * Q, dh), amp_free=(self.amp == "free"),
                                       ckpt=bool(self.cfg.ckpt_kernel), dil=P.get("dil"))
        return [K.view(n, Q, *K.shape[-2:]) for K in Ks], logZ.view(n, Q)

    # ------------------------------------------------------------------ the forward model on a set of pixels of one exposure
    def finish(self, e_t, pix, P, c, f=None, e_host=None):
        f = int(self.e_filt[e_t]) if f is None else f
        gam = self.gamma_at(e_t, pix["xi"], P)
        amp = torch.exp(gam)
        S = (self.wq[f][:, None] * amp * c).sum(0)
        m = S + self.background(e_t, pix["xi"], pix["row"], pix["col"], e_host=e_host)
        return {"S": S, "m": m}

    def render_batch(self, items, P, view=None, detach_sky=False, jitter=False, region_px=None):
        """Batched overlap-add FFT render of several patches: items = list of (eid, pix). Patches are grouped by channel (same level grids);
        one fixed window per level and group, one grouped kernel evaluation, one batched FFT per level. Returns a list of dicts S, m."""
        A = self.sky.coeffs(view)
        if detach_sky:
            A = A.detach()
        region = region_px or self.cfg.region_px
        outs = [None] * len(items)
        groups = {}
        for k, (e, pix) in enumerate(items):
            groups.setdefault(self.l_chan[e], []).append(k)
        sh = (torch.rand(2) - 0.5) if jitter else None                       # host tensor: no device sync when read
        Q = self.fset.Q
        for ch, ks in groups.items():
            lv = self.levels_ch[ch]
            poss, wts, codes, Rs, Phis, nps = [], [], [], [], [], []
            for k in ks:
                e, pix = items[k]
                e_t = self.et(e)
                pos = self.positions(e_t, pix["X"], pix["Y"], pix["xi"], P, c0=pix.get("c0"))
                ny, nx = self.l_shape[e]
                nodes, w = region_nodes(pix["xi_px"], (ny, nx), region, bounds=pix.get("bounds"))
                xin = torch.stack([nodes[:, 0] / (nx - 1) * 2 - 1, nodes[:, 1] / (ny - 1) * 2 - 1], -1).float()
                codes.append(self.codes_at(e_t, xin, P).permute(1, 0, 2))      # (n, Q, dh)
                poss.append(pos)
                wts.append(w)
                Rs.append(self.e_R[e_t])
                Phis.append(self.Phi[self.l_filt[e]])
                nps.append(pos.shape[1])
            B, nmax, Npm = len(ks), max(c.shape[0] for c in codes), max(nps)
            dh = codes[0].shape[-1]
            Cb = torch.stack([torch.cat([c, c[:1].expand(nmax - c.shape[0], -1, -1)]) for c in codes])          # (B, nmax, Q, dh)
            Wb = torch.stack([torch.nn.functional.pad(w, (0, Npm - w.shape[1], 0, nmax - w.shape[0])) for w in wts])  # (B, nmax, Npm)
            Pb = torch.stack([torch.cat([p_, p_[:, -1:].expand(-1, Npm - p_.shape[1], -1)], 1) for p_ in poss])        # (B, Q, Npm, 2)
            det_frame = tuple(l for l in range(lv.L + 1) if not pooled_level(lv, self.sky.pitch, l)) if self.cfg.render_frame == "detector" else ()
            Rb = torch.stack(Rs)
            Ks, _ = kernel_level_values_grouped(self.net, lv, Rb, Cb.reshape(B, nmax * Q, dh), amp_free=(self.amp == "free"),
                                                shift0=sh, ckpt=bool(self.cfg.ckpt_kernel), dil=P.get("dil"), det_frame=det_frame)
            Ks = [K.view(B, nmax, Q, *K.shape[-2:]) for K in Ks]
            c = render_fft_batch(self.sky, A, torch.stack(Phis), lv, Ks, Wb, Pb, shift0=sh, Rs=Rb, det_frame=det_frame)
            for b, k in enumerate(ks):
                e, pix = items[k]
                outs[k] = self.finish(self.et(e), pix, P, c[b, :, :nps[b]], f=self.l_filt[e], e_host=e)
        return outs

    @torch.no_grad()
    def project_shift(self, train_ids):
        """Common-shift gauge (exact, F18): the LEARNED translations of the training exposures have zero mean (the initial registration
        shift0, e.g. from register_xcorr, is kept as given)."""
        self.affine.data[train_ids, 4:6] -= self.affine[train_ids, 4:6].mean(0)

    @torch.no_grad()
    def project_gauges(self, eids_by_filter=None, eids=None):
        """Pedestal gauge (F18): per filter, the training exposures' background constants measured from their medians sum to zero
        (sum_e (b_e0 - median_e) = 0); NIRCam stripes have zero mean over the rows / columns each exposure uses."""
        if eids_by_filter:
            for ids in eids_by_filter.values():
                self.bkg.data[ids, 0] -= (self.bkg[ids, 0] - self.bkg0[ids]).mean()
        if eids is not None:
            for T, M in ((self.stripe_rows, self.row_mask), (self.stripe_cols, self.col_mask)):
                mk = M[eids].float()
                mean = (T[eids] * mk).sum(1) / mk.sum(1).clamp(min=1)
                T.data[eids] -= mean[:, None] * mk

    # ------------------------------------------------------------------ priors / gauges
    def priors(self, eids):
        """Random-effect priors (N(0, I) codes and gains, and the affine and stripe priors), mean over the given exposures."""
        z = self.codes.z[eids].pow(2).sum(-1).mean() * 0.5
        u = self.u[eids].pow(2).mean() * 0.5
        aff = (self.affine[eids, :4] * 1e-3).pow(2).sum(-1).mean() * 0.5 / self.cfg.affine_prior ** 2
        st = (self.stripe_rows[eids].pow(2).mean() + self.stripe_cols[eids].pow(2).mean()) / (self.e_sigbar[eids].mean() ** 2) * self.cfg.stripe_l2
        return z + u + aff + st

    def shift_gauge(self, P):
        """Common shift: mean LEARNED displacement (Omega - X_wcs - shift0) over the training exposures at the 3x3 lattice, all lambda nodes [arcsec]
        (F18; logged only; the learned translations are projected to zero mean)."""
        tr = self.ref_eids
        acc = []
        for e in tr[:: max(1, tr.numel() // 64)]:
            xi = LATTICE.to(self.dev)
            X = torch.zeros(xi.shape[0], device=self.dev)
            pos = self.positions(e, X, X, xi, P)
            acc.append(pos.mean((0, 1)) - self.shift0[e])
        return torch.stack(acc).mean(0)

    @torch.no_grad()
    def instrument_mean_log_throughput(self, P=None, names=None):
        """Audit finding 5: mean log-throughput gamma (relative to the published curves, after the gauge) of the training exposures of each
        instrument, T-weighted over the Lambda nodes and averaged over the lattice. The gauge sets it to zero for the reference set; for the other
        instruments it is the flux-scale offset the fit has learned relative to that set. {instrument: {"mean_log_g", "n_train_exposures"}}."""
        P = self.prepare() if P is None else P
        if self.amp != "structured":
            return {}
        keep = self.gref
        out = {}
        try:
            for i in range(self.n_inst):
                ids = [int(e) for e in self.ref_eids.tolist() if int(self.e_inst[e]) == i]
                if not ids:
                    continue
                self.gref = torch.as_tensor(ids, device=self.dev, dtype=torch.long)
                g = self.gamma_raw_ref(P) - P["m"].flatten()[None]
                Tw = self.T_ref[self.e_filt[self.gref]]
                out[names[i] if names and i < len(names) else str(i)] = {"mean_log_g": float((Tw * g).sum() / Tw.sum().clamp(min=1e-12)),
                                                                        "n_train_exposures": len(ids)}
        finally:
            self.gref = keep
        return out

    @torch.no_grad()
    def gauge_report(self, exps=None, P=None):
        """Declared gauges, logged (F17, F18): throughput (enforced exactly: max |T-weighted mean of gamma over the reference set|, per lambda
        node), common shift (mas), common blur (mean core second moment of the kernels of the training exposures, arcsec^2, and its value at the
        warm start if recorded), background vs smooth sky (mean background constant per filter over training exposures and the 5th
        percentile of the band image of the sky)."""
        P = self.prepare() if P is None else P
        out = {}
        if self.amp == "structured":
            g = self.gamma_raw_ref(P) - P["m"].flatten()[None]
            Tw = self.T_ref[self.e_filt[self.gref]]
            den = Tw.sum(0)
            mean_g = torch.where(den > 0, (Tw * g).sum(0) / den.clamp(min=1e-12), torch.zeros_like(den))
            if self.unseen:      # nodes of filters without training exposures take the gauge shift of the nearest covered node: not part of the check
                Qn = self.fset.Q
                keep = torch.as_tensor([n for n in range(mean_g.numel()) if n // Qn not in self.unseen], device=mean_g.device)
                mean_g = mean_g[keep]
            out["throughput_gauge_max_abs_mean_log_g"] = float(mean_g.abs().max())
        out["common_shift_mas"] = (self.shift_gauge(P) * 1e3).tolist()
        m2 = []
        for e in self.ref_eids[:: max(1, self.ref_eids.numel() // 32)]:
            Ks, _ = self.node_kernels(e, torch.zeros(1, 2, device=self.dev), P)
            lv = self.levels_of(e)
            K0, u = Ks[0][0], lv.u[0]
            r2 = (u ** 2).sum(-1)
            core = r2 <= (0.5 * lv.R[0]) ** 2
            m2.append(float(((K0 * (r2 * core)).sum((-1, -2)) / (K0 * core).sum((-1, -2))).mean()))
        out["common_blur_core_m2_arcsec2"] = float(np.mean(m2))
        out["background_mean_per_filter"] = {self.fset.names[f]: float(self.bkg[self.ref_eids][self.e_filt[self.ref_eids] == f, 0].mean())
                                             for f in range(len(self.fset)) if bool((self.e_filt[self.ref_eids] == f).any())}
        S = self.sky.band_images(self.Phi_band)
        if self.unseen:
            out["filters_without_training_exposures"] = [self.fset.names[f] for f in self.unseen]
        out["sky_band_p05"] = {self.fset.names[f]: float(torch.quantile(S[f].flatten()[:: max(1, S[f].numel() // 100000)], 0.05)) for f in range(len(self.fset))}
        return out

    @torch.no_grad()
    def effective_variability(self, P=None, n_exp=24):
        """Size of each learned per-exposure variability in OUTPUT space (the code-space tau of the kernel is not identifiable on its own, F21):
        kernel: rms over training exposures x 3x3 lattice of the change of the normalised level-0 kernel when the code term is removed, relative
        to the rms of the kernel; throughput: the tau-scaled gain term [log g] (tau times the rms of the random effect over the training exposures)."""
        P = self.prepare() if P is None else P
        out = {}
        eids = self.ref_eids[:: max(1, self.ref_eids.numel() // n_exp)]
        saved = dict(self.tau.mode)
        ax = "k_exp"
        if self.tau(ax) is None:
            out[ax] = 0.0
        else:
            num, den = 0.0, 0.0
            for e in eids:
                xi = LATTICE.to(self.dev)
                K1 = self.node_kernels(e, xi, P)[0][0]
                self.tau.mode[ax] = "off"
                K0 = self.node_kernels(e, xi, P)[0][0]
                self.tau.mode[ax] = saved[ax]
                num += float(((K1 - K0) ** 2).sum())
                den += float((K1 ** 2).sum())
            out[ax] = math.sqrt(num / max(den, 1e-30))
        tr = self.ref_eids
        out["g_exp"] = self.tau.value("g_exp") * float(self.u[tr].pow(2).mean().sqrt()) if self.amp == "structured" else 0.0
        self.tau.mode.update(saved)
        return out

    def exposure_effects(self, nuis):
        """The per-exposure random-effect tensors that share one optimiser group (gain u, kernel code z)."""
        return [nuis["u"], nuis["z"]]

    def param_groups(self):
        """Operator (global) vs per-exposure nuisance vs sky vs tau parameters."""
        nuis = {"bkg": self.bkg, "affine": self.affine, "u": self.u, "z": self.codes.z, "stripe_rows": self.stripe_rows, "stripe_cols": self.stripe_cols}
        nuis_ids = {id(p) for p in nuis.values()}
        tau_ids = {id(p) for p in self.tau.parameters()}
        sky_ids = {id(p) for p in self.sky.parameters()}
        op = [p for p in self.parameters() if id(p) not in nuis_ids | tau_ids | sky_ids]
        return op, nuis, list(self.tau.parameters()), list(self.sky.parameters())


def region_nodes(xi_px, shape, region, bounds=None):
    """Region node lattice of spacing ~region px over the detector (shape (ny, nx)); bilinear weights of the pixels xi_px (M, 2) [col, row].
    Returns the nodes used by these pixels (n, 2) [col, row] and weights (n, M). With bounds = (col0, col1, row0, row1) (host ints) the node set
    is computed on the host (no device synchronisation)."""
    ny, nx = int(shape[0]), int(shape[1])
    nnx, nny = int(math.ceil((nx - 1) / region)) + 1, int(math.ceil((ny - 1) / region)) + 1
    sx, sy = (nx - 1) / (nnx - 1), (ny - 1) / (nny - 1)
    fx, fy = (xi_px[:, 0] / sx).clamp(0, nnx - 1 - 1e-6), (xi_px[:, 1] / sy).clamp(0, nny - 1 - 1e-6)
    ix, iy = fx.floor().long(), fy.floor().long()
    tx, ty = fx - ix, fy - iy
    if bounds is not None:
        c0, c1, r0, r1 = bounds
        ix0 = min(max(int(math.floor(c0 / sx)), 0), nnx - 2)
        ix1 = min(max(int(math.floor((c1 - 1) / sx)), 0), nnx - 2) + 1
        iy0 = min(max(int(math.floor(r0 / sy)), 0), nny - 2)
        iy1 = min(max(int(math.floor((r1 - 1) / sy)), 0), nny - 2) + 1
        nxl, nyl = ix1 - ix0 + 1, iy1 - iy0 + 1
        W = torch.zeros(nxl * nyl, xi_px.shape[0], device=xi_px.device)
        cols = torch.arange(xi_px.shape[0], device=xi_px.device)
        for dy in (0, 1):
            for dx in (0, 1):
                loc = (iy + dy - iy0).clamp(0, nyl - 1) * nxl + (ix + dx - ix0).clamp(0, nxl - 1)
                W.index_put_((loc, cols), (tx if dx else 1 - tx) * (ty if dy else 1 - ty), accumulate=True)
        gy, gx = torch.meshgrid(torch.arange(iy0, iy1 + 1, device=xi_px.device), torch.arange(ix0, ix1 + 1, device=xi_px.device), indexing="ij")
        nodes = torch.stack([gx.flatten().float() * sx, gy.flatten().float() * sy], -1)
        return nodes, W
    ids, ws = [], []
    for dy in (0, 1):
        for dx in (0, 1):
            ids.append((iy + dy) * nnx + (ix + dx))
            ws.append((tx if dx else 1 - tx) * (ty if dy else 1 - ty))
    ids, ws = torch.stack(ids), torch.stack(ws)               # (4, M)
    uniq, inv = torch.unique(ids, return_inverse=True)
    W = torch.zeros(uniq.numel(), xi_px.shape[0], device=xi_px.device)
    W.scatter_add_(0, inv, ws)
    nodes = torch.stack([(uniq % nnx).float() * sx, (uniq // nnx).float() * sy], -1)
    return nodes, W
