"""The field operator: everything between the sky s and the detector pixel d (equations in DESIGN.md, approximations F1... in __init__).

  d_e(xi) = N_e[ (1/Q) sum_q alpha_e(xi) exp(gamma_e(xi, lambda_q)) c_q(xi) ] + B_e(xi) + n_e(xi)
  c_q(xi) = int d^2u k_e(u; xi, lambda_q) s(Omega_e(xi, lambda_q) + u, lambda_q)          (render.py)
  Omega_e(xi, lambda) = x'(xi) + M_e x'(xi) + t_e,
      x' = X_wcs(xi) + shift0_e + tau_Ds Dst^(det, xi, lambda) + tau_Dd Ddr^(det, xi, t) + tau_De sum_j zD_ej B_j(xi)
  gamma_e(xi, lambda) = gbar(lambda, filter input, instrument) + tau_gd gdet^(det, xi, lambda) + tau_ge gep^(t, lambda, instrument)
                        + tau_gx u_e  -  m(lambda)                                     (m = gauge projection, F17)
  alpha = 1 (_cal / _flt) or |det dOmega/dxi| (_rate)
  B_e(xi) = sum_6 b_ej P_j(xi) + pattern_det(xi) + rows_e[row] + cols_e[col] (NIRCam)
  sigma^2 = (sigma_pipe exp(h_sigma[det, filter]))^2 + (f_floor[channel] |S|)^2
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import Codes, KernelNet, LogPolarEncoding, Tau, centred_normalised, inv_softplus, lam_feats, mlp, t_feats, xi_feats
from .render import Levels, kernel_level_values, kernel_level_values_grouped, pooled_level, quad_points, render_fft, render_fft_batch, render_quad

LATTICE = torch.tensor([[x, y] for y in (-0.8, 0.0, 0.8) for x in (-0.8, 0.0, 0.8)], dtype=torch.float32)


def legendre6(xi):
    x, y = xi[..., 0], xi[..., 1]
    return torch.stack([torch.ones_like(x), x, y, 1.5 * x * x - 0.5, x * y, 1.5 * y * y - 0.5], -1)


def quad_fields(xi):
    """Orthonormal (on [-1,1]^2, uniform) second-order fields for the per-exposure distortion: 6 vector fields (x and y components)."""
    x, y = xi[..., 0], xi[..., 1]
    p2x, p2y, pxy = math.sqrt(5) * (1.5 * x * x - 0.5), math.sqrt(5) * (1.5 * y * y - 0.5), 3.0 * x * y
    z = torch.zeros_like(x)
    return torch.stack([torch.stack([p2x, z], -1), torch.stack([pxy, z], -1), torch.stack([p2y, z], -1),
                        torch.stack([z, p2x], -1), torch.stack([z, pxy], -1), torch.stack([z, p2y], -1)], -2)  # (..., 6, 2)


class FieldOperator(nn.Module):
    def __init__(self, cfg, fset, meta, sky, dev):
        """meta: dict of per-exposure lists/arrays: filt, inst, chan, det, t, train(bool), R (E,2,2), sigbar, det_shape (E,2), shift0 (E,2),
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
        self.register_buffer("e_t", t(meta["t"], torch.float32))
        self.register_buffer("e_R", t(meta["R"], torch.float32))
        self.register_buffer("e_sigbar", t(meta["sigbar"], torch.float32))
        self.register_buffer("e_shape", t(meta["det_shape"]))
        self.register_buffer("shift0", t(meta["shift0"], torch.float32))
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
            self.levels_ch.append(Levels(pr, cfg.level_factor, cfg.level_nh, cfg.kernel_extent, a=cfg.level_overlap, device=dev))
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
        self.g_det_emb = nn.Embedding(n_det, 8)
        self.gdet = mlp(8 + 10 + 9, 32, 1, 2)
        self.gep = mlp(5 + 9 + 4, 32, 1, 2)
        self.u = nn.Parameter(torch.zeros(E, device=dev))
        # ---- geometry
        self.affine = nn.Parameter(torch.zeros(E, 6, device=dev))   # raw; physical = raw * aff_scale (matrix 1e-3, shift 0.01 arcsec per unit)
        self.register_buffer("aff_scale", torch.tensor([1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-2], device=dev))
        self.zD = nn.Parameter(torch.zeros(E, 6, device=dev))
        self.D_det = nn.Embedding(n_det, 8)
        self.Dst = mlp(8 + 10 + 9, 32, 2, 2)
        self.Ddr = mlp(8 + 10 + 5, 32, 2, 2)
        # ---- detector
        self.bkg = nn.Parameter(torch.zeros(E, 6, device=dev))
        with torch.no_grad():
            self.bkg[:, 0] = torch.as_tensor(np.asarray(meta["bkg0"]), dtype=torch.float32, device=dev)
        G = cfg.pattern_grid
        self.pattern = nn.Parameter(torch.zeros(n_det, G, G, device=dev)) if G > 0 else None
        self.stripe_rows = nn.Parameter(torch.zeros(E, 2048, device=dev))
        self.stripe_cols = nn.Parameter(torch.zeros(E, 2048, device=dev))
        self.register_buffer("e_stripes", t(meta["stripes"], torch.bool))
        self.hs = nn.Parameter(torch.zeros(n_det, len(fset), device=dev))
        self.floor_raw = nn.Parameter(inv_softplus(torch.full((n_chan,), cfg.floor0)).to(dev))
        self.nonlin = mlp(2 + 4, 32, 1, 2, zero_last=True) if cfg.nonlin else None
        self.nl_chan = nn.Embedding(n_chan, 4)
        # ---- variability scales
        self.tau = Tau(self.tau_spec(cfg, meta))
        # ---- reference sets (training exposures)
        tr = np.nonzero(self.train_mask)[0]
        self.ref_eids = torch.as_tensor(tr, device=dev, dtype=torch.long)
        dets = sorted({int(meta["det"][i]) for i in tr})
        L = LATTICE.to(dev)
        self.register_buffer("ref_det", torch.as_tensor(dets, device=dev, dtype=torch.long).repeat_interleave(L.shape[0]))
        self.register_buffer("ref_xi", L.repeat(len(dets), 1))
        ep = sorted({(float(meta["t"][i]), int(meta["inst"][i])) for i in tr})
        self.register_buffer("ref_t", torch.tensor([x[0] for x in ep], device=dev))
        self.register_buffer("ref_tinst", torch.tensor([x[1] for x in ep], device=dev, dtype=torch.long))
        dt_pairs = sorted({(int(meta["det"][i]), float(meta["t"][i])) for i in tr})
        self.register_buffer("ref_dt_det", torch.tensor([x[0] for x in dt_pairs], device=dev, dtype=torch.long).repeat_interleave(L.shape[0]))
        self.register_buffer("ref_dt_t", torch.tensor([x[1] for x in dt_pairs], device=dev).repeat_interleave(L.shape[0]))
        self.register_buffer("ref_dt_xi", L.repeat(len(dt_pairs), 1))
        # (detector, Lambda node) pairs observed in training: the throughput / distortion fields are normalised over these only
        Qn = fset.Q
        obs = torch.zeros(len(dets), len(fset) * Qn, dtype=torch.bool)
        for i in tr:
            k = dets.index(int(meta["det"][i]))
            f = int(meta["filt"][i])
            obs[k, f * Qn:(f + 1) * Qn] = True
        self.register_buffer("obs_idx", obs.repeat_interleave(L.shape[0], 0).flatten().nonzero()[:, 0].to(dev), persistent=False)  # flat (point, node)
        gref = tr if cfg.gauge_ref_inst < 0 else np.array([i for i in tr if int(meta["inst"][i]) == cfg.gauge_ref_inst])
        self.register_buffer("gref", torch.as_tensor(gref, device=dev, dtype=torch.long))
        Tt = fset.T_table(lam.flatten().cpu().numpy())        # (nf, nf*Q)
        self.register_buffer("T_ref", torch.as_tensor(Tt, dtype=torch.float32, device=dev))
        self.amp = cfg.amplitude
        self.input_level = cfg.input_level
        # ---- pedestal / stripe gauges (F18): per-exposure reference levels and the detector rows / columns each exposure uses
        self.register_buffer("bkg0", t(meta["bkg0"], torch.float32))
        rm = torch.zeros(E, 2048, dtype=torch.bool, device=dev)
        cm = torch.zeros(E, 2048, dtype=torch.bool, device=dev)
        for i, (r0, c0, nr, nc) in enumerate(meta.get("cut", [(0, 0, 2048, 2048)] * E)):
            rm[i, r0:r0 + nr] = True
            cm[i, c0:c0 + nc] = True
        self.register_buffer("row_mask", rm)
        self.register_buffer("col_mask", cm)
        # ---- blur gauge (F18): the sharpest kernel class (smallest generic width) may not be sharper than one sky tent
        fpix = {}
        for i in range(E):
            fpix.setdefault(int(meta["filt"][i]), meta["chan_pix"][int(meta["chan"][i])])
        diam = meta.get("diameter_m", 6.5)
        sig2 = {f: (0.42 * float(self.piv[f]) * 1e-6 / diam * 206264.8) ** 2 + p_ ** 2 / 12 for f, p_ in fpix.items()}
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
    def tau_spec(cfg, meta):
        t = np.asarray(meta["t"])[np.asarray(meta["train"], bool)]
        multi_epoch = len(np.unique(np.round(t, 6))) > 1 and cfg.epoch_axes
        glob = "learned" if cfg.tau_mode == "learned" else "on"

        def km(mode):
            return "on" if (mode == "learned" and cfg.tau_mode == "fixed") else mode
        spec = {
            "k_space": (km(cfg.k_space), cfg.tau0_code),
            "k_time": (km(cfg.k_time) if multi_epoch else "off", cfg.tau0_code),
            "k_lambda": (km(cfg.k_lambda), 1.0),
            "k_exp": (km(cfg.k_exp), cfg.tau0_code),
        }
        if cfg.amplitude == "structured":
            spec.update({"g_space": (glob, cfg.tau0_g), "g_time": (glob if multi_epoch else "off", cfg.tau0_g), "g_exp": (glob, cfg.tau0_g)})
        if cfg.distortion:
            spec.update({"D_static": (glob, cfg.tau0_D), "D_drift": (glob if multi_epoch else "off", cfg.tau0_D), "D_exp": (glob, cfg.tau0_D)})
        return spec

    def g_filt(self, f_idx):
        if self.cfg.filter_input == "curve":
            return self.g_fin(self.codes.curves[f_idx])
        return self.g_fin(f_idx)

    # ------------------------------------------------------------------ blur gauge
    def mean_codes(self, f):
        """Mean kernel code of filter f at its lambda nodes (every centred variability term at its mean, z = 0): (Q, dh)."""
        f = torch.as_tensor(f, device=self.dev)
        Q = self.fset.Q
        ins = self.e_inst[(self.e_filt == f).nonzero()[0, 0]]
        fq, iq = f.expand(Q), ins.expand(Q)
        hp = self.codes.hbar_at(self.piv[f].expand(Q), fq, iq)
        tl = self.tau("k_lambda")
        return hp if tl is None else hp + tl * (self.codes.hbar_at(self.lam[f], fq, iq) - hp)

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
        tau = self.tau
        d = self.dilation()
        if d is not None:
            P["dil"] = d
        # kernel det / epoch codes
        if tau("k_space") is not None:
            ref = self.codes.hdet_raw(self.ref_det, self.ref_xi)
            _, P["hdet_mu"], P["hdet_s"] = centred_normalised(ref, ref)
        if tau("k_time") is not None:
            ref = self.codes.hep_raw(self.ref_t, self.ref_tinst)
            _, P["hep_mu"], P["hep_s"] = centred_normalised(ref, ref)
        lamf = self.lam.flatten()                                           # Lambda: all nodes (nf*Q,)
        nL = lamf.numel()
        if self.amp == "structured":
            # throughput normalisations over (det lattice x Lambda) and (epochs x Lambda)
            if tau("g_space") is not None:
                nr = self.ref_det.numel()
                inp = torch.cat([self.g_det_emb(self.ref_det)[:, None].expand(-1, nL, -1), xi_feats(self.ref_xi)[:, None].expand(-1, nL, -1),
                                 lam_feats(lamf)[None].expand(nr, -1, -1)], -1)
                raw = self.gdet(inp)                                            # (nr, nL, 1)
                _, P["gdet_mu"], P["gdet_s"] = centred_normalised(raw, raw.reshape(-1, raw.shape[-1])[self.obs_idx])
                P["gdet_tab"] = ((raw[..., 0] - P["gdet_mu"]) / P["gdet_s"]).view(-1, LATTICE.shape[0], nL).mean(1)   # (n_ref_det, nL)
                P["gdet_dets"] = self.ref_det.view(-1, LATTICE.shape[0])[:, 0]
            if tau("g_time") is not None:
                ne = self.ref_t.numel()
                inp = torch.cat([t_feats(self.ref_t)[:, None].expand(-1, nL, -1), lam_feats(lamf)[None].expand(ne, -1, -1),
                                 self.g_inst(self.ref_tinst)[:, None].expand(-1, nL, -1)], -1)
                raw = self.gep(inp)
                _, P["gep_mu"], P["gep_s"] = centred_normalised(raw, raw.reshape(-1, 1))
                P["gep_tab"] = (raw[..., 0] - P["gep_mu"]) / P["gep_s"]       # (n_epochs, nL)
            P["m"] = self.gauge_mean(P)                                         # (nf, Q)
        if self.cfg.distortion:
            if tau("D_static") is not None:
                nr = self.ref_det.numel()
                inp = torch.cat([self.D_det(self.ref_det)[:, None].expand(-1, nL, -1), xi_feats(self.ref_xi)[:, None].expand(-1, nL, -1),
                                 lam_feats(lamf)[None].expand(nr, -1, -1)], -1)
                raw = self.Dst(inp)
                _, P["Dst_mu"], P["Dst_s"] = centred_normalised(raw, raw.reshape(-1, raw.shape[-1])[self.obs_idx])
            if tau("D_drift") is not None:
                raw = self.Ddr(torch.cat([self.D_det(self.ref_dt_det), xi_feats(self.ref_dt_xi), t_feats(self.ref_dt_t)], -1))
                _, P["Ddr_mu"], P["Ddr_s"] = centred_normalised(raw, raw)
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
        if "gdet_tab" in P and self.tau("g_space") is not None:
            pos = (P["gdet_dets"][None] == self.e_det[e][:, None]).float().argmax(1)
            g = g + self.tau("g_space") * P["gdet_tab"][pos]
        if "gep_tab" in P and self.tau("g_time") is not None:
            pos = ((self.ref_t[None] - self.e_t[e][:, None]).abs() + 10 * (self.ref_tinst[None] != ins[:, None]).float()).argmin(1)
            g = g + self.tau("g_time") * P["gep_tab"][pos]
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
        return m.view(len(self.fset), self.fset.Q)

    # ------------------------------------------------------------------ components at points of one exposure
    def codes_at(self, e, xi, P, lam=None):
        """Kernel codes for exposure e at normalised positions xi (M, 2): (Q, M, dh)."""
        f, ins = self.e_filt[e], self.e_inst[e]
        lam = self.lam[f] if lam is None else lam
        Q, M = lam.numel(), xi.shape[0]
        fq = f.expand(Q)
        iq = ins.expand(Q)
        hp = self.codes.hbar_at(self.piv[f].expand(Q), fq, iq)                # (Q, dh) pivot code
        tl = self.tau("k_lambda")
        h = hp if tl is None else hp + tl * (self.codes.hbar_at(lam, fq, iq) - hp)
        h = h[:, None].expand(Q, M, -1)
        if "hdet_mu" in P and self.tau("k_space") is not None:
            hd = (self.codes.hdet_raw(self.e_det[e].expand(M), xi) - P["hdet_mu"]) / P["hdet_s"]
            h = h + self.tau("k_space") * hd[None]
        if "hep_mu" in P and self.tau("k_time") is not None:
            he = (self.codes.hep_raw(self.e_t[e][None], ins[None]) - P["hep_mu"]) / P["hep_s"]
            h = h + self.tau("k_time") * he[None]
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
        if "gdet_mu" in P and self.tau("g_space") is not None:
            inp = torch.cat([self.g_det_emb(self.e_det[e].expand(Q, M)), xi_feats(xi)[None].expand(Q, -1, -1), lam_feats(lam)[:, None].expand(-1, M, -1)], -1)
            g = g + self.tau("g_space") * ((self.gdet(inp)[..., 0] - P["gdet_mu"][0]) / P["gdet_s"])
        if "gep_mu" in P and self.tau("g_time") is not None:
            inp = torch.cat([t_feats(self.e_t[e]).expand(Q, -1), lam_feats(lam), self.g_inst(ins.expand(Q))], -1)
            g = g + self.tau("g_time") * ((self.gep(inp)[..., 0] - P["gep_mu"][0]) / P["gep_s"])[:, None]
        if self.tau("g_exp") is not None:
            g = g + self.tau("g_exp") * self.u[e]
        return g - P["m"][f][:, None]

    def distortion(self, e, xi, P):
        """D (Q, M, 2) [arcsec] or None."""
        if not self.cfg.distortion:
            return None
        f = self.e_filt[e]
        lam = self.lam[f]
        Q, M = lam.numel(), xi.shape[0]
        D = torch.zeros(Q, M, 2, device=xi.device)
        if "Dst_mu" in P and self.tau("D_static") is not None:
            inp = torch.cat([self.D_det(self.e_det[e].expand(Q, M)), xi_feats(xi)[None].expand(Q, -1, -1), lam_feats(lam)[:, None].expand(-1, M, -1)], -1)
            D = D + self.tau("D_static") * (self.Dst(inp) - P["Dst_mu"]) / P["Dst_s"]
        if "Ddr_mu" in P and self.tau("D_drift") is not None:
            inp = torch.cat([self.D_det(self.e_det[e].expand(M)), xi_feats(xi), t_feats(self.e_t[e]).expand(M, -1)], -1)
            D = D + self.tau("D_drift") * ((self.Ddr(inp) - P["Ddr_mu"]) / P["Ddr_s"])[None]
        if self.tau("D_exp") is not None:
            D = D + self.tau("D_exp") * torch.einsum("j,mjc->mc", self.zD[e], quad_fields(xi))[None]
        return D

    def positions(self, e, X, Y, xi, P):
        """Omega_e(xi, lambda_q): (Q, M, 2) [arcsec]."""
        Q = self.fset.Q
        x = torch.stack([X, Y], -1)[None].expand(Q, -1, -1) + self.shift0[e]
        D = self.distortion(e, xi, P)
        if D is not None:
            x = x + D
        a = self.affine[e] * self.aff_scale
        Mx = torch.stack([a[0] * x[..., 0] + a[1] * x[..., 1], a[2] * x[..., 0] + a[3] * x[..., 1]], -1)
        return x + Mx + a[4:6]

    def pixel_area(self, e, Jpix, xi, P):
        """alpha = |det dOmega/dxi| per pixel [arcsec^2] (F15). Jpix (M, 2, 2) header-WCS Jacobian d(X,Y)/d(col,row) per pixel.
        dD/dxi by central differences of D (1 detector pixel); (I + M_e) applied. Returns (Q, M)."""
        a = self.affine[e] * self.aff_scale
        A = torch.eye(2, device=Jpix.device) + torch.stack([a[0:2], a[2:4]])
        J = Jpix[None].expand(self.fset.Q, -1, -1, -1)
        if self.cfg.distortion:
            ny, nx = int(self.e_shape[e][0]), int(self.e_shape[e][1])
            dxn, dyn = 2.0 / (nx - 1), 2.0 / (ny - 1)
            ex = torch.tensor([dxn, 0.0], device=xi.device)
            ey = torch.tensor([0.0, dyn], device=xi.device)
            Dx = (self.distortion(e, xi + ex, P) - self.distortion(e, xi - ex, P)) / 2.0   # per detector pixel
            Dy = (self.distortion(e, xi + ey, P) - self.distortion(e, xi - ey, P)) / 2.0
            J = J + torch.stack([Dx, Dy], -1)
        JA = torch.einsum("ij,qmjk->qmik", A, J)
        return (JA[..., 0, 0] * JA[..., 1, 1] - JA[..., 0, 1] * JA[..., 1, 0]).abs()

    def background(self, e, xi, row, col):
        B = (legendre6(xi) * self.bkg[e]).sum(-1)
        if self.pattern is not None:
            pat = self.pattern[self.e_det[e]]
            pat = pat - pat.mean()
            B = B + F.grid_sample(pat[None, None], xi[None, None], mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0]
        if bool(self.e_stripes[e]) and self.cfg.stripes:
            B = B + self.stripe_rows[e, row] + self.stripe_cols[e, col]
        return B

    def apply_nonlin(self, e, S):
        if self.nonlin is None:
            return S
        sb = self.e_sigbar[e]
        gate = S * S / (S * S + (10 * sb) ** 2)
        inp = torch.cat([torch.stack([torch.asinh(S / sb) / 5, gate], -1), self.nl_chan(self.e_chan[e]).expand(S.shape[0], -1)], -1)
        return S + 0.2 * S * gate * torch.tanh(self.nonlin(inp)[..., 0])

    def sigma(self, e, S, sig_pipe):
        h = 3 * torch.tanh(self.hs[self.e_det[e], self.e_filt[e]] / 3)
        fl = F.softplus(self.floor_raw[self.e_chan[e]])
        return torch.sqrt((sig_pipe * torch.exp(h)) ** 2 + (fl * S) ** 2)

    def levels_of(self, e):
        return self.levels_ch[self.l_chan[int(e)] if isinstance(e, int) else int(self.e_chan[e])]

    # ------------------------------------------------------------------ kernels at region nodes
    def node_kernels(self, e, xi_nodes, P, shift0=None):
        """Discrete level kernels for the region nodes of exposure e: list over levels of (n, Q, kh, kw)."""
        h = self.codes_at(e, xi_nodes, P)                        # (Q, n, dh)
        Q, n, dh = h.shape
        Ks, logZ = kernel_level_values(self.net, self.levels_of(e), self.e_R[e], h.permute(1, 0, 2).reshape(n * Q, dh), amp_free=(self.amp == "free"),
                                       shift0=shift0, ckpt=bool(self.cfg.ckpt_kernel), dil=P.get("dil"))
        return [K.view(n, Q, *K.shape[-2:]) for K in Ks], logZ.view(n, Q)

    # ------------------------------------------------------------------ the forward model on a set of pixels of one exposure
    def render(self, e, pix, P, method="fft", view=None, region_px=None, quad=None, detach_sky=False, jitter=False):
        """pix: dict with X, Y (M,), xi (M, 2) normalised, xi_px (M, 2) detector pixel coords, row, col (M,) long, Jpix (M,2,2) optional.
        Returns dict S (signal before N), m (model mean), pos."""
        e_t = torch.as_tensor(e, device=self.dev)
        A = self.sky.coeffs(view)
        if detach_sky:
            A = A.detach()
        f = int(self.e_filt[e_t])
        Phi_q = self.Phi[f]
        pos = self.positions(e_t, pix["X"], pix["Y"], pix["xi"], P)
        if method == "fft" and self.cfg.batch_render:
            return self.render_batch([(int(e), pix)], P, view=view, detach_sky=detach_sky, jitter=jitter, region_px=region_px)[0] | {"pos": pos}
        if method == "fft":
            region = region_px or self.cfg.region_px
            nodes, wts = region_nodes(pix["xi_px"], self.e_shape[e_t], region)
            xin = torch.stack([nodes[:, 0] / (self.e_shape[e_t][1] - 1) * 2 - 1, nodes[:, 1] / (self.e_shape[e_t][0] - 1) * 2 - 1], -1).float()
            sh = (torch.rand(2) - 0.5) if jitter else None
            Ks, _ = self.node_kernels(e_t, xin, P, shift0=sh)
            c = render_fft(self.sky, A, Phi_q, self.levels_of(e_t), Ks, wts, pos, self.os_ch[int(self.e_chan[e_t])], shift0=sh, interp=self.cfg.interp)
        else:
            qpts, qw = quad
            h = self.codes_at(e_t, pix["xi"], P)                  # (Q, M, dh)
            c = render_quad(self.sky, A, Phi_q, self.net, self.e_R[e_t], h, pos, qpts, qw, amp_free=(self.amp == "free"), dil=P.get("dil"))
        gam = self.gamma_at(e_t, pix["xi"], P)
        amp = torch.exp(gam)
        if self.input_level == "rate":
            amp = amp * self.pixel_area(e_t, pix["Jpix"], pix["xi"], P)
        S = (self.wq[f][:, None] * amp * c).sum(0)
        m = self.apply_nonlin(e_t, S) + self.background(e_t, pix["xi"], pix["row"], pix["col"])
        return {"S": S, "m": m, "pos": pos}

    def finish(self, e_t, pix, P, c, f=None):
        f = int(self.e_filt[e_t]) if f is None else f
        gam = self.gamma_at(e_t, pix["xi"], P)
        amp = torch.exp(gam)
        if self.input_level == "rate":
            amp = amp * self.pixel_area(e_t, pix["Jpix"], pix["xi"], P)
        S = (self.wq[f][:, None] * amp * c).sum(0)
        m = self.apply_nonlin(e_t, S) + self.background(e_t, pix["xi"], pix["row"], pix["col"])
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
                e_t = torch.as_tensor(e, device=self.dev)
                pos = self.positions(e_t, pix["X"], pix["Y"], pix["xi"], P)
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
            c = render_fft_batch(self.sky, A, torch.stack(Phis), lv, Ks, Wb, Pb, shift0=sh, interp=self.cfg.interp, Rs=Rb, det_frame=det_frame)
            for b, k in enumerate(ks):
                e, pix = items[k]
                outs[k] = self.finish(torch.as_tensor(e, device=self.dev), pix, P, c[b, :, :nps[b]], f=self.l_filt[e])
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
        """Random-effect priors (N(0, I) codes, gains and per-exposure distortion), mean over the given exposures."""
        z = self.codes.z[eids].pow(2).sum(-1).mean() * 0.5
        u = self.u[eids].pow(2).mean() * 0.5
        zd = self.zD[eids].pow(2).sum(-1).mean() * 0.5
        aff = (self.affine[eids, :4] * 1e-3).pow(2).sum(-1).mean() * 0.5 / self.cfg.affine_prior ** 2
        st = (self.stripe_rows[eids].pow(2).mean() + self.stripe_cols[eids].pow(2).mean()) / (self.e_sigbar[eids].mean() ** 2) * self.cfg.stripe_l2
        return z + u + zd + aff + st

    def shift_gauge(self, P):
        """Common shift: mean LEARNED displacement (Omega - X_wcs - shift0) over the training exposures at the 3x3 lattice, all lambda nodes [arcsec]
        (F18; logged, and penalised only if --gauge-shift > 0; the learned translations are also projected to zero mean)."""
        tr = self.ref_eids
        acc = []
        for e in tr[:: max(1, tr.numel() // 64)]:
            xi = LATTICE.to(self.dev)
            X = torch.zeros(xi.shape[0], device=self.dev)
            pos = self.positions(e, X, X, xi, P)
            acc.append(pos.mean((0, 1)) - self.shift0[e])
        return torch.stack(acc).mean(0)

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
        out["sky_band_p05"] = {self.fset.names[f]: float(torch.quantile(S[f].flatten()[:: max(1, S[f].numel() // 100000)], 0.05)) for f in range(len(self.fset))}
        return out

    @torch.no_grad()
    def effective_variability(self, P=None, n_exp=24):
        """Size of each learned variability in OUTPUT space (the code-space tau of the kernel axes is not identifiable on its own, F21):
        kernel axes: rms over training exposures x 3x3 lattice of the change of the normalised level-0 kernel when the axis term is removed,
        relative to the rms of the kernel; throughput axes: rms of the tau-scaled term [log g]; distortion axes: rms displacement [mas]."""
        P = self.prepare() if P is None else P
        out = {}
        eids = self.ref_eids[:: max(1, self.ref_eids.numel() // n_exp)]
        xi = LATTICE.to(self.dev)
        saved = dict(self.tau.mode)
        for ax in ("k_space", "k_time", "k_lambda", "k_exp"):
            if self.tau(ax) is None:
                out[ax] = 0.0
                continue
            num, den = 0.0, 0.0
            for e in eids:
                K1 = self.node_kernels(e, xi, P)[0][0]
                self.tau.mode[ax] = "off"
                K0 = self.node_kernels(e, xi, P)[0][0]
                self.tau.mode[ax] = saved[ax]
                num += float(((K1 - K0) ** 2).sum())
                den += float((K1 ** 2).sum())
            out[ax] = math.sqrt(num / max(den, 1e-30))
        # throughput and distortion terms are normalised to unit rms over their reference sets, so their output-space size is tau itself
        # (log g; arcsec -> mas); the per-exposure ones are tau x rms of the random effects over the training exposures
        tr = self.ref_eids
        out["g_space"] = self.tau.value("g_space") if self.amp == "structured" else 0.0
        out["g_time"] = self.tau.value("g_time") if self.amp == "structured" else 0.0
        out["g_exp"] = self.tau.value("g_exp") * float(self.u[tr].pow(2).mean().sqrt()) if self.amp == "structured" else 0.0
        out["D_static"] = self.tau.value("D_static") * 1e3 if self.cfg.distortion else 0.0
        out["D_drift"] = self.tau.value("D_drift") * 1e3 if self.cfg.distortion else 0.0
        out["D_exp"] = self.tau.value("D_exp") * 1e3 * float(self.zD[tr].pow(2).sum(-1).mean().sqrt()) if self.cfg.distortion else 0.0
        self.tau.mode.update(saved)
        return out

    def param_groups(self):
        """Operator (global) vs per-exposure nuisance vs sky vs tau parameters."""
        nuis = {"bkg": self.bkg, "affine": self.affine, "u": self.u, "z": self.codes.z, "zD": self.zD, "stripe_rows": self.stripe_rows, "stripe_cols": self.stripe_cols}
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
