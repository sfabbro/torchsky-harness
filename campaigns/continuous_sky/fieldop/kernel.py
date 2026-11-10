"""Kernel shape k_e(u; xi, lambda): ONE conditional neural field of the offset u (F5-F8), and its conditioning code h.

  k(u; h) = exp(l(v, h)) / Z,     v = R_e^T u   (u: sky-plane offset [arcsec]; R_e: rotation (with parity) of the exposure's mean
                                                 pixel-to-sky Jacobian, so v is the offset in the DETECTOR-aligned frame, arcsec)
  l(v, h) = FiLM-MLP( enc(v); h ),   enc = multi-scale log-radius x angle encoding (LogPolarEncoding)
  Z       = sum over the RENDERING quadrature of exp(l) * window * cell area   (unit integral, F7)

  h = hbar(lambda_pivot, filter input, instrument) + tau_lam (hbar(lambda) - hbar(lambda_pivot))
      + tau_det hdet^(detector, xi) + tau_epoch hep^(t, instrument) + tau_exp z_e,        z_e ~ N(0, I)

hdet^ and hep^ are CENTRED and RMS-NORMALISED over a reference set of the training data (detector x 3x3 lattice; training epochs), so
hbar is the mean code and every tau is the amplitude of its variability in code space; tau = 0 freezes the axis.  The filter enters as an
INPUT: either its published transmission curve sampled on a fixed log-lambda grid (`curve`) or a learned token (`token`).
No basis is imposed on the shape: no Gaussian, no polynomial, no low rank.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sky import lam_unit


def inv_softplus(y):
    y = torch.as_tensor(y, dtype=torch.float32)
    return y + torch.log(-torch.expm1(-y))


def mlp(i, h, o, n_hidden=2, zero_last=False, act=nn.SiLU):
    layers, d = [], i
    for _ in range(n_hidden):
        layers += [nn.Linear(d, h), act()]
        d = h
    last = nn.Linear(d, o)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)


def lam_feats(lam_um, n=4):
    u = lam_unit(lam_um)
    k = torch.arange(1, n + 1, device=u.device, dtype=u.dtype)
    a = math.pi * u[..., None] * k
    return torch.cat([u[..., None], torch.sin(a), torch.cos(a)], -1)  # 1 + 2n


def xi_feats(xi, n=2):
    """xi (..., 2) normalised detector coordinates in [-1, 1]."""
    k = torch.arange(1, n + 1, device=xi.device, dtype=xi.dtype)
    a = math.pi * xi[..., :, None] * k * 0.5
    return torch.cat([xi, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], -1)  # 2 + 4n


def t_feats(t):
    """t (...,) normalised epoch in [-1, 1]."""
    return torch.stack([t, torch.sin(math.pi * t), torch.cos(math.pi * t), torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], -1)


class LogPolarEncoding(nn.Module):
    """enc(v), a SMOOTH function of v everywhere (no cone tip at v = 0, so grid quadratures of the kernel converge fast):
    rho = log(sqrt(r^2 + r0^2)) mapped to [-1, 1] over [r0, r_max]; radial Fourier features of rho (frequencies 1..n_rad: resolution uniform
    in log radius, i.e. sharp core and long wings); angular harmonics Re/Im[((x + i y) / sqrt(r^2 + r0^2))^m], m = 1..m_max (polynomial in v
    near 0, cos/sin(m theta) far out); products of the first n_cross radial features with the angular ones."""

    def __init__(self, r0=0.005, r_max=20.0, n_rad=8, m_max=12, r_gate=None, n_cross=3):
        super().__init__()
        self.r0, self.r_max, self.n_rad, self.m_max, self.n_cross = r0, r_max, n_rad, m_max, n_cross
        self.lo, self.hi = math.log(r0), math.log(math.hypot(r_max, r0))
        self.dim = 1 + 2 * n_rad + 2 * m_max + 2 * m_max * n_cross

    def forward(self, v):
        r2 = (v * v).sum(-1) + self.r0 ** 2
        rho = (0.5 * torch.log(r2) - self.lo) / (self.hi - self.lo) * 2 - 1
        k = torch.arange(1, self.n_rad + 1, device=v.device, dtype=v.dtype)
        a = math.pi * rho[..., None] * k * 0.5
        rad = [rho[..., None], torch.sin(a), torch.cos(a)]
        s = torch.rsqrt(r2)
        zr, zi = v[..., 0] * s, v[..., 1] * s
        re, im, cr, ci = [], [], zr, zi
        for _ in range(self.m_max):
            re.append(cr)
            im.append(ci)
            cr, ci = cr * zr - ci * zi, cr * zi + ci * zr
        ang = torch.cat([torch.stack(re, -1), torch.stack(im, -1)], -1)
        cross = [ang * torch.cos(math.pi * rho[..., None] * j * 0.5) for j in range(1, self.n_cross + 1)]
        return torch.cat(rad + [ang] + cross, -1)


class KernelNet(nn.Module):
    """l(v, h): MLP on enc(v) with FiLM modulation of every hidden layer by h."""

    def __init__(self, enc: LogPolarEncoding, dh=16, hidden=64, layers=3):
        super().__init__()
        self.enc, self.dh, self.hidden, self.layers = enc, dh, hidden, layers
        self.inp = nn.Linear(enc.dim, hidden)
        self.hid = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers - 1)])
        self.out = nn.Linear(hidden, 1)
        self.film = nn.Linear(dh, 2 * hidden * layers)
        nn.init.normal_(self.film.weight, std=0.1 / math.sqrt(dh))
        nn.init.zeros_(self.film.bias)

    def forward(self, feats, h):
        """feats (P, F) shared offsets (or (C, P, F) per code), h (C, dh) -> log density (C, P).
        Grouped: feats (G, P, F) and h (G, C, dh) -> (G, C, P) (one offset set per group, e.g. per exposure rotation)."""
        if h.dim() == 3:
            G, C = h.shape[:2]
            gb = self.film(h).view(G, C, self.layers, 2, self.hidden)
            a = self.inp(feats)[:, None]                                   # (G, 1, P, H)
            x = F.silu(a * (1 + gb[:, :, 0, 0, None]) + gb[:, :, 0, 1, None])
            for l, lin in enumerate(self.hid, start=1):
                x = F.silu(lin(x) * (1 + gb[:, :, l, 0, None]) + gb[:, :, l, 1, None])
            return self.out(x)[..., 0].clamp(max=30.0)
        C = h.shape[0]
        gb = self.film(h).view(C, self.layers, 2, self.hidden)
        a = self.inp(feats)
        if a.dim() == 2:
            a = a[None]
        x = F.silu(a * (1 + gb[:, 0, 0, None]) + gb[:, 0, 1, None])
        for l, lin in enumerate(self.hid, start=1):
            x = F.silu(lin(x) * (1 + gb[:, l, 0, None]) + gb[:, l, 1, None])
        return self.out(x)[..., 0].clamp(max=30.0)


class Tau(nn.Module):
    """Variability scales. mode per axis: 'off' (tau = 0, axis frozen), 'on' (tau fixed at its initial value), 'learned' (fitted by the
    validation likelihood, F21). tau = softplus(raw) for learned axes, so it can approach 0 but stays differentiable."""

    def __init__(self, spec):
        super().__init__()
        self.mode, self.init = {}, {}
        self.raw = nn.ParameterDict()
        for name, (mode, t0) in spec.items():
            self.mode[name], self.init[name] = mode, float(t0)
            if mode == "learned":
                self.raw[name] = nn.Parameter(inv_softplus(torch.tensor(float(t0))))

    def __call__(self, name):
        m = self.mode.get(name, "off")
        if m == "off":
            return None
        if m == "on":
            return self.init[name]
        return F.softplus(self.raw[name])

    def value(self, name):
        v = self(name)
        return 0.0 if v is None else float(v.detach() if torch.is_tensor(v) else v)

    def table(self):
        return {k: {"mode": self.mode[k], "init": self.init[k], "value": self.value(k)} for k in self.mode}

    def occam(self, exclude=()):
        """Weak exponential prior sum_a tau_a / tau0_a over learned axes (F21)."""
        tot = None
        for k in self.raw:
            if k in exclude:
                continue
            v = F.softplus(self.raw[k]) / max(self.init[k], 1e-6)
            tot = v if tot is None else tot + v
        return tot


def centred_normalised(f, ref):
    """(f - mean(ref)) / rms(ref - mean(ref)) with the statistics over the reference rows (last dim = feature)."""
    mu = ref.mean(0)
    s = torch.sqrt(((ref - mu) ** 2).sum(-1).mean() / ref.shape[-1] + 1e-12)
    return (f - mu) / s, mu, s


class Codes(nn.Module):
    """Conditioning code h of the kernel field."""

    def __init__(self, n_filters, n_inst, n_det, n_exp, curves, filter_input="curve", dh=16, hidden=64):
        super().__init__()
        self.dh, self.filter_input = dh, filter_input
        self.register_buffer("curves", curves)  # (nf, CURVE_N)
        if filter_input == "curve":
            self.f_in = nn.Linear(curves.shape[1], 16)
        else:
            self.f_tok = nn.Embedding(n_filters, 16)
        self.inst = nn.Embedding(n_inst, 4)
        self.hbar = mlp(9 + 16 + 4, hidden, dh, 2)
        self.det = nn.Embedding(n_det, 8)
        self.hdet = mlp(8 + 10, hidden, dh, 2)
        self.hep = mlp(5 + 4, 32, dh, 2)
        self.z = nn.Parameter(torch.zeros(n_exp, dh))

    def filt(self, f_idx):
        if self.filter_input == "curve":
            return self.f_in(self.curves[f_idx])
        return self.f_tok(f_idx)

    def hbar_at(self, lam, f_idx, inst_idx):
        return self.hbar(torch.cat([lam_feats(lam), self.filt(f_idx), self.inst(inst_idx)], -1))

    def hdet_raw(self, det_idx, xi):
        return self.hdet(torch.cat([self.det(det_idx), xi_feats(xi)], -1))

    def hep_raw(self, t, inst_idx):
        return self.hep(torch.cat([t_feats(t), self.inst(inst_idx)], -1))
