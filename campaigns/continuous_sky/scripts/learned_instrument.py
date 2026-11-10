#!/usr/bin/env python3
"""Fully learned instrument (track L of forward-operator-spec.md) applied to Spitzer IRAC, sky frozen from JWST NIRCam LW.

    d_i(p) = F_theta[ sky patch around p in the detector frame ; meta ; z_i ]   (+ learned noise)

APPROXIMATION (stated here and in the report): the sky is the NIRCam LW coadd of jwst_lw_sky.py (bands F277W/F356W/F444W/F410M as
channels). The NIRCam LW PSF (0.12-0.15 arcsec) is negligible next to IRAC's (~1.7 arcsec, 1.2 arcsec pixels), so the coadd stands
in for the true sky; the 3D (wavelength) quadrature of the spec is replaced by the NIRCam band images; no registration beyond the
header WCS (its errors are absorbed by the learned operator).

Per detector pixel p of frame i
    * starting sky position from the frame's header WCS (TAN-SIP) -> sky-grid pixel position pos(p); local Jacobian J(p) =
      d(sky px)/d(col, row) from central differences of the same WCS (rotation, scale, shear, parity, distortion);
    * patch: P x P samples (default 25 x 25 at 0.3 arcsec = +-3.6 arcsec = +-3 IRAC pixels) on a regular grid in the DETECTOR frame,
      mapped to the sky by J (the patch rotates with the detector), bilinear; the sky is first pre-smoothed with a box of the
      patch pitch (normalised Gaussian, NaN-aware) to avoid aliasing the 0.12 arcsec sky at 0.3 arcsec; a sample is valid only if
      all four bilinear neighbours are covered (NaN -> value 0 + mask channel);
    * regularisers of the learned model (generic, no optics): roughness (Laplacian) of W_base and of the bank, energy of the bank, size
      of the net output of g (all relative to the base-kernel energy), L2 of z_i (cfg.smooth, bank_l2, g_l2, z_l2);
    * learned model (--model learned):
        L(p)    = sum_{band, sample} W[band, sample](detector x, y, channel, epoch, z_i) * patch[band, sample]       (LINEAR in the sky)
                  W = (W_base[channel] + sum_k a_k(cond) B_k) / (its total weight),  a = hypernetwork MLP(cond), B_k a learned basis
                  bank; the unit total weight (flat sky in = flat sky out) fixes the scale of L so that g cannot trade a gain against
                  the kernel; colour mixing of the bands is free (negative weights allowed)
        pred(p) = L + b_i + g(L, patch summary, t_exp, DCE index, time since previous frame, x, y, channel, z_i)    (g NONLINEAR, last
                  layer zero-initialised so g == 0 at start; g = gate(L) * (s + |L|) * net(...) with gate = L^2/(L^2 + s_g^2), s_g = 10 noise units: its size
                  follows the signal and the slope of pred(L) at faint signal is 1 = the low-signal gauge of the spec)
        sigma(p)= cbunc(p) * exp(h(L, patch summary, meta, z_i)), h = 3 tanh(net/3) (learned per-pixel log-variance on top of cbunc)
        loss    = Gaussian NLL 0.5 r^2/sigma^2 + log sigma (--loss nll, default) or the same with Huber(delta) instead of 0.5 z^2 (--loss huber)
    * structured baseline (--model structured), same sampler and loss form: L = gain_i * sum_band c[channel, band] * (PSF ⊗ box)
      with the PSF a mixture of M centred anisotropic Gaussians (weights, sigmas, angle = MLP of detector x, y, channel, tabulated on
      a 9 x 9 grid of detector nodes per channel and bilinearly interpolated), convolved with the 1.2 arcsec pixel box (4 x 4
      midpoint sub-samples), unit flux (sum of kernel = 1), zero centroid; pred = L + b_i; sigma = cbunc; no g, no variance head.

Data-only spike mask: single-pixel spikes (cosmic rays / hot pixels that the bimsk fatal bits do not flag; > 8 sigma over a 7x7 median with
neighbours < 12 percent) are removed from `good` (counted per AOR as n_spikes_masked); they are not sky and no sky-based model can predict them.

Split (deterministic, by AOR, written to split.json): 2 test AORs + 1 validation AOR from three different programs (programs
with >= 2 AORs preferred, so every program keeps training data), the rest train. Usable pixel = good (bimsk & 32520 == 0, finite,
cbunc > 0) AND the whole patch inside JWST coverage of the required bands (default F356W and F444W; other bands enter with their
mask channel). Training uses all usable pixels of train frames; scoring uses only usable pixels.

Held-out frames have their own nuisance parameters (b_i, z_i, or b_i and log-gain for the baseline). They are fitted with all shared
weights frozen on a deterministic checkerboard half of the usable pixels ((row + col) even) and every chi2 metric is computed on the
other half; aperture photometry uses the whole frame (the fitted nuisance has 1 + 4 numbers per frame).

Metrics (same code for both models, per channel and all): chi2 = <r^2/cbunc^2> (mean; median/0.4549; (1.4826 MAD)^2; mean of
min(z^2, 25)), the same on bright pixels (S/N = (L+g)/cbunc > 5), chi2 and NLL with the model's own sigma, and aperture-photometry
flux error on isolated, PSF-like (not single-pixel) S/N > 20 sources (r = 3 px aperture, 3.5-6 px annulus median background removed from data and prediction alike).

CLI
    python3 learned_instrument.py --sky SKY.fits --irac-root DIR --out DIR [--model learned|structured|both] [--quick]
        [--budget-min M] [--steps N] [--batch B] [--device auto|cpu|cuda] [--split-json F] [--aors a,b] [--max-frames-per-aor N]
        [--resume] [--count-only] [--loss nll|huber] [--patch 25 --pitch 0.3] ...
Outputs in --out: split.json, usable_pixels.json, <model>/ckpt.pt, <model>/train_log.txt, <model>/resid_<frame>.png, metrics.json
(models run into the same --out are merged into one metrics.json).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

warnings.filterwarnings("ignore", category=FITSFixedWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from jwst_lw_sky import FrozenSky, load_sky  # noqa: E402

IRAC_FATAL_BITS = 32520  # IRAC Instrument Handbook 7.1.1 (= pixelstore.IRAC_FATAL_BITS; checked in the tests)
IRAC_PIX_ARCSEC = 1.2
MATCH_BAND = {0: "F356W", 1: "F444W"}  # starting guess only: IRAC1 -> F356W, IRAC2 -> F444W
MJD_REF, MJD_SCALE = 57000.0, 1500.0


def log(msg, fh=None):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    if fh is not None:
        fh.write(s + "\n")
        fh.flush()


@dataclass
class Cfg:
    patch: int = 25
    pitch: float = 0.3
    hidden: int = 64
    K: int = 32
    dz: int = 4
    n_mix: int = 4
    psf_nodes: int = 9
    loss: str = "nll"
    huber_delta: float = 2.0
    batch: int = 4096
    lr: float = 2e-3
    lr_kernel: float = 1e-3
    steps: int = 20000
    budget_min: float = 0.0
    nuis_steps: int = 200
    z_l2: float = 1e-3
    smooth: float = 0.1     # roughness penalty of the linear kernels (relative to their energy)
    bank_l2: float = 0.1    # energy penalty of the basis bank (relative to the base-kernel energy)
    g_l2: float = 0.05      # penalty on the size of the nonlinear part (net output^2): prefer explaining by the linear part
    snr_bright: float = 5.0
    require_bands: tuple = ("F356W", "F444W")
    bright_frac: float = 0.25
    val_every: int = 0
    sa: float = 3.0  # asinh scale of the network inputs, in noise units
    gate: float = 10.0  # scale of the low-signal gate of g, in noise units (g = O(L^2) below it)
    seed: int = 0
    prefilter: bool = True
    det_size: int = 256


QUICK = dict(hidden=32, K=8, batch=1024, steps=150, nuis_steps=60)


# ====================================================================================================== sky + patch sampler
def _gauss1d(sigma_px):
    r = max(1, int(math.ceil(3 * sigma_px)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2)
    return k / k.sum()


def _blur(x, k):
    """Separable blur of x [1,C,H,W] with 1-d kernel k (zero padding)."""
    C = x.shape[1]
    r = (k.numel() - 1) // 2
    kx = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    x = F.conv2d(x, kx, padding=(0, r), groups=C)
    return F.conv2d(x, ky, padding=(r, 0), groups=C)


class SkyTensor:
    """Frozen sky on the device. vals [1,nb,H,W] (0 where uncovered, divided by `scale`), mask [1,nb,H,W] in {0,1}."""

    def __init__(self, sky: FrozenSky, device, pitch_arcsec=0.3, prefilter=True, scale=1.0):
        data = torch.from_numpy(np.ascontiguousarray(sky.data))[None]
        mask = torch.isfinite(data).float()
        v = torch.nan_to_num(data, nan=0.0) * mask
        if prefilter:
            s = pitch_arcsec / sky.scale_arcsec / math.sqrt(12.0)  # std of a box of width = pitch
            if s > 0.3:
                k = _gauss1d(s)
                v = _blur(v, k) / _blur(mask, k).clamp(min=1e-6) * mask
        self.names = list(sky.names)
        self.nb = len(self.names)
        self.scale_arcsec = sky.scale_arcsec
        self.vals = (v / scale).to(device)
        self.mask = mask.to(device)
        self.H, self.W = int(v.shape[2]), int(v.shape[3])
        self.cat = torch.cat([self.vals, self.mask], 1)  # one grid_sample for values and mask
        self.device = device

    def sample(self, pos, J, offs, thr=0.999):
        """pos [B,2] (x=col, y=row of the sky grid, 0-based), J [B,4] = (dx/du, dx/dv, dy/du, dy/dv) with (u, v) the detector
        (col, row) offsets, offs [Q,2] detector offsets (u, v) in detector pixels. Returns vals [B,nb,Q], valid [B,nb,Q] (bool)."""
        sx = pos[:, 0:1] + J[:, 0:1] * offs[None, :, 0] + J[:, 1:2] * offs[None, :, 1]
        sy = pos[:, 1:2] + J[:, 2:3] * offs[None, :, 0] + J[:, 3:4] * offs[None, :, 1]
        grid = torch.stack([2.0 * sx / (self.W - 1) - 1.0, 2.0 * sy / (self.H - 1) - 1.0], -1)[None]  # [1,B,Q,2]
        out = F.grid_sample(self.cat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]  # [2nb,B,Q]
        out = out.permute(1, 0, 2)
        valid = out[:, self.nb:] > thr
        vals = torch.where(valid, out[:, : self.nb], torch.zeros_like(out[:, : self.nb]))
        return vals, valid


def patch_offsets(P, pitch_arcsec, pix_arcsec=IRAC_PIX_ARCSEC):
    """Detector-frame sample offsets (u = col, v = row) in detector pixels, row-major over (v, u): [P*P, 2]."""
    c = (P - 1) / 2.0
    a = (torch.arange(P, dtype=torch.float32) - c) * pitch_arcsec / pix_arcsec
    vv, uu = torch.meshgrid(a, a, indexing="ij")
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], 1)


def init_kernel(names, P, pitch, nb, sigma_arcsec=0.72, pix=IRAC_PIX_ARCSEC):
    """Warm-start kernel [2, nb, P*P]: Gaussian (FWHM 1.7 arcsec) ⊗ 1.2 arcsec box, unit sum, on the matching band of each channel."""
    o = patch_offsets(P, pitch, pix) * pix  # arcsec
    sub = torch.tensor([-0.45, -0.15, 0.15, 0.45])
    sx, sy = torch.meshgrid(sub, sub, indexing="ij")
    pts = o[None] + torch.stack([sx.reshape(-1), sy.reshape(-1)], 1)[:, None]
    g = torch.exp(-0.5 * (pts ** 2).sum(-1) / sigma_arcsec ** 2).mean(0)
    g = g / g.sum()
    k = torch.zeros(2, nb, P * P)
    for ch in (0, 1):
        b = names.index(MATCH_BAND[ch]) if MATCH_BAND[ch] in names else 0
        k[ch, b] = g
    return k


# ====================================================================================================== IRAC frames
def _frame_geometry(w, ny, nx, sky: FrozenSky):
    """Sky-grid position [N,2] and Jacobian [N,4] (per pixel, central differences of the header WCS)."""
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    X, Y = xx.ravel(), yy.ravel()

    def px(dx, dy):
        ra, dec = w.all_pix2world(X + dx, Y + dy, 0)
        return sky.world_to_px(ra, dec)

    x0, y0 = px(0, 0)
    xr, yr = px(0.5, 0)
    xl, yl = px(-0.5, 0)
    xu, yu = px(0, 0.5)
    xd, yd = px(0, -0.5)
    J = np.stack([xr - xl, xu - xd, yr - yl, yu - yd], 1)
    return np.stack([x0, y0], 1).astype(np.float32), J.astype(np.float32)


def spike_mask(d, unc, good, snr=8.0, ratio=0.12):
    """Data-only mask of single-pixel spikes (cosmic rays / hot pixels the bimsk fatal bits miss): pixel more than `snr` sigma above
    the 7 x 7 running median while the median of its 8 neighbours is below `ratio` of its own significance (a real IRAC source,
    FWHM ~1.4 px, has neighbours at >= ~20 percent of its peak). Independent of the sky and of the models."""
    from scipy.ndimage import median_filter

    ny, nx = d.shape
    fill = np.where(good, d, np.median(d[good]) if good.any() else 0.0)
    z = (d - median_filter(fill, size=7, mode="nearest")) / unc
    pad = np.pad(z, 1, mode="edge")
    nb = np.stack([pad[1 + dy: 1 + dy + ny, 1 + dx: 1 + dx + nx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    return good & (z > snr) & (np.median(nb, 0) < ratio * z)


def read_irac_frame(cbcd, sky: FrozenSky, spike_snr=8.0):
    """One IRAC frame as a dict (numpy, flat row-major pixel arrays). cbcd/cbunc MJy/sr, bimsk fatal bits 32520."""
    unc_p, msk_p = cbcd.replace("_cbcd.fits", "_cbunc.fits"), cbcd.replace("_cbcd.fits", "_bimsk.fits")
    with fits.open(cbcd, memmap=False) as h:
        hdr = h[0].header
        d = h[0].data.astype(np.float32)
    if str(hdr.get("BUNIT", "")).strip() != "MJy/sr":
        raise ValueError(f"{cbcd}: BUNIT is not MJy/sr")
    unc = fits.getdata(unc_p).astype(np.float32)
    bim = fits.getdata(msk_p).astype(np.int64) & 0xFFFF
    ny, nx = d.shape
    w = WCS(hdr)
    pos, J = _frame_geometry(w, ny, nx, sky)
    good = np.isfinite(d) & np.isfinite(unc) & (unc > 0) & ((bim & IRAC_FATAL_BITS) == 0)
    n_spike = 0
    if spike_snr:
        sp = spike_mask(np.nan_to_num(d, nan=0.0), np.where(good, unc, 1.0), good, spike_snr)
        n_spike = int(sp.sum())
        good = good & ~sp
    stem = os.path.basename(cbcd).replace("_cbcd.fits", "")
    parts = stem.split("_")
    frame = dict(
        name=stem, path=cbcd, aor=str(int(hdr.get("AORKEY", parts[2]))), program=str(hdr.get("PROGID", "")),
        ch=int(hdr["CHNLNUM"]) - 1, mjd=float(hdr["MJD_OBS"]), texp=float(hdr.get("EXPTIME", np.nan)), dce=int(parts[3]),
        ny=ny, nx=nx, d=np.nan_to_num(d, nan=0.0).ravel(), unc=np.where(np.isfinite(unc) & (unc > 0), unc, 1.0).ravel(),
        good=good.ravel(), n_spike=n_spike, pos=pos, J=J, row=np.repeat(np.arange(ny), nx), col=np.tile(np.arange(nx), ny))
    return frame


def discover_irac(root, aors=None, channels=(1, 2), max_per_aor=None):
    """cbcd files under root/<aor>/ (also root/*.fits). Frames are taken evenly spaced if max_per_aor is set (per AOR and channel)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "SPITZER_I*_cbcd.fits")) + glob.glob(os.path.join(root, "SPITZER_I*_cbcd.fits"))):
        m = re.match(r"SPITZER_I(\d)_(\d+)_(\d+)_", os.path.basename(f))
        if not m or int(m.group(1)) not in channels:
            continue
        if aors and m.group(2) not in aors:
            continue
        if not (os.path.exists(f.replace("_cbcd", "_cbunc")) and os.path.exists(f.replace("_cbcd", "_bimsk"))):
            continue
        out.setdefault((m.group(2), int(m.group(1))), []).append(f)
    files = []
    for k in sorted(out):
        fl = sorted(out[k])
        if max_per_aor and len(fl) > max_per_aor:
            idx = np.unique(np.round(np.linspace(0, len(fl) - 1, max_per_aor)).astype(int))
            fl = [fl[i] for i in idx]
        files += fl
    return files


def add_time_features(frames):
    """Time since the previous available frame of the same (AOR, channel), sorted by DCE index (log feature in [0, 1])."""
    groups = {}
    for fr in frames:
        groups.setdefault((fr["aor"], fr["ch"]), []).append(fr)
    for g in groups.values():
        g.sort(key=lambda f: f["dce"])
        for i, fr in enumerate(g):
            fr["dt"] = (fr["mjd"] - g[i - 1]["mjd"]) * 86400.0 if i > 0 else float("nan")


def frame_tables(frames, device):
    """Per-frame metadata tensors (index = frame['gid'])."""
    def t(x):
        return torch.tensor(np.asarray(x, np.float32), device=device)

    dt = np.array([fr.get("dt", np.nan) for fr in frames], np.float32)
    dtf = np.where(np.isfinite(dt), np.log1p(np.clip(dt, 0, 3600.0)) / math.log1p(3600.0), 1.0)
    return dict(
        ch=torch.tensor([fr["ch"] for fr in frames], device=device, dtype=torch.long),
        tau=t([(fr["mjd"] - MJD_REF) / MJD_SCALE for fr in frames]),
        ltexp=t([math.log(max(fr["texp"], 1e-3)) / 5.0 for fr in frames]),
        dcef=t([math.log1p(fr["dce"]) / math.log1p(300.0) for fr in frames]),
        dtf=t(dtf), det_n=t([fr["nx"] for fr in frames]))


def prepare_frames(frames, skyT: SkyTensor, cfg: Cfg, init_k, unit, device, chunk=4096):
    """Scale to noise units, compute `usable` (good & whole patch valid in the required bands) and `proxy` (init-kernel prediction).
    Everything done per frame in chunks; the patch is never stored."""
    names = skyT.names
    req = [names.index(b) for b in cfg.require_bands if b in names] or list(range(skyT.nb))
    offs = patch_offsets(cfg.patch, cfg.pitch).to(device)
    ik = init_k.to(device)
    for fr in frames:
        fr["d"] = fr["d"] / unit
        fr["unc"] = fr["unc"] / unit
        pos = torch.from_numpy(fr["pos"]).to(device)
        J = torch.from_numpy(fr["J"]).to(device)
        N = pos.shape[0]
        # centre pre-filter
        _, vc = skyT.sample(pos, J, torch.zeros(1, 2, device=device))
        cand = torch.nonzero(vc[:, req, 0].all(1) & torch.from_numpy(fr["good"]).to(device)).squeeze(1)
        usable = torch.zeros(N, dtype=torch.bool, device=device)
        proxy = torch.full((N,), float("nan"), device=device)
        for s in range(0, cand.numel(), chunk):
            ii = cand[s: s + chunk]
            v, ok = skyT.sample(pos[ii], J[ii], offs)
            okr = ok[:, req].reshape(ii.numel(), -1).all(1)
            prox = (v * ik[fr["ch"]][None]).sum((1, 2))
            usable[ii[okr]] = True
            proxy[ii[okr]] = prox[okr]
        fr["usable"] = usable.cpu().numpy()
        fr["proxy"] = proxy.cpu().numpy()
        fr["n_usable"] = int(fr["usable"].sum())
        fr["b0"] = float(np.median(fr["d"][fr["usable"]] - fr["proxy"][fr["usable"]])) if fr["n_usable"] else 0.0


class Pool:
    """Flat pixel pool (device tensors) over selected frames. parity: None = all usable, 0 = fit half, 1 = score half."""

    def __init__(self, frames, device, parity=None):
        cols = {k: [] for k in ("gid", "pix", "pos", "J", "d", "unc", "xy", "proxy")}
        for fr in frames:
            m = fr["usable"].copy()
            if parity is not None:
                m &= ((fr["row"] + fr["col"]) % 2 == parity)
            ii = np.nonzero(m)[0]
            if ii.size == 0:
                continue
            cols["gid"].append(np.full(ii.size, fr["gid"], np.int64))
            cols["pix"].append(ii.astype(np.int64))
            cols["pos"].append(fr["pos"][ii]), cols["J"].append(fr["J"][ii]), cols["d"].append(fr["d"][ii])
            cols["unc"].append(fr["unc"][ii]), cols["proxy"].append(fr["proxy"][ii])
            n = fr["nx"]
            cols["xy"].append(np.stack([(fr["col"][ii] - (n - 1) / 2) / ((n - 1) / 2), (fr["row"][ii] - (fr["ny"] - 1) / 2) / ((fr["ny"] - 1) / 2)], 1))
        self.N = int(sum(len(c) for c in cols["gid"]))
        self.device = device
        for k, v in cols.items():
            arr = np.concatenate(v) if v else np.zeros((0,))
            setattr(self, k, torch.from_numpy(arr.astype(np.int64 if k in ("gid", "pix") else np.float32)).to(device))
        self.unc = self.unc if self.N else self.unc
        self.bright_idx = torch.nonzero((self.proxy / self.unc.clamp(min=1e-6)) > 5.0).squeeze(1) if self.N else self.gid

    def get(self, sel):
        return dict(gid=self.gid[sel], pos=self.pos[sel], J=self.J[sel], d=self.d[sel], unc=self.unc[sel], xy=self.xy[sel])


# ====================================================================================================== models
def mlp(i, h, o, n_hidden=2, zero_last=True):
    layers, d = [], i
    for _ in range(n_hidden):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    last = nn.Linear(d, o)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


def fourier_xy(xy):
    x, y = xy[:, 0:1], xy[:, 1:2]
    f = [x, y]
    for k in (1, 2):
        f += [torch.sin(k * math.pi * x), torch.cos(k * math.pi * x), torch.sin(k * math.pi * y), torch.cos(k * math.pi * y)]
    return torch.cat(f, 1)  # 10


class Instrument(nn.Module):
    """Common frame tables and nuisance handling."""

    def __init__(self, nF, tables):
        super().__init__()
        for k, v in tables.items():
            self.register_buffer("t_" + k, v.clone())
        self.nF = nF

    def meta(self, gid):
        return self.t_ch[gid], self.t_tau[gid]

    def set_tables(self, tables):
        for k, v in tables.items():
            setattr(self, "t_" + k, v.clone().to(self.frame_b.weight.device))

    def nuisance_modules(self):
        raise NotImplementedError

    def reset_nuisance(self, frames):
        with torch.no_grad():
            for fr in frames:
                self.frame_b.weight[fr["gid"], 0] = fr["b0"]
            for m in self.nuisance_modules()[1:]:
                for fr in frames:
                    m.weight[fr["gid"]] = 0.0

    def reg(self):
        return 0.0

    def loss_terms(self, out, d, unc, kind, delta):
        r = d - out["pred"]
        z = r / out["sigma"]
        if kind == "huber":
            a = z.abs()
            core = torch.where(a <= delta, 0.5 * z * z, delta * (a - 0.5 * delta))
        else:
            core = 0.5 * z * z
        return core + torch.log(out["sigma"])


class LearnedInstrument(Instrument):
    kind = "learned"

    def __init__(self, nb, nF, cfg: Cfg, tables, ik):
        super().__init__(nF, tables)
        P2 = cfg.patch ** 2
        self.nb, self.P, self.P2, self.cfg = nb, cfg.patch, P2, cfg
        self.K, self.dz = cfg.K, cfg.dz
        self.base = nn.Parameter(ik.clone())  # [2, nb, P2]
        self.basis = nn.Parameter(torch.randn(cfg.K, nb * P2) * 0.01)
        self.hyper = mlp(10 + 2 + 1 + cfg.dz, cfg.hidden, cfg.K)
        self.frame_b = nn.Embedding(nF, 1)
        self.frame_z = nn.Embedding(nF, cfg.dz)
        nn.init.zeros_(self.frame_b.weight)
        nn.init.normal_(self.frame_z.weight, std=0.0)
        gin = 1 + 4 * nb + 5 + 2 + cfg.dz
        self.g = mlp(gin, cfg.hidden, 1)
        self.var = mlp(gin, cfg.hidden, 1)
        self.register_buffer("centre", torch.tensor([(cfg.patch ** 2 - 1) // 2]))
        self._g_pen = 0.0

    def nuisance_modules(self):
        return [self.frame_b, self.frame_z]

    def _rough(self, W):
        """Sum of squared second differences (Laplacian) of kernels W [n, P, P]."""
        lap = W[:, 1:-1, 1:-1] * 4 - W[:, :-2, 1:-1] - W[:, 2:, 1:-1] - W[:, 1:-1, :-2] - W[:, 1:-1, 2:]
        return (lap ** 2).sum()

    def reg(self):
        c = self.cfg
        r = c.z_l2 * (self.frame_z.weight ** 2).sum(1).mean()
        if c.smooth > 0 or c.bank_l2 > 0:
            e0 = (self.base ** 2).sum().detach() + 1e-8
            Wb = self.base.reshape(-1, self.P, self.P)
            Bk = self.basis.reshape(-1, self.P, self.P)
            r = r + c.smooth * (self._rough(Wb) + self._rough(Bk)) / e0 + c.bank_l2 * (self.basis ** 2).sum() / e0
        return r + c.g_l2 * self._g_pen

    def _coef(self, ch, xy, tau, z):
        return self.hyper(torch.cat([fourier_xy(xy), F.one_hot(ch, 2).float(), tau[:, None], z], 1))

    def linear(self, vals, ch, xy, tau, z):
        """L = W(cond) . patch, W linear bank, normalised to unit total weight (flat sky in = flat sky out; gauge of the scale of L,
        so that the nonlinear part g cannot trade a global gain against the kernel). vals [B,nb,P2]."""
        B = vals.shape[0]
        v = vals.reshape(B, -1)
        coef = self._coef(ch, xy, tau, z)
        L0 = (v * self.base[ch].reshape(B, -1)).sum(1)
        raw = L0 + ((v @ self.basis.T) * coef).sum(1)
        norm = self.base.sum((1, 2))[ch] + (coef @ self.basis.sum(1))
        return raw / norm.clamp(min=0.2)

    def effective_kernel(self, ch, xy, tau, z=None):
        """W [nb, P2] at one condition (xy: 2-vector in [-1,1]): the linear part as a kernel (unit total weight)."""
        dev = self.base.device
        xy = torch.as_tensor(xy, dtype=torch.float32, device=dev)[None]
        ch_t = torch.tensor([ch], device=dev)
        z = torch.zeros(1, self.dz, device=dev) if z is None else z
        coef = self._coef(ch_t, xy, torch.tensor([tau], dtype=torch.float32, device=dev), z)
        W = self.base[ch] + (coef @ self.basis).reshape(self.nb, self.P2)
        return (W / W.sum().clamp(min=0.2)).detach()

    def summary(self, vals, valid):
        sa = self.cfg.sa
        c = vals[:, :, self.centre[0]]
        vf = valid.float()
        mean = (vals.sum(2) / vf.sum(2).clamp(min=1.0))
        mx = vals.max(2).values
        return torch.cat([torch.asinh(c / sa), torch.asinh(mean / sa), torch.asinh(mx / sa), vf.mean(2)], 1)

    def forward(self, vals, valid, gid, xy, unc):
        sa = self.cfg.sa
        ch, tau = self.meta(gid)
        z = self.frame_z(gid)
        L = self.linear(vals, ch, xy, tau, z)
        b = self.frame_b(gid)[:, 0]
        gin = torch.cat([torch.asinh(L / sa)[:, None], self.summary(vals, valid), self.t_ltexp[gid][:, None],
                         self.t_dcef[gid][:, None], self.t_dtf[gid][:, None], xy, F.one_hot(ch, 2).float(), z], 1)
        gate = L * L / (L * L + self.cfg.gate ** 2)  # low-signal gauge: g = O(L^2) for |L| << sa, so the response to faint sky is L alone
        o = self.g(gin)[:, 0]
        g = gate * (sa + L.abs()) * o
        h = 3.0 * torch.tanh(self.var(gin)[:, 0] / 3.0)
        self._g_pen = (gate * o).pow(2).mean() if self.training else 0.0
        return dict(pred=L + b + g, src=L + g, L=L, g=g, b=b, sigma=unc * torch.exp(h))

    def param_groups(self, cfg):
        kern = [self.base, self.basis]
        ids = {id(p) for p in kern}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [dict(params=kern, lr=cfg.lr_kernel), dict(params=rest, lr=cfg.lr)]

    def counts(self):
        n = lambda ps: int(sum(p.numel() for p in ps))  # noqa: E731
        return {"linear_base_kernels": n([self.base]), "linear_basis_bank": n([self.basis]), "hypernetwork_mlp": n(self.hyper.parameters()),
                "residual_net_g": n(self.g.parameters()), "variance_head": n(self.var.parameters()),
                "per_frame_b": n(self.frame_b.parameters()), "per_frame_z": n(self.frame_z.parameters()),
                "total": n(self.parameters())}


class StructuredInstrument(Instrument):
    kind = "structured"

    def __init__(self, nb, nF, cfg: Cfg, tables, ik):
        super().__init__(nF, tables)
        self.nb, self.P, self.P2, self.cfg = nb, cfg.patch, cfg.patch ** 2, cfg
        M, n = cfg.n_mix, cfg.psf_nodes
        self.M, self.n = M, n
        self.psf = mlp(10 + 2, cfg.hidden, 4 * M, zero_last=False)
        with torch.no_grad():
            self.psf[-1].weight.mul_(0.02)
            sig0 = torch.tensor([0.5, 0.8, 1.4, 2.5])[:M] if M <= 4 else torch.linspace(0.5, 3.0, M)
            pi0 = torch.tensor([0.35, 0.35, 0.2, 0.1])[:M] if M <= 4 else torch.ones(M) / M
            bias = torch.zeros(M, 4)
            bias[:, 0] = torch.log(pi0)
            bias[:, 1] = torch.log(sig0)
            bias[:, 2] = torch.log(sig0)
            self.psf[-1].bias.copy_(bias.reshape(-1))
        col = torch.zeros(2, nb)
        for c in (0, 1):
            col[c] = ik[c].sum(1)  # 1 on the matched band
        self.colour = nn.Parameter(col)
        self.frame_b = nn.Embedding(nF, 1)
        self.frame_lg = nn.Embedding(nF, 1)
        nn.init.zeros_(self.frame_b.weight)
        nn.init.zeros_(self.frame_lg.weight)
        xs = torch.linspace(-1, 1, n)
        gy, gx = torch.meshgrid(xs, xs, indexing="ij")  # [n, n]
        self.register_buffer("nodes", torch.stack([gx.reshape(-1), gy.reshape(-1)], 1))
        o = patch_offsets(cfg.patch, cfg.pitch) * IRAC_PIX_ARCSEC
        sub = torch.tensor([-0.45, -0.15, 0.15, 0.45])
        sx, sy = torch.meshgrid(sub, sub, indexing="ij")
        pts = o[None] + torch.stack([sx.reshape(-1), sy.reshape(-1)], 1)[:, None]  # [16, P2, 2]
        self.register_buffer("pts", pts)

    def nuisance_modules(self):
        return [self.frame_b, self.frame_lg]

    def kernels(self):
        """Tabulated kernel field [2, n, n, P2] (unit sum, zero centroid, PSF ⊗ pixel box)."""
        n, M = self.n, self.M
        ks = []
        for ch in (0, 1):
            inp = torch.cat([fourier_xy(self.nodes), F.one_hot(torch.full((n * n,), ch, device=self.nodes.device), 2).float()], 1)
            p = self.psf(inp).reshape(n * n, M, 4)
            pi = torch.softmax(p[..., 0], -1)
            s1 = torch.exp(p[..., 1]).clamp(0.2, 8.0)
            s2 = torch.exp(p[..., 2]).clamp(0.2, 8.0)
            th = p[..., 3]
            u = self.pts[None, None, :, :, 0]  # [1,1,16,P2]
            v = self.pts[None, None, :, :, 1]
            c, s = torch.cos(th)[..., None, None], torch.sin(th)[..., None, None]
            a = c * u + s * v
            b = -s * u + c * v
            q = (a / s1[..., None, None]) ** 2 + (b / s2[..., None, None]) ** 2
            G = (pi[..., None, None] * torch.exp(-0.5 * q) / (2 * math.pi * s1 * s2)[..., None, None]).sum(1)  # [n*n,16,P2]
            k = G.mean(1)
            ks.append((k / k.sum(-1, keepdim=True)).reshape(n, n, self.P2))
        return torch.stack(ks)

    def forward(self, vals, valid, gid, xy, unc, kern=None):
        ch, _ = self.meta(gid)
        kern = self.kernels() if kern is None else kern
        n = self.n
        fx = (xy[:, 0].clamp(-1, 1) + 1) * 0.5 * (n - 1)
        fy = (xy[:, 1].clamp(-1, 1) + 1) * 0.5 * (n - 1)
        x0 = fx.floor().clamp(0, n - 2).long()
        y0 = fy.floor().clamp(0, n - 2).long()
        wx, wy = (fx - x0)[:, None], (fy - y0)[:, None]
        k = ((1 - wx) * (1 - wy) * kern[ch, y0, x0] + wx * (1 - wy) * kern[ch, y0, x0 + 1]
             + (1 - wx) * wy * kern[ch, y0 + 1, x0] + wx * wy * kern[ch, y0 + 1, x0 + 1])
        mix = (vals * self.colour[ch][:, :, None]).sum(1)
        L = torch.exp(self.frame_lg(gid)[:, 0]) * (mix * k).sum(1)
        b = self.frame_b(gid)[:, 0]
        return dict(pred=L + b, src=L, L=L, g=torch.zeros_like(L), b=b, sigma=unc)

    def param_groups(self, cfg):
        return [dict(params=list(self.parameters()), lr=cfg.lr)]

    def counts(self):
        n = lambda ps: int(sum(p.numel() for p in ps))  # noqa: E731
        return {"psf_mlp": n(self.psf.parameters()), "colour": n([self.colour]), "per_frame_b": n(self.frame_b.parameters()),
                "per_frame_log_gain": n(self.frame_lg.parameters()), "total": n(self.parameters())}


def build_model(kind, nb, nF, cfg, tables, ik):
    return (LearnedInstrument if kind == "learned" else StructuredInstrument)(nb, nF, cfg, tables, ik)


# ====================================================================================================== training
def run_model(model, skyT, offs, batch, kern_cache=None):
    vals, valid = skyT.sample(batch["pos"], batch["J"], offs)
    if isinstance(model, StructuredInstrument):
        return model(vals, valid, batch["gid"], batch["xy"], batch["unc"], kern=kern_cache)
    return model(vals, valid, batch["gid"], batch["xy"], batch["unc"])


def draw(pool, B, bright_frac, gen):
    n_b = int(B * bright_frac) if pool.bright_idx.numel() > 0 else 0
    sel = torch.randint(0, pool.N, (B - n_b,), device=pool.device, generator=gen)
    if n_b:
        sel = torch.cat([sel, pool.bright_idx[torch.randint(0, pool.bright_idx.numel(), (n_b,), device=pool.device, generator=gen)]])
    return sel


def train(model, pool, skyT, cfg: Cfg, budget_s, out_dir, fh, resume=False, val_fn=None):
    dev = pool.device
    offs = patch_offsets(cfg.patch, cfg.pitch).to(dev)
    ck = os.path.join(out_dir, "ckpt.pt")
    step0, el0, hist = 0, 0.0, []
    if resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"])
        step0, el0, hist = st["step"], st["elapsed"], st.get("hist", [])
        log(f"resumed {ck} at step {step0}", fh)
    opt = torch.optim.AdamW(model.param_groups(cfg), weight_decay=0.0)
    base_lrs = [g["lr"] for g in opt.param_groups]
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed + step0)
    t0, last_ck, step = time.time(), time.time(), step0
    best = (float("inf"), None)
    model.train()
    while step < cfg.steps:
        el = el0 + time.time() - t0
        if budget_s and el >= budget_s:
            log(f"budget reached at step {step}", fh)
            break
        prog = max(step / cfg.steps, (el / budget_s) if budget_s else 0.0)
        f = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        for g, l0 in zip(opt.param_groups, base_lrs):
            g["lr"] = l0 * f
        sel = draw(pool, cfg.batch, cfg.bright_frac, gen)
        bt = pool.get(sel)
        out = run_model(model, skyT, offs, bt)
        loss = model.loss_terms(out, bt["d"], bt["unc"], cfg.loss, cfg.huber_delta).mean() + model.reg()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        step += 1
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                chi = float((((bt["d"] - out["pred"]) / bt["unc"]) ** 2).mean())
            hist.append((step, float(loss.detach()), chi))
            if step % 100 == 0 or step == 1:
                log(f"step {step}  loss {float(loss.detach()):.4f}  chi2(cbunc) {chi:.3f}  lr x{f:.2f}  {(time.time() - t0) / max(step - step0, 1):.3f} s/step", fh)
        if val_fn is not None and cfg.val_every and step % cfg.val_every == 0:
            v = val_fn()
            log(f"  val chi2_clip25 {v:.4f}", fh)
            if v < best[0]:
                best = (v, {k: t.detach().clone() for k, t in model.state_dict().items()})
            model.train()
        if time.time() - last_ck > 120:
            torch.save(dict(model=model.state_dict(), step=step, elapsed=el0 + time.time() - t0, hist=hist, cfg=asdict(cfg)), ck)
            last_ck = time.time()
    if best[1] is not None:
        model.load_state_dict(best[1])
        log(f"restored best-validation weights ({best[0]:.4f})", fh)
    torch.save(dict(model=model.state_dict(), step=step, elapsed=el0 + time.time() - t0, hist=hist, cfg=asdict(cfg)), ck)
    return dict(steps=step, train_seconds=el0 + time.time() - t0, samples_per_s=cfg.batch * (step - step0) / max(time.time() - t0, 1e-9), hist=hist)


def fit_nuisance(model, frames, skyT, cfg: Cfg, steps, dev, fh=None):
    """Fit b_i and z_i (or log-gain) of held-out frames on the fit half of their usable pixels; shared weights frozen."""
    model.reset_nuisance(frames)
    pool = Pool(frames, dev, parity=0)
    if pool.N == 0 or steps <= 0:
        return
    offs = patch_offsets(cfg.patch, cfg.pitch).to(dev)
    saved = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    nm = model.nuisance_modules()
    for m in nm:
        m.weight.requires_grad_(True)
    opt = torch.optim.Adam([dict(params=[nm[0].weight], lr=0.05), dict(params=[nm[1].weight], lr=0.02)])
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed + 1)
    kern = None
    model.eval()
    for s in range(steps):
        sel = torch.randint(0, pool.N, (min(cfg.batch, pool.N),), device=dev, generator=gen)
        bt = pool.get(sel)
        kern = model.kernels().detach() if isinstance(model, StructuredInstrument) else None
        out = run_model(model, skyT, offs, bt, kern)
        loss = model.loss_terms(out, bt["d"], bt["unc"], cfg.loss, cfg.huber_delta).mean() + model.reg()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    for p, rg in saved:
        p.requires_grad_(rg)


@torch.no_grad()
def predict_frame(model, fr, skyT, cfg: Cfg, dev, chunk=4096):
    """Images [ny,nx] (NaN where not usable) of data, unc, pred, src (= L + g), sigma, usable; noise units."""
    offs = patch_offsets(cfg.patch, cfg.pitch).to(dev)
    N = fr["nx"] * fr["ny"]
    ii = np.nonzero(fr["usable"])[0]
    out_all = {k: np.full(N, np.nan, np.float32) for k in ("pred", "src", "sigma", "L", "g")}
    kern = model.kernels() if isinstance(model, StructuredInstrument) else None
    n = fr["nx"]
    for s in range(0, ii.size, chunk):
        j = ii[s: s + chunk]
        bt = dict(gid=torch.full((j.size,), fr["gid"], device=dev, dtype=torch.long), pos=torch.from_numpy(fr["pos"][j]).to(dev),
                  J=torch.from_numpy(fr["J"][j]).to(dev), unc=torch.from_numpy(fr["unc"][j]).to(dev),
                  xy=torch.from_numpy(np.stack([(fr["col"][j] - (n - 1) / 2) / ((n - 1) / 2), (fr["row"][j] - (fr["ny"] - 1) / 2) / ((fr["ny"] - 1) / 2)], 1).astype(np.float32)).to(dev))
        o = run_model(model, skyT, offs, bt, kern)
        for k in out_all:
            out_all[k][j] = o[k].cpu().numpy()
    shp = (fr["ny"], fr["nx"])
    rec = {k: v.reshape(shp) for k, v in out_all.items()}
    us = fr["usable"].reshape(shp)
    rec["d"] = np.where(us, fr["d"].reshape(shp), np.nan)
    rec["unc"] = np.where(us, fr["unc"].reshape(shp), np.nan)
    rec["usable"] = us
    rec["score"] = us & (((fr["row"] + fr["col"]) % 2 == 1).reshape(shp))
    rec.update(ch=fr["ch"], name=fr["name"], aor=fr["aor"])
    return rec


def evaluate(model, frames, skyT, cfg, dev, fh=None, nuis_steps=None):
    fit_nuisance(model, frames, skyT, cfg, cfg.nuis_steps if nuis_steps is None else nuis_steps, dev, fh)
    model.eval()
    return [predict_frame(model, fr, skyT, cfg, dev) for fr in frames]


# ====================================================================================================== metrics (one code path)
def _pix_stats(r, unc, sig):
    z = r / unc
    zs = r / sig
    out = dict(n=int(z.size))
    if z.size == 0:
        return out
    mad = 1.4826 * np.median(np.abs(z - np.median(z)))
    out.update(chi2_mean=float(np.mean(z ** 2)), chi2_median=float(np.median(z ** 2) / 0.4549), chi2_mad=float(mad ** 2),
               chi2_clip25=float(np.mean(np.minimum(z ** 2, 25.0))), chi2_sigmodel_mean=float(np.mean(zs ** 2)),
               nll_sigmodel=float(np.mean(0.5 * zs ** 2 + np.log(sig))), mean_resid_sigma=float(np.mean(z)))
    return out


def _local_max(img, size=3):
    t = torch.from_numpy(np.where(np.isfinite(img), img, -np.inf).astype(np.float32))[None, None]
    return (F.max_pool2d(t, size, 1, size // 2)[0, 0].numpy() == t[0, 0].numpy()) & np.isfinite(img)


def aperture_errors(rec, snr_min=20.0, r_ap=3.0, r_in=3.5, r_out=6.0, iso_r=7.0, iso_snr=5.0, psf_ratio=0.12):
    """Relative aperture flux error (F_pred - F_data) / F_data on isolated bright sources of one frame (see module docstring). A peak must be
    resolved at the IRAC PSF scale: the median S/N of its 8 neighbours >= psf_ratio * its own S/N (rejects single-pixel spikes)."""
    d, unc, pred, us = rec["d"], rec["unc"], rec["pred"], rec["usable"]
    if not us.any():
        return []
    med = np.nanmedian(d[us])
    snr = np.where(us, (d - med) / unc, np.nan)
    lm = _local_max(snr)
    peaks = np.argwhere(lm & (snr > snr_min))
    others = lm & (snr > iso_snr)
    R = int(math.ceil(iso_r))
    yy, xx = np.mgrid[-R: R + 1, -R: R + 1]
    rr = np.hypot(yy, xx)
    ny, nx = d.shape
    res = []
    for (r0, c0) in peaks:
        if r0 < R or c0 < R or r0 >= ny - R or c0 >= nx - R:
            continue
        sl = (slice(r0 - R, r0 + R + 1), slice(c0 - R, c0 + R + 1))
        ap = rr <= r_ap
        ann = (rr >= r_in) & (rr <= r_out)
        u = us[sl]
        if not u[ap].all() or u[ann].mean() < 0.6:
            continue
        nbr = snr[r0 - 1: r0 + 2, c0 - 1: c0 + 2].ravel()
        if not np.nanmedian(np.delete(nbr, 4)) >= psf_ratio * snr[r0, c0]:
            continue
        oth = others[sl] & (rr <= iso_r)
        oth[R, R] = False
        if oth.any():
            continue
        ds, ps = d[sl], pred[sl]
        a = ann & u
        Fd = float((ds[ap] - np.median(ds[a])).sum())
        Fm = float((ps[ap] - np.median(ps[a])).sum())
        sF = float(np.sqrt((unc[sl][ap] ** 2).sum()))
        if Fd > 0:
            res.append(dict(frame=rec["name"], ch=rec["ch"], snr_peak=float(snr[r0, c0]), rel_err=(Fm - Fd) / Fd, pull=(Fm - Fd) / sF))
    return res


def compute_metrics(recs, snr_bright=5.0):
    """Metrics for a list of frame records (output of predict_frame). Same code for every model."""
    out = {}
    for name, chs in (("all", (0, 1)), ("IRAC1", (0,)), ("IRAC2", (1,))):
        rr = [r for r in recs if r["ch"] in chs]
        if not rr:
            continue
        sc = [r["score"] for r in rr]
        resid = np.concatenate([(r["d"] - r["pred"])[m] for r, m in zip(rr, sc)])
        unc = np.concatenate([r["unc"][m] for r, m in zip(rr, sc)])
        sig = np.concatenate([r["sigma"][m] for r, m in zip(rr, sc)])
        src = np.concatenate([r["src"][m] for r, m in zip(rr, sc)])
        bright = (src / unc) > snr_bright
        ap = [e for r in rr for e in aperture_errors(r)]
        a = {"n_sources": len(ap)}
        if ap:
            re_ = np.array([e["rel_err"] for e in ap])
            pu = np.array([e["pull"] for e in ap])
            a.update(median_rel_err=float(np.median(re_)), median_abs_rel_err=float(np.median(np.abs(re_))),
                     rms_rel_err=float(np.sqrt(np.mean(re_ ** 2))), rms_pull=float(np.sqrt(np.mean(pu ** 2))),
                     median_snr_peak=float(np.median([e["snr_peak"] for e in ap])))
        out[name] = dict(all=_pix_stats(resid, unc, sig), bright=_pix_stats(resid[bright], unc[bright], sig[bright]), aperture=a)
    return out


def save_png(rec, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    us = rec["usable"]
    if us.any():  # crop to the usable region (JWST coverage is partial)
        ys, xs = np.nonzero(us)
        sl = (slice(max(ys.min() - 3, 0), ys.max() + 4), slice(max(xs.min() - 3, 0), xs.max() + 4))
        us = us[sl]
        rec = dict(rec, d=rec["d"][sl], pred=rec["pred"][sl], unc=rec["unc"][sl])
    d, p, unc = rec["d"], rec["pred"], rec["unc"]
    res = (d - p) / unc
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    v = np.concatenate([d[us], p[us]]) if us.any() else np.zeros(1)
    lo, hi = np.percentile(v, [1, 99.7])
    for a, im, t in ((ax[0], d, "data"), (ax[1], p, "prediction (L + b + g)")):
        h = a.imshow(np.arcsinh(np.where(us, im, np.nan) / max(hi - lo, 1e-6) * 5), origin="lower", cmap="magma")
        a.set_title(t)
    h = ax[2].imshow(np.where(us, res, np.nan), origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
    ax[2].set_title("(data - pred) / cbunc")
    fig.colorbar(h, ax=ax[2], fraction=0.046)
    for a in ax:
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(title or rec["name"], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


# ====================================================================================================== split
def _h(s, seed=0):
    return int(hashlib.sha1(f"{seed}:{s}".encode()).hexdigest(), 16)


def make_split(aor_program: dict, seed=0):
    """Deterministic: 2 test AORs + 1 val AOR from three different programs (programs with >= 2 AORs first), rest train."""
    progs = {}
    for a, p in aor_program.items():
        progs.setdefault(p, []).append(a)
    order = sorted(progs, key=lambda p: (len(progs[p]) < 2, _h(p, seed)))
    pick = lambda p: min(progs[p], key=lambda a: _h(a, seed))  # noqa: E731
    if len(order) >= 3:
        test, val = [pick(order[0]), pick(order[1])], [pick(order[2])]
    else:  # fewer than three programs: fall back to distinct AORs by hash
        al = sorted(aor_program, key=lambda a: _h(a, seed))
        if len(al) < 3:
            raise ValueError("need at least 3 AORs for a test/val/train split")
        test, val = al[:2], al[2:3]
    train = sorted(a for a in aor_program if a not in test and a not in val)
    return dict(test=sorted(test), val=sorted(val), train=train, programs=aor_program, seed=seed,
                rule="2 test AORs + 1 val AOR from different programs (programs with >=2 AORs first, sha1(seed:name) order); rest train")


# ====================================================================================================== main pipeline
def count_usable(frames):
    by = {}
    for fr in frames:
        a = by.setdefault(fr["aor"], dict(program=fr["program"], n_frames=0, n_pixels=0, n_usable=0, per_channel={}))
        a["n_frames"] += 1
        a["n_pixels"] += fr["nx"] * fr["ny"]
        a["n_usable"] += fr["n_usable"]
        a["n_spikes_masked"] = a.get("n_spikes_masked", 0) + fr.get("n_spike", 0)
        c = a["per_channel"].setdefault(f"IRAC{fr['ch'] + 1}", dict(n_frames=0, n_frames_with_usable=0, n_usable=0))
        c["n_frames"] += 1
        c["n_frames_with_usable"] += int(fr["n_usable"] > 0)
        c["n_usable"] += fr["n_usable"]
    return by


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sky", required=True, help="jwst_lw_sky.fits")
    ap.add_argument("--irac-root", required=True, help="dir with <aor>/SPITZER_I{1,2}_<aor>_<dce>_*_{cbcd,cbunc,bimsk}.fits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["learned", "structured", "both"], default="learned")
    ap.add_argument("--quick", action="store_true", help="tiny networks, ~150 steps (CPU smoke run)")
    ap.add_argument("--budget-min", type=float, default=0.0, help="training wall-clock per model [min] (0 = steps only)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--patch", type=int, default=25)
    ap.add_argument("--pitch", type=float, default=0.3)
    ap.add_argument("--loss", choices=["nll", "huber"], default="nll")
    ap.add_argument("--nuis-steps", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=0)
    ap.add_argument("--require-bands", default="F356W,F444W")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--aors", default=None, help="comma list; default all found")
    ap.add_argument("--channels", default="1,2")
    ap.add_argument("--max-frames-per-aor", type=int, default=None, help="per AOR and channel, evenly spaced")
    ap.add_argument("--spike-snr", type=float, default=8.0, help="data-only single-pixel spike mask threshold (0 = off)")
    ap.add_argument("--max-png", type=int, default=12)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--count-only", action="store_true", help="only write usable_pixels.json / split.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    cfg = Cfg(patch=a.patch, pitch=a.pitch, loss=a.loss, val_every=a.val_every, require_bands=tuple(a.require_bands.split(",")), seed=a.seed)
    if a.quick:
        for k, v in QUICK.items():
            setattr(cfg, k, v)
    for k in ("steps", "batch", "lr", "hidden", "K"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if a.nuis_steps is not None:
        cfg.nuis_steps = a.nuis_steps
    cfg.budget_min = a.budget_min

    t_all = time.time()
    sky = load_sky(a.sky)
    log(f"sky {sky.names} {sky.shape} at {sky.scale_arcsec:.3f} arcsec; device {dev}")
    files = discover_irac(a.irac_root, set(a.aors.split(",")) if a.aors else None, tuple(int(c) for c in a.channels.split(",")), a.max_frames_per_aor)
    if not files:
        raise SystemExit("no IRAC frames found")
    log(f"reading {len(files)} IRAC frames")
    frames = [read_irac_frame(f, sky, a.spike_snr) for f in files]
    for i, fr in enumerate(frames):
        fr["gid"] = i
    add_time_features(frames)
    unit = float(np.median([np.median(fr["unc"][fr["good"]]) for fr in frames]))
    skyT = SkyTensor(sky, dev, cfg.pitch, cfg.prefilter, scale=unit)
    ik = init_kernel(skyT.names, cfg.patch, cfg.pitch, skyT.nb)
    prepare_frames(frames, skyT, cfg, ik, unit, dev)
    usable = count_usable(frames)

    aor_prog = {fr["aor"]: fr["program"] for fr in frames}
    if a.split_json and os.path.exists(a.split_json):
        split = json.load(open(a.split_json))
    else:
        split = make_split(aor_prog, a.split_seed)
        json.dump(split, open(a.split_json or os.path.join(a.out, "split.json"), "w"), indent=1)
    role = {x: r for r in ("train", "val", "test") for x in split[r]}
    for aor, rec_ in usable.items():
        rec_["role"] = role.get(aor, "unused")
    json.dump(dict(unit_MJy_sr=unit, patch=cfg.patch, pitch_arcsec=cfg.pitch, require_bands=list(cfg.require_bands), aors=usable),
              open(os.path.join(a.out, "usable_pixels.json"), "w"), indent=1)
    for aor, r_ in sorted(usable.items()):
        log(f"AOR {aor} [{r_['role']}] program {r_['program']}: {r_['n_frames']} frames, usable pixels {r_['n_usable']} of {r_['n_pixels']}")
    if a.count_only:
        return
    sel = lambda r: [fr for fr in frames if role.get(fr["aor"]) == r]  # noqa: E731
    tr, va, te = sel("train"), sel("val"), sel("test")
    pool = Pool(tr, dev)
    if pool.N == 0:
        raise SystemExit("no usable training pixels")
    log(f"train pixels {pool.N}, val frames {len(va)}, test frames {len(te)}")
    tables = frame_tables(frames, dev)
    kinds = ["learned", "structured"] if a.model == "both" else [a.model]
    mpath = os.path.join(a.out, "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
    metrics.update(config=asdict(cfg), split=split, usable_pixels=usable, unit_MJy_sr=unit, device=str(dev), n_frames=len(frames))
    metrics.setdefault("models", {})
    for kind in kinds:
        mdir = os.path.join(a.out, kind)
        os.makedirs(mdir, exist_ok=True)
        fh = open(os.path.join(mdir, "train_log.txt"), "a")
        torch.manual_seed(a.seed)
        model = build_model(kind, skyT.nb, len(frames), cfg, tables, ik).to(dev)
        log(f"[{kind}] parameters {model.counts()}", fh)
        with torch.no_grad():  # warm start of nuisance: b_i from the init kernel
            for fr in tr:
                model.frame_b.weight[fr["gid"], 0] = fr["b0"]
        val_fn = None
        if va and cfg.val_every:
            def val_fn():  # noqa: E306
                recs = evaluate(model, va, skyT, cfg, dev, nuis_steps=40)
                m = compute_metrics(recs)["all"]["all"]
                return m.get("chi2_clip25", float("inf"))
        budget = a.budget_min * 60.0 / len(kinds) if a.budget_min else 0.0
        tinfo = train(model, pool, skyT, cfg, budget, mdir, fh, a.resume, val_fn)
        t_ev = time.time()
        res = {}
        for tag, fl in (("val", va), ("test", te)):
            if not fl:
                continue
            recs = evaluate(model, fl, skyT, cfg, dev, fh)
            res[tag] = compute_metrics(recs, cfg.snr_bright)
            if tag == "test":
                for r in recs[: a.max_png]:
                    save_png(r, os.path.join(mdir, f"resid_{r['name']}.png"), f"{kind}  {r['name']}  AOR {r['aor']}")
        log(f"[{kind}] test (all): {json.dumps(res.get('test', {}).get('all', {}).get('all', {}))}", fh)
        metrics["models"][kind] = dict(parameters=model.counts(), train=dict(steps=tinfo["steps"], seconds=tinfo["train_seconds"],
                                       samples_per_s=tinfo["samples_per_s"], loss_hist_tail=tinfo["hist"][-5:]),
                                       eval_seconds=time.time() - t_ev, metrics=res)
        fh.close()
    metrics["wall_seconds"] = time.time() - t_all
    json.dump(metrics, open(mpath, "w"), indent=1, default=float)
    log(f"wrote {mpath}")


if __name__ == "__main__":
    main()
