"""Synthetic fields for the CPU tests, rendered by an INDEPENDENT analytic renderer (no FFT, no kernel field, no level grids).

Truth: point sources and round Gaussian galaxies with power-law spectra; PSF = two concentric Gaussians (core sigma_c(lambda, xi), wing
3 sigma_c, wing fraction f_w); pixel response = the pixel box (rotations are multiples of 90 degrees, so the box stays axis-aligned and
everything is a product of error functions); throughput = T(lambda)/(lambda N) x flat(xi); lambda integral with a dense quantile rule
(n_dense nodes); per-exposure registration offsets that the header WCS does not know; Gaussian noise of known sigma; constant background.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .bandpass import FilterSet, box_filter, quantile_nodes
from .data import Exposure, build_meta, make_patches


@dataclass
class SynthInstrument:
    name: str
    pixel: float
    npix: int
    diameter_m: float
    sigma0: float           # core sigma [arcsec] at lam0
    lam0: float
    space_grad: float = 0.0  # injected spatial PSF variation: sigma_c *= (1 + space_grad xi_x)
    chromatic: float = 1.0   # sigma_c proportional to (lam / lam0) ** chromatic
    flat_grad: float = 0.0   # injected throughput variation: flat = 1 + flat_grad xi_y (mean 1 over the detector)
    f_wing: float = 0.1
    wing_ratio: float = 3.0
    filters: list = field(default_factory=list)  # (name, lo, hi)


def phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def box_gauss(dx, dy, sig, p):
    """Mean over a pixel box of side p centred at offset (dx, dy) from a unit-flux Gaussian of width sig (per unit area)."""
    a = (phi((dx + p / 2) / sig) - phi((dx - p / 2) / sig))
    b = (phi((dy + p / 2) / sig) - phi((dy - p / 2) / sig))
    return a * b / (p * p)


@dataclass
class Truth:
    pts: np.ndarray    # (n, 4): x, y, amp, beta
    gals: np.ndarray   # (n, 6): x, y, sigma, amp, beta, _

    def flux(self, amp, beta, lam):
        return amp * (lam / 2.0) ** beta


def make_truth(rng, half=1.5, n_pts=14, n_gal=2, fmin=0.3, fmax=7.0):
    pts = np.stack([rng.uniform(-half, half, n_pts), rng.uniform(-half, half, n_pts), np.exp(rng.uniform(math.log(fmin), math.log(fmax), n_pts)),
                    rng.uniform(-1.5, 1.0, n_pts)], 1)
    gals = np.stack([rng.uniform(-half * 0.7, half * 0.7, n_gal), rng.uniform(-half * 0.7, half * 0.7, n_gal), rng.uniform(0.08, 0.2, n_gal),
                     rng.uniform(3.0, 8.0, n_gal), rng.uniform(-0.5, 1.5, n_gal), np.zeros(n_gal)], 1)
    return Truth(pts, gals)


def render_truth(truth, inst, filt, X, Y, xi, n_dense=24, epoch_effect=0.0):
    """Analytic noiseless pixel values at true pixel centres X, Y (arcsec, (M,)), normalised detector coords xi (M, 2)."""
    lam, w = quantile_nodes(filt.lam_um, filt.T, n_dense)
    out = torch.zeros_like(X, dtype=torch.float64)
    X, Y = X.double(), Y.double()
    flat = 1 + inst.flat_grad * xi[:, 1].double()
    for q in range(n_dense):
        l = float(lam[q])
        sc = inst.sigma0 * (l / inst.lam0) ** inst.chromatic * (1 + inst.space_grad * xi[:, 0].double()) * (1 + epoch_effect)
        sw = inst.wing_ratio * sc
        acc = torch.zeros_like(X)
        for x, y, amp, beta in truth.pts:
            F = amp * (l / 2.0) ** beta
            acc += F * ((1 - inst.f_wing) * box_gauss(X - x, Y - y, sc, inst.pixel) + inst.f_wing * box_gauss(X - x, Y - y, sw, inst.pixel))
        for x, y, sg, amp, beta, _ in truth.gals:
            F = amp * (l / 2.0) ** beta
            acc += F * ((1 - inst.f_wing) * box_gauss(X - x, Y - y, torch.sqrt(sc ** 2 + sg ** 2), inst.pixel)
                        + inst.f_wing * box_gauss(X - x, Y - y, torch.sqrt(sw ** 2 + sg ** 2), inst.pixel))
        out += w[q] * flat * acc
    return out.float()


def make_field(seed=0, n_per_filter=8, n_val=2, n_held=1, noise=1.0, Q=3, dev="cpu", insts=None, epochs=(0.0, 1.0),
               shift_rms=0.03, rot_choices=(0, 1), epoch_effect=0.0, centre_range=0.35, patch_px=64, truth_half=1.5):
    """Exposures (header WCS without the true offsets), FilterSet, meta, truth and per-exposure true offsets."""
    rng = np.random.default_rng(seed)
    if insts is None:
        insts = [SynthInstrument("A", 0.06, 40, 6.5, 0.05, 1.5, space_grad=0.12, chromatic=1.0, flat_grad=0.06,
                                 filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
                 SynthInstrument("B", 0.11, 22, 6.5, 0.09, 3.5, space_grad=0.12, chromatic=1.0, flat_grad=0.06, filters=[("B1", 3.0, 4.0)])]
    filters = []
    for ins in insts:
        for name, lo, hi in ins.filters:
            filters.append(box_filter(name, ins.name, lo, hi, Q))
    fset = FilterSet(filters)
    truth = make_truth(rng, half=truth_half)
    exps, dets, chans = [], [], []
    k = 0
    for ii, ins in enumerate(insts):
        dets.append(ins.name)
        chans.append(ins.name)
        for name, lo, hi in ins.filters:
            f = fset.index[name]
            for j in range(n_per_filter):
                rot = int(rng.choice(rot_choices))
                th = rot * math.pi / 2
                Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
                c = rng.uniform(-centre_range, centre_range, 2)
                n = ins.npix
                rr, cc = np.mgrid[0:n, 0:n].astype(np.float64)
                off = np.stack([cc - (n - 1) / 2, rr - (n - 1) / 2], -1) * ins.pixel
                pos = off @ Rm.T + c
                Xh, Yh = torch.tensor(pos[..., 0], dtype=torch.float32), torch.tensor(pos[..., 1], dtype=torch.float32)
                true_shift = rng.normal(0, shift_rms, 2)
                xi = torch.stack([torch.tensor(cc.ravel() / (n - 1) * 2 - 1), torch.tensor(rr.ravel() / (n - 1) * 2 - 1)], -1).float()
                t = epochs[j % len(epochs)]
                eff = epoch_effect * (1 if t > 0 else -1)
                clean = render_truth(truth, ins, fset.filters[f], Xh.reshape(-1) + float(true_shift[0]), Yh.reshape(-1) + float(true_shift[1]), xi,
                                     epoch_effect=eff).view(n, n)
                bk = float(rng.normal(5.0, 1.0))
                d = clean + bk + noise * torch.randn(n, n, generator=torch.Generator().manual_seed(1000 + k))
                split = "held" if j < n_held else ("val" if j < n_held + n_val else "train")
                ins_name = ins.name
                e = Exposure(name=f"{name}_{j}", filt=f, inst=ii, chan=ii, det=ii, t=float(t), mjd=float(t), det_shape=(n, n), cut0=(0, 0),
                             d=d.to(dev), sig=torch.full((n, n), noise, device=dev), valid=torch.ones(n, n, dtype=torch.bool, device=dev),
                             X=Xh.to(dev), Y=Yh.to(dev), R=Rm, split=split, group=f"{name}_{j}", stripes=False, bkg0=float(d.median()),
                             Jpix=torch.tensor(np.broadcast_to(Rm * ins.pixel, (n, n, 2, 2)).copy(), dtype=torch.float32, device=dev),
                             extra={"J": (Rm * ins.pixel).tolist(), "true_shift": true_shift.tolist(), "clean": clean.to(dev), "bkg_true": bk,
                                    "diameter_m": ins.diameter_m})
                e.eid = k
                exps.append(e)
                k += 1
    for e in exps:
        make_patches(e, patch_px)
    meta = build_meta(exps, dets, chans, [i.name for i in insts])
    return exps, fset, meta, truth, insts
