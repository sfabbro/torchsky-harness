#!/usr/bin/env python3
"""Check of my independent renderer against the package's own renderer (render.render_fft with the package's level kernels,
kernel_level_values), on the FULL fitted sky, at detector-like points (0.031" / 0.063" lattice rotated by R_e) within 0.6" of two
sources, for F090W and F444W (one exposure each, code z_e, throughput exp(gamma)).
Mine = fine part (kernel |u| <= 2.7" on a 0.005" grid, exact tent sky, FFT) + coarse part (cell-integrated kernel |u| > 2.7" on the
0.04" grid, applied to the node values), interpolated bilinearly to the points.
Run: python3 crosscheck_render.py  (about 1-2 min)"""
import json, math, os, sys
import numpy as np
import torch
from scipy import ndimage
from scipy.signal import fftconvolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kc_common import EXTENT, FILTERS, HERE, PIX, SKY_PITCH, Fit, smooth_down
from q2_q4_render import kernel_table
from fieldop.sky import SkyGrid, RBFBasis
from fieldop.render import kernel_level_values, render_fft
torch.set_num_threads(1)
H, KR = 0.005, 2.7
SRC = [(8.36, -3.28), (0.24, -2.40)]


def outer_cells(fit, h, logZ, R):
    n = int(round(EXTENT / SKY_PITCH))
    g = np.arange(-n, n + 1) * SKY_PITCH
    CX, CY = np.meshgrid(g, g)
    W = np.zeros_like(CX)
    sel = np.hypot(CX, CY) > KR
    s = ((np.arange(2) + 0.5) / 2 - 0.5) * SKY_PITCH
    for a in s:
        for b in s:
            ux, uy = CX[sel] + a, CY[sel] + b
            v = np.stack([ux, uy], -1) @ R.numpy().astype(np.float64)
            T = smooth_down(torch.tensor(np.hypot(ux, uy)), 0.85 * EXTENT, EXTENT).numpy()
            W[sel] += np.exp(fit.logk(torch.tensor(v), h).numpy() - logZ) * T * (SKY_PITCH / 2) ** 2
    return W


fit = Fit()
N = fit.N
coef = fit.sky_coeffs()
sky = SkyGrid(coef.float(), SKY_PITCH, RBFBasis(12))
res = {}
for f in (0, 5):
    chan = 0 if f < 3 else 1
    e = int([i for i in fit.s["gref"].numpy() if fit.e_filt[i] == f][0])
    R = fit.R[e]
    h = fit.code(f, det=int(fit.e_det[e]), xi=(0.0, 0.0), e=e)
    Q = fit.s["lam"].shape[1]
    hq = h.expand(Q, -1)
    with torch.no_grad():
        Ks, logZq = kernel_level_values(fit.net, fit.levels[chan], R.float(), hq)
    logZ = float(logZq[0])
    gam = fit.gamma(e).double().numpy()
    amp = np.exp(gam) * fit.s["wq"][f].double().numpy()
    Phi = fit.s["Phi"][f]
    K, g = kernel_table(fit, h, logZ, R, KR, H)
    K = np.where(np.hypot(*np.meshgrid(g, g)) <= KR, K, 0.0)
    Wout = outer_cells(fit, h, logZ, R)
    skies = np.einsum("qk,khw->qhw", Phi.double().numpy(), coef.double().numpy())
    rows = []
    for (X0, Y0) in SRC:
        pix = PIX[chan]
        kk = np.arange(-int(0.6 / pix), int(0.6 / pix) + 1)
        KX, KY = np.meshgrid(kk, kk)
        u = (np.stack([KX.ravel(), KY.ravel()], -1) * pix + 0.37 * pix) @ R.numpy().T.astype(np.float64) + np.array([X0, Y0])   # u = R v
        pos = torch.tensor(u, dtype=torch.float32)[None].expand(Q, -1, -1).contiguous()
        with torch.no_grad():
            c_pkg = render_fft(sky, sky.coeffs(), Phi, fit.levels[chan], [k[None] for k in Ks], torch.ones(1, u.shape[0]), pos, int(round(SKY_PITCH / fit.levels[chan].p_r)))
        S_pkg = (torch.tensor(amp)[:, None] * c_pkg.double()).sum(0).numpy()
        # mine: fine part on a +-(0.8 + KR) window
        half = 0.8 + KR
        gf = np.arange(-round(half / H), round(half / H) + 1) * H
        CC, RR = np.meshgrid((X0 + gf) / SKY_PITCH + (N - 1) / 2, (Y0 + gf) / SKY_PITCH + (N - 1) / 2)
        S_me = np.zeros(u.shape[0])
        for q in range(Q):
            s_f = ndimage.map_coordinates(skies[q], [RR, CC], order=1, mode="constant")
            cf = fftconvolve(s_f, K[::-1, ::-1], mode="same") * H * H
            co = fftconvolve(skies[q], Wout[::-1, ::-1], mode="same")
            pf = ndimage.map_coordinates(cf, [(u[:, 1] - Y0 - gf[0]) / H, (u[:, 0] - X0 - gf[0]) / H], order=1)
            po = ndimage.map_coordinates(co, [u[:, 1] / SKY_PITCH + (N - 1) / 2, u[:, 0] / SKY_PITCH + (N - 1) / 2], order=1)
            S_me += amp[q] * (pf + po)
        d = S_me - S_pkg
        o = dict(filter=FILTERS[f], exposure=e, source=[X0, Y0], n_points=int(u.shape[0]), peak_pkg=float(S_pkg.max()), peak_mine=float(S_me.max()),
                 mean_pkg=float(S_pkg.mean()), mean_mine=float(S_me.mean()), max_abs_diff_over_peak=float(np.abs(d).max() / S_pkg.max()),
                 rms_diff_over_peak=float(np.sqrt((d ** 2).mean()) / S_pkg.max()), logZ_levels=logZ)
        rows.append(o)
        print(o, flush=True)
    res[FILTERS[f]] = rows
json.dump(res, open(os.path.join(HERE, "results", "crosscheck_renderer.json"), "w"), indent=1)
