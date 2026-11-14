#!/usr/bin/env python3
"""Questions 2 and 4: what the whole model predicts, written independently of the package renderer.

Model (read from model.py / render.py):  S(x) = sum_q w_q exp(gamma_q) c_q(x),  c_q(x) = int k(u) s_q(x + u) d^2u,  s_q = tent interpolant of the
node values Phi[f, q] . A;  k(u) = exp(l(R_e^T u, h_e)) T(|u|) / Z (unit integral on the package's level quadrature).  Data pixels read
S at their centres (no extra pixel box: the kernel is the whole pixel-in-kernel response).

Question 2 (per filter, one exposure: the first training exposure of the filter, its rotation R_e and code z_e):
  for 5 stars (the most compact bright peaks of the fitted sky, see SOURCES), the SOURCE-ONLY sky (nodes within 1" of the source, minus
  the 3-sigma-clipped median of the annulus 1.0-1.5") is upsampled bilinearly (= the tent interpolant) to a
  0.005" grid and correlated by FFT with the kernel tabulated on the same grid to +-2.7" (enough for every output point within 2"; the
  kernel beyond 2.7" only moves flux beyond 2"). Measured on the sky alone, on the prediction c, and on the mosaic (bicubic upsampling,
  same annulus background): FWHM (half-maximum area, equivalent diameter), encircled fractions; the prediction is also sampled on the
  exposure's detector pixel lattice (nominal scale, no distortion) to give the FWHM in detector pixels.
Question 4 (per filter, 4 exposures): discrete kernel on the 0.04" sky grid to +-10" (cell integrals, 16x16 sub-samples within 1.2",
  2x2 beyond), applied to the whole sky (5 wavelength nodes, throughput exp(gamma_q) of the exposure); mean over the data region
  (|X|,|Y| <= 11.24") of the prediction with the full kernel and with the kernel truncated at +-32 detector pixels (square in the detector
  frame, not renormalised), against the fitted background constant and the exposure's data median (bkg0).
Run: python3 q2_q4_render.py q2  |  q4     (each about 2-3 min, one thread)
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy import ndimage
from scipy.signal import fftconvolve

from kc_common import EXTENT, FILTERS, HERE, PIX, PUB_FWHM, SKY_PITCH, Fit, smooth_down
from q3_flux import Img, load_mosaic

torch.set_num_threads(1)
SOURCES = [(-9.00, -5.00), (10.28, -2.84), (-9.00, 4.72), (9.68, 6.76), (9.00, -0.40)]   # sky-frame arcsec (X west, Y north)
# stars: diffraction spikes in the fitted sky (figs/q4_wing_map_F200W.png), mosaic 0.15"/0.5" flux ratio 0.7-0.8 at F200W and FWHM 0.077" there
# (the most compact of the brightest sky peaks); the fifth (9.0, -0.4) is compact but has a neighbour (ratio 0.47-0.57).
MASK_R, ANN = 1.0, (1.0, 1.5)   # source-only sky: nodes within 1" (keeps the inner diffraction spikes), background annulus 1.0-1.5"
H = 0.005          # fine grid pitch [arcsec]
KR = 2.7           # kernel table half-size [arcsec]
# effective (pixel-convolved) widths implied by the published optical FWHM: sqrt(FWHM^2 + (2.355 p / sqrt 12)^2), a Gaussian approximation
EPSF = {f: math.sqrt(PUB_FWHM[f] ** 2 + (2.3548 * PIX[0 if i < 3 else 1] / math.sqrt(12)) ** 2) for i, f in enumerate(FILTERS)}


def kernel_table(fit, h, logZ, R, half, pitch):
    """k(u) on a square grid of sky offsets u (rows = Y, cols = X) [arcsec^-2]; v = R^T u."""
    g = np.arange(-round(half / pitch), round(half / pitch) + 1) * pitch
    X, Y = np.meshgrid(g, g)
    u = np.stack([X.ravel(), Y.ravel()], -1)
    v = u @ R.numpy().astype(np.float64)            # row vectors: v = u R  (= R^T u), as render.kernel_level_values
    lg = fit.logk(torch.tensor(v), h).numpy()
    T = smooth_down(torch.tensor(np.hypot(X, Y).ravel()), 0.85 * EXTENT, EXTENT).numpy()
    return (np.exp(lg - logZ) * T).reshape(X.shape), g


def half_max_fwhm(img, pitch):
    pk = img.max()
    lab, _ = ndimage.label(img >= pk / 2)
    i0, j0 = np.unravel_index(np.argmax(img), img.shape)
    area = (lab == lab[i0, j0]).sum() * pitch ** 2
    return 2 * math.sqrt(area / math.pi)


def enc(img, pitch, cx, cy, radii):
    n = img.shape[0]
    g = (np.arange(n) - (n - 1) / 2) * pitch
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X - cx, Y - cy)
    return {str(r): float(img[R <= r].sum() * pitch ** 2) for r in radii}


def q2():
    fit = Fit()
    N = fit.N
    radii = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    res = {"sources_sky_arcsec": SOURCES, "effective_psf_fwhm_approx": EPSF, "filters": {}}
    for f in range(6):
        t0 = time.time()
        name = FILTERS[f]
        chan = 0 if f < 3 else 1
        e = int(np.nonzero((fit.e_filt == f) & fit.s["gref"].new_zeros(123, dtype=torch.bool).index_fill(0, fit.s["gref"], True).numpy())[0][0])
        R = fit.R[e]
        h = fit.code(f, det=int(fit.e_det[e]), xi=(0.0, 0.0), e=e)
        logZ = fit.logZ_levels(h, chan, R)
        K, g = kernel_table(fit, h, logZ, R, KR, H)
        kin = float(K.sum() * H * H)                   # kernel integral inside the +-2.7" table
        B = fit.band_image(f).double().numpy()
        sky = Img(B, SKY_PITCH, (N - 1) / 2, (N - 1) / 2, SKY_PITCH ** 2)
        d, p, c0, r0, a = load_mosaic(name)
        q3 = json.load(open(os.path.join(HERE, "results", "q3_flux.json")))[name]
        dX, dY = q3["offset_sky_minus_mosaic_arcsec"]
        mos = Img(d, p, c0 - dX / p, r0 - dY / p, a)
        rows = []
        for (X0, Y0) in SOURCES:
            # source-only sky on the fine grid (+-KR about the node nearest the source)
            jc, ic = int(round(X0 / SKY_PITCH + (N - 1) / 2)), int(round(Y0 / SKY_PITCH + (N - 1) / 2))
            Xc, Yc = (jc - (N - 1) / 2) * SKY_PITCH, (ic - (N - 1) / 2) * SKY_PITCH
            lvl = sky.annulus_level(Xc, Yc, *ANN)
            jj, ii = np.meshgrid(np.arange(N), np.arange(N))
            rr = np.hypot((jj - jc) * SKY_PITCH, (ii - ic) * SKY_PITCH)
            src = np.where(rr <= MASK_R, B - lvl, 0.0)
            n = len(g)
            fx = Xc + g                                           # fine grid coordinates (same spacing as the kernel table)
            cols = fx / SKY_PITCH + (N - 1) / 2
            rows_ = (Yc + g) / SKY_PITCH + (N - 1) / 2
            CC, RR = np.meshgrid(cols, rows_)
            s_fine = ndimage.map_coordinates(src, [RR, CC], order=1, mode="constant")       # tent interpolant
            flux_sky = float(s_fine.sum() * H * H)
            # c(x) = sum_u k(u) s(x + u) h^2: correlation = convolution with the flipped kernel
            c = fftconvolve(s_fine, K[::-1, ::-1], mode="same") * H * H
            # centroid of the sky source
            cxs = float((s_fine.clip(0) * g[None, :]).sum() / s_fine.clip(0).sum())
            cys = float((s_fine.clip(0) * g[:, None]).sum() / s_fine.clip(0).sum())
            inner = np.hypot(*np.meshgrid(g, g)) <= 2.0
            # mosaic on the same fine grid (bicubic), annulus background, restricted to 2"
            mc, mr = mos.to_px(Xc + g, Yc + g)
            MC, MR = np.meshgrid(mc, mr)
            mlvl = mos.annulus_level(Xc, Yc, *ANN)
            m_fine = ndimage.map_coordinates(np.nan_to_num(mos.d), [MR, MC], order=3, mode="constant") - mlvl
            m_fine_in = np.where(inner, m_fine, 0.0)
            # detector-pixel sampling of the prediction (peak at a pixel centre), nominal scale, lattice rotated by R_e
            pix = PIX[chan]
            ipk = np.unravel_index(np.argmax(c), c.shape)
            kk = np.arange(-12, 13)
            KX, KY = np.meshgrid(kk, kk)
            vdet = np.stack([KX.ravel(), KY.ravel()], -1) * pix
            usky = vdet @ R.numpy().T.astype(np.float64)          # u = R v
            px = ndimage.map_coordinates(c, [(usky[:, 1] + g[ipk[0]] - g[0]) / H, (usky[:, 0] + g[ipk[1]] - g[0]) / H], order=1).reshape(KX.shape)
            o = dict(X=X0, Y=Y0, sky_flux_uJy=flux_sky * 1e12 / 4.2545170296e10, kernel_integral_in_table=kin,
                     fwhm_sky_arcsec=half_max_fwhm(s_fine, H), fwhm_pred_arcsec=half_max_fwhm(c, H),
                     fwhm_mosaic_arcsec=half_max_fwhm(np.where(inner, m_fine, 0), H),
                     fwhm_pred_detector_px=half_max_fwhm(c, H) / pix,
                     peak_pixel_value_pred_over_mosaic_peak=float(c.max() / m_fine_in.max()),
                     enc_sky_over_sky_flux={k: v / flux_sky for k, v in enc(s_fine, H, cxs, cys, radii).items()},
                     enc_pred_over_sky_flux={k: v / flux_sky for k, v in enc(c, H, cxs, cys, radii).items()},
                     enc_mosaic_over_sky_flux={k: v / flux_sky for k, v in enc(m_fine_in, H, cxs, cys, radii).items()},
                     pred_over_mosaic_within={k: (v / w if w > 0 else None) for (k, v), w in zip(enc(c, H, cxs, cys, radii).items(), enc(m_fine_in, H, cxs, cys, radii).values())},
                     detector_px_image_peak_fraction=float(px.max() * pix * pix / flux_sky))
            o["sky_over_mosaic_within"] = {k: (v / w if w > 0 else None) for (k, v), w in zip(enc(s_fine, H, cxs, cys, radii).items(), enc(m_fine_in, H, cxs, cys, radii).values())}
            o["enc_pred_rel_2arcsec"] = {k: v / o["enc_pred_over_sky_flux"]["2.0"] for k, v in o["enc_pred_over_sky_flux"].items()}
            o["enc_mosaic_rel_2arcsec"] = {k: v / o["enc_mosaic_over_sky_flux"]["2.0"] for k, v in o["enc_mosaic_over_sky_flux"].items()}
            rows.append(o)
            if (X0, Y0) == SOURCES[0]:
                np.savez(os.path.join(HERE, "results", f"q2_profiles_{name}.npz"), g=g, sky=s_fine, pred=c, mosaic=m_fine_in)
        res["filters"][name] = dict(exposure=e, sources=rows)
        med = lambda k: float(np.median([r[k] for r in rows]))
        print(f"{name} (exp {e}): FWHM sky {med('fwhm_sky_arcsec'):.3f}  pred {med('fwhm_pred_arcsec'):.3f}  mosaic {med('fwhm_mosaic_arcsec'):.3f}"
              f"  (published {PUB_FWHM[name]}, pixel-convolved ~{EPSF[name]:.3f}); pred EE/skyflux "
              f"{[round(float(np.median([r['enc_pred_over_sky_flux'][str(x)] for r in rows])), 3) for x in radii]}; mosaic EE/skyflux "
              f"{[round(float(np.median([r['enc_mosaic_over_sky_flux'][str(x)] for r in rows])), 3) for x in radii]}; pred/mosaic "
              f"{[round(float(np.median([r['pred_over_mosaic_within'][str(x)] for r in rows])), 3) for x in radii]}; sky/mosaic "
              f"{[round(float(np.median([r['sky_over_mosaic_within'][str(x)] for r in rows])), 3) for x in radii]} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(HERE, "results", "q2_point_source.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def discrete_kernel(fit, h, logZ, R, chan, trunc_px=None):
    """Cell integrals of k on the 0.04" sky grid to +-10" (cells centred on the grid offsets), optionally only the part inside the
    +-trunc_px square of the DETECTOR frame. 16x16 sub-samples per cell within 1.2", 2x2 beyond."""
    n = int(round(EXTENT / SKY_PITCH))
    g = np.arange(-n, n + 1) * SKY_PITCH
    W = np.zeros((len(g), len(g)))
    Rn = R.numpy().astype(np.float64)
    pix = PIX[chan]
    for (lo, hi, sub) in ((0.0, 1.2, 16), (1.2, 1e9, 2)):
        s = ((np.arange(sub) + 0.5) / sub - 0.5) * SKY_PITCH
        CX, CY = np.meshgrid(g, g)
        rc = np.maximum(np.abs(CX), np.abs(CY))
        sel = (rc >= lo - 1e-9) & (rc < hi) if lo > 0 else (rc < hi)
        ci, cj = np.nonzero(sel)
        for a in s:
            for b in s:
                ux, uy = g[cj] + a, g[ci] + b
                u = np.stack([ux, uy], -1)
                v = u @ Rn
                T = smooth_down(torch.tensor(np.hypot(ux, uy)), 0.85 * EXTENT, EXTENT).numpy()
                val = np.exp(fit.logk(torch.tensor(v), h).numpy() - logZ) * T
                if trunc_px is not None:
                    val = val * ((np.abs(v[:, 0]) <= trunc_px * pix) & (np.abs(v[:, 1]) <= trunc_px * pix))
                W[ci, cj] += val * (SKY_PITCH / sub) ** 2
    return W


def q4():
    fit = Fit()
    N = fit.N
    A = fit.sky_coeffs().double().numpy()
    gnode = (np.arange(N) - (N - 1) / 2) * SKY_PITCH
    GX, GY = np.meshgrid(gnode, gnode)
    region = (np.abs(GX) <= 11.24) & (np.abs(GY) <= 11.24)
    res = {}
    for f in range(6):
        t0 = time.time()
        name = FILTERS[f]
        chan = 0 if f < 3 else 1
        es = np.nonzero(fit.e_filt == f)[0]
        tr = set(fit.s["gref"].numpy().tolist())
        es_tr = [int(e) for e in es if int(e) in tr]
        pick = es_tr[:: max(1, len(es_tr) // 4)][:4]
        Phi = fit.s["Phi"][f].double().numpy()                 # (Q, K)
        skies = np.einsum("qk,khw->qhw", Phi, A)              # (Q, N, N) node values per wavelength node
        out = []
        for e in pick:
            R = fit.R[e]
            h = fit.code(f, det=int(fit.e_det[e]), xi=(0.0, 0.0), e=e)
            logZ = fit.logZ_levels(h, chan, R)
            Wf = discrete_kernel(fit, h, logZ, R, chan)
            Wi = discrete_kernel(fit, h, logZ, R, chan, trunc_px=32)
            gam = fit.gamma(e).double().numpy()
            amp = np.exp(gam) * fit.s["wq"][f].double().numpy()
            # correlation c(x) = sum_d W(d) s(x + d): convolution with the flipped kernel
            Sf = sum(amp[q] * fftconvolve(skies[q], Wf[::-1, ::-1], mode="same") for q in range(len(amp)))
            Si = sum(amp[q] * fftconvolve(skies[q], Wi[::-1, ::-1], mode="same") for q in range(len(amp)))
            wing = Sf - Si
            band = (amp[:, None, None] * skies).sum(0)
            b0, med = float(fit.s["bkg"][e, 0]), float(fit.s["bkg0"][e])
            o = dict(exposure=e, detector=["NRCB1", "NRCB2", "NRCB3", "NRCB4", "NRCBLONG"][int(fit.e_det[e])],
                     kernel_integral_full=float(Wf.sum()), kernel_integral_inside_32px=float(Wi.sum()),
                     mean_sky_band_MJysr=float(band[region].mean()),
                     mean_pred_full_MJysr=float(Sf[region].mean()), mean_pred_trunc32_MJysr=float(Si[region].mean()),
                     mean_wing_MJysr=float(wing[region].mean()), wing_rms_over_region_MJysr=float(wing[region].std()),
                     wing_p05_p95_MJysr=[float(np.percentile(wing[region], 5)), float(np.percentile(wing[region], 95))],
                     background_constant_MJysr=b0, data_median_bkg0_MJysr=med, gain_exp_gamma_mean=float(np.exp(gam).mean()))
            o["wing_over_background_constant"] = o["mean_wing_MJysr"] / b0
            o["wing_over_data_median"] = o["mean_wing_MJysr"] / med
            o["wing_over_full_prediction"] = o["mean_wing_MJysr"] / o["mean_pred_full_MJysr"]
            out.append(o)
            print(f"{name} exp {e}: kernel inside +-32px {o['kernel_integral_inside_32px']:.3f} of {o['kernel_integral_full']:.3f}; mean over data region: "
                  f"sky band {o['mean_sky_band_MJysr']:.4f}, prediction full {o['mean_pred_full_MJysr']:.4f}, truncated {o['mean_pred_trunc32_MJysr']:.4f}, "
                  f"wings {o['mean_wing_MJysr']:.4f} (rms {o['wing_rms_over_region_MJysr']:.4f}, p05-p95 {o['wing_p05_p95_MJysr'][0]:.4f}-{o['wing_p05_p95_MJysr'][1]:.4f}); "
                  f"background constant {b0:.4f}, data median {med:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if e == pick[0]:
                np.savez(os.path.join(HERE, "results", f"q4_wing_{name}.npz"), wing=wing.astype(np.float32), full=Sf.astype(np.float32), band=band.astype(np.float32))
        res[name] = out
    with open(os.path.join(HERE, "results", "q4_wings.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    {"q2": q2, "q4": q4}[sys.argv[1]]()
