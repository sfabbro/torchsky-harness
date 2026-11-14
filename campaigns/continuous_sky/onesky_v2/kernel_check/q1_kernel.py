#!/usr/bin/env python3
"""Question 1: shape of the learned kernel, measured independently from the network weights.

For each filter: kernel at the centre and at two corners of the training footprint of the detector with the most exposures of that
filter (mean code, no per-exposure term), plus the spread over the per-exposure codes of that filter (tau_exp z_e).
Measured: radial profile (azimuthal mean about the origin and about the peak), FWHM (half-maximum area and x/y cuts through the peak,
0.0005" grid), encircled fraction at 0.05, 0.1, 0.2, 0.5, 1, 2 arcsec and at the 10" edge, fraction beyond +-32 and +-128 detector pixels
(square in the detector frame, as in the earlier claim), negative fraction, normalisation (package level quadrature vs an independent
polar quadrature of the same continuous function), and a three-component decomposition of the radial profile.
Run: python3 q1_kernel.py   (about 2-3 min, one thread)
"""
import json
import math
import os
import time

import numpy as np
import torch

from kc_common import EXTENT, FILTERS, PIX, PUB_FWHM, Fit, smooth_down, HERE

torch.set_num_threads(1)
OUT = os.path.join(HERE, "results")
RAD = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, EXTENT]


def polar_nodes():
    """Radii: 0.0001" steps to 0.06", then log-spaced to 10" (dr/r = 0.25 %); trapezoid weights r dr. 720 angles."""
    r1 = np.arange(0.0, 0.06, 0.0001)
    r2 = np.exp(np.arange(math.log(0.06), math.log(EXTENT), 0.0025))
    r = np.unique(np.concatenate([r1, r2, [EXTENT]]))
    w = np.zeros_like(r)
    w[1:-1] = 0.5 * (r[2:] - r[:-2])
    w[0] = 0.5 * (r[1] - r[0])
    w[-1] = 0.5 * (r[-1] - r[-2])
    na = 720
    th = (np.arange(na) + 0.5) * 2 * math.pi / na
    return r, w * r, th


def kernel_polar(fit, h, r, th, chunk_r=150):
    """exp(l) * taper on the polar grid (nr, na), relative to its value scale exp(lmax) (returned separately)."""
    vals = np.empty((len(r), len(th)))
    for i in range(0, len(r), chunk_r):
        rr = r[i:i + chunk_r]
        vx = (rr[:, None] * np.cos(th)[None]).ravel()
        vy = (rr[:, None] * np.sin(th)[None]).ravel()
        v = torch.tensor(np.stack([vx, vy], -1))
        vals[i:i + chunk_r] = fit.logk(v, h).numpy().reshape(len(rr), len(th))
    T = smooth_down(torch.tensor(r), 0.85 * EXTENT, EXTENT).numpy()
    return vals, T


def measure(fit, f, det, xi, e_rot, e_code=None, full=True):
    chan = 0 if f < 3 else 1
    pix = PIX[chan]
    h = fit.code(f, det=det, xi=xi, e=e_code)
    R = fit.R[e_rot]
    logZ_lev = fit.logZ_levels(h, chan, R)
    r, wr, th = polar_nodes()
    L, T = kernel_polar(fit, h, r, th)
    da = 2 * math.pi / len(th)
    # independent continuous integral of exp(l) T over the 10" disc
    Zc = float((np.exp(L - logZ_lev) * T[:, None] * wr[:, None]).sum() * da)   # in units of Z_lev
    k = np.exp(L - logZ_lev) * T[:, None]                                       # package-normalised density [arcsec^-2]
    ring = (k * wr[:, None]).sum(1) * da                                        # d(integral)/ring
    cum = np.cumsum(ring)
    out = dict(filter=FILTERS[f], det=det, xi=[float(x) for x in xi], exposure_code=e_code, logZ_levels=logZ_lev,
               integral_continuous_over_levels=Zc, n_negative=int((k < 0).sum()), min_over_peak=float(k.min() / k.max()))
    out["encircled_about_origin_levels_norm"] = {str(a): float(np.interp(a, r, cum)) for a in RAD}
    out["encircled_about_origin_cont_norm"] = {str(a): float(np.interp(a, r, cum) / Zc) for a in RAD}
    # fraction beyond +-L detector pixels (square in the detector frame), with the package normalisation
    vx, vy = r[:, None] * np.cos(th)[None], r[:, None] * np.sin(th)[None]
    for Lp in (32, 128):
        inside = (np.abs(vx) <= Lp * pix) & (np.abs(vy) <= Lp * pix)
        out[f"frac_beyond_pm{Lp}px_square"] = float(1 - (k * inside * wr[:, None]).sum() * da)
        out[f"frac_beyond_pm{Lp}px_square_cont_norm"] = float(1 - (k * inside * wr[:, None]).sum() * da / Zc)
    out["frac_beyond_r_1arcsec"] = float(1 - np.interp(1.0, r, cum))
    # azimuthal mean profile about the origin
    prof = k.mean(1)
    if not full:
        return out, (r, prof)
    # fine cartesian grid for the FWHM (0.0005", +-0.15")
    g = np.arange(-0.15, 0.15 + 1e-9, 0.0005)
    X, Y = np.meshgrid(g, g)
    lg = fit.logk(torch.tensor(np.stack([X.ravel(), Y.ravel()], -1)), h).numpy().reshape(X.shape)
    kk = np.exp(lg - logZ_lev)
    i0, j0 = np.unravel_index(np.argmax(kk), kk.shape)
    pk = kk[i0, j0]
    area = (kk >= pk / 2).sum() * 0.0005 ** 2
    out["peak_density_arcsec-2"] = float(pk)
    out["peak_offset_arcsec"] = [float(g[j0]), float(g[i0])]
    out["fwhm_area_equivalent_arcsec"] = float(2 * math.sqrt(area / math.pi))

    def cut(p):
        idx = np.where(p >= pk / 2)[0]
        a, b = idx[0], idx[-1]
        lf = a - 1 + (pk / 2 - p[a - 1]) / (p[a] - p[a - 1])
        rt = b + (p[b] - pk / 2) / (p[b] - p[b + 1])
        return float((rt - lf) * 0.0005)
    out["fwhm_x_cut_arcsec"] = cut(kk[i0, :])
    out["fwhm_y_cut_arcsec"] = cut(kk[:, j0])
    out["fwhm_x_cut_px"] = out["fwhm_x_cut_arcsec"] / pix
    out["fwhm_y_cut_px"] = out["fwhm_y_cut_arcsec"] / pix
    # encircled about the peak, inside 0.15" (fine grid; normalised by the package Z)
    Rp = np.hypot(X - g[j0], Y - g[i0])
    out["encircled_about_peak_levels_norm"] = {str(a): float(kk[Rp <= a].sum() * 0.0005 ** 2) for a in (0.01, 0.02, 0.05, 0.1)}
    out["centroid_within_0.15_arcsec"] = [float((kk * X).sum() / kk.sum()), float((kk * Y).sum() / kk.sum())]
    return out, (r, prof)


def decompose(r, p, ee_peak):
    """Non-parametric decomposition of the azimuthal-mean profile p(r) about the origin [arcsec^-2] (a parametric spike + Gaussian +
    Moffat fit was tried first and left log residuals of ~1 dex because the profile has a flat shoulder and a flat outer floor):
      spike   = integral within 0.02" of the PEAK (fine grid) minus the shoulder level x pi 0.02^2;
      shoulder= the flat part at 0.03-0.1" (density ~1-2 arcsec^-2) up to its edge at 0.2", about the origin, minus the spike;
      halo    = 0.2" to 2";   outer   = 2" to the 10" edge (the density is flat from ~5" to the 8.5" taper start: a pedestal).
    Also the 'floor': median density at 5-8" times the area of the tapered 10" disc (what a constant would carry)."""
    ring = 2 * math.pi * r * p
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (ring[1:] + ring[:-1]) * np.diff(r))])
    I = lambda a, b: float(np.interp(b, r, cum) - np.interp(a, r, cum))
    shoulder_level = float(np.median(p[(r > 0.04) & (r < 0.08)]))
    spike = ee_peak["0.02"] - shoulder_level * math.pi * 0.02 ** 2
    inner = I(0, 0.2)
    floor = float(np.median(p[(r > 5) & (r < 8)]))
    rr = np.linspace(0, EXTENT, 20001)
    T = smooth_down(torch.tensor(rr), 0.85 * EXTENT, EXTENT).numpy()
    area = float(np.trapezoid(2 * math.pi * rr * T, rr))
    tot = float(cum[-1])
    return dict(spike_fraction=spike / tot, shoulder_fraction=(inner - spike) / tot, shoulder_level_arcsec2=shoulder_level,
                halo_0p2_2_fraction=I(0.2, 2.0) / tot, outer_2_10_fraction=I(2.0, EXTENT) / tot,
                floor_density_5_8_arcsec2=floor, floor_constant_fraction=floor * area / tot, tapered_disc_area_arcsec2=area,
                total_integral_of_profile=tot)


def main():
    t0 = time.time()
    fit = Fit()
    # blur gauge: recompute the reference moment (dilation is 1 when M >= floor)
    M, floor = fit.blur_moment(0)
    res = dict(blur_gauge=dict(M_ref_arcsec2=M, floor_arcsec2=floor, dilation=1.0 if M >= floor else math.sqrt(floor / M),
                               note="dilation applied below only if M < floor"), tau=fit.tau, filters={})
    assert M >= floor, "blur floor active: dilation would have to be applied"
    profiles = {}
    for f in range(6):
        es = np.nonzero(fit.e_filt == f)[0]
        dets, cnt = np.unique(fit.e_det[es], return_counts=True)
        det = int(dets[np.argmax(cnt)])
        e_rot = int(es[fit.e_det[es] == det][0])
        lat = fit.ref_lattice_xi(det).numpy()
        pos = {"footprint_centre": lat[4], "corner_a": lat[0], "corner_b": lat[8]}
        rf = {"detector": ["NRCB1", "NRCB2", "NRCB3", "NRCB4", "NRCBLONG"][det], "rotation_from_exposure": e_rot, "positions": {}}
        for name, xi in pos.items():
            o, (r, p) = measure(fit, f, det, xi, e_rot, full=True)
            rf["positions"][name] = o
            if name == "footprint_centre":
                profiles[FILTERS[f]] = (r, p)
                rf["decomposition"] = decompose(r, p, o["encircled_about_peak_levels_norm"])
        # spread over per-exposure codes (5 exposures of this filter)
        spread = []
        for e in es[:: max(1, len(es) // 5)][:5]:
            o, _ = measure(fit, f, int(fit.e_det[e]), (0.0, 0.0), int(e), e_code=int(e), full=True)
            spread.append({k: o[k] for k in ("exposure_code", "fwhm_area_equivalent_arcsec", "frac_beyond_pm32px_square", "frac_beyond_r_1arcsec",
                                              "encircled_about_origin_levels_norm", "integral_continuous_over_levels")})
        rf["per_exposure_codes"] = spread
        res["filters"][FILTERS[f]] = rf
        c = rf["positions"]["footprint_centre"]
        print(f"{FILTERS[f]} det {rf['detector']}: FWHM area-eq {c['fwhm_area_equivalent_arcsec']:.4f}\" cuts {c['fwhm_x_cut_arcsec']:.4f}/{c['fwhm_y_cut_arcsec']:.4f}"
              f" (published {PUB_FWHM[FILTERS[f]]}); EE {[round(v, 3) for v in c['encircled_about_origin_levels_norm'].values()]};"
              f" beyond +-32px {c['frac_beyond_pm32px_square']:.3f} +-128px {c['frac_beyond_pm128px_square']:.3f}; Zcont/Zlev {c['integral_continuous_over_levels']:.4f};"
              f" decomposition {json.dumps({k: round(v, 4) for k, v in rf['decomposition'].items()})} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, "q1_kernel.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez(os.path.join(OUT, "q1_profiles.npz"), **{k: np.stack([v[0], v[1]]) for k, v in profiles.items()})
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (k, (r, p)) in enumerate(profiles.items()):
        ax[0].loglog(r[1:], p[1:], label=k, color=plt.cm.viridis(i / 5))
        ee = np.cumsum(2 * math.pi * p * np.gradient(r) * r)
        ax[1].semilogx(r[1:], ee[1:], color=plt.cm.viridis(i / 5), label=k)
    ax[0].set_xlabel("radius [arcsec]"); ax[0].set_ylabel("azimuthal mean density [arcsec$^{-2}$]"); ax[0].legend(fontsize=8)
    ax[0].set_title("learned kernel, footprint centre, mean code")
    ax[1].set_xlabel("radius [arcsec]"); ax[1].set_ylabel("encircled fraction (package normalisation)")
    for a in (0.031 * 32, 0.063 * 32):
        ax[1].axvline(a, ls=":", c="grey")
    ax[1].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "q1_kernel_profiles.png"), dpi=110)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
