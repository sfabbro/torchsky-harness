#!/usr/bin/env python3
"""Tabulate the LEARNED NIRCam kernels of the fieldop reference run (SMACS, fieldop_smacs_fix_200_s0) and summarise them.

What is reproduced from fieldop (files in fieldop_ref/, read-only copy of the package that made the run):
  * the kernel network (LogPolarEncoding + KernelNet with FiLM code), the code network (Codes) and every weight, loaded from fieldop_state.pt
    (the FieldOperator object itself is NOT rebuilt: it needs the exposure files. The pieces it uses are rebuilt one by one);
  * the code  h = hbar(pivot, filter curve, instrument) + tau_lam (hbar(lam) - hbar(pivot)) + tau_det hdet^(det, xi) + tau_exp z_e
    with the final taus of the run (state: softplus(tau.raw.*)): tau_lam = tau_det = 1e-6 (pruned), tau_exp = 0.0587 (not used: no per-exposure code);
    hdet^ is centred and normalised over the run's own reference lattice (ref_det, ref_xi in the state); detector centre xi = (0, 0);
  * l(v, h) = net(enc(v), h) with v the offset in the DETECTOR-aligned frame in arcsec (v = R^T u in the package; we tabulate directly in v);
  * the blur-gauge dilation `dil` (v -> v / dil), recomputed from the state exactly as FieldOperator.dilation does (result: 1, floor inactive);
  * the normalisation Z: sum over the package's nested-level quadrature (Levels, render pitch 0.005" SW / 0.01" LW, 24 half-nodes, factor 2,
    10" extent, taper from 8.5") of exp(l) x window x cell area, with the rotation R_e of a real exposure of the same filter and detector;
  * the taper: effective kernel = exp(l) * smooth_down(r, 8.5", 10") / Z (the level windows sum to this taper).
What is approximated: pixel sizes 0.031" (short-wavelength) and 0.063" (long-wavelength) are taken from the task, not from the run's header medians
(the median |det J|^0.5 per channel was not saved; the render pitches in the log, 0.005" and 0.01", are consistent with them);
the table is evaluated at the filter pivot wavelength with the mean code, not at the five wavelength nodes of the run (tau_lam = 1e-6, so the
difference is below 1e-5 of the kernel); the detector is NRCB3 (short-wavelength filters) and NRCBLONG (long-wavelength filters).
The learned kernel is used here as the full "pixel-in-kernel" response (see report): no extra pixel box is applied.

Output: results/cache/learned_tables.npz (kernels at 16 samples per pixel, +-32 px), results/learned_kernels.json, figs/learned_kernels.png
Run: python3 learned_kernels.py      (about 1 minute, one thread)
"""
import os
import sys
import json
import math
import time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import torch
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fieldop_ref.kernel import Codes, KernelNet, LogPolarEncoding
from fieldop_ref.render import Levels, kernel_level_values, smooth_down

STATE = os.path.join(HERE, "dl", "fieldop_state.pt")
CACHE = os.path.join(HERE, "results", "cache"); FIGS = os.path.join(HERE, "figs")
os.makedirs(CACHE, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

FILTERS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
PIX = {0: 0.031, 1: 0.063}                  # arcsec per detector pixel, channel 0 = short wavelength, 1 = long wavelength
DETNAME = {0: "NRCB1", 1: "NRCB2", 2: "NRCB3", 3: "NRCB4", 4: "NRCBLONG"}
CHAN_RENDER_PITCH = {0: 0.005, 1: 0.01}     # from fieldop.log (render pitch per channel)
N = 16                                      # samples per pixel of the stored table
LSUP = 32                                   # +- pixels
EXTENT = 10.0                               # arcsec, kernel extent of the run
RADII = [1, 2, 4, 8, 16, 32]


def load_model():
    s = torch.load(STATE, map_location="cpu", weights_only=False)
    enc = LogPolarEncoding(r0=0.5 * 0.005, r_max=10.0, n_rad=8, m_max=12)
    net = KernelNet(enc, dh=16, hidden=64, layers=3)
    codes = Codes(6, 1, 5, 123, s["codes.curves"], "curve", dh=16, hidden=64)
    net.load_state_dict({k[4:]: v for k, v in s.items() if k.startswith("net.")}, strict=True)
    codes.load_state_dict({k[6:]: v for k, v in s.items() if k.startswith("codes.")}, strict=True)
    net.eval(); codes.eval()
    tau = {k[8:]: float(torch.nn.functional.softplus(v)) for k, v in s.items() if k.startswith("tau.raw.")}
    return s, net, codes, tau


def mean_code(s, codes, tau, f, det, xi=(0.0, 0.0), at_lambda=None):
    """h at the pivot (or at wavelength at_lambda, um) for filter f, detector det, normalised position xi; no per-exposure code."""
    f_t = torch.tensor([f]); i_t = torch.tensor([0])
    hp = codes.hbar_at(s["piv"][f].reshape(1), f_t, i_t)
    h = hp
    if at_lambda is not None:
        h = hp + tau["k_lambda"] * (codes.hbar_at(torch.tensor([float(at_lambda)]), f_t, i_t) - hp)
    ref = codes.hdet_raw(s["ref_det"], s["ref_xi"])
    mu = ref.mean(0); sc = torch.sqrt(((ref - mu) ** 2).sum(-1).mean() / ref.shape[-1] + 1e-12)
    hd = (codes.hdet_raw(torch.tensor([det]), torch.tensor([list(xi)], dtype=torch.float32)) - mu) / sc
    return h + tau["k_space"] * hd


def dilation(s, net, codes, tau, blur_floor_cfg=1.0, sky_pitch=0.04, diam=6.5):
    """FieldOperator.dilation() reproduced: returns (dil or None, M_ref at dil=1, floor, blur_ref filter)."""
    pix = {0: 0.031, 1: 0.063}
    fchan = {int(f): int(c) for f, c in zip(s["e_filt"], s["e_chan"])}
    sig2 = {f: (0.42 * float(s["piv"][f]) * 1e-6 / diam * 206264.8) ** 2 + pix[c] ** 2 / 12 for f, c in fchan.items()}
    ref_f = min(sig2, key=sig2.get)
    floor = blur_floor_cfg * sky_pitch ** 2 / 3.0
    Rw = 2.0 * math.sqrt(floor); pm = 0.005
    nh = int(math.ceil(4 * Rw / pm))
    g = torch.arange(-nh, nh + 1, dtype=torch.float32) * pm
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    bu = torch.stack([xx.flatten(), yy.flatten()], -1); r2 = (bu ** 2).sum(-1); logw = -0.5 * r2 / Rw ** 2
    Q = s["lam"].shape[1]

    def moment(dil):
        h = []
        for q in range(Q):
            h.append(mean_code(s, codes, tau, ref_f, 0, at_lambda=float(s["lam"][ref_f, q])))
        h = torch.cat(h, 0)
        with torch.no_grad():
            lg = net(net.enc(bu / dil), h) + logw[None]
        w = torch.softmax(lg, -1)
        return (w * r2[None]).sum(-1).mean()
    M1 = float(moment(1.0))
    s1 = math.sqrt(max(1.0, floor / M1))
    if s1 == 1.0:
        return None, M1, floor, FILTERS[ref_f]
    M = moment(s1)
    return float(s1 * torch.sqrt(torch.clamp(floor / M, min=1.0))), M1, floor, FILTERS[ref_f]


def log_density(net, h, v_arcsec, dil=None, chunk=120000):
    """l(v, h) for v (P, 2) arcsec (detector frame)."""
    out = []
    with torch.no_grad():
        for i in range(0, v_arcsec.shape[0], chunk):
            v = v_arcsec[i:i + chunk]
            if dil is not None:
                v = v / dil
            out.append(net(net.enc(v), h)[0])
    return torch.cat(out)


def level_quadrature(net, h, chan, R, dil, extent=EXTENT):
    """Package quadrature: returns log Z, and per-node detector-frame offsets v (P,2) and normalised weights K (P,) summing to 1."""
    lev = Levels(CHAN_RENDER_PITCH[chan], 2, 24, extent, a=0.5, taper=0.85)
    with torch.no_grad():
        Ks, logZ = kernel_level_values(net, lev, R, h, dil=dil)
    vs, ws = [], []
    for l in range(lev.L + 1):
        m = lev.mask[l]
        u = lev.u[l][m]
        vs.append(u @ R)
        ws.append(Ks[l][0][m])
    return float(logZ[0]), torch.cat(vs), torch.cat(ws), lev


def table(net, h, logZ, pix, dil, n=N, L=LSUP):
    """Learned kernel at n samples per pixel, +-L px, as density per pixel^2 (integral 1 over the plane): exp(l) T(r) pix^2 / Z."""
    g = (torch.arange(-L * n, L * n + 1, dtype=torch.float64) / n)
    X, Y = torch.meshgrid(g, g, indexing="xy")
    v = torch.stack([X.flatten(), Y.flatten()], -1).float() * pix
    lg = log_density(net, h, v, dil).double()
    r = v.double().norm(dim=-1)
    T = smooth_down(r, 0.85 * EXTENT, EXTENT)
    return (torch.exp(lg - logZ) * T * pix ** 2).reshape(X.shape).numpy()


def continuous_Z(net, h, dil, pix):
    """Independent normalisation: polar quadrature (log-spaced radius, 720 angles) of exp(l) T(r) over the disc r <= extent. Returns log Z."""
    nr, na = 3000, 720
    rr = torch.cat([torch.zeros(1), torch.logspace(math.log10(1e-3), math.log10(EXTENT), nr)]).double()
    ang = (torch.arange(na, dtype=torch.float64) + 0.5) * 2 * math.pi / na
    # trapezoid in r on the nodes (r d r d theta)
    w_r = torch.zeros_like(rr)
    w_r[1:-1] = 0.5 * (rr[2:] - rr[:-2]); w_r[0] = 0.5 * (rr[1] - rr[0]); w_r[-1] = 0.5 * (rr[-1] - rr[-2])
    tot = 0.0
    for i in range(0, nr + 1, 200):
        r = rr[i:i + 200]
        v = torch.stack([(r[:, None] * torch.cos(ang)[None]).reshape(-1), (r[:, None] * torch.sin(ang)[None]).reshape(-1)], -1).float()
        lg = log_density(net, h, v, dil).double().reshape(len(r), na)
        T = smooth_down(r, 0.85 * EXTENT, EXTENT)[:, None]
        tot += float((torch.exp(lg) * T * (r * w_r[i:i + 200])[:, None]).sum() * 2 * math.pi / na)
    return math.log(tot)


def fwhm_cut(T, n, i0, j0, ax):
    prof = T[i0, :] if ax == 1 else T[:, j0]
    pk = prof.max(); idx = np.where(prof >= pk / 2)[0]
    a, b = idx[0], idx[-1]
    lft = a - 1 + (pk / 2 - prof[a - 1]) / (prof[a] - prof[a - 1])
    rgt = b + (prof[b] - pk / 2) / (prof[b] - prof[b + 1])
    return float((rgt - lft) / n)


def summarise(T, n=N, outside=0.0):
    """Properties of the table T (density per pixel^2, sampled n per pixel, origin at the centre)."""
    c = T.shape[0] // 2
    g = (np.arange(T.shape[0]) - c) / n
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X, Y)
    i0, j0 = np.unravel_index(np.argmax(T), T.shape)
    pk = float(T.max())
    inside = float(T.sum() / n ** 2)
    out = dict(peak_per_pixel2=pk, peak_at_px=[float(g[j0]), float(g[i0])], min_value=float(T.min()), n_negative=int((T < 0).sum()),
               integral_inside_table=inside, integral_outside_table_levels=outside,
               centroid_px=[float((T * X).sum() / T.sum()), float((T * Y).sum() / T.sum())])
    out["fwhm_x_px"] = fwhm_cut(T, n, i0, j0, 1); out["fwhm_y_px"] = fwhm_cut(T, n, i0, j0, 0)
    out["fwhm_mean_cut_px"] = 0.5 * (out["fwhm_x_px"] + out["fwhm_y_px"])
    out["fwhm_area_equiv_px"] = float(2 * math.sqrt((T >= pk / 2).sum() / n ** 2 / math.pi))
    # encircled energy about the table origin, with the Riemann sum at n/px (cell-centred weights), normalised by the TOTAL integral (inside + outside)
    tot = inside + outside
    out["encircled_energy_about_origin"] = {str(r): float(T[R <= r].sum() / n ** 2 / tot) for r in RADII}
    Rp = np.hypot(X - g[j0], Y - g[i0])
    out["encircled_energy_about_peak"] = {str(r): float(T[Rp <= r].sum() / n ** 2 / tot) for r in RADII[:5]}
    # structure: azimuthal scatter and spikes in annuli
    ann = []
    for r0, r1 in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]:
        sel = (R >= r0) & (R < r1) & (T > 0)
        lg = np.log(T[sel])
        # azimuthal rms of log T at fixed radius: remove the radial trend with the median of narrow rings
        rr = R[sel]; bins = np.floor(rr * 4).astype(int)
        res = np.empty_like(lg)
        for b in np.unique(bins):
            m = bins == b; res[m] = lg[m] - np.median(lg[m])
        med_ann = {b: np.median(T[sel][bins == b]) for b in np.unique(bins)}
        Tsel = T[sel]; mb = np.array([med_ann[b] for b in bins])
        spike = Tsel > 2 * mb
        ann.append(dict(annulus_px=[r0, r1], azimuthal_rms_dex=float(res.std() / math.log(10)), max_over_median_dex=float(np.log10(Tsel / mb).max()),
                        frac_pixels_above_2x_median=float(spike.mean()), flux_share_in_spike_pixels=float(Tsel[spike].sum() / Tsel.sum())))
    out["structure_by_annulus"] = ann
    # radial profile (azimuthal median), count of local rises (rings)
    edges = np.arange(0.5, 31.5, 0.5)
    prof = np.array([np.median(T[(R >= a) & (R < a + 0.5)]) for a in edges])
    rises = [(float(edges[i + 1]), float(prof[i + 1] / prof[i])) for i in range(len(prof) - 1) if prof[i + 1] > 1.02 * prof[i] and edges[i] > 1.0]
    out["radial_median_profile_rises_gt_2pct_beyond_1px"] = rises[:20]
    out["n_radial_rises"] = len(rises)
    out["dynamic_range_dex_peak_to_32px_median"] = float(math.log10(pk / prof[-1]))
    return out


def main():
    t00 = time.time()
    s, net, codes, tau = load_model()
    dil, M1, floor, bref = dilation(s, net, codes, tau)
    print("taus", tau); print("dilation", dil, "M_ref", M1, "floor", floor, "ref filter", bref, flush=True)
    res = dict(source="fieldop_smacs_fix_200_s0/fieldop_state.pt", taus_final=tau, dilation=dil, blur_moment_ref_filter=bref, blur_moment_at_dil1=M1,
               blur_floor=floor, pixel_arcsec=PIX, kernels={})
    tabs = {}
    for f, name in enumerate(FILTERS):
        ch = int(s["e_chan"][(s["e_filt"] == f).nonzero()[0, 0]])
        det = 4 if ch == 1 else 2
        pix = PIX[ch]
        h = mean_code(s, codes, tau, f, det)
        # rotations of the real exposures of this filter on this detector
        ex = ((s["e_filt"] == f) & (s["e_det"] == det)).nonzero()[:, 0]
        Rs = s["e_R"][ex]
        info = dict(channel=ch, detector=DETNAME[det], pixel_arcsec=pix, pivot_um=float(s["piv"][f]), n_exposures_for_R=len(ex))
        logZs = []
        for i in (0, len(ex) // 2, len(ex) - 1):
            lz, v, w, lev = level_quadrature(net, h, ch, Rs[i], dil)
            logZs.append(lz)
        info["logZ_levels_three_exposure_rotations"] = logZs
        info["Z_levels_spread_over_rotations_rel"] = float(np.ptp(np.exp(logZs)) / np.exp(logZs[0]))
        info["rotation_matrix_exposure0"] = Rs[0].tolist()
        lz, v, w, lev = level_quadrature(net, h, ch, Rs[0], dil)
        # independent continuous normalisation
        lzc = continuous_Z(net, h, dil, pix)
        info["logZ_levels"] = lz; info["logZ_continuous_polar_quadrature"] = lzc
        info["Z_levels_over_Z_continuous"] = float(math.exp(lz - lzc))
        # flux of the level-quadrature kernel outside the +-32 px square (detector frame) and outside the +-LSUP circle
        vp = (v / pix).numpy()
        sq = np.maximum(np.abs(vp[:, 0]), np.abs(vp[:, 1]))
        wn = w.numpy().astype(np.float64)
        info["level_quadrature_total_weight"] = float(wn.sum())
        info["fraction_outside_square_pm32px_levels"] = float(wn[sq > LSUP].sum() / wn.sum())
        info["fraction_outside_circle_32px_levels"] = float(wn[np.hypot(vp[:, 0], vp[:, 1]) > LSUP].sum() / wn.sum())
        for L_ in (8, 16, 64, 128):
            info[f"fraction_outside_square_pm{L_}px_levels"] = float(wn[sq > L_].sum() / wn.sum())
        # same fraction with the continuous (polar) quadrature, independent of the level grids
        T = table(net, h, lz, pix, dil)
        tabs[name] = T
        inside_sq = float(T.sum() / N ** 2)
        info["table_integral_inside_square_pm32px"] = inside_sq
        # kernel-bias check: table integral at 16/px against level quadrature inside the square
        info["levels_weight_inside_square_pm32px"] = float(wn[sq <= LSUP].sum() / wn.sum())
        outside = float(wn[sq > LSUP].sum() / wn.sum())
        res["kernels"][name] = dict(**info, **summarise(T, N, outside))
        # variation with detector position (check that the pruned spatial term does nothing) and with wavelength node
        h_c = mean_code(s, codes, tau, f, det, xi=(0.9, 0.9))
        d_pos = float((log_density(net, h_c, torch.tensor([[0.0, 0.0], [0.5 * pix, 0.0], [3 * pix, 0.0]]), dil) -
                       log_density(net, h, torch.tensor([[0.0, 0.0], [0.5 * pix, 0.0], [3 * pix, 0.0]]), dil)).abs().max())
        h_l = mean_code(s, codes, tau, f, det, at_lambda=float(s["lam"][f, 0]))
        d_lam = float((log_density(net, h_l, torch.tensor([[0.0, 0.0], [0.5 * pix, 0.0], [3 * pix, 0.0]]), dil) -
                       log_density(net, h, torch.tensor([[0.0, 0.0], [0.5 * pix, 0.0], [3 * pix, 0.0]]), dil)).abs().max())
        res["kernels"][name]["max_abs_change_log_kernel_detector_corner_vs_centre"] = d_pos
        res["kernels"][name]["max_abs_change_log_kernel_first_vs_pivot_wavelength_node"] = d_lam
        k = res["kernels"][name]
        print(name, ch, "FWHM cuts %.3f %.3f area %.3f" % (k["fwhm_x_px"], k["fwhm_y_px"], k["fwhm_area_equiv_px"]), "EE", {a: round(b, 4) for a, b in k["encircled_energy_about_origin"].items()},
              "outside32 %.4f" % outside, "Zlev/Zc %.4f" % k["Z_levels_over_Z_continuous"], "min %.2e" % k["min_value"], "t=%.0f" % (time.time() - t00), flush=True)
    np.savez_compressed(os.path.join(CACHE, "learned_tables.npz"), **{f"T16_{k}": v for k, v in tabs.items()})
    json.dump(res, open(os.path.join(HERE, "results", "learned_kernels.json"), "w"), indent=1, default=float)
    make_figure(tabs, res)


def make_figure(tabs, res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(3, 6, figsize=(21, 10.5), gridspec_kw=dict(height_ratios=[1, 1, 0.8]))
    for j, (name, T) in enumerate(tabs.items()):
        c = T.shape[0] // 2
        pk = T.max()
        for i, (hw, ttl) in enumerate([(32, "+-32 px"), (4, "+-4 px")]):
            sub = T[c - hw * N:c + hw * N + 1, c - hw * N:c + hw * N + 1] / pk
            im = axs[i, j].imshow(sub, origin="lower", extent=[-hw, hw, -hw, hw], norm=LogNorm(vmin=1e-7, vmax=1), cmap="magma")
            axs[i, j].set_title(f"{name} {ttl}" + (f"  (FWHM {res['kernels'][name]['fwhm_mean_cut_px']:.2f} px)" if i == 1 else ""), fontsize=10)
            if j == 0: axs[i, j].set_ylabel("y, detector pixels")
            axs[i, j].set_xlabel("x, detector pixels")
        g = (np.arange(T.shape[0]) - c) / N
        X, Y = np.meshgrid(g, g); R = np.hypot(X, Y)
        edges = np.arange(0, 32, 0.25)
        med = np.array([np.median(T[(R >= a) & (R < a + 0.25)]) for a in edges]) / pk
        mx = np.array([T[(R >= a) & (R < a + 0.25)].max() for a in edges]) / pk
        mn = np.array([T[(R >= a) & (R < a + 0.25)].min() for a in edges]) / pk
        ax = axs[2, j]
        ax.fill_between(edges + 0.125, mn, mx, color="#9ecae1", alpha=0.6, label="min to max in ring")
        ax.plot(edges + 0.125, med, color="#08519c", lw=1.5, label="median in ring")
        ax.set_yscale("log"); ax.set_xscale("log"); ax.set_xlim(0.1, 32); ax.set_ylim(1e-6, 1.5); ax.set_xlabel("radius, detector pixels"); ax.grid(alpha=0.3, which="both")
        if j == 0: ax.set_ylabel("kernel / peak"); ax.legend(fontsize=8)
    cb = fig.colorbar(im, ax=axs[:2, :], shrink=0.6, pad=0.01); cb.set_label("kernel / peak (logarithmic)")
    fig.suptitle("Learned NIRCam kernels (fieldop, SMACS 200-minute run), pivot wavelength, mean code, detector centre, detector frame; "
                 "short-wavelength 0.031\"/px, long-wavelength 0.063\"/px", fontsize=11)
    fig.savefig(os.path.join(FIGS, "learned_kernels.png"), dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
