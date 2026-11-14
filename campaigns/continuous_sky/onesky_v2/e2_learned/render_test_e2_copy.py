#!/usr/bin/env python3
"""E2: rendering an arbitrary profile through an undersampled diffraction kernel.

Route 1 (closed form): profile = K non-negative Gaussians, kernel = non-negative circular Gaussians on a
          pyramid of sizes; image = double sum of Gaussians (covariances add).
Route 2 (local sampling): sample the profile on a grid n times finer than the detector pixels,
          convolve with the kernel tabulated at the same sampling, bin to pixels.

STAND-IN KERNELS: analytic pupil (hexagon + central obscuration + 3 struts) -> |FFT|^2 -> pixel box -> small
Gaussian. They are placeholders to be replaced later by kernels learned from real data.

Usage: python3 render_test.py            (all stages, in order; the reference images are cached in results/cache)
       python3 render_test.py --stage kernels|fits|refs|route1|route2|timing|final   (one stage)
       python3 render_test.py --fresh   (delete the cache first)
All lengths are detector pixels. Single thread everywhere.
"""
import os, sys, json, time, argparse, math, shutil
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import scipy.fft as sfft
from scipy.signal import fftconvolve
from scipy.optimize import nnls, minimize
from scipy.special import gammaincinv, gamma
import torch
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CACHE = os.path.join(RES, "cache")
FIGS = os.path.join(HERE, "figs")
os.makedirs(CACHE, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

JEXT = 256                    # periodic extent of the kernel calculation, pixels (so +-128)
LSUP = 32                     # tabulated kernel support, +-32 pixels
NLIST = [1, 2, 3, 4, 6, 8, 12, 16]
CASES = {"sharp": 1.15, "wide": 2.1}    # target FWHM (mean of x and y cuts), pixels. 1.0 is unreachable: a 1 px box plus 0.2 px diffusion alone gives 1.07
SIG_DIFF = 0.2                # detector diffusion Gaussian width, pixels
REF_N = 16                    # reference sub-samples per pixel
NUM_N = 8                     # sub-samples used for the "part" numerics of route 1
KS = [2, 4, 8, 16]
SEED = 7


def jload(name, default=None):
    p = os.path.join(CACHE, name)
    return json.load(open(p)) if os.path.exists(p) else default


def jsave(name, obj, folder=CACHE):
    json.dump(obj, open(os.path.join(folder, name), "w"), indent=1, default=float)


# ----------------------------------------------------------------------------------------------
# stand-in kernels
# ----------------------------------------------------------------------------------------------
def pupil_mask(d, ss=3):
    B = d + 6
    n = B * ss
    c = (np.arange(n) + 0.5 - n / 2) / (d * ss)          # units of D (flat-to-flat)
    x, y = np.meshgrid(c, c)
    hexm = (np.abs(x) <= 0.5) & (np.abs(x) * 0.5 + np.abs(y) * np.sqrt(3) / 2 <= 0.5)
    m = hexm & ~(x ** 2 + y ** 2 < (0.11 / 2) ** 2)     # central obscuration, diameter 0.11 D
    for ang in (90, 210, 330):                           # three struts, width 0.02 D
        a = np.deg2rad(ang)
        along = x * np.cos(a) + y * np.sin(a)
        perp = np.abs(-x * np.sin(a) + y * np.cos(a))
        m &= ~((along > 0) & (perp < 0.01))
    return m.reshape(B, ss, B, ss).mean((1, 3))


def build_kernels(d, n, want_T=True):
    """Return K (optics+diffusion, unit sum per fine cell) and T (K convolved with a 1x1 pixel box, unit
    integral) at n samples per pixel, cropped to +-LSUP. Pixel box and diffusion applied exactly in Fourier space."""
    M = n * JEXT
    B = pupil_mask(d).shape[0]
    P = np.zeros((M, M))
    s0 = M // 2 - B // 2
    P[s0:s0 + B, s0:s0 + B] = pupil_mask(d)
    A = sfft.fft2(P); del P
    I = (A.real ** 2 + A.imag ** 2); del A
    Ih = sfft.rfft2(I); del I
    fy = sfft.fftfreq(M, d=1.0 / n)[:, None]
    fx = sfft.rfftfreq(M, d=1.0 / n)[None, :]
    Hg = np.exp(-2 * np.pi ** 2 * SIG_DIFF ** 2 * (fx ** 2 + fy ** 2))
    Ih *= Hg
    c = M // 2
    sl = slice(c - LSUP * n, c + LSUP * n + 1)
    K = sfft.fftshift(sfft.irfft2(Ih, s=(M, M)))[sl, sl].copy()
    T = None
    if want_T:
        Ih *= np.sinc(fx) * np.sinc(fy)
        T = sfft.fftshift(sfft.irfft2(Ih, s=(M, M)))[sl, sl].copy()
    del Ih
    K /= K.sum()
    if T is not None:
        T /= T.sum() / n ** 2
    return K, T


def fwhm_equiv(T, n):
    return 2 * math.sqrt((T >= T.max() / 2).sum() / n ** 2 / math.pi)


def calibrate(target):
    cache = {}

    def f(d):
        if d not in cache:
            _, T = build_kernels(d, 4)
            cache[d] = 0.5 * (fwhm_axis(T, 4, 0) + fwhm_axis(T, 4, 1))
        return cache[d]
    lo, hi = 30, 600
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) > target: lo = mid      # FWHM in pixels falls as the pixel grows (d grows)
        else: hi = mid
    best = min(range(lo - 3, hi + 4), key=lambda d: abs(f(d) - target))
    return best, f(best)


def spectrum(d, nb):
    M = nb * JEXT
    B = pupil_mask(d).shape[0]
    P = np.zeros((M, M)); s0 = M // 2 - B // 2
    P[s0:s0 + B, s0:s0 + B] = pupil_mask(d)
    A = sfft.fft2(P); del P
    I = (A.real ** 2 + A.imag ** 2); del A
    Ih = sfft.rfft2(I); del I
    fy = sfft.fftfreq(M, d=1.0 / nb)[:, None]; fx = sfft.rfftfreq(M, d=1.0 / nb)[None, :]
    Ih *= np.exp(-2 * np.pi ** 2 * SIG_DIFF ** 2 * (fx ** 2 + fy ** 2))
    return Ih, fx, fy


def table_from_spectrum(Ih, fx, fy, nb, n, pixel_box, cell_avg):
    """Kernel at n samples per pixel from the nb-sample spectrum (n divides nb). cell_avg: average over a cell of
    width 1/n at the source side (T tables) or at both source and destination side (K tables, binned)."""
    M = nb * JEXT; st = nb // n
    H = Ih
    if pixel_box: H = H * (np.sinc(fx) * np.sinc(fy))
    if cell_avg:
        H = H * (np.sinc(fx / n) * np.sinc(fy / n)) ** (1 if pixel_box else 2)
    c = M // 2
    full = sfft.fftshift(sfft.irfft2(H, s=(M, M)))
    sl = slice(c - LSUP * nb, c + LSUP * nb + 1, st)
    t = full[sl, sl].copy()
    return t / t.sum() if not pixel_box else t / (t.sum() / n ** 2)


def stage_kernels():
    out = {}
    for case, tgt in CASES.items():
        fn = os.path.join(CACHE, f"kern_{case}.npz")
        t0 = time.time()
        d, fw = calibrate(tgt)
        tabs = {}
        for nb, ns in ((16, (16, 8, 4, 2, 1)), (12, (12, 6, 3))):
            Ih, fx, fy = spectrum(d, nb)
            for n in ns:
                tabs[f"K{n}"] = table_from_spectrum(Ih, fx, fy, nb, n, False, True)    # binned route: cell-integrated, unit sum
                tabs[f"T{n}"] = table_from_spectrum(Ih, fx, fy, nb, n, True, True)     # pixel box inside, source-cell averaged, unit integral
                tabs[f"Kp{n}"] = table_from_spectrum(Ih, fx, fy, nb, n, False, False)  # point-sampled (no cell averaging), unit sum
                tabs[f"Tp{n}"] = table_from_spectrum(Ih, fx, fy, nb, n, True, False)   # point-sampled pixel-integrated kernel, unit integral
            if nb == 16:
                tabs["P16"] = table_from_spectrum(Ih, fx, fy, 16, 16, True, False)      # pixel-integrated kernel at points (fit target / exact kernel)
            del Ih
        np.savez(fn, d=d, **tabs)
        T16 = tabs["P16"]
        out[case] = dict(pupil_cells_across_D=d, pixel_in_lambda_over_D=d / JEXT, fwhm_axis_mean_px=fw, fwhm_area_equiv_px=fwhm_equiv(T16, 16), target_fwhm_px=tgt,
                         fwhm_x_px=fwhm_axis(T16, 16, 0), fwhm_y_px=fwhm_axis(T16, 16, 1),
                         peak_pixel_integrated_kernel=float(T16.max()), seconds=time.time() - t0)
        print(case, out[case], flush=True)
    jsave("kernel_info.json", out)


def load_tabs(case):
    z = np.load(os.path.join(CACHE, f"kern_{case}.npz"))
    return {k: z[k] for k in z.files if k != "d"}, int(z["d"])


def fwhm_axis(T, n, ax):
    c = T.shape[0] // 2
    prof = T[c, :] if ax == 1 else T[:, c]
    pk = prof.max(); idx = np.where(prof >= pk / 2)[0]
    i0, i1 = idx[0], idx[-1]
    l = i0 - 1 + (pk / 2 - prof[i0 - 1]) / (prof[i0] - prof[i0 - 1])
    r = i1 + (prof[i1] - pk / 2) / (prof[i1] - prof[i1 + 1])
    return float((r - l) / n)


# ----------------------------------------------------------------------------------------------
# profiles (black-box functions f(u, v): surface brightness per square pixel, unit flux)
# ----------------------------------------------------------------------------------------------
class Sersic:
    def __init__(s, n, re, q, th, cx, cy):
        s.n, s.re, s.q, s.th, s.cx, s.cy = n, re, q, th, cx, cy
        s.b = float(gammaincinv(2 * n, 0.5))
        s.I0 = 1.0 / (2 * np.pi * q * re ** 2 * n * gamma(2 * n) * s.b ** (-2 * n))

    def rell(s, u, v):
        x = u - s.cx; y = v - s.cy
        c, sn = np.cos(s.th), np.sin(s.th)
        xp = x * c + y * sn; yp = -x * sn + y * c
        return np.sqrt(xp ** 2 + (yp / s.q) ** 2), np.arctan2(yp / s.q, xp)

    def __call__(s, u, v):
        r, _ = s.rell(u, v)
        return s.I0 * np.exp(-s.b * (r / s.re) ** (1.0 / s.n))


CLUMPS = [((10.0, -5.0), 1.5, 0.03), ((-14.0, 8.0), 2.5, 0.05), ((5.0, 16.0), 3.0, 0.02)]


class Irregular:
    """Sersic n=1 disc (re=8, q=0.7) x two-armed logarithmic spiral (amplitude 0.5) + three clumps."""
    def __init__(s, th, cx, cy):
        s.disc = Sersic(1.0, 8.0, 0.7, th, cx, cy)
        s.cx, s.cy, s.th = cx, cy, th
        s.fdisc = 1.0 - sum(c[2] for c in CLUMPS)
        s.pitch = np.deg2rad(25.0)

    def __call__(s, u, v):
        r, phi = s.disc.rell(u, v)
        r = np.maximum(r, 1e-9)
        mod = 1.0 + 0.5 * np.cos(2 * phi - 2.0 / np.tan(s.pitch) * np.log(r / 8.0))
        out = s.fdisc * s.disc(u, v) * mod
        for (dx, dy), sg, fl in CLUMPS:
            out = out + fl / (2 * np.pi * sg ** 2) * np.exp(-((u - s.cx - dx) ** 2 + (v - s.cy - dy) ** 2) / (2 * sg ** 2))
        return out


class GaussMix:
    def __init__(s, a, m, S):
        s.a, s.m, s.S = np.asarray(a), np.asarray(m), np.asarray(S)

    def __call__(s, u, v):
        out = 0.0
        for k in range(len(s.a)):
            Sk = s.S[k]; det = Sk[0, 0] * Sk[1, 1] - Sk[0, 1] ** 2
            dx = u - s.m[k, 0]; dy = v - s.m[k, 1]
            q = (Sk[1, 1] * dx ** 2 - 2 * Sk[0, 1] * dx * dy + Sk[0, 0] * dy ** 2) / det
            out = out + s.a[k] / (2 * np.pi * np.sqrt(det)) * np.exp(-0.5 * q)
        return out


def make_items():
    rng = np.random.default_rng(SEED)
    th = float(rng.uniform(0, np.pi))                       # one orientation
    offs = np.round(rng.uniform(-0.5, 0.5, (3, 2)) * 8) / 8  # 3 sub-pixel positions, quantised to 1/8 px
    items = [dict(id="point", kind="point", re=0.0, S=129)]
    for nn in (1, 4):
        for re in (0.5, 2.0, 8.0, 32.0):
            items.append(dict(id=f"sersic{nn}_re{re:g}", kind="sersic", n=nn, re=re, S=257 if re == 32 else 129))
    items.append(dict(id="irregular", kind="irregular", re=8.0, S=129))
    return items, th, offs


def make_profile(it, th, off):
    cx, cy = float(off[0]), float(off[1])
    if it["kind"] == "sersic": return Sersic(it["n"], it["re"], 0.5, th, cx, cy)
    if it["kind"] == "irregular": return Irregular(th, cx, cy)
    return None


def refine_radius(prof):
    re = getattr(prof, "re", None)
    if re is None: re = getattr(getattr(prof, "disc", None), "re", 0.5)
    return min(4.0 * re, 6.0)


def sample_grid(prof, xs, ys, cell, refine, cx, cy):
    """Flux per cell = f(centre)*cell^2 ; optionally the cells within 5 cells of the item centre use 8x8 sub-sampling
    (cusp handling). xs, ys are cell-centre coordinates."""
    F = np.empty((len(ys), len(xs)))
    for i0 in range(0, len(ys), 128):
        F[i0:i0 + 128] = prof(xs[None, :], ys[i0:i0 + 128, None])
    F *= cell ** 2
    if refine:
        rr = max(refine_radius(prof), 5.5 * cell)
        ix = np.where(np.abs(xs - cx) <= rr)[0]; iy = np.where(np.abs(ys - cy) <= rr)[0]
        ns = 6
        if len(ix) and len(iy):
            sub = (np.arange(ns) + 0.5) / ns - 0.5
            X = (xs[ix][:, None] + sub[None, :] * cell).ravel(); Y = (ys[iy][:, None] + sub[None, :] * cell).ravel()
            V = np.empty((len(iy), ns, len(ix), ns))
            for j0 in range(0, len(iy), 16):
                V[j0:j0 + 16] = prof(X[None, :], Y[j0 * ns:(j0 + 16) * ns, None]).reshape(-1, ns, len(ix), ns)
            F[np.ix_(iy, ix)] = V.mean((1, 3)) * cell ** 2
    return F


# ----------------------------------------------------------------------------------------------
# rendering: route 2 and the numerical reference
# ----------------------------------------------------------------------------------------------
def crop(tab, n, L):
    c = tab.shape[0] // 2
    return tab[c - L * n:c + L * n + 1, c - L * n:c + L * n + 1]


def render_bin(prof, Kt, n, S, L, off, refine=False, delta=False):
    """Profile sampled at cell centres of an n-times finer grid (margin L px beyond the stamp), convolved with the
    fine kernel Kt (already cropped to +-L px), cells binned (summed) into pixels."""
    h = S // 2; half = h + L; Npx = 2 * half + 1
    xs = (np.arange(Npx * n) + 0.5) / n - 0.5 - half
    if delta:
        F = np.zeros((Npx * n, Npx * n))
        ix = int(np.floor((off[0] + 0.5 + half) * n)); iy = int(np.floor((off[1] + 0.5 + half) * n))
        F[iy, ix] = 1.0
    else:
        F = sample_grid(prof, xs, xs, 1.0 / n, refine, off[0], off[1])
    out = fftconvolve(F, Kt, mode="valid")
    return out.reshape(S, n, S, n).sum((1, 3))


def render_T(prof, Tt, n, S, L, off, refine=False, delta=False, gm=None):
    """Same, but the kernel table Tt already contains the exact 1x1 pixel box; the sampled grid has points on the
    pixel centres and the convolved grid is read at the pixel centres (no binning)."""
    h = S // 2; half = h + L
    xs = np.arange(-half * n, half * n + 1) / n
    if delta:
        F = np.zeros((len(xs), len(xs)))
        F[int(round(off[1] * n)) + half * n, int(round(off[0] * n)) + half * n] = 1.0
    else:
        F = sample_grid(prof, xs, xs, 1.0 / n, refine, off[0], off[1])
    out = fftconvolve(F, Tt, mode="valid")
    return out[::n, ::n]


def metrics(img, ref):
    pk = ref.max(); e = img - ref
    return dict(max=float(np.abs(e).max() / pk), rms=float(np.sqrt(np.mean(e ** 2)) / pk),
                flux=float((img.sum() - ref.sum()) / ref.sum()))


def reference(it, th, off, tabs):
    """Brute force: profile on 16 sub-samples per pixel, FFT convolution with the 16x kernel, block sum."""
    if it["kind"] == "point":
        c = tabs["P16"].shape[0] // 2; S = it["S"]; h = S // 2
        g = np.arange(-h, h + 1)
        X, Y = np.meshgrid(g - off[0], g - off[1])
        ix = np.round(X * 16).astype(int) + c; iy = np.round(Y * 16).astype(int) + c
        ok = (ix >= 0) & (ix < 2 * c + 1) & (iy >= 0) & (iy < 2 * c + 1)
        img = np.zeros((S, S)); img[ok] = tabs["P16"][iy[ok], ix[ok]]
        return img, None
    prof = make_profile(it, th, off)
    S = it["S"]
    r = {}
    for flav in ("Kp", "K"):
        for n in (8, 16):
            r[flav, n] = render_bin(prof, crop(tabs[f"{flav}{n}"], n, LSUP), n, S, LSUP, off, refine=True)
    Rp = (4 * r["Kp", 16] - r["Kp", 8]) / 3.0          # Richardson extrapolation of the point-sampled scheme
    Rc = (4 * r["K", 16] - r["K", 8]) / 3.0            # independent scheme (cell-integrated kernel)
    pk = Rp.max()
    info = dict(raw_16_vs_8_max_over_peak=float(np.abs(r["Kp", 16] - r["Kp", 8]).max() / pk),
                raw_16_vs_8_rms_over_peak=float(np.sqrt(np.mean((r["Kp", 16] - r["Kp", 8]) ** 2)) / pk),
                raw_16_vs_reference_max_over_peak=float(np.abs(r["Kp", 16] - Rp).max() / pk),
                richardson_vs_cellintegrated_scheme_max_over_peak=float(np.abs(Rp - Rc).max() / pk),
                flux_16_vs_8=float((r["Kp", 16].sum() - r["Kp", 8].sum()) / r["Kp", 16].sum()),
                peak=float(pk), flux_in_stamp=float(Rp.sum()))
    return Rp, info


def stage_refs():
    items, th, offs = make_items()
    conv = jload("ref_convergence.json", {})
    for case in CASES:
        tabs, _ = load_tabs(case)
        for it in items:
            for p in range(3):
                key = f"{case}|{it['id']}|{p}"
                fn = os.path.join(CACHE, f"ref_{case}_{it['id']}_{p}.npy")
                if os.path.exists(fn) and key in conv: continue
                t0 = time.time()
                ref, info = reference(it, th, offs[p], tabs)
                np.save(fn, ref)
                conv[key] = info if info is not None else dict(note="point source: reference is the tabulated pixel-integrated kernel at the exact 1/8-px offset; no 8-vs-16 test")
                conv[key]["seconds"] = time.time() - t0
                print(key, {k: v for k, v in conv[key].items() if k != "note"}, flush=True)
                jsave("ref_convergence.json", conv)


def load_ref(case, iid, p):
    return np.load(os.path.join(CACHE, f"ref_{case}_{iid}_{p}.npy"))


# ----------------------------------------------------------------------------------------------
# route 1 pieces: kernel Gaussians
# ----------------------------------------------------------------------------------------------
KCONFIGS = [(3, 1, 1.0), (4, 1, 1.0), (5, 2, 1.0), (6, 2, 1.0), (7, 2, 1.0), (8, 2, 1.0), (7, 3, 1.0),
            (7, 3, 0.5), (7, 4, 0.5)]   # last two: diagnostic, stencil spacing half a width (not the specified layout)
SIG0 = 0.35


def stencil(L, m, sp=1.0):
    P = []
    for l in range(L):
        s = SIG0 * 2 ** l
        g = np.arange(-m, m + 1) * s * sp
        P += [(cx, cy, s) for cx in g for cy in g]
    return np.array(P)


def gauss_design(P, X, Y):
    A = np.exp(-((X.ravel()[:, None] - P[:, 0]) ** 2 + (Y.ravel()[:, None] - P[:, 1]) ** 2) / (2 * P[:, 2] ** 2)) / (2 * np.pi * P[:, 2] ** 2)
    return A


def eval_kernel_gauss(P, w, X, Y, chunk=100000):
    out = np.empty(X.size); Xr, Yr = X.ravel(), Y.ravel()
    nz = w > 0
    for i in range(0, X.size, chunk):
        A = gauss_design(P[nz], Xr[i:i + chunk], Yr[i:i + chunk])
        out[i:i + chunk] = A @ w[nz]
    return out.reshape(X.shape)


def spike_mask(T, n):
    c = T.shape[0] // 2
    g = (np.arange(T.shape[0]) - c) / n
    X, Y = np.meshgrid(g, g); R = np.hypot(X, Y)
    msk = np.zeros(T.shape, bool)
    bins = np.arange(3, 30, 1.0)
    for b0 in bins:
        sel = (R >= b0) & (R < b0 + 1)
        med = np.median(T[sel])
        msk |= sel & (T > 2.0 * med)
    return msk, R


def stage_fits():
    """Kernel Gaussian fits (non-negative least squares via QR) and profile Gaussian fits."""
    out = {}
    save = {}
    for case in CASES:
        tabs, d = load_tabs(case)
        T16 = tabs["P16"]; T4 = T16[::4, ::4]
        c4 = T4.shape[0] // 2
        g = (np.arange(T4.shape[0]) - c4) / 4.0
        X, Y = np.meshgrid(g, g)
        pk = T4.max()
        out[case] = []
        for (L, m, sp) in KCONFIGS:
            t0 = time.time()
            P = stencil(L, m, sp)
            A = gauss_design(P, X, Y)
            sc = np.linalg.norm(A, axis=0)
            Q, R = np.linalg.qr(A / sc)
            y = Q.T @ T4.ravel()
            w, _ = nnls(R, y, maxiter=20000)
            w = w / sc
            wu = np.linalg.lstsq(R, y, rcond=1e-13)[0]               # diagnostic: same stencil, weights allowed to be negative
            fit_u = (Q @ (R @ wu)).reshape(X.shape)
            unc_max = float(np.abs(fit_u - T4).max() / pk); unc_rms = float(np.sqrt(np.mean((fit_u - T4) ** 2)) / pk)
            neg_frac = float(np.abs(wu[wu < 0] / sc[wu < 0]).sum() / max(np.abs(wu / sc).sum(), 1e-300))
            del A, Q, fit_u
            fit = eval_kernel_gauss(P, w, X, Y).reshape(X.shape)
            e = (fit - T4) / pk
            msk, R_ = spike_mask(T4, 4)
            rec = dict(L=L, m=m, spacing_in_widths=sp, unconstrained_sign_max_err=unc_max, unconstrained_sign_rms_err=unc_rms, unconstrained_negative_weight_share=neg_frac, n_stencil=len(P), n_nonzero=int((w > 0).sum()), max_err=float(np.abs(e).max()),
                       rms_err=float(np.sqrt(np.mean(e ** 2))), flux_captured=float(w.sum() / (T4.sum() / 16.0)),
                       core_max_err_r_lt_2=float(np.abs(e[R_ < 2]).max()),
                       mid_max_err_2_to_8=float(np.abs(e[(R_ >= 2) & (R_ < 8)]).max()),
                       outer_max_err_gt_8=float(np.abs(e[R_ >= 8]).max()),
                       spike_pixels=int(msk.sum()), spike_rms_err=float(np.sqrt(np.mean(e[msk] ** 2))),
                       spike_max_err=float(np.abs(e[msk]).max()),
                       spike_flux_ratio_fit_over_true=float(fit[msk].sum() / T4[msk].sum()), seconds=time.time() - t0)
            out[case].append(rec)
            save[f"{case}_{L}_{m}_{sp}"] = np.concatenate([P, w[:, None]], 1)
            print(case, rec, flush=True)
    np.savez(os.path.join(CACHE, "kernel_gauss.npz"), **save)
    jsave("kernel_fits.json", out)


def kernel_choice(case):
    """The standard kernel configuration: the smallest stencil whose max error is within 10 percent of the best one."""
    fits = jload("kernel_fits.json")[case]
    fits = [f for f in fits if f["spacing_in_widths"] == 1.0]
    best = min(f["max_err"] for f in fits)
    cand = [f for f in fits if f["max_err"] <= 1.1 * best]
    f = min(cand, key=lambda r: r["n_nonzero"])
    return f["L"], f["m"]


def load_kgauss(case, L, m, sp=1.0):
    z = np.load(os.path.join(CACHE, "kernel_gauss.npz"))[f"{case}_{L}_{m}_{sp}"]
    nz = z[:, 3] > 0
    return z[nz, :2], z[nz, 2], z[nz, 3]     # centres, sigma, weight


def sersic_gauss_fit(n, re, q, K, rmax):
    b = float(gammaincinv(2 * n, 0.5))
    I0 = 1.0 / (2 * np.pi * q * re ** 2 * n * gamma(2 * n) * b ** (-2 * n))
    r = np.concatenate([[0.0], np.geomspace(1e-3 * re, rmax, 400)])
    dl = np.log(r[2] / r[1])
    tgt = I0 * np.exp(-b * (r / re) ** (1.0 / n))
    # objective: sum (error)^2 r^2 dln r / (I + 0.01 I(re)); i.e. area-weighted, divided by the local brightness (flux-weighted relative error)
    wt = np.sqrt(np.concatenate([[1e-6], r[1:] ** 2 * dl]) / (tgt + 0.01 * I0 * np.exp(-b)))

    def design(lsmin, lsmax):
        sg = np.exp(np.linspace(lsmin, lsmax, K))
        return sg, np.exp(-r[:, None] ** 2 / (2 * sg ** 2)) / (2 * np.pi * q * sg ** 2)

    def solve(A):
        sc = np.linalg.norm(A * wt[:, None], axis=0)
        w, rn = nnls(A * wt[:, None] / sc, tgt * wt, maxiter=5000)
        return w / sc, rn

    def cost(p):
        sg, A = design(*p)
        if not (np.log(0.01 * re) < p[0] < p[1] < np.log(50 * re)): return 1e3
        try: return solve(A)[1]
        except RuntimeError: return 1e3

    best = None
    for s0, s1 in [(np.log(0.1 * re), np.log(3 * re)), (np.log(0.03 * re), np.log(6 * re)), (np.log(0.3 * re), np.log(2 * re))]:
        res = minimize(cost, [s0, s1], method="Nelder-Mead", options=dict(maxiter=250, xatol=1e-3, fatol=1e-14))
        if best is None or res.fun < best.fun: best = res
    lo = np.log(0.01 * re)
    p = best.x
    sg, A = design(*p)
    w, rn = solve(A)
    nz = w > 0
    resid = (A @ w - tgt)
    area_w = np.concatenate([[1e-6], r[1:] ** 2 * dl])
    pk = tgt.max()
    return dict(sigma=sg[nz].tolist(), flux=w[nz].tolist(), rms_over_peak_profile=float(np.sqrt(np.sum(resid ** 2 * area_w) / np.sum(area_w)) / pk),
                nelder_mead_evals=int(best.nfev), converged=bool(best.success), total_flux_fit=float(w.sum()))


def irregular_gauss_fit(K, iters=60):
    """Free Gaussians (free centre and covariance) fitted to the irregular profile on a 0.5 px grid, greedy addition
    followed by joint L-BFGS refinement. Fitted about the item centre (origin), orientation as the item."""
    items, th, offs = make_items()
    prof = Irregular(th, 0.0, 0.0)
    g = np.arange(-56, 56.01, 0.5)
    X, Y = np.meshgrid(g, g)
    Z = prof(X, Y)
    Xt = torch.tensor(X.ravel()); Yt = torch.tensor(Y.ravel()); Zt = torch.tensor(Z.ravel())
    pk = Z.max()
    Rg = np.hypot(X, Y); eps = 0.01 * float(Z[Rg < 8].mean())
    wz = 1.0 / (Zt + eps); norm = float((wz * Zt ** 2).sum())      # same flux-weighted objective as the Sersic fits
    comps = []

    def model(par):
        la, mx, my, l11, l21, l22 = [par[:, i] for i in range(6)]
        a = torch.exp(la); s11 = 0.25 + torch.nn.functional.softplus(l11); s22 = 0.25 + torch.nn.functional.softplus(l22)
        dx = Xt[:, None] - mx; dy = Yt[:, None] - my
        z1 = dx / s11; z2 = (dy - l21 * z1) / s22
        return (a / (2 * np.pi * s11 * s22) * torch.exp(-0.5 * (z1 ** 2 + z2 ** 2))).sum(1)

    def optimise(par, n_it):
        p = par.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([p], lr=1.0, max_iter=n_it, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)
        losses = []

        def closure():
            opt.zero_grad(); l = (wz * (model(p) - Zt) ** 2).sum() / norm; l.backward(); losses.append(float(l)); return l
        opt.step(closure)
        return p.detach(), losses

    def inv_sp(x): return math.log(math.expm1(x))
    snaps = {}
    par = torch.zeros((0, 6), dtype=torch.float64)
    for k in range(1, max(KS) + 1):
        resid = (Zt - (model(par) if len(par) else 0 * Zt)).reshape(X.shape).numpy()
        from scipy.ndimage import gaussian_filter
        sm = gaussian_filter(resid, 1.5); i = np.unravel_index(np.argmax(sm), sm.shape)
        amp = max(sm[i] * 2 * np.pi * 1.5, 1e-4)
        new = torch.tensor([[math.log(amp), X[i], Y[i], inv_sp(1.2), 0.0, inv_sp(1.2)]], dtype=torch.float64)
        par = torch.cat([par, new], 0)
        par, losses = optimise(par, 40)
        if k in KS:
            fin, losses = optimise(par, 400)
            m = model(fin).numpy()
            a = np.exp(fin[:, 0].numpy()); s11 = 0.25 + torch.nn.functional.softplus(fin[:, 3]).numpy()
            s22 = 0.25 + torch.nn.functional.softplus(fin[:, 5]).numpy(); l21 = fin[:, 4].numpy()
            S = np.zeros((k, 2, 2)); S[:, 0, 0] = s11 ** 2; S[:, 0, 1] = S[:, 1, 0] = s11 * l21; S[:, 1, 1] = l21 ** 2 + s22 ** 2
            snaps[k] = dict(flux=a.tolist(), mean=fin[:, 1:3].numpy().tolist(), cov=S.tolist(),
                            rms_over_peak_profile=float(np.sqrt(np.mean((m - Z.ravel()) ** 2)) / pk),
                            max_over_peak_profile=float(np.abs(m - Z.ravel()).max() / pk),
                            lbfgs_iterations_final=len(losses), loss_start_end=[losses[0], losses[-1]],
                            loss_still_falling=bool(len(losses) > 5 and (losses[-6] - losses[-1]) > 1e-3 * losses[-1]),
                            total_flux_fit=float(a.sum()))
            print("irregular K", k, {kk: v for kk, v in snaps[k].items() if kk in ("rms_over_peak_profile", "max_over_peak_profile", "loss_still_falling", "lbfgs_iterations_final")}, flush=True)
            par = fin
    return snaps


def stage_profile_fits():
    t0 = time.time()
    items, th, offs = make_items()
    out = {}
    for it in items:
        if it["kind"] == "sersic":
            rmax = min(20 * it["re"], (it["S"] // 2 + LSUP) * 1.0)
            out[it["id"]] = {str(K): sersic_gauss_fit(it["n"], it["re"], 0.5, K, rmax) for K in KS}
            print(it["id"], {K: round(v["rms_over_peak_profile"], 5) for K, v in out[it["id"]].items()}, flush=True)
    out["irregular"] = {str(k): v for k, v in irregular_gauss_fit(max(KS)).items()}
    out["_seconds"] = time.time() - t0
    jsave("profile_fits.json", out)


def gauss_profile_params(it, th, off, K, pf):
    """(flux (K,), mean (K,2), cov (K,2,2)) absolute positions for the item at offset off."""
    if it["kind"] == "sersic":
        f = pf[it["id"]][str(K)]
        sg = np.array(f["sigma"]); a = np.array(f["flux"])
        c, s = np.cos(th), np.sin(th); Rm = np.array([[c, -s], [s, c]])
        S = np.array([Rm @ np.diag([x ** 2, (0.5 * x) ** 2]) @ Rm.T for x in sg])
        return a, np.tile(np.array(off, float), (len(a), 1)), S
    f = pf["irregular"][str(K)]
    return np.array(f["flux"]), np.array(f["mean"]) + np.array(off, float), np.array(f["cov"])


# ----------------------------------------------------------------------------------------------
# route 1: closed-form double sum
# ----------------------------------------------------------------------------------------------
def closed_form(a, m, S, kc, ks, kw, Spix, delta_off=None, maxel=3_000_000):
    """image(pixel) = sum_k sum_j a_k w_j N(x - m_k - mu_j ; S_k + s_j^2 I). a=None -> point source at delta_off."""
    h = Spix // 2
    g = np.arange(-h, h + 1.0)
    X, Y = np.meshgrid(g, g); Xr = X.ravel(); Yr = Y.ravel()
    out = np.zeros(Xr.size)
    s2 = ks ** 2
    if a is None:
        a = np.array([1.0]); m = np.array([delta_off], float); S = np.zeros((1, 2, 2))
    J = len(kw); ch = max(1000, maxel // J)
    for k in range(len(a)):
        sxx = S[k, 0, 0] + s2; syy = S[k, 1, 1] + s2; sxy = S[k, 0, 1] * np.ones(J)
        det = sxx * syy - sxy ** 2
        pref = a[k] * kw / (2 * np.pi * np.sqrt(det))
        cx = m[k, 0] + kc[:, 0]; cy = m[k, 1] + kc[:, 1]
        for i in range(0, Xr.size, ch):
            dx = Xr[None, i:i + ch] - cx[:, None]; dy = Yr[None, i:i + ch] - cy[:, None]
            q = (syy[:, None] * dx * dx - 2 * sxy[:, None] * dx * dy + sxx[:, None] * dy * dy) / det[:, None]
            np.exp(-0.5 * q, out=q)
            out[i:i + ch] += pref @ q
    return out.reshape(Spix, Spix)


def approx_T_table(case, L, m, n, Lsup=LSUP):
    """Gaussian-approximated pixel-integrated kernel, point-sampled on the n grid."""
    kc, ks, kw = load_kgauss(case, L, m)
    c = np.arange(-Lsup * n, Lsup * n + 1) / n
    X, Y = np.meshgrid(c, c)
    return eval_kernel_gauss(np.concatenate([kc, ks[:, None]], 1), kw, X, Y)


def stage_route1():
    items, th, offs = make_items()
    pf = jload("profile_fits.json")
    res = jload("route1_results.json", {})
    for case in CASES:
        tabs, _ = load_tabs(case)
        L, m = kernel_choice(case)
        kc, ks, kw = load_kgauss(case, L, m)
        Ta = approx_T_table(case, L, m, NUM_N)
        T8 = crop(tabs[f"Tp{NUM_N}"], NUM_N, LSUP)
        for it in items:
            for p in range(3):
                key = f"{case}|{it['id']}|{p}"
                if key in res: continue
                t0 = time.time()
                ref = load_ref(case, it["id"], p)
                off = offs[p]; S = it["S"]
                r = dict(kernel_config=[L, m], n_gauss_kernel=len(kw))
                if it["kind"] == "point":
                    img = closed_form(None, None, None, kc, ks, kw, S, delta_off=off)
                    r["both"] = {"point": metrics(img, ref)}
                    r["kernel_only"] = r["both"]
                    r["profile_only"] = None
                else:
                    prof = make_profile(it, th, off)
                    # numerical floor and kernel-part: exact profile, 8 sub-samples, exact vs Gaussian kernel
                    img0 = render_T(prof, T8, NUM_N, S, LSUP, off, refine=True)
                    r["numerics_floor_exactprofile_exactkernel_n8"] = metrics(img0, ref)
                    imgk = render_T(prof, Ta, NUM_N, S, LSUP, off, refine=True)
                    # kernel part: difference between the same numerical scheme with the Gaussian-approximated and the exact kernel
                    r["kernel_only"] = {"all": metrics(imgk, img0)}
                    r["kernel_only"]["all"]["vs_reference"] = metrics(imgk, ref)
                    r["profile_only"] = {}; r["both"] = {}
                    for K in KS:
                        a, mm, SS = gauss_profile_params(it, th, off, K, pf)
                        gp = GaussMix(a, mm, SS)
                        imgp = render_T(gp, T8, NUM_N, S, LSUP, off, refine=True)
                        r["profile_only"][str(K)] = metrics(imgp, img0)       # same scheme, approximate vs exact profile
                        r["profile_only"][str(K)]["vs_reference"] = metrics(imgp, ref)
                        imgb = closed_form(a, mm, SS, kc, ks, kw, S)
                        r["both"][str(K)] = metrics(imgb, ref)
                r["seconds"] = time.time() - t0
                res[key] = r
                jsave("route1_results.json", res)
                b = r["both"]
                print(key, {k: round(v["max"], 5) for k, v in b.items()}, "t=%.1f" % r["seconds"], flush=True)


# ----------------------------------------------------------------------------------------------
# route 2: sweeps
# ----------------------------------------------------------------------------------------------
def n_adapt(it, nb):
    mult = 4 if it["re"] <= 1.0 else (2 if it["re"] <= 4.0 else 1)
    return min(16, nb * mult)


R2_NB = [1, 2, 3, 4, 6]
R2_TRUNC = [32, 16, 8]


def r2_configs():
    cf = []
    for variant in ("bin", "boxK"):
        for ad in (False, True):
            for nb in R2_NB:
                cf.append((variant, ad, nb, 32))
        for ad in (False, True):
            for L in (16, 8):
                cf.append((variant, ad, 4, L))
    return cf


class Tabs:
    def __init__(s, case):
        s.t, _ = load_tabs(case); s.c = {}

    def get(s, variant, n, L):
        k = (variant, n, L)
        if k not in s.c:
            s.c[k] = crop(s.t[("Kp" if variant == "bin" else "Tp") + str(n)], n, L)
        return s.c[k]


def r2_render(variant, tb, it, prof, off, S, n, L):
    if variant == "bin":
        return render_bin(prof, tb.get("bin", n, L), n, S, L, off, delta=(it["kind"] == "point"))
    return render_T(prof, tb.get("boxK", n, L), n, S, L, off, delta=(it["kind"] == "point"))


def stage_route2():
    items, th, offs = make_items()
    res = jload("route2_results.json", {})
    for case in CASES:
        tb = Tabs(case)
        for it in items:
            for p in range(3):
                key = f"{case}|{it['id']}|{p}"
                if key in res: continue
                t0 = time.time()
                ref = load_ref(case, it["id"], p); off = offs[p]
                prof = make_profile(it, th, off); r = {}
                for (variant, ad, nb, L) in r2_configs():
                    n = n_adapt(it, nb) if ad else nb
                    img = r2_render(variant, tb, it, prof, off, it["S"], n, L)
                    r[f"{variant}|adapt={int(ad)}|nb={nb}|L={L}"] = dict(n=n, **metrics(img, ref))
                res[key] = r
                jsave("route2_results.json", res)
                print(key, "t=%.1f" % (time.time() - t0), flush=True)


# ----------------------------------------------------------------------------------------------
# timing (median of repeats after warm-up, one thread)
# ----------------------------------------------------------------------------------------------
def timeit(fn, rep=5):
    fn()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def stage_timing():
    items, th, offs = make_items()
    pf = jload("profile_fits.json")
    byid = {it["id"]: it for it in items}
    res = {}
    # items to time: (id, stamp). small = re 2 on a 65x65 stamp; large = re 32 on 257x257; also 129 stamp for re 2.
    tl = [("point", 65), ("sersic1_re2", 65), ("sersic1_re2", 129), ("sersic1_re32", 257), ("sersic4_re2", 65), ("sersic4_re32", 257), ("irregular", 129)]
    for case in CASES:
        tb = Tabs(case)
        L, m = kernel_choice(case)
        kc, ks, kw = load_kgauss(case, L, m)
        for iid, S in tl:
            it = dict(byid[iid]); it["S"] = S
            off = offs[0]
            prof = make_profile(it, th, off)
            tag = f"{case}|{iid}|S{S}"
            res[tag] = {}
            # route 1
            if it["kind"] == "point":
                res[tag]["route1|point"] = timeit(lambda: closed_form(None, None, None, kc, ks, kw, S, delta_off=off), 5)
            else:
                for K in KS:
                    a, mm, SS = gauss_profile_params(it, th, off, K, pf)
                    res[tag][f"route1|K={K}"] = timeit(lambda: closed_form(a, mm, SS, kc, ks, kw, S), 3 if S >= 257 else 5)
            res[tag]["route1_n_kernel_gaussians"] = len(kw)
            # route 2
            for (variant, ad, nb, Lt) in r2_configs():
                n = n_adapt(it, nb) if ad else nb
                big = S >= 257
                if big and not (Lt == 32 or nb == 4): continue
                res[tag][f"route2|{variant}|adapt={int(ad)}|nb={nb}|L={Lt}"] = timeit(
                    lambda: r2_render(variant, tb, it, prof, off, S, n, Lt), 3 if big else 5)
            print(tag, "done", flush=True)
    jsave("timing.json", res, CACHE)


# ----------------------------------------------------------------------------------------------
# assemble, estimate, figure
# ----------------------------------------------------------------------------------------------
def agg(rows, key):
    """worst-of-positions max error; mean-of-positions rms; mean flux error; (rows = list of metric dicts)."""
    return dict(max_worst=max(r["max"] for r in rows), max_median=float(np.median([r["max"] for r in rows])), rms_mean=float(np.mean([r["rms"] for r in rows])),
                flux_abs_worst=max(abs(r["flux"]) for r in rows))


def stage_final():
    items, th, offs = make_items()
    r1 = jload("route1_results.json"); r2 = jload("route2_results.json"); tm = jload("timing.json")
    kinfo = jload("kernel_info.json"); kfit = jload("kernel_fits.json"); conv = jload("ref_convergence.json")
    pf = jload("profile_fits.json")
    out = dict(description="E2 numerical results. Errors are relative to the peak pixel of the reference image; "
                           "'max_worst' = worst over the 3 sub-pixel positions of the max abs error; 'rms_mean' = mean over positions.",
               stand_in_kernels="STAND-INS (analytic pupil), to be replaced by kernels learned from real data",
               kernel_info=kinfo, kernel_fits=kfit, reference_convergence_8_vs_16=conv, profile_fits=pf,
               config=dict(seed=SEED, orientation_rad=float(th), subpixel_offsets=offs.tolist(), offsets_quantised_to="1/8 px",
                           kernel_support=LSUP, sub_samples_reference=REF_N, numerics_sub_samples=NUM_N),
               timing_seconds=tm)
    summ = {"route1": {}, "route2": {}}
    snap = {}
    for case in CASES:
        tabs, _ = load_tabs(case); P = tabs["P16"]; pk = P.max()
        snap[case] = {f"shift_{k}_over_16_px": float(np.abs(P[k:, :] - P[:-k, :]).max() / pk) for k in (1, 2)}
    out["point_source_snap_error_scale"] = dict(
        note="max |T(x)-T(x+shift)|/peak of the exact pixel-integrated kernel: the error a point source suffers when snapped to a "
             "grid point shift px away. Route 2 snaps to the nearest sample (<= 1/(2n) px). The test offsets are multiples of 1/8 px, "
             "so grids with n multiple of 8 hit them exactly: those point-source numbers are not representative of generic positions.",
        per_case=snap)
    for case in CASES:
        for it in items:
            iid = it["id"]
            rows1 = [r1[f"{case}|{iid}|{p}"] for p in range(3)]
            s1 = {}
            if iid == "point":
                s1["kernel_only"] = agg([r["both"]["point"] for r in rows1], "")
            else:
                s1["numerics_floor_n8"] = agg([r["numerics_floor_exactprofile_exactkernel_n8"] for r in rows1], "")
                s1["kernel_only"] = agg([r["kernel_only"]["all"] for r in rows1], "")
                for K in KS:
                    s1[f"profile_only_K{K}"] = agg([r["profile_only"][str(K)] for r in rows1], "")
                    s1[f"both_K{K}"] = agg([r["both"][str(K)] for r in rows1], "")
            summ["route1"][f"{case}|{iid}"] = s1
            rows2 = [r2[f"{case}|{iid}|{p}"] for p in range(3)]
            s2 = {}
            for k in rows2[0]:
                a = agg([r[k] for r in rows2], "")
                a["n"] = rows2[0][k]["n"]
                s2[k] = a
            summ["route2"][f"{case}|{iid}"] = s2
    out["summary"] = summ
    out["route1_kernel_config_used_L_m"] = {c: list(kernel_choice(c)) for c in CASES}
    out["route1_per_position"] = r1
    out["route2_per_position"] = r2
    # --- 2048 x 2048 estimate
    est = {}
    for case in CASES:
        e = {}
        ts = tm[f"{case}|sersic1_re2|S65"]; tl_ = tm[f"{case}|sersic1_re32|S257"]
        for K in KS:
            e[f"route1_K{K}"] = dict(small_s=ts[f"route1|K={K}"], large_s=tl_[f"route1|K={K}"],
                                     total_s=1e4 * ts[f"route1|K={K}"] + 100 * tl_[f"route1|K={K}"])
        for k in ts:
            if k.startswith("route2|") and "L=32" in k and k in tl_:
                e[k] = dict(small_s=ts[k], large_s=tl_[k], total_s=1e4 * ts[k] + 100 * tl_[k])
        est[case] = e
    out["estimate_2048x2048_10000_small_100_large"] = dict(
        assumptions=["items are Sersic n=1 (half-light radius 2 px and 32 px, axis ratio 0.5)",
                     "small items rendered on a 65x65 stamp, large on 257x257; kernel +-32 px (route 2 includes a 32 px margin of fine grid)",
                     "per-item times measured on one thread of a shared machine, additive; no overlap/batching, no culling, no GPU",
                     "route 1 time excludes fitting the Gaussians (they would be the model's parameters); route 2 excludes kernel tabulation",
                     "images of overlapping items are summed at negligible extra cost (stamp add not timed)"],
        per_case=est)
    jsave("e2_results.json", out, RES)
    make_figure(out)
    print("wrote results/e2_results.json and figs/e2_errors.png")


def make_figure(out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tm = out["timing_seconds"]; summ = out["summary"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=False, sharey=True)
    cols = {"r1": "#0072B2", "r2": "#D55E00", "r2b": "#009E73"}
    for ci, case in enumerate(CASES):
        for ri, (iid, S, lab) in enumerate([("sersic1_re2", 65, "small item: Sersic n=1, half-light radius 2 px (65x65 stamp)"),
                                             ("sersic1_re32", 257, "large item: Sersic n=1, half-light radius 32 px (257x257 stamp)")]):
            ax = axs[ri, ci]
            t = tm[f"{case}|{iid}|S{S}"]
            s1 = summ["route1"][f"{case}|{iid}"]; s2 = summ["route2"][f"{case}|{iid}"]
            xs = [1e3 * t[f"route1|K={K}"] for K in KS]; ys = [s1[f"both_K{K}"]["max_worst"] for K in KS]
            ax.plot(xs, ys, "o-", color=cols["r1"], lw=2, ms=8, label="Route 1 closed form, K = 2,4,8,16")
            for x, y, K in zip(xs, ys, KS): ax.annotate(f"K={K}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9, color=cols["r1"])
            for variant, ls, mk, lab2 in (("bin", "s-", "s", "Route 2 sampling + binning"), ("boxK", "^--", "^", "Route 2 pixel box inside kernel")):
                for ad, fill in ((False, True), (True, False)):
                    ks = [k for k in s2 if k.startswith(f"{variant}|adapt={int(ad)}") and k.endswith("L=32") and f"route2|{k}" in t]
                    ks.sort(key=lambda k: int(k.split("nb=")[1].split("|")[0]))
                    xs = [1e3 * t[f"route2|{k}"] for k in ks]; ys = [s2[k]["max_worst"] for k in ks]
                    c = cols["r2"] if variant == "bin" else cols["r2b"]
                    ax.plot(xs, ys, ls, color=c, lw=1.6, ms=7, mfc=(c if fill else "white"),
                            label=f"{lab2}, {'adapted to size' if ad else 'fixed'} n=1,2,3,4,6")
            ax.axhline(1e-3, color="k", ls=":", lw=1.5); ax.text(0.02, 1.15e-3, "target 1e-3 of peak", transform=ax.get_yaxis_transform(), fontsize=10)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{case} kernel (FWHM {out['kernel_info'][case]['fwhm_axis_mean_px']:.2f} px): {lab}", fontsize=9)
            ax.set_xlabel("time per item, ms (one thread)"); 
            if ci == 0: ax.set_ylabel("max |error| / peak pixel (worst of 3 positions)")
    axs[0, 0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("E2: render accuracy against cost (stand-in kernels, kernel support +-32 px)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "e2_errors.png"), dpi=130)


STAGES = dict(kernels=stage_kernels, refs=stage_refs, fits=stage_fits, pfits=stage_profile_fits, route1=stage_route1, route2=stage_route2,
              timing=stage_timing, final=stage_final)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all"); ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    if a.fresh: shutil.rmtree(CACHE, ignore_errors=True); os.makedirs(CACHE, exist_ok=True)
    for name in (["kernels", "refs", "fits", "pfits", "route1", "route2", "timing", "final"] if a.stage == "all" else [a.stage]):
        t0 = time.time(); print("== stage", name, flush=True); STAGES[name](); print("== stage", name, "took %.0f s" % (time.time() - t0), flush=True)
