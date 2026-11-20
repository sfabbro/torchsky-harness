#!/usr/bin/env python3
"""E2 repeated with LEARNED kernels (fieldop, SMACS 200-minute run) instead of the hexagon stand-ins.

Needs results/learned_kernels.json and the state file (run learned_kernels.py first). Re-uses the functions of render_test_e2_copy.py (a copy of
e2/render_test.py: Sersic, make_items, make_profile, sample_grid, render_T, crop, metrics, stencil, gauss_design, spike_mask).

Conventions (all lengths in detector pixels of the kernel's own channel, short-wavelength 0.031 arcsec, long-wavelength 0.063 arcsec):
  * the learned kernel k(v) is a density per pixel^2, includes the pixel response implicitly (fieldop reads the continuous sky at pixel centres, no
    box), so it is used as the "pixel in kernel" kernel: image(p) = sum_x f(x) k(x - p) h^2 at pixel centres p. fieldop's operator is a CORRELATION
    (c = integral k(u) s(Omega + u) du), so the table is flipped, k(-v), before the FFT convolution of render_T.
  * kernels at any sampling n are evaluated directly from the network (same code, Z, taper as learned_kernels.py), point-sampled.
  * reference image = Richardson extrapolation (4 R16 - R8) / 3 of the point-sampled scheme at 16 and 8 sub-samples, kernel cut at +-32 px, source
    sampled with 6x6 sub-samples in the cells near its centre (as e2). Errors are max |error| / peak of the reference, worst of 3 sub-pixel positions.
Stages (each resumable; --budget seconds): fits, refs, trunc, local, final.  Usage: python3 render_test_learned.py --stage fits --kernel F090W
"""
import os
import sys
import json
import time
import argparse
import math
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import torch
torch.set_num_threads(1)
from scipy.optimize import nnls
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_test_e2_copy as E2
import learned_kernels as LK

CACHE = os.path.join(HERE, "results", "cache"); FIGS = os.path.join(HERE, "figs")
os.makedirs(CACHE, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
FILTERS = LK.FILTERS
LSUP = 32
KCONFIGS = [(3, 1, 1.0), (4, 1, 1.0), (5, 2, 1.0), (6, 2, 1.0), (7, 2, 1.0), (8, 2, 1.0), (7, 3, 1.0), (7, 3, 0.5),
            (8, 3, 1.0)]       # last = the extra, larger stencil (392 candidates). (7,4,0.5) of e2 (567 candidates, diagnostic) is NOT repeated.
EXTRA_FITS = [(10, 2, 1.0, 0.08, True)]   # NOT in the e2 protocol: pyramid starting at sigma = 0.08 px (10 levels x 25), fitted on the 16 samples/px table
NLOC = [2, 4, 6, 8]
POINT_OFFSETS = [(0.137, -0.291), (-0.382, 0.213), (0.4707, 0.0589)]   # NOT multiples of 1/8 px (e2 used 1/8 px steps, which fall on the n=4, 8, 16 grids)
REF_METHOD = {"sersic1_re2": "split_in16-32_out8-16", "sersic1_re8": "split_in16-32_out8-16", "sersic1_re32": "rich8-16"}
SOURCES = [("sersic1_re2", 2.0, 129), ("sersic1_re8", 8.0, 129), ("sersic1_re32", 32.0, 257)]


def jload(name, default=None):
    p = os.path.join(CACHE, name)
    return json.load(open(p)) if os.path.exists(p) else default


def jsave(name, obj):
    json.dump(obj, open(os.path.join(CACHE, name), "w"), indent=1, default=float)


class Kern:
    """Model pieces for one filter (loaded once)."""
    _shared = None

    def __init__(self, name):
        if Kern._shared is None:
            s, net, codes, tau = LK.load_model()
            Kern._shared = (s, net, codes, tau, json.load(open(os.path.join(HERE, "results", "learned_kernels.json"))))
        s, net, codes, tau, meta = Kern._shared
        self.name = name; self.f = FILTERS.index(name)
        self.meta = meta["kernels"][name]
        self.ch = self.meta["channel"]; self.pix = LK.PIX[self.ch]
        det = 4 if self.ch == 1 else 2
        self.net = net; self.h = LK.mean_code(s, codes, tau, self.f, det)
        self.dil = meta["dilation"]; self.logZ = self.meta["logZ_levels"]
        self.tabs = {}

    def raw(self, n, L):
        k = (n, L)
        if k not in self.tabs:
            self.tabs[k] = LK.table(self.net, self.h, self.logZ, self.pix, self.dil, n=n, L=L)
        return self.tabs[k]

    @property
    def mass32(self):
        """integral of the kernel over the +-32 px square, converged in the sampling: Richardson (4 S32 - S16) / 3 of the sums at 32 and 16 samples/px (cached)."""
        mj = jload("kernel_mass.json", {})
        if self.name not in mj:
            s16 = self.raw(16, LSUP).sum() / 256.0
            s32 = LK.table(self.net, self.h, self.logZ, self.pix, self.dil, n=32, L=LSUP).sum() / 1024.0
            mj[self.name] = dict(S16=float(s16), S32=float(s32), mass=float((4 * s32 - s16) / 3.0))
            jsave("kernel_mass.json", mj)
        return mj[self.name]["mass"]

    def table(self, n, L, flip=False, norm=False):
        """learned kernel, density per pixel^2, n samples per pixel, +-L px, point-sampled. flip -> k(-v) (for rendering).
        norm -> scaled so that the +-32 px table sums (x 1/n^2) to the converged integral mass32 (as a practical implementation would normalise
        its table; point-sampling the sharp core at 2 to 6 samples per pixel otherwise changes the mass by up to 18 percent)."""
        T = self.raw(n, L)
        if norm:
            T = T * (self.mass32 / (self.raw(n, LSUP).sum() / n ** 2))
        return T[::-1, ::-1].copy() if flip else T

    def k_at(self, vx, vy):
        """exact kernel (density per pixel^2) at detector-frame offsets (px)."""
        v = torch.tensor(np.stack([np.ravel(vx), np.ravel(vy)], -1) * self.pix, dtype=torch.float32)
        lg = LK.log_density(self.net, self.h, v, self.dil).double()
        r = v.double().norm(dim=-1)
        T = LK.smooth_down(r, 0.85 * LK.EXTENT, LK.EXTENT)
        return (torch.exp(lg - self.logZ) * T * self.pix ** 2).numpy().reshape(np.shape(vx))


# ----------------------------------------------------------------------------------------------
# (a) kernel fits: non-negative and signed Gaussians on the pyramid; streaming QR so the big design matrix is never stored
# ----------------------------------------------------------------------------------------------
def col_norms(P, g):
    """exact column norms of the design matrix on the square grid g x g (separable)."""
    out = np.empty(len(P))
    for j, (cx, cy, s) in enumerate(P):
        a = np.exp(-(g - cx) ** 2 / s ** 2); b = np.exp(-(g - cy) ** 2 / s ** 2)
        out[j] = a.sum() ** 0.5 * b.sum() ** 0.5 / (2 * np.pi * s ** 2)
    return out


def stream_qr(P, sc, X, Y, y, chunk=16384):
    n = len(P)
    Raug = np.zeros((0, n + 1))
    Xr, Yr, yr = X.ravel(), Y.ravel(), y.ravel()
    for i in range(0, Xr.size, chunk):
        A = E2.gauss_design(P, Xr[i:i + chunk], Yr[i:i + chunk]) / sc
        aug = np.concatenate([Raug, np.concatenate([A, yr[i:i + chunk, None]], 1)], 0)
        Raug = np.linalg.qr(aug, mode="r")
        del A, aug
    return Raug[:n, :n], Raug[:n, n], abs(Raug[n, n])


def eval_w(P, w, X, Y, chunk=60000):
    """sum_j w_j N(x; c_j, s_j^2 I); weights of either sign (E2.eval_kernel_gauss drops negative weights, so it cannot be used for signed fits)."""
    out = np.empty(X.size); Xr, Yr = X.ravel(), Y.ravel()
    nz = w != 0
    for i in range(0, X.size, chunk):
        out[i:i + chunk] = E2.gauss_design(P[nz], Xr[i:i + chunk], Yr[i:i + chunk]) @ w[nz]
    return out.reshape(X.shape)


def stencil2(L, m, sp, sig0):
    P = []
    for l in range(L):
        s = sig0 * 2 ** l
        g = np.arange(-m, m + 1) * s * sp
        P += [(cx, cy, s) for cx in g for cy in g]
    return np.array(P)


def stage_fits(names, budget):
    t_end = time.time() + budget
    for name in names:
        fn = f"fits_{name}.json"
        out = jload(fn, {})
        K = Kern(name)
        T16 = K.table(16, LSUP); T4 = T16[::4, ::4]
        pk4, pk16 = T4.max(), T16.max()
        g4 = (np.arange(T4.shape[0]) - T4.shape[0] // 2) / 4.0
        X4, Y4 = np.meshgrid(g4, g4)
        g16 = (np.arange(T16.shape[0]) - T16.shape[0] // 2) / 16.0
        X16, Y16 = np.meshgrid(g16, g16)
        R4 = np.hypot(X4, Y4)
        for (L, m, sp, sig0, fit16) in [c + (0.35, False) for c in KCONFIGS] + EXTRA_FITS:
            key = f"{L}_{m}_{sp}" + ("" if sig0 == 0.35 else f"_s{sig0}")
            if key in out: continue
            if time.time() > t_end: print("budget reached"); return
            t0 = time.time()
            P = stencil2(L, m, sp, sig0)
            if fit16:       # extra configuration: fitted directly on the 16 samples/px table (the smallest Gaussians are narrower than the 4/px grid can resolve)
                sc = col_norms(P, g16)
                Rm, qty, resn = stream_qr(P, sc, X16, Y16, T16, chunk=30000)
            else:
                sc = col_norms(P, g4)
                Rm, qty, resn = stream_qr(P, sc, X4, Y4, T4)
            w, _ = nnls(Rm, qty, maxiter=20000)
            wu = np.linalg.lstsq(Rm, qty, rcond=1e-13)[0]
            w = w / sc; wu = wu / sc
            rec = dict(L=L, m=m, spacing_in_widths=sp, smallest_sigma_px=sig0, fitted_on_16px_grid=fit16, n_stencil=len(P), n_nonzero=int((w > 0).sum()))
            for tag, ww in (("nonneg", w), ("signed", wu)):
                f4 = eval_w(P, ww, X4, Y4)
                e4 = (f4 - T4) / pk4
                f16 = eval_w(P, ww, X16, Y16)
                e16 = (f16 - T16) / pk16
                rec[tag] = dict(max_err_4px_grid_over_grid_peak=float(np.abs(e4).max()), rms_err_4px_grid=float(np.sqrt(np.mean(e4 ** 2))),
                                max_err_16px_grid_over_true_peak=float(np.abs(e16).max()), rms_err_16px_grid=float(np.sqrt(np.mean(e16 ** 2))),
                                max_err_at_peak_pixel_16=float(abs(e16[np.unravel_index(np.argmax(T16), T16.shape)])),
                                flux_captured=float(ww.sum() / (T4.sum() / 16.0)),
                                core_max_err_r_lt_2_4px=float(np.abs(e4[R4 < 2]).max()), mid_max_err_2_to_8_4px=float(np.abs(e4[(R4 >= 2) & (R4 < 8)]).max()),
                                outer_max_err_gt_8_4px=float(np.abs(e4[R4 >= 8]).max()), min_fit_value_over_peak=float(f16.min() / pk16))
                del f16, e16
            rec["signed"]["negative_weight_share"] = float(np.abs(wu[wu < 0]).sum() / np.abs(wu).sum())
            rec["tables_peak_ratio_4px_grid_over_true"] = float(pk4 / pk16)
            rec["seconds"] = time.time() - t0
            out[key] = rec
            jsave(fn, out)
            print(name, key, "nonneg max4 %.2e max16 %.2e | signed max4 %.2e max16 %.2e | flux %.3f | %.0fs" % (
                rec["nonneg"]["max_err_4px_grid_over_grid_peak"], rec["nonneg"]["max_err_16px_grid_over_true_peak"],
                rec["signed"]["max_err_4px_grid_over_grid_peak"], rec["signed"]["max_err_16px_grid_over_true_peak"], rec["nonneg"]["flux_captured"], rec["seconds"]), flush=True)


# ----------------------------------------------------------------------------------------------
# references (Richardson of 8 and 16 sub-samples, kernel +-32 px)
# ----------------------------------------------------------------------------------------------
_ITEMS = None


def items():
    global _ITEMS
    if _ITEMS is None:
        _ITEMS = E2.make_items()
    return _ITEMS


def mk_item(iid):
    its, th, offs = items()
    it = [x for x in its if x["id"] == iid][0]
    return it, th, offs


def render_corr(K, prof, n, S, L, off, refine=False):
    """image(p) = sum_x f(x) k(x - p) h^2 at pixel centres, kernel cut at +-L px, n sub-samples per pixel."""
    Tt = E2.crop(K.table(n, LSUP, flip=True, norm=True), n, L)
    return E2.render_T(prof, Tt, n, S, L, off, refine=refine)


def win_in(n, L):
    """smooth radial window w(r) = 1 for r <= 8 px, 0 for r >= 12 px, on the n/px table grid (+-L px)."""
    g = np.arange(-L * n, L * n + 1) / n
    X, Y = np.meshgrid(g, g)
    return LK.smooth_down(torch.tensor(np.hypot(X, Y)), 8.0, 12.0).numpy()


def render_split(K, prof, n, S, off, part, L, refine=True):
    """image of the kernel part K*w ('in') or K*(1-w) ('out'), cut at +-L, n sub-samples per pixel."""
    T = K.table(n, L, flip=True, norm=True)
    w = win_in(n, L)
    Tt = T * w if part == "in" else T * (1.0 - w)
    return E2.render_T(prof, Tt, n, S, L, off, refine=refine)


def stage_refs(names, budget):
    """Reference. re 2 and re 8: kernel split smoothly at r = 8-12 px; inner part Richardson(16, 32 sub-samples), outer part Richardson(8, 16).
    re 32: Richardson(8, 16) of the whole kernel (n = 32 does not fit in memory on a 257 px stamp). Kernel cut at +-32 px."""
    t_end = time.time() + budget
    conv = jload("lref_convergence.json", {})
    for name in names:
        K = Kern(name)
        for iid, re, S in SOURCES:
            it, th, offs = mk_item(iid)
            for p in range(3):
                key = f"{name}|{iid}|{p}"
                fn = os.path.join(CACHE, f"lref_{name}_{iid}_{p}.npy")
                if os.path.exists(fn) and key in conv and conv[key].get("ref_method") == REF_METHOD[iid]: continue
                if time.time() > t_end: print("budget reached", flush=True); return
                t0 = time.time()
                prof = E2.make_profile(it, th, offs[p])
                if iid == "sersic1_re32":
                    r8 = render_corr(K, prof, 8, S, LSUP, offs[p], refine=True)
                    r16 = render_corr(K, prof, 16, S, LSUP, offs[p], refine=True)
                    ref = (4 * r16 - r8) / 3.0; pk = ref.max()
                    info = dict(ref_method=REF_METHOD[iid], raw_16_vs_8_max_over_peak=float(np.abs(r16 - r8).max() / pk), plain16_vs_ref_max_over_peak=float(np.abs(r16 - ref).max() / pk))
                else:
                    i16 = render_split(K, prof, 16, S, offs[p], "in", 12); i32 = render_split(K, prof, 32, S, offs[p], "in", 12)
                    o8 = render_split(K, prof, 8, S, offs[p], "out", LSUP); o16 = render_split(K, prof, 16, S, offs[p], "out", LSUP)
                    rin = (4 * i32 - i16) / 3.0; rout = (4 * o16 - o8) / 3.0
                    ref = rin + rout; pk = ref.max()
                    info = dict(ref_method=REF_METHOD[iid], inner_plain32_vs_ref_max_over_peak=float(np.abs(i32 + rout - ref).max() / pk),
                                inner_plain16_vs_ref_max_over_peak=float(np.abs(i16 + rout - ref).max() / pk),
                                outer_plain16_vs_ref_max_over_peak=float(np.abs(rin + o16 - ref).max() / pk),
                                inner_richardson_vs_plain32_over_peak_error_estimate=float(np.abs(rin - i32).max() / pk))
                np.save(fn, ref)
                info.update(peak=float(pk), flux_in_stamp=float(ref.sum()), seconds=time.time() - t0)
                conv[key] = info
                jsave("lref_convergence.json", conv)
                print(key, {k: (round(v, 6) if not isinstance(v, str) else v) for k, v in info.items()}, flush=True)


def stage_refcheck(names, budget):
    """Check of the Richardson reference: kernel cut at +-12 px (memory), re 2 and re 8, position 0: Richardson(8,16) vs Richardson(16,32) and plain 32."""
    res = jload("lref_check.json", {})
    for name in names:
        K = Kern(name)
        for iid, re, S in SOURCES[:2]:
            key = f"{name}|{iid}"
            if key in res: continue
            it, th, offs = mk_item(iid); off = offs[0]; prof = E2.make_profile(it, th, off)
            L = 12
            r = {n: render_corr(K, prof, n, S, L, off, refine=True) for n in (8, 16, 32)}
            R1 = (4 * r[16] - r[8]) / 3.0; R2 = (4 * r[32] - r[16]) / 3.0; pk = R2.max()
            res[key] = dict(kernel_cut_px=L, richardson_8_16_vs_16_32_max_over_peak=float(np.abs(R1 - R2).max() / pk), plain16_vs_R2=float(np.abs(r[16] - R2).max() / pk),
                            plain8_vs_R2=float(np.abs(r[8] - R2).max() / pk), plain32_vs_R2=float(np.abs(r[32] - R2).max() / pk))
            jsave("lref_check.json", res); print(key, res[key], flush=True)


# ----------------------------------------------------------------------------------------------
# (b) truncation
# ----------------------------------------------------------------------------------------------
def stage_trunc(names, budget):
    t_end = time.time() + budget
    res = jload("trunc_results.json", {})
    for name in names:
        K = Kern(name)
        E = int(math.ceil(LK.EXTENT / K.pix))
        T1 = K.table(1, E, flip=True)
        wing = T1.copy(); c = E
        wing[c - LSUP:c + LSUP + 1, c - LSUP:c + LSUP + 1] = 0.0           # keep only |x|,|y| > 32 px
        core_sum = K.mass32
        wing_sum = float(wing.sum())
        for iid, re, S in [s for s in SOURCES if s[0] in ("sersic1_re2", "sersic1_re32")]:
            it, th, offs = mk_item(iid)
            for p in range(3):
                key = f"{name}|{iid}|{p}"
                if key in res: continue
                if time.time() > t_end: print("budget reached", flush=True); return
                t0 = time.time()
                off = offs[p]
                prof = E2.make_profile(it, th, off)
                imgs = {L: render_corr(K, prof, 8, S, L, off, refine=True) for L in (8, 16, 32)}
                wimg = E2.render_T(prof, wing, 1, S, E, off, refine=True)
                full = imgs[32] + wimg
                pk = full.max()
                r = dict(kernel_integral_core32=float(core_sum), kernel_integral_wing_beyond_32_n1=wing_sum, wing_table_half_width_px=E,
                         image_peak=float(pk), image_flux_stamp=float(full.sum()), wing_flux_over_total=float(wimg.sum() / full.sum()))
                tot = core_sum + wing_sum
                for L in (8, 16, 32):
                    inside = float(K.table(8, LSUP, norm=True)[LSUP * 8 - L * 8:LSUP * 8 + L * 8 + 1, LSUP * 8 - L * 8:LSUP * 8 + L * 8 + 1].sum() / 64.0)
                    e = imgs[L] - full
                    e2 = imgs[L] * (tot / inside) - full
                    r[f"L{L}"] = dict(kernel_fraction_inside=inside / tot, max_err=float(np.abs(e).max() / pk), rms_err=float(np.sqrt(np.mean(e ** 2)) / pk),
                                      flux_lost=float(-e.sum() / full.sum()),
                                      rescaled_max_err=float(np.abs(e2).max() / pk), rescaled_rms_err=float(np.sqrt(np.mean(e2 ** 2)) / pk))
                # error that remains in the L=32 image if the kernel is only extended to +-E (SW: E = 32 x ... full 10 arcsec = %d px)
                r["seconds"] = time.time() - t0
                res[key] = r
                jsave("trunc_results.json", res)
                print(key, {L: (round(r[f"L{L}"]["max_err"], 5), round(r[f"L{L}"]["rescaled_max_err"], 5)) for L in (8, 16, 32)}, "wing flux share %.3f" % r["wing_flux_over_total"], "%.0fs" % r["seconds"], flush=True)


# ----------------------------------------------------------------------------------------------
# (c) local sampling
# ----------------------------------------------------------------------------------------------
def point_images(K, off, S, n_list):
    h = S // 2
    g = np.arange(-h, h + 1.0)
    X, Y = np.meshgrid(g, g)
    vx, vy = off[0] - X, off[1] - Y                       # k(x0 - p): kernel argument for the pixel at p
    ref = np.where((np.abs(vx) <= LSUP) & (np.abs(vy) <= LSUP), K.k_at(vx, vy), 0.0)    # exact network value at the exact offset
    out = {}
    for n in n_list:
        Tn = K.table(n, LSUP)
        ix = (vx + LSUP) * n; iy = (vy + LSUP) * n
        inside = (np.abs(vx) <= LSUP) & (np.abs(vy) <= LSUP)
        cub = map_coordinates(Tn, [iy, ix], order=3, mode="nearest", prefilter=True)
        near = Tn[np.clip(np.round(iy).astype(int), 0, Tn.shape[0] - 1), np.clip(np.round(ix).astype(int), 0, Tn.shape[1] - 1)]
        out[n] = dict(cubic=np.where(inside, cub, 0.0), snap=np.where(inside, near, 0.0))
    return ref, out


def stage_local(names, budget):
    t_end = time.time() + budget
    res = jload("local_results.json", {})
    for name in names:
        K = Kern(name)
        for iid, re, S in SOURCES:
            it, th, offs = mk_item(iid)
            for p in range(3):
                key = f"{name}|{iid}|{p}"
                if key in res: continue
                fn = os.path.join(CACHE, f"lref_{name}_{iid}_{p}.npy")
                if not os.path.exists(fn): print("missing reference", key); continue
                if time.time() > t_end: print("budget reached", flush=True); return
                t0 = time.time()
                ref = np.load(fn); prof = E2.make_profile(it, th, offs[p]); r = {}
                for n in NLOC:
                    img = render_corr(K, prof, n, S, LSUP, offs[p], refine=False)
                    r[f"n={n}"] = E2.metrics(img, ref)
                    img = render_corr(K, prof, n, S, LSUP, offs[p], refine=True)
                    r[f"n={n}|centre_refined"] = E2.metrics(img, ref)
                r["seconds"] = time.time() - t0
                res[key] = r; jsave("local_results.json", res)
                print(key, {k: round(v["max"], 5) for k, v in r.items() if k.startswith("n=") and "refined" not in k}, "%.0fs" % r["seconds"], flush=True)
        # point source
        for p in range(3):
            key = f"{name}|point|{p}"
            if key in res: continue
            ref, out = point_images(K, POINT_OFFSETS[p], 129, [2, 4, 6, 8, 16])
            r = {}
            for n, d in out.items():
                for kind, img in d.items():
                    r[f"n={n}|{kind}"] = E2.metrics(img, ref)
            res[key] = r; jsave("local_results.json", res)
            print(key, {k: round(v["max"], 4) for k, v in r.items() if "cubic" in k}, flush=True)


# ----------------------------------------------------------------------------------------------
def worst(rows, field="max"):
    return float(max(r[field] for r in rows))


def stage_final():
    out = dict(description="E2 with learned kernels. Errors relative to the peak of the reference image; 'worst' = worst of 3 sub-pixel positions, 'median' = median of 3.",
               kernel_properties=json.load(open(os.path.join(HERE, "results", "learned_kernels.json"))),
               stand_in_numbers_e2=dict(fwhm_sharp_px=1.15, fwhm_wide_px=2.11, kernel_fit_nonneg_max_sharp=3.2e-3, kernel_fit_nonneg_max_wide=1.2e-2,
                                        kernel_fit_signed_max_sharp=9.7e-4, kernel_fit_signed_max_wide=2.6e-3,
                                        truncation_re32_L16_sharp_wide=[2.7e-3, 5.3e-3], truncation_re32_L8_sharp_wide=[1.1e-2, 1.8e-2]),
               reference_convergence=jload("lref_convergence.json"), kernel_fits={}, truncation={}, local_sampling={})
    fits = {k: jload(f"fits_{k}.json") for k in FILTERS}
    out["kernel_fits"] = {k: v for k, v in fits.items() if v}
    tr = jload("trunc_results.json", {}); lo = jload("local_results.json", {})
    for k in FILTERS:
        out["truncation"][k] = {}
        for iid in ("sersic1_re2", "sersic1_re32"):
            rows = [tr[f"{k}|{iid}|{p}"] for p in range(3) if f"{k}|{iid}|{p}" in tr]
            if not rows: continue
            d = {}
            for L in (8, 16, 32):
                d[f"L{L}"] = dict(kernel_fraction_inside=rows[0][f"L{L}"]["kernel_fraction_inside"], max_worst=worst([r[f"L{L}"] for r in rows], "max_err"),
                                  rms_mean=float(np.mean([r[f"L{L}"]["rms_err"] for r in rows])), flux_lost=float(np.mean([r[f"L{L}"]["flux_lost"] for r in rows])),
                                  rescaled_max_worst=worst([r[f"L{L}"] for r in rows], "rescaled_max_err"))
            d["wing_flux_over_total"] = float(np.mean([r["wing_flux_over_total"] for r in rows]))
            out["truncation"][k][iid] = d
        out["local_sampling"][k] = {}
        for iid in [s[0] for s in SOURCES] + ["point"]:
            rows = [lo[f"{k}|{iid}|{p}"] for p in range(3) if f"{k}|{iid}|{p}" in lo]
            if not rows: continue
            d = {}
            for key in rows[0]:
                if key == "seconds": continue
                d[key] = dict(max_worst=worst([r[key] for r in rows]), max_median=float(np.median([r[key]["max"] for r in rows])),
                              rms_mean=float(np.mean([r[key]["rms"] for r in rows])))
            out["local_sampling"][k][iid] = d
    json.dump(out, open(os.path.join(HERE, "results", "e2_learned_results.json"), "w"), indent=1, default=float)
    make_figure(out)
    print("wrote results/e2_learned_results.json, figs/e2_learned_errors.png")


def make_figure(out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    ax = axs[0, 0]
    for i, k in enumerate(FILTERS):
        f = out["kernel_fits"].get(k)
        if not f: continue
        keys = [f"{L}_{m}_{sp}" for (L, m, sp) in KCONFIGS if f"{L}_{m}_{sp}" in f] + [q for q in f if q.endswith("_s0.08")]
        x = np.arange(len(keys))
        ax.plot(x, [f[q]["nonneg"]["max_err_16px_grid_over_true_peak"] for q in keys], "o-", color=cols[i], label=k + " non-negative")
        ax.plot(x, [f[q]["signed"]["max_err_16px_grid_over_true_peak"] for q in keys], "s--", color=cols[i], mfc="white", label=k + " signed")
    ax.set_xticks(np.arange(len(KCONFIGS) + 1)); ax.set_xticklabels([f"{L}x{(2*m+1)**2}\n{'' if sp==1 else 'half sp.'}" for (L, m, sp) in KCONFIGS] + ["extra:\nsigma0\n0.08 px"], fontsize=7)
    ax.axhline(1e-3, color="k", ls=":"); ax.set_yscale("log"); ax.set_ylabel("max |error| / true peak (16 samples per pixel)"); ax.set_title("(a) Gaussian pyramid fit of the kernel (levels x candidates per level)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3, which="both")
    for j, iid in enumerate(("sersic1_re2", "sersic1_re32")):
        ax = axs[0, 1 + j]
        for i, k in enumerate(FILTERS):
            t = out["truncation"].get(k, {}).get(iid)
            if not t: continue
            ax.plot([8, 16, 32], [t[f"L{L}"]["max_worst"] for L in (8, 16, 32)], "o-", color=cols[i], label=k)
        ax.axhline(1e-3, color="k", ls=":"); ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([8, 16, 32]); ax.set_xticklabels(["8", "16", "32"])
        ax.set_xlabel("kernel cut at +- L pixels"); ax.set_ylabel("max |error| / peak (vs full 10 arcsec kernel)")
        ax.set_title("(b) truncation, " + ("small source, half-light radius 2 px" if j == 0 else "large source, half-light radius 32 px")); ax.grid(alpha=0.3, which="both")
        if j == 0: ax.legend(fontsize=6, ncol=2)
    ax = axs[1, 0]
    for q, (iid, ls, mk) in enumerate((("sersic1_re2", "-", "o"), ("sersic1_re8", "--", "s"), ("sersic1_re32", ":", "^"))):
        for i, k in enumerate(FILTERS):
            d = out["local_sampling"].get(k, {}).get(iid)
            if not d: continue
            ax.plot(NLOC, [d[f"n={n}"]["max_worst"] for n in NLOC], ls, marker=mk, color=cols[i], label=f"{k} {iid[7:]}" if q == 0 else None)
    ax.axhline(1e-3, color="k", ls=":"); ax.set_yscale("log"); ax.set_xlabel("sub-samples per pixel"); ax.set_ylabel("max |error| / peak (worst of 3 positions)")
    ax.set_title("(c) local sampling, Sérsic n=1: solid re 2, dashed re 8, dotted re 32"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    ax = axs[1, 1]
    for i, k in enumerate(FILTERS):
        d = out["local_sampling"].get(k, {}).get("point")
        if not d: continue
        ax.plot([2, 4, 6, 8], [d[f"n={n}|cubic"]["max_worst"] for n in (2, 4, 6, 8)], "o-", color=cols[i], label=k + " cubic interpolation")
        ax.plot([2, 4, 6, 8], [d[f"n={n}|snap"]["max_worst"] for n in (2, 4, 6, 8)], "s--", color=cols[i], mfc="white", label=k + " nearest sample")
    ax.axhline(1e-3, color="k", ls=":"); ax.set_yscale("log"); ax.set_xlabel("kernel table samples per pixel"); ax.set_ylabel("max |error| / peak")
    ax.set_title("(c) point source at exact offset (vs exact kernel value)"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6, ncol=2)
    ax = axs[1, 2]
    for i, k in enumerate(FILTERS):
        kp = out["kernel_properties"]["kernels"][k]
        ax.plot([1, 2, 4, 8, 16, 32], [kp["encircled_energy_about_origin"][str(r)] for r in (1, 2, 4, 8, 16, 32)], "o-", color=cols[i], label=k)
    ax.set_xscale("log", base=2); ax.set_xlabel("radius, detector pixels"); ax.set_ylabel("encircled fraction of the integral"); ax.set_ylim(0, 1)
    ax.set_title("learned kernels: encircled energy (the rest lies beyond 32 px)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("E2 repeated with learned kernels (dotted line: 1e-3 of peak)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "e2_learned_errors.png"), dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--kernel", default="all"); ap.add_argument("--budget", type=float, default=200)
    a = ap.parse_args()
    names = FILTERS if a.kernel == "all" else a.kernel.split(",")
    t0 = time.time()
    if a.stage == "fits": stage_fits(names, a.budget)
    elif a.stage == "refs": stage_refs(names, a.budget)
    elif a.stage == "refcheck": stage_refcheck(names, a.budget)
    elif a.stage == "trunc": stage_trunc(names, a.budget)
    elif a.stage == "local": stage_local(names, a.budget)
    elif a.stage == "final": stage_final()
    print("stage %s took %.0f s" % (a.stage, time.time() - t0))
