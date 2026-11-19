"""Uncertainty of the fitted sky (added 18 Nov 2026; report: UNCERTAINTY_2026-11-18.md). Runs AFTER a fit; nothing here changes the fit.

Product. For a band with spectral weights phi (a filter's bandpass-averaged basis, model.Phi_band[f], or a unit vector for one spectral
coefficient), the sky is s_phi(x) = sum_j (phi . A_j) T(x - x_j) (A the coefficient grid, T the bilinear tent at the sky pitch). The product at
resolution s is the continuous sky convolved with a unit-integral circular Gaussian g_s, evaluated on an output grid:
    I_s(x) = sum_j (phi . A_j) (T * g_s)(x - x_j)          (separable; weights sum to 1, so a constant sky is unchanged)

Analytic estimator (linearised, Gauss-Newton posterior). The model of the training pixels is linear in A and in the per-exposure background
coefficients b (6 Legendre terms per exposure) once the response, registration, gain and noise terms are held at their fitted values:
    m = J_A A + L b + (terms held fixed),   weights W = diag(w_i),  w_i = min(1, delta / |z_i|) / sigma_i^2  (Huber IRLS weight at the fit)
    F = [J_A L]^T W [J_A L] + diag(mu I, 0)   restricted to the background gauge (per filter, the training mean of each of the 6
                                               background terms held; see SkyNormalOperator)
    Cov(I_s(x)) = k_x^T [F^{-1}]_AA k_x,  k_x = phi (x) (T * g_s)(x - .)
F^{-1} is never formed. Two ways, both by preconditioned conjugate gradients with the model's own renders (J_A v by a forward render with the
sky coefficients replaced by v, J_A^T r by the backward pass of that render):
  * samples: x = F^{-1} ([J_A L]^T W^{1/2} z + [sqrt(mu) z', 0]), z, z' standard normal, has covariance F^{-1} exactly ("perturb and solve");
    smoothing the A part of n such draws at any s gives sigma_s(x) for every output pixel and scale (relative error 1 / sqrt(2 n)).
  * exact: one solve F x = k_x per requested pixel and scale (sigma^2 = k_x . x_A), for a subsample of pixels (cross-check of the samples).
mu = (median node information) / rho^2 is a weak Gaussian prior on each coefficient (rho = --unc-prior, default 10 node-noise units); it only
matters at scales finer than the supported resolution, where the data do not constrain the sky (checked in the report).

Supported resolution. alpha(s, x) = -d ln sigma_s(x) / d ln s. For white noise alpha = 1 (the noise per element falls as its linear size);
for a sky that is a deconvolution, alpha > 1 at small s. For a Gaussian response of width p, sigma_s^2 is proportional to 1 / (s^2 - p^2) and
alpha = s^2 / (s^2 - p^2) = the variance of the element divided by the variance it would have without the blur. The supported resolution
s*(x) is the smallest s with alpha(s, x) <= 2 (for a Gaussian response s* = sqrt(2) p).

Empirical estimators. h = (I_s[fit on half a] - I_s[fit on half b]) / 2 for two fits on disjoint halves of the training exposures (alternate
dither groups, --train-half 0 / 1): its variance equals the full fit's if the noise variance scales as 1 / (number of exposures), and biases
common to both halves cancel. For two fits that differ only by the seed, sigma_opt = rms(I_a - I_b) / sqrt(2).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

from .model import legendre6

# ============================================================================= product: band vector, smoothing operator
SCALES_DEFAULT = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)     # in units of the sky pitch


def band_vector(model, band):
    """Spectral weights phi (K,): an int or a filter name -> model.Phi_band[f]; 'coef:k' -> unit vector on spectral coefficient k."""
    K = model.sky.K
    if isinstance(band, str) and band.startswith("coef:"):
        v = torch.zeros(K, device=model.Phi_band.device)
        v[int(band.split(":")[1])] = 1.0
        return v
    if isinstance(band, str):
        band = model.fset.names.index(band) if hasattr(model.fset, "names") else int(band)
    return model.Phi_band[int(band)].detach().clone()


def tent_gauss_1d(x_out, x_nodes, pitch, s, n_quad=48):
    """(n_out, N) matrix of (T * g_s)(x_out - x_j): tent of half-width pitch convolved with a unit Gaussian of width s (s = 0: the tent)."""
    x_out = torch.as_tensor(x_out, dtype=torch.float64)
    x_nodes = torch.as_tensor(x_nodes, dtype=torch.float64)
    u = x_out[:, None] - x_nodes[None]
    if s <= 0:
        return (1 - u.abs() / pitch).clamp(min=0)
    t, wq = np.polynomial.legendre.leggauss(n_quad)          # integral over the tent support [-p, p] in two halves
    out = torch.zeros_like(u)
    for a, b in ((-pitch, 0.0), (0.0, pitch)):
        tt = torch.tensor(0.5 * (b - a) * t + 0.5 * (b + a))
        ww = torch.tensor(0.5 * (b - a) * wq)
        tent = 1 - tt.abs() / pitch
        g = torch.exp(-0.5 * ((u[..., None] - tt) / s) ** 2) / (math.sqrt(2 * math.pi) * s)
        out += (g * (tent * ww)).sum(-1)
    return out                      # sum over nodes = 1 (the tent is a partition of unity)


class Smoother:
    """Separable linear map from a node image (N, N) to the product at resolution s on an output axis (default: the node positions)."""

    def __init__(self, N, pitch, s, x_out=None, dtype=torch.float32, device="cpu"):
        x_nodes = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * pitch
        self.x_out = x_nodes if x_out is None else torch.as_tensor(x_out, dtype=torch.float64)
        self.s, self.pitch = float(s), float(pitch)
        self.M = tent_gauss_1d(self.x_out, x_nodes, pitch, s).to(dtype=dtype, device=device)    # (n_out, N)

    def __call__(self, img):
        """img (..., N, N) rows = Y, columns = X -> (..., n_out, n_out)."""
        M = self.M.to(img.dtype)
        return M @ img @ M.T

    def weights_at(self, ix, iy):
        """Node weights (N, N) of output pixel (row iy, column ix): k = M[iy] (outer) M[ix]."""
        return self.M[iy][:, None] * self.M[ix][None, :]


# ============================================================================= the linear operator of the fit
@contextlib.contextmanager
def frozen(model):
    """Every model parameter without gradient while the operator renders (only the sky coefficients passed in carry one)."""
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, f in flags:
            p.requires_grad_(f)


@contextlib.contextmanager
def sky_override(sky, A):
    """Render with the coefficient grid A in place of the fitted sky (instance attribute shadows the method; removed on exit)."""
    sky.coeffs = lambda view=None: A
    try:
        yield
    finally:
        del sky.coeffs


def default_tile(e, cfg):
    J = np.asarray(e.extra.get("J", [[cfg.pix_default, 0], [0, cfg.pix_default]]))
    return max(32, int(round(cfg.patch_arcsec / max(abs(np.linalg.det(J)) ** 0.5, 1e-6))))


DEFAULT_BATCH_PX = 16000        # pixels per batched render on a CUDA device (about the fit's batch: 5 patches of ~2,500 pixels)
DEFAULT_TILE_CUDA = 64          # detector pixels per tile edge on a CUDA device (CPU: --region-px, the fastest per pixel on one thread)
LEGACY_CHUNK = 32               # tiles per batched render when no pixel budget is in force (CPU default, the validated one)


def plan_passes(K, method="draws", polish=True, n_samples=8, maxiter=60, n_exact=0, n_bands=1):
    """Planned cost in applications of F (one application = one forward and one adjoint render of every training pixel; at most: a conjugate-
    gradient solve stops earlier when it reaches --unc-tol). The set-up also has one forward-only pass (the weights), counted separately."""
    setup = K + 1                                        # coverage (J^T w) + K delta responses of the Fourier preconditioner
    pol = (1 + maxiter) if polish else 0                 # right-hand side + iterations
    dr = n_samples * (1 + maxiter) if (method == "draws" and n_samples > 0) else 0
    ex = n_exact * n_bands * 41                          # typical (measured on the truth problem); the cap is 600 per solve
    return {"setup": setup, "forward_only": 1, "polish": pol, "draws": dr, "exact": ex, "total": setup + pol + dr + ex}


def resolve_batching(device_type, region_px, tile=None, chunk=None, batch_px=None):
    """(tile, batch_px, chunk). Explicit tile / chunk win. CUDA: tile 64 and a budget of 16000 pixels per batched render unless given. CPU: tile =
    region_px and 32 tiles per render (the validated default) unless a budget or a chunk is given."""
    cuda = device_type == "cuda"
    if batch_px is None and not chunk and cuda:
        batch_px = DEFAULT_BATCH_PX
    tile = tile or (DEFAULT_TILE_CUDA if cuda else region_px)
    batch_px = None if chunk else batch_px
    return tile, batch_px, (chunk or (None if batch_px else LEGACY_CHUNK))


def pack_batches(sizes, batch_px=None, chunk=None):
    """Indices of tiles grouped into batched renders: fixed number of tiles (chunk), or greedily up to batch_px pixels (a tile larger than the
    budget goes alone). Order is kept."""
    out, cur, s = [], [], 0
    for i, n in enumerate(sizes):
        if chunk:
            if len(cur) == chunk:
                out.append(cur)
                cur = []
        elif cur and s + n > batch_px:
            out.append(cur)
            cur, s = [], 0
        cur.append(i)
        s += n
    if cur:
        out.append(cur)
    return out


class Tile:
    """One detector tile of a training exposure. Holds ONLY what cannot be recomputed cheaply: the exposure, the slice, the validity mask
    (bool) and the weights w and w r (float32): about 9 bytes per pixel. The pixel dictionary (positions, data, 10 more arrays) and the
    background basis are rebuilt for each batch and freed after it (keys 'pix' and 'Lb' build them on access, for the tests)."""
    __slots__ = ("op", "eid", "row", "e", "sl", "n", "mask", "w", "wr")

    def __init__(self, op, e, sl, mask):
        self.op, self.e, self.sl, self.mask = op, e, sl, mask
        self.eid, self.row = int(e.eid), op.row_of[int(e.eid)]
        self.n = (sl[1] - sl[0]) * (sl[3] - sl[2])
        self.w = self.wr = None

    def __getitem__(self, k):
        if k == "pix":
            return self.e.pix(self.sl)
        if k == "Lb":
            return self.op._lb(self, self.e.pix(self.sl))
        return getattr(self, k) if k != "e" else self.e


class Monitor:
    """Progress lines and the time budget of a run. stop_now() is asked before every application of F: True when the application (estimated
    from the last one, plus 15 per cent and the reserve for writing the outputs) would end after the deadline. Once it said True, .hit is set."""

    def __init__(self, op, log=print, max_minutes=0.0, t0=None, planned_apps=None, every=5, reserve=60.0, clock=time.perf_counter):
        self.op, self.log, self.every, self.reserve, self.clock = op, log, every, reserve, clock
        self.t0 = clock() if t0 is None else t0
        self.deadline = self.t0 + 60.0 * max_minutes if max_minutes and max_minutes > 0 else None
        self.planned = planned_apps
        self.hit, self.where, self.residual = False, "", float("nan")

    def elapsed(self):
        return self.clock() - self.t0

    def eta(self):
        if not self.planned or self.op.n_apply == 0:
            return float("nan")
        return max(self.planned - self.op.n_apply, 0) * self.op.t_apply / self.op.n_apply

    def stop_now(self, where=""):
        if self.deadline is None:
            return False
        if self.clock() + 1.15 * getattr(self.op, "t_last", 0.0) + self.reserve > self.deadline:
            self.hit, self.where = True, where
            return True
        return False

    def report(self, label, it, rel):
        self.residual = rel
        if it and it % self.every == 0:
            self.log(f"  {label}: iteration {it}, relative residual {rel:.2e}, {self.op.n_apply} passes so far, elapsed {self.elapsed() / 60:.1f} min, "
                     f"estimated time left (planned, upper bound) {self.eta() / 60:.1f} min")

    def tail(self):
        return f"{self.op.n_apply} passes, elapsed {self.elapsed() / 60:.1f} min, estimated time left (upper bound) {self.eta() / 60:.1f} min"


def _eigh_rows(fh, fn, rows=32):
    """Apply fn(block of fh (R, N, K, K)) over blocks of rows (bounds the memory of the batched eigendecomposition)."""
    # On the CPU in every case: the GPU's batched eigensolver refused the real field's batch (cusolver INVALID_VALUE, 18 Nov), and
    # N * N matrices of K x K take seconds on the host. A non-finite entry is reported here rather than as a solver error.
    if not bool(torch.isfinite(fh.real).all() and torch.isfinite(fh.imag).all()):
        raise RuntimeError(f"Fourier blocks of the normal operator hold {int((~torch.isfinite(fh.real)).sum())} non-finite entries")
    dev = fh.device
    return torch.cat([fn(fh[r:r + rows].cpu()) for r in range(0, fh.shape[0], rows)]).to(dev)


class SkyNormalOperator:
    """F = [J_A L]^T W [J_A L] + diag(mu, 0) on the flat vector x = [A (K N N), b (n_train_rows x 6)], restricted to a background gauge.

    Gauge. A sky polynomial of order <= 2 over the field is exactly degenerate with the per-exposure quadratic backgrounds (a sky gradient g x
    seen by an exposure dithered by c_e is g x_det + g c_e: a linear plus a constant background term). The fit fixes only the mean constant
    (pedestal gauge); the other five directions are left to the optimiser and the warm start. bkg_gauge="all" (default) holds the training mean
    of each of the 6 terms at its fitted value, so those sky modes are NOT in sigma (listed in the report); "constant" uses the pedestal gauge
    only, and the five degenerate sky modes then get the prior variance 1 / mu (sigma then depends on --unc-prior at every scale).

    Held at the fitted values (not marginalised): response (kernel, throughput, distortion), registration, gain u_e, pixel gain, detector
    pattern, stripes, noise scale and floor (the weights). Marginalised: every training exposure's 6 background coefficients.

    Batching (real-data adaptation). Every render goes through self.batches (lists of tiles): up to batch_px pixels per batched render (default
    16000 on a CUDA device with tiles of 64 detector pixels), or chunk tiles per render when chunk is given; on CPU without either, tiles of
    cfg.region_px and 32 per render (the validated default). Pixel dictionaries, background bases and every render output are built for one batch
    and released after it; what stays on the device is the weights (two float32 and one bool per training pixel) plus fixed-size arrays."""

    def __init__(self, model, exps, cfg, P=None, chunk=None, rho=10.0, tile=None, huber=True, precond_mode="fourier", bkg_gauge="all", log=print,
                 batch_px=None, plan=None, mu=None, n_regions=8, region_smooth=1.5, coverage="quadratic", n_probes=1, precision="float32",
                 region_box=None, exclude_bright=0.0, bkg_basis="raw", bkg_rcond=1e-4):
        """precond_mode: 'fourier' (the 18 Nov preconditioner: one Fourier model probed at the centre node, scaled by the all-filter coverage;
        bit-identical to the calibrated truth-problem runs; class default), 'jacobi', or 'filter' (19 Nov, SOLVER_DIAGNOSIS_2026-11-19.md: one
        Fourier model per filter probed where that filter is well covered, per-node K x K scaling by each filter's own coverage, and the prior
        handled exactly where the data are absent; the command-line default). mu: prior weight given explicitly (default: from the data).
        precision (19 Nov, second part of the report): 'float32' (as before, bit-identical) or 'mixed': the renders stay in float32, but the adjoint
        sums are accumulated over batches in float64, the background blocks are built and inverted in float64 (eigenvalue pseudo-inverse with a
        relative floor), and the operator's vectors (apply, preconditioner, right-hand sides) are float64.
        region_box (arcsec, half width; lab only): keep the training pixels whose header position is within it. exclude_bright (lab only): drop
        the pixels whose noise-floor variance exceeds this fraction of their total variance (0 = keep all). Both change the estimator."""
        if precond_mode not in ("fourier", "jacobi", "filter", "none", "bjacobi"):
            raise ValueError(f"precond_mode {precond_mode!r}")
        if precision not in ("float32", "mixed"):
            raise ValueError(f"precision {precision!r}")
        self.precision = precision
        self.vdtype = torch.float64 if precision == "mixed" else torch.float32
        self.setup_mode = "fourier" if precond_mode in ("jacobi", "none") else ("filter" if precond_mode == "bjacobi" else precond_mode)
        if bkg_basis not in ("raw", "orth"):
            raise ValueError(f"bkg_basis {bkg_basis!r}")
        if bkg_basis == "orth" and precision != "mixed":
            raise ValueError("bkg_basis='orth' needs precision='mixed'")
        self.bkg_basis, self.bkg_rcond = "raw", float(bkg_rcond)        # 'raw' until the weights pass has built the orthonormal basis
        self.model, self.cfg = model, cfg
        self.n_regions, self.region_smooth, self.coverage_kind, self.n_probes = int(n_regions), float(region_smooth), coverage, int(n_probes)
        self.bkg_gauge = bkg_gauge          # "constant": the fit's pedestal gauge only; "all": the training mean of each of the 6 terms held
        self.log = log
        sky = model.sky
        self.K, self.N = sky.K, sky.N
        self.nA = self.K * self.N * self.N
        dev = model.dev
        tile, self.batch_px, self.chunk = resolve_batching(torch.device(dev).type, int(cfg.region_px), tile, chunk, batch_px)
        self.tile = tile
        with torch.no_grad():
            self.P = P if P is not None else {k: (v.detach() if torch.is_tensor(v) else v) for k, v in model.prepare().items()}
        tr = [e for e in exps if e.split == "train"]
        self.rows = sorted({int(e.eid) for e in tr})
        self.row_of = {r: i for i, r in enumerate(self.rows)}
        self.nb = 6 * len(self.rows)
        # pedestal gauge groups: per filter, the training rows (sum of their b_e0 fixed)
        gf = {}
        for e in tr:
            gf.setdefault(int(e.filt), set()).add(self.row_of[int(e.eid)])
        self.gauge_groups = [torch.tensor(sorted(v), device=dev) for v in gf.values()] if getattr(cfg, "pedestal_gauge", 1) else []
        h = sky.half_extent - cfg.pred_margin
        self.tiles = []
        A_fit = sky.coeffs().detach()
        n_pix, n_down = 0, 0
        with torch.no_grad():
            for e in tr:
                ny, nx = e.shape
                for y0 in range(0, ny, tile):
                    for x0 in range(0, nx, tile):
                        sl = (y0, min(ny, y0 + tile), x0, min(nx, x0 + tile))
                        pix = e.pix(sl)
                        inside = ((pix["X"] + e.shift0[0]).abs() < h) & ((pix["Y"] + e.shift0[1]).abs() < h)
                        mask = pix["valid"] & inside
                        if region_box:
                            mask = mask & ((pix["X"] + e.shift0[0]).abs() < region_box) & ((pix["Y"] + e.shift0[1]).abs() < region_box)
                        k = int(mask.sum())
                        if k == 0:
                            continue
                        n_pix += k
                        self.tiles.append(Tile(self, e, sl, mask))
        self.tile_filt = [int(t.e.filt) for t in self.tiles]
        self.filters = sorted(set(self.tile_filt))
        if precond_mode in ("filter", "bjacobi"):     # one filter per batched render, so that each filter's pass can carry its own probe (same pixels per pass)
            self.batches = []
            for f in self.filters:
                idx = [i for i, ff in enumerate(self.tile_filt) if ff == f]
                self.batches += [[idx[j] for j in b] for b in pack_batches([self.tiles[i].n for i in idx], self.batch_px, self.chunk)]
        else:
            self.batches = pack_batches([t.n for t in self.tiles], self.batch_px, self.chunk)
        self.n_pix = n_pix
        self.A_fit = A_fit
        self.dev = dev
        self.n_apply = 0
        self.t_apply = 0.0
        self.t_last = 0.0
        self.n_forward = 0
        self.t_forward = 0.0
        K = self.K
        pl = plan_passes(K, **plan) if plan else None
        px_render = sum(t.n for t in self.tiles)
        big = max((sum(self.tiles[i].n for i in b) for b in self.batches), default=0)
        log(f"uncertainty plan: {len(self.tiles)} tiles of up to {tile} x {tile} detector pixels, {n_pix} training pixels ({px_render} rendered), "
            f"{len(self.batches)} batches per pass (largest {big} pixels; budget {self.batch_px if self.batch_px else 'none'}, "
            f"{self.chunk if self.chunk else 'by budget'} tiles per batch)"
            + (f"; planned passes (one pass = forward + adjoint over all training pixels): set-up {pl['setup']} + 1 forward-only, "
               f"Gauss-Newton step {pl['polish']}, draws {pl['draws']}, exact solves ~{pl['exact']}; total up to {pl['total']}" if pl else ""))
        # weights at the fit (forward only): w, w r per tile, the 6 x 6 background blocks (exact, block-Jacobi)
        Bb = torch.zeros(len(self.rows), 6, 6, device=dev, dtype=torch.float64 if precision == "mixed" else torch.float32)
        n_bright = 0
        t0 = time.perf_counter()
        with torch.no_grad(), sky_override(sky, A_fit):
            for ch in self._batches():
                pixs = [t["pix"] for t in ch]
                outs = model.render_batch([(t.eid, p) for t, p in zip(ch, pixs)], self.P)
                for t, pix, o in zip(ch, pixs, outs):
                    et = model.et(t.eid)
                    S, m = o["S"], o["m"]
                    sig = model.sigma(et, S, pix["sig"])
                    z = (pix["d"] - m) / sig
                    w = 1.0 / sig ** 2
                    if huber and cfg.loss == "huber":
                        hw = torch.clamp(cfg.huber_delta / z.abs().clamp(min=1e-12), max=1.0)
                        n_down += int(((hw < 1) & t.mask).sum())
                        w = w * hw
                    if exclude_bright and exclude_bright > 0:     # lab: drop the pixels whose variance is mostly the noise floor (f S)^2
                        fl = torch.nn.functional.softplus(model.floor_raw[model.e_chan[et]])
                        br = (fl * S) ** 2 > exclude_bright * sig ** 2
                        n_bright += int((br & t.mask).sum())
                        w = torch.where(br, torch.zeros_like(w), w)
                    t.w = torch.where(t.mask, w, torch.zeros_like(w))
                    t.wr = t.w * (pix["d"] - m)            # weighted residual at the fit (Gauss-Newton step)
                    if precond_mode in ("filter", "bjacobi"):           # weights splatted at the pixels' sky positions (filter mode: quadratic coverage)
                        self._splat_weights(t, pix)
                    Lb = self._lb(t, pix)
                    if precision == "mixed":
                        Lb64 = Lb.double()
                        Bb[t.row] += Lb64.T @ (t.w.double()[:, None] * Lb64)
                    else:
                        Bb[t.row] += Lb.T @ (t.w[:, None] * Lb)
                del outs, pixs
        self.n_bright_excluded = n_bright
        self.Bb = Bb
        if precision == "mixed":
            self._setup_bkg_basis(Bb, log)
            self.bkg_basis = bkg_basis
        self.n_huber = n_down
        self.n_forward += 1
        self.t_forward += time.perf_counter() - t0
        log(f"first pass (forward only, the weights): {self.t_forward:.1f} s for {len(self.batches)} batches; a full application "
            f"(forward + adjoint) is expected to take about {1.8 * self.t_forward:.1f} s")
        # preconditioner of the sky block (F_AA + mu): Fourier (circulant) model of F_AA from its response to a delta at the centre node,
        # scaled by the relative coverage d_j = (J^T w)_j / (J^T w)_centre; Jacobi on the nodes with d < 0.05 (edges, uncovered)
        t0 = time.perf_counter()
        self.precond_mode = precond_mode
        self.rho = rho
        self._qcache = {}
        if self.setup_mode == "filter":
            self._setup_filter_precond(rho, mu, log, pl)
            self.Bb_inv = self._bb_inverse(Bb)
            self.t_setup_precond = time.perf_counter() - t0
            log(f"uncertainty operator: {len(self.tiles)} tiles, {n_pix} training pixels ({n_down} with Huber weight < 1), {len(self.rows)} exposures; "
                f"coefficients {self.nA}, background terms {self.nb}; node noise (median, coefficient units) {self.node_noise:.4g}; prior rho {rho} -> mu {self.mu:.4g}; "
                f"preconditioner 'filter' {self.t_setup_precond:.1f} s ({self.n_apply} applications + {self.n_forward} forward-only)")
            return
        self.cov = self._JT_w().sum(0)                                     # (N, N): sum_k (J_A^T w)
        log(f"first full application (forward + adjoint): {self.t_last:.1f} s" + (f"; planned total up to {pl['total']} passes = about "
            f"{pl['total'] * self.t_last / 3600:.2f} h" if pl else ""))
        c = (self.N - 1) // 2
        cols = []
        for k in range(self.K):
            v = torch.zeros(self.K, self.N, self.N, device=dev)
            v[k, c, c] = 1.0
            cols.append(self._data_normal(v.reshape(-1), torch.zeros(self.nb, device=dev))[0].view(self.K, self.N, self.N).float())
        self.fc = fc = torch.stack(cols, 1)                                # (K', K, N, N): F[(k', j), (k, centre)]
        dcen = torch.stack([fc[k, k, c, c] for k in range(self.K)])         # (K,) diagonal entries at the centre
        rel = (self.cov / self.cov[c, c].clamp(min=1e-30)).clamp(min=0)
        self.covered = rel > 1e-3
        dA = dcen[:, None, None] * rel[None]
        dmed = float(torch.stack([dA[k][self.covered].median() for k in range(self.K)]).max())
        self.mu = dmed / rho ** 2 if mu is None else float(mu)
        self.rho = rho
        self.node_noise = 1.0 / math.sqrt(dmed)
        self.dA = dA + self.mu
        self.fmask = rel > 0.05
        self.fscale = torch.where(self.fmask, rel, torch.ones_like(rel)).rsqrt()
        fh = torch.fft.fft2(torch.roll(fc, shifts=(-c, -c), dims=(-2, -1)))  # (K', K, N, N)
        fh = fh.permute(2, 3, 0, 1)                                          # (N, N, K', K)

        def _inv(b):
            b = 0.5 * (b + b.conj().transpose(-1, -2))
            ev, V = torch.linalg.eigh(b)
            ev = ev.clamp(min=0) + self.mu
            return (V * (1.0 / ev)[..., None, :]) @ V.conj().transpose(-1, -2)
        self.finv = _eigh_rows(fh, _inv)                                    # (N, N, K, K)
        del fh
        self.kappa = (dcen / self.cov[c, c].clamp(min=1e-30)).tolist()
        kap = self.kappa
        self.Bb_inv = self._bb_inverse(Bb)
        self.t_setup_precond = time.perf_counter() - t0
        log(f"uncertainty operator: {len(self.tiles)} tiles, {n_pix} training pixels ({n_down} with Huber weight < 1), {len(self.rows)} exposures; "
            f"coefficients {self.nA}, background terms {self.nb}; node noise (median, coefficient units) {self.node_noise:.4g}; prior rho {rho} -> mu {self.mu:.4g}; "
            f"kappa {np.round(kap, 5).tolist()}; preconditioner {self.t_setup_precond:.1f} s ({self.n_apply} applications + {self.n_forward} forward-only)")

    def _batches(self):
        for b in self.batches:
            yield [self.tiles[i] for i in b]

    def _bb_inverse(self, Bb):
        """Inverses of the per-exposure 6 x 6 background blocks. float32: inv(Bb + 1e-9 I) (the 18 Nov code). mixed: float64 eigenvalue
        pseudo-inverse with a relative floor 1e-12 x the block's largest eigenvalue (the 6 Legendre terms of a detector that the box crosses only
        in part are nearly collinear; an absolute 1e-9 is no regulariser for blocks of order 1e10, and float32 cannot invert them).
        Records the blocks' condition numbers (self.bb_cond)."""
        B64 = 0.5 * (Bb.double() + Bb.double().transpose(-1, -2)).cpu()
        ev = torch.linalg.eigvalsh(B64)
        self.bb_cond = (ev[:, -1] / ev[:, 0].clamp(min=1e-300)).tolist()
        if self.precision != "mixed":
            return torch.linalg.inv(Bb + 1e-9 * torch.eye(6, device=Bb.device, dtype=Bb.dtype))
        self.Bb_inv_f32 = torch.linalg.inv(Bb.float() + 1e-9 * torch.eye(6, device=Bb.device))   # the 18 Nov inverse (for set_precision)
        lam, V = torch.linalg.eigh(B64)
        floor = 1e-12 * lam[:, -1:].clamp(min=1e-300)
        inv = torch.where(lam > floor, 1.0 / lam.clamp(min=1e-300), torch.zeros_like(lam))
        self.Bb_inv_f64 = ((V * inv[:, None, :]) @ V.transpose(-1, -2)).to(self.dev)
        return self.Bb_inv_f64

    def _setup_bkg_basis(self, Bb, log):
        """Orthonormal background basis (bkg_basis 'orth', 19 Nov, section 11 of the report). Per exposure, the six Legendre functions times the
        gain are orthonormalised over the exposure's own weighted valid pixels: Bb_r = V diag(lam) V^T (float64), b_r = T_r c_r with
        T_r = V diag(lam^{-1/2}), so the background block in c is the identity. Directions with sqrt(lam / lam_max) < bkg_rcond (combinations of the
        six terms that the cut cannot tell apart: they change the model by less than bkg_rcond of the strongest one) are dropped, i.e. held at
        the fit; they are degenerate with large-scale sky modes. The gauge (the training mean of each raw term per filter, or of the constant only)
        is imposed exactly by the orthogonal projector onto {C c = 0, dropped slots 0} per group."""
        B64 = 0.5 * (Bb.double() + Bb.double().transpose(-1, -2)).cpu()
        lam, V = torch.linalg.eigh(B64)
        lmax = lam[:, -1:].clamp(min=1e-300)
        keep = lam > (self.bkg_rcond ** 2) * lmax
        T = V * torch.where(keep, lam.clamp(min=1e-300).rsqrt(), torch.zeros_like(lam))[:, None, :]
        self.bkgT = T.to(self.dev)
        self.bkg_keep = keep.to(self.dev)
        self.n_bkg_dropped = int((~keep).sum())
        terms = list(range(6)) if self.bkg_gauge == "all" else [0]
        proj, in_group = [], torch.zeros(len(self.rows), dtype=torch.bool)
        for g in self.gauge_groups:
            rows = g.cpu()
            in_group[rows] = True
            C = torch.cat([T[r][terms, :] for r in rows.tolist()], dim=1)                 # (n_terms, 6 n_g)
            D = keep[rows].reshape(-1).double()
            CD = C * D[None]
            Pg = torch.diag(D) - CD.T @ torch.linalg.pinv(CD @ CD.T, hermitian=True, rtol=1e-12) @ CD
            proj.append((rows.to(self.dev), Pg.to(self.dev)))
        self.bkg_proj, self.bkg_in_group = proj, in_group.to(self.dev)
        log(f"background basis: {len(self.rows)} exposures x 6 terms, orthonormalised over each exposure's weighted pixels; "
            f"{self.n_bkg_dropped} directions dropped (singular value below {self.bkg_rcond:g} of the largest; held at the fit); block condition "
            f"numbers before: max {float((lam[:, -1] / lam[:, 0].clamp(min=1e-300)).max()):.3e}")

    def bkg_to_raw(self, xb):
        """Background terms in the raw Legendre coefficients (n_rows, 6) from the solver's vector (raw already, or orthonormal coordinates)."""
        xb = xb.reshape(-1, 6)
        if self.bkg_basis != "orth":
            return xb
        return torch.einsum("rij,rj->ri", self.bkgT.to(xb.dtype), xb)

    def _bkg_grad(self, gb):
        """A gradient with respect to the raw terms (n_rows, 6) -> with respect to the solver's background coordinates."""
        if self.bkg_basis != "orth":
            return gb
        return torch.einsum("rji,rj->ri", self.bkgT.to(gb.dtype), gb.reshape(-1, 6))

    def set_bkg_basis(self, basis):
        """Switch between the raw background terms and the orthonormal basis (an operator built with precision 'mixed'). For the lab."""
        if basis == "orth" and not hasattr(self, "bkgT"):
            raise ValueError("the orthonormal background basis needs precision='mixed' at construction")
        self.bkg_basis = basis

    def set_precision(self, precision):
        """Switch an operator built with precision 'mixed' between 'mixed' and 'float32' (float32: float32 vectors, float32 adjoint sums, the 18 Nov
        background inverse). For the lab."""
        if precision == self.precision:
            return
        if not hasattr(self, "Bb_inv_f64"):
            raise ValueError("set_precision needs an operator built with precision='mixed'")
        self.precision = precision
        self.vdtype = torch.float64 if precision == "mixed" else torch.float32
        self.Bb_inv = self.Bb_inv_f64 if precision == "mixed" else self.Bb_inv_f32

    # ---------------------------------------------------------------- per-filter preconditioner (precond_mode 'filter', 19 Nov)
    def _splat_weights(self, t, pix):
        """Add the tile's weights w_i, bilinearly, at its pixels' sky positions (the model's positions, mean over the lambda nodes) to the
        per-filter weight map W_f (N, N) on the node grid."""
        if not hasattr(self, "wsplat"):
            self.wsplat = {}
        f = int(t.e.filt)
        N, sky = self.N, self.model.sky
        W = self.wsplat.setdefault(f, torch.zeros(N * N, device=self.dev, dtype=torch.float64))
        pos = self.model.positions(self.model.et(t.eid), pix["X"], pix["Y"], pix["xi"], self.P, c0=pix.get("c0")).mean(0)
        u = (pos - sky.x0) / sky.pitch                                                # node coordinates (x = column, y = row)
        i0 = torch.floor(u).long()
        fr = u - i0
        w = t.w.double()
        for dy in (0, 1):
            for dx in (0, 1):
                ix, iy = i0[:, 0] + dx, i0[:, 1] + dy
                wt = (fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1])
                ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (w > 0)
                W.index_add_(0, (iy * N + ix)[ok], (w * wt.double())[ok])

    def _render_sum_by_filter(self, xAs, rfun):
        """One application in which the tiles of filter f render with their own coefficient grid xAs[f] (batches hold one filter each): returns
        {f: J_f^T r_f (K, N, N)}. Same pixels and cost as _render_sum."""
        model = self.model
        t0 = time.perf_counter()
        leaves = {f: x.detach().float().clone().requires_grad_(True) for f, x in xAs.items()}
        mixed = self.precision == "mixed"
        acc = {f: torch.zeros(l.shape, dtype=torch.float64, device=l.device) for f, l in leaves.items()} if mixed else None
        with torch.enable_grad(), frozen(model):
            for b in self.batches:
                ch = [self.tiles[i] for i in b]
                lf = leaves[self.tile_filt[b[0]]]
                with sky_override(model.sky, lf):
                    pixs = [t["pix"] for t in ch]
                    outs = model.render_batch([(t.eid, p) for t, p in zip(ch, pixs)], self.P)
                    Ss = [o["S"] for o in outs]
                    Rs = [rfun(t, o["S"].detach(), None) for t, o in zip(ch, outs)]
                    if mixed:                     # this batch's adjoint alone (float32), added to a float64 sum
                        g_ = torch.autograd.grad(Ss, lf, Rs, allow_unused=True)[0]
                        if g_ is not None:
                            acc[self.tile_filt[b[0]]] += g_.double()
                    else:
                        torch.autograd.backward(Ss, Rs)
                del outs, Ss, Rs, pixs
        self.n_apply += 1
        self.t_last = time.perf_counter() - t0
        self.t_apply += self.t_last
        if mixed:
            return acc
        return {f: (l.grad.detach() if l.grad is not None else torch.zeros_like(l)) for f, l in leaves.items()}

    def _setup_filter_precond(self, rho, mu, log, pl, n_regions=None, smooth=None):
        """Preconditioner of precond_mode 'filter' (SOLVER_DIAGNOSIS_2026-11-19.md, section 4). Pieces:
          c_f(j)   = sum_k (J_f^T w_f)_{kj}: linear coverage of node j by the training pixels of filter f (one application for all filters);
          p_f      = probe node of filter f: the node nearest the box centre with c_f within 10 per cent of its 90th percentile;
          G_f(d)   = response of filter f's normal operator to a unit coefficient k at p_f, read at p_f + d (K x K per lag d; K applications for
                     all filters together, each filter's pixels carrying the delta at its own probe); d_f(j) = c_f(j) / c_f(p_f);
          M_j      = sum_f d_f(j) G_f(0) + mu I: the local K x K information of node j;
          regions  r = 1..R: k-means of theta_f(j) = d_f(j) g_f / (d_f(j) g_f + mu) (g_f the largest eigenvalue of G_f(0)), i.e. nodes grouped by
                     which filters constrain them (gaps, edges, bright-source cores, the uncovered margin); a_r(j) >= 0, sum_r a_r = 1, the
                     indicator of region r smoothed over `smooth` nodes; dbar_rf the mean of d_f over region r;
          C_r(w)   = sum_f dbar_rf FFT[G_f](w) + mu I  (K x K per frequency w),  M_r = sum_f dbar_rf G_f(0) + mu I;
          P^{-1}   = M_j^{-1/2} [ sum_r a_r^{1/2} M_r^{1/2} C_r^{-1} M_r^{1/2} a_r^{1/2} ] M_j^{-1/2}.
        Symmetric positive definite. Exact where one region's coverage is uniform (P = C_r) and where no data reach a node (P = M_j = mu I)."""
        K, N, dev = self.K, self.N, self.dev
        R_ = int(n_regions or getattr(self, "n_regions", 8))
        sm_nodes = float(smooth or getattr(self, "region_smooth", 1.5))
        c = (N - 1) // 2
        Z = torch.zeros(K, N, N, device=dev)
        covf = self._render_sum_by_filter({f: Z for f in self.filters}, lambda t, S, Lb: t.w)
        log(f"first full application (forward + adjoint, per-filter coverage): {self.t_last:.1f} s" + (f"; planned total up to {pl['total']} passes = "
            f"about {pl['total'] * self.t_last / 3600:.2f} h" if pl else ""))
        cf = {f: g.sum(0) for f, g in covf.items()}                                  # (N, N) per filter
        self.cov = sum(cf.values())
        self.cov_filter = cf
        yy, xx = torch.meshgrid(torch.arange(N, device=dev), torch.arange(N, device=dev), indexing="ij")
        r2 = ((yy - c) ** 2 + (xx - c) ** 2).float()
        # probes: up to n_probes per filter, at least 2 Rw + 1 nodes apart (Chebyshev), nearest the centre first, among the nodes whose coverage
        # is within 10 per cent of the 90th percentile; their windowed responses are averaged (the Fourier model averaged over sub-pixel phases).
        # With room for one probe only: one probe, no window (the response of the whole field).
        Rw = int(max(6, min(40, N // 7)))
        nmax = int(getattr(self, "n_probes", 1))
        probes, plist = {}, {}
        for f, m in cf.items():
            pos = m > 0.05 * m.max()
            thr = 0.9 * torch.quantile(m[pos].float().cpu(), 0.9).item() if bool(pos.any()) else 0.0
            i = int(torch.argmin(torch.where(m >= thr, r2, torch.full_like(r2, float("inf")))))
            probes[f] = (i // N, i % N)
            inside = (yy >= Rw) & (yy < N - Rw) & (xx >= Rw) & (xx < N - Rw)
            cand = torch.nonzero((m >= thr) & inside).cpu()
            order = torch.argsort(r2[cand[:, 0], cand[:, 1]].cpu())
            pk = []
            for q in order.tolist():
                y_, x_ = int(cand[q, 0]), int(cand[q, 1])
                if all(max(abs(y_ - a), abs(x_ - b)) >= 2 * Rw + 1 for a, b in pk):
                    pk.append((y_, x_))
                if len(pk) >= nmax:
                    break
            plist[f] = pk if len(pk) >= 2 else [probes[f]]
            probes[f] = plist[f][0]
        self.probes, self.probe_list = probes, plist
        u = torch.fft.fftfreq(N, 1.0 / N).round().abs()                              # |lag| per index
        t1 = torch.where(u <= Rw / 2, torch.ones(N), torch.where(u <= Rw, torch.cos(0.5 * math.pi * (u - Rw / 2) / (Rw / 2)) ** 2, torch.zeros(N)))
        win = (t1[:, None] * t1[None, :]).float()                                     # separable taper, lag 0 at index 0
        nF = len(self.filters)
        kernF = torch.zeros(nF, K, K, N, N)                                          # host: (filter, K', K, lag y, lag x), lag 0 at index 0
        G0 = torch.zeros(nF, K, K, dtype=torch.float64)
        for k in range(K):
            xAs = {}
            for f, pk in plist.items():
                v = torch.zeros(K, N, N, device=dev)
                for (py, px) in pk:
                    v[k, py, px] = 1.0
                xAs[f] = v
            out = self._render_sum_by_filter(xAs, lambda t, S, Lb: t.w * S)
            for i, f in enumerate(self.filters):
                pk = plist[f]
                o = out[f].cpu()
                for (py, px) in pk:
                    r_ = torch.roll(o, shifts=(-py, -px), dims=(-2, -1))
                    kernF[i, :, k] += (r_ * win if len(pk) > 1 else r_) / len(pk)
                    G0[i, :, k] += o[:, py, px].double() / len(pk)
            del out
        G0 = 0.5 * (G0 + G0.transpose(-1, -2))
        self.G0 = G0
        self.fc = torch.roll(kernF.sum(0), shifts=(c, c), dims=(-2, -1)).to(dev)    # all filters at their probes' coverage (origin at the centre node)
        cref = {f: torch.stack([cf[f][p] for p in plist[f]]).mean() for f in self.filters}   # coverage the averaged model stands for
        d_lin = torch.stack([(cf[f] / cref[f].clamp(min=1e-30)).clamp(min=0) for f in self.filters]).double().cpu()   # (F, N, N)
        if getattr(self, "coverage_kind", "quadratic") == "quadratic" and hasattr(self, "wsplat"):
            # quadratic coverage q_f = W_f (*) psi_f^2: the information sum_i w_i psi_ij^2 of node j (at a bright source the weights fall inside
            # the footprint, and the linear coverage sum_i w_i psi_ij overstates it). psi_f^2(l) ~ g_f(2 l), g_f(l) = 1^T G_f(l) 1 the
            # flat-spectrum delta response (the footprint's autocorrelation): exact for a Gaussian footprint.
            dq = []
            for i, f in enumerate(self.filters):
                g = kernF[i].double().sum((0, 1))                                    # (N, N), lag 0 at index 0
                lag = torch.fft.fftfreq(N, 1.0 / N).round().long()                    # signed lags 0, 1, ..., -1
                L2 = (2 * lag) % N
                h = g[L2][:, L2] * ((2 * lag.abs() < N // 2)[:, None] & (2 * lag.abs() < N // 2)[None, :]).double()
                h = h.clamp(min=0)
                Wf = self.wsplat.get(f, torch.zeros(N * N, dtype=torch.float64)).view(N, N).cpu()
                n2 = 2 * N
                hp = torch.zeros(n2, n2, dtype=torch.float64)
                hl = lag.clone()
                hp[hl[:, None] % n2, hl[None, :] % n2] = h
                q = torch.fft.irfft2(torch.fft.rfft2(Wf, s=(n2, n2)) * torch.fft.rfft2(hp), s=(n2, n2))[:N, :N].clamp(min=0)
                dq.append(q / max(float(torch.stack([q[p] for p in plist[f]]).mean()), 1e-300))
            d = torch.stack(dq)
        else:
            d = d_lin
        self.d_filter = d
        self.d_filter_linear = d_lin
        diag = torch.einsum("fhw,fk->khw", d, torch.diagonal(G0, dim1=-2, dim2=-1))  # (K, N, N): diagonal of the data part of M_j
        covc = self.cov.double().cpu()
        # the data footprint: nodes with a training pixel within one node (the weight splat, dilated). Kernel wings (to 10 arcsec) give a
        # small coverage to almost every node, so 'coverage > 0' is not the footprint; the prior weight mu (median node information / rho^2)
        # and the reference coverage are taken over the footprint (19 Nov, section 11; on the 4-arcsec lab box mu was 6e-10 otherwise)
        if hasattr(self, "wsplat") and self.wsplat:
            Wt = sum(w.view(N, N).cpu() for w in self.wsplat.values())
            foot = torch.nn.functional.max_pool2d((Wt > 0).double()[None, None], 3, 1, 1)[0, 0] > 0
        else:
            foot = covc > 0
        self.footprint = foot.to(dev)
        ref = torch.quantile(covc[foot & (covc > 0)], 0.9).item() if bool((foot & (covc > 0)).any()) else 1.0
        rel = (covc / max(ref, 1e-300)).clamp(min=0)
        self.covered = ((rel > 1e-3) & foot).to(dev)
        dmed = float(torch.stack([diag[k][self.covered.cpu()].median() for k in range(K)]).max())
        self.mu = dmed / rho ** 2 if mu is None else float(mu)
        mu_ = self.mu
        self.node_noise = 1.0 / math.sqrt(dmed)
        self.dA = (diag + mu_).float().to(dev)
        self.fmask = (rel > 0.05).to(dev)
        self.fscale = torch.where(self.fmask, rel.float().to(dev), torch.ones_like(rel, dtype=torch.float32, device=dev)).rsqrt()
        self.kappa = (diag[:, c, c] / float(covc[c, c]) if float(covc[c, c]) > 0 else diag[:, c, c] * 0).tolist()
        # regions by coverage pattern: k-means (k-means++ start, fixed seed) on theta_f(j)
        gf = torch.linalg.eigvalsh(G0)[:, -1].clamp(min=1e-300)
        dl = (mu_ / gf)[:, None, None]                                               # coverage at which filter f's information equals the prior
        th = torch.log(d + dl).reshape(nF, -1).T                                     # (N^2, F): log coverage, so distances are ratios
        gen = torch.Generator().manual_seed(12345)
        cent = [th.mean(0)]
        for _ in range(1, R_):
            dist = torch.stack([((th - c0) ** 2).sum(1) for c0 in cent]).min(0).values
            cent.append(th[int(torch.multinomial(dist / dist.sum(), 1, generator=gen))] if float(dist.sum()) > 0 else th[0])
        cent = torch.stack(cent)
        for _ in range(30):
            lab = torch.cdist(th, cent).argmin(1)
            cent = torch.stack([th[lab == r].mean(0) if bool((lab == r).any()) else cent[r] for r in range(R_)])
        lab = torch.cdist(th, cent).argmin(1).view(N, N)
        used = [r for r in range(R_) if bool((lab == r).any())]
        S1 = Smoother(N, 1.0, sm_nodes, dtype=torch.float64).M if sm_nodes > 0 else torch.eye(N, dtype=torch.float64)
        a = torch.stack([S1 @ (lab == r).double() @ S1.T for r in used])
        a = a / a.sum(0, keepdim=True).clamp(min=1e-300)
        self.region_label = lab
        self.region_sqrt_a = a.sqrt().float().to(dev)                               # (R, N, N)
        Mr_half, finv_r, empty, dbar_all = [], [], [], []
        for i, r in enumerate(used):
            m = (lab == r)
            dbar = (torch.exp(th.T.reshape(nF, N, N)[:, m].mean(1)) - dl[:, 0, 0]).clamp(min=0)   # geometric mean (in log coverage) per filter
            dbar_all.append(dbar.tolist())
            Mr = torch.einsum("f,fkl->kl", dbar, G0) + mu_ * torch.eye(K, dtype=torch.float64)
            ev, V = torch.linalg.eigh(0.5 * (Mr + Mr.T))
            Mr_half.append(((V * ev.clamp(min=0).sqrt()[None]) @ V.T).float())
            if float((dbar * gf).max()) < 1e-2 * mu_:                               # no data in this region (below 1 per cent of the prior): C_r = mu I
                empty.append(True)
                finv_r.append(None)
                continue
            empty.append(False)
            kr = torch.einsum("f,fklhw->klhw", dbar.float(), kernF)
            fh = torch.fft.fft2(kr).permute(2, 3, 0, 1)

            def _inv(b):
                b = 0.5 * (b + b.conj().transpose(-1, -2))
                ev, V = torch.linalg.eigh(b)
                ev = ev.clamp(min=0) + mu_
                return (V * (1.0 / ev)[..., None, :]) @ V.conj().transpose(-1, -2)
            finv_r.append(_eigh_rows(fh, _inv).to(dev))
            del fh, kr
        self.kernF = kernF                       # host: per-filter lag kernels (filter, K', K, N, N), kept for the restricted operator
        self.region_Mhalf = [m.to(dev) for m in Mr_half]
        self.region_finv = finv_r
        self.region_empty = empty
        self.region_dbar = dbar_all
        self.finv = next((x for x in finv_r if x is not None), None)                 # (for code that reads op.finv: a covered region's inverse)
        Mi = []
        for r0 in range(0, N, 32):                                                   # per-node M_j^{-1/2}, in blocks of rows, on the host
            Mj = torch.einsum("fhw,fkl->hwkl", d[:, r0:r0 + 32], G0) + mu_ * torch.eye(K, dtype=torch.float64)
            lam, Vj = torch.linalg.eigh(0.5 * (Mj + Mj.transpose(-1, -2)))
            Mi.append(((Vj * lam.clamp(min=mu_).rsqrt()[..., None, :]) @ Vj.transpose(-1, -2)).float())
        self.Minvh = torch.cat(Mi).to(dev)                                           # (N, N, K, K)
        names = (lambda f: self.model.fset.names[f]) if hasattr(self.model.fset, "names") else (lambda f: f)
        cen = {names(f): round(float(d[i, c, c]), 4) for i, f in enumerate(self.filters)}
        log(f"preconditioner 'filter': probe nodes (row, column) per filter {({names(f): p for f, p in probes.items()})} ({[len(plist[f]) for f in self.filters]} "
            f"probes per filter averaged, window +-{Rw} nodes); coverage of the box centre "
            f"relative to each filter's probe {cen} (0 = the filter does not reach the centre); {len(used)} coverage regions (nodes per region "
            f"{[int((lab == r).sum()) for r in used]}; mean relative coverage per filter "
            f"{[[round(x, 2) for x in db] for db in dbar_all]}; {sum(empty)} without data)")

    def _region_solve(self, i, u):
        """C_r^{-1} u for region i (u: (K, N, N))."""
        if self.region_empty[i]:
            return u / self.mu
        Uh = torch.fft.fft2(u).permute(1, 2, 0)
        return torch.fft.ifft2(torch.einsum("hwij,hwj->hwi", self.region_finv[i], Uh).permute(2, 0, 1)).real

    def local_fourier_sigma(self, phi, s):
        """sigma of the product at scale s with the 'filter' preconditioner taken as the approximation of F^{-1}, M_j and the region weights
        assumed constant over the smoothing kernel g_s around each output node x:
            sigma^2(x) = sum_r a_r(x) u_rx^T Q_rs u_rx,   u_rx = M_r^{1/2} M_x^{-1/2} phi,   Q_rs = N^-2 sum_w |g_s^(w)|^2 Re C_r^{-1}(w).
        Follows each filter's own coverage (gaps, edges, bright sources), unlike circulant_sigma of the 'fourier' mode. (N, N), NaN where the
        all-filter coverage is below 0.05 of its 90th percentile."""
        N, c = self.N, (self.N - 1) // 2
        sm = Smoother(N, self.model.sky.pitch, s, dtype=torch.float64)
        g = sm.weights_at(c, c)
        gh2 = torch.fft.fft2(torch.roll(g, (-c, -c), (0, 1))).abs() ** 2             # (N, N)
        ph = phi.double().cpu()
        if not hasattr(self, "_Minvh_host"):
            self._Minvh_host = self.Minvh.cpu().double()
            self._Qcache = {}
        w = torch.einsum("hwkl,l->hwk", self._Minvh_host, ph)                         # M_x^{-1/2} phi
        var = torch.zeros(N, N, dtype=torch.float64)
        for i in range(len(self.region_Mhalf)):
            key = (i, round(float(s), 9))
            if key in self._Qcache:
                Q = self._Qcache[key]
            elif self.region_empty[i]:
                Q = float(gh2.sum()) / N ** 2 / self.mu * torch.eye(self.K, dtype=torch.float64)
            else:
                fi = self.region_finv[i]
                g2 = gh2.to(device=fi.device, dtype=torch.float32).reshape(-1)
                Q = (g2.to(torch.complex64) @ fi.reshape(N * N, -1)).to(torch.complex128).real.cpu().view(self.K, self.K) / N ** 2
            self._Qcache[key] = Q
            Mh = self.region_Mhalf[i].cpu().double()
            u = torch.einsum("kl,hwl->hwk", Mh, w)
            var += self.region_sqrt_a[i].cpu().double() ** 2 * torch.einsum("hwk,kl,hwl->hw", u, Q, u)
        sig = var.clamp(min=0).sqrt()
        return torch.where(self.fmask.cpu(), sig, torch.full_like(sig, float("nan")))

    def _lb(self, t, pix):
        """Background basis of a tile (Legendre terms times the exposure gain), no gradient."""
        with torch.no_grad():
            Lb = legendre6(pix["xi"])
            G = self.model.gain_of(t.eid, pix)
            return Lb * G[:, None] if G is not None else Lb

    # ---------------------------------------------------------------- fast approximation (no solve)
    def _circ_q(self, phi):
        """q(freq) = sum over the eigenmodes of the Fourier model of |<v, phi>|^2 / (lambda + mu), double precision, in blocks of rows; cached
        per band vector (the eigendecomposition does not depend on the scale)."""
        key = tuple(float(x) for x in phi.double().cpu().tolist())
        if key not in self._qcache:
            N, c = self.N, (self.N - 1) // 2
            fh = torch.fft.fft2(torch.roll(self.fc.double(), (-c, -c), (-2, -1))).permute(2, 3, 0, 1)
            ph = phi.double().to(fh.device)

            def _q(b):
                b = 0.5 * (b + b.conj().transpose(-1, -2))
                ev, V = torch.linalg.eigh(b)
                lam = ev.clamp(min=0) + self.mu
                proj = (V.conj().transpose(-1, -2) @ ph.to(device=V.device, dtype=V.dtype)).abs() ** 2
                return (proj / lam).sum(-1)
            self._qcache[key] = _eigh_rows(fh, _q)
        return self._qcache[key]

    def circulant_sigma(self, phi, s):
        """sigma of the product at scale s from the Fourier model of F alone (uniform-coverage approximation): sigma_c(s) at the centre's
        coverage, scaled by 1 / sqrt(relative coverage) elsewhere. Costs nothing beyond the preconditioner's K + 1 applications; ignores the
        background terms, the edges and coverage gradients. Returns (N, N) (NaN where the relative coverage is below 0.05).
        With precond_mode 'filter' this is local_fourier_sigma (each filter's own coverage)."""
        if self.setup_mode == "filter":
            return self.local_fourier_sigma(phi, s)
        N, c = self.N, (self.N - 1) // 2
        q = self._circ_q(phi)
        g = Smoother(N, self.model.sky.pitch, s, dtype=torch.float64, device=q.device).weights_at(c, c)
        gh = torch.fft.fft2(torch.roll(g, (-c, -c), (0, 1)))
        sc = math.sqrt(float((gh.abs() ** 2 * q).sum() / N ** 2))
        rel = (self.cov / self.cov[c, c]).double()
        return torch.where(self.fmask, sc / rel.clamp(min=1e-12).sqrt(), torch.full_like(rel, float("nan"))).cpu()

    # ---------------------------------------------------------------- pieces
    def gauge(self, xb):
        """Project the background part onto the gauge (per filter, sum over training rows of b_e0, or of every term, = 0). With the orthonormal
        basis: the exact orthogonal projector onto the gauge constraints and the kept directions."""
        if getattr(self, "bkg_basis", "raw") == "orth":
            x = xb.reshape(-1, 6)
            out = x * self.bkg_keep.to(x.dtype)
            for rows, Pg in self.bkg_proj:
                out[rows] = (Pg.to(x.dtype) @ x[rows].reshape(-1)).view(-1, 6)
            return out.reshape(-1)
        xb = xb.view(-1, 6).clone()
        cols = slice(0, 6) if self.bkg_gauge == "all" else slice(0, 1)
        for g in self.gauge_groups:
            xb[g, cols] -= xb[g, cols].mean(0)
        return xb.reshape(-1)

    def _render_sum(self, xA, rfun, want_gb=False):
        """One application: forward render of every tile with sky coefficients xA (requires grad), batch by batch, and backward of
        sum_i r_i S_i with r = rfun(tile, S detached, Lb). Each batch's graph is freed by its own backward and its tensors are deleted before the
        next batch. Returns J_A^T r (K, N, N) and, if want_gb, L^T r per training row (n_rows, 6) accumulated per batch."""
        model = self.model
        t0 = time.perf_counter()
        mixed = self.precision == "mixed"
        xA = xA.detach().float().requires_grad_(True)
        gb = torch.zeros(len(self.rows), 6, device=self.dev, dtype=torch.float64 if mixed else torch.float32) if want_gb else None
        acc = torch.zeros(xA.shape, dtype=torch.float64, device=xA.device) if mixed else None
        with torch.enable_grad(), frozen(model), sky_override(model.sky, xA):
            for ch in self._batches():
                pixs = [t["pix"] for t in ch]
                outs = model.render_batch([(t.eid, p) for t, p in zip(ch, pixs)], self.P)
                Ss, Rs = [], []
                for t, p, o in zip(ch, pixs, outs):
                    Lb = self._lb(t, p)
                    r = rfun(t, o["S"].detach(), Lb)
                    Ss.append(o["S"])
                    Rs.append(r)
                    if want_gb:
                        gb[t.row] += (Lb.double().T @ r.double()) if mixed else (Lb.T @ r)
                if mixed:                         # this batch's adjoint alone (float32), added to a float64 sum
                    acc += torch.autograd.grad(Ss, xA, Rs)[0].double()
                else:
                    torch.autograd.backward(Ss, Rs)
                del outs, Ss, Rs, pixs
        self.n_apply += 1
        self.t_last = time.perf_counter() - t0
        self.t_apply += self.t_last
        if mixed:
            return acc, gb
        g = xA.grad if xA.grad is not None else torch.zeros_like(xA)
        return g.detach(), gb

    def _JT_w(self):
        g, _ = self._render_sum(torch.zeros(self.K, self.N, self.N, device=self.dev), lambda t, S, Lb: t.w)
        return g.float()

    def _data_normal(self, xA_flat, xb_flat):
        xA = xA_flat.view(self.K, self.N, self.N)
        xb = self.bkg_to_raw(xb_flat).float()
        gA, gb = self._render_sum(xA, lambda t, S, Lb: t.w * (S + Lb @ xb[t.row]), want_gb=True)
        return gA.reshape(-1), self._bkg_grad(gb).reshape(-1)

    # ---------------------------------------------------------------- operator, preconditioner, right-hand sides
    def apply(self, x):
        xA, xb = x[:self.nA], self.gauge(x[self.nA:])
        gA, gb = self._data_normal(xA, xb)
        return torch.cat([gA.to(x.dtype) + self.mu * xA, self.gauge(gb.to(x.dtype))])

    def precond(self, r):
        rA, rb = r[:self.nA], r[self.nA:].view(-1, 6)
        if self.bkg_basis == "orth":            # the background block is the identity in the orthonormal basis: exact inverse on the gauge subspace
            zb = rb.reshape(-1)
        else:
            zb = torch.einsum("rij,rj->ri", self.Bb_inv, rb.to(self.Bb_inv.dtype)).reshape(-1).to(r.dtype)
        if self.precond_mode == "bjacobi":      # per-node K x K block Jacobi M_j^{-1} from the 'filter' set-up's local information
            R = rA.view(self.K, self.N, self.N).to(self.Minvh.dtype)
            zA = torch.einsum("hwkl,lhw->khw", self.Minvh, torch.einsum("hwkl,lhw->khw", self.Minvh, R))
            return torch.cat([zA.reshape(-1).to(r.dtype), self.gauge(zb)])
        if self.precond_mode == "none":          # no preconditioner on the sky block (the background blocks keep their exact inverses)
            return torch.cat([rA, self.gauge(zb)])
        if self.precond_mode == "jacobi":
            return torch.cat([rA / self.dA.reshape(-1).to(r.dtype), self.gauge(zb)])
        R = rA.view(self.K, self.N, self.N).float()
        if self.precond_mode == "filter":
            y = torch.einsum("hwkl,lhw->khw", self.Minvh, R)
            z = torch.zeros_like(y)
            for i, Mh in enumerate(self.region_Mhalf):
                sa = self.region_sqrt_a[i]
                v = self._region_solve(i, torch.einsum("kl,lhw->khw", Mh, sa * y))
                z += sa * torch.einsum("kl,lhw->khw", Mh, v)
            zA = torch.einsum("hwkl,lhw->khw", self.Minvh, z)
            return torch.cat([zA.reshape(-1).to(r.dtype), self.gauge(zb)])
        u = torch.where(self.fmask, R * self.fscale, torch.zeros_like(R))
        Uh = torch.fft.fft2(u).permute(1, 2, 0)                                   # (N, N, K)
        z = torch.fft.ifft2(torch.einsum("hwij,hwj->hwi", self.finv, Uh).permute(2, 0, 1)).real
        zA = torch.where(self.fmask, z * self.fscale, R / self.dA)
        return torch.cat([zA.reshape(-1).to(r.dtype), self.gauge(zb)])

    def sample_rhs(self, gen):
        """[J_A L]^T W^{1/2} z + [sqrt(mu) z', 0] (z, z' standard normal from generator gen; z is drawn tile by tile, in tile order, on the host)."""
        gA, gb = self._render_sum(torch.zeros(self.K, self.N, self.N, device=self.dev),
                                  lambda t, S, Lb: t.w.sqrt() * torch.randn((t.n,), generator=gen).to(self.dev), want_gb=True)
        zA = torch.randn(self.nA, generator=gen).to(self.dev)
        return torch.cat([gA.reshape(-1) + math.sqrt(self.mu) * zA.to(gA.dtype), self.gauge(self._bkg_grad(gb).reshape(-1))])

    def residual_rhs(self):
        """Gradient of the log posterior at the fit: [J_A^T W r - mu A_fit, L^T W r] (gauge-projected)."""
        gA, gb = self._render_sum(torch.zeros(self.K, self.N, self.N, device=self.dev), lambda t, S, Lb: t.wr, want_gb=True)
        return torch.cat([gA.reshape(-1) - self.mu * self.A_fit.reshape(-1).to(gA.dtype), self.gauge(self._bkg_grad(gb).reshape(-1))])

    def unit_rhs(self, kA):
        """Right-hand side [k (A part), 0] for an exact variance solve."""
        return torch.cat([kA.reshape(-1).to(self.dev, self.vdtype), torch.zeros(self.nb, device=self.dev, dtype=self.vdtype)])


def pcg(op, b, tol=1e-3, maxiter=300, x0=None, snapshots=None, monitor=None, label="solve", keep_every=0, keep_vectors=False):
    """Preconditioned conjugate gradients for op.apply (symmetric positive definite on the gauge subspace). Returns x, info.
    monitor (optional): progress line every few iterations and the time budget; when it asks to stop, the iteration ends cleanly before the next
    application and info['truncated'] is True (x is the last iterate; info['rel_residual'] the residual reached)."""
    with torch.no_grad():
        x = torch.zeros_like(b) if x0 is None else x0.clone()
        r = b - op.apply(x) if x0 is not None else b.clone()
        z = op.precond(r)
        p = z.clone()
        rz = torch.dot(r, z)
        bn = float(b.norm())
        hist, alphas, betas, rzs = [], [], [], []
        x_mark, it_mark, x_mark_prev, it_mark_prev = None, 0, None, 0
        breakdown = ""
        lvecs = []
        if not (math.isfinite(float(rz)) and float(rz) > 0):
            breakdown = f"r.P^-1 r = {float(rz):.3e} at the start (the preconditioner as applied is not positive)"
            maxiter = 0
        it, truncated = 0, False
        if monitor is not None:
            monitor.residual = 1.0
        for it in range(1, maxiter + 1):
            if monitor is not None and monitor.stop_now(label):
                truncated, it = True, it - 1
                break
            Ap = op.apply(p)
            pAp = torch.dot(p, Ap)
            if not (math.isfinite(float(pAp)) and float(pAp) > 0):      # p.Fp <= 0: impossible for a positive operator; stop, say so
                breakdown = f"p.Fp = {float(pAp):.3e} at iteration {it} (the operator as applied is not positive on the search direction)"
                it -= 1
                break
            a = rz / pAp
            if keep_vectors:                          # Lanczos vector (-1)^j z_j / sqrt(r_j.z_j), on the host (for Ritz vectors)
                lvecs.append((((-1) ** (it - 1)) * z / math.sqrt(float(rz))).double().cpu())
            alphas.append(float(a))
            rzs.append(float(rz))
            x += a * p
            r -= a * Ap
            rel = float(r.norm()) / max(bn, 1e-30)
            hist.append(rel)
            if monitor is not None:
                monitor.report(label, it, rel)
            if snapshots is not None and it in snapshots:
                snapshots[it] = x.clone()
            if keep_every and it % keep_every == 0:          # marks every keep_every iterations (two copies held: the last two marks)
                x_mark_prev, it_mark_prev = x_mark, it_mark
                x_mark, it_mark = x.clone(), it
            if rel < tol:
                break
            z = op.precond(r)
            rz_new = torch.dot(r, z)
            if not (math.isfinite(float(rz_new)) and float(rz_new) > 0):  # r.P^{-1}r <= 0: the preconditioner is not positive
                breakdown = f"r.P^-1 r = {float(rz_new):.3e} at iteration {it} (the preconditioner as applied is not positive)"
                break
            betas.append(float(rz_new / rz))
            p = z + (rz_new / rz) * p
            rz = rz_new
    ritz = lanczos_ritz(alphas, betas)
    # error of the iterate in the operator's own metric, ||x* - x||_F (CG minimises it): Hestenes-Stiefel / Strakos-Tichy estimate with a delay
    # of d iterations, ||x* - x_{m-d}||_F^2 >= sum_{j=m-d}^{m-1} alpha_j r_j.z_j; for a converging solve it approximates (from below) the error d
    # iterations before the end, which bounds the final error (the F-norm error decreases monotonically). Relative to ||x||_F ~ sqrt(x.b).
    d = min(10, len(alphas) - 1)
    errF = math.sqrt(max(sum(alphas[j] * rzs[j] for j in range(len(alphas) - d, len(alphas))), 0.0)) if d >= 1 else float("nan")
    xb = abs(float(torch.dot(x, b))) if x0 is None else float("nan")
    extra = {}
    if keep_every:                                            # an iterate keep_every / 2 to 3 keep_every / 2 iterations before the last one
        ref, ref_it = (x_mark, it_mark) if (x_mark is not None and it - it_mark >= max(1, keep_every // 2)) else (x_mark_prev, it_mark_prev)
        if ref is not None and 0 < ref_it < it:
            extra = {"x_back": ref, "iter_back": ref_it}
    return x, {**extra, "iterations": it, "rel_residual": hist[-1] if hist else 1.0, "history": hist[::10], "truncated": truncated,
               "ritz_min": ritz[0] if ritz else float("nan"), "ritz_max": ritz[-1] if ritz else float("nan"),
               "err_F_est": errF, "rel_err_F_est": errF / math.sqrt(xb) if xb > 0 else float("nan"), "breakdown": breakdown,
               "history_all": hist, **({"lanczos_vectors": lvecs, "alphas": alphas, "betas": betas} if keep_vectors else {})}


class OperatorCheckError(RuntimeError):
    pass


def band_synthesis(Phi_band, smooth=1e-3):
    """E (K, F): coefficients from band values, reproducing every band exactly (Phi_band E = I) with the smoothest spectrum (least second
    difference across k): E = W Phi^T (Phi W Phi^T)^{-1}, W = (D^T D + smooth I)^{-1}."""
    Phi = Phi_band.double().cpu()
    K = Phi.shape[1]
    D = torch.zeros(K - 2, K, dtype=torch.float64)
    for i in range(K - 2):
        D[i, i:i + 3] = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
    W = torch.linalg.inv(D.T @ D + smooth * torch.eye(K, dtype=torch.float64))
    return W @ Phi.T @ torch.linalg.inv(Phi @ W @ Phi.T)


class RestrictedOperator:
    """The normal operator restricted to a sky at a stated resolution in the observed bands (19 Nov, second part of the report).

    Unknowns y = (band images on a coarse node grid of pitch q = factor x the sky pitch, the background terms). The sky coefficients are
        x = B y,   B = E (x) B1 (x) B1,
    E (K x F) the band synthesis (band_synthesis: exact band reproduction, smoothest spectrum), B1 (N x Nc) the coarse tent of half-width q
    (optionally convolved with a Gaussian of width s_extra), a partition of unity. The operator is B^T F B (the prior becomes mu B^T B), solved
    in y. Directions that F leaves to the prior (spatial frequencies above the coarse grid's, the spectral directions no filter reads) are not
    unknowns any more: they stay at the fit's values. Preconditioner: G^{-1} B^T P^{-1} B G^{-1}, G = B^T B (Kronecker, exact inverse),
    P^{-1} the base operator's (exact if range(B) is invariant under P). Everything else (rows, gauge, A_fit, the cheap sigma, timings) is the
    base operator's."""

    def __init__(self, op, factor=2, s_extra=0.0, bands=None, log=print, precond_kind=None):
        self.base = op
        model = op.model
        dev = op.dev
        self.vdtype = torch.float64
        self.factor = int(factor)
        bands = sorted(set(op.filters)) if bands is None else list(bands)
        self.bands = bands
        self.E = band_synthesis(model.Phi_band.detach()[bands]).to(dev)               # (K, F)
        N, p = op.N, model.sky.pitch
        Nc = int(math.ceil((N - 1) / self.factor)) + 1
        x = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * p
        z = (torch.arange(Nc, dtype=torch.float64) - (Nc - 1) / 2) * self.factor * p
        self.B1 = tent_gauss_1d(x, z, self.factor * p, s_extra).to(dev)               # (N, Nc)
        G1 = self.B1.T @ self.B1
        self.G1inv = torch.linalg.inv(G1)
        self.GEinv = torch.linalg.inv(self.E.T @ self.E)
        self.F, self.Nc = len(bands), Nc
        self.K, self.N = self.F, Nc                                                    # shape of the unknown sky part (bands, Nc, Nc)
        self.nA, self.nb = self.F * Nc * Nc, op.nb
        self.precision = "mixed"
        self.pitch_restricted = self.factor * p
        self.s_extra = float(s_extra)
        if precond_kind is None:
            precond_kind = "schwarz" if getattr(op, "kernF", None) is not None and s_extra == 0 else "projected"
        self.precond_kind = precond_kind
        self.precond_mode = f"restricted/{precond_kind}"
        if precond_kind == "schwarz":
            self._setup_schwarz(log)
        log(f"restricted operator: band images of {self.F} filters on {Nc} x {Nc} nodes at {self.factor * p:.4f} arcsec "
            f"({self.nA} unknowns instead of {op.nA}); preconditioner G^-1 B^T P^-1 B G^-1 with the base '{op.precond_mode}'")

    def __getattr__(self, k):                     # everything not defined here is the base operator's
        if k == "base":
            raise AttributeError(k)
        return getattr(self.base, k)

    def lift_A(self, yA):
        """x = B y: (F Nc Nc,) -> (K, N, N), float64."""
        y = yA.reshape(self.F, self.Nc, self.Nc).double()
        return torch.einsum("kb,bmn->kmn", self.E, self.B1 @ y @ self.B1.T)

    def restrict_A(self, xA):
        """B^T x: (K N N,) -> (F Nc Nc,)."""
        X = xA.reshape(self.base.K, self.base.N, self.base.N).double()
        return (self.B1.T @ torch.einsum("kb,kmn->bmn", self.E, X) @ self.B1).reshape(-1)

    def _Ginv(self, yA):
        y = yA.reshape(self.F, self.Nc, self.Nc)
        return torch.einsum("ab,bmn->amn", self.GEinv, self.G1inv @ y @ self.G1inv).reshape(-1)

    def _full(self, xA, yb):
        return torch.cat([xA.reshape(-1).to(self.base.vdtype), yb.to(self.base.vdtype)])

    def apply(self, y):
        out = self.base.apply(self._full(self.lift_A(y[:self.nA]), y[self.nA:]))
        return torch.cat([self.restrict_A(out[:self.base.nA]), out[self.base.nA:].double()])

    def _setup_schwarz(self, log):
        """The 'filter' preconditioner rebuilt for B^T F B on the coarse grid, from the base operator's per-filter probe kernels (no application):
        on a coarse grid of pitch q, B^T F B of a translation-invariant F is translation invariant by one coarse node, with kernel
            Gy_f(D) = E^T [ (G_f * c2)(q D) ] E,   c2 = the autocorrelation of the fine-grid tent profile of B1 (separable),
        and the prior mu B^T B has kernel mu E^T E c2(q D). Regions, their weights and the per-filter coverage are the base's, read at the fine
        nodes under the coarse ones; P_y^{-1} = M^{-1/2} [sum_r a_r^{1/2} M_r^{1/2} C_r^{-1} M_r^{1/2} a_r^{1/2}] M^{-1/2} as in the base."""
        op, q, N, Nc, F = self.base, self.factor, self.base.N, self.Nc, self.F
        E = self.E.double().cpu()
        prof = self.B1[:, Nc // 2].double().cpu()                                    # the tent profile of an interior coarse node on the fine grid
        b = prof[prof > 1e-12]                                                       # (aligned for odd N: q = 2 -> 0.5, 1, 0.5; offset for even N)
        nb_ = len(b)
        cv = {v: float((b[max(0, -v):nb_ - max(0, v)] * b[max(0, v):nb_ - max(0, -v)]).sum()) for v in range(-nb_ + 1, nb_)}
        H = (N - 1) // 2 // q
        lagD = torch.arange(-H, H + 1)
        fine_idx = (q * lagD) % N
        coarse_idx = lagD % Nc

        def to_coarse(G):                                                            # (a, b, N, N) lag kernel -> (a, b, Nc, Nc)
            Gc = torch.zeros_like(G)
            for v, w in cv.items():                                                  # rows
                Gc += w * torch.roll(G, shifts=v, dims=-2)
            G2 = torch.zeros_like(G)
            for v, w in cv.items():                                                  # columns
                G2 += w * torch.roll(Gc, shifts=v, dims=-1)
            out = torch.zeros(*G.shape[:-2], Nc, Nc, dtype=G.dtype)
            out[..., coarse_idx[:, None], coarse_idx[None, :]] = G2[..., fine_idx[:, None], fine_idx[None, :]]
            return out
        Gy = torch.stack([to_coarse(torch.einsum("ka,klhw,lb->abhw", E, op.kernF[i].double(), E)) for i in range(len(op.filters))])   # (Ff, F, F, Nc, Nc)
        G0y = Gy[..., 0, 0]                                                          # (Ff, F, F)
        c00 = cv[0] ** 2
        # prior mu B^T B: symbol mu (E^T E) chat(w_y) chat(w_x), chat the coarse-lag spectrum of the tent autocorrelation (>= cv[0] - 2 cv[q] > 0)
        wq = 2 * math.pi * torch.fft.fftfreq(Nc, dtype=torch.float64)
        chat = cv[0] + sum(2 * cv.get(q * D_, 0.0) * torch.cos(D_ * wq) for D_ in range(1, nb_ // q + 2))
        prior_sym = op.mu * chat[:, None] * chat[None, :]                               # (Nc, Nc) scalar part
        ev_e, V_e = torch.linalg.eigh(E.T @ E)
        EtE_mhalf = (V_e * ev_e.rsqrt()[None]) @ V_e.T                                  # (E^T E)^{-1/2}
        # coarse node m sits over fine node j = (N-1)/2 + (m - (Nc-1)/2) q (nearest)
        jj = torch.clamp(torch.round((N - 1) / 2 + (torch.arange(Nc, dtype=torch.float64) - (Nc - 1) / 2) * q).long(), 0, N - 1)
        d = op.d_filter[:, jj][:, :, jj].double()                                    # (Ff, Nc, Nc)
        a = op.region_sqrt_a.cpu().double()[:, jj][:, :, jj] ** 2
        a = a / a.sum(0, keepdim=True).clamp(min=1e-300)
        self.ry_sqrt_a = a.sqrt().to(op.dev)
        mu0 = op.mu * (E.T @ E) * c00
        Mh, finv = [], []
        for i, dbar in enumerate(op.region_dbar):
            dbar = torch.tensor(dbar, dtype=torch.float64)
            Mr = torch.einsum("f,fab->ab", dbar, G0y) + mu0
            ev, V = torch.linalg.eigh(0.5 * (Mr + Mr.T))
            Mh.append(((V * ev.clamp(min=0).sqrt()[None]) @ V.T).to(op.dev))
            kr = torch.einsum("f,fabhw->abhw", dbar, Gy)                             # data part only
            fh = torch.fft.fft2(kr).permute(2, 3, 0, 1)
            fh = 0.5 * (fh + fh.conj().transpose(-1, -2))
            # C = Pr^{1/2} (I + S) Pr^{1/2}, S = Pr^{-1/2} D Pr^{-1/2} with its eigenvalues clamped at 0 (a one-probe data symbol can dip below 0)
            Wm = EtE_mhalf.to(fh.dtype)[None, None] / prior_sym.sqrt()[..., None, None]          # Pr^{-1/2} per frequency
            Sm = Wm @ fh @ Wm
            ev, V = torch.linalg.eigh(0.5 * (Sm + Sm.conj().transpose(-1, -2)))
            inner_inv = (V * (1.0 / (1.0 + ev.clamp(min=0)))[..., None, :]) @ V.conj().transpose(-1, -2)
            finv.append((Wm @ inner_inv @ Wm).to(op.dev))
        Mj = torch.einsum("fhw,fab->hwab", d, G0y) + mu0
        lam, Vj = torch.linalg.eigh(0.5 * (Mj + Mj.transpose(-1, -2)))
        self.ry_Minvh = ((Vj * lam.clamp(min=float(op.mu * c00 * 1e-6)).rsqrt()[..., None, :]) @ Vj.transpose(-1, -2)).to(op.dev)
        self.ry_Mhalf, self.ry_finv = Mh, finv
        log(f"restricted preconditioner (Schwarz, from the base's probe kernels, no application): {len(finv)} regions on {Nc} x {Nc} coarse nodes")

    def circulant_sigma(self, phi, s):
        """Cheap sigma of the RESTRICTED estimator (Schwarz kind): sigma^2 = k_y^T P_y^{-1} k_y with k_y = B^T k_x = (E^T phi) (x) B1^T g_s (for a
        band, E^T phi_f = e_f exactly), M and the region weights taken as constant over the kernel:
            sigma^2(m) = sum_r a_r(m) u_rm^T Q_r u_rm,  u_rm = M_r^{1/2} M_m^{-1/2} E^T phi,  Q_r = Nc^-2 sum_w |k^(w)|^2 Re C_r^{-1}(w),
        on the coarse nodes, read at the nearest coarse node for every fine node. Other kinds: the base operator's (full-space) Fourier sigma."""
        if self.precond_kind != "schwarz":
            return self.base.circulant_sigma(phi, s)
        N, Nc, q = self.base.N, self.Nc, self.factor
        c = (N - 1) // 2
        g1 = Smoother(N, self.base.model.sky.pitch, s, dtype=torch.float64).M[c]          # (N,) fine weights of the centre output node
        w1 = self.B1.double().cpu().T @ g1                                              # (Nc,) on the coarse nodes
        mc = int(torch.argmax(w1))
        k2 = torch.roll(w1[:, None] * w1[None, :], shifts=(-mc, -mc), dims=(0, 1))
        kh2 = torch.fft.fft2(k2).abs() ** 2
        ph = (self.E.double().cpu().T @ phi.double().cpu())                            # (F,)
        w = torch.einsum("hwab,b->hwa", self.ry_Minvh.cpu().double(), ph)
        var = torch.zeros(Nc, Nc, dtype=torch.float64)
        for i, Mh in enumerate(self.ry_Mhalf):
            Q = torch.einsum("hw,hwij->ij", kh2.to(torch.complex128), self.ry_finv[i].cpu().to(torch.complex128)).real / Nc ** 2
            u = torch.einsum("ab,hwb->hwa", Mh.cpu().double(), w)
            var += self.ry_sqrt_a[i].cpu().double() ** 2 * torch.einsum("hwa,ab,hwb->hw", u, Q, u)
        m_of = torch.clamp(torch.round((torch.arange(N, dtype=torch.float64) - (N - 1) / 2) / q + (Nc - 1) / 2).long(), 0, Nc - 1)
        sig = var.clamp(min=0).sqrt()[m_of][:, m_of]
        return torch.where(self.base.fmask.cpu(), sig, torch.full_like(sig, float("nan")))

    def precond(self, r):
        if self.precond_kind == "schwarz":
            zb = self.base.precond(torch.cat([torch.zeros(self.base.nA, device=self.base.dev, dtype=self.base.vdtype),
                                              r[self.nA:].to(self.base.vdtype)]))[self.base.nA:].double()
            R = r[:self.nA].reshape(self.F, self.Nc, self.Nc).double()
            y = torch.einsum("hwab,bhw->ahw", self.ry_Minvh, R)
            z = torch.zeros_like(y)
            for i, Mh in enumerate(self.ry_Mhalf):
                sa = self.ry_sqrt_a[i]
                v = torch.einsum("ab,bhw->ahw", Mh, sa * y)
                Vh = torch.fft.fft2(v).permute(1, 2, 0)
                v = torch.fft.ifft2(torch.einsum("hwij,hwj->hwi", self.ry_finv[i], Vh).permute(2, 0, 1)).real
                z += sa * torch.einsum("ab,bhw->ahw", Mh, v)
            zA = torch.einsum("hwab,bhw->ahw", self.ry_Minvh, z)
            return torch.cat([zA.reshape(-1), zb])
        z = self.base.precond(self._full(self.lift_A(self._Ginv(r[:self.nA])), r[self.nA:]))
        return torch.cat([self._Ginv(self.restrict_A(z[:self.base.nA])), z[self.base.nA:].double()])

    def gauge(self, xb):
        return self.base.gauge(xb)

    def _map(self, v):
        return torch.cat([self.restrict_A(v[:self.base.nA]), v[self.base.nA:].double()])

    def residual_rhs(self):
        return self._map(self.base.residual_rhs())

    def sample_rhs(self, gen):
        return self._map(self.base.sample_rhs(gen))

    def unit_rhs(self, kA):
        return torch.cat([self.restrict_A(kA.reshape(-1).to(self.base.dev)), torch.zeros(self.nb, device=self.base.dev, dtype=torch.float64)])


def operator_checks(op, seed=7, log=print, sym_tol=1e-4, null_tol=0.5, raise_on_fail=False, label=""):
    """Checks of the operator F and the preconditioner P^{-1} on the actual problem (19 Nov, second part of the report). Costs 4 applications.
      * symmetry: |u.Fv - v.Fu| / (|u| |Fv|) and the same for P^{-1}, u, v random (gauge-projected);
      * positivity: u.Fu / u.u and u.P^{-1}u / u.u for random u;
      * the prior-only direction: u in the spectral null space of the filters (every node's coefficient vector orthogonal to all rows of Phi, the
        directions no band reads), random in space. Exactly, J u is then small and u.Fu >= mu u.u; the ratio q_null = u.Fu / (mu u.u) must be
        >= 1 up to the data's small share. q_null < null_tol (or negative) means the operator as applied cannot resolve the prior against the
        data curvature (round-off of the large components swamps mu): CG on it will break down;
      * background only: u.Fu for a random background vector (the 6 x 6 blocks), and the blocks' condition numbers.
    Prints one line; returns a dict with 'failed' (list of reasons). raise_on_fail: raise OperatorCheckError with the reasons."""
    g = torch.Generator().manual_seed(seed)
    dt = getattr(op, "vdtype", torch.float32)
    dev = op.dev

    def rv():
        u = torch.randn(op.nA + op.nb, generator=g, dtype=torch.float64).to(dev, dt)
        return torch.cat([u[:op.nA], op.gauge(u[op.nA:])])
    u, v = rv(), rv()
    Fu, Fv = op.apply(u), op.apply(v)
    Pu, Pv = op.precond(u), op.precond(v)
    out = {"F_symmetry": abs(float(u @ Fv - v @ Fu)) / max(float(u.norm() * Fv.norm()), 1e-300),
           "P_symmetry": abs(float(u @ Pv - v @ Pu)) / max(float(u.norm() * Pv.norm()), 1e-300),
           "F_positive": min(float(u @ Fu), float(v @ Fv)) / float(u @ u), "P_positive": min(float(u @ Pu), float(v @ Pv)) / float(u @ u)}
    # spectral null space of the filters (base operator only: a restricted operator has no such directions)
    base = getattr(op, "base", None)
    if base is None and hasattr(op, "model"):
        Phi = op.model.Phi_band.detach().double().cpu()
        Q, _ = torch.linalg.qr(Phi.T)                                   # (K, F): the read directions
        Pn = torch.eye(op.K, dtype=torch.float64) - Q @ Q.T
        n = torch.randn(op.K, op.N, op.N, generator=g, dtype=torch.float64)
        n = torch.einsum("kl,lhw->khw", Pn, n)
        un = torch.cat([n.reshape(-1).to(dev, dt), torch.zeros(op.nb, device=dev, dtype=dt)])
        Fn = op.apply(un)
        out["q_null"] = float(un @ Fn) / (op.mu * float(un @ un))
        out["P_null"] = float(un @ op.precond(un)) * op.mu / float(un @ un)        # ~1 when P^{-1} gives the prior's 1/mu there
    ub = torch.zeros(op.nA + op.nb, device=dev, dtype=dt)
    ub[op.nA:] = op.gauge(torch.randn(op.nb, generator=g, dtype=torch.float64).to(dev, dt))
    out["F_background_positive"] = float(ub @ op.apply(ub)) / max(float(ub @ ub), 1e-300)
    bc = getattr(op, "bb_cond", None) or getattr(base, "bb_cond", None) or []
    if bc:
        out["background_block_condition_max"] = float(max(bc))
        out["background_blocks_above_1e7"] = int(sum(c > 1e7 for c in bc))
    fails = []
    if not out["F_symmetry"] < sym_tol:
        fails.append(f"operator not symmetric ({out['F_symmetry']:.2e} > {sym_tol:g})")
    if not out["P_symmetry"] < sym_tol:
        fails.append(f"preconditioner not symmetric ({out['P_symmetry']:.2e} > {sym_tol:g})")
    if not out["F_positive"] > 0:
        fails.append(f"operator not positive on a random vector (u.Fu/u.u = {out['F_positive']:.3e})")
    if not out["P_positive"] > 0:
        fails.append(f"preconditioner not positive on a random vector ({out['P_positive']:.3e})")
    if "q_null" in out and not out["q_null"] > null_tol:
        fails.append(f"the operator does not resolve the prior in the filters' spectral null space (u.Fu / (mu u.u) = {out['q_null']:.3e}, "
                     f"exactly >= 1): round-off of the data curvature swamps mu")
    if not out["F_background_positive"] > 0:
        fails.append(f"background block not positive ({out['F_background_positive']:.3e})")
    out["failed"] = fails
    log(f"operator checks{(' [' + label + ']') if label else ''} (precision {getattr(op, 'precision', getattr(base, 'precision', '?'))}): "
        + ", ".join(f"{k} {v:.3e}" if isinstance(v, float) else f"{k} {v}" for k, v in out.items() if k != "failed")
        + ("; ALL PASSED" if not fails else "; FAILED: " + "; ".join(fails)))
    if fails and raise_on_fail:
        raise OperatorCheckError("; ".join(fails))
    return out


def ritz_vectors(info, which=(-1, 0)):
    """Ritz pairs of P^{-1} F from a pcg run with keep_vectors=True: [(value, vector)] for the indices `which` of the ascending Ritz values."""
    al, be, V = info["alphas"], info["betas"], info["lanczos_vectors"]
    m = min(len(al), len(V))
    T = np.zeros((m, m))
    for j in range(m):
        T[j, j] = 1.0 / al[j] + (be[j - 1] / al[j - 1] if j > 0 else 0.0)
        if j + 1 < m and j < len(be):
            T[j, j + 1] = T[j + 1, j] = math.sqrt(max(be[j], 0.0)) / al[j]
    ev, S = np.linalg.eigh(T)
    Vm = torch.stack(V[:m])
    return [(float(ev[i]), (torch.tensor(S[:, i], dtype=torch.float64)[:, None] * Vm).sum(0)) for i in which]


def mode_report(op, y):
    """Where a vector lives: share of its squared norm in the sky block and the background block; the background share per exposure (the
    largest three) and per term; the sky share per spectral coefficient (or band, restricted)."""
    nA = op.nA
    ya, yb = y[:nA].double(), y[nA:].double().view(-1, 6)
    tot = float((y.double() ** 2).sum()) or 1.0
    sky = float((ya ** 2).sum()) / tot
    per_row = (yb ** 2).sum(1) / tot
    top = torch.argsort(per_row, descending=True)[:3].tolist()
    rows = getattr(op, "rows", None)
    model = getattr(op, "model", None)
    names = []
    for r in top:
        eid = rows[r] if rows is not None else r
        try:
            names.append(f"{model.fset.names[int(model.e_filt[eid])]}#{eid}: {float(per_row[r]):.3f}")
        except Exception:
            names.append(f"row {r}: {float(per_row[r]):.3f}")
    per_term = ((yb ** 2).sum(0) / tot).tolist()
    K = op.K
    per_k = (ya.view(K, -1) ** 2).sum(1) / tot
    return {"sky_share": sky, "background_share": 1 - sky, "background_top_exposures": names, "background_per_term": [round(v, 4) for v in per_term],
            "sky_per_coefficient": [round(float(v), 4) for v in per_k]}


def lanczos_ritz(alphas, betas):
    """Ritz values (ascending) of the preconditioned operator P^{-1} F from the conjugate-gradient coefficients (the Lanczos tridiagonal matrix
    T: T_jj = 1/alpha_j + beta_{j-1}/alpha_{j-1}, T_j,j+1 = sqrt(beta_j)/alpha_j). Their spread is the condition number CG has met so far."""
    m = len(alphas)
    if m == 0 or not all(math.isfinite(a) and a != 0 for a in alphas):
        return []
    T = np.zeros((m, m))
    for j in range(m):
        T[j, j] = 1.0 / alphas[j] + (betas[j - 1] / alphas[j - 1] if j > 0 else 0.0)
        if j + 1 < m and j < len(betas):
            T[j, j + 1] = T[j + 1, j] = math.sqrt(max(betas[j], 0.0)) / alphas[j]
    return np.sort(np.linalg.eigvalsh(T)).tolist()


# ============================================================================= estimators
def posterior_samples(op, n, seed=0, tol=1e-3, maxiter=300, log=print, monitor=None, accept_tol=None):
    """n draws of the coefficient error from the linearised posterior: (n, K, N, N) and the solver records. With a monitor, a draw that the
    time budget cuts is DISCARDED (an unconverged draw has too small a variance); the draws completed so far are returned (None if none) and
    monitor.hit is set. accept_tol (default None: every finished draw is kept, the 18 Nov behaviour): a draw whose relative residual is above
    it is discarded too, and said so in the log (record kept with 'rejected': True)."""
    gen = torch.Generator().manual_seed(seed)
    out, recs = [], []
    with torch.no_grad():
        for i in range(n):
            if monitor is not None and monitor.stop_now(f"draw {i + 1}/{n}"):
                break
            t0 = time.perf_counter()
            x, info = pcg(op, op.sample_rhs(gen), tol=tol, maxiter=maxiter, monitor=monitor, label=f"draw {i + 1}/{n}")
            if info["truncated"]:
                log(f"  posterior draw {i + 1}/{n}: stopped by the time budget at iteration {info['iterations']} (relative residual "
                    f"{info['rel_residual']:.2e}); this draw is discarded")
                break
            if info.get("breakdown"):
                log(f"  WARNING posterior draw {i + 1}/{n}: CONJUGATE-GRADIENT BREAKDOWN ({info['breakdown']}); this draw is discarded")
                continue
            if accept_tol is not None and not info["rel_residual"] <= accept_tol:
                log(f"  WARNING posterior draw {i + 1}/{n}: NOT CONVERGED (relative residual {info['rel_residual']:.2e} above the tolerance "
                    f"{accept_tol:.1e} after {info['iterations']} iterations); this draw is discarded")
                continue
            out.append((op.lift_A(x[:op.nA]) if hasattr(op, "lift_A") else x[:op.nA].view(op.K, op.N, op.N)).cpu())
            info["seconds"] = time.perf_counter() - t0
            recs.append(info)
            tail = f"; {monitor.tail()}" if monitor is not None else ""
            log(f"  posterior draw {i + 1}/{n}: {info['iterations']} iterations, relative residual {info['rel_residual']:.2e}, {info['seconds']:.1f} s{tail}")
    return (torch.stack(out) if out else None), recs


def gauss_newton_polish(op, tol=1e-4, maxiter=60, monitor=None, keep_every=0):
    """One Gauss-Newton step from the fit with the response held: A_pol = A_fit + [F^{-1} g]_A, g = residual_rhs(). For the grid sky (linear
    in A) this is the minimiser of the weighted least squares with the weak prior, i.e. the estimator the linearised posterior describes;
    positivity is not imposed. The backgrounds move too (returned) within the gauge. Returns (A_pol (K, N, N), delta_b (n_rows, 6), info).
    With a monitor the step may be truncated (info['truncated']): the last conjugate-gradient iterate is used and info['rel_residual'] says how far
    the solve got (1.0 if not even the right-hand side was computed: A_pol = A_fit)."""
    if monitor is not None and monitor.stop_now("Gauss-Newton step"):
        return (op.A_fit.clone().detach(), torch.zeros(len(op.rows), 6), {"iterations": 0, "rel_residual": 1.0, "history": [], "truncated": True})
    with torch.no_grad():
        x, info = pcg(op, op.residual_rhs(), tol=tol, maxiter=maxiter, monitor=monitor, label="Gauss-Newton step", keep_every=keep_every)
        lift = op.lift_A if hasattr(op, "lift_A") else (lambda v: v.view(op.K, op.N, op.N))     # restricted unknowns -> coefficient grid
        if "x_back" in info:                                  # the sky of the step some iterations earlier (for the change diagnostic)
            info["A_back"] = (op.A_fit + lift(info.pop("x_back")[:op.nA])).detach().cpu()
        db = op.bkg_to_raw(x[op.nA:]) if hasattr(op, "bkg_to_raw") else x[op.nA:].view(-1, 6)   # raw Legendre terms
        return (op.A_fit + lift(x[:op.nA])).detach(), db.detach(), info


def exact_variance(op, phi, smoother, pts, tol=1e-4, maxiter=600, monitor=None):
    """Exact sigma of the product at output pixels pts [(ix, iy), ...]: one solve F x = k per pixel (fewer if the time budget stops them)."""
    out = []
    for ix, iy in pts:
        k = phi[:, None, None].cpu() * smoother.weights_at(ix, iy)[None].to(torch.float32)
        rhs = op.unit_rhs(k)
        x, info = pcg(op, rhs, tol=tol, maxiter=maxiter, monitor=monitor, label=f"exact ({ix}, {iy})")
        if info["truncated"]:
            break
        out.append((float(torch.dot(rhs, x)) ** 0.5, info["iterations"]))
    return out


def band_of(A, phi):
    return torch.einsum("k,...khw->...hw", phi.to(A.dtype).to(A.device), A)


def sigma_maps(samples, phi, pitch, scales):
    """sigma_s (n_scales, N, N) at the node positions, from posterior draws (n, K, N, N)."""
    B = band_of(samples, phi)                     # (n, N, N)
    N = B.shape[-1]
    out = []
    for s in scales:
        sm = Smoother(N, pitch, s)
        out.append(sm(B.double()).pow(2).mean(0).sqrt())
    return torch.stack(out)


def supported_resolution(sig, scales, alpha_max=2.0):
    """s*(x): smallest scale with alpha = -d ln sigma / d ln s <= alpha_max (log-linear between ladder points; scales > 0 only).
    Returns s* (N, N) (NaN where never reached on the ladder) and alpha (n_scales - 1, N, N) at the mid-points."""
    sc = np.asarray(scales, dtype=np.float64)
    keep = sc > 0
    sc, sg = sc[keep], sig[torch.as_tensor(keep)]
    ls = torch.tensor(np.log(sc))
    lsg = torch.log(sg.clamp(min=1e-30))
    alpha = -(lsg[1:] - lsg[:-1]) / (ls[1:] - ls[:-1])[:, None, None]
    mids = 0.5 * (ls[1:] + ls[:-1])
    # scan from the largest scale down: s* = the scale below which alpha first exceeds alpha_max (at the finest scales the prior takes over and
    # alpha falls again towards 1, the white-noise rate of the prior: that regime is not resolution)
    n = alpha.shape[0]
    s_star = torch.full(sig.shape[1:], float("nan"), dtype=torch.float64)
    ok = alpha[n - 1] <= alpha_max
    s_star = torch.where(ok, torch.full_like(s_star, math.exp(float(mids[n - 1]))), s_star)
    alive = ok.clone()
    for i in range(n - 2, -1, -1):
        a_lo, a_hi = alpha[i], alpha[i + 1]
        stop = alive & (a_lo > alpha_max)
        f = ((a_lo - alpha_max) / (a_lo - a_hi).clamp(min=1e-12)).clamp(0, 1)
        val = torch.exp(mids[i] + f * (mids[i + 1] - mids[i]))
        s_star = torch.where(stop, val, s_star)
        cont = alive & ~stop
        s_star = torch.where(cont, torch.full_like(s_star, math.exp(float(mids[i]))), s_star)
        alive = cont
    return s_star, alpha


def correlation_length(samples, phi, pitch, s, covered, max_lag=None):
    """Azimuthally averaged correlation of the product errors at scale s over the covered nodes; length = lag where it falls to 1/2 [arcsec]."""
    B = band_of(samples, phi).double()
    N = B.shape[-1]
    E = Smoother(N, pitch, s)(B)
    m = covered.double().cpu()
    E = E / E.pow(2).mean(0).sqrt().clamp(min=1e-30) * m
    F = torch.fft.rfft2(E, s=(2 * N, 2 * N))
    ac = torch.fft.irfft2((F * F.conj()).mean(0), s=(2 * N, 2 * N))
    Fm = torch.fft.rfft2(m, s=(2 * N, 2 * N))
    nm = torch.fft.irfft2(Fm * Fm.conj(), s=(2 * N, 2 * N))
    ac = ac / nm.clamp(min=1.0)
    L = max_lag or N // 2
    yy, xx = np.meshgrid(np.r_[0:L], np.r_[0:L], indexing="ij")
    r = np.hypot(xx, yy)
    prof = []
    for k in range(L):
        sel = (r >= k - 0.5) & (r < k + 0.5)
        prof.append(float(ac[:L, :L].numpy()[sel].mean()))
    prof = np.array(prof) / prof[0]
    below = np.nonzero(prof < 0.5)[0]
    if below.size == 0:
        return float("nan"), prof
    k = below[0]
    lag = (k - 1) + (prof[k - 1] - 0.5) / max(prof[k - 1] - prof[k], 1e-12)
    return float(lag * pitch), prof


def difference_map(img_a, img_b, phi, pitch, s, mode="halves", window=None, covered=None):
    """Empirical estimator from two coefficient grids (K, N, N) (or band images (N, N)): the product difference at scale s and the implied
    sigma of the full fit. mode 'halves': h = (I_a - I_b) / 2 (fits on disjoint halves; Var h = Var of the full fit when the noise variance
    scales as 1 / exposures); mode 'seeds': h = (I_a - I_b) / sqrt(2) (two fits differing by the seed; Var h = optimisation scatter of one fit).
    Implied sigma: global rms of h over the covered nodes and a local map sqrt(g_window * h^2) (window default 6 s)."""
    def band(x):
        x = torch.as_tensor(x).double()
        return band_of(x, phi.double()) if x.dim() == 3 else x
    a, b = band(img_a), band(img_b)
    N = a.shape[-1]
    sm = Smoother(N, pitch, s, dtype=torch.float64)
    fac = 0.5 if mode == "halves" else 1 / math.sqrt(2)
    h = fac * (sm(a) - sm(b))
    cov = torch.ones(N, N, dtype=torch.bool) if covered is None else torch.as_tensor(covered).bool()
    glob = float(h[cov].pow(2).mean().sqrt())
    w = window or max(6 * s, 3 * pitch)
    sw = Smoother(N, pitch, w, dtype=torch.float64)
    num = sw(h.pow(2) * cov.double())
    den = sw(cov.double()).clamp(min=1e-6)
    local = (num / den).sqrt() * cov.double()
    return {"h": h, "sigma_global": glob, "sigma_local": local, "window": w, "mode": mode}


# ============================================================================= half split of the training exposures
def alternate_groups(groups, half):
    """Training dither groups kept by --train-half: groups sorted by name, rank parity == half. Returns the kept set."""
    gs = sorted(set(groups))
    return {g for i, g in enumerate(gs) if i % 2 == half}


# ============================================================================= one call: the whole map
def solve_shared(op, n_samples=8, method="draws", polish=True, tol=1e-4, maxiter=60, seed=0, log=print, monitor=None, accept_tol=None):
    """The part of the map that does not depend on the band: the Gauss-Newton step (all K spectral coefficients at once) and the posterior
    draws (K coefficients each). Done once for any number of bands. The step comes first (it is the product), the draws after it. With a
    monitor, the time budget can cut either: the step is then the last iterate, a cut draw is discarded, and no further draw is started.
    Returns a dict: A_pol (None without the step), db, polish (info), samples (None without draws), recs, seconds_samples, truncated, where,
    residual (of the solve that was cut, else of the step), n_samples_requested.
    accept_tol (default None: the 18 Nov behaviour, the last iterate is the product whatever its residual): the step counts as converged only
    when its relative residual is at most accept_tol; otherwise 'polish_converged' is False, the log says so plainly, and band_readout writes
    the fit's own sky as SKY (POLISHED false) with the unconverged step aside. Draws above accept_tol are discarded."""
    sh = {"A_pol": None, "db": None, "polish": None, "samples": None, "recs": [], "seconds_samples": 0.0, "truncated": False, "where": "",
          "residual": float("nan"), "n_samples_requested": n_samples if method == "draws" else 0, "method": method, "accept_tol": accept_tol,
          "polish_converged": None}
    with torch.no_grad():
        if polish:
            A_pol, db, info = gauss_newton_polish(op, tol=tol, maxiter=maxiter, monitor=monitor, keep_every=10 if accept_tol is not None else 0)
            sh.update(A_pol=A_pol.cpu(), db=db.cpu(), polish=info, residual=info["rel_residual"])
            ritz = (f"; preconditioned spectrum (Lanczos estimate from the iterations) {info['ritz_min']:.3g} to {info['ritz_max']:.3g}, ratio "
                    f"{info['ritz_max'] / info['ritz_min']:.3g}" if info.get("ritz_min", 0) and math.isfinite(info.get("ritz_min", float("nan"))) else "")
            log(f"Gauss-Newton step from the fit: {info['iterations']} iterations, relative residual {info['rel_residual']:.2e}"
                + (" (TRUNCATED by the time budget)" if info["truncated"] else "")
                + (f"; estimated error of the step in the posterior metric (F-norm, 10 iterations before the end) {info['err_F_est']:.3g}, relative "
                   f"{info['rel_err_F_est']:.2e}" if math.isfinite(info.get("err_F_est", float("nan"))) else "")
                + f"; background constants moved by {db[:, 0].abs().max().item():.4g} at most (gauge-preserving)" + ritz)
            if info.get("breakdown"):
                log(f"WARNING: CONJUGATE-GRADIENT BREAKDOWN in the Gauss-Newton step: {info['breakdown']}. The solve stopped there.")
            if accept_tol is not None:
                conv = info["iterations"] > 0 and info["rel_residual"] <= accept_tol and not info.get("breakdown")
                sh["polish_converged"] = bool(conv)
                if not conv:
                    log(f"WARNING: the Gauss-Newton step did NOT CONVERGE (relative residual {info['rel_residual']:.2e} after {info['iterations']} "
                        f"iterations; tolerance {accept_tol:.1e}). SKY is the fit's own sky and is NOT polished (header POLISHED = F, CONVERGD = F); "
                        f"the unconverged step is written as SKY_UNCONV for inspection only")
                else:
                    log(f"Gauss-Newton step converged (relative residual {info['rel_residual']:.2e} <= tolerance {accept_tol:.1e}): SKY is polished")
        t0 = time.perf_counter()
        if method == "draws" and n_samples > 0 and not (monitor is not None and monitor.hit):
            samp, recs = posterior_samples(op, n_samples, seed=seed, tol=tol, maxiter=maxiter, log=log, monitor=monitor, accept_tol=accept_tol)
            sh.update(samples=samp, recs=recs)
        sh["seconds_samples"] = time.perf_counter() - t0
    if monitor is not None and monitor.hit:
        sh.update(truncated=True, where=monitor.where, residual=monitor.residual)
        log(f"time budget reached in '{monitor.where}' (relative residual reached {monitor.residual:.2e}); {len(sh['recs'])} of "
            f"{sh['n_samples_requested']} draws complete; outputs are written from what was solved")
    return sh


def band_readout(model, op, shared, band, resolution="auto", alpha_max=2.0, scales_pitch=SCALES_DEFAULT, seed=0, n_exact=0, log=print, var_window=3.0,
                 monitor=None, rho=None, tol=1e-4):
    """Everything that depends on the band, from the shared solve: the circulant sigma, sigma from the draws, supported resolution, the product at
    its resolution, the error correlation length and (optional) exact solves. No application of F except for the exact solves."""
    sky = model.sky
    pitch, N = sky.pitch, sky.N
    phi = band_vector(model, band).float().cpu()
    scales = [f * pitch for f in scales_pitch]
    cov = op.covered.cpu()
    samp, recs, method = shared["samples"], shared["recs"], shared["method"]
    circ = torch.stack([op.circulant_sigma(phi, sc) if sc > 0 else torch.full((N, N), float("nan"), dtype=torch.float64) for sc in scales])
    sig = sigma_maps(samp, phi, pitch, scales) if samp is not None else circ
    s_star, alpha = supported_resolution(sig, scales, alpha_max)
    s_med = float(np.nanmedian(s_star[cov].numpy())) if bool(cov.any()) else float("nan")
    s_use = s_med if resolution == "auto" else float(resolution)
    sm = Smoother(N, pitch, s_use, dtype=torch.float64)
    A_fit = op.A_fit.cpu().double()
    img_fit = sm(band_of(A_fit, phi.double()))
    pol_info = shared["polish"]
    img = sm(band_of(shared["A_pol"].double(), phi.double())) if shared["A_pol"] is not None else img_fit
    conv = shared.get("polish_converged")
    chg = {}
    if pol_info and pol_info.get("A_back") is not None:      # change of the product over the last iterations, in units of the Fourier sigma
        sc_ = op.circulant_sigma(phi, s_use) if s_use > 0 else None
        if sc_ is not None:
            dimg = img - sm(band_of(pol_info["A_back"].double(), phi.double()))
            ok_ = cov & torch.isfinite(sc_) & (sc_ > 0)
            z_ = (dimg / sc_)[ok_]
            if z_.numel():
                chg = {"change_rms": float(z_.pow(2).mean().sqrt()), "change_max": float(z_.abs().max()), "change_from": int(pol_info["iter_back"])}
                log(f"band {band}: the step's product changed by {chg['change_rms']:.3g} sigma rms, {chg['change_max']:.3g} sigma at most, between "
                    f"iteration {chg['change_from']} and the last ({pol_info['iterations']}); for a solve that converges at least twice as fast per "
                    f"10 iterations, the remaining error is below that change")
    img_unconv = None
    if conv is False:                       # an unconverged step is not the estimator sigma describes: SKY is the fit, the step is kept aside
        img_unconv, img = img, img_fit
    sig_c = op.circulant_sigma(phi, s_use)
    if samp is not None:
        var = sm(band_of(samp, phi).double()).pow(2).mean(0)
        w = Smoother(N, pitch, var_window * s_use, dtype=torch.float64)
        m = cov.double()
        sigma = (w(var * m) / w(m).clamp(min=1e-6)).sqrt() * m
        ell, prof = correlation_length(samp, phi, pitch, s_use, cov)
    else:
        sigma = torch.nan_to_num(sig_c, nan=0.0)
        ell, prof = float("nan"), None
    exact = []
    if n_exact:
        rng = np.random.default_rng(seed)
        idx = torch.nonzero(cov)
        pick = idx[torch.as_tensor(rng.choice(len(idx), size=min(n_exact, len(idx)), replace=False))]
        ex = exact_variance(op, phi, sm, [(int(p[1]), int(p[0])) for p in pick], monitor=monitor)
        exact = [{"row": int(p[0]), "col": int(p[1]), "sigma_exact": v, "iterations": it, "sigma_map": float(sigma[p[0], p[1]]),
                  "sigma_circulant": float(sig_c[p[0], p[1]])} for p, (v, it) in zip(pick, ex)]
    ok = cov & torch.isfinite(sig_c) & (sigma > 0)
    ratio = float((sigma[ok] / sig_c[ok]).median()) if bool(ok.any()) else float("nan")
    iters = [r["iterations"] for r in recs]
    return {"image": img, "image_fit": img_fit, "sigma": sigma, "sigma_circulant": sig_c, "draws_over_circulant": ratio, "s_star": s_star, "alpha": alpha,
            "sigma_scales": sig, "scales": scales, "resolution": s_use, "resolution_median": s_med, "corr_length": ell, "corr_profile": prof, "covered": cov,
            "samples": samp, "phi": phi, "exact": exact, "cg_iterations": iters, "seconds_samples": shared["seconds_samples"], "n_apply": op.n_apply,
            "seconds_per_apply": op.t_apply / max(op.n_apply, 1), "mu": op.mu, "rho": op.rho if rho is None else rho, "node_noise": op.node_noise,
            "n_pix": op.n_pix, "n_huber_downweighted": op.n_huber, "polish": pol_info, "method": method, "op": op,
            "truncated": shared["truncated"], "truncated_in": shared["where"], "residual": shared["residual"],
            "n_samples_requested": shared["n_samples_requested"], "sigma_source": "draws" if samp is not None else "circulant",
            "polish_converged": conv, "accept_tol": shared.get("accept_tol"), "image_unconverged": img_unconv, **chg,
            "precond": getattr(op, "precond_mode", "")}


def uncertainty_map(model, exps, cfg, band, n_samples=8, rho=10.0, scales_pitch=SCALES_DEFAULT, resolution="auto", alpha_max=2.0, tol=1e-4,
                    maxiter=60, seed=0, n_exact=0, log=print, P=None, op=None, method="draws", polish=True, var_window=3.0, monitor=None):
    """Sky product, sigma, supported resolution and correlation length for one band (node grid). Returns a dict of tensors / numbers.
    method 'draws': sigma from n_samples posterior draws, the variance map averaged locally over a Gaussian window of var_window x the product
    resolution (reduces the sampling error of few draws; assumes sigma varies slowly on that scale); the Fourier (circulant) sigma is computed
    too and written alongside. method 'circulant': no draws (cost: the preconditioner's K + 1 applications only).
    polish: the product SKY is the fit plus one Gauss-Newton step (the estimator the posterior describes; calibrated on the known-truth
    problem); SKY_FIT is the fit's own sky at the same resolution (its sigma is NOT the posterior's: see the report).
    Several bands: call solve_shared once and band_readout per band (what the command line does); this function is the two for one band."""
    op = op or SkyNormalOperator(model, exps, cfg, P=P, rho=rho, log=log)
    shared = solve_shared(op, n_samples=n_samples, method=method, polish=polish, tol=tol, maxiter=maxiter, seed=seed, log=log, monitor=monitor)
    return band_readout(model, op, shared, band, resolution=resolution, alpha_max=alpha_max, scales_pitch=scales_pitch, seed=seed, n_exact=n_exact,
                        log=log, var_window=var_window, monitor=monitor, rho=rho, tol=tol)


# ============================================================================= files
def write_product(path, res, band, pitch, meta=None):
    """Write the product. FITS (astropy, already used by the package's loader): one image per extension, readable by any astronomy viewer,
    header carries the band, resolution, units and the conventions; falls back to .npz when astropy is missing."""
    meta = dict(meta or {})
    N = res["image"].shape[-1]
    arrays = {"SKY": res["image"].numpy().astype(np.float32), "SIGMA": res["sigma"].numpy().astype(np.float32),
              **({"SKY_FIT": res["image_fit"].numpy().astype(np.float32)} if "image_fit" in res else {}),
              **({"SIGMA_CIRC": torch.nan_to_num(res["sigma_circulant"], nan=0.0).numpy().astype(np.float32)} if "sigma_circulant" in res else {}),
              "RESOLUTION": res["s_star"].numpy().astype(np.float32), "COVERED": res["covered"].numpy().astype(np.uint8),
              "SIGMA_SCALES": res["sigma_scales"].numpy().astype(np.float32),
              **({"SKY_UNCONV": res["image_unconverged"].numpy().astype(np.float32)} if res.get("image_unconverged") is not None else {}),
              **{k: v.numpy().astype(np.float32) for k, v in (res.get("extra_images") or {}).items()}}
    conv = res.get("polish_converged")
    polished = bool(res.get("polish")) and res["polish"].get("iterations", 1) > 0 and conv is not False
    hdr = {"BAND": str(band), "PRODRES": (res["resolution"], "[arcsec] width of the Gaussian smoothing of SKY and SIGMA"),
           "RESMED": (res["resolution_median"], "[arcsec] median supported resolution over covered nodes"),
           "CORRLEN": (res["corr_length"], "[arcsec] lag where the error correlation falls to 1/2"),
           "PRIORRHO": (res["rho"], "prior sd per coefficient in node-noise units"), "NSAMPLE": (len(res["cg_iterations"]), "posterior draws"),
           "POLISHED": (polished, "SKY = fit + one converged Gauss-Newton step"),
           **({"CONVERGD": (bool(conv), "step residual <= ACCTOL (F: SKY = SKY_FIT)"),
               "ACCTOL": (float(res.get("accept_tol") or float("nan")), "rel. residual accepted as converged")} if conv is not None else {}),
           **({"PRECOND": (str(res["precond"]), "preconditioner of the solves")} if res.get("precond") else {}),
           **({"RITZMIN": (float(res["polish"].get("ritz_min", float("nan"))), "Lanczos min of the precond. spectrum"),
               "RITZMAX": (float(res["polish"].get("ritz_max", float("nan"))), "Lanczos max of the precond. spectrum"),
               "POLERRF": (float(res["polish"].get("rel_err_F_est", float("nan"))), "est. rel. F-norm error of the step")}
              if res.get("polish") and "ritz_min" in res["polish"] else {}),
           **({"POLCHG": (res["change_rms"], "step change, last iters, rms [sigma]"), "POLCHGMX": (res["change_max"], "same, max [sigma]"),
               "POLCHGIT": (res["change_from"], "change measured from this iteration")} if "change_rms" in res else {}),
           "UNITS": "band surface brightness (sky units)",
           "NSAMPREQ": (res.get("n_samples_requested", len(res["cg_iterations"])), "posterior draws requested"),
           "SIGMETH": (res.get("sigma_source", "draws" if res["cg_iterations"] else "circulant"), "SIGMA from draws or circulant"),
           "TRUNCATD": (bool(res.get("truncated", False)), "stopped by --unc-max-minutes"),
           "TRUNCIN": (str(res.get("truncated_in", "")), "solve that was cut"),
           "RESIDUAL": (float(res.get("residual", float("nan"))), "rel. residual of the cut solve"),
           "POLRESID": (float((res.get("polish") or {}).get("rel_residual", float("nan"))), "rel. residual of the step"),
           "POLITER": (int((res.get("polish") or {}).get("iterations", 0)), "CG iterations of the step"),
           "NPASSES": (int(res.get("n_apply", 0)), "operator applications (passes)"),
           "SIGRATIO": (res.get("draws_over_circulant", float("nan")), "median SIGMA / SIGMA_CIRC"),
           "PITCH": (pitch, "[arcsec] node spacing"), "XREF": (-(N - 1) / 2 * pitch, "[arcsec] X of column 0 (west); Y of row 0 the same (north)"),
           "SCALES": (",".join(f"{s:.4f}" for s in res["scales"]), "[arcsec] SIGMA_SCALES planes")}
    for k, v in meta.items():
        hdr[k[:8].upper()] = v
    if path.endswith(".npz"):
        np.savez_compressed(path, **arrays, header=json.dumps({k: (v[0] if isinstance(v, tuple) else v) for k, v in hdr.items()}))
        return path
    try:
        from astropy.io import fits
    except ImportError:
        p = os.path.splitext(path)[0] + ".npz"
        return write_product(p, res, band, pitch, meta)
    hl = [fits.PrimaryHDU()]
    for k, v in hdr.items():
        val = v[0] if isinstance(v, tuple) else v
        if isinstance(val, float) and not math.isfinite(val):          # FITS headers cannot hold NaN: written as a string
            v = ("NaN", v[1]) if isinstance(v, tuple) else "NaN"
        hl[0].header[k] = v
    for name, a in arrays.items():
        h = fits.ImageHDU(a, name=name)
        h.header["CTYPE1"], h.header["CTYPE2"] = "OFFSET-X", "OFFSET-Y"
        h.header["CUNIT1"] = h.header["CUNIT2"] = "arcsec"
        h.header["CRPIX1"] = h.header["CRPIX2"] = (N + 1) / 2
        h.header["CRVAL1"] = h.header["CRVAL2"] = 0.0
        h.header["CDELT1"] = h.header["CDELT2"] = pitch
        hl.append(h)
    fits.HDUList(hl).writeto(path, overwrite=True)
    return path


def read_coeffs(model, skies, st, log=print, cfg=None):
    """Load a saved state into the model and return the coefficient grid (K, N, N) of the first tile."""
    from .cli import load_saved_state
    load_saved_state(model, skies, st, log=log, cfg=cfg)
    with torch.no_grad():
        return model.sky.coeffs().detach().clone()


# ============================================================================= command line
def build_from_args(cfg, log=print):
    """Data and model as the package's command line builds them (cli.main up to the state load), single tile only."""
    from . import data as _data
    from . import instruments as _ins
    from .cli import make_sky
    from .data import load_field, make_patches
    from .model import FieldOperator, resolve_instrument_options
    from .sky import RBFBasis
    from .tiles import TileSet
    _data.DQ_IGNORE_BITS = int(cfg.dq_ignore_bits)
    _ins.DQ_IGNORE_OVERRIDE.clear()
    _ins.DQ_IGNORE_OVERRIDE.update(_ins.parse_dq_ignore(getattr(cfg, "dq_ignore", "")))
    _ins.HST_WCS = getattr(cfg, "hst_wcs", "apriori")
    _data.RATE_WCS = cfg.rate_wcs
    _data.LOADER = getattr(cfg, "loader", "astropy")
    dev = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else ("cpu" if cfg.device == "auto" else cfg.device))
    if not cfg.splits:
        cfg.splits = os.path.join(cfg.protocol_root, f"splits_{cfg.field}.json")
    exps, fset, meta, info = load_field(cfg, dev, log=log)
    _ins.spectral_setup(fset, cfg, log=log)
    resolve_instrument_options(cfg, len(info["insts"]), log=log)
    if info["multi_tile"]:
        raise SystemExit("uncertainty: one tile (box) per run; select one box")
    half = info["half_data"] - cfg.data_margin + cfg.sky_margin
    basis = RBFBasis(cfg.spec_k).to(dev)
    for e in exps:
        e.to_dev(dev)
    sky, N = make_sky(exps, fset, cfg, half, dev, basis=basis)
    skies = {info["tiles"][0]: sky}
    model = FieldOperator(cfg, fset, meta, sky, dev).to(dev)
    model.attach_pixel_gain(exps, cfg, fset_names=fset.names, det_names=info["dets"])
    model.attach_filter_shift(exps, cfg, fset_names=fset.names, det_names=info["dets"])
    TileSet(model, exps, skies, cfg, log=log)
    for e in exps:
        pix = float(abs(np.linalg.det(np.asarray(e.extra["J"]))) ** 0.5)
        make_patches(e, max(16, int(round(cfg.patch_arcsec / pix))))
    return model, exps, fset, skies


def add_args(ap):
    g = ap.add_argument_group("uncertainty (python -m fieldop.uncertainty)")
    g.add_argument("--unc-cmd", choices=["map", "diff", "diff-files", "lab"], default="map",
                   help="map: product, sigma and resolution from --state; diff: the fitted skies of two states; diff-files: two products of 'map'")
    g.add_argument("--unc-file-a", default="")
    g.add_argument("--unc-file-b", default="")
    g.add_argument("--unc-config", default="", help="metrics.json of the fit: its configuration is used (options given here win); "
                   "--state defaults to fieldop_state.pt next to it")
    g.add_argument("--unc-band", default="0", help="filter name or index, or coef:k for one spectral coefficient; a comma-separated list, or 'all' "
                   "(every filter of the fit): ONE solve, one file per band")
    g.add_argument("--unc-out", default="sky_uncertainty.fits", help="output file; with several bands a pattern containing {band}")
    g.add_argument("--unc-samples", type=int, default=8)
    g.add_argument("--unc-method", choices=["draws", "circulant"], default="draws")
    g.add_argument("--unc-polish", type=int, choices=[0, 1], default=1, help="SKY = fit + one Gauss-Newton step (the estimator sigma describes)")
    g.add_argument("--unc-tile", type=int, default=0, help="render tile [detector pixels]; 0 = 64 on a CUDA device, --region-px on CPU")
    g.add_argument("--unc-chunk", type=int, default=0, help="tiles per batched render; 0 = follow --unc-batch-px (CPU without --unc-batch-px: 32)")
    g.add_argument("--unc-batch-px", type=int, default=DEFAULT_BATCH_PX, help="pixel budget of one batched render (default 16000, about the fit's "
                   "batch); on CPU it is used only when given explicitly; --unc-chunk wins")
    g.add_argument("--unc-max-minutes", type=float, default=0.0, help="time budget of the whole run in minutes (0 = none): when the next pass would "
                   "end after it, the solve stops cleanly, outputs are written with TRUNCATD/RESIDUAL in the header, exit status 0")
    g.add_argument("--unc-prior", type=float, default=10.0, help="prior sd of each coefficient in units of the median node noise")
    g.add_argument("--unc-resolution", default="auto", help="'auto' (median supported resolution) or a width in arcsec")
    g.add_argument("--unc-alpha", type=float, default=2.0)
    g.add_argument("--unc-tol", type=float, default=1e-4, help="relative residual at which a conjugate-gradient solve stops")
    g.add_argument("--unc-maxiter", type=int, default=100, help="iterations per solve at most (19 Nov: 100, was 60)")
    g.add_argument("--unc-precond", choices=["filter", "fourier", "jacobi", "bjacobi"], default="filter",
                   help="filter (19 Nov, default): one Fourier model per filter and per-node K x K scaling by each filter's coverage; fourier: the 18 Nov "
                        "preconditioner (one Fourier model probed at the centre node; reproduces the truth-problem calibration bit for bit); jacobi")
    g.add_argument("--unc-precision", choices=["mixed", "float32"], default="mixed",
                   help="mixed (19 Nov, default): float64 vectors, inner products, adjoint sums over batches and background blocks (renders stay float32); "
                        "float32: as on 18 Nov")
    g.add_argument("--unc-restrict", type=float, default=0.0,
                   help="arcsec: solve for band images on a coarse grid of this pitch (a multiple of the sky pitch; e.g. 0.08), the directions the data "
                        "leave to the prior held at the fit (RestrictedOperator); 0 = the full coefficient grid (the 18 Nov formulation)")
    g.add_argument("--unc-bkg-basis", choices=["orth", "raw"], default="orth",
                   help="orth (19 Nov, section 11; needs --unc-precision mixed): each exposure's six background functions orthonormalised over its "
                        "weighted pixels, directions below --unc-bkg-rcond dropped (held at the fit), the gauge imposed exactly; raw: as before")
    g.add_argument("--unc-bkg-rcond", type=float, default=1e-4,
                   help="orth: drop background directions whose singular value is below this fraction of the largest (the cut cannot tell them apart)")
    g.add_argument("--unc-check", type=int, choices=[0, 1], default=1,
                   help="1: check the operator and the preconditioner (symmetry, positivity, the prior-only direction; 4 applications) before solving "
                        "and ABORT with the reason if a check fails")
    g.add_argument("--unc-lab-box", type=float, default=2.0, help="lab: half width [arcsec] of the central region whose training pixels are used")
    g.add_argument("--unc-lab-iters", type=int, default=30, help="lab: conjugate-gradient iterations per variant")
    g.add_argument("--unc-lab-variants", default=LAB_VARIANTS, help="lab: comma-separated variants (see run_lab)")
    g.add_argument("--unc-regions", type=int, default=8, help="--unc-precond filter: number of coverage regions, each with its own Fourier model "
                   "(device memory about 8 K^2 N^2 bytes per region: 433 MB at K = 12, N = 613)")
    g.add_argument("--unc-probes", type=int, default=1, help="--unc-precond filter: probes per filter whose windowed responses are averaged (the "
                   "Fourier model averaged over sub-pixel phases; placed in the same passes, no extra cost); 1 = one probe, no window (the tested default)")
    g.add_argument("--unc-accept-tol", type=float, default=1e-3,
                   help="a solve whose final relative residual is above this did NOT converge: the step is then not applied (SKY = SKY_FIT, "
                        "POLISHED F, CONVERGD F, step written as SKY_UNCONV) and a draw is discarded; the log says so. 0 = the 18 Nov behaviour (no check)")
    g.add_argument("--unc-exact", type=int, default=0, help="exact solves at this many random covered pixels (cross-check)")
    g.add_argument("--unc-state-b", default="", help="diff: second state (first is --state)")
    g.add_argument("--unc-diff-mode", choices=["halves", "seeds"], default="halves")
    return ap


def _local_peaks(img, mask, n, min_sep):
    """Local maxima (3 x 3) inside mask, brightest first, at least min_sep nodes apart: [(row, col)]."""
    t = torch.as_tensor(img, dtype=torch.float64)
    mx = torch.nn.functional.max_pool2d(t[None, None], 3, 1, 1)[0, 0]
    m = torch.as_tensor(mask) & (t == mx) & torch.isfinite(t)
    cand = torch.nonzero(m)
    vals = t[cand[:, 0], cand[:, 1]]
    out = []
    for i in torch.argsort(-vals).tolist():
        y, x = int(cand[i, 0]), int(cand[i, 1])
        if all((y - yy) ** 2 + (x - xx) ** 2 >= min_sep ** 2 for yy, xx in out):
            out.append((y, x))
        if len(out) >= n:
            break
    return out


def calibration_difference(img_a, img_b, sig_a, sig_b, covered, pitch, n_src=40, win=6, min_snr=30.0):
    """Response and registration differences between two fits (e.g. the halves), from their products at one resolution. Per bright source (local
    maxima of I = (I_a + I_b) / 2 with peak S/N >= min_snr, window +-win nodes) a weighted least-squares fit of
        I_a - I_b = (s - 1) I - delta . grad I + v lap I + c
    (flux scale s, position difference delta [arcsec], width term v = (V_a - V_b) / 2 [arcsec^2] for Gaussian variances V, constant c).
    Returns the medians over sources (a difference common to the whole image: zero point, astrometric frame, resolution), the robust spreads
    (source to source), and maps: the difference with the common terms removed, and sigma_cal = the error that the source-to-source part implies
    for ONE fit with the halves convention (spread / 2, as h = (I_a - I_b) / 2):
        sigma_cal^2 = (e_s I)^2 + (e_x dI/dx)^2 + (e_y dI/dy)^2 + (e_v lap I)^2.
    Arrays are numpy (N, N), rows = Y, columns = X."""
    a, b = np.asarray(img_a, np.float64), np.asarray(img_b, np.float64)
    cov = np.asarray(covered, bool) & np.isfinite(a) & np.isfinite(b)
    sig = np.sqrt(np.asarray(sig_a, np.float64) ** 2 + np.asarray(sig_b, np.float64) ** 2)
    cov &= sig > 0
    I = 0.5 * (a + b)
    gy, gx = np.gradient(I, pitch)
    lap = np.gradient(np.gradient(I, pitch, axis=0), pitch, axis=0) + np.gradient(np.gradient(I, pitch, axis=1), pitch, axis=1)
    inner = cov.copy()
    e = win + 2
    inner[:e], inner[-e:], inner[:, :e], inner[:, -e:] = False, False, False, False
    snr = np.where(cov, I / np.where(sig > 0, sig, 1.0), 0.0)
    pk = [p for p in _local_peaks(np.where(inner, I, -np.inf), inner, 4 * n_src, 2 * win) if snr[p] >= min_snr][:n_src]
    rows = []
    for (y, x) in pk:
        sl = (slice(y - win, y + win + 1), slice(x - win, x + win + 1))
        m = cov[sl].ravel()
        D = (a - b)[sl].ravel()[m]
        W = 1.0 / sig[sl].ravel()[m]
        X = np.stack([I[sl].ravel()[m], -gx[sl].ravel()[m], -gy[sl].ravel()[m], lap[sl].ravel()[m], np.ones(int(m.sum()))], 1)
        coef = np.linalg.lstsq(X * W[:, None], D * W, rcond=None)[0]
        rows.append([coef[0], coef[1], coef[2], coef[3], float(np.sqrt(np.mean((D * W) ** 2))), float(np.sqrt(np.mean(((D - X @ coef) * W) ** 2)))])
    if not rows:
        return {"n_sources": 0}
    R = np.asarray(rows)
    med = np.median(R[:, :4], 0)
    rob = 1.4826 * np.median(np.abs(R[:, :4] - med), 0)
    aligned = (a - b) - (med[0] * I - med[1] * gx - med[2] * gy + med[3] * lap)
    eps = rob / 2.0
    sig_cal = np.sqrt((eps[0] * I) ** 2 + (eps[1] * gx) ** 2 + (eps[2] * gy) ** 2 + (eps[3] * lap) ** 2) * cov
    return {"n_sources": len(rows), "scale_minus_1": float(med[0]), "shift_arcsec": [float(med[1]), float(med[2])], "width_arcsec2": float(med[3]),
            "spread_scale": float(rob[0]), "spread_shift_arcsec": [float(rob[1]), float(rob[2])], "spread_width_arcsec2": float(rob[3]),
            "source_pull_rms_before_median": float(np.median(R[:, 4])), "source_pull_rms_after_median": float(np.median(R[:, 5])),
            "aligned": aligned, "sigma_cal": sig_cal, "mean_image": I}


def diff_products(path_a, path_b, out, mode="halves", window_px=8.0, ext="SKY", calib=True):
    """Difference of two product files written by 'map' (same grid and resolution): e.g. the polished skies of fits on the two halves
    (--train-half 0 / 1), or of two seeds. Writes h (SKY), the implied sigma (SIGMA: local rms of h over a Gaussian window of window_px nodes,
    within the nodes covered in both) and returns the global rms over the common covered nodes."""
    from astropy.io import fits
    with fits.open(path_a) as ha, fits.open(path_b) as hb:
        a, b = ha[ext].data.astype(np.float64), hb[ext].data.astype(np.float64)
        cov = (ha["COVERED"].data > 0) & (hb["COVERED"].data > 0)
        pitch, res_a, res_b = float(ha[ext].header["CDELT1"]), ha[0].header["PRODRES"], hb[0].header["PRODRES"]
        sa = ha["SIGMA"].data.astype(np.float64) if "SIGMA" in ha else None
        sb = hb["SIGMA"].data.astype(np.float64) if "SIGMA" in hb else None
    if abs(res_a - res_b) > 1e-6:
        raise SystemExit(f"products at different resolutions: {res_a} and {res_b} arcsec")
    fac = 0.5 if mode == "halves" else 1 / math.sqrt(2)
    h = torch.tensor(fac * (a - b))
    m = torch.tensor(cov).double()
    sw = Smoother(h.shape[-1], pitch, window_px * pitch, dtype=torch.float64)
    local = (sw(h.pow(2) * m) / sw(m).clamp(min=1e-6)).sqrt() * m
    glob = float(h[torch.tensor(cov)].pow(2).mean().sqrt())
    res = {"image": h, "sigma": local, "s_star": torch.full_like(h, float("nan")), "covered": torch.tensor(cov), "sigma_scales": local[None],
           "resolution": res_a, "resolution_median": float("nan"), "corr_length": float("nan"), "rho": float("nan"), "cg_iterations": [], "scales": [res_a]}
    meta = {"SIGGLOB": glob, "MODE": mode}
    if calib and sa is not None and sb is not None and mode == "halves":
        cd = calibration_difference(a, b, sa, sb, cov, pitch)
        if cd.get("n_sources", 0) > 0:
            # what the two fits' stated sigma cannot explain on bright sources: the common terms (zero point, frame, resolution) in the header, the
            # source-to-source part as SIGMA_CAL; pulls of h before and after, against the stated sigma of h ((sa^2 + sb^2)^(1/2) / 2)
            sh = 0.5 * np.sqrt(sa ** 2 + sb ** 2)
            ok = cov & (sh > 0)
            pull0 = (fac * (a - b))[ok] / sh[ok]
            pull1 = (fac * cd["aligned"])[ok] / np.sqrt(sh[ok] ** 2 + cd["sigma_cal"][ok] ** 2)
            I = cd["mean_image"][ok]
            br = I >= np.percentile(I, 90)
            meta.update({"CALNSRC": cd["n_sources"], "ZPDIFF": (cd["scale_minus_1"], "flux scale a/b - 1, common to sources"),
                         "FRAMEDX": (cd["shift_arcsec"][0] * 1e3, "[mas] position a - b, common"), "FRAMEDY": (cd["shift_arcsec"][1] * 1e3, "[mas]"),
                         "WIDTHDIF": (cd["width_arcsec2"], "[arcsec2] (V_a - V_b) / 2, common"),
                         "CALSCALE": (cd["spread_scale"], "source-to-source spread of the scale"),
                         "CALSHX": (cd["spread_shift_arcsec"][0] * 1e3, "[mas] spread of the shift"), "CALSHY": (cd["spread_shift_arcsec"][1] * 1e3, "[mas]"),
                         "PULLBRT0": (float(np.sqrt(np.mean(pull0[br] ** 2))), "rms h/sigma, brightest 10%"),
                         "PULLBRT1": (float(np.sqrt(np.mean(pull1[br] ** 2))), "same, aligned, with SIGMA_CAL")})
            res["extra_images"] = {"H_ALIGNED": torch.tensor(fac * cd["aligned"]), "SIGMA_CAL": torch.tensor(cd["sigma_cal"])}
            print(f"calibration difference of the two fits ({cd['n_sources']} bright sources): flux scale {cd['scale_minus_1']:+.4f}, position "
                  f"({cd['shift_arcsec'][0] * 1e3:+.1f}, {cd['shift_arcsec'][1] * 1e3:+.1f}) mas, width {cd['width_arcsec2']:+.2e} arcsec^2 common; "
                  f"source-to-source spreads {cd['spread_scale']:.4f}, ({cd['spread_shift_arcsec'][0] * 1e3:.1f}, {cd['spread_shift_arcsec'][1] * 1e3:.1f}) mas; "
                  f"pull rms of the brightest 10 per cent {meta['PULLBRT0'][0]:.2f} -> {meta['PULLBRT1'][0]:.2f} after alignment with SIGMA_CAL")
    write_product(out, res, f"difference ({mode})", pitch, meta)
    return glob


def main(argv=None):
    from .cli import build_parser
    T_START = time.perf_counter()
    torch.set_num_threads(max(1, int(os.environ.get("FIELDOP_THREADS", "1"))))
    ap = add_args(build_parser())
    cfg = ap.parse_args(argv)
    explicit_batch = any(a == "--unc-batch-px" or a.startswith("--unc-batch-px=") for a in (sys.argv[1:] if argv is None else argv))
    if cfg.unc_config:          # the fit's own configuration (metrics.json "config"), then every option given on this command line on top
        stored = json.load(open(cfg.unc_config))
        stored = stored.get("config", stored)
        defaults = vars(ap.parse_args([]))
        ap2 = add_args(build_parser())
        for act in ap2._actions:
            act.default = argparse.SUPPRESS
        given = vars(ap2.parse_args(argv))                 # only the options written on this command line
        for k, v in stored.items():
            if k in defaults and not k.startswith("unc_"):
                setattr(cfg, k, v)
        for k, v in given.items():
            setattr(cfg, k, v)
        if not cfg.state:
            cfg.state = os.path.join(os.path.dirname(os.path.abspath(cfg.unc_config)), "fieldop_state.pt")
    if not cfg.state and cfg.unc_cmd != "diff-files":
        raise SystemExit("--state FILE (a fieldop_state.pt) or --unc-config metrics.json is required")
    T0 = T_START
    spec = [b.strip() for b in str(cfg.unc_band).split(",") if b.strip()]
    if cfg.unc_cmd == "map" and (len(spec) > 1 or (spec and spec[0].lower() == "all")) and "{band}" not in cfg.unc_out:
        raise SystemExit("several bands (--unc-band list or all): --unc-out must be a pattern containing {band}, e.g. OUT/sky_{band}_uncertainty.fits")
    if cfg.unc_cmd == "diff-files":
        g = diff_products(cfg.unc_file_a, cfg.unc_file_b, cfg.unc_out, mode=cfg.unc_diff_mode)
        print(f"difference ({cfg.unc_diff_mode}) of {cfg.unc_file_a} and {cfg.unc_file_b}: implied sigma (global rms) {g:.5g}; written {cfg.unc_out}")
        return

    def log(msg):
        print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)
    model, exps, fset, skies = build_from_args(cfg, log=log)
    st = torch.load(cfg.state, map_location="cpu", weights_only=False)
    A = read_coeffs(model, skies, st, log=log, cfg=cfg)
    bands = resolve_bands(model, spec)
    band = bands[0]
    log(f"bands ({len(bands)}, one solve for all): {bands}")
    if cfg.unc_cmd == "diff":
        if len(bands) != 1:
            raise SystemExit("diff takes one band")
        if not cfg.unc_state_b:
            raise SystemExit("diff needs --unc-state-b")
        B = read_coeffs(model, skies, torch.load(cfg.unc_state_b, map_location="cpu", weights_only=False), log=log, cfg=cfg)
        phi = band_vector(model, band).cpu()
        s = float(cfg.unc_resolution) if cfg.unc_resolution != "auto" else 2.0 * model.sky.pitch
        d = difference_map(A.cpu(), B.cpu(), phi, model.sky.pitch, s, mode=cfg.unc_diff_mode)
        res = {"image": d["h"], "sigma": d["sigma_local"], "s_star": torch.full_like(d["h"], float("nan")), "covered": torch.ones_like(d["h"], dtype=torch.bool),
               "sigma_scales": d["sigma_local"][None], "resolution": s, "resolution_median": float("nan"), "corr_length": float("nan"), "rho": float("nan"),
               "cg_iterations": [], "scales": [s]}
        p = write_product(cfg.unc_out, res, band, model.sky.pitch, {"MODE": cfg.unc_diff_mode, "SIGGLOB": d["sigma_global"]})
        log(f"difference ({cfg.unc_diff_mode}) at {s:.4f} arcsec: implied sigma (global rms) {d['sigma_global']:.5g}; written {p} (SKY = difference h)")
        return
    # the sky used is the saved one: --state loaded above (the op reads model.sky.coeffs())
    plan = dict(method=cfg.unc_method, polish=bool(cfg.unc_polish), n_samples=cfg.unc_samples, maxiter=cfg.unc_maxiter, n_exact=cfg.unc_exact,
                n_bands=len(bands))
    cuda = torch.device(model.dev).type == "cuda"
    if cfg.unc_cmd == "lab":
        budget = cfg.unc_max_minutes if cfg.unc_max_minutes and cfg.unc_max_minutes > 0 else 20.0
        log(f"solver lab: training pixels within +-{cfg.unc_lab_box} arcsec of the centre, {cfg.unc_lab_iters} iterations per variant, variants "
            f"{cfg.unc_lab_variants}; budget {budget:g} min from the start")
        recs = run_lab(model, exps, cfg, log=log, box=cfg.unc_lab_box, iters=cfg.unc_lab_iters, variants=cfg.unc_lab_variants,
                       deadline=T0 + 60.0 * budget - 90.0)
        outp = os.path.join(os.path.dirname(os.path.abspath(cfg.unc_out.format(band="lab"))), "solver_lab.json")
        json.dump(recs, open(outp, "w"), indent=1, default=str)
        log(f"solver lab done in {(time.perf_counter() - T0) / 60:.1f} min; records written to {outp}")
        return
    op = SkyNormalOperator(model, exps, cfg, rho=cfg.unc_prior, tile=cfg.unc_tile or None, chunk=cfg.unc_chunk or None, log=log,
                           batch_px=cfg.unc_batch_px if (explicit_batch or cuda) else None, plan=plan, precond_mode=cfg.unc_precond,
                           n_regions=cfg.unc_regions, n_probes=cfg.unc_probes, precision=cfg.unc_precision,
                           bkg_basis=cfg.unc_bkg_basis if cfg.unc_precision == "mixed" else "raw", bkg_rcond=cfg.unc_bkg_rcond)
    acc = cfg.unc_accept_tol if cfg.unc_accept_tol and cfg.unc_accept_tol > 0 else None
    op_full = op
    if cfg.unc_restrict and cfg.unc_restrict > 0:
        f_ = int(round(cfg.unc_restrict / model.sky.pitch))
        if f_ < 1 or abs(f_ * model.sky.pitch - cfg.unc_restrict) > 1e-6 * max(cfg.unc_restrict, 1e-9) + 1e-9:
            raise SystemExit(f"--unc-restrict {cfg.unc_restrict}: must be a multiple of the sky pitch {model.sky.pitch}")
        op = RestrictedOperator(op_full, factor=f_, log=log)
    if cfg.unc_check:
        fails = operator_checks(op_full, log=log, label="full")["failed"]
        if op is not op_full:
            fails += operator_checks(op, log=log, label="restricted")["failed"]
        if fails:
            log("ABORT: the operator checks failed: " + "; ".join(fails) + ". No solve was run (a conjugate-gradient solve on this operator would "
                "break down). See SOLVER_DIAGNOSIS_2026-11-19.md section 10 for the options.")
            raise SystemExit("uncertainty: operator checks failed: " + "; ".join(fails))
    pl = plan_passes(op.K, **plan)
    mon = Monitor(op, log=log, max_minutes=cfg.unc_max_minutes, t0=T0, planned_apps=pl["setup"] + pl["polish"] + pl["draws"] + pl["exact"],
                  reserve=60.0 + 30.0 * len(bands))            # time kept for the per-band read-outs (Fourier eigendecomposition, smoothing) and the files
    if cfg.unc_max_minutes:
        log(f"time budget {cfg.unc_max_minutes:g} min from the start of the run (reserve {mon.reserve:.0f} s for writing)")
    shared = solve_shared(op, n_samples=cfg.unc_samples, method=cfg.unc_method, polish=bool(cfg.unc_polish), tol=cfg.unc_tol, maxiter=cfg.unc_maxiter,
                          log=log, monitor=mon, accept_tol=acc)
    for b in bands:
        res = band_readout(model, op, shared, b, resolution=cfg.unc_resolution, alpha_max=cfg.unc_alpha, n_exact=cfg.unc_exact, log=log, monitor=mon,
                           rho=cfg.unc_prior, tol=cfg.unc_tol)
        out = cfg.unc_out.format(band=str(b).replace(":", "")) if "{band}" in cfg.unc_out else cfg.unc_out
        p = write_product(out, res, b, model.sky.pitch, {"STATE": os.path.basename(cfg.state)[:60]})
        cov = res["covered"]
        log(f"band {b}: product resolution {res['resolution']:.4f} arcsec (median supported {res['resolution_median']:.4f}); median sigma {float(res['sigma'][cov].median()):.4g}; "
            f"correlation length {res['corr_length']:.4f} arcsec; draws / circulant sigma {res['draws_over_circulant']:.3f}; CG iterations {res['cg_iterations']}; "
            f"{res['n_apply']} operator applications at {res['seconds_per_apply']:.2f} s"
            + (f"; TRUNCATED in '{res['truncated_in']}', relative residual {res['residual']:.2e}" if res["truncated"] else "") + f"; written {p}")
        if res["exact"]:
            log("exact / sampled sigma: " + json.dumps(res["exact"]))
    log(f"done: {len(bands)} band(s), {op.n_apply} applications + {op.n_forward} forward-only pass, {mon.elapsed() / 60:.1f} min"
        + (" (solve truncated by the time budget)" if shared["truncated"] else "")
        + ("; the Gauss-Newton step did NOT CONVERGE: SKY is NOT polished" if shared.get("polish_converged") is False else ""))


LAB_VARIANTS = "jacobi64,jacobi64_orth,bjacobi64_orth,filter64,filter64_orth,restrict64_orth"   # one operator, held once


def run_lab(model, exps, cfg, log=print, box=2.0, iters=25, variants=LAB_VARIANTS, factor=None, deadline=None, clock=time.perf_counter):
    """Solver lab (19 Nov, second part of the report): the operator restricted to the training pixels within +-box arcsec of the box centre (one
    application takes seconds), and short solves of the Gauss-Newton step with each variant:
      none64 / jacobi64     no preconditioner / diagonal (on the 'filter' operator), float64 vectors and sums
      fourier32             the 18 Nov preconditioner, float32 throughout (what ran on 18-19 Nov)
      filter32 / filter64   the 'filter' preconditioner, float32 throughout / float64 vectors, sums and background blocks
      restrict64            the restricted operator (band images at factor x the sky pitch, default 0.08 arcsec) on the 'filter' operator
      *_nobright            the same without the pixels whose variance is mostly the noise floor ((f S)^2 > 0.5 sigma^2)
    For each operator: the checks of operator_checks in float32 and in float64; for each variant: the relative residual at iterations 1, 2, 5,
    10, 15, 20, 25 (or iters), the Lanczos range, any breakdown, and the time. Returns a list of records."""
    names = [v.strip() for v in str(variants).split(",") if v.strip()]
    factor = factor or max(2, int(round(0.08 / model.sky.pitch)))
    ops, recs = {}, []
    kw = dict(rho=cfg.unc_prior, tile=cfg.unc_tile or None, chunk=cfg.unc_chunk or None, batch_px=cfg.unc_batch_px, region_box=box, log=log)

    def get(kind):
        if kind not in ops:
            for k_ in list(ops):                  # one operator on the device at a time (a 'filter' operator holds about 4 GB at SMACS size)
                del ops[k_]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"lab: building the '{kind}' operator (training pixels within +-{box} arcsec of the centre)")
            op = SkyNormalOperator(model, exps, cfg, precond_mode="fourier" if kind == "fourier" else "filter", precision="mixed",
                                   exclude_bright=0.5 if kind.endswith("nobright") else 0.0, bkg_rcond=getattr(cfg, "unc_bkg_rcond", 1e-4), **kw)
            if kind.endswith("nobright"):
                log(f"lab: {op.n_bright_excluded} pixels excluded as floor-dominated")
            log(f"lab: background blocks: condition number max {max(op.bb_cond):.3e}, median {float(np.median(op.bb_cond)):.3e}, "
                f"{sum(c > 1e7 for c in op.bb_cond)} of {len(op.bb_cond)} above 1e7")
            for prec in ("float32", "mixed"):
                op.set_precision(prec)
                recs.append({"operator": kind, "precision": prec, "checks": operator_checks(op, log=log, label=f"{kind}, {prec}")})
            ops[kind] = op
        return ops[kind]
    for name in names:
        if deadline is not None and clock() > deadline:
            log(f"lab: variant {name} SKIPPED (time budget)")
            continue
        kind = "fourier" if name.startswith("fourier") else "filter"
        if "_nobright" in name:
            kind += "_nobright"
        op = get(kind)
        op.set_precision("float32" if "32" in name else "mixed")
        op.set_bkg_basis("orth" if "_orth" in name else "raw")
        base_ = name.split("_")[0].rstrip("0123456789")
        op.precond_mode = {"none": "none", "jacobi": "jacobi", "bjacobi": "bjacobi"}.get(base_, op.setup_mode)
        target = op
        if base_ == "restrict":
            target = RestrictedOperator(op, factor=factor, log=log)
        if "_orth" in name or base_ == "restrict":
            operator_checks(target, log=log, label=name)
        t0, n0 = clock(), op.n_apply
        b = target.residual_rhs()
        x, info = pcg(target, b, tol=1e-12, maxiter=iters, keep_vectors=True)
        modes = {}
        try:
            for lab_, i_ in (("top", -1), ("bottom", 0)):
                val, y_ = ritz_vectors(info, which=(i_,))[0]
                modes[lab_] = {"ritz": val, **mode_report(target, y_)}
        except Exception as ex:                    # a report problem must not stop the lab
            modes = {"error": repr(ex)}
        info.pop("lanczos_vectors", None)
        h = info["history_all"]
        pts = [i for i in (1, 2, 5, 10, 15, 20, 25, 30, 40, 50) if i <= len(h)]
        rec = {"variant": name, "residual": {i: h[i - 1] for i in pts}, "final": info["rel_residual"], "iterations": info["iterations"],
               "ritz": [info["ritz_min"], info["ritz_max"]], "breakdown": info["breakdown"], "applications": op.n_apply - n0, "seconds": clock() - t0,
               "rel_err_F_est": info["rel_err_F_est"], "modes": modes}
        recs.append(rec)
        for lab_, m_ in modes.items():
            if isinstance(m_, dict):
                log(f"lab variant {name}: {lab_} Ritz value {m_['ritz']:.3e}: sky share {m_['sky_share']:.3f}, background share {m_['background_share']:.3f} "
                    f"(largest exposures {m_['background_top_exposures']}; per term {m_['background_per_term']}); sky per coefficient or band "
                    f"{m_['sky_per_coefficient']}")
        log(f"lab variant {name}: relative residual " + ", ".join(f"{i}: {h[i - 1]:.3e}" for i in pts)
            + f"; Lanczos {info['ritz_min']:.3e} to {info['ritz_max']:.3e}; estimated relative F-norm error {info['rel_err_F_est']:.2e}; "
            + (f"BREAKDOWN: {info['breakdown']}; " if info["breakdown"] else "") + f"{rec['applications']} applications, {rec['seconds']:.0f} s")
        op.precond_mode = op.setup_mode
        op.set_precision("mixed")
        op.set_bkg_basis("raw")
    return recs


def resolve_bands(model, spec):
    """--unc-band: list of names / indices / coef:k, or 'all' (every filter of the model, by name)."""
    if len(spec) == 1 and spec[0].lower() == "all":
        names = getattr(model.fset, "names", None)
        return list(names) if names is not None else list(range(model.Phi_band.shape[0]))
    return [int(b) if b.isdigit() else b for b in spec] or [0]


if __name__ == "__main__":
    main(sys.argv[1:])
