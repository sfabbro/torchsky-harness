"""Sky s(Omega, lambda): the only function of sky position (F1-F3).

s(X, Y, lambda) = sum_k phi_k(lambda) A_k(X, Y),  k = 1..K (K = 12 RBF bumps uniform in log lambda, partition of unity; v40's
RBFSpectralBasis without the learned correction).  A_k is a grid of coefficients at pitch `pitch` [arcsec]; the CONTINUOUS sky is the
bilinear (tent) interpolant of the grid, zero outside it.  Both renderers read this same continuous function.

Coordinates: tangent plane at the box centre, arcsec, X along the north-up target pixel x axis (x to the WEST, as the target header
CD1_1 < 0 of v40 / register_xcorr), Y to the north.  Grid node (i, j) sits at X = (j - (N-1)/2) pitch, Y = (i - (N-1)/2) pitch.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LAM_LO, LAM_HI = 0.70, 5.30  # um: reference spectral domain of the sky field (as v40); the domain in force is SPECTRAL_DOMAIN
SPECTRAL_DOMAIN = [LAM_LO, LAM_HI]   # set from the loaded filter curves by instruments.spectral_setup (audit finding 4); the reference by default


def set_spectral_domain(lo, hi):
    """Spectral domain [um] of the sky basis and of every wavelength coordinate of the networks (lam_unit). Plain module state, like ITEM_CHUNK."""
    SPECTRAL_DOMAIN[0], SPECTRAL_DOMAIN[1] = float(lo), float(hi)


def lam_unit(lam_um):
    lo, hi = SPECTRAL_DOMAIN
    return (torch.log(lam_um) - math.log(lo)) / (math.log(hi) - math.log(lo))


class RBFBasis(nn.Module):
    """phi_k(lambda): K normalised Gaussian bumps uniform in log lambda (partition of unity). Equal to v40 RBFSpectralBasis(k, learned=False)."""

    def __init__(self, k=12):
        super().__init__()
        self.k = k
        self.register_buffer("c", torch.linspace(0.0, 1.0, k))
        self.width = 1.0 / (k - 1)

    def forward(self, lam_um):  # (M,) -> (M, K)
        u = lam_unit(lam_um)
        g = torch.exp(-0.5 * ((u[:, None] - self.c[None]) / self.width) ** 2)
        return g / g.sum(1, keepdim=True)


def spectral_init(S_band, Phi_band, mu=1e-5):
    """Coefficients per pixel from band images (nb, H, W) and band-integrated basis Phi_band (nb, K): ridge least squares with a second
    difference prior across k (v40 spectral_init). Returns (K, H, W)."""
    with torch.no_grad():
        k = Phi_band.shape[1]
        D = torch.zeros(k - 2, k, dtype=torch.float64)
        for i in range(k - 2):
            D[i, i:i + 3] = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        P64 = Phi_band.double().cpu()
        M = P64.T @ P64 + mu * D.T @ D + 1e-6 * torch.eye(k, dtype=torch.float64)
        P = torch.linalg.solve(M, P64.T).to(S_band.device, S_band.dtype)  # K nb
        return torch.einsum("kb,bhw->khw", P, S_band)


class SkyGrid(nn.Module):
    """Spectral grid sky with optional cross-fitting views (V copies of the coefficients)."""
    ped = None        # --sky-positivity-margin: constant spectral coefficients p (K,); the positivity penalty acts on sky + p

    def __init__(self, A0, pitch, basis, n_views=1, precond=None):
        """precond (N, N): optional fixed diagonal preconditioner, coefficients = precond x A (Adam then moves bright pixels proportionally
        faster; F3). None = identity."""
        super().__init__()
        K, N, N2 = A0.shape
        assert N == N2, "square sky grid"
        self.N, self.K, self.pitch, self.n_views = N, K, float(pitch), n_views
        self.register_buffer("pre", torch.ones(N, N, device=A0.device, dtype=A0.dtype) if precond is None else precond.to(A0.dtype))
        self.A = nn.Parameter((A0 / self.pre)[None].repeat(n_views, 1, 1, 1).clone())
        self.basis = basis
        self.x0 = -(N - 1) / 2 * self.pitch  # coordinate of node 0 (both axes)

    @property
    def half_extent(self):
        return (self.N - 1) / 2 * self.pitch

    def n_fitted(self):
        """Fitted sky numbers (one view)."""
        return self.K * self.N * self.N

    @torch.no_grad()
    def reset_from_coeffs(self, A0):
        """Warm start (rebuilt by the registration pre-pass): every view = A0 / preconditioner."""
        self.A.data[:] = (A0 / self.pre)[None]

    def coeffs(self, view=None):
        if view is None:
            return (self.A.mean(0) if self.n_views > 1 else self.A[0]) * self.pre
        return self.A[view] * self.pre

    # --- continuous reads
    def sample_coeffs(self, A, X, Y):
        """Tent interpolation of coefficient grid A (K, N, N) at positions X, Y (any equal shape) -> (K, *shape). Zero outside."""
        shp = X.shape
        h = (self.N - 1) / 2 * self.pitch
        g = torch.stack([X.reshape(-1) / h, Y.reshape(-1) / h], -1)[None, None]
        out = F.grid_sample(A[None], g.to(A.dtype), mode="bilinear", padding_mode="zeros", align_corners=True)[0, :, 0]
        return out.view(A.shape[0], *shp)

    def pooled(self, A, m):
        """Box average of the coefficient grid over m x m blocks (level grids, F3). Returns (K, Nm, Nm) and the coordinate of pooled node 0."""
        if m == 1:
            return A, self.x0
        N = self.N
        Np = int(math.ceil(N / m)) * m
        Ap = F.pad(A, (0, Np - N, 0, Np - N))
        P = F.avg_pool2d(Ap[None], m)[0]
        return P, self.x0 + (m - 1) / 2 * self.pitch

    def band_images(self, Phi_band, view=None):
        """Band images on the grid nodes: (nb, N, N) = sum_k Phi_band[b, k] A_k (quadrature-averaged basis per band)."""
        return torch.einsum("bk,khw->bhw", Phi_band, self.coeffs(view))


# ============================================================================= item sky (--sky-form items)
ITEM_R = 4.0          # patch half-size in units of the larger effective width (fraction of a 2-d Gaussian beyond 4 widths: exp(-8) = 3e-4)
ITEM_HALF = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128)   # patch half-size buckets [nodes]
ITEM_CHUNK = 4_000_000  # elements (items x patch nodes x K) per rasterisation chunk

# --item-raster pyramid (F40): Gaussian-semigroup pyramid, cost per item independent of its size
PYR_BLUR = 1.0        # cascade blur from level l to l-1: s_l = PYR_BLUR x p_l (p_l = 2^l p)
PYR_MIN = 1.0         # an item goes to the coarsest level l whose remaining smaller principal width sqrt(T_min - S_l^2) >= PYR_MIN x p_l
PYR_R = 4.0           # level patch half-size in units of the larger REMAINING width (as ITEM_R)
PYR_TAP = 5.0         # cascade kernel half-length in units of s_l (truncation changes its variance by 1.5e-5 relative)
PYR_HALF = ITEM_HALF + (160, 192, 256, 320, 384, 448, 512)
PYR_SHEAR_GAIN = 1.5  # --item-patch sheared: an item gets the sheared patch only if it has 1.5x fewer nodes than its square patch   # patch buckets of the pyramid mode (direct keeps ITEM_HALF, capped at 128)


def _halton(n, base):
    out = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        f, r, k = 1.0, 0.0, i + 1
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        out[i] = r
    return out


def _gauss_smooth(A, s_nodes):
    """Separable Gaussian smoothing of (K, N, N) by s_nodes [nodes] (zero padding, renormalised by the smoothed coverage)."""
    if s_nodes < 0.3:
        return A
    h = int(math.ceil(3 * s_nodes))
    t = torch.arange(-h, h + 1, device=A.device, dtype=A.dtype)
    g = torch.exp(-0.5 * (t / s_nodes) ** 2)
    g = g / g.sum()
    K = A.shape[0]
    x = A[:, None]
    one = torch.ones_like(x[:1])
    kx, ky = g.view(1, 1, 1, -1), g.view(1, 1, -1, 1)
    num = F.conv2d(F.conv2d(x, kx, padding=(0, h)), ky, padding=(h, 0))
    den = F.conv2d(F.conv2d(one, kx, padding=(0, h)), ky, padding=(h, 0))
    return (num / den.clamp(min=1e-6))[:, 0].view(K, *A.shape[1:])


def item_lattice(n_items, half, n_levels, floor, max_width):
    """Data-blind seeding: coarse-to-fine square lattices over [-half, half]^2. Level l (0 = finest) receives a share of the items
    proportional to 4^-l (3 levels: 76 / 19 / 5 per cent). Level l: an m x m lattice, m = floor(sqrt(n_l)), spacing d = 2 half / m, cell
    centres; the finest level's remainder (n_l - m^2, plus whatever coarser levels could not place) goes on a Halton set. Initial width =
    max(d / 2, floor), capped at max_width, round. Returns x, y, width, level, cell area (all (n,) float64 / long)."""
    w = torch.tensor([4.0 ** -l for l in range(n_levels)], dtype=torch.float64)
    share = (w / w.sum() * n_items).floor().long()
    share[0] += n_items - int(share.sum())
    xs, ys, ws, lv, area = [], [], [], [], []
    carry = 0
    for l in range(n_levels - 1, -1, -1):          # coarse first; unplaced items carried to the next finer level
        nl = int(share[l]) + carry
        m = int(math.floor(math.sqrt(nl)))
        if m < 1:
            carry = nl
            continue
        d = 2 * half / m
        g = (torch.arange(m, dtype=torch.float64) + 0.5) * d - half
        YY, XX = torch.meshgrid(g, g, indexing="ij")
        n_lat = m * m
        rem = nl - n_lat
        if l > 0:
            carry = rem
            rem = 0
        else:
            carry = 0
        x, y = XX.flatten(), YY.flatten()
        if rem > 0:
            x = torch.cat([x, (_halton(rem, 2) * 2 - 1) * half])
            y = torch.cat([y, (_halton(rem, 3) * 2 - 1) * half])
        n_here = x.numel()
        a = (2 * half) ** 2 / n_here             # area represented by one item of this level
        dd = math.sqrt(a)
        xs.append(x)
        ys.append(y)
        ws.append(torch.full((n_here,), min(max(0.5 * dd, floor), max_width), dtype=torch.float64))
        lv.append(torch.full((n_here,), l, dtype=torch.long))
        area.append(torch.full((n_here,), a, dtype=torch.float64))
    out = [torch.cat(v) for v in (xs, ys, ws, lv, area)]
    assert out[0].numel() == n_items, (out[0].numel(), n_items)
    return out

class _ItemRaster(torch.autograd.Function):
    """Item scatter with a recompute-in-chunks backward pass (first order only)."""

    @staticmethod
    def forward(ctx, pos, wid, ang, c, sky):
        ctx.sky = sky
        ctx.save_for_backward(pos, wid, ang, c)
        return sky._raster_forward(pos, wid, ang, c)

    @staticmethod
    def backward(ctx, gout):
        pos, wid, ang, c = ctx.saved_tensors
        return ctx.sky._raster_vjp(pos, wid, ang, c, gout) + (None,)


class _LevelGrid:
    """One coarse level (l >= 1) of the pyramid rasteriser: square grid of n nodes at pitch p_l, node 0 at x0 (both axes). Items are
    point-sampled with covariance T - S_l^2 I = Sigma + (p^2/12 - S_l^2) I, i.e. the effective variance offset is qoff = p^2/12 - S_l^2
    (negative), on patches of +-PYR_R larger remaining widths, normalised so that p_l^2 sum_patch = 1. The scatter and its recompute-in-chunks
    backward pass are ItemSky's own (_raster_forward / _raster_vjp), reused through _ItemRaster."""

    def __init__(self, x0, pitch, n, K, qoff):
        self.x0, self.pitch, self.N, self.K, self.qoff = float(x0), float(pitch), int(n), int(K), float(qoff)

    def _buckets(self, wid, ang=None):
        p = self.pitch
        need = torch.ceil(PYR_R * torch.sqrt((wid.detach().amax(1) ** 2 + self.qoff).clamp(min=0)) / p).long()
        hs = torch.tensor(PYR_HALF, device=need.device)
        bucket = torch.searchsorted(hs, need.clamp(max=PYR_HALF[-1]))
        for bi in torch.unique(bucket).tolist():
            h = PYR_HALF[bi]
            idx = torch.nonzero(bucket == bi).squeeze(1)
            Pn = 2 * h + 1
            chunk = max(1, ITEM_CHUNK // (Pn * Pn * max(self.K, 1)))
            for s0 in range(0, idx.numel(), chunk):
                yield h, idx[s0:s0 + chunk]

    def _patch(self, pos, wid, ang, h):
        N, p = self.N, self.pitch
        ia, ib = 1.0 / (wid[:, 0] ** 2 + self.qoff), 1.0 / (wid[:, 1] ** 2 + self.qoff)
        cth, sth = torch.cos(ang), torch.sin(ang)
        Aq = cth * cth * ia + sth * sth * ib
        Cq = sth * sth * ia + cth * cth * ib
        Bq = cth * sth * (ia - ib)
        off = torch.arange(-h, h + 1, device=pos.device)
        x, y = pos[:, 0], pos[:, 1]
        jx = torch.round((x.detach() - self.x0) / p).long()[:, None] + off[None]
        jy = torch.round((y.detach() - self.x0) / p).long()[:, None] + off[None]
        dx = (self.x0 + jx.to(x.dtype) * p) - x[:, None]
        dy = (self.x0 + jy.to(y.dtype) * p) - y[:, None]
        arg = (0.5 * Aq[:, None, None] * dx[:, None, :] ** 2 + 0.5 * Cq[:, None, None] * dy[:, :, None] ** 2
               + Bq[:, None, None] * dx[:, None, :] * dy[:, :, None])
        g = torch.exp(-arg)
        g = g / (g.sum((1, 2), keepdim=True) * p * p)
        okx = (jx >= 0) & (jx < N)
        oky = (jy >= 0) & (jy < N)
        flat = torch.where(oky[:, :, None] & okx[:, None, :], jy[:, :, None] * N + jx[:, None, :],
                           torch.full_like(jy[:, :, None] * jx[:, None, :], N * N))
        return g.reshape(g.shape[0], -1), flat.reshape(g.shape[0], -1)


class _ShearGrid(_LevelGrid):
    """Same as _LevelGrid (any level, including level 0 with qoff = p^2/12), but the patch is a SHEARED box in the item's decorrelated
    coordinates instead of a square of +-PYR_R larger widths (--item-patch sheared, F40): rows |dy| <= hy p with hy = ceil(PYR_R sqrt(V_yy) / p),
    and in each row the columns round(x + (V_xy / V_yy) dy) +- hx with hx = ceil(PYR_R sqrt(det V / V_yy) / p) (V = the covariance sampled
    at this level; x | y has mean x + V_xy / V_yy dy and variance det V / V_yy). The quadratic form is a sum of the two decorrelated squares, so
    the box holds the whole PYR_R-sigma ellipse, and the node count is ~ (2 PYR_R)^2 sqrt(det V) / p^2, proportional to the item's area
    instead of its larger width squared. Same normalisation (p^2 sum_patch = 1)."""

    def _hw(self, wid, ang):
        va = (wid[:, 0].detach() ** 2 + self.qoff).clamp(min=0)
        vb = (wid[:, 1].detach() ** 2 + self.qoff).clamp(min=0)
        c, s = torch.cos(ang.detach()), torch.sin(ang.detach())
        Vyy = s * s * va + c * c * vb
        Vxy = c * s * (va - vb)
        return Vyy, Vxy, va * vb / Vyy

    def _buckets(self, wid, ang=None):
        p = self.pitch
        Vyy, _, cx = self._hw(wid, ang)
        hs = torch.tensor(PYR_HALF, device=wid.device)
        by = torch.searchsorted(hs, torch.ceil(PYR_R * torch.sqrt(Vyy) / p).long().clamp(max=PYR_HALF[-1]))
        bx = torch.searchsorted(hs, torch.ceil(PYR_R * torch.sqrt(cx) / p).long().clamp(max=PYR_HALF[-1]))
        need = torch.ceil(PYR_R * torch.sqrt((wid.detach().amax(1) ** 2 + self.qoff).clamp(min=0)) / p).long()
        bs = torch.searchsorted(hs, need.clamp(max=PYR_HALF[-1]))
        n_sq = (2 * hs[bs] + 1) ** 2
        n_sh = (2 * hs[by] + 1) * (2 * hs[bx] + 1)
        nb = len(PYR_HALF)
        # the sheared patch costs more per node (per-row column indices): used only where it saves > PYR_SHEAR_GAIN of the nodes
        key = torch.where(n_sh.double() * PYR_SHEAR_GAIN < n_sq.double(), by * nb + bx, nb * nb + bs)
        for kk in torch.unique(key).tolist():
            idx = torch.nonzero(key == kk).squeeze(1)
            if kk >= nb * nb:
                h = PYR_HALF[kk - nb * nb]
                h2 = (h, h)
            else:
                h = h2 = (PYR_HALF[kk // nb], PYR_HALF[kk % nb])
            chunk = max(1, ITEM_CHUNK // ((2 * h2[0] + 1) * (2 * h2[1] + 1) * max(self.K, 1)))
            for s0 in range(0, idx.numel(), chunk):
                yield h, idx[s0:s0 + chunk]

    def _patch(self, pos, wid, ang, h):
        if not isinstance(h, tuple):                      # square patch (items for which shearing saves little)
            return _LevelGrid._patch(self, pos, wid, ang, h)
        hy, hx = h
        N, p = self.N, self.pitch
        ia, ib = 1.0 / (wid[:, 0] ** 2 + self.qoff), 1.0 / (wid[:, 1] ** 2 + self.qoff)
        cth, sth = torch.cos(ang), torch.sin(ang)
        Aq = cth * cth * ia + sth * sth * ib
        Cq = sth * sth * ia + cth * cth * ib
        Bq = cth * sth * (ia - ib)
        Vyy, Vxy, _ = self._hw(wid, ang)
        offy = torch.arange(-hy, hy + 1, device=pos.device)
        offx = torch.arange(-hx, hx + 1, device=pos.device)
        x, y = pos[:, 0], pos[:, 1]
        jy = torch.round((y.detach() - self.x0) / p).long()[:, None] + offy[None]             # (m, Py)
        dy = (self.x0 + jy.to(y.dtype) * p) - y[:, None]
        xc = x.detach()[:, None] + (Vxy / Vyy)[:, None] * dy.detach()                       # conditional mean of x in each row
        jx = torch.round((xc - self.x0) / p).long()[:, :, None] + offx[None, None]           # (m, Py, Px)
        dx = (self.x0 + jx.to(x.dtype) * p) - x[:, None, None]
        arg = (0.5 * Aq[:, None, None] * dx ** 2 + 0.5 * Cq[:, None, None] * dy[:, :, None] ** 2 + Bq[:, None, None] * dx * dy[:, :, None])
        g = torch.exp(-arg)
        g = g / (g.sum((1, 2), keepdim=True) * p * p)
        ok = (jx >= 0) & (jx < N) & ((jy >= 0) & (jy < N))[:, :, None]
        flat = torch.where(ok, jy[:, :, None] * N + jx, torch.full_like(jx, N * N))
        return g.reshape(g.shape[0], -1), flat.reshape(g.shape[0], -1)


class PyramidPlan:
    """Geometry of the Gaussian-semigroup pyramid for a fine grid of N nodes at pitch p, node 0 at x0 (F40).

    Level l (0 = the fine grid) has pitch p_l = 2^l p. Going from level l to l-1, every level-l node deposits a discretely normalised
    Gaussian of width s_l = PYR_BLUR p_l onto level l-1 (stride-2 transposed convolution, separable, taps +-ceil(PYR_TAP s_l / p_{l-1}),
    1-d weights 2 g_t / sum g, so flux p_l^2 a is conserved). S_l^2 = sum_{j<=l} s_j^2 is the blur variance from level l to level 0.
    Level l (l >= 1) has a margin of E_l nodes beyond the fine grid's extent, E_l = ceil(reach_l / p_l) + 1 where reach_l = sum_{j<=l}
    (tap radius_j x p_{j-1}) is the farthest a level-l node can deposit on level 0; so everything that can reach the fine grid is kept and
    the fine-grid result equals that of unbounded coarse grids. Item thresholds: level l takes items with T_min >= thr_l = (PYR_MIN p_l)^2
    + S_l^2 (T_min = smaller principal variance of T = Sigma + p^2/12 I); the coarsest such level is chosen."""

    def __init__(self, N, pitch, x0, max_var):
        self.N, self.p, self.x0 = int(N), float(pitch), float(x0)
        self.pl, self.s, self.S2, self.thr, self.taps, self.rad, self.E, self.n, self.org = [self.p], [0.0], [0.0], [0.0], [None], [0], [0], \
            [self.N], [self.x0]
        reach = 0.0
        l = 0
        while True:
            l += 1
            pl = self.p * 2 ** l
            s = PYR_BLUR * pl
            S2 = self.S2[-1] + s * s
            thr = (PYR_MIN * pl) ** 2 + S2
            if thr > max_var:
                break
            r = int(math.ceil(PYR_TAP * s / (pl / 2)))
            reach += r * pl / 2
            E = int(math.ceil(reach / pl)) + 1
            n = int(math.ceil((self.N - 1) * self.p / pl)) + 2 * E + 1
            self.pl.append(pl); self.s.append(s); self.S2.append(S2); self.thr.append(thr); self.rad.append(r); self.E.append(E)
            self.n.append(n); self.org.append(self.x0 - E * pl)
        self.L = len(self.pl) - 1                   # number of coarse levels (0 = pyramid not used)

    def assign(self, wid):
        """Level of each item (detached widths (n, 2)): coarsest l with T_min >= thr_l, else 0."""
        if self.L == 0:
            return torch.zeros(wid.shape[0], dtype=torch.long, device=wid.device)
        tmin = wid.detach().amin(1).double() ** 2 + self.p ** 2 / 12.0
        thr = torch.tensor(self.thr[1:], dtype=torch.float64, device=wid.device)
        return torch.searchsorted(thr, tmin, right=True)        # number of thresholds <= tmin

    def level_grid(self, l, K, shear=False):
        return (_ShearGrid if shear else _LevelGrid)(self.org[l], self.pl[l], self.n[l], K, self.p ** 2 / 12.0 - self.S2[l])

    def kernel(self, l, dtype, device):
        r = self.rad[l]
        t = torch.arange(-r, r + 1, dtype=torch.float64) * (self.pl[l] / 2)
        g = torch.exp(-0.5 * (t / self.s[l]) ** 2)
        return (2.0 * g / g.sum()).to(device=device, dtype=dtype)

    def up(self, G, l):
        """Level-l grid (K, n_l, n_l) -> its deposit on level l-1 (K, n_{l-1}, n_{l-1}). Ordinary autograd."""
        K = G.shape[0]
        k = self.kernel(l, G.dtype, G.device)
        T = k.numel()
        r = self.rad[l]
        y = F.conv_transpose2d(G[:, None], k.view(1, 1, 1, T), stride=(1, 2))
        y = F.conv_transpose2d(y, k.view(1, 1, T, 1), stride=(2, 1))[:, 0]   # (K, 2(n-1)+T, 2(n-1)+T); out index o <-> node o + c - r
        c = self.E[l - 1] - 2 * self.E[l]                                    # level-(l-1) index of level-l node 0
        start = r - c
        m = self.n[l - 1]
        Lt = y.shape[-1]
        return F.pad(y, (-start, start + m - Lt, -start, start + m - Lt))


class ItemSky(nn.Module):
    """Sky made of Gaussian items, RASTERISED onto the same coefficient grid as SkyGrid:

        s(X, Y, lambda) = sum_n [sum_k c_nk phi_k(lambda)] G(X - x_n, Y - y_n; Sigma_n)

    Item n: position (x_n, y_n) [arcsec, continuous], widths w_i = sqrt(floor^2 + exp(2 l_i)) (never below --item-floor, capped at
    --item-max-width), angle theta_n, coefficients c_nk = exp(r_nk) > 0 on the package's basis phi_k (so s >= 0 at every lambda from the items).

    Rasterisation (approximate cell integral): A_k(node) = sum_n c_nk g_n(node), g_n = the Gaussian with covariance Sigma_n + (p^2 / 12) I (the
    pixel box replaced by the isotropic Gaussian of equal variance, which commutes with the rotation) evaluated at the node, on a local patch
    of +-ITEM_R effective widths, renormalised so that p^2 sum_patch g_n = 1 (exact flux, including the nodes that fall outside the grid, which
    are then dropped: the sky is zero outside, as for SkyGrid). Scatter-added with index_add (differentiable in every item parameter).
    Everything downstream (tent read, pooled levels, FFT and quadrature renders) reads coeffs() exactly as for SkyGrid, so the sky is STILL
    read through a tent grid at the sky pitch: what changes is the parametrisation, not the renderer.

    raster='direct' (default) is the above, unchanged; raster='pyramid' (F40) rasterises items wider than ~2.8 pitches on coarser grids with
    the covariance left after a fixed Gaussian cascade (PyramidPlan), so the cost per item does not grow with its size; patch='sheared'
    (pyramid only) also makes it grow with the item's area rather than its larger width squared."""

    sample_coeffs = SkyGrid.sample_coeffs
    pooled = SkyGrid.pooled
    band_images = SkyGrid.band_images
    half_extent = SkyGrid.half_extent

    def __init__(self, A0, pitch, basis, n_items, floor=0.0, max_width=0.0, n_levels=3, n_views=1, coef_min=None,
                 seed=0, raster="direct", patch="square", impl="prototype"):
        super().__init__()
        assert impl in ("prototype", "torchsky"), impl
        self.item_impl = impl            # --item-impl: "torchsky" rasterises through torchsky's GaussianItemScene (tsky_backends.py)
        assert raster in ("direct", "pyramid"), raster
        assert patch in ("square", "sheared") and (patch == "square" or raster == "pyramid"), (raster, patch)
        self.raster_mode = raster
        self.patch_shape = patch
        self._halves = ITEM_HALF if raster == "direct" else PYR_HALF   # level-0 patch buckets (pyramid: larger buckets above 128 nodes)
        self._plan = None
        K, N, N2 = A0.shape
        assert N == N2, "square sky grid"
        self.N, self.K, self.pitch, self.n_views = N, K, float(pitch), n_views
        self.basis = basis
        self.x0 = -(N - 1) / 2 * self.pitch
        self.n_items, self.n_levels = int(n_items), int(n_levels)
        self.floor = float(floor) if floor > 0 else 0.4 * self.pitch
        self.max_width = float(max_width) if max_width > 0 else 16.0 * self.pitch
        self.seed = seed
        dev, dt = A0.device, A0.dtype
        self.pos = nn.Parameter(torch.zeros(n_views, self.n_items, 2, device=dev, dtype=dt))
        self.lw = nn.Parameter(torch.zeros(n_views, self.n_items, 2, device=dev, dtype=dt))
        self.ang = nn.Parameter(torch.zeros(n_views, self.n_items, device=dev, dtype=dt))
        self.lc = nn.Parameter(torch.zeros(n_views, self.n_items, K, device=dev, dtype=dt))
        self.register_buffer("level", torch.zeros(self.n_items, dtype=torch.long, device=dev))
        self.ped = None   # --sky-positivity-margin: constant spectral coefficients p (K,); the items carry sky + p, the sky is items - p
        self.coef_min = coef_min
        self._cache = None
        self._mut = 0
        self.reset_from_coeffs(A0)

    # ------------------------------------------------------------------ bookkeeping
    def n_fitted(self):
        """Fitted sky numbers (one view): items x (2 position + 2 width + 1 angle + K)."""
        return self.n_items * (5 + self.K)

    def widths(self, view):
        lw = self.lw[view].clamp(max=math.log(self.max_width))
        return torch.sqrt(self.floor ** 2 + torch.exp(2 * lw))

    @torch.no_grad()
    def reset_from_coeffs(self, A0):
        """(Re)seed: lattice positions and widths (data-blind), coefficients read from the warm-start coadd coefficients A0 (K, N, N): level l
        gets the share of the items it holds, c_nk = (n_l / n) x (cell area of its items) x max(A0_k smoothed at the item width, coef_min);
        Used at construction and when the registration pre-pass rebuilds the warm start."""
        if getattr(self, "ped", None) is not None:
            A0 = A0 + self.ped.to(A0)[:, None, None]
        x, y, w, lv, area = item_lattice(self.n_items, self.half_extent, self.n_levels, self.floor, self.max_width)
        dev, dt = self.pos.device, self.pos.dtype
        cmin = self.coef_min
        if cmin is None:
            cmin = 1e-3 * float(A0.abs().amax(0).median().clamp(min=1e-12)) + 1e-12
        c = torch.zeros(self.n_items, self.K, dtype=dt, device=dev)
        for l in range(self.n_levels):
            sel = lv == l
            if not bool(sel.any()):
                continue
            wl = float(w[sel][0])
            As = _gauss_smooth(A0, wl / self.pitch)
            X, Y = x[sel].to(dev, dt), y[sel].to(dev, dt)
            vals = self.sample_coeffs(As, X, Y).T                          # (n_l, K)
            frac = float(sel.sum()) / self.n_items
            c[sel.to(dev)] = frac * area[sel].to(dev, dt)[:, None] * vals.clamp(min=cmin)
        rng = torch.Generator().manual_seed(1234 + self.seed)
        ang = torch.rand(self.n_items, generator=rng, dtype=torch.float64) * math.pi
        lw0 = 0.5 * torch.log((w ** 2 - self.floor ** 2).clamp(min=(0.05 * self.floor) ** 2))
        self.pos.data[:] = torch.stack([x, y], -1).to(dev, dt)[None]
        self.lw.data[:] = lw0[:, None].repeat(1, 2).to(dev, dt)[None]
        self.ang.data[:] = ang.to(dev, dt)[None]
        self.lc.data[:] = torch.log(c.clamp(min=1e-30))[None]
        self.level.copy_(lv.to(dev))
        self._mut += 1

    def opt_groups(self, cfg):
        """Adam parameter groups with their own step sizes (each scaled by the run's schedule through 'base')."""
        p = self.pitch
        gs = [("pos", self.pos, cfg.item_lr_pos * p), ("lw", self.lw, cfg.item_lr_width), ("ang", self.ang, cfg.item_lr_angle),
              ("lc", self.lc, cfg.item_lr_coef)]
        return [{"params": [t], "lr": lr, "base": lr, "name": n} for n, t, lr in gs]

    # ------------------------------------------------------------------ rasterisation
    def _key(self, view):
        ps = [self.pos, self.lw, self.ang, self.lc]
        # version counters miss writes through .data, so the key also holds two content checksums per parameter
        with torch.no_grad():
            sums = tuple((float(t.sum()), float((t * t).sum())) for t in ps)
        return (view, self._mut, str(self.pos.device)) + tuple((t._version, t.data_ptr()) for t in ps) + sums

    def _buckets(self, wid, ang=None):
        """Index sets of items sharing a patch half-size (from the larger effective width, detached), cut in chunks of ITEM_CHUNK elements."""
        p = self.pitch
        q = p * p / 12.0
        need = torch.ceil(ITEM_R * torch.sqrt(wid.detach().amax(1) ** 2 + q) / p).long()
        halves = self._halves
        hs = torch.tensor(halves, device=need.device)
        bucket = torch.searchsorted(hs, need.clamp(max=halves[-1]))
        for bi in torch.unique(bucket).tolist():
            h = halves[bi]
            idx = torch.nonzero(bucket == bi).squeeze(1)
            Pn = 2 * h + 1
            chunk = max(1, ITEM_CHUNK // (Pn * Pn * max(self.K, 1)))
            for s0 in range(0, idx.numel(), chunk):
                yield h, idx[s0:s0 + chunk]

    def _patch(self, pos, wid, ang, h):
        """Normalised Gaussian patches g (m, P^2) of the given items on +-h nodes and their flat node indices (m, P^2; N*N = outside)."""
        N, p = self.N, self.pitch
        q = p * p / 12.0
        ia, ib = 1.0 / (wid[:, 0] ** 2 + q), 1.0 / (wid[:, 1] ** 2 + q)
        cth, sth = torch.cos(ang), torch.sin(ang)
        Aq = cth * cth * ia + sth * sth * ib
        Cq = sth * sth * ia + cth * cth * ib
        Bq = cth * sth * (ia - ib)
        off = torch.arange(-h, h + 1, device=pos.device)
        x, y = pos[:, 0], pos[:, 1]
        jx = torch.round((x.detach() - self.x0) / p).long()[:, None] + off[None]
        jy = torch.round((y.detach() - self.x0) / p).long()[:, None] + off[None]   # (m, P)
        dx = (self.x0 + jx.to(x.dtype) * p) - x[:, None]
        dy = (self.x0 + jy.to(y.dtype) * p) - y[:, None]
        arg = (0.5 * Aq[:, None, None] * dx[:, None, :] ** 2 + 0.5 * Cq[:, None, None] * dy[:, :, None] ** 2
               + Bq[:, None, None] * dx[:, None, :] * dy[:, :, None])              # (m, P_y, P_x)
        g = torch.exp(-arg)
        g = g / (g.sum((1, 2), keepdim=True) * p * p)                       # exact flux: p^2 sum = 1 over the full patch
        okx = (jx >= 0) & (jx < N)
        oky = (jy >= 0) & (jy < N)
        flat = torch.where(oky[:, :, None] & okx[:, None, :], jy[:, :, None] * N + jx[:, None, :],
                           torch.full_like(jy[:, :, None] * jx[:, None, :], N * N))
        return g.reshape(g.shape[0], -1), flat.reshape(g.shape[0], -1)

    def _raster_forward(self, pos, wid, ang, c):
        """(N*N + 1, K) scatter of every item (last row: sink for nodes outside the grid). No graph is kept by the caller."""
        out = torch.zeros(self.N * self.N + 1, self.K, device=pos.device, dtype=pos.dtype)
        for h, i in self._buckets(wid, ang):
            g, flat = self._patch(pos[i], wid[i], ang[i], h)
            out.index_add_(0, flat.reshape(-1), (g[:, :, None] * c[i, None, :]).reshape(-1, self.K))
        return out

    def _raster_vjp(self, pos, wid, ang, c, gout):
        """Gradient of sum(gout * raster) with respect to (pos, wid, ang, c), chunk by chunk: each chunk's patches are recomputed with a
        graph, differentiated and freed, so the memory held is one chunk (ITEM_CHUNK elements), not items x patch nodes x K."""
        gs = [torch.zeros_like(t) for t in (pos, wid, ang, c)]
        for h, i in self._buckets(wid, ang):
            with torch.enable_grad():
                leaf = [t[i].detach().requires_grad_(True) for t in (pos, wid, ang, c)]
                g, flat = self._patch(leaf[0], leaf[1], leaf[2], h)
                val = (g[:, :, None] * leaf[3][:, None, :] * gout[flat]).sum()
                d = torch.autograd.grad(val, leaf)
            for G, di in zip(gs, d):
                G[i] = di
        return tuple(gs)

    def raster(self, v):
        """Items of view v rasterised on the coefficient grid: (K, N, N). Same values and gradients as a plain autograd scatter; the
        backward pass recomputes the patches chunk by chunk (see _raster_vjp) instead of storing them."""
        N, K = self.N, self.K
        if self.item_impl == "torchsky":
            from .tsky_backends import item_raster_torchsky
            return item_raster_torchsky(self, self.pos[v], self.lw[v], self.ang[v], self.lc[v])
        if self.raster_mode == "pyramid":
            return self._raster_pyramid(v)
        out = _ItemRaster.apply(self.pos[v], self.widths(v), self.ang[v], torch.exp(self.lc[v]), self)
        return out[:N * N].T.reshape(K, N, N)

    def plan(self):
        """Pyramid geometry (F40), built once: the number of levels follows from the largest possible item variance
        floor^2 + max_width^2 + p^2/12."""
        if self._plan is None:
            self._plan = PyramidPlan(self.N, self.pitch, self.x0, self.floor ** 2 + self.max_width ** 2 + self.pitch ** 2 / 12.0)
        return self._plan

    def _raster_pyramid(self, v):
        """--item-raster pyramid (F40): items whose smaller principal variance of T is below the level-1 threshold are rasterised on the
        fine grid by the direct code path; the others on the coarsest level l they qualify for (detached widths), with covariance
        T - S_l^2 I, and the levels are combined coarse to fine by the fixed upsample-and-blur cascade (ordinary autograd on dense grids).
        Per-level scatters use _ItemRaster (recompute-in-chunks backward). With patch='sheared' every level, level 0 included, uses
        the sheared patch (_ShearGrid) instead: same Gaussian and normalisation, fewer nodes for elongated items."""
        N, K = self.N, self.K
        pl = self.plan()
        pos, wid, ang, c = self.pos[v], self.widths(v), self.ang[v], torch.exp(self.lc[v])
        lev = pl.assign(wid)
        G = None
        for l in range(int(lev.max()) if lev.numel() else 0, 0, -1):
            idx = torch.nonzero(lev == l).squeeze(1)
            if idx.numel():
                lg = pl.level_grid(l, K, shear=self.patch_shape == "sheared")
                R = _ItemRaster.apply(pos[idx], wid[idx], ang[idx], c[idx], lg)[:lg.N * lg.N].T.reshape(K, lg.N, lg.N)
                G = R if G is None else G + R
            if G is not None:
                G = pl.up(G, l)
        idx = torch.nonzero(lev == 0).squeeze(1)
        if idx.numel():
            lg = pl.level_grid(0, K, shear=True) if self.patch_shape == "sheared" else self
            R = _ItemRaster.apply(pos[idx], wid[idx], ang[idx], c[idx], lg)[:N * N].T.reshape(K, N, N)
            G = R if G is None else G + R
        if G is None:
            G = torch.zeros(K, N, N, device=pos.device, dtype=pos.dtype) + 0.0 * (pos.sum() + wid.sum() + ang.sum() + c.sum())
        return G

    def view_coeffs(self, v):
        A = self.raster(v)
        if self.ped is not None:
            A = A - self.ped.to(A)[:, None, None]
        return A

    def coeffs(self, view=None):
        live = torch.is_grad_enabled() and any(t.requires_grad for t in self.parameters())
        if not live:
            key = self._key(view)
            if self._cache is not None and self._cache[0] == key:
                return self._cache[1]
        if view is None:
            A = torch.stack([self.view_coeffs(v) for v in range(self.n_views)]).mean(0) if self.n_views > 1 else self.view_coeffs(0)
        else:
            A = self.view_coeffs(view)
        if not live:
            self._cache = (key, A.detach())
        else:
            self._cache = None
        return A

    def _load_from_state_dict(self, state_dict, prefix, *a, **k):
        state_dict.pop(prefix + "resid", None)       # saved before the item relocation was removed: an all-zero buffer, no longer kept
        super()._load_from_state_dict(state_dict, prefix, *a, **k)
        self._mut += 1
        self._cache = None


_LevelGrid._raster_forward = ItemSky._raster_forward
_LevelGrid._raster_vjp = ItemSky._raster_vjp
