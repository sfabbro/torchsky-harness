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

LAM_LO, LAM_HI = 0.70, 5.30  # um: spectral domain of the sky field (as v40)


def lam_unit(lam_um):
    return (torch.log(lam_um) - math.log(LAM_LO)) / (math.log(LAM_HI) - math.log(LAM_LO))


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
