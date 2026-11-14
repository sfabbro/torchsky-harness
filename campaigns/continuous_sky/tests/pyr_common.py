"""Helpers for the pyramid rasteriser tests (tests/test_items_pyramid.py): one-item skies and the exact reference."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fieldop.sky import ItemSky, RBFBasis  # noqa: E402

P = 0.04


def one_item(N, x, y, wa, wb, th, mode, K=1, dtype=torch.float64, max_width=None, floor=None, p=P, patch="square"):
    """One item with principal widths (wa, wb) [arcsec] of Sigma (before the p^2/12 term), angle th, c_k = 1."""
    floor = 0.05 * p if floor is None else floor
    mw = max_width if max_width is not None else 1.01 * max(wa, wb)
    # built with a small width cap, then the cap is raised: the one-item seeding would otherwise smooth the (zero) warm start at the
    # field size, which is slow and memory-hungry (an existing property of ItemSky's seeding, not of the rasteriser)
    s = ItemSky(torch.zeros(K, N, N, dtype=dtype), p, RBFBasis(max(K, 2)), 1, floor=floor, max_width=2 * p, n_levels=1, raster=mode,
                patch=patch if mode == "pyramid" else "square")
    s.max_width = mw
    s._plan = None
    s.K = K
    s.lc = torch.nn.Parameter(torch.zeros(1, 1, K, dtype=dtype))
    with torch.no_grad():
        s.pos.data[0, 0] = torch.tensor([x, y], dtype=dtype)
        s.lw.data[0, 0] = torch.tensor([0.5 * math.log(wa * wa - floor * floor), 0.5 * math.log(wb * wb - floor * floor)], dtype=dtype)
        s.ang.data[0, 0] = th
        s._mut += 1
    return s


def reference(N, x, y, wa, wb, th, p=P):
    """Point-sampled Gaussian with covariance T = Sigma + p^2/12 I on the N x N grid, normalised to exact flux over the INFINITE lattice
    (Poisson summation of the lattice sum, |k| <= 4 per axis): p^2 sum_{all nodes} g = 1."""
    q = p * p / 12
    va, vb = wa * wa + q, wb * wb + q
    c, s = math.cos(th), math.sin(th)
    Txx = c * c * va + s * s * vb
    Tyy = s * s * va + c * c * vb
    Txy = c * s * (va - vb)
    det = Txx * Tyy - Txy * Txy
    Ixx, Iyy, Ixy = Tyy / det, Txx / det, -Txy / det
    x0 = -(N - 1) / 2 * p
    g = x0 + torch.arange(N, dtype=torch.float64) * p
    dx = (g - x)[None, :]
    dy = (g - y)[:, None]
    val = torch.exp(-0.5 * (Ixx * dx * dx + Iyy * dy * dy + 2 * Ixy * dx * dy))
    # lattice sum of exp(-0.5 d^T T^-1 d) over nodes = (2 pi sqrt(det) / p^2) sum_k exp(-2 pi^2 k^T T k / p^2) cos(2 pi k . mu / p)
    mux, muy = (x - x0) / p, (y - x0) / p
    tot = 0.0
    for kx in range(-4, 5):
        for ky in range(-4, 5):
            e = 2 * math.pi ** 2 * (kx * kx * Txx + ky * ky * Tyy + 2 * kx * ky * Txy) / (p * p)
            if e > 700:
                continue
            tot += math.exp(-e) * math.cos(2 * math.pi * (kx * mux + ky * muy))
    Z = 2 * math.pi * math.sqrt(det) / (p * p) * tot
    return val / (Z * p * p)
