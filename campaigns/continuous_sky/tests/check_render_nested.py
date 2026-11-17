"""Measurement for --render-impl torchsky (D): the prototype's overlap-add FFT render (render.render_fft) against the same render built on
torchsky's NestedKernelPSF, for the SAME sky, kernel values and exposure geometry.   python tests/check_render_nested.py [--out FILE]

Case: float64; sky = tent interpolant of a coefficient grid at pitch 0.1 (arcsec, one channel); render pitch = sky pitch = detector pixel pitch = 0.1
and unrotated (so that NestedKernelPSF, a convolution on one unit-pitch grid, can be applied at all); kernel field k(u) = Gaussian core (sigma 1.5
pixels) + power-law wing, n_h = 8 nodes per side per level, extent 30 pixels (levels 0, 1, 2: three nesting levels), window parameters at the
prototype's defaults (0.5, 0.85, which NestedKernelPSF shares); output positions = node positions + a sub-pixel offset (0.37, 0.21) pixels.
  prototype : render.render_fft with discrete weights K_l = k w_l p_l^2 / Z  (box-averaged pooled sky on levels >= 1, F12; cubic B-spline read).
  nested    : sky tent-sampled on the level-0 lattice -> NestedKernelPSF (hat-function pooling and read-back) -> the same cubic B-spline read at the
              output positions.
  dense     : dense sum over a lattice 4x finer than the pixel, same tapered kernel, tent sky read exactly (differs from both by the tent kinks
              between sky nodes, which neither method samples: not a fair reference for a sharp sky).
  lattice reference: the node-sampled sky image convolved with the dense tapered stamp, same B-spline read (the quantity both methods approximate).
Errors are max |a - b| / max |b| over the output pixels."""
import argparse
import json
import math
import os
import sys

import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
torch.set_default_dtype(torch.float64)

P = 0.1            # pitch [arcsec]
NH = 8
EXT_PX = 30.0
OFFSET = (0.37, 0.21)


def kfun(r2_px):
    """kernel density as a function of squared radius in PIXELS (shared by both renders)."""
    return torch.exp(-r2_px / (2 * 1.5 ** 2)) + 0.02 * (1.0 + r2_px / 9.0) ** -1.5


def make_sky(kind, N=161, seed=0):
    from fieldop.sky import SkyGrid
    g = torch.Generator().manual_seed(seed)
    c = torch.arange(N, dtype=torch.float64)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    A = torch.zeros(1, N, N)

    def blob(x0, y0, s, a):
        return a * torch.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / s ** 2)
    if kind == "smooth":       # galaxies and diffuse light: widths 4 to 20 pixels
        for _ in range(12):
            A[0] += blob(*(torch.rand(2, generator=g) * (N - 1)).tolist(), float(4 + 16 * torch.rand(1, generator=g)), float(torch.rand(1, generator=g)) + 0.2)
    elif kind == "sharp":      # plus stars: widths 0.7 to 1.2 pixels
        for _ in range(8):
            A[0] += blob(*(torch.rand(2, generator=g) * (N - 1)).tolist(), float(4 + 16 * torch.rand(1, generator=g)), float(torch.rand(1, generator=g)) + 0.2)
        for _ in range(30):
            A[0] += blob(*(torch.rand(2, generator=g) * (N - 1)).tolist(), float(0.7 + 0.5 * torch.rand(1, generator=g)), 5 * float(torch.rand(1, generator=g)) + 1.0)
    elif kind == "noise":      # white noise: all power at the pixel scale (worst case of the nested scheme)
        A[0] = torch.randn(N, N, generator=g)
    else:
        raise ValueError(kind)
    return SkyGrid(A, P, None)


def run(kind):
    import torchsky.map.operators.nested_kernel as nk
    from fieldop.render import Levels, bspline_prefilter, bspline_read, render_fft
    sky = make_sky(kind)
    levels = Levels(P, 2, NH, EXT_PX * P, a=0.5, taper=0.85)
    # pixels: 36 x 36 at the sky nodes around the centre plus the sub-pixel offset
    c0 = (sky.N - 1) // 2 - 18
    j = torch.arange(36, dtype=torch.float64)
    yy, xx = torch.meshgrid(j, j, indexing="ij")
    X = sky.x0 + (c0 + xx + OFFSET[0]) * P
    Y = sky.x0 + (c0 + yy + OFFSET[1]) * P
    pos = torch.stack([X.reshape(-1), Y.reshape(-1)], -1)[None]                     # (Q=1, Npix, 2)
    # discrete kernel weights of the prototype, from the SAME analytic field
    logs = []
    for l in range(levels.L + 1):
        u, w, m = levels.u[l].double(), levels.w[l].double(), levels.mask[l]
        r2 = ((u / P) ** 2).sum(-1)
        logs.append(kfun(r2) * w * levels.pitch[l] ** 2)
    Z = sum(float(t[m].sum()) for t, m in zip(logs, levels.mask))
    kernels = [(t * m / Z)[None, None] for t, m in zip(logs, levels.mask)]            # (n=1, Q=1, kh, kw)
    node_w = torch.ones(1, pos.shape[1])
    Phi_q = torch.ones(1, 1)
    out_proto = render_fft(sky, sky.A[0].detach(), Phi_q, levels, kernels, node_w, pos, 1, shift0=None, interp="bspline")[0]
    # nested route
    def field(offsets, condition):
        return kfun((offsets ** 2).sum(-1))[None]
    psf = nk.NestedKernelPSF(field, core_half_width=NH, extent=EXT_PX)
    marg = int(EXT_PX) + 12
    lo = c0 - marg
    n = 36 + 2 * marg
    ax = torch.arange(n, dtype=torch.float64)
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    img = sky.sample_coeffs(sky.A[0].detach(), sky.x0 + (lo + gx) * P, sky.x0 + (lo + gy) * P)[0]       # the level-0 lattice
    cimg = psf(img, spatial_x=gx, spatial_y=gy, wavelength=1.0)
    org = (sky.x0 + lo * P, sky.x0 + lo * P)
    out_nested = bspline_read(bspline_prefilter(cimg[None, None]), (torch.tensor([org[0]]), torch.tensor([org[1]])), P, pos[..., 0], pos[..., 1])[0, 0]
    # dense reference: lattice 4x finer than the pixel, radius extent, same taper, tent sky exact
    f = 4
    rr = torch.arange(-int(EXT_PX * f), int(EXT_PX * f) + 1, dtype=torch.float64) / f
    uy, ux = torch.meshgrid(rr, rr, indexing="ij")
    r = torch.sqrt(ux ** 2 + uy ** 2)
    from fieldop.render import smooth_down
    wk = kfun(r ** 2) * smooth_down(r, 0.85 * EXT_PX, EXT_PX)
    wk = wk / wk.sum()
    ref = torch.zeros(pos.shape[1])
    for s0 in range(0, pos.shape[1], 64):
        pp = pos[0, s0:s0 + 64]
        xs = pp[:, 0, None, None] + ux[None] * P
        ys = pp[:, 1, None, None] + uy[None] * P
        v = sky.sample_coeffs(sky.A[0].detach(), xs, ys)[0]
        ref[s0:s0 + 64] = (v * wk[None]).sum((1, 2))
    # lattice reference: the same node-sampled sky image convolved with the dense tapered stamp (what both methods approximate), same B-spline read
    from fieldop.render import correlate_same
    kk = int(EXT_PX)
    ar = torch.arange(-kk, kk + 1, dtype=torch.float64)
    sy, sx = torch.meshgrid(ar, ar, indexing="ij")
    rs = torch.sqrt(sx ** 2 + sy ** 2)
    stamp = kfun(rs ** 2) * smooth_down(rs, 0.85 * EXT_PX, EXT_PX)
    stamp = stamp / stamp.sum()
    cref = correlate_same(img[None], stamp[None])[0]
    out_lat = bspline_read(bspline_prefilter(cref[None, None]), (torch.tensor([org[0]]), torch.tensor([org[1]])), P, pos[..., 0], pos[..., 1])[0, 0]
    mx = float(ref.abs().max())
    res = {"sky": kind, "n_levels": levels.L + 1, "nested_num_levels": psf.num_levels, "image_max": mx,
           "nested_vs_prototype": float((out_nested - out_proto).abs().max()) / mx,
           "prototype_vs_dense": float((out_proto - ref).abs().max()) / mx,
           "nested_vs_dense": float((out_nested - ref).abs().max()) / mx,
           "nested_vs_lattice_reference": float((out_nested - out_lat).abs().max()) / mx,
           "prototype_vs_lattice_reference": float((out_proto - out_lat).abs().max()) / mx}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = [run(k) for k in ("smooth", "sharp", "noise")]
    for r in rows:
        print(json.dumps(r))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
