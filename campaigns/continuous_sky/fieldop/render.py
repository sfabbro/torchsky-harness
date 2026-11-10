"""Rendering of  c_q(xi) = int d^2u k(u; xi, lambda_q) s(Omega_q(xi) + u, lambda_q)  (F9-F14).

Two independent quadratures of the same continuous integral (the sky is the tent interpolant of its grid, sky.py):

1. OVERLAP-ADD FFT (Nagy & O'Leary 1998), `render_fft`.
   * Kernel quadrature = nested square LEVELS: level l has pitch p_l = p_r f^l and (2 n_h + 1)^2 nodes, radius R_l = n_h p_l; a smooth
     partition of unity in r splits the kernel between levels (window w_l = c_l - c_(l-1), c_l = 1 for r <= a R_l falling to 0 at R_l,
     times the extent taper that ends at --kernel-extent).  The finest pitch p_r (--render-pitch) is finer than the finest detector pixel.
   * Normalisation: Z = sum_l sum_nodes exp(l) w_l p_l^2, so the kernel has unit integral ON THIS QUADRATURE (F7).
   * Levels with pitch below 2 sky pixels read the sky by tent interpolation of the coefficient grid on their nodes (exact); coarser levels
     read the coefficient grid box-averaged over p_l / p_sky cells (F12: box average of the coefficients, not of the tent function).
   * Each exposure is cut into REGIONS of --region-px detector pixels; K is evaluated at the region nodes (corners), the sky is correlated
     with each node kernel by FFT and the results are blended with the bilinear (piecewise-linear) weights of the nodes at each pixel.
   * The correlated level images are read at the pixel positions Omega_q(xi) by prefiltered cubic B-spline interpolation (default) or
     Keys bicubic (--interp keys) (F13).
2. PER-PIXEL QUADRATURE, `render_quad` (reference): k evaluated with each pixel's OWN code at points around the pixel position: a dense
   square lattice (pitch p_c) in the core disk, and stratified, jittered, log-uniform-in-radius rings outward (importance sampling
   proportional to 1/r^2) with the same windows; the sky is read exactly (tent) at every point; Z from the same points.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


def smooth_down(r, a, b):
    """1 for r <= a, 0 for r >= b, C1 cubic in between."""
    if b <= a:
        return (r <= a).to(r.dtype)
    t = ((r - a) / (b - a)).clamp(0, 1)
    return 1 - t * t * (3 - 2 * t)


class Levels:
    """Nested level grids of the FFT kernel quadrature (and the windows reused by the per-pixel quadrature)."""

    def __init__(self, p_r, factor, n_h, r_ext, a=0.5, taper=0.85, device="cpu"):
        self.p_r, self.f, self.n_h, self.r_ext, self.a, self.taper = float(p_r), int(factor), int(n_h), float(r_ext), a, taper
        L = 0
        while n_h * p_r * factor ** L < r_ext * (1 - 1e-9):
            L += 1
        self.L = L
        self.pitch = [p_r * factor ** l for l in range(L + 1)]
        self.R = [n_h * p for p in self.pitch]
        idx = torch.arange(-n_h, n_h + 1, dtype=torch.float32, device=device)
        iy, ix = torch.meshgrid(idx, idx, indexing="ij")
        self.u, self.w, self.mask = [], [], []
        for l in range(L + 1):
            u = torch.stack([ix * self.pitch[l], iy * self.pitch[l]], -1)  # (kh, kw, 2): X along columns, Y along rows
            r = u.norm(dim=-1)
            w = self.window(r, l)
            self.u.append(u)
            self.w.append(w)
            self.mask.append(w > 1e-7)
        self.n_points = int(sum(int(m.sum()) for m in self.mask))

    def c(self, r, l):
        T = smooth_down(r, self.taper * self.r_ext, self.r_ext)
        if l < 0:
            return torch.zeros_like(r)
        if l >= self.L:
            return T
        return smooth_down(r, self.a * self.R[l], self.R[l]) * T

    def window(self, r, l):
        return self.c(r, l) - self.c(r, l - 1)


def next_fast(n):
    m = n
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def correlate_same(img, ker):
    """out[..., y, x] = sum_ij ker[..., i, j] img[..., y + i - ph, x + j - pw], zero padding (same as F.conv2d with padding).
    img (..., H, W) broadcastable against ker (..., kh, kw) on the leading dims."""
    H, W = img.shape[-2:]
    kh, kw = ker.shape[-2:]
    ph, pw = kh // 2, kw // 2
    Hf, Wf = next_fast(H + kh - 1), next_fast(W + kw - 1)
    Fi = torch.fft.rfft2(img, s=(Hf, Wf))
    Fk = torch.fft.rfft2(ker.flip(-2, -1), s=(Hf, Wf))
    full = torch.fft.irfft2(Fi * Fk, s=(Hf, Wf))
    return full[..., ph:ph + H, pw:pw + W]


def kernel_level_values(net, levels, R, codes, amp_free=False, shift0=None, ckpt=False, dil=None):
    """Evaluate the kernel field for codes (C, dh) on every level grid of an exposure with rotation R (2x2).
    shift0 (2,) in units of the level-0 pitch: random sub-pitch phase of the level-0 grid (training, F11); None = 0.
    Returns list over levels of (C, kh, kw) DISCRETE weights K_l = k w_l p_l^2 / Z (Z omitted when amp_free), and log Z (C,)."""
    C = codes.shape[0]
    logs, masks = [], []
    for l in range(levels.L + 1):
        if l == 0 and shift0 is not None:
            u_full = levels.u[0] + shift0.to(levels.u[0].device) * levels.pitch[0]
            w_full = levels.window(u_full.norm(dim=-1), 0)
            m = w_full > 1e-7
            u, w = u_full[m], w_full[m]
        else:
            m = levels.mask[l]
            u, w = levels.u[l][m], levels.w[l][m]
        v = u @ R                                # v = R^T u  (row vectors: u R)
        if dil is not None:                      # common dilation of every kernel (blur gauge, F18)
            v = v / dil
        if ckpt and torch.is_grad_enabled():     # recompute the field in the backward pass (memory, F26)
            lg = torch.utils.checkpoint.checkpoint(lambda vv, cc: net(net.enc(vv), cc), v, codes, use_reentrant=False)
        else:
            lg = net(net.enc(v), codes)          # (C, P)
        logs.append(lg + torch.log(w * levels.pitch[l] ** 2)[None])
        masks.append(m)
    logZ = torch.logsumexp(torch.cat(logs, 1), 1)                    # (C,)
    out = []
    for l in range(levels.L + 1):
        m = masks[l]
        kh = m.shape[0]
        K = torch.zeros(C, kh, kh, device=codes.device, dtype=codes.dtype)
        wl = logs[l] if amp_free else logs[l] - logZ[:, None]
        K[:, m] = torch.exp(wl)
        out.append(K)
    return out, logZ


def pooled_level(levels, sky_pitch, l):
    """Levels with pitch >= 2 sky pixels read box-averaged coefficients (sky-aligned grids); finer levels read the tent sky exactly and can
    use a grid aligned with the detector (F31)."""
    m = levels.pitch[l] / sky_pitch
    return l > 0 and abs(m - round(m)) < 1e-6 and round(m) >= 2


def kernel_level_values_grouped(net, levels, Rs, codes, amp_free=False, shift0=None, ckpt=False, dil=None, det_frame=()):
    """Batched kernel_level_values: Rs (G, 2, 2) one rotation per group (patch), codes (G, C, dh).
    Returns list over levels of (G, C, kh, kw) discrete weights and log Z (G, C)."""
    G, C = codes.shape[:2]
    logs, masks = [], []
    for l in range(levels.L + 1):
        if l == 0 and shift0 is not None:
            u_full = levels.u[0] + shift0.to(levels.u[0].device) * levels.pitch[0]
            w_full = levels.window(u_full.norm(dim=-1), 0)
            m = w_full > 1e-7
            u, w = u_full[m], w_full[m]
        else:
            m = levels.mask[l]
            u, w = levels.u[l][m], levels.w[l][m]
        if l in det_frame:                       # grid aligned with the detector: the offsets ARE detector-frame offsets
            v = u[None].expand(G, -1, -1)
        else:
            v = torch.einsum("pi,gij->gpj", u, Rs)   # (G, P, 2) detector-frame offsets of a sky-aligned grid
        if dil is not None:
            v = v / dil
        if ckpt and torch.is_grad_enabled():
            lg = torch.utils.checkpoint.checkpoint(lambda vv, cc: net(net.enc(vv), cc), v, codes, use_reentrant=False)
        else:
            lg = net(net.enc(v), codes)          # (G, C, P)
        logs.append(lg + torch.log(w * levels.pitch[l] ** 2)[None, None])
        masks.append(m)
    logZ = torch.logsumexp(torch.cat(logs, -1), -1)                  # (G, C)
    out = []
    for l in range(levels.L + 1):
        m = masks[l]
        kh = m.shape[0]
        K = torch.zeros(G, C, kh, kh, device=codes.device, dtype=codes.dtype)
        wl = logs[l] if amp_free else logs[l] - logZ[..., None]
        K[:, :, m] = torch.exp(wl)
        out.append(K)
    return out, logZ


def bicubic_read(img, X0, pitch, X, Y):
    """img (B, C, H, W) on nodes X = X0[0] + j pitch, Y = X0[1] + i pitch; read at (X, Y) of shape (B, N) -> (B, C, N)."""
    B, C, H, W = img.shape
    ox = X0[0][:, None] if torch.is_tensor(X0[0]) else X0[0]
    oy = X0[1][:, None] if torch.is_tensor(X0[1]) else X0[1]
    gx = (X - ox) / (pitch * (W - 1)) * 2 - 1
    gy = (Y - oy) / (pitch * (H - 1)) * 2 - 1
    g = torch.stack([gx, gy], -1)[:, None]  # (B, 1, N, 2)
    return F.grid_sample(img, g, mode="bicubic", padding_mode="zeros", align_corners=True)[:, :, 0]


def bspline_prefilter(img):
    """Cubic B-spline interpolation coefficients of img (..., H, W): division by the B-spline sampling spectrum (2 + cos w)/3 per axis
    (periodic boundary; the IIR tail decays as 0.268^k, far inside the n_h + 3 margin of every window)."""
    H, W = img.shape[-2:]
    wy = torch.fft.fftfreq(H, device=img.device) * 2 * math.pi
    wx = torch.fft.rfftfreq(W, device=img.device) * 2 * math.pi
    den = ((2 + torch.cos(wy))[:, None] / 3) * ((2 + torch.cos(wx))[None] / 3)
    return torch.fft.irfft2(torch.fft.rfft2(img) / den, s=(H, W))


def bspline_read(coef, X0, pitch, X, Y):
    """Cubic B-spline evaluation of prefiltered coefficients coef (B, C, H, W) at (X, Y) (B, N) -> (B, C, N); zero outside."""
    B, C, H, W = coef.shape
    ox = X0[0][:, None] if torch.is_tensor(X0[0]) else X0[0]
    oy = X0[1][:, None] if torch.is_tensor(X0[1]) else X0[1]
    fx = (X - ox) / pitch
    fy = (Y - oy) / pitch
    ix, iy = torch.floor(fx), torch.floor(fy)
    tx, ty = fx - ix, fy - iy

    def wts(t):
        t2, t3 = t * t, t * t * t
        return torch.stack([(1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6], -1)  # (B, N, 4)
    wx, wy = wts(tx), wts(ty)
    offs = torch.arange(-1, 3, device=coef.device)
    cx = ix.long()[..., None] + offs            # (B, N, 4)
    cy = iy.long()[..., None] + offs
    okx = (cx >= 0) & (cx < W)
    oky = (cy >= 0) & (cy < H)
    idx = (cy.clamp(0, H - 1)[..., :, None] * W + cx.clamp(0, W - 1)[..., None, :])        # (B, N, 4, 4)
    w = (wy * oky)[..., :, None] * (wx * okx)[..., None, :]                                  # (B, N, 4, 4)
    flat = coef.reshape(B, C, H * W)
    g = torch.gather(flat, 2, idx.reshape(B, 1, -1).expand(B, C, -1)).view(B, C, *idx.shape[1:])
    return (g * w[:, None]).sum((-1, -2))


def crop_padded(img, i0, j0, h, w):
    """img (..., H, W): rows i0..i0+h, cols j0..j0+w with zero padding outside."""
    H, W = img.shape[-2:]
    out = img.new_zeros(*img.shape[:-2], h, w)
    a0, a1 = max(i0, 0), min(i0 + h, H)
    b0, b1 = max(j0, 0), min(j0 + w, W)
    if a1 > a0 and b1 > b0:
        out[..., a0 - i0:a1 - i0, b0 - j0:b1 - j0] = img[..., a0:a1, b0:b1]
    return out


def render_fft(sky, A, Phi_q, levels, kernels, node_w, pos, os_, shift0=None, interp="bspline"):
    """Overlap-add FFT render of one patch.

    sky: SkyGrid; A (K, N, N) coefficients (one view); Phi_q (Q, K) basis at the patch's lambda nodes;
    kernels: list over levels of (n_nodes, Q, kh, kw) discrete weights; node_w (n_nodes, Npix) blending weights;
    pos (Q, Npix, 2) positions Omega_q(xi) [arcsec]; os_ = sky pitch / render pitch (integer); shift0: the level-0 phase used for the kernels.
    Returns c (Q, Npix)."""
    Q, Np, _ = pos.shape
    n_h = levels.n_h
    out = 0.0
    lo = pos.reshape(-1, 2).min(0).values.detach()
    hi = pos.reshape(-1, 2).max(0).values.detach()
    for l in range(levels.L + 1):
        p = levels.pitch[l]
        m = p / sky.pitch
        pooled = l > 0 and abs(m - round(m)) < 1e-6 and round(m) >= 2
        if pooled:   # coarse level: box average of the coefficient grid over m x m cells (F12)
            Ap, X0g = sky.pooled(A, int(round(m)))
        else:        # fine level: tent interpolation of the coefficient grid on the level nodes (exact); level 0 is aligned with sky node 0
            X0g = sky.x0
        pr = p
        marg = n_h + 3
        j0 = int(math.floor((float(lo[0]) - X0g) / pr)) - marg
        i0 = int(math.floor((float(lo[1]) - X0g) / pr)) - marg
        j1 = int(math.ceil((float(hi[0]) - X0g) / pr)) + marg
        i1 = int(math.ceil((float(hi[1]) - X0g) / pr)) + marg
        h, w = i1 - i0 + 1, j1 - j0 + 1
        if not pooled:
            xs = X0g + (j0 + torch.arange(w, device=A.device, dtype=A.dtype)) * pr
            ys = X0g + (i0 + torch.arange(h, device=A.device, dtype=A.dtype)) * pr
            YY, XX = torch.meshgrid(ys, xs, indexing="ij")
            if l == 0 and shift0 is not None:   # kernel node i sits at (i + delta) p: read the sky at the shifted nodes, output stays on the nodes
                XX, YY = XX + float(shift0[0]) * pr, YY + float(shift0[1]) * pr
            Sk = sky.sample_coeffs(A, XX, YY)                        # (K, h, w)
        else:
            Sk = crop_padded(Ap, i0, j0, h, w)
        S = torch.einsum("qk,khw->qhw", Phi_q, Sk)                   # (Q, h, w)
        Cimg = correlate_same(S[None], kernels[l])                   # (n, Q, h, w)
        org = (X0g + j0 * pr, X0g + i0 * pr)
        n = Cimg.shape[0]
        Ci = Cimg.permute(1, 0, 2, 3)
        if interp == "bspline":
            rd = bspline_read(bspline_prefilter(Ci), org, pr, pos[..., 0], pos[..., 1])         # (Q, n, Npix)
        else:
            rd = bicubic_read(Ci, org, pr, pos[..., 0], pos[..., 1])
        out = out + (rd * node_w[None]).sum(1)
    return out


def render_fft_batch(sky, A, Phi_b, levels, kernels, node_w, pos, shift0=None, interp="bspline", Rs=None, det_frame=()):
    """Overlap-add FFT render of B patches at once (same channel = same level grids), one fixed window size per level.

    Phi_b (B, Q, K); kernels: list over levels of (B, n, Q, kh, kw); node_w (B, n, Np) (zero for padding nodes / pixels);
    pos (B, Q, Np, 2). Rs (B, 2, 2) sky <- detector rotations; levels in det_frame use a detector-aligned window (smaller for rotated
    detectors: the sky-aligned bounding box of a patch rotated by 45 deg has twice its area). Returns c (B, Q, Np)."""
    B, Q, Np, _ = pos.shape
    n_h = levels.n_h
    dev, dt = A.device, A.dtype
    flat = pos.detach().reshape(B, -1, 2)
    lo_s, hi_s = flat.min(1).values.cpu().double(), flat.max(1).values.cpu().double()
    if det_frame:
        posd = torch.einsum("bqni,bij->bqnj", pos, Rs)                      # detector-frame coordinates (about the box centre)
        fd = posd.detach().reshape(B, -1, 2)
        lo_d, hi_d = fd.min(1).values.cpu().double(), fd.max(1).values.cpu().double()
    out = 0.0
    for l in range(levels.L + 1):
        p = levels.pitch[l]
        m = p / sky.pitch
        pooled = pooled_level(levels, sky.pitch, l)
        dfr = l in det_frame
        lo, hi = (lo_d, hi_d) if dfr else (lo_s, hi_s)
        if pooled:
            Ap, X0g = sky.pooled(A, int(round(m)))
        else:
            X0g = 0.0 if dfr else sky.x0
        marg = n_h + 3
        j0 = torch.floor((lo[:, 0] - X0g) / p).long() - marg
        i0 = torch.floor((lo[:, 1] - X0g) / p).long() - marg
        j1 = torch.ceil((hi[:, 0] - X0g) / p).long() + marg
        i1 = torch.ceil((hi[:, 1] - X0g) / p).long() + marg
        h, w = int((i1 - i0).max()) + 1, int((j1 - j0).max()) + 1
        ar_h, ar_w = torch.arange(h, device=dev), torch.arange(w, device=dev)
        i0d, j0d = i0.to(dev), j0.to(dev)
        if not pooled:
            xs = X0g + (j0d[:, None] + ar_w[None]).to(dt) * p                 # (B, w)
            ys = X0g + (i0d[:, None] + ar_h[None]).to(dt) * p                 # (B, h)
            if l == 0 and shift0 is not None:
                xs, ys = xs + float(shift0[0]) * p, ys + float(shift0[1]) * p
            XX = xs[:, None, :].expand(B, h, w)
            YY = ys[:, :, None].expand(B, h, w)
            if dfr:   # node (a, b) of the detector-aligned grid sits at sky position R (a, b)
                XX, YY = (Rs[:, 0, 0, None, None] * XX + Rs[:, 0, 1, None, None] * YY, Rs[:, 1, 0, None, None] * XX + Rs[:, 1, 1, None, None] * YY)
            Sk = sky.sample_coeffs(A, XX.reshape(B * h, w), YY.reshape(B * h, w)).view(A.shape[0], B, h, w)
        else:
            Hn = Ap.shape[-1]
            pad = int(max(0, -int(i0.min()), -int(j0.min()), int((i0 + h).max()) - Hn, int((j0 + w).max()) - Hn)) + 1
            App = F.pad(Ap, (pad, pad, pad, pad))
            rows = (i0d[:, None] + ar_h[None] + pad)[:, :, None]
            cols = (j0d[:, None] + ar_w[None] + pad)[:, None, :]
            Sk = App[:, rows, cols]                                         # (K, B, h, w)
        S = torch.einsum("bqk,kbhw->bqhw", Phi_b, Sk)
        Cimg = correlate_same(S[:, None], kernels[l])                       # (B, n, Q, h, w)
        n = Cimg.shape[1]
        Ci = Cimg.permute(0, 2, 1, 3, 4).reshape(B * Q, n, h, w)
        org = ((X0g + j0d.to(dt) * p).repeat_interleave(Q), (X0g + i0d.to(dt) * p).repeat_interleave(Q))
        P2 = (posd if dfr else pos).reshape(B * Q, Np, 2)
        if interp == "bspline":
            rd = bspline_read(bspline_prefilter(Ci), org, p, P2[..., 0], P2[..., 1])
        else:
            rd = bicubic_read(Ci, org, p, P2[..., 0], P2[..., 1])
        out = out + (rd.view(B, Q, n, Np) * node_w[:, None]).sum(2)
    return out


def quad_points(levels, p_c, n_rad=12, n_az=48, gen=None, device="cpu"):
    """Per-pixel quadrature points (P, 2) and weights (P,) [arcsec^2, windows included].
    Core: square lattice of pitch p_c in r <= R_0 (weights c_0(r) p_c^2).  Levels l >= 1: the annulus a R_(l-1) < r < min(R_l, r_ext) is cut into
    n_rad log-uniform radial bins x n_az azimuth bins (alternate rings rotated by half a bin); one jittered point per bin (importance density
    proportional to 1/r^2); weight = bin area x w_l(r)."""
    R0 = levels.R[0] if levels.L > 0 else levels.r_ext
    nh = int(math.ceil(R0 / p_c))
    idx = torch.arange(-nh, nh + 1, dtype=torch.float32, device=device) * p_c
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    u = torch.stack([xx.flatten(), yy.flatten()], -1)
    r = u.norm(dim=-1)
    w = levels.window(r, 0) * p_c ** 2
    keep = w > 1e-9
    us, ws = [u[keep]], [w[keep]]
    for l in range(1, levels.L + 1):
        r_in = levels.a * levels.R[l - 1]
        r_out = min(levels.R[l], levels.r_ext)
        if r_out <= r_in:
            continue
        e = torch.linspace(math.log(r_in), math.log(r_out), n_rad + 1, device=device)
        jr = torch.rand(n_rad, n_az, generator=gen, device=device) if gen is not None else torch.full((n_rad, n_az), 0.5, device=device)
        ja = torch.rand(n_rad, n_az, generator=gen, device=device) if gen is not None else torch.full((n_rad, n_az), 0.5, device=device)
        rr0, rr1 = torch.exp(e[:-1])[:, None], torch.exp(e[1:])[:, None]
        # area-uniform radius inside the bin
        rr = torch.sqrt(rr0 ** 2 + jr * (rr1 ** 2 - rr0 ** 2))
        off = (torch.arange(n_rad, device=device) % 2)[:, None] * 0.5
        th = (torch.arange(n_az, device=device)[None] + off + (ja - 0.5)) * (2 * math.pi / n_az)
        area = math.pi * (rr1 ** 2 - rr0 ** 2) / n_az
        uu = torch.stack([rr * torch.cos(th), rr * torch.sin(th)], -1).reshape(-1, 2)
        ww = (area.expand(n_rad, n_az).reshape(-1)) * levels.window(rr.reshape(-1), l)
        k = ww > 1e-12
        us.append(uu[k])
        ws.append(ww[k])
    return torch.cat(us), torch.cat(ws)


def render_quad(sky, A, Phi_q, net, R, codes_pix, pos, qpts, qw, chunk=256, amp_free=False, dil=None):
    """Per-pixel quadrature: codes_pix (Q, Npix, dh), pos (Q, Npix, 2), points qpts (P, 2) with weights qw (P,). Returns c (Q, Npix)."""
    Q, Np, _ = pos.shape
    feats = net.enc(qpts @ R / (1.0 if dil is None else dil))   # (P, F) shared pattern (detector frame)
    out = torch.empty(Q, Np, device=A.device, dtype=A.dtype)
    lw = torch.log(qw)
    for q in range(Q):
        for s0 in range(0, Np, chunk):
            sl = slice(s0, min(Np, s0 + chunk))
            lg = net(feats, codes_pix[q, sl])        # (c, P)
            lwt = lg + lw[None]
            if amp_free:
                wt = torch.exp(lwt)
            else:
                wt = torch.softmax(lwt, 1)
            X = pos[q, sl, 0][:, None] + qpts[None, :, 0]
            Y = pos[q, sl, 1][:, None] + qpts[None, :, 1]
            sk = sky.sample_coeffs(A, X, Y)           # (K, c, P)
            s = torch.einsum("k,kcp->cp", Phi_q[q], sk)
            out[q, sl] = (wt * s).sum(1)
    return out
