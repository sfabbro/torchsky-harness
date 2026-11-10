"""Physical PSF for JWST/NIRCam: pupil amplitude (18 hexagonal segments, gaps, 3 struts; approximate geometry, soft
edges), learned wavefront W = sum_j c_j Z_j (Noll Z4..Z22 per channel + Z4..Z11 per detector), Fraunhofer
propagation by matrix Fourier transform (MFT) at any wavelength, then detector pixel box, charge diffusion + jitter
(Gaussian) and inter-pixel capacitance applied as MTFs, binned to the sky grid. Kernels are flux-normalised by
Parseval: they sum to the energy fraction inside the window (wings beyond the window are lost, and reported).
Interface matches the other PSF models: basis(), basis_lam(lam_um, chan_idx) -> (M, ndet, n, n); weights() one-hot
per detector (PSF varies between detectors, not within one)."""
import math
import torch
import torch.nn as nn

ARCSEC = math.pi / (180 * 3600)


def noll_to_nm(j):
    """Noll index -> (n, m); m > 0 cosine, m < 0 sine (aotools convention)."""
    n, j1 = 0, j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    return n, m


def zernike(j, rho, phi):
    n, m = noll_to_nm(j)
    am = abs(m)
    R = torch.zeros_like(rho)
    for s in range((n - am) // 2 + 1):
        c = (-1) ** s * math.factorial(n - s) / (math.factorial(s) * math.factorial((n + am) // 2 - s) * math.factorial((n - am) // 2 - s))
        R = R + c * rho ** (n - 2 * s)
    norm = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
    if m > 0:
        return norm * R * torch.cos(am * phi)
    if m < 0:
        return norm * R * torch.sin(am * phi)
    return norm * R


def jwst_pupil(N, extent_m, rot_deg=0.0, seg_f2f=1.32, gap=0.007, strut_w=0.08, flip=False):
    """Soft-edged amplitude on an N x N grid spanning extent_m; segments are flat-top hexagons on a hex lattice."""
    u = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * (extent_m / N)
    yy, xx = torch.meshgrid(u, u, indexing="ij")
    if flip:
        xx = -xx
    t = math.radians(rot_deg)
    x = xx * math.cos(t) + yy * math.sin(t)
    y = -xx * math.sin(t) + yy * math.cos(t)
    eps = extent_m / N  # soft edge width: one pupil sample
    s = seg_f2f + gap
    a1 = (s * math.cos(math.pi / 6), s * math.sin(math.pi / 6))
    a2 = (0.0, s)
    A = torch.zeros_like(x)
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) > 2 or abs(r) > 2 or abs(q + r) > 2 or (q == 0 and r == 0):
                continue
            cx, cy = q * a1[0] + r * a2[0], q * a1[1] + r * a2[1]
            dx, dy = x - cx, y - cy
            h = seg_f2f / 2
            m = torch.minimum(torch.minimum(h - dy.abs(), h - (math.sqrt(3) / 2 * dx + dy / 2).abs()), h - (math.sqrt(3) / 2 * dx - dy / 2).abs())
            A = torch.maximum(A, torch.sigmoid(m / eps * 4))
    for ang in (90.0, 240.0, 300.0):  # struts from the centre outwards
        c, sn = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        along = x * c + y * sn
        perp = (-x * sn + y * c).abs()
        strut = torch.sigmoid((strut_w / 2 - perp) / eps * 4) * (along > 0).double()
        A = A * (1 - strut)
    return A.float(), x.float(), y.float()


class PupilPSF(nn.Module):
    def __init__(self, bands, lam_pivot_um, chan_of_band, det_chan, scale_arcsec, n=121, os=4, N=256, extent_m=6.7,
                 rot_deg=0.0, nz_global=22, nz_det=11, pix_arcsec=(0.031, 0.063)):
        super().__init__()
        self.n, self.os, self.N = n, os, N
        self.ndet = len(det_chan)
        self.scale = scale_arcsec
        self.register_buffer("lam_pivot", torch.tensor(lam_pivot_um, dtype=torch.float32))
        self.register_buffer("chan_idx", torch.tensor(chan_of_band, dtype=torch.long))
        self.register_buffer("det_chan", torch.tensor(det_chan, dtype=torch.long))
        self.extent = extent_m
        self.rot = nn.Parameter(torch.tensor(float(rot_deg)), requires_grad=False)  # set by a grid search, then fixed
        A, x, y = jwst_pupil(N, extent_m, 0.0)
        self.register_buffer("x", x)
        self.register_buffer("y", y)
        self._set_amp(rot_deg)
        R = extent_m / 2 * 0.985
        rho = (x ** 2 + y ** 2).sqrt() / R
        phi = torch.atan2(y, x)
        Z = torch.stack([zernike(j, rho, phi) * (rho <= 1) for j in range(4, nz_global + 1)])  # Noll 4..
        self.register_buffer("Z", Z)
        self.nzd = nz_det - 3
        self.cg = nn.Parameter(torch.zeros(2, Z.shape[0]))  # global wavefront per channel (SW, LW), in 100 nm
        self.cd = nn.Parameter(torch.zeros(self.ndet, self.nzd))  # per-detector low-order deltas, in 100 nm
        self.log_cd = nn.Parameter(torch.log(torch.tensor([0.25, 0.25])))  # charge diffusion + jitter, detector pixels
        self.ipc = nn.Parameter(torch.tensor([0.006, 0.006]))
        self.register_buffer("pix", torch.tensor(pix_arcsec))
        no = n * os
        th = (torch.arange(no, dtype=torch.float32) - (no - 1) / 2) * (scale_arcsec / os) * ARCSEC
        self.register_buffer("theta", th)
        f = torch.fft.fftfreq(no, d=scale_arcsec / os)  # cycles per arcsec
        self.register_buffer("fx", f[None, :].expand(no, no).clone())
        self.register_buffer("fy", f[:, None].expand(no, no).clone())
        self.ee_inside = None

    def _set_amp(self, rot_deg):
        A, _, _ = jwst_pupil(self.N, self.extent, rot_deg)
        self.register_buffer("A", A.to(self.x.device) if hasattr(self, "A") else A)

    def set_rotation(self, rot_deg, flip=False):
        self.rot.data.fill_(rot_deg)
        self.flip = bool(flip)
        A, _, _ = jwst_pupil(self.N, self.extent, rot_deg, flip=flip)
        self.A = A.to(self.x.device)

    @torch.no_grad()
    def fit_orientation(self, img, band, det, nstars=4, step=2.0, r_in=3, r_out=40):
        """Pupil rotation and parity that best fit the brightest compact sources of a coadd of that band: per star
        and angle, least-squares amplitude + constant background in the annulus r_in..r_out px (the spikes),
        robust noise-weighted residual summed over stars. Returns (score, angle, flip) and the scores."""
        H, W = img.shape
        I = img - img.median()
        sig = 1.4826 * (I - I.median()).abs().median()
        k = 5
        mx = torch.nn.functional.max_pool2d(I[None, None], 2 * k + 1, 1, k)[0, 0]
        pk = (I == mx) & (I > 50 * sig)
        ys, xs = torch.nonzero(pk, as_tuple=True)
        ok = (ys >= 8) & (ys < H - 8) & (xs >= 8) & (xs < W - 8)
        ys, xs = ys[ok], xs[ok]
        # point-like: peak over the mean of the 3 x 3 neighbourhood excluding the peak (sharp sources score high)
        conc = torch.stack([I[y, x] / (I[y - 1:y + 2, x - 1:x + 2].sum() - I[y, x]).clamp(min=1e-12) * 8 for y, x in zip(ys, xs)]) if len(ys) else torch.zeros(0)
        order = torch.argsort(conc, descending=True)[:nstars]
        ys, xs = ys[order], xs[order]
        r = torch.arange(-r_out, r_out + 1, device=img.device)
        rr = (r[:, None] ** 2 + r[None, :] ** 2).double().sqrt().to(img.dtype)
        ann0 = (rr >= r_in) & (rr <= r_out)
        Ipad = torch.nn.functional.pad(I, (r_out,) * 4, value=float("nan"))
        cuts, anns = [], []
        for y, x in zip(ys.tolist(), xs.tolist()):
            cut = Ipad[y:y + 2 * r_out + 1, x:x + 2 * r_out + 1]
            ann = ann0 & torch.isfinite(cut)
            cuts.append(torch.nan_to_num(cut)); anns.append(ann)
        c0 = self.n // 2
        best, scores = (float("inf"), 0.0, False), []
        for flip in (False, True):
            for ang in torch.arange(0.0, 180.0, step):
                self.set_rotation(float(ang), flip)
                K = self.basis_lam(self.lam_pivot[band:band + 1], self.chan_idx[band:band + 1])[0, det].to(img.dtype)
                T = K[c0 - r_out:c0 + r_out + 1, c0 - r_out:c0 + r_out + 1]
                tot = 0.0
                for cut, ann in zip(cuts, anns):
                    t_ = T[ann] / T[ann].norm().clamp(min=1e-30)
                    A = torch.stack([t_, torch.ones_like(t_) / t_.numel() ** 0.5], 1).double()
                    w = 1.0 / (sig ** 2 + (0.05 * cut[ann].clamp(min=0)) ** 2)
                    Aw = A * w[:, None].sqrt().double()
                    yw = (cut[ann] * w.sqrt()).double()
                    coef = torch.linalg.lstsq(Aw, yw[:, None]).solution
                    res = (yw - (Aw @ coef)[:, 0])
                    tot += float(torch.nn.functional.huber_loss(res, torch.zeros_like(res), delta=3.0, reduction="sum"))
                scores.append((float(ang), flip, tot))
                if tot < best[0]:
                    best = (tot, float(ang), flip)
        self.set_rotation(best[1], best[2])
        return best, scores, list(zip(ys.tolist(), xs.tolist()))

    def _mono(self, lam_um, chan, det):
        """Oversampled monochromatic PSF (no detector effects), flux-normalised by Parseval. lam_um (M,), chan (M,), det (D,)."""
        lam = lam_um[:, None, None, None] * 1e-6  # M 1 1 1 (m)
        W = (torch.einsum("mj,jhw->mhw", self.cg[chan], self.Z)[:, None] +
             torch.einsum("dj,jhw->dhw", self.cd, self.Z[:self.nzd])[None]) * 1e-7  # M D H W (m)
        E = self.A * torch.exp(1j * (2 * math.pi * W / lam))  # complex
        du = self.extent / self.N
        u = (torch.arange(self.N, device=E.device, dtype=torch.float32) - (self.N - 1) / 2) * du
        ph = -2 * math.pi * self.theta[None, :, None] * u[None, None, :] / (lam_um[:, None, None] * 1e-6)  # M no N
        Mx = torch.exp(1j * ph)  # M no N
        F = torch.einsum("mau,mduv,mbv->mdab", Mx, E, Mx)  # M D no no  (rows: y, cols: x)
        I = F.real ** 2 + F.imag ** 2
        dth = (self.scale / self.os) * ARCSEC
        I = I * (du ** 2 / (lam_um[:, None, None, None] * 1e-6)) ** 2 * dth ** 2 / ((self.A ** 2).sum() * du ** 2)
        return I

    def _detector(self, I, chan):
        """Detector pixel box, charge diffusion/jitter Gaussian and IPC as MTFs (per channel), on the oversampled grid."""
        p = self.pix[chan][:, None, None, None]  # arcsec
        sig = self.log_cd.exp()[chan][:, None, None, None] * p
        a = self.ipc[chan][:, None, None, None]
        fx, fy = self.fx[None, None], self.fy[None, None]
        mtf = (torch.sinc(fx * p) * torch.sinc(fy * p) * torch.exp(-2 * math.pi ** 2 * sig ** 2 * (fx ** 2 + fy ** 2))
               * (1 - 4 * a + 2 * a * (torch.cos(2 * math.pi * fx * p) + torch.cos(2 * math.pi * fy * p))))
        If = torch.fft.fft2(torch.fft.ifftshift(I, dim=(-2, -1)))
        return torch.fft.fftshift(torch.fft.ifft2(If * mtf).real, dim=(-2, -1))

    def basis_lam(self, lam_um, chan_idx):
        dets = torch.arange(self.ndet, device=lam_um.device)
        I = self._mono(lam_um.float(), chan_idx.long(), dets)
        I = self._detector(I, chan_idx.long())
        M, D = I.shape[:2]
        K = I.reshape(M, D, self.n, self.os, self.n, self.os).sum((3, 5))  # bin to the sky grid
        self.ee_inside = K.detach().sum((-2, -1))
        return K

    def basis(self):
        return self.basis_lam(self.lam_pivot, self.chan_idx)

    def weights(self, b, det, dx, dy):
        det = det if torch.is_tensor(det) else torch.full_like(dx, int(det), dtype=torch.long)
        if det.numel() == 1 and dx.numel() > 1:
            det = det.expand(dx.shape[0])
        return torch.nn.functional.one_hot(det.long(), self.ndet).to(dx.dtype)

    def penalty(self):  # weak physical prior: wavefront of order 100 nm rms
        return 1e-4 * (self.cg.pow(2).sum() + self.cd.pow(2).sum())
