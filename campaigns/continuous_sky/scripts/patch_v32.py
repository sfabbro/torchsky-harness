"""v31 -> v32 (5 Nov 2026): elaborated Gaussian splats, catalogue-free (--sky-coeffs splat2).
Sky coefficients a_k(x) = smooth_k(x) + sum_j c_jk G_j(x): a coarse bilinear grid (--splat-smooth-px) for diffuse light plus
anisotropic Gaussians (width floor --splat-sigma-floor, pixel-integrated, +-4 sigma windows, tile/size-class vectorised, chunked)
whose spectra c_j share the sky's RBF basis. What is exact and what is not:
  EXACT (to projected-CG tolerance / iteration cap, warm-started): the amplitudes (smooth coefficients and c_jk), given the shapes and
    the instrument, by (IRLS-weighted) linear least squares with the relative second-difference spectral penalty and
    phi(lam_ref).c_j >= 0 per splat (active-set projected conjugate gradients, Splat2Problem / projected_cg).
  APPROXIMATE: shapes and the instrument (Adam between solves, amplitudes frozen); the solve excludes TV / LAM_* terms (set to 0 for
    splat2); the amplitude S/N uses the conditional Fisher diagonal with an isotropic-Gaussian PSF; the densification statistics use
    the residual coadd (bilinear splat of the training residuals) matched-filtered with a Gaussian of each band's PSF core width.
Densification (per round, no catalogue, no detection image): prune splats with amplitude S/N < --splat-prune-snr (3); split splats
whose local residual is significantly better described by 2 Gaussians than 1 (chi-square test, dof 5 + nb, alpha = fpr / n tested);
add splats at the positive peaks of the band-chi-square residual statistic above a threshold taken from the empirical distribution of
the NEGATIVE-residual peaks (null_threshold); stop when the validation score has not improved for 2 rounds (best state restored)."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


BLOCK = r'''# ----------------------------------------------------------------------------- v32: elaborated Gaussian splats (--sky-coeffs splat2)
_GL4_X = (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526)
_GL4_W = (0.34785484513745385, 0.6521451548625461, 0.6521451548625461, 0.34785484513745385)
_WIN_CLASSES = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def splat_raster(mu, sigma, theta, H, W, nsub=4, mem_mb=256.0):
    """Pixel-integrated anisotropic unit-flux Gaussians. mu (J,2) [x, y] in target pixels (0-based centres, pixel i covers
    [i-0.5, i+0.5]), sigma (J,2) = (s1, s2) along the axes rotated by theta, theta (J,). Each Gaussian is integrated over each
    pixel by an nsub x nsub Gauss-Legendre rule (nsub=4: 16 points, exact for polynomials of degree 7 per axis) inside a window of
    +-ceil(4 sigma_max + 0.5) pixels, windows grouped in size classes and processed in chunks that stay under mem_mb.
    Differentiable in mu, sigma, theta (the integer window position is not). Returns (jid, lin, vals): splat index, flat pixel
    index (y*W+x) and value of every in-image footprint pixel (vals sum to ~1 - truncation (1e-4) - image edge loss)."""
    J = mu.shape[0]
    dev, dt = mu.device, mu.dtype
    if J == 0:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z, torch.zeros(0, dtype=dt, device=dev)
    gx, gw = (np.polynomial.legendre.leggauss(nsub) if nsub != 4 else (np.array(_GL4_X), np.array(_GL4_W)))
    s = torch.tensor(gx * 0.5, dtype=dt, device=dev)
    w1 = torch.tensor(gw * 0.5, dtype=dt, device=dev)
    ww = (w1[:, None] * w1[None, :]).reshape(-1)  # nsub^2, sums to 1
    sx, sy = torch.meshgrid(s, s, indexing="xy")
    sx, sy = sx.reshape(-1), sy.reshape(-1)
    hneed = torch.ceil(4.0 * torch.nan_to_num(sigma.detach(), nan=0.5, posinf=1e3).max(1).values + 0.5).clamp(1, _WIN_CLASSES[-1]).long()  # window cap +-128 px
    jids, lins, vals = [], [], []
    lo = 0
    for hc in _WIN_CLASSES:
        sel = ((hneed > lo) & (hneed <= hc)).nonzero().flatten() if lo > 0 else (hneed <= hc).nonzero().flatten()
        lo = hc
        if sel.numel() == 0:
            continue
        r = torch.arange(-hc, hc + 1, device=dev)
        P = (2 * hc + 1) ** 2
        n_ch = max(1, int(mem_mb * 1e6 / (P * nsub * nsub * 4 * 8)))
        for a in range(0, sel.numel(), n_ch):
            ix = sel[a:a + n_ch]
            c = torch.nan_to_num(mu[ix].detach(), nan=-1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6).round().long()  # n,2 (non-finite -> off-image, masked below)
            ox = c[:, 0, None, None] + r[None, None, :]  # n,1,2h+1
            oy = c[:, 1, None, None] + r[None, :, None]  # n,2h+1,1
            ox = ox.expand(-1, 2 * hc + 1, -1).reshape(ix.numel(), P)
            oy = oy.expand(-1, -1, 2 * hc + 1).reshape(ix.numel(), P)
            ok = (ox >= 0) & (ox < W) & (oy >= 0) & (oy < H)
            dx = (ox.to(dt) - mu[ix, 0, None])[:, :, None] + sx  # n,P,S
            dy = (oy.to(dt) - mu[ix, 1, None])[:, :, None] + sy
            ct, st = torch.cos(theta[ix])[:, None, None], torch.sin(theta[ix])[:, None, None]
            u, v = ct * dx + st * dy, -st * dx + ct * dy
            s1, s2 = sigma[ix, 0, None, None], sigma[ix, 1, None, None]
            g = torch.exp(-0.5 * ((u / s1) ** 2 + (v / s2) ** 2)) / (2 * math.pi * s1 * s2)
            val = (g * ww).sum(-1)  # n,P
            jj = ix[:, None].expand(-1, P)
            jids.append(jj[ok])
            lins.append((oy.clamp(0, H - 1) * W + ox.clamp(0, W - 1))[ok])
            vals.append(val[ok])
    return torch.cat(jids), torch.cat(lins), torch.cat(vals)


def splat_render(c, fp, K, H, W, mem_mb=256.0):
    """sum_j c_jk G_j(x) -> (K,H,W) from footprints fp = (jid, lin, vals); chunked over footprint pixels."""
    jid, lin, vals = fp
    out = torch.zeros(K, H * W, device=c.device, dtype=c.dtype)
    if jid.numel() == 0:
        return out.view(K, H, W)
    ch = max(1, int(mem_mb * 1e6 / (K * 4 * 3)))
    for a in range(0, jid.numel(), ch):
        sl = slice(a, a + ch)
        out = out.index_add(1, lin[sl], c[jid[sl]].T * vals[sl][None])
    return out.view(K, H, W)


class Splat2Coeffs(nn.Module):
    """s(x, lambda) coefficients a_k(x) = smooth_k(x) + sum_j c_jk G_j(x). Shapes (mu, rho, theta) are Parameters (Adam); the linear
    amplitudes (smooth: K x hs x ws coarse grid, bilinearly upsampled; c: J x K splat spectra in the sky's RBF basis) are buffers
    set by solve_splat2_amplitudes. sigma_i = floor + softplus(rho_i) >= floor."""

    def __init__(self, K, H, W, smooth_px=4, sigma_floor=0.5, smooth_init=None, nsub=4):
        super().__init__()
        self.K, self.H, self.W, self.px, self.floor, self.nsub = K, H, W, smooth_px, sigma_floor, nsub
        self.hs, self.ws = int(math.ceil(H / smooth_px)), int(math.ceil(W / smooth_px))
        dev = smooth_init.device if smooth_init is not None else torch.device("cpu")
        self.register_buffer("smooth", torch.zeros(K, self.hs, self.ws, device=dev))
        self.register_buffer("c", torch.zeros(0, K, device=dev))
        self.mu = nn.Parameter(torch.zeros(0, 2, device=dev))
        self.rho = nn.Parameter(torch.zeros(0, 2, device=dev))
        self.theta = nn.Parameter(torch.zeros(0, device=dev))
        self._fp = None
        if smooth_init is not None:  # low-pass of the coadd initialisation: area average onto the coarse grid
            with torch.no_grad():
                self.smooth.copy_(F.adaptive_avg_pool2d(smooth_init[None], (self.hs, self.ws))[0])

    @property
    def n(self):
        return self.mu.shape[0]

    def sigma(self):
        return self.floor + F.softplus(self.rho)

    def upsample(self, smooth=None):
        s = self.smooth if smooth is None else smooth
        return F.interpolate(s[None], size=(self.H, self.W), mode="bilinear", align_corners=False)[0]

    def footprints(self, detach=False):
        if detach and self._fp is not None:
            return self._fp
        fp = splat_raster(self.mu, self.sigma(), self.theta, self.H, self.W, self.nsub)
        if detach:
            self._fp = tuple(x.detach() for x in fp)
            return self._fp
        return fp

    def forward(self):
        return self.upsample() + splat_render(self.c, self.footprints(), self.K, self.H, self.W)

    def invalidate(self):
        self._fp = None

    def set_splats(self, mu, sigma, theta, c):
        dev = self.smooth.device
        self.mu = nn.Parameter(mu.detach().clone().to(dev))
        self.rho = nn.Parameter(_inv_softplus((sigma.detach() - self.floor).clamp(min=0.05)).to(dev))
        self.theta = nn.Parameter(theta.detach().clone().to(dev))
        self.c = c.detach().clone().to(dev)
        self.invalidate()

    def sanitize(self):
        """Drop splats with non-finite parameters or a centre outside the image (the Adam step can push one out); clamp nothing
        silently. Returns the number dropped."""
        if self.n == 0:
            return 0
        with torch.no_grad():
            ok = (torch.isfinite(self.mu).all(1) & torch.isfinite(self.rho).all(1) & torch.isfinite(self.theta) & torch.isfinite(self.c).all(1)
                  & (self.mu[:, 0] >= -0.5) & (self.mu[:, 0] <= self.W - 0.5) & (self.mu[:, 1] >= -0.5) & (self.mu[:, 1] <= self.H - 0.5))
        nbad = int((~ok).sum())
        if nbad:
            self.keep(ok)
        return nbad

    def keep(self, mask):
        self.set_splats(self.mu[mask], self.sigma()[mask], self.theta[mask], self.c[mask])


def _inv_softplus(y):
    return torch.log(torch.expm1(y.clamp(min=1e-6)))


def projected_cg(matvec, b, x0, con, iters=60, tol=1e-6, minv=None):
    """min 0.5 x'Hx - b'x subject to x[con] >= 0 (H = matvec, symmetric positive semi-definite), flat vectors. Active-set conjugate
    gradients (More & Toraldo 1991 style): CG on the variables not held at a bound, the step truncated where a constrained variable
    reaches 0 (that variable joins the active set), the active set re-derived from the exact gradient after every CG sweep (a variable
    at 0 with negative gradient is released). `minv`: optional diagonal (Jacobi) preconditioner, the inverse of an approximate diag(H).
    `iters` caps the number of matvecs. Returns (x, info)."""
    x = x0.clone()
    x[con] = x[con].clamp(min=0)
    bn = float(b.norm()) + 1e-30
    g = matvec(x) - b
    nmv = 1
    pg = float("inf")
    while nmv < iters:
        free = ~(con & (x <= 0) & (g > 0))
        r = -g * free
        pg = float(r.norm())
        if pg <= tol * bn:
            break
        z = r if minv is None else r * minv * free
        p, rs, hit = z.clone(), float(r @ z), False
        while nmv < iters:
            Hp = matvec(p) * free
            nmv += 1
            pHp = float(p @ Hp)
            if pHp <= 0:
                break
            alpha = rs / pHp
            neg = con & free & (p < 0)
            amax = float((-x[neg] / p[neg]).min()) if bool(neg.any()) else float("inf")
            if alpha >= amax:
                x = x + amax * p
                x[con] = x[con].clamp(min=0)
                hit = True
                break
            x = x + alpha * p
            r = r - alpha * Hp
            if float(r.norm()) <= tol * bn * 0.1:
                break
            z = r if minv is None else r * minv * free
            rs_new = float(r @ z)
            p = z + (rs_new / rs) * p
            rs = rs_new
        x[con & (x < 0)] = 0
        g = matvec(x) - b
        nmv += 1
    free = ~(con & (x <= 0) & (g > 0))
    pg = float((g * free).norm())
    return x, {"matvecs": nmv, "proj_grad_rel": pg / bn, "n_active": int((con & (x <= 0)).sum())}


def _second_diff(K, device, dtype=torch.float32):
    D = torch.zeros(K - 2, K, device=device, dtype=dtype)
    for i in range(K - 2):
        D[i, i:i + 3] = torch.tensor([1.0, -2.0, 1.0], device=device, dtype=dtype)
    return D


def _householder(v):
    """Orthogonal Q (K,K) with first column v/|v|."""
    K = v.numel()
    e = torch.zeros(K, dtype=v.dtype, device=v.device)
    e[0] = 1.0
    u = v / v.norm()
    w = u - e
    if float(w.norm()) < 1e-12:
        return torch.eye(K, dtype=v.dtype, device=v.device)
    w = w / w.norm()
    return torch.eye(K, dtype=v.dtype, device=v.device) - 2.0 * torch.outer(w, w)  # H e = u


class Splat2Problem:
    """The linear sub-problem of --sky-coeffs splat2. With the shapes (mu, sigma, theta), the PSF, geometry, background, stripes and
    gain frozen, the sky part of every prediction is linear in x = (smooth coefficients, splat spectra c):
        m_i = off_i + (Lin x)_i,   off_i = prediction of the same model with a zero sky (background, stripes),
    and the training objective is  (1/ntot) sum_i W_i (d_i - off_i - (Lin x)_i)^2 + pen(x),  W = ivar (Gaussian) or ivar * IRLS weight
    of the Huber losses. pen = the relative second-difference spectral smoothness of SpectralSky.penalty evaluated on the amplitudes
    (scale fixed at the current values, i.e. the same reweighting as the detached scale of SpectralSky.penalty). Constraint: the
    broadband amplitude phi(lam_ref).c_j >= 0 for every splat; in coordinates z_j = Q^T c_j with Q orthogonal and first column
    phi(lam_ref)/|phi(lam_ref)| it is the box constraint z_j0 >= 0 (the objective is unchanged: Q is orthogonal)."""

    def __init__(self, model, batches, ntot, phi_ref, band_index, loss_kind="gauss", delta=3.0, Phi=None):
        self.model, self.batches, self.ntot = model, batches, ntot
        self.sky, self.co = model.sky, model.sky.coeffs
        self.band_index, self.loss_kind, self.delta = band_index, loss_kind, delta
        self.sp_scale = 1.0
        self.Phi = Phi
        dev = self.co.smooth.device
        self.K = self.co.K
        self.Q = _householder(phi_ref.to(dev).float())
        self.D2 = _second_diff(self.K, dev)
        self.cache = []
        with torch.no_grad():
            _, CB = model.prepare()
            self.CB_shape = CB.shape
            CB0 = torch.zeros_like(CB)
            for e in batches:
                b = band_index[e["band"]]
                i = e["eidx"]
                det = e["detv"]
                x, y = model.geom(i, det, e["tx"], e["ty"], e["dx"], e["dy"])
                w = model.psf.weights(b, det, e["dx"], e["dy"])
                gain = model.bkg.gain(vec(i, e["dx"]))
                off = model.predict(CB0, None, e)
                self.cache.append({"b": b, "x": x, "y": y, "w": w, "gain": gain, "y_res": e["d"] - off, "ivar": e["ivar"], "W": e["ivar"]})
        self.fp = None

    # --- one linear map
    def _lin(self, CBb, c):
        v = sample(CBb, c["x"], c["y"])
        return c["gain"] * (c["w"] * v.T).sum(1)

    def weight_map(self):
        """Om_b(x) = sum_i W_i lambda_i(x) (bilinear weights of datum i at its target position), (nb,H,W)."""
        H, W = self.co.H, self.co.W
        om = torch.zeros(len(self.band_index), H * W, device=self.co.smooth.device)
        with torch.no_grad():
            for c in self.cache:
                x0, y0 = torch.floor(c["x"]), torch.floor(c["y"])
                fx, fy = c["x"] - x0, c["y"] - y0
                x0, y0 = x0.long(), y0.long()
                for dxi, wx in ((0, 1 - fx), (1, fx)):
                    for dyi, wy in ((0, 1 - fy), (1, fy)):
                        xi, yi = x0 + dxi, y0 + dyi
                        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                        om[c["b"]].index_add_(0, yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1), c["W"] * wx * wy * ok)
        return om.view(len(self.band_index), H, W)

    def approx_diag(self, smooth, cmat, sig, sig_psf, Phi):
        """Approximate diagonal of M = G/ntot + P in the solve's coordinates (smooth, z = Q^T c), for the Jacobi preconditioner:
        data part from Om and the analytic profile overlap (see amplitude_snr), penalty part from the diagonals of the quadratic forms."""
        om = self.weight_map()
        K = self.K
        px2 = float(self.co.px) ** 2
        omc = F.adaptive_avg_pool2d(om[None], (self.co.hs, self.co.ws))[0]  # nb hs ws
        ds = px2 * (4.0 / 9.0) * torch.einsum("bk,bhw->khw", Phi ** 2, omc) / self.ntot
        ds = ds + (self.w_sp / self.ntot) * 4.0
        dz = torch.zeros_like(cmat)
        if cmat.shape[0]:
            omj = sample(om, self.co.mu[:, 0].detach(), self.co.mu[:, 1].detach()).T  # J nb
            se1 = torch.sqrt(sig[:, 0:1] ** 2 + sig_psf[None] ** 2)
            se2 = torch.sqrt(sig[:, 1:2] ** 2 + sig_psf[None] ** 2)
            pq = Phi @ self.Q  # nb K (rotated)
            dz = torch.einsum("bm,jb->jm", pq ** 2, omj / (4 * math.pi * se1 * se2)) / self.ntot
            if self.lam > 0:
                dz = dz + (6.0 * self.coef_c / self.sc_c ** 2)[:, None]
        if self.lam > 0:
            ds = ds + 6.0 * self.coef_s / self.sc_s ** 2
        return torch.cat([ds.flatten(), dz.flatten()]).clamp(min=1e-12)

    def set_irls(self, smooth, cmat):
        """W_i from the residuals at the current solution (Huber losses); Gaussian: W = ivar."""
        with torch.no_grad():
            A = self.co.upsample(smooth) + splat_render(cmat, self.fp, self.K, self.co.H, self.co.W)
            CB = self.model.prepare(A=A)[1]
            for c in self.cache:
                if self.loss_kind == "gauss" or HUBER_DELTA is None:
                    c["W"] = c["ivar"]
                    continue
                z = (c["ivar"] * (c["y_res"] - self._lin(CB[c["b"]], c)) ** 2).sqrt()
                d = self.delta
                f = torch.where(z <= d, torch.ones_like(z), d / z.clamp(min=1e-12)) if self.loss_kind == "huber" else 1.0 / torch.sqrt(1.0 + z * z / (d * d))
                c["W"] = c["ivar"] * f

    def _adjoint(self, gCB, theta0):
        A0 = self.co.upsample(theta0[0]) + splat_render(theta0[1], self.fp, self.K, self.co.H, self.co.W)
        return A0

    def setup_graph(self):
        """Graph of the linear map (smooth, c) -> CB, built once per solve; its adjoint is applied with retain_graph."""
        s0 = torch.zeros_like(self.co.smooth, requires_grad=True)
        c0 = torch.zeros_like(self.co.c, requires_grad=True)
        A0 = self.co.upsample(s0) + splat_render(c0, self.fp, self.K, self.co.H, self.co.W)
        self.graph = (s0, c0, self.model.prepare(A=A0)[1])

    def gram(self, smooth, cmat):
        """(Lin^T W Lin x) for x = (smooth, c)."""
        with torch.no_grad():
            A = self.co.upsample(smooth) + splat_render(cmat, self.fp, self.K, self.co.H, self.co.W)
            CB = self.model.prepare(A=A)[1]
        gCB = torch.zeros_like(CB)
        for c in self.cache:
            leaf = CB[c["b"]].detach().requires_grad_(True)
            y = self._lin(leaf, c)
            g, = torch.autograd.grad(y, leaf, grad_outputs=c["W"] * y.detach())
            gCB[c["b"]] += g
        return self._vjp(gCB)

    def forward_lin(self, smooth, cmat):
        """Lin x as one vector over all training data (batch order), for tests."""
        with torch.no_grad():
            A = self.co.upsample(smooth) + splat_render(cmat, self.fp, self.K, self.co.H, self.co.W)
            CB = self.model.prepare(A=A)[1]
            return torch.cat([self._lin(CB[c["b"]], c) for c in self.cache])

    def rhs(self):
        gCB = torch.zeros(self.CB_shape, device=self.co.smooth.device)
        for c in self.cache:
            leaf = torch.zeros(self.CB_shape[1:], device=gCB.device, requires_grad=True)
            y = self._lin(leaf, c)
            g, = torch.autograd.grad(y, leaf, grad_outputs=c["W"] * c["y_res"])
            gCB[c["b"]] += g
        return self._vjp(gCB)

    def _vjp(self, gCB):
        s0, c0, CB0 = self.graph
        gs, gc = torch.autograd.grad(CB0, [s0, c0], grad_outputs=gCB, retain_graph=True, allow_unused=True)
        return (torch.zeros_like(s0) if gs is None else gs), (torch.zeros_like(c0) if gc is None else gc)

    # --- penalty (quadratic form x'Px; returns Px)
    def set_penalty(self, smooth, cmat, sigma):
        s = self.sky
        K, H, W = self.K, self.co.H, self.co.W
        self.lam = s.lam_smooth
        wsum = sum(float(c["ivar"].sum()) for c in self.cache)
        self.w_sp = float(self.co.px) ** 2 * wsum / (H * W * len(self.band_index)) * self.sp_scale
        if self.lam <= 0:
            return
        if s.relative:
            self.sc_s = (smooth.abs().mean(0, keepdim=True) + s.floor_sigma * s.sig_ref)
        else:
            self.sc_s = torch.full_like(smooth[:1], s.sig_ref)
        self.coef_s = self.lam / ((K - 2) * smooth.shape[1] * smooth.shape[2])
        area = 2 * math.pi * sigma[:, 0] * sigma[:, 1]
        if s.relative:
            self.sc_c = cmat.abs().mean(1) + s.floor_sigma * s.sig_ref * area
        else:
            self.sc_c = s.sig_ref * area
        self.coef_c = self.lam * 2.0 * area / ((K - 2) * H * W)  # footprint 4 pi s1 s2 (area of the 2-sigma ellipse), pixels

    def spatial_prior(self, smooth):
        """P_sp s: random-walk prior on the smooth grid, (w_sp / ntot) sum_neighbours (s_a - s_b)^2 per coefficient, with
        w_sp = (cell area in target px) x (mean inverse variance per target pixel and band): neighbouring cells may differ by about
        the noise of a cell's own measurement. Without it a 4 px cell grid is degenerate with extended splats (diffuse light and a
        sigma ~ 3 px galaxy are the same thing). Neumann boundaries."""
        out = torch.zeros_like(smooth)
        dxs = smooth[:, :, 1:] - smooth[:, :, :-1]
        out[:, :, 1:] += dxs
        out[:, :, :-1] -= dxs
        dys = smooth[:, 1:, :] - smooth[:, :-1, :]
        out[:, 1:, :] += dys
        out[:, :-1, :] -= dys
        return (self.w_sp / self.ntot) * out

    def pen_apply(self, smooth, cmat):
        sp = self.spatial_prior(smooth)
        if self.lam <= 0:
            return sp, torch.zeros_like(cmat)
        d2 = torch.einsum("jk,khw->jhw", self.D2, smooth)
        ps = self.coef_s * torch.einsum("jk,jhw->khw", self.D2, d2 / self.sc_s ** 2) + sp
        if cmat.shape[0] == 0:
            return ps, torch.zeros_like(cmat)
        e2 = cmat @ self.D2.T  # J, K-2
        pc = (self.coef_c / self.sc_c ** 2)[:, None] * (e2 @ self.D2)
        return ps, pc

    # --- the solve
    def system(self, smooth, cmat, sig):
        """The quadratic problem  min x'Mx - 2 b'x  in the coordinates x = (smooth, z = Q^T c), z_j0 >= 0: returns a dict with
        matvec (v -> 2 M v), b2 (= 2 b), pack/unpack, the constraint mask, the Jacobi inverse diagonal."""
        co = self.co
        dev = co.smooth.device
        J, K = cmat.shape
        ns = smooth.numel()
        con = torch.zeros(ns + J * K, dtype=torch.bool, device=dev)
        con[ns::K] = True
        Q = self.Q

        def unpack(v):
            return v[:ns].view_as(smooth), (v[ns:].view(J, K) @ Q.T)

        def pack(s_, c_):
            return torch.cat([s_.flatten(), (c_ @ Q).flatten()])
        self.set_irls(smooth, cmat)
        self.set_penalty(smooth, cmat, sig)
        rs_, rc_ = self.rhs()
        b = pack(rs_, rc_) / self.ntot

        def matvec(v):  # H = 2 M
            s_, c_ = unpack(v)
            gs, gc = self.gram(s_, c_)
            ps, pc = self.pen_apply(s_, c_)
            return 2.0 * pack(gs / self.ntot + ps, gc / self.ntot + pc)
        minv = None
        if self.Phi is not None:
            minv = 1.0 / (2.0 * self.approx_diag(smooth, cmat, sig, psf_core_sigma(self.model), self.Phi))
        return {"matvec": matvec, "b2": 2.0 * b, "pack": pack, "unpack": unpack, "con": con, "minv": minv}

    def solve(self, iters=40, tol=1e-5, irls=2):
        co = self.co
        sig = co.sigma().detach()
        co.invalidate()
        with torch.enable_grad():
            self.fp = co.footprints(detach=True)
            self.setup_graph()
        smooth, cmat = co.smooth.clone(), co.c.clone()
        n_pass = 1 if (self.loss_kind == "gauss" or HUBER_DELTA is None) else irls
        tot_mv, info = 0, {}
        for ip in range(n_pass):
            sy = self.system(smooth, cmat, sig)
            x, info = projected_cg(sy["matvec"], sy["b2"], sy["pack"](smooth, cmat), sy["con"], iters=iters, tol=tol, minv=sy["minv"])
            tot_mv += info["matvecs"]
            smooth, cmat = sy["unpack"](x)
        info["matvecs"] = tot_mv
        info["irls_passes"] = n_pass
        with torch.no_grad():
            co.smooth.copy_(smooth)
            co.c = cmat.detach().clone()
        self.graph = None
        return info


# ----------------------------------------------------------------------------- v32: catalogue-free densification
def psf_core_sigma(model):
    """(nb,) Gaussian-equivalent sigma [target px] of each band's PSF core kernel (second moment / 2)."""
    with torch.no_grad():
        B = model.psf.basis()[:, 0]  # nb n n
        B = B / B.sum((-2, -1), keepdim=True)
        n = B.shape[-1]
        r = torch.arange(n, device=B.device, dtype=B.dtype) - n // 2
        r2 = r[None, :] ** 2 + r[:, None] ** 2
        return torch.sqrt(((B * r2).sum((-2, -1)) / 2.0).clamp(min=0.25))


def residual_coadd(model, batches, band_index, loss_kind="gauss", delta=3.0, chunk=1_000_000):
    """Training residuals of the full current model (sky + background + stripes + gain) on the target grid, per band:
    num_b(x) = sum_i W_i r_i lambda_i(x), om_b(x) = sum_i W_i lambda_i(x), with lambda_i(x) the bilinear weights of datum i at its
    (geometry-corrected) target position and W = ivar (x Huber IRLS weight for the robust losses). R_b = num_b / om_b is the residual
    coadd, om_b its inverse variance (no resampling covariance, no outlier rejection beyond the IRLS weight)."""
    H, W = model.sky.coeffs.H, model.sky.coeffs.W
    nb = len(band_index)
    dev = model.sky.coeffs.smooth.device
    num = torch.zeros(nb, H * W, device=dev)
    om = torch.zeros(nb, H * W, device=dev)
    with torch.no_grad():
        S, CB = model.prepare()
        for e in batches:
            b = band_index[e["band"]]
            m = model.predict(CB, None, e)
            r = e["d"] - m
            Wt = e["ivar"]
            if loss_kind != "gauss" and HUBER_DELTA is not None:
                z = (Wt * r * r).sqrt()
                f = torch.where(z <= delta, torch.ones_like(z), delta / z.clamp(min=1e-12)) if loss_kind == "huber" else 1.0 / torch.sqrt(1.0 + z * z / (delta * delta))
                Wt = Wt * f
            x, y = model.geom(e["eidx"], e["detv"], e["tx"], e["ty"], e["dx"], e["dy"])
            x0, y0 = torch.floor(x), torch.floor(y)
            fx, fy = x - x0, y - y0
            x0, y0 = x0.long(), y0.long()
            for dxi, wx in ((0, 1 - fx), (1, fx)):
                for dyi, wy in ((0, 1 - fy), (1, fy)):
                    xi, yi = x0 + dxi, y0 + dyi
                    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                    lin = (yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1))
                    wgt = (Wt * wx * wy) * ok
                    om[b].index_add_(0, lin, wgt)
                    num[b].index_add_(0, lin, wgt * r)
    return num.view(nb, H, W), om.view(nb, H, W)


def significance_maps(num, om, sig_psf):
    """PSF-matched point-source filter of the residual coadd, per band: Z_b = sum Om R p / sqrt(sum Om p^2) with p the band's
    Gaussian PSF core (sigma sig_psf_b), then divided by the robust (1.4826 MAD) scatter of Z_b over covered pixels, so that Z_b is
    ~N(0,1) for pure noise whatever the noise calibration. Returns Z (nb,H,W), valid (H,W) bool (every band covered), scale (nb,)."""
    nb = num.shape[0]
    n = int(2 * math.ceil(4 * float(sig_psf.max())) + 1)
    p = gauss_kernel(sig_psf, n, num.device)[:, None]  # nb 1 n n
    conv = lambda a, k: F.conv2d(a[None], k, padding=n // 2, groups=nb)[0]
    N = conv(num, p)
    D2 = conv(om, p * p)
    cov = D2 > 0.05 * D2[D2 > 0].median() if bool((D2 > 0).any()) else D2 > 0
    Z = torch.where(cov, N / D2.clamp(min=1e-30).sqrt(), torch.zeros_like(N))
    scale = torch.ones(nb, device=num.device)
    for b in range(nb):
        v = Z[b][cov[b]]
        if v.numel() > 10:
            scale[b] = (1.4826 * (v - v.median()).abs().median()).clamp(min=1e-6)
    return Z / scale[:, None, None], cov.all(0), scale


def local_maxima(T, r):
    k = 2 * r + 1
    return (T == F.max_pool2d(T[None, None], k, stride=1, padding=r)[0, 0]) & (T > 0)


def null_threshold(Tneg_peaks, fpr):
    """Detection threshold from the symmetric null: peaks of the negative-residual statistic (same construction as the positive one)
    give the null distribution of positive peaks. Robust exponential-tail extrapolation from the BODY of that distribution (the
    top of it is contaminated by genuine negative residuals: over-subtraction around sources the model has not captured yet):
    u = median of the N null peaks, s = (median of the peaks above u - u) / ln 2; the expected number of null peaks above t > u is
    (N/2) exp(-(t-u)/s), and tau is the level where that equals fpr (expected false additions per round); equivalent to the (1-p)
    quantile of the null peaks with p = fpr / N. The chi-square-like true tail decays faster than the exponential fitted at the
    median, so tau is conservative. Returns (tau, info); tau = inf when fewer than 30 null peaks exist."""
    v = torch.sort(Tneg_peaks).values
    N = v.numel()
    if N < 30:
        return float("inf"), {"n_null_peaks": N}
    u = float(v[N // 2])
    up = v[N // 2:]
    s = max((float(up[up.numel() // 2]) - u) / math.log(2.0), 1e-6)
    tau = u + s * math.log(N / (2.0 * fpr))
    return tau, {"n_null_peaks": N, "u": u, "scale": s, "max_null": float(v[-1])}


def init_shapes(Zpos_sum, peaks_xy, sig_psf_mean, floor, valid):
    """Initial (mu, sigma, theta) of new splats from the second moment of the positive band-summed significance map A = sum_b (Z_b)_+
    around each peak, over the connected region where A >= A_peak / 2 (within R = clip(ceil(4 sqrt(2) sig_psf) + 2, 4, 12) pixels).
    For a Gaussian profile the covariance inside its half-maximum contour is 0.307 x the full covariance (2 ln 2 = 1.386 =
    chi^2 quantile; M2 / (2 M0) = [2 - (c+2) e^{-c/2}] / (2 (1 - e^{-c/2})) with c = 2 ln 2), so the moment is divided by 0.307
    (a geometric constant, not a tuning parameter). The matched-filter map is the source convolved with the PSF twice (data PSF, then
    the filter), so 2 sig_psf^2 is subtracted from the eigenvalues, which are then clipped to [floor^2, (R/3)^2]. Centre = centroid of
    the same weights."""
    H, W = Zpos_sum.shape
    dev = Zpos_sum.device
    R = int(min(12, max(4, math.ceil(4 * math.sqrt(2) * sig_psf_mean) + 2)))
    r = torch.arange(-R, R + 1, device=dev)
    oy, ox = torch.meshgrid(r, r, indexing="ij")
    inside = (ox ** 2 + oy ** 2) <= R * R
    mu, sg, th = [], [], []
    for (x0, y0) in peaks_xy.tolist():
        xs, ys = (x0 + ox).clamp(0, W - 1), (y0 + oy).clamp(0, H - 1)
        inb = ((x0 + ox) >= 0) & ((x0 + ox) < W) & ((y0 + oy) >= 0) & ((y0 + oy) < H) & inside
        A = Zpos_sum[ys, xs] * inb
        top = float(A[R, R])
        reg = (A >= 0.5 * top) & inb
        seed = torch.zeros_like(reg)
        seed[R, R] = True
        for _ in range(R):  # connected component of the half-maximum region containing the peak
            grown = (F.max_pool2d(seed[None, None].float(), 3, stride=1, padding=1)[0, 0] > 0) & reg
            if bool((grown == seed).all()):
                break
            seed = grown
        w = A * seed
        sw = w.sum().clamp(min=1e-12)
        cx, cy = float((w * ox).sum() / sw), float((w * oy).sum() / sw)
        dx, dy = ox - cx, oy - cy
        cxx, cyy, cxy = [float((w * q).sum() / sw) / 0.307 for q in (dx * dx, dy * dy, dx * dy)]
        blur = 2.0 * sig_psf_mean ** 2
        C = torch.tensor([[cxx - blur, cxy], [cxy, cyy - blur]])
        ev, evec = torch.linalg.eigh(C)
        ev = ev.clamp(min=floor ** 2).clamp(max=(R / 3.0) ** 2)
        v1 = evec[:, 1]
        mu.append([x0 + cx, y0 + cy])
        sg.append([float(ev[1].sqrt()), float(ev[0].sqrt())])
        th.append(math.atan2(float(v1[1]), float(v1[0])))
    if not mu:
        return torch.zeros(0, 2, device=dev), torch.zeros(0, 2, device=dev), torch.zeros(0, device=dev)
    return (torch.tensor(mu, device=dev), torch.tensor(sg, device=dev), torch.tensor(th, device=dev))


def band_phi(basis, lam_nodes, w_nodes):
    """Phi_bk = sum_q w_bq phi_k(lam_bq): band-integrated basis (nb,K)."""
    with torch.no_grad():
        nb, q = lam_nodes.shape
        phi = basis(lam_nodes.flatten()).view(nb, q, -1)
        return (phi * w_nodes[:, :, None]).sum(1)


def amplitude_snr(co, om, sig_psf, Phi, q1):
    """S/N of the broadband amplitude z_j0 = q1.c_j of every splat from the Fisher diagonal of the linear problem, in the conditional
    approximation (other parameters fixed; the full marginal variance is larger): a unit step of z_j0 changes the band-b sky by
    s_bj = Phi_b . q1 times the unit-flux profile, seen through the PSF as a Gaussian of sigma_i^2 + sig_psf_b^2, so
    F_j = sum_b s_bj^2 Om_b(mu_j) / (4 pi sigma1_b sigma2_b)  (sum_x g^2 = 1/(4 pi sigma1 sigma2)). Om_b = inverse-variance weight
    per target pixel (residual_coadd). Returns snr (J,), F (J,)."""
    J = co.n
    if J == 0:
        z = torch.zeros(0, device=om.device)
        return z, z
    with torch.no_grad():
        sig = co.sigma()
        s_b = Phi @ q1  # nb
        omj = sample(om, co.mu[:, 0], co.mu[:, 1]).T  # J nb
        se1 = torch.sqrt(sig[:, 0:1] ** 2 + sig_psf[None] ** 2)
        se2 = torch.sqrt(sig[:, 1:2] ** 2 + sig_psf[None] ** 2)
        Fj = ((s_b[None] ** 2) * omj / (4 * math.pi * se1 * se2)).sum(1)
        t = co.c @ q1
        return t * Fj.sqrt(), Fj


def chi2_isf(alpha, dof):
    """Upper-tail chi-square quantile (Wilson-Hilferty; relative error < 1% for dof >= 3)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - min(max(alpha, 1e-12), 0.5))
    a = 2.0 / (9.0 * dof)
    return dof * (1.0 - a + z * math.sqrt(a)) ** 3


def _stamp_images(mu, sigma, theta, sig_psf, S):
    """(n,nb,S,S): pixel-integrated unit-flux Gaussians of covariance R diag(sigma^2) R' + sig_psf_b^2 I at stamp-local mu."""
    n, nb = mu.shape[0], sig_psf.shape[0]
    out = []
    for b in range(nb):
        jid, lin, val = splat_raster(mu, torch.sqrt(sigma ** 2 + sig_psf[b] ** 2), theta, S, S)
        out.append(torch.zeros(n * S * S, device=mu.device, dtype=mu.dtype).index_add(0, jid * S * S + lin, val).view(n, S, S))
    return torch.stack(out, 1)


def _fit_stamps(T, Om, mu, sig, th, sig_psf, floor, steps, lr=0.1):
    """Weighted least-squares fit of m = 1 or 2 Gaussians (shapes by Adam, per-band amplitudes >= 0 in closed form) to the stamps
    T (n,nb,S,S) with weights Om. mu/sig (n,m,2), th (n,m). Returns chi2 (n,), mu, sig, th, amplitudes (n,m,nb)."""
    n, m = mu.shape[:2]
    S = T.shape[-1]
    mu_p = mu.clone().requires_grad_(True)
    rho = _inv_softplus((sig - floor).clamp(min=0.05)).clone().requires_grad_(True)
    th_p = th.clone().requires_grad_(True)
    OT = (Om * T).flatten(2)
    Of = Om.flatten(2)
    T2 = (Om * T * T).sum((1, 2, 3))

    def evaluate():
        sg = floor + F.softplus(rho)
        G = _stamp_images(mu_p.reshape(n * m, 2), sg.reshape(n * m, 2), th_p.reshape(n * m), sig_psf, S).view(n, m, -1, S * S)  # n m nb P
        r = torch.einsum("nbp,nmbp->nmb", OT, G)
        M = torch.einsum("nbp,nmbp,nlbp->nmlb", Of, G, G)
        if m == 1:
            a = (r[:, 0] / M[:, 0, 0].clamp(min=1e-30)).clamp(min=0)[:, None]
        else:
            det = (M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] ** 2).clamp(min=1e-30)
            af = torch.stack([(M[:, 1, 1] * r[:, 0] - M[:, 0, 1] * r[:, 1]) / det, (M[:, 0, 0] * r[:, 1] - M[:, 0, 1] * r[:, 0]) / det], 1)
            a1 = torch.stack([(r[:, 0] / M[:, 0, 0].clamp(min=1e-30)).clamp(min=0), torch.zeros_like(r[:, 0])], 1)
            a2 = torch.stack([torch.zeros_like(r[:, 1]), (r[:, 1] / M[:, 1, 1].clamp(min=1e-30)).clamp(min=0)], 1)

            def gain(a):
                return 2 * (a * r).sum(1) - torch.einsum("nmb,nmlb,nlb->nb", a, M, a)
            feas = (af >= 0).all(1)
            gf = torch.where(feas, gain(af), torch.full_like(gain(af), -1e30))
            g1, g2 = gain(a1), gain(a2)
            best = torch.stack([gf, g1, g2], 0).argmax(0)  # per (n, b)
            a = torch.where(best[:, None] == 0, af, torch.where(best[:, None] == 1, a1, a2))
        gain_ = 2 * (a * r).sum(1) - torch.einsum("nmb,nmlb,nlb->nb", a, M, a)
        return T2 - gain_.sum(1), a, sg

    opt = torch.optim.Adam([mu_p, rho, th_p], lr=lr)
    for k in range(steps):
        for g_ in opt.param_groups:  # cosine decay of the step size
            g_["lr"] = lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * k / max(steps, 1))))
        opt.zero_grad()
        c2, _, _ = evaluate()
        c2.sum().backward()
        opt.step()
        with torch.no_grad():
            mu_p.clamp_(1.0, S - 2.0)
    with torch.no_grad():
        c2, a, sg = evaluate()
    return c2, mu_p.detach(), sg.detach(), th_p.detach(), a.detach()


def split_tests(co, R, om, Tpeaks_xy, tau, sig_psf, Phi, infl, fpr, floor, steps=60, max_tests=2000):
    """Which splats to split. Candidates = splats with a residual statistic peak (either sign) above the detection threshold tau
    within 2 sigma_max + 2 sigma_psf of their centre. For each: the local 'total signal' T_b = residual coadd + the splat's own
    model (a_jb x its profile, seen through a Gaussian PSF of sigma_psf_b) in a stamp; fit 1 Gaussian vs 2 Gaussians (shapes Adam
    from the splat / from a split of the 1-Gaussian fit along each axis, best of the two restarts; per-band amplitudes >= 0, weights
    Om). Split if (chi2_1 - chi2_2) / infl exceeds the chi-square quantile of dof = 5 + nb at alpha = fpr / n_candidates (5 = extra
    mu, sigma, theta; nb = extra amplitudes: conservative since the amplitudes are spectrally smooth). infl = mean squared robust
    scatter of the significance maps (noise mis-calibration). Returns list of dicts and a log dict."""
    J = co.n
    log = {"candidates": 0, "split": 0}
    if J == 0 or Tpeaks_xy.shape[0] == 0 or not math.isfinite(tau):
        return [], log
    dev = co.smooth.device
    with torch.no_grad():
        sig = co.sigma()
        rad = 2 * sig.max(1).values + 2 * float(sig_psf.mean())
        d = torch.cdist(co.mu, Tpeaks_xy.float().to(dev))
        inside = (torch.isfinite(co.mu).all(1) & torch.isfinite(sig).all(1) & (co.mu[:, 0] >= 0) & (co.mu[:, 0] <= co.W - 1)
                  & (co.mu[:, 1] >= 0) & (co.mu[:, 1] <= co.H - 1))  # stamps are cut from the image: centres must be on it
        log["skipped_off_image"] = int((~inside).sum())
        cand = ((d < rad[:, None]).any(1) & inside).nonzero().flatten()[:max_tests]
        log["candidates"] = int(cand.numel())
        if cand.numel() == 0:
            return [], log
        nb, H, W = R.shape
        hcls = torch.clamp(torch.ceil(3 * sig[cand].max(1).values + 3 * float(sig_psf.max())).long() + 1, 4, 14)
        hcls = torch.tensor([min([h for h in (4, 6, 8, 10, 14) if h >= int(x)]) for x in hcls], device=dev)
        a_b = (co.c @ Phi.T).clamp(min=0)  # J nb
        dof = 5 + nb
        thr = chi2_isf(fpr / cand.numel(), dof)
        log["chi2_threshold"], log["dof"] = thr, dof
        out = []
        gains = []
        for h in sorted(set(hcls.tolist())):
            jj = cand[hcls == h]
            S = 2 * h + 1
            cj = torch.stack([co.mu[jj, 0].round().clamp(0, co.W - 1), co.mu[jj, 1].round().clamp(0, co.H - 1)], 1).long()  # in [0, W-1]: indices below stay inside the padded image
            ar = torch.arange(-h, h + 1, device=dev)
            Rp, Op = F.pad(R, (h,) * 4), F.pad(om, (h,) * 4)
            ys = (cj[:, 1, None] + h + ar[None])[:, :, None].expand(-1, -1, S)
            xs = (cj[:, 0, None] + h + ar[None])[:, None, :].expand(-1, S, -1)
            assert int(ys.min()) >= 0 and int(ys.max()) < Rp.shape[1] and int(xs.min()) >= 0 and int(xs.max()) < Rp.shape[2], "stamp outside the padded image"
            Rs = Rp[:, ys, xs].permute(1, 0, 2, 3)
            Os = Op[:, ys, xs].permute(1, 0, 2, 3)
            mu_l = co.mu[jj] - cj.float() + h
            s_j, t_j = sig[jj], co.theta[jj]
            own = a_b[jj][:, :, None, None] * _stamp_images(mu_l, s_j, t_j, sig_psf, S)
            T = Rs + own
            with torch.enable_grad():
                c1, mu1, sg1, th1, am1 = _fit_stamps(T, Os, mu_l[:, None], s_j[:, None], t_j[:, None], sig_psf, floor, steps)
                best = None
                for axis in (0, 1):
                    u = torch.stack([torch.cos(th1[:, 0] + axis * math.pi / 2), torch.sin(th1[:, 0] + axis * math.pi / 2)], 1)
                    ext = sg1[:, 0, axis][:, None]
                    mu2 = torch.stack([mu1[:, 0] - 0.5 * ext * u, mu1[:, 0] + 0.5 * ext * u], 1)
                    s2 = (sg1 / 1.6).expand(-1, 2, -1).clone()
                    c2, mu2f, sg2f, th2f, am2 = _fit_stamps(T, Os, mu2, s2, th1.expand(-1, 2).clone(), sig_psf, floor, steps)
                    if best is None:
                        best = [c2, mu2f, sg2f, th2f, am2]
                    else:
                        better = c2 < best[0]
                        best = [torch.where(better, c2, best[0]), torch.where(better[:, None, None], mu2f, best[1]),
                                torch.where(better[:, None, None], sg2f, best[2]), torch.where(better[:, None], th2f, best[3]),
                                torch.where(better[:, None, None], am2, best[4])]
            c2, mu2f, sg2f, th2f, am2 = best
            dchi = (c1 - c2) / infl
            gains += dchi.tolist()
            for q in range(jj.numel()):
                if float(dchi[q]) > thr and bool((am2[q].sum(1) > 0).all()):
                    share = (am2[q] / am2[q].sum(0, keepdim=True).clamp(min=1e-30)).mean(1)  # (2,) fraction of the flux
                    out.append({"j": int(jj[q]), "mu": mu2f[q] + (cj[q].float() - h), "sigma": sg2f[q], "theta": th2f[q], "share": share,
                                "dchi2": float(dchi[q])})
        log["split"] = len(out)
        log["dchi2_median"] = float(np.median(gains)) if gains else None
        return out, log


def splat2_snapshot(model):
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def splat2_restore(model, sd):
    """Load a state dict whose splat count differs from the current one (resizes mu, rho, theta, c first)."""
    co = model.sky.coeffs
    dev = co.smooth.device
    pre = "sky.coeffs."
    co.mu = nn.Parameter(sd[pre + "mu"].to(dev).clone())
    co.rho = nn.Parameter(sd[pre + "rho"].to(dev).clone())
    co.theta = nn.Parameter(sd[pre + "theta"].to(dev).clone())
    co.c = sd[pre + "c"].to(dev).clone()
    co.invalidate()
    model.load_state_dict({k: v.to(dev) for k, v in sd.items()})


def splat2_solve(model, batches, ntot, band_index, phi_ref, a, tag="", Phi=None):
    """One exact amplitude solve (see Splat2Problem); returns its info dict (also logged)."""
    t = time.perf_counter()
    co = model.sky.coeffs
    co.invalidate()
    prob = Splat2Problem(model, batches, ntot, phi_ref, band_index, LOSS_KIND, HUBER_DELTA or 3.0, Phi=Phi)
    info = prob.solve(iters=a.splat_solve_iters, tol=1e-5)
    info["seconds"] = round(time.perf_counter() - t, 2)
    info["n_splats"] = co.n
    del prob
    return info


def splat2_densify(model, batches, band_index, Phi, q1, a, quick_log):
    """prune -> (re-solve done by the caller) ; returns (stats, need_resolve). See the report in patch_v32.py."""
    co = model.sky.coeffs
    stats = {"pruned": 0, "split": 0, "added": 0}
    stats["sanitized"] = co.sanitize()
    if stats["sanitized"]:
        tick(f"WARNING: dropped {stats['sanitized']} splats with non-finite parameters or a centre outside the image")
    sig_psf = psf_core_sigma(model)
    num, om = residual_coadd(model, batches, band_index, LOSS_KIND, HUBER_DELTA or 3.0)
    if co.n:
        snr, _ = amplitude_snr(co, om, sig_psf, Phi, q1)
        keep = snr >= a.splat_prune_snr
        stats["pruned"] = int((~keep).sum())
        if stats["pruned"]:
            co.keep(keep)
            stats["need_solve"] = True
    return stats, (num, om, sig_psf)


def splat2_add_split(model, num, om, sig_psf, Phi, a):
    co = model.sky.coeffs
    dev = co.smooth.device
    H, W = co.H, co.W
    stats = {"split": 0, "added": 0}
    R = torch.where(om > 0, num / om.clamp(min=1e-30), torch.zeros_like(num))
    Z, valid, scale = significance_maps(num, om, sig_psf)
    Zp, Zn = F.relu(Z), F.relu(-Z)
    Tpos, Tneg = (Zp ** 2).sum(0), (Zn ** 2).sum(0)
    ok = F.avg_pool2d(valid[None, None].float(), 3, stride=1, padding=1)[0, 0] > 0.999  # one pixel in from uncovered pixels
    Tpos, Tneg = Tpos * ok, Tneg * ok
    rpk = int(max(1, math.ceil(float(sig_psf.mean())) + 1))
    pk_n = local_maxima(Tneg, rpk)
    tau, ninfo = null_threshold(Tneg[pk_n], a.splat_fpr)
    stats.update(ninfo)
    stats["tau"] = tau
    if not math.isfinite(tau):
        return stats
    pk_p = local_maxima(Tpos, rpk) & (Tpos > tau)
    yp, xp = torch.nonzero(pk_p, as_tuple=True)
    pos_xy = torch.stack([xp, yp], 1).float()
    pk_n2 = local_maxima(Tneg, rpk) & (Tneg > tau)
    yn, xn = torch.nonzero(pk_n2, as_tuple=True)
    all_xy = torch.cat([pos_xy, torch.stack([xn, yn], 1).float()], 0)
    infl = float((scale ** 2).mean())
    # --- splits
    try:
        sp, slog = split_tests(co, R, om, all_xy, tau, sig_psf, Phi, infl, a.splat_fpr, co.floor, steps=a.splat_split_steps)
        if dev.type == "cuda":
            torch.cuda.synchronize()  # surface an asynchronous device-side error here, not in a later call
    except Exception as exc:  # noqa: BLE001
        sp, slog = [], {"candidates": 0, "split": 0, "error": repr(exc)[:300]}
        tick(f"WARNING: split test failed ({repr(exc)[:300]}); no splits this round")
    stats["split_test"] = slog
    if sp:
        keep = torch.ones(co.n, dtype=torch.bool, device=dev)
        new_mu, new_sg, new_th, new_c = [], [], [], []
        claimed = []
        for s in sp:
            j = s["j"]
            keep[j] = False
            for q in range(2):
                new_mu.append(s["mu"][q]); new_sg.append(s["sigma"][q]); new_th.append(s["theta"][q])
                new_c.append(co.c[j] * s["share"][q])
            claimed.append(co.mu[j])
        mu = torch.cat([co.mu.detach()[keep], torch.stack(new_mu)])
        sg = torch.cat([co.sigma().detach()[keep], torch.stack(new_sg)])
        th = torch.cat([co.theta.detach()[keep], torch.stack(new_th)])
        cc = torch.cat([co.c[keep], torch.stack(new_c)])
        co.set_splats(mu, sg, th, cc)
        stats["split"] = len(sp)
        claimed = torch.stack(claimed)
        if pos_xy.shape[0]:  # a split already explains the positive residual around it
            rad = 2 * float(sig_psf.max()) + 3
            pos_xy = pos_xy[(torch.cdist(pos_xy, claimed) > rad).all(1)]
    # --- additions at the remaining positive peaks, brightest first, within the budget
    room = max(0, a.splat_max - co.n)
    if pos_xy.shape[0] and room:
        vals = Tpos[pos_xy[:, 1].long(), pos_xy[:, 0].long()]
        order = torch.argsort(vals, descending=True)[:room]
        pk = pos_xy[order].long()
        mu, sg, th = init_shapes(Zp.sum(0), pk, float(sig_psf.mean()), co.floor, valid)
        co.set_splats(torch.cat([co.mu.detach(), mu]), torch.cat([co.sigma().detach(), sg]), torch.cat([co.theta.detach(), th]),
                      torch.cat([co.c, torch.zeros(mu.shape[0], co.K, device=dev)]))
        stats["added"] = int(mu.shape[0])
        stats["n_positive_peaks_above_tau"] = int(pos_xy.shape[0])
    stats["n_negative_peaks_above_tau"] = int(yn.numel())
    return stats


def fit_splat2(name, model, exps, steps_round, lrs_e, sig_b, train_idx, band_index, cb, a, Phi, dev):
    """--sky-coeffs splat2 outer loop. Round r: [prune by amplitude S/N] -> [re-solve] -> split / add from the training-residual
    significance -> exact solve -> Adam on the splat shapes + instrument (fit(), amplitudes frozen, exact re-solve every
    --splat-solve-every steps through its step_hook) -> exact solve -> validation score (cb). Stops when the score has not improved
    for 2 rounds (best state restored), when a round neither adds, splits nor prunes, at --splat-max-rounds, or at the time
    budget. Without validation exposures (cb returns None) it runs until nothing changes or the round limit, and keeps the last."""
    co = model.sky.coeffs
    basis = model.sky.basis
    lam_ref = float(np.median([REG.lam_um[b] for b in REG.bands]))
    with torch.no_grad():
        phi_ref = basis(torch.tensor([lam_ref], device=dev))[0]
    q1 = _householder(phi_ref.float())[:, 0]
    batches = band_batches(exps, train_idx, 4)
    ntot = sum(exps[i]["d"].numel() for i in train_idx)
    t_start = time.perf_counter()
    budget_s = a.budget_explicit * 60
    max_rounds = min(a.splat_max_rounds, 3) if a.quick else a.splat_max_rounds
    rounds, best, bad, it_tot = [], None, 0, 0
    LOG[f"{name}_splat_rounds"] = rounds
    lrs = dict(lrs_e, **({"sky": a.sky_lr} if a.sky_lr > 0 else {}), **({"psf": a.psf_lr} if a.psf_lr > 0 else {}))

    def solve(tag):
        info = splat2_solve(model, batches, ntot, band_index, phi_ref, a, tag, Phi)
        tick(f"{name} {tag}: exact amplitude solve, {info['n_splats']} splats, {info['matvecs']} matvecs, projected-gradient residual "
             f"{info['proj_grad_rel']:.2e} (rel), {info['n_active']} amplitudes at 0, {info['seconds']} s")
        return info

    def hook(m_, it, bt):
        if a.splat_solve_every > 0 and co.n > 0 and (it + 1) % a.splat_solve_every == 0:
            splat2_solve(m_, bt, ntot, band_index, phi_ref, a, Phi=Phi)
    solve("initial (smooth component only)")
    for r in range(max_rounds):
        rec = {"round": r}
        st, (num, om, sig_psf) = splat2_densify(model, batches, band_index, Phi, q1, a, None)
        rec["pruned"] = st["pruned"]
        rec["sanitized"] = st["sanitized"]
        if st["sanitized"]:
            solve(f"round {r} after sanitising")
            num, om = residual_coadd(model, batches, band_index, LOSS_KIND, HUBER_DELTA or 3.0)
        if st.get("need_solve"):
            solve(f"round {r} after prune")
            num, om = residual_coadd(model, batches, band_index, LOSS_KIND, HUBER_DELTA or 3.0)
        st2 = splat2_add_split(model, num, om, sig_psf, Phi, a)
        rec.update({k: st2.get(k, 0) for k in ("split", "added")})
        rec["added_before_prune"] = rec["added"]
        rec["tau"], rec["n_null_peaks"] = st2.get("tau"), st2.get("n_null_peaks")
        rec["split_test"] = st2.get("split_test")
        rec["solve"] = solve(f"round {r} after densify")
        for _ in range(3):  # joint exact solve first, then drop what the joint solution does not need (e.g. peaks that were only the
            if co.n == 0:  # ringing of a brighter unmodelled source), so Adam never sees them
                break
            _, om_ = residual_coadd(model, batches, band_index, LOSS_KIND, HUBER_DELTA or 3.0)
            snr_, _ = amplitude_snr(co, om_, sig_psf, Phi, q1)
            drop = snr_ < a.splat_prune_snr
            if not bool(drop.any()):
                break
            co.keep(~drop)
            rec["pruned"] += int(drop.sum())
            rec["solve"] = solve(f"round {r} after post-densify prune")
        rec["total"] = co.n
        changed = rec["pruned"] + rec["split"] + rec["added"]
        if co.n:
            fit(f"{name}_r{r}", model, exps, steps_round, lrs, 0.0, sig_b, train_idx, log_every=max(steps_round // 2, 1),
                budget_s=max(a.splat_round_min * 60, 1.0), nchunks=4, callback=None, cb_every_s=None, patience=0, batches=batches,
                no_ckpt=True, step_hook=hook)
            it_tot += LOG.get(f"{name}_r{r}_steps_done", 0)
            rec["solve_final"] = solve(f"round {r} final")
        score = cb(model, it_tot, time.perf_counter() - t_start) if cb is not None else None
        rec["val_score"] = score
        tick(f"{name} splat round {r}: pruned {rec['pruned']}, split {rec['split']}, added {rec['added']}, total {rec['total']}, "
             f"threshold {rec['tau']}, validation score {score}")
        rounds.append(rec)
        if score is not None:
            if best is None or score < best[0]:
                best, bad = (score, r, splat2_snapshot(model)), 0
            else:
                bad += 1
                if bad >= 2:
                    tick(f"{name}: validation score not improved for 2 rounds, stopping densification")
                    break
        if changed == 0:
            tick(f"{name}: round {r} changed nothing, stopping")
            break
        if time.perf_counter() - t_start > budget_s:
            tick(f"{name}: time budget reached after round {r}")
            break
    if best is not None:
        splat2_restore(model, best[2])
        LOG[f"{name}_best_score"], LOG[f"{name}_best_round"] = best[0], best[1]
        tick(f"{name}: restored the best validation state (round {best[1]}, score {best[0]:.4f}, {co.n} splats)")
    LOG[f"{name}_n_splats"] = co.n
    if SAVE_DIR is not None:
        np.savez(SAVE_DIR / f"{name}_splats.npz", mu=co.mu.detach().cpu().numpy(), sigma=co.sigma().detach().cpu().numpy(),
                 theta=co.theta.detach().cpu().numpy(), c=co.c.cpu().numpy(), smooth=co.smooth.cpu().numpy())
        torch.save(model.state_dict(), SAVE_DIR / f"{name}_state.pt")
    del batches
    return model
'''

rep("""def splat_seeds(init, sig_b, A0, nmax, snr, sig_px):""", BLOCK.lstrip() + """

def splat_seeds(init, sig_b, A0, nmax, snr, sig_px):""")

# --- SpectralSky: optional coefficient override, param groups
rep("""    def nodes(self):
        A = self.coeffs()  # K H W
""", """    def nodes(self, A=None):
        A = self.coeffs() if A is None else A  # K H W (v32: A given = the solve's linear map)
""")
rep("""        if isinstance(self.coeffs, HybridCoeffs):
            sp = self.coeffs.splats""", """        if isinstance(self.coeffs, Splat2Coeffs):  # v32: shapes only; the amplitudes are solved exactly
            co = self.coeffs
            return [{"params": [co.mu], "lr": 1e-2}, {"params": [co.rho, co.theta], "lr": 5e-3}] if co.n else []
        if isinstance(self.coeffs, HybridCoeffs):
            sp = self.coeffs.splats""")
# --- SkyModel.prepare
rep("""    def prepare(self):
        if hasattr(self.sky, "nodes"):
            Sn, S = self.sky.nodes()  # nb Q H W ; nb H W""", """    def prepare(self, A=None):
        if hasattr(self.sky, "nodes"):
            Sn, S = self.sky.nodes() if A is None else self.sky.nodes(A)  # nb Q H W ; nb H W""")
# --- fit(): reuse batches, no checkpoint file, step hook
rep("""        callback=None, cb_every_s=None, ckpt_every_s=1800.0, patience=0):
    if getattr(args_global, "optimizer", "adam") == "lbfgs":""", """        callback=None, cb_every_s=None, ckpt_every_s=1800.0, patience=0, batches=None, no_ckpt=False, step_hook=None):
    if getattr(args_global, "optimizer", "adam") == "lbfgs":""")
rep("""    batches = band_batches(exps, train_idx, nchunks)
    _LCACHE.clear()
    hist = []""", """    batches = band_batches(exps, train_idx, nchunks) if batches is None else batches  # v32: splat2 passes its own
    _LCACHE.clear()
    hist = []""")
rep("""    ck = SAVE_DIR / f"{name}_ckpt.pt" if SAVE_DIR is not None else None""", """    ck = SAVE_DIR / f"{name}_ckpt.pt" if (SAVE_DIR is not None and not no_ckpt) else None""")
rep("""        opt.step()
        el = time.perf_counter() - t_start
        set_lr(""", """        opt.step()
        if step_hook is not None:  # v32: exact amplitude solve between Adam steps
            step_hook(model, it, batches)
        el = time.perf_counter() - t_start
        set_lr(""")
# --- state loading
rep("""    res = m.load_state_dict(torch.load(path, map_location=dev), strict=False)""", """    sd_ = torch.load(path, map_location=dev)
    if isinstance(getattr(getattr(m, "sky", None), "coeffs", None), Splat2Coeffs):  # v32: the splat count is part of the state
        splat2_restore(m, {k: v.cpu() for k, v in sd_.items()})
    res = m.load_state_dict(sd_, strict=False)""")
# --- arguments
rep("""    ap.add_argument("--sky-coeffs", choices=("grid", "hybrid", "convdecoder"), default="grid", help="explicit spectral sky coefficients")""",
    """    ap.add_argument("--sky-coeffs", choices=("grid", "hybrid", "convdecoder", "splat2"), default="grid", help="explicit spectral sky coefficients (splat2 = v32 catalogue-free elaborated Gaussian splats, needs --spectral rbf)")
    ap.add_argument("--splat-smooth-px", type=int, default=4, help="v32: coarse grid spacing (target px) of the smooth component")
    ap.add_argument("--splat-sigma-floor", type=float, default=0.5, help="v32: minimum splat sigma (target px)")
    ap.add_argument("--splat-solve-every", type=int, default=50, help="v32: exact amplitude solve every N Adam steps (0 = only at round boundaries)")
    ap.add_argument("--splat-solve-iters", type=int, default=40, help="v32: matvec cap of one projected-CG solve")
    ap.add_argument("--splat-round-min", type=float, default=5.0, help="v32: minutes of Adam per densification round")
    ap.add_argument("--splat-max-rounds", type=int, default=12, help="v32: maximum densification rounds")
    ap.add_argument("--splat-max", type=int, default=20000, help="v32: maximum number of splats")
    ap.add_argument("--splat-fpr", type=float, default=0.1, help="v32: expected false additions per round allowed by the null threshold (also the false-split budget)")
    ap.add_argument("--splat-prune-snr", type=float, default=3.0, help="v32: prune splats whose broadband amplitude S/N is below this")
    ap.add_argument("--splat-split-steps", type=int, default=100, help="v32: Adam steps of each local 1- and 2-Gaussian fit of the split test")""")
# --- construction
rep("""            elif kind == "convdecoder":
                s0 = float(sig_b.mean())""", """            elif kind == "splat2":
                if a.uq or a.zip or a.optimizer == "lbfgs":
                    raise SystemExit("--sky-coeffs splat2 does not support --uq, --zip or --optimizer lbfgs")
                coeffs = Splat2Coeffs(a.spec_k, A0.shape[-2], A0.shape[-1], a.splat_smooth_px, a.splat_sigma_floor, smooth_init=A0, nsub=4).to(dev)
                LOG[f"{name}_splat2"] = {"smooth_px": a.splat_smooth_px, "sigma_floor": a.splat_sigma_floor, "tv_in_objective": 0.0}
                tick(f"{name}: splat2 sky, smooth component {coeffs.hs}x{coeffs.ws} cells from the coarse low-pass of the coadd initialisation, no splats yet")
            elif kind == "convdecoder":
                s0 = float(sig_b.mean())""")
rep("""        else:
            fit(name, m, exps, steps_e, dict(dict(lrs_e, psf=5e-4)""", """        elif spec and getattr(a, "sky_coeffs", "grid") == "splat2":
            fit_splat2(name, m, exps, steps_e, dict(lrs_e, psf=5e-4) if a.explicit_psf == "chrom" else lrs_e, sig_b, train_idx, band_index,
                       make_cb(name), a, band_phi(basis, lam_nodes, w_nodes), dev)
        else:
            fit(name, m, exps, steps_e, dict(dict(lrs_e, psf=5e-4)""")
rep("""        return cb if getattr(a, "eval_every_min", 0) > 0 else None""", """        return cb if (getattr(a, "eval_every_min", 0) > 0 or getattr(a, "sky_coeffs", "grid") == "splat2") else None""")
rep("""    spec = getattr(a, "spectral", "none") != "none"
    if spec:
        lam_nodes, w_nodes, curves""", """    spec = getattr(a, "spectral", "none") != "none"
    if getattr(a, "sky_coeffs", "grid") == "splat2" and not spec:
        raise SystemExit("--sky-coeffs splat2 needs --spectral rbf")
    if spec:
        lam_nodes, w_nodes, curves""")
open(dst, "w").write(s)
print("wrote", dst)
