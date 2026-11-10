"""v32 -> v33 (5 Nov 2026): roll-aware PSF, phases A0 and A1 of forward-operator-spec.md.
Problem: the learned PSF lived on the sky target grid and was conditioned on band/channel/detector only, but the HST data come from
several telescope rolls (the diffraction pattern rotates with the telescope, the pixel box with the detector).

--psf-frame detector (A0; default sky = v32 behaviour, bit-identical):
  * Per exposure the local linear map J_e from detector pixel offsets (col, row) to target-grid offsets is fitted by least squares
    over the exposure's pixels from the LOADED starting coordinates (tx, ty); the learned Geometry corrections are ignored.
    J_e = R(theta_e) M F_p with F_p = diag(1, p), p = sign det J (parity), M ~ diag(sx, sy) (detector pixel size in target px, the
    residual off-diagonal part is the shear); theta_e is the rotation of the polar decomposition of J_e F_p.
  * PSF groups = (band, theta_e rounded to --psf-group-deg (0.5), parity). The learned kernel K(xi) is a function of the offset xi in
    the DETECTOR-aligned frame (target-pixel units); for a group it is rendered on the sky grid at xi = F_p R(theta)^T delta:
    ChromaticPSF / ImplicitVarPSF (coordinate networks) are evaluated at the rotated coordinates (exact, no interpolation; the
    lambda/D-scaled coordinates are rotated the same way); ExplicitVarPSF (pixel grids) is rotated by bilinear resampling
    (APPROXIMATION: smoothing ~1/6 px^2 variance for a generic angle, zero padding at the window edge, core renormalised to unit sum,
    the other kernels re-centred to zero sum). The mixing-weight field is unchanged.
  * SkyModel.prepare convolves once per PSF group (both for the band image and for the spectral path with per-quadrature-node
    kernels) instead of once per band: CB has shape (G, nk, H, W), predict() reads CB[group of the batch]; band_batches splits a
    band's batches per group. prepare() is called once per training step (the per-step cache); held-out scoring convolves only the
    group of the scored exposure (cached within one heldout_chi2 call), creating a group on the fly for an unseen theta.
    Memory and time scale with the number of groups G relative to the number of bands.
--pixel-box (A1, needs --psf-frame detector): the learned kernel is the optical PSF only; the detector pixel's box response is applied
  in the detector frame: kernel_eff(delta) = mean_u K(delta - J_e u), u on an S x S midpoint grid over one detector pixel,
  S = ceil(2 max(sx, sy)); in detector coordinates this is xi - F M F u, i.e. a box of sx x sy target px. Midpoint rule: exact for
  polynomials of degree 1 per axis; error O((w/S)^2 K''/24) with w the pixel width. The raw (pre-normalisation) core and residual
  kernels are averaged, then the core is normalised as before. The Gaussian warm start is narrowed by the box variance.
Not supported with the detector frame: --zip, --uq, pupil PSFs. Per-channel --psf-n is not implemented (one window for all bands)."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


BLOCK = r'''# ----------------------------------------------------------------------------- v33: roll-aware PSF (--psf-frame detector, --pixel-box)
PSF_FRAME = "sky"
PIXEL_BOX = False
PSFG = None  # PSFGroups of the current run (detector frame only)


def configure_psf_frame(a):
    global PSF_FRAME, PIXEL_BOX
    PSF_FRAME = getattr(a, "psf_frame", "sky")
    PIXEL_BOX = bool(getattr(a, "pixel_box", False))
    if PIXEL_BOX and PSF_FRAME != "detector":
        raise SystemExit("--pixel-box needs --psf-frame detector")
    if PSF_FRAME == "detector":
        if getattr(a, "zip", False) or getattr(a, "uq", False):
            raise SystemExit("--psf-frame detector does not support --zip or --uq")
        if getattr(a, "explicit_psf", "") == "pupil" or getattr(a, "psf_model", "") == "pupil":
            raise SystemExit("--psf-frame detector does not support the pupil PSF")


def estimate_jacobian(e):
    """Local linear map J from detector pixel offsets (col, row) to target-grid offsets (tx, ty), least squares over the exposure's
    loaded pixels (starting coordinates, learned Geometry ignored). J = R(theta) M F_p, F_p = diag(1, p), p = sign det J.
    Returns theta_deg in (-180, 180], parity p, scale = sqrt|det J| (target px per detector px), sx, sy = diagonal of M, shear = the
    off-diagonal part of M relative to its trace, M (2x2), J (2x2) and the rms residual of the linear fit [target px]."""
    row, col = e["row"].detach().double().cpu(), e["col"].detach().double().cpu()
    Y = torch.stack([e["tx"].detach().double().cpu(), e["ty"].detach().double().cpu()], 1)
    X = torch.stack([torch.ones_like(col), col - col.mean(), row - row.mean()], 1)
    coef = torch.linalg.solve(X.T @ X, X.T @ Y)  # 3 x 2
    rms = float((Y - X @ coef).pow(2).sum(1).mean().sqrt())
    J = coef[1:].T.numpy().copy()  # J[i, j] = d(target i) / d(detector j)
    det = float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
    p = 1 if det > 0 else -1
    Jf = J * np.array([1.0, float(p)])[None, :]
    th = math.atan2(Jf[1, 0] - Jf[0, 1], Jf[0, 0] + Jf[1, 1])
    c, s_ = math.cos(th), math.sin(th)
    M = np.array([[c, s_], [-s_, c]]) @ Jf
    return {"theta_deg": math.degrees(th), "parity": p, "scale": math.sqrt(abs(det)), "sx": float(M[0, 0]), "sy": float(M[1, 1]),
            "shear": float((M[0, 1] + M[1, 0]) / (M[0, 0] + M[1, 1])), "M": M.tolist(), "J": J.tolist(), "fit_rms_px": rms, "n": int(row.numel())}


class PSFGroups:
    """PSF groups (band, theta rounded to `step` degrees, parity). Groups seen in training come first (ids 0..n_train-1); a group created
    afterwards (held-out exposure with an unseen theta) gets a later id and is not part of the training convolutions."""

    chunk = 8  # groups convolved per FFT call (bounds the temporaries)

    def __init__(self, step_deg=0.5, box=False):
        self.step, self.box = float(step_deg), bool(box)
        self.groups, self.index, self.n_train = [], {}, None

    def theta_key(self, th):
        k = math.floor(th / self.step + 0.5) * self.step
        if k <= -180.0:
            k += 360.0
        return k + 0.0

    def key(self, e):
        j = e["jac"]
        return (REG.index[e["band"]], self.theta_key(j["theta_deg"]), int(j["parity"]))

    def add(self, e, train=True):
        k = self.key(e)
        g = self.index.get(k)
        j = e["jac"]
        if g is None:
            g = len(self.groups)
            self.index[k] = g
            S = int(math.ceil(2.0 * max(j["sx"], j["sy"]))) if self.box else 1
            self.groups.append({"id": g, "b": k[0], "band": e["band"], "theta_deg": k[1], "parity": k[2], "M": j["M"], "sx": j["sx"], "sy": j["sy"],
                                "S": max(S, 1), "n_exp": 0, "n_train": 0, "n_pix": 0, "det": int(e.get("det", 0)), "names": []})
        info = self.groups[g]
        info["n_exp"] += 1
        info["n_train"] += int(train)
        info["n_pix"] += int(e["d"].numel())
        info["names"].append(e["name"])
        return g

    def freeze_train(self):
        self.n_train = len(self.groups)

    def group_id(self, e):
        """Group of an exposure, created on the fly for an unseen (band, theta, parity)."""
        g = self.index.get(self.key(e))
        return self.add(e, train=False) if g is None else g

    def own_groups(self):
        """theta = 0, parity +1 group per band (display cubes *_ownpsf), pixel size of the band's first group."""
        out = []
        for b in range(len(REG.bands)):
            gb = next((g for g in self.groups if g["b"] == b), None)
            sx, sy = (gb["sx"], gb["sy"]) if gb is not None else (1.0, 1.0)
            S = int(math.ceil(2.0 * max(sx, sy))) if self.box else 1
            out.append({"id": -1, "b": b, "band": REG.bands[b], "theta_deg": 0.0, "parity": 1, "M": [[sx, 0.0], [0.0, sy]], "sx": sx, "sy": sy, "S": max(S, 1)})
        return out

    def select(self, groups=None):
        if groups is None:
            return self.groups[:self.n_train]
        if isinstance(groups, str):
            assert groups == "own", groups
            return self.own_groups()
        return [g if isinstance(g, dict) else self.groups[g] for g in groups]

    def table(self):
        return [dict({k: v for k, v in g.items() if k not in ("names", "M")}, first_exposure=(g["names"][0] if g["names"] else None)) for g in self.groups]


def box_shifts(g, dev):
    """(S*S, 2) detector-frame shifts F M F u of the midpoint sub-positions u in [-1/2, 1/2]^2 of one detector pixel [target px]."""
    S = g["S"]
    t = (torch.arange(S, dtype=torch.float64) + 0.5) / S - 0.5
    uy, ux = torch.meshgrid(t, t, indexing="ij")
    u = torch.stack([ux.reshape(-1), uy.reshape(-1)], 1)
    Fm = torch.diag(torch.tensor([1.0, float(g["parity"])], dtype=torch.float64))
    Mm = torch.tensor(g["M"], dtype=torch.float64)
    return (u @ (Fm @ Mm @ Fm).T).float().to(dev)


def _cs(theta_deg):
    """cos, sin of an angle in degrees, exact for multiples of 90 degrees."""
    t = math.radians(theta_deg)
    c, s_ = math.cos(t), math.sin(t)
    return (0.0 if abs(c) < 1e-12 else c), (0.0 if abs(s_) < 1e-12 else s_)


def group_rot(groups, dev, box):
    """Rotation spec of the PSF classes: one row per group (cs = cos/sin of theta, parity, shifts = box sub-positions)."""
    return {"band": torch.tensor([g["b"] for g in groups], dtype=torch.long, device=dev),
            "cs": torch.tensor([_cs(g["theta_deg"]) for g in groups], dtype=torch.float32, device=dev),
            "parity": torch.tensor([float(g["parity"]) for g in groups], dtype=torch.float32, device=dev),
            "shifts": [box_shifts(g, dev) if box else torch.zeros(1, 2, device=dev) for g in groups]}


def rot_repeat(rot, q):
    return {"band": rot["band"].repeat_interleave(q), "cs": rot["cs"].repeat_interleave(q, 0), "parity": rot["parity"].repeat_interleave(q),
            "shifts": [sh for sh in rot["shifts"] for _ in range(q)]}


def rot_xy(xy, cs, par, sh):
    """Target-grid offsets xy (P, 2) -> detector-frame coordinates xi = F R(theta)^T delta, minus the box shifts: (T*P, 2)."""
    c, s_ = cs[0], cs[1]
    x, y = xy[:, 0], xy[:, 1]
    xi = torch.stack([c * x + s_ * y, par * (c * y - s_ * x)], -1)
    return (xi[None] - sh[:, None, :]).reshape(-1, 2)


def run_ckpt(fn, *args):
    """Evaluate a kernel row; with a box (many sub-positions) and gradients on, recompute in the backward pass to bound memory."""
    if torch.is_grad_enabled() and args[-1].shape[0] > 1:
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def resample_rot(B, rot, n):
    """Grid kernels B (nb, nk, n, n) in the detector frame -> (M, nk, n, n) rotated onto the sky grid by bilinear resampling (zero
    padding), averaged over the box sub-positions. Core (index 0) renormalised to unit sum, the others re-centred to zero sum."""
    dev = B.device
    r = torch.arange(n, device=dev, dtype=torch.float32) - n // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    xy = torch.stack([xx, yy], -1).reshape(-1, 2)
    h = (n - 1) / 2.0
    out = []
    for m in range(rot["band"].numel()):
        pts = rot_xy(xy, rot["cs"][m], rot["parity"][m], rot["shifts"][m])
        T = pts.shape[0] // (n * n)
        g = ((pts + (n // 2) - h) / h).reshape(T, n, n, 2)
        img = B[rot["band"][m]][None].expand(T, -1, -1, -1)
        out.append(F.grid_sample(img, g, mode="bilinear", padding_mode="zeros", align_corners=True).mean(0))
    K = torch.stack(out)
    core = K[:, 0]
    core = core / core.sum((-2, -1), keepdim=True)
    rest = K[:, 1:] - K[:, 1:].mean((-2, -1), keepdim=True)
    return torch.cat([core[:, None], rest], 1)


def psf_group_kernels(psf, groups, lam_nodes=None):
    """Basis kernels of PSF groups on the sky grid: (G, nk, n, n), or (G*Q, nk, n, n) when lam_nodes (nb, Q) is given (kernels at the
    band's quadrature wavelengths for ChromaticPSF, the band kernel repeated otherwise)."""
    dev = psf.det_emb.weight.device if hasattr(psf, "det_emb") else psf.res.device
    sel = torch.tensor([g["b"] for g in groups], dtype=torch.long, device=dev)
    rot = group_rot(groups, dev, PSFG.box)
    q = 1 if lam_nodes is None else lam_nodes.shape[1]
    if hasattr(psf, "_basis"):  # ChromaticPSF
        if lam_nodes is not None and hasattr(psf, "basis_lam"):
            return psf.basis_lam(lam_nodes[sel].flatten(), psf.chan_idx[sel].repeat_interleave(q), rot=rot_repeat(rot, q))
        return psf._basis(psf.pix_per_lod[sel], psf.loglam[sel], psf.chan[sel], rot=rot)
    B = psf.basis(rot=rot)
    return B if q == 1 else B.repeat_interleave(q, 0)


def group_cb(model, e, cache):
    """Convolved basis images (nk, H, W) of the PSF group of exposure e (a group is created for an unseen theta); cached in `cache`."""
    gid = PSFG.group_id(e)
    if gid not in cache:
        cache[gid] = model.prepare(groups=[gid])[1][0].detach()
    return cache[gid]


def box_adjusted_sigma(sig0):
    """Starting Gaussian width of the OPTICAL PSF: remove the detector pixel's box variance w^2/12 per axis (target px^2)."""
    out = sig0.clone()
    for b in range(len(REG.bands)):
        gs = [g for g in PSFG.groups if g["b"] == b]
        if not gs:
            continue
        v = float(np.mean([(g["sx"] ** 2 + g["sy"] ** 2) / 24.0 for g in gs]))
        s0 = float(sig0[b])
        if s0 >= 0.5:
            out[b] = math.sqrt(max(s0 * s0 - v, 0.25))
    return out


def setup_psf_frame(exps, train_idx, a, dev):
    """J_e of every exposure, PSF groups of the training exposures (then the held-out / validation exposures are attached to existing
    groups or new ones), log and print."""
    global PSFG
    PSFG = PSFGroups(getattr(a, "psf_group_deg", 0.5), PIXEL_BOX)
    for e in exps:
        e["jac"] = estimate_jacobian(e)
    tr = set(train_idx)
    for i in train_idx:
        exps[i]["pgrp"] = PSFG.add(exps[i], train=True)
    PSFG.freeze_train()
    for i, e in enumerate(exps):
        if i not in tr:
            e["pgrp"] = PSFG.group_id(e)
    nk = a.nk
    mb = PSFG.n_train * nk * a.size * a.size * 4 / 1e6
    per_band = {}
    for g in PSFG.groups[:PSFG.n_train]:
        per_band.setdefault(g["band"], []).append(g)
    box_txt = " + pixel box" if PIXEL_BOX else ""
    tick(f"PSF frame: detector{box_txt}; {PSFG.n_train} PSF groups in training for {len(REG.bands)} bands "
         f"(+{len(PSFG.groups) - PSFG.n_train} held-out-only), group rounding {PSFG.step} deg; convolved basis images ~{mb:.0f} MB per copy "
         f"({PSFG.n_train} x {nk} x {a.size}^2 float32, vs {len(REG.bands)} x {nk} x {a.size}^2 in the sky frame)")
    for b in REG.bands:
        for g in per_band.get(b, []):
            tick(f"  group {g['id']:3d} {b:14s} theta {g['theta_deg']:8.2f} deg parity {g['parity']:+d} pixel {g['sx']:.3f}x{g['sy']:.3f} target px "
                 f"S={g['S']} n_exp {g['n_train']} n_pix {g['n_pix']}")
    LOG["psf_frame"] = {"frame": PSF_FRAME, "pixel_box": PIXEL_BOX, "group_deg": PSFG.step, "n_train_groups": PSFG.n_train, "groups": PSFG.table(),
                        "exposures": {e["name"]: dict({k: e["jac"][k] for k in ("theta_deg", "parity", "scale", "sx", "sy", "shear", "fit_rms_px")}, group=e["pgrp"]) for e in exps}}
    return PSFG


'''

# --------------------------------------------------------------------------- PSF classes
rep('''    def basis(self):
        core = gauss_kernel(self.log_sig.exp(), self.n, self.res.device) + self.res[:, 0]''',
    '''    def basis(self, rot=None):
        core = gauss_kernel(self.log_sig.exp(), self.n, self.res.device) + self.res[:, 0]''')
rep('''        return torch.cat([core[:, None], rest], 1)  # nb nk n n''',
    '''        B = torch.cat([core[:, None], rest], 1)  # nb nk n n
        return B if rot is None else resample_rot(B, rot, self.n)  # v33: detector-frame kernels rotated onto the sky grid (bilinear)''')

rep('''    def basis(self):
        dev = self.det_emb.weight.device
        r = (torch.arange(self.n, device=dev, dtype=torch.float32) - self.n // 2) / (self.n // 2)''',
    '''    def basis(self, rot=None):
        dev = self.det_emb.weight.device
        if rot is not None:
            return self._basis_rot(rot)
        r = (torch.arange(self.n, device=dev, dtype=torch.float32) - self.n // 2) / (self.n // 2)''')
rep('''    def weights(self, b, det, dx, dy):
        N = dx.shape[0]
        oh = torch.zeros(N, self.nb, device=dx.device)
        oh[:, b] = 1
        inp = torch.cat([fourier(torch.stack([dx, dy], -1) * 0.5, 3), oh, self.det_emb.weight[vec(det, dx)]], 1)''',
    '''    def _basis_rot(self, rot):
        """v33: the MLP kernel field evaluated at the detector-frame coordinates xi = F R(theta)^T delta (minus the box sub-positions)."""
        dev = self.det_emb.weight.device
        half = self.n // 2
        r = torch.arange(self.n, device=dev, dtype=torch.float32) - half
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        xy = torch.stack([xx, yy], -1).reshape(-1, 2)  # pixels
        P = xy.shape[0]
        out = []
        for m in range(rot["band"].numel()):
            b = int(rot["band"][m])

            def row(cs, par, sh, b=b):
                pts = rot_xy(xy, cs, par, sh) / half
                oh = torch.zeros(pts.shape[0], self.nb, device=dev)
                oh[:, b] = 1
                k = self.kern(torch.cat([fourier(pts * 0.5, 6), oh], 1)).reshape(sh.shape[0], P, self.nk)
                return torch.cat([torch.nn.functional.softplus(k[..., 0]).mean(0)[None], k[..., 1:].mean(0).T], 0)  # nk, P
            k = run_ckpt(row, rot["cs"][m], rot["parity"][m], rot["shifts"][m]).reshape(self.nk, self.n, self.n)
            core = k[0] / k[0].sum()
            rest = 0.01 * (k[1:] - k[1:].mean((-2, -1), keepdim=True))
            out.append(torch.cat([core[None], rest], 0))
        return torch.stack(out)

    def weights(self, b, det, dx, dy):
        N = dx.shape[0]
        oh = torch.zeros(N, self.nb, device=dx.device)
        oh[:, b] = 1
        inp = torch.cat([fourier(torch.stack([dx, dy], -1) * 0.5, 3), oh, self.det_emb.weight[vec(det, dx)]], 1)''')

rep('''    def basis_lam(self, lam_um, chan_idx):
        """Basis kernels at arbitrary wavelengths (um) and registry channel ids: (M, nk, n, n)."""
        lod = LOD_COEF * lam_um * (D_REF_M / self.chan_D[chan_idx.long()])
        loglam = (torch.log(lam_um) - math.log(2.0)) / 0.8
        chan = torch.nn.functional.one_hot(chan_idx.long(), self.nchan).float()
        return self._basis(self.scale_arcsec / lod, loglam, chan)

    def _basis(self, pix_per_lod, loglam, chan):
        dev = self.loglam.device''',
    '''    def basis_lam(self, lam_um, chan_idx, rot=None):
        """Basis kernels at arbitrary wavelengths (um) and registry channel ids: (M, nk, n, n)."""
        lod = LOD_COEF * lam_um * (D_REF_M / self.chan_D[chan_idx.long()])
        loglam = (torch.log(lam_um) - math.log(2.0)) / 0.8
        chan = torch.nn.functional.one_hot(chan_idx.long(), self.nchan).float()
        return self._basis(self.scale_arcsec / lod, loglam, chan, rot=rot)

    def _basis_rot(self, pix_per_lod, loglam, chan, rot):
        """v33: row m evaluated at the detector-frame coordinates xi = F R(theta_m)^T delta (minus the box sub-positions): both the
        lambda/D-scaled and the pixel coordinates are rotated; the raw core / residual kernels are box-averaged before normalising."""
        dev = self.loglam.device
        r = torch.arange(self.n, device=dev, dtype=torch.float32) - self.n // 2
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        xy = torch.stack([xx, yy], -1).reshape(-1, 2)
        P = xy.shape[0]
        M = loglam.shape[0]
        rows = []
        for b in range(M):
            def row(cs, par, sh, b=b):
                pts = rot_xy(xy, cs, par, sh)
                rho = pts * pix_per_lod[b] / 24.0
                cond = torch.cat([loglam[b].view(1, 1), chan[b].view(1, -1)], 1).expand(pts.shape[0], 1 + self.nchan)
                k = self.kern(torch.cat([fourier(rho, 6), fourier(pts / self.n, 4), cond], 1)).reshape(sh.shape[0], P, self.nk)
                return torch.cat([torch.nn.functional.softplus(k[..., 0]).mean(0)[None], k[..., 1:].mean(0).T], 0)  # nk, P
            rows.append(run_ckpt(row, rot["cs"][b], rot["parity"][b], rot["shifts"][b]))
        k = torch.stack(rows).reshape(M, self.nk, self.n, self.n)
        core = k[:, 0] / k[:, 0].sum((-2, -1), keepdim=True)
        rest = 0.01 * (k[:, 1:] - k[:, 1:].mean((-2, -1), keepdim=True))
        return torch.cat([core[:, None], rest], 1)

    def _basis(self, pix_per_lod, loglam, chan, rot=None):
        if rot is not None:
            return self._basis_rot(pix_per_lod, loglam, chan, rot)
        dev = self.loglam.device''')

# --------------------------------------------------------------------------- batches, SkyModel
rep('''def band_batches(exps, idx, nchunks=1, seed=0):
    out = []
    g = torch.Generator().manual_seed(seed)
    for b in REG.bands:
        sel = [i for i in idx if exps[i]["band"] == b]
        if not sel:
            continue
        dev = exps[sel[0]]["d"].device
        cat = lambda k: torch.cat([exps[i][k] for i in sel])
        full = {"tx": cat("tx"), "ty": cat("ty"), "d": cat("d"), "ivar": cat("ivar"), "dx": cat("dx"), "dy": cat("dy"),
                "row": cat("row"), "col": cat("col"),
                "eidx": torch.cat([torch.full((exps[i]["d"].numel(),), i, dtype=torch.long, device=dev) for i in sel]),
                "detv": torch.cat([torch.full((exps[i]["d"].numel(),), exps[i]["det"], dtype=torch.long, device=dev) for i in sel])}
        n = full["d"].numel()
        perm = torch.randperm(n, generator=g).to(dev)
        for c in range(nchunks):
            ix = perm[c::nchunks]
            ch = {k: v[ix].contiguous() for k, v in full.items()}
            ch.update(band=b, det=exps[sel[0]]["det"], chunk=c)
            out.append(ch)
        del full
    return out
''',
    '''def band_batches(exps, idx, nchunks=1, seed=0):
    """Batches per band (sky frame) or per (band, PSF group) (v33 detector frame: batch["pgrp"] = group id)."""
    out = []
    g = torch.Generator().manual_seed(seed)
    for b in REG.bands:
        sel0 = [i for i in idx if exps[i]["band"] == b]
        if not sel0:
            continue
        if PSF_FRAME == "sky":
            parts = [sel0]
        else:
            parts = [[i for i in sel0 if exps[i]["pgrp"] == gid] for gid in sorted({exps[i]["pgrp"] for i in sel0})]
        for sel in parts:
            dev = exps[sel[0]]["d"].device
            cat = lambda k: torch.cat([exps[i][k] for i in sel])
            full = {"tx": cat("tx"), "ty": cat("ty"), "d": cat("d"), "ivar": cat("ivar"), "dx": cat("dx"), "dy": cat("dy"),
                    "row": cat("row"), "col": cat("col"),
                    "eidx": torch.cat([torch.full((exps[i]["d"].numel(),), i, dtype=torch.long, device=dev) for i in sel]),
                    "detv": torch.cat([torch.full((exps[i]["d"].numel(),), exps[i]["det"], dtype=torch.long, device=dev) for i in sel])}
            n = full["d"].numel()
            perm = torch.randperm(n, generator=g).to(dev)
            for c in range(nchunks):
                ix = perm[c::nchunks]
                ch = {k: v[ix].contiguous() for k, v in full.items()}
                ch.update(band=b, det=exps[sel[0]]["det"], chunk=c)
                if PSF_FRAME != "sky":
                    ch["pgrp"] = exps[sel[0]]["pgrp"]
                out.append(ch)
            del full
    return out
''')

rep('''    def prepare(self, A=None):
        if hasattr(self.sky, "nodes"):
            Sn, S = self.sky.nodes() if A is None else self.sky.nodes(A)  # nb Q H W ; nb H W''',
    '''    def prepare_groups(self, A=None, groups=None):
        """v33 (detector frame): sky convolved once per PSF group: S (nb,H,W), CB (G, nk, H, W). groups: None = the groups of the
        training exposures, "own" = theta 0 per band (display cubes), or a list of group ids."""
        G = PSFG.select(groups)
        sel = torch.tensor([g["b"] for g in G], dtype=torch.long)
        step = PSFG.chunk
        if hasattr(self.sky, "nodes"):
            Sn, S = self.sky.nodes() if A is None else self.sky.nodes(A)  # nb Q H W ; nb H W
            q = Sn.shape[1]
            outs = []
            for a0 in range(0, len(G), step):
                Gc, sc = G[a0:a0 + step], sel[a0:a0 + step].to(Sn.device)
                B = partial_kernels(psf_group_kernels(self.psf, Gc, self.sky.lam_nodes), TARGET_SIG_PX)  # (Gc Q) nk n n
                CBn = convolve_basis(Sn[sc].flatten(0, 1), B)
                outs.append((CBn.view(len(Gc), q, *CBn.shape[1:]) * self.sky.w_nodes[sc][:, :, None, None, None]).sum(1))
            return S, torch.cat(outs)
        S = self.sky()
        outs = []
        for a0 in range(0, len(G), step):
            Gc, sc = G[a0:a0 + step], sel[a0:a0 + step].to(S.device)
            outs.append(convolve_basis(S[sc], partial_kernels(psf_group_kernels(self.psf, Gc), TARGET_SIG_PX)))
        return S, torch.cat(outs)

    def prepare(self, A=None, groups=None):
        if PSF_FRAME != "sky":
            return self.prepare_groups(A, groups)
        if hasattr(self.sky, "nodes"):
            Sn, S = self.sky.nodes() if A is None else self.sky.nodes(A)  # nb Q H W ; nb H W''')
rep('''        v = sample(CB[b], x, y)  # nk N
        w = self.psf.weights(b, det, e["dx"], e["dy"])  # N nk''',
    '''        v = sample(CB[e["pgrp"] if "pgrp" in e else b], x, y)  # nk N  (v33: CB is indexed by PSF group in the detector frame)
        w = self.psf.weights(b, det, e["dx"], e["dy"])  # N nk''')

# --------------------------------------------------------------------------- held-out scoring
rep('''    per_exposure = []  # v22: one record per scored exposure
    heldout_chi2.per_exposure = per_exposure
''', '''    per_exposure = []  # v22: one record per scored exposure
    heldout_chi2.per_exposure = per_exposure
    gcache = {}  # v33: convolved basis images of the PSF groups met so far in this call
''')
rep('''                _, CB = (None, None) if zipm else m0.prepare()''',
    '''                _, CB = (None, None) if zipm else (m0.prepare() if PSF_FRAME == "sky" else (None, group_cb(m0, e, gcache)))''')
rep('''            CB = CB[b].detach() if CB is not None else None''',
    '''            CB = (CB[b] if PSF_FRAME == "sky" else CB).detach() if CB is not None else None''')

# --------------------------------------------------------------------------- splat2 problem: CB is indexed by group
rep('''                self.cache.append({"b": b, "x": x, "y": y, "w": w,''', '''                self.cache.append({"b": b, "g": e.get("pgrp", b), "x": x, "y": y, "w": w,''')
rep('''                z = (c["ivar"] * (c["y_res"] - self._lin(CB[c["b"]], c)) ** 2).sqrt()''',
    '''                z = (c["ivar"] * (c["y_res"] - self._lin(CB[c["g"]], c)) ** 2).sqrt()''')
rep('''            leaf = CB[c["b"]].detach().requires_grad_(True)
            y = self._lin(leaf, c)
            g, = torch.autograd.grad(y, leaf, grad_outputs=c["W"] * y.detach())
            gCB[c["b"]] += g''',
    '''            leaf = CB[c["g"]].detach().requires_grad_(True)
            y = self._lin(leaf, c)
            g, = torch.autograd.grad(y, leaf, grad_outputs=c["W"] * y.detach())
            gCB[c["g"]] += g''')
rep('''            return torch.cat([self._lin(CB[c["b"]], c) for c in self.cache])''',
    '''            return torch.cat([self._lin(CB[c["g"]], c) for c in self.cache])''')
rep('''            g, = torch.autograd.grad(y, leaf, grad_outputs=c["W"] * c["y_res"])
            gCB[c["b"]] += g''',
    '''            g, = torch.autograd.grad(y, leaf, grad_outputs=c["W"] * c["y_res"])
            gCB[c["g"]] += g''')

# --------------------------------------------------------------------------- run(): setup, warm start, outputs
rep('''    set_registry(registry_from_exposures(exps, psf_n=getattr(a, "psf_n", 31)))
    LOG["band_registry"] = REG.summary()''',
    '''    set_registry(registry_from_exposures(exps, psf_n=getattr(a, "psf_n", 31)))
    configure_psf_frame(a)
    LOG["band_registry"] = REG.summary()''')
rep('''    band_index = dict(REG.index)

    if getattr(a, "coarse_shift_jwst", 0.0) > 0:''',
    '''    band_index = dict(REG.index)
    if PSF_FRAME == "detector":  # v33: J_e, PSF groups
        setup_psf_frame(exps, train_idx, a, dev)

    if getattr(a, "coarse_shift_jwst", 0.0) > 0:''')
rep('''    sig0 = torch.tensor([REG.fwhm_arcsec[b] / a.scale / 2.3548 for b in REG.bands], device=dev)
''', '''    sig0 = torch.tensor([REG.fwhm_arcsec[b] / a.scale / 2.3548 for b in REG.bands], device=dev)
    if PIXEL_BOX:  # v33: the learned kernel is the optical PSF only: start from the nominal width without the detector pixel's variance
        sig0 = box_adjusted_sigma(sig0)
''')
rep('''            S, CB = m.prepare()
            B = m.psf.basis()
            results[name + "_ownpsf"] = CB[:, 0]''',
    '''            S, CB = m.prepare() if PSF_FRAME == "sky" else m.prepare(groups="own")  # v33: ownpsf = theta 0 (and the pixel box when on)
            B = m.psf.basis()
            results[name + "_ownpsf"] = CB[:, 0]''')
rep('''            np.save(out / f"{name}_psf.npy", psfs, allow_pickle=True)
''', '''            np.save(out / f"{name}_psf.npy", psfs, allow_pickle=True)
            if PSF_FRAME == "detector":  # v33: effective kernel of every PSF group on the sky grid (box included), at the group's first detector centre
                gk = {}
                for g_ in PSFG.groups[:PSFG.n_train]:
                    Bg = psf_group_kernels(m.psf, [g_])[0]
                    z = torch.zeros(1, device=dev)
                    gk[g_["id"]] = {"band": g_["band"], "theta_deg": g_["theta_deg"], "parity": g_["parity"], "S": g_["S"],
                                    "kernel": effective_psf(Bg, m.psf.weights(g_["b"], g_["det"], z, z)[0]).cpu().numpy()}
                np.save(out / f"{name}_psf_groups.npy", gk, allow_pickle=True)
''')

# --------------------------------------------------------------------------- arguments
rep('''    ap.add_argument("--budget-implicit", type=float, default=35.0, help="minutes for the implicit fit")
    a = ap.parse_args()
''', '''    ap.add_argument("--budget-implicit", type=float, default=35.0, help="minutes for the implicit fit")
    ap.add_argument("--psf-frame", choices=("sky", "detector"), default="sky", help="v33: sky = the PSF kernel lives on the sky grid (v32); detector = kernel defined in the detector frame and rotated per (band, roll) PSF group")
    ap.add_argument("--pixel-box", action="store_true", help="v33: (needs --psf-frame detector) the learned kernel is the optical PSF only; the detector pixel's box response is applied in the detector frame")
    ap.add_argument("--psf-group-deg", type=float, default=0.5, help="v33: roll angles are grouped by rounding theta_e to this many degrees")
    a = ap.parse_args()
    configure_psf_frame(a)
''')

anchor = "def load_state_checked(m, path, dev):"
rep(anchor, BLOCK + anchor)
open(dst, "w").write(s)
print("wrote", dst, len(s.splitlines()), "lines")
