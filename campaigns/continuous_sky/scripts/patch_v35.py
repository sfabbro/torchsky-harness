"""v34 -> v35 (6 Nov 2026): experiment E2 of psf-geometry-audit.md, physics warm start of the PSF, wide window, deconvolved sky start.
Problem (audit #1, #2, #5, #7): the sky started from the drizzle coadd, the PSF from a narrow Gaussian, the window was 1.24 arcsec and one
lambda/D-scaled network served every telescope, so the fit settled in the basin where the sky holds the PSF and the kernel is only a relative
kernel. E2 breaks that degeneracy with warm starts only (the library is never a hard constraint):

--psf-model lib --explicit-psf lib  (needs --psf-frame detector, NOT --pixel-box, --psf-lib DIR, --skip-implicit; default off = v34, bit-identical)
  LibPSF replaces ChromaticPSF for the explicit model. K(u) = L_{d,b,q}(u) + eps_{d,b,q} r_c(u; log lambda_q, detector embedding) in the
  DETECTOR frame (u = detector-pixel offset), L the library effective PSF (ePSF; already contains the pixel response, hence no --pixel-box)
  read by bicubic grid_sample from the oversampled array (zero outside), eps = 0.1 max(L) (fixed), r_c one MLP per REG channel (no sharing
  across telescopes, lambda only an input feature), last layer zero-initialised so the warm start is exactly the library. Group kernel on
  the sky grid: mean over 3 x 3 midpoint sub-positions of the sky CELL of K(u), u = (F M F)^-1 xi, xi = rot_xy(delta + v), then unit sum.
  nk = 1 (forced), weights() = ones: no spatial variation in E2. PSF groups include the detector index in their key.
--psf-lib DIR  nircam_epsf.npz ("NRCB3/F150W" -> (Q, N, N)), wfc3ir_epsf.npz ("WFC3IR_F105W" -> Anderson 3 x 3 grid, centre index 4), meta.json.
--psf-window-arcsec W  kernel window = odd(round(W / scale)) sky pixels (0 = --psf-n).
--sky-init deconv  --rl-iters N (30): Richardson-Lucy of each band's init coadd with the library kernel (middle node, first training group).
--sample-mode {bilinear,bicubic}  interpolation of SkyModel.predict and of the held-out scorer's model read-out (not the masks).
--lib-res-l2 X (0)  optional L2 on the residual outputs; --lib-res-hidden H (64).
Library-specific --psf-centre acts as a centroid gauge on K relative to the library's first moment (detector px)."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:80])
    s = s.replace(old, new)


# ------------------------------------------------------------------ arguments
rep('''    ap.add_argument("--psf-model", choices=("band", "chrom", "pupil"), default="band",''',
    '''    ap.add_argument("--psf-model", choices=("band", "chrom", "pupil", "lib"), default="band",''')
rep('''    ap.add_argument("--explicit-psf", choices=("band", "chrom", "pupil"), default="band")''',
    '''    ap.add_argument("--explicit-psf", choices=("band", "chrom", "pupil", "lib"), default="band")''')
rep('''    ap.add_argument("--psf-group-deg", type=float, default=0.5, help="v33: roll angles are grouped by rounding theta_e to this many degrees")
''', '''    ap.add_argument("--psf-group-deg", type=float, default=0.5, help="v33: roll angles are grouped by rounding theta_e to this many degrees")
    ap.add_argument("--psf-lib", default=None, help="v35: directory with nircam_epsf.npz, wfc3ir_epsf.npz and meta.json (--psf-model lib --explicit-psf lib)")
    ap.add_argument("--psf-window-arcsec", type=float, default=0.0, help="v35: kernel window on the sky grid [arcsec]; psf_n = odd(round(W / scale)); 0 = use --psf-n")
    ap.add_argument("--sky-init", choices=("keep", "deconv"), default="keep", help="v35: deconv = Richardson-Lucy deconvolution of the init coadd with the library kernel before the spectral initialisation")
    ap.add_argument("--rl-iters", type=int, default=30, help="v35: Richardson-Lucy iterations of --sky-init deconv")
    ap.add_argument("--sample-mode", choices=("bilinear", "bicubic"), default="bilinear", help="v35: interpolation used to read the convolved model at pixel positions (predict and held-out scorer)")
    ap.add_argument("--lib-res-l2", type=float, default=0.0, help="v35: L2 weight on the residual-network outputs of the library PSF (a soft prior toward the library; default off)")
    ap.add_argument("--lib-res-hidden", type=int, default=64, help="v35: hidden width of the library PSF residual networks")
''')

rep('''    a = ap.parse_args()
    configure_psf_frame(a)
''', '''    a = ap.parse_args()
    if a.psf_window_arcsec > 0:  # v35: window in arcsec -> odd number of sky pixels
        n_w = int(round(a.psf_window_arcsec / a.scale))
        a.psf_n = n_w + 1 - n_w % 2
        LOG["psf_window"] = {"window_arcsec": a.psf_window_arcsec, "scale": a.scale, "psf_n": a.psf_n, "realised_arcsec": round(a.psf_n * a.scale, 4)}
    configure_psf_frame(a)
''')

# ------------------------------------------------------------------ configuration hook
rep('''def configure_psf_frame(a):
    global PSF_FRAME, PIXEL_BOX
''', '''def configure_psf_frame(a):
    configure_v35(a)
    global PSF_FRAME, PIXEL_BOX
''')

# ------------------------------------------------------------------ PSF groups: detector in the key for the library PSF
rep('''        return (REG.index[e["band"]], self.theta_key(j["theta_deg"]), int(j["parity"]))''',
    '''        k = (REG.index[e["band"]], self.theta_key(j["theta_deg"]), int(j["parity"]))
        return k + (int(e["det"]),) if self.det_key else k  # v35: the library PSF differs per detector''')
rep('''    chunk = 8  # groups convolved per FFT call (bounds the temporaries)
''', '''    chunk = 8  # groups convolved per FFT call (bounds the temporaries)
    det_key = False  # v35: True with the library PSF
''')
rep('''            sx, sy = (gb["sx"], gb["sy"]) if gb is not None else (1.0, 1.0)
            S = int(math.ceil(2.0 * max(sx, sy))) if self.box else 1
            out.append({"id": -1, "b": b, "band": REG.bands[b], "theta_deg": 0.0, "parity": 1, "M": [[sx, 0.0], [0.0, sy]], "sx": sx, "sy": sy, "S": max(S, 1)})''',
    '''            sx, sy = (gb["sx"], gb["sy"]) if gb is not None else (1.0, 1.0)
            S = int(math.ceil(2.0 * max(sx, sy))) if self.box else 1
            out.append({"id": -1, "b": b, "band": REG.bands[b], "theta_deg": 0.0, "parity": 1, "M": [[sx, 0.0], [0.0, sy]], "sx": sx, "sy": sy, "S": max(S, 1),
                        "det": gb["det"] if gb is not None else 0})  # v35: first detector of the band (library PSF)''')
rep('''    PSFG = PSFGroups(getattr(a, "psf_group_deg", 0.5), PIXEL_BOX)
''', '''    PSFG = PSFGroups(getattr(a, "psf_group_deg", 0.5), PIXEL_BOX)
    PSFG.det_key = LIB_ON
''')
rep('''    LOG["psf_frame"] = {"frame": PSF_FRAME, "pixel_box": PIXEL_BOX, "group_deg": PSFG.step, "n_train_groups": PSFG.n_train, "groups": PSFG.table(),''',
    '''    if LIB_ON:
        tick("library PSF: PSF groups keyed by (band, theta, parity, detector); nk = 1, no spatial variation of the PSF over the detector (E2)")
    LOG["psf_frame"] = {"frame": PSF_FRAME, "pixel_box": PIXEL_BOX, "group_deg": PSFG.step, "n_train_groups": PSFG.n_train, "groups": PSFG.table(),''')

# ------------------------------------------------------------------ dispatch of the group kernels
rep('''    dev = psf.det_emb.weight.device if hasattr(psf, "det_emb") else psf.res.device
    sel = torch.tensor([g["b"] for g in groups], dtype=torch.long, device=dev)''',
    '''    if isinstance(psf, LibPSF):  # v35
        return psf.group_kernels(groups, lam_nodes)
    dev = psf.det_emb.weight.device if hasattr(psf, "det_emb") else psf.res.device
    sel = torch.tensor([g["b"] for g in groups], dtype=torch.long, device=dev)''')

# ------------------------------------------------------------------ read-out interpolation
rep('''        v = sample(CB[e["pgrp"] if "pgrp" in e else b], x, y)  # nk N  (v33: CB is indexed by PSF group in the detector frame)''',
    '''        v = (sample if SAMPLE_MODE == "bilinear" else sample_bicubic)(CB[e["pgrp"] if "pgrp" in e else b], x, y)  # nk N  (v33: CB is indexed by PSF group in the detector frame; v35: --sample-mode)''')
rep('''            return (w * sample(CB, x, y).T).sum(1)
''', '''            return (w * (sample if SAMPLE_MODE == "bilinear" else sample_bicubic)(CB, x, y).T).sum(1)
''')

# ------------------------------------------------------------------ run(): explicit model, learning rate, name suffix, sky init
rep('''    esfx = ("_chrom" if a.explicit_psf == "chrom" else "") + ("_3d" if spec else "") + a.tag''',
    '''    esfx = ("_chrom" if a.explicit_psf == "chrom" else ("_lib" if a.explicit_psf == "lib" else "")) + ("_3d" if spec else "") + a.tag''')
rep('''        elif a.explicit_psf == "chrom":
            epsf = ChromaticPSF(REG.bands, ndet, a.scale, a.nk, n=a.psf_n).to(dev)
            if not getattr(a, "load_dir", None):''', '''        elif a.explicit_psf == "lib":  # v35
            epsf = make_libpsf(a, dets, exps, band_index, lam_nodes if spec else None, dev)
            tick(f"{name}: library PSF warm start = the library itself (residual networks zero-initialised): "
                 f"{sum(p_.numel() for p_ in epsf.parameters()) / 1e3:.1f} k residual parameters")
        elif a.explicit_psf == "chrom":
            epsf = ChromaticPSF(REG.bands, ndet, a.scale, a.nk, n=a.psf_n).to(dev)
            if not getattr(a, "load_dir", None):''')
rep('''dict(lrs_e, psf=5e-4) if a.explicit_psf == "chrom" else lrs_e, sig_b, train_idx, band_index,''',
    '''explicit_lrs(a, lrs_e), sig_b, train_idx, band_index,''')
rep('''fit(name, m, exps, steps_e, dict(dict(lrs_e, psf=5e-4) if a.explicit_psf == "chrom" else lrs_e, **({"sky"''',
    '''fit(name, m, exps, steps_e, dict(explicit_lrs(a, lrs_e), **({"sky"''')
rep('''    models = {}
    esfx = ''', '''    if getattr(a, "sky_init", "keep") == "deconv":  # v35: sky warm start deconvolved with the library PSF, not the drizzle/SWarp coadd
        init = deconv_init(init, stats, sig_b, make_libpsf(a, dets, exps, band_index, lam_nodes if spec else None, dev), a, dev)
    models = {}
    esfx = ''')

# ------------------------------------------------------------------ diagnostics
rep('''                np.save(out / f"{name}_psf_groups.npy", gk, allow_pickle=True)''',
    '''                if LIB_ON:  # v35: detector-frame K(u) on the library's oversampled grid (middle node) per (detector, band), with the library
                    gk["detector_frame"] = m.psf.detector_dump()
                np.save(out / f"{name}_psf_groups.npy", gk, allow_pickle=True)''')

# ------------------------------------------------------------------ the new code
BLOCK = r'''# ----------------------------------------------------------------------------- v35: library PSF (E2), RL sky start, bicubic read-out
LIB_ON = False
SAMPLE_MODE = "bilinear"
LIB_STORE = {}


def configure_v35(a):
    """Flags of v35: validate and set the module switches (idempotent). The library PSF is an effective PSF (pixel response included)."""
    global LIB_ON, SAMPLE_MODE
    SAMPLE_MODE = getattr(a, "sample_mode", "bilinear")
    ex, im = getattr(a, "explicit_psf", ""), getattr(a, "psf_model", "")
    LIB_ON = "lib" in (ex, im)
    if LIB_ON:
        if ex != "lib" or im != "lib":
            raise SystemExit("--psf-model lib and --explicit-psf lib go together")
        if getattr(a, "psf_frame", "sky") != "detector":
            raise SystemExit("the library PSF needs --psf-frame detector")
        if getattr(a, "pixel_box", False):
            raise SystemExit("the library PSFs are effective PSFs that already include the detector pixel response: do not combine with --pixel-box")
        if not getattr(a, "psf_lib", None):
            raise SystemExit("the library PSF needs --psf-lib DIR")
        if not getattr(a, "skip_implicit", False):
            raise SystemExit("the library PSF is explicit-only in v35: add --skip-implicit")
        a.nk = 1  # one kernel per group: no spatial variation in E2
    if getattr(a, "sky_init", "keep") == "deconv" and not LIB_ON:
        raise SystemExit("--sky-init deconv needs the library kernels (--psf-model lib --explicit-psf lib --psf-lib DIR)")
    if getattr(a, "rl_iters", 30) < 0:
        raise SystemExit("--rl-iters must be >= 0")


def sample_bicubic(img, tx, ty):
    """As `sample` (0-based pixel centres, zero padding) with bicubic interpolation (Keys, a = -0.75, torch grid_sample)."""
    squeeze = img.dim() == 2
    if squeeze:
        img = img[None]
    C, H, W = img.shape
    gx = (tx + 0.5) / W * 2 - 1
    gy = (ty + 0.5) / H * 2 - 1
    g = torch.stack([gx, gy], -1)[None, None]
    out = F.grid_sample(img[None], g, mode="bicubic", padding_mode="zeros", align_corners=False)[0, :, 0]
    return out[0] if squeeze else out


def explicit_lrs(a, lrs_e):
    """Learning rates of the explicit fit: PSF 5e-4 for the chromatic field (v34), 1e-3 for the library residuals, else unchanged."""
    if a.explicit_psf == "chrom":
        return dict(lrs_e, psf=5e-4)
    if a.explicit_psf == "lib":
        return dict(lrs_e, psf=1e-3)
    return lrs_e


from scipy.ndimage import spline_filter  # noqa: E402


class PSFLibrary:
    """Reader of the physics PSF library directory (see scripts/gen_psf_lib.py): nircam_epsf.npz, wfc3ir_epsf.npz, meta.json."""

    def __init__(self, path):
        self.path = Path(path)
        meta = json.load(open(self.path / "meta.json"))
        self.oversample = int(meta["oversample"])
        self.nodes_um = {b: [float(x) for x in v] for b, v in meta["nodes_um"].items()}
        self.nir = np.load(self.path / "nircam_epsf.npz") if (self.path / "nircam_epsf.npz").exists() else None
        self.wfc = np.load(self.path / "wfc3ir_epsf.npz") if (self.path / "wfc3ir_epsf.npz").exists() else None
        self.notes = []

    @staticmethod
    def libdet(detname):
        d = detname.strip().upper()
        return d[:-4] + "5" if d.endswith("LONG") else d  # NRCBLONG <-> NRCB5

    def get(self, detname, band, channel):
        """-> (array (Q, N, N) float32 unit sum per node, library node wavelengths [um] or None (one array for every node))."""
        if channel.startswith("NIRCAM"):
            key = f"{self.libdet(detname)}/{band}"
            if self.nir is None or key not in self.nir.files:
                raise SystemExit(f"PSF library {self.path}: no NIRCam entry {key!r} (detector {detname}, band {band})")
            arr = np.asarray(self.nir[key], np.float32)
            if arr.ndim != 3 or band not in self.nodes_um or arr.shape[0] != len(self.nodes_um[band]):
                raise SystemExit(f"PSF library {key}: shape {arr.shape} does not match the {len(self.nodes_um.get(band, []))} nodes of meta.json")
            return arr / arr.sum((-2, -1), keepdims=True), self.nodes_um[band]
        if channel == "WFC3_IR":
            key = "WFC3IR_" + band.split("_", 1)[-1]
            if self.wfc is None or key not in self.wfc.files:
                raise SystemExit(f"PSF library {self.path}: no WFC3/IR entry {key!r}")
            arr = np.asarray(self.wfc[key], np.float32)
            shp = arr.shape
            if arr.ndim == 2:
                c, how = arr, "single 2-D array"
            else:
                if arr.shape[-1] <= 16 and arr.shape[-2] > 16:  # (N, N, K): grid axis last
                    arr = np.moveaxis(arr, -1, 0)
                a3 = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
                ic = 4 if a3.shape[0] == 9 else a3.shape[0] // 2
                c, how = a3[ic], f"grid of {a3.shape[0]}, index {ic}" + ("" if a3.shape[0] == 9 else " (UNEXPECTED grid size, middle entry taken)")
            self.notes.append(f"{key}: array shape {shp}: {how}")
            return (c / c.sum())[None].astype(np.float32), None
        raise SystemExit(f"no PSF library for channel {channel} (band {band})")


class LibPSF(nn.Module):
    """E2 PSF: per-instrument library effective PSF (warm start) + a bounded residual from one MLP per REG channel (see patch_v35.py).
    Interface of ChromaticPSF where the SkyModel needs it: weights(), penalty(), centre_weight, basis(), det_emb, loglam; the kernels of PSF
    groups come from group_kernels() (dispatched by psf_group_kernels)."""

    def __init__(self, bands, det_names, pairs, lib, lam_nodes, scale_arcsec, n=31, hidden=64, nf=5, res_l2=0.0, sc=3, reg=None):
        super().__init__()
        reg = reg or REG
        self.n, self.nk, self.nb, self.nf, self.sc = n, 1, len(bands), nf, sc
        self.os, self.res_l2, self.scale_arcsec = lib.oversample, float(res_l2), scale_arcsec
        self.det_names, self.bands = list(det_names), list(bands)
        self.chan = [reg.channel[b] for b in bands]
        lam = torch.tensor([reg.lam_um[b] for b in bands])
        self.register_buffer("loglam", (torch.log(lam) - math.log(2.0)) / 0.8)
        din = 4 * nf + 1 + 4
        self.nets = nn.ModuleDict()
        for c in reg.channels:
            net = nn.Sequential(nn.Linear(din, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.nets[c] = net
        self.det_emb = nn.Embedding(len(det_names), 4)
        self._L, self._C, self._eps, self.lam_q, self._Rdet = {}, {}, {}, {}, {}
        self.interp = "bspline"  # "bspline": exact cubic B-spline of the library (prefiltered); "grid_sample": torch bicubic (Keys a = -0.75, 2nd order)
        msgs = []
        for d, b in sorted(pairs):
            band = bands[b]
            arr, lnodes = lib.get(det_names[d], band, reg.channel[band])
            if lam_nodes is not None:
                run_nodes = [float(x) for x in lam_nodes[b]]
            elif lnodes is not None:
                run_nodes = list(lnodes)
            else:
                run_nodes = [reg.lam_um[band]]
            if lnodes is None:  # WFC3/IR: the same ePSF for every node (APPROXIMATION)
                idx = [0] * len(run_nodes)
                if b not in self.lam_q:
                    msgs.append(f"APPROXIMATION: {band} {det_names[d]}: one ePSF (library centre PSF) for every quadrature node, no within-band chromatic change")
            else:
                idx = [int(np.argmin([abs(x - r) for x in lnodes])) for r in run_nodes]
                dev_ = max(abs(lnodes[i] - r) for i, r in zip(idx, run_nodes))
                if dev_ > 2e-3 and b not in self.lam_q:
                    msgs.append(f"WARNING: {band}: run nodes {np.round(run_nodes, 4).tolist()} differ from the library nodes {np.round(lnodes, 4).tolist()} by up to "
                                f"{dev_:.4f} um (> 2e-3): nearest library node used for each run node (indices {idx})")
            if b not in self.lam_q:
                self.lam_q[b] = run_nodes
            Lt = torch.from_numpy(np.ascontiguousarray(arr[idx]))
            self._L[(d, b)] = Lt
            cache = {i: spline_filter(arr[i].astype(np.float64), order=3, mode="mirror").astype(np.float32) for i in sorted(set(idx))}
            self._C[(d, b)] = torch.from_numpy(np.stack([cache[i] for i in idx]))
            self._eps[(d, b)] = [0.1 * float(Lt[i].max()) for i in range(Lt.shape[0])]
        self.messages = msgs
        self.centre_weight = 0.0

    # ---- geometry of one group
    def _pair(self, g):
        b = g["b"]
        d = int(g.get("det", 0))
        if (d, b) in self._L:
            return d
        for (d2, b2) in sorted(self._L):
            if b2 == b:
                return d2
        raise KeyError(f"no library PSF for band {self.bands[b]}")

    def _R(self, g, d, b):
        """Half window in detector pixels (scale of the residual coordinates)."""
        if (d, b) not in self._Rdet:
            self._Rdet[(d, b)] = (self.n // 2) / math.sqrt(abs(g["M"][0][0] * g["M"][1][1] - g["M"][0][1] * g["M"][1][0]))
        return self._Rdet[(d, b)]

    def _points(self, g, dev):
        """Detector-pixel offsets u (S_c^2 * n^2, 2) of the sub-positions of every sky cell of the n x n window of group g."""
        n, sc = self.n, self.sc
        r = torch.arange(n, device=dev, dtype=torch.float32) - n // 2
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        xy = torch.stack([xx, yy], -1).reshape(-1, 2)
        t = (torch.arange(sc, device=dev, dtype=torch.float32) + 0.5) / sc - 0.5
        vy, vx = torch.meshgrid(t, t, indexing="ij")
        v = torch.stack([vx.reshape(-1), vy.reshape(-1)], 1)
        pts = (xy[None] + v[:, None]).reshape(-1, 2)
        cs = torch.tensor(_cs(g["theta_deg"]), dtype=torch.float32, device=dev)
        xi = rot_xy(pts, cs, torch.tensor(float(g["parity"]), device=dev), torch.zeros(1, 2, device=dev))  # F R^T (delta + v), target px
        Fm = torch.diag(torch.tensor([1.0, float(g["parity"])], dtype=torch.float64))
        Mm = torch.tensor(g["M"], dtype=torch.float64)
        Minv = torch.linalg.inv(Fm @ Mm @ Fm)  # xi = F M F u
        return (xi.double().cpu() @ Minv.T).float().to(dev)

    def _read(self, d, b, q, u):
        """Library value at detector offsets u (npts, 2) [detector px] from the oversampled array, zero outside: (npts,). Default: cubic
        B-spline interpolation (coefficients from scipy spline_filter at load; 4th-order accurate). self.interp = "grid_sample": torch bicubic."""
        L = self._get_L(d, b)
        N = L.shape[-1]
        if self.interp == "grid_sample":
            h = (N - 1) / 2.0
            grid = torch.stack([(u[:, 0] * self.os + N // 2 - h) / h, (u[:, 1] * self.os + N // 2 - h) / h], -1)
            return F.grid_sample(L[q][None, None], grid.view(1, 1, -1, 2), mode="bicubic", padding_mode="zeros", align_corners=True).view(-1)
        C = self._get_C(d, b)[q].reshape(-1)
        px, py = u[:, 0] * self.os + N // 2, u[:, 1] * self.os + N // 2
        ix, iy = torch.floor(px), torch.floor(py)
        tx, ty = px - ix, py - iy
        ix, iy = ix.long(), iy.long()

        def wts(t):
            return [(1 - t) ** 3 / 6.0, (3 * t ** 3 - 6 * t ** 2 + 4) / 6.0, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6.0, t ** 3 / 6.0]
        wx, wy = wts(tx), wts(ty)
        out = torch.zeros_like(px)
        for l in range(4):
            yy = iy - 1 + l
            oky = (yy >= 0) & (yy < N)
            for k in range(4):
                xx = ix - 1 + k
                ok = oky & (xx >= 0) & (xx < N)
                out = out + torch.where(ok, C[yy.clamp(0, N - 1) * N + xx.clamp(0, N - 1)], torch.zeros_like(px)) * (wy[l] * wx[k])
        return out

    def _res(self, ch, d, lamf, u, R):
        x = torch.cat([fourier(u / R, self.nf), lamf.view(1, 1).expand(u.shape[0], 1), self.det_emb.weight[d].view(1, 4).expand(u.shape[0], 4)], 1)
        return self.nets[ch](x)[:, 0]

    def _get_L(self, d, b):
        dev = self.det_emb.weight.device
        t = self._L[(d, b)]
        if t.device != dev:
            t = self._L[(d, b)] = t.to(dev)
        return t

    def _get_C(self, d, b):
        dev = self.det_emb.weight.device
        t = self._C[(d, b)]
        if t.device != dev:
            t = self._C[(d, b)] = t.to(dev)
        return t

    # ---- kernels
    def group_kernels(self, groups, lam_nodes=None, residual=True):
        """(G*Q, 1, n, n) sky-grid kernels at the Q nodes (lam_nodes (nb, Q) given), or (G, 1, n, n) at the middle node."""
        dev = self.det_emb.weight.device
        n, S2 = self.n, self.sc ** 2
        out = []
        for g in groups:
            b = g["b"]
            d = self._pair(g)
            L = self._get_L(d, b)
            Qt = L.shape[0]
            R = self._R(g, d, b)
            u = self._points(g, dev)
            if lam_nodes is None:
                qs = [Qt // 2]
                lams = [torch.tensor((math.log(self.lam_q[b][Qt // 2]) - math.log(2.0)) / 0.8, device=dev)]
            else:
                if lam_nodes.shape[1] != Qt:
                    raise SystemExit(f"library PSF built for {Qt} nodes, asked for {lam_nodes.shape[1]}")
                qs = list(range(Qt))
                lams = [(torch.log(lam_nodes[b, q]) - math.log(2.0)) / 0.8 for q in qs]
            for q, lamf in zip(qs, lams):
                def row(lamf, u, q=q, b=b, eps=self._eps[(d, b)][q], ch=self.chan[b], d=d, R=R):
                    K = self._read(d, b, q, u)
                    if residual:
                        K = K + eps * self._res(ch, d, lamf, u, R)
                    K = K.view(S2, n * n).mean(0)
                    return (K / K.sum()).view(n, n)
                out.append(run_ckpt(row, lamf.reshape(()), u))
        return torch.stack(out)[:, None]

    def basis(self):
        """(nb, 1, n, n): theta = 0, parity +1 kernels at the middle node, first detector of every band (display products)."""
        return self.group_kernels(PSFG.select("own"), None)

    def weights(self, b, det, dx, dy):
        return torch.ones(dx.shape[0], 1, device=dx.device)

    def _moment_grid(self, d, b, dev):
        R = self._Rdet.get((d, b))
        if R is None:
            R = (self.n // 2) / 1.0
        m = int(math.ceil(4 * R)) | 1
        ax = (torch.arange(m, device=dev, dtype=torch.float32) - m // 2) / 2.0
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], 1), R

    def penalty(self):
        """Centroid gauge (first moment of K minus the library's, detector px) times centre_weight, plus res_l2 times the mean squared residual."""
        wc, wl = float(getattr(self, "centre_weight", 0.0)), self.res_l2
        dev = self.det_emb.weight.device
        if not wc and not wl:
            return torch.zeros((), device=dev)
        tot = torch.zeros((), device=dev)
        for (d, b), L in self._L.items():
            L = self._get_L(d, b)
            u, R = self._moment_grid(d, b, dev)
            for q in range(L.shape[0]):
                Lv = self._read(d, b, q, u)
                lamf = (torch.log(torch.tensor(float(self.lam_q[b][q]), device=dev)) - math.log(2.0)) / 0.8
                r = self._res(self.chan[b], d, lamf, u, R)
                K = Lv + self._eps[(d, b)][q] * r
                if wc:
                    mk = (K[:, None] * u).sum(0) / K.sum()
                    ml = (Lv[:, None] * u).sum(0) / Lv.sum()
                    tot = tot + wc * (mk - ml).pow(2).sum()
                if wl:
                    tot = tot + wl * r.pow(2).mean()
        return tot

    @torch.no_grad()
    def detector_dump(self):
        """K(u) (learned, zero outside the square window |u| <= half_window_det_px, unit sum) and L(u) (library, whole lattice, unit sum) on the
        library's oversampled lattice at the middle node, per 'DET/BAND'."""
        dev = self.det_emb.weight.device
        out = {}
        for (d, b), L in sorted(self._L.items()):
            L = self._get_L(d, b)
            q = L.shape[0] // 2
            N = L.shape[-1]
            ax = (torch.arange(N, device=dev, dtype=torch.float32) - N // 2) / self.os
            yy, xx = torch.meshgrid(ax, ax, indexing="ij")
            u = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)
            R = self._Rdet.get((d, b), float(self.n // 2))
            lamf = (torch.log(torch.tensor(float(self.lam_q[b][q]), device=dev)) - math.log(2.0)) / 0.8
            Lv = L[q].reshape(-1)  # the lattice itself: the interpolant equals the samples there
            K = torch.cat([Lv[i:i + 65536] + self._eps[(d, b)][q] * self._res(self.chan[b], d, lamf, u[i:i + 65536], R) for i in range(0, u.shape[0], 65536)])
            K = K * ((u.abs() <= R).all(1)).float()  # the residual is only trained inside the window (|u| <= R, square footprint): zero outside
            out[f"{self.det_names[d]}/{self.bands[b]}"] = {"det": self.det_names[d], "band": self.bands[b], "node_index": q, "lam_um": self.lam_q[b][q],
                                                           "oversample": self.os, "half_window_det_px": R,
                                                           "library": (Lv / Lv.sum()).view(N, N).cpu().numpy(), "learned": (K / K.sum()).view(N, N).cpu().numpy()}
        return out


def make_libpsf(a, dets, exps, band_index, lam_nodes, dev):
    """LibPSF for the (detector, band) pairs present in the loaded exposures; the library is read once per run."""
    key = str(a.psf_lib)
    if key not in LIB_STORE:
        LIB_STORE[key] = PSFLibrary(a.psf_lib)
    lib = LIB_STORE[key]
    pairs = {(int(e["det"]), band_index[e["band"]]) for e in exps}
    psf = LibPSF(REG.bands, dets, pairs, lib, lam_nodes.cpu() if lam_nodes is not None else None, a.scale, n=a.psf_n, hidden=a.lib_res_hidden, res_l2=a.lib_res_l2)
    if a.psf_centre > 0:
        psf.centre_weight = a.psf_centre
    psf = psf.to(dev)
    for g_ in PSFG.groups:  # window scale (detector px) of every (detector, band) pair from its first group
        if (int(g_["det"]), g_["b"]) in psf._L:
            psf._R(g_, int(g_["det"]), g_["b"])
    for m_ in lib.notes + psf.messages:
        tick(m_)
    lib.notes = []
    LOG["lib_psf"] = {"dir": key, "oversample": lib.oversample, "pairs": [[dets[d], REG.bands[b]] for d, b in sorted(pairs)], "window_px": a.psf_n,
                      "residual_hidden": a.lib_res_hidden, "res_l2": a.lib_res_l2, "subpositions_per_cell": psf.sc ** 2, "messages": psf.messages,
                      "run_nodes_um": {REG.bands[b]: v for b, v in psf.lam_q.items()}}
    return psf


def deconv_init(init, stats, sig_b, psf, a, dev):
    """Richardson-Lucy deconvolution of every band image of `init` with the band's library kernel (middle node, first training PSF group,
    rendered on the sky grid), on (image - robust background) clipped at >= 0; the background is added back. The forward operator is
    convolve_basis (the convention of the model), the adjoint the same with the flipped kernel (reflect padding: approximate at the border)."""
    t0 = time.perf_counter()
    out = init.clone()
    rep_ = {}
    with torch.no_grad():
        for k, b in enumerate(REG.bands):
            g = next((g_ for g_ in PSFG.groups[:PSFG.n_train] if g_["b"] == k), None)
            if g is None:
                continue
            Kk = psf_group_kernels(psf, [g])  # (1, 1, n, n)
            Kt = Kk.flip(-2, -1)
            bg = float(stats[k][0])
            d = torch.nan_to_num(init[k] - bg).clamp(min=0.0)
            floor = 0.01 * float(sig_b[k])
            est = d + floor
            for _ in range(a.rl_iters):
                den = convolve_basis(est[None], Kk)[0, 0].clamp(min=1e-3 * floor)
                est = est * convolve_basis(((d + floor) / den)[None], Kt)[0, 0]
            new = est - floor + bg
            old_ex = float((init[k] - bg).abs().sum())
            rep_[b] = {"fraction_flux_changed_l1": float((new - init[k]).abs().sum()) / max(old_ex, 1e-30),
                       "flux_ratio": float((new - bg).sum()) / max(float((init[k] - bg).sum()), 1e-30), "peak_ratio": float((new - bg).max()) / max(float((init[k] - bg).max()), 1e-30),
                       "group": g["id"], "kernel_peak": float(Kk.max())}
            out[k] = new
    dt = time.perf_counter() - t0
    LOG["sky_init_deconv"] = {"iters": a.rl_iters, "seconds": round(dt, 1), "per_band": rep_,
                              "note": "RL of (init - median) clipped at >= 0 with the library kernel at the middle node of the first training PSF group; median added back"}
    LOG["timings"]["sky_init_deconv"] = round(dt, 1)
    tick(f"sky init: Richardson-Lucy x {a.rl_iters} with the library kernels in {dt:.1f} s; fraction of flux changed (L1 of the change over L1 of image - background) "
         + ", ".join(f"{b} {v['fraction_flux_changed_l1']:.3f}" for b, v in rep_.items()))
    return out


'''
rep('''def run(exps, thead, dev, a, out, classical_fn=None):''', BLOCK + '''def run(exps, thead, dev, a, out, classical_fn=None):''')
open(dst, "w").write(s)
print("ok", dst)
