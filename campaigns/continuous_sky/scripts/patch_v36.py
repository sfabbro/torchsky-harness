"""v35 -> v36 (6 Nov 2026): the DATA-DRIVEN arm of E2 (no physics library; degeneracies are to be broken by multi-instrument diversity).
All flags act on the library-PSF path only (--psf-model lib --explicit-psf lib); without them v36 is bit-identical to v35.

--lib-source {file,gauss}  file = v35 (--psf-lib DIR). gauss: no directory; for every (detector, band) of the training set an in-memory
  "library" ePSF that is a round Gaussian of sigma --lib-gauss-sigma detector px (default 1.0, generic, the same for all instruments),
  tabulated with oversampling --lib-table-os (4) of that detector's pixel, half-width ceil(R) detector px with R = (psf_n // 2) /
  sqrt|det M| (the window converted to detector px; the corners of a rotated window beyond R read zero), identical for every node.
--lib-learn-table  the table is a parameter: ONE table per (detector, band), shared by all quadrature nodes of the band (APPROXIMATION:
  achromatic within a band; with --lib-source file it starts from the middle-node table). T = softplus(theta) / sum softplus(theta)
  (positive, unit sum); theta = log expm1(max(T0, 1e-8 max T0)). Rendering as v35 (cubic B-spline of the table, then 3 x 3 cell average,
  unit sum on the sky grid). The spline prefilter is differentiable because it is applied as the dense matrix P = B^-1 (B the mirror-boundary
  tridiagonal 1/6, 2/3, 1/6 system, the scipy spline_filter(order 3, mode "mirror") operator): C = P T P^T, identical to scipy to float
  precision, so the interpolation accuracy is that of v35 (tests/test_v36_lib.py).
--lib-res {on,off}  off removes the per-channel residual networks.   --lib-table-tv X  total variation of T / max T (mean |dx| + mean |dy|).
--psf-centre  now also penalises the first moment of every learnable table about its centre (detector px).
Learning rate of the table parameters: --psf-lr if given else 1e-2 (own optimiser group; the other PSF parameters keep the psf rate).
The *_psf_groups.npy dump gains, per 'DET/BAND': 'table' (learned table), 'library' (the starting table), 'learned' (K = table + residual)."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:80])
    s = s.replace(old, new)


def between(start, end, new):
    global s
    assert s.count(start) == 1 and s.count(end) == 1, (start, end)
    i0, i1 = s.index(start), s.index(end)
    assert i0 < i1
    s = s[:i0] + new + s[i1:]


rep('''    ap.add_argument("--lib-res-hidden", type=int, default=64,''', '''    ap.add_argument("--lib-source", choices=("file", "gauss"), default="file", help="v36: file = --psf-lib directory (v35); gauss = generic round Gaussian tables, no physics library")
    ap.add_argument("--lib-gauss-sigma", type=float, default=1.0, help="v36: sigma [detector px] of the generic Gaussian table (--lib-source gauss)")
    ap.add_argument("--lib-table-os", type=int, default=4, help="v36: oversampling of the generic table relative to the detector pixel")
    ap.add_argument("--lib-learn-table", action="store_true", help="v36: the (detector, band) table is a learnable positive unit-sum parameter, shared by the nodes of the band")
    ap.add_argument("--lib-res", choices=("on", "off"), default="on", help="v36: off removes the per-channel residual networks of the library PSF")
    ap.add_argument("--lib-table-tv", type=float, default=0.0, help="v36: total-variation weight on the learnable table (of T / max T)")
    ap.add_argument("--lib-res-hidden", type=int, default=64,''')

rep('''        if not getattr(a, "psf_lib", None):
            raise SystemExit("the library PSF needs --psf-lib DIR")''', '''        if getattr(a, "lib_source", "file") == "file" and not getattr(a, "psf_lib", None):
            raise SystemExit("the library PSF needs --psf-lib DIR (or --lib-source gauss)")''')
rep('''    if getattr(a, "rl_iters", 30) < 0:''', '''    if not LIB_ON and (getattr(a, "lib_source", "file") != "file" or getattr(a, "lib_learn_table", False) or getattr(a, "lib_res", "on") != "on"
                       or getattr(a, "lib_table_tv", 0.0) > 0):
        raise SystemExit("--lib-source/--lib-learn-table/--lib-res/--lib-table-tv need --psf-model lib --explicit-psf lib")
    if LIB_ON and getattr(a, "lib_res", "on") == "off" and getattr(a, "lib_source", "file") == "gauss" and not getattr(a, "lib_learn_table", False):
        raise SystemExit("--lib-source gauss with --lib-res off and no --lib-learn-table has no free PSF parameters")
    if getattr(a, "rl_iters", 30) < 0:''')

# ---------------------------------------------------------------- classes
NEW_HELPERS = r'''class GaussLibrary:
    """In-memory stand-in of PSFLibrary for --lib-source gauss: a round Gaussian effective PSF of sigma `sigma` detector px for every
    (detector, band), tabulated at `oversample` samples per detector pixel on (2 ceil(R os) + 1)^2 samples centred at N // 2, unit sum,
    R = half window in detector px of that pair (Rtab). No node dependence (one array for every node)."""

    def __init__(self, sigma, oversample, Rtab):
        self.sigma, self.oversample, self.Rtab, self.notes = float(sigma), int(oversample), Rtab, []

    def get(self, detname, band, channel):
        R = self.Rtab[(detname, band)]
        N = 2 * int(math.ceil(R * self.oversample)) + 1
        ax = (np.arange(N) - N // 2) / self.oversample
        x, y = np.meshgrid(ax, ax)
        k = np.exp(-0.5 * (x * x + y * y) / self.sigma ** 2)
        return (k / k.sum())[None].astype(np.float32), None


_PMAT = {}


def spline_prefilter_matrix(N, dtype=torch.float32):
    """P = B^-1 with B the cubic B-spline interpolation matrix with mirror boundaries (scipy spline_filter order 3, mode 'mirror'): the
    coefficients of a 2-D table T are P T P^T (differentiable)."""
    key = (N, dtype)
    if key not in _PMAT:
        B = torch.zeros(N, N, dtype=torch.float64)
        i = torch.arange(N)
        B[i, i] = 2.0 / 3.0
        B[i[:-1], i[:-1] + 1] = 1.0 / 6.0
        B[i[1:], i[1:] - 1] = 1.0 / 6.0
        B[0, 1], B[N - 1, N - 2] = 1.0 / 3.0, 1.0 / 3.0
        _PMAT[key] = torch.linalg.inv(B).to(dtype)
    return _PMAT[key]


def psf_param_groups(psf, lr):
    """Optimiser groups of the PSF: one group at `lr` (v34/v35), plus the learnable library tables at --psf-lr or 1e-2 (v36)."""
    tp = list(psf.tab.values()) if isinstance(psf, LibPSF) and psf.learn else []
    ids = {id(p) for p in tp}
    g = [{"params": [p for p in psf.parameters() if p.requires_grad and id(p) not in ids], "lr": lr}]
    if tp:
        g.append({"params": [p for p in tp if p.requires_grad], "lr": args_global.psf_lr if args_global.psf_lr > 0 else 1e-2})
    return g


'''.replace("\\'\\'\\'", "'''")
rep("class LibPSF(nn.Module):", NEW_HELPERS + "class LibPSF(nn.Module):")

rep('''res_l2=0.0, sc=3, reg=None):
        super().__init__()''', '''res_l2=0.0, sc=3, reg=None, learn=False, res_on=True, tv=0.0):
        super().__init__()''')
rep('''        self.os, self.res_l2, self.scale_arcsec = lib.oversample, float(res_l2), scale_arcsec
''', '''        self.os, self.res_l2, self.scale_arcsec = lib.oversample, float(res_l2), scale_arcsec
        self.learn, self.res_on, self.tv = bool(learn), bool(res_on), float(tv)
''')
rep('''        for c in reg.channels:
            net = nn.Sequential(nn.Linear(din''', '''        for c in (reg.channels if res_on else []):
            net = nn.Sequential(nn.Linear(din''')
rep('''        self.det_emb = nn.Embedding(len(det_names), 4)
        self._L, self._C,''', '''        self.det_emb = nn.Embedding(len(det_names), 4)
        self.tab = nn.ParameterDict()  # v36: learnable tables, key "<detector index>_<band index>"
        self._L, self._C,''')
rep('''            self._eps[(d, b)] = [0.1 * float(Lt[i].max()) for i in range(Lt.shape[0])]
''', '''            self._eps[(d, b)] = [0.1 * float(Lt[i].max()) for i in range(Lt.shape[0])]
            if self.learn:  # v36: one table per (detector, band) from the middle-node table; theta = inverse softplus with a floor
                T0 = torch.from_numpy(np.ascontiguousarray(arr[idx[len(idx) // 2]])).double()
                T0 = T0 / T0.sum()
                y = T0.clamp(min=1e-8 * float(T0.max()))
                self.tab[f"{d}_{b}"] = nn.Parameter(torch.log(torch.expm1(y)).float())
''')
rep('''    def _read(self, d, b, q, u):''', '''    def _read(self, d, b, q, u, Cf=None):''')
rep('''        L = self._get_L(d, b)
        N = L.shape[-1]
        if self.interp == "grid_sample":
            h = (N - 1) / 2.0
            grid = torch.stack([(u[:, 0] * self.os + N // 2 - h) / h, (u[:, 1] * self.os + N // 2 - h) / h], -1)
            return F.grid_sample(L[q][None, None], grid''', '''        N = self._get_L(d, b).shape[-1]
        if self.interp == "grid_sample":
            h = (N - 1) / 2.0
            grid = torch.stack([(u[:, 0] * self.os + N // 2 - h) / h, (u[:, 1] * self.os + N // 2 - h) / h], -1)
            return F.grid_sample(self._tab_q(d, b, q)[None, None], grid''')
rep('''        C = self._get_C(d, b)[q].reshape(-1)
        px, py''', '''        C = Cf if Cf is not None else self._cf(d, b, q)
        px, py''')
rep('''    def _res(self, ch, d, lamf, u, R):''', '''    def _table(self, d, b):
        """Learnable unit-sum positive table (N, N)."""
        sp = F.softplus(self.tab[f"{d}_{b}"])
        return sp / sp.sum()

    def _tab_q(self, d, b, q):
        return self._table(d, b) if self.learn else self._get_L(d, b)[q]

    def _cf(self, d, b, q):
        """Flat cubic B-spline coefficients of the table of node q (learnable: differentiable P T P^T)."""
        if not self.learn:
            return self._get_C(d, b)[q].reshape(-1)
        T = self._table(d, b)
        P = spline_prefilter_matrix(T.shape[-1]).to(T.device)
        return (P @ T @ P.T).reshape(-1)

    def _res(self, ch, d, lamf, u, R):''')

between("    # ---- kernels\n", "    def basis(self):\n        \"\"\"(nb, 1, n, n): theta = 0", '''    # ---- kernels
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
            Cs = self._cf(d, b, 0) if self.learn else None  # one table for all nodes: coefficients once per group
            for q, lamf in zip(qs, lams):
                Cf = Cs if self.learn else self._cf(d, b, q)

                def row(lamf, Cf, u, q=q, b=b, eps=self._eps[(d, b)][q], ch=self.chan[b], d=d, R=R):
                    K = self._read(d, b, q, u, Cf)
                    if residual and self.res_on:
                        K = K + eps * self._res(ch, d, lamf, u, R)
                    K = K.view(S2, n * n).mean(0)
                    return (K / K.sum()).view(n, n)
                out.append(run_ckpt(row, lamf.reshape(()), Cf, u))
        return torch.stack(out)[:, None]

'''.replace("\\'\\'\\'", "'''"))

between("    def penalty(self):\n        \"\"\"Centroid gauge (first moment of K", "def make_libpsf(", '''    def penalty(self):
        """Centroid gauge of K relative to the table (first moment, detector px) times centre_weight, res_l2 times the mean squared residual,
        and (learnable tables) the first moment of every table about its centre times centre_weight plus tv times its total variation."""
        wc, wl, tv = float(getattr(self, "centre_weight", 0.0)), (self.res_l2 if self.res_on else 0.0), self.tv if self.learn else 0.0
        dev = self.det_emb.weight.device
        if not wc and not wl and not tv:
            return torch.zeros((), device=dev)
        tot = torch.zeros((), device=dev)
        for (d, b), L in self._L.items():
            L = self._get_L(d, b)
            if self.learn:
                T = self._table(d, b)
                N = T.shape[-1]
                ax = (torch.arange(N, device=dev, dtype=T.dtype) - N // 2) / self.os
                if wc:
                    tot = tot + wc * ((T.sum(0) * ax).sum().pow(2) + (T.sum(1) * ax).sum().pow(2))
                if tv:
                    Tn = T / T.max()
                    tot = tot + tv * ((Tn[:, 1:] - Tn[:, :-1]).abs().mean() + (Tn[1:] - Tn[:-1]).abs().mean())
            if not self.res_on or not (wc or wl):
                continue
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
        """K(u) (learned, zero outside the square window |u| <= half_window_det_px, unit sum), L(u) (the STARTING table / library, whole
        lattice, unit sum) and, with learnable tables, 'table' (the learned table T, unit sum), on the oversampled lattice at the middle node."""
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
            Ls = L[q].reshape(-1)  # the lattice itself: the interpolant equals the samples there
            Lv = self._table(d, b).reshape(-1) if self.learn else Ls
            if self.res_on:
                K = torch.cat([Lv[i:i + 65536] + self._eps[(d, b)][q] * self._res(self.chan[b], d, lamf, u[i:i + 65536], R) for i in range(0, u.shape[0], 65536)])
            else:
                K = Lv
            K = K * ((u.abs() <= R).all(1)).float()  # the residual is only trained inside the window (|u| <= R, square footprint): zero outside
            ent = {"det": self.det_names[d], "band": self.bands[b], "node_index": q, "lam_um": self.lam_q[b][q], "oversample": self.os,
                   "half_window_det_px": R, "library": (Ls / Ls.sum()).view(N, N).cpu().numpy(), "learned": (K / K.sum()).view(N, N).cpu().numpy()}
            if self.learn:
                ent["table"] = Lv.view(N, N).cpu().numpy()
            out[f"{self.det_names[d]}/{self.bands[b]}"] = ent
        return out


'''.replace("\\'\\'\\'", "'''"))

between("def make_libpsf(", "def deconv_init(", '''def make_libpsf(a, dets, exps, band_index, lam_nodes, dev):
    """LibPSF for the (detector, band) pairs present in the loaded exposures; the library is read once per run (--lib-source file) or
    built in memory as generic Gaussian tables (--lib-source gauss)."""
    src_ = getattr(a, "lib_source", "file")
    if src_ == "gauss":
        Rtab = {}
        for g_ in PSFG.groups:
            k_ = (dets[int(g_["det"])], REG.bands[g_["b"]])
            if k_ not in Rtab:
                Rtab[k_] = (a.psf_n // 2) / math.sqrt(abs(g_["M"][0][0] * g_["M"][1][1] - g_["M"][0][1] * g_["M"][1][0]))
        lib, key = GaussLibrary(a.lib_gauss_sigma, a.lib_table_os, Rtab), "gauss"
    else:
        key = str(a.psf_lib)
        if key not in LIB_STORE:
            LIB_STORE[key] = PSFLibrary(a.psf_lib)
        lib = LIB_STORE[key]
    pairs = {(int(e["det"]), band_index[e["band"]]) for e in exps}
    psf = LibPSF(REG.bands, dets, pairs, lib, lam_nodes.cpu() if lam_nodes is not None else None, a.scale, n=a.psf_n, hidden=a.lib_res_hidden, res_l2=a.lib_res_l2,
                 learn=getattr(a, "lib_learn_table", False), res_on=getattr(a, "lib_res", "on") == "on", tv=getattr(a, "lib_table_tv", 0.0))
    if a.psf_centre > 0:
        psf.centre_weight = a.psf_centre
    psf = psf.to(dev)
    for g_ in PSFG.groups:  # window scale (detector px) of every (detector, band) pair from its first group
        if (int(g_["det"]), g_["b"]) in psf._L:
            psf._R(g_, int(g_["det"]), g_["b"])
    for m_ in lib.notes + psf.messages:
        tick(m_)
    lib.notes = []
    if src_ == "gauss":
        tick(f"data-driven arm: generic Gaussian tables sigma {a.lib_gauss_sigma} detector px, oversampling {a.lib_table_os}, no physics library; "
             f"learnable tables: {psf.learn} (APPROXIMATION when learnable: one achromatic table per (detector, band)), residual networks: {psf.res_on}")
    LOG["lib_psf"] = {"dir": key, "source": src_, "oversample": lib.oversample, "pairs": [[dets[d], REG.bands[b]] for d, b in sorted(pairs)], "window_px": a.psf_n,
                      "residual_hidden": a.lib_res_hidden, "res_l2": a.lib_res_l2, "subpositions_per_cell": psf.sc ** 2, "messages": psf.messages,
                      "learn_table": psf.learn, "residual_networks": psf.res_on, "table_tv": psf.tv, "gauss_sigma_det_px": getattr(a, "lib_gauss_sigma", None),
                      "run_nodes_um": {REG.bands[b]: v for b, v in psf.lam_q.items()}}
    return psf


'''.replace("\\'\\'\\'", "'''"))

rep('''              {"params": [p for p in model.psf.parameters() if p.requires_grad], "lr": lrs["psf"]},
''', '''              *psf_param_groups(model.psf, lrs["psf"]),
''')
open(dst, "w").write(s)
print("ok", dst)
