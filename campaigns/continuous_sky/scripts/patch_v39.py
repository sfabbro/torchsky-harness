"""v38 -> v39 (7 Nov 2026): within-band chromatic learnable library-PSF tables.
Without the new flag v39 is bit-identical to v38 (tests/check_v39_equiv.py).

WHY. v36's learnable tables (--psf-model lib --explicit-psf lib --lib-learn-table) are ONE table per (detector, band) shared by every quadrature node of the
band, i.e. achromatic within a band. On the real data the NIRCam LW bands were clearly worse than with ChromaticPSF (which evaluates its kernel at every
node), while the SW bands were on par; the LW PSF width changes by about 25 per cent across F444W.

--lib-chromatic {none,endpoints,nodes}   (default none = v38; needs --lib-learn-table)
  nodes      one learnable table per (detector, band, node q = 0..Q-1), all initialised from the SAME start table (the middle-node table of the
             library, or the Gaussian of --lib-source gauss; as v36). Plus a smoothness penalty across adjacent nodes (--lib-chrom-smooth).
  endpoints  two learnable tables per (detector, band): at the first (q = 0) and the last (q = Q-1) node. The inner nodes use the table
             T_q = (1 - w_q) T_0 + w_q T_{Q-1},  w_q = (lambda_q - lambda_0) / (lambda_{Q-1} - lambda_0)   (linear in lambda, lambda_q = the
             run's quadrature node wavelengths in um). Both free tables start from the same start table, so the starting model is achromatic.
             Convex combination of positive unit-sum tables: positive, unit sum. No smoothness penalty (interpolation is smooth by construction).
  Q = 1: one table (all modes the same as v38 for that pair; only the parameter name differs).
--lib-chrom-smooth W (default 1.0): penalty added to the model penalty (summed over (detector, band) pairs), nodes only:
     W * mean_q  || T_{q+1} - T_q ||_2^2 / ( 0.5 ( ||T_q||_2^2 + ||T_{q+1}||_2^2 ) )
  with T_q the unit-sum (positive) table of node q on its oversampled lattice. It is dimensionless (relative to the table norm), zero for equal
  tables, and for a pure width change of a few per cent per node of order 1e-3 (so W = 1 hardly resists genuine chromaticity; it suppresses node-to-node noise).
  It is an L2 smoothness in the NODE direction only; there is no smoothness prior in the spatial directions except --lib-table-tv.

APPROXIMATIONS AND LIMITS (all stated in the log and metrics.json["lib_psf"]):
  * NO physics: no lambda/D scaling is imposed; lambda enters only through which table / interpolation weight a node gets. The starting model is
    achromatic (every table equals the same start), the chromaticity is learned from the data only. With --lib-source file the per-node library
    tables are NOT used as the start (v36 behaviour: the middle node); they still define the residual-network amplitudes eps_q.
  * Gauges per table as in v36 (unit sum by construction; centroid: --psf-centre times the squared first moment of every FREE table about its
    centre; --lib-table-tv on every free table). With endpoints the gauge acts on the two free tables; the centroid is linear in T, so it then
    holds exactly for the interpolated tables too (TV is not exactly linear and is applied to the free tables only).
  * Interpolation weights are linear in lambda between the first and last quadrature node, not in 1/lambda or in a width; a PSF whose profile
    changes non-linearly across the band is represented only to first order by endpoints.
  * The learned per-node tables are the tables only; the (optional) residual networks are evaluated at the node wavelength on top, as before.
  * WFC3/IR tables are chromatic in the same way (they have Q nodes too); their start is the same ePSF for every node.
  * --freeze-instruments-from does not freeze chromatic tables (stops with a message if combined).
  * Cost: the B-spline coefficients P T P^T of a table are computed once per (detector, band, node) per forward call (not once per PSF group).
  * Memory: see the log line 'chromatic tables: ...' printed at start (parameter count and bytes of tables incl. gradient and two Adam moments).

OUTPUT: <model>_psf_groups.npy["detector_frame"]["<det>/<band>"] gets, with the flag, 'chromatic' (mode), 'lam_nodes_um', 'tables_nodes' (Q, N, N)
  float32, the unit-sum learned table of EVERY node (for endpoints the interpolated ones) and 'free_table_nodes' (node indices that are free
  parameters), in addition to the existing keys ('table', 'learned', 'library' at the middle node). Encircled energy versus lambda can be read from these.
"""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:90])
    s = s.replace(old, new)


# ===================================================================================== docstring
rep('''All off by default: bit-identical to v37. See patch_v38.py for every approximation.
"""''', '''All off by default: bit-identical to v37. See patch_v38.py for every approximation.

v39: --lib-chromatic {none,endpoints,nodes} (+ --lib-chrom-smooth): within-band chromatic learnable library-PSF tables (one per node, or two
endpoint tables interpolated linearly in lambda). Default none: bit-identical to v38. No lambda/D physics imposed. See patch_v39.py.
"""''')

# ===================================================================================== arguments and checks
rep('''    a = ap.parse_args()
    if a.seed is not None:''', '''    ap.add_argument("--lib-chromatic", choices=("none", "endpoints", "nodes"), default="none",
                    help="v39: within-band chromatic learnable tables (needs --lib-learn-table): none = one table per (detector, band) (v38); nodes = one per quadrature node; "
                         "endpoints = first and last node, linear interpolation in lambda for the inner nodes. No lambda/D physics imposed")
    ap.add_argument("--lib-chrom-smooth", type=float, default=1.0,
                    help="v39: (nodes only) weight of the node-to-node smoothness penalty mean_q ||T_q+1 - T_q||^2 / (0.5 (||T_q||^2 + ||T_q+1||^2)) of the unit-sum tables")
    a = ap.parse_args()
    if a.seed is not None:''')

rep('''        raise SystemExit("--lib-source gauss with --lib-res off and no --lib-learn-table has no free PSF parameters")
''', '''        raise SystemExit("--lib-source gauss with --lib-res off and no --lib-learn-table has no free PSF parameters")
    if getattr(a, "lib_chromatic", "none") != "none":
        if not LIB_ON or not getattr(a, "lib_learn_table", False):
            raise SystemExit("--lib-chromatic needs --psf-model lib --explicit-psf lib --lib-learn-table")
        if getattr(a, "freeze_instruments_from", None):
            raise SystemExit("--lib-chromatic is not supported together with --freeze-instruments-from (chromatic tables are not frozen)")
        if getattr(a, "lib_chrom_smooth", 1.0) < 0:
            raise SystemExit("--lib-chrom-smooth must be >= 0")
''')

# ===================================================================================== optimiser groups
rep('''    tp = list(psf.tab.values()) if isinstance(psf, LibPSF) and psf.learn else []
''', '''    tp = (list(psf.tab.values()) + list(psf.tabq.values())) if isinstance(psf, LibPSF) and psf.learn else []
''')

# ===================================================================================== LibPSF constructor
rep('''res_l2=0.0, sc=3, reg=None, learn=False, res_on=True, tv=0.0):
        super().__init__()''', '''res_l2=0.0, sc=3, reg=None, learn=False, res_on=True, tv=0.0,
                 chrom="none", chrom_smooth=1.0):
        super().__init__()''')
rep('''        self.learn, self.res_on, self.tv = bool(learn), bool(res_on), float(tv)
        self.det_names, self.bands''', '''        self.learn, self.res_on, self.tv = bool(learn), bool(res_on), float(tv)
        self.chrom, self.chrom_smooth = (chrom if learn else "none"), float(chrom_smooth)  # v39
        self.det_names, self.bands''')
rep('''        self.tab = nn.ParameterDict()  # v36: learnable tables, key "<detector index>_<band index>"
''', '''        self.tab = nn.ParameterDict()  # v36: learnable tables, key "<detector index>_<band index>"
        self.tabq = nn.ParameterDict()  # v39: chromatic learnable tables, key "<detector index>_<band index>_<node index>" (empty without --lib-chromatic)
''')
rep('''                self.tab[f"{d}_{b}"] = nn.Parameter(torch.log(torch.expm1(y)).float())
''', '''                th0 = torch.log(torch.expm1(y)).float()
                if self.chrom == "none":
                    self.tab[f"{d}_{b}"] = nn.Parameter(th0)
                else:  # v39: every free table starts from the same table (achromatic start; NO physics)
                    Qn = len(idx)
                    free = list(range(Qn)) if self.chrom == "nodes" else sorted({0, Qn - 1})
                    for qq in free:
                        self.tabq[f"{d}_{b}_{qq}"] = nn.Parameter(th0.clone())
''')

# ===================================================================================== tables
rep('''    def _table(self, d, b):
        """Learnable unit-sum positive table (N, N)."""
        sp = F.softplus(self.tab[f"{d}_{b}"])
        return sp / sp.sum()

    def _tab_q(self, d, b, q):
        return self._table(d, b) if self.learn else self._get_L(d, b)[q]
''', '''    def _unit(self, key):
        sp = F.softplus(self.tabq[key])
        return sp / sp.sum()

    def _wint(self, b, q):
        """v39 (endpoints): linear interpolation weight of the last-node table at node q (linear in lambda; 0 at the first, exactly 1 at the last node)."""
        lam = self.lam_q[b]
        Q = len(lam)
        if q <= 0 or Q < 2 or lam[-1] <= lam[0]:
            return 0.0
        if q >= Q - 1:
            return 1.0
        return (float(lam[q]) - float(lam[0])) / (float(lam[-1]) - float(lam[0]))

    def _table(self, d, b, q=0):
        """Learnable unit-sum positive table (N, N). v39: with --lib-chromatic the table of node q (nodes: its own; endpoints: the linear
        interpolation of the first- and last-node tables); without it q is ignored (one table per (detector, band))."""
        if self.chrom == "none":
            sp = F.softplus(self.tab[f"{d}_{b}"])
            return sp / sp.sum()
        Q = self._L[(d, b)].shape[0]
        if self.chrom == "nodes":
            return self._unit(f"{d}_{b}_{q}")
        w = self._wint(b, q)
        if Q < 2 or w == 0.0:
            return self._unit(f"{d}_{b}_0")
        if w == 1.0:
            return self._unit(f"{d}_{b}_{Q - 1}")
        return (1.0 - w) * self._unit(f"{d}_{b}_0") + w * self._unit(f"{d}_{b}_{Q - 1}")

    def _tables_free(self, d, b):
        """The free (learned) tables of the pair, for the gauges / penalties."""
        if self.chrom == "none":
            return [self._table(d, b)]
        Q = self._L[(d, b)].shape[0]
        qs = range(Q) if self.chrom == "nodes" else sorted({0, Q - 1})
        return [self._unit(f"{d}_{b}_{q}") for q in qs]

    def _tab_q(self, d, b, q):
        return self._table(d, b, q) if self.learn else self._get_L(d, b)[q]
''')
rep('''        T = self._table(d, b)
        P = spline_prefilter_matrix(T.shape[-1]).to(T.device)
        return (P @ T @ P.T).reshape(-1)
''', '''        T = self._table(d, b, q)
        P = spline_prefilter_matrix(T.shape[-1]).to(T.device)
        return (P @ T @ P.T).reshape(-1)
''')

# ===================================================================================== kernels
rep('''        dev = self.det_emb.weight.device
        n, S2 = self.n, self.sc ** 2
        out = []
        for g in groups:''', '''        dev = self.det_emb.weight.device
        n, S2 = self.n, self.sc ** 2
        out = []
        ccache = {}  # v39: chromatic tables: spline coefficients once per (detector, band, node) per call, shared by the groups
        for g in groups:''')
rep('''            Cs = self._cf(d, b, 0) if self.learn else None  # one table for all nodes: coefficients once per group
            for q, lamf in zip(qs, lams):
                Cf = Cs if self.learn else self._cf(d, b, q)
''', '''            chromq = self.learn and self.chrom != "none"
            Cs = self._cf(d, b, 0) if (self.learn and not chromq) else None  # one table for all nodes: coefficients once per group
            for q, lamf in zip(qs, lams):
                if chromq:
                    if (d, b, q) not in ccache:
                        ccache[(d, b, q)] = self._cf(d, b, q)
                    Cf = ccache[(d, b, q)]
                else:
                    Cf = Cs if self.learn else self._cf(d, b, q)
''')

# ===================================================================================== penalty
rep('''        if not wc and not wl and not tv:
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
''', '''        sm = self.chrom_smooth if (self.learn and self.chrom == "nodes") else 0.0  # v39
        if not wc and not wl and not tv and not sm:
            return torch.zeros((), device=dev)
        tot = torch.zeros((), device=dev)
        for (d, b), L in self._L.items():
            L = self._get_L(d, b)
            if self.learn:
                Ts = self._tables_free(d, b)
                for T in Ts:
                    N = T.shape[-1]
                    ax = (torch.arange(N, device=dev, dtype=T.dtype) - N // 2) / self.os
                    if wc:
                        tot = tot + wc * ((T.sum(0) * ax).sum().pow(2) + (T.sum(1) * ax).sum().pow(2))
                    if tv:
                        Tn = T / T.max()
                        tot = tot + tv * ((Tn[:, 1:] - Tn[:, :-1]).abs().mean() + (Tn[1:] - Tn[:-1]).abs().mean())
                if sm and len(Ts) > 1:  # v39: node-to-node smoothness relative to the table norm (nodes only)
                    terms = [(Ts[i + 1] - Ts[i]).pow(2).sum() / (0.5 * (Ts[i].pow(2).sum() + Ts[i + 1].pow(2).sum())) for i in range(len(Ts) - 1)]
                    tot = tot + sm * torch.stack(terms).mean()
''')

# ===================================================================================== dump
rep('''            Lv = self._table(d, b).reshape(-1) if self.learn else Ls
''', '''            Lv = self._table(d, b, q).reshape(-1) if self.learn else Ls
''')
rep('''                ent["table"] = Lv.view(N, N).cpu().numpy()
            out[f"{self.det_names[d]}/{self.bands[b]}"] = ent''', '''                ent["table"] = Lv.view(N, N).cpu().numpy()
                if self.chrom != "none":  # v39: the learned table of every node (endpoints: the interpolated ones), for EE versus lambda
                    Qn = L.shape[0]
                    ent["chromatic"] = self.chrom
                    ent["lam_nodes_um"] = [float(x) for x in self.lam_q[b]]
                    ent["tables_nodes"] = np.stack([self._table(d, b, qq).detach().cpu().numpy().astype(np.float32) for qq in range(Qn)])
                    ent["free_table_nodes"] = list(range(Qn)) if self.chrom == "nodes" else sorted({0, Qn - 1})
            out[f"{self.det_names[d]}/{self.bands[b]}"] = ent''')

# ===================================================================================== construction and log
rep('''learn=getattr(a, "lib_learn_table", False), res_on=getattr(a, "lib_res", "on") == "on", tv=getattr(a, "lib_table_tv", 0.0))
''', '''learn=getattr(a, "lib_learn_table", False), res_on=getattr(a, "lib_res", "on") == "on", tv=getattr(a, "lib_table_tv", 0.0),
                 chrom=getattr(a, "lib_chromatic", "none"), chrom_smooth=getattr(a, "lib_chrom_smooth", 1.0))
''')
rep('''                      "run_nodes_um": {REG.bands[b]: v for b, v in psf.lam_q.items()}}
    return psf''', '''                      "run_nodes_um": {REG.bands[b]: v for b, v in psf.lam_q.items()}}
    if psf.chrom != "none":  # v39
        npar = sum(p.numel() for p in psf.tabq.values())
        npar_ach = sum(int(psf._L[k].shape[-1]) ** 2 for k in psf._L)
        LOG["lib_psf"].update({"chromatic": psf.chrom, "chrom_smooth": psf.chrom_smooth if psf.chrom == "nodes" else None,
                               "n_table_params": npar, "n_table_params_achromatic_equivalent": npar_ach,
                               "table_bytes_with_grad_and_adam": 16 * npar, "n_free_tables": len(psf.tabq)})
        tick(f"chromatic tables ({psf.chrom}): {len(psf.tabq)} free tables, {npar} table parameters ({npar / 1e6:.2f} M; the achromatic v38 tables have {npar_ach}), "
             f"{16 * npar / 2 ** 30:.3f} GiB incl. gradient and two Adam moments (fp32); smoothness weight {psf.chrom_smooth if psf.chrom == 'nodes' else 'n/a'}. "
             f"APPROXIMATION: no lambda/D physics, every free table starts from the same table (achromatic start); "
             f"{'inner nodes linearly interpolated in lambda between the first- and last-node tables' if psf.chrom == 'endpoints' else 'node-to-node L2 smoothness relative to the table norm'}")
    return psf''')

open(dst, "w").write(s)
print("ok")
