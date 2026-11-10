"""Build smacs_four_v21.py from v20: Q1 zip-field capacity control and Q2 deterministic renderer (4 Nov 2026)."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new, after=None):
    global s
    if after is None:
        assert s.count(old) == 1, (old[:70], s.count(old))
        s = s.replace(old, new)
    else:
        i = s.index(after)
        head, tail = s[:i], s[i:]
        assert old in tail, old[:70]
        s = head + tail.replace(old, new, 1)


# ---------------------------------------------------------------- ZipSkyModel: keep the PSF, deterministic renderer
rep("""        I = I.clamp(min=0)
        no = I.shape[-1]
""", """        I = I.clamp(min=0)
        no = I.shape[-1]
        self.I = I  # (nbQ, D, no, no), kept for the deterministic renderer (v21)
""")

rep("""    def offsets(self, b, det, N, n, gen=None):""", '''    render_mode, render_os, _R, _Kf = "mc", 2, None, None

    @torch.no_grad()
    def build_kernels(self, os_r, band_dets):
        """Deterministic renderer (v21). The same physical PSF as the sampler (detector response included, energy in
        the window kept), binned from the sampler's oversampling P.os to os_r points per target pixel, bins centred on
        offsets k/os_r. The outermost ring of the fine grid that does not fill a whole bin is dropped (reported).
        Kernel FFTs are cached on the padded render grid for the (band, detector) pairs that occur."""
        f = self.os // os_r
        assert f * os_r == self.os and f % 2 == 0 and self.no % 2 == 0, \
            "bins centred on k/os_r need an even number of fine pixels per bin (even PSF grid)"
        no = self.no
        d = (torch.arange(no, device=self.I.device, dtype=torch.float64) - (no - 1) / 2) / self.os  # fine offsets, target px
        k = torch.floor(d * os_r + 0.5).long()
        kmax = (no // f - 1) // 2
        M = torch.zeros(2 * kmax + 1, no, device=self.I.device, dtype=self.I.dtype)
        ok = k.abs() <= kmax
        M[k[ok] + kmax, torch.nonzero(ok)[:, 0]] = 1.0
        Kr = torch.einsum("ki,ndij,lj->ndkl", M, self.I, M)  # nbQ D nk nk
        self.render_os, self.kmax = os_r, kmax
        self.pad_t = int(math.ceil(kmax / os_r))  # render margin in target px (no circular wrap in the inner region)
        H = self.sky.size
        self.Ne = (H + 2 * self.pad_t) * os_r
        nk = 2 * kmax + 1
        Q = self.lam_nodes.shape[1]
        self._Kf, lost = {}, []
        for (b, det) in sorted(band_dets):
            for q in range(Q):
                K = torch.zeros(self.Ne, self.Ne, device=self.I.device)
                K[:nk, :nk] = Kr[b * Q + q, det]
                K = torch.roll(K, shifts=(-kmax, -kmax), dims=(0, 1))  # offset 0 at index 0
                self._Kf[(b, det, q)] = torch.fft.rfft2(K)
                lost.append(float(1 - Kr[b * Q + q, det].sum() / self.I[b * Q + q, det].sum().clamp(min=1e-30)))
        self.band_dets = sorted(band_dets)
        LOG["zip_render"] = {"os": os_r, "kernel_px": nk, "margin_target_px": self.pad_t, "grid": self.Ne,
                             "energy_dropped_edge_ring_max": max(lost),
                             "note": "field evaluated on the oversampled grid, FFT-convolved, then bilinear-sampled at detector positions"}
        tick(f"deterministic renderer: {nk}^2 kernels at {os_r}x, render grid {self.Ne}^2, "
             f"edge-ring energy dropped <= {max(lost):.2e}")

    def render(self):
        """Band images through the PSF, per (band, detector), on the os_r-oversampled target grid (inner region)."""
        os_r, H, pad = self.render_os, self.sky.size, self.pad_t
        u = (torch.arange(self.Ne, device=self.lam_nodes.device, dtype=torch.float32) + 0.5) / os_r - 0.5 - pad
        yy, xx = torch.meshgrid(u, u, indexing="ij")
        A = self.sky(xx.flatten(), yy.flatten()).T.reshape(-1, self.Ne, self.Ne)  # K Ne Ne
        Q = self.lam_nodes.shape[1]
        out = {}
        for b in sorted({b for b, _ in self.band_dets}):
            Fq = torch.fft.rfft2(torch.einsum("qk,khw->qhw", self.phi[b], A))
            for (bb, det) in self.band_dets:
                if bb != b:
                    continue
                acc = sum(self.w_nodes[b, q] * Fq[q] * self._Kf[(b, det, q)] for q in range(Q))
                R = torch.fft.irfft2(acc, s=(self.Ne, self.Ne))
                out[(b, det)] = R[pad * os_r:(pad + H) * os_r, pad * os_r:(pad + H) * os_r]
        return out

    def offsets(self, b, det, N, n, gen=None):''')

rep("""    def core(self, b, det, x, y, n=8, gen=None, chunk=200000):
        det = int(det)
""", """    def core(self, b, det, x, y, n=8, gen=None, chunk=200000):
        det = int(det)
        if self.render_mode == "det":
            if self._R is None:  # evaluation: render once without gradients, cached until the next training step
                with torch.no_grad():
                    self._R = self.render()
            os_r = self.render_os
            return sample(self._R[(b, det)], (x + 0.5) * os_r - 0.5, (y + 0.5) * os_r - 0.5)
""")

# ---------------------------------------------------------------- fit_zip: one render per step in det mode
rep("""    while True:
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for i in train_idx:
            e = exps[i]
            N = e["d"].numel()
            idx = torch.randperm(N, device=e["d"].device, generator=gen)[: max(1, int(frac * N))]
            m = model.predict_zip(i, e, idx, nsamp, gen)
            li = rho(e["ivar"][idx] * (e["d"][idx] - m) ** 2).sum() / (frac * ntot)
            li.backward()
            loss += float(li.detach())
        reg = model.penalty_at()""", """    det_mode = model.render_mode == "det"
    while True:
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        if det_mode:  # render once with gradients; per-exposure losses back-propagate into detached leaves
            R_full = model.render()
            model._R = {k: v.detach().requires_grad_(True) for k, v in R_full.items()}
        for i in train_idx:
            e = exps[i]
            N = e["d"].numel()
            idx = torch.randperm(N, device=e["d"].device, generator=gen)[: max(1, int(frac * N))]
            m = model.predict_zip(i, e, idx, nsamp, gen)
            li = rho(e["ivar"][idx] * (e["d"][idx] - m) ** 2).sum() / (frac * ntot)
            li.backward()
            loss += float(li.detach())
        if det_mode:
            keys = [k for k in R_full if model._R[k].grad is not None]
            torch.autograd.backward([R_full[k] for k in keys], [model._R[k].grad for k in keys])
            model._R, R_full = None, None
        reg = model.penalty_at()""")
rep("""    if best is not None:
        model.load_state_dict""", """    model._R = None
    if best is not None:
        model.load_state_dict""", after="def fit_zip(")

# ---------------------------------------------------------------- Q1 capacity control
rep("""def sky_fisher(m, exps, train_idx, nz=24):""", '''def zip_capacity(a, init, sig_b, src_masks, lam_nodes, w_nodes, dev, out):
    """Q1 capacity control (v21): fit the zip-field directly to target spectral-coefficient grids, with no
    instrument, no PSF and no exposures. It measures whether the representation can hold each target at the grid
    nodes; it says nothing about deconvolution or about values between nodes. Objective = the v20 warm start's
    (coefficient mse / mean(sig_b)^2). Scored in band space against each target's band images: mse/sig_b^2 on all
    pixels and on the S/N > 5 pixels of the SWarp-like coadd (the mask of the held-out 'src' score)."""
    basis = RBFSpectralBasis(a.spec_k, learned=False).to(dev)
    targets = {"coadd_" + a.init: spectral_init(init, lam_nodes, w_nodes, basis).detach()}
    if a.capacity_target_state:
        st = torch.load(a.capacity_target_state, map_location=dev)
        cand = [(k, v) for k, v in st.items() if k.startswith("sky.") and v.dim() == 3 and v.shape[0] == a.spec_k
                and v.shape[-1] == a.size]
        assert len(cand) == 1, [k for k, _ in cand]
        targets["grid3d"] = cand[0][1].float().detach()
        LOG["capacity_target_grid3d"] = {"file": a.capacity_target_state, "key": cand[0][0]}
    phi = basis(lam_nodes.flatten()).view(*lam_nodes.shape, -1)  # nb Q K

    def bands(A):
        return torch.einsum("bqk,khw->bhw", phi * w_nodes[..., None], A)

    sig = sig_b[:, None, None]
    H = a.size
    ar = torch.arange(H, device=dev, dtype=torch.float32)
    yy, xx = torch.meshgrid(ar, ar, indexing="ij")
    res = {}
    for cfg in a.capacity_configs.split(";"):
        cname, feat, hidden, levels = cfg.split(":")
        for tname, A0 in targets.items():
            torch.manual_seed(0)
            field = ZipField(H, a.spec_k, levels=tuple(int(v) for v in levels.split(",")), feat=int(feat),
                             hidden=int(hidden), scale=float(sig_b.mean()) * 20).to(dev)
            npar = sum(p_.numel() for p_ in field.parameters())
            opt = torch.optim.Adam(field.parameters(), lr=a.capacity_lr)
            with torch.no_grad():
                S0 = bands(A0)
            t0, it, hist, budget = time.perf_counter(), 0, [], a.capacity_min * 60
            while True:
                el = time.perf_counter() - t0
                ft = el / budget
                for g in opt.param_groups:  # warm-up 3 %, cosine to 1 %
                    g["lr"] = a.capacity_lr * (ft / 0.03 if ft < 0.03 else
                                               0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (ft - 0.03) / 0.97))))
                opt.zero_grad(set_to_none=True)
                A = field(xx.flatten(), yy.flatten()).T.reshape(-1, H, H)
                l = ((A - A0) / sig_b.mean()).pow(2).mean()
                l.backward()
                opt.step()
                it += 1
                done = el > budget
                if it % 500 == 0 or done or it == 800:
                    with torch.no_grad():
                        r2 = ((bands(A) - S0) / sig).pow(2)
                        row = [it, round(el, 1), float(l), float(r2.mean()), float(r2[src_masks].mean())]
                    hist.append(row)
                    tick(f"CAPACITY {cname}/{tname} step {it} ({el / 60:.1f} min): coeff mse/sig2 {row[2]:.4f} | "
                         f"band mse/sig2 all {row[3]:.4f} bright {row[4]:.4f}")
                if done:
                    break
            with torch.no_grad():
                A = field(xx.flatten(), yy.flatten()).T.reshape(-1, H, H)
                rb = (bands(A) - S0) / sig
                per_band = {BANDS[j]: {"all": float(rb[j].pow(2).mean()), "bright": float(rb[j][src_masks[j]].pow(2).mean()),
                                       "bright_max_abs": float(rb[j][src_masks[j]].abs().max())} for j in range(len(BANDS))}
                np.save(out / f"capacity_{cname}_{tname}_resid.npy", rb.cpu().numpy())
            res[f"{cname}/{tname}"] = {"params": npar, "steps": it, "minutes": a.capacity_min, "config": cfg,
                                       "coeff_mse_sig2": hist[-1][2], "band_mse_sig2_all": hist[-1][3],
                                       "band_mse_sig2_bright": hist[-1][4], "per_band": per_band, "history": hist}
            LOG["capacity"] = res
            json.dump(LOG, open(out / "metrics_partial.json", "w"), indent=1, default=float)
            del field, opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return res


def sky_fisher(m, exps, train_idx, nz=24):''')

# ---------------------------------------------------------------- run(): capacity mode, renderer, init state, eval only
rep("""    if getattr(a, "zip", False):
        from pupil_psf import PupilPSF""", """    if getattr(a, "zip_capacity", False):
        zip_capacity(a, init, sig_b, src_masks, lam_nodes, w_nodes, dev, out)
        json.dump(LOG, open(out / "metrics.json", "w"), indent=1, default=float)
        tick("done (capacity control only)")
        return

    if getattr(a, "zip", False):
        from pupil_psf import PupilPSF""")
rep("""        zfield = ZipField(a.size, a.spec_k, scale=float(sig_b.mean()) * 20).to(dev)""",
    """        zfield = ZipField(a.size, a.spec_k, levels=tuple(int(v) for v in a.zip_levels.split(",")), feat=a.zip_feat,
                          hidden=a.zip_hidden, scale=float(sig_b.mean()) * 20).to(dev)""")
rep("""        A0 = spectral_init(init, lam_nodes, w_nodes, zbasis)
        optw = torch.optim.Adam(m.sky.parameters(), lr=5e-3)
        for it_w in range(20 if a.quick else 800):
            optw.zero_grad()
            lw = ((m.grid_coeffs() - A0) / sig_b.mean()).pow(2).mean()
            lw.backward()
            optw.step()
        tick(f"{zname}: field warm start on the drizzle spectral coefficients, mse/sig2 {float(lw):.3f}")
        fit_zip(""", """        if a.zip_render == "det":
            m.build_kernels(a.zip_render_os, {(band_index[e["band"]], int(e["det"])) for e in exps})
            m.render_mode = "det"
        LOG[f"{zname}_render"] = m.render_mode
        if a.zip_init_state:
            st0 = torch.load(a.zip_init_state, map_location=dev)
            m.sky.load_state_dict({k[4:]: v for k, v in st0.items() if k.startswith("sky.")})  # strict
            LOG[f"{zname}_init_state"] = a.zip_init_state
            tick(f"{zname}: field weights loaded from {a.zip_init_state} (no warm start)")
        else:
            A0 = spectral_init(init, lam_nodes, w_nodes, zbasis)
            optw = torch.optim.Adam(m.sky.parameters(), lr=5e-3)
            for it_w in range(20 if a.quick else a.zip_warm_steps):
                optw.zero_grad()
                lw = ((m.grid_coeffs() - A0) / sig_b.mean()).pow(2).mean()
                lw.backward()
                optw.step()
            LOG[f"{zname}_warm_mse_sig2"] = float(lw)
            tick(f"{zname}: field warm start on the {a.init} coadd's spectral coefficients ({a.zip_warm_steps} steps), mse/sig2 {float(lw):.3f}")
        if a.zip_eval_only:
            tick(f"{zname}: evaluation only, no fit")
        else:
            fit_zip(""")
rep("""                a.eval_every_min * 60, a.patience, frac=a.zip_frac, nsamp=a.zip_samples)
        LOG["timings"][zname]""", """                    a.eval_every_min * 60, a.patience, frac=a.zip_frac, nsamp=a.zip_samples)
        LOG["timings"][zname]""")

# ---------------------------------------------------------------- arguments
rep("""    ap.add_argument("--zip-frac", type=float, default=0.125)""", """    ap.add_argument("--zip-frac", type=float, default=0.125)
    ap.add_argument("--zip-render", choices=("mc", "det"), default="mc", help="mc: Monte Carlo PSF samples per pixel (v20); det: field on an oversampled grid, FFT-convolved with the same PSF, bilinear at detector positions")
    ap.add_argument("--zip-render-os", type=int, default=2, help="render points per target pixel for --zip-render det")
    ap.add_argument("--zip-init-state", default=None, help="load the field weights (sky.*) from this zip state file instead of the warm start")
    ap.add_argument("--zip-eval-only", action="store_true", help="score the zip model without fitting")
    ap.add_argument("--zip-warm-steps", type=int, default=800)
    ap.add_argument("--zip-levels", default="16,32,64,128,256,512,1024")
    ap.add_argument("--zip-feat", type=int, default=4)
    ap.add_argument("--zip-hidden", type=int, default=96)
    ap.add_argument("--zip-capacity", action="store_true", help="Q1: fit the field directly to target coefficient grids (no instrument), report, exit")
    ap.add_argument("--capacity-target-state", default=None, help="explicit 3D state file whose grid sky is a second capacity target")
    ap.add_argument("--capacity-configs", default="base:4:96:16,32,64,128,256,512,1024;big:8:256:16,32,64,128,256,512,1024,2048",
                    help="name:feat:hidden:levels, separated by ;")
    ap.add_argument("--capacity-min", type=float, default=12.0, help="minutes per (config, target)")
    ap.add_argument("--capacity-lr", type=float, default=5e-3)""")

open(dst, "w").write(s)
print("wrote", dst)
