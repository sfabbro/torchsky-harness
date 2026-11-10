"""Tests for scripts/smacs_four_v32.py --sky-coeffs splat2. Run: python3 tests/test_splat2.py  (CPU; needs the torchsky venv for (e))
(a) pixel-integrated Gaussian rasterisation vs a 32x32 sub-sampled reference, max error per sigma
(b) float64 gradcheck of the rasteriser w.r.t. mu, sigma, theta
(c) synthetic recovery on 64x64 (5 bands, known Gaussian PSF, noise): no splats -> densify / solve / prune; position, flux, false splats
(d) the projected-CG amplitude solve equals dense bounded least squares (scipy) on a tiny case (+ unconstrained and Huber-IRLS variants),
    and the Fisher-diagonal approximation vs the exact diagonal
(e) end-to-end --quick run of smacs_four_v32.py with --sky-coeffs splat2 --spectral rbf on testdata/nircam (256 px: 128 px has <30 null peaks)
(f) regression for the CANFAR round-2 index-out-of-bounds in split_tests (off-image / NaN / huge-sigma splats; failure in split_tests is survived)
"""
import json
import math
import os
import subprocess
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import smacs_four_v32 as S  # noqa: E402

CPU = torch.device("cpu")
NIRCAM_DIR = os.path.join(ROOT, "testdata", "nircam")
SVO = os.path.join(ROOT, "testdata", "svo_cache")


# ---------------------------------------------------------------------------------------------------- (a)
def reference_image(mu, sg, th, H, W, n=32):
    o = (torch.arange(n, dtype=torch.float64) + 0.5) / n - 0.5
    ys = (torch.arange(H, dtype=torch.float64)[:, None] + o[None, :]).reshape(-1)
    xs = (torch.arange(W, dtype=torch.float64)[:, None] + o[None, :]).reshape(-1)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    dx, dy = X - mu[0], Y - mu[1]
    u, v = math.cos(th) * dx + math.sin(th) * dy, -math.sin(th) * dx + math.cos(th) * dy
    g = torch.exp(-0.5 * ((u / sg[0]) ** 2 + (v / sg[1]) ** 2)) / (2 * math.pi * sg[0] * sg[1])
    return g.view(H, n, W, n).mean((1, 3))


def raster_image(mu, sg, th, H, W, **kw):
    j, l, v = S.splat_raster(torch.tensor([mu], dtype=torch.float64), torch.tensor([sg], dtype=torch.float64),
                             torch.tensor([th], dtype=torch.float64), H, W, **kw)
    return torch.zeros(H * W, dtype=torch.float64).index_add(0, l, v).view(H, W)


def test_a_rasterisation():
    H = W = 48
    print("(a) pixel-integrated Gaussian, 4x4 Gauss-Legendre vs 32x32 sub-sampling; max |error| over the image, relative to the peak")
    worst = 0.0
    rng = np.random.default_rng(1)
    for s in (0.5, 0.7, 1.0, 1.5, 2.0, 4.0, 8.0):
        errs, rel = [], []
        for _ in range(6):
            mu = (20 + rng.uniform(-0.5, 0.5) + 4, 24 + rng.uniform(-0.5, 0.5))
            sg = (s, s * rng.uniform(0.5, 1.0) if s * 0.5 >= 0.5 else s)
            th = rng.uniform(0, math.pi)
            ref = reference_image(mu, sg, th, H, W)
            im = raster_image(mu, sg, th, H, W)
            errs.append(float((im - ref).abs().max()))
            rel.append(errs[-1] / float(ref.max()))
        worst = max(worst, max(rel))
        print(f"    sigma {s:4.1f} (axis ratio 0.5-1): max abs error {max(errs):.2e}, relative to peak {max(rel):.2e}")
    # exact reference for the axis-aligned case: product of erf differences
    s = 0.5
    mu = (20.3, 24.4)
    ref = reference_image(mu, (s, s), 0.0, H, W)
    ex = torch.erf((torch.arange(W, dtype=torch.float64)[None] + 0.5 - mu[0]) / (s * math.sqrt(2))) - torch.erf((torch.arange(W, dtype=torch.float64)[None] - 0.5 - mu[0]) / (s * math.sqrt(2)))
    ey = torch.erf((torch.arange(H, dtype=torch.float64)[:, None] + 0.5 - mu[1]) / (s * math.sqrt(2))) - torch.erf((torch.arange(H, dtype=torch.float64)[:, None] - 0.5 - mu[1]) / (s * math.sqrt(2)))
    exact = 0.25 * ey * ex
    im = raster_image(mu, (s, s), 0.0, H, W)
    print(f"    axis-aligned sigma 0.5 against the exact erf-product: GL 4x4 max error {float((im - exact).abs().max()):.2e}, "
          f"32x32 midpoint reference {float((ref - exact).abs().max()):.2e}")
    # old naive alternative for comparison: 4x4 midpoint
    assert worst < 5e-4, worst
    # flux conservation
    assert abs(float(raster_image((24.0, 24.0), (2.0, 1.0), 0.3, H, W).sum()) - 1.0) < 1e-4
    # window classes: a wide Gaussian near the edge keeps the in-image part
    assert float(raster_image((2.0, 24.0), (6.0, 6.0), 0.0, H, W).sum()) < 1.0


# ---------------------------------------------------------------------------------------------------- (b)
def test_b_gradcheck():
    torch.manual_seed(0)
    H = W = 14
    mu = torch.tensor([[6.3, 7.1], [9.2, 5.4]], dtype=torch.float64, requires_grad=True)
    sg = torch.tensor([[1.2, 0.8], [0.9, 1.4]], dtype=torch.float64, requires_grad=True)
    th = torch.tensor([0.4, 1.9], dtype=torch.float64, requires_grad=True)

    def f(mu_, sg_, th_):
        j, l, v = S.splat_raster(mu_, sg_, th_, H, W)
        return torch.zeros(2 * H * W, dtype=torch.float64).index_add(0, j * H * W + l, v)
    ok = torch.autograd.gradcheck(f, (mu, sg, th), eps=1e-6, atol=1e-7, rtol=1e-5)
    # through the whole coefficient provider: sigma = floor + softplus(rho)
    co = S.Splat2Coeffs(3, H, W, 4, 0.5).double()
    co.set_splats(mu.detach(), sg.detach(), th.detach(), torch.rand(2, 3, dtype=torch.float64))
    co.mu.data, co.rho.data, co.theta.data = co.mu.double(), co.rho.double(), co.theta.double()
    ok2 = torch.autograd.gradcheck(lambda m_, r_, t_: S.splat_render(co.c, S.splat_raster(m_, 0.5 + torch.nn.functional.softplus(r_), t_, H, W), 3, H, W),
                                   (co.mu, co.rho, co.theta), eps=1e-6, atol=1e-7, rtol=1e-5)
    print(f"(b) gradcheck float64 (mu, sigma, theta) of the rasteriser: {ok}; of the rendered coefficient image through sigma=floor+softplus(rho): {ok2}")
    assert ok and ok2


# ---------------------------------------------------------------------------------------------------- synthetic world
BANDS = ["F090W", "F150W", "F200W", "F277W", "F356W"]
K_SYN = 4


def synth_setup(N=64, nexp=4, noise=0.5, sigma_psf=1.2, seed=0, sources=True, dtype=torch.float32):
    """5 bands x (nexp + 1 validation) full-frame exposures (sub-pixel dithers) of a fake sky seen through a known Gaussian PSF.
    Returns dict(model, exps, train_idx, val_idx, truth, basis, Phi, a, ...)."""
    g = torch.Generator().manual_seed(seed)
    S.set_registry(S.build_registry(BANDS, psf_n=15))
    nb = len(BANDS)
    lam = torch.tensor([[S.REG.lam_um[b]] for b in BANDS])
    w_nodes = torch.ones(nb, 1)
    S.LAM_LO, S.LAM_HI = 0.7, 5.3
    basis = S.RBFSpectralBasis(K_SYN, learned=False)
    Phi = S.band_phi(basis, lam, w_nodes)  # nb K
    shifts = [(0.0, 0.0), (0.5, 0.5), (0.25, -0.25), (-0.25, 0.5), (0.4, -0.4)]
    yy, xx = torch.meshgrid(torch.arange(N, dtype=torch.float32), torch.arange(N, dtype=torch.float32), indexing="ij")
    exps = []
    for b in BANDS:
        for k in range(nexp + 1):
            sx, sy = shifts[k]
            exps.append({"name": f"syn_{b}_{k}", "band": b, "det": 0, "detname": "D0", "tx": (xx + sx).flatten(), "ty": (yy + sy).flatten(),
                         "d": torch.zeros(N * N), "ivar": torch.full((N * N,), 1.0 / noise ** 2),
                         "row": yy.flatten().long(), "col": xx.flatten().long(), "dx": (xx.flatten() / N) * 2 - 1, "dy": (yy.flatten() / N) * 2 - 1,
                         "bkg0": 0.0, "holdout": False, "val": k == nexp})
    train_idx = [i for i, e in enumerate(exps) if not e["val"]]
    val_idx = [i for i, e in enumerate(exps) if e["val"]]
    band_index = dict(S.REG.index)
    sig0 = torch.full((nb,), sigma_psf)

    def build(coeffs):
        sky = S.SpectralSky(coeffs, basis, lam, w_nodes, lam_smooth=1e-3, sig_ref=noise, relative=True)
        return S.SkyModel(sky, S.ExplicitVarPSF(sig0, 1, 2, n=15), S.ExplicitBackground(exps, 1), S.Geometry(exps, 1, "fixed"), band_index, stripes=None, sig=noise)
    # ---- truth
    spec = lambda flux, slope: flux * torch.tensor([1.0, 1.0 - 0.15 * slope, 1.0 - 0.3 * slope, 1.0 - 0.45 * slope])
    truth = []
    if sources:
        for (x, y, f, sl) in [(14.3, 15.6, 60.0, 0.5), (45.7, 12.2, 30.0, 1.0), (30.1, 40.4, 15.0, 0.0), (52.4, 50.8, 8.0, 1.5)]:
            truth.append({"kind": "point", "mu": (x, y), "sigma": (0.5, 0.5), "theta": 0.0, "c": spec(f, sl)})
        truth.append({"kind": "galaxy", "mu": (20.5, 46.2), "sigma": (3.2, 1.4), "theta": 0.6, "c": spec(160.0, 0.8)})
        truth.append({"kind": "galaxy", "mu": (47.8, 31.4), "sigma": (2.4, 1.6), "theta": 2.2, "c": spec(90.0, 0.3)})
    tc = S.Splat2Coeffs(K_SYN, N, N, 4, 0.5)
    if truth:
        tc.set_splats(torch.tensor([t["mu"] for t in truth]), torch.tensor([t["sigma"] for t in truth]), torch.tensor([t["theta"] for t in truth]),
                      torch.stack([t["c"] for t in truth]))
    grad = (0.02 + 0.00002 * (xx[:: 4, :: 4] * 0 + torch.arange(0, N, 4, dtype=torch.float32)[None, :] * 3 + torch.arange(0, N, 4, dtype=torch.float32)[:, None] * 1.5))
    tc.smooth.copy_(grad[None, : tc.hs, : tc.ws].expand(K_SYN, -1, -1) * torch.tensor([1.0, 0.9, 0.8, 0.7])[:, None, None] * 20)
    tm = build(tc)
    with torch.no_grad():
        _, CB = tm.prepare()
        for e in exps:
            e["d"] = tm.predict(CB, None, dict(e, eidx=exps.index(e), detv=e["det"])) + noise * torch.randn(N * N, generator=g)
    return dict(exps=exps, train_idx=train_idx, val_idx=val_idx, truth=truth, truth_model=tm, build=build, basis=basis, Phi=Phi, lam=lam,
                w_nodes=w_nodes, band_index=band_index, N=N, noise=noise, sigma_psf=sigma_psf)


def synth_args(**kw):
    a = types.SimpleNamespace(quick=False, budget_explicit=20.0, splat_solve_iters=40, splat_round_min=0.5, splat_max_rounds=8, splat_max=20000,
                              splat_fpr=0.1, splat_prune_snr=3.0, splat_split_steps=100, splat_solve_every=25, sky_lr=0.0, psf_lr=0.0, optimizer="adam",
                              splat_smooth_px=4, splat_sigma_floor=0.5)
    a.__dict__.update(kw)
    return a


def make_model(W_, coeffs=None):
    N = W_["N"]
    coeffs = coeffs or S.Splat2Coeffs(K_SYN, N, N, 4, 0.5)
    return W_["build"](coeffs)


# ---------------------------------------------------------------------------------------------------- (c)
def val_callback(W_):
    ex = W_["exps"]

    def cb(model, it, el):
        with torch.no_grad():
            _, CB = model.prepare()
            tot, n = 0.0, 0
            for i in W_["val_idx"]:
                e = ex[i]
                m = model.predict(CB, None, dict(e, eidx=i, detv=e["det"]))
                tot += float((e["ivar"] * (e["d"] - m) ** 2).sum())
                n += e["d"].numel()
        return tot / n
    return cb


def run_synthetic(seed=0, sources=True, steps=150, quiet=False):
    torch.manual_seed(seed)
    W_ = synth_setup(seed=seed, sources=sources)
    m = make_model(W_)
    a = synth_args(splat_round_min=1.0)
    S.args_global, S.SAVE_DIR = a, None
    S.LOG.clear()
    S.fit_splat2("syn", m, W_["exps"], steps, {"sky": 0.1, "psf": 2e-3, "bkg": 1e-3, "geom": 2e-3}, torch.tensor(W_["noise"]), W_["train_idx"],
                 W_["band_index"], val_callback(W_), a, W_["Phi"], CPU)
    return W_, m, a


def associate(W_, co):
    """per truth source: the recovered splats within 3.5 sigma_max(truth) + 1 px; splats attached to no truth source = false."""
    Phi = W_["Phi"]
    mu, sg = co.mu.detach(), co.sigma().detach()
    band_flux = (co.c @ Phi.T).detach()  # J nb
    owner = torch.full((co.n,), -1, dtype=torch.long)
    for t, tr in enumerate(W_["truth"]):
        d = (mu - torch.tensor(tr["mu"])).norm(dim=1)
        rad = 3.5 * max(tr["sigma"]) + 1.0
        for j in torch.nonzero(d < rad).flatten().tolist():
            if owner[j] < 0:
                owner[j] = t
    return owner, band_flux


def test_c_recovery():
    t0 = time.time()
    W_, m, a = run_synthetic(0)
    co = m.sky.coeffs
    owner, bf = associate(W_, co)
    Phi = W_["Phi"]
    print(f"(c) synthetic recovery: 64x64, 5 bands x 4 dithers (+1 validation), Gaussian PSF sigma {W_['sigma_psf']} px, noise {W_['noise']} per pixel "
          f"[{time.time() - t0:.0f} s]")
    print("    rounds:", [(r["round"], r["pruned"], r["split"], r["added"], r["total"], None if r["val_score"] is None else round(r["val_score"], 4)) for r in S.LOG["syn_splat_rounds"]],
          "(round, pruned, split, added, total, validation chi2/pix)")
    print(f"    final: {co.n} splats; best validation round {S.LOG.get('syn_best_round')}")
    nb = len(BANDS)
    fails = []
    for t, tr in enumerate(W_["truth"]):
        js = torch.nonzero(owner == t).flatten()
        tf = (tr["c"] @ Phi.T)
        if js.numel() == 0:
            fails.append(f"truth {t} ({tr['kind']}) not recovered")
            print(f"    truth {t} {tr['kind']:6s} at {tr['mu']}: NOT RECOVERED")
            continue
        fl = bf[js].sum(0)
        cen = (co.mu.detach()[js] * bf[js].sum(1, keepdim=True)).sum(0) / bf[js].sum()
        dpos = float((cen - torch.tensor(tr["mu"])).norm())
        if tr["kind"] == "point":
            best = js[(co.mu.detach()[js] - torch.tensor(tr["mu"])).norm(dim=1).argmin()]
            dpos = float((co.mu.detach()[best] - torch.tensor(tr["mu"])).norm())
        # expected flux error per band: sqrt(4 pi s_eff1 s_eff2 / Om), Om = 4 dithers x ivar
        se = [math.sqrt(s ** 2 + W_["sigma_psf"] ** 2) for s in tr["sigma"]]
        sF = math.sqrt(4 * math.pi * se[0] * se[1] / (4 * (1 / W_["noise"] ** 2)))
        pull = ((fl - tf) / sF)
        print(f"    truth {t} {tr['kind']:6s} at ({tr['mu'][0]:.1f},{tr['mu'][1]:.1f}): {js.numel()} splat(s), position error {dpos:.3f} px, band flux "
              f"{[round(float(x), 1) for x in fl]} vs {[round(float(x), 1) for x in tf]}, pulls {[round(float(x), 1) for x in pull]} (sigma_F ~ {sF:.2f})")
        snr_tot = float(((tf / sF) ** 2).sum().sqrt())
        sx = math.sqrt(se[0] * se[1]) / snr_tot  # Cramer-Rao-like position error of a Gaussian blob
        print(f"        total S/N {snr_tot:.1f}, expected position error ~ {sx:.3f} px, flux deviation {float(((fl - tf) / tf).abs().max()) * 100:.1f} % (max over bands)")
        if tr["kind"] == "point":
            if dpos > max(0.2, 3 * sx):
                fails.append(f"truth {t}: position error {dpos:.3f} px (expected {sx:.3f})")
            if float(pull.abs().max()) > 4.0:
                fails.append(f"truth {t}: flux pulls {pull.tolist()}")
        else:  # extended: the 4 px smooth grid is partly degenerate with a sigma ~ 3 px blob, so only a fractional flux criterion
            if dpos > 0.2 or float(((fl - tf) / tf).abs().max()) > 0.12:
                fails.append(f"truth {t}: position {dpos:.3f} px, flux deviation {((fl - tf) / tf).tolist()}")
    nfalse = int((owner < 0).sum())
    print(f"    false splats (attached to no true source): {nfalse}")
    # pure-noise control: no sources, three noise realisations
    ctr = []
    for sd in (11, 12, 13):
        Wn, mn, _ = run_synthetic(sd, sources=False, steps=40)
        ctr.append(mn.sky.coeffs.n)
    print(f"    pure-noise control (no sources, 3 realisations): splats created {ctr}; smooth-only validation chi2/pix {S.LOG['syn_splat_rounds'][0]['val_score']:.4f}")
    assert not fails, fails
    assert nfalse == 0, nfalse
    assert sum(ctr) == 0, ctr


# ---------------------------------------------------------------------------------------------------- (d)
def dense_system(prob, sy, ns, J, K):
    """Dense H and b2 of prob.system by applying the matvec to unit vectors."""
    n = ns + J * K
    H = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        e = torch.zeros(n)
        e[i] = 1.0
        H[:, i] = sy["matvec"](e).double()
    return 0.5 * (H + H.T), sy["b2"].double()


def test_d_linear_solve():
    from scipy.optimize import lsq_linear
    torch.manual_seed(0)
    W_ = synth_setup(N=12, nexp=3, noise=0.5, seed=5, sources=False)
    # tiny truth: one real point source, one faint, and a NEGATIVE one (a dip) where the fit model has a splat: its constraint is active
    ex = W_["exps"]
    tm = make_model(W_)
    tc = tm.sky.coeffs
    tc.set_splats(torch.tensor([[4.2, 5.1], [8.4, 3.3], [9.5, 9.0]]), torch.tensor([[0.5, 0.5], [0.8, 0.5], [0.6, 0.6]]), torch.tensor([0.0, 0.5, 0.0]),
              torch.stack([torch.tensor([1.0, 0.9, 0.8, 0.7]) * 12.0, torch.tensor([1.0, 0.9, 0.8, 0.7]) * 2.0, torch.tensor([1.0, 0.9, 0.8, 0.7]) * -6.0]))
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        _, CB = tm.prepare()
        for i, e in enumerate(ex):
            e["d"] = tm.predict(CB, None, dict(e, eidx=i, detv=e["det"])) + W_["noise"] * torch.randn(e["d"].numel(), generator=g)
    tc.set_splats(torch.zeros(0, 2), torch.zeros(0, 2), torch.zeros(0), torch.zeros(0, K_SYN))
    m = make_model(W_)
    co = m.sky.coeffs
    # fit model: three splats - near source 1, a mis-placed one next to source 2 and one on empty sky
    co.set_splats(torch.tensor([[4.1, 5.0], [8.0, 3.0], [9.5, 9.0]]), torch.tensor([[0.6, 0.6], [0.7, 0.5], [0.6, 0.6]]), torch.tensor([0.0, 0.3, 0.0]),
                  torch.ones(3, K_SYN))
    S.args_global, S.SAVE_DIR = synth_args(), None
    batches = S.band_batches(ex, W_["train_idx"], 4)
    ntot = sum(ex[i]["d"].numel() for i in W_["train_idx"])
    phi_ref = W_["basis"](torch.tensor([2.0]))[0]
    prob = S.Splat2Problem(m, batches, ntot, phi_ref, W_["band_index"], "gauss", 3.0, Phi=W_["Phi"])
    co.invalidate()
    prob.fp = co.footprints(detach=True)
    prob.setup_graph()
    smooth, cmat = co.smooth.clone(), co.c.clone()
    sy = prob.system(smooth, cmat, co.sigma().detach())
    ns, J, K = smooth.numel(), cmat.shape[0], K_SYN
    H, b2 = dense_system(prob, sy, ns, J, K)
    n = ns + J * K
    con = sy["con"].numpy()
    # dense bounded least squares: H = 2 R'R  ->  rows sqrt(2) * sqrt(Lambda) V'; solves min 0.5 x'Hx - b2'x = 0.5||R x - r0||^2 + const
    lam_, V = torch.linalg.eigh(H)
    keep = lam_ > 1e-12 * float(lam_.max())
    Rm = (lam_[keep].sqrt()[:, None] * V[:, keep].T).numpy()
    r0 = (V[:, keep].T @ b2 / lam_[keep].sqrt()).numpy()
    lb = np.where(con, 0.0, -np.inf)
    ref = lsq_linear(Rm, r0, bounds=(lb, np.full(n, np.inf)), method="bvls", tol=1e-14, max_iter=2000)
    x_ref = torch.tensor(ref.x)
    # projected CG (the production routine; float32 operator), with and without the Jacobi preconditioner
    res = {}
    for name, mv in (("projected CG + Jacobi", sy["minv"]), ("projected CG", None)):
        x, info = S.projected_cg(sy["matvec"], sy["b2"], sy["pack"](smooth, cmat), sy["con"], iters=300, tol=1e-7, minv=mv)
        obj = lambda v: float(0.5 * v.double() @ H @ v.double() - b2 @ v.double())
        rel = float((x.double() - x_ref).norm() / x_ref.norm())
        res[name] = (rel, info)
        print(f"(d) {name:22s}: n = {n} unknowns ({J} splats), {info['matvecs']} matvecs, {info['n_active']} amplitudes at 0 (bvls: {int(((x_ref < 1e-9) & torch.tensor(con)).sum())}), "
              f"|x - x_bvls| / |x_bvls| = {rel:.2e}, objective gap {obj(x) - obj(x_ref.float()):.2e} (objective scale {abs(obj(x_ref.float())):.2e})")
        assert rel < 2e-3, rel
    assert res["projected CG"][1]["n_active"] >= 1, "test case has no active constraint: change the seed"
    # unconstrained splats: plain CG == normal equations
    xu, _ = S.projected_cg(sy["matvec"], sy["b2"], sy["pack"](smooth, cmat), torch.zeros_like(sy["con"]), iters=300, tol=1e-7, minv=sy["minv"])
    xs = torch.linalg.pinv(H) @ b2
    print(f"    unconstrained: |x - pinv| / |pinv| = {float((xu.double() - xs).norm() / xs.norm()):.2e}")
    assert float((xu.double() - xs).norm() / xs.norm()) < 2e-3
    # penalty check: the quadratic form of pen_apply on the smooth grid equals the SpectralSky.penalty formula (spectral part) + spatial prior
    s_ = torch.randn_like(smooth) * 0.3
    ps, _ = prob.pen_apply(s_, torch.zeros(0, K))
    quad = float((s_ * ps).sum())
    d2 = torch.einsum("jk,khw->jhw", prob.D2, s_)
    spec = float(m.sky.lam_smooth * (d2 / prob.sc_s).pow(2).mean())
    spat = float(prob.w_sp / ntot * (((s_[:, :, 1:] - s_[:, :, :-1]) ** 2).sum() + ((s_[:, 1:] - s_[:, :-1]) ** 2).sum()))
    print(f"    penalty quadratic form on the smooth grid: {quad:.6e} vs SpectralSky.penalty-form {spec:.6e} + spatial prior {spat:.6e} = {spec + spat:.6e}")
    assert abs(quad - (spec + spat)) < 1e-4 * (spec + spat), (quad, spec, spat)
    # Fisher-diagonal approximation vs the exact diagonal e'Ge / ntot of the broadband amplitude of every splat (W = ivar)
    S.set_registry(S.REG)
    num, om = S.residual_coadd(m, batches, W_["band_index"])
    prob.fp = co.footprints(detach=True)
    sp_ = S.psf_core_sigma(m)
    q1 = prob.Q[:, 0]
    snr, Fj = S.amplitude_snr(co, om, sp_, W_["Phi"], q1)
    ex_diag = []
    for j in range(J):
        z = torch.zeros(K)
        z[0] = 1.0
        c_ = torch.zeros(J, K)
        c_[j] = prob.Q @ z
        y = prob.forward_lin(torch.zeros_like(smooth), c_)
        w_ = torch.cat([c["ivar"] for c in prob.cache])
        ex_diag.append(float((w_ * y * y).sum()))
    print(f"    Fisher diagonal of the broadband amplitude, analytic approximation / exact: {[round(float(Fj[j]) / ex_diag[j], 2) for j in range(J)]}")
    assert all(0.4 < float(Fj[j]) / ex_diag[j] < 2.5 for j in range(J))
    # robust loss: outlier pixels (cosmic-ray like) pull the Gaussian solution, the IRLS-Huber solution resists
    ex0 = [e["d"].clone() for e in ex]
    for e in ex[: 3]:
        e["d"][5 * 12 + 4] += 400.0
        e["d"][5 * 12 + 5] += 300.0
    outs = {}
    for kind in ("gauss", "huber"):
        S.LOSS_KIND, S.HUBER_DELTA = kind, (3.0 if kind == "huber" else None)
        co.set_splats(co.mu.detach(), co.sigma().detach(), co.theta.detach(), torch.ones(J, K))
        co.smooth.zero_()
        S.splat2_solve(m, S.band_batches(ex, W_["train_idx"], 4), ntot, W_["band_index"], phi_ref, synth_args(splat_solve_iters=200), Phi=W_["Phi"])
        outs[kind] = float((co.c[0] @ W_["Phi"].T).mean())
    S.LOSS_KIND, S.HUBER_DELTA = "gauss", None
    for e, d0 in zip(ex, ex0):
        e["d"] = d0
    print(f"    source-1 mean band flux with cosmic-ray-like outliers next to it: Gaussian {outs['gauss']:.2f}, Huber-IRLS {outs['huber']:.2f}, truth {float((tm.sky.coeffs.c[0] @ W_['Phi'].T).mean()) if tm.sky.coeffs.n else float((torch.tensor([1.0, 0.9, 0.8, 0.7]) * 12.0 @ W_['Phi'].T).mean()):.2f}")
    assert outs["huber"] < outs["gauss"]


# ---------------------------------------------------------------------------------------------------- (e)
def have_torchsky():
    try:
        import torchsky  # noqa: F401
        from torchsky.wcs import WCS  # noqa: F401
        return True
    except Exception as exc:  # noqa: BLE001
        print("    torchsky not importable:", repr(exc)[:120])
        return False


def test_e_end_to_end_quick():
    if not have_torchsky() or not os.path.isdir(NIRCAM_DIR):
        print("(e) SKIPPED: torchsky (or the NIRCam test data) not available")
        return
    out = os.path.join(ROOT, "runs", "v32_quick")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "smacs_four_v32.py"), "--data-dir", NIRCAM_DIR, "--out", out, "--size", "256", "--quick",
           "--no-fallback", "--no-holdout", "--score-all", "--boot-b", "50", "--budget-explicit", "1000", "--budget-implicit", "1000",
           "--spectral", "rbf", "--svo-cache", SVO, "--spec-q", "3", "--skip-implicit", "--explicit-configs", "fixed",
           "--sky-coeffs", "splat2", "--splat-solve-every", "10", "--stripes", "1"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = p.stdout + p.stderr
    open(os.path.join(out, "stdout.log"), "w").write(log)
    print(f"(e) end-to-end --quick run: exit {p.returncode} in {time.time() - t0:.0f} s; log {out}/stdout.log")
    for ln in log.splitlines():
        if any(k in ln for k in ("splat round", "splat2 sky", "Traceback", "Error", "restored", "stopping")):
            print("    " + ln[:260])
    assert p.returncode == 0, log[-3000:]
    M = json.load(open(os.path.join(out, "metrics.json")))
    rounds = M["explicit_fixedgeom_3d_splat_rounds"]
    assert len(rounds) >= 2 and sum(r["added"] + r["split"] for r in rounds) > 0, rounds
    print("    rounds (round, pruned, split, added, total):", [(r["round"], r["pruned"], r["split"], r["added"], r["total"]) for r in rounds])
    from astropy.io import fits
    cube = fits.getdata(os.path.join(out, "cubes", "explicit_fixedgeom_3d_sky.fits"))
    assert cube.shape == (6, 256, 256) and np.isfinite(cube).all()
    z = np.load(os.path.join(out, "explicit_fixedgeom_3d_splats.npz"))
    print(f"    splat catalogue: {z['mu'].shape[0]} splats, sigma min {z['sigma'].min() if z['sigma'].size else float('nan')}; sky cube finite {cube.shape}")
    assert z["mu"].shape[0] > 0 and z["sigma"].min() >= 0.5 - 1e-6


# ---------------------------------------------------------------------------------------------------- (f) regression: CANFAR crash
def test_f_edge_regression():
    """Round-2 CUDA 'index out of bounds' in split_tests -> splat_raster on CANFAR: a splat whose centre had left the image (Adam) or a
    large-sigma splat near the edge made the stamp gather index outside the padded residual map (CPU: IndexError; negative indices
    wrapped silently). Now: off-image / non-finite splats are skipped by the split test and dropped by sanitize(); stamps of on-image
    centres are clamped inside the padded map; huge or NaN sigma cannot create giant windows; a failure inside the split test is
    logged and skipped."""
    torch.manual_seed(0)
    W_ = synth_setup()
    N, nb = W_["N"], len(BANDS)
    m = make_model(W_)
    co = m.sky.coeffs
    mu = torch.tensor([[-5.0, 30.0], [30.0, 70.0], [32.0, 32.0], [63.0, 63.0], [0.0, 0.0], [float("nan"), 3.0], [31.0, 5.0], [-0.4, 20.0]])
    sg = torch.tensor([[8.0, 8.0], [8.0, 8.0], [30.0, 20.0], [8.0, 6.0], [8.0, 8.0], [1.0, 1.0], [1e5, 1e5], [3.0, 3.0]])
    co.set_splats(mu, sg.clamp(max=1e5), torch.zeros(8), torch.ones(8, K_SYN))
    # (1) rasteriser: NaN centre / huge sigma are masked or capped, not turned into garbage indices or giant windows
    j, l, v = S.splat_raster(co.mu, co.sigma(), co.theta, N, N)
    assert l.numel() == 0 or (int(l.min()) >= 0 and int(l.max()) < N * N and torch.isfinite(v).all())
    # (2) split test with a peak next to every splat, including the off-image ones
    R, om = torch.randn(nb, N, N), torch.ones(nb, N, N)
    xy = mu.nan_to_num(0.0).clamp(0, N - 1)
    out, log = S.split_tests(co, R, om, xy, 10.0, torch.full((nb,), 1.2), W_["Phi"], 1.0, 0.1, 0.5, steps=5)
    print(f"(f) split_tests with off-image, NaN, huge-sigma and edge splats: no exception; candidates {log['candidates']}, skipped off-image/non-finite {log['skipped_off_image']}")
    assert log["skipped_off_image"] >= 3
    # (3) sanitize drops exactly the off-image / non-finite ones (centre outside [-0.5, N-0.5])
    n0 = co.n
    nd = co.sanitize()
    print(f"    sanitize(): {nd} of {n0} splats dropped")
    assert nd == 4 and co.n == 4, (nd, co.n)
    # (4) fit_splat2 survives an exception inside split_tests (monkeypatched) and records it
    S_orig = S.split_tests

    def boom(*a_, **k_):
        raise RuntimeError("synthetic device-side failure")
    S.split_tests = boom
    try:
        W2 = synth_setup(seed=0)
        torch.manual_seed(0)
        m2 = make_model(W2)
        a = synth_args(splat_round_min=0.2, splat_max_rounds=3)
        S.args_global, S.SAVE_DIR = a, None
        S.LOG.clear()
        S.fit_splat2("rg", m2, W2["exps"], 40, {"sky": 0.1, "psf": 2e-3, "bkg": 1e-3, "geom": 2e-3}, torch.tensor(W2["noise"]), W2["train_idx"],
                     W2["band_index"], val_callback(W2), a, W2["Phi"], CPU)
    finally:
        S.split_tests = S_orig
    rr = S.LOG["rg_splat_rounds"]
    errs = [r["split_test"].get("error") for r in rr if r.get("split_test")]
    print(f"    fit_splat2 with split_tests raising: {len(rr)} rounds completed, {m2.sky.coeffs.n} splats, split_test errors logged: {len(errs)}")
    assert len(rr) >= 2 and all(e and "synthetic" in e for e in errs) and m2.sky.coeffs.n > 0
    assert all(r["split"] == 0 for r in rr)


if __name__ == "__main__":
    torch.set_num_threads(2)
    tests = [test_a_rasterisation, test_b_gradcheck, test_c_recovery, test_d_linear_solve, test_e_end_to_end_quick, test_f_edge_regression]
    only = sys.argv[1:]
    for t in tests:
        if only and t.__name__.split("_")[1] not in only:
            continue
        t()
        print(f"{t.__name__}: OK\n")
    print("ALL OK")
