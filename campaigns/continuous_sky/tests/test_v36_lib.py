"""Tests for scripts/smacs_four_v36.py (v35 library PSF tests a-d, plus the data-driven arm: generic Gaussian tables, learnable tables). Plain asserts, no pytest:
    python tests/test_v35_lib.py [a b c d e]
(a) synthetic PSF library in the exact npz / meta.json format (Gaussian core + power-law wing ePSFs, different per detector and band)
(b) kernel rendering: theta = 0, parity +1, M = diag(s, s): library integrated onto the sky grid (1e-3 relative L1), unit sum, centroid;
    theta = 90 deg is a rot90; parity -1 is a flip
(c) warm start: zero residual gives exactly the library kernels; gradients reach the residual networks; penalty is zero at the warm start
(d) smoke run of smacs_four_v36.py on testdata/nircam with the new flags (library nodes deliberately off by 0.01 um: nearest-node fallback)
(f) generic Gaussian tables and learnable-table properties (g) smoke run, gauss + learnable table, no residual networks (h) multi-instrument identifiability / common-mode diagnostic
tests/check_v36_equiv.py is the bitwise comparison with v35."""
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import smacs_four_v36 as S  # noqa: E402

NIR = os.path.join(ROOT, "testdata", "nircam")
SVO = os.path.join(ROOT, "testdata", "svo_cache")
CPU = torch.device("cpu")
OS = 4
NODES = {"F090W": [0.8122, 0.8475, 0.8822, 0.9175, 0.9525, 0.9884], "F150W": [1.3565, 1.4128, 1.4682, 1.5248, 1.5824, 1.6403],
         "F200W": [1.7897, 1.8651, 1.9418, 2.0205, 2.1008, 2.1849], "F277W": [2.4704, 2.5793, 2.6889, 2.8031, 2.9227, 3.0596],
         "F356W": [3.2065, 3.3499, 3.4929, 3.6308, 3.7679, 3.9094], "F444W": [3.9578, 4.1177, 4.2832, 4.4593, 4.6524, 4.8787]}


def epsf_fn(x, y, sig, wing=0.002, pw=2.5, ell=0.0, lobe=0.0):
    """Analytic effective PSF (unnormalised): elliptical Gaussian core + power-law wing (+ an off-centre lobe to break symmetry)."""
    core = np.exp(-0.5 * ((x / (sig * (1 + ell))) ** 2 + (y / (sig * (1 - ell))) ** 2))
    w = wing / (1.0 + (x * x + y * y) / (4 * sig * sig)) ** (pw / 2)
    lb = lobe * np.exp(-0.5 * (((x - 1.5 * sig) / (0.7 * sig)) ** 2 + ((y + 0.8 * sig) / (0.7 * sig)) ** 2))
    return core + w + lb


def epsf_array(N, sig, **kw):
    ax = (np.arange(N) - N // 2) / OS
    x, y = np.meshgrid(ax, ax)
    k = epsf_fn(x, y, sig, **kw)
    return (k / k.sum()).astype(np.float32)


def make_library(path, node_shift=0.0, N=161):
    """nircam_epsf.npz (NRCB1-4 for SW bands, NRCB5 for LW), wfc3ir_epsf.npz (3 x 3 grid), meta.json: the format of gen_psf_lib.py."""
    os.makedirs(path, exist_ok=True)
    lib = {}
    for b, nodes in NODES.items():
        dets = ["NRCB1", "NRCB2", "NRCB3", "NRCB4"] if b in ("F090W", "F150W", "F200W") else ["NRCB5"]
        for di, det in enumerate(dets):
            stack = []
            for lam in nodes:
                sig = 0.50 * lam / 1.5 * (1.0 + 0.04 * di) if det != "NRCB5" else 0.45 * lam / 1.5 * 0.5 + 0.35
                stack.append(epsf_array(N, max(sig, 0.5), wing=0.002 * (1 + di), ell=0.03 * di, lobe=0.02 * (di % 2)))
            lib[f"{det}/{b}"] = np.stack(stack)
    np.savez_compressed(os.path.join(path, "nircam_epsf.npz"), **lib)
    wl = {}
    for f, sg in (("F105W", 0.55), ("F125W", 0.60), ("F140W", 0.66), ("F160W", 0.72)):
        wl[f"WFC3IR_{f}"] = np.stack([epsf_array(101, sg * (1 + 0.01 * i)) for i in range(9)])
    np.savez_compressed(os.path.join(path, "wfc3ir_epsf.npz"), **wl)
    json.dump({"oversample": OS, "fov_arcsec": 4.0, "nodes_um": {b: [x + node_shift for x in v] for b, v in NODES.items()}}, open(os.path.join(path, "meta.json"), "w"))
    return path


class FakeLib:
    """In-memory stand-in of PSFLibrary for the synthetic experiments (one array per (detector, band), same for every node)."""

    def __init__(self, arrays, oversample=OS):
        self.arrays, self.oversample, self.notes = arrays, oversample, []

    def get(self, detname, band, channel):
        a = self.arrays[(detname, band)]
        return a / a.sum((-2, -1), keepdims=True), None


def setup_reg(bands):
    S.set_registry(S.build_registry(bands, psf_n=31))
    S.PSF_FRAME = "detector"


def group(b=0, theta=0.0, parity=1, s=2.0, det=0):
    return {"id": 0, "b": b, "theta_deg": theta, "parity": parity, "M": [[s, 0.0], [0.0, s]], "sx": s, "sy": s, "S": 1, "det": det}


def rel_l1(a, b):
    return float(np.abs(a - b).sum() / np.abs(b).sum())


def centroid(k):
    n = k.shape[-1]
    r = np.arange(n) - n // 2
    return float((k.sum(0) * r).sum() / k.sum()), float((k.sum(1) * r).sum() / k.sum())


# ------------------------------------------------------------------------------------------------------------------------------ (a)
def test_a():
    with tempfile.TemporaryDirectory() as td:
        make_library(td)
        lib = S.PSFLibrary(td)
        assert lib.oversample == OS
        a, nodes = lib.get("NRCB3", "F150W", "NIRCAM_SW")
        assert a.shape == (6, 161, 161) and abs(float(a[0].sum()) - 1) < 1e-5 and len(nodes) == 6
        a5, _ = lib.get("NRCBLONG", "F277W", "NIRCAM_LW")  # NRCBLONG <-> NRCB5
        assert a5.shape[0] == 6
        w, nd = lib.get("WFC3_IR", "WFC3IR_F105W", "WFC3_IR")
        assert nd is None and w.shape == (1, 101, 101)
        assert np.allclose(w[0], np.load(os.path.join(td, "wfc3ir_epsf.npz"))["WFC3IR_F105W"][4], atol=1e-7)
        assert any("index 4" in n for n in lib.notes), lib.notes
        np.savez(os.path.join(td, "wfc3ir_epsf.npz"), WFC3IR_F105W=np.random.rand(5, 51, 51).astype(np.float32))  # unexpected shape: robust
        lib2 = S.PSFLibrary(td)
        w2, _ = lib2.get("WFC3_IR", "WFC3IR_F105W", "WFC3_IR")
        assert w2.shape == (1, 51, 51) and any("UNEXPECTED" in n for n in lib2.notes)
    print("(a) PASS: synthetic library read back; NRCBLONG -> NRCB5; WFC3 centre index 4; unexpected grid shape handled and logged")


# ------------------------------------------------------------------------------------------------------------------------------ (b)
def mk_psf(arrs, bands, dets, pairs, lam_nodes=None, n=31, scale=0.04):
    setup_reg(bands)
    return S.LibPSF(bands, dets, pairs, FakeLib(arrs), lam_nodes, scale, n=n, hidden=16)


def test_b():
    n, s = 31, 2.0
    sig = 1.5  # the 1e-3 tolerance holds for cores resolved by the S_c = 3 midpoint rule (sigma >~ 1.3 detector px at this scale); see the info lines
    arr = {("D", "F150W"): epsf_array(161, sig)[None]}
    psf = mk_psf(arr, ["F150W"], ["D"], {(0, 0)}, n=n)
    with torch.no_grad():
        K = psf.group_kernels([group(s=s)], None)[0, 0].numpy()
    # reference: analytic ePSF averaged over the sky cell (21 x 21 midpoints), same window, unit sum
    t = (np.arange(21) + 0.5) / 21 - 0.5
    r = np.arange(n) - n // 2
    ref = np.zeros((n, n))
    for vy in t:
        for vx in t:
            X, Y = np.meshgrid((r + vx) / s, (r + vy) / s)
            ref += epsf_fn(X, Y, sig)
    ref /= ref.sum()
    e1 = rel_l1(K, ref)
    # second reference: the oversampled library trapezoid-integrated over the 2 x 2 oversampled-sample cell (s = 2: cell = OS/s = 2 samples)
    A = arr[("D", "F150W")][0].astype(np.float64)
    c = A.shape[0] // 2
    w1 = np.array([0.5, 1.0, 0.5]) / 2.0
    ref2 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            yy, xx = c + (i - n // 2) * 2, c + (j - n // 2) * 2
            ref2[i, j] = (A[yy - 1:yy + 2, xx - 1:xx + 2] * np.outer(w1, w1)).sum()
    ref2 /= ref2.sum()
    e2 = rel_l1(K, ref2)
    cx, cy = centroid(K)
    psf.interp = "grid_sample"
    with torch.no_grad():
        Kgs = psf.group_kernels([group(s=s)], None)[0, 0].numpy()
    psf.interp = "bspline"
    print(f"    info: torch bicubic grid_sample (Keys a = -0.75) instead of the cubic B-spline: rel L1 vs analytic cell mean {rel_l1(Kgs, ref):.2e}")
    print(f"(b) theta 0, parity +1, M = diag({s}, {s}), core sigma {sig} detector px: rel L1 vs analytic cell mean {e1:.2e}; vs trapezoid-integrated library {e2:.2e}; "
          f"sum {K.sum():.6f}; centroid ({cx:.1e}, {cy:.1e}) px")
    assert e1 < 1e-3, e1
    assert e2 < 1e-2, e2  # the trapezoid rule on the 2-sample cell is itself only accurate to a few 1e-3
    assert abs(K.sum() - 1) < 1e-5 and max(abs(cx), abs(cy)) < 0.01
    # sharper core / non-integer scale: information only (the S_c = 3 midpoint rule is the stated approximation)
    for sig_, s_ in ((1.2, 2.0), (0.43, 1.29), (0.43, 2.0), (0.7, 1.29)):
        arr2 = {("D", "F150W"): epsf_array(161, sig_)[None]}
        p2 = mk_psf(arr2, ["F150W"], ["D"], {(0, 0)}, n=n)
        with torch.no_grad():
            K2 = p2.group_kernels([group(s=s_)], None)[0, 0].numpy()
        ref3 = np.zeros((n, n))
        for vy in t:
            for vx in t:
                X, Y = np.meshgrid((r + vx) / s_, (r + vy) / s_)
                ref3 += epsf_fn(X, Y, sig_)
        ref3 /= ref3.sum()
        print(f"    info: core sigma {sig_} px, s = {s_}: rel L1 vs analytic cell mean {rel_l1(K2, ref3):.2e}")
    # asymmetric library: rotation and parity
    arr3 = {("D", "F150W"): epsf_array(161, 1.2, lobe=0.5, ell=0.1)[None]}
    p3 = mk_psf(arr3, ["F150W"], ["D"], {(0, 0)}, n=n)
    with torch.no_grad():
        K0 = p3.group_kernels([group(s=2.0)], None)[0, 0].numpy()
        K90 = p3.group_kernels([group(theta=90.0, s=2.0)], None)[0, 0].numpy()
        Kp = p3.group_kernels([group(parity=-1, s=2.0)], None)[0, 0].numpy()
        K180 = p3.group_kernels([group(theta=180.0, s=2.0)], None)[0, 0].numpy()
    d_cw, d_ccw = rel_l1(K90, np.rot90(K0, -1)), rel_l1(K90, np.rot90(K0, 1))
    k_flip = np.flipud(K0)
    print(f"    theta 90 vs rot90(K0, -1) {d_cw:.1e}, vs rot90(K0, +1) {d_ccw:.1e}; theta 180 vs rot180 {rel_l1(K180, np.rot90(K0, 2)):.1e}; "
          f"parity -1 vs flipud(K0) {rel_l1(Kp, k_flip):.1e} (vs fliplr {rel_l1(Kp, np.fliplr(K0)):.1e}); asymmetry of K0 (vs its own flip) {rel_l1(K0, k_flip):.2f}")
    assert min(d_cw, d_ccw) < 1e-4 and max(d_cw, d_ccw) > 0.05
    assert rel_l1(K180, np.rot90(K0, 2)) < 1e-4 and rel_l1(Kp, k_flip) < 1e-4
    print("(b) PASS")


# ------------------------------------------------------------------------------------------------------------------------------ (c)
def test_c():
    arr = {("D", "F150W"): epsf_array(161, 1.2, lobe=0.3)[None], ("E", "F277W"): epsf_array(161, 2.0)[None]}
    psf = mk_psf(arr, ["F150W", "F277W"], ["D", "E"], {(0, 0), (1, 1)})
    assert set(psf.nets.keys()) == set(S.REG.channels) and set(psf.nets.keys()) >= {"NIRCAM_SW", "NIRCAM_LW"}
    with torch.no_grad():
        psf.det_emb.weight.normal_()
    gs = [group(s=1.7, theta=33.0, parity=-1), group(b=1, s=1.7, theta=-12.5, det=1)]
    with torch.no_grad():
        K = psf.group_kernels(gs, None)
        Klib = psf.group_kernels(gs, None, residual=False)
    assert torch.equal(K, Klib), float((K - Klib).abs().max())
    assert K.shape == (2, 1, 31, 31) and torch.allclose(K.sum((-2, -1)), torch.ones(2, 1), atol=1e-6)
    lam = torch.tensor([[1.5, 1.6], [2.7, 2.8]])
    psf2 = mk_psf({("D", "F150W"): np.repeat(arr[("D", "F150W")], 2, 0), ("E", "F277W"): np.repeat(arr[("E", "F277W")], 2, 0)}, ["F150W", "F277W"], ["D", "E"], {(0, 0), (1, 1)}, lam_nodes=lam)
    Kq = psf2.group_kernels(gs, lam)
    assert Kq.shape == (4, 1, 31, 31)
    # gradients: nonzero for the last layer of the used channel nets, none for others; penalty is exactly 0 at the warm start
    psf.zero_grad()
    w = torch.randn_like(K)
    (psf.group_kernels(gs, None) * w).sum().backward()
    g_last = {c: psf.nets[c][-1].weight.grad for c in psf.nets}
    assert all(g_last[c] is not None and torch.isfinite(g_last[c]).all() and g_last[c].abs().sum() > 0 for c in ("NIRCAM_SW", "NIRCAM_LW")), g_last
    assert psf.nets["NIRCAM_SW"][0].weight.grad.abs().sum() == 0  # first layer: zero until the last layer moves (zero-initialised output)
    # after one step of the last layers the earlier layers receive gradient as well
    with torch.no_grad():
        for c in ("NIRCAM_SW", "NIRCAM_LW"):
            psf.nets[c][-1].weight.add_(0.01 * torch.randn_like(psf.nets[c][-1].weight))
    psf.zero_grad()
    (psf.group_kernels(gs, None) * w).sum().backward()
    assert psf.nets["NIRCAM_SW"][0].weight.grad.abs().sum() > 0 and psf.det_emb.weight.grad.abs().sum() > 0
    psf3 = mk_psf(arr, ["F150W", "F277W"], ["D", "E"], {(0, 0), (1, 1)})
    psf3.centre_weight, psf3.res_l2 = 1.0, 1.0
    assert float(psf3.penalty().detach()) == 0.0
    with torch.no_grad():
        psf3.nets["NIRCAM_SW"][-1].bias.add_(0.5)
        pen = float(psf3.penalty())
    assert pen > 0 and math.isfinite(pen), pen
    # the dump has the library and the learned kernel on the oversampled grid
    dmp = psf3.detector_dump()
    k0 = dmp["D/F150W"]
    assert k0["library"].shape == (161, 161) and abs(k0["learned"].sum() - 1) < 1e-4 and k0["oversample"] == OS
    print(f"(c) PASS: zero-residual kernels == library kernels bitwise ({tuple(K.shape)}), node path {tuple(Kq.shape)}, last-layer grads finite and nonzero, "
          f"penalty 0 at warm start and {pen:.3g} after a residual offset, detector-frame dump {k0['learned'].shape}")


# ------------------------------------------------------------------------------------------------------------------------------ (d)
SMOKE_BASE = ["--data-dir", NIR, "--size", "128", "--quick", "--no-fallback", "--no-holdout", "--score-all", "--boot-b", "50",
              "--budget-explicit", "100000", "--budget-implicit", "100000", "--stripes", "1", "--eval-every-min", "0"]


def test_d():
    with tempfile.TemporaryDirectory() as td:
        lib = make_library(os.path.join(td, "lib"), node_shift=0.01)
        out = os.path.join(ROOT, "runs", "test_v35_smoke")
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "smacs_four_v36.py"), "--out", out] + SMOKE_BASE + [
            "--psf-frame", "detector", "--psf-model", "lib", "--explicit-psf", "lib", "--psf-lib", lib, "--psf-window-arcsec", "1.0",
            "--sky-init", "deconv", "--rl-iters", "5", "--sample-mode", "bicubic", "--halo-px", "40", "--spectral", "rbf", "--svo-cache", SVO,
            "--spec-q", "3", "--skip-implicit", "--explicit-configs", "explicit"]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
        print(f"    smoke run: exit {p.returncode} in {time.time() - t0:.0f} s")
        if p.returncode:
            print(p.stdout[-3000:], p.stderr[-3000:])
        assert p.returncode == 0
        M = json.load(open(os.path.join(out, "metrics.json")))
        keys = [k for k in M["metrics"]["heldout_chi2"] if "lib" in k]
        assert keys, list(M["metrics"]["heldout_chi2"])
        vals = []
        for k in keys:
            for key in ("heldout_chi2", "heldout_chi2_sources", "heldout_robust", "heldout_robust_sources"):
                v = M["metrics"][key][k]
                vals.append(v)
                assert v is not None and math.isfinite(v), (key, k, v)
        lines = p.stdout.splitlines()
        want = ["WARNING: ", "nearest library node", "Richardson-Lucy", "library PSF warm start", "PSF groups keyed by (band, theta, parity, detector)"]
        for w in want:
            assert any(w in ln for ln in lines), w
        assert M["psf_window"]["psf_n"] == 25, M["psf_window"]
        g = np.load(os.path.join(out, f"{keys[0]}_psf_groups.npy"), allow_pickle=True).item()
        assert "detector_frame" in g and any(k.startswith("NRCB3/") for k in g["detector_frame"]) and any(k.startswith("NRCBLONG/") for k in g["detector_frame"])
        assert any(isinstance(k, int) for k in g)
        for ln in lines:
            if any(w in ln for w in ("WARNING: F", "Richardson", "library PSF", "TEST explicit")):
                print("    |", ln[:260])
        print(f"(d) PASS: exit 0, finite held-out scores {['%.4f' % v for v in vals]}, window {M['psf_window']['psf_n']} px, detector-frame dump keys {sorted(g['detector_frame'])}")


# ------------------------------------------------------------------------------------------------------------------------------ (f)
def gauss_psf(sig, R, learn=False, res_on=False, n=31, os_=4, tv=0.0):
    setup_reg(["F150W"])
    lib = S.GaussLibrary(sig, os_, {("D", "F150W"): R})
    return S.LibPSF(["F150W"], ["D"], {(0, 0)}, lib, None, 0.04, n=n, hidden=8, learn=learn, res_on=res_on, tv=tv)


def cell_ref(sig, s, n):
    t = (np.arange(21) + 0.5) / 21 - 0.5
    r = np.arange(n) - n // 2
    ref = np.zeros((n, n))
    for vy in t:
        for vx in t:
            X, Y = np.meshgrid((r + vx) / s, (r + vy) / s)
            ref += np.exp(-0.5 * (X * X + Y * Y) / sig ** 2)
    return ref / ref.sum()


def test_f():
    from scipy.ndimage import spline_filter
    # gauss table == analytic Gaussian
    psf = gauss_psf(1.0, 7.3)
    T = psf._L[(0, 0)][0].numpy()
    N = T.shape[0]
    assert N == 2 * int(math.ceil(7.3 * 4)) + 1 and T.shape[0] == T.shape[1]
    ax = (np.arange(N) - N // 2) / 4.0
    x, y = np.meshgrid(ax, ax)
    ref = np.exp(-0.5 * (x * x + y * y))
    ref /= ref.sum()
    e_tab = rel_l1(T, ref)
    assert e_tab < 1e-6 and abs(T.sum() - 1) < 1e-6 and T.argmax() == (N // 2) * N + N // 2
    # differentiable prefilter == scipy spline_filter
    P = S.spline_prefilter_matrix(N).double()
    C = (P @ torch.from_numpy(T).double() @ P.T).numpy()
    C0 = spline_filter(T.astype(np.float64), order=3, mode="mirror")
    e_sp = float(np.abs(C - C0).max() / np.abs(C0).max())
    assert e_sp < 1e-5, e_sp
    # interpolation accuracy of the learnable path on sigma = 1.2 and 1.5 px cores (same cell-mean test as (b))
    out = []
    for sg in (1.2, 1.5):
        pl = gauss_psf(sg, 8.0, learn=True)
        with torch.no_grad():
            K = pl.group_kernels([group(s=2.0)], None)[0, 0].numpy()
        out.append((sg, rel_l1(K, cell_ref(sg, 2.0, 31))))
    assert out[1][1] < 1e-3
    # learnable table: starts at the Gaussian, positive, unit sum, gradients reach theta, stays positive after a random step
    pn = gauss_psf(1.0, 7.3, learn=False)
    pl = gauss_psf(1.0, 7.3, learn=True)
    g = group(s=1.7, theta=33.0)
    with torch.no_grad():
        Kn, Kl = pn.group_kernels([g], None), pl.group_kernels([g], None)
    e_init = float((Kn - Kl).abs().sum() / Kn.abs().sum())
    assert e_init < 1e-5, e_init
    th = pl.tab["0_0"]
    assert th.requires_grad and not any(p is th for p in pl.nets.parameters())
    w = torch.randn(1, 1, 31, 31)
    (pl.group_kernels([g], None) * w).sum().backward()
    gr = th.grad
    assert gr is not None and torch.isfinite(gr).all() and gr.abs().sum() > 0
    with torch.no_grad():
        th.add_(0.5 * torch.randn_like(th))
        Kp = pl.group_kernels([g], None)
        Tl = pl._table(0, 0)
    neg = float(Kp.min()) / float(Kp.max())  # cubic B-spline interpolation of a rough positive table can undershoot slightly
    assert neg > -0.05 and abs(float(Kp.sum()) - 1) < 1e-5 and float(Tl.min()) > 0 and abs(float(Tl.sum()) - 1) < 1e-5
    # gauge and TV penalties: ~0 at the symmetric start, positive after noise
    pl2 = gauss_psf(1.0, 7.3, learn=True, tv=1.0)
    pl2.centre_weight = 1.0
    pl2.tv = 0.0
    p0 = float(pl2.penalty().detach())
    pl2.tv = 1.0
    ptv0 = float(pl2.penalty().detach())
    with torch.no_grad():
        pl2.tab["0_0"].add_(0.3 * torch.randn_like(pl2.tab["0_0"]))
    p1 = float(pl2.penalty().detach())
    assert p0 < 1e-12 and p1 > ptv0 and math.isfinite(p1), (p0, ptv0, p1)
    d = pl2.detector_dump()["D/F150W"]
    assert "table" in d and abs(d["table"].sum() - 1) < 1e-5 and d["table"].shape == T.shape
    assert len(gauss_psf(1.0, 7.3, learn=True, res_on=False).nets) == 0 and len(gauss_psf(1.0, 7.3, learn=True, res_on=True).nets) == len(S.REG.channels)
    print(f"(f) PASS: gauss table vs analytic rel L1 {e_tab:.1e}; differentiable prefilter vs scipy max rel diff {e_sp:.1e}; learnable-table cell-mean error "
          f"sigma 1.2: {out[0][1]:.2e}, sigma 1.5: {out[1][1]:.2e}; start kernel == fixed-table kernel (rel L1 {e_init:.1e}); theta grad finite and nonzero; "
          f"table positive, unit sum; sky kernel min/max {neg:.1e} (spline undershoot), sum {float(Kp.sum()):.6f} after a random theta step; gauge penalty at start {p0:.1e}, tv+gauge {ptv0:.2e} -> {p1:.2e} after noise")


# ------------------------------------------------------------------------------------------------------------------------------ (g)
def test_g():
    out = os.path.join(ROOT, "runs", "test_v36_smoke")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "smacs_four_v36.py"), "--out", out] + SMOKE_BASE + [
        "--psf-frame", "detector", "--psf-model", "lib", "--explicit-psf", "lib", "--lib-source", "gauss", "--lib-learn-table", "--lib-res", "off",
        "--psf-window-arcsec", "1.0", "--sky-init", "deconv", "--rl-iters", "5", "--sample-mode", "bicubic", "--scale", "0.04", "--halo-px", "40",
        "--spectral", "rbf", "--svo-cache", SVO, "--spec-q", "3", "--skip-implicit", "--explicit-configs", "explicit", "--psf-centre", "1.0", "--lib-table-tv", "0.01"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    print(f"    smoke run: exit {p.returncode} in {time.time() - t0:.0f} s")
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:])
    assert p.returncode == 0
    M = json.load(open(os.path.join(out, "metrics.json")))
    keys = [k for k in M["metrics"]["heldout_chi2"] if "lib" in k]
    assert keys
    vals = []
    for k in keys:
        for key in ("heldout_chi2", "heldout_chi2_sources", "heldout_robust", "heldout_robust_sources"):
            v = M["metrics"][key][k]
            assert v is not None and math.isfinite(v), (key, k, v)
            vals.append(v)
    g = np.load(os.path.join(out, f"{keys[0]}_psf_groups.npy"), allow_pickle=True).item()
    df = g["detector_frame"]
    assert all("table" in v and abs(v["table"].sum() - 1) < 1e-4 for v in df.values()) and len(df) == 6
    st = torch.load(os.path.join(out, f"{keys[0]}_state.pt"), weights_only=False)
    assert any(k.startswith("psf.tab.") for k in st) and not any(k.startswith("psf.nets.") for k in st)
    assert M["lib_psf"]["source"] == "gauss" and M["lib_psf"]["learn_table"] and not M["lib_psf"]["residual_networks"]
    moved = max(float(np.abs(v["table"] - v["library"]).sum()) for v in df.values())
    assert moved > 0
    for ln in p.stdout.splitlines():
        if any(w in ln for w in ("data-driven arm", "TEST explicit", "Richardson")):
            print("    |", ln[:250])
    print(f"(g) PASS: exit 0, finite held-out scores {['%.4f' % v for v in vals]}, learned tables in the dump {sorted(df)}, state has psf.tab.*, no psf.nets.*; max L1 table change {moved:.3f}")


# ------------------------------------------------------------------------------------------------------------------------------ (h)
def ee(k, os_, radii):
    N = k.shape[-1]
    ax = (np.arange(N) - N // 2) / os_
    x, y = np.meshgrid(ax, ax)
    r = np.hypot(x, y)
    return [float(k[r <= rr].sum() / k.sum()) for rr in radii]


def khat(K, P=128):
    pad = np.zeros((P, P))
    n = K.shape[0]
    pad[:n, :n] = K
    pad = np.roll(pad, (-(n // 2), -(n // 2)), (0, 1))
    return np.fft.fft2(pad)


def common_mode(Kt, Kl, fmax=0.25, thr=0.05):
    """Is K_learned_i * g = K_true_i for ONE g across instruments? Kt, Kl: lists of sky-grid kernels. Residuals relative to |K_true|:
    raw (g = 1), free per-frequency g (least squares over the instruments) and Gaussian g = exp(-2 pi^2 s2 f^2) (one parameter s2, sky px^2)."""
    P = 128
    f = np.fft.fftfreq(P)
    fr = np.hypot(f[:, None], f[None, :])
    Th, Lh = [khat(k) for k in Kt], [khat(k) for k in Kl]
    mask = (fr <= fmax) & np.all([np.abs(t) > thr for t in Th], 0)
    den = sum(np.abs(l) ** 2 for l in Lh)
    gfree = sum(t * np.conj(l) for t, l in zip(Th, Lh)) / np.maximum(den, 1e-30)
    norm = sum((np.abs(t[mask]) ** 2).sum() for t in Th)

    def res(g):
        return math.sqrt(sum((np.abs(t[mask] - g[mask] * l[mask]) ** 2).sum() for t, l in zip(Th, Lh)) / norm)
    best = min(((res(np.exp(-2 * math.pi ** 2 * s2 * fr ** 2)), s2) for s2 in np.arange(-4.0, 4.0, 0.02)))
    return {"raw": res(np.ones_like(fr)), "free_g": res(gfree), "gauss_g": best[0], "gauss_s2": float(best[1])}


def rel_kernel_err(Kt, Kl, s, fmax=0.2, thr=0.1):
    P = 128
    f = np.fft.fftfreq(P)
    fr = np.hypot(f[:, None], f[None, :])
    Th, Lh = [khat(k) for k in Kt], [khat(k) for k in Kl]
    mask = (fr <= fmax) & (np.abs(Th[s]) > thr) & (np.abs(Lh[s]) > thr)
    return [None if i == s else float(np.linalg.norm((Lh[i] / Lh[s] - Th[i] / Th[s])[mask]) / np.linalg.norm((Th[i] / Th[s])[mask])) for i in range(len(Kt))]


def run_multi(insts, start, steps=1000, seed=0, sky_lr=0.03, tab_lr=1e-2):
    """Data-driven arm diagnostic. insts: list of dict(name, band, pix [sky px per detector px], sig [effective sigma, detector px], theta [deg]).
    One shared sky; every instrument has a learnable table (generic Gaussian sigma 1.0 detector px start, no residual networks)."""
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    H = W = 72
    n = 41
    N = 201
    nI = len(insts)
    sky = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    pts = gen.uniform(14, 58, size=(14, 2))
    for (px, py), f in zip(pts, gen.uniform(300, 2000, 14)):
        sky += f * (np.abs(xx - round(px)) + np.abs(yy - round(py)) == 0)
    sky += 40.0 * np.exp(-0.5 * (((xx - 36) / 7.0) ** 2 + ((yy - 30) / 5.0) ** 2))
    sky += 25.0 * np.exp(-0.5 * (((np.hypot(xx - 28, yy - 44)) - 8.0) / 1.5) ** 2)
    sky_t = torch.from_numpy(sky)
    bands = [i["band"] for i in insts]
    names = [i["name"] for i in insts]
    setup_reg(bands)
    groups = [group(b=k, s=i["pix"], det=k, theta=i["theta"]) for k, i in enumerate(insts)]
    pairs = {(k, k) for k in range(nI)}
    truth = {(i["name"], i["band"]): epsf_array(N, i["sig"], wing=0.004)[None] for i in insts}
    tpsf = S.LibPSF(bands, names, pairs, FakeLib(truth), None, 0.04, n=n, hidden=8, res_on=False)
    with torch.no_grad():
        KT = tpsf.group_kernels(groups, None)
        CBt = [S.convolve_basis(sky_t[None], KT[i:i + 1])[0, 0] for i in range(nI)]

    def positions(inst, seed_):
        g_ = np.random.default_rng(seed_)
        pix, th = inst["pix"], math.radians(inst["theta"])
        ncol = int(math.ceil(1.5 * H / pix))
        c = (np.arange(ncol) - ncol / 2 + 0.5) * pix
        X, Y = np.meshgrid(c, c)
        sh = g_.uniform(-0.5 * pix, 0.5 * pix, 2)
        px = (H - 1) / 2 + math.cos(th) * X - math.sin(th) * Y + sh[0]
        py = (H - 1) / 2 + math.sin(th) * X + math.cos(th) * Y + sh[1]
        keep = (px > 9) & (px < W - 10) & (py > 9) & (py < H - 10)
        return torch.from_numpy(px[keep]).float(), torch.from_numpy(py[keep]).float()
    exps, held = [], []
    for k, inst in enumerate(insts):
        for j in range(10):
            x, y = positions(inst, 100 * k + j)
            d = S.sample_bicubic(CBt[k], x, y) + torch.randn(x.shape[0], generator=torch.Generator().manual_seed(1000 * k + j))
            exps.append({"inst": k, "x": x, "y": y, "d": d})
        for j in range(3):  # held-out dithers: noise-free truth
            x, y = positions(inst, 5000 + 100 * k + j)
            held.append({"inst": k, "x": x, "y": y, "d": S.sample_bicubic(CBt[k], x, y)})

    def splat(sel):
        num, den = torch.zeros(H * W), torch.zeros(H * W)
        for e in sel:
            x, y = e["x"], e["y"]
            x0, y0 = x.floor().long(), y.floor().long()
            for dx_ in (0, 1):
                for dy_ in (0, 1):
                    wgt = (1 - (x - x0 - dx_).abs()) * (1 - (y - y0 - dy_).abs())
                    ok = (x0 + dx_ >= 0) & (x0 + dx_ < W) & (y0 + dy_ >= 0) & (y0 + dy_ < H)
                    idx = (y0 + dy_)[ok] * W + (x0 + dx_)[ok]
                    num.index_add_(0, idx, (wgt * e["d"])[ok])
                    den.index_add_(0, idx, wgt[ok])
        return (num / den.clamp(min=1e-3)).view(H, W)
    coadd = [splat([e for e in exps if e["inst"] == i]) for i in range(nI)]
    Rtab = {(i["name"], i["band"]): (n // 2) / i["pix"] for i in insts}
    psf = S.LibPSF(bands, names, pairs, S.GaussLibrary(1.0, 4, Rtab), None, 0.04, n=n, learn=True, res_on=False)
    with torch.no_grad():
        K_start = psf.group_kernels(groups, None).clone()
    if start == "rl":
        S.PSFG = S.PSFGroups()
        S.PSFG.groups = [dict(g_, id=k, band=bands[k]) for k, g_ in enumerate(groups)]
        S.PSFG.n_train = nI
        stats = [(c.median(), c.std()) for c in coadd]
        rl = S.deconv_init(torch.stack(coadd), stats, torch.ones(nI), psf, SimpleNamespace(rl_iters=30), CPU)
        init = rl.mean(0)
    else:
        init = splat(exps)
    sky0 = init.clone()[None]
    scale = 0.1 * (sky0.abs() + 3.0)
    delta = torch.nn.Parameter(torch.zeros_like(sky0))

    def smk(a_):
        k_ = torch.exp(-0.5 * (torch.arange(-8, 9).float() / 2.0) ** 2)
        k_ = k_[:, None] * k_[None, :]
        return torch.nn.functional.conv2d(a_[None, None], (k_ / k_.sum())[None, None], padding=8)[0, 0]
    skyerr = lambda s_: float((smk(s_) - smk(sky_t)).norm() / smk(sky_t).norm())
    err0 = skyerr(init)
    opt = torch.optim.Adam([{"params": [delta], "lr": sky_lr}, {"params": list(psf.tab.values()), "lr": tab_lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for it in range(steps):
        opt.zero_grad()
        K = psf.group_kernels(groups, None)
        sk = (sky0 + scale * delta)[0]
        loss, cnt = 0.0, 0
        CB = [S.convolve_basis(sk[None], K[i:i + 1])[0, 0] for i in range(nI)]
        for e in exps:
            r = e["d"] - S.sample_bicubic(CB[e["inst"]], e["x"], e["y"])
            loss = loss + (r * r).sum()
            cnt += r.numel()
        loss = loss / cnt
        loss.backward()
        opt.step()
        sched.step()
    chi2 = float(loss.detach())

    def hold(CBs_):
        return [float(sum(((S.sample_bicubic(CBs_[e["inst"]], e["x"], e["y"]) - e["d"]) ** 2).sum() for e in held if e["inst"] == i).sqrt()
                      / sum((e["d"] ** 2).sum() for e in held if e["inst"] == i).sqrt()) for i in range(nI)]
    with torch.no_grad():
        sk = (sky0 + scale * delta)[0]
        Kl = psf.group_kernels(groups, None)
        hp = hold([S.convolve_basis(sk[None], Kl[i:i + 1])[0, 0] for i in range(nI)])
        hp0 = hold([S.convolve_basis(init[None], K_start[i:i + 1])[0, 0] for i in range(nI)])
    dmp = psf.detector_dump()
    rows = []
    for k, inst in enumerate(insts):
        key = f"{inst['name']}/{inst['band']}"
        Rw = dmp[key]["half_window_det_px"]
        Nt = dmp[key]["learned"].shape[0]
        c0 = N // 2 - Nt // 2
        trc = truth[(inst["name"], inst["band"])][0][c0:c0 + Nt, c0:c0 + Nt]
        ax = (np.arange(Nt) - Nt // 2) / 4.0
        win = (np.abs(ax)[None, :] <= Rw) & (np.abs(ax)[:, None] <= Rw)
        rows.append({"inst": key, "truth": ee(trc * win, 4, [1.0, 2.0]), "start": ee(dmp[key]["library"], 4, [1.0, 2.0]), "learned": ee(dmp[key]["learned"], 4, [1.0, 2.0])})
    Kt_l = [KT[i, 0].numpy() for i in range(nI)]
    Kl_l = [Kl[i, 0].numpy() for i in range(nI)]
    Ks_l = [K_start[i, 0].numpy() for i in range(nI)]
    r_ = np.arange(n) - n // 2
    sharp = int(np.argmin([math.sqrt((((r_[None, :] ** 2 + r_[:, None] ** 2) * k).sum()) / k.sum()) for k in Kt_l]))
    return {"rows": rows, "chi2": chi2, "skyerr": (err0, skyerr(sk)), "heldout_start": hp0, "heldout": hp, "common": common_mode(Kt_l, Kl_l),
            "common_start": common_mode(Kt_l, Ks_l), "sharp": names[sharp], "relk": rel_kernel_err(Kt_l, Kl_l, sharp), "relk_start": rel_kernel_err(Kt_l, Ks_l, sharp),
            "dt": time.time() - t0}


CASES = {
    "2 instr, drizzle start": ([dict(name="A", band="F150W", pix=1.0, sig=1.5, theta=0.0), dict(name="B", band="F277W", pix=3.0, sig=0.75, theta=0.0)], "drizzle"),
    "2 instr, RL start": ([dict(name="A", band="F150W", pix=1.0, sig=1.5, theta=0.0), dict(name="B", band="F277W", pix=3.0, sig=0.75, theta=0.0)], "rl"),
    "2 instr, rolls 0 and 70 deg, RL start": ([dict(name="A", band="F150W", pix=1.0, sig=1.5, theta=0.0), dict(name="B", band="F277W", pix=3.0, sig=0.75, theta=70.0)], "rl"),
    "3 instr, RL start": ([dict(name="A", band="F150W", pix=1.0, sig=1.5, theta=0.0), dict(name="B", band="F277W", pix=3.0, sig=0.75, theta=0.0),
                           dict(name="C", band="F444W", pix=2.0, sig=1.1, theta=0.0)], "rl"),
}


def test_h(steps=1000, only=None):
    for lab, (insts, start) in CASES.items():
        if only and lab not in only:
            continue
        r = run_multi(insts, start, steps)
        e0, e1 = r["skyerr"]
        print(f"    [{lab}] {r['dt']:.0f} s, final chi2/N {r['chi2']:.3f}, sky rel. L2 error (2 px smoothed) {e0:.3f} -> {e1:.3f}")
        for w in r["rows"]:
            print(f"        {w['inst']}: EE(1 px) truth {w['truth'][0]:.3f} start {w['start'][0]:.3f} learned {w['learned'][0]:.3f} | "
                  f"EE(2 px) truth {w['truth'][1]:.3f} start {w['start'][1]:.3f} learned {w['learned'][1]:.3f}")
        print(f"        held-out dithers, rel. L2 prediction error per instrument: start {['%.3f' % v for v in r['heldout_start']]} -> learned {['%.3f' % v for v in r['heldout']]}")
        c, c0 = r["common"], r["common_start"]
        print(f"        common-mode (sky-frame kernels, f <= 0.25 cyc/px), residual of K_true vs g*K_learned: raw {c['raw']:.3f}, one free g(f) {c['free_g']:.3f}, "
              f"one Gaussian g (s2 = {c['gauss_s2']:+.2f} px^2) {c['gauss_g']:.3f}   [start kernels: raw {c0['raw']:.3f}, free g {c0['free_g']:.3f}, Gaussian g {c0['gauss_g']:.3f}]")
        print(f"        relative kernels K_i/K_{r['sharp']} (Fourier, where |K_{r['sharp']}| > 0.1), rel. error: learned {['-' if v is None else '%.3f' % v for v in r['relk']]} "
              f"vs start {['-' if v is None else '%.3f' % v for v in r['relk_start']]}")
    print("(h) diagnostic complete (no pass/fail criterion)")


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b", "c", "d", "f", "g", "h"]
    torch.set_num_threads(2)
    for w in which:
        globals()["test_" + w]()
    print("\nALL REQUESTED TESTS DONE")
