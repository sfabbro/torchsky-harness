"""Tests for scripts/smacs_four_v35.py (E2: library PSF warm start, RL sky start, bicubic read-out). Plain asserts, no pytest:
    python tests/test_v35_lib.py [a b c d e]
(a) synthetic PSF library in the exact npz / meta.json format (Gaussian core + power-law wing ePSFs, different per detector and band)
(b) kernel rendering: theta = 0, parity +1, M = diag(s, s): library integrated onto the sky grid (1e-3 relative L1), unit sum, centroid;
    theta = 90 deg is a rot90; parity -1 is a flip
(c) warm start: zero residual gives exactly the library kernels; gradients reach the residual networks; penalty is zero at the warm start
(d) smoke run of smacs_four_v35.py on testdata/nircam with the new flags (library nodes deliberately off by 0.01 um: nearest-node fallback)
(e) identifiability diagnostic (synthetic, CPU): two instruments, library PSF 15% too broad, sky started from the drizzle-like coadd or from RL
tests/check_v35_equiv.py is the bitwise comparison with v34."""
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
import smacs_four_v35 as S  # noqa: E402

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
        cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "smacs_four_v35.py"), "--out", out] + SMOKE_BASE + [
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


# ------------------------------------------------------------------------------------------------------------------------------ (e)
def ee(k, os_, radii):
    N = k.shape[-1]
    ax = (np.arange(N) - N // 2) / os_
    x, y = np.meshgrid(ax, ax)
    r = np.hypot(x, y)
    return [float(k[r <= rr].sum() / k.sum()) for rr in radii]


def run_ident(start, shared, steps, seed=0, verbose=True, psf_lr=1e-3, psf_true=False, sky_lr=0.03, optimizer="adam"):
    """Two instruments: A (detector pixel = 1 sky px, core sigma 1.5 px) and B (pixel = 3 sky px, effective sigma 0.75 px: undersampled).
    The library PSFs are 15% too broad. start in ('drizzle', 'rl'); shared: one sky for both instruments, else one sky per instrument."""
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    H = W = 72
    n = 41
    sigA, sigB = 1.5, 0.75
    N = 201
    truth = {("A", "F150W"): epsf_array(N, sigA, wing=0.004)[None], ("B", "F277W"): epsf_array(N, sigB, wing=0.004)[None]}
    broad = {k: epsf_array(N, 1.15 * (sigA if k[0] == "A" else sigB), wing=0.004)[None] for k in truth}
    # scene: point sources + an extended blob, units of the noise sigma
    sky = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    pts = gen.uniform(14, 58, size=(14, 2))
    for (px, py), f in zip(pts, gen.uniform(300, 2000, 14)):
        sky += f * (np.abs(xx - round(px)) + np.abs(yy - round(py)) == 0)  # delta on a sky cell
    sky += 40.0 * np.exp(-0.5 * (((xx - 36) / 7.0) ** 2 + ((yy - 30) / 5.0) ** 2))
    sky += 25.0 * np.exp(-0.5 * (((np.hypot(xx - 28, yy - 44)) - 8.0) / 1.5) ** 2)
    sky_t = torch.from_numpy(sky)
    setup_reg(["F150W", "F277W"])
    dets = ["A", "B"]
    gA, gB = group(b=0, s=1.0, det=0), group(b=1, s=3.0, det=1)
    truth_psf = S.LibPSF(S.REG.bands, dets, {(0, 0), (1, 1)}, FakeLib(truth), None, 0.04, n=n, hidden=8)
    with torch.no_grad():
        KT = truth_psf.group_kernels([gA, gB], None)
        CBt = [S.convolve_basis(sky_t[None], KT[i:i + 1])[0, 0] for i in range(2)]
    # exposures: detector pixel centres in sky-grid coordinates (dithers), noise sigma 1
    exps = []
    for inst, (g, ndith) in enumerate(((gA, 10), (gB, 10))):
        pix = g["M"][0][0]
        ncol = int(H / pix)
        for k in range(ndith):
            sh = gen.uniform(-0.5 * pix, 0.5 * pix, 2)
            c = (np.arange(ncol) + 0.5) * pix - 0.5 + sh[0]
            r = (np.arange(ncol) + 0.5) * pix - 0.5 + sh[1]
            X, Y = np.meshgrid(c, r)
            keep = (X > 9) & (X < W - 10) & (Y > 9) & (Y < H - 10)  # interior: the kernel never reads beyond the grid
            x, y = torch.from_numpy(X[keep]).float(), torch.from_numpy(Y[keep]).float()
            d = S.sample_bicubic(CBt[inst], x, y) + torch.randn(x.shape[0], generator=torch.Generator().manual_seed(1000 * inst + k))
            exps.append({"inst": inst, "x": x, "y": y, "d": d})
    # drizzle-like coadd per instrument: bilinear splat of the pixel values onto the sky grid, weighted mean (pixel area weighting for B)
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
        return (num / den.clamp(min=1e-3)).view(H, W), den.view(H, W)
    coadd = [splat([e for e in exps if e["inst"] == i])[0] for i in range(2)]
    all_coadd = splat(exps)[0]
    # RL start
    lib_psf = S.LibPSF(S.REG.bands, dets, {(0, 0), (1, 1)}, FakeLib(broad), None, 0.04, n=n, hidden=a_hidden)
    S.PSFG = S.PSFGroups()
    S.PSFG.groups = [dict(gA, id=0, band="F150W"), dict(gB, id=1, band="F277W")]
    S.PSFG.n_train = 2
    stats = [(c.median(), c.std()) for c in coadd]
    sig_b = torch.ones(2)
    rl = S.deconv_init(torch.stack(coadd), stats, sig_b, lib_psf, SimpleNamespace(rl_iters=30), CPU)
    if shared:
        inits = {"drizzle": all_coadd[None], "rl": rl.mean(0, keepdim=True), "truth": sky_t[None]}[start]
    else:
        inits = {"drizzle": torch.stack(coadd), "rl": rl, "truth": torch.stack([sky_t, sky_t])}[start]
    sky0 = inits.clone()
    skyscale = 0.1 * (sky0.abs() + 3.0)  # per-pixel preconditioning of the Adam step: sky = start + scale * delta
    delta = torch.nn.Parameter(torch.zeros_like(sky0))
    class _Sky:  # light wrapper so the closure and the error code can read the current sky
        @property
        def val(self):
            return sky0 + skyscale * delta
    skyp_obj = _Sky()
    def sm0(a_):
        k_ = torch.exp(-0.5 * (torch.arange(-8, 9).float() / 2.0) ** 2)
        k_ = k_[:, None] * k_[None, :]
        return torch.nn.functional.conv2d(a_[None, None], (k_ / k_.sum())[None, None], padding=8)[0, 0]
    err0 = float((sm0((inits[0] if shared else inits.mean(0)).detach()) - sm0(sky_t)).norm() / sm0(sky_t).norm())
    psf = S.LibPSF(S.REG.bands, dets, {(0, 0), (1, 1)}, FakeLib(truth if psf_true else broad), None, 0.04, n=n, hidden=a_hidden)
    gl = [gA, gB]
    t0 = time.time()
    hist = []

    def closure():
        opt.zero_grad()
        K = psf.group_kernels(gl, None)
        loss, nn_ = 0.0, 0
        for i in range(2):
            CB = S.convolve_basis(skyp_obj.val[0 if shared else i][None], K[i:i + 1])[0, 0]
            for e in [e for e in exps if e["inst"] == i]:
                r = e["d"] - S.sample_bicubic(CB, e["x"], e["y"])
                loss = loss + (r * r).sum()
                nn_ += r.numel()
        loss = loss / nn_
        loss.backward()
        return loss

    params = [delta] + ([] if psf_lr <= 0 else list(psf.parameters()))
    if optimizer == "lbfgs":
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=steps // 10, history_size=30, line_search_fn="strong_wolfe")
        for outer in range(10):
            l_ = opt.step(closure)
            hist.append((outer, float(l_.detach())))
    else:
        opt = torch.optim.Adam([{"params": [delta], "lr": sky_lr}, {"params": list(psf.parameters()), "lr": max(psf_lr, 1e-12)}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for it in range(steps):
            l_ = opt.step(closure)
            sched.step()
            if it % max(steps // 6, 1) == 0 or it == steps - 1:
                hist.append((it, float(l_.detach())))
    dmp = psf.detector_dump()
    res = {}
    for (kk, name) in ((("A", "F150W"), "A/F150W"), (("B", "F277W"), "B/F277W")):
        key = f"{kk[0]}/{kk[1]}"
        rad = [1.0, 2.0]
        Rw = dmp[key]["half_window_det_px"]  # EE normalised inside the trained window for all three kernels
        ax = (np.arange(N) - N // 2) / OS
        win = (np.abs(ax)[None, :] <= Rw) & (np.abs(ax)[:, None] <= Rw)
        res[key] = {"truth": ee(truth[kk][0] * win, OS, rad), "library": ee(broad[kk][0] * win, OS, rad), "learned": ee(dmp[key]["learned"], OS, rad)}
    # sky error: relative L2 of the model sky against the truth, after convolving both with a common 2 px Gaussian (the resolution the data allow)
    def sm(a_):
        k_ = torch.exp(-0.5 * (torch.arange(-8, 9).float() / 2.0) ** 2)
        k_ = (k_[:, None] * k_[None, :])
        k_ = k_ / k_.sum()
        return torch.nn.functional.conv2d(a_[None, None], k_[None, None], padding=8)[0, 0]
    skyv = skyp_obj.val
    skyf = skyv[0] if shared else skyv.mean(0)
    err = float((sm(skyf.detach()) - sm(sky_t)).norm() / sm(sky_t).norm())
    return res, hist, time.time() - t0, (err0, err)


a_hidden = 16


def test_e(steps=700):
    out = {}
    for shared in (True, False):
        for start in ("drizzle", "rl"):
            res, hist, dt, err = run_ident(start, shared, steps)
            out[(shared, start)] = res
            lab = "shared sky" if shared else "separate skies"
            print(f"    [{lab}, sky start {start:7s}] {dt:.0f} s, loss/N {hist[0][1]:.3f} -> {hist[-1][1]:.3f}, sky rel. L2 error (2 px smoothed) at start {err[0]:.3f} -> end {err[1]:.3f}")
            for key, v in res.items():
                print(f"        {key}: EE(1 px) truth {v['truth'][0]:.3f} library {v['library'][0]:.3f} learned {v['learned'][0]:.3f} | "
                      f"EE(2 px) truth {v['truth'][1]:.3f} library {v['library'][1]:.3f} learned {v['learned'][1]:.3f}")
    print("(e) diagnostic complete (no pass/fail criterion)")
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b", "c", "d", "e"]
    torch.set_num_threads(2)
    for w in which:
        globals()["test_" + w]()
    print("\nALL REQUESTED TESTS DONE")
