"""Tests for scripts/smacs_four_v39.py --lib-chromatic {none,endpoints,nodes} (see scripts/patch_v39.py). Plain asserts, no pytest:
    python tests/test_v39_chrom.py [a b c d e]
(a) endpoints: the first / last node tables are exactly the two free tables, the inner ones linear in lambda; nodes: every node has its own table;
    all free tables start identical; unit sum and positivity
(b) gradients reach every free table (nodes: Q per pair; endpoints: 2 per pair); the smoothness penalty is 0 at the start, > 0 after a perturbation,
    scales with the weight, and is absent for endpoints / none; the centroid gauge acts on every free table
(c) synthetic chromatic truth (Gaussian, sigma proportional to lambda, 25 per cent across the band): kernel-level fit; nodes and endpoints fit it far
    better than one achromatic table
(d) parameter counts and GPU-memory estimate for the real configuration (6 NIRCam bands on 5 detectors + 4 WFC3/IR bands, spec_q 6), built from the
    real LibPSF class with the generic Gaussian tables
(e) quick testdata smoke runs of each mode (E2 invocation as tests/test_v36_lib.py (g)): exit 0, finite held-out scores, per-node tables in the dump
APPROXIMATIONS: (c) fits the KERNELS (no data, no noise, no sky); it shows capacity, not the real-data gain. (d) is an estimate: the rest of the
model (sky, caches, activations of the rest of the model) was not measured here (no GPU)."""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import smacs_four_v39 as S  # noqa: E402

NIR = os.path.join(ROOT, "testdata", "nircam")
SVO = os.path.join(ROOT, "testdata", "svo_cache")
OS = 4
LAMQ = [3.9578, 4.1177, 4.2832, 4.4593, 4.6524, 4.8787]  # F444W run nodes (6)


class FakeLib:
    def __init__(self, arrays, oversample=OS):
        self.arrays, self.oversample, self.notes = arrays, oversample, []

    def get(self, detname, band, channel):
        a = self.arrays[(detname, band)]
        return a / a.sum((-2, -1), keepdims=True), None


def gauss_arr(N, sig):
    ax = (np.arange(N) - N // 2) / OS
    x, y = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * (x * x + y * y) / sig ** 2)
    return (k / k.sum()).astype(np.float32)


def mk(chrom, bands=("F444W",), ndet=1, N=121, sig=1.5, smooth=1.0, n=31, nodes=LAMQ, res_on=False):
    S.set_registry(S.build_registry(list(bands), psf_n=n))
    S.PSF_FRAME = "detector"
    dets = [f"D{i}" for i in range(ndet)]
    arrs = {(d, b): gauss_arr(N, sig)[None] for d in dets for b in bands}
    pairs = {(i, j) for i in range(ndet) for j in range(len(bands))}
    ln = torch.tensor([list(nodes)] * len(bands))
    return S.LibPSF(list(S.REG.bands), dets, pairs, FakeLib(arrs), ln, 0.04, n=n, hidden=8, learn=True, res_on=res_on, chrom=chrom, chrom_smooth=smooth), ln


def group(b=0, s=2.0, det=0):
    return {"id": 0, "b": b, "theta_deg": 0.0, "parity": 1, "M": [[s, 0.0], [0.0, s]], "sx": s, "sy": s, "S": 1, "det": det}


def theta_of(T):
    return torch.log(torch.expm1(T.clamp(min=1e-8 * float(T.max()))))


def test_a():
    for mode in ("nodes", "endpoints"):
        psf, ln = mk(mode)
        Q = len(LAMQ)
        keys = sorted(psf.tabq.keys())
        want = [f"0_0_{q}" for q in range(Q)] if mode == "nodes" else ["0_0_0", f"0_0_{Q - 1}"]
        assert keys == sorted(want), keys
        assert len(psf.tab) == 0
        for k in keys[1:]:
            assert torch.equal(psf.tabq[keys[0]], psf.tabq[k])  # same start
        with torch.no_grad():
            for k in psf.tabq:  # distinct values so that interpolation is testable
                psf.tabq[k].add_(torch.randn_like(psf.tabq[k]) * 0.3)
            T = [psf._table(0, 0, q) for q in range(Q)]
        for t in T:
            assert float(t.min()) > 0 and abs(float(t.sum()) - 1) < 1e-5
        if mode == "endpoints":
            with torch.no_grad():
                A, B = psf._unit("0_0_0"), psf._unit(f"0_0_{Q - 1}")
            assert torch.equal(T[0], A) and torch.equal(T[-1], B), "endpoints must reproduce the two endpoint tables exactly"
            for q in range(1, Q - 1):
                lq = psf.lam_q[0]  # the run's node wavelengths (float32 values of the tensor)
                w = (lq[q] - lq[0]) / (lq[-1] - lq[0])
                assert float((T[q] - ((1 - w) * A + w * B)).abs().max()) < 1e-9
                assert abs(psf._wint(0, q) - w) < 1e-12
        else:
            assert len({id(psf.tabq[k]) for k in keys}) == Q
    print("(a) PASS: nodes: Q own tables, endpoints: exact end tables + linear-in-lambda inner tables, same start, unit sum, positive")


def test_b():
    n_nodes = len(LAMQ)
    for mode in ("nodes", "endpoints"):
        psf, ln = mk(mode, bands=("F444W",), ndet=2)
        psf.centre_weight = 1.0
        out = psf.group_kernels([group(det=0), group(det=1)], ln)
        assert out.shape[0] == 2 * n_nodes
        loss = sum(((out[i, 0] * torch.randn(out.shape[-2:])).sum()) for i in range(out.shape[0]))
        loss.backward()
        miss = [k for k, p in psf.tabq.items() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        assert not miss, ("no gradient reaches", miss)
        assert len(psf.tabq) == (2 * n_nodes if mode == "nodes" else 4)
        with torch.no_grad():
            p0 = float(psf.penalty())
        assert abs(p0) < 1e-12, p0  # equal tables, centred Gaussians: zero penalty
        with torch.no_grad():
            for k in psf.tabq:
                psf.tabq[k].add_(torch.randn_like(psf.tabq[k]) * 0.2)
            psf.centre_weight = 0.0
            psf.chrom_smooth = 1.0
            p1 = float(psf.penalty())
            psf.chrom_smooth = 4.0
            p4 = float(psf.penalty())
            psf.chrom_smooth = 0.0
            p0w = float(psf.penalty())
        if mode == "nodes":
            assert p1 > 0 and abs(p4 - 4 * p1) < 1e-5 * max(1.0, p4) and abs(p0w) < 1e-12, (p1, p4, p0w)
        else:
            assert abs(p4) < 1e-12 and abs(p1) < 1e-12 and abs(p0w) < 1e-12  # no node smoothness term for endpoints
        # the centroid gauge acts on every free table: shifting any one of them raises the penalty
        psf.centre_weight = 1.0
        for k0 in sorted(psf.tabq):
            with torch.no_grad():
                before = float(psf.penalty())
                saved = psf.tabq[k0].clone()
                psf.tabq[k0].copy_(torch.roll(psf.tabq[k0], 8, dims=1))
                after = float(psf.penalty())
                psf.tabq[k0].copy_(saved)
            assert after > before + 1e-7, (mode, k0, before, after)
    psf, ln = mk("none")
    assert len(psf.tabq) == 0 and len(psf.tab) == 1
    print("(b) PASS: gradients reach all free tables (nodes 2x6, endpoints 2x2); smoothness: 0 at start, linear in the weight (nodes), absent (endpoints); centroid gauge on every free table")


def kernel_fit(mode, truth, ln, steps=250, lr=0.1):
    psf, _ = mk(mode, sig=1.5)
    g = group()
    opt = torch.optim.Adam([p for p in psf.tabq.values()] + [p for p in psf.tab.values()], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        K = psf.group_kernels([g], ln)
        loss = ((K - truth) ** 2).sum() / (truth ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        K = psf.group_kernels([g], ln)
        return float(((K - truth) ** 2).sum() / (truth ** 2).sum()), K


def test_c():
    Q = len(LAMQ)
    # truth: Gaussian with sigma proportional to lambda (1.3 -> 1.6 detector px; +23 per cent) built through the model itself (nodes mode)
    psf_t, ln = mk("nodes")
    with torch.no_grad():
        for q in range(Q):
            sig = 1.3 * LAMQ[q] / LAMQ[0]
            psf_t.tabq[f"0_0_{q}"].copy_(theta_of(torch.from_numpy(gauss_arr(121, sig)).double()).float())
        truth = psf_t.group_kernels([group()], ln)
    res = {}
    for mode in ("none", "endpoints", "nodes"):
        res[mode], K = kernel_fit(mode, truth, ln)
    print(f"    relative kernel error (L2^2) after the fit: none {res['none']:.2e}, endpoints {res['endpoints']:.2e}, nodes {res['nodes']:.2e}")
    assert res["endpoints"] < res["none"] / 3 and res["nodes"] < res["none"] / 3
    print("(c) PASS: a chromatic truth (sigma proportional to lambda, +23 per cent across the band) is fit better by endpoints and nodes than by one table")


def test_d():
    """Real configuration: NRCB1-4 x {F090W, F150W, F200W}, NRCBLONG x {F277W, F356W, F444W}, WFC3/IR x 4 bands; spec_q 6; scale 0.04, 3 arcsec window."""
    hst = [{"band": f"WFC3IR_{f}", "instrument": "WFC3/IR", "filter": f, "photplam": p, "svo_id": None}
           for f, p in (("F105W", 10552.0), ("F125W", 12486.0), ("F140W", 13923.0), ("F160W", 15369.0))]
    S.set_registry(S.build_registry(["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"], hst, psf_n=75))
    S.PSF_FRAME = "detector"
    bands = list(S.REG.bands)
    dets = ["NRCB1", "NRCB2", "NRCB3", "NRCB4", "NRCBLONG", "WFC3_IR"]
    pairs = set()
    for b, band in enumerate(bands):
        if band in ("F090W", "F150W", "F200W"):
            pairs |= {(d, b) for d in range(4)}
        elif band in ("F277W", "F356W", "F444W"):
            pairs.add((4, b))
        else:
            pairs.add((5, b))
    pix = {"NRCB": 0.031, "NRCBLONG": 0.063, "WFC3_IR": 0.128}
    scale, n = 0.04, 75
    Rtab = {}
    for d, b in pairs:
        p = pix["NRCBLONG"] if dets[d] == "NRCBLONG" else (pix["WFC3_IR"] if dets[d] == "WFC3_IR" else pix["NRCB"])
        Rtab[(dets[d], bands[b])] = (n // 2) / (p / scale)
    lib = S.GaussLibrary(1.0, 4, Rtab)
    Q = 6
    ln = torch.stack([torch.linspace(S.REG.lam_um[b] * 0.93, S.REG.lam_um[b] * 1.07, Q) for b in bands])
    counts = {}
    for mode in ("none", "endpoints", "nodes"):
        psf = S.LibPSF(bands, dets, pairs, lib, ln, scale, n=n, hidden=8, learn=True, res_on=False, chrom=mode)
        counts[mode] = sum(p.numel() for p in list(psf.tab.values()) + list(psf.tabq.values()))
    sizes = {(dets[d], bands[b]): int(psf._L[(d, b)].shape[-1]) for d, b in pairs}
    GiB = 2 ** 30
    print(f"    pairs: {len(pairs)} (12 NIRCam SW + 3 LW + 4 WFC3/IR); table side N: SW {sizes[('NRCB1', 'F090W')]}, LW {sizes[('NRCBLONG', 'F444W')]}, "
          f"WFC3/IR {sizes[('WFC3_IR', bands[-1])]}")
    for m, c in counts.items():
        # fp32: parameter + gradient + two Adam moments = 16 B; the forward graph of the spline coefficients (softplus, normalised, P T, P T P^T: 4 N^2 floats)
        # = 16 B per parameter, once per table per call (chromatic modes compute the coefficients once per table per call, shared by the groups)
        print(f"    {m:9s}: {c:>11,d} table parameters  ->  states {16 * c / GiB:.3f} GiB, + forward graph of the coefficients {16 * c / GiB:.3f} GiB")
    assert counts["nodes"] == 6 * counts["none"] and counts["endpoints"] == 2 * counts["none"]
    tot_nodes = 2 * 16 * counts["nodes"] / GiB
    assert tot_nodes < 1.0
    print(f"(d) PASS: nodes x6, endpoints x2 of the achromatic count {counts['none']:,d}; table-related memory of 'nodes' about {tot_nodes:.2f} GiB, "
          f"endpoints about {2 * 16 * counts['endpoints'] / GiB:.2f} GiB (estimates, fp32)")


SMOKE_BASE = ["--data-dir", NIR, "--size", "128", "--quick", "--no-fallback", "--no-holdout", "--score-all", "--boot-b", "50",
              "--budget-explicit", "100000", "--budget-implicit", "100000", "--stripes", "1", "--eval-every-min", "0"]


def smoke(mode, extra=()):
    out = os.path.join(ROOT, "runs", f"test_v39_smoke_{mode}{'_res' if extra else ''}")
    os.makedirs(out, exist_ok=True)
    res = ["--lib-res", "off"] if not extra else []
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "smacs_four_v39.py"), "--out", out] + SMOKE_BASE + [
        "--psf-frame", "detector", "--psf-model", "lib", "--explicit-psf", "lib", "--lib-source", "gauss", "--lib-learn-table"] + res + [
        "--psf-window-arcsec", "1.0", "--sky-init", "deconv", "--rl-iters", "5", "--sample-mode", "bicubic", "--scale", "0.04", "--halo-px", "40",
        "--spectral", "rbf", "--svo-cache", SVO, "--spec-q", "3", "--skip-implicit", "--explicit-configs", "explicit", "--psf-centre", "1.0",
        "--lib-table-tv", "0.01", "--lib-chromatic", mode] + list(extra)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    print(f"    smoke {mode}{' ' + ' '.join(extra) if extra else ''}: exit {p.returncode} in {time.time() - t0:.0f} s")
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
    if mode == "none":
        assert all("tables_nodes" not in v for v in df.values())
    else:
        for v in df.values():
            Tn = v["tables_nodes"]
            assert Tn.shape[0] == 3 and Tn.shape[1:] == v["table"].shape and np.allclose(Tn.sum((1, 2)), 1, atol=1e-4) and len(v["lam_nodes_um"]) == 3
            assert v["free_table_nodes"] == ([0, 1, 2] if mode == "nodes" else [0, 2])
        assert M["lib_psf"]["chromatic"] == mode and M["lib_psf"]["n_free_tables"] == len(df) * (3 if mode == "nodes" else 2)
        st = torch.load(os.path.join(out, f"{keys[0]}_state.pt"), weights_only=False)
        assert sum(k.startswith("psf.tabq.") for k in st) == M["lib_psf"]["n_free_tables"]
        assert any("chromatic tables (" in ln for ln in p.stdout.splitlines())
    return vals


def test_e():
    for mode, extra in (("none", ()), ("endpoints", ()), ("nodes", ()), ("endpoints", ("--lib-res-hidden", "16"))):
        v = smoke(mode, extra)
        print(f"    {mode} {' '.join(extra)}: finite held-out scores {['%.4f' % x for x in v]}")
    print("(e) PASS: smoke runs of none, endpoints, nodes (and endpoints with residual networks on): exit 0, finite scores, per-node tables in the psf_groups dump, tabq in the state")


if __name__ == "__main__":
    todo = sys.argv[1:] or list("abcde")
    for t in todo:
        globals()[f"test_{t}"]()
    print("ALL PASS")
