"""Acceptance tests of the pyramid item rasteriser (the only rasteriser, sheared patches; sky.ItemSky._raster_pyramid, F40). CPU, one thread, float64
unless stated.

  1. accuracy: single items, larger width 0.5 to 200 nodes (log-spaced), axis ratios 1 / 2 / 5, three angles, four sub-node offsets; plus
     items exactly on both sides of every level-switch width; plus wide items centred on the grid edge and corner. Error = max over the grid
     of |pyramid - reference| / peak of the reference, the reference being the point-sampled Gaussian of covariance T = Sigma + p^2/12 I
     normalised to exact flux over the infinite lattice (tests/pyr_common.reference). Target <= 1e-3.
  2. flux: p^2 x sum of the grid = c_k to 1e-10 relative, items well inside the grid (+-7.5 effective widths).
  3. gradients: central finite differences of sum(W * raster) against the analytic (recompute-in-chunks) gradient for 300 random items on
     levels 0 to 3.
  4. float32: the accuracy sweep (subset) in float32: no NaN, worst error within 3x of float64 (or of float32 rounding, 1e-6).
Results are keyed <test>_sheared (the key of the unpruned package's sheared run, kept).
Run:  python -m pytest -q tests/test_items_pyramid.py      (numbers in tests/items_results/pyramid.json)
"""
import json
import math
import os
import sys

import pytest
import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyr_common import P, one_item, reference  # noqa: E402
from fieldop.sky import ItemSky, RBFBasis  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "items_results")
os.makedirs(OUT, exist_ok=True)
RES = os.path.join(OUT, "pyramid.json")
TARGET = 1e-3


def save(key, val):
    d = json.load(open(RES)) if os.path.exists(RES) else {}
    d[key] = val
    json.dump(d, open(RES, "w"), indent=1)


def grid_n(wa_nodes):
    return int(min(2 * math.ceil(4.5 * wa_nodes) + 41, 801)) | 1


def err_one(N, x, y, wa, wb, th, dtype=torch.float64, ref=None):
    s = one_item(N, x, y, wa, wb, th, dtype=dtype)
    with torch.no_grad():
        A = s.coeffs()[0].double()
    ref = reference(N, x, y, wa, wb, th) if ref is None else ref
    lev = int(s.plan().assign(s.widths(0))[0])
    return float((A - ref).abs().max() / ref.max()), lev, bool(torch.isfinite(A).all()), A


ANGLES = (0.0, 0.3, 1.1)
OFFS = ((0.0, 0.0), (0.5, 0.5), (0.37, -0.21), (-0.13, 0.44))


def sweep(widths, ratios, angles, offs, dtype=torch.float64):
    rows = []
    for ratio in ratios:
        for w in widths:
            wa, wb = w * P, max(w / ratio, 0.3) * P            # smaller width floored at 0.3 nodes (the default item floor is 0.4)
            N = grid_n(w)
            for th in angles:
                for ox, oy in offs:
                    x, y = ox * P, oy * P
                    ref = reference(N, x, y, wa, wb, th)
                    e, lev, fin, _ = err_one(N, x, y, wa, wb, th, dtype, ref)
                    rows.append(dict(w_nodes=w, wb_nodes=wb / P, ratio=ratio, angle=th, off=[ox, oy], level=lev, err=e, finite=fin))
    return rows


def worst(rows, key="err"):
    r = max(rows, key=lambda r: r[key])
    return {k: r[k] for k in ("w_nodes", "wb_nodes", "ratio", "angle", "off", "level", key)}


def test_accuracy_sweep():
    widths = [0.5 * 400 ** (i / 31) for i in range(32)]
    rows = sweep(widths, (1, 2, 5), ANGLES, OFFS)
    by_level = {}
    for r in rows:
        b = by_level.setdefault(r["level"], [0.0, 0])
        b[0] = max(b[0], r["err"])
        b[1] += 1
    out = {"n_cases": len(rows), "worst_pyramid": worst(rows),
           "by_level": {str(k): {"max_err_pyramid": v[0], "cases": v[1]} for k, v in sorted(by_level.items())},
           "max_err_pyramid_levels_ge1": max([r["err"] for r in rows if r["level"] >= 1], default=0.0)}
    save("accuracy_sweep_sheared", out)
    print(json.dumps(out, indent=1))
    assert all(r["finite"] for r in rows)
    assert out["worst_pyramid"]["err"] <= TARGET


def test_level_switches():
    """Both sides of every level-switch width (relative step 1e-6 in the smaller width), ratios 1 / 2 / 5; and the jump of the pyramid
    result across the switch (|below - above| / peak), which should be of the same order as the error itself."""
    s0 = one_item(41, 0, 0, P, P, 0, max_width=200 * P)
    pl = s0.plan()
    q = P * P / 12
    rows = []
    for l in range(1, pl.L + 1):
        wsw = math.sqrt(pl.thr[l] - q)                         # smaller width at the switch [arcsec]
        for ratio in (1, 2, 5):
            wa_n = ratio * wsw / P
            N = grid_n(min(wa_n, 200))
            for th in (0.0, 0.7):
                for ox, oy in ((0.0, 0.0), (0.37, -0.21)):
                    x, y = ox * P, oy * P
                    res = []
                    for f in (1 - 1e-6, 1 + 1e-6):
                        wb = wsw * f
                        wa = ratio * wb
                        ref = reference(N, x, y, wa, wb, th)
                        e, lev, fin, A = err_one(N, x, y, wa, wb, th, ref=ref)
                        res.append((e, lev, A, float(ref.max())))
                    jump = float((res[0][2] - res[1][2]).abs().max()) / res[0][3]
                    rows.append(dict(switch_level=l, wb_nodes=wsw / P, ratio=ratio, angle=th, off=[ox, oy], err_below=res[0][0],
                                     level_below=res[0][1], err_above=res[1][0], level_above=res[1][1], jump=jump))
    out = {"switch_widths_nodes": [math.sqrt(pl.thr[l] - q) / P for l in range(1, pl.L + 1)],
           "max_err_below": max(r["err_below"] for r in rows), "max_err_above": max(r["err_above"] for r in rows),
           "max_jump": max(r["jump"] for r in rows),
           "per_level": {str(l): {k: max(r[k] for r in rows if r["switch_level"] == l) for k in ("err_below", "err_above", "jump")}
                         for l in range(1, pl.L + 1)},
           "worst_row": max(rows, key=lambda r: max(r["err_below"], r["err_above"]))}
    save("level_switches_sheared", out)
    print(json.dumps(out, indent=1))
    assert all(r["level_below"] == r["switch_level"] - 1 and r["level_above"] == r["switch_level"] for r in rows)
    assert out["max_err_below"] <= TARGET and out["max_err_above"] <= TARGET and out["max_jump"] <= 2 * TARGET


def test_edges():
    """Wide items centred on the grid edge, the corner, and 3 widths outside the grid: the part inside the grid must be right (the coarse
    grids' margins hold everything that reaches the fine grid)."""
    N = 201
    h = (N - 1) / 2 * P
    rows = []
    for w in (3.0, 8.0, 20.0, 60.0, 150.0):
        for ratio, th in ((1, 0.0), (3, 0.6)):
            wa, wb = w * P, w / ratio * P
            for cx, cy in ((h, 0.0), (h, h), (-h - 0.3 * P, 0.4 * h), (h + 3 * wb, 0.0)):
                ref = reference(N, cx, cy, wa, wb, th)
                if float(ref.max()) <= 0:
                    continue
                pk = float(reference(N, 0.0, 0.0, wa, wb, th).max())     # peak of the whole item (the in-grid maximum may be a tail)
                s = one_item(N, cx, cy, wa, wb, th)
                with torch.no_grad():
                    A = s.coeffs()[0]
                rows.append(dict(w_nodes=w, ratio=ratio, centre_nodes=[cx / P, cy / P], level=int(s.plan().assign(s.widths(0))[0]),
                                 err_over_item_peak=float((A - ref).abs().max()) / pk))
    out = {"max_err_over_item_peak": max(r["err_over_item_peak"] for r in rows), "rows": rows}
    save("edges_sheared", out)
    print(json.dumps(out["max_err_over_item_peak"]))
    assert out["max_err_over_item_peak"] <= TARGET


def test_flux():
    K = 3
    rows = []
    for w in (0.5, 1.0, 2.0, 2.9, 5.0, 6.5, 12.0, 25.0, 50.0):
        for ratio, th in ((1, 0.0), (2, 0.4), (5, 1.2)):
            wa, wb = w * P, max(w / ratio, 0.3) * P
            sig = math.sqrt(wa * wa + P * P / 12) / P
            N = 2 * int(math.ceil(7.5 * sig)) + 21
            s = one_item(N, 0.31 * P, -0.17 * P, wa, wb, th, K=K)
            with torch.no_grad():
                s.lc.data[0, 0] = torch.tensor([0.0, -1.3, 2.1], dtype=torch.float64)
                s._mut += 1
                A = s.coeffs()
            flux = (A.sum((1, 2)) * P * P)
            c = torch.exp(s.lc.data[0, 0])
            rel = float(((flux - c) / c).abs().max())
            rows.append(dict(w_nodes=w, ratio=ratio, level=int(s.plan().assign(s.widths(0))[0]), rel_flux_err=rel))
    out = {"max_rel_flux_err": max(r["rel_flux_err"] for r in rows), "rows": rows}
    save("flux_sheared", out)
    print(json.dumps(out["max_rel_flux_err"]))
    assert out["max_rel_flux_err"] <= 1e-10


def many_items(n=300, N=121, K=3, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    A0 = torch.zeros(K, N, N, dtype=dtype)
    s = ItemSky(A0, P, RBFBasis(K), n, max_width=30 * P, n_levels=1)
    half = (N - 1) / 2 * P
    with torch.no_grad():
        w = torch.exp(math.log(0.45 * P) + torch.rand(n, 2, generator=g, dtype=torch.float64) * math.log(25 / 0.45))
        s.lw.data[0] = (0.5 * torch.log(w ** 2 - s.floor ** 2)).to(dtype)
        s.pos.data[0] = ((torch.rand(n, 2, generator=g, dtype=torch.float64) * 2 - 1) * (half + 10 * P)).to(dtype)
        s.ang.data[0] = (torch.rand(n, generator=g, dtype=torch.float64) * math.pi).to(dtype)
        s.lc.data[0] = (torch.randn(n, K, generator=g, dtype=torch.float64) * 0.5).to(dtype)
        s._mut += 1
    W = torch.randn(K, N, N, generator=g, dtype=torch.float64).to(dtype)
    return s, W


def loss(s, W):
    return (W * s.raster(0)).sum()


def test_gradients_fd():
    s, W = many_items()
    lev = s.plan().assign(s.widths(0))
    counts = torch.bincount(lev, minlength=s.plan().L + 1).tolist()
    for t in (s.pos, s.lw, s.ang, s.lc):
        t.grad = None
    loss(s, W).backward()
    grads = {"pos": s.pos.grad[0].clone(), "lw": s.lw.grad[0].clone(), "ang": s.ang.grad[0].clone(), "lc": s.lc.grad[0].clone()}
    params = {"pos": s.pos, "lw": s.lw, "ang": s.ang, "lc": s.lc}
    steps = {"pos": 1e-4 * P, "lw": 1e-4, "ang": 1e-4, "lc": 1e-4}   # central differences: truncation O(h^2) ~ 1e-8 relative, rounding ~ 1e-12 / h
    g = torch.Generator().manual_seed(5)
    # items: 8 per level (or all of a level if fewer)
    pick = torch.cat([torch.nonzero(lev == l).squeeze(1)[:8] for l in range(s.plan().L + 1)])
    res = {}
    skipped = 0
    for name, t in params.items():
        worst_rel = 0.0
        scale = float(grads[name].abs().max())
        for i in pick.tolist():
            sub = range(t.shape[2]) if t.dim() == 3 else [None]
            for j in sub:
                idx = (0, i, j) if j is not None else (0, i)
                with torch.no_grad():
                    v0 = float(t.data[idx])
                    t.data[idx] = v0 + steps[name]
                    s._mut += 1
                    lp, levp = float(loss(s, W)), s.plan().assign(s.widths(0))
                    t.data[idx] = v0 - steps[name]
                    s._mut += 1
                    lm, levm = float(loss(s, W)), s.plan().assign(s.widths(0))
                    t.data[idx] = v0
                    s._mut += 1
                if not (torch.equal(levp, lev) and torch.equal(levm, lev)):
                    skipped += 1
                    continue
                fd = (lp - lm) / (2 * steps[name])
                an = float(grads[name][idx[1:]])
                worst_rel = max(worst_rel, abs(fd - an) / (abs(an) + 1e-6 * scale))
        # directional derivative over ALL items
        dirs = []
        for _ in range(3):
            v = torch.randn(t.shape, generator=g, dtype=t.dtype)
            # every coordinate moves at once, so a small step: the patch layout (rounded patch centre; with sheared patches also the
            # rounded column centre of every row) is a discrete function of the detached parameters, and a layout change is a jump of
            # at most ~exp(-8) of the item peak (truncation level) which a larger step is likely to straddle
            hdir = 0.01 * steps[name]
            with torch.no_grad():
                t.data += hdir * v
                s._mut += 1
                lp = float(loss(s, W))
                t.data -= 2 * hdir * v
                s._mut += 1
                lm = float(loss(s, W))
                t.data += hdir * v
                s._mut += 1
            fd = (lp - lm) / (2 * hdir)
            an = float((grads[name] * v[0]).sum())
            dirs.append(abs(fd - an) / abs(an))
        res[name] = {"max_rel_err_per_coordinate": worst_rel, "max_rel_err_directional_all_items": max(dirs)}
    out = {"items_per_level": counts, "checked_items": int(pick.numel()), "skipped_level_switch": skipped, "groups": res}
    save("gradients_fd_sheared", out)
    print(json.dumps(out, indent=1))
    for r in res.values():
        assert r["max_rel_err_per_coordinate"] < 1e-5 and r["max_rel_err_directional_all_items"] < 1e-5


def test_float32():
    widths = [0.5 * 400 ** (i / 11) for i in range(12)]
    r64 = sweep(widths, (1, 5), (0.3,), ((0.37, -0.21),))
    r32 = sweep(widths, (1, 5), (0.3,), ((0.37, -0.21),), dtype=torch.float32)
    ratios = [max(a["err"], 1e-6) / max(b["err"], 1e-6) for a, b in zip(r32, r64)]
    s, W = many_items(dtype=torch.float32)
    loss(s, W).backward()
    gfin = all(bool(torch.isfinite(t.grad).all()) for t in (s.pos, s.lw, s.ang, s.lc))
    out = {"worst_f64": max(r["err"] for r in r64), "worst_f32": max(r["err"] for r in r32), "max_ratio_per_case_floor_1e-6": max(ratios),
           "all_finite": all(r["finite"] for r in r32), "grad_finite": gfin}
    save("float32_sheared", out)
    print(json.dumps(out, indent=1))
    assert out["all_finite"] and gfin
    assert out["worst_f32"] <= 3 * out["worst_f64"] and out["max_ratio_per_case_floor_1e-6"] <= 3


if __name__ == "__main__":
    for f in (test_accuracy_sweep, test_level_switches, test_edges, test_flux, test_gradients_fd, test_float32):
        f()
