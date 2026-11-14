"""Unit tests of the item sky (fieldop.sky.ItemSky, --sky-form items). CPU, seconds.

  1. flux and cell integration: one item rasterised on the coefficient grid, for widths from the floor to 10 x the floor, sub-node offsets
     0 / 0.25 / 0.5 pitch, angles 0 and 30 deg: (a) flux after renormalisation (p^2 x sum of the grid), (b) flux BEFORE the renormalisation
     (error of the point-sampled Gaussian sum, what the renormalisation removes), (c) per-node error against the exact cell mean (erf box
     integral; axis-aligned items only, where it is available in closed form), as a fraction of the peak, (d) centroid error of the grid
     (sub-node placement) in units of the pitch, (e) the same centroid read through the tent interpolant on a fine grid.
  2. gradients with respect to positions, log widths, angles, log coefficients and the smooth grid against central finite differences
     (float64).
  3. a sky made of known Gaussians (exact cell integrals, incl. point-like items between nodes) is reproduced by fitting items seeded on
     the lattices from it (Adam, 600 steps).
Run:  python tests/test_items_unit.py     (numbers in tests/items_results/unit.json)
"""
import json
import math
import os
import sys

import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fieldop.sky import RBFBasis, ItemSky  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "items_results")
os.makedirs(OUT, exist_ok=True)
P = 0.04


def phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def exact_cell_mean(x, y, a, b, N, p):
    """Axis-aligned unit-flux Gaussian: mean over each cell (side p, centred on the node) -> (N, N)."""
    g = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * p
    fx = phi((g + p / 2 - x) / a) - phi((g - p / 2 - x) / a)
    fy = phi((g + p / 2 - y) / b) - phi((g - p / 2 - y) / b)
    return fy[:, None] * fx[None, :] / (p * p)


def one_item_sky(N, x, y, a, b, th, K=3, dtype=torch.float64, smooth=0.0):
    A0 = torch.zeros(K, N, N, dtype=dtype)
    s = ItemSky(A0, P, RBFBasis(K), 1, n_levels=1, smooth_pitch=smooth)
    with torch.no_grad():
        s.pos.data[0, 0] = torch.tensor([x, y], dtype=dtype)
        fl = s.floor
        s.lw.data[0, 0] = torch.tensor([0.5 * math.log(max(a * a - fl * fl, 1e-30)), 0.5 * math.log(max(b * b - fl * fl, 1e-30))], dtype=dtype)
        s.ang.data[0, 0] = th
        s.lc.data[0, 0] = 0.0          # c = 1 for every k
        s._mut += 1
    return s


def test_flux_and_cells():
    N = 81
    rows = []
    fl = 0.4 * P
    for wf in (1.0, 1.25, 1.5, 2.0, 4.0, 10.0):
        for off in (0.0, 0.25, 0.5):
            for th in (0.0, math.pi / 6):
                a, b = wf * fl, wf * fl * (1.0 if th == 0 else 1.6)
                x, y = off * P + 0.003 * 0, -off * P
                s = one_item_sky(N, x, y, a, b, th)
                with torch.no_grad():
                    A = s.coeffs()[0]
                    flux = float(A.sum() * P * P)
                    # before renormalisation: point-sampled Gaussian with variance + p^2/12, summed over an untruncated grid
                    q = P * P / 12
                    g = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * P
                    YY, XX = torch.meshgrid(g, g, indexing="ij")
                    dx, dy = XX - x, YY - y
                    c, sn = math.cos(th), math.sin(th)
                    u, v = c * dx + sn * dy, -sn * dx + c * dy
                    a2, b2 = a * a + q, b * b + q
                    G = torch.exp(-0.5 * (u * u / a2 + v * v / b2)) / (2 * math.pi * math.sqrt(a2 * b2))
                    flux_pre = float(G.sum() * P * P)
                    cx = float((A * XX).sum() / A.sum())
                    cy = float((A * YY).sum() / A.sum())
                    # centroid of the tent interpolant (continuous sky) on a fine grid
                    gf = torch.linspace(-(N - 1) / 2 * P, (N - 1) / 2 * P, 4 * (N - 1) + 1, dtype=torch.float64)
                    YF, XF = torch.meshgrid(gf, gf, indexing="ij")
                    T = s.sample_coeffs(A[None], XF, YF)[0]
                    tx = float((T * XF).sum() / T.sum())
                    row = {"width_over_floor": wf, "offset_pitch": off, "angle_deg": round(math.degrees(th)), "flux_error": flux - 1.0,
                           "flux_error_before_renorm": flux_pre - 1.0, "centroid_error_x_pitch": (cx - x) / P, "centroid_error_y_pitch": (cy - y) / P,
                           "tent_centroid_error_x_pitch": (tx - x) / P}
                    if th == 0:
                        E = exact_cell_mean(x, y, a, b, N, P)
                        row["cell_error_over_peak"] = float((A - E).abs().max() / E.max())
                        row["cell_flux_exact"] = float(E.sum() * P * P) - 1.0
                    rows.append(row)
    worst = {k: max(abs(r[k]) for r in rows if k in r) for k in ("flux_error", "flux_error_before_renorm", "centroid_error_x_pitch",
                                                                  "centroid_error_y_pitch", "tent_centroid_error_x_pitch", "cell_error_over_peak")}
    by_w = {}
    for r in rows:
        d = by_w.setdefault(r["width_over_floor"], {"flux_error_before_renorm": 0, "cell_error_over_peak": 0, "centroid_error_pitch": 0})
        d["flux_error_before_renorm"] = max(d["flux_error_before_renorm"], abs(r["flux_error_before_renorm"]))
        d["cell_error_over_peak"] = max(d["cell_error_over_peak"], abs(r.get("cell_error_over_peak", 0)))
        d["centroid_error_pitch"] = max(d["centroid_error_pitch"], abs(r["centroid_error_x_pitch"]), abs(r["centroid_error_y_pitch"]))
    res = {"rows": rows, "worst": worst, "worst_by_width": by_w}
    json.dump(res, open(os.path.join(OUT, "unit_flux.json"), "w"), indent=1)
    print("flux / cells, worst over offsets and angles, by width / floor:")
    for k, v in by_w.items():
        print(f"  {k:5.2f}: {json.dumps({kk: float(f'{vv:.2e}') for kk, vv in v.items()})}")
    print("  worst:", {k: float(f"{v:.2e}") for k, v in worst.items()})
    assert worst["flux_error"] < 1e-9, worst
    assert worst["centroid_error_x_pitch"] < 0.03 and worst["centroid_error_y_pitch"] < 0.03
    return res


def test_gradients():
    torch.manual_seed(0)
    N, K = 41, 3
    A0 = torch.rand(K, N, N, dtype=torch.float64)
    s = ItemSky(A0, P, RBFBasis(K), 7, n_levels=2, smooth_pitch=0.2)
    with torch.no_grad():
        s.pos.data += 0.013 * torch.randn_like(s.pos)
        s.lw.data = torch.log(torch.tensor(0.03, dtype=torch.float64)) + 0.5 * torch.randn_like(s.lw)
        s.ang.data += torch.randn_like(s.ang)
        s.lc.data += 0.3 * torch.randn_like(s.lc)
        s.S.data += torch.randn_like(s.S)
    W = torch.randn(K, N, N, dtype=torch.float64)

    def f():
        return (W * s.coeffs()).sum()
    L = f()
    L.backward()
    out = {}
    for name in ("pos", "lw", "ang", "lc", "S"):
        t = getattr(s, name)
        g = t.grad.clone().flatten()
        idx = torch.randperm(g.numel())[:12]
        fd = []
        for i in idx.tolist():
            eps = 1e-6
            with torch.no_grad():
                t.data.view(-1)[i] += eps
                lp = float(f())
                t.data.view(-1)[i] -= 2 * eps
                lm = float(f())
                t.data.view(-1)[i] += eps
            fd.append((lp - lm) / (2 * eps))
        fd = torch.tensor(fd, dtype=torch.float64)
        err = float((g[idx] - fd).abs().max() / fd.abs().max().clamp(min=1e-12))
        out[name] = err
    json.dump(out, open(os.path.join(OUT, "unit_grad.json"), "w"), indent=1)
    print("gradients, max |autograd - finite difference| / max |finite difference|:", {k: float(f"{v:.1e}") for k, v in out.items()})
    assert all(v < 1e-5 for v in out.values()), out
    return out


def test_known_gaussians():
    """Target: 6 Gaussians (exact cell means, axis-aligned), 3 of them point-like (width = floor) between nodes; K = 2 (the same image in both terms) to keep it small."""
    torch.manual_seed(1)
    N = 51
    fl = 0.4 * P
    truth = [(0.13 * P + 0.11, -0.37 * P, fl, fl, 5.0), (-0.31, 0.21 + 0.5 * P, fl, fl, 3.0), (0.4, 0.45, fl, fl, 2.0),
             (0.0, 0.0, 0.12, 0.06, 4.0), (-0.5, -0.4, 0.2, 0.2, 6.0), (0.55, -0.5, 0.05, 0.09, 1.5)]
    T = sum(F_ * exact_cell_mean(x, y, a, b, N, P) for x, y, a, b, F_ in truth)[None].repeat(2, 1, 1)
    A0 = T + 0.0
    res = {}
    for n_items in (60, 200):
        s = ItemSky(A0.float(), P, RBFBasis(2), n_items, n_levels=3)
        from types import SimpleNamespace
        cfg = SimpleNamespace(item_lr_pos=0.05, item_lr_width=0.03, item_lr_angle=0.05, item_lr_coef=0.05)
        opt = torch.optim.Adam(s.opt_groups(0.0, cfg))
        Tt = T.float()
        e0 = float(((s.coeffs() - Tt) ** 2).sum().sqrt() / (Tt ** 2).sum().sqrt())
        nst = 600
        for it in range(nst):
            f = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * it / nst))
            for g in opt.param_groups:
                g["lr"] = g["base"] * f
            loss = ((s.coeffs() - Tt) ** 2).sum() / (Tt ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            A = s.coeffs()
        e1 = float(((A - Tt) ** 2).sum().sqrt() / (Tt ** 2).sum().sqrt())
        # flux recovered within 3 pitches of each point-like truth item
        g = (torch.arange(N, dtype=torch.float32) - (N - 1) / 2) * P
        YY, XX = torch.meshgrid(g, g, indexing="ij")
        fr = []
        for x, y, a, b, F_ in truth[:3]:
            m = ((XX - x) ** 2 + (YY - y) ** 2) < (3 * P) ** 2
            fr.append(float(A[0][m].sum() / Tt[0][m].sum()))
        res[n_items] = {"rel_l2_error_seed": e0, "rel_l2_error_fit": e1, "flux_ratio_point_items": fr, "steps": nst}
        print(f"known Gaussians, {n_items} items: relative L2 error {e0:.3f} at the seed -> {e1:.4f} after {nst} steps; "
              f"flux ratio near point-like items {[round(v, 3) for v in fr]}")
    json.dump(res, open(os.path.join(OUT, "unit_known.json"), "w"), indent=1)
    assert res[200]["rel_l2_error_fit"] < 0.05, res
    return res


if __name__ == "__main__":
    test_flux_and_cells()
    test_gradients()
    test_known_gaussians()
