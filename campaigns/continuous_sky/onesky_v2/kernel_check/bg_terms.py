#!/usr/bin/env python3
"""Mean of the fitted detector background (constant + quadratic Legendre + per-detector 8x8 pattern, zero mean over the whole
detector) over each exposure's training footprint, written from model.background (stripes have zero mean over the rows/columns used
and are left out). The footprint is recovered from the 3x3 reference lattice in the state (cell centres of the footprint box)."""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kc_common import Fit, FILTERS, HERE
torch.set_num_threads(1)


def footprint(fit, det):
    lat = fit.ref_lattice_xi(det).numpy()          # x = x0 + {1/6, 1/2, 5/6} (x1 - x0)
    xs, ys = np.unique(lat[:, 0].round(6)), np.unique(lat[:, 1].round(6))
    wx, wy = (xs[2] - xs[0]) / (2 / 3), (ys[2] - ys[0]) / (2 / 3)
    return xs[1] - wx / 2, xs[1] + wx / 2, ys[1] - wy / 2, ys[1] + wy / 2


def bg_mean(fit, e, n=64):
    det = int(fit.e_det[e])
    x0, x1, y0, y1 = footprint(fit, det)
    gx, gy = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    X, Y = np.meshgrid(gx, gy)
    xi = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1), dtype=torch.float32)
    x, y = xi[:, 0], xi[:, 1]
    L = torch.stack([torch.ones_like(x), x, y, 1.5 * x * x - 0.5, x * y, 1.5 * y * y - 0.5], -1)
    b = fit.s["bkg"][e]
    leg = (L * b).sum(-1)
    pat = fit.s["pattern"][det]
    pat = pat - pat.mean()
    pv = F.grid_sample(pat[None, None], xi[None, None], mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0]
    return dict(footprint_xi=[float(x0), float(x1), float(y0), float(y1)], constant=float(b[0]), legendre_mean=float(leg.mean()),
                pattern_mean=float(pv.mean()), pattern_min_max=[float(pv.min()), float(pv.max())], total_mean=float((leg + pv).mean()),
                data_median=float(fit.s["bkg0"][e]))


if __name__ == "__main__":
    fit = Fit()
    q4 = json.load(open(os.path.join(HERE, "results", "q4_wings.json")))
    for f, rows in q4.items():
        for o in rows:
            o["background_over_footprint"] = bg_mean(fit, o["exposure"])
            bo = o["background_over_footprint"]
            o["model_mean_full"] = o["mean_pred_full_MJysr"] + bo["total_mean"]
            print(f, o["exposure"], o["detector"], "footprint", [round(v, 2) for v in bo["footprint_xi"]], "const %.4f leg %.4f pattern %.4f (min %.3f max %.3f) total bg %.4f | sky term %.4f wings %.4f | model mean %.4f data median %.4f"
                  % (bo["constant"], bo["legendre_mean"], bo["pattern_mean"], *bo["pattern_min_max"], bo["total_mean"], o["mean_pred_full_MJysr"], o["mean_wing_MJysr"], o["model_mean_full"], bo["data_median"]))
    json.dump(q4, open(os.path.join(HERE, "results", "q4_wings.json"), "w"), indent=1)
