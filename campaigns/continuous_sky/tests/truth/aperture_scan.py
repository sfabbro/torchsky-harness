"""Question (1) of the sky-zero change: why is the error of a signed (not clipped) sky in empty regions 1.6 to 2.1 times the ideal noise?
Re-measures saved skies (run_truth.py writes <out>_sky.npz) at the 25 empty positions with Gaussian apertures of several widths, each in
units of the ideal noise of THAT aperture (the ideal estimator knows the profile, the response and the background). If the ratio falls to
about 1 for wide apertures, the excess is the price of a free-form sky at the resolution element (it does not know that the light has the
aperture's shape); if it stays, the excess is noise the fit adds at every scale (under-convergence, optimiser noise).

    python tests/truth/aperture_scan.py tests/truth/results/sky_zero/mm_g_n8_s?.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import truth_problem as tp  # noqa: E402

torch.set_num_threads(1)
WIDTHS = (0.035, 0.07, 0.14, 0.28)


def main(paths):
    pop = tp.make_population()
    fl = tp.flux_for_snr(pop, n_ref=8)
    empties = tp.empty_positions(pop)
    pf = [(p[0], p[1], p[2], float(F)) for p, F in zip(pop, fl)]
    runs = [json.load(open(p)) for p in paths]
    n = runs[0]["n_train"]
    acc = {w: {"Z": [], "I": []} for w in WIDTHS}
    for p, r in zip(paths, runs):
        assert r["n_train"] == n
        exps, fset, meta, truth, _, _ = tp.build_field(n, seed=r["config"]["seed"], pop=pop, fluxes=fl)
        tr = [e for e in exps if e.split == "train"]
        sk = np.load(os.path.splitext(p)[0] + "_sky.npz")
        img = torch.as_tensor(sk["sky"], dtype=torch.float64)
        db = float(sk["bkg"][:, 0].mean()) - tp.BKG
        N = img.shape[0]
        timg = tp.true_sky_nodes(pf, N, 0.05)
        d_tr = torch.stack([e.d.reshape(-1) for e in tr]).double()
        c_tr = torch.stack([e.extra["clean"].reshape(-1) for e in tr]).double()
        for w in WIDTHS:
            cache = tp.IdealCache([(x, y, w) for x, y in empties], tr, fset)
            rec, ref = [], []
            for (x, y) in empties:
                idx, P, W = tp.aperture_weights(N, 0.05, x, y, w)
                rec.append(tp.sky_aperture_flux(img, idx, P, W) + db * tp.const_factor(P, W))
                ref.append(tp.sky_aperture_flux(timg, idx, P, W))
            ide = cache.estimate(d_tr.float()) - cache.estimate(c_tr.float() + tp.BKG)
            acc[w]["Z"].append((np.array(rec) - np.array(ref)) / cache.sigma)
            acc[w]["I"].append(ide / cache.sigma)
    out = {}
    for w in WIDTHS:
        Z, I = np.array(acc[w]["Z"]), np.array(acc[w]["I"])
        out[w] = {"rms_fit": float(np.sqrt((Z ** 2).mean())), "rms_ideal_empirical": float(np.sqrt((I ** 2).mean())),
                  "sd_fit_about_mean": float(Z.std()), "mean_fit": float(Z.mean())}
        print(f"aperture width {w:.3f} arcsec: fit rms {out[w]['rms_fit']:.2f} (sd about mean {out[w]['sd_fit_about_mean']:.2f}, mean {out[w]['mean_fit']:+.2f}), "
              f"ideal empirical rms {out[w]['rms_ideal_empirical']:.2f}  [ideal-noise units of that aperture]")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
