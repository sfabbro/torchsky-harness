"""CPU analogue of the SMACS background conditioning (SOLVER_DIAGNOSIS_2026-11-19.md section 11): the reproducer with detectors three times
wider than the data box, so that each exposure's training cut covers a third of its detector and the six Legendre background terms are nearly
collinear on it. Reports the background blocks' condition numbers, and for each variant the residual curve, the Lanczos range and where the top
and bottom Ritz vectors live (sky against background, by exposure and term).

    python tests/solver_repro/bkg_analogue.py --name NAME [--variants none,jacobi,filter,restrict] [--bkg-basis raw|orth] [--iters 30]
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from repro_field import build_model  # noqa: E402
from fieldop import uncertainty as U  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--variants", default="none,jacobi,jacobi_orth,bjacobi_orth,filter,filter_orth,restrict,restrict_orth")
ap.add_argument("--bkg-basis", default="raw")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--sw-npix", type=int, default=180)
ap.add_argument("--lw-npix", type=int, default=120)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--det2048", type=int, default=0, help="1: cuts are off-centre patches of 2048-pixel detectors (see below)")
ap.add_argument("--box", type=float, default=0.4, help="region_box: training pixels within this half width [arcsec] (the lab's patch)")
a = ap.parse_args()
torch.set_num_threads(a.threads)
t0 = time.time()
cache = os.path.join(HERE, "results", f"_field_cache_{a.sw_npix}_{a.lw_npix}.pt")
if os.path.exists(cache):
    model, exps, cfg = torch.load(cache, weights_only=False)
else:
    model, exps, cfg, truth, o = build_model(sw_npix=a.sw_npix, lw_npix=a.lw_npix)
    try:
        torch.save((model, exps, cfg), cache)
    except Exception as ex:
        print("field not cached:", ex)
if a.det2048:
    # the SMACS geometry of the background terms: each cut is a small, off-centre patch of a 2048-pixel detector (normalised detector
    # coordinates span 0.05 to 0.2, as a 4 to 20 arcsec box on 63 to 129 arcsec NIRCam detectors); the model's own kernel-field shapes are not
    # changed, the background, throughput and distortion fields are evaluated at the patch's coordinates
    # (only det_shape changes: the cut then sits at a detector corner; xi_px and the cut offset, which the kernel field uses, are unchanged)
    for e in exps:
        e.det_shape = (2048, 2048)
logs = []
kw = dict(tile=64, batch_px=16000, region_box=a.box if a.box > 0 else None)
op = U.SkyNormalOperator(model, exps, cfg, rho=10.0, precond_mode="filter", precision="mixed", log=logs.append, **kw)
rec = {"name": a.name, "args": vars(a), "mu": op.mu, "bb_cond_max": max(op.bb_cond), "bb_cond_sorted": sorted(op.bb_cond)[-8:],
       "n_pix": op.n_pix, "checks": U.operator_checks(op, log=logs.append), "variants": {}}
print(a.name, "background block condition numbers (largest 8):", [f"{c:.2e}" for c in rec["bb_cond_sorted"]], "mu", op.mu, flush=True)
for v in [x for x in a.variants.split(",") if x]:
    op.set_bkg_basis("orth" if v.endswith("_orth") else "raw")
    op.precond_mode = {"none": "none", "jacobi": "jacobi", "filter": "filter", "restrict": "filter", "bjacobi": "bjacobi"}[v.split("_")[0]]
    target = U.RestrictedOperator(op, factor=2, log=logs.append) if v.startswith("restrict") else op
    x, info = U.pcg(target, target.residual_rhs(), tol=1e-12, maxiter=a.iters, keep_vectors=True)
    h = info["history_all"]
    modes = {}
    for lab, i in (("top", -1), ("bottom", 0)):
        val, y = U.ritz_vectors(info, which=(i,))[0]
        modes[lab] = {"ritz": val, **U.mode_report(target, y.to(op.dev) if hasattr(y, "to") else y)}
    rec["variants"][v] = {"residual": {i: h[i - 1] for i in (1, 5, 10, 15, 20, 25, 30, 40, 60, 100) if i <= len(h)}, "breakdown": info["breakdown"],
                          "ritz": [info["ritz_min"], info["ritz_max"]], "modes": modes}
    print(f"{a.name} {v}: residual " + ", ".join(f"{i}: {h[i - 1]:.2e}" for i in (1, 5, 10, 20, 30, 60, 100) if i <= len(h))
          + f"; Lanczos {info['ritz_min']:.3g} to {info['ritz_max']:.3g}; top mode: sky {modes['top']['sky_share']:.3f}, background "
          f"{modes['top']['background_share']:.3f} {modes['top']['background_top_exposures']} terms {modes['top']['background_per_term']}; bottom mode: sky "
          f"{modes['bottom']['sky_share']:.3f}" + (f"; BREAKDOWN {info['breakdown']}" if info["breakdown"] else ""), flush=True)
    op.precond_mode = op.setup_mode
    op.set_bkg_basis("raw")
rec["n_bkg_dropped"] = op.n_bkg_dropped
rec["log"] = logs
rec["seconds"] = time.time() - t0
json.dump(rec, open(os.path.join(HERE, "results", f"bkg_{a.name}.json"), "w"), indent=1, default=str)
