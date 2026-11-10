"""CPU tests of the pixel-area path alpha = |det dOmega/dxi| (count-rate `_rate` input, fieldop F15).

1. Analytic distortion: a quadratic map (col, row) -> (X, Y) with a known Jacobian determinant; alpha from the header-Jacobian maps
   (central differences of the per-pixel positions, as the loader computes them) must match it.
2. The per-exposure affine multiplies alpha by det(I + M) exactly; the rate renderer equals the cal renderer times alpha.
3. Real NIRCam _cal file: |det J| / PIXAR_A2 from the SIP header WCS, through the same loader, against the pipeline's AREA extension.
"""
import glob
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fieldop.cli import default_config, make_sky  # noqa: E402
from fieldop.data import load_exposure_file, pixel_jacobian  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import make_field  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "fieldop_results")
os.makedirs(OUT, exist_ok=True)
NIRCAM = os.path.join(os.path.dirname(__file__), "..", "testdata", "nircam")


def model_for(exps, fset, meta, **kw):
    b = dict(sky_pitch=0.04, render_pitch=0.02, level_nh=16, kernel_extent=0.6, n_lambda=3, k_hidden=16, pattern_grid=0, stripes=0,
             warm_kernel_steps=0, input_level="rate", amplitude="unit", distortion=0, sky_precond=0.0)
    b.update(kw)
    cfg = default_config(**b)
    sky, _ = make_sky(exps, fset, cfg, 2.0, "cpu")
    return cfg, FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))


def test_alpha_analytic_distortion():
    exps, fset, meta, truth, insts = make_field(seed=0, n_per_filter=4)
    cfg, m = model_for(exps, fset, meta)
    n = 60
    rr, cc = np.mgrid[0:n, 0:n].astype(np.float64)
    p, eps = 0.06, 2e-4
    X = p * cc + eps * cc ** 2 - 0.5 * eps * rr * cc
    Y = p * rr + 0.3 * eps * rr ** 2 + eps * cc * rr
    # analytic Jacobian
    dXc, dXr = p + 2 * eps * cc - 0.5 * eps * rr, -0.5 * eps * cc
    dYc, dYr = eps * rr, p + 0.6 * eps * rr + eps * cc
    det_true = np.abs(dXc * dYr - dXr * dYc)
    J = torch.from_numpy(pixel_jacobian(X, Y))
    xi = torch.stack([torch.from_numpy(cc.ravel() / (n - 1) * 2 - 1), torch.from_numpy(rr.ravel() / (n - 1) * 2 - 1)], -1).float()
    with torch.no_grad():
        a = m.pixel_area(torch.tensor(0), J.view(-1, 2, 2), xi, {})[0].view(n, n).double().numpy()
    inner = (slice(1, -1), slice(1, -1))
    err_in = np.abs(a[inner] / det_true[inner] - 1).max()
    err_all = np.abs(a / det_true - 1).max()
    json.dump({"max_rel_err_interior": float(err_in), "max_rel_err_with_edges": float(err_all),
               "area_range_rel": float(det_true.max() / det_true.min() - 1)}, open(os.path.join(OUT, "area_analytic.json"), "w"), indent=1)
    print(f"alpha vs analytic |det J|: interior max rel err {err_in:.2e}, with one-sided edges {err_all:.2e}; area varies by {det_true.max() / det_true.min() - 1:.1%}")
    assert err_in < 1e-5          # central differences are exact for a quadratic map
    assert err_all < 2e-2


def test_alpha_affine_and_rate_render():
    exps, fset, meta, truth, insts = make_field(seed=0, n_per_filter=4)
    cfg, m = model_for(exps, fset, meta)
    cfg_c, mc = model_for(exps, fset, meta, input_level="cal")
    mc.load_state_dict(m.state_dict())
    e = exps[0]
    pix = e.pix(e.patches[0])
    with torch.no_grad():
        m.sky.A.data[:] = 1.0
        mc.sky.A.data[:] = 1.0
        a0 = m.pixel_area(torch.tensor(e.eid), pix["Jpix"], pix["xi"], {})
        m.affine.data[e.eid, :4] = torch.tensor([3.0, -1.0, 2.0, 5.0])   # x 1e-3
        mc.affine.data[:] = m.affine.data
        a1 = m.pixel_area(torch.tensor(e.eid), pix["Jpix"], pix["xi"], {})
        M = (m.affine[e.eid, :4] * 1e-3).view(2, 2) + torch.eye(2)
        assert torch.allclose(a1 / a0, torch.det(M).expand_as(a1), rtol=1e-5)
        P = m.prepare()
        s_rate = m.render(e.eid, pix, P)["S"]
        s_cal = mc.render(e.eid, pix, mc.prepare())["S"]
        ratio = s_rate / s_cal
        assert torch.allclose(ratio, a1[0], rtol=1e-4)


@pytest.mark.skipif(not glob.glob(os.path.join(NIRCAM, "*_cal.fits")), reason="no NIRCam test file")
def test_alpha_against_pipeline_area_map():
    rows = []
    for f in sorted(glob.glob(os.path.join(NIRCAM, "*_cal.fits")))[:2]:
        from astropy.io import fits
        h = fits.getheader(f, "SCI")
        from torchsky.wcs import WCS
        w = WCS(header={k: h[k] for k in h.keys() if k and k not in ("COMMENT", "HISTORY", "")})
        ra, dec = w.pixel_to_world_values(1024.0, 1024.0) if hasattr(w, "pixel_to_world_values") else (h["CRVAL1"], h["CRVAL2"])
        box = {"ra": float(np.asarray(ra)), "dec": float(np.asarray(dec))}
        r = load_exposure_file(f, box, 30.0, want_jac=True)
        J = r["Jpix"]
        det = np.abs(J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0])
        rel = det / r["pixar_a2"]
        area = r["area_ext"]
        ok = np.isfinite(area) & (area > 0)
        ok[[0, -1], :] = False
        ok[:, [0, -1]] = False
        d = rel[ok] / area[ok] - 1
        rows.append({"file": os.path.basename(f), "n_pixels": int(ok.sum()), "median_rel_diff": float(np.median(d)), "rms_rel_diff": float(d.std()),
                     "max_abs_rel_diff": float(np.abs(d).max()), "area_map_range": [float(area[ok].min()), float(area[ok].max())]})
        print(rows[-1])
    json.dump(rows, open(os.path.join(OUT, "area_vs_pipeline.json"), "w"), indent=1)
    for r in rows:
        assert r["rms_rel_diff"] < 2e-3 and abs(r["median_rel_diff"]) < 5e-3
