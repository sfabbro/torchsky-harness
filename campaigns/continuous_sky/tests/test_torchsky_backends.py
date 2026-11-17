"""Tests of the four optional torchsky back ends (pieces A to D of TORCHSKY_INTEGRATION_REPORT.md).  CPU, one thread, whole file under 90 s.

  A  --item-impl torchsky   items through GaussianItemScene against ItemSky's own rasteriser (float64, wide, narrow and rotated items, near the edge):
                            raster to 1e-5 of the image maximum, gradients to 1e-4 relative (direct and pyramid)
  B  --rate-wcs telescope   count-rate positions from the telescope frame against the calibrated file's header WCS on a sample file (3 to 60 mas)
  C  --loader torchfits     identical arrays and header values at usable pixels, identical cut dictionaries, and the astropy fallback
  D  (no switch) the measured agreement of NestedKernelPSF with the prototype's render (1e-3 on a resolved sky)
  and that every default is the prototype's.
Run:  PYTHONPATH=<torchsky>/src python -m pytest tests/test_torchsky_backends.py -q"""
import math
import os
import sys

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
SAMPLE = os.path.join(HERE, "..", "..", "fieldop_gain", "dl")
RATE = os.path.join(SAMPLE, "rate", "jw02736001001_02101_00009_nrcb1_rate.fits")
CAL_FULL = os.path.join(SAMPLE, "cal", "jw02736001001_02101_00009_nrcb1_cal.fits")
need_files = pytest.mark.skipif(not (os.path.exists(RATE) and os.path.exists(CAL_FULL)), reason="12-exposure sample not on this machine")


# ----------------------------------------------------------------------------- defaults
def test_defaults_are_the_prototypes():
    from fieldop.cli import build_parser
    a = build_parser().parse_args([])
    assert (a.item_impl, a.rate_wcs, a.loader) == ("prototype", "cal", "astropy")
    from fieldop import data
    assert data.LOADER == "astropy" and data.RATE_WCS == "cal"


# ----------------------------------------------------------------------------- A
def _items(raster, impl, n=160, N=72, pitch=0.04, K=3, seed=0, dtype=torch.float64):
    from fieldop.sky import ItemSky, RBFBasis
    g = torch.Generator().manual_seed(seed)
    sky = ItemSky(torch.rand(K, N, N, generator=g, dtype=dtype), pitch, RBFBasis(K).to(dtype), n, raster=raster, impl=impl)
    half = sky.half_extent
    with torch.no_grad():
        sky.pos.data[0] = ((torch.rand(n, 2, generator=g, dtype=dtype) * 2 - 1) * half * 1.05)           # a few items sit beyond the edge
        w = torch.exp(torch.rand(n, 2, generator=g, dtype=dtype) * math.log(30 / 0.3) + math.log(0.3)) * pitch   # 0.3 to 30 pixels, narrow and wide
        sky.lw.data[0] = 0.5 * torch.log((w ** 2 - sky.floor ** 2).clamp(min=1e-6))
        sky.ang.data[0] = torch.rand(n, generator=g, dtype=dtype) * 2 * math.pi - math.pi / 2
        sky.lc.data[0] = torch.randn(n, K, generator=g, dtype=dtype) * 0.5
    return sky


@pytest.mark.parametrize("raster", ["direct", "pyramid"])
def test_A_items_through_gaussian_item_scene(raster):
    a, b = _items(raster, "prototype"), _items(raster, "torchsky")
    ra, rb = a.raster(0), b.raster(0)
    assert float((ra - rb).abs().max() / ra.abs().max()) < 1e-5
    w = torch.randn(ra.shape, generator=torch.Generator().manual_seed(1), dtype=ra.dtype)
    for s in (a, b):
        (s.raster(0) * w).sum().backward()
    for name in ("pos", "lw", "ang", "lc"):
        ga, gb = getattr(a, name).grad, getattr(b, name).grad
        assert float((ga - gb).norm() / ga.norm()) < 1e-4, name


def test_A_sheared_patch_is_refused_and_width_floor_is_the_prototypes():
    from fieldop.sky import ItemSky, RBFBasis
    with pytest.raises(RuntimeError, match="sheared"):
        s = ItemSky(torch.rand(2, 30, 30, dtype=torch.float64), 0.04, RBFBasis(2).to(torch.float64), 20, raster="pyramid", patch="sheared", impl="torchsky")
        s.raster(0)
    # one very narrow item: both impls floor its width at --item-floor and give it the same flux
    outs = []
    for impl in ("prototype", "torchsky"):
        s = _items("direct", impl, n=1, N=40)
        with torch.no_grad():
            s.pos.data[0, 0] = torch.tensor([0.013, -0.021], dtype=torch.float64)
            s.lw.data[0] = -20.0                                    # width = floor
            s.lc.data[0] = 0.0
        outs.append(s.raster(0).detach())
    assert float((outs[0] - outs[1]).abs().max() / outs[0].abs().max()) < 1e-5
    assert abs(float(outs[1].sum()) * 0.04 ** 2 / 3 - 1.0) < 1e-6     # K = 3 planes, unit coefficient, flux exact


# ----------------------------------------------------------------------------- B
@need_files
def test_B_telescope_positions_near_the_calibrated_solution():
    import torchfits
    from fieldop.tsky_backends import TelescopeRateWCS
    from torchsky.wcs import WCS
    g = torch.linspace(0.0, 2047.0, 17, dtype=torch.float64)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    pc = torch.stack([gx.ravel(), gy.ravel()], -1)
    sh = dict(torchfits.read_header(CAL_FULL, "SCI"))
    wc = WCS(header={k: v for k, v in sh.items() if k and k not in ("COMMENT", "HISTORY", "")}).pixel_to_world_differentiable(pc, origin=0)
    tw = TelescopeRateWCS(RATE)
    assert tw.aperture == "NRCB1_FULL"
    wt = tw.pixel_to_world_differentiable(pc, origin=0)
    d = torch.deg2rad(wt - wc)
    sep = torch.sqrt((d[:, 0] * torch.cos(torch.deg2rad(wc[:, 1]))) ** 2 + d[:, 1] ** 2) * 180 / math.pi * 3.6e6      # mas
    assert 1.0 < float(sep.pow(2).mean().sqrt()) < 10.0           # measured 4.1 mas rms, 10 mas maximum for this file (not below 5 mas for all 12)
    assert abs(tw.pixar_a2 - float(sh["PIXAR_A2"])) / float(sh["PIXAR_A2"]) < 5e-3
    # the plain header WCS of the count-rate file is far worse (about 90 mas rms): the telescope frame is the better stand-in
    shr = dict(torchfits.read_header(RATE, "SCI"))
    wr = WCS(header={k: v for k, v in shr.items() if k and k not in ("COMMENT", "HISTORY", "")}).pixel_to_world_differentiable(pc, origin=0)
    dr = torch.deg2rad(wr - wc)
    sr = torch.sqrt((dr[:, 0] * torch.cos(torch.deg2rad(wc[:, 1]))) ** 2 + dr[:, 1] ** 2) * 180 / math.pi * 3.6e6
    assert float(sr.pow(2).mean().sqrt()) > 5 * float(sep.pow(2).mean().sqrt())


@need_files
def test_B_data_path_with_rate_wcs_telescope_uses_the_telescope_frame():
    from fieldop import data
    data.RATE_WCS, data.LOADER, data.DQ_IGNORE_BITS = "telescope", "astropy", 0
    try:
        raw = data.read_exposure_file(CAL_FULL, RATE)
    finally:
        data.RATE_WCS = "cal"
    assert type(raw["w"]).__name__ == "TelescopeRateWCS" and raw["sci"].shape == (2048, 2048)


# ----------------------------------------------------------------------------- C
@need_files
def test_C_loader_identical_arrays_headers_and_cuts():
    import torchfits
    from fieldop import data
    from astropy.io import fits
    data.RATE_WCS, data.DQ_IGNORE_BITS = "cal", 0
    try:
        data.LOADER = "astropy"
        a = data.read_exposure_file(CAL_FULL, RATE)
        data.LOADER = "torchfits"
        b = data.read_exposure_file(CAL_FULL, RATE)
        ref = np.isfinite(a["sci"]) & np.isfinite(a["err"]) & (a["err"] > 0) & (a["dq"].astype(np.int64) == 0)
        assert np.array_equal(b["usable"], ref)
        assert np.array_equal(a["sci"][ref], b["sci"][ref]) and np.array_equal(a["err"][ref], b["err"][ref])
        assert a["detector"] == b["detector"] and a["instrument"] == b["instrument"] and a["pixar_a2"] == b["pixar_a2"]
        sh = fits.getheader(CAL_FULL, "SCI")
        hd = {k: sh[k] for k in sh.keys() if k and k not in ("COMMENT", "HISTORY", "")}
        assert hd.keys() == b["header_hd"].keys() and all(hd[k] == b["header_hd"][k] for k in hd)
        assert np.array_equal(a["area"], b["area"])
        # the whole cut dictionary of the loader: identical wherever a pixel is valid, and the same pixels are valid
        hr = dict(torchfits.read_header(RATE, "SCI"))
        box = {"ra": float(hr["RA_REF"]), "dec": float(hr["DEC_REF"])}
        data.LOADER = "astropy"
        ca = data.load_exposure_file(CAL_FULL, box, 2.0, rate_path=RATE)
        data.LOADER = "torchfits"
        cb = data.load_exposure_file(CAL_FULL, box, 2.0, rate_path=RATE)
        assert ca is not None and cb is not None and ca["valid"].sum() > 2000
        assert np.array_equal(ca["valid"], cb["valid"]) and ca["cut0"] == cb["cut0"]
        v = ca["valid"]
        assert np.array_equal(ca["d"][v], cb["d"][v]) and np.array_equal(ca["sig"][v], cb["sig"][v])
        assert np.array_equal(ca["X"], cb["X"]) and np.array_equal(ca["Y"], cb["Y"]) and ca["bkg0"] == cb["bkg0"]
        # dq_ignore_bits 2 gives the astropy mask too
        data.DQ_IGNORE_BITS = 2
        data.LOADER = "astropy"
        a2 = data.read_exposure_file(CAL_FULL, RATE)
        data.LOADER = "torchfits"
        b2 = data.read_exposure_file(CAL_FULL, RATE)
        ref2 = np.isfinite(a2["sci"]) & np.isfinite(a2["err"]) & (a2["err"] > 0) & ((a2["dq"].astype(np.int64) & ~2) == 0)
        assert np.array_equal(b2["usable"], ref2) and ref2.sum() > ref.sum()
    finally:
        data.LOADER, data.DQ_IGNORE_BITS = "astropy", 0


@need_files
def test_C_import_failure_falls_back_to_astropy_with_a_message(capsys):
    from fieldop import data, tsky_backends as tb
    saved, note = dict(tb._LOADER_STATE), list(data._LOADER_NOTE)
    tb._LOADER_STATE.update(checked=True, ok=False, err="ImportError: simulated")
    data._LOADER_NOTE.clear()
    try:
        data.LOADER, data.RATE_WCS, data.DQ_IGNORE_BITS = "torchfits", "cal", 0
        raw = data.read_exposure_file(CAL_FULL, RATE)
        assert raw.get("usable") is None and raw["dq"] is not None            # the astropy dictionary
        out = capsys.readouterr().out
        assert "FALLING BACK to the astropy reader" in out and "simulated" in out
    finally:
        data.LOADER = "astropy"
        tb._LOADER_STATE.update(saved)
        data._LOADER_NOTE[:] = note


# ----------------------------------------------------------------------------- D
def test_D_nested_kernel_agrees_with_the_prototype_render_on_a_resolved_sky():
    sys.path.insert(0, HERE)
    import check_render_nested as c
    torch.set_default_dtype(torch.float64)
    try:
        r = c.run("sharp")
    finally:
        torch.set_default_dtype(torch.float32)
    assert r["n_levels"] >= 2 and r["nested_num_levels"] == r["n_levels"]
    assert r["nested_vs_prototype"] < 1e-3                        # measured 1.8e-4 (stars and galaxies); white noise reaches 1.2e-3, see the report
