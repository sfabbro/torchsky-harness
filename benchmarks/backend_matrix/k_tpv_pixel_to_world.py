"""Backend-matrix adapter: TPV WCS ``pixel_to_world`` (forward TPV polynomial + TAN).

Exercises ``torchsky.WCS(header).pixel_to_world(x, y)`` on a 1024x1024 TPV
header. The forward path (CD matrix, ``TPV.distort``, TAN deprojection, sky
rotation) is pure torch on every device; there is no native kernel, so no
``native`` variant.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torchsky

SIZES = {"small": 100, "medium": 10_000, "large": 1_000_000}
DTYPES = {"cpu": torch.float64, "cuda": torch.float64, "mps": torch.float32}
_SEEDS = {"small": 22_001, "medium": 22_002, "large": 22_003}


def _tpv_header() -> dict:
    # Copied from torchsky tests/test_wcs_tpv.py::_tpv_header (torchsky main 6c0a911a).
    header = {}
    header["NAXIS"] = 2
    header["NAXIS1"] = 1024
    header["NAXIS2"] = 1024
    header["CTYPE1"] = "RA---TPV"
    header["CTYPE2"] = "DEC--TPV"
    header["CRPIX1"] = 512.0
    header["CRPIX2"] = 512.0
    header["CRVAL1"] = 200.0
    header["CRVAL2"] = -20.0
    header["CD1_1"] = -2.8e-4
    header["CD1_2"] = 1.2e-7
    header["CD2_1"] = -1.1e-7
    header["CD2_2"] = 2.8e-4

    # Keep a mix of linear and non-linear terms; this remains stable for forward parity.
    header["PV1_0"] = 0.0
    header["PV1_1"] = 1.0
    header["PV1_2"] = 0.0
    header["PV1_4"] = 2.0e-4
    header["PV1_5"] = -3.0e-4
    header["PV1_6"] = 1.5e-4
    header["PV1_7"] = 2.0e-6
    header["PV1_39"] = 2.0e-11

    header["PV2_0"] = 0.0
    header["PV2_1"] = 0.0
    header["PV2_2"] = 1.0
    header["PV2_4"] = -1.0e-4
    header["PV2_5"] = 2.5e-4
    header["PV2_6"] = -2.0e-4
    header["PV2_7"] = -1.5e-6
    header["PV2_39"] = -1.0e-11

    return header


def _bench_header() -> dict:
    # Same fixture as k_tpv_world_to_pixel: tests/test_wcs_tpv.py::
    # _well_conditioned_tpv_header (the astropy-parity fixture). The forward cost
    # does not depend on the choice; it keeps the two adapters on one header.
    header = _tpv_header()
    header["PV2_1"] = 1.0
    header["PV2_2"] = 2.0e-4
    return header


HEADER = _bench_header()
# Smallest CD column norm (deg / pixel); converts sky offsets to pixels.
_PIXEL_SCALE_DEG = min(
    math.hypot(HEADER["CD1_1"], HEADER["CD2_1"]),
    math.hypot(HEADER["CD1_2"], HEADER["CD2_2"]),
)

# float64 budget: outputs are RA ~200 deg, Dec ~-20 deg. float64 spacing at RA in
# [128, 256) is 2.8e-14 deg = 1.0e-10 px; eager vs compiled kernels may reorder or
# fuse a handful of rounded ops (TPV polynomial, atan2/asin, deg<->rad), so allow
# ~10 ulp: 1e-9 px.
_F64_TOL_PX = 1.0e-9
# float32 budget (MPS): torchsky's float32 contract,
# torchsky.wcs._core_support._MPS_PIXEL_IDENTIFICATION_LIMIT_PX = 0.5 px (WCS warns
# when its float32 error estimate reaches it; 0.10 px for this header, measured
# max 0.11 px vs CPU float64).
_F32_TOL_PX = 0.5

NOTES = (
    "Calls torchsky.WCS(header).pixel_to_world(x, y) (origin 0) on the 1024x1024 TPV "
    "fixture tests/test_wcs_tpv.py::_well_conditioned_tpv_header (= _tpv_header() with "
    "PV2_1=1, PV2_2=2e-4; same header as k_tpv_world_to_pixel). Inputs: pixel "
    "coordinates uniform over the image, float64 on CPU/CUDA, float32 on MPS; the WCS is "
    "moved with WCS.to(device) (casts cached params to float32 on MPS). Pure torch on all "
    "devices: no native kernel and no switch, so 'native' is absent. 'compile' is "
    "torch.compile(fn, dynamic=False) of the same call; 'compile_fullgraph' records "
    "whether the forward path traces as one graph (it does on CPU). MPS (torch 2.14.1): "
    "Inductor's MPS backend raises NotImplementedError on torch.hypot (TPV._evaluate), so "
    "'compile' fails; fullgraph breaks first at WCS._mps_float32_error_pixels "
    "(`if not scale > 0.0`), which _as_wcs_coord runs on every MPS call (a device->host "
    "sync per call unless a precision warning already fired). check() converts the RA/Dec "
    "difference to pixels (dRA cos(Dec), dDec over the smallest CD column norm): float64 "
    "1e-9 px (~10 ulp at RA 200 deg), float32 0.5 px (torchsky's MPS "
    "pixel-identification limit); NaN masks must match."
)


_WCS_CACHE: dict[str, torchsky.WCS] = {}


def _wcs_on(device: torch.device) -> torchsky.WCS:
    key = str(device)
    wcs = _WCS_CACHE.get(key)
    if wcs is None:
        wcs = torchsky.WCS(dict(HEADER)).to(device)
        _WCS_CACHE[key] = wcs
    return wcs


def make_inputs(size_key: str, device: torch.device, dtype: torch.dtype) -> dict:
    device = torch.device(device)
    n = SIZES[size_key]
    gen = torch.Generator(device="cpu").manual_seed(_SEEDS[size_key])
    x = torch.rand(n, generator=gen, dtype=torch.float64) * HEADER["NAXIS1"] - 0.5
    y = torch.rand(n, generator=gen, dtype=torch.float64) * HEADER["NAXIS2"] - 0.5
    return {
        "wcs": _wcs_on(device),
        "x": x.to(dtype).to(device).contiguous(),
        "y": y.to(dtype).to(device).contiguous(),
    }


def items(size_key: str) -> int:
    return SIZES[size_key]


def _pure(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return inputs["wcs"].pixel_to_world(inputs["x"], inputs["y"], origin=0)


# Same body under a distinct code object: Dynamo caches compiled frames per code
# object, so compiling _pure twice would let the fullgraph wrapper reuse the
# (possibly graph-broken) cache entries of 'compile'.
def _pure_fullgraph(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return inputs["wcs"].pixel_to_world(inputs["x"], inputs["y"], origin=0)


def variants(device: torch.device) -> dict[str, Callable[[dict], Any]]:
    return {
        "eager": _pure,
        "compile": torch.compile(_pure, dynamic=False),
        "compile_fullgraph": torch.compile(
            _pure_fullgraph, dynamic=False, fullgraph=True
        ),
    }


def check(reference: Any, candidate: Any) -> tuple[bool, str]:
    # Copy to CPU before casting: on torch 2.14.1, mps_f32.to("cpu", torch.float64)
    # returns zeros.
    rra, rdec = (t.detach().cpu().to(torch.float64).reshape(-1) for t in reference)
    cra, cdec = (t.detach().cpu().to(torch.float64).reshape(-1) for t in candidate)
    if rra.shape != cra.shape:
        return False, f"shape mismatch {tuple(rra.shape)} vs {tuple(cra.shape)}"
    float32_rule = (
        candidate[0].dtype != torch.float64 or reference[0].dtype != torch.float64
    )
    tol = _F32_TOL_PX if float32_rule else _F64_TOL_PX
    ref_nan = torch.isnan(rra) | torch.isnan(rdec)
    cand_nan = torch.isnan(cra) | torch.isnan(cdec)
    n = rra.numel()
    extra = int((cand_nan & ~ref_nan).sum())
    missing = int((ref_nan & ~cand_nan).sum())
    ok_mask = ~ref_nan & ~cand_nan
    dra = torch.remainder(cra - rra + 180.0, 360.0) - 180.0
    dra = dra * torch.cos(torch.deg2rad(rdec))
    err = torch.hypot(dra, cdec - rdec)[ok_mask] / _PIXEL_SCALE_DEG
    max_err = float(err.max()) if err.numel() else 0.0
    msg = (
        f"max sky offset {max_err:.3e} px over {int(ok_mask.sum())}/{n} finite "
        f"(tol {tol:g} px, {'float32' if float32_rule else 'float64'}); "
        f"NaN: ref {int(ref_nan.sum())}, cand-only {extra}, ref-only {missing}"
    )
    ok = extra == 0 and missing == 0 and max_err <= tol
    return ok, msg
