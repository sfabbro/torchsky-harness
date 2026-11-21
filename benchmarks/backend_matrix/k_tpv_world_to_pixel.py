"""Backend-matrix adapter: TPV WCS ``world_to_pixel`` (Newton inverse of the TPV polynomial).

Exercises ``torchsky.WCS(header).world_to_pixel(ra, dec)`` on a 1024x1024 TPV
header. The TPV inverse ``torchsky.wcs.tpv.TPV.invert`` takes the native
``torchsky._cpp_ext.wcs_tpv_invert`` kernel only when the class flag
``TPV._USE_NATIVE_INVERT`` is true and the inputs are CPU float64 with no
gradient requested; otherwise it runs the pure-torch Newton sweeps in the same
function body. No other native op sits on this path.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torchsky
from torchsky.wcs.tpv import TPV

SIZES = {"small": 100, "medium": 10_000, "large": 1_000_000}
DTYPES = {"cpu": torch.float64, "cuda": torch.float64, "mps": torch.float32}
_SEEDS = {"small": 21_001, "medium": 21_002, "large": 21_003}


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
    # tests/test_wcs_tpv.py::_well_conditioned_tpv_header: the fixture torchsky
    # checks against astropy. TPV axis 2 is a polynomial in (eta, xi), so the raw
    # _tpv_header() (PV2_1=0, PV2_2=1) makes eta ~ xi at linear order: a
    # near-singular Jacobian, 27% of in-image points round-trip to a different
    # root (up to ~800 px) and a 1-ulp RA change moves the solution by 0.2 px.
    # That would benchmark a degenerate solve, so use the conditioned variant.
    header = _tpv_header()
    header["PV2_1"] = 1.0
    header["PV2_2"] = 2.0e-4
    return header


HEADER = _bench_header()
# Smallest CD column norm (deg / pixel); converts world-plane quantities to pixels.
_PIXEL_SCALE_DEG = min(
    math.hypot(HEADER["CD1_1"], HEADER["CD2_1"]),
    math.hypot(HEADER["CD1_2"], HEADER["CD2_2"]),
)

# float64 budget: TPV.invert stops when the intermediate-plane residual is
# <= tol = 1e-11 deg. Two solutions of the same target that both meet it differ by
# <= 2 * tol / sigma_min(J); this header's Jacobian is identity + O(1e-4), so
# sigma_min ~ 1 and the gap is <= 2e-11 deg / 2.8e-4 deg/px = 7.1e-8 px. Round up
# to 1e-7 px. (Measured eager vs native: 2.3e-13 px at 1e5 points.)
_F64_TOL_PX = 1.0e-7
# float32 budget (MPS): torchsky's own float32 contract,
# torchsky.wcs._core_support._MPS_PIXEL_IDENTIFICATION_LIMIT_PX = 0.5 px; WCS warns
# when its float32 error estimate reaches it. For this header the estimate is
# 0.10 px and the measured max error vs CPU float64 is ~0.10 px.
_F32_TOL_PX = 0.5

NOTES = (
    "Calls torchsky.WCS(header).world_to_pixel(ra, dec) (origin 0) on the 1024x1024 TPV "
    "fixture tests/test_wcs_tpv.py::_well_conditioned_tpv_header (= _tpv_header() with "
    "PV2_1=1, PV2_2=2e-4; the raw _tpv_header() has a near-singular linear part and 27% "
    "wrong-root round trips, so it is not benchmarked). Inputs: points uniform over the "
    "image mapped through pixel_to_world in CPU float64, then cast to the device dtype; "
    "the WCS is moved with WCS.to(device) (casts cached params to float32 on MPS). "
    "Switch: torchsky.wcs.tpv.TPV._USE_NATIVE_INVERT (class flag). 'eager'/'compile'/"
    "'compile_fullgraph' set it False for the call; 'native' (CPU only, float64, no grad) "
    "sets it True and runs _cpp_ext.wcs_tpv_invert. 'compile' is torch.compile(fn, "
    "dynamic=False) of the same function; the flag is set outside the compiled callable. "
    "Graph breaks in TPV.invert: bool(update.any()) per Newton sweep, rows = rows[still] "
    "(data-dependent shape; with dynamic=False each new active-row count is a recompile "
    "of the resumed frame), int((~valid).sum().item()) after the loop, so "
    "'compile_fullgraph' is expected to raise. This header converges in one Newton sweep "
    "from the affine seed, so row shrinkage is barely exercised (the raw header takes ~10 "
    "sweeps). A compile-friendly formulation would need: a fixed iteration count (or "
    "torch.while_loop/cond) with masked torch.where updates over all points instead of row "
    "compaction, no host syncs (.any()/.item()/int()), trace bookkeeping and "
    "final_active computed outside the compiled region. Known MPS issue: TPV.invert uses "
    "tol=1e-11 deg regardless of dtype, below float32 resolution of xi/eta, so points "
    "whose float32 residual is not exactly 0 come back NaN (~0.1% of points); check() "
    "reports that as a NaN-mask mismatch (smoke: 11/1e4 points). MPS compile (torch "
    "2.14.1): Inductor's MPS backend raises NotImplementedError on torch.hypot (used by "
    "TPV._evaluate), so 'compile' fails on MPS; fullgraph on MPS breaks first at "
    "WCS._mps_float32_error_pixels (`if not scale > 0.0`), which _as_wcs_coord runs on "
    "every MPS call (a device->host sync per call unless a precision warning already "
    "fired). Tolerances: float64 1e-7 px (Newton tol / "
    "pixel scale), float32 0.5 px (torchsky's MPS pixel-identification limit)."
)


def native_available() -> bool:
    from torchsky import _cpp_ext
    from torchsky._native import cpp

    return cpp is not None and getattr(_cpp_ext, "wcs_tpv_invert", None) is not None


@contextlib.contextmanager
def native_invert(enabled: bool) -> Iterator[None]:
    """Set ``TPV._USE_NATIVE_INVERT`` for the duration of the block."""
    original = TPV._USE_NATIVE_INVERT
    TPV._USE_NATIVE_INVERT = enabled
    try:
        yield
    finally:
        TPV._USE_NATIVE_INVERT = original


_WCS_CACHE: dict[str, torchsky.WCS] = {}


def _wcs_on(device: torch.device) -> torchsky.WCS:
    key = str(device)
    wcs = _WCS_CACHE.get(key)
    if wcs is None:
        wcs = torchsky.WCS(dict(HEADER)).to(device)
        _WCS_CACHE[key] = wcs
    return wcs


def _uniform_pixels(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand(n, generator=gen, dtype=torch.float64) * HEADER["NAXIS1"] - 0.5
    y = torch.rand(n, generator=gen, dtype=torch.float64) * HEADER["NAXIS2"] - 0.5
    return x, y


def make_inputs(size_key: str, device: torch.device, dtype: torch.dtype) -> dict:
    device = torch.device(device)
    x, y = _uniform_pixels(SIZES[size_key], _SEEDS[size_key])
    ra, dec = _wcs_on(torch.device("cpu")).pixel_to_world(x, y, origin=0)
    return {
        "wcs": _wcs_on(device),
        "ra": ra.to(dtype).to(device).contiguous(),
        "dec": dec.to(dtype).to(device).contiguous(),
    }


def items(size_key: str) -> int:
    return SIZES[size_key]


def _pure(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return inputs["wcs"].world_to_pixel(inputs["ra"], inputs["dec"], origin=0)


# Same body under a distinct code object: Dynamo caches compiled frames per code
# object, so compiling _pure twice would let the fullgraph wrapper reuse the
# graph-broken cache entries of 'compile' and never report the breaks.
def _pure_fullgraph(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return inputs["wcs"].world_to_pixel(inputs["ra"], inputs["dec"], origin=0)


def _with_native(fn: Callable[[dict], Any], enabled: bool) -> Callable[[dict], Any]:
    def run(inputs: dict) -> Any:
        with native_invert(enabled):
            return fn(inputs)

    return run


def variants(device: torch.device) -> dict[str, Callable[[dict], Any]]:
    device = torch.device(device)
    out: dict[str, Callable[[dict], Any]] = {
        "eager": _with_native(_pure, False),
        "compile": _with_native(torch.compile(_pure, dynamic=False), False),
        "compile_fullgraph": _with_native(
            torch.compile(_pure_fullgraph, dynamic=False, fullgraph=True), False
        ),
    }
    if device.type == "cpu" and native_available():
        out["native"] = _with_native(_pure, True)
    return out


def check(reference: Any, candidate: Any) -> tuple[bool, str]:
    # Copy to CPU before casting: on torch 2.14.1, mps_f32.to("cpu", torch.float64)
    # returns zeros.
    rx, ry = (t.detach().cpu().to(torch.float64).reshape(-1) for t in reference)
    cx, cy = (t.detach().cpu().to(torch.float64).reshape(-1) for t in candidate)
    if rx.shape != cx.shape:
        return False, f"shape mismatch {tuple(rx.shape)} vs {tuple(cx.shape)}"
    float32_rule = (
        candidate[0].dtype != torch.float64 or reference[0].dtype != torch.float64
    )
    tol = _F32_TOL_PX if float32_rule else _F64_TOL_PX
    ref_nan = torch.isnan(rx) | torch.isnan(ry)
    cand_nan = torch.isnan(cx) | torch.isnan(cy)
    n = rx.numel()
    extra = int((cand_nan & ~ref_nan).sum())
    missing = int((ref_nan & ~cand_nan).sum())
    ok_mask = ~ref_nan & ~cand_nan
    err = torch.hypot(cx - rx, cy - ry)[ok_mask]
    max_err = float(err.max()) if err.numel() else 0.0
    msg = (
        f"max |dpix| {max_err:.3e} px over {int(ok_mask.sum())}/{n} finite "
        f"(tol {tol:g} px, {'float32' if float32_rule else 'float64'}); "
        f"NaN: ref {int(ref_nan.sum())}, cand-only {extra}, ref-only {missing}"
    )
    ok = extra == 0 and missing == 0 and max_err <= tol
    return ok, msg
