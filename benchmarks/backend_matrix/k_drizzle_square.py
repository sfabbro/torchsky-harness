"""Backend-matrix adapter: exact square-kernel drizzle accumulation.

A source image (TAN, square pixels) is drizzled with ``pixfrac = 0.8`` onto a
target TAN grid rotated by 37 degrees with 1.2x coarser pixels (geometry of
``tests/test_map_drizzle_semantics.py::test_square_drizzle_native_matches_torch_path``,
scaled up). The timed call is
``torchsky.map.operators.geometry_drizzle._accumulate_quadrilateral_overlaps``, the
stage of ``exact_square_drizzle_project`` that owns the native shortcut
(``torch.ops.torchsky.drizzle_square_accumulate``).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from typing import Any

import torch

import torchsky
from torchsky.map.operators import geometry_drizzle as _gd

SIZES = {"small": 64, "medium": 512, "large": 2048}
DTYPES = {"cpu": torch.float64, "cuda": torch.float64, "mps": torch.float32}
CALL_STYLE = "dict"
_SEEDS = {"small": 31_001, "medium": 31_002, "large": 31_003}

PIXFRAC = 0.8
ROT_DEG = 37.0
SCALE_RATIO = 1.2  # target pixel / source pixel
_SRC_CDELT_DEG = 1.0e-4

NOTES = (
    "Square drizzle, pixfrac 0.8, source n^2 (n = 64/512/2048) onto a target TAN grid "
    "rotated 37 deg with 1.2x coarser pixels, n_tgt = ceil(1.2 n) + 4 so every drop lands "
    "inside the target (about 3-4 candidate target boxes per drop). make_inputs does what "
    "exact_square_drizzle_project does before accumulating: the pixfrac-scaled corners of "
    "every source pixel go through source.pixel_to_world -> target.world_to_pixel (float64 "
    "on CPU) and the [N,4] target-pixel quads are stored with numer = image*weight and "
    "weight (float32), so the WCS cost is excluded and the projection is separable. The "
    "timed call allocates three zeroed float32 maps and runs "
    "geometry_drizzle._accumulate_quadrilateral_overlaps on all N quads at once (no "
    "chunk_pixels source chunking; the torch path still batches 65,536 (CPU) / 1,048,576 "
    "(accelerator) candidate pairs internally), with the variance map, then applies the "
    "weight > 0 masking of exact_square_drizzle_project. Switch: module global "
    "geometry_drizzle._NATIVE_OVERLAP_ENABLED (same as env TORCHSKY_DRIZZLE_NATIVE=0), "
    "set False for the duration of each 'eager'/'compile' call and True for 'native'; "
    "'native' (CPU only, when torch.ops.torchsky.drizzle_square_accumulate exists) "
    "therefore calls the operator once per run. 'compile' is torch.compile(dynamic=False) "
    "of the same function; the torch path has data-dependent control flow (.tolist(), "
    "int()/bool() of tensors, repeat_interleave and boolean-mask indexing with "
    "data-dependent sizes, a Python loop over pair chunks), so Dynamo graph-breaks many "
    "times (3 breaks at small) and 'compile_fullgraph' raises (data-dependent guard at 'if not bool(finite.all())', torch 2.14.1). MPS caveat: torchsky's public "
    "exact square drizzle refuses MPS (require_float64); here the internal accumulator is "
    "run with float32 quads on MPS as a what-if. float32 vertices near |x| ~ n_tgt carry "
    "~eps32*n_tgt px of rounding and the grazing-overlap cutoff 16*eps*|x|*extent drops "
    "slivers up to ~4e-3 px^2 at n_tgt=2462, so MPS results differ from CPU at the "
    "1e-3..1e-2 level of the drop flux; check() scales its tolerance by "
    "the eps of the vertex dtype (returned as 'coord_eps') * n_tgt. Returns a dict of float32 "
    "maps numer / weight / variance_weight plus the float 'coord_eps'."
)


def _target_size(n_src: int) -> int:
    return int(math.ceil(SCALE_RATIO * n_src)) + 4


def _header(n: int, crpix: float, cdelt: float, rot_deg: float) -> dict:
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    return {
        "NAXIS": 2,
        "NAXIS1": n,
        "NAXIS2": n,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CRPIX1": crpix,
        "CRPIX2": crpix,
        "CD1_1": -cdelt * c,
        "CD1_2": cdelt * s,
        "CD2_1": cdelt * s,
        "CD2_2": cdelt * c,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }


@contextlib.contextmanager
def native_overlap(enabled: bool) -> Iterator[None]:
    """Set geometry_drizzle._NATIVE_OVERLAP_ENABLED inside the block."""
    previous = _gd._NATIVE_OVERLAP_ENABLED
    _gd._NATIVE_OVERLAP_ENABLED = enabled
    try:
        yield
    finally:
        _gd._NATIVE_OVERLAP_ENABLED = previous


def native_available() -> bool:
    from torchsky._native import cpp

    return cpp is not None and hasattr(torch.ops.torchsky, "drizzle_square_accumulate")


def make_inputs(size_key: str, device: torch.device, dtype: torch.dtype) -> dict:
    n_src = SIZES[size_key]
    n_tgt = _target_size(n_src)
    # Source centred on CRVAL; target centred there too with a sub-pixel offset so no
    # drop edge is pixel-aligned with the target grid.
    src_wcs = torchsky.WCS(_header(n_src, 0.5 * (n_src + 1), _SRC_CDELT_DEG, 0.0))
    tgt_wcs = torchsky.WCS(
        _header(n_tgt, 0.5 * (n_tgt + 1) + 0.17, _SRC_CDELT_DEG * SCALE_RATIO, ROT_DEG)
    )
    gen = torch.Generator(device="cpu").manual_seed(_SEEDS[size_key])
    yy, xx = torch.meshgrid(
        torch.arange(n_src, dtype=torch.float64),
        torch.arange(n_src, dtype=torch.float64),
        indexing="ij",
    )
    # Smooth structure plus noise; inverse-variance-like weights in [0.5, 3.5).
    image = (
        1.0
        + 0.5
        * torch.sin(2.0 * math.pi * xx / max(n_src / 4.0, 1.0))
        * torch.cos(2.0 * math.pi * yy / max(n_src / 5.0, 1.0))
        + 0.1 * torch.randn(n_src, n_src, generator=gen, dtype=torch.float64)
    ).to(torch.float32)
    weight = (torch.rand(n_src, n_src, generator=gen) * 3.0 + 0.5).to(torch.float32)
    # Exactly the corner construction of exact_square_drizzle_project.
    half = 0.5 * PIXFRAC
    x = xx.reshape(-1)
    y = yy.reshape(-1)
    sx = torch.stack((x - half, x + half, x + half, x - half), dim=1)
    sy = torch.stack((y - half, y - half, y + half, y + half), dim=1)
    ra, dec = src_wcs.pixel_to_world(sx.reshape(-1), sy.reshape(-1))
    qx, qy = tgt_wcs.world_to_pixel(ra, dec)
    qx = qx.reshape(-1, 4).to(torch.float64)
    qy = qy.reshape(-1, 4).to(torch.float64)
    if not (bool(torch.isfinite(qx).all()) and bool(torch.isfinite(qy).all())):
        raise RuntimeError("non-finite projected drizzle corners")
    if (
        float(qx.min()) < 0.0
        or float(qx.max()) > n_tgt - 1
        or float(qy.min()) < 0.0
        or float(qy.max()) > n_tgt - 1
    ):
        raise RuntimeError("drizzle drops leave the target grid; enlarge n_tgt")
    numer = (image * weight).reshape(-1)
    return {
        "quad_x": qx.to(dtype).to(device).contiguous(),
        "quad_y": qy.to(dtype).to(device).contiguous(),
        "numer": numer.to(device).contiguous(),
        "weight": weight.reshape(-1).to(device).contiguous(),
        "width": n_tgt,
        "height": n_tgt,
    }


def items(size_key: str) -> int:
    """Source pixels (drops) accumulated."""
    return SIZES[size_key] ** 2


def _pure(inputs: dict) -> dict[str, Any]:
    width = int(inputs["width"])
    height = int(inputs["height"])
    quad_x = inputs["quad_x"]
    flat_numer = torch.zeros(width * height, dtype=torch.float32, device=quad_x.device)
    flat_weight = torch.zeros_like(flat_numer)
    flat_variance = torch.zeros_like(flat_numer)
    _gd._accumulate_quadrilateral_overlaps(
        quad_x,
        inputs["quad_y"],
        inputs["numer"],
        inputs["weight"],
        width=width,
        height=height,
        flat_numer=flat_numer,
        flat_weight=flat_weight,
        flat_variance_weight=flat_variance,
    )
    numer = flat_numer.reshape(height, width)
    weight = flat_weight.reshape(height, width)
    variance = flat_variance.reshape(height, width)
    positive = weight > 0.0
    zero = torch.zeros_like(numer)
    return {
        "numer": torch.where(positive, numer, zero),
        "weight": torch.where(positive, weight, zero),
        "variance_weight": torch.where(positive, variance, zero),
        # Python float: run_matrix moves outputs to CPU before check(), so the
        # vertex precision cannot be read from the candidate's device.
        "coord_eps": float(torch.finfo(quad_x.dtype).eps),
    }


def _pure_fullgraph(inputs: dict) -> dict[str, Any]:
    # Distinct code object: Dynamo caches compiled frames per code object, so
    # torch.compile(_pure, fullgraph=True) after torch.compile(_pure) would reuse the
    # graph-broken cache entry instead of raising.
    return _pure(inputs)


def _with_native(fn: Callable[[dict], Any], enabled: bool) -> Callable[[dict], Any]:
    def run(inputs: dict) -> Any:
        with torch.no_grad(), native_overlap(enabled):
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


# Tolerance, relative to the largest reference value of each map (the image scale):
# * float64 vertices (CPU/CUDA): the overlap areas agree to ~1e-15; the maps are float32
#   sums of ~3 drops with different accumulation order (C++ loop vs scatter_add), i.e. a
#   few float32 ulp. 1e-5 is the bound used by the torchsky native-vs-torch test.
# * float32 vertices (MPS): a vertex at |x| <= n_tgt is rounded by eps32*n_tgt px, which
#   moves a 0.44 px^2 drop area by ~4*0.67*eps32*n_tgt/0.44 ~ 6 eps32*n_tgt (relative),
#   and the grazing-overlap cutoff 16*eps32*|x|*extent (extent ~0.94 px) discards up to
#   ~15*eps32*n_tgt px^2, i.e. ~34 eps32*n_tgt of a drop. Allow 64*eps*n_tgt.
_REL_TOL_F64 = 1.0e-5
_NAMES = ("numer", "weight", "variance_weight")


def check(reference: Any, candidate: Any) -> tuple[bool, str]:
    coord_eps = float(candidate.get("coord_eps", torch.finfo(torch.float64).eps))
    n_tgt = int(reference["numer"].shape[-1])
    if coord_eps <= torch.finfo(torch.float64).eps:
        rel_tol = _REL_TOL_F64
    else:
        rel_tol = max(_REL_TOL_F64, 64.0 * coord_eps * n_tgt)
    ok = True
    parts = []
    for name in _NAMES:
        ref_t, cand_t = reference[name], candidate[name]
        if ref_t.shape != cand_t.shape:
            return False, f"{name}: shape {tuple(ref_t.shape)} vs {tuple(cand_t.shape)}"
        # Two-step transfer: MPS .to("cpu", float64) in one call is wrong in torch 2.14.1.
        ref = ref_t.detach().cpu().to(torch.float64)
        cand = cand_t.detach().cpu().to(torch.float64)
        if not bool(torch.isfinite(cand).all()):
            return False, f"{name}: non-finite values in candidate"
        scale = max(1.0, float(ref.abs().max()))
        err = float((ref - cand).abs().max()) / scale
        total = abs(float(cand.sum()) - float(ref.sum())) / max(
            abs(float(ref.sum())), 1e-30
        )
        ok = ok and err <= rel_tol
        parts.append(f"{name} max|err|/scale={err:.2e} sum rel={total:.1e}")
    return ok, "; ".join(parts) + f" (tol {rel_tol:.1e}, coord eps {coord_eps:.1e})"
