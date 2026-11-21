"""Backend-matrix adapter: HEALPix bilinear ``get_interp_val`` on an nside-256 RING map.

Exercises ``torchsky.sphere.healpix.get_interp_val(m, lon, lat, nest=False, lonlat=True)``.
On CPU with ``torchsky._ops.has_fixed_dispatcher()`` true (and no grad) it calls the
native op ``torch.ops.torchsky.healpix_get_interp_val_ring``; otherwise it runs the
pure-torch ``get_interp_weights`` (whose own native shortcut is gated by the same
switch) followed by a gather and a weighted sum.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import torch

import torchsky._ops as _ts_ops
from torchsky.sphere import healpix as _hp

NSIDE = 256
SIZES = {"small": 100, "medium": 10_000, "large": 1_000_000}
DTYPES = {"cpu": torch.float64, "cuda": torch.float64, "mps": torch.float32}
_SEEDS = {"small": 12_001, "medium": 12_002, "large": 12_003}
_MAP_SEED = 12_000

# Map values are uniform in [0, 1), so |error| <= sum_i |dw_i| (weights sum to 1).
# float64: native and pure-torch paths evaluate the same formulas; differences are
# a few ulp of the trig/division chain, so atol 1e-10 is ~1e5 ulp of headroom.
_ATOL_F64 = 1e-10
# float32: the azimuthal weight is frac(phi/dphi - shift/2) with phi/dphi up to
# 4*nside = 1024, where float32 spacing is 2**-13 ~ 1.2e-4; input rounding of lon
# in [256, 360) adds 1.5e-5 deg * 1024/360 ~ 4e-5; the polar-angle weight has the
# same order. Each of the two weight pairs can therefore move by ~2e-4 and the
# value by <= 2 * 2 * 2e-4 ~ 8e-4 in the worst case. Allow 2e-3.
# Measured (torch 2.14.1, M2): MPS float32 eager max 1.5e-4 at 1e6 points; CPU
# float64 eager vs native max 1.7e-13.
_ATOL_F32 = 2e-3

NOTES = (
    "Calls torchsky.sphere.healpix.get_interp_val(map_nside256_ring, lon, lat, nest=False, "
    "lonlat=True). 'eager' and 'compile' run with torchsky._ops.has_fixed_dispatcher "
    "monkeypatched to return False for the call; that disables both the "
    "healpix_get_interp_val_ring shortcut in get_interp_val and the "
    "healpix_get_interp_weights_ring shortcut in get_interp_weights, so the path is "
    "get_interp_weights pure torch (_ring_above, _get_ring_info2) + gather + weighted sum "
    "(verified by reading the code and by counting dispatcher calls in the smoke run). "
    "'native' (CPU only) is the same public call unpatched. 'compile' is "
    "torch.compile(fn, dynamic=False) of the same pure-torch function. Graph breaks: "
    "npix2nside uses int(shape) arithmetic (static, fine), _validate_lonlat does "
    "bool(tensor), get_interp_weights branches on `if torch.any(mask):` four times and uses "
    "boolean-mask gather/scatter (dynamic output shapes), so Dynamo splits into many graphs "
    "and 'compile_fullgraph' is expected to raise. Map and positions are float64 on "
    "CPU/CUDA and float32 on MPS; check() uses atol 1e-10 (float64) / 2e-3 (float32) on "
    "map values in [0, 1)."
)


def _no_native() -> bool:
    return False


@contextlib.contextmanager
def force_pure_torch() -> Iterator[None]:
    """Make torchsky's fixed-shape dispatcher look unavailable inside the block."""
    original = _ts_ops.has_fixed_dispatcher
    _ts_ops.has_fixed_dispatcher = _no_native
    try:
        yield
    finally:
        _ts_ops.has_fixed_dispatcher = original


def _uniform_sky(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.rand(n, generator=gen, dtype=torch.float64)
    v = torch.rand(n, generator=gen, dtype=torch.float64)
    lon = 360.0 * u
    lat = torch.rad2deg(torch.asin(torch.clamp(2.0 * v - 1.0, -1.0, 1.0)))
    return lon, lat


def make_inputs(size_key: str, device: torch.device, dtype: torch.dtype) -> dict:
    n = SIZES[size_key]
    lon, lat = _uniform_sky(n, _SEEDS[size_key])
    gen = torch.Generator(device="cpu").manual_seed(_MAP_SEED)
    m = torch.rand(12 * NSIDE * NSIDE, generator=gen, dtype=torch.float64)
    return {
        "map": m.to(dtype).to(device).contiguous(),
        "lon": lon.to(dtype).to(device).contiguous(),
        "lat": torch.clamp(lat.to(dtype), -90.0, 90.0).to(device).contiguous(),
    }


def items(size_key: str) -> int:
    return SIZES[size_key]


def _pure(inputs: dict) -> torch.Tensor:
    return _hp.get_interp_val(
        inputs["map"], inputs["lon"], inputs["lat"], nest=False, lonlat=True
    )


def _pure_fullgraph(inputs: dict) -> torch.Tensor:
    # Distinct code object on purpose: Dynamo caches compiled frames per code
    # object, so torch.compile(_pure, fullgraph=True) after torch.compile(_pure)
    # would silently reuse the graph-broken cache entry instead of raising.
    return _pure(inputs)


def _under_pure_torch(fn: Callable[[dict], Any]) -> Callable[[dict], Any]:
    def run(inputs: dict) -> Any:
        with force_pure_torch():
            return fn(inputs)

    return run


def variants(device: torch.device) -> dict[str, Callable[[dict], Any]]:
    device = torch.device(device)
    out: dict[str, Callable[[dict], Any]] = {
        "eager": _under_pure_torch(_pure),
        "compile": _under_pure_torch(torch.compile(_pure, dynamic=False)),
        "compile_fullgraph": _under_pure_torch(
            torch.compile(_pure_fullgraph, dynamic=False, fullgraph=True)
        ),
    }
    if device.type == "cpu" and _ts_ops.has_fixed_dispatcher():
        out["native"] = _pure
    return out


def check(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[bool, str]:
    if reference.shape != candidate.shape:
        return (
            False,
            f"shape mismatch {tuple(reference.shape)} vs {tuple(candidate.shape)}",
        )
    low = candidate.dtype in (torch.float32, torch.float16, torch.bfloat16)
    atol = _ATOL_F32 if low else _ATOL_F64
    # Two-step transfer: MPS .to('cpu', dtype) in one call returns garbage for
    # float64 targets in torch 2.14.1; .cpu() then .to(dtype) is correct.
    ref = reference.detach().cpu().to(torch.float64)
    cand = candidate.detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(cand).all()):
        return False, "non-finite values in candidate"
    err = (ref - cand).abs()
    max_err = float(err.max()) if err.numel() else 0.0
    ok = max_err <= atol
    return (
        ok,
        f"max|err|={max_err:.3e} (atol {atol:g}, {'float32' if low else 'float64'})",
    )
