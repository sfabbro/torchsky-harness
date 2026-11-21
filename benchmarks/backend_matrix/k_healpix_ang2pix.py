"""Backend-matrix adapter: HEALPix ``ang2pix`` (NESTED, lon/lat degrees, nside 1024).

Exercises ``torchsky.sphere.healpix.ang2pix(nside, lon, lat, nest=True, lonlat=True)``,
which routes to ``ang2pix_nested``. That function takes the native dispatcher op
``torch.ops.torchsky.healpix_ang2pix_nested`` only when
``torchsky._ops.has_fixed_dispatcher()`` is true and both inputs are on CPU;
otherwise it runs the pure-torch path in the same function body.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import torch

import torchsky._ops as _ts_ops
from torchsky.sphere import healpix as _hp

NSIDE = 1024
SIZES = {"small": 100, "medium": 10_000, "large": 1_000_000}
DTYPES = {"cpu": torch.float64, "cuda": torch.float64, "mps": torch.float32}
_SEEDS = {"small": 11_001, "medium": 11_002, "large": 11_003}

# float32 boundary-tie budget (MPS only). The pure-torch kernel maps a point to
# continuous face coordinates nside*(0.5 + tt) +/- nside*0.75*z with magnitude
# up to ~4.5*nside = 4608; float32 spacing there is 2**-11 ~ 4.9e-4 pixel units,
# plus input rounding of lon in [256, 360) (half-ulp 1.5e-5 deg, i.e.
# 1.5e-5 * 4/360 * 1024 ~ 1.7e-4 pixel units). A uniformly random point
# flips across a pixel edge with probability ~ |coordinate error| per axis, so the
# expected mismatch fraction is O(1e-3). Allow 1e-2 (one decade of headroom) and
# require every mismatched pixel to be a HEALPix neighbour of the reference.
# Measured (torch 2.14.1, M2 MPS eager): 2/1e4 and 192/1e6 (1.9e-4), all neighbours.
_F32_TIE_FRACTION = 1e-2

NOTES = (
    "Calls torchsky.sphere.healpix.ang2pix(nside=1024, lon, lat, nest=True, lonlat=True) "
    "-> ang2pix_nested. 'eager' and 'compile' run with torchsky._ops.has_fixed_dispatcher "
    "monkeypatched to return False for the duration of the call, which makes "
    "ang2pix_nested skip torch.ops.torchsky.healpix_ang2pix_nested and run its pure-torch "
    "body (verified by reading the code and by counting dispatcher calls in the smoke run). "
    "'native' (CPU only) calls the same public function unpatched. 'compile' is "
    "torch.compile(fn, dynamic=False) of the same pure-torch function; the patch is applied "
    "outside the compiled callable so Dynamo inlines the constant-False replacement. "
    "Graph breaks: _validate_lonlat does bool(tensor) (data-dependent), the equatorial/polar "
    "split uses `if mask.any():` and boolean-mask indexing/assignment (dynamic output shapes), "
    "so Dynamo splits the function into several graphs; 'compile_fullgraph' is expected to "
    "raise. Inputs are float64 on CPU/CUDA and float32 on MPS (torchsky's work dtype); "
    "check() is exact except on MPS, where neighbour-only boundary ties up to "
    f"{_F32_TIE_FRACTION:g} of points are allowed."
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
    lat = torch.clamp(lat.to(dtype), -90.0, 90.0)
    return {
        "nside": NSIDE,
        "lon": lon.to(dtype).to(device).contiguous(),
        "lat": lat.to(device).contiguous(),
    }


def items(size_key: str) -> int:
    return SIZES[size_key]


def _pure(inputs: dict) -> torch.Tensor:
    return _hp.ang2pix(
        inputs["nside"], inputs["lon"], inputs["lat"], nest=True, lonlat=True
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
    float32_rule = candidate.device.type == "mps"
    # Two-step transfer: MPS .to('cpu', dtype) in one call returns garbage for
    # float64 targets in torch 2.14.1; .cpu() then .to(dtype) is correct.
    ref = reference.detach().cpu().to(torch.int64).reshape(-1)
    cand = candidate.detach().cpu().to(torch.int64).reshape(-1)
    if ref.shape != cand.shape:
        return False, f"shape mismatch {tuple(ref.shape)} vs {tuple(cand.shape)}"
    bad = ref != cand
    nbad = int(bad.sum())
    n = ref.numel()
    if nbad == 0:
        return True, f"exact ({n} pixels)"
    npix = 12 * NSIDE * NSIDE
    if bool(((cand < 0) | (cand >= npix)).any()):
        return False, f"{nbad}/{n} mismatches, some out of range [0, {npix})"
    nb = _hp.neighbors(NSIDE, ref[bad], nest=True)  # (nbad, 8), -1 for missing
    all_neighbours = bool((nb == cand[bad].unsqueeze(1)).any(dim=1).all())
    frac = nbad / n
    msg = f"{nbad}/{n} mismatches (frac {frac:.2e}), all neighbours={all_neighbours}"
    if not float32_rule:
        return False, msg + "; exact match required for float64"
    ok = all_neighbours and frac <= _F32_TIE_FRACTION
    return ok, msg + f"; float32 boundary-tie budget {_F32_TIE_FRACTION:g}"
