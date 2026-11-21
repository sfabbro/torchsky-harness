"""Backend-matrix adapter: catalogue radius search (N vs N, 2 arcsec, ~5 deg field).

Exercises the zone-index engine behind ``torchsky.catalogs.match_within`` /
``SkyIndex.query_within``: ``ZoneIndex(right)`` (sort by zone and RA) followed by
``zone_query(index, left, radius, find="all")``. The timed call is index build plus
query, i.e. what ``match_within`` does minus input validation.

The internal pair (``ZoneIndex`` + ``zone_query``) is called directly instead of the
public ``match_within`` because the public API relocates MPS inputs to the CPU in
float64 (``sky_index._coords``: "MPS has no float64"), so ``match_within(device=mps)``
would time a CPU run. ``zone_query`` itself runs on any device in the input dtype.
"""

from __future__ import annotations

import contextlib
import math
import types
from collections.abc import Callable, Iterator
from typing import Any

import torch

from torchsky.catalogs import _zones

SIZES = {"small": 1_000, "medium": 100_000, "large": 1_000_000}
DTYPES = {"cpu": torch.float64, "cuda": torch.float64, "mps": torch.float32}
_SEEDS = {"small": 31_001, "medium": 31_002, "large": 31_003}

RADIUS_ARCSEC = 2.0
_RA0, _DEC0, _SIDE_DEG = 150.0, 2.0, 5.0
# Fraction of right rows placed near a left row (true counterparts, 0.5" scatter),
# so the pair count is not dominated by chance coincidences at small N.
_COUNTERPART_FRAC = 0.5
_SCATTER_ARCSEC = 0.5

# Pairs that differ between reference and candidate are accepted only when their
# float64 separation lies within this distance of the radius. float64: none
# (both paths decide on the same atan2 separation, padded prefilter). float32:
# positions are float32-representable in every dtype (see make_inputs), so only
# arithmetic differs. Unit-vector components carry ~1 ulp (6e-8); the atan2
# separation's cross product is a difference of O(1) products, so a ~1e-5 rad
# angle carries ~4 ulp ~ 2.4e-7 rad ~ 0.05 arcsec of error (seen on the MPS smoke
# run as max |dsep| on common pairs). 0.2 arcsec leaves 4x headroom.
_BORDER_F64 = 0.0
_BORDER_F32 = 0.2

NOTES = (
    "Times ZoneIndex(right_ra, right_dec, zone_height_deg=h) + "
    "zone_query(index, left_ra, left_dec, radius, find='all') from "
    "torchsky.catalogs._zones (the engine of match_within/SkyIndex.query_within; the public "
    "API is bypassed because it moves MPS work to CPU float64). N left vs N right in a "
    "5x5 deg field at (150, 2), radius 2 arcsec, half the right rows are 0.5-arcsec "
    "counterparts of left rows. h is fixed per size in make_inputs (default_zone_height_deg "
    "with occupied_spacing_deg, as match_within does). Native gate: zone_query calls "
    "_zones._native_ext() once and takes the xmatch_count/xmatch_fill kernel only for CPU "
    "float64 with accept=None and stats=None; that is the only native gate on this path "
    "(ZoneIndex, segments, _blocked_separation are pure torch; native_knn is not reached "
    "for find='all'). 'eager'/'compile' monkeypatch _zones._native_ext to return None for "
    "the call; the smoke run counts _cpp_ext.xmatch_* calls (0 for eager). 'native' (CPU "
    "only) is the same call unpatched. 'compile' is torch.compile(fn) with default dynamic "
    "shapes: the path is data-dependent (int(count.sum()) host sync, ragged "
    "repeat_interleave, boolean-mask filtering, nonzero, Python batch loop over candidate "
    "ranges), so Dynamo graph-breaks into many small graphs and recompiles per size; "
    "'compile_fullgraph' (fullgraph=True) is expected to raise. Output: (i, j) pairs and "
    "float64 separations on CPU; check() compares pair sets order-independently, allowing "
    "only pairs within 0.2 arcsec of the radius to differ in float32 (float32 separations "
    "themselves are off by ~0.05 arcsec at 2 arcsec). Compiled variants use a separate code "
    "object each (_fresh) so the fullgraph compile does not reuse the default compile's "
    "Dynamo cache entry."
)


def _no_native() -> None:
    return None


def _fresh(fn: Callable, tag: str) -> Callable:
    """Copy of ``fn`` with its own code object.

    Dynamo caches compiled frames on the code object, so ``torch.compile(fn,
    fullgraph=True)`` after ``torch.compile(fn)`` silently reuses the graph-broken
    entry and never raises (torch 2.14.1). Each compiled variant gets its own copy.
    """
    code = fn.__code__.replace(co_name=f"{fn.__name__}_{tag}")
    return types.FunctionType(
        code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__
    )


@contextlib.contextmanager
def force_pure_torch() -> Iterator[None]:
    """Make the zone engine's native extension look unavailable inside the block."""
    original = _zones._native_ext
    _zones._native_ext = _no_native
    try:
        yield
    finally:
        _zones._native_ext = original


def _field(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform on the sphere inside the box (uniform in RA and sin(dec))."""
    s0 = math.sin(math.radians(_DEC0))
    s1 = math.sin(math.radians(_DEC0 + _SIDE_DEG))
    ra = _RA0 + _SIDE_DEG * torch.rand(n, generator=gen, dtype=torch.float64)
    u = torch.rand(n, generator=gen, dtype=torch.float64)
    dec = torch.rad2deg(torch.asin(s0 + (s1 - s0) * u))
    return ra, dec


def _scene(n: int, seed: int) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    lra, ldec = _field(n, gen)
    rra, rdec = _field(n, gen)
    m = int(n * _COUNTERPART_FRAC)
    src = torch.randperm(n, generator=gen)[:m]
    off = torch.randn(m, 2, generator=gen, dtype=torch.float64) * (
        _SCATTER_ARCSEC / 3600.0
    )
    rdec[:m] = ldec[src] + off[:, 1]
    rra[:m] = lra[src] + off[:, 0] / torch.cos(torch.deg2rad(ldec[src]))
    perm = torch.randperm(n, generator=gen)
    rra, rdec = rra[perm], rdec[perm]
    # Round through float32 so every device sees the same positions; only the
    # arithmetic precision differs between float64 and float32 runs.
    return tuple(t.to(torch.float32).to(torch.float64) for t in (lra, ldec, rra, rdec))


def make_inputs(size_key: str, device: torch.device, dtype: torch.dtype) -> dict:
    n = SIZES[size_key]
    lra, ldec, rra, rdec = _scene(n, _SEEDS[size_key])
    h = _zones.default_zone_height_deg(
        n, RADIUS_ARCSEC, _zones.occupied_spacing_deg(rra, rdec)
    )

    def put(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype).to(device).contiguous()

    return {
        "left_ra": put(lra),
        "left_dec": put(ldec),
        "right_ra": put(rra),
        "right_dec": put(rdec),
        "radius": put(torch.full((n,), RADIUS_ARCSEC, dtype=torch.float64)),
        "zone_height_deg": float(h),
        "n_right": n,
    }


def items(size_key: str) -> int:
    return SIZES[size_key]


def _pure(inputs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = _zones.ZoneIndex(
        inputs["right_ra"],
        inputs["right_dec"],
        zone_height_deg=inputs["zone_height_deg"],
    )
    return _zones.zone_query(
        index, inputs["left_ra"], inputs["left_dec"], inputs["radius"], find="all"
    )


def _wrap(fn: Callable[[dict], Any], *, pure_torch: bool) -> Callable[[dict], dict]:
    def run(inputs: dict) -> dict:
        ctx = force_pure_torch() if pure_torch else contextlib.nullcontext()
        with ctx:
            li, rj, sep = fn(inputs)
        return {"li": li, "rj": rj, "sep": sep, "n_right": inputs["n_right"]}

    return run


def variants(device: torch.device) -> dict[str, Callable[[dict], Any]]:
    device = torch.device(device)
    out: dict[str, Callable[[dict], Any]] = {
        "eager": _wrap(_pure, pure_torch=True),
        "compile": _wrap(torch.compile(_fresh(_pure, "compile")), pure_torch=True),
        "compile_fullgraph": _wrap(
            torch.compile(_fresh(_pure, "fullgraph"), fullgraph=True), pure_torch=True
        ),
    }
    if device.type == "cpu" and _zones._native_ext() is not None:
        out["native"] = _wrap(_pure, pure_torch=False)
    return out


def _keys(res: dict) -> tuple[torch.Tensor, torch.Tensor]:
    li = res["li"].detach().to("cpu").to(torch.int64)
    rj = res["rj"].detach().to("cpu").to(torch.int64)
    sep = res["sep"].detach().to("cpu").to(torch.float64)
    key = li * int(res["n_right"]) + rj
    order = torch.argsort(key)
    return key[order], sep[order]


def check(reference: dict, candidate: dict) -> tuple[bool, str]:
    rk, rs = _keys(reference)
    ck, cs = _keys(candidate)
    if rk.numel() and bool((rk[1:] == rk[:-1]).any()):
        return False, "duplicate pairs in reference"
    if ck.numel() and bool((ck[1:] == ck[:-1]).any()):
        return False, "duplicate pairs in candidate"
    only_r = ~torch.isin(rk, ck)
    only_c = ~torch.isin(ck, rk)
    low = candidate["sep"].dtype in (torch.float32, torch.float16, torch.bfloat16)
    border = _BORDER_F32 if low else _BORDER_F64
    diff_sep = torch.cat((rs[only_r], cs[only_c]))
    n_diff = int(diff_sep.numel())
    worst = float((diff_sep - RADIUS_ARCSEC).abs().max()) if n_diff else 0.0
    ok = n_diff == 0 or worst <= border
    msg = (
        f"pairs ref={rk.numel()} cand={ck.numel()} only_ref={int(only_r.sum())} "
        f"only_cand={int(only_c.sum())}"
    )
    if n_diff:
        msg += f" max|sep-r| of differing={worst:.3e} arcsec (allowed {border:g})"
    common = torch.isin(ck, rk)
    if bool(common.any()):
        err = (cs[common] - rs[torch.isin(rk, ck)]).abs().max()
        msg += f"; max|dsep| common={float(err):.2e} arcsec"
    return ok, msg
