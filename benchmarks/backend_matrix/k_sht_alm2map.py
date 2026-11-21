"""Backend-matrix adapter: scalar HEALPix alm2map (spherical harmonic synthesis).

Calls torchsky's real implementation, ``torchsky.sphere.spectral.alm2map``
(the function behind ``torchsky.sphere.compat.alm2map``), RING ordering, no
pixel window, ``mmax = lmax = 2 * nside``.

Path selection (torchsky main 6c0a911a, default environment):

* ``TORCHSKY_SCALAR_ALM2MAP_RING_TORCH=auto`` with
  ``TORCHSKY_SCALAR_RING_AUTO_MIN_BYTES=0`` always selects
  ``_scalar_alm2map_ring_torch`` (the opt-in ``TORCHSKY_SCALAR_ALM2MAP_CPP``
  direct kernel is never reached).
* Legendre synthesis: ``torch.ops.torchsky.spectral_scalar_interpolate_recurrence``
  (via ``ScalarSHTInterpolate``) when ``torchsky._ops.has_spectral_dispatcher()``
  and device is CPU; otherwise the chunked pure-torch recurrence
  ``_scalar_ring_legendre_synthesize``.
* ring synthesis: ``torch.ops.torchsky.spectral_ring_fourier_synthesis`` when the
  dispatcher is present, device is CPU and ``mmax <= 1024``; otherwise
  ``torch.fft.ifft`` per ring group with alias folding (``_accumulate_add``).

Every native branch is gated on a live lookup of
``torchsky._ops.has_spectral_dispatcher`` / ``has_fixed_dispatcher`` (module
attribute, never a by-name import), so ``force_pure_torch()`` patching those two
attributes removes all ``torch.ops.torchsky`` calls from the transform.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Callable, Iterator

import torch

import torchsky._ops as _ts_ops
from torchsky.sphere import spectral as _spectral

SIZES = {"small": 64, "medium": 256, "large": 512}  # nside; lmax = mmax = 2 * nside
DTYPES = {"cpu": torch.float64, "cuda": torch.float64, "mps": torch.float32}
NOTES = (
    "torchsky.sphere.spectral.alm2map (== sphere.compat.alm2map), scalar, RING, "
    "no pixwin, lmax=mmax=2*nside; input alm complex128 (cpu/cuda) or complex64 "
    "(mps), m=0 imaginary parts zero; items = npix. "
    "Pure-torch switch: force_pure_torch() patches torchsky._ops.has_spectral_dispatcher "
    "and has_fixed_dispatcher to a constant False for the duration of the call "
    "(every native branch reads them live), giving the chunked Legendre "
    "recurrence (_scalar_ring_legendre_synthesize) + torch.fft.ifft ring "
    "synthesis. native = default CPU dispatch: "
    "torchsky::spectral_scalar_interpolate_recurrence + "
    "torchsky::spectral_ring_fourier_synthesis. "
    "Shared caches (ring layout, ring grouping, phi0 phases, alias indices, pixel "
    "angles) are pure geometry keyed by nside/mmax/nph only, identical for every "
    "backend; the pure path has no Legendre plan cache (recurrence recomputed per "
    "call), so no native plan can leak into eager. "
    "Graph breaks (dynamo.explain, nside 64: 10 graphs / 9 breaks, 0 torchsky "
    "nodes): Tensor.item() on the cached ring layout in _ring_fourier_synthesis "
    "(capture_scalar_outputs=False), threading.RLock in "
    "SpectralCacheManager.get_or_build, data-dependent "
    "`if torch.isfinite(out).all()`. _accumulate_add (alias folding, one call per "
    "polar ring group with a different shape) hits dynamo recompile_limit=8 under "
    "dynamic=False and runs eager for the remaining groups. The Legendre scan over "
    "ell and the per-ring-group loop are unrolled (~5.6k traced ops at lmax=128, "
    "first compile ~75 s on CPU; grows with lmax). compile_fullgraph fails "
    "(RLock). Inductor has no complex codegen (falls back to aten). "
    "MPS: float32/complex64; the Legendre recurrence runs in float64 on the host "
    "(_scan_device) and each per-ell slice is copied to MPS; complex index_add is "
    "split into real/imag (_accumulate_add). compile on MPS fails "
    "(InductorError 'float64 is not supported by MPS': the float64 host scan "
    "result is device_put to MPS before the float32 cast). "
    "Eager CPU float64 at large (nside 512, lmax 1024) is ~31 s per call on a "
    "shared M2 (single probe; medium ~2.2 s), far over the 5 s budget; native "
    "~0.15 s."
)

_ENV_DEFAULTS = {
    "TORCHSKY_SCALAR_ALM2MAP_RING_TORCH": "auto",
    "TORCHSKY_SCALAR_RING_AUTO_MIN_BYTES": "0",
}

_NATIVE_SPECTRAL = _ts_ops.has_spectral_dispatcher
_NATIVE_FIXED = _ts_ops.has_fixed_dispatcher


def _native_off() -> bool:
    # One persistent object: torch.compile guards on the patched attribute's
    # identity, so reusing it avoids recompiles across calls.
    return False


@contextlib.contextmanager
def force_pure_torch() -> Iterator[None]:
    """Disable every torchsky native branch reachable from the SHT for one call."""
    saved = (_ts_ops.has_spectral_dispatcher, _ts_ops.has_fixed_dispatcher)
    _ts_ops.has_spectral_dispatcher = _native_off
    _ts_ops.has_fixed_dispatcher = _native_off
    try:
        yield
    finally:
        _ts_ops.has_spectral_dispatcher, _ts_ops.has_fixed_dispatcher = saved


def native_available() -> bool:
    return bool(_NATIVE_SPECTRAL())


def _check_env() -> None:
    for key, default in _ENV_DEFAULTS.items():
        value = os.environ.get(key, default)
        if value != default:
            raise RuntimeError(
                f"{key}={value!r} changes the SHT path; unset it for the backend matrix"
            )


def _lmax(nside: int) -> int:
    return 2 * int(nside)


def _complex_of(dtype: torch.dtype) -> torch.dtype:
    return {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
        torch.complex64: torch.complex64,
        torch.complex128: torch.complex128,
    }[dtype]


def make_inputs(size_key: str, device: torch.device, dtype: torch.dtype) -> dict:
    nside = int(SIZES[size_key])
    lmax = _lmax(nside)
    nalm = (lmax + 1) * (lmax + 2) // 2
    gen = torch.Generator(device="cpu").manual_seed(20261121 + nside)
    re = torch.randn(nalm, generator=gen, dtype=torch.float64)
    im = torch.randn(nalm, generator=gen, dtype=torch.float64)
    im[: lmax + 1] = 0.0  # m = 0 block (healpy order): real for a real map
    alm = torch.complex(re, im)
    return {
        # Cast on the host, then move: never a fused device+dtype .to() (MPS bug).
        "alm": alm.to(_complex_of(dtype)).to(device),
        "nside": nside,
        "lmax": lmax,
    }


def _alm2map(alm: torch.Tensor, nside: int, lmax: int) -> torch.Tensor:
    return _spectral.alm2map(alm, nside, lmax=lmax, mmax=lmax)


def _alm2map_fullgraph(alm: torch.Tensor, nside: int, lmax: int) -> torch.Tensor:
    # Same body as _alm2map but a distinct code object: dynamo caches compiled
    # frames per code object, so sharing one would let the fullgraph wrapper
    # silently reuse the graph-broken entry produced by the default compile.
    return _spectral.alm2map(alm, nside, lmax=lmax, mmax=lmax)


def variants(device: torch.device) -> dict[str, Callable[[dict], torch.Tensor]]:
    _check_env()
    device = torch.device(device)
    compiled = torch.compile(_alm2map, dynamic=False)
    compiled_full = torch.compile(_alm2map_fullgraph, dynamic=False, fullgraph=True)

    def eager(inputs: dict) -> torch.Tensor:
        with force_pure_torch():
            return _alm2map(inputs["alm"], inputs["nside"], inputs["lmax"])

    def compile_(inputs: dict) -> torch.Tensor:
        with force_pure_torch():
            return compiled(inputs["alm"], inputs["nside"], inputs["lmax"])

    def compile_fullgraph(inputs: dict) -> torch.Tensor:
        with force_pure_torch():
            return compiled_full(inputs["alm"], inputs["nside"], inputs["lmax"])

    out: dict[str, Callable[[dict], torch.Tensor]] = {
        "eager": eager,
        "compile": compile_,
        "compile_fullgraph": compile_fullgraph,
    }
    if device.type == "cpu" and native_available():

        def native(inputs: dict) -> torch.Tensor:
            return _alm2map(inputs["alm"], inputs["nside"], inputs["lmax"])

        out["native"] = native
    return out


def items(size_key: str) -> int:
    nside = int(SIZES[size_key])
    return 12 * nside * nside


def _nside_from_npix(npix: int) -> int:
    nside = int(round(math.sqrt(npix / 12.0)))
    if 12 * nside * nside != npix:
        raise ValueError(f"map length {npix} is not a HEALPix npix")
    return nside


def _unit_roundoff(t: torch.Tensor) -> float:
    real = t.real.dtype if t.is_complex() else t.dtype
    return float(torch.finfo(real).eps) / 2.0


def check(reference, candidate) -> tuple[bool, str]:
    """Relative-to-norm comparison of synthesized maps.

    Both are moved to the host first and then cast to float64 (two steps; a
    fused device+dtype ``.to()`` from MPS returns wrong values in torch 2.14.1).
    The error budget is ``C * u * (lmax + 1)`` relative to ``||map_ref||_2``
    with ``lmax = 2 * nside`` (this module's convention) and ``u`` the unit
    roundoff of the less precise operand: each ring Fourier coefficient sums
    O(lmax) recurrence terms and the inverse FFT adds O(log nph). C = 8 for
    float32/complex64 (MPS eager vs CPU-native float64 observed rel_l2 3.0e-7 at
    lmax=128 against a budget of 6.2e-5); for float64 the 1e-11 floor covers
    native-vs-torch differences in recurrence rescaling and summation order
    (observed rel_l2 2.5e-14 at lmax=128, 1.1e-13 at 512, 2.5e-13 at 1024).
    """
    if not isinstance(candidate, torch.Tensor) or not isinstance(
        reference, torch.Tensor
    ):
        return False, f"expected tensors, got {type(reference)} / {type(candidate)}"
    if tuple(candidate.shape) != tuple(reference.shape):
        return False, f"shape {tuple(candidate.shape)} != {tuple(reference.shape)}"
    if candidate.is_complex() or reference.is_complex():
        return False, "alm2map must return a real map"
    u = max(_unit_roundoff(reference), _unit_roundoff(candidate))
    ref = reference.detach().cpu().to(torch.float64)
    cand = candidate.detach().cpu().to(torch.float64)
    if not torch.isfinite(cand).all():
        return False, "non-finite values in candidate"
    lmax = _lmax(_nside_from_npix(int(ref.shape[-1])))
    ref_norm = float(torch.linalg.vector_norm(ref))
    diff = cand - ref
    rel_l2 = float(torch.linalg.vector_norm(diff)) / max(ref_norm, 1e-300)
    rel_max = float(diff.abs().max()) / max(float(ref.abs().max()), 1e-300)
    tol = max(8.0 * u * (lmax + 1), 1e-11)
    ok = rel_l2 <= tol
    return ok, f"rel_l2={rel_l2:.3e} rel_max={rel_max:.3e} tol={tol:.1e} lmax={lmax}"
