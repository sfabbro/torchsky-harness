"""Backend-matrix adapter: scalar HEALPix map2alm (spherical harmonic analysis).

Calls torchsky's real implementation, ``torchsky.sphere.spectral.map2alm``
(the function behind ``torchsky.sphere.compat.map2alm``), RING ordering,
``iter=0``, no weights, ``mmax = lmax = 2 * nside``.

Path selection (torchsky main 6c0a911a, default environment):

* ``map2alm`` -> ``_map2alm_single_pass`` -> ``_scalar_map2alm_ring_torch``
  (``TORCHSKY_SCALAR_MAP2ALM_RING_TORCH=auto`` with
  ``TORCHSKY_SCALAR_RING_AUTO_MIN_BYTES=0`` always selects the ring path).
* ring FFT: ``_ring_fourier_modes`` uses ``torch.ops.torchsky.spectral_ring_fourier_modes``
  when ``torchsky._ops.has_spectral_dispatcher()`` and device is CPU and
  ``mmax <= 1024``; otherwise ``torch.fft.fft`` per ring group.
* Legendre integration: ``torch.ops.torchsky.spectral_scalar_integrate_recurrence``
  (via ``ScalarSHTIntegrate``) when the dispatcher is present and device is CPU;
  otherwise the chunked pure-torch recurrence ``_scalar_ring_legendre_integrate``.

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
    "torchsky.sphere.spectral.map2alm (== sphere.compat.map2alm), scalar, RING, "
    "iter=0, lmax=mmax=2*nside; items = npix. "
    "Pure-torch switch: force_pure_torch() patches torchsky._ops.has_spectral_dispatcher "
    "and has_fixed_dispatcher to a constant False for the duration of the call "
    "(every native branch reads them live), giving torch.fft ring FFT + chunked "
    "Legendre recurrence (_scalar_ring_legendre_integrate). native = default CPU "
    "dispatch: torchsky::spectral_ring_fourier_modes + "
    "torchsky::spectral_scalar_integrate_recurrence. "
    "Shared caches (ring layout, ring grouping, phi0 phases, pixel angles) are pure "
    "geometry keyed by nside/mmax only, identical for every backend; the pure path "
    "has no Legendre plan cache (recurrence recomputed per call), so no native plan "
    "can leak into eager. "
    "Graph breaks (dynamo.explain, nside 64: 7 graphs / 6 breaks, 0 torchsky nodes): "
    "threading.RLock in SpectralCacheManager.get_or_build ('Unsupported context "
    "manager'), data-dependent `if torch.isfinite(alm).all()`. The Legendre scan "
    "(a generator looped over ell) and the per-ring-group loop are unrolled: "
    "~5.4k traced ops at lmax=128, first compile ~70 s on CPU; compile time grows "
    "~lmax^1+ so medium/large compiles may take very long. compile_fullgraph fails "
    "(RLock). Inductor has no complex codegen (falls back to aten). "
    "MPS: float32/complex64; the Legendre recurrence runs in float64 on the host "
    "(_scan_device) and each per-ell slice is copied to MPS. compile on MPS fails: "
    "inductor lowers the float64(cpu)->float32(mps) _to() as prims.device_put "
    "before the dtype cast, and MPS rejects float64. "
    "Eager CPU float64 at large (nside 512, lmax 1024) is ~14 s per call on a "
    "shared M2 (single probe), over the 5 s budget; native ~0.13 s."
)

_ENV_DEFAULTS = {
    "TORCHSKY_SCALAR_MAP2ALM_RING_TORCH": "auto",
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


def make_inputs(size_key: str, device: torch.device, dtype: torch.dtype) -> dict:
    nside = int(SIZES[size_key])
    npix = 12 * nside * nside
    gen = torch.Generator(device="cpu").manual_seed(20261120 + nside)
    values = torch.randn(npix, generator=gen, dtype=torch.float64)
    return {
        # Cast on the host, then move: never a fused device+dtype .to() (MPS bug).
        "map": values.to(dtype).to(device),
        "nside": nside,
        "lmax": _lmax(nside),
    }


def _map2alm(map_values: torch.Tensor, nside: int, lmax: int) -> torch.Tensor:
    return _spectral.map2alm(map_values, nside=nside, lmax=lmax, mmax=lmax)


def _map2alm_fullgraph(map_values: torch.Tensor, nside: int, lmax: int) -> torch.Tensor:
    # Same body as _map2alm but a distinct code object: dynamo caches compiled
    # frames per code object, so sharing one would let the fullgraph wrapper
    # silently reuse the graph-broken entry produced by the default compile.
    return _spectral.map2alm(map_values, nside=nside, lmax=lmax, mmax=lmax)


def variants(device: torch.device) -> dict[str, Callable[[dict], torch.Tensor]]:
    _check_env()
    device = torch.device(device)
    compiled = torch.compile(_map2alm, dynamic=False)
    compiled_full = torch.compile(_map2alm_fullgraph, dynamic=False, fullgraph=True)

    def eager(inputs: dict) -> torch.Tensor:
        with force_pure_torch():
            return _map2alm(inputs["map"], inputs["nside"], inputs["lmax"])

    def compile_(inputs: dict) -> torch.Tensor:
        with force_pure_torch():
            return compiled(inputs["map"], inputs["nside"], inputs["lmax"])

    def compile_fullgraph(inputs: dict) -> torch.Tensor:
        with force_pure_torch():
            return compiled_full(inputs["map"], inputs["nside"], inputs["lmax"])

    out: dict[str, Callable[[dict], torch.Tensor]] = {
        "eager": eager,
        "compile": compile_,
        "compile_fullgraph": compile_fullgraph,
    }
    if device.type == "cpu" and native_available():

        def native(inputs: dict) -> torch.Tensor:
            return _map2alm(inputs["map"], inputs["nside"], inputs["lmax"])

        out["native"] = native
    return out


def items(size_key: str) -> int:
    nside = int(SIZES[size_key])
    return 12 * nside * nside


def _lmax_from_nalm(nalm: int) -> int:
    # nalm = (L + 1)(L + 2) / 2 for mmax == lmax.
    lmax = int(round((math.sqrt(8.0 * nalm + 1.0) - 3.0) / 2.0))
    if (lmax + 1) * (lmax + 2) // 2 != nalm:
        raise ValueError(f"alm length {nalm} is not a triangular mmax=lmax layout")
    return lmax


def _unit_roundoff(t: torch.Tensor) -> float:
    real = t.real.dtype if t.is_complex() else t.dtype
    return float(torch.finfo(real).eps) / 2.0


def check(reference, candidate) -> tuple[bool, str]:
    """Relative-to-norm comparison of alm vectors.

    Both are cast to complex128 on CPU. The error budget is
    ``C * u * (lmax + 1)`` relative to ``||a_ref||_2`` where ``u`` is the unit
    roundoff of the less precise operand: the three-term Legendre recurrence
    accumulates O(lmax) roundings per coefficient and the ring FFT O(log nph)
    more. C = 8 for float32/complex64 (MPS eager vs CPU-native float64 observed
    rel_l2 2.6e-7 at lmax=128 against a budget of 6.2e-5, which is mostly the
    float32 rounding of the input map); for float64 the 1e-11 floor covers
    native-vs-torch differences in recurrence rescaling and summation order
    (observed rel_l2 2.4e-14 at lmax=128, 1.1e-13 at 512, 2.5e-13 at 1024).
    Transfers are host-first, so the candidate may be on any device and dtype.
    """
    if not isinstance(candidate, torch.Tensor) or not isinstance(
        reference, torch.Tensor
    ):
        return False, f"expected tensors, got {type(reference)} / {type(candidate)}"
    if tuple(candidate.shape) != tuple(reference.shape):
        return False, f"shape {tuple(candidate.shape)} != {tuple(reference.shape)}"
    u = max(_unit_roundoff(reference), _unit_roundoff(candidate))
    # Two-step move: torch 2.14.1 MPS returns garbage for the fused
    # complex64(mps) -> .to(device="cpu", dtype=complex128) conversion.
    ref = reference.detach().cpu().to(torch.complex128)
    cand = candidate.detach().cpu().to(torch.complex128)
    if not torch.isfinite(torch.view_as_real(cand)).all():
        return False, "non-finite values in candidate"
    lmax = _lmax_from_nalm(int(ref.shape[-1]))
    ref_norm = float(torch.linalg.vector_norm(ref))
    diff = cand - ref
    rel_l2 = float(torch.linalg.vector_norm(diff)) / max(ref_norm, 1e-300)
    rel_max = float(diff.abs().max()) / max(float(ref.abs().max()), 1e-300)
    tol = max(8.0 * u * (lmax + 1), 1e-11)
    ok = rel_l2 <= tol
    return ok, f"rel_l2={rel_l2:.3e} rel_max={rel_max:.3e} tol={tol:.1e} lmax={lmax}"
