"""Backend-matrix adapter: interpolating (gather) resample onto another WCS grid.

Times the sampling stage of torchsky's image-plane resampling operator
(``torchsky.map.image_plane_projection._resample_weighted_to_target``, inner
``_sample_on_source_grid``, non-variance branch) for a TAN -> TAN pair: stack the
weighted image and weight planes, sample them with
``torchsky.map.operators.resampling.sample_image_plane_batched_channels``
(``F.grid_sample`` for bilinear, torchsky's chunked separable sampler for Lanczos),
renormalise the edge-truncated kernel with ``in_bounds_kernel_sum`` and zero samples
outside the source. The kernel is ``lanczos3`` here; ``k_resample_bilinear.py`` runs
the same adapter with ``bilinear``. There is no native kernel for this path.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

import torchsky
from torchsky.map.operators.resampling import (
    in_bounds_kernel_sum,
    sample_image_plane_batched_channels,
    supports_in_bounds_kernel_sum,
)

KERNEL = "lanczos3"
CALL_STYLE = "dict"
SIZES = {"small": 256, "medium": 1024, "large": 4096}
DTYPES = {"cpu": torch.float64, "cuda": torch.float64, "mps": torch.float32}
_SEEDS = {"small": 41_001, "medium": 41_002, "large": 41_003}

ROT_DEG = 37.0
SCALE_RATIO = 0.9  # target pixel / source pixel
_SRC_CDELT_DEG = 1.0e-4


def _notes(kernel: str) -> str:
    sampler = (
        "F.grid_sample(mode='bilinear', align_corners=True, zeros padding) on the "
        "normalised grid"
        if kernel == "bilinear"
        else "torchsky custom_separable_sample -> _separable_sample_chunked (6x6 taps, "
        "chunks of ~116k samples from a 48 MB working-set budget, index_select + addcmul_ "
        "per tap, one host sync for the finite-coordinate check)"
    )
    return (
        f"Kernel {kernel}. Source n^2 (n = 256/1024/4096) TAN image; target n^2 TAN grid "
        "rotated 37 deg with 0.9x the source pixel size, so the target corners fall outside "
        "the source and the edge renormalisation / masking is exercised. make_inputs "
        "precomputes the target->source pixel map (target pixel -> pixel_to_world -> "
        "source world_to_pixel, float64 on CPU), then rounds it to float32-representable "
        "values for every device: the operator's samplers cast coordinates to the float32 "
        "plane dtype anyway, and the rounding makes in-bounds masks identical across "
        "devices so check() measures sampling arithmetic, not coordinate rounding. "
        "Coordinates are stored in DTYPES[device] (float64 CPU/CUDA, float32 MPS); planes "
        "are float32 (image*weight, weight) as in the operator. Timed: torch.stack of the "
        "two planes, normalised grid construction, "
        f"sample_image_plane_batched_channels ({sampler}), in_bounds_kernel_sum edge "
        "renormalisation, in-bounds and weight > 0 masking -- the body of "
        "_sample_on_source_grid for surface_brightness, no flux areas, no variance. "
        "Excluded: WCS mapping, the round-off snapping of edge coordinates, weighting, "
        "accumulation into a coadd. No native variant (pure torch on every device). "
        "'compile' is torch.compile(dynamic=False) of the same function; "
        + (
            "grid_sample and the elementwise tail should compile into few graphs. "
            if kernel == "bilinear"
            else "custom_separable_sample has a bool(isfinite(...).all()) host check "
            "(graph break) and _separable_sample_chunked has a Python loop over chunks "
            "that Dynamo unrolls (about 144 chunks x 36 taps at 4096^2), so compile time "
            "grows with size and can dominate the large case (cold compile ~15 s on CPU "
            "already at 256^2, 6 graph breaks; 'compile_fullgraph' raises at the isfinite "
            "check, torch 2.14.1). "
        )
        + "Bilinear 'compile_fullgraph' compiles with 0 breaks. Returns (numer, weight) float32 "
        "maps on the target grid."
    )


NOTES = _notes(KERNEL)


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


def make_inputs_for(
    kernel: str, size_key: str, device: torch.device, dtype: torch.dtype
) -> dict:
    n = SIZES[size_key]
    src_wcs = torchsky.WCS(_header(n, 0.5 * (n + 1), _SRC_CDELT_DEG, 0.0))
    tgt_wcs = torchsky.WCS(
        _header(n, 0.5 * (n + 1) + 0.23, _SRC_CDELT_DEG * SCALE_RATIO, ROT_DEG)
    )
    gen = torch.Generator(device="cpu").manual_seed(_SEEDS[size_key])
    yy, xx = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    image = (
        1.0
        + 0.5
        * torch.sin(2.0 * math.pi * xx / (n / 6.0))
        * torch.cos(2.0 * math.pi * yy / (n / 7.0))
        + 0.05 * torch.randn(n, n, generator=gen, dtype=torch.float64)
    ).to(torch.float32)
    weight = (torch.rand(n, n, generator=gen) * 3.0 + 0.5).to(torch.float32)
    ra, dec = tgt_wcs.pixel_to_world(xx.reshape(-1), yy.reshape(-1))
    x_src, y_src = src_wcs.world_to_pixel(ra, dec)
    # float32-representable coordinates on every device (see NOTES).
    x_src = x_src.reshape(n, n).to(torch.float32).to(torch.float64)
    y_src = y_src.reshape(n, n).to(torch.float32).to(torch.float64)
    return {
        "weighted_image": (image * weight).to(device).contiguous(),
        "weighted_weight": weight.to(device).contiguous(),
        "x_src": x_src.to(dtype).to(device).contiguous(),
        "y_src": y_src.to(dtype).to(device).contiguous(),
        "kernel": kernel,
    }


def make_inputs(size_key: str, device: torch.device, dtype: torch.dtype) -> dict:
    return make_inputs_for(KERNEL, size_key, device, dtype)


def items(size_key: str) -> int:
    """Target pixels sampled."""
    return SIZES[size_key] ** 2


def _sample(
    weighted_image: torch.Tensor,
    weighted_weight: torch.Tensor,
    x_src: torch.Tensor,
    y_src: torch.Tensor,
    kernel: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_src, w_src = (int(v) for v in weighted_image.shape)
    stacked_planes = torch.stack([weighted_image, weighted_weight], dim=0)
    x_norm = (2.0 * x_src / max(w_src - 1, 1)) - 1.0
    y_norm = (2.0 * y_src / max(h_src - 1, 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).to(dtype=torch.float32)
    sampled_planes = sample_image_plane_batched_channels(
        stacked_planes, x_src=x_src, y_src=y_src, grid=grid, kernel=kernel
    )
    sampled_numer = sampled_planes[0]
    sampled_weight = sampled_planes[1]
    if supports_in_bounds_kernel_sum(kernel):
        kernel_sum = in_bounds_kernel_sum(
            x_src, y_src, height=h_src, width=w_src, kernel=kernel
        )
        kernel_sum = kernel_sum.to(dtype=sampled_weight.dtype)
        edge_scale = torch.where(
            kernel_sum > torch.finfo(kernel_sum.dtype).eps,
            1.0 / kernel_sum,
            torch.zeros_like(kernel_sum),
        )
        sampled_numer = sampled_numer * edge_scale
        sampled_weight = sampled_weight * edge_scale
    in_bounds = (
        (x_src >= 0.0)
        & (x_src <= (w_src - 1))
        & (y_src >= 0.0)
        & (y_src <= (h_src - 1))
    )
    zero = torch.zeros_like(sampled_numer)
    sampled_numer = torch.where(in_bounds, sampled_numer, zero)
    sampled_weight = torch.where(in_bounds, sampled_weight, zero)
    sampled_weight = torch.where(sampled_weight > 0.0, sampled_weight, zero)
    sampled_numer = torch.where(sampled_weight > 0.0, sampled_numer, zero)
    return sampled_numer, sampled_weight


def _pure(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return _sample(
        inputs["weighted_image"],
        inputs["weighted_weight"],
        inputs["x_src"],
        inputs["y_src"],
        inputs["kernel"],
    )


def _pure_fullgraph(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    # Distinct code object: Dynamo caches compiled frames per code object, so
    # torch.compile(_pure, fullgraph=True) after torch.compile(_pure) would reuse the
    # graph-broken cache entry instead of raising.
    return _pure(inputs)


def _no_grad(fn: Callable[[dict], Any]) -> Callable[[dict], Any]:
    def run(inputs: dict) -> Any:
        with torch.no_grad():
            return fn(inputs)

    return run


def variants(device: torch.device) -> dict[str, Callable[[dict], Any]]:
    return {
        "eager": _no_grad(_pure),
        "compile": _no_grad(torch.compile(_pure, dynamic=False)),
        "compile_fullgraph": _no_grad(
            torch.compile(_pure_fullgraph, dynamic=False, fullgraph=True)
        ),
    }


# Coordinates are bit-identical across devices (float32-representable, see NOTES), so
# differences come from float32 sampling arithmetic only.
# * Lanczos-3 (torchsky separable sampler): the float32-representable coordinates are
#   used as is; each output is a sum of 36 float32 products with sum|coefficient| <= ~1.5,
#   then scaled by an edge factor computed in DTYPES[device]. Rounding/reordering (FMA vs
#   multiply-add, tap order) is bounded by 64 * eps32 of the image scale ~ 7.6e-6.
# * bilinear (F.grid_sample, align_corners=True): the operator passes a float32
#   normalised grid 2x/(W-1)-1 that grid_sample unnormalises again; the round trip moves
#   the sample position by up to ~2 eps32 * W px, and the normalisation is evaluated in
#   float64 on CPU/CUDA but float32 on MPS and fused differently by inductor. The value
#   moves by that shift times the neighbour difference (<= the image scale), so the bound
#   adds 4 * eps32 * W to the arithmetic term (1.3e-4 at W = 256, 2e-3 at W = 4096).
_REL_TOL = 64.0 * torch.finfo(torch.float32).eps
_NAMES = ("numer", "weight")


def _tolerance(kernel: str, width: int) -> float:
    if kernel == "bilinear":
        return _REL_TOL + 4.0 * torch.finfo(torch.float32).eps * width
    return _REL_TOL


def make_check(kernel: str) -> Callable[[Any, Any], tuple[bool, str]]:
    def check(reference: Any, candidate: Any) -> tuple[bool, str]:
        if len(reference) != len(candidate):
            return False, f"output count {len(reference)} vs {len(candidate)}"
        tol = _tolerance(kernel, int(reference[0].shape[-1]))
        ok = True
        parts = []
        for name, ref_t, cand_t in zip(_NAMES, reference, candidate):
            if ref_t.shape != cand_t.shape:
                return (
                    False,
                    f"{name}: shape {tuple(ref_t.shape)} vs {tuple(cand_t.shape)}",
                )
            # Two-step transfer: MPS .to("cpu", float64) in one call is wrong in torch 2.14.1.
            ref = ref_t.detach().cpu().to(torch.float64)
            cand = cand_t.detach().cpu().to(torch.float64)
            if not bool(torch.isfinite(cand).all()):
                return False, f"{name}: non-finite values in candidate"
            scale = max(1.0, float(ref.abs().max()))
            # A support flip (sample kept on one side, zeroed on the other) is a real
            # difference unless the reference value is itself within rounding of zero.
            flipped = ((ref > 0) != (cand > 0)) & (ref.abs() > tol * scale)
            support = int(flipped.sum())
            err = float((ref - cand).abs().max()) / scale
            ok = ok and err <= tol and support == 0
            parts.append(
                f"{name} max|err|/scale={err:.2e} support mismatches={support}"
            )
        return ok, "; ".join(parts) + f" (tol {tol:.1e}, {kernel})"

    return check


check = make_check(KERNEL)
