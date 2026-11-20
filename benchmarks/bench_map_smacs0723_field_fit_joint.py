#!/usr/bin/env python3
"""Joint multiband real-data field-fit benchmark on JWST SMACS J0723 products."""

from __future__ import annotations
import torchfits

import argparse
import collections
import json
import math
import hashlib
import os
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks"

from torchsky_harness.benchmarks.bench_map_field_fit import (
    _field_fit_kwargs_for_model,
    _field_model_kwargs_for_case,
)  # noqa: E402
from torchsky_harness.benchmarks.bench_map_smacs0723_field_fit import (  # noqa: E402
    _PerExposureScalarBias,
    _copy_header_with_crop,
    _crop_field_inputs_to_target,
    _field_target_observation,
    _training_intensity_scale,
    _scaled_field_observation,
)
from torchsky_harness.benchmarks.bench_map_smacs0723_real import (  # noqa: E402
    DATA,
    DEFAULT_EXPOSURE_DIR,
    _approx_overlap_grid,
    _best_overlap_cutout,
    _center_cutout,
    _cutout_header,
    _discover_exposure_paths,
    _field_observation_kwargs_for_band,
    _filter_exposure_paths_by_patterns,
    _injection_manifest_sign,
    _injection_manifest_tensors,
    _jwst_dq_support_and_contamination_prior,
    _metric_mask,
    _prepare_aux_files,
    _read_ext,
    _read_optional_ext,
    _reference_path_for_band,
    _sci_header,
)
from torchsky_harness.benchmarks.bench_multiband_depth import (  # noqa: E402
    gaussian_psf_contract,
    inject_common_world_exposures,
)
import torchsky_harness.benchmarks.fit_smacs0723_operator_calibration as calibration_fit  # noqa: E402

DEFAULT_JSON_OUT = DATA / "benchmark_results" / "smacs0723_field_fit_joint.json"
DEFAULT_MARKDOWN_OUT = DATA / "benchmark_results" / "smacs0723_field_fit_joint.md"
DEFAULT_EXPORT_DIR = DATA / "benchmark_results" / "smacs0723_field_fit_joint_exports"

import torchsky  # noqa: E402
from torchsky.map.field.core import ExplicitSpectralGridField  # noqa: E402
from torchsky.map.field import (
    render_field_observation,
    render_spectral_power_foam_observation_exact,
)
from torchsky.scene import FieldObservation
from torchsky.field import (
    ForwardContext,
    estimate_domain as estimate_field_domain,
    fit as fit_field_model,
    fit_noise_crossfit as fit_structured_noise_crossfit,
    model as make_field_model,
)
from torchsky.map import coadd as coadd_image_plane
from torchsky.scene import ImageObservation
from torchsky.map.field import (
    IRLSConfig,
    LinearBandGridField,
    adapt_stationary_calibrated_band_observation,
    bind_linear_band_grid_observations_to_operator_set,
    prepare_linear_band_grid_observation_data,
    solve_joint_multiband_grid_cg,
    solve_linear_band_grid_cg,
    solve_linear_band_grid_lsqr,
    solve_power_pixel_affine_least_squares,
    solve_power_pixel_least_squares,
    solve_spectral_cell_coefficients,
    solve_spectral_power_pixel_affine_least_squares,
    spectral_basis_whitening_transform,
)
from torchsky.field import (
    FitResult as FieldFitResult,
    save_artifact as save_field_artifact,
)
from torchsky.map.field.bandpass import bandpass_quadrature  # noqa: E402
from torchsky.map.field.forwarding import type_method_accepts_kwarg  # noqa: E402
from torchsky.instrument import nircam_imaging_band  # noqa: E402
from torchsky.map.operators import (  # noqa: E402
    apply_learned_psf_spec_adjoint,
    apply_psf_spec,
    pixel_area_sr_at,
)
from torchsky.map.operators.psf import gaussian_kernel1d, psf_support_radius  # noqa: E402
from torchsky.calibration_evidence import calibration_exposure_id, load_operator_calibration_split  # noqa: E402
from torchsky.map.operators import (
    ConditionedBackgroundField,
    LearnedBackgroundOp,
    LearnedPSFOp,
    LowRankChromaticPSF,
    AdaptiveSpatioSpectralTemporalEncoder,
    StructuredDetectorNoise,
    StructuredNoiseOp,
    VAR_FLAT,
    VAR_POISSON,
    VAR_RNOISE,
    apply_operator_calibration_pruning,
    field_observation_operator_bundle,
    profile_amplifier_row_offsets,
)
from torchsky.scene import ObservationOperatorBundle
from torchsky.field import PSFSpec
from torchsky.map.product import render_field_product
from torchsky.map import BandpassSpec, ProductResult, ProductSpec
from torchsky.spectrophotometry import Passband  # noqa: E402


REAL_JOINT_CASE = "tpv_field_fit_deep_multiband_broad"
FOAM_FOOTPRINT_EXTENT = 0.55
FOAM_MAX_SMOOTH_PAIRS = 64 * 1024 * 1024
FOAM_TEMPERATURE_START_CELL_UNITS = 1.0


def _contexts_consumed_by_model(
    model: torch.nn.Module,
    contexts: Sequence[ForwardContext | None],
) -> tuple[ForwardContext | None, ...]:
    methods = ["forward"]
    if callable(getattr(model, "render_bandpass", None)):
        methods.append("render_bandpass")
    if all(
        type_method_accepts_kwarg(type(model), method, "forward_context")
        for method in methods
    ):
        return tuple(contexts)
    return (None,) * len(contexts)
FOAM_TEMPERATURE_END_CELL_UNITS = 16.0
SCENE_INITIALIZATION_MODEL_KINDS = frozenset({
    "gaussian_basis",
    "coadd_bilinear",
    "spectral_basis_field",
    "learned_spectral_basis_field",
    "sky_encoder_resmlp",
    "sky_lambda_feature_field",
    "operator_constrained_feature_field",
    "explicit_spectral_grid_field",
    "linear_band_grid_field",
    "sine_implicit",
    "wire_implicit",
    "sasnet",
    "flair_bla",
    "spectral_power_foam",
    "foam_initialized_inr",
    "diffuse_foam_hybrid",
})

INR_PCG_MODEL_KINDS = frozenset({
    "sine_implicit", "wire_implicit", "fourier_mlp", "sky_encoder_mlp",
    "sky_encoder_resmlp", "hashgrid_mlp", "hashgrid_resmlp",
})


def _refine_neural_scene_pcg(
    result: FieldFitResult,
    observations: Sequence[FieldObservation],
    *,
    fit_kwargs: dict[str, Any],
    steps: int,
    cg_iterations: int = 12,
    damping: float = 1.0e-3,
    seed: int = 101,
) -> FieldFitResult:
    """Refine neural parameters against training detector pixels with fixed operators."""
    from torchsky.map.field._gauss_newton import gauss_newton_pcg_step  # white-box: private solver
    from torchsky.map.field.observation_ops import observation_support_mask
    from torchsky.map.field.training import (
        _compute_full_observation_losses,
        compute_field_fit_diagnostics,
    )

    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("neural PCG refinement steps must be a positive integer")
    if (
        fit_kwargs.get("loss_kind", "mse") != "mse"
        or fit_kwargs.get("loss_reduction", "global") != "global"
        or float(fit_kwargs.get("gradient_residual_weight", 0.0)) != 0.0
    ):
        raise ValueError("neural PCG refinement requires global weighted MSE")
    contexts = tuple(fit_kwargs.get("observation_forward_contexts") or [None] * len(observations))
    bundles = result.observation_operator_bundles or tuple(
        field_observation_operator_bundle(observation) for observation in observations
    )
    poses = result.exposure_poses or (None,) * len(observations)
    if not observations or not (len(observations) == len(contexts) == len(bundles) == len(poses)):
        raise ValueError("neural PCG observations, contexts, poses and bundles must align")
    if any(context is not None and context.extra.get("split", "train") != "train" for context in contexts):
        raise ValueError("neural PCG refinement accepts training observations only")
    render_contexts = _contexts_consumed_by_model(result.model, contexts)
    for bundle in bundles:
        if isinstance(bundle.noise, StructuredNoiseOp):
            raise ValueError("neural PCG does not provide structured-noise whitening")
        for _, operator in bundle.iter_slots():
            operator_model = getattr(operator, "model", None)
            if isinstance(operator_model, torch.nn.Module) and any(
                parameter.requires_grad for parameter in operator_model.parameters()
            ):
                raise ValueError("neural PCG requires frozen observation operators")
    render_kwargs = {
        "num_spectral_samples": int(fit_kwargs.get("num_spectral_samples", 9)),
        "spectral_sampling": "deterministic_quadrature",
        "spatial_sampling": fit_kwargs.get("spatial_sampling", "center"),
        "spatial_tile_hw": fit_kwargs.get("render_spatial_tile_hw"),
    }
    render_observations = []
    for observation in observations:
        if observation.detector_to_sky is None:
            # Fixed WCS needs no derivative. torchsky's WCS is pure torch, so it
            # composes with torch.func's wrapped tensors as-is.
            wcs = torchsky.WCS(observation.header)
            render_observations.append(replace(
                observation, detector_to_sky=wcs,
                detector_x0_pix=0.0, detector_y0_pix=0.0,
            ))
        else:
            render_observations.append(observation)

    class DetectorScene(torch.nn.Module):
        def __init__(self, scene: torch.nn.Module) -> None:
            super().__init__()
            self.scene = scene

        def forward(self, observation_index: int) -> torch.Tensor:
            return render_field_observation(
                self.scene, render_observations[observation_index],
                forward_context=render_contexts[observation_index],
                exposure_pose=poses[observation_index],
                operator_bundle=bundles[observation_index],
                **render_kwargs,
            )

    wrapper = DetectorScene(result.model).eval()
    scene_parameters = [
        (name, parameter) for name, parameter in result.model.named_parameters()
        if parameter.requires_grad and not (
            isinstance(result.model, _PerExposureScalarBias) and name == "bias"
        )
    ]
    if not scene_parameters:
        raise ValueError("neural PCG requires trainable scene parameters")
    state = torch.cat([parameter.detach().reshape(-1) for _, parameter in scene_parameters])
    lengths = [parameter.numel() for _, parameter in scene_parameters]
    supports = [observation_support_mask(observation) for observation in observations]
    weights = [
        (observation.weight if observation.weight is not None else torch.ones_like(observation.image))[support].detach().double()
        for observation, support in zip(observations, supports, strict=True)
    ]
    total_weight = sum(float(weight.sum()) for weight in weights)
    if not math.isfinite(total_weight) or total_weight <= 0 or any(weight.numel() == 0 for weight in weights):
        raise ValueError("neural PCG requires finite positive weight on each exposure's support")

    def residual_function(flat: torch.Tensor, observation_index: int) -> torch.Tensor:
        parameter_map = {
            f"scene.{name}": value.reshape_as(parameter)
            for (name, parameter), value in zip(
                scene_parameters, flat.split(lengths), strict=True
            )
        }
        prediction = torch.func.functional_call(wrapper, parameter_map, (observation_index,))
        support = supports[observation_index]
        target = observations[observation_index].image[support].to(device=prediction.device)
        whitening = (weights[observation_index] / total_weight).sqrt().to(
            device=prediction.device, dtype=prediction.dtype
        )
        return (prediction[support.to(device=prediction.device)] - target) * whitening

    # Warm fixed WCS caches outside functional transforms; no heldout images enter here.
    with torch.no_grad():
        for index in range(len(observations)):
            wrapper(index)
    residual_functions = [
        lambda flat, index=index: residual_function(flat, index)
        for index in range(len(observations))
    ]
    start = time.perf_counter()
    reports = []
    for step in range(steps):
        state, report = gauss_newton_pcg_step(
            residual_functions, state, cg_iterations=cg_iterations,
            damping=damping, seed=seed + step,
        )
        reports.append(report)
        print(f"Neural PCG: {step + 1}/{steps}, loss={report['final_loss']:.6g}, scale={report['accepted_step_scale']}", flush=True)
        if report["accepted_step_scale"] == 0.0:
            break
    with torch.no_grad():
        for (_, parameter), value in zip(scene_parameters, state.split(lengths), strict=True):
            parameter.copy_(value.reshape_as(parameter))
        final_loss = _compute_full_observation_losses(
            observations, result.model, render_contexts, poses, bundles,
            num_spectral_samples=render_kwargs["num_spectral_samples"],
            loss_kind="mse", charbonnier_eps=float(fit_kwargs.get("charbonnier_eps", 1.0e-3)),
            huber_delta=float(fit_kwargs.get("huber_delta", 1.0)), loss_reduction="global",
            spatial_sampling=render_kwargs["spatial_sampling"],
            spatial_tile_hw=fit_kwargs.get("diagnostic_spatial_tile_hw"),
        )
    diagnostics = dict(result.diagnostics or {})
    previous_duration = float(diagnostics.get("performance_profile", {}).get("total_duration_sec", 0.0))
    max_mem = int(diagnostics.get("performance_profile", {}).get("max_memory_allocated_bytes", 0))
    if state.device.type == "cuda":
        torch.cuda.synchronize(state.device)
        max_mem = max(max_mem, torch.cuda.max_memory_allocated(state.device))
    diagnostics.update(compute_field_fit_diagnostics(
        result.model, observations, contexts, poses, bundles,
        num_spectral_samples=render_kwargs["num_spectral_samples"],
        fit_duration_sec=previous_duration + time.perf_counter() - start,
        steps=result.completed_steps, total_pixels=sum(int(support.sum()) for support in supports),
        max_mem=max_mem,
        spatial_sampling=render_kwargs["spatial_sampling"],
        spatial_tile_hw=fit_kwargs.get("diagnostic_spatial_tile_hw"),
    ))
    diagnostics["inr_pcg_refinement"] = {
        "objective": "training_global_weighted_mse", "steps": reports,
        "observations": len(observations), "pixels": sum(int(support.sum()) for support in supports),
        "initial_loss": reports[0]["initial_loss"], "final_loss": final_loss,
        "adam_final_loss": result.final_loss,
        "optimizer_status": "discarded after post-Adam parameter refinement",
    }
    return replace(result, final_loss=final_loss, diagnostics=diagnostics,
                   optimizer_state=None, amp_scaler_state=None)


def _save_scene_initialization_product(
    path: Path,
    product: ProductResult,
    *,
    elapsed_ms: float,
    pixel_response_convention: str,
    intensity_scale: float,
) -> dict[str, Any]:
    if not math.isfinite(float(intensity_scale)) or float(intensity_scale) <= 0.0:
        raise ValueError("scene initialization intensity scale must be positive")
    path.mkdir(parents=True, exist_ok=False)
    if product.weight is None or product.mask is None:
        raise ValueError("scene initialization persistence requires weight and support")
    provenance = dict(product.provenance or {})
    value_unit = str(provenance.get("value_unit") or "unknown")
    data_header = dict(product.header)
    data_header["BUNIT"] = value_unit
    data_header["TSKYSCL"] = float(intensity_scale)
    weight_header = dict(product.header)
    weight_header["BUNIT"] = f"1/({value_unit})^2"
    weight_header["TSKYSCL"] = float(intensity_scale)
    support_header = dict(product.header)
    support_header["BUNIT"] = "1"
    support_header["TSKYSCL"] = float(intensity_scale)
    outputs = {
        "data": (
            path / "data.fits",
            product.data * float(intensity_scale),
            data_header,
        ),
        "weight": (
            path / "weight.fits",
            product.weight / float(intensity_scale) ** 2,
            weight_header,
        ),
        "support": (
            path / "support.fits",
            product.mask.to(dtype=torch.float32),
            support_header,
        ),
    }
    files = {}
    for name, (output_path, tensor, header) in outputs.items():
        torchfits.write(
            str(output_path), tensor.detach().cpu(), header=header,
            overwrite=False,
        )
        files[name] = {
            "path": str(output_path.resolve()),
            "size": output_path.stat().st_size,
            "sha256": calibration_fit._sha256(output_path),
        }
    bands = []
    for band in product.product.bands:
        metadata = {
            "label": band.label,
            "transmission_center": band.transmission_center,
            "spectral_unit": band.spectral_unit,
        }
        if band.spectral_min is not None:
            metadata["spectral_min"] = band.spectral_min
            metadata["spectral_max"] = band.spectral_max
        if band.transmission_sigma is not None:
            metadata["transmission_sigma"] = band.transmission_sigma
        if band.passband is not None:
            metadata["passband_wavelength"] = band.passband.wavelength.tolist()
            metadata["passband_throughput"] = band.passband.throughput.tolist()
        bands.append(metadata)
    metadata = {
        "policy": "common_product",
        "elapsed_ms": float(elapsed_ms),
        "value_unit": provenance.get("value_unit"),
        "value_semantics": product.product.value_semantics,
        "stored_value_space": "input_surface_brightness",
        "intensity_scale": float(intensity_scale),
        "pixel_response_convention": str(pixel_response_convention),
        "training_exposure_ids": list(provenance.get("training_exposure_ids", [])),
        "initializer_model": provenance.get("initializer_model"),
        "initializer_steps": provenance.get("initializer_steps"),
        "initializer_lr": provenance.get("initializer_lr"),
        "initializer_initial_loss": provenance.get("initializer_initial_loss"),
        "initializer_final_loss": provenance.get("initializer_final_loss"),
        "target_header": dict(product.header),
        "shape": list(product.data.shape),
        "bands": bands,
        "files": files,
    }
    (path / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _save_joint_field_artifact(
    path: Path, *, result: Any, model_kind: str, model_kwargs: dict[str, Any],
    fit_kwargs: dict[str, Any], run_config: dict[str, Any],
    contexts: Sequence[ForwardContext], input_identities: list[dict[str, Any]],
    calibration_audit: dict[str, Any] | None, intensity_scale: float,
    normalization_training_count: int, unfrozen_extras: bool = False,
    exact_refined: bool = False,
    scene_initialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the scene and explicitly describe external replay requirements."""
    if isinstance(result.model, _PerExposureScalarBias):
        return {
            "status": "unsupported", "reason": "per-exposure bias wrapper is not registered",
            "detector_replay_available": False, "training_resume_available": False,
        }
    # Operator instances remain external; their frozen checkpoint identities are saved.
    scalar_fit_config = {
        key: value for key, value in fit_kwargs.items()
        if key not in {"observation_forward_contexts", "observation_operator_bundles"}
    }
    replay = {
        "detector_replay_available": not unfrozen_extras,
        "training_resume_available": not unfrozen_extras and not exact_refined,
        "requirements": (
            "scene only: PSF/noise extras have no saved checkpoint references"
            if unfrozen_extras else
            "rebuild observations and operators from verified inputs and saved run configuration"
        ),
        "optimizer_status": (
            "discarded: neural PCG changed parameters after optimizer steps"
            if (result.diagnostics or {}).get("inr_pcg_refinement") else
            "discarded: exact refinement changed parameters after optimizer steps"
            if exact_refined else "saved from field fit"
        ),
    }
    if model_kind == "linear_band_grid_field":
        path.mkdir(parents=True, exist_ok=True)
        weights_path = path / "weights.pt"
        state_cpu = {
            k: v.detach().cpu() for k, v in result.model.state_dict().items()
        }
        torch.save({"state_dict": state_cpu}, weights_path)
        manifest_payload = {
            "schema": "torchsky.linear_band_grid_artifact.v1",
            "model_kind": model_kind,
            "model_kwargs": _report_configuration(model_kwargs),
            "domain": asdict(result.model.domain),
            "fit_config": {
                "fit_kwargs": _report_configuration(scalar_fit_config),
                "run_config": _report_configuration(run_config),
                "scene_initialization": _report_configuration(scene_initialization),
                "normalization": {
                    "intensity_scale": float(intensity_scale),
                    "training_exposures": int(normalization_training_count),
                    "operation": "SCI/scale; ERR/scale; VAR/scale**2; no additive subtraction",
                    "prediction_to_original_units": "multiply by intensity_scale",
                },
            },
            "calibration_manifest": _report_configuration(calibration_audit),
            "input_identities": input_identities,
        }
        (path / "manifest.json").write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "saved",
            "path": str(path.resolve()),
            "detector_replay_available": not unfrozen_extras,
            "training_resume_available": False,
            "requirements": replay["requirements"],
            "optimizer_status": "linear_solver_closed_form",
        }
    save_field_artifact(
        path, model=result.model, model_kind=model_kind, model_kwargs=model_kwargs,
        fit_result=(replace(result, optimizer_state=None, amp_scaler_state=None)
                    if exact_refined else result),
        fit_config={
            "fit_kwargs": scalar_fit_config, "run_config": run_config,
            "scene_initialization": scene_initialization,
            **({"inr_pcg_refinement": result.diagnostics["inr_pcg_refinement"]}
               if result.diagnostics and "inr_pcg_refinement" in result.diagnostics else {}),
            "normalization": {
                "intensity_scale": float(intensity_scale),
                "training_exposures": int(normalization_training_count),
                "operation": "SCI/scale; ERR/scale; VAR/scale**2; no additive subtraction",
                "prediction_to_original_units": "multiply by intensity_scale",
            },
            "replay": replay,
        },
        calibration_manifest=calibration_audit,
        operator_manifest={
            "contexts": [asdict(context) if context is not None else None for context in contexts],
            "frozen_calibration": calibration_audit,
            "live_operator_objects_serialized": False,
        },
        input_identities=input_identities,
    )
    return {"status": "saved", "path": str(path.resolve()), **replay}


def _fitted_gaussian_psf_sigma_pix(
    band: str,
    header: dict[str, Any],
) -> float:
    """Convert the frozen fitted PSF FWHM to the local image pixel scale."""

    width = int(header.get("NAXIS1", 0))
    height = int(header.get("NAXIS2", 0))
    if width <= 0 or height <= 0:
        raise ValueError("PSF conversion requires positive NAXIS1/NAXIS2")
    wcs = torchsky.WCS(header)
    area_sr = pixel_area_sr_at(
        wcs,
        torch.tensor(0.5 * (width - 1), dtype=torch.float64),
        torch.tensor(0.5 * (height - 1), dtype=torch.float64),
    )
    pixel_scale_arcsec = float(torch.sqrt(area_sr) * (180.0 / torch.pi) * 3600.0)
    contract = gaussian_psf_contract(band)
    sigma_pix = contract.fitted_fwhm_arcsec / 2.354820045 / pixel_scale_arcsec
    if not math.isfinite(sigma_pix) or sigma_pix <= 0.0:
        raise ValueError("fitted PSF has a non-finite image-pixel width")
    return sigma_pix


def _parse_csv(value: str) -> list[str]:
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def _safe_slug(value: str) -> str:
    text = str(value).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "item"


def _calibration_exposure_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for exposure in payload["exposures"]:
        lookup[str(exposure["path"])] = exposure
        lookup[str(exposure["resolved_path"])] = exposure
        lookup[str(exposure["filename"])] = exposure
    return lookup


def _calibration_exposure(
    lookup: dict[str, dict[str, Any]], path: Path
) -> dict[str, Any]:
    exposure = lookup.get(str(path)) or lookup.get(str(path.resolve()))
    exposure = exposure or lookup.get(path.name)
    if exposure is None:
        raise ValueError(f"calibration manifest does not contain {path}")
    return exposure


def _training_calibration_exposure_paths(
    paths: list[Path], lookup: dict[str, dict[str, Any]]
) -> tuple[list[Path], dict[str, int]]:
    training = []
    omitted = {"validation": 0, "test": 0}
    for path in paths:
        split = _calibration_exposure(lookup, path).get("split")
        if split == "train":
            training.append(path)
        elif split in omitted:
            omitted[split] += 1
        else:
            raise ValueError(f"invalid calibration split {split!r} for {path}")
    if paths and not training:
        raise ValueError("calibration selection contains no training exposures")
    return training, omitted


def _exclude_calibration_samples(
    mask: np.ndarray, rows: list[dict[str, Any]], *,
    x0: float, y0: float, psf_radius: int,
) -> np.ndarray:
    """Exclude full-detector sample footprints and their PSF influence region."""
    result = mask.copy()
    if psf_radius < 0:
        raise ValueError("PSF exclusion radius must be nonnegative")
    height, width = result.shape
    for row in rows:
        bounds = row.get("calibration_bounds_xy")
        if bounds is None or len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
            raise ValueError("calibration evidence requires finite footprint bounds")
        if bounds[0] > bounds[1] or bounds[2] > bounds[3]:
            raise ValueError("calibration footprint bounds are reversed")
        left = max(0, math.floor(bounds[0] - psf_radius - x0))
        right = min(width, math.ceil(bounds[1] + psf_radius - x0) + 1)
        bottom = max(0, math.floor(bounds[2] - psf_radius - y0))
        top = min(height, math.ceil(bounds[3] + psf_radius - y0) + 1)
        if left < right and bottom < top:
            result[bottom:top, left:right] = 0
    return result


class _ZeroScene(torch.nn.Module):
    """Zero celestial brightness for measuring the fixed detector contribution."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), device=device), requires_grad=False)

    def forward(
        self,
        ra: torch.Tensor,
        dec: torch.Tensor,
        wavelength: Any,
        *,
        forward_context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del dec, wavelength, forward_context
        return torch.zeros_like(ra) + self.anchor


def _export_native_training_view(
    output_dir: Path,
    observations: Sequence[FieldObservation],
    contexts: Sequence[ForwardContext],
    bundles: Sequence[ObservationOperatorBundle],
    *,
    intensity_scale: float,
    fit_kwargs: dict[str, Any],
    calibration_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Export the fixed-background-corrected detector inputs for classical parity."""
    from torchsky.map.operators.noise import ExplicitNoiseOp

    scale = float(intensity_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("native training view requires a finite positive intensity scale")
    if not observations or not (len(observations) == len(contexts) == len(bundles)):
        raise ValueError("native training view observations, contexts and bundles must align")
    if any(context.extra.get("split") != "train" for context in contexts):
        raise ValueError("native training view accepts only TRAIN contexts")
    for bundle in bundles:
        if bundle.noise is not None and not isinstance(bundle.noise, ExplicitNoiseOp):
            raise ValueError("native training view requires diagonal pipeline ERR weighting")
        for _, operator in bundle.iter_slots():
            model = getattr(operator, "model", None)
            if isinstance(model, torch.nn.Module) and any(
                parameter.requires_grad for parameter in model.parameters()
            ):
                raise ValueError("native training view requires frozen calibrated operators")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        # ponytail: render/write one exposure at a time; PSF-halo tile blocks
        # are the upgrade if a single detector crop exceeds the memory budget.
        for index, (observation, context, bundle) in enumerate(
            zip(observations, contexts, bundles, strict=True)
        ):
            if observation.err is None or observation.mask is None or observation.weight is None:
                raise ValueError("native training view requires ERR, mask and likelihood weights")
            image = observation.image.detach().cpu()
            err = observation.err.detach().cpu()
            weight = observation.weight.detach().cpu()
            valid = (observation.mask.detach().cpu() > 0) & torch.isfinite(image)
            valid &= torch.isfinite(err) & (err > 0) & torch.isfinite(weight) & (weight > 0)
            if not bool(valid.any()) or not torch.allclose(
                (weight[valid].double() * err[valid].double().square()),
                torch.ones_like(weight[valid], dtype=torch.float64), rtol=1.e-5, atol=1.e-7,
            ):
                raise ValueError("native training view requires nonempty support and ERR^-2 weights")
            if str(observation.header.get("BUNIT", "")).strip() != "MJy/sr":
                raise ValueError("native training view requires MJy/sr detector SCI")
            fixed = render_field_observation(
                _ZeroScene(observation.image.device).eval(), observation,
                num_spectral_samples=int(fit_kwargs.get("num_spectral_samples", 9)),
                spectral_sampling="deterministic_quadrature",
                spatial_sampling=fit_kwargs.get("spatial_sampling", "center"),
                spatial_tile_hw=fit_kwargs.get("render_spatial_tile_hw", (256, 256)),
                forward_context=context, operator_bundle=bundle,
            ).detach().cpu()
            if fixed.shape != image.shape or not bool(torch.isfinite(fixed[valid]).all()):
                raise ValueError("native training view has invalid fixed detector prediction")
            arrays = {
                "sci": torch.where(valid, (image - fixed) * scale, 0.),
                "wht": torch.where(valid, weight / (scale * scale), 0.),
                "mask": valid.to(image.dtype),
            }
            paths: dict[str, str] = {}
            hashes: dict[str, str] = {}
            for kind, array in arrays.items():
                path = output_dir / f"{index:03d}_{kind}.fits"
                header = dict(observation.header)
                header["BUNIT"] = {"sci": "MJy/sr", "wht": "(MJy/sr)^-2", "mask": "1"}[kind]
                torchfits.write(str(path), array, header=header, overwrite=False)
                paths[kind] = str(path)
                hashes[kind] = calibration_fit._sha256(path)
            source_path = Path(str(context.extra["source_path"]))
            rows.append({
                "band": str(context.survey), "detector": str(context.detector),
                "exposure_id": str(context.exposure_id),
                "calibration_exposure_id": context.extra.get("calibration_exposure_id"),
                "split": "train", "source_path": str(source_path.resolve()),
                "source_sha256": calibration_fit._sha256(source_path),
                "valid_pixels": int(valid.sum()), "shape": list(image.shape),
                "detector_xy0": [observation.detector_x0_pix, observation.detector_y0_pix],
                "image_observation": {
                    "path": paths["sci"], "hdu": 0,
                    "weight_path": paths["wht"], "weight_hdu": 0,
                    "mask_path": paths["mask"], "mask_hdu": 0,
                    "weight_is_inverse_variance": True,
                },
                "file_sha256": hashes,
            })
    manifest_path = output_dir / "manifest.json"
    report = {
        "schema_version": 1, "profile": "smacs-native-training-view",
        "depth_claim": False, "resampled": False, "intensity_scale": scale,
        "science": "(normalized SCI - frozen zero-scene detector prediction) * intensity_scale",
        "weight": "normalized likelihood weight / intensity_scale^2; pipeline ERR^-2",
        "support": "exact positive finite neural likelihood support; calibration footprints excluded",
        "contamination": "isolated JUMP_DET retained at pipeline ERR weight; no learned noise or reweighting",
        "calibration": calibration_metadata, "render_settings": {
            key: fit_kwargs[key] for key in (
                "num_spectral_samples", "spatial_sampling", "render_spatial_tile_hw"
            ) if key in fit_kwargs
        },
        "rows": rows, "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _training_scene_initialization_product(
    observations: Sequence[FieldObservation],
    contexts: Sequence[ForwardContext],
    bundles: Sequence[ObservationOperatorBundle],
    *,
    target_header: dict[str, Any],
    fit_kwargs: dict[str, Any],
) -> ProductResult:
    """Coadd normalized training SCI minus its fixed additive detector signal."""
    if not observations or not (len(observations) == len(contexts) == len(bundles)):
        raise ValueError("scene initialization requires aligned training observations")
    if any(context.extra.get("split") != "train" for context in contexts):
        raise ValueError("scene initialization accepts only training exposures")
    value_units = {observation.value_unit for observation in observations}
    value_semantics = {observation.value_semantics for observation in observations}
    spectral_units = {str(observation.spectral_unit).lower() for observation in observations}
    if len(value_units) != 1 or None in value_units or value_semantics != {"surface_brightness"}:
        raise ValueError(
            "scene initialization requires the same calibrated surface-brightness unit"
        )
    if spectral_units not in ({"angstrom"}, {"angstroms"}, {"aa"}):
        raise ValueError("scene initialization requires one Angstrom spectral unit")
    records = []
    identities = set()
    for observation, context, bundle in zip(observations, contexts, bundles, strict=True):
        identity = context.exposure_id or context.extra.get("source_path")
        if not identity or identity in identities:
            raise ValueError("scene initialization requires unique training exposure identities")
        identities.add(identity)
        records.append((str(identity), observation, context, bundle))
    records.sort(key=lambda item: item[0])
    height, width = int(target_header["NAXIS2"]), int(target_header["NAXIS1"])
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float64),
                            torch.arange(width, dtype=torch.float64), indexing="ij")
    ra, dec = torchsky.WCS(target_header).pixel_to_world(xx, yy)
    zero = _ZeroScene(torch.device("cpu"))
    accumulated: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for _, observation, context, bundle in records:
            if observation.wavelength is None or observation.weight is None or observation.mask is None:
                raise ValueError("scene initialization requires wavelength, weight and mask")
            fixed = render_field_observation(
                zero, observation, forward_context=context, operator_bundle=bundle,
                num_spectral_samples=int(fit_kwargs.get("num_spectral_samples", 9)),
                spectral_sampling="deterministic_quadrature",
                spatial_sampling=fit_kwargs.get("spatial_sampling", "center"),
                spatial_tile_hw=fit_kwargs.get("render_spatial_tile_hw", (256, 256)),
            ).detach().cpu()
            residual = observation.image.detach().cpu() - fixed
            weight = observation.weight.detach().cpu()
            valid = (observation.mask.detach().cpu() > 0) & torch.isfinite(residual)
            valid &= torch.isfinite(weight) & (weight > 0)
            weight = torch.where(valid, weight, 0.0)
            residual = torch.where(valid, residual, 0.0)
            px, py = torchsky.WCS(observation.header).world_to_pixel(ra, dec)
            source_h, source_w = residual.shape
            grid = torch.stack((2 * (px + .5) / source_w - 1,
                                2 * (py + .5) / source_h - 1), dim=-1).float()[None]
            # ponytail: bilinear coadds are a PSF-blurred warm start; the detector
            # objective, rather than this interpolation, performs reconstruction.
            projected = torch.nn.functional.grid_sample(
                torch.stack((residual * weight, weight)).float()[None], grid,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )[0].double()
            wavelength = float(observation.wavelength)
            label = str(observation.band_name or "").strip()
            if not label:
                raise ValueError("scene initialization requires band names")
            numer, denom = accumulated.setdefault(
                wavelength,
                (torch.zeros_like(projected[0]), torch.zeros_like(projected[1])),
            )
            numer.add_(projected[0])
            denom.add_(projected[1])
    centers = sorted(accumulated)
    weights = torch.stack([accumulated[center][1] for center in centers])
    data = torch.stack([accumulated[center][0] for center in centers])
    data = torch.where(weights > 0, data / weights.clamp_min(torch.finfo(data.dtype).tiny), 0.)
    if not bool((weights > 0).any()):
        raise ValueError("scene initialization has no supported training pixels")
    band_specs = {}
    for _, observation, _, _ in records:
        center = float(observation.wavelength)
        label = str(observation.band_name)
        if center in band_specs and band_specs[center].label != label:
            raise ValueError("scene initialization band labels must match by wavelength")
        if center in band_specs:
            continue
        if observation.passband is not None:
            band_specs[center] = BandpassSpec(
                label=label,
                spectral_min=observation.spectral_min,
                spectral_max=observation.spectral_max,
                passband=observation.passband,
            )
        elif (
            observation.transmission_wavelength is not None
            and observation.transmission_throughput is not None
        ):
            band_specs[center] = BandpassSpec(
                label=label,
                spectral_min=observation.spectral_min,
                spectral_max=observation.spectral_max,
                passband=Passband.from_samples(
                    observation.transmission_wavelength,
                    observation.transmission_throughput,
                    name=label,
                    wavelength_unit="angstrom",
                ),
            )
        else:
            band_specs[center] = BandpassSpec(
                label=label,
                spectral_min=observation.spectral_min,
                spectral_max=observation.spectral_max,
                transmission_center=center,
                transmission_sigma=observation.transmission_sigma,
            )
    output_dtype = observations[0].image.dtype
    product = ProductSpec(
        target_header=dict(target_header),
        bands=tuple(band_specs[center] for center in centers),
    )
    return ProductResult(
        product=product,
        data=data.to(dtype=output_dtype),
        weight=weights.to(dtype=output_dtype),
        header=dict(target_header),
        mask=weights > 0,
        provenance={
            "value_unit": next(iter(value_units)),
            "value_semantics": "surface_brightness",
            "spectral_unit": next(iter(spectral_units)),
            "training_exposure_ids": [item[0] for item in records],
        },
    )


def _initialize_joint_scene(
    model: torch.nn.Module,
    model_kind: str,
    product: ProductResult,
    *,
    policy: str = "common_product",
) -> dict[str, Any]:
    """Seed the grid image or a positive constant INR from the same sky data."""
    if policy == "none":
        return {"scene_initialization_policy": "none"}
    if policy != "common_product":
        raise ValueError(f"unsupported scene initialization policy {policy!r}")
    assert product.weight is not None
    valid = product.weight > 0
    mean = float(product.data[valid].mean())
    if not math.isfinite(mean):
        raise ValueError("scene initialization mean must be finite")

    def initialize_positive_output(
        output: torch.nn.Linear,
        *,
        target: float,
        floor: float,
    ) -> None:
        epsilon = torch.finfo(output.weight.dtype).eps
        matched_target = max(target, floor + epsilon)
        value = matched_target - floor
        with torch.no_grad():
            output.weight.zero_()
            output.bias.fill_(value + math.log(-math.expm1(-value)))

    def initialize_foam(foam_model: torch.nn.Module, target: float) -> None:
        coefficients = getattr(foam_model, "cell_coefficients", None)
        if not isinstance(coefficients, torch.nn.Parameter):
            raise TypeError("Foam scene initialization requires cell coefficients")
        value = max(target, torch.finfo(coefficients.dtype).tiny)
        log_intensity = bool(getattr(getattr(foam_model, "foam", None),
                                     "log_intensity_spectra", False))
        with torch.no_grad():
            coefficients.zero_()
            coefficients[:, 0].fill_(math.log(value) if log_intensity else value)
            gradients = getattr(foam_model, "cell_gradient_coefficients", None)
            if isinstance(gradients, torch.nn.Parameter):
                gradients.zero_()

    def match_center_output(
        field_model: torch.nn.Module,
        output: torch.nn.Linear,
        *,
        target: float,
        floor: float,
    ) -> None:
        epsilon = torch.finfo(output.weight.dtype).eps
        matched_target = max(target, floor + epsilon)
        value = matched_target - floor
        device = output.weight.device
        dtype = output.weight.dtype
        domain = field_model.domain
        ra = torch.tensor([domain.center_ra_deg], device=device, dtype=dtype)
        dec = torch.tensor([domain.center_dec_deg], device=device, dtype=dtype)
        wavelength = torch.tensor(
            [domain.wavelength_center], device=device, dtype=dtype
        )
        with torch.no_grad():
            output.bias.fill_(value + math.log(-math.expm1(-value)))
            for _ in range(8):
                prediction = field_model(ra, dec, wavelength)
                positive = (prediction - floor).clamp_min(
                    torch.finfo(dtype).eps
                )
                derivative = -torch.expm1(-positive)
                correction = (prediction - matched_target) / derivative
                output.bias.sub_(correction.mean())

    if model_kind in {
        "explicit_spectral_grid_field",
        "linear_band_grid_field",
        "gaussian_basis",
    }:
        model.initialize_from_product(product)
    elif model_kind == "spectral_power_foam":
        initialize_foam(model, mean)
    elif model_kind == "foam_initialized_inr":
        pass
    elif model_kind == "diffuse_foam_hybrid":
        diffuse = model.diffuse_field
        diffuse_floor = float(getattr(diffuse, "intensity_floor", 1.0e-4))
        initialize_positive_output(
            diffuse.output,
            target=diffuse_floor,
            floor=diffuse_floor,
        )
        initialize_foam(model.localized_foam, mean - diffuse_floor)
    elif model_kind in {"spectral_basis_field", "learned_spectral_basis_field"}:
        floor = 1.0e-4
        initialize_positive_output(model.bias_head, target=mean, floor=floor)
        with torch.no_grad():
            model.coeff_head.weight.zero_()
            model.coeff_head.bias.zero_()
            if model_kind == "spectral_basis_field":
                model.spectral_amp_head[-1].weight.zero_()
                model.spectral_amp_head[-1].bias.zero_()
    elif model_kind in {"sky_lambda_feature_field", "operator_constrained_feature_field"}:
        initialize_positive_output(model.base_output, target=mean, floor=1.0e-4)
        with torch.no_grad():
            model.refine[-1].weight.zero_()
            model.refine[-1].bias.zero_()
            if model_kind == "operator_constrained_feature_field":
                model.residual_decoder[-1].weight.zero_()
                model.residual_decoder[-1].bias.zero_()
    else:
        output = model.readout[-1] if model_kind == "sasnet" else model.output
        if isinstance(output, torch.nn.Sequential):
            output = output[-1]
        if not isinstance(output, torch.nn.Linear):
            raise TypeError(f"{model_kind} has no supported initialization readout")
        floor = float(getattr(model, "intensity_floor", 1.e-4))
        match_center_output(model, output, target=mean, floor=floor)
    return {
        "scene_initialization_policy": "common_product",
        "scene_initialization_training_mean": mean,
        "scene_initialization_supported_pixels": float(valid.sum()),
    }


def _fit_training_scene_initialization_product(
    observations: Sequence[FieldObservation],
    contexts: Sequence[ForwardContext],
    bundles: Sequence[ObservationOperatorBundle],
    *,
    target_header: dict[str, Any],
    fit_kwargs: dict[str, Any],
    steps: int,
    lr: float,
) -> ProductResult:
    if int(steps) <= 0:
        raise ValueError("scene initialization steps must be positive")
    if not math.isfinite(float(lr)) or float(lr) <= 0.0:
        raise ValueError("scene initialization learning rate must be positive")
    records = sorted(
        zip(observations, contexts, bundles, strict=True),
        key=lambda item: str(
            item[1].exposure_id or item[1].extra.get("source_path") or ""
        ),
    )
    ordered_observations = tuple(item[0] for item in records)
    ordered_contexts = tuple(item[1] for item in records)
    ordered_bundles = tuple(item[2] for item in records)
    seed = _training_scene_initialization_product(
        ordered_observations,
        ordered_contexts,
        ordered_bundles,
        target_header=target_header,
        fit_kwargs=fit_kwargs,
    )
    height, width = seed.product.shape
    domain = estimate_field_domain(ordered_observations, coord_mode="local_tangent")
    model = make_field_model(
        "explicit_spectral_grid_field",
        domain,
        spatial_resolution=max(height, width),
        spectral_resolution=max(seed.n_bands, 2),
    )
    model.initialize_from_product(seed)
    model_contexts = _contexts_consumed_by_model(model, ordered_contexts)
    parameter_states = []
    seen_parameters = set()
    for bundle in ordered_bundles:
        for operator in (bundle.psf, bundle.background, bundle.noise):
            operator_model = getattr(operator, "model", None)
            if not isinstance(operator_model, torch.nn.Module):
                continue
            for parameter in operator_model.parameters():
                if id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                parameter_states.append((parameter, parameter.requires_grad))
                parameter.requires_grad_(False)
    try:
        result = fit_field_model(
            ordered_observations,
            model,
            steps=int(steps),
            lr=float(lr),
            num_spectral_samples=int(fit_kwargs.get("num_spectral_samples", 9)),
            loss_kind="mse",
            loss_reduction="global",
            training_spectral_sampling="deterministic_quadrature",
            spatial_sampling=fit_kwargs.get("spatial_sampling", "center"),
            observations_per_step=len(ordered_observations),
            training_patch_size=None,
            pixel_samples_per_observation=None,
            observation_forward_contexts=model_contexts,
            observation_operator_bundles=ordered_bundles,
            render_spatial_tile_hw=fit_kwargs.get("render_spatial_tile_hw"),
        )
    finally:
        for parameter, requires_grad in parameter_states:
            parameter.requires_grad_(requires_grad)
    latent = render_field_product(
        result.model,
        seed.product,
        num_spectral_samples=int(fit_kwargs.get("num_spectral_samples", 9)),
        spatial_sampling="center",
        spatial_tile_hw=fit_kwargs.get("render_spatial_tile_hw"),
    )
    latent_data = latent.data.detach().cpu()
    if seed.mask is not None:
        latent_data = torch.where(seed.mask, latent_data, torch.zeros_like(latent_data))
    return replace(
        latent,
        data=latent_data,
        weight=seed.weight,
        mask=seed.mask,
        provenance={
            **dict(seed.provenance or {}),
            "initializer_model": "explicit_spectral_grid_field",
            "initializer_steps": int(steps),
            "initializer_lr": float(lr),
            "initializer_initial_loss": float(result.initial_loss),
            "initializer_final_loss": float(result.final_loss),
        },
    )


def _foam_initialization_entries(
    product: ProductResult,
    band_entries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if product.mask is None:
        raise ValueError("Foam scene initialization requires product support")
    by_band = {str(entry["band"]): entry for entry in band_entries}
    if len(by_band) != len(band_entries):
        raise ValueError("Foam scene initialization requires unique band entries")
    entries = []
    for band_index, band in enumerate(product.product.bands):
        label = str(band.label or "")
        if label not in by_band:
            raise ValueError(f"Foam scene initialization has no target for {label!r}")
        source = by_band[label]
        entries.append({
            **source,
            "target_header": dict(product.header),
            "target_observation": replace(
                source["target_observation"],
                header=dict(product.header),
                psf_sigma_pix=0.0,
                psf_spec=PSFSpec(),
            ),
            "coadd_pred": product.data[band_index].detach().cpu().numpy(),
            "support_mask": product.mask[band_index].detach().cpu().numpy(),
        })
    return entries


class _MultiBandLinearGridScene(torch.nn.Module):
    """Per-band signed linear grid wrapper for multi-band SMACS evaluation."""

    def __init__(
        self,
        domain: Any,
        *,
        band_entries: Sequence[dict[str, Any]],
        spatial_resolution: int,
        spatial_bounds_local_deg: tuple[float, float, float, float],
    ) -> None:
        super().__init__()
        if not band_entries:
            raise ValueError("_MultiBandLinearGridScene requires non-empty band_entries")
        self.domain = domain
        self.spatial_resolution = int(spatial_resolution)
        self.spatial_bounds_local_deg = tuple(
            float(v) for v in spatial_bounds_local_deg
        )
        self.band_models = torch.nn.ModuleDict()
        self.band_wavelengths: dict[str, float] = {}
        for entry in band_entries:
            band = str(entry["band"]).strip().upper()
            target_obs = entry["target_observation"]
            self.band_models[band] = LinearBandGridField(
                domain,
                spatial_resolution=self.spatial_resolution,
                spatial_bounds_local_deg=self.spatial_bounds_local_deg,
                band_name=band,
            )
            self.band_wavelengths[band] = float(target_obs.wavelength)

    def model_for_band(self, band: str) -> LinearBandGridField:
        key = str(band).strip().upper()
        if key not in self.band_models:
            raise KeyError(f"unknown band {band!r} in _MultiBandLinearGridScene")
        model = self.band_models[key]
        assert isinstance(model, LinearBandGridField)
        return model

    def initialize_from_product(self, product: ProductResult) -> None:
        for band, model in self.band_models.items():
            assert isinstance(model, LinearBandGridField)
            model.initialize_from_product(product, band_name=band)

    def _closest_band(self, wavelength_angstrom: torch.Tensor | float | None) -> str:
        if len(self.band_models) == 1 or wavelength_angstrom is None:
            return next(iter(self.band_models.keys()))
        lam = float(torch.as_tensor(wavelength_angstrom).detach().double().mean())
        return min(
            self.band_wavelengths,
            key=lambda band: abs(self.band_wavelengths[band] - lam),
        )

    def _resolve_band(
        self,
        wavelength_angstrom: torch.Tensor | float | None,
        forward_context: ForwardContext | None = None,
    ) -> str:
        if forward_context is not None and forward_context.survey:
            key = str(forward_context.survey).strip().upper()
            if key in self.band_models:
                return key
        return self._closest_band(wavelength_angstrom)

    def forward_with_context(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelength_angstrom: torch.Tensor | float,
        context: ForwardContext | None = None,
    ) -> torch.Tensor:
        return self.forward(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
            forward_context=context,
        )

    def render_bandpass(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelengths: torch.Tensor,
        weights: torch.Tensor,
        *,
        forward_context: ForwardContext | None = None,
    ) -> torch.Tensor:
        w = torch.as_tensor(weights).detach().double()
        lam = torch.as_tensor(wavelengths).detach().double()
        w_sum = float(w.sum().item())
        mean_lam = (
            float((lam * w).sum().item() / w_sum)
            if w_sum > 0.0 and lam.numel() == w.numel()
            else float(lam.mean().item())
        )
        band = self._resolve_band(mean_lam, forward_context=forward_context)
        model = self.band_models[band]
        assert isinstance(model, LinearBandGridField)
        return model.render_bandpass(ra_deg, dec_deg, wavelengths, weights)

    def forward(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelength_angstrom: torch.Tensor | float | None = None,
        *,
        forward_context: ForwardContext | None = None,
    ) -> torch.Tensor:
        band = self._resolve_band(
            wavelength_angstrom,
            forward_context=forward_context,
        )
        return self.band_models[band](ra_deg, dec_deg, wavelength_angstrom)



def _observation_to_device(
    observation: FieldObservation,
    device: torch.device,
) -> FieldObservation:
    def _move(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else tensor.to(device=device)

    psf_spec = observation.psf_spec
    if psf_spec is not None and psf_spec.kernel2d is not None:
        psf_spec = replace(
            psf_spec,
            kernel2d=psf_spec.kernel2d.to(device=device),
        )
    return replace(
        observation,
        image=observation.image.to(device=device),
        weight=_move(observation.weight),
        mask=_move(observation.mask),
        err=_move(observation.err),
        var_poisson=_move(observation.var_poisson),
        var_rnoise=_move(observation.var_rnoise),
        var_flat=_move(observation.var_flat),
        contamination_prior=_move(observation.contamination_prior),
        amplifier_index=_move(observation.amplifier_index),
        psf_spec=psf_spec,
    )


def _fit_linear_band_grid_scene(
    observations: Sequence[FieldObservation],
    model: _MultiBandLinearGridScene,
    *,
    contexts: Sequence[ForwardContext] | None,
    operator_bundles: Sequence[ObservationOperatorBundle] | None,
    background_reference: str,
    solver: str = "cg",
    max_iterations: int = 64,
    irls_steps: int = 1,
    edge_preserving: bool = False,
    non_negative_weight: float = 0.0,
    data_weight_fn: str = "none",
    irls_use_mad_scaling: bool = False,
    damping: float = 5.0e-3,
    smoothness_weight: float = 2.0e-2,
    relative_regularization: bool = True,
    prior_reference: str = "initial",
    preconditioner: str = "diagonal_majorizer",
    tolerance: float = 1.0e-6,
    exposure_chunk_size: int = 16,
) -> FieldFitResult:
    """Solve linear or IRLS band-grid subproblems under frozen operators."""
    if not observations:
        raise ValueError("observations must not be empty")
    if contexts is not None and len(contexts) != len(observations):
        raise ValueError("contexts must align with observations")
    if operator_bundles is not None and len(operator_bundles) != len(observations):
        raise ValueError("operator_bundles must align with observations")
    requested_device = os.environ.get("TORCHSKY_DEVICE", "").strip().lower()
    if requested_device:
        solve_device = torch.device(requested_device)
    elif torch.cuda.is_available():
        solve_device = torch.device("cuda")
    else:
        solve_device = next(model.parameters()).device
    model.to(device=solve_device)
    if solve_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(solve_device)
    solve_dtype = torch.float32 if solve_device.type in ("cuda", "mps") else torch.float64

    grouped_indices: dict[str, list[int]] = {
        band: [] for band in model.band_models.keys()
    }
    for idx, obs in enumerate(observations):
        ctx = None if contexts is None else contexts[idx]
        band_key = str(
            obs.band_name or (ctx.survey if ctx is not None else "")
        ).strip().upper()
        if band_key not in grouped_indices:
            raise ValueError(
                f"observation {idx} has band {band_key!r} not in {tuple(grouped_indices)}"
            )
        grouped_indices[band_key].append(idx)

    total_initial_weighted_sq = 0.0
    total_final_weighted_sq = 0.0
    total_weight_sum = 0.0
    linear_solver_by_band: dict[str, dict[str, Any]] = {}

    operators_by_band: dict[str, Any] = {}
    targets_by_band: dict[str, list[torch.Tensor]] = {}
    init_by_band: dict[str, torch.Tensor] = {}
    initial_sq_by_band: dict[str, float] = {}
    weight_sum_by_band: dict[str, float] = {}
    valid_pixels_by_band: dict[str, int] = {}

    for band_key, indices in grouped_indices.items():
        if not indices:
            raise ValueError(f"no training observations for band {band_key!r}")
        band_model = model.model_for_band(band_key)
        band_observations: list[FieldObservation] = []
        band_targets: list[torch.Tensor] = []

        for idx in indices:
            obs_dev = _observation_to_device(observations[idx], solve_device)
            ctx = None if contexts is None else contexts[idx]
            if operator_bundles is not None:
                bundle = operator_bundles[idx]
                for op in (bundle.psf, bundle.background, bundle.noise):
                    op_model = getattr(op, "model", None)
                    if isinstance(op_model, torch.nn.Module):
                        op_model.to(device=solve_device)
                adapted = adapt_stationary_calibrated_band_observation(
                    obs_dev,
                    bundle,
                    forward_context=ctx,
                    expected_value_unit=str(obs_dev.value_unit or "MJy/sr"),
                    background_gauge_policy=str(background_reference),
                )
                band_observations.append(adapted.observation)
                band_targets.append(adapted.prepared_data.to(dtype=solve_dtype))
            else:
                psf_spec = obs_dev.psf_spec
                sigma_pix = float(
                    psf_spec.sigma_pix
                    if psf_spec is not None and psf_spec.mode == "gaussian_sigma_pix"
                    else obs_dev.psf_sigma_pix
                )
                if psf_spec is not None and psf_spec.mode == "kernel2d":
                    if psf_spec.kernel2d is None:
                        raise ValueError("kernel2d PSFSpec requires kernel2d tensor")
                    effective_psf = replace(
                        psf_spec,
                        frozen=True,
                        pixel_response_included=True,
                    )
                elif sigma_pix > 0.0:
                    k1 = gaussian_kernel1d(
                        sigma_pix,
                        device=solve_device,
                        dtype=solve_dtype,
                    )
                    k2 = torch.outer(k1, k1)
                    k2 = k2 / k2.sum()
                    effective_psf = PSFSpec(
                        mode="kernel2d",
                        sigma_pix=sigma_pix,
                        kernel2d=k2,
                        frozen=True,
                        pixel_response_included=True,
                    )
                else:
                    effective_psf = PSFSpec(
                        mode="none",
                        sigma_pix=0.0,
                        frozen=True,
                        pixel_response_included=True,
                    )
                obs_effective = replace(obs_dev, psf_spec=effective_psf)
                prepared = prepare_linear_band_grid_observation_data(obs_effective)
                band_observations.append(obs_effective)
                band_targets.append(prepared.to(dtype=solve_dtype))

        op_set = bind_linear_band_grid_observations_to_operator_set(
            band_observations,
            band_model.domain,
            spatial_resolution=int(band_model.spatial_resolution),
            spatial_bounds_local_deg=band_model.spatial_bounds_local_deg,
            exposure_chunk_size=int(exposure_chunk_size),
        )
        x0 = band_model.grid.detach().to(dtype=solve_dtype).clone()
        band_initial_sq = 0.0
        band_weight_sum = 0.0
        band_valid_pixels = 0
        with torch.no_grad():
            has_nonzero_x0 = bool(torch.any(x0 != 0.0))
            preds0 = (
                op_set.apply(x0)
                if has_nonzero_x0
                else tuple(torch.zeros_like(t) for t in band_targets)
            )
            for bound_op, pred0, target in zip(
                op_set.operators,
                preds0,
                band_targets,
                strict=True,
            ):
                diff0 = pred0 - target
                weighted_diff0 = bound_op.apply_noise_model(diff0)
                band_initial_sq += float(
                    torch.sum(diff0.double() * weighted_diff0.double()).cpu()
                )
                w_ones = bound_op.apply_noise_model(torch.ones_like(target))
                band_weight_sum += float(torch.sum(w_ones.double()).cpu())
                band_valid_pixels += int((w_ones > 0).sum().cpu())

        operators_by_band[band_key] = op_set
        targets_by_band[band_key] = band_targets
        init_by_band[band_key] = x0
        initial_sq_by_band[band_key] = band_initial_sq
        weight_sum_by_band[band_key] = band_weight_sum
        valid_pixels_by_band[band_key] = band_valid_pixels

    diagnostics_by_band: dict[str, dict[str, Any]] = {}
    solve_results_by_band: dict[str, Any] = {}
    irls_cfg = (
        IRLSConfig(
            max_iter=max(1, int(irls_steps)),
            weight_fn=str(data_weight_fn),
            use_mad_scaling=bool(irls_use_mad_scaling),
        )
        if str(data_weight_fn).lower() != "none"
        else None
    )
    if (
        solver == "cg"
        and int(irls_steps) > 1
        and (bool(edge_preserving) or str(data_weight_fn).lower() != "none")
        and len(operators_by_band) > 1
    ):
        solve_results_by_band = solve_joint_multiband_grid_cg(
            operators_by_band,
            targets_by_band,
            initial_solutions=init_by_band,
            iterations=int(max_iterations),
            irls_steps=int(irls_steps),
            damping=float(damping),
            smoothness_weight=float(smoothness_weight),
            relative_regularization=bool(relative_regularization),
            prior_reference=str(prior_reference),  # type: ignore[arg-type]
            preconditioner=str(preconditioner),  # type: ignore[arg-type]
            tolerance=float(tolerance),
            edge_preserving=bool(edge_preserving),
            non_negative_weight=float(non_negative_weight),
            data_weight_fn=str(data_weight_fn),
            irls_config=irls_cfg,
            diagnostics_by_band=diagnostics_by_band,
        )
    elif solver == "cg":
        for band_key in grouped_indices:
            solver_diag: dict[str, Any] = {}
            solve_results_by_band[band_key] = solve_linear_band_grid_cg(
                operators_by_band[band_key],
                targets_by_band[band_key],
                initial_solution=init_by_band[band_key],
                iterations=int(max_iterations),
                damping=float(damping),
                smoothness_weight=float(smoothness_weight),
                relative_regularization=bool(relative_regularization),
                prior_reference=str(prior_reference),  # type: ignore[arg-type]
                preconditioner=str(preconditioner),  # type: ignore[arg-type]
                tolerance=float(tolerance),
                irls_steps=int(irls_steps),
                edge_preserving=bool(edge_preserving),
                non_negative_weight=float(non_negative_weight),
                data_weight_fn=str(data_weight_fn),
                irls_config=irls_cfg,
                diagnostics=solver_diag,
            )
            diagnostics_by_band[band_key] = solver_diag
    elif solver == "lsqr":
        if float(smoothness_weight) > 0.0:
            raise ValueError("lsqr does not support smoothness_weight > 0")
        for band_key in grouped_indices:
            solver_diag = {}
            solve_results_by_band[band_key] = solve_linear_band_grid_lsqr(
                operators_by_band[band_key],
                targets_by_band[band_key],
                initial_solution=init_by_band[band_key],
                iterations=int(max_iterations),
                damping=float(damping),
                prior_reference=str(prior_reference),  # type: ignore[arg-type]
                tolerance=float(tolerance),
                diagnostics=solver_diag,
            )
            diagnostics_by_band[band_key] = solver_diag
    else:
        raise ValueError(f"unsupported linear_band_grid_field solver: {solver!r}")

    for band_key in grouped_indices:
        band_model = model.model_for_band(band_key)
        op_set = operators_by_band[band_key]
        band_targets = targets_by_band[band_key]
        solve_result = solve_results_by_band[band_key]
        solver_diag = diagnostics_by_band[band_key]
        band_initial_sq = initial_sq_by_band[band_key]
        band_weight_sum = weight_sum_by_band[band_key]
        band_valid_pixels = valid_pixels_by_band[band_key]

        with torch.no_grad():
            sol_t = solve_result.solution
            band_model.grid.copy_(sol_t.to(dtype=band_model.grid.dtype))
            band_final_sq = 0.0
            preds1 = op_set.apply(sol_t)
            for bound_op, pred1, target in zip(
                op_set.operators,
                preds1,
                band_targets,
                strict=True,
            ):
                diff1 = pred1 - target
                weighted_diff1 = bound_op.apply_noise_model(diff1)
                band_final_sq += float(
                    torch.sum(diff1.double() * weighted_diff1.double()).cpu()
                )

        total_initial_weighted_sq += band_initial_sq
        total_final_weighted_sq += band_final_sq
        total_weight_sum += band_weight_sum
        denom = max(band_weight_sum, 1.0e-30)
        linear_solver_by_band[band_key] = {
            "linear_solver_kind": str(solver_diag.get("solver", solver)),
            "linear_solver_irls_steps": int(solver_diag.get("irls_steps", irls_steps)),
            "linear_solver_edge_preserving": bool(
                solver_diag.get("edge_preserving", edge_preserving)
            ),
            "linear_solver_non_negative_weight": float(
                solver_diag.get("non_negative_weight", non_negative_weight)
            ),
            "linear_solver_data_weight_fn": str(
                solver_diag.get("data_weight_fn", data_weight_fn)
            ),
            "linear_solver_damping": float(damping),
            "linear_solver_smoothness_weight": float(smoothness_weight),
            "linear_solver_relative_regularization": bool(relative_regularization),
            "linear_solver_effective_damping": float(
                solver_diag.get("effective_damping", float(damping))
            ),
            "linear_solver_effective_smoothness_weight": float(
                solver_diag.get(
                    "effective_smoothness_weight", float(smoothness_weight)
                )
            ),
            "linear_solver_prior_reference": str(prior_reference),
            "linear_solver_preconditioner": str(
                solver_diag.get(
                    "preconditioner",
                    str(preconditioner) if solver == "cg" else "none",
                )
            ),
            "linear_solver_iterations": int(solver_diag.get("iterations", 0)),
            "linear_solver_converged": bool(solver_diag.get("converged", False)),
            "linear_solver_relative_residual": float(
                solver_diag.get("relative_normal_residual", float("nan"))
            ),
            "linear_solver_initial_normal_residual_norm": float(
                solver_diag.get("initial_normal_residual_norm", float("nan"))
            ),
            "linear_solver_final_normal_residual_norm": float(
                solver_diag.get("final_normal_residual_norm", float("nan"))
            ),
            "linear_solver_normal_operator_applications": int(
                solver_diag.get("normal_operator_applications", 0)
            ),
            "linear_solver_rhs_and_preconditioner_seconds": float(
                solver_diag.get("rhs_and_preconditioner_seconds", float("nan"))
            ),
            "linear_solver_solve_seconds": float(
                solver_diag.get("solve_seconds", float("nan"))
            ),
            "linear_solver_grid_covered_fraction": float(
                solver_diag.get("grid_covered_fraction", float("nan"))
            ),
            "linear_solver_grid_sensitivity_median": float(
                solver_diag.get("grid_sensitivity_median", float("nan"))
            ),
            "linear_solver_grid_sensitivity_p10_p90_ratio": float(
                solver_diag.get("grid_sensitivity_p10_p90_ratio", float("nan"))
            ),
            "linear_solver_initial_detector_mse": band_initial_sq / denom,
            "linear_solver_final_detector_mse": band_final_sq / denom,
            "linear_solver_valid_pixels": int(band_valid_pixels),
        }

    max_memory_allocated_bytes = (
        int(torch.cuda.max_memory_allocated(solve_device))
        if solve_device.type == "cuda"
        else 0
    )
    global_denom = max(total_weight_sum, 1.0e-30)
    initial_loss = total_initial_weighted_sq / global_denom
    final_loss = total_final_weighted_sq / global_denom
    return FieldFitResult(
        model=model,
        loss_history=(initial_loss, final_loss),
        final_loss=final_loss,
        _pre_train_loss=initial_loss,
        diagnostics={
            "linear_solver_by_band": linear_solver_by_band,
            "performance_profile": {
                "max_memory_allocated_bytes": max_memory_allocated_bytes,
            },
        },
    )


def _render_joint_scene_product(
    model: torch.nn.Module, observation: FieldObservation,
    fit_kwargs: dict[str, Any], num_spectral_samples: int,
    context: ForwardContext | None,
) -> torch.Tensor:
    """Export using the fitted pixel integration, without detector backgrounds."""
    target_model = getattr(model, "inner", model)
    render_sampling = str(fit_kwargs["spatial_sampling"])
    render_obs = observation
    crop_slice: tuple[slice, slice] | None = None
    if isinstance(target_model, _MultiBandLinearGridScene) and observation.band_name:
        model = target_model.model_for_band(str(observation.band_name))
        num_spectral_samples = 1
        if (
            observation.psf_spec is not None
            and observation.psf_spec.pixel_response_included is False
        ):
            render_sampling = "gauss_legendre_2x2"
        halo_y, halo_x = psf_support_radius(observation.psf_spec)
        height, width = observation.image.shape
        if (
            (halo_y > 0 or halo_x > 0)
            and observation.detector_to_sky is None
            and height > 4 * halo_y
            and width > 4 * halo_x
        ):
            padded_h = height + 2 * halo_y
            padded_w = width + 2 * halo_x
            padded_header = _copy_header_with_crop(
                observation.header,
                x0=-halo_x,
                y0=-halo_y,
                width=padded_w,
                height=padded_h,
            )
            render_obs = replace(
                observation,
                image=torch.zeros(
                    (padded_h, padded_w),
                    dtype=observation.image.dtype,
                    device=observation.image.device,
                ),
                header=padded_header,
                weight=None,
                mask=None,
            )
            crop_slice = (
                slice(halo_y, halo_y + height),
                slice(halo_x, halo_x + width),
            )
    with torch.no_grad():
        rendered = render_field_observation(
            model, render_obs, num_spectral_samples=num_spectral_samples,
            spectral_sampling="deterministic_quadrature",
            spatial_sampling=render_sampling,
            spatial_tile_hw=fit_kwargs["render_spatial_tile_hw"],
            forward_context=context,
        )
        if crop_slice is not None:
            rendered = rendered[crop_slice]
        return rendered


def _validation_detector_scores(
    model: torch.nn.Module,
    observations: Sequence[FieldObservation],
    contexts: Sequence[ForwardContext],
    bundles: Sequence[ObservationOperatorBundle],
    *,
    intensity_scale: float,
    fit_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score fixed detector predictions under pipeline ERR, without residual fitting."""
    if not (len(observations) == len(contexts) == len(bundles)):
        raise ValueError("validation observations, contexts and bundles must align")
    if any(context.extra.get("split") != "validation" for context in contexts):
        raise ValueError("detector scoring accepts only the validation split")
    scale = float(intensity_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("validation intensity_scale must be finite and positive")
    scores: dict[str, dict[str, Any]] = {}
    was_training = model.training
    model.eval()
    try:
        device = next((parameter.device for parameter in model.parameters()), torch.device("cpu"))
        zero_scene = _ZeroScene(device).eval()
        for bundle in bundles:
            for operator in (getattr(bundle, "psf", None), getattr(bundle, "background", None)):
                operator_model = getattr(operator, "model", None)
                if isinstance(operator_model, torch.nn.Module):
                    operator_model.to(device=device)
        with torch.no_grad():
            for observation, context, bundle in zip(observations, contexts, bundles, strict=True):
                if observation.err is None or observation.mask is None:
                    raise ValueError("validation detector scoring requires ERR and mask")
                image = observation.image.detach().cpu().double()
                err = observation.err.detach().cpu().double()
                valid = (observation.mask.detach().cpu() > 0) & torch.isfinite(image)
                valid &= torch.isfinite(err) & (err > 0)
                if observation.weight is not None:
                    weight = observation.weight.detach().cpu()
                    valid &= torch.isfinite(weight) & (weight > 0)
                count = int(valid.sum())
                if count == 0:
                    raise ValueError(f"validation exposure has no valid pixels: {context.exposure_id}")
                prediction = render_field_observation(
                    model, observation,
                    num_spectral_samples=int(fit_kwargs.get("num_spectral_samples", 9)),
                    spectral_sampling="deterministic_quadrature",
                    spatial_sampling=fit_kwargs.get("spatial_sampling", "center"),
                    spatial_tile_hw=fit_kwargs.get("render_spatial_tile_hw", (256, 256)),
                    forward_context=context, operator_bundle=bundle,
                ).detach().cpu().double()
                if prediction.shape != image.shape or not torch.isfinite(prediction[valid]).all():
                    raise ValueError("validation prediction has invalid shape or nonfinite valid pixels")
                fixed_prediction = render_field_observation(
                    zero_scene, observation,
                    num_spectral_samples=int(fit_kwargs.get("num_spectral_samples", 9)),
                    spectral_sampling="deterministic_quadrature",
                    spatial_sampling=fit_kwargs.get("spatial_sampling", "center"),
                    spatial_tile_hw=fit_kwargs.get("render_spatial_tile_hw", (256, 256)),
                    forward_context=context, operator_bundle=bundle,
                ).detach().cpu().double()
                if fixed_prediction.shape != image.shape or not torch.isfinite(fixed_prediction[valid]).all():
                    raise ValueError("zero-scene detector prediction is invalid")
                available_scene = (image[valid] - fixed_prediction[valid]) * scale
                chi2 = ((prediction[valid] - image[valid]) / err[valid]).square()
                # Restore the density to original SCI units after train-only scaling.
                nll = .5 * (chi2 + math.log(2 * math.pi)) + torch.log(err[valid]) + math.log(scale)
                band = str(context.survey)
                row = scores.setdefault(band, {
                    "band": band, "exposures": 0, "pixels": 0,
                    "gaussian_nll_sum": 0., "chi2_sum": 0.,
                    "value_unit": observation.value_unit,
                    "per_exposure": [],
                })
                if row["value_unit"] != observation.value_unit:
                    raise ValueError("validation exposures must share units within a band")
                row["exposures"] += 1
                row["pixels"] += count
                row["gaussian_nll_sum"] += float(nll.sum())
                row["chi2_sum"] += float(chi2.sum())
                row["per_exposure"].append({
                    "exposure_id": context.exposure_id,
                    "source_path": context.extra.get("source_path"),
                    "pixels": count,
                    "gaussian_nll_per_pixel": float(nll.mean()),
                    "chi2_per_pixel": float(chi2.mean()),
                    "observed_mean": float(image[valid].mean()) * scale,
                    "fixed_detector_mean": float(fixed_prediction[valid].mean()) * scale,
                    "predicted_scene_mean": float((prediction[valid] - fixed_prediction[valid]).mean()) * scale,
                    "observed_minus_fixed_median": float(available_scene.median()),
                    "observed_minus_fixed_negative_fraction": float((available_scene < 0).double().mean()),
                    "brightness_unit": observation.value_unit,
                })
    finally:
        model.train(was_training)
    for row in scores.values():
        row["gaussian_nll_per_pixel"] = row["gaussian_nll_sum"] / row["pixels"]
        row["chi2_per_pixel"] = row["chi2_sum"] / row["pixels"]
    return list(scores.values())


_CalibrationKey = str | tuple[str, str, str]


def _exposure_calibration_key(context: ForwardContext) -> tuple[str, str, str]:
    exposure = context.extra.get("calibration_exposure_id")
    if not isinstance(exposure, str) or not exposure.strip() or not context.survey:
        raise ValueError("exposure calibration requires exact manifest exposure identity and band")
    return str(context.detector or "").upper(), exposure, str(context.survey).upper()


def _load_calibrated_operator_models(
    manifest: Path,
    fit_report: Path,
    *,
    detectors: set[str],
    intensity_scale: float,
    family: str = "adaptive",
    contexts: Sequence[ForwardContext] = (),
) -> tuple[dict[_CalibrationKey, dict[str, torch.nn.Module]], dict[str, Any]]:
    if family not in {"stationary", "adaptive"}:
        raise ValueError("calibration family must be stationary or adaptive")
    manifest = manifest.resolve()
    report = json.loads(fit_report.read_text(encoding="utf-8"))
    if report.get("manifest_sha256") != calibration_fit._sha256(manifest):
        raise ValueError("operator fit report does not match the calibration manifest")
    scope = report.get("calibration_scope", "detector")
    if scope not in {"detector", "exposure"}:
        raise ValueError("unsupported calibration scope")
    if scope == "exposure" and not contexts:
        raise ValueError("exposure calibration requires observation contexts")
    indexed = {}
    for group in report["groups"]:
        key = (
            str(group["detector"]).upper(),
            group.get("exposure_id") if scope == "exposure" else None,
            str(group.get("band", "")).upper() if scope == "exposure" else None,
            str(group["role"]),
            str(group["family"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate calibrated operator group {key}")
        indexed[key] = group
    keys: list[_CalibrationKey] = (
        list(sorted({_exposure_calibration_key(context) for context in contexts}))
        if scope == "exposure" else list(sorted(detectors))
    )
    models: dict[_CalibrationKey, dict[str, torch.nn.Module]] = {}
    audit = []
    for model_key in keys:
        detector, exposure_id, band = (
            model_key if isinstance(model_key, tuple) else (model_key, None, None)
        )
        models[model_key] = {}
        for role in ("background", "psf"):
            group = indexed.get((detector, exposure_id, band, role, family))
            if group is None:
                raise ValueError(f"fit report has no {family} {model_key}/{role} group; independent exposure calibration is required")
            checkpoint_path = Path(group["checkpoint"])
            if calibration_fit._sha256(checkpoint_path) != group["checkpoint_sha256"]:
                raise ValueError(f"{detector}/{role} checkpoint hash mismatch")
            arrays = load_operator_calibration_split(
                manifest,
                split="train",
                role=role,
                detector=detector,
                exposure_id=exposure_id,
                band=band,
                verify_hashes=True,
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model, _initializer = calibration_fit._make_model(
                arrays,
                role=role,
                family=family,
                device=torch.device("cpu"),
                psf_model=str(checkpoint.get("psf_model") or "fitted_gaussian_kernel"),
                exposure_scope=exposure_id is not None,
            )
            if (
                checkpoint.get("schema_version") != 2
                or checkpoint.get("detector") != detector
                or checkpoint.get("role") != role
                or checkpoint.get("family") != family
                or (scope == "exposure" and (
                    checkpoint.get("exposure_id") != exposure_id
                    or checkpoint.get("band") != band
                    or checkpoint.get("calibration_scope") != "exposure"
                ))
            ):
                raise ValueError(f"invalid {detector}/{role} checkpoint metadata")
            if role == "psf":
                pixel_response_included = checkpoint.get("pixel_response_included")
                pixel_response_provenance = checkpoint.get(
                    "pixel_response_provenance"
                )
                if (
                    not isinstance(pixel_response_included, bool)
                    or not isinstance(pixel_response_provenance, str)
                    or not pixel_response_provenance.strip()
                    or group.get("pixel_response_included")
                    != pixel_response_included
                    or group.get("pixel_response_provenance")
                    != pixel_response_provenance
                ):
                    raise ValueError(
                        f"invalid {detector}/psf pixel-response metadata"
                    )
                model._torchsky_pixel_response_included = pixel_response_included
                model._torchsky_pixel_response_provenance = (
                    pixel_response_provenance
                )
            model.load_state_dict(checkpoint["state_dict"])
            if isinstance(
                getattr(model, "encoder", None),
                AdaptiveSpatioSpectralTemporalEncoder,
            ):
                pruning = apply_operator_calibration_pruning(
                    model, group["validation_pruning"]
                )
            else:
                pruning = {
                    "pruned_bands": [],
                    "note": "model has no adaptive gates to prune",
                }
            _freeze_calibrated_operator_model(
                model, role=role, intensity_scale=intensity_scale
            )
            models[model_key][role] = model
            audit.append(
                {
                    "detector": detector,
                    "exposure_id": exposure_id,
                    "band": band,
                    "role": role,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": group["checkpoint_sha256"],
                    **(
                        {
                            "pixel_response_included": pixel_response_included,
                            "pixel_response_provenance": pixel_response_provenance,
                        }
                        if role == "psf"
                        else {}
                    ),
                    **pruning,
                }
            )
    return models, {
        "calibration_scope": scope,
        "manifest": str(manifest),
        "manifest_sha256": report["manifest_sha256"],
        "fit_report": str(fit_report.resolve()),
        "intensity_scale": float(intensity_scale),
        "models": audit,
    }


def _freeze_calibrated_operator_model(
    model: torch.nn.Module, *, role: str, intensity_scale: float
) -> None:
    scale = float(intensity_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("calibrated operator intensity_scale must be positive")
    if role == "background":
        if not isinstance(model, ConditionedBackgroundField):
            raise TypeError("background checkpoint produced the wrong model")
        output = model.network[-1]
        if not isinstance(output, torch.nn.Linear):
            raise TypeError("background model must end with a linear layer")
        with torch.no_grad():
            output.weight.div_(scale)
            output.bias.div_(scale)
    elif role != "psf":
        raise ValueError("calibrated operator role must be background or psf")
    model.requires_grad_(False)
    model.eval()


def _reference_calibrated_backgrounds_to_training_mean(
    models: dict[_CalibrationKey, dict[str, torch.nn.Module]],
    contexts: Sequence[ForwardContext],
    *,
    manifest: Path | dict[str, Any],
    intensity_scale: float,
) -> dict[str, Any]:
    """Choose a relative background gauge from independent training evidence."""
    if not math.isfinite(intensity_scale) or intensity_scale <= 0:
        raise ValueError("background reference requires a positive intensity scale")
    model_bands: dict[int, str] = {}
    for key, operators in models.items():
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError("training_mean background reference requires exposure scope")
        model = operators.get("background")
        if not isinstance(model, ConditionedBackgroundField) or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise ValueError("background reference requires frozen learned backgrounds")
        if not isinstance(model.network[-1], torch.nn.Linear):
            raise TypeError("background model must end with a linear layer")
        if model_bands.setdefault(id(model), key[2]) != key[2]:
            raise ValueError("background reference cannot share models across bands")
    training_keys = sorted({
        _exposure_calibration_key(context) for context in contexts
        if context.extra.get("split") == "train"
    })
    bands: dict[str, Any] = {}
    with torch.no_grad():
        for key in training_keys:
            detector, exposure_id, band = key
            model = models[key]["background"]
            parameter = next(model.parameters())
            arrays = load_operator_calibration_split(
                manifest, split="train", role="background", detector=detector,
                exposure_id=exposure_id, band=band, verify_hashes=True,
            )
            coordinates = [arrays[name].to(parameter) for name in
                           ("detector_x", "detector_y", "wavelength", "time_mjd")]
            values = model(*coordinates).detach().cpu().double()
            weights = arrays["sample_weight"].detach().cpu().double()
            if (values.shape != weights.shape or not torch.isfinite(values).all()
                    or not torch.isfinite(weights).all() or bool((weights < 0).any())
                    or float(weights.sum()) <= 0):
                raise ValueError("background reference requires finite positive-weight evidence")
            entry = bands.setdefault(band, {
                "weighted_sum": 0., "sample_weight_sum": 0., "sample_count": 0,
                "training_exposures": [],
            })
            weight_sum = float(weights.sum())
            entry["weighted_sum"] += float((values * weights).sum())
            entry["sample_weight_sum"] += weight_sum
            entry["sample_count"] += int(weights.numel())
            entry["training_exposures"].append({
                "detector": detector, "exposure_id": exposure_id,
                "sample_count": int(weights.numel()), "sample_weight_sum": weight_sum,
            })
        if not bands or any(band not in bands for band in model_bands.values()):
            raise ValueError("background reference needs training evidence in every band")
        for entry in bands.values():
            offset = entry.pop("weighted_sum") / entry["sample_weight_sum"]
            entry["offset_normalized"] = offset
            entry["offset_mjy_sr"] = offset * float(intensity_scale)
        shifted = set()
        for key, operators in models.items():
            model = operators["background"]
            if id(model) not in shifted:
                model.network[-1].bias.sub_(bands[key[2]]["offset_normalized"])
                shifted.add(id(model))
    return {
        "policy": "training_mean",
        "evidence": "science-training exposures; full-exposure calibration train samples",
        "transformation": "after intensity normalization subtract band offset from final background readout bias once per model, including validation models",
        "absolute_foreground": "unestablished; common pedestal retained in scene",
        "bands": bands,
    }


def _calibrated_operator_bundles(
    observations: list[FieldObservation],
    contexts: list[ForwardContext],
    models: dict[_CalibrationKey, dict[str, torch.nn.Module]],
) -> tuple[ObservationOperatorBundle, ...]:
    if len(observations) != len(contexts):
        raise ValueError("calibrated observations and contexts must have equal length")
    bundles = []
    exposure_scope = any(isinstance(key, tuple) for key in models)
    if exposure_scope and any(isinstance(key, str) for key in models):
        raise ValueError("cannot mix exposure and detector calibration scopes")
    for observation, context in zip(observations, contexts, strict=True):
        detector = str(context.detector or "").upper()
        key = _exposure_calibration_key(context) if exposure_scope else detector
        if key not in models:
            raise ValueError(f"no calibrated operators for {key!r}")
        psf_model = models[key]["psf"]
        pixel_response_included = getattr(
            psf_model, "_torchsky_pixel_response_included", None
        )
        pixel_response_provenance = getattr(
            psf_model, "_torchsky_pixel_response_provenance", None
        )
        if not isinstance(pixel_response_included, bool) or not isinstance(
            pixel_response_provenance, str
        ):
            raise ValueError(
                f"calibrated PSF for {key!r} lacks pixel-response metadata"
            )
        frozen = not any(parameter.requires_grad for parameter in psf_model.parameters())
        if not frozen:
            raise ValueError(f"calibrated PSF for {key!r} is not frozen")
        bundles.append(
            field_observation_operator_bundle(observation).with_updates(
                psf=LearnedPSFOp(
                    psf_model,
                    frozen=True,
                    pixel_response_included=pixel_response_included,
                    pixel_response_provenance=pixel_response_provenance,
                ),
                background=LearnedBackgroundOp(
                    models[key]["background"], coordinate_frame="detector"
                ),
            )
        )
    return tuple(bundles)


def _spatial_sampling_for_operator_bundles(
    requested: str,
    bundles: Sequence[ObservationOperatorBundle],
) -> str:
    psfs = [bundle.psf for bundle in bundles if bundle.psf is not None]
    if not psfs:
        return requested
    conventions = {getattr(psf, "pixel_response_included", None) for psf in psfs}
    if None in conventions:
        raise ValueError("PSF pixel-response convention is unknown")
    if len(conventions) != 1:
        raise ValueError(
            "one fit cannot mix PSFs with different pixel-response conventions"
        )
    if conventions == {True}:
        return "center"
    return "gauss_legendre_2x2" if requested == "center" else requested


def _pixel_response_convention(
    bundles: Sequence[ObservationOperatorBundle],
) -> str:
    psfs = [bundle.psf for bundle in bundles if bundle.psf is not None]
    if not psfs:
        return "none"
    conventions = {getattr(psf, "pixel_response_included", None) for psf in psfs}
    if conventions == {True}:
        return "effective_includes_pixel_response"
    if conventions == {False}:
        return "optical_only"
    if None in conventions:
        return "unknown"
    return "mixed"


def _apply_learned_psf_arm(
    observations: list[FieldObservation],
    bands: list[str],
    *,
    grid_side: int = 4,
    spectral_degree: int = 2,
) -> dict[str, float]:
    """Swap fitted Gaussian PSF specs for shared per-band learned models.

    One :class:`LowRankChromaticPSF` per band (base width from the fitted
    Gaussian contract); every exposure evaluates it at its own detector
    control-cell centers. ponytail: in-band spectral coordinate is the
    band index normalized to [0, 1]; upgrade path is true bandpass-weighted
    central wavelengths.
    """
    models: dict[str, LowRankChromaticPSF] = {}
    # True in-band spectral coordinate: band pivot wavelengths normalized
    # across the run's band set (not a bare band index).
    pivots = {
        band: float(nircam_imaging_band(band).pivot_wavelength_angstrom)
        for band in bands
    }
    pivot_lo, pivot_hi = min(pivots.values()), max(pivots.values())
    for observation in observations:
        band = str(observation.band_name)
        if band not in bands:
            raise ValueError(f"observation band {band!r} not in {bands}")
        model = models.get(band)
        if model is None:
            model = LowRankChromaticPSF(
                grid_side=grid_side,
                spectral_degree=spectral_degree,
                base_sigma_pix=float(observation.psf_sigma_pix),
            )
            # Calibration-anchored nuisance: frozen during scene fitting so
            # renders through it build no autograd graph.
            model.requires_grad_(False)
            models[band] = model.to(
                next(iter({o.image.device for o in observations}), torch.device("cpu"))
            ).to(
                next(
                    iter({o.image.device for o in observations if hasattr(o, "image")}),
                    torch.device("cpu"),
                )
            )
        height, width = (int(v) for v in observation.image.shape[-2:])
        xs = torch.linspace(0.0, width - 1.0, grid_side) + float(
            observation.detector_x0_pix
        )
        ys = torch.linspace(0.0, height - 1.0, grid_side) + float(
            observation.detector_y0_pix
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        cell_u_pix = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        lam_norm = (pivots[band] - pivot_lo) / max(pivot_hi - pivot_lo, 1e-9)
        lam_norm = min(max(lam_norm, 0.0), 1.0)
        observation.psf_spec = replace(
            PSFSpec.learned(
                model,
                base_sigma_pix=float(observation.psf_sigma_pix),
                cell_u_pix=cell_u_pix,
                cell_lam_norm=lam_norm,
            ),
            pixel_response_included=False,
        )
    return {band: float(model.base_sigma_pix) for band, model in models.items()}


def _regular_foam_model_kwargs(
    *,
    grid_side: int,
    basis_count: int,
    affine_spatial_basis: bool = False,
    extent: float = 1.0,
    bounds_xy: tuple[tuple[float, float], tuple[float, float]] | None = None,
    log_intensity_spectra: bool = False,
    initial_intensity: float = 1.0e-2,
) -> dict[str, Any]:
    """Deterministic Power-Foam initialization independent of science pixels."""
    side = int(grid_side)
    basis = int(basis_count)
    limit = float(extent)
    if side < 2:
        raise ValueError("grid_side must be at least 2")
    if basis < 1:
        raise ValueError("basis_count must be positive")
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("extent must be finite and positive")
    if not math.isfinite(initial_intensity) or initial_intensity <= 0:
        raise ValueError("initial_intensity must be finite and positive")
    bounds = bounds_xy or ((-limit, limit), (-limit, limit))
    (x_min, x_max), (y_min, y_max) = bounds
    if not all(math.isfinite(value) for pair in bounds for value in pair):
        raise ValueError("bounds_xy must be finite")
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("bounds_xy must have increasing x/y limits")
    step_x = (x_max - x_min) / float(side)
    step_y = (y_max - y_min) / float(side)
    centers_x = torch.linspace(
        x_min + 0.5 * step_x,
        x_max - 0.5 * step_x,
        steps=side,
        dtype=torch.float32,
    )
    centers_y = torch.linspace(
        y_min + 0.5 * step_y,
        y_max - 0.5 * step_y,
        steps=side,
        dtype=torch.float32,
    )
    yy, xx = torch.meshgrid(centers_y, centers_x, indexing="ij")
    sites = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    radii = torch.full(
        (side * side,),
        math.hypot(step_x, step_y),
        dtype=torch.float32,
    )
    coefficients = torch.zeros((side * side, basis), dtype=torch.float32)
    if log_intensity_spectra:
        coefficients[:, 0] = math.log(float(initial_intensity))
    inverse_cell_area = 1.0 / (step_x * step_y)
    result = {
        "sites": sites,
        "radii": radii,
        "cell_coefficients": coefficients,
        "minimum_radius": 0.25 * min(step_x, step_y),
        "temperature_start": FOAM_TEMPERATURE_START_CELL_UNITS * inverse_cell_area,
        "temperature_end": FOAM_TEMPERATURE_END_CELL_UNITS * inverse_cell_area,
        "log_intensity_spectra": bool(log_intensity_spectra),
    }
    if affine_spatial_basis:
        result["cell_gradient_coefficients"] = torch.zeros(
            (side * side, basis, 2), dtype=torch.float32
        )
    return result


def _target_boundary_world(header: dict[str, Any]) -> torch.Tensor:
    width = int(header.get("NAXIS1", 0))
    height = int(header.get("NAXIS2", 0))
    if width <= 0 or height <= 0:
        raise ValueError("target WCS requires positive NAXIS1/NAXIS2")
    # ponytail: 17 samples per edge bound ordinary distorted imaging WCS;
    # upgrade to adaptive extrema only if a high-order WCS violates the box.
    x = torch.linspace(-0.5, width - 0.5, 17, dtype=torch.float64)
    y = torch.linspace(-0.5, height - 0.5, 17, dtype=torch.float64)
    boundary = torch.cat(
        (
            torch.stack((x, torch.full_like(x, -0.5)), dim=-1),
            torch.stack((torch.full_like(y, width - 0.5), y), dim=-1),
            torch.stack((x, torch.full_like(x, height - 0.5)), dim=-1),
            torch.stack((torch.full_like(y, -0.5), y), dim=-1),
        )
    )
    return torchsky.WCS(header).pixel_to_world(boundary)


def _target_footprint_local_bounds_deg(
    domain: Any,
    band_entries: list[dict[str, Any]],
) -> tuple[float, float, float, float]:
    if not band_entries:
        raise ValueError("band_entries must not be empty")
    world = torch.cat(
        [_target_boundary_world(dict(entry["target_header"])) for entry in band_entries]
    )
    delta_ra = (
        torch.remainder(
            world[:, 0] - float(domain.center_ra_deg) + 180.0,
            360.0,
        )
        - 180.0
    )
    x = delta_ra * math.cos(math.radians(float(domain.center_dec_deg)))
    y = world[:, 1] - float(domain.center_dec_deg)
    return (
        float(torch.min(x)),
        float(torch.max(x)),
        float(torch.min(y)),
        float(torch.max(y)),
    )


def _foam_target_footprint_bounds(
    domain: Any,
    band_entries: list[dict[str, Any]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return target-WCS bounds in the Foam's normalized tangent plane."""
    x_min, x_max, y_min, y_max = _target_footprint_local_bounds_deg(
        domain, band_entries
    )
    scale = float(domain.spatial_scale_deg)
    return (
        (x_min / scale, x_max / scale),
        (y_min / scale, y_max / scale),
    )


def _grid_target_footprint_bounds(
    domain: Any,
    band_entries: list[dict[str, Any]],
    *,
    margin_sigma: float = 4.0,
) -> tuple[float, float, float, float]:
    """Return target bounds in tangent-plane degrees with a Gaussian PSF halo."""
    if not math.isfinite(float(margin_sigma)) or margin_sigma < 0.0:
        raise ValueError("margin_sigma must be finite and non-negative")
    x_min, x_max, y_min, y_max = _target_footprint_local_bounds_deg(
        domain, band_entries
    )
    fwhm_values = [float(entry["fitted_psf_fwhm_arcsec"]) for entry in band_entries]
    if not all(math.isfinite(value) and value >= 0.0 for value in fwhm_values):
        raise ValueError("fitted_psf_fwhm_arcsec must be finite and non-negative")
    fwhm_arcsec = max(fwhm_values)
    sigma_deg = fwhm_arcsec / math.sqrt(8.0 * math.log(2.0)) / 3600.0
    margin_deg = float(margin_sigma) * sigma_deg
    return (
        x_min - margin_deg,
        x_max + margin_deg,
        y_min - margin_deg,
        y_max + margin_deg,
    )


def _grid_target_sampling_diagnostics(
    model: ExplicitSpectralGridField | LinearBandGridField,
    target_header: dict[str, Any],
) -> dict[str, Any]:
    world = _target_boundary_world(target_header).to(model.grid.device)
    wavelength = torch.full_like(world[:, 0], float(model.domain.wavelength_center))
    unit = model._coords_unit(world[:, 0], world[:, 1], wavelength)
    x_span = float(unit[:, 0].max() - unit[:, 0].min())
    y_span = float(unit[:, 1].max() - unit[:, 1].min())
    local = model._encoded_coords(world[:, 0], world[:, 1], wavelength)
    scale = float(model.domain.spatial_scale_deg)
    x_width_arcsec = float((local[:, 0].max() - local[:, 0].min()) * scale * 3600.0)
    y_width_arcsec = float((local[:, 1].max() - local[:, 1].min()) * scale * 3600.0)
    intervals_x = x_span * (model.spatial_resolution - 1)
    intervals_y = y_span * (model.spatial_resolution - 1)
    if (
        not math.isfinite(intervals_x)
        or not math.isfinite(intervals_y)
        or intervals_x <= 0.0
        or intervals_y <= 0.0
    ):
        raise ValueError("target WCS must span positive finite grid intervals")
    return {
        "grid_spatial_mapping": (
            "target_footprint_linear"
            if model.spatial_bounds_local_deg is not None
            else "legacy_tanh_domain"
        ),
        "grid_spatial_bounds_local_deg": model.spatial_bounds_local_deg,
        "grid_target_intervals_x": intervals_x,
        "grid_target_intervals_y": intervals_y,
        "grid_mean_cell_arcsec_x": x_width_arcsec / intervals_x,
        "grid_mean_cell_arcsec_y": y_width_arcsec / intervals_y,
    }


def _foam_training_tile_side(*, crop_size: int, grid_side: int) -> int:
    """Bound dense smooth memberships while increasing fixed-grid capacity."""
    crop = int(crop_size)
    side = int(grid_side)
    if crop <= 0:
        raise ValueError("crop_size must be positive")
    if side < 2:
        raise ValueError("grid_side must be at least 2")
    pair_limited = math.isqrt(FOAM_MAX_SMOOTH_PAIRS // (side * side))
    if pair_limited < 8:
        raise ValueError(
            "Foam grid exceeds the dense smooth-training capacity at the "
            f"{FOAM_MAX_SMOOTH_PAIRS:,}-pair limit"
        )
    return min(crop, 128, pair_limited)


def _set_foam_geometry_trainable(model: torch.nn.Module, trainable: bool) -> None:
    """Select fixed-grid or adaptive-geometry fitting without freezing appearance."""
    foam = getattr(model, "foam", model)
    sites = getattr(foam, "sites", None)
    raw_radii = getattr(foam, "raw_radii", None)
    if not isinstance(sites, torch.nn.Parameter) or not isinstance(
        raw_radii, torch.nn.Parameter
    ):
        raise TypeError("model must expose trainable Foam sites and raw_radii")
    sites.requires_grad_(bool(trainable))
    raw_radii.requires_grad_(bool(trainable))


def _initialize_foam_from_band_coadds(
    model: torch.nn.Module,
    band_entries: list[dict[str, Any]],
    *,
    intensity_scale: float,
    num_spectral_samples: int,
    operator_max_candidate_pairs: int = 10_000_000,
    whiten_spectral_basis: bool = False,
) -> dict[str, float]:
    """Warm-start fixed Foam appearance from exact observed pixel footprints."""
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise TypeError("model must have parameters")
    # TORCHSKY_DEVICE=cpu opts out of CUDA entirely -- e.g. on MIG slices
    # where the caching allocator's NVML queries can assert (torch 2.10).
    requested_device = os.environ.get("TORCHSKY_DEVICE", "").strip().lower()
    if requested_device:
        initialization_device = torch.device(requested_device)
    else:
        initialization_device = torch.device(
            "cuda" if torch.cuda.is_available() else first_parameter.device
        )
    model.to(initialization_device)
    sites = getattr(model, "sites", None)
    coefficients = getattr(model, "cell_coefficients", None)
    gradient_coefficients = getattr(model, "cell_gradient_coefficients", None)
    spectral_basis = getattr(model, "spectral_basis", None)
    normalized_spatial_coordinates = getattr(
        model,
        "normalized_spatial_coordinates",
        None,
    )
    exact_operator = getattr(model, "exact_operator", None)
    if (
        not isinstance(sites, torch.Tensor)
        or not isinstance(coefficients, torch.nn.Parameter)
        or not callable(spectral_basis)
        or not callable(normalized_spatial_coordinates)
        or not callable(exact_operator)
    ):
        raise TypeError(
            "model must expose Foam sites, coefficients, basis, and operator"
        )
    scale = float(intensity_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("intensity_scale must be finite and positive")
    if not band_entries:
        raise ValueError("band_entries must not be empty")
    candidate_limit = int(operator_max_candidate_pairs)
    if candidate_limit <= 0:
        raise ValueError("operator_max_candidate_pairs must be positive")

    cell_values: list[torch.Tensor] = []
    cell_weights: list[torch.Tensor] = []
    response_basis: list[torch.Tensor] = []
    pixel_systems: list[
        tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor | None]
    ] = []
    for entry in band_entries:
        image = torch.as_tensor(
            entry["coadd_pred"],
            dtype=sites.dtype,
            device=sites.device,
        )
        support = torch.as_tensor(
            entry["support_mask"],
            dtype=torch.bool,
            device=sites.device,
        )
        if image.ndim != 2 or support.shape != image.shape:
            raise ValueError("coadd image and support mask must share 2-D shape")
        valid = support & torch.isfinite(image)
        height, width = image.shape
        grid_device = (
            torch.device("cpu") if sites.device.type == "mps" else sites.device
        )
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float64, device=grid_device),
            torch.arange(width, dtype=torch.float64, device=grid_device),
            indexing="ij",
        )
        offsets = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=torch.float64,
            device=grid_device,
        )
        wcs = torchsky.WCS(dict(entry["target_header"]))
        wcs.to(grid_device)
        ra_deg, dec_deg = wcs.pixel_to_world(
            x[..., None] + offsets[:, 0],
            y[..., None] + offsets[:, 1],
        )
        polygons = normalized_spatial_coordinates(
            ra_deg.to(device=sites.device),
            dec_deg.to(device=sites.device),
        ).reshape(height * width, 4, 2)
        operator = exact_operator(
            polygons,
            max_candidate_pairs=candidate_limit,
        )
        next_vertices = torch.roll(polygons, shifts=-1, dims=-2)
        pixel_areas = 0.5 * torch.abs(
            torch.sum(
                polygons[..., 0] * next_vertices[..., 1]
                - polygons[..., 1] * next_vertices[..., 0],
                dim=-1,
            )
        )
        if bool(torch.any(pixel_areas <= 0)):
            raise ValueError("target WCS produced a degenerate pixel polygon")
        normalized = torch.where(
            valid,
            image / scale,
            torch.zeros_like(image),
        ).reshape(-1)
        pixel_weight = valid.to(dtype=sites.dtype).reshape(-1)
        integrated = normalized * pixel_areas
        psf_spec = entry["target_observation"].psf_spec
        if getattr(psf_spec, "mode", "none") not in (
            "none",
            "gaussian_sigma_pix",
            "learned_lowrank",
        ):
            raise ValueError("Foam initialization requires a symmetric target PSF")

        def psf_forward(
            values: torch.Tensor,
            areas: torch.Tensor = pixel_areas,
            spec: Any = psf_spec,
            shape: tuple[int, int] = (height, width),
        ) -> torch.Tensor:
            with torch.no_grad():
                surface_brightness = values / areas
                return (
                    apply_psf_spec(
                        surface_brightness.reshape(shape),
                        spec,
                    ).reshape(values.shape)
                    * areas
                )

        def psf_adjoint(
            values: torch.Tensor,
            areas: torch.Tensor = pixel_areas,
            spec: Any = psf_spec,
            shape: tuple[int, int] = (height, width),
        ) -> torch.Tensor:
            with torch.no_grad():
                if getattr(spec, "mode", "none") == "learned_lowrank":
                    core = apply_learned_psf_spec_adjoint(
                        (values * areas).reshape(shape),
                        spec,
                    )
                else:
                    core = apply_psf_spec(
                        (values * areas).reshape(shape),
                        spec,
                    )
                return core.reshape(values.shape) / areas

        solved_gradients = None
        if isinstance(gradient_coefficients, torch.nn.Parameter):
            solved_cells, solved_gradients = solve_power_pixel_affine_least_squares(
                operator,
                integrated,
                pixel_weight,
                ridge=1.0e-6,
                max_iterations=50,
                tolerance=1.0e-5,
                pixel_matvec=psf_forward,
                pixel_rmatvec=psf_adjoint,
            )
        else:
            solved_cells = solve_power_pixel_least_squares(
                operator,
                integrated,
                pixel_weight,
                ridge=1.0e-6,
                max_iterations=50,
                tolerance=1.0e-5,
                pixel_matvec=psf_forward,
                pixel_rmatvec=psf_adjoint,
            )
        coverage = torch.clamp(
            operator.rmatvec(pixel_weight, semantics="integrated_flux"),
            min=0.0,
            max=1.0,
        )
        cell_values.append(solved_cells)
        cell_weights.append(coverage)
        pixel_systems.append(
            (
                operator,
                pixel_areas,
                normalized,
                pixel_weight,
                psf_forward,
                solved_gradients,
            )
        )

        wavelengths, response = bandpass_quadrature(
            entry["target_observation"],
            num_spectral_samples=int(num_spectral_samples),
            device=sites.device,
        )
        basis_samples = spectral_basis(wavelengths).to(dtype=sites.dtype)
        response_basis.append(
            torch.sum(response.to(dtype=sites.dtype)[:, None] * basis_samples, dim=0)
        )

    values = torch.stack(cell_values)
    weights = torch.stack(cell_weights)
    basis = torch.stack(response_basis)
    raw_basis_condition = torch.linalg.cond(basis)
    raw_basis_rank = torch.linalg.matrix_rank(basis)
    if whiten_spectral_basis:
        set_basis_transform = getattr(model, "set_spectral_basis_transform", None)
        if not callable(set_basis_transform):
            raise TypeError("model must expose set_spectral_basis_transform")
        transform = spectral_basis_whitening_transform(basis)
        set_basis_transform(transform)
        basis = basis @ transform
    solved = solve_spectral_cell_coefficients(basis, values, weights)
    with torch.no_grad():
        coefficients.copy_(solved)
    solved_gradient_coefficients = None
    if isinstance(gradient_coefficients, torch.nn.Parameter):
        band_gradients = torch.stack(
            [
                gradient
                for _, _, _, _, _, gradient in pixel_systems
                if gradient is not None
            ]
        )
        solved_gradient_coefficients = torch.stack(
            (
                solve_spectral_cell_coefficients(
                    basis, band_gradients[..., 0], weights
                ),
                solve_spectral_cell_coefficients(
                    basis, band_gradients[..., 1], weights
                ),
            ),
            dim=-1,
        )
        with torch.no_grad():
            gradient_coefficients.copy_(solved_gradient_coefficients)
    reconstructed = basis @ solved.mT
    reconstructed_gradients = (
        torch.einsum("bk,ckd->bcd", basis, solved_gradient_coefficients)
        if solved_gradient_coefficients is not None
        else None
    )
    initial_cell_variations: list[torch.Tensor] = []
    residual_square_sum = sites.new_zeros(())
    observed_count = sites.new_zeros(())
    for band_index, (band_cells, pixel_system) in enumerate(
        zip(reconstructed, pixel_systems, strict=True)
    ):
        operator, pixel_areas, normalized, pixel_weight, psf_forward, _ = pixel_system
        integrated_prediction = (
            operator.matvec_affine(
                band_cells,
                reconstructed_gradients[band_index],
            )
            if reconstructed_gradients is not None
            else operator.matvec(band_cells)
        )
        if reconstructed_gradients is not None:
            observed_cells = weights[band_index] > 0
            cell_variation = torch.sqrt(operator.cell_areas) * torch.linalg.vector_norm(
                reconstructed_gradients[band_index],
                dim=-1,
            )
            observed_variation = cell_variation[observed_cells]
            if observed_variation.numel():
                initial_cell_variations.append(observed_variation)
        prediction = psf_forward(integrated_prediction) / pixel_areas
        residual_square_sum += torch.sum(
            pixel_weight * torch.square(prediction - normalized)
        )
        observed_count += torch.sum(pixel_weight)
    residual_rmse = torch.sqrt(
        residual_square_sum / torch.clamp(observed_count, min=1.0)
    )
    coverage_count = torch.sum(weights > 0, dim=0)
    variation = (
        torch.cat(initial_cell_variations)
        if initial_cell_variations
        else sites.new_zeros((1,))
    )
    return {
        "foam_initialized_cell_fraction": float(
            torch.mean((coverage_count > 0).float())
        ),
        "foam_fully_covered_cell_fraction": float(
            torch.mean((coverage_count == len(band_entries)).float())
        ),
        "foam_initialization_rmse": float(residual_rmse),
        "foam_initial_coefficient_abs_max": float(torch.max(torch.abs(solved))),
        "foam_initial_gradient_coefficient_abs_max": (
            float(torch.max(torch.abs(solved_gradient_coefficients)))
            if solved_gradient_coefficients is not None
            else 0.0
        ),
        "foam_spectral_basis_condition_number": float(torch.linalg.cond(basis)),
        "foam_spectral_basis_raw_condition_number": float(raw_basis_condition),
        "foam_spectral_basis_rank": float(raw_basis_rank),
        "foam_initial_cell_variation_abs_p95": float(torch.quantile(variation, 0.95)),
        "foam_initial_cell_variation_abs_max": float(torch.max(variation)),
    }


def gpu_mem_note(label: str) -> None:
    """Log CUDA allocator residency when TORCHSKY_GPU_MEM_LOG=1."""
    if os.environ.get("TORCHSKY_GPU_MEM_LOG") != "1" or not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(
        f"[gpu-mem] {label}: allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB",
        flush=True,
    )


def _refine_foam_from_observations_exact(
    model: torch.nn.Module,
    observations: list[FieldObservation],
    *,
    num_spectral_samples: int,
    operator_max_candidate_pairs: int = 10_000_000,
    max_iterations: int = 50,
    operator_storage_device: str = "model",
    observation_batch_size: int | str | None = None,
    preconditioner: str = "auto",
) -> dict[str, float]:
    """Refit fixed spectral-affine appearance through native detector pixels."""
    coefficients = getattr(model, "cell_coefficients", None)
    gradient_coefficients = getattr(model, "cell_gradient_coefficients", None)
    spectral_basis = getattr(model, "spectral_basis", None)
    normalized_coordinates = getattr(model, "normalized_spatial_coordinates", None)
    exact_operator = getattr(model, "exact_operator", None)
    if (
        not isinstance(coefficients, torch.nn.Parameter)
        or not isinstance(gradient_coefficients, torch.nn.Parameter)
        or not callable(spectral_basis)
        or not callable(normalized_coordinates)
        or not callable(exact_operator)
    ):
        raise TypeError("exact observation refinement requires affine spectral Foam")
    if not observations:
        raise ValueError("observations must not be empty")
    device = coefficients.device

    gpu_mem_note("refit-entry")
    dtype = coefficients.dtype
    storage_device = (
        device
        if str(operator_storage_device).strip().lower() == "model"
        else torch.device(operator_storage_device)
    )
    operators = []
    response_basis = []
    integrated_values = []
    integrated_weights = []
    pixel_areas_by_observation = []
    forwards = []
    adjoints = []
    observed_pixel_count = 0
    for observation_index, observation in enumerate(observations):
        image = observation.image.to(device=device, dtype=dtype)
        height, width = image.shape
        grid_device = torch.device("cpu") if device.type == "mps" else device
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float64, device=grid_device),
            torch.arange(width, dtype=torch.float64, device=grid_device),
            indexing="ij",
        )
        offsets = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=torch.float64,
            device=grid_device,
        )
        wcs = torchsky.WCS(dict(observation.header))
        wcs.to(grid_device)
        ra_deg, dec_deg = wcs.pixel_to_world(
            x[..., None] + offsets[:, 0],
            y[..., None] + offsets[:, 1],
        )
        polygons = normalized_coordinates(
            ra_deg.to(device=device), dec_deg.to(device=device)
        ).reshape(height * width, 4, 2)
        operator = exact_operator(
            polygons,
            max_candidate_pairs=int(operator_max_candidate_pairs),
        )
        next_vertices = torch.roll(polygons, shifts=-1, dims=-2)
        pixel_areas = 0.5 * torch.abs(
            torch.sum(
                polygons[..., 0] * next_vertices[..., 1]
                - polygons[..., 1] * next_vertices[..., 0],
                dim=-1,
            )
        )
        if bool(torch.any(pixel_areas <= 0)):
            raise ValueError("observation WCS produced a degenerate pixel polygon")
        mask = (
            observation.mask.to(device=device, dtype=torch.bool)
            if observation.mask is not None
            else torch.ones_like(image, dtype=torch.bool)
        )
        surface_weight = (
            observation.weight.to(device=device, dtype=dtype)
            if observation.weight is not None
            else torch.ones_like(image)
        )
        valid = (
            mask
            & torch.isfinite(image)
            & torch.isfinite(surface_weight)
            & (surface_weight > 0)
        ).reshape(-1)
        area = pixel_areas.reshape(-1)
        stored_area = area.to(storage_device)
        integrated_values.append(
            torch.where(valid, image.reshape(-1) * area, torch.zeros_like(area)).to(
                storage_device
            )
        )
        integrated_weights.append(
            torch.where(
                valid,
                surface_weight.reshape(-1) / area.square(),
                torch.zeros_like(area),
            ).to(storage_device)
        )
        pixel_areas_by_observation.append(stored_area)
        observed_pixel_count += int(torch.sum(valid))
        if observation_index % 25 == 0:
            gpu_mem_note(f"refit-obs-{observation_index}")
        psf_spec = observation.psf_spec
        if getattr(psf_spec, "mode", "none") not in (
            "none",
            "gaussian_sigma_pix",
            "learned_lowrank",
        ):
            raise ValueError("exact observation refinement requires a symmetric PSF")

        def psf_forward(
            values: torch.Tensor,
            areas: torch.Tensor = stored_area,
            spec: Any = psf_spec,
            shape: tuple[int, int] = (height, width),
        ) -> torch.Tensor:
            areas = areas.to(device=values.device)
            with torch.no_grad():
                return (
                    apply_psf_spec((values / areas).reshape(shape), spec).reshape(-1)
                    * areas
                )

        def psf_adjoint(
            values: torch.Tensor,
            areas: torch.Tensor = stored_area,
            spec: Any = psf_spec,
            shape: tuple[int, int] = (height, width),
        ) -> torch.Tensor:
            areas = areas.to(device=values.device)
            with torch.no_grad():
                if getattr(spec, "mode", "none") == "learned_lowrank":
                    core = apply_learned_psf_spec_adjoint(
                        (values * areas).reshape(shape), spec
                    )
                else:
                    core = apply_psf_spec((values * areas).reshape(shape), spec)
                return core.reshape(-1) / areas

        wavelengths, response = bandpass_quadrature(
            observation,
            num_spectral_samples=int(num_spectral_samples),
            device=device,
        )
        basis_samples = spectral_basis(wavelengths).to(dtype=dtype)
        response_basis.append(
            torch.sum(response.to(dtype=dtype)[:, None] * basis_samples, dim=0).to(
                storage_device
            )
        )
        operators.append(operator.to(storage_device))
        forwards.append(psf_forward)
        adjoints.append(psf_adjoint)
    gpu_mem_note("refit-preloop-done")

    initial_coefficients = coefficients.detach().clone()
    initial_gradients = gradient_coefficients.detach().clone()
    solve_diagnostics: dict[str, float] = {}
    solved, solved_gradients = solve_spectral_power_pixel_affine_least_squares(
        operators,
        torch.stack(response_basis),
        integrated_values,
        integrated_weights,
        ridge=1.0e-6,
        max_iterations=int(max_iterations),
        tolerance=1.0e-5,
        pixel_matvecs=forwards,
        pixel_rmatvecs=adjoints,
        initial_coefficients=initial_coefficients,
        initial_gradient_coefficients=initial_gradients,
        solve_device=device,
        observation_batch_size=observation_batch_size,
        preconditioner=preconditioner,
        diagnostics=solve_diagnostics,
    )
    gpu_mem_note("refit-solve-done")
    with torch.no_grad():
        coefficients.copy_(solved)
        gradient_coefficients.copy_(solved_gradients)

    weighted_square_sum = coefficients.new_zeros(())
    initial_weighted_square_sum = coefficients.new_zeros(())
    weight_sum = coefficients.new_zeros(())
    for operator, basis, values, weight, area, forward in zip(
        operators,
        response_basis,
        integrated_values,
        integrated_weights,
        pixel_areas_by_observation,
        forwards,
        strict=True,
    ):
        solve_operator = operator.to(device)
        solve_basis = basis.to(device)
        solve_area = area.to(device)
        solve_values = values.to(device)
        solve_weight = weight.to(device)
        prediction = forward(
            solve_operator.matvec_affine(
                solved @ solve_basis,
                torch.einsum("ckd,k->cd", solved_gradients, solve_basis),
            )
        )
        initial_prediction = forward(
            solve_operator.matvec_affine(
                initial_coefficients @ solve_basis,
                torch.einsum("ckd,k->cd", initial_gradients, solve_basis),
            )
        )
        surface_weight = solve_weight * solve_area.square()
        weighted_square_sum += torch.sum(
            surface_weight * torch.square((prediction - solve_values) / solve_area)
        )
        initial_weighted_square_sum += torch.sum(
            surface_weight
            * torch.square((initial_prediction - solve_values) / solve_area)
        )
        weight_sum += torch.sum(surface_weight)
    return {
        "foam_exact_refinement_initial_rmse": float(
            torch.sqrt(initial_weighted_square_sum / torch.clamp(weight_sum, min=1.0))
        ),
        "foam_exact_refinement_rmse": float(
            torch.sqrt(weighted_square_sum / torch.clamp(weight_sum, min=1.0))
        ),
        "foam_exact_refinement_observations": float(len(observations)),
        "foam_exact_refinement_pixels": float(observed_pixel_count),
        "foam_exact_refinement_max_iterations": float(max_iterations),
        "foam_exact_refinement_iterations": solve_diagnostics["iterations"],
        "foam_exact_refinement_normal_operator_applications": solve_diagnostics[
            "normal_operator_applications"
        ],
        "foam_exact_refinement_initial_relative_residual": solve_diagnostics[
            "initial_relative_residual"
        ],
        "foam_exact_refinement_ridge_centered_on_initial": solve_diagnostics[
            "ridge_centered_on_initial"
        ],
        "foam_exact_refinement_relative_residual": solve_diagnostics[
            "relative_residual"
        ],
        "foam_exact_refinement_converged": solve_diagnostics["converged"],
        "foam_exact_refinement_preconditioner_setup_seconds": solve_diagnostics[
            "preconditioner_setup_seconds"
        ],
        "foam_exact_refinement_rhs_seconds": solve_diagnostics["rhs_seconds"],
        "foam_exact_refinement_initial_residual_seconds": solve_diagnostics[
            "initial_residual_seconds"
        ],
        "foam_exact_refinement_pcg_seconds": solve_diagnostics["pcg_seconds"],
        "foam_exact_refinement_preconditioner_cell_block": solve_diagnostics[
            "preconditioner_cell_block"
        ],
        "foam_exact_refinement_preconditioner_block_fallback_cells": (
            solve_diagnostics["preconditioner_block_fallback_cells"]
        ),
        "foam_exact_refinement_preconditioner_storage_bytes": solve_diagnostics[
            "preconditioner_storage_bytes"
        ],
        "foam_exact_refinement_preconditioner_auto": solve_diagnostics[
            "preconditioner_auto"
        ],
        "foam_exact_refinement_preconditioner_response_condition": (
            solve_diagnostics["preconditioner_response_condition"]
        ),
        "foam_exact_refinement_preconditioner_auto_condition_threshold": (
            solve_diagnostics["preconditioner_auto_condition_threshold"]
        ),
        "foam_exact_refinement_preconditioner_auto_budget_bytes": solve_diagnostics[
            "preconditioner_auto_budget_bytes"
        ],
        "foam_exact_refinement_preconditioner_block_construction_peak_bytes": (
            solve_diagnostics["preconditioner_block_construction_peak_bytes"]
        ),
        "foam_exact_refinement_observation_batch_size": float(
            solve_diagnostics["observation_batch_size"]
        ),
        "foam_exact_refinement_observation_batch_auto": solve_diagnostics[
            "observation_batch_auto"
        ],
        "foam_exact_refinement_observation_batch_transfer_budget_bytes": (
            solve_diagnostics["observation_batch_transfer_budget_bytes"]
        ),
        "foam_exact_refinement_max_observation_transfer_bytes": (
            solve_diagnostics["max_observation_transfer_bytes"]
        ),
        "foam_exact_refinement_operator_storage_bytes": float(
            solve_diagnostics["operator_storage_bytes"]
        ),
        "foam_exact_refinement_peak_operator_batch_bytes": solve_diagnostics[
            "peak_operator_batch_bytes"
        ],
        "foam_exact_refinement_host_offload": float(storage_device.type == "cpu"),
    }


def _preview_uint8(image: np.ndarray) -> np.ndarray:
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros(image.shape, dtype=np.uint8)
    values = image[finite].astype(np.float64, copy=False)
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    clipped = np.clip((image.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
    clipped = np.where(finite, clipped, 0.0)
    return np.rint(255.0 * clipped).astype(np.uint8)


def _write_export_products(
    export_dir: Path,
    *,
    band: str,
    model: str,
    psf_mode: str = "fitted_gaussian",
    header: dict[str, Any],
    reference: np.ndarray,
    prediction: np.ndarray,
    support_mask: np.ndarray,
    source_mask: np.ndarray,
) -> None:
    model_dir = export_dir / _safe_slug(psf_mode) / _safe_slug(band) / _safe_slug(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    residual = prediction - reference
    torchfits.write(
        str(model_dir / "reference.fits"),
        torch.from_numpy(reference.astype(np.float32, copy=False)),
        header=dict(header),
        overwrite=True,
    )
    torchfits.write(
        str(model_dir / "prediction.fits"),
        torch.from_numpy(prediction.astype(np.float32, copy=False)),
        header=dict(header),
        overwrite=True,
    )
    torchfits.write(
        str(model_dir / "residual.fits"),
        torch.from_numpy(residual.astype(np.float32, copy=False)),
        header=dict(header),
        overwrite=True,
    )
    torchfits.write(
        str(model_dir / "support_mask.fits"),
        torch.from_numpy(support_mask.astype(np.float32, copy=False)),
        header=dict(header),
        overwrite=True,
    )
    torchfits.write(
        str(model_dir / "source_mask.fits"),
        torch.from_numpy(source_mask.astype(np.float32, copy=False)),
        header=dict(header),
        overwrite=True,
    )
    Image.fromarray(_preview_uint8(reference), mode="L").save(
        model_dir / "reference.png"
    )
    Image.fromarray(_preview_uint8(prediction), mode="L").save(
        model_dir / "prediction.png"
    )
    Image.fromarray(_preview_uint8(residual), mode="L").save(model_dir / "residual.png")
    Image.fromarray(support_mask.astype(np.uint8) * 255, mode="L").save(
        model_dir / "support_mask.png"
    )
    Image.fromarray(source_mask.astype(np.uint8) * 255, mode="L").save(
        model_dir / "source_mask.png"
    )


def _source_metric_mask(
    pred: np.ndarray,
    reference: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    base = _metric_mask(pred, reference, support)
    if not np.any(base):
        return base
    ref_values = reference[base].astype(np.float64, copy=False)
    median = float(np.median(ref_values))
    mad = float(np.median(np.abs(ref_values - median)))
    robust_sigma = 1.4826 * mad
    percentile_threshold = float(np.percentile(ref_values, 90.0))
    if not np.isfinite(robust_sigma) or robust_sigma <= 0.0:
        threshold = percentile_threshold
    else:
        threshold = min(median + 3.0 * robust_sigma, percentile_threshold)
    source = base & (reference >= threshold)
    if np.any(source):
        return source
    source = base & (reference >= float(np.percentile(ref_values, 80.0)))
    if np.any(source):
        return source
    return base


def _background_subtracted_source_stats(
    pred: np.ndarray,
    reference: np.ndarray,
    support: np.ndarray,
) -> dict[str, float]:
    base = _metric_mask(pred, reference, support)
    if not np.any(base):
        return {
            "source_rmse": float("nan"),
            "source_contrast_rmse": float("nan"),
            "source_contrast_slope": float("nan"),
            "source_flux_ratio": float("nan"),
            "source_frac": 0.0,
        }
    source = _source_metric_mask(pred, reference, support)
    pred64 = pred.astype(np.float64, copy=False)
    ref64 = reference.astype(np.float64, copy=False)
    source_pred = pred64[source]
    source_ref = ref64[source]
    source_diff = source_pred - source_ref
    background = base & ~source
    if np.any(background):
        pred_bg = float(np.median(pred64[background]))
        ref_bg = float(np.median(ref64[background]))
    else:
        pred_bg = float(np.median(pred64[base]))
        ref_bg = float(np.median(ref64[base]))
    source_pred_contrast = source_pred - pred_bg
    source_ref_contrast = source_ref - ref_bg
    contrast_diff = source_pred_contrast - source_ref_contrast
    if source_ref_contrast.size:
        design = source_ref_contrast.reshape(-1, 1)
        slope_arr, *_ = np.linalg.lstsq(design, source_pred_contrast, rcond=None)
        contrast_slope = float(slope_arr[0])
    else:
        contrast_slope = float("nan")
    source_flux_ratio = (
        float(np.sum(source_pred_contrast) / np.sum(source_ref_contrast))
        if source_ref_contrast.size and np.sum(source_ref_contrast) != 0.0
        else float("nan")
    )
    return {
        "source_rmse": float(np.sqrt(np.mean(source_diff * source_diff)))
        if source_diff.size
        else float("nan"),
        "source_contrast_rmse": float(np.sqrt(np.mean(contrast_diff * contrast_diff)))
        if contrast_diff.size
        else float("nan"),
        "source_contrast_slope": contrast_slope,
        "source_flux_ratio": source_flux_ratio,
        "source_frac": float(source.mean()),
    }


def _high_frequency_laplacian_rms(
    image: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Estimate pixel-scale noise RMS via the 5-point discrete Laplacian.

    For white noise with pixel standard deviation sigma, the 5-point
    Laplacian (4*c - n - s - w - e) has variance 20 * sigma^2 and vanishes
    on any affine plane.
    """
    if image.ndim != 2 or mask.shape != image.shape:
        raise ValueError("image and mask must share a 2-D shape")
    if image.shape[0] < 3 or image.shape[1] < 3:
        return float("nan")
    valid = mask & np.isfinite(image)
    interior = (
        valid[1:-1, 1:-1]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
    )
    if not np.any(interior):
        return float("nan")
    img64 = image.astype(np.float64, copy=False)
    laplacian = (
        4.0 * img64[1:-1, 1:-1]
        - img64[:-2, 1:-1]
        - img64[2:, 1:-1]
        - img64[1:-1, :-2]
        - img64[1:-1, 2:]
    )
    values = laplacian[interior]
    return float(np.sqrt(np.mean(values * values) / 20.0))


def _surface_brightness_noise_diagnostics(
    pred: np.ndarray,
    coadd_pred: np.ndarray,
    reference: np.ndarray,
    support: np.ndarray,
) -> dict[str, float]:
    """Quantify negative-value support and pixel-scale noise amplification vs coadd."""
    base = _metric_mask(pred, reference, support)
    if not np.any(base):
        return {
            "support_negative_fraction": float("nan"),
            "background_negative_fraction": float("nan"),
            "background_median": float("nan"),
            "background_rms": float("nan"),
            "background_mad_sigma": float("nan"),
            "background_high_freq_rms": float("nan"),
            "coadd_background_high_freq_rms": float("nan"),
            "background_high_freq_ratio_vs_coadd": float("nan"),
        }
    source = _source_metric_mask(pred, reference, support)
    background = base & ~source
    eval_mask = background if np.any(background) else base
    pred64 = pred.astype(np.float64, copy=False)
    support_vals = pred64[base]
    bg_vals = pred64[eval_mask]
    bg_median = float(np.median(bg_vals))
    bg_mad_sigma = float(1.4826 * np.median(np.abs(bg_vals - bg_median)))
    bg_rms = float(np.sqrt(np.mean((bg_vals - bg_median) ** 2)))
    bg_hf_rms = _high_frequency_laplacian_rms(pred64, eval_mask)
    coadd_hf_rms = _high_frequency_laplacian_rms(
        coadd_pred.astype(np.float64, copy=False), eval_mask
    )
    hf_ratio = (
        float(bg_hf_rms / coadd_hf_rms)
        if math.isfinite(bg_hf_rms)
        and math.isfinite(coadd_hf_rms)
        and coadd_hf_rms > 0.0
        else float("nan")
    )
    return {
        "support_negative_fraction": float(np.mean(support_vals < 0.0)),
        "background_negative_fraction": float(np.mean(bg_vals < 0.0)),
        "background_median": bg_median,
        "background_rms": bg_rms,
        "background_mad_sigma": bg_mad_sigma,
        "background_high_freq_rms": bg_hf_rms,
        "coadd_background_high_freq_rms": coadd_hf_rms,
        "background_high_freq_ratio_vs_coadd": hf_ratio,
    }


def _joint_sasnet_kwargs(observations: list[FieldObservation]) -> dict[str, Any]:
    """Resolve the experimental frequency cap from actual training pixels.

    The area-equivalent pixel scale and fitted Gaussian FWHM cap frequencies.
    Frozen PSFs remain in the observation operator.
    """
    if not observations:
        raise ValueError("SASNet requires training observations")
    scales = []
    psf_supports = []
    for observation in observations:
        height, width = observation.image.shape[-2:]
        area = pixel_area_sr_at(
            torchsky.WCS(observation.header),
            torch.tensor((width - 1) / 2, dtype=torch.float64),
            torch.tensor((height - 1) / 2, dtype=torch.float64),
        )
        scale = float(torch.sqrt(area) * (180.0 / torch.pi))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("SASNet requires finite positive training pixel areas")
        scales.append(scale)
        sigma_pix = float(observation.psf_sigma_pix or 0.0)
        if math.isfinite(sigma_pix) and sigma_pix > 0.0:
            psf_supports.append(math.sqrt(8.0 * math.log(2.0)) * sigma_pix * scale)
    return {
        "pixel_footprint_deg": min(scales),
        "psf_support_deg": min(psf_supports) if psf_supports else None,
    }


def _joint_real_fit_kwargs(
    model_kind: str,
    *,
    crop_size: int,
    steps: int,
    num_observations: int,
    foam_grid_side: int = 16,
    foam_lr: float = 2.0e-2,
    foam_fixed_geometry: bool = False,
    optimizer_lr: float | None = None,
    full_observation_training: bool = False,
) -> dict[str, Any]:
    if optimizer_lr is not None and (
        not math.isfinite(optimizer_lr) or optimizer_lr <= 0
    ):
        raise ValueError("optimizer_lr must be finite and positive")
    fit_lookup_kind = (
        "explicit_spectral_grid_field"
        if model_kind == "linear_band_grid_field"
        else model_kind
    )
    kwargs = _field_fit_kwargs_for_model(
        fit_lookup_kind,
        case=REAL_JOINT_CASE,
        size=int(crop_size),
        benchmark_steps=int(steps),
    )
    if model_kind == "coadd_bilinear":
        return kwargs
    if model_kind == "spectral_power_foam":
        foam_learning_rate = float(foam_lr)
        if not math.isfinite(foam_learning_rate) or foam_learning_rate <= 0.0:
            raise ValueError("foam_lr must be finite and positive")
        tile_side = (
            min(int(crop_size), 128)
            if foam_fixed_geometry
            else _foam_training_tile_side(
                crop_size=int(crop_size),
                grid_side=int(foam_grid_side),
            )
        )
        kwargs["lr"] = foam_learning_rate
        kwargs["curriculum"] = "feature_scale_ramp"
        kwargs["num_spectral_samples"] = 5
        kwargs["training_spectral_sampling"] = "deterministic_quadrature"
        kwargs["spectral_anchor_observations_per_step"] = 0
        kwargs["spectral_anchor_loss_weight"] = 0.0
        kwargs["training_patch_size"] = tile_side
        kwargs["streaming_accumulation"] = True
        kwargs["diagnostic_spatial_tile_hw"] = (min(tile_side, 64),) * 2
    kwargs["render_spatial_tile_hw"] = (256, 256)
    if model_kind == "spectral_power_foam":
        # ponytail: cap point-by-cell work, not total CUDA memory. Spectral
        # samples retain multiple matrices and PSF margins add pixels; a sparse
        # cell index is the upgrade for much larger grids.
        render_side = min(
            tile_side, 32,
            max(1, math.isqrt(1024 * 1024 // (int(foam_grid_side) ** 2))),
        )
        kwargs["render_spatial_tile_hw"] = (render_side, render_side)
        kwargs["diagnostic_spatial_tile_hw"] = (render_side, render_side)
    kwargs["observations_per_step"] = min(max(num_observations, 1), 12)
    kwargs["observation_sampling"] = "band_balanced"
    if model_kind != "spectral_power_foam":
        kwargs["training_patch_size"] = None
    kwargs["pixel_samples_per_observation"] = max(
        256, min(int(crop_size) * int(crop_size), 1024)
    )
    if model_kind == "spectral_power_foam":
        kwargs["pixel_samples_per_observation"] = None
        kwargs["shared_sky_patch_sampling"] = True
    kwargs["loss_kind"] = "mse"
    kwargs["loss_reduction"] = "global"
    kwargs["spatial_sampling"] = "gauss_legendre_2x2"
    kwargs["grad_clip_norm"] = 1.0
    if model_kind != "spectral_power_foam":
        kwargs["lr"] = min(float(kwargs.get("lr", 1.0e-3)), 1.0e-3)
    kwargs["lr_warmup_steps"] = max(int(kwargs.get("lr_warmup_steps", 0)), 4)
    kwargs["context_observations_per_step"] = min(
        int(kwargs.get("context_observations_per_step", 0)),
        1,
    )
    kwargs["full_observations_per_step"] = 0
    if model_kind in {"spectral_basis_field", "sky_encoder_resmlp", "sky_encoder_mlp"}:
        kwargs["num_spectral_samples"] = min(
            int(kwargs.get("num_spectral_samples", 9)), 5
        )
    if model_kind == "sky_lambda_feature_field":
        kwargs["num_spectral_samples"] = min(
            int(kwargs.get("num_spectral_samples", 9)), 5
        )
        kwargs["spectral_anchor_observations_per_step"] = min(
            int(kwargs.get("spectral_anchor_observations_per_step", 0)),
            1,
        )
    if model_kind == "operator_constrained_feature_field":
        kwargs["num_spectral_samples"] = min(
            int(kwargs.get("num_spectral_samples", 7)), 5
        )
        kwargs["context_loss_weight"] = min(
            float(kwargs.get("context_loss_weight", 0.0)), 0.05
        )
    if model_kind == "spectral_basis_field":
        kwargs["lr"] = min(float(kwargs.get("lr", 1.0e-3)), 5.0e-4)
    if model_kind != "spectral_power_foam":
        # Pixel samples cannot represent the PSF convolution or pixel quadrature.
        kwargs.update(
            pixel_samples_per_observation=None,
            streaming_accumulation=True,
            context_observations_per_step=0,
            spectral_anchor_observations_per_step=0,
            gradient_residual_weight=0.0,
        )
    if model_kind in {
        "sasnet",
        "flair_bla",
        "sine_implicit",
        "explicit_spectral_grid_field",
        "linear_band_grid_field",
    }:
        side = min(int(crop_size), 32)
        kwargs.update(
            training_patch_size=side,
            shared_sky_patch_sampling=True,
            render_spatial_tile_hw=(side, side),
            diagnostic_spatial_tile_hw=(side, side),
            num_spectral_samples=1 if model_kind == "linear_band_grid_field" else 5,
            training_spectral_sampling="deterministic_quadrature",
            spectral_anchor_loss_weight=0.0,
            context_loss_weight=0.0,
        )
    if optimizer_lr is not None:
        kwargs["lr"] = float(optimizer_lr)
    if full_observation_training:
        kwargs.update(
            training_patch_size=None,
            shared_sky_patch_sampling=False,
            pixel_samples_per_observation=None,
            streaming_accumulation=True,
        )
    return kwargs


def _structured_scene_fit_kwargs(
    base: dict[str, Any], *, num_observations: int
) -> dict[str, Any]:
    """Make a full-exposure fit compatible with the structured likelihood."""
    kwargs = dict(base)
    kwargs.update(
        loss_kind="structured_nll",
        loss_reduction="global",
        training_spectral_sampling="deterministic_quadrature",
        observations_per_step=max(int(num_observations), 1),
        training_patch_size=None,
        pixel_samples_per_observation=None,
        shared_sky_patch_sampling=False,
        diagnostic_spatial_tile_hw=None,
        gradient_residual_weight=0.0,
        spectral_anchor_observations_per_step=0,
        component_anchor_terms=None,
        component_centroid_anchor_terms=None,
        context_observations_per_step=0,
    )
    return kwargs


def _cross_fitted_scene_predictions(
    observations: Sequence[FieldObservation],
    *,
    model_factory: Callable[[], torch.nn.Module],
    steps: int,
    fit_kwargs: dict[str, Any],
    render_kwargs: dict[str, Any],
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, float | int], ...]]:
    """Fit one scene per held-out exposure and render only that exposure."""
    if len(observations) < 2:
        raise ValueError("cross-fitting requires at least two exposures")
    if int(steps) < 1:
        raise ValueError("cross-fit scene steps must be positive")
    predictions: list[torch.Tensor] = []
    diagnostics: list[dict[str, float | int]] = []
    for held_out_index, held_out in enumerate(observations):
        training = tuple(
            observation
            for index, observation in enumerate(observations)
            if index != held_out_index
        )
        torch.manual_seed(int(seed) + held_out_index)
        result = fit_field_model(
            training,
            model_factory(),
            steps=int(steps),
            **fit_kwargs,
        )
        result.model.eval()
        with torch.no_grad():
            prediction = render_field_observation(
                result.model,
                held_out,
                **render_kwargs,
            )
        if prediction.shape != held_out.image.shape:
            raise RuntimeError("held-out render shape does not match exposure")
        predictions.append(prediction.detach().cpu())
        diagnostics.append(
            {
                "held_out_index": held_out_index,
                "training_exposures": len(training),
                "initial_loss": float(result.initial_loss),
                "final_loss": float(result.final_loss),
            }
        )
    return tuple(predictions), tuple(diagnostics)


def _structured_noise_bundles(
    observations: Sequence[FieldObservation],
    models: Sequence[StructuredDetectorNoise],
) -> tuple[ObservationOperatorBundle, ...]:
    if len(observations) != len(models):
        raise ValueError("noise models must match observations")
    return tuple(
        field_observation_operator_bundle(observation).with_updates(
            noise=StructuredNoiseOp(model, frozen=True)
        )
        for observation, model in zip(observations, models, strict=True)
    )


@torch.no_grad()
def _structured_noise_diagnostics(
    observations: Sequence[FieldObservation],
    predictions: Sequence[torch.Tensor],
    models: Sequence[StructuredDetectorNoise],
    losses: Sequence[float],
) -> list[dict[str, Any]]:
    if not (len(observations) == len(predictions) == len(models) == len(losses)):
        raise ValueError("noise diagnostics inputs must have equal length")
    rows: list[dict[str, Any]] = []
    for index, (observation, prediction, model, loss) in enumerate(
        zip(observations, predictions, models, losses, strict=True)
    ):
        calibration = model.variance_calibration
        row_noise = model.row_noise
        residual = prediction.to(dtype=torch.float32) - observation.image.to(
            dtype=torch.float32
        )
        support = torch.isfinite(residual)
        components: dict[str, torch.Tensor] = {}
        for key, value in (
            (VAR_POISSON, observation.var_poisson),
            (VAR_RNOISE, observation.var_rnoise),
            (VAR_FLAT, observation.var_flat),
        ):
            if value is None:
                continue
            plane = value.to(dtype=residual.dtype)
            plane_valid = torch.isfinite(plane) & (plane >= 0.0)
            support &= plane_valid
            components[key] = torch.where(plane_valid, plane, torch.zeros_like(plane))
        err = (
            None
            if observation.err is None
            else observation.err.to(dtype=residual.dtype)
        )
        if not components and err is None:
            if observation.weight is None:
                raise ValueError("noise diagnostics require variance, ERR, or weight")
            weight = observation.weight.to(dtype=residual.dtype)
            err = torch.where(
                weight > 0.0,
                weight.clamp_min(torch.finfo(weight.dtype).tiny).rsqrt(),
                torch.ones_like(weight),
            )
        elif not components and err is not None:
            err_valid = torch.isfinite(err) & (err >= 0.0)
            support &= err_valid
            err = torch.where(err_valid, err, torch.ones_like(err))
        variance = calibration(components or None, err=err).to(dtype=residual.dtype)
        if variance.shape != residual.shape:
            variance = torch.broadcast_to(variance, residual.shape)
        support &= torch.isfinite(variance) & (variance > 0)
        if observation.mask is not None:
            support &= observation.mask.to(dtype=torch.bool)
        safe_residual = torch.where(support, residual, torch.zeros_like(residual))
        amplifier_index = (
            model.default_amplifier_index(tuple(residual.shape), residual.device)
            if observation.amplifier_index is None
            else observation.amplifier_index.to(device=residual.device)
        )
        _, corrected = profile_amplifier_row_offsets(
            safe_residual,
            variance,
            amplifier_index,
            valid=support,
            num_amplifiers=model.num_amplifiers,
        )
        inlier_probability, _ = model.contamination_responsibilities(
            safe_residual,
            components=components or None,
            err=err,
            valid=support,
            amplifier_index=amplifier_index,
            contamination_prior=observation.contamination_prior,
        )
        z_abs = torch.abs(corrected) / torch.sqrt(variance)
        inlier_weight = torch.where(
            support, inlier_probability, torch.zeros_like(inlier_probability)
        )
        inlier_weight_sum = inlier_weight.sum().clamp_min(
            torch.finfo(inlier_weight.dtype).tiny
        )
        active_amplifiers = torch.unique(amplifier_index[support]).detach().cpu()
        rows.append(
            {
                "exposure_index": index,
                "crossfit_nll": float(loss),
                "poisson_scale": float(calibration.poisson_scale.detach().cpu()),
                "rnoise_scale": float(calibration.rnoise_scale.detach().cpu()),
                "flat_scale": float(calibration.flat_scale.detach().cpu()),
                "err_scale": float(calibration.err_scale.detach().cpu()),
                "variance_floor": float(calibration.variance_floor.detach().cpu()),
                "row_white_psd": row_noise.white_psd.detach().cpu().tolist(),
                "row_knee_frequency": (
                    row_noise.knee_frequency.detach().cpu().tolist()
                ),
                "row_exponent": row_noise.exponent.detach().cpu().tolist(),
                "outlier_probability": float(model.outlier_probability.detach().cpu()),
                "outlier_scale": float(model.outlier_scale.detach().cpu()),
                "active_amplifiers": active_amplifiers.tolist(),
                "valid_pixels": int(support.sum()),
                "posterior_weighted_coverage_68": float(
                    torch.sum(inlier_weight * (z_abs <= 1.0)) / inlier_weight_sum
                ),
                "posterior_weighted_coverage_95": float(
                    torch.sum(inlier_weight * (z_abs <= 1.959963984540054))
                    / inlier_weight_sum
                ),
            }
        )
    return rows


def _summarize_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "OK":
            grouped.setdefault(str(row["model"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for model, model_rows in grouped.items():
        item: dict[str, Any] = {
            "model": model,
            "bands": len(model_rows),
            "mean_support_rmse": float(
                np.mean([float(r["support_rmse"]) for r in model_rows])
            ),
            "mean_source_rmse": float(
                np.mean([float(r["source_rmse"]) for r in model_rows])
            ),
            "mean_source_contrast_rmse": float(
                np.mean([float(r["source_contrast_rmse"]) for r in model_rows])
            ),
            "mean_uniform_rmse": float(
                np.mean([float(r["uniform_rmse"]) for r in model_rows])
            ),
            "mean_aligned_rmse": float(
                np.mean([float(r["aligned_rmse"]) for r in model_rows])
            ),
            "mean_affine_slope": float(
                np.mean([float(r["affine_slope"]) for r in model_rows])
            ),
            "mean_source_contrast_slope": float(
                np.mean([float(r["source_contrast_slope"]) for r in model_rows])
            ),
            "mean_support_flux_ratio": float(
                np.mean([float(r["support_flux_ratio"]) for r in model_rows])
            ),
            "mean_source_flux_ratio": float(
                np.mean([float(r["source_flux_ratio"]) for r in model_rows])
            ),
            "total_fit_time_ms": float(
                np.max([float(r["fit_time_ms"]) for r in model_rows])
            ),
        }
        for key in (
            "support_negative_fraction",
            "background_negative_fraction",
            "background_high_freq_rms",
            "background_high_freq_ratio_vs_coadd",
        ):
            if all(key in r for r in model_rows):
                item[f"mean_{key}"] = float(
                    np.mean([float(r[key]) for r in model_rows])
                )
        summary.append(item)
    summary.sort(key=lambda row: row["mean_source_contrast_rmse"])
    return summary


def _format_markdown(
    rows: list[dict[str, Any]],
    *,
    validation_scores: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    ok_rows = [row for row in rows if row["status"] == "OK"]
    summary = _summarize_by_model(ok_rows)
    lines = [
        "# SMACS J0723 Joint Multiband Field-Fit",
        "",
        "## Prediction of exposures excluded from training",
        "",
    ]
    if validation_scores:
        lines.extend([
            "Lower negative log likelihood means better prediction on the same validation pixels. "
            "These scores use pipeline errors and assume independent pixels. "
            "The final test exposures were not evaluated. Differences alone do not establish "
            "improvement without paired uncertainty estimates and flux/color recovery checks.",
            "",
            "| Method | Band | Exposures | Pixels | Negative log likelihood per pixel | Chi-squared per pixel | Brightness units |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for method, band_scores in validation_scores.items():
            for score in band_scores:
                lines.append(
                    f"| {method} | {score['band']} | {score['exposures']} | {score['pixels']} | "
                    f"{score['gaussian_nll_per_pixel']:.6f} | {score['chi2_per_pixel']:.6f} | "
                    f"{score.get('value_unit') or 'not measured'} |"
                )
    else:
        lines.append("Validation prediction error: not measured.")
    lines.extend([
        "",
        "## Secondary reference-image comparisons",
        "",
        "| model | bands | mean source contrast rmse | mean source rmse | mean support rmse | mean aligned rmse | mean source slope | fit ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['bands']} | {row['mean_source_contrast_rmse']:.6f} | "
            f"{row['mean_source_rmse']:.6f} | {row['mean_support_rmse']:.6f} | "
            f"{row['mean_aligned_rmse']:.6f} | {row['mean_source_contrast_slope']:.4f} | "
            f"{row['total_fit_time_ms']:.2f} |"
        )
    exact_rows: dict[str, dict[str, Any]] = {}
    for row in ok_rows:
        if "foam_exact_refinement_rmse" in row:
            exact_rows.setdefault(str(row["model"]), row)
    if exact_rows:
        lines.extend(
            [
                "",
                "## Exact Refinement",
                "",
                "| model | initial rmse | final rmse | iterations | residual | setup s | rhs s | pcg s | peak operator MiB | ridge center |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for model, row in exact_rows.items():
            peak_mib = (
                float(row["foam_exact_refinement_peak_operator_batch_bytes"]) / 2**20
            )
            ridge_center = (
                "initial"
                if float(row["foam_exact_refinement_ridge_centered_on_initial"])
                else "zero"
            )
            lines.append(
                f"| {model} | {row['foam_exact_refinement_initial_rmse']:.6f} | "
                f"{row['foam_exact_refinement_rmse']:.6f} | "
                f"{row['foam_exact_refinement_iterations']:.0f} | "
                f"{row['foam_exact_refinement_relative_residual']:.3e} | "
                f"{row['foam_exact_refinement_preconditioner_setup_seconds']:.3f} | "
                f"{row['foam_exact_refinement_rhs_seconds']:.3f} | "
                f"{row['foam_exact_refinement_pcg_seconds']:.3f} | "
                f"{peak_mib:.2f} | {ridge_center} |"
            )
    lines.extend(
        [
            "",
            "## Per-Band Rows",
            "",
            "| band | model | status | n exposures | source contrast rmse | source rmse | support rmse | uniform rmse | source slope | support flux ratio | source flux ratio | support frac | source frac | fit ms |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["status"] != "OK":
            lines.append(
                f"| {row['band']} | {row['model']} | {row['status']} | {row.get('n_exposures', 0)} | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |"
            )
            continue
        lines.append(
            f"| {row['band']} | {row['model']} | OK | {row['n_exposures']} | "
            f"{row['source_contrast_rmse']:.6f} | {row['source_rmse']:.6f} | {row['support_rmse']:.6f} | {row['uniform_rmse']:.6f} | "
            f"{row['source_contrast_slope']:.4f} | {row['support_flux_ratio']:.4f} | {row['source_flux_ratio']:.4f} | "
            f"{row['support_frac']:.4f} | {row['source_frac']:.4f} | "
            f"{row['fit_time_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _parse_model_kwargs(value: str) -> dict[str, dict[str, Any]]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("model kwargs must be valid JSON") from exc
    if not isinstance(parsed, dict) or any(
        not isinstance(kwargs, dict) for kwargs in parsed.values()
    ):
        raise argparse.ArgumentTypeError("model kwargs must map model names to objects")
    return parsed


def _comparison_model_kwargs(
    model_kind: str, defaults: dict[str, Any], overrides: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {**defaults, **overrides.get(model_kind, {})}


def _report_configuration(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.nn.Module):
        return {"model_class": type(value).__name__, "source": "fitted Foam dependency"}
    if isinstance(value, dict):
        return {key: _report_configuration(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_report_configuration(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-kwargs", type=_parse_model_kwargs, default={},
        help='JSON constructor overrides by model, e.g. {"gaussian_basis":{"splat":true}}',
    )
    parser.add_argument(
        "--optimizer-lr", type=float, default=None,
        help="Explicit learning rate for all compared scenes, overriding model defaults",
    )
    parser.add_argument(
        "--inr-pcg-refine-steps", type=int, default=0,
        help="Opt-in post-Adam neural Gauss-Newton steps on training detector pixels",
    )
    parser.add_argument("--inr-pcg-iterations", type=int, default=12)
    parser.add_argument("--inr-pcg-damping", type=float, default=1.0e-3)
    parser.add_argument(
        "--full-observation-training", action="store_true",
        help="Fit every valid pixel in selected exposures with streamed PSF-aware tiles",
    )
    parser.add_argument(
        "--grid-fit-target-footprint",
        action="store_true",
        help="Map explicit grid cells linearly over the target WCS plus a PSF halo",
    )
    parser.add_argument(
        "--grid-psf-margin-sigma",
        type=float,
        default=4.0,
        help="Gaussian PSF halo around target-footprint grid bounds",
    )
    parser.add_argument("--bands", default="F090W,F277W,F356W,F444W")
    parser.add_argument("--module", default="b")
    parser.add_argument(
        "--exposure-dir",
        type=Path,
        default=DEFAULT_EXPOSURE_DIR,
        help="Directory containing native CRF/CAL products (aligned only for explicit comparator runs)",
    )
    parser.add_argument(
        "--input-product",
        choices=("auto", "crf", "cal", "aligned"),
        default="auto",
        help="Native training defaults to CRF with per-exposure CAL fallback",
    )
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument(
        "--crop-strategy", choices=("center", "best_overlap"), default="best_overlap"
    )
    parser.add_argument("--crop-x0", type=int, default=None)
    parser.add_argument("--crop-y0", type=int, default=None)
    parser.add_argument(
        "--target-header-path",
        type=Path,
        default=None,
        help="FITS image supplying one exact target WCS for all output bands",
    )
    parser.add_argument(
        "--injection-manifest",
        type=Path,
        default=None,
        help="Frozen common-world detector-level injection manifest",
    )
    parser.add_argument(
        "--patt-nums",
        default="",
        help="Comma-separated PATT_NUM identifiers; calibration manifests restrict fitting to the training split",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--scene-initialization",
        choices=("common_product", "none"),
        default="common_product",
        help="Training-only common-WCS scene initialization policy",
    )
    parser.add_argument("--scene-initialization-steps", type=int, default=32)
    parser.add_argument("--scene-initialization-lr", type=float, default=5.0e-2)
    parser.add_argument("--foam-grid-side", type=int, default=32)
    parser.add_argument(
        "--foam-exact-max-candidate-pairs",
        type=int,
        default=10_000_000,
        help="Explicit allocation ceiling for exact Foam sparse geometry",
    )
    parser.add_argument("--foam-basis-count", type=int, default=4)
    parser.add_argument(
        "--foam-affine-spatial-basis",
        action="store_true",
        help="Learn cell-local x/y slopes and integrate them by exact first moments",
    )
    parser.add_argument("--foam-lr", type=float, default=2.0e-2)
    parser.add_argument(
        "--foam-log-intensity-spectra",
        action="store_true",
        help="Parameterize each Foam cell spectrum as exp(Legendre modes)",
    )
    parser.add_argument("--foam-initial-intensity", type=float, default=1.0e-2)
    parser.add_argument(
        "--foam-train-geometry",
        action="store_true",
        help="Train sites and radii; off by default for the fixed-grid control",
    )
    parser.add_argument(
        "--foam-initialize-from-coadd",
        action="store_true",
        help="Warm-start fixed cell spectra from observed per-band coadds",
    )
    parser.add_argument(
        "--foam-fit-target-footprint",
        action="store_true",
        help="Place the fixed grid over target-WCS bounds instead of the full domain",
    )
    parser.add_argument(
        "--foam-whiten-spectral-basis",
        action="store_true",
        help=(
            "Orthonormalize passband-integrated basis columns before fitting; "
            "requires at least as many bands as basis modes"
        ),
    )
    parser.add_argument(
        "--foam-exact-observation-refine",
        action="store_true",
        help=(
            "Finish fixed affine Foam with a joint matrix-free solve through "
            "native detector WCS pixels, weights, bandpasses, and symmetric PSFs"
        ),
    )
    parser.add_argument(
        "--foam-exact-operator-storage",
        choices=("model", "cpu"),
        default="model",
        help="Store all exact operators with the model or offload them to CPU",
    )
    parser.add_argument(
        "--foam-exact-observation-batch-size",
        default="auto",
        help="Maximum exact observations resident on solve device; auto or positive integer",
    )
    parser.add_argument(
        "--foam-exact-max-iterations",
        type=int,
        default=50,
        help="Maximum PCG iterations for exact detector-pixel refinement",
    )
    parser.add_argument(
        "--foam-exact-preconditioner",
        choices=("diagonal", "cell_block", "auto"),
        default="auto",
        help="Exact-solve preconditioner; auto applies a device-memory guard",
    )
    parser.add_argument(
        "--foam-extent",
        type=float,
        default=FOAM_FOOTPRINT_EXTENT,
        help=(
            "Half-width of the deterministic normalized Foam grid. The field "
            "domain maps the exposure-footprint union inside +/-0.5."
        ),
    )
    parser.add_argument(
        "--models",
        default="coadd_bilinear,spectral_basis_field,sky_encoder_resmlp,sky_lambda_feature_field,operator_constrained_feature_field",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument(
        "--fail-on-method-error",
        action="store_true",
        help="Exit nonzero when any requested model fails or is missing",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument(
        "--artifact-dir", type=Path, default=None,
        help="Saved model artifacts (default: <json-out stem>_artifacts); existing artifacts are not overwritten",
    )
    parser.add_argument(
        "--native-training-view-dir", type=Path, default=None,
        help="Export fixed-background-corrected native SCI/weight/mask inputs for matched classical coadds",
    )
    parser.add_argument(
        "--operator-calibration-manifest",
        type=Path,
        default=None,
        help="135-CRF calibration evidence manifest for frozen learned operators",
    )
    parser.add_argument(
        "--operator-calibration-fit-report",
        type=Path,
        default=None,
        help="Fit report containing adaptive checkpoints and validation pruning",
    )
    parser.add_argument(
        "--forward-context",
        action="store_true",
        help=(
            "Per-exposure ForwardContext for fit_field_model; wraps INR with "
            "_PerExposureScalarBias (see bench_map_smacs0723_field_fit.py)."
        ),
    )
    parser.add_argument(
        "--structured-noise-crossfit",
        action="store_true",
        help=(
            "Calibrate one frozen structured-noise model per exposure from "
            "leave-one-exposure-out Foam predictions, then use structured NLL"
        ),
    )
    parser.add_argument("--crossfit-scene-steps", type=int, default=4)
    parser.add_argument("--structured-noise-steps", type=int, default=50)
    parser.add_argument("--structured-noise-lr", type=float, default=1.0e-2)
    parser.add_argument(
        "--foam-initialized-inr",
        action="store_true",
        help="Fit a positive zero-correction INR initialized from the fitted Foam",
    )
    parser.add_argument("--inr-steps", type=int, default=None)
    parser.add_argument("--inr-lr", type=float, default=1.0e-3)
    parser.add_argument("--inr-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--tile-ra",
        type=float,
        default=None,
        help="COSMOS-scale tiling skeleton: tile center R.A. [deg]. "
        "Recorded in outputs this pass; per-tile FieldDomain partitioning "
        "is a later run.",
    )
    parser.add_argument(
        "--tile-dec",
        type=float,
        default=None,
        help="COSMOS-scale tiling skeleton: tile center Dec [deg].",
    )
    parser.add_argument(
        "--tile-size-deg",
        type=float,
        default=0.25,
        help="COSMOS-scale tiling skeleton: tile edge length [deg].",
    )
    parser.add_argument("--inr-hidden-layers", type=int, default=3)
    parser.add_argument(
        "--measured-passband-dir",
        type=Path,
        default=None,
        help="SVO NPZ curves; constant-I_nu photon-weighted response T/lambda",
    )
    parser.add_argument(
        "--independent-frozen-calibration",
        action="store_true",
        help="Require measured passbands and a matching frozen evidence/fit pair",
    )
    parser.add_argument(
        "--calibration-family",
        choices=("stationary", "adaptive"),
        default="adaptive",
        help="Predeclared frozen calibration family; do not select using test results",
    )
    parser.add_argument(
        "--psf-model",
        choices=("fitted_gaussian", "learned_lowrank"),
        default="fitted_gaussian",
        help="Control arm: nominal Gaussian width (legacy fitted_gaussian name) or frozen nominal low-rank initialization; independent fits require calibration evidence.",
    )
    parser.add_argument(
        "--background-model",
        choices=("none", "learned"),
        default="none",
        help="No additional background model, or the independently fitted "
        "ConditionedBackgroundField from the operator calibration fit.",
    )
    parser.add_argument(
        "--background-reference", choices=("calibrated", "training_mean"),
        default="calibrated",
        help="Keep calibrated levels or retain the common training background pedestal in the scene",
    )
    args = parser.parse_args()
    if args.inr_pcg_refine_steps < 0 or args.inr_pcg_iterations < 1:
        parser.error("neural PCG steps must be non-negative and iterations positive")
    if not math.isfinite(args.inr_pcg_damping) or args.inr_pcg_damping < 0:
        parser.error("--inr-pcg-damping must be finite and non-negative")
    if args.optimizer_lr is not None and (
        not math.isfinite(args.optimizer_lr) or args.optimizer_lr <= 0
    ):
        parser.error("--optimizer-lr must be finite and positive")
    if args.foam_initialize_from_coadd and args.scene_initialization == "none":
        parser.error("--foam-initialize-from-coadd conflicts with --scene-initialization none")
    if args.scene_initialization_steps <= 0:
        parser.error("--scene-initialization-steps must be positive")
    if not math.isfinite(args.scene_initialization_lr) or args.scene_initialization_lr <= 0:
        parser.error("--scene-initialization-lr must be finite and positive")
    if args.foam_train_geometry and (
        args.foam_initialize_from_coadd or args.foam_whiten_spectral_basis
    ):
        parser.error("coadd initialization and whitening require fixed Foam geometry")
    if args.foam_whiten_spectral_basis and args.scene_initialization != "common_product":
        parser.error(
            "--foam-whiten-spectral-basis requires common-product initialization"
        )
    if args.foam_exact_observation_refine and (
        args.foam_train_geometry or not args.foam_affine_spatial_basis
    ):
        parser.error(
            "--foam-exact-observation-refine requires fixed affine Foam geometry"
        )
    if args.foam_log_intensity_spectra and (
        args.foam_affine_spatial_basis
        or args.foam_initialize_from_coadd
        or args.foam_whiten_spectral_basis
        or args.foam_exact_observation_refine
    ):
        parser.error(
            "log-intensity Foam currently excludes affine spectra, coadd initialization, "
            "and the linear exact refiner"
        )
    exact_batch: int | str
    if str(args.foam_exact_observation_batch_size).strip().lower() == "auto":
        exact_batch = "auto"
    else:
        try:
            exact_batch = int(args.foam_exact_observation_batch_size)
        except ValueError:
            parser.error(
                "--foam-exact-observation-batch-size must be auto or a positive integer"
            )
        if exact_batch <= 0:
            parser.error(
                "--foam-exact-observation-batch-size must be auto or a positive integer"
            )
    if args.foam_exact_max_iterations <= 0:
        parser.error("--foam-exact-max-iterations must be positive")
    if args.foam_exact_max_candidate_pairs <= 0:
        parser.error("--foam-exact-max-candidate-pairs must be positive")
    if (args.operator_calibration_manifest is None) != (
        args.operator_calibration_fit_report is None
    ):
        parser.error(
            "--operator-calibration-manifest and "
            "--operator-calibration-fit-report must be provided together"
        )
    if args.foam_exact_observation_refine and (
        args.operator_calibration_manifest is not None or args.forward_context
    ):
        parser.error(
            "exact Foam observation refinement does not yet support "
            "conditioned operators or per-exposure bias"
        )
    if args.structured_noise_crossfit:
        if args.crossfit_scene_steps <= 0 or args.structured_noise_steps <= 0:
            parser.error("cross-fit scene and structured-noise steps must be positive")
        if not math.isfinite(args.structured_noise_lr) or args.structured_noise_lr <= 0:
            parser.error("--structured-noise-lr must be finite and positive")
        if args.forward_context or args.operator_calibration_manifest is not None:
            parser.error(
                "structured-noise cross-fitting does not yet support conditioned operators"
            )
        if args.foam_initialize_from_coadd:
            parser.error(
                "structured-noise cross-fitting forbids full-data coadd initialization"
            )
        if args.foam_exact_observation_refine:
            parser.error(
                "structured-noise cross-fitting cannot be followed by the MSE exact refiner"
            )
    if args.foam_initialized_inr:
        if args.inr_steps is not None and args.inr_steps <= 0:
            parser.error("--inr-steps must be positive")
        if not math.isfinite(args.inr_lr) or args.inr_lr <= 0:
            parser.error("--inr-lr must be finite and positive")
        if int(args.inr_hidden_dim) <= 0 or int(args.inr_hidden_layers) <= 0:
            parser.error("INR hidden dimensions must be positive")
        if not args.foam_log_intensity_spectra:
            parser.error("--foam-initialized-inr requires --foam-log-intensity-spectra")
    if (args.tile_ra is None) != (args.tile_dec is None):
        parser.error("--tile-ra and --tile-dec must be given together")
    if args.tile_ra is not None:
        for name, value in (
            ("--tile-ra", args.tile_ra),
            ("--tile-dec", args.tile_dec),
        ):
            if not math.isfinite(value) or not -180.0 <= value <= 360.0:
                parser.error(f"{name} must be a finite angle in [-180, 360] deg")
        if (
            not math.isfinite(args.tile_size_deg)
            or args.tile_size_deg <= 0.0
            or args.tile_size_deg > 2.0
        ):
            parser.error("--tile-size-deg must be in (0, 2] deg")
    tile_center = (
        None if args.tile_ra is None else (float(args.tile_ra), float(args.tile_dec))
    )
    calibrated_operators = args.operator_calibration_manifest is not None
    if args.native_training_view_dir is not None and (
        not args.independent_frozen_calibration or args.structured_noise_crossfit
    ):
        parser.error("--native-training-view-dir requires independent frozen calibration and diagonal pipeline ERR weighting")
    if args.background_reference == "training_mean" and (
        not calibrated_operators or args.background_model != "learned"
    ):
        parser.error("--background-reference training_mean requires calibrated learned backgrounds")
    if args.independent_frozen_calibration and (
        not calibrated_operators
        or args.measured_passband_dir is None
        or args.background_model != "learned"
        or args.psf_model == "learned_lowrank"
        or args.forward_context
    ):
        parser.error(
            "independent frozen calibration requires manifest, fit report, measured passbands, --background-model learned, and no nominal learned_lowrank or free-offset --forward-context override"
        )
    if args.background_model == "learned" and not calibrated_operators:
        parser.error(
            "--background-model learned requires --operator-calibration-manifest"
        )
    if (
        not math.isfinite(args.foam_initial_intensity)
        or args.foam_initial_intensity <= 0
    ):
        parser.error("--foam-initial-intensity must be finite and positive")

    bands = _parse_csv(args.bands)
    if args.foam_whiten_spectral_basis and args.foam_basis_count > len(bands):
        parser.error(
            "--foam-whiten-spectral-basis requires at least as many bands as modes"
        )
    pattern_numbers = tuple(int(value) for value in _parse_csv(args.patt_nums))
    model_kinds = [part.strip() for part in str(args.models).split(",") if part.strip()]
    if args.inr_pcg_refine_steps and not INR_PCG_MODEL_KINDS.intersection(model_kinds):
        parser.error("--inr-pcg-refine-steps requires a supported neural model in --models")
    unused_overrides = set(args.model_kwargs) - (set(model_kinds) - {"coadd_bilinear"})
    if unused_overrides:
        parser.error("--model-kwargs has models not fitted: " + ", ".join(sorted(unused_overrides)))
    unsupported_initializers = sorted(
        set(model_kinds) - SCENE_INITIALIZATION_MODEL_KINDS
    )
    if args.scene_initialization == "common_product" and unsupported_initializers:
        parser.error(
            "common-product scene initialization does not support: "
            + ", ".join(unsupported_initializers)
            + "; use --scene-initialization none"
        )
    if "diffuse_foam_hybrid" in model_kinds and not args.foam_log_intensity_spectra:
        parser.error("diffuse_foam_hybrid requires --foam-log-intensity-spectra")
    if (args.structured_noise_crossfit or args.foam_initialized_inr) and (
        "spectral_power_foam" not in model_kinds
    ):
        parser.error(
            "structured noise and Foam-initialized INR require spectral_power_foam"
        )
    if args.foam_initialized_inr and "foam_initialized_inr" not in model_kinds:
        foam_index = model_kinds.index("spectral_power_foam")
        model_kinds.insert(foam_index + 1, "foam_initialized_inr")
    if "foam_initialized_inr" in model_kinds:
        if "spectral_power_foam" not in model_kinds or model_kinds.index(
            "foam_initialized_inr"
        ) < model_kinds.index("spectral_power_foam"):
            parser.error("foam_initialized_inr must follow spectral_power_foam")
        if not args.foam_log_intensity_spectra:
            parser.error("foam_initialized_inr requires --foam-log-intensity-spectra")
    if "linear_band_grid_field" in model_kinds:
        if (
            args.forward_context
            or args.structured_noise_crossfit
            or args.psf_model == "learned_lowrank"
        ):
            parser.error(
                "linear_band_grid_field requires stationary operators without "
                "--forward-context, --structured-noise-crossfit, or learned_lowrank PSF"
            )
        if calibrated_operators and args.calibration_family != "stationary":
            parser.error(
                "linear_band_grid_field with calibrated operators requires "
                "--calibration-family stationary"
            )
    rows: list[dict[str, Any]] = []
    field_artifacts: dict[str, dict[str, Any]] = {}
    effective_model_configs: dict[str, dict[str, Any]] = {}
    artifact_dir = args.artifact_dir or args.json_out.with_name(
        f"{args.json_out.stem}_artifacts"
    )
    if not args.forward_context:
        for model_kind in model_kinds:
            if (artifact_dir / _safe_slug(model_kind) / "manifest.json").exists():
                parser.error(f"artifact already exists for {model_kind}: {artifact_dir}")

    band_entries: list[dict[str, Any]] = []
    held_out_exposures_omitted: dict[str, dict[str, int]] = {}
    all_field_observations: list[FieldObservation] = []
    all_forward_contexts: list[ForwardContext] = []
    validation_observations: list[FieldObservation] = []
    validation_contexts: list[ForwardContext] = []
    validation_scores: dict[str, list[dict[str, Any]]] = {}
    injection_rows: list[dict[str, Any]] = []
    injection_manifest = (
        json.loads(args.injection_manifest.read_text(encoding="utf-8"))
        if args.injection_manifest is not None
        else None
    )
    calibration_manifest_payload = (
        json.loads(args.operator_calibration_manifest.read_text(encoding="utf-8"))
        if args.operator_calibration_manifest is not None
        else None
    )
    calibration_lookup = (
        _calibration_exposure_lookup(calibration_manifest_payload)
        if calibration_manifest_payload is not None
        else {}
    )
    calibration_samples: dict[tuple[str, str], list[dict[str, Any]]] = {}
    calibration_psf_radius = 0
    if calibration_manifest_payload and "calibration_partitioning" in calibration_manifest_payload:
        extraction = calibration_manifest_payload["extraction"]
        calibration_psf_radius = math.ceil(extraction.get("settings", extraction)["psf_radius_px"])
        for sample in calibration_manifest_payload["evidence"]:
            key = (calibration_exposure_id(sample), str(sample["band"]).upper())
            calibration_samples.setdefault(key, []).append(sample)
    explicit_target_header = (
        dict(torchfits.read(str(args.target_header_path), return_header=True)[1])
        if args.target_header_path is not None
        else None
    )
    if (args.crop_x0 is None) != (args.crop_y0 is None):
        raise ValueError("--crop-x0 and --crop-y0 must be provided together")
    if explicit_target_header is not None and (
        int(explicit_target_header.get("NAXIS1", -1)) != int(args.crop_size)
        or int(explicit_target_header.get("NAXIS2", -1)) != int(args.crop_size)
    ):
        raise ValueError("explicit target header shape must match --crop-size")

    with tempfile.TemporaryDirectory(prefix="torchsky_smacs_joint_") as tmp:
        tmpdir = Path(tmp)
        for band in bands:
            reference_path = _reference_path_for_band(band)
            exposure_paths = _discover_exposure_paths(
                args.exposure_dir,
                input_product=args.input_product,
                band=band,
                module=args.module,
                filename_contains="nrcblong"
                if band in {"F277W", "F356W", "F444W"}
                else None,
            )
            if pattern_numbers:
                exposure_paths = _filter_exposure_paths_by_patterns(
                    exposure_paths,
                    pattern_numbers,
                )
            validation_paths = []
            if calibrated_operators:
                validation_paths = [
                    path for path in exposure_paths
                    if _calibration_exposure(calibration_lookup, path).get("split") == "validation"
                ]
                exposure_paths, omitted = _training_calibration_exposure_paths(
                    exposure_paths, calibration_lookup
                )
                held_out_exposures_omitted[band] = omitted
            if not reference_path.exists() or not exposure_paths:
                message = (
                    f"missing reference={reference_path.exists()} "
                    f"n_exposures={len(exposure_paths)}"
                )
                if args.skip_missing:
                    for model_kind in model_kinds:
                        rows.append(
                            {
                                "band": band,
                                "model": model_kind,
                                "status": "MISSING",
                                "message": message,
                                "n_exposures": len(exposure_paths),
                                "forward_context": bool(args.forward_context),
                            }
                        )
                    continue
                raise FileNotFoundError(message)

            ref_data = _read_ext(reference_path, "SCI")
            ref_header = _sci_header(reference_path)
            overlap_headers = [_sci_header(path) for path in exposure_paths]
            if args.crop_x0 is not None:
                x0 = int(args.crop_x0)
                y0 = int(args.crop_y0)
                if (
                    x0 < 0
                    or y0 < 0
                    or x0 + int(args.crop_size) > ref_data.shape[1]
                    or y0 + int(args.crop_size) > ref_data.shape[0]
                ):
                    raise ValueError("fixed crop lies outside the reference image")
                ref_crop = ref_data[
                    y0 : y0 + int(args.crop_size),
                    x0 : x0 + int(args.crop_size),
                ]
            elif args.crop_strategy == "best_overlap":
                coarse_step = 4
                overlap_count = _approx_overlap_grid(
                    overlap_headers,
                    ref_header,
                    coarse_step=coarse_step,
                )
                coarse_ref = ref_data[::coarse_step, ::coarse_step]
                coarse_size = max(1, int(np.ceil(int(args.crop_size) / coarse_step)))
                _, coarse_x0, coarse_y0 = _best_overlap_cutout(
                    coarse_ref,
                    overlap_count,
                    coarse_size,
                )
                x0 = min(
                    int(coarse_x0 * coarse_step),
                    max(0, ref_data.shape[1] - int(args.crop_size)),
                )
                y0 = min(
                    int(coarse_y0 * coarse_step),
                    max(0, ref_data.shape[0] - int(args.crop_size)),
                )
                ref_crop = ref_data[
                    y0 : y0 + int(args.crop_size), x0 : x0 + int(args.crop_size)
                ]
            else:
                ref_crop, x0, y0 = _center_cutout(ref_data, int(args.crop_size))
            target_header = (
                dict(explicit_target_header)
                if explicit_target_header is not None
                else _cutout_header(ref_header, x0, y0, int(args.crop_size))
            )
            psf_contract = gaussian_psf_contract(band)
            target_psf_sigma_pix = _fitted_gaussian_psf_sigma_pix(
                band,
                target_header,
            )
            target_observation = _field_target_observation(
                target_header,
                band_name=band,
                psf_sigma_pix=target_psf_sigma_pix,
            )
            if args.measured_passband_dir is not None:
                target_observation = replace(
                    target_observation,
                    **_field_observation_kwargs_for_band(
                        band, measured_passband_dir=args.measured_passband_dir
                    ),
                )
            science_overrides = None
            if injection_manifest is not None:
                injection_world, injection_flux = _injection_manifest_tensors(
                    injection_manifest,
                    band=band,
                )
                science_overrides, injection_audit = inject_common_world_exposures(
                    exposure_paths + validation_paths,
                    band=band,
                    world_radec_deg=injection_world,
                    flux_jy=injection_flux,
                    injection_sign=_injection_manifest_sign(injection_manifest),
                )
                injection_rows.extend({"band": band, **row} for row in injection_audit)

            image_observations: list[ImageObservation] = []
            field_observations: list[FieldObservation] = []
            band_validation_observations: list[FieldObservation] = []
            band_validation_contexts: list[ForwardContext] = []
            for exposure_path in exposure_paths + validation_paths:
                science_override = (
                    science_overrides.get(exposure_path)
                    if science_overrides is not None
                    else None
                )
                science = (
                    _read_ext(exposure_path, "SCI")
                    if science_override is None
                    else np.asarray(science_override)
                )
                err = _read_ext(exposure_path, "ERR")
                dq = _read_ext(exposure_path, "DQ")
                dq_support, contamination_prior = (
                    _jwst_dq_support_and_contamination_prior(dq)
                )
                valid = np.isfinite(err) & (err > 0) & dq_support
                calibration_exposure = (
                    _calibration_exposure(calibration_lookup, exposure_path)
                    if calibrated_operators else None
                )
                if calibration_samples and calibration_exposure is not None:
                    exposure_identity = calibration_exposure.get("exposure_id", calibration_exposure["path"])
                    valid = _exclude_calibration_samples(
                        valid, calibration_samples.get((exposure_identity, str(band).upper()), []),
                        x0=float(calibration_exposure["detector_bounds_xy"][0]),
                        y0=float(calibration_exposure["detector_bounds_xy"][2]),
                        psf_radius=calibration_psf_radius,
                    )
                if exposure_path in exposure_paths:
                    science_path, weight_path, mask_path = _prepare_aux_files(
                        exposure_path, tmpdir, linearize_sip=False,
                        science_override=science_override, additional_mask=valid,
                    )
                    image_observations.append(ImageObservation(
                        str(science_path), hdu=0, weight_path=str(weight_path),
                        weight_hdu=0, mask_path=str(mask_path), mask_hdu=0,
                    ))
                weight = np.zeros_like(err, dtype=np.float32)
                weight[valid] = 1.0 / np.square(err[valid])
                header = dict(_sci_header(exposure_path))
                cropped = _crop_field_inputs_to_target(
                    image=science,
                    weight=weight,
                    mask=valid.astype(np.float32, copy=False),
                    header=header,
                    target_header=target_header,
                )
                if cropped is None:
                    continue
                crop_image, crop_weight, crop_mask, crop_header = cropped
                obs_index = len(all_field_observations) + len(field_observations)
                calibration_exposure = (
                    _calibration_exposure(calibration_lookup, exposure_path)
                    if calibrated_operators
                    else None
                )
                detector_x0 = float(crop_header["SUBSTRT1"]) - 1.0
                detector_y0 = float(crop_header["SUBSTRT2"]) - 1.0
                source_x0 = int(crop_header["SUBSTRT1"]) - int(
                    header.get("SUBSTRT1", 1)
                )
                source_y0 = int(crop_header["SUBSTRT2"]) - int(
                    header.get("SUBSTRT2", 1)
                )
                if calibration_exposure is not None:
                    detector_x0 += float(
                        calibration_exposure["detector_bounds_xy"][0]
                    ) - (float(header.get("SUBSTRT1", 1)) - 1.0)
                    detector_y0 += float(
                        calibration_exposure["detector_bounds_xy"][2]
                    ) - (float(header.get("SUBSTRT2", 1)) - 1.0)
                    crop_header["SUBSTRT1"] = int(detector_x0) + 1
                    crop_header["SUBSTRT2"] = int(detector_y0) + 1
                source_slice = (
                    slice(source_y0, source_y0 + crop_image.shape[0]),
                    slice(source_x0, source_x0 + crop_image.shape[1]),
                )
                # Retain only cropped variance arrays between exposures.
                variance_crops = []
                for extension in ("VAR_POISSON", "VAR_RNOISE", "VAR_FLAT"):
                    variance = _read_optional_ext(exposure_path, extension)
                    variance_crops.append(
                        None if variance is None else torch.from_numpy(variance[source_slice].copy())
                    )
                del variance
                amplifier_columns = torch.clamp(
                    (
                        torch.arange(crop_image.shape[1], dtype=torch.int64)
                        + int(detector_x0)
                    )
                    // 512,
                    min=0,
                    max=3,
                )
                is_validation = exposure_path in validation_paths
                (band_validation_observations if is_validation else field_observations).append(
                    FieldObservation(
                        image=torch.from_numpy(crop_image),
                        header=crop_header,
                        weight=torch.from_numpy(crop_weight),
                        mask=torch.from_numpy(crop_mask),
                        err=torch.from_numpy(err[source_slice].copy()),
                        var_poisson=variance_crops[0],
                        var_rnoise=variance_crops[1],
                        var_flat=variance_crops[2],
                        contamination_prior=torch.from_numpy(
                            contamination_prior[source_slice].copy()
                        ),
                        amplifier_index=amplifier_columns[None, :].expand(
                            crop_image.shape[0], crop_image.shape[1]
                        ),
                        detector_x0_pix=detector_x0,
                        detector_y0_pix=detector_y0,
                        psf_sigma_pix=_fitted_gaussian_psf_sigma_pix(
                            band,
                            crop_header,
                        ),
                        psf_spec=replace(
                            PSFSpec.gaussian(
                                _fitted_gaussian_psf_sigma_pix(band, crop_header)
                            ),
                            pixel_response_included=False,
                        ),
                        **_field_observation_kwargs_for_band(
                            band, measured_passband_dir=args.measured_passband_dir
                        ),
                    )
                )
                (band_validation_contexts if is_validation else all_forward_contexts).append(
                    ForwardContext(
                        survey=str(band),
                        instrument="NIRCAM" if calibration_exposure else None,
                        detector=(
                            str(calibration_exposure["detector"])
                            if calibration_exposure
                            else None
                        ),
                        exposure_id=f"{band}_{exposure_path.stem}",
                        time_mjd=(
                            float(calibration_exposure["timing"]["midpoint_mjd_utc"])
                            if calibration_exposure
                            else None
                        ),
                        extra={
                            "calibration_exposure_id": calibration_exposure.get(
                                "exposure_id", calibration_exposure["path"]
                            )
                            if calibration_exposure else None,
                            "exposure_index": obs_index,
                            "source_path": str(exposure_path),
                            "split": calibration_exposure.get("split")
                            if calibration_exposure
                            else "train",
                            "spectral_response_convention": "constant_I_nu_photon_mean_T_over_lambda"
                            if args.measured_passband_dir
                            else "nominal_gaussian_control",
                            "measured_passband_path": str(
                                args.measured_passband_dir
                                / f"JWST_NIRCam2025.{band}.npz"
                            )
                            if args.measured_passband_dir
                            else None,
                            "measured_passband_sha256": hashlib.sha256(
                                (
                                    args.measured_passband_dir
                                    / f"JWST_NIRCam2025.{band}.npz"
                                ).read_bytes()
                            ).hexdigest()
                            if args.measured_passband_dir
                            else None,
                        },
                    )
                )
            if not field_observations:
                if args.skip_missing:
                    for model_kind in model_kinds:
                        rows.append(
                            {
                                "band": band,
                                "model": model_kind,
                                "status": "MISSING",
                                "message": "no overlapping field observations",
                                "n_exposures": len(exposure_paths),
                                "forward_context": bool(args.forward_context),
                            }
                        )
                    continue
                raise RuntimeError(f"no overlapping field observations for {band}")

            validation_observations.extend(band_validation_observations)
            validation_contexts.extend(band_validation_contexts)
            coadd_reference = coadd_image_plane(
                image_observations,
                target=dict(target_header),
                kernel="bilinear",
                semantics="surface_brightness",
                method="weighted_mean",
            )
            coadd_weight = coadd_reference.weight.cpu().numpy()
            coadd_data = (
                coadd_reference.data.cpu().numpy().astype(np.float32, copy=False)
            )
            support_mask = coadd_weight > 0
            if not np.any(support_mask):
                support_mask = np.isfinite(ref_crop) & np.isfinite(coadd_data)

            band_entries.append(
                {
                    "band": band,
                    "reference_crop": (
                        coadd_data
                        if explicit_target_header is not None
                        else ref_crop.astype(np.float32, copy=False)
                    ),
                    "target_header": dict(target_header),
                    "target_observation": target_observation,
                    "truth_psf_fwhm_arcsec": psf_contract.truth_fwhm_arcsec,
                    "fitted_psf_fwhm_arcsec": psf_contract.fitted_fwhm_arcsec,
                    "input_psf_sigma_pix_min": min(
                        observation.psf_sigma_pix for observation in field_observations
                    ),
                    "input_psf_sigma_pix_max": max(
                        observation.psf_sigma_pix for observation in field_observations
                    ),
                    "target_psf_sigma_pix": target_psf_sigma_pix,
                    "support_mask": support_mask.astype(bool, copy=False),
                    "coadd_pred": coadd_data,
                    "n_exposures": len(exposure_paths),
                }
            )
            all_field_observations.extend(field_observations)

        if not band_entries:
            if args.skip_missing:
                payload = {
                    "rows": rows,
                    "summary": [],
                    "status": "skipped",
                    "reason": "no bands available (missing reference or exposures)",
                }
                text = json.dumps(payload, indent=2)
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                args.json_out.write_text(text, encoding="utf-8")
                args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
                args.markdown_out.write_text(_format_markdown(rows), encoding="utf-8")
                print(text)
                return 0
            raise RuntimeError("no bands available for joint fit")

        intensity_scale, normalization_training_count = _training_intensity_scale(
            all_field_observations,
            [str(context.extra.get("split")) for context in all_forward_contexts],
        )
        normalized_observations = [
            _scaled_field_observation(observation, intensity_scale=intensity_scale)
            for observation in all_field_observations
        ]
        normalized_validation = [
            _scaled_field_observation(observation, intensity_scale=intensity_scale)
            for observation in validation_observations
        ]

        psf_arm_audit: dict[str, float] | None = None
        if args.psf_model == "learned_lowrank":
            psf_arm_audit = _apply_learned_psf_arm(normalized_observations, bands)
        spherical_domain = estimate_field_domain(
            normalized_observations,
            coord_mode="sphere_xyz",
        )
        local_domain = estimate_field_domain(
            normalized_observations,
            coord_mode="local_tangent",
        )
        forward_ctx_tuple: tuple[ForwardContext | None, ...] | None = None
        if args.forward_context or calibrated_operators:
            if len(all_forward_contexts) != len(normalized_observations):
                raise RuntimeError(
                    "internal: joint forward context list length mismatch"
                )
            forward_ctx_tuple = tuple(all_forward_contexts)

        calibrated_bundles: tuple[ObservationOperatorBundle, ...] | None = None
        validation_bundles: tuple[ObservationOperatorBundle, ...] = ()
        calibration_audit: dict[str, Any] | None = None
        native_training_view_metadata: dict[str, Any] | None = None
        if calibrated_operators:
            assert args.operator_calibration_manifest is not None
            assert args.operator_calibration_fit_report is not None
            detectors = {
                str(context.detector).upper()
                for context in all_forward_contexts + validation_contexts
            }
            calibrated_models, calibration_audit = _load_calibrated_operator_models(
                args.operator_calibration_manifest,
                args.operator_calibration_fit_report,
                detectors=detectors,
                intensity_scale=float(intensity_scale),
                family=args.calibration_family,
                contexts=all_forward_contexts + validation_contexts,
            )
            if args.background_reference == "training_mean":
                calibration_audit["background_reference"] = (
                    _reference_calibrated_backgrounds_to_training_mean(
                        calibrated_models, all_forward_contexts + validation_contexts,
                        manifest=calibration_manifest_payload,
                        intensity_scale=float(intensity_scale),
                    )
                )
            calibrated_bundles = _calibrated_operator_bundles(
                normalized_observations, all_forward_contexts, calibrated_models
            )
            validation_bundles = _calibrated_operator_bundles(
                normalized_validation, validation_contexts, calibrated_models
            )
            if args.background_model == "none":
                # Preserve supplied offsets but apply no fitted background field.
                calibrated_bundles = tuple(
                    bundle.with_updates(background=None)
                    for bundle in calibrated_bundles
                )
                validation_bundles = tuple(
                    bundle.with_updates(background=None)
                    for bundle in validation_bundles
                )

            if args.native_training_view_dir is not None:
                native_training_view_metadata = _export_native_training_view(
                    args.native_training_view_dir, normalized_observations,
                    all_forward_contexts, calibrated_bundles,
                    intensity_scale=float(intensity_scale),
                    fit_kwargs={
                        "num_spectral_samples": 9,
                        "spatial_sampling": _spatial_sampling_for_operator_bundles(
                            "gauss_legendre_2x2", calibrated_bundles
                        ),
                        "render_spatial_tile_hw": (256, 256),
                    },
                    calibration_metadata={
                        "manifest": str(args.operator_calibration_manifest.resolve()),
                        "manifest_sha256": calibration_fit._sha256(args.operator_calibration_manifest),
                        "fit_report": str(args.operator_calibration_fit_report.resolve()),
                        "fit_report_sha256": calibration_fit._sha256(args.operator_calibration_fit_report),
                        "family": args.calibration_family,
                        "background_model": args.background_model,
                        "background_reference": args.background_reference,
                        "background_gauge": calibration_audit.get("background_reference"),
                        "injection_manifest": str(args.injection_manifest.resolve())
                        if args.injection_manifest is not None else None,
                        "injection_manifest_sha256": calibration_fit._sha256(args.injection_manifest)
                        if args.injection_manifest is not None else None,
                    },
                )

        foam_bounds = (
            _foam_target_footprint_bounds(local_domain, band_entries)
            if args.foam_fit_target_footprint
            else None
        )
        grid_bounds = (
            _grid_target_footprint_bounds(
                local_domain,
                band_entries,
                margin_sigma=float(args.grid_psf_margin_sigma),
            )
            if args.grid_fit_target_footprint
            else None
        )
        foam_model_kwargs = _regular_foam_model_kwargs(
            grid_side=int(args.foam_grid_side),
            basis_count=int(args.foam_basis_count),
            affine_spatial_basis=bool(args.foam_affine_spatial_basis),
            extent=float(args.foam_extent),
            bounds_xy=foam_bounds,
            log_intensity_spectra=bool(args.foam_log_intensity_spectra),
            initial_intensity=float(args.foam_initial_intensity),
        )
        foam_model_kwargs["compact_fixed_geometry"] = not bool(args.foam_train_geometry)
        foam_model_kwargs = _comparison_model_kwargs(
            "spectral_power_foam", foam_model_kwargs, args.model_kwargs,
        )

        def make_foam_model() -> torch.nn.Module:
            model = make_field_model(
                "spectral_power_foam", local_domain, **foam_model_kwargs
            )
            _set_foam_geometry_trainable(model, args.foam_train_geometry)
            return model

        structured_bundles: tuple[ObservationOperatorBundle, ...] | None = None
        noise_crossfit_audit: dict[str, Any] | None = None
        if args.structured_noise_crossfit:
            crossfit_kwargs = _joint_real_fit_kwargs(
                "spectral_power_foam",
                crop_size=int(args.crop_size),
                steps=int(args.crossfit_scene_steps),
                num_observations=max(len(normalized_observations) - 1, 1),
                foam_grid_side=int(args.foam_grid_side),
                foam_lr=float(args.foam_lr),
                foam_fixed_geometry=not bool(args.foam_train_geometry),
            )
            crossfit_predictions, crossfit_diagnostics = (
                _cross_fitted_scene_predictions(
                    normalized_observations,
                    model_factory=make_foam_model,
                    steps=int(args.crossfit_scene_steps),
                    fit_kwargs=crossfit_kwargs,
                    render_kwargs={
                        "num_spectral_samples": int(
                            crossfit_kwargs.get("num_spectral_samples", 1)
                        ),
                        "spatial_sampling": str(
                            crossfit_kwargs.get("spatial_sampling", "center")
                        ),
                    },
                    seed=int(args.seed),
                )
            )
            noise_models, crossfit_noise_losses = fit_structured_noise_crossfit(
                normalized_observations,
                crossfit_predictions,
                steps=int(args.structured_noise_steps),
                lr=float(args.structured_noise_lr),
                device="cpu",
            )
            structured_bundles = _structured_noise_bundles(
                normalized_observations, noise_models
            )
            noise_crossfit_audit = {
                "scene_folds": list(crossfit_diagnostics),
                "noise_models": _structured_noise_diagnostics(
                    normalized_observations,
                    crossfit_predictions,
                    noise_models,
                    crossfit_noise_losses,
                ),
            }

        scene_initialization_product: ProductResult | None = None
        scene_initialization_metadata: dict[str, Any] = {
            "policy": str(args.scene_initialization),
            "elapsed_ms": 0.0,
        }
        default_operator_bundles = tuple(
            field_observation_operator_bundle(observation)
            for observation in normalized_observations
        )
        linear_grid_override = (
            args.model_kwargs.get("linear_band_grid_field", {})
            if isinstance(args.model_kwargs, dict)
            else {}
        )
        linear_grid_needs_common_init = (
            str(linear_grid_override.get("prior_reference", "zero")) == "initial"
        )
        needs_common_product_init = any(
            kind not in {"coadd_bilinear", "linear_band_grid_field"}
            or (kind == "linear_band_grid_field" and linear_grid_needs_common_init)
            for kind in model_kinds
        )
        if args.scene_initialization == "common_product" and needs_common_product_init:
            active_initialization_bundles = (
                structured_bundles
                or calibrated_bundles
                or default_operator_bundles
            )
            initialization_sampling = _spatial_sampling_for_operator_bundles(
                "gauss_legendre_2x2", active_initialization_bundles
            )
            initialization_target = min(
                band_entries,
                key=lambda entry: (
                    float(entry["target_observation"].wavelength),
                    str(entry["band"]),
                ),
            )
            initialization_start = time.perf_counter()
            scene_initialization_product = _fit_training_scene_initialization_product(
                normalized_observations,
                all_forward_contexts,
                active_initialization_bundles,
                target_header=dict(initialization_target["target_header"]),
                fit_kwargs={
                    "num_spectral_samples": 9,
                    "spatial_sampling": initialization_sampling,
                    "render_spatial_tile_hw": (256, 256),
                },
                steps=int(args.scene_initialization_steps),
                lr=float(args.scene_initialization_lr),
            )
            initialization_elapsed_ms = (
                time.perf_counter() - initialization_start
            ) * 1000.0
            scene_initialization_metadata = _save_scene_initialization_product(
                artifact_dir / "scene_initialization",
                scene_initialization_product,
                elapsed_ms=initialization_elapsed_ms,
                pixel_response_convention=_pixel_response_convention(
                    active_initialization_bundles
                ),
                intensity_scale=float(intensity_scale),
            )

        artifact_paths = {
            Path(value).resolve()
            for context in all_forward_contexts + validation_contexts
            for key in ("source_path", "measured_passband_path")
            if (value := context.extra.get(key)) is not None
        }
        artifact_paths.update(_reference_path_for_band(str(entry["band"])).resolve()
                              for entry in band_entries)
        artifact_paths.update(
            path.resolve() for path in (
                args.operator_calibration_manifest, args.operator_calibration_fit_report,
                args.injection_manifest, args.target_header_path,
            ) if path is not None
        )
        if calibration_audit is not None:
            artifact_paths.update(Path(item["checkpoint"]).resolve()
                                  for item in calibration_audit["models"])
        # Hash each source once for all models, including SCI/ERR/DQ/variance HDUs.
        artifact_inputs = [
            {"path": str(path), "size": path.stat().st_size,
             "sha256": calibration_fit._sha256(path)}
            for path in sorted(artifact_paths)
        ]
        artifact_run_config = {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        fitted_models: dict[
            str,
            tuple[Any, float, dict[str, Any], dict[str, Any]],
        ] = {}
        for model_kind in model_kinds:
            if model_kind == "coadd_bilinear":
                continue
            if model_kind == "spectral_power_foam":
                model_kwargs = foam_model_kwargs
                model_steps = int(args.steps)
            elif model_kind == "diffuse_foam_hybrid":
                model_kwargs = {
                    "foam": make_foam_model(),
                    "diffuse_hidden_dim": int(args.inr_hidden_dim),
                    "diffuse_hidden_layers": int(args.inr_hidden_layers),
                }
                model_steps = int(args.steps)
            elif model_kind == "foam_initialized_inr":
                foam_reference = fitted_models["spectral_power_foam"][0]
                foam_reference = getattr(foam_reference, "inner", foam_reference)
                model_kwargs = {
                    "foam": foam_reference,
                    "hidden_dim": int(args.inr_hidden_dim),
                    "hidden_layers": int(args.inr_hidden_layers),
                }
                model_steps = int(args.inr_steps or args.steps)
            elif model_kind == "linear_band_grid_field":
                linear_grid_bounds = (
                    grid_bounds
                    if grid_bounds is not None
                    else _grid_target_footprint_bounds(
                        local_domain,
                        band_entries,
                        margin_sigma=float(args.grid_psf_margin_sigma),
                    )
                )
                model_kwargs = {
                    "spatial_resolution": max(2 * int(args.crop_size), 256),
                    "spatial_bounds_local_deg": linear_grid_bounds,
                    "solver": "cg",
                    "max_iterations": min(max(int(args.steps), 48), 64),
                    "irls_steps": 1,
                    "edge_preserving": False,
                    "non_negative_weight": 0.0,
                    "data_weight_fn": "none",
                    "irls_use_mad_scaling": False,
                    "damping": 5.0e-3,
                    "smoothness_weight": 2.0e-2,
                    "relative_regularization": True,
                    "prior_reference": "zero",
                    "preconditioner": "diagonal_majorizer",
                    "tolerance": 1.0e-6,
                }
                model_steps = int(model_kwargs["max_iterations"])
            else:
                model_kwargs = _field_model_kwargs_for_case(
                    model_kind,
                    case=REAL_JOINT_CASE,
                    size=int(args.crop_size),
                )
                if model_kind == "sasnet":
                    model_kwargs.update(_joint_sasnet_kwargs(normalized_observations))
                if model_kind == "explicit_spectral_grid_field" and grid_bounds is not None:
                    model_kwargs["spatial_bounds_local_deg"] = grid_bounds
                model_steps = int(args.steps)
            model_kwargs = _comparison_model_kwargs(model_kind, model_kwargs, args.model_kwargs)
            if model_kind == "linear_band_grid_field" and "max_iterations" in model_kwargs:
                model_steps = int(model_kwargs["max_iterations"])
            fit_kwargs = _joint_real_fit_kwargs(
                model_kind,
                crop_size=int(args.crop_size),
                steps=model_steps,
                optimizer_lr=args.optimizer_lr,
                full_observation_training=args.full_observation_training,
                num_observations=len(normalized_observations),
                foam_grid_side=int(args.foam_grid_side),
                foam_lr=float(args.foam_lr),
                foam_fixed_geometry=not bool(args.foam_train_geometry),
            )
            if forward_ctx_tuple is not None:
                fit_kwargs = dict(fit_kwargs)
                fit_kwargs["observation_forward_contexts"] = forward_ctx_tuple
            if calibrated_bundles is not None:
                fit_kwargs = dict(fit_kwargs)
                fit_kwargs["spatial_sampling"] = _spatial_sampling_for_operator_bundles(
                    str(fit_kwargs["spatial_sampling"]),
                    (*calibrated_bundles, *validation_bundles),
                )
                fit_kwargs["observation_operator_bundles"] = calibrated_bundles
            if structured_bundles is not None:
                fit_kwargs = _structured_scene_fit_kwargs(
                    fit_kwargs, num_observations=len(normalized_observations)
                )
                fit_kwargs["spatial_sampling"] = _spatial_sampling_for_operator_bundles(
                    str(fit_kwargs["spatial_sampling"]), structured_bundles
                )
                fit_kwargs["observation_operator_bundles"] = structured_bundles
            if calibrated_bundles is None and structured_bundles is None:
                fit_kwargs = dict(fit_kwargs)
                fit_kwargs["spatial_sampling"] = _spatial_sampling_for_operator_bundles(
                    str(fit_kwargs["spatial_sampling"]), default_operator_bundles
                )
            if model_kind == "foam_initialized_inr" and args.optimizer_lr is None:
                fit_kwargs["lr"] = float(args.inr_lr)
            effective_model_configs[model_kind] = _report_configuration({
                "model_kwargs": model_kwargs,
                "fit_kwargs": {
                    "steps": model_steps,
                    **{key: value for key, value in fit_kwargs.items() if key not in {
                        "observation_forward_contexts", "observation_operator_bundles",
                    }},
                },
            })
            domain = (
                local_domain
                if model_kind in {
                    "spectral_power_foam", "foam_initialized_inr",
                    "diffuse_foam_hybrid", "sasnet",
                    "explicit_spectral_grid_field", "linear_band_grid_field",
                    "sine_implicit", "wire_implicit",
                    "flair_bla",
                    "gaussian_basis",
                }
                else spherical_domain
            )
            torch.manual_seed(int(args.seed))
            if model_kind == "spectral_power_foam":
                model: torch.nn.Module = make_foam_model()
            elif model_kind == "linear_band_grid_field":
                bounds_tuple = tuple(
                    float(v) for v in model_kwargs["spatial_bounds_local_deg"]
                )
                assert len(bounds_tuple) == 4
                model = _MultiBandLinearGridScene(
                    domain,
                    band_entries=band_entries,
                    spatial_resolution=int(model_kwargs["spatial_resolution"]),
                    spatial_bounds_local_deg=bounds_tuple,  # type: ignore[arg-type]
                )
            else:
                model = make_field_model(model_kind, domain, **model_kwargs)
            gpu_mem_note("model-built")
            model_initialization_start = time.perf_counter()
            if scene_initialization_product is None or (
                model_kind == "linear_band_grid_field"
                and str(model_kwargs.get("prior_reference", "zero")) == "zero"
            ):
                initialization_diagnostics: dict[str, Any] = {
                    "scene_initialization_policy": "none"
                }
            else:
                initialization_diagnostics = _initialize_joint_scene(
                    model,
                    model_kind,
                    scene_initialization_product,
                )
                if (
                    model_kind == "spectral_power_foam"
                    and not args.foam_train_geometry
                    and not args.foam_log_intensity_spectra
                ):
                    initialization_diagnostics.update(_initialize_foam_from_band_coadds(
                        model,
                        _foam_initialization_entries(
                            scene_initialization_product, band_entries
                        ),
                        intensity_scale=1.0,
                        num_spectral_samples=int(
                            fit_kwargs.get("num_spectral_samples", 1)
                        ),
                        operator_max_candidate_pairs=int(
                            args.foam_exact_max_candidate_pairs
                        ),
                        whiten_spectral_basis=bool(args.foam_whiten_spectral_basis),
                    ))
                gpu_mem_note("foam-coadd-init-done")
            model_initialization_ms = (
                time.perf_counter() - model_initialization_start
            ) * 1000.0
            initialization_diagnostics["scene_initialization_model_elapsed_ms"] = (
                model_initialization_ms
            )
            initialization_diagnostics["scene_initialization_common_elapsed_ms"] = float(
                0.0
                if (
                    model_kind == "linear_band_grid_field"
                    and str(model_kwargs.get("prior_reference", "zero")) == "zero"
                )
                else scene_initialization_metadata["elapsed_ms"]
            )
            if args.forward_context:
                model = _PerExposureScalarBias(model, len(normalized_observations))
            gpu_mem_note("pre-field-fit")
            t0 = time.perf_counter()
            progress_steps = {max(1, math.ceil(model_steps * fraction))
                              for fraction in (0.1, 0.25, 0.5, 0.75, 1.0)}

            def report_progress(step_index: int, loss: float) -> None:
                if step_index + 1 in progress_steps:
                    print(
                        f"{model_kind}: step {step_index + 1}/{model_steps}, "
                        f"loss={loss:.6g}, elapsed={time.perf_counter() - t0:.1f}s",
                        flush=True,
                    )

            print(f"Starting {model_kind}: {model_steps} steps", flush=True)
            if model_kind == "linear_band_grid_field":
                assert isinstance(model, _MultiBandLinearGridScene)
                result = _fit_linear_band_grid_scene(
                    normalized_observations,
                    model,
                    contexts=forward_ctx_tuple,
                    operator_bundles=calibrated_bundles,
                    background_reference=str(args.background_reference),
                    solver=str(model_kwargs.get("solver", "cg")),
                    max_iterations=int(model_kwargs.get("max_iterations", model_steps)),
                    irls_steps=int(model_kwargs.get("irls_steps", 3)),
                    edge_preserving=bool(model_kwargs.get("edge_preserving", True)),
                    non_negative_weight=float(
                        model_kwargs.get("non_negative_weight", 0.25)
                    ),
                    data_weight_fn=str(
                        model_kwargs.get("data_weight_fn", "none")
                    ),
                    irls_use_mad_scaling=bool(
                        model_kwargs.get("irls_use_mad_scaling", False)
                    ),
                    damping=float(model_kwargs.get("damping", 1.0e-3)),
                    smoothness_weight=float(
                        model_kwargs.get("smoothness_weight", 2.0e-3)
                    ),
                    relative_regularization=bool(
                        model_kwargs.get("relative_regularization", True)
                    ),
                    prior_reference=str(
                        model_kwargs.get("prior_reference", "zero")
                    ),
                    preconditioner=str(
                        model_kwargs.get("preconditioner", "diagonal_majorizer")
                    ),
                    tolerance=float(model_kwargs.get("tolerance", 1.0e-7)),
                )
            else:
                result = fit_field_model(
                    normalized_observations,
                    model,
                    steps=model_steps,
                    _progress_callback=report_progress,
                    **fit_kwargs,
                )
                if args.inr_pcg_refine_steps and model_kind in INR_PCG_MODEL_KINDS:
                    result = _refine_neural_scene_pcg(
                        result, normalized_observations, fit_kwargs=fit_kwargs,
                        steps=int(args.inr_pcg_refine_steps),
                        cg_iterations=int(args.inr_pcg_iterations),
                        damping=float(args.inr_pcg_damping), seed=int(args.seed),
                    )
            prerequisite_time_ms = (
                float(fitted_models["spectral_power_foam"][1])
                if model_kind == "foam_initialized_inr"
                else 0.0
                if (
                    model_kind == "linear_band_grid_field"
                    and str(model_kwargs.get("prior_reference", "zero")) == "zero"
                )
                else float(scene_initialization_metadata["elapsed_ms"])
            )
            initialization_diagnostics["scene_initialization_prerequisite_elapsed_ms"] = (
                prerequisite_time_ms
            )
            fit_time_ms = (
                (time.perf_counter() - t0) * 1000.0
                + model_initialization_ms
                + prerequisite_time_ms
            )
            exact_refinement_diagnostics: dict[str, float] = {}
            if (
                model_kind == "spectral_power_foam"
                and args.foam_exact_observation_refine
            ):
                if calibrated_bundles is not None or args.forward_context:
                    raise ValueError(
                        "exact Foam observation refinement does not yet support "
                        "conditioned operators or per-exposure bias"
                    )
                hard_model = getattr(result.model, "inner", result.model)
                refine_t0 = time.perf_counter()
                exact_refinement_diagnostics = _refine_foam_from_observations_exact(
                    hard_model,
                    normalized_observations,
                    num_spectral_samples=int(fit_kwargs.get("num_spectral_samples", 1)),
                    operator_max_candidate_pairs=int(
                        args.foam_exact_max_candidate_pairs
                    ),
                    operator_storage_device=str(args.foam_exact_operator_storage),
                    observation_batch_size=exact_batch,
                    max_iterations=int(args.foam_exact_max_iterations),
                    preconditioner=str(args.foam_exact_preconditioner),
                )
                fit_time_ms += (time.perf_counter() - refine_t0) * 1000.0
            field_artifacts[model_kind] = _save_joint_field_artifact(
                artifact_dir / _safe_slug(model_kind), result=result,
                model_kind=model_kind, model_kwargs=model_kwargs,
                fit_kwargs={"steps": model_steps, **fit_kwargs},
                run_config=artifact_run_config, contexts=all_forward_contexts,
                input_identities=artifact_inputs, calibration_audit=calibration_audit,
                intensity_scale=intensity_scale,
                normalization_training_count=normalization_training_count,
                scene_initialization={
                    **scene_initialization_metadata,
                    "model": model_kind,
                    "model_elapsed_ms": model_initialization_ms,
                    "diagnostics": initialization_diagnostics,
                },
                unfrozen_extras=(structured_bundles is not None
                                 or args.psf_model == "learned_lowrank"),
                exact_refined=bool(exact_refinement_diagnostics) or bool(
                    (result.diagnostics or {}).get("inr_pcg_refinement")
                ),
            )
            if validation_bundles and not args.forward_context:
                validation_scores[model_kind] = _validation_detector_scores(
                    result.model, normalized_validation, validation_contexts,
                    validation_bundles, intensity_scale=intensity_scale,
                    fit_kwargs=fit_kwargs,
                )
            fitted_models[model_kind] = (
                result.model,
                fit_time_ms,
                fit_kwargs,
                {
                    **dict(result.diagnostics or {}),
                    "initial_loss": float(result.initial_loss),
                    "final_loss": float(result.final_loss),
                    "last_training_loss": float(result.loss_history[-1]),
                    **initialization_diagnostics,
                    **exact_refinement_diagnostics,
                },
            )

        for entry in band_entries:
            band = str(entry["band"])
            ref_crop = entry["reference_crop"]
            target_header = entry["target_header"]
            support_mask = entry["support_mask"]
            target_observation = entry["target_observation"]
            for model_kind in model_kinds:
                row = {
                    "band": band,
                    "model": model_kind,
                    "n_exposures": int(entry["n_exposures"]),
                    "assumed_reference_psf_fwhm_arcsec": float(entry["truth_psf_fwhm_arcsec"]),
                    "assumed_control_psf_fwhm_arcsec": float(entry["fitted_psf_fwhm_arcsec"]),
                    "psf_width_status": "not measured; Gaussian widths are approximation controls",
                    "input_psf_sigma_pix_min": float(entry["input_psf_sigma_pix_min"]),
                    "input_psf_sigma_pix_max": float(entry["input_psf_sigma_pix_max"]),
                    "target_psf_sigma_pix": float(entry["target_psf_sigma_pix"]),
                    "forward_context": bool(args.forward_context),
                    "calibrated_operators": bool(
                        calibrated_operators and model_kind != "coadd_bilinear"
                    ),
                    "structured_noise_crossfit": bool(
                        args.structured_noise_crossfit
                        and model_kind != "coadd_bilinear"
                    ),
                    "optimizer_steps": (
                        int(args.inr_steps or args.steps)
                        if model_kind == "foam_initialized_inr"
                        else int(args.steps)
                        if model_kind != "coadd_bilinear"
                        else 0
                    ),
                    "status": "OK",
                }
                try:
                    if model_kind == "coadd_bilinear":
                        pred = entry["coadd_pred"]
                        fit_time_ms = 0.0
                    else:
                        (
                            result_model,
                            fit_time_ms,
                            fit_kwargs,
                            fit_diagnostics,
                        ) = fitted_models[model_kind]
                        row["parameter_count"] = int(
                            sum(
                                parameter.numel()
                                for parameter in result_model.parameters()
                            )
                        )
                        row["trainable_parameter_count"] = int(
                            sum(
                                parameter.numel()
                                for parameter in result_model.parameters()
                                if parameter.requires_grad
                            )
                        )
                        grid_model = getattr(result_model, "inner", result_model)
                        if isinstance(grid_model, ExplicitSpectralGridField):
                            row.update(
                                _grid_target_sampling_diagnostics(
                                    grid_model, dict(target_header)
                                )
                            )
                        elif isinstance(grid_model, _MultiBandLinearGridScene):
                            row.update(
                                _grid_target_sampling_diagnostics(
                                    grid_model.model_for_band(band),
                                    dict(target_header),
                                )
                            )
                        linear_solver_band = (
                            fit_diagnostics.get("linear_solver_by_band", {}).get(
                                str(band).strip().upper(), {}
                            )
                        )
                        for diag_key, diag_val in linear_solver_band.items():
                            row[diag_key] = diag_val
                        if "linear_solver_iterations" in linear_solver_band:
                            row["optimizer_steps"] = int(
                                linear_solver_band["linear_solver_iterations"]
                            )
                        row["spectral_samples"] = int(
                            fit_kwargs.get("num_spectral_samples", 1)
                        )
                        row["spatial_sampling"] = str(
                            fit_kwargs.get("spatial_sampling", "center")
                        )
                        row["intensity_scale"] = float(intensity_scale)
                        row["initial_loss"] = float(fit_diagnostics["initial_loss"])
                        row["final_loss"] = float(fit_diagnostics["final_loss"])
                        row["last_training_loss"] = float(
                            fit_diagnostics["last_training_loss"]
                        )
                        if "inr_pcg_refinement" in fit_diagnostics:
                            row["inr_pcg_refinement"] = fit_diagnostics["inr_pcg_refinement"]
                        row["foam_initialized_from_coadd"] = bool(
                            model_kind == "spectral_power_foam"
                            and args.scene_initialization == "common_product"
                        )
                        for diagnostic_name, diagnostic_value in fit_diagnostics.items():
                            if diagnostic_name.startswith("scene_initialization_"):
                                row[diagnostic_name] = diagnostic_value
                        for diagnostic_name in (
                            "foam_initialized_cell_fraction",
                            "foam_fully_covered_cell_fraction",
                            "foam_initialization_rmse",
                            "foam_initial_coefficient_abs_max",
                            "foam_initial_gradient_coefficient_abs_max",
                            "foam_spectral_basis_condition_number",
                            "foam_spectral_basis_raw_condition_number",
                            "foam_spectral_basis_rank",
                            "foam_initial_cell_variation_abs_p95",
                            "foam_initial_cell_variation_abs_max",
                            "foam_exact_refinement_rmse",
                            "foam_exact_refinement_initial_rmse",
                            "foam_exact_refinement_observations",
                            "foam_exact_refinement_pixels",
                            "foam_exact_refinement_max_iterations",
                            "foam_exact_refinement_iterations",
                            "foam_exact_refinement_normal_operator_applications",
                            "foam_exact_refinement_initial_relative_residual",
                            "foam_exact_refinement_ridge_centered_on_initial",
                            "foam_exact_refinement_relative_residual",
                            "foam_exact_refinement_converged",
                            "foam_exact_refinement_preconditioner_setup_seconds",
                            "foam_exact_refinement_rhs_seconds",
                            "foam_exact_refinement_initial_residual_seconds",
                            "foam_exact_refinement_pcg_seconds",
                            "foam_exact_refinement_preconditioner_cell_block",
                            "foam_exact_refinement_preconditioner_block_fallback_cells",
                            "foam_exact_refinement_preconditioner_storage_bytes",
                            "foam_exact_refinement_preconditioner_auto",
                            "foam_exact_refinement_preconditioner_response_condition",
                            "foam_exact_refinement_preconditioner_auto_condition_threshold",
                            "foam_exact_refinement_preconditioner_auto_budget_bytes",
                            "foam_exact_refinement_preconditioner_block_construction_peak_bytes",
                            "foam_exact_refinement_observation_batch_size",
                            "foam_exact_refinement_observation_batch_auto",
                            "foam_exact_refinement_observation_batch_transfer_budget_bytes",
                            "foam_exact_refinement_max_observation_transfer_bytes",
                            "foam_exact_refinement_operator_storage_bytes",
                            "foam_exact_refinement_peak_operator_batch_bytes",
                            "foam_exact_refinement_host_offload",
                        ):
                            if diagnostic_name in fit_diagnostics:
                                row[diagnostic_name] = float(
                                    fit_diagnostics[diagnostic_name]
                                )
                        row["training_patch_size"] = int(
                            fit_kwargs.get("training_patch_size") or args.crop_size
                        )
                        row["optimizer_lr"] = float(fit_kwargs["lr"])
                        performance = fit_diagnostics.get("performance_profile", {})
                        row["peak_accelerator_memory_bytes"] = int(
                            performance.get("max_memory_allocated_bytes", 0)
                        )
                        render_num_spectral_samples = int(
                            fit_kwargs.get("num_spectral_samples", 1)
                        )
                        render_ctx = (
                            ForwardContext(survey=str(band))
                            if args.forward_context
                            else None
                        )
                        if model_kind == "spectral_power_foam":
                            hard_model = getattr(result_model, "inner", result_model)
                            coefficients = hard_model.cell_coefficients.detach()
                            sites = hard_model.sites.detach()
                            radii = hard_model.radii.detach()
                            probe = torch.tensor(
                                [
                                    [0.0, 0.0],
                                    [float(args.crop_size - 1), 0.0],
                                    [0.0, float(args.crop_size - 1)],
                                    [
                                        float(args.crop_size - 1),
                                        float(args.crop_size - 1),
                                    ],
                                    [
                                        0.5 * float(args.crop_size - 1),
                                        0.5 * float(args.crop_size - 1),
                                    ],
                                ],
                                dtype=torch.float64,
                                device=sites.device,
                            )
                            world = (
                                torchsky.WCS(target_header)
                                .to(sites.device)
                                .pixel_to_world(probe)
                            )
                            target_coordinates = (
                                hard_model.normalized_spatial_coordinates(
                                    world[:, 0], world[:, 1]
                                )
                            )
                            row["foam_coefficient_abs_max"] = float(
                                coefficients.abs().max()
                            )
                            row["foam_log_intensity_spectra"] = bool(
                                hard_model.foam.log_intensity_spectra
                            )
                            gradients = hard_model.cell_gradient_coefficients
                            row["foam_affine_spatial_basis"] = gradients is not None
                            x_centers = torch.unique(sites[:, 0])
                            y_centers = torch.unique(sites[:, 1])
                            if x_centers.numel() < 2 or y_centers.numel() < 2:
                                raise ValueError(
                                    "regular Foam diagnostics require a 2-D grid"
                                )
                            cell_width_x = float(
                                (x_centers.max() - x_centers.min())
                                / (x_centers.numel() - 1)
                            )
                            cell_width_y = float(
                                (y_centers.max() - y_centers.min())
                                / (y_centers.numel() - 1)
                            )
                            cell_scale = math.sqrt(cell_width_x * cell_width_y)
                            if gradients is not None:
                                row["foam_gradient_coefficient_abs_max"] = float(
                                    gradients.detach().abs().max()
                                )
                                wavelengths, response = bandpass_quadrature(
                                    target_observation,
                                    num_spectral_samples=render_num_spectral_samples,
                                    device=sites.device,
                                )
                                band_basis = torch.sum(
                                    response.to(dtype=sites.dtype)[:, None]
                                    * hard_model.spectral_basis(wavelengths).to(
                                        dtype=sites.dtype
                                    ),
                                    dim=0,
                                )
                                band_gradients = torch.einsum(
                                    "k,ckd->cd",
                                    band_basis,
                                    gradients.detach(),
                                )
                                cell_variation = cell_scale * torch.linalg.vector_norm(
                                    band_gradients,
                                    dim=-1,
                                )
                                row["foam_band_cell_variation_abs_p95"] = float(
                                    torch.quantile(cell_variation, 0.95)
                                )
                                row["foam_band_cell_variation_abs_max"] = float(
                                    torch.max(cell_variation)
                                )
                            row["foam_site_x_range"] = [
                                float(sites[:, 0].min()),
                                float(sites[:, 0].max()),
                            ]
                            row["foam_site_y_range"] = [
                                float(sites[:, 1].min()),
                                float(sites[:, 1].max()),
                            ]
                            row["foam_radius_range"] = [
                                float(radii.min()),
                                float(radii.max()),
                            ]
                            row["foam_extent"] = float(args.foam_extent)
                            row["foam_grid_cell_width_x"] = cell_width_x
                            row["foam_grid_cell_width_y"] = cell_width_y
                            row["foam_grid_cell_scale"] = cell_scale
                            row["foam_temperature_start"] = float(
                                hard_model.temperature_start
                            )
                            row["foam_temperature_end"] = float(
                                hard_model.temperature_end
                            )
                            row["foam_geometry_trainable"] = bool(
                                hard_model.sites.requires_grad
                                or hard_model.foam.raw_radii.requires_grad
                            )
                            row["foam_compact_fixed_geometry"] = bool(
                                hard_model.compact_fixed_geometry
                            )
                            row["foam_target_x_range"] = [
                                float(target_coordinates[:, 0].min()),
                                float(target_coordinates[:, 0].max()),
                            ]
                            row["foam_target_y_range"] = [
                                float(target_coordinates[:, 1].min()),
                                float(target_coordinates[:, 1].max()),
                            ]
                            with torch.no_grad():
                                if args.foam_exact_observation_refine:
                                    rendered = render_spectral_power_foam_observation_exact(
                                        hard_model, target_observation,
                                        num_spectral_samples=render_num_spectral_samples,
                                        operator_kwargs={
                                            "max_candidate_pairs": int(args.foam_exact_max_candidate_pairs)
                                        },
                                    )
                                    row["render_mode"] = "hard_pixel_footprint"
                                else:
                                    rendered = _render_joint_scene_product(
                                        result_model, target_observation, fit_kwargs,
                                        render_num_spectral_samples, render_ctx,
                                    )
                                    row["render_mode"] = (
                                        "trained_soft_"
                                        f"{fit_kwargs.get('spatial_sampling', 'center')}"
                                    )
                            row["product_interpretation"] = (
                                "target-grid scene with assumed target PSF; "
                                "not a calibrated detector prediction"
                            )
                        else:
                            rendered = _render_joint_scene_product(
                                result_model, target_observation, fit_kwargs,
                                render_num_spectral_samples, render_ctx,
                            )
                            row["render_mode"] = (
                                f"trained_{fit_kwargs.get('spatial_sampling', 'center')}"
                            )
                            row["product_interpretation"] = (
                                "target-grid scene with assumed target PSF; "
                                + (
                                    "retains common training foreground pedestal; absolute foreground unestablished"
                                    if args.background_reference == "training_mean"
                                    else "excludes fixed detector background"
                                )
                            )
                        pred = (
                            rendered.detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        ) * intensity_scale
                        row["prediction_abs_max"] = float(np.max(np.abs(pred)))
                        row["prediction_nonzero_fraction"] = float(
                            np.count_nonzero(pred) / pred.size
                        )
                    support_metric_mask = _metric_mask(pred, ref_crop, support_mask)
                    uniform_metric_mask = np.isfinite(pred) & np.isfinite(ref_crop)
                    if not np.any(uniform_metric_mask):
                        raise RuntimeError(
                            "render produced no finite comparison pixels"
                        )
                    support_diff = (
                        pred[support_metric_mask] - ref_crop[support_metric_mask]
                    )
                    uniform_diff = (
                        pred[uniform_metric_mask] - ref_crop[uniform_metric_mask]
                    )
                    source_stats = _background_subtracted_source_stats(
                        pred,
                        ref_crop,
                        support_mask,
                    )
                    noise_stats = _surface_brightness_noise_diagnostics(
                        pred,
                        entry["coadd_pred"],
                        ref_crop,
                        support_mask,
                    )

                    # Robust affine fit to ignore background offsets
                    source_metric_mask = _source_metric_mask(
                        pred, ref_crop, support_mask
                    )
                    ref_pts = ref_crop[source_metric_mask].astype(np.float64)
                    pred_pts = pred[source_metric_mask].astype(np.float64)
                    design = np.stack([ref_pts, np.ones_like(ref_pts)], axis=1)
                    coeffs, *_ = np.linalg.lstsq(design, pred_pts, rcond=None)
                    affine_slope = float(coeffs[0])
                    affine_intercept = float(coeffs[1])
                    aligned_pred = (
                        (pred_pts - affine_intercept) / max(affine_slope, 1e-6)
                        if affine_slope > 0.5
                        else pred_pts
                    )
                    aligned_diff = aligned_pred - ref_pts

                    row.update(
                        {
                            "source_rmse": source_stats["source_rmse"],
                            "source_contrast_rmse": source_stats[
                                "source_contrast_rmse"
                            ],
                            "source_contrast_slope": source_stats[
                                "source_contrast_slope"
                            ],
                            "support_rmse": float(
                                np.sqrt(np.mean(support_diff * support_diff))
                            )
                            if support_diff.size
                            else float("nan"),
                            "uniform_rmse": float(
                                np.sqrt(np.mean(uniform_diff * uniform_diff))
                            )
                            if uniform_diff.size
                            else float("nan"),
                            "aligned_rmse": float(
                                np.sqrt(np.mean(aligned_diff * aligned_diff))
                            )
                            if aligned_diff.size
                            else float("nan"),
                            "affine_slope": affine_slope,
                            "support_flux_ratio": (
                                float(np.sum(pred_pts) / np.sum(ref_pts))
                                if ref_pts.size and np.sum(ref_pts) != 0.0
                                else float("nan")
                            ),
                            "source_flux_ratio": source_stats["source_flux_ratio"],
                            "support_frac": float(support_metric_mask.mean()),
                            "source_frac": source_stats["source_frac"],
                            **noise_stats,
                            "fit_time_ms": float(fit_time_ms),
                        }
                    )
                    _write_export_products(
                        args.export_dir,
                        band=band,
                        model=model_kind,
                        psf_mode=str(args.psf_model),
                        header=target_header,
                        reference=ref_crop,
                        prediction=pred,
                        support_mask=support_mask,
                        source_mask=source_metric_mask,
                    )
                except Exception as exc:
                    row["status"] = "FAIL"
                    row["message"] = str(exc)
                rows.append(row)

    ok_rows = [row for row in rows if row["status"] == "OK"]
    fail_rows = [row for row in rows if row["status"] != "OK"]
    ok_rows.sort(
        key=lambda row: (
            str(row["model"]),
            float(row["support_rmse"]),
            str(row["band"]),
        )
    )
    fail_rows.sort(key=lambda row: (str(row["model"]), str(row["band"])))
    if calibrated_bundles is None:
        exposure_pixel_response = [
            (
                getattr(observation.psf_spec, "pixel_response_included", None),
                "unknown; no calibrated PSF operator was supplied",
            )
            for observation in all_field_observations
        ]
    else:
        exposure_pixel_response = [
            (
                None if bundle.psf is None else bundle.psf.pixel_response_included,
                None if bundle.psf is None else bundle.psf.pixel_response_provenance,
            )
            for bundle in calibrated_bundles
        ]
    if validation_bundles:
        validation_pixel_response = [
            (
                None if bundle.psf is None else bundle.psf.pixel_response_included,
                None if bundle.psf is None else bundle.psf.pixel_response_provenance,
            )
            for bundle in validation_bundles
        ]
    else:
        validation_pixel_response = [
            (
                getattr(observation.psf_spec, "pixel_response_included", None),
                "unknown; no calibrated validation PSF operator was supplied",
            )
            for observation in normalized_validation
        ]
    validation_response_values = {item[0] for item in validation_pixel_response}
    validation_response_provenance = {
        item[1] for item in validation_pixel_response
    }
    payload = {
        "experiment": {
            "input_product": str(args.input_product),
            "exposure_dir": str(args.exposure_dir),
            "bands": bands,
            "module": str(args.module),
            "psf_model": str(args.psf_model),
            "background_model": str(args.background_model),
            "background_reference": str(args.background_reference),
            "independent_frozen_calibration": bool(args.independent_frozen_calibration),
            "calibration_family": args.calibration_family,
            "spectral_response_convention": "constant_I_nu_photon_mean_T_over_lambda"
            if args.measured_passband_dir
            else "nominal_gaussian_control",
            "measured_passband_dir": str(args.measured_passband_dir)
            if args.measured_passband_dir
            else None,
            "normalization_training_exposures": normalization_training_count,
            "normalization_operation": "SCI/scale; ERR/scale; VAR/scale**2; no additive subtraction",
            "scene_initialization": scene_initialization_metadata,
            "jwst_reference_convention_verified": False,
            "held_out_validation_pipeline": (
                "unavailable: training-exposure scalar biases cannot transfer to validation"
                if args.forward_context else "fixed-model detector prediction; no held-out residual fitting"
            ) if calibrated_operators else "unavailable: no calibration manifest",
            "held_out_exposures_omitted": held_out_exposures_omitted,
            "validation_detector_scoring": {
                "split": "validation",
                "test_exposures_evaluated": 0,
                "pixel_domain": "all valid pixels in detector crops overlapping the target footprint",
                "likelihood": "diagonal Gaussian using supplied pipeline ERR squared; ignores pixel correlations and fitted structured noise",
                "normalization": "training-only scale; NLL restored to original SCI units by adding log(scale) per pixel",
                "pixel_response_included": next(iter(validation_response_values))
                if len(validation_response_values) == 1
                else None,
                "pixel_response_provenance": next(
                    iter(validation_response_provenance)
                )
                if len(validation_response_provenance) == 1
                else "mixed pixel-response provenance",
                "scores_by_model": validation_scores,
            },
            "exposure_provenance": [
                {
                    "source_path": context.extra.get("source_path"),
                    "photometric_header": {key: observation.header.get(key)
                        for key in ("BUNIT", "PHOTMJSR", "PIXAR_SR", "CRDS_CTX")},
                    "pixel_response_included": pixel_response[0],
                    "pixel_response_provenance": pixel_response[1],
                    "measured_passband_path": context.extra.get("measured_passband_path"),
                    "measured_passband_sha256": context.extra.get("measured_passband_sha256"),
                }
                for observation, context, pixel_response in zip(
                    all_field_observations,
                    all_forward_contexts,
                    exposure_pixel_response,
                    strict=True,
                )
            ],
            "psf_arm_audit": psf_arm_audit,
            "tile_center_deg": tile_center,
            "tile_size_deg": (
                float(args.tile_size_deg) if tile_center is not None else None
            ),
            "crop_size": int(args.crop_size),
            "export_dir": str(args.export_dir.resolve()),
            "optimizer_steps": int(args.steps),
            "seed": int(args.seed),
            "foam_grid_side": int(args.foam_grid_side),
            "foam_affine_spatial_basis": bool(args.foam_affine_spatial_basis),
            "foam_exact_max_candidate_pairs": int(args.foam_exact_max_candidate_pairs),
            "foam_basis_count": int(args.foam_basis_count),
            "foam_extent": float(args.foam_extent),
            "foam_lr": float(args.foam_lr),
            "foam_log_intensity_spectra": bool(args.foam_log_intensity_spectra),
            "foam_initial_intensity": float(args.foam_initial_intensity),
            "foam_train_geometry": bool(args.foam_train_geometry),
            "foam_initialize_from_coadd": bool(args.foam_initialize_from_coadd),
            "foam_fit_target_footprint": bool(args.foam_fit_target_footprint),
            "grid_fit_target_footprint": bool(args.grid_fit_target_footprint),
            "grid_psf_margin_sigma": float(args.grid_psf_margin_sigma),
            "foam_whiten_spectral_basis": bool(args.foam_whiten_spectral_basis),
            "foam_exact_observation_refine": bool(args.foam_exact_observation_refine),
            "foam_exact_operator_storage": str(args.foam_exact_operator_storage),
            "foam_exact_observation_batch_size": exact_batch,
            "foam_exact_max_iterations": int(args.foam_exact_max_iterations),
            "foam_exact_preconditioner": str(args.foam_exact_preconditioner),
            "structured_noise_crossfit": bool(args.structured_noise_crossfit),
            "crossfit_scene_steps": int(args.crossfit_scene_steps),
            "structured_noise_steps": int(args.structured_noise_steps),
            "structured_noise_lr": float(args.structured_noise_lr),
            "foam_initialized_inr": bool(args.foam_initialized_inr),
            "inr_steps": int(args.inr_steps or args.steps),
            "inr_lr": float(args.inr_lr),
            "inr_hidden_dim": int(args.inr_hidden_dim),
            "inr_hidden_layers": int(args.inr_hidden_layers),
            "intensity_scale": float(intensity_scale),
            "patt_nums": list(pattern_numbers),
            "crop_x0": args.crop_x0,
            "crop_y0": args.crop_y0,
            "target_header_path": (
                str(args.target_header_path.resolve())
                if args.target_header_path is not None
                else None
            ),
            "injection_manifest": (
                str(args.injection_manifest.resolve())
                if args.injection_manifest is not None
                else None
            ),
            "operator_calibration_manifest": (
                str(args.operator_calibration_manifest.resolve())
                if args.operator_calibration_manifest is not None
                else None
            ),
            "operator_calibration_fit_report": (
                str(args.operator_calibration_fit_report.resolve())
                if args.operator_calibration_fit_report is not None
                else None
            ),
        },
        "injection_audit": injection_rows,
        "exposures": [
            {
                "exposure_index": index,
                "exposure_id": context.exposure_id,
                "band": context.survey,
                "source_path": (context.extra or {}).get("source_path"),
            }
            for index, context in enumerate(all_forward_contexts)
        ],
        "operator_calibration": calibration_audit,
        "field_artifacts": field_artifacts,
        "effective_model_configs": effective_model_configs,
        "psf_arm_audit": psf_arm_audit,
        "structured_noise_crossfit": noise_crossfit_audit,
        "rows": ok_rows + fail_rows,
        "summary": _summarize_by_model(ok_rows),
    }
    if native_training_view_metadata is not None:
        payload["native_training_view"] = native_training_view_metadata
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(
        _format_markdown(payload["rows"], validation_scores=validation_scores),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    if args.fail_on_method_error and fail_rows:
        return 1
    return 0


if __name__ == "__main__" and os.environ.get("TORCHSKY_GPU_MEM_LOG") == "1":
    torch.cuda.memory._record_memory_history(
        enabled="all", context="all", stacks="python", max_entries=100000
    )
    try:
        _code = main()
    except (torch.OutOfMemoryError, RuntimeError) as _exc:
        if "NVML_SUCCESS" not in str(_exc) and not isinstance(
            _exc, torch.OutOfMemoryError
        ):
            raise
        _snapshot = torch.cuda.memory._snapshot()
        _by_trace = collections.Counter()
        for _seg_key, _segment in (
            _snapshot.items() if isinstance(_snapshot, dict) else enumerate(_snapshot)
        ):
            if not isinstance(_segment, list):
                continue
            for _block in _segment:
                if not isinstance(_block, dict):
                    continue
                _size = int(_block.get("allocated_size") or _block.get("size", 0) or 0)
                _frames = []
                _hist = _block.get("history")
                if isinstance(_hist, list):
                    for _h in reversed(_hist):
                        _frames = (_h or {}).get("frames") or []
                        if _frames:
                            break
                if not _frames:
                    _frames = _block.get("frames") or []
                if _frames:
                    _last = _frames[-1]
                    _prev = _frames[-2] if len(_frames) > 1 else {}
                    _key = (
                        f"{_prev.get('filename', '?')}:{_prev.get('line', 0)}"
                        f" -> {_last.get('filename', '?')}:{_last.get('line', 0)}"
                        f" [{_seg_key}]"
                    )
                else:
                    _key = f"<no-frames> [{_seg_key}]"
                _by_trace[_key] += _size
        print("[gpu-mem] OOM top allocation sites:")
        for _key, _size in _by_trace.most_common(15):
            print(f"[gpu-mem] {_size / 1024**3:8.3f} GiB  {_key}", flush=True)
        raise
    raise SystemExit(_code)
elif __name__ == "__main__":
    raise SystemExit(main())
