from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchfits

from torchsky.map import calibration as calibration_mod
from torchsky.map import image_plane_projection as image_plane_projection_mod
from torchsky.map import interpolated_mapping as interpolated_mapping_mod
from torchsky.map import windowing as image_grid_mod
from torchsky.map.operators import validate_image_plane_2d

pytest.importorskip("torchsky._cpp")


from torchsky_harness.benchmarks import bench_map_ground_truth as _truth_bench


@pytest.mark.parametrize(
    ("case", "rmse_max", "mae_max", "coverage_min"),
    [
        ("tan_aligned_low_noise", 0.03, 0.02, 0.99),
        ("tan_aligned_high_noise", 0.11, 0.09, 0.99),
        ("tan_multires_noise", 0.085, 0.06, 0.99),
        ("tan_multires_extreme", 0.20, 0.15, 0.97),
        ("tan_low_overlap_exptime_transmission", 0.32, 0.22, 0.99),
        ("tpv_multires_noise", 0.24, 0.12, 0.99),
        ("tan_flux_multires", 0.12, 0.08, 0.99),
        ("tpv_flux_multires_mild", 0.25, 0.12, 0.80),
    ],
)
def test_map_ground_truth_cases_stay_bounded(
    tmp_path: Path,
    case: str,
    rmse_max: float,
    mae_max: float,
    coverage_min: float,
) -> None:
    row = _truth_bench.evaluate_torchsky_case(
        tmp_path,
        case=case,
        size=64,
        seed=1234,
        runs=1,
    )

    assert row["library"] == "torchsky"
    assert row["interpolation_kernel"] == "bilinear"
    assert row["resampling_method"] == "resample_weighted_numerator_and_denominator"
    assert row["stacking_method"] == "weighted_mean"
    assert row["coverage_frac"] >= coverage_min
    assert row["rmse_vs_truth"] <= rmse_max
    assert row["mae_vs_truth"] <= mae_max
    assert row["weighted_residual_rms"] >= 0.0
    assert row["background_residual_rms"] >= 0.0
    assert row["background_leakage_fraction"] >= 0.0
    assert row["detrended_residual_rms"] >= 0.0


@pytest.mark.parametrize(
    ("case", "flux_rel_err_max"),
    [
        ("tan_flux_multires", 0.08),
        ("tpv_flux_multires_mild", 0.16),
    ],
)
def test_map_ground_truth_flux_cases_stay_flux_conservative(
    tmp_path: Path,
    case: str,
    flux_rel_err_max: float,
) -> None:
    row = _truth_bench.evaluate_torchsky_case(
        tmp_path,
        case=case,
        size=64,
        seed=1234,
        runs=1,
    )

    assert row["value_semantics"] == "flux"
    assert row["total_flux_rel_err"] <= flux_rel_err_max


def test_map_ground_truth_high_noise_degrades_against_low_noise(tmp_path: Path) -> None:
    (tmp_path / "low").mkdir()
    (tmp_path / "high").mkdir()
    low = _truth_bench.evaluate_torchsky_case(
        tmp_path / "low",
        case="tan_aligned_low_noise",
        size=64,
        seed=1234,
        runs=1,
    )
    high = _truth_bench.evaluate_torchsky_case(
        tmp_path / "high",
        case="tan_aligned_high_noise",
        size=64,
        seed=1234,
        runs=1,
    )

    assert low["projection"] == "TAN"
    assert high["projection"] == "TAN"
    assert high["mean_noise_sigma_flux"] > low["mean_noise_sigma_flux"]
    assert high["rmse_vs_truth"] > low["rmse_vs_truth"]
    assert high["mae_vs_truth"] > low["mae_vs_truth"]


def test_map_ground_truth_emits_kernel_recommendations(tmp_path: Path) -> None:
    rows = [
        _truth_bench.evaluate_torchsky_case(
            tmp_path,
            case="tan_flux_multires",
            size=64,
            seed=1234,
            runs=1,
            resampling_kernel=kernel,
        )
        for kernel in ("bilinear", "catmull_rom", "mitchell_netravali")
    ]
    recs = _truth_bench._kernel_recommendation_rows(rows)

    assert any(row["profile"] == "default_general" for row in recs)
    assert any(row["profile"] == "high_accuracy_interpolatory" for row in recs)
    assert any(row["profile"] == "smooth_reconstruction" for row in recs)


def test_map_ground_truth_conservative_polygon_reports_conservative_geometry(
    tmp_path: Path,
) -> None:
    row = _truth_bench.evaluate_torchsky_case(
        tmp_path,
        case="tan_aligned_low_noise",
        size=64,
        seed=1234,
        runs=1,
        resampling_kernel="conservative_polygon",
    )

    assert row["library"] == "torchsky"
    assert row["variant"] == "conservative_polygon"
    assert row["geometry_family"] == "conservative_overlap"
    assert row["coverage_frac"] > 0.95
    assert row["rmse_vs_truth"] >= 0.0


@pytest.mark.parametrize("operator_mode", ["explicit", "learned_shell"])
def test_map_ground_truth_scene_render_rows_share_product_metrics(
    tmp_path: Path,
    operator_mode: str,
) -> None:
    row = _truth_bench.evaluate_torchsky_scene_case(
        tmp_path,
        case="tan_aligned_low_noise",
        size=64,
        seed=1234,
        runs=1,
        operator_mode=operator_mode,
    )

    assert row["library"] == "torchsky_scene"
    assert row["method"] == "scene_render"
    assert row["variant"] == operator_mode
    assert row["geometry_family"] == "scene_target_render"
    assert row["operator_mode"] == operator_mode
    assert row["coverage_frac"] == pytest.approx(1.0)
    assert row["rmse_vs_truth"] == pytest.approx(0.0)
    assert row["background_leakage_fraction"] >= 0.0


def test_map_ground_truth_field_fit_row_is_finite_and_optimized(tmp_path: Path) -> None:
    row = _truth_bench.evaluate_torchsky_field_case(
        tmp_path,
        case="tan_aligned_low_noise",
        size=24,
        seed=1234,
        runs=1,
        field_model_kind="wire_implicit",
        fit_steps=4,
        max_observations=2,
    )

    assert row["library"] == "torchsky_field"
    assert row["method"] == "field_fit"
    assert row["variant"] == "wire_implicit"
    assert row["geometry_family"] == "latent_scene_field"
    assert row["fit_observations"] == 2
    assert row["field_final_loss"] <= row["field_initial_loss"]
    assert row["coverage_frac"] > 0.95
    assert row["rmse_vs_truth"] >= 0.0


@pytest.mark.parametrize(
    ("spherical_mode", "geometry_family"),
    [
        ("healpix_bridge", "spherical_spectral_healpix_bridge"),
        ("harmonic_direct", "spherical_harmonic_direct"),
    ],
)
def test_map_ground_truth_spherical_rows_are_finite(
    tmp_path: Path,
    spherical_mode: str,
    geometry_family: str,
) -> None:
    row = _truth_bench.evaluate_torchsky_spherical_case(
        tmp_path,
        case="tan_aligned_low_noise",
        size=32,
        seed=1234,
        runs=1,
        spherical_mode=spherical_mode,
    )

    assert row["library"] == "torchsky_spherical"
    assert row["method"] == "spherical_reproject"
    assert row["variant"] == spherical_mode
    assert row["geometry_family"] == geometry_family
    assert row["operator_bundle_geometry"] == "spherical_spectral_geometry"
    assert row["healpix_nside"] >= 1
    assert row["coverage_frac"] > 0.95
    assert row["rmse_vs_truth"] >= 0.0


def test_map_ground_truth_joint_teb_inverse_row_is_finite_and_improves_fit(
    tmp_path: Path,
) -> None:
    row = _truth_bench.evaluate_torchsky_joint_teb_case(
        tmp_path,
        case="tan_aligned_low_noise",
        size=32,
        seed=1234,
        runs=1,
        lmax=3,
    )

    assert row["library"] == "torchsky_spherical"
    assert row["method"] == "spherical_inverse"
    assert row["variant"] == "joint_teb_cl_prior"
    assert row["geometry_family"] == "spherical_harmonic_joint_teb"
    assert row["operator_bundle_geometry"] == "spherical_spectral_geometry"
    assert row["temp_observations"] == 2
    assert row["pol_observations"] == 2
    assert row["residual_norm_final"] < row["residual_norm_initial"]
    assert row["temp_rmse_mean"] >= 0.0
    assert row["pol_rmse_mean"] >= 0.0
    assert row["teb_cross_spectrum_prior_norm"] >= 0.0


def test_map_ground_truth_random10_large_union_metadata(tmp_path: Path) -> None:
    row = _truth_bench.evaluate_torchsky_case(
        tmp_path,
        case="tan_random10_large_union",
        size=256,
        seed=1234,
        runs=1,
    )
    assert row["n_obs"] == 10
    assert row["min_input_width"] >= 500
    assert row["max_input_width"] <= 4000
    assert row["min_input_height"] >= 500
    assert row["max_input_height"] <= 4000


def test_map_ground_truth_random10_rect_weighted_masked_metadata(
    tmp_path: Path,
) -> None:
    row = _truth_bench.evaluate_torchsky_case(
        tmp_path,
        case="tan_random10_large_union_rect_weighted_masked",
        size=320,
        width=320,
        height=240,
        seed=1234,
        runs=1,
    )
    assert row["n_obs"] == 10
    assert row["product_geometry"] == "rectangular"
    assert row["target_width"] == 320
    assert row["target_height"] == 240
    assert row["n_weight_maps"] == 10
    assert row["n_mask_maps"] == 10
    assert row["n_warped_inputs"] > 0
    assert row["mean_explicit_mask_fraction_input"] > 0.0
    assert row["background_residual_rms"] >= 0.0
    assert row["background_leakage_fraction"] >= 0.0


def test_tpv_roundtrip_validity_mask_bypasses_filtering_for_tpv(tmp_path: Path) -> None:
    observations, target_header, _truth = _truth_bench.make_case(
        tmp_path,
        case="tan_random10_large_union_rect_weighted_masked",
        size=256,
        width=256,
        height=192,
        seed=1234,
    )
    tpv_obs = next(
        item for item in observations if "TPV" in str(item.header.get("CTYPE1", ""))
    )
    target_wcs = _truth_bench.WCS(target_header)
    y, x = torch.meshgrid(
        torch.arange(target_header["NAXIS2"], dtype=torch.float64),
        torch.arange(target_header["NAXIS1"], dtype=torch.float64),
        indexing="ij",
    )
    ra, dec = target_wcs.pixel_to_world(x, y)
    flat_ra = ra.reshape(-1)
    flat_dec = dec.reshape(-1)
    source_wcs = _truth_bench.WCS(tpv_obs.header)
    x_src, y_src = source_wcs.world_to_pixel(flat_ra, flat_dec)
    x_src = x_src.reshape_as(ra)
    y_src = y_src.reshape_as(dec)
    valid = image_plane_projection_mod._roundtrip_validity_mask(
        source_wcs,
        x_src=x_src,
        y_src=y_src,
        flat_ra=flat_ra,
        flat_dec=flat_dec,
        image_shape=(int(tpv_obs.header["NAXIS2"]), int(tpv_obs.header["NAXIS1"])),
    )

    assert bool(valid.all())


def test_tpv_roundtrip_validity_gate_not_enabled_by_default(tmp_path: Path) -> None:
    observations, _target_header, _truth = _truth_bench.make_case(
        tmp_path,
        case="tan_random10_large_union_rect_weighted_masked",
        size=256,
        width=256,
        height=192,
        seed=1234,
    )
    tpv_obs = next(
        item for item in observations if "TPV" in str(item.header.get("CTYPE1", ""))
    )
    source_wcs = _truth_bench.WCS(tpv_obs.header)
    assert not image_plane_projection_mod._requires_roundtrip_validity_check(source_wcs)


def test_distorted_sources_enable_target_pixel_quadrature(tmp_path: Path) -> None:
    observations, _target_header, _truth = _truth_bench.make_case(
        tmp_path,
        case="tan_random10_large_union_rect_weighted_masked",
        size=256,
        width=256,
        height=192,
        seed=1234,
    )
    tpv_obs = next(
        item for item in observations if "TPV" in str(item.header.get("CTYPE1", ""))
    )
    tan_obs = next(
        item for item in observations if "TAN" in str(item.header.get("CTYPE1", ""))
    )
    assert image_plane_projection_mod._requires_target_pixel_quadrature(
        _truth_bench.WCS(tpv_obs.header),
        value_semantics="flux",
    )
    assert not image_plane_projection_mod._requires_target_pixel_quadrature(
        _truth_bench.WCS(tpv_obs.header),
        value_semantics="surface_brightness",
    )
    assert not image_plane_projection_mod._requires_target_pixel_quadrature(
        _truth_bench.WCS(tan_obs.header),
        value_semantics="flux",
    )


def test_prepare_target_caches_quadrature_world_grids(tmp_path: Path) -> None:
    observations, target_header, _truth = _truth_bench.make_case(
        tmp_path,
        case="tan_random10_large_union_rect_weighted_masked",
        size=256,
        width=256,
        height=192,
        seed=1234,
    )
    target = image_grid_mod._prepare_target(
        header=target_header,
        target_wcs=_truth_bench.WCS(target_header),
        target_shape=(int(target_header["NAXIS2"]), int(target_header["NAXIS1"])),
        device=torch.device("cpu"),
        include_quadrature=True,
    )
    assert target.quadrature_world_grids is not None
    assert len(target.quadrature_world_grids) == 4
    for ra, dec in target.quadrature_world_grids:
        assert tuple(ra.shape) == target.shape
        assert tuple(dec.shape) == target.shape


def test_prepare_target_skips_quadrature_world_grids_by_default(tmp_path: Path) -> None:
    _observations, target_header, _truth = _truth_bench.make_case(
        tmp_path,
        case="tan_random10_large_union_rect_weighted_masked",
        size=256,
        width=256,
        height=192,
        seed=1234,
    )
    target = image_grid_mod._prepare_target(
        header=target_header,
        target_wcs=_truth_bench.WCS(target_header),
        target_shape=(int(target_header["NAXIS2"]), int(target_header["NAXIS1"])),
        device=torch.device("cpu"),
    )
    assert target.quadrature_world_grids is None


def _count_distorted_quadrature_world_to_pixel_calls(
    tmp_path: Path, *, size: int, mapping: str
) -> int:
    tmp_path.mkdir(parents=True, exist_ok=True)
    observations, target_header, _truth = _truth_bench.make_case(
        tmp_path,
        case="tan_random10_large_union_rect_weighted_masked",
        size=size,
        width=size,
        height=size * 3 // 4,
        seed=1234,
    )
    tpv_obs = next(
        item for item in observations if "TPV" in str(item.header.get("CTYPE1", ""))
    )
    image, _header = torchfits.read(
        tpv_obs.observation.path,
        hdu=tpv_obs.observation.hdu,
        return_header=True,
    )
    image = validate_image_plane_2d(image).to(dtype=torch.float32)
    source_wcs = _truth_bench.WCS(tpv_obs.header)
    prepared_target = image_grid_mod._prepare_target(
        header=target_header,
        target_wcs=_truth_bench.WCS(target_header),
        target_shape=(int(target_header["NAXIS2"]), int(target_header["NAXIS1"])),
        device=torch.device("cpu"),
    )
    source_weight = calibration_mod._load_observation_weight_map(
        tpv_obs.observation,
        image,
        device=torch.device("cpu"),
    )

    call_count = 0
    orig_world_to_pixel = source_wcs.world_to_pixel

    def _counted_world_to_pixel(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig_world_to_pixel(*args, **kwargs)

    source_wcs.world_to_pixel = _counted_world_to_pixel
    try:
        with interpolated_mapping_mod.mapping_mode(mapping):
            image_plane_projection_mod._resample_weighted_to_target(
                image,
                source_weight,
                source_wcs,
                prepared_target,
                resampling_kernel="bilinear",
                value_semantics="flux",
            )
    finally:
        source_wcs.world_to_pixel = orig_world_to_pixel
    return call_count


def test_distorted_quadrature_batches_world_to_pixel_calls(tmp_path: Path) -> None:
    """Exact mapping: every quadrature sub-grid goes through one batched call."""
    assert (
        _count_distorted_quadrature_world_to_pixel_calls(
            tmp_path / "exact", size=256, mapping="exact"
        )
        == 1
    )


def test_interpolated_mapping_call_count_does_not_grow_with_the_image(
    tmp_path: Path,
) -> None:
    """Interpolated mapping (the default for TPV/SIP since torchsky 0.5.0): the
    verified lattice is a fixed chain of batched solves, so the number of
    world->pixel calls is independent of the image size (12 at 128-512 px)."""
    small = _count_distorted_quadrature_world_to_pixel_calls(
        tmp_path / "small", size=128, mapping="interpolated"
    )
    large = _count_distorted_quadrature_world_to_pixel_calls(
        tmp_path / "large", size=512, mapping="interpolated"
    )
    assert small == large
    assert large <= 16


pytestmark = pytest.mark.slow


def test_random10_rect_tpv_headers_are_locally_invertible(tmp_path: Path) -> None:
    observations, _target_header, _truth = _truth_bench.make_case(
        tmp_path,
        case="tan_random10_large_union_rect_weighted_masked",
        size=512,
        width=512,
        height=384,
        seed=1234,
    )
    for item in observations:
        if "TPV" not in str(item.header.get("CTYPE1", "")):
            continue
        wcs = _truth_bench.WCS(item.header)
        height = int(item.header["NAXIS2"])
        width = int(item.header["NAXIS1"])
        ys, xs = torch.meshgrid(
            torch.linspace(64, max(64, height - 64), 17, dtype=torch.float64),
            torch.linspace(64, max(64, width - 64), 17, dtype=torch.float64),
            indexing="ij",
        )
        rel_x = xs.reshape(-1) - wcs._crpix0
        rel_y = ys.reshape(-1) - wcs._crpix1
        u, v = wcs._apply_cd_2d(rel_x, rel_y)
        _xi, _eta, j11, j12, j21, j22 = wcs.tpv._distort_and_jacobian(u, v)
        det = j11 * j22 - j12 * j21
        assert bool((det > 0).all() or (det < 0).all())
