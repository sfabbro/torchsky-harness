"""Megapipe-scale benchmark: torchsky vs SWarp on CFHT-representative data."""

from __future__ import annotations
import torchfits

from pathlib import Path

import pytest
import torch

pytest.importorskip("torchsky._cpp")

from torchsky.map import coadd as coadd_image_plane
from torchsky.scene import ImageObservation

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"

import torchsky_harness.benchmarks.bench_map_compat as bench_map_compat  # noqa: E402

_CFHT_SIZE = 1024
_CFHT_N_OBS = 5
_SPEED_FACTOR_MIN = 2.0
_SPEED_RUNS = 3
_QUALITY_TOLERANCE = 0.5
_BORDER_CROP = 50


def _cfht_scale_header(h: int, w: int) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": w,
        "NAXIS2": h,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": w / 2.0,
        "CRPIX2": h / 2.0,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CD1_1": -2.77778e-05,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 2.77778e-05,
        "RADESYS": "ICRS",
        "EQUINOX": 2000.0,
    }


@pytest.mark.skipif(
    bench_map_compat._find_swarp_binary() is None,
    reason="SWarp binary not installed or not on PATH",
)
def test_megapipe_scale_torchsky_beats_swarp_on_speed(tmp_path: Path) -> None:
    h = w = _CFHT_SIZE
    obs_list: list[ImageObservation] = []
    for i in range(_CFHT_N_OBS):
        hdr = dict(_cfht_scale_header(h, w))
        dither_pix = (i - _CFHT_N_OBS // 2) * 3.0
        hdr["CRPIX1"] = float(hdr["CRPIX1"]) + dither_pix
        hdr["CRPIX2"] = float(hdr["CRPIX2"]) + dither_pix * 0.7

        img = torch.ones(h, w, dtype=torch.float32) * 100.0
        wt = torch.ones(h, w, dtype=torch.float32)

        path = tmp_path / f"obs_{i}.fits"
        wt_path = tmp_path / f"obs_{i}_wt.fits"
        torchfits.write(str(path), img, header=hdr, overwrite=True)
        torchfits.write(str(wt_path), wt, header=hdr, overwrite=True)
        obs_list.append(ImageObservation(str(path), weight_path=str(wt_path)))

    target_header = dict(_cfht_scale_header(h, w))

    import time

    # Best of _SPEED_RUNS for each tool: a single cold sample swings by more
    # than the margin (SWarp 1.2-1.7 s, torchsky first-call setup), so the gate
    # compares steady-state throughput, not one noisy run.
    # Same CPU budget for both tools: SWarp runs with NTHREADS 0 (every core),
    # while conftest pins torch to one thread for determinism.
    import os

    threads_before = torch.get_num_threads()
    torch.set_num_threads(os.cpu_count() or 1)
    try:
        t_torchsky = float("inf")
        for _ in range(_SPEED_RUNS):
            t0 = time.perf_counter()
            ts = coadd_image_plane(
                obs_list,
                target=target_header,
                kernel="bilinear",
            )
            t_torchsky = min(t_torchsky, time.perf_counter() - t0)
    finally:
        torch.set_num_threads(threads_before)

    t_swarp = float("inf")
    for _ in range(_SPEED_RUNS):
        t0 = time.perf_counter()
        sw_data, sw_w = bench_map_compat._coadd_with_swarp(
            obs_list, target_header=target_header, resampling_type="BILINEAR"
        )
        t_swarp = min(t_swarp, time.perf_counter() - t0)

    speed_ratio = t_swarp / t_torchsky
    mpix = h * w / 1e6
    print(
        f"\nMegapipe-scale ({h}x{w}, {_CFHT_N_OBS} obs, {os.cpu_count()} threads):"
        f"\n  torchsky: {t_torchsky * 1000:.1f}ms ({mpix / t_torchsky:.1f} mpix/s)"
        f"\n  SWarp:    {t_swarp * 1000:.1f}ms ({mpix / t_swarp:.1f} mpix/s)"
        f"\n  speed ratio: {speed_ratio:.1f}x"
    )
    assert speed_ratio >= _SPEED_FACTOR_MIN, (
        f"torchsky too slow: {speed_ratio:.1f}x vs SWarp "
        f"(threshold {_SPEED_FACTOR_MIN}x)"
    )

    inner_y = slice(_BORDER_CROP, h - _BORDER_CROP)
    inner_x = slice(_BORDER_CROP, w - _BORDER_CROP)
    mask = (ts.weight[inner_y, inner_x] > 0.0) & torch.from_numpy(
        sw_w[inner_y, inner_x] > 0.0
    ).to(ts.weight.device)
    assert mask.any(), "no overlapping pixels in inner region"
    td = ts.data[inner_y, inner_x][mask].cpu()
    sd = torch.from_numpy(sw_data[inner_y, inner_x][mask.cpu().numpy()]).to(td.device)
    max_diff = float((td - sd).abs().max().item())
    rmse = float((td - sd).pow(2).mean().sqrt().item())
    print(f"  quality: max_diff={max_diff:.4f}  rmse={rmse:.4f}")
    assert max_diff < _QUALITY_TOLERANCE, (
        f"torchsky and SWarp disagree: max_diff={max_diff:.4f} "
        f"(tolerance {_QUALITY_TOLERANCE})"
    )


@pytest.mark.benchmark
def test_megapipe_scale_torchsky_no_oom(tmp_path: Path) -> None:
    h, w = 2048, 4612
    n_obs = 3
    obs_list: list[ImageObservation] = []
    for i in range(n_obs):
        hdr = dict(_cfht_scale_header(h, w))
        dither_pix = (i - n_obs // 2) * 3.0
        hdr["CRPIX1"] = float(hdr["CRPIX1"]) + dither_pix

        img = torch.ones(h, w, dtype=torch.float32) * 100.0
        img += torch.randn(h, w, dtype=torch.float32) * 5.0
        wt = torch.ones(h, w, dtype=torch.float32)

        path = tmp_path / f"obs_{i}.fits"
        wt_path = tmp_path / f"obs_{i}_wt.fits"
        torchfits.write(str(path), img, header=hdr, overwrite=True)
        torchfits.write(str(wt_path), wt, header=hdr, overwrite=True)
        obs_list.append(ImageObservation(str(path), weight_path=str(wt_path)))

    target_header = dict(_cfht_scale_header(h, w))
    result = coadd_image_plane(
        obs_list,
        target=target_header,
        kernel="bilinear",
    )
    assert result.data.shape == (h, w)
    assert result.weight.shape == (h, w)
