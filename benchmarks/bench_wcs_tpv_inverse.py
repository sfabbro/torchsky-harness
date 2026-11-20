#!/usr/bin/env python3
"""Dedicated large-N TPV world-to-pixel benchmark.

Compares torchsky's TPV inverse (the pure-torch tensor-core Newton solve that
backs the public ``WCS.world_to_pixel``) against Astropy ``all_world2pix``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from astropy.wcs import WCS as AstropyWCS

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from torchsky_harness.benchmarks.bench_wcs import _resolve_device, _sync  # noqa: E402
from torchsky import WCS as TorchWCS


@dataclass(frozen=True)
class CaseSpec:
    name: str
    header: dict
    description: str


def _tpv_nonlinear_header() -> dict:
    header = dict()
    header["NAXIS"] = 2
    header["NAXIS1"] = 2048
    header["NAXIS2"] = 1536
    header["CTYPE1"] = "RA---TPV"
    header["CTYPE2"] = "DEC--TPV"
    header["CRPIX1"] = 1024.0
    header["CRPIX2"] = 768.0
    header["CRVAL1"] = 12.0
    header["CRVAL2"] = -3.0
    header["CD1_1"] = -0.004465
    header["CD1_2"] = 0.0005488
    header["CD2_1"] = 0.0005488
    header["CD2_2"] = 0.004465
    header["PV1_0"] = 0.0
    header["PV1_1"] = 1.0
    header["PV1_2"] = -0.02
    header["PV1_4"] = 2.0e-4 * 0.22
    header["PV1_5"] = -3.0e-4 * 0.22
    header["PV1_6"] = 1.5e-4 * 0.22
    header["PV1_7"] = 2.0e-6 * 0.22
    header["PV1_39"] = 2.0e-11 * 0.22
    header["PV2_0"] = 0.0
    header["PV2_1"] = 0.02
    header["PV2_2"] = 1.0
    header["PV2_4"] = -1.0e-4 * 0.22
    header["PV2_5"] = 2.5e-4 * 0.22
    header["PV2_6"] = -2.0e-4 * 0.22
    header["PV2_7"] = -1.5e-6 * 0.22
    header["PV2_39"] = -1.0e-11 * 0.22
    return header


def _tpv_affine_header() -> dict:
    header = _tpv_nonlinear_header()
    for key in list(header.keys()):
        if key.startswith("PV1_") or key.startswith("PV2_"):
            del header[key]
    header["PV1_0"] = 0.05
    header["PV1_1"] = 1.12
    header["PV1_2"] = -0.08
    header["PV2_0"] = -0.03
    header["PV2_1"] = 0.04
    header["PV2_2"] = 0.91
    return header


CASES: dict[str, CaseSpec] = {
    "tpv_affine": CaseSpec(
        name="tpv_affine",
        header=_tpv_affine_header(),
        description="Affine-only TPV inverse",
    ),
    "tpv_nonlinear": CaseSpec(
        name="tpv_nonlinear",
        header=_tpv_nonlinear_header(),
        description="Non-linear broad-field TPV inverse",
    ),
}


def _sample_pixels(
    header: dict, n_points: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    width = int(header["NAXIS1"])
    height = int(header["NAXIS2"])
    margin_x = min(64.0, max(width / 16.0, 8.0))
    margin_y = min(64.0, max(height / 16.0, 8.0))
    x = rng.uniform(
        margin_x, max(margin_x + 1.0, width - 1.0 - margin_x), size=n_points
    )
    y = rng.uniform(
        margin_y, max(margin_y + 1.0, height - 1.0 - margin_y), size=n_points
    )
    return x.astype(np.float64), y.astype(np.float64)


def _time_many(
    fn, runs: int, sync_device: torch.device | None = None
) -> tuple[float, Any]:
    result = fn()
    if sync_device is not None:
        _sync(sync_device)
    samples: list[float] = []
    for _ in range(int(runs)):
        t0 = time.perf_counter()
        result = fn()
        if sync_device is not None:
            _sync(sync_device)
        samples.append(time.perf_counter() - t0)
    return float(min(samples)), result


def _error_summary(
    x_true: np.ndarray, y_true: np.ndarray, x_pred: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    dx = x_pred - x_true
    dy = y_pred - y_true
    err = np.sqrt(dx * dx + dy * dy)
    return {
        "mean_inverse_pixel_error": float(np.mean(err)),
        "p99_inverse_pixel_error": float(np.percentile(err, 99.0)),
        "max_inverse_pixel_error": float(np.max(err)),
    }


def _bench_torchsky(
    *,
    header: dict,
    ra_t: torch.Tensor,
    dec_t: torch.Tensor,
    x_true: np.ndarray,
    y_true: np.ndarray,
    runs: int,
    device: torch.device,
) -> dict[str, Any]:
    twcs = TorchWCS(dict(header)).to(device)

    best_s, (x_pred_t, y_pred_t) = _time_many(
        lambda: twcs.world_to_pixel(ra_t, dec_t, origin=0),
        runs=runs,
        sync_device=device,
    )
    x_pred = x_pred_t.detach().cpu().numpy()
    y_pred = y_pred_t.detach().cpu().numpy()
    row = {
        "library": "torchsky",
        "method": "tensor_core",
        "status": "OK",
        "device": str(device),
        "time_ms": best_s * 1000.0,
        "mpts_s": float(ra_t.numel() / max(best_s, 1.0e-12) / 1.0e6),
    }
    row.update(_error_summary(x_true, y_true, x_pred, y_pred))
    twcs.tpv.set_invert_trace(True)
    twcs.world_to_pixel(ra_t, dec_t, origin=0)
    trace = twcs.tpv.get_last_invert_trace() or {}
    twcs.tpv.set_invert_trace(False)
    row["mean_point_iterations"] = float(trace.get("mean_point_iterations", 0.0))
    row["iterations_max"] = int(trace.get("iterations_max", trace.get("iterations", 0)))
    return row


def _bench_astropy_method(
    *,
    awcs: AstropyWCS,
    ra_np: np.ndarray,
    dec_np: np.ndarray,
    x_true: np.ndarray,
    y_true: np.ndarray,
    runs: int,
) -> dict[str, Any]:
    try:
        best_s, (x_pred, y_pred) = _time_many(
            lambda: awcs.all_world2pix(ra_np, dec_np, 0),
            runs=runs,
        )
    except Exception as exc:
        return {
            "library": "astropy",
            "method": "all_world2pix",
            "status": "SKIPPED",
            "skip_reason": str(exc),
            "time_ms": None,
            "mpts_s": None,
            "mean_inverse_pixel_error": None,
            "p99_inverse_pixel_error": None,
            "max_inverse_pixel_error": None,
            "mean_point_iterations": None,
            "iterations_max": None,
        }

    row = {
        "library": "astropy",
        "method": "all_world2pix",
        "status": "OK",
        "time_ms": best_s * 1000.0,
        "mpts_s": float(ra_np.size / max(best_s, 1.0e-12) / 1.0e6),
        "mean_point_iterations": None,
        "iterations_max": None,
    }
    row.update(_error_summary(x_true, y_true, np.asarray(x_pred), np.asarray(y_pred)))
    return row


def _format_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "case",
        "library",
        "method",
        "status",
        "n_points",
        "time_ms",
        "mpts_s",
        "mean_inverse_pixel_error",
        "p99_inverse_pixel_error",
        "max_inverse_pixel_error",
        "mean_point_iterations",
        "iterations_max",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals: list[str] = []
        for key in headers:
            value = row.get(key)
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            elif value is None:
                vals.append("n/a")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="tpv_affine,tpv_nonlinear")
    parser.add_argument("--n-points", type=int, default=1000000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/wcs_tpv_inverse.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("benchmark_results/wcs_tpv_inverse.md"),
    )
    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []

    for case_name in [part.strip() for part in args.cases.split(",") if part.strip()]:
        spec = CASES[case_name]
        header = spec.header
        awcs = AstropyWCS(header)
        x_true, y_true = _sample_pixels(header, args.n_points, rng)
        ra_np, dec_np = awcs.all_pix2world(x_true, y_true, 0)
        ra_t = torch.from_numpy(np.asarray(ra_np, dtype=np.float64)).to(device)
        dec_t = torch.from_numpy(np.asarray(dec_np, dtype=np.float64)).to(device)

        torchsky_row = _bench_torchsky(
            header=header,
            ra_t=ra_t,
            dec_t=dec_t,
            x_true=x_true,
            y_true=y_true,
            runs=args.runs,
            device=device,
        )
        astropy_row = _bench_astropy_method(
            awcs=awcs,
            ra_np=np.asarray(ra_np, dtype=np.float64),
            dec_np=np.asarray(dec_np, dtype=np.float64),
            x_true=x_true,
            y_true=y_true,
            runs=args.runs,
        )
        astropy_row["device"] = "cpu"
        for row in (torchsky_row, astropy_row):
            row["case"] = spec.name
            row["description"] = spec.description
            row["n_points"] = int(args.n_points)
            rows.append(row)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    args.markdown_out.write_text(_format_markdown(rows), encoding="utf-8")
    print(args.json_out)
    print(args.markdown_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
