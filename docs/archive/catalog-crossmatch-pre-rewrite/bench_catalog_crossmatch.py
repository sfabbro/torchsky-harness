#!/usr/bin/env python3
"""Benchmark catalog crossmatch candidate-generation strategies."""

from __future__ import annotations

from pathlib import Path
import time  # noqa: E402
import argparse  # noqa: E402
import json  # noqa: E402
from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import torchsky.catalogs.core as catalogs_core  # noqa: E402
from torchsky.catalogs import crossmatch_sky  # noqa: E402
from torchsky_harness.benchmarks.config import DEFAULT_OUTPUT_DIR


def _time_many(
    fn: Callable[[], Any], runs: int, device: torch.device | None = None
) -> float:
    fn()
    if device is not None:
        _sync(device)
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if device is not None:
            _sync(device)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if choice == "mps":
        raise RuntimeError("catalog benches need float64; MPS is float32-only")
    return torch.device(choice)


def _random_sphere(
    rng: np.random.Generator,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    lon = rng.uniform(0.0, 360.0, size=n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    return lon.astype(np.float64), lat.astype(np.float64)


def _make_catalogs(
    *,
    n_left: int,
    n_right_band: int,
    n_right_background: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    left_center_ra = 42.0
    left_center_dec = 4.0

    left_ra = left_center_ra + rng.normal(0.0, 0.12, size=n_left)
    left_dec = left_center_dec + rng.normal(0.0, 0.08, size=n_left)
    left_ra = np.remainder(left_ra, 360.0).astype(np.float64)
    left_dec = np.clip(left_dec, -85.0, 85.0).astype(np.float64)

    matched = min(max(16, n_left // 2), n_left)
    matched_ra = np.remainder(
        left_ra[:matched] + rng.normal(0.0, 0.00008, size=matched), 360.0
    )
    matched_dec = np.clip(
        left_dec[:matched] + rng.normal(0.0, 0.00008, size=matched),
        -89.999,
        89.999,
    )

    band_ra = rng.uniform(0.0, 360.0, size=n_right_band)
    band_dec = left_center_dec + rng.normal(0.0, 0.12, size=n_right_band)
    bg_ra, bg_dec = _random_sphere(rng, n_right_background)

    right_ra = np.concatenate((matched_ra, band_ra, bg_ra)).astype(np.float64)
    right_dec = np.concatenate(
        (
            matched_dec,
            np.clip(band_dec, -89.999, 89.999),
            np.clip(bg_dec, -89.999, 89.999),
        )
    ).astype(np.float64)

    return (
        torch.from_numpy(left_ra),
        torch.from_numpy(left_dec),
        torch.from_numpy(right_ra),
        torch.from_numpy(right_dec),
    )


def _assert_same_matches(
    got: catalogs_core.CrossmatchResult,
    expected: catalogs_core.CrossmatchResult,
) -> None:
    if got.left_index.tolist() != expected.left_index.tolist():
        raise RuntimeError("left_index mismatch between crossmatch strategies")
    if got.right_index.tolist() != expected.right_index.tolist():
        raise RuntimeError("right_index mismatch between crossmatch strategies")
    torch.testing.assert_close(
        got.separation_arcsec,
        expected.separation_arcsec,
        atol=1e-9,
        rtol=0.0,
    )


def _recommendation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decl = next(
        (
            row
            for row in rows
            if row.get("operation") == "crossmatch_declination_pruned"
        ),
        None,
    )
    coarse = next(
        (row for row in rows if row.get("operation") == "crossmatch_coarse_healpix"),
        None,
    )
    indexed = next(
        (
            row
            for row in rows
            if row.get("operation") == "crossmatch_declination_indexed"
        ),
        None,
    )
    public = next(
        (row for row in rows if row.get("operation") == "crossmatch_sky"),
        None,
    )
    if coarse is None or indexed is None:
        return []
    candidates = [row for row in (decl, coarse, indexed, public) if row is not None]
    recommended_row = min(candidates, key=lambda row: float(row["time_s"]))
    recommended = str(recommended_row["operation"])
    note = "Lowest measured exact candidate-generation time on this workload"
    return [
        {
            "operation": "crossmatch_recommendation",
            "library": "torchsky_recommendation",
            "recommended_operation": recommended,
            "time_s_declination": None if decl is None else decl["time_s"],
            "time_s_coarse": coarse["time_s"],
            "time_s_indexed": indexed["time_s"],
            "time_s_public": None if public is None else public["time_s"],
            "candidate_row_reduction_vs_declination": coarse.get(
                "candidate_row_reduction_vs_declination",
                float("nan"),
            ),
            "pair_reduction_vs_declination": coarse.get(
                "pair_reduction_vs_declination",
                float("nan"),
            ),
            "coarse_nside": coarse.get("coarse_nside"),
            "note": note,
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-left", type=int, default=20_000)
    parser.add_argument("--n-right-band", type=int, default=250_000)
    parser.add_argument("--n-right-background", type=int, default=50_000)
    parser.add_argument("--radius-arcsec", type=float, default=2.0)
    parser.add_argument("--left-chunk-size", type=int, default=4096)
    parser.add_argument("--right-chunk-size", type=int, default=8192)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--skip-decl-pruned",
        action="store_true",
        help="Skip the dense declination-pruned path (use for N>=1e6–1e7 CANFAR rows).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "catalog_crossmatch.json",
    )
    args = parser.parse_args()
    device = _resolve_device(args.device)

    left_ra, left_dec, right_ra, right_dec = _make_catalogs(
        n_left=args.n_left,
        n_right_band=args.n_right_band,
        n_right_background=args.n_right_background,
        seed=args.seed,
    )
    left_ra = left_ra.to(device)
    left_dec = left_dec.to(device)
    right_ra = right_ra.to(device)
    right_dec = right_dec.to(device)
    right_points = (
        args.n_right_band
        + args.n_right_background
        + min(max(16, args.n_left // 2), args.n_left)
    )
    match_kwargs = dict(
        radius_arcsec=args.radius_arcsec,
        left_chunk_size=args.left_chunk_size,
        right_chunk_size=args.right_chunk_size,
    )
    radius_left = torch.full(
        (int(left_ra.numel()),),
        float(args.radius_arcsec),
        dtype=left_ra.dtype,
        device=left_ra.device,
    )
    indexed_kwargs = dict(
        radius_arcsec=radius_left,
        left_chunk_size=args.left_chunk_size,
        right_chunk_size=args.right_chunk_size,
    )

    coarse_result, coarse_stats = catalogs_core._crossmatch_sky_coarse_indexed(
        left_ra, left_dec, right_ra, right_dec, return_stats=True, **match_kwargs
    )
    indexed_result, indexed_stats = catalogs_core._crossmatch_sky_declination_indexed(
        left_ra, left_dec, right_ra, right_dec, return_stats=True, **indexed_kwargs
    )
    public_result = crossmatch_sky(
        left_ra, left_dec, right_ra, right_dec, **match_kwargs
    )
    coarse_nside = catalogs_core._choose_default_coarse_nside(
        radius_arcsec=args.radius_arcsec
    )
    _assert_same_matches(indexed_result, coarse_result)
    _assert_same_matches(public_result, indexed_result)

    decl_stats = None
    t_decl = float("nan")
    if not args.skip_decl_pruned:
        decl_result, decl_stats = catalogs_core._crossmatch_sky_declination_pruned(
            left_ra, left_dec, right_ra, right_dec, return_stats=True, **match_kwargs
        )
        _assert_same_matches(coarse_result, decl_result)
        t_decl = _time_many(
            lambda: catalogs_core._crossmatch_sky_declination_pruned(
                left_ra, left_dec, right_ra, right_dec, **match_kwargs
            ),
            runs=args.runs,
            device=device,
        )

    t_coarse = _time_many(
        lambda: catalogs_core._crossmatch_sky_coarse_indexed(
            left_ra, left_dec, right_ra, right_dec, **match_kwargs
        ),
        runs=args.runs,
        device=device,
    )
    t_indexed = _time_many(
        lambda: catalogs_core._crossmatch_sky_declination_indexed(
            left_ra, left_dec, right_ra, right_dec, **indexed_kwargs
        ),
        runs=args.runs,
        device=device,
    )
    t_public = _time_many(
        lambda: crossmatch_sky(left_ra, left_dec, right_ra, right_dec, **match_kwargs),
        runs=args.runs,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    if decl_stats is not None:
        rows.append(
            {
                "operation": "crossmatch_declination_pruned",
                "library": "torchsky",
                "left_points": args.n_left,
                "right_points": right_points,
                "radius_arcsec": args.radius_arcsec,
                "time_s": t_decl,
                "crossmatches_s": args.n_left / t_decl,
                "matched_rows": decl_stats.matched_rows,
                "candidate_groups": decl_stats.candidate_groups,
                "candidate_rows_considered": decl_stats.candidate_rows_considered,
                "pairs_evaluated": decl_stats.pairs_evaluated,
            }
        )
    coarse_row: dict[str, Any] = {
        "operation": "crossmatch_coarse_healpix",
        "library": "torchsky",
        "left_points": args.n_left,
        "right_points": right_points,
        "radius_arcsec": args.radius_arcsec,
        "time_s": t_coarse,
        "crossmatches_s": args.n_left / t_coarse,
        "matched_rows": coarse_stats.matched_rows,
        "candidate_groups": coarse_stats.candidate_groups,
        "candidate_rows_considered": coarse_stats.candidate_rows_considered,
        "pairs_evaluated": coarse_stats.pairs_evaluated,
        "coarse_nside": coarse_nside,
        "speedup_vs_declination": (
            t_decl / t_coarse if np.isfinite(t_decl) else float("nan")
        ),
    }
    if decl_stats is not None:
        coarse_row["candidate_row_reduction_vs_declination"] = (
            decl_stats.candidate_rows_considered
            / max(coarse_stats.candidate_rows_considered, 1)
        )
        coarse_row["pair_reduction_vs_declination"] = decl_stats.pairs_evaluated / max(
            coarse_stats.pairs_evaluated, 1
        )
    rows.append(coarse_row)
    indexed_row: dict[str, Any] = {
        "operation": "crossmatch_declination_indexed",
        "library": "torchsky",
        "left_points": args.n_left,
        "right_points": right_points,
        "radius_arcsec": args.radius_arcsec,
        "time_s": t_indexed,
        "crossmatches_s": args.n_left / t_indexed,
        "matched_rows": indexed_stats.matched_rows,
        "candidate_groups": indexed_stats.candidate_groups,
        "candidate_rows_considered": indexed_stats.candidate_rows_considered,
        "pairs_evaluated": indexed_stats.pairs_evaluated,
        "speedup_vs_declination": (
            t_decl / t_indexed if np.isfinite(t_decl) else float("nan")
        ),
        "speedup_vs_coarse": t_coarse / t_indexed,
    }
    if decl_stats is not None:
        indexed_row["pair_reduction_vs_declination"] = decl_stats.pairs_evaluated / max(
            indexed_stats.pairs_evaluated, 1
        )
    rows.append(indexed_row)
    rows.append(
        {
            "operation": "crossmatch_sky",
            "library": "torchsky",
            "left_points": args.n_left,
            "right_points": right_points,
            "radius_arcsec": args.radius_arcsec,
            "time_s": t_public,
            "crossmatches_s": args.n_left / t_public,
            "speedup_vs_coarse": t_coarse / t_public,
            "speedup_vs_indexed": t_indexed / t_public,
        }
    )
    rows.extend(_recommendation_rows(rows))
    for row in rows:
        row["device"] = str(device)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2))

    print("Catalog crossmatch benchmark")
    if decl_stats is not None:
        print(
            f"  declination_pruned: {t_decl:.4f}s "
            f"pairs={decl_stats.pairs_evaluated} "
            f"candidate_rows={decl_stats.candidate_rows_considered}"
        )
    print(
        f"  coarse_healpix:     {t_coarse:.4f}s "
        f"pairs={coarse_stats.pairs_evaluated} "
        f"candidate_rows={coarse_stats.candidate_rows_considered} "
        f"nside={coarse_nside}"
    )
    print(
        f"  declination_indexed:{t_indexed:.4f}s "
        f"pairs={indexed_stats.pairs_evaluated} "
        f"candidate_rows={indexed_stats.candidate_rows_considered} "
        f"speedup_vs_coarse={t_coarse / t_indexed:.2f}x"
    )
    print(
        f"  crossmatch_sky:     {t_public:.4f}s "
        f"speedup_vs_coarse={t_coarse / t_public:.2f}x"
    )
    recommendation = rows[-1]
    print(
        f"  recommendation:    {recommendation['recommended_operation']} "
        f"({recommendation['note']})"
    )
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
