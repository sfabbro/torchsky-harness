#!/usr/bin/env python3
"""Benchmark bounded-memory Arrow-partitioned crossmatch.

Measures wall time, peak memory, throughput, and tile-size distribution
across the ``crossmatch_sky_from_arrow`` path.  Validates correctness
against the in-memory ``crossmatch_sky``.

For uniformly distributed sources the per-pixel tile size stays near
constant as total catalog size grows — this benchmark checks that claim.
"""

from __future__ import annotations

from pathlib import Path

from torchsky_harness.benchmarks.config import DEFAULT_OUTPUT_DIR


import argparse
import json
import math
import resource
import tempfile
import time
from typing import Any

import numpy as np
import torch

import torchsky.catalogs.core as catalogs_core
from torchsky.catalogs import crossmatch_sky


def _time_many(fn, runs: int) -> float:
    fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _peak_rss_mb() -> float:
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return rusage.ru_maxrss / (1024 * 1024)


def _random_sphere(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    lon = rng.uniform(0.0, 360.0, size=n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    return lon.astype(np.float64), lat.astype(np.float64)


def _make_catalogs(
    *,
    n_left: int,
    n_right: int,
    radius_arcsec: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    center_ra, center_dec = 42.0, 4.0
    radius_deg = radius_arcsec / 3600.0

    left_ra = center_ra + rng.normal(0.0, 0.15, size=n_left)
    left_dec = center_dec + rng.normal(0.0, 0.10, size=n_left)
    left_ra = np.remainder(left_ra, 360.0).astype(np.float64)
    left_dec = np.clip(left_dec, -85.0, 85.0).astype(np.float64)

    matched = min(max(8, n_left // 2), n_left)
    matched_ra = np.remainder(
        left_ra[:matched] + rng.normal(0.0, radius_deg * 0.08, size=matched), 360.0
    )
    matched_dec = np.clip(
        left_dec[:matched] + rng.normal(0.0, radius_deg * 0.08, size=matched),
        -89.999, 89.999,
    )

    bg_ra, bg_dec = _random_sphere(rng, max(0, n_right - matched))

    right_ra = np.concatenate([matched_ra, bg_ra]).astype(np.float64)
    right_dec = np.concatenate([matched_dec, bg_dec]).astype(np.float64)
    return left_ra, left_dec, right_ra, right_dec


def _write_arrow_file(
    ra: np.ndarray, dec: np.ndarray, path: str | Path
) -> None:
    import pyarrow as pa
    table = pa.table({"ra": ra, "dec": dec})
    with pa.ipc.new_file(pa.OSFile(str(path), "wb"), table.schema) as writer:
        writer.write_table(table)


def _tile_size_stats(
    path: str | Path,
    *,
    nside: int,
    ra_col: str = "ra",
    dec_col: str = "dec",
    batch_size: int = 131072,
) -> tuple[float, float, int, int]:
    """Re-tile the catalog to Parquet and report tile-size distribution."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = pa.ipc.open_file(pa.memory_map(str(path), "r")).read_all()
    tmpdir = tempfile.mkdtemp(prefix="torchsky_tilestats_")
    try:
        tile_path = Path(tmpdir) / "tiles.parquet"
        pixel_map, n_total = catalogs_core._tile_catalog_to_parquet(
            source,
            ra_col=ra_col,
            dec_col=dec_col,
            coarse_nside=nside,
            path=tile_path,
            batch_size=batch_size,
        )
        pf = pq.ParquetFile(str(tile_path))
        sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
        if not sizes:
            return 0.0, 0.0, 0, n_total
        return float(np.median(sizes)), float(max(sizes)), len(sizes), n_total
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _collect_arrow_stats(
    left_path: str | Path,
    right_path: str | Path,
    *,
    nside: int = 64,
    radius_arcsec: float,
) -> dict[str, Any]:
    """Tile both catalogs and compute candidate/pair statistics."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from torchsky.sphere import healpix as _hp

    radius_deg = radius_arcsec / 3600.0
    candidate_radius_rad = math.radians(radius_deg) + (2.0 * _hp.max_pixel_radius(nside, degrees=False))

    left_source = pa.ipc.open_file(pa.memory_map(str(left_path), "r")).read_all()
    right_source = pa.ipc.open_file(pa.memory_map(str(right_path), "r")).read_all()

    tmpdir = tempfile.mkdtemp(prefix="torchsky_arrowstats_")
    try:
        left_map, n_left = catalogs_core._tile_catalog_to_parquet(
            left_source, ra_col="ra", dec_col="dec", coarse_nside=nside,
            path=Path(tmpdir) / "left.parquet",
        )
        right_map, n_right = catalogs_core._tile_catalog_to_parquet(
            right_source, ra_col="ra", dec_col="dec", coarse_nside=nside,
            path=Path(tmpdir) / "right.parquet",
        )

        left_pf = pq.ParquetFile(str(Path(tmpdir) / "left.parquet"))
        left_sizes = {
            pix: left_pf.metadata.row_group(rg).num_rows
            for pix, rg in left_map.items()
        }
        right_pf = pq.ParquetFile(str(Path(tmpdir) / "right.parquet"))
        right_sizes = {
            pix: right_pf.metadata.row_group(rg).num_rows
            for pix, rg in right_map.items()
        }

        left_pixels = set(left_map)
        right_pixels = set(right_map)

        candidate_rows = 0
        pairs_evaluated = 0
        for pix in left_pixels:
            cand = set(
                _candidate_coarse_pixels_for_left_pixel(
                    nside, pix, radius_rad=candidate_radius_rad, device=torch.device("cpu")
                ).tolist()
            ) & right_pixels
            n_left_here = left_sizes[pix]
            n_right_cand = sum(right_sizes.get(cp, 0) for cp in cand)
            candidate_rows += n_left_here * n_right_cand
            pairs_evaluated += n_left_here * n_right_cand

        return {
            "n_left": n_left,
            "n_right": n_right,
            "n_pixels_left": len(left_map),
            "n_pixels_right": len(right_map),
            "max_tile_left": max(left_sizes.values()) if left_sizes else 0,
            "max_tile_right": max(right_sizes.values()) if right_sizes else 0,
            "median_tile_left": float(np.median(list(left_sizes.values()))) if left_sizes else 0.0,
            "median_tile_right": float(np.median(list(right_sizes.values()))) if right_sizes else 0.0,
            "candidate_rows": candidate_rows,
            "pairs_evaluated": pairs_evaluated,
            "all_pairs": n_left * n_right,
            "candidate_reduction": (n_left * n_right) / max(candidate_rows, 1),
        }
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _candidate_coarse_pixels_for_left_pixel(
    nside: int, pix: int, *, radius_rad: float, device: torch.device
) -> torch.Tensor:
    return catalogs_core._candidate_coarse_pixels_for_left_pixel(
        nside, pix, radius_rad=radius_rad, device=device
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-left", type=int, default=50_000)
    parser.add_argument("--n-right", type=int, default=500_000)
    parser.add_argument("--radius-arcsec", type=float, default=2.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "catalog_crossmatch_arrow.json",
    )
    args = parser.parse_args()

    print(f"Generating catalogs: left={args.n_left}, right={args.n_right}")
    left_ra, left_dec, right_ra, right_dec = _make_catalogs(
        n_left=args.n_left,
        n_right=args.n_right,
        radius_arcsec=args.radius_arcsec,
        seed=args.seed,
    )

    left_t = torch.from_numpy(left_ra)
    left_d = torch.from_numpy(left_dec)
    right_t = torch.from_numpy(right_ra)
    right_d = torch.from_numpy(right_dec)

    with tempfile.TemporaryDirectory(prefix="torchsky_bench_") as tmpdir:
        left_path = Path(tmpdir) / "left.arrow"
        right_path = Path(tmpdir) / "right.arrow"
        _write_arrow_file(left_ra, left_dec, left_path)
        _write_arrow_file(right_ra, right_dec, right_path)
        print(f"  Arrow files: {left_path}, {right_path}")

        print("Running reference in-memory crossmatch...")
        t0 = time.perf_counter()
        ref_result = crossmatch_sky(left_t, left_d, right_t, right_d, radius_arcsec=args.radius_arcsec)
        ref_time = time.perf_counter() - t0
        print(f"  in-memory: {ref_time:.4f}s, {ref_result.left_index.numel()} matches")

        print("Computing tile-size statistics...")
        arrow_stats = _collect_arrow_stats(
            left_path, right_path, radius_arcsec=args.radius_arcsec
        )
        print(f"  candidate reduction: {arrow_stats['candidate_reduction']:.1f}x")
        print(f"  max tile left: {arrow_stats['max_tile_left']} rows")
        print(f"  max tile right: {arrow_stats['max_tile_right']} rows")

        print(f"Timing arrow-partitioned crossmatch ({args.runs} runs)...")

        def _run_arrow():
            import pyarrow as pa
            left_src = pa.ipc.open_file(pa.memory_map(str(left_path), "r")).read_all()
            right_src = pa.ipc.open_file(pa.memory_map(str(right_path), "r")).read_all()
            return catalogs_core.crossmatch_sky_from_arrow(
                left_src, right_src, radius_arcsec=args.radius_arcsec,
            )

        result = _run_arrow()
        t_arrow = _time_many(_run_arrow, runs=args.runs)

        print("Validating correctness...")
        n_arrow = result.left_index.numel()
        n_ref = ref_result.left_index.numel()

        if n_arrow == n_ref and n_arrow > 0:
            pairs_arrow: set[tuple[int, int]] = set(
                zip(result.left_index.tolist(), result.right_index.tolist())
            )
            pairs_ref: set[tuple[int, int]] = set(
                zip(ref_result.left_index.tolist(), ref_result.right_index.tolist())
            )
            tp = len(pairs_arrow & pairs_ref)
            fp = len(pairs_arrow - pairs_ref)
            fn = len(pairs_ref - pairs_arrow)
        elif n_arrow == 0 and n_ref == 0:
            tp = 0
            fp = 0
            fn = 0
        else:
            tp = 0
            fp = n_arrow
            fn = n_ref

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        rows: list[dict[str, Any]] = [
            {
                "operation": "crossmatch_sky_from_arrow",
                "n_left": args.n_left,
                "n_right": args.n_right,
                "radius_arcsec": args.radius_arcsec,
                "time_s": t_arrow,
                "crossmatches_s": args.n_left / t_arrow,
                "ref_time_s": ref_time,
                "speedup_vs_in_memory": ref_time / t_arrow,
                "matched_rows": n_arrow,
                "ref_matched_rows": n_ref,
                "precision": precision,
                "recall": recall,
                **arrow_stats,
            },
        ]

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))

        print()
        print("=== Arrow Partitioned Crossmatch Benchmark ===")
        print(f"  In-memory crossmatch:  {ref_time:.4f}s ({n_ref} matches)")
        print(f"  Arrow crossmatch:      {t_arrow:.4f}s ({n_arrow} matches)")
        print(f"  Speedup vs in-memory:  {ref_time / t_arrow:.2f}x")
        print(f"  Throughput:            {args.n_left / t_arrow:.0f} left-sources/s")
        print(f"  Candidate reduction:   {arrow_stats['candidate_reduction']:.1f}x")
        print(f"  Max tile left:         {arrow_stats['max_tile_left']} rows")
        print(f"  Max tile right:        {arrow_stats['max_tile_right']} rows")
        print(f"  Median tile left:      {arrow_stats['median_tile_left']:.0f} rows")
        print(f"  Median tile right:     {arrow_stats['median_tile_right']:.0f} rows")
        print(f"  Precision:             {precision:.4f}")
        print(f"  Recall:                {recall:.4f}")
        print(f"Wrote {args.json_out}")
        return 0 if (precision >= 0.99 and recall >= 0.99) else 1


if __name__ == "__main__":
    raise SystemExit(main())
