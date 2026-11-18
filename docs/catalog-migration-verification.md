# Catalog and shared benchmark migration

This note records the catalog/common evaluation boundary for the harness
extraction. The benchmark implementations and evaluator tests are staged under
`torchsky_harness.benchmarks` and `tests`; no catalog or geometry algorithms,
torchsky production APIs, or source-repository files were changed by this work.

## Catalog evaluation

- `bench_catalog_compat.py` compares `torchsky.catalogs.crossmatch_sky` with
  Astropy `SkyCoord` on synthetic catalogs and vendored VOTable inputs. The
  FITS roundtrip fixture uses the public `torchfits.table` API. Astropy remains
  a comparison oracle; FITS serialization and reads belong to torchfits.
- `bench_catalog_crossmatch.py` and `bench_catalog_crossmatch_arrow.py` retain
  their candidate-generation, parity, throughput, and memory measurements.
  Their references into `torchsky.catalogs.core` are version-coupled benchmark
  diagnostics, not a supported torchsky API. This extraction does not widen
  that API or change their numerical work.
  Both were retired on 2026-11-18 to `docs/archive/catalog-crossmatch-pre-rewrite/`
  when torchsky removed those internals; `bench_catalog_match_exact.py` and
  `bench_catalog_match_perf.py` replace them through the public API.
- Catalog evaluator tests remain separate from torchsky's production behavior
  and interop tests. Keep the package's catalog and opt-in torchfits facade
  tests in torchsky; retain evaluator tests here when the source snapshot is
  finalized.

## FITS and table ownership

FITS file handling, table predicate execution, and read/write performance are
owned by torchfits. The harness keeps historical evaluation evidence without
presenting I/O as a torchsky capability:

- `bench_pipeline_table_sphere.py` remains a mixed evaluation. It preserves all
  five output rows (`read_where` for `cpp_filtered` and `torch`,
  `pipeline_where` for both paths, and `pipeline_no_pushdown`), selected-row
  parity, pipeline checksum parity, the read-speed ratio, both pipeline ratio
  gates, and the borderline retry behavior. Table creation, filtered reads,
  full reads, and row counts now use public `torchfits` APIs. The standalone
  `read_filtered_vs_torch` metric is torchfits-owned; the HEALPix reduction is
  the torchsky compute portion of the mixed path.
- `bench_cold_targets.py` is a historical cold-read evaluator and is wholly
  torchfits-owned, including rows that exercise the thin torchsky delegation
  facade. Preserve its results and provenance until a dedicated torchfits
  migration replaces the evaluator.
- FITS-loading measurements in `bench_gpu_memory.py` are torchfits-owned. Its
  transform measurements continue to exercise torchsky transforms.
- Keep `tests/test_perf_gate.py` focused on torchsky hot paths such as WCS,
  HEALPix, indexing, resampling, and operators. Any existing I/O-only gate is
  ownership debt for a separate torchfits migration; do not drop it during this
  extraction.

No Astropy FITS reader or writer is used by the catalog/common files covered
here. Astropy WCS, coordinates, and tables remain optional parity/reference
tools.

## Focused verification

Ran against the already materialized torchsky test environment, with the
harness installed editable:

```text
pixi run --manifest-path $TORCHSKY_ROOT/pixi.toml --frozen --no-install -e test -- pytest /private/tmp/torchsky-harness-20261011/tests/test_catalog_compat.py /private/tmp/torchsky-harness-20261011/tests/test_real_data_validation.py /private/tmp/torchsky-harness-20261011/tests/test_bootstrap_map_compat_tools.py -q --tb=short
9 passed in 2.81s
```

The pipeline smoke ran with 10,000 rows, one timing sample, `nside=8`, and
zero timing thresholds so it checked row execution and parity without making a
performance claim. All five rows were emitted. Both read paths selected 1,220
rows with checksum 2,020; all three pipeline paths selected 1,220 rows with
checksum 10,980. This confirms the filtered and full-read paths agree on the
selected catalog and spherical reduction result. The normal gate thresholds
and default retry settings remain in the benchmark.

```text
pixi run --manifest-path $TORCHSKY_ROOT/pixi.toml --frozen --no-install -e test -- python -m torchsky_harness.benchmarks.bench_pipeline_table_sphere --table-path /private/tmp/torchsky-harness-pipeline-smoke.fits --json-out /private/tmp/torchsky-harness-pipeline-smoke.json --n-rows 10000 --runs 1 --nside 8 --min-read-ratio-filtered-vs-torch 0 --min-pipeline-ratio-filtered-vs-no-pushdown 0 --min-pipeline-ratio-torch-vs-no-pushdown 0 --retry-borderline-frac 0
```

Ruff passed for the touched catalog/common modules and migrated helper tests.

The initial Pixi invocation from the staging directory selected its empty local
environment and failed with `pytest: command not found`; the focused run above
uses the explicit core manifest and `--no-install`, so it did not install or
modify packages.
