# Catalog crossmatch benchmarks before the zone-index rewrite

Retired 2026-11-18 when torchsky replaced its declination-strip, HEALPix-ring
and declination-window crossmatch engines with one zone index (torchsky commit
`4092d891`), and removed `crossmatch_sky_from_arrow` / `crossmatch_sky_streamed`.

- `bench_catalog_crossmatch.py` compared private torchsky engines
  (`_crossmatch_sky_declination_pruned`, `_crossmatch_sky_coarse_indexed`,
  `_crossmatch_sky_declination_indexed`) against each other. Those internals no
  longer exist.
- `bench_catalog_crossmatch_arrow.py` measured the removed on-disk Arrow tiling
  path.

They are kept verbatim as the source of their recorded results (for example
`benchmark_results/catalog_crossmatch.json` and the CANFAR `catalog_n1e6` /
`catalog_n1e7` artifacts). They run only against torchsky 0.5.0 or earlier.

Replacements:

- `benchmarks/bench_catalog_match_exact.py`: public-API exactness against
  brute force, mpmath, scipy `cKDTree` and astropy on adversarial geometry.
- `benchmarks/bench_catalog_match_perf.py`: fair performance against scipy
  `cKDTree` and astropy, with one subprocess per engine (per-engine peak RSS),
  equal thread counts, build and query timed separately, and pair-ID parity
  checked first.

`scripts/run_catalog_scale_contract.py` stays in place, frozen, as the
source of `benchmarks/scorecards/20260821_catalog_scale.md`. It times the
cKDTree build together with the query and flattens `query_ball_point` with a
Python loop, so prefer the new performance bench for comparisons.
