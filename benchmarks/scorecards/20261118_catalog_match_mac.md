# Catalog match: exactness and fair performance (2026-11-18)

Mac (Apple silicon), wiring and regression evidence; the claim machine for
large N is CANFAR (`catalog`, `catalog-match-1e7`, `catalog-match-1e8`, `cuda`
modes, not yet run). `bench_catalog_match_perf`: one subprocess per engine,
equal threads, cKDTree build + query (vectorised flatten), 2 arcsec radius,
right = left jittered 0.3 arcsec (sparse/dense) or independent Plummer
clusters (crowded). `ratio` is cKDTree time / torchsky time (> 1: torchsky
faster). Pair-ID parity is checked before timing; every row has zero
mismatches outside the 1e-8 arcsec tie band.

## Exactness (`bench_catalog_match_exact`, subject `ca6308f6`)

All 19 required cases `OK` (`scientific_validity: established`): poles incl.
exact +-90 deg, RA wrap, wrap at the pole, duplicates, equidistant ties,
radius ladder 1 mas to 1 deg, per-row radii over 5 dex, Plummer cluster,
two-sided sum / quadrature / max, mpmath boundary separations (oracle and
subject <= 1 uas). scipy `cKDTree` and astropy `search_around_sky` agree with
the brute-force oracle on every case. JSON: `catalog_match_exact_mac_ca6308f6.json`.

## Native kernel (`ca6308f6`), 4 threads

| layout | mode | N | torchsky | cKDTree | ratio | parity mismatches |
|---|---|---|---|---|---|---|
| sparse | best | 1e+04 | 0.003 s | 0.003 s | 1.08 | 0 |
| sparse | all | 1e+04 | 0.003 s | 0.008 s | 2.56 | 0 |
| sparse | knn2 | 1e+04 | 0.009 s | 0.004 s | 0.40 | 0 |
| sparse | best | 1e+05 | 0.030 s | 0.071 s | 2.42 | 0 |
| sparse | all | 1e+05 | 0.037 s | 0.180 s | 4.84 | 0 |
| sparse | knn2 | 1e+05 | 0.058 s | 0.045 s | 0.78 | 0 |
| sparse | best | 1e+06 | 0.329 s | 0.622 s | 1.89 | 0 |
| sparse | all | 1e+06 | 0.501 s | 1.391 s | 2.77 | 0 |
| sparse | knn2 | 1e+06 | 0.697 s | 0.788 s | 1.13 | 0 |
| dense | best | 1e+04 | 0.003 s | 0.003 s | 1.07 | 0 |
| dense | all | 1e+04 | 0.003 s | 0.008 s | 2.57 | 0 |
| dense | knn2 | 1e+04 | 0.006 s | 0.004 s | 0.56 | 0 |
| dense | best | 1e+05 | 0.021 s | 0.035 s | 1.65 | 0 |
| dense | all | 1e+05 | 0.039 s | 0.107 s | 2.74 | 0 |
| dense | knn2 | 1e+05 | 0.049 s | 0.044 s | 0.91 | 0 |
| dense | best | 1e+06 | 0.316 s | 0.594 s | 1.88 | 0 |
| dense | all | 1e+06 | 0.509 s | 1.366 s | 2.68 | 0 |
| dense | knn2 | 1e+06 | 0.626 s | 0.768 s | 1.23 | 0 |
| crowded | best | 1e+04 | 0.004 s | 0.003 s | 0.82 | 0 |
| crowded | all | 1e+04 | 0.006 s | 0.013 s | 2.06 | 0 |
| crowded | knn2 | 1e+04 | 0.006 s | 0.004 s | 0.58 | 0 |
| crowded | best | 1e+05 | 0.038 s | 0.037 s | 0.98 | 0 |
| crowded | all | 1e+05 | 0.129 s | 0.271 s | 2.10 | 0 |
| crowded | knn2 | 1e+05 | 0.051 s | 0.043 s | 0.86 | 0 |
| crowded | best | 1e+06 | 0.565 s | 0.653 s | 1.16 | 0 |
| crowded | all | 1e+06 | 1.516 s | 2.711 s | 1.79 | 0 |
| crowded | knn2 | 1e+06 | 0.735 s | 0.742 s | 1.01 | 0 |

## Native kernel (`ca6308f6`), 1 thread

| layout | mode | N | torchsky | cKDTree | ratio | parity mismatches |
|---|---|---|---|---|---|---|
| sparse | best | 1e+04 | 0.004 s | 0.006 s | 1.26 | 0 |
| sparse | all | 1e+04 | 0.007 s | 0.014 s | 2.02 | 0 |
| sparse | knn2 | 1e+04 | 0.021 s | 0.008 s | 0.40 | 0 |
| sparse | best | 1e+05 | 0.062 s | 0.089 s | 1.43 | 0 |
| sparse | all | 1e+05 | 0.092 s | 0.177 s | 1.93 | 0 |
| sparse | knn2 | 1e+05 | 0.169 s | 0.112 s | 0.67 | 0 |
| sparse | best | 1e+06 | 0.827 s | 1.581 s | 1.91 | 0 |
| sparse | all | 1e+06 | 1.755 s | 7.533 s | 4.29 | 0 |
| sparse | knn2 | 1e+06 | 2.982 s | 5.015 s | 1.68 | 0 |
| dense | best | 1e+04 | 0.007 s | 0.007 s | 0.96 | 0 |
| dense | all | 1e+04 | 0.009 s | 0.016 s | 1.82 | 0 |
| dense | knn2 | 1e+04 | 0.012 s | 0.009 s | 0.77 | 0 |
| dense | best | 1e+05 | 0.071 s | 0.110 s | 1.54 | 0 |
| dense | all | 1e+05 | 0.115 s | 0.349 s | 3.05 | 0 |
| dense | knn2 | 1e+05 | 0.234 s | 0.144 s | 0.61 | 0 |
| dense | best | 1e+06 | 1.051 s | 2.098 s | 2.00 | 0 |
| dense | all | 1e+06 | 2.733 s | 3.119 s | 1.14 | 0 |
| dense | knn2 | 1e+06 | 2.066 s | 4.147 s | 2.01 | 0 |
| crowded | best | 1e+04 | 0.010 s | 0.008 s | 0.82 | 0 |
| crowded | all | 1e+04 | 0.025 s | 0.028 s | 1.12 | 0 |
| crowded | knn2 | 1e+04 | 0.017 s | 0.010 s | 0.58 | 0 |
| crowded | best | 1e+05 | 0.201 s | 0.193 s | 0.96 | 0 |
| crowded | all | 1e+05 | 1.269 s | 0.754 s | 0.59 | 0 |
| crowded | knn2 | 1e+05 | 0.252 s | 0.216 s | 0.86 | 0 |
| crowded | best | 1e+06 | 1.585 s | 1.508 s | 0.95 | 0 |
| crowded | all | 1e+06 | 4.109 s | 4.804 s | 1.17 | 0 |
| crowded | knn2 | 1e+06 | 2.201 s | 1.920 s | 0.87 | 0 |

## Torch path only (`52685864`, before the native kernel), 4 threads

| layout | mode | N | torchsky | cKDTree | ratio | parity mismatches |
|---|---|---|---|---|---|---|
| sparse | best | 1e+04 | 0.004 s | 0.003 s | 0.66 | 0 |
| sparse | all | 1e+04 | 0.005 s | 0.008 s | 1.55 | 0 |
| sparse | knn2 | 1e+04 | 0.031 s | 0.003 s | 0.11 | 0 |
| sparse | best | 1e+05 | 0.030 s | 0.034 s | 1.11 | 0 |
| sparse | all | 1e+05 | 0.044 s | 0.088 s | 1.99 | 0 |
| sparse | knn2 | 1e+05 | 0.164 s | 0.040 s | 0.25 | 0 |
| sparse | best | 1e+06 | 0.390 s | 0.536 s | 1.38 | 0 |
| sparse | all | 1e+06 | 0.514 s | 1.114 s | 2.17 | 0 |
| sparse | knn2 | 1e+06 | 1.894 s | 0.680 s | 0.36 | 0 |
| dense | best | 1e+04 | 0.004 s | 0.003 s | 0.65 | 0 |
| dense | all | 1e+04 | 0.007 s | 0.008 s | 1.13 | 0 |
| dense | knn2 | 1e+04 | 0.021 s | 0.003 s | 0.16 | 0 |
| dense | best | 1e+05 | 0.031 s | 0.033 s | 1.07 | 0 |
| dense | all | 1e+05 | 0.044 s | 0.086 s | 1.94 | 0 |
| dense | knn2 | 1e+05 | 0.157 s | 0.041 s | 0.26 | 0 |
| dense | best | 1e+06 | 0.378 s | 0.608 s | 1.61 | 0 |
| dense | all | 1e+06 | 0.608 s | 1.462 s | 2.40 | 0 |
| dense | knn2 | 1e+06 | 2.225 s | 0.857 s | 0.39 | 0 |
| crowded | best | 1e+04 | 0.014 s | 0.005 s | 0.36 | 0 |
| crowded | all | 1e+04 | 0.030 s | 0.016 s | 0.52 | 0 |
| crowded | knn2 | 1e+04 | 0.054 s | 0.005 s | 0.09 | 0 |
| crowded | best | 1e+05 | 0.205 s | 0.045 s | 0.22 | 0 |
| crowded | all | 1e+05 | 0.680 s | 0.258 s | 0.38 | 0 |
| crowded | knn2 | 1e+05 | 0.502 s | 0.056 s | 0.11 | 0 |
| crowded | best | 1e+06 | 2.782 s | 0.728 s | 0.26 | 0 |
| crowded | all | 1e+06 | 8.566 s | 3.197 s | 0.37 | 0 |
| crowded | knn2 | 1e+06 | 10.689 s | 0.982 s | 0.09 | 0 |

Peak RSS (per-engine subprocess) is 1.1-3.8x cKDTree's; most of it is the
~230 MB cost of importing torch, which dominates below 1e6 rows. The plan's
<= 1.5x RSS target is not met for kNN at 1e6 (3x) and is recorded, not gated.

JSON: `catalog_match_perf_mac_native_ca6308f6.json`,
`catalog_match_perf_mac_torchpath_52685864.json`.
