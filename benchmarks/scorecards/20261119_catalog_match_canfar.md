# Catalog match on CANFAR (2026-11-19)

Linux CANFAR headless sessions (`images.canfar.net/astroai/notebook`), torchsky
subjects from `sfabbro/torchsky`, harness from `sfabbro/torchsky-harness`.
`bench_catalog_match_perf`: one subprocess per engine, equal threads,
cKDTree build + query with a vectorised flatten, 2 arcsec, pair-ID parity
checked first. Every row has **zero parity mismatches** outside the 1e-8
arcsec tie band. `ratio` = cKDTree time / torchsky time.

## Exactness at 4x scale (`catalog-20261119-170039`, subject `49714a57`)

All 19 required cases `OK` (126 rows).

## CPU, 1e6 (`catalog-20261119-170039`, native kernel, RSS via VmHWM)

| layout | mode | 1 thread | 8 threads |
|---|---|---|---|
| sparse | best / all / knn2 | 2.16 / 2.76 / 1.25 | 1.63 / 3.39 / 1.55 |
| dense | best / all / knn2 | 1.86 / 2.79 / 1.21 | 1.88 / 3.44 / 1.64 |
| crowded | best / all / knn2 | 1.20 / 1.06 / 0.89 | 1.49 / 3.00 / 1.18 |

## CPU, 1e7 and 1e8, 16 threads (subject `0cf5264d`)

| N | layout | best | all |
|---|---|---|---|
| 1e7 | sparse | 1.9 s vs 9.0 s (4.7x) | 2.2 s vs 21.8 s (10.0x) |
| 1e7 | dense | 1.5 s vs 9.0 s (6.1x) | 2.2 s vs 22.3 s (10.0x) |
| 1e8 | sparse | 11.7 s vs 70.0 s (6.0x) | 16.3 s vs 162.8 s (10.0x) |
| 1e8 | dense | 11.5 s vs 71.6 s (6.2x) | 19.1 s vs 172.8 s (9.0x) |

1e8 peak RSS (that subject, before query vectors moved into the kernel):
best 24.9 GB vs 14.0 GB, all 25.6 GB vs 44.5 GB.

## CUDA (10.75 GB GPU) vs cKDTree on 8 CPU threads (subject `49714a57`)

| N | layout | best | all | knn2 |
|---|---|---|---|---|
| 1e6 | sparse | 10.5x | 36.1x | 6.7x |
| 1e6 | dense | 13.0x | 32.4x | 6.6x |
| 1e6 | crowded | 4.3x | 13.3x | 2.7x |
| 1e7 | sparse | 14.4x | 37.9x | 8.7x |
| 1e7 | dense | 14.9x | 36.4x | 8.8x |
| 1e7 | crowded | 5.8x | OOM (fixed in `359c3ff7`) | - |

## Real data: Gaia DR3 x 2MASS PSC x CatWISE2020 (HATS on /arc)

`bench_catalog_match_real`, seven fields. Run 1 (`49714a57`, single median
epoch, published errors) and run 2 (`359c3ff7`, pair epochs, `scale_floor`
calibration):

- Barnard's star (10.4 arcsec/yr) lost in run 1 against 2MASS (median epoch),
  recovered in run 2 (pair epochs). Other > 200 mas/yr stars recovered in both;
  a naive 2 arcsec match recovers none.
- CatWISE published errors are underestimated: confident-match mean d^2 2.6-5.0
  (theory 2.0) in run 1; fitted right-error scale 1.8-2.6 with calibration in
  uncrowded fields, after which d^2 mean 1.74-1.88 and tail fractions 0.11-0.12
  (theory 0.10), and Gaia->CatWISE completeness rises (north pole 0.75 -> 0.88).
- 2MASS errors are close to right (scale 0.6-1.3, floor 18-96 mas).
- Crowded fields (Baade, LMC bar) over-inflate CatWISE errors (scale 4-7,
  floor > 100 mas, confident d^2 mean ~0.85): blended WISE sources have
  genuinely larger, non-Gaussian offsets; a single global error model is too coarse there
  (magnitude / crowding dependent calibration is the next step).
- N-way MAP agrees with pairwise on 98.8-99.6 % of rows; certified residual 0.
