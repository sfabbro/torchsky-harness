# Real data: exclusive association, direct ML calibration (2026-11-20)

`bench_catalog_match_real` on CANFAR, run `catalog-match-real-20261119-230010`
(subject torchsky `0a682079`, harness `203c434`). The inputs are Gaia DR3,
2MASS PSC and CatWISE2020, read from HATS on `/arc/projects/hats`. JSON (six
fields; the LMC bar was still running when this was written):
`canfar_catalog-match-real-20261119-230010_catalog_match_real_partial.json`.

All CatWISE rows calibrate per blending-count bin. The scale shown is the first
bin, sources with no other Gaia star within 6″. The mean d² is over confident
matches (posterior > 0.9); for correct errors theory gives 2.0 (1.999 within the gate).

| field | Gaia rows | 2MASS: scale / mean d² | CatWISE exclusive: completeness / scale / mean d² | CatWISE independent: completeness / scale | CatWISE with transferred calibration: completeness / mean d² |
|---|---|---|---|---|---|
| north_pole | 1360 | 0.63 / 1.97 | 0.891 / 1.97 / 2.01 | 0.899 / 1.98 | – |
| south_pole | 3009 | 0.77 / 1.96 | 0.687 / 2.43 / 1.81 | 0.696 / 2.40 | – |
| ra_wrap | 1651 | 1.06 / 1.95 | 0.843 / 1.77 / 1.94 | 0.843 / 1.76 | – |
| pleiades | 9675 | 1.07 / 2.00 | 0.726 / 2.22 / 1.84 | 0.741 / 2.21 | – |
| barnards_star | 9958 | 1.18 / 2.00 | 0.574 / 3.47 / 1.61 | 0.640 / 3.28 | – |
| baade_window | 51580 | 1.57 / 1.70 | 0.071 / 34.4 / 1.03 | 0.505 / 137 / – | 0.005 / 2.58 |

## Reading

- **Uncrowded fields are calibrated.** CatWISE errors are underestimated by
  about 2× (1.8–2.4), and 2MASS needs 0.6–1.2. With the fitted calibration,
  confident matches follow χ²₂: mean d² 1.81–2.01 against 2.0. Exclusive and
  independent posteriors agree there, as they should when nothing is shared.
- **In Baade's window, positions alone cannot fix the error scale.** A single
  Gaussian error model, fitted to convergence, absorbs the Gaia stars that
  cluster around every blended CatWISE detection: scale 34 exclusive, 137
  independent. The old 8-round EM showed 8.5 only because it had not
  converged.
  - Exclusivity limits the damage: completeness 0.07 against 0.51, in a field
    where Gaia is far deeper than CatWISE.
  - Transferring the uncrowded field's per-W1 calibration (`right_scale=`,
    `floor_mas2=`) is the other bound: completeness 0.005 with mean d² 2.58.
    Blending does add error, but nothing like 34×.
- **Next step: a blend-aware model.** Predict each CatWISE position as the
  flux-weighted centroid of the Gaia stars in the beam, from Gaia fluxes and
  a W1 − G colour model, rather than inflating a Gaussian.
