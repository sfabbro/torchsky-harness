# Crowded-field association: exclusive vs independent vs nway (2026-11-20)

`bench_catalog_match_injected --scenes uniform,crowded` (Mac, subject
`421957eb`, nway 4.8.0 out of process). JSON:
`catalog_match_injected_crowded_mac_421957eb.json`. All gated rows pass
(0 failures).

Methods:

- **torchsky**: `associate` / `associate_many` defaults. These are exclusive (one-to-one Bethe
  posteriors) with completeness fitted by maximum marginal likelihood. Crowded
  pairwise scenes also fit the error scale and floor (`calibrate="scale_floor"`).
- **torchsky_indep**: the same fits with `exclusive=False`, which is nway's
  independent posterior model.
- **torchsky_plain**: independent model, true completeness, no calibration.
- **nway**: true completeness as prior, published errors. In `blend_shallow` nway
  was also asked for its W1 magnitude prior (`--mag SECOND:MAG auto`).

## Uniform scenes (release gate, unchanged setup)

| scene | method | completeness@0.9 | purity@0.9 | ECE(p_any) | MAP tuple | time |
|---|---|---|---|---|---|---|
| pairwise_dense | torchsky | 0.905 | 0.955 | 0.022 | – | 1.6 s |
| pairwise_dense | torchsky_indep | 0.902 | 0.955 | 0.022 | – | 0.04 s |
| pairwise_dense | nway | 0.714 | 0.966 | 0.035 | – | 5.4 s |
| nway_three | torchsky | 0.967 (any) | 0.989 | 0.006 | 0.929 | 0.24 s |
| nway_three | torchsky_indep | 0.965 (any) | 0.990 | 0.006 | 0.928 | 0.16 s |
| nway_three | nway | 0.658 (any) | 0.996 | 0.126 | 0.817 | 7.9 s |

Without crowding, the exclusive and independent models agree, as designed: ρ = 1
wherever no right row is shared.

## Crowded scenes (0.05° box, 12 000 primaries = 0.37 per arcsec²)

### crowded_miscal

The shallow secondary holds 20 % of the primaries (plus 300 background
sources). Its published errors (0.15–0.4″) are half the true ones.

| method | completeness@0.5 | purity@0.5 | completeness@0.9 | purity@0.9 | ECE | fitted completeness | fitted scale |
|---|---|---|---|---|---|---|---|
| torchsky | **0.470** | **0.627** | **0.086** | **0.904** | **0.023** | 0.196 (true 0.2005) | 1.94 (true 2) |
| torchsky_indep | 0.431 | 0.522 | 0.000 | – | 0.033 | 0.205 | 1.98 |
| torchsky_plain | 0.362 | 0.510 | 0.034 | 0.638 | 0.087 | – | – |
| nway | 0.333 | 0.524 | 0.015 | 0.607 | 0.090 | – | – |

### blend_shallow

The secondary is WISE-like: flux-weighted PSF centroids, W1 < 16.5, a 6″
deblending radius and 1.5× underestimated errors. Truth is the seed star.
torchsky adds a pair-colour term (W1 − G).

| method | completeness@0.5 | purity@0.5 | completeness@0.9 | purity@0.9 | ECE |
|---|---|---|---|---|---|
| torchsky | **0.736** | 0.753 | **0.489** | **0.935** | **0.006** |
| torchsky_indep | 0.713 | 0.791 | 0.295 | 0.929 | 0.009 |
| torchsky_plain | 0.253 | 0.492 | 0.020 | 0.467 | 0.013 |
| nway | 0.256 | 0.497 | 0.020 | 0.500 | 0.020 |

nway refused its automatic magnitude histogram ("too few secure matches"), so
its row ran without a magnitude prior.

### nway_crowded

Three catalogues. The two secondaries hold 20 % and 15 % of the primaries,
with correct errors, so this scene isolates exclusivity.

| method | MAP tuple | any completeness@0.9 | any purity@0.9 | ECE | time |
|---|---|---|---|---|---|
| torchsky | **0.881** | **0.528** | **0.959** | 0.010 | 1.5 s |
| torchsky_indep | 0.841 | 0.204 | 0.917 | 0.009 | 0.22 s |
| torchsky_plain | 0.840 | 0.201 | 0.915 | 0.008 | 0.20 s |
| nway | did not finish in 1200 s | | | | |

## Reading

- **Crowding is a model problem, not a search problem.** In the independent model a
  shallow source inside the gates of many deep primaries counts as a counterpart
  for each of them. The exclusive model counts it once. That alone raises
  confident completeness by a factor 1.7–2.6 in the crowded scenes, at
  equal or better purity.
- **Calibration of misestimated errors works in crowded fields** once the error
  scale is fitted by direct marginal likelihood. Alternating EM stalled at
  1.45 for a true 2.0. The two models fit the scale equally well (1.94 vs 1.98),
  but only the exclusive one turns it into confident matches.
- **Cost.** The exclusive model is 2.5–7× slower than the independent one on these
  scenes (belief propagation inside every likelihood evaluation). Locally, at
  200 000 primaries, 390 000 pairs and with calibration, it takes about 90 s
  against 12 s.
