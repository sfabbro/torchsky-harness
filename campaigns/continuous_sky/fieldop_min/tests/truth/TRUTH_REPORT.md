# Known-truth measurements of the field operator (grid sky and item sky)

All numbers come from `tests/truth/` (driver `run_truth.py`, problem and measurements `truth_problem.py`, smoke test `test_truth_smoke.py`, raw results
`results/*.json`, aggregated `results/summary_*.json`, configuration trials in `results/task1/`). Nothing in `fieldop/` or the existing tests was edited.
One core, one thread, no background processes. Machine shared with other agents, so wall times are upper bounds.

## 1. The problem and what is frozen

Synthetic problem (built by `truth_problem.build_field`, which copies the structure of `fieldop/synth.make_field` and uses its analytic renderer):

* One instrument, one filter (flat spectrum, so the sky is one band image). Detector 44 x 44 pixels of 0.10 arcsec, noise 1.0 per pixel (surface-brightness units),
  flat background 5.0 in every exposure (same value in all exposures; the package's own synthetic field scatters it by 1.0 per exposure, which I did not want).
* Response: two concentric Gaussians (core width 0.07 arcsec, wings 3 times wider with 10 per cent of the flux) integrated over the square pixel; same everywhere,
  not varying with colour, position or time. Registration offsets 0 (dither centres random within +-0.3 arcsec, rotation 0).
* Sky: 36 injected round Gaussian sources of known total flux (30 point-like with intrinsic width 0.03 arcsec, 6 extended with width 0.12 or 0.20 arcsec).
  Fluxes are set so that the total signal-to-noise with the first 8 training exposures (optimal weights, true profile through the true response) is log-uniform in
  each bin 0.3-1, 1-3, 3-10, 10-30, 30-100 (12, 10, 6, 4, 4 sources). The population, positions and fluxes are the same in every run; only the noise (and the fit seed) change.
  Unit: flux is surface brightness times arcsec^2 (a flux of 1 is a peak of about 12 noise units per exposure for a point-like source).
* 25 empty positions (spacing 0.2 arcsec, kept at 0.25 to 0.6 arcsec plus twice the source width away from every source, farther for brighter ones, so the response
  wings of a source do not leak in), all inside the region seen by every exposure.
* Exposures: 1 held out (exposure 0) + N training exposures (N = 8 or 32; the first 8 are common to both). No validation exposures.
* Model settings: sky pitch 0.05 arcsec (101 x 101 nodes, half width 2.5 arcsec), render pitch 0.025 arcsec, response extent 1.0 arcsec, 3 spectral basis functions,
  4 patches per step, 1 sky inner step, `--amplitude unit`, no distortion, no stripes, no pattern, all variability axes off, blur gauge off (`blur_floor` 0).
  Grid sky: package default soft positivity penalty (`sky_neg` 1.0). Item sky: 1200 positive Gaussian items on three seeding lattices, no relocation.

**Frozen (to the truth):** the response (kernel network and its code are trained by `fit_kernel_to_truth` on the log of the true pixel-integrated kernel, 2500 steps,
then removed from the optimiser groups by a subclass of the model; error of the started kernel: relative L1 0.0068, rms log error 0.012 where the kernel exceeds 1e-3 of its peak,
integral ratio 1.0000); the noise scale and fractional noise floor (floor set to 1e-6); throughput is exactly 1 by construction (`unit` amplitude); distortion and
stripes are off. **Learned:** the sky, the per-exposure background (6 Legendre coefficients), the per-exposure registration (6 affine numbers; truth is 0), the
registration pre-pass; for the held-out exposure its background and registration. The sky always starts from the data coadd, never from the truth.

## 2. Task 1: a converging configuration

Score = held-out chi-squared per pixel from the package's own `fit_heldout` + `predict_exposure` + `chi2_exposure`, over the pixels it scores; noise-only value =
the same pixels scored with the true noiseless model (0.994 for noise seed 0). "Within 5 per cent" = ratio below 1.05. Wall time is the whole run (setup, kernel start 25 s,
pre-pass, fit, held-out fit, score) except where stated. The six configurations are the whole set tried for this table; earlier development runs
(same problem, kernel started with only 150 steps so with a 9 per cent kernel error; grid 100 steps gave 1.026, grid 400 steps 1.014) are not counted.

| # | sky | response | training exposures | steps | fit s | total wall s | held-out score | noise-only | ratio |
|---|---|---|---|---|---|---|---|---|---|
| C1 | grid | frozen at truth | 8 | 100 | 22 | 58 | 1.0208 | 0.9938 | 1.027 |
| C2 | grid | frozen at truth | 8 | 400 | 91 | 130 | 1.0076 | 0.9938 | 1.014 |
| C3 | items | frozen at truth | 8 | 150 | 67 | 103 | 1.0013 | 0.9938 | 1.008 |
| C4 | items | frozen at truth | 8 | 300 | 169 | 208 | 1.0026 | 0.9938 | 1.009 |
| C5 | grid | learned, package diffraction warm start, blur gauge on | 8 | 400 | 136 | 150 | 2.219 | 0.9938 | 2.23 |
| C6 | items | learned, same | 8 | 300 | 186 | 200 | 2.321 | 0.9938 | 2.34 |

Result: with the response frozen at the truth, both sky forms converge to within 3 per cent of the noise-only value in under 1 minute of fit (grid) or 3 minutes (items)
on one core; every frozen-response configuration is below 5 per cent, and C1 (grid, 100 steps, 58 s) is the smallest. The training-batch chi-squared per pixel of C2 and C4
oscillates between 0.95 and 1.10 from step 75 on (batch noise), without trend. I did not try smaller fields, fewer exposures or one-step shorter fits.
With the response learned (C5, C6) nothing converges in the time allowed: the batch chi-squared falls to about 0.9 early, then climbs to 2.3 to 3.2 (grid, steps 150 to 300) and 1.4 to 2.3
(items) while the kernel network is the only thing that differs from C2 and C4; I did not diagnose which kernel directions move (it would need a separate study). The tables below use C2 (400 steps) and C4 (300 steps).

**Caution:** a held-out score within 1 to 2 per cent of the floor is not a converged sky at the node scale. The sky error against the truth at the nodes (rms over |x|, |y| < 1.5 arcsec,
in surface-brightness units where the noise per pixel is 1) is 1.76 for the grid and 0.53 for the items (N = 8, seed 0); at 32 exposures 1.83 and 0.47. The grid error does not fall with exposures (see 3a, 3c).

## 3. Task 2: measurements (converged configuration C2 for the grid, C4 for the items)

Realisations: 3 noise realisations (seeds 0, 1, 2) for N = 8; **2** (seeds 0, 1) for N = 32 (a third did not fit in the 40 minute budget). Held-out score / noise-only ratio: grid N = 8
1.014, 1.016, 1.016; items N = 8 1.009, 1.011, 1.009; grid N = 32 1.005, 1.011; items N = 32 1.000, 0.999. Compute: 36 minutes of fits for the whole of this section (including the diagnostics in 3b, 3c).

**Ideal estimator (stated):** for a source or a position with a given profile, F = sum m (d - b) / sum m^2 over all training exposures and pixels, where m is the noiseless
model of a unit-flux source of that profile through the true response and pixel box, b is the true background, noise is 1 per pixel. This is the inverse-variance weighted coadd of the
noiseless-kernel-matched data; it knows position, profile, response and background exactly (more than the fit does). Its noise is 1 / sqrt(sum m^2), also the definition of the total signal-to-noise.

**Gauge (important for 3b).** A constant sky and the exposure backgrounds are degenerate; the package fixes the split by a pedestal gauge (the mean fitted background of the training
exposures equals the mean of their medians). With sources in the field the median is 0.19 above the true background (5.19 against 5.00), so the package's gauge asks the sky to sit 0.19 *below* zero in empty regions.
Every "recovered" number below is therefore reported two ways: **raw** (sky as the package returns it) and **total** (raw plus the offset between the fitted and true mean background,
acting as a constant sky in the aperture). Only the total is gauge-independent. "total" is the primary number.

**Recovered flux estimator on the sky** (the fitted sky image at the nodes): F = sum(W s) / sum(W P), P the true profile sampled at the nodes, W the aperture weight, out to 4 widths (at least 0.11 arcsec).
* *strict*: W = P (the source's own true profile; for point-like sources this is a 5 to 9 node aperture, smaller than the response core).
* *matched*: W = Gaussian of width sqrt(width^2 + 0.07^2) (the true profile seen at the resolution of the response). Same normalisation, so a perfect sky gives F exactly in both.

### 3a. Flux bias: recovered / true flux by total signal-to-noise (N = 8, 3 realisations)

Entries: mean of (recovered flux / true flux) over sources and realisations, with the error of the mean from the realisation-to-realisation spread of the bin mean (3 values) and, in
brackets, the scatter over sources of the realisation-averaged ratio. n = sources in the bin. Ideal estimator for the same sources and noise.

| total S/N | n | grid strict | items strict | grid matched | items matched | ideal |
|---|---|---|---|---|---|---|
| 0.3 to 1 | 12 | 0.38 +- 0.07 (0.21) | 0.39 +- 0.02 (0.15) | 1.33 +- 0.17 (0.59) | 1.47 +- 0.11 (0.57) | 1.16 +- 0.63 (0.94) |
| 1 to 3 | 10 | 0.46 +- 0.05 (0.23) | 0.43 +- 0.03 (0.27) | 0.94 +- 0.04 (0.25) | 0.91 +- 0.07 (0.25) | 1.36 +- 0.06 (0.27) |
| 3 to 10 | 6 | 0.58 +- 0.03 (0.26) | 0.63 +- 0.02 (0.25) | 0.77 +- 0.03 (0.22) | 0.79 +- 0.03 (0.22) | 0.97 +- 0.04 (0.14) |
| 10 to 30 | 4 | 0.78 +- 0.02 (0.19) | 0.96 +- 0.01 (0.07) | 0.96 +- 0.02 (0.07) | 0.98 +- 0.02 (0.03) | 1.03 +- 0.02 (0.02) |
| 30 to 100 | 4 | 0.72 +- 0.01 (0.19) | 1.02 +- 0.01 (0.03) | 0.95 +- 0.01 (0.03) | 1.01 +- 0.01 (0.01) | 1.00 +- 0.00 (0.01) |

The same, as a bias in units of the ideal noise of each source, mean of (recovered - true) / sigma (matched aperture; strict in brackets); ideal in the last column:

| total S/N | grid | items | ideal |
|---|---|---|---|
| 0.3 to 1 | +0.10 (-0.39) | +0.18 (-0.39) | +0.15 |
| 1 to 3 | -0.15 (-0.91) | -0.19 (-0.92) | +0.54 |
| 3 to 10 | -1.24 (-2.33) | -1.14 (-2.06) | -0.17 |
| 10 to 30 | -0.72 (-4.34) | -0.22 (-0.58) | +0.51 |
| 30 to 100 | -2.74 (-16.5) | +0.37 (+1.43) | +0.23 |

N = 32 (2 realisations; total S/N is twice as large for the same fluxes, so the bins hold different sources: n = 6, 10, 8, 5, 4):
grid strict 0.48, 0.23, 0.36, 0.68, 0.70; items strict 0.49, 0.22, 0.40, 0.82, 0.97; grid matched 1.71, 0.84, 0.68, 0.83, 0.93; items matched 1.78, 0.84, 0.67, 0.90, 0.99; ideal 1.55, 1.03, 1.12, 0.99, 1.00
(errors of the mean 0.01 to 0.1 except the lowest-S/N ideal, 0.8; the full table with errors is in `results/summary_g_n32.json` and `summary_i_n32.json`).

What this measures and how to read it. The ideal estimator measures each source alone (positions known, neighbours ignored), so it is unbiased only up to crowding: the ratio 1.36 (S/N 1 to 3) and 1.03 at S/N 10 to 30 contain
about +0.5 noise units of leakage from bright neighbours (the same +0.54 / +0.51 in the second table); the entries 1.16 and 1.55 at S/N below 1 are mostly noise. The ideal values are the reference, not a perfect 1. Findings:
* Bright sources (S/N above 10), item sky: unbiased in the strict aperture (0.96 to 1.02 at N = 8; 0.82 to 0.97 at N = 32) and in the matched one (0.98 to 1.01). Grid sky: 20 to 30 per cent of the flux is missing from the strict (true-profile) aperture at every N (0.72 to 0.78 at N = 8, 0.68 to 0.70 at N = 32), and 3 to 7 per cent from the matched aperture. The grid sky recovers the total flux at the resolution of the response but not the sub-response structure (a 0.03 arcsec point source ends up broader). It does not improve from 8 to 32 exposures.
* Faint end (S/N below 10): both sky forms lose flux in the strict aperture (ratio 0.2 to 0.6) and the matched ratio is 0.67 to 0.94 at S/N 1 to 10 with a bias of -1.1 to -1.2 noise units at S/N 3 to 10, in both forms and at both N; the ideal estimator has none. Below S/N 1 the matched ratio is above 1 (1.3 to 1.8) because of the positive empty-sky offset measured in 3b (+0.73 noise units for point-like sources at N = 8, from the gauge-independent total), partly cancelled by lost flux. I could not separate the two at the lowest S/N with 12 sources.
* In the noise units the faint-end flux bias is at most about 1.2 sigma of a single source at N = 8, but it grows in flux ratio as the fixed offset is divided by a smaller flux.

### 3b. Empty-sky bias: mean of (recovered sky - true sky) in source-free regions, in units of the ideal coadd's noise per resolution element

Resolution element = Gaussian of width 0.07 arcsec (the response core); ideal noise per element: 0.0138 (N = 8) and 0.0069 (N = 32) in flux units. The reference value for each position is what the true sky
gives in the same aperture (zero away from the faint-source tails). 25 positions, spacing 0.2 arcsec (neighbours are correlated). Errors: "real." = standard error from the realisation means (3 or 2 values); "pos." = standard error over positions of the realisation-averaged value.

| sky | N | realisations | raw | total (gauge-independent) | error (real.) | error (pos.) |
|---|---|---|---|---|---|---|
| grid, positivity penalty on (default) | 8 | 3 | +0.03 | **+0.86** | 0.03 | 0.05 |
| items (positive by construction) | 8 | 3 | +0.06 | **+0.89** | 0.01 | 0.03 |
| grid, positivity penalty on | 32 | 2 | -0.08 | **+1.60** | 0.04 | 0.01 |
| items | 32 | 2 | +0.00 | **+1.68** | 0.02 | 0.00 |
| grid, no positivity penalty (diagnostic) | 8 | 2 | -0.85 | -0.01 | 0.17 | 0.29 |
| grid, no positivity penalty (diagnostic) | 32 | 2 | -2.28 | -0.60 | 0.07 | 0.26 |
| ideal coadd | 8 | 3 | | -0.04 | 0.15 | 0.12 |
| ideal coadd | 32 | 2 | | -0.35 | 0.15 | 0.18 |

Sign and size: **positive: a surface-brightness offset of +0.19 (3.8 per cent of the background), which is +0.86 to +0.89 ideal noise units per resolution element at 8 exposures and +1.6 to +1.7 at 32 (it grows as the square root of the number of exposures because it is a fixed
offset in surface brightness while the noise falls)**. It appears for both sky forms with the same size. The raw sky is consistent with zero (that is the trap: the package's raw sky has the right empty level), because the pedestal gauge puts the +0.19 into the fitted
backgrounds (mean 5.19 against the true 5.00; 5.19, 5.19 and 5.18 in the three N = 8 realisations); the model is then 0.19 too bright in empty regions and pays the chi-squared for it, because the sky cannot go negative (items) or is held up by the soft penalty (grid).
Switching the penalty off (grid, `sky_neg` 0) lets the raw sky go to -0.85 (N = 8) and the total comes back to zero within errors (-0.01 +- 0.17 +- 0.29; at 32 exposures -0.60 +- 0.07 +- 0.26, 2 sigma by the position error). So positivity plus the median pedestal gauge is the cause. The diagnostic grid fits are less converged in the score (held-out ratio 1.075 to 1.079 at N = 8, 1.019 to 1.028 at N = 32): without the penalty the noise amplification at the node scale is large (3c).
The ideal estimator's means (-0.04, -0.35) are zero within about 2 sigma; the -0.35 at 32 is from 2 realisations of 25 correlated positions.

### 3c. Depth: rms of (recovered - true sky) in empty regions, units of the analytic ideal noise per resolution element

| sky | N | rms (total), real.-by-real. | mean part | scatter about the mean | empirical ideal rms (check, expect 1) |
|---|---|---|---|---|---|
| grid, positivity on | 8 | **0.93** (0.94, 0.89, 0.98) | 0.86 | 0.37 | 0.95 |
| grid, positivity on | 32 | **1.60** (1.64, 1.56) | 1.60 | 0.10 | 1.24 |
| items | 8 | **0.92** (0.88, 0.93, 0.93) | 0.89 | 0.21 | 0.95 |
| items | 32 | **1.68** (1.71, 1.66) | 1.68 | 0.02 | 1.24 |
| grid, no positivity (diagnostic) | 8 | **2.05** (2.00, 2.11) | -0.01 | 2.05 | 0.99 |
| grid, no positivity (diagnostic) | 32 | **1.70** (1.80, 1.60) | -0.60 | 1.59 | 1.24 |

Ratios to the ideal noise (rms in ideal-noise units, 3c table): with the default positivity (both forms) the ratio is 0.93 (grid) and 0.92 (items) at 8 exposures but 1.60 and 1.68 at 32. It is near 1 at 8 exposures only by accident and it does **not**
follow the square root of the number of exposures: in flux units the rms should halve from 8 to 32 exposures; it falls only to 0.86 (grid) and 0.91 (items) of its 8-exposure value, because it is the offset of 3b (+0.19 surface brightness, independent of N), not noise.
The scatter about the mean is 0.37 -> 0.10 (grid) and 0.21 -> 0.02 (items) ideal-noise units, far below 1: the empty sky sits against the positivity boundary and hides its noise (clipping, not extra depth).
Without the penalty (grid) the error is noise-like: 2.05 and 1.70 ideal-noise units, in flux units a fall by a factor 2.4 from 8 to 32 exposures (expected 2.0; sampling error with 2 realisations about 0.3), but at 1.7 to 2.1 times the ideal noise, not 1 (and the held-out score of those fits is 2 to 8 per cent above the floor).
The ideal coadd's own empirical rms is 0.95 at N = 8 and 1.24 at N = 32 (expected 1; with 2 realisations x 25 correlated positions the sampling error is about 0.15, so the N = 32 value is 1.6 sigma high; not repeated).

**Conclusions.** (a) Item sky: no flux bias above S/N 10 (strict and matched); grid sky: 20 to 30 per cent flux deficit in the true-profile aperture, 3 to 7 per cent in the resolution-matched one; both: a deficit of about 1.1 to 1.2 noise units at S/N 3 to 10 that the ideal estimator does not have. (b) Empty sky: positive, +0.19 surface brightness = +0.9 (8 exposures) and +1.6 to +1.7 (32 exposures) ideal-noise units, both forms, caused by positivity together with the median pedestal gauge (absent without the penalty). (c) The depth test is not passed: the rms ratio is 0.9 (8) and 1.6 to 1.7 (32) with positivity, dominated by the offset and not by noise; without positivity the noise is 1.7 to 2.1 times the ideal and scales roughly as the square root of the number of exposures (2.4 for a factor 4).

## 4. Approximations and shortcuts

* The problem is a single-filter, one-instrument, flat-spectrum field with a round, constant, pixel-integrated two-Gaussian response and no colour, position or time variation; nothing in it tests the chromatic, spatial or temporal parts of the response.
* The response is frozen at (a numerical fit of) the truth, 0.7 per cent relative L1 error; noise scale frozen; registration and background learned. The learned-response case does not converge (C5, C6), so none of the Task 2 numbers describes a fit with a learned response.
* The point-like sources have width 0.03 arcsec (0.6 sky pitch), not zero: a delta function cannot be represented by the sky grid. Their true profile is sampled at the nodes; the strict aperture is 5 to 9 nodes.
* The ideal estimator knows position, profile, response and background; it is an upper bound on what any sky model can do, so ratios below 1 are impossible except by bias, and ratios above 1 include the price of not knowing the profile.
* Source S/N is defined with the first 8 training exposures; at N = 32 the same fluxes have twice the S/N, and the bins in 3a (N = 32) are re-binned with that S/N.
* Sources keep a separation (about 0.25 to 0.85 arcsec by brightness) and empty positions an exclusion (0.25 to 0.6 arcsec plus twice the source width); crowding and confusion are not tested. The responses' wings of S/N 100 sources may still leak about 0.05 noise units into the nearest empty positions; the reference value (true sky in the same aperture) removes the direct overlap but not for the ideal estimator's reference, which uses the noiseless data, equally.
* Numbers of sources per bin are 4 to 12 and positions 25 (correlated): the scatter over sources is a sample scatter, and the errors in 3a are from only 3 (2) realisations and are themselves uncertain by about 50 per cent (3 values) to 70 per cent (2 values).
* The fits are stopped by step count (grid 400, items 300), not by a convergence test; the held-out score is within 2 per cent of the floor but the node-scale sky is not converged (grid strict aperture). Longer fits were not tried; the grid sky error does not change from N = 8 to 32.
* The "recovered" fluxes use the gauge-independent total (fitted mean background minus the true background added as a constant sky); this uses the truth of the background, which a real analysis does not have.
* The truncation of apertures at 4 widths, the sampling of the true profile at the nodes (not the integral over the node cell) and the tent interpolation of the sky are not corrected; the true sky in its own aperture gives the true flux to under 5 per cent (checked in the smoke test).
* Tests are one noise realisation per seed on one population and one geometry; population and dithers are not re-drawn.

## 5. Not finished / open

* Why the learned-response fits (C5, C6) diverge and which kernel parameters move; a learned response was not used in Task 2.
* A third realisation at N = 32 (budget), the positivity-off diagnostic only for the grid (the item sky is positive by construction, so there is no switch).
* Whether a pedestal gauge that uses a robust, source-masked level (instead of the exposure median) removes the +0.19 offset; the position/flux dependence of the offset; item count and seeding sensitivity of the item results.
* Why the grid sky loses 20 to 30 per cent of the flux of point-like sources in the strict aperture and does not improve with more exposures (spectral bias of the optimiser at the node scale, the soft penalty, or the missing node-scale regularisation are all candidates; not separated).
* Hooks in the package that would have helped: (1) an option to exclude named parameter groups (response, noise model, registration) from the optimisers, which I did by subclassing `FieldOperator.param_groups`; (2) a supported way to start the response at a given kernel function; (3) a choice of the pedestal gauge (known level, or none) and a way to report the fitted background next to the sky; (4) `make_field` accepting a truth list and a constant background. Also: `fit_heldout` indexes `model.affine.grad` directly, which fails if the registration is frozen by replacing it in `param_groups`; I left registration learned for this reason.

## 6. Sky zero level: the new option `--sky-zero masked` and the positivity margin (added 17 Nov 2026, after the audit)

What changed (details, formula and reasoning in `AUDIT_2026-11-17.md`, section F): `--sky-zero masked` replaces each training exposure's median in the pedestal
gauge by the median of its *data* over the pixels where the current model's rendered sky, smoothed by a Gaussian of 1.5 detector pixels, satisfies |S| < 0.3 sigma
(re-evaluated at fit progress 0, 0.1, 0.25, 0.5, 0.75). `--sky-positivity-margin 3` lets the sky go down to -p before positivity acts (grid: the penalty acts on sky + p;
items: the items carry sky + p), p = 3 x the noise of the warm-start coadd at one node (median over covered nodes; 2.5 at 8 exposures, 1.1 at 32, in surface-brightness units
where the pixel noise is 1). Same problem, same configurations C2 (grid, 400 steps) and C4 (items, 300 steps) as sections 2 and 3; noise realisations (seeds) 0 and 1 for
every new row. New files: `results/sky_zero/` (`mk_*` = masked, `mm_*` = masked + margin 3; `summary_*.json`; `table.json` from `sky_zero_table.py`).
Two fits ran at a time (two cores, one thread each); one items fit (masked, N = 32, seed 1) took 267 s of wall time, over the 4-minute rule.

**Tables (a) to (c) with the new rows.** Units: ideal-noise units per resolution element (empty regions) or per source (flux bias). Errors in brackets: from the
realisation means (real.) and over the 25 correlated positions (pos.). "fit - ideal": the same difference taken position by position against the ideal estimator on the
same noise, which removes most of the noise the two share (the ideal estimator's own empty-region mean is -0.14 at 8 and -0.35 at 32 exposures on these two seeds).
"rms 32/8": the rms in flux units at 32 exposures over that at 8 (square-root scaling = 0.5). Flux columns: matched aperture, sources pooled over S/N 3 to 10 and over S/N 10 to 100
(bins by the S/N of each N, as in 3a). The "median (today)" rows are recomputed from the section 3 files (3 realisations at N = 8).

| convention | sky | N | total offset (real.) (pos.) | fit - ideal (real.) (pos.) | raw sky offset | fitted - true background [surface brightness] | rms | rms 32/8 | matched ratio S/N 3-10 | bias S/N 3-10 (ideal) | matched ratio S/N >= 10 | bias S/N >= 10 (ideal) | held-out / floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| median (today) | grid | 8 | +0.86 (0.03) (0.05) | +0.90 (0.12) (0.11) | +0.03 | +0.186 | 0.93 | | 0.77 +- 0.07 | -1.24 (-0.17) | 0.954 | -1.73 (+0.37) | 1.014, 1.016, 1.016 |
| median (today) | grid | 32 | +1.60 (0.04) (0.01) | +1.94 (0.11) (0.18) | -0.08 | +0.189 | 1.60 | 0.86 | 0.68 +- 0.05 | -1.80 (+0.42) | 0.876 | -3.19 (+0.14) | 1.005, 1.011 |
| median (today) | items | 8 | +0.89 (0.01) (0.02) | +0.93 (0.16) (0.11) | +0.06 | +0.186 | 0.92 | | 0.79 +- 0.07 | -1.14 (-0.17) | 0.995 | +0.08 (+0.37) | 1.009, 1.011, 1.009 |
| median (today) | items | 32 | +1.68 (0.02) (0.00) | +2.03 (0.12) (0.18) | +0.00 | +0.189 | 1.68 | 0.92 | 0.67 +- 0.05 | -1.68 (+0.42) | 0.939 | -0.99 (+0.14) | 1.000, 0.999 |
| masked | grid | 8 | +0.57 (0.09) (0.10) | +0.71 (0.11) (0.12) | +0.44 | +0.030 | 0.87 | | 0.84 +- 0.08 | -0.82 (-0.10) | 0.954 | -1.75 (+0.13) | 1.013, 1.019 |
| masked | grid | 32 | +0.61 (0.06) (0.05) | +0.95 (0.09) (0.16) | +0.19 | +0.046 | 0.70 | 0.40 | 0.76 +- 0.06 | -1.32 (+0.42) | 0.908 | -2.51 (+0.14) | 1.001, 1.007 |
| masked | items | 8 | +0.40 (0.02) (0.08) | +0.54 (0.19) (0.10) | +0.23 | +0.037 | 0.62 | | 0.86 +- 0.07 | -0.71 (-0.10) | 1.001 | +0.25 (+0.13) | 1.011, 1.014 |
| masked | items | 32 | +0.43 (0.05) (0.00) | +0.77 (0.10) (0.18) | +0.01 | +0.046 | 0.43 | 0.34 | 0.81 +- 0.10 | -0.92 (+0.42) | 0.969 | -0.23 (+0.14) | 0.994, 0.999 |
| masked + margin 3 | grid | 8 | +0.12 (0.22) (0.27) | +0.26 (0.02) (0.25) | -0.09 | +0.046 | 1.92 | | 0.95 +- 0.06 | -0.25 (-0.10) | 0.976 | -0.91 (+0.13) | 1.072, 1.068 |
| masked + margin 3 | grid | 32 | -0.54 (0.03) (0.23) | -0.19 (0.11) (0.16) | -0.96 | +0.047 | 1.58 | 0.41 | 0.95 +- 0.08 | -0.44 (+0.42) | 0.947 | -1.30 (+0.14) | 1.023, 1.020 |
| masked + margin 3 | items | 8 | +0.07 (0.12) (0.22) | +0.21 (0.09) (0.17) | -0.08 | +0.034 | 1.70 | | 0.94 +- 0.04 | -0.37 (-0.10) | 1.022 | +0.70 (+0.13) | 1.050, 1.052 |
| masked + margin 3 | items | 32 | -0.19 (0.14) (0.18) | +0.16 (0.01) (0.14) | -0.60 | +0.046 | 1.25 | 0.37 | 0.93 +- 0.09 | -0.46 (+0.42) | 0.988 | +0.37 (+0.14) | 1.009, 1.006 |

Reading:
* `masked` alone puts the fitted background within +0.03 to +0.05 of the truth (was +0.19) and the total empty-region offset no longer grows with the number of exposures
  (+0.4 to +0.6 at both N), but it does not vanish: that remainder is positivity (raw sky +0.2 to +0.4 above zero at 8 exposures with the zero level nearly right),
  and its scatter about the mean (0.05 to 0.66) is still below 1, i.e. clipped noise.
* With the margin the offset is within about 0.2 to 0.3 of zero (fit - ideal: +0.26, -0.19, +0.21, +0.16, each with an error of 0.15 to 0.25 from 2 realisations x 25
  correlated positions: the 0.2 target is met within the sampling error, not demonstrated below it); the rms falls from 8 to 32 exposures by 0.41 (grid) and 0.37 (items)
  in flux units (target 0.5; the sampling error of these ratios is roughly 0.1); the faint-source flux deficit at S/N 3 to 10 goes from -1.1 to -1.8 to -0.25 to -0.46
  noise units (ratio 0.67 to 0.79 -> 0.93 to 0.95); bright sources are unchanged or closer to 1, except the item sky's matched bias at N = 8 (+0.70 against +0.08; ideal +0.13).
  Costs: the error is 1.6 to 1.9 times the ideal noise (2 realisations), and the held-out score is 5 to 7 per cent above the floor at 8 exposures (was 1 to 2) and 1 to 2 per cent at 32.
* The raw sky with the margin sits below zero by 0.04 to 0.05 in surface brightness (-0.6 and -0.96 noise units at 32 exposures): the masked reference contains faint
  source light the model does not separate (stated limit of the convention), and with the margin the sky can follow it downwards. The total is unaffected.
* Not shown in the table: at N = 32 the S/N 1 to 3 bin loses flux in every new row (matched 0.30 to 0.63 against the ideal 1.03); today's 0.84 there was lifted by the +1.6 offset.

**Why the signed sky is 1.6 to 2.1 times the ideal noise (question 1).** `aperture_scan.py` re-measures the saved skies with Gaussian apertures of other widths, each in the ideal
noise of that aperture. Masked + margin, grid, 8 exposures (2 realisations): 1.59, 1.92, 1.21, 1.96 for widths 0.035, 0.07, 0.14, 0.28 arcsec; 32 exposures: 0.99, 1.58, 1.75, 2.34;
items 8: 1.09, 1.70, 1.84, 3.34; items 32: 0.57, 1.25, 1.43, 2.16 (the ideal estimator's own empirical rms 0.8 to 1.2). The excess does not shrink for wider apertures, so it is not the
price of resolving the resolution element. One fit of 800 steps (grid, 8 exposures, seed 0) has more error, not less: 2.10 against 1.91 at 0.07 arcsec and 3.72 against 2.12 at
0.28 arcsec, so it is not under-convergence either. The excess grows with scale; the per-exposure quadratic backgrounds (6 numbers each, dithers of only +-0.3 arcsec on a
4.4 arcsec detector) are nearly degenerate with the sky's large-scale structure, and the sky grid has about as many free nodes in the covered region (about 7,700) as half the
data at 8 exposures (15,500 pixels): both are candidates; I did not separate them (the test would freeze the background's higher terms, which needs a code hook).

**Positivity at the faint end once the zero level is right (question 2).** Yes: with `masked` the fitted background is 0.03 to 0.05 above the truth, which explains +0.13 to +0.40 of
the total offset, and the rest, raw sky +0.44 (grid) and +0.23 (items) noise units at 8 exposures, is the rectified noise of a sky held at or above zero. It stays at +0.4 to
+0.6 in noise units at 32 exposures (constant in noise units, so it shrinks only as the square root of the number of exposures in surface brightness), and it disappears with
the margin. Development runs (not in the tables, `results/sky_zero/dev_*`): backgrounds left free with no gauge (`--pedestal-gauge 0`) let a constant layer of +0.89 noise units grow in
the sky (the backgrounds fell to 4.08): positivity rectifies the noise and the free constant follows it, so a convention is needed and a tie to the residual (which moves with any layer) is not one.

Approximations of this section: the mask threshold (0.3 sigma) and smoothing (1.5 pixels) were chosen on grid seed 0 with 8 exposures (development runs: one-sided threshold 1.0 gave
a background +0.10 high; one-sided 0.3 with the margin gave -0.09 from selection on the noise of S; unsmoothed symmetric 0.3 gave +0.07 to +0.09), and seed 0 is also in the tables; the margin
was tried at one value (3) only; every other caveat of sections 3 and 4 applies.
