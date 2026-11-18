# Flat field, pixel gain and additive detector pattern in the count-rate fits: diagnosis (17 Nov 2026)

Material: the two 200-minute SMACS 0723 count-rate fits (`checkin/c_fieldop_rate_gain_grid3_*` with the learned pixel gain,
`checkin/c_fieldop_rate_grid3_*` without), the two complete calibrated files in `dl_full/` (NRCB1 F090W, NRCBLONG F277W), and the 12 local
count-rate files with their calibrated twins (all 12 are held out in both fits). Scripts and their raw output: `diag/` (`q1_flat_area.py`,
`q1_in_box.py`, `q2_pattern.py`, `q2_heldout_footprint.py`, `q3_gain_variants.py`, `common.py`, one `.json` each). Synthetic test:
`tests/synth_flat_pattern.py`, results in `synth_out/`. Patch: `flat_pattern.diff` (against `fieldop_orig/`, the untouched copy).

"F" below is the official flat as the data define it: F = count rate x PHOTMJSR / calibrated value, pixel by pixel (exact: both files come
from the same ramp fit). "Box" is the detector region the fit actually used (the bounding box of the training pixels of each detector;
NRCB1 664 x 588 pixels, NRCBLONG 691 x 694, about a tenth of each detector). "sigma" is the median pipeline error of a pixel.
Correction to the brief: the fits have 4 to 6 training exposures per detector pixel (median; `flat_results/rgg3`), not 30; the 30 to 35
"visits" quoted in the package (F45) are optimiser visits.

---------------------------------------------------------------------------------------------------------------------------------------

## Q1. What the pipeline flat contains, and what the count-rate model should multiply by

Measured (`diag/q1_flat_area.json`, 64 x 64-pixel block medians over the whole detector; `diag/q1_in_box.json`, 32 x 32 blocks inside the box):

| | NRCB1 (F090W) | NRCBLONG (F277W) |
|---|---|---|
| AREA extension against the area from the header distortion (SIP), rms of the log ratio, full resolution | 1.0e-4 | 2.7e-4 |
| correlation of large-scale log F with log AREA, whole detector | **+0.79** | **-0.09** |
| slope of log F on log AREA, whole detector | 1.82 | -0.17 |
| large-scale rms of log F / of log (F / AREA), whole detector | 0.0144 / 0.0102 | 0.0290 / 0.0345 |
| the same after a degree-2 polynomial is removed | 0.0046 / 0.0047 | 0.0151 / 0.0151 |
| inside the box: rms of log F / of log (F / AREA) about the mean | 0.0073 / 0.0063 | 0.0066 / 0.0077 |
| inside the box: the same after a plane is removed | 0.0046 / 0.0046 | 0.0022 / 0.0023 |
| inside the box: rms of log AREA about its mean / after a plane | 0.0024 / 0.00007 | 0.0052 / 0.0003 |

What this says.
* The calibrated value is count rate x PHOTMJSR / F, and the AREA extension is attached but not applied: if an extra area factor were
  applied, F would contain AREA exactly on every detector; on NRCBLONG it does not contain it at all (correlation -0.09). (Strictly, the
  ratio defines F, so this is an inference from the structure of F, not a comparison with the reference file `jwst_nircam_flat_0755`, which
  I could not fetch.)
* On the short-wavelength detector the flat contains the pixel-area variation and more (slope 1.8, correlation 0.79): dividing by AREA makes
  it smoother. On the long-wavelength detector the flat's large-scale structure is unrelated to the area: multiplying a model by AREA there
  adds a structure (1.6 per cent rms over the detector, a 0.5 per cent rms tilt over the box) that the official calibration does not have.
  This settles yesterday's doubt: the effect is real and comes from the large scale only (after a degree-2 polynomial is removed the area has
  no structure left, 1e-4).
* The count-rate model must reproduce rate = s x F / C. Today it multiplies by (a / a_nom) x exp(throughput) x G, with G free of every
  monomial of degree <= 2 and the per-detector throughput field pruned (its scale fell to 1e-6 in BOTH fits). So the only large-scale
  multiplicative structure the model has for sources is a / a_nom, once. Nothing counts the area twice today. The residual it leaves,
  F / (a / a_nom) inside the box, is 0.63 per cent rms (NRCB1) and 0.77 per cent (NRCBLONG; 0.66 per cent without the area factor), mostly a
  plane. The area WOULD be counted twice if a learned term were made to equal F (for example a gain map without the polynomial gauge, or a
  throughput field started from the official flat): the learned large-scale term must stand for F / (a / a_nom), not F.
* Which term is supposed to carry the large-scale flat: for sources, the per-detector smooth throughput field (the spatial part of
  gamma, scale `g_space`); for the background, the six per-exposure Legendre terms (degree <= 2, multiplied by G), which take it exposure
  by exposure. Can it? The field can represent it, but on a 20-arcsecond box it is not detected: 0.2 to 0.5 per cent beyond a plane, seen
  only through sources, is below what the validation-likelihood rule keeps, so it is pruned. For the background, degree <= 2 is absorbed per
  exposure and only 0.2 to 0.5 per cent (box) remains beyond it.

What the model should multiply by: keep the physical area factor a / a_nom from the WCS (it is instrument-independent and right where the
flat contains the area), and let the learned multiplicative terms (smooth field plus pixel gain) stand for F / (a / a_nom). On NRCBLONG
that term must then undo the area (about 0.5 per cent rms over this box); with a box this small the cost is negligible, on a full detector it
is 1.6 per cent and needs the throughput field not to be pruned.

## Q2. What the additive detector pattern is doing

Measured from the saved states (`diag/q2_pattern.json`, `diag/q2_heldout_footprint.json`), sizes in units of the detector's median sigma:

| | NRCB1 | NRCB2 | NRCB3 | NRCB4 | NRCBLONG |
|---|---|---|---|---|---|
| logged size (rms over all 64 grid nodes), gain fit / control | 2.06 / 2.11 | 1.52 / 1.65 | 1.75 / 2.12 | 1.53 / 1.95 | 3.01 / 3.69 |
| rms over the training footprint of the detector, gain / control | 0.17 / 0.27 | 0.23 / 0.32 | 0.36 / 0.17 | 0.21 / 0.30 | 0.84 / 0.78 |
| of which beyond a degree-2 polynomial, gain / control | 0.096 / 0.096 | 0.065 / 0.052 | 0.062 / 0.053 | 0.072 / 0.071 | 0.19 / 0.19 |
| correlation of the pattern with minus the mean per-exposure background shape, gain / control | 0.09 / -0.24 | -0.23 / -0.85 | -0.45 / 0.05 | -0.57 / -0.23 | -0.90 / -0.83 |
| correlation of the pattern between the two fits | 0.30 | 0.55 | 0.86 | 0.49 | 0.97 |
| beyond degree 2: correlation with the learned gain smoothed to the pattern grid, gain fit / control | -0.32 / -0.31 | -0.49 / -0.06 | -0.44 / -0.30 | -0.55 / -0.19 | -0.41 / -0.05 |
| flat x background that one additive map would have to stand for (beyond degree 2, pattern scale) | 0.012 | | 0.028 | | 0.041 |
| correlation of the pattern (beyond degree 2) with that flat x background, gain / control | -0.05 / -0.10 | | -0.77 / -0.57 | | +0.18 / +0.16 |
| regression of the pattern on log F (pattern scale), DN/s, gain / control; mean background DN/s | -0.04 / -0.09; 0.09 | | -0.15 / -0.09; 0.09 | | +0.40 / +0.34; 0.47 |
| on the held-out cuts: beyond each exposure's own quadratic, rms (range over its exposures), gain fit | 0.004-0.005 | | 0.023-0.029 | | 0.11-0.18 |

Answer.
* The logged 1.5 to 3.7 sigma is mostly not in the data: (1) grid nodes that no training pixel touches (the box touches about 3 x 3 to 4 x 4 of the
  8 x 8 nodes on the short-wavelength detectors) move freely through the zero-mean constraint; (2) inside the footprint 68 to 97 per cent of
  the variance is a degree-2 surface that trades with the per-exposure background shapes (an exact degeneracy: correlation with minus their
  mean up to -0.85 and -0.90). The part that is determined (beyond degree 2, over the footprint) is 0.05 to 0.10 sigma (about 0.0005 to
  0.0009 DN/s) on the short-wavelength detectors and 0.19 sigma (0.003 DN/s) on the long-wavelength one.
* It is not the flat field in disguise: a single additive map standing in for flat x background would be 0.012 to 0.041 sigma at that scale,
  3 to 8 times smaller than the pattern, and the pattern does not follow it (correlation -0.05 and -0.77, the second with the wrong sign, on
  the short-wavelength detectors; +0.16 to +0.18 on the long-wavelength one, where the regression coefficient 0.34 to 0.40 DN/s happens to
  be close to the mean background 0.47 DN/s but explains only about 3 per cent of the pattern's variance). Whether the rest is a physical
  residual dark could not be told from these data (one shared map per detector cannot be split by filter after the fact); its size is
  compatible with one.
* There is a mild trade between the pattern and the gain map at the pattern's scale: beyond degree 2 they correlate -0.3 to -0.55 in the
  gain fit (against -0.05 to -0.3 for the control's pattern against the same map). The amplitudes are small: the gain map's pattern-scale
  part is 0.0035 rms in log G, about 0.04 to 0.09 sigma times the background.
* On held-out exposures the frozen pattern contributes, beyond what each exposure's own quadratic background re-fits, 0.004 to 0.03 sigma
  (short wavelength) and 0.11 to 0.18 sigma (long wavelength): at most about 0.001 and 0.035 in held-out chi-squared per pixel. 0 to 2.4 per
  cent of held-out pixels lie outside the training footprint. The pattern therefore explains or causes almost none of the held-out residual
  structure; I could not measure the held-out residual maps themselves (the predictions are on CANFAR storage only).

## Q3. Why the learned gain hurts faint pixels, and what shrinkage and a perfect flat would give

Done without re-rendering (`diag/q3_gain_variants.py`): rebuilding the model needs the 123-exposure field and the filter curves on CANFAR
storage, which this machine must not read. Instead, on the 12 local held-out exposures, the excess held-out chi-squared that each gain
variant causes through its pixel-scale mismatch with the true flat, in the limit of a perfect sky and perfect other terms:
excess_p = (1 - exp(x_p - t_p))^2 x (d_p^2 / sigma_p^2 - 1), with t = log of the exposure's own official flat and x = log G of the variant,
both with the degree-2 gauge removed over the box; signal-to-noise bins by (data - median) / sigma. Held fixed: everything else (sky,
backgrounds, pattern, noise model); per-exposure terms were not re-fitted.

Per-pixel background signal-to-noise of these exposures: 9 to 11 (short wavelength), 19 to 30 (long wavelength), so a log-gain error
delta costs about delta^2 x 100 to 900 in chi-squared even on pixels with no source.

| variant (all 12 exposures, 3 detectors) | rms error of log G against the exposure's own flat (NRCB1 / NRCB3 / NRCBLONG) | excess chi2, S/N < 1 | 1-5 | 5-20 | 20-50 | 50-200 |
|---|---|---|---|---|---|---|
| (a) learned gain as is | 0.047 / 0.042 / 0.031 | 0.358 | **0.416** | 0.674 | 1.69 | 26.7 |
| (b1) shrunk, prior width from the map and its noise limit (moments) | 0.043 / 0.030 / 0.021 | 0.212 | 0.170 | 0.318 | 0.94 | 21.4 |
| (b2) shrunk, Gaussian prior of width = official robust rms (0.015-0.019) | 0.040 / 0.028 / 0.020 | 0.201 | 0.158 | 0.291 | 0.82 | 18.7 |
| (b3) best single scale factor (oracle, 0.27-0.42) | 0.037 / 0.027 / 0.020 | 0.189 | 0.146 | 0.272 | 0.75 | 7.5 |
| (c) official flat, one map per detector (mean of the local filters' maps) | 0.003 / 0.003 / 0.006 | 0.010 | 0.010 | 0.019 | 0.03 | 0.11 |
| (d) no gain | 0.043 / 0.030 / 0.022 | 0.260 | **0.168** | 0.330 | 0.98 | 9.1 |

Per detector (`diag/q3_gain_variants.json`): on NRCB1 the learned map helps the bright pixels (50-200: 29.5 against 58.2 without gain), as
in the real scores, and Gaussian shrinkage gives that back (54.6); on NRCBLONG the learned map is worse than nothing at every level.

Answer.
* The learned gain hurts faint pixels because it is noise-dominated: with 4 to 6 training exposures per pixel its noise limit (0.042,
  0.042, 0.019 per pixel) equals or exceeds the true pixel-to-pixel flat (robust rms 0.019, 0.016, 0.014), and its error against the true
  flat (0.031 to 0.047) is LARGER than the flat itself (0.022 to 0.043): applying it adds error. Because faint pixels sit on a background of
  signal-to-noise 9 to 30, that error costs 0.42 against 0.17 (no gain) in chi-squared at S/N 1-5 in the perfect-sky limit. The real fits
  show the same sign with a smaller size (these 12 exposures, reference-model S/N bins: < 1: 1.25 against 1.15; 1-5: 1.44 against 1.39; 5-20: 1.82 against 1.85;
  20-50: 1.58 against 1.68), since there the multiplicative error competes with sky and kernel errors and is partly taken by the learned
  noise terms.
* Shrinkage alone would not make the gain pay at this depth: shrunk maps (b1, b2) are no better than no gain on faint pixels (0.158 to 0.170
  against 0.168) and a Gaussian prior gives back what the map gained on bright, strongly deviant pixels (the official flats have 3 to 6 per
  cent of pixels beyond 5 robust rms). The moment estimate of the prior width from the map itself is 0 on the short-wavelength detectors
  (the map's variance is below its noise limit, i.e. the 200-minute map is not at its maximum-likelihood value), which reduces it to no gain.
* Ceiling with a perfect flat: 0.010 at S/N 1-5 and 0.11 at 50-200: the flat costs 0.16 in chi-squared on faint pixels and about 9 at
  S/N 50-200 when nothing corrects it; this is what a gain map can win, and it needs a per-pixel noise limit well below 0.02.

## Q4. Form of the detector terms for count-rate input

```latex
\hat d_{ep} \;=\; e^{\,g_{d p}}\Big[\frac{a_{ep}}{a^{\mathrm{nom}}_{e}\,C_e}\sum_q w_{fq}\,
   e^{\gamma_{epq}}\,(K_{epq}\ast s_q) \;+\; \sum_{j=1}^{6} b_{ej}\,P_j(\xi_p)\Big] \;+\; A_d(\xi_p) \;+\; \rho_{e,r(p)} + \kappa_{e,c(p)}

\gamma_{epq} = \bar g(\lambda_q) + \tau_{gd}\,\phi_d(\xi_p,\lambda_q) + \tau_{ge}\,u_e ,\qquad
g_d \perp \operatorname{span}\{x^i y^j\,(i+j\le 2),\; h_{d,n}(\xi)\}\ \text{over the observed pixels of } d ,\qquad
g_{dp} \sim \textstyle\sum_{k=1}^{2}\pi_{dk}\,\mathcal N(0,\tau_{dk}^2),

\tau_{dk}^2 \leftarrow \frac{\sum_p r_{pk}\,\big(g_{dp}^2 + (I_p + \tau_{dk}^{-2})^{-1}\big)}{\sum_p r_{pk}},\quad
\pi_{dk} \leftarrow \langle r_{pk}\rangle_p,\quad r_{pk}\propto \pi_{dk}\,\mathcal N(g_{dp};0,\tau_{dk}^2)
```

d = (instrument, detector); a_ep the pixel's solid angle from the WCS and the learned geometry; C_e the conversion of that exposure's
instrument and filter; phi_d the smooth per-detector response field (with the area factor in front it stands for F / (a / a_nom), not F);
g the pixel gain; A_d the additive pattern in that detector's DN/s, the same for every filter, on a bilinear grid with hat functions h_{d,n};
I_p the stored information of log G at pixel p (sum over training cuts of max(d^2 / sigma^2 - 1, 0)).

* (i) Second instrument: every term is keyed by (instrument, detector) or by exposure; the area comes from each exposure's WCS, the conversion
  from its own calibration, the sky stays in MJy/sr; the learned flat means "response per unit solid angle", which has one definition for
  every instrument whatever its official flat convention (NIRCam's short-wavelength flats contain the area, its long-wavelength flat
  apparently does not; only a starting value taken from an official flat needs to be divided by a / a_nom).
* (ii) Gain and pattern are not degenerate: G multiplies light (sources and the filter-dependent background, 0.06 to 0.71 DN/s here), A does
  not; and G is now projected orthogonal to the pattern grid's own functions as well as to the degree-2 monomials, so they share no
  component even for a single filter. The cost is the flat's structure at the pattern scale beyond degree 2, measured at 0.0011 to 0.0028 in
  log (Q2), which the smooth field and the per-exposure background carry instead.
* (iii) Shrinkage without a hand-set strength: a two-component scale mixture (narrow core, wide tail, because real flats have heavy tails)
  whose weights and widths are learned by EM from the map and its per-pixel information, with the posterior-variance term (so a width does
  not collapse merely because the map is shrunk). The prior enters the training objective with weight 1 / (number of training pixels), the
  same scale as the per-pixel mean loss.

### Implementation (`flat_pattern.diff`, 280 added lines including the 170-line test; defaults unchanged)

* `--pixel-gain-prior {off,gauss,mix2}` (default off): `PixelGain.set_prior`, `prior_nll`, `em_update`, `prior_table` (model.py); prior added
  to the joint-step objective divided by the number of training pixels, EM every `--pixel-gain-em-every` (25) steps from progress
  `--pixel-gain-em-start` (0.3) (train.py); learned widths and weights logged and written to `metrics.json["pixel_gain"]["prior"]`.
  Starting values: core width = median per-pixel noise limit of the map, tail = 5 x core, weights 0.9 / 0.1.
* `--pixel-gain-gauge {poly,poly+pattern}` (default poly): with poly+pattern the gain map's gauge also projects out the bilinear hat
  functions of the pattern grid that touch observed pixels (model.py, `PixelGain` and `build_pixel_gain`).
* With the defaults the state after a fit is bit-identical to the original package (`tests/synth_gain.py --inject-rms 0.03 --pixel-gain
  detector --steps 20`: same state hash with `fieldop_orig/` and `fieldop/`).
* Tried and dropped: a gauge on the PATTERN (no degree-2 component over the observed footprint, zero at untouched nodes, by projection in
  node space). On the synthetic field it degraded the recovered pattern (correlation 0.45 and 0.64 against 0.79 and 0.80 today) with no gain
  in held-out score, so it is not in the patch. The reporting of the pattern's size (Q2) should be changed instead (over the observed
  footprint and beyond degree 2); not done.

### Synthetic test (`tests/synth_flat_pattern.py`, 200 steps, about 175 s per run, two seeds)

Field of `tests/synth_gain.py` (two instruments, three filters, 6 training and 2 held-out exposures per filter) with an injected heavy-tailed
flat (log G: N(0, 0.02) for every pixel plus N(0, 0.15) for 5 per cent of them; rms after the gauge 0.037 to 0.043), filter-dependent
backgrounds (0.5, 1.5 and 1 x 8 sigma, so the per-pixel noise limit, 0.026 and 0.046, is above the core width: the SMACS regime), and an
additive pattern on the model's own 5 x 5 grid (rms 0.5 sigma). Held-out chi-squared by bins of the true source signal-to-noise:

| seed | variant | S/N < 1 | 1-5 | 5-20 | > 20 | all | flat: correlation A / B | flat: rms error A / B (injected alone) | pattern: correlation A / B |
|---|---|---|---|---|---|---|---|---|---|
| 0 | no gain | 1.293 | 1.614 | 2.562 | 34.4 | 2.651 | | (0.0413 / 0.0426) | 0.84 / 0.83 |
| 0 | today: gain, no prior | 1.265 | 1.645 | 2.577 | 37.6 | 2.757 | 0.558 / 0.416 | 0.0369 / 0.0445 | 0.79 / 0.80 |
| 0 | prior mix2 | 1.242 | 1.562 | 2.373 | 37.0 | 2.693 | 0.595 / 0.434 | 0.0333 / 0.0385 | 0.81 / 0.82 |
| 0 | **prior mix2 + gauge poly+pattern** | 1.245 | 1.571 | 2.387 | 36.9 | 2.695 | 0.589 / 0.466 | 0.0336 / 0.0377 | 0.84 / 0.84 |
| 1 | no gain | 1.293 | 1.624 | 2.966 | 69.2 | 4.082 | | (0.0368 / 0.0427) | 0.87 / 0.87 |
| 1 | today: gain, no prior | 1.274 | 1.571 | 2.951 | 60.8 | 3.733 | 0.505 / 0.402 | 0.0376 / 0.0442 | 0.79 / 0.87 |
| 1 | **prior mix2 + gauge poly+pattern** | 1.255 | 1.548 | 2.788 | 65.7 | 3.893 | 0.498 / 0.436 | 0.0320 / 0.0384 | 0.87 / 0.88 |

Pixels per bin about 7000, 750 to 920, 530 to 650, 330. Truth-model chi-squared 0.988 (all).

* Recovery of the injected flat: rms error lower than today on both detectors and both seeds (by 9 to 15 per cent), and now BELOW the error
  of using no gain at all (today's map is worse than nothing on detector B, 0.0445 against 0.0426, as on SMACS); correlation equal or higher
  (0.589 / 0.466 against 0.558 / 0.416; seed 1 0.498 / 0.436 against 0.505 / 0.402). The comparison uses the degree-2 gauge only, so it is
  slightly unfavourable to the poly+pattern gauge.
* Held-out prediction: better than today at S/N < 1 (-0.020, -0.019), 1-5 (-0.074, -0.023) and 5-20 (-0.190, -0.163) in both seeds, and
  better than no gain at every S/N < 20. Above S/N 20 (330 pixels, dominated by the unconverged sky) the sign differs between the seeds.
* Learned prior: core width 0.026 to 0.034 (injected 0.020), tail 0.062 to 0.067 on A with weight 0.02 (injected 0.15, 0.05); on B the two
  components merged (0.031 / 0.032), i.e. the tail is not resolved at a noise limit of 0.046. The deviant pixels are shrunk more than today
  (tail-pixel error 0.13 to 0.18 against 0.10 to 0.15): the mixture keeps only part of what a free map catches on strongly deviant pixels.
* The pattern is recovered as well as or better than today; the gain-pattern trade is removed by construction.

## Approximations and shortcuts

* Q1: F's large-scale structure from block medians (64 or 32 pixels) of the exact per-pixel ratio, two exposures, one filter per channel;
  no comparison with the flat reference file itself; header area by central differences of the SIP solution on a 32-pixel grid.
* Q2: per-filter mean backgrounds are the means of the stored per-exposure medians (they contain some source light); "flat x background
  at the pattern scale" uses a least-squares projection of the official log flat onto the 8 x 8 bilinear grid restricted to the box; flats
  are the local exposures' (3 filters for each of 3 detectors; NRCB2 and NRCB4 have no flat here).
* Q3: perfect-sky limit (sky, backgrounds, pattern, noise model held fixed; no re-render, no re-fit of per-exposure terms); its S/N bins are
  (data - median) / sigma, not the scorer's reference-model S/N; the per-detector official map (c) is the equal-weight mean of the local
  exposures' own maps, which includes the scored exposure's map (optimistic, at most by a third of the filter-to-filter difference); the
  noise limit uses the pipeline sigma (the fits learned noise scales of 0.69 to 0.83, so the true information is about 1.5 to 2 times
  larger) with the factor N / (N - 1) of `flat_check.py`.
* Patch: the information I_p in the EM step uses the pipeline sigma (same caveat: the posterior-variance term is overstated, which biases the
  widths up a little); the EM uses the current map, which may not be converged (this biases the widths down); Adam moments are not projected
  after the gauge (as for the existing gauge); starting values and EM schedule (from progress 0.3, every 25 steps) are fixed, the strength is
  not; one mixture per gain map (per detector, or per detector and filter).
* Synthetic test: two seeds, 200 steps (the loss was still falling; the sky is not converged above S/N 20), one regime of depth; the injected
  pattern lies in the model's span.

## What I did not check

* The real held-out predictions with variants (a) to (d): not possible on this machine without the CANFAR files (field, filter curves,
  predictions). Q3 is the perfect-sky limit only.
* The patch on real data (no count-rate fit was run).
* Whether the long-wavelength official flat lacks the area by construction (reference-file documentation) or the illumination happens to
  cancel it; per-filter flats on the long-wavelength detector (official maps correlate only 0.85 to 0.90 between its filters).
* Why the throughput field is pruned, quantitatively (its evidence trace), and the pattern's size reporting.

## Recommended arguments for the next count-rate fit on a field with many more exposures per pixel

```
--input-level rate --rate-units sky --rate-wcs cal --nuisance-steps exposure --dq-ignore-bits 2 --sky-zero masked
--pixel-gain detector --pixel-gain-poly 2 --pixel-gain-gauge poly+pattern --pixel-gain-prior mix2
--pixel-gain-em-start 0.3 --pixel-gain-em-every 25 --lr-gain 3e-3 --pixel-gain-start 0.1 --pattern-grid 8
```

With 6 times the exposures per pixel of SMACS the noise limit falls from 0.042 to about 0.017 (short wavelength) and 0.008 (long
wavelength), at or below the official robust rms (0.015 to 0.019): the regime where the learned map can pay, with the learned prior
deciding how much. Check in the log that the core width stays above the median noise limit and that the tail weight is not at its 0.01
floor; check the throughput field's scale (`tau.g_space`): on a field wider than 20 arcseconds it should no longer be pruned, since it is the
only term that can carry the large-scale F / (a / a_nom) for sources (1.6 per cent rms over a whole long-wavelength detector).
