# Rate-level input and a learned per-pixel gain (flat field): build, tests, launch recipe

16 Nov 2026. Folder `/home/claude/work/fieldop_gain/`. Question: does the flat field calibrate itself from the science exposures alone?
Drop rule (from the request): drop the idea for this data volume if (1) the learned gain map correlates below 0.3 with the official flat's
pixel-to-pixel part where the expected noise limit allows more, or (2) the rate-level fit with a gain map predicts held-out rate pixels
worse than the calibrated-level fit predicts held-out calibrated pixels by more than 0.05 in chi2/N at S/N 5 to 50.

**Status:** everything below is built and tested on this CPU machine. No real-data fit was run: those are GPU jobs, and the recipe is in
section 6. The command-line path with `--input-level rate` could not run here because torchsky is not installed on this machine (see
section 7).

## 1. What differs between `_cal` and `_rate` (measured on jw02736001001_02101_00009_nrcb1, both CAL_VER 3.0.0, CRDS 1584)

| | `_rate` | `_cal` |
|---|---|---|
| units | DN/s | MJy/sr (PHOTMJSR 3.592 for this one) |
| extensions | SCI, ERR, DQ, VAR_POISSON, VAR_RNOISE, ASDF | adds AREA and VAR_FLAT |
| ERR | sqrt(VAR_POISSON + VAR_RNOISE) | cal ERR² = (rate ERR / C)² + VAR_FLAT exactly (median ratio 1.0000000); VAR_FLAT is 0.4 % of the variance at low S/N, 1.3 % at S/N 20-50, 6 % at 50-200, 37 % above 200 |
| DQ | identical except: cal adds NO_FLAT_FIELD (value 2^18) on 3,673 pixels, and DO_NOT_USE on 2 more | |
| header WCS | linear RA---TAN from the pointing (CRVAL, PC, CDELT), **no distortion** | RA---TAN-SIP (order 5) from assign_wcs |
| stage-1 steps and references | the same (dark 0344, linearity 0057, same ramp fit) | plus flat 0674, photom 0175, area 0300 |

* The ratio C = rate SCI / cal SCI is exact pixel by pixel. Both files come from the same ramp fit, so the noise cancels. The scatter of
  C is the same at |rate| > 10 ERR and at |rate| < 0.5 ERR (1.6 % robust). It is finite on 99.1 % of the detector, and its median
  times PHOTMJSR is 1.004 to 1.008 over the 12 sample exposures.
* WCS: the `_rate` header WCS is 0.07″ (median) and 0.24″ (maximum) away from the `_cal` SIP WCS over NRCB1, or 0.09″ rms after
  removing the mean. That is 2 to 8 short-wavelength pixels, too large to use. **For rate input the package therefore takes the header
  WCS (and PIXAR_A2) from the matching `_cal` file's SCI header** (`--rate-wcs cal`, the default; `--rate-wcs rate` uses the `_rate`
  header). The WCS is a geometric step, not a photometric one. On CANFAR the `_cal` file sits at its splits path, so the run needs both
  folders.
* Many pixels carry JUMP_DET (value 4): 434,017 here, about 10 %, identical in both files. They are excluded with the default flag
  policy, and also with `--dq-ignore-bits 2`, which keeps only SATURATED. This is unchanged by this work, but it is a large pixel loss
  worth a look.

## 2. What was built

### 2.1 Per-pixel gain in the model (`fieldop/model.py`: `PixelGain`, `build_pixel_gain`, `FieldOperator.attach_pixel_gain`, `gain_of`)

    d_e(p) = N_e[ G(p) S_e(p) ] + G(p) sum_j b_ej P_j(xi) + pattern_det(xi) + stripes_e(row, col),   G(p) = exp(logg_k(e)(p))

* `--pixel-gain {off, detector, detector-filter}` (default off). There is one map per detector, or per detector and filter.
* G multiplies the sky signal S and the per-exposure quadratic background, which is sky and zodiacal light falling on the pixel.
* The per-detector 8×8 pattern and the NIRCam row and column stripes are added after G. They stand for additive detector terms (residual
  dark, amplifier offsets, 1/f noise), which do not pass through the flat.
* The optional non-linearity acts on G·S.
* Allocation: the bounding box of the detector pixels valid in at least one *training* cut. Pixels outside the box, or never observed in
  training, keep G = 1, so held-out pixels beyond the training footprint get no flat correction.
* Stored with the map: an observed mask, the training cuts per pixel, and an information map I_p = Σ_e max(d²/σ² − 1, 0).
  1/√I_p is the Cramér–Rao bound of logg if all else were known.
* **Gauge (F44), exact, by projection after every gain step, no penalty.** Over each map's observed pixels, logg is made orthogonal
  (uniform weights) to the monomials of total degree ≤ `--pixel-gain-poly` (default 2: mean, two tilts, three curvatures) of the pixel
  coordinates scaled to [−1, 1]. Unobserved pixels are set to 0. This uses a QR basis per map, computed once.
  * The mean is degenerate with the throughput gauge and the per-exposure gains u_e.
  * The degree ≤ 2 part is degenerate with the smooth throughput detector field and the quadratic backgrounds.
  * Smooth structure above degree 2 stays free in the map. Where the throughput detector axis is on (it was pruned in the SMACS runs),
    that part is shared with it, which is a residual degeneracy.
  * There is no smoothness or other prior.
* **Optimisation (F45).** The map has its own Adam with step `--lr-gain` (default 3e-3 in log G), following the same warm-up and cosine
  factor as the operator and with no gradient clipping. It is stepped only in the joint step, from progress `--pixel-gain-start` 0.1 on,
  so that the sky exists first. It is frozen in the held-out and validation fits and kept out of the operator group (`param_groups`).
  * **How the step was chosen:** a scan on the synthetic field over 3e-4, 1e-3, 3e-3 and 1e-2 (section 4), taking the best recovery and
    held-out score. This is the only tuning of the new group. No other parameter or method was tuned.
  * **Caveat:** a synthetic pixel gets about 130 gradient visits in 200 steps; a SMACS pixel gets only about 30 to 35 in a 200-minute fit
    (10,528 steps × 5 patches; median 4 training cuts per pixel out of about 6,000 patches). The map may be under-converged on real data
    (at 3e-4 the synthetic map reached only a quarter of its amplitude).
* **Default off is bit-identical.** `tests/synth_small.py --steps 40` gives state sha256 `ff0f1183004ab6a1…` for both
  `/home/claude/work/fieldop_items` and this package with the gain off (checked again on 16 Nov).
* Prediction files of rate-trained fits carry `meta.units = "rate"` (and `units_note`, `pixel_gain`). `fieldop_state.pt` gains
  `pixgain.*` tensors and `pixel_gain_meta` (labels, boxes, offsets, gauge residual), and `metrics.json` gains `pixel_gain`. The log
  prints, per map, the box, observed pixels, cuts per pixel and the information limit.

### 2.2 Rate-level scoring (`score_paired.py --level rate --rate-dir DIR`, also a new `score` subcommand)

* Data are the `_rate` SCI; sigma is the `_rate` ERR × the protocol's per-pixel `noise_factor`. Pixels are the protocol mask, and
  `--dq-zero` applies the `_rate` file's flags.
* Predictions in rate units (meta `units = "rate"`) are used as they are. Calibrated-unit predictions are multiplied by
  C = rate SCI / cal SCI of the same exposure (`conversion_map`).
* C is accepted where both values are finite and non-zero and C × PHOTMJSR lies in [0.2, 5]. Scored pixels where C is rejected get the
  median of accepted C in their 5×5 neighbourhood (the smooth flat only), else 1/PHOTMJSR. On the sample, 0 scored pixels needed a fill.
* The `_cal` and `_rate` headers are compared (CAL_VER, CRDS context, stage-1 references) and any mismatch is reported. There were none.
* Output goes to a `rate/` subfolder. The `score` subcommand reports `score.py`'s own bins plus a pooled S/N 5–50 column, which is the
  drop rule's range. A rate-unit prediction scored at cal level is refused.

### 2.3 Official flat as the truth (`flat_check.py`)

* From `_cal`/`_rate` pairs of one detector, the official map is log(C × PHOTMJSR), averaged over pairs of a filter. For one map per
  detector it is weighted over filters by each filter's training background information. The same gauge projection as the map is then
  applied.
* It is compared with the learned map from `fieldop_state.pt` (`--state`). Reported:
  * correlation, also high-passed (minus a 9×9 median) and on background-dominated pixels;
  * rms of the difference and the slope;
  * the noise limit 1/√I_p and the correlation expected at that limit, overall and in bins of the per-pixel limit;
  * the drop-rule verdict: correlation where the expected correlation exceeds 0.5.
* With `--ref-state` (a fit without a gain map) it predicts the footprint, coverage and noise limit from the state's geometry and
  background S/N. Section 5 uses this mode.
* Tested end to end on a constructed state: the official NRCB1 map plus noise at the noise limit (`tests/check_flat_check.py`).
  flat_check returns correlation 0.668 against 0.667 expected and slope 1.001 (`flat_results/check_constructed_NRCB1.json`).

### 2.4 Synthetic field and tests

* `fieldop/synth.make_field(pix_gain_rms=…, bkg_mean=…)` injects a white per-pixel log-gain map per detector from its own random
  stream: d = G·(signal + background) + noise. With the defaults the field is unchanged, which the checksum above confirms.
* `tests/test_pixel_gain.py`: 6 unit tests, all passing.
  * Off leaves the model inert.
  * The gauge is exact (projected components below 1e-6), idempotent, removes a pure degree-2 polynomial entirely, and with order 0
    removes only the mean.
  * Lookup and box handling are correct.
  * G multiplies S and the quadratic background but not the pattern and stripes, in both the batched and the plain renderer.
  * The count and observed maps match the cuts, and `detector-filter` gives 3 maps.
  * The gradient reaches exactly the 36 pixels of a 6×6 patch.
* `tests/synth_gain.py`: the validation runs below. `tests/check_rate_scoring.py`: the scorer identity check.

## 3. Rate-level scoring check (12-exposure sample, `scores_sample/`, `tests/gain_results/check_rate_scoring.json`)

chi2/N pooled over the sample (without offset, then with offset):

| method | level | S/N <1 | 1–5 | 5–20 | 20–50 | 50–200 | >200 | S/N 5–50 |
|---|---|---|---|---|---|---|---|---|
| stage3_blot | cal | 1.046 / 0.985 | 1.144 / 1.066 | 1.307 / 1.248 | 1.113 / 1.058 | 4.310 / 4.194 | 66.96 / 66.72 | **1.251 / 1.193** |
| stage3_blot | rate | 1.072 / 1.009 | 1.178 / 1.099 | 1.339 / 1.279 | 1.133 / 1.077 | 4.410 / 4.290 | 73.37 / 73.12 | **1.280 / 1.220** |
| fieldop_fix_200 | cal | 1.134 / 1.029 | 1.278 / 1.180 | 1.574 / 1.499 | 1.393 / 1.313 | 2.555 / 2.416 | 69.33 / 69.25 | **1.522 / 1.446** |
| fieldop_fix_200 | rate | 1.162 / 1.054 | 1.318 / 1.216 | 1.612 / 1.535 | 1.418 / 1.335 | 2.655 / 2.512 | 82.39 / 82.31 | **1.556 / 1.477** |

* The rate − cal difference at S/N 5–50 (without offset) is **+0.029 for stage3_blot and +0.034 for fieldop_fix_200**, about 2.3 %.
* Scoring at rate level with the calibrated sigma carried over (C × protocol sigma) reproduces the calibrated scores to **4e-16** in
  every bin. C and the pixel handling are therefore exact, and the whole difference comes from the sigma definitions.
* Decomposition, measured on the exposure with a full `_cal`:
  * the protocol sigma is 1.0125 × (current cal ERR × noise factor) on every pixel, because the masks were built from the earlier
    `_cal` files: about 2.5 % in variance;
  * the flat-field variance missing from the rate ERR adds 0.4 % at low S/N, 1.3 % at 20–50, 6 % at 50–200 and 37 % above 200 (hence
    67 → 73 in the top bin).
  * Over the sample, the variance ratio of protocol sigma to rate sigma/C is 1.025, 1.031, 1.021, 1.017, 1.021 and 1.059 by bin.
* **Consequence for the drop rule's second criterion.** Comparing a rate-level fit at rate level with a calibrated fit at calibrated
  level carries about +0.03 at S/N 5–50 for free, more than half of the 0.05 margin. The fair comparison is both fits at rate level,
  paired on identical pixels (`paired --level rate`), with the calibrated fit's prediction converted by C. I recommend judging
  criterion (2) that way and reporting the cross-level number next to it.

## 4. Synthetic validation (`tests/gain_results/`, 200 steps each, 116–143 s fits, under 3 minutes per run)

The field is `tests/synth_small.py`'s (detectors A 40×40 pixels with 12 training exposures per pixel, B 34×34 with 6; injected PSF and
smooth throughput variation). Changes: the background was raised from 5 to 40 (noise 1), so that a pixel's per-exposure background S/N is
40, and a white injected log-gain of 3 % rms per detector (0.030 and 0.029 rms after the gauge projection). The held-out chi2 is on 4,356
pixels of 3 held-out exposures; "background" means the 3,313 pixels whose true signal is below 0.5 σ. The truth (noise only) scores
0.990 on all pixels and 0.985 on background pixels.

| run | injected flat | gain map, step | held-out chi2 all / background | corr A / B (expected at noise limit 0.976 / 0.950) | slope A / B | rms(learned − injected) A / B (noise limit 0.0067 / 0.0096) |
|---|---|---|---|---|---|---|
| r0 | none | off | **3.048 / 1.105** | | | |
| r1 | 3 % | off | **4.744 / 2.853** | | | |
| r3 | 3 % | on, 3e-4 | 4.094 / 2.244 | 0.877 / 0.834 | 0.23 / 0.18 | 0.0235 / 0.0244 |
| r2 | 3 % | on, 1e-3 | 3.567 / 1.645 | 0.903 / 0.862 | 0.61 / 0.49 | 0.0148 / 0.0172 |
| r4 | 3 % | on, 3e-3 (chosen) | 3.684 / **1.480** | **0.920 / 0.888** | 0.91 / 0.84 | 0.0120 / 0.0136 |
| r6 | 3 % | on, 1e-2 | 3.806 / 1.503 | 0.920 / 0.891 | 0.97 / 0.96 | 0.0125 / 0.0144 |
| r5 | none | on, 3e-3 | 3.630 / 1.323 | (learned rms 0.0115 / 0.0130) | | |

* **(a) Without the map, the injected flat degrades the held-out score:** background pixels go from 1.105 to 2.853, all pixels from 3.05
  to 4.74.
* **(b) With the map, the score improves but does not return to the no-flat-error level.**
  * Background pixels reach 1.48, against 1.105. This removes 79 % of the excess on background pixels and 62 % on all pixels
    (3.68 against 3.05 without a flat error and 4.74 with one).
  * The injected map is recovered with correlation 0.92 (A) and 0.89 (B), against 0.98 and 0.95 expected at the noise limit.
  * The map error is **1.8 × (A) and 1.4 × (B) the Cramér–Rao noise limit**, so it does not reach the expected noise limit.
  * r5 (map on, nothing injected) shows the same floor: the map learns 0.0115 rms of spurious structure (1.7 × the limit), and costs
    +0.22 on background pixels and +0.58 on all pixels against r0.
  * The excess in r4 is about what that map error predicts (background 40 × 0.012 = 0.5 σ, so about +0.23 in chi2) plus the r0 → r5
    cost.
  * Likely causes: Adam's stochastic noise at a fixed step (the cosine decay only goes to 0.05 of the step), and the map absorbing
    sky and kernel model errors. The fits themselves are not converged: the held-out chi2 on all pixels is 3.05 against 0.99 for the
    truth, and the logged loss was still fluctuating at the end. I did not test which cause dominates.
* **(c) The gauge holds**, tracked every 10 steps in every run: the largest projected component (mean and degree-2 part, as an rms over
  observed pixels) is at most 6e-10, the largest |mean logg| at most 5e-10, and unobserved pixels are exactly 0.
* The step is a trade-off: 3e-4 and 1e-3 under-converge (slope 0.2 to 0.6); 1e-2 gives the full amplitude (slope 0.97) but more noise,
  with the same correlation and a slightly worse held-out score. 3e-3 was chosen.

## 5. Real data: how large is the official flat, and how well can the dithers measure it? (`flat_results/pred_*.json`, `.png`)

* Sources: the reference state `fieldop_fix_200_state.pt` and its `metrics.json` (geometry of the 86 training exposures; per-exposure
  background S/N = median data / median pipeline sigma).
* The official maps come from the 12 sample pairs: NRCB1 and NRCB3 in F090W, F150W and F200W; NRCBLONG in F277W, F356W and F444W, two
  exposures each. They were fetched to `dl/rate`, with the `_cal` SCI from `score_paired/tests/data/cal`.
* The degree-2 gauge was applied in every case. The noise limit is the background-only Cramér–Rao bound (not inflated).

| detector | footprint pixels | training cuts per pixel, median (10–90 %) | background S/N per pixel per exposure | official p2p rms (robust) | noise limit per pixel, median (10–90 %) | expected corr, best case (inflated ×N/(N−1)) |
|---|---|---|---|---|---|---|
| NRCB1 (SW) | 254,670 | 4 (2–10); 124k pixels seen twice | 8.9 (F090W), 12.1, 12.2 | 0.042 (0.019) | 0.043 (0.028–0.058) | **0.67 (0.63)** |
| NRCB3 (SW) | 168,781 | 6 (3–11) | 8.8, 11.4, 11.6 | 0.031 (0.017) | 0.037 (0.028–0.052) | **0.61 (0.57)** |
| NRCBLONG (LW) | 334,157 | 5 (2–15) | 21.1 (F277W), 22.0, 30.9 | 0.021 (0.014) | 0.017 (0.010–0.032) | **0.69 (0.65)** |

* The official short-wavelength flat has heavy tails: 6 % (B1) and 3 % (B3) of pixels lie beyond 5 robust rms. Its rms of 3 to 4 % is
  driven by them, while the robust rms is 1.7 to 1.9 %. The long-wavelength flat is closer to Gaussian (0.5 % beyond 5 robust rms).
* Between filters, the official maps correlate at 0.98 to 0.99 (SW) but only **0.85 to 0.90 (LW)**. One map per detector is adequate in
  SW; in LW it caps the attainable agreement with any single filter's flat. `--pixel-gain detector-filter` exists for this, at a third
  of the data per map.
* **The test is marginal.** Even at the Cramér–Rao limit (background only, perfect sky and kernel model), the expected correlation is
  0.6 to 0.7. The synthetic runs reached a map error of 1.4 to 1.8 × the limit. At 1.6 × the limit the expected correlations fall to
  about **0.49 (B1), 0.44 (B3) and 0.51 (LONG)** (from the rms noise limits 0.047, 0.039, 0.022): above the 0.3 threshold, but not by much.
* Model errors near bright sources come on top. In SMACS the 20″ box is a crowded cluster core.
* The step-size caveat (about 4 × fewer gradient visits per pixel than on the synthetic field) could push the result below 0.3 through
  under-convergence alone.
* A pass would be a correlation of 0.4 or more. A failure in the 0.2–0.3 range would not distinguish "the flat cannot be learned" from
  "this optimiser and data volume are too small"; the by-noise-bin table in flat_check separates part of that.

## 6. Launch recipe (CANFAR)

    BASE="--field SMACS --split dither --protocol-root /arc/projects/ots/adass2026/protocol/v1 --svo-cache /arc/projects/ots/adass2026/svo_cache"
    REF="--budget-min 200 --batch-patches 5 --ckpt-kernel 1 --reg-prepass-steps 100 --held-steps 200 --held-reg-steps 150 --log-every 100 --dq-ignore-bits 2"
    RATE="--input-level rate --rate-dir /arc/projects/ots/torchsky-data/smacs0723_raw/rate --rate-wcs cal"
    GAIN="--pixel-gain detector --pixel-gain-poly 2 --lr-gain 3e-3 --pixel-gain-start 0.1"
    ITEMS="--sky-form items --item-count 60000 --item-levels 3 --item-raster pyramid --item-patch sheared --item-buckets coarse --item-chunk 32000000 --item-max-width 2.56"

    # (i)   grid, rate, no gain map
    python -m fieldop $BASE --method-name fieldop_rate_grid       --out /arc/projects/ots/adass2026/fieldop/fieldop_rate_grid       $REF $RATE
    # (ii)  grid, rate, gain map
    python -m fieldop $BASE --method-name fieldop_rate_gain_grid  --out /arc/projects/ots/adass2026/fieldop/fieldop_rate_gain_grid  $REF $RATE $GAIN
    # (iii) items, rate, no gain map / with gain map
    python -m fieldop $BASE --method-name fieldop_rate_items      --out /arc/projects/ots/adass2026/fieldop/fieldop_rate_items      $REF $RATE $ITEMS
    python -m fieldop $BASE --method-name fieldop_rate_gain_items --out /arc/projects/ots/adass2026/fieldop/fieldop_rate_gain_items $REF $RATE $GAIN $ITEMS
    # calibrated twins with IDENTICAL arguments (needed for criterion 2, unless runs with exactly $REF [$ITEMS] already exist)
    python -m fieldop $BASE --method-name fieldop_cal_grid        --out /arc/projects/ots/adass2026/fieldop/fieldop_cal_grid        $REF
    python -m fieldop $BASE --method-name fieldop_cal_items       --out /arc/projects/ots/adass2026/fieldop/fieldop_cal_items       $REF $ITEMS

* All six argument lines were parsed with the package's own parser.
* The rate runs read the `_cal` files at their splits paths too (header WCS only), so `/arc/projects/ots/torchsky-data/jwst_smacs0723/cal`
  must be readable. The package must be this folder's `fieldop/`.
* Things to look for in `fieldop.log`:
  * the `pixel gain map …` lines (box, cuts per pixel, information limit);
  * `pixel gain: … maps` at the fit start;
  * `pixel_gain_gauge` in metrics.json (should be about 1e-9);
  * the held-out lines (in DN/s for rate runs, against the pipeline ERR).

Scoring (put `score_paired.py` from this folder next to `score.py` in `protocol/v1/code/`; RD = the rate folder):

    S=/arc/projects/ots/adass2026/protocol/v1/code/score_paired.py; RD=/arc/projects/ots/torchsky-data/smacs0723_raw/rate
    # every method at rate level (calibrated predictions converted by C), pooled S/N 5-50 column = the drop-rule range
    python $S score  --field SMACS --split dither --dq-zero --level rate --rate-dir $RD --methods fieldop_rate_grid fieldop_rate_gain_grid fieldop_rate_items fieldop_rate_gain_items fieldop_cal_grid fieldop_cal_items stage3_blot
    # calibrated twins at their own level (for the cross-level number)
    python $S score  --field SMACS --split dither --dq-zero --methods fieldop_cal_grid fieldop_cal_items stage3_blot
    # criterion 2, fair form: rate fits against their calibrated twin, same level, identical pixels
    python $S paired --field SMACS --split dither --dq-zero --level rate --rate-dir $RD --methods fieldop_rate_gain_grid fieldop_rate_grid --reference fieldop_cal_grid
    python $S paired --field SMACS --split dither --dq-zero --level rate --rate-dir $RD --methods fieldop_rate_gain_items fieldop_rate_items --reference fieldop_cal_items
    # criterion 1: learned map vs official flat, per detector (pairs: exposure 00009 of the three activities; any training or held-out pair works)
    for d in nrcb1 nrcb2 nrcb3 nrcb4 nrcblong; do
      python flat_check.py --state /arc/projects/ots/adass2026/fieldop/fieldop_rate_gain_grid/fieldop_state.pt \
        --cal-dir /arc/projects/ots/torchsky-data/jwst_smacs0723/cal --rate-dir $RD \
        --roots jw02736001001_02101_00009_$d jw02736001001_02103_00009_$d jw02736001001_02105_00009_$d \
        --label $(echo $d | tr a-z A-Z) --out flat_$d.json --png flat_$d.png
    done      # and the same with fieldop_rate_gain_items

`flat_check.py` imports `score_paired` and `fieldop.model`, so run it from a folder that holds both (this folder layout). It runs on a CPU
in about a minute per detector; the state plus three pairs are about 0.5 GB.

## 7. Approximations and shortcuts (new: F43–F46 in `fieldop/__init__.py`; P7 in `score_paired.py`)

* **F43 model form.** G multiplies the sky signal and the quadratic background, not the pattern and stripes. Maps cover the training
  footprint; G = 1 elsewhere. One map per detector assumes a filter-independent flat (0.85–0.90 between LW filters). Multi-tile overlaps
  count twice in the diagnostic maps.
* **F44 gauge.** Degree ≤ 2 projected out per map with uniform weights. Smooth structure above degree 2 is free and partly shared with
  the throughput detector field when that axis is on. The information map is a background-dominated Cramér–Rao bound.
* **F45 optimiser.** Adam with step 3e-3, chosen by a 4-value scan on the synthetic field; not clipped; started at 10 % of the fit;
  momentum moves pixels between visits; possibly under-converged on SMACS (about 30 visits per pixel).
* **F46 rate input.** WCS from the `_cal` header. The rate ERR has no flat variance. The sky still has per-filter units, because the
  photometric conversion is not an input (F15, unchanged).
* **P7 rate scoring.** C is exact where accepted, with a local-median fill otherwise (none was needed). Sigma has a different
  provenance from the protocol sigma (about 2.5 % in variance plus the flat-variance term). The offset is constant in rate units.
* **flat_check.** The truth includes the flat file's large-scale part (removed by the gauge). The `--ref-state` footprint uses a rotated
  square without distortion (1–2 pixels at the edges). The detector-mode truth weights filters by background information.
* **Synthetic validation.** Background raised to 40 to make the test informative. The injected map is white, while the real flat has some
  structure. 200 steps, not converged (all-pixel held-out chi2 about 3). The PSF errors of this small field dominate the all-pixel score.

## 8. Not done / surprises

* **Not done:**
  * any real-data fit (GPU, coordinator);
  * a CPU run of the `cli` path with `--input-level rate` (torchsky is not installed here). The save, log and units-meta code in
    `cli.py` is therefore untested by execution: it was parse-checked, and the recipe arguments parse;
  * a test of which cause sets the 1.4–1.8 × noise-limit floor of the learned map;
  * maps per detector and filter on the synthetic field;
  * checks of NRCB2 and NRCB4 statistics: no pairs were fetched for those detectors.
* **Surprises:**
  * the `_rate` WCS has no distortion (0.24″ errors), and the package would have used it;
  * most footprint pixels are seen by only 2 to 6 training exposures, not about 20 (the 20″ box moves across the detector with the
    dithers);
  * the protocol sigma is 1.25 % larger than the current files' ERR;
  * 10 % of pixels carry JUMP_DET and are excluded under every current flag setting;
  * the official LW flats differ noticeably between filters (correlation 0.85–0.90).

Files fetched from CANFAR (read only): 12 `_rate` files of the sample (`dl/rate/`), the full `_cal` of jw02736001001_02101_00009_nrcb1
(`dl/cal/`), and `fieldop/fieldop_smacs_fix_200_s0/metrics.json` (`dl/fix200_metrics.json`). About 1.1 GB in `dl/`, which can be deleted.
