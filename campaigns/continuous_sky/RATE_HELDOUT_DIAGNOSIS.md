# Count-rate fits fail on held-out exposures: diagnosis and fix

17 Nov 2026. Fits concerned: `fieldop_rate_grid` (count-rate input, grid sky, no pixel gain map) and `fieldop_rate_gain_grid` (the same with a
gain map per detector pixel), both 200 minutes on a GPU with the `fieldop_gain` package. All checks below ran on this CPU machine. They used
the saved states in `/home/claude/work/checkin/`, the 12 local count-rate exposures, and synthetic fields.

## 1. Cause

The warm start of the sky was built in the wrong units for count-rate input.

* The count-rate model multiplies the rendered sky by the pixel area alpha (arcsec², approximation F15). Its sky is therefore a count rate
  per arcsec².
* The warm start (`data.splat_coadd`, used in `cli.make_sky` and again after the registration pre-pass) splatted the count rate per *pixel*.
* As a result, every source began too faint by the pixel area: 0.00095 arcsec² for short-wavelength pixels and 0.0040 for long-wavelength
  pixels, a factor of 250 to 1000.

The fit never recovered. While the sky was this far off, the learned noise model grew until the source pixels no longer counted:

* The fractional noise floor (`floor_raw`, one per channel) went from 0.01 to 10.8 (short wavelength) and 10.0 (long wavelength). That is a
  floor of 1000 per cent of the model signal. In the gain run the per-(detector, filter) noise scale took over instead: exp(h) = 10.8 for
  F277W and 6.75 for F444W.
* With bright pixels this down-weighted, the sky stayed far too faint on sources.
* The training chi-squared in the log is computed with that inflated model sigma, so it looked normal (0.93 to 1.03).
* The held-out score uses the pipeline ERR, so it showed the failure.

None of the four suspects in the brief is the cause:

* (a) No per-exposure factor carries the conversion.
* (b) The pixel area applied to held-out exposures is correct.
* (c) and (d) No held-out procedure could close the gap, because no per-exposure term can change the amplitude by more than a few per cent.

## 2. Evidence (numbers, and what each measures)

### 2.1 The saved control state reproduces the logged scores on CPU

The fit was rebuilt from `rg_state.pt` with the package's own classes. The local held-out exposures were loaded with the package loader and
predicted with `predict_exposure`, using their saved held-out terms. The rebuild script is in my scratch folder and is not part of the package.

| held-out exposure | chi²/N, pipeline ERR (log value) | chi²/N with the model's own sigma | model/data summed over pixels > 20 sigma | correlation |
|---|---|---|---|---|
| 02101_00004_nrcblong (F277W) | 159.93 (159.932) | 7.91 | **0.152** (9,390 px) | 0.62 |
| 02101_00009_nrcb1 (F090W) | 3.147 (3.147) | 1.51 | **0.077** (21 px) | 0.53 |
| 02103_00009_nrcb1 (F150W) | 6.717 (6.717) | 1.44 | **0.074** (64 px) | 0.50 |

* On bright long-wavelength pixels the model's sigma is a median 27.4 times the pipeline sigma.
* The frozen sky plus response predicts bright sources at **7 to 15 per cent** of the data.
* The pixel area used for these exposures is right: median 0.003973 against PIXAR_A2 0.003969 (long wavelength), and 0.000956 against
  0.000945 (short wavelength).

### 2.2 No per-exposure term can carry the missing factor, for training or held-out exposures

Every row of the saved state (123 exposures) was evaluated on a 7×7 lattice per detector, grouped by filter, detector and split:

* **Gain factor exp(tau_g u):** 0.990 to 1.030 for all training exposures, and 1.004 to 1.014 for held-out ones. Here tau_g = 0.0104.
  * The u = 0.834 reported for held-out 02101_00004 in the gain run therefore means a gain of 1.0087, not exp(0.834) = 2.3.
* **Pixel-area factor alpha / header area** (affine plus per-exposure distortion): 0.998 to 1.007 for training exposures, and 0.999 to 1.004
  for held-out ones.
* **Largest per-exposure distortion displacement:** up to 14 mas for training exposures, and 0.5 to 2.5 mas for held-out ones.

Training and held-out exposures of the same filter therefore get the same amplitude to within 3 per cent. The factor of 7 to 13 shortfall in
2.1 is a property of the frozen sky and response, so it applies to training exposures too. The model is not sound at count-rate level: the
training fit is "good" only because of the inflated noise model. The noise floor in two calibrated-input fits is 0.043 and 0.016
(`states/grid004_state.pt`, `states/fieldop_fix_200_state.pt`); in the count-rate fits it is 10.8 and 10.0.

Training chi-squared in `rg.log`, which uses the model sigma: 80 at step 0, 88 at step 200, 1.41 at step 300. The noise model absorbed the
mismatch within the first 300 steps.

### 2.3 The sky is wrong in shape as well as amplitude, so the saved fits cannot be rescued

I fitted one free amplitude k (plus an offset) to each saved held-out prediction by weighted least squares:

| held-out exposure | k | chi²/N after |
|---|---|---|
| 02101_00004_nrcblong | 12.3 | 35.7 |
| 02101_00009_nrcb1 | 7.6 | 2.11 |
| 02103_00009_nrcb1 | 10.6 | 3.45 |

Even a free scale per exposure, which the model does not have, leaves a long-wavelength chi² of 35.7. Re-running only the held-out stage on
the saved states cannot fix this. **Both 200-minute fits must be re-run.**

### 2.4 Synthetic reproduction and fix

The field is `tests/synth_small.py`'s: two instruments, three filters, 60 fit steps. For the count-rate version, every exposure's data and
sigma were multiplied by its pixel area, which is exactly what the count-rate model computes. The quantity "model/truth" is the summed model
signal over the summed true signal on pixels above 20 sigma.

| run | model/truth, warm start | after registration | end, training | end, held-out | held-out chi²/N |
|---|---|---|---|---|---|
| count rate, before fix (`fieldop_gain`) | 0.0012 | 0.0090 | 0.022 | 0.022 | **158.3** |
| count rate, after fix (`fieldop_tsky`) | 0.104 | 0.809 | 0.957 | 0.955 | **5.53** |
| calibrated input, control | 0.101 | 0.806 | 0.956 | 0.956 | **5.57** |

After the fix, the count-rate fit matches the calibrated fit. The noise floor did not run away within 60 synthetic steps even before the fix.
On SMACS the runaway took about 250 steps, so the synthetic run shows the faint start and its persistence, not the runaway itself.

## 3. Fix (in `fieldop_tsky` only)

* `fieldop/data.py`:
  * New `pixel_area_of(e)`: |det Jpix| per pixel, or |det J| of the mean header Jacobian when Jpix is absent.
  * `splat_coadd(..., per_area=False)`: with `per_area=True`, data and sigma are divided by the header pixel area before the
    inverse-variance splat, so the weights become (alpha/sigma)².
* `fieldop/cli.py` (`make_sky`) and `fieldop/train.py` (`register_prepass`, which rebuilds the warm start) pass
  `per_area = (input level == rate)`.
* The default (calibrated) path is untouched. `tests/synth_small.py --pkg . --sky-form items --seed 0 --steps 150` gives state sha256
  `b20eb534b3f7d588…`, the hash recorded in `tests/tsky_results/synth_2_tsky_defaults.json`.
* The band noise `sig_b` and the sky preconditioner are computed from the same splat, so in count-rate mode they are now in sky units too.
  The positivity penalty is in units of that band noise.

**Approximation:** the warm start uses the *header* pixel area. The model uses the learned area |det((I + M_e)(J + dD/dxi))|. These differ by
less than 0.7 per cent in the saved state.

Regression test: `tests/test_rate_warm_start.py` (2 tests, 4 to 5 s). The same synthetic field is used in calibrated and count-rate form. At
the warm start, and after the registration pre-pass, the count-rate signal of every training exposure must equal the pixel area times the
calibrated signal: the tolerance is 0.2 per cent at the warm start and 2 per cent after registration. Learned geometry is off in this test,
so that the model's area equals the header area exactly. On a copy of `fieldop_gain` both tests fail: the ratio is 0.010 (the pixel area),
and `per_area` does not exist there. With the fix both pass.

Test counts in `fieldop_tsky/tests`:

* 41 passed, with `PYTHONPATH=/home/claude/torchsky/src` for `test_torchsky_backends.py`.
* Without that path, 8 of the 10 tests in `test_torchsky_backends.py` fail because torchsky cannot be imported. That is environmental and
  unrelated to this change.

## 4. What to re-run on a GPU

Re-run both fits with this package (`fieldop_tsky/fieldop`), using the arguments of `RATE_GAIN_REPORT.md` section 6, unchanged:

    python -m fieldop $BASE --method-name fieldop_rate_grid      --out .../fieldop_rate_grid      $REF $RATE
    python -m fieldop $BASE --method-name fieldop_rate_gain_grid --out .../fieldop_rate_gain_grid $REF $RATE $GAIN

The values are those in the metrics of the failed runs:

* REF = `--budget-min 200 --batch-patches 5 --ckpt-kernel 1 --reg-prepass-steps 100 --held-steps 200 --held-reg-steps 150 --log-every 100
  --dq-ignore-bits 2`
* RATE = `--input-level rate --rate-dir /arc/projects/ots/torchsky-data/smacs0723_raw/rate --rate-wcs cal`
* GAIN = `--pixel-gain detector --pixel-gain-poly 2 --lr-gain 3e-3 --pixel-gain-start 0.1`

Use new output folders or method names so the failed runs stay for comparison. The item-sky count-rate runs, if launched, need the same
package. The calibrated twins do not need re-running: the calibrated path is bit-identical.

What to check in the new logs and states:

* `floor_raw` should map through softplus to about 0.01 to 0.05, not 10.
* exp(h) should stay near 1.
* Training chi-squared at steps 0 to 300 should start near the calibrated run's value, not at 80.
* The held-out per-exposure scores on the long-wavelength detector should be about 2, not 70 to 160.

## 5. Other problems found (reported, not fixed)

1. **The training chi-squared in the log hides failures.** `chi2/px(batch)` uses the model's own sigma, which includes the learned floor and
   noise scale. A run whose noise model has absorbed a factor of 10 mismatch looks perfect. Suggestion: also log chi-squared against the
   pipeline ERR, plus the floor and noise-scale values.
2. **The fractional noise floor has no upper bound.** It is a softplus of a free parameter. The per-(detector, filter) noise scale is bounded
   at a factor of 20. Both can make the bright pixels weightless. A bound, or a prior, of a few per cent would have turned this failure into
   a visible one.
3. **Per-detector photometric conversion at count-rate level.** PHOTMJSR differs between short-wavelength detectors B1 and B3 by 3.5 per cent
   (F090W), 1.5 per cent (F150W) and 2.7 per cent (F200W), from the local `_cal` headers. At count-rate level only the throughput detector
   field (variability axis `g_space`) can carry this, and the evidence rule pruned that axis in both failed runs (tau = 1e-6). The per-filter
   conversion is absorbed by the sky spectrum: approximation F15, "the sky has per-filter units". If `g_space` is pruned again after the
   re-run, expect a residual of up to 3.5 per cent on bright short-wavelength sources. A learned conversion per (detector, filter), shared
   across exposures, would be the smallest modelling change. I did not make it, because the warm start is the demonstrated cause.
4. In the count-rate fits, exp(h) on the short-wavelength detectors is 0.38 to 0.76, against 0.79 to 0.89 in calibrated fits. The model
   believes the count-rate ERR overestimates the noise by up to a factor of 2.6. This may be a consequence of the failure; check it after the
   re-run.
5. The largest per-exposure distortion displacement of training exposures (up to 14 mas) is 5 to 10 times that of held-out exposures. This
   is consistent with the reported D_exp of 1.88 and 2.46 against 0.62 for calibrated input. It is probably geometry absorbing the
   mis-fitted sources; check it after the re-run.
6. The brief's reading of the gain run, "u = 0.834, exp(u) = 2.3", is a misreading. The per-exposure gain is exp(tau_g u) = 1.009.

## 6. What I did not check

* **The suggested first experiment.** Predicting a training exposure with its trained terms, then as held-out, was not possible: all 12
  local exposures are held-out in the dither split, and storage on the cluster was off limits. The per-exposure term sizes in 2.2 replace it.
* **A full re-run of the held-out procedure from reset terms.** One held-out step costs about 13 s on this CPU, so 350 steps did not fit the
  4-minute rule. It is not needed for the conclusion: no reachable per-exposure term changes the amplitude by more than 3 per cent, and even
  a free amplitude leaves chi² 35.7 (2.3).
* **Whether the real-data fit converges with the fix.** That needs the GPU re-run. The synthetic fit is only 60 steps.
* **The gain run's sky amplitude on CPU.** Its saved prediction shows the same faintness: 99th percentile 0.73 against 3.74 DN/s. Its noise
  scales are inflated the same way. I did not rebuild it.
* **The multi-tile and item-sky paths.** They use the same `splat_coadd` call with `per_area`, but I did not test them on count-rate data.
