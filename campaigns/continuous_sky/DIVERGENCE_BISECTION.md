# Divergence of the learned-response fit: bisection (17 Nov 2026)

Everything here was run in `fieldop_dbgB/` (copy of the package and `tests/truth/`). The driver is `dbg/dbg_run.py`; raw per-run results are `dbg/runs/*.json`
(per-step losses, per-group gradient norms and parameter movement, dilation); the diff is `dbg/blur_dilation_cap.diff` (against `fieldop_orig/`). One core, one thread.

## Result in one paragraph

The climb is caused by the blur gauge (the common dilation of every kernel, `FieldOperator.dilation`, active in C5 because `--blur-floor` defaults to 1.0; the
frozen-response runs of the truth report had it off). While the kernel network is learned, the dilation factor grows from 1 to tens or hundreds within the first
60 to 130 steps, then to 900 to 10^5 in the bad runs. Once it passes a few hundred, every kernel position is divided by it, falls below the encoding's smoothing
radius (0.005 arcsec), the network output no longer depends on position, and the kernel network receives exactly zero gradient from then on (it never recovers; kernel parameters
stop moving at step 140 in the traced run). The rendered kernel is then not the learned kernel and the training and held-out scores sit at 2 to 3 times the noise value.
With the gauge off, or with the dilation capped at 4, the same fit converges (ratio 1.11 at 8 exposures, 1.03 at 32). The other response groups cannot be the cause in this configuration: they have no gradient (details below).

## What each number measures

* held-out: chi-squared per pixel of the held-out exposure from the package's `heldout_stage` (all-pixel fit, the number of the earlier tables; cross-fitted score not used, except where stated), against the noise-only value 0.994 (the same pixels scored with the true noiseless model). ratio = held-out / noise-only.
* batch chi2 (model noise) is the training-batch chi-squared per pixel with the model's own noise (floor and scale); (input noise) uses the input sigma. They are identical in every run here (noise terms stay at floor 1e-6, scale 1).
  A batch is 4 patches, so a single logged value scatters by about 0.1 to 0.3 in healthy runs.
* gradient norm: L2 norm over the parameters of the group, computed after backward and before clipping, of the joint step only. Groups: kernel = kernel field network, kernel codes and the code networks (a); sky; bkg = per-exposure background; reg = the 6 affine numbers per exposure (registration).
* move: L2 distance of the group's parameters from their values 10 logged steps earlier (in the trace; 25 in other runs).
* dilation: the common factor the kernels' positions are divided by (1 = gauge inactive). M(core) is the windowed core second moment of the kernel after dilation (the gauge wants it at least 833e-6 arcsec^2, one sky tent).
* lr factor: `lr_factor(progress)`, the multiplier on every base rate (warm-up 3 per cent from 0.1, then cosine to 0.05).

## 1. Reproduction and trace of C5 (grid sky, 400 steps, response learned, legacy defaults, 8 exposures)

Final held-out 2.219 (noise-only 0.994, ratio 2.233): identical to the report's C5, with the extra logging (log every 10 steps, held-out evaluated on a copy of the model every 50 steps; I restored the random state, and the final number is unchanged to the last digit).
Held-out score every 50 steps: step 50 1.076, 100 1.056, 150 2.312, 200 2.274, 250 2.244, 300 2.225, 350 2.224, end 2.219. So the held-out score was fine at step 100 and bad at step 150: the damage happens in steps 100 to 150.
Training batch score (model noise): 0.89 at step 0, 0.9 to 1.2 until step 100 (one spike of 2.51 at step 60), 1.82 at step 120, 2.91 at step 130, and 2.3 to 3.4 afterwards (the report's "falls to 0.9, then climbs" is this).

Noise floor stays at 1e-6 and noise scale exp(h) = 1.000 (frozen in C5 by the truth driver), see the table. Group (f) has no gradient except the prior and never moves (cumulative movement 0.0); throughput, distortion and variability scales have no gradient at all (no entry in the gradient record).

What moves before the climb (rows of the trace below): the dilation, nothing else.
* Dilation 1.0 (step 10), 1.2 (20), 1.5 (30), 3.7 (40), 10.9 (50), 123 (60: the batch score spikes to 2.51 and the kernel gradient to 3.5, 10 to 100 times its usual value), then oscillates 25 to 134 (steps 70 to 120), then 1044 (130), 2746 (140), and settles at 1600.7.
* The kernel gradient is exactly 0 from step 140 and the kernel parameters stop moving (movement 0.000 per 10 steps from step 180 on, cumulative movement frozen at 3.03).
  Before that the kernel moves about 0.2 to 0.5 per 10 steps. The sky, background and registration keep moving and settle as the schedule decays but cannot repair a flat kernel.
* The sky gradient, background gradient and registration gradient are small (0.001 to 0.5) and show no change before step 100. The sky movement falls smoothly with the rate schedule. They do not lead the climb.
* lr factor is 1.0 at steps 20 to 40, 0.88 at 100, 0.73 at 150: the climb starts at about 60 % of the peak rate, not at a restart or a rate jump (the schedule is monotone after step 12).

| step | lr factor | batch/model noise | batch/input noise | noise floor | noise scale exp(h) | grad norm kernel (pre-clip) | grad sky | grad bkg | grad reg | move kernel (since prev row) | move sky | move bkg | move reg | dilation | M(core) e-6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.10 | 0.89 | 0.89 | 1e-06 | 1.000 | 0.372 | 0.015 | 0.019 | 0.0023 | 0.013 | 14.0 | 0.71 | 1.56 | 1.0 | 1745 |
| 10 | 0.85 | 1.16 | 1.16 | 1e-06 | 1.000 | 0.0286 | 0.019 | 0.53 | 0.0026 | 0.512 | 101.0 | 0.66 | 0.71 | 1.0 | 970 |
| 20 | 1.00 | 1.11 | 1.11 | 1e-06 | 1.000 | 0.0433 | 0.0081 | 0.13 | 0.0038 | 0.507 | 77.6 | 0.90 | 1.06 | 1.2 | 816 |
| 30 | 0.99 | 0.90 | 0.90 | 1e-06 | 1.000 | 0.162 | 0.016 | 0.61 | 0.0049 | 0.387 | 76.3 | 0.78 | 0.81 | 1.5 | 792 |
| 40 | 0.99 | 0.94 | 0.94 | 1e-06 | 1.000 | 0.26 | 0.01 | 0.28 | 0.0081 | 0.260 | 77.9 | 0.73 | 0.76 | 3.7 | 1379 |
| 50 | 0.98 | 1.23 | 1.23 | 1e-06 | 1.000 | 0.0991 | 0.008 | 0.27 | 0.0029 | 0.387 | 74.9 | 0.60 | 0.77 | 10.9 | 4534 |
| 60 | 0.96 | 2.51 | 2.51 | 1e-06 | 1.000 | 3.47 | 0.0032 | 0.11 | 0.0017 | 0.455 | 63.4 | 0.52 | 0.55 | 123.0 | 6683 |
| 70 | 0.95 | 1.21 | 1.21 | 1e-06 | 1.000 | 0.266 | 0.0054 | 0.12 | 0.0045 | 0.492 | 54.9 | 0.51 | 0.29 | 24.7 | 4413 |
| 80 | 0.93 | 1.32 | 1.32 | 1e-06 | 1.000 | 0.478 | 0.0035 | 0.081 | 0.0032 | 0.422 | 51.7 | 0.52 | 0.46 | 57.8 | 5545 |
| 90 | 0.91 | 1.02 | 1.02 | 1e-06 | 1.000 | 0.27 | 0.012 | 0.23 | 0.0027 | 0.495 | 46.2 | 0.86 | 0.60 | 15.1 | 819 |
| 100 | 0.88 | 0.92 | 0.92 | 1e-06 | 1.000 | 0.0842 | 0.0066 | 0.09 | 0.002 | 0.348 | 79.3 | 0.46 | 0.69 | 25.3 | 1883 |
| 110 | 0.86 | 1.17 | 1.17 | 1e-06 | 1.000 | 0.146 | 0.0098 | 0.52 | 0.0028 | 0.172 | 62.4 | 0.48 | 0.44 | 49.1 | 3795 |
| 120 | 0.83 | 1.82 | 1.82 | 1e-06 | 1.000 | 0.481 | 0.0031 | 0.13 | 0.0013 | 0.200 | 51.1 | 0.52 | 0.41 | 134.3 | 5845 |
| 130 | 0.80 | 2.91 | 2.91 | 1e-06 | 1.000 | 0.124 | 0.0029 | 0.16 | 0.00034 | 0.357 | 51.0 | 0.36 | 0.29 | 1043.8 | 6635 |
| 140 | 0.77 | 2.64 | 2.64 | 1e-06 | 1.000 | 0 | 0.0034 | 0.36 | 0.0029 | 0.240 | 40.2 | 0.57 | 0.37 | 2746.2 | 6666 |
| 150 | 0.73 | 2.32 | 2.32 | 1e-06 | 1.000 | 0 | 0.0017 | 0.15 | 0.00046 | 0.086 | 32.8 | 0.42 | 0.24 | 1931.5 | 6666 |
| 160 | 0.70 | 2.50 | 2.50 | 1e-06 | 1.000 | 0 | 0.0028 | 0.3 | 0.0007 | 0.030 | 28.3 | 0.41 | 0.18 | 1707.0 | 6666 |
| 200 | 0.55 | 3.01 | 3.01 | 1e-06 | 1.000 | 0 | 0.0014 | 0.058 | 0.00014 | 0.000 | 15.9 | 0.27 | 0.29 | 1602.1 | 6666 |
| 250 | 0.36 | 3.18 | 3.18 | 1e-06 | 1.000 | 0 | 0.002 | 0.2 | 0.00042 | 0.000 | 13.8 | 0.23 | 0.09 | 1600.7 | 6666 |
| 300 | 0.20 | 2.25 | 2.25 | 1e-06 | 1.000 | 0 | 0.0019 | 0.2 | 0.00013 | 0.000 | 8.3 | 0.06 | 0.04 | 1600.7 | 6666 |
| 350 | 0.09 | 1.63 | 1.63 | 1e-06 | 1.000 | 0 | 0.0028 | 0.29 | 9e-05 | 0.000 | 3.7 | 0.05 | 0.03 | 1600.7 | 6666 |
| 390 | 0.05 | 3.01 | 3.01 | 1e-06 | 1.000 | 0 | 0.0018 | 0.11 | 0.00074 | 0.000 | 2.0 | 0.03 | 0.03 | 1600.7 | 6666 |

(kernel movement per row is over the 10 steps before it; step 0 movement is from the start.)

## 2. Bisection (one group learned, rest at the truth; then all but one)

Groups: (a) kernel field network and codes; (b) throughput; (c) distortion; (d) noise terms (floor and scale); (e) variability scales; (f) per-exposure codes (u, zD, z) and stripes.
Always learned in every run (as in the truth report): sky, per-exposure background, registration. "At the truth" for (a) means the response fitted to the true kernel by `fit_kernel_to_truth` (same 2500-step fit as the report, cached) and removed from the optimiser by `TruthOperator`; (b),(c),(e),(f) are at their start values, which are the truth in this problem
(unit amplitude, no distortion, no stripes, no variability), and (d) is at floor 1e-6, scale 1 (the truth: sigma as given).
Where the blur gauge was on is stated: it is the setting `--blur-floor 1.0` that C5 and the package default use; the frozen runs of the report had it off.

| run | description | held-out | noise-only | ratio | batch chi2 (model noise) max after step 100 | first logged step with batch >= 1.5 (3-row mean) | max dilation | end dilation |
|---|---|---|---|---|---|---|---|---|
| c5_trace | C5 reproduced (legacy defaults): a learned, blur gauge on, 8 exp | 2.219 | 0.994 | 2.233 | 3.38 | 100 | 2.75e+03 | 1.6e+03 |
| a_blur0 | a learned, blur gauge off | 1.105 | 0.994 | 1.112 | 1.21 | none | 1 | 1 |
| frozenk_blur1 | nothing in a-f learned (kernel at truth), blur gauge on | 1.008 | 0.994 | 1.014 | 1.20 | none | 1 | 1 |
| d_only | d only (noise terms), kernel at truth, blur off | 1.008 | 0.994 | 1.014 | 1.15 | none | 1 | 1 |
| bcef_only | b,c,e,f learned, kernel at truth, blur off | 1.008 | 0.994 | 1.014 | 1.20 | none | 1 | 1 |
| all_blur1 | all of a-f learned, blur on | 2.231 | 0.994 | 2.245 | 2.19 | 175 | 9.85e+04 | 9.85e+04 |
| all_but_a_blur1 | all but a learned, blur on | 1.008 | 0.994 | 1.014 | 1.15 | none | 1 | 1 |
| all_but_d_blur1 | all but d learned, blur on | 2.219 | 0.994 | 2.233 | 4.02 | 75 | 1.93e+03 | 1.6e+03 |
| a_lr3 | a learned, blur on, kernel rate /3 | 1.125 | 0.994 | 1.132 | 1.25 | none | 2.02 | 1.61 |
| a_lr10 | a learned, blur on, kernel rate /10 | 1.080 | 0.994 | 1.087 | 1.27 | none | 1.55 | 1.55 |
| a_const | a learned, blur on, constant rate 0.1 x base | 1.837 | 0.994 | 1.848 | 1.98 | 250 | 111 | 67.4 |
| a_clip5 | a learned, blur on, clip 5 (was 10) | 2.202 | 0.994 | 2.215 | 3.68 | 75 | 890 | 671 |
| n16 | a learned, blur on, 16 exposures | 1.010 | 0.994 | 1.016 | 1.59 | none | 38.7 | 30.2 |
| n32 | a learned, blur on, 32 exposures | 2.259 | 0.994 | 2.273 | 3.57 | 125 | 890 | 890 |
| a_today | a learned, blur on, TODAY's defaults (shortcuts, see text) | 1.098 | 0.994 | 1.105 | 2.35 | 50 | 243 | 1 |
| a_cap4 | a learned, blur on, dilation capped at 4 (diagnostic edit) | 1.109 | 0.994 | 1.115 | 1.23 | none | 6.5 | 4.78 |
| n32_cap4 | as previous, 32 exposures | 1.023 | 0.994 | 1.029 | 1.24 | none | 5.18 | 5.17 |

Reading it:
* Learning (a) with the gauge on diverges (2.233); with the gauge off it does not (1.112); the kernel at the truth with the gauge on does not (1.014, dilation stays 1.0: the truth kernel is above the floor).
* Learning (d) alone: 1.014. Floor and scale move by at most 0.03 in log scale and not at all in the floor over 400 steps. Not a cause (nor involved when learned with the others: all-but-d reproduces C5 exactly, 2.233).
* (b), (c), (e) have no parameters that receive any gradient in this configuration (amplitude `unit`, distortion 0, no validation exposures so the variability scales are never fitted); (f) receives only its prior gradient and does not move. The run `bcef_only` (all four learned, kernel at the truth) is identical to the frozen run (1.014 and the same batch scores). So "one at a time" for these four is the same run as `bcef_only` and I did not repeat it four times.
  This says nothing about real data, where throughput, distortion and variability are active; it only says that in this problem they cannot be the cause of this climb.
* All of a to f learned (gauge on): 2.245. All but a: 1.014. All but d: 2.233. So (a) is necessary, and with the gauge on it is sufficient.

## 3. Rate, schedule and clipping (group a learned, gauge on, 8 exposures)

| change | ratio | climb | max dilation | note |
|---|---|---|---|---|
| none (C5) | 2.233 | yes, about step 120 | 2.7e3 | |
| kernel rate / 3 (1e-3 to 3.3e-4; other groups unchanged) | 1.132 | no | 2.0 | dilation 1.4 to 2.0, drifts up |
| kernel rate / 10 | 1.087 | no | 1.55 | dilation still rises slowly (1.55 at the end) |
| constant rate 0.1 x base for EVERY group (replaces warm-up and cosine; sky and background also at 0.1 x) | 1.848 | yes, later (step 250) | 111 | delays the runaway to 250; the sky is also slowed, so this run is not clean |
| clip threshold 5 (was 10) | 2.215 | yes, earlier (step 75) | 890 | the kernel gradient is mostly below 0.5, so clipping rarely acts except at the spikes |

A lower kernel rate removes the climb in 400 steps, but the dilation still rises with it (1.0 to 1.5 at /10); I take that as a delay, not a cure. I did not run longer fits to check that.
The constant-rate run shows the same mechanism later: dilation 1.6 at step 200, 52 at step 300, 111 at step 350.

## 4. More data (group a learned, gauge on, same 400 steps)

| training exposures | ratio | max dilation | climb |
|---|---|---|---|
| 8 | 2.233 | 2.7e3 | yes |
| 16 | 1.016 | 38.7 (ends 30.2) | no (batch score reaches 1.59 once at step 225) |
| 32 | 2.273 | 890 | yes (from step 125) |

More data does not remove it: 16 exposures escaped by staying below the point of no return (dilation 30 to 40), 32 exposures did not. One realisation each, so I cannot say how often 16 escapes.
The dilation runaway is a feature of the kernel's own dynamics against the gauge, not of the amount of data.

## 5. Today's defaults

`a_today`: group a learned, gauge on, defaults of 18 Nov 2026 (masked sky-zero, noise bounds, exposure-wise nuisance steps, cross-fitted held-out, epoch rule), 8 exposures. Result ratio 1.105 (held-out 1.098), no persistent climb:
the batch score spiked to 2.55 at step 50 and 2.35 at step 100 while the dilation went to 44 and 243, then the dilation fell back to 1.0 at step 125 and stayed there. So under today's defaults the same runaway occurs but happened to recover (dilation below about 250 is recoverable; the legacy C5 went past 1000).
It is a single run: I cannot say the new defaults are safer. Shortcuts for this run: the masked sky-zero was re-evaluated only at progress 0 (the default re-evaluates at 0, 0.1, 0.25, 0.5, 0.75; five refreshes did not fit in 4 minutes) and the held-out score is the all-pixel fit (`held_xfit` 0 for scoring), not cross-fitted. The kernel noise bounds were irrelevant (noise terms frozen).

## 6. Mechanism and the diagnostic edit

`dilation()` sets s1 = sqrt(max(1, floor / M(1))) assuming the core second moment M grows as the square of the dilation, then one more pass. M is measured on a 0.025 arcsec grid inside a Gaussian window of width 0.058 arcsec; when the learned kernel has most of its weight on the central grid point, M(1) falls towards 0 (about 3e-10 if the dilation 1600 came from the first pass alone: inferred as floor / 1600^2, not measured), s1 explodes, and M(s1) is not floor (the logged M is 6666e-6 at dilation 1600, against the 833e-6 wanted: the fixed point has not been reached).
The rendered kernel is unchanged by the gauge in the ideal case (it is a pure reparametrisation, a free direction for the data), so nothing in the data pulls the kernel network back from a sharp core: it drifts along that direction under Adam and the dilation compensates until it saturates the encoding.
That is my reading; I did not isolate it further (for example by measuring M(1) at each step or the shape of the learned kernel; the recorded quantities are dilation and M at the current dilation).

Diagnostic edit (one line, `fieldop/model.py`, `dbg/blur_dilation_cap.diff`): cap s1 at 4.0 before the second pass. The value 4 is arbitrary and the cap is a mitigation, not the repair of a bug: I found no sign, detach or restart error. Results: 8 exposures 1.115 (was 2.233), 32 exposures 1.029 (was 2.273); dilation stays 4.8 to 6.5 permanently (the gauge is active all the time, which means the learned kernel core sits at about a fifth of the tent width in the gauge direction, so it is the gauge that is constraining the fit, and the effect of that on the response and sky is not measured here).
Not a delivered fix; the package's own copy was not touched.

## What was held fixed

Problem, population, noise seed 0 (noise and fit seed 0 in every run), 400 steps, 4 patches per step, grid sky, kernel network of the package, learning rates as in `base_config`, noise terms frozen except where group d is learned, legacy defaults except `a_today`.
Same noise realisation for all runs; one seed per configuration.

## Shortcuts and what I did not run

* One seed per configuration. A run that escapes (N = 16, `a_today`) could be luck; the escape rate is unknown.
* The held-out score of the traced run was evaluated every 50 steps with a deep copy of the model and all-pixel nuisance fit (`held_xfit` 0), not cross-fitted. All final held-out values are all-pixel fits (cross-fitted scores were not computed).
* Groups (b), (c), (e), (f) were not run singly: they have no gradient here (shown by the gradient record and by `bcef_only`). Throughput, distortion and variability scales were not tested under settings where they are active (free amplitude, distortion on, validation exposures). Not run.
* Rate runs: the /3 and /10 scale only the main operator group (kernel network, codes, and the unused throughput and distortion networks); noise-term rate and the sky and background rates unchanged. The constant-rate run changes every group.
* Constant rate: only 0.1 x base was tried. A smaller one, longer runs (to see if the /10 run drifts into the runaway later) and the combination of rate with cap were not run.
* Items sky not run (grid only; C6 not reproduced).
* No longer fits; no other seeds; I did not try `--blur-floor` values between 0 and 1, or other caps.
* The kernel warm start, as in C5, is the package's diffraction start (150 steps); the frozen-kernel start is the same 2500-step fit as the report.
