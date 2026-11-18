# The minimal preset (`--minimal`), 18 November 2026

This note describes the first pruning step on the field-operator prototype. The command-line flag `--minimal` selects the small model described in section 1 of `torchsky-end-to-end-plan.md`. It works like `--legacy-defaults`: it changes defaults only, and any option given explicitly on the command line still wins. Combining `--minimal` with `--legacy-defaults` is an error. No code was deleted and nothing was restructured. All commands are run from the package root with `PYTHONNOUSERSITE=1 python ...`. The machine had only a CPU, so all runs are small.

Acronyms: EM is expectation-maximisation (here, the damped update of the random-effect scales); FFT is fast Fourier transform; GPU is graphics processing unit; PSF is point-spread function (the "kernel" here); S/N is signal to noise; chi2 is the reduced chi-squared of the fit.

## 1. Options the preset sets

The preset sets 47 options; 17 differ from the unpreset default. Two options are new: `--g-fields {default,off}` and `--variability-evidence {0,1}` (default 1). The mechanism is the dictionary `MINIMAL_DEFAULTS` in `fieldop/cli.py`.

| Group | Option | Value |
|---|---|---|
| Sky | sky_form | items |
| | item_count | 60000 |
| | item_levels | 3 |
| | item_raster | pyramid |
| | item_patch | sheared |
| | item_buckets | coarse |
| | item_max_width | 2.56 (item_chunk left at its default) |
| Kernel | k_space, k_time, k_lambda | off, off, off |
| | k_exp | learned (per-exposure kernel code kept) |
| | tau_mode | learned |
| | filter_input | curve |
| Throughput | amplitude | structured |
| | g_fields (new) | off |
| Geometry | distortion | 0 |
| | filter_shift | off |
| | affine_prior | 0.01 |
| Detector | stripes | 1 (where declared) |
| | pattern_grid | 0 |
| | nonlin | 0 |
| | pixel_gain | off |
| | input_level, rate_units | cal, dn |
| Noise, loss | noise_scale_max | 2.0 |
| | noise_floor_max | 0.2 |
| | loss, huber_delta | huber, 3.0 |
| Gauges | sky_zero | median |
| | sky_positivity_margin | 0.0 |
| | pedestal_gauge | 1 |
| | blur_floor | 1.0 |
| | kernel_norm_radius | 2.0 arcsec |
| Rendering | render, interp, render_jitter | fft, bspline, 1 |
| Fit machinery | batch_patches | 5 |
| | inner_sky | 2 |
| | is_frac (importance sampling) | 0.5 |
| | em_damp | 0.5 |
| | nuisance_steps | exposure |
| | held_xfit, held_step_rows | 1, 1 (cross-fitted held-out prediction) |
| | epoch_per_instrument | 1 |
| Variability evidence | variability_evidence (new) | 0 |
| | n_val_groups, ab_every | 0, 0 |

Not set by the preset, because they depend on the data: `--gauge-ref-inst` (needed with several instruments), `--coarse-reg` (opt-in), `--dq-ignore-bits`.

## 2. How each switched-off item is off

"Absent" means the term is not computed in the forward model and its parameters are in no optimiser group. "Scale fixed at zero" would mean the term exists but is multiplied by zero; none of the items below needed that. The one caveat is that the tensors of absent terms are still allocated in the state dictionary (41 tensors); they receive no gradient and are never changed. A test checks this.

| Item | How it is off |
|---|---|
| Kernel variation across the detector, with time, extra with wavelength | Absent. Their variability scales are mode `off`, so the term is not built. The extra tensors (detector and epoch codes) are in no optimiser group. With k_lambda off there is one code per filter (at the pivot wavelength). |
| Throughput field across the detector and with time | Absent (`--g-fields off` turns both scales off). |
| Static and drifting distortion, per-exposure distortion code | Absent (`--distortion 0`). The per-exposure distortion code is left out of the exposure-effects optimiser group and of its prior. |
| Detector pattern (8x8) | Absent (`--pattern-grid 0`). |
| Gated non-linearity | Absent (`--nonlin 0`); its channel tensor is excluded from the optimiser. |
| Count-rate path and pixel gain map | Absent (`--pixel-gain off`, `--rate-units dn`). |
| Learned per-(detector, filter) shift | Absent (`--filter-shift off`). |
| Masked zero rule, zero-level refresh, positivity margin | Absent (median rule; refresh not triggered; margin 0). |
| Variability-evidence machinery | Absent: no validation groups are carved out of the training exposures (they become ordinary training exposures), no gradient steps on the scales, no likelihood-ratio checks. |
| Quadrature rendering, Keys interpolation | Absent (fft render, bspline interpolation). |

## 3. What could not be switched off cleanly

1. The EM updates of the per-exposure kernel and gain scales used to run only inside the validation block. To keep them (the spec keeps the per-exposure terms and their priors), `--variability-evidence 0` makes `train.fit` call the same EM update on its own schedule, without validation. This is a small change to `fit`, not a redesign. In the 150-step run the kernel scale moved from 0.3 to 0.019 and the gain scale from 0.02 to 0.0011.
2. There was no existing switch for the throughput fields, hence `--g-fields`.
3. The item-sky bucket sizes are module state set in `cli.main`; a helper `apply_sky_module_options(cfg)` now does this, so scripts and tests can reuse it.
4. Absent tensors remain allocated (41 of 83), and one of them (the per-exposure distortion code) still sits in the per-exposure dictionary and is copied by the row-stepping helpers, but is in no optimiser group.
5. The off axes remain in the scale table as `off`, and diagnostic summaries (effective variability, size report) are still computed. The at-cap report is empty when evidence is off.
6. The truth harness passes its own explicit options (amplitude unit, fixed scales, blur floor 0, 1200 items), which override the preset. "Standard settings" are therefore what it ran; see section 5.

## 4. Hashes and commands

The state hash is the sha256 over the sorted state-dictionary keys and bytes, as in `tests/synth_small.py`.

Default run (no `--minimal`), before and after the changes, identical:

`python tests/synth_small.py --pkg . --sky-form items --seed 0 --steps 150 --gauge-ref-inst 0 --kernel-norm-radius 2.0 --truth-eval 0 --out tests/oracle/results/default_items150_{before,after}.json`

`d20ae945fe90d16bad9fc4efe451c75285d037ac17fa08796face53a51c1d398`

Minimal run, twice, identical:

`python -W ignore tests/synth_small.py --pkg . --sky-form items --seed 0 --steps 150 --gauge-ref-inst 0 --minimal 1 --truth-eval 0 --out tests/oracle/results/minimal_items150_run{1,2}.json`

`436d7a87c0a2459cc2795a518e40a1d70048098e712c37adb7a42a5e02ca78f3`

That run had 21 training and 0 validation exposures, held-out chi2 2.591 and 0.959, and took 363 s. The blur dilation reached a maximum of 3.73 and ended at 2.41.

`tests/oracle/run_oracles.sh` repeats these (modes hash, truth, golden, default).

## 5. Truth problem (8 exposures, seeds 0, 1, 2, 300 steps, items)

Commands: `python -W ignore tests/truth/run_truth.py fit --form items --n-train 8 --seed S --steps 300 --minimal 1 --out ...` (response frozen at truth, the harness's standard setting) and the same with `--freeze-kernel 0 --blur-floor 1.0` (response learned).

| Run | chi2, all pixels | noise-only chi2 | ratio | cross-fitted chi2 (ratio) | sky rms | blur dilation |
|---|---|---|---|---|---|---|
| frozen s0 | 1.0026 | 0.9938 | 1.0088 | 1.0086 (1.0149) | 0.525 | 1 |
| frozen s1 | 1.0274 | 1.0167 | 1.0106 | 1.0339 (1.0169) | 0.547 | 1 |
| frozen s2 | 0.9855 | 0.9772 | 1.0085 | 0.9942 (1.0173) | 0.509 | 1 |
| learned s0 | 1.0243 | 0.9938 | 1.0306 | 1.0305 (1.0369) | 2.445 | max 963, final 36.2 |
| learned s1 | 1.1053 | 1.0167 | 1.0872 | 1.1229 (1.1045) | 1.974 | max 176, final 62.2 |
| learned s2 | 1.8921 | 0.9772 | 1.9362 | 1.9187 (1.9635) | 2.574 | max 532, final 316 |

With the response frozen, the fit reaches the noise floor within about 1 per cent, and reproduces the earlier recorded results for this problem. Strict flux ratios by S/N bin were 0.39, 0.43, 0.63, 0.96, 1.02 (averaged over the three seeds); the mean over the empty sky was +0.89 sigma.

With the response learned, the known divergence of the blur dilation occurs: it crosses 2 at steps 96, 149 and 164, reaches 10 to 1000 and never returns below about 36. Each run logs three warnings (above 2, above 10, final maximum). The sky rms grows to 2 to 2.6, strict flux ratios fall to 0.33 to 0.53, and seed 2 fails (chi2 ratio 1.94). This was recorded, not fixed.

Files: `tests/oracle/results/truth_minimal_{frozen,learned}_s{0,1,2}.json`, matching `_sky.npz`, and the two summary files.

## 6. Golden steps and tests

`tests/oracle/golden_step.py` makes one optimiser step's inputs and outputs for three configurations: one instrument, two instruments, one instrument with stripes. Each `.pt` file holds the full parameter state, the batch, the scalar loss, and the gradient of every parameter, in float64. Files: `tests/oracle/results/golden_*.pt` (1.2 to 1.5 MB each). Losses: 24.5005857828, 23.8209781831, 24.4938606177. Each has 83 parameters; 42 have gradients (the 41 absent tensors have none). `tests/test_oracle_golden.py` reloads, recomputes and agrees to 1e-6 relative (observed worst gap about 6e-8). Gradients were also checked against finite differences (1e-8 to 1e-5; the smallest are limited by round-off).

New tests: `tests/test_minimal_preset.py` (12 tests: the preset parses and matches the table; explicit options win; the legacy conflict; the default parse is unchanged; no absent tensor in any optimiser group while the kept ones are; absent tensors are untouched after a short fit; no validation exposures are held back, with a control showing the default does hold one back; EM runs without validation; explicit options bring terms back; `cli.main` runs end to end) and `tests/test_oracle_golden.py` (3 tests).

Full suite, run as `PYTHONPATH=/home/claude/torchsky/src python -m pytest tests` (the path that `tests/test_torchsky_backends.py` asks for): `118 passed in 546.36s (0:09:06)`. Without that path, 8 tests in `test_torchsky_backends.py` fail with `ModuleNotFoundError: No module named 'torchsky'` (an environment issue, not a code one): `8 failed, 109 passed, 1 skipped in 541.42s`.

## 7. Lines that are dead under the preset (rough)

Counts are executable lines (as Python reports them) inside whole functions that the preset never calls. They exclude branches inside kept functions (for example the variability branches of `prepare` in `model.py` and the validation block of `fit` in `train.py`, perhaps 100 to 150 more lines), so treat them as a lower bound of about 590 lines out of about 7,600 for the package.

| File | Dead under the preset (switched off) | What |
|---|---|---|
| `model.py` | 272 | pixel gain, filter shift, non-linearity, distortion, polynomial bases, quadrature fields |
| `train.py` | 148 | zero-level refresh, smoothing, evidence steps, ablation evidence, per-item losses |
| `sky.py` | 78 | the grid sky and the direct rasteriser |
| `render.py` | 58 | quadrature and Keys interpolation |
| `data.py` | 25 | count-rate conversion |
| `kernel.py` | 6 | time and position feature helpers |
| Total | 587 | |

Other code not used by the preset but not part of this switch list (counted separately, judgement): `tsky_backends.py` (135 lines, an alternative item rasteriser that needs an outside package) and the saved-state tolerant loaders in `cli.py` (50 lines). `coarse_reg.py`, `uncertainty.py` and `tiles.py` are not counted: they are used when asked for or are separate tools.

As a cross-check, a line-coverage run of the 150-step minimal synthetic fit never executes: `model.py` 448 of 920, `train.py` 453 of 1092, `render.py` 143 of 325, `sky.py` 85 of 493, `kernel.py` 19 of 142, `cli.py` 276 of 593, `data.py` 376 of 522. These larger numbers include real-data loading and options not in the synthetic path. Data: `tests/oracle/results/lines/`.

## 8. Real-data GPU fit with the preset

Arguments from the reference launches, plus the preset:

`--field SMACS --split dither --protocol-root /arc/projects/ots/adass2026/protocol/v1 --svo-cache /arc/projects/ots/adass2026/svo_cache --minimal --budget-min 200 --ckpt-kernel 1 --reg-prepass-steps 100 --held-steps 200 --held-reg-steps 150 --log-every 100 --dq-ignore-bits 2`

The preset supplies `--batch-patches 5`. For two instruments, add `--gauge-ref-inst 0 --coarse-reg xcorr`. An optional speed-up that does not change the model is `--item-chunk 32000000`. These were not run (no GPU, and no remote job was started).

## 9. Approximations and shortcuts

- Float64 golden steps use a context manager outside the package that sets the default dtype and patches `Tensor.float`, because a few places in the package hard-code float32. Level grids and exposure tensors are converted. The package itself is unchanged.
- Render jitter is off in the golden steps, so the step is deterministic.
- The golden comparison uses a gradient floor of 1e-9 times the largest gradient, because gauge-direction gradients are pure round-off.
- Dead-line counts are rough (whole functions only, plus a coverage cross-check).
- The truth harness's own explicit options override the preset, so "frozen" is its standard setting and "learned" needed explicit `--freeze-kernel 0 --blur-floor 1.0`.
- The baseline copy for a line-by-line difference was lost (another process deleted it from the shared scratch area), so exact changed-line counts were not measured. The default hash proof was saved beforehand.
- Golden batch: 6 patches, with a draw chosen so at least 4 exposures and every filter appear.
- The state hash under the preset includes the 41 inert tensors; a variant hash with the legacy residual buffer (`162b0e16...`) was also recorded.
