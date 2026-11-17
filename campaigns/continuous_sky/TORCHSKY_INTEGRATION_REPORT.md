# Four torchsky pieces behind switches in the field-operator prototype

16 Nov 2026. Folder `/home/claude/work/fieldop_tsky/` (a copy of `fieldop_gain/` without `dl/`; the sample files are read from `../fieldop_gain/dl`, and the
11 other calibrated sample files from `../score_paired/tests/data/cal`). `fieldop_gain/` and `/home/claude/torchsky/` were not edited.
Run with `PYTHONPATH=/home/claude/torchsky/src PYTHONNOUSERSITE=1`. One thread throughout. Every default is the prototype's own code path: the
default synthetic fit gives the same state hash as `fieldop_gain` (section "Synthetic fit"), and the 40-step grid fit still gives the hash `ff0f1183004ab6a1…`
recorded in `RATE_GAIN_REPORT.md`.

| piece | switch (default first) | adopted | key number |
|---|---|---|---|
| A. items through `GaussianItemScene` | `--item-impl {prototype,torchsky}` | yes | raster agrees to 2.2e-7 of the image maximum; gradients to 1e-5 |
| B. count-rate positions from the telescope frame | `--rate-wcs {cal,rate,telescope}` | switch built; NOT usable by the 5 mas rule | rms 3.0 to 59.8 mas against the calibrated solution |
| C. exposures read by torchsky | `--loader {astropy,torchfits}` | yes | 0 differences on usable pixels and in 38 comparisons of headers |
| D. render through `NestedKernelPSF` | `--render-impl {prototype,torchsky}` | no, not wired in | 1.8e-4 (stars and galaxies), 1.2e-3 (white noise) of the image maximum |

New and changed files: `fieldop/tsky_backends.py` (new, all torchsky code; imported lazily), `fieldop/sky.py` (`impl` argument of `ItemSky`, 2 branches),
`fieldop/data.py` (reader dispatcher, `--loader`, `--rate-wcs telescope`), `fieldop/cli.py` (4 switches, passed through), `tests/test_torchsky_backends.py` (new),
`tests/check_rate_wcs_telescope.py`, `tests/check_loader_equivalence.py`, `tests/check_render_nested.py`, `tests/bench_items_torchsky.py` (new measurement scripts),
`tests/synth_small.py` (one added optional flag `--item-impl`, nothing else), and the measurement results in `tests/tsky_results/`.
Running the existing tests rewrote the copies of their result files in `tests/items_results/` inside this folder only.

## A. Item sky through `GaussianItemScene` (`--item-impl torchsky`): adopted

* What it does. `ItemSky.raster` hands the item parameters to the scene: sky position (arcsec, west and north) goes to longitude and latitude through a
  tangent-plane header whose pixel grid IS the prototype's coefficient grid (0.04 arcsec pixels, centre on the box centre); widths enter as `exp(min(lw, log max_width))` and the scene
  adds the prototype's width floor in quadrature (`min_sigma_arcsec = floor`), so the floor, the cap and the positivity (`exp` of the log coefficient) are the prototype's; the angle is
  converted (scene angle = prototype angle minus 90 degrees, east of north against measured from the west axis). The scene returns flux per pixel, the prototype's grid is a density
  per square arcsec, so the output is divided by the pixel area. `--item-raster direct|pyramid` maps to the scene's `mode`. Gradients flow through to the prototype's parameters
  (the scene is called through `torch.func.functional_call` so that its parameters can be non-leaf tensors).
* Acceptance (float64, 400 items, 96 x 96 grid at 0.04 arcsec, three coefficient planes; widths log-uniform from 0.3 to 40 pixels per axis so both narrow and wide items, random
  angles over the full circle, items up to 5 per cent beyond the edge of the grid; error = largest absolute difference over the image maximum, 744):
  | | raster | gradient, position | width | angle | coefficient |
  |---|---|---|---|---|---|
  | direct | 2.2e-7 | 1.1e-6 | 7.8e-7 | 5.0e-6 | 5.5e-7 |
  | pyramid | 2.2e-7 | 1.2e-6 | 9.2e-7 | 7.7e-6 | 4.7e-7 |
  The gradient columns are the relative difference of the two gradient vectors (norm of the difference over the norm of the prototype's) for the loss `sum(random weights x raster)`.
  Limits were 1e-5 and 1e-4. `test_A_*` in the new test file repeats this on 160 items, 72 x 72 grid.
* Cost, one forward plus backward, 60,000 items, 601 x 601 grid at 0.04 arcsec (the grid of the SMACS runs), 12 coefficient planes, float32, one thread, shared machine
  (`tests/bench_items_torchsky.py`, `tests/tsky_results/bench_items.jsonl`; items from the prototype's lattice seed with log-normal width scatter, median width 1.6 pixels, 99th percentile 12 pixels;
  chunk of 4 million elements; best of two after a warm-up; peak = the process's resident-set maximum, of which about 530 MiB is the interpreter and imports):
  | rasteriser | implementation | forward | forward + backward | peak memory |
  |---|---|---|---|---|
  | direct | prototype | 4.1 s | 12.4 s | 879 MiB |
  | direct | torchsky | 4.7 s | 11.6 s | 878 MiB |
  | pyramid, square patch | prototype | 3.3 s | 9.8 s | 895 MiB |
  | pyramid, square patch | torchsky | 3.3 s | 8.4 s | 1009 MiB |
  Timings on a shared two-core machine are indicative to about 10 per cent. Both implementations are within that of each other; the scene's pyramid holds 13 per cent more memory (dense level grids in the autograd graph).
* Approximations and differences from the prototype's raster.
  * The cell-integral stand-in (1/12 pixel squared added to the covariance) and the exact-flux normalisation are the same in both.
  * Patch sizes differ: the scene uses a patch of 4 larger principal widths with every integer half-size, the prototype uses its ladder of half-sizes (2, 3, 4, 5, 6, 8, 10, ...) and, in direct mode, caps the patch at 128 nodes
    (wider items are silently truncated); the scene has no cap. This is why the two rasters are not identical (2e-7), and an item wider than 32 pixels is rendered more completely by the scene.
  * Not available with the scene: `--item-patch sheared` (the run stops with a clear message) and `--item-buckets coarse` (ignored; the scene has its own buckets). `--item-chunk` is honoured (read when the scene is first built).
  * Positions go through longitude and latitude in float64 (a cast of the float32 positions); a float32 longitude would resolve only 0.05 arcsec.
  * The gradient through the scene treats the patch window and the pyramid level as fixed (as the prototype does).
* Not finished: no run on the real SMACS field (it needs a GPU job); the benchmark uses seeded, not fitted, items.

## B. Count-rate positions from the telescope frame (`--rate-wcs telescope`): built, not usable by the 5 mas rule

* What it does. `--input-level rate --rate-wcs telescope` builds each count-rate file's pixel-to-sky map from the file's own header (pointing `RA_REF`, `DEC_REF`, roll `ROLL_REF`, reference point `V2_REF`, `V3_REF`,
  `VPARITY`, `V3I_YANG`) and the instrument aperture file's polynomial (read with `pysiaf`, the aperture named by `APERNAME` in the primary header; polynomial degree 5, coefficients of the installed `pysiaf` 0.29.0, PRD `PRDOPSSOC-075`), through
  `TelescopeFrameDetectorToSky`. The matching calibrated file is no longer needed. The pixel solid angle (`PIXAR_A2`, absent from count-rate files, kept only as a note in the exposure record) comes from the polynomial's Jacobian at the reference pixel
  and agrees with the calibrated file's value to 1.5e-4 or better.
* Acceptance (`tests/check_rate_wcs_telescope.py`, `tests/tsky_results/rate_wcs_telescope_vs_cal.json`): 65 x 65 lattice of pixels from 0 to 2047 on each detector, great-circle separation from the calibrated file's own header
  solution (the SIP world-coordinate system the prototype uses today), all 12 exposures, milliarcseconds; rule: usable if the rms is below 5 mas.
  | detector | exposures | filters | maximum | rms (pooled, per exposure) |
  |---|---|---|---|---|
  | NRCB1 | 3 | F090W, F150W, F200W | 60.2 | 34.4 (4.1, 3.0, 59.4) |
  | NRCB3 | 3 | F090W, F150W, F200W | 61.5 | 34.7 (5.2, 3.0, 59.8) |
  | NRCBLONG | 6 | F277W, F356W, F444W (two each) | 14.9 | 11.8 (13.5, 10.7, 10.8) |
  The 0.08 of a long-wavelength pixel is 5 mas. **The rule is met by 3 of the 12 exposures** (F090W and F150W on NRCB1, F150W on NRCB3), is missed marginally by F090W on NRCB3 (5.2), and is missed by the long-wavelength detector (10.7 to 13.5) and by every F200W exposure (59 to 60).
  Verdict: not usable as a replacement for the calibrated solution. It is 7 to 30 times closer than the count-rate file's own plain header (about 90 mas rms on NRCB1, 340 on the long-wavelength detector) except for F200W (1.5 times).
  Removing one constant offset per exposure (diagnostic only, not applied) leaves 0.7 to 2.7 mas rms and 3.6 to 7.8 mas maximum, so the miss is a per-filter constant shift, not a distortion error. The shift is the pipeline's filter-dependent offset, which the
  aperture polynomial does not contain (my reading, not checked against the reference files, which are not on this machine; torchsky's own notes say the same).
* What is neglected or assumed.
  * Velocity aberration (header `VA_SCALE` = 0.999994 here, an effect of about 0.4 mas at the detector corners by my estimate; not implemented). The calibrated solution includes it.
  * The pipeline's filter-dependent offset and distortion, and any time dependence.
  * Assumed: `APERNAME` of the primary header is the aperture's name in the instrument file (true for the 12 files; the long-wavelength detector is `NRCB5_FULL` there, which is how the file spells it); the reference pixel of the polynomial
    (`XSciRef`, `YSciRef`, 1024.5) comes from `pysiaf`, not from the header; the planar ideal-to-telescope step of the instrument file convention (torchsky note: differs from the spherical option by far less than 1 mas over a detector).
  * The calibrated file's own solution is itself the pipeline's SIP approximation of its full model, so the differences above are between two approximations.
* Not finished: no fit on real data with these positions (GPU job); no learnable filter offset.

## C. Exposure files read by torchsky (`--loader torchfits`): adopted

* What it does. The prototype's reader dictionary is built from `field_observation_from_exposure_file` (whole detector, `float64` planes so that `1/err^2` cannot overflow), with the data-quality policy "reject every bit except `--dq-ignore-bits`".
  The reader also supplies the header, `AREA` (read through `torchfits` because the observation record has no slot for it), `DETECTOR` and `INSTRUME` (primary header read again).
* Acceptance (`tests/check_loader_equivalence.py`, `tests/tsky_results/loader_equivalence.json`; 38 comparisons: count-rate files with the calibrated header, with the count-rate header, `--dq-ignore-bits` 0 and 2):
  * Science and error at usable pixels: bit equal (largest difference 0.0); 12 count-rate files and the one complete calibrated file.
  * Usable mask: identical to the prototype's own validity expression (finite science, finite positive error, data-quality bits) in every comparison; data-quality plane read by torchfits identical to astropy's.
  * Header: every key, value and type of the science header identical (0 differences in 38 comparisons plus 11 calibrated headers); `PIXAR_A2`, `DETECTOR`, `INSTRUME` identical; `AREA` identical; pixel-to-sky positions of the two world-coordinate systems bit equal.
  * Only 1 of the 12 local calibrated files is complete (`fieldop_gain/dl/cal`); the other 11 (`score_paired/tests/data/cal`) are cut at 20 MB, so on them only the header and the science plane (complete) could be compared: identical.
  * The loader's whole cut dictionary (`load_exposure_file`, 2 arcsec half-size box, one file) is identical at every valid pixel, with the same valid pixels, positions and background (in the new tests).
* Differences by construction (not errors): the reader writes 0 into science at pixels it calls unusable, where astropy keeps the file's value (including NaN); it returns the usable mask instead of the data-quality plane, which the prototype only ever uses to form that mask; the downstream code multiplies by the
  mask. Reading is slower: 0.28 s against 0.05 s for one count-rate file with the calibrated header (warm cache).
* Can the torchfits import fail? Yes: torchfits is a compiled extension, so a missing or mismatched wheel or shared library raises at import. `--loader torchfits` checks once, prints
  "`--loader torchfits: torchfits or ... could not be imported (<error>); FALLING BACK to the astropy reader`" and uses the astropy path. This is tested by simulating the failure (the real import works here). `--rate-wcs telescope` also reads headers with astropy if torchfits is missing.
* Not finished: the reader's own sky-box cut is not used (the prototype cuts after its own position test); no HST file tried.

## D. Rendering through `NestedKernelPSF` (`--render-impl torchsky`): not adopted, not wired in

* Measurement (`tests/check_render_nested.py`, `tests/tsky_results/render_nested_vs_prototype.json`, float64). Same sky (tent interpolant of a grid at 0.1 arcsec), same kernel values (a Gaussian core of sigma 1.5 pixels plus a power-law wing, 8 nodes per side per level,
  extent 30 pixels: 3 nesting levels in both), same geometry (unrotated, render pitch = sky pitch = pixel pitch, output positions at the nodes plus a sub-pixel offset of 0.37 and 0.21 pixel, 36 x 36 pixels); the
  prototype's `render_fft` against the nested operator on the node-sampled sky followed by the prototype's own cubic B-spline read at the output positions. Differences over the image maximum:
  | sky | nested against prototype | prototype against the lattice reference | nested against the lattice reference |
  |---|---|---|---|
  | smooth galaxies | 6.1e-5 | 1.3e-5 | 6.1e-5 |
  | stars and galaxies (30 stars of 0.7 to 1.2 pixels) | 1.8e-4 | 2.4e-4 | 3.9e-4 |
  | white noise | 1.2e-3 | 1.3e-3 | 1.2e-3 |
  The lattice reference is the same node-sampled sky convolved with the dense tapered kernel. So the two renders agree to better than 1e-3 on resolved skies in this idealised case, and miss 1e-3 only on white noise, where both are about 1.2e-3 from the reference: the two schemes use different pooling for the coarse levels
  (the prototype averages the coefficients over blocks, torchsky uses the transpose of a hat-function interpolant) and cannot represent pixel-scale power in the wings, and the difference of two such approximations is as large as either.
  (Both differ by 4 to 5 per cent from a dense evaluation of the tent sky on a finer lattice for the sharp skies: the tent between nodes is not sampled by either; not a property of the nested operator.)
* Why it is not wired in, whatever the table says. `NestedKernelPSF` is a convolution of one image with a kernel on the same unit-pitch grid. The model's render path is different in these ways, none tested to agree:
  1. it reads the correlated level images at arbitrary exposure positions (rotated, scaled, distorted detector pixels, up to 6 times finer than the pixel for the kernel core) with a cubic B-spline read; the nested operator has no such read, so it would need an image at the render pitch (several thousand pixels per side per patch);
  2. the kernel comes from the network for each patch, rotated into the sky frame, with a random sub-pitch lattice phase in training, a common dilation, the optional unit integral inside a radius (flux convention) and the option of a free amplitude; the nested operator normalises every node kernel to unit sum over the whole extent;
  3. the prototype blends region kernels by output pixel position; the nested operator blends by source position (torchsky's note: the prototype's way does not conserve a point source's flux);
  4. coarse levels of the prototype follow the sky grid; the nested ones follow detector coordinates.
  Wiring it would be a rewrite of `render_fft_batch` and the kernel evaluation, with a new acceptance test, not an adapter. I did not start it.
* The switch exists so the command line is complete; `--render-impl torchsky` stops at start-up with a message that gives these numbers.

## Synthetic fit (`tests/synth_small.py --sky-form items --item-count 1500 --steps 150 --truth-eval 0`, 1,500 items, direct raster, seed 0, one thread, about 170 s each)

| run | held-out chi-squared per pixel (all pixels) | at signal-to-noise above 5 | state hash |
|---|---|---|---|
| `fieldop_gain`, defaults | 3.33148374 | 24.1092 | `b20eb534b3f7…` |
| `fieldop_tsky`, defaults | 3.33148374 | 24.1092 | `b20eb534b3f7…` (identical) |
| `fieldop_tsky`, `--item-impl torchsky` (the only adopted switch the synthetic field can exercise) | 3.33121305 | 24.1065 | `ec2fb16cda44…` |

* Defaults identical to every digit and in the final-state hash. With the scene the held-out score differs by 0.008 per cent (limit 1 per cent).
* The fit is not converged (150 steps; noise-only floor is 0.989 on the same pixels), as in the earlier synthetic runs of this package. Only A can act on the synthetic field: B and C read files, and D is not wired in.
* The 40-step grid fit gives the recorded hash `ff0f1183004ab6a1…` (`tests/tsky_results/synth_grid40_tsky_defaults.json`).

## Tests

* Existing tests, unchanged, in `fieldop_tsky`: `test_items_unit.py` 3 passed (87 s), `test_items_tiles.py` 1 passed, `test_pixel_gain.py` 7 passed, `test_items_pyramid.py` 14 passed (7 square, 7 sheared; run in two halves because the whole file takes over 2 minutes). Total 25 of 25.
* New `tests/test_torchsky_backends.py`: 10 passed in 7 s (defaults; A direct and pyramid; A floor and the refusal of the sheared patch; B against the calibrated solution and the data path; C arrays, headers, cuts and the fall-back; D refusal and the 1e-3 check on the resolved sky). B and C tests are skipped if the sample files are not at `../fieldop_gain/dl`.

## For the torchsky maintainer

1. `GaussianItemScene` copies its inputs into `nn.Parameter`s, so a caller that computes the item parameters from its own parameters (as a fitting harness does) loses the graph. I wrapped the scene in a module and used `torch.func.functional_call` to substitute tensors.
   A functional entry point taking tensors (and the public pixel-space rasteriser `_rasterize_gaussians`, which is private now) would remove the workaround.
2. The scene's input is longitude and latitude and its output is flux per pixel. A harness that already works in a tangent plane has to round-trip through a header and rescale by the pixel area; a pixel-grid entry point would also avoid the float32 longitude limit (0.05 arcsec).
3. In direct mode there is no cap on the patch (a width of 300 pixels builds a 2401 x 2401 patch); the prototype capped at 128 nodes. The cap changes results for items wider than 32 pixels, and memory is unbounded in the item width; a documented optional cap would help.
4. `TelescopeFrameDetectorToSky` has `pixel_to_world(x, y)` but not the `WCS` method `pixel_to_world_differentiable(pixels (n, 2), origin)`; a small adapter was needed. A constructor from a header plus a `pysiaf` aperture name would save every caller the coefficient and reference-pixel plumbing
   (the flat 21-term coefficient arrays of `pysiaf.get_polynomial_coefficients()` were accepted directly, which is good).
5. The telescope-frame solution is 3 to 13 mas from the pipeline's solution for four filters and 60 mas for F200W, almost entirely a constant per-filter shift. Until a filter offset is available (their notes' open question 3) the transform cannot meet a 5 mas rule on most of the sample.
6. `field_observation_from_exposure_file` does not return the data-quality plane (only the mask), does not expose `DETECTOR` or `INSTRUME` (primary header), and zero-fills science at unusable pixels; a caller that wants the raw planes (as this prototype's downstream code could) has to read them again. Reading `AREA` is by hand.
   In `float32` the usable mask also drops pixels whose `1/err^2` overflows; I asked for `float64` to avoid it (no such pixels in the sample).
7. `NestedKernelPSF` has no read at arbitrary output positions, no rotation of the kernel with the exposure, and always normalises to unit sum; each is a reasonable choice for its design but together they block its use as a drop-in for the prototype's render. Its mutual difference with the prototype's own coarse-level scheme on white noise (1.2e-3) is of the size of either's error against a dense reference.
8. Observed while testing, not fixed: only 1 of the 12 sample calibrated files is complete; the others are cut at 20 MB (`score_paired/tests/data/cal`), which makes `astropy` fail on their `ERR` plane. This is a property of the sample, not of torchsky.

## What was not done

* No real-data fit with any switch on (those are GPU jobs); the rate-position and loader switches were checked on files only.
* D was measured, not wired. B was built and measured, not adopted by the rule.
* No timing of A on a GPU, no float32 accuracy table for A beyond the synthetic fit (which ran in float32 and agrees to 0.008 per cent).

## E. Learned shift per detector and filter (`--filter-shift {off,learned}`, default `off`)

16 Nov 2026. Built and unit-tested; the synthetic target was missed (numbers below); no real-data fit (needs a GPU).

* What was built. One learned 2-vector per (detector, filter) pair, in detector pixels (x = column, y = row, zero based), shared by all exposures of the pair (training, validation and held out).
  Pair names are `DETECTOR/FILTER` (for example `NRCB1/F200W`; the filter name is the text after the last dot of the bandpass name). Switches: `--filter-shift`, `--filter-shift-init FILE` (JSON `{"NRCB1/F200W": [dx, dy]}`, unknown names stop the run),
  `--lr-filter-shift` (0.03 pixel per Adam step, same warm-up and cosine factor as the other parameters). The shift has its own optimiser, is stepped from step 0 only for the pairs present in the batch, is not in the operator's group and not in the held-out fit.
  Files: `fieldop/model.py` (`attach_filter_shift`, `filter_shift_sky`, `project_filter_shift`, one line in `positions`, one in `param_groups`), `fieldop/train.py` (optimiser, step, projection, prepass), `fieldop/cli.py` (3 switches, attach, log line, `filter_shift` in `metrics.json`),
  `fieldop/synth.py` (one optional argument `pair_shift_px`, default `None` leaves the field unchanged).
* Default path unchanged. With `off` the model has no new parameter or buffer (state dictionary identical) and every added line is behind `if model.fs_on`. The default synthetic fit (`synth_small.py --sky-form items --item-count 1500 --steps 150 --truth-eval 0`) gives state hash `b20eb534b3f7d588…` and held-out 3.33148374,
  identical to `synth_2_tsky_defaults.json` (`tests/tsky_results/synth_4_filter_shift_defaults.json`).
* Sign convention. The position of pixel (col, row) is W(col + sx, row + sy), W the header (or telescope-frame) pixel-to-sky map: positive (sx, sy) reads the sky where the header puts pixel (col + sx, row + sy). Implemented to first order as a sky displacement J_e s added to the header position
  before the distortion and the per-exposure affine, J_e = the exposure's mean header Jacobian d(X, Y)/d(col, row) (same frame as X, Y; `extra["J"]`). Tested: positions change by exactly J_e s. The files check below uses the same convention (W_telescope(p + s) against the pipeline's solution), so its numbers are valid starting values.
* Free origin and the per-exposure affine. The affine's translation (t_e, 0.01 arcsec per raw unit) is free (the prior covers only its four matrix terms) and the old gauge zeroed the mean of t_e over the training exposures. A shared shift is therefore redundant with the pair mean of t_e. With the switch on, `project_shift` is replaced, after every step (and after the registration pre-pass), by an exact two-part projection
  on the training exposures of the shift component: (1) the mean over exposures of the whole learned displacement d_e = J_e s_pair(e) + t_e is removed (t_e -= mean d): the free origin, no penalty; (2) per pair, the translations are made J-orthogonal, sum_e J_e^T t_e = 0, by moving ds = (sum J_e^T J_e)^-1 sum J_e^T t_e into s and taking J_e ds out of t_e.
  Step 2 does not change d_e: it is a re-expression, so nothing learned is thrown away and the model function is unchanged by it. When the exposures of a pair share one Jacobian (one roll) it is exactly "the mean of t_e is zero within the pair, and the shifts, converted to the sky tangent plane with each pair's own Jacobian, have zero weighted mean" (weights = training exposures per pair, as the old gauge).
  With different rolls it is the least-squares split (translations are the smallest residuals), which I chose because a pixel shift then maps to a different sky displacement in each exposure, and a plain zero mean would leave the split undetermined. Pairs differ in pixel scale and orientation only through J. The affine matrix terms are untouched. Validation and held-out exposures keep their own free translation, fitted around the pair's shift.
* Held-out exposures use the fitted shift of their pair, fixed (tested: positions equal the pair's J_e s, and `fit_heldout` leaves the shift bit-identical). A pair with no training exposure keeps its starting value.
* A. Files only (`tests/check_filter_shift_files.py`, `tests/tsky_results/filter_shift_files.json`). Same 65 x 65 lattice as section B; one shift per pair by Gauss-Newton least squares of W_telescope(p + s) against the calibrated solution of all the pair's exposures; rms and maximum are of the separation in mas after the shift (before: rms 3.0 to 59.8 mas).
  | pair | exposures | shift (x, y) pixels | rms mas | max mas |
  |---|---|---|---|---|
  | NRCB1/F090W | 1 | -0.068, -0.085 | 2.32 | 7.08 |
  | NRCB1/F150W | 1 | -0.076, -0.054 | 0.78 | 4.11 |
  | NRCB1/F200W | 1 | 1.045, 1.620 | 0.60 | 3.71 |
  | NRCB3/F090W | 1 | -0.103, -0.108 | 2.41 | 7.82 |
  | NRCB3/F150W | 1 | -0.071, -0.054 | 1.17 | 3.99 |
  | NRCB3/F200W | 1 | 1.054, 1.621 | 0.99 | 3.45 |
  | NRCBLONG/F277W | 2 | 0.098, 0.189 | 1.11 | 4.47 |
  | NRCBLONG/F356W | 2 | 0.043, 0.163 | 0.84 | 4.65 |
  | NRCBLONG/F444W | 2 | 0.024, -0.165 | 2.57 | 4.66 |
  For the three long-wavelength pairs the two exposures' own best shifts differ by 1.6e-7, 6.5e-8 and 1.1e-7 pixel (numerical level), so one shared shift per pair is enough on these files: the miss in section B is one constant per filter in detector pixels. Limits of this: the six short-wavelength pairs have one exposure each (nothing to compare), all 12 files are from one programme and visit family,
  and the pipeline's solution is itself an approximation. The F200W shifts of the two short-wavelength detectors agree to 0.01 pixel. All nine pairs meet the 5 mas rms rule after the shift; the largest maximum is 7.8 mas. Starting values for a real run: `tests/tsky_results/filter_shift_init_from_files.json`.
* B. Synthetic (`tests/synth_filter_shift.py`; `fs_synth_items_*.json`). Generator: it already has two detectors and three filters (A/A1, A/A2, B/B1; pixels 0.10 and 0.12 arcsec); I added only `pair_shift_px`. Injected into the data positions (not the header positions the model gets): A/A1 (1.2, -0.8), A/A2 (-0.6, 1.6), B/B1 (0.5, 0.7) pixel (1.4, 1.7, 0.9 pixel).
  8 exposures per filter (6 training, 1 validation, 1 held out), all with the same roll (so that a common sky translation is a pixel shift per pair), seed 0, item sky with 1500 items, 180 steps each (about 190 s), float32, one thread. Held-out score = mean chi-squared per pixel over the 3 held-out exposures, the package's own (noise-only value 0.989).
  | run | held-out chi-squared | against reference | max remaining shift error after removing the common sky translation |
  |---|---|---|---|
  | reference: no injected shift, switch off | 1.426 | - | - |
  | injected, switch off | 1.751 | +22.8 % | (shift not modelled) 1.57 pixel |
  | injected, learned, 0.03 pixel per step | 1.248 | -12.5 % | 1.21 pixel (recovered minus injected, per pair: (-1.09, 0.44), (0.21, -1.05), (-0.26, -0.86)) |
  | injected, learned, 0.15 pixel per step (second and last attempt) | 1.379 | -3.3 % | 0.98 pixel (recovered: (0.24, -0.54), (-0.52, 0.72), (0.23, -0.16)) |
  The common translation (the origin the data cannot fix) is removed by the best sky-plane vector c* (weights = training exposures), residual = recovered minus injected minus J_pair^-1 c*; in the 0.03 run: (-0.69, 0.99), (0.61, -0.51), (0.07, -0.40) pixel. The injected-but-unmodelled reference value in the last column is what a zero shift would give.
  **Target missed on both counts.** Differences were recovered to 0.98 to 1.21 pixel, not 0.05 (a quarter to a third of the way from zero to the injected values, in the right direction), and the held-out score is not "within 2 %" of the reference: it is 3 to 12 % better, and the switch-off run is 23 % worse. With 3 held-out exposures, one seed, and a fit not converged
  (chi-squared 1.4 against 0.99), differences of this size between runs are not distinguishable from run-to-run variation (the unshifted and shifted references themselves differ by 23 %); I do not claim an improvement.
  Diagnosis (my reading, not tested further, no more tuning): the shift moves at 0.03 to 0.15 pixel per step against a sky that adapts faster; 180 steps are too few to travel 1.5 pixel while the item positions are also moving; the rate raised fivefold gave a better score and a smaller error, so the fit is limited by steps and rate, not by the sign.
  Sign and machinery check: with the grid sky, starting the learned fit at the injected values (`grid_sky/fs_synth_inj_learned_truthinit.json`) kept the shifts at the injected values apart from the common translation (remaining error 0.19 pixel).
  **Grid sky: not identifiable.** With `--sky-form grid` (12 spectral coefficients per node, 3 filters) each band can place its sources independently, so shifts between filters are not constrained by the data. Results (200 steps; `tests/tsky_results/grid_sky/`): reference 1.579, injected off 1.827 (+15.7 %), injected learned 1.866 (+18.1 %), recovered shifts 0.05 to 0.10 pixel (not learned). Only a sky in which a source has one position at all wavelengths (the item sky) lets filter shifts be learned; real runs
  with the grid sky should not use the switch.
* C. Tests. `tests/test_filter_shift.py`, 4 passed in 5 s: off is a no-op on the forward model (no new state; zero shift gives bit-identical output); positions change by J_e s and the gradient reaches the shift of the exposure's pair only; the projection removes exactly the common sky translation (adding any common translation first gives the same state afterwards; the removed part is exactly the mean; J-orthogonal translations; idempotent; pairs of different scale and orientation);
  held-out exposures take their pair's shift and the held-out fit leaves it fixed. Existing tests, unchanged, in this folder: `test_items_unit.py` 3, `test_items_tiles.py` 1, `test_pixel_gain.py` 7, `test_items_pyramid.py` 14 (run as two halves), `test_torchsky_backends.py` 10: all pass. Total 39 passed (35 existing, 4 new).
* Approximations and shortcuts. (1) First-order, one Jacobian per exposure (cut mean) instead of re-evaluating the pixel-to-sky map at p + s; the error is the Jacobian's variation over the detector times |s| (about 1 per cent of 1 pixel, under 1 mas for the F200W shift by my estimate; not measured). (2) The shift is not in the registration pre-pass; the pass's translations enter the shift through the projection. The initial coadd that warm-starts the sky in `make_sky` ignores `--filter-shift-init`; only the pre-pass coadd includes it. (3) The projection uses the training exposures only and is not weighted by the exposures' information.
  (4) With several disconnected common-shift components (tiles) the switch stops the run with a message; not implemented. (5) A common translation is not exactly a pixel shift when rolls differ; the projection then puts the difference in the translations. (6) The synthetic test used one roll and the item sky; the unit test covers differing orientations.
* Not done. No fit on real data with the switch (GPU job), so the cases `--rate-wcs telescope`, `cal` and calibrated input were not run through a fit: for `cal` and calibrated input the shift should come out near zero (the header is the pipeline's solution), which is not demonstrated; the files check above only supports `telescope`. The switch reads `extra["J"]`, present for every real cut, and the pair names match the keys of the file written from A. No second seed, no convergence run, no test of more than three pairs, no velocity-aberration term.
