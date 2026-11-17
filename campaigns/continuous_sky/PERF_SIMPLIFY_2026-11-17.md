# Step cost, speed-ups that keep results, and simplification (17 Nov 2026)

All timings: this machine, 2 CPU cores shared with nothing else, one thread, `torch.set_num_threads(1)`, the small synthetic field of
`tests/synth_small.py` (3 filters, 2 instruments, 18 training exposures, 4 patches per step unless stated, sky pitch 0.06 arcsec, item sky
1,500 items, 12 kernel nodes per level, 32 hidden units). No GPU was available: every GPU statement below is an inference from counted
synchronisation points, not a measurement. Run-to-run scatter of a step time on this machine is about 5 per cent; gains below that are not resolved.
The package before my changes is kept at `/tmp/claude-0/-home-claude/286b9f86-e33d-5aff-95e8-42cb96d595fe/scratchpad/base/` (not part of the deliverable).

## Part 1. Where a step spends its time

Method. Forward-pass pieces: wall-clock timers wrapped around the package functions (time inside a nested function is not counted in its
caller), 10 steps after 3 warm-up steps, set-up of `fit()` subtracted. Backward pass: one timer around `Tensor.backward` (there are two per step:
the sky-only step and the joint step), then split by operator family with `torch.profiler` (below). A step = sky-only step + joint step
(`--inner-sky 1` here; the default is 2, see Part 2).

Rows (a) to (h) are the categories you asked for. "Backward" is reported once for (a) to (d) together, because the same backward operators serve all
of them; the operator-family split follows the tables. (g) is the sum of the two measured parts: the parameter-only tables `prepare()` (called twice
per step before my change) and the per-patch quantities that do not depend on any parameter (kernel-node lists, data cut-outs).

### Grid sky, milliseconds per step (fraction of the step)

| part | 4 patches per step | 8 patches per step |
|---|---|---|
| (a) kernel field, forward | 88.3 (15.2 %) | 177.6 (16.2 %) |
| (b) render: FFT, overlap-add, reading level images at pixels | 169.0 (29.0 %) | 320.0 (29.1 %) |
| (c) reading the sky grid | 6.6 (1.1 %) | 11.3 (1.0 %) |
| (d) per-exposure terms, loss, regularisers | 19.1 (3.3 %) | 33.4 (3.0 %) |
| (e) backward pass, both backward calls of the step | 279.5 (48.0 %) | 533.0 (48.5 %) |
| (f) optimiser steps, gauge projections, gradient clipping | 2.5 (0.4 %) | 2.3 (0.2 %) |
| (g) recomputed, independent of the parameters updated in the step | 10.8 (1.9 %) | 14.7 (1.3 %) |
| (h) logging, bookkeeping, patch drawing | 0.8 (0.1 %) | 0.9 (0.1 %) |
| other forward glue | 5.6 (1.0 %) | 5.7 (0.5 %) |
| total | 582 | 1099 |

### Item sky (1,500 items, direct rasteriser), milliseconds per step

| part | 4 patches per step | 8 patches per step |
|---|---|---|
| (a) kernel field, forward | 94.6 (11.6 %) | 188.6 (13.6 %) |
| (b) render | 176.4 (21.6 %) | 351.8 (25.4 %) |
| (c) rasterising the items onto the grid, and reading it | 72.7 (8.9 %) | 78.5 (5.7 %) |
| (d) per-exposure terms, loss | 21.4 (2.6 %) | 38.2 (2.8 %) |
| (e) backward pass | 429.9 (52.7 %) | 704.6 (50.8 %) |
| (f) optimiser, projections | 2.4 (0.3 %) | 2.6 (0.2 %) |
| (g) recomputed, parameter-independent | 11.8 (1.4 %) | 15.1 (1.1 %) |
| (h) logging, bookkeeping | 0.9 (0.1 %) | 1.0 (0.1 %) |
| other forward glue | 5.7 (0.7 %) | 6.0 (0.4 %) |
| total | 816 | 1386 |

Note for the item sky: (c) is the raster forward only; it is run three times per step (twice in the sky-only and joint forward passes with gradients,
once more for the logged band image on refresh steps), and its backward (recomputing patches chunk by chunk) is inside (e). The raster does not
scale with patches per step (72.7 to 78.5 ms for twice the patches): it is paid once per forward evaluation of the sky.

### Backward pass by operator family (grid sky, 4 patches, `torch.profiler`, which inflates absolute times; shares are what matter)

Backward 324 ms profiled (forward 342 ms profiled in the same run). Fast Fourier transforms (real-to-complex and complex-to-complex) 93 ms (29 %);
matrix products and SiLU backward of the kernel network 37 ms (11 %); grid-sampler backward and scatter-add (sky and level-image reads) 26 ms (8 %);
element-wise multiply, copy, fill, divide, sum, add 126 ms (39 %), which belong to the overlap-add multiplication, the spline read and the
kernel exponent in proportions I did not separate; the rest is small operators. The kernel field is therefore about a quarter, the
render (FFT plus spline read) about half, of the backward pass; this is an operator-family assignment, not a causal split.

### Host synchronisation points per training step

Counted by wrapping `Tensor.item`, `float`, `int`, `bool`, `.cpu()`, `.tolist()`, `.numpy()` and device `torch.as_tensor` for 10 steps (grid sky and
item sky, 4 patches per step, two renders per step; set-up and logged steps excluded by differencing two runs). Calls on host tensors (the
Adam step counters, quantities already copied to the host) cost nothing on a GPU and are marked "host". The ten most frequent call sites, before my
changes (file and line in the package as it was):

| rank | per step | call | file:line (before) | on a GPU |
|---|---|---|---|---|
| 1 | 69.0 | `.item()` inside `Adam.step` (step counters) | train.py:685 (joint step), also :662, :699 | host, free |
| 2 | 30.4 | `int()` of window extents | render.py:331 | host (extents were copied once), free |
| 3 | 8.0 | `torch.as_tensor(e, device)` per patch | model.py:704 | host-to-device copy |
| 4 | 8.0 | same, in the per-patch finish | model.py:728 | host-to-device copy |
| 5 | 8.0 | `bool(self.e_stripes[e])` | model.py:602 | blocking read |
| 6 | 8.0 | `torch.as_tensor(e.eid, device)` per patch in the loss | train.py:132 | host-to-device copy |
| 7 | 7.6 | `.cpu()` of patch minimum and maximum (sky frame) | render.py:310, 314 | blocking read |
| 8 | 7.6 | `.cpu()` of patch extents (detector frame) | render.py:310, 314 | blocking read |
| 9 | 7.6 | `float(shift0[0])` | render.py:338 | host, free |
| 10 | 7.0 | `.item()` inside `Adam.step` of the nuisance step | train.py:428 | host, free |

Not in the top ten but blocking: `float(blur_moment)` and `nonzero` in the blur gauge, 2 + 2 per step (model.py:393, 415, called by every `prepare()`);
the noise-floor read (train.py:117) 1 per step; on the item sky the raster bucket sizes `.tolist()` (sky.py:528) 4 per step.
Python loops that issue separate tensor operations (not counted above): per patch, three loops (`render_batch` per-patch set-up,
`finish`, `pix_loss`), so 6 x patches iterations per step (24 at 4 patches), plus one loop over the levels (4 to 6 iterations) in each render.

| per step, grid sky, 4 patches | before | after |
|---|---|---|
| blocking device reads | 28.2 | 5.8 |
| host-to-device copies | 27.2 | 2.1 |
| `.item()` / `float` / `int` on host tensors | about 140 | about 120 |
| item sky, blocking device reads | 32.2 | 9.8 |

### Cost model (measured on this machine; extrapolations stated)

| variable | points measured | ms per step | reading |
|---|---|---|---|
| patches per step (grid) | 2, 4, 8 | 301, 618 (582 in the timed run), 1099 | linear: about 150 ms per patch |
| patches per step (items) | 4, 8 | 816, 1386 | 204 ms then 173 ms per patch: a fixed raster cost of about 75 ms plus about 150 per patch |
| training exposures in the field (grid, 4 patches) | 18, 42, 90 | 618, 577, 559 | none: step cost does not depend on the number of exposures; `fit()` set-up grows 0.57, 0.56, 0.72 s |
| items in the sky (4 patches, 0.06 arcsec grid) | 1,500; 6,000; 24,000 | 816; 754; 877 | weak, because the grid raster is small here; at 60,000 items on a 0.02 arcsec grid the raster will dominate: EXTRAPOLATED, not measured. The recorded GPU figures (1.45 s grid, 2.7 s items) imply about 1.25 s for the raster there |
| tiles | the test generator gave 2 tiles in all three cases (27, 30, 60 cuts) | 437, 486, 590 | per-step cost grows with the number of cuts only through patch and kernel lookups, about 4 ms per extra cut; the number of tiles itself was NOT varied (the generator does not offer it), so the tile-count dependence is not measured |

Conclusion for the aim (many exposures, tiles and items): step time is set by patches per step and, for the item sky, by the raster; it is
independent of the number of exposures. The cost of holding exposures and tiles is in memory and swapping, which this machine cannot show.

## Part 2. Speed-ups that keep the results

Hash checks (the defaults, same command as the record): grid sky, 40 steps: `ff0f1183...0e234a`, identical to
`synth_grid40_tsky_defaults.json`. Item sky, 150 steps: see the note under Part 3 (the state no longer contains the all-zero buffer `sky.resid`, so the
plain hash of the whole state necessarily changes; with that buffer put back by the check script the hash is `b20eb534...aedd1`, identical to
`synth_2_tsky_defaults.json`). Every other tensor of the state is bit-identical before and after.

| change | what | CPU gain | GPU effect (expected) |
|---|---|---|---|
| 1 | One `prepare()` per step instead of one in the sky-only step (no gradient) plus one in the joint step: it does not depend on the sky, so the joint-step tables are detached for the sky-only step (train.py). Saves 1 of 2 calls at `--inner-sky 1`, and 2 of 3 at the default `--inner-sky 2` | about 3.6 ms per step (of 600), under the noise | removes 1 to 2 blocking reads (the blur gauge reads and `nonzero`) and about 2 small kernels per call |
| 2 | Level-extent read of the batched render: the four minimum and maximum extents are stacked and copied to the host once instead of four times (render.py) | under 1 ms | 4 blocking reads per render group become 1: -11.4 per step at 4 patches |
| 3 | `bool(self.e_stripes[e])` read from a host copy, and not at all when `--stripes 0` (model.py) | under 1 ms | -8 blocking reads per step |
| 4 | Device index tensors of the exposures made once (`model.et`), instead of a new `torch.as_tensor(e, device)` three times per patch (model.py, train.py) | about 1 ms | -25 host-to-device copies per step |
| 5 | Noise-floor warning: the check that read the floor to the host every step now keeps a running "first step above 0.2" on the device and reads it at logged steps and at the end (train.py). The recorded step is the same; the warning line appears at the next logged step, and the level is compared as round(x, 6) > 0.2 up to a difference of 5e-7 (unrounded float) | under 1 ms | -1 blocking read per step |

All five together, same machine, interleaved runs of the old and the new package, 10 steps each: grid sky 640, 624 ms before; 621, 574 after
(mean 632 to 598, -5 %). Item sky 854, 849 ms before; 781, 863 after (mean 852 to 822, -3.5 %). Individual gains are inside the scatter and are
not separately resolved; the synchronisation counts, not the milliseconds, are the evidence. On a GPU the removal of about 23 blocking reads and
25 copies per step matters more than on this CPU; the size of the gain is not known without a measurement (`tests/bench_step.py` gives it).

Changes considered and NOT applied (would change the hash at rounding level, or are not provably identical):
- Replace the data-dependent branch of the blur gauge (`dilation()`: `if s1 == 1.0: return None`) by a device-side `torch.where`: removes the last
  per-step blocking read of `prepare()` (1 per step), but the dilation tensor then enters the graph when it is 1, so gradients pick up exact
  zeros and the kernel evaluation divides by 1.0 (exact); I did not prove bit identity and did not apply it.
- Batch `finish` and `pix_loss` over patches (loop of 6 x patches small operator groups per step): the sum over patches would be accumulated in a
  different order; expected CPU gain 2 to 4 ms per step, GPU gain larger (the kernel launches of 24 small groups). Rounding-level change.
- Skip padding: patches with fewer region nodes are padded to the largest (`nmax`, `Npm`) in the batch; sorting patches by size would change the order of the
  sum. Not measured.

Also written: `tests/bench_step.py` (grid or items; sizes as lists of values: `--batch-patches 4 8 --n-per-filter 8 16 --item-count 1500 6000
--sky-pitch 0.06 0.04`; `--device cpu|cuda|auto`; prints one JSON line per configuration with milliseconds per step and peak memory;
set-up excluded by differencing two fits; CUDA clock after `torch.cuda.synchronize()`, peak from `max_memory_allocated`; on the CPU, peak resident size
of the process). Checked on this CPU: grid, 2 and 4 patches: 353.9 and 617.8 ms, 946 and 1153 MiB; items, 4 patches, 1,500 items: 834.4 ms, 1175 MiB.
The CUDA branch has NOT been run (no GPU here).

## Part 3. Simplification

Removed entirely, with options, code and tests: `--item-relocate` (ItemSky.relocate, the residual buffer and hook, the four relocation marks in `fit`,
`rec["relocated"]`), `--item-smooth-pitch` (the coarse grid parameter `S`, `smooth_up`, its Adam group, its regulariser `reg_terms`, its
averaging and zeroing; `sky_regs` now returns 0 for the item sky, as before for the default settings), `--render-impl` (option, refusal, and the
test of the refusal). `--kernel-norm-radius` kept. Dead code removed: `train_ids_by_filter`, imports `quad_points` (model.py) and `LATTICE` (train.py);
the unused `lr_sky_sig` argument of `opt_groups`. Nothing else was unreachable by search; the search was: every function name that occurs once in the
package and not in a test, plus unused imports, which I checked by running the tests.

Tests changed: removed `test_D_render_impl_torchsky_is_refused`; removed the smooth-grid part of the finite-difference gradient test and the
`smooth` argument in `test_items_unit.py`; `test_items_tiles.py` no longer sets smooth grid or relocation; `synth_small.py` and `synth_filter_shift.py`
lost the two options; `test_pixel_gain.py` sets the host copy of the stripe flags next to the buffer where it edits the buffer after construction
(a model edited after construction must update both: this is the one fragile point of change 3).

| file | lines before | lines after |
|---|---|---|
| fieldop/__init__.py | 192 | 190 |
| fieldop/__main__.py | 5 | 5 |
| fieldop/bandpass.py | 102 | 102 |
| fieldop/cli.py | 472 | 464 |
| fieldop/data.py | 573 | 573 |
| fieldop/kernel.py | 204 | 204 |
| fieldop/model.py | 967 | 978 |
| fieldop/render.py | 423 | 427 |
| fieldop/sky.py | 741 | 660 |
| fieldop/synth.py | 241 | 241 |
| fieldop/tiles.py | 172 | 172 |
| fieldop/train.py | 1041 | 1030 |
| fieldop/tsky_backends.py | 204 | 204 |
| total | 5,337 | 5,250 |

Options: 161 before, 158 after. Lines added by Part 2 (host-copy and index helpers, device-side floor check): about 25; removed by Part 3: about 110.

State files: the item sky no longer registers the all-zero buffer `resid`. States saved earlier still load (the loader drops that key). The plain
state hash of the item run therefore changes to `840b7a11fced01f92c6ce3e2d23ca1403d428f3fef6fe233d38a8f841fd10f23`; every other tensor is bit-identical (the
check script now also prints `state_sha256_with_legacy_resid`, which equals the recorded `b20eb534...aedd1`). I did not overwrite the recorded
result files; the maintainer may record the new hash.

### Option inventory (158 options)

Status rules. "used by the adopted configurations (named on a launch line)" = appears on one of the three launch lines. "switch between adopted
alternatives" = the launch lines or tests use more than one value (grid or items sky, direct or pyramid raster, cal or rate input, ...). "diagnostic
only" = output, prediction or debugging controls. "used ... (read at its default)" = not set by the launch lines, but the adopted path reads its default
value every run (a tunable or an alternative that is on by default). "never exercised: candidate" = no test sets it, no launch line sets it, and
it guards an alternative that is off by default; the number is how many package lines mention the option (a guard count, not the size of the
guarded branch; branch sizes added where I measured them). Whether a test sets an option was found by searching the test files for its name, so an option
set only through a configuration file would be missed. Candidates are not removed.

| option | default | meaning | status |
|---|---|---|---|
| --field | SMACS | field name (selects the protocol files) | used by the adopted configurations (read at its default) |
| --split | dither | data split name inside the protocol | used by the adopted configurations (read at its default; also set by tests) |
| --protocol-root | /arc/projects/ots/adas | folder of the split files | used by the adopted configurations (read at its default) |
| --splits |  | default: <protocol-root>/splits_<FIELD>.json | used by the adopted configurations (read at its default; also set by tests) |
| --box-id | 0 | single scoring box number | never exercised: candidate; guards 1 reference lines |
| --tiles |  | several scoring boxes (sky patches) under one operator: 'all' or a comma list of box ids ('' = --box-id only) | used by the adopted configurations (named on a launch line) |
| --tile-weight-power | 0.5 | tile k is visited with probability proportional to (training pixels)^power | never exercised: candidate; guards 1 reference lines |
| --visit-steps | 10 | optimisation steps per tile visit | used by the adopted configurations (read at its default; also set by tests) |
| --gpu-resident-gb | 4.0 | pixel data of tiles kept on the device (LRU); tiles beyond it are swapped | used by the adopted configurations (named on a launch line) |
| --load-workers | 8 | threads reading files (several tiles) | never exercised: candidate; guards 1 reference lines |
| --sky-swap | 0 | several tiles: inactive skies and their Adam state are kept on the host | switch between adopted alternatives |
| --held-rounds | 4 | several tiles: rounds over the tiles in the held-out nuisance fit | used by the adopted configurations (read at its default; also set by tests) |
| --held-ref-cuts | 24.0 | several tiles: per-tile held-out steps = --held-steps x (held-out cuts of the tile) / this | never exercised: candidate; guards 2 reference lines |
| --all-box-files | 0 | also load split files not listed as overlapping the box | never exercised: candidate; guards 1 reference lines |
| --method-name | fieldop | name used in output file names | diagnostic only |
| --out | out_fieldop | output folder | diagnostic only |
| --data-root |  | directory of the _cal files (default: paths in the splits file) | never exercised: candidate; guards 1 reference lines |
| --rate-dir |  | folder of the count-rate files | used by the adopted configurations (named on a launch line) |
| --svo-cache |  | cache folder for filter curves | diagnostic only |
| --shifts-from |  | register_xcorr.json (LMC): initial per-exposure registration shifts | used by the adopted configurations (named on a launch line) |
| --filters |  | filters to use | used by the adopted configurations (read at its default; also set by tests) |
| --max-exposures | 0 | cap on exposures read (debugging) | diagnostic only |
| --n-val-groups | 1 | dither groups per filter carved from training for the tau validation likelihood | used by the adopted configurations (named on a launch line) |
| --epoch-min-days | 1.0 | epoch axes need a training time span at least this long | never exercised: candidate; guards 1 reference lines |
| --epoch-axes | 1 | allow kernel / throughput variation with observing epoch | never exercised: candidate; guards 1 reference lines |
| --write-pred | 1 | write held-out predictions to disk | diagnostic only |
| --pix-default | 0.031 | pixel scale used when a file has none [arcsec] | used by the adopted configurations (read at its default) |
| --sky-pitch | 0.04 | sky grid spacing [arcsec] | used by the adopted configurations (named on a launch line) |
| --sky-margin | 2.0 | sky domain = box + 2 x margin [arcsec] | used by the adopted configurations (read at its default; also set by tests) |
| --data-margin | 1.0 | data region = box + 2 x margin [arcsec] (< sky margin) | used by the adopted configurations (read at its default; also set by tests) |
| --spec-k | 12 | number of spectral basis functions | used by the adopted configurations (read at its default) |
| --sky-neg | 1.0 | weight of the negative-sky penalty (grid sky) | used by the adopted configurations (read at its default; also set by tests) |
| --sky-precond | 0.02 | diagonal preconditioner of the sky: 1 + f x (local warm-start S/N), 0 = off | used by the adopted configurations (read at its default) |
| --sky-precond-cap | 2000.0 | cap of the sky preconditioner | used by the adopted configurations (read at its default) |
| --sky-tv | 0.0 | weight of total-variation penalty on the grid sky | never exercised: candidate; guards 3 reference lines (3 lines in sky_regs) |
| --xfit | 1 | cross-fitting sky views (1 = off, 2 = two folds) | never exercised: candidate; guards 8 reference lines (about 45 lines (fit loop, ItemSky.average_views, SkyGrid.average_views)) |
| --xfit-frac | 0.6 | fraction of the fit before the two sky views are averaged | never exercised: candidate; guards 1 reference lines |
| --sky-form | grid | sky parametrisation: coefficient grid, or Gaussian items rasterised onto that grid (sky.ItemSky) | switch between adopted alternatives |
| --item-count | 20000 | items per sky (tile); split over --item-levels lattices in proportion 4^-l | used by the adopted configurations (named on a launch line) |
| --item-levels | 3 | coarse-to-fine seeding lattices | used by the adopted configurations (named on a launch line) |
| --item-floor | 0.0 | minimum item width [arcsec] (0 = 0.4 x --sky-pitch) | used by the adopted configurations (read at its default; also set by tests) |
| --item-max-width | 0.0 | maximum item width [arcsec] (0 = 16 x --sky-pitch); with --item-raster direct, patches are capped at +-128 ... | used by the adopted configurations (named on a launch line) |
| --item-chunk | 0 | elements (items x patch nodes x K) per rasterisation chunk; 0 = 4 M. On a GPU the item scatter is bound by ... | used by the adopted configurations (named on a launch line) |
| --item-buckets | fine | patch-size ladder: fine (18 to 25 sizes) or coarse (powers of two: fewer distinct patch shapes, so fewer la... | switch between adopted alternatives |
| --item-raster | direct | item rasteriser: direct (patch of +-4 widths at the sky pitch, cost per item ~ (width / pitch)^2) or pyrami... | switch between adopted alternatives |
| --item-impl | prototype | item rasteriser code: the prototype's own (ItemSky, default) or torchsky's GaussianItemScene (same paramete... | used by the adopted configurations (read at its default; also set by tests) |
| --item-patch | square | pyramid only: patch of each item, square (+-4 larger widths) or sheared (+-4 widths in the item's decorrela... | switch between adopted alternatives |
| --item-lr-pos | 0.05 | Adam step of item positions [sky pitches] | used by the adopted configurations (read at its default; also set by tests) |
| --item-lr-width | 0.03 | Adam step of item log widths | used by the adopted configurations (read at its default; also set by tests) |
| --item-lr-angle | 0.05 | Adam step of item angles [rad] | used by the adopted configurations (read at its default; also set by tests) |
| --item-lr-coef | 0.05 | Adam step of item log spectral coefficients | used by the adopted configurations (read at its default; also set by tests) |
| --kernel-extent | 10.0 | kernel half-size [arcsec] | used by the adopted configurations (read at its default; also set by tests) |
| --kernel-norm-radius | 0.0 | arcsec; > 0: the kernel's unit integral is taken inside this radius only (flux convention F41); 0 = whole s... | never exercised: candidate; guards 1 reference lines (kept by instruction (Levels.norm_log and 6 lines)) |
| --render-pitch | 0.0 | 0 = per channel: largest sky_pitch/k <= pixel / --render-os-pix | used by the adopted configurations (read at its default; also set by tests) |
| --render-os-pix | 6.0 | render oversampling per detector pixel | used by the adopted configurations (read at its default) |
| --interp | bspline | interpolation of level images at pixel positions (bspline or bicubic) | used by the adopted configurations (read at its default; also set by tests) |
| --level-factor | 2 | pitch ratio between kernel levels | used by the adopted configurations (read at its default) |
| --level-nh | 24 | kernel nodes per level half-width | used by the adopted configurations (read at its default; also set by tests) |
| --level-overlap | 0.5 | level l hands over to l+1 between a R_l and R_l | used by the adopted configurations (read at its default) |
| --region-px | 256 | pixels per kernel-node region | used by the adopted configurations (read at its default; also set by tests) |
| --n-lambda | 5 | wavelength nodes per filter | used by the adopted configurations (read at its default; also set by tests) |
| --render | fft | renderer: fft (default) or quad (per-pixel quadrature) | used by the adopted configurations (read at its default; also set by tests) |
| --quad-nrad | 12 | radial points of the quadrature renderer | never exercised: candidate; guards 1 reference lines (with render=quad: render_quad 23 + quad_points 34 lines (render=quad itself is exercised by a test)) |
| --quad-naz | 48 | azimuthal points of the quadrature renderer | never exercised: candidate; guards 1 reference lines (see quad-nrad) |
| --k-hidden | 64 | kernel network width | used by the adopted configurations (read at its default; also set by tests) |
| --k-layers | 3 | kernel network depth | used by the adopted configurations (read at its default) |
| --dh | 16 | kernel code length | used by the adopted configurations (read at its default) |
| --enc-r0 | 0.0 | core radius of the log-radius encoding (0 = half the render pitch) | used by the adopted configurations (read at its default) |
| --render-jitter | 1 | random sub-pitch phase of the level-0 kernel grid in training (F11) | used by the adopted configurations (read at its default) |
| --enc-nrad | 8 | radial terms of the log-polar encoding | used by the adopted configurations (read at its default) |
| --enc-mmax | 12 | angular terms of the encoding | used by the adopted configurations (read at its default) |
| --filter-input | curve | filter described to the network by its curve or an index | used by the adopted configurations (read at its default) |
| --k-space | learned | kernel variation over the detector: learned / off | used by the adopted configurations (read at its default) |
| --k-time | learned | kernel variation over time: learned / off | used by the adopted configurations (read at its default) |
| --k-lambda | learned | kernel variation with wavelength: learned / off | used by the adopted configurations (read at its default) |
| --k-exp | learned | per-exposure kernel variation: learned / off | used by the adopted configurations (read at its default) |
| --warm-kernel-steps | 400 | steps of the physics-only kernel warm start | used by the adopted configurations (read at its default; also set by tests) |
| --tau-mode | learned | variability scales learned or fixed | used by the adopted configurations (read at its default; also set by tests) |
| --tau0-code | 0.3 | starting scale of kernel code variability | used by the adopted configurations (read at its default) |
| --tau0-g | 0.02 | starting scale of throughput variability | used by the adopted configurations (read at its default) |
| --tau0-D | 0.005 | starting scale of distortion variability | used by the adopted configurations (read at its default) |
| --tau-prior | 0.001 | (read nowhere) | never exercised: candidate; guards 0 reference lines (0 lines: the value is never read, the option can go) |
| --tau-start | 0.35 | fit fraction at which scale learning starts | used by the adopted configurations (read at its default) |
| --val-every | 4 | steps between validation updates | used by the adopted configurations (read at its default; also set by tests) |
| --val-nuis-steps | 2 | nuisance steps on validation exposures per update | used by the adopted configurations (read at its default) |
| --lr-tau | 0.05 | step of log tau per evidence update | used by the adopted configurations (read at its default) |
| --tau-accum | 4 | validation evaluations accumulated per tau step | used by the adopted configurations (read at its default) |
| --tau-beta | 0.95 | EMA factor of the validation-gradient evidence | used by the adopted configurations (read at its default) |
| --tau-occam-z | 2.0 | evidence (in sd) an axis must show to grow | used by the adopted configurations (read at its default) |
| --tau-warm-evals | 3 | validation evaluations before scales may grow | used by the adopted configurations (read at its default) |
| --tau-code-max | 1.5 | cap of the code-space kernel variability scales | used by the adopted configurations (read at its default) |
| --tau-g-max | 0.3 | cap of the throughput function-axis scales (rms log g on the footprint lattice) | used by the adopted configurations (read at its default) |
| --tau-d-max | 0.03 | cap of the static / drift distortion scales [arcsec] | used by the adopted configurations (read at its default) |
| --ref-footprint | 1 | detector fields normalised (and the throughput gauge anchored) on a 3x3 lattice spanning each detector's tr... | used by the adopted configurations (read at its default) |
| --ab-every | 2 | likelihood-ratio (axis off vs on) evidence every N validation evaluations (0 = off) | used by the adopted configurations (read at its default) |
| --ab-beta | 0.95 | forgetting factor of the likelihood-ratio evidence | used by the adopted configurations (read at its default) |
| --em-damp | 0.5 | exponent of the damped EM step of the random-effect scales | used by the adopted configurations (read at its default) |
| --amplitude | structured | throughput form: structured or free | used by the adopted configurations (read at its default) |
| --gauge-ref-inst | -1 | instrument fixing the throughput gauge (-1 = automatic) | used by the adopted configurations (read at its default) |
| --gauge-shift | 0.0 | extra penalty on the common learned shift (the exact projection is always on) | never exercised: candidate; guards 2 reference lines |
| --blur-floor | 1.0 | blur gauge: core m2 of the sharpest kernel class >= this x (sky pitch)^2 / 3 (0 = off) | used by the adopted configurations (read at its default) |
| --pedestal-gauge | 1 | per filter, training background constants (relative to the exposure medians) sum to zero | used by the adopted configurations (read at its default) |
| --sky-zero | median | reference level of each training exposure for the pedestal gauge (the split between a constant sky and the ... | used by the adopted configurations (read at its default; also set by tests) |
| --sky-zero-snr | 0.3 | --sky-zero masked: a pixel counts as source-free when /S/ < this x its sigma | used by the adopted configurations (read at its default) |
| --sky-zero-smooth | 1.5 | --sky-zero masked: Gaussian width [detector pixels] of the smoothing of S before the threshold | used by the adopted configurations (read at its default) |
| --sky-zero-at | 0,0.1,0.25,0.5,0.75 | --sky-zero masked: fit progress fractions at which the levels are re-evaluated | used by the adopted configurations (read at its default) |
| --sky-positivity-margin | 0.0 | positivity (grid penalty, item exponentials) applies to sky + p instead of the sky, p = this x the noise of... | used by the adopted configurations (read at its default; also set by tests) |
| --distortion | 1 | learn geometric distortion (1 = on) | used by the adopted configurations (read at its default; also set by tests) |
| --affine-prior | 0.01 | prior width of per-exposure affine terms | used by the adopted configurations (read at its default) |
| --input-level | cal | data level: cal (surface brightness) or rate (count rate) | switch between adopted alternatives |
| --rate-wcs | cal | --input-level rate: header WCS (pixel positions) from the matching _cal file (distortion-corrected SIP, def... | switch between adopted alternatives |
| --rate-units | dn | --input-level rate: units of the sky. 'dn' = count rate per arcsec^2 (today; differs per filter and detecto... | switch between adopted alternatives |
| --rate-photom |  | --rate-units sky: JSON table {"DETECTOR/FILTER": MJy/sr per DN/s} (and optionally {"DETECTOR/FILTER/area": ... | never exercised: candidate; guards 2 reference lines |
| --nuisance-steps | mean | Adam step of the data-unit nuisances (background, stripes): lr x mean pipeline sigma over all exposures ('m... | switch between adopted alternatives |
| --loader | astropy | exposure files read by the prototype's astropy code (default) or by torchsky's field_observation_from_expos... | used by the adopted configurations (read at its default; also set by tests) |
| --pixel-gain | off | per-pixel log-gain map (flat field) learned from the exposures: one per detector, or per detector and filte... | switch between adopted alternatives |
| --pixel-gain-poly | 2 | gauge of the gain map: monomials of total degree <= this are projected out over the observed pixels of each... | used by the adopted configurations (named on a launch line) |
| --lr-gain | 0.003 | Adam step of the log-gain map (F45; chosen on the synthetic field among 3e-4, 1e-3, 3e-3, 1e-2) | used by the adopted configurations (named on a launch line) |
| --pixel-gain-start | 0.1 | the gain map is stepped from this fraction of the fit on | used by the adopted configurations (named on a launch line) |
| --filter-shift | off | learned position shift (2 numbers, detector pixels, applied to the pixel coordinates before the pixel-to-sk... | used by the adopted configurations (read at its default; also set by tests) |
| --filter-shift-init |  | JSON file {"DETECTOR/FILTER": [dx, dy]} (detector pixels) giving the starting shifts | used by the adopted configurations (read at its default; also set by tests) |
| --lr-filter-shift | 0.03 | Adam step of the pair shift, detector pixels (same warm-up and cosine factor as the others) | used by the adopted configurations (read at its default; also set by tests) |
| --dq-ignore-bits | 0 | data-quality bits that do not exclude a pixel (F42); 0 = strict DQ == 0; 2 = keep pixels flagged SATURATED ... | used by the adopted configurations (named on a launch line) |
| --nonlin | 0 | learn a detector non-linearity (0 = off) | never exercised: candidate; guards 1 reference lines (about 25 lines (network, apply_nonlin, report)) |
| --pattern-grid | 8 | detector pattern grid size (0 = off) | used by the adopted configurations (read at its default; also set by tests) |
| --stripes | 1 | learn row and column stripes (1 = on) | used by the adopted configurations (read at its default; also set by tests) |
| --stripe-l2 | 0.001 | penalty on stripe size | never exercised: candidate; guards 1 reference lines |
| --floor0 | 0.01 | starting fractional noise floor | used by the adopted configurations (read at its default; also set by tests) |
| --noise-floor-max | 0.0 | upper bound of the learned fractional noise floor (softplus(floor_raw)), enforced by clamping floor_raw aft... | used by the adopted configurations (read at its default; also set by tests) |
| --loss | huber | huber or squared loss | used by the adopted configurations (read at its default; also set by tests) |
| --huber-delta | 3.0 | Huber threshold [sigma] | used by the adopted configurations (read at its default) |
| --budget-min | 30.0 | wall-clock budget of the fit [min] | used by the adopted configurations (named on a launch line) |
| --steps | 1000000 | fixed step count (tests, benchmarks) | diagnostic only |
| --batch-patches | 8 | patches per step | used by the adopted configurations (named on a launch line) |
| --patch-arcsec | 2.5 | patch size [arcsec] | used by the adopted configurations (read at its default) |
| --inner-sky | 2 | sky-only steps before each joint step | used by the adopted configurations (read at its default; also set by tests) |
| --lr-op | 0.001 | Adam step of the operator | used by the adopted configurations (read at its default) |
| --lr-sky | 0.3 | Adam step of the grid sky | used by the adopted configurations (read at its default) |
| --lr-bkg | 0.1 | Adam step of backgrounds | used by the adopted configurations (read at its default) |
| --lr-affine | 0.1 | Adam lr of the raw affine (x 1e-3 matrix, x 10 mas shift) | used by the adopted configurations (read at its default) |
| --lr-codes | 0.02 | Adam step of kernel codes | used by the adopted configurations (read at its default) |
| --clip | 10.0 | gradient clipping norm of the operator | used by the adopted configurations (read at its default) |
| --prior-pixels | 0.0 | random-effect priors are divided by this pixel count (0 = mean valid pixels per training exposure) | used by the adopted configurations (read at its default) |
| --is-frac | 0.5 | share of each batch drawn from bright patches | used by the adopted configurations (read at its default) |
| --is-snr | 20.0 | peak signal-to-noise defining a bright patch | used by the adopted configurations (read at its default; also set by tests) |
| --is-refresh | 200 | steps between refreshes of the bright-patch list | used by the adopted configurations (read at its default) |
| --held-steps | 150 | held-out nuisance-fit steps | used by the adopted configurations (named on a launch line) |
| --held-reg-steps | 100 | held-out exposures: translation + background pre-fit before the nuisance fit | used by the adopted configurations (named on a launch line) |
| --reg-prepass-iters | 2 | registration pre-pass: translation-only fits against the warm-start sky | used by the adopted configurations (named on a launch line) |
| --reg-prepass-steps | 150 |  | used by the adopted configurations (named on a launch line) |
| --pred-margin | 0.3 | margin of predicted images [arcsec] | diagnostic only |
| --log-every | 25 | steps between log lines | used by the adopted configurations (named on a launch line) |
| --seed | 0 | random seed | used by the adopted configurations (read at its default; also set by tests) |
| --device | auto | cpu, cuda or auto | used by the adopted configurations (read at its default; also set by tests) |
| --ckpt-kernel | 0 | activation checkpointing of the kernel field (GPU memory) | switch between adopted alternatives |
| --batch-render | 1 | batched FFT rendering of the patches of a step | used by the adopted configurations (read at its default; also set by tests) |
| --pred-batch | 8 | pixels per prediction batch | diagnostic only |
| --render-frame | detector | frame of the fine (tent-sampled) level grids | never exercised: candidate; guards 1 reference lines |

Candidates for later removal, summed: about 20 options. Together they guard roughly 150 lines (cross-fitting about 45, non-linearity about 25,
`--tau-prior` 0 lines, because its value is read nowhere, and the rest a few lines each). `--render quad` and `--batch-render 0` (not-batched renderer: `render`,
`render_fft`, `kernel_level_values`, `node_kernels`, `patch_loss`, about 150 lines) are exercised by tests and are not candidates here.

## Largest remaining costs and what would remove them (ranked, grid sky 4 patches; item sky in brackets)

1. Backward pass through the render, 48 % [53 %] of the step (280 ms [430 ms]): about a third in FFTs, a quarter in the kernel network.
   Removal: evaluate the FFT overlap-add in half precision or with fewer levels, or use `--ckpt-kernel` only where memory demands it (it adds a kernel forward to
   the backward pass). Expected gain 20 to 30 % of the step. Results WOULD change (rounding or model).
2. Forward render (FFT, overlap-add, spline read) 29 % [22 %] (169 ms [176 ms]). Removal: skip level 0 for patches whose region is unrotated
   and aligned with the render pitch; reuse the forward FFT of the sky between the sky-only and joint step is impossible (the sky changes between them).
   Expected gain 5 to 10 %. Results would change only for the skip, not for the reuse.
3. Kernel field forward, 15 % [12 %] (88 ms [95 ms]), evaluated twice per step on different patches (with and without gradient). Removal: share the kernel
   values of nodes that two patches of the same exposure have in common (rare with random patches), or lower `--k-hidden`; reducing hidden units changes the model.
   Expected gain 3 to 8 %. Results would change.
4. Item raster, 9 % [forward 73 ms] plus its backward (item sky only, grows with items and with the grid): at 60,000 items on a 0.02 arcsec grid I extrapolate
   it to the largest single cost of the item step (about 1.25 s of 2.7 s on the recorded GPU). Removal: rasterise once per step and share it between the
   sky-only and joint step is not possible (the sky parameters move between them); skip the sky-only step for the item sky (`--inner-sky 1`; the launch lines use the default 2,
   which runs three rasters per step): about 30 % of the step on the item sky, but the fit changes.
5. Synchronisation and launch overhead, small on this CPU (under 3 %) but not on a GPU: 5.8 blocking reads and about 24 patch-level loop groups per step remain. Removal:
   batch `finish` and `pix_loss` over patches, the last blur-gauge read (see "not applied"); results change at rounding level. Gain on a GPU not known until measured with `tests/bench_step.py`.

## Not done

- No GPU measurement (none available). The synchronisation counts and the extrapolated item cost are inferences.
- The number of tiles was not varied; the tile dependence is not measured.
- Backward pass is split by operator family, not by the categories (a) to (d).
- Per-change CPU gains are below the run-to-run scatter; only the five changes together were timed.
- The inventory's "never exercised" test is a text search of the test files; the guard counts are reference lines, and branch sizes were measured for four options only.
