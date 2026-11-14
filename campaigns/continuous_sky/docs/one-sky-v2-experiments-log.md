# One sky, version 2: log of the small experiments (newest first)

Plan and design: `claude/one-sky-v2-raw-anyscale-ensemble.md` (living doc, tab "Version 2": https://claude.ai/artifact/DYP9UJuvosgpaz7BGG4yr5#bcefa2e8-f671).

## 14 Nov 2026, 04:10 to 06:00 UTC: first real-data results for the item sky; the re-fetched files differ from the protocol's

Scores below are mean squared held-out residuals in noise units (1 is ideal, lower is better). "Paired" = difference on identical pixels with a 68% interval from a bootstrap over the 12 dither groups; negative = the first-named fit predicts better. All fits: SMACS, 86 training exposures, 24 held out, 200 minutes on one GPU slice.

**1. The re-fetched SMACS files are not the files the protocol was built on.** Tonight's grid fit (`onesky_grid004`: same code path and arguments as the 1.75 reference `fieldop_fix_200`) scored 2.38, with F277W 4.78 against 1.97. Cause, checked on one exposure and on the whole set:
- The `_cal` files re-fetched from MAST on 13 Nov (CAL_VER 3.0.0, CRDS context 1584, processed 24 Oct 2026) flag the cores of the brightest sources SATURATED (bit 2: some groups saturated, slope from the others). The protocol's held-out masks were built from the earlier files with DQ == 0 and call those pixels valid: 61 of the 864 pixels above signal-to-noise 200 in one F277W exposure; 7,334 of 2.11 M scored pixels overall (0.35%).
- The fit uses DQ == 0, so it never saw those cores and under-predicts them (0.84 of the data; 415 against 53 in that bin). In every other signal-to-noise bin tonight's grid fit equals the reference to within 0.03.
- Two additions (mine): `score_paired.py --dq-zero` also requires the scored file's flag to be zero; `fieldop --dq-ignore-bits 2` keeps pixels flagged only SATURATED. With `--dq-zero`: `onesky_grid004` 1.58, `fieldop_fix_200` 1.56, official pipeline 1.61 (paired against the pipeline: -0.027 [-0.169, +0.087] and -0.047 [-0.183, +0.059]). So the reference is reproduced on the new files. All paired numbers below use `--dq-zero`.
- A grid fit with `--dq-ignore-bits 2` is running (`onesky_grid004_dq2`) to confirm that the 1.75 comes back on the protocol's own mask.
- Not checked: the earlier files are gone, so "the earlier files had flag 0 there" is inferred from the masks.

**2. Item sky against grid sky, same data, same 200 minutes** (60,000 Gaussian items = 1.02 M sky numbers against 4.51 M; 2,327 steps against 9,353). Paired, item minus grid, by signal-to-noise of the official mosaic:

| Signal-to-noise | pixels | without the scorer's offset | with the scorer's offset |
|---|---|---|---|
| below 1 | 11,287 | +0.079 [+0.054, +0.103] | -0.157 [-0.173, -0.150] |
| 1 to 5 | 664,410 | -0.145 [-0.166, -0.133] (0.95 against 1.10) | +0.017 [-0.046, +0.056] |
| 5 to 20 | 1,043,065 | -0.181 [-0.210, -0.158] (1.01 against 1.19) | -0.002 [-0.049, +0.034] |
| 20 to 50 | 283,221 | -0.102 [-0.144, -0.053] (1.22 against 1.32) | -0.017 [-0.068, +0.041] |
| 50 to 200 | 98,484 | +5.85 [+3.83, +8.62] (9.0 against 3.2) | +5.85 [+3.84, +8.60] |
| above 200 | 3,798 | -6.1 [-30.8, +17.4] | -6.2 [-30.9, +17.2] |
| headline (5 and above) | 1,428,568 | +0.235 [+0.075, +0.419] | +0.382 [+0.226, +0.569] |

- As a predictor of the whole exposure (no offset added by the scorer) the item sky is better than the grid from signal-to-noise 1 to 50, which is 95% of the scored pixels, and better than the official pipeline from 1 to 20 (-0.141 [-0.147, -0.134] and -0.093 [-0.108, -0.080]); equal to it from 20 to 50.
- It is much worse from 50 to 200 (bright cores): too few steps (2,327) and a blunter sharpest point source (kernel reference moment 853e-6 arcsec^2 against the floor 533e-6).
- **The gain from 1 to 50 disappears when the scorer adds its own background offset** (the protocol's headline form; the offset is the clipped median residual over the faintest pixels). So the item fit is inconsistent between the faintest pixels and the rest by roughly 0.45 noise units: a real defect for faint diffuse light. Likely cause: 27% of the items sit at the width cap (0.32 arcsec) and 5% have both widths there, so the items cannot carry diffuse light and the per-exposure background compensates on average. Round 3 tests this.
- Item sky at 0.02 arcsec raster pitch (`onesky_items002p`, 1,850 steps): same gains without offset (1 to 5: -0.139; 5 to 20: -0.179; 20 to 50: -0.149 [-0.181, -0.115]) and a smaller bright-core deficit (+3.8 [+2.5, +5.5]). Against the pipeline without offset: -0.135, -0.091, -0.065 [-0.121, -0.024] and unresolved above 50; headline +0.03 [-0.15, +0.19]. The grid sky at 0.02 arcsec is worse than at 0.04 in every bin (headline 3.43 against 1.58): a finer grid hurts the grid form and helps the item form.

**3. The item sky is flux-calibrated and gives a diffraction-like kernel; the grid sky is neither.** Fitted sky over official mosaic on the brightest galaxy (my script `flux_check/flux_ratio.py`; one object, annulus background):

| Fit | radius 1 arcsec | radius 0.3 arcsec | box total | kernel fraction inside 2 arcsec (short / long wavelength) |
|---|---|---|---|---|
| grid 0.04 (`fieldop_fix_200`, reference) | 2.40 to 3.37 | 2.63 to 4.16 | 1.99 to 4.19 | 0.39 to 0.49 |
| grid 0.04 tonight (`onesky_grid004`) | 2.31 to 3.35 | 2.66 to 4.37 | 1.87 to 4.28 | 0.41 to 0.51 / 0.39 to 0.45 |
| grid 0.04 + unit integral inside 2 arcsec (`onesky_grid004_norm2`) | 1.20 to 1.47 | 1.58 to 2.09 | 1.05 to 2.30 | (all inside 2 arcsec by construction; 20% to 26% of it between 1 and 2 arcsec) |
| items 0.04 (`onesky_items004b`) | 1.06 to 1.16 | 1.15 to 1.34 | 0.92 to 1.03 | 0.95 to 0.97 / 0.90 |
| items 0.04, any-size rasteriser (`onesky_items004p`) | 1.03 to 1.15 | 1.08 to 1.31 | 0.92 to 1.02 | not measured |
| items 0.02, any-size rasteriser (`onesky_items002p`) | 0.96 to 1.03 | 0.97 to 1.11 | 0.91 to 1.02 | not measured |

- With the item sky the kernel's fraction inside 0.05 arcsec falls with wavelength as diffraction says (0.72 at F090W, 0.61 at F200W, 0.31 at F444W) and 0.83 to 0.86 (short) and 0.74 (long) lies inside 0.2 arcsec. With the grid sky the kernel is the same in all six filters and half of it lies beyond 2 arcsec.
- Reading: a sky of positive, compact items has no free diffuse level for the kernel's far wings to trade against, so the gauge is fixed by the form of the sky, with no added convention. Caveat: the item fits ran 1,850 to 2,327 steps; whether the drift appears later is not known.
- The flux convention on the kernel (`--kernel-norm-radius 2`) is a partial fix only: fluxes 1.2 to 1.5 times the mosaic at 1 arcsec, the plateau moved inside 2 arcsec, and prediction slightly worse than the plain grid (headline +0.143 [+0.099, +0.187]; +0.02 to +0.03 from 1 to 20). Not adopted.

**4. Speed of the item fits.** 5.2 s per step against 1.3 s for the grid. Measured on the GPU (`onesky_v2/gpu_bench`): the item scatter is bound by the number of small kernel launches (64,000 to 75,000 per pass), not by arithmetic, and the fit rasterises three times per step (two sky steps and one joint step). The any-size rasteriser gave no speed-up as run (5.3 s per step) but did no harm to the fit (same scores). Added: `--item-chunk` (larger chunks) and `--item-buckets coarse` (fewer distinct patch shapes).

**5. Round 3, launched 05:46 UTC** (package `fieldop-items-a542655b.tar.gz`; all with `--dq-ignore-bits 2`, any-size rasteriser with sheared patches, coarse buckets, 32 M chunks):
- `onesky_smacs_items004c_s0`: width cap 0.32 arcsec (isolates the speed and data-quality changes);
- `onesky_smacs_items004w_s0`: width cap 2.56 arcsec (can wide items carry the diffuse light and remove the faint-pixel inconsistency?);
- `onesky_smacs_items004ws_s0`: cap 2.56 arcsec plus a signed smooth grid at 0.64 arcsec.
- Check-in scheduled for 09:50 UTC.

**Repository.** Four commits on main in Seb's local clone `~/src/astroai/torchsky-harness` (dc087c1, 770ee83, a5f02b3, 7091887): the 12 Nov package as run on CANFAR, the item sky and tonight's changes with tests, the paired scorer, the small experiments. Not pushed: the sandbox has no GitHub key; `git push origin main` from the Mac does it.

## 14 Nov 2026, 01:20 to 02:20 UTC: item fits out of memory (fixed), any-size rasteriser, kernel check, flux convention

**Both item fits of the first batch failed with GPU out-of-memory** (10.75 GB slice). Cause: autograd kept every item patch (items x patch nodes x 12 coefficients), and the patches grew as items widened during the fit: 3.8 GB at step 0, 9.7 GB at step 700 at 0.04 arcsec; the 0.02 arcsec fit died at step 1. The two grid fits were not affected.
- Fix (mine): the backward pass of the item scatter recomputes patches in chunks instead of storing them (`_ItemRaster`). Values and gradients bit-identical to the old code (float32 and float64). The relaunched 0.04 arcsec item fit holds 3.3 GB. Package `fieldop-items-e752bce3.tar.gz`, run `onesky_smacs_items004b_s0` (method `onesky_items004b`), started 01:27 UTC.
- The item fit is slow: about 4 s per step against 1.2 s for the grid, so it gets about a third of the steps in the same 200 minutes. Equal wall time is the stated comparison; the step deficit is a handicap of the item form as implemented.

**Any-size rasteriser (Opus agent; checked by me).** Flag `--item-raster pyramid` (default `direct`, bit-identical), `--item-patch {square,sheared}`. Gaussian semigroup: an item wider than 2.8 nodes is point-sampled on a coarser grid (pitch 2^l p) with the cascade's blur removed from its covariance, and the levels are combined by fixed Gaussian upsampling steps. Cost per item no longer grows with its size.
- Accuracy: worst 2.1e-4 of the item peak against an exact point-sampled Gaussian (target 1e-3; the 1e-4 aim is missed); jump across a level switch at most 2.3e-4; flux exact to 2e-15; gradients against finite differences at most 7e-7. My own check against the analytic density (independent reference): worst 6.3e-4, at 0.6 nodes wide (the existing sub-node renormalisation, same as the direct path).
- The direct path truncates items wider than about 32 nodes (error 10% of the peak at 60 nodes in my check; 0.05 to 3.4 in the agent's sweep). The pyramid path does not.
- Cost, CPU, one thread, 60,000 items, forward + backward: round widened items 7.1 s to 2.3 s on 613^2, 23.9 s to 4.5 s on 1226^2. Elongated items gain only with sheared patches (16.1 s to 4.4 s; 53.5 s to 8.7 s).
- Not done: GPU timings; a real fit (whether the 2e-4 switch jumps disturb the optimiser). Report: `PYRAMID_REPORT.md`.

**Independent check of the learned kernels and of the fitted sky's flux scale (Opus agent; flux ratio re-measured by me).** State of `fieldop_smacs_fix_200_s0`.
- The earlier kernel claim holds (independent code): core width 0.021 to 0.023 arcsec in all six filters; 53% to 66% of the integral beyond +-32 pixels. The profile is the same in all six filters: a spike at the sampling scale (12% to 24% of the integral), a flat shoulder to 0.1 arcsec (12% to 15%), a 1/r halo from 0.2 to 2 arcsec (12% to 15%), and 51% to 60% beyond 2 arcsec, most of it a flat floor out to the 10 arcsec edge. No diffraction-like part.
- **The fitted sky is not flux-calibrated: it carries about twice the official mosaic's flux.** Fitted sky over official mosaic, same apertures: 2.7 to 4.1 for 20 compact sources (radius 0.3 arcsec), 1.7 to 2.2 for 8 bright galaxies (radius 1 arcsec), 1.9 to 2.9 for the box total. My own measurement on the brightest galaxy: 2.4 to 3.4 at 1 arcsec, 2.0 to 4.2 box total. The factor tracks 1 / (kernel fraction inside 2 arcsec).
- Through the whole model a star has a plausible core (predicted width 0.067 to 0.153 arcsec, official mosaic 0.063 to 0.169) and the right flux inside 0.3 to 1 arcsec (0.83 to 1.06 of the mosaic), but only 23% to 36% of its flux inside 0.3 arcsec (real NIRCam: about 80%).
- Cause: nothing pins the split between the kernel's far wings acting on the sky and the background terms. The unit integral is imposed over the whole 10 arcsec disc; the far floor times the sky is absorbed mainly by the per-detector 8x8 background pattern (only the cells under the 20 arcsec footprint see data) and the quadratic terms. Held-out scores are not affected (they are in data space); every sky-space product is.
- Report: `kernel_check/KERNEL_CHECK_REPORT.md`. Approximations there: nominal pixel scales, one exposure per filter for the star test, annulus backgrounds in a crowded field.

**Flux convention added (mine): `--kernel-norm-radius R`** (default 0 = unchanged; default path checked identical on the small synthetic run). With R > 0 the kernel's unit integral is taken inside R only; the kernel beyond R is free. The sky's flux scale is then set by the light inside R and the far-wing/background split can no longer move it. Stated bias: fluxes are "inside R" fluxes, low by the true fraction beyond R (NIRCam: about 1% to 3% at 2 arcsec). This is a convention, not a prior on the kernel's shape.

**Launched 02:17 UTC** (package `fieldop-items-2da1ecf7.tar.gz`, same arguments as the 1.75 reference run, 200 minutes):
- `onesky_smacs_grid004_norm2_s0` (method `onesky_grid004_norm2`): grid sky, 0.04 arcsec, `--kernel-norm-radius 2.0`. Questions: does the held-out score stay at 1.75, and does the sky flux come back to the mosaic's?
- `onesky_smacs_items004p_s0` and `onesky_smacs_items002p_s0`: 60,000 items at 0.04 and 0.02 arcsec with the any-size rasteriser and sheared patches.
- Still running: `onesky_smacs_grid004_s0`, `onesky_smacs_grid002_s0` (end about 03:30 UTC), `onesky_smacs_items004b_s0` (end about 04:50 UTC).

**Repository.** GitHub is not reachable from the cloud workspace. Seb granted the local clone `~/src/astroai/torchsky-harness` (main, origin = his fork). Found: the repository's `campaigns/continuous_sky/fieldop` was the 10-11 Nov version; the 12 Nov version that produced the reference scores (`fieldop-ce8616f0`, with tiles) had never been committed.

## 13-14 Nov 2026 (night): CANFAR access, three builds, four real-data fits launched, and a missing data folder

**Access.** Seb attached a CADC proxy certificate (valid to 13 Dec 2026). In the cloud workspace: `vls` works; `vcp` and `vmkdir` are refused, so reads go through a direct GET (`/home/claude/arc/arcget.sh`: `curl --cert ~/.ssl/cadcproxy.pem https://ws-uv.canfar.net/arc/files/<path>`), writes through `arcput.py`, new folders through a PUT of a ContainerNode to `https://ws-uv.canfar.net/arc/nodes/<path>`. `canfar` 1.5.2 needs `typer==0.27.3` (0.29 breaks `canfar create`).

**BLOCKER: the exposure folder is missing.** `/arc/projects/ots/torchsky-data` did not exist at about 22:55 UTC on 13 Nov (a listing returned "not found" before any job of this session ran). It held `jwst_smacs0723/cal`, `hst_smacs0723`, `aggregate` (LMC, 121 GB), `smacs0723_union` and the pixel store. The last run that read it finished on 12 Nov 10:47 UTC. A search two levels deep under `/arc/projects` and three under `/arc/home/sfabbro` found no copy. Not known whether it was moved or deleted; Seb to say. The folder now at that path was created by this session and holds only:
- `jwst_smacs0723/cal/`: the 135 SMACS NIRCam module B `_cal` files, re-fetched from MAST (15 GB), so the SMACS protocol works again;
- `smacs0723_raw/rate/` and `smacs0723_raw/uncal/`: `_rate` and `_uncal` for module B (the cluster field) and, by mistake, module A (the neighbouring field): 270 files each.
LMC work (and anything needing HST, Spitzer, the union or the pixel store) is blocked until the folder is found or re-fetched.

**Raw data facts (MAST).** LMC programme 1476 NIRCam imaging ramps are short: RAPID with 2 groups or BRIGHT1 with 5. SMACS programme 2736: MEDIUM8 with 8 groups. Ramp files are smaller than calibrated files here. For the ramp test on LMC the diversity is the same stars at two exposure lengths, not curvature inside a ramp.

**Build 1 (Opus agent): Gaussian items as the sky inside `fieldop`.** Flag `--sky-form {grid,items}`; tarball `scripts/fieldop-items-13917a37.tar.gz` (base: `fieldop-ce8616f0`). Items (position, two widths with a floor of 0.4 x sky pitch, angle, 12 positive spectral coefficients: 17 numbers each) are rasterised onto the same coefficient grid the renderers read, so this tests the parametrisation, not a grid-free renderer. Seeding on data-blind lattices, coefficients from the same warm-start coadd as the grid. With `--sky-form grid` the package gives identical numbers to the untouched one (checksums). Synthetic field (small, short runs, not converged), held-out score: 100 steps 4.04 against 4.51 (seed 0) and 2.31 against 5.17 (seed 1); 250 steps 1.52 against 2.52; with 25,500 against 108,300 fitted sky numbers. Caveat: read through the tent grid, an item-made star has second moment 1.15 pitch^2 against 0.67 for a grid node, so items should lose on the sharpest cores at 0.04 arcsec. Report: `ITEMS_REPORT.md` in the session archive.

**Build 2 (Sonnet agent): rendering test with the kernels learned on SMACS** (state of `fieldop_smacs_fix_200_s0`). No route reaches 1e-3 of the peak, because of the kernels themselves. NOT YET CHECKED INDEPENDENTLY:
- core width 0.023 to 0.024 arcsec in all six filters (0.77 px short channel, 0.38 px long channel): no growth with wavelength; the sharpest kernel sits at the blur floor (M_ref 545e-6 against floor 533e-6 arcsec^2 in the run log);
- 52% to 66% of each kernel's integral lies beyond +-32 px; 35% to 40% beyond +-128 px (4 arcsec) in the short channel; encircled energy at 1 px is 0.19 to 0.30;
- the variability across the detector and with wavelength was pruned (scales 1e-6): one kernel everywhere;
- the kernel peak is off the origin by 0.06 to 0.25 px.
Reading: a tail near 1e-5 of the peak over 10 arcsec outweighs a point-like core and trades against the background: the gauge "far kernel wings against background" is not pinned in `fieldop`. To check: fitted sky fluxes against the official mosaic (a common factor of 3 to 5 is possible).

**Build 3 (Sonnet agent): paired and by-scale scoring**, `protocol/v1/code/score_paired.py`, launcher `scripts/onesky/run_score_paired.sh`, output `protocol/v1/scores_paired/<FIELD>/`. Reproduces `score.py` exactly. Full SMACS result (24 held-out exposures, 12 dither groups), field operator minus official pipeline, mean squared residual in noise units on identical pixels (negative = field operator better):

| Quantity | `fieldop_fix_200` minus `stage3_blot` | 68% interval |
|---|---|---|
| Headline, signal-to-noise >= 5 (1.74 against 1.79) | -0.055 | [-0.226, +0.081], P(better) 0.64 |
| Signal-to-noise 1 to 5 | +0.032 | [0.006, 0.070] |
| 5 to 20 | +0.126 | [0.095, 0.160] |
| 20 to 50 | +0.155 | [0.102, 0.198] |
| 50 to 200 | -3.08 | [-5.13, -1.64] |
| above 200 | +12.5 | [-3.9, +31.1] |
| F090W | -1.47 | [-3.23, -0.07] |
| F277W | +0.238 | [0.190, 0.286] |

The headline is unresolved and is a cancellation: the pipeline is better from signal-to-noise 1 to 50, the field operator from 50 to 200 and in F090W. The paired interval is 0.31 wide against 0.45 for two separate intervals.

**Running since about 00:06 UTC, 14 Nov** (CANFAR sessions `oneskyg004`, `oneskyi004`, `oneskyg002`, `oneskyi002`, `oneskysmoke`; run folders in `/arc/projects/ots/adass2026/fieldop/`; all with `fieldop-items-13917a37.tar.gz`, the arguments of the 1.75 reference run, 200-minute fits):

| Run | Method name in the protocol | Sky |
|---|---|---|
| `onesky_smacs_grid004_s0` | `onesky_grid004` | grid, 0.04 arcsec |
| `onesky_smacs_items004_s0` | `onesky_items004` | 60,000 items (`--item-levels 3 --item-max-width 0.32`), 0.04 arcsec |
| `onesky_smacs_grid002_s0` | `onesky_grid002` | grid, 0.02 arcsec |
| `onesky_smacs_items002_s0` | `onesky_items002` | 60,000 items, 0.02 arcsec |
| `onesky_smoke_items_s0` | `onesky_smoke_items` | 4-minute smoke test of the item path |

At step 0 the items start at a higher batch score (18 to 21 against 4 to 5): their seed is blurrier. The smoke test ran at about 3.8 s per step against about 1.2 s for the grid reference: to confirm on the long runs. When they finish: paired scoring with `REF=stage3_blot` and `METHODS="onesky_grid004 onesky_items004 onesky_grid002 onesky_items002 fieldop_fix_200"`.

**Session files.** `arc:projects/ots/adass2026/onesky_v2/one-sky-v2-experiments-e1-e2-20261113.tar.gz` (the two first tests). The second round's code (`fieldop_items/`, `e2_learned/`, `score_paired/`) is in the cloud workspace and, for the parts that run, on /arc as listed above.

## 13 Nov 2026: E1 (any-scale representations) and E2 (rendering through a kernel), small scale

**Where it ran.** The cloud workspace: 2 CPU cores, no GPU, torch 2.15 CPU. Three agents in parallel: Sonnet (baselines), Opus (learned profile), Sonnet (rendering). Each fit capped near 220 s. Code and results: the file `one-sky-v2-experiments-e1-e2.tar.gz` sent to Seb in the chat on 13 Nov (the workspace is ephemeral; it is not on /arc yet). Layout: `README.md` (rules), `e1/common.py` (synthetic sky and scoring, written by the orchestrating session), `e1/baselines.py`, `e1/profiles.py`, `e1/summarise.py`, `e1/*_REPORT.md`, `e2/render_test.py`, `e2/E2_REPORT.md`, `results/` and `figs/` in each.

**Check done.** All 45 saved model images were re-scored against a freshly rendered truth: no mismatch with the stored scores. No second agent has reviewed the code.

### E1: setting

- Synthetic sky from `common.make_spec(0)`, 256 x 256 pixels: stars (exact Gaussians of width 0.45 px), galaxies (Sérsic discs and bulges, spiral arms, clumps, sizes 1 to 60 px), diffuse light (8 to 512 px). Noise: sqrt(truth + 1). One sky, one noise draw; sky 10 for the transfer test.
- Every method fits the noisy image with weights 1/sigma^2. Score: mean squared error against the truth in noise units (`chi2_truth`); the noisy image itself scores 0.992.
- Budgets: 5,000, 15,000 and 45,000 fitted numbers for 65,536 pixels.

### E1: results (sky 0)

| Representation | Noisy, 5,000 numbers | Noise-free, 15,000 numbers |
|---|---|---|
| Pixel grid, bicubic | 2.79 | 2.17 |
| Hash-grid neural field (12 levels, 2 features, 2 x 32 network) | 1.13 | 0.007 |
| Gaussian items, baseline code (lattice / adaptive seeding) | 0.86 / 0.84 | 0.010 / 0.004 |
| Gaussian items, second code (inside profiles.py) | 0.48 / 0.43 | 0.005 / 0.004 |
| Learned-profile items (4 latent numbers per item, 1,152 shared weights) | 1.10 / 1.14 | 0.062 / 0.021 |

- Transfer: profile trained jointly on skies 0, 1, 2 and frozen, items fitted on sky 10: 1.07 / 1.03 at 5,000 numbers against 0.12 / 0.13 for Gaussians; 0.250 against 0.232 at 15,000; 0.474 against 0.498 at 45,000.
- Scale covariance (frozen profile, sky 10 at magnifications 1, 1.4, 2, 2.8): scores 0.250, 0.209, 0.203, 0.225. Matched items: log size shifts by 0.25, 0.74, 0.87 against log(zoom) 0.34, 0.69, 1.03. Latent distance between matched items over that between random pairs: 1.08, 1.17, 1.21, i.e. no better than random.
- At equal item count on a training sky (not budget-matched), the frozen profile ties the Gaussian (0.0040 against 0.0040 to 0.0051).

### E1: reading

- Robust: items with continuous size beat a fixed pixel grid by a factor 3 (noisy) to several hundred (noise-free).
- Gaussian items against the hash-grid field: Gaussians better in noise-weighted error; the field better in the 1 to 2 pixel band on noise-free data (0.009 against 0.04 to 0.10) and in peak signal-to-noise ratio (66.5 against 55 to 61 dB). The two Gaussian codes differ by a factor 2 at 5,000 numbers: that is the implementation scatter.
- Learned profile: worse than Gaussians by a factor 2 to 10 in the two informative settings. By the drop rule of the plan (less than a factor 2 in size or 2 dB over the best alternative) the element is dropped from the measured sky in this form.
- Structural point: splitting light into many small overlapping items is not unique, so per-item latents are not identifiable, and the shape of each item hardly matters. A learned profile can only pay when one item is one coherent structure.

### E1: caveats

- Design flaw of the test: at 15,000 and 45,000 numbers fitted to noisy data, every method sits near the noise-fitting floor (numbers / pixels = 0.23 and 0.69); those rows do not separate methods.
- The learned profile's renderer is inaccurate: up to 26% of the peak at widths near 3 px (point sub-sampling of a bilinear table). Not fixed.
- Stars in the synthetic sky are exact Gaussians, which favours Gaussian items.
- The learned profile got 135 to 272 optimiser steps against 355 to 1,150 for Gaussians (wall-clock cap under load). The 45,000 runs are under-converged for all item methods.
- Baseline Gaussians were capped at a width of 7.8 px for compute.
- One sky, one noise draw, no error bars.

### E2: setting

- Stand-in kernels from an analytic hexagonal pupil (obscuration 0.11 D, three struts of 0.02 D), |FFT|^2, 1 x 1 pixel box, 0.2 px Gaussian. "Sharp": 1.15 px wide; "wide": 2.11 px. Support +-32 px.
- Profiles: point source; Sérsic index 1 and 4 with half-light radius 0.5 to 32 px; an irregular spiral with clumps. Three sub-pixel positions each.
- Reference: 16 sub-samples per pixel, Richardson-extrapolated with 8; agrees with an independent scheme to 3e-5.
- Target: maximum error below 1e-3 of the peak pixel.

### E2: results

| Case | Maximum error / peak (sharp ; wide) |
|---|---|
| Kernel fitted by positive Gaussians on a pyramid (50 ; 91 Gaussians) | 3.2e-3 ; 1.2e-2. Spike signal captured: 0.44 ; 0.57 |
| Same pyramid, signed weights | 9.7e-4 ; 2.6e-3 |
| Closed form, 16 Gaussians per profile, Sérsic 1, radius 2 px | 1.7e-3 ; 3.5e-3 |
| Closed form, 16 Gaussians, irregular profile | 0.19 ; 0.06 |
| Local sampling, pixel box inside the kernel, 4 sub-samples, Sérsic 1, radius 32 px | 1.3e-4 ; 6.6e-5 |
| Local sampling, pixel box inside the kernel, 6 sub-samples, Sérsic 1, radius 2 px | 4.6e-4 ; 2.6e-4 |
| Local sampling, 6 sub-samples, Sérsic 4, radius 2 px | 5.5e-2 ; 3.8e-2 |
| Local sampling, point source snapped to the grid | 0.07 to 0.27 ; 0.03 to 0.13 |
| Kernel cut at +-16 px, Sérsic 1, radius 32 px | 2.7e-3 ; 5.3e-3 (flux lost 1.1% ; 2.3%) |

Speed, one thread, unbatched Python: 170 to 1,150 s per 2048 x 2048 image with 10,000 small and 100 large items.

### E2: reading

- A kernel made of positive Gaussians cannot reach the target; positivity is the limit, not the stencil. Signed weights nearly reach it for the sharp kernel.
- Local sampling works for smooth extended light with the pixel box folded into the kernel. Cusps and point sources need the kernel evaluated at the exact offset.
- Far kernel wings cannot be cut.
- Renderer to build: signed-weight kernel pyramid, exact-offset evaluation for compact light, sampling for extended light, batched on a GPU.
- To repeat with the learned NIRCam kernels on /arc.

### Effect on the design

| Element of version 2 | Status |
|---|---|
| Items with continuous position, size and orientation | Kept |
| Learned profile per item | Dropped from the measured sky by the rule; one fair retest on real mosaics is optional (Seb to decide) |
| Kernel as positive Gaussian blobs | Dropped: signed pyramid plus exact-offset evaluation |
| Raw inputs, calibration frames, ensemble predictor | Untested, unchanged |

### Seb's steer of 13 Nov (evening)

- Version 2 is ultraviolet to infrared. Radio and X-ray wait for a version 3.
- Fields: SMACS and LMC, then COSMOS added.
- Telescopes one at a time: JWST, HST, Spitzer, GALEX, CFHT, Subaru, VISTA. Note: CFHT and Subaru cannot observe SMACS or the LMC, so ground data arrive with COSMOS unless VISTA goes first.
- Use subagents: Sonnet for well-specified builds, Opus for new model code and review.

### Blocked

- E3 (flat from `_rate` images), E4 (ramps) and the real-data repeats of E1 and E2 need CANFAR. The cloud workspace of 13 Nov had no CADC proxy certificate (`~/.ssl/cadcproxy.pem`); see the setup notes in `claude/continuous-sky-handoff.md`.
