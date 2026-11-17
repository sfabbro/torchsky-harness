# Handoff: one sky, many instruments — field operator track (10 Nov 2026)

Start here in a new session. This supersedes the "Latest" sections of `claude/continuous-sky-handoff.md`; that file still holds the setup notes (CANFAR, certificates, conventions) and the older history.

## Read first

0. **Proposal of 13 Nov (version 2), requested by Seb, small experiments first:** `claude/one-sky-v2-raw-anyscale-ensemble.md` ([living doc, tab "Version 2"](https://claude.ai/artifact/DYP9UJuvosgpaz7BGG4yr5#bcefa2e8-f671)).
   - Three changes from version 1: raw inputs (ramps, raw frames, calibration frames as exposures of trivial scenes); a sky of items with continuous position, size and orientation and a learned profile; a shared ensemble predictor across scales, scored by CRPS on raw held-out pixels and never imposed on the measured sky.
   - Six small tests, E1 to E6, each with a rule for dropping its element. E1 (any-scale profiles), E2 (rendering) and E3 (does the flat calibrate itself from `_rate` images) come first. Nothing has been run.
   - Seb's steer: as raw as possible; any scale, not fixed levels; as free of priors as feasible and fit for unknown unknowns; calibration frames may be included.
   - **Data folder (resolved 16 Nov):** `/arc/projects/ots/torchsky-data` was deleted by mistake by the cleaning team (Seb). Re-fetch started 16 Nov 23:14 UTC with the surviving manifests: LMC NIRCam `_cal` (1,032 files), HST SMACS (116), the SMACS union (JWST, GALEX, SPHEREx, Spitzer). The SMACS NIRCam `_cal`, `_rate` and `_uncal` files were re-fetched on 13 Nov. The re-fetched files are a newer calibration than the protocol masks: use `--dq-ignore-bits 2`. State in the log.
   - **Results so far:** `claude/one-sky-v2-experiments-log.md`. E1 and E2 ran on 13 Nov at small scale: the learned per-item profile did not beat Gaussian items (dropped by the rule; items with continuous size kept), and a kernel of positive Gaussians misses 1e-3 (signed pyramid plus exact-offset evaluation instead). E3 and E4 are blocked on a CADC certificate in the cloud workspace.
   - **14 Nov, key findings** (details: `claude/one-sky-v2-experiments-log.md`, entries of 14 Nov; living doc section "Third round"):
     - The reference grid sky (`fieldop_smacs_fix_200_s0`) carries 2.3 to 3.4 times the official mosaic's flux and half of its kernel lies beyond 2 arcsec: the split between far kernel wings and background terms is unpinned. Held-out scores are unaffected; sky-space products are.
     - A sky of 60,000 positive Gaussian items (`fieldop --sky-form items`) comes out flux-calibrated (1.06 to 1.16; 0.96 to 1.03 at a 0.02 arcsec raster) with a diffraction-like kernel, and predicts held-out pixels better than the grid sky and the official pipeline from signal-to-noise 1 to 50 when the full prediction is scored. It is worse on bright cores (4 times fewer steps in equal time) and inconsistent at the faintest pixels (the gain disappears with the scorer's offset). Three follow-up fits were launched at 05:46 UTC (wider items, smooth grid).
     - The SMACS `_cal` files re-fetched on 13 Nov are a newer calibration than the protocol's: they flag the brightest cores SATURATED where the masks say valid. Use `score_paired.py --dq-zero` for fits made with DQ == 0 on the new files, or fit with `--dq-ignore-bits 2`.
     - **16 Nov:** with wide items allowed (`--item-max-width 2.56`) the item sky is level with or ahead of the official pipeline in every signal-to-noise bin (headline -0.133 [-0.270, -0.005], full prediction scored) and its fluxes agree with the mosaic to 2%. The "faint-pixel inconsistency" was an artefact of the protocol's offset (its pixels are chosen from a mosaic that contains the held-out exposure): use the no-offset scores or `score_paired.py --offset-rank stage3_blot --offset-rank-smooth 1`. Use `--dq-ignore-bits 2` for all new fits (the grid then scores 1.73 on the protocol's own mask). Raw-input step built, not yet run on real data: `--input-level rate` with `--pixel-gain detector` (package `fieldop-gain-a7950e8c.tar.gz`, `RATE_GAIN_REPORT.md`).
     - **17 Nov, 02:30 UTC:** (a) item relocation (faint items moved to positive residual) is worse than none: headline +0.286 [+0.087, +0.474]; dropped. (b) Count-rate input with a learned gain per detector pixel, grid sky (`fieldop_rate_gain_grid`): fine on training exposures, held-out headline 69.6 against 1.95 for the calibrated twin at count-rate level; the gain map has 3% to 6% large-scale structure (official flat 0.5% to 1.6%). Reading (not established): gain trades against the scene with one pointing and nine dithers. Dropped on SMACS; the control without gain (`fieldop_rate_grid`) ends about 05:45 UTC; the planned form of the test is LMC programme 1476. Do not run rate + gain + items on SMACS. (c) Four pieces are now in torchsky itself (commit `3df2abe5` on main of Seb's clone, unpushed): `GaussianItemScene`, `TelescopeFrameDetectorToSky`, `field_observation_from_exposure_file`, `NestedKernelPSF`; next build (Sonnet): make the harness prototype use them. Open questions for Seb: `claude/torchsky-upstream-notes.md`. (d) Re-fetch at 02:25 UTC: HST SMACS, Spitzer, GALEX complete; SPHEREx 28 of 718 cut-outs fail with HTTP 422; LMC calibrated 321 of 1,032 and count-rate 279 of 1,032; JWST union in progress. (e) Only one healthy GPU slice: launch with `/home/claude/work/launch_gpu.sh`, one job per call.
     - **17 Nov, 11:25 UTC, round of debugging, performance and simplification (Seb's request; log entry "07:45 to 11:25 UTC"):** audit report `AUDIT_2026-11-17.md` (15 ranked findings, the interface a second instrument must supply, two bugs fixed, end-of-fit sizes of all learned terms now reported); a converging truth problem in `tests/truth/` (response frozen at the truth; a learned response diverges there, not diagnosed); with today's defaults the empty sky is biased high by an amount that does not fall with more exposures (background tied to data medians plus positivity); new options, off by default: `--sky-zero masked --sky-positivity-margin 3` (bias removed within errors, faint fluxes to 5% to 7%, but error still 1.25 to 1.9 times the ideal coadd) and `--rate-units sky --nuisance-steps exposure` (count-rate fits in MJy/sr; use for every count-rate fit); performance report `PERF_SIMPLIFY_2026-11-17.md` (half a step is the backward pass through the renderer; step time independent of the number of exposures; device reads per step cut from 28 to 6; benchmark `tests/bench_step.py` to run on a GPU). Six decisions are waiting for Seb (log section 5). Current archive: `fieldop-tsky-67a98fa5.tar.gz`; 20 GPU launches have failed since 05:45 (faulty node); the launch line for the count-rate fit is in the scheduled check-in E.
     - **17 Nov, 07:15 UTC (corrects item (b) of 02:30 below):** the count-rate fits failed because the starting sky was in DN/s per pixel while the model works per square arcsecond (sources 250 to 1,000 times too faint; the learned noise floor rose to 1,000% and hid it). The control without gain failed the same way, so nothing was learned about the gain map. Fixed in `fieldop_tsky` (`RATE_HELDOUT_DIAGNOSIS.md`); new archive `fieldop-tsky-9538c031.tar.gz` (also prints the score against input noise and the learned noise floor at every logged step, and warns above 0.2) is on CANFAR and `/home/claude/work/launch_gpu.sh` now reads `/home/claude/work/CURRENT_SCRIPT` and takes `FIELD` from the environment. To run when a healthy GPU slice frees (none since 05:45; every launch lands on the node with the NVML fault): `fieldop_rate_gain_grid2`, `fieldop_rate_grid2`, then the LMC item sky `onesky_lmc_items002w_s0` (arguments in the log entry of 06:00 to 07:15). After every fit check the learned noise floor (a few per cent is normal). Re-fetch is complete (LMC 1,032 calibrated and 1,032 count-rate files).
     - **17 Nov, 04:00 UTC:** `/home/claude/work/fieldop_tsky/` (committed, `0c4d62d` and `ee6221a`) is the prototype with optional torchsky backends, all off by default: `--item-impl torchsky` and `--loader torchfits` adopted (equivalent to 1e-7 and identical); `--rate-wcs telescope` plus `--filter-shift learned` gives count-rate positions without a calibrated file (0.6 to 2.6 mas rms on files with one shift per detector and filter; learning the shifts from zero in a short synthetic fit is NOT shown; start them from a table with `--filter-shift-init`); `--render-impl torchsky` not wired. The package used on CANFAR is still `fieldop_gain` (archive `322a22f9`); before switching, build a new archive from `fieldop_tsky`, a new torchsky source archive from `/home/claude/torchsky`, check torchfits and pysiaf exist in the job image, and make one real-data smoke run. Report: `campaigns/continuous_sky/TORCHSKY_INTEGRATION_REPORT.md`.
     - Code: commits on main in Seb's local clones `~/src/astroai/torchsky-harness` (11 or more ahead of origin) and `~/src/astroai/torchsky` (1 ahead); not pushed: no GitHub key in the sandbox, so `git push origin main` has to come from Seb.
   - Seb's order of work (13 Nov): ultraviolet to infrared only (radio and X-ray in a version 3); SMACS and LMC, then COSMOS; telescopes one at a time: JWST, HST, Spitzer, GALEX, CFHT, Subaru, VISTA; Sonnet agents for builds, Opus for new model code and review.
   - Version 1 (12 Nov, the "photon mixture" of Gaussian blobs): `claude/one-sky-at-every-scale-plan.md`. Superseded, but its review of `claude/field-operator-formulation.md` (eight findings), gauge table, likelihood and kernel pyramid still stand. Seb's decisions of 12 Nov: scope is imagers, spectrographs and photon-counting instruments (no radio, no polarisation); COSMOS after the SMACS 8-arcminute union.
1. **Plan (living doc):** [Baselines, general field operator and ablation plan](https://claude.ai/artifact/5u2o5CeELJcPMQ1VpcLGPj).
   - Phases: 0 evaluation protocol → 1 published baselines → 2 general field operator → 3 seventeen ablations → 4 scaling with data diversity.
   - Seb approved it on 10 Nov.
2. **Write-up:** [Field operator: mathematical formulation and groundings](https://claude.ai/artifact/Tb4B5qdZHFo1zyQouzZcB3).
   - The equations as implemented, the gauges, the variability scales, the rendering and its errors, groundings in published practice, the evidence and the open problems.
3. **Run log:** `claude/continuous-sky-runlog.md`, newest first.
4. **Older formulation doc:** [One sky, many instruments: formulation](https://claude.ai/artifact/Nj91AWRxgJf94mZyifAc56). It holds Seb's ticked decisions:
   - grid plus splats for the sky;
   - one shared trunk across instruments;
   - the blur gauge as "sky at the resolution of the sharpest kernel".

## Seb's direction (keep to it)

- **Bitter lesson.** Learned and data-driven. Physics only as inputs or a warm start, never as constraints. He has never seen sim-to-real work in observational astrophysics.
- **Break degeneracies with diversity.** Multi-instrument data, data splitting and strong empirical tests, not physics.
- **Catalogue-free model.** Catalogues are allowed for evaluation metrics only.
- **Baselines must be the published, standard methods.** Never a home-made stand-in.
- **The non-sky part is ONE field operator.**
  - No imposed basis: Seb finds Alard's Gaussian × polynomial basis too rigid.
  - Transmission is inside the kernel.
  - Shape normalised, amplitude structured: amplitude = pixel area from the geometry × learned throughput.
- **Communication.**
  - Plain names, no unexplained acronyms or run codes.
  - State every approximation and shortcut.
  - Concise options first.
  - Canadian English.
- **Workflow.**
  - CANFAR headless jobs use `images.canfar.net/astroai/base:latest`. No `~/.local` installs.
  - Git only on the sfabbro fork main, then a PR to astroai; no tags, releases or pushes without Seb's OK.
  - The cloud session has no GitHub credentials.

## State at handoff

| Phase | Item | State |
|---|---|---|
| 0 Protocol | Splits, official alignment, reference mosaics, held-out masks, noise factors, scorer | **Done for SMACS and LMC.** |
| 1 Baselines | Official JWST stage-3 pipeline + blot (`stage3_blot`) | **SMACS scored: 1.80 [1.61, 2.02].** **LMC scored: 6.14 [5.25, 7.22]** (F090W 28.1, the badly aligned 1476 observation 001; other filters 3.6–16). HST WFC3/IR AstroDrizzle + blot ran with a provisional split. |
| 1 Baselines | DOLPHOT (NIRCam module) | **Blocked.** Web access to americano.dolphinsim.com and arxiv.org needs Seb's approval in the app. |
| 1 Baselines | Empirical PSFs (Anderson-style), galaxy fitter (ForcePho or The Tractor), PIFF | Not started. |
| 2 Field operator | `fieldop` package v0.1 | **SMACS 1.77 [1.68, 1.86]**, a tie with the pipeline. Synthetic recovery partly works (write-up, "Evidence"). Latest code (`fieldop_v5`, 150-min fit) scored 1.94 [1.86, 2.05], worse than the 3.3 h run; rerun at matched length to tell whether the cause is fit length or the new scale rule. |
| 3, 4 | Ablations, scaling | Not started. |
| Data | LMC all filters (1476, 4447), NIRISS LMC (1501, 4471), M92 (1334) | **Downloaded** (2164 files) to `/arc/projects/ots/torchsky-data/aggregate/<prog>_<instrument>/`. |

Score definition: mean (data − model)²/σ² on held-out dither exposures, on pixels where the official mosaic has S/N ≥ 5, with σ = pipeline error × per-detector noise factor, and a 68% bootstrap over dither groups.

## To finish this round (minimum viable)

1. Done: LMC protocol and LMC pipeline score (6.14).
2. Check why F090W scores 28 on LMC and what drives the fractional residual of 1.47 at S/N > 50 (saturated cores?).
3. **Field operator:** rerun the latest code at matched length (≥ 3.3 h) on SMACS.

## Next steps (in order)

1. **Phase 2 fixes** (write-up, "Open problems"):
   - speed (1.3 s/step; CUDA graphs or larger batched FFTs);
   - port the multi-tile loop from `opjoint.py` and run LMC tiles;
   - HST loading;
   - a refitted likelihood-ratio or Laplace evidence for the variability scales;
   - the `_rate` input conversion.
2. **Phase 1:** DOLPHOT once web access is approved, then empirical PSFs, a galaxy fitter on SMACS and PIFF.
3. **Phase 3:** the 17 ablations of the plan, each in both directions with 3 seeds.
4. **Phase 4:** the scaling experiment.

## Where things live

- **Code (cloud workspace, ephemeral; copies on /arc):**
  - `/home/claude/work/fieldop/` (+ `DESIGN.md`)
  - `/home/claude/work/protocol/` (`INTERFACE.md`, `make_splits.py`, `official_align.py`, `make_heldout_masks.py`, `score.py`)
  - `/home/claude/work/baselines/` (`stage3_blot.py`, `stage3_blot_hst.py`)
  - `/home/claude/work/tests/` (`test_fieldop_*`, `test_register_xcorr.py`, …)
  - `/home/claude/work/scripts/` (`register_xcorr.py`, `tilefit.py`, `opfield.py`, `opjoint.py`, smacs_four v15–v40)
- **CANFAR:**
  - scripts: `/arc/projects/ots/adass2026/scripts/` (`run_fieldop.sh`, `fieldop-<hash>.tar.gz`, `run_protocol_masks.sh`, `run_official_align.sh`, `run_score.sh`, `run_stage3_blot.sh`, `run_register_xcorr.sh`)
  - protocol: `/arc/projects/ots/adass2026/protocol/v1/` (splits, align, reference mosaics, heldout, pred/<FIELD>/<METHOD>/<SPLIT>/, scores/)
  - LMC registration: `/arc/projects/ots/adass2026/p1/register_xcorr_v2_r3/`
  - CRDS cache: `/arc/projects/ots/adass2026/crds_cache`
- **Bundle of everything:** `/arc/projects/ots/adass2026/continuous-sky-<date>.tar.gz` (newest).

## Lessons worth keeping

- **Registration came first.** Header WCS errors up to 1.8″ in programme 1476 broke the first LMC experiment. A global, catalogue-free cross-correlation solve with a ±3″ search and cross-filter links fixed it; it agrees with the official tweakreg to 2.6 mas.
- **Per-run masks made scores incomparable.** Each method used to score on its own masks, so the same coadd scored 2.93 or 3.25 on SMACS. One shared protocol is essential.
- **Two gauges need a condition each.** The common blur and the sky pedestal must each be pinned by one declared condition. Otherwise held-out error rises with fit length, because the sky goes sharper than its grid.
- **Raw χ² does not compare across fields.** It scales with S/N squared (a 5% model error at S/N 100 gives 25). Use the fractional residual for that.
- **Earlier operators failed in transfer and in joint fits.** The previous field operator (`opfield`/`opjoint`, fixed stencil and per-filter tables) lost to the coadd and transferred negatively. The registered per-detector PSF tables transferred frozen from LMC to SMACS at 2.03 against 2.20, but they are not a published method and are no longer pursued.
