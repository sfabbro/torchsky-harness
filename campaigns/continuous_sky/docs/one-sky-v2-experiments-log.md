# One sky, version 2: log of the small experiments (newest first)

Plan and design: `claude/one-sky-v2-raw-anyscale-ensemble.md` (living doc, tab "Version 2": https://claude.ai/artifact/DYP9UJuvosgpaz7BGG4yr5#bcefa2e8-f671).

## 18 Nov 2026, 10:10 UTC: first joint JWST + HST fit: one sky fits both, fluxes right in all six JWST bands, held-out JWST prediction unchanged

Fit `onesky_smacs_jwsthst_items_s0`: item sky, 85 JWST and 60 HST WFC3/IR training exposures (plus 28 validation), 200 minutes, 3,133 steps, new defaults, kernel normalised within 2 arcsec, HST positions started from the archive solution (shortcut, see 06:15). Collection by a Haiku agent; flux check by me.

**Held-out JWST exposures (cross-fitted, no scorer offset), against the JWST-only item fit with the same defaults (`onesky_items004w_v2`):**

| Signal-to-noise | JWST only | JWST + HST | Difference |
|---|---|---|---|
| headline (5 and above) | 2.026 | 2.063 | +0.037 [-0.060, +0.137] |
| 1 to 5 | 0.977 | 0.974 | -0.003 [-0.006, +0.001] |
| 5 to 20 | 1.089 | 1.094 | +0.006 [+0.002, +0.011] |
| 20 to 50 | 1.161 | 1.291 | +0.130 [+0.096, +0.161] |
| 50 to 200 | 8.90 | 10.23 | +1.33 [+0.64, +2.08] |
| above 200 | 138 | 108 | -30 [-67, -4] |

- Adding 74 coarser HST exposures leaves the prediction of held-out JWST exposures unchanged in the headline: slightly worse at 20 to 200, better above 200. This is the expected first result: HST adds wavelength coverage between F090W and F150W, which JWST held-out exposures cannot reward. The test that can is to hold out HST exposures (or a JWST filter) and predict them; not yet set up (the protocol's masks and scorer are JWST only).
- **The two instruments agree on one sky to 5%:** the learned mean log-throughput of WFC3/IR relative to NIRCam is -0.051 (the reference instrument is fixed at 0). The files-only check had given F160W over F150W of 0.92 to 0.94.
- WFC3/IR learned noise scales 0.81 to 0.91, matching the files-only finding that its error planes are 10% to 20% too large. Its noise floor ended at 8% (it sat at the 20% bound early in the fit). NIRCam noise scales 0.96 to 1.09.
- Translations learned on top of the archive positions: 5.3 mas rms, 23 mas at most.
- **Sky flux against the official mosaic at 1 arcsec: 0.98 to 1.00 in all six JWST bands** (the JWST-only fit with the same defaults: 0.98 to 1.00 short-wavelength, 1.15 to 1.17 long-wavelength). The difference between the two fits is the 2 arcsec kernel normalisation, which the joint fit gets by default: less than 0.1% of each kernel lies beyond 2.5 arcsec, where the JWST-only fits put 46% to 56%. So that normalisation pins the flux scale for the item sky, and should be tried as the default for single-instrument fits too (it was only a partial fix for the grid sky on 14 Nov).
- The zero-level rule was active for 130 to 132 of 145 training exposures.
- A fault in my flux tool, caught before reporting: it took the first six filters of the state as the six JWST filters; with ten filters ordered by wavelength that compared the model's F105W with the mosaic's F150W and so on (ratios 0.55 to 1.53). It now selects filters by pivot wavelength.
- Caveat: the JWST-only reference is itself worse on bright pixels than the old defaults (ablations running); read the comparison per bin.

**Also at 09:50:** the third ablation had landed on node g0125 at 9.3 s per step (2.7 times slow); deleted and relaunched on g0123 (`onesky_smacs_items004w_v2_nobound_s1`), so it ends about 13:25.

## 18 Nov 2026, 09:20 UTC: the new defaults on real data: better for the grid sky, worse for the item sky on bright pixels; three ablations launched

Collection by a Haiku agent; flux checks by me. Both fits: SMACS, 200 minutes, package `fieldop-tsky-90ff3b37`, cross-fitted held-out scoring, no scorer offset. The comparison at equal scoring is with the re-scored old fits.

| Signal-to-noise | Pipeline | Grid, old defaults | Grid, new defaults | Items, old defaults | Items, new defaults |
|---|---|---|---|---|---|
| headline (5 and above) | 1.822 | 1.768 | 1.692 | 1.733 | 2.026 |
| 1 to 5 | 1.094 | 1.098 | 1.146 | 0.976 | 0.977 |
| 5 to 20 | 1.174 | 1.259 | 1.308 | 1.092 | 1.089 |
| 20 to 50 | 1.237 | 1.325 | 1.349 | 1.169 | 1.161 |
| 50 to 200 | 6.70 | 3.35 | 2.86 | 6.87 | 8.90 |
| above 200 | 92 | 127 | 97 | 83 | 138 |

- **Grid sky: the new defaults help**, -0.076 [-0.090, -0.063] in the headline, all of it on bright pixels (50 to 200: -0.49; above 200: -30); slightly worse at 1 to 50 (+0.02 to +0.05). 9,708 steps. Final noise floor 4.4% and 1.6%; noise scales 0.71 to 0.91 (two long-wavelength scales sat at the bound of 2 early on, not at the end). The zero-level rule fell back to the old median from the second refresh on, so for the grid this is the effect of the other defaults.
- **Item sky: the new defaults hurt**, +0.292 [+0.148, +0.435] in the headline, all of it on bright pixels (50 to 200: +2.0 [+1.1, +3.2]; above 200: +56); identical below 50. Against the pipeline the new item fit is now worse in the headline (+0.20 [+0.01, +0.39]) while still better at 1 to 50.
  - Things that differ from the old item fit and could cause it, not yet separated: 3,131 steps instead of about 4,400 (3.83 s per step against 2.7; the zero-level refresh takes about 10 minutes of the 200, the rest is unexplained); the spatial variation of the kernel was kept this time (its scale ended at 1.5, where every earlier fit pruned it); final noise floor 9.0% and 4.4% (not reported for the old fit); the zero-level rule itself (active here: 75 to 86 of 86 exposures re-referenced at each refresh).
  - Flux of the new item sky against the official mosaic at 1 arcsec: 0.98 to 1.00 in the three short-wavelength filters, but 1.15 to 1.17 in the three long-wavelength ones (the old item fit: 0.98 to 1.00 in all six). So the long-wavelength amplitude is off by 16%, which fits a bright-pixel loss.
  - Its sky in empty regions is at 0.002 to 0.004 MJy/sr (the mosaic's own background level there is 0.012 to 0.037): the zero convention does what it says.
- **So the defaults approved on synthetic evidence are not a clear gain on real data.** Three 200-minute ablations of the item fit were launched at 09:00 to 09:10 on the same package to find which default does it: (1) all old defaults with cross-fitted scoring (`onesky_items004w_legacy`; should reproduce 1.733 if nothing else in the package changed); (2) new defaults with the old zero-level rule (`onesky_items004w_v2_zmed`); (3) new defaults without the noise bounds (`onesky_items004w_v2_nobound`). Until they are read, I treat `--sky-zero masked` and the noise bounds as provisional for the item sky.

### Builds finished while the fits ran (Sonnet, `BUILD_2026-11-18B.md`; not yet in the archive used by the running fits)

- Zero-level refresh on a subset of pixels (`--sky-zero-px`, default 20,000 per exposure; 0 = all), several tiles supported. On CPU 2.4 times faster at the default on a field of 40,000-pixel cuts; levels differ from the all-pixel ones by 0.007 noise units (sampling alone predicts 0.007). No GPU timing yet.
- **Catalogue-free coarse registration (`--coarse-reg xcorr`)**: each exposure of a non-reference instrument is cross-correlated against the starting sky made from the reference instrument alone. Synthetic: injected shifts of 0.27 to 0.48 arcsec recovered to 7 mas (0.06 pixel). Real files: 28 HST exposures against two JWST exposures give shifts of 409 to 662 mas, which agree with the archive's Gaia-aligned solution to 15 mas (x) and 9 mas (y) rms without using any catalogue. The synthetic held-out comparison after the coarse step is inconclusive (unconverged fits). This replaces the `--hst-wcs archive` shortcut in the next joint fit.
- Guard on the blur dilation: warnings above 2 and 10, the maximum in the metrics, `--dilation-stop`. The agent's two-instrument synthetic field also ran away in the dilation (171 in 60 steps), so that fault is not confined to the truth problem.

### Queue at 09:20 UTC

- g0123: joint JWST + HST fit (ends about 09:40), LMC tile 7 grid twin (started 08:56), ablations 1 and 2. g0126: LMC tile 7 item fit (ends about 10:15). g0125: ablation 3 (slow node yesterday; speed to check).

## 18 Nov 2026, 06:15 UTC: how much the old held-out scores were flattered; first joint JWST + HST run; where the new zero-level rule works

### Decision 3 measured on real data: re-scoring the two saved SMACS fits with cross-fitted held-out terms

Same fitted sky and response as before (`onesky_grid004_dq2`, `onesky_items004w`); only the held-out exposures' own terms are now fitted on one half of the pixels and predict the other half. No scorer offset; 24 held-out exposures; 68% intervals.

| | Old score | Cross-fitted | Difference | Against the official pipeline (1.822), cross-fitted |
|---|---|---|---|---|
| Item sky, headline | 1.689 | 1.733 | +0.044 [+0.012, +0.073] | -0.088 [-0.230, +0.041] |
| Grid sky, headline | 1.713 | 1.768 | +0.055 [+0.040, +0.073] | -0.054 [-0.230, +0.085] |

- By bin the item sky loses 0.02 to 0.03 at every signal-to-noise below 50 (all significant) and nothing measurable above.
- **Consequence for what I told Seb on 16 Nov:** "the item sky is ahead of the official pipeline in the headline (-0.133 [-0.270, -0.005])" becomes -0.088 [-0.230, +0.041]: the headline lead is no longer significant. What survives clearly: the item sky is better than the pipeline at signal-to-noise 1 to 5 (-0.118 [-0.123, -0.110]), 5 to 20 (-0.082 [-0.096, -0.069]) and 20 to 50 (-0.068 [-0.121, -0.025]), level at 50 to 200 and above 200. The grid sky is worse than the pipeline at 5 to 50 and better at 50 to 200.
- So the flattery was real but small (2% to 3% of the score); it does not change which form is better.
- Old states load in the new package without loss (all tensors found; one removed buffer ignored).

### First joint JWST + HST run (10-minute smoke run, item sky, 123 JWST + 74 HST WFC3/IR exposures)

- The joint loader, the two instrument descriptions, the filter curves from the cluster's cache and the 2 arcsec kernel normalisation all ran: 145 training and 28 validation exposures, 8,309 patches.
- **Registration is the first real problem.** With HST's a priori pointing the offsets are 0.4 to 0.5 arcsec (three to four HST pixels), and the registration pre-pass, which is gradient steps, moved the translations by only 4 to 6 mas rms. The fit then starts at an input-noise score of 74 (18 for JWST alone). A gradient pre-pass cannot capture offsets larger than about a kernel width.
- **Shortcut taken for the first long joint fit (launched 06:04 UTC, `onesky_smacs_jwsthst_items_s0`, 200 minutes):** HST positions start from the archive's solution (`--hst-wcs archive`), which was aligned to Gaia by the archive (within 25 mas of the JWST frame); the fit refines from there. This is a catalogue-informed starting value. The catalogue-free way is a coarse cross-correlation against the starting sky before the gradient pre-pass (the harness already has such a tool for LMC: `--shifts-from`); to build next.
- The zero-level refresh costs about 5 minutes each time on 173 exposures (it renders every exposure), five times per fit: 12% of a 200-minute fit. To be made cheaper (subsample).

### Where the new zero-level rule works

- Item sky on SMACS: 83 to 86 of 86 training exposures find source-free pixels at each refresh (5% to 100% of their pixels): the rule is active.
- Grid sky on SMACS: 77 exposures at the first refresh, 1 at the second, none at the third: as the grid sky takes up diffuse light and background, no pixel counts as source-free any more and the rule falls back to the old data median. So with the grid sky in a field full of diffuse light the new rule is inoperative. (With the item sky it holds.)
- HST exposures of the cluster core (0.128 arcsec pixels): 0 to 96 of 30,500 pixels source-free; reference level kept.
- This says something general: the zero level can only be set where an exposure sees empty sky. A 20-arcsecond box in a cluster core has none for the coarser instruments; larger areas (tiles reaching blank sky) are needed for the zero level, not only for depth.

### GPU queue at 06:15 UTC

- Running on g0123: `onesky_smacs_grid004_v2_s0`, `onesky_smacs_items004w_v2_s0` (new defaults; end about 08:30), `onesky_smacs_jwsthst_items_s0` (joint; ends about 09:40).
- Running on g0126 (slow yesterday; speed being checked): `onesky_lmc_t7_items002w_s0`, the item sky on LMC tile 7 alone (the tile with the most held-out cuts and both programmes), 200 minutes, new defaults. Its grid twin follows.

## 18 Nov 2026, 05:45 UTC: registry back; official LMC score of the item fit (corrects my "failed"); queue running

- The image registry answered again at about 05:00 UTC (last error 04:55:22Z; first 01:40:13Z). The two SMACS fits with the new defaults started at about 05:02 and run at normal speed (grid 1.14 s per step, items 3.3 s per step); they end about 08:30.
- **LMC item fit, official paired score (10 tiles, 62 held-out exposures, 12.1 million pixels above signal-to-noise 5; no scorer offset; old all-pixel held-out terms for both of our fits):**

| Signal-to-noise | Official pipeline | Item sky (17 Nov) | Earlier grid fit (`fieldop_tiles_fix_sky002`) |
|---|---|---|---|
| headline (5 and above) | 6.11 | 7.32 (+1.20 [+0.74, +1.70]) | 8.03 (+1.92 [+1.31, +2.48]) |
| below 1 | 1.92 | 1.68 (-0.24) | 1.84 (-0.09) |
| 1 to 5 | 2.15 | 1.97 (-0.17) | 2.04 (-0.11) |
| 5 to 20 | 2.21 | 2.17 (-0.04 [-0.09, +0.03]) | 2.17 (-0.04) |
| 20 to 50 | 4.52 | 5.60 (+1.08) | 4.00 (-0.52) |
| 50 to 200 | 28.9 | 40.4 (+11.4) | 29.1 (+0.2) |
| above 200 | 259 | 308 (+49 [+8, +93]) | 531 (+272) |

  - So the item fit did not fail outright: with about 380 steps per tile it is better than the official pipeline below signal-to-noise 5, level at 5 to 20, worse from 20 to 200, and far better than the grid fit on the brightest pixels. In the headline it is worse than the pipeline and better than the grid fit. Its learned noise floor (0.34 and 0.47) still says the bright pixels were not fitted, which is where it loses.
  - My earlier statement to Seb ("failed; median 91 where the pipeline scores about 6") compared two different measures and is withdrawn.
  - Every LMC score here is far above 1 for all methods including the pipeline (crowded field, bright stars), so differences at 50 and above are dominated by a few stars.
- Launched at 05:36 to 05:40 UTC on the new package (`fieldop-tsky-83b17095.tar.gz`): re-scoring of the saved grid and item fits with cross-fitted held-out terms (old states load: all tensors found, one extra key ignored), and the 10-minute joint JWST + HST smoke run (197 files loading).
- First look at the grid fit with new defaults: the learned noise scale sits at its bound of 2.0 for two long-wavelength filters at step 300, and the noise floor is 0.14 and 0.09 at 15% progress (the old grid fit ended at 0.016 to 0.043). To be read at the end: the bound is doing what it is for, but a fit pressing on it has a misfit that needs explaining.

## 18 Nov 2026, 01:45 to 03:10 UTC: instrument interface with HST as second instrument; the truth-test divergence is the blur convention running away; LMC item fit failed; image registry down

### Instrument interface (decision 6; Opus build, `INSTRUMENT_INTERFACE.md`; I re-ran 30 tests, the agent reports 82)

- New `fieldop/instruments.py`: one description per camera (NIRCam, HST WFC3/IR, HST ACS/WFC) holding the reader, the unit of the science plane and its factor to MJy/sr, the camera's own data-quality bits, detector names, pixel scales, telescope diameter and the filter-curve identifier. Every science extension of a multi-chip file is its own detector. NIRCam now runs through it with identical results (both recorded default state hashes reproduced exactly).
- HST conversion to surface brightness:

```latex
S\,[\mathrm{MJy/sr}] = \frac{\mathrm{PHOTFNU}\times 10^{-6}}{\Omega_{\rm nom}\; t_{\rm unit}}\; x ,\qquad t_{\rm unit} = \mathrm{EXPTIME}\ \text{for files in electrons, 1 for electrons per second}
```

  with the nominal pixel solid angle from `IDCSCALE` and `PHOTFNU` derived from `PHOTFLAM` and the pivot wavelength when absent.
- The spectral range and the filter-curve grid now come from the union of the loaded bandpasses (audit finding 4); with NIRCam alone they are the old constants. `--spec-k-rule span` grows the number of spectral coefficients with the span (default keeps 12). Data-quality bits are per instrument (finding 7).
- Checks on real files, no fit (5 WFC3/IR exposures and 1 ACS exposure of SMACS against NIRCam):
  - Units: HST over JWST aperture flux, median over 16 to 27 sources: F105W against F090W 1.20 to 1.28; F160W against F150W 0.92 to 0.94; F140W against F150W 0.81 to 0.83; ACS F814W against F090W 0.76 to 0.83. All of order 1 (a units error would give 17 or 2.6, the pixel-area ratios); the spread is mostly the colour of red cluster galaxies.
  - Geometry: translations of 0.39 to 0.51 arcsec (WFC3/IR) and 0.10 arcsec (ACS) with HST's a priori pointing, which the registration pre-pass removes; scale and rotation zero within 1.8 sigma (about 200 to 250 parts per million, 40 to 50 arcsec of angle); residual scatter 22 to 27 mas, which is centroid noise of galaxies.
  - Noise: blank-sky scatter over the file's error: NIRCam 0.96 to 1.03; WFC3/IR 0.78 to 0.91 (its error is 10% to 20% too large); ACS 1.12 to 1.22 (unflagged cosmic rays).
- Choices to confirm with Seb: HST positions use the a priori guide-star solution, not the archive's Gaia-aligned one (my reading of "catalogue-free"; `--hst-wcs archive` switches); which HST data-quality bits are kept; ACS chips as independent exposures.
- Not supported yet: WFC3/UVIS (6 files) and WFPC2 (8).
- **Joint list of exposures built** (`joint/make_joint_splits.py`, on CANFAR under `scripts/onesky/`): the SMACS list plus 74 HST WFC3/IR exposures as training only (F105W 26, F125W 8, F140W 20, F160W 20; 24,700 s in all; every one covers the box), held-out set unchanged (24 JWST exposures). The four filter curves are already in the cluster's cache. First joint fit: `--splits /arc/projects/ots/adass2026/scripts/onesky/splits_SMACS_jwst_wfc3ir.json --gauge-ref-inst 0 --epoch-axes 0` plus the item arguments; the first run should be a 10-minute smoke run, since the joint loader has only been tested on a synthetic field and on single real files.
- Package archive: `fieldop-tsky-83b17095.tar.gz`.

### Why the truth test diverges with a learned response (Sonnet bisection, `fieldop_dbgB/DIVERGENCE_BISECTION.md`)

- Cause: the common-blur convention (the rule that no kernel may be sharper than one sky node, enforced by a dilation of all kernels) runs away while the kernel network is learned: the dilation goes 1.0, 3.7, 10.9, 123 and then about 1,000 between steps 20 and 130; the kernel positions then fall inside the encoding's innermost radius and the kernel network's gradient is exactly zero from step 140.
- Evidence: kernel learned with the convention on: held-out 2.23 times the noise value; with it off: 1.11; kernel at the truth with it on: 1.01; every other group of parameters learned: 1.01. Dividing the kernel's learning rate by 3 or 10 delays it (1.13, 1.09 in 400 steps; the dilation still creeps). More data does not cure it (16 exposures 1.02, 32 exposures 2.27; one seed each).
- **Real fits have not shown this:** the dilation is printed at every logged step and stayed between 1.0 and 1.33 in every SMACS and LMC fit read so far. The truth problem is harsher (a kernel core of 0.07 arcsec on 0.10 arcsec pixels). But nothing prevents it, so it is a robustness fault to fix: a guard (warning when the dilation passes 2) and a better-posed convention. A cap at 4 was tried as a stopgap (1.12 and 1.03) and not adopted: the value is arbitrary and the convention is then permanently active.
- Not known: why the kernel core sharpens along that direction; the rate across seeds; the item sky.

### LMC item fit (`onesky_lmc_items002w_s0b`): failed, as its noise floor warned

> **Correction, 18 Nov 05:45 UTC:** "failed" and "median 91 against about 6" were wrong. The 91 is the fit's own number on pixels where the *model* is above signal-to-noise 5, an unweighted median over cuts; it is not the protocol's score. The official paired score is in the entry of 18 Nov 05:45: headline 7.32 against 6.11 for the official pipeline and 8.03 for the earlier grid fit.

- 3,775 steps over 10 tiles (4.5 s per step), final learned noise floor 0.34 and 0.47. The fit's own held-out numbers: 3.28 over all pixels against input noise, and a median of 91 on pixels above signal-to-noise 5 (the official pipeline scores about 6 there). Official paired scoring is waiting on the registry (below).
- Reading: about 380 steps per tile and 60,000 items for a crowded 20-arcsecond star field are far too few; with the new default bound on the noise floor the fit could not have hidden it. Next LMC test: one tile, 200 minutes, items against grid.

### Infrastructure

- Since 01:40 UTC every new session stays "Pending": the image registry answers HTTP 500 ("failed to resolve reference images.canfar.net/astroai/base:latest: unexpected status from HEAD request ... 500 Internal Server Error", last seen 03:02 UTC, node g0123). Waiting: the two SMACS fits with the new defaults (sessions oneskyg2, oneskyi2) and the LMC scoring job (scorelmc1). They start by themselves when the registry answers.

## 18 Nov 2026, 00:10 to 01:45 UTC: Seb approved the six decisions; new defaults, fair held-out scoring, and what the flat-field test on SMACS really measured

Seb (17 Nov, late): "six decisions: ok, yes. go ahead. keep debugging the stuff from your audit." I read "go ahead" as also covering the GPU fits proposed in my last message (one LMC tile, items against grid; the count-rate fit with learned gain on LMC programme 1476).

### New defaults (Sonnet build, `DEFAULTS_V2_2026-11-18.md`; I re-ran 30 tests on the final tree, the agent reports 75 in all)

- `--sky-zero masked` (zero level from pixels the model calls source-free); `--noise-floor-max 0.2`; new `--noise-scale-max 2.0` (learned noise scale per detector and filter kept within a factor of 2, with a warning at the bound); `--nuisance-steps exposure`.
- Several instruments: the fit refuses to start without an explicit `--gauge-ref-inst`, and normalises the kernel within 2 arcsec unless told otherwise.
- **Held-out scoring is now cross-fitted (`--held-xfit 1`)**: each held-out exposure's own terms (background, affine, gain, codes, stripes) are fitted on one half of interleaved 4 by 4 pixel blocks and predict the other half, then the reverse, so no predicted pixel was seen by the terms that predict it. The old all-pixel score is still printed next to it. On the two short synthetic fits the old way was flattering by 0.2% to 4.6% against input noise and 5.5% to 9.0% against the model's noise (3 held-out exposures, unconverged fits: indicative only). The real size is measured by re-scoring the saved SMACS states (queued).
- `--heldout-only --state FILE` re-runs only the held-out stage from a saved state (older states load; missing and extra keys are logged); the launch configuration is now stored inside the state.
- Audit findings fixed: 12 (time axes decided per instrument), 14 (held-out optimiser steps only the rows present). Finding 15 (start the throughput, distortion and noise fields at exactly zero) is available as `--field-init zero` but **not** default: the exact zero start is singular (the fields are normalised by their rms, so the first gradient is about 10,000 times larger and gradient clipping scales every other operator gradient down on that step); to be tested on a GPU before it becomes default.
- Removed: `--tau-prior`, `--sky-tv`, `--xfit`, `--xfit-frac`, `--gauge-shift` and their code. **I removed 5 options, not the 20 I had proposed:** the other 15 "never exercised" options select data or resources (`--box-id`, `--data-root`, `--load-workers`, ...), belong to planned tests (`--nonlin`, `--rate-photom`), or are now used (`--kernel-norm-radius`).
- `--legacy-defaults` restores every old default; with it the two small synthetic fits reproduce the recorded state hashes exactly (grid `ff0f1183...`, items `840b7a11...`), which is the proof that nothing else changed. New default hashes: grid `579e2b1c...`, items `ba5f9c80...`.
- Package archive: `fieldop-tsky-90ff3b37.tar.gz`.

### What the flat-field test on SMACS measured (Opus diagnosis on the saved states and the real files, `FLAT_PATTERN_DIAGNOSIS.md`)

- **Correction to my entry of 23:15:** a SMACS detector pixel is seen by 4 to 6 training exposures, not about 30 (I misread a count of gradient visits). With so few, the learned gain's error against the true flat (3.1% to 4.7%) is larger than the flat itself (2.2% to 4.3%). That is why it cannot help faint pixels: on a background at signal-to-noise 9 to 30 per pixel it costs 0.42 where no gain costs 0.17 (signal-to-noise 1 to 5; perfect-sky calculation on the 12 local held-out exposures, per-exposure terms not re-fitted).
- With a perfect flat the excess would be 0.010 at signal-to-noise 1 to 5 and 0.11 at 50 to 200: this is the ceiling the learned gain is aiming for.
- Shrinking the map by its own noise makes it harmless on faint pixels (0.16 to 0.17, the same as no gain) and gives back the bright-pixel advantage. So shrinkage is necessary, and more exposures per pixel are what makes the gain pay.
- **What the pipeline's flat contains:** calibrated = count rate times conversion over flat; the pixel-area map is not applied. On NRCB1 the flat contains the pixel-area variation (large-scale correlation +0.79); on the long-wavelength detector it does not (-0.09). So yesterday's doubt was real: on the long-wavelength detector the model's area factor leaves a large-scale term of 0.77% rms (mostly a tilt) that only the per-detector throughput field could carry, and that field is pruned on a 20-arcsecond box. Nothing is counted twice. Not known whether the long-wavelength flat leaves the area out by design.
- **The additive detector pattern is not the flat in disguise.** Its reported size (1.5 to 3.7 sigma) was mostly grid nodes no training pixel touches plus a degree-2 surface that trades with the per-exposure background shapes; beyond degree 2 it is 0.05 to 0.19 sigma, and it adds at most 0.001 to 0.035 to held-out scores. The end-of-fit report now gives the size over the observed footprint and beyond degree 2.
- New form, off by default (`--pixel-gain-prior mix2 --pixel-gain-gauge poly+pattern`): the gain multiplies sky plus background; the additive pattern stays in detector units; the gain is projected off what the pattern and the polynomial can represent; its prior is a two-component mixture whose weights and widths are learned from the data (expectation-maximisation with the posterior variance), so the shrinkage strength is not set by hand.

```latex
d = e^{g}\Big[\frac{a}{a_{\rm nom}\,C}\sum_q w_q\,e^{\gamma_q}\,(K_q * s)_q + b\Big] + A_{\rm det} + \text{stripes}
```

- Synthetic check (2 seeds): flat error 0.0336 and 0.0377 against 0.0369 and 0.0445 for today's form; held-out prediction better than today's form and than no gain at every signal-to-noise below 20; above 20 the sign differs between seeds. No real-data run yet.
- Open: the long-wavelength flats differ between filters (correlation 0.85 to 0.90), so one gain map per detector may not be enough there; the mixture's tail component did not separate at a noise limit of 4.6%.

### What this means for the plan

- The flat-field test needs many exposures per detector pixel, which a 20-arcsecond box cannot give on SMACS. Before launching it on LMC programme 1476 I count the exposures per detector pixel there from the LMC state (the tool exists: `flat_check.py --ref-state`). If that count is also small, the right test covers more of each detector (more or larger tiles), not just more exposures.

### GPU queue (18 Nov 01:40 UTC)

- Launched on node g0123: `onesky_smacs_grid004_v2_s0` (method `onesky_grid004_v2`) and `onesky_smacs_items004w_v2_s0` (`onesky_items004w_v2`): the two SMACS reference fits with the new defaults and cross-fitted held-out scoring. Both are waiting: the image registry returned HTTP 500 on the image pull at 01:40 (`images.canfar.net`), the sessions retry by themselves.
- Next: re-score the saved states of `onesky_grid004_dq2` and `onesky_items004w` with `--heldout-only --legacy-defaults --held-xfit 1` (measures the flattery of the old scores on real data); one LMC tile, items against grid; then the count-rate fit with the new gain prior where the exposure count per pixel allows.
- The LMC item fit of 17 Nov finished its fit at 00:24 UTC (3,775 steps; learned noise floor 0.34 and 0.47 at the end); collection at 02:13.

## 17 Nov 2026, 23:15 UTC: first sound count-rate fits on SMACS: the fit learns the true flat field, but not yet well enough to help prediction

Collection by a Haiku agent; flat comparison and flux check by me. Fits: `fieldop_rate_grid3` (control, 9,140 steps, 1.31 s per step) and `fieldop_rate_gain_grid3` (learned gain per detector pixel, 8,821 steps), both from count-rate files in sky units, 200 minutes, grid sky.

**The count-rate path is sound now.** Learned noise floor 2% to 5% (1,000% in the broken fits); learned noise scale 0.68 to 0.88; held-out score against input noise 2.78 (control) and 2.70 (gain); the fitted sky is 1.9 to 3.1 times the official mosaic's flux, the same factor as the calibrated grid fit (the known split between far kernel wings and background), so the sky units agree between input levels.

**Paired scores at count-rate level** (24 held-out exposures; no scorer offset; difference from the calibrated twin `onesky_grid004_dq2`, 68% interval):

| Signal-to-noise | Calibrated twin | Count-rate, no gain | Count-rate, learned gain |
|---|---|---|---|
| headline (5 and above) | 1.95 | 2.50 (+0.55 [+0.32, +0.78]) | 2.48 (+0.53 [+0.45, +0.61]) |
| 1 to 5 | 1.11 | 1.24 (+0.13) | 1.35 (+0.24) |
| 5 to 20 | 1.28 | 1.50 (+0.22) | 1.57 (+0.29) |
| 20 to 50 | 1.35 | 1.72 (+0.37) | 1.70 (+0.35) |
| 50 to 200 | 3.51 | 7.22 (+3.71) | 5.16 (+1.66) |
| above 200 | 182 | 204 (+23) | 228 (+47) |

Against the official pipeline at count-rate level: headline +0.54 (control) and +0.51 (gain); the gain fit is better than the pipeline at 50 to 200 (-2.0 [-3.8, -0.7]).

**The learned gain is the real flat field, at about 70% to 80% of the correlation the noise allows.** Against the official flat (ratio of count-rate to calibrated files, same polynomial removed):

| Detector | Official flat rms | Learned map rms | Correlation | Correlation the noise allows | Slope |
|---|---|---|---|---|---|
| NRCB1 | 4.3% | 4.9% | 0.49 | 0.67 | 0.56 |
| NRCB3 | 3.0% | 4.4% | 0.40 | 0.56 | 0.58 |
| NRCBLONG | 2.1% | 3.4% | 0.44 | 0.67 | 0.71 |

- So a flat field can be learned from the sky exposures alone, with no flat-field frames: criterion 1 of test 3 passes (rule: at least 0.3 where the noise allows more than 0.5; measured 0.45 to 0.50 there).
- Criterion 2 fails on SMACS: the count-rate fit with gain is worse than its calibrated twin by 0.29 to 0.35 at signal-to-noise 5 to 50 (rule: at most 0.05), and it is not better than the control without gain except on bright pixels (50 to 200: 5.2 against 7.2). At faint pixels it is worse (1.35 against 1.24).
- (Corrected 18 Nov: 4 to 6 training exposures per pixel, not 30; see the entry of 18 Nov.) Reading: each SMACS detector pixel is seen about 30 times, so the per-pixel noise of the learned gain (median 1.7% to 3.9%) is as large as the flat itself (2% to 4%); the map is half signal, half noise (slope 0.56 to 0.71), and the noise half costs more on faint pixels than the signal half gains. Two things follow, neither tested yet: shrink the map by its own noise (a learned prior on its amplitude), and use more visits per pixel, which is what LMC programme 1476 provides (680 exposures).
- **Verdict for test 3 on SMACS:** the element is not dropped and not adopted. It learns the right thing but does not pay yet; the LMC fit decides. I am not applying the letter of the drop rule (which says drop) because its first criterion passed clearly and the failure of the second has a specific, testable cause.
- Open doubt (long-wavelength pixel area counted twice) not settled: the per-detector throughput field was pruned in both fits, the detector pattern is larger on the long-wavelength detector (3.0 to 3.7 sigma against 1.5 to 2.1), consistent with it but not proof.
- Also visible in the new end-of-fit report: 53% to 56% of the F090W kernel lies beyond 2.5 arcsec (the wings and background trade again); held-out background constants sit 1.2 sigma from the data median (training: 0.27).

**LMC item fit (`onesky_lmc_items002w_s0b`), still running (68% at 22:55): it is not fitting.** The guard warned at step 674: the learned noise floor passed 0.2 and is now 0.36 and 0.50; the score against input noise is 27 to 28 while the model-noise score reads 1.1. At 4.9 s per step it will have about 4,000 steps over 10 tiles (400 per tile). Expect a poor score; the cause to check first is too few steps per tile and too few items for a crowded star field (60,000 per tile).

## 17 Nov 2026, 18:30 to 19:30 UTC: GPU back; three GPU nodes are 10 to 30 times slow; fits relaunched on the fast one

- The node with the driver fault is `keel-prod-g0123.arbutus.uvic.ca` (captured at 17:54:27 UTC: "failed to initialize NVML: Driver/library version mismatch"; 29 of our launches failed there between 02:25 and 17:54). It was repaired before 18:30.
- At 18:32 to 18:35 the three queued fits started, on `g0126`, `g0125` and `g0124` (one H100 each). They ran 10 to 30 times slower than yesterday: registration pre-pass of 100 steps in 981 s and 1,475 s (yesterday 35 s), first fit step after 85 s (yesterday 2 s), no step 100 after 11 minutes; the LMC fit had not finished its pre-pass after 40 minutes.
- Cause separated with four 5-minute runs on the same data: yesterday's package and today's package (with and without the new count-rate options) all ran at normal speed on `g0123` (pre-pass 27 to 32 s, about 1.2 s per step). So it is the nodes, not the code. Not known why (the GPU-bound kernel warm start took its usual 24 s on the slow nodes, so the processors or storage of those nodes are the likely limit, not the GPUs).
- I stopped my three slow jobs and relaunched them at 19:21 to 19:24; all landed on `g0123`: `fieldop_rate_gain_grid3` (session oneskyrgg3), `fieldop_rate_grid3` (oneskyrg3), `onesky_lmc_items002w_s0b` (oneskylmcj, method `onesky_lmc_items002w`). The runs `fieldop_rate_gain_grid2`, `fieldop_rate_grid2` and `onesky_lmc_items002w_s0` are incomplete and void.
- First steps of the relaunched count-rate fit with gain: 1.16 s per step (yesterday 1.43), learned noise floor 0.011, score against input noise 18.5 at step 0 and 13.4 at step 100. Count-rate fits end about 22:50 UTC, the LMC fit about 00:30 UTC.
- Package archive in use: `fieldop-tsky-67a98fa5.tar.gz`. Count-rate arguments now include `--rate-units sky --nuisance-steps exposure`.
- Lesson for launches: after starting a fit, read the pre-pass time in its log (about 30 s per 100 steps on SMACS); if it is several times longer, find the node with `canfar events <id>` and relaunch.

## 17 Nov 2026, 07:45 to 11:25 UTC: a round of debugging, performance and simplification while the GPU is blocked (Seb's request)

Seb: "do a round of debugging, performance, simplifying, but always with the goal of multi-survey joint analysis for new science discovery, ultimately probing the deepest sky latent field." Five agents (Opus: audit, then two model changes; Sonnet: truth test, then performance and simplification). I re-ran the new tests on the final tree (25 pass in the seven files touched; the agents report 54 in all). All work is in `fieldop_tsky/`; reports: `AUDIT_2026-11-17.md`, `tests/truth/TRUTH_REPORT.md`, `PERF_SIMPLIFY_2026-11-17.md`.

### 1. Audit by an agent that did not write the code (units, terms that hide errors, single-instrument assumptions)

Two bugs fixed, default results unchanged: (a) every filter curve was looked up under the first instrument's name (an HST filter in a NIRCam field would have been fetched as a NIRCam filter); (b) the blur convention read a telescope-diameter key no loader sets and always used 6.5 m. End-of-fit reporting added for every learned term that can absorb an error (noise scale per detector and filter, backgrounds, detector pattern, stripes, affine terms, gain, codes, pixel gain, non-linearity, variability scales at their caps, kernel fraction beyond 0.5, 1 and 2.5 arcsec).

Fifteen ranked findings; the ones that matter most for a joint deep fit:
1. Count-rate input had no photometric conversion, so its sky was in different units per filter and detector (now optional, section 3).
2. The sky's zero level is tied to the median of each exposure's data, which contains sky light (section 3).
3. The learned noise scale can change the noise by a factor of 20 either way per detector and filter, and the noise floor is unbounded; either hides a units error. The factor "10.8" of the failed count-rate run was this noise scale. Proposed default bounds: floor at most 0.2, noise scale within a factor of 2. Not applied (changes every fit).
4. The spectral range (0.70 to 5.30 micrometres) and the filter-curve grid (0.55 to 5.6) are constants: bluer HST filters, GALEX and Spitzer 5.8 and 8.0 micrometres would silently get wrong colours.
5. With two instruments the absolute flux scale becomes an average weighted by exposure counts; an explicit reference instrument should be required.
6. The loader is specific to JWST (extension names, first science extension only, detector names, a 2,000-pixel minimum, data-quality bits shared across instruments). The audit lists the short interface an instrument must supply, so that HST becomes a data-description task.
7. Held-out scores are flattered by terms fitted on the held-out pixels themselves (closed-form stripes with no prior, background, affine, gain, codes). The official-pipeline baseline gets none of these. Proposed: fit them on one half of interleaved blocks and score the other half. This changes every score we have quoted; not yet measured how much.
8. Kernel far wings trade with the background by default (known); proposed default normalisation radius of 2 arcsec for joint fits.
Others: detector pattern shared across filters in calibrated units; time axes switching on for every instrument when one extends the time span; step sizes using one mean over all exposures; held-out optimiser not masked by row; random starting values for throughput and distortion fields.

### 2. A small problem with known truth that converges, and what it shows about depth

Until now no synthetic fit converged (held-out 1.4 to 3.3 against a noise value of 0.99), so nothing measured faint-end bias or depth. New: `tests/truth/` (one filter, 44 by 44 pixels, 36 sources, flat background, 8 or 32 exposures). It converges to within 1% to 3% of the noise value in 1 to 4 minutes **only with the response frozen at the truth**; with the response learned from the usual start the fit diverges after an initial descent (2.2 to 2.3; not diagnosed). So every number below is for a known response.

With today's defaults, on that problem:
- **The recovered sky in empty regions is biased high and the bias does not fall with more exposures**: +0.86 ideal-noise units at 8 exposures, +1.60 at 32 (grid and items alike). The scatter falls; the offset (+0.19 in surface brightness, 3.8% of the background) does not. Cause: the background is tied to the data median, which contains source light, and positivity stops the sky from compensating.
- Sources at total signal-to-noise 3 to 10 lose about 1.1 to 1.2 noise units of flux (ratio 0.77 to 0.79 in an aperture matched to the response) which the ideal coadd does not lose.
- Items recover point-source flux above signal-to-noise 10 without bias; the grid loses 20% to 30% in a strict true-profile aperture (3% to 7% in the matched one).
- Without positivity the grid has no offset at 8 exposures but its error is 1.7 to 2.1 times the ideal coadd's. This is not the resolution element and not under-convergence (800 steps are worse than 400); the excess grows with scale. Candidates, not separated: the per-exposure quadratic backgrounds against the sky's large scales, and a node count comparable to the data.

### 3. Two model changes, off by default (Opus)

**Sky zero level, `--sky-zero masked` with `--sky-positivity-margin 3`.** Convention: for each filter the training backgrounds average to the median of each exposure's data over pixels where the model's own rendered sky is below 0.3 sigma (smoothed by 1.5 pixels), re-evaluated five times per fit; the sky may go negative down to three times the starting node noise before positivity acts. On the truth problem (2 noise realisations per row; errors on the new offsets 0.15 to 0.27):

| Sky | Exposures | Empty-sky offset, before to after | Error over ideal, before to after | Flux ratio at signal-to-noise 3 to 10, before to after |
|---|---|---|---|---|
| grid | 8 | +0.86 to +0.12 | 0.93 to 1.92 | 0.77 to 0.95 |
| grid | 32 | +1.60 to -0.54 | 1.60 to 1.58 | 0.68 to 0.95 |
| items | 8 | +0.89 to +0.07 | 0.92 to 1.70 | 0.79 to 0.94 |
| items | 32 | +1.68 to -0.19 | 1.68 to 1.25 | 0.67 to 0.93 |

("Before" error ratios below 1 at 8 exposures were the offset, not noise.) The error now falls from 8 to 32 exposures by 0.41 (grid) and 0.37 (items) in flux units (0.5 expected; 0.86 and 0.92 before). Costs and limits: held-out score 5% to 7% above the noise value at 8 exposures (1% to 2% before); the error is still 1.25 to 1.9 times the ideal; item flux at signal-to-noise above 10 at 8 exposures is +0.70 noise units high (was +0.08); the mask threshold and smoothing were tuned on one of the seeds in the table; one margin value tried; no real-data fit; no two-instrument test. With the masked zero alone (no margin) positivity still biases empty sky by +0.2 to +0.6 noise units (rectified noise), so the margin is needed.
**Verdict: the depth test is improved, not passed.** The zero level is no longer biased within the sampling error and faint fluxes are recovered to 5% to 7%, but the sky's error in empty regions is still above the ideal coadd's, for a reason not yet found.

**Count-rate input in sky units, `--rate-units sky` (with `--nuisance-steps exposure`).** Count rate per pixel = surface brightness times (pixel area over nominal area) over the conversion, where the conversion (`PHOTMJSR`) is read from the matching calibrated file's header or from a table; it is an input and a starting value, and the learned throughput terms stay free. Starting values converted consistently. Checks: synthetic, five different conversions: fitted fluxes agree with the calibrated-style fit to 0.5% and held-out scores to 1.6% (both fits unconverged); on the 12 real exposure pairs, without fitting, count rate times conversion against calibrated data: median ratio 0.9998 to 1.0084 per detector and filter, pixel-to-pixel scatter 1.8% to 3.6% (the flat field the fit is meant to learn). Open doubt: on the long-wavelength detector dividing by the relative pixel area increases that scatter (3.4% to 3.6% against 2.5% to 3.2%), so the area may be counted twice there; to be settled on the first real fit (the per-detector throughput field would show it). The queued count-rate fits now use these options.

### 4. Performance and simplification (Sonnet; CPU only, no GPU today)

- Where a step goes (4 patches per step, CPU): grid sky 582 ms: backward pass 48%, forward rendering 29%, kernel field 15%, per-exposure terms and loss 3%. Item sky 816 ms: backward 53%, rendering 22%, kernel field 12%, item rasterisation 9%. Optimiser, projections, logging and parameter-independent recomputation together are under 3%.
- Cost model: about 150 ms per patch on this CPU; step time is independent of the number of exposures in the field (618, 577, 559 ms at 18, 42, 90 exposures) and doubles with patches per step. So cost per unit of data is set by rendering, not by bookkeeping; larger areas scale by tiles and patches.
- Applied without changing results: blocking device reads per step from 28 to 6 (grid) and 32 to 10 (items), host-to-device copies from 27 to 2; 3% to 5% faster on CPU (inside the 5% run-to-run scatter). The effect on a GPU is unmeasured; `tests/bench_step.py` is the benchmark to run there.
- Removed: item relocation, the smooth grid under items, the `--render-impl` stub. 5,337 to 5,250 lines; 161 to 158 options. An inventory of all 158 options with their status is in the report; 20 are never exercised by any test or adopted launch line (candidates for removal; `--tau-prior` is read nowhere).
- The item-sky state hash changed only because an all-zero buffer was removed (all other tensors bit-identical; old states still load). The grid hash is identical.
- Largest remaining costs: the backward pass through the renderer (half the step); the kernel field evaluated twice per step; for the item sky, three rasterisations per step because of the sky-only inner step (dropping it would save about 30% of an item step and changes the fit). Each of these changes results and needs a GPU comparison.

### 5. Decisions for Seb (each changes results, so none is applied)

1. Make `--sky-zero masked` the default (and the positivity margin for the deep-sky product only).
2. Default bounds on the learned noise terms (floor at most 0.2; noise scale within a factor of 2).
3. Score held-out exposures on pixels not used to fit their per-exposure terms (changes every quoted score; makes the comparison with the official pipeline fair).
4. Require an explicit reference instrument for the flux scale as soon as two instruments are loaded; make the 2 arcsec kernel normalisation the default for joint fits.
5. Remove the 20 never-exercised options; drop the sky-only inner step if a GPU comparison shows no loss.
6. Next model work for the second instrument: the instrument interface of audit section C (reader, units, data-quality policy, spectral range from the union of bandpasses), then HST WFC3/IR on SMACS.

**GPU.** Still blocked at 11:22 UTC: 20 launches since 05:45 have landed on the node with the driver fault. Next retry about 14:25 UTC. Archive for the queued fits: `fieldop-tsky-67a98fa5.tar.gz`.

## 17 Nov 2026, 06:00 to 07:15 UTC: the count-rate failure was a fault in the fit's starting sky, not the gain map (corrects the 02:20 entry)

Collection by a Haiku agent; diagnosis and fix by an Opus agent (`campaigns/continuous_sky/RATE_HELDOUT_DIAGNOSIS.md`); I re-ran the new tests (pass).

**The control without a gain map (`fieldop_rate_grid`, 8,099 steps) fails on held-out exposures in the same way as the fit with the gain map:** 1.7 to 160 per exposure, growing with the amount of source signal (long-wavelength detector 70 to 160). So the 02:20 failure was not caused by the gain map, and my reading there (gain trading against the scene with small dithers) is withdrawn: it was never tested, because both fits were broken.

**Cause.** With count-rate input the model treats the sky as DN/s per square arcsecond and multiplies by the pixel area, but the sky's starting estimate (the inverse-variance coadd of the training pixels) was built in DN/s per pixel. Every source started 250 to 1,000 times too faint. The fit did not recover; instead the learned fractional noise floor rose from 0.01 to about 10 (1,000%), which removed bright pixels from the loss. The training score in the log is computed with the model's own noise, which is why training looked fine (0.93 to 1.03).
- Evidence (control state rebuilt on CPU; reproduces the logged held-out scores exactly): model over data summed over pixels above 20 sigma is 0.15 on a held-out F277W exposure and 0.07 to 0.08 on two short-wavelength ones; on bright long-wavelength pixels the model's noise is 27 times the pipeline's; per-exposure gain factors of training exposures are 0.99 to 1.03, so no per-exposure term carries the missing factor: the frozen sky is too faint for training exposures as well.
- A mistake of mine in the 02:20 entry: "per-exposure gain exp(0.83)" was a misreading. The stored value is multiplied by a scale of 0.0104; the gain factor is 1.009.
- The saved states cannot be rescued by re-running the held-out stage (a free amplitude per exposure still leaves 35.7 on the F277W exposure).

**Fix** (in `fieldop_tsky`; count-rate input only; calibrated path unchanged to the state hash): the starting coadd divides data and noise by the header pixel area. Approximation: the header's pixel area, not the learned one (they differ by less than 0.7%). Synthetic count-rate fit, 60 steps: held-out score 158.3 before, 5.53 after, against 5.57 for the calibrated control; model over truth amplitude 0.022 before, 0.955 after. Two new tests fail on the old package and pass on the new one; 41 tests pass.

**Reported, not fixed (design weaknesses this exposed).**
- The learned noise floor has no upper bound, so a fit can hide a gross misfit by inflating its own noise.
- The training log reports the score only against the model's own noise. It should also print it against the pipeline's noise, and warn when the floor exceeds a few per cent. Until that is added, check the floor at the end of every fit (calibrated fits: 0.016 to 0.043).
- The DN/s conversion differs between short-wavelength detectors by 1.5% to 3.5%; only the per-detector throughput field can carry it, and the evidence rule pruned that field in both failed runs. If it is pruned again, expect residuals of up to 3.5% on bright short-wavelength sources.

**Guard added at 07:35 UTC (Sonnet build, `fieldop_tsky`, archive `fieldop-tsky-9538c031.tar.gz`).** Every logged step now also prints the batch score against the input noise alone (`chi2/px(input noise)`) and the learned noise floor (`noise-floor`); a `WARNING` line is printed the first time a floor exceeds 0.2 and at the end of the fit; the metrics file gets `noise_floor_final` and the held-out score against input noise. New option `--noise-floor-max` (default 0: no bound, so existing results are reproducible) clamps the floor after each step. The default synthetic fit gives the recorded state hash; 46 tests pass.

**Status of the raw-input test (test 3).** Open again. Both count-rate fits need re-running with the fixed package (archive `fieldop-tsky-9538c031.tar.gz`, uploaded; `/home/claude/work/CURRENT_SCRIPT`). The drop rule has not been applied to a sound fit yet.

**GPU.** No healthy slice since the control ended at 05:45: twelve launches (four of the LMC item fit at 06:20, eight of the gain re-run at 07:10 and 07:35) all landed on the node with the driver fault ("failed to initialize NVML: Driver/library version mismatch") and failed at container start. Queue, in order, when a slice frees: `fieldop_rate_gain_grid2` (arguments of `fieldop_rate_gain_grid`), `fieldop_rate_grid2` (control), LMC item sky (`onesky_lmc_items002w_s0`: the tile arguments of the reference run plus items, 60,000 per tile, pitch 0.02 arcsec, 285 minutes, `--dq-ignore-bits 2`, `FIELD=LMC`).

**Re-fetch: complete.** LMC calibrated 680 + 352 and count-rate 680 + 352; HST SMACS 122 files in the folder (116 expected by the manifest: 6 extra, not yet looked at); JWST union 587 files in seven programme folders (the 344 I had written down was wrong or counted another product level: to be checked against the manifest); Spitzer and GALEX complete; SPHEREx: 24 cut-outs fail with HTTP 422 (same request form as the ones that work; left). All five fetch jobs ended with exit 0. Scores of the control at count-rate level were written but not read (they are void).

## 17 Nov 2026, 04:00 UTC: one learned shift per detector and filter closes the position gap on files; learning it in a fit is not yet shown

Build by a Sonnet agent in `fieldop_tsky/` (`--filter-shift {off,learned}`, default off, default synthetic fit unchanged to the state hash). I re-ran its tests with the neighbouring ones (21 pass). Report: section E of `TORCHSKY_INTEGRATION_REPORT.md`.

**On files (12 SMACS exposures): one constant shift per (detector, filter) pair brings the telescope-frame positions to 0.6 to 2.6 mas rms of the pipeline's solution (maximum 3.5 to 7.8 mas).** Shifts in detector pixels: F200W (1.05, 1.62) on both short-wavelength detectors; all other filters below 0.2 pixel. The two exposures of each long-wavelength pair want the same shift to 1e-7 pixel. So the positions of count-rate files need the aperture polynomial plus nine pairs of numbers, and no calibrated file.

**Learned inside a fit (synthetic, item sky, 1,500 items, 180 steps, one seed, injected shifts of 0.5 to 2 pixels): not validated.** Held-out score 1.426 without injected shifts; 1.751 with them and the switch off; 1.248 and 1.379 with the switch on (two step sizes). The shifts move a quarter to a third of the way (largest remaining error 1.0 to 1.2 pixels against 1.57 for no correction); the target of 0.05 pixel is missed. The fits are far from converged (1.4 against a noise floor of 0.99) and with one seed and three held-out exposures the score differences are inside run-to-run scatter, so the lower scores with the switch on are not a result.

- With a grid sky the shifts between filters are not identifiable: with 12 spectral coefficients per pixel each filter can hold its own shifted image. Only a sky whose sources have one position at all wavelengths (the item sky) can pin them. This is one more argument for the item form, and a warning for the grid twin of any count-rate fit.
- Free origin: the common sky translation of all shifts is removed by projection after each step; within a pair the per-exposure translations are made zero-mean and the excess is moved into the pair's shift (the model function is unchanged).
- Approximations and limits: the shift is applied to first order with one Jacobian per exposure (a few per cent of the shift at most, the agent's estimate, not measured); several disconnected tiles are refused; the first sky coadd ignores a starting table; no run on calibrated input (where the shifts should come out near zero); no real-data fit.
- What I take from it: start the shifts from a table (`--filter-shift-init`, values above, or the pipeline's reference file) and let the fit refine them, in line with physics as a starting value only. Learning from zero needs a longer or staged fit and is a GPU test, queued behind the count-rate control.

## 17 Nov 2026, 03:30 UTC: the prototype can now run on three of the four torchsky pieces; two are adopted

Build by a Sonnet agent in `fieldop_tsky/` (a copy of the current package; every default keeps the old code path, and the small synthetic fit gives the identical state hash). I re-ran its 10 new tests (pass); the existing 25 pass unchanged. Report: `campaigns/continuous_sky/TORCHSKY_INTEGRATION_REPORT.md`.

| Piece | Switch | Adopted | Number |
|---|---|---|---|
| Item sky by `GaussianItemScene` | `--item-impl torchsky` | yes | raster agrees to 2.2e-7 of the image maximum, gradients to 1e-5; synthetic held-out score 3.33121 against 3.33148; same speed |
| Exposure files by `field_observation_from_exposure_file` (torchfits) | `--loader torchfits` | yes | identical to the astropy loader on the local sample (38 comparisons, none differ); only 1 of the 12 calibrated sample files is complete, the other 11 were compared on header and science plane |
| Count-rate positions by `TelescopeFrameDetectorToSky` | `--rate-wcs telescope` | built, not by my 5 mas rule | rms against the pipeline's solution 3 to 13 mas, 60 mas for F200W; 0.7 to 2.7 mas after one constant shift per exposure |
| Rendering by `NestedKernelPSF` | `--render-impl torchsky` | no | agrees to 6e-5 to 1.2e-3 of the maximum, but has no read at exposure positions, no kernel rotation, and forces unit sum |

- The position result is better than the rule's verdict suggests: the polynomial distortion from the aperture file is right to 1 to 3 mas, and what is missing is one constant shift per filter (the pipeline applies a filter offset the aperture file does not hold). Next small build: a learned shift per (detector, filter) with one reference filter fixed, started at zero or at the reference value. Then count-rate input needs no calibrated file at all. Until then `--rate-wcs cal` stays (shortcut).
- Shortcuts and limits: velocity aberration neglected (about 0.4 mas); `--item-impl torchsky` does not support the sheared patches used with the pyramid rasteriser in the fast configuration; no real-data fit has run with any switch (needs a new torchsky source archive on CANFAR and torchfits in the job image, both unchecked).
- Findings and three new questions for the library: `claude/torchsky-upstream-notes.md`, section D.
- The package used for CANFAR fits is unchanged (`fieldop_gain`, archive `322a22f9`) until one real-data smoke run with the switches on.

## 17 Nov 2026, 02:20 to 02:50 UTC: relocation dropped; first rate-level fit with learned gain fails on held-out exposures

> **Correction, 07:15 UTC:** the count-rate part of this entry is superseded. The failure was a fault in the fit's starting sky (see the entry of 06:00 to 07:15); the reading about the gain map below was not tested by these fits, and "exp(0.83)" was a misreading. The relocation result stands.

Collection by a Haiku agent; diagnosis by me.

**Items with relocation (`onesky_items004wr`, 3,382 steps, four relocations of 3,000 faint items each): worse than without.** Paired against the same fit without relocation (`onesky_items004w`), no scorer offset: headline +0.286 [+0.087, +0.474]; 20 to 50: +0.044 [+0.019, +0.067]; 50 to 200: +1.66 [+0.32, +3.20]; above 200: +56 [+15, +110]. Against the grid: headline +0.263 [+0.054, +0.485]. Relocation as implemented (faint items moved to where the back-projected residual is positive) is dropped.

**Rate input with a learned per-pixel gain, grid sky (`fieldop_rate_gain_grid`, 8,385 steps): training looks fine (batch score about 0.95) but held-out exposures are predicted very badly.** At rate level, paired against its calibrated twin (`onesky_grid004_dq2`): headline 69.6 against 1.95; by bin 3.1 / 9.4 / 36 / 520 / 7,050 against 1.11 / 1.28 / 1.35 / 3.5 / 182. The fit's own held-out numbers agree (7 to 165 per exposure), so this is the fit, not the scorer.
- On one held-out F277W exposure the prediction is nearly featureless: its 99th percentile is 0.73 DN/s where the data's is 3.7, its correlation with the data over the bright region is 0.24, and the per-exposure gain fitted for it is exp(0.83), where training exposures have 1% scatter.
- The learned gain map is far from the official flat's size: rms of log gain 0.08 per detector, 3% to 6% of it on scales above 16 pixels (official flat after the same polynomial removal: 0.5% to 1.6% on 32-pixel scales), extremes of 3.2 and 0.27.
- The fitted sky does contain the sources (in DN/s per square arcsecond), so the sky was not absorbed wholesale by the gain.
- Reading, not yet established: with one pointing and nine dithers, a free gain per pixel can trade against the scene on every scale above the dither pattern, and a held-out exposure at a new dither position then meets a gain map that encodes part of the scene. The agent's synthetic test had shown the symptom (spurious structure at 1.7 times the noise limit with no injected flat). Whether the failure is the gain map or rate input as such is decided by the control without gain (`fieldop_rate_grid`, running since 02:21 UTC, ends about 05:45).
- By the drop rule (rate fit worse than its calibrated twin by more than 0.05 at signal-to-noise 5 to 50) the learned gain is dropped **for SMACS**. The planned form of the test is on LMC programme 1476, where each detector pixel sees many different pieces of sky, which is what breaks this trade.
- Queue change: the rate fit with items and gain is not run on SMACS. One healthy GPU slice was free at 02:25 (four launches of a second job died on the node with the driver fault).

**Re-fetch at 02:25 UTC.** HST SMACS: complete (116 files, 4.6 GB). Spitzer (10,458 files, 2.56 GB) and GALEX (60): complete. SPHEREx cut-outs: 28 of 718 failed with HTTP 422 (listed in `smacs0723_union/spherex/fetch_failed.csv`; not yet looked at). LMC calibrated: 321 of 680 for programme 1476, programme 4447 not started. LMC count-rate: 279 of 680. JWST union: second programme in progress. No failures in the MAST jobs.

## 17 Nov 2026, 00:30 UTC: four pieces promoted to torchsky; back to the plan; delegation

**Seb, 16 Nov:** keep to the initial plan; delegate trivial tasks to cheap models; why is nothing committed to torchsky? Answer: the work was a prototype in the harness that used torchsky only for the header WCS and the bandpasses and read FITS with astropy. Seb chose all four candidate pieces for promotion, and the LMC count-rate download for the planned form of the flat-field test.

**torchsky, commit `3df2abe5` on main of the local clone `~/src/astroai/torchsky` (1 ahead of upstream, not pushed):** 23 files, 4 new modules with tests, guides, changelog, manifest and reference pages. Built by Sonnet agents from one specification each, integrated by a fifth; checks re-run by me.
- `torchsky.scene.GaussianItemScene`: batched Gaussian items through a WCS, chunked backward pass, pyramid mode. Worst error 2.2e-4 of an item's peak; agrees with the existing `SourceCatalogScene` to 4e-5; 10,000 mixed-width items on 512 x 512: 71 s direct against 2.8 s pyramid (forward and backward, one CPU thread).
- `torchsky.wcs.PolynomialDetectorDistortion`, `TelescopeAttitude`, `TelescopeFrameDetectorToSky`: positions for exposures whose header has no distortion. **Correction to what I told Seb:** a count-rate file carries no distortion model at all, not even in its ASDF extension; the pipeline adds it later from reference files. The new code takes the standard aperture polynomial as tensors (torchsky ships no calibration data). Against the calibrated files' WCS on 12 real NIRCam pairs: 3 to 15 mas median (the plain count-rate header: 87 to 340 mas), except F200W at about 60 mas; almost all of the remainder is one constant pixel offset per filter and detector (0.5 to 2.5 mas median after removing it).
- `torchsky.field.field_observation_from_exposure_file`: one exposure file into a `FieldObservation` through torchfits with an explicit data-quality policy and a sky-box cut that reads only the needed region (a full NIRCam detector in 0.4 s).
- `torchsky.map.operators.NestedKernelPSF`: wide, spatially varying kernels by nested shells and FFT, exact adjoint, PSF conventions respected. Error against direct convolution 4e-5 (point, smooth) and 2e-4 (white noise) of the peak; 0.03 s for a 512-pixel extent on 512 x 512 where a dense convolution takes about 80 s.
- Checks: the four new test files (158 tests), API manifest, docstring ratchet, lazy and cold import, torchfits boundary, device-conversion and docs tests pass with plain pytest on CPU. Lint and format clean with the pinned ruff 0.15.20 (five lint errors that the newer ruff in the workspace had missed were fixed by me). Committed with `--no-verify` because the sandbox has no pre-commit tool. Not run: type check, strict docs build, CUDA, MPS, the C++ extension.
- Open questions and library oddities found on the way: `claude/torchsky-upstream-notes.md` (14 questions, 19 findings).
- Next for this track: make the harness prototype call these four pieces, so that the campaign validates the library code rather than its own copies.

**Plan status (13 Nov plan):** test 1 (any-scale sky form) and test 2 (rendering) done; test 3 (flat field from count-rate images) running on SMACS, to be run as planned on LMC programme 1476 once the count-rate files are in (download started 23:25 UTC, 87 GB); tests 4 (ramps), 5 (predictor across scales) and 6 (depth, surprise map) not started; telescopes: JWST only so far, HST next.

**Delegation from here on:** Haiku for log reading, file counts, job launches and scoring runs; Sonnet for specified builds, integration and figures; Opus for new model code and independent checks only.

## 16 Nov 2026, 23:15 UTC: re-fetch of the deleted data folder started

Seb, 16 Nov: `/arc/projects/ots/torchsky-data` was deleted by mistake by the cleaning team; re-fetch. The manifests and fetch scripts had survived in `adass2026/scripts/`. Four CPU jobs started at 23:14 UTC, all idempotent:
- `refetchlmc` (`scripts/onesky/run_fetch_lmc.sh`): the 1,032 LMC NIRCam `_cal` files of protocol v1 (programmes 1476: 680, 4447: 352; about 121 GB) to `aggregate/{1476,4447}_nircam/`. Manifests rebuilt from `splits_LMC.json`, so exactly the protocol's files.
- `refetchhst` (`run_fetch_hst.sh`): the 116 unique HST exposures of SMACS (`scripts/hst_imaging_unique.csv`) to `hst_smacs0723/raw/`.
- `refetchunionm` (`scripts/union/fetch_union_mast.sh`): JWST `_cal` of all programmes on the SMACS 8-arcminute union (344 files, 37 GB) and 4 extra HST files, to `smacs0723_union/`.
- `refetchunioni` (`scripts/union/fetch_union_irsa.sh`): GALEX (60 files), SPHEREx cut-outs (718), Spitzer IRAC (10,458 files), to `smacs0723_union/`.
- Not re-fetched: the pixel store (derived; `scripts/run_pixelstore.sh` rebuilds it).
- Expect the LMC files, like the SMACS ones, to be a newer calibration than the protocol masks: fits use `--dq-ignore-bits 2`.

## 16 Nov 2026, 22:30 to 23:00 UTC: two crashes fixed, raw-input fits started, where the bright-pixel deficit sits, sharpness

**Two code crashes found by the first runs on real data, both fixed (package `fieldop-gain-322a22f9.tar.gz`):**
- Pixel-gain builder: the gain box is the bounding box of the valid pixels, and a cut's array with invalid margins stuck out of it (never the case in the synthetic tests). Fixed by keeping the overlap; regression test added (7 tests pass).
- Item relocation (`--item-relocate 1`): a CPU random generator was used with a GPU tensor. It crashed at the first relocation, 30 minutes into the fit. Fixed. The relocation path had never run on a GPU.
- The six-minute rate-level smoke run then passed: five gain maps (1.17 M pixels seen in training, 4 to 6 training exposures per pixel), batch score 107 at step 0 falling to 7 after 200 steps, predictions written.

**CANFAR.** One GPU node fails every job at container start ("failed to initialize NVML: Driver/library version mismatch"); six of my launches died there. Two healthy slices take my jobs, so the remaining fits run two at a time. Running since about 22:35 and 22:45 UTC: items with relocation (`onesky_smacs_items004wr_s0`) and rate input with learned gain, grid sky (`fieldop_rate_gain_grid`). Waiting: rate grid without gain, rate items with and without gain, 120,000 items.

**Count-rate files against calibrated files, measured on 12 pairs** (log of rate over calibrated): the per-filter conversion is a factor 0.28 (F090W) to 2.8 (F444W); detectors of one filter differ by 1% to 3%; the flat's structure on 32-pixel scales is 1.0% to 2.9% rms, 0.5% to 1.6% after removing a degree-2 polynomial. So the model's smooth throughput field at its default starting scale (2%) is of the right size, and the per-filter factor goes into the sky's spectrum: a rate-level sky is in instrument units per filter, not in flux units (absolute flux cannot calibrate itself, as stated in the design).

**Where the item sky still loses to the grid sky (signal-to-noise 50 to 200).** Splitting that bin into connected bright regions on four held-out exposures: one bright star carries 96% of the excess in an F090W exposure and all of it in an F277W exposure; in the other two exposures the item sky is level or better. At the nine brightest pixels of those stars the item fit is right to 0.1% (F090W) while the grid fit is 7% low; the loss is in the pixels around the peak (rings and spikes).
- Reading: both sky forms paint part of the telescope's diffraction pattern into the sky around bright stars (spikes are visible in both fitted skies), the grid more freely. With one telescope orientation, as in SMACS, nothing separates "spike on the sky" from "spike in the kernel". This needs exposures at several orientations, not a change of sky form.
- Not a number to chase on SMACS alone.

**Sharpness of the fitted skies** (`flux_check/sharpness.py`; median width at half maximum of the 12 most concentrated sources, against the official mosaic):

| Filter | official mosaic | grid sky 0.04 | item sky, raster 0.04 | item sky, raster 0.02 |
|---|---|---|---|---|
| F090W | 0.127 | 0.084 | 0.108 | 0.093 |
| F200W | 0.128 | 0.092 | 0.106 | 0.089 |
| F277W | 0.233 | 0.153 | 0.097 | 0.085 |
| F444W | 0.264 | 0.199 | 0.102 | 0.087 |

- The item sky shows the same widths in all six filters: 0.40 of the mosaic's at F444W (0.33 at the 0.02 raster). The grid sky keeps wavelength-dependent widths (0.74 of the mosaic's).
- **This is an assumption of the item form, to be stated:** one item has one position, size and orientation at every wavelength and only its spectrum varies, so the sharp short-wavelength exposures set the shapes that the long-wavelength exposures are then asked to confirm. Colour gradients need several overlapping items. The support for it is predictive only: the long-wavelength held-out exposures are predicted as well as or better than by the grid sky (-0.094 [-0.238, +0.060] in round 2) and better than by the pipeline. A source whose shape truly changes with wavelength inside one item's footprint would be misrepresented.
- Approximations: nearest-pixel peaks, no deblending, sources are compact galaxies and stars together, widths are limited by the raster (smallest 0.053 and 0.028 arcsec).

## 16 Nov 2026: round 3 collected; the "faint-pixel bias" was the scorer's; raw-input build ready

The session stopped on 14 Nov at about 06:20 UTC (weekly usage limit) and resumed on 16 Nov at 21:05 UTC. The six fits launched before the stop all finished.

**1. Reference confirmed on the protocol's own mask.** The grid fit that keeps the saturated-flagged cores (`onesky_grid004_dq2`, `--dq-ignore-bits 2`) scores 1.73 [1.65, 1.81] against 1.75 for `fieldop_fix_200`. From here on the plain protocol mask is used, with fits made with `--dq-ignore-bits 2`.

**2. Item sky, round 3** (60,000 items, any-size rasteriser with sheared patches, coarse buckets, 32 M chunks; 200 minutes). Paired on identical pixels, scoring each method's full prediction (no scorer offset), 68% intervals:

| Fit | steps (s per step) | headline against grid | headline against official pipeline | 1 to 5 | 5 to 20 | 20 to 50 | 50 to 200 | above 200 |
|---|---|---|---|---|---|---|---|---|
| items, raster 0.04, cap 0.32 (`items004c`) | 4,428 (2.7) | +0.043 [-0.101, +0.204] | not run | -0.136 | -0.181 | -0.160 | +4.07 [+2.82, +5.71] | -26 [-54, +1] |
| items, raster 0.04, cap 2.56 (`items004w`) | 2,983 (4.0) | -0.024 [-0.120, +0.097] | -0.133 [-0.270, -0.005] | -0.139 | -0.183 | -0.176 | +3.33 [+2.39, +4.64] | -31 [-56, -10] |
| items, cap 2.56 + smooth grid 0.64 (`items004ws`) | 3,665 (3.3) | +0.336 [+0.183, +0.519] | not run | -0.143 | -0.177 | -0.086 | +7.1 [+5.0, +10.1] | -3 [-29, +21] |
| items, raster 0.02, cap 2.56 (`items002w`) | 2,494 (4.8) | -0.047 [-0.120, +0.042] | -0.156 [-0.311, -0.014] | -0.132 | -0.177 | -0.146 | +2.96 [+2.12, +4.07] | -33 [-59, -17] |
| items, raster 0.01, cap 2.56 (`items001w`) | 1,193 (10.1) | worse (2.56 against 1.71) | not run | | | | | |

Bin columns are item minus grid (`onesky_grid004_dq2`: 1.09, 1.25, 1.32, 3.23, 112 in those bins). Against the official pipeline, `items004w` by bin: below 1: -0.047; 1 to 5: -0.141 [-0.146, -0.135]; 5 to 20: -0.103 [-0.117, -0.091]; 20 to 50: -0.095 [-0.148, -0.055]; 50 to 200: -0.14 [-1.64, +1.20]; above 200: -10 [-30, +11]. The grid against the pipeline: -0.109 [-0.293, +0.029] headline, worse from 5 to 50 (+0.08), better from 50 to 200 (-3.5).
- With wide items allowed the item sky is ahead of or level with the official pipeline in every signal-to-noise bin, and level with the grid on the headline. 20% of the items end wider than 0.32 arcsec (carrying 29% of the summed coefficients), 2.5% at the 2.56 arcsec cap.
- The bright-core deficit against the grid (50 to 200) remains: +3.0 to +3.3, from +5.9 in round 2. Above 200 the item sky is better than the grid.
- The smooth grid makes things worse (kernel reference moment 1145e-6 arcsec^2 against the floor 533e-6): dropped.
- The larger chunks and coarser patch ladder halve the step time (2.7 s against 5.2 s at the same cap). Wide items cost more (4.0 s); a 0.01 arcsec raster is too slow for a 200-minute fit.
- Flux scale of `items004w` (sky over official mosaic, brightest galaxy): 0.98 to 1.00 inside 1 arcsec in all six filters, 1.00 to 1.09 inside 0.3 arcsec, 0.93 to 1.00 box total. `items002w`: 1.00 to 1.05 and 1.03 to 1.16. `items004c`: 1.00 to 1.09.

**3. The "faint-pixel bias" of the item sky was an artefact of the scorer's background offset.** My explanation of 14 Nov (items cannot carry diffuse light) was wrong: wider items changed nothing (below 1: 0.86 with the offset, 1.15 without, for every item fit).
- The protocol fits its offset on the 2,000 scored pixels with the lowest signal-to-noise in the official mosaic. That mosaic is built from all exposures, the held-out one included, so those pixels are the ones where the held-out exposure's own noise is low. A model that does not follow the noise is then given a negative offset; a model that follows the training exposures' noise (the grid) is not.
- Check (five held-out exposures): median (data - model) / sigma on the faintest 2% of pixels ranked by the official mosaic: -0.38 to -0.57 for the item fit and -0.13 to -0.21 for the grid fit in the short-wavelength exposures. Ranked instead by the other method's prediction, which shares no noise with the held-out exposure: +0.13 to +0.17 for both, with no trend across the whole range.
- Fix (mine): `score_paired.py --offset-rank METHOD --offset-rank-smooth 1`: the offset pixels are the 2,000 with the lowest value of a training-only prediction, box-averaged over 1 arcsec so that it carries almost none of the training exposures' pixel noise either (an unsmoothed training mosaic biases the other way: it costs the pipeline 0.05 on the five-exposure sample). On that sample the scores with the new offset equal those without any offset to 0.01 for all three methods (item 1.392 and 1.390; grid 1.854 and 1.848; pipeline 1.678 and 1.680). Full set (24 held-out exposures): with the new offset the headline is 1.691 for the wide-item fit, 1.719 for the grid and 1.813 for the pipeline, against 1.689, 1.713 and 1.822 with no offset; item minus pipeline -0.121 [-0.261, +0.006], item minus grid -0.028 [-0.127, +0.092]. Output: `protocol/v1/scores_paired/SMACS/offrank/`.
- Consequence: the protocol's headline "with offset" form understates the item sky by 0.14 (1.83 against 1.69) and should not be used to compare smooth models with noise-following ones. All conclusions of 14 Nov drawn from the "with offset" column are withdrawn; those from the "without offset" column stand.

**4. Raw inputs: rate-level fits with a learned per-pixel gain (Opus agent; unit tests re-run by me, 6 pass).** Folder `fieldop_gain/`, report `RATE_GAIN_REPORT.md`, package `fieldop-gain-a7950e8c.tar.gz`.
- `--pixel-gain {off,detector,detector-filter}`: a log-gain map per detector over the pixels seen in training, multiplying sky signal and sky background (not the detector pattern or stripes); gauge by exact projection (zero mean and zero polynomial part to degree 2 over the observed pixels), no prior; `off` is bit-identical to the item package.
- `_rate` files: DN/s; ERR without flat-field variance; no AREA extension; **the `_rate` header WCS has no distortion correction** (0.07 arcsec median, 0.24 arcsec maximum from the `_cal` WCS), so positions come from the matching `_cal` header (`--rate-wcs cal`): a stated shortcut, not yet raw.
- Rate-level scoring (`score_paired.py --level rate`): calibrated-unit predictions are converted with the per-pixel ratio rate/cal of the same exposure (exact to 4e-16). At rate level the scores are 0.03 higher (pipeline 1.251 to 1.280 on 12 exposures, signal-to-noise 5 to 50): compare rate fits with calibrated fits at the same level, paired.
- Synthetic test (3% injected flat): the gain map removes 79% of the excess on background pixels (2.85 to 1.48; 1.105 without flat error) and recovers the injected map with correlation 0.92 and 0.89 against 0.98 and 0.95 at the noise limit. With no injected flat it learns spurious structure at 1.7 times the noise limit and costs +0.22 on background pixels: cause not established.
- Expectation on SMACS: marginal. Each detector pixel is seen in 4 to 6 training exposures at background signal-to-noise 9 to 31; the official flat's pixel-to-pixel rms is 0.02 to 0.04 against a noise limit of 0.017 to 0.043 per pixel. Best possible correlation 0.6 to 0.7; expected about 0.45 to 0.5. The drop rule is a correlation below 0.3.
- Not yet run: any real fit. The command-line path for rate input needs torchsky, absent from the cloud workspace; a six-minute smoke run is queued.
- Noted by the agent: about 10% of pixels carry the cosmic-ray jump flag and are excluded under every current setting.

**5. Figures (Sonnet agent; looked at by me):** `figs_round2/`: fitted skies beside the official mosaic, enclosed flux over mosaic against radius, kernel profiles and encircled fractions, item sizes. Notes and approximations in `FIGS_NOTES.md`.

**CANFAR, 16 Nov 21:45 UTC.** The first rate-level smoke run failed at container start with a GPU driver error on the node ("failed to initialize NVML: Driver/library version mismatch"), before any code ran; relaunched as `onesky_smacs_rate_smoke2_s0`, queued. The four 200-minute rate fits wait for it. Queued on the tested item package (`fieldop-items-a542655b.tar.gz`), both aimed at the bright-core deficit: `onesky_smacs_items004wr_s0` (wide items plus `--item-relocate 1`: faint items moved to where the back-projected residual is positive) and `onesky_smacs_items004w120_s0` (120,000 items). The account was running 14 other jobs, so mine wait their turn.

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
- **The gain from 1 to 50 disappears when the scorer adds its own background offset** (the protocol's headline form; the offset is the clipped median residual over the faintest pixels). So the item fit is inconsistent between the faintest pixels and the rest by roughly 0.45 noise units: a real defect for faint diffuse light. Likely cause: 27% of the items sit at the width cap (0.32 arcsec) and 5% have both widths there, so the items cannot carry diffuse light and the per-exposure background compensates on average. Round 3 tests this. Sign and size, from the scorer's per-exposure offsets: the item fit predicts the faintest pixels too high, by 0.008 MJy/sr in the short-wavelength exposures (median; up to 0.017) and 0.0015 in the long-wavelength ones; the grid fit by 0.002 and 0.0001.
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
