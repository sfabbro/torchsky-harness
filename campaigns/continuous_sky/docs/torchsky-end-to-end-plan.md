# Plan: one end-to-end path in torchsky, from exposure files to sky, uncertainty and fitted response

Sources I read: torchsky `AGENTS.md` and the upstream notes; the audit (sections B to D, ranked findings, F3); the 17 and 18 Nov log entries; `DEFAULTS_V2`, `BUILD_2026-11-18B`, `PERF_SIMPLIFY`, the truth report (sections 1 to 3); and the closing `fit done … tau` and `sizes of the learned terms` lines of the 13 finished 200-minute fits in `/home/claude/work/checkin/`. I did not modify anything.

**Terms used below:**
- **Pruned:** the learned scale of a term ended at about 1e-6, or the term was off.
- **Median rule:** the current zero-level convention. For each filter, the training exposures' background constants average to each exposure's data median.
- **Masked rule:** the alternative zero-level rule. It failed on real data.

The model being fitted, for exposure $e$ and pixel $p$:

$$
d_{e,p} = g_e\,\frac{1}{Q}\sum_q T_{i,f}(\lambda_q)\,\big[k_{c}(\cdot;\lambda_q,z_e) * s(\cdot,\lambda_q)\big]\big(A_e(x_p)\big) + B_e(x_p) + \text{stripes}_e(p) + n_{e,p}
$$

$$
\sigma_{e,p}^2 = \big(\sigma^{\rm pipe}_{e,p}\,e^{h_{d,f}}\big)^2 + (f_c\,S_{e,p})^2
$$

with $s(\Omega,\lambda) = \sum_n \sum_k c_{nk}\,\phi_k(\lambda)\,G(\Omega - x_n;\Sigma_n)$.

---

## 1. The minimal model

Fits read: 13 finished 200-minute fits — 9 SMACS JWST-only, 2 SMACS JWST+HST, 2 LMC tile 7. 11 of the 13 used the median rule.

| Component | Decision | Evidence (one line) |
|---|---|---|
| Sky as positive Gaussian items (log fluxes, width floor, data-blind seeding on 3 lattices) | **keep, only form** | All 9 SMACS item fits beat the official pipeline at signal-to-noise 1 to 50 (−0.12, −0.08, −0.08), repeatable to 0.01. In the truth problem the item error is 0.64 of the ideal, against 0.97 for the grid. Items pin one position per source across filters; a grid sky cannot. |
| Spectral basis: 12 bumps uniform in log wavelength (partition of unity), range from the union of loaded bandpasses | keep | One sky fits JWST and HST (throughput difference −4.3% to −5.1%). On the synthetic field, a filter never trained on scores 4.35 with the second instrument against 9.09 without. torchsky already has the same basis: `LogLambdaRBFBasis`. |
| Kernel network per (instrument channel, wavelength), with nested-level rendering, render jitter during training, per-exposure rotation and spline read-back | keep | This is the response itself. |
| Per-exposure kernel code | keep | Kept by every fit (scale 0.016 to 0.061). |
| Kernel variation across the detector, with time, and extra variation with wavelength | **drop** | Pruned in every median-rule fit. Across the detector it ended at its cap of 1.5 in the two masked-rule fits, which were the worse fits. *Evidence comes only from 20-arcsecond boxes; re-test at full-detector size.* |
| Throughput per (instrument, filter), smooth in wavelength, with a reference-instrument gauge | keep | WFC3/IR against NIRCam: −5.1% (archive positions), −4.3% (cross-correlation start). |
| Throughput field across the detector, and with time | drop | Pruned in every fit. |
| Per-exposure gain (one number per exposure, with a prior) | keep | Kept, but small: 0.1% to 0.15% rms in log gain for items, 0.5% to 0.6% for grid and count-rate fits. *Thin evidence; it is cheap, so I keep it.* |
| Header WCS plus per-exposure affine (6 numbers) | keep | Translations 5 mas rms (JWST) and 276 mas (HST, catalogue-free). Matrix 2e-4 rms. |
| Coarse cross-correlation registration for non-reference instruments | keep | 74 of 74 HST exposures registered in 1.7 s. Agrees with the archive's Gaia-aligned solution to 15 mas (x) and 9 mas (y). |
| Static and drifting distortion fields, and per-exposure distortion code | drop | Static and drifting: pruned in every median-rule fit. The per-exposure code ends at 0.2 to 0.4 mas, negligible. |
| Per-exposure background: constant plus quadratic shape terms | keep | Shape terms 0.45 sigma rms (max 2.3). |
| Detector pattern on an 8 × 8 grid | drop | Beyond degree 2 it is 0.01 to 0.05 sigma, and adds at most 0.035 to held-out scores (flat-field diagnosis). |
| Row and column stripes, declared by the instrument (NIRCam yes) | keep | 0.17 to 0.22 sigma rms, maximum 1.2 to 1.7 sigma, in training and held-out exposures alike. |
| Noise scale per (detector, filter), bounded to a factor of 2 | keep | NIRCam 0.95 to 1.02; WFC3/IR 0.81 to 0.91, which matches the file check (its error planes are 10% to 20% too large). Required for an honest uncertainty. |
| Fractional noise floor, bounded at 0.2 | keep | Ends at 4% to 10% in every SMACS fit; at the bound in the under-fitted LMC item fit. Read it as a misfit indicator, not as physics. |
| Huber loss (delta 3) | keep | ACS blank-sky scatter is 1.12 to 1.22 of its error plane (unflagged cosmic rays). |
| Gauges: reference-instrument throughput; common shift (or a catalogue tie); zero level by the median rule; blur convention with guard; kernel flux within 2 arcsec | keep | Median rule: the masked rule biased long-wavelength flux by +8% to +17% in both fits that used it. Normalisation radius: items keep less than 0.4% of kernel light beyond 2.5 arcsec, so it changes little. One convention is needed for joint fits. *Evidence thin; I pick 2 arcsec everywhere and watch the blur dilation (it reached 2.38 once).* |
| Fit machinery: Adam groups, 5 patches per step with importance sampling, 2 sky-only inner steps, registration pre-pass, damped expectation-maximisation updates of the random-effect scales | keep | Needed to reproduce the prototype. Dropping the inner sky steps (about 30% of an item step) is a later, measured change. |
| Cross-fitted prediction of held-out exposures (per-exposure terms fitted on one half of 4 × 4 pixel blocks, scored on the other half) | keep, as the prediction method | Removes 2% to 3% of score flattery on real data. |

**What the end-to-end path does not carry:**
- The grid sky and its preconditioner and positivity penalty.
- The masked zero rule, the zero-level refresh and the positivity margin.
- Item relocation and the smooth grid under items (already removed).
- The variability evidence machinery: validation dither groups, gradients of the variability scales, likelihood-ratio checks, caps, the at-cap reports. Every axis it governs, other than the random effects, is pruned anyway.
- The pruned and dropped fields themselves: kernel across the detector, with time and with extra wavelength dependence; throughput field across the detector and with time; static and drifting distortion; per-exposure distortion code.
- The detector pattern and the gated non-linearity.
- The count-rate path, unit options and pixel gain map; the learned per-(detector, filter) shift; the telescope-frame positions.
- Cross-fitting of the sky, already removed.
- Several tiles, sky swapping and pinned-memory residency (deferred, not dropped).
- `--legacy-defaults`, the archive-WCS shortcut, `--heldout-only`.
- The command-line diagnostics: wing fractions, nuisance-size tables, the per-step input-noise score. These become optional fields of a fit report.
- Per-pixel quadrature rendering, Keys interpolation, the `direct` item rasteriser (`pyramid` becomes the only path).
- About 160 of the 183 command-line options. About 15 configuration fields remain.

---

## 2. The end-to-end interface: 9 public names

New package `torchsky.map.exposure_fit`, re-exported from the `torchsky.field` facade.

```python
InstrumentDescription        # dataclass: name, diameter_m, pixel_scale per channel, detectors, channel_of,
                             #   DQPolicy (own bit table), detector_terms ("stripes", "amp_offsets"), science-plane unit
                             #   and factor to MJy/sr (from header cards), bandpass id, HDU layout (names, EXTVER per chip)
instrument_description(name: str) -> InstrumentDescription      # "NIRCAM", "WFC3_IR", "ACS_WFC"; lives in torchsky.instrument

load_exposures(paths, instruments, sky_box, *, dq_policy=None) -> list[FieldObservation]
    # one per (file, chip), cut to sky_box through torchfits; sigma, mask, DQ plane, MJy/sr factor,
    # detector/instrument/band/MJD in metadata. Application boundary: the only function that opens files.

ReferenceCatalogue           # dataclass of tensors: ra, dec, epoch, optional pm, optional flux per band,
                             #   is_point flag, uncertainties
SkyFitConfig                 # dataclass, ~15 fields: sky_pitch, item_count, item_floor, spectral_k, kernel_extent,
                             #   kernel_norm_radius=2", background_order=2, noise bounds, reference_instrument,
                             #   budget (steps or minutes), patches_per_step, patch_size, seed, device
fit_sky(observations, config, *, catalogue=None, init=None, log=None) -> SkyFit
SkyFit                       # nn.Module: .sky (items + basis), .response (kernels, throughputs, noise terms),
                             #   .exposure_terms (table per training exposure), .config, .report;
                             #   .save(path) / SkyFit.load(path) (torch state + JSON config, via map.field.artifacts)
SkyProduct                   # dataclass: image [B,H,W], std [B,H,W], header (tangent-plane WCS dict), bands,
                             #   resolution_fwhm_arcsec per band, unit "MJy/sr", zero_level_convention + its std, valid mask
predict_exposure(fit, observation, *, cross_fit=True) -> ExposurePrediction   # NamedTuple: model, sigma, terms, chi2
```

`SkyFit.product(bands, pitch_arcsec, resolution="native" | fwhm_arcsec, region=None) -> SkyProduct` is a method, not a separate name.

**What comes out:**
- **Sky image and standard deviation:** `fit.product(...)`. The sky is the items convolved with a stated Gaussian target beam and sampled at the stated pitch; the uncertainty uses the same target beam.
- **Fitted response:** `fit.response`, with documented readers:
  - `kernel(instrument, band, at=None, exposure=None) -> stamp`
  - `throughput(instrument, band) -> curve`
  - `noise_terms() -> table`
- **Prediction of any exposure:** `predict_exposure`. It fits only that exposure's own terms (background, affine, gain, kernel code, stripes), on one half of the pixels, and predicts the other half.

**Where catalogues enter (all optional):**

| Input | Effect | Without it |
|---|---|---|
| Positions of stars (Gaia) | Ties the absolute frame. Replaces the common-shift gauge, so the absolute translation is measured, not fixed to the mean of the exposures. | Frame = mean of the reference instrument's header positions. |
| Positions of any sources | Starting translations for exposures far off. An alternative to cross-correlation, used when cross-correlation refuses an exposure. | Cross-correlation against the starting sky. |
| `is_point` sources | Seeded as items fixed at the floor width. Point sources pin the kernel core, so the blur convention stops being the only thing that fixes sharpness where stars exist. | Blur convention alone. |
| Star fluxes per band | Default: a reported check of the throughput scale. Optional: an anchor that replaces the "reference instrument equals its published curve" gauge. | Published curves of the reference instrument. |

**Not intermediate products:**
- No resampled exposure anywhere.
- No coadd in the likelihood: the likelihood is a sum over exposure pixels only.
- Per-exposure terms are fitted parameters, not products.
- Nothing is written between loading and the fitted state.

**The starting coadd** (inverse-variance splat at header positions plus a ridge spectral start) survives only until the equivalence check of step 6 passes. Then it is replaced by the back-projection of the model's own operator: weighted adjoint of the data divided by the weighted adjoint of ones. That is computed in memory, uses no separate resampling code, and is thrown away after seeding the items. The cross-correlation reference sky is replaced the same way. A catalogue can seed items in addition.

---

## 3. Where each piece lives

| Piece (prototype source) | torchsky home | Reuse or replace |
|---|---|---|
| Loader and instrument descriptions (`data.py`, `instruments.py`) | `torchsky/instrument.py` (descriptions, DQ bit tables); `map/field/exposure_io.py` (extended) plus `load_exposures` in the new package | Reuse `field_observation_from_exposure_file`, `DQPolicy`, `filter_profiles` (SVO). Output type is `FieldObservation`; no new observation class. |
| Item sky (`sky.py` items part) | `scene/gaussians.py` gets a functional, pixel-grid entry. Item parametrisation and seeding go in a private `exposure_fit/_items.py`. | Reuse `GaussianItemScene` (pyramid mode) and `LogLambdaRBFBasis` from `map/field/chromatic.py`. |
| Renderer (`render.py`) | New private operator `map/operators/_exposure_render.py`: sky grid to exposure pixel positions, kernel rotated per exposure, region kernels blended by output pixel, free amplitude and normalisation radius | Answers question D17: a second operator, sharing the level and pooling helpers of `NestedKernelPSF`. `NestedKernelPSF` stays a detector-plane operator. |
| Kernel network (`kernel.py`) | Private `map/operators/_kernel_field.py`, following `NestedKernelPSF`'s `kernel(offsets, condition)` contract | Not `ConditionedKernelPSF`: different encoding, and the warm start is needed. |
| Response: throughput, gain, affine, background, stripes, noise, gauges (`model.py`) | Private `exposure_fit/_response.py` and `_gauges.py` | Reuse `polynomial_background` if its basis can be Legendre on detector coordinates. The noise floor is fractional in signal, which `PipelineVarianceCalibration` (absolute floor) does not cover, so it is new. |
| Fit loop, pre-pass, expectation-maximisation, cross-fitted prediction (`train.py`); cross-correlation (`coarse_reg.py`) | Private `exposure_fit/_fit.py`, `_register.py`, `_predict.py` | Reuse `map/field/artifacts.py` for saving. |
| Uncertainty (`fieldop/uncertainty.py`, being built elsewhere) | `exposure_fit/_uncertainty.py` behind `SkyFit.product` | Port after its validation (step 8). |
| Command line (`cli.py`) | `torchsky/cli` subcommand `sky-fit`; writes products with torchfits | Replace. |
| Not used in version 1 | `telescope_frame.py`, `NestedKernelPSF` in the fit | Kept for the count-rate path. |

**What torchsky must change first** (references are to the sections of the upstream notes):
- **DQ fixes:** A1 (`reject_bits=None` and `=0` mean opposite things) and A2 (signed int32 DQ planes). Add the bit tables to `torchsky.instrument` (C4).
- **Loader fixes:**
  - D7: return the DQ plane plus `DETECTOR`, `INSTRUME`, `AREA`; no zero-filling.
  - B4: parse HST `FILTER1`/`FILTER2`.
  - A8 and C2: an enumerated `value_semantics`, including surface brightness.
  - A7 and C6: the crop passes `fail_closed=False`.
  - A5: `.to(dtype)` must not cast float64 transforms.
- **Item scene:** a functional form (D2, D16), a pixel-grid entry (D3), a patch-size cap (D4) and `pyramid` as the default mode (B1).
- **torchfits 1.3.1 (separate repository):** `EXTVER` reading and a usable error for an unknown extension name. ACS/WFC needs both of its chips.
- **Housekeeping:** the API manifest gets `torchsky.map.exposure_fit`, and the import-order lint failures in the touched `__init__` files are fixed (A13, A19).
- **Deferred with the count-rate path:** A3 (a learnable attitude is silently frozen).

---

## 4. Order of work

**Oracles.** Bit-identical state hashes cannot survive a re-implementation: operation order changes and the fits are chaotic over hundreds of steps. So there are two oracles:
- **Golden step:** a frozen state and batch dumped by the prototype. torchsky must reproduce its loss and every gradient to 1e-5 relative.
- **End-of-fit statistics:** on the small synthetic field and the truth problem, within the seed scatter.

The prototype's recorded hashes pin the oracle runs themselves.

| # | Scope | Done when | Agent | Lines | Parallel |
|---|---|---|---|---|---|
| 1 | **Prototype: a "minimal" preset and the oracles.** A `--minimal` preset that switches off everything in the "not carried" list. Record its state hash on the 150-step item synthetic run and on the truth problem (8 exposures, seeds 0 to 2). Write golden-step dumps for 3 configurations (single instrument, two instruments, with stripes). Launch 2 seeds of a 200-minute SMACS item fit with the preset. | The minimal SMACS fits score within 0.02 of the reference fits at signal-to-noise 5 to 50, with flux over the mosaic at 0.98 to 1.01 and bright bins within the measured scatter (1.59 to 1.77 headline). **If this fails, the minimal model is wrong; stop and revise section 1.** | mid; cheap to launch and collect | 200 | with 2 |
| 2 | **Library prerequisites** (section 3 list) | Each fix has a regression test that fails before and passes after; existing map tests pass | cheap (mechanical), mid (scene functional form) | 450 | with 1 |
| 3 | **Instruments and loading:** `InstrumentDescription`, `instrument_description`, `load_exposures` | On the 12 NIRCam, 5 WFC3/IR and 1 ACS sample files: pixels, masks, sigma, MJy/sr factors and positions (to 1e-6 arcsec) equal the prototype loader's; repeat the `check_instruments_real` unit ratios | mid | 500 | with 4, 5 |
| 4 | **Item sky and spectral basis** through the functional `GaussianItemScene` | Raster to 1e-6 of the maximum and gradients to 1e-5 against the prototype; seeding identical for the same random generator | mid | 300 | with 3, 5 |
| 5 | **Renderer and kernel network** (critical path) | Golden-step forward values and gradients to 1e-5; exact adjoint test; agreement with direct convolution as in the prototype tests; one GPU timing within 10% of the prototype | **strongest** (design against `NestedKernelPSF`), mid for tests | 900 | with 3, 4 |
| 6 | **Response, noise, loss and gauges** | Full golden step (loss and all gradients) to 1e-5 in the 3 configurations; each gauge projection checked to round-off | **strongest** | 700 | after 4, 5 |
| 7 | **Fit loop, registration, `fit_sky`, `SkyFit`, `predict_exposure`.** **This is the first end-to-end result in torchsky:** single instrument, calibrated input, one tile. | Synthetic and truth problem, 3 seeds each side: held-out chi-squared within 1% of the step-1 oracle; flux ratios within 2%; empty-sky offset inside its error. One real SMACS 200-minute GPU fit through `load_exposures` within 0.02 of step 1 at signal-to-noise 5 to 50. Then swap the starting coadd for the back-projection and repeat within the seed scatter. | mid from a written specification, strongest reviews | 900 | after 3, 6 |
| 8 | **Uncertainty and product:** port `uncertainty.py` into `SkyFit.product` | Truth problem: (recovered − true)/std has rms 0.9 to 1.1 in empty regions and for source fluxes at signal-to-noise 3 to 100. Real data: two fits on disjoint halves of the exposures, difference/√(std₁² + std₂²) with rms 0.9 to 1.2 per band | strongest (validation design), mid (port) | 400 | port can start on prototype states during 6 and 7 |
| 9 | **Second instrument:** WFC3/IR joint fit with cross-correlation start; reference-instrument gauge | Joint SMACS fit: WFC3/IR throughput −4% to −5%; JWST fluxes 0.99 to 1.01; held-out JWST at signal-to-noise 5 to 50 within 0.02 of the prototype's joint fit | mid | 300 | with 10, after 7 |
| 10 | **Catalogue inputs,** then command line, documentation, changelog, manifest; retire the prototype | Synthetic: an injected frame offset of 100 mas recovered to under 5 mas with a star catalogue. SMACS: absolute offset against Gaia reported. Docs contract and API manifest checks pass. The prototype is retired after 7, 8 and 9 pass. | mid; cheap for docs and manifest | 400 | with 9 |

**Shortest path to a first result inside torchsky:** step 1 and step 2 (only the loader and scene items) → steps 4 and 5 in parallel (3 alongside) → 6 → 7. That gives a slow, NIRCam-only, calibrated, one-tile fit from files with no uncertainty yet. About 3,500 lines including tests.

**Deferred:**
- Count-rate input, the learned flat field, telescope-frame positions and the per-(detector, filter) shift. Decided by the LMC programme 1476 test.
- Several tiles and large areas.
- Spatially varying kernel and throughput: re-test at full-detector size.
- ACS beyond loading; GALEX, Spitzer and ground-based data.
- Speed work.

---

## 5. Risks and open decisions for the owner

1. **Items as the only sky form.** On LMC the grid was better at signal-to-noise 20 to 200 (5.9 against 8.3), but with 3.6 times more steps. Items win at the faint end, pin positions across filters, and lose 20% to 30% less flux than the grid on the truth problem. Positive fluxes bias empty sky upward (rectified noise) and make the uncertainty there asymmetric. **Recommend:** items only; carry the faint-end bias as a documented limit.
2. **Count-rate path and learned flat out of version 1.** The count-rate fit is +0.55 worse than its calibrated twin in the headline. The flat does not pay with 4 to 6 exposures per pixel. **Recommend:** out; decide on the LMC programme 1476 result.
3. **Zero level.** The median rule ties the zero to data that contain sky light. The masked rule biased fluxes on real data. A 20-arcsecond box in a cluster core has no empty sky for the coarser instruments. **Recommend:** keep the median rule as a stated convention. Report a per-band zero-level standard deviation in `SkyProduct`. Fix it once tiles reach blank sky, not by another rule on one box.
4. **What "highest resolution achievable" means.** **Recommend:** the product is the sky convolved with a stated Gaussian target beam per band. "Native" means the item floor width (0.4 × sky pitch), the same for all bands. The finest beam is the smallest one for which the half-split uncertainty test of step 8 passes; publish that number per band rather than claim more. The blur convention, which ran away in the truth problem though it stayed between 1.0 and 1.33 in real fits, is what gives "native" a meaning. Stars from a catalogue are the remedy where they exist.
5. **Uncertainty method and its validation on real data.** A curvature (Laplace) estimate from one fit misses the fit-to-fit scatter: 0.17 in the headline on bright pixels. The fractional noise floor (4% to 10%) also inflates the stated error of bright pixels. **Recommend:** curvature estimate plus at least 2 seeds; ship no product until the disjoint-half test passes per band.
6. **Speed.** About 200 GPU minutes per 20-arcsecond tile. LMC needed the whole budget on one tile to beat the pipeline. **Recommend:** parity only in version 1. Then measured changes, each checked against the oracles:
   - drop the sky-only inner step (about 30% of an item step);
   - half-precision fast Fourier transforms in the backward pass (20% to 30%);
   - share kernel evaluations between patches.
7. **Pruning evidence comes from small boxes.** Every "drop" for a field across the detector or for distortion was measured on 20-arcsecond boxes. **Recommend:** keep the slots in the response design, switched off, and re-test when several tiles arrive.
8. **When to retire the prototype.** **Recommend:** after steps 7, 8 and 9 pass against the step-1 oracles (golden step to 1e-5; end-of-fit statistics within the seed scatter). Freeze the prototype at the step-1 archive and make no further model work there; otherwise the oracle moves.

---


### Critical Files for Implementation
- /home/claude/work/fieldop_tsky/fieldop/model.py
- /home/claude/work/fieldop_tsky/fieldop/render.py
- /home/claude/work/fieldop_tsky/fieldop/train.py
- /home/claude/torchsky/src/torchsky/map/field/exposure_io.py
- /home/claude/torchsky/src/torchsky/scene/gaussians.py