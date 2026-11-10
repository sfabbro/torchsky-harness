# PSF and geometry operator: audit, probes and literature (6 Nov 2026)

## What was actually tested

A0/A1 (v33) did not test a learned continuous PSF+geometry field. The field sketched in `claude/forward-operator-spec.md` (A2) has not been built.

v33 is the old explicit pipeline with two changes:
- the kernel field is evaluated in detector coordinates and resampled onto the 0.04″ sky grid once per (band, roll) group;
- the kernel is averaged over the detector pixel box.

The gate failure therefore says nothing yet about the continuous operator.

## Main finding: the learned "PSF" is not the instrument PSF

The PSF and the sky are degenerate. The fit sits in the basin where the sky holds the PSF and the kernel is only a relative kernel.

**Probe 1: learned kernels in the detector frame, with pixel box (B run, `v33_A0_jwst_wfc3ir_3d_detframe`).**

Encircled energy in a circle, learned vs the WFC3 Instrument Handbook (Table 7.6):

| Band | r = 0.13″ | 0.26″ | 0.39″ | 0.52″ |
|---|---|---|---|---|
| F105W learned | 0.985 | 0.993 | 0.991 | 0.994 |
| F105W handbook | 0.577 | 0.811 | 0.852 | 0.882 |
| F160W learned | 0.896 | 0.991 | 0.990 | 0.993 |
| F160W handbook | 0.494 | 0.736 | 0.833 | 0.855 |

In the JWST-only run (`v33_A0_jwst_cal_3d_detframe`):
- The short-wavelength (SW) kernels are near delta functions: EE(0.05″) is 0.98–0.99 for F090W, F150W and F200W.
- The long-wavelength (LW) kernels are narrow relative kernels: F444W EE(0.1″) = 0.93.

**Probe 2: star profiles in the fitted sky vs the drizzle coadd** (bright star at grid (131, 30); EE at 0.04, 0.08, 0.13, 0.26″ over EE at 0.6″):

| | F150W sky | F150W drizzle | F444W sky | F444W drizzle | F160W sky (B) | F160W drizzle (B) |
|---|---|---|---|---|---|---|
| EE | 0.38 0.53 0.73 0.89 | 0.36 0.52 0.71 0.87 | 0.37 0.69 0.77 0.91 | 0.16 0.39 0.73 0.87 | 0.30 0.48 0.70 0.88 | 0.11 0.23 0.45 0.82 |

Three more stars in the B run give the same picture.

The fitted sky has the resolution of the NIRCam SW drizzle. It is not deconvolved. The model behaves as a coadd at SW resolution plus learned relative kernels:
- SW kernels ≈ identity;
- LW kernels ≈ (LW PSF) ⊗ (SW PSF)⁻¹;
- the WFC3/IR kernel is far too narrow.

**Why the fit lands there:**
1. **Initialisation.** The sky starts from the drizzle coadd (`--init drizzle`), which already contains each band's PSF and the drizzle kernel. The PSF starts as a Gaussian core with the nominal FWHM, and narrower still with the pixel box (`box_adjusted_sigma`). It has no wings.
2. **Window.** The kernel window is 31 × 0.04″ = 1.24″. The real wings carry 12–15% of the WFC3/IR flux beyond 0.6″ (handbook), and they cannot exist in the kernel.
3. **Weak spatial modulation.** The residual basis kernels are multiplied by 0.01 in code, and the mixing-weight network starts at zero. The PSF can barely vary over the detector.
4. **Learning rate.** The PSF learning rate is 5e-4 against a free sky. The sky absorbs everything faster.
5. **One network for all telescopes.** ChromaticPSF is a single network for NIRCam SW, NIRCam LW and WFC3/IR, with coordinates scaled by λ/D. Once the NIRCam solution collapses to a delta in λ/D units, the same function rescaled by 6.5/2.4 is still narrow for HST. This is the over-rigid physics.

**Why this explains the gate:**
- With one instrument, the degeneracy is harmless for prediction. That is why the JWST-only scores look good.
- With two instruments, one sky cannot carry two different sets of PSF wings, so both instruments degrade.
- The bright stars dominate the S/N > 5 metric, and they show it most.

Q5 and the other wins are prediction wins (a better coadd). They are not evidence that the sky is a deconvolved, instrument-free sky.

## Other shortcuts found in the code

| # | Shortcut | Where | Expected effect |
|---|---|---|---|
| 1 | Sky and PSF degenerate; sky starts from drizzle; PSF starts as a narrow Gaussian | above | **main cause** of the gate failure |
| 2 | Kernel window 1.24″; wings truncated and renormalised to unit sum | `psf_n=31` | about 10–15% of HST IR flux and the LW wings land in the sky |
| 3 | Rendering is resampling: sky ⊗ kernel on the 0.04″ grid, then bilinear interpolation at pixel centres. A detector-frame kernel is itself resampled onto the grid (bilinear rotation for the explicit PSF) | `convolve_basis`, `sample`, `resample_rot` | the 0.04″ grid undersamples NIRCam SW (0.031″ pixels, F090W FWHM ≈ 0.03″) and is about 1× for ACS/UVIS; the literature needs ≥ 4× supersampling of the native pixel |
| 4 | Data pixels are kept in a 24-pixel ring outside the 512² sky grid, and the model predicts background only there (bilinear sampling into zeros) | `load_exposures`, `load_hst_exposures` keep `-pad < tp < size+pad` | about 16% of a fully covering exposure's pixels can never be fitted. The four bright stars sit 0.9–1.9″ from the edges. The bright-pixel score excludes the ring, but training does not (Huber-bounded). |
| 5 | Field of 20.5″ | `size 512, scale 0.04` | the PSF spatial variation and the distortion are learned from a few stars covering 15–30% of each detector; the field cannot be identified |
| 6 | No time or focus term: one PSF per band for 2017 and 2022 | ChromaticPSF has no exposure code | HST breathing ±3% core and > 20% off focus (Bellini+ 2018); JWST epoch-to-epoch 3–4% (Nardiello+ 2022) |
| 7 | One shared PSF network for all instruments, with only a one-hot | ChromaticPSF | no per-instrument capacity or adaptation |
| 8 | Spatial variation of rank 4, residual kernels scaled by 0.01 | ChromaticPSF | the literature needs about 20 components for ACS (Jee+ 2007), and grids of 3×3 (WFC3/IR) to 9×10 (ACS) |
| 9 | JWST geometry starts from the FITS SIP approximation in the `_cal` header, not the gwcs; no filter-dependent distortion term | `WCS(header=hd)` | small (tolerance of the SIP fit; NIRCam filter-dependent scale is detectable at 14σ, Griggio+ 2023); the per-detector cubic absorbs most of it |
| 10 | HST a-priori WCS, with registration only against NIRCam | v31 | visits are offset by up to 0.6″, plausible for the guide-star WCS (0.1″ typical with GSC 2.4, 0.3″ with GSC 2.3, 0.75″ worst; DrizzlePac 4.5); no HST-only anchor |
| 11 | Optimiser: Adam, 300 min budget, validation still improving at the end (8.58 → 5.38, last half hour −1%) | `fit` | not converged |
| 12 | Detector effects absent: IPC (~5% WFC3/IR), brighter-fatter, charge transfer, IRAC intrapixel gain, cosmic rays in single CCD exposures | N_i not built | inflates bright-pixel χ² on its own |

Correct in the current code:
- The HST WCS is read with `fobj`, so the D2IM and NPOL lookup tables are applied.
- Data units are surface brightness, consistent with pixel-centre sampling, so the pixel-area convention is consistent.

## What the literature says a strong operator needs

Sources and numbers: `claude/psf-literature-review.md`.

1. **Detector-frame effective PSF (ePSF), supersampled ≥ 4× the native pixel.** Interpolated continuously. The standard practice since Anderson & King 2000. Oversampling of 1 gives up to about 20% flux errors; 3–4× gives about 0.1% (Godden & Blundell 2025, arXiv:2512.16764).
2. **Spatial variation over the whole detector.** NIRCam and NIRISS use 5×5 grids per detector (Nardiello+ 2022; Libralato+ 2023), WFC3/IR 3×3, ACS 9×10, IRAC 5×5.
3. **A time latent per exposure.** HST focus lies on a one-dimensional manifold, and about 5 stars per exposure fix it (Bellini+ 2018). JWST needs a term per wavefront-sensing epoch (about every 2 days).
4. **Something must break the PSF–scene degeneracy.**
   - STARRED (Millon+ 2024) and ALPACA (arXiv:2609.04312) need a prior or anchor on the PSF.
   - On HST, STARRED found the best scene grid is 3× finer than the data.
   - A differentiable optical model plus a data-driven residual (WaveDiff, Liaudat+ 2023; dLux, Desdoigts+ 2024) beats pure pixel models by large factors and handles chromatic variation.
5. **The full distortion chain as a start, then small learned residuals.**
   - ACS linear terms drift about 25 mas over 20 years.
   - WFC3/UVIS filter-dependent terms are 0.3 px.
   - IRAC needs 7th/8th order and a time-dependent scale (Esplin & Luhman 2016).
6. **A robust likelihood and a model-error floor.** The best ePSFs still leave 3–4% residuals.
7. **No published neural field learns PSF and distortion jointly from dithered exposures without a catalogue.** The closest are "neural aberration fields" (JOSA A 2026, vision) and WaveDiff (catalogue of stars). This is an open niche.

## Proposed stronger operator (to discuss)

**Gauge-fixing and degeneracy.**
- The sky is the only place an unresolved source can be a point. It must be represented at a resolution finer than every PSF, or as continuous splats.
- The PSF must carry all the blur.
- Options to make the fit go there, from weakest to strongest prior:
  - **(a)** Physics warm start only: STPSF with dated OPDs for NIRCam; the STScI WFC3/IR, ACS and UVIS ePSF libraries (4× supersampled, spatial grid, focus-diverse); the IRAC PRF. Plus a sky warm start deconvolved with those PSFs, not drizzled.
  - **(b)** (a) plus a soft prior toward the warm start, with its weight chosen on held-out data and annealed.
  - **(c)** Cross-instrument consistency alone: a shared sky at the same λ for NIRCam, WFC3/IR and NIRISS. It breaks the degeneracy only if the sky has enough resolution and the PSFs start near the truth.
- (a) is within the "physics only as warm start" rule. (b) bends it slightly.

**Representation.** One PSF field per instrument (per detector, through an embedding):
- P(offset in native pixels; detector x, y; λ; epoch/focus code z_i), evaluated continuously. No kernel on the sky grid.
- Core: a supersampled grid (≥ 4×) or Gaussian mixture plus a neural residual. Wings: a coarse radial or Fourier component out to ≥ 3″ (5″ for WFC3/IR and LW).
- λ/D enters only as an input feature, never as shared structure across telescopes.

**Rendering.** For each exposure and detector pixel:
- integrate sky × PSF × pixel box (plus an intrapixel map) in the detector frame;
- splats in closed form (Gaussian mixture), grids by quadrature on a supersampled local grid;
- no bilinear resampling of a convolved image.

**Geometry.**
- Start from the full archive chain: `fobj` for HST; the gwcs (asdf) for JWST, not the SIP approximation.
- Learn per-detector, per-filter residual fields, per-epoch linear terms, and a per-exposure affine with a coarse-to-fine scan against an internal reference (no Gaia in the model).

**Domain.** The sky grid or tile extends one PSF wing radius beyond the data, with no ring of unpredictable pixels. The PSF field is learned on full detectors from the union pixel store (tiles), not on 20″.

**Likelihood.** A cosmic-ray and outlier mixture plus a fractional model-error floor, learned.

## Experiment ladder (cheap first, each answers one question)

| Exp | Change | Question | Cost |
|---|---|---|---|
| E1 | Fix the data ring (sky grid with a halo of 1.5″, score the interior only); rerun A and B | how much of the gate is the edge bug? | v34 patch, 2 × 5 h GPU |
| E2 | Physics warm start (STPSF NIRCam with dated OPD, STScI WFC3/IR ePSF library), per-instrument PSF networks, 3″ window, sky warm start by deconvolution, sky grid 0.02″; rerun A and B | does breaking the degeneracy pass the gate? Do the learned EE curves stay near the handbook? | new PSF module, 2 × 5–8 h GPU |
| E3 | E2 plus an epoch/focus code per exposure, B with 2017 + 2022 HST | does time variation matter? | small |
| E4 | E2 on 4–9 union tiles (full detectors) | can the spatial variation be learned? | U1 tile loop |
| E5 | A2 continuous operator (closed-form splat rendering) | does the continuous operator beat E2 on held-out data and injection–recovery? | larger build |

Gate for E2:
- the learned PSF EE matches the handbook and STPSF to a few percent;
- NIRCam held-out with HST added is no worse than JWST-only;
- WFC3/IR held-out is well below the coadd baseline (14.8 now, against the learned model's 14.2).

## Addendum (6 Nov, later): reformulation after Seb's steer, and E1/E2 status

**Steer from Seb.** Do not expect physics to break degeneracies, since sim-to-real has not worked in observational astrophysics. Break them with multi-instrument diversity, data splitting, good intuitions and a strong set of empirical tests. The library warm start is therefore only a control arm.

**The common-mode gauge cannot be broken by data diversity.**
- For any common kernel g, (s ⊗ g, K_i ⊗ g⁻¹ for every i) predicts every exposure equally well.
- Diversity can only limit g:
  - more instruments, fields, dithers or epochs constrain the RELATIVE kernels, not g;
  - rolls force g to be isotropic;
  - positivity of the sky and kernels bounds how sharp the sky can be made.
- With positivity the natural data-driven gauge is "the sky at the resolution of the sharpest kernel". The real fits went exactly there: the NIRCam SW kernels became delta functions.
- "Deconvolved sky" statements therefore need a declared reference resolution.
- The JWST+HST gate failure is not this degeneracy, which predicts equally well for any g. It is the inability to represent the relative kernels: a λ/D-shared network, a 1.24″ window, rank-4 spatial variation, no time term.

**Synthetic check (v36, Sonnet, same renderer for data and model, so not independent of the renderer).**
- Setup: generic Gaussian start, learnable positive tables, shared sky.
- Absolute kernels are wrong, but a single common g explains almost all of the error. The residual against the truth after removing one free g(f) is 0.009–0.010; it is 0.31–0.52 before.
- Relative kernels are right to about 0.03 (0.55 at the start). Held-out prediction errors fall from about 0.2 to about 0.01.
- This holds with two rolls and with three instruments.
- Physics warm start (v35, library 15% too broad): it overshoots and does not hold the PSF.

**Runs (6 Nov):**
- E1, v34 halo: `v34_E1_jwst_cal_halo`, `v34_E1_jwst_wfc3ir_halo`.
- E2-data, v36: generic Gaussian tables, learned per (detector, band), positive, achromatic within a band; 3″ window; bicubic sampling; drizzle start.
  - `v36_E2data_jwst_s04` (0.04″)
  - `v36_E2data_jwst_wfc3ir_s04` (0.04″)
  - `v36_E2data_jwst_wfc3ir_s02` (0.02″)
- E2-lib control: `v36_E2lib_jwst_wfc3ir_s04`. Same model, tables started from the STPSF and STDPSF libraries.
- The physics libraries are in `/arc/projects/ots/adass2026/psflib/v1`:
  - STPSF with the OPD R2022060802, 1 day from the observation;
  - the Anderson WFC3/IR STDPSF;
  - the orientation check is inconclusive.
- Caveat: at 0.02″ the bright-pixel masks come from a different coadd. Compare s02 with s04 on the per-exposure all-pixel χ² and robust scores, not on the bright-pixel score alone.

**Empirical test suite (proposed gate for every operator change):**

| Test | Split | What passing means |
|---|---|---|
| T1 | held-out dithers (now) | prediction within a visit |
| T2 | held-out visits and rolls | the operator follows the roll in the detector frame |
| T3 | held-out epochs | time terms and a static sky separated |
| T4 | new-instrument transfer: sky frozen from a fit that never saw instrument X; fit only X's operator on half its exposures, predict the other half | the sky is instrument-agnostic up to the declared gauge |
| T5 | held-out tiles/fields | the operator transfers across fields |
| T6 | injection–recovery by transplanting real data patches (no simulated PSF) | flux and shape recovered, including bright and saturated sources |
| T7 | residual null tests stacked by detector position, pixel phase, roll, time, readout direction | no structure left in the operator |
