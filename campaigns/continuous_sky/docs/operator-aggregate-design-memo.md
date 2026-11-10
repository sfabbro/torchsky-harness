# Design memo: learning the operator in aggregate (7 Nov 2026)

## Summary
My hypothesis (to be tested, not established) is that the shoulder and wing deficit is mainly **data starvation**. The 20″ SMACS box has about four bright stars, covers one corner of each SW detector, and has one epoch. That is too little to separate wings, focal-plane variation and flux-dependent broadening. The elegant fix is the ePSF route, made catalogue-free: learn one operator per instrument from many dithered pointings, give each field its own continuous sky, and profile the sky out.

## Options

1. **Aggregate operator ("catalogue-free ePSF").**
   - One operator per (instrument, detector, band), shared across many fields and thousands of exposures, with a small code z_i per exposure.
   - Each tile has its own sky, treated as a nuisance.
2. **Cross-fitted variable projection (VarPro).**
   - Solve the sky s_A* by CG on split A of the exposures.
   - Train θ only on Σ_B ‖d_B − A_B,θ s_A*(θ)‖²_w over the other split B, with one adjoint solve per step.
   - The sky cannot absorb view-specific kernel errors, so diversity (detector position, roll, instrument) becomes the training signal: Noise2Noise logic and self-calibration (Holmes, Hogg & Rix 2012) applied to operators.
3. **Equivariant detector-frame operator with a learned low-rank manifold.**
   - μ = N(Σ_k w_k(p, z_i) · [P_k ⊗ (sky mapped to the detector)]).
   - Rendered by FFT on whole exposures, with wings to 5″.
   - The modes P_k and the weights w_k span focal-plane position and time: a data-driven focus manifold (Bellini+ 2018).
   - There is no Λ gauge, and no wing pixels are lost to minibatches.
4. **Sky-free cross-relation.**
   - For two views of one sky, K_j⊗d_i − K_i⊗d_j ≈ 0, so the sky never appears.
   - Multichannel blind deconvolution recovers kernels up to a common factor, which is exactly our g gauge.

Options 2 and 3 are the engine and the representation for option 1, and they compose with it.

## Option 1 in detail

**Fit.** Alternate two steps.
- **Sky.** Per tile, s_f* = argmin −log L + ρ‖s‖², with s ≥ 0.
- **Operator.** A θ step with gradients summed over tiles, then update z_i.
- At a converged s_f*, the envelope theorem gives ∂L*/∂θ = ∂L/∂θ|_{s*}. The alternation is then exact VarPro (Golub & Pereyra 2003), with no backpropagation through CG.
- This ends the race in which the sky "absorbs everything faster" (audit, item 4). It is the global/local split of `union-fit-design.md` §3 with the U2 solver.
- Option 2 adds cross-fitting on top.

**What aggregation breaks.**
- **Focal-plane variation.** Large dithers move the same stars across detectors (PID 1476: 24″ steps, 9 positions).
- **Wings and shoulders.**
  - The constraint grows with Σ flux² over bright stars: likely of order 10³ per detector against about 4 (not verified).
  - Saturated stars contribute too: their cores are DQ-masked, their wings stay, and the sky takes their flux freely.
- **Time variation.** Repeated monitors make z_i identifiable as a manifold rather than a per-exposure fudge.
- **Brighter-fatter versus PSF.** The SMACS shoulder deficit (+7–9% at 0.5–1.5 px around S/N>50 peaks) may be flux-dependent broadening (5–6% for H2RG; Plazas+ 2018). Only many stars over a wide flux range separate N from K.

**What it does not break.**
- **The common blur g remains a gauge.** In stellar fields the sky is mostly unresolved, so positivity drives the declared "sharpest kernel" gauge towards the physical PSF, up to the sky's resolution. This is a property of the data, not a theorem, and it needs a sky at 0.015–0.02″ or splats. At 0.04″ the sky undersamples the SW core (audit, item 3).
- **Far wings beyond the window** are degenerate with the low-order background B. Vary the window (3″, 5″, 8″).
- **Flux scale.** A declared gauge.
- **Within-band chromaticity.** Stellar SEDs lack galaxy emission lines.
- **SMACS's own epoch.** z_i must be re-fitted.

**Data.**

| Field | Programmes (verified) | Bands |
|---|---|---|
| LMC calibration field, NIRCam | 1476 (C1 L-flat, photometric and distortion monitor; 16 filter pairs, 9-point dithers); 4447 (C2; F150W/F277W, F200W/F356W, F070W/F444W, F210M/F335M) | SMACS bands |
| LMC astrometric field, NIRISS | 1501 and 4471 (all filters, 4 dithers, two sets about 150 d apart); 1515 and 4475 (bimonthly F200W) | NIRISS; time series for z |
| M92, NIRCam | ERS 1334 (source of the Nardiello+ 2022 PSFs) | F090W, F150W, F277W, F444W |
| 47 Tuc, NGC 6397, M4 | 2559; 1979 (with NIRISS F150W in parallel) | F150W2/F322W2 only; λ-generalisation test |
| ω Cen, WFC3/IR | 11928, 12340, 12353, 12694, 12714 (270 IR exposures: F160W 59, F125W 71, F105W 4, F140W 4); monitor 17974 | F105W and F140W thin |
| 47 Tuc and LMC, WFC3/IR | SMOV 11445 | F160W |
| Extragalactic | PRIMER 1837 (F090W–F444W, two epochs at 180°); COSMOS-Web 1727 (152 visits); CANUCS 1208 and 3362 | SMACS-like control |

Not verified: the exact filter pairs and observation dates of 1476. Take them from MAST and prefer the visits nearest June 2022.

**Scale.**
- A NIRCam detector image is 4.2 Mpx, about 130 MB in the store.
- The C1+C2 LMC NIRCam data are of order 10³ images, 4–8 × 10⁹ px: SMACS-union scale.
- Stratified subsampling should do. Target a few hundred bright-star wings per (detector, 5×5 cell, band).
- **Compute (assumed).** About 0.5 s per detector image for forward plus backward (FFT, 9 modes × 4 λ nodes) and about 30 CG matvecs per outer iteration. That is 2–3 h per outer iteration over 2000 images on 4 slices, so about a day.
- **Memory per slice.** One 30″ tile sky (0.02″ × 12 spectral terms ≈ 3 × 10⁷) plus the operator (about 7 × 10⁷ parameters) fits in 10.75 GiB.

**Catalogue trap.**
- Fit the dense fields exactly like SMACS: a continuous positive sky (grid or residual-densified splats).
- No detection, no star selection, no external positions, and every pixel used, including galaxies and diffuse light.
- The residual risk is curation: an operator trained only on point sources may favour point-like skies. Mix in the extragalactic programmes. The SMACS transfer test is the judge.

**Tests.**
- T5: held-out LMC tiles.
- T3: held-out epochs (C1 against C2).
- T2: held-out rolls (PRIMER).
- **The decisive test: frozen transfer to SMACS.**
- T7: null stacks binned by peak flux, to separate N from K.

## Ranking

| Option | Gain on shoulders/wings | Elegance and generality | Cost | Risk |
|---|---|---|---|---|
| 1 Aggregate | high | high | medium | epoch and SED mismatch at transfer |
| 3 Equivariant FFT | medium-high | high | medium | the equivariance assumption |
| 2 Cross-fitted VarPro | medium (conditioning; adds no information) | very high | low; needs U2 CG | small alone |
| 4 Cross-relation | low | conceptually the cleanest | low | aliasing; needs resampling |

## First experiment (1–2 days): frozen transfer from the LMC field

1. **Data.** Download the LMC-CAL-FIELD NIRCam exposures in the six SMACS bands.
2. **Tiles.** Extract 8 tiles of 20″, each with a 3″ halo, chosen so that the dithers spread each tile across detector cells.
3. **Model.** E2-data (the current best), extended with 3×3 spatial modes, a 5″ window and a 0.02″ sky. Tables are shared; each tile has its own sky, with k inner sky steps per table step (VarPro-lite until CG lands). Use 4 slices with about 2 tiles each.
4. **Optional second arm.** The field operator (Λ=J, log shape, importance sampling) on the same tiles, giving a 2×2 of data starvation against representation.
5. **Transfer.** Freeze the tables. Fit the SMACS sky, geometry, background, flux and a 2-d z on the training exposures. Score on the split and masks of `v38_E2data_jwst_dump`.

**Success.**
- NIRCam held-out S/N>5 ≤ 2.26 (the top of E2's [2.14, 2.26]).
- Shoulder plus wing χ² share ≤ 11% (E2: 13.5%).
- Shoulder deficit at 1 px below 3% (E2: about 7%).

**Falsified if:**
- frozen transfer scores above 2.40 while fine-tuning on SMACS recovers 2.20 (the operator is field- or epoch-specific); or
- the shoulder deficit persists and grows with peak flux (the cause is N, brighter-fatter or IPC, not K).

## Principle flags

- **Option 1.** Field choice is curation, and unresolved stars effectively pin g. Declare this; do not impose it.
- **Option 2.** The ridge ρ is a sky prior; choose it on held-out data.
- **Option 3.** Local shift-equivariance (the isoplanatic patch) is a physics-shaped constraint. Choose capacity on held-out data and run pixel-phase null tests.
- **Option 4.** It needs a common grid, which violates no-resampling, and it fails for undersampled WFC3/IR and NIRCam SW. Use it as an LW diagnostic only.
- **None of the options breaks g, the flux scale, or far wings beyond the window.**

## Sources

- JWST programmes:
  - 1476: https://www.stsci.edu/jwst-program-info/download/jwst/pdf/1476/
  - 4447: https://www.stsci.edu/jwst-program-info/download/jwst/pdf/4447/
  - 4471: https://www.stsci.edu/jwst-program-info/download/jwst/pdf/4471/
  - 4475: https://www.stsci.edu/jwst-program-info/download/jwst/pdf/4475/
  - 2559: https://www.stsci.edu/jwst-program-info/download/jwst/pdf/2559/
  - 1979: https://www.stsci.edu/jwst/phase2-public/1979.pdf
- ERS 1334: Nardiello+ 2022, https://arxiv.org/abs/2209.06547
- NIRISS LMC: Libralato+ 2023, https://arxiv.org/abs/2303.00009
- WFC3/IR:
  - ω Cen exposures: Bellini+ 2017, https://arxiv.org/abs/1704.07425
  - L-flats: https://hst-docs.stsci.edu/wfc3dhb/chapter-7-wfc3-ir-sources-of-error/7-8-ir-flat-fields
  - SMOV 11445: https://hst-docs.stsci.edu/wfc3dhb/chapter-4-wfc3-images-distortion-correction-and-astrodrizzle/4-1-wfc3-geometric-distortion
  - Programme 17974: https://www.stsci.edu/hst-program-info/download/hst/pdf/17974/
- Extragalactic programmes:
  - PRIMER: https://primer-jwst.github.io/observations.html
  - COSMOS-Web: https://arxiv.org/abs/2506.03256
  - CANUCS: https://archive.stsci.edu/hlsp/canucs
- Methods:
  - Focus manifold: Bellini+ 2018, https://arxiv.org/abs/1706.00386
  - Self-calibration: Holmes, Hogg & Rix 2012, https://arxiv.org/abs/1203.6255
  - Noise2Noise: https://arxiv.org/abs/1803.04189
  - Plazas+ 2018: https://arxiv.org/abs/1712.06642
  - Golub & Pereyra 2003, Inverse Problems 19, R1 (from memory)
- Programme 1501 is cited via 4471's own text; 1515 via 4475's.
