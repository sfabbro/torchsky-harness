# Literature review: PSF and distortion models for HST, JWST and Spitzer (6 Nov 2026)

Compiled by a subagent; two arXiv IDs (2512.16764, 2609.04312) were spot-checked. Items marked "derived" are arithmetic, not quotes. Not retrieved in full: ACS ISR 2006-01, WFC3 ISR 2016-12, the hst1pass ISR, a NIRCam IPC value.

## 1. Effective PSF practice

- **ePSF.** Optical PSF ⊗ pixel response, tabulated 4× supersampled, iterated with star positions using dithers (Anderson & King 2000, astro-ph/0006325).
- **ACS/WFC.** 9×10 grid of 4× ePSFs. Focus-diverse versions exist for 12 filters (STScI ACS PSF pages; Bellini+ ACS ISR 2018-08; Anand+ ACS ISR 2023-06).
- **WFC3/UVIS.** 7×8 grid in total, 4×, focus-diverse for 7 filters.
- **WFC3/IR.**
  - A 3×3 grid is adequate; the largest deviation from the mean PSF is about 10% of the peak.
  - A centred star puts 43% of its light in the central pixel (F105W, WFC3 IHB 7.6).
  - Handbook FWHM: 0.128″ at 1.0 µm to 0.151″ at 1.6 µm.
  - Handbook EE (IHB Table 7.6), F160W: 0.494 at 0.13″, 0.736 at 0.26″, 0.833 at 0.39″, 0.855 at 0.52″, 0.918 at 1.0″, 0.966 at 2.0″.
- **NIRCam.**
  - 5×5 ePSF grid per detector at 4×.
  - Spatial variation about 9% peak-to-peak (F150W) and 15–20% (F090W); about 3–4% between epochs.
  - Perturbing the PSF per image halved the fit-quality metric of bright stars (Nardiello+ 2022, arXiv:2209.06547).
- **NIRISS.** 5×5 grid at 4×; central-pixel flux varies by less than about 2.5–5% over 51 days (Libralato+ 2023, arXiv:2303.00009).
- **IRAC.** Core PRFs on a 5×5 grid, 5× and 100× supersampled (IRSA PRF pages).
- **Oversampling.** For FWHM ≈ 1.35 px, oversampling of 1 gives up to about 20% flux error and 0.3 px centroid error; oversampling of 3 gives about 0.1%. At least 4 grid points per FWHM (Godden & Blundell 2025, arXiv:2512.16764).

## 2. Geometric distortion and the a-priori WCS

- **ACS/WFC.**
  - The axes meet at 84.9° and 86.1°; pixel area ranges 0.88–1.06.
  - Linear terms drift about 25 mas over about 20 years (ACS IHB 10.4; Anderson 2007).
  - The current solution (IDCTAB + D2IM + NPOL) corrects to about 0.02 px (Kozhurina-Platais 2015).
- **WFC3.**
  - UVIS distortion is 7–11% across the field, with lithographic patterns.
  - Filter-dependent tables bring residuals below 0.008 px (Bellini, Anderson & Bedin 2011, arXiv:1102.5218).
  - UVIS scale and skew drift less than 0.2 px over 13 years, with ±0.3 px scatter per image (O'Connor+ 2024, arXiv:2602.04041).
- **NIRCam.**
  - A 3rd-order polynomial plus a 16×16 residual table reaches better than about 0.2 mas.
  - The pixel scale is filter-dependent (F090W vs F150W at 14σ); velocity aberration changes the scale by about 1e-4 (Griggio+ 2023, arXiv:2212.03256).
- **IRAC.** The pipeline's 3rd-order solution leaves about 100 mas; 7th/8th order with a time-dependent scale reaches about ±10 mas. Pixel-phase centroid bias exceeds 100 mas (Esplin & Luhman 2016, arXiv:1511.06403).
- **A-priori HST pointing.** About 0.3″ with GSC 2.3 (0.75″ worst case) and about 0.1″ with GSC 2.4 (adopted October 2017); Gaia alignment gives about 10 mas (DrizzlePac Handbook 4.5). Offsets of 0.6″ between visits are plausible.
- **astropy WCS** without `fobj` drops the D2IM and NPOL tables (our hst_io passes `fobj`).

## 3. Time variability of the PSF

- **HST breathing.** Focus moves 3–5 µm per orbit (Jee+ 2007, arXiv:0710.5560), over −10 to +5 µm across visits (Rhodes+ 2006).
- **WFC3/UVIS focus manifold** (Bellini+ 2018, arXiv:1706.00386):
  - PSFs lie on a one-dimensional focus manifold.
  - Library-PSF residuals are about 4% near good focus and exceed 20% off focus; a focus-fitted PSF stays near 3%.
  - About 5 stars per exposure set the focus.
- **ACS.** About 20 principal components with a 5th-order spatial polynomial (Jee+ 2007).
- **WFC3/IR.** Breathing is small beyond 1.1 µm.
- **JWST.** The wavefront is sensed about every 2 days and corrected above about 70 nm rms. F200W EE within 2.5 px varied between 0.661 and 0.673 (STPSF documentation on measured OPDs).

## 4. Detector effects that mimic PSF errors

- **Charge transfer (ACS).** Trails out to about 70 px; flux losses of 20–50% in short exposures (Anderson & Bedin 2010, arXiv:1007.3987).
- **Cosmic rays.** About 1.2 cm⁻² s⁻¹ on ACS and UVIS (Miles+ 2021, arXiv:2006.00909). That is roughly 2–3% of pixels per 1000 s (derived). Single `_flc` files carry no cosmic-ray flags unless AstroDrizzle updated their DQ.
- **IPC.** About 5% on WFC3/IR (WFC3 ISR 2011-10).
- **Brighter-fatter and charge migration.**
  - H2RG lab spots grow 5–6% (Plazas+ 2018, arXiv:1712.06642).
  - WFC3/IR stars grow about 2% (Finner+ 2023, arXiv:2301.07725).
  - NIRISS charge-migration correction raised the peak pixel by 13% (Goudfrooij+ 2023, arXiv:2311.16301).
- **IRAC intrapixel gain.** Up to about 8% in channel 1 and 2–3% in channel 2 (Ingalls+ 2012; IRSA pixel-phase pages).

## 5. Data-driven and differentiable PSF models

- **PSFEx, PIFF** (Jarvis+ 2021) and **MCCD** (Liaudat+ 2021, arXiv:2011.09835).
- **WaveDiff** (Liaudat+ 2023, arXiv:2203.04908): a differentiable optical model in wavefront space plus a data-driven part. Pixel error is 6× lower at native resolution and 44× lower at 3× super-resolution; it handles chromatic variation.
- **dLux** (Desdoigts+ 2024, arXiv:2406.08703): differentiable optics; recovers phase and flat field jointly from science data.
- **STARRED** (Millon+ 2024, arXiv:2402.08725): joint PSF and scene fit over many epochs. On HST the best grid was 3× finer than the data; the native grid biased photometry by 3–11%.
- **ALPACA** (arXiv:2609.04312): joint PSF and lens fit on NIRCam. A fixed PSF with 12% RMSE biased results by 14%. Joint fitting cut that to about 2%, but only with a Gaussian prior anchoring the PSF.
- **Others:** HybPSF (arXiv:2308.14065), a hybrid JWST NIRCam PSF; pixellated posterior sampling of PSFs (arXiv:2511.19594); "Neural aberration fields" (JOSA A 2026), an implicit field for spatially varying PSFs (vision, not astronomy).
- **The niche is open.** No published catalogue-free neural field learns PSF and distortion jointly from dithered exposures.

## 6. Undersampled reconstruction

- **Lauer 1999** (astro-ph/9810394): K× undersampled data need at least K² dithers, acting as translations constant to a few percent of a pixel.
- **Drizzle:** Fruchter & Hook 2002.
- **IMCOM** (Rowe, Hirata & Rhodes 2011, arXiv:1105.2852) needs a known PSF for every exposure and pixel.
- **Dither patterns:** uniform sub-pixel grids beat random dithers (Godden & Blundell 2025).
- **Self-calibration** needs good dither coverage (Arendt+ 2000).

## Ranked ingredients most likely missing (subagent's ranking)

1. Supersampled detector-frame ePSF (≥ 4× native).
2. Spatial variation over each detector.
3. A focus or epoch latent per exposure.
4. A robust likelihood (cosmic rays and outliers) plus a model-error floor.
5. The full distortion chain plus small learned residuals.
6. Coarse-to-fine registration within the basin of convergence.
7. Detector terms: brighter-fatter, IPC, charge transfer, IRAC intrapixel gain.
8. A chromatic PSF and a consistent pixel-area convention.
