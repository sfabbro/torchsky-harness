# One sky at every scale, version 2: raw inputs, any scale, a scored ensemble

Nov 13, 2026 · Sébastien Fabbro

Living doc (the editable copy, tab "Version 2"): https://claude.ai/artifact/DYP9UJuvosgpaz7BGG4yr5#bcefa2e8-f671

Status: proposal. Seb asked for this rewrite on 13 Nov and wants the small experiments run before any large build. Nothing has been run. It supersedes `claude/one-sky-at-every-scale-plan.md` (version 1), whose review, gauge table, likelihood and kernel pyramid still stand.

## Summary

Version 2 changes four things: inputs become raw, scale becomes continuous with learned profiles, the sky gains a shared predictor that is scored but never imposed, and the method is trained on a sample and then applied. Nothing has been run. Each new element gets one small test, with a rule for dropping it, before any large build.

|  | Version 1 | Version 2 | Why |
| --- | --- | --- | --- |
| Inputs | Calibrated exposures | Raw ramps and frames, plus calibration frames as exposures of trivial scenes | Raw data add the diversity that identifies the detector, and nothing is baked in |
| Sky | Gaussian blobs on discrete levels | Items with continuous position, size and orientation, and a learned profile | Any scale; the same weights serve billions of similar objects |
| Across the sky | Every place fitted alone | A shared ensemble predictor of finer from coarser scales, scored on raw held-out pixels | Uses the redundancy of the sky; gives a null model for unknown unknowns |
| Scale-up | Optimisation per field | Train on a sample, apply in a few passes | At 3.3 GPU-hours per 20 arcsec box, per-field fitting cannot reach the sky |

**On calibration data.** Include calibration frames; do not bet on self-calibration alone. They are cheaper than science frames because they need no sky, and three quantities cannot be self-calibrated at any data volume. The bet itself is tested as experiment E3.

**Your steer, 13 Nov.**

- As raw as possible, with approximations and lossy steps removed. Otherwise the best outcome is parity.
- Any scale, not a fixed set of levels.
- As free of assumptions and priors as feasible; fit for unknown unknowns.
- So different from the present code that it needs small experiments first.

## Formulation

Three learned parts: a sky made of items with learned profiles, an instrument operator that predicts raw data, and an ensemble predictor across scales. Only the first two enter the measured sky.

**1. Sky: items with a learned profile, at any scale.**

```latex
I(\boldsymbol\theta,\lambda)=\sum_n a_n\;\psi\!\Big(\frac{\mathsf R_n^{\top}(\boldsymbol\theta-\boldsymbol\theta_n)}{s_n},\ \frac{\lambda}{\kappa_n};\ \mathbf z_n\Big),\qquad a_n\ge0,\quad \psi\ge0 .
```

| Symbol | Meaning |
| --- | --- |
| a | Flux of item n |
| θ, s, R | Its position, size and orientation, all continuous and fitted |
| κ | Its stretch in wavelength; for a redshifted source this is 1 + redshift |
| z | A short latent vector that selects its shape and spectrum |
| ψ | One shared network: the learned family of profiles, normalised to unit integral |

- **Scale is factored out of the network.** The network sees coordinates in units of the item's own size. It needs about one decade of bandwidth, for a 10 mas core and for a degree of cirrus alike.
- **All of the range in scale sits in one number per item,** the logarithm of its size, which has no spectral bias.
- **Items are seeded by coverage,** on the footprint of the data, with no detection step. An item is removed when its flux goes to zero.
- **An item is not a source.** A galaxy may be one item or several; the model has no notion of an object.
- **Version 1 is the special case** where the profile is a fixed Gaussian.

**2. Instrument: one operator from scene to raw data.** The charge rate in a pixel, then the raw value at read r:

```latex
q_e(\mathbf x)=A_e(\mathbf x)\,\big[k_e\star S_e\big](\mathbf x)+\beta_e(\mathbf x),
\qquad
\mu_e(\mathbf x,r)=\mathcal D_e\big[\,q_e\,t_r\,\big](\mathbf x)+o_e(\mathbf x,r).
```

```latex
S_e=\begin{cases} I+I^{\mathrm{fg}}_e & \text{science frame}\\ L_e & \text{lamp or twilight flat: a free, smooth illumination field}\\ 0 & \text{dark or bias}\end{cases}
```

- A, k and β are the amplitude, kernel and additive detector terms of version 1, as learned fields.
- D is the learned detector map: non-linearity, brighter-fatter, saturation. It acts on accumulated charge, at read time t.
- o is the offset per read: bias, reset level and stripes.
- The channel is now a pixel and a read. For a CCD there is one read.

**3. Predictor: an ensemble of finer scales given coarser ones.**

```latex
\big\{\,n:\ s_n<s\,\big\}^{(k)}=G_\phi\Big(\big\{\,n:\ s_n\ge s\,\big\},\ s,\ \boldsymbol\xi^{(k)}\Big),\qquad \boldsymbol\xi^{(k)}\sim\mathcal N(0,\mathbf I_m),\quad m\ \text{small}.
```

- G is one network shared over the whole sky. The size s is a continuous input, so there are no levels.
- The noise vector enters the normalisation layers of the network, as in WeatherNext 2 and 3. Each draw is one coherent sky.

**4. Score: a proper scoring rule on raw held-out pixels.** No density is normalised and no prior is written down.

```latex
\mathcal L(\phi)=\sum_{(e,\mathbf x,r)\ \text{held out}}\mathrm{CRPS}\Big(\big\{\mu^{(k)}_e(\mathbf x,r)+n^{(k)}\big\}_k,\ d_e(\mathbf x,r)\Big),
```

where each ensemble member is rendered through the instrument and given one draw n of the noise model.

**5. Three products, kept apart.**

| Product | How it is made | Role of the predictor |
| --- | --- | --- |
| Measured sky | Items fitted to the raw data alone, down to the size the data constrain | None |
| Predicted continuation | Ensemble below that size, or where data are shallow | All of it; labelled as prediction |
| Surprise map | Per-pixel score of the raw data against the ensemble | Null model: what it fails to predict is flagged |

Going deeper than the data allow at one place means borrowing from other places. That is a prior by any name, so it is confined to the second product and never enters the first.

## Raw inputs and calibration data

Include the calibration frames. They cost less than science frames, they constrain exactly the terms that science frames constrain slowly, and no amount of sky diversity fixes the three gauges below.

**What can calibrate itself, and what cannot.**

| Class | Quantities | What fixes them | Cost |
| --- | --- | --- | --- |
| Cannot calibrate itself | Absolute flux and bandpass shape; absolute wavelength; absolute position, rotation and plate scale | One external reference each: a standard star or reference throughput curve, an arc or known lines, one astrometric reference | Negligible. Declared, not fitted |
| Can calibrate itself, slowly | Dark current, bias and hot pixels; pixel-to-pixel response; non-linearity and gain | Science frames with enough dithers and a range of background levels. Calibration frames give the same far faster | Low: no sky and no kernel, only per-pixel terms |
| Only science frames can give | Kernel; distortion residuals; large-scale illumination; fringing on the sky; foreground; drifts between calibration epochs | Dithers, rolls, epochs and instruments | The main fit |

**How calibration frames enter.** They are exposures of trivial scenes, through the same operator (the scene S in the formulation).

- **Dark and bias:** the scene is zero. They constrain the additive detector terms and the offsets.
- **Lamp or twilight flat:** the scene is a free smooth field per lamp and epoch. Only what is common with the science frames is constrained: the pixel-scale response and the detector map. The lamp's own large-scale pattern and colour stay in the free field, so they cannot leak into the sky. This is the mis-specification that forces illumination corrections in classical pipelines.
- **Arc:** the scene is lines at known wavelengths. It pins the wavelength gauge.
- **Ramps of flats at several levels:** every pixel is seen at high charge, which science ramps reach only under bright stars.

**Cost.** The expensive part of a science frame is rendering the sky through the kernel, and calibration frames skip it. Ramps multiply the data volume by the number of reads kept. Once the detector map is trained on a sample, ramps can be compressed per pixel; that compression is an approximation to be checked.

**Pointing.** Raw frames carry telemetry pointing but no distortion. The reference distortion is a starting value, and the catalogue-free registration already built handles header errors (1.8 arcsec in programme 1476).

**Reference files.** They are a digest of calibration frames made by the instrument teams. Under your rule they serve as starting values and inputs, with a learned residual, never as fixed corrections.

**Recommendation, as three arms of experiment E3.**

| Arm | Calibration information used | Role |
| --- | --- | --- |
| a | Reference files as starting values, learned residual | Default to start |
| b | Arm a plus raw calibration frames as exposures | Expected best; needs an instrument with real calibration frames, such as MegaCam at CADC |
| c | None | The pure bet on diversity and quantity |

The bet is won if arm c reaches arm a within the number of exposures the archive holds.

**Levels of rawness, JWST as the example.**

| Level | File | Already applied by the pipeline | What using it adds |
| --- | --- | --- | --- |
| Calibrated | `_cal` | Ramp fit, dark, linearity, flat, flux calibration, distortion | The present inputs |
| Rate | `_rate` | Ramp fit, dark, linearity | The flat and the foreground-as-flat constraint become testable, at the same data volume |
| Ramps | `_uncal` | Nothing | Non-linearity, brighter-fatter, saturation, cosmic rays and stripes become identifiable |
| Calibration frames | Darks, flats, arcs | Nothing | Detector terms at high precision, with no sky |

## Small experiments

Six tests, each a few days, each on one new element, each with a rule for dropping that element. None needs the full method to exist.

(Diagram in the living doc: baseline first; then E1, E2 and E3 in parallel; then E4, E5 and E6 in order.)

E1 and E3 are highlighted because they test your two strongest hunches: any scale with learned profiles, and calibration from diversity alone.

| Test | Question | Data | Measure | Drop the element if | Size |
| --- | --- | --- | --- | --- | --- |
| Baseline | What does the present code score, fairly compared? | SMACS 20 arcsec box | Paired difference against the official pipeline; residual power by scale | Not a test: the reference for all others | Days |
| E1 | Does a learned profile with scale factored out represent a wide range of scales better than Gaussians, a grid and a hash-grid field? | The official SMACS and LMC mosaics as stand-in images; a synthetic sky with structure over three decades of scale. No instrument | Residual power by scale band at equal size; fit time; the same latent recovered when an image is resampled at 1, 1.4, 2 and 2.8 times | Less than a factor 2 in size, or 2 dB at equal size, over the best alternative | 2 to 3 days |
| E2 | Can a learned profile be rendered through a kernel accurately and cheaply? | The present learned NIRCam kernels | Error against brute-force quadrature for closed-form sub-profiles and for local sampling; time per exposure | Neither reaches 1e-3 within 10 times the present renderer's cost | 1 to 2 days |
| E3 | Does the flat calibrate itself from diversity alone? | LMC programme 1476 as `_rate` images, to download | Held-out score against number of exposures (50, 200, 800) for arm a (reference flat as a start) and arm c (nothing); recovered flat against the reference, as a comparison only | Not a drop rule: if arm c has not reached arm a by 800 exposures, calibration inputs are required for JWST | 3 to 4 days |
| E4 | Do raw ramps identify the detector and recover bright cores? | One NIRCam short-wavelength detector, about 20 exposures with bright and saturated stars, as `_uncal`, to download | Held-out late reads; held-out exposures at signal-to-noise above 50; saturated cores recovered from early reads | No gain over the calibrated-input fit above signal-to-noise 50, and no saturated core recovered | About a week |
| E5 | Is the finer sky predictable from the coarser sky, beyond a linear predictor? | Skies fitted so far on SMACS and LMC; raw held-out pixels | CRPS against zero prediction and against a linear predictor; rank histogram of the ensemble | Less than 10% gain over the linear predictor | 3 to 4 days |
| E6 | Does the predictor buy depth, and does it flag what it has not seen? | As E5, with a quarter of the exposures; structures absent from training injected into held-out raw exposures (a ring, a jet, an odd spectrum) | Equivalent exposure-time gain of the assisted fit; injected structures intact in the measured sky and flagged in the surprise map | Gain below 10%, or an injected structure is weakened in the measured sky | 3 to 4 days |

**Cost is measured in every test.** The target for the applied method is within 10 times one coadd pass over the same exposures.

**Arm b of the calibration test** needs real calibration frames. It waits for a ground instrument: one UNIONS tile of MegaCam raw frames with its biases and twilight flats.

**After the tests.** The scale ladder of version 1 stands: the SMACS box, the SMACS 8 arcmin union, COSMOS, then the all-sky backbone.

**To check first.**

- [ ] Sizes of the `_rate` and `_uncal` downloads for programme 1476 and for SMACS.
- [ ] Access to MegaCam raw and calibration frames at CADC for one UNIONS tile.

## Results to date (14 Nov 2026)

The living doc (https://claude.ai/artifact/DYP9UJuvosgpaz7BGG4yr5, tab "Version 2") is the current text; it also holds the sections "Scope and order", "First results", "Second round" and "Third round". The numbers are in `one-sky-v2-experiments-log.md`. In short:

- Kept: sky items with continuous position, size and orientation. Dropped by its own rule: a learned per-item profile (it did not beat Gaussian items in the synthetic test).
- On real data (SMACS, JWST NIRCam, six filters) a sky of 60,000 positive Gaussian items comes out flux-calibrated (sky over official mosaic 1.06 to 1.16 inside 1 arcsec; 0.96 to 1.03 at a 0.02 arcsec raster) with a diffraction-like kernel (90% to 97% inside 2 arcsec, core widening with wavelength). The grid sky carries 2.3 to 3.4 times the mosaic's flux with half of its kernel beyond 2 arcsec. The form of the sky fixes the gauge "far kernel wings against background"; no convention was added.
- Scoring each method's full prediction, the item sky predicts held-out pixels better than the grid sky and than the official pipeline from signal-to-noise 1 to 50 (95% of scored pixels), with a quarter of the sky numbers and a quarter of the steps. It is worse on bright cores and inconsistent at the faintest pixels (the gain disappears when the scorer adds its own background offset).
- Open: speed of the item rasteriser on the GPU (launch-bound; three rasterisations per step), wider items or a smooth component for diffuse light, bright cores, and the tests on raw inputs (flat from `_rate` images, ramps), the predictor across scales and the surprise map, none of which has run.

## What carries over from version 1

Most of version 1 stands as the measurement machinery; what changes is the sky form, the inputs and what is learned across the sky.

| From version 1 | Status in version 2 |
| --- | --- |
| Review of the note: eight findings | Stands |
| Gauge rule and table | Stands. Calibration frames and one reference each now supply the pins |
| Sky as photons, instrument as a probability rule, data channel | Stands. The channel gains the read index and the scene |
| Kernel pyramid; each instrument reads the sky at its own scale | Stands as the renderer. Levels are internal to rendering, no longer part of the sky |
| Likelihood from the model variance; Poisson for events | Stands. The ramp likelihood moves from later to first |
| Gaussian blobs for the sky | The special case of a fixed profile. Kept as the comparison arm in E1 |
| Lattice levels; new blobs at residual peaks | Dropped. Items are seeded by coverage |
| Convex sky step | Kept for fluxes at fixed profiles; lost for positions, sizes and latents |
| Steps 2 to 5: SMACS box, SMACS union, COSMOS, all-sky backbone | Kept as the scale ladder, after the small experiments |

## Declared approximations and open problems

Version 2 adds seven approximations and leaves five problems unsolved. This design is a synthesis and none of it has been tested.

| Approximation | Check |
| --- | --- |
| Profiles are non-negative and come from one learned family | Anything outside the family costs more items, never accuracy; E1 measures how many |
| A lamp or twilight scene is smooth and constant during the exposure | Residuals of flat frames |
| The detector map is trained on a sample of ramps, and ramps are then compressed | Compressed against full ramps on held-out exposures (E4) |
| A learned profile is rendered through the kernel by sub-profiles or local sampling | E2 |
| The score is per pixel, so joint structure is not scored directly | Rank histograms on pooled pixels; the weather papers find that joint structure emerges but report mesh artefacts |
| The predictor is trained on skies fitted by the same model | It is scored on raw held-out pixels, not on those skies |
| The sky is static, and radio and polarisation are out of scope | As in version 1 |

**Open problems.**

1. **Feed-forward inference is not designed.** The network that proposes items from raw data in one pass, which the scale-up relies on, does not exist yet.
2. **Crowded fields.** Seeding by coverage may leave too few or too many items where sources overlap.
3. **Non-convexity.** Positions, sizes and latents are fitted by gradient steps, with no guarantee.
4. **Error propagation.** The Fisher operator is the error model, and storing it compactly at survey scale is unsolved.
5. **Hard limits of the data.** Nothing passes the diffraction cutoff without assuming sparsity or positivity, and the largest scales stay tied to the foreground.

## References

Read in part on 13 Nov (abstract, or methods and results), not in full.

| Topic | Reference |
| --- | --- |
| Ensemble by perturbing the network, scored by CRPS; training against observations | [WeatherNext 3](https://arxiv.org/html/2609.03582v1); [Skillful joint probabilistic weather forecasting from marginals](https://arxiv.org/html/2506.10772v1) |
| Embedding field from heterogeneous sensors, at planetary scale | [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291v1) |
| Any-sensor inputs conditioned on wavelength | [Panopticon](https://arxiv.org/pdf/2503.10845) |
| Scale as a network input is only approximate | [Neural Gaussian Scale-Space Fields](https://arxiv.org/pdf/2405.20980) |
| Explicit primitives against learned radial bases, on large images | [Large Images are Gaussians](https://arxiv.org/html/2502.09039v1) |
| The instrument as a learned field | [Learning Lens Blur Fields](https://arxiv.org/pdf/2310.11535) |
| A short series of networks on data residuals | [R2D2](https://arxiv.org/abs/2403.05452) |
| No common sky frame in the astronomy foundation model | [AION-1](https://arxiv.org/pdf/2510.17960) |
| Priors from corrupted, multi-view data | [A Data-Driven Prism](https://arxiv.org/html/2510.05205v1) |
| A latent sky from many ground exposures | [ImageMM](https://arxiv.org/abs/2501.03002) |
| Fusion of JWST imaging and spectroscopy | [Fusion of JWST data](https://arxiv.org/pdf/2603.05014) |

Known by title or from memory only; verify before quoting.

| Topic | Reference |
| --- | --- |
| A coordinate network as a dictionary set by its first-layer frequencies | [Yüce et al. 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yuce_A_Structured_Dictionary_Perspective_on_Implicit_Neural_Representations_CVPR_2022_paper.html); [BACON](https://computationalimaging.org/?p=5514) |
| A shared network with a latent per item; a learned basis with coefficients | Functa (Dupont et al. 2022); DeepONet (Lu et al. 2021) |
| Self-calibration of the flat from dithered science frames | Holmes, Hogg & Rix 2012 |
| Generation across scales | [Inverse heat dissipation](https://arxiv.org/pdf/2206.13397); [Wavelet conditional renormalisation group](https://arxiv.org/pdf/2207.04941) |

**Not read.** Three November 2026 arXiv papers were rate-limited, one titled [Forward modeling linear variable filter data](https://arxiv.org/pdf/2611.12846). They may overlap with this design.
