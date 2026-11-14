# One sky at every scale: functional forms and plan

Nov 12, 2026 · Sébastien Fabbro

Living doc (the editable copy): https://claude.ai/artifact/DYP9UJuvosgpaz7BGG4yr5

Status: proposal from the review of `claude/field-operator-formulation.md`. Nothing has been run and Seb has not yet approved it.

## Summary

There is a more elegant form than the atom pyramid, and the pyramid turns out to be one way of computing it. I call it the **photon mixture**. It rests on three statements.

1. **Photons, not pixels.** The sky is a non-negative distribution of photons over direction, wavelength and time. An exposure is a rule that gives the probability that a photon lands in each data channel. Expected data are the sky pushed through that rule.
2. **Size is a coordinate.** Sky and instrument are both clouds of non-negative Gaussian blobs in (direction, log wavelength, angular size, spectral size). A datum is a sum of closed-form overlaps between sky blobs and that channel's blobs.
3. **Fitting is a mixture fit.** For a fixed instrument the problem is convex, with positivity as the only constraint and no prior. New blobs are added where the back-projected residual peaks, in position, wavelength and size.

| Requirement | How the photon mixture meets it |
| --- | --- |
| Any scale | A blob carries its own size, from 0.4 mas to the whole sky. A coarse instrument reads merged blobs, so its cost follows its own pixels. |
| Any wavelength | Log wavelength is a coordinate like position. A filter is a wide blob in wavelength, a spectrograph a narrow one. |
| Any instrument in scope | Imagers, spectrographs and photon-counting instruments differ only in the channel and in the blob cloud per channel. |
| Accuracy on point sources | A point source is one blob at the declared minimum size, at a continuous position. |
| Scalable and federated | Blob updates are sums over exposures, so each data centre ships sums, not pixels. |
| Gauges | The common blur becomes a translation along the size axis, pinned by one declared floor on the sky. |

**Your decisions, 12 Nov.**

- Scope: imagers, spectrographs and photon-counting instruments. Radio and polarisation are out of the first design.
- Sky form: an elegant form if one fits every requirement, otherwise the atom pyramid.
- Test bed after the SMACS 8-arcminute union: COSMOS.

**Status.** Nothing here has been run. Each part is established in its own field; the combination is untested. One quick web search found no equivalent method; that is not a literature review.

## Review of the current formulation

The note is sound as a ladder for near-infrared imagers; eight things stop it from being general. I read the project copy dated 13 Nov and assumed the copy on the `torchregress-harness` branch is the same text.

| # | Finding | Change to the note |
| --- | --- | --- |
| 1 | The gauge table is a special case. In §1.1 any invertible linear map common to all exposures is a gauge. What survives is every such map that keeps the response inside its allowed family. | State the rule in §2 and derive the table from it. Add the missing gauges listed in "Gauges" below. |
| 2 | "One absolute calibration" understates the flux gauge. It is a whole function of wavelength. | Pin it with one reference throughput curve. |
| 3 | "Pixel" is too narrow. A real, non-negative, localised response per pixel excludes photon events, measured-energy redistribution and ghosts. For a fringe response A3 is a restriction, not a reparametrisation. | Replace the pixel by a data channel. Say that A3 holds for localised responses only. |
| 4 | Wavelength is second class: a published filter curve, 5 nodes, 12 bumps. The quadrature error of 1.2e-3 holds for smooth spectra only. | Make the kernel a function of offset and wavelength jointly, with a centre in both. Transmission is then inside the kernel and a spectrograph is the same object. |
| 5 | Galactic cirrus is listed as foreground. It is static, fixed on the celestial sphere and passes through the response, so no data separate it from the sky. | Define foreground as radiance that is not fixed in celestial coordinates. Cirrus is sky. |
| 6 | One resolution per patch (A8) blocks both accuracy and scale. It gives the 0.26 residual in F090W. The full sky at 0.04 arcsec with 12 spectral terms is 4e15 coefficients. | Replace A8 with a sky that carries its own sizes. |
| 7 | The blur gauge is enforced on the wrong side. Dilating kernels to a floor puts a limit of the sky representation inside the operator. It is also where the gradient defect lives. | Declare the floor on the sky and leave kernels untouched. |
| 8 | Kernel positivity (the softmax) is an undeclared approximation. Once a blur gauge is declared, the effective kernel is the true one deconvolved by the reference, which need not be positive. | Add it to the ladder as a step, with its cost. |

**The score.** It cannot see angular scale, nor how blur is split between sky and kernel. Two bootstraps do not separate 1.77 \[1.68, 1.86\] from 1.80 \[1.61, 2.02\]. The plan adds a held-out residual power spectrum by scale, cross-instrument prediction, and paired differences per exposure.

**§7 question 1.** I agree with all four candidates. Foreground through the response and model-based variance matter most.

## The central idea: the photon mixture

One equation covers every instrument in scope: the expected data are the photon distribution of the sky pushed through a probability rule per exposure.

```latex
\mu_e(c)=\int_{\Omega}P_e(c\mid\omega)\,\big[\nu(d\omega)+\nu^{\mathrm{fg}}_e(d\omega)\big]+\beta_e(c),
\qquad \omega=(\boldsymbol\theta,\ \log\lambda,\ t),\qquad \nu\ge0,\quad P_e\ge0 .
```

- The sky is the non-negative measure ν on photon space Ω.
- The channel c is a pixel, a spectral channel or a single photon event.
- The rule P gives the probability that a photon at ω lands in channel c. Its total is the throughput, its mean is the geometry and its spread is the kernel. These are the three parts of step A3, now including wavelength.
- The foreground of exposure e is a second measure, not fixed on the sky, through the same rule.
- The term β holds what does not pass through the optics: instrument emission and dark current.
- The recorded data follow from the detector map and the noise law, as in §1.1 and §1.2 of the note.

**Size as a coordinate.** Write both the sky and the rule as sums of normalised Gaussian blobs in ω. Each blob has a centre and a covariance: angular size s and spectral size r for the sky, and the same for each kernel blob.

```latex
\nu=\sum_n a_n\,\mathcal N\big(\omega;\ \omega_n,\ \Sigma_n\big),\quad a_n\ge0,\quad \Sigma_n=\mathrm{diag}\big(s_n^2,\,s_n^2,\,r_n^2\big),
```

```latex
P_e(c\mid\omega)=A_e(c)\sum_m w_{cm}\,\mathcal N\big(\omega;\ \bar\omega_e(c)+\delta_{cm},\ S_{cm}\big),\quad w_{cm}\ge0,\quad \sum_m w_{cm}=1 .
```

The expected datum is then a double sum of closed-form overlaps (static sky; time is added the same way):

```latex
\mu_e(c)=A_e(c)\sum_n\sum_m a_n\,w_{cm}\;\mathcal N\big(\omega_n-\bar\omega_e(c)-\delta_{cm};\ \Sigma_n+S_{cm}\big)+\dots
```

Four consequences follow from this one line.

1. **The blur gauge is a translation.** Only the sum of the two covariances appears. Adding a common matrix to every sky blob and removing it from every kernel blob changes nothing. The same holds along the spectral size. One declared floor on the sky sizes pins both.
2. **Any scale at fixed cost.** When a kernel blob is much larger than a group of sky blobs, the group acts as one blob with the summed flux, the mean position and the total covariance. A coarse instrument reads these merged blobs and never touches the fine ones.
3. **Any instrument.** An imager has kernel blobs wide in wavelength. A spectrograph has narrow ones, with a centre that moves across the detector. For a photon-counting instrument the channel is the event itself.
4. **The fit is a mixture fit.** For a fixed instrument and Poisson data the amplitude update is the Richardson–Lucy step written over blobs. It keeps positivity and needs no step size.

```latex
a_n\ \leftarrow\ a_n\;\frac{\sum_{e,c}R_e(c,n)\,d_e(c)/\mu_e(c)}{\sum_{e,c}R_e(c,n)},
\qquad R_e(c,n)=A_e(c)\sum_m w_{cm}\,\mathcal N\big(\omega_n-\bar\omega_e(c)-\delta_{cm};\ \Sigma_n+S_{cm}\big).
```

**When to add a blob.** Back-project the weighted residual onto a trial blob at any position, wavelength and size. The sky is complete when this map is below a threshold everywhere. Otherwise a blob is added at its peak.

```latex
\eta(\hat\omega)=\sum_{e,c}R_e(c,\hat\omega)\,\frac{d_e(c)-\mu_e(c)}{V_e(c)}\ \le\ \tau
\qquad\text{for all}\quad \hat\omega=(\boldsymbol\theta,\ \log\lambda,\ s,\ r).
```

This is the optimality condition of a convex problem over measures, not a detection step. A blob is not a source: a galaxy is many blobs and the model has no notion of an object.

| Part | Established as | New here |
| --- | --- | --- |
| Sky as a positive measure, convex fit, blobs added at the residual peak | Off-the-grid sparse recovery by conditional gradient; CLEAN is the same step | Applied to optical, infrared and X-ray exposures jointly |
| Positivity as the only regulariser | Non-negative least squares; super-resolution of positive measures | Used to select the blur gauge |
| Mixture fit with a blur per datum | Extreme deconvolution; Richardson–Lucy as an EM algorithm | Blur is the learned kernel, per channel |
| Size as a coordinate | Scale space; multi-scale CLEAN; anti-aliased Gaussian splatting | Kernel and sky share one blob space |
| Fast sum of Gaussian overlaps | Fast Gauss transform | Sizes vary per blob |
| Instrument as a probability rule | X-ray practice: effective area, energy redistribution, point-spread function | Extended to every instrument, and learned |

## Sky model

The sky is one list of non-negative blobs, each with a position, a wavelength, an angular size, a spectral size and a flux. Your earlier choice of "grid plus splats" is the special case with a single lattice level.

**Two ways of placing the same object.**

|  | Lattice blobs | Free blobs |
| --- | --- | --- |
| Position | Centres of nested HEALPix cells at one order | Continuous |
| Angular size | 0.8 × the cell size; the sum is then flat to 1.3e-5 (checked numerically in one dimension on a regular lattice; HEALPix centres are not regular, so this is to be re-checked on the sphere) | Continuous, never below the floor |
| Fitted | Flux only: linear and convex | Flux, position and size |
| Carries | Diffuse and extended light | Compact structure; a point source is one blob at the floor |

Both are stored under one key: the HEALPix order given by the size, and the cell index given by the position.

| HEALPix order | 12 | 16 | 20 | 22 | 24 | 29 |
| --- | --- | --- | --- | --- | --- | --- |
| Cell size (arcsec) | 52 | 3.2 | 0.20 | 0.050 | 0.013 | 0.0004 |

**Wavelength.** A spatial blob with a spectrum is a set of blobs that share position and angular size and differ in log wavelength and spectral size. The spectral blobs sit on a lattice in log wavelength, refined per spatial blob where a spectrograph observes it. A narrow free blob is an emission line. This replaces the 12 fixed bumps.

**Positivity.** Every flux is non-negative. This is sufficient for a non-negative sky and slightly restrictive: a sharp hole in smooth light must be built from the blobs around it.

**The floor.** A declared map gives the smallest angular and spectral size allowed at each place. I propose one third of the sharpest kernel core that covers the place. The map is a product of the fit, reported with the sky.

**Many descriptions of one sky.** A wide blob equals a spread of narrow ones, so the blob list is not unique although the sky is. The rule: a blob is split only where the residual map demands it, and merged back when its children carry no evidence. Predictions do not depend on this choice.

**Time.** The sky is static first, with residuals flagged. Variable sources are blobs with a finite extent in time, added later in the same way.

**Later, if data ask for it.** Blobs with a full 2×2 angular covariance. The overlap stays closed-form.

## Response model

The response keeps the three parts of step A3 (geometry, kernel, amplitude) and extends each to wavelength. It stays one learned field operator; only its output changes, from samples on one fine grid to blobs on a pyramid of sizes.

**Geometry: where and at what wavelength a channel looks.** It stays separate from the kernel, whose centroid is pinned to zero (§7 question 5). Scales much larger than the kernel need the geometry and amplitude alone, so it must not be hidden in the kernel.

- Imager: pointing, then focal-plane distortion, then detector placement, as in A3. The wavelength centre comes from the filter.
- Spectrograph: position and wavelength centre both vary across the detector.
- Ground: add refraction as a learned function of airmass and parallactic angle, and a low-order warp per exposure.

**Kernel: one field that outputs blobs.**

```latex
\big\{\,w_{cm},\ \delta_{cm},\ S_{cm}\,\big\}_{m=1}^{M}=f_\vartheta\big(m\ \big|\ \mathbf p_c,\ \tilde{\mathbf x}_c,\ h_e\big),
\qquad h_e=h_{\mathrm{inst},f}+\tau_{\mathrm{det}}\,h_d+\tau_{\mathrm{ep}}\,h(t_e)+\tau_{\mathrm{exp}}\,\mathbf z_e .
```

- **First version.** Blob offsets and sizes are fixed on a pyramid: a small stencil at each size level, out to the full field. The network gives only the weights, through one softmax over all blobs. This is the present kernel network with its samples rearranged. It samples the kernel at about 1200 points; it is not a low-order basis.
- **Later.** Offsets and sizes are outputs too.
- **Inputs.** Both the focal-plane position (continuous optics) and the detector position (pixel effects), which uses optical continuity across detectors (§7 question 4). An optics kernel convolved with a detector kernel is a sum of covariances in blob form, so the composition is available as an ablation.
- **Frame.** Blobs are defined in the detector frame and carried to the sky by the local Jacobian of the geometry, so rolls are exact.
- **Wavelength.** A filter is a few blobs in log wavelength that cover the band. They replace the 5 quadrature nodes. The published curve is a starting value only.
- **No window.** The coarsest level spans the field, so far wings and scattered light join the kernel continuously.

**Amplitude: pixel area from the geometry × structured gains.**

```latex
\log A_e(c)=\log\Big|\det\frac{\partial\boldsymbol\theta_e}{\partial\mathbf x}\Big|+\gamma_{\mathrm{inst},f}+\gamma_d+\gamma(t_e)+\gamma_e+\eta_d(\tilde{\mathbf x},\lambda),
\qquad \int\eta_d\,d^2\tilde x=0 .
```

The scalars are random effects and the spatial field has zero mean per detector. Gains can then no longer hide in the field, which is defect 2 of the note (§7 question 6). At survey scale this is relative photometric calibration from overlaps, solved with the sky and without a catalogue.

**Foreground.** A smooth learned field of focal-plane position, wavelength and time. Its inputs are metadata only: Sun angle, ecliptic latitude, airmass, Moon. It passes through the same amplitude as the sky (step A4), which ties the flat field to the background. No physical model is used (§7 question 7).

**Detector map.** A later, separate learned stage with unit slope at low signal. The sky step stays linear by linearising it at each outer iteration.

| Class | Channel | What is particular |
| --- | --- | --- |
| Space imager | Pixel | Kernel wide in wavelength |
| Ground imager | Pixel | The exposure code dominates the kernel (seeing); strong, variable foreground |
| Slitless spectrograph | Pixel | Detector position depends on direction and wavelength; kernel narrow in wavelength |
| Integral-field or fibre spectrograph | Pixel on a trace | Aperture footprint on the sky; line spread in wavelength |
| SPHEREx | Pixel | Wavelength centre varies across the detector; 6 arcsec pixels |
| X-ray and GALEX events | Photon event: position, measured energy, time | Pointing per event; kernel includes the energy redistribution; Poisson likelihood on events |

**Fallback.** If a diffraction-limited kernel needs too many Gaussian blobs, each level keeps a signed stencil with tabulated overlaps instead. The sky side does not change.

## Rendering and solver

Rendering is a sum of Gaussian overlaps between two blob clouds, organised by one tree: HEALPix order for size, cell index for position. Each instrument then costs in proportion to its own channels, not to the finest detail on the sky.

**Rendering.**

- **Reading level.** A kernel blob reads sky blobs merged to about one third of its own size. The merge keeps total flux, mean position and total covariance. Its error is third order and is checked against one level finer.
- **Upward pass.** Merged blobs are computed once per sky step and shared by all exposures.
- **Cost.** A 10 arcsec kernel on a 0.04 arcsec sky is 250 000 taps when applied directly. A 13×13 stencil on 6 to 7 size levels is about 225 overlap-equivalents per pixel. That count assumes coarse levels are evaluated on a coarser pixel grid and interpolated.
- **Not known.** The gain over the present overlap-add FFT renderer (1.2 s per step). It must be measured.

**Sky step, operator fixed.** The problem is convex in the fluxes, and convex as a problem over measures. The positions and sizes of free blobs are not convex parameters.

- **Flux.** Poisson data use the Richardson–Lucy step over blobs, accelerated. Read-noise-limited data use non-negative least squares by projected conjugate gradients, as the earlier splat code already does.
- **Multigrid.** Coarse levels are fitted on residuals binned to their size. This cures the slow convergence of large scales that stopped Adam at 1e8 to 1e9 coefficients.
- **Free blobs.** Positions and sizes take gradient steps after each flux solve. New blobs enter at peaks of the residual map; blobs at zero flux leave.

**Operator step, sky fixed.** Gradient steps on the kernel field, codes, geometry and gains, summed over sky cells. At a converged sky the gradient is exact without differentiating through the sky solve (the variable-projection argument of the 7 Nov design memo). Cross-fitting is an option: the sky from one half of the exposures, the operator trained on the other half.

**Federation.** Every sum above runs over exposures. A centre that holds exposures computes two numbers per blob, plus the operator gradient, and ships those. Pixels never move. Sky cells are HEALPix cells of low order with a halo of one kernel reach.

**Photon events.** The likelihood needs the model at each event and the integral of the model over all channels. The integral is closed-form per blob: amplitude × exposure time. No pixel grid is needed.

## Gauges

One rule generates every gauge: any invertible linear map G on the sky, common to all exposures, that keeps each response inside its allowed family.

```latex
I\ \to\ G\,I,\qquad \mathcal R_e\ \to\ \mathcal R_e\,G^{-1}\quad\text{for every exposure } e .
```

A narrower response family leaves fewer gauges. Diversity removes a gauge only when it moves the response relative to the sky, as dithers and rolls do for distortion.

| Gauge | What changes | Broken in part by | Pin: one declared condition |
| --- | --- | --- | --- |
| Flux scale, a function of wavelength | Sky × c(λ), amplitude ÷ c(λ) | Nothing | The throughput curve of one reference instrument, taken as the definition |
| Common blur | Sky sizes + B, kernel sizes − B | Positivity on both sides; unresolved sources | The floor map on sky sizes; the sharpest non-negative sky is selected |
| Common spectral blur | The same along spectral size | Spectrographs that see narrow lines | A spectral floor |
| Common shift, per wavelength | Sky shifted by a(λ) | Nothing | Zero mean shift for one reference instrument, at every wavelength |
| Common rotation and plate scale | Sky rotated and rescaled | Nothing; rolls do not break it | The mean header pointing of one reference instrument |
| Common distortion beyond those | Sky warped smoothly | Dithers and rolls, since distortion is tied to the detector | None needed with large dithers; report the residual |
| Common wavelength warp | Wavelength axis remapped | Nothing | The wavelength solution of one reference instrument |
| Kernel centroid against geometry | Internal to the response | Nothing | Kernel centroid is zero |
| Sky against foreground | Smooth light moved between them | Foreground that varies with time or pointing | Light static on the celestial sphere is sky, by definition |
| Far kernel wings against background | Wing light moved to a smooth additive term | Bright sources at many detector positions | No kernel window; the foreground has no sky-fixed part |
| Detector slope against throughput | Low-signal slope × k, amplitude ÷ k | Nothing | Slope is 1 |
| Detector gain against throughput field | Internal to the amplitude | Nothing | The spatial field has zero mean per detector |

**Not a gauge of the expected signal, but pinned the same way:** the noise scale against the model-error floor. Both are fitted on held-out data and reported.

**Caveat.** Only the Gaussian part of the common blur is a translation. I expect positivity of the kernel weights to suppress the rest, but this is not proven. The synthetic test in step 1 of the plan checks it.

## Likelihood and scoring

The variance comes from the model, not the data, and photon events use the exact Poisson form. This is level L1 of step A7, plus the event case.

**Integrated pixels.** The variance is recomputed from the current model at each step and held fixed in the gradient. This avoids both the flux bias of data-based errors and the bias from differentiating through the variance.

```latex
-\log L=\frac12\sum_{e,c}\frac{\big(d_e(c)-\mu_e(c)\big)^2}{V_e(c)},
\qquad V_e(c)=\sigma_r^2+\frac{\bar\mu_e(c)}{g}+\big(f\,\bar\mu_e(c)\big)^2,\quad \bar\mu=\text{current model, held fixed}.
```

**Photon events.** The first term is the expected number of events, closed-form per blob through its exposure E. The second runs over the recorded events.

```latex
-\log L=\sum_n a_n\,E_e(n)+\int\!\beta_e\,dc\;-\sum_{i\,\in\,\text{events}}\log\mu_e(c_i).
```

**Correlated noise.** Row and column stripes stay as nuisances, as now. Inter-pixel capacitance belongs to the detector map.

**Outliers without a mixture (§7 question 9).** The earlier mixture failed because it could explain bright cores. Core errors scale with flux; cosmic-ray hits are positive and do not. I propose the fractional floor for cores and a heavy tail on positive residuals only, with a scale that does not grow with the model. This is untested and goes in the ablations.

**Variability scales (§7 question 8).** Keep the type-II update for random effects. Decide each function axis by paired held-out differences per exposure, with a bootstrap over dither groups. Pairing removes the exposure-to-exposure scatter, so weak real effects become visible as data grow. A refitted likelihood ratio is the cross-check.

**Additions to the score.**

| Addition | What it shows |
| --- | --- |
| Paired difference per exposure between two methods | Whether 1.77 and 1.80 differ |
| Cross-power spectrum of held-out residuals, between two disjoint exposure groups, by angular scale | At which scales the model fails; the noise cancels |
| Cross-instrument prediction: fit the sky without one instrument, predict it with a frozen operator | Whether blur is split correctly between sky and kernel |
| Injection and recovery by scale band | Recovery of small and large scales where data support them |
| Peak of the residual map over position, wavelength and size | Whether anything is left to add |

## Plan and test ladder

Six steps, each on a wider scale than the last, and none starts before the gate of the one before is passed. The SMACS data stay as the engineering bed because they are on disk; COSMOS is the first real test of scale.

(Diagram in the living doc: steps 0 to 5 in order, step 1 highlighted.)

Step 1 is highlighted because it decides, before any large build, whether kernels are Gaussian blobs or stencils and whether the blur gauge behaves as claimed.

| Step | Data | Build | Gate |
| --- | --- | --- | --- |
| 0 Repair and measure | SMACS 20 arcsec box, present `fieldop` code | Zero-mean throughput field with scalar gains; "static is sky" in place of the pedestal at exposure medians (the mean background over exposures is assigned to the sky); variance from the model; paired and by-scale scores; rerun at matched length | The regression from 1.77 to 3.29 is explained; the paired difference against the official pipeline has an interval |
| 1a Blob count | Present learned kernels and the public empirical kernel libraries, as targets only | Fit each with pyramid blobs; error against count | At most 1e-3 of the peak with about 1200 weights per control point (13×13 on 7 levels). Otherwise the stencil fallback |
| 1b Synthetic scales | A known sky with structure from 0.01 arcsec to 10 arcmin, seen by three mock instruments: sharp and narrow, medium, coarse and wide | Fit sky and kernels jointly | Each scale band is recovered where an instrument supports it; the floor recovers the true kernel widths when unresolved sources exist; a common blur leaves predictions unchanged |
| 1c Renderer | Same sky and kernels as step 0 | Tree renderer against the present renderer | Agreement to 1e-3; time per step measured |
| 2 Photon mixture, small | SMACS 20 arcsec box, NIRCam then HST | Sky blobs, kernel pyramid, convex sky step | Paired held-out difference against the pipeline (1.80) below zero; worst F090W point-source residual from 0.26 to below 0.05; no worse than the pipeline at signal-to-noise above 50 |
| 3 Many instruments | SMACS 8 arcmin union: JWST, HST, Spitzer, SPHEREx, GALEX | Reading levels and merging; SPHEREx as the first spectral instrument; a slitless spectrograph and photon events if the data exist | Adding an instrument worsens no other held-out score (the gate that failed before); cost per instrument follows its own pixels; cross-instrument prediction passes |
| 4 Degree scale | COSMOS: ground imagers, HST, JWST, Spitzer, GALEX, X-ray events, grism spectra | Sky cells on HEALPix, sums shipped between cells, foreground through the response | Light on arcminute-to-degree scales agrees between instruments; exposures at different seeing predict each other; the JWST operator transfers frozen |
| 5 All-sky backbone | SPHEREx and WISE exposures | Coarse levels over the whole sky, refined where sharper data exist | Cost per sky cell is flat with area; the coarse sky in the COSMOS cell agrees with step 4 |

**If step 2 fails at bright cores.** If the deficit at signal-to-noise above 50 persists and grows with peak flux, the cause is the detector, not the sky form. The learned detector map comes next, before step 3.

**Sizes, as guesses.** Step 0 is days. Step 1 is about a week. Step 2 is one to two weeks. Steps 3 to 5 depend on what steps 1 and 2 show.

**To check before step 3 and step 4.**

- [ ] Are the NIRISS slitless exposures of programme 2736 in the union download?
- [ ] Does the Chandra archive hold event files for SMACS J0723, and are GALEX photon lists reachable?
- [ ] Which COSMOS ground exposures are reachable as individual exposures: CFHT at CADC, Subaru HSC?
- [ ] Download size and compute for COSMOS; neither is estimated here.

## Declared approximations, risks and falsifiers

The design adds nine approximations to the ladder of the note. Each has a stated check.

| Approximation | Check or control |
| --- | --- |
| The sky is a sum of non-negative Gaussian blobs | The finest level alone represents any non-negative sky down to the floor; sharp holes are the weak case |
| The kernel is a sum of non-negative Gaussian blobs on a pyramid | Step 1a; signed stencil as fallback |
| Sky blobs are merged at the reading level | Third-order error; compared with one level finer |
| Coarse kernel levels are evaluated on a coarser pixel grid | Compared with full evaluation on a sample of exposures |
| The geometry is linear across one blob, and a spectral trace is straight across one blob | Blobs are refined until it holds |
| The sky is static, and nothing changes within an exposure | Residuals flagged; photon events carry their own pointing |
| Integrated pixels are Gaussian, with the model variance held fixed in the gradient | Exact ramp likelihood later |
| No detector map yet: inter-pixel capacitance and brighter-fatter stay in the kernel | Residual stacks by peak flux; the detector stage follows if they grow |
| Radio and polarisation are out of scope | The atom-to-channel contract leaves room for both |

**Risks.**

1. **Kernels may need too many blobs.** Diffraction spikes and rings are far from Gaussian. Step 1a measures it.
2. **Smooth light may converge slowly.** Overlapping blobs are ill-conditioned. Multigrid is the planned cure; lattice-only levels for diffuse light are the fallback.
3. **The operator fit is not convex.** Only the sky step is. Alternation, cross-fitting and the gates carry the rest.
4. **Positivity biases faint light upward.** A non-negative fit can only follow noise in one direction. The foreground term absorbs the mean; injection on empty sky measures what is left.
5. **Adding blobs at residual peaks could bias crowded fields.** The threshold comes from the null of negative peaks, as in the earlier splat code. Injection by flux checks it.
6. **An elegant form is not a better score.** The atom pyramid stays available inside the same framework.

**What would falsify the design.**

| Result | Conclusion |
| --- | --- |
| Step 1b: with unresolved sources present, the floor does not recover the kernel widths | The gauge claim is wrong. Fall back to one declared reference kernel |
| Step 2: the F090W point-source residual does not drop | The sky form was not the cause. Look at the kernel core and the detector |
| Step 3: adding an instrument worsens another | One shared blob space is not enough. Diagnose by scale with the cross-power score |
| Step 4: large scales disagree between instruments | The foreground form is too weak to separate sky from foreground |

## References

All cited from memory and not re-checked today. Verify before quoting them elsewhere.

| Topic | Reference |
| --- | --- |
| Blobs added by conditional gradient, then moved | Boyd, Schiebinger & Recht 2017, SIAM Journal on Optimization; Denoyelle, Duval, Peyré & Soubies 2019, Inverse Problems |
| Positive measures need no separation and no regulariser | Schiebinger, Robeva & Recht 2018, Information and Inference; Slawski & Hein 2013, Electronic Journal of Statistics |
| Gradient descent on blob positions and fluxes reaches the global optimum when over-parametrised | Chizat 2022, Mathematical Programming |
| Mixture fit with a blur per datum | Bovy, Hogg & Roweis 2011, Annals of Applied Statistics |
| Richardson–Lucy as an EM algorithm | Shepp & Vardi 1982, IEEE Transactions on Medical Imaging |
| Size as a coordinate | Lindeberg 1994, Scale-Space Theory in Computer Vision; Cornwell 2008, multi-scale CLEAN; Yu et al. 2024, Mip-Splatting |
| Fast sums of Gaussians; sparse operators in multi-scale bases | Greengard & Strain 1991; Beylkin, Coifman & Rokhlin 1991 |
| Deconvolution to a declared finite resolution | Magain, Courbin & Sohy 1998, ApJ 494, 472 |
| Relative calibration from overlaps; self-calibration | Padmanabhan et al. 2008, ApJ 674, 1217; Holmes, Hogg & Rix 2012 |
| Variable projection | Golub & Pereyra 2003, Inverse Problems 19, R1 |

**Surfaced by today's search, not read.** "Joint Deconvolution of Astronomical Images in the Presence of Poisson Noise" (arXiv:2403.13933) and PolyCLEAN (conditional gradient for radio imaging). Both are neighbours of this design and should be read before any claim of novelty.

**Project documents used.** `claude/field-operator-formulation.md`, `claude/field-operator-handoff.md`, `claude/operator-aggregate-design-memo.md`, `claude/forward-operator-spec.md`, `claude/union-fit-design.md`.
