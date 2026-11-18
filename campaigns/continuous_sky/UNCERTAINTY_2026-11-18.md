# An uncertainty for the fitted sky, and its calibration on the known-truth problem (18 Nov 2026)

New code: `fieldop/uncertainty.py` (estimators, product, command line), two small hooks (`--train-half` in `fieldop/cli.py` and
`fieldop/data.py`), `tests/test_uncertainty.py` (11 tests, 6 s), `tests/truth/uncertainty_calibration.py` (calibration driver).
Results: `tests/truth/results/uncertainty/` (saved fits `state_*.pt`, posterior draws `post_*.pt`, Gauss-Newton steps of the half
and seed fits `pol_*.pt`, `summary_grid_n8.json`, `summary_items_n8.json`, `exact_grid_s0_centre.json`, `corr_and_step.json`, an example
product file `example_grid_s0_circulant.fits`; 45 MB).

Nothing in the fit was changed. `--train-half` defaults to -1 (all exposures) and its branch in `load_field` is not entered then; the model,
sky, training and kernel code are untouched, so the model state of a default run is unchanged and the recorded default state hashes
(`tests/tsky_results/synth_*_defaults_v2.json`) stay valid. I did not re-run the hash proof (`tests/synth_small.py`); I ran
`tests/test_defaults_v2.py` (19 passed).

## 1. The product

A band is a vector of spectral weights $\phi$ over the $K$ spectral coefficients: a filter's bandpass-averaged basis (`model.Phi_band[f]`) or
a unit vector for one coefficient. The sky of the package is the bilinear (tent) interpolant of the coefficient grid $A_{j}$ at nodes $x_j$
(pitch $p$). The product at resolution $s$ is that continuous sky convolved with a unit-integral circular Gaussian $g_s$, evaluated on an
output grid (by default the node grid; any output positions are possible, the weights are separable):

```latex
% product (sky image of band phi at resolution s), its standard deviation, its error correlation length
I_s(x) \;=\; \sum_j (\phi\cdot A_j)\,(T * g_s)(x - x_j), \qquad
  T(u) = \prod_{i=1,2}\max\!\big(0,\,1 - |u_i|/p\big),\quad \sum_j (T*g_s)(x-x_j) = 1

\sigma_s(x)^2 \;=\; k_{x}^{\mathsf T}\,\big[F^{-1}\big]_{AA}\,k_{x}, \qquad
  k_{x} = \phi \otimes (T*g_s)(x - x_{\cdot}), \qquad
  F = [J_A\;L]^{\mathsf T}\,W\,[J_A\;L] + \mathrm{diag}(\mu I,\,0)\ \text{on the background gauge}

\rho_s(r) = \frac{\langle \delta I_s(x)\,\delta I_s(x+r)\rangle}{\sigma_s(x)\,\sigma_s(x+r)}, \qquad
  \ell_s \;=\; r \ \text{such that}\ \overline{\rho_s}(r) = \tfrac12 \quad(\text{azimuthal average over the covered region})

% supported resolution
\alpha(s, x) = -\frac{\partial \ln \sigma_s(x)}{\partial \ln s}, \qquad
  s^\ast(x) = \min\{\, s : \alpha(s', x) \le 2 \ \ \forall\, s' \ge s \,\}
```

Why $\alpha = 2$. For white noise, the noise of an element falls as its linear size ($\alpha = 1$). For a sky that is a deconvolution of a
Gaussian response of width $p_{\rm eff}$, $\sigma_s^2 \propto 1/(s^2 - p_{\rm eff}^2)$, so $\alpha = s^2/(s^2 - p_{\rm eff}^2)$, which is also the
ratio of the element's variance to the variance it would have without the blur. $\alpha = 2$ is the scale where the noise per element is
twice (in variance) what the smoothing alone would give, i.e. where the noise starts rising faster than the resolution gained; for a Gaussian
response $s^\ast = \sqrt2\,p_{\rm eff}$. The scan goes from coarse to fine and stops at the first crossing, because at the finest scales the
weak prior takes over and $\alpha$ falls back towards 1 (that regime is prior, not resolution). $s^\ast$ is computed per node (a map).

On the truth problem (response core 0.07 arcsec, pixels 0.10 arcsec, sky pitch 0.05 arcsec, 8 exposures): $s^\ast$ = **0.111 arcsec**
(median over the region seen by every exposure; 10 to 90 per cent of nodes 0.102 to 0.132). For a Gaussian response that is
$p_{\rm eff} = 0.078$ arcsec, which is the quadrature sum of the core (0.07), the pixel (0.10/$\sqrt{12}$ = 0.029) and the tent (0.05/$\sqrt6$ =
0.020): 0.078. Noise against scale (median over the inner region, surface-brightness units, pixel noise 1 per exposure):

| smoothing width s [arcsec] | 0.0625 | 0.075 | 0.10 | 0.125 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|---|
| sigma_s, posterior draws | 0.99 | 0.43 | 0.163 | 0.106 | 0.081 | 0.056 | 0.035 |
| sigma_s, Fourier approximation (section 2c) | 0.90 | 0.41 | 0.165 | 0.107 | 0.081 | 0.056 | 0.035 |
| alpha between this and the next width | 4.6 | 3.4 | 1.9 | 1.5 | 1.25 | 1.17 | |

Correlation length of the errors (lag where the correlation falls to 1/2): 0.10 arcsec at s = 0.10, **0.13 arcsec at s = 0.111**, 0.21 at
0.15, 0.31 at 0.20 (the same for both sky forms). Prior sensitivity (Fourier model): dividing the prior weight $\mu$ by 10 changes
$\sigma_s$ by 0.2 per cent at s = 0.10 arcsec and by 16 per cent at s = 0.07 arcsec: below the supported resolution the number depends on the
prior, above it on the data only.

## 2. The analytic estimator (linearised posterior)

### 2a. What is held and what is marginalised

The model of the training pixels is linear in the sky coefficients $A$ and in the per-exposure background coefficients $b_e$ (6 Legendre
terms) once everything else is held at its fitted value:
$m = J_A A + L b + (\text{held terms})$. `SkyNormalOperator` builds $F$ above with
* $W = \mathrm{diag}(w_i)$, $w_i = \min(1, \delta/|z_i|)/\sigma_i^2$: the inverse variance from the model's own noise terms at the fit
  (`model.sigma`, learned scale and floor included) times the Huber weight of the fit's loss at the fit's residual $z_i$ (Gauss-Newton weights
  of the Huber loss; 40 of 14,746 pixels below 1 on the truth problem);
* held, not marginalised: the response (kernel, throughput, distortion), registration, gain, per-pixel gain, detector pattern, stripes, noise
  terms;
* marginalised: all 6 background terms of every training exposure;
* gauge. A sky polynomial of order up to 2 over the field is exactly degenerate with the per-exposure quadratic backgrounds (a sky gradient
  $g\,x$ seen by an exposure dithered by $c_e$ is $g\,x_{\rm det} + g\,c_e$: a linear plus a constant background term). The fit fixes only
  the mean constant (pedestal gauge) and leaves the other five directions to the optimiser and the warm start. The estimator (default
  `bkg_gauge="all"`) holds the training mean of each of the 6 terms at its fitted value, so **those six large-scale sky modes are not in
  sigma** (their size is measured in section 4). With `bkg_gauge="constant"` they get the prior variance $1/\mu$ instead (sigma then depends on
  the prior at every scale).
* a weak prior $\mu = \bar d/\rho^2$ per coefficient, $\bar d$ the median node information, $\rho$ = 10 (`--unc-prior`): it only matters below
  the supported resolution (above).

$J_A v$ is the model's own batched render with the sky coefficients replaced by $v$ (`sky_override`), and $J_A^{\mathsf T} r$ the backward
pass of the same render; one application of $F$ is one forward and one backward render of every training pixel (measured: the backward costs
0.8 of the forward, so one application is 1.8 forward-render equivalents). The inverse is never formed.

### 2b. Solving

Preconditioned conjugate gradients. Preconditioner of the sky block: a Fourier (circulant) model of $F_{AA}$ built from its response to a
delta at the centre node for each spectral coefficient ($K$ applications), $K\times K$ per frequency, plus $\mu$, scaled by the relative
coverage $d_j = (J_A^{\mathsf T} w)_j/(J_A^{\mathsf T} w)_{\rm centre}$; Jacobi on nodes with $d_j < 0.05$; background block: exact $6\times6$
blocks. On the truth problem it reaches a relative residual of 1e-2 in 35 iterations where plain Jacobi was still at 0.36 after 40. Two ways
of getting $\sigma$:
* **posterior draws** ("perturb and solve"): $x = F^{-1}\big([J_A\;L]^{\mathsf T} W^{1/2} z + [\sqrt\mu\, z', 0]\big)$ with $z, z'$ standard
  normal has covariance exactly $F^{-1}$; smoothing the sky part of $n$ draws gives $\sigma_s$ at every pixel and every scale (relative error
  $1/\sqrt{2n}$). Default: 60 iterations per draw. Measured on one draw against a 100-iteration solution: variance at s = 0.10 arcsec 0.979 of
  the converged value (2 per cent low), 0.987 at 0.07.
* **exact**: one solve $F x = k_x$ per requested pixel and scale, $\sigma^2 = k_x\cdot x$. A smooth functional converges fast: at the centre
  at s = 0.111 arcsec the value was converged to 1e-7 by iteration 40 (114 to reach a residual of 1e-6). Exact 0.1324, posterior draws
  (pooled, inner median) 0.1330, Fourier approximation 0.1321.

### 2c. A cheap approximation

`circulant_sigma`: the Fourier model alone (uniform coverage, background terms ignored), scaled by $1/\sqrt{d_j}$. It costs nothing beyond the
preconditioner. On the truth problem it agrees with the draws to 1 per cent at s >= 0.10 arcsec, 5 per cent at 0.075 and 11 per cent at
0.05 (inner region; table in section 1). It does not know about chip gaps, edges or coverage gradients beyond the local scaling.

### 2d. The item sky

Propagated through the grid it is rasterised on: the same $F$ with $A$ the raster. This assumes that the item parametrisation can move the
raster in any direction near the fit, which it cannot exactly (items are positive and finite in number). I did not linearise in the item
parameters: an item with a coefficient near zero has zero derivative with respect to its position, widths and angle, and items can exchange
roles, so that Gauss-Newton matrix is singular by construction (not measured).

### 2e. The estimator the posterior describes: one Gauss-Newton step

The calibration (section 4) shows that the fit's own sky is NOT the estimator the linearised posterior describes: positivity and an
incomplete optimisation make its noise about half the posterior's and add a bias. The command line therefore writes, by default
(`--unc-polish 1`), **SKY = the fit plus one Gauss-Newton step** with the response held,
$A_{\rm pol} = A_{\rm fit} + [F^{-1}(J^{\mathsf T}W r - \mu A_{\rm fit})]_A$ (one more solve; for the grid sky this is exactly the weighted
least-squares minimiser with the weak prior; positivity is not imposed, so the product can be negative), and SKY_FIT = the fit's own sky at
the same resolution. The step changes the sky by 1.6 (grid) to 1.7 (items) sigma rms at the supported resolution, 1.4 of which is the zero
level (section 4).

## 3. The empirical estimator (no linearisation)

* `--train-half 0|1` (new option of the fit, default -1 = all): per filter, the training dither groups sorted by name; rank parity = half
  is kept. Implemented in `load_field` (`fieldop/data.py`) and identically in the truth driver. Not run on real files here (only parsed).
* `difference_map` (fitted skies of two states, from `--unc-cmd diff --unc-state-b`) and `diff_products` (`--unc-cmd diff-files`, two
  products of `map`, e.g. the polished skies of the two halves): $h = (I_a - I_b)/2$ for halves (its variance equals the full fit's if the
  noise variance scales as one over the number of exposures; biases common to both halves cancel), $h = (I_a - I_b)/\sqrt2$ for two seeds
  (the optimisation scatter of one fit). Implied sigma: the global rms of $h$ over the common covered region and a local map (rms in a
  Gaussian window). Tested on synthetic states with known noise (to 8 per cent).

## 4. Calibration on the known-truth problem

Set-up: `tests/truth/` problem (one filter, 44 x 44 pixels of 0.10 arcsec, 36 sources, 25 empty positions), **8 exposures, 6 noise
realisations (seeds 0 to 5), both sky forms**, configurations of the truth report (grid 400 steps, items 300 steps; response frozen at the
truth, fitted once and shared by every run: relative L1 error 0.0051; noise model frozen; default zero-level rule `--sky-zero median`).
Per realisation and form: the full fit, two fits on alternate halves (4 exposures each), and for realisations 0 and 1 a second fit with
another seed; for each full fit 3 posterior draws (60 iterations) and the Gauss-Newton step; for each half and seed fit the Gauss-Newton step.
The sigma maps use the 18 draws of all 6 realisations of a form pooled (the operators differ only through the fitted registration and the
Huber weights; per-realisation sigma / pooled sigma = 0.98 to 1.02), the variance map averaged locally over a Gaussian window of 0.25 arcsec,
read at the node nearest to each point. The grid and item runs used the same random numbers for their draws, so their sigma maps are not
independent estimates (they agree to 4 digits).

Pull $= (\text{recovered} - \text{truth})/\sigma_{s}$ at s = 0.111 arcsec (the supported resolution); truth = the true sky convolved
analytically with $g_s$. "Total" adds the offset of the mean fitted background from the true one (acting as a constant sky; this uses the
truth of the background, so it is the gauge-independent number of the truth report); "raw" is the sky as returned. Mean and rms over points
and realisations; errors from the spread of the 6 per-realisation values. Empty: the 25 positions; inner: every node with |x|, |y| < 1.5
arcsec (3,481 strongly correlated nodes); sources: at the source centres, by total signal-to-noise (12, 10, 6, 4, 4 sources).
A calibrated sigma gives mean 0, rms 1.

**Product = fit plus one Gauss-Newton step (the default product), s = 0.111 arcsec**

| region | grid mean | grid rms | items mean | items rms |
|---|---|---|---|---|
| empty, total | +0.32 +- 0.20 | 1.27 +- 0.07 | +0.18 +- 0.19 | 1.14 +- 0.05 |
| empty, raw | -1.18 +- 0.18 | 1.67 +- 0.12 | -1.33 +- 0.18 | 1.72 +- 0.13 |
| inner, total | -0.41 +- 0.06 | 1.10 +- 0.02 | -0.20 +- 0.05 | 1.02 +- 0.02 |
| inner, best quadratic over the region removed | 0 | 0.96 +- 0.02 | 0 | 0.96 +- 0.02 |
| sources S/N 0.3-1 | -0.38 +- 0.14 | 1.19 +- 0.09 | -0.23 +- 0.14 | 1.12 +- 0.09 |
| sources S/N 1-3 | +0.24 +- 0.18 | 1.07 +- 0.08 | +0.24 +- 0.17 | 1.03 +- 0.09 |
| sources S/N 3-10 | -0.34 +- 0.09 | 1.11 +- 0.09 | -0.28 +- 0.10 | 0.99 +- 0.09 |
| sources S/N 10-30 | -0.09 +- 0.21 | 0.93 +- 0.14 | +0.04 +- 0.22 | 0.88 +- 0.12 |
| sources S/N 30-100 | -0.56 +- 0.20 | 1.03 +- 0.13 | -0.39 +- 0.21 | 0.98 +- 0.14 |
| half-difference h / analytic sigma (empty + sources) | +0.03 +- 0.09 | 1.03 +- 0.04 | +0.03 +- 0.09 | 1.04 +- 0.04 |
| half-difference sigma / analytic sigma (inner, global) | 1.00 +- 0.03 | | 1.00 +- 0.03 | |
| empty, total, pull with the half-difference sigma | +0.29 +- 0.17 | 1.18 +- 0.05 | +0.16 +- 0.17 | 1.06 +- 0.04 |

**The fit's own sky (as it comes out of the optimiser), same sigma, s = 0.111 arcsec**

| region | grid mean | grid rms | items mean | items rms |
|---|---|---|---|---|
| empty, total | +1.50 +- 0.07 | 1.59 +- 0.08 | +1.60 +- 0.05 | 1.62 +- 0.05 |
| empty, raw | -0.00 +- 0.04 | 0.49 +- 0.05 | +0.10 +- 0.03 | 0.24 +- 0.04 |
| inner, total | +0.79 +- 0.05 | 1.08 +- 0.02 | +1.05 +- 0.05 | 1.27 +- 0.04 |
| inner, quadratic removed | 0 | 0.69 +- 0.03 | 0 | 0.67 +- 0.02 |
| sources S/N 0.3-1 | +0.67 +- 0.07 | 0.82 +- 0.09 | +0.90 +- 0.05 | 0.96 +- 0.07 |
| sources S/N 1-3 | +0.33 +- 0.11 | 0.92 +- 0.07 | +0.32 +- 0.09 | 0.84 +- 0.08 |
| sources S/N 3-10 | -0.57 +- 0.12 | 1.26 +- 0.12 | -0.55 +- 0.13 | 1.14 +- 0.12 |
| sources S/N 10-30 | -0.53 +- 0.25 | 1.25 +- 0.20 | -0.26 +- 0.21 | 0.97 +- 0.10 |
| sources S/N 30-100 | -1.17 +- 0.20 | 1.65 +- 0.21 | -0.34 +- 0.25 | 1.07 +- 0.10 |
| half-difference h / analytic sigma | +0.02 +- 0.05 | 0.71 +- 0.02 | +0.04 +- 0.04 | 0.65 +- 0.01 |
| half-difference sigma / analytic sigma | 0.54 +- 0.02 | | 0.50 +- 0.01 | |
| empty, total, pull with the half-difference sigma | +2.62 +- 0.17 | 2.78 +- 0.18 | +3.05 +- 0.14 | 3.10 +- 0.14 |
| optimisation scatter / analytic sigma (2 seed pairs) | 0.17, 0.19 | | 0.14, 0.17 | |

Other scales (inner rms, total / with the quadratic removed; half-difference sigma over analytic): product grid 1.06/0.96, 1.00 at 0.10
arcsec; 1.27/0.92, 1.00 at 0.15; 1.51/0.86, 0.99 at 0.20; items 1.00/0.96, 1.00; 1.07/0.92, 1.00; 1.15/0.85, 0.99. Fit's own sky: half-difference
over analytic 0.49, 0.62, 0.65 (grid) and 0.46, 0.56, 0.59 (items) at 0.10, 0.15, 0.20. Full tables: `summary_*_n8.json`.

### Reading

* **The product with the Gauss-Newton step is calibrated** at the supported resolution once the large-scale modes are set aside: the two
  independent estimators agree (half-difference over analytic 1.00 +- 0.03 for both forms, at every scale tested); pulls on the inner region
  have rms 0.96 with a quadratic removed (the removal itself takes out a little noise, so slightly below 1 is expected), sources 0.88 to 1.19,
  empty regions 1.14 to 1.27. The empty-region mean (+0.32 +- 0.20 grid, +0.18 +- 0.19 items) is consistent with zero at 1.6 and 1 sigma of
  its error. Nothing was rescaled.
* **What is missing in the product's sigma, by size:**
  1. *The six large-scale sky modes degenerate with the backgrounds* (held at the fitted values by the gauge): they raise the inner rms from
     0.96 to 1.10 (grid) and 1.02 (items), i.e. **0.54 sigma (grid) and 0.34 sigma (items) in quadrature** at 0.111 arcsec, 1.24 and 0.77 at 0.20
     arcsec. The inner mean (-0.41, -0.20) is the same effect, and it explains the faintest-source means (-0.38, -0.23), which sit on it.
  2. *The zero level* (pedestal rule, one number per filter): the default `--sky-zero median` puts the backgrounds 0.19 high in surface
     brightness, **1.42 sigma per element at 0.111 arcsec** (it grows as the square root of the number of exposures). The product's raw sky
     carries it (raw empty mean -1.18, -1.33); only the total, which needs the true background, is free of it. The rule `--sky-zero masked`
     cut it to 0.03 to 0.05 (truth report section 6); not recalibrated here.
  3. *Bright sources* (S/N 30 to 100): -0.56 +- 0.20 (grid), -0.39 +- 0.21 (items). Candidates: the frozen response's 0.5 per cent error (an
     error that is a fixed fraction of the flux gives about that fraction times the signal-to-noise in sigma units: 0.3 to 0.5 at S/N 60 to 100)
     and the representation of the 0.03 arcsec sources on the 0.05 arcsec tent grid. Not separated.
* **Why the fit's own sky is not calibrated** (and why the product is the polished sky):
  1. *Zero level and positivity*: the backgrounds sit 0.19 high (1.42 sigma), and positivity holds the raw sky at zero instead of letting it go to
     -1.42 sigma, so the total is +1.50 (grid) and +1.60 (items): +1.42 of it is the zero-level rule, the rest is within errors. After the step
     (no positivity) the raw sky goes down and the total comes back to +0.32 / +0.18.
  2. *Less noise than the posterior*: the fit's sky has 0.54 (grid) and 0.50 (items) of the posterior's noise at 0.111 arcsec (0.29 and 0.25
     of its variance), 0.65 / 0.59 at 0.20. In empty regions the raw scatter is 0.49 (grid, soft positivity penalty) and 0.24 (items, positive by
     construction): positivity clips the noise. Elsewhere the optimiser has not reached the least-squares sky (the step changes it by 0.7 sigma rms
     beyond the zero level). I did not separate positivity from incomplete convergence.
  3. *Flux at the faint and bright ends*: -0.57 at S/N 3 to 10 and -1.17 (grid) at S/N 30 to 100 against -0.34 and -0.56 after the step: the
     shrinkage of item 2 acting on sources (the grid's bright-source deficit is the resolution loss already seen in the truth report).
  So the analytic sigma over-states the noise of the fit's own sky by a factor of about 2 while missing a bias of about 1.5 sigma in empty
  regions; the half-difference of the fit's own skies under-states the error by the same factor (empty pulls with it: rms 2.8 to 3.1). Neither
  number should be quoted for SKY_FIT.

## 5. What the uncertainty does not include

| item | size on the truth problem (s = 0.111 arcsec, 8 exposures) |
|---|---|
| the response being learned (held at its fitted value) | not measurable here: the response is frozen at the truth (0.5 per cent error) because learned-response fits do not converge on this problem (truth report C5, C6). An error that is a fraction f of the flux contributes about f x (signal-to-noise) sigma; the bright-source means (-0.4 to -0.6 at S/N 30 to 100) bound the effect of f = 0.005 together with the representation |
| calibration errors common to all exposures (throughput, flux scale, bandpass) | not measurable (throughput is exactly 1 here); same scaling as the response: f x (signal-to-noise) |
| the zero level (pedestal rule) | 0.19 surface brightness = 1.42 sigma per element with `--sky-zero median`; one number per filter, grows as sqrt(exposures) in sigma units |
| the six background-degenerate large-scale sky modes | 0.54 sigma (grid), 0.34 sigma (items) in quadrature; 1.24 / 0.77 at 0.20 arcsec |
| positivity bias | none in the product (positivity not imposed after the step; the product can be negative). For SKY_FIT: +1.5 sigma total in empty regions with the median rule, and the noise halved |
| optimisation scatter | 0.14 to 0.19 sigma for the fit's own sky (2 to 4 per cent in variance); 0.11 to 0.21 sigma after the step (conjugate-gradient truncation plus the different fitted registrations of two seeds) |
| registration, gain, per-pixel gain, noise terms held | not measured (registration truth is 0 and fitted values are near 0 here) |
| conjugate-gradient truncation (60 iterations) | variance 2 per cent low at 0.10 arcsec (one draw); exact solve at the centre agrees with the draws to 0.5 per cent |
| Huber weights at the fit | 40 of 14,746 pixels down-weighted: negligible here |

## 6. Cost and real data

Unit: one application of $F$ = one forward and one backward render of every training pixel (1.8 forward-render equivalents). Measured on the
truth problem, one CPU thread: 1.45 s for 14,746 pixels (98 microseconds per pixel; tiles of `--region-px` pixels were 4 times faster than
the 22-pixel tiles of the prediction code); a 400-step grid fit took 89 s = 61 applications.

| piece | applications | truth problem, one thread |
|---|---|---|
| set-up (weights pass, coverage, K delta responses for the preconditioner) | K + 2 (14 at K = 12; 5 here) | 6 s |
| Gauss-Newton step (the product) | 61 | 90 s |
| one posterior draw | 61 | 90 s |
| exact sigma at one pixel and one scale | about 41 | 60 s |
| map with the Fourier approximation and the step | K + 63 (75 at K = 12) | 1.2 fits |
| map with 8 draws and the step | K + 63 + 61 x 8 (563 at K = 12) | 9 fits |

Scaling to a real field (86 exposures, 600 x 600 nodes, 12 spectral coefficients): an application is linear in the number of training
pixels (about 86 x 260,000 = 22 million if half the exposures are short-wavelength NIRCam at 0.031 arcsec and half long-wavelength at 0.063
arcsec in a 20 arcsec box; my assumption); the preconditioner is a $600^2$ FFT with $12\times12$ blocks per frequency (415 MB in complex
single precision, negligible time); vectors are 4.3 million coefficients (17 MB each, about 6 held). The iteration count is set by the
conditioning after preconditioning, not by the size, so I expect about 60 again, but only one field size was tested, and chip gaps and six
filters (twelve spectral coefficients, several constrained by only one filter) may need more. In units of the real fit: the map costs about
75 + 61 n full passes, where one full pass costs (training pixels) / (pixels per optimisation step) steps of the fit divided by about 2.5
(on the truth problem a step costs 2.5 times a pass over the same pixels); with one thread on a CPU an application would take about 36 minutes,
so the real map needs the GPU.

Command on a saved real fit (run where the fit's data paths are valid; the fit's configuration is read from its `metrics.json`, the state
from `fieldop_state.pt` next to it; options written here win):

```
PYTHONNOUSERSITE=1 python -m fieldop.uncertainty --unc-config OUT/metrics.json --device cuda --unc-band F200W \
    --unc-out OUT/sky_F200W_uncertainty.fits [--unc-method circulant] [--unc-samples 8] [--unc-resolution auto]
```

Output: one FITS file per band (FITS because it is the format every astronomy viewer reads, astropy is already used by the loader, and one
file holds the images with their conventions in the headers; `.npz` if astropy is missing): SKY (product at the resolution PRODRES: by default
the median supported resolution; fit plus one Gauss-Newton step), SIGMA, SKY_FIT, SIGMA_CIRC, RESOLUTION (the map $s^\ast(x)$), COVERED,
SIGMA_SCALES (sigma at the scales listed in the header), header keys PRODRES, RESMED, CORRLEN, PRIORRHO, NSAMPLE, POLISHED, SIGRATIO
(median SIGMA / SIGMA_CIRC: a check of the cheap approximation on the real field). Half-difference: fit with `--train-half 0` and `--train-half 1`,
run the command above on each, then `python -m fieldop.uncertainty --unc-cmd diff-files --unc-file-a A.fits --unc-file-b B.fits --unc-out D.fits`.
The command line was tested for option handling only (`tests/test_uncertainty.py`); it has not been run on real files (no data here).

## 7. Approximations and shortcuts (all of them)

* Linearisation with the response, registration, gain and noise terms held; for the grid sky the model is exactly linear in the sky and the
  backgrounds, so the posterior is exact given those; for the item sky it is the posterior of a free grid at the raster (section 2d).
* Gaussian likelihood with the Huber weights of the fit's loss at the fit's residual (iteratively reweighted least squares weights), not a
  sandwich estimate.
* Background gauge: the training mean of each of the 6 background terms held (the fit holds only the constant).
* Weak prior rho = 10 node-noise units; it changes sigma by 0.2 per cent at 0.10 arcsec.
* Draws: 3 per realisation, pooled over the 6 realisations of a form (18), grid and items with the same random numbers; variance map averaged
  over a 0.25 arcsec Gaussian window and read at the nearest node; 60 iterations (about 2 per cent variance deficit).
* Truth problem: response fitted to the truth once and shared by all runs (the truth report fitted it per run); fits stopped by step count
  (grid 400, items 300) as in the truth report; one population, one geometry; 8 exposures only.
* Half-difference: assumes the noise variance scales as one over the number of exposures and that the two halves see the sky alike (their
  dithers differ; the Gauss-Newton step of each half uses its own operator).
* The supported resolution uses the alpha = 2 criterion and a ladder of 10 scales with log-linear interpolation; the correlation length is a
  global number (azimuthal average over the inner region), not a map.
* The "total" pulls use the true background (not available on real data); the representation of the true sky on the node grid was not
  modelled (an attempt using the truth sampled at the nodes gave -1.3 sigma at S/N 30 to 100, larger than the measured -0.56, so it is not a
  usable proxy and is not used).

## 8. Not done

* 32 exposures (no time: the 8-exposure set took 40 fits and 77 solves of 1 to 3 minutes each on two cores).
* The calibration with `--sky-zero masked` (expected to remove most of the zero-level item).
* Separating positivity from incomplete convergence in the fit's own sky; separating the response error from the representation in the
  bright-source means.
* Linearising the item sky in its own parameters.
* A run on real files (option `--train-half` and the command line are untested on real data); a correlation-length map.
