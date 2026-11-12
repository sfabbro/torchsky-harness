---
title: "One sky, many instruments: general formulation and approximation ladder"
date: "13 November 2026"
---

**Purpose.** This note starts from the most general measurement equation. No form is assumed for the noise, the background or the instrument response.

It then lists, one at a time, the approximations that can be made. For each it states what the approximation breaks or creates in the way of degeneracies, and what the current implementation (the `fieldop` package) does. The aim is to choose the functional forms deliberately, for an outside deep-thinking review.

# 0. Notation

| Symbol | Meaning |
|---|---|
| $e$ | exposure, i.e. one detector image; $d(e)$ its detector, $f(e)$ its filter, $[t_e, t_e+\Delta t_e]$ its time span |
| $\mathbf{x}$ | pixel position on the detector (pixels); $\tilde{\mathbf{x}}\in[-1,1]^2$ normalised over the detector |
| $\mathbf{p} = F_{d}(\mathbf{x})$ | position in the telescope focal plane (e.g. JWST $V2,V3$) |
| $\boldsymbol\theta$ | position on the sky (tangent plane of a sky patch, arcsec) |
| $\boldsymbol\theta_e(\mathbf{x},\lambda)$ | sky position imaged at pixel $\mathbf{x}$ (the geometric mapping) |
| $\mathbf{u} = \boldsymbol\theta - \boldsymbol\theta_e(\mathbf{x},\lambda)$ | sky offset from that position |
| $\lambda,\ t$ | wavelength, time |
| $I(\boldsymbol\theta,\lambda,t)$ | astronomical sky (specific intensity) |
| $I_{\mathrm{fg}}(\boldsymbol\theta,\lambda,t)$ | foreground radiance: zodiacal light, Galactic cirrus, stray light from sources outside the field |
| $E_e(\mathbf{x},\lambda,t)$ | instrument emission reaching the pixel without passing through the sky: thermal self-emission, glow |
| $D_e(\mathbf{x},t)$ | dark current |
| $d_e(\mathbf{x})$ | recorded value |

# 1. General measurement equation

## 1.1 Expected signal (no approximation)

The charge collected in pixel $\mathbf{x}$ during exposure $e$ is a functional of everything that reaches it. The most general form linear in incoming radiance, followed by a general detector map, is

$$
Q_e(\mathbf{x}) = \int_{t_e}^{t_e+\Delta t_e}\!\!dt\int d\lambda\left[\int d^2\theta\;\mathcal{R}_e(\mathbf{x},\boldsymbol\theta,\lambda,t)\,\big(I + I_{\mathrm{fg}}\big)(\boldsymbol\theta,\lambda,t) \;+\; E_e(\mathbf{x},\lambda,t)\right] + \int dt\; D_e(\mathbf{x},t),
$$

$$
\mu_e(\mathbf{x}) \;=\; \mathcal{D}_e\!\big[\,Q_{e}(\cdot),\ \mathcal{H}_e\,\big](\mathbf{x}).
$$

- **Response.** $\mathcal{R}_e(\mathbf{x},\boldsymbol\theta,\lambda,t)$ is the full response of pixel $\mathbf{x}$ to radiance from direction $\boldsymbol\theta$ at wavelength $\lambda$ and time $t$. It covers geometry, optics, filter transmission, quantum efficiency, intra-pixel response and charge diffusion. It is not assumed local, separable or shift-invariant.
- **Detector map.** $\mathcal{D}_e$ is non-linear and non-local, acting on the whole charge image.
  - It includes non-linearity, inter-pixel capacitance, brighter-fatter and saturation.
  - It includes ramp fitting and the calibration steps that turn ramps into `_rate` or `_cal` products.
  - $\mathcal{H}_e$ is the detector history: persistence from earlier exposures.
- **Background.** Background is not a separate additive term here. Foreground radiance passes through the same response as the sky; instrument emission and dark current enter separately.

## 1.2 Likelihood (no approximation)

The data are a random draw given the expected signal and the detector state:

$$
d_e \;\sim\; p\big(d_e \,\big|\, \mu_e,\ \mathcal{H}_e\big).
$$

In general this includes:

- Poisson statistics of the collected charge;
- read noise;
- correlated noise (1/f, appearing as row and column structure; inter-pixel capacitance correlating neighbours);
- cosmic-ray hits and other outliers;
- the non-Gaussian, data-dependent errors introduced by ramp fitting.

The noise is neither additive nor Gaussian in general.

## 1.3 What is estimated

The sky $I$ is estimated jointly with every nuisance: $\mathcal{R}_e$, $I_{\mathrm{fg}}$, $E_e$, $D_e$, $\mathcal{D}_e$ and the noise law. Everything that is not $I$ is the **instrument operator** of exposure $e$.

# 2. Gauges (exact degeneracies of the general equation)

These transformations leave every expected signal $\mu_e$ unchanged, so no data can fix them. They must be declared and pinned by one condition each.

| Gauge | Transformation | Comment |
|---|---|---|
| Flux scale per $\lambda$ | $I\to c(\lambda)\,I,\;\; I_{\mathrm{fg}}\to c(\lambda)\,I_{\mathrm{fg}},\;\; \mathcal{R}_e\to\mathcal{R}_e/c(\lambda)$ | pinned by one absolute calibration |
| Common blur | $I\to I\ast b,\;\; \mathcal{R}_e(\mathbf{x},\cdot)\to\mathcal{R}_e(\mathbf{x},\cdot)\ast b^{-1}$ | the sky is defined only up to a common kernel |
| Common shift | $I(\boldsymbol\theta)\to I(\boldsymbol\theta+\mathbf{a}),\;\; \mathcal{R}_e(\mathbf{x},\boldsymbol\theta)\to\mathcal{R}_e(\mathbf{x},\boldsymbol\theta+\mathbf{a})$ | pinned by one astrometric reference |
| Sky against foreground | $I\to I+J,\;\; I_{\mathrm{fg}}\to I_{\mathrm{fg}}-J$ | exact if both are static and pass through the same response |

The last gauge is broken only by structure that differs between the two components: foreground that varies with time or pointing (zodiacal light changes with season and elongation), or stray light that depends on what lies outside the field.

# 3. Approximation ladder

Each step is one approximation. For each: the equation, why it might help identifiability, its cost, and what `fieldop` does today.

## A1. Static astronomical sky

$$
I(\boldsymbol\theta,\lambda,t)=I(\boldsymbol\theta,\lambda).
$$

- **Breaks:** a time-varying response against a static sky. Any change between epochs must be instrumental. This also breaks the sky–foreground gauge when $I_{\mathrm{fg}}$ varies.
- **Cost:** variable sources, transients and proper motions are absorbed elsewhere. They could be added back as a sparse time-varying component.
- **fieldop:** assumed.

## A2. No change within an exposure

The response, foreground and emission are constant over $[t_e,t_e+\Delta t_e]$:

$$
\int dt\,\mathcal{R}_e(\mathbf{x},\boldsymbol\theta,\lambda,t)\to\Delta t_e\,\mathcal{R}_e(\mathbf{x},\boldsymbol\theta,\lambda).
$$

- **Cost:** jitter and drift within an exposure are absorbed into the effective kernel; time-variable foreground is averaged.
- **fieldop:** assumed.

## A3. Localised response: geometry ∘ kernel ∘ amplitude

$$
\mathcal{R}_e(\mathbf{x},\boldsymbol\theta,\lambda) \;=\; A_e(\mathbf{x},\lambda)\; k_e\!\big(\boldsymbol\theta-\boldsymbol\theta_e(\mathbf{x},\lambda);\;\mathbf{x},\lambda\big),
\qquad \int k_e(\mathbf{u};\mathbf{x},\lambda)\,d^2u = 1.
$$

This is a reparametrisation of any response with a finite first moment, not a restriction. It splits $\mathcal{R}_e$ into three parts:

- **Geometry** $\boldsymbol\theta_e(\mathbf{x},\lambda)$: where the pixel looks.
- **Shape** $k_e$: how the light is spread around that point.
- **Amplitude** $A_e$: how much light is collected.

It introduces a new **internal gauge:** the first moment of $k_e$ trades against $\boldsymbol\theta_e$. It is pinned by fixing the centroid of the core of $k_e$ to zero.

**Geometry.**

$$
\boldsymbol\theta_e(\mathbf{x},\lambda) \;=\; W_e\Big(\mathbf{x} + \Delta_{\mathrm{fp}}\big(F_d(\mathbf{x}),\lambda,t_e\big) + \Delta_d(\mathbf{x})\Big).
$$

- $W_e$ is the pointing: the header WCS, refined by a registration shift and a per-exposure affine.
- $\Delta_{\mathrm{fp}}$ is optical distortion, continuous over the focal plane and possibly chromatic and slowly varying.
- $\Delta_d$ is a per-detector term: mounting and pixel-grid irregularities.

**Shape.** The shape splits naturally into a part continuous over the focal plane (optics) and a per-detector part (pixel response, charge diffusion):

$$
k_e(\mathbf{u};\mathbf{x},\lambda) \;\simeq\; \big[\,k^{\mathrm{opt}}(\,\cdot\,;\mathbf{p},\lambda,t_e)\ast k^{\mathrm{det}}_{d}(\,\cdot\,;\mathbf{x},\lambda)\,\big](\mathbf{u}),
\qquad \mathbf{p}=F_d(\mathbf{x}).
$$

- **Breaks:** continuity of the optics across detector boundaries ties neighbouring detectors together. The detector part carries discontinuities.
- **fieldop:** uses one shape field conditioned on (detector embedding, $\tilde{\mathbf{x}}$, $\lambda$, epoch, exposure code), with no focal-plane coordinate. Optical continuity across detectors is therefore not used.

**Amplitude.** It splits into the projected pixel area and the throughput:

$$
A_e(\mathbf{x},\lambda) \;=\; \Big|\det\frac{\partial\boldsymbol\theta_e}{\partial\mathbf{x}}\Big|\;\;\underbrace{T_{f}(\lambda;\mathbf{p})\;\eta_d(\mathbf{x},\lambda)\;g_e}_{\text{filter}\,\times\,\text{QE}\,\times\,\text{gain}}.
$$

- **Breaks:** the pixel area is tied to the geometry, so photometry and astrometry constrain each other.
- **Breaks:** filter transmission (smooth over the focal plane) is separable from quantum efficiency and flat structure (per detector).
- **fieldop:** uses an area term (1 for flat-fielded input), a published filter curve as input, and one learned log-throughput field over (detector, $\tilde{\mathbf{x}}$, $\lambda$, epoch) plus a per-exposure gain.

## A4. Foreground through the same response

Insert A3 into §1.1 and use the fact that the foreground is smooth on the scale of the kernel, $k_e\ast I_{\mathrm{fg}}\simeq I_{\mathrm{fg}}$:

$$
Q_e(\mathbf{x}) \simeq \Delta t_e\!\int\! d\lambda\,A_e(\mathbf{x},\lambda)\Big[\,(k_e\star I)\big(\boldsymbol\theta_e(\mathbf{x},\lambda),\lambda\big) + I_{\mathrm{fg}}\big(\boldsymbol\theta_e(\mathbf{x},\lambda),\lambda,t_e\big)\Big] + \Delta t_e\!\int\! d\lambda\,E_e(\mathbf{x},\lambda) + \Delta t_e\,\bar D_e(\mathbf{x}).
$$

- **Breaks:** the foreground is imprinted through the *same* amplitude $A_e$ as the sky. It acts as a "sky flat", which ties the flat field to the background and separates throughput from additive background.
  - Additive terms not multiplied by $A_e$ (instrument emission, dark current, stray light from outside the field) remain separate.
- **Cost:** a model for $I_{\mathrm{fg}}$ is needed: smooth in $\boldsymbol\theta$ and $\lambda$, varying with epoch and pointing.
- **fieldop:** does **not** do this. The background is an additive surface $B_e(\tilde{\mathbf{x}})$ with no wavelength dependence and no multiplication by throughput (quadratic per exposure, plus an 8×8 pattern per detector, plus row/column stripes). This is the "integrated over wavelength" approximation.

## A5. Linear detector, separated

$$
\mathcal{D}_e[Q](\mathbf{x}) \;\simeq\; \Big(\mathcal{C}_d \ast \big(Q + \mathcal{B}[Q]\big)\Big)(\mathbf{x}) \;+\; \mathcal{P}_e(\mathbf{x}) \;+\; \text{(non-linearity, saturation mask)}.
$$

- $\mathcal{C}_d$ is inter-pixel capacitance, a small fixed kernel per detector.
- $\mathcal{B}$ is brighter-fatter, a charge-dependent redistribution.
- $\mathcal{P}_e$ is persistence from the detector history $\mathcal{H}_e$.
- **Breaks:** inter-pixel capacitance and brighter-fatter act *after* the optics and depend on charge. They are distinguishable from the kernel by their dependence on flux.
- **Cost:** the extra terms are only partly constrained by data.
- **fieldop:** none of these are modelled. Inter-pixel capacitance and charge diffusion are absorbed into the shape; there is an optional gated non-linearity, untested; saturated pixels are masked.

## A6. Wavelength quadrature

$$
\int d\lambda\,(\cdots)\;\simeq\;\sum_{q=1}^{Q} w_q\,(\cdots)\big|_{\lambda_q}.
$$

- **fieldop:** $Q=5$ nodes at the quantiles of $T_f(\lambda)/\lambda$, with error $1.2\times10^{-3}$. Chromatic kernels and geometry are kept at each node.

## A7. Noise model

There are several levels of approximation to $p(d_e\mid\mu_e)$:

| Level | Likelihood | Comment |
|---|---|---|
| L0 | exact ramp likelihood (Poisson + read noise + jumps) | needs the ramps (`_uncal`) |
| L1 | $d \sim \mathcal{N}\big(\mu,\ \sigma_r^2 + \mu/g + (f\mu)^2\big) + \text{correlated}$ | variance from the **model**, not the data; avoids flux bias |
| L2 | $d \sim \mathcal{N}\big(\mu,\ (\nu\,\sigma^{\mathrm{pipe}})^2 + (f\mu)^2\big)$ | pipeline error from the data, times a factor |
| L3 | L2 with a robust loss (Huber) for outliers | **fieldop today** |

Correlated noise can be modelled as covariance, or as deterministic nuisances (row/column offsets). fieldop uses deterministic stripes.

- **Breaks:** at L1, the variance depends on the model's flux. Bright sources then constrain the gain $g$ and the throughput together, and model-based variance removes the bias of data-based errors (faint pixels that fluctuate low get small errors and too much weight).
- **fieldop measurements:** the pipeline $\sigma$ is about 29% too small, and the fractional floor is about 5%.

## A8. Sky representation

$I(\boldsymbol\theta,\lambda)$ must be represented by a finite model.

- **fieldop:** 12 spectral bumps in $\log\lambda$ times a coefficient grid at 0.04″ with bilinear interpolation, one grid per sky patch:

$$
I(\boldsymbol\theta,\lambda) = \sum_{j=1}^{12}\phi_j(\log\lambda)\sum_{m,n}a_{j,mn}\,\Lambda\!\Big(\tfrac{\theta_1-\theta_{1,m}}{p_s}\Big)\Lambda\!\Big(\tfrac{\theta_2-\theta_{2,n}}{p_s}\Big).
$$

- **Known limitation:** this cannot represent short-wavelength point sources. The worst residual relative to the peak is 0.26 in F090W and 0.16 in F150W, against $5\times10^{-6}$ in F444W.
- Because of it, fieldop pins the blur gauge with a **floor:** no kernel may be sharper than one sky tent, enforced by dilating all kernels. A continuous sky representation needs a different declaration of this gauge.

# 4. Current implementation in one equation

With A1–A3, A6, L3 noise, the additive background and the grid sky:

$$
d_e(\mathbf{x}) \;\approx\; \sum_{q=1}^{Q} w_q\,\alpha_e(\mathbf{x})\,g_e(\mathbf{x},\lambda_q)\!\int\! d^2u\;k_e(\mathbf{u};\mathbf{x},\lambda_q)\,I\big(\boldsymbol\theta_e(\mathbf{x},\lambda_q)+\mathbf{u},\lambda_q\big) \;+\; B_e(\mathbf{x}) \;+\; n_e(\mathbf{x}),
$$

$$
n_e \sim \text{Huber}\big(0,\ \sigma_e\big),\qquad
\sigma_e^2 = \big(\nu_{d,f}\,\sigma^{\mathrm{pipe}}_e\big)^2 + \big(f\,S_e\big)^2.
$$

## 4.1 Pointing and distortion

$$
\boldsymbol\theta_e(\mathbf{x},\lambda) = (I+M_e)\big[\boldsymbol\theta^{\mathrm{wcs}}(\mathbf{x}) + \boldsymbol\delta_e + \Delta(\mathbf{x};\lambda,t_e) - \mathbf{c}_e\big] + \mathbf{c}_e + \mathbf{t}_e.
$$

## 4.2 Kernel shape

$$
k_e(\mathbf{u};\mathbf{x},\lambda) = \frac{\exp \ell_\vartheta\big(R_e^{\top}\mathbf{u};\,h_e(\mathbf{x},\lambda)\big)}{\sum_{i}w_i\exp\ell_\vartheta\big(R_e^{\top}\mathbf{u}_i;\,h_e\big)}.
$$

The conditioning code is a sum of parts, each scaled by its own variability scale:

$$
\begin{aligned}
h_e(\mathbf{x},\lambda) = \bar h\big(\lambda_{\mathrm{piv}},\mathbf{T}_f,\mathrm{inst}\big)
&+ \tau_\lambda\big[\bar h(\lambda)-\bar h(\lambda_{\mathrm{piv}})\big]
+ \tau_{\mathrm{det}}\,\hat h(d,\tilde{\mathbf{x}})\\
&+ \tau_{\mathrm{ep}}\,\hat h(t_e)
+ \tau_{\mathrm{exp}}\,\mathbf{z}_e,
\qquad \mathbf{z}_e\sim\mathcal{N}(0,I).
\end{aligned}
$$

- $\ell_\vartheta$ is an MLP with FiLM conditioning.
- Its input is a log-radius and angular-harmonic encoding of the offset, out to 10″.
- $R_e$ rotates into the detector frame.

## 4.3 Throughput

$$
\begin{aligned}
g_e(\mathbf{x},\lambda) = \frac{T_f(\lambda)}{\lambda N_f}\exp\Big[\,&\bar\gamma(\lambda,\mathbf{T}_f,\mathrm{inst})
+\tau_{gd}\,\hat\gamma(d,\tilde{\mathbf{x}},\lambda)\\
&+\tau_{ge}\,\hat\gamma(t_e,\lambda)
+\tau_{gx}\,u_e - m(\lambda)\Big].
\end{aligned}
$$

## 4.4 Background

$$
B_e(\mathbf{x}) = \sum_{i+j\le 2} b_{e,ij}\,P_i(\tilde x_1)P_j(\tilde x_2) + \Pi_d(\tilde{\mathbf{x}}) + r_e(\text{row}) + c_e(\text{col}).
$$

## 4.5 Rendering, variability scales and gauges

- **Rendering:** overlap-add FFT, checked against a per-pixel quadrature to $6\times10^{-4}$–$2\times10^{-3}$.
- **Variability scales** $\tau$:
  - random effects by a type-II (EM) update;
  - function axes by held-out validation evidence with an off-switch check.
- **Gauges:**
  - **flux:** exact, via $m(\lambda)$;
  - **blur:** a floor;
  - **shift:** zero-mean translations;
  - **pedestal:** training background constants sum to zero about each exposure's median.

# 5. Evaluation and current results

$$
\chi^2/N = \frac{1}{|\mathcal{M}|}\sum_{(e,\mathbf{x})\in\mathcal{M}}\left(\frac{d_e(\mathbf{x})-\hat d_e(\mathbf{x})-b_e}{\nu_d\,\sigma^{\mathrm{pipe}}_e(\mathbf{x})}\right)^2,
\qquad \mathcal{M}=\{\text{held-out pixels with S/N}_{\mathrm{ref}}\ge5\}.
$$

- **Held-out pixels:** these are held-out dither exposures. Only per-exposure nuisances may be refitted on them.
- **Masks:** the masks and $\mathrm{S/N}_{\mathrm{ref}}$ come from the official stage-3 mosaic.

| Field | fieldop | Official JWST pipeline (mosaic blotted back) | Empty sky |
|---|---|---|---|
| SMACS NIRCam, 20″ box | 1.77 [1.68, 1.86] (older package); 3.29 (newer, regressed) | 1.80 [1.61, 2.02] | 126 |
| LMC NIRCam, 10 patches | 8.54 [7.32, 9.90] | 6.14 [5.25, 7.22] | 30.3 |

The field operator is better than the pipeline at S/N < 20 and worse at S/N > 50, the bright star cores.

# 6. Defects found in review

1. **Confirmed: blur-floor gradient.** The dilation's gradient is wrong: one pass runs without autograd. Kernel gradients are biased by 8–26%, and the sign is wrong at large dilation.
2. **Design defect: the throughput field absorbs gains.** It also absorbs per-detector gain constants, so it is always kept by the variability rule. This is the likely cause of the SMACS regression.
3. **Design defect: the pedestal gauge.** It is pinned at exposure medians, which conflicts with sky positivity in crowded fields.
4. **Minor:**
   - the `--tau-prior` flag is ignored;
   - the logged batch χ² is not importance-weighted;
   - there is slight validation leakage in registration;
   - the WCS source for `_rate` input is wrong;
   - inf can turn into NaN.

Geometry, orientation, flux, held-out isolation and the renderer were checked numerically and are correct.

# 7. Questions for the choice of functional forms

1. **Which approximations to keep or drop.** Which steps of the ladder should be dropped because keeping the more general form *breaks* degeneracies? Candidates:
   - foreground through the response (A4: a sky flat);
   - focal-plane continuity of the optics (A3);
   - model-based Poisson variance (A7 L1);
   - an explicit inter-pixel capacitance / brighter-fatter stage (A5).
2. **Sky** $I(\boldsymbol\theta,\lambda)$. Which continuous representation keeps point sources sharp below the finest pixel, represents extended emission compactly, optimises as well as a grid and scales to the full sky? Is a sparse continuous component acceptable without a catalogue?
3. **Blur gauge for a continuous sky.** Band-limit the sky at the sharpest kernel, fix a reference kernel, or only declare and report it?
4. **Kernel family.** One conditional field, or the composition $k^{\mathrm{opt}}(\mathbf{p})\ast k^{\mathrm{det}}_d(\mathbf{x})$, each a field? Which conditioning (FiLM, hypernetwork, additive codes) gives smooth, identifiable variation across $\mathbf{p}$, $\lambda$ and $t$?
5. **Geometry and the kernel.** Keep the distortion separate, or let it be the kernel's first moment, with the internal gauge pinned?
6. **Throughput.** Should it factor into filter $T_f(\lambda;\mathbf{p})$ × quantum efficiency/flat $\eta_d(\mathbf{x},\lambda)$ × gain $g_e$, so that per-detector gains are separated from spatial structure?
7. **Foreground.** What form for $I_{\mathrm{fg}}$ (smooth in $\boldsymbol\theta$ and $\lambda$, varying with epoch and pointing) breaks the sky–foreground gauge without a physical zodiacal-light model?
8. **Variability scales.** What principled evidence decides whether an axis is kept: a Laplace or marginal likelihood on a linearised operator, refitted likelihood ratios, or cross-validation? How can weak but real effects (the PSF's colour dependence within a filter) be kept?
9. **Noise.** Should the model work at L1 with model-based variance, with correlated noise as covariance or as nuisances? How should outliers be handled without a mixture? An earlier mixture classified bright cores as outliers and failed.
10. **Cost.** About 1.2 s per step on an 11 GB GPU slice, dominated by fine-level FFTs. Which forms make the operator cheap at survey scale?
