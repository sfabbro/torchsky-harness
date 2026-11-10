# fieldop: design note (Phase 2, general field operator), 10 Nov 2026

Package `fieldop/` (sky.py, bandpass.py, kernel.py, render.py, model.py, data.py, train.py, cli.py, synth.py), runner `run_fieldop.sh`
(on /arc as `scripts/run_fieldop.sh`), tests `tests/test_fieldop_{render,synth,gauge,area}.py`. Every approximation is listed as F1...F30
in `fieldop/__init__.py` (printed at the start of every run, written to metrics.json). Nothing here is catalogue-based: no source detection,
no source list, no PSF library. Physics enters only as inputs (filter curves, wavelengths, aperture diameter and pixel scale for the kernel
warm start, header WCS, pipeline ERR) or as a warm start.

## Equations as implemented

For exposure e, detector pixel xi (normalised to [-1, 1] over the detector), filter f with Q lambda nodes:

    d_e(xi)   = N_e[ S_e(xi) ] + B_e(xi) + n_e(xi)
    S_e(xi)   = (1/Q) sum_q  alpha_e(xi) exp(gamma_e(xi, lambda_q)) c_q(xi)
    c_q(xi)   = int d^2u  k_e(u; xi, lambda_q)  s(Omega_e(xi, lambda_q) + u, lambda_q)

* **Sky** (sky.py): `s(X, Y, lambda) = sum_k phi_k(lambda) A_k(X, Y)`, 12 RBF bumps over log lambda (v40 basis); `A_k` is a grid at `--sky-pitch`
  (0.04" or 0.02"), and the continuous sky is its bilinear (tent) interpolant, zero outside. Coordinates: tangent plane at the box centre,
  arcsec (X to the west, Y north; the register_xcorr convention).
* **lambda integral** (bandpass.py): nodes at the mid-quantiles of `T_f(lambda)/lambda` (published SVO curve), equal weights 1/Q. This is
  `int dlambda T/(lambda N_f) exp(gamma) [...]` with `N_f = int T dlambda/lambda`, so a flat f_nu sky gives the _cal surface brightness when gamma = 0.
* **Geometry** (model.py): `Omega = (I + M_e)(X_wcs + shift0_e + D) + t_e` about the box centre, with `M_e`, `t_e` the per-exposure affine.
  `shift0_e` comes from register_xcorr.json (LMC, `--shifts-from`) or is 0 (SMACS). In that case the translation is learned: first by the
  registration pre-pass, then jointly. `D = tau_Ds Dst^(det, xi, lambda) + tau_Dd Ddr^(det, xi, t) + tau_De sum_j zD_ej B_j(xi)`: a chromatic
  static field, a slow drift and a per-exposure quadratic (zD ~ N(0, I)). Dst^ and Ddr^ are centred and rms-normalised over reference sets,
  so each tau is the rms displacement in arcsec.
* **Kernel shape** (kernel.py): `k(u; h) = exp(l(R_e^T u, h)) / Z`, with `l` = FiLM-MLP of a smooth multi-scale encoding of the offset
  (log sqrt(r^2 + r0^2) with radial Fourier features, angular harmonics `((x + iy)/sqrt(r^2 + r0^2))^m`, cross terms). One field, no imposed basis.
  `R_e` = rotation of the exposure's mean Jacobian (the kernel lives in the detector frame). The code is
  `h = hbar(lambda_piv, T_f or token, instrument) + tau_lam (hbar(lambda) - hbar(lambda_piv)) + tau_det hdet^(det, xi) + tau_epoch hep^(t) + tau_exp z_e`,
  with `z_e ~ N(0, I)`, and hdet^ and hep^ centred and rms-normalised over the training references. The filter curve (48 samples on a fixed
  log-lambda grid) is an input vector (`--filter-input curve`) or a learned token. `Z` = sum of `exp(l)` over the rendering quadrature, so
  the integral is 1 on that quadrature.
* **Amplitude**: `A = alpha g`, `g = T/(lambda N) exp(gamma)`, with
  `gamma = gbar(lambda, filter input, instrument) + tau_gd gdet^(det, xi, lambda) + tau_ge gep^(t, lambda) + tau_gx u_e - m(lambda)`.
  `alpha = 1` for _cal/_flt input and `|det dOmega/dxi|` for `_rate` input (header Jacobian from central differences of the per-pixel
  positions, times (I + M_e), plus dD/dxi). `--amplitude unit` sets gamma = 0 (integral of K = alpha T); `free` drops the normalisation of k
  (the integral is learned).
* **Gauges** (all declared, logged by `gauge_report`):
  1. Throughput, exact: `m(lambda)` = T-weighted mean of the raw gamma over the reference set (training exposures, or one instrument with
     `--gauge-ref-inst`) on a 3x3 lattice, subtracted at every lambda node. The mean g of the reference set equals the published curve: one
     condition per lambda.
  2. Common blur ("sky at the resolution of the sharpest kernel"), hard reparametrisation. Every kernel is evaluated at `v / s`, with
     `s = sqrt(max(1, floor / M_ref))` (two fixed-point passes). `M_ref` is the core second moment (Gaussian window of width 2 sqrt(floor))
     of the mean kernel of the sharpest kernel class (filter with the smallest generic width). `floor = --blur-floor x pitch^2 / 3` is the
     second moment of one sky tent. No kernel class can then be sharper than the sky grid supports; above the floor the blur stays free.
     Choice: a floor (not an equality) leaves the data free wherever they constrain the blur, and only stops the degenerate direction that
     produced sub-grid kernels in the first synthetic runs (kernel core 15x narrower than truth, sky carrying the blur, held-out chi2 rising
     with fit length). A common dilation is a reparametrisation of the kernel family, not the exact convolution gauge (F18).
  3. Pedestal: per filter, `sum over training exposures of (b_e0 - median_e) = 0`, projected after every nuisance step. NIRCam stripes are
     projected to zero mean over the rows and columns each exposure uses. A constant sky pedestal can then no longer trade against the
     backgrounds.
  4. Common shift: the learned translations of the training exposures are projected to zero mean after every step. The initial registration
     (register_xcorr shifts for LMC) is kept as given. An earlier penalty version also counted the initial shifts and acted only on the
     exposures of each batch, which pulled LMC exposures by up to ~200 mas; it was replaced.
* **Detector**: `N_e` = optional gated non-linearity (`--nonlin`). `B_e` = quadratic Legendre per exposure + per-detector 8x8 pattern
  (zero mean) + NIRCam row and column stripes. Noise: `sigma^2 = (sigma_pipe exp(h[det, filter]))^2 + (f_chan S)^2`. Loss: Huber (delta 3)
  or Gaussian on `z = (d - m)/sigma`, plus `log sigma`.

## Rendering (render.py): two quadratures of the same integral, both converge

1. **Overlap-add FFT** (Nagy & O'Leary 1998). The kernel is sampled on nested square levels: pitch `p_r f^l` (default f = 2,
   `(2 n_h + 1)^2` nodes, n_h = 24), with a C1 partition of unity in r between levels (hand-over between `a R_l` and `R_l`, a = 0.5), out to
   `--kernel-extent` (default 10"). The render pitch `p_r` is set per channel: the largest `sky_pitch / k` at or below pixel/6 (NIRCam SW
   0.005", LW 0.01"). This grid is finer than the finest detector pixel, as the plan requires. Fine levels read the tent sky exactly;
   levels of 2 or more sky pixels read box-averaged coefficients. Each exposure is cut into regions of `--region-px`. K is evaluated at the
   region nodes, the sky is correlated with each node kernel by FFT, and the results are read at the pixel positions by prefiltered cubic
   B-splines and blended with the bilinear node weights. In training the level-0 grid gets a random sub-pitch phase each render, so the
   field is learned as a continuous function (without it, the field was unconstrained between grid nodes and the reference did not converge).
2. **Per-pixel quadrature** (reference): k is evaluated with each pixel's own code at a dense square lattice in the core plus stratified,
   jittered, log-uniform rings outward (importance density proportional to 1/r^2). The sky is read exactly at every point, and Z is computed
   from the same points.

Measured (tests/test_fieldop_render.py; sharpest synthetic kernel sigma ~ 0.022", 144 pixels around a bright source, reference
self-consistency 3e-4):

| quantity | values (max relative error of the FFT render against the reference) |
|---|---|
| render pitch 0.02 / 0.01 / 0.005" (B-spline) | 2.0e-2 / 2.0e-3 / 5.8e-4 |
| same, Keys bicubic, 0.01 / 0.005" | 5.9e-3 / 2.5e-3 |
| region 40 / 20 / 10 / 5 / 2 px (kernel core width varying by 28% across the 40-px detector) | 6.1e-3 / 2.5e-3 / 2.0e-3 / 1.5e-3 / 6.5e-4 |
| lambda nodes Q = 1 / 2 / 3 / 5 / 8 (vs Q = 32) | 1.3e-2 / 4.1e-3 / 2.3e-3 / 1.2e-3 / 6.2e-4 |

## Learning (train.py)

* **Warm start**: the kernel field is fitted (400 Adam steps, continuous random offsets) to a generic profile:
  `log[0.85 N(sigma) + 0.15 Moffat]`, with `sigma^2 = (0.42 lambda/D)^2 + p^2/12`. The sky starts from an inverse-variance splat coadd of
  the training pixels (ridge spectral fit, v40). A **registration pre-pass** (2 x `--reg-prepass-steps`) then fits per-exposure
  translations and backgrounds only, and the coadd is rebuilt at the registered positions. Catalogue-free; it plays the role of the coarse
  shift. On the synthetic field it takes the offset residual from 41 to 10 mas before the main fit.
* **Joint fit**: per step, `--inner-sky` Adam steps on the sky only (operator frozen), then one joint step on the operator, the
  per-exposure nuisances (Adam rows of exposures absent from the batch are not moved) and the sky. Optional cross-fitting `--xfit 2`.
  The sky has a fixed diagonal preconditioner (brightness-adaptive step). The learning rate warms up, then follows a cosine on
  progress = max(step fraction, time fraction).
* **Sampling**: patches of `--patch-arcsec`. A fraction `--is-frac` is drawn around local maxima of the current model band images (no
  catalogue; the importance sampling of opjoint, at patch level), with exact `1/(N pi)` reweighting.
* **Variability scales tau.**
  * Function axes (kernel space, time, lambda; throughput space, time; static and drift distortion): held-out (validation) likelihood, as a
    sequential evidence rule.
    1. One dither group per filter is carved out of training. Its nuisances are fitted on half of interleaved 4x4-pixel blocks; the
       gradient of the loss on the other half with respect to log tau is accumulated over `--tau-accum` evaluations.
    2. An exponential moving mean and variance (`--tau-beta`) give a significance `z`, and log tau moves by
       `-lr_tau clip(z + z_occam, -3, 3)`. An axis grows only while the held-out data ask for it at more than `z_occam` (default 2)
       standard deviations; otherwise it shrinks geometrically.
    3. Likelihood-ratio check (`--ab-every`): every few validation evaluations, the held-out loss on the same validation patches is
       recomputed with each function axis switched off. Paired differences are accumulated with forgetting (`--ab-beta`). An axis is KEPT
       while switching it off is significantly worse (z > `z_occam`); kept axes follow the validation gradient without Occam pressure,
       others shrink geometrically. Without this check, the gradient rule had no stable fixed point (an injected axis whose effect was
       captured showed z ~ 0, was shrunk, and regrew), which made held-out chi2 non-monotone. Limitation: switching off a co-adapted term
       is not a refitted likelihood ratio, so epoch terms (throughput time, drift distortion) can be kept although their output-space
       effect is small.
    4. Code-space kernel scales are capped at `--tau-code-max`, because they are not identifiable on their own and ran away (0.9 to 6.7)
       while held-out chi2 rose.
  * Random-effect axes (kernel z_e, gain u_e, distortion zD_e): a damped parameter-expanded EM step, a type-II estimate without the
    posterior-variance term.
  * Why not plain validation gradients: first-order Adam steps on noisy validation gradients random-walk (the first synthetic runs drifted
    the wrong way). Why not a Laplace evidence for the function axes: it needs Gauss-Newton log-determinants of neural fields.
  * The checks are gauge-invariant output-space quantities (relative kernels as Fourier ratios, relative throughput), not the taus
    themselves.
* **Held-out exposures**: first a translation + background pre-fit (`--held-reg-steps`, as the training pre-pass; without it one held-out
  synthetic exposure kept a 10+ mas offset and chi2 6.8), then only background, affine, gain u_e, codes z_e and zD_e, and stripes are fitted (protocol v1). The whole cut is
  then rendered by FFT and written as `pred/<FIELD>/<METHOD>/<SPLIT>/<rootname>.npz` (model, model_valid, meta).

## CLI ablation axes

`--k-space/--k-time/--k-lambda/--k-exp {on,off,learned}`, `--kernel-extent`, `--render-pitch` (+ `--render-os-pix`, `--level-factor`,
`--level-nh`, `--region-px`, `--n-lambda`, `--interp`, `--render {fft,quad}`), `--sky-pitch`, `--amplitude {unit,structured,free}`,
`--filter-input {token,curve}`, `--input-level {cal,rate}`, `--loss {gauss,huber}`, `--tau-mode {fixed,learned}`.

## Speed

* Batched rendering (`--batch-render`, default): the patches of a pass are rendered together per channel, with one window size per level,
  one grouped kernel-field evaluation and batched FFTs. It equals the per-patch render to 5e-7 (values) and 3e-5 (gradients). Device syncs
  were removed from the hot path: no `.item()`-type reads, boolean indexing, `torch.unique` or float conversion of GPU tensors per patch.
* Detector-aligned fine levels (`--render-frame detector`, default): levels finer than 2 sky pixels sample the tent sky on a grid aligned
  with the detector. The FFT window is then the patch itself rather than its sky-aligned bounding box, which for NIRCam at the SMACS
  orientation (~35 deg) is up to 1.9 times larger. Accuracy on a 35-deg detector against the per-pixel reference: 9.7e-4 (detector frame)
  vs 1.6e-3 (sky frame).
* Measured on the 1g.10gb H100 MIG slice (both SW render pitch 0.005", Q = 5, extent 10"):

| | patch renders (with backward) per second | s/step (batch) |
|---|---|---|
| first smoke run, per-patch loop | ~10 | 2.26 (6, incl. validation) |
| batched, checkpointed kernel (SMACS long run, package 34da161c) | ~10.9 | 1.73 (5, incl. validation) |
| + detector-aligned fine levels, sync-free (benchmark, c356b2d1) | ~16 | 1.15 (5, incl. validation from 35%) |
| same, batch 8 (final run, 268992e1; peak 5.4 GB) | ~18 | 1.31 (8, before validation) |

  Batching did not raise throughput: the slice is compute-bound by the fine-level FFTs (about 100 channels of ~800^2 per pass of 5 SW
  patches), not by launches or syncs. The detector-aligned windows give the measured 1.5x; batch 8 (memory now allows it) brings the total to ~1.8x. Further levers, with their measured accuracy cost: SW render pitch 0.01" (4x fewer
  level-0 nodes, render error 2e-3 vs 6e-4), Q = 3 (1.7x fewer channels, lambda error 2.3e-3 vs 1.2e-3), `--inner-sky 1` (one sky pass
  fewer per step). The 5x target is NOT met.

## Expected cost per step and memory (~11 GB GPU slice)

* Kernel evaluations per patch: (region nodes touched, usually 4) x Q (5) x (levels' nodes: SW 12.5 k, LW 10.8 k at 10" extent), i.e.
  about 0.25 M field evaluations per patch, ~3 kFLOP each with a 64-wide, 3-layer FiLM MLP, forward.
* FFTs per patch and level: 20 correlations on a window of (patch + 2 R_l)/p_l + 2 n_h + 6 nodes. Level 0 dominates: about 600^2 for a
  2.5" SW patch at 0.005", about 350^2 for LW at 0.01". There is one extra FFT pair for the B-spline prefilter.
* Measured on the smoke run (A100 slice, 6 patches): see "Smoke run" below. On 2 CPU cores the synthetic test takes 1.1-1.4 s/step
  (4 patches of 36^2 px, render pitch 0.02"), and real SW patches about 2.7 s/step for 2 patches.
* Memory: exposures on the GPU about 17 B/px (SMACS box + 2": about 30 M px, ~0.5 GB). Sky 4.5 M coefficients plus Adam (~55 MB). Per
  patch with autograd, about 0.2-0.4 GB (kernel-field activations about 20 x 12.5k x 64 x 9 floats, plus level-0 FFT buffers). Batch 6:
  2-3 GB. `--ckpt-kernel 1` recomputes the field in the backward pass if a larger batch or extent is needed. The 11 GB slice is not the limit;
  time per step is.

## Smoke run (SMACS NIRCam, dither split, 10 Nov 2026)

RUN `fieldop_smoke_smacs_s0`, session c96g2njj, package `fieldop-517c79cd.tar.gz` (the version before the EM update of the random-effect
scales and the automatic prior weight), torchsky 9749c80, H100 NVL MIG slice of 11 GB. Arguments:
`--budget-min 22 --batch-patches 6 --reg-prepass-steps 60 --held-steps 150`, all other flags at their defaults (sky 0.04", render
0.005"/0.01", extent 10", region 256 px, Q = 5, every axis learned).

* Data: 123 exposures (86 train, 13 validation = one dither group per filter, 24 held out). Load 74 s; kernel warm start 21 s;
  registration pre-pass 2 x 60 steps, 84 s (translation rms 7.7 mas).
* Fit: 585 steps in 1321 s, **2.26 s/step** for batch 6 (2 inner sky passes + 1 joint pass + validation every 4 steps: about 22 patch
  renders with backward per step, ~0.1 s each). Held-out nuisance fit 89 s; prediction of 24 exposures 85 s; total 28 min of Python.
  No out-of-memory at batch 6 without checkpointing (peak memory was not logged in this version; it is now).
* Protocol score (score.py, 24/24 exposures, 0% missing pixels):

| | chi2/N S/N >= 5 (headline) | <1 | 1-5 | 5-20 | 20-50 | 50-200 | >200 | frac. resid. rms S/N > 50 |
|---|---|---|---|---|---|---|---|---|
| fieldop smoke, with offset | **1.90 [1.82, 2.00]** | 0.98 | 1.17 | 1.33 | 1.33 | 4.63 | 125 | 0.076 |
| stage3_blot, with offset | 1.80 [1.61, 2.02] | 1.07 | 1.12 | 1.17 | 1.21 | 6.48 | 91.6 | 0.051 |
| fieldop smoke, without offset | 1.81 [1.73, 1.92] | 1.16 | 1.07 | 1.20 | 1.32 | 4.68 | 125 | 0.076 |
| stage3_blot, without offset | 1.81 [1.64, 2.02] | 1.16 | 1.11 | 1.18 | 1.22 | 6.49 | 91.6 | 0.051 |

  Per filter (with offset), fieldop vs stage3_blot: F090W 2.15 vs 3.13, F150W 1.99 vs 1.66, F200W 1.72 vs 1.35, F277W 1.72 vs 1.74,
  F356W 2.11 vs 2.05, F444W 2.05 vs 2.12. The headline is a statistical tie with the stage-3 baseline after a 22-minute smoke run. fieldop
  is better at S/N 50-200 and on F090W, and worse in the brightest bin (>200), at S/N 5-50 and on F150W/F200W. Learned tau at the end:
  kernel space 0.14, kernel lambda 0.35, kernel exposure 0.54 (code space); throughput space 0.026, exposure 0.010; static distortion
  3.8 mas, exposure distortion 2.0 mas. Epoch axes were off: SMACS spans 0.28 days.

## Long SMACS run (dither split): RUN `fieldop_smacs_long_s1`, package 34da161c, 200-min fit, 6950 steps

Package 34da161c has the blur floor, the pedestal gauge, held-out registration, batched rendering and the first evidence rule. It does not
have the likelihood-ratio check or the detector-aligned levels. Batch 5, checkpointed kernel.

| | chi2/N S/N >= 5 | 5-20 | 20-50 | 50-200 | >200 | frac. resid. S/N > 50 |
|---|---|---|---|---|---|---|
| fieldop, with offset | **1.77 [1.68, 1.86]** | 1.29 | 1.36 | 3.35 | 116 | 0.057 |
| stage3_blot, with offset | 1.80 [1.61, 2.02] | 1.17 | 1.21 | 6.48 | 91.6 | 0.051 |
| fieldop, without offset | 1.76 [1.68, 1.85] | 1.28 | 1.36 | 3.35 | 116 | 0.057 |

Per filter vs stage3_blot: F090W 1.69 vs 3.13, F150W 1.73 vs 1.66, F200W 1.50 vs 1.35, F277W 2.02 vs 1.74, F356W 2.15 vs 2.05,
F444W 2.01 vs 2.12. This is still a statistical tie on the headline (smoke run: 1.90). On SMACS the evidence rule pruned the spatial and
chromatic kernel variation and the throughput variation (tau -> 1e-6). This is plausible: the 20" box covers a small part of each detector.

A final run with the current package (268992e1: likelihood-ratio tau, detector-aligned levels, batch 8, 150-min fit) is running as method
`fieldop_v5` (session mq2grca9, RUN `fieldop_smacs_v5_s0`). Its score will be in `protocol/v1/scores/SMACS/fieldop_v5/dither.json`.

## Synthetic recovery (tests/test_fieldop_synth.py; numbers in tests/fieldop_results/synth_recovery*.json)

Two instruments (0.10" and 0.18" pixels), 3 filters, 12 exposures per filter (9 train, 2 validation, 1 held out), two epochs. Injected:
PSF width x (1 + 0.15 xi_x), PSF width proportional to lambda, throughput 1 + 0.08 xi_y, per-exposure offsets of 41 mas rms. Analytic truth
gives held-out chi2/N 1.04. The test FAILS its "approaches the noise" threshold (1.5):

| | 10 CPU-min (428 steps) | 30 CPU-min (1359 steps) |
|---|---|---|
| held-out chi2/N (warm start 16.6, truth 1.04) | 2.95 | 4.00 |
| held-out chi2/N, model S/N > 5 | 11.3 | 14.6 |
| offset residual, training / held-out (truth 41 / 54 mas) | 9.5 / 4.5 mas | 11.2 / 6.6 mas |
| effective kernel variability, space (injected) / time / exposure | 0.25 / 0.034 / 0.033 | 0.23 / 0.039 / 0.041 |
| tau throughput space (injected, truth ~0.065) / time / exposure | 0.012 / 0.012 / 0.0016 | 0.0066 / 0.010 / 0.0031 |
| distortion static / drift / exposure (none injected) | 8.1 / 3.0 / 0.6 mas | 14.0 / 1.1 / 1.0 mas (exposure rms 2.3) |

What works:
* The registration pre-pass and the joint fit recover 73-92% of the offsets.
* The random-effect scales (EM) collapse when nothing is injected.
* The spatial kernel axis is 6-7 times larger than the non-injected kernel axes in output space.
* The throughput gauge holds to 1e-8.

What does not:
1. The throughput spatial term is not separated from the time term.
2. The static distortion absorbs ~10 mas that was not injected.
3. Longer fitting makes held-out prediction worse. The learned kernel core is about 15 times narrower in second moment than the truth
   (m2 inside 0.3": 0.00053 vs 0.0082 arcsec^2) and its variation across the detector is overestimated (ratio 5.4 vs 1.32). The sky has
   absorbed the blur, and a pedestal of ~2.3 units has moved from the per-exposure background into the sky.

These are exactly the two gauges declared but left unconstrained (common blur; smooth sky vs background, F18). They have to be fixed, or
broken by data diversity, before the Phase 2 gate. In the synthetic set the two instruments share no wavelength, so nothing breaks them.

### Second iteration: gauge-invariant checks (30 CPU-min each)

Field redesigned for identifiability: 4" detectors, dithers +-0.6", 4 rotations, 12 training dithers per filter, B1 overlapping A2 in
wavelength, sky pitch 0.06". The checks are gauge-invariant: held-out chi2 at checkpoints (nuisances refitted each time), relative kernels as
Fourier ratios |K_a(f)| / |K_b(f)| (f = 1.5-4.5 cycles/arcsec) against the true ratios, relative throughput, and relative offsets.

| run (changes) | held-out chi2/N at 0 / 15 / 30 / 50 / 75 / 100 % (truth 1.00) | space_x learned / true | non-injected kernel axes (y, epoch, exposure) | lambda learned / true | throughput y learned / true (x, epoch) | offsets train / held (truth 41 / 46 mas) |
|---|---|---|---|---|---|---|
| v2 (blur floor, pedestal, held-out registration, evidence rule z0 = 2) | 21.1 / 1.45 / 1.33 / 1.27 / 1.38 / 1.36 | 0.68 / 0.81 | 0.017 / 0.006 / 0.004 | 0 / 0.69 | 0.096 / 0.128 (0.019, 0.001) | 14.3 / 5.9 mas |
| v4 (+ code-space cap, fields normalised on observed pairs, B pixel 0.12") | 19.3 / 1.36 / 1.22 / 1.56 / 1.36 / 1.27 | 0.70 / 0.81 | 0.002 / 0.037 / 0.022 | 0 / 0.69 | 0.105 / 0.128 (0.016, 0.014) | 23 / 22 mas |
| v5 (+ likelihood-ratio check) | 19.3 / 1.38 / 1.28 / 1.27 / 1.24 / 1.38 | **0.80 / 0.81** | **0.006 / 0.012 / 0.010** | 0 / 0.69 | **0.116 / 0.128** (0.022, 0.007) | **6.8 / 6.3 mas** |

* The first-iteration failure (held-out 2.95 -> 4.0) is gone: with the blur floor and the pedestal gauge, held-out is 1.24-1.38 from 15%
  of the fit onward. Five of six held-out exposures end at 1.11-1.25. One sharp-band exposure (A1_0) stays at 1.8-2.4 and causes the late
  rises; the curve is NOT monotone yet.
* The injected spatial PSF variation is recovered (gauge-invariant ratio within 14%), as are the injected throughput gradient (9%) and
  the offsets (85%). Non-injected kernel axes are 65-130x smaller than the injected one in output space.
* The within-band chromatic PSF variation is NOT recovered: the lambda axis is pruned in every run. Its effect on broadband data is
  second order (it shows only through source colour differences), so the held-out evidence for it stays below the 2-sd threshold at this
  data volume.
* Gauge-dependent (reported only): the blur floor is active (dilation 1.56; sharpest-class core m2 1.18e-3 = floor). Common-blur core m2
  2.3e-3 arcsec^2.

## Not yet done

* Speed. Python-level loops over patches and levels; one CUDA graph or batched patches would cut the GPU time per step several-fold. This is
  the main limit on budget-matched comparisons.
* HST (_flt/_flc) and LMC tiles through this package. The loader handles the protocol splits and the xcorr shifts, but only NIRCam detector
  naming has been run, and there is no multi-tile loop (opjoint's tile swapping is not ported).
* `_rate` input: alpha is implemented and tested, but the per-filter photometric conversion is not an input yet, and rate-trained
  predictions are not scorable against _cal.
* The non-linearity N_e exists (`--nonlin 1`) but is untested on data. There is no persistence, brighter-fatter or IPC term.
* The throughput-epoch field is still normalised over all lambda nodes (the detector fields now use observed pairs only), so its tau value
  is meaningless (15.7 in v5 for an output-space effect of 0.007). The likelihood-ratio check is not a refitted ratio.
* tau by validation likelihood is first order and noisy; it may need longer runs or accumulated validation gradients. A type-II estimate
  for z_e (16-dimensional, Laplace) would be a cheap complement.
* Cross-fitting (`--xfit 2`) is implemented but not exercised in tests.
* The Phase 2 gate itself. Recover the injected PSF variation and offsets: offsets partly, PSF width not (blur gauge), see above. Beat the
  best baseline on held-out dithers: the smoke run ties stage3_blot. Needed next:
  1. Fix the common-blur and sky-pedestal gauges (e.g. kernel core moment tied to the warm start, or the sky band-limited at the
     resolution of the sharpest kernel; background mean fixed per filter).
  2. Make the function-axis taus discriminate (larger and more validation sets, or a Laplace evidence on a linearised operator).
  3. Batched rendering for 5-10x more steps per GPU hour.
  4. Longer SMACS runs plus LMC tiles.
