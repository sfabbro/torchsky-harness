# fieldop: design note (Phase 2, general field operator), 10 Nov 2026; several sky patches added 11-12 Nov 2026

Package `fieldop/` (sky.py, bandpass.py, kernel.py, render.py, model.py, data.py, train.py, cli.py, synth.py, tiles.py), runner `run_fieldop.sh`
(on /arc as `scripts/run_fieldop.sh`), tests `tests/test_fieldop_{render,synth,gauge,area,tiles}.py` (item sky: `tests/test_items_*.py`, `tests/synth_small.py`). Every approximation is listed as F1...F39
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
* **Item sky, alternative** (`--sky-form items`, sky.py `ItemSky`, 13 Nov 2026; section "Item sky" below): the same `A_k` grid is produced
  by rasterising Gaussian items with continuous positions, widths, angles and non-negative spectral coefficients (+ an optional coarse
  smooth grid). The renderers are unchanged and still read the tent interpolant of that grid.
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

## Several sky patches (tiles.py; F32-F36), 11-12 Nov 2026

One operator, one sky per patch (the protocol's scoring boxes, "tiles"), one row of per-exposure nuisances per exposure. Ported from
opjoint's multi-tile loop (tile swapping, shared exposure rows, tile probabilities), rebuilt on fieldop's own data structures.

* **Data.** `--tiles all` (or a list of box ids). Each `_cal` file is read once (threads, `--load-workers`) and cut once per box it overlaps
  at its REGISTERED position (header + register_xcorr shift: with 1.8" header errors, a cut at header positions would fall partly outside
  the sky domain). A cut is a `data.Exposure` with its own pixels, header positions X, Y in its box's tangent plane, the tile id, the tile
  offset c_k and the GLOBAL exposure id `eid` shared by every cut of the file. Per-exposure metadata (kernel frame, noise level, pedestal
  reference median) come from the cut with most valid pixels (F33). With one box, cuts = exposures and the code path is the single-box one
  (bit-identical: all 120 state tensors equal after warm start, registration, fit with tau learning and the held-out fit, checked against
  the previous package).
* **Geometry across tiles.** The per-exposure affine acts about the exposure's own reference point r_e in the field frame (centroid of its
  valid pixels over all its cuts): `Omega = x + M_e (x + c_k - r_e) + t_e`, c_k = gnomonic offset of tile k's centre from the centroid of the
  selected boxes (F32). One (M_e, t_e) per exposure is then consistent in every tile (test: the same detector pixel lands on the same global
  position from either tile to 1e-5"), and the lever arm of M_e is the exposure's own extent. The first version used the field centroid
  as the reference (lever arm 110-136" for the far LMC boxes); it failed, see "What went wrong" below. The north directions of tile frames
  differ by 0.13 deg across the LMC boxes; ignored (< 0.2 mas). r_e is a plain device tensor, not a registered buffer, so the single-box
  path allocates nothing new (an extra allocation alone shifts float32 rounding by ~1e-6, which Adam amplifies to 0.06-0.2 differences in
  the end state; checked).
* **Shared rows.** Background quadratic, affine, gain u_e, kernel code z_e, distortion code zD_e and NIRCam stripes are per exposure. The
  stripe gauge masks are the union of the rows / columns of all cuts. The gradient on a shared row is the sum over its tiles (test).
* **Gauges.** Common shift per connected component of the (exposure, tile) training graph; pedestal per (filter, component of that
  filter's graph); throughput gauge global (F35). LMC: two components, boxes {0, 1, 2, 3, 5, 6, 7} and {4, 8, 9}; 12 pedestal groups.
* **Schedule.** A visit = `--visit-steps` (10) steps on one tile drawn with p_k proportional to (training pixels)^0.5 (LMC: dense boxes
  0.11-0.13, boxes 4, 8, 9 about 0.06). Within a visit the step is the single-box step (inner sky steps on that tile's sky, then the joint
  step of operator, nuisances and that sky; validation tau steps on the tile's validation cuts when it has any). One Adam per tile sky.
* **Memory (11 GB slice).** Pixel tensors live on the host (pinned) and are copied per tile into an LRU cache of `--gpu-resident-gb`
  (LMC: 11.0 GB of pixels, 0.43-1.67 GB per box). Skies stay on the device at 0.04" (~75 MB each with Adam); with `--sky-swap 1` the inactive
  skies and their Adam state go to the host too (needed at 0.02": ~0.3 GB per sky). Activation checkpointing of the kernel field
  (`--ckpt-kernel 1`) is needed at batch 8 on LMC: without it the first joint step ran out of memory (10.2 GB allocated); with it the peak
  is 7.3-8.0 GB including 2.1-2.5 GB of resident pixels.
* **Held-out exposures.** The nuisance fit cycles 4 times over the tiles (`--held-rounds`), with `--held-steps` x (held-out cuts of the tile
  / 24) steps per tile; each held-out cut is rendered with its tile's sky; the cuts of an exposure are merged into one file by the nearest
  tile centre, as stage3_blot merges its tiles.
* **Tests** (`tests/test_fieldop_tiles.py`, CPU, 4.5 min): shared-exposure handling (one row, consistent geometry, gradient additivity,
  gauge components); gradients identical (exactly, max difference 0) with the tile resident or evicted and copied back; Adam updates of a
  tile sky identical with and without sky swapping; a two-patch synthetic fit (independent analytic renderer, 2 instruments, 3 filters, 9
  exposures per filter of which 6 cover both patches): in 3 CPU-min (241 steps, both tiles visited ~30 times) held-out chi2/N 32.1 (warm
  start) -> 2.07 (truth 0.98), offsets 41.8 -> 23.4 mas rms, merged predictions rms 0.82-1.19 against the noiseless truth (noise 1). On
  the CPU the swap tests exercise the bookkeeping (copies, eviction, optimiser state) but not a second device; the first GPU run with
  `--sky-swap` found a device mismatch (band noise taken from a host-side sky), fixed.

### LMC runs (protocol v1, dither split, 62 held-out exposures; pipeline baseline `stage3_blot` 6.14 [5.25, 7.22])

All: 1 GPU (H100 MIG slice, 10.75 GB), batch 8, `--ckpt-kernel 1`, register_xcorr shifts, 1 registration pre-pass of 40 steps per tile,
`--n-val-groups 2` (10-tile runs). Setup ~13 min (9 of them reading 518 files from /arc), held-out fit + prediction + scoring ~35 min.

| run (method name) | tiles, fit | s/step, steps | headline chi2/N S/N >= 5 | F090W | F150W | F200W | F277W | F356W | F444W |
|---|---|---|---|---|---|---|---|---|---|
| stage3_blot (pipeline) | - | - | **6.14** [5.25, 7.22] | 28.1 | 3.96 | 3.56 | 16.2 | 7.74 | 8.26 |
| smoke (fieldop_lmc2t_smoke), sky 0.04" | 0, 4; 25 min | 1.97, 764 | 6.74 on its 35 exposures | 34.5 | 6.86 | 4.55 | 8.58 | 6.54 | 3.98 |
| fieldop_tiles, sky 0.04", tau learned | all; 290 min | 2.02, 8609 | 11.86 [10.38, 13.52] | 55.0 | 8.38 | 8.60 | 18.0 | 15.0 | 27.6 |
| fieldop_tiles_sky002, sky 0.02", tau learned, sky swap | all; 285 min | 2.25, 7592 | 13.54 [11.41, 16.12] | 113 | 9.57 | 8.06 | 16.8 | 18.2 | 24.9 |

Smoke run, same tiles as the pipeline (box 0 / box 4): 5.45 / 20.9 against 4.67 / 20.6. Per filter it already split the two ways it splits
below: better than the pipeline in the long-wavelength filters (F277W 8.6 / 8.2 vs 15.4 / 19.6; F444W box 4 8.5 vs 24.5), worse in the
short-wavelength ones (F090W box 0 37.8 vs 24.1).

Per box (with offset), pipeline / fieldop 0.04" / fieldop 0.02", sorted by the distance |c_k| of the box from the field centroid:

| box | 3 | 1 | 5 | 2 | 7 | 0 | 6 | 8 | 9 | 4 |
|---|---|---|---|---|---|---|---|---|---|---|
| \|c_k\| [arcsec] | 34 | 41 | 47 | 63 | 65 | 72 | 111 | 132 | 134 | 136 |
| stage3_blot | 8.01 | 4.93 | 7.24 | 4.47 | 4.21 | 4.67 | 8.41 | 21.0 | 17.3 | 20.6 |
| fieldop 0.04" | 9.84 | 6.21 | 9.77 | 5.96 | 5.17 | 5.80 | 23.5 | 139 | 78.4 | 53.4 |
| fieldop 0.02" | 9.58 | 5.39 | 11.6 | 5.85 | 4.91 | 5.20 | 21.6 | 175 | 114 | 106 |

**What went wrong, and the fix (F32 revised).** The ratio to the pipeline is 1.1-1.35 for the six boxes within 75" of the centroid and
2.6-8.4 for the four beyond 110" (box 6 is not sparse: 116 training cuts). The first version applied the per-exposure affine about the field
centroid, so M_e had a 110-136" lever arm in the far boxes: one Adam step of the matrix (~1e-4 at the start of the fit, 3e-5 in the
held-out fit's last 30%) moves positions by 4-14 mas there. Symptoms: the held-out predictions of `fieldop_tiles` carry a residual
translation of 9.5 mas median (F090W 18, F444W 26 mas) against 2.6 mas for the pipeline (lmc_anomalies.py), and the training chi2 per patch of
boxes 4, 8, 9 rose from ~0.6-1.0 to 1.3-3.5 over the fit while the near boxes improved. The affine now acts about each exposure's own
centroid r_e in the field frame (`Omega = x + M_e (x + c_k - r_e) + t_e`), still one (M_e, t_e) per exposure for all its tiles but with a
lever arm of the exposure's own extent. The SMACS path is unchanged (bit-identical). Runs with the fix: see below.

**Variability scales (learned runs, end of fit).** Output-space sizes (kernel axes: relative rms change of the level-0 kernel; throughput:
log g; distortion: mas):

| axis | 0.04" run | 0.02" run | smoke (2 tiles, 25 min) |
|---|---|---|---|
| kernel across the detector | 0 (pruned) | 0 (pruned) | tau 0.027, not kept (z 0.4) |
| kernel between epochs | 0 | 0 | tau 0.29, kept (z 3) |
| kernel within the band | 0 | 0 | tau 0.11, not kept |
| kernel per exposure (random effect) | 0.23 | 0.12 | |
| throughput across the detector / epochs | 0 / 0 | 0 / 0 | ~0.005 / 0.005 |
| gain per exposure | 0.0058 | 0.0049 | |
| static distortion | 0 | **2.35 mas, kept (z 4.6)** | 6.2 mas, kept (z 9) |
| distortion drift / per exposure | 0 / 0.69 mas | 0 / 0.64 mas | |

So no: in the 10-tile fits the evidence rule did NOT keep kernel variation across the detector, even though the ten boxes sample far more
of each detector than the SMACS box. The function axes were shrunk between 35% (tau start) and ~55% of the fit while the likelihood-ratio
z stayed below 2; once an axis is below 1e-5 it is skipped by the check and cannot return (absorbing). Two caveats make this a statement about
the rule, not yet about the instrument: (1) the tile probabilities put 82% of the visits, hence of the validation evaluations, on boxes 0-3 and 5-7
and 18% on boxes 4, 8, 9, and the far boxes were mis-registered by the lever-arm flaw above, so their evidence was corrupted; (2) the 2-tile smoke run kept the static
distortion (6 mas) and the kernel epoch term. A matched run with every axis on (`--tau-mode fixed`) tests whether the pruning costs held-out
accuracy (below).

### Matched 150-min runs: the affine fix, and variability on vs learned

| run (method name) | geometry | tau | s/step, steps | headline | F090W | F150W | F200W | F277W | F356W | F444W | frac. resid. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stage3_blot | - | - | - | **6.14** [5.25, 7.22] | 28.1 | 3.96 | 3.56 | 16.2 | 7.74 | 8.26 | 1.47 |
| fieldop_tiles_taufixed | about the field centroid | fixed (all on) | 1.69, 5326 | 12.19 [10.48, 14.47] | 55.1 | 5.91 | 9.91 | 28.6 | 12.8 | 21.7 | 27.1 |
| fieldop_tiles_affref_taufixed | about each exposure (fixed) | fixed (all on) | 1.67, 5402 | **8.86** [7.33, 10.74] | 37.5 | 4.90 | 4.64 | 26.2 | 12.1 | 17.0 | 2.41 |
| **fieldop_tiles_affref** | about each exposure | learned | 1.99, 4533 | **8.54** [7.32, 9.90] | 38.3 | 5.21 | 4.97 | 19.1 | 12.6 | 21.0 | 1.79 |

Per box, pipeline / old geometry / fixed geometry (both tau fixed): 0: 4.67 / 7.15 / 6.50; 1: 4.93 / 7.41 / 7.10; 2: 4.47 / 7.50 / 6.51;
3: 8.01 / 10.6 / 10.3; 4: 20.6 / 63.6 / 29.9; 5: 7.24 / 12.1 / 10.7; 6: 8.41 / 17.7 / 12.4; 7: 4.21 / 5.95 / 5.64; 8: 21.0 / 94.5 / 30.4;
9: 17.3 / 82.3 / 33.9. The fix removes most of the far-box failure (factor 2-3) and the fractional-residual outliers (27 -> 2.4); fieldop
is still behind the pipeline everywhere, by 1.1-1.5x in the near boxes and 1.4-2x in boxes 4, 8, 9. Per (filter, box), the
short-wavelength cells of the dense boxes are close (F150W / F200W in boxes 0, 1, 2, 7: 2.6-4.6 vs 2.3-3.4); the long-wavelength cells
are not (F277W 23.6-31.0 vs 13.2-25.4), unlike the 25-min learned-tau smoke run (F277W boxes 0 / 4: 8.6 / 8.2 vs pipeline 15.4 / 19.6).
The matched learned run (last row) scores the same as the fixed one (8.54 vs 8.86, intervals overlap): F277W is better with learned scales
(19.1 vs 26.2), F444W worse (21.0 vs 17.0), the rest within 0.4. Per box, pipeline / fieldop_tiles_affref: 0: 4.67 / 5.67; 1: 4.93 / 6.42;
2: 4.47 / 6.05; 3: 8.01 / 9.28; 4: 20.6 / 31.8; 5: 7.24 / 9.96; 6: 8.41 / 12.1; 7: 4.21 / 5.22; 8: 21.0 / 32.6; 9: 17.3 / 43.2.

**Where fieldop loses (fieldop_tiles_affref vs stage3_blot, by S/N of the official mosaic, with offset):** <1: 1.73 vs 1.85; 1-5: 2.01 vs
2.11; 5-20: 2.10 vs 2.23; 20-50: 4.72 vs 4.56; 50-200: 35.9 vs 29.0; >200: 536 vs 259. fieldop is better on the faint and intermediate
pixels (79% of the headline pixels are at S/N 5-20) and loses on star cores: the >200 bin (0.7% of the headline pixels) contributes 4.0 of
the 8.54 against 1.9 of the 6.14. Consistent with this, the blur gauge was strongly active in the corrected runs (common dilation 2.6-4.2
logged mid-fit: the kernel field pushes the sharpest kernel class, F090W, well below one sky tent). On LMC (stars) the 0.04" sky cannot hold
point sources sharp enough for the short-wavelength cores; the 0.02" run was made with the old geometry, so its effect is still unmeasured.

**Variability scales, corrected learned run (end of fit, output space):** kernel across the detector 0, between epochs 0, within the band
0; throughput across the detector 0, between epochs 0; static distortion 0, drift 0 (all pruned, likelihood-ratio z <= 0.1 at the end);
random effects: kernel per exposure 0.053 (relative rms change of the kernel), gain per exposure 0.0031, distortion per exposure 0.60 mas.
Answer to "does the operator keep variation across the detector when patches sample much more of each detector than SMACS does": no,
neither in the kernel nor in the throughput, with the geometry fixed as well; and switching every axis on at its initial scale does not
improve held-out prediction (8.86 vs 8.54). Only the 0.02" run (old geometry) and the 2-tile smoke run kept the static distortion (2.4 and
6 mas). The evidence rule remains first order (F21); a refitted likelihood ratio per axis would be a stronger test.

**Timings and memory (10 tiles):** setup 13 min (reading 518 files: 9 min); registration pre-pass 40 steps x 10 tiles, 3 min; fit 1.65-1.70
s/step before validation, 2.0-2.3 s/step with validation; tile swaps cost 20 s per 290-min fit at 0.04" (0.1%) and 110 s with sky swapping
at 0.02" (0.6%); held-out nuisance fit (198 held-out cuts of 62 exposures) ~15 min; prediction ~10 min; scoring ~5 min. Peak GPU memory
7.9-8.8 GB of 10.75 GB.

**Rendering speed (task 4 of the plan): not attempted** this round (no GPU slot free while the LMC runs ran).

## Item sky (`--sky-form items`, sky.py `ItemSky`; F38, F39), 13 Nov 2026

    s(X, Y, lambda) = sum_n [sum_k c_nk phi_k(lambda)] G(X - x_n, Y - y_n; Sigma_n)  (+ tent interpolant of a coarse smooth grid, optional)

* **Parameters per item**: position (x_n, y_n) [arcsec, continuous]; widths `w_i = sqrt(floor^2 + exp(2 l_i))` (floor `--item-floor`,
  default 0.4 x sky pitch; cap `--item-max-width`, default 16 x pitch); angle; `c_nk = exp(r_nk)` on the package basis (K = 12). 5 + K = 17
  fitted numbers per item. Optional smooth grid `--item-smooth-pitch` (signed; `--sky-neg` applies to its band images only).
* **Minimal integration**: `coeffs(view)` returns the items rasterised on the K x N x N grid at `--sky-pitch`; everything downstream (tent
  read, pooled levels, FFT and quadrature renders, band images, bright-patch strata, held-out prediction) is unchanged. So the sky is still
  read through a tent grid at the sky pitch: the test is of the parametrisation, not of a grid-free renderer.
* **Rasterisation**: the cell integral is approximated by adding p^2/12 to both principal variances (box -> isotropic Gaussian of equal
  variance, which commutes with the rotation) and point-sampling at the nodes, on a patch of +-4 effective widths (patch half-sizes in
  18 buckets), renormalised so that p^2 x (patch sum) = 1 (exact flux; nodes outside the grid are then dropped, as the sky is zero there),
  scatter-added with `index_add` in chunks of 4 M elements. Measured (tests/test_items_unit.py): flux error before renormalisation 3.3e-2
  at the floor, 5.6e-3 at 1.25 floor, 6e-4 at 1.5 floor; per-node error against the exact cell mean 2.2 / 1.2 / 0.45 / 0.03 per cent of
  the peak at 1 / 1.5 / 2 / 4 x floor; centroid error of the grid 0.025 pitch at the floor, < 1e-3 pitch above 1.5 x floor; gradients
  (positions, log widths, angles, log coefficients, smooth grid) agree with finite differences to 5e-9 (float64).
* **Seeding (data-blind)**: square lattices at `--item-levels` sizes, level l holding a share 4^-l of `--item-count` (3 levels: 76 / 19 /
  5 per cent), width half the spacing; coefficients read from the warm-start coefficients (the same coadd the grid starts from) smoothed
  at the item width, times the level share and the item area, floored at 1e-3 x their median peak. Reseeded when the registration
  pre-pass rebuilds the warm start (`reset_from_coeffs`).
* **Optimiser**: one Adam per sky with groups positions (`--item-lr-pos` x pitch), log widths, angles, log coefficients (`--item-lr-*`),
  smooth grid (as the grid sky); untuned (steps from the image-only experiment). Optional relocation (`--item-relocate 1`, off): see F39.
* **Blur gauge**: unchanged (one tent, F18). A rasterised item read by the tent has a 2-d second moment of at least
  2 floor^2 + p^2/6 (cell) + p^2/3 (tent) = 0.82 p^2 at the default floor, against p^2/3 for one grid node: with the kernel floor of p^2/3
  the sharpest point source the item model can draw is m2 = 1.15 p^2 (grid: 0.67 p^2). At 0.04" this is 1.8e-3 arcsec^2 against an
  estimated F090W core m2 of ~5e-4 (0.42 lambda/D and the pixel box); i.e. at 0.04" both forms are already too blunt for the SW cores,
  the items more so (the long SMACS grid run fieldop_smacs_fix_200_s0 ends with the sharpest kernel class at the floor: M_ref 545e-6
  against a floor of 533e-6 arcsec^2). This did not show in the synthetic test (true m2 >= 6e-3), not even at a 0.10" sky pitch where it
  should bind (100 steps, far from convergence).
* **Synthetic comparison and default-path checks**: tables in ITEMS_REPORT.md (beside the package folder); default `--sky-form grid` gives
  bit-identical states to the untouched package (single box, 40 steps; two tiles with sky swapping, 12 steps).
* **Cost**: per rasterisation pass (forward + backward), CPU one thread: 0.08 s for 1500 items on 95^2 (synthetic, ~+15-35 per cent per
  step), 3.0 s for 60 000 items on 613^2 (SMACS at 0.04", 21.6 M patch nodes, ~0.4 GB saved for the backward pass).
* **Not done**: a grid-free renderer (items read without the tent); deconvolved (prefiltered) rasterisation to remove the cell + tent blur;
  cross-fitting with items (implemented, parameter-wise view average, untested); the smooth grid on a synthetic recovery; tuning of the item
  step sizes and counts.

## Not yet done

* Speed. Python-level loops over patches and levels; one CUDA graph or batched patches would cut the GPU time per step several-fold. This is
  the main limit on budget-matched comparisons.
* HST (_flt/_flc) through this package: only NIRCam detector naming has been run. (LMC tiles: done, see "Several sky patches".)
* Several tiles, not done: checkpoint / resume of long runs; skies of different sizes; cross-fitting (`--xfit`) with several tiles (refused);
  a per-cut kernel frame (one R_e per exposure, F33); the tile probabilities are not tuned (power 0.5 is a choice); loading reads 518 files
  in ~9 min with threads (I/O-bound on /arc), a process pool or a cached cut store would cut it.
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

## Item sky, 14 Nov 2026: memory-bounded backward pass and any-size rasteriser (F40)

* **Why**: the first two real-data item fits (SMACS, 60,000 items, sky pitch 0.04" and 0.02") ran out of GPU memory (10.75 GB): autograd
  kept every item patch (items x patch nodes x K) and the patches grew as the items widened during the fit (3.8 GB at step 0, 9.7 GB at
  step 700 at 0.04"; out of memory at step 1 at 0.02").
* **Backward pass recomputed in chunks** (`_ItemRaster`, `_raster_vjp`): the forward scatter keeps no graph; the backward pass recomputes
  each chunk of patches (4 M elements), differentiates it against the gathered grid gradient and frees it. Values and gradients are
  bit-identical to the plain autograd scatter (float32 and float64, 1500 items on 95^2); first order only (no double backward). The 0.04"
  item fit then holds 3.3 GB.
* **Any-size rasteriser** (`--item-raster pyramid`, default `direct` unchanged and bit-identical; `--item-patch {square,sheared}`): Gaussian
  semigroup. Level l has pitch 2^l p; going down one level is a fixed separable transposed convolution with a discretely normalised
  Gaussian of width s_l = p_l (taps +-5 s_l); an item is point-sampled at the coarsest level where the smaller principal width left after
  removing the cascade blur is still >= p_l (switch widths 2.81, 5.99, 12.2, 24.4, ... pitches), with covariance T_n - S_l^2 I. Measured
  (tests/test_items_pyramid.py and PYRAMID_REPORT.md beside the package folder): worst error 2.1e-4 of the item peak against the exact
  point-sampled Gaussian (target 1e-3), jump across a level switch <= 2.3e-4 of the peak, flux exact to 2e-15, gradients against finite
  differences <= 7e-7. The direct path truncates items wider than about 32 nodes (errors 0.05 to 3.4 of the peak); the pyramid path does
  not up to +-512 nodes per level. Cost (CPU, one thread, 60,000 items, K = 12, forward + backward): round widened items 7.1 s -> 2.3 s at
  613^2 and 23.9 s -> 4.5 s at 1226^2; elongated items need `--item-patch sheared` to gain (16.1 s -> 4.4 s; 53.5 s -> 8.7 s).
* **Not yet done**: GPU timings; a fit longer than 30 steps with the pyramid path (whether the 2e-4 switch jumps disturb Adam is untested).

## Flux convention and data-quality policy, 14 Nov 2026 (F41, F42)

* **`--kernel-norm-radius R`** (default 0 = unchanged): the kernel's unit integral is taken inside R only (window 1 to 0.85 R, C1 fall to 0
  at R). Why: with the unit integral over the whole +-10" support, a flat far pedestal of the kernel acting on the sky is absorbed by the
  background terms, which nothing pins; the reference SMACS sky (`fieldop_smacs_fix_200_s0`) carries about twice the official mosaic's
  flux (1.7 to 4 depending on aperture; onesky_v2/kernel_check/KERNEL_CHECK_REPORT.md). Fluxes then are "inside R" fluxes.
* **`--dq-ignore-bits B`** (default 0 = strict DQ == 0): data-quality bits that do not exclude a pixel. Why: the SMACS `_cal` files
  re-fetched from MAST on 13 Nov 2026 (CAL_VER 3.0.0, CRDS context 1584) flag the brightest cores SATURATED (bit 2) where the protocol's
  held-out masks (built from the earlier files) call them valid; with strict DQ == 0 the fit never sees those cores. `B = 2` keeps them.
* **Scoring** on re-fetched files: `protocol/score_paired.py --dq-zero` also requires the scored file's flag to be zero (0.35 per cent of
  the SMACS held-out pixels dropped), so that fits made before and after the re-fetch are compared on pixels all of them could train on.

## Rate input and per-pixel gain, 16 Nov 2026 (F43-F46)

* **Question**: does the flat field calibrate itself from the science exposures? `--input-level rate` fits `_rate` images (DN/s; no flat,
  no photometric conversion; WCS from the matching `_cal` header, F46); `--pixel-gain detector` adds a per-pixel log-gain map per detector
  that multiplies the light on the pixel (sky signal and sky background; not the pattern and the stripes), gauge by exact projection
  (mean and degree-2 monomials over the observed pixels, F44), its own Adam (F45). Default off, bit-identical (synthetic checksum).
* **Scoring**: `score_paired.py --level rate` (P7); `flat_check.py` compares a learned map with the official flat (C = rate / cal).
* **Results and launch recipe**: RATE_GAIN_REPORT.md beside the package. Synthetic: an injected 3 per cent flat is recovered with
  correlation 0.92 / 0.89 (0.98 / 0.95 expected at the noise limit), map error 1.4-1.8 x the noise limit; held-out background chi2 2.85
  without the map, 1.48 with it, 1.105 with no flat error. SMACS prediction: noise limit per pixel about equal to the official
  pixel-to-pixel rms (median 2 to 6 training cuts per pixel), expected correlation 0.57-0.69 at best: the real test is marginal.
* **Not done**: no real-data fit (GPU); the `cli` path with `--input-level rate` was not run here (torchsky is not installed on the CPU
  machine); per-filter photometric conversion is still not an input (F15).
