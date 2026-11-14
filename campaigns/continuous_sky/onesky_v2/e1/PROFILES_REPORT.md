# E1: items with a learned, size-scaled profile (profiles.py)

**Verdict:** at equal size, the learned profile does **not** beat the fixed Gaussian. Where it differs, it is worse. Running psi frozen on an unseen sky works mechanically, but at budgets of 15,000 and 45,000 it only ties the Gaussian, and at 5,000 it is much worse. The latent does not stay the same across zooms.

## What was built
- **Item parameters:** each item has a position (2 numbers), two log widths (2, so log size and axis ratio), an angle (1) and a log flux (1). That is 6 numbers for a Gaussian item and 10 for a learned-profile item, which also carries a latent of dimension 4. There is one background scalar per image. The width is w = sqrt(0.45² + e^(2l)), so no width parameter can go below 0.45 px. This is the same parametrisation as the Gaussian baseline.
- **The profile psi:** it is defined in polar coordinates. A radial log-profile on 12 knots, uniform in log(1+rho), is multiplied by [1 + harmonics m = 1, 2, 3 whose coefficients depend on radius], with a cos² taper from 2.55 to 3 widths.
  - A shared network (latent → 12 tanh units → 84 table values) produces the knot values. It has **1152 weights**.
  - For each item this polar form is evaluated on a 19×19 table spanning ±3 widths (node spacing 1/3 width). psi is *defined* as the bilinear interpolant of that table.
  - Non-negativity is exact: the sum of harmonic amplitudes is capped below 0.98, and bilinear interpolation of non-negative nodes stays non-negative.
  - The integral is exactly (spacing)² × the sum of the nodes, which is normalised to 1. Brute-force quadrature gives 1.00005.
  - The support is bounded and the bandwidth is fixed by the table.
  - The fixed-Gaussian mode uses the same items, renderer, seeding and optimiser, with an analytic Gaussian truncated at 3 widths and renormalised.
- **Rendering:** each item is evaluated on a square patch (side ≥ 6 w_max + 2, from 21 bucket sizes), clamped inside the image, and added into the image with index_add.
  - Pixel integration uses o×o point sub-samples with o = clip(ceil(3/w_min), 1, 7).
- **Seeding (identical for both profiles, never looks at the data):**
  - coarse lattices at widths 32, 16, 8 and 4 px with spacing 2w, up to 25% of the items;
  - one third of the remaining items on a covering lattice;
  - two thirds on an offset lattice at width 1 px.
  - The adaptive variant, at 20%, 35% and 50% of the run, moves at most 5% of the items: those whose S/N proxy flux/(sigma·sqrt(4π w1 w2 + 1)) is below 1. They go to positions drawn in proportion to the positive residual in noise units, smoothed over 1 px.
- **Optimiser:** Adam with a cosine schedule over **wall-clock time** (220 s per fit and 480 s for the joint training), so the iteration counts vary with machine load (see table).

## Sanity checks (profiles_selftest.py)
- **Gaussian mode against exact erf pixel integrals:**
  - per-pixel error up to 1.6% of the peak at R=3, caused by truncating and renormalising at 3 widths;
  - 0.5–0.7% of the peak with R=4, before I reduced the support;
  - integrated flux within 6e-4 of the flux.
- **Learned psi:**
  - integral 1.0000;
  - rendered flux/flux between 0.996 and 1.005 for the trained psi at all widths.
- **Pixel integration for the trained psi** (my sub-sampling against 16×16 sub-samples):
  - widths 0.45–0.7 px: max error/peak 0.6–1.6%;
  - widths 1–1.6 px: 3–7%;
  - width 3 px: **19–26%**.

## Results (all runs from profiles.py; n=256; the bands are taken from the result JSON files)
| method | variant | target | budget | fitted image/shared | chi2_truth | PSNR | s / iter. | frac. error, bands 1-2, 2-4, 4-8, 8-16, 16-32, 32-64, 64-128, 128-1024 px |
|---|---|---|---|---|---|---|---|---|
| profile_lattice | s0 z1.0 | noisy | 5000 | 3841/1152 | 1.099 | 28.7 | 220/649 | 0.570 0.577 0.502 0.391 0.286 0.234 0.170 0.077 |
| profile_lattice | s0 z1.0 | noisy | 15000 | 13841/1152 | 0.203 | 49.3 | 221/222 | 0.100 0.056 0.040 0.032 0.023 0.019 0.012 0.004 |
| profile_lattice | s0 z1.0 | noisy | 45000 | 43841/1152 | 0.398 | 49.4 | 221/136 | 0.102 0.057 0.039 0.027 0.020 0.015 0.010 0.004 |
| profile_lattice | s0 z1.0 | truth | 15000 | 13841/1152 | 0.062 | 44.5 | 221/265 | 0.073 0.090 0.085 0.066 0.051 0.037 0.029 0.014 |
| profile_adaptive | s0 z1.0 | noisy | 5000 | 3841/1152 | 1.144 | 28.6 | 220/706 | 0.587 0.588 0.511 0.398 0.290 0.239 0.165 0.078 |
| profile_adaptive | s0 z1.0 | noisy | 15000 | 13841/1152 | 0.258 | 41.2 | 221/233 | 0.126 0.133 0.123 0.097 0.074 0.057 0.042 0.016 |
| profile_adaptive | s0 z1.0 | noisy | 45000 | 43841/1152 | 0.493 | 40.3 | 222/135 | 0.244 0.160 0.116 0.087 0.067 0.057 0.041 0.017 |
| profile_adaptive | s0 z1.0 | truth | 15000 | 13841/1152 | 0.021 | 54.0 | 221/272 | 0.042 0.030 0.027 0.020 0.016 0.013 0.009 0.004 |
| profile_fixedgauss_lattice | s0 z1.0 | noisy | 5000 | 4999/0 | 0.482 | 33.3 | 221/312 | 0.301 0.327 0.309 0.243 0.177 0.145 0.127 0.041 |
| profile_fixedgauss_lattice | s0 z1.0 | truth | 15000 | 14995/0 | 0.005 | 54.8 | 220/355 | 0.108 0.028 0.007 0.003 0.002 0.002 0.001 0.001 |
| profile_fixedgauss_adaptive | s0 z1.0 | noisy | 5000 | 4999/0 | 0.434 | 34.0 | 221/315 | 0.298 0.310 0.279 0.213 0.156 0.128 0.112 0.035 |
| profile_fixedgauss_adaptive | s0 z1.0 | truth | 15000 | 14995/0 | 0.004 | 55.1 | 220/363 | 0.103 0.028 0.006 0.003 0.002 0.001 0.001 0.000 |
| profile_fixedgauss_lattice | s10 z1.0 | noisy | 5000 | 4999/0 | 0.119 | 45.7 | 220/426 | 0.093 0.050 0.044 0.049 0.041 0.031 0.022 0.009 |
| profile_fixedgauss_lattice | s10 z1.0 | noisy | 15000 | 14995/0 | 0.232 | 46.1 | 221/219 | 0.087 0.048 0.045 0.041 0.034 0.026 0.018 0.006 |
| profile_fixedgauss_lattice | s10 z1.0 | noisy | 45000 | 44995/0 | 0.498 | 43.5 | 220/170 | 0.117 0.074 0.050 0.041 0.034 0.026 0.018 0.007 |
| profile_fixedgauss_adaptive | s10 z1.0 | noisy | 5000 | 4999/0 | 0.127 | 43.3 | 221/431 | 0.110 0.070 0.057 0.058 0.048 0.038 0.029 0.014 |
| profile_fixedgauss_adaptive | s10 z1.0 | noisy | 15000 | 14995/0 | 0.233 | 46.1 | 221/220 | 0.087 0.048 0.045 0.041 0.034 0.026 0.018 0.006 |
| profile_fixedgauss_adaptive | s10 z1.0 | noisy | 45000 | 44995/0 | 0.501 | 43.4 | 222/173 | 0.119 0.075 0.050 0.041 0.034 0.026 0.018 0.007 |
| profile_frozen_lattice | s10 z1.0 | noisy | 5000 | 4991/0 | 1.074 | 23.2 | 220/737 | 0.707 0.679 0.658 0.624 0.548 0.445 0.306 0.082 |
| profile_frozen_lattice | s10 z1.0 | noisy | 15000 | 14991/0 | 0.250 | 44.6 | 220/383 | 0.089 0.058 0.054 0.053 0.044 0.032 0.024 0.007 |
| profile_frozen_lattice | s10 z1.0 | noisy | 45000 | 44991/0 | 0.474 | 44.8 | 221/194 | 0.093 0.060 0.049 0.042 0.035 0.026 0.018 0.007 |
| profile_frozen_adaptive | s10 z1.0 | noisy | 5000 | 4991/0 | 1.029 | 23.4 | 220/727 | 0.670 0.660 0.654 0.625 0.530 0.466 0.309 0.115 |
| profile_frozen_adaptive | s10 z1.0 | noisy | 15000 | 14991/0 | 0.248 | 44.6 | 220/386 | 0.084 0.057 0.054 0.052 0.044 0.032 0.024 0.007 |
| profile_frozen_adaptive | s10 z1.0 | noisy | 45000 | 44991/0 | 0.476 | 44.7 | 220/191 | 0.096 0.061 0.049 0.043 0.035 0.026 0.018 0.008 |
| profile_frozen_lattice | s10 z1.4 | noisy | 15000 | 14991/0 | 0.209 | 42.4 | 220/406 | 0.284 0.082 0.051 0.051 0.048 0.034 0.032 0.007 |
| profile_frozen_lattice | s10 z2.0 | noisy | 15000 | 14991/0 | 0.203 | 41.8 | 220/385 | 1.847 0.148 0.060 0.053 0.045 0.040 0.030 0.010 |
| profile_frozen_lattice | s10 z2.8 | noisy | 15000 | 14991/0 | 0.225 | 38.5 | 220/476 | 3.945 0.484 0.102 0.064 0.048 0.043 0.028 0.012 |

**Diagnostic, not budget-matched** (profiles_diag.py, sky 0 noise-free, 220 s):
- frozen psi with 2499 items (the Gaussian's count): chi2_truth 0.0040, compared with 0.0051 (lattice) and 0.0040 (adaptive) for the Gaussian;
- Gaussian with 1384 items (the learned run's count): 0.0121, compared with 0.062 and 0.021 for the learned psi trained on this image alone.

## What the numbers show
- **T1 (standard image, psi counted in the budget): the learned profile loses.**
  - Noise-free target at 15,000: chi2_truth is 0.062 (lattice) and 0.021 (adaptive). The fixed Gaussian in the same code is at 0.005 and 0.004, and the baseline gauss_* runs are at 0.010 and 0.004.
  - At 5,000: 1.10 and 1.14, against 0.48 and 0.43 for the same-code Gaussian (baseline gauss_*: 0.86 and 0.84).
  - At 15,000 and 45,000 on noisy data, every method sits near the noise-fitting level. A model with P free numbers that fits noise reaches about P/65,536, which is 0.23 and 0.69. So those runs say little about representation: the learned runs score 0.20–0.26 and 0.40–0.49, the baselines about the same.
  - The diagnostic splits the loss into two causes: (a) 4 extra numbers per item plus 1152 weights buy about 45% fewer items; (b) training psi from scratch together with the items in about 250 iterations is worse than a Gaussian even at the same item count (0.062 against 0.012).
- **T2 (frozen psi on unseen sky 10):**
  - At 15,000: 0.250/0.248 against 0.232/0.233 for the fixed Gaussian (Gaussian slightly better).
  - At 45,000: 0.474/0.476 against 0.498/0.501 (learned slightly better; both are noise-fitting).
  - At 5,000: **1.07/1.03 against 0.12/0.13**. In the frozen 5,000 run, 83% of the chi2 is in 100 pixels: bright stars that the 499 items never reached. The Gaussian had 833 items.
  - At equal item count, a pretrained frozen psi is roughly as good as a Gaussian or marginally better (diagnostic above, on a sky it was trained on). The latent's cost per item outweighs this at equal budget. The redundancy claim is not supported at these budgets.
- **T3 (frozen psi, sky 10, zooms 1, 1.4, 2, 2.8; lattice; 15,000):**
  - chi2_truth: 0.250, 0.209, 0.203, 0.225.
  - Matching (profiles_t3.py: same scene position within 0.5 × size, flux within ×1.5, size within ×2) finds only 41/354, 27/152 and 8/66 of the bright resolved zoom-1 items.
  - For the matched items, the median shift in log size (in pixels) is 0.25, 0.74 and 0.87, against log(zoom) = 0.34, 0.69 and 1.03. That roughly follows log(zoom), with wide spread (interquartile range about 0.5). The size window in the matching biases this towards agreement.
  - The latent does **not** stay the same: the median distance between matched latents divided by the median distance between random pairs is 1.08, 1.17 and 1.21. That is no better than random, because the decomposition into overlapping items is not unique.
  - Fractional error in the 1–2 px band exceeds 1 at zoom ≥ 2, because the truth has almost no power there and fitted noise dominates the ratio.

## Approximations, shortcuts, special treatment
- **Pixel integration:** point sub-sampling as above, with errors as measured. With the trained psi it is poor at widths of 1–3 px, so the effective profile depends a little on the absolute size. That breaks exact scale covariance. Halving the sub-sample spacing would fix it at about twice the cost; not done.
- **Width floor:** the floor applies to the scale parameter only. A learned psi can concentrate light inside its unit (the most compact latents have an RMS radius of 0.72 units, against 1.41 for the Gaussian), so a learned item can be **effectively narrower than 0.45 px**. A Gaussian item cannot.
- **Support and patches:** support is 3 widths for both modes; the Gaussian is truncated and renormalised. Patches are clamped into the image, so pixels outside the image are not computed.
- **Wall-clock schedule:**
  - At 15,000 and 45,000 the learned T1 runs got 222–272 and 135 iterations, against 355–363 for the same-code Gaussian and 650–1150 for the baselines (different code).
  - Most runs report "still falling" as False under the 1%-over-10% test, but several truth-target runs report True.
  - The 45,000 runs are under-converged.
- **Joint psi training:** 204 iterations over 480 s, on skies 0, 1 and 2 at 1499 items each (items-only budget 15,000), adaptive. The final chi2_truth was 0.26, 0.31 and 0.47.
- **T3 uses one method only** (frozen lattice), at budget 15,000.

## Tuning (modest, about 25 minutes of short development runs, all applied equally to both modes)
- Learning-rate multipliers of 0.5, 1 and 2, plus one test of flux rate 0.1, on a 128 px crop.
- One global rise (position rate 0.05 → 0.2 px, flux 0.03 → 0.05, etc.), checked once at 5,000 on the standard image.
- The seeding redesigned once, because the first version put too few compact items in at 5,000.
- Relocation softened: from 25% to 5% of items, threshold S/N 2 → 1, 3 events, after it damaged a crop fit.
- Support reduced from 4 to 3 widths, and psi switched from a per-sample polar evaluation to the bilinear table, for speed (about 7× faster).
- No tuning was done on sky 10.

## Not done
- No repeat runs over noise seeds or sky seeds, so I have no error bars. Run-to-run scatter from machine load looked like 10–30% in chi2 at 5,000.
- No longer runs for the 45,000 budget.
- No variant with a smaller latent (dimension 1–2) or a cheaper psi, which is the obvious next test of the per-item cost.
- No improvement to sub-sampling for the learned psi.
- T3 not run with the adaptive variant, and no matching on stars specifically.
