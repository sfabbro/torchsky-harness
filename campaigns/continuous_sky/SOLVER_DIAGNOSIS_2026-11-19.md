# Why the uncertainty solver stalls on the SMACS halves, and the fix (19 Nov 2026)

Scope: `fieldop/uncertainty.py` (the only package file changed), its tests (`tests/test_uncertainty.py`: one fixture line;
`tests/test_uncertainty_solver.py`: new), the reproducer and its results (`tests/solver_repro/`). Everything was run on a CPU with two
cores; no GPU, no real exposures. Real-data evidence comes from the two fitted states, the fit and uncertainty logs, and the products of the
19 Nov runs in `/home/claude/work/checkin/`.

Abbreviations: CG = conjugate gradients (the iterative solver); PCG = preconditioned conjugate gradients; FFT = fast Fourier transform;
SW / LW = the NIRCam short-wavelength (0.031 arcsec pixels) and long-wavelength (0.063 arcsec pixels) channels; S/N = signal-to-noise ratio;
"application" = one forward and one adjoint render of every training pixel (60 s on the GPU for a half fit).

## 1. Summary

* **Cause of the stall.** The preconditioner of the 18 Nov tool (mode `fourier`) is a Fourier model of the normal operator taken from the
  response to a unit coefficient at **one node, the box centre**, and scaled elsewhere by the coverage summed over all filters. On the SMACS
  box the centre sits (a) in the short-wavelength chip gap of half 0's F090W dithers (the log's `kappa` is 0.0, 0.0, 5e-05 for the three
  bluest coefficients against 1e-3 to 3e-3 for the others) and of half 1's F150W dithers (1e-4 against 3e-3), and (b) on bright extended
  emission (97th to 98th percentile of the field), where the noise floor depresses the weights. The Fourier model then has no information in the
  F090W (half 0) or F150W (half 1) direction of the spectrum, while that filter constrains the rest of the field: the preconditioned operator
  has eigenvalues spread over three to four decades and CG makes no progress. The rising residual (above 1) is what PCG does on such a
  spectrum: the residual's 2-norm is not monotone (the error in the operator's own norm is). The operator itself is symmetric to 1e-9 and
  repeatable bit for bit, and the preconditioner is symmetric and positive: no adjoint or determinism problem.
  On a CPU reproducer with F090W's gap on the box centre, the old solver shows the same `kappa` signature (0.0, 0.0, 8e-05) and the same
  behaviour (relative residual 0.86, 0.32, then 0.65 at 60 iterations; largest Lanczos estimate 4,250).
* **Why half 0's F090W sigma is 7.7 times half 1's.** Not a shortage of exposures (F090W: 12 and 11 training exposures; F150W 11 and 12; the
  others 11/11 and 3/3), not the validation carve-out (the same 13 validation exposures in both halves: same seed), not the band read-out
  (identical `Phi_band`). It is the same probe artefact: the products were made with `--unc-method circulant`, whose sigma is one number per
  band (the Fourier model at the centre) times a coverage pattern common to all bands. The half-0 / half-1 ratio of the F090W sigma is
  7.86 at every node once that pattern is divided out (`tests/solver_repro/results/halves_bright.json`, section 6). Half 1's F150W sigma is
  inflated the same way (0.0083 against 0.0018).
* **Fix (new default on the command line, `--unc-precond filter`).** One Fourier model per filter, each probed where that filter is well
  covered, combined per region of similar per-filter coverage (k-means in log coverage), with an exact per-node K x K scaling by each filter's
  own information (quadratic coverage), the regions blended by a smooth partition of unity (an additive Schwarz preconditioner with Fourier
  local solvers). Set-up cost unchanged (K + 1 applications). On the 56 x 56 reproducer: relative residual 1e-3 after 41 to 75
  iterations in all nine configurations (54 to 88 applications with the set-up), where the old one needs more than 100 in eight of them and
  stalls in two. Draws converge as fast. **Limit:** on smaller test fields (33 x 33 and 41 x 41 nodes) it does not converge either (section 5),
  so convergence on SMACS is expected, not shown. The old preconditioner is still the class default and selectable
  (`--unc-precond fourier`), bit-identical to the calibrated truth-problem runs (rerun checked).
* **Convergence reporting.** A solve whose final relative residual is above `--unc-accept-tol` (default 1e-3) is declared not converged:
  plainly in the log, `POLISHED = F`, `CONVERGD = F` in the header, SKY = the fit's own sky, the unconverged step written aside as SKY_UNCONV;
  unconverged draws are discarded. The log and header also give a Lanczos estimate of the preconditioned spectrum, and, per band, the
  change of the product over the last 10 iterations in sigma units (`POLCHG`, `POLCHGMX`). The relative residual alone does not say how good
  the product is (section 3d).
* **Bright sources.** The two halves differ on bright sources mainly by terms common to all sources of a band: a position difference of
  about 19 mas (the same in all bands), a flux-scale difference of 3 to 7 per cent, and in F444W a width difference. The source-to-source rest
  is a few per cent in flux and 2 to 6 mas in position. Fitting those three terms per source brings the source pulls from 7 to 8 down to 0.7
  (F200W) and 1.3 (F444W). This is the **response differing between the halves** (and a gauge of the fit that is not fixed: the kernel
  centroid against the sky), not noise the posterior misses. `diff-files` now measures it and writes SIGMA_CAL (section 7).

## 2. The reproducer

`tests/solver_repro/repro_field.py` builds a field from the package's own pieces (`synth.render_truth` for the data, `data.Exposure`,
`model.FieldOperator`, `train.warm_kernel` for a kernel start, `instruments.spectral_setup`), with the real-data features switchable:

| feature | reproducer | SMACS half |
|---|---|---|
| filters, spectral coefficients | F090W, F150W, F200W (SW) and F277W, F356W, F444W (LW), box curves; K = 12 on 0.70-5.30 um (the real `Phi_band` rows) | the same six, published curves, K = 12 |
| channels, pixels | SW 0.031 arcsec, 60 x 60 pixels; LW 0.063 arcsec, 30 x 30 | SW 0.031, LW 0.063 |
| exposures per filter | SW 3, LW 2 (15 in all) | SW 11-12, LW 3 (43) |
| chip gap | cross of 10 SW pixels through the detector centre (the 2 x 2 SW mosaic seen from a box at the module centre); dithers up to 0.45 arcsec | 2 x 2 SW mosaic, box across the cross |
| gap on the probe | option: one filter dithered by less than the gap (F090W) | F090W in half 0, partly F150W in half 1 |
| bright sources | four point sources, peak S/N 3000, 1000, 300, 100 per SW pixel | cluster galaxies and stars |
| noise floor | 9 per cent (SW), 4 per cent (LW) of the model signal, in the data and in the weights | 9.0 / 3.7 (half 0), 7.6 / 4.2 per cent (half 1) |
| Huber weights | at the fit's residual, delta 3 | the same |
| sky | grid, or item sky (1500 items) | item sky (60,000 items) |
| size | 56 x 56 nodes at 0.04 arcsec (37,632 coefficients), 90 background terms, 15,621 training pixels | 613 x 613 (4.5 million), 258, 3.4-3.6 million |
| tiling | 64-pixel tiles, 16,000-pixel batches (the CUDA defaults of the real runs) | the same |

Approximations of the reproducer: the "fit" is the package's warm start (an inverse-variance splat mapped to coefficients), not an optimiser's
result, so the right-hand side of the Gauss-Newton step is larger than on real data and 13.6 per cent of the pixels have Huber weight
below 1 (0.5 per cent on SMACS); the kernel is the warm start of the kernel network (120 steps), not a fitted response; there are far fewer
exposures and nodes; Q = 3 wavelength nodes per filter (5 on SMACS).

Scripts: `run_repro.py` (one configuration and one preconditioner: checks of the operator, the step's solve and a draw's solve with the
residual of every iteration, Lanczos estimates), `run_isolation.sh` (features one at a time), `top_modes.py` and `lanczos_modes.py` (where the
extreme eigenvectors of the preconditioned operator live), `halves_bright.py` (the real halves), `summarise.py` (tables). Results in
`tests/solver_repro/results/`, logs in `tests/solver_repro/logs/` (together 1.2 MB); `medium_field.py` for the fields of section 5.
Versions of the new preconditioner in the result files: `all_filter` and `topmode_filter` are the first, discarded version (section 3c);
`all_filter6`, `gapc_filter6`, `lanczos_filter`, `lanczos_filter_gapc` the second (saturating region features, linear coverage);
`all_filterq`, `gapc_filterq`, every `iso_*_filter`, `lanczos_filter_small*` the final one.

## 3. Measurements

### 3a. The operator and the old preconditioner are sound

With the real-data tiling (`run_repro.py`, every configuration): $|u^\mathsf{T}Fv - v^\mathsf{T}Fu| / (|u|\,|Fv|) = 1.3\times10^{-9}$ to
$4\times10^{-9}$; $F u$ computed twice is identical (difference 0.0: the operator renders with `jitter=False`, so `--render-jitter 1` in the
fit's configuration plays no part); $u^\mathsf{T}Fu > 0$. The preconditioner of either mode: symmetric to $10^{-9}$ to $6\times10^{-9}$,
$u^\mathsf{T}P^{-1}u > 0$. A non-symmetric or non-deterministic operator is therefore not the explanation.

### 3b. Residual curves (Gauss-Newton step; relative residual after 10, 30, 60, 100 iterations)

Feature isolation (`run_isolation.sh`; the old preconditioner is run at the new one's prior weight mu, so both solve the same system):

| configuration | preconditioner | 10 | 30 | 60 | 100 | iterations to 1e-3 | Lanczos range |
|---|---|---|---|---|---|---|---|
| six filters, K = 12, two channels; no gap, no bright sources, no floor, no Huber | fourier (old) | 1.9e-01 | 2.4e-02 | 7.4e-03 | 1.5e-03 | >100 | 0.031 to 107 |
| six filters, K = 12, two channels; no gap, no bright sources, no floor, no Huber | filter (new) | 3.1e-02 | 3.1e-03 | 1.3e-04 | 4.2e-06 | 44 | 0.019 to 9.64 |
| + chip gap (dithers fill it) | fourier (old) | 1.3e-01 | 3.0e-02 | 1.6e-02 | 8.8e-03 | >100 | 0.011 to 65.2 |
| + chip gap (dithers fill it) | filter (new) | 8.4e-02 | 9.1e-03 | 7.1e-04 | 6.2e-05 | 59 | 0.019 to 14.4 |
| + chip gap, F090W's gap on the box centre | fourier (old) | 8.4e-01 | 5.2e-01 | 3.8e-01 | 2.2e-01 | >100 | 0.838 to 5.01e+03 |
| + chip gap, F090W's gap on the box centre | filter (new) | 8.1e-02 | 9.2e-03 | 9.2e-04 | 4.8e-05 | 60 | 0.018 to 14.1 |
| six filters, no gap; bright sources (peak S/N 3000) and floor 9 / 4 per cent | fourier (old) | 3.2e-01 | 1.2e-01 | 6.3e-02 | 2.0e-02 | >100 | 0.013 to 130 |
| six filters, no gap; bright sources (peak S/N 3000) and floor 9 / 4 per cent | filter (new) | 1.7e-01 | 7.8e-03 | 4.4e-04 | 1.2e-05 | 49 | 0.019 to 11.3 |
| + Huber weights | fourier (old) | 3.7e-01 | 1.3e-01 | 2.4e-02 | 3.9e-03 | >100 | 0.050 to 138 |
| + Huber weights | filter (new) | 2.4e-01 | 1.3e-02 | 4.7e-04 | 1.5e-05 | 52 | 0.017 to 11.4 |
| ONE filter (K = 3), all other features | fourier (old) | 1.1e-01 | 1.3e-02 | 1.9e-03 | 8.2e-04 | 73 | 0.015 to 20 |
| ONE filter (K = 3), all other features | filter (new) | 2.5e-02 | 6.9e-03 | 4.2e-04 | 3.4e-06 | 41 | 0.006 to 3.74 |
| all features, one channel (all six filters at 0.031 arcsec) | fourier (old) | 5.4e-01 | 3.2e-01 | 1.3e-01 | 8.3e-02 | >100 | 0.314 to 1.64e+03 |
| all features, one channel (all six filters at 0.031 arcsec) | filter (new) | 8.4e-02 | 6.5e-03 | 8.2e-04 | 1.1e-04 | 55 | 0.006 to 8.26 |
| all features, item sky | fourier (old) | 3.0e-01 | 8.0e-02 | 1.3e-02 | 3.9e-03 | >100 | 0.038 to 108 |
| all features, item sky | filter (new) | 1.5e-01 | 1.3e-02 | 2.1e-03 | 3.9e-04 | 75 | 0.017 to 24.2 |
| all features (gap, bright, floor, Huber, two channels) | fourier (old) | 2.9e-01 | 5.4e-02 | 2.6e-02 | 5.2e-03 | >100 | 0.020 to 85.4 |
| all features (gap, bright, floor, Huber, two channels) | filter (new) | 1.8e-01 | 1.7e-02 | 2.0e-03 | 1.7e-04 | 71 | 0.017 to 18 |

The two full configurations (each mode at its own mu; also one posterior draw):

| field | preconditioner | 30 | 60 | 100 | iterations to 1e-3 | Lanczos range | draw at 60 / 100 |
|---|---|---|---|---|---|---|---|
| one filter, K = 3, no gap, no bright sources (the truth-problem regime) | fourier (old) | 6.7e-3 | 2.7e-4 | 5.0e-6 | 43 | 0.043 to 19.9 | 4.2e-4 / 6.6e-6 |
| all features | fourier (old) | 4.2e-2 | 2.2e-2 | 3.5e-3 | > 100 | 0.020 to 64 | 1.7e-2 / 1.3e-2 |
| all features | filter (new) | 1.7e-2 | 2.0e-3 | 1.7e-4 | 71 | 0.017 to 18 | 3.7e-3 / 4.8e-4 |
| all features, F090W's gap on the box centre | fourier (old) | 0.32 | **0.65** | 0.28 | never | 0.54 to **4,250** | 0.21 / 0.15 |
| all features, F090W's gap on the box centre | filter (new) | 1.8e-2 | 1.7e-3 | 2.0e-4 | 63 | 0.016 to 17 | 3.8e-3 / 3.7e-4 |

Reading. Old preconditioner: with one filter (K = 3) it converges even with every other feature on (73 iterations). Six filters alone
already keep it above 1e-3 at 100 iterations (largest Lanczos value 107). A filter whose gap lies on the probe node makes it stall, as on
SMACS half 0 (0.22 at 100 iterations, largest value 5,010). So does the one-channel variant, where the long-wavelength filters also get
chip gaps and one of them misses the centre (8.3e-2, 1,640). Bright sources with the noise floor (130), and gaps elsewhere (65), each slow it.
Huber weights and the item sky change little: same operator form, and Huber weights act like the floor. New preconditioner: 1e-3 in 41
to 75 iterations in every configuration, with largest Lanczos values of 3.7 to 24. The draws behave like the step (table below).

### 3c. Where the slow modes live

Power iteration and Ritz vectors (Lanczos vectors from the CG run) on the all-features field:

* old preconditioner: the largest mode (eigenvalue 64) lies in the interior of the field, in the long-wavelength coefficients 8 to 10. The
  Fourier model probed at the centre, scaled by the all-filter coverage, misstates each filter's share of the information wherever the
  filters' coverages differ;
* a first version of the new preconditioner, which mixed the Fourier inverse into weakly covered nodes in proportion to their data
  information, had its largest mode (120) at the edge of the data (96 per cent of the mode). The Fourier inverse amplifies high frequencies
  by up to about $\rho^2 = 100$ wherever the data are weak; that is what the regions in section 4 avoid;
* a second version with regions grouped by the saturating feature $d g/(d g + \mu)$ lumped all chip-gap nodes with the interior: largest mode
  23 (48 with the probe gap), entirely in the gaps, in the blue coefficients 0 to 2. Grouping in log coverage fixed it;
* the smallest modes of every version (0.02 to 0.18) sit at the bright sources (57 to 94 per cent of the vector). Their cores are a hole in
  the weights about three PSF widths across (the floor), where the linear coverage $\sum_i w_i\psi_{ij}$ overstates the information
  $\sum_i w_i\psi_{ij}^2$ by up to 50 times. The quadratic coverage and the log-coverage regions of section 4, introduced together, brought
  the residual at 60 iterations from 9.2e-3 to 2.0e-3 (largest Lanczos value 23 to 18; the smallest, 0.017, did not change: the bright cores
  still set it);
* background terms do not appear in the extreme modes (background share of the Ritz vectors below 1e-3). The coupling of the 6 background
  terms per exposure with the large-scale sky is not what limits the solve, so no coarse-grid (multigrid) correction is called for.

### 3d. What a residual means for the product

Error of the step after k iterations against the step after 100 iterations (relative residual 4e-6 to 1e-4), as band products at 2 sky pitches
(0.08 arcsec), in units of the tool's own sigma at that resolution, over the covered nodes (`product_error_vs_final` in the result files):

| field (new preconditioner) | iteration | relative residual | error rms / sigma (range over the 6 bands) | largest error / sigma |
|---|---|---|---|---|
| six filters alone | 30 | 3.1e-03 | 0.07 to 0.89 | 9.35 |
| six filters alone | 40 | 1.3e-03 | 0.04 to 0.37 | 3.37 |
| six filters alone | 60 | 1.3e-04 | 0.01 to 0.05 | 0.37 |
| six filters alone | 80 | 2.3e-05 | 0.00 to 0.01 | 0.10 |
| gap on the probe (F090W) | 40 | 3.3e-03 | 0.14 to 2.44 | 25.1 |
| gap on the probe (F090W) | 60 | 9.2e-04 | 0.05 to 0.61 | 6.26 |
| gap on the probe (F090W) | 80 | 2.2e-04 | 0.01 to 0.12 | 0.99 |
| bright sources, floor, Huber | 40 | 3.3e-03 | 0.11 to 1.06 | 10.0 |
| bright sources, floor, Huber | 60 | 4.7e-04 | 0.03 to 0.28 | 2.97 |
| bright sources, floor, Huber | 80 | 4.8e-05 | 0.00 to 0.05 | 0.41 |
| all features, one channel | 60 | 8.2e-04 | 0.21 to 0.70 | 4.59 |
| all features, one channel | 80 | 3.5e-04 | 0.10 to 0.20 | 1.68 |

**A relative residual of 1e-3 is not enough for a trustworthy product on this field.** It leaves errors of 0.05 to 0.7 sigma (rms) and
up to 3 to 6 sigma at single nodes. About 2e-4 is needed for errors below 1 sigma everywhere. The size of the error at a given relative
residual scales with the right-hand side, which is large here (the warm start is far from the optimum) and should be smaller on SMACS (the
optimiser's fit, which the step moved by about 1.6 sigma rms on the truth problem). So the relative residual alone cannot tell the user
how good the product is. Two consequences in the code:
* the solve still runs to `--unc-tol 1e-4` (or 100 iterations); `--unc-accept-tol 1e-3` is the least that counts as converged;
* every run now measures, per band, **how much the product changed over the last 10 iterations, in units of its own (Fourier) sigma**
  (log line and header `POLCHG` rms, `POLCHGMX` largest, `POLCHGIT` the iteration it is measured from). When the residual falls at least
  two-fold per 10 iterations (as in every converging run here), the remaining error is below that change. This is the number to read
  before trusting SKY. Checked in the one case of the table measured 10 iterations apart (six filters, iterations 30 and 40: error 0.89
  then 0.37 sigma rms in the worst band, so a change of at least 0.52 against a remaining 0.37); not checked more widely.

## 4. The fix: per-filter Fourier models, coverage regions, per-node K x K scaling (`precond_mode="filter"`)

Notation: $j$ a node, $k$ a spectral coefficient, $f$ a filter, $F = J^\mathsf{T}WJ + \mu R$ restricted to the sky block, $\psi_{ij}$ the
footprint of node $j$ on pixel $i$, $w_i$ the weights.

```latex
% per-filter coverage (one application for all filters: every batch holds one filter and renders with that filter's own coefficient grid)
c_f(j) = \sum_k \big(J_f^{\mathsf T} w_f\big)_{kj}

% probe of filter f: the node nearest the box centre with c_f within 10 per cent of its 90th percentile
p_f = \arg\min_j \{\, |x_j| : c_f(j) \ge 0.9\, Q_{0.9}(c_f) \,\}

% per-filter response to a unit coefficient k at p_f (K applications for all filters together), read at lag d; zero lag G_f(0)
G_f(d)_{k'k} = \big(J_f^{\mathsf T} W_f J_f\big)_{(k', p_f + d),\,(k, p_f)}

% quadratic coverage: W_f = the weights splatted at the pixels' sky positions; psi_f^2(l) ~ g_f(2l), g_f = 1^T G_f 1 (exact for a Gaussian footprint)
q_f(j) = \sum_i w_i \psi_{ij}^2 \approx (W_f * \psi_f^2)(j), \qquad d_f(j) = q_f(j) / q_f(p_f)

% local information (K x K) and the prior weight
M_j = \sum_f d_f(j)\, G_f(0) + \mu I, \qquad \mu = \tfrac{1}{\rho^2}\max_k \operatorname{median}_{j\ \text{covered}} \big(M_j - \mu I\big)_{kk}

% regions: k-means (8 clusters, k-means++ start, fixed seed) of t_f(j) = \ln(d_f(j) + \mu / g_f), g_f the largest eigenvalue of G_f(0);
% region r: geometric-mean coverage \bar d_{rf}; a_r(j) = the region's indicator smoothed over 1.5 nodes, normalised so that \sum_r a_r = 1
C_r(\omega) = \sum_f \bar d_{rf}\, \widehat{G}_f(\omega) + \mu I \quad (K\times K \text{ per frequency}), \qquad M_r = \sum_f \bar d_{rf}\, G_f(0) + \mu I

% the preconditioner (symmetric positive definite)
P^{-1} = M^{-1/2} \Big[\sum_r a_r^{1/2}\, M_r^{1/2}\, C_r^{-1}\, M_r^{1/2}\, a_r^{1/2}\Big] M^{-1/2}
```

Properties. Where one region's coverage is uniform, $M_j = M_r$ and $P = C_r$ (the Fourier model of that coverage, exact for a
translation-invariant operator). Where no data reach a node, $P = M_j = \mu I$ exactly. A spectral direction that a filter constrains elsewhere
but not at node $j$ (a chip gap) is in a region whose model does not contain it. Each $C_r^{-1}$ is one FFT solve; the background block keeps
its exact 6 x 6 inverses. Set-up: one forward-only pass (weights, plus the weight splat), one application (per-filter coverage), K
applications (probes): **13 + 1 at K = 12, as before**. Host work: one complex K x K eigendecomposition per frequency per region, one real one per
node, the k-means; seconds to a minute at N = 613. Device memory: $8K^2N^2$ bytes per region (433 MB at K = 12, N = 613; 3.5 GB for 8
regions), plus $4K^2N^2$ for $M^{-1/2}$ (216 MB) and the summed probe kernel (216 MB).

The cheap sigma (`--unc-method circulant`, `circulant_sigma`) in this mode is the same approximation used as a variance, $\sigma^2 = k^\mathsf{T}P^{-1}k$
with $M$ and $a_r$ taken as constant over the smoothing kernel $g_s$:

```latex
\sigma_s^2(x) \approx \sum_r a_r(x)\, u_{rx}^{\mathsf T} Q_{rs}\, u_{rx}, \qquad u_{rx} = M_r^{1/2} M_x^{-1/2}\phi, \qquad
Q_{rs} = N^{-2}\sum_\omega |\hat g_s(\omega)|^2\, \operatorname{Re} C_r^{-1}(\omega)
```

It follows each filter's own coverage (gap, edge, bright source), so it can no longer copy one node's value to the whole field.

Other changes in `fieldop/uncertainty.py`:
* `pcg` records its coefficients. `info` gains `ritz_min`, `ritz_max` (Lanczos estimates of the preconditioned spectrum) and `err_F_est`,
  `rel_err_F_est`. The latter is the error of the iterate in the operator's own norm, from the CG coefficients 10 iterations before the end
  (Hestenes-Stiefel / Strakos-Tichy estimate, a lower bound of the error 10 iterations earlier, which bounds the final error approximately from
  above). The iterates are unchanged.
* `solve_shared(..., accept_tol)`, `posterior_samples(..., accept_tol)`, `band_readout`, `write_product`: the convergence rule of section 1;
  header keys `CONVERGD`, `ACCTOL`, `PRECOND`, `RITZMIN`, `RITZMAX`, `POLERRF`, `POLCHG`, `POLCHGMX`, `POLCHGIT`; extension `SKY_UNCONV`.
  `pcg(..., keep_every)` keeps one earlier iterate (one extra vector of 18 MB at SMACS size) for the change diagnostic. With `accept_tol=None` (the library
  default) everything is as on 18 Nov.
* `SkyNormalOperator(..., precond_mode, mu, n_regions, region_smooth, coverage, n_probes)`; the class default stays `fourier`, so library callers
  (the truth calibration driver) are unchanged. Command line: `--unc-precond filter|fourier|jacobi` (default filter), `--unc-regions` (8),
  `--unc-probes` (1),
  `--unc-accept-tol` (1e-3; 0 = no check), `--unc-maxiter` default 100 (was 60).
* `calibration_difference` and the `diff-files` additions (section 7).

Why the new mode is the command-line default: it is the one that converges on every multi-filter configuration tested, including the SMACS
signature. The old one stays the class default because the truth-problem calibration (one filter, uniform coverage), where it converges, is
built on it. Bit for bit: with `precond_mode="fourier"` the batching, the operator, mu, the preconditioner and the iterates are the code of
18 Nov (the only edits on that path are bookkeeping). `test_old_mode_is_default_and_unchanged` checks the default against an explicit
`fourier` operator; the 20 tests of `tests/test_uncertainty.py` pass. `tests/check_uncertainty_rerun.py` was rerun on a copy of the results
(noise realisation 0, grid, both parts: 3 posterior draws and the Gauss-Newton step): draws and polished coefficients are bit-identical to the
saved ones (maximum absolute difference 0.0; mu 1.0178e-4 in both; 60 iterations each).

Tests: `tests/test_uncertainty_solver.py`, 8 new tests (symmetry and positivity of the new preconditioner and repeatability of the operator
with the real-data tiling; the probe-gap signature and the improvement; Lanczos estimates on a dense problem; the convergence reporting in
log, header and files, and the discarding of unconverged draws; the cheap sigma following each filter's coverage, and agreeing with the old one
on a uniform one-filter field; the calibration difference on known terms; the old mode as the class default, unchanged, and the
command-line defaults). With `tests/test_uncertainty.py`: 28 passed in 90 s on two loaded cores. In `tests/test_uncertainty.py` one line
changed: the command-line fixture caps solves at 12 iterations, so it now passes `--unc-accept-tol 0` (the 18 Nov behaviour, which those
tests are about).

## 5. Verified, and only expected

Verified on the CPU reproducer (56 x 56 nodes): the stall and its signature with the old preconditioner; convergence of the new one to
1e-3 in 41 to 75 iterations in nine configurations (63 and 71 on the two full-feature fields, with and without the probe gap), and of the
draws; symmetry and positivity; the cheap sigma follows each filter's coverage; the old path bit-identical to the truth calibration. Verified on the real products: the 7.86 ratio is one number per band; the bright-source discrepancy is common terms.

Expected but not verified (no GPU here): the iteration count on SMACS. The real field has more of everything that slows the solve:
4.5 million coefficients, 43 exposures with several dither groups per filter (more distinct coverage patterns than 8 regions can represent
exactly), many bright sources, LW channels with 3 exposures per filter. I expect the residual to reach 1e-3 within the 100 iterations on
each half and the 7.7-fold imbalance of F090W to disappear from SIGMA, but I cannot exclude that one half needs more iterations. In that
case the log and header will say NOT CONVERGED and SKY will be the fit's own sky.

**A limit found on the way.** On smaller fields the new preconditioner does not converge either. On 33 x 33 nodes with one or two LW
exposures per filter (`lanczos_modes.py --small 1`) the largest Lanczos estimate is 356 to 383. On 41 x 41 nodes with two or three LW
exposures per filter it is 709 to 734, with relative residual 4e-2 to 8e-2 at 100 iterations; the old preconditioner there: 0.43 at 60,
38,000. Changing Q from 2 to 3 did not help. The slow mode lies in coefficients 7 and 8 (where F277W and F356W overlap) at the highest
spatial frequencies, in the main covered region. Two mechanisms are involved; I could not separate them:
(a) with few undersampled exposures the high-frequency information depends on each node's sub-pixel phase relative to the pixels, and the
single-probe Fourier model reads it as about zero. Its high-frequency part changes sign between neighbouring probe nodes (-3.2e-4 to
+6.2e-5, against 1e-2 at low frequency), while other nodes carry hundreds of times mu;
(b) on these grids the probe response (radius about 15 nodes, kernel extent 0.6 arcsec) is comparable to the grid, so the periodic Fourier
model wraps around, and the data cover only 40 per cent of the sky grid (the rest is the margin and its edge band).
The 56 x 56 reproducer (data on 50 per cent of the grid) converges. The real field (613 x 613, response core about 25 nodes, a thin edge
band, 3 LW and 11 to 12 SW exposures per filter) is closer to it than to the small fields, but this is the main reason I cannot promise
convergence on SMACS. For (a), `--unc-probes n` averages up to n windowed probe responses per filter, placed in the same passes (no extra
cost; on the 613-node field 9 probes per filter fit with windows of +-40 nodes). It is implemented but not shown to help: the small fields
have no room for several probes of a filter. The default is one probe, the tested path.

## 6. Real data: the two halves

* `kappa` (centre node, per coefficient): half 0: 0.0, 0.0, 5e-05, 8e-4, 2.6e-3, ...; half 1: 9e-4, 3.9e-3, 3.0e-3, **3.9e-4, 1.1e-4, 7.3e-4**,
  1.1e-3, ... F090W is absent from the centre node in half 0, F150W weak there in half 1.
* Training exposures per filter (from `e_filt[gref]` in the states): half 0: 12, 11, 11, 3, 3, 3; half 1: 11, 12, 11, 3, 3, 3. The SW
  exposures of both halves fall on all four SW detectors (3 per detector, 2 for one), so the box straddles the cross of the 2 x 2 SW mosaic
  and the centre node is near the chip gap in every half; which filter misses it depends on the dithers kept.
* Validation: 13 exposures, the same in both halves (one dither group per filter, chosen with the fit's seed before the half split; both
  fits have seed 0). Band read-out: `Phi_band` identical in both states.
* SIGMA in the products: `SIGMETH = circulant`. Ratio half 0 / half 1 per node: F090W 6.7 to 9.1 (5 to 95 per cent), F200W and F444W 0.85
  to 1.15. Divided by the common pattern, the F090W ratio is 7.86 at every node: one number set at the centre.
* The box centre is at the 97th (F090W) to 98th (F200W) percentile of the field's brightness: the probe of the old preconditioner sat on
  bright extended emission, where the floor depresses the weights.
* The step did nothing: max |SKY - SKY_FIT| is 1.2e-3 (F090W) against a sky rms of 0.24. The products are the optimiser's skies.

## 7. The bright-source discrepancy

`halves_bright.py` and the new `calibration_difference` fit, per bright source (local maxima, window +-6 nodes), the half-difference with a
flux-scale term, a shift and a width term (section 4 of the code's docstring). On the 19 Nov products (40 sources per band, 9 in F090W
above S/N 30):

| band | common scale (a/b - 1) | common shift x, y [mas] | common width (V_a - V_b)/2 [arcsec^2] | spread: scale, shift x, y | source pull rms before -> after the fit | brightest-10% pull rms, raw -> aligned with SIGMA_CAL |
|---|---|---|---|---|---|---|
| F090W | -0.048 to -0.067 | -19, +3 to +6 | ~0 | 0.08 to 0.09, 2 to 9 mas | 1.1 -> 0.2 | 4.0 -> 1.1 |
| F200W | -0.038 to -0.040 | -17, +6 to +7 | +6.6e-5 | 0.04, 5 to 6 mas | 7.0 -> 0.7 | 11.8 -> 2.0 |
| F444W | +0.028 | -19 to -20, +7 | -4.9e-4 | 0.035, 3 to 6 mas | 8.2 -> 1.3 | 23.1 -> 2.1 |

(Ranges: the two estimators, `halves_bright.py` with all 40 peaks and `calibration_difference` with S/N >= 30 only.)

Reading:
1. **A position difference of 17 to 20 mas common to all bands** is not registration. The 37 exposures shared by both halves (validation
   and held out) are registered to the two skies within -3, +1 mas on average (matched by their data median; per filter -1.6 to -5.5 mas).
   The model has no gauge on the centroid of the learned kernels: a common shift of all kernels against the sky leaves every model pixel
   unchanged, and the two fits have landed at different points along it. Fixing it in the fit is the clean remedy (a model change, not made
   here: e.g. hold the T-weighted mean first moment of the reference class's kernels at zero, projected after each step like the other
   gauges, in `fieldop/model.py`). In the product it is an astrometric-frame uncertainty of about 19/2 mas, common to the whole image.
2. **A flux-scale difference of 3 to 7 per cent**, of one sign in SW and the other in LW, with a source-to-source spread of 3.5 to 9 per cent.
   That is the size of the fitted noise floor (4 to 9 per cent), which is consistent with the response being uncertain at that level on
   bright sources and the floor absorbing it in the fit. The floor is in the weights (so in sigma), but as independent pixel noise, which
   averages down over the many pixels of a source; a response error is the same in all of them. So the floor does not make the stated sigma
   honest on sources, even though it enters it.
3. **A width difference in F444W**: a resolution (PSF) difference between the halves' kernels. In a +-6-node window the width term is
   correlated with the scale term (on synthetic sources it came out 1.8 times the injected value), so read it as a sign and a size, not a
   measurement.
4. What remains after the three terms is about the stated sigma (pulls 0.2 to 1.3; F090W low because half 0's sigma is inflated). The
   faint-pixel pulls (0.52, 0.72, 1.41) come from the circulant sigma itself: F090W inflated (section 6), F444W probably understated by the
   single-probe model of an undersampled 3-exposure channel (section 5).

**Not decidable without a new GPU run:** how much of the bright-source difference is the optimiser's fit-to-fit scatter. The products are the
optimiser's skies (the step did not move them), and on the truth problem the seed-to-seed scatter was 0.14 to 0.19 sigma, but SMACS's item
sky with 60,000 items may behave differently. **The run that decides it:** the two half fits through the new solver (section 8) to a
converged step, then `diff-files` on the two polished SKY products. If the common terms and SIGMA_CAL are unchanged, the discrepancy is
the response (and the centroid gauge). If the polished halves agree much better than the fits' own skies, it was the optimiser.

How the product carries it (implemented in `diff-files`, mode halves): header keys `ZPDIFF` (flux scale), `FRAMEDX`, `FRAMEDY` (mas),
`WIDTHDIF`, `CALSCALE`, `CALSHX`, `CALSHY` (source-to-source spreads), `CALNSRC`, and `PULLBRT0` / `PULLBRT1` (brightest-10 per cent pull rms
before and after); extensions H_ALIGNED (the half-difference with the common terms removed) and SIGMA_CAL:

```latex
\sigma_{\rm cal}^2(x) = (\epsilon_s I)^2 + (\epsilon_x\,\partial_x I)^2 + (\epsilon_y\,\partial_y I)^2 + (\epsilon_v\,\nabla^2 I)^2,
\qquad \epsilon = \tfrac12\,\text{(robust source-to-source spread of the half difference's terms)}
```

with $I$ the mean of the two halves; $\epsilon$ uses the halves convention ($h = (I_a - I_b)/2$ is the error of one full fit). The total
per-pixel error to quote is $\sqrt{\sigma^2 + \sigma_{\rm cal}^2}$; the common terms are image-wide (frame, zero point, resolution) and
are quoted once, not added per pixel. Not done: writing SIGMA_CAL into the full fit's product file (a `--unc-cal-from D.fits` option would
copy it in), and a model of how the response error scales from half to full data (assumed here like noise, a factor 1/2 on the half
difference).

## 8. The next real-data run (one half, six bands, one solve; 10.75 GB GPU slice)

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1 python -m fieldop.uncertainty \
    --unc-config OUT/metrics.json --device cuda \
    --unc-band F090W,F150W,F200W,F277W,F356W,F444W --unc-out OUT/unc_filter/sky_{band}_uncertainty.fits \
    --unc-precond filter --unc-regions 8 --unc-method circulant --unc-polish 1 \
    --unc-maxiter 100 --unc-tol 1e-4 --unc-accept-tol 1e-3 --unc-resolution 0.08 \
    --unc-batch-px 16000 --unc-max-minutes 150
```

Time at 60 s per application: set-up 1 forward-only pass (40 s) + 13 applications (13 min) + host work (about 1 min); the step's
right-hand side and at most 100 iterations (101 min; less if 1e-4 is reached); read-outs and files (about 5 min): **about 2 h, budget 150
minutes** (the budget stops the step cleanly if needed, and an unfinished step is reported as not converged). One job per half, as on
19 Nov. Then `python -m fieldop.uncertainty --unc-cmd diff-files --unc-file-a H0/sky_F200W_uncertainty.fits --unc-file-b
H1/sky_F200W_uncertainty.fits --unc-out D_F200W.fits` per band. That gives the half-difference pulls with the new sigma, and the
calibration terms of the polished skies (section 7).

What to read in the log: the line `preconditioner 'filter': probe nodes ...; coverage of the box centre relative to each filter's probe`
(expect F090W near 0 in half 0), the Gauss-Newton line with the relative residual, the Lanczos range and the estimated error, and
`Gauss-Newton step converged` or `did NOT CONVERGE`. Memory: about 4 GB for the preconditioner on top of the render. If the set-up runs out of
memory, `--unc-regions 4` (1.7 GB) is the first thing to try, then `--unc-batch-px 8000`.

Posterior draws (the calibrated sigma, rather than the Fourier one) cost up to 101 applications each with `--unc-maxiter 100`. Once the
step is known to converge, two draws per half (`--unc-method draws --unc-samples 2`, about 3.4 h more) are enough to check the cheap sigma
(`SIGRATIO`); eight draws is a 15 to 16 h job.

## 9. Approximations (all of them) and what I would do next

The approximations of the estimator itself are those of UNCERTAINTY_2026-11-18.md section 7 (linearisation with the response held,
Huber weights at the fit, the background gauge, the weak prior); nothing here changes them.

Approximations of the new preconditioner (they affect the iteration count, never the answer of a converged solve): one probe per filter,
read as a translation-invariant model; the quadratic coverage $W_f * \psi_f^2$ with $\psi_f^2(l) \approx g_f(2l)$ (exact for a Gaussian
footprint) and the weights splatted at the mean of the Q wavelength nodes' positions; 8 coverage regions with geometric-mean coverage per
filter; the smoothing of the region indicators (1.5 nodes); the probe chosen at the 90th percentile of linear coverage. The cheap sigma of the
new mode additionally takes $M$ and $a_r$ as constant over the smoothing kernel. Prior weight: mu is now the median of the quadratic-coverage
diagonal (1.5 to 1.8 times smaller than the old centre-probe value on the reproducer), so sigma at scales below the supported resolution
(which depends on the prior) is not comparable between modes. The acceptance tolerance (1e-3) is a relative residual, not an error in
sigma units (section 3d). The error estimate in the log is an estimate, not a bound. The change diagnostic (`POLCHG`) bounds the remaining
error only for a solve converging at least two-fold per 10 iterations, and it uses the Fourier sigma. The calibration term (section 7) is
first order in the shift, scale and width differences, fitted in +-6-node windows around local maxima above S/N 30. It assumes the response
error of a full fit is half the half-difference, and it takes the common terms as image-wide.

Next, in order:
1. If the real run is slow and the log's Lanczos estimate is large: rerun with `--unc-probes 9` (section 5). Test it first on a CPU
   reproducer large enough to hold several probes per filter (about 100 x 100 nodes; about 20 minutes per solve on two cores).
   Zero-padding the Fourier solves (to remove the wrap-around of section 5 (b)) costs four times the memory and is the next option.
2. Recycle the Lanczos vectors of the step's solve to deflate the draws' solves (the operator is the same).
3. The kernel-centroid gauge in the fit (section 7), then the decisive run of section 7.
4. A sigma-calibrated stopping rule (the product error in sigma units, section 3d) instead of a relative residual.
5. Recalibrate the polished product on the truth problem with the new preconditioner. It should not change: same estimator, converged.
