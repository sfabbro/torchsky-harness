# Independent check of the learned kernel and flux scale, fit fieldop_smacs_fix_200_s0 (13 Nov 2026)

Fit: one continuous sky plus a learned kernel field, JWST NIRCam SMACS J0723, six filters, 123 exposures (86 training), 20.48" scoring box, sky grid 613 x 613 at 0.04" (24.5"). The state (`e2_learned/dl/fieldop_state.pt`) loads with the package version that made it (`/home/claude/arc/fieldop_latest/fieldop`; all 12 source hashes equal those in `fieldop.log`), strict load of every kernel, code and throughput weight. Only the model *definitions* (network classes, the level quadrature that defines the package's normalisation, the spectral basis) were taken from the package; every measurement is new code (`kc_common.py`, `q1_kernel.py`, `q2_q4_render.py`, `q2_node_response.py`, `q3_flux.py`, `bg_terms.py`; driver `kernel_check.py`). The earlier agent's measuring code was read only to learn how the state is laid out.

Conventions read from the code: `k(u) = exp(l(R_e^T u, h)) T(|u|) / Z`, T = C1 taper from 8.5" to 10"; Z = sum over the nested level quadrature (pitch 0.005" short-wavelength / 0.01" long-wavelength, doubling to 0.64", radius 10"), so the kernel sums to one **over the whole 10"-radius disc** on that quadrature (approximation F7). The model is `sum_q w_q exp(gamma_q) [k * s_q] + background`, input level `cal`, so the sky is a surface brightness in **MJy/sr** (bandpass.py: a flat f_nu sky gives the cal surface brightness at gamma = 0; the throughput gauge F17 makes mean gamma = 0; I measured gamma = -0.002 to +0.001 for the exposures used). Blur gauge recomputed: core moment of the sharpest class 5.40e-4 arcsec^2 vs floor 5.33e-4, so the common dilation is 1 (inactive). Learned variability: kernel detector and wavelength terms pruned (tau 1e-6), per-exposure kernel code tau 0.059.

## 1. Kernel shape (results/q1_kernel.json, figs/q1_kernel_profiles.png)

Evaluated on a polar grid (0.0001" radial steps to 0.06", then 0.25 % log steps to 10"; 720 angles) and on a 0.0005" Cartesian grid for the widths; detector with most exposures of the filter (NRCB1 / NRCBLONG), centre and two corners of the training footprint, mean code.

| | F090W | F150W | F200W | F277W | F356W | F444W |
|---|---|---|---|---|---|---|
| FWHM, half-maximum area (arcsec) | 0.0227 | 0.0214 | 0.0213 | 0.0229 | 0.0234 | 0.0233 |
| FWHM, x / y cuts through the peak (arcsec) | 0.021/0.025 | 0.020/0.024 | 0.026/0.019 | 0.021/0.025 | 0.021/0.025 | 0.021/0.025 |
| same in detector pixels (area) | 0.73 | 0.69 | 0.69 | 0.36 | 0.37 | 0.37 |
| published optical FWHM | 0.033 | 0.050 | 0.066 | 0.092 | 0.116 | 0.145 |
| encircled at 0.05 / 0.1 / 0.2 / 0.5 / 1 / 2 / 10" | .21 .25 .27 .29 .33 .41 1.00 | .29 .32 .34 .36 .39 .46 1.00 | .31 .34 .37 .39 .42 .49 1.00 | .23 .27 .31 .33 .38 .45 .99 | .19 .23 .26 .29 .33 .41 .99 | .18 .22 .25 .27 .32 .39 .99 |
| beyond +-32 px (square, detector frame) | 0.657 | 0.599 | 0.572 | 0.531 | 0.569 | 0.587 |
| beyond +-128 px | 0.406 | 0.367 | 0.347 | 0.059 | 0.060 | 0.062 |
| continuous integral / package quadrature sum | 0.9990 | 0.9986 | 0.9987 | 0.9896 | 0.9925 | 0.9927 |
| decomposition: spike / shoulder / 0.2-2" / 2-10" | .15 .12 .14 .59 | .22 .12 .12 .54 | .24 .13 .12 .51 | .16 .15 .15 .54 | .13 .14 .15 .59 | .12 .13 .15 .60 |
| flat floor 5-8.5" (arcsec^-2) and its share | 1.9e-3, 0.51 | 1.7e-3, 0.46 | 1.6e-3, 0.44 | 1.7e-3, 0.46 | 1.9e-3, 0.50 | 1.9e-3, 0.52 |

Encircled fractions are about the kernel origin with the package normalisation. The three detector positions give identical numbers to four digits (detector term pruned); five per-exposure codes per filter move the FWHM by 0.0209-0.0241" and the fraction beyond +-32 px by at most 0.02. Negative values: none, fraction 0 (the density is an exponential times a non-negative taper; it reaches 0 only at the 10" edge). Peak 0.004-0.009" from the origin.

**Verdict: the earlier claim holds.** FWHM 0.021-0.023" (area) or 0.019-0.026" (cuts) in all six filters, against 0.033-0.145" published; 53-66 % beyond +-32 px and 35-41 % beyond +-128 px (short wavelength). The earlier report's 0.023-0.024" is the mean of the two cuts, mine the area-equivalent diameter: same kernel, different width measure, both right. Its fractions agree with mine to 0.004.

**Decomposition.** A parametric spike + Gaussian + Moffat fit failed (log residual 1 dex), because the profile is not of that form. It has four parts with the **same radii in all six filters** (no lambda/D scaling at all): (i) a spike of FWHM about 0.022" (0.55 of the 0.04" sky pitch), 12-24 % of the integral; (ii) a flat shoulder of 1.4-2.3 arcsec^-2 out to 0.1", edge at 0.15-0.2" (log slope -4), 12-15 %; this is not diffraction-like, since it does not grow with wavelength; (iii) a halo falling as r^-1 from 0.3" to 5", 12-15 % between 0.2" and 2"; (iv) a flat floor from about 5" to the taper at 8.5", density 1.6-1.9e-3 arcsec^-2, which alone carries 44-52 % of the unit integral. So yes, the "core" is a spike at the sky-sampling scale, but what sits under it is a filter-independent plateau and a 10"-wide pedestal, not a diffraction pattern.

## 2. Stars through the whole model (results/q2_point_source.json, q2_single_node_response.json, figs/q2_source_profiles.png)

No compact source inside 8.5" of the centre is a star (the most compact have mosaic FWHM 0.12-0.27"). Four stars lie at 9-10.3" (diffraction spikes visible in the fitted sky, figs/q4_wing_map_F200W.png; mosaic 0.15"/0.5" flux ratio 0.7-0.8 at F200W): (-9.0,-5.0), (10.28,-2.84), (-9.0,4.72), (9.68,6.76) arcsec (X west, Y north); a fifth, (9.0,-0.4), has a neighbour and is listed in the json but left out of the medians. Method: source-only sky (nodes within 1", minus the annulus 1.0-1.5" median), tent-interpolated to 0.005", correlated by FFT with the kernel of one training exposure per filter (its rotation and code), kernel table to +-2.7"; mosaic bicubic-upsampled, same annulus. Medians over the four stars:

| | F090W | F150W | F200W | F277W | F356W | F444W |
|---|---|---|---|---|---|---|
| FWHM sky alone (arcsec) | 0.051 | 0.052 | 0.059 | 0.083 | 0.127 | 0.132 |
| FWHM prediction (arcsec / detector px) | 0.068 / 2.2 | 0.067 / 2.2 | 0.072 / 2.3 | 0.099 / 1.6 | 0.149 / 2.4 | 0.153 / 2.4 |
| FWHM in the mosaic (drizzled) | 0.063 | 0.066 | 0.077 | 0.135 | 0.147 | 0.169 |
| published, pixel-convolved (approx.) | 0.039 | 0.054 | 0.069 | 0.101 | 0.124 | 0.151 |
| prediction, fraction of the star's total flux within 0.1 / 0.2 / 0.3 / 0.5 / 1 / 2" | .17 .23 .25 .28 .33 .40 | .25 .31 .33 .36 .40 .46 | .26 .33 .36 .39 .43 .49 | .19 .28 .31 .34 .38 .45 | .13 .22 .25 .28 .32 .40 | .12 .20 .23 .26 .30 .38 |
| prediction / mosaic flux within 0.3" / 1" | 0.88 / 1.03 | 0.83 / 0.88 | 0.90 / 0.86 | 0.90 / 0.93 | 0.88 / 0.97 | 1.01 / 1.06 |
| sky / mosaic flux within 0.3" / 1" | 3.1 / 3.1 | 2.4 / 2.2 | 2.4 / 2.0 | 2.8 / 2.5 | 3.4 / 3.0 | 4.1 / 3.5 |

Sharpest object the sky grid can hold (one tent node) through the kernel: FWHM 0.049-0.051" in every filter; fraction of the total within 0.3" 0.25-0.38, within 2" 0.38-0.49.

**Verdict.** In data space the core is plausible. The predicted star matches the mosaic in flux within 0.3-1" to 0.83-1.06, and in FWHM to within 9 % in five filters (F277W: 0.099" against 0.135" in the drizzled mosaic). Against the published pixel-convolved widths it is within 4 % for F200W, F277W and F444W, and 20-24 % broader for F150W and F356W. For F090W it is 0.068" against 0.039", 1.7 times broader, limited by the 0.04" sky grid: the sharpest possible sky object already gives 0.050". The sky carries the band-dependent blur, as the gauge allows. But the predicted star holds only 23-36 % of its total flux within 0.3" and 38-49 % within 2", against roughly 80 % within 0.15-0.3" for real NIRCam point sources (and, from memory of the published encircled-energy curves, above about 90 % within 2"; not re-checked here). The other 51-62 % goes into a halo of about 2e-3 arcsec^-2 per unit flux spread over 10". **The predicted star is plausible only as a profile shape within about 1"; as a flux distribution it is not physical, and the sky star is 2-4 times brighter than the real one.**

## 3. Flux scale against the official mosaics (results/q3_flux.json, figs/q3_flux_ratio.png)

Both images are in MJy/sr. The mosaic is a north-up gnomonic projection about the box centre, the same frame as the sky grid, so no conversion is applied beyond surface brightness times area (mosaic PIXAR_A2 0.000961 / 0.003969 arcsec^2, sky cell 0.0016 arcsec^2). Residual offset sky - mosaic: +5 to +9 mas west, +23 to +32 mas north, measured from 15 centroids and removed. Same aperture code for both (7 x 7 sub-cells, annulus 3-sigma-clipped median: 0.6-1.0" for r = 0.3", 2.5-3.5" for 1", 3-4" for 2").

| median sky / mosaic | F090W | F150W | F200W | F277W | F356W | F444W |
|---|---|---|---|---|---|---|
| (a) 20 compact sources, r = 0.3" (scatter, 1.48 MAD) | 3.65 (0.37) | 2.92 (0.35) | 2.74 (0.11) | 3.16 (0.27) | 3.79 (0.31) | 4.10 (0.42) |
| (b) 8 bright galaxies, r = 0.3 / 1 / 2" | 3.32 / 2.07 / 2.31 | 2.14 / 1.72 / 1.74 | 1.98 / 1.92 / 2.00 | 2.62 / 1.86 / 2.13 | 3.46 / 2.09 / 2.28 | 1.88 / 2.18 / 2.67 |
| (c) box total, each minus its blank level | 2.85 | 2.04 | 1.91 | 2.00 | 2.38 | 2.68 |
| (c) box total, raw | 3.48 | 2.12 | 1.83 | 2.11 | 2.79 | 3.44 |
| 1 / (kernel fraction within 2") | 2.45 | 2.17 | 2.05 | 2.22 | 2.43 | 2.54 |
| blank level sky / mosaic (MJy/sr) | 0.118 / 0.028 | 0.099 / 0.044 | 0.088 / 0.053 | 0.093 / 0.039 | 0.099 / 0.026 | 0.106 / 0.019 |

The compact sample is compact galaxies with mosaic fluxes 0.2-55 microjansky (the stars are near the edge; see question 2: 2.0-3.5 within 1"). **There is a common factor of about 2 (1.7-2.9) in every filter.** It is larger at 0.3" (2.7-4.1) because the sky is sharper than the mosaic, and at 2" and in the box totals it tracks 1/(kernel fraction within 2"), to 0.1-0.4. The sky also has a diffuse level 2-6 times the mosaic's blank level. This is the signature asked about: the flux deficit of the kernel core is made up by inflating the sky, and the kernel's far wings put the excess into a smooth level.

## 4. Far wings against background (results/q4_wings.json, figs/q4_wing_map_F200W.png)

**From the code, nothing pins the split.** The kernel's unit integral is imposed over the whole 10" disc (F7), and the throughput gauge (F17) fixes the gain against the published curves, but nothing fixes how much of the unit integral lies at radii comparable to the 20" footprint. There, a kernel wing acting on the sky is a smooth, almost constant level. The pedestal gauge (F18b) fixes only the mean of the per-exposure *constant* term to the exposure medians. The per-detector 8 x 8 pattern has zero mean over the *whole* detector, but only the 3 x 3 or so cells under the footprint see data, so it is effectively a free smooth background there. Each exposure also has free quadratic Legendre terms. The only other weak anchors are the soft sky positivity penalty and the zero sky outside the 24.5" domain.

Measured with the full kernel against the kernel truncated at +-32 detector pixels (square, detector frame, not renormalised), over the data region (|X|,|Y| <= 11.24"), 4 training exposures per filter, wavelength nodes and throughput included:

| MJy/sr (range over 4 exposures) | F090W | F150W | F200W | F277W | F356W | F444W |
|---|---|---|---|---|---|---|
| kernel integral inside +-32 px | 0.34 | 0.39-0.41 | 0.42-0.43 | 0.47 | 0.42 | 0.40 |
| sky term, full kernel (mean) | 0.185-0.186 | 0.231-0.232 | 0.245-0.247 | 0.248-0.249 | 0.215-0.216 | 0.210-0.211 |
| carried by kernel beyond +-32 px (mean; rms over region) | 0.108-0.109 (0.035) | 0.121-0.124 (0.052) | 0.124-0.127 (0.059) | 0.112-0.113 (0.048) | 0.109 (0.039) | 0.110-0.111 (0.036) |
| fitted background constant | 0.209-0.236 | 0.229-0.262 | 0.186-0.230 | 0.138-0.141 | 0.136-0.139 | 0.259 |
| exposure data median | 0.203-0.236 | 0.224-0.263 | 0.185-0.233 | 0.140-0.141 | 0.135-0.138 | 0.259-0.260 |
| 8 x 8 pattern, mean under the footprint | -0.072 to -0.105 | -0.072 to -0.105 | -0.072 to -0.105 | -0.127 | -0.127 | -0.127 |
| total background (constant + Legendre + pattern) | 0.087-0.121 | 0.106-0.139 | 0.083-0.100 | -0.014 to -0.008 | -0.010 to -0.006 | 0.109-0.110 |

The kernel parts beyond +-32 px carry 0.11-0.13 MJy/sr, which is 44-59 % of the predicted sky signal and 0.42-0.81 of the fitted background constant. The constant itself sits at the data median, as the gauge demands. The detector pattern then cancels almost all of the wing level (-0.07 to -0.13 MJy/sr), and for the long-wavelength exposures the total background is slightly negative. The wing level is not constant over the region (5-95 % range 0.05-0.24 MJy/sr, peaked on the cluster core), so the trade is made jointly with the quadratic and pattern terms and the sky's own diffuse level.

## 5. Is the fitted sky a flux-calibrated product?

No. Its fluxes are too high by a factor of about 2 (1.7-2.9 depending on filter and aperture, up to 4 for compact sources within 0.3"). The factor matches the fraction of the kernel's unit integral that lies beyond about 2", and the sky carries a diffuse level 2-6 times the mosaic's. Shapes and positions are usable, and the model's predictions in data space (sky through kernel plus background) match the data in the core. The flux *scale* and anything diffuse are not, because they are set by an unconstrained trade between far kernel wings and the detector background terms, which the per-exposure constant gauge does not reach (the 8 x 8 pattern and Legendre terms absorb the level). **Single most useful pin:** define the flux normalisation over the scales the data can separate from a background, i.e. make the kernel's unit integral hold within a fixed radius well inside the footprint (for example 1-2", a fixed-aperture flux convention as in PSF photometry with an aperture correction), with whatever lies beyond counted as not belonging to the source. The far wings then cannot change the sky's flux scale, however they trade with the background. A weaker alternative is to make the pedestal gauge act on the *total* background (constant + pattern + Legendre) over the footprint rather than on the constant alone; on its own that still leaves the sky's diffuse level free to take the trade. Not implemented.

## Checks of my own code

- Renderer against the package's own `render_fft` (results/crosscheck_renderer.json): F090W agrees to 0.2 % of the peak (rms 1e-4 of peak) at 1521 points around two sources. For F444W mine is 3-5 % lower at the peak and 2.7 % lower in the mean. That matches the package's own quadrature error for the long-wavelength kernels (level-0 pitch 0.01" against a 0.023" core; continuous integral / quadrature sum 0.990-0.993). It is irrelevant to the factor of 2.
- Sky reading checked against the log: my 5th percentiles of the band images equal the logged `sky_band_p05` to all printed digits (coefficients = stored parameter x preconditioner).
- Matching peaks between the package's held-out prediction files and my render failed (no WCS available: `asdf`/`gwcs` are not installed, and I installed nothing). It was replaced by the renderer check above; `crosscheck_pred.py` is kept, but its output is not used.

## Approximations and shortcuts

1. Kernel at the pivot code (wavelength term pruned to 1e-6); detector term included at the footprint lattice (also 1e-6); per-exposure code included where an exposure is named (questions 2, 4, and the spread in question 1).
2. Nominal pixel scales 0.031" and 0.063" (the run used header medians, not saved) for "detector pixels", the +-32 px truncation and the detector lattices.
3. Throughput rebuilt from the state (mean gamma term and per-exposure gain; detector term, tau 1e-6, dropped).
4. Question 2: detector lattice at nominal scale with the exposure's mean rotation, no distortion or affine terms; source-only sky within 1" (cuts the diffraction spikes beyond 1"); kernel beyond 2.7" not rendered (it only moves flux beyond 2"); the mosaic profile includes neighbours and resampling blur; published FWHM from the task's list, pixel-convolved values by a Gaussian quadrature sum with the pixel box (sigma = p/sqrt 12); the published encircled-energy figures are approximate.
5. Question 3: the mosaic is taken as the flux reference (pipeline calibration and its own background handling); annulus backgrounds in a crowded cluster field; the sky treated as an image of 0.04" cells holding the node values; the compact sample is compact galaxies; the box-total "blank levels" are each image's source-masked clipped median.
6. Question 4: the averaging region is the whole data region, not each exposure's own cut; footprints are recovered from the reference lattice (training bounding box per detector); stripes are left out (zero mean by gauge); the kernel is cell-integrated on the 0.04" grid (affects local values, not means); the truncation is not renormalised.
7. Question 1 decomposition is non-parametric, with boundaries chosen by eye from the profile (0.02", 0.2", 2", floor 5-8").

## Not done

No comparison against the `_cal` exposures themselves (only the mosaic and the package's renderer); no instrument point-spread-function models; question 2 uses one exposure per filter; the held-out prediction files could not be placed on the sky (no WCS tools); no attempt to implement or test the proposed pin.

Files: `kernel_check.py` (driver), `kc_common.py`, `q1_kernel.py`, `q3_flux.py`, `q2_q4_render.py`, `q2_node_response.py`, `bg_terms.py`, `crosscheck_render.py`, `crosscheck_pred.py`, `fetch_mosaics.py`, `make_figs.py`; `results/*.json`, `results/*.npz`; `figs/q1_kernel_profiles.png`, `figs/q2_source_profiles.png`, `figs/q3_flux_ratio.png`, `figs/q4_wing_map_F200W.png`; mosaic cut-outs in `dl/`.
