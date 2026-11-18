"""fieldop: a general field operator mapping a continuous sky to individual JWST/HST exposures (Phase 2 of the ablation plan).

    d_e(xi) = N_e[ (1/Q) sum_q alpha_e(xi) exp(gamma_e(xi, lambda_q)) int d^2u k_e(u; xi, lambda_q) s(Omega_e(xi, lambda_q) + u, lambda_q) ] + B_e(xi) + n_e(xi)

Catalogue-free: no source detection, identification or catalogue anywhere; physics enters only as inputs (filter curves, wavelengths,
aperture diameter and pixel scale in the kernel warm start, header WCS, pipeline ERR) or as a warm start. See DESIGN.md for the equations
as implemented, the choices, the cost and what is not done. Every approximation is listed below (F1...), printed at the start of every run
and written to metrics.json.
"""
__version__ = "0.1.0"

APPROXIMATIONS = [
    "F1 sky is static: s(Omega, lambda, t) = s(Omega, lambda) (no variable sources); spectrum = 12 RBF bumps uniform in log lambda on [0.70, 5.30] um "
    "(partition of unity, v40 basis without the learned correction): spectral resolution ~ 0.18 dex, extrapolated by the end bumps outside the domain",
    "F2 sky in space = tent (bilinear) interpolant of a coefficient grid at --sky-pitch over the box + 2 x --sky-margin; ZERO outside: sources outside the "
    "sky domain and their wings are absent; data are used only inside the box + 2 x --data-margin (< the sky margin)",
    "F3 sky optimisation: fixed diagonal preconditioner 1 + f min(S/N_warm, cap) (5x5 dilated) on the coefficients; soft positivity penalty --sky-neg on the "
    "band images (in units of the band noise); TV off by default; warm start = inverse-variance bilinear splat of the training pixels at header-WCS "
    "(+ initial registration) positions, mapped to coefficients by v40's ridge spectral_init (PSF-blurred start)",
    "F4 lambda integral: Q equal-weight nodes at the mid-quantiles of T(lambda)/lambda over the filter support (T >= 1e-3 max) of the PUBLISHED curve; "
    "exp(gamma) is evaluated at the nodes only; error vs Q measured in tests/test_fieldop_render.py",
    "F5 kernel frame: v = R_e^T u with R_e the polar rotation (with parity) of the exposure's MEAN header Jacobian; the variation of the local WCS "
    "rotation/skew across the detector is left to the learned spatial code; the level grids are axis-aligned on the sky",
    "F6 kernel field: FiLM MLP (--k-layers x --k-hidden) on a smooth log-radius encoding (core r0 = half the finest render pitch, radial Fourier "
    "frequencies 1..--enc-nrad over [r0, extent]) and angular harmonics m <= --enc-mmax (12: features narrower than ~15 deg in azimuth, e.g. thin "
    "diffraction spikes far out, are smoothed); log density clamped at 30",
    "F7 unit integral of k on each renderer's OWN quadrature (FFT levels or per-pixel points); amplitude 'free' skips it (integral learned)",
    "F8 code h additive in code space; hdet / hep centred and rms-normalised over reference sets (training detectors x 3x3 lattice; training epochs), "
    "recomputed every step; a code-space tau is not identifiable on its own (the field can rescale its sensitivity), so the output-space effective "
    "variability is reported next to it; the chromatic term tau_lam (hbar(lambda) - hbar(lambda_pivot)) is not normalised",
    "F9 kernel truncated by a C1 taper between 0.85 and 1.0 x --kernel-extent; nothing beyond",
    "F10 FFT kernel quadrature = nested square level grids (pitch p_r f^l, (2 n_h + 1)^2 nodes) with a C1 partition of unity in r (hand-over between "
    "a R_l and R_l); measured against the per-pixel quadrature (tests)",
    "F11 training renders shift the level-0 grid by a random sub-pitch phase (--render-jitter) so that the field is learned as a continuous function; "
    "prediction uses phase 0",
    "F12 levels with pitch >= 2 sky pixels read the box average of the COEFFICIENTS (not of the tent function); finer levels read the tent sky exactly",
    "F13 correlated level images are read at the pixel positions by prefiltered cubic B-splines (periodic prefilter, valid inside the n_h + 3 margin) "
    "or Keys bicubic (--interp keys); error measured",
    "F14 spatial variation inside a region: K evaluated at region nodes (lattice of --region-px per detector), correlated images blended with bilinear "
    "weights (= kernel interpolated bilinearly in xi); error vs region size measured",
    "F15 pixel area alpha (--input-level rate) = |det((I + M_e)(J_wcs + dD/dxi))| with J_wcs from central differences of the per-pixel header positions "
    "(one-sided at the cut edges) and dD/dxi by 1-px central differences; the per-filter photometric conversion of _rate data is NOT an input yet, "
    "so in rate mode the sky has per-filter units; predictions of rate-trained models are in DN/s (not scorable against _cal)",
    "F16 throughput log g: filter-level MLP of (lambda, filter input, instrument) + detector field (MLP of 2 Fourier frequencies of xi: no structure "
    "finer than ~1/4 detector) + epoch field, both normalised over their references (detector field: over the (detector, lambda node) pairs "
    "observed in training), + per-exposure gain random effect u_e ~ N(0, 1)",
    "F17 throughput gauge (exact): the T-weighted mean of gamma over the reference set (training exposures; --gauge-ref-inst) evaluated on a 3x3 lattice "
    "per exposure is subtracted at every lambda node: the mean g of the reference set equals the published curves AT THE NODES; wavelengths not "
    "covered by the reference set get no condition",
    "F18 gauges besides throughput: (a) COMMON BLUR, hard reparametrisation: every kernel is evaluated at v / s with s = sqrt(max(1, floor / M_ref)) "
    "(two fixed-point passes), M_ref = core second moment (Gaussian window of width 2 sqrt(floor)) of the mean kernel of the sharpest kernel class "
    "(filter with the smallest generic width (0.42 lambda/D)^2 + p^2/12) averaged over its lambda nodes, floor = --blur-floor x pitch^2 / 3 (one sky "
    "tent): no kernel class can be sharper than the sky grid supports; above the floor the blur is free (a common dilation is a reparametrisation, "
    "not the exact convolution gauge); (b) PEDESTAL: per filter, sum over training exposures of (background constant - exposure median) = 0, "
    "projected after every nuisance step; NIRCam stripes projected to zero mean over the rows / columns each exposure uses; (c) common shift: the "
    "LEARNED translations of the training exposures are projected to zero mean after every step (the initial registration shift0 is kept); "
    "all logged (gauge_report)",
    "F19 geometry: Omega = (I + M_e)(X_wcs + shift0_e + D) + t_e about the BOX centre; D = tau_Ds Dst(det, xi, lambda) + tau_Dd Ddr(det, xi, t) + "
    "tau_De sum_j zD_ej B_j(xi) (6 fixed orthonormal quadratic fields, zD ~ N(0, I)); Dst / Ddr normalised over their references; shift0 from "
    "register_xcorr.json (LMC) or 0 (SMACS: the per-exposure translation is learned; the v31 coarse-shift code is not ported)",
    "F20 sampling: patches of --patch-arcsec drawn uniformly plus --is-frac around local maxima of the current model band images above --is-snr "
    "(patch-level strata, exact 1/(N pi) reweighting: unbiased for the mean loss per pixel)",
    "F21 variability scales: FUNCTION axes (kernel space / time / lambda, throughput space / time, static and drift distortion) by the held-out "
    "(validation) likelihood as a sequential evidence rule: one dither group per filter is carved out of training (--n-val-groups), its nuisances are "
    "fitted on one half of interleaved 4x4-pixel blocks, the gradient of the loss on the other half w.r.t. log tau is accumulated (--tau-accum), its "
    "exponential moving mean / sd (--tau-beta) give a significance z, and log tau moves by -lr_tau clip(z, -3, 3) while the axis is KEPT, i.e. while a "
    "likelihood-ratio check (held-out loss on the same validation patches with the axis switched off minus on, paired differences accumulated with "
    "forgetting --ab-beta) is significant at > --tau-occam-z sd, and shrinks geometrically otherwise (first order, no hypergradient; consecutive gradients are correlated, so z is "
    "optimistic; code-space kernel scales capped at --tau-code-max); RANDOM-EFFECT axes (kernel z_e, gain u_e, distortion zD_e) by a damped parameter-expanded EM step (tau <- tau x rms of the MAP "
    "effects; no posterior-variance term, so weakly constrained scales are biased low)",
    "F22 detector: optional gated non-linearity (<= 20 per cent); background = quadratic Legendre per exposure + per-detector 8x8 bilinear pattern "
    "(zero mean) + NIRCam row/column stripes (L2 prior); no persistence, brighter-fatter, IPC or saturation model beyond the DQ mask",
    "F23 noise sigma^2 = (sigma_pipe exp(h[det, filter]))^2 + (f_chan S)^2, h bounded +-3; Huber (delta 3) on z plus log sigma (Huber NLL not "
    "normalised); random-effect priors are divided by --prior-pixels (their weight against the per-pixel loss is a choice)",
    "F24 optimisation: Adam; sky by --inner-sky steps per operator step (not an exact solve); warm-up + cosine on progress = max(step, time "
    "fraction); operator gradient clipping",
    "F25 held-out exposures: ONLY per-exposure nuisances: first translation + background (--held-reg-steps, larger steps), then background, affine, "
    "gain u, codes z and zD, stripes by Adam on the exposure's own pixels (--held-steps), operator, sky and tau frozen; full-cut FFT render (phase 0), stripes re-solved in closed form (3 alternations); pixels within "
    "--pred-margin of the sky edge are not claimed (model_valid False)",
    "F26 float32 throughout (positions relative to the box centre: ~1e-6 arcsec resolution); kernel activation checkpointing optional",
    "F27 data: SCI/ERR/DQ of _cal; valid = DQ == 0, finite, ERR > 0, inside the data region; header (SIP) WCS through torchsky; epoch t normalised "
    "over the field's time span; epoch axes off when the training span < --epoch-min-days (SMACS: one day, so off)",
    "F28 kernel warm start (physics only as a start): l fitted to log[0.85 N(sigma) + 0.15 Moffat(2 sigma, beta 1.5)], sigma^2 = (0.42 lambda/D)^2 + "
    "p^2/12, on random continuous offsets, with every variability term at zero; no PSF library, no STPSF",
    "F29 (removed 18 Nov 2026): sky cross-fitting by dither-group fold (--xfit); the sky keeps its leading view axis of length 1 so saved states keep their shapes",
    "F31 batched rendering (--batch-render): the patches of a step are rendered together per channel with ONE window size per level (the largest of "
    "the batch) and one random level-0 phase per batch; identical to the per-patch render to 5e-7 (values) / 3e-5 (gradients)",
    "F30 instruments: the loader maps JWST NIRCam detectors; HST _flt/_flc loading through this package is not implemented yet",
    "F32 several sky patches (--tiles): one operator, one sky per tile (each in the tangent plane of its own box centre), one row of per-exposure "
    "nuisances per exposure shared by its cuts in every tile; the per-exposure affine acts about the exposure's own reference point r_e in "
    "the FIELD frame (centroid of its valid pixels over all its cuts): Omega = x + M_e (x + c_k - r_e) + t_e with c_k the gnomonic offset of "
    "tile k's centre from the centroid of the selected boxes (about the field centroid instead, M_e had a 60-130 arcsec lever arm on LMC and "
    "each Adam step of M_e moved positions by ~10 mas); the north directions of two tile frames differ by dRA sin(dec) (0.13 deg across the "
    "LMC tiles), ignored in M_e (x + c_k - r_e) and t_e (effect < 0.2 mas for |M_e| < 1e-3)",
    "F33 per-exposure metadata with several tiles (kernel frame R_e, sigbar, the reference median bkg0 of the pedestal gauge, pixel scale) come "
    "from the exposure's cut with most valid pixels; R_e of a cut in another tile differs by the frame rotation above (< 0.13 deg); the sky warm "
    "start of each tile subtracts that cut's own median",
    "F34 tile schedule: visits of --visit-steps steps on one tile drawn with p_k proportional to (training pixels)^--tile-weight-power, so the "
    "objective is sum_k p_k L_k (per-pixel mean loss of tile k), the pooled pixel mean only for power 1; rows of exposures absent from a "
    "visit are not stepped (as F24); validation (tau) steps use the visited tile's validation cuts (no step on tiles without any)",
    "F35 gauges with several skies: the common-shift projection acts per connected component of the (exposure, tile) training graph, the "
    "pedestal projection per (filter, component of that filter's graph); the throughput gauge stays global, so a sky scale of one component "
    "against the gains of its exposures is held only by the N(0, 1) prior on u_e (exposures shared across tiles tie the components together)",
    "F36 data residency: pixel tensors of the cuts on the host (pinned), copied to the device per tile (LRU cache of --gpu-resident-gb); skies "
    "and their Adam states stay on the device (with --sky-swap they are moved to the host too when their tile is inactive); held-out nuisances are fitted cycling --held-rounds times over the tiles (shared rows), "
    "--held-steps scaled by held-out cuts / --held-ref-cuts per tile; each held-out cut is predicted with its tile's sky and the cuts are "
    "merged per exposure by the nearest tile centre (as stage3_blot); cuts are taken at REGISTERED positions (header + shift0)",
    "F37 reference lattice and caps (12 Nov 2026, after the SMACS 200-min regression): the detector fields (kernel code, throughput, static / "
    "drift distortion) are normalised, and the throughput gauge is evaluated, on a 3x3 lattice at the cell centres of each detector's "
    "TRAINING footprint (bounding box of its valid training pixels; --ref-footprint 1), so each function-axis scale is the rms variation "
    "where the data are (on the fixed +-0.8 lattice, far outside the 20-arcsec SMACS footprint, the throughput scale ran away to 17.5 while "
    "the field varied by up to 1.9 in log g across the footprint); throughput scales are capped at --tau-g-max (rms log g) and static / drift "
    "distortion at --tau-d-max, because an axis kept by the likelihood-ratio check follows the validation gradient with no Occam pressure; "
    "the footprint is a bounding box (a box straddling a detector corner yields a lattice point without data)",
    "F38 item sky (--sky-form items, 13 Nov 2026; default grid unchanged): s = sum_n [sum_k c_nk phi_k] G(. - x_n; Sigma_n); "
    "c_nk = exp(r_nk) > 0; widths sqrt(floor^2 + "
    "exp(2 l)) >= --item-floor (0 = 0.4 x sky pitch), capped at --item-max-width (0 = 16 x pitch). The items are RASTERISED on the SkyGrid "
    "coefficient grid at --sky-pitch and read by the unchanged renderers through the tent interpolant (so the sky is still a tent grid; the "
    "change is the parametrisation). Cell integral approximated by adding p^2/12 to both principal variances (the box replaced by an isotropic "
    "Gaussian of equal variance) and point-sampling, on a patch of +-4 effective widths, renormalised to exact flux (error before "
    "renormalisation 3 per cent at the floor, < 1e-3 above 1.5 x floor; per-node error vs the exact cell mean <= 2.2 per cent of the peak "
    "at the floor; sub-node centroid error <= 0.025 pitch); an item's rendered second moment is therefore at least 2 floor^2 + p^2/2 "
    "(0.82 p^2 at the default floor) against p^2/3 for one grid node; the blur gauge (F18) is unchanged and still set by one tent",
    "F39 item seeding and updates: data-blind square lattices, level l (0 finest) holding a share 4^-l of --item-count (3 levels: 76/19/5 "
    "per cent), spacing 2 half / m, width half the spacing (>= floor), random angle, fine-level remainder on a Halton set; coefficients = level "
    "share x item area x max(warm-start coefficients smoothed at the item width, 1e-3 x their median peak); reseeded when the registration "
    "pre-pass rebuilds the warm start; Adam groups: positions --item-lr-pos x pitch, log widths, angles, log coefficients (their own steps, "
    "untuned, from the image-only experiment); the rasterised grid is cached while no sky parameter requires a gradient",
    "F40 item pyramid rasteriser (--item-raster pyramid, 13 Nov 2026; default direct, bit-identical to F38): Gaussian semigroup. Level l has "
    "pitch p_l = 2^l p (level 0 = the sky grid); level l is carried to level l-1 by a fixed separable upsample-and-blur (stride-2 transposed "
    "convolution with a discretely normalised Gaussian of width s_l = p_l, taps +-5 s_l, flux conserved), so a level-l node arrives on the sky "
    "grid as a Gaussian of variance S_l^2 = sum_{j<=l} s_j^2 (sampling error of each step ~ exp(-2 pi^2 sigma_eff^2 / p_l^2), sigma_eff^2 >= "
    "p_l^2 / 2 for the first step). Item n (target covariance T_n = Sigma_n + p^2/12 I, as F38) goes to the coarsest level l whose remaining "
    "covariance T_n - S_l^2 I has its smaller principal width >= p_l (detached widths: a discrete switch; switch widths 2.81, 5.99, 12.2, 24.4, "
    "48.9 ... sky pitches), and is point-sampled there with that covariance on +-4 larger remaining widths, normalised to exact flux; items "
    "below the level-1 threshold use the F38 path on the sky grid (with patch buckets extended from +-128 to +-512 nodes). Coarse grids extend "
    "beyond the sky grid by the cascade's reach, so items at or beyond the edge are right inside the grid; flux leaving the sky grid is dropped "
    "(as F38). Measured against an exact-flux point-sampled Gaussian of covariance T_n (larger width 0.5 to 200 nodes, axis ratios 1, 2, 5): "
    "worst node error 2.1e-4 of the peak (just above the level-1 switch; direct path 2.8e-4 from its +-4-width truncation, in its range), "
    "jump across a switch <= 2.3e-4 of the peak, flux exact to 2e-15. The number of levels follows from --item-max-width (allowed up to the "
    "field size). Cost per item depends on the item's axis ratio, not its size: with square patches an item costs (8 x larger remaining width "
    "/ p_l)^2 nodes at the level set by its SMALLER width, so narrow elongated items stay on the sky grid at (larger width / p)^2; "
    "--item-patch sheared samples instead a box of +-4 widths in the item's decorrelated coordinates (rows along y, each row's columns "
    "centred on the conditional mean of x), ~ 64 sqrt(det) / p_l^2 nodes, used per item where it has 1.5x fewer nodes than the square, on "
    "every level including the sky grid (then level-0 items differ from F38 at the truncation level, <= 2.8e-4 of the peak). The patch layout "
    "(rounded centres, and with shearing each row's rounded column centre) is a discrete function of the detached parameters, so the raster "
    "jumps by up to ~exp(-8) of an item's peak where it changes (as F38 does when an item crosses a half node)",
    "F41 flux convention of the kernel (--kernel-norm-radius R, 14 Nov 2026; default 0 = unit integral over the whole support, unchanged): "
    "with R > 0 the kernel's unit integral is taken inside R only (window 1 to 0.85 R, C1 fall to 0 at R), the kernel beyond R being free in "
    "amplitude. Reason: with the unit integral over the full +-10 arcsec support, a flat far pedestal of the kernel acting on the sky is "
    "absorbed by the background terms (per-detector pattern, quadratic terms), which nothing pins; the fitted SMACS sky then carries about "
    "2 x the official mosaic's flux (1.7 to 4 depending on aperture) with 53 to 66 per cent of the kernel integral beyond +-32 pixels. With "
    "R the sky's flux scale is set by the light inside R: fluxes are 'inside R' fluxes, low against total-flux calibration by the true "
    "encircled fraction beyond R (NIRCam: about 1 to 3 per cent at 2 arcsec); the split between far wings and background stays unpinned "
    "but no longer moves the sky",
    "F42 data-quality policy (--dq-ignore-bits B, 14 Nov 2026; default 0 = F27 unchanged): pixels whose only flags are among the bits B stay "
    "valid. Reason: the SMACS _cal files re-fetched from MAST on 13 Nov (CAL_VER 3.0.0, CRDS context 1584) flag the cores of the brightest "
    "sources SATURATED (bit 2: some groups saturated, slope fitted from the others) where the protocol's held-out masks, built from the "
    "earlier files with DQ == 0, call them valid; with strict DQ == 0 the fit never sees those cores and under-predicts them (0.84 of the "
    "data at reference S/N > 200 in one F277W exposure; chi2/N there 415 against 53). B = 2 keeps them; their slopes come from fewer groups, "
    "so they carry more non-linearity error than ERR states",
    "F43 per-pixel gain (--pixel-gain detector | detector-filter, 16 Nov 2026; default off, bit-identical): d = N_e[G(p) S] + G(p) x "
    "(per-exposure quadratic background) + per-detector pattern + NIRCam stripes, G = exp(logg) per detector pixel p (one map per detector, "
    "or per detector and filter); the sky signal and the sky background pass through G, the pattern and the stripes (additive detector "
    "terms: residual dark, amplifier offsets, 1/f noise) do not; the non-linearity acts on G S (background excluded, as before); maps are "
    "allocated over the bounding box of the detector pixels valid in at least one TRAINING cut; pixels outside it or never observed in "
    "training (held-out pixels beyond the training footprint) get G = 1, i.e. no flat correction; one map per detector assumes a "
    "filter-independent pixel-to-pixel flat (official flats of one detector correlate at 0.98-0.99 between SW filters but only 0.85-0.90 "
    "between LW filters); with several tiles, cuts of one exposure that overlap on the detector are counted twice in the count / "
    "information maps (diagnostics only)",
    "F44 gain gauge (exact, by projection after every gain step, no penalty, no prior): over the observed pixels of each map, logg is made "
    "orthogonal (uniform weights) to the monomials of total degree <= --pixel-gain-poly (default 2: mean, tilts, curvatures) of the pixel "
    "coordinates scaled to [-1, 1] over the box; this removes the mean (degenerate with the throughput and the per-exposure gains u_e) and "
    "the lowest-order smooth part (degenerate with the smooth throughput detector field and the quadratic backgrounds); smooth structure "
    "above degree 2 stays in the map, partly degenerate with the throughput detector field where that axis is on; no smoothness prior on "
    "the pixel-to-pixel part. Diagnostic stored with the map: information I_p = sum over training cuts of max(d^2 / sig^2 - 1, 0) (the "
    "Cramer-Rao bound 1 / sqrt(I_p) of logg if all else were known and m ~ d)",
    "F45 gain optimisation: its own Adam (lr --lr-gain, default 3e-3 in log G, same warm-up + cosine factor as the operator, no gradient "
    "clipping), stepped in the joint step only, from progress --pixel-gain-start (0.1) on, so that the sky exists first; frozen in the "
    "held-out fit and in the validation steps. The step was chosen on the synthetic field (tests/synth_gain.py: 3e-4, 1e-3, 3e-3, 1e-2 "
    "scanned, about 130 gradient visits per pixel in 200 steps); a SMACS pixel gets only ~30-35 visits in a 200-minute fit, so the map may "
    "be under-converged there. Adam's momentum moves pixels between visits (no row masking as for per-exposure nuisances)",
    "F46 _rate input (--input-level rate): SCI / ERR / DQ from the _rate file (DN/s; ERR without the flat-field variance; DQ without the "
    "NO_FLAT_FIELD bit the _cal adds), header WCS and PIXAR_A2 from the SCI header of the matching _cal file (--rate-wcs cal, default): the "
    "_rate header has only the linear TAN WCS of the pointing (0.07 arcsec median, 0.24 arcsec maximum from the _cal SIP WCS over NRCB1); "
    "the WCS is a geometric step of calwebb_image2, kept, while flat field and photometric conversion are not applied; predictions are "
    "written in DN/s with meta units = 'rate' (score_paired.py --level rate converts calibrated-unit predictions with C = rate / cal)",
]
