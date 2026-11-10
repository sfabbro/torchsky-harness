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
    "F29 cross-fitting (--xfit 2, optional): two sky views by dither-group fold, the operator reads the other fold's sky (detached), views averaged "
    "after --xfit-frac",
    "F31 batched rendering (--batch-render): the patches of a step are rendered together per channel with ONE window size per level (the largest of "
    "the batch) and one random level-0 phase per batch; identical to the per-patch render to 5e-7 (values) / 3e-5 (gradients)",
    "F30 instruments: the loader maps JWST NIRCam detectors; HST _flt/_flc loading through this package is not implemented yet",
]
