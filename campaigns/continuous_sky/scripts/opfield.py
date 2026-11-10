#!/usr/bin/env python3
"""opfield.py: one learned field operator for everything that is not sky (main track of claude/forward-operator-spec.md, 6 Nov 2026).

For exposure i, kept pixel p (target-grid position x0 = header WCS, detector offsets (dx, dy) in [-1, 1]), quadrature node q of the band
(nodes and weights from load_bandpasses, spec_q = 6):

    L_i(p)   = exp(a) sum_q w_q sum_k W_k(q) s(x0 + mu + J Lc (v_k - c), lambda_q)         W_k(q) = Psi_k(q) dv_k / sum_k Psi_k(q) dv_k
    mu_pred  = L + g(L, ctx) + B_i(p) + stripes_i(p)          g gated: slope 1 at faint signal (learned_instrument.py)
    sigma(p) = sigma_pipe(p) exp(h(ctx)),   outlier probability pi(ctx)   (pi is a function of the MODEL context only, never of the data)
    loss     = 2 mean_batch[ -log( (1-pi) N(d; mu_pred, sigma) + pi U ) ]      U = 1 / (200 sigma_pipe)  (uniform, width 200 sigma_pipe)

(mu, Lc, a, Psi) come from ONE hypernetwork (shared trunk, separate heads) of the context c = (Fourier features of the detector
position, detector and instrument embeddings, log lambda of the band, epoch, exposure time, roll angle / parity / log pixel scale from
estimate_jacobian, readout direction of NIRCam, per-exposure code z_i of dimension 8, L2-penalised). Physics enters only as these inputs.
J is the exposure's linear detector-pixel Jacobian (estimate_jacobian): v lives in DETECTOR pixels and Lc (log scales, angle, shear,
zero-initialised = identity) is a learned correction on top of it. This is a spatial-transformer-style internal frame: a DESIGN CHOICE
(translation equivariance, v in detector pixels), not physics. mu is a registration/distortion field in target pixels, Psi the shape
(optics + pixel response + charge spreading), exp(a) the transmission x gain.

Psi (log-density on a fixed canonical stencil) = base[channel] + sum_k coef_k(h, lambda_q) bank_k   (K_bank = 32 learned basis vectors),
density = softplus(.), mass_k = Psi_k dv_k normalised to unit sum; the stencil is re-centred on the mass-weighted centroid (band
weighted, core window), so Psi has zero core centroid and position lives only in mu. The sky is read at every stencil node by bicubic interpolation.
Sky = 12 RBF coefficients per pixel over log lambda (like v36), initialised from the drizzle coadd (spectral_init), soft positivity
(lam_neg) and TV (0.02) as v36. Coarse to fine (BARF-like): the sky is read through a Gaussian whose sigma anneals from
--c2f-start-arcsec (0.4) to 0 over the first --c2f-frac (0.3) of the budget (also during the held-out nuisance fit).

GAUGES (declared; small penalties, values logged): common shift (mean mu over the training exposures, pixel-count weighted, = 0);
flux (mean a over the training exposures of --ref-instrument = 0); common blur (--gauge-blur, default 1e-4, relative: mean second moment
of the core of the effective kernels, i.e. the sky ends at the resolution of the sharpest kernel); background vs smooth sky: NONE beyond
the low capacity of B (quadratic per exposure + quartic per detector).

EVERY APPROXIMATION / SHORTCUT (also written to metrics.json["approximations"] and to the log):
  S1  Stencil. Deterministic (no Monte Carlo). Dense core: square lattice of pitch 0.25 detector px kept inside a DISK of radius 3 px
      (not the +-3 px square); wings: log-polar rings (ratio 1.25 in radius, 24 azimuths, alternate rings offset by half a step) from 3 px
      out to --wing-arcsec (3 arcsec) converted to detector px PER CHANNEL (the stencil of a channel is a prefix of the common one).
      The wing pitch is therefore NOT a constant 1 px (that would be ~29000 samples for NIRCam SW) but about 0.26 r (0.9 px at r = 3.5 px,
      growing with r); the azimuthal resolution is 15 degrees, so diffraction spikes / ghosts narrower than that are not resolved.
      Beyond the channel radius there is no kernel. Quadrature weights dv are cell areas (core: pitch^2, wings: annulus area / 24); the core
      lattice does not tile the disk exactly; both errors are absorbed by the learned Psi (the learned object is the measure Psi dv).
  S2  Wing samples read a Gaussian-pre-smoothed sky (not a box), at 6 DISCRETE levels (sigma 0, 0.7, 1.4, 2.8, 5.6, 11.2 sky px), the level
      nearest to 0.2887 x (sample pitch in sky px) (no smoothing if the pitch is below 1 sky px). The pre-smoothing is also applied by
      the coarse-to-fine blur (sigma_tot^2 = sigma_c2f^2 + sigma_level^2), computed on the coefficient images (linear, commutes with the
      spectral map), replicate-padded, truncated at 4 sigma.
  S3  Bicubic interpolation = torch grid_sample (Keys a = -0.75): exact for constants and linear fields only; relative error on a
      Gaussian of sigma 1 sky px read at random sub-pixel positions is measured in tests/test_opfield.py (a few percent of the peak).
      Outside the sky grid the sky is replicated from the border (padding_mode = border), not zero.
  S4  Zero centroid by re-centring the stencil on the mass-weighted centroid of the CORE window (r <= 3 detector px, renormalised),
      averaged over the band's nodes with weights w_q: the band-integrated CORE has zero centroid, the first moment of the wings is NOT
      constrained (it is dominated by a few far, faint nodes: with the full-stencil centroid a 0.2 percent asymmetric wing mass moved
      the core by 0.04 px in a unit test, and wings that grew moved it by ~10 px in a first smoke run). The kernel at an individual
      lambda_q may keep a (chromatic) centroid. mu does not depend on lambda_q (no chromatic registration).
  S5  Lc is a correction to the exposure's single global linear Jacobian J_i (least squares over the loaded pixels); WCS non-linearity is
      left to mu and Lc. |det Lambda| is not an extra factor: Psi is normalised to unit sum, so a flat sky gives exp(a) x sky.
  S6  Initial Psi: a generic round Gaussian of sigma 1.0 detector px plus a floor 1e-7 per px^2 for every channel (generic start of
      v36 E2-data, NOT the nominal PSF width, no PSF library, no STPSF). Priors: L2 on the bank, on (base - base_init) and on the Lambda
      correction (--lc-l2), no smoothness prior on Psi (the wings in the poorly constrained annuli can drift; the L2 limits it).
  S7  The readout direction is the unit vector of the detector x axis on the sky (J[:, 0]) for the instruments that use stripes (NIRCam),
      0 otherwise: it duplicates the roll angle for NIRCam (no header readout/amplifier flags); epoch from EXPSTART, exposure time from
      EFFEXPTM (JWST) / EXPTIME (HST).
  S8  Nonlinearity g: gated residual (gate L^2/(L^2+gs^2), gs = 10 per-exposure median pipeline sigma); its neighbourhood context is
      the centre / mean / max of the band-averaged sky samples within 1.5 detector px, not a learned neighbourhood network.
  S9  Noise: sigma = sigma_pipe exp(h), h = 3 tanh(net/3). Outlier mixture with a uniform density of width 200 sigma_pipe centred on the
      model (not normalised for the truncation); pi = 0.2 sigmoid(net), initial 0.005. The outlier POSTERIOR (used for the CR report)
      does use the data; pi does not.
  S10 The loss is 2 x mean mixture NLL of a pixel minibatch with bright-pixel oversampling (--bright-frac of the batch drawn from pixels
      above S/N 5 of the SWarp-like coadd): the objective is not an unbiased estimate of the all-pixel NLL (deliberate, as
      learned_instrument.py). Weights of the regularisers are those of v36 (TV 0.02, lam_neg) applied to the band images of the sky.
  S11 Gauges are evaluated on a 3 x 3 lattice per training exposure (not on all pixels). The blur gauge uses the second moment of the
      CORE of the kernel (r <= 3 detector px, renormalised), not the full second moment (wing-dominated, r^2 weighted). 1e-4 is a
      tie-breaker: in a short budget the gauge may not be reached (the measured second moments are logged).
  S12 Held-out protocol (T1): theta and the sky are frozen; z_i (init 0) and the quadratic background are fitted by Adam on minibatches of
      the learned-noise mixture NLL with the coarse-to-fine blur, then the quadratic background is re-solved by exact weighted least
      squares with the PIPELINE ivar, and NIRCam stripes are fitted by the v36 closed form (3 alternations of ivar-weighted row / column
      means). Unlike v36 there is NO affine registration refit and NO separate exact gain: registration and gain enter only through z_i.
  S13 Scores use the pipeline ivar exactly as heldout_chi2 (chi2, pseudo-Huber at 3 sigma, src mask = bilinear sample of the SWarp-like
      S/N > 5 coadd mask restricted to the original box with --halo-px, aperture-photometry pulls from the same isolated-source
      catalogue); the aperture lookup uses the registered position x0 + mu (v36 uses its refit affine); the data are the same exposures.
  S14 HST coadds for the sky initialisation and masks are the approximate tent-splat coadds of v36 (splat_coadd). The pixel data ring
      (pad 24 px - halo) is kept as in v36. HST data are in surface brightness per nominal pixel (no area_ratio correction), as v36.
  S15 --coarse-shift (v36 grid scan) is wired in but default OFF and untested here.
  S16 Pixel response is not a separate factor: the 0.25 px core lattice is the quadrature of the pixel box inside the learned Psi.
NEW FLAGS (all default to the original behaviour; S24-S32 below): --eval-only/--init-state/--args-from/--dump-heldout-pixels (diagnostics plumbing),
--base-per, --shape-head-per, --wings/--cart-steps, --base-table/--mod-rank/--base-init-from (more instrument-specific operator).
--lambda-gauge jacobian (Lambda == I, M = J; S30), --lam-in-band (within-band wavelength features in the shape MLP; S31); always on: metrics.json["kernel_diagnostics"] (S32).
WINGS / SHOULDERS (8 Nov, all default OFF, S33-S37): --psi-param log (Psi as a log-density), --importance-sampling R1,R2,... (+ --is-frac/--is-refresh/--is-snr/--is-max-peaks:
annulus-stratified, exactly reweighted sampling around the bright peaks of the current model), --two-scale R_CORE_ARCSEC (+ --wing-mod/--wing-group-deg/--wing-chunk/--wing-lr-mult/--wing-l2:
per-pixel stencil truncated at R_CORE, the wings = one free log table per (detector, band) in the detector frame, FFT-convolved with the band image per roll group).
Outputs (--out): metrics.json, opfield.log, cubes/opfield_sky.fits (+ coefficients), kernels/ (effective kernels), mu_fields.npz,
opfield_state.pt, opfield_best.pt, holdout_selection.json.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smacs_four_v36 as V  # noqa: E402  (data loading, coadds, selection, bootstrap: reused by import)

APPROXIMATIONS = [
    "S1 stencil: core disk r<=3 detector px at pitch 0.25; wings log-polar (ratio 1.25, 24 azimuths) to --wing-arcsec per channel; wing pitch ~0.26 r (not a constant 1 px); azimuthal resolution 15 deg (spikes/ghosts unresolved); no kernel beyond the radius; quadrature weights are cell areas, errors absorbed in Psi",
    "S2 wing samples read a Gaussian-pre-smoothed sky at 6 discrete levels (sigma 0..11.2 sky px), nearest to 0.2887 x pitch (not a box); c2f blur combined in quadrature; replicate padded, 4 sigma truncation",
    "S3 bicubic interpolation (Keys a=-0.75, torch grid_sample): exact for linear fields only; border padding outside the sky grid",
    "S4 zero centroid by re-centring on the band-weighted centroid of the CORE window (r<=3 px); wing first moment unconstrained; per-lambda_q centroids may remain; mu independent of lambda_q",
    "S5 Lambda = J_i Lc with one global linear Jacobian per exposure; unit-sum Psi, no extra |det Lambda| factor",
    "S6 generic initial Psi (round Gaussian sigma 1.0 detector px + floor, every channel); L2 priors on bank and on base - base_init; no smoothness prior",
    "S7 readout direction = unit vector of detector x on the sky for stripe (NIRCam) instruments; duplicates roll; epoch from EXPSTART, exposure time EFFEXPTM/EXPTIME",
    "S8 g gated residual with core-sample statistics as neighbourhood context (not a learned neighbourhood)",
    "S9 noise sigma_pipe exp(h), h bounded +-3; outlier uniform of width 200 sigma_pipe, not normalised for truncation; pi<=0.2 from model context only",
    "S10 loss = 2 x mean mixture NLL of a minibatch with bright-pixel oversampling (biased on purpose); v36 regulariser weights",
    "S11 gauges on a 3x3 lattice per training exposure; blur gauge uses the CORE second moment (r<=3 px); 1e-4 may not reach the gauge in a short budget",
    "S12 held-out nuisance: z_i by Adam (learned-noise NLL, c2f), quadratic background by exact WLS with pipeline ivar, NIRCam stripes by the v36 closed form; no affine registration refit, no separate exact gain (both only through z_i)",
    "S13 scores as heldout_chi2 (pipeline ivar); aperture lookup at the registered position x0+mu rather than a refit affine",
    "S14 HST coadds are v36's approximate tent-splat coadds; data ring and surface-brightness units as v36",
    "S15 --coarse-shift wired but default off and untested",
    "S16 pixel response not a separate factor (core lattice = quadrature of the pixel box inside Psi)",
    "S17 --operator factorised (diagnostic baseline): mu = free per-exposure shift + 2x2 matrix (target px, coordinates dx,dy in [-1,1]) + static cubic Legendre distortion per detector; Lambda = J; ONE free Psi table per channel (no lambda_q or position dependence); free per-exposure log gain; z zeroed in the trunk. Loss, sky, gauges and scoring unchanged (base_l2 prior still pulls Psi to the initial Gaussian)",
    "S18 --mu-skip: mu += mu_skip_scale x W z_i (linear, W random init std 1/sqrt(dz), own lr); z_i is still L2-penalised",
    "S19 factorised held-out fit: shift and matrix start at 0, log gain at the class mean of the training exposures of the channel (the hypernetwork gets the same information from the instrument token)",
    "S20 --sky-c2f: BARF-like release of sky resolution approximated by an extra Gaussian band limit on the READ (quadrature with the c2f blur) plus a low-pass of the initial sky; not a parameter pyramid; the sky parameters keep full resolution but their high frequencies get no gradient while blurred",
    "S21 --xfit-reg K: K sky views, view g trained only on fold-g pixels (gradient to the view only, registration/operator detached); registration, operator, noise and gauges trained on the pass that reads the OTHER view(s) (detached); folds = training exposures grouped by (channel, roll rounded to 1 degree), alternated over sorted groups; cost ~2x per step; regularisers on each view weighted 1/K; after --xfit-frac of the budget the views are averaged into the sky (the average of views registered to different skies is approximate); K>2 uses the mean of the other views; a channel with fewer roll groups than folds has its exposures alternated across folds",
    "S22 --restart-cycles: the sky is re-initialised from a v36 TENT-SPLAT coadd (no PSF matching, no outlier rejection, approximate for every instrument) at x0 + learned mu, (d - background - stripes)/exp(a); under --xfit-reg each view is rebuilt from its own fold only; Adam moments of the sky are reset; theta and z kept; restart points are fractions of the budget",
    "S24 --eval-only: theta, sky, nuisance of the training exposures are loaded strictly from --init-state (opfield_best.pt = best validation checkpoint; opfield_state.pt = the model after the restore of the best); the held-out protocol (S12-S13) then runs with the same fixed generator seed (1234) as at the end of training, so the scores reproduce up to floating-point non-determinism (threads, GPU atomics) of the Adam nuisance fit; the training pool is not built, the gauge report has no untrained reference (m2_ref = nan)",
    "S25 --wings cartesian: nested annuli of square lattices aligned to the origin beyond the core radius, pitch (--cart-steps, default 1 detector px to r = 24, 2 px to r = 48, 4 px beyond to the wing radius), node weight = pitch^2 (cells straddling an annulus boundary are not cut; absorbed by the learned Psi as in S1); wings read the sky pre-smoothed at the S2 levels by the node pitch (box-equivalent Gaussian, not a box); the azimuthal resolution of a lattice of pitch p at radius r is p / r rad (pitch 1: 5.7 deg at r = 10 to 2.4 deg at r = 24; pitch 2: 4.8 to 2.4 deg for r = 24-48; pitch 4: 4.8 to 2.4 deg for r = 48-97), i.e. about 5 deg or better but spikes narrower than that at large r are still unresolved; about 3-4 times more nodes than the log-polar wings, so memory and time per pixel grow in proportion",
    "S26 --base-per instrument|detector|detector_band: one base row per key (a row is a free log-density on the common stencil prefix of its channel); rows with no TRAINING exposure are aliased to the trained row of the same band (else channel) with most exposures (deterministic); the L2 prior stays mean over ALL base elements of (base - base_init)^2 x --base-l2 (so the pull per kernel relative to its own data is independent of the number of rows, but per-kernel it weakens in proportion to the number of rows); --base-l2 0 = no pull. 'instrument' is coarser than 'channel' for NIRCam (SW and LW share one row defined on the common stencil, in detector px of each channel)",
    "S27 --shape-head-per channel: one shape MLP and one bank (n_bank x channel stencil prefix) per channel; the trunk, the heads and the sky stay shared; the bank L2 is the mean over all bank elements",
    "S28 --base-table detector_band: base-per detector_band + Cartesian wings + low-rank modulation (--mod-rank vectors instead of --n-bank); the modulation acts in log-density before the softplus (multiplicative at high density), not on the table itself; --base-init-from copies the E2 learned tables (key learned: table + residual network, unit sum, zero outside its window) onto the stencil by the exact box integral of the piecewise-constant table over a box of side sqrt(dv) at every node (equal-area box for log-polar nodes), floor 1e-7, and resets the L2 anchor base0 to it: a DATA-DRIVEN warm start, not physics, and E2's table is a 4x-oversampled (0.25 px) object while the core stencil is --core-pitch (0.5 px in the huber configuration)",
    "S29 --dump-heldout-pixels: keys and dtypes of smacs_four_v38 dump_exposure_pixels; tx, ty = registered position x0 + mu (v38: after the held-out refit), model_sky = L + g (source part, gain included), gain = mean exp(a) over the exposure, noise_scale = 1, mjd = e['mjd'] else SCI MJD-AVG else primary EXPSTART (v38: JWST MJD-AVG, HST EXPSTART), split from the exposure flags",
    "S30 --lambda-gauge jacobian: Lambda == I, so M = J EXACTLY (J = estimate_jacobian of the exposure: the linear fit of the header WCS over the detector; a pure linear map, no distortion, no per-pixel Jacobian variation). head_lam stays in the module (state dicts of --lambda-gauge learned runs load strictly) but its output is NOT used: no gradient reaches it, the lc-l2 term is 0, and its weights keep their initial (or loaded) values. Stencil coverage with Lambda fixed (canonical px = detector px): the dense core r <= --core-radius (10 in the real runs) det px is +-0.31 arcsec on NIRCam SW (0.031 arcsec/px), +-0.63 arcsec on LW (0.063), +-1.28 arcsec on WFC3/IR (0.128); the wings reach --wing-arcsec (3 arcsec). The per-pixel variation of the Jacobian across the detector (the geometric-distortion change of scale and shear; not quantified here) is NOT modelled by Lambda any more and must come from mu (the shift field) and Psi (the shape); the original model could in principle absorb it in Lambda(x). Lambda-Psi gauge redundancy is removed by construction, at the price of any genuine local anisotropy that Psi would have to carry per detector position",
    "S31 --lam-in-band: adds to the shape-MLP input (alongside the 5 global-lambda features, 5 + 4 = 9 columns) the within-band position t = (lambda_q - lambda_mid) / (lambda_max - lambda_min) in [-0.5, 0.5] of each quadrature node q (lambda_min, lambda_max, lambda_mid = the extremes / midpoint of THAT band's own nodes; t = 0 for a band with one node) as (t, t^2, sin pi t, cos pi t): the shape can then vary across a band with a resolution the global lambda features (period 1 over 0.7-5.3 um) do not have. New input columns only (the shape MLP's last layer still starts at zero, so the initial kernel is unchanged). The chromatic freedom within a band is NOT constrained by physics: it can absorb sky colour (SED) through the PSF if the band nodes' weights are mis-specified",
    "S32 kernel_diagnostics (always on, new metrics.json key 'kernel_diagnostics'): exp(a) mean / std / p05 / p95 over at most 4000 randomly chosen TRAINING pixels per (instrument, detector, band) cell (even over its training exposures, fixed seed, the model's own per-exposure nuisance); EE at 0.1 / 0.2 / 0.5 arcsec from the sky-frame node offsets pos of the first training exposure of each roll group at its detector centre, averaged over the cell's roll groups (halo = 1 - EE; node-based, a log-polar ring is a step; nothing beyond the stencil is counted); 'a' is free (no penalty); kernels/effective_kernels.npz also stores |J and |Lc (Lambda) per group, TRUE detector-px node offsets = J^-1 pos (compare_kernels.py)",
    "S33 --psi-param log (default softplus): Psi = exp(base + coef @ bank), a LOG-density on the stencil; mass_k = Psi_k dv_k normalised to unit sum by a softmax of (x + log dv) (max subtracted inside: stable for log-densities spanning +-300). START: base_init = log(max(G, 1e-7)) with the same generic Gaussian + floor as S6, which is exactly the density the softplus start represents (softplus(inv_softplus(y)) = y), so the start kernel equals the softplus start to float32 round-off (6e-7 relative in tests/test_opfield_wings.py). Consequences, not removed: the bank / MLP coefficients act MULTIPLICATIVELY (relative precision in the wings, which softplus loses once a node is below ~1e-3 of the peak); Adam at --lr-kernel moves a log-density by at most ~3e-3 per step, so changing a wing by a factor 100 takes >= 1500 steps at full lr; the L2 prior (base - base0)^2 is now in log units (the same as softplus units at low density, different in the core); a node whose mass underflows float32 gets no gradient (softmax gradient is proportional to the node mass) and cannot recover; state dicts of softplus and log runs have the same shapes and DIFFERENT meaning (always load with the run's own args: --args-from). --base-init-from writes log(density) in this mode",
    "S34 --importance-sampling R1,R2,... (default off): per step --is-frac of the batch is drawn from annuli [0,R1), [R1,R2), ... (detector px of the pixel's own exposure) around the bright peaks of the CURRENT MODEL, the rest as before (uniform + --bright-frac); empty annuli are dropped and the share is split equally among the non-empty ones; every pixel carries w = 1 / (N q) with q the exact mixture pmf of the deterministic counts (balance heuristic), so mean(w nll) is an UNBIASED estimate of the all-pixel mean NLL of the training pool (checked on a toy problem); w <= B / n_uniform (defensive mixture). With IS on the S10 bright-pixel oversampling no longer biases the objective (it is only part of the proposal): the objective is the all-pixel loss, as E2-data; the regularisers, gauges and the held-out protocol are unchanged and not reweighted; the weights are not self-normalised. Peaks: 3 x 3 local maxima of the model band image above --is-snr x the coadd band noise sig_b (the SWarp-like coadd's median sigma, not the per-exposure noise), at most --is-max-peaks (200) brightest per band, integer-pixel positions (no sub-pixel refinement), refreshed every --is-refresh (200) steps from model.band_image() (no catalogue; the model of the first steps is the drizzle initial sky). Peak -> pixel mapping: the pool is sorted once by the key (band, round(y0), round(x0)); for each peak the pixels of the band within +-ceil(R_last s) sky px (s = largest sky px per detector px of the band) are gathered as one sorted run per cell row (torch.searchsorted), then the exact distance r = |J_e^-1 (x0 - peak)| decides, using the pixel's exposure Jacobian and its HEADER-WCS position x0 (not x0 + mu: mu of up to ~1 px is not included, comparable to the innermost annulus); a pixel belongs to the annulus of its NEAREST peak of the band; 30 detector px is 0.9 arcsec on NIRCam SW, 1.9 on LW and 3.8 on WFC3/IR (the annuli are in detector px, not arcsec); peaks of another band are not used. The effective sample size (sum w)^2 / sum w^2 of the batch weights (not of the loss) is logged every --log-every steps and stored in metrics.json['importance_sampling']. Extra memory: key (int32), permutation (int32) and stratum label (int8) = 9 bytes per pool pixel (+4 transient during a refresh, +8 during the initial sort); --importance-sampling is not implemented with --xfit-reg",
    "S35 --two-scale R_CORE_ARCSEC (default 0 = off): the per-pixel stencil (position / lambda / time dependent, as before) is TRUNCATED at r <= R_CORE (arcsec -> detector px per channel; a prefix of the radius-sorted stencil, also inside the dense core) and carries mass (1 - f_w) (unit sum over the truncated stencil, S4 centroid over the truncated core window); the WINGS r in (R_CORE, --wing-arcsec] are one FREE table per (detector, band) with a learned mass fraction f_w = sigmoid(phi) per (detector, band) (init = the mass of the generic Gaussian + floor outside R_CORE; separate for every row), so the kernel is exp(a) [(1 - f_w) K_core + f_w mod K_wing] with K_core, K_wing of unit sum; the split is a hard disk in the detector frame (the core nodes have cell areas dv, the wing lattice has 1 sky-pixel cells: the two quadratures do not tile the boundary exactly, the difference is absorbed by the learned masses). The wing is ACHROMATIC within the band (one table per detector and band), so the sum over the quadrature nodes commutes with the convolution (exact): ONE convolution per (detector, band, roll group) of the band image (not Q = 6). mu lives in the core only: the wing image is read at x0 + mu (bicubic, border replicated) but the wing kernel has no position of its own; the first moment of the wing table is unconstrained (as S4 for the wings). The wing uses the group Jacobian J, not J Lambda (with --lambda-gauge learned the wing ignores Lambda). --two-scale is not implemented with --xfit-reg or --operator factorised",
    "S36 --two-scale wing table and rendering: the table is a (2 nh + 1)^2 array of log-masses, nh = ceil(wing_arcsec / scale), detector frame, ONE NODE PER SKY PIXEL of the channel's nominal scale (node (i, j) = detector offset (i, j) / s with s = median sky px per detector px of the channel; 151^2 nodes for 3 arcsec at 0.04; the pitch in detector px differs between channels, 1.3 px on NIRCam SW to 0.31 px on WFC3/IR), only nodes with R_CORE < r <= nh/s are used. A roll group is rendered onto the sky lattice by BICUBIC interpolation of the LOG table at t = s J^-1 D (grid_sample, border padding), exp, annulus mask in the detector frame, unit-sum normalisation (softmax): the absolute scale of the table and the sky/detector cell-area ratio (|det J| / s^2, within ~1 percent between exposures) are absorbed by this normalisation; interpolation smooths structures narrower than ~2 nodes (spikes narrower than ~1 sky pixel are not representable), the orientation convention is that of the core (a pixel reads the sky at x + D, D = J u; checked by tests). The convolution is done by FFT on the replicate-padded band image of the c2f-blurred sky (padding nh, FFT size next 2^a 3^b 5^c >= H + 2 nh), exact (< 1e-6 relative) against the direct sum; groups = (detector, band, roll angle rounded to --wing-group-deg = 0.5 degree, parity) of ALL exposures (training, validation, held out), group Jacobian = MEAN J of its exposures (rotation error up to 0.25 degree = 0.3 sky px at 3 arcsec; not each exposure's own J), wing images computed lazily per step and cached (chunks of --wing-chunk groups per FFT call); (detector, band) rows with no training exposure are aliased like S26 (same band, else channel, most exposures). The wing image of a pixel is NOT affected by its own z_i (no per-pixel context in the wing), so the held-out nuisance fit only sees the wing through mu and a. Not modelled: wavelength / time / roll dependence of the wing table beyond the group Jacobian",
    "S37 --two-scale training: wing tables, logits and modulation have their own Adam group at --lr-kernel x --wing-lr-mult (5), a pull --wing-l2 (1e-4) x [mean (table - table_start)^2 + mean (logit - logit_start)^2 + mean c^2] (the L2 anchor is the START, no smoothness prior, so far wings with no constraint stay at the start). START (--wing-init powerlaw, default): the generic Gaussian + floor of S6 PLUS a power-law halo rho ~ r^-3 (the asymptotic slope of a diffraction-limited PSF; an ASSUMPTION, not the nominal PSF) carrying a mass fraction --wing-f0 (0.01) outside R_CORE, so f_w0 ~ 0.01 and the START KERNEL DIFFERS from the single-scale start by that 1 percent of mass moved to the wings (--wing-init floor: only the floor, f_w0 ~ 1e-3; in log parameters a wing 5 orders of magnitude below the start takes hundreds of Adam steps to grow, which slowed the synthetic fit); --wing-mod K: multiplies the wing amplitude by exp(sum_k c_k P_k(dx, dy)) over the NON-constant Legendre terms of total degree <= K in the detector coordinates (K = 0: none), one coefficient vector per (detector, band): it changes the wing MASS across the detector (flux is not conserved by the modulation; exp(a) and f_w absorb the mean), NOT its shape; kernel dumps (effective_kernels.npz, kernel_points) include the wing lattice nodes (sky offsets D, mass f_w mod K) after the core nodes; EE and halo diagnostics therefore count the wing out to --wing-arcsec. Memory / time: see metrics.json['two_scale'] (analytic estimate of the FFT temporaries and the autograd graph)",
    "S23 --z-hier: epochs are MJD clusters (gap 90 d) and visits come from the file names (JWST program/observation/visit; HST ipppss); the exposure part is penalised z-l2 x z-exp-mult; held-out exposures of unseen visits fit a free visit code (L2 z-l2-visit); epoch and visit codes of ids with no training exposure stay 0",
]

LEVEL_SIGMAS = (0.0, 0.7, 1.4, 2.8, 5.6, 11.2)  # sky px
PSI_FLOOR = 1e-7
U_WIDTH = 200.0
PI_MAX, PI_INIT = 0.2, 0.005
MJD_REF, MJD_SCALE = 58500.0, 1000.0
LOGF = None
T0 = time.perf_counter()


def log(msg):
    s = f"[{time.perf_counter() - T0:8.1f}s] {msg}"
    print(s, flush=True)
    if LOGF is not None:
        LOGF.write(s + "\n")
        LOGF.flush()


# ====================================================================================================================== registry
class OpReg:
    """Per band: pivot lambda [um], instrument name, channel name, stripes flag. Built from the v36 registry or by hand (tests)."""

    def __init__(self, bands, lam_um, instrument, channel, stripes):
        self.bands, self.lam_um = list(bands), [float(x) for x in lam_um]
        self.instrument, self.channel, self.stripes = list(instrument), list(channel), [bool(s) for s in stripes]
        self.index = {b: k for k, b in enumerate(self.bands)}
        self.instruments = list(dict.fromkeys(self.instrument))
        self.channels = list(dict.fromkeys(self.channel))
        self.inst_of_band = [self.instruments.index(i) for i in self.instrument]
        self.chan_of_band = [self.channels.index(c) for c in self.channel]
        self.multi = len(self.instruments) > 1

    @classmethod
    def from_v36(cls, r):
        return cls(r.bands, [r.lam_um[b] for b in r.bands], [r.instrument[b] for b in r.bands], [r.channel[b] for b in r.bands],
                   [r.is_nircam[b] for b in r.bands])


# ====================================================================================================================== stencil
CART_STEPS_DEFAULT = ((24.0, 1.0), (48.0, 2.0), (float("inf"), 4.0))   # --wings cartesian beyond the core radius (see S25)


def parse_cart_steps(txt):
    """'24:1,48:2,inf:4' -> ((24.0, 1.0), (48.0, 2.0), (inf, 4.0))."""
    if not txt:
        return CART_STEPS_DEFAULT
    out = []
    for item in str(txt).split(","):
        r_, p_ = item.split(":")
        out.append((float(r_), float(p_)))
    return tuple(out)


class Stencil:
    """Canonical stencil in detector pixels, sorted by radius (core first, then wing rings). See S1 of the module docstring."""

    def __init__(self, core_r=3.0, core_pitch=0.25, ratio=1.25, nang=24, rmax=100.0, wings="logpolar", cart_steps=None):
        self.wings = wings
        n = int(round(core_r / core_pitch))
        ax = np.arange(-n, n + 1) * core_pitch
        uu, vv = np.meshgrid(ax, ax)
        r = np.hypot(uu, vv).ravel()
        keep = r <= core_r + 1e-9
        core = np.stack([uu.ravel()[keep], vv.ravel()[keep]], 1)
        o = np.argsort(np.hypot(core[:, 0], core[:, 1]), kind="stable")
        core = core[o]
        v, dv, ring_of = [core], [np.full(len(core), core_pitch ** 2)], [np.full(len(core), -1)]
        edges = [core_r]
        pitch_cart = [np.full(len(core), core_pitch)]
        if wings == "cartesian":
            # NEW (--wings cartesian): nested annuli of square lattices (aligned to the origin) whose pitch grows in steps; node cell area = pitch^2.
            # cart_steps = [(outer radius [detector px], pitch [detector px]), ...] beyond the core radius; the last step extends to rmax.
            steps = list(cart_steps) if cart_steps else list(CART_STEPS_DEFAULT)
            r_in = core_r
            for (r_out, pit) in steps:
                r_out = min(float(r_out), rmax) if np.isfinite(r_out) else rmax
                if r_out <= r_in + 1e-9:
                    continue
                m_ = int(math.ceil(r_out / pit))
                ax_ = np.arange(-m_, m_ + 1) * pit
                uu_, vv_ = np.meshgrid(ax_, ax_)
                rr_ = np.hypot(uu_, vv_).ravel()
                kp_ = (rr_ > r_in + 1e-9) & (rr_ <= r_out + 1e-9)
                pts = np.stack([uu_.ravel()[kp_], vv_.ravel()[kp_]], 1)
                pts = pts[np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")]
                v.append(pts)
                dv.append(np.full(len(pts), pit ** 2))
                ring_of.append(np.full(len(pts), -1))
                pitch_cart.append(np.full(len(pts), pit))
                r_in = r_out
                if r_in >= rmax - 1e-9:
                    break
            edges = [core_r]   # no log-polar rings
        else:
            while edges[-1] < rmax:
                edges.append(edges[-1] * ratio)
        self.ring_r = []
        for j in range(len(edges) - 1):
            ri, ro = edges[j], edges[j + 1]
            rc = (2.0 / 3.0) * (ro ** 3 - ri ** 3) / (ro ** 2 - ri ** 2)  # area-weighted mean radius of the annulus
            ang = (np.arange(nang) + 0.5 * (j % 2)) * 2 * np.pi / nang
            v.append(np.stack([rc * np.cos(ang), rc * np.sin(ang)], 1))
            dv.append(np.full(nang, np.pi * (ro ** 2 - ri ** 2) / nang))
            ring_of.append(np.full(nang, j))
            self.ring_r.append(rc)
        self.v = np.concatenate(v)
        self.dv = np.concatenate(dv)
        self.ring_of = np.concatenate(ring_of)
        self.r = np.hypot(self.v[:, 0], self.v[:, 1])
        self.nc, self.nang, self.core_pitch, self.core_r = len(core), nang, core_pitch, core_r
        self.n15 = int((self.r[: self.nc] <= 1.5).sum())
        self.pitch = np.where(self.ring_of < 0, core_pitch, self.r * 2 * np.pi / nang)  # sample pitch in detector px
        if wings == "cartesian":
            self.pitch = np.concatenate(pitch_cart)
        self.n_rings = len(self.ring_r)

    def __len__(self):
        return len(self.v)

    def nk_for(self, R):
        """Number of stencil points (prefix) for a wing radius R [detector px]: core + the rings whose mean radius <= R (at least one)."""
        if self.wings == "cartesian":
            return max(self.nc, int(np.searchsorted(self.r, R + 1e-9, side="right")))   # r is sorted: prefix of nodes with r <= R
        nr = max(1, int(sum(1 for rc in self.ring_r if rc <= R))) if self.n_rings else 0
        return self.nc + nr * self.nang

    def nk_core_for(self, R):
        """NEW (--two-scale): prefix of the stencil with r <= R [detector px] for ANY R (nk_for never returns less than the whole dense core):
        R <= core radius -> the core nodes with r <= R (the core is sorted by radius); R > core radius -> nk_for(R)."""
        if R <= self.core_r + 1e-9:
            return max(1, int(np.searchsorted(self.r[: self.nc], R + 1e-9, side="right")))
        return self.nk_for(R)

    def levels_for(self, nk, scale_sky_per_det):
        """Pre-smoothing level of every point of the prefix (monotone in radius), for a channel with `scale` sky px per detector px."""
        sig_req = np.where(self.pitch[:nk] * scale_sky_per_det > 1.0, 0.2887 * self.pitch[:nk] * scale_sky_per_det, 0.0)
        lev = np.array([int(np.argmin([abs(s - t) for t in LEVEL_SIGMAS])) for s in sig_req])
        return np.maximum.accumulate(lev)


def level_slices(lev):
    sl, s = [], 0
    for i in range(1, len(lev) + 1):
        if i == len(lev) or lev[i] != lev[s]:
            sl.append((int(lev[s]), s, i))
            s = i
    return sl


# ====================================================================================================================== sky reading
def gauss_blur_img(img, sigma):
    """Separable Gaussian blur of img [K, H, W] (replicate padding, truncated at 4 sigma)."""
    if sigma < 0.05:
        return img
    r = int(math.ceil(4 * sigma))
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    K = img.shape[0]
    p = F.pad(img[None], (r, r, r, r), mode="replicate")
    p = F.conv2d(p, k.view(1, 1, 1, -1).repeat(K, 1, 1, 1), groups=K)
    return F.conv2d(p, k.view(1, 1, -1, 1).repeat(K, 1, 1, 1), groups=K)[0]


def sample_levels(levels, pts, slices):
    """levels: list of [K, H, W] (None = unused); pts [B, nk, 2] target-pixel positions (0-based pixel centres); slices [(level, s, e)].
    Returns [K, B, nk] (bicubic, border padding)."""
    outs = []
    for (l, s, e) in slices:
        img = levels[l]
        H, W = img.shape[-2:]
        g = pts[:, s:e]
        grid = torch.stack([(g[..., 0] + 0.5) / W * 2 - 1, (g[..., 1] + 0.5) / H * 2 - 1], -1)[None]
        outs.append(F.grid_sample(img[None], grid, mode="bicubic", padding_mode="border", align_corners=False)[0])
    return torch.cat(outs, 2) if len(outs) > 1 else outs[0]


def stencil_positions(x0, mu, M, v, cbar):
    """Sky positions of the stencil nodes: x0 + mu + M (v - cbar). x0, mu [B,2], M [B,2,2], v [nk,2], cbar [B,2] -> [B,nk,2]."""
    u = v[None] - cbar[:, None]
    return x0[:, None] + mu[:, None] + torch.einsum("bij,bnj->bni", M, u)


# ====================================================================================================================== two-scale wings (NEW, --two-scale)
class Levels(list):
    """list of pre-smoothed sky levels (index = level) that also carries `.wing` (a WingCache) in the --two-scale operator."""
    wing = None


def next_fast_len(n):
    """Smallest 2^a 3^b 5^c >= n (FFT sizes)."""
    n = int(n)
    best = 1 << max(n - 1, 0).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best


def wing_lattice(nh, device):
    """[2nh+1, 2nh+1, 2] sky-pixel offsets D = (x, y) (row index = y, column index = x)."""
    ax = torch.arange(-nh, nh + 1, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    return torch.stack([xx, yy], -1)


def render_wing_kernels(tab, Jinv, s_ch, rc_det, rw_det, nh):
    """Sky-frame wing kernels of G groups from detector-frame log tables (S33). tab [G, Nt, Nt] (Nt = 2 nh + 1; node (i, j) = detector offset
    (i - nh, j - nh) / s_ch, i.e. ONE NODE PER SKY PIXEL of the channel's nominal scale), Jinv [G,2,2] (sky px -> detector px), s_ch [G] sky px per
    detector px of the channel, rc_det / rw_det [G] inner / outer radius in detector px. Each lattice offset D (sky px) reads the table at
    t = s_ch Jinv D (nodes) by BICUBIC interpolation of the LOG table (grid_sample, border padding), exp, mask rc < |Jinv D| <= rw, normalised to
    unit sum (softmax: stable). Returns K [G, 2nh+1, 2nh+1] (K[y, x] = mass at sky offset D = (x - nh, y - nh); a pixel at x reads sum_D K(D) s(x + D))
    and the annulus mask."""
    lat = wing_lattice(nh, tab.device)
    u = torch.einsum("gij,yxj->gyxi", Jinv, lat)                    # detector px
    r = u.norm(dim=-1)
    grid = u * (s_ch / nh)[:, None, None, None]                     # align_corners=True: +-1 <-> nodes +-nh
    samp = F.grid_sample(tab[:, None], grid, mode="bicubic", padding_mode="border", align_corners=True)[:, 0]
    mask = (r > rc_det[:, None, None]) & (r <= rw_det[:, None, None])
    K = torch.softmax(samp.masked_fill(~mask, float("-inf")).flatten(1), -1).view_as(samp)
    return K, mask


def pad_fft_image(img, nh, Hp, Wp):
    """rfft2 of the replicate-padded (nh) image, zero-extended to (Hp, Wp) >= (H + 2 nh, W + 2 nh): [Hp, Wp // 2 + 1] complex."""
    p = F.pad(img[None, None], (nh, nh, nh, nh), mode="replicate")[0, 0]
    z = img.new_zeros(Hp, Wp)
    z[: p.shape[0], : p.shape[1]] = p
    return torch.fft.rfft2(z)


def correlate_fft(Fimg, K, nh, Hp, Wp, H, W):
    """out[g, y, x] = sum_D K[g](D) S(x + D) (S replicate-padded) by one FFT product per group: Fimg [G or 1, Hp, Wp // 2 + 1] from pad_fft_image."""
    G = K.shape[0]
    z = K.new_zeros(G, Hp, Wp)
    z[:, : 2 * nh + 1, : 2 * nh + 1] = K.flip(-1, -2)            # K'(D') = K(-D'): true convolution = correlation with K
    z = torch.roll(z, (-nh, -nh), (-2, -1))
    out = torch.fft.irfft2(Fimg * torch.fft.rfft2(z), s=(Hp, Wp))
    return out[:, nh:nh + H, nh:nh + W]


class WingCache:
    """Wing images of ONE step (S33): W_g = K_g * S_band (S_band = band image of the c2f-blurred sky, achromatic wing so the quadrature sum commutes with
    the convolution: one convolution per (detector, band, roll group), not one per node). Groups are computed lazily, `chunk` per batched FFT call, and
    cached (graph included) until the next make_levels."""

    def __init__(self, model, Sband):
        self.m, self.S, self.Fb, self.img = model, Sband, {}, {}

    def _fb(self, b):
        if b not in self.Fb:
            self.Fb[b] = pad_fft_image(self.S[b], self.m.wing_nh, self.m.wing_Hp, self.m.wing_Wp)
        return self.Fb[b]

    def ensure(self, gs):
        m = self.m
        todo = [g for g in gs if g not in self.img]
        for s0 in range(0, len(todo), m.wing_chunk):
            part = todo[s0:s0 + m.wing_chunk]
            gi = torch.tensor(part, dtype=torch.long, device=m.sv.device)
            K, _ = m.wing_kernels(gi)
            Fi = torch.stack([self._fb(b) for b in m.wg_band[gi].tolist()])
            out = correlate_fft(Fi, K, m.wing_nh, m.wing_Hp, m.wing_Wp, *self.S.shape[-2:])
            for k, g in enumerate(part):
                self.img[g] = out[k]


# ====================================================================================================================== nuisance
class Nuis(nn.Module):
    """Per-exposure free parameters: code z (dz), quadratic Legendre background (6) and NIRCam stripes (rows/cols)."""

    def __init__(self, n, dz, dev):
        super().__init__()
        self.z = nn.Parameter(torch.zeros(n, dz, device=dev))
        self.pe = nn.Parameter(torch.zeros(n, V.nterms(2), device=dev))
        self.zv = nn.Parameter(torch.zeros(n, dz, device=dev))   # z-hier: visit-level code of a held-out exposure whose visit is not in the training set
        self.aff = nn.Parameter(torch.zeros(n, 6, device=dev))   # factorised operator only: shift (2) + 2x2 matrix, target px (unused by the field operator)
        self.lg = nn.Parameter(torch.zeros(n, device=dev))       # factorised operator only: per-exposure log gain
        self.stripes = V.Stripes(n, dev)


def fourier_xy(dx, dy, nf=3):
    f = [dx[:, None], dy[:, None]]
    for k in range(1, nf + 1):
        f += [torch.sin(k * math.pi * dx)[:, None], torch.cos(k * math.pi * dx)[:, None],
              torch.sin(k * math.pi * dy)[:, None], torch.cos(k * math.pi * dy)[:, None]]
    return torch.cat(f, 1)  # 2 + 4 nf = 14


def lam_features(u):
    """u in [0, 1] (any shape) -> [..., 5]."""
    return torch.stack([u, torch.sin(math.pi * u), torch.cos(math.pi * u), torch.sin(2 * math.pi * u), torch.cos(2 * math.pi * u)], -1)


def inv_softplus(y):
    return np.log(np.expm1(y))


def mlp(i, h, o, n_hidden=2, zero_last=True):
    layers, d = [], i
    for _ in range(n_hidden):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    last = nn.Linear(d, o)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


def lambda_matrix(r):
    """r [B,4] raw -> Lc [B,2,2] = R(angle) diag(exp s1, exp s2) [[1, shear],[0, 1]]; zero input = identity."""
    s1, s2 = torch.tanh(r[:, 0]), torch.tanh(r[:, 1])
    ang, sh = math.pi * torch.tanh(r[:, 2]), 0.5 * torch.tanh(r[:, 3])
    c, s = torch.cos(ang), torch.sin(ang)
    e1, e2 = torch.exp(s1), torch.exp(s2)
    # R @ D @ Sh
    m00, m01 = c * e1, c * e1 * sh - s * e2
    m10, m11 = s * e1, s * e1 * sh + c * e2
    return torch.stack([torch.stack([m00, m01], -1), torch.stack([m10, m11], -1)], -2)


LOSS_KIND = "mixture"  # --loss; "huber" = v36-style Huber(delta) on (d - mu) / sigma, no outlier mixture (posterior = |z| > delta, for reporting)
HUBER_DELTA = 3.0


def nll_terms(d, mu, sp, h, pi):
    """Per-pixel mixture NLL and outlier posterior. sigma = sp exp(h); outlier density 1/(U_WIDTH sp).
    --loss huber: 0.5 z^2 for |z| <= delta, delta (|z| - delta/2) beyond, plus log sigma (h is 0 when the noise head is off)."""
    if LOSS_KIND == "huber":
        sigma = sp * torch.exp(h)
        z = (d - mu) / sigma
        az = z.abs()
        rho = torch.where(az <= HUBER_DELTA, 0.5 * z * z, HUBER_DELTA * (az - 0.5 * HUBER_DELTA))
        return rho + torch.log(sigma) + 0.9189385332, (az > HUBER_DELTA).float()
    sigma = sp * torch.exp(h)
    logn = -0.5 * ((d - mu) / sigma) ** 2 - torch.log(sigma) - 0.9189385332
    logu = -torch.log(U_WIDTH * sp)
    a = torch.log1p(-pi) + logn
    b = torch.log(pi) + logu
    lp = torch.logaddexp(a, b)
    return -lp, torch.exp(b - lp)


# ====================================================================================================================== the operator
class OpField(nn.Module):
    def __init__(self, a, reg, meta, stencil, A0, Phi_nodes, w_nodes, lam_nodes, n_det, scale_ch, dev, det_names=None):
        super().__init__()
        self.a_cfg, self.reg, self.stencil = a, reg, stencil
        for k, v in meta.items():
            self.register_buffer("m_" + k, v.to(dev))
        n_exp = int(meta["band"].numel())
        self.n_exp, self.dz = n_exp, int(a.dz)
        H = int(a.hidden)
        self.H = H
        nb, Q, K = Phi_nodes.shape
        self.nb, self.Q, self.K = nb, Q, K
        self.register_buffer("m_stripe", torch.tensor([float(s) for s in reg.stripes], device=dev))
        self.register_buffer("Phi", Phi_nodes.to(dev))
        self.register_buffer("wq", w_nodes.to(dev))
        self.register_buffer("Phi_band", (Phi_nodes * w_nodes[:, :, None]).sum(1).to(dev))
        self.register_buffer("lamf", lam_features(V.lam_unit(lam_nodes)).to(dev))  # nb, Q, 5
        self.register_buffer("lam_um", lam_nodes.detach().clone().to(dev), persistent=False)   # nb, Q [um] (kernel dumps; not in the state dict)
        self.register_buffer("bandf", lam_features(V.lam_unit(torch.tensor(reg.lam_um, dtype=torch.float32))).to(dev))  # nb, 5
        # NEW (--lambda-gauge, --lam-in-band): defaults reproduce the original model exactly (no new parameter, buffer in the state dict, or input column)
        self.lam_gauge = getattr(a, "lambda_gauge", "learned")
        self.lam_in_band = bool(getattr(a, "lam_in_band", False))
        _ln = lam_nodes.detach().clone().float()
        _lo, _hi = _ln.min(1, keepdim=True).values, _ln.max(1, keepdim=True).values
        _t = (_ln - 0.5 * (_lo + _hi)) / (_hi - _lo).clamp_min(1e-9)       # [nb, Q] in [-0.5, 0.5]: position of lambda_q inside ITS OWN band's node range
        self.register_buffer("lamb", torch.stack([_t, _t * _t, torch.sin(math.pi * _t), torch.cos(math.pi * _t)], -1).to(dev), persistent=False)  # nb, Q, 4
        self.sky = nn.Parameter(A0.clone().to(dev))
        # stencil per channel
        self.register_buffer("sv", torch.tensor(stencil.v, dtype=torch.float32, device=dev))
        self.register_buffer("sdv", torch.tensor(stencil.dv, dtype=torch.float32, device=dev))
        self.nch = len(reg.channels)
        self.nk_ch, self.slices_ch, self.R_ch = [], [], []
        used = set()
        self.two_scale = float(getattr(a, "two_scale", 0.0)) > 0      # NEW (--two-scale): per-pixel stencil truncated at R_core, wings = one FFT-convolved table per (detector, band)
        self.psi_log = getattr(a, "psi_param", "softplus") == "log"   # NEW (--psi-param log): Psi is a log-density (softmax over the stencil)
        for c in range(self.nch):
            if self.two_scale:
                R = a.two_scale / (scale_ch[c]["arcsec_per_det_px"])
                nk = stencil.nk_core_for(R)
            else:
                R = a.wing_arcsec / (scale_ch[c]["arcsec_per_det_px"])
                nk = stencil.nk_for(R)
            lev = stencil.levels_for(nk, scale_ch[c]["sky_per_det_px"])
            self.nk_ch.append(nk)
            self.R_ch.append(R)
            self.slices_ch.append(level_slices(lev))
            used |= {l for l, _, _ in self.slices_ch[-1]}
        self.used_levels = sorted(used)
        nk_max = max(self.nk_ch)
        self.nk_max = nk_max
        # shape: base per channel (log-density), bank
        G = np.exp(-0.5 * stencil.r[:nk_max] ** 2 / a.psf_init_sigma ** 2) / (2 * np.pi * a.psf_init_sigma ** 2)
        x0 = np.log(np.maximum(G, PSI_FLOOR)) if self.psi_log else inv_softplus(np.maximum(G, PSI_FLOOR))   # log: same start density max(G, floor)
        # NEW (--base-per): the base kernel is one row per (channel | instrument | detector | detector_band); default channel = the original model
        self.base_per = getattr(a, "base_per", "channel")
        self.det_names = list(det_names) if det_names is not None else [f"D{k}" for k in range(max(n_det, 1))]
        row_of_exp, self.row_keys = self._base_rows(meta)
        self.n_rows = len(self.row_keys)
        self.register_buffer("m_brow", row_of_exp.to(dev), persistent=False)   # exposure -> base row (re-mapped by set_train_rows for cells never trained)
        self.register_buffer("m_brow0", row_of_exp.to(dev).clone(), persistent=False)
        self.base = nn.Parameter(torch.tensor(np.tile(x0, (self.n_rows, 1)), dtype=torch.float32, device=dev))
        self.register_buffer("base0", self.base.detach().clone())
        # NEW (--shape-head-per channel): one shape MLP + one bank per channel; --base-table detector_band: bank rank = --mod-rank
        self.head_per = getattr(a, "shape_head_per", "shared")
        self.n_bank = int(a.n_bank)
        if getattr(a, "base_table", "none") != "none":
            self.n_bank = int(getattr(a, "mod_rank", 8))
        if self.head_per == "channel":
            self.bank = nn.ParameterList([nn.Parameter(torch.randn(self.n_bank, self.nk_ch[c], device=dev) * 0.01) for c in range(self.nch)])
        else:
            self.bank = nn.Parameter(torch.randn(self.n_bank, nk_max, device=dev) * 0.01)
        self.zh = bool(getattr(a, 'z_hier', False))
        in_dim = 14 + 4 + 4 + 5 + 1 + 1 + 3 + 1 + 2 + self.dz + (4 if self.zh else 0)
        self.det_emb = nn.Embedding(max(n_det, 1), 4)
        self.inst_emb = nn.Embedding(len(reg.instruments), 4)
        nn.init.normal_(self.det_emb.weight, std=0.5)
        nn.init.normal_(self.inst_emb.weight, std=0.5)
        self.trunk = nn.Sequential(nn.Linear(in_dim, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.head_mu = mlp(H, 0, 2, n_hidden=0)
        self.head_lam = mlp(H, 0, 4, n_hidden=0)
        self.head_a = mlp(H, 0, 1, n_hidden=0)
        if self.head_per == "channel":
            self.shape_mlp = nn.ModuleList([mlp(H + 5 + 4 * self.lam_in_band, int(a.shape_hidden), self.n_bank, n_hidden=1) for _ in range(self.nch)])
        else:
            self.shape_mlp = mlp(H + 5 + 4 * self.lam_in_band, int(a.shape_hidden), self.n_bank, n_hidden=1)
        gin = 4 + H
        self.g_net = mlp(gin, 32, 1, n_hidden=1)
        self.var_net = mlp(gin, 32, 1, n_hidden=1)
        self.pi_net = mlp(gin, 32, 1, n_hidden=1)
        with torch.no_grad():
            self.pi_net[-1].bias.fill_(math.log((PI_INIT / PI_MAX) / (1 - PI_INIT / PI_MAX)))
        self.mu_scale = 2.0
        self.pd = nn.Parameter(torch.zeros(max(n_det, 1), V.nterms(4), device=dev))
        self.operator = getattr(a, "operator", "field")
        self.fact = self.operator == "factorised"
        self.use_skip = bool(getattr(a, "mu_skip", False)) and not self.fact
        self.skip_scale = float(getattr(a, "mu_skip_scale", 4.0))
        if self.use_skip:
            self.mu_skip = nn.Linear(self.dz, 2, bias=False)
            nn.init.normal_(self.mu_skip.weight, std=1.0 / math.sqrt(self.dz))
        if self.zh:
            self.Ze = nn.Parameter(torch.zeros(int(self.m_ep_id.max()) + 1, self.dz, device=dev))
            self.Zv = nn.Parameter(torch.zeros(int(self.m_vis_id.max()) + 1, self.dz, device=dev))
        self.pdist = nn.Parameter(torch.zeros(max(n_det, 1), V.nterms(3), 2, device=dev))  # factorised: static cubic distortion per detector [target px]
        self.register_buffer("lg_class", torch.zeros(self.nch, device=dev))
        self.register_buffer("m_fold", torch.full((n_exp,), -1, dtype=torch.long, device=dev))
        self.nuis = Nuis(n_exp, self.dz, dev)
        if self.two_scale:
            self._init_wings(a, meta, scale_ch, tuple(A0.shape[-2:]), dev)
        self.to(dev)
        self._g_pen = torch.zeros((), device=dev)

    # ------------------------------------------------------------------------------------------------ two-scale wings (S33-S35)
    def _init_wings(self, a, meta, scale_ch, hw, dev):
        """Wing tables, fractions, modulation and roll groups of the --two-scale operator. One table per (detector, band) row; start = the generic
        round Gaussian + floor of S6 evaluated on the table lattice, f_w0 = its mass outside R_core / total (so the START kernel equals the start of the
        single-scale operator up to the node quadrature)."""
        if not (0.0 < a.two_scale < a.wing_arcsec):
            raise SystemExit(f"--two-scale {a.two_scale} arcsec must lie in (0, --wing-arcsec = {a.wing_arcsec})")
        K = int(getattr(a, "wing_mod", 0))
        self.wing_chunk = max(1, int(getattr(a, "wing_chunk", 8)))
        nh = int(math.ceil(a.wing_arcsec / a.scale - 1e-9))
        self.wing_nh, Nt = nh, 2 * nh + 1
        self.wing_Hp, self.wing_Wp = next_fast_len(hw[0] + 2 * nh), next_fast_len(hw[1] + 2 * nh)
        det, band, chan = (meta[k].cpu().numpy() for k in ("det", "band", "chan"))
        keys = sorted({(int(d), int(b)) for d, b in zip(det, band)})
        row0 = np.array([keys.index((int(d), int(b))) for d, b in zip(det, band)], dtype=np.int64)
        nr = len(keys)
        self.wing_keys, self.n_wrows = keys, nr
        self.wing_row_chan = [int(chan[int(np.nonzero(row0 == r)[0][0])]) for r in range(nr)]
        self.wing_row_band = [k[1] for k in keys]
        self.untrained_wing_cells = []
        dev_ = dev
        self.register_buffer("m_wrow0", torch.as_tensor(row0, device=dev_), persistent=False)
        self.register_buffer("m_wrow", torch.as_tensor(row0, device=dev_).clone(), persistent=False)
        self.register_buffer("wing_alias", torch.arange(nr, device=dev_), persistent=False)
        # roll groups: (row, roll angle rounded to --wing-group-deg, parity); the group Jacobian is the MEAN J of its exposures (S34)
        roll = meta["roll"].cpu().numpy().astype(np.float64)
        Jn = meta["J"].cpu().numpy().astype(np.float64)
        step = float(getattr(a, "wing_group_deg", 0.5))
        nbin = max(1, int(round(360.0 / step)))
        th = np.degrees(np.arctan2(roll[:, 1], roll[:, 0]))
        gk = [(int(row0[i]), int(np.floor(th[i] / step + 0.5)) % nbin, int(roll[i, 2])) for i in range(self.n_exp)]
        gl = sorted(set(gk))
        grp = np.array([gl.index(k) for k in gk], dtype=np.int64)
        Jg = np.stack([Jn[grp == g].mean(0) for g in range(len(gl))])
        self.wing_groups = [{"row": k[0], "det": keys[k[0]][0], "band": keys[k[0]][1], "theta_bin_deg": k[1] * step, "parity": k[2], "n_exp": int((grp == g).sum())} for g, k in enumerate(gl)]
        sk = np.array([scale_ch[self.wing_row_chan[k[0]]]["sky_per_det_px"] for k in gl])
        rc = np.array([a.two_scale / scale_ch[self.wing_row_chan[k[0]]]["arcsec_per_det_px"] for k in gl])
        f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev_)  # noqa: E731
        self.register_buffer("m_wgrp", torch.as_tensor(grp, device=dev_), persistent=False)
        self.register_buffer("wg_row0", torch.tensor([k[0] for k in gl], dtype=torch.long, device=dev_), persistent=False)
        self.register_buffer("wg_band", torch.tensor([keys[k[0]][1] for k in gl], dtype=torch.long, device=dev_), persistent=False)
        self.register_buffer("wg_Jinv", f32(np.linalg.inv(Jg)), persistent=False)
        self.register_buffer("wg_s", f32(sk), persistent=False)
        self.register_buffer("wg_Rc", f32(rc), persistent=False)
        self.register_buffer("wg_Rw", f32(nh / sk), persistent=False)
        # start: generic Gaussian + floor on the lattice (mass per node), f_w0 from the mass partition
        ii = np.arange(-nh, nh + 1)
        tab0, f0 = np.zeros((nr, Nt, Nt)), np.zeros(nr)
        sig = float(a.psf_init_sigma)
        for r in range(nr):
            c = self.wing_row_chan[r]
            s_ = scale_ch[c]["sky_per_det_px"]
            rdet = np.hypot(*np.meshgrid(ii, ii)) / s_
            dens = np.maximum(np.exp(-0.5 * rdet ** 2 / sig ** 2) / (2 * np.pi * sig ** 2), PSI_FLOOR)
            rcd = a.two_scale / scale_ch[c]["arcsec_per_det_px"]
            ring = (rdet > rcd) & (rdet <= nh / s_)
            if getattr(a, "wing_init", "powerlaw") == "powerlaw":
                # generic halo start: rho(r) = f0 r^-3 / (2 pi (1/R_core - 1/R_wing)) [mass per detector px^2] on the annulus (an ASSUMPTION: the asymptotic slope of a
                # diffraction-limited PSF, NOT the nominal PSF; f0 = --wing-f0), added to the Gaussian + floor
                dens = np.where(ring, dens + a.wing_f0 * np.where(ring, rdet, 1.0) ** -3 / (2 * np.pi * (1.0 / rcd - s_ / nh)), dens)
            tab0[r] = np.log(dens)
            m_w = float(dens[ring].sum() / s_ ** 2)
            nkc = self.nk_ch[c]
            Gs = np.maximum(np.exp(-0.5 * self.stencil.r[:nkc] ** 2 / sig ** 2) / (2 * np.pi * sig ** 2), PSI_FLOOR)
            m_c = float((Gs * self.stencil.dv[:nkc]).sum())
            f0[r] = m_w / (m_w + m_c)
        self.wing_tab = nn.Parameter(torch.tensor(tab0, dtype=torch.float32, device=dev_))
        self.register_buffer("wing_tab0", self.wing_tab.detach().clone(), persistent=False)
        lg0 = np.log(f0 / (1.0 - f0))
        self.wing_logit = nn.Parameter(torch.tensor(lg0, dtype=torch.float32, device=dev_))
        self.register_buffer("wing_logit0", self.wing_logit.detach().clone(), persistent=False)
        self.wing_mod_order = K
        if K > 0:
            self.wing_mod = nn.Parameter(torch.zeros(nr, V.nterms(K) - 1, device=dev_))
        else:
            self.register_parameter("wing_mod", None)

    def wing_kernels(self, gi):
        """Unit-sum sky-frame wing kernels of the groups gi [G] (render_wing_kernels), with the untrained-cell alias applied."""
        rows = self.wing_alias[self.wg_row0[gi]]
        return render_wing_kernels(self.wing_tab[rows], self.wg_Jinv[gi], self.wg_s[gi], self.wg_Rc[gi], self.wg_Rw[gi], self.wing_nh)

    def wing_amp(self, eid, dx, dy):
        """(f_w, modulation) per pixel: f_w = sigmoid(logit) of the (aliased) row, mod = exp(sum_k c_k P_k(dx, dy)) over the non-constant Legendre terms."""
        row = self.m_wrow[eid]
        fw = torch.sigmoid(self.wing_logit[row])
        if self.wing_mod is None:
            return fw, torch.ones_like(fw)
        leg = V._legendre2d(dx, dy, self.wing_mod_order)[:, 1:]
        return fw, torch.exp((leg * self.wing_mod[row]).sum(1))

    def wing_sample(self, cache, eid, pos):
        """Wing image of every pixel's roll group sampled at pos [B,2] (= x0 + mu) by bicubic interpolation: [B]."""
        g_all = self.m_wgrp[eid]
        gl = torch.unique(g_all).tolist()
        cache.ensure(gl)
        parts, idxs = [], []
        for g in gl:
            idx = torch.nonzero(g_all == g).squeeze(1)
            parts.append(sample_levels([cache.img[g][None]], pos[idx][:, None, :], [(0, 0, 1)])[0, :, 0])
            idxs.append(idx)
        perm = torch.cat(idxs)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        return torch.cat(parts)[inv]

    def wing_prior(self):
        d = ((self.wing_tab - self.wing_tab0) ** 2).mean() + ((self.wing_logit - self.wing_logit0) ** 2).mean()
        return d + (self.wing_mod ** 2).mean() if self.wing_mod is not None else d

    def _alias_wing_rows(self, train_idx):
        """Wing rows (detector, band) with no TRAINING exposure are aliased to the trained row of the same band (else channel) with most exposures."""
        train = torch.as_tensor(list(train_idx), dtype=torch.long)
        cnt = np.bincount(self.m_wrow0.cpu()[train].numpy(), minlength=self.n_wrows)
        alias, moves = np.arange(self.n_wrows), []
        lab = lambda r: f"{self.det_names[self.wing_keys[r][0]]}/{self.reg.bands[self.wing_keys[r][1]]}"  # noqa: E731
        for r in range(self.n_wrows):
            if cnt[r] > 0:
                continue
            cand = [q for q in range(self.n_wrows) if cnt[q] > 0 and self.wing_row_chan[q] == self.wing_row_chan[r]]
            pick = [q for q in cand if self.wing_row_band[q] == self.wing_row_band[r]] or cand
            if pick:
                q = max(pick, key=lambda x: cnt[x])
                alias[r] = q
                moves.append(f"{lab(r)} -> {lab(q)}")
        self.wing_alias.copy_(torch.as_tensor(alias, device=self.wing_alias.device))
        self.m_wrow.copy_(self.wing_alias[self.m_wrow0])
        return moves

    # ------------------------------------------------------------------------------------------------ context
    def zparts(self, eid, nid, nuis):
        """z-hier: (epoch + visit code, exposure code). Epoch / visit codes of ids without training exposures stay exactly 0 (no data gradient, L2 only)."""
        return self.Ze[self.m_ep_id[eid]] + self.Zv[self.m_vis_id[eid]] + nuis.zv[nid], nuis.z[nid]

    def zeff(self, eid, nid, nuis):
        if self.fact:
            return nuis.z[nid] * 0.0
        if self.zh:
            zs, zx = self.zparts(eid, nid, nuis)
            return zs + zx
        return nuis.z[nid]

    def ctx_input(self, eid, dx, dy, nid, nuis):
        tf = []
        if self.zh:   # smooth time input: Fourier features of the epoch (units of 1000 d)
            e_ = self.m_epoch[eid][:, None]
            tf = [torch.sin(2 * math.pi * e_ / 3.6), torch.cos(2 * math.pi * e_ / 3.6), torch.sin(2 * math.pi * e_ / 1.8), torch.cos(2 * math.pi * e_ / 1.8)]
        return torch.cat([fourier_xy(dx, dy), self.det_emb(self.m_det[eid]), self.inst_emb(self.m_inst[eid]), self.bandf[self.m_band[eid]],
                          self.m_epoch[eid][:, None], self.m_ltexp[eid][:, None], self.m_roll[eid], self.m_lscale[eid][:, None],
                          self.m_ro[eid], self.zeff(eid, nid, nuis)] + tf, 1)

    def heads(self, h, eid, cx=None):
        """cx = (dx, dy, nid, nuis) is needed for mu and a of the factorised operator and for the mu skip; with cx None only M (and the
        hypernetwork mu / a) are valid (used by core_m2, which needs M only)."""
        if self.fact:
            Lc = torch.eye(2, device=h.device)[None].expand(h.shape[0], 2, 2)
            M = self.m_J[eid]
            if cx is None:
                return torch.zeros(h.shape[0], 2, device=h.device), Lc, M, torch.zeros(h.shape[0], device=h.device)
            dx, dy, nid, nuis = cx
            ap = nuis.aff[nid]
            xy = torch.stack([dx, dy], 1)
            mu = ap[:, :2] + torch.einsum("bij,bj->bi", ap[:, 2:].reshape(-1, 2, 2), xy)
            mu = mu + torch.einsum("bt,bti->bi", V._legendre2d(dx, dy, 3), self.pdist[self.m_det[eid]])
            return mu, Lc, M, nuis.lg[nid]
        mu = self.mu_scale * self.head_mu(h)
        if self.use_skip and cx is not None:
            zin = self.zparts(eid, cx[2], cx[3])[0] if self.zh else cx[3].z[cx[2]]   # z-hier: the skip reads the epoch + visit code only
            mu = mu + self.skip_scale * self.mu_skip(zin)
        if self.lam_gauge == "jacobian":
            Lc = torch.eye(2, device=h.device)[None].expand(h.shape[0], 2, 2)    # Lambda == I: M = J exactly; head_lam is NOT evaluated here (no gradient reaches it)
        else:
            Lc = lambda_matrix(self.head_lam(h))
        a = 3.0 * torch.tanh(self.head_a(h)[:, 0] / 3.0)
        M = self.m_J[eid] if self.lam_gauge == "jacobian" else torch.einsum("bij,bjk->bik", self.m_J[eid], Lc)
        return mu, Lc, M, a

    # ------------------------------------------------------------------------------------------------ base rows / heads
    def _base_rows(self, meta):
        """(row index per exposure [n_exp], list of row keys). Keys: channel -> c; instrument -> i; detector -> d; detector_band -> (d, band)."""
        chan, inst, det, band = (meta[k].cpu().numpy() for k in ("chan", "inst", "det", "band"))
        per = self.base_per
        if per == "channel":
            keys = [int(c) for c in range(self.nch)]
            row = chan.copy()
        else:
            if per == "instrument":
                kk = [int(i) for i in inst]
            elif per == "detector":
                kk = [int(d) for d in det]
            elif per == "detector_band":
                kk = [(int(d), int(b)) for d, b in zip(det, band)]
            else:
                raise ValueError(f"unknown --base-per {per}")
            keys = sorted(set(kk))
            row = np.array([keys.index(k) for k in kk], dtype=np.int64)
        return torch.as_tensor(row, dtype=torch.long), keys

    def row_label(self, r):
        k = self.row_keys[r]
        if self.base_per == "channel":
            return self.reg.channels[k]
        if self.base_per == "instrument":
            return self.reg.instruments[k]
        if self.base_per == "detector":
            return self.det_names[k]
        return f"{self.det_names[k[0]]}/{self.reg.bands[k[1]]}"

    def set_train_rows(self, train_idx):
        """Rows (cells) with no TRAINING exposure would stay at their initial kernel and be used by held-out exposures of that cell. They are aliased:
        detector_band -> a trained row of the same band (most training exposures), else of the same channel; detector -> a trained row of the same
        channel. Returns the list of re-mapped (label -> label). Deterministic given the training set (so --eval-only reproduces it)."""
        if self.two_scale:
            self.untrained_wing_cells = self._alias_wing_rows(train_idx)   # NEW (--two-scale): the same rule for the wing tables
        if self.base_per in ("channel", "instrument"):
            return []
        train = torch.as_tensor(list(train_idx), dtype=torch.long)
        self.m_brow.copy_(self.m_brow0)
        cnt = np.bincount(self.m_brow0.cpu()[train].numpy(), minlength=self.n_rows)
        row_chan, row_band = {}, {}
        for e_ in range(self.n_exp):
            row_chan[int(self.m_brow0[e_])] = int(self.m_chan[e_])
            row_band[int(self.m_brow0[e_])] = int(self.m_band[e_])
        alias, moves = {}, []
        for r in range(self.n_rows):
            if cnt[r] > 0:
                continue
            cand = [q for q in range(self.n_rows) if cnt[q] > 0 and row_chan[q] == row_chan[r]]
            same_band = [q for q in cand if row_band[q] == row_band[r]]
            pick = same_band or cand
            if pick:
                q = max(pick, key=lambda x: cnt[x])
                alias[r] = q
                moves.append(f"{self.row_label(r)} -> {self.row_label(q)}")
        if alias:
            m = self.m_brow0.clone()
            for r, q in alias.items():
                m[self.m_brow0 == r] = q
            self.m_brow.copy_(m)
        self.untrained_cells = moves
        return moves

    @torch.no_grad()
    def init_base_from_e2(self, e2_dir, key="learned"):
        """NEW (--base-init-from): start every (detector, band) base row from the E2-data learned table of that pair (data-driven warm start, NOT
        physics). E2 file: <E2_RUN>/*_psf_groups.npy, ['detector_frame']['DET/BAND'][key] (square table, oversample os samples per detector px,
        centre at N // 2, mass per sample). Mass of each stencil node = the table mass inside the node's cell (a box of side sqrt(dv) centred on the
        node: exact cell for the Cartesian core / wings, an equal-area box for log-polar nodes) by the exact summed-area-table integral of the
        piecewise-constant table; density = mass / dv, base = inverse softplus of the density (floor PSI_FLOOR). base0 (the L2 anchor) is reset to it.
        Returns a report dict."""
        if self.base_per != "detector_band":
            raise SystemExit("--base-init-from needs --base-per detector_band (or --base-table detector_band)")
        p = Path(e2_dir)
        if p.is_file():
            fs = [p]
        else:
            fs = sorted(p.glob("explicit_lib_3d_psf_groups.npy")) or sorted(p.glob("*_psf_groups.npy"))
        if len(fs) != 1:
            raise SystemExit(f"--base-init-from {e2_dir}: expected one *_psf_groups.npy (explicit_lib_3d preferred), found {[f.name for f in fs]}")
        g = np.load(fs[0], allow_pickle=True).item()["detector_frame"]
        v = self.sv[: self.nk_max].cpu().numpy().astype(np.float64)      # [:nk_max]: the base rows have nk_max columns (the --two-scale stencil can be shorter than the whole one)
        side = np.sqrt(self.sdv[: self.nk_max].cpu().numpy().astype(np.float64))
        rep_ = {"file": str(fs[0]), "key": key, "matched": [], "unmatched": [], "mass_in_stencil": {}}
        for r in range(self.n_rows):
            lab = self.row_label(r)
            if lab not in g or key not in g[lab]:
                rep_["unmatched"].append(lab)
                continue
            K = np.asarray(g[lab][key], np.float64)
            K = K / K.sum()
            N, os_ = K.shape[0], float(g[lab]["oversample"])
            S = np.zeros((N + 1, N + 1))
            S[1:, 1:] = K.cumsum(0).cumsum(1)

            def Fcum(t, s_):
                t, s_ = np.clip(t, 0, N), np.clip(s_, 0, N)
                i, j = np.minimum(np.floor(t).astype(int), N - 1), np.minimum(np.floor(s_).astype(int), N - 1)
                ft, fs_ = t - i, s_ - j
                return S[j, i] * (1 - ft) * (1 - fs_) + S[j, i + 1] * ft * (1 - fs_) + S[j + 1, i] * (1 - ft) * fs_ + S[j + 1, i + 1] * ft * fs_

            ex = lambda u: u * os_ + N // 2 + 0.5   # noqa: E731  edge coordinate of detector offset u
            x0_, x1_, y0_, y1_ = ex(v[:, 0] - side / 2), ex(v[:, 0] + side / 2), ex(v[:, 1] - side / 2), ex(v[:, 1] + side / 2)
            mass = Fcum(x1_, y1_) - Fcum(x0_, y1_) - Fcum(x1_, y0_) + Fcum(x0_, y0_)
            dens = np.maximum(mass / (side ** 2), PSI_FLOOR)
            self.base[r] = torch.tensor(np.log(dens) if self.psi_log else inv_softplus(dens), dtype=torch.float32, device=self.base.device)
            rep_["matched"].append(lab)
            rep_["mass_in_stencil"][lab] = float(np.clip(mass, 0, None).sum())
        self.base0.copy_(self.base.detach())
        return rep_

    def kernel_params(self):
        return [self.base] + (list(self.bank) if self.head_per == "channel" else [self.bank])

    def _bank(self, c):
        return self.bank[c] if self.head_per == "channel" else self.bank[:, : self.nk_ch[c]]

    def _mlp(self, c):
        return self.shape_mlp[c] if self.head_per == "channel" else self.shape_mlp

    def bank_l2(self):
        return torch.cat([b.reshape(-1) for b in self.kernel_params()[1:]]).pow(2).mean()

    def param_counts(self):
        n = lambda ps: int(sum(p.numel() for p in ps))  # noqa: E731
        d = {"sky": n([self.sky]), "base": n([self.base]), "bank": n(self.kernel_params()[1:]), "shape_mlp": n(self.shape_mlp.parameters()),
             "trunk_heads_embeddings": n(list(self.trunk.parameters()) + list(self.head_mu.parameters()) + list(self.head_lam.parameters()) + list(self.head_a.parameters())
                                         + list(self.det_emb.parameters()) + list(self.inst_emb.parameters())),
             "g_var_pi_nets": n(list(self.g_net.parameters()) + list(self.var_net.parameters()) + list(self.pi_net.parameters())),
             "per_exposure_and_detector_nuisance": n(list(self.nuis.parameters())) + n([self.pd, self.pdist])}
        d["kernel_total (base+bank+shape_mlp)"] = d["base"] + d["bank"] + d["shape_mlp"]
        if self.two_scale:
            d["wing (tables + logit + mod)"] = n([p_ for p_ in (self.wing_tab, self.wing_logit, self.wing_mod) if p_ is not None])
        d["total"] = n(self.parameters())
        d["n_base_rows"] = self.n_rows
        return d

    def shape(self, c, f, bi, brow=None):
        """f [B,H] trunk features, bi [B] band index, brow [B] base-row index (None: the first row of channel c) -> w [B,Q,nk] (unit sum per q),
        cbar [B,2], wq [B,Q]."""
        nk = self.nk_ch[c]
        lamf = self.lamf[bi]
        if self.lam_in_band:
            lamf = torch.cat([lamf, self.lamb[bi]], -1)       # + (t, t^2, sin pi t, cos pi t), t = (lambda_q - lambda_mid) / (lambda_max - lambda_min) of the band
        B_, Q = lamf.shape[:2]
        cond = torch.cat([f[:, None, :].expand(B_, Q, f.shape[1]), lamf], -1)
        coef = torch.zeros(B_, Q, self.n_bank, device=f.device) if self.fact else self._mlp(c)(cond)
        if self.base_per == "channel":
            base = self.base[c, :nk]                       # original path (no per-pixel gather)
        else:
            if brow is None:
                brow = torch.full((B_,), self._first_row(c), dtype=torch.long, device=f.device)
            base = self.base[brow, :nk][:, None, :]
        x = base + coef @ self._bank(c)
        if self.psi_log:
            w = torch.softmax(x + torch.log(self.sdv[:nk]), -1)     # NEW (--psi-param log): Psi = exp(x) (log-density), mass = Psi dv normalised to unit sum (max subtracted inside softmax)
        else:
            m = F.softplus(x) * self.sdv[:nk]
            w = m / m.sum(-1, keepdim=True)
        nc = min(self.stencil.nc, nk)  # centroid of the CORE window (S4): the wings' first moment is dominated by a few far, faint nodes
        cq = (w[..., :nc] @ self.sv[:nc]) / w[..., :nc].sum(-1, keepdim=True)
        wq = self.wq[bi]
        cbar = (cq * wq[:, :, None]).sum(1)
        return w, cbar, wq

    def _first_row(self, c):
        for e_ in range(self.n_exp):
            if int(self.m_chan[e_]) == c:
                return int(self.m_brow[e_])
        return 0

    def make_levels(self, sigma_c2f=0.0, sky=None):
        A = self.sky if sky is None else sky
        out = []
        for l, s in enumerate(LEVEL_SIGMAS):
            out.append(gauss_blur_img(A, math.sqrt(sigma_c2f ** 2 + s ** 2)) if l in self.used_levels else None)
        if self.two_scale:   # NEW: the wing images of this step (lazy, cached) read the c2f-blurred band images
            out = Levels(out)
            out.wing = WingCache(self, gauss_blur_img(self.band_image(sky), sigma_c2f))
        return out

    def band_image(self, sky=None):
        return torch.einsum("bk,khw->bhw", self.Phi_band, self.sky if sky is None else sky)

    # ------------------------------------------------------------------------------------------------ forward
    def forward(self, bt, nuis, levels):
        """bt: eid, nid, x0 [B,2], dx, dy, row, col, sp (pipeline sigma). NO data value enters (pi and sigma are functions of the model only)."""
        eid, nid, x0 = bt["eid"], bt["nid"], bt["x0"]
        h = self.trunk(self.ctx_input(eid, bt["dx"], bt["dy"], nid, nuis))
        mu, Lc, M, a = self.heads(h, eid, (bt["dx"], bt["dy"], nid, nuis))
        lraw = torch.zeros(h.shape[0], 4, device=h.device) if self.lam_gauge == "jacobian" else self.head_lam(h)   # jacobian gauge: head_lam output unused (tanh(0)^2 = 0 in the lc-l2 term)
        B = eid.numel()
        ch, band = self.m_chan[eid], self.m_band[eid]
        Ls, idxs, stats = [], [], []
        for c in torch.unique(ch).tolist():
            idx = torch.nonzero(ch == c).squeeze(1)
            f, bi = h[idx], band[idx]
            w, cbar, wq = self.shape(c, f, bi, self.m_brow[eid[idx]])
            nk = self.nk_ch[c]
            pts = stencil_positions(x0[idx], mu[idx], M[idx], self.sv[:nk], cbar)
            S12 = sample_levels(levels, pts, self.slices_ch[c])  # K,B_,nk
            st = torch.einsum("kbn,bqk->bqn", S12, self.Phi[bi])
            Lq = (w * st).sum(-1)
            if self.two_scale:
                # NEW: kernel = exp(a) [(1 - f_w) K_core + f_w mod K_wing]; the wing image is read at x0 + mu (bicubic)
                fw, modv = self.wing_amp(eid[idx], bt["dx"][idx], bt["dy"][idx])
                Wv = self.wing_sample(levels.wing, eid[idx], (x0 + mu)[idx])
                Ls.append(torch.exp(a[idx]) * ((1.0 - fw) * (Lq * wq).sum(-1) + fw * modv * Wv))
            else:
                Ls.append(torch.exp(a[idx]) * (Lq * wq).sum(-1))
            sbar = (st * wq[:, :, None]).sum(1)[:, : self.stencil.n15]
            stats.append(torch.stack([sbar[:, 0], sbar.mean(1), sbar.max(1).values], 1))
            idxs.append(idx)
        perm = torch.cat(idxs)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(B, device=perm.device)
        L = torch.cat(Ls)[inv]
        stat = torch.cat(stats)[inv]
        unit = self.m_unit[eid]
        sa, gs = 3.0 * unit, 10.0 * unit
        gin = torch.cat([torch.asinh(L / sa)[:, None], torch.asinh(stat / sa[:, None]), h], 1)
        gate = L * L / (L * L + gs * gs)
        o = self.g_net(gin)[:, 0]
        g = gate * (sa + L.abs()) * o
        hv = 3.0 * torch.tanh(self.var_net(gin)[:, 0] / 3.0)
        pi = PI_MAX * torch.sigmoid(self.pi_net(gin)[:, 0])
        self._g_pen = (gate * o).pow(2).mean()
        leg2 = V._legendre2d(bt["dx"], bt["dy"], 2)
        leg4 = V._legendre2d(bt["dx"], bt["dy"], 4)
        bkg = self.m_b0[eid] + (leg2 * nuis.pe[nid]).sum(1) + (leg4 * self.pd[self.m_det[eid]]).sum(1)
        if bool(self.m_stripe[band].any()):
            st_ = nuis.stripes(nid, bt["row"].clamp(max=2047), bt["col"].clamp(max=2047)) * self.m_stripe[band]
        else:
            st_ = torch.zeros_like(L)
        src = L + g
        return dict(L=L, g=g, src=src, bkg=bkg, st=st_, pred=src + bkg + st_, h=hv, pi=pi, sigma=bt["sp"] * torch.exp(hv), mu=mu,
                    pos=x0 + mu, a=a, hfeat=h, lraw=lraw)

    # ------------------------------------------------------------------------------------------------ gauges and regularisers
    def lattice_inputs(self, eids, n):
        lat = torch.linspace(-0.8, 0.8, n, device=eids.device) if n > 1 else torch.zeros(1, device=eids.device)
        yy, xx = torch.meshgrid(lat, lat, indexing="ij")
        ne = eids.numel()
        return (eids.repeat_interleave(n * n), xx.reshape(-1).repeat(ne), yy.reshape(-1).repeat(ne))

    def core_m2(self, h_c, eid_c):
        """Second moment [target px^2] of the CORE (r <= core radius) of the band-integrated effective kernel at the given trunk features."""
        _, _, M, _ = self.heads(h_c, eid_c)
        ch, band = self.m_chan[eid_c], self.m_band[eid_c]
        out = torch.zeros(eid_c.numel(), device=h_c.device)
        parts, idxs = [], []
        for c in torch.unique(ch).tolist():
            idx = torch.nonzero(ch == c).squeeze(1)
            w, cbar, wq = self.shape(c, h_c[idx], band[idx], self.m_brow[eid_c[idx]])
            nc = min(self.stencil.nc, self.nk_ch[c])
            wk = (w * wq[:, :, None]).sum(1)[:, :nc]
            u = self.sv[:nc][None] - cbar[:, None]
            pos = torch.einsum("bij,bnj->bni", M[idx], u)
            parts.append((wk * (pos ** 2).sum(-1)).sum(1) / wk.sum(1))
            idxs.append(idx)
        perm = torch.cat(idxs)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        return torch.cat(parts)[inv]

    def gauge_terms(self, eids, npix, ref_mask, n=3):
        """mean mu [2] (pixel-count weighted), mean a over the reference-instrument exposures, core second moment per exposure."""
        eid, dx, dy = self.lattice_inputs(eids, n)
        h = self.trunk(self.ctx_input(eid, dx, dy, eid, self.nuis))
        mu, _, _, a = self.heads(h, eid, (dx, dy, eid, self.nuis))
        ne = eids.numel()
        mu_e = mu.view(ne, n * n, 2).mean(1)
        a_e = a.view(ne, n * n).mean(1)
        mu_bar = (mu_e * npix[:, None]).sum(0) / npix.sum()
        a_ref = a_e[ref_mask].mean() if bool(ref_mask.any()) else torch.zeros((), device=a.device)
        hc = h.view(ne, n * n, -1)[:, (n * n) // 2]
        return mu_bar, a_ref, self.core_m2(hc, eids)

    def prior_terms(self, train_eids):
        z = self.nuis.z[train_eids]
        zterm = (z ** 2).sum(1).mean()
        if self.zh:   # effective weight relative to --z-l2: exposure part x mult, epoch and visit parts by their own weights
            a_ = self.a_cfg
            zl = max(float(a_.z_l2), 1e-12)
            zterm = a_.z_exp_mult * zterm + (a_.z_l2_epoch / zl) * (self.Ze ** 2).sum(1).mean() + (a_.z_l2_visit / zl) * (self.Zv ** 2).sum(1).mean()
        d = dict(z=zterm, bank=self.bank_l2(), base=((self.base - self.base0) ** 2).mean(), g=self._g_pen)
        if self.two_scale:
            d["wing"] = self.wing_prior()
        return d


# ====================================================================================================================== data helpers
def exposure_numbers(e):
    """(mjd, exposure time [s]) of an exposure dict (HST dicts carry them; JWST from the primary header)."""
    mjd, tex = e.get("mjd"), e.get("exptime")
    if mjd is None or tex is None:
        try:
            h0 = fits.getheader(e["path"], 0)
            mjd = h0.get("EXPSTART", float("nan")) if mjd is None else mjd
            tex = h0.get("EFFEXPTM", h0.get("EXPTIME", float("nan"))) if tex is None else tex
        except Exception:  # noqa: BLE001
            mjd, tex = float("nan") if mjd is None else mjd, float("nan") if tex is None else tex
    return float(mjd if mjd is not None else float("nan")), float(tex if tex is not None else float("nan"))


_JW_RE = re.compile(r"^jw(\d{5})(\d{3})(\d{3})_")
_HSTP_RE = re.compile(r"^hst_(\d+)_([a-z0-9]{2})_", re.I)
_HST_RE = re.compile(r"^([ij])([a-z0-9]{3})([a-z0-9]{2})([a-z0-9]{2})([a-z0-9])_", re.I)


def visit_key(name):
    """Visit identifier from an exposure file / exposure name. JWST jwPPPPPOOOVVV_gggss_eeeee_det_cal -> 'jwPPPPPOOOVVV' (program, observation,
    visit); HST ipppssoot (_flt / _flc) -> first six characters 'ipppss' (instrument, program, visit). Anything else: the text before the first
    underscore (each such name is then its own visit unless names share a prefix)."""
    base = os.path.basename(str(name))
    m = _JW_RE.match(base)
    if m:
        return "jw" + m.group(1) + m.group(2) + m.group(3)
    m = _HST_RE.match(base)
    if m:
        return base[:6].lower()
    m = _HSTP_RE.match(base)   # MAST-style archive name hst_<program>_<visit>_<instrument>_<detector>_<filter>_<ipppssoot>_flt.fits
    if m:
        return "hst" + m.group(1) + "_" + m.group(2).lower()
    return base.split("_")[0]


def assign_codes(names, mjds, gap_days=90.0):
    """(epoch_id, visit_id) integer arrays. Visits from visit_key; epochs = clusters of the exposure start times separated by more than gap_days
    (all instruments together; a visit is never split by the gap in practice)."""
    m = np.asarray(mjds, dtype=float)
    order = np.argsort(m)
    ep = np.zeros(len(m), dtype=np.int64)
    c = 0
    for a_, b_ in zip(order[:-1], order[1:]):
        if m[b_] - m[a_] > gap_days:
            c += 1
        ep[b_] = c
    keys = [(visit_key(n), int(e_)) for n, e_ in zip(names, ep)]   # (visit key, epoch): HST 3-character program codes repeat across cycles
    uniq = sorted(set(keys))
    vis = np.array([uniq.index(k) for k in keys], dtype=np.int64)
    return ep, vis


def build_meta(exps, reg, dev, jac_key="jac"):
    """Per-exposure tables (index = position in exps)."""
    n = len(exps)
    mjd = np.array([exposure_numbers(e)[0] for e in exps])
    tex = np.array([exposure_numbers(e)[1] for e in exps])
    mjd = np.where(np.isfinite(mjd), mjd, np.nanmedian(mjd) if np.isfinite(mjd).any() else MJD_REF)
    tex = np.where(np.isfinite(tex) & (tex > 0), tex, np.nanmedian(tex[tex > 0]) if (tex > 0).any() else 1000.0)
    J = np.array([e[jac_key]["J"] for e in exps], np.float32).reshape(n, 2, 2)
    roll = np.array([[math.cos(math.radians(e[jac_key]["theta_deg"])), math.sin(math.radians(e[jac_key]["theta_deg"])),
                      float(e[jac_key]["parity"])] for e in exps], np.float32)
    band = np.array([reg.index[e["band"]] for e in exps])
    stripe = np.array([float(reg.stripes[b]) for b in band], np.float32)
    ro = np.array([(J[i, :, 0] / np.linalg.norm(J[i, :, 0])) * stripe[i] for i in range(n)], np.float32)
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)  # noqa: E731
    ep_id, vis_id = assign_codes([e.get('name', '') for e in exps], mjd)
    return dict(ep_id=t(ep_id, torch.long), vis_id=t(vis_id, torch.long), band=t(band, torch.long), inst=t([reg.inst_of_band[b] for b in band], torch.long),
                chan=t([reg.chan_of_band[b] for b in band], torch.long), det=t([e["det"] for e in exps], torch.long),
                epoch=t((mjd - MJD_REF) / MJD_SCALE), ltexp=t(np.log(tex) / 7.0), roll=t(roll),
                lscale=t([math.log(e[jac_key]["scale"]) for e in exps]), ro=t(ro), J=t(J),
                unit=t([float((1.0 / e["ivar"].sqrt()).median()) for e in exps]), b0=t([e["bkg0"] for e in exps]),
                stripe_exp=t(stripe)), mjd, tex


def channel_scales(exps, reg, meta_band, target_scale_arcsec):
    """Per channel: median J scale (sky px per detector px) and arcsec per detector px."""
    out = []
    for c in range(len(reg.channels)):
        sc = [e["jac"]["scale"] for e in exps if reg.chan_of_band[reg.index[e["band"]]] == c]
        s = float(np.median(sc)) if sc else 1.0
        out.append({"sky_per_det_px": s, "arcsec_per_det_px": s * target_scale_arcsec})
    return out


class Pool:
    """Flat pixel pool of the training exposures (device tensors)."""

    def __init__(self, exps, train_idx, src_masks, band_index, dev):
        cols = {k: [] for k in ("eid", "x0", "dx", "dy", "d", "sp", "row", "col", "bright")}
        for i in train_idx:
            e = exps[i]
            n = e["d"].numel()
            cols["eid"].append(torch.full((n,), i, dtype=torch.long, device=dev))
            cols["x0"].append(torch.stack([e["tx"], e["ty"]], 1))
            cols["dx"].append(e["dx"])
            cols["dy"].append(e["dy"])
            cols["d"].append(e["d"])
            cols["sp"].append(1.0 / e["ivar"].sqrt())
            cols["row"].append(e["row"].to(torch.int32))
            cols["col"].append(e["col"].to(torch.int32))
            b = band_index[e["band"]]
            cols["bright"].append(V.sample(src_masks[b][None].float(), e["tx"], e["ty"])[0] > 0.5)
        for k, v in cols.items():
            setattr(self, k, torch.cat(v))
        self.N = int(self.d.numel())
        self.bright_idx = torch.nonzero(self.bright).squeeze(1)
        self.device = dev

    def draw(self, B, bright_frac, gen):
        nb = int(B * bright_frac) if self.bright_idx.numel() > 0 else 0
        sel = torch.randint(0, self.N, (B - nb,), device=self.device, generator=gen)
        if nb:
            sel = torch.cat([sel, self.bright_idx[torch.randint(0, self.bright_idx.numel(), (nb,), device=self.device, generator=gen)]])
        return self.get(sel)

    def get(self, sel):
        eid = self.eid[sel]
        return dict(eid=eid, nid=eid, x0=self.x0[sel], dx=self.dx[sel], dy=self.dy[sel], d=self.d[sel], sp=self.sp[sel],
                    row=self.row[sel].long(), col=self.col[sel].long())


def parse_is_radii(txt):
    """'1,3,8,15,30' -> [1.0, 3.0, 8.0, 15.0, 30.0] (outer radii in detector px; annulus j = [R_(j-1), R_j), the first starts at 0)."""
    r = [float(x) for x in str(txt).split(",") if x.strip()]
    if not r or any(x <= 0 for x in r) or any(b <= a_ for a_, b in zip(r[:-1], r[1:])):
        raise SystemExit(f"--importance-sampling {txt!r}: need strictly increasing positive radii")
    return r


class PeakSampler:
    """NEW (--importance-sampling R1,R2,...): annulus-stratified, exactly reweighted pixel sampling around the bright peaks of the CURRENT MODEL (S36).

    Peaks: local maxima (3 x 3) of the model band image (model.band_image(), no catalogue) above --is-snr x sig_b of the band, at most --is-max-peaks per
    band (the brightest), refreshed every --is-refresh steps. Peak -> pixel mapping: every training pixel has its header-WCS target position x0; the pool is
    sorted once by the key (band, row, column) of the integer cell round(x0) (a grid lookup with one sorted run per cell row); for a peak the pixels of the
    band within +-ceil(R_max s) sky px (s = largest sky px per detector px of the band's exposures) are gathered as one contiguous run per cell row
    (torch.searchsorted), and the EXACT detector-frame distance r = |J_e^-1 (x0 - peak)| (J_e: the pixel's exposure Jacobian, detector px) assigns the annulus
    of the NEAREST peak (r_min over peaks of the band). Annuli [0, R1), [R1, R2), ...; a pixel beyond the last radius is in no stratum.
    Sampling pmf of a batch of B draws (deterministic counts, balance-heuristic mixture): n_u uniform draws over the pool, n_b uniform draws over the bright
    (coadd S/N > 5) pixels (as --bright-frac), n_j uniform draws in stratum j (empty strata dropped, --is-frac of the batch shared equally);
    q(p) = [n_u / N + n_b 1_bright(p) / N_b + sum_j n_j 1_j(p) / |S_j|] / B, and every pixel carries the weight w = 1 / (N q(p)), so E[mean_batch w nll] =
    mean over ALL pool pixels of nll exactly (unbiased); the uniform component bounds w <= B / n_u."""

    def __init__(self, pool, model, radii, a, sig_b):
        dev = pool.device
        self.pool, self.model, self.a, self.dev = pool, model, a, dev
        self.edges = torch.tensor(radii, dtype=torch.float32, device=dev)
        self.n_ann = len(radii)
        self.frac = float(a.is_frac)
        self.sig_b = sig_b
        nb = len(model.reg.bands)
        lo_, hi_ = pool.x0.round().amin(0), pool.x0.round().amax(0)
        self.x_min, self.y_min = int(lo_[0]), int(lo_[1])
        self.Wc, self.Hc = int(hi_[0]) - self.x_min + 1, int(hi_[1]) - self.y_min + 1
        kd = torch.int32 if nb * self.Hc * self.Wc < 2 ** 31 - 1 else torch.int64
        key = torch.empty(pool.N, dtype=kd, device=dev)
        for s0 in range(0, pool.N, 1 << 22):                       # chunked: no N-sized int64 temporaries
            sl = slice(s0, s0 + (1 << 22))
            cxy = pool.x0[sl].round().long()
            key[sl] = ((model.m_band[pool.eid[sl]] * self.Hc + (cxy[:, 1] - self.y_min)) * self.Wc + (cxy[:, 0] - self.x_min)).to(kd)
        sk, perm = torch.sort(key)
        del key
        self.skey, self.perm = sk, perm.to(torch.int32 if pool.N < 2 ** 31 - 1 else torch.int64)
        del perm
        self.kd = kd
        self.lab = torch.full((pool.N,), -1, dtype=torch.int8, device=dev)
        self.strata, self.counts = [None] * self.n_ann, [0] * self.n_ann
        # per-exposure inverse Jacobian and per-band largest scale
        J = model.m_J.detach().double()
        self.Jinv = torch.linalg.inv(J).float()
        sc = torch.linalg.matrix_norm(J, ord=2).float()
        self.Rs = []
        for b in range(nb):
            ids = (model.m_band == b).nonzero().squeeze(1)
            self.Rs.append(int(math.ceil(float(radii[-1]) * (float(sc[ids].max()) if ids.numel() else 1.0))) + 1)
        self.bright_n = int(pool.bright_idx.numel())
        self.n_refresh = 0
        self.history, self.ess_hist = [], []
        self.last = {}

    @torch.no_grad()
    def refresh(self, step):
        t0 = time.perf_counter()
        m, pool, dev = self.model, self.pool, self.dev
        img = m.band_image()
        mx = F.max_pool2d(img[None], 3, 1, 1)[0]
        rmin = torch.full((pool.N,), float("inf"), device=dev)
        n_peaks = []
        for b in range(img.shape[0]):
            thr = float(self.a.is_snr) * float(self.sig_b[b])
            ys, xs = torch.nonzero((img[b] >= mx[b]) & (img[b] > thr), as_tuple=True)
            if ys.numel() > int(self.a.is_max_peaks):
                top = torch.topk(img[b][ys, xs], int(self.a.is_max_peaks)).indices
                ys, xs = ys[top], xs[top]
            n_peaks.append(int(ys.numel()))
            if ys.numel() == 0:
                continue
            R = self.Rs[b]
            rows = torch.arange(-R, R + 1, device=dev)
            for p0 in range(0, ys.numel(), 8):
                py, px = ys[p0:p0 + 8], xs[p0:p0 + 8]
                yy = py[:, None] + rows[None] - self.y_min                                  # [P, nr] cell rows
                xl = (px - R - self.x_min).clamp(min=0)[:, None].expand_as(yy)
                xr = (px + R - self.x_min).clamp(max=self.Wc - 1)[:, None].expand_as(yy)
                ok = (yy >= 0) & (yy < self.Hc) & (xr >= xl)
                yc = yy.clamp(0, self.Hc - 1)
                lo = ((b * self.Hc + yc) * self.Wc + xl).to(self.kd)
                hi = ((b * self.Hc + yc) * self.Wc + xr).to(self.kd)
                st = torch.searchsorted(self.skey, lo.flatten(), right=False)
                en = torch.searchsorted(self.skey, hi.flatten(), right=True)
                ln = torch.where(ok.flatten(), en - st, torch.zeros_like(en))
                tot = int(ln.sum())
                if tot == 0:
                    continue
                rep_ = torch.repeat_interleave(torch.arange(ln.numel(), device=dev), ln)
                off = torch.arange(tot, device=dev) - (torch.cumsum(ln, 0) - ln)[rep_]
                pix = self.perm[(st[rep_] + off).long()].long()
                pk = rep_ // rows.numel()
                d = pool.x0[pix] - torch.stack([px, py], 1).float()[pk]
                u = torch.einsum("nij,nj->ni", self.Jinv[pool.eid[pix]], d)
                r = u.norm(dim=-1)
                rmin.scatter_reduce_(0, pix, r, "amin")
        self.lab = torch.empty(pool.N, dtype=torch.int8, device=dev)
        for s0 in range(0, pool.N, 1 << 23):                                                 # chunked: r < R1 -> 0, ..., r >= R_last -> -1
            lb = torch.bucketize(rmin[s0:s0 + (1 << 23)], self.edges, right=True)
            self.lab[s0:s0 + (1 << 23)] = torch.where(lb >= self.n_ann, torch.full_like(lb, -1), lb).to(torch.int8)
        del rmin
        for j in range(self.n_ann):
            idx = torch.nonzero(self.lab == j).squeeze(1)
            self.strata[j], self.counts[j] = idx, int(idx.numel())
        self.n_refresh += 1
        rec = {"step": int(step), "peaks_per_band": n_peaks, "stratum_pixels": list(self.counts), "seconds": round(time.perf_counter() - t0, 2)}
        self.history.append(rec)
        log(f"IS refresh at step {step}: peaks per band {n_peaks} (S/N > {self.a.is_snr:g}); pixels per annulus {self.counts} of {pool.N} ({time.perf_counter() - t0:.1f} s)")

    @torch.no_grad()
    def draw(self, B, bright_frac, gen):
        pool, dev = self.pool, self.dev
        act = [j for j in range(self.n_ann) if self.counts[j] > 0]
        n_is = int(round(self.frac * B)) if act else 0
        nj = [0] * self.n_ann
        for k, j in enumerate(act):
            nj[j] = n_is // len(act) + (1 if k < n_is % len(act) else 0)
        n_def = B - sum(nj)
        nb = int(n_def * bright_frac) if self.bright_n > 0 else 0
        n_u = n_def - nb
        parts = [torch.randint(0, pool.N, (n_u,), device=dev, generator=gen)]
        if nb:
            parts.append(pool.bright_idx[torch.randint(0, self.bright_n, (nb,), device=dev, generator=gen)])
        for j in act:
            parts.append(self.strata[j][torch.randint(0, self.counts[j], (nj[j],), device=dev, generator=gen)])
        sel = torch.cat(parts)
        q = torch.full((B,), n_u / pool.N, device=dev)
        if nb:
            q = q + nb * pool.bright[sel].float() / self.bright_n
        coef = torch.tensor([nj[j] / self.counts[j] if self.counts[j] > 0 else 0.0 for j in range(self.n_ann)], device=dev)
        ls = self.lab[sel].long()
        q = q + torch.where(ls >= 0, coef[ls.clamp(min=0)], torch.zeros_like(q))
        w = 1.0 / (pool.N * q / B)
        bt = pool.get(sel)
        bt["w"] = w
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        self.last = {"ess": ess, "n_is": sum(nj), "w_max": float(w.max()), "w_min": float(w.min())}
        return bt


def exp_batch(e, eid, sel=None, nid=0):
    """Batch dict of (a subset of) one exposure; nid indexes a Nuis object (0 for a single-exposure Nuis)."""
    s = slice(None) if sel is None else sel
    x0 = torch.stack([e["tx"], e["ty"]], 1)[s]
    n = x0.shape[0]
    dev = x0.device
    return dict(eid=torch.full((n,), eid, dtype=torch.long, device=dev), nid=torch.full((n,), nid, dtype=torch.long, device=dev), x0=x0,
                dx=e["dx"][s], dy=e["dy"][s], d=e["d"][s], sp=1.0 / e["ivar"][s].sqrt(), row=e["row"][s], col=e["col"][s])


def estimate_wing_memory_mb(n_groups, Hp, Wp, chunk, n_bands):
    """Analytic estimate [MB] (float32 / complex64) of the --two-scale wing images with autograd: per group the kernel FFT and the gathered band FFT kept for the
    backward pass (2 complex Hp x (Wp/2+1) = 8 Hp Wp bytes) plus the real IFFT output (4 Hp Wp); per band one image FFT (4 Hp Wp); per chunk the temporaries
    (zero-padded kernels, kernel FFT, product: 12 Hp Wp per group of the chunk); x 1.3."""
    hw = float(Hp) * float(Wp)
    return 1.3 * (n_groups * 12.0 * hw + n_bands * 4.0 * hw + chunk * 12.0 * hw) / 2 ** 20


def estimate_batch_memory_mb(B, nk, Q, K=12, n_levels=6, H=96, hw=(640, 640), n_bank=32):
    """Analytic estimate [MB] of the peak training memory (float32) for a pixel batch B, stencil nk (largest channel)."""
    per_pix = nk * (2 + 6 + K + Q + 4 * Q + Q + 2) + (2 * H + 5 * 96)          # positions, S12, contraction, Psi chain, products; trunk
    sky = n_levels * K * hw[0] * hw[1] * 3 + K * hw[0] * hw[1] * 4              # level images (+ conv temporaries / grads), A, Adam
    return (4.0 * B * per_pix * 1.25 + 4.0 * sky) / 2 ** 20


# ====================================================================================================================== nuisance fit and scoring
@contextlib.contextmanager
def frozen(model):
    req = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in req:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, r in req:
            p.requires_grad_(r)


def c2f_sigma(progress, a):
    s0 = a.c2f_start_arcsec / a.scale
    s_read = s0 * max(0.0, 1.0 - progress / max(a.c2f_frac, 1e-6)) if a.c2f_frac > 0 else 0.0
    s_sky = a.sky_c2f_arcsec / a.scale * max(0.0, 1.0 - progress / a.sky_c2f) if a.sky_c2f > 0 else 0.0
    return math.sqrt(s_read ** 2 + s_sky ** 2)


def fit_nuisance(model, e, eid, a, steps, gen, lr_bkg):
    """Held-out nuisance fit: theta and sky frozen; z_i (init 0) and the quadratic background by Adam on minibatches of the learned-noise
    mixture NLL, with the coarse-to-fine blur over the first half of the steps. Returns a Nuis object holding one exposure."""
    dev = e["d"].device
    nuis = Nuis(1, model.dz, dev)
    groups = [{"params": [nuis.z] + ([nuis.zv] if model.zh else []), "lr": a.lr_z}, {"params": [nuis.pe], "lr": lr_bkg}]
    if model.fact:
        with torch.no_grad():
            nuis.lg.fill_(float(model.lg_class[model.m_chan[eid]]))   # class-mean log gain of the training exposures of this channel (S19)
        groups += [{"params": [nuis.aff], "lr": a.lr_aff}, {"params": [nuis.lg], "lr": a.lr_gain}]
    opt = torch.optim.Adam(groups)
    base = [g["lr"] for g in opt.param_groups]
    N = e["d"].numel()
    bs = min(N, int(a.nuis_batch))
    lv0 = None
    with frozen(model):
        for it in range(steps):
            frac = it / max(steps, 1)
            sig = max(a.c2f_start_arcsec, a.sky_c2f_arcsec if a.sky_c2f > 0 else 0.0) / a.scale * max(0.0, 1.0 - frac / 0.5) if a.nuis_c2f else 0.0
            with torch.no_grad():
                if sig > 0 or lv0 is None:
                    lv = model.make_levels(sig)
                    if sig <= 0:
                        lv0 = lv
                else:
                    lv = lv0
            f = 0.5 * (1 + math.cos(math.pi * frac)) * 0.98 + 0.02
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            sel = torch.randint(0, N, (bs,), device=dev, generator=gen)
            bt = exp_batch(e, eid, sel)
            out = model(bt, nuis, lv)
            nll, _ = nll_terms(bt["d"], out["pred"], bt["sp"], out["h"], out["pi"])
            loss = 2 * nll.mean() + a.z_l2 * (a.z_exp_mult if model.zh else 1.0) * (nuis.z ** 2).sum(1).mean() + (a.z_l2_visit * (nuis.zv ** 2).sum(1).mean() if model.zh else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return nuis


def predict_exposure(model, e, eid, nuis, chunk=None):
    """Model quantities for all pixels of an exposure (no grad): src = L + g, pattern (detector), leg2, h, pi, pos, stat arrays."""
    chunk = chunk or (8192 if e["d"].is_cuda else 2048)
    with torch.no_grad():
        lv = model.make_levels(0.0)
        keys = ("src", "L", "g", "h", "pi", "pos", "mu", "a")
        acc = {k: [] for k in keys}
        N = e["d"].numel()
        for s in range(0, N, chunk):
            bt = exp_batch(e, eid, slice(s, min(N, s + chunk)))
            out = model(bt, nuis, lv)
            for k in keys:
                acc[k].append(out[k])
        res = {k: torch.cat(v) for k, v in acc.items()}
        res["pattern"] = (V._legendre2d(e["dx"], e["dy"], 4) * model.pd[model.m_det[eid]]).sum(1)
        return res


def finalize_exposure(model, e, eid, nuis, use_stripes):
    """Exact WLS (pipeline ivar) for the quadratic background and the v36 closed-form NIRCam stripes, given z (v36 heldout_chi2 structure).
    Returns the full prediction m and the model quantities."""
    P = predict_exposure(model, e, eid, nuis)
    with torch.no_grad():
        ivar, d = e["ivar"], e["d"]
        A = V._legendre2d(e["dx"], e["dy"], 2)
        sw = ivar.sqrt()
        Aw = (A * sw[:, None]).double()
        AtA = (Aw.T @ Aw).cpu()
        AtA = AtA + 1e-9 * torch.eye(AtA.shape[0], dtype=AtA.dtype) * AtA.diagonal().mean()

        def solve_pe(st):
            y = d - e["bkg0"] - P["pattern"] - P["src"] - st
            coef = torch.linalg.solve(AtA, (Aw.T @ (y * sw).double()).cpu()).to(A.device).float()
            return e["bkg0"] + P["pattern"] + A @ coef, coef

        st = torch.zeros_like(d)
        bk, coef = solve_pe(st)
        if use_stripes:
            for _ in range(3):
                for key in ("row", "col"):
                    r = d - P["src"] - bk - st
                    idx = e[key]
                    num = torch.zeros(2048, device=r.device).index_add_(0, idx, ivar * r)
                    den = torch.zeros(2048, device=r.device).index_add_(0, idx, ivar).clamp(min=1e-12)
                    st = st + (num / den)[idx]
                bk, coef = solve_pe(st)
        m = P["src"] + bk + st
        P.update(m=m, bkg=bk, st=st, pe=coef)
    return m, P


def build_apertures(swarp, sig_b, a, reg, det_bands):
    """The isolated-source aperture catalogue of v36 run(): peaks > 5 sigma of the SWarp-like detection image (copied logic)."""
    Rpx = 0.2 / a.scale
    sw = det_bands
    Dd = ((swarp[sw] - swarp[sw].flatten(1).median(1).values[:, None, None]) / sig_b[sw][:, None, None]).sum(0) / math.sqrt(len(sw))
    k = 2 * int(round(Rpx)) + 1
    mx = F.max_pool2d(Dd[None, None], k, stride=1, padding=k // 2)[0, 0]
    pk = (Dd == mx) & (Dd > 5)
    Hh, Ww = Dd.shape
    mrg = int(math.ceil(2 * Rpx))
    pk[:mrg], pk[-mrg:], pk[:, :mrg], pk[:, -mrg:] = False, False, False, False
    yy_, xx_ = torch.nonzero(pk, as_tuple=True)
    cxy = torch.stack([xx_, yy_], 1).float()
    if cxy.shape[0] > 1:
        d2 = torch.cdist(cxy, cxy)
        d2.fill_diagonal_(1e9)
        cxy = cxy[d2.min(1).values > 2.0 * Rpx]
    lab = torch.full((Hh, Ww), -1, dtype=torch.long, device=Dd.device)
    r = torch.arange(-int(math.ceil(Rpx)), int(math.ceil(Rpx)) + 1, device=Dd.device)
    oy, ox = torch.meshgrid(r, r, indexing="ij")
    disk = (ox ** 2 + oy ** 2) <= Rpx ** 2
    oy, ox = oy[disk], ox[disk]
    for j, (x0, y0) in enumerate(cxy.long().tolist()):
        lab[y0 + oy, x0 + ox] = j
    return lab, cxy, Rpx


def dump_mjd(e):
    """MJD of an exposure for the pixel dumps, as smacs_four_v38 (exposure_mjd): e['mjd'] (HST: EXPSTART), else the SCI header MJD-AVG (JWST,
    mid-exposure), else the primary EXPSTART; nan if none is found."""
    if e.get("mjd") is not None and np.isfinite(e["mjd"]):
        return float(e["mjd"])
    for ext in ("SCI", 0):
        try:
            hd = fits.getheader(e["path"], ext)
            for k in ("MJD-AVG", "MJD-BEG", "EXPSTART"):
                if k in hd:
                    return float(hd[k])
        except Exception:  # noqa: BLE001
            pass
    return float("nan")


def dump_exposure_pixels(dump_dir, e, m, P, a, reg, model, eid):
    """One npz per scored exposure in EXACTLY the key set / dtypes of smacs_four_v38.py --dump-heldout-pixels (patch_v38.dump_exposure_pixels):
    name, band, instrument, det, det_index, split, method, mjd, roll_deg, parity, pixscale_arcsec, gain, noise_scale, row, col, dx, dy, tx, ty (here the
    REGISTERED position x0 + mu of the field operator), tx_hdr, ty_hdr (x0, the header-WCS position), data, model (total = background + stripes +
    source), model_sky (= L + g, the source part), model_bkg (= model - model_sky), sigma (pipeline ERR). gain = mean exp(a) over the pixels
    (the field operator has no single gain); noise_scale = 1 (the learned noise head is not used by the scores)."""
    j = e.get("jac") or V.estimate_jacobian(e)
    f32 = lambda t: t.detach().float().cpu().numpy()  # noqa: E731
    split = "test" if e.get("holdout") else ("val" if e.get("val") else "train")
    stem = Path(e["name"]).name
    stem = stem[:-5] if stem.endswith(".fits") else stem
    np.savez_compressed(
        Path(dump_dir) / f"{stem}.npz", name=e["name"], band=e["band"], instrument=reg.instrument[reg.index[e["band"]]], det=e["detname"], det_index=int(e["det"]),
        split=split, method="opfield", mjd=dump_mjd(e), roll_deg=float(j["theta_deg"]), parity=int(j["parity"]), pixscale_arcsec=float(j["scale"] * a.scale),
        gain=float(torch.exp(P["a"]).mean()), noise_scale=float(e.get("noise_scale", 1.0)),
        row=e["row"].cpu().numpy().astype(np.int32), col=e["col"].cpu().numpy().astype(np.int32), dx=f32(e["dx"]), dy=f32(e["dy"]),
        tx=f32(P["pos"][:, 0]), ty=f32(P["pos"][:, 1]), tx_hdr=f32(e["tx"]), ty_hdr=f32(e["ty"]), data=f32(e["d"]), model=f32(m), model_sky=f32(P["src"]),
        model_bkg=f32(m - P["src"]), sigma=f32(1.0 / e["ivar"].sqrt()))


def evaluate(model, exps, idx_list, a, src_masks, band_index, apertures, reg, steps, gen, lr_bkg, keep_nuis=None, tag="", dump_dir=None):
    """The v36 heldout_chi2 protocol with the nuisance fit of this model (S12). Returns (chi2_all, chi2_src, last, photometry, per_exposure, extra)."""
    tot, n, tot_s, n_s, rob, rob_s = 0.0, 0, 0.0, 0, 0.0, 0.0
    per_band, per_inst, phot_acc, per_exposure = {}, {}, {}, []
    nll_acc = {}
    mk = lambda: {"chi2": 0.0, "robust": 0.0, "n": 0, "chi2_src": 0.0, "robust_src": 0.0, "n_src": 0, "n_exp": 0}  # noqa: E731
    for i in idx_list:
        e = exps[i]
        b = band_index[e["band"]]
        inst = reg.instrument[b]
        nuis = fit_nuisance(model, e, i, a, steps, gen, lr_bkg)
        use_st = bool(a.stripes) and reg.stripes[b]
        m, P = finalize_exposure(model, e, i, nuis, use_st)
        if keep_nuis is not None:
            keep_nuis[i] = (nuis, P)
        if dump_dir is not None:
            dump_exposure_pixels(dump_dir, e, m, P, a, reg, model, i)
        with torch.no_grad():
            r = e["d"] - m
            chi = e["ivar"] * r * r
            d3 = 9.0
            rh = 2.0 * d3 * (torch.sqrt(1.0 + chi / d3) - 1.0)
            tot += float(chi.sum()); rob += float(rh.sum()); n += r.numel()
            acc = per_band.setdefault(e["band"], mk()); acc["chi2"] += float(chi.sum()); acc["robust"] += float(rh.sum()); acc["n"] += r.numel(); acc["n_exp"] += 1
            ia = per_inst.setdefault(inst, mk()); ia["chi2"] += float(chi.sum()); ia["robust"] += float(rh.sum()); ia["n"] += r.numel(); ia["n_exp"] += 1
            pe = {"name": e["name"], "band": e["band"], "det": int(e["det"]), "n": int(r.numel()), "n_src": 0, "chi2_sum": float(chi.sum()),
                  "chi2_src_sum": 0.0, "robust_sum": float(rh.sum()), "robust_src_sum": 0.0, "pulls": [], "instrument": inst}
            per_exposure.append(pe)
            sm = V.sample(src_masks[b][None].float(), e["tx"], e["ty"])[0] > 0.5
            tot_s += float(chi[sm].sum()); rob_s += float(rh[sm].sum()); n_s += int(sm.sum())
            acc["chi2_src"] += float(chi[sm].sum()); acc["robust_src"] += float(rh[sm].sum()); acc["n_src"] += int(sm.sum())
            ia["chi2_src"] += float(chi[sm].sum()); ia["robust_src"] += float(rh[sm].sum()); ia["n_src"] += int(sm.sum())
            pe["n_src"], pe["chi2_src_sum"], pe["robust_src_sum"] = int(sm.sum()), float(chi[sm].sum()), float(rh[sm].sum())
            # learned-noise NLL (reported separately; NOT used for the pipeline-ivar scores)
            sp = 1.0 / e["ivar"].sqrt()
            nll, pout = nll_terms(e["d"], m, sp, P["h"], P["pi"])
            sig = sp * torch.exp(P["h"])
            na = nll_acc.setdefault(inst, {"nll": 0.0, "chi2_learned": 0.0, "n": 0, "pi_mean": 0.0, "n_out05": 0})
            na["nll"] += float(nll.sum()); na["chi2_learned"] += float((r * r / sig ** 2).sum()); na["n"] += r.numel()
            na["pi_mean"] += float(P["pi"].sum()); na["n_out05"] += int((pout > 0.5).sum())
            if apertures is not None:
                lab, cxy, Rpx = apertures
                xg, yg = P["pos"][:, 0], P["pos"][:, 1]
                sky_pred = P["src"]
                data_sky = e["d"] - (m - sky_pred)
                Hh, Ww = lab.shape
                ix = xg.round().long().clamp(0, Ww - 1)
                iy = yg.round().long().clamp(0, Hh - 1)
                sid = lab[iy, ix]
                ok = sid >= 0
                ns = int(cxy.shape[0])
                idx_ = sid[ok]
                if ns and ok.any():
                    bsum = lambda v: torch.zeros(ns, device=v.device, dtype=torch.float64).index_add_(0, idx_, v[ok].double())  # noqa: E731
                    npx = bsum(torch.ones_like(data_sky))
                    fd, fm = bsum(data_sky), bsum(sky_pred)
                    var = bsum(1.0 / e["ivar"].clamp(min=1e-20))
                    xd, xm = bsum(data_sky * xg), bsum(sky_pred * xg)
                    yd, ym = bsum(data_sky * yg), bsum(sky_pred * yg)
                    full = npx >= 0.9 * npx[npx > 0].median() if (npx > 0).any() else npx > 0
                    good = full & (fm > 0) & (fd > 0) & (fm / var.clamp(min=1e-30).sqrt() > 20)
                    if good.any():
                        dm = -2.5 * torch.log10(fd[good] / fm[good])
                        pull = (fd[good] - fm[good]) / var[good].sqrt()
                        dx_ = (xd[good] / fd[good] - xm[good] / fm[good]) * a.scale * 1000
                        dy_ = (yd[good] / fd[good] - ym[good] / fm[good]) * a.scale * 1000
                        Pa = phot_acc.setdefault(e["band"], {"dm": [], "pull": [], "dx": [], "dy": []})
                        Pa["dm"].append(dm.cpu()); Pa["pull"].append(pull.cpu()); Pa["dx"].append(dx_.cpu()); Pa["dy"].append(dy_.cpu())
                        pe["pulls"] = [float(x) for x in pull.cpu().tolist()]
        log(f"{tag}nuisance+score {e['name']} [{inst} {e['band']}]: chi2 {float(chi.mean()):.3f}, robust {float(rh.mean()):.3f}, "
            f"|z| {float(nuis.z.norm()):.2f}, learned NLL {float(nll.mean()):.3f}")
    for bnd, acc in list(per_band.items()) + list(per_inst.items()):
        nn_, ns = max(acc.pop("n"), 1), acc.pop("n_src")
        acc["chi2"] /= nn_; acc["robust"] /= nn_
        acc["chi2_src"] = acc["chi2_src"] / ns if ns else None
        acc["robust_src"] = acc["robust_src"] / ns if ns else None
    nmad_ = lambda v: float(1.4826 * (v - v.median()).abs().median()) if v.numel() else float("nan")  # noqa: E731
    phot, alls = {}, {"dm": [], "pull": [], "dx": [], "dy": []}
    for bnd, Pa in phot_acc.items():
        c_ = {k: torch.cat(v) for k, v in Pa.items()}
        for k in alls:
            alls[k].append(c_[k])
        phot[bnd] = {"n": int(c_["dm"].numel()), "dm_median_mmag": float(c_["dm"].median() * 1000), "dm_nmad_mmag": nmad_(c_["dm"]) * 1000,
                     "pull_nmad": nmad_(c_["pull"]), "centroid_nmad_mas": float(math.hypot(nmad_(c_["dx"]), nmad_(c_["dy"])) / math.sqrt(2))}
    if alls["dm"]:
        c_ = {k: torch.cat(v) for k, v in alls.items()}
        phot["all"] = {"n": int(c_["dm"].numel()), "dm_median_mmag": float(c_["dm"].median() * 1000), "dm_nmad_mmag": nmad_(c_["dm"]) * 1000,
                       "pull_nmad": nmad_(c_["pull"]), "centroid_nmad_mas": float(math.hypot(nmad_(c_["dx"]), nmad_(c_["dy"])) / math.sqrt(2))}
    for k, v in nll_acc.items():
        nn_ = max(v.pop("n"), 1)
        v["nll"] /= nn_; v["chi2_learned"] /= nn_; v["pi_mean"] /= nn_
        v["frac_outlier_posterior_gt_0.5"] = v.pop("n_out05") / nn_
    last = {"robust": rob / max(n, 1), "robust_src": rob_s / max(n_s, 1) if n_s else None, "per_band": per_band, "per_instrument": per_inst}
    return tot / max(n, 1), (tot_s / max(n_s, 1) if n_s else None), last, phot, per_exposure, nll_acc


# ====================================================================================================================== dumps
def kernel_points(model, eid, dx=0.0, dy=0.0, nuis=None):
    """Band-integrated effective kernel of exposure eid at detector position (dx, dy): node offsets [nk,2] in TARGET px, masses [nk]
    (unit sum), exp(a), core second moment. No grad."""
    nuis = model.nuis if nuis is None else nuis
    with torch.no_grad():
        dev = model.sv.device
        e_ = torch.tensor([eid], device=dev)
        h = model.trunk(model.ctx_input(e_, torch.tensor([dx], device=dev), torch.tensor([dy], device=dev), torch.zeros(1, dtype=torch.long, device=dev) if nuis is not model.nuis else e_, nuis))
        mu, Lc, M, a = model.heads(h, e_, (torch.tensor([dx], device=dev), torch.tensor([dy], device=dev), torch.zeros(1, dtype=torch.long, device=dev) if nuis is not model.nuis else e_, nuis))
        c, bi = int(model.m_chan[eid]), model.m_band[eid][None]
        w, cbar, wq = model.shape(c, h, bi, model.m_brow[eid][None])
        nk = model.nk_ch[c]
        wk = (w * wq[:, :, None]).sum(1)[0]
        vdet = model.sv[:nk] - cbar[0][None]
        pos = torch.einsum("ij,nj->ni", M[0], vdet)
        if model.two_scale:
            # NEW: effective kernel = (1 - f_w) core nodes + f_w mod x wing lattice nodes (sky offsets D, TRUE detector px J^-1 D); per-lambda_q masses: the wing is achromatic
            gi = model.m_wgrp[eid][None]
            K, mask = model.wing_kernels(gi)
            dxy = torch.tensor([dx], device=dev), torch.tensor([dy], device=dev)
            fw, modv = model.wing_amp(e_, *dxy)
            lat = wing_lattice(model.wing_nh, dev)[mask[0]]
            Kw = K[0][mask[0]]
            vd_w = torch.einsum("ij,nj->ni", model.wg_Jinv[gi[0]], lat)
            mw = fw * modv * Kw
            rcat = lambda x_, y_: np.concatenate([x_, y_])  # noqa: E731
            return dict(pos=rcat(pos.cpu().numpy(), lat.cpu().numpy()), mass=rcat(((1 - fw) * wk).cpu().numpy(), mw.cpu().numpy()), amp=float(torch.exp(a[0])),
                        mu=mu[0].cpu().numpy(), M=M[0].cpu().numpy(), nc=min(model.stencil.nc, nk), r_det=rcat(model.stencil.r[:nk], vd_w.norm(dim=-1).cpu().numpy()),
                        J=model.m_J[eid].cpu().numpy(), Lc=Lc[0].cpu().numpy(), vdet=rcat(vdet.cpu().numpy(), vd_w.cpu().numpy()),
                        massq=np.concatenate([((1 - fw) * w[0]).cpu().numpy(), (mw[None] * torch.ones_like(w[0][:, :1])).cpu().numpy()], 1),
                        lamq=model.lam_um[bi[0]].cpu().numpy(), wq=wq[0].cpu().numpy())
        return dict(pos=pos.cpu().numpy(), mass=wk.cpu().numpy(), amp=float(torch.exp(a[0])), mu=mu[0].cpu().numpy(), M=M[0].cpu().numpy(),
                    nc=model.stencil.nc, r_det=model.stencil.r[:nk],
                    J=model.m_J[eid].cpu().numpy(), Lc=Lc[0].cpu().numpy(), vdet=vdet.cpu().numpy(), massq=w[0].cpu().numpy(), lamq=model.lam_um[bi[0]].cpu().numpy(), wq=wq[0].cpu().numpy())


def rasterize(pos, mass, half, pitch):
    """Bilinear mass-conserving splat of point masses onto a square grid (offsets in target px, centred): density per target px^2."""
    n = int(round(2 * half / pitch)) + 1
    g = np.zeros((n, n))
    fx = (pos[:, 0] + half) / pitch
    fy = (pos[:, 1] + half) / pitch
    ok = (fx >= 0) & (fx <= n - 1) & (fy >= 0) & (fy <= n - 1)
    x0, y0 = np.floor(fx[ok]).astype(int).clip(0, n - 2), np.floor(fy[ok]).astype(int).clip(0, n - 2)
    wx, wy = fx[ok] - x0, fy[ok] - y0
    m = mass[ok]
    for dxx, dyy, wgt in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        np.add.at(g, (y0 + dyy, x0 + dxx), m * wgt)
    return g / pitch ** 2


def enclosed_energy(pos, mass, radii_px):
    r = np.hypot(pos[:, 0], pos[:, 1])
    return [float(mass[r <= x].sum() / mass.sum()) for x in radii_px]


def make_stencil(a, Rmax):
    """Stencil of the run from the CLI (--wings, --cart-steps; defaults = the original log-polar wings)."""
    return Stencil(a.core_radius, a.core_pitch, a.wing_ratio, a.wing_nang, Rmax, wings=getattr(a, "wings", "logpolar"),
                   cart_steps=parse_cart_steps(getattr(a, "cart_steps", None)))


def load_init_state(model, path, dev, LOG):
    """--eval-only: load opfield_best.pt / opfield_state.pt (a state_dict) or a checkpoint {'model': state_dict}; strict (every tensor, same shapes)."""
    st = torch.load(path, map_location=dev, weights_only=False)
    if isinstance(st, dict) and isinstance(st.get("model"), dict):
        st = st["model"]
    try:
        model.load_state_dict(st, strict=True)
    except RuntimeError as exc:
        raise SystemExit(f"--init-state {path} does not match this model (use the run's own args: --args-from RUN/metrics.json): {str(exc)[:1500]}")
    LOG["eval_only"] = {"init_state": str(path), "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]}
    log(f"loaded {path} (strict)")


def check_split_matches(init_state, LOG):
    """If metrics.json sits next to the state, the held-out / validation / training exposure lists must be identical (same selection args)."""
    mp = Path(init_state).parent / "metrics.json"
    if not mp.exists():
        log(f"WARNING: {mp} not found: the held-out split could not be compared with the run that produced the state")
        LOG["eval_only"]["split_check"] = "no metrics.json next to the state"
        return
    old = json.load(open(mp)).get("split")
    if old is None:
        LOG["eval_only"]["split_check"] = "metrics.json has no split"
        return
    for k in ("holdout", "validation", "train"):
        if sorted(old.get(k, [])) != sorted(LOG["split"][k]):
            raise SystemExit(f"--eval-only: the {k} exposure list differs from the run that produced the state ({len(old.get(k, []))} vs {len(LOG['split'][k])}); "
                             "use the same data and selection args (--args-from RUN/metrics.json)")
    LOG["eval_only"]["split_check"] = "identical train / validation / held-out lists"
    log("held-out / validation / training exposure lists identical to the run that produced the state")


def train_eval_only(model, exps, train_idx, reg, a, LOG):
    """Eval-only replacement of train(): the gauge report (m2_ref unknown: the untrained value is not available; reported as nan)."""
    dev = model.sv.device
    npix = torch.tensor([float(exps[i]["d"].numel()) for i in train_idx], device=dev)
    ref_inst = a.ref_instrument if a.ref_instrument in reg.instruments else max(reg.instruments, key=lambda s: sum(1 for i in train_idx if reg.instrument[reg.index[exps[i]["band"]]] == s))
    ref_mask = torch.tensor([reg.instrument[reg.index[exps[i]["band"]]] == ref_inst for i in train_idx], device=dev)
    LOG["train_steps"], LOG["train_minutes"] = 0, 0.0
    LOG["gauges"] = gauge_report(model, train_idx, npix, ref_mask, reg, a, float("nan"))
    log("GAUGES of the loaded state: " + json.dumps(LOG["gauges"]))
    model.eval()


# ====================================================================================================================== run
def run(exps, thead, dev, a, out, classical_fn=None, reg=None, bandpass=None, preselected=False):
    global LOGF
    out = Path(out)
    (out / "cubes").mkdir(parents=True, exist_ok=True)
    (out / "kernels").mkdir(exist_ok=True)
    LOGF = open(out / "opfield.log", "a")
    LOG = {"args": {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v)) for k, v in vars(a).items()},
           "timings": {}, "metrics": {}, "approximations": APPROXIMATIONS}
    try:
        src_hash = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    except Exception:  # noqa: BLE001
        src_hash = "?"
    log(f"opfield start; torch {torch.__version__}; script sha256 {src_hash[:16]}; device {dev}")
    log("APPROXIMATIONS (all stated in the module docstring):")
    for s in APPROXIMATIONS:
        log("  " + s)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    V.LAM_LO, V.LAM_HI = a.lam_lo, a.lam_hi
    if reg is None:
        V.set_registry(V.registry_from_exposures(exps, psf_n=31))
        reg = OpReg.from_v36(V.REG)
    log("bands: " + ", ".join(f"{b} [{reg.instrument[k]} {reg.channel[k]} {reg.lam_um[k]:.3f}um]" for k, b in enumerate(reg.bands)))
    band_index = dict(reg.index)
    dets = sorted({e["detname"] for e in exps})
    for e in exps:
        e["det"] = dets.index(e["detname"])
    ndet = len(dets)
    if not preselected:
        n_test_arg = getattr(a, "n_test", None)
        sel = V.select_exposures(exps, n_test=1 if n_test_arg is None else n_test_arg, n_val=a.n_val,
                                 holdout_names=(V.read_holdout_names(a.holdout_from) if a.holdout_from else None),
                                 no_holdout=a.no_holdout, bands=reg.bands, fill_missing_bands=n_test_arg is not None)
        for w_ in sel["warnings"]:
            log("selection warning: " + w_)
        json.dump({"holdout": sel["holdout"], "validation": sel["validation"], "warnings": sel["warnings"]}, open(out / "holdout_selection.json", "w"), indent=1)
    train_idx = [i for i, e in enumerate(exps) if not e["holdout"] and not e["val"]]
    hold_idx = [i for i, e in enumerate(exps) if e["holdout"]]
    val_idx = [i for i, e in enumerate(exps) if e["val"]]
    LOG["split"] = {"train": [exps[i]["name"] for i in train_idx], "holdout": [exps[i]["name"] for i in hold_idx], "validation": [exps[i]["name"] for i in val_idx]}
    log(f"exposures: {len(train_idx)} train, {len(val_idx)} validation, {len(hold_idx)} held out; training pixels "
        f"{sum(exps[i]['d'].numel() for i in train_idx) / 1e6:.2f} M")
    for e in exps:
        e["jac"] = V.estimate_jacobian(e)
    if a.coarse_shift > 0 and reg.multi:
        V.args_global = a
        V.coarse_register_hst(exps, thead, dev, a)
    elif a.coarse_shift_jwst > 0:
        V.args_global = a
        V.coarse_register_all(exps, thead, dev, a)

    # ---- classical coadds (sky init, masks, noise)
    t = time.perf_counter()
    cfn = classical_fn or V.classical
    swarp = cfn(exps, thead, "lanczos3", 1.0, dev)
    driz = cfn(exps, thead, "drizzle_square", 0.8, dev)
    cover = dict(getattr(cfn, "last_cover", {}))
    LOG["timings"]["coadds"] = round(time.perf_counter() - t, 1)
    stats = [V.band_median_sigma(c, cover.get(b)) for b, c in zip(reg.bands, swarp)]
    sig_b = torch.stack([s_ for _, s_ in stats]).clamp(min=1e-6)
    src_masks = []
    for (m_, _), b, c, sb in zip(stats, reg.bands, swarp, sig_b):
        sm_ = (c - m_) > 5 * sb
        if cover.get(b) is not None:
            sm_ = sm_ & cover[b]
        src_masks.append(sm_)
    src_masks = torch.stack(src_masks)
    if a.halo_px > 0:
        hp = a.halo_px
        inner = torch.zeros_like(src_masks[0])
        inner[hp:hp + getattr(a, 'size_data', a.size), hp:hp + getattr(a, 'size_data', a.size)] = True
        src_masks = src_masks & inner
    det_bands = [k for k, ch in enumerate(reg.channel) if ch == "NIRCAM_SW"] or sorted(range(len(reg.bands)), key=lambda k: reg.lam_um[k])[:3]
    apertures = build_apertures(swarp, sig_b, a, reg, det_bands)
    log(f"aperture catalogue: {apertures[1].shape[0]} isolated sources")

    # ---- bandpasses, spectral sky
    if bandpass is None:
        lam_nodes, w_nodes, _ = V.load_bandpasses(reg.bands, a.spec_q, a.svo_cache)
    else:
        lam_nodes, w_nodes = bandpass
    lam_nodes, w_nodes = lam_nodes.to(dev), w_nodes.to(dev)
    basis = V.RBFSpectralBasis(a.spec_k, learned=False).to(dev)
    A0 = V.spectral_init(driz, lam_nodes, w_nodes, basis)
    Phi_nodes = basis(lam_nodes.flatten()).view(len(reg.bands), -1, a.spec_k).detach()

    # ---- operator
    meta, mjd, tex = build_meta(exps, reg, dev)
    scale_ch = channel_scales(exps, reg, meta["band"], a.scale)
    Rmax = max((a.two_scale if a.two_scale > 0 else a.wing_arcsec) / s["arcsec_per_det_px"] for s in scale_ch)   # --two-scale: the per-pixel stencil ends at R_core
    stencil = make_stencil(a, Rmax)
    if a.sky_c2f > 0:   # S20: the stale high frequencies of the (mis-registered) drizzle initial sky must not return when the band limit is released
        A0 = gauss_blur_img(A0, a.sky_c2f_arcsec / a.scale)
    model = OpField(a, reg, meta, stencil, A0, Phi_nodes, w_nodes, lam_nodes, ndet, scale_ch, dev, det_names=dets)
    moves = model.set_train_rows(train_idx)
    if moves:
        log(f"base rows with no training exposure (--base-per {model.base_per}) aliased to a trained row: " + "; ".join(moves))
    LOG["base_rows"] = {"base_per": model.base_per, "n_rows": model.n_rows, "labels": [model.row_label(r) for r in range(model.n_rows)],
                        "aliased_untrained": moves, "exposures_per_row_train": {model.row_label(r): int(sum(1 for i in train_idx if int(model.m_brow[i]) == r)) for r in range(model.n_rows)}}
    if getattr(a, "base_init_from", None) and not a.eval_only:
        rp = model.init_base_from_e2(a.base_init_from, getattr(a, "base_init_key", "learned"))
        LOG["base_init_from"] = rp
        log(f"base rows initialised from E2 tables ({rp['file']}, key {rp['key']}): matched {len(rp['matched'])}/{model.n_rows}; "
            f"unmatched (kept at the Gaussian start): {rp['unmatched']}; table mass inside the stencil per cell (min/median): "
            + (f"{min(rp['mass_in_stencil'].values()):.3f}/{float(np.median(list(rp['mass_in_stencil'].values()))):.3f}" if rp["mass_in_stencil"] else "n/a"))
    LOG["stencil"] = {"wings": stencil.wings, "core_points": stencil.nc, "rings": stencil.n_rings, "points_total": len(stencil),
                      "per_channel": {ch: {"nk": model.nk_ch[c], "wing_radius_det_px": round(model.R_ch[c], 1),
                                           "wing_radius_arcsec": a.wing_arcsec, "sky_per_det_px": round(scale_ch[c]["sky_per_det_px"], 4),
                                           "level_slices": model.slices_ch[c]} for c, ch in enumerate(reg.channels)}}
    if model.two_scale:
        for c, ch in enumerate(reg.channels):
            LOG["stencil"]["per_channel"][ch]["per_pixel_stencil_radius_arcsec"] = a.two_scale   # --two-scale: 'wing_radius_det_px' above is the radius of the PER-PIXEL stencil (R_core)
    log("STENCIL (" + stencil.wings + "): core " + str(stencil.nc) + " points (disk r<=" + str(a.core_radius) + " px, pitch " + str(a.core_pitch) + "), "
        + (str(stencil.n_rings) + " rings x " + str(a.wing_nang) + " = " + str(stencil.n_rings * a.wing_nang) + " wing points" if stencil.wings == "logpolar"
           else str(len(stencil) - stencil.nc) + " Cartesian wing points, steps (outer radius, pitch) " + str(parse_cart_steps(getattr(a, "cart_steps", None)))) + "; per channel: "
        + "; ".join(f"{ch} nk={model.nk_ch[c]} R={model.R_ch[c]:.0f} px, levels {[(l, e - s) for l, s, e in model.slices_ch[c]]}" for c, ch in enumerate(reg.channels)))
    wing_mb = 0.0
    if model.two_scale:
        wing_mb = estimate_wing_memory_mb(len(model.wing_groups), model.wing_Hp, model.wing_Wp, model.wing_chunk, len(reg.bands))
        LOG["two_scale"] = {"r_core_arcsec": a.two_scale, "wing_arcsec": a.wing_arcsec, "table_nodes": 2 * model.wing_nh + 1, "table_pitch": "1 sky px (a.scale) in the detector frame",
                            "rows": [f"{model.det_names[k[0]]}/{reg.bands[k[1]]}" for k in model.wing_keys], "aliased_untrained": model.untrained_wing_cells,
                            "n_groups": len(model.wing_groups), "groups": model.wing_groups, "fft_size": [model.wing_Hp, model.wing_Wp], "wing_mod_degree": a.wing_mod,
                            "estimated_wing_memory_mb": round(wing_mb, 1),
                            "core_nk_per_channel": {ch: {"nk": model.nk_ch[c], "r_core_det_px": round(model.R_ch[c], 2)} for c, ch in enumerate(reg.channels)}}
        log(f"TWO-SCALE: per-pixel stencil r <= {a.two_scale}\" ({', '.join(f'{ch}: {model.R_ch[c]:.1f} det px, nk {model.nk_ch[c]}' for c, ch in enumerate(reg.channels))}); wings {a.two_scale}\"-{a.wing_arcsec}\": "
            f"{model.n_wrows} tables ({2 * model.wing_nh + 1}^2 nodes at 1 sky px), {len(model.wing_groups)} roll groups, FFT {model.wing_Hp}x{model.wing_Wp}, chunk {model.wing_chunk}, "
            f"estimated wing memory {wing_mb:.0f} MB, f_w start " + ", ".join(f"{float(torch.sigmoid(model.wing_logit[r_])):.1e}" for r_ in range(model.n_wrows)))
        if model.untrained_wing_cells:
            log("wing tables with no training exposure aliased to a trained one: " + "; ".join(model.untrained_wing_cells))
    n_par = sum(p.numel() for p in model.parameters())
    log(f"parameters: {n_par / 1e6:.2f} M total; sky {model.sky.numel() / 1e6:.2f} M; operator "
        f"{sum(p.numel() for n_, p in model.named_parameters() if not n_.startswith(('sky', 'nuis', 'pd'))) / 1e3:.1f} k")
    LOG["param_counts"] = model.param_counts()
    log("parameter counts: " + json.dumps(LOG["param_counts"]))

    unit = float(np.mean([float(model.m_unit[i]) for i in train_idx]))
    lr_bkg = a.lr_bkg if a.lr_bkg > 0 else 0.1 * unit
    lr_sky = a.lr_sky if a.lr_sky > 0 else 0.3 * float(sig_b.mean())
    if a.eval_only:
        pool, B = None, 0
        LOG["batch"] = 0
        log("EVAL ONLY: no pool, no training")
        load_init_state(model, a.init_state, dev, LOG)
        check_split_matches(a.init_state, LOG)
        train_eval_only(model, exps, train_idx, reg, a, LOG)
    else:
        pool = Pool(exps, train_idx, src_masks, band_index, dev)
        log(f"pool: {pool.N / 1e6:.2f} M pixels, {int(pool.bright_idx.numel()) / 1e6:.2f} M bright (coadd S/N > 5)")
    # batch
    if a.eval_only:
        pass
    elif a.batch > 0:
        B = int(a.batch)
    elif dev.type == "cuda":
        free = torch.cuda.mem_get_info()[0] / 2 ** 20
        B = 1024
        while B < 32768 and estimate_batch_memory_mb(B * 2, model.nk_max, a.spec_q, a.spec_k, hw=A0.shape[-2:], H=a.hidden) + wing_mb < 0.5 * free:
            B *= 2
    else:
        B = 2048
    if not a.eval_only:
        log(f"batch {B} pixels; estimated peak memory {estimate_batch_memory_mb(B, model.nk_max, a.spec_q, a.spec_k, hw=A0.shape[-2:], H=a.hidden):.0f} MB")
        LOG["batch"] = B

    # ---- training
    def restart_fn(idx_set):
        """S22: sky coefficients from a v36 tent-splat coadd of the exposures in idx_set at their LEARNED positions x0 + mu, background / stripes
        removed and divided by exp(a); no catalogue, no cross-correlation."""
        idx_set = set(idx_set)
        es = []
        with torch.no_grad():
            lv = model.make_levels(0.0)
            for i, e in enumerate(exps):
                e2 = dict(e)
                if i not in idx_set:
                    e2["holdout"], e2["val"] = True, False
                    es.append(e2)
                    continue
                N = e["d"].numel()
                pos, corr = [], []
                for s0 in range(0, N, 4096):
                    bt = exp_batch(e, i, slice(s0, min(N, s0 + 4096)), nid=i)
                    o = model(bt, model.nuis, lv)
                    pos.append(o["pos"])
                    corr.append((o["bkg"] + o["st"], torch.exp(o["a"])))
                pos = torch.cat(pos)
                bk = torch.cat([c_[0] for c_ in corr])
                ga = torch.cat([c_[1] for c_ in corr])
                e2["tx"], e2["ty"] = pos[:, 0].contiguous(), pos[:, 1].contiguous()
                e2["d"] = (e["d"] - bk) / ga + e["bkg0"]
                e2["ivar"] = e["ivar"] * ga ** 2
                e2["holdout"], e2["val"] = False, False
                es.append(e2)
            imgs = []
            for b in reg.bands:
                img, _ = V.splat_coadd([e2 for e2 in es if e2["band"] == b and not e2["holdout"] and not e2.get("val")], thead, dev)
                imgs.append(torch.nan_to_num(img).float())
            return V.spectral_init(torch.stack(imgs), lam_nodes, w_nodes, basis)

    if not a.eval_only:
        train(model, pool, exps, train_idx, val_idx, a, B, lr_sky, lr_bkg, sig_b, src_masks, band_index, apertures, reg, out, LOG, dev, restart_fn=restart_fn)
    if model.fact and not a.eval_only:
        with torch.no_grad():
            for c_ in range(model.nch):
                ids_ = [i for i in train_idx if int(model.m_chan[i]) == c_]
                if ids_:
                    model.lg_class[c_] = model.nuis.lg[ids_].mean()

    # ---- evaluation
    keep = {}
    score_idx = train_idx if a.score_all else hold_idx
    LOG["scored_exposures"] = "all (in-sample)" if a.score_all else "test (left out)"
    gen = torch.Generator(device=dev).manual_seed(1234)
    t = time.perf_counter()
    if score_idx and not a.skip_eval:
        dump_dir = None
        if a.dump_heldout_pixels:
            dump_dir = out / "heldout_pixels" / "opfield"
            dump_dir.mkdir(parents=True, exist_ok=True)
        c2, c2s, L, phot, per_exp, nlls = evaluate(model, exps, score_idx, a, src_masks, band_index, apertures, reg, a.nuis_steps, gen, lr_bkg, keep_nuis=keep, tag="TEST ",
                                                   dump_dir=dump_dir)
        if dump_dir is not None:
            log(f"--dump-heldout-pixels: {len(list(dump_dir.glob('*.npz')))} exposure files written to {dump_dir}")
        M = LOG["metrics"]
        M["heldout_chi2"] = {"opfield": c2}
        M["heldout_chi2_sources"] = {"opfield": c2s}
        M["heldout_robust"] = {"opfield": L["robust"]}
        M["heldout_robust_sources"] = {"opfield": L["robust_src"]}
        M["heldout_per_band"] = {"opfield": L["per_band"]}
        M["heldout_per_instrument"] = {"opfield": L["per_instrument"]}
        M["heldout_photometry"] = {"opfield": phot}
        M["heldout_per_exposure"] = {"opfield": [{k: v for k, v in r_.items() if k != "pulls"} for r_ in per_exp]}
        M["heldout_learned_noise"] = {"opfield": nlls}
        try:
            bs = V.bootstrap_scores(per_exp, B=a.boot_b, seed=0)
            M["heldout_bootstrap"] = {"opfield": bs}
            log(V.boot_line("opfield", bs))
        except Exception as exc:  # noqa: BLE001
            log(f"bootstrap failed: {exc!r}")
        f_ = lambda v: "nan" if v is None else f"{v:.4f}"  # noqa: E731
        for b_, v in L["per_band"].items():
            log(f"TEST-BAND opfield {b_} [{reg.instrument[reg.index[b_]]}] (n_exp {v['n_exp']}): chi2 all {f_(v['chi2'])} src {f_(v['chi2_src'])} "
                f"robust all {f_(v['robust'])} src {f_(v['robust_src'])}")
        log(f"TEST opfield: chi2 all {c2:.4f} src {f_(c2s)} | robust all {L['robust']:.4f} src {f_(L['robust_src'])}"
            + "".join(f" | {ins} (n_exp {v['n_exp']}): chi2 all {f_(v['chi2'])} src {f_(v['chi2_src'])} robust all {f_(v['robust'])} src {f_(v['robust_src'])}" for ins, v in L["per_instrument"].items()))
        for ins, v in nlls.items():
            log(f"LEARNED-NOISE {ins}: NLL {v['nll']:.4f}/px, chi2 with learned sigma {v['chi2_learned']:.4f}, mean pi {v['pi_mean']:.4f}, outlier posterior>0.5 {v['frac_outlier_posterior_gt_0.5']:.4f}")
    LOG["timings"]["evaluation"] = round(time.perf_counter() - t, 1)

    # ---- outputs
    write_outputs(model, exps, train_idx, keep, thead, a, reg, out, LOG, dev, sig_b)
    json.dump(LOG, open(out / "metrics.json", "w"), indent=1, default=float)
    log("done")
    LOGF.close()
    LOGF = None
    return dict(model=model, exps=exps, train_idx=train_idx, hold_idx=hold_idx, val_idx=val_idx, keep=keep, log=LOG, reg=reg, sig_b=sig_b,
                src_masks=src_masks, lr_bkg=lr_bkg, driz=driz, swarp=swarp)


def gauge_report(model, train_idx, npix, ref_mask, reg, a, m2_ref):
    with torch.no_grad():
        eids = torch.tensor(train_idx, device=model.sv.device)
        mu_bar, a_ref, m2 = model.gauge_terms(eids, npix, ref_mask)
        rep = {"mean_mu_target_px": [float(x) for x in mu_bar], "mean_a_ref_instrument": float(a_ref), "m2_core_mean_target_px2": float(m2.mean()),
               "m2_core_ref_target_px2": float(m2_ref)}
        by = {}
        for k, i in enumerate(train_idx):
            b = int(model.m_band[i])
            by.setdefault(f"{reg.instrument[b]}/{reg.bands[b]}", []).append(float(m2[k]))
        rep["core_second_moment_sigma_arcsec"] = {key: float(np.sqrt(np.mean(v) / 2.0) * a.scale) for key, v in by.items()}  # per-axis sigma
        return rep


def xfit_step(model, pool, a, views, K, sc, p, noise_start, B, gen, sig_b, sig_mean, tr, npix, ref_mask, m2_ref, unit_sig, opt):
    """One cross-fitted step (S21). Pass 1 per fold f: loss against the fold's OWN view, gradient to that view only. Pass 2 per fold f: loss against
    the OTHER view(s), detached, gradient to registration / operator / noise. Returns the pieces the training loop logs."""
    lv_own = [model.make_levels(sc, sky=v) for v in views]
    lv_det = [[None if t is None else t.detach() for t in lv] for lv in lv_own]
    bt = pool.draw(B, a.bright_frac, gen)
    fold = model.m_fold[bt["eid"]]
    n_b = fold.shape[0]
    opt.zero_grad(set_to_none=True)
    nlls, pis, lraws, d2 = [], [], [], 0.0
    gviews = []
    for f in range(K):
        idx = torch.nonzero(fold == f).squeeze(1)
        if idx.numel() == 0:
            gviews.append(None)
            continue
        sub = {k_: (v_[idx] if torch.is_tensor(v_) and v_.shape[:1] == (n_b,) else v_) for k_, v_ in bt.items()}
        # pass 1: sky fit of view f on its own fold
        o1 = model(sub, model.nuis, lv_own[f])
        if p < noise_start:
            n1, _ = nll_terms(sub["d"], o1["pred"], sub["sp"], torch.zeros_like(o1["h"]), torch.full_like(o1["pi"], PI_INIT))
        else:
            n1, _ = nll_terms(sub["d"], o1["pred"], sub["sp"], o1["h"], o1["pi"])
        S = model.band_image(sky=views[f])
        l1 = 2.0 * n1.sum() / n_b + (a.lam_tv * V.tv(S, sig_b) + (a.lam_neg * F.relu(-S / sig_b[:, None, None] - 2.0).pow(2).mean() if a.lam_neg > 0 else 0.0)) / K
        gviews.append(torch.autograd.grad(l1, [views[f]])[0])
        # pass 2: registration against the other view(s), sky detached
        if K == 2:
            lv_o = lv_det[1 - f]
        else:
            oth = torch.stack([views[g].detach() for g in range(K) if g != f]).mean(0)
            lv_o = model.make_levels(sc, sky=oth)
        o2 = model(sub, model.nuis, lv_o)
        if p < noise_start:
            n2, _ = nll_terms(sub["d"], o2["pred"], sub["sp"], torch.zeros_like(o2["h"]), torch.full_like(o2["pi"], PI_INIT))
        else:
            n2, _ = nll_terms(sub["d"], o2["pred"], sub["sp"], o2["h"], o2["pi"])
        d2 = d2 + 2.0 * n2.sum() / n_b
        nlls.append(n2.detach()); pis.append(o2["pi"].detach()); lraws.append(o2["lraw"])
    pri = model.prior_terms(tr)
    mu_bar, a_ref, m2 = model.gauge_terms(tr, npix, ref_mask)
    gp = a.gauge_shift * (mu_bar ** 2).sum() + a.gauge_flux * a_ref ** 2 + a.gauge_blur * m2.mean() / m2_ref
    lraw = torch.cat(lraws)
    reg_t = gp + a.lc_l2 * (torch.tanh(lraw) ** 2).mean() + a.z_l2 * pri["z"] + a.g_l2 * pri["g"] + a.bank_l2 * pri["bank"] + a.base_l2 * pri["base"]
    reg_t = reg_t + model.nuis.stripes.penalty(unit_sig)
    (d2 + reg_t).backward()
    for v, g in zip(views, gviews):
        v.grad = g
    opt.step()
    nll = torch.cat(nlls)
    return d2.detach(), nll, dict(pi=torch.cat(pis)), gp, mu_bar, a_ref, m2


def train(model, pool, exps, train_idx, val_idx, a, B, lr_sky, lr_bkg, sig_b, src_masks, band_index, apertures, reg, out, LOG, dev, restart_fn=None):
    tr = torch.tensor(train_idx, device=dev)
    K_x = int(a.xfit_reg)
    views, switched = [], False
    if K_x >= 2:
        keys = {}
        for i in train_idx:
            keys.setdefault((int(model.m_chan[i]), round(exps[i]['jac']['theta_deg'])), []).append(i)
        per_ch = {}
        for k_, ids_ in sorted(keys.items()):
            per_ch.setdefault(k_[0], []).append(ids_)
        for ch_, groups_ in per_ch.items():
            if len(groups_) >= K_x:
                for n_, ids_ in enumerate(groups_):
                    for i in ids_:
                        model.m_fold[i] = (n_ + ch_) % K_x
            else:   # fewer roll groups than folds: alternate the exposures themselves (near-duplicates are then split; stated in S21)
                for n_, i in enumerate([i for ids_ in groups_ for i in ids_]):
                    model.m_fold[i] = (n_ + ch_) % K_x
        log('xfit folds: ' + str({f: sum(int((model.m_fold[i] == f)) for i in train_idx) for f in range(K_x)}) + ' training exposures per fold; groups ' + str(len(keys)))
        views = [nn.Parameter(model.sky.detach().clone()) for _ in range(K_x)]
    npix = torch.tensor([float(exps[i]["d"].numel()) for i in train_idx], device=dev)
    ref_inst = a.ref_instrument if a.ref_instrument in reg.instruments else max(reg.instruments, key=lambda s: sum(1 for i in train_idx if reg.instrument[reg.index[exps[i]["band"]]] == s))
    if ref_inst != a.ref_instrument:
        log(f"WARNING: reference instrument {a.ref_instrument} absent; flux gauge uses {ref_inst}")
    ref_mask = torch.tensor([reg.instrument[reg.index[exps[i]["band"]]] == ref_inst for i in train_idx], device=dev)
    with torch.no_grad():
        _, _, m2_0 = model.gauge_terms(tr, npix, ref_mask)
    m2_ref = float(m2_0.mean().clamp(min=1e-6))
    unit_sig = float(model.m_unit[tr].mean())
    ops = [p for n_, p in model.named_parameters() if not n_.startswith(("sky", "nuis", "pd", "base", "bank", "mu_skip", "Ze", "Zv", "wing"))]
    groups = [{"params": [model.sky], "lr": lr_sky}, {"params": ops, "lr": a.lr_op}, {"params": model.kernel_params(), "lr": a.lr_kernel},
              {"params": [model.nuis.z] + ([model.Ze, model.Zv] if model.zh else []), "lr": a.lr_z}]
    if model.two_scale:   # NEW: wing tables, fractions and modulation at lr-kernel x --wing-lr-mult (S35)
        groups.append({"params": [p_ for p_ in (model.wing_tab, model.wing_logit, model.wing_mod) if p_ is not None], "lr": a.lr_kernel * a.wing_lr_mult})
    sampler = None
    if getattr(a, "importance_sampling", None):   # NEW (--importance-sampling): S36
        sampler = PeakSampler(pool, model, parse_is_radii(a.importance_sampling), a, sig_b)
        log(f"importance sampling: annuli (detector px) {[0.0] + parse_is_radii(a.importance_sampling)}, fraction {a.is_frac} of the batch, refresh every {a.is_refresh} steps, "
            f"peaks above S/N {a.is_snr:g} (max {a.is_max_peaks} per band); loss reweighted by 1 / (N q) (unbiased for the all-pixel loss, S10 bright-bias replaced)")
    if model.use_skip:
        groups.append({"params": list(model.mu_skip.parameters()), "lr": a.lr_mu_skip})
    if model.fact:
        groups += [{"params": [model.nuis.aff], "lr": a.lr_aff}, {"params": [model.pdist], "lr": a.lr_dist}, {"params": [model.nuis.lg], "lr": a.lr_gain}]
    groups += [{"params": [model.nuis.pe, model.pd] + list(model.nuis.stripes.parameters()), "lr": lr_bkg}]
    if views:
        groups.append({"params": views, "lr": lr_sky})
    opt = torch.optim.Adam(groups)
    base_lr = [g["lr"] for g in groups]
    steps = a.steps if a.steps > 0 else 10 ** 12
    budget = a.budget_min * 60.0
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    ck = out / "opfield_ckpt.pt"
    it0, el0, hist = 0, 0.0, []
    if a.resume and ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        it0, el0, hist = st["it"] + 1, st["elapsed"], st["hist"]
        log(f"resumed at step {it0}, {el0 / 60:.1f} min done")
    t_start = time.perf_counter() - el0
    t_ck = t_val = time.perf_counter()
    best, bad = (float("inf"), None), 0
    val_every = a.val_every_min * 60.0

    def set_lr(frac):
        wu = 0.03
        f = max(frac, 1e-3) / wu if frac < wu else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - wu) / (1 - wu))))
        for k, (g, lr) in enumerate(zip(opt.param_groups, base_lr)):
            g["lr"] = lr * f * (0.0 if (k == 0 and frac < a.sky_warmup) else 1.0)
    prog = lambda it, el: max(it / steps, el / budget if budget > 0 else 0.0)  # noqa: E731
    set_lr(prog(it0, el0))
    model.train()
    log(f"training: budget {a.budget_min} min, max steps {a.steps or 'inf'}, batch {B}, lr sky {lr_sky:.3g} op {a.lr_op} kernel {a.lr_kernel} z {a.lr_z} bkg {lr_bkg:.3g}, "
        f"c2f {a.c2f_start_arcsec}\" -> 0 over the first {a.c2f_frac} of the budget")
    it = it0
    sig_mean = float(sig_b.mean())
    noise_start = a.c2f_frac if a.noise_start < 0 else a.noise_start
    hook = getattr(a, 'hook', None)
    n_rs = int(a.restart_cycles)
    rs_pts = [] if n_rs <= 0 else ([0.5 * (a.restart_first + a.restart_last)] if n_rs == 1 else list(np.linspace(a.restart_first, a.restart_last, n_rs)))
    rs_i = 0
    accum, ess_acc = [], []
    while True:
        el = time.perf_counter() - t_start
        p = prog(it, el)
        if p >= 1.0:
            break
        sc = c2f_sigma(p, a)
        if restart_fn is not None and rs_i < len(rs_pts) and p >= rs_pts[rs_i]:
            rs_i += 1
            t_rs = time.perf_counter()
            if views and not switched:
                for g_, v_ in enumerate(views):
                    with torch.no_grad():
                        v_.copy_(restart_fn([i for i in train_idx if int(model.m_fold[i]) == g_]))
                    opt.state.pop(v_, None)
            else:
                with torch.no_grad():
                    model.sky.copy_(restart_fn(train_idx))
                opt.state.pop(model.sky, None)
            log(f"RESTART {rs_i}/{len(rs_pts)} at step {it} (p {p:.2f}): sky re-initialised from the learned-mu coadd ({time.perf_counter() - t_rs:.0f} s)")
        use_x = K_x >= 2 and p < a.xfit_frac
        if K_x >= 2 and not use_x and not switched:
            with torch.no_grad():
                model.sky.copy_(torch.stack([v for v in views]).mean(0))
            switched = True
            log('xfit: views averaged into the sky; normal training continues')
        if use_x:
            data, nll, out_, gp, mu_bar, a_ref, m2 = xfit_step(model, pool, a, views, K_x, sc, p, noise_start, B, gen, sig_b, sig_mean, tr, npix, ref_mask, m2_ref, unit_sig, opt)
        else:
            levels = model.make_levels(sc)
            if sampler is not None:
                if (it - it0) % max(1, int(a.is_refresh)) == 0:
                    sampler.refresh(it)
                bt = sampler.draw(B, a.bright_frac, gen)
                ess_acc.append((sampler.last["ess"], sampler.last["n_is"], sampler.last["w_max"]))
            else:
                bt = pool.draw(B, a.bright_frac, gen)
            out_ = model(bt, model.nuis, levels)
            if p < noise_start:   # noise head off: sigma = sigma_pipe, pi = initial (the misfit of an unregistered model must not be excused by h)
                nll, _ = nll_terms(bt["d"], out_["pred"], bt["sp"], torch.zeros_like(out_["h"]), torch.full_like(out_["pi"], PI_INIT))
            else:
                nll, _ = nll_terms(bt["d"], out_["pred"], bt["sp"], out_["h"], out_["pi"])
            data = 2.0 * (nll.mean() if "w" not in bt else (bt["w"] * nll).mean())
            S = model.band_image()
            reg_t = a.lam_tv * V.tv(S, sig_b)
            if a.lam_neg > 0:
                reg_t = reg_t + a.lam_neg * F.relu(-S / sig_b[:, None, None] - 2.0).pow(2).mean()
            if a.spec_smooth > 0:
                A = model.sky
                reg_t = reg_t + a.spec_smooth * ((A[2:] - 2 * A[1:-1] + A[:-2]) / sig_mean).pow(2).mean()
            pri = model.prior_terms(tr)
            mu_bar, a_ref, m2 = model.gauge_terms(tr, npix, ref_mask)
            gp = a.gauge_shift * (mu_bar ** 2).sum() + a.gauge_flux * a_ref ** 2 + a.gauge_blur * m2.mean() / m2_ref
            reg_t = reg_t + gp + a.lc_l2 * (torch.tanh(out_["lraw"]) ** 2).mean() + a.z_l2 * pri["z"] + a.g_l2 * pri["g"] + a.bank_l2 * pri["bank"] + a.base_l2 * pri["base"]
            reg_t = reg_t + model.nuis.stripes.penalty(unit_sig)
            if model.two_scale:
                reg_t = reg_t + a.wing_l2 * pri["wing"]
            loss = data + reg_t
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        set_lr(prog(it + 1, time.perf_counter() - t_start))
        accum.append((float(data), float(nll.mean()), float(out_["pi"].mean()), float(gp.detach())))
        now = time.perf_counter()
        if it % a.log_every == 0:
            ac = np.mean(accum, 0)
            accum = []
            hist.append([it, float(ac[0])])
            log(f"step {it} ({(now - t_start) / 60:.1f} min, p {p:.2f}) 2NLL {ac[0]:.4f} NLL {ac[1]:.4f} pi {ac[2]:.4f} c2f {sc:.2f}px gauge-pen {ac[3]:.2e} "
                f"|mu_bar| {float(mu_bar.norm()):.3f}px a_ref {float(a_ref):+.3f} m2 {float(m2.mean()):.3f}px2")
            if ess_acc:
                ea = np.mean(ess_acc, 0)
                ess_acc = []
                sampler.ess_hist.append([it, float(ea[0])])
                log(f"   IS: effective sample size {ea[0]:.0f} of {B} ({100 * ea[0] / B:.0f}%), {ea[1]:.0f} stratified draws per batch, max weight {ea[2]:.2f}")
            if model.two_scale:
                with torch.no_grad():
                    log("   WING f_w (sigmoid) per (detector/band): " + ", ".join(f"{model.det_names[k[0]]}/{reg.bands[k[1]]} {float(torch.sigmoid(model.wing_logit[r_])):.2e}" for r_, k in enumerate(model.wing_keys)))
        if hook is not None and it % a.log_every == 0:
            hook(it, model)
        if now - t_ck > a.ckpt_every_min * 60:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "elapsed": now - t_start, "hist": hist}, ck)
            t_ck = now
        if val_idx and val_every > 0 and now - t_val > val_every and p > a.c2f_frac:
            g2 = torch.Generator(device=dev).manual_seed(77)
            _, _, L, _, _, _ = evaluate(model, exps, val_idx, a, src_masks, band_index, None, reg, a.val_nuis_steps, g2, lr_bkg, tag="VAL ")
            score = L["robust"] + (L["robust_src"] or 0.0)
            t_val = time.perf_counter()
            log(f"VAL at step {it} ({(t_val - t_start) / 60:.1f} min): robust all {L['robust']:.4f} src {L['robust_src']} score {score:.4f}")
            LOG.setdefault("val_history", []).append([it, (t_val - t_start) / 60, score])
            if score < best[0]:
                best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()})
                bad = 0
                torch.save(best[1], out / "opfield_best.pt")
                log("new best validation checkpoint")
            else:
                bad += 1
                if a.patience and bad >= a.patience:
                    log(f"no validation improvement in {bad} checks: early stop")
                    break
        it += 1
    # final validation if any, then restore best
    if val_idx:
        g2 = torch.Generator(device=dev).manual_seed(77)
        _, _, L, _, _, _ = evaluate(model, exps, val_idx, a, src_masks, band_index, None, reg, a.val_nuis_steps, g2, lr_bkg, tag="VAL(final) ")
        score = L["robust"] + (L["robust_src"] or 0.0)
        LOG.setdefault("val_history", []).append([it, (time.perf_counter() - t_start) / 60, score])
        if best[1] is not None and score >= best[0]:
            model.load_state_dict(best[1])
            log(f"restored best validation checkpoint (score {best[0]:.4f} vs final {score:.4f})")
        LOG["val_best_score"] = min(best[0], score)
    LOG["train_steps"] = it
    if sampler is not None:
        eh = [x[1] for x in sampler.ess_hist]
        LOG["importance_sampling"] = {"radii_det_px": parse_is_radii(a.importance_sampling), "frac": a.is_frac, "refresh": a.is_refresh, "snr": a.is_snr,
                                      "ess_mean": float(np.mean(eh)) if eh else None, "ess_min": float(np.min(eh)) if eh else None, "batch": B,
                                      "refreshes": sampler.history[-20:], "n_refreshes": sampler.n_refresh, "ess_history": sampler.ess_hist[-200:]}
    LOG["train_minutes"] = round((time.perf_counter() - t_start) / 60, 2)
    LOG["gauges"] = gauge_report(model, train_idx, npix, ref_mask, reg, a, m2_ref)
    log("GAUGES at the end: " + json.dumps(LOG["gauges"]))
    torch.save(model.state_dict(), out / "opfield_state.pt")
    model.eval()


def write_outputs(model, exps, train_idx, keep, thead, a, reg, out, LOG, dev, sig_b):
    with torch.no_grad():
        hdr = fits.Header(thead)
        S = model.band_image()
        fits.writeto(out / "cubes" / "opfield_sky.fits", S.cpu().numpy().astype(np.float32), hdr, overwrite=True)
        fits.writeto(out / "cubes" / "opfield_sky_coeffs.fits", model.sky.cpu().numpy().astype(np.float32), hdr, overwrite=True)
        # effective kernels at the detector centre of one training exposure per (instrument, detector, band, roll rounded to 1 deg)
        groups, rows = {}, {}
        for i in train_idx:
            e = exps[i]
            key = (reg.instrument[reg.index[e["band"]]], e["detname"], e["band"], round(e["jac"]["theta_deg"]))
            groups.setdefault(key, []).append(i)
        radii_arcsec = np.array([0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0])
        npz, meta_rows = {}, []
        for key, ids in groups.items():
            i = ids[0]
            kp = kernel_points(model, i)
            name = "|".join(str(k) for k in key)
            ee = enclosed_energy(kp["pos"], kp["mass"], radii_arcsec / a.scale)
            g = rasterize(kp["pos"], kp["mass"], a.kernel_half_px, a.kernel_pitch_px)
            core_m2 = float((kp["mass"][: kp["nc"]] * (kp["pos"][: kp["nc"]] ** 2).sum(1)).sum() / kp["mass"][: kp["nc"]].sum())
            full_m2 = float((kp["mass"] * (kp["pos"] ** 2).sum(1)).sum())
            npz[name + "|grid"] = g.astype(np.float32)
            npz[name + "|pos"] = kp["pos"].astype(np.float32)
            npz[name + "|mass"] = kp["mass"].astype(np.float32)
            # NEW (compare_kernels.py): detector-frame node offsets (v - cbar, detector px; [:, 0] = column, [:, 1] = row), the sky->detector matrix M
            # (pos = M vdet), per-lambda_q masses (unit sum each), the lambda_q [um] and their band weights. Old keys unchanged.
            npz[name + "|vdet"] = kp["vdet"].astype(np.float32)
            npz[name + "|M"] = kp["M"].astype(np.float32)
            # NEW: J (exposure detector -> target-px Jacobian; M = J Lambda) and Lambda: TRUE detector-px node offsets = J^-1 pos = Lambda vdet
            # ('vdet' is the CANONICAL stencil coordinate, NOT detector px unless Lambda = I; compare_kernels.py uses J)
            npz[name + "|J"] = kp["J"].astype(np.float32)
            npz[name + "|Lc"] = kp["Lc"].astype(np.float32)
            npz[name + "|massq"] = kp["massq"].astype(np.float32)
            npz[name + "|lamq"] = kp["lamq"].astype(np.float32)
            npz[name + "|wq"] = kp["wq"].astype(np.float32)
            meta_rows.append({"group": name, "exposure": exps[i]["name"], "instrument": key[0], "det": key[1], "band": key[2], "roll_deg": key[3],
                              "channel": reg.channel[reg.index[key[2]]], "pixscale_arcsec_per_det_px": float(exps[i]["jac"]["scale"] * a.scale),
                              "n_exposures_in_group": len(ids), "amp_exp_a": kp["amp"], "ee_radii_arcsec": radii_arcsec.tolist(),
                              "ee": ee, "lambda_singular_values": np.linalg.svd(kp["Lc"].astype(np.float64), compute_uv=False).tolist(), "core_m2_target_px2": core_m2, "full_m2_target_px2": full_m2, "mu_centre_target_px": kp["mu"].tolist()})
        np.savez_compressed(out / "kernels" / "effective_kernels.npz", **npz)
        json.dump({"grid_half_width_target_px": a.kernel_half_px, "grid_pitch_target_px": a.kernel_pitch_px, "target_px_arcsec": a.scale, "kernels": meta_rows,
                   "stencil": {"wings": model.stencil.wings, "core_points": model.stencil.nc, "core_pitch_det_px": float(model.stencil.core_pitch),
                               "core_radius_det_px": float(model.stencil.core_r), "base_per": model.base_per, "shape_head_per": model.head_per}},
                  open(out / "kernels" / "kernels.json", "w"), indent=1)
        LOG["kernel_summary"] = {r["group"]: {"EE": dict(zip([f"{x:.2f}" for x in r["ee_radii_arcsec"]], [round(v, 4) for v in r["ee"]])), "exp_a": r["amp_exp_a"],
                                              "core_m2": r["core_m2_target_px2"]} for r in meta_rows}
        # mu fields on a 9x9 lattice for every exposure (training: the model's own nuisance; held out: the fitted one)
        lat = torch.linspace(-0.9, 0.9, 9, device=dev)
        yy, xx = torch.meshgrid(lat, lat, indexing="ij")
        mu_out = {"lattice_dx": xx.cpu().numpy(), "lattice_dy": yy.cpu().numpy()}
        names, mus = [], []
        for i, e in enumerate(exps):
            nz, nid = (keep[i][0], 0) if i in keep else (model.nuis, i)
            eid = torch.full((81,), i, device=dev, dtype=torch.long)
            h = model.trunk(model.ctx_input(eid, xx.reshape(-1), yy.reshape(-1), torch.full((81,), nid, device=dev, dtype=torch.long), nz))
            mu, _, _, _ = model.heads(h, eid, (xx.reshape(-1), yy.reshape(-1), torch.full((81,), nid, device=dev, dtype=torch.long), nz))
            names.append(e["name"])
            mus.append((mu * a.scale * 1000).reshape(9, 9, 2).cpu().numpy())
        np.savez_compressed(out / "mu_fields.npz", names=np.array(names), mu_mas=np.stack(mus), **mu_out)
        LOG["mu_summary_mas"] = {n_: {"mean": [float(x) for x in m_.mean((0, 1))], "rms_about_mean": float(np.sqrt(((m_ - m_.mean((0, 1))) ** 2).sum(-1).mean()))} for n_, m_ in zip(names, mus)}
        kernel_diagnostics(model, exps, train_idx, reg, a, rows_by_cell(meta_rows), LOG, dev)
    LOG["approximations"] = APPROXIMATIONS


def rows_by_cell(meta_rows):
    cells = {}
    for r in meta_rows:
        cells.setdefault(f"{r['instrument']}|{r['det']}|{r['band']}", []).append(r)
    return cells


def kernel_diagnostics(model, exps, train_idx, reg, a, cells, LOG, dev, n_pix=4000):
    """ALWAYS ON (metrics.json["kernel_diagnostics"] and the log), per (instrument, detector, band) cell: the mean and spread of the kernel sum exp(a) over TRAINING
    pixels (a random subset of at most n_pix pixels per cell, taken evenly over the cell's training exposures; the model's own per-exposure nuisance; fixed seed),
    and the core / halo enclosed energy of the effective kernel at 0.1, 0.2, 0.5 arcsec from the sky-frame node offsets 'pos' (EE = sum of the node masses within r of
    the core centroid; halo = 1 - EE; mean over the roll groups of the cell; kernel of the first training exposure of each roll group at its detector centre).
    'a' stays free: no penalty is added. Approximations: node-based EE (a log-polar ring is a step: exact at the node radii only; the stencil's wing radius bounds the
    halo, so the halo beyond the stencil is not counted); a subset of pixels, not all of them."""
    g = torch.Generator().manual_seed(4321)
    by = {}
    for i in train_idx:
        by.setdefault(f"{reg.instrument[reg.index[exps[i]['band']]]}|{exps[i]['detname']}|{exps[i]['band']}", []).append(i)
    out = {}
    with torch.no_grad():
        for cell, ids in sorted(by.items()):
            per = max(1, n_pix // len(ids))
            vals = []
            for i in ids:
                n = int(exps[i]["dx"].numel())
                sel = torch.randperm(n, generator=g)[:per].to(dev)
                dx, dy = exps[i]["dx"][sel], exps[i]["dy"][sel]
                eid = torch.full((sel.numel(),), i, device=dev, dtype=torch.long)
                nid = eid.clone()
                h = model.trunk(model.ctx_input(eid, dx, dy, nid, model.nuis))
                _, _, _, a_ = model.heads(h, eid, (dx, dy, nid, model.nuis))
                vals.append(torch.exp(a_).cpu())
            v = torch.cat(vals).numpy().astype(np.float64)
            rr = cells.get(cell, [])
            ee = {}
            if rr:
                for rad in (0.1, 0.2, 0.5):
                    k = int(np.argmin(np.abs(np.array(rr[0]["ee_radii_arcsec"]) - rad)))
                    ee[f"{rad:g}"] = float(np.mean([r_["ee"][k] for r_ in rr]))
            out[cell] = {"n_pixels": int(v.size), "n_training_exposures": len(ids), "n_roll_groups": len(rr), "exp_a_mean": float(v.mean()), "exp_a_std": float(v.std()),
                         "exp_a_p05": float(np.percentile(v, 5)), "exp_a_p95": float(np.percentile(v, 95)),
                         "lambda_singular_values": np.mean([r_["lambda_singular_values"] for r_ in rr], 0).tolist() if rr else None, "ee_core_arcsec": ee, "halo_1_minus_ee_arcsec": {k_: 1.0 - x for k_, x in ee.items()}}
            log(f"KERNEL-DIAG {cell}: exp(a) mean {v.mean():.3f} std {v.std():.3f} [p05 {np.percentile(v, 5):.3f}, p95 {np.percentile(v, 95):.3f}] (n_pix {v.size}); "
                + (" ".join(f"EE({k_})={x:.3f}/halo {1 - x:.3f}" for k_, x in ee.items())))
    LOG["kernel_diagnostics"] = {"cells": out, "note": "exp(a) over training pixels (kernel sum = unit-sum kernel x exp(a)); EE at 0.1/0.2/0.5 arcsec from the sky-frame node offsets; a is free (no penalty)",
                                 "lambda_gauge": model.lam_gauge, "lam_in_band": model.lam_in_band}
    for ch in sorted({r_["channel"] for rr in cells.values() for r_ in rr}):
        m_ = [out[c]["exp_a_mean"] for c, rr in cells.items() if c in out and rr[0]["channel"] == ch]
        if m_:
            log(f"KERNEL-DIAG channel {ch}: exp(a) cell means range {min(m_):.3f}-{max(m_):.3f} over {len(m_)} cells")


# ====================================================================================================================== CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="opfield: one learned field operator (see module docstring)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stripes", type=int, default=1, help="model NIRCam 1/f stripes (training and held-out closed form)")
    ap.add_argument("--svo-cache", default=None)
    ap.add_argument("--data-dir", default="/arc/projects/ots/torchsky-data/jwst_smacs0723/aligned")
    ap.add_argument("--hst-dir", default=None)
    ap.add_argument("--hst-manifest", default=None)
    ap.add_argument("--hst-programs", default=None)
    ap.add_argument("--hst-instruments", default=None)
    ap.add_argument("--hst-filters", default=None)
    ap.add_argument("--hst-mjd-min", type=float, default=None)
    ap.add_argument("--hst-mjd-max", type=float, default=None)
    ap.add_argument("--hst-wcs", choices=("apriori", "archive"), default="apriori")
    ap.add_argument("--hst-dq-policy", default="default")
    ap.add_argument("--holdout-from", default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=0)
    ap.add_argument("--no-holdout", action="store_true")
    ap.add_argument("--score-all", action="store_true", help="score the training exposures (in-sample, with the held-out nuisance protocol)")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--ra", type=float, default=110.8268)
    ap.add_argument("--dec", type=float, default=-73.4544)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--scale", type=float, default=0.04)
    ap.add_argument("--halo-px", type=int, default=0)
    ap.add_argument("--lam-lo", type=float, default=0.70)
    ap.add_argument("--lam-hi", type=float, default=5.30)
    ap.add_argument("--spec-k", type=int, default=12)
    ap.add_argument("--spec-q", type=int, default=6)
    ap.add_argument("--spec-smooth", type=float, default=1e-3)
    ap.add_argument("--coarse-shift", type=float, default=0.0, help="v36 grid scan (arcsec) for HST, default OFF (S15)")
    ap.add_argument("--coarse-shift-jwst", type=float, default=0.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot-b", type=int, default=1000)
    # operator
    ap.add_argument("--dz", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--shape-hidden", type=int, default=64)
    ap.add_argument("--n-bank", type=int, default=32)
    ap.add_argument("--core-radius", type=float, default=3.0)
    ap.add_argument("--core-pitch", type=float, default=0.25)
    ap.add_argument("--wing-arcsec", type=float, default=3.0)
    ap.add_argument("--wing-ratio", type=float, default=1.25)
    ap.add_argument("--wing-nang", type=int, default=24)
    ap.add_argument("--psf-init-sigma", type=float, default=1.0, help="generic initial Gaussian sigma [detector px], all channels")
    # more instrument-specific operator (all default to the original model)
    ap.add_argument("--base-per", choices=("channel", "instrument", "detector", "detector_band"), default="channel",
                    help="one base kernel per channel (default = original: shared by all detectors and bands of the channel), per instrument (COARSER than channel for NIRCam: SW + LW share one row), "
                         "per detector, or per (detector, band) (a free table per cell, like E2-data)")
    ap.add_argument("--shape-head-per", choices=("shared", "channel"), default="shared", help="shape MLP + bank shared by all instruments (default) or one pair per channel")
    ap.add_argument("--wings", choices=("logpolar", "cartesian"), default="logpolar",
                    help="logpolar = original wings (rings of ratio --wing-ratio x --wing-nang azimuths); cartesian = nested square lattices with the pitch growing in steps (--cart-steps)")
    ap.add_argument("--cart-steps", default="24:1,48:2,inf:4", help="--wings cartesian: 'outer_radius_detector_px:pitch' list beyond the core radius (last step extends to the wing radius)")
    ap.add_argument("--base-table", choices=("none", "detector_band"), default="none",
                    help="detector_band: a free 2-D table per (detector, band) on the Cartesian stencil as the base (implies --base-per detector_band and --wings cartesian); "
                         "the hypernetwork only modulates it with --mod-rank basis vectors")
    ap.add_argument("--mod-rank", type=int, default=8, help="--base-table: number of modulation basis vectors (replaces --n-bank)")
    ap.add_argument("--base-init-from", default=None, help="E2-data run dir (or *_psf_groups.npy): start the (detector, band) base tables from its learned tables (data-driven warm start; needs detector_band rows)")
    ap.add_argument("--base-init-key", choices=("learned", "table", "library"), default="learned")
    ap.add_argument("--lambda-gauge", choices=("learned", "jacobian"), default="learned",
                    help="learned = M = J Lambda(h) (default, original). jacobian = Lambda == I, M = J exactly (head_lam kept for state-dict compatibility but unused); removes the Lambda / Psi-width gauge redundancy")
    ap.add_argument("--lam-in-band", action="store_true",
                    help="add (t, t^2, sin pi t, cos pi t), t = (lambda_q - lambda_mid) / (lambda_max - lambda_min) of the quadrature node inside its band, to the shape-MLP input (changes the shape-MLP input size: new state dict)")
    # evaluation / diagnostics
    # NEW (8 Nov 2026): wings / shoulders (all default to the original behaviour; S33-S37 in the module docstring)
    ap.add_argument("--psi-param", choices=("softplus", "log"), default="softplus",
                    help="shape parameterisation: softplus (default) = density softplus(base + coef @ bank); log = Psi proportional to exp(base + coef @ bank), mass = Psi dv normalised to unit sum (softmax), same start kernel")
    ap.add_argument("--importance-sampling", default=None, metavar="R1,R2,...",
                    help="annulus-stratified sampling around the bright peaks of the CURRENT MODEL: outer radii [detector px] of the annuli (first starts at 0), e.g. 1,3,8,15,30; exact reweighting 1/(N q)")
    ap.add_argument("--is-frac", type=float, default=0.5, help="--importance-sampling: fraction of each batch drawn from the annuli (shared equally by the non-empty ones)")
    ap.add_argument("--is-refresh", type=int, default=200, help="--importance-sampling: steps between peak-list refreshes (from the model band images)")
    ap.add_argument("--is-snr", type=float, default=50.0, help="--importance-sampling: peaks above this S/N (x the coadd band noise)")
    ap.add_argument("--is-max-peaks", type=int, default=200, help="--importance-sampling: at most this many (brightest) peaks per band")
    ap.add_argument("--two-scale", type=float, default=0.0, metavar="R_CORE_ARCSEC",
                    help="two-scale operator: per-pixel stencil truncated at r <= R_CORE (arcsec); beyond it (to --wing-arcsec) one free log table per (detector, band) in the detector frame at the sky pitch, FFT-convolved with the band image per roll group; 0 = off")
    ap.add_argument("--wing-mod", type=int, default=0, help="--two-scale: degree K of the Legendre modulation (in detector x, y) of the wing amplitude, exp(sum c_k P_k), constant term excluded; 0 = none")
    ap.add_argument("--wing-group-deg", type=float, default=0.5, help="--two-scale: roll angles are grouped in bins of this width (degrees) per (detector, band, parity); the group Jacobian is the mean J")
    ap.add_argument("--wing-chunk", type=int, default=8, help="--two-scale: groups per batched FFT call (bounds the temporaries)")
    ap.add_argument("--wing-init", choices=("powerlaw", "floor"), default="powerlaw",
                    help="--two-scale: start of the wing table: powerlaw = generic r^-3 halo of mass --wing-f0 outside R_core (+ the Gaussian start); floor = only the 1e-7 floor (f_w0 ~ 1e-3)")
    ap.add_argument("--wing-f0", type=float, default=0.01, help="--two-scale --wing-init powerlaw: initial wing mass fraction")
    ap.add_argument("--wing-lr-mult", type=float, default=5.0, help="--two-scale: learning-rate multiplier (x --lr-kernel) of the wing tables / fractions / modulation")
    ap.add_argument("--wing-l2", type=float, default=1e-4, help="--two-scale: weight of mean (table - table_init)^2 + (logit f_w - start)^2 + c^2 (the pull toward the generic start)")
    ap.add_argument("--eval-only", action="store_true", help="no training: load --init-state and run the held-out protocol (and the outputs / dumps)")
    ap.add_argument("--init-state", default=None, help="--eval-only: opfield_best.pt / opfield_state.pt of a finished run (same data and args)")
    ap.add_argument("--args-from", default=None, help="metrics.json of a finished run: its 'args' become the DEFAULTS of this command line (explicit flags win; out / eval-only / init-state / dumps are not taken)")
    ap.add_argument("--dump-heldout-pixels", action="store_true", help="write OUT/heldout_pixels/opfield/<exposure>.npz per scored exposure (format of smacs_four_v38 --dump-heldout-pixels; for null_stacks.py / compare_operators.py)")
    # coarse to fine, gauges
    ap.add_argument("--c2f-start-arcsec", type=float, default=0.4)
    ap.add_argument("--c2f-frac", type=float, default=0.3)
    ap.add_argument("--nuis-c2f", type=int, default=1)
    ap.add_argument("--noise-start", type=float, default=-1.0, help="training progress before which the learned noise head is off (h = 0, pi = initial); -1 = --c2f-frac")
    ap.add_argument("--sky-warmup", type=float, default=0.0, help="fraction of the budget with the sky learning rate at 0 (operator and nuisance only)")
    ap.add_argument("--ref-instrument", default="NIRCam")
    ap.add_argument("--gauge-shift", type=float, default=10.0)
    ap.add_argument("--gauge-flux", type=float, default=10.0)
    ap.add_argument("--gauge-blur", type=float, default=1e-4)
    # training
    ap.add_argument("--budget-min", type=float, default=420.0)
    ap.add_argument("--steps", type=int, default=0, help="maximum steps (0 = time budget only)")
    ap.add_argument("--batch", type=int, default=0, help="pixel minibatch (0 = from available GPU memory; CPU 2048)")
    ap.add_argument("--bright-frac", type=float, default=0.25)
    ap.add_argument("--loss", choices=("mixture", "huber"), default="mixture", help="mixture = Gaussian + uniform outlier (default); huber = v36-style Huber on the whitened residual (no outlier component)")
    ap.add_argument("--huber-delta", type=float, default=3.0)
    ap.add_argument("--lr-sky", type=float, default=0.0)
    ap.add_argument("--lr-op", type=float, default=2e-3)
    ap.add_argument("--lr-kernel", type=float, default=3e-3)
    ap.add_argument("--lr-z", type=float, default=1e-2)
    ap.add_argument("--z-hier", action="store_true", help="z_i = z_epoch + z_visit + z_exp (visit / epoch ids from the exposure names, epochs = MJD clusters) + Fourier features of the epoch; the mu skip reads epoch + visit only")
    ap.add_argument("--z-exp-mult", type=float, default=10.0, help="z-hier: the exposure part is penalised z-l2 x this")
    ap.add_argument("--z-l2-epoch", type=float, default=1e-4)
    ap.add_argument("--z-l2-visit", type=float, default=1e-3)
    ap.add_argument("--restart-cycles", type=int, default=0, help="N: re-initialise the sky N times from a splat coadd built with the LEARNED mu (keeps theta, z)")
    ap.add_argument("--restart-first", type=float, default=0.35)
    ap.add_argument("--restart-last", type=float, default=0.55)
    ap.add_argument("--sky-c2f", type=float, default=0.0, help="fraction of the budget over which the sky band limit (extra Gaussian blur, quadrature with the read blur, plus low-pass of the initial sky) is released; 0 = off")
    ap.add_argument("--sky-c2f-arcsec", type=float, default=0.8)
    ap.add_argument("--xfit-reg", type=int, default=0, help="K >= 2: cross-fitted registration; fold-k exposures register against a sky view that never saw fold k; 0 = off")
    ap.add_argument("--xfit-frac", type=float, default=0.7, help="fraction of the budget spent cross-fitting; afterwards the views are averaged and normal training continues")
    ap.add_argument("--operator", choices=["field", "factorised"], default="field",
                    help="field = hypernetwork (default); factorised = free per-exposure affine + per-detector cubic distortion, Lambda = J, one free Psi table per channel, free per-exposure log gain (diagnostic baseline)")
    ap.add_argument("--mu-skip", action="store_true", help="field operator: add a direct linear skip z_i -> mu (target px = --mu-skip-scale * W z)")
    ap.add_argument("--mu-skip-scale", type=float, default=4.0)
    ap.add_argument("--lr-mu-skip", type=float, default=1e-2)
    ap.add_argument("--lr-aff", type=float, default=2e-2, help="factorised: lr of the per-exposure shift / matrix [target px]")
    ap.add_argument("--lr-dist", type=float, default=5e-3, help="factorised: lr of the cubic distortion")
    ap.add_argument("--lr-gain", type=float, default=1e-2, help="factorised: lr of the per-exposure log gain")
    ap.add_argument("--lr-bkg", type=float, default=0.0)
    ap.add_argument("--lam-tv", type=float, default=0.02)
    ap.add_argument("--lam-neg", type=float, default=1.0)
    ap.add_argument("--z-l2", type=float, default=1e-3)
    ap.add_argument("--g-l2", type=float, default=0.05)
    ap.add_argument("--lc-l2", type=float, default=1e-2, help="L2 prior on the Lambda correction (log scales, angle, shear) toward the detector Jacobian")
    ap.add_argument("--bank-l2", type=float, default=1e-2)
    ap.add_argument("--base-l2", type=float, default=1e-4, help="weight of mean (base - base_init)^2 (the pull toward the generic Gaussian start, or the E2 tables with --base-init-from); 0 = no pull")
    ap.add_argument("--val-every-min", type=float, default=30.0)
    ap.add_argument("--val-nuis-steps", type=int, default=100)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--ckpt-every-min", type=float, default=30.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--skip-eval", action="store_true", help="development: skip the final held-out evaluation")
    ap.add_argument("--nuis-steps", type=int, default=300)
    ap.add_argument("--nuis-batch", type=int, default=4096)
    ap.add_argument("--kernel-half-px", type=float, default=24.0)
    ap.add_argument("--kernel-pitch-px", type=float, default=0.25)
    ap.add_argument("--eval-every-min", type=float, default=0.0, help="accepted for compatibility (validation uses --val-every-min)")
    pre, _ = ap.parse_known_args(argv)
    if pre.args_from:
        sa = dict(json.load(open(pre.args_from)).get("args", {}))
        if sa.get("size_data") is not None:   # main() adds 2 halo to a.size and stores size_data = the requested size
            sa["size"] = sa["size_data"]
        valid = {act.dest for act in ap._actions}
        skip = {"out", "eval_only", "init_state", "args_from", "dump_heldout_pixels", "resume", "size_data", "hook", "help"}
        ap.set_defaults(**{k: v for k, v in sa.items() if k in valid and k not in skip})
    a = ap.parse_args(argv)
    if a.base_table != "none":
        a.base_per, a.wings = "detector_band", "cartesian"
    if a.base_init_from and a.base_per != "detector_band":
        ap.error("--base-init-from needs --base-per detector_band (or --base-table detector_band)")
    if a.eval_only != bool(a.init_state):
        ap.error("--eval-only and --init-state go together")
    if a.importance_sampling:
        parse_is_radii(a.importance_sampling)
        if not 0.0 < a.is_frac <= 1.0:
            ap.error("--is-frac must be in (0, 1]")
        if a.xfit_reg >= 2:
            ap.error("--importance-sampling is not implemented together with --xfit-reg")
    if a.two_scale > 0:
        if a.two_scale >= a.wing_arcsec:
            ap.error("--two-scale R_CORE must be smaller than --wing-arcsec")
        if a.xfit_reg >= 2:
            ap.error("--two-scale is not implemented together with --xfit-reg")
        if a.operator == "factorised":
            ap.error("--two-scale is not implemented for --operator factorised")
    if a.quick:
        a.hidden, a.n_bank, a.shape_hidden = min(a.hidden, 32), min(a.n_bank, 8), min(a.shape_hidden, 32)
        a.batch = a.batch or 512
        a.steps = a.steps or 60
        a.nuis_steps = min(a.nuis_steps, 30)
        a.val_nuis_steps = min(a.val_nuis_steps, 20)
        a.log_every = min(a.log_every, 20)
    return a


def main(argv=None):
    global LOSS_KIND, HUBER_DELTA
    a = parse_args(argv)
    LOSS_KIND, HUBER_DELTA = a.loss, a.huber_delta
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.size_data = a.size
    if a.halo_px > 0:
        a.size = a.size + 2 * a.halo_px
    thead = V.target_header(a.ra, a.dec, a.size, a.scale)
    pad_ = int(round(24 * 0.04 / a.scale)) - a.halo_px
    V.args_global = a
    V.CALIBRATE_NOISE = False
    exps = V.load_all_exposures(a, thead, dev, pad_)
    print(f"loaded {len(exps)} exposures, {sum(e['d'].numel() for e in exps) / 1e6:.2f} M pixels", flush=True)
    if not exps:
        raise SystemExit("no exposures loaded")
    run(exps, thead, dev, a, out)


if __name__ == "__main__":
    main()
