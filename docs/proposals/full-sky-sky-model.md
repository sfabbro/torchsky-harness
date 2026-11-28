# Toward a full-sky, all-survey sky model

Date: 2026-11-04
Status: proposal and brainstorm, for discussion; nothing here is implemented
unless marked as measured.
Torchsky snapshot: `0.5.0` (`main`).
Companion notes: [INR and Power Foam problem formulation](../inr_foam_problem_formulation.md)
(single-field formulation and controls), [map tool parity audit](map-tool-parity-2026.md)
(classical baselines).

## Summary

Goal: one continuous sky field `s(x, λ, t)`, fitted to raw pixels from every
survey (HST, JWST, Euclid, Roman, UNIONS and other ground surveys) through a
learned instrument model, scalable to the full sky, and able to run where the
data live instead of moving petabytes.

Main directions:

1. **Learn the instrument, do not model it effect by effect.** Keep only two
   structural facts: optics are linear in sky brightness; everything nonlinear
   happens at or after the detector. Learn PSF, pixel response, detector
   effects, background and noise from data, with one network shared by all
   instruments and a learned code per instrument, detector and exposure.
2. **Stronger sky representations** read at each detector sample's own sky
   position, with resolution earned on held-out exposures, a spectral axis
   shared across surveys, and an explicit time dependence.
3. **Train by predicting unseen exposures** ("novel-exposure synthesis"):
   self-supervised on the raw archive, no ground truth, no simulations.
   Amortise with a feed-forward multi-exposure model, refine per field.
4. **Scale by splitting global from local**: small shared instrument model,
   huge but local sky on HEALPix tiles; alternate between them.
5. **Federate**: the log-likelihood is a sum over exposures, so each data
   centre computes its own terms; models and summaries move, pixels do not.

## 1. Where we stand (measured on SMACS J0723)

NIRCam, 6 bands, 123 exposures, 20″ field at 0.04″. Scores on unseen test
exposures, one per band; ⟨r²/σ²⟩ on bright pixels (1 = noise only) and
aperture-flux error. No error bars yet (one field, one test exposure per band).

| Sky model | Instrument | Bright ⟨r²/σ²⟩ | Flux error |
|---|---|---|---|
| Drizzle-like coadd | none | 2.77 | 2.7σ |
| Pixel grid, per band | rigid (Gaussian PSF, pipeline WCS) | 2.26 | 2.5σ |
| Pixel grid, per band | learned jointly | 1.84 | 2.1σ |
| Pixel grid, `s(λ)` basis | learned jointly | 2.18 | 2.6σ |
| "Neural field", `s(λ)` basis | learned jointly | 2.35 | 2.5σ |
| Pixel grid, `s(λ)`, L-BFGS | frozen from the 2D fit | 2.53 | – |

What these runs do and do not show:

- **The instrument carries the gain** (2.26 → 1.84 on the same sky grid).
- **The "neural field" was a reparametrised pixel grid.** Dense feature grids
  plus an MLP, evaluated only at the 512² grid nodes, then convolved and
  interpolated like the grid. Its continuity was never used.
- **The learned PSF was a shortcut:** an MLP basis of 4 kernels, 31×31 at
  0.04″ (cut at 0.62″), evaluated at the pivot wavelength only in the 2D fits.
- **The optimiser is not the limit.** L-BFGS lowers the training loss much
  faster than Adam; validation stops improving within minutes.
- **The χ² floor (≈5.6 on all pixels) is model error:** PSF wings, bright
  stars, F277W.
- **Uncertainties:** a Laplace sky variance (Hutchinson Fisher diagonal) brings
  test coverage from 62/85 % to 68/89 % at the 68/90 % levels; temperature
  scaling and split conformal (torchregress) give 70/90 and 71/92.

Running as baselines (not the direction): a physical pupil PSF (approximate
segment and strut geometry, Zernike wavefront per channel and detector,
detector MTF, 4.8″ window) with the pixel grid, and a continuous zip-field sky
read at detector positions through Monte Carlo samples of that PSF.

## 2. Goals and constraints

- **Catalogue-free:** no source detection, identification or separation in
  the model. Catalogues may be used for scoring only.
- **Any scene:** crowded fields, bright extended or moving targets (Jupiter,
  comets), diffuse emission, empty deep fields, with the same settings.
- **Any instrument and band set,** space and ground.
- **Learn from data, and get better with more data** (bitter lesson): the
  instrument model is learned; physics enters only as structure that holds
  for every telescope, or as an optional warm start.
- **Sky priors:** no prior learned on external data or simulations. Whether a
  decoder shared across the sky and learned from the same data is allowed is an
  open decision (§4.5, §10).
- **Scale:** cost close to linear in pixels and exposures; tileable; runnable
  at several data centres.

## 3. Forward model: a learned instrument

```
d = N( A[s; z] ; z ) + B + ε
```

`s` sky, `A` optics (linear in `s`), `N` detector (nonlinear, local), `B`
background, `ε` noise, `z` learned codes per instrument, detector and exposure.

Two structural facts are kept because they hold for every telescope:

1. **Incoherent imaging is linear in sky brightness**, including scattered
   light and ghosts. This is what lets one sky be shared by all instruments.
2. **Nonlinearity lives at or after the detector:** brighter-fatter,
   nonlinearity, saturation, persistence.

Everything else is learned.

| Component | Representation | Learns | Depends on |
|---|---|---|---|
| Optics `A` | PSF as a neural field over the offset, or a conditional normalising flow sampled at render time | diffraction, aberrations, variation across the field and in time, jitter, seeing, refraction, wings, ghosts | offset, detector position, λ, t, `z` |
| Pixel response | small field over intra-pixel and detector position | pixel footprint and sensitivity variations, linear inter-pixel coupling | position, `z` |
| Detector `N` | small convolutional network from ideal electrons to expected counts | brighter-fatter, nonlinearity, saturation, persistence (earlier frames as input), unknown local effects | exposure time, `z` |
| Background `B` | smooth field per exposure in the detector frame + static pattern per detector; deliberately low capacity | sky level, stray light, 1/f stripes | exposure, detector |
| Noise | network for per-pixel variance, an outlier term, a learned whitening for correlated noise | Poisson, read noise, correlated noise, cosmic rays, artefacts | predicted counts, `z` |

### 3.1 PSF without convolution

Rendering by sampling: for each detector sample, draw offsets from the PSF
density and read the sky at those positions. No kernel window, free wings,
cost independent of kernel size, works for any PSF shape and for a PSF that
varies continuously across the field. A conditional normalising flow gives
both the density and exact samples; a neural field needs inverse-CDF sampling
on a fine grid. Cost: Monte Carlo variance (for an `S`-sample mean, the
squared loss gains a `Var/S` term that acts as weak smoothing) — use control
variates or a deterministic core plus sampled wings.

Where data do not constrain the PSF, physics can enter as a warm start or an
extra input (for example the pupil PSF now running), never as a hard
constraint.

### 3.2 Scaling across instruments

One network for all instruments; each instrument, detector and exposure gets
a learned code fitted jointly (auto-decoder, as in NeRF in the Wild). A new
survey adds codes; the shared weights improve with every dataset. Per-
instrument output heads (as in WeatherNext 3's observation heads) handle
instrument-specific outputs.

### 3.3 Identifiability

A flexible operator with a flexible sky is blind deconvolution. What breaks
the degeneracy:

- the same sky through several operators: instruments, bands, dithers,
  rotations, epochs. This is why data diversity should reveal more structure;
- one operator shared across many fields, so it cannot fit field-specific
  structure;
- gauges: unit-flux PSF with a fixed centroid per band, low-capacity
  background, detector response close to identity at low signal, absolute
  frame and flux anchors (§6.6);
- capacity decided by held-out exposures only.

Stop criterion: if more operator flexibility makes held-out sky prediction or
injection recovery worse, the operator is absorbing sky.

## 4. Sky representation `s(x, λ, t)`

### 4.1 Requirements

- Read at each detector sample's own sky position through the operator, with
  no intermediate grid.
- Resolution earned locally: add capacity while held-out likelihood improves;
  stop per region and band when it does not.
- Same footing for compact, crowded and diffuse light.
- Tileable with halos.

### 4.2 Spatial candidates

| Representation | How the PSF enters | Resolution adapts by | Main risk |
|---|---|---|---|
| Multi-resolution hash/feature grids + MLP (Instant-NGP, Mip-/Zip-NeRF style, with area-integrated queries) | sampled PSF offsets | finest level used | MLP smooths compact sources; anti-aliasing must match the pixel footprint |
| Band-limited Fourier field | exact product in Fourier space, NUFFT to samples | frequency support per band from the PSF transfer function | ringing around bright compact sources |
| Multiscale B-splines or starlets | exact convolution per basis function, or FFT per scale | finest scale kept per region | many fine coefficients for edges and compact sources |
| Gaussian splats | analytic for Gaussian-mixture PSFs, else sampled PSF | splits and prunes on held-out significance; width floor | densification rules, degeneracy of overlapping splats |

Splats are viable only with: amplitudes solved exactly (non-negative least
squares, §5.1), a width floor at the target resolution, splits and prunes
decided by held-out significance, a smooth spectrum per splat, tile-based
rendering. Splats are a representation, not source identification: nothing
requires one splat per source.

### 4.3 Spectral axis

Shared basis learned over the whole sky, `s(x, λ) = Σₖ aₖ(x) φₖ(λ)`, smooth in
λ and integrated through each passband at quadrature nodes. A rest-frame
variant `φₖ(λ / (1 + z(x)))` with a continuous redshift field `z(x)` is close
to USPM and needs no source identification; it breaks where sources at
different redshifts overlap (a mixture of layers would be needed).

### 4.4 Time

Surveys span about 20 years (HST over SMACS J0723 since 2008, JWST in 2022),
so a static sky breaks. Catalogue-free form, as in dynamic NeRFs:

```
s(x, λ, t) = s₀( x − v(x)(t − t₀), λ ) + Σⱼ bⱼ(x, λ) ψⱼ(t)
```

`v(x)` a smooth proper-motion field, `bⱼ ψⱼ` a low-rank variable part. Risk: a
moving source and a varying one look alike when epochs are sparse.

### 4.5 Compressed latents shared across the sky

The sky is very redundant. Options:

1. **Auto-decoder over tiles:** one decoder `D(x, λ, t; z_tile)` trained over
   all tiles; each tile fits only its code, from a meta-learned start
   ("functions as data").
2. **Latent grid + local decoder** (as in learned image codecs): a coarse grid
   of latents on HEALPix, decoded locally. Tileable; the latents double as
   tokens for a sky foundation model.
3. **Generative model on the latents** (latent diffusion, or an FGN head,
   §5.4) for posterior samples.

Open points:

- A shared decoder is a learned sky prior. Proposed rule if adopted: learned
  only from the data being modelled, across all fields, judged on held-out
  exposures, trained jointly with the forward model on raw exposures (never on
  our own reconstructions, which carry instrument errors).
- A decoder only expresses what it has seen. Keep a free residual,
  `s = D(z) + r`, with `r` under the held-out capacity rule. Where `r` is
  needed is a discovery map.
- Gate: the decoder-constrained fit must beat a free fit on held-out
  exposures of shallow fields (single-epoch ground, few exposures) and not
  lose on deep ones.

## 5. Inference and architectures

### 5.1 Per-field optimisation

- **Variable projection:** for a fixed instrument, the sky amplitudes are
  linear; solve them by preconditioned conjugate gradients on the normal
  equations (IRLS for the Huber loss), preconditioned by the Fisher diagonal.
  Outer loop on the instrument only.
- **Coarse to fine,** doubling as the resolution rule.
- **Uncertainty from the same operator:** Laplace by conjugate gradients,
  perturb-and-optimise samples, coverage checked on held-out exposures.

### 5.2 Amortised, feed-forward reconstruction

A transformer reads all exposures of a field as a set of tokens (pixels plus
WCS, band, time, instrument code) and outputs a latent sky; the sky is then
queried at any `(x, λ, t)` through cross-attention, so it stays continuous.
Precedents: Scene Representation Transformer, Perceiver IO, feed-forward
Gaussian splatting (pixelSplat, MVSplat, GS-LRM), large reconstruction models.

Training objective: **predict held-out exposures** from the others, through the
forward model, across the raw archive. A "view" is any instrument, band,
epoch or dither. No ground truth, no simulations; it is our held-out score
used as the training loss.

### 5.3 Test-time refinement

Refine the amortised output per field: a few steps of §5.1 through the
explicit operator, or test-time training of the network on that field's
exposures.

### 5.4 Probabilistic output: FGN rather than diffusion

Functional Generative Networks (WeatherNext 2 and 3): a low-dimensional noise
vector conditions the normalisation layers, so each forward pass is one
coherent sample; training uses a per-pixel proper score (CRPS) on marginals,
yet samples are coherent jointly. One pass per sample, versus many denoising
steps for diffusion.

For us: CRPS on held-out raw exposures (ensemble predictions plus the noise
model) instead of the Huber loss. Checks: joint scores (energy, variogram),
sample power spectra, coverage as in §1. Risk: a low-dimensional noise vector
may not capture local ambiguities everywhere at once (blends in crowded
fields). The spread is an implicit learned prior: hallucination is checked by
held-out and injection tests.

### 5.5 Precedents from Earth observation

Weather has the same problem: many heterogeneous instruments, one global
state. ECMWF GraphDOP and Aardvark Weather learn from observations only.
WeatherNext 3 (Google DeepMind, 2026, arXiv:2609.03582) uses one FGN mesh
transformer with dense, sparse-station and track heads, takes satellite
mosaics as input, trains on raw station observations, and is still anchored on
reanalysis (ERA5/HRES). Astronomy has no reanalysis; candidate anchors: coadds
as warm start, or the highest-resolution instruments acting as the "analysis"
(§10).

### 5.6 Learned solvers

RIM-type recurrent solvers, using the likelihood gradient of our own operator,
only after the above is settled, to amortise speed. Trained on simulations,
they inherit a prior from the simulator, which must then be documented and
varied across scene types.

### 5.7 Scaling laws

Measure held-out exposure prediction against data, compute and model size
early, before building at full sky.

## 6. Full-sky scaling

### 6.1 Global/local split and alternation

- **Global, small:** shared instrument network and codes; trained by
  data-parallel gradient steps over random batches of tiles.
- **Local, huge:** the sky on HEALPix tiles with margins of one PSF window;
  with the instrument fixed, each tile is an independent solve.
- Alternate: solve tiles, update instrument, repeat. Gaia's AGIS (sources,
  attitude, calibration) and SDSS ubercal follow the same block-iterative
  pattern.

### 6.2 Adaptive resolution

A dense sky at 0.04″ is about 10¹⁴ values per band. Each region gets the
hierarchy depth its data support (Euclid or Roman where observed, ground
seeing elsewhere).

### 6.3 Seams

Blending in overlaps; if residuals show seams, a few rounds of exchange
between neighbouring tiles (Schwarz domain decomposition).

### 6.4 Incremental updates

With a Gaussian likelihood, the normal equations add over exposures: new data
re-solve only the tiles they touch, plus a fine-tune of the instrument
(approximate with the Huber loss).

### 6.5 Serving

The sky is a function queried at `(α, δ, λ, t, resolution)`, exposed as HiPS
so existing viewers work.

### 6.6 Anchors

Absolute astrometric frame (Gaia used only to fix the frame, never inside the
model, or self-calibration with a fixed gauge) and absolute flux scale.

### 6.7 Cost (order of magnitude, not measured)

Euclid + UNIONS + Roman ≈ 10¹⁴ pixel-exposures, ≈ 400 TB read per pass.
Reading data dominates, so computation must run where data are stored, each
tile reading only its own cutouts. Today's SMACS fit takes hours for a 20″
field: the fast solver (§5.1) is a prerequisite, not an optimisation.

## 7. Federated fitting: move models, not pixels

**Principle.** Exposures are independent given the sky and the instrument, so
the log-likelihood is a sum over exposures, and the exposures live at
different centres (MAST, ESA, CADC, IRSA, Rubin, ...). Each centre evaluates
its own terms, with their gradients and curvature, on its own hardware.
"Adding up likelihoods" is literally right; the methods below differ in how
many rounds of exchange they need.

**What moves:** the shared instrument network (small); the sky tiles where
two centres' data overlap; per-tile summaries. Instrument codes and output
heads stay at the instrument's home centre. Raw exposures never move.

| Mechanism | What is exchanged | Exact? | Rounds |
|---|---|---|---|
| Federated gradients (FedSGD/FedAvg) | gradients or weight updates | yes, in the limit | many |
| Additive normal equations | per-tile `AᵀWA` blocks (or their action) and `AᵀWd` | exact for the linear sky with the instrument fixed (Gaussian) | one per instrument update |
| Consensus ADMM | local tile copies and dual variables | converges to the joint optimum for the convex sky problem (Huber, instrument fixed) | tens |
| Sequential Bayesian update (Kalman-like) | sky mean + precision (diagonal or low-rank) | exact for linear-Gaussian, approximate otherwise | one per new dataset, any infrastructure |

Hard parts:

- **Coupling:** when the shared instrument changes, every centre's summaries
  go stale; each centre re-runs locally (an extra outer iteration).
- **Size:** a Euclid-resolution sky is tens of TB per band; exchange is
  per tile and only where data overlap; adaptive resolution and latents shrink
  it.
- **Proprietary data:** the sky model is effectively the image, so federation
  does not protect proprietary periods. Governance, not technology.
- **Bookkeeping:** sequential updates double-count data without a strict
  ledger of which exposures entered which tile version.
- **Reproducibility and trust:** each update carries signed provenance
  (exposures, code version, instrument-model version); tiles are versioned
  like code (content-addressed). Determinism across heterogeneous hardware
  is checked by replay on a reference tile.

Precedents: Gaia DPAC distributed processing, SKA Regional Centres Network
(moving compute to data), FedAvg, ADMM consensus (Boyd et al. 2011), ensemble
Kalman filters in weather data assimilation.

## 8. Evaluation protocol

Scores, all with bootstrap error bars over exposures:

1. **Held-out prediction:** ⟨r²/σ²⟩ on all and bright pixels, robust
   variants, aperture flux error, on several test exposures per band; CRPS and
   coverage for probabilistic models.
2. **Injection and recovery:** point-like sources, resolved discs and arcs,
   smooth diffuse light added to raw exposures through an independent PSF.
   Measure flux and position bias, recovered against injected size
   (resolution), and limiting depth. Catalogues for scoring only.
3. **Cost:** GPU-hours per square degree per exposure, peak memory, data read,
   bytes exchanged (federated runs).
4. **Federated equivalence:** a federated fit must match the joint fit on
   held-out data within error bars.

Pointing suite (same settings everywhere; no per-scene tuning):

| Scene | Example | Stresses |
|---|---|---|
| Multi-instrument overlap | SMACS J0723 JWST NIRCam + HST ACS/WFC3 (RELICS, programme 14096, plus earlier ACS and WFPC2); UNIONS + Euclid overlap | shared sky, learned operators, data diversity, time |
| Lensing cluster | SMACS J0723 NIRCam (have) | diffuse light, arcs, few stars |
| Crowded | globular-cluster or Galactic-plane NIRCam field | blends everywhere, no empty sky |
| Bright, extended, moving | Jupiter, NIRCam early release science | saturation, scattered light, non-sidereal frame |
| Empty deep | JADES / GOODS-S | background-limited faint sources |
| Ground | UNIONS MegaCam tile (have) | seeing, refraction, many CCDs |

## 9. Milestones and gates

Each phase ends with a gate; a failed gate stops that direction and the
reason is written down.

| Phase | Work | Gate | Stop if |
|---|---|---|---|
| 0. Federation proof | SMACS J0723, JWST and HST as two "sites" (HST via astroquery/MAST); consensus or additive fit vs joint fit | federated = joint on held-out data | not reachable in a few tens of rounds |
| 1. Learned operator, one instrument | PSF flow with sampled rendering, detector network, learned noise, on SMACS NIRCam | below 1.84 (2D) and below the physical pupil PSF | cannot beat the physical PSF |
| 2. Two instruments | shared operator with codes, JWST + HST, then UNIONS + Euclid | adding an instrument improves held-out prediction of the other; smaller injected structure recovered | more flexibility makes the sky worse |
| 3. Representations | grid vs Fourier vs splines/starlets vs hash grids vs splats, same operator; time term on the 2008–2022 baseline | a continuous model ≥ grid on held-out data and wins on injection resolution | all stay behind the grid |
| 4. Optimisation | variable projection, PCG/Gauss–Newton, coarse to fine | 10× faster at equal score | speed-up below 3× |
| 5. Amortised + probabilistic | feed-forward multi-exposure model trained on held-out exposures; FGN head with CRPS; scaling laws | matches per-field fits after refinement; calibrated coverage; scores improve with data | no gain with more data |
| 6. Suite and scale | every scene with one setting; tiling to a UNIONS tile; then a multi-survey region | holds on every scene; new instrument needs only its codes; no seams | scene-specific switches needed |
| 7. Full-sky pilot | federated across two centres (e.g. CADC and MAST) | federated = joint; cost per square degree acceptable | exchange or staleness dominate |

## 10. Open decisions

- Learned sky prior (shared decoder) allowed, and under which rule (§4.5).
- Anchor for training without a reanalysis: coadd warm start, or the
  sharpest instruments as reference (§5.5).
- Gaia for the absolute frame only, or self-calibration (§6.6).
- Time model: proper-motion field plus low-rank variability, or more general.
- Proprietary data in federated fits (§7).

## References (by name)

- Instant-NGP (Müller et al. 2022); Mip-NeRF 360 and Zip-NeRF (Barron et al.
  2022, 2023); NeRF in the Wild (Martin-Brualla et al. 2021); D-NeRF and
  Nerfies (dynamic fields); 3D Gaussian splatting (Kerbl et al. 2023).
- DeepSDF (Park et al. 2019); "From data to functa" (Dupont et al. 2022).
- Scene Representation Transformer (Sajjadi et al. 2022); Perceiver IO
  (Jaegle et al. 2022); pixelSplat, MVSplat, GS-LRM.
- GraphDOP (ECMWF); Aardvark Weather (Allen et al. 2025); WeatherNext 2
  (Functional Generative Networks); WeatherNext 3 (arXiv:2609.03582).
- AION-1, AstroCLIP, 4M (multimodal masked modelling).
- Recurrent Inference Machines (Putzky & Welling 2017).
- Gaia AGIS (Lindegren et al. 2012); SDSS ubercal (Padmanabhan et al. 2008).
- ADMM (Boyd et al. 2011); FedAvg (McMahan et al. 2017); ensemble Kalman
  filter (Evensen 1994); SKA Regional Centres Network.
