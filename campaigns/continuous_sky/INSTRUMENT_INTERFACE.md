# Instrument interface, and HST as the second instrument (18 Nov 2026)

Package: `/home/claude/work/fieldop_tsky/fieldop/`. Two cores, one thread, nothing installed, no network, nothing on CANFAR touched, no fit beyond the synthetic checks.
Words: *instrument* = one camera (NIRCam, WFC3 infrared channel, ACS wide-field channel). *Chip* = one science extension of a file. *Exposure* (in the package) = one chip of one file.
*Data-quality bits* = the flag plane (DQ) of the file. *Surface brightness* = MJy/sr per nominal pixel area, the unit of the JWST calibrated (`_cal`) files.

## 1. The interface (`fieldop/instruments.py`, new, 380 lines)

One `Instrument` description per camera, in a registry `INSTRUMENTS` keyed by the value of the `instrument` field of the list of exposures (`NIRCAM`, `WFC3_IR`, `ACS_WFC`;
aliases `WFC3` with detector `IR`, `ACS` with detector `WFC`; `WFC3` with any other detector, for example `UVIS`, is refused, not read as the infrared channel).
Adding an instrument = one description + one reader function. No other file reads instrument facts any more.

| field | meaning | NIRCAM | WFC3_IR | ACS_WFC |
|---|---|---|---|---|
| `telescope`, `diameter_m` | aperture for the kernel warm start and the blur gauge (0.42 lambda / D) | JWST, 6.5 | HST, 2.4 | HST, 2.4 |
| `pixel_scale` | nominal scale per channel [arcsec] (the model itself uses the header Jacobian) | short 0.031, long 0.063 | 0.12825 | 0.05 |
| `detectors`, `channel_of` | detector names (unique across instruments), channel = kernel render pitch and level grids | NRCA1-NRCB4, NRCALONG, NRCBLONG; short / long | WFC3_IR; one channel | ACS_WFC1, ACS_WFC2 (from CCDCHIP); one channel |
| `levels`, `sci_units` | input levels and the unit of the science plane | cal: MJy/sr; rate: DN/s | cal: ELECTRONS/S | cal: ELECTRONS |
| `dq_bits`, `dq_ignore_default`, `dq_kind` | the instrument's own bit meanings; bits that do NOT reject; word size | 32-bit; ignore `--dq-ignore-bits` (default 0: every flag rejects) | 16-bit; ignore 16 (hot) + 512 (bad or uncertain flat) = 528 | 16-bit; ignore 16 (hot) + 64 (warm) = 80 |
| `stripes` | per-exposure row and column stripes (detector terms) | yes | no | no |
| `min_valid_pixels` | a cut with fewer valid pixels is dropped | 2,000 | 2,000 | 2,000 |
| `bandpass_prefix` | filter-curve identifier = prefix + filter (SVO Filter Profile Service id) | `JWST/NIRCam.` | `HST/WFC3_IR.` | `HST/ACS_WFC.` |
| `reader` | file -> list of exposure records, one per chip | `read_nircam` (the existing reader, unchanged) | `read_hst` | `read_hst` |
| `saturation` | what the files carry | DQ 2; pipeline linearity step | DQ 256, 2048; calwf3 non-linearity correction and ramp fit | DQ 256, 2048; no non-linearity correction below full well |

The common exposure record (one dict per chip; `data.load_exposure_file` turns it into a cut, unchanged otherwise):
`sci`, `err` (file unit), `dq` (or `usable` from the torchfits reader), `dq_ok` (the instrument's policy as a function), `w` (header solution with `pixel_to_world_differentiable`),
`detector`, `instrument`, `chip`, `name_suffix` (`""` for one chip, `_sci<chip>` otherwise), `level`, `sci_unit`, `to_sky`, `pixar_a2` (nominal pixel area, arcsec^2), `mjd`, `exptime`, `filter`,
`wcs_name`, `saturation`, plus `area` and `photmjsr` for NIRCam.

Conversions to surface brightness (also in the module docstring); S is in MJy/sr per nominal pixel area, x is the science plane:

```latex
\text{JWST calibrated: } S = x, \qquad \text{JWST count rate: } S = \mathrm{PHOTMJSR}\; x \ \ (\text{applied by the model through } \kappa_e = 1/(a_{\rm nom}\,\mathrm{PHOTMJSR}))
```
```latex
\text{HST: } S = \frac{\mathrm{PHOTFNU}\times 10^{-6}}{\Omega_{\rm nom}\; t_{\rm unit}}\; x, \quad
\mathrm{PHOTFNU} = \frac{10^{23}\,\mathrm{PHOTFLAM}\,\lambda_p^{2}}{c}\ \text{(if absent)}, \quad
\Omega_{\rm nom} = \mathrm{IDCSCALE}^2\,\Big(\frac{\pi}{648000}\Big)^2\,\mathrm{sr}, \quad
t_{\rm unit} = \begin{cases}\mathrm{EXPTIME} & \mathrm{BUNIT=ELECTRONS}\\ 1 & \mathrm{BUNIT=ELECTRONS/S}\end{cases}
```
with PHOTFNU in Jy s per electron, PHOTFLAM in erg cm^-2 A^-1 per electron, lambda_p = PHOTPLAM in A and c = 2.99792458e18 A/s. The factor is applied to SCI and ERR at load time
(calibrated level), so the model, its equations and its units are unchanged. HST `_flt` / `_flc` are flat-fielded so that a uniform sky is uniform (the pixel-area map is not applied),
the same convention as the JWST `_cal` files; this is why HST enters at the calibrated level only. Values on the samples: WFC3/IR F105W 0.2955, F140W 0.2472, F160W 0.3966 MJy/sr per e-/s
(from PHOTFNU; the PHOTFLAM route agrees to better than 1e-3, tested); ACS/WFC F814W 0.004793 MJy/sr per electron (PHOTFNU absent, from PHOTFLAM, PHOTPLAM and EXPTIME 544 s).

Header solution of HST chips: astropy with every distortion term (polynomial plus the ACS look-up tables `D2IMARR` and `WCSDVARR`, which torchsky's reader does not read; audit section C).
`--hst-wcs apriori` (default) takes the linear part from the a priori alternate solution (the guide-star catalogue one, `IDC_x-GSC...`, else the plain `IDC_x`) and never an a posteriori
`-FIT_*` or `-HSC*` one (derived by registering the images, which the fit must not use, as for JWST); `--hst-wcs archive` takes the primary solution as served (Gaia-aligned on these files).

What else changed, item by item of the audit list (section C):

| audit item | now |
|---|---|
| `data.py` reads `SCI`, `ERR`, `DQ`, first `SCI` only | the instrument's reader; every `SCI` extension is its own exposure and detector |
| detector name from primary `DETECTOR` | from the reader (`ACS_WFC1`, `ACS_WFC2`, `WFC3_IR`; NIRCam unchanged) |
| `svo_id` (UVIS wrong, everything else ACS) | `Instrument.bandpass_id`; `data.svo_id` kept as a wrapper |
| filters sorted by the digits of the name | one instrument: the old order (names without digits after, by name); several: by pivot wavelength |
| 2,000-pixel minimum | `min_valid_pixels` per instrument (2,000 for all three; Spitzer or GALEX would need a few hundred) |
| count-rate file by name substitution | `nircam_rate_path`; HST at `--input-level rate` is refused with the reason |
| diameters for three names only | `diameter_m` per instrument, carried per exposure (`extra["diameter_m"]`), read by `build_meta` and `filt_info_of` as before |
| one global `--dq-ignore-bits` | per-instrument policy; `--dq-ignore-bits` is now NIRCam's own; `--dq-ignore WFC3_IR:528,ACS_WFC:80` changes any instrument |
| no lookup-table distortion | astropy reader for HST (above) |
| ACS in electrons, unconverted | `to_sky` with EXPTIME |
| stripes `chan.startswith("NIRCAM")` | `Instrument.stripes` |
| spectral domain and curve grid constants (finding 4) | section 2 |

Also: the filter name of the list of exposures is checked against the file header (refusal on a mismatch); `info["insts"]` holds the instrument keys.

Deviations from the audit's "smallest interface": no `kernel_start(lambda)` for seeing (no ground-based instrument yet); no amplifier-quadrant terms (no model change allowed; ACS and WFC3
quadrant offsets are not modelled); the per-exposure `pixel_to_sky` is evaluated once at load time, not differentiated (the model's learned geometry acts on top, as for NIRCam); the record
gives `level` per chip but the model still has one level per run (`--input-level`), so mixed levels are refused rather than supported.

## 2. Spectral domain, filter-curve grid and number of spectral coefficients (finding 4)

`instruments.spectral_setup(fset, cfg)` runs in `cli.main` right after loading (also on `--heldout-only`), and in `tests/synth_small.py`:

* `--spectral-domain union` (default): domain = hull(0.70-5.30 um, [smallest quadrature node / 1.02, largest node x 1.02]) over all loaded filters. It sets `sky.SPECTRAL_DOMAIN`, read by
  `sky.lam_unit`, i.e. by the spectral basis and by every wavelength input of the kernel, throughput and distortion networks. `fixed` = the old constants.
* Filter-curve input grid = hull(0.55-5.6 um, union of the curves' supports T >= 1e-3 max), log-spaced at the old density (48 points over ln(5.6 / 0.55)); a filter entirely outside
  the old grid (ACS F435W) no longer becomes an all-zero input (tested).
* `--spec-k-rule span`: K = 1 + round((`--spec-k` - 1) ln(hi / lo) / ln(5.30 / 0.70)), i.e. the same bump spacing in ln(lambda) (0.184, 0.080 dex; bump width = spacing) as 12 bumps over the
  old domain. Default `fixed` (K = `--spec-k` = 12 whatever the span). NIRCam alone gives K = 12 under both rules.
* Approximation behind the K rule: the spectrum is resolved uniformly in ln(lambda); the bump spacing that suits 0.7-5.3 um is assumed to suit the blue (4000 A break, emission lines in
  the HST blue filters are narrower than one bump), and nothing adapts the resolution to where filters are dense. With `fixed` and a wider domain the resolution per filter gets coarser
  (12 bumps over 0.38-5.3 um = 0.104 dex spacing).
* Why a hull with the old domain rather than the pure union: the pure union shrinks the domain for any narrower filter set (0.80-4.98 um for the SMACS NIRCam filters), which would change
  every NIRCam fit; the hull keeps NIRCam fits identical whenever their nodes lie inside 0.70-5.30 um (all NIRCam wide filters F090W-F444W; F070W's nodes may fall below 0.70 um and then widen it,
  which is the fix the audit asked for, and a change for such a fit).
* For the cluster's JWST plus HST list (ACS F435W, F606W, F814W; WFC3/IR F105W-F160W; NIRCam F090W-F444W), with stand-in curves (pivot and width) and Q = 4 to 8: domain about 0.38-5.30 um,
  K = 15 with `span`, curve grid 0.34-5.6 um with 58 points. With the published curves the numbers will differ slightly; the log line `spectral domain ...` and `metrics["spectral_setup"]` give them.

## 3. Checks on the real files (no fit): `tests/check_instruments_real.py`

Every file read by its instrument's reader and cut by the package's own `load_exposure_file` around a virtual box of 140 arcsec centred on the NIRCam NRCB1 F090W exposure
(`jw02736001001_02101_00009_nrcb1_cal`), so positions are header positions in one tangent plane and science planes are in MJy/sr after conversion. Sources: 23 bright compact sources
found in that exposure (peaks above 10 robust sigma after a 2-pixel smoothing, isolated by 1.5 arcsec, more than half the 0.8-arcsec flux inside 0.4 arcsec), and 27 (F090W) and
24 (F150W) found the same way in the official mosaic cut-outs. Results: `tests/instrument_results/real_checks_apriori.json` and `real_checks_archive.json` (60 s each).

### (a) Units: HST / JWST aperture-flux ratio, median over sources [quartiles] (n)

Apertures of radius r centred on the JWST centroid plus the measured HST translation (b); background = median of the 1.5-2.5 arcsec annulus; flux = sum of (S - background) x pixel area.

| HST exposure | JWST reference | r = 0.4" | r = 0.8" | r = 1.2" |
|---|---|---|---|---|
| WFC3/IR F105W iczgz2nsq | exposure F090W | 1.215 [1.15, 1.24] (20) | 1.198 | 1.221 |
| WFC3/IR F105W iczgz4kcq | exposure F090W | 1.232 [1.17, 1.27] (20) | 1.226 | 1.249 |
| WFC3/IR F105W iczgz2nsq, iczgz4kcq | mosaic F090W | 1.245, 1.266 (27) | 1.267, 1.276 | 1.272, 1.274 |
| WFC3/IR F160W iczgz2nxq, iczgzbmgq | mosaic F150W | 0.916, 0.918 [0.89, 0.95] (24) | 0.927, 0.930 | 0.941, 0.927 |
| WFC3/IR F140W iczgz6j9q | mosaic F150W | 0.814 [0.79, 0.85] (24) | 0.821 | 0.826 |
| ACS/WFC F814W jczgz3k8q chip 2 | exposure F090W | 0.770 [0.74, 0.87] (21) | 0.794 | 0.827 |
| ACS/WFC F814W chip 1, chip 2 | mosaic F090W | 0.812 (8), 0.758 (16) | 0.818, 0.788 | 0.818, 0.831 |

All ratios are 0.76 to 1.28: of order 1, and nowhere near a factor 10 or the pixel-area ratios (17 for WFC3/IR against NIRCam short-wave pixels, 2.6 for ACS). What limits them: colour (red
cluster galaxies: F105W / F090W = 1.23 means f_nu rising as lambda^1.3, which predicts F814W / F090W 0.86, F140W / F160W 0.88; measured 0.79-0.83 and 0.88), the broader HST point-spread
functions (aperture losses a few per cent larger for HST at these radii), the absolute calibrations of the two observatories (a few per cent), and for F150W a mosaic rather than an exposure
(no F150W exposure on this machine; the F090W mosaic gives ratios 2-6 % higher than the F090W exposure). The F150W comparisons are about 10 % below the lambda^1.3 extrapolation (expected
F160W / F150W 1.03, measured 0.93); I cannot separate colour, aperture loss and calibration here, and a units error would be 10 times larger.

### (b) Geometry: HST positions against the NRCB1 F090W exposure, each through its own header solution

Windowed centroids (Gaussian window 0.12-0.15 arcsec); similarity fit HST = (1 + s) R(theta) JWST + t over 17-20 sources spanning 32-35 arcsec; uncertainties from the residuals.

| HST chip | t (a priori WCS) [mas] | t (archive WCS) [mas] | scale s [ppm] | rotation theta [arcsec of angle] | residual rms [mas] |
|---|---|---|---|---|---|
| WFC3/IR F105W iczgz2nsq | (-162, -476) | (1, 9) | 259 +- 230 | -48 +- 47 | 22.0 |
| WFC3/IR F105W iczgz4kcq | (-153, -488) | (14, 15) | 92 +- 238 | 9 +- 49 | 22.6 |
| WFC3/IR F160W iczgz2nxq | (-172, -458) | (-7, 3) | 220 +- 251 | -54 +- 52 | 23.8 |
| WFC3/IR F160W iczgzbmgq | (-239, -314) | (-19, 17) | 66 +- 191 | -70 +- 39 | 22.4 |
| WFC3/IR F140W iczgz6j9q | (384, -303) | (-7, 15) | 61 +- 218 | -81 +- 45 | 25.5 |
| ACS/WFC F814W chip 2 | (30, 90) | (-11, 13) | -89 +- 253 | -60 +- 52 | 27.4 |
| floor: NIRCam F277W NRCBLONG (same visit) | (3, 1) | | -337 +- 292 | 61 +- 60 | 21.9 |

(scale and rotation from the a priori run; the archive run gives the same within the errors.) Translations with the a priori HST solution: 0.39-0.51 arcsec (WFC3/IR) and 0.10 arcsec (ACS):
the expected HST pointing errors, removed by the registration pre-pass. With the Gaia-aligned archive solution they drop to 25 mas or less, so this JWST `_cal` header is within about 25 mas of the
Gaia frame here. Scale and rotation are zero within 1.8 sigma for every chip; a scale or rotation error of 750 ppm or 150 arcsec of angle (3 sigma, about 25 mas at the edge of the
35-arcsec span) would have been seen. The 22-27 mas residuals are centroid noise of galaxies in different bands at different resolutions (the NIRCam pair from one visit shows the same floor).
ACS chip 1 matched no source of the NRCB1 exposure (it lies outside it); against the F090W mosaic its translation is the same as chip 2's (0.13 arcsec, archive solution).

### (c) Valid pixels and noise

| file | kept by the policy, whole chip | DQ == 0 only | inside the box | median ERR [MJy/sr] | blank-sky scatter / ERR | first differences / sqrt 2 / ERR |
|---|---|---|---|---|---|---|
| NIRCam F090W NRCB1 (ignore 0) | 0.887 (0.990 ignoring the jump bit 4) | 0.887 | 0.887 | 0.0243 | 1.03 | 0.96 |
| NIRCam F277W NRCBLONG (ignore 0) | 0.840 (0.976 ignoring bit 4) | 0.840 | 0.846 | 0.0055 | 1.27 | 1.01 |
| WFC3/IR, 5 exposures (ignore 528) | 0.984 | 0.960 | 0.984-0.986 | 0.016-0.023 | 0.82-0.91 | 0.78-0.85 |
| ACS/WFC F814W, 2 chips (ignore 80) | 0.965, 0.966 | 0.932 | 0.960, 0.964 | 0.044, 0.045 | 1.19, 1.12 | 1.22, 1.13 |

Blank sky = pixels kept by the policy after three passes of masking above 3 robust sigma, dilated by 3 pixels; scatter = 1.4826 x median absolute deviation. What it measures: whether the
pipeline sigma matches the pixel-to-pixel noise of empty sky. NIRCam: yes (F277W's excess in the plain scatter is faint structure; the differences remove it). WFC3/IR: ERR is 10-20 % larger
than the scatter (the fit's learned noise scale, bounded to a factor 2, can take it; it would otherwise down-weight HST). ACS: ERR is 12-22 % smaller (unflagged cosmic rays in a single
exposure, and the hot and warm pixels the policy keeps). The WFC3/IR bit counts (fraction of the chip): 4: 0.4 %, 8: 0.3 %, 16: 0.8 %, 32: 1.3 %, 256: 0.1 %, 512: 1.9 %, 4096: 0.2-0.5 %; ACS: 16: 1.6 %, 64: 2.0 %, 128: 0.8 %, 1024: 1.4 %, 4096: 1.0 %.

## 4. Proofs that nothing else changed

| check | before | now |
|---|---|---|
| grid, 40 steps, new defaults, `--gauge-ref-inst 0` (`synth_grid40_defaults_v2.json`) | `579e2b1c1d6490dc5b85d59c1f48f8797d114b3a77c921e2d38da44d1872473f`, held-out 7.320529686517952 | identical, 7.320529686517952 (155 s) |
| items, 150 steps (`synth_items150_defaults_v2.json`) | `ba5f9c80a78e5e7f0f9a4dd313a04639dd895f7c8569c89f48d89026522d228c`, held-out 3.413583226768767 | identical, 3.413583226768767 (278 s; over the 4-minute guide, as recorded before) |
| two instruments through `cli.main` (`tests/synth_two_instrument_cli.py`), 6 steps | package copied before any edit: state `fdb9ab9c...1dab0a`, held-out 469.4703 (input noise) / 448.4043 (model noise) on 1,552 pixels | identical (state hash, summary and per-exposure scores), test `test_instruments.py` |

`tests/synth_small.py` now calls `spectral_setup` (when the package has it), so the two hash runs go through the interface; their output files (in my scratch folder, not over the recorded
ones) report domain 0.70-5.30 um, curve grid 0.55-5.6 um x 48, K = 12. The recorded result files were not overwritten. One edit to `data.load_field` came after the two hash runs (the check
of the header filter against the list); `tests/synth_small.py` does not call `load_field`. The legacy hashes (`--legacy-defaults`) were not rerun.

The two-instrument run without `--gauge-ref-inst` stops (`SystemExit`), as does `resolve_instrument_options` on the real NIRCam + WFC3/IR + ACS field loaded through `load_field` (tested):
the flux scale of a several-instrument fit needs an explicit reference instrument.

## 5. Tests

Run file by file: `test_instruments.py` 7 (new; the real-file test needs `PYTHONPATH=/home/claude/torchsky/src` and skips without it), audit fixes 4, noise floor guard 5, sky zero 3,
pixel gain 7, filter shift 4, item tiles 1, rate warm start 2, rate units 2, item unit 3, defaults v2 19, torchsky backends 9, truth smoke 2, item pyramid 14 (190 s). 82 pass, 0 fail.
Two edits to an older test, both from the interface: `test_audit_fixes.py` expects the instrument KEY `WFC3_IR` as the curve's label (was the list's `WFC3`), and its stand-in for `_read_cuts`
takes the new `inst` argument.

New tests: data-quality policies (NIRCam bit-identical to the old expression for ignore masks 0, 2, 4, 6; HST 16-bit with the top bit); the conversion formula on the real WFC3/IR header
(PHOTFNU and PHOTFLAM routes agree, by-hand value, ELECTRONS divided by EXPTIME, other units refused); names, aliases and refusals (unknown instrument, WFC3/UVIS, HST at the count-rate level);
spectral domain unchanged for the synthetic and NIRCam-like filters at Q = 3 and 8 (curve vectors bit-identical, lam_unit equal to the old formula) and widened for a blue filter (its nodes
inside the basis, partition of unity, K by the rule, non-zero curve input); real NIRCam + WFC3/IR + two-chip ACS files through `load_field` (instruments, filter order by pivot, one exposure per
chip with detectors ACS_WFC1 and ACS_WFC2, splits by rootname, stripes only NIRCam, diameters 6.5 and 2.4 m, WFC3/IR cut = file e-/s x factor at every valid pixel, NIRCam cut = file values exactly,
pixel scales 0.128, 0.050, 0.031 arcsec from the headers, refusal without a reference instrument, HST at the count-rate level refused); the two-instrument command-line equality above.

## 6. Every approximation and shortcut

* Stand-in filter curves in the tests: no curve cache exists on this machine (`~/.cache/torchsky/svo` absent, no `$TORCHSKY_SVO_CACHE`) and no network. The real-file test uses a box of
  full width sqrt(12) x PHOTBW centred on PHOTPLAM for HST and a box at 0.90 um for NIRCam F090W (`instruments.stand_in_filter`, set through `data.FILTER_CURVE`; never used by a real run).
* ACS chips are separate exposures with their own background, affine registration, gain and codes: the rigid tie between the two chips of one pointing is not used (the same as NIRCam's
  detectors today). Prediction files of a two-chip exposure would be named `<rootname>_sci<chip>.npz`.
* HST header positions come from astropy (not differentiable) once at load time; the learned geometry acts on top, as for NIRCam.
* The HST conversion uses IDCSCALE^2 as the nominal area (WFC3/IR pixels are 0.135 x 0.121 arcsec; PHOTFNU is defined for the drizzled scale IDCSCALE), and assumes the flat-field convention
  (uniform sky gives a uniform `_flt`), the same as JWST `_cal`.
* Data-quality defaults are a choice: drizzlepac's final bits for these detectors without saturation (WFC3/IR keeps hot 16 and bad-flat 512 pixels, ACS hot 16 and warm 64). Unflagged cosmic
  rays remain in single ACS exposures (check c). Persistence in WFC3/IR is not flagged in `_flt` and is not handled.
* The a priori HST solution is my reading of "never pre-aligned"; `--hst-wcs archive` is there if the maintainer decides otherwise.
* The spectral domain is a hull with the old domain (section 2), margin 2 % on the nodes; the curve-grid point count rounds up; the K rule rounds.
* With several instruments the filters are ordered by pivot wavelength (the digits of HST and NIRCam names are not comparable); one instrument keeps the old order.
* Check (a) uses the official mosaic for F150W (no F150W exposure here); check (b) uses galaxies, not stars, so centroids depend on band and resolution (floor about 22 mas).

## 7. What an HST entry in the list of exposures must contain

The same keys as a NIRCam entry, with HST values; the reader takes everything else from the file:

* `file` (`iczgz2nsq_flt.fits`, `jczgz3k8q_flc.fits`), `path` (or `--data-root` + `file`), `rootname` (MAST rootname, `iczgz2nsq`; the splits lists use it; both chips of an ACS file follow it).
* `instrument`: `WFC3_IR` or `ACS_WFC` (or `WFC3` / `ACS` with `detector` `IR` / `WFC`); `detector`: `IR` or `WFC` (the reader names the detectors itself).
* `filter`: the header's `FILTER` (WFC3) or the non-CLEAR one of `FILTER1` / `FILTER2` (ACS); checked against the header at load.
* `mjd`: `EXPSTART` of the primary header (not the association start in `hst_imaging_unique.csv`; that file's exptime is also the association's, 2,573 s against 353 s in the header of iczgz2nsq).
* `boxes` {box id: coverage} as for NIRCam; `dither_group`: one group per pointing and filter (for example `<rootname[:6]>|<filter>` or the association), used for the validation carve-out.
* Filter names must be unique across instruments: F606W exists on ACS/WFC and WFC3/UVIS in the cluster's list (the package refuses a shared name; WFC3/UVIS is not described yet anyway), F814W likewise if WFC3/UVIS is added.
* Supported: WFC3/IR `_flt` (88 of the 116: F105W 26, F110W 12, F125W 8, F140W 20, F153M 2, F160W 20) and ACS/WFC `_flc` (14: F435W 4, F606W 5, F814W 5). Not supported: WFC3/UVIS `_flc` (6;
  needs a description: two chips, its own DQ bits and CTE, `HST/WFC3_UVIS2.` curves whose SVO id I could not verify offline) and WFPC2 `_c1m` (8; different format and units).
* HST exposures should be training exposures for the first fit: the held-out masks and the scorer of the protocol are built for JWST `_cal` files; a held-out HST prediction would be in MJy/sr, not in the file's e-/s (divide by `to_sky`, logged per exposure).

## 8. What a cluster run with JWST plus HST needs

* Files: the `_flt` / `_flc` files on `/arc` (ACS files are 168 MB each and read whole), and astropy (already the default loader).
* Curve cache: in the cache directory the run uses (`--svo-cache`, else `$TORCHSKY_SVO_CACHE`, else `~/.cache/torchsky/svo`), torchsky's files `filter_HST_WFC3_IR.F105W.json`, `filter_HST_WFC3_IR.F110W.json`,
  `filter_HST_WFC3_IR.F125W.json`, `filter_HST_WFC3_IR.F140W.json`, `filter_HST_WFC3_IR.F153M.json`, `filter_HST_WFC3_IR.F160W.json`, `filter_HST_ACS_WFC.F435W.json`, `filter_HST_ACS_WFC.F606W.json`,
  `filter_HST_ACS_WFC.F814W.json` (`instruments.curve_cache_file(id)` gives the name), beside the NIRCam ones the cluster already has. Without them torchsky tries the network and, without astroquery, stops.
* Arguments: `--input-level cal`; `--gauge-ref-inst I` where I is NIRCam's index in the sorted instrument keys (`["NIRCAM", "WFC3_IR"]` gives 0; with ACS `["ACS_WFC", "NIRCAM", "WFC3_IR"]` gives 1;
  the log line `instruments [...]` shows it); `--spec-k-rule span` if ACS blue filters are in (otherwise 12 bumps cover 0.38-5.3 um); `--hst-wcs apriori` (default); `--dq-ignore-bits` as in the NIRCam runs
  (NIRCam only now); the kernel normalisation radius resolves to 2 arcsec by itself with two instruments.
* Epochs: the HST exposures of SMACS span about 63 days (MJD 57817-57880), so with `--epoch-per-instrument 1` the epoch axes switch on for the whole model because HST has several epochs of its own
  (one switch for the model, DEFAULTS_V2 item 5). For a first joint fit comparable with the NIRCam ones, `--epoch-axes 0` (or `--epoch-min-days 100`).
* Memory per HST exposure in a 20-arcsec box: about 200 x 250 WFC3/IR pixels, two ACS cuts of about 300 x 500; small next to NIRCam.

## 9. Not done

* No fit on real data, no new list of exposures (yours), no change to the model's equations, defaults or recorded hashes.
* WFC3/UVIS, WFPC2, Spitzer, GALEX, ground-based: not described. Ground-based needs a seeing kernel start (audit "smallest interface" item 1) and the blur reference (`model.py:318-326`) still assumes
  diffraction-limited kernels.
* Amplifier-quadrant offsets (ACS, WFC3) and a per-filter detector pattern (finding 10): model changes, not made.
* The count-rate scorer in `score_paired.py` still reads PHOTMJSR (JWST only); not touched.
* HST at the count-rate level: refused (it would need the pixel-area map multiplied back into `_flt`).
* Check (a) for F160W / F140W against a F150W exposure: no such exposure on this machine; done against the official mosaic.

## 10. Files

New: `fieldop/instruments.py`; `tests/test_instruments.py`; `tests/check_instruments_real.py`; `tests/synth_two_instrument_cli.py`; `tests/tsky_results/two_instrument_cli_before_interface.json`;
`tests/instrument_results/real_checks_apriori.json`, `real_checks_archive.json`; this file.
Changed: `fieldop/data.py` (reader dispatch, per-instrument data-quality policy, conversion, chips, filter order, constants from the descriptions), `fieldop/sky.py` (`SPECTRAL_DOMAIN`,
`set_spectral_domain`), `fieldop/bandpass.py` (`FilterSet.set_curve_grid`), `fieldop/cli.py` (options `--spectral-domain`, `--spec-k-rule`, `--dq-ignore`, `--hst-wcs`; `spectral_setup` and
instrument log lines; `metrics["spectral_setup"]`; three options added to the structure list), `fieldop/__init__.py` (description, F1 text); `tests/synth_small.py` (calls `spectral_setup`),
`tests/test_audit_fixes.py` (two edits, section 5). Package 5,727 to 6,215 lines.
