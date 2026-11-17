# Upstream notes for the maintainer

Collected while integrating `GaussianItemScene`, `PolynomialDetectorDistortion` /
`TelescopeAttitude` / `TelescopeFrameDetectorToSky`, and
`field_observation_from_exposure_file`. Nothing below was fixed; line numbers are in the
tree as integrated. Delete this file once triaged.

## A. Existing library behaviour that looks wrong or inconsistent

1. `src/torchsky/map/quality.py:48-52` (`DQPolicy.usability_mask`): `reject_bits=None` accepts only `dq == 0`, but `reject_bits=0` accepts everything. Two spellings of "no rejection bits" mean opposite things.
2. `src/torchsky/map/quality.py:72-80` (`DQPolicy._coerce`): `.to(torch.int64)` on a signed int32 DQ plane (BITPIX 32, no BZERO) is negative where bit 31 is set, so high-bit masks misbehave. JWST uint32 planes are fine.
3. `src/torchsky/wcs/detector_astrometry.py:674` (`DetectorToSkyTransform.__init__`): `attitude.requires_grad_(False)` silently stops a learnable attitude from learning.
4. `src/torchsky/wcs/detector_astrometry.py:132-155` (`RegularGridTransform2D.inverse` tolerance): `rtol` scales the target value, not an output scale, so `rtol` alone is a poor criterion near zero.
5. `src/torchsky/map/field/types.py:601` (`FieldObservation.to`): `dtype=float32` casts every module in `detector_to_sky`, silently degrading float64 detector tables. The new telescope-frame modules guard against it with a private `_apply` override; other transforms do not.
6. `src/torchsky/map/field/types.py:96` (`DetectorToSkyTransform` Protocol): documented as forward-only, but `map/field/observation_ops.py:422` reads `world_to_pixel` through `getattr`. Declare it (optional) in the protocol, or stop probing for it.
7. `src/torchsky/map/field/observation_ops.py:385-422` (`crop_field_observation_to_shared_sky_patch`): calls `world_to_pixel` with defaults, so a sky point far from the detector makes the new transform raise "did not converge" and abort the crop instead of reporting "not on this detector". `world_to_pixel(..., fail_closed=False)` returns NaN, which the crop already handles.
8. `src/torchsky/map/field/types.py:418` (`FieldObservation.value_semantics`): free string checked only for non-empty; consumers compare with literals (`"surface_brightness"`, `"single_integrated_read"`, Foam `"integrated_flux"`), so a typo fails late. No string exists for count rate per pixel.
9. `src/torchsky/map/field/observation_ops.py:48,261` (`copy_header_with_crop`, `crop_field_observation`): `PIXAR_*` and other cards are left as they are. Correct for a crop, but undocumented.
10. `src/torchsky/map/scenes.py:52-288` (`SceneRepresentation.render_to_target`): no documented shape, unit or `kwargs` contract; implementations return different things (`DiscreteTargetImage` returns its data unchanged; the field model and `SourceCatalogScene` return `(H, W)`). `as_scene_representation` wraps any `nn.Module` as `ContinuousFieldModel`, so a scene that is also a module must subclass `SceneRepresentation`.
11. `src/torchsky/scene/sources.py:765-773` (`SourceCatalogScene.render_to_target`): reads `parameter_values`, `source_indices`, `chunk_size` via `kwargs.get` and ignores unknown keys silently; `GaussianItemScene` raises `TypeError` for them.
12. `src/torchsky/scene/sources.py:615` (`SourceCatalogScene._gaussian_stamp`): box of `4 x major sigma + PSF halo + 2` pixels, no renormalisation, so flux is not exactly the coefficient (agrees with `GaussianItemScene` to 3e-4 of peak).
13. `src/torchsky/scene/__init__.py:153` and `src/torchsky/wcs/__init__.py:174`, `src/torchsky/field/__init__.py:203`, `src/torchsky/map/field/__init__.py:317`: `__all__` lists are not sorted (RUF022), and `wcs/__init__.py` and `field/__init__.py` fail I001 with the local ruff, with or without the new lines. `field/__init__.py` imports builders eagerly while `map/field/__init__.py` is a lazy facade, so the two registration patterns differ.
14. torchfits 1.3.1 (outside this repo): `torchfits.read(path, "NAME")` ignores `EXTVER` and raises an opaque `RuntimeError ... 'list' object has no attribute 'get'` for an unknown name; `TensorHDU(data=None)` as primary breaks `HDUList.write` with `ValueError('No file handle available')`. `exposure_io.py` works around both.
15. `PSFSpec.learned(model=..., base_sigma_pix=...)` builds a `learned_lowrank` spec and `apply_psf_spec` calls `model(u_pix, lam_norm)`; a `ConditionedKernelPSF`-style model (including `NestedKernelPSF`) is accepted at construction but fails when applied (`TypeError`, forward takes 2 positional arguments but 3 given). Only `LearnedPSFOp(model=...)` works for them (`src/torchsky/map/psf*`; spec construction in `PSFSpec.learned`).
16. `src/torchsky/map/operators/_inverse_stationary.py`: the stationary `LearnedPSFOp` adapter requires `isinstance(model, ConditionedKernelPSF)`, so `NestedKernelPSF` cannot be bound to the stationary inverse path; any model exposing `effective_kernels` and `kernel_size` would do.
17. `tests/test_psf_transport_invariants.py` (`_scatter`): shape error when a stamp is larger than the image (empty slices); the copy in `tests/test_nested_kernel_psf.py` skips those offsets.
18. `apply_kernel2d` normalises by the sum and accepts negative entries while `PSFSpec.kernel` rejects negative kernels (pinned by `test_signed_kernel_helper_keeps_its_existing_normalization_contract`).
19. `map/operators/__init__.py:298`: `__all__` not sorted (RUF022), plus one more ruff error, with or without `NestedKernelPSF`.

## B. Limits of the new code the maintainer should know about

1. `scene/gaussians.py`: the default `mode` is `"direct"`; `"pyramid"` is far cheaper with wide items (10,000 items, 512 x 512, K = 4: 22 s against 0.95 s forward). Consider changing the default.
2. `scene/gaussians.py`: the cell integral is an isotropic Gaussian of variance 1/12 pixel^2, not an erf; it is a poor model below about 1.5 px (1.9e-2 of peak at 0.5 px). Dense pyramid cascade grids are not covered by `chunk_elements`. Second derivatives are unsupported. `lat` outside [-90, 90] and minor > major are checked only at construction. No CUDA or MPS run.
3. `wcs/telescope_frame.py`: no filter-dependent offsets, filter-dependent distortion or velocity aberration. Against the pipeline's calibrated SIP, most filters differ by 3 to 15 mas median; the two F200W exposures differ by about 60 mas (a constant pixel shift, about 1 and 1.6 px). Parameters stay float64 under `.to(dtype)` and MPS is unsupported. No pixel coordinates beyond the residual grid are accepted. Reference coefficients are not shipped (author's question: where should they live: a data package, the harness repository, or read from `pysiaf`).
4. `map/field/exposure_io.py`: empty sky-box cuts return `None`; HST `FILTER1`/`FILTER2` band names are not parsed; the JWST `AREA` plane is not read because `FieldObservation` has no pixel-area slot; no real SIP file, no HST file, no CUDA or MPS run; the coarse-to-fine box search is exact only when the pixel scale varies by less than the inflation margin within a 32-pixel cell.
5. `map/operators/nested_kernel.py`: not exactly shift-invariant (about 3e-5 of the peak with source phase modulo the coarsest pitch); dense match is to an interpolation error (4e-5 of peak for point and smooth sources, 2e-4 for white noise at extent 200), not round-off; kernels must be non-negative; coordinates must be a unit-pitch grid; `kernel_size` is the exact support (above `2 extent + 1`), so the tiling halo can equal the image for a large extent. No variance propagation, no CUDA/MPS run, no batch dimension; non-callable `kernel` raises `TypeError` (TRY004 style deviation from `ValueError`). Not ported from the prototype: training-time lattice jitter, sky-aligned level grids, per-pixel quadrature, B-spline continuous readback, log-kernel network conventions and checkpointing.

## C. Open questions from the authors

1. Should `FieldObservation` get a pixel-area plane (from `AREA` or the WCS) so count-rate input can be forward-modelled directly?
2. Which `value_semantics` string should a count rate per pixel carry (`"integrated_flux"` or a new one)? It stays `None` until decided.
3. Empty cuts: keep returning `None`, or add a `raise_if_empty` switch for single-file use?
4. Should instrument DQ bit tables (JWST, HST) live in a named module (for example `torchsky.instrument`) so callers can write `DQPolicy(reject_bits=JWST.DO_NOT_USE)`? `torchsky.quality.QualityBit` coincides with JWST only for bits 1, 2, 4.
5. Is `FILTER-PUPIL` the wanted band-name spelling for NIRCam filters with a pupil element? It is not matched against `instrument.nircam_imaging_band` names.
6. Should the crop code pass `fail_closed=False`, or should `TelescopeFrameDetectorToSky.world_to_pixel` default to NaN? (see A7)
7. Should the filter offset be a learnable constant pixel offset on the distortion, initialised from the reference file?
8. Should the grid of `AffineFreeResidualGrid2D` extend half a pixel beyond the detector for training crops?
9. Should the `FieldObservation.detector_to_sky` protocol declare `world_to_pixel`? (see A6)
10. `NestedKernelPSF`: keep `kernel_size` as the exact support (needed for exact tiling) or expose a smaller tile-friendly value (tiling then approximate)?
11. Linear (default, non-negative response) or cubic read-back (1.2x to 1.5x better on smooth sources, small negative lobes)?
12. Are the defaults `core_half_width = 16` and `extent = 128` acceptable, or should the constructor require both?
13. Open the stationary-inverse adapter check to any model with `effective_kernels` and `kernel_size` (see A16)?
14. Track `torchsky.map.operators.nested_kernel` in the API manifest and registry (add to `NAMESPACES` in `scripts/api_manifest.py` plus a registry page), or wait as for the other learned PSF classes?
