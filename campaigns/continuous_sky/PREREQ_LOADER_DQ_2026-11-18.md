# Loader and data-quality prerequisites, 2026-11-18

Scope: `map/quality.py`, `instrument.py`, `map/field/exposure_io.py`, `map/field/types.py`,
`map/field/observation_ops.py`, and their tests. Nothing was committed. DQ means data quality.
Tests: `tests/test_dq_bits_and_value_semantics.py` (new) and `tests/test_field_exposure_io.py` (extended).
Changelog entries were added to `docs/project/changelog.md`.

## Per fix

1. **`reject_bits` meaning.** Documented in the `DQPolicy` docstring: `None` rejects any set bit
   (strict); `0` rejects nothing. Added `DQPolicy.strict()` and `DQPolicy.accept_all()`.
   Test: `test_named_constructors_spell_both_meanings` (plus the pin
   `test_reject_bits_none_is_strict_and_zero_accepts_everything`). Behaviour unchanged for existing
   callers (several in the tree use `reject_bits=0` as "accept all").
2. **Signed int32 DQ.** `_coerce` masks signed int8/int16/int32 back to their own width, so the
   plane is an unsigned bit field. Tests: `test_signed_int32_plane_is_read_as_an_unsigned_bit_field`,
   `test_every_signed_width_masks_to_its_own_width`, `test_quality_product_stores_the_unsigned_pattern`.
   Changed only for negative values; `QualityProduct.dq` and `propagated()` now return the non-negative pattern.
3. **Bit tables.** `JWSTDataQuality`, `WFC3IRDataQuality`, `ACSWFCDataQuality` (`IntFlag`) in
   `torchsky.instrument`. Tests: `test_jwst_table_matches_the_documented_bits`,
   `test_hst_tables_name_only_the_certain_bits`, `test_tables_compose_into_a_dq_policy`.
   New names only. No handbook was available offline, so the bits come from memory of the JWST pipeline
   documentation and the WFC3 and ACS data handbooks; the docstrings cite them without table numbers.
   Evidence from the real files: in both NIRCam files, bit 31 (REFERENCE_PIXEL) is set on exactly 32,704
   pixels, equal to the 4-pixel border of a 2048 x 2048 detector, and bits 19, 20, 21 (NO_GAIN_VALUE,
   NO_LIN_CORR, NO_SAT_CHECK) coincide with it, as expected. The HST bits could not be checked this way.
   - JWST left out: bit 7 (value 128, reserved) and any bit added after this table was written.
   - WFC3/IR named: 1, 2, 4, 16, 256, 512. Left out: 8, 32, 64, 128, 1024, 2048, 4096 and above.
   - ACS/WFC named: 1, 2, 4, 16, 32, 256, 512, 2048. Left out: 8, 64, 128, 1024, 4096, 8192, 16384
     (including the cosmic-ray flags).
   The owner should check the HST values against the handbooks before relying on them.
4. **Loader.**
   - (a) `with_extras=True` returns `ExposureFileData(observation, quality, metadata)`: the DQ plane
     (int64, same cut), and `INSTRUME`, `DETECTOR`, `AREA` (when present). `FieldObservation` has no slot
     for these, so they come beside it; the default return is unchanged. Tests:
     `test_with_extras_*`, `test_default_call_still_returns_a_bare_observation_without_instrument_cards`.
   - (b) `zero_fill_unusable=False` keeps file values (including NaN) in unusable pixels. Test:
     `test_zero_fill_can_be_switched_off`.
   - (c) Band from `FILTER1`/`FILTER2` when there is no `FILTER`, ignoring `CLEAR*`; two real elements
     join as `A-B`. Test: `test_hst_band_names_from_filter_wheels`.
   - (d) Float32 weights that overflow are clipped to the largest finite value and the pixel stays usable.
     Test: `test_weight_overflow_in_output_precision_stays_usable_with_clipped_weight`. **Behaviour change**
     (requested); the old test that pinned the drop was rewritten. A weight that underflows to zero is still unusable.
   - Also: `value_semantics` is validated before the file is read.
   - `ExposureFileData` was added to `__all__`, so `test_exported_through_lazy_facade` was updated.
5. **`value_semantics`.** `FieldValueSemantics` (`StrEnum`) and `VALUE_SEMANTICS` in `types.py`; the
   constructor raises `ValueError` listing the choices and stores a plain string. Accepted:
   surface_brightness, integrated_flux, integrated_flux_per_pixel, single_integrated_read, detector_counts,
   flux, count_rate_per_pixel. I found these literals by instrumenting the test suite, and `flux` by reading
   `map/product.py` and `map/scenes.py`, which copy a map-product value into an observation. Tests:
   `test_every_literal_used_in_the_tree_is_accepted`, `test_unknown_value_semantics_fails_at_construction_with_the_choices`,
   `test_enum_member_is_stored_as_a_plain_string`. **Behaviour change** (requested): custom strings now raise.
   A custom string on a `TargetRenderSpec` would now fail in `scenes.py`. The loader still leaves
   count-rate files at `None` by default; the new value is opt-in.
6. **Crop.** `fail_closed=False` is passed when `world_to_pixel` declares that keyword. Tests:
   `test_crop_far_from_the_detector_reports_none_instead_of_raising` (and two no-regression tests).
   Behaviour change only where it used to raise.
7. **`.to(dtype)`.** A `detector_to_sky` module is never cast to a narrower float; float32 modules still widen
   under float64; device moves still apply. Tests: `test_to_float32_keeps_a_float64_detector_table`,
   `test_to_float64_still_widens_a_float32_transform`, `test_device_move_still_applies_to_the_transform`.
   A float64 module under a float32 request is the case that changed.
8. **Housekeeping.** `ruff check --fix` on the five source files and the extended test file (import order,
   sorted `__all__`, `collections.abc`, unquoted annotations). `types.py` has no `__all__`. One
   pre-existing SIM102 remains in `types.py` and was left alone.

## Fail-before check

Run against the original versions of the five source files (with import stubs): 29 of the new tests fail
before the change, as intended. The two pin tests (`reject_bits` meaning, unusable `err`) pass before and after by design.

## Checks run (no Pixi here; `PYTHONPATH=src`)

- `tests/test_dq_bits_and_value_semantics.py` + `tests/test_field_exposure_io.py`: 95 passed.
- `test_quality`, `test_map_dq_parity`, `test_instrument`: 17 passed, 1 skipped (before the new tests).
- Wider run (every test file that mentions FieldObservation, value_semantics, DQPolicy or instrument, plus
  `test_device_to` and `test_docstrings`): `1 failed, 1270 passed, 88 skipped, 1 warning, 11 errors`. Identical
  to the same run before my changes (1 failed, 11 errors): `test_cli_map.py::test_cli_map_dry_run_writes_resolved_plan_json`
  and 11 `tests/test_map_*.py` files that cannot import the uncompiled `torchsky._cpp_ext`. Neither is related.
- `scripts/api_manifest.py --check docs/api-manifest.json`: "API manifest matches". `pyright` on the five files: 0 errors.
- Not run: `pixi run docs-contract` and `preflight-push` (no Pixi). `ruff check` and `ruff format --check` were run on touched files.

## Real-file check (report only)

| File | Shape | Band | INSTRUME / DETECTOR | Usable (default) | Usable (DO_NOT_USE only) |
|---|---|---|---|---|---|
| JWST nrcb1 `_cal` | 2048 x 2048 | F090W | NIRCAM / NRCB1 | 0.8872 | 0.9909 |
| JWST nrcblong `_cal` | 2048 x 2048 | F277W | NIRCAM / NRCBLONG | 0.8404 | 0.9890 |
| WFC3/IR `iczgz2npq_flt` | 1014 x 1014 | F140W | WFC3 / IR | 0.9604 | 0.9952 (bad detector + saturation) |
| ACS `jczgz3k8q_flc`, EXTVER 1 | 2048 x 4096 | F814W | ACS / WFC | 0.9324 | 0.9999 (bad detector + saturation) |
| same, EXTVER 2 | 2048 x 4096 | F814W | ACS / WFC | 0.9319 | |

Both JWST files return an `AREA` plane (2048 x 2048). HST files have none, so the key is absent.
The loader's EXTVER workaround covers ACS: it picks the extension by scanning headers and reading by HDU index.
Without `extver` the loader raises a clear error for the two chips; with `extver=1` or `2` the two chips differ
(EXTVER 1 has CCDCHIP 2 and EXTVER 2 has CCDCHIP 1, the usual ACS order), and each equals the direct
HDU-index read on usable pixels. WFC3 band, unit (`ELECTRONS/S`) and exposure time were read correctly.

## Left undone / for the owner

- HST bit tables are deliberately partial (see 3); confirm against the handbooks and extend.
- `count_rate_per_pixel` is not applied automatically to count-rate files (default stays `None`).
- Public exports: `ExposureFileData`, `FieldValueSemantics` are not added to the `map.field` facade or the API manifest.
- Underflowing weights (huge `err`) still make a pixel unusable.
- Docs guide (`docs/models/fields.md`) not edited: the user-visible paragraph about the unread `AREA` plane is now out of date.
