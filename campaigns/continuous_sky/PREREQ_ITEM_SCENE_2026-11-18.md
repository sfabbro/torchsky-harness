# Prerequisite: functional Gaussian-item rasteriser (2026-11-18)

Scope: `src/torchsky/scene/gaussians.py`, the export in `src/torchsky/scene/__init__.py`,
the Gaussian-scene tests, and the matching docs lines. Nothing in `map/quality.py`,
`map/field/*` or `instrument.py` was touched. No commits were made.

## 1. Signature

```python
torchsky.scene.render_gaussian_items(
    centers_pix,            # (C, 2) item centres (x, y) in pixels; 0-indexed pixel centres, x along columns
    shape_pix,              # (C, 2, 2) covariance in pixels^2, or (C, 3) rows (major, minor, angle_rad), 0 < minor <= major
    flux,                   # (C, K) fluxes; K planes at once; sets the output dtype
    shape,                  # (H, W)
    *,
    mode="direct",          # or "pyramid"
    output="flux_per_pixel",        # or "surface_brightness"
    pixel_area=None,        # required for, and only allowed with, output="surface_brightness"
    pixel_integration=True, # add 1/12 pixel^2 to the covariance (stand-in for the pixel cell integral)
    chunk_elements=4_000_000,
    truncation_sigma=4.0,
    patch_cap=128,          # largest patch half-size in nodes; None = unbounded (the old behaviour)
) -> Tensor                 # (K, H, W)
```

I kept the module's own convention (`centers_pix` as a `(C, 2)` tensor, then shape, then
coefficients) rather than separate `x`, `y` arguments, so that the bridge's `pix` tensor passes straight in.

Units. Everything is in pixels. `flux_per_pixel`: each pixel holds the flux in it, so a plane sums
to the flux of the items inside the grid (flux off the grid is dropped, not renormalised).
`surface_brightness`: that divided by `pixel_area` (the area of one pixel in the unit wanted, for
example arcsec^2). The prototype's coefficient grid is this form with `pixel_area = pitch**2`.

Differentiable to first order in `centers_pix`, `shape_pix` and `flux` (a custom backward that
recomputes patches chunk by chunk, so memory stays bounded by `chunk_elements`). Second
derivatives are not supported (as before). Nothing is held: no `nn.Parameter`, no WCS, no header.

Position precision: geometry is done in float64 inside the kernel whatever the input dtype
(this was already so in the private kernel). A float32 position at pixel 20000.3 is stored as
20000.30078, and the item lands there to better than 1e-3 pixel. The 0.05 arcsec loss mentioned in the
brief came from float32 longitude and latitude in the module's WCS step, which the functional form avoids,
since the caller supplies pixel positions.

## 2. What changed

- New public `render_gaussian_items` (exported from `torchsky.scene`; `docs/api-manifest.json` and the
  generated `docs/reference/scene.md` and `index.md` regenerated; changelog and the scenes guide updated).
- `_rasterize_gaussians` is now a three-line wrapper around it (kept because the existing tests import it).
- `GaussianItemScene.render_coefficients` and `render_to_target` call `render_gaussian_items`. The scene has a new
  keyword `patch_cap` (default 128). One implementation.
- Only user-visible change to existing behaviour: the new patch cap (section 3). Error messages for the
  coefficients argument still contain the word "coefficients".
- `_Level` gained a trailing `cap` field (default `None`), so existing private calls still work.

## 3. Patch cap

Direct mode builds a square patch of half-size `ceil(4 sigma_major / pitch)` nodes, bucketed. Now
`patch_cap` bounds it, so a patch has at most (2 cap + 1)^2 nodes. Default 128 equals the prototype's direct
cap and the end of the bucket table. Items with a larger principal width up to about 32 pixels
(4 sigma plus the pixel term at most 128) are bit-for-bit unchanged (test asserts equality with `patch_cap=None`).

An item that would need more:

1. If its smaller principal variance qualifies for a coarse pyramid level, it is sampled there, even in `mode="direct"`.
   This is the prototype's pyramid behaviour, and its error is the pyramid's, about 2e-4 of the item's peak. A round item of
   sigma 300 pixels now uses patches of at most 255 x 255 instead of 2401 x 2401, and agrees with the unbounded result to
   3e-4 of its peak and to 1e-3 in total flux (test).
2. Otherwise (long thin item) the patch is cut at the cap and renormalised. Flux stays exact; the profile does not. This is
   a real approximation. Error as a fraction of the item's own peak (minor sigma 1.5 px, cap 128): major sigma
   40 px 0.002; 60 px 0.07; 100 px 0.38; 200 px 0.94. For such items, pass a larger cap (memory grows as its square) or `None`.
   The prototype's direct mode truncated silently the same way; its pyramid mode allowed 512 nodes, so for
   agreement with it on thin items above about 30 pixels, use `patch_cap=512`.

I did not implement an exact tiled patch for case 2 (it would need a two-pass normalisation across tiles and a
matching backward pass). Left undone.

## 4. Tests (`tests/test_scene_gaussians_functional.py`, 18 tests)

- `test_output_units_and_flux[direct|pyramid]`: flux per pixel sums to flux; surface brightness is that over `pixel_area`.
- `test_output_argument_validation`: bad `output`, `pixel_area`, `patch_cap`.
- `test_planes_are_independent_columns_of_flux[...]`: 12 planes at once equal single-plane renders; covariance and
  (major, minor, angle) forms agree.
- `test_module_form_is_the_functional_form[...]`: scene output equals the functional call on the scene's own geometry
  exactly, and equals a hand-built geometry to 2e-6 of the peak.
- `test_module_form_patch_cap_is_passed_through`.
- `test_gradcheck_float64[direct|pyramid]`: `torch.autograd.gradcheck`, float64, 3 items, 2 planes, both shape forms and
  surface brightness.
- `test_agrees_with_prototype_rasteriser[direct-30, pyramid-30, pyramid-60 with cap 512]`: skipped if
  `/home/claude/work/fieldop_tsky` is missing. Imports the prototype's `ItemSky` (read only).
- `test_float32_position_far_from_origin[...]`: item at pixel 20000.3 on a 20100-wide grid.
- `test_patch_cap_leaves_ordinary_widths_unchanged`, `test_patch_cap_bounds_the_patch_of_a_wide_item`,
  `test_patch_cap_truncates_a_long_thin_item_and_keeps_its_flux`.

Summary lines (run with `PYTHONNOUSERSITE=1 python -m pytest`; pixi is not installed here):

- `tests/test_scene_gaussians.py tests/test_scene_gaussians_functional.py`: `54 passed in 19.72s` (36 existing, unchanged and
  passing, plus 18 new).
- plus `test_api_manifest`, `test_docs_reference`, `test_docstrings`: `59 passed in 22.82s` together with the two above.
- `test_docs_claims/coverage/integrity/snippets`: `177 passed, 1 skipped`.
- `scripts/run_docs_snippets.py`: `115/115 runnable snippets passed.`
- `scripts/gen_reference.py --check`: `Reference pages are current (38 generated pages).`
- `scripts/api_manifest.py --check`: `API manifest matches docs/api-manifest.json`.
- `ruff check` and `ruff format --check` clean on the two files I edited and the new test.
  `ruff check src/torchsky/scene` reports 14 older issues in `__init__.py`, `operators.py` and `sources.py` (unsorted `__all__`,
  import order); not touched. Not run: `pixi run typecheck` (pyright is not installed), `docs-contract` strict build, `preflight-push`.

## 5. Agreement with the prototype

300 items, 96 x 96 grid, 3 planes, widths log-uniform 0.5 to 30 pixels, pitch 0.7 arcsec, floor 0.28 arcsec:

- Direct and pyramid: image difference 5e-15 of the maximum; gradients (position, log width, angle, log coefficient) 1e-14 to 7e-14 relative.
  The requirement was 1e-6 and 1e-5.
- Widths to 60 px, pyramid, `patch_cap=512`: image 5.6e-8; gradients 7e-8 to 4.8e-6 relative (log width is the worst).
  With the default cap 128 the same problem differs by 1.5e-4 of the maximum from the prototype (thin items truncated, section 3).
- Prototype direct with widths to 60 px truncates at 128 nodes; the new default instead goes to the pyramid, so they differ
  (3e-5 of the maximum). Not a bug in either; stated for the bridge.

## 6. Mode accuracy (for the owner's decision on the default; the default stays `direct`)

150 items, 256 x 256 grid, minor sigma log-uniform 1 to 40 pixels, major 1 to 4 times minor, random angles, float64. Reference: the analytic
Gaussian density per pixel with the 1/12 pixel^2 term.

| render | max error against analytic, fraction of image maximum |
|---|---|
| direct, cap 128 or None | 3.1e-5 |
| pyramid, cap 128, 512 or None | 3.0e-5 |

Direct against pyramid: 1.5e-5 of the maximum at worst, 4e-7 rms (same with the default cap). Per item, from `INTEGRATION_gaussians.md`:
up to about 2e-4 of the item's peak in either mode. So accuracy does not separate the modes; the differences are that pyramid values and
gradients have small jumps where an item changes level (about 2e-4 of its peak) and pyramid keeps dense grids per level in memory.
Cost, from the earlier note (10,000 items, 512 x 512, 4 planes, widths 0.5 to 30 px, forward, one core): direct 22 s, pyramid 0.95 s. I did not
re-measure that row. For widths up to 8 px they are 2.6 s against 0.9 s.

## 7. Timing, one CPU core (`torch.set_num_threads(1)`)

60,000 items, 601 x 601, 12 planes, float64, widths log-uniform 0.5 to 10 px (my assumption; the fit's width distribution was not given),
same items in both. The machine has 2 cores shared with other work, so each figure moves by 10 to 20 percent between runs.
Two runs of the same script:

| | prototype fwd | prototype bwd | functional fwd | functional bwd |
|---|---|---|---|---|
| direct, run 1 | 8.1 s | 28.3 s | 9.1 s | 19.8 s |
| direct, run 2 | 11.2 s | 33.4 s | 10.1 s | 27.3 s |
| pyramid, run 1 | 10.3 s | 31.1 s | 9.7 s | 18.9 s |
| pyramid, run 2 | 7.3 s | 19.7 s | 6.7 s | 9.0 s |

"bwd" is the backward pass alone (it recomputes the patches). The forward pass is the same speed as the prototype within noise (reference
11.6 s and 12.4 s). Images agreed to 1e-15. Not a performance change: no speed claim is made beyond "no slower".

## 8. Approximations still present (all from the earlier module, plus the cap)

Pixel cell integral replaced by a 1/12 pixel^2 Gaussian; patches truncated at 4 widths and renormalised to exact flux; discrete patch window
and bucket and pyramid level choice (gradients ignore the switches); first derivatives only; the new cap (section 3).
Details and sizes are in `INTEGRATION_gaussians.md`, section 7.

## 9. Left undone, and open questions

- Exact (tiled) treatment of long thin items above the cap. Open question: is `patch_cap=128` the right default for the fit, or 512 to
  match the prototype's pyramid? 512 costs up to 1025^2 x K elements for one item.
- The `(major, minor, angle)` form still requires `minor <= major`, as before. The prototype's widths are not ordered; the fit should pass
  the covariance form (the prototype test does) or sort.
- Default mode left as `direct` (owner's decision).
- No CUDA or MPS run. Typecheck and the strict docs build were not run (tools absent here).
