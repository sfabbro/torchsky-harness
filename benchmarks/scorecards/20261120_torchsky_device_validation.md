# torchsky device validation: own test suite on CUDA and MPS (20261120)

Subject: torchsky `601968a797bcf522c9cdf309c298490f778114a5` (sfabbro/torchsky, 0.6.0rc1).
Question: does torchsky's own test suite pass on the devices it claims, and
which of its skips exist only because no such device was present?

The suite is torchsky's `pixi run test` (main suite, `--ignore-glob='tests/test_map_*.py'`)
plus the separately run map tests (`KMP_DUPLICATE_LIB_OK=TRUE pytest -n 4 tests/test_map_*.py`).
Device coverage comes from torchsky's own mechanisms: the `compute_device`
fixture (`cpu`, `cuda`, `mps` params, skipped when absent), tests that check
`torch.cuda.is_available()` / `torch.backends.mps.is_available()`, and
`default_compute_device()`, which is CUDA when present and CPU otherwise.

## Runs

| Device | Host | Run | Environment |
|---|---|---|---|
| CUDA | CANFAR headless, NVIDIA H100 NVL MIG 1g.12gb (cap. 9.0), 8 CPU, 64 GB | `torchsky-tests-cuda-20261120-044438` (session `a1llaabr`, harness `f8d1f6c`) | harness `cuda-bench` (Python 3.13.14, torch 2.14.0+cu130, torchfits 1.3.1) + torchsky `[feature.test]` packages; native extension built from the subject (`avx2+fma`); default device `cuda` |
| MPS | local Apple M2, 8 CPU, 24 GB, macOS 14.8.2 | `torchsky-tests-mps-20261120-045851` | torchsky's own `test` env in a fresh clone at the subject SHA (torch 2.14.1); extension built (`neon`); default device `cpu`, MPS reached through device params |

Artifacts: CUDA JUnit XML, summary, device probe and session log in
`/arc/projects/ots/torchsky-runs/torchsky-tests-cuda-20261120-044438/` (VOS tarball
upload failed; the gist fallback is
<https://gist.github.com/sfabbro/182508644e210a17136628643dd5f5ad>). MPS JUnit XML
and logs in `benchmarks_results/torchsky-tests-mps-20261120-045851/` (local,
gitignored). Machine-readable summaries next to this card:
[`20261120_torchsky_device_validation_cuda.json`](20261120_torchsky_device_validation_cuda.json),
[`20261120_torchsky_device_validation_mps.json`](20261120_torchsky_device_validation_mps.json)
(`scripts/summarize_junit.py`).

## Counts

| Device | Suite | Total | Passed | Failed | Skipped | of which device-absent | Wall time |
|---|---|---:|---:|---:|---:|---:|---|
| CUDA | main | 4305 | 4140 | 6 | 159 | 115 (MPS absent) | 6 min 24 s (`-n 4`) |
| CUDA | map | 1449 | 1424 | 0 | 25 | 23 (MPS absent) | 1 min 2 s (`-n 4`) |
| MPS | main | 4303 | 4164 | 2 | 137 | 87 (CUDA absent) | 33 min 17 s (`-n 4`, shared host) |
| MPS | map | 1420 | 1388 | 0 | 32 | 29 (CUDA absent) | 1 min 55 s (`-n 4`) |

No errors and no xfails in any suite. Totals differ by host because device
params are collected per host. Tests with `cuda` in the id: 154 of 156 passed
on CUDA (the two others are the operator-bundle failures below, not
`cuda`-param tests: 100/102 main, 54/54 map). Tests with `mps` in the id or
class: 149 of 156 passed on MPS; of the 7 others, 6 are float64-capability
skips and 1 is the HEALPix `query_disc` binding skip described below.

## Failures

| Device | File | Test | Error | Class |
|---|---|---|---|---|
| CUDA, MPS | `tests/test_device_to.py` | `test_no_fused_device_dtype_conversions_outside_helper` | 20 call sites use a fused `.to(device, dtype)` / `torch.as_tensor(..., device=, dtype=)` instead of `torchsky.device._to` / `_as_tensor` (MPS float64 copy bug): `map/exposure_fit/_items.py` (8), `_response.py` (7), `_uncertainty.py` (2), `_tiles.py` (1), `catalogs/_exclusive.py:170`, `catalogs/association.py:963` | torchsky bug (device-independent static audit; new `exposure_fit` and association code bypasses the helper) |
| CUDA | `tests/test_implicit_forward.py` | `test_predict_observation_mean_forwards_operator_bundle` | `The values for attribute 'device' do not match: cuda:0 != cpu` | torchsky bug: `render_field_observation` places output with `resolve_device(None, model)`; a parameter-free model has no device, so it falls back to `default_compute_device()` (CUDA) and ignores the CPU observation tensors, against the documented placement rule in `torchsky/device.py` |
| CUDA | `tests/test_implicit_forward.py` | `test_multi_survey_loss_forwards_operator_bundles` | `cuda:0 != cpu` | torchsky bug, same root cause |
| CUDA | `tests/test_field_diagnostics.py` | `test_render_field_quality_helpers_accept_operator_bundles` | `cuda:0 != cpu` | torchsky bug, same root cause |
| CUDA | `tests/test_field_diagnostics.py` | `test_render_field_fit_quality_uses_fit_operator_bundles` | `cuda:0 != cpu` | torchsky bug, same root cause |
| CUDA | `tests/test_hosted_native_validation.py` | `test_cuda_diagnostic_explains_cpu_only_upstream_torch` | expects `installed upstream PyTorch wheel is CPU-only`; on a CUDA torch the diagnostic correctly reports `CUDA_HOME or CUDAToolkit_ROOT is not set; cuda.h ... NVRTC is unavailable` | test assumption (assumes the installed torch is CPU-only; should skip when `torch.version.cuda` is set) |
| MPS | `tests/test_examples_runner.py` | `test_every_example_script_runs` | `examples/test_examples.py` exit 1; rerun on the idle host: 25/32 passed, 6 skipped, `legacy_field_fit.py` `timeout after 120s` (passes in the CUDA run) | test assumption / environment: the fixed 120 s per-example budget is too short for `legacy_field_fit.py` on an M2 CPU; the default device there is CPU, not MPS |

The four operator-bundle failures appear only where `default_compute_device()`
is an accelerator, which is why CPU and MPS hosts never saw them. A fix in
the library (follow observation tensors when the model has none) or in the
tests (give `_ZeroField` a buffer, or compare with `assert_close_on_device`)
both clear them; the library fix matches the documented rule.

## Skips that exist only because a device was absent

Classified from the skip message (`summarize_junit.py` flags absence; the
MPS-host "MPS unavailable or float64 unsupported" and "MPS does not support
float64" skips are capability skips on a present device and are excluded).

- **On the MPS host, CUDA absent: 116** (87 main, 29 map). Reasons: "CUDA
  unavailable" (47), "CUDA not available" (45), "cuda unavailable" (10),
  "needs CUDA" (7), "CUDA is unavailable" (5), "cuda not available" (2).
  **115 of the 116 passed in the CUDA run.** The one exception,
  `test_healpix_primitives::test_query_disc_cuda_vec_matches_cpu`, skips on
  CUDA too ("torch-native HEALPix query_disc C++ binding unavailable").
  Largest groups: `test_wcs_sphere_devices` (10), `test_wcs_extended_projections`
  (7), `test_native_integrated_read` (6), `test_map_field_render_precision` (4).
- **On the CUDA host, MPS absent: 138** (115 main, 23 map). Reasons: "MPS not
  available" (61), "MPS unavailable" (49), "mps unavailable" (10), "MPS
  unavailable or float64 unsupported" (10), "MPS is unavailable" (4), "MPS is
  not available on this machine" (2), "needs MPS" (1), "no MPS" (1).
  **131 of the 138 passed in the MPS run.** Of the 7 others, 6 skip on MPS
  for float64 (`[mps-dtype1]` params of `test_continuous_response`,
  `test_joint_native_likelihood`, `test_pixel_calibration`,
  `test_native_integrated_read` x2, `test_map_field_losses`), and
  `test_query_disc_mps_vec_matches_cpu` hits the missing binding as above.
  Largest groups: `test_device_to` (13), `test_nuisance_spectral_device` (11),
  `test_crossmatch_engine` (10), `test_wcs_sphere_devices` (10).

Together the two hosts turn 246 of the 254 device-absent skips into passes;
none of them failed on the device it needed.

## Skips on both hosts (not device-related)

53 test ids skip on both hosts; 7 of them are the MPS float64 and
`query_disc` cases above. The rest: missing optional comparators (torchsky's
`test` env does not install them): healpy (14 across messages),
dust_extinction (5), mocpy (2), cdshealpix, gwcs, healsparse, hpgeom,
mhealpy, pysiaf, speclite, synphot (1 each); a pyarrow-gated doc snippet (1);
"collection skipped" (2, map); "scalar symbol hidden after final linking"
(1). Native-path skips with the extension built on both hosts, worth a look
because they validate nothing anywhere: "C++ sphere geometry kernels
unavailable" (4, `test_sphere_polygons`), "torch-native HEALPix ... C++
binding(s) unavailable" (5 including the two `query_disc` device tests,
`test_healpix_primitives`), "native crossmatch kernel not built" (1,
`test_crossmatch_engine`), "prototype not available" (3). Arm64 only:
"x86_64 binary instruction inspection" (1).

## Verdict

- CUDA: 5564 of 5754 collected tests pass; 6 failures (4 torchsky placement
  bugs with one root cause, 1 static-audit bug shared with MPS, 1 test
  assumption). Not green.
- MPS: 5552 of 5723 pass; 2 failures (the shared static-audit bug, 1
  example-timeout test assumption). Not green.
- Device-absent skips are now covered: 115/116 CUDA-only and 131/138
  MPS-only skips pass on the matching device. The remaining device gap is the
  HEALPix `query_disc` binding check that skips on every host.
- Not covered: multi-GPU, full (non-MIG) GPUs, `TORCHSKY_DEVICE=mps` as the
  library default on Mac, and the optional comparator tests listed above.

Rerun: `TORCHSKY_COMMIT=<sha> TORCHSKY_SUBJECT_REPO=sfabbro/torchsky
TORCHSKY_HARNESS_REPO=sfabbro/torchsky-harness TORCHSKY_CANFAR_CPU=8
TORCHSKY_CANFAR_MEMORY=64 bash scripts/launch_canfar_wcs_sphere.sh
torchsky-tests-cuda`.
