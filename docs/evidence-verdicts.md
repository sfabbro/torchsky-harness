# Execution records and evidence verdicts

Every `torchsky-harness run` writes `<run-dir>/execution.json`. The record is the
only place the harness states what a run did and what it is allowed to claim.
`stdout.log`, `stderr.log`, and any `--json-out` report sit beside it.

## Record fields

| Field | Meaning |
|---|---|
| `schema_version` | record layout version (currently `1`) |
| `evaluation`, `command`, `working_directory` | what ran, exactly |
| `subject` | installed subject distribution, version, import path, package-content SHA-256, source identity, Python, platform. The subject is never a checkout on `sys.path`. `subject.source.kind` is `archive` for a wheel install (with the wheel SHA-256, and no VCS revision), `source_tree` for an editable/dir install (with the source repo revision and dirty flag), or `registry` when there is no direct URL. A wheel install never inherits the revision of the repository that happens to contain `site-packages`. |
| `harness` | harness version and source identity |
| `inputs` | SHA-256 of every command argument that is an existing file at launch |
| `execution` | `running`, `completed`, `failed`, or `interrupted` |
| `returncode` | child process status once finished |
| `logs`, `evidence_report` | SHA-256 of the logs; absolute path of the `--json-out` report |
| `scientific_validity` | `not_assessed`, `established`, or `not_established` |
| `comparison_eligibility` | `not_assessed`, `suite_owned`, or `baseline_compared` |
| `verdict_reasons` | why validity is not `established` (empty when established) |
| `comparison_notes` | baseline supplied but not enforced, or similar |
| `claim` | the declared bounded claim this evaluation may certify (absent when none) |
| `metrics`, `device` | `suite_owned`: rows carry their own metrics/device declaration |

## Verdict vocabulary

A run never guesses. Validity is derived only from declarations in
`src/torchsky_harness/evidence.py` plus observable facts:

* `not_assessed` — the evaluation has no declared claim (most evaluations), or
  the run did not finish.
* `established` — the evaluation has a declared claim, the child exited `0`, and
  every declared required row is present in the evidence report with
  `status=OK`; when several rows share a required identity (for example one row
  per comparator library), every one of them must pass. Summary-report
  evaluations declare a `summary_failures_key` instead of rows and are
  established when that failure list is empty.
* `not_established` — a declared claim exists but one of the above failed; each
  failure is listed in `verdict_reasons`.

`comparison_eligibility` is independent of scientific validity:

* `suite_owned` — the evaluation reports its own metrics; no baseline coverage
  was enforced.
* `baseline_compared` — a `--baseline-json` was supplied together with
  `--require-complete-baseline`, so missing rows, missing metrics, and
  regressions are failures.

## Claim scopes

| Scope | What a pass certifies |
|---|---|
| `synthetic_truth` | the declared synthetic/analytic cases only; says nothing about real detector data, real sky, or other regimes |
| `reference_comparison` | parity against the named independent comparators on the declared cases |
| `real_data` | real observations with an explicit, checked reference |

Each declared claim also lists `limitations`. Those limitations are part of the
claim: for example the HEALPix multi-band gate does not claim flux-conserving
reprojection, harmonic resampling, variance propagation, or comparator
throughput.

## Declared release gates

| Evaluation | Rows | Task |
|---|---|---|
| `bench_map_cube` | `cube_constant_bands` | `bench-map-cube-gate` |
| `bench_wcs` | 30 projection rows (`TAN` … `TPV`) | `bench-wcs-release-gate` |
| `bench_healpix` | `ang2pix_ring`, `ang2pix_nested`, `pix2ang_ring`, `pix2ang_nested`, `ring2nest`, `nest2ring` | `bench-healpix-release-gate` |
| `bench_healpix_advanced` | `neighbors_ring`, `neighbors_nested`, `interp_weights_ring`, `interp_weights_nested`, `interp_val_ring`, `interp_val_nested` | `bench-healpix-advanced-release-gate` |
| `bench_sphere_matrix` | summary `failed_thresholds` must be empty | `bench-sphere-matrix-release-gate` |
| `bench_spin_matrix` | summary `failures` must be empty | `bench-spin-matrix-release-gate` |
| `bench_catalog_compat` | `synthetic_clustered`, `real_astropy_votable_catalogs`, `real_astropy_votable_fits_roundtrip` | `bench-catalog-release-gate` |
| `bench_catalog_match_exact` | poles, RA wrap, wrap at the pole, duplicates, equidistant ties, radius ladder 1 mas to 1 deg, per-row radii over 5 dex, Plummer cluster, two-sided sum/quadrature/max, mpmath boundary | `bench-catalog-match-exact-release-gate` |
| `bench_map_ifu` | `ifu_matched_sparse_spectrum` | `bench-map-ifu-gate` |
| `bench_map_ifu_spectral_axis` | `ifu_nonuniform_axis_line_localization`, `ifu_nonuniform_axis_constant_spectrum`, `ifu_regular_axis_spectrum_interpolation`, `ifu_nonuniform_axis_interpolation_envelope` | `bench-map-ifu-spectral-axis-gate` |
| `bench_map_healpix_multiband` | `healpix_multiband_flat_multiscale`, `healpix_multiband_sed_registration`, `healpix_multiband_plane_photometry`, `healpix_multiband_roundtrip_cube` | `bench-map-healpix-multiband-gate` |
| `bench_native_response` | `native_gaussian_star` | `bench-native-response-gate` |
| `bench_native_likelihood` | `native_gaussian_likelihood` | `bench-native-likelihood-gate` |
| `bench_native_nuisance` | `native_nuisance_precision` | `bench-native-nuisance-gate` |

Adding an evaluation to `CLAIMS` is a deliberate act: it widens what a passing
run may claim. Do not add one without a committed baseline, a completeness
check, and a limitations list.

## Reading a bundle

For each gate run keep: `execution.json` (including `subject.source.revision`
and `package_content_sha256`), the `--json-out` report, `stdout.log`, and the
task command. A release bundle is the set of runs whose `scientific_validity` is
`established` for a declared claim and whose subject identity matches the release
artifact. Anything else is supporting evidence, not a pass.

## Evidence bundle

`pixi run release-readiness` collects the run records and applies the mandatory
paired performance verdict. `pixi run evidence-bundle` is its bundle step and
also fails closed when the performance artifacts are absent or invalid:

| Field | Meaning |
|---|---|
| `release_ready` | `scientific_release_ready` is true and the finite paired performance contract is ready |
| `scientific_release_ready` | every declared claim has a completed and `established` run, all declared-claim runs share one subject identity, and no run directory is unreadable. Undeclared supporting runs may record a different subject without blocking readiness; `claim_subject_identities` is the gated set |
| `performance` | fail-closed paired candidate/baseline verdict, per-row ratios, fingerprints, and blockers |
| `missing_claims` | declared claims without an `established` run |
| `extra_established` | established runs for evaluations that declare no claim (supporting evidence) |
| `subject_identities`, `subject_consistent` | package-content SHA-256, source kind, source digest, and revision of every selected run |
| `harness_revisions`, `claim_harness_revisions` | harness revisions observed in the bundle; the `claim_`-prefixed field is the declared-claim subset that the release evidence pins |
| `incomplete_runs`, `incomplete_claims`, `unreadable_runs` | runs that cannot certify anything; `incomplete_claims` is the declared subset that blocks readiness |
| `superseded_runs` | older runs that were not selected |
| `entries` | per evaluation: verdict, claim, subject, harness, `execution.json` and report SHA-256, log SHA-256 |

Selection is newest `established` run per evaluation, falling back to the newest
run when none is established. The bundle hashes the files it names, so a bundle
plus the run directories is self-checking; it is not a signature.
