# torchsky Benchmarks

> FITS image/table I/O benchmarks live in
> [`torchfits`](https://github.com/astroai/torchfits). Torchsky benchmarks begin
> with decoded tensors or Arrow batches.

torchsky is benchmarked for both **correctness** and **throughput** against upstream libraries across WCS, sphere, catalog, map, and field workloads. This page documents the methodology, comparator environments, real-data benchmark setup, and how to reproduce the results.

## Optimization evidence program

The harness-first optimization program is defined in
[optimization-evidence-plan.md](optimization-evidence-plan.md). Its quick local
entry point is:

```bash
export TORCHSKY_EXPECTED_COMMIT=<full-subject-sha>
pixi run bench-optimization-evidence-quick
```

Focused tasks cover prepared map targets, PCG synchronization, calibration
graph scaling, MOC/sparse random sampling, and triplane batching. They emit raw
JSON under `benchmark_results/` and durable execution records under `runs/`
when invoked through the harness CLI. The shared
`torchsky-harness.optimization-evidence.v1` contract records installed-subject
identity, runtime/device information, repeated timing distributions, checksums,
and explicit limitations. Baseline and candidate runs must use the same
workload and hardware; a microbenchmark improvement is descriptive until its
representative end-to-end row also passes.
The aggregate task refuses dirty subjects and verifies
`TORCHSKY_EXPECTED_COMMIT` (or the CANFAR `TORCHSKY_COMMIT`) before execution.

Record installed comparator versions with `pixi run bench-comparator-versions-all` (writes under `benchmarks_results/`).

### Scorecard history (committed)

Full local runs write to gitignored `benchmarks_results/<run_id>/`. To keep a
compact, diff-friendly history in the repo, archive a snapshot after a
representative WCS/sphere sweep:

```bash
pixi run bench-archive-scorecard -- \
  --label main_domains \
  --runs fits=<fits_run_id>,fitstable=<fitstable_run_id>,wcs=<wcs_run_id>,sphere=<sphere_run_id>
```

Committed snapshots live under [`benchmarks/scorecards/`](../benchmarks/scorecards/)
(`index.md` plus dated `*.json` / `*.md`). Older narrative notes remain in
[`docs/archive/benchmark-history.md`](archive/benchmark-history.md).

The external packages torchsky replaces, compares against, or merely coexists with are tracked in [replacement-targets.md](replacement-targets.md).

For adversarial contract and audit methodology (native boundary, invalid
domains, tensor/device contracts, concurrency, packaging), see
[adversarial-evaluation-playbook.md](adversarial-evaluation-playbook.md).

Artifact roots used in this document:

- `benchmarks_results/` is the default root for generic benchmark-suite outputs
  and comparator-version captures.
- `benchmark_results/` is used by several Pixi tasks for focused JSON/CSV
  outputs.
- `data/benchmark_results/` is for heavyweight local real-data runs, especially
  SMACS/UPdec artifacts that should not be committed.

## Comparator Environments

Torchsky benchmarks are not Astropy- or healpy-only. Local and CI runs combine
several Pixi environments because upstream wheels and ABI pins differ.

| Environment | Purpose | Typical comparators |
|---|---|---|
| `bench-all` | Main `bench_all.py` WCS/sphere domains | Astropy (`astropy.wcs`), torch, optional legacy bridge rows |
| `bench-legacy` | Legacy WCS libraries | PyAST (`starlink.Ast`) for legacy WCS comparison |
| `sphere-bench` | HEALPix and spherical geometry ecosystem | healpy, hpgeom, astropy-healpix, healsparse, spherical-geometry, mocpy, optional healpix/mhealpy/skyproj |
| `sht-bench` | Optional HEALPix SHT peer | same as `sphere-bench`, plus **s2fft** via `pixi run sht-bench-bootstrap` (JAX is not in `sphere-bench`; healpy stays the science oracle) |
| `map-compat` | Optical/IR stacker compatibility | reproject, dfreproject, drizzle, Montage, SWarp, casutools |

Representative commands:

```bash
pixi run -e bench-all -- python benchmarks/bench_all.py --profile user --wcs-only
pixi run sphere-bench-geometry-ecosystem
pixi run bench-map-compat-external
pixi run bench-comparator-versions-all
pixi run wcs-sphere-acceptance
```

Linux CPU/CUDA confirmation uses `scripts/launch_canfar_wcs_sphere.sh`. Modes:

- `cpu`, `cuda`, `all` — full WCS/sphere acceptance on CPU and/or CUDA
- `suite` / `many` — parallel `spectral`, `sphere`, `sparse`, `spin`, `geometry`, `cuhpx`
- `sht` — `spectral` + `sparse` + `spin` only
- world-layer jobs (also `world` to submit them together): `wcs-paper2`, `wcs-tpv`, `wcs-spectral`, `wcs-tabular`, `sht-profile`, `moc`, `catalog`, `observed`, `geom-cuda`, `world-cuda`. `catalog` also runs `bench_catalog_compat.py` (Astropy SkyCoord peer). `wcs-paper2` writes fused=0/1 CSC A/B JSON.
- `world-cuda` puts WCS/catalog/observed/sparse/polygon tensors on the GPU (`--device cuda`). `geom-cuda` remains the polygon-only CUDA job. `gpu-validate` is the current-subject device check: it installs the subject into the harness `cuda-bench` environment, runs the device-named package tests, and runs the harness WCS comparison on GPU (including the reference-offset cases) with a junit and comparison JSON artifact. Use `gpu-validate` for release evidence; the older modes still call benchmarks from the subject checkout, which no longer owns them.
- `torchsky-tests-cuda` runs torchsky's own test suite at the subject commit on one GPU: the main suite as torchsky's `pixi run test` (`--ignore-glob='tests/test_map_*.py'`) and then `KMP_DUPLICATE_LIB_OK=TRUE pytest -n 4 tests/test_map_*.py`, in the harness `cuda-bench` environment plus torchsky's `[feature.test]` packages. On a CUDA host the library default device is CUDA and the `compute_device` fixture adds `cuda` params, so this covers what local CPU/MPS runs skip. Artifacts: `torchsky_tests_{main,map}_cuda.xml` (JUnit, per-test outcomes), `torchsky_tests_cuda_summary.json` (counts, failures, skip reasons grouped, device-absent skips flagged; `scripts/summarize_junit.py`) and `device_probe.json`. Set `TORCHSKY_CANFAR_MEMORY=64`.

That submits AstroAI headless sessions which run `scripts/run_canfar_wcs_sphere.sh` at an exact `TORCHSKY_COMMIT`, write `/arc/projects/ots/torchsky-runs/<run_id>/`, and copy the same tree to `vos:sfabbro/torchsky-runs/<run_id>/`. Pull a finished run with `bash scripts/fetch_canfar_wcs_sphere_run.sh <run_id>` (`vcp` / `pip install vos`). Do not reuse the ADASS/SMACS runner for this domain. Local Mac smoke stays `--quick` via `pixi run wcs-sphere-acceptance`; the full oracle ingest is `pixi run wcs-sphere-acceptance-full` or the CANFAR `cpu` job. CANFAR `cpu`/`spin`/`geometry` jobs pass `--no-throughput-gate` so Mac-tuned healpy ratio thresholds do not fail the job; rel L2 and index/angle mismatch gates still apply. Mac pixi tasks (`sphere-bench-geometry-gate`, `sphere-upstream-spin-matrix-gate`) keep the ratio gates. The cuHPX scalar SHT spike is a separate CUDA-only job: `scripts/run_canfar_cuhpx_sht_spike.sh` or the `suite` `cuhpx` session. Current Linux ingest: [`benchmarks/scorecards/20260817_world_layer.md`](../benchmarks/scorecards/20260817_world_layer.md) (main WCS 87/90 CPU, 90/90 CUDA large-N) plus [`benchmarks/scorecards/20260821_world_layer.md`](../benchmarks/scorecards/20260821_world_layer.md) (Paper II / TPV / spectral / tabular / observed / moc / `world-cuda` on SHA `775762c2`). Claim-or-close ingest: [`benchmarks/scorecards/20260821_wcs_sphere_claim_close.md`](../benchmarks/scorecards/20260821_wcs_sphere_claim_close.md) on SHA `d8d706f6` stamp `20260821-1744*` plus relaunch SHA `7564070c` stamp `20260821-1852*` (`world-cuda` pytest 18/18; CUDA CSC host-copy 8.35 mas **not** claimed; `catalog_compat` vs SkyCoord 0.116× **not** claimed; N=1e6/N=1e7 still running). World-layer throughput is **not claimed** as a suite win; see that card.

Latest release GPU evidence on torchsky `1611fad2` / harness `0d30d635`:

- `gpu-validate` run `gpu-validate-20261020-153814` (session `m5x5s624`): the
  17-case GPU WCS comparison is clean (`bad=[]`, max angular **8.23e-3 asec**,
  pole-excluded round-trip **7.11e-7 asec**), but the device-named package
  tests are **6 passed / 1 failed** on
  `test_wcs_extended_projections.py::test_paper2_cuda_matches_cpu[CSC-0.3-0.0]`
  (CSC CUDA/CPU samples differ by up to 8.3e-7 against a 1e-10 tolerance). This
  is known CSC seam conditioning, not a mapping regression, but the test and
  implementation must be reconciled before the device check is claimed green.
- `cuda` run `wcs-sphere-cuda-20261020-153833` (session `z2ilmrs7`, rc=0): WCS
  large-N CUDA acceptance had **0 deficits**; the sphere CUDA section recorded
  8 comparator large-N deficits and is not claimed as a throughput win. VOS
  upload is unavailable on that image (no `vcp`); the gist session log is the
  retained artifact.

For pip-only parity benches against the HEALPix / geometry comparator stack,
install `pip install -e ".[bench-all]"` after installing a compatible PyTorch.

Benchmark policy: on comparable rows in emitted CSV/JSON artifacts, torchsky
should rank first on throughput while staying within parity tolerances. Rows
where a comparator is skipped or has different semantics are not counted as
losses; expand coverage carefully so skips shrink over time.

## Performance vs comparators

`benchmarks/bench_comparator_perf.py` (Pixi: `bench-comparator-perf-gate`; a campaign `performance` gate, rerun with
`--device cuda` in `MODE=gpu`) records, for each torchsky primitive, its time and throughput next to every comparator in
the map, in one normalized JSON (`benchmark_results/comparator_perf.json`, `comparator_perf_cuda.json`). Nothing here is a
wall-time gate: shared nodes are noisy, the file is the evidence. A comparator that is missing is a row marked
`unavailable` with the reason.

| area | operations | sizes | comparators |
|---|---|---|---|
| `wcs` | pixel to world, world to pixel (TAN, TAN-SIP) | 1e5, 1e6 | astropy.wcs, PyAST |
| `healpix` | ang2pix, pix2ang, neighbours (nside 1024) | 1e5, 1e6 | healpy, hpgeom, astropy-healpix |
| `catalogs` | nearest-match crossmatch | 1e5, 1e6 | astropy SkyCoord, STILTS `tmatch2`, `tskymatch2` |
| `moc` | from_circle, union, intersection, contains (order 10) | single call; 1e5, 1e6 points | mocpy |
| `sparse` | get_values, ud_grade | 1e5, 1e6; 20k covered pixels | healsparse |
| `sht` | map2alm, alm2map (nside 256/512, lmax 512) | 786k, 3.1M pixels | healpy, s2fft |
| `map` | bilinear and exact reprojection | 2048^2, 4096^2 | reproject, dfreproject, SWarp, Montage |

Conventions: speedup is `comparator_s / torchsky_s` (above 1: torchsky faster); throughput is `size / torchsky_s`.
PyAST runs in its own interpreter (`scripts/oracle/wcs_oracle.py`) and are timed inside it; STILTS and the
SWarp/Montage binaries report wall time including process start and file I/O (noted per row); healpy and s2fft SHTs run
in a child process. The parity gates `test_stilts_parity.py` add per-mode STILTS timings on the real Gaia DR3 x 2MASS cone
through `tests/_perf_record.py` (`comparator_perf.jsonl`). `scripts/canfar/summarize.py` merges both into the
"Comparator performance" table of every campaign `summary.md` (and `summary.json`, key `comparator_performance`).
Known limit seen while writing it: `map2alm` at `nside >= 512` with `lmax > 512` takes torchsky's dense-basis fallback and
fails (`basis-based fallback requires ... GB`), so the SHT rows use lmax 512.

Local use: `pixi run bench-comparator-perf-quick`; the oracles with `pixi run test-comparator-oracles`
(env `oracle`); STILTS with `pixi run fetch-stilts fetch-real-catalogs test-stilts-parity` (env `stilts`, Java).

## CI: map compat gate

Reusable map truth metrics and benchmark artifact/report helpers are exposed by
[`torchsky_harness.metrics`](map-metrics.md). Evaluation modules import that
harness API directly; the torchsky runtime package has no fallback to harness
code.

Scientific map gates are **not** run on every push. The push/PR workflow
`.github/workflows/ci.yml` is `Harness infrastructure` and runs preflight plus
infrastructure contracts only. The `Map mosaic bench + compat gate` workflow in
`.github/workflows/map_compat_weekly.yml` is manual (`workflow_dispatch`, no
scheduled trigger despite the file name):
it installs the subject and harness, runs the map/compat test set, runs
`benchmarks/bench_map_compat.py` on `synthetic_shifted_wcs` with
`benchmarks/baselines/map_compat_ci_gate.json` enforcing status and comparator
diff ceilings for the bilinear `reproject_interp` row, then builds the pinned
official SWarp and Montage source revisions and requires Drizzle, SWarp, and
Montage rows to satisfy the committed absolute acceptance contracts. A skipped
official comparator is a failed job, not a parity pass. Run the same gates
locally with the pixi tasks in this document; `pixi run ci-local` is the
optional ruff + compileall + full pytest pass.

**Optional map-compat wall-time regression:** after recording representative `time_ms` rows on a pinned runner, pass **`--time-baseline-json`** (format in **`benchmarks/baselines/map_compat_time_gate_example.json`**) and **`--time-regression-tolerance`** (default `0.35`) to fail if OK rows slow down beyond the baseline. This is intentionally **off** in default CI because wall times vary by hardware.

## Real-data map parity

`benchmarks/bench_map_real_cutout.py` reprojects a real 256x256 cutout of the
CFHT MegaPipe g-band mosaic (fixture provenance in
`benchmarks/fixtures/real_map/PROVENANCE.md`) onto a rotated/rescaled TAN grid
with `torchsky.map.resample`, and compares it with astropy `reproject` on the
interior of the shared footprint (the coverage-edge convention is recorded but
not gated).

## Real-data sphere parity

`benchmarks/bench_sphere_real_map.py` compares torchsky and healpy on a compact
real-sky map derived from the local 2 GB CFHT MegaPipe g-band mosaic
(`scripts/derive_real_sphere_fixture.py`; provenance in
`benchmarks/fixtures/real_sphere/PROVENANCE.md`). The fixture stores only
per-pixel means and real sky positions. It gates pixel lookup, interpolation,
`ud_grade`, `map2alm`/`alm2map`/`alm2cl`, and `smoothmap`.

## Real-data WCS parity

`benchmarks/bench_wcs_real_headers.py` compares torchsky and astropy on eleven
real instrument headers and small real images vendored from the Astropy WCS test
suite (BSD-3; see `benchmarks/fixtures/real_wcs/PROVENANCE.md` for sources and
descriptions): INT WFC ZPN, Spitzer IRAC SIP, PTF TPV, HST WFPC2 SIP, validation
TPV/SIP headers, and a real HST `(RA, Dec, TIME-TAB)` product. It gates forward
angular parity and inverse pixel parity per fixture, records `SKIPPED` rows with
reasons for lookup-distortion products whose parity is still open, and found
two real-data `-TAB` defects that the synthetic cases never exercised.

## Real-Data Benchmark Setup

Large downloaded inputs under `data/`, generated benchmark artifacts under
`data/benchmark_results/`, and temporary staging files under `data/_tmp/` are
local products and are intentionally not committed.

### SMACS J0723 Public Data

Download public references plus public JWST inputs:

```bash
cd "$(git rev-parse --show-toplevel)"
pixi run -e test -- python scripts/download_smacs0723_multiband_public.py \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --module b \
  --updec-iterations 0 \
  --jwst-output-dir data/jwst_smacs0723 \
  --updec-output-dir data/updec_webb
```

Fill one missing band when needed:

```bash
cd "$(git rev-parse --show-toplevel)"
pixi run -e test -- python scripts/download_jwst_smacs0723_mast.py \
  --band F277W \
  --module b \
  --subgroup CAL \
  --output-dir data/jwst_smacs0723
```

Run the full non-INR multiband stacker sweep:

```bash
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=src:benchmarks pixi run -e map-compat -- python benchmarks/bench_map_smacs0723_stackers_band_sweep.py \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --module b \
  --input-product cal \
  --crop-size 256 \
  --crop-mode best_overlap \
  --skip-missing \
  --json-out data/benchmark_results/smacs0723_stackers_band_sweep_full_public_256.json \
  --markdown-out data/benchmark_results/smacs0723_stackers_band_sweep_full_public_256.md
```

`best_overlap` is the recommended crop mode across bands. Use `--crop-size 512`
for a more detailed pass when memory and runtime allow.

### UPdec-Webb References

```bash
cd "$(git rev-parse --show-toplevel)"
pixi run -e test -- python scripts/download_updec_webb.py \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --iterations 0,10 \
  --output-dir data/updec_webb
```

### Field-Fit Smoke Runs

Single-band real field fit:

```bash
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=src:benchmarks pixi run -e test -- python benchmarks/bench_map_smacs0723_field_fit.py \
  --band-name F277W \
  --input-product cal \
  --filename-contains nrcblong \
  --crop-size 128 \
  --crop-mode best_overlap \
  --steps 4
```

Multiband joint field fit:

```bash
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=src:benchmarks pixi run -e test -- python benchmarks/bench_map_smacs0723_field_fit_joint.py \
  --bands F090W,F277W,F356W,F444W \
  --exposure-dir "${TORCHSKY_DATA}/jwst_smacs0723" \
  --input-product auto \
  --crop-size 32 \
  --crop-strategy best_overlap \
  --steps 4 \
  --models coadd_bilinear,spectral_power_foam
```

`auto` selects an untweaked CRF per exposure and falls back to its CAL only
when that CRF is absent. To run the immutable tweakreg comparator, pass the
`aligned` directory and `--input-product aligned` explicitly; never use that
command as the native astrometric training input.

Real-data field-fit debug runs can be expensive. Treat their outputs as local
artifacts unless a baseline is deliberately reviewed and committed.

## Main Scripts

| Script | Description |
|--------|-------------|
| `bench_all.py` | WCS and sphere benchmark orchestrator. |
| `bench_wcs_suite.py` | Authoritative WCS projection sweep runner with forward+inverse and `N={1k..10M}` tiers. |
| `bench_sphere_suite.py` | Authoritative sphere-domain runner aggregating geometry/advanced/sparse/spectral/polygon/core benches. |
| `bench_operator_chunking.py` | Lightweight synthetic multi-exposure executor check. It compares eager and matrix-free chunked `A^T N^-1 A` application in isolated subprocesses, reporting deterministic checksums, throughput, peak RSS, and the analytic prediction working-set bound without running the full benchmark suite. |
| `bench_map_image_plane.py` | Image-plane mapmaking benchmark covering synthetic identical/shifted WCS coadds, a synthetic low-overlap TPV MEF case, a grouped synthetic multi-band compatibility cube, a header-derived planned spectral-cube row on a shared target grid, plus both single-HDU and generated MEF real Astropy SIP FITS cases in identical-WCS and shifted-WCS variants, including overlap and footprint diagnostics. |
| `bench_updec_webb_real.py` | Real-data smoke benchmark on public UPdec-Webb JWST SMACS J0723 products. It compares the published iteration-0 Drizzle product against the iteration-10 UPDC product on the same WCS grid, reports product-format details plus same-grid delta and sharpness ratios, and can auto-download selected public bands from NADC. |
| `bench_map_smacs0723_real.py` | Real-exposure JWST SMACS J0723 smoke benchmark. It coadds local public NIRCam exposures onto a cropped target grid derived from the published F277W Drizzle product, preferring local `*_crf.fits` inputs when present and falling back to `*_cal.fits`, then reports support-domain agreement against that real reference patch. The current real JWST path also accepts `drizzle_point` / `drizzle_square` plus explicit `--drizzle-pixfrac` so the in-tree map code can be compared against the published drizzle contract directly. |
| `bench_map_smacs0723_reproduce.py` | Real-data reproduction sweep against the published SMACS J0723 F277W Drizzle patch. It runs several `torchsky.map` resampling/value-semantics variants on the same local calibrated exposures, prefers local `*_crf.fits` inputs when present, and can optionally add true external `drizzle` rows on the same crop to separate accumulation-contract bugs from benchmark mismatches. For the public JWST CRF inputs, the drizzle rows are treated as `surface_brightness` products rather than flux-per-pixel products. |
| `bench_map_smacs0723_ladder.py` | Real-data exposure-ladder benchmark against the published SMACS J0723 F277W Drizzle patch. It runs one chosen `torchsky.map` configuration on prefixes of a matched real-exposure set, preferring local `*_crf.fits` when present, so overlap, coverage, and support error can be tracked as more real visits are added. |
| `bench_map_smacs0723_field_fit.py` | Real-data single-band field-fit smoke on JWST SMACS J0723 exposures. It compares `coadd_bilinear` against selected INR field models on a crop of the public drizzle target, now using band-aware field observations with finite bandpass bounds plus weighted-RMS normalization instead of a delta-wavelength proxy, and it also supports an `exact_wavelength` observation mode as a 2D no-integration control. |
| `bench_map_smacs0723_field_fit_joint.py` | Real-data joint multiband field-fit benchmark on JWST SMACS J0723 products. It fits one latent field across several public bands simultaneously, then renders each band back onto its own published reference crop so per-band coadds can be compared against a shared multiband INR. |
| `bench_map_smacs0723_field_fit_band_sweep.py` | Real-data per-band field-fit sweep on JWST SMACS J0723 module-B products. It iterates across multiple public UPdec-Webb reference bands, auto-discovers matching local public exposures by filter and module, and reports per-band `coadd_bilinear` vs INR field-fit results on a common crop. It supports both finite-bandpass fitting and an `exact_wavelength` 2D control mode. |
| `bench_map_smacs0723_stackers.py` | Real-data stacker comparison benchmark against the published SMACS J0723 Drizzle patch. It runs legacy stackers (`drizzle`, `reproject`, `dfreproject`, `Montage`, `SWarp`) plus matching `torchsky` kernels and selected INR rows on the same paper-faithful crop, with backend-appropriate target projections and band-aware INR rendering. |
| `bench_map_smacs0723_stackers_band_sweep.py` | Real-data multiband non-INR stacker sweep on public SMACS J0723 products. It iterates over multiple public UPdec-Webb reference bands, auto-discovers matching local public JWST exposures by filter and module, then runs the full legacy/`torchsky` stacker matrix sequentially per band without INR rows so the baseline stack pipeline can be validated before multiband field fitting. |
| `bench_map_compat.py` | Mapmaking compatibility benchmark comparing torchsky image-plane coadds against an explicit external-tool matrix. `reproject_interp`, `dfreproject`, low-level `drizzle`, source-built `Montage`, source-built `SWarp`, and local-source `casutools imstack` are wired where their target/grid semantics are comparable; `drizzlepac` and unsupported or backend-failing target/grid combinations appear as explicit `SKIPPED` rows with concrete reasons. |
| `bench_map_ground_truth.py` | Synthetic mapmaking correctness benchmark against known sky truth. It evaluates final requested map products on a shared `MapProductSpec`, emits machine-readable JSON plus a concise Markdown summary table, and can optionally enforce representative baseline JSON gates. |
| `diagnose_map_large_union_reprojection.py` | Focused diagnostic runner for the large-union truth cases. It emits per-observation reprojection rows for `torchsky` and `dfreproject`, pairwise diffs before stacking, and final stack rows on the same target grid so TPV or large-rectangle gaps can be localized to specific inputs instead of inferred from the final coadd alone. |
| `diagnose_map_large_union_accumulation.py` | Follow-up diagnostic runner for the large-union TAN weighted cases. It compares grouped vs per-observation accumulation for `torchsky` and `SWarp`, bins residuals by overlap count, and reports numerator/denominator hotspots so overlap-region edge-support mismatches can be localized to specific pixels and observations. |
| `diagnose_map_projection_tail.py` | Focused wide-projection stack diagnostic. It compares stack methods on one projection case, bins residuals by overlap count, and reports the worst improved pixel with contributing single-observation values so small tail differences can be separated from broad reprojection failures. |
| `bench_map_boundary_support.py` | Focused edge-support sweep for bilinear TAN weighted coadds on the large-union truth cases. It evaluates explicit source-boundary margins against ground-truth reconstruction metrics and `SWarp` agreement so edge-policy ideas can be benchmarked before touching the production resampler. |
| `bench_map_multifield_truth.py` | Synthetic multiwavelength latent-field cube benchmark. It simulates wavelength-dependent continuous sky fields with mixed bandpasses, PSFs, WCS distortions, throughput, and overlap, using both analytic fields and an initial frozen implicit-neural-field variant, then scores `coadd_image_plane_cube(...)` against latent cube truth and emits per-case torchsky kernel recommendations. |
| `bench_map_field_fit.py` | Experimental continuous-field fitting benchmark. It renders fitted field models and direct stack baselines onto the same requested final product, emits JSON plus Markdown summaries, and can optionally enforce representative baseline JSON gates. |
| `bench_strainer_phase2.py` | Small JSON harness for Strainer-style field models: baseline vs trunk pretrain + head reset vs pretrain with frozen trunk, reporting initial / early-step / final training losses on a synthetic TAN observation. Requested `--early-steps` above `--fit-steps` are omitted (see JSON `early_steps_skipped`) rather than emitted as NaNs. |
| `bench_map_smacs0723_strainer_transfer.py` | Optional real-data Strainer transfer smoke on SMACS J0723 (same crop stack as `bench_map_smacs0723_field_fit.py`): random init vs multi-noise pretrain + `reset_decoder_head()`, with training losses and support RMSE vs the drizzle reference. Exits 0 with `status: skipped` when `data/` inputs are absent. |
| `bench_wcs.py` | WCS transform throughput checks, including small-batch center-projection `default vs python-only` fields on CPU. |
| `bench_wcs_small_n_paths.py` | Small-`N` dispatch matrix: Astropy `all_pix2world` / `all_world2pix` vs torchsky two-arg vs stacked `(N,2)` vs stacked after `compile()` for **forward** (fused pixels→rel→world + Torch pole rotation) and **inverse** (fused world→pixels + Torch inverse pole rotation), FITS `origin` 0/1; median ms plus max |ΔRA|,|ΔDec| and max |Δx|,|Δy| vs Astropy on the stacked paths. |
| `bench_wcs_tpv_inverse.py` | Dedicated large-`N` TPV inverse benchmark comparing torchsky fused TPV `world_to_pixel`, torchsky tensor-core TPV inverse, and Astropy on affine-only and non-linear TPV headers. |
| `bench_healpix.py` | HEALPix CPU/CUDA parity + throughput + quality gates (`ang2pix`, `pix2ang`, `ring/nest`). |
| `bench_healpix_advanced.py` | Advanced HEALPix parity + throughput checks (`get_all_neighbours`, `get_interp_weights`, `get_interp_val`). Timed loops measure torchsky tensor work only; NumPy export for diffs is outside timing so torchsky is not charged extra copy cost vs healpy. |
| `bench_sphere_geometry.py` | Cross-library spherical geometry benchmark (torchsky vs HEALPix ecosystem releases). |
| `bench_sphere_core.py` | Sphere-core primitives benchmark (multi-band sampling, pairwise distances, ellipse query). `--moc-only` times torchsky vs mocpy cone / contains / union. |
| `bench_sphere_polygons.py` | Non-convex spherical polygon benchmark (contains/intersects/query/area with optional spherical-geometry parity). `--device cpu|cuda|mps`; C++ vs Python rows are CPU-only. |
| `bench_observed_place.py` | Pinned-EOP ICRS→GCRS/CIRS/TETE/AltAz timing; CIRS/TETE vs ERFA `c2i06a`/`pnm06a`. |
| `bench_sphere_geom_exact_contains.py` | Bulk ``(lon, lat)`` containment: ``ExactSphericalRegion.contains`` (``spherical_geometry`` backend) vs the same vectorized ``contains_lonlat``/``contains_radec`` numpy path used as the upstream reference, plus throughput vs ``spherical_polygon_contains``. Explicit **SKIPPED** rows document non-comparable APIs for **hpgeom** / **healpy** (HEALPix-centric; use `bench_sphere_geometry.py` for those). Wired into `bench_sphere_suite.py` as suite `sphere_geom_exact_contains`. **Regression gate:** `--baseline-json benchmarks/baselines/sphere_geom_exact_contains_gate.json` enforces parity and throughput floors (with default `--points-s-tolerance` 0.35); disable per-row throughput via `skip_points_s_regression: true` if a runner is too variable. Pixi: `sphere-bench-geom-exact-contains-gate`. |
| `bench_catalog_match_exact.py` | Exact crossmatch correctness of the torchsky public API (`crossmatch_sky`, `match_within`, `match_nearest`, `count_within`) against brute force, mpmath, scipy `cKDTree` and astropy on named adversarial geometry (poles, RA wrap, ties, radius ladder, per-row radii, clusters, two-sided rules). |
| `bench_catalog_match_perf.py` | Fair crossmatch performance against scipy `cKDTree` and astropy: one subprocess per engine (peak RSS per engine), equal threads, build and query timed separately, pair-ID parity first; sparse, dense and crowded layouts; `best`, `all`, `knn2`. GPU nearest-neighbour peers on 3-D unit vectors (`--gpu-peers`: CuPy `KDTree`, FAISS GPU flat, cuVS brute force, `torch_cluster`, Open3D NNS; `UNAVAILABLE` when not installed, `INEXACT` when pair IDs differ, peak GPU memory). CANFAR mode `catalog-match-gpu-peers` installs them at job time with `scripts/install_gpu_nn_peers.sh`. |
| `bench_catalog_compat.py` | Catalog crossmatch compatibility benchmark against `astropy.coordinates.SkyCoord`, covering both synthetic clustered catalogs and vendored real Astropy VOTable fixtures, plus a real-data FITS roundtrip when native table I/O is available. |
| `bench_sphere_spectral.py` | CPU-first spherical harmonic primitive benchmark (`map2alm`, `map2alm_lsq`, `alm2map`, `almxfl`, `alm2cl`, `anafast`, `map2alm_spin`, `alm2map_spin`) plus compatibility-generation paths (`synalm`, `synfast`, `bl2beam`, `beam2bl`) as needed. Optional `backend=s2fft` rows when s2fft is importable (`sampling="healpix"` only; `jax_healpy` is not a peer). Ingested by `bench_sphere_suite.py` the same way optional mocpy/healsparse rows are. |
| `bench_pipeline_table_sphere.py` | End-to-end benchmark: table predicate pushdown (`where`) + HEALPix spherical reduction pipeline. |

## Methodology

### Throughput

Each case measures **median wall-clock time** over multiple repetitions (typically 3-9 runs). Cases are grouped into two families:

- **smart** &mdash; the idiomatic high-level torchsky compute API. Historical
  FITS smart-path rows are owned by `torchfits`.
- **specialized** &mdash; lower-level or library-specific call paths with explicit mmap control. Used for controlled I/O comparisons.

Fairness controls:

- **mmap parity** &mdash; rows where libraries have mismatched mmap behavior are marked `SKIPPED` and excluded from rankings. Python `fitsio` is not a torchsky comparator; FITS I/O vs `fitsio` is owned by `torchfits`.
- **Comparator provenance** &mdash; sphere-domain comparators must come from official released packages, not local or editable installs.
- **Warm cache** &mdash; FITS benchmarks use configurable file/handle caching (default: `cache=10`, `handle_cache=16`). Cold-start variants available via `--profile cold`.

### Correctness gates

Correctness is validated separately using upstream test fixtures and public reference data. Each gate runs as a pass/fail check:

| Gate | Command | Validates |
|---|---|---|
| Local CI parity (optional) | `pixi run ci-local` | Ruff, compileall, and the full pytest suite in the test env. Git pre-push is `preflight-push` only; the GitHub harness workflow runs preflight plus infrastructure contracts. |
| WCS parity/perf | `pixi run bench-wcs` and `pixi run bench-wcs-tpv-inverse` | WCS projections, spectral/temporal paths, and TPV inverse benchmarks |
| FITS boundary integration | focused cross-package smoke | `torchfits` output preserves torchsky WCS/map/catalog semantics |
| HEALPix geometry | `pixi run sphere-bench-geometry-gate` | HEALPix geometry matrix with correctness and median ratio gates |
| HEALPix advanced | `pixi run sphere-bench-healpix-advanced-gate` | Neighbors and interpolation checks vs healpy |
| Spin transforms | `pixi run sphere-upstream-spin-matrix-gate` | Spin transform matrix benchmark against the committed baseline |
| Polygon parity | `pixi run sphere-bench-geom-exact-contains-gate` | Non-convex containment checks vs spherical-geometry-backed exact regions |
| Mapmaking acceptance | `pixi run bench-map-acceptance-smoke` | Synthetic map truth, multifield, field-fit, and real-fixture map compat gates |

### Output format

Every benchmark run produces three artifacts in `benchmarks_results/<run_id>/`:

| File | Contents |
|---|---|
| `results.csv` | One row per (case, library, family) with median time, throughput, rank, lag ratio |
| `torchsky_deficits.csv` | Only cases where torchsky is not first in its family |
| `summary.md` | Scorecard with win rates, adoption checks, and deficit table |

---

## Reproducing

### Full suite

```bash
# Contract run (WCS and sphere)
pixi run bench-all

# Exact scope aliases
pixi run bench-wcs        # == pixi run bench-all -- --wcs-only
pixi run bench-sphere     # == pixi run bench-all -- --sphere-only

# Direct CLI (equivalent)
pixi run -e bench-all -- python benchmarks/bench_all.py --scope all
pixi run -e bench-all -- python benchmarks/bench_all.py --wcs-only
pixi run -e bench-all -- python benchmarks/bench_all.py --sphere-only

# Output contract
# benchmarks_results/<run_id>/
#   - results.csv
#   - torchsky_deficits.csv
#   - summary.md

# Image-plane mapmaking run (synthetic + real FITS fixture; native extension required)
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_image_plane.py --size 512 --n-obs 4 --runs 5 --footprint-order 8

# Image-plane mapmaking correctness run against synthetic truth (native extension required)
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_ground_truth.py --size 128 --runs 3

# Same run, but also emit the concise Markdown summary next to the JSON artifact
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_ground_truth.py --size 128 --runs 3 --json-out /tmp/map_truth.json

# Image-plane method matrix against synthetic truth, including optional external backends
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_ground_truth.py --size 128 --runs 3 --include-external

# Diagnose per-observation reprojection differences on the rectangular large-union truth case
pixi run -e map-compat python benchmarks/diagnose_map_large_union_reprojection.py --size 512 --width 512 --height 384

# Diagnose grouped-vs-per-observation accumulation differences on the same large-union TAN weighted case
pixi run -e map-compat python benchmarks/diagnose_map_large_union_accumulation.py --size 2048 --width 2048 --height 1536 --scenario no_mask

# Diagnose residual tail differences between stack methods on a wide-projection row
pixi run -e map-compat python benchmarks/diagnose_map_projection_tail.py --case ARC:tan_multires_noise@768 --left-method mean --right-method median

# Sweep bilinear source-boundary support margins against truth and SWarp on the large-union TAN weighted case
pixi run -e map-compat python benchmarks/bench_map_boundary_support.py --size 2048 --width 2048 --height 1536 --scenario no_mask

# Multiwavelength latent-field cube truth run
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_multifield_truth.py --size 96 --runs 3

# Rectangular large-scale multifield run with mixed input image sizes
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_multifield_truth.py --cases tan_multifield_rectangular_scale,tpv_multifield_rectangular_scale --width 2048 --height 1536 --runs 1 --torchsky-kernels bilinear

# Ten-image large-scale random multifield stress run with mixed resolutions and some non-overlap
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_multifield_truth.py --cases mixed_multifield_random10 --width 2048 --height 1536 --runs 1 --torchsky-kernels bilinear

# Multiwavelength latent-field cube smoke baseline gate
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_multifield_truth.py --size 48 --runs 2 --cases tan_multifield_cube --torchsky-kernels bilinear --baseline-json benchmarks/baselines/map_multifield_truth_smoke.json

# Larger representative multifield gate with timing checks
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_multifield_truth.py --size 96 --runs 3 --cases tan_multifield_cube,tpv_multifield_cube --torchsky-kernels bilinear --baseline-json benchmarks/baselines/map_multifield_truth_rep.json

# Experimental latent-field fitting run
# Includes exact-transmission spectral rows such as `tan_field_fit_exact_transmission`
# and `tpv_field_fit_deep_multiband_exact`, plus harder same-field broad-band
# deep-field rows such as `tan_sip_field_fit_deep_multiband_broad` and
# `tpv_field_fit_deep_multiband_broad`, for direct projected-field vs per-band
# stack comparisons on the same product grid. New sparse larger-mosaic rows such as
# `tan_sip_field_fit_sparse_large_mosaic_broad` and
# `tpv_field_fit_sparse_large_mosaic_exact` keep the visits small while scaling
# the requested product to a larger field. Experimental pure-INR families
# include fixed and learned spectral bases, factorized sky-lambda planes,
# multiscale sky-lambda K-planes with refinement, transmission-conditioned latent
# fields, sky-encoder bandpass aggregators, conditioned spectral decoders, and
# hybrid hash-grid spatial encoders with explicit wavelength decoders.
# The broad-band deep-field rows currently use training-time shared mixture
# Monte Carlo supervision plus a light deterministic full-band anchor on one
# observation per step, while keeping deterministic quadrature for evaluation.
# The tuned `sky_encoder_resmlp` branch is now case-split too: the TAN-SIP broad
# row keeps the stronger long-budget non-multiscale setting, while the TPV broad
# row keeps the multiscale readout variant that currently wins its support-aware
# benchmark.
# The sparse broad-band mosaic row remains on deterministic quadrature because
# both sampled training paths regressed that larger regime in current benchmarks.
# The structured sparse TPV exact `sky_lambda_feature_field` row now uses the
# mixed `band_global_mix` reduction, which slightly improves both support-aware
# and full-grid RMSE over the earlier pure `per_band` default on the current
# `128x128` frontier probe.
# The sparse TPV exact benchmark family now also includes a limiting-case
# `tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit` row with only
# three weakly coupled exact bands but fifteen visits per band, so we can test
# whether the latent field can beat three separate coadds by pooling signal
# across forty-five total exposures without relying on dense spectral sampling.
# Experimental coherent shared-sky patch sampling is available in the fitter,
# but current large sparse TPV probes keep the older independent patch
# minibatches because the shared-sky variant improved support-region errors in
# some longer runs while clearly regressing the full-grid frontier metrics.
# That 3-band sparse TPV frontier now uses the new chunked sparse-training
# contract for `sky_lambda_feature_field`: band-balanced observation minibatches
# plus local training patches. On the hard `128x128` row this improves the field
# over the earlier full-frame fit, even though the coadd still remains better.
# The same chunked contract is now also used on the sparse TPV broad-overlap
# frontier. On the current `32x32` smoke it improves support-aware field
# accuracy relative to the old full-frame feature-field path, and at a larger
# `16`-step budget on that same smoke row it does beat the direct coadd.
# Field-fit rows now report both full-grid product metrics and support-aware
# metrics using the `coadd_bilinear` support mask as the comparison domain.
# Recommendation rows prefer `support_reconstruction_rmse` when it is available.
# Sparse frontier rows now also emit cheap region-split diagnostics:
# - supported single-overlap vs multi-overlap RMSE
# - fringe-extrapolated vs far-extrapolated RMSE
# so large sparse-mosaic failures can be localized without a separate notebook.
# They also emit a true uniform-grid metric (`uniform_reconstruction_rmse`) that
# ignores each method's native support mask, so extrapolating fields and masked
# coadds can be compared on the same target grid without ambiguity.
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 48 --steps 120 --runs 1

# Focused sparse-mosaic frontier profile with case-specific step budgets
# - `tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit` runs at 32 steps
# - `tpv_field_fit_sparse_large_mosaic_broad_denseoverlap` runs at 16 steps
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 32 --steps 8 --step-profile sparse_mosaic_frontier --runs 1 --cases tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit,tpv_field_fit_sparse_large_mosaic_broad_denseoverlap --models coadd_bilinear,sky_lambda_feature_field,sky_encoder_resmlp

# Public UPdec-Webb real-data smoke on one or more bands. This currently
# benchmarks the published product pairs directly (Drizzle iter0 vs UPDC iter10)
# rather than the raw JWST exposure-level forward model.
pixi run -e test python scripts/download_updec_webb.py --bands F277W --iterations 0,10
pixi run -e test python benchmarks/bench_updec_webb_real.py --bands F277W

# Sequentially download the public multiband references plus a manageable
# module-B public JWST exposure subset for each band into data/.
pixi run -e test python scripts/download_smacs0723_multiband_public.py --bands F090W,F150W,F200W,F277W,F356W,F444W --module b --updec-iterations 0

# First real exposure-level JWST SMACS J0723 smoke, using local public F277W
# calibrated exposures plus the published F277W iter0 drizzle product as the
# target/reference grid. Native SIP headers now run by default; `--linearize-sip`
# remains available as a fallback/debug option.
pixi run -e test python scripts/download_jwst_smacs0723_mast.py --band F277W --max-exposures 6
pixi run -e test python benchmarks/bench_map_smacs0723_real.py --crop-size 768

# Reproduction sweep against the published F277W drizzle patch with several
# torchsky map variants on the same real exposures. Native SIP now runs by
# default; `--linearize-sip` remains available as a fallback/debug option.
# If local `*_crf.fits` files are present they are preferred over `*_cal.fits`.
pixi run -e test python benchmarks/bench_map_smacs0723_reproduce.py --crop-size 512

# Same sweep, but also add true external drizzle rows on the matched CRF set.
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_reproduce.py --crop-size 512 --include-drizzle --drizzle-kernels point,square

# Direct in-tree drizzle-style reproduction on the same matched CRF set.
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_real.py --crop-size 512 --resampling-kernel drizzle_square --value-semantics surface_brightness --stacking-method weighted_mean --drizzle-pixfrac 1.0

# Real-data stacker comparison on the paper-faithful 9-visit F277Wb crop.
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_stackers.py --band F277W --module b --input-product cal --filename-contains nrcblong --crop-size 512 --crop-mode center

# Real-data multiband non-INR stacker sweep across all public UPdec-Webb bands,
# reusing the same legacy and torchsky stackers sequentially per band.
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_stackers_band_sweep.py --bands F090W,F150W,F200W,F277W,F356W,F444W --module b --input-product cal --crop-size 512 --crop-mode best_overlap

# Exposure ladder on the same real SMACS J0723 F277W target, using the current
# best quick reproduction setting (`bilinear + flux + weighted_mean`) and an
# overlap-aware crop so we can see how agreement changes as more real visits
# are added.
pixi run -e test python benchmarks/bench_map_smacs0723_ladder.py --tiers 2,4,6 --crop-size 512 --crop-mode best_overlap --resampling-kernel bilinear --value-semantics flux --stacking-method weighted_mean

# Real-data INR smoke on the same F277W target. This currently treats the
# public F277W exposures as an exact single-band field-fit problem and compares
# them against the matching coadd support region on the published drizzle grid.
pixi run -e test python benchmarks/bench_map_smacs0723_field_fit.py --crop-size 128 --crop-mode best_overlap --steps 4 --models coadd_bilinear,spectral_basis_field,sky_encoder_resmlp,sky_lambda_feature_field,operator_constrained_feature_field

# Real-data per-band INR sweep across multiple published UPdec-Webb bands,
# using local module-B public JWST exposures when present.
pixi run -e test python benchmarks/bench_map_smacs0723_field_fit_band_sweep.py --bands F090W,F150W,F200W,F277W,F356W,F444W --module b --crop-size 32 --steps 2

# Real-data joint multiband INR smoke on the currently complete public bands.
# This fits one field across several filters simultaneously, then scores each
# rendered band against its own published iter0 reference crop.
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_field_fit_joint.py --bands F090W,F277W,F356W,F444W --crop-size 32 --crop-strategy best_overlap --steps 4 --models coadd_bilinear,spectral_basis_field,sky_encoder_resmlp,sky_lambda_feature_field,operator_constrained_feature_field

# Focused large sparse-mosaic frontier profile
# - keeps the exact 3-band multivisit row at 32 steps
# - drops the broad dense-overlap row to a low-memory 12-step profile at 128+
# - the 128+ broad row also switches to a smaller, deterministic-quadrature
#   `sky_lambda_feature_field` contract and weighted global MSE fitting so the
#   benchmark stays runnable
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 128 --steps 6 --step-profile sparse_mosaic_large_frontier --runs 1 --cases tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit,tpv_field_fit_sparse_large_mosaic_broad_denseoverlap --models coadd_bilinear,sky_lambda_feature_field,sky_encoder_resmlp

# The unified operator-aware follow-up family can be compared on the same sparse
# frontiers. It keeps a continuous spectral scaffold, adds observability-aware
# spectral regularization, and now uses a width-gated support-refinement head so
# exact or delta-like bands stay on the pure continuous path.
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 32 --steps 8 --step-profile sparse_mosaic_frontier --runs 1 --cases tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit,tpv_field_fit_sparse_large_mosaic_broad_denseoverlap --models coadd_bilinear,sky_lambda_feature_field,operator_constrained_feature_field

# Focused scoreboards from these runs are local generated artifacts; keep them
# under `benchmark_results/` or `data/benchmark_results/` unless deliberately
# promoted into a committed baseline.

# Staged stronger non-INR baseline: a coadd-initialized explicit spectral grid
# refined directly against the observations through the same forward model.
# This is currently useful on the sparse exact 3-band gate, but not yet on the
# sparse broad-overlap gate.
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 32 --steps 8 --step-profile sparse_mosaic_frontier --runs 1 --cases tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit,tpv_field_fit_sparse_large_mosaic_broad_denseoverlap --models coadd_bilinear,explicit_spectral_grid_field

# Experimental latent-field fitting run with baseline regression gate
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 24 --steps 8 --runs 2 --cases tan_field_fit --models sine_implicit,coadd_bilinear --baseline-json benchmarks/baselines/map_field_fit_smoke.json

# Focused spectral-basis exact-band sweep helper
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/diagnose_field_fit_sweep.py --size 24 --steps 32,64 --hidden-dims 96,128 --hidden-layers 2 --spatial-frequencies 6,8 --spectral-frequencies 4,6

# Convenience Pixi gate tasks for the finalized map contract
pixi run bench-map-ground-truth-gate
pixi run bench-map-multifield-gate
pixi run bench-map-field-fit-gate
pixi run bench-map-compat-gate

# Larger representative gates with active timing checks on stack, multifield, and field-fit
pixi run bench-map-ground-truth-gate-rep
pixi run bench-map-multifield-gate-rep
pixi run bench-map-field-fit-gate-rep

# Aggregate acceptance entry points
pixi run bench-map-acceptance-smoke
pixi run bench-map-acceptance

# Mapmaking compatibility run (torchsky vs optional external reprojection libraries)
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_compat.py --size 256 --n-obs 3 --runs 3

# Competitive large-union stack benchmark on the fixed 10-image rectangular case
pixi run bench-map-ground-truth-competitive-large-union

# Aggregate the 512 / 1024 / 2048 competitive large-union runs into one report
pixi run bench-map-ground-truth-competitive-large-union-aggregate

# Competitive large-union stack benchmark on the rectangular target with explicit weight maps and masks
pixi run bench-map-ground-truth-competitive-large-union-rect-masked

# Aggregate the 512x384 / 1024x768 / 2048x1536 rectangular weight+mask runs into one report
pixi run bench-map-ground-truth-competitive-large-union-rect-masked-aggregate

# Combined TAN/SIP scoreboard, quick routine profile:
# - large-union TAN truth runs at 512x384 / 1024x768
# - weighted-stack diversity rows across aligned, noisy, multires, low-overlap, and mosaic TAN cases
# - weighted-mean final-stack comparisons against fair external tools on those diverse TAN cases
# - representative mean/median method sweep on the weighted+masked TAN family
# - real SIP compatibility runs against fair external tools
pixi run bench-map-tan-sip-suite

# Combined TAN/SIP scoreboard, full graduation profile:
# - large-union TAN truth runs at 512x384 / 1024x768 / 2048x1536
# - broader weighted-stack diversity coverage including stronger multires and larger mosaic cases
# - weighted-mean final-stack comparisons against fair external tools on those diverse TAN cases
# - representative mean/median method sweep on the weighted+masked TAN family
# - real SIP compatibility runs against the extended comparator set
pixi run bench-map-tan-sip-suite-full

# Combined wide-projection mosaicking scoreboard, quick routine profile:
# - synthetic truth-backed weighted-mean stacks across all current target projection families
#   (`SIN`, `ARC`, `ZEA`, `STG`, `CEA`, `MER`, `AIT`, `MOL`)
# - equal-PSF aligned controls on representative projections so weighted-mean behavior can be judged
#   without PSF-depth coupling
# - hard-row Projection Stack Methods sweep on CEA / AIT with `weighted_mean`, `mean`, and `median`
#   under `bilinear` and `lanczos3`
# - Montage-first compatibility rows on representative cylindrical and all-sky targets
# - INR Common-PSF rows:
#   - same-grid direct-stack vs INR projected products
#   - one shared implicit blur level per case (`common_psf_sigma_pix`)
#   - coordinate-mode comparison for INR (`local_tangent` vs `sphere_xyz`)
#   - pure observation-space INR fitting, with no direct-stack prefit
#   - robust Charbonnier loss plus cosine LR schedule / gradient clipping for INR rows
#   - quick profile keeps the strongest current future-facing INR candidates
#     (`wire_implicit@sphere_xyz`, `sky_encoder_mlp@sphere_xyz`, `hashgrid_mlp@sphere_xyz`)
#   - exact-wavelength product regressions live in the unit-test surface and compare
#     `render_field_product(...)` directly against per-band stacks on the same target grid
#   - direct-vs-INR RMSE/time deltas against the best direct stack baseline
# - support-stratified truth metrics (`support_fringe_q25_*`, `support_core_q75_*`) so wide-projection
#   rows can distinguish low-support fringe errors from high-support core behavior
pixi run bench-map-projection-mosaic-suite

# Combined wide-projection mosaicking scoreboard, full expansion profile:
# - aligned, shifted-overlap, multiresolution, low-overlap, and large-union rows across the target projection set
# - equal-PSF aligned controls across representative zenithal, cylindrical, and all-sky projections
# - large hard-row Projection Stack Methods sweep across zenithal, cylindrical, and all-sky families
#   (`ARC`, `STG`, `CEA`, `MER`, `AIT`, `MOL`) with `weighted_mean`, `mean`, and `median`
#   under `bilinear` and `lanczos3`
# - Montage compatibility rows on the fair medium/large projection cases
# - medium and large INR Common-PSF rows with both direct stacks and INR coordinate-mode sweeps
pixi run bench-map-projection-mosaic-suite-full

# Exact-band field-fit support-aware representative slice
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/bench_map_field_fit.py --size 24 --steps 64 --runs 1 --cases tan_field_fit_exact_transmission,tpv_field_fit_deep_multiband_exact --models spectral_basis_field,coadd_bilinear,sky_encoder_mlp
# Exact-band field-fit JSON rows now include:
# - `band_centers_angstrom`
# - `band_reconstruction_rmse`
# - `band_flux_relative_error`
# - `support_band_reconstruction_rmse`
# - `support_band_flux_relative_error`
# - uncertainty diagnostics:
#   - `uncertainty_kind`
#   - `uncertainty_error_corr`
#   - `uncertainty_top_bottom_abs_err_ratio`
#   - `uncertainty_support_mean_sigma`
#   - `uncertainty_extrapolated_mean_sigma`

# Residual diagnostic for the exact-band field-fit gates, using the tuned per-case model defaults
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/diagnose_field_fit_residuals.py --case tpv_field_fit_deep_multiband_exact --model-kind sky_encoder_mlp
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/diagnose_field_fit_residuals.py --case tan_field_fit_exact_transmission --model-kind spectral_basis_field

# Broad-band deep-field representative slice
# These rows emulate the next-stage same-field HST/JWST-style challenge more closely:
# repeated visits per broad band, mixed scale/depth/coverage, known transmissions,
# shared PSF, and TAN-SIP / TPV source distortion.
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/bench_map_field_fit.py --size 24 --steps 64 --runs 1 --cases tan_sip_field_fit_deep_multiband_broad,tpv_field_fit_deep_multiband_broad --models coadd_bilinear,sky_encoder_bandpass_field,sky_encoder_resmlp

# Fast, deterministic exposure-and-sky holdout contract. The split is made
# before product planning, coaddition, domain estimation, initialization, and
# fitting. NE and SW training quadrants are removed in sky coordinates, while
# evaluation uses PSF/resampling-guarded cores. Field rows report held-out
# detector-space weighted RMSE; every method reports the same held-out-sky
# truth metrics and zero training support in those cores.
pixi run bench-map-field-heldout-contract

# Sparse larger-mosaic deep-field smoke
# These rows keep the individual visits small while growing the requested product
# field, closer to the "many HST/JWST visits on one larger mosaic" regime.
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/bench_map_field_fit.py --size 128 --steps 8 --runs 1 --cases tan_sip_field_fit_sparse_large_mosaic_broad,tpv_field_fit_sparse_large_mosaic_exact --models coadd_bilinear,sky_encoder_bandpass_field,sky_encoder_mlp

# The simulation and evaluation flow for these map truth benchmarks is documented in:
# docs/map.md

# Real-fixture compatibility smoke gate on the fixed Astropy SIP products
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_compat.py --size 32 --n-obs 2 --runs 1 --cases real_astropy_sip_fixture,real_astropy_sip_shifted_wcs,real_astropy_sip_mef,real_astropy_sip_mef_shifted_wcs --torchsky-kernels bilinear --reproject-kernels bilinear --drizzle-kernels point --swarp-kernels bilinear --baseline-json benchmarks/baselines/map_compat_real_smoke.json

# External-tool compatibility run in the dedicated Python-3.13 env
pixi run -e map-compat python scripts/run_local_sphere_bench.py benchmarks/bench_map_compat.py --size 256 --n-obs 3 --runs 3

# Bootstrap pinned official source builds for external map comparators
pixi run -e map-compat python scripts/bootstrap_map_compat_tools.py --tool swarp --force-source
pixi run -e map-compat python scripts/bootstrap_map_compat_tools.py --tool montage --force-source
pixi run -e map-compat python scripts/bootstrap_map_compat_tools.py --tool casutools
# `casutools` defaults to downloading `casutools-1.0.43.tar.gz`; override with
# `CASUTOOLS_SOURCE=/path/to/tree` or `CASUTOOLS_URL=https://...tar.gz` if needed.

# Required official-tool rows plus committed absolute numerical acceptance bounds
pixi run bench-map-drizzle-required
pixi run bench-map-swarp-required
pixi run bench-map-swarp-clipped-required
pixi run bench-map-montage-required

# WCS parity+throughput run with mixed interior/boundary sampling and p99 gates
pixi run -e bench-all -- python benchmarks/bench_wcs.py --cases TAN,SIN,ARC,AIT,MOL,HPX,CEA,MER,TAN_SIP,TPV --sample-profile mixed --n-points 200000 --runs 5 --p99-angular-error-arcsec 1e-2 --p99-inverse-pixel-error 1e-2

# WCS small-N center-projection check (bench_wcs emits center_cpp_* fields when active)
pixi run -e bench-all -- python benchmarks/bench_wcs.py --cases AIT,HPX,CEA,MER,CAR,SFL --sample-profile mixed --n-points 512 --runs 9

# WCS small-N: Astropy vs torchsky forward + inverse (two-arg vs stacked vs stacked+compile)
pixi run -e bench-all -- python benchmarks/bench_wcs_small_n_paths.py --cases TAN,SIN,ARC --n-points 256 --runs 11

# WCS spectral/time benchmark against Astropy/WCSLIB as comparator only.
# The torchsky runtime must never fall back to Astropy/WCSLIB for these paths.
pixi run bench-wcs-spectral-temporal

# Coupled FITS -TAB lookup and main-WCS forward/inverse benchmark.
pixi run bench-wcs-tabular

# Dedicated TPV inverse primitive benchmark
pixi run bench-wcs-tpv-inverse

# HEALPix parity+throughput run on CPU (strict parity gates)
pixi run -e sphere-bench -- python benchmarks/bench_healpix.py --device cpu --sample-profile mixed --nside 1024 --n-points 200000 --runs 5 --max-index-mismatches 0 --max-pix2ang-dra-deg 1e-10 --max-pix2ang-ddec-deg 1e-10

# HEALPix CUDA run with CPU comparison gate
pixi run -e sphere-bench -- python benchmarks/bench_healpix.py --device cuda --compare-cpu --sample-profile mixed --nside 1024 --n-points 200000 --runs 5 --min-cuda-speedup-vs-cpu 2.0

# HEALPix Apple GPU (MPS) run with CPU comparison (float32 thresholds)
pixi run -e sphere-bench -- python benchmarks/bench_healpix.py --device mps --compare-cpu --sample-profile uniform --nside 1024 --n-points 200000 --runs 5 --max-index-mismatches 10 --max-pix2ang-dra-deg 5e-4 --max-pix2ang-ddec-deg 5e-4

# Advanced HEALPix benchmark (neighbors + interpolation)
pixi run sphere-bench-healpix-advanced

# Advanced HEALPix benchmark gate (ratio thresholds vs healpy)
pixi run sphere-bench-healpix-advanced-gate

# Advanced HEALPix small-N gate (overhead-sensitive)
pixi run sphere-bench-healpix-advanced-gate-small

# Sphere-core benchmark (multi-band sampling + geometry primitives)
pixi run sphere-bench-core

# Non-convex polygon benchmark (contains/intersects/query/area)
pixi run sphere-bench-polygons

# Catalog crossmatch refinement benchmark (chunk-wide, HEALPix, and per-row indexing)
pixi run catalog-match-exact
pixi run catalog-match-perf

# Catalog crossmatch compatibility benchmark (torchsky vs astropy SkyCoord)
pixi run sphere-bench-catalog-compat

# Spectral primitive benchmark (CPU-first scalar + spin transforms)
pixi run sphere-bench-spectral

# End-to-end table pushdown + sphere pipeline benchmark
pixi run sphere-bench-pipeline

# End-to-end table pushdown + sphere pipeline gate (ratio thresholds)
pixi run sphere-bench-pipeline-gate

# Install optional sphere benchmark comparators from official releases
pixi run sphere-bench-bootstrap

# Install only core HEALPix comparator packages (lean setup)
pixi run sphere-bench-bootstrap-core

# Sync upstream release source artifacts and extract tests/data fixtures
pixi run sphere-upstream-sync

# Optional extended spin matrix (higher-complexity spin-2 cases)
pixi run sphere-upstream-spin-matrix

# Cross-library sphere geometry benchmark (fails if comparator comes from local/editable/VCS install)
pixi run sphere-bench-geometry-fast

# Ecosystem benchmark snapshot against released packages (healpy/hpgeom/astropy-healpix/healpix/mhealpy)
pixi run sphere-bench-geometry-ecosystem

# Cross-library sphere geometry matrix with correctness + median ratio gates
pixi run sphere-bench-geometry-gate

# Small-N matrix/profile gate to catch overhead-sensitive regressions
pixi run sphere-bench-geometry-gate-small
```

### Parallel by domain

```bash
pixi run bench-wcs &              # WCS projections
pixi run bench-sphere &           # HEALPix + harmonics + polygons
wait
```

### Specialized runs

```bash
# Cross-library sphere geometry ecosystem snapshot
pixi run sphere-bench-geometry-ecosystem
```

---

## Historical Release Snapshot (0.3.2)

Run ID: `20260318_083620`. Run date: 2026-03-18. Profile: `user`. Platform: macOS arm64 (Apple Silicon).

### Scorecard

| Domain | Family | Cases | torchsky First | Win Rate |
|---|---|---:|---:|---:|
| fits | smart | 84 | 84 | 100.0% |
| fits | specialized | 165 | 165 | 100.0% |
| fitstable | smart | 70 | 70 | 100.0% |
| fitstable | specialized | 70 | 70 | 100.0% |
| wcs | smart | 150 | 128 | 85.3% |
| sphere | specialized | 18 | 16 | 88.9% |

**Large-N leadership** (N &ge; 100K): 100% across all domains. Zero large-N deficits.

Note: Python `fitsio` is not a torchsky peer. Historical FITS/table rankings vs `fitsio` live in `torchfits`. The torchsky compute comparators are vs astropy WCS / SkyCoord and sphere libraries.

---

### FITS Image I/O

Compared against **astropy** (smart family). Coverage: 7 dtypes (int8 &ndash; float64), 1D/2D/3D, 4 size tiers, Rice/HCOMPRESS/GZIP compressed, BSCALE/BZERO scaled, MEF, cutouts, header-only reads.

**Representative `read_full` results (smart family, torchsky vs astropy):**

| Case | torchsky | astropy | Ratio (astropy / tf) |
|---|---:|---:|---:|
| tiny_int16_1d | 73 &mu;s | 6.5 ms | 89x |
| tiny_int32_1d | 31 &mu;s | 2.6 ms | 87x |
| small_float32_1d | 55 &mu;s | 1.1 ms | 19x |
| small_float64_2d | 145 &mu;s | 3.6 ms | 25x |
| medium_float32_1d | 104 &mu;s | 3.7 ms | 35x |
| medium_int16_1d | 125 &mu;s | 2.0 ms | 16x |
| medium_float64_2d | 1.5 ms | 5.8 ms | 3.8x |
| large_float32_1d | 621 &mu;s | 1.6 ms | 2.6x |
| large_int16_2d | 1.3 ms | 4.8 ms | 3.6x |
| large_float64_2d | 6.0 ms | 8.5 ms | 1.4x |
| compressed_rice_1 | 6.6 ms | 19.8 ms | 3.0x |
| compressed_gzip_2 | 7.7 ms | 45.9 ms | 6.0x |
| compressed_hcompress_1 | 21.9 ms | 28.4 ms | 1.3x |

Win rate: **100%** across all 84 smart-family and 165 specialized-family comparable cases. No deficits.

---

### FITS Table I/O

Compared against **astropy** (smart family). Coverage: narrow/wide/mixed/varlen schemas, 1K &ndash; 1M rows, column projection, row slicing, predicate filter, streaming scan.

**Representative results (smart family, torchsky vs astropy):**

| Case | torchsky | astropy | Ratio |
|---|---:|---:|---:|
| narrow_1000 [read_full] | 58 &mu;s | 1.6 ms | 27x |
| narrow_1000000 [read_full] | 131 &mu;s | 22.1 ms | 168x |
| wide_100000 [read_full] | 67 &mu;s | 25.8 ms | 383x |
| mixed_1000 [row_slice] | 57 &mu;s | 3.1 ms | 54x |
| mixed_100000 [row_slice] | 61 &mu;s | 3.9 ms | 63x |
| mixed_1000000 [read_full] | 78 &mu;s | 38.9 ms | 502x |
| varlen_100000 [row_slice] | 67 &mu;s | 8.2 ms | 123x |
| mixed_100000 [scan_count] | 142 &mu;s | 3.0 ms | 21x |
| narrow_1000 [predicate_filter] | 1.3 ms | 2.1 ms | 1.6x |

Win rate: **100%** across all 70 smart-family and 70 specialized-family comparable cases. No deficits.

---

### WCS

Compared against **astropy.wcs** (smart family). PyAST is available as a legacy comparator when bridge data is present but were not included in this run. Coverage: 15 projections &times; forward/inverse &times; N = {1K, 10K, 100K, 1M, 10M}.

**Results at N = 10M (smart family, torchsky vs astropy.wcs):**

| Projection | Forward | Inverse | Forward ratio | Inverse ratio |
|---|---:|---:|---:|---:|
| TAN | 455 ms | 232 ms | 2.3x | 5.2x |
| TAN_SIP | 317 ms | 839 ms | 3.3x | 2.9x |
| TPV | 413 ms | 5.01 s | 2.9x | &mdash; |
| SIN | 301 ms | 320 ms | 2.8x | 3.4x |
| ARC | 396 ms | 460 ms | 2.3x | 2.8x |
| ZPN | 267 ms | 284 ms | 3.1x | 3.8x |
| ZEA | 419 ms | 425 ms | 2.4x | 2.8x |
| STG | 404 ms | 432 ms | 2.3x | 2.9x |
| CEA | 56 ms | 61 ms | 5.6x | 2.8x |
| CAR | 31 ms | 48 ms | 6.4x | 3.4x |
| MER | 77 ms | 98 ms | 5.8x | 4.1x |
| AIT | 521 ms | 224 ms | 1.4x | 2.0x |
| MOL | 263 ms | 521 ms | 1.8x | 16.2x |
| SFL | 80 ms | 60 ms | 3.9x | 2.9x |
| HPX | 200 ms | 123 ms | 1.9x | 2.8x |

**Results at N = 100K:**

| Projection | Forward | Inverse | Forward ratio | Inverse ratio |
|---|---:|---:|---:|---:|
| TAN | 4.6 ms | 2.3 ms | 2.4x | 5.7x |
| TAN_SIP | 4.0 ms | 9.1 ms | 2.6x | 2.5x |
| SIN | 4.8 ms | 4.7 ms | 1.9x | 2.4x |
| ARC | 3.7 ms | 3.6 ms | 2.2x | 2.8x |
| ZPN | 3.0 ms | 3.2 ms | 2.6x | 3.2x |
| ZEA | 8.3 ms | 8.2 ms | 1.3x | 1.5x |
| CEA | 442 &mu;s | 483 &mu;s | 7.0x | 3.1x |
| CAR | 228 &mu;s | 340 &mu;s | 8.4x | 3.8x |
| HPX | 1.9 ms | 1.0 ms | 1.8x | 2.9x |

Win rate: **85.3%** overall. At N &ge; 100K: **100%** (90/90 cases).

**Deficits:** 22 deficits total: 21 at N = 1K and 1 at N = 10K (ZEA forward, 2.9x). PyTorch dispatch overhead dominates at small N. No deficits at N &ge; 100K.

---

### Sphere

Compared against **healpy**, **hpgeom**, **astropy-healpix**, **healsparse**, **mocpy** (optional), **s2fft** (optional HEALPix SHT peer, not an oracle), and **spherical-geometry**. Coverage: `ang2pix` (ring/nested), `pix2ang`, `nest2ring`, `ring2nest`, neighbors, interpolation weights/values, sparse map `get_values`/`ud_grade` vs healsparse when installed, MOC contains vs mocpy when installed, non-convex polygon contains/area/query, and spin-weighted harmonic transforms.

**HEALPix geometry (CPU):**

| Operation | torchsky | healpy | Ratio (healpy / tf) |
|---|---:|---:|---:|
| ang2pix ring | 2.6 ms | 5.6 ms | 2.1x |
| ang2pix nested | 3.9 ms | 5.8 ms | 1.5x |
| pix2ang ring | 2.2 ms | 3.2 ms | 1.5x |
| pix2ang nested | 3.5 ms | 4.1 ms | 1.2x |
| ring2nest | 1.4 ms | 1.9 ms | 1.4x |
| nest2ring | 1.8 ms | 1.8 ms | 1.0x |

**Advanced HEALPix:**

| Operation | torchsky | healpy | Ratio |
|---|---:|---:|---:|
| neighbors nested | 1.6 ms | 4.1 ms | 2.6x |
| neighbors ring | 6.7 ms | 9.4 ms | 1.4x |
| interp_weights nested | 20.5 ms | 29.2 ms | 1.4x |
| interp_weights ring | 14.3 ms | 17.2 ms | 1.2x |
| interp_val nested | 33.4 ms | 35.9 ms | 1.1x |
| interp_val ring | 18.4 ms | 26.1 ms | 1.4x |

**Spherical polygons:**

| Operation | torchsky | spherical-geometry | Ratio |
|---|---:|---:|---:|
| contains (non-convex, N points) | 9.2 ms | 28.2 s | 3072x |

**Sparse maps:**

| Operation | torchsky | dense CPU baseline | Ratio |
|---|---:|---:|---:|
| ud_grade | 24.8 ms | 60.5 ms | 2.4x |

**Spherical harmonics:**

The microseconds below are the **pre-Ishioka** Mac rows (same snapshot as
the 88.9% win rate). After the 2026-08 Ishioka skip-2 CPU work, Mac
`map2alm_spin` leads healpy and `alm2map_spin` is ~1.6× healpy at nside=128. See
[`benchmarks/scorecards/20260817_world_layer.md`](../benchmarks/scorecards/20260817_world_layer.md)
(Mac snapshot: [`20260816_world_layer.md`](../benchmarks/scorecards/20260816_world_layer.md)).
Do not treat these 0.1–0.2× ratios as current.

| Operation | torchsky | healpy | Ratio (healpy / tf) |
|---|---:|---:|---:|
| map2alm_spin | 1.1 ms | 170 &mu;s | 0.2x |
| alm2map_spin | 926 &mu;s | 111 &mu;s | 0.1x |

Win rate: **88.9%** (16/18). Large-N: **100%** (13/13). Two deficits listed below.

---

### Known Deficits

| Domain | Case | Lag | Notes |
|---|---|---:|---|
| wcs | N &le; 10K forward/inverse (22 cases) | 1.0x &ndash; 2.9x | Fixed PyTorch dispatch overhead at small N (21 at N = 1K, 1 at N = 10K). All under 3 ms absolute. |
| sphere | `alm2map_spin` CPU | Mac ~1.6× healpy at nside=128; Linux unclaimed | Same-machine Mac: nside=128 `alm2map_spin` 8.0 ms vs healpy 5.1 ms; `map2alm_spin` 5.3 ms vs 6.5 ms (leads). Polar CPU RING FFTs are in-process pocketfft (Accelerate radix-2 on Apple for large pow2). Linux CANFAR SHA `b1f02069` scalar polar FFT at 8 threads: interpolate ~2.5 ms, synthesize ~280 ms, `alm2map_spin` 284 ms vs healpy 3.62 ms. SHA `425af888` uncapped pool was interpolate-bound (193 ms). Remeasure pocketfft on Linux. CUDA/MPS SHT stays on device on the ATen ring path; fused Ishioka/Wigner C++ stays CPU. |
| sphere | fused SHT on GPU | ATen ring on device; no fused CUDA Ishioka | Ishioka/Wigner is CPU C++. CUDA/MPS maps stay on device (ATen ring FFT). Do not claim a fused CUDA Ishioka kernel or Linux CPU SHT leadership. CANFAR CUDA/cuHPX spikes are separate (`scripts/launch_canfar_wcs_sphere.sh`). |

---

## Benchmark Scripts

| Script | Domain | Description |
|---|---|---|
| `bench_all.py` | wcs/sphere | WCS and sphere orchestrator |
| `bench_wcs_suite.py` | wcs | Projection sweep, forward + inverse, N = 1K &ndash; 10M |
| `bench_sphere_suite.py` | sphere | Aggregates geometry, advanced, sparse, spectral, polygon benchmarks |
| `bench_wcs.py` | wcs | WCS throughput with parity gates |
| `bench_wcs_spectral_temporal.py` | wcs | Comparator-only spectral FITS-WCS algorithm and temporal scale-conversion benchmark against Astropy/WCSLIB |
| `bench_wcs_tabular.py` | wcs | Coupled FITS `-TAB` lookup/WCS forward and ambiguity-safe inverse throughput |
| `bench_healpix.py` | sphere | HEALPix CPU/CUDA/MPS parity + throughput |
| `bench_healpix_advanced.py` | sphere | Neighbors + interpolation benchmarks |
| `bench_sphere_geometry.py` | sphere | Cross-library HEALPix ecosystem comparison |
| `bench_sphere_core.py` | sphere | Multi-band sampling, pairwise distances, ellipse queries; `--moc-only` mocpy cone/contains/union |
| `bench_sphere_polygons.py` | sphere | Non-convex polygon operations vs spherical-geometry (`--device` for CUDA/MPS) |
| `bench_observed_place.py` | wcs | Pinned-EOP CIRS/TETE/GCRS/AltAz timing vs ERFA |
| `bench_sphere_spectral.py` | sphere | Spherical harmonic primitives (scalar + spin) |
| `bench_pipeline_table_sphere.py` | mixed | Table pushdown + HEALPix reduction pipeline |

### Upstream-style gates

The current tree does not include the older `benchmarks/replays/` scripts.
Use the committed benchmark and pytest gates above as the supported local parity
surface. Reintroduce replay scripts only with corresponding docs, Pixi tasks,
and `tests/test_docs_integrity.py` updates in the same PR.

## Sphere Benchmark Policy

- Comparator libraries must come from official released distributions (pip/conda), not local repo clones.
- `bench_healpix.py` and `bench_sphere_geometry.py` now print comparator package provenance and fail by default on local/editable/VCS installs.
- `bench_sphere_matrix.py` runs `uniform/boundary/mixed` profiles and can enforce median torchsky/healpy speed-ratio floors.
- `bench_sphere_polygons.py` benchmarks non-convex polygon contains/intersects/query/area, optionally checks parity against released `spherical-geometry`, and emits `torchsky_recommendation` rows describing the preferred contains/intersection backend on the measured workload.
- `scripts/run_local_sphere_bench.py` is used by the polygon bench/replay tasks to editable-install the local package into the active `sphere-bench` env on demand, so those tasks can see the native `torchsky._cpp` extension instead of falling back to Python-only imports.
- `bench_catalog_crossmatch.py` and `bench_catalog_crossmatch_arrow.py` are retired to `docs/archive/catalog-crossmatch-pre-rewrite/` (they compared torchsky internals removed by the zone-index rewrite); their recorded results stay as historical evidence. `bench_catalog_match_exact.py` and `bench_catalog_match_perf.py` replace them.
- `bench_catalog_compat.py` benchmarks `torchsky.catalogs.crossmatch_sky` against `astropy.coordinates.SkyCoord.match_to_catalog_sky`, using both synthetic clustered catalogs and vendored real Astropy VOTable fixtures; when the native table extension is present it also runs a real-data FITS roundtrip case derived from those fixtures. Exact covering NSIDE is `nside_for_cap` (match radius vs pixel radius), not catalog-size bins. Pair-ID vs unit-sphere cKDTree: `scripts/run_catalog_scale_contract.py`. Linux 1e7 vs xmatch `fast` is the survey-scale claim; Mac numbers are wiring. See `docs/research/catalog_matching.md`.
- `bench_map_image_plane.py` benchmarks the current image-plane coadd path on both synthetic observations and real Astropy SIP FITS image fixtures, including shifted-SIP reprojection, generated-MEF cases, and a synthetic rotated low-overlap TPV MEF case, and also times both the multiband compatibility wrapper and the header-derived planned spectral-cube path on a shared target grid; it reports wall time, Mpix/s, coverage fraction, mean accumulated weight, max overlap, footprint area, spectral-axis metadata, and summed spectral projection weight. Run it through `scripts/run_local_sphere_bench.py` so the active env sees the local native extension.
- `bench_map_ground_truth.py` is the mapmaking correctness harness. It does not use `reproject`, `Montage`, `drizzle`, `SWarp`, or any other package as truth. Instead it generates a latent analytic sky field, simulates calibrated observations with known exposure time / transmission / PSF / noise / mask structure, and scores methods against that ground truth with `rmse_vs_truth`, `mae_vs_truth`, `p95_abs_err_vs_truth`, `max_abs_err_vs_truth`, `signal_rms_truth`, `residual_rms`, `output_snr`, `output_snr_db`, and integrated-flux metrics (`total_flux_truth`, `total_flux_output`, `total_flux_abs_err`, `total_flux_rel_err`). Torchsky rows now form a kernel matrix across `nearest`, `bilinear`, `bicubic`, `lanczos3`, `lanczos5`, `mitchell_netravali`, and `catmull_rom`, the synthetic case set includes harder resolution-ratio sweeps plus explicit `flux`-semantics cases, and `--include-external` extends the same truth scoring to the available external backends and variants. Flux-semantics rows also record the synthetic `noise_model`; the current contract uses area-scaled flux noise for flux-per-pixel cases so distorted TPV exposures are judged under a physically consistent variance model instead of a constant-flux-noise artifact. The TPV flux set is split into a milder regression-gated case and a harder stress case that stays in the matrix as an optimization target rather than being hidden. The synthetic TPV benchmark family now uses a non-degenerate linear TPV component and sampled local-Jacobian checks so the warped cases remain locally invertible rather than benchmarking fold/branch ambiguity. In the current torchsky path, distorted `SIP`/`TPV` reprojection uses fixed 2x2 Gauss-Legendre target-pixel quadrature for `flux` products, while the default `surface_brightness` path uses a single center sample; the quadrature world grids are cached once per prepared target so distorted large-map benchmarks measure reprojection quality rather than repeated target-grid recomputation overhead. CPU fused WCS is now enabled by default for supported projections, including TPV, and can be disabled with `TORCHSKY_WCS_FUSED=0` for debugging or comparator studies. The script also emits `torchsky_recommendation` rows per case, including a practical default (`fastest_within_1pct_rmse`), the best interpolatory kernel, and the best smooth reconstruction kernel from the measured torchsky rows.
- `bench_map_ground_truth.py` now evaluates methods on the final requested product contract instead of an implicit script-local output shape. The primary metric vocabulary is standardized with the field-fit benchmark: `reconstruction_rmse`, `reconstruction_mae`, `reconstruction_p95_abs_err`, `reconstruction_max_abs_err`, `signal_to_residual_ratio`, `signal_to_residual_db`, `total_flux_relative_error`, `band_flux_relative_error_mean/max`, `coverage_fraction`, plus color-ratio errors where multiple bands exist. The script writes JSON plus a concise Markdown summary table via the shared map benchmark helpers. `benchmarks/baselines/map_ground_truth_smoke.json` is the lightweight smoke baseline; it gates correctness/SNR/flux on a small TAN case and intentionally skips time regression because the smoke-scale timing is too noisy to be meaningful on its own. `benchmarks/baselines/map_ground_truth_rep.json` is the larger representative stack baseline with active timing checks on `tan_multires_noise` and `tpv_flux_multires_mild`.
- `bench_map_multifield_truth.py` is the first dedicated continuous-field, multiwavelength mapmaking benchmark. It constructs latent spectral-spatial fields with two model families: an explicit analytic basis field and an initial frozen implicit-neural-field (`implicit_siren`) variant. Observation synthesis now uses normalized Gaussian transmission filters across each observation bandpass rather than a pure top-hat average, and the core cube stack path is evaluated on the same product-oriented metric vocabulary as the other map truth benches. The scalar throughput parameter still controls noise level under the assumption that photometric calibration is already known. The benchmark simulates overlapping observations with mixed TAN/TPV geometry, PSF width, bandpass, exposure time, and throughput, then scores the recovered final product against latent cube truth with `reconstruction_*`, SNR, flux, and per-band summaries while still carrying the older `cube_rmse_vs_truth` aliases for continuity. Like `bench_map_ground_truth.py`, it also emits `torchsky_recommendation` rows so the kernel choice is benchmark-backed rather than static. Its synthetic TPV family now matches the locally invertible benchmark policy used in the map ground-truth harness, so the continuous-field warped cases stay physically coherent. It now supports rectangular target grids via `--width/--height`, explicit large-scale rectangular cases with mixed per-observation `NAXIS1/NAXIS2`, and a deterministic ten-image stress case with random widths/heights in the 500–4000 range, mixed TAN/TPV geometry, mixed pixel scales, random shifts/rotations, and deliberate partial non-overlap. `benchmarks/baselines/map_multifield_truth_smoke.json` is the lightweight multifield baseline for `tan_multifield_cube` and skips time regression only where the smoke-scale run is too noisy; `benchmarks/baselines/map_multifield_truth_rep.json` is the larger representative multifield baseline with active timing checks on `tan_multifield_cube` and `tpv_multifield_cube`.
- `bench_map_field_fit.py` is now fully product-oriented too: direct stack baselines and fitted field models are both rendered onto the same requested final product before scoring. The current case matrix spans `TAN`, `TAN-SIP`, and `TPV`; implicit and analytic latent fields; low-overlap / mixed-depth / multiresolution / PSF-mismatch environments; exact-transmission as well as Gaussian-filter spectral targets; and both smaller multicolor and deeper same-field repeated-visit broad-band products. It also now includes sparse larger-mosaic rows where many small visits are fit onto a much larger target product, which is the current bridge from local deep-field INR success to realistic mosaic-scale latent modeling. The model matrix includes `sky_encoder_bandpass_field`, a transmission-aware sky-encoder branch that aggregates bandpass-weighted multiscale sky features before decoding, `sky_encoder_resmlp`, which is currently the strongest broad-band deep-field INR branch after tuning its architecture, `mixture_mc`-based training setup, and case-specific readout policy, and the newer `operator_constrained_feature_field`, which keeps a continuous spectral scaffold but constrains its degrees of freedom through the actual bandpass operator and a small zero-initialized bandpass residual path. Reported metrics use the same standardized product vocabulary as the stack-truth benchmark, plus `fit_time_ms`, observation-fit summaries, support-aware product metrics keyed off the `coadd_bilinear` support mask, and true uniform-grid metrics that help separate supported-pixel fidelity from global-field coherence. `benchmarks/baselines/map_field_fit_smoke.json` is the lightweight smoke baseline and intentionally skips timing on its tiny coadd helper rows. `benchmarks/baselines/map_field_fit_rep.json` is the larger representative field baseline with active timing checks on the field-fit defaults across the current representative rows.
- `diagnose_field_fit_sparse_frontier.py` is the focused scale-ladder surface for those sparse larger-mosaic rows. By default it keeps the honest four-way comparison in view: `coadd_bilinear`, the regular `sky_encoder_mlp` INR, `sky_lambda_kplanes_field`, and the newer `sky_lambda_feature_field`. It sweeps one or more sparse cases across target sizes, fit-step budgets, and selected field models, then reports full-grid and support-aware product metrics on the same final grids. This is the script to use before concluding that a sparse-mosaic INR gap is architectural rather than budget-limited.
- `bench_wcs_tpv_inverse.py` isolates the current warped-map hotspot from the full stack contract. It benchmarks only TPV `world_to_pixel` at large `N`, using Astropy-generated world coordinates from known pixel samples and scoring inverse-pixel reconstruction against the original pixel coordinates. The current matrix is `torchsky` fused TPV inverse, `torchsky` tensor-core TPV inverse with fused WCS disabled, and Astropy `all_world2pix`, across both an affine-only TPV header and a non-linear broad-field TPV header. Reported metrics are `time_ms`, `mpts_s`, `mean_inverse_pixel_error`, `p99_inverse_pixel_error`, `max_inverse_pixel_error`, and tensor-path iteration summaries where available. This is the authoritative surface for further TPV inverse kernel work; map-level TPV optimizations should be justified here before being inferred from the full mapmaking benchmark.
- `bench_wcs_spectral_temporal.py` is the focused comparator-only surface for the current spectral/temporal WCS roadmap slice. It times torchsky against Astropy/WCSLIB for one-axis spectral `FREQ`, `WAVE`, `WAVN`, `FREQ-LOG`, `WAVE-LOG`, `WAVN-LOG`, the supported F/W/A/V compound matrix (`FREQ-*2F`, `WAVE-*2W`, `AWAV-*2A`, `VELO-*2V`), and grism cases including `WAVE-GRI`, `WAVE-GRA`, and `FREQ-GRA`; it also times Astropy time-scale conversions for UTC/TAI/TT/GPS MJD vectors. It records throughput plus max absolute parity error. Astropy/WCSLIB is intentionally confined to this benchmark/test surface and is not a runtime fallback.
- `bench_wcs_tabular.py` is the focused surface for coupled FITS `-TAB` lookup behavior. It times direct `CoupledTabularLookupWCS` and main `WCS(header, tables=...)` forward transforms plus the ambiguity-safe 2D square-grid inverse, and records roundtrip pixel error. It has no Astropy/WCSLIB runtime dependency because arbitrary coupled table-grid ambiguity is a torchsky-native policy surface.
- `bench_map_compat.py` is the external mapmaking comparison harness. It now emits a concrete variant matrix across torchsky `nearest`/`bilinear`/`bicubic`/`lanczos3`/`lanczos5`/`mitchell_netravali`/`catmull_rom` kernels, `reproject_interp`, `dfreproject`, low-level `drizzle`, `drizzlepac`, `Montage`, `SWarp`, and `casutools imstack`, plus explicit pairwise comparison rows for every successful method pair in a case. Each external variant is compared with the semantically matching torchsky variant: Drizzle point/square with `drizzle_point`/`drizzle_square`, SWarp Lanczos3 with torchsky Lanczos3, and Montage with conservative polygon overlap. `--require-comparators` rejects missing or skipped selected rows, while `--acceptance-json` applies committed identity checks and absolute minimum/maximum numerical bounds. `reproject_interp` and `dfreproject` are wired with kernel variants, `drizzle` is wired with multiple low-level kernels, and `SWarp` is wired with multiple resampling modes where the requested target grid is representable. For SWarp, the primary numerical comparison uses pixels with a complete interpolation footprint; common-support and full-grid metrics remain in the same row, together with support IoU and symmetric-difference fractions, so border-policy differences stay visible. For the pure resampling/stacking milestone, the SWarp benchmark helper writes `FSCALASTRO_TYPE NONE` by default so hidden astrometric flux scaling does not leak calibration semantics into the comparator rows. The low-level `drizzle` backend is wired through upstream-style `drizzle.utils.calc_pixmap(...)` and `drizzle.resample.Drizzle.add_image(...)`, and includes an additional CRPIX-translation case patterned after drizzle's own `test_translated_map`. The `Montage` backend is wired through source-built official binaries and uses `mProjectPP` automatically for the fast plane-to-plane cases it actually supports, while falling back to `mProject` for SIP/distorted inputs. The harness also includes generated MEF cases built from real Astropy SIP data plus a synthetic rotated low-overlap TPV MEF case so extension-addressed `SCI/ERR/DQ` style workflows are exercised in addition to single-HDU inputs. `SWarp` is wired through a source-built CLI path for simple equatorial non-distorted target grids and can therefore participate in the TPV-input/TAN-target synthetic case, while SIP targets are reported as explicit skips instead of being benchmarked against an inexact surrogate grid. `casutools imstack` is wired from a local source build and is only exercised on the documented CASU WCS regime (`TAN`/`ZPN`, no `SIP`/`PV` distortion); because its confidence maps are renormalized confidence images rather than direct inverse-variance weights, its comparison rows use coverage-mask semantics for weight/coverage agreement and common-support semantics for data deltas. Some comparators can still fail on the harder TPV inverse problem; those surface as explicit `SKIPPED` rows with the upstream error text instead of aborting the matrix. Missing, unsupported, or not-yet-wired comparators show up as explicit `SKIPPED` rows rather than silently disappearing, and import/ABI failures are reported verbatim when available. The script records device, dtype, seeds, package versions, and pinned source commits in every row, writes JSON plus a concise Markdown summary table, and can enforce a committed real-fixture compat baseline through `benchmarks/baselines/map_compat_real_smoke.json`. The default `sphere-bench` env stays on the main current Python stack; the dedicated `map-compat` env pins Python `3.13` so lagging external packages can still be evaluated when upstream packaging allows it. The bench env pins PyPI `drizzle==2.2.0` and `dfreproject==1.3.1` explicitly so we do not accidentally benchmark older lineages or conflate `drizzle` with `drizzlepac`.
- The opt-in `swarp_clipped` row runs official `COMBINE_TYPE CLIPPED` against
  torchsky's scratch-backed sigma-clipped reducer on a localized-outlier fixture.
  Its weight metric is positive-median normalized because SWarp rescales the
  output weight plane internally; both raw scales remain recorded in the row.
- `run_ground_truth_competitive_large_union.py` is the heavy end-to-end competitor runner for the large union-stack cases. It compares torchsky against the current external stackers on a fixed final target product and writes incremental JSON plus Markdown artifacts while the run is in progress. Torchsky rows now default to a warmed median of three timed runs (`--torchsky-runs 3`) so the heavy warped cases have a more stable internal baseline than a single-shot wall clock; external comparator rows still use the requested `--runs` contract unless a backend requires a minimum of two runs.
- `diagnose_map_large_union_reprojection.py` is the follow-up diagnostic surface for those same large-union cases. It is intentionally narrower than `bench_map_compat.py`: instead of surveying many backends, it compares `torchsky` and `dfreproject` observation-by-observation on the same synthetic truth case, then emits pairwise reprojection deltas and stack summaries so regressions can be localized to specific warped inputs or to the post-reprojection accumulation stage.
- `diagnose_map_large_union_accumulation.py` is the companion diagnostic for the surviving TAN weighted overlap gap. It compares grouped and per-observation accumulation for both `torchsky` and `SWarp`, reports overlap-count histograms and overlap-binned residual statistics, and emits explicit data/weight/numerator hotspots so a final stack mismatch can be traced to a particular overlap regime and, when needed, to a specific contributing observation at a specific target pixel.
- `bench_map_boundary_support.py` is the decision surface for explicit bilinear edge-support policies on the large-union TAN weighted cases. It monkeypatches candidate source-boundary margins only for the benchmark run, then scores each margin against analytic truth and against grouped `SWarp` output. This is the benchmark to use before changing any production boundary-support rule in `image_plane.py`.
- The current large-union TAN edge hunt found an important comparator nuance: on the representative `2048x1536` weighted hotspot, `torchsky`, `reproject_interp`, and `dfreproject` agree on the single-observation bilinear sample while `SWarp` returns zero support. That means the surviving grouped `SWarp` mismatch is not automatically evidence that torchsky's bilinear boundary semantics are wrong; benchmark-only edge-policy experiments should be judged against synthetic truth and at least one additional comparator, not only against `SWarp`.
- `scripts/bootstrap_map_compat_tools.py` clones and builds the official upstream SWarp and Montage sources at the revisions named in that script and can also stage a CASUtools build into `.benchmarks/external-tools/prefix/{swarp,montage,casutools}` from inside the `map-compat` env, either from `CASUTOOLS_SOURCE` or from the default public `casutools-1.0.43.tar.gz` URL. An existing source tree at another revision fails closed. Pass `--force-source` for SWarp to avoid a host/Conda executable taking precedence over the pinned source build. The CASUtools bootstrap expects `cfitsio` and `wcslib` development headers in the active `map-compat` env (or `WCSLIB_INCLUDE` / `WCSLIB_LIB` overrides). `bench_map_compat.py` checks those repo-managed install prefixes before falling back to `PATH`, so source-built external tools show up in the matrix even when Python wheels are unusable on the host architecture.
- The current `SWarp` benchmark path is intentionally narrow: it only runs when the requested target grid can be represented by SWarp's manual output-grid controls. SIP-distorted, PV-distorted, rotated, sheared, or non-equatorial target WCSes are reported as `SKIPPED` rather than benchmarked against an inexact surrogate grid.
- External mapmaking comparisons should source cases in this order:
  1. official upstream tests/fixtures from the comparator package when they exist
  2. popular pipeline scenarios that actually exercise the package in production, for example DrizzlePac/HST and JWST level-3 style resampling flows
  3. torchsky-owned synthetic stress cases used to probe scaling, boundary behavior, and cross-device performance
- `sync_upstream_sphere_sources.py` downloads release artifacts for selected packages and extracts upstream `tests/` + `data/` trees with a pinned manifest. The legacy replay runners that consumed those extracted fixtures are not present in this tree; use the supported benchmark gates listed above unless a future PR restores the replay scripts and docs together.
- `bench_spin_matrix.py` exercises spin transform cases (`map2alm_spin`, `alm2map_spin`) against healpy rel L2 gates. Mac pixi tasks also enforce min-ratio / baseline-regression throughput gates. CANFAR passes `--no-throughput-gate` so Linux jobs record ratios without using Mac thresholds as a science failure.
- Spin matrix runs can use validated `TORCHSKY_*` profiles for local experiments; see **Default `TORCHSKY_*` knobs (WCS / sphere)** below.
- Override only for local development experiments with:
  - `bench_healpix.py --allow-nonrelease-healpy`
  - `bench_sphere_geometry.py --allow-nonrelease-distributions`

## Default `TORCHSKY_*` knobs (WCS / sphere)

These are the **code defaults** when the environment variable is unset (see `torchsky/wcs/core.py` and `torchsky/sphere/spectral.py`). Benchmark and replay drivers may call `os.environ.setdefault(...)` to pin a stricter profile.

| Variable | Default | Purpose |
|----------|---------|---------|
| `TORCHSKY_WCS_FUSED` | `1` | CPU fused `pixel_to_world` / `world_to_pixel` for supported projections (TAN, SIN, ARC, CEA, AIT, HPX, CAR, SFL, MOL, TPV when enabled). Set `0` to force the non-fused tensor path. |
| `TORCHSKY_PERF_GATE_WCS_N` | `200000` | Number of points used by the CI-friendly WCS TAN forward/inverse perf smoke gates in `tests/test_perf_gate.py`. |
| `TORCHSKY_PERF_GATE_WCS_P2W_S` | `2.0` | Maximum median seconds allowed by the WCS TAN bulk `pixel_to_world` perf smoke gate. |
| `TORCHSKY_PERF_GATE_WCS_W2P_S` | `2.0` | Maximum median seconds allowed by the WCS TAN bulk `world_to_pixel` perf smoke gate. |
| `TORCHSKY_RING_FOURIER_CPP` | `1` | RING Fourier analysis/synthesis helpers in spectral transforms. |
| `TORCHSKY_RING_FOURIER_MODES_CPP` | `1` | C++ path for Fourier mode assembly within `mmax` caps. |
| `TORCHSKY_RING_FOURIER_SYNTH_CPP` | `1` | C++ path for Fourier synthesis within `mmax` caps. |
| `TORCHSKY_SPIN_MAP2ALM_RING_CONCAT_CPP` | `1` | Spin `map2alm` ring concat integration. |
| `TORCHSKY_SPIN_RING_FINALIZE_CPP` | `1` | Spin ring finalize stage. |
| `TORCHSKY_SPIN_RING_INTERP_CONCAT_CPP` | `1` | Spin `alm2map` interpolate concat. |
| `TORCHSKY_SPIN_RING_INTEGRATE_CONCAT_CPP` | `1` | Spin `map2alm` integrate concat. |
| `TORCHSKY_SPIN_ISHIOKA` | `1` | Ishioka 2018 skip-2 for spin-1/2 CPU recurrence. Set `0` for Wigner. |
| `TORCHSKY_SPIN_THREADS` | unset | Cap CPU spin `m`-pool workers. Unset uses affinity/cgroup/ATen on Linux. |
| `TORCHSKY_SHT_PROFILE` | unset | stderr split of `alm2map_spin` interpolate vs synthesize. |
| `TORCHSKY_SCALAR_MAP2ALM_RECURRENCE_CPP` | `1` | Scalar `map2alm` recurrence backend. |
| `TORCHSKY_SCALAR_ALM2MAP_RECURRENCE_CPP` | `1` | Scalar `alm2map` recurrence backend. |
| `TORCHSKY_SPIN_RING_AUTO_MIN_BYTES` | `0` | Minimum map size (bytes) before auto-selecting some ring fast paths; replays often set `32 MiB`. |

Phase 0 benchmark procedure for WCS + sphere: [`benchmarks/PHASE0_WCS_SPHERE.md`](../benchmarks/PHASE0_WCS_SPHERE.md). Pixi: `pixi run wcs-sphere-acceptance` (benches + full WCS/sphere pytest + spin matrix report).

## Optimization History

Older benchmark snapshots and optimization history now live in
[archive/benchmark-history.md](archive/benchmark-history.md). Keep current
methodology, commands, and supported benchmark tasks in this file.

## Historical Release Snapshot (0.3.1)

Run metadata:
- Date: `2026-03-06`
- Exhaustive output: `benchmarks_results/20260306_191113`
- Profile: `user` (`cache=10`, `handle_cache=16`, `hot_cache=10`)
- Tables included: `True`

### Astronomer Scorecard (Win Rates)

| Domain | Family | torchsky First | Win Rate | Legacy In Ranking |
|---|---|---:|---:|---:|
| fits | smart | 83/84 | 98.8% | 0 |
| fits | specialized | 165/165 | 100.0% | 0 |
| fitstable | smart | 70/70 | 100.0% | 0 |
| fitstable | specialized | 70/70 | 100.0% | 0 |
| sphere | specialized | 16/18 | 88.9% | 0 |
| wcs | smart | 121/150 | 80.7% | 75 |

### FITS I/O Representative Medians (vs `fitsio`)

| Case | torchsky median (s) | Fitsio median (s) | Speedup vs Fitsio |
|------|---------------------:|------------------:|------------------:|
| `tiny_int8_1d [read_full]` | 0.000044 | 0.000185 | 4.20x |
| `small_float32_2d [read_full]` | 0.000075 | 0.000212 | 2.82x |
| `medium_int16_3d [read_full]` | 0.000524 | 0.000782 | 1.49x |
| `large_float64_2d [read_full]` | 0.005312 | 0.005470 | 1.03x |

### Known Deficits (Target for 0.4.0)

- **Sphere (Spin Transforms)**: Mac fused Ishioka `map2alm_spin` leads healpy; `alm2map_spin` is ~1.6× healpy at nside=128 on that machine. Polar CPU RING FFTs are in-process pocketfft. Linux CANFAR SHA `b1f02069` scalar polar FFT at 8 threads: interpolate ~2.5 ms, synthesize ~280 ms (`alm2map_spin` ~79× healpy at nside=128). SHA `425af888` uncapped `SpinMPool` was interpolate-bound (~52×). Remeasure pocketfft on Linux. Do not claim Linux CPU SHT. Fused SHT C++ is CPU; CUDA/MPS use the ATen ring path on device. Do not claim a fused CUDA Ishioka kernel.
- **Sparse `get_values`**: Mac matches healsparse at 50k nest queries. Linux CANFAR SHA `934b02fa` interleaved is 0.84× (nside=256, 200k) and 0.95× (nside=512, 500k). Not claimed on Linux.
- **WCS (Small-N)**: Some forward small-N rows still lag vs `pyast` (legacy), especially away from the center-equator projections. `bench_wcs.py` now tracks the CPU center-helper crossover explicitly through `center_cpp_*` fields for `AIT/CAR/CEA/HPX/MER/SFL`. The CPU fused path (`TORCHSKY_WCS_FUSED=1`) includes **ARC** alongside TAN/SIN and the other supported codes; re-profile after native extension rebuilds. For torchsky-vs-Astropy timing on **two-arg vs stacked `(N,2)` vs `compile()`** for both **pixel→world and world→pixel**, use `bench_wcs_small_n_paths.py`; pytest gates stacked and compiled-stacked parity (forward and inverse) in `tests/test_wcs_parity.py`.
- **ZPN/AIT Projections**: Slight visible deficits in intermediate size tiers.
