# Comparator coverage matrix

Which torchsky functionality is evaluated against which comparator, and whether it is a
release gate. "Gate" means `TORCHSKY_HARNESS_REQUIRE_GATES=1` makes a missing comparator a
failure (`tests/conftest.py`, `RELEASE_GATE_FILES`); the CANFAR campaign sets it. Exhaustive
domain gates are listed in [releases/0.1.0/gate-matrix.md](releases/0.1.0/gate-matrix.md).

## torchsky 0.5 comparator map

Every torchsky primitive area against the full comparator map ("own API, compared in the harness").
Parity = bounded deterministic gate listed in `scripts/canfar/run_gates.py` (`PARITY_TESTS`);
Performance = throughput and speedup recorded (never gated) by `benchmarks/bench_comparator_perf.py`
(`bench-comparator-perf-gate`, JSON `benchmark_results/comparator_perf*.json`, summary table
"Comparator performance") or an older gate named below. "Oracle" = out-of-process comparator
(`tests/_oracle.py`, NPZ exchange, own interpreter).

| Area | Comparator | Parity gate | Performance | Status |
|---|---|---|---|---|
| wcs | astropy.wcs / WCSLIB | `test_wcs_parity.py`, `test_wcs_upstream_astropy.py`, `test_wcs_sip.py`, `test_wcs_tpv.py` (+ `bench-wcs-release-gate`, `bench-wcs-real-headers-gate`) | `bench-wcs-release-gate`; comparator_perf `wcs` (TAN, TAN-SIP; 1e5, 1e6) | release gate |
| wcs | PyAST (starlink-pyast) | `test_wcs_pyast_parity.py` (TAN, TAN pole, SIN, ARC, ZEA, STG, CAR, CAR wrap, AIT, MOL, HPX, CEA, MER, SFL, ZPN, AZP, TAN-SIP, TPV; 1e-4 mas, SIP/TPV 1e-3 mas; 1e-6 px) | comparator_perf `wcs` (PyAST timed inside its oracle) | release gate, Oracle (own venv) |
| wcs | WCSTools (extra C implementation) | `test_wcstools_parity.py` | - | required on `improc` |
| coords | astropy.coordinates | `test_wcs_frames.py`, `bench_coords_astropy` (`bench-coords-astropy-gate`) | `bench-coords-astropy-gate` | release gate |
| coords | ERFA (pyerfa) | `test_erfa_observed_place_parity.py` | `bench-observed-place` (local task, not a campaign gate) | release gate |
| sphere | healpy | `test_healpix_parity.py`, `test_sphere_compat.py`, `test_synalm_parity.py` | `bench-healpix-release-gate`, `bench-healpix-advanced-release-gate`; comparator_perf `healpix` (ang2pix, pix2ang, neighbours) | release gate |
| sphere | healpy `pixwin` tables + independent reference (`tests/_pixwin_reference.py`) | `test_pixwin_parity.py`: T vs healpy 1e-11 for nside 2-128 (`l <= 4 nside`); P vs healpy within the documented frame term; T vs the independent reference 1e-11 for nside 256 (`l <= 256`; `l <= 767` slow), 512 (`l <= 128`), 1024/2048 (`l <= 64`, slow); healpy deviation recorded (see "pixwin" below) | - | release gate (healpy-data download) |
| sphere | hpgeom | `test_healpix_parity.py` | `bench-healpix-release-gate`; comparator_perf `healpix` | release gate |
| sphere | astropy-healpix | `test_healpix_parity.py` | comparator_perf `healpix` | release gate |
| sphere | healsparse | `test_healsparse_upstream.py` | comparator_perf `sparse` (get_values, ud_grade) | release gate |
| sphere | mocpy | `test_sphere_moc.py` | comparator_perf `moc` (from_circle, union, intersection, contains) | release gate |
| sphere | spherical_geometry | `test_sphere_polygons.py`, `test_sphere_exact_regions.py` | `sphere-bench-polygons` (local task, not a campaign gate) | release gate |
| sphere | s2fft (and healpy SHT) | `test_sphere_s2fft.py`, `test_synalm_parity.py`; healpy SHT in `bench-spin-matrix-release-gate` | comparator_perf `sht` (map2alm, alm2map vs healpy and s2fft), `bench-spin-matrix-release-gate` | release gate |
| map | reproject | `test_map_conservative_external_parity.py`, `test_map_reproject_fft_parity.py` | `bench-map-compat-gate`; comparator_perf `map` (2048, 4096 bilinear and exact) | release gate |
| map | drizzle (STScI package) | `test_map_drizzle_external_parity.py` | `bench-map-drizzle-required` (local) | release gate |
| map | AstroDrizzle (drizzlepac `do_driz`/`cdriz.tdriz`) | `test_map_astrodrizzle_parity.py` (square/point, pixfrac 0.7/1.0, shift/rotate/rescale; 1e-4 abs). The `drizzle` package gate covers the same algorithm re-implemented by STScI; this gate ties torchsky to the original C kernel. AstroDrizzle scales output by the pixel-area ratio (flux conserving), the test divides it out | `bench_map_compat` lists a `drizzlepac` label (local) | release gate, Oracle (own venv) |
| map | dfreproject | `test_map_dfreproject_parity.py` | comparator_perf `map` | release gate |
| map | SWarp | `test_map_swarp_optional_parity.py`, `test_map_external_tools_smoke.py` | comparator_perf `map` (binary wall time), `bench-map-swarp-required` (local) | required on `improc` |
| map | Montage (`mProject`) | `test_montage_projection_parity.py` | comparator_perf `map`, `bench-map-montage-required` (local) | required on `improc` |
| catalogs | astropy SkyCoord matching | `test_catalog_compat.py` | `bench-catalog-release-gate`; comparator_perf `catalogs` (1e5, 1e6) | release gate |
| catalogs | exact crossmatch (brute force, mpmath, scipy cKDTree, astropy) | — | `bench-catalog-match-exact-release-gate` | release gate |
| catalogs | crossmatch performance vs scipy cKDTree / astropy | — | `catalog-match-perf`; CANFAR `catalog`, `catalog-match-1e7`, `catalog-match-1e8`, `cuda` | record |
| catalogs | STILTS (`tmatch2`, `tskymatch2`, `tmatch1`, `tmatchn`) | `test_stilts_parity.py`: synthetic (1e5, RA wrap, pole cap) and **real Gaia DR3 x 2MASS x AllWISE** (`scripts/fetch_real_catalogs.py`, sha256-checked against the manifest); exact row-tuple equality | `test_stilts_parity.py` records per-mode STILTS vs torchsky timings (comparator performance table, area `catalogs`); comparator_perf `catalogs` (tmatch2, tskymatch2; 1e5, 1e6) | required on `improc` (`REQUIRE_TOOLS=1`) |

### pixwin

`torchsky.sphere.pixwin` computes the T and P windows from the pixel shapes for any nside. healpy
reads the HEALPix tables, which HEALPix documents as exact only for `nside <= 128`; above that T is
extrapolated from nside 128 with a top-hat scaling and P is T times a low-`l` factor
([HEALPix docs](https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm)). The gate
therefore uses healpy only where its tables are exact, and an independent reference elsewhere:
`tests/_pixwin_reference.py` integrates `int_p Y_lm^*` exactly in azimuth and with Gauss-Legendre in
the ring coordinate for every pixel class (NumPy only; cost `O(lmax^2 nside^2)`, hence the bounded
`lmax`). The reference matches the exact healpy tables to 1e-11 for nside 2-64 (`l <= 4 nside`).

| nside | `l` checked vs reference | torchsky vs reference | healpy vs reference | healpy vs torchsky, T / P, `l <= 4 nside` | test time (reference) |
|---|---|---|---|---|---|
| 2-128 | `4 nside` (healpy exact) | < 1.4e-13 (via healpy) | < 1.3e-13 | T < 1.3e-13; P ~0.94e-6 (8/nside)^4 (frame term) | < 1 s |
| 256 | 256 / 767 (slow) | 1.2e-14 / 2.8e-14 | 2.4e-6 / 7.3e-6 | 7.3e-6 / 7.0e-6 | 8 s / 68 s |
| 512 | 128 | 1.6e-14 | 1.3e-6 | 1.06e-5 / 1.03e-5 | 7 s |
| 1024 | 64 (slow) | 1.3e-14 | 1.3e-6 | 1.21e-5 / 1.20e-5 | 7 s |
| 2048 | 64 (slow) | 2.7e-14 | 1.3e-6 | 1.28e-5 / 1.28e-5 | 26 s |

Known discrepancy (informational, not a torchsky defect): the healpy tables for nside >= 256 are high
by a smooth `~4e-6 (l/nside)^2` relative term plus irregular low-`l` errors (nside 256, `l = 1`: +4.0e-7,
confirmed with a healpy-only boundary-integral check of `w_1`). Rows are appended to
`benchmark_results/pixwin_healpy_deviation.jsonl`; the test bounds them at 2e-5 so a change in the
tables is noticed. Draft upstream report: [healpix-pixwin-upstream-issue.md](healpix-pixwin-upstream-issue.md).

P frame term: torchsky parallel-transports the polarization frame between the two points of each pair
(kernel `d^l_22`); the HEALPix exact tables differ by ~0.94e-6 (8/nside)^4 (2.4e-4 at nside 2, 1.4e-11
at 128), bounded in the gate by 1.2e-6 (8/nside)^4 + 1e-11. No independent P reference exists for
nside >= 256: a single-pixel frame differs from pairwise transport by `O(Omega^2)` (estimated ~1e-10 at nside 256),
above the 1e-11 target, so P there is covered only by the informational healpy comparison.

Runtime: the reference times above are on one core; torchsky computes the nside 1024/2048 moments once
(about 17 s / 60 s) and caches them under `~/.cache/torchsky/pixwin`. Non-slow cases take about 20 s in total.

STILTS modes and torchsky equivalents (`tests/test_stilts_parity.py`). Parity means equal multisets of matched row tuples
(`-1` = absent side); the only exclusion is a pair whose separation is within 1e-6 arcsec of its threshold, dropped from both sides together
with the tuples that involve its rows (bounded by `max(2, 1e-5 n)`).

| STILTS | torchsky | Status |
|---|---|---|
| `tmatch2 matcher=sky find=best1 / best2 / all` | `crossmatch_sky(find="best1" / "best2" / "all")` (`"best"` is a legacy alias of `best1`) | tuple-equal |
| `tmatch2 find=best` | `crossmatch_sky(find="mutual")` (exactly `best1 & best2`); `"symmetric"` is a greedy one-to-one assignment, deliberately not STILTS | tuple-equal, every join |
| `tmatch2 join=1and2 / 1or2 / all1 / all2 / 1not2 / 2not1` | `join_rows(result, n1, n2, join)` | tuple-equal for every find |
| `tskymatch2` | same as `matcher=sky` | tuple-equal |
| `tmatch2 matcher=skyerr` (separation <= err1 + err2) | `left_err_arcsec=`, `right_err_arcsec=` (per row, both sides) | tuple-equal, `find=all`, every join; real Gaia x 2MASS and 2MASS x AllWISE (errors 10x the catalogue values) |
| `tmatch2 matcher=exact` | `crossmatch_exact` | tuple-equal for `find=all` and joins; for `best*` only matched row counts (STILTS' choice among equal scores is unspecified, torchsky takes the lowest rows) |
| `tmatch1 matcher=sky action=identify` | `crossmatch_self(group=True)` | same partition and group sizes (ids not compared), full Gaia cone |
| `tmatchn multimode=pairs` | `crossmatch_many(mode="pairs", reference=0)`: each reference row takes its `best1` partner per catalogue (non-exclusive), then unreferenced rows alone | tuple-equal, synthetic and real Gaia x 2MASS x AllWISE |
| `tmatchn multimode=group` | `crossmatch_many(mode="group")` | same partition and group sizes (STILTS keeps an arbitrary member per catalogue, so representatives are not compared) |

Intentional differences, deliberately not tested against STILTS:

- `distance_tol` is a `|d1 - d2| <= tol` cut on top of the sky test; STILTS `sky3d` uses a Euclidean 3-d metric.
- `skyellipse` is not provided.
- Ranking (`best*`) always uses plain separation; STILTS `skyerr` ranks by an error-normalised score, so `find=best*` with errors is not compared.
- `matcher=sky3d`, `1d/2d/3d` matchers are not provided.

AstroDrizzle installability (checked with CPython 3.13, NumPy 2.5, macOS arm64, 2026-11):
`starlink-pyast 4.2.0` and `drizzlepac 3.11.0` install from wheels (and drizzlepac's `cdriz` runs on 3.13).

Deferred to 0.6/0.7 (global solvers): SCAMP-like global astrometric/photometric fits and solve-field pipelines.
Only the presence gates (`test_external_tools_required.py`) remain.

## Earlier coverage rows

| Domain | Comparator | Evidence | Status |
|---|---|---|---|
| WCS projections, SIP/TPV, real headers | astropy.wcs | `tests/test_wcs_*.py`, `bench_wcs*` | release gate |
| Coordinate frames | astropy.coordinates | `bench_coords_astropy` | release gate |
| HEALPix geometry, advanced ops | healpy, hpgeom, astropy-healpix | `tests/test_healpix_parity.py`, `bench_healpix*` | release gate |
| Spherical transforms (scalar, spin) | healpy, s2fft | `bench_sphere_spectral`, `bench_spin_matrix`, `tests/test_sphere_s2fft.py` | release gate (s2fft pinned `1.5.0` with `jax<0.8`; s2fft HEALPix needs `L >= 2*nside`, the previous test silently skipped) |
| MOC | mocpy | `tests/test_sphere_moc.py` | release gate |
| Polygons / exact regions | spherical_geometry | `tests/test_sphere_polygons.py`, `test_sphere_exact_regions.py` | release gate |
| Sparse maps | healsparse | `tests/test_healsparse_upstream.py` | release gate |
| Catalog crossmatch | astropy.coordinates | `tests/test_catalog_compat.py`, `bench_catalog_compat` | release gate |
| Resampling / coadd | reproject, drizzle, dfreproject | `tests/test_map_*_external_parity.py`, `bench_map_compat` | release gate |
| SWarp, Montage | SWarp binary, Montage `mProject` | `tests/test_map_swarp_optional_parity.py`, `test_montage_projection_parity.py`, `test_map_external_tools_smoke.py` | **required on the `improc` image** (`IMAGE_PROFILE=improc` sets `REQUIRE_MAP_COMPAT=1`, `REQUIRE_TOOLS=1`); optional on `astroai/base` (binaries absent there). Not verified locally (no binaries on the dev machine) |
| External binaries present and runnable | SWarp, SCAMP, SExtractor, PSFEx, solve-field, Montage, WCSTools; `ASTROMETRY_INDEX_DIR` | `tests/test_external_tools_required.py` | required on `improc` (`REQUIRE_TOOLS=1`) |
| WCS vs an independent C implementation | WCSTools `xy2sky`/`sky2xy` | `tests/test_wcstools_parity.py` | required on `improc`; TAN/SIN/ARC/ZEA/STG/CAR, 1 mas / 2e-3 px |
| ERFA / observed place | pyerfa | `tests/test_erfa_observed_place_parity.py` (`atco13`, `c2i06a`, `pnm06a`, `starpm`), `bench_observed_place` | release gate. Documents the Keplerian Earth-velocity approximation (max 0.56 arcsec vs ERFA, claim "about 1 arcsec") |
| GWCS freeze | gwcs | `tests/test_gwcs_live_parity.py` | release gate (`freeze_live_gwcs` vs live gwcs, sub-mas) |
| TLE / ephemeris / SkyCoord interop | sgp4, jplephem (+astropy, ERFA `epv00`), astropy.coordinates | `tests/test_wcs_optional_extras_parity.py` | release gate; DE440s kernel from `TORCHSKY_HARNESS_DE_KERNEL` or NAIF download. Small DE440s has planet-centre segments only for Mercury/Venus/Earth/Moon |
| Multi-order HEALPix interchange | mhealpy, cdshealpix | `tests/test_multiorder_interchange_parity.py` | release gate. One strict xfail: `from_mhealpy` always reports `values_are_densities=True` (torchsky bug) |
| Spectral rebin / Doppler | specutils | `tests/test_specutils_parity.py` | release gate (`FluxConservingResampler`, `shift_spectrum_to`) |
| CAR spherical harmonics | torch-harmonics | `tests/test_torch_harmonics_parity.py` | release gate (native backend vs the torch-harmonics adapter) |
| Cosmology (distances, H(z), dV/dz) | astropy.cosmology | `tests/test_cosmology_astropy_parity.py` | release gate. Ages are not in torchsky. One strict xfail records python-float precision loss |
| Extinction (CCM89, O94, F99, Calzetti) | dust_extinction, published Calzetti polynomial | `tests/test_extinction_parity.py` | release gate. G23 is not provided by torchsky |
| Synthetic photometry, filter profiles | speclite, synphot | `tests/test_spectrophotometry_parity.py` | release gate |
| Redshift, log-lambda | astropy.units, astropy.constants | `tests/test_redshift_units_parity.py` | release gate |
| Transforms | astropy.visualization, NumPy | `tests/test_transforms_correctness.py` | release gate; two strict xfails record subject deviations (ZScale upper limit, even-count median) |
| Field models / INRs | analytic truth, drizzle/coadd baselines | `bench_map_field_fit`, `tests/test_inr_*` | bounded gates; the SMACS campaigns are evidence, not gates |
| Encoders | none | `torchsky.encoders` | **experimental, not gated**: learned components with no established external reference; covered by bounded subject unit tests only |
| Scene model | none | `torchsky.scene` | **experimental, not gated**: no comparator implements the same contract; exercised through the map/field gates |
| Quality / mask contract | none | `torchsky.quality` | **experimental, not gated**: a torchsky-defined convention, not a comparator-defined one; subject unit tests only |
| Spectral cube readers | specutils | not covered | **not gated**: only rebin/Doppler are compared (above) |
| Astrometric calibration vs peers | SCAMP, solve-field (astrometry.net) | binaries only (`test_external_tools_required.py`) | **deferred to 0.6/0.7 (global solvers)**: presence gates only; `campaigns/phase0` manifests remain manifests |
| Deblending peers, TweakReg | Tractor, scarlet, tweakwcs/jwst | `campaigns/phase0` manifests | manifests only; torchsky makes no parity claim for these |
| SExtractor / PSFEx | source-extractor, psfex | binary presence only | torchsky makes no photometry/PSF parity claim; no comparison suite |
| astroquery SVO filter profiles | astroquery.svo_fps | `tests/test_svo_passband_bootstrap.py` (offline bootstrap) | live SVO query not gated (network) |
