# Ecosystem compatibility

torchsky is **torch-native first**: primitives, kernels, and public tensor paths are designed for performance, differentiability, and scale. On top of that, the project maintains **compatibility surfaces** so workflows built on common astronomy stacks can migrate incrementally and be checked with **parity tests** against trusted libraries.

The goal is **not** to match every internal detail of upstream packages. The intent is to match **behavior users rely on** at useful API boundaries (see `AGENTS.md`).

## World-layer replacement matrix

A **job** is claimed only when all four exist: a native implementation with no
runtime import of the peer, oracle tests or packaged fixtures, a throughput row
versus the peer on CPU (and CUDA/MPS only where the device is claimed), and
docs that fail loud on gaps. Linux leadership is not claimed from Mac numbers.
Thin shims (`torchsky.sphere.compat`, `torchsky.wcs.WCS`) stay for migration;
this is not `import healpy` and not a public-symbol clone.

| Peer | Jobs | torchsky entry | Oracle | Status |
|---|---|---|---|---|
| healpy | Pixel API; scalar/spin SHT; synfast/anafast; `gal_cut`; RING quadrature weights | `sphere.healpix`, `sphere.harmonics`, `sphere.compat` | healpy/ducc | **partial** — fused Ishioka `map2alm_spin` leads healpy on Mac CPU; `alm2map_spin` is ~1.6× healpy at nside=128 on that machine. Polar CPU RING FFTs are in-process pocketfft (Accelerate radix-2 on Apple for large pow2); equatorial CPU rings use the same in-process FFT. Linux CANFAR SHA `d8d706f6` `sht-profile` nside=128 / 8 CPU: `map2alm_spin` 0.41× healpy (10.9 ms vs 4.5 ms), `alm2map_spin` 0.32× (10.7 ms vs 3.3 ms). Synthesize ~6 ms vs interpolate ~2.1 ms. Do not claim Linux SHT. Healpy stays a runtime peer for SHT speed. `use_weights` is computed RING quadrature. `use_pixel_weights` loads `full_weights/healpix_full_weights_nside_NNNN.fits` through torchfits (`datapath` / `TORCHSKY_HEALPIX_DATAPATH`); files are not vendored or downloaded (`scripts/fetch_healpix_pixel_weights.sh` can populate that datapath). CUDA/MPS SHT stays on device via the ATen ring FFT path; fused Ishioka/Wigner C++ stays CPU. `rotate_alm` is healpy zyz Euler, returns a new tensor, and stays on device (ATen). |
| hpgeom | `query_disc` / box / ellipse / strip | `sphere.healpix` | hpgeom | **claimed** for query semantics. Inclusive `fact==4` pads by max pixel radius; `fact!=4` supersamples at `nside*fact` then parent-degrades. |
| astropy-healpix | Same pixel jobs as healpy/hpgeom | `sphere.healpix` | astropy-healpix | Oracle only; no third API. |
| healsparse | construct/update, `get_values`/`valid_pixels`, coverage, `ud_grade`, FITS `.hs`, union/intersection sums | `sphere.HealSparseMap` | healsparse | **partial** — jobs implemented and valid-pixel values match (`maxabs 0.0`). Fused CPU `get_values` matches healsparse on Mac at nside=256 / 50k nest queries (~0.20 ms median interleaved, ratio ~1.03). Linux CANFAR SHA `d8d706f6` interleaved: `get_values` 0.89× healsparse at nside=256 / 200k and 0.92× at nside=512 / 500k (**not claimed**). Linux interleaved `ud_grade` is 2.86× / 4.63× torchsky/healsparse (**claimed** on that stamp). 1-D value maps write `PIXTYPE=HEALSPARSE` COV+SPARSE images (coverage nside from `nside_coverage` or `nside//16` for pixel-list maps); `tests/test_healsparse_upstream.py` cross-opens with `healsparse.HealSparseMap.read` / `write`. Extra value dimensions stay PIXEL/VALUE tables. |
| mocpy | cone/polygon, contains, union/area, FITS/JSON/ASCII, empty/full, wrap | `sphere.MOC`, `sphere.STMOC` | mocpy | **partial** — spatial MOC jobs as before. STMOC is native (time order 61 µs since JD=0 × spatial `MOC`); RANGE FITS and `t<depth>/<c0>-<c1>` ASCII roundtrip are torchsky-owned. This is not a mocpy STMOC API clone. Linux CANFAR SHA `d8d706f6` order-8 contains is 0.98× mocpy at 50k points; cone is 0.061× and union is 8.7e-4× (not claimed). |
| spherical-geometry | contains, intersect, union, area | `sphere.geom` | spherical_geometry | **claimed** for contains/area. Exact backend vs spherical-geometry area atol `1e-12` steradians (`tests/test_sphere_exact_regions.py`). Approximate contains/intersect/`query_polygon_general` run on CUDA/MPS via ATen; exact `NativeExactSphericalRegion` stays CPU. Linux CUDA throughput exists as torchsky-only (`world-cuda` `geom_cuda.json`, N=2e5 contains 1.03e8 pts/s); cuda-bench had no spherical-geometry, so Linux/CUDA is not claimed vs the peer. |
| astropy.wcs | FITS pixel↔world, SIP/TPV/TNX/ZPX, spectral/temporal, slicing | `wcs.WCS` | astropy.wcs | **partial** — Linux CANFAR SHA `9e65cbc` large-N CPU 87/90 first. SHA `775762c2` production (`astroai/notebook:latest`, 8 CPU): Paper II N=1e5 mixed vs astropy 8.0.1 is first for PCO 1.91× / CSC 1.48× / NCP 3.63× and behind for XPH 0.92× / QSC 0.73× / TSC 0.53×. TPV forward N=2e5 is 0.92× astropy on a rank-deficient header (`PV2_2=1`, no `PV2_1`); identity mixed TPV is `PV1_1=1` and `PV2_1=1`. Dedicated TPV inverse N=1e6 tensor_core leads (affine 9.3 ms vs 3.43 s). Coupled `-TAB` inverse is accurate (5e-11 px) and slow (27 s CPU / 165 s CUDA at N=10k). Spectral vs WCSLIB 26/55 first at N=1e5. CUDA Paper II/TPV first vs CPU astropy (`world-cuda` on SHA `775762c2` pytest 15/15 `-k cuda`). SHA `d8d706f6` stamp `20260821-1744*`: Paper II N=1e5 CPU is first for PCO 1.89× / NCP 2.55× and behind for XPH 0.88× / CSC 0.61× / QSC 0.64× / TSC 0.53× (per-code, never 6/6). CSC matches WCSLIB to double rounding on CPU (about 2.8e-14° at torchsky `87f8c3ce`; `test_csc_cpu_matches_wcslib`) and CUDA/MPS use the same polynomial rounding, so the earlier mas-class CSC discrepancy no longer applies. CUDA Paper II vs CPU astropy is first (CSC 3.9×, PCO 32×). Coupled `-TAB` inverse N=10k is 53 ms CPU / 9.1 ms CUDA at 5e-11 px (the old 27 s / 165 s Newton leftover is gone). Spectral vs WCSLIB 13/55 first; linear FREQ 0.40×; `FREQ-GRA` leftover 0.625 Hz. `WCS.to("mps")` is float32, arcsecond-class vs CPU float64. Fused C++ WCS is CPU; CUDA/MPS use ATen. |
| gwcs | Calibration-time detector→sky callable | `wcs.sample_gwcs_detector_to_tangent`, `FrozenDetectorEpochTables.evaluate` / `.inverse`; optional `wcs.gwcs_live.freeze_live_gwcs` | gwcs at sample time only | **partial** — core runtime never imports `gwcs`. Frozen tables stay the production path. Live graphs are the `gwcs` extra, then freeze. |
| astropy.coordinates / ERFA | Geometric frames, space motion | `wcs.CatalogState`, `wcs.frames`, `wcs.transform` | Astropy/ERFA | **partial** — not a `SkyCoord` clone. `wcs.transform(lon, lat, from_frame=..., to_frame=...)` is a tensor façade. Observed AltAz needs explicit IERS DUT1 (`wcs.observed`). `pn='iau2006'` is Fukushima-Williams plus full IAU 2000A nutation with IAU 2006/P03 adjustments (SOFA `nut06a` / `c2i06a` / `pnm06a`; requires `tt_minus_utc_s`; leaps are never inferred). `GCRS`/`CIRS`/`TETE` share that obstime and EOP. Aberration is Keplerian Earth; refraction is Saemundsson/Meeus and opt-in. Linux CANFAR SHA `775762c2` N=4096 pinned EOP: CIRS 2.13e-9″ / TETE 5.36e-9″ vs ERFA (CPU and CUDA). AltAz is throughput only. CUDA/MPS stay on device via ATen (CUDA float64, MPS float32). There is no observed-place C++ kernel. Ecliptic-of-date, Galactocentric (explicit solar params), and LSR (explicit UVW) are tensor-native. `get_body` / TLE are extras (`jplephem` / `sgp4`) and fail closed without the kernel or library. No `SkyCoord` class. |

Out of this program: ducc0 internals, lsst.sphgeom, wcslib/pyast (optional WCS benches only), reproject/SWarp/drizzle, astropy.io.fits, map stackers.

## Reference libraries (non-exhaustive)

| Area | Typical references | torchsky entry / notes |
|------|-------------------|-------------------------|
| HEALPix pixel + harmonic API | **healpy**, **hpgeom** (strict queries), **astropy-healpix** | `torchsky.sphere.healpix`, `torchsky.sphere.compat`; spectral transforms are torch-native—compare to healpy in tests/benchmarks only |
| WCS / projections | **Astropy** `astropy.wcs`, FITS WCS | `torchsky.wcs.WCS` (pure-torch path); parity tests vs Astropy where scoped |
| FITS I/O & tables | **torchfits** | `torchfits` only; torchsky has no I/O or table namespace; never `astropy.io.fits` or Python `fitsio` |
| Spherical geometry / polygons | **spherical_geometry**, **astropy-healpix** (where applicable) | `torchsky.sphere` and `torchsky.sphere.geom`; exact regions and benchmarks vs upstream fixtures |
| Catalogs / coordinates | **Astropy** `SkyCoord` / ERFA | `torchsky.wcs.CatalogState` and `torchsky.wcs.frames`; see [coordinates.md](https://github.com/astroai/torchsky/blob/main/docs/coordinates.md). Not a SkyCoord clone. |

## Where compatibility lives

- **HEALPix native pixel API** — `torchsky.sphere.healpix` owns indexing, neighbors, interpolation, range helpers, and query primitives.
- **HEALPix / healpy-shaped API** — [`torchsky.sphere.compat`](https://github.com/astroai/torchsky/blob/main/docs/sphere.md#compatibility); use `import torchsky.sphere.compat as hp` when porting healpy-shaped code.
- **Sphere native + extended geometry** — `torchsky.sphere` (regions, MOC, sparse maps, etc.), often compared to healpy/hpgeom/spherical_geometry in tests and benchmarks.
- **WCS** — `torchsky.wcs` (see tests and WCS-related benchmarks).
- **FITS / tables** — `torchfits`; torchsky accepts decoded tensors, headers,
  masks/weights, and table batches (see `io.md` and `table.md`).
- **Mapmaking** — `torchsky.map` for native map products, with external stackers used only as parity and benchmark comparators (see `map.md` and `benchmarks.md`).

## Upstream Validation Matrix

“Upstream” means reference libraries, fixture families, and public data surfaces
used to validate behavior. It does not mean torchsky depends on those packages
at runtime.

### Astropy

| Upstream surface | Source | Local coverage | Contract |
|---|---|---|---|
| WCS core fixture-heavy tests | `astropy/wcs/tests/test_wcs.py` | `tests/test_wcs_parity.py`, `tests/test_wcs_allsky.py`, `tests/test_wcs_tpv.py`, `tests/test_wcs_legacy.py`, `tests/test_wcs_zpx.py`, `benchmarks/bench_wcs.py`, `benchmarks/bench_wcs_suite.py` | parity-test, benchmark-hook |
| FITS docs/examples | `astropy.io.fits` docs | Owned and validated by `torchfits`; torchsky retains only a cross-package compute smoke | integration |

### Healpy

| Upstream surface | Source | Local coverage | Contract |
|---|---|---|---|
| HEALPix primitive tests/tutorial flows | healpy tests/tutorials | `tests/test_healpix_parity.py`, `benchmarks/bench_healpix.py`, `benchmarks/bench_healpix_advanced.py`, `benchmarks/bench_spin_matrix.py` | parity-test, benchmark-hook |

### HealSparse

| Upstream surface | Source | Local coverage | Contract |
|---|---|---|---|
| I/O lifecycle | `tests/test_io.py` | `tests/test_healsparse_upstream.py`, `benchmarks/bench_sphere_sparse.py` | adapter-test, benchmark-hook |
| Degrade semantics | `tests/test_degrade.py` | `tests/test_healsparse_upstream.py`, `benchmarks/bench_sphere_sparse.py` | adapter-test, benchmark-hook |
| Geometry and map operations | `tests/test_geom.py`, `tests/test_operations.py` | `tests/test_healsparse_upstream.py`, `benchmarks/bench_sphere_sparse.py` | adapter-test |
| Quickstart/basic interface | `basic_interface.html` | `tests/test_healsparse_upstream.py` | example-smoke |

### Fitsio / Astropy

FITS I/O parity, compression, checksum, HDU, table, and throughput matrices
live in `torchfits`. Duplicating those matrices in torchsky would blur ownership
and waste CI. Torchsky validates only that decoded data and metadata cross the
boundary without changing WCS/map/catalog semantics. The supported
cross-package baseline is `torchfits 1.0.0` or newer.

### Real-Data Reference Fixtures

| Real-data surface | Source | Local coverage | Contract |
|---|---|---|---|
| Installed scaled/VLA/WCS FITS fixtures | `astropy` package test data | `tests/test_real_data_validation.py` | adapter-test |
| Installed HEALPix weight table | `healpy` package data | `tests/test_real_data_validation.py` | adapter-test |
| Public sample/reference FITS image | Astropy data server tutorial image | `tests/test_release_smoke.py` when network/fixture access is available | example-smoke |

If upstream replay scripts return, add their source inventory and restore the
corresponding docs/tests in the same PR. `healsparse` remains adapter-based
rather than exact API replay because torchsky intentionally exposes sparse-map
functionality through its own spherical namespace.

## Optional dependencies

Many parity and benchmark paths import **healpy**, **Astropy**, **hpgeom**, or others **only when installed**. They are used to **validate** torchsky behavior, not as the default implementation of core tensor APIs.

## Adding new compatibility

1. Prefer implementing behavior in **torch-native** primitives, then expose a **thin** compatibility wrapper with the upstream name/signature.
2. Add **tests** against the reference library or packaged fixtures when available.
3. Document the mapping here or in the domain doc (`sphere.md`, `api.md`, etc. in the torchsky package docs).
