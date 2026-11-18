# Replacement and parity targets
The execution plan derived from this inventory is [0.4.0-completion-plan.md](0.4.0-completion-plan.md).


This is the working inventory of external packages whose user-facing
functionality torchsky (or its sibling `torchfits`) is intended to replace,
complement, or interoperate with. It separates production replacement targets
from optional adapters and from packages that are only references; a package
listed here is not automatically a release claim.

Status vocabulary:

- **gated** — a declared release claim compares this package on the listed
  behaviour.
- **covered** — a benchmark or acceptance task exists, but it is not in the
  release claim set.
- **unit-only** — torchsky has bounded package tests, no cross-package
  benchmark.
- **gap** — no meaningful cross-package comparison yet.
- **interop** — coexistence/adapter, not a replacement claim.
- **owner: torchfits** — FITS/table implementation lives in the sibling repo.

## 1. FITS and tables

| User-facing package | Current torchsky/torchfits surface | Status | Evidence |
|---|---|---|---|
| `astropy.io.fits` | `torchfits` HDU/image/table I/O | owner: torchfits | torchfits owns FITS/HDU/table parity and throughput |
| `fitsio` (python-fitsio) | `torchfits` | owner: torchfits | do not import in torchsky |
| CFITSIO | `torchfits` backend | owner: torchfits | package-level |
| `astropy.table` (FITS tables) | `torchfits.table`, `torchsky.catalogs` batches | partial | torchfits owns table codec; torchsky owns catalog semantics |
| pandas / polars / PyArrow / DuckDB | Arrow/tensor batches at boundaries | interop | catalog tests; tables are read by torchfits/pyarrow/polars and passed to torchsky as tensors; not replacements |

## 2. WCS, astrometry, coordinates, time

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `astropy.wcs` / WCSLIB | `torchsky.wcs.WCS` | **gated** | 38-projection release gate, real instrument headers, domain fuzz |
| PyAST (`starlink.Ast`) | `torchsky.wcs.WCS`, `convert_celestial`, `plan` | covered (legacy env) | `bench-wcs-legacy`/`bench-wcs-ecosystem`; not a release claim; env currently lacks harness |
| `gwcs` | `torchsky.wcs.sample_gwcs*`, `gwcs_live` | unit-only | build-time sampling/live adapter tests; no harness parity |
| `astropy.coordinates` / `astropy.time` / `astropy.units` | `torchsky.coords`, `torchsky.transforms`, `torchsky.wcs.frames` | **gated** (geometric frames) | `bench-coords-astropy-gate`: ICRS/Galactic/FK5/ecliptic <1e-6 asec; time/units remain unit-only |
| `pyerfa` / ERFA | `torchsky.wcs.iau2006`, observed frames | covered | `bench_observed_place` compares CIRS/TETE/AltAz against ERFA; not a release claim |
| SOFA | same IAU 2006/2000A path | unit-only | no direct SOFA binding comparison |
| `jplephem` | `torchsky.wcs.ephemeris` adapter | unit-only/interop | optional extra; no parity gate |
| `sgp4` | `torchsky.wcs.tle` adapter | unit-only/interop | optional extra; no parity gate |
| `skyfield` | none | gap | not a target; possible future coordinate/ephemeris comparison |
| `astrometry.net` (`solve-field`), SCAMP, `drizzlepac.tweakreg` | `torchsky.wcs.solve_astrometry`, `fit_sky_affine` (partial) | gap | scripts exist for calibration, no parity campaign |

## 3. Sphere, HEALPix, MOC, harmonics

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `healpy` | `torchsky.sphere.healpix`, `spectral`, `compat` | **gated** | index/angle/advanced/spin gates, real-sky map gate |
| `hpgeom` | `torchsky.sphere.healpix` queries | covered | `bench_sphere_geometry*`, ecosystem matrix; not release-gated |
| `astropy-healpix` | HEALPix core and WCS HPX | covered | ecosystem benchmark/upstream sync; not release-gated |
| `healpix` (C++/Python) | HEALPix core | covered | ecosystem benchmark; package missing in current sphere env |
| `cdshealpix` (Python/Rust) | `sphere.healpix`, MOC/STMOC | gap | mentioned in archived matrix; no harness comparator |
| `healsparse` | `torchsky.sphere.sparse` | covered | upstream fixtures/sync; no release gate |
| `mocpy` | `torchsky.sphere.MOC`, `BMOC`, `STMOC` | covered | `bench_sphere_core --moc-only`; not release-gated |
| `mhealpy` | multi-order HEALPix maps | gap | ecosystem benchmark entry; package missing |
| `spherical_geometry` | `torchsky.sphere.geom` | covered | polygon/area comparisons; not release-gated |
| `s2fft` | `torchsky.sphere.spectral` peer | covered (optional) | `sht-bench` extra; not currently installed/release-gated |
| `ducc0` / `ducc0.sht` | `torchsky.sphere.spectral` | gap | no comparator wired |
| `pixell`/`enmap` | WCS map + SHT | gap | no CAR-native SHT comparison |
| `NaMaster`/`pymaster` | pseudo-C_l, mode coupling | gap | not implemented as a target yet |
| `cuhpx` | CUDA SHT spike | covered (optional) | CANFAR `cuhpx` mode rc=0; not a release claim |

## 4. Mapmaking, reprojection, coaddition

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `reproject` | `torchsky.map.resample/coadd` | **gated** | real CFHT cutout gate |
| `drizzle` (LSST) | `ResampleConfig(kernel="drizzle*")` | **gated (bounded)** | `bench-map-drizzle-required`; tiny synthetic CRPIX translation only, not real-data replacement or rejection parity |
| `drizzlepac` / AstroDrizzle | drizzle/blot/outlier/context paths | diagnostic | Phase-0 compares one compact real WFC3/IR exposure with exact square drizzle on AstroDrizzle's output grid in identical `ELECTRONS/S` units. It reports shared-support science, flux, centroid, support, and finite-fraction metrics without certifying tolerances; raw weight, DQ/rejection, blot, background, and context parity remain unavailable. |
| `dfreproject` | DEIMOS-style reprojection | covered | `bench_map_compat`/bench env; not release-gated |
| Montage (`MontagePy`, `mProject`, `mFitplane`) | exact spherical overlap + sky matching | covered | `bench-map-montage-required`; `MontagePy` missing locally |
| SWarp | resampling/coadd/background/`.head` | covered | `bench-map-swarp-required`, clipped variant; not release-gated |
| casutools/imcore | aperture/resample compatible operations | gap | bootstrap script only |
| `stcal` | variance/outlier/cube build | gap | JWST pipeline not compared |
| `jwst.cube_build` | IFU cubes | gap | useful real-data comparator |
| T-PHOT / SCARLET / Tractor | forward/deblended photometry | gap | planned surfaces |
| MaxiMask / WeightWatcher | defect/weight masks | gap | `torchsky.quality` target, no comparator |

## 5. Detection, photometry, PSF, calibration

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `photutils` | `torchsky.catalogs`, `torchsky.map.psf` (partial) | gap | used in SMACS scripts, no parity gate |
| SExtractor / SEP | source detection, catalogs | gap | scripts only; no numerical comparison |
| PSFEx | PSF modelling | gap | no comparator |
| `stpsf` / WebbPSF | JWST PSF | gap | SMACS feature; no parity gate |
| `poppy` | optical PSF | gap | no comparator |
| SCAMP / `tweakreg` | astrometric bundle | gap | calibration scripts, no parity campaign |

## 6. Spectral, photometric, cosmology, extinction

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `astropy.cosmology` | `torchsky.cosmology`, `redshift` | unit-only | package tests; no harness comparator |
| `astropy.units` spectral equivalencies | `torchsky.spectral`, `spectrophotometry` | partial | unit tests; no astropy equivalency parity |
| `synphot` / `pysynphot` | filter integration, bandpasses | unit-only | no comparator |
| `speclite` | filter response integration | unit-only | no comparator |
| `specutils` | spectral axes/line analysis | unit-only | no comparator |
| `astroquery` SVO FPS | filter profile lookup | unit-only/interop | fake/offline tests; no live parity |
| `dust_extinction` | `torchsky.extinction` | unit-only | package tests; no harness parity |
| `dustmaps` | extinction maps | gap | no wrapper yet |
| `sncosmo` / `sedpy` / `prospect` | SED/spectral modelling | gap | not targets yet |

## 7. Catalogs and cross-matching

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `astropy.table`/`SkyCoord` matching | `torchsky.catalogs.crossmatch` | **gated** | catalog compatibility gate vs SkyCoord |
| `scipy.spatial.cKDTree` | spatial index | reference | used in tests/benchmarks |
| `xmatch` (CDS) | crossmatch engine | gap | scorecard only; package not installed |
| `duckdb`/`polars`/PyArrow/pandas | table ingestion | interop | catalog I/O tests; not replacements |

## 8. Field models, rendering, visualization

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `tiny-cuda-nn` / `tinycudann` | hash-grid encoders | gap (research) | no direct comparator; field models validated on truth |
| `nerfstudio`, `torch-ngp`, `gsplat` | neural/radiance fields | gap (research) | not applicable to survey maps yet |
| `scipy.interpolate` / `scipy.ndimage` | continuous field/interp baselines | unit-only | no harness comparator |
| `matplotlib` | visualization | interop | examples/site only |
| `astropy.visualization`/WCSAxes | WCS plotting | interop | torchsky.viz is optional |
| Aladin Lite / JS9 / skyview | browser FITS viewer | interop | torchsky.viz serves Aladin Lite |

## 9. Ownership note for FITS

Torchsky's compatibility policy forbids `astropy.io.fits`, `fitsio`, and
`astropy.Table.write(..., format="fits")` in production paths. The replacement
for those packages is **torchfits**, evaluated in the torchfits repository; it
is listed here only so the boundary is explicit.

## Current release evidence (2026-10-21)

Release-gated comparisons: astropy WCS/WCSLIB, healpy (core/advanced/spin/real
map), astropy SkyCoord catalog matching, reproject real cutout, real instrument
WCS headers. Contract/fuzz gates cover structural/domain behaviour.

Non-release coverage exists for hpgeom, astropy-healpix, healsparse, mocpy,
spherical_geometry, drizzle, dfreproject, SWarp, Montage, s2fft, cuhpx, PyAST,
ERFA, and pandas/polars/Arrow. Everything marked **gap/unit-only** is
the next campaign list.

## Functional coverage against sections 2–4 (2026-10-21)

This is a capability assessment, not a test-suite count. “Core” means the
compute path exists and is exercised; “full replacement” means a user could drop
the peer package from a representative workflow without losing behaviour.

### 2. WCS, astrometry, coordinates, time

| Capability | Status | Missing for full replacement |
|---|---|---|
| WCS Paper II celestial projections, forward/inverse, CPU/CUDA/MPS | **core, gated** | all 30 codes covered; no known projection gap |
| WCS Paper III spectral axes, `-TAB`, air/vacuum, rest frequency | **core** | gate covers spectral/temporal rows, not every WCSLIB alias |
| WCS Paper IV distortions (SIP/TPV/TNX/ZPX/DSS/CPDIS/DET2IM) | **core** | lookup fixtures are partially skipped; CPDIS/DET2IM full parity not certified |
| Tabular `-TAB` separable and coupled | **core** | coupled CUDA path now host-parity; broad randomized fuzz still missing |
| Sky affine fitting / alignment | **partial** | no global SCAMP/tweakreg-style bundle solver; no Astrometry.net replacement |
| `gwcs` pipeline | **partial/interop** | sampling/frozen tables and live freeze adapter, but no general GWCS transform graph or arbitrary compound mappings |
| Astropy/ERFA geometric frames | **core, gated** | ICRS/Galactic/FK5/ecliptic now astropy-exact; full Astropy frame graph and units/time semantics are not cloned |
| Observed place, DCR, aberration, ephemeris tracking | **core** | ERFA CIRS/TETE/AltAz parity benchmark exists but is not a release claim; jplephem/sgp4 adapters are optional and untested against live kernels |
| PyAST arbitrary FrameSet/Mapping, AST plotting | **partial** | compound mappings and `ast.Plot` are not implemented |
| WCSAxes/visualisation | **not replaced** | `torchsky.viz` is optional; no WCS plotting parity |

Bottom line: **the WCS transform engine is close to a full replacement; the
astrometric bundle solver and general GWCS/PyAST object model are not.**

### 3. Sphere, HEALPix, MOC, harmonics

| Capability | Status | Missing for full replacement |
|---|---|---|
| HEALPix indexing, neighbours, interpolation, `ud_grade`, ring/nest | **core, gated** | healpy class wrappers (`Alm`, `Rotator`) and a few aliases (`max_pixrad`, `isnsideok`, `isnpixok`, `pix2ring`, `pix2xyf`, `xyf2pix`) are missing |
| Scalar/spin SHT, `anafast`, `synfast`, `alm2cl`, beam/pixwin, smoothing, rotate | **core, gated** | no `Alm`/`Rotator` object API; s2fft/ducc0 are not interchangeable backends |
| Region queries | **core** | hpgeom vector-query variants (`query_*_vec`, `angle_to_pixel`, `pixel_to_angle`) are not exposed under those names |
| Sparse maps / WideBitMask / SkyMaskPipe | **core** | `HealSparseCoverage`, `WIDE_MASK` constant, and complete `.hs` cross-open parity |
| MOC / BMOC / STMOC | **core** | no `FrequencyMOC`, `SFMOC`, or `TimeMOC` equivalents; STMOC covers space-time |
| Exact spherical geometry/regions | **core** | no full `spherical_geometry` plotting/convex-hull API; boolean-region coverage is partial |
| Multi-order maps (`mhealpy`) | **gap** | no multi-order map type |
| `cdshealpix` range algebra | **partial** | range/uniq helpers exist; bit-interleaving (`spread_bits`/`compact_bits`) is not public |
| `pixell`/`enmap`, `NaMaster`, `ducc0`, `s2fft` | **gap/optional** | no CAR-native SHT, mode-coupling, or interchangeable SHT backend |

Bottom line: **the HEALPix geometry/harmonic core and main sparse/MOC surfaces
are largely covered; the compatibility-class API, multi-order maps, and CMB
post-processing layers are not.**

### 4. Mapmaking, reprojection, coaddition

| Capability | Status | Missing for full replacement |
|---|---|---|
| Reprojection kernels (nearest/bilinear/bicubic/Lanczos/Mitchell/FFT/polygon) | **core, gated for reproject real cutout** | adaptive reprojection and every reproject kernel are not separately certified |
| Drizzle kernels, `pixfrac`, blot, context bitplane, outlier rejection | **core** | no full AstroDrizzle/stcal DQ/CR parameter matrix or external comparator certification |
| SWarp kernels, combine modes, sky mesh, `.head` sidecars | **core** | pinned-source comparator could not be built locally; acceptance task not passed on this host |
| Montage exact spherical overlap/planar sky matching | **core** | `MontagePy` is x86_64-only here; exact comparator not certified locally |
| Multi-band/cube/HEALPix coadds, variance propagation, tiling | **core** | stcal/JWST full variance contracts and real IFU cube-build comparison remain open |
| casutools/imcore compatibility | **gap** | no comparator implementation |
| T-PHOT/SCARLET/Tractor deblending/forward photometry | **gap** | these are catalog/forward-model surfaces, not yet implemented |
| MaxiMask/WeightWatcher defect masks | **gap** | `torchsky.quality` is a target, no comparator |
| Source detection/photometry/PSF (SExtractor/SEP/photutils/PSFEx/stpsf) | **gap** | adjacent to mapmaking but required for an end-to-end replacement |

Bottom line: **the resampling/coadd engine is close to a full replacement for
reproject/drizzle/SWarp/Montage core paths; end-to-end mapmaking (calibration,
detection, photometry, PSF, deblending) is not.** Those gaps are sections 5–8 of
the inventory, not the map engine itself.
