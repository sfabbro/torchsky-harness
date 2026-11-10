# Union fit of all space data around SMACS J0723: design and scale (draft, 5 Nov 2026)

## What changes

Today's fit covers a 20.5″ box: about 10⁷ pixels, one GPU, about 4 h.

The union of space data (all JWST/HST footprints, plus Spitzer, GALEX and SPHEREx) covers about 8′ × 8′, about 580× the area:

- JWST cal: 722 detector images, about 3 × 10⁹ pixels.
- HST: about 120 exposures, about 1 × 10⁹ pixels.
- Spitzer IRAC: 3486 frames, about 2 × 10⁸ pixels.
- SPHEREx: 718 cutouts, about 7 × 10⁶ pixels.
- GALEX: 3 visits.

At 0.04″ with 12 spectral terms, one sky grid would need 1.7 × 10⁹ coefficients, plus the optimiser state. That does not fit in one GPU, and the data do not fit in GPU memory either.

## Principles kept

- No pre-registration and no resampling of the data.
- One sky s(α, δ, λ, t).
- Learned instruments shared across fields.
- Resolution only where the data support it.

## Proposed structure

### 1. Extraction to a pixel store (CPU, once per data release)

For every exposure, write its usable pixels as table rows:

- data, inverse variance, DQ;
- detector row and column;
- starting sky position from the header WCS;
- exposure, detector, band and epoch ids.

The rows are partitioned by sky tile, with a halo of one PSF window (largest for Spitzer and SPHEREx). The store holds Arrow/Parquet tables through torchfits. This is extraction, not resampling: pixels keep their detector identity, and the model still maps them through a learned transform.

### 2. Tiles

Use TAN tiles of about 30″ (750² px at 0.04″) inside HEALPix cells, giving roughly 250 tiles for the union. Each tile's sky resolution is set by the finest data covering it:

- 0.02–0.04″ under JWST/HST;
- about 0.3″ where only Spitzer is present;
- coarser where only SPHEREx or GALEX are present.

### 3. Split between global and local parameters

| Scope | Parameters | How they are fitted |
|---|---|---|
| Global, small | PSF networks per instrument; per-detector distortion, pixel response and background pattern; per-exposure transform, gain and background | gradients summed over tiles (data-parallel) |
| Local, large | sky coefficients of each tile | solved independently with the global parameters fixed |

The fit alternates: solve the tiles, update the global parameters, solve the tiles again. An exposure spanning many tiles sums its gradients over them. This is workstream D of the plan, and the same loop serves federation (tiles where the data live).

### 3b. Sky representation per tile

The candidates for a tile's sky are a pixel grid at the tile's resolution, a linear B-spline field, or elaborated Gaussian splats on top of a smooth grid.

Splats suit the union naturally:
- their density and widths follow the data, so resolution comes for free where JWST/HST resolve structure, and stays coarse under Spitzer or SPHEREx;
- they render per tile with a window, and a splat belongs to the tile that holds its centre, read through the halo by its neighbours;
- their amplitudes are linear, so they fit the CG/VarPro solve.

v32 implements them catalogue-free (densification from residual significance only). It is being compared with the 3D grid on SMACS (run splat2a).

### 4. Solver

Adam on 10⁸–10⁹ coefficients will not converge in reasonable time. The B-spline runs already showed it. The sky solve per tile needs the conjugate-gradient / VarPro solver of workstream C. This is now on the critical path.

### 5. Order of work

| Step | Work | Size |
|---|---|---|
| U0 | Pixel store for the union | CPU jobs on CANFAR, roughly 50 GB |
| U1 | Tile loop on 4–9 tiles around the cluster, global/local alternation, current Adam solver | proves the mechanics |
| U2 | CG/VarPro sky solve per tile | speed gate: 10× |
| U3 | All tiles, adaptive resolution per tile; add Spitzer, GALEX, SPHEREx operators (their PSFs and pixel responses learned like the others) | |
| U4 | Ground data and spectra later | |

## Open choices

- Tile size and halo, and whether tiles share a resolution ladder or not.
- Pixel store format and where it lives: arc (Parquet) or a database.
- How time enters for the union (several epochs from 2007 to 2026): a static sky first, with residuals flagged, or s(α, δ, λ, t) from the start.
