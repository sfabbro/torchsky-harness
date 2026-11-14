# Item sky for fieldop (`--sky-form items`), 13 Nov 2026

**Verdict.** On the small synthetic test, the item sky beats the grid sky on every seed and run length tried. The items use 4.2 times fewer fitted sky numbers, and the gap widens with fit length. The caveat is that every run is short: held-out $\chi^2$ is still 1.5 to 7, against a noise floor of 0.99. Two things remain open. Whether the advantage holds at convergence is not established. Whether it holds for real star cores at 0.04" is in doubt (see "Blur gauge").

## What was built
- **Model.** The sky is
$s(X,Y,\lambda)=\sum_n\big[\sum_k c_{nk}\phi_k(\lambda)\big]\,G(X-x_n,Y-y_n;\Sigma_n)$, plus an optional coarse smooth grid.
  - Each item has a continuous position, two widths $w_i=\sqrt{f^2+e^{2l_i}}\ge f$ and an angle.
  - Its coefficients are $c_{nk}=e^{r_{nk}}>0$, on the package's 12-term spectral basis.
  - That is $5+K=17$ fitted numbers per item.
  - Default floor $f$ = `--item-floor` = $0.4\times$ sky pitch: 0.016" at 0.04", 0.008" at 0.02". Two reasons for this choice: below about $0.4p$, a rasterised item can no longer be placed between nodes (centroid error 0.025 pitch at the floor, measured); and the sharpest kernel core, at about 0.012" for F090W, is below what the tent can carry anyway.
- **Minimal integration.** `ItemSky.coeffs(view)` rasterises the items onto the same $K\times N\times N$ grid at `--sky-pitch`. The tent read, pooled levels, FFT and quadrature renders, band images, bright-patch strata and prediction are untouched. **The sky is therefore still read through a tent grid at the sky pitch.** What is tested is the parametrisation: continuous positions and sizes, positivity, fewer numbers. It is not a grid-free renderer.
- **Code changes.**
  - `sky.py`: new class `ItemSky`; `SkyGrid` gains `n_fitted`, `reset_from_coeffs` and `average_views`, which run the same operations as before.
  - `train.py`: `sky_optimizer`, a `sky_regs` hook, the relocation hook, and the two places that wrote `model.sky.A`.
  - `tiles.py`: one optimiser per tile sky.
  - `cli.py`: the flags, listed under "Recommended GPU arguments".
- **Optimiser.** Adam, with separate steps for positions (0.05 pitch), log widths (0.03), angles (0.05 rad) and log coefficients (0.05). The smooth grid uses the grid sky's step. All follow the run's schedule.
- **Relocation.** Optional and off by default. It is cheap: the back-projected residual is minus the gradient of the data loss with respect to the coefficient grid, captured with a hook and averaged over steps.
- **Several tiles.** One item sky per tile works with sky swapping. Saving and loading the state round-trip exactly (`tests/test_items_tiles.py`).

## Approximations and shortcuts (also F38 and F39 in `fieldop/__init__.py`)
1. **Cell integral.** The cell integral is approximated by adding $p^2/12$ to both principal variances (the box replaced by an isotropic Gaussian), then point-sampling. Each item is evaluated on a patch of ±4 effective widths, renormalised to exact flux. Nodes outside the grid are dropped, because the sky is zero there.
2. **Seeding.** Data-blind lattices. Level $l$ gets a share of the items proportional to $4^{-l}$: 76, 19 and 5 per cent for 3 levels. Spacing is $2h/m$, width is half the spacing, and the angles are random. The finest level's remainder goes on a Halton set. Coefficients are the level share × the item area × the warm-start coadd coefficients smoothed at the item width. They are floored at $10^{-3}$ × their median peak, because $c=e^r$ cannot be zero. The seeds are rebuilt when the registration pre-pass rebuilds the warm start.
3. **Untuned settings.** The item step sizes and counts were taken from the image-only experiment and not tuned here; the grid's settings were not tuned either.
4. **Width cap.** Widths are capped at `--item-max-width` (default $16p$). Items may drift outside the sky domain; nothing stops them.
5. **Positivity penalty.** `--sky-neg` acts only on the smooth grid, since the items are positive by construction. TV is not applied to items.
6. **Cross-fitting.** Views are averaged parameter by parameter. This is not the average of the two skies. It is implemented but untested.
7. **Relocation details.** Items whose peak band brightness is below 0.5 band noise are moved, at most 5 per cent of them, at 15, 30, 45 and 60 per cent of the run, and only on the active tile.
8. **Cache.** The rasterised grid is cached while no sky parameter needs a gradient. The cache key includes content checksums, because writes through `.data` do not bump PyTorch's version counter.

## Tests (CPU, one thread, shared machine)
**1. Default path unchanged.** The untouched copy is kept in `untouched/fieldop`. I compared it against the new package with `--sky-form grid` in two runs:
- single box, 40 steps;
- two tiles with sky swapping, 12 steps.

In both, the full final state checksum (operator, nuisances, sky) is identical, and so are every logged loss and the held-out $\chi^2$ (6.9017).

**2. Item sky unit tests** (`tests/test_items_unit.py`).
- Gradients against central finite differences (float64): maximum relative error is $4\times10^{-10}$ for positions, $1\times10^{-9}$ for log widths, $1\times10^{-9}$ for angles, $4\times10^{-10}$ for log coefficients and $5\times10^{-9}$ for the smooth grid.
- Known sky: six known Gaussians, three of them point-like at the floor and placed between nodes, fitted for 600 steps. The relative L2 error goes from 0.95 at the seed to 0.027. Flux within 3 pitches of each point-like Gaussian is recovered to 0.99 to 1.05. The remaining error is about the size of the cell-integration error at the floor.
- Flux and placement of one item: the table below. Each value is the worst case over sub-node offsets of 0, ¼ and ½ pitch and over angles of 0 and 30 degrees.

| width / floor | 1 | 1.25 | 1.5 | 2 | 4 | 10 |
|---|---|---|---|---|---|---|
| flux error after renormalisation | <1e-15 | <1e-15 | <1e-15 | <1e-15 | <1e-15 | <1e-15 |
| flux error before renormalisation | 3.3e-2 | 5.6e-3 | 6.3e-4 | 2.5e-6 | 0 | 7e-12 |
| worst node vs exact cell mean, per cent of peak (angle 0) | 2.2 | 1.9 | 1.2 | 0.45 | 0.03 | 0.001 |
| centroid error of the grid, in pitches (same for the tent read) | 0.025 | 0.006 | 9e-4 | 6e-6 | 2e-6 | 4e-6 |

**3. Synthetic recovery** (`tests/synth_small.py`). This is the reference synthetic field made smaller, to respect the 4-minute rule:
- 8 exposures per filter (6 train, 1 validation, 1 held out);
- kernel extent 1.2";
- 4 patches per step;
- variability scales fixed;
- sky pitch 0.06";
- fixed step counts, and everything else identical between the two forms.

What the columns measure:
- **Held-out $\chi^2$**: the package's own score on the 3 held-out exposures (4356 pixels). Only per-exposure nuisances are refitted. "S/N > 5" means the model's bright pixels, about 430. Noise floor 0.99.
- **Sky error**: relative rms error of the fitted sky's band images against the true sky. Both are smoothed by a Gaussian of 0.06", and the truth is shifted by the common-shift gauge. The error is split into nodes within 0.15" of a true point source and the rest.

  It is reported after the best common relative blur. Without that adjustment, every run is at 0.46 to 0.54, because every fitted sky is 0.07" to 0.09" blurrier than the truth (the kernels are correspondingly sharper). That is the common-blur gauge, which is free above the floor.

| run | steps | fitted sky numbers | held-out $\chi^2$ | held-out $\chi^2$, S/N > 5 | sky error near point sources | sky error elsewhere |
|---|---|---|---|---|---|---|
| seed 0, grid | 100 | 108,300 | 4.51 | 35.6 | 0.193 | 0.187 |
| seed 0, items | 100 | 25,500 (1500 items) | **4.04** | **31.5** | **0.177** | **0.159** |
| seed 1, grid | 100 | 108,300 | 5.17 | 39.0 | 0.177 | 0.191 |
| seed 1, items | 100 | 25,500 | **2.31** | **12.9** | **0.176** | **0.159** |
| seed 0, grid (longer run, a stated exception to the 4-minute rule) | 250 | 108,300 | 2.52 | 15.3 | 0.210 | 0.202 |
| seed 0, items (same) | 250 | 25,500 | **1.52** | **5.99** | **0.156** | **0.158** |
| seed 0, grid, sky pitch 0.10" | 100 | 38,988 | 7.04 | 61.0 | 0.138 | 0.191 |
| seed 0, items, sky pitch 0.10" | 100 | 25,500 | **4.15** | **31.8** | **0.132** | **0.136** |
| seed 0, items with relocation | 100 | 25,500 | 3.92 | 30.1 | 0.178 | 0.157 |

**4. What looks wrong or weak.**
- **Pre-convergence regime.** Every run is far from convergence. The batch training $\chi^2$ was still falling, and the grid's was *lower* than the items' at 100 steps while its held-out score was worse. That pattern suggests the grid fits more noise, but this is not proven.
- **Grid sky error.** Between 100 and 250 steps on seed 0, the grid's sky error rose from 0.19 to 0.21 while its held-out $\chi^2$ improved.
- **Item warm start.** The items' seeded sky is blurrier than the grid's start: first-step loss 7.0 against 3.5.
- **Cost.** Items took about 15 to 35 per cent more time per step on CPU. One rasterisation pass (forward and backward) costs 0.08 s. The measured gap is noisy because the machine was shared.
- **Relocation** changed nothing that matters, the same result as the image-only experiment.

## Blur gauge and the item floor
- **Sharpest sky shapes.** The kernel floor (`--blur-floor`, unchanged) is one tent: $m_2=p^2/3$. A rasterised item read through the tent has $m_2\ge 2f^2+p^2/6+p^2/3=0.82\,p^2$. One grid node has $p^2/3$.
- **Sharpest model star.** With the kernel at its floor, the sharpest star the model can draw is $1.15\,p^2$ with items, against $0.67\,p^2$ with the grid.
- **What it means at 0.04".** Both are blunter than an estimated F090W core ($\sim5\times10^{-4}$ arcsec², against $1.8\times10^{-3}$ for items and $1.1\times10^{-3}$ for the grid). The existing 200-minute SMACS grid run ends with its sharpest kernel exactly at the floor: $M_{ref}$ = 545e-6 against 533e-6 arcsec², read from its log on CANFAR. This may be part of why the grid loses on star cores. Items at 0.04" should be expected to do worse there.
- **Not shown by the synthetic test.** The synthetic test cannot show this effect: its true $m_2 \ge 6\times10^{-3}$. At 0.10" pitch, where the effect should bind, items still won, but only at 100 steps.

## Recommended GPU arguments (same settings for both forms; give each a different `METHOD`)
- **SMACS grid** (reference): `--budget-min 200 --batch-patches 5 --ckpt-kernel 1 --reg-prepass-steps 100 --held-steps 200 --held-reg-steps 150 --log-every 100 --sky-form grid`
- **SMACS items:** the same, plus `--sky-form items --item-count 60000 --item-levels 3 --item-max-width 0.32`. That is 1.02 M fitted numbers against 4.5 M for the grid.
- **SMACS, second pair at `--sky-pitch 0.02`** for both forms, because of the floor argument above. Grid: 18 M numbers. Items: the same item arguments.
- **LMC grid:** `--tiles all --shifts-from /arc/projects/ots/adass2026/p1/register_xcorr_v2_r3/register_xcorr.json --budget-min 285 --reg-prepass-iters 1 --reg-prepass-steps 40 --gpu-resident-gb 2.0 --ckpt-kernel 1 --n-val-groups 2 --sky-pitch 0.02 --sky-swap 1 --sky-form grid`
- **LMC items:** the same, plus `--sky-form items --item-count 60000 --item-levels 2 --item-max-width 0.16`. That is 1.02 M numbers per tile, against 18 M.

**Memory and time on a ~10 GB GPU.** These are estimates, not measurements.
- **SMACS at 0.04":** 21.6 M patch nodes at seeding.
  - Memory: about 0.4 GB saved for the backward pass, plus 12 MB of Adam state, against 55 MB for the grid with its Adam state. The peak should stay under about 6 GB.
  - Time: about 40 to 80 ms per rasterisation pass, 3 passes per step, so about +0.12 to 0.25 s on top of the grid's 1.15 to 1.3 s/step. That is 10 to 20 per cent fewer steps in the same budget. CPU reference: 3.0 s per pass with one thread.
- **LMC at 0.02":** 58.6 M patch nodes.
  - Memory: about 1.1 GB, on top of the grid run's 7.3 to 8.0 GB peak. That is tight on 10.75 GB. If it runs out of memory, use `--item-levels 1` (0.7 GB).
  - Time: about +15 to 35 per cent per step.
- **Width risk.** Items that widen during the fit raise the patch sizes; the width cap bounds this.

## Not done or risky
- **Not done:**
  - convergence-length synthetic runs;
  - a synthetic field with cores sharper than the floor;
  - the smooth grid in a recovery run (it has only unit and smoke tests);
  - cross-fitting with items;
  - a grid-free or deconvolved rasterisation, which would remove the extra $p^2/2$ of blur;
  - any tuning of item counts or step sizes;
  - the CLI path on real data: only flag parsing is tested, there are no local files.
- **Risky:** star cores at 0.04" (see "Blur gauge and the item floor"); GPU time and memory are estimates; the per-step cost grows if items widen.

Files: `fieldop/`, `tests/` (results in `tests/items_results/`), `untouched/fieldop`, `fieldop-items.tar.gz`.
