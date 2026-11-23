# onesky_v3: the torchsky library path and the comparators on real exposures

Design and plan: `torchsky/onesky.md` (kept out of git on purpose). This folder holds what runs on CANFAR.

| File | Role |
|---|---|
| `run_library_fit.py` | torchsky `load_exposures -> fit_sky -> predict_exposure`, protocol-v1 predictions (`--internal key:value` sets a private setting; `--load-fit` predicts from a saved fit and also saves the model's noise) |
| `comparators/ssf.py` | the representation and recipe of Spectro-Spatial Survey Fusion (arXiv:2611.21798), re-implemented for calibrated exposures |
| `run_job_v2.sh` | job script: builds the torchsky wheel from a working-tree tarball, runs `PY` (default the library runner), scores with `protocol/v1/code/score.py` |
| `collect_scores.py` | markdown table of scores across runs |
| `conformal.py` | calibrated intervals on held-out exposures (split conformal within brightness classes, leave one dither group out) |

On CANFAR the scripts are copies under `/arc/projects/ots/adass2026/scripts/onesky_v3/` with versioned names (a file a running job uses is
never overwritten). Runs write to `/arc/projects/ots/adass2026/onesky_v3/<RUN>/`.

Launch, one GPU (an env value must not contain `=`: the launcher drops it):

    canfar create headless images.canfar.net/astroai/base:latest -n os3-<name> -c 8 -m 64 -g 1 -e RUN=<run> -e TSKY_SRC=<tarball> \
      -e SPLIT=dither -e PY=run_library_fit_v2.py -e "ARGS=--budget-min 200 --seed 0 --internal direct_compact:1" \
      -- bash /arc/projects/ots/adass2026/scripts/onesky_v3/run_job_v2.sh

Checks made here: the comparator's renderer against a separately written dense sum (8e-5 of the peak); the library runner end to end on
24 and on 123 SMACS exposures (22 Nov 2026).

## Cross-instrument test and latent compression (22 Nov 2026, evening)

- `make_cross_splits.py`: the joint JWST + HST WFC3/IR list with 15 HST exposures held out (`splits_SMACS_cross.json`, passed with `--splits`).
- `run_library_fit.py` (on the cluster as `run_library_fit_v5.py`): `--instruments` (keep some instruments), `--sky-from DIR` (response-only
  fit against the sky of a saved fit, torchsky's private `_fit_response_to_sky`), `--latents kind:d,...` with `--load-fit` (every item rebuilt
  from d latents, held-out exposures predicted again under `<method>_<kind><d>`), side files `<run>/xpred/` for every held-out exposure.
- `run_job_v3.sh`: as `run_job_v2.sh`, and scores the variants `<METHOD>_*` as well.
- `score_common.py`: protocol scores of several methods on identical pixels that every library fit saw (the protocol's pixels minus those the
  loader dropped), with bootstrap intervals and paired differences. Use it, not the protocol's table, to compare library fits with anything.
- `score_cross.py`: paired scores of held-out exposures from the `xpred` side files (HST has no protocol mask); classes from a reference
  method's own prediction, one noise factor per exposure from blank sky.
- Runs: `x_hst_*` (HST alone), `x_joint_*` (both), `x_h2j_*` (HST sky, JWST response fitted), `x_j2h_*` (JWST sky, HST response fitted),
  `lat_*` (latent variants of the JWST fits); `_base` = reference renderer, `_direct` = compact items at exact positions.
- Comparator on arc: `ssf_v3.py` (22 Nov, 23:30 UTC) bounds the (pixel, component) pairs per piece (`--max-pairs`); `ssf_v2.py` ran out of GPU
  memory near 115,000 components. Runs: `ssf3_11ep`, `ssf3_200m`, `ssf3_11ep_noF200W`.
- `run_library_fit.py --shifts-from JOINT_FIT` (on arc: `run_library_fit_v6.py`, tarball `torchsky-wt-45fc2a44-4d7276da.tar.gz`): with `--sky-from`, every
  exposure starts at the translation a joint fit found for it (cards `TSSHIFTX`, `TSSHIFTY`); no search. Runs `x_h2jk_*`: the control for
  `x_h2j_*`, whose JWST exposures ended a median 59 mas from where the joint fit puts them.
- Comparator `ssf_v4.py` (23 Nov): `--fit-background 1` fits one constant per exposure (the median of a cut is not one level across cuts of a
  cluster box). Runs `ssf4_200m`, `ssf4_200m_noF200W`. `score_common.py --bands F150W` scores one filter. `lib_base_s0_400m`: the baseline at the
  new renderer's 400 minutes.

## Kernel start, noise and coverage tests, a second field (23 Nov 2026)

- `run_library_fit.py` (on the cluster as `run_library_fit_v12.py`, through `torchsky.map.exposure_fit.experimental`): `--kernel-start NPZ`
  and `--kernel-start-wfc3ir NPZ` (the kernel starts from point-source images), `--flat-noise 1` (the pipeline error of an exposure replaced
  by one number, its blank-sky median), `--noise-floor-max`, `--train-half 0|1` (every second training exposure of each filter and detector),
  `--train-filters`, `--spectral-k`.
- `mask_check.py`: share of in-box pixels the loader drops, per flag.
- Job script: `FIELD=LMC` runs the second field; `SCORE=0` skips scoring.
- A field with several visits needs `--internal coarse_reg:xcorr --internal coarse_reg_reference:visit` and a search radius that covers the
  offsets between visits. LMC box 0: up to 1.84 arcsec (the 2022 commissioning visits), so `coarse_reg_radius:3`. A probe job
  (`--budget-min 1 --no-predict`, `SCORE=0`) logs every visit's offset in about ten minutes. Runs `lmc3_*` (radius 1.5) left 70 of 220
  exposures unplaced and are invalid; `lmc4_*` place all 220.
- `diagnostics/`: the analysis scripts behind the recorded numbers (see its README).
