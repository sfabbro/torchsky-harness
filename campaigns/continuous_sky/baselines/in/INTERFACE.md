# Evaluation protocol v1: shared interface (plan: https://claude.ai/artifact/5u2o5CeELJcPMQ1VpcLGPj)

Root on CANFAR: /arc/projects/ots/adass2026/protocol/v1/

## Fields (v1)

- **SMACS**
  - NIRCam `_cal` (programme 2736): /arc/projects/ots/torchsky-data/jwst_smacs0723/cal
  - Scoring box: RA 110.8268, Dec -73.4544, 20.48" square (as in all previous runs).
  - WFC3/IR is added in v1.1.
- **LMC**
  - NIRCam `_cal` (programmes 1476 and 4447): /arc/projects/ots/torchsky-data/aggregate/{1476,4447}_nircam/
  - Scoring boxes: the 10 tiles of /arc/projects/ots/adass2026/p1/plan_lmc.json.
  - The six SMACS filters first; other filters arrive as the download finishes.

## Files written by the protocol (Phase 0)

- `splits_<FIELD>.json`
  - `exposures`: list of {file, rootname, program, observation, visit, exposure, detector, filter, pupil, mjd, pa_v3}.
  - `splits`: {"dither": {"train": [...], "heldout": [...]}, "visit": {...}, "epoch": {...}, "filter_<F>": {...}}. Each list holds rootnames.
  - Each held-out list is fixed once. The rule and the random seed are recorded in the file.
- `align_<FIELD>/`: official-pipeline alignment (calwebb_image3 tweakreg, absolute to Gaia when possible) of ALL exposures of the field.
  - Holds one aligned WCS per exposure (asdf), plus the tweakreg log.
  - Used ONLY to locate held-out pixels in the reference mosaic for the masks. A method may use it only if the method's own published recipe uses tweakreg.
- `ref_<FIELD>_<FILTER>_i2d.fits`: reference mosaic per filter (official calwebb_image3 resample of ALL exposures). Used only for the S/N mask.
- `heldout/<FIELD>/<rootname>.npz`, one per held-out exposure:
  - `valid` (bool, detector shape): DQ == 0, finite SCI/ERR, and pixel centre inside a scoring box (aligned WCS).
  - `snr_ref` (float32): S/N of the reference mosaic sampled at the pixel (aligned WCS).
  - `sigma` (float32): ERR × noise_factor[detector].
  - `noise_factor`, `box_id` per pixel (int16, -1 outside).
- `noise_factors_<FIELD>.json`: {detector/filter: factor}, with the method used.

## Files written by every method

`pred/<FIELD>/<METHOD>/<SPLIT>/<rootname>.npz` containing:

- `model`: float32, detector shape, same units as SCI of the `_cal`. This is the method's prediction of the held-out exposure.
- `model_valid` (optional, bool): pixels the method claims to predict. Others are scored as missing and reported.
- `meta` (optional, json string): what the method fitted on the held-out exposure. Allowed: per-exposure nuisances only (background, global offset/affine, gain, 1/f stripes). Per-source amplitudes, positions or shapes fitted to held-out data are NOT allowed.

## Scorer (`score.py`)

- Applies one robust constant background offset per held-out exposure, the same for every method, fitted on valid pixels with snr_ref < 1. It reports scores with and without it.
- Metrics:
  - χ²/N in snr_ref bins [<1, 1–5, 5–20, 20–50, 50–200, >200];
  - headline S/N>5 χ²/N;
  - fractional residual rms((d−m)/m) for snr_ref > 50;
  - all per field, filter, detector and box, with a bootstrap over dither groups (68% intervals).
- Output: `scores/<FIELD>/<METHOD>/<SPLIT>.json` and a comparison table across methods.
