# Handoff: continuous sky model after ADASS 2026 (4 Nov 2026)

> **Superseded (10 Nov 2026):** start with `claude/field-operator-handoff.md`. This file keeps the CANFAR setup notes and older history.


Start here in a new session. The talk is done; this is the research thread that follows it.

## Where things live

- **Research plan (living doc):** https://claude.ai/artifact/QNCwd6ZCqgutp8fC7xFQyX. It covers the results table, lessons, workstreams A (learned forward model), B (sky representations plus "Experiments toward a proper sky field"), C (optimisation), D (full sky, all surveys), the evaluation protocol, the milestones, and (since 4 Nov) the **Run queue** (Q1–Q6) and **Model allocation** (which tasks go to Haiku/Sonnet, which stay on Opus). Read it first.
- **Repo proposal (long form):** `torchsky-harness/docs/proposals/full-sky-sky-model.md` on Seb's machine (`~/src/torchsky-harness`).
  - It covers the learned instrument, representations, s(x, λ, t), shared latents, feed-forward/FGN, full-sky scaling and federation.
  - It is not committed. Ask Seb before committing; his workflow is the sfabbro fork main, then a PR to astroai main, with no extra branches.
- **Other project docs:**
  - `claude/adass-talk-narrative-and-qa.md`: talk narrative and Q&A.
  - `claude/torchsky-0.5.0rc1-tpv-regression.md`: the TPV PV1_3 bug (fixed in 0.5.0).
- **CANFAR (all persistent):**
  - Scripts: `/arc/projects/ots/adass2026/scripts/`, including:
    - `smacs_four_v15.py` … `smacs_four_v20.py`;
    - `pupil_psf.py`;
    - `smacs_run.sh` (provenance launcher);
    - `wait_then_run.sh` (waits for `$WAIT_FOR`, then runs `smacs_run.sh`);
    - `holdout_main.json`;
    - `uq_calibrate.py`;
    - the SED, sweep and cartoon figure scripts.
  - Runs: `/arc/projects/ots/adass2026/smacs_four/<RUN>/`, mirrored to `vos:sfabbro/adass2026/smacs_four/<RUN>` on exit. Each has `metrics.json`, `session.log`, `cubes/`, state `.pt` files and `provenance/`.
  - Data: `/arc/projects/ots/torchsky-data/jwst_smacs0723/aligned`. These are symlinks to tweakreg-aligned crf files (a pre-alignment the model does not need; to be replaced by un-registered inputs): 135 = 3 visits × 9 dithers × (4 short-wavelength + 1 long-wavelength) detectors. Each visit carries one SW+LW band pair (F090W+F277W, F150W+F356W, F200W+F444W), so each band has 9 dithers in one visit; 123 files cover the field.
  - torchsky tarball used for the latest runs: `torchsky-src-9749c80.tar.gz`.
- **Overnight 4–5 Nov:** `claude/continuous-sky-runlog.md` lists every job, script version and result; the research plan has a section "Overnight results (5 November)".
- **Scripts added (all in `/arc/projects/ots/adass2026/scripts/`, each built by its `patch_vNN.py` from the previous version):** v21 deterministic zip renderer + capacity control; v22 `--n-test` dither protocol + dither bootstrap (Sonnet); v23 B-spline field (Sonnet); v24/v25 PSF convention flags; v26 `--zip-warm-lr`; v27 CG warm start; v28 oversampled warm start. Tests: `test_det_render.py`, `test_q5a.py`, `test_bspline.py`, `test_v24_orientation.py`, `test_cg_warm.py`; figures: `star_zooms.py`.
- **HST for SMACS J0723:** `/arc/projects/ots/torchsky-data/hst_smacs0723/raw` (222 files = 116 unique exposures; `hst_imaging_unique.csv`). Not to be pre-aligned: the model reads each exposure in its detector frame, starts from the header WCS, and fits a continuous per-exposure transform together with PSF and sky.
- **The cloud session workspace is ephemeral.** Anything not listed above should be treated as gone.

## Setting up a fresh cloud session

A new cloud workspace has no CADC certificate, no `canfar` CLI and no torch. Before any CANFAR work:

- get a proxy certificate from Seb (`cadc-get-cert -u sfabbro` on his machine, then attach `~/.ssl/cadcproxy.pem`) and put it at `~/.ssl/cadcproxy.pem` (the one from 4 Nov is valid to 28 Nov);
- `pip install --break-system-packages canfar vos cadcutils` (canfar 1.5.2 is on PyPI);
- patch cadcutils for the agent proxy: in `cadcutils/net/ws.py` set `self.verify = (os.environ.get('REQUESTS_CA_BUNDLE') or True) if not insecure else False` (the session sets `trust_env = False`, so the proxy CA is otherwise ignored);
- reads (`vls`, `vcp` from arc/vault) then work; uploads with `vcp` fail with HTTP 500, so use `arcput.py` (in the scripts folder: a direct PUT with the cert). New arc folders: PUT a ContainerNode XML to `https://ws-uv.canfar.net/arc/nodes/<path>`;
- `canfar delete` needs `-f` in a non-interactive shell; a headless command should be a script file on /arc (an inline `bash -c "..."` exited without running).

## How to launch a SMACS run

```
C="--data-dir /arc/projects/ots/torchsky-data/jwst_smacs0723/aligned --ra 110.8268 --dec -73.4544 --size 512 --scale 0.04 --loss huber --n-val 1 --holdout-from /arc/projects/ots/adass2026/scripts/holdout_main.json"
# add for 3D: --spectral rbf --spec-k 8 --spec-q 6 --spec-smooth 1.0 --spec-relative   (best 3D used --spec-k 12)
canfar create headless images.canfar.net/astroai/base:latest -n NAME -c 8 -m 64 -g 1 \
  -e RUN=<run> -e SCRIPT=smacs_four_v20.py -e TSKY_SRC=torchsky-src-9749c80.tar.gz \
  -e "ARGS=$C <flags>" -- bash /arc/projects/ots/adass2026/scripts/smacs_run.sh
```

- Useful flags:
  - `--explicit-psf {band,chrom,pupil}`, `--psf-n 121`, `--psf-lr`;
  - `--skip-explicit`, `--skip-implicit`;
  - `--budget-explicit` / `--budget-implicit` (minutes), `--eval-every-min 15`, `--patience 4`;
  - `--freeze-instrument-from DIR`, `--optimizer {adam,lbfgs}`, `--rigid`, `--uq`;
  - `--zip --instrument-file <state.pt> --zip-samples 8 --zip-frac 0.125`;
  - `--quick` for a smoke test.
- `canfar logs <id>` gives progress. Lines `VAL` / `TEST` / `PHOT` report scores.
- Conventions:
  - image `astroai/base:latest` (`improc` for SWarp);
  - no `~/.local` installs;
  - no tags, PyPI releases or GitHub pushes without Seb's OK;
  - there are no GitHub credentials in the cloud.

## Results on test exposures (S/N > 5 ⟨r²/σ²⟩, flux error)

| Run | Bright | Flux | Note |
|---|---|---|---|
| Drizzle-like coadd | 2.77 | 2.7σ | |
| Rigid instrument, grid (v15) | 2.26 | 2.5σ | |
| Learned instrument, grid, 2D (v7_e04_huber) | **1.84** | 2.1σ | the main talk result |
| Learned, grid, 3D, 8 spectral terms (v11) | 2.18 | 2.6σ | |
| Learned, grid, 3D, 12 spectral terms (v12_e3d_04_k12_rel) | **1.91** | – | best 3D |
| "Neural field" 3D (v11 n3d) | 2.35 | 2.5σ | really a reparametrised grid |
| Pupil PSF, 2D (v19_pg2d_pupil) | 1.88 | 2.2σ | better on validation, worse on test |
| Pupil PSF, 3D (v19_pg3d_pupil) | 2.20 | 2.6σ | |
| Zip-field, frozen pupil PSF (v20_zip3d_pupil) | 23.1 | 4.0σ | failed |

All-pixel scores are about 5.6 for every fit (the model-error floor), 5.88 for Drizzle-like and 7.01 for the zip-field.

UQ (uq16b run), coverage at the 68/90 % levels:

- noise only: 62/85;
- plus Laplace sky variance: 68/89;
- temperature-scaled: 70/90;
- split conformal: 71/92.

## What we learned (be upfront about these)

- **The learned instrument carries the gain.** Neither the sky representation nor the optimiser matters yet: L-BFGS stops early on validation.
- **The old "neural field" was evaluated only at grid nodes,** so its continuity was never used.
- **The old learned PSF had two shortcuts:**
  - it was cut at 0.62″ (31×31);
  - the 2D fits used only each band's pivot wavelength.
- **The fitted skies are not deconvolved.** Star zooms show the diffraction spikes still in the sky, plus a box at the PSF window edge. This is the PSF/sky degeneracy in practice. Held-out χ² can't detect the split; only injection–recovery and multi-instrument data can.
- **The physical pupil PSF did not help on test.** This supports Seb's direction: a flexible, learned PSF and instrument shared across instruments, not physics effect by effect.
- **The zip-field fails on bright pixels from the first evaluation.** Two suspects:
  - Monte Carlo PSF sampling, with 8 samples per pixel, is far noisier than photon noise on compact sources;
  - the field's capacity: its warm start reached only mse/σ² 3.8 against drizzle.
- **One test exposure per band:** differences of a few percent are within noise.

## Seb's direction and constraints

- **Catalogue-free:** no source detection, identification or separation in the model.
- **Learned sky priors:** none from external data. A shared decoder learned from the same data is an open decision.
- **Instrument:** learn it (PSF without convolution or hard window, detector nonlinearities, background, noise) and let it scale with data (HST, Euclid, Roman, ground). Bitter lesson.
- **Scale and federation:** the full sky; s(x, λ, t); federated fitting where data live.
- **Working style:**
  - concise options first, details on request;
  - Canadian English;
  - state every approximation and shortcut explicitly; Seb felt misled earlier about the neural field and the PSF.

## Latest (8 Nov, 14:30 UTC)

- **P1 gate failed.**
  - The LMC-trained NIRCam tables transferred frozen to SMACS score 4.66, against E2-data's 2.20.
  - Fine-tuned from the LMC tables, they score 2.16, a tie with E2-data.
  - Details and the per-band split are in the run log (8 Nov 14:20).
- **Leading diagnosis: the LMC tables learned misregistration blur.**
  - Within a channel the bands have no λ trend (F150W ≈ F200W per detector).
  - Widths vary threefold across the SW detectors.
  - The fitted shifts are ≤ 14 mas at the 99th percentile.
  - tilefit had no coarse shift.
- **Testing it:**
  - `tilefit.py --coarse-shift-jwst 0.2 --register-only` (Sonnet; synthetic recovery within 1.1 mas).
  - Job p1reg (`p1_lmc_register_only`) measures the LMC per-exposure shifts.
  - If they are 25–55 mas, run P1b: P1 with coarse registration, then the transfer again.
- **p1tilea** is still evaluating: coadd scores and the T5 control are pending.

- **Update (8 Nov 18:20 UTC):** the misregistration diagnosis is confirmed.
  - `register_xcorr.py` (catalogue-free pairwise cross-correlation, 3″ radius, all adjacent band links) finds whole 1476 observations off by 0.5–1.8″ in the header WCS. Observation 001, which holds all the F090W data, is off by 1.8″. 04447 agrees to within 33 mas. Closure is 2.5 mas.
  - P1b (`p1_lmc_tilefit_e2_xreg_s0b`, p1tilec, after a header-shift fix; the first launch p1tileb was killed) runs with `--shifts-from .../p1/register_xcorr_v2_r3/register_xcorr.json`.
  - Next: convert the P1b snapshot and rerun the frozen and fine-tuned transfers.

## Latest (7 Nov, 22:30 AWST)

- **Direction (Seb):** a more ambitious, scalable formulation, one sky and many instruments. The formulation doc is at https://claude.ai/artifact/Nj91AWRxgJf94mZyifAc56. Read it and `claude/operator-aggregate-design-memo.md` first.
- **Next experiment (P1):** NIRCam operator on 8 LMC tiles (PID 1476 + 4447; 1032 files, 121 GB, downloading), then frozen transfer to SMACS.
- **Running:** opfield v5 wing runs (log shape, importance sampling, two-scale), with a check-in at 20:28 UTC.
- **Best SMACS models so far** (same masks, halo 64):
  - E1 JWST-only: 1.95.
  - E2-data: 2.20 JWST-only; on JWST + WFC3/IR (2017 + 2022), NIRCam 2.33 and WFC3/IR 5.62.
  - opfield Λ = J on JWST + WFC3/IR: WFC3/IR 3.40, the best WFC3/IR yet.

## Latest (7 Nov, 08:45 AWST)

- **Recap doc:** https://claude.ai/artifact/HDJ7Wvmi86Wc4SB9HrwkJx (has an "Overnight update" section).
- **Seeds confirm the WFC3/IR ranking:** E2-data 5.53 ± 0.26 ≪ E1 9.73 < E2-lib 11.5. NIRCam is equal across models, at about 2.3.
- **NIRISS joint fit:** no harm to NIRCam (2.19). Held-out NIRISS scores 2.46, against the coadd's 2.92.
- **Field operator with the Huber loss:**
  - JWST-only 2.81.
  - JWST + WFC3/IR: NIRCam 3.63, WFC3/IR 4.14 (best WFC3/IR).
- **Null stacks:**
  - error floor about 5.5%;
  - pipeline σ low by about 29%;
  - kernels too peaked;
  - NIRISS column 1/f;
  - faint bias.
- **Running:** v39 chromatic tables, and the T3 reruns after the make_holdout rootname fix.

## Latest (6 Nov, 23:30 AWST)

- **Recap and what-next:** https://claude.ai/artifact/HDJ7Wvmi86Wc4SB9HrwkJx, with a copy at `claude/continuous-sky-recap-2026-11-06.md`. Read this first.
- **Best real models:**
  - E2-data (v36, learned tables, Gaussian start): JWST+WFC3/IR gives NIRCam 2.32, WFC3/IR 5.82.
  - E1 (v34): JWST-only 1.92.
- **The field operator fails on real data.** A Huber-loss ablation is running: opfah, opfbhub.

## Latest (6 Nov, 15:30 AWST)

- **Main track: the field operator** (`scripts/opfield.py`; spec `claude/forward-operator-spec.md`, "Main track").
  - Factorised v33–v36 is the baseline.
  - Synthetic ladder and time tests are in the run log.
- **E1 (halo) fixed a bug that dominated every earlier all-pixel χ².**
  - Best real model so far: v34 E1. JWST-only S/N>5 1.92, all pixels 1.35.
  - JWST+WFC3/IR: NIRCam 2.38, WFC3/IR 9.82 (coadd 2.90 / 12.8).
- **Running:**
  - opfield v2 with core r10 / pitch 0.5: opfa2, opfbh0, opfbh1, opfbf0.
  - E2-data B and E2-lib B.
  - Collection check-in at 15:27 UTC.
- **Seb on time dependence:** agreed to try hierarchical time codes plus a restart cycle. Identifiability comes from the operator varying coherently across the field and being low-dimensional in time, not from the time dependence itself.

## Latest (6 Nov, ~08:30 AWST)

- **Read first:** `claude/psf-geometry-audit.md`, including the addendum, and `claude/psf-literature-review.md`.
- **Seb's steer.** Break degeneracies with multi-instrument diversity, data splitting and empirical tests (T1–T7 in the addendum), not physics. The physics library is a control only.
- **The common-mode gauge** (one shared kernel g between the sky and all operators) cannot be broken by data. Declare it.
- **Running:**
  - E1 halo fix: v34, e1ajw / e1bjwhst.
  - E2-data: v36, generic learnable tables, 3″ window. e2dataa / e2datab at 0.04″; e2datab2 at 0.02″, which had OOM warnings on an 11.5 GB GPU.
  - E2-lib control: e2libb.
  - Scheduled check-ins at 01:30 and 08:50 UTC.
- **New scripts:** smacs_four_v34/v35/v36, gen_psf_lib.py, run_psflib.sh.
- **Libraries:** `/arc/projects/ots/adass2026/psflib/v1`.

## Latest (6 Nov, ~00:30 AWST)

- Operator spec: `claude/forward-operator-spec.md` (A_i linear shift-variant optics; N_i learned detector with physics only as warm start; a fully learned instrument as a separate track).
- v33 detector-frame PSF + pixel box: no gain on the JWST+HST gate (NIRCam 3.34, WFC3/IR 14.2; JWST-only 2.31). HST visits are misregistered by up to ~0.6″ in the a-priori WCS; the in-model shift fixes it only when a NIRCam anchor exists.
- Running: epoch test bjv1hw6n (JWST + 2022 WFC3/IR), learned IRAC g2mpa593 (`/arc/projects/ots/adass2026/learned_instrument/irac_li_v1/`), Spitzer pixel-store extraction (psspz0/1) with finalize queued (psfinal2).

## Latest (5 Nov afternoon)

- Inputs are JWST `_cal` and HST `_flc`/`_flt` with the a-priori WCS; registration is in the model (`--coarse-shift`, `--coarse-shift-jwst`, v31).
- Best without pre-registration: 3D K = 12 grid on JWST cal, 2.30 [2.20, 2.40] (Q5 test set). 2D on cal: 2.87 (v31 coarse shift run k0e7ihxq pending).
- Phase-2 gate (JWST + WFC3/IR) fails as run: NIRCam 2.30 → 2.82 with HST added; WFC3/IR 77.8 → 10.4 with JWST added. See the run log for hypotheses.
- Union data on arc (`smacs0723_union/`), pixel store `pixelstore_v1` built for JWST/HST/SPHEREx/GALEX (Spitzer pending); design in `claude/union-fit-design.md`.
- Splat sky v32 (`--sky-coeffs splat2`) relaunched after a bounds fix (splat2b, ekxdjlm3).

## Where the thread stands (5 Nov morning)

- The zip failure was Monte Carlo PSF sampling. The deterministic renderer (`--zip-render det`) gives test 2.48 and a deconvolved sky (star zooms).
- The learned-instrument gain over coadds holds under the 2-dither protocol: 2.42 [2.33, 2.51] vs Drizzle-like 3.77 [3.57, 3.96]; the pupil 3D grid scores 2.70 [2.59, 2.80].
- Q4 first step (121-px learned PSF window): 2D 1.96 (no gain), 3D K = 12 1.81 (best 3D).
- Q3 (B-spline) is optimiser-limited under Adam; needs CG/VarPro.
- The zip under the 2-dither protocol (instrument fitted without the test dithers): 2.67 [2.59, 2.76] vs the pupil grid on the same instrument 2.70 [2.59, 2.80]; all pixels 5.29 vs 5.14. A tie on bright pixels; the 2D learned grid (2.42) is still the best model.

## Next steps (proposed)

1. Decide what the deconvolved zip sky is for, given it ties the grid on prediction (injection–recovery of resolution is the test that could separate them).
2. Generalise the deterministic renderer to any PSF kernel (today it needs the pupil PSF tables), so the zip can use the learned chromatic PSF, and unfreeze the instrument (joint fit).
3. Q3 with a linear solver (CG on the normal equations, IRLS for Huber) instead of Adam.
4. Q6: HST loader (detector-frame pixels, DQ, ERR, header WCS + distortion as the starting transform; Sonnet task), then the JWST + HST fit. No tweakreg: registration is part of the model. The JWST inputs are still tweakreg-aligned crf files; switching them to un-registered cal files is part of the same step.
5. Q4 proper: a learned PSF with free wings beyond 121 px (flow or field with sampled wings, now that sampling is known to need a deterministic core).
