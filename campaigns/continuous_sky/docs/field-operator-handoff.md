# Handoff: one sky, many instruments — field operator track (10 Nov 2026)

Start here in a new session. This supersedes the "Latest" sections of `claude/continuous-sky-handoff.md`; that file still holds the setup notes (CANFAR, certificates, conventions) and the older history.

## Read first

1. **Plan (living doc):** [Baselines, general field operator and ablation plan](https://claude.ai/artifact/5u2o5CeELJcPMQ1VpcLGPj).
   - Phases: 0 evaluation protocol → 1 published baselines → 2 general field operator → 3 seventeen ablations → 4 scaling with data diversity.
   - Seb approved it on 10 Nov.
2. **Write-up:** [Field operator: mathematical formulation and groundings](https://claude.ai/artifact/Tb4B5qdZHFo1zyQouzZcB3).
   - The equations as implemented, the gauges, the variability scales, the rendering and its errors, groundings in published practice, the evidence and the open problems.
3. **Run log:** `claude/continuous-sky-runlog.md`, newest first.
4. **Older formulation doc:** [One sky, many instruments: formulation](https://claude.ai/artifact/Nj91AWRxgJf94mZyifAc56). It holds Seb's ticked decisions:
   - grid plus splats for the sky;
   - one shared trunk across instruments;
   - the blur gauge as "sky at the resolution of the sharpest kernel".

## Seb's direction (keep to it)

- **Bitter lesson.** Learned and data-driven. Physics only as inputs or a warm start, never as constraints. He has never seen sim-to-real work in observational astrophysics.
- **Break degeneracies with diversity.** Multi-instrument data, data splitting and strong empirical tests, not physics.
- **Catalogue-free model.** Catalogues are allowed for evaluation metrics only.
- **Baselines must be the published, standard methods.** Never a home-made stand-in.
- **The non-sky part is ONE field operator.**
  - No imposed basis: Seb finds Alard's Gaussian × polynomial basis too rigid.
  - Transmission is inside the kernel.
  - Shape normalised, amplitude structured: amplitude = pixel area from the geometry × learned throughput.
- **Communication.**
  - Plain names, no unexplained acronyms or run codes.
  - State every approximation and shortcut.
  - Concise options first.
  - Canadian English.
- **Workflow.**
  - CANFAR headless jobs use `images.canfar.net/astroai/base:latest`. No `~/.local` installs.
  - Git only on the sfabbro fork main, then a PR to astroai; no tags, releases or pushes without Seb's OK.
  - The cloud session has no GitHub credentials.

## State at handoff

| Phase | Item | State |
|---|---|---|
| 0 Protocol | Splits, official alignment, reference mosaics, held-out masks, noise factors, scorer | **Done for SMACS and LMC.** |
| 1 Baselines | Official JWST stage-3 pipeline + blot (`stage3_blot`) | **SMACS scored: 1.80 [1.61, 2.02].** **LMC scored: 6.14 [5.25, 7.22]** (F090W 28.1, the badly aligned 1476 observation 001; other filters 3.6–16). HST WFC3/IR AstroDrizzle + blot ran with a provisional split. |
| 1 Baselines | DOLPHOT (NIRCam module) | **Blocked.** Web access to americano.dolphinsim.com and arxiv.org needs Seb's approval in the app. |
| 1 Baselines | Empirical PSFs (Anderson-style), galaxy fitter (ForcePho or The Tractor), PIFF | Not started. |
| 2 Field operator | `fieldop` package v0.1 | **SMACS 1.77 [1.68, 1.86]**, a tie with the pipeline. Synthetic recovery partly works (write-up, "Evidence"). Latest code (`fieldop_v5`, 150-min fit) scored 1.94 [1.86, 2.05], worse than the 3.3 h run; rerun at matched length to tell whether the cause is fit length or the new scale rule. |
| 3, 4 | Ablations, scaling | Not started. |
| Data | LMC all filters (1476, 4447), NIRISS LMC (1501, 4471), M92 (1334) | **Downloaded** (2164 files) to `/arc/projects/ots/torchsky-data/aggregate/<prog>_<instrument>/`. |

Score definition: mean (data − model)²/σ² on held-out dither exposures, on pixels where the official mosaic has S/N ≥ 5, with σ = pipeline error × per-detector noise factor, and a 68% bootstrap over dither groups.

## To finish this round (minimum viable)

1. Done: LMC protocol and LMC pipeline score (6.14).
2. Check why F090W scores 28 on LMC and what drives the fractional residual of 1.47 at S/N > 50 (saturated cores?).
3. **Field operator:** rerun the latest code at matched length (≥ 3.3 h) on SMACS.

## Next steps (in order)

1. **Phase 2 fixes** (write-up, "Open problems"):
   - speed (1.3 s/step; CUDA graphs or larger batched FFTs);
   - port the multi-tile loop from `opjoint.py` and run LMC tiles;
   - HST loading;
   - a refitted likelihood-ratio or Laplace evidence for the variability scales;
   - the `_rate` input conversion.
2. **Phase 1:** DOLPHOT once web access is approved, then empirical PSFs, a galaxy fitter on SMACS and PIFF.
3. **Phase 3:** the 17 ablations of the plan, each in both directions with 3 seeds.
4. **Phase 4:** the scaling experiment.

## Where things live

- **Code (cloud workspace, ephemeral; copies on /arc):**
  - `/home/claude/work/fieldop/` (+ `DESIGN.md`)
  - `/home/claude/work/protocol/` (`INTERFACE.md`, `make_splits.py`, `official_align.py`, `make_heldout_masks.py`, `score.py`)
  - `/home/claude/work/baselines/` (`stage3_blot.py`, `stage3_blot_hst.py`)
  - `/home/claude/work/tests/` (`test_fieldop_*`, `test_register_xcorr.py`, …)
  - `/home/claude/work/scripts/` (`register_xcorr.py`, `tilefit.py`, `opfield.py`, `opjoint.py`, smacs_four v15–v40)
- **CANFAR:**
  - scripts: `/arc/projects/ots/adass2026/scripts/` (`run_fieldop.sh`, `fieldop-<hash>.tar.gz`, `run_protocol_masks.sh`, `run_official_align.sh`, `run_score.sh`, `run_stage3_blot.sh`, `run_register_xcorr.sh`)
  - protocol: `/arc/projects/ots/adass2026/protocol/v1/` (splits, align, reference mosaics, heldout, pred/<FIELD>/<METHOD>/<SPLIT>/, scores/)
  - LMC registration: `/arc/projects/ots/adass2026/p1/register_xcorr_v2_r3/`
  - CRDS cache: `/arc/projects/ots/adass2026/crds_cache`
- **Bundle of everything:** `/arc/projects/ots/adass2026/continuous-sky-<date>.tar.gz` (newest).

## Lessons worth keeping

- **Registration came first.** Header WCS errors up to 1.8″ in programme 1476 broke the first LMC experiment. A global, catalogue-free cross-correlation solve with a ±3″ search and cross-filter links fixed it; it agrees with the official tweakreg to 2.6 mas.
- **Per-run masks made scores incomparable.** Each method used to score on its own masks, so the same coadd scored 2.93 or 3.25 on SMACS. One shared protocol is essential.
- **Two gauges need a condition each.** The common blur and the sky pedestal must each be pinned by one declared condition. Otherwise held-out error rises with fit length, because the sky goes sharper than its grid.
- **Raw χ² does not compare across fields.** It scales with S/N squared (a 5% model error at S/N 100 gives 25). Use the fractional residual for that.
- **Earlier operators failed in transfer and in joint fits.** The previous field operator (`opfield`/`opjoint`, fixed stencil and per-filter tables) lost to the coadd and transferred negatively. The registered per-detector PSF tables transferred frozen from LMC to SMACS at 2.03 against 2.20, but they are not a published method and are no longer pursued.
