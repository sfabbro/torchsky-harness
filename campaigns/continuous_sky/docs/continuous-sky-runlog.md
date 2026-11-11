# Continuous sky: overnight run log (4–5 Nov 2026)

## 11 Nov 04:50 UTC: evaluation protocol complete for LMC; official pipeline scored there

- **LMC protocol complete.**
  - Official alignment: all 10 sky patches × 6 filters.
  - Held-out masks: 518 files.
  - The trivial "zero" model scores 30.3 on source pixels.
- **Official JWST stage-3 pipeline on LMC held-out dithers** (62 exposures, none missing).
  - Mean (data − model)²/σ² on source pixels (S/N ≥ 5): **6.14 [5.25, 7.22]**.
  - By S/N of the official mosaic: <1: 1.85; 1–5: 2.11; 5–20: 2.23; 20–50: 4.56; 50–200: 29; >200: 259.
  - Fractional residual rms at S/N > 50: 1.47. Likely dominated by saturated or near-saturated star cores; to check.
  - Per filter: F090W 28.1, F150W 3.96, F200W 3.56, F277W 16.2, F356W 7.74, F444W 8.26.
  - F090W is all in programme 1476 observation 001, the one with the 1.8″ header error. Its poor score suggests residual misalignment or crowding in that observation; to check.
- **Next for the field operator:** run on LMC patches (needs the multi-patch loop ported from opjoint) and compare with 6.14.

## 10 Nov 19:10 UTC: field operator with the latest code is slightly worse; LMC alignment still running

- **Field operator, latest code** (`fieldop_v5`, 150-min fit, likelihood-ratio rule for the variability scales, faster rendering): SMACS held-out S/N≥5 score **1.94 [1.86, 2.05]**. The earlier 3.3 h fit scored 1.77, and the official pipeline 1.80.
  - Per filter: F090W 1.71, F150W 1.89, F200W 1.74, F277W 2.21, F356W 2.33, F444W 2.10.
  - Fractional residual at S/N > 50: 0.061.
  - Shorter fit and changed scale rule, so the cause of the drop is unclear. Next session: a matched-length run of the latest code.
- **LMC official alignment rerun** (protoalignlmc3, 6 h so far):
  - tiles 0 and 1 now complete;
  - tile 5 has 1 of 6 filters;
  - tiles 2, 3 and 7 are not started.
  - LMC masks, and the scoring of the pipeline baseline on LMC, follow once it ends.

## 10 Nov 13:30 UTC: new plan under way. Protocol, first published baseline, first general field operator

Plan: [Baselines, general field operator and ablation plan](https://claude.ai/artifact/5u2o5CeELJcPMQ1VpcLGPj). All scores below are the protocol scorer's: held-out dithers, mean (data − model)²/σ² on pixels where the official mosaic has S/N ≥ 5, 68% bootstrap intervals.

**Evaluation protocol (Phase 0)**
- Done for SMACS:
  - splits;
  - official pipeline alignment;
  - reference mosaics;
  - held-out masks;
  - noise factors;
  - scorer.
- LMC alignment job died partway; tiles 4, 6, 8 and 9 are done. Relaunched for tiles 0, 1, 2, 3, 5 and 7 (protoalignlmc3) with fewer workers. LMC masks follow.
- Files are in `/arc/projects/ots/adass2026/protocol/v1/`.

**Published baseline 1: JWST stage-3 pipeline plus blot (`stage3_blot`)**
- Tweakreg runs on all exposures; skymatch, outlier detection and resample run on the training exposures. The mosaic is blotted into each held-out exposure.
- **SMACS:** 1.80 [1.61, 2.02].
  - Per filter: F090W 3.13, F150W 1.66, F200W 1.35, F277W 1.74, F356W 2.05, F444W 2.12.
  - Default tweakreg tolerance mis-aligned F090W (fit rms 318 mas). Tightened to 0.2″.
- **LMC:** predictions exist for all 10 tiles, but are unscored until the masks exist.
  - Tweakreg agrees with our cross-correlation shifts to 2.6 mas (median).
- **HST WFC3/IR:** AstroDrizzle plus blot ran with a provisional split.

**Published baseline 2: DOLPHOT.** Blocked. Reading its documentation and the Weisz et al. 2024 paper needs web access (americano.dolphinsim.com, arxiv.org), and the permission prompts went unanswered.

**General field operator (`fieldop`, Phase 2, first version)**
- One conditional neural field for the kernel shape, with no basis imposed. The amplitude is pixel area × learned throughput.
- Rendering uses overlap-add FFT, checked against per-pixel integration to 6e-4–2e-3.
- Variability (across the detector, between epochs, between exposures) is switched on or off by held-out checks.
- **SMACS, 3.3 h fit:** 1.77 [1.68, 1.86], against stage-3's 1.80. A tie on the headline number.
  - Better at S/N 50–200 (3.35 against 6.48), in F090W (1.69 against 3.13) and in F444W.
  - Worse at S/N 5–50 and above 200, and in F200W and F277W.
- **Synthetic recovery** (gauge-invariant checks):
  - injected PSF variation across the detector recovered to 14%;
  - variability that was not injected is pruned (65–130× smaller);
  - throughput gradient recovered to about 10%;
  - offsets recovered to about 6 mas.
  - Not yet working: the PSF's colour dependence within a filter; held-out error is not yet monotone with fit length.
- **Speed:** only 1.8× faster, not the 5× targeted.
- A run with the latest code (`fieldop_v5`) is in progress.

## 9 Nov 16:35 UTC: joint field operator (opjoint) final scores. It fails against the coadd; its shared operator transfers negatively

All scores are held-out χ²/N on S/N>5 pixels. Masks are opjoint's own, so they differ from the v38 masks (its SMACS drizzle coadd NIRCam scores 3.25 here, against 2.93 under v38). This is one more reason for the fixed protocol in the new plan.

**Joint run `p1j_joint_s0`, 420 min:**

| Held-out set | Field operator | Drizzle coadd |
|---|---|---|
| SMACS NIRCam | 4.45 [4.30, 4.59] | 3.25 |
| SMACS WFC3/IR | 18.8 [14.9, 22.6] | 11.2 |
| LMC dithers, pooled | 48.6 [44.2, 53.0] | 28.4 |
| LMC dithers, per tile | 39–80 | – |

- The field operator is worse than the coadd on every field.
- **Held-out LMC tiles:** the shared operator with only the local parts refit scores 59.8 (tile 5) and 65.9 (tile 9). Refitting the operator on tile 5 alone (the control) gives 31.4. The shared operator transfers negatively.

**Leave-SMACS-out `p1j_lomo_smacs_s0`:** SMACS pooled 104 local-only, 120 after fine-tuning. WFC3/IR has no training field there, so its generic kernel dominates. The NIRCam-only split was not printed. Under any reading it is far worse than the joint run.

**Continuation `p1j_joint_s0_cont`:** SMACS validation plateaued at 4.95–5.09 (639–765 min). Its best before the warm restart was 4.84.

**PSF tables plus registration (P1b, `p1_lmc_tilefit_e2_xreg_s0b`):** LMC dithers pooled 25.4; held-out tile 5 scores 20.8 (P1: 71.4). The coadd comparison and control are still pending.

**Conclusion:** the field operator as implemented fails on real data at this budget, both on its own fields and in transfer. The PSF tables with registration are the only configuration that beats the coadd so far. This, and Seb's critique (reinvented baseline, semi-classical design), led to the new plan: [Baselines, general field operator and ablation plan](https://claude.ai/artifact/5u2o5CeELJcPMQ1VpcLGPj). No new runs until its protocol and baselines are in place.

## 9 Nov 12:40 UTC: joint field operator at 420 min is behind; continuation recovering from its warm restart

**p1j_joint_s0** (evaluation still running), held-out S/N>5:
- **SMACS T1 pooled** (NIRCam + WFC3/IR, 32 detector images): **5.45 [5.18, 5.71]**; all pixels 1.91; flux error 3.46σ.
  - Single-field references on the same split: opfield all-three B, NIRCam 3.20 and WFC3/IR 2.33; JWST-only 2.23.
  - Per-instrument split at S/N>5 is still pending.
  - All-pixel χ² per exposure: NIRCam SW 1.25–1.43, LW 2.6–2.7; WFC3/IR 1.9–7.6.
- **LMC T1:**

| Tile | Joint opjoint | P1b tilefit (tables + registration) |
|---|---|---|
| 0 | 39.2 | 23.2 |
| 1 | 46.1 | 27.9 |
| 2 | 45.3 | 23.5 |
| 3 | 62.4 | 38.6 |

  - Caveat: the held-out protocols differ. opfield refits only z and the background (S12, no affine refit); tilefit refits the affine. The two are not strictly comparable.
- **Reading:** at 420 min the joint field operator is worse than both the single-field operator (SMACS) and the registered tables (LMC). Suspected causes:
  - SMACS gets only 20% of the steps;
  - the field operator needs more steps than tables;
  - crowded LMC fields.

**p1j_lomo_smacs_s0:** LMC T1 tile 0 28.8, tile 1 39.3, tile 2 32.7, tile 3 59.2. The SMACS local-only and fine-tune scores come after the LMC evaluation.

**p1j_joint_s0_cont:** the warm restart (learning rate back to about half its peak) set it back.
- SMACS logged loss jumped from −4.3 to about +60, and is recovering to about 21–35.
- SMACS validation: 6.61 at 451 min, 5.89 at 483 min, 5.45 at 514 min. The best of the first run was 4.84.
- It is recovering as the learning rate decays again. Approximation noted: the warm restart costs about 100 min of the extra 420.

## 9 Nov 10:40 UTC: P1b frozen transfer PASSES. Registered LMC tables beat SMACS-only tables

**`p1b_transfer_frozen`:** the LMC-trained arm-A tables from P1b (registered), frozen; only the SMACS sky and nuisances are fitted. Same split and arguments as `p1_transfer_frozen`. Held-out NIRCam:

| Model | S/N>5 | All pixels | Flux error |
|---|---|---|---|
| P1b frozen | **2.03 [1.95, 2.11]** | 1.41 | 1.47σ |
| P1 frozen (unregistered) | 4.66 [3.93, 5.41] | 1.89 | – |
| E2-data (SMACS only) | 2.20 [2.14, 2.26] | 1.44 | – |
| E1 | 1.95 [1.84, 2.05] | – | – |
| Drizzle coadd | 2.93 | – | – |

- **Gate (memo):** frozen ≤ 2.26. Met, and the frozen transfer beats E2-data with non-overlapping intervals.
- **`compare_operators` against E2-data**, all-pixel χ² per band:
  - P1b better in F090W, F277W, F356W and especially F444W (1.27 against 1.60);
  - equal in F150W and F200W.
  - Shoulder share 6.6% against 6.9%. Shoulder plus wing share 13.9% against 13.6%, so the memo's ≤ 11% criterion fails for both.
- **Reading:** with registration fixed, PSF tables learned from 8 crowded LMC tiles in another epoch and programme transfer frozen to SMACS, and predict better than tables learned on SMACS itself. The largest gain is in LW, where SMACS alone constrains least.

**P1b LMC T1** (`p1_lmc_tilefit_e2_xreg_s0b`), S/N>5:

| | Tile 0 | Tile 1 | Tile 2 | Tile 3 | Tile 4 | Tile 6 | Tile 7 | Tile 8 | Pooled |
|---|---|---|---|---|---|---|---|---|---|
| P1b | 23.2 | 27.9 | 23.5 | 38.6 | 20.9 | 14.3 | 23.7 | 19.4 | **25.4** |
| P1 (unregistered) | 89.8 | 101.7 | 71.2 | 136.2 | 21.0 | 29.0 | 112.5 | 19.9 | 88.1 |

- Still high in absolute terms (crowded field). The coadd comparison and T5 are still pending.

## 9 Nov 10:00 UTC: joint run finished training; continuation launched

**p1j_joint_s0:** budget reached at about cycle 24k. Its evaluation (T1 per field, T5 with control) is running.

SMACS validation score (robust all + src) over the run:

| Minute | 127 | 159 | 190 | 221 | 253 | 284 | 315 | 347 | 378 | 409 |
|---|---|---|---|---|---|---|---|---|---|---|
| Score | 10.75 | 11.48 | 7.04 | 6.59 | 6.05 | 5.76 | 5.83 | 5.00 | 5.29 | **4.84** |

- Still falling at the end.
- The single-field opfield_v5_wfc3ir_all3 reached 4.27 at 91 min.
- Final logged SMACS loss −4.3, against −5.2 at the single-field end.
- Dense LMC tiles are at 0.5–1.4.

**Continuation:** p1jjoint2 (ryzflowo), `p1j_joint_s0_cont`.
- It copies `p1j_joint_s0/ckpt.pt` into a new run directory, using `scripts/continue_opjoint.sh`, so the first run's evaluation outputs stay intact.
- Same arguments, with `--budget-min 840`, so about 420 more minutes.
- **Approximation:** the cosine learning-rate schedule runs on elapsed time over the budget. Raising the budget restarts the learning rate at about half its peak, a warm restart.
- Purpose: about twice the SMACS steps before judging the gate. The first run's T1 is reported as the 420-min point.

## 9 Nov 05:20 UTC: opjoint progress. SMACS is learning, but under-trained, as predicted

**p1j_joint_s0:**
- **First SMACS validation** at cycle 7123 (127 min): score 10.75 (robust all 3.14 + src 7.61). Per exposure, WFC3/IR χ² is 2.4–7.0.
- **Single-field reference:** opfield_v5_wfc3ir_all3 scored 4.27 at 91 min, but that was 19.4k SMACS steps. The joint run has given SMACS about 1.4k joint cycles, so the two are not comparable yet.
- **Logged loss is falling:** SMACS 107 → 64 from cycle 3000 to 8000; coarse-to-fine reached 0 px. Dense LMC tiles are now at 2.1–3.1 (registered P1b, tilefit: 3.0–3.8 at cycle 60, different quantity).
- **Projection:** about 22k cycles in the budget, so about 4.4k SMACS cycles, still far below the single field's 61.7k steps. The SMACS T1 from this run will be confounded by under-training.
- **Plan:** resume from `ckpt.pt` (automatic on relaunch) for a second 420 min before judging the gate. Or raise SMACS's floor share in a resumed run.

**p1j_lomo_smacs_s0:** cycle 7800, LMC tiles 2.8–5.0.

## 9 Nov 03:35 UTC: opjoint health; P1 final LMC metrics

**opjoint runs:** healthy.
- No OOM. Selfcheck ok (inner sky steps leave the global parameters and Adam state unchanged).
- A13: 518 LMC exposures shifted. SMACS WFC3/IR coarse shifts are 0.15–0.5″, χ² ×0.35–0.42, as in the single-field runs.
- About 1.15 s per cycle; 2000 cycles in 38 min, so about 22k cycles in the 420-min budget.
- Tile probabilities: SMACS 0.20; dense LMC tiles 0.13–0.15; thin tiles 0.026.
- Logged per-tile data loss at cycle 2000 (coarse-to-fine still at a 7 px blur):
  - LMC 4.3–7.0, falling;
  - SMACS 100–160, noisy.
  - This is the importance-weighted batch loss. It is not yet comparable with the single-field opfield log, which reports a different quantity. Judge by the SMACS validation score instead (single-field all-three: VAL 4.27 at 91 min).

**P1 (`p1_lmc_tilefit_e2_s0`, unregistered) final `tile_metrics`**, S/N>5 χ²:

| Tile | Model | Drizzle coadd |
|---|---|---|
| 0 | 89.8 | 72.8 |
| 1 | 101.7 | 82.6 |
| 2 | 71.2 | 61.6 |
| 3 | 136.2 | 127.1 |
| 4 (thin) | 21.0 | 21.0 |
| 6 | 29.0 | 28.5 |
| 7 | 112.5 | 91.9 |
| 8 (thin) | 19.9 | 19.7 |

| Held-out tile | Shared instrument | Control (refit on the tile) | Drizzle coadd |
|---|---|---|---|
| 5 | 71.4 | 75.2 | 60.7 |
| 9 | 118.2 | 112.5 | 126.2 |

- The model is worse than the drizzle coadd on every dense tile.
- The shared instrument is no better than the control: pooled T5 91.3 against 91.0.
- This confirms that the unregistered P1 fit was broken, not just under-trained. (The SWarp coadd on the thin tiles, about 130, is an outlier; the drizzle coadd is the reference.)

## 9 Nov 02:50 UTC: P1b training done; table diagnostics; frozen transfer launched (secondary)

- **Training:** budget reached at cycle 1210; best validation 10.30 at cycle 1002, against 17.4 in P1.
- **Tables** (`/arc/projects/ots/adass2026/p1/instrument_state_p1b.pt`) compared with P1:
  - **SW spread across detectors (cv of m2) dropped:** F150W 0.44 → 0.26; F200W 0.47 → 0.42; F090W 0.12 → 0.12.
  - **NRCBLONG now looks like NRCALONG:**
    - EE(1 px): 0.84–0.85 in both, against 0.53–0.58 for NRCBLONG in P1;
    - m2: 0.64–0.73 for NRCBLONG, 0.46–0.70 for NRCALONG.
  - **Still wide in module B:** NRCB1 and NRCB2 F150W/F200W (m2 1.7–1.9) against NRCB3 and NRCB4 (0.8–1.1).
  - **Still no λ trend within SW** (F090W is the broadest). Under the declared blur gauge that is not a physical test: absolute widths carry g(λ).
- **Frozen transfer launched:** p1btfz (u6xc4yyc), `p1b_transfer_frozen`, with the args of `p1_transfer_frozen` and only `--load-tables-from` changed. This is a diagnostic only; the main track is opjoint.

## 9 Nov 02:20 UTC: P1 redefined as a joint global–local inversion with the field operator (Seb)

**Seb's correction:**
- Frozen transfer is not the method. The problem is one global–local inversion of the sky and all the instruments.
- P1 had drifted to factorised PSF tables (arm A), when the formulation is one learned field operator.

**Build (Sonnet): `opjoint.py`**
- **Global:** one OpField operator shared across all fields. It holds the trunk and hypernetwork, the instrument and detector embeddings, the base and wing tables, the bank and the noise nets.
- **Local:**
  - the sky per tile;
  - codes, background and stripes per exposure, shared across tiles;
  - hierarchical codes optional.
- **Loop:** per visit (25 cycles) on one tile, K = 2 inner sky steps, then a joint step. Tiles are picked in proportion to their pixels, with a field floor of 0.2. Inactive skies and pools stay on the CPU.
- **Fields as data:** `fields_p1j.json` holds the LMC plan (with A13 shifts) and SMACS (NIRCam plus WFC3/IR 14096 + 16729).
- **Tests:**
  - T1 per tile, field and instrument;
  - T5 against the control;
  - `--holdout-field`: local-only refit, then an optional fine-tune.
- **Approximations J1–J20** are in the docstring. The main ones:
  - J5: the objective is Σ pₖ Lₖ;
  - J6: the inner solve is Adam steps;
  - J8: validation comes from SMACS only;
  - J16: when SMACS is left out, WFC3/IR has no training field, so only the NIRCam score is meaningful.
- **Equivalence:** bit-identical to opfield on the CPU in a restricted single-tile configuration.
- **Synthetic test:** the thin field gains from the joint fit (1.20 against 1.49). Leave-field-out with only local parameters refit scores 1.175. T5 with the operator fixed scores 1.13, against 1.18 for the control.

**Launched** (opfield all-three flags: Λ = J, log Ψ, importance sampling, two-scale 0.4″; budget 420 min):
- p1jjoint (r3fyi89r): `p1j_joint_s0`, with `--ht-control`;
- p1jlomosmacs (o4w4vtm5): `p1j_lomo_smacs_s0`, with `--holdout-field SMACS --lo-steps 6000 --lo-ft-cycles 600`.

**SMACS references, same split** (held-out S/N>5):
- opfield all-three: JWST-only 2.23; on B, NIRCam 3.20 and WFC3/IR 2.33;
- E2-data: 2.20; on B, NIRCam 2.33.

**Risk:** SMACS gets far fewer steps than in the single-field run (≥ 20% share of 17–25k cycles, against 61.7k steps), so it may be under-trained.

**P1b** (the arm-A tables with registration) keeps running as a registration diagnostic. Its transfer reruns are now secondary.

## 8 Nov 19:55 UTC: P1b (fixed) is healthy

- **Shifts applied:** A13 on 518 exposures, 1719 headers shifted. Selfcheck ok.
- **χ²/pix against P1, at the same cycle:**
  - thin tiles 4 and 8 (cycle 20): 1.55 / 1.39, against 1.55 / 1.38. They are identical, as expected for a common shift.
  - dense tiles (cycle 60): 3.0–3.8, against 5.5–6.0 in P1. They are still falling.
- **Training ends** at about 02:30 UTC. The snapshot conversion and the SMACS transfers are scheduled for 02:40 UTC.
- **P1 held-out tile 5:** 71.4 with the LMC-trained instrument, against 75.2 for the control fit on that tile alone. P1 was misregistered.

## 8 Nov 19:10 UTC: P1b relaunched after a header-shift bug

- **Bug.** The first P1b run (p1tileb) applied the A13 shifts only to tx/ty. `V.classical` builds the drizzle sky init and the `--score-coadds` references from `e["header"]`, so it used the unshifted WCS.
- **Symptom.** Tiles 4 and 8 (all 01476 obs 001, one common shift of about 1.13″) started at χ²/pix 6.5, against 1.5 in P1. Every tile started worse; t0 was at 8.8 against 5.7 at cycle 1.
- **Killed p1tileb.**
- **Fix (Sonnet).** `shift_exposure_header` iterates CRVAL1/2 so that the WCS built from the header moves the exposure pivot by exactly the shift; SIP is untouched. It applies to both A12 and A13, once per header.
  - Test: SW and LW SIP headers at dec −69.5°, shifts up to 2.5″; worst deviation from tx/ty is 0.004 tile px.
  - `V.classical` coadd: within 0.05 px.
  - Existing tests are unchanged.
  - The same bug affects A12 runs with large shifts. The smacs_four v31 coarse shift is unaffected in practice (its shifts were 21 mas or less).
- **Relaunched:** p1tilec (jbh8fu5o), `p1_lmc_tilefit_e2_xreg_s0b`.

## 8 Nov 18:20 UTC: xcorr registration of LMC: large, coherent header-WCS offsets; P1b launched

**rxcorr1** (radius 1.5″, xband F150W:F200W, F277W:F356W, F356W:F444W) was not usable:
- the graph split into 3 components;
- 49.5% of edges were rejected, 5426 of them with the peak at the search boundary.
Some relative offsets exceed 1.5″, and the v1 solution aliased several visits.

**rxcorr2** (`register_xcorr_v2_r3`: radius 3.0″, all adjacent band pairs from F090W:F150W to F356W:F444W):
- **Graph:** 1 component; 14 079 edges, 97% good, 3.2% rejected.
- **Consistency:**
  - closure p50 2.5 mas, p90 5.1 mas;
  - dither-group agreement p50 3.2 mas;
  - spread across tiles p50 2.5 mas (1 of 352 above 10 mas).
- **Offsets relative to 04447 visit 001001:**
  - **04447:** all visits within 33 mas.
  - **01476, observation 001** (all 10 visits, coherent to about 2 mas): (+1349, −1214) mas, |1.81″|.
  - **01476, observations 003, 004 and 006:** mostly 5–41 mas. Outliers:
    - 003001: 0.77″;
    - 003004: 1.79″;
    - 006003: 0.49″ and 1.05″, two activities that disagree with each other;
    - 103001: 0.88″;
    - 104001: 0.75″.
- **The LMC F090W data are all in 01476 obs 001**, so every F090W exposure is about 1.8″ off.
- **Reading:** header-WCS errors of 0.5–1.8″ on whole observations in Cycle 1 calibration programme 1476. P1 ran on these unregistered. The tables could not absorb arcsecond offsets, which explains:
  - the poor LMC T1 (the sky sees multiple images);
  - broad tables with no λ trend;
  - the failed transfer.
  The A12 χ² scan (±0.4″) could not reach these offsets.
- **Approximations:** R1–R10 in the JSON (translation only; one shift per image across tiles; edge σ from a formula). Formal σ is about 0.1 mas, an underestimate next to the 2.5 mas closure.
- **Untested:** an independent absolute check, Gaia as a diagnostic only, never an input.

**Launched P1b:** p1tileb (djcn6ppp), `p1_lmc_tilefit_e2_xreg_s0`. Same arguments as `p1_lmc_tilefit_e2_s0`, plus `--shifts-from .../register_xcorr_v2_r3/register_xcorr.json` (A13).
- When training ends, convert `instrument_best_snapshot.pt` and rerun the frozen and fine-tuned SMACS transfers. Don't wait for the evaluation.

## 8–9 Nov: LMC registration diagnostics

**p1reg** (`p1_lmc_register_only`, A12 χ² scan against cross-band splat coadds, R = 0.2″) measured 518 exposures.
- **|shift|:** p50 20 mas, p90 311, p99 449.
  - Per programme: 01476 median 223 mas; 04447 median 13 mas.
  - By band: F090W p50 98 mas; the others 12–23 mas.
- **The scan is not reliable:**
  - The median χ² ratio (best/zero) is 0.992, so the likelihood is almost flat.
  - 28% of the tile-exposures land on the grid edge.
  - 231 of the 352 multi-tile exposures disagree across tiles by more than 10 mas, up to about 1″.
  - Likely cause: the reference coadds are themselves built from misregistered exposures, in a crowded field.
- **Not applied.** P1b was not launched on these shifts.
- **Reading:** large header-WCS errors in 01476 (Cycle 1) are plausible but not measured reliably. The misregistration hypothesis is still open; it is neither confirmed nor refuted.

**New registration stage** (Sonnet): `register_xcorr.py`.
- **Method:** pairwise masked normalised cross-correlation of exposures in the same band, resampled onto each tile's TAN grid using the header SIP WCS (the same WCS the engine uses). Sub-pixel peak, with the second-peak ratio as a reliability metric.
- **Ties:**
  - SW/LW simultaneity: one unknown per dither group, so the detectors of a dither share one shift.
  - Cross-band edges F150W:F200W, F277W:F356W and F356W:F444W (`XBAND=1`). Without them the LMC plan's graph splits into 3 components.
- **Solve:** one joint robust LSQ over all tiles; one unknown per exposure group across tiles.
- **Approximations R1–R10** are in the JSON. The main ones: translation only; one shift per image across all tiles; edge σ from a formula with one inflation factor; thresholds tuned on synthetic data.
- **Synthetic test** (crowded field, 0–500 mas injected, rolls up to 15°): recovered to within 0.54 mas; closure p90 1.1 mas; the aliasing case is flagged.
- **tilefit:** `--shifts-from FILE` (A13) applies the shifts with A12's exact arithmetic. The existing tests are unchanged.
- **Launched:** rxcorr1 (spr4dz2q), CPU, output in `/arc/projects/ots/adass2026/p1/register_xcorr_v1/`. It compares itself against the A12 shifts.

**p1tilea:** still running (19 h). The T5 tile 5 control is at step 481 of 1200, about 17 s per step. Tile 9 and its control are still to come, about 12 more hours. The coadd scores are written only to `tile_metrics.json` at the end.

## 8 Nov 14:20 UTC: P1 transfer result: the gate fails (frozen), fine-tune ties

**SMACS, `v38_E2data_jwst_dump` split, NIRCam held-out S/N>5 (all pixels):**

| Run | S/N>5 | All pixels |
|---|---|---|
| E2-data (SMACS only, reference) | 2.20 [2.14, 2.26] | 1.44 |
| E1 halo 64 | 1.95 [1.84, 2.05] | – |
| **LMC tables, frozen** (`p1_transfer_frozen`) | **4.66** [3.93, 5.41] | 1.89 |
| **LMC tables, fine-tuned** (`p1_transfer_finetune`, psf-lr 2e-3) | **2.16** [2.06, 2.26] | 1.46 |

Per band, S/N>5:

| Band | Frozen | Fine-tuned |
|---|---|---|
| F090W | 2.34 | 2.13 |
| F150W | 5.06 | 2.23 |
| F200W | 4.86 | 1.74 |
| F277W | 9.51 | 3.36 |
| F356W | 5.55 | 3.05 |
| F444W | 1.57 | 1.73 |

- **Gate (memo):** frozen > 2.40 while the fine-tune recovers, so P1 as run is **falsified**. The LMC-trained tables do not transfer frozen. Fine-tuning from them reaches E2-data, but does not beat it.
- **`compare_operators`, frozen against E2-data:**
  - The shoulders (1–3 px) take 24% of χ², against 7% for E2.
  - The excess is in F150W, F200W, F277W and F356W: +0.34, +0.51, +0.25 and +0.17 of E2's χ².
  - F090W and F444W show no shoulder loss. Both are poorly sampled at LMC: F090W has thin coverage, and F444W is the broadest kernel.
  - The fine-tune removes most of the gap; the residual is about +0.06 to +0.09 in the shoulders.
- **First reading (weakened by the diagnosis below; the SMACS sky did converge):** the LMC tables are broad (broad; NRCB1/F150W EE at 1 px = 0.36), whereas the SMACS sky converges under the SMACS gauge (sharpest kernel ≈ delta). The frozen tables impose a gauge g that is different from the one the SMACS sky is fit in. That gauge is held fixed while the sky has not adapted. The sky is fit from a drizzle start with K inner steps, so it may never deconvolve far enough in the budget. Untested alternatives: the LMC tables are wrong (the T1 held-out problem below), or they are field- or epoch-specific.
- **LMC held-out is poor:** T1 S/N>5 is 71–136 on dense tiles 0, 1, 2, 3 and 7; about 20 on thin tiles 4 and 8; 29 on tile 6. T5 tile 5 is 71.4. The coadd scores and the T5 control are still pending: p1tilea is still evaluating, 17 h in. Until they arrive we cannot tell whether the LMC fit itself is bad (crowding, under-training at about 150 cycles per tile) or whether the scoring is simply hard in a crowded field.

**Diagnosis: the LMC tables look like misregistration blur, not PSF.**
- **No λ trend within SW.** Per detector, the F150W and F200W tables are near-identical, both in EE and in centroid; for example NRCB2 has EE(1 px) 0.21 and 0.20. Optically, F200W should be about 1.3× broader. The same holds for F277W and F356W.
- **Threefold width spread across SW detectors.** Core m2 runs from 0.75 (NRCA3) to 2.97 (NRCB2) in os-px². The SW detectors' optical PSFs are similar to each other.
- **The fitted per-exposure shifts are tiny:** |t| median 1.5 mas, 99th percentile 14 mas. On SMACS, pipeline `_cal` WCS errors were 25–55 mas. tilefit had no coarse shift, and gradient-only fitting is known to stall at more than half a PSF core. The shared tables would then absorb the average misregistration, and a crowded field makes this worse. That would explain both the poor LMC T1 and the failed transfer.
- **The frozen SMACS transfer converged** (validation flat at 2.81 from step about 7000), so this is not under-optimisation of the SMACS sky.
- **Build (Sonnet):** `tilefit.py --coarse-shift-jwst R [--register-only]` runs the engine's grid scan per tile; an exposure that appears in several tiles gets the pixel-weighted mean of its shifts; A12 is stated.
  - Synthetic test: 45–57 mas injected shifts are recovered to within 1.1 mas.
  - Caveat: the shifts are relative to cross-band references, so errors in mutually referencing bands can drift by about 19 mas.
- **Launched p1reg** (`p1_lmc_register_only`, R = 0.2″): it measures the LMC shift distribution only. If the shifts are 25–55 mas, the next run is P1b, i.e. P1 with coarse registration, followed by the transfer again.

## 8 Nov 06:10 UTC: P1 training done; transfer to SMACS launched

**p1_lmc_tilefit_e2_s0:**
- Budget reached at cycle 1220 (about 150 cycles per training tile). The best validation snapshot is from cycle 812; validation has been flat since (robust S/N>5 17.4).
- Training χ²/pix: about 4.8–5.2 on the dense tiles, about 0.82 on the thin ones.
- **Held-out T1 so far** (the evaluation is still running, about 50 min per tile):
  - tile 0: S/N>5 89.8 [71.9, 107.4], all pixels 10.8;
  - tile 1: S/N>5 101.7, all pixels 11.4.
  - These are far above training. The coadd scores (`--score-coadds`) are needed before judging. Candidates: crowding, held-out affine refit in dense fields, under-training.
- **Learned tables differ from the SMACS ones:**
  - NRCB1/F150W EE(1, 2, 4 detector px) = 0.36, 0.69, 1.00: broad, not the near-delta of the SMACS E2 tables.
  - NRCBLONG/F444W: 0.84, 0.93, 0.99.
  - With many stars, the gauge is pulled toward the true PSF, as the memo predicted.

**Transfer launched without waiting for the evaluation:**
- `instrument_state_from_snapshot.pt` was built from `instrument_best_snapshot.pt`. The (detector, band) order comes from the run log; band order F090W..F444W and oversample 4 are assumed from the E2 preset.
- Runs, both with the `v38_E2data_jwst_dump` args:
  - **p1tfz** (`p1_transfer_frozen`, `--freeze-tables`);
  - **p1tff** (`p1_transfer_finetune`, `--psf-lr 2e-3`).
- **Gate** (memo): NIRCam S/N>5 ≤ 2.26; falsified if frozen > 2.40 while fine-tune recovers. Reference: E2-data 2.20 [2.14, 2.26].

## 7 Nov 22:10 UTC: P1 health check

- **p1tilea is healthy.** All 10 tiles loaded; the self-check passed (inner sky steps leave the instrument unchanged; checkpointed gradients match); no OOM.
- **About 17 s per cycle** (1 tile, K=2), so 420 min gives about 1500 cycles, about 190 per training tile.
- **χ²/pix** on the dense tiles falls slowly: 5.69 → 5.43 in 80 cycles; the thin tiles are at 1.3–1.5. It will likely be under-converged at the end of the budget; a chained resume run is possible.

## 7 Nov 21:30 UTC: P1 launched (LMC multi-tile, arm A = E2-data tables)

- **Builds (Sonnet, CPU tests pass, nothing yet on GPU or real data):**
  - `plan_tiles.py`;
  - `tilefit.py`: shared instrument across tiles, sky per tile, K inner sky steps per cycle, T1 and T5 with the `--ht-control` control;
  - `smacs_four_v40.py`: `--load-tables-from`, `--freeze-tables`; bit-identical to v39 without them.
  - Arm B (`--operator opfield`) is only an extension point.
  - Approximations A1–A11 are in the `tilefit.py` docstring.
- **LMC plan** (`/arc/projects/ots/adass2026/p1/plan_lmc.json`):
  - 1032 files, 10 detectors (modules A and B);
  - 10 tiles of 20.48″, covering all 90 (detector, 3×3 cell) positions and 267 of 540 (detector, band, cell);
  - F090W coverage is thin (4–8 exposures per tile).
- **Run p1tilea** (`p1_lmc_tilefit_e2_s0`):
  - 8 training tiles; tiles 5 and 9 held out (T5);
  - 0.04″, 3″ window, budget 420 min, inner sky 2, spec-k 12, Huber;
  - GPU slice: H100 MIG 1g.12gb.
- **Next, scheduled:** frozen and fine-tuned transfer to SMACS (v40) on the `v38_E2data_jwst_dump` split.

## 7 Nov 20:40 UTC: wing options (opfield v5, Λ = J), and the LMC download is done

**Held-out S/N>5 against the Λ = J baselines** (JWST 3.40; B NIRCam 3.04, WFC3/IR 3.40):

| Run | JWST-only / NIRCam | WFC3/IR | All pixels |
|---|---|---|---|
| log shape | 3.00 [2.64, 3.34] | – | 1.49 |
| **importance sampling** | **2.24** [2.15, 2.32] | – | 1.37 |
| two-scale | 4.05 | – | 1.77 |
| **all three** | **2.23** [2.17, 2.28] | – | 1.36 |
| B, two-scale | 3.23 | 3.64 | – |
| **B, all three** | 3.20 [3.01, 3.40] | **2.33** [1.98, 2.67] | 0.96 (WFC3/IR) |

- **Importance sampling is the lever.** The JWST-only field operator now ties E2-data (2.20) and is still behind E1 (1.95).
- **On B, WFC3/IR is the best by far:** 2.33, against E2-data 5.62 and the coadd about 12.
- **NIRCam with HST is still degraded** (3.20, against 2.23 without HST).

**`compare_operators`, all three against E2-data, same split:**
- **JWST-only:**
  - all pixels 1.356 against 1.444 (operator better);
  - shoulders 6.0% against 7.9% of χ² (better);
  - wings 7.1% against 5.8% (slightly worse).
- **B:**
  - all pixels 1.478 against 1.452;
  - WFC3/IR better in every band (shoulders −0.10 to −0.20 of E2's χ²);
  - NIRCam slightly worse in the F200W and F356W shoulders.

**LMC download:** done at 17:46 UTC, 1032 files (1476 + 4447, SMACS bands) in `/arc/projects/ots/torchsky-data/aggregate/{1476,4447}_nircam/`.

## 7 Nov 14:30 UTC: formulation doc, aggregate inventory, LMC download

- **Formulation doc** ("One sky, many instruments: formulation"): https://claude.ai/artifact/Nj91AWRxgJf94mZyifAc56. It covers:
  - the model and the operator family (shared trunk, instrument tokens, detector embeddings, core and wing tables per (detector, band), hierarchical codes);
  - gauges, plus what diversity breaks;
  - learning by variable projection with cross-fitting;
  - scale (tiles, pixel store, workers), data and curriculum, the gate tests, phases P0–P5, and 5 decisions for Seb.
- **MAST inventory** (`/arc/projects/ots/torchsky-data/aggregate/aggregate_inventory.{csv,json}`):

| Programme | Files | GB |
|---|---|---|
| 1476 | 2268 | 267 |
| 4447 | 1628 | 191 |
| 1501 | 200 | 24 |
| 4471 | 100 | 12 |
| 1515 | 48 | 6 |
| 4475 | 24 | 3 |
| 1334 | 264 | 31 |
| 1837 (PRIMER) | 14740 | 1615 |
| ω Cen WFC3 (11928, 12340, 12353, 12694, 12714, 17974) | 54–810 each | 3–37 each |

- In the six SMACS bands, 1476 + 4447 NIRCam come to **1032 CAL files, 121 GB** after exact filter matching. A substring bug that would have pulled F150W2 combos was fixed first.
- **Download launched:** aggfetch, into `/arc/projects/ots/torchsky-data/aggregate/{1476,4447}_nircam/`.

## 7 Nov 13:50 UTC: Λ = J gauge results, E1 halo 64, wing builds launched, design memo

**Λ = J gauge, against the learned-Λ Huber runs:**

| Run | NIRCam S/N>5 | WFC3/IR S/N>5 |
|---|---|---|
| JWST-only, jac | 3.40 [3.19, 3.60] (was 2.81) | – |
| JWST-only, jac + band | 3.28 | – |
| B, jac | **3.04** [2.79, 3.30] (was 3.63) | **3.40** [3.20, 3.61] (was 4.14; best WFC3/IR yet) |
| B, jac + band | 3.24 | 5.06 |

- The results are mixed and within the large run-to-run spread; there are no seeds.
- **Per-detector flux exp(a) under the gauge:** JWST-only SW is 0.76–1.00, against 0.78–1.94 before. LW stays at about 0.6.
- **Correction to my earlier claim:** a per BAND is degenerate with the spectral normalisation of the sky. Only ratios between detectors within one band are a test. For F200W these are 0.94 / 0.86 / 0.78 / 0.76, a 20% spread. Real detector calibration differences are a few percent, so shape compensation remains.

**E1 with halo 64** (`v38_E1_jwst_halo64_dump`, same masks as E2 and opfield):
- 1.95 [1.84, 2.05].
- LW bands: F277W 2.08, F356W 1.78, F444W 1.40, against E2-data's 3.44 / 2.52 / 2.90.
- **E1's LW advantage is real, not a mask artefact.** Candidate explanation, untested: E2's free LW tables (189² per band, one detector) overfit, while E1's ChromaticPSF network is smoother.

**Wing builds (Sonnet), opfield flags:**
- `--psi-param log`;
- `--importance-sampling` (annuli around bright peaks of the current model; unbiased reweighting);
- `--two-scale R_core` (per-pixel core plus an FFT wing table per detector and band, log-parametrised, with a learned wing fraction).

Synthetic wing check:
- **Passes only for two-scale (12 px = 6σ core) + IS:** 3–15 px χ² 5.1 against 11.4 for the default.
- **Fails at 6 px (3σ).**
- **IS without the bright-pixel bias worsens the core.**

**Launched** (Λ = J, batch 8192):
- JWST-only: wja (log), wjb (IS), wjc (two-scale 0.4″), wjd (all three);
- B: wbc (two-scale), wbd (all three).

**Design memo** ("learn the operator from many pointings in aggregate", Opus agent): `claude/operator-aggregate-design-memo.md`.
- Recommended first experiment: share per-(detector, band) tables across 8 tiles of the LMC NIRCam calibration field (PID 1476 / 4447; 9-point dithers, 16 filter pairs), each tile with its own catalogue-free sky. Then freeze the tables and transfer to SMACS, scored on the E2 split.
- Success criteria:
  - NIRCam ≤ 2.26;
  - shoulder plus wing χ² share ≤ 11%;
  - shoulder deficit below 3%.
- Falsified if frozen transfer scores above 2.40 while fine-tuning recovers.

## 7 Nov 10:10 UTC: same-split comparison with HST (JWST + WFC3/IR 14096 + 16729)

E2-data reference `v38_E2data_jwst_wfc3ir1729_dump`, S/N>5:
- NIRCam 2.33. WFC3/IR 5.62 (all pixels 1.36).
- The operator (Huber) has NIRCam 3.63 and **WFC3/IR 4.14** on the same exposures.

`compare_operators.py`, operator against E2-data:
- **All pixels:** 1.65 against 1.45.
- **Share of χ² by region:**
  - core: 3.2% against 1.4%;
  - shoulders: 16.9% against 10.4%;
  - wings: 11.0% against 7.0%;
  - non-peak: 62% against 73%.
- **Per band:** NIRCam loses in the shoulders (F200W +0.14, F356W +0.13, F444W +0.17 of E2's χ²) and in the wings (F090W, F150W). WFC3/IR is mixed: cores worse in F105W and F125W, better in F140W.
- **On WFC3/IR bright pixels the operator wins** (4.14 against 5.62). On NIRCam it loses the bright shoulders and wings.

## 7 Nov 08:40 UTC: opfield ablations (JWST-only) and same-split comparison

**Ablations** (one change each on top of the Huber run's args; baseline 2.81 [2.62, 2.97] at batch 8192):

| Ablation | S/N>5 |
|---|---|
| batch 4096 (control) | 3.65 [3.35, 3.95] |
| base per detector and band, base_l2 0 | 3.25 |
| shape head per channel | 2.98 |
| 72 wing azimuths | 2.90 |
| **base table per detector and band** (batch 4096) | **2.71 [2.51, 2.90]** |

- Run-to-run variation is large. The batch-4096 control is 0.8 worse than the batch-8192 baseline, and there are no seeds.
- Only the base table stands out: it beats its own control by 0.9 and is the best field operator so far. It is still behind E2-data on the same split: 2.20 [2.14, 2.26] (`v38_E2data_jwst_dump`).

**`compare_operators.py`, operator (Huber eval) against E2-data, same 24 held-out exposures:**
- **All pixels:** 1.50 against 1.44 χ²/pixel.
- **Share of χ² by region:**
  - shoulders (1–3 px from model peaks): 11.9% against 7.8%;
  - wings (3–15 px): 9.3% against 5.7%;
  - non-peak areas (more than 30 px from peaks): 70% against 79%. The operator is slightly better there in F090W and F444W.
- **Band with the largest loss:** shoulders in F150W and F200W; wings in F277W and F356W.
- **Conclusion:** the gap is localised to the shoulders and wings of bright sources. This is consistent with the shrunk Λ frame and the broad halos. The Λ = J gauge runs test this directly.

## 7 Nov 06:50 UTC: v39 chromatic tables, T3 (fixed), mask confound

**v39 chromatic tables** (S/N>5 per band, then the bootstrap):

| Run | F090W | F150W | F200W | F277W | F356W | F444W | NIRCam | WFC3/IR |
|---|---|---|---|---|---|---|---|---|
| JWST-only, nodes | 2.25 | 2.15 | 1.57 | 3.24 | 3.29 | 2.89 | 2.19 [2.11, 2.27] | – |
| JWST-only, endpoints | 2.34 | 2.30 | 1.79 | 3.22 | 3.22 | 2.10 | 2.26 [2.14, 2.37] | – |
| B, endpoints | – | – | – | – | – | – | 2.37 | 10.18 [8.80, 11.40] |

- Against E2-data (2.24 JWST-only; 5.53 WFC3/IR on B), there is no gain, and B gets worse on WFC3/IR.

**T3b** (`make_holdout` fixed; score is held-out F105W at S/N>5, against the coadd):
- 2022 held out, 2017 trains: 2.33 [2.05, 2.58] against 4.98. This direction is clean: 2022 has only F105W, so the epoch is never seen.
- 2017 held out: 7.41 against 13.9. The other 2017 bands remain in training.
- Null stacks on WFC3/IR held-out pixels: noise factor 0.65–0.82 (the pipeline σ is too high for IR); fractional floor 14–21%.

**Mask confound:** the S/N>5 masks depend on `--halo-px`.
- E1 (halo 48) and E2 (halo 64) score different pixels per band. The drizzle F444W score is 2.73 under one and 4.38 under the other.
- Relative to its own coadd, E1 is still better in LW (about 0.43 against 0.70).
- Launched **e1h64** (`v38_E1_jwst_halo64_dump`): E1 with halo 64 and pixel dumps, for a like-for-like comparison with E2 and the field operator.

## 7 Nov 06:45 UTC: Λ gauge and within-band λ (Seb: build it)

**New opfield flags** (defaults bitwise unchanged):
- `--lambda-gauge jacobian`: Λ ≡ I, so M = J, the exposure's WCS Jacobian. This is a gauge choice for the redundancy between Λ scale and Ψ width, not a prior toward the calibration. Per-pixel Jacobian variation must then come from μ and Ψ.
- `--lam-in-band`: the node's position within its band as an input.
- The kernel flux a stays FREE. A new metrics key, `kernel_diagnostics`, records per (instrument, detector, band) the exp(a) distribution, EE at 0.1, 0.2 and 0.5″, and the singular values of Λ. a is used as a null test: it should return to about 1 if the shape is right.

**`compare_kernels.py` fixed** (true detector offsets J⁻¹·pos; a regression test added). On the JWST-only Huber run, operator against E2:
- Λ singular values are about 0.42 / 0.33 (SW) and 0.41 / 0.35 (LW).
- Core rms: operator 1.8–2.6 px against E2 0.8 (SW); 2.5 against 0.6 (LW).
- 17–35% of the operator's mass lies beyond 12 detector px; E2 has none beyond about 4 px.
- After the common-blur gauge, EE at 2 px is 0.81 against 0.91 (SW) and 0.75 against 0.84 (LW).
- E2 has an LW m=2 elongation that the operator lacks.

**Launched:**
- opjac: `opfield_v4_jwst_jac`;
- opjacb: `opfield_v4_jwst_jac_band`;
- opwjac: `opfield_v4_wfc3ir_jac`;
- opwjacb: `opfield_v4_wfc3ir_jac_band`.

## 7 Nov 03:40 UTC: field-operator diagnosis, first results (JWST-only Huber run, eval-only)

- Eval-only reproduces the training run's scores exactly: 2.8056 / 3.6613 / 4.1445.

**Null stacks** (opfield JWST-only, 24 held-out exposures). The reference is the E2+NIRISS run, which is not the same split; the same-split comparison will come from e2dumpa.
- Noise factor 1.69, against 1.29.
- Shoulder deficit around S/N>50 model peaks: +10.6 / +9.1 / +7.6% at 0.5 / 1.0 / 1.5 px, against +8.7 / +6.9 / +2.3%.
- Position maps are dominated by bright-star bins: NRCB3 χ²/dof 602 against 92, NRCB4 368 against 73. The 20″ box covers only one corner of each SW detector, so focal-plane variation is barely sampled at all.

**Kernels** (from `kernels.json`, sky-frame `pos`):
- **The learned local frame Λ shrinks the stencil.** M = J·Λ has scale about 0.29 target px per canonical px for SW, against 0.78 expected: Λ ≈ 0.37. For LW it is about 0.6 against 1.6. The dense core therefore spans only about ±0.11″ for SW.
- **Broad halos:**
  - SW: EE 0.33–0.52 within 0.1″, 0.41–0.60 within 0.2″, and 17–22% of the flux beyond 0.5″;
  - LW: EE about 0.21 within 0.1″ and 0.71–0.72 within 0.5″;
  - E2 SW tables are close to a delta function under the declared gauge.
- **Kernel sums exp(a):** 0.78–1.94 across the SW detectors of one band, and 0.55–0.68 for LW. This is unphysical for flux-calibrated `_cal` data. Aperture photometry is still fine (median −7 mmag, as for E2), so the sky compensates.
- **Conclusion:** beyond capacity, the operator's internal split into Λ (scale), Ψ (width/halo) and a (flux) is under-determined, and it drifts.

**Tool bug:** `compare_kernels.py` treats `vdet` (canonical stencil coordinates, scaled by Λ) as detector pixels, about 2.7× off for SW. Its EE and blur numbers are invalid; it needs fixing to use `pos` with the exposure Jacobian.

**Proposed (data-driven) constraints:**
- tie Λ to the WCS Jacobian, with deviations of ±10% at most;
- the flux a per instrument plus small per-detector deviations (a flat-field), with a gauge per instrument;
- base tables per detector and band (ablation btab, running);
- the node's position within its band as an input (within-band λ).

## 7 Nov 02:20 UTC: field-operator diagnosis and instrument-specific ablations (Seb's request)

**New opfield flags** (Sonnet; defaults unchanged and bitwise-equal; 21 unit tests pass):
- diagnostics: `--eval-only --init-state --args-from --dump-heldout-pixels`;
- instrument-specific options:
  - `--base-per {channel,instrument,detector,detector_band}`, `--base-l2`;
  - `--shape-head-per {shared,channel}`;
  - `--wings {logpolar,cartesian}`, `--wing-nang`;
  - `--base-table detector_band` (free table per detector and band, plus rank-8 modulation by the hypernetwork);
  - `--base-init-from E2_RUN`.

**New scripts:**
- `compare_operators.py`: χ² by S/N; radial and azimuthal residuals around model peaks; position and readout maps; χ² share in core, shoulders, wings and the rest.
- `compare_kernels.py`: the operator's kernels against the E2 tables after removing a common blur per channel; EE; m=2 and m=6 azimuthal power.

**Audit facts:**
- The base is one kernel per channel, shared by 4–5 detectors and 3 bands.
- The bank (32 kernels) and the shape MLP are shared by all instruments.
- Wings have 24 azimuths. The base L2 is weak (1e-4).

**Launched:**
- eval-only with dumps of the two Huber runs: opevala, opevalb;
- E2-data references with dumps on the same splits: e2dumpa (JWST-only), e2dumpb (JWST + WFC3/IR 14096 + 16729);
- JWST-only ablations, one change each against the Huber run's args:
  - ablb4096: batch 4096, the control for the batch change;
  - ablbasedb: base per detector and band, base L2 0;
  - ablheadch: shape head per channel;
  - ablnang72: 72 wing azimuths;
  - ablbtab: base table per detector and band, batch 4096.

## 7 Nov 00:45 UTC: morning report

**Seeds (B = JWST + WFC3/IR 14096; S/N>5; seed 0 from v34/v36):**

| Model | NIRCam | WFC3/IR |
|---|---|---|
| E1, seeds 0–2 | 2.38 / 2.37 / 2.21 → 2.32 ± 0.09 | 9.82 / 10.02 / 9.35 → 9.73 ± 0.34 |
| E2-data, seeds 0–2 | 2.32 / 2.26 / 2.25 → 2.28 ± 0.04 | 5.82 / 5.45 / 5.32 → 5.53 ± 0.26 |
| E2-lib, seeds 0–1 | 2.31 / 2.27 | 11.83 / 11.23 |

- The WFC3/IR ranking holds across seeds. NIRCam does not separate the models.

**NIRISS joint fit** (`v38_E2data_nircam_niriss_2022`, NIRCam B + NIRISS 2736, same epoch):
- NIRCam: 2.19 [2.13, 2.26], against E2-data JWST-only 2.24.
- NIRISS: 2.46 [2.44, 2.49], against the coadd's 2.92.

**Null stacks** (`null_stacks.py` on that run; 28 held-out exposures, 3.0 M pixels):
- **Detector-position maps:** structure beyond the shuffle null for all detectors (perm p < 0.005, except NRCB2 at p = 0.035).
- **Readout direction:** NIRISS columns χ²/dof 131 (1/f is not modelled for NIRISS). NIRCam rows are weaker.
- **Per-exposure means:** χ²/dof 149 over 28 exposures, from per-exposure offsets. Mean z is +0.04.
- **Fractional residual against model S/N:**
  - noise factor 1.29 (pipeline σ low by about 29%);
  - floor f = 5.5%;
  - at S/N 1–3 the model exceeds the data by 3–6% (faint bias).
- **Peaks** (9 at S/N > 50): data exceed the model by 7–9% at 0.5–1.5 px from the peak, so the kernels are too peaked.

**T3 runs (`v38_T3_*`) were invalid:**
- `make_holdout.py` used the manifest's 8-character "root" column, so the holdout matched no exposure.
- Fixed to use obs_id (9 characters).
- Relaunched as t3fwdb (`v38_T3b_hst2022_heldout`) and t3revb (`v38_T3b_hst2017_heldout`).

**Still running:** v39 chrna, chrea, chreb (started about 21:30–22:00; 420 min budgets).

## 6 Nov 21:50 UTC: field operator (Huber), JWST+WFC3/IR

- **opfield_v3_jwst_wfc3ir_huber_cshift:** Huber loss, grid-scan coarse shift, HST 14096 + 16729, no xfit, skip or restart.
- **Held-out S/N>5:**
  - NIRCam: **3.63 [3.37, 3.89]**, against 2.81 for JWST only.
  - WFC3/IR: **4.14** (robust 1.91), against E2-data B 5.82 and coadd about 12.
  - All pixels: 1.65.
- **WFC3/IR is the best so far.** Caveat: this run includes the 2022 HST data (E2-data B was 2017 only), so the auto-held-out WFC3/IR exposures may differ.
- **Adding HST costs NIRCam about 0.8 here,** against about 0.1 for E2-data.

## 6 Nov 21:30 UTC: v39 chromatic PSF tables (Sonnet) launched

- **New flag `--lib-chromatic {none,endpoints,nodes}`.** Bit-identical to v38 without it.
  - nodes: one free table per (detector, band, node), with a smoothness penalty across nodes.
  - endpoints: two tables per (detector, band), interpolated linearly in λ.
  - No λ/D physics is imposed.
- **Parameters:** 1.9 M (achromatic), 3.8 M (endpoints), 11.4 M (nodes).
- **Synthetic kernel fit with σ ∝ λ (+23%), relative error:** none 9.7e-3, endpoints 6.4e-5, nodes 1.3e-7. This checks capacity only.
- **Runs:**
  - chrna: `v39_E2chrom_nodes_jwst`.
  - chrea: `v39_E2chrom_endpoints_jwst`.
  - chreb: `v39_E2chrom_endpoints_jwst_wfc3ir`.
- **Compare with** E2-data (JWST-only 2.24; LW 3.0–3.5) and E1 (1.92; LW 1.2–2.0).

## 6 Nov 21:05 UTC: Huber ablation of the field operator (JWST-only)

- **opfield_v3_jwst_cal_huber** (Huber δ = 3, noise head off, no outlier mixture; core r10; stopped early at 218 min on patience).
  - Held-out S/N>5: **2.81 [2.62, 2.97]**, against 9.81 with the mixture loss.
  - All pixels: 1.50. Robust S/N>5: 1.78.
  - The outlier mixture was the main cause of the real-data failure.
- **Per band, S/N>5:**

| Band | Field operator | E2-data | E1 |
|---|---|---|---|
| F090W | 2.42 | 2.31 | 2.36 |
| F150W | 2.87 | 2.22 | 2.19 |
| F200W | 2.20 | 1.62 | 1.64 |
| F277W | 4.09 | 3.51 | 2.05 |
| F356W | 4.51 | 2.95 | 1.72 |
| F444W | 3.22 | 3.03 | 1.23 |

- **The gap to E1 is mostly in the LW bands, and E2-data shares it.**
  - E1's ChromaticPSF evaluates the kernel at every quadrature wavelength. E2 tables are achromatic within a band (a stated approximation). The F444W PSF width changes by about 25% across the band.
  - Lead for E2: λ-dependent tables within a band.
  - CORRECTION (checked in code after the first message to Seb): the field operator's shape Ψ DOES depend on the node wavelength (its `lamf` features). So its LW gap has another cause, not yet diagnosed. Candidates: wing sampling (log-polar, 15° azimuths), bank capacity, early stop.
- **Decision rule set before leaving:** the field operator stays the main track only if JWST-only ≤ 2.5. At 2.81 that is not met.
  - The factorised model carries the real-data work.
  - The field operator stays a parallel track. Its SW bands are on par with the factorised models, and its LW gap is not the achromatic-table issue (see the correction above).
  - The B Huber run with coarse shift is still running.

## 6 Nov 16:30 UTC: overnight queue (Seb away 8–10 h)

- **v37:** `--seed`, otherwise bit-identical to v36. GPU kernels stay non-deterministic.
- **Seed repeats on B** (JWST + WFC3/IR 14096):
  - E2-data: seeds 1 and 2 (e2ds1, e2ds2);
  - E1: seeds 1 and 2 (e1s1, e1s2);
  - E2-lib: seed 1 (e2ls1).
- **v38 (Sonnet), bit-identical to v37 without its flags.** New flags:
  - `--extra-jwst-dirs`: NIRISS, NIRCam module A, medium bands; MIRI only with `--include-miri`;
  - `--freeze-sky-from` and `--freeze-instruments-from`: T4 transfer;
  - `--dump-heldout-pixels`: per-exposure npz files, fixing the old coords bug.
- **New scripts:**
  - `make_holdout.py`, splits T1–T4. T2 needs PA_V3 from the FITS headers.
  - `null_stacks.py`, T7. Synthetic checks: noise-only χ²/dof 0.93; a 3% error floor recovered as 3.2%.
- **v38 approximation flagged by Sonnet:** registry pivot wavelengths, pixel scales and FWHM starts for NIRISS, NIRCam medium bands and MIRI were typed from memory, to about 1%. Check them against JDox.
- **Launched (E2-data config + `--dump-heldout-pixels`):**
  - **t3fwd:** `v38_T3_hst2022_heldout`. The 6 F105W exposures of 2022 are held out, and 2017 trains.
  - **t3rev:** `v38_T3_hst2017_heldout`. The 20 F105W exposures of 2017 are held out.
  - In both, only F105W is common to the two epochs. Bands with no counterpart stay in training, so the held-out epoch is partly seen through its other bands.
  - **nirissj:** `v38_E2data_nircam_niriss_2022`. NIRCam B plus NIRISS 2736 (same epoch), with auto held-out NIRISS (2 per band).
- **GPU queue is saturated.** 9 runs are pending behind other users' jobs, so not all will finish by morning.

## 6 Nov 15:30 UTC: field operator real runs (v2) and recap

Held-out S/N>5 (all pixels):
- **opfield_v2_jwst_cal_core10:** 9.81 (2.85). The wider core did not help (v1 was 9.87). It stopped early at 223 min.
- **JWST+WFC3/IR with xfit, mu-skip and restart:**

| Run | NIRCam | WFC3/IR |
|---|---|---|
| z-hier, seed 0 | 597 | 160 |
| z-hier, seed 1 | 1015 | 139 |
| flat z, seed 0 | 372 | 161 |

- Validation got worse from 129 min on, before the restart. The best checkpoint was restored, but the run had already degraded.

Hypothesis: the outlier mixture (π ≤ 0.2, uniform over 200σ) treats misfit bright cores as outliers, so their gradient is near zero. The learned NLL is good while the pipeline χ² is bad.

Ablation launched: `--loss huber --huber-delta 3 --noise-start 2.0`, so the noise head is off and the loss is Huber as in v36.
- opfah (xo3zj4wz): opfield_v3_jwst_cal_huber.
- opfbhub (a4r3vesm): opfield_v3_jwst_wfc3ir_huber_cshift, using the v36 grid-scan coarse shift, with no xfit, skip or restart.

**Recap doc:** https://claude.ai/artifact/HDJ7Wvmi86Wc4SB9HrwkJx, with a copy at `claude/continuous-sky-recap-2026-11-06.md`.

## 6 Nov 08:50 UTC: learned PSF tables vs handbook (E2, JWST+WFC3/IR, 0.04″)

Encircled energy in the detector frame at 0.13, 0.26, 0.39 and 0.52″. Core second moment m2 within r < 0.4″, in arcsec².

**E2-data (generic σ = 1 px start):**
- NIRCam SW tables collapse to about a delta function: m2 0.0012–0.0015, against 0.0019 at the start. LW tables also narrow, from 0.0079 to about 0.0045.
- WFC3/IR tables broaden from the start but stay much narrower than the handbook. F160W EE is 0.73, 0.97, 0.98, 0.98, against the handbook's 0.49, 0.74, 0.83, 0.86. F105W at 0.13″ is 0.96 against 0.58.
- The λ trend is right: the F105W table is the sharpest relative to its PSF.
- This is the declared gauge, the sky at NIRCam SW resolution. The WFC3/IR tables are relative kernels (HST ⊗ SW⁻¹), not instrument PSFs.

**E2-lib (STPSF and STDPSF start):** every table sharpens from its start, so the sky absorbs a common blur.
- Core m2 changes by −0.001 to −0.006 for SW, −0.009 to −0.012 for LW, and −0.005 to −0.010 for WFC3/IR.
- The drift is λ-dependent, consistent with a gauge g(λ) carried by the spectral sky.
- The physics start does not hold the PSF at the physical truth, as in the synthetic test.
- It also predicts WFC3/IR worse (11.8 against 5.8). Two gauge-equivalent solutions would predict equally, so this is a worse optimisation basin within the budget, not just a different gauge.

## 6 Nov 07:40 UTC: E2-data vs E2-lib, JWST+WFC3/IR at 0.04″ (finished)

Held-out S/N>5 (all pixels), bootstrap 68%:

| Run | NIRCam | WFC3/IR |
|---|---|---|
| E2-data, generic Gaussian start | 2.32 [2.23, 2.41] (1.46) | **5.82 [5.05, 6.50]** (1.38) |
| E2-lib, physics-library start | 2.31 [2.16, 2.46] (1.48) | 11.83 [10.50, 13.05] (1.99) |
| E1, ChromaticPSF + halo | 2.38 [2.21, 2.56] (1.40) | 9.82 [8.98, 10.54] (1.81) |
| coadd | 2.93 | 12.2 |

- **Learned per-detector tables from a generic start are the best model on WFC3/IR so far,** 5.82 against the coadd's 12.2.
- **Gate within the E2-data model:** NIRCam with HST is 2.32, against 2.24 [2.18, 2.30] JWST-only. The gap is small and the intervals nearly touch.
- **E1's ChromaticPSF is still better on JWST alone** (1.92).
- **The physics-library start is clearly worse on WFC3/IR** (11.8 against 5.8), with the same model and budget. This is one run each.

## 6 Nov 08:00 UTC: E1 result (edge-ring bug), first field-operator results

**E1 (v34, sky halo; same model as v33 A0/A1).** Held-out ⟨r²/σ²⟩, bootstrap 68%:

| | v33 (ring bug) | v34 E1 (halo) | coadd (v34) |
|---|---|---|---|
| JWST only, S/N>5 | 2.31 [2.19, 2.42] | **1.92 [1.84, 2.01]** | 2.90 |
| JWST only, all pixels | 5.20 | **1.35** | 1.48 |
| JWST+WFC3/IR, NIRCam S/N>5 | 3.34 [3.10, 3.59] | **2.38 [2.21, 2.56]** | 2.90 |
| JWST+WFC3/IR, WFC3/IR S/N>5 | 14.2 [13.0, 15.2] | **9.82 [8.98, 10.54]** | 12.8 |

- **The all-pixel χ² of every earlier run was dominated by the unfittable ring**, including the coadd baselines, the Q5 numbers and the LW "noise problem" (8–15). Re-read old all-pixel numbers with that in mind; the S/N>5 metric excluded the ring but training did not.
- **The gate still fails, but the gap shrank.** NIRCam with HST is 2.38 against 1.92 without (it was 3.34 against 2.31). WFC3/IR now beats the coadd.

**E2-data (v36, generic learnable tables, one per detector and band).**
- JWST-only at 0.04″: 2.24 [2.18, 2.30], worse than E1's 1.92.
- At 0.02″ with a 2″ window (B): NIRCam 4.63, WFC3/IR 16.6. Its bright masks come from a different coadd (drizzle 3.16 there), but it is clearly worse.
- E2-data B and E2-lib B at 0.04″ are still running.

**Field operator (opfield.py, Sonnet). Synthetic, independent renderer:**
- Without large shifts (3 instruments, rolls) it reaches near-noise χ² (2.3–3.0 against 1.0), with registration at 1–6 mas. That is 10× better than the factorised baseline in the same harness (Ψ without λ or position dependence).
- Shifts larger than about half a PSF core lead to a joint sky/registration local minimum. Cross-fit plus mu-skip brings the 0.6″ instrument to about 150 mas, not solved.
- Time synthetic: an unseen epoch (T3) is barely worse than held-out dithers. A restart cycle halves χ². z-hier gives no gain (single seed). Proper motion is not absorbed by the coherent μ, and the bright cores are the dominant error.

**Field operator, real data. opfield_v1_jwst_cal (JWST only, 240 min, stopped early at 166 min):**
- S/N>5: 9.87 [9.37, 10.34]. All pixels: 2.82. Far worse than E1.
- Cause found: the canonical core was r ≤ 3 detector px, i.e. ±0.09″ for SW and ±0.19″ for LW. Everything beyond that is sparse log-polar wings reading a pre-smoothed sky, so the PSF cores cannot be represented.
- **Relaunched** with core r ≤ 10 detector px at pitch 0.5 (SW ±0.31″ at 0.016″; LW ±0.63″; WFC3/IR ±1.28″ at 0.064″), batch 8192:
  - opfa2: JWST only.
  - opfbh0 / opfbh1: JWST + WFC3/IR 2017 + 2022, z-hier, seeds 0 and 1.
  - opfbf0: flat z, seed 0.
  - All with xfit 2, mu-skip ×8 and restart 1.
- The first launch of the B runs (wrong stencil) was killed after a few minutes. Their session.log files are appended in the same run directories.

## 6 Nov 01:30 UTC check-in

- **e2datab2** (0.02″, 3″ window) died of CUDA OOM: the GPU slice has 10.75 GiB. Relaunched as e2datab3 (o9r9224a), `v36_E2data_jwst_wfc3ir_s02_w2`, with a 2″ window and halo 100.
- **All other runs are stepping:**
  - E1 A: 6000 steps. E1 B: 3300 steps.
  - E2-data A: 1500 steps at about 2.8 s/step. E2-data B: about 4.5 s/step.
- **Early validation at about 1 h, S/N>5 pixels (same validation exposure as v33):**
  - E2-data B: 4.33.
  - E2-lib B: 8.36.
  - For reference, v33 A0 B was 8.58 at 0.5 h and 5.38 at 4.7 h.
  - The physics-library start is behind the generic Gaussian start so far. This is early and not yet a test result.

## 6 Nov (AWST ~19:00): E1, E2-data and E2-lib launched; common-mode gauge

- **Seb's steer.** Data-driven by diversity, data splitting and empirical tests; no reliance on physics or sim-to-real. The physics library is a control only.
- **Results from earlier runs:**
  - Epoch test: JWST + 2022 WFC3/IR F105W (6 exposures). NIRCam 2.32 [2.18, 2.46], no degradation; WFC3/IR 6.07 vs coadd 9.27. Confounded by fewer HST exposures.
  - IRAC full run, held-out bright pixels: learned 20.1 vs structured 25.5. IRAC1 15.0 vs 27.7; IRAC2 24.6 vs 23.7. Only 9 of the 150 minutes were used. The bright-pixel sets differ by model.
- **Builds:**
  - v34 (halo, E1): equivalence tests pass.
  - v35 (LibPSF, physics library) and v36 (learnable generic tables), both by Sonnet: bit-identical without the new flags.
  - Synthetic: the common-mode gauge is confirmed. Absolute kernels are off by one shared g; relative kernels and held-out predictions are right. See the audit addendum.
- **Runs:**
  - e1ajw (zg46vjfn), e1bjwhst (ask8fufs).
  - e2dataa (jrhajeps), e2datab (myyqojnu), e2datab2 (mcy5l47x, 0.02″), e2libb (fj9kwf88).
  - Budget 420 min each.

## 6 Nov (AWST ~14:00): PSF/geometry audit (no new runs)

Full write-up: `claude/psf-geometry-audit.md`.

- **The learned PSF is a relative kernel, not the instrument PSF.**
  - WFC3/IR learned EE(0.13″) is 0.90–0.99, against 0.49–0.58 in the handbook.
  - NIRCam SW kernels are about delta functions.
  - The fitted sky has NIRCam SW drizzle resolution: it is not deconvolved.
  - Causes: drizzle initialisation, a narrow Gaussian PSF warm start, a 1.24″ window, residual kernels multiplied by 0.01, and one λ/D-scaled network shared by all telescopes.
  - With two instruments, one sky cannot hold both instruments' wings, which explains the gate failure.
- **Bug: unfittable data ring.** Data pixels are kept in a 24-px ring outside the sky grid and predicted as background only: about 16% of the pixels of a fully covering exposure. The bright stars sit 0.9–1.9″ from the grid edges.
- **The B run was still improving when its 300-min budget ended.**
- **Literature review** (subagent; arXiv IDs 2512.16764 and 2609.04312 spot-checked):
  - ePSF practice: ≥ 4× supersampling, spatial grids over the detector, an HST focus manifold, a JWST term per wavefront-sensing epoch.
  - A prior or anchor is needed to break the PSF–scene degeneracy (STARRED, ALPACA).
  - WaveDiff and dLux for differentiable optics plus a data-driven residual.
  - The full HST distortion chain plus small learned residuals.
  - A-priori HST pointing errors of 0.1–0.75″ are normal.
  - No published catalogue-free neural field learns PSF and distortion jointly.
- **Proposed ladder:**
  - E1: data-ring fix.
  - E2: physics warm start, per-instrument PSF fields, 3″ window, sky at 0.02″ initialised by deconvolution.
  - E3: epoch/focus code per exposure.
  - E4: full-detector tiles.
  - E5: A2 continuous operator.

## 6 Nov (AWST ~00:30): A0/A1 result, HST registration, learned IRAC track

**A0 + A1 (detector-frame PSF rotated per roll group, detector pixel box) did not fix the gate:**

| Held-out bright pixels | sky-frame PSF (v30) | detector frame + pixel box (v33) |
|---|---|---|
| JWST only, NIRCam | 2.30 [2.20, 2.40] | 2.31 [2.19, 2.42] |
| JWST + WFC3/IR, NIRCam | 2.82 | 3.34 |
| JWST + WFC3/IR, WFC3/IR | 10.4 | 14.2 |
| WFC3/IR only | 77.8 | 70.2 (all pixels 5.11 → 4.67) |

- Roll and undersampling are not the main cause.
- HST-only runs fail because visits are misregistered relative to each other by up to ~0.6″ in the a-priori WCS: B's in-model coarse shifts per visit range from (−434, 487) to (+242, 323) mas, consistent within a visit (5–12 mas scatter). C had no anchor band, so no shift. v31 only shifts HST against NIRCam; an HST-only reference is still to add.
- With registration fine in B, the NIRCam degradation (worst in F150W and the LW bands, near the grid edges where the four bright stars sit) points to epoch differences (2017 vs 2022: proper motions, variability) or to stellar cores. Gaia (ESA archive overloaded, VizieR blocked through the proxy) could not be queried to check the proper motions.
- Epoch test launched: JWST + WFC3/IR from 2022 only (program 16729 F105W, 6 exposures), session bjv1hw6n.

**Fully learned instrument, IRAC (Sonnet, reviewed):**
- Model: `learned_instrument.py`. Frozen sky = NIRCam LW coadd at 0.12″ (`jwst_lw_sky.py`; the NIRCam PSF is negligible against IRAC's).
- Learned model: a linear patch operator (25×25 samples at 0.3″ in the detector frame, weights from a hypernetwork of detector position, channel, epoch and per-frame code; unit total weight), a gated nonlinear residual g, a per-frame offset and a learned noise head.
- Structured baseline: a 4-Gaussian PSF field plus the 1.2″ pixel box, a per-frame gain and offset.
- Synthetic test: learned 1.04 vs structured 14.2 on held-out frames (soft saturation present).
- Real mini-run (CPU, 168 frames): bright-pixel χ² 16.6 vs 23.2; aperture flux rms 0.26 vs 0.41 (9 sources).
- Full run on CANFAR: `iracli` (g2mpa593), all 3486 frames, budget 150 min per model, outputs in `/arc/projects/ots/adass2026/learned_instrument/irac_li_v1/`.
- Spitzer pixel-store extraction: psspz0/1.


## 5 Nov evening (AWST): operator spec, A0/A1, splat and step 1b results

- **Forward-operator spec** (`claude/forward-operator-spec.md`): d = N_i(A_i[s]) + B_i + ε; A_i linear but shift-variant (geometry T_i with Jacobian, detector-frame PSF field, pixel box and intra-pixel response; not a convolution), N_i nonlinear detector (response curve and saturation, brighter-fatter/IPC, charge-transfer trails with epoch, persistence, residual), gauges and identifiability tests, phases A0–A2, N1–N4.
- **HST DQ fix:** saturated pixels (bit 256) no longer "good" for ACS/UVIS (80; 96 before 2012). The pixel store keeps native DQ, so no re-extraction.
- **Roll angles:** WFC3/IR 14096 spans PA_V3 254–359° (6 rolls), ACS 4/74/254°; NIRCam 2736 one roll. Held-out WFC3/IR by roll in B was inconclusive (8 exposures; F160W 21.7 at 254°, 24.7 at 359°).
- **v33 (A0 + A1, Sonnet):** `--psf-frame detector` (PSF field in detector coordinates, rotated per exposure group by the starting-WCS Jacobian; one convolution per (band, roll) group), `--pixel-box` (detector-pixel box by S×S quadrature; S = 7 for WFC3/IR). Bit-identical to v32 without the flags; synthetic two-roll test: sky-frame kernel 3.1–3.3 on bright pixels, detector-frame 0.4–0.7. Gate reruns: a0ajw (mhziuq8q), a0bjwhst (nr0rkvuh), a0chst (aeuds6e7), check-in scheduled.
- **Step 1b (2D, cal, v31 coarse shift for JWST):** 2.69 [2.51, 2.85] vs 2.87 without and 2.42 on tweakreg crf. The JWST coarse shifts are small (median 3 mas, max 21 mas). Robust bright-pixel scores are nearly equal between cal and crf (1.55 vs 1.53), so most of the cal–crf gap sits in the χ² tails: cal files carry unflagged cosmic rays and outliers that crf flagged (stage-3 outlier step). Needs the outlier term of the noise model (spec), not registration.
- **Splat sky (v32, rerun after the fix):** 3.94 [3.61, 4.24], all 5.31, flux 2.46σ vs the 3D grid 2.30 / 5.20 / 2.20 on the same setup. 1599 splats after 8 rounds. Behind the grid; densification on residual coadds with an isotropic PSF approximation and Adam shapes are the likely limits.


## Phase-2 gate, JWST + WFC3/IR (5 Nov ~15:30 AWST): fails as run

Same JWST test dithers (24 images) in A and B; same 8 WFC3/IR test exposures in B and C; 3D K = 12 learned chromatic PSF, unregistered inputs, HST a-priori WCS + coarse shift in B.

| Held-out | A: JWST only | B: JWST + WFC3/IR | C: WFC3/IR only |
|---|---|---|---|
| NIRCam S/N>5 | **2.30** [2.20, 2.40] | 2.82 [2.65, 2.99] | – |
| NIRCam all pixels | 5.20 | 5.25 | – |
| NIRCam flux pull NMAD | 2.20 | 2.17 | – |
| WFC3/IR S/N>5 | – | **10.4** [9.7, 11.0] | 77.8 |
| WFC3/IR all pixels | – | 5.44 | **5.11** |
| WFC3/IR flux pull NMAD | – | 2.55 | 13.5 |

- JWST helps HST strongly on sources (77.8 → 10.4; flux error 13.5 → 2.6σ). HST makes JWST worse on sources (2.30 → 2.82), in every NIRCam band except F090W, most in F150W (2.48 → 3.55) and F356W (2.66 → 3.36).
- In the first test exposure per band the extra χ² concentrates in a few regions (F150W: one corner of the grid; F277W/F356W: grid edges and corners), not uniformly.
- WFC3/IR source χ² grows with wavelength in B (F105W 3.0, F125W 6.9, F140W 8.0, F160W 23.2).
- Hypotheses to test: (1) edge handling for the undersampled IR pixels (3.2 target px) and the pad; (2) epoch 2017 vs 2022 (proper motion of stars, variability), since the sky is static; (3) IR PSF window and capacity in the shared PSF network; (4) residual HST registration (translation-only coarse shift, rotation/scale left to the gradient fit).

Splat run (v32): crashed in round 2 (splat centre moved off-grid; stamp index out of bounds). Fixed by Sonnet (bounds clamp, sanitize per round, split failures logged and skipped); relaunched as splat2b (ekxdjlm3). Round 1 before the crash: 623 splats, validation S/N>5 24 at 0.58 h (grid ~4).


## 5 Nov afternoon (AWST): registration, pixel store, splats

- **Step 1 (2D learned grid, unregistered JWST cal, Q5 test set):** S/N>5 2.87 [2.71, 3.02], all 5.31, flux pull NMAD 2.29 vs 2.42 [2.33, 2.51] on tweakreg crf. Drizzle on cal 3.59 [3.34, 3.84]. Gradient-only transforms do not fully absorb 25–55 mas starting offsets in 2D.
- **A (3D K = 12 grid, unregistered cal):** 2.30 [2.20, 2.40], all 5.20 — better than both 2D runs.
- **C (WFC3/IR only):** S/N>5 77.8 (coadd 84); all-pixel 5.11. The HST source pixels are badly modelled in every method (undersampled IR); to diagnose with residual maps.
- **v31** (`--coarse-shift-jwst`): in-model translation scan for JWST groups too, two passes, cross-band references from training pixels. Step 1b (session k0e7ihxq) tests it.
- **U0 pixel store** (`pixelstore.py`, Sonnet; store `/arc/projects/ots/torchsky-data/smacs0723_union/pixelstore_v1`): JWST 722 exposures (6 jobs), HST 128 chips (561 M rows), SPHEREx 689 cutouts (31 M rows), GALEX 6 — extracted in ~15 min; Spitzer waits for its download. 30″ tiles, halos per instrument, one part file per tile per job. Skipped as calibration darks: UVIS 12342 F218W, 14534 and 16567 entries.
- **v32 elaborated splats** (`--sky-coeffs splat2`, Sonnet): smooth coarse grid + anisotropic Gaussians with σ floor, spectra in the RBF basis, amplitudes by projected CG (non-negative broadband), shapes by Adam; catalogue-free densification from residual significance with a threshold from the negative-peak null, 1-vs-2 Gaussian split test, Fisher S/N prune, validation-based stopping. Bit-identical to v31 without the flag. Run splat2a (qcwu7poa) uses A's exact setup to compare with 2.30.


## Union data download (5 Nov, ~afternoon AWST)

Scope: the union of all space footprints, an 8′ × 8′ box centred at RA 110.8020, Dec −73.4586, individual exposures only. Manifests in `/arc/projects/ots/adass2026/scripts/union/` (built by Sonnet agents from MAST JWST keyword services, HST headers, IRSA SIA `spitzer_sha`, IRSA ObsCore SPHEREx, MAST GALEX CAOM). Data go to `/arc/projects/ots/torchsky-data/smacs0723_union/` (logs `fetch_mast.log`, `fetch_irsa.log`), one subfolder per program and instrument (NIRISS F150W must not be read as NIRCam F150W).

| Set | Files | Size |
|---|---|---|
| JWST cal, programs 2736 (NIRCam module A, NIRISS, MIRI), 4043, 4368, 6882, 12774 | 587 to fetch (+135 present) | 65.5 GB |
| HST extra flc (UVIS F606W 14534/16567, UVIS F218W 12342) | 6 | ~1 GB |
| Spitzer IRAC cbcd/cbunc/bimsk (12005, 12123, 14017, 90233) | 3486 frames | 2.6 GB |
| SPHEREx L2 10′ cutouts (qr2, qr3; all extensions incl. PSF and WCS-WAVE) | 718 | ~3.6 GB |
| GALEX AIS visits (int, rrhr, skybg, flags + extras) | 3 visits × 2 bands | 0.26 GB |

Not available as calibrated exposures: ACS F502N/short F435W calibration programs (no flc/flt in MAST). No MIPS. GALEX photon lists unconfirmed (gPhoton endpoint 404).

Design draft for fitting the union: `claude/union-fit-design.md`.


## Step 2: JWST + HST, no pre-registration (5 Nov, launched ~09:30 AWST)

Inputs by default: JWST `_cal` (pipeline guide-star WCS, no tweakreg, no outlier step) and HST `_flc` (ACS/WFC, WFC3/UVIS) / `_flt` (WFC3/IR). MAST's primary HST WCS is an a-posteriori Gaia fit made by registering these images ('-FIT_REL_GAIAeDR3'); v30 starts from the a-priori alternate WCS instead (`--hst-wcs apriori`: GSC-corrected guide-star solution when present, else plain IDC). A-priori vs archive offsets: 405 mas (WFC3/IR sample), 74–87 mas (ACS). JWST cal vs the old tweakreg files: median 27 mas, max 54 mas.

Code: `hst_io.py` (Sonnet; flc/flt loader, SIP + D2IM/NPOL lookup tables via astropy, MJy/sr per nominal pixel like NIRCam cal, drizzlepac good bits; a-priori WCS policy added by me), v29 (Sonnet; band registry for any instrument, learned chromatic PSF with λ/D per telescope, HST splat coadds for init, NIRCam-only stripes, mixed test selection; bit-identical to v28 without HST flags), v30 (me; `--hst-wcs`, `--coarse-shift`: per-exposure translation from a likelihood scan against the nearest-λ JWST coadd, JWST as the anchor frame, before the joint fit). Local check on 3 WFC3/IR samples: a-priori start → shift (230, 340) mas, |d| ≈ 410 mas = the archive-fit offset; archive start → ≤ 25 mas.

Units note (from Sonnet, verified in headers + JDox): NIRCam cal and HST flt/flc are flat-fielded surface brightness per nominal pixel; the AREA map (0.98–1.05) matters for flux sums, not for a surface-brightness forward model. No change needed.

| Job | Session | RUN | What |
|---|---|---|---|
| s1cal2d | njwvxb3v | v28_step1_cal_unregistered_e2d | Step 1: 2D learned grid on JWST cal (unregistered), Q5 test set mapped to cal names; compare with 2.42 [2.33, 2.51] |
| s2ajw | n85m5gfm | v30_s2A_jwst_cal_3d | A: JWST cal only, 3D K = 12, same JWST test set |
| s2bjwhst | ys3c1cig | v30_s2B_jwst_wfc3ir_3d | B: JWST cal + WFC3/IR 14096 (56 exposures, F105W–F160W, 2017), a-priori WCS + coarse shift 1″; JWST test set as A, 2 HST test exposures per band |
| s2chst | mayninup | v30_s2C_wfc3ir_only_3d | C: WFC3/IR only, same HST test set as B |

Gate (phase 2): B beats A on held-out JWST and B beats C on held-out HST. In B the coarse shifts are ~(155, 480) mas with χ² ×0.37 against F090W, consistent across exposures.


Live log of the Run queue in the research plan. Times AWST unless marked; job clocks are UTC.

## Jobs launched (CANFAR, astroai/base:latest, 1 GPU unless noted)

| Job | Session | RUN | Script | What |
|---|---|---|---|---|
| q1cap | n8egwt1c | v21_q1_capacity | v21 | Q1: zip-field fitted directly to drizzle and v19 3D-grid coefficients, 2 configs × 12 min |
| q2aeval | nyq8gsp4 | v21_q2a_det_eval | v21 | Q2a: v20 zip weights scored with the deterministic renderer (no fit) — DONE |
| q2brefit | gaxow52t | v21_q2b_det_refit | v21 | Q2b: v20 zip weights refitted 90 min with the deterministic renderer |
| q2cmlp | kft7phgw | v23_q2c_mlp_det_warm | v23 | Q2c: zip MLP field, deterministic renderer, fresh warm start (3000 steps), 120 min |
| q3bs1 | parz86np | v23_q3_bspline_1px | v23 | Q3: linear B-spline field, spacings 8,4,2,1 px, deterministic renderer, 120 min |
| q3bs05 | f6m76giq | v23_q3_bspline_05px | v23 | Q3: same, spacings 8,4,2,1,0.5 px |
| q4w2d | bwr2opr2 | v23_q4_wings121_2d | v23 | Q4 (first step): learned chromatic PSF with 121-px (4.84″) window instead of 31, 2D grid, 240 min |
| q4w3d | icnuhfwk | v23_q4_wings121_3d_k12 | v23 | Q4: same, 3D grid K = 12, 330 min |
| q5smoke | gea2l8n3 | v22_q5_smoke | v22 | Q5 smoke test (--quick) — DONE, selection + BOOT lines work |
| q5e2d | eoderw4d | v22_q5_ntest2_e2d | v22 | Q5: 2D learned-instrument grid under the new protocol (2 test dithers per band, dither bootstrap), 240 min |
| hstfetch | l6av9s5m | – | fetch_hst.py | Q6: download 222 HST imaging exposures (9.7 GB) to /arc/projects/ots/torchsky-data/hst_smacs0723/raw (CPU) |

## Code

- `smacs_four_v21.py` (patch_v21.py): `--zip-render det` deterministic renderer (same pupil PSF as the sampler, binned from 4× to 2×, FFT convolution, bilinear read at detector positions); `--zip-init-state`, `--zip-eval-only`, `--zip-warm-steps`, `--zip-levels/feat/hidden`; `--zip-capacity` (Q1). Unit test: det vs 4096-sample MC agree to the MC noise (rel rms 6e-3 on a pixel-scale random field); mirrored PSF gives 7e-2, so orientation is checked.
- `smacs_four_v22.py` (patch_v22.py, written by Sonnet, reviewed): `--n-test N` dithers per band, `holdout_selection.json`, per-exposure scores, stratified dither (cluster) bootstrap, `BOOT` lines.
- `smacs_four_v23.py` (patch_v23.py, Sonnet, reviewed): `--zip-field bspline`, `--bspline-spacings` (linear, 11.3 M coefficients at 8,4,2,1,0.5 px).
- All in `/arc/projects/ots/adass2026/scripts/`. Uploads: `vcp` push returns HTTP 500 from the cloud session; a direct PUT with the proxy cert works (`arcput.py`, kept in the scripts folder).

## Bug found: zip PSF rotated by 180° (v20–v23)

The explicit model's `convolve_basis` is a correlation (a point source at p0 responds at p0 − δ for a kernel tap at δ); the zip Monte Carlo sampler (v20) used s(x − δ), a convolution (p0 + δ). The frozen pupil instrument was fitted in the explicit convention, so every zip run saw its PSF rotated by 180°. The fitted pupil PSF is far from point-symmetric (L1 asymmetry 0.30–0.65 per band). Shown by `test_v24_orientation.py`; v24 flips the tabulated PSF once (`--zip-psf-convention explicit`, default). My v21 deterministic renderer inherited the old convention, so Q2a/Q2b/Q2c measure the renderer with the rotated PSF; v24 runs (Q2d, Q2e, Q3) use the fixed one. The two Q3 runs launched on v23 were stopped and relaunched on v24.

| Job | Session | RUN | Script | What |
|---|---|---|---|---|
| q2dmlp | jmuq2v92 | v24_q2d_mlp_det_fixed | v24 | MLP zip, deterministic renderer, fixed PSF orientation, fresh warm start, 120 min |
| q2emc | b49ns4ph | v24_q2e_mlp_mc_fixed | v24 | same with the Monte Carlo renderer (8 samples) |
| q3bs1f | cvxf4thr | v24_q3_bspline_1px | v24 | Q3 B-spline 8,4,2,1 px, deterministic, fixed |
| q3bs05f | ox2hin6m | v24_q3_bspline_05px | v24 | Q3 B-spline 8,4,2,1,0.5 px, deterministic, fixed |
| hstfetch2 | uwhvc0e0 | – | run_fetch.sh | Q6 download (the first hstfetch exited without running; relaunched from a script file) |

Validation references (same validation exposures): drizzle S/N>5 4.71, v19 3D pupil grid 3.08, v20 zip (MC) 34.5.

## Correction (5 Nov ~00:30 AWST): which convention is right is an open question

The pupil PSF's orientation search (`PupilPSF.fit_orientation`) matches the kernel directly to star images, i.e. it assumes a convolution, the v20 zip convention. Only the explicit models' `convolve_basis` uses a correlation. Early validation favours the v20 zip convention: Q2c (old) S/N>5 2.56 at 0.79 h vs Q2d (v24 flip) 5.37 at 0.52 h. v25 adds one flag for every model, `--psf-convention {correlation, convolution}` (default correlation = v19/v24), and these runs test it:

| Job | Session | RUN | What |
|---|---|---|---|
| pconv3d | zt7wr8k9 | v25_pupil3d_convolution | v19 3D pupil-PSF explicit run, same settings, true convolution |
| pconv2d | vkg7q9zo | v25_pupil2d_convolution | v19 2D pupil-PSF explicit run, true convolution |
| q3bs1c | vo9wm45z | v25_q3_bspline_1px_conv | Q3 B-spline 8,4,2,1 px under convolution |

## Results (5 Nov ~03:00 AWST)

- **Q2c TEST (MLP zip, deterministic renderer, v20 orientation, fresh warm start, 2 h):** S/N>5 2.48, all 5.82, flux pull NMAD 2.55. Drizzle 2.77 / 5.88 / 2.69; v19 pupil grid 2.20 / 5.62. v20 Monte Carlo zip: 23.1.
- **Q2b TEST (v20 weights refitted with the deterministic renderer, 90 min):** S/N>5 2.92, all 5.87.
- **Q2e (Monte Carlo, fixed orientation):** validation S/N>5 stays at 34 for 1.7 h. Q2d (deterministic, flipped): 2.77 at 1.6 h. Monte Carlo PSF sampling is the cause of the zip failure; orientation is second order.
- **Explicit pupil model:** the convention does not matter (v25 convolution runs track v19's validation to 3 decimals for 0.8 h; the odd wavefront terms flip sign). Stopped.
- **Star zooms (star_zooms_q2_20261105.png):** the deterministic zip skies are deconvolved: stars collapse to ~0.1″ cores, no spikes, three decades down by 0.25″; the pupil grid keeps a broad halo; the v20 Monte Carlo sky is full of spikes. Q2c F444W shows a dark ring around bright stars (ringing).
- **B-spline (Q3):** the first runs failed in the warm start (Adam lr 5e-3 left mse/σ² 1865 after 3000 steps; the MLP reaches 0.02). v26 adds `--zip-warm-lr`; relaunched with warm lr 0.2 and fit lr 0.05 under convolution (sessions sx0tuc56, tp0jce0e).
- **HST:** 222 files downloaded (9.7 GB) to `/arc/projects/ots/torchsky-data/hst_smacs0723/raw`; they are 116 unique exposures (HAP-named and original copies of the same exposure); `hst_imaging_unique.csv` lists one per exposure.

## Results (5 Nov ~07:45 AWST)

- **q5zip, the deciding run:** deterministic MLP zip on the pupil instrument fitted without the test dithers, 2-dither protocol: S/N>5 2.67 [2.59, 2.76], all 5.29 [5.22, 5.37], flux pull NMAD 2.82. The pupil grid on the same instrument and test dithers: 2.70 [2.59, 2.80], 5.14, 2.73. They tie on bright pixels and the grid is better on all pixels. The 2D learned-instrument grid is better than both (2.42). Deconvolution does not yet buy better prediction.
- **Q3 v28 B-spline (oversampled CG warm start):** TEST 12.7 (fit lr 5e-3) and 3.75 / 5.86 (lr 2e-2). Still worse than drizzle; optimiser-limited.

## Results (5 Nov ~05:20 AWST)

- **Q4 3D K = 12, 121-px learned PSF window (5.5 h):** TEST S/N>5 1.81, all 5.57, flux pull NMAD 2.27 (31-px window: 1.91 / 5.58). Q4 2D: 1.96 / 5.61 / 2.30 (31 px: 1.84 / 5.60 / 2.1). Star zooms (`star_zooms_psf_window_20261105.png`): the 121-px window removes the box at the 31-px window edge, but the spikes (F200W) and the ring (F444W) stay in the sky.
- **Q2f (deterministic zip, 5 h budget):** best validation at 2.1 h, early-stopped; TEST 2.63 / 5.80. Longer is not better than Q2c (2.48).
- **Q3 B-spline 0.5 px (Adam warm start):** TEST 4.56 / 6.10, worse than drizzle. v28 runs (oversampled CG warm start): validation 17.1 (lr 5e-3) and 7.6 (lr 2e-2) at 1.6 h; MLP field 2.4 at the same time.
- **Q5, pupil 3D grid (convolution):** 2.70 [2.59, 2.80], all 5.14, flux pull NMAD 2.73; same test dithers as the 2D learned grid (2.42 [2.33, 2.51]) and the coadds (identical coadd scores confirm the same selection).
- **q5zip (xb6riwak, RUN v28_q5_ntest2_zip_det):** the deterministic MLP zip on that pupil instrument (fitted without the test dithers), 2-dither protocol, 150 min. ETA ~08:00 AWST.

## Results (5 Nov ~03:30 AWST)

- **Q5, learned instrument vs coadds with error bars** (2 test dithers per band = 24 detector images, dither-cluster bootstrap, 68 % intervals; 2D learned-instrument grid, 4 h): S/N>5 ⟨r²/σ²⟩ learned 2.42 [2.33, 2.51], Drizzle-like 3.77 [3.57, 3.96], SWarp-like 5.54 [5.39, 5.71]; all pixels 5.15 / 5.42 / 6.02; flux error 2.43σ [2.31, 2.49] / 3.26σ [3.01, 3.39] / 3.11σ. The main talk result holds with intervals that do not overlap. Caveat: 2 dithers per band is a coarse bootstrap. Scores are not comparable with the old single-image protocol.
- **Q2e TEST (Monte Carlo, flipped PSF):** S/N>5 22.86 (v20: 23.1). **Q2d TEST (deterministic, flipped):** 2.69. **Q2c (deterministic, v20 orientation):** 2.48.
- **Q3 B-spline: the warm start was the problem twice.** (1) Adam lr 5e-3 left mse/σ² 1865. (2) CG (v27) fits the grid nodes to mse/σ² 1.06, but a linear field with more coefficients than nodes is then free between nodes: validation S/N>5 138 after 15 min, because the renderer reads the field on a 2× grid. CPU test: node mse 0.012 but half-pixel mse 123. v28 fits the warm start on the 2× grid against the bilinearly upsampled coadd (half-pixel mse 2.1). Relaunched: oqgv69eq (fit lr 5e-3), t17wgc29 (2e-2). Still running: tp0jce0e (0.5 px, Adam warm start), validation 9.3 at 1.06 h.
- **q5pupil3d (gid5exjh):** pupil 3D explicit under the 2-dither protocol, convolution, 4 h, so the zip can be scored with error bars on an instrument that has not seen the test dithers.

## Earlier results

- **Q2a (same v20 weights, renderer swapped):** TEST S/N>5 19.2 deterministic vs 23.1 Monte Carlo (all pixels 7.79 vs 7.01). Sampling noise at scoring time explains only a small part of the failure; the weights themselves are poor.
- **Q2b (refit with deterministic renderer, old PSF orientation):** validation S/N>5 7.39 at 15 min, 4.51 at 31 min (drizzle 4.71); first validation at 15 min: S/N>5 7.39 (v20 Monte Carlo run at the same stage: 34.2). Training χ²/pixel 2.1–2.5 vs 2.8–3.7 with Monte Carlo, and 0.8 s/step vs 1.7 s/step.
- **Q1 (capacity):** the base field (5.6 M parameters) fits the drizzle coefficients to mse/σ² < 1e-4 in 12 min (42 740 steps), on bright pixels too. The v20 warm start's 3.8 was 800 steps, i.e. undertrained, not a capacity limit. Caveat: this is at grid nodes only.
- **Q5 smoke:** with 2 test dithers per band, the coadds score much worse on S/N>5 pixels (drizzle 11.9 [9.9, 13.7]) than under the old single-image test (2.77). Scores under the two protocols are not comparable.

## Data facts found on the way (Sonnet, from file names; to verify)

- Each band sits in a single visit group (F090W+F277W 02101, F150W+F356W 02103, F200W+F444W 02105), 9 dithers per band; 123 of 135 files survive loading (12 SW detector images do not cover the field).
- Today's "test exposure" is one detector image (SW) per band, not one dither.
- HST manifest: RELICS 14096 ACS has some exposures whose observation-level start time matches 12166/12884 (2011/2013); use FITS header dates, not the manifest, for the time baseline. Also on disk soon: WFC3/IR 18297 (F110W, F160W, 2026) and 16729.
