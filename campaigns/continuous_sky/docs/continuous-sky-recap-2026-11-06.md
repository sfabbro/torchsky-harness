# Continuous sky: recap and what next (6 Nov 2026)

Living doc: https://claude.ai/artifact/HDJ7Wvmi86Wc4SB9HrwkJx. This file is a copy for the project; the doc is the version to edit.

## Summary

The best model so far is factorised, with learned PSF tables per detector and band and a 3″ window (E2-data).
- Held-out WFC3/IR: 5.8, against 12.2 for the coadd.
- NIRCam with HST added: 2.32, against 2.24 without.
- NIRCam alone: the edge-fixed ChromaticPSF model (E1) is still best, at 1.92.

The single learned field operator works on synthetic data without large offsets. On real data it is not yet competitive: 9.8 on JWST alone, and it breaks down with HST. A loss ablation is running.

Three results change how earlier numbers should be read:
- an edge-ring bug inflated every earlier all-pixel score;
- the sky–PSF degeneracy leaves a shared blur that no data diversity can remove;
- starting from physics PSF libraries fitted worse than a plain Gaussian start.

## Results

Held-out ⟨r²/σ²⟩ on S/N>5 pixels, with the all-pixel score in brackets. The 68% bootstrap range is in square brackets where measured. Lower is better; 1 is the noise level.

| Model | JWST only: NIRCam | JWST+WFC3/IR: NIRCam | JWST+WFC3/IR: WFC3/IR |
|---|---|---|---|
| Drizzle coadd (v34) | 2.90 (1.48) | 2.90 | 12.8 |
| v33 A0 (with the edge-ring bug) | 2.31 (5.20) | 3.34 | 14.2 |
| **E1**: ChromaticPSF + halo | **1.92** [1.84, 2.01] (1.35) | 2.38 [2.21, 2.56] | 9.82 [8.98, 10.54] |
| **E2-data**: learned tables, Gaussian start, 0.04″ | 2.24 [2.18, 2.30] (1.45) | **2.32** [2.23, 2.41] | **5.82** [5.05, 6.50] |
| E2-lib: learned tables, STPSF/STDPSF start | – | 2.31 [2.16, 2.46] | 11.83 [10.50, 13.05] |
| E2-data at 0.02″, 2″ window | – | 4.63 | 16.6 |
| Field operator, JWST only | 9.81 (2.85) | – | – |
| Field operator, B, flat z, seed 0 | – | 372 | 161 |
| Field operator, B, z-hier, seeds 0 and 1 | – | 597 / 1015 | 160 / 139 |

The learned IRAC instrument scores 20.1 on held-out bright pixels, against 25.5 for the structured model.

## Key findings

See the living doc for the full text. In short:
1. **Edge-ring bug.** About 16% of pixels sat outside the sky grid. It affected every earlier all-pixel score.
2. **Relative kernels.** The learned PSF is a relative kernel, and the sky sits at NIRCam SW resolution.
3. **Common-mode gauge.** A shared blur g moves freely between the sky and all kernels; data diversity cannot fix it, so it is declared ("the sharpest kernel"). Confirmed in synthetic tests.
4. **The gate failure was representation, not degeneracy.** Free tables per detector take WFC3/IR from 9.82 to 5.82.
5. **Physics starts are worse** (11.8 against 5.8). Their kernels drift sharper and the sky absorbs a λ-dependent blur.
6. **Field operator, synthetic.** Near the noise floor without large offsets, 10× better than the factorised model. Offsets beyond about half a PSF core fall into a joint sky/registration local minimum. Restarts help; z-hier gives no gain (single seed).
7. **Field operator, real data.** It fails even on JWST alone. The leading hypothesis is that the outlier mixture turns misfit bright cores into outliers. A Huber-loss ablation is running: opfield_v3_jwst_cal_huber and opfield_v3_jwst_wfc3ir_huber_cshift.

## What next, ranked

1. Read the Huber ablation.
2. Add a seed flag and run 3 seeds each of E1, E2-data and E2-lib on B.
3. Test suite on the best factorised model: T2, T3, T4 and T7.
4. Field operator, depending on step 1.
5. Registration without anchors: per-visit codes, multi-scale sky, restart.
6. Add NIRISS and the 2024 and 2026 NIRCam epochs.
7. A learned detector response N for the bright cores.
8. Union tiles (U1).
9. A time-dependent sky.
10. The A2 continuous operator and a CG/VarPro solver.
