# Diagnostics behind the numbers of 23 Nov 2026

Analysis scripts run on a workstation against files copied from CANFAR (predictions, protocol masks, `xpred` side files, saved fits).
They are kept as they ran; paths are arguments, or the folder named by `ONESKY_SCRATCH` (`canfar/` inputs, `summary/` outputs).

| File | What it measures |
|---|---|
| `diag.py` | held-out pixels on the common set: reduced chi-square, NMAD of the pulls, outlier fraction, fractional residual, flux ratio on sources; residual images and cut-outs |
| `diag_shift.py` | the shift per held-out exposure, whole cut and quadrants, that best explains the residual on bright pixels |
| `diag_invented.py` | structure one fit has and another does not: unreproduced fraction, unmatched-peak fraction, unmatched flux fraction |
| `diag_seeds.py` | the same comparison across seeds (not a test of invented structure: same data and same starting items) |
| `reg_check.py`, `reg_move.py` | where a fit left its exposures against another fit of the same exposures |
| `kern_stay.py`, `kern_orient.py`, `kern_warm.py` | whether a kernel started from point-source images stays near them, its orientation, and the warm-start error |
