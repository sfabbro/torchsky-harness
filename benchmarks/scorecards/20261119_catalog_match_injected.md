# Injected-truth association: torchsky vs nway (2026-11-19)

`bench_catalog_match_injected` (Mac, subject `833de488`, nway 4.8.0 in the
`nway` Pixi environment, run out of process on FITS written with torchfits).
0.25 deg^2 box, 3000 primaries, 20000 background sources per extra
catalogue, per-row isotropic errors (nway's error model): primary
0.05-0.3", second 0.3-1.0", third 0.2-0.6". Both tools get the true
completeness as prior (nway: the product of the two for three catalogues,
its single prior); `torchsky_em` fits completeness by EM instead.

## Pairwise (completeness 0.7)

| method | completeness@0.5 | purity@0.5 | completeness@0.9 | purity@0.9 | ECE(p_any) | time |
|---|---|---|---|---|---|---|
| torchsky | 0.974 | 0.941 | 0.902 | 0.955 | 0.022 | 0.015 s |
| torchsky (EM, fit 0.692) | 0.974 | 0.941 | 0.901 | 0.955 | 0.021 | 0.016 s |
| nway | 0.956 | 0.950 | 0.715 | 0.966 | 0.035 | 2.2 s |

## Three catalogues (completeness 0.7, 0.6)

| method | MAP tuple | cat1 / cat2 component | any-counterpart completeness@0.9 | purity@0.9 | ECE(p_any) | time |
|---|---|---|---|---|---|---|
| torchsky | 0.928 | 0.953 / 0.973 | 0.965 | 0.990 | 0.006 | 0.06 s |
| torchsky (EM) | 0.928 | 0.953 / 0.973 | 0.963 | 0.990 | 0.007 | 0.06 s |
| nway | 0.817 | 0.875 / 0.939 | 0.658 | 0.996 | 0.126 | 3.3 s |

torchsky's N-way hypotheses include every absent-counterpart combination with
per-catalogue completeness and density, normalised per primary, with full
covariances; that is where the tuple accuracy and calibration gains come from.
