# One sky, version 2: small experiments (13-14 Nov 2026)

Design: `../docs/one-sky-v2-raw-anyscale-ensemble.md`. Log, newest first: `../docs/one-sky-v2-experiments-log.md`. Rules given to every
agent that worked here (machine limits, reporting rules, test specifications): `README.md`.

| Folder | Question | Report |
|---|---|---|
| `e1/` | Do sky items with a learned, scale-factored profile represent a wide range of angular scales better than Gaussians, a pixel grid and a hash-grid neural field, at equal size? (synthetic, no instrument) | `e1/BASELINES_REPORT.md`, `e1/PROFILES_REPORT.md` |
| `e2/` | Can a profile be rendered through an undersampled, diffraction-limited kernel to 1e-3 of the peak, cheaply? (stand-in kernels) | `e2/E2_REPORT.md` |
| `e2_learned/` | The same with the kernels learned on SMACS (`fieldop_smacs_fix_200_s0`) | `e2_learned/E2_LEARNED_REPORT.md` |
| `kernel_check/` | Independent check of those learned kernels and of the fitted sky's flux scale against the official mosaic | `kernel_check/KERNEL_CHECK_REPORT.md` |

Paths inside the scripts are those of the cloud workspace they ran in (`/home/claude/work/...`, `/home/claude/arc/arcget.sh`).

Not in Git (size): caches, kernel tables, the fitted state, mosaic cut-outs, per-source profile arrays, five E1 result files with item
dumps, the test data of the paired scorer. Archives on CANFAR storage, `/arc/projects/ots/adass2026/onesky_v2/`:
`one-sky-v2-experiments-e1-e2-20261113.tar.gz` and `one-sky-v2-experiments-round2-20261114.tar.gz`.
