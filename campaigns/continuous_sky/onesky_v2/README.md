# Small experiments for "One sky at every scale, version 2" (13 Nov 2026)

Context: we are testing pieces of a new way to model the astronomical sky from telescope
exposures. Each experiment tests ONE element and has a rule for dropping it. Be honest:
a negative result is as useful as a positive one. Never tune one method more than another.

Machine: 2 CPU cores, 7 GB RAM, no GPU, shared by three agents working at the same time.
- Call `torch.set_num_threads(1)` at the top of every script.
- No single fit may run longer than about 4 minutes; reduce iterations rather than exceed it,
  and record the iteration count and whether the loss was still falling.
- Run long commands with a timeout; do not leave background processes running.
- Python 3, torch 2.15 (CPU), numpy, scipy, matplotlib are installed. Install nothing heavy.
- Write only inside your own files/folders named below. Do not edit `e1/common.py`.

Reporting rules (they apply to code comments, the report file and your final message):
- Plain names, no codenames or unexplained acronyms. Canadian English.
- State every approximation and shortcut you took, and every place a method got special treatment.
- Give numbers with what they measure. Say what you did not finish.

## E1: any-scale sky representations, no instrument (folder `e1/`)

Question: does a sky made of items with continuous position, size and orientation and a
LEARNED profile (a shared network evaluated in coordinates scaled by each item's own size)
represent a wide range of angular scales better than Gaussians, a pixel grid, and a
hash-grid neural field, at equal size?

Shared ground truth and scoring: `e1/common.py` (read it first).
- `make_spec(seed)` builds an analytic synthetic sky: stars (unresolved, width 0.45 px),
  galaxies (Sérsic discs and bulges, spiral arms, clumps, sizes 1 to 60 px) and diffuse
  light (power-law structure, 8 to 512 px). `make_data(spec, n, zoom)` returns
  `truth`, `noisy`, `sigma`. `score` and `save_result` write the comparable numbers.
- STANDARD SETTING for every method: `make_data(make_spec(0), n=256, zoom=1.0)`.
  Fit the `noisy` image with weights `1/sigma**2`; all scores are against `truth`.
  The headline number is `chi2_truth` (mean squared error against the truth in units of
  the noise; the noisy image itself scores 1.0; lower is better), then the fractional
  error by octave of scale.
- BUDGETS (total number of fitted numbers for this one image): 5000, 15000, 45000.
  The image has 65,536 pixels. Count every fitted scalar. For a shared network, report
  its weights separately (`n_params_shared`) AND include them in the budget when the
  network is trained on this image alone.
- Also run each method once at budget 15000 on the noise-free `truth` (target='truth'),
  to separate capacity from noise handling.
- Result files: `save_result(method, budget, variant, model, data, ...)` with
  variant like `seed0_zoom1.0`. Method names: `grid`, `gauss_lattice`, `gauss_adaptive`,
  `hash`, `profile_lattice`, `profile_adaptive`, and `profile_frozen_*` for the transfer test.
- Seeding rule, identical for Gaussians and learned profiles:
  `*_lattice`: items start on a regular lattice covering the image (several sizes allowed),
  nothing in the seeding looks at the data. `*_adaptive`: in addition, items whose flux
  falls below a threshold are moved to positions drawn in proportion to the smoothed
  positive residual. No source detection, no catalogue, in either.

## E2: rendering a profile through a telescope kernel (folder `e2/`)

Question: can an arbitrary (learned) profile be rendered through a diffraction-limited,
undersampled kernel to 1e-3 of the peak, cheaply, either with closed-form Gaussian
sub-profiles or by local sampling?

## Reading files from CANFAR storage (added 13 Nov, evening)

- Read a file: `/home/claude/arc/arcget.sh <path under arc:> <local file>`, for example
  `/home/claude/arc/arcget.sh projects/ots/adass2026/fieldop/fieldop_smacs_fix_200_s0/metrics.json metrics.json`.
- List a folder: `vls -l arc:projects/ots/adass2026/<folder>`.
- NEVER print, copy, move or upload the certificate file in `~/.ssl`. Do not write anything to CANFAR storage,
  and do not create, stop or delete CANFAR sessions (jobs): the orchestrating session does that.
- Other people's jobs are running on the same account. Leave them alone.
