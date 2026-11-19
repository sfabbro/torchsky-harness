# Catalog match: torchsky CUDA vs GPU nearest-neighbour libraries (2026-11-20)

CANFAR headless GPU sessions on `images.canfar.net/astroai/notebook`, with one
H100 NVL MIG 1g.12gb slice per run (16 SMs, 10.75 GiB). The torchsky subject is
`sfabbro/torchsky@1684524f` (0.5.0, torch 2.14.0+cu130, Pixi `cuda-bench`).
`bench_catalog_match_perf.py --device cuda`, 2 arcsec, `--runs 3` (median), 8 CPU
threads for cKDTree.

| run | harness | peers |
|---|---|---|
| `catalog-match-gpu-peers-20261119-211019` | `bdfcf7e` | CuPy, cuVS, torch_cluster, FAISS |
| `catalog-match-gpu-peers-20261119-213251` | `f06063c` | Open3D only (other columns repeat) |

Peers were installed at job time by `scripts/install_gpu_nn_peers.sh`, with pip
constrained to the solved env, so `pixi.lock` is unchanged:

| peer | version | status |
|---|---|---|
| CuPy `cupyx.scipy.spatial.KDTree` | cupy-cuda13x 14.2.0 | float64, exact |
| cuVS `brute_force` | cuvs-cu13 26.10 | float32 only: **inexact** at arcsec scale |
| torch_cluster `knn` / `radius` | 1.6.3, built from git for sm_90 (no PyG wheel for torch 2.14) | float64, exact; CUDA kernels are brute force |
| Open3D `core.nns` | 0.20.0 | float64, exact; needs GL/X11/libusb/udev from conda-forge (`pixi exec`) |
| FAISS GPU `IndexFlatL2` | faiss-gpu-cu12 1.14.1 | **error**: `no kernel image is available` (wheel has no sm_90 code) |

The peers work on 3-D float64 unit vectors with chord radius `2 sin(r/2)`.
Inputs are on the device before timing, as torchsky's ra/dec are. Results are
copied to the host inside the query time. Pair IDs are checked against torchsky
before any time is reported.

- Every exact (float64) peer row has **zero parity mismatches**.
- cuVS rows are `INEXACT`: 89-100 % of rows differ. Its float32 expanded-form
  `|x|^2 + |y|^2 - 2x.y` loses the 1e-10 chord^2 scale entirely, so its times
  are for wrong answers.
- `best` for kNN-only peers is `k=1` filtered to the radius. `all` has no row
  for them.
- `cupy_kdtree` `all` is `query_ball_point`, which allocates an
  `(n_query, n_tree)` int64 buffer. It is chunked at 1e5-1e6 and skipped at 1e7.
- `cupy_kdtree_kgrow` is an exact `all` built from `query(k, distance_upper_bound)`.
  Rows whose last slot is a pair are re-queried with 2k.
- torch_cluster `radius` regrows `max_num_neighbors` the same way, so it never
  truncates.
- Brute-force peers (cuVS, torch_cluster, FAISS) are skipped above 1e6.
- Open3D `knn_search` is brute force too: 11.6 s at 1e6, and it hit the 900 s
  worker timeout at 1e7.

Cells show `build + query = total s (torchsky ratio, peak GPU GiB)`.
`ratio = peer time / torchsky time`, so a ratio below 1 means torchsky is slower.
GPU memory is the device-wide `cudaMemGetInfo` peak during the warm-up call,
above the post-context baseline. It includes inputs and library pools.
torchsky's `best`/`all` build its index inside the call (build 0). Its `knn2`
builds a `SkyIndex`. The last column is torchsky in the Open3D run, which
Open3D's ratio uses.

## Where torchsky loses

Every loss is on query cost, never on parity, and torchsky's own index build
is cheap.

| case | torchsky | peer | torchsky slower by |
|---|---|---|---|
| crowded best 1e7 | 1.71 s | Open3D hybrid 1.17 s | 1.46x |
| crowded best 1e7 | 1.64 s | CuPy KDTree 1.48 s | 1.11x |
| crowded knn2 1e7 | 2.54 s | CuPy KDTree 1.99 s (build 1.03 + query 0.96) | 1.28x |
| crowded best 1e6 | 0.17 s | Open3D 0.091 s / CuPy 0.127 s | 1.85x / 1.35x |
| crowded knn2 1e6 | 0.26 s | CuPy 0.176 s | 1.50x |
| crowded all 1e6 / 1e7 | 0.33 s / 3.04 s | Open3D fixed-radius 0.31 s / 3.23 s | 1.04x at 1e6; torchsky 1.06x faster at 1e7 |
| 1e5, several cases | 8-40 ms | Open3D 4-21 ms, CuPy 12-22 ms | 1.1-3.7x (launch-overhead regime) |

- **Sparse and dense at 1e6-1e7:** torchsky wins every mode.
  - Against Open3D: `best` 1.2-1.8x, `all` 2.3-3.2x.
  - Against CuPy: `best` 2-4x, `knn2` 1.4-2.1x, `all` 4.4-5.8x over the
    exact k-regrow.
- **Query-only, with the tree already built** (the case where an index is
  reused), CuPy's KD-tree traversal is faster than torchsky's query path:
  - crowded `best` 1e7: 0.45 s vs 1.64 s (3.7x)
  - crowded `knn2` 1e7: 0.96 s vs 2.50 s (2.6x)
  - sparse `knn2` 1e7: 0.52 s vs 1.05 s (2.0x)

  Open3D has no separate build in this timing (its spatial hash is built inside
  the search).
- **GPU memory:** torchsky uses more in crowded fields.
  - crowded `all` 1e7: 9.9 GiB of the 10.75 GiB slice, vs 3.1 GiB for Open3D
    and 8.3 GiB for CuPy k-regrow.
  - `knn2` 1e7: 4.0-5.1 GiB vs CuPy's 1.5 GiB.
  - `best` 1e7: 1.8 GiB vs 0.7 GiB for Open3D.

Reading: a dedicated CUDA kernel is justified by the crowded `best`/`knn2`
query phase and by GPU memory. Open3D's hashed fixed-radius/hybrid search
(about 1.5x faster in crowded `best` at 1e7, about 3x smaller peak memory) and
CuPy's KD-tree traversal (3.7x faster per query once built) bound what a kernel
should reach. Sparse and dense fields do not need one.

## Results

### N = 1e+05

| layout | mode | torchsky CUDA s (GiB) | cKDTree 8 thr | cupy_kdtree | cupy_kdtree_kgrow | torch_cluster | cuvs_brute_force | open3d_nns | torchsky (open3d run) |
|---|---|---|---|---|---|---|---|---|---|
| sparse | best | 0.008 (0.05) | 0.071 (9.0x) | 0.010 + 0.002 = 0.012 (1.52x, 0.02) | n/a | 0.000 + 0.246 = 0.246 (30.99x, 0.04) | 0.000 + 0.282 = 0.282 (35.55x, 0.02) INEXACT 89327 | 0.000 + 0.004 = 0.004 (0.68x, 0.03) | 0.006 (0.05) |
| sparse | all | 0.009 (0.05) | 0.200 (21.5x) | 0.010 + 1.559 = 1.569 (169.13x, 1.65) | 0.010 + 0.011 = 0.021 (2.29x, 0.05) | 0.000 + 0.200 = 0.200 (21.56x, 0.04) | n/a | 0.000 + 0.007 = 0.007 (1.11x, 0.03) | 0.006 (0.05) |
| sparse | knn2 | 0.022 (0.12) | 0.086 (3.9x) | 0.010 + 0.004 = 0.014 (0.65x, 0.03) | n/a | 0.000 + 0.283 = 0.283 (12.94x, 0.04) | 0.000 + 0.285 = 0.286 (13.08x, 0.03) INEXACT 96690 | 0.000 + 0.117 = 0.117 (7.22x, 0.04) | 0.016 (0.12) |
| dense | best | 0.008 (0.05) | 0.062 (8.1x) | 0.010 + 0.003 = 0.012 (1.60x, 0.02) | n/a | 0.000 + 0.246 = 0.246 (32.01x, 0.04) | 0.000 + 0.279 = 0.279 (36.38x, 0.02) INEXACT 99954 | 0.000 + 0.004 = 0.004 (0.51x, 0.03) | 0.009 (0.05) |
| dense | all | 0.008 (0.05) | 0.196 (24.1x) | 0.010 + 1.483 = 1.492 (183.47x, 1.65) | 0.010 + 0.013 = 0.022 (2.74x, 0.05) | 0.000 + 0.200 = 0.200 (24.65x, 0.04) | n/a | 0.000 + 0.007 = 0.008 (0.89x, 0.03) | 0.009 (0.05) |
| dense | knn2 | 0.018 (0.11) | 0.079 (4.3x) | 0.010 + 0.006 = 0.015 (0.85x, 0.03) | n/a | 0.000 + 0.282 = 0.282 (15.48x, 0.04) | 0.000 + 0.280 = 0.280 (15.37x, 0.03) INEXACT 100000 | 0.000 + 0.119 = 0.119 (5.98x, 0.04) | 0.020 (0.11) |
| crowded | best | 0.019 (0.37) | 0.069 (3.7x) | 0.009 + 0.005 = 0.014 (0.76x, 0.02) | n/a | 0.000 + 0.245 = 0.245 (13.02x, 0.04) | 0.000 + 0.277 = 0.277 (14.72x, 0.02) INEXACT 99991 | 0.000 + 0.005 = 0.005 (0.27x, 0.03) | 0.018 (0.37) |
| crowded | all | 0.035 (0.39) | 0.473 (13.7x) | 0.009 + 1.669 = 1.678 (48.59x, 1.65) | 0.009 + 0.119 = 0.128 (3.71x, 0.15) | 0.000 + 0.492 = 0.492 (14.24x, 0.12) | n/a | 0.000 + 0.020 = 0.021 (0.65x, 0.03) | 0.031 (0.39) |
| crowded | knn2 | 0.040 (0.52) | 0.079 (2.0x) | 0.009 + 0.013 = 0.022 (0.54x, 0.03) | n/a | 0.000 + 0.283 = 0.283 (7.02x, 0.04) | 0.000 + 0.277 = 0.278 (6.90x, 0.03) INEXACT 100000 | 0.000 + 0.116 = 0.117 (2.71x, 0.04) | 0.043 (0.52) |

### N = 1e+06

| layout | mode | torchsky CUDA s (GiB) | cKDTree 8 thr | cupy_kdtree | cupy_kdtree_kgrow | torch_cluster | cuvs_brute_force | open3d_nns | torchsky (open3d run) |
|---|---|---|---|---|---|---|---|---|---|
| sparse | best | 0.068 (0.33) | 0.742 (11.0x) | 0.117 + 0.016 = 0.133 (1.97x, 0.14) | n/a | 0.000 + 23.192 = 23.192 (343.03x, 0.21) | 0.001 + 28.071 = 28.071 (415.20x, 0.06) INEXACT 985502 | 0.000 + 0.071 = 0.072 (1.27x, 0.09) | 0.057 (0.33) |
| sparse | all | 0.053 (0.34) | 2.144 (40.3x) | 0.117 + 131.547 = 131.664 (2475.09x, 1.77) | 0.117 + 0.118 = 0.235 (4.42x, 0.33) | 0.000 + 18.768 = 18.768 (352.82x, 0.27) | n/a | 0.000 + 0.141 = 0.141 (2.28x, 0.10) | 0.062 (0.34) |
| sparse | knn2 | 0.114 (0.50) | 0.848 (7.4x) | 0.117 + 0.044 = 0.161 (1.41x, 0.16) | n/a | 0.000 + 26.544 = 26.544 (231.94x, 0.21) | 0.001 + 28.159 = 28.159 (246.05x, 0.05) INEXACT 999333 | 0.000 + 11.639 = 11.639 (106.10x, 0.10) | 0.110 (0.50) |
| dense | best | 0.056 (0.33) | 0.598 (10.6x) | 0.111 + 0.016 = 0.128 (2.27x, 0.14) | n/a | 0.000 + 23.150 = 23.150 (412.11x, 0.21) | 0.001 + 28.087 = 28.088 (500.01x, 0.06) INEXACT 999951 | 0.000 + 0.068 = 0.069 (1.21x, 0.09) | 0.057 (0.33) |
| dense | all | 0.050 (0.34) | 1.830 (36.3x) | 0.111 + 130.930 = 131.042 (2599.73x, 1.77) | 0.111 + 0.134 = 0.246 (4.88x, 0.33) | 0.000 + 18.822 = 18.822 (373.41x, 0.27) | n/a | 0.000 + 0.145 = 0.145 (2.27x, 0.10) | 0.064 (0.34) |
| dense | knn2 | 0.113 (0.49) | 0.800 (7.1x) | 0.111 + 0.049 = 0.161 (1.42x, 0.16) | n/a | 0.000 + 26.578 = 26.578 (235.44x, 0.21) | 0.001 + 28.103 = 28.103 (248.96x, 0.05) INEXACT 1000000 | 0.000 + 11.635 = 11.635 (105.06x, 0.10) | 0.111 (0.49) |
| crowded | best | 0.172 (0.60) | 0.760 (4.4x) | 0.085 + 0.041 = 0.127 (0.74x, 0.14) | n/a | 0.000 + 23.125 = 23.125 (134.72x, 0.21) | 0.001 + 28.052 = 28.053 (163.43x, 0.06) INEXACT 999960 | 0.000 + 0.091 = 0.091 (0.54x, 0.09) | 0.168 (0.60) |
| crowded | all | 0.324 (1.33) | 3.919 (12.1x) | 0.085 + 141.587 = 141.673 (437.88x, 1.77) | 0.086 + 1.384 = 1.469 (4.54x, 1.30) | 0.000 + 43.650 = 43.650 (134.91x, 1.65) | n/a | 0.000 + 0.313 = 0.313 (0.96x, 0.31) | 0.327 (1.33) |
| crowded | knn2 | 0.264 (1.10) | 0.816 (3.1x) | 0.085 + 0.091 = 0.176 (0.67x, 0.16) | n/a | 0.000 + 26.536 = 26.536 (100.56x, 0.21) | 0.001 + 28.066 = 28.067 (106.36x, 0.05) INEXACT 1000000 | 0.000 + 11.640 = 11.641 (44.38x, 0.10) | 0.262 (1.10) |

### N = 1e+07

| layout | mode | torchsky CUDA s (GiB) | cKDTree 8 thr | cupy_kdtree | cupy_kdtree_kgrow | torch_cluster | cuvs_brute_force | open3d_nns | torchsky (open3d run) |
|---|---|---|---|---|---|---|---|---|---|
| sparse | best | 0.520 (1.79) | 7.571 (14.5x) | 1.748 + 0.184 = 1.932 (3.71x, 1.36) | n/a | skip | skip | 0.008 + 0.947 = 0.955 (1.80x, 0.72) | 0.530 (1.79) |
| sparse | all | 0.536 (1.91) | 18.408 (34.3x) | skip | 1.747 + 1.372 = 3.120 (5.82x, 2.86) | skip | n/a | 0.008 + 1.826 = 1.833 (3.23x, 0.75) | 0.568 (1.91) |
| sparse | knn2 | 1.093 (4.18) | 8.606 (7.9x) | 1.748 + 0.516 = 2.264 (2.07x, 1.51) | n/a | skip | skip | timeout (900 s) | 1.093 (4.18) |
| dense | best | 0.698 (1.79) | 9.305 (13.3x) | 1.653 + 0.225 = 1.879 (2.69x, 1.36) | n/a | skip | skip | 0.008 + 0.928 = 0.936 (1.57x, 0.72) | 0.595 (1.79) |
| dense | all | 0.640 (2.00) | 21.160 (33.1x) | skip | 1.652 + 1.534 = 3.186 (4.98x, 2.86) | skip | n/a | 0.008 + 1.905 = 1.913 (3.16x, 0.78) | 0.605 (2.00) |
| dense | knn2 | 1.119 (4.02) | 10.667 (9.5x) | 1.653 + 0.552 = 2.205 (1.97x, 1.51) | n/a | skip | skip | timeout (900 s) | 1.104 (4.02) |
| crowded | best | 1.643 (1.84) | 8.194 (5.0x) | 1.026 + 0.449 = 1.475 (0.90x, 1.36) | n/a | skip | skip | 0.008 + 1.165 = 1.172 (0.69x, 0.72) | 1.710 (1.84) |
| crowded | all | 3.081 (9.93) | 35.430 (11.5x) | skip | 1.027 + 15.347 = 16.374 (5.31x, 8.28) | skip | n/a | 0.007 + 3.226 = 3.234 (1.06x, 3.10) | 3.037 (9.93) |
| crowded | knn2 | 2.542 (5.08) | 8.916 (3.5x) | 1.026 + 0.958 = 1.985 (0.78x, 1.51) | n/a | skip | skip | timeout (900 s) | 2.525 (5.08) |

JSON: `canfar_catalog-match-gpu-peers-20261119-211019_catalog_match_gpu_peers.json`, `canfar_catalog-match-gpu-peers-20261119-213251_catalog_match_gpu_peers.json`.
