#!/usr/bin/env bash
# Install GPU nearest-neighbour comparators for bench_catalog_match_perf.py into
# the active Pixi environment (run as `pixi run -e cuda-bench -- bash THIS OUT_DIR`).
#
# Job-time pip keeps pixi.lock untouched. Every install is constrained to the
# versions already in the environment (torch, numpy, nvidia-* ...), so a peer can
# add packages but never move the solved stack; one that cannot install is logged
# and shows as UNAVAILABLE in the bench. Logs: OUT_DIR/pip_<peer>.log; summary:
# OUT_DIR/GPU_PEERS.json.
set -uo pipefail

out=${1:?usage: install_gpu_nn_peers.sh OUT_DIR}
mkdir -p "${out}"
cons="${TMPDIR:-/tmp}/gpu_peers_constraints.txt"

freeze() {
  python -m pip list --format=freeze --exclude-editable | grep -v ' @ ' > "${cons}"
}

try_pip() {  # label seconds pip-args...
  local label=$1 secs=$2
  shift 2
  freeze
  echo "peer_install ${label}: pip install $* (utc=$(date -u +%FT%TZ))"
  if timeout "${secs}" python -m pip install --progress-bar off -c "${cons}" "$@" \
    > "${out}/pip_${label}.log" 2>&1; then
    echo "peer_install ${label}: ok"
    return 0
  fi
  echo "peer_install ${label}: FAILED"
  tail -n 8 "${out}/pip_${label}.log"
  return 1
}

check() {  # label python-statement
  if python -c "$2" > "${out}/check_${1}.log" 2>&1; then
    echo "peer_check ${1}: ok"
    return 0
  fi
  echo "peer_check ${1}: FAILED"
  tail -n 5 "${out}/check_${1}.log"
  return 1
}

python -m pip --version
try_pip scipy 900 scipy || true

# CuPy: match torch's CUDA 13 runtime first; the cu12 wheel is the fallback.
cupy_ok=0
for spec in "cupy-cuda13x[ctk]" cupy-cuda13x "cupy-cuda12x[ctk]"; do
  if try_pip "cupy_${spec//[^a-z0-9]/_}" 1200 "${spec}" \
    && check cupy 'import cupy, cupyx.scipy.spatial as s; t = s.KDTree(cupy.random.rand(64, 3)); t.query(cupy.random.rand(4, 3), k=2); print(cupy.__version__)'; then
    cupy_ok=1
    break
  fi
  python -m pip uninstall -y cupy-cuda13x cupy-cuda12x > /dev/null 2>&1 || true
done
echo "cupy_ok=${cupy_ok}"

try_pip faiss 1200 faiss-gpu-cu12 && check faiss 'import faiss; assert faiss.get_num_gpus() > 0; print(faiss.__version__)' || true
try_pip cuvs 1800 cuvs-cu13 --extra-index-url https://pypi.nvidia.com \
  && check cuvs 'import torch, cuvs; from cuvs.neighbors import brute_force; x = torch.rand(64, 3, device="cuda"); i = brute_force.build(x); brute_force.search(i, x[:4].contiguous(), 2); print(cuvs.__version__)' \
  || true
try_pip open3d 1200 open3d && check open3d 'import open3d.core as o3c; assert o3c.cuda.is_available(); print("cuda devices", o3c.cuda.device_count())' || true

# torch_cluster has no wheel for torch 2.14 on data.pyg.org: build it for this GPU.
(
  export CUDA_HOME="${CONDA_PREFIX}" FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS:-$(nproc)}"
  TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
  export TORCH_CUDA_ARCH_LIST
  nv_inc="$(python - <<'PY'
import glob
import os
import site

dirs = []
for sp in site.getsitepackages():
    dirs += sorted(glob.glob(os.path.join(sp, "nvidia", "*", "include")))
print(":".join(dirs))
PY
)"
  export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CONDA_PREFIX}/include${nv_inc:+:${nv_inc}}${CPATH:+:${CPATH}}"
  export LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  echo "torch_cluster build: arch=${TORCH_CUDA_ARCH_LIST} nvcc=$(command -v nvcc) gxx=$(command -v g++ || echo none)"
  try_pip build_tools 600 setuptools wheel || true
  try_pip torch_cluster 2700 --no-build-isolation --no-deps "git+https://github.com/rusty1s/pytorch_cluster.git" \
    || try_pip torch_cluster_sdist 2700 --no-build-isolation --no-deps torch-cluster \
    || true
)
check torch_cluster 'import torch, torch_cluster; x = torch.rand(64, 3, device="cuda", dtype=torch.float64); print(torch_cluster.knn(x, x[:4], 2).shape)' || true

python - "${out}/GPU_PEERS.json" <<'PY'
import importlib
import json
import sys

rows = {}
for name, mod in [("cupy", "cupy"), ("faiss", "faiss"), ("cuvs", "cuvs"), ("open3d", "open3d"),
                  ("torch_cluster", "torch_cluster"), ("scipy", "scipy"), ("torch", "torch"), ("numpy", "numpy")]:
    try:
        m = importlib.import_module(mod)
        rows[name] = {"status": "ok", "version": getattr(m, "__version__", "unknown")}
    except Exception as exc:
        rows[name] = {"status": "missing", "error": f"{type(exc).__name__}: {exc}"}
json.dump(rows, open(sys.argv[1], "w"), indent=2)
print(json.dumps(rows))
PY
