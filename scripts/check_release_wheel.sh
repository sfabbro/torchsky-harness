#!/usr/bin/env bash
# Check one published torchsky wheel in a fresh uv venv on one Python.
#
#   check_release_wheel.sh PYTHON WHEEL TESTS_DIR QUICKSTART_DIR WORKDIR
#
#   PYTHON          e.g. 3.15 (uv fetches a managed interpreter when needed)
#   WHEEL           the release wheel, already checked against its manifest
#   TESTS_DIR       a copy of torchsky's tests/ at the release tag, with
#                   smoke_installed_wheel.py beside it. The copy's conftest
#                   prepends <copy>/../src, which does not exist, so the
#                   installed wheel is what the tests import.
#   QUICKSTART_DIR  qs*.py: the runnable snippets of the quickstart guide
#   WORKDIR         venvs and logs (venv<PY>/, install<PY>.log, pytest<PY>.log)
#
# Environment: RELEASE_CHECK_WORKERS (pytest -n, default 2),
# RELEASE_CHECK_K (pytest -k, default "not mps": PyTorch 2.15's MPS matmul
# aborts on macOS 14, a PyTorch bug recorded in torchsky's release runbook).
# Exit status is non-zero when any step fails; every step still runs.
set -uo pipefail
PY="$1"; WHEEL="$2"; TESTS="$3"; QS="$4"; WORK="$5"
WORKERS="${RELEASE_CHECK_WORKERS:-2}"
KEXPR="${RELEASE_CHECK_K:-not mps}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
VENV="${WORK}/venv${PY}"
status=0
step() { printf '%-26s ' "py${PY} $1:"; }

rm -rf "${VENV}"
step venv
uv venv -q -p "${PY}" --no-project "${VENV}" && echo ok || { echo FAIL; exit 1; }
step install
if uv pip install -q -p "${VENV}/bin/python" "${WHEEL}" pytest pytest-xdist astropy scipy psutil \
  >"${WORK}/install${PY}.log" 2>&1; then
  echo ok
else
  echo FAIL; tail -5 "${WORK}/install${PY}.log"; exit 1
fi
"${VENV}/bin/python" -I -c "
import platform, sys, numpy, torch, torchfits, torchsky
from torchsky import _cpp
print('  ', platform.platform(), 'python', sys.version.split()[0], 'torch', torch.__version__,
      'torchfits', torchfits.__version__, 'numpy', numpy.__version__,
      'torchsky', torchsky.__version__, 'isa', _cpp.selected_cpu_isa())
assert '/site-packages/torchsky/' in torchsky.__file__, torchsky.__file__
" || status=1

step smoke
if out=$(cd "${WORK}" && "${VENV}/bin/python" -I "${TESTS}/../smoke_installed_wheel.py" 2>&1); then
  echo ok
else
  echo FAIL; echo "${out}" | tail -8; status=1
fi
for f in "${QS}"/qs*.py; do
  step "$(basename "${f}")"
  if out=$(cd "${WORK}" && "${VENV}/bin/python" -I "${f}" 2>&1); then
    echo ok
  else
    echo FAIL; echo "${out}" | tail -8; status=1
  fi
done

step tests
cd "$(dirname "${TESTS}")" || exit 1
"${VENV}/bin/python" -I -m pytest -n "${WORKERS}" -q -p no:cacheprovider -k "${KEXPR}" \
  tests/test_wcs_compile.py tests/test_wcs_tpv.py tests/test_wcs_projection_domains.py \
  tests/test_wcs_projection_matrix.py tests/test_healpix_primitives.py tests/test_healpix_compile.py \
  tests/test_association.py tests/test_crossmatch_engine.py >"${WORK}/pytest${PY}.log" 2>&1
rc=$?
tail -1 "${WORK}/pytest${PY}.log"
if [[ ${rc} -ne 0 ]]; then
  grep -E "^(FAILED|ERROR)" "${WORK}/pytest${PY}.log" | head -20
  status=1
fi
exit "${status}"
