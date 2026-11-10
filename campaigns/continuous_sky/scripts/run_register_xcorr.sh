#!/usr/bin/env bash
# CPU job (no GPU): catalogue-free cross-correlation registration of the plan's _cal exposures (register_xcorr.py), parallel over tile x band tasks (multiprocessing).
# Launch (8 cores, 64 GB):
#   canfar create headless images.canfar.net/astroai/base:latest -n rxcorr1 -c 8 -m 64 -e XBAND=1 -- bash /arc/projects/ots/adass2026/scripts/run_register_xcorr.sh
# env (all optional):
#   OUT        output dir            (default /arc/projects/ots/adass2026/p1/register_xcorr_v1)
#   PLAN       plan json             (default /arc/projects/ots/adass2026/p1/plan_lmc.json)
#   WORKERS    processes             (default 8)
#   XBAND      1 = add cross-band correlation edges of adjacent bands (F150W:F200W, F277W:F356W); WITHOUT it the SW/LW simultaneity links only F150W+F277W and F200W+F356W
#              (+F090W via F356W) separately: the relative offset between those families is then not measured (the summary says so)
#   UNIT       group (default) | detector        SIMILARITY  1 = per dither group scale + rotation (implies UNIT=detector)
#   WCS_MODE   fits (default, SCI-header WCS with SIP as tilefit / the engine) | gwcs (ASDF extension; asdf + gwcs only, no jwst)
#   GAUGE      all (default) | prog:<id>        TILES / BANDS  comma lists        RESUME  1 = reuse OUT/work/edges_*.json
#   COMPARE    coarse_shifts.json of tilefit A12 (default: OUT/inputs/coarse_shifts_A12.json when present)
#   ARGS       extra register_xcorr.py flags
set -uo pipefail
A=/arc/projects/ots/adass2026
OUT=${OUT:-$A/p1/register_xcorr_v1}
PLAN=${PLAN:-$A/p1/plan_lmc.json}
mkdir -p "$OUT/provenance" "$OUT/work"
exec > >(tee -a "$OUT/session.log") 2>&1
echo "start $(date -u +%FT%TZ) host=$(hostname) nproc=$(nproc) out=$OUT"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvX && /scratch/venvX/bin/pip install -q numpy scipy astropy gwcs asdf || { echo "pip failed"; exit 4; }
P=$OUT/provenance
cp "$A/scripts/register_xcorr.py" "$A/scripts/run_register_xcorr.sh" "$P/" 2>/dev/null
sha256sum "$A/scripts/register_xcorr.py" "$PLAN" > "$P/sha256.txt"
/scratch/venvX/bin/pip freeze > "$P/pip-freeze.txt"
printf '%s\n' "PLAN=$PLAN" "WORKERS=${WORKERS:-8}" "XBAND=${XBAND:-}" "UNIT=${UNIT:-group}" "SIMILARITY=${SIMILARITY:-}" "WCS_MODE=${WCS_MODE:-fits}" "GAUGE=${GAUGE:-all}" "TILES=${TILES:-}" "BANDS=${BANDS:-}" "ARGS=${ARGS:-}" > "$P/launch.env"
CMP=${COMPARE:-}
[ -z "$CMP" ] && [ -f "$OUT/inputs/coarse_shifts_A12.json" ] && CMP=$OUT/inputs/coarse_shifts_A12.json
FLAGS="--plan $PLAN --out $OUT --workers ${WORKERS:-8} --unit ${UNIT:-group} --wcs ${WCS_MODE:-fits} --gauge ${GAUGE:-all}"
[ -n "${XBAND:-}" ] && FLAGS="$FLAGS --xband"
[ -n "${SIMILARITY:-}" ] && FLAGS="$FLAGS --similarity"
[ -n "${TILES:-}" ] && FLAGS="$FLAGS --tiles $TILES"
[ -n "${BANDS:-}" ] && FLAGS="$FLAGS --bands $BANDS"
[ -n "${RESUME:-}" ] && FLAGS="$FLAGS --resume"
[ -n "$CMP" ] && FLAGS="$FLAGS --compare $CMP"
echo "register_xcorr.py $FLAGS ${ARGS:-}"
# shellcheck disable=SC2086
/scratch/venvX/bin/python -u "$A/scripts/register_xcorr.py" $FLAGS ${ARGS:-}
rc=$?
echo "exit=$rc done $(date -u +%FT%TZ)"
exit $rc
