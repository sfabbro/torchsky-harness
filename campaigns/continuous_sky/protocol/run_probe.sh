#!/usr/bin/env bash
A=/arc/projects/ots/adass2026
OUT=$A/protocol/v1/logs; mkdir -p $OUT
exec > >(tee $OUT/probe.log) 2>&1
echo "start $(date -u +%FT%TZ) $(hostname) nproc=$(nproc)"; free -g|head -2; df -h /scratch | tail -1; python3 --version
for u in https://jwst-crds.stsci.edu https://gea.esac.esa.int/tap-server/tap https://vizier.cds.unistra.fr/viz-bin/votable https://vizier.cfa.harvard.edu https://gsss.stsci.edu/webservices/vo/CatalogSearch.aspx https://mast.stsci.edu https://pypi.org/simple/jwst/ https://archive.stsci.edu; do
  echo "$u -> $(curl -s -m 20 -o /dev/null -w '%{http_code}' $u)"; done
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp; mkdir -p $PIP_CACHE_DIR $TMPDIR
python3 -m venv /scratch/venvJ && /scratch/venvJ/bin/pip install -q jwst 2>&1 | tail -3
/scratch/venvJ/bin/python -c "import jwst,astropy,numpy;print('jwst',jwst.__version__,astropy.__version__,numpy.__version__)"
/scratch/venvJ/bin/pip freeze > $OUT/probe_pipfreeze.txt
/scratch/venvJ/bin/python - <<'P'
import inspect
from jwst.tweakreg import TweakRegStep
from jwst.resample import ResampleStep
from jwst.outlier_detection import OutlierDetectionStep
from jwst.skymatch import SkyMatchStep
for S in (TweakRegStep,ResampleStep,OutlierDetectionStep,SkyMatchStep):
    print(S.__name__); print(S.spec)
P
echo done
