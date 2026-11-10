#!/usr/bin/env bash
A=/arc/projects/ots/adass2026
OUT=$A/protocol/v1/logs; mkdir -p $OUT
exec > >(tee $OUT/probe2.log) 2>&1
echo "start $(date -u +%FT%TZ)"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp; mkdir -p $PIP_CACHE_DIR $TMPDIR
python3 -m venv /scratch/venvJ && /scratch/venvJ/bin/pip install -q jwst scipy 2>&1 | grep -v notice | tail -3
echo "installed $(date -u +%FT%TZ)"
(cd /scratch && tar cf $A/protocol/v1/code/venvJ.tar venvJ && ls -la $A/protocol/v1/code/venvJ.tar; echo "tarred $(date -u +%FT%TZ)")
/scratch/venvJ/bin/python - <<'P' 2>&1 | grep -v Warning
import inspect, jwst
from jwst.tweakreg import TweakRegStep
from jwst.resample import ResampleStep
from jwst.outlier_detection import OutlierDetectionStep
from jwst.skymatch import SkyMatchStep
from jwst.datamodels import ModelLibrary
print(jwst.__file__)
for S in (TweakRegStep,ResampleStep,OutlierDetectionStep,SkyMatchStep):
    print('=====',S.__name__, inspect.signature(S.process)); print(inspect.getdoc(S.process) or ''); 
print('=====TWEAKREG SOURCE'); print(inspect.getsource(TweakRegStep.process))
print('=====ModelLibrary'); print(inspect.getdoc(ModelLibrary)); print(inspect.signature(ModelLibrary.__init__))
print([m for m in dir(ModelLibrary) if not m.startswith('__')])
P
echo done
