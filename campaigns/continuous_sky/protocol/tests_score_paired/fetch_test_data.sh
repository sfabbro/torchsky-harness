#!/usr/bin/env bash
# Fetch the small real-data sample used by the tests (read-only).  Run from this folder.
#  * predictions and masks: CANFAR protocol folder, via /home/claude/arc/arcget.sh
#  * SCI arrays: the public _cal files from MAST (first 20 MB of each file only: primary header + SCI).
#    The _cal files on CANFAR (/arc/projects/ots/torchsky-data/jwst_smacs0723/cal) could not be read from this machine.
set -euo pipefail
G=/home/claude/arc/arcget.sh; P=projects/ots/adass2026/protocol/v1; D=${1:-data}
ROOTS="jw02736001001_02101_00003_nrcb3 jw02736001001_02101_00004_nrcblong jw02736001001_02101_00007_nrcblong jw02736001001_02101_00009_nrcb1
jw02736001001_02103_00003_nrcblong jw02736001001_02103_00005_nrcblong jw02736001001_02103_00007_nrcb3 jw02736001001_02103_00009_nrcb1
jw02736001001_02105_00002_nrcblong jw02736001001_02105_00003_nrcblong jw02736001001_02105_00007_nrcb3 jw02736001001_02105_00009_nrcb1"
for m in fieldop_fix_200 stage3_blot; do mkdir -p $D/pred/SMACS/$m/dither; done
mkdir -p $D/heldout/SMACS $D/cal
for r in $ROOTS; do
  for m in fieldop_fix_200 stage3_blot; do [ -s $D/pred/SMACS/$m/dither/$r.npz ] || $G $P/pred/SMACS/$m/dither/$r.npz $D/pred/SMACS/$m/dither/$r.npz; done
  [ -s $D/heldout/SMACS/$r.npz ] || $G $P/heldout/SMACS/$r.npz $D/heldout/SMACS/$r.npz
  [ -s $D/cal/${r}_cal.fits ] || curl -sS -L --fail --retry 3 -r 0-19999999 -o $D/cal/${r}_cal.fits "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/${r}_cal.fits"
done
cp ../out_splits_SMACS.json $D/splits_SMACS.json
du -sh $D
