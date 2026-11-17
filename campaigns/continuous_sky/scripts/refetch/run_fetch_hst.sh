#!/usr/bin/env bash
# 16 Nov 2026: restore the HST exposures of SMACS J0723 (116 unique exposures of scripts/hst_imaging_unique.csv) after the deletion of
# /arc/projects/ots/torchsky-data. Public data from MAST. Idempotent (sizes are checked).
S=/arc/projects/ots/adass2026/scripts; D=/arc/projects/ots/torchsky-data/hst_smacs0723/raw
mkdir -p $D; L=/arc/projects/ots/torchsky-data/hst_smacs0723/fetch_hst.log
echo "start $(date -u +%FT%TZ)" >> $L
python3 - $S/hst_imaging_unique.csv /scratch/hst_manifest.csv <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
w = csv.writer(open(sys.argv[2], "w", newline="")); w.writerow(["filename", "dataURI", "size_bytes"])
for r in rows: w.writerow([r["product_filename"], r["dataURI"], r["size_bytes"]])
print(len(rows), "rows")
PY
python3 $S/union/fetch_mast.py --workers 4 /scratch/hst_manifest.csv $D >> $L 2>&1; echo "hst exit=$?" >> $L
cp $S/hst_imaging_unique.csv /arc/projects/ots/torchsky-data/hst_smacs0723/ 2>/dev/null
echo "$(ls $D | wc -l) files, $(du -sh $D | cut -f1)" >> $L; echo "end $(date -u +%FT%TZ)" >> $L
