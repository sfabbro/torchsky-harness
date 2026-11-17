#!/usr/bin/env bash
# 16 Nov 2026: restore the LMC NIRCam _cal exposures of protocol v1 (programmes 1476 and 4447; 1032 files, about 121 GB) at the paths the
# protocol splits expect. /arc/projects/ots/torchsky-data was deleted by mistake (found missing 13 Nov); public data re-fetched from MAST.
# Idempotent (a file already present is skipped only when the manifest gives its size; here sizes are blank, so present files are re-checked
# by name and skipped by the wrapper below).
S=/arc/projects/ots/adass2026/scripts; D=/arc/projects/ots/torchsky-data/aggregate
mkdir -p $D; L=$D/fetch_lmc.log
echo "start $(date -u +%FT%TZ)" >> $L
for p in 1476_nircam 4447_nircam; do
  mkdir -p $D/$p
  python3 - "$S/onesky/lmc_${p}_cal.csv" "$D/$p" "/scratch/todo_$p.csv" <<'PY'
import csv, os, sys
rows = list(csv.DictReader(open(sys.argv[1])))
todo = [r for r in rows if not (os.path.exists(os.path.join(sys.argv[2], r["filename"])) and os.path.getsize(os.path.join(sys.argv[2], r["filename"])) % 2880 == 0 and os.path.getsize(os.path.join(sys.argv[2], r["filename"])) > 50_000_000)]
w = csv.DictWriter(open(sys.argv[3], "w", newline=""), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(todo)
print(len(rows), "in manifest,", len(todo), "to fetch")
PY
  python3 $S/union/fetch_mast.py --workers 6 /scratch/todo_$p.csv $D/$p >> $L 2>&1; echo "$p exit=$?" >> $L
  echo "$p: $(ls $D/$p | grep -c _cal.fits) files, $(du -sh $D/$p | cut -f1)" >> $L
done
echo "end $(date -u +%FT%TZ)" >> $L
