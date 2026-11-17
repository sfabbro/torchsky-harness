#!/usr/bin/env bash
# 17 Nov 2026: LMC NIRCam _rate images (count rates: no flat field, no flux conversion) of the protocol exposures, programmes 1476 then 4447,
# for the test "does the flat field calibrate itself from the science exposures" (680 + 352 files, about 87 GB). Public data from MAST.
S=/arc/projects/ots/adass2026/scripts; D=/arc/projects/ots/torchsky-data/aggregate_rate
mkdir -p $D; L=$D/fetch_lmc_rate.log
echo "start $(date -u +%FT%TZ)" >> $L
for p in 1476_nircam 4447_nircam; do
  mkdir -p $D/$p
  python3 - "$S/onesky/lmc_${p}_rate.csv" "$D/$p" "/scratch/todo_rate_$p.csv" <<'PY'
import csv, os, sys
rows = list(csv.DictReader(open(sys.argv[1])))
todo = [r for r in rows if not (os.path.exists(os.path.join(sys.argv[2], r["filename"])) and os.path.getsize(os.path.join(sys.argv[2], r["filename"])) % 2880 == 0 and os.path.getsize(os.path.join(sys.argv[2], r["filename"])) > 50_000_000)]
w = csv.DictWriter(open(sys.argv[3], "w", newline=""), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(todo)
print(len(rows), "in manifest,", len(todo), "to fetch")
PY
  python3 $S/union/fetch_mast.py --workers 4 /scratch/todo_rate_$p.csv $D/$p >> $L 2>&1; echo "$p exit=$?" >> $L
  echo "$p: $(ls $D/$p | grep -c _rate.fits) files, $(du -sh $D/$p | cut -f1)" >> $L
done
echo "end $(date -u +%FT%TZ)" >> $L
