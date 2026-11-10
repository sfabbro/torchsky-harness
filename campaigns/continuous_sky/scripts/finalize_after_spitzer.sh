#!/usr/bin/env bash
# Wait for both Spitzer extraction jobs, then finalize the pixel store (6 Nov 2026).
ST=/arc/projects/ots/torchsky-data/smacs0723_union/pixelstore_v1; t=0
until [ -s $ST/done/job-spitzer-0of2.json ] && [ -s $ST/done/job-spitzer-1of2.json ]; do sleep 60; t=$((t+60)); [ $t -ge 14400 ] && { echo "gave up"; exit 3; }; done
STORE=$ST FINALIZE=1 bash /arc/projects/ots/adass2026/scripts/run_pixelstore.sh
