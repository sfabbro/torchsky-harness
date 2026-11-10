#!/usr/bin/env bash
# Wait for $WAIT_FOR (a file on /arc) to exist, then run smacs_run.sh with the same env. Gives up after $WAIT_MAX_S.
W=${WAIT_FOR:?}; M=${WAIT_MAX_S:-28800}; t=0
while [ ! -s "$W" ]; do sleep 60; t=$((t+60)); [ $t -ge $M ] && { echo "gave up waiting for $W"; exit 3; }; done
sleep 120  # let the writer finish
echo "found $W after ${t}s"
exec bash /arc/projects/ots/adass2026/scripts/smacs_run.sh
