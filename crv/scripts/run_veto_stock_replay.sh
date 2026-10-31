#!/bin/bash
# Run one bounded stock replay after the user builds CrvReplayEventFromTable; never overwrite an output tag.
set -e
AREA=${1:?built crv/art muse area}
TABLE=${2:?raw replay ROOT table}
OUT=${3:?new output directory under data area}
EVENTS=${4:?event count}
SEED=${5:-7311}
[ "${MUSE_WORK_DIR:-}" = "$AREA" ] || { echo "wrong MUSE_WORK_DIR=${MUSE_WORK_DIR:-unset}"; exit 2; }
[ -f "$TABLE" ] || { echo "missing replay table"; exit 2; }
[[ "$EVENTS" =~ ^[1-9][0-9]*$ ]] || { echo "positive integer event count required"; exit 2; }
case "$OUT" in "${M2FS_DATA:?set M2FS_DATA to the writable data area (see site.env.example)}"/*) ;; *) echo "output must be in the data area"; exit 2;; esac
mkdir "$OUT"
cat > "$OUT/replay.fcl" <<EOF
#include "CrvFastSim/fcl/replay_crv_run1a.fcl"
physics.producers.EWMProducer.tableFile : "$TABLE"
physics.producers.fastCrvSteps.tableFile : "$TABLE"
source.maxEvents : $EVENTS
services.SeedService.baseSeed : $SEED
EOF
printf 'area %s\nbacking %s\ntable %s\nevents %s\n' "$AREA" "$(readlink "$AREA/backing")" "$TABLE" "$EVENTS" > "$OUT/provenance.txt"
sha256sum "$TABLE" "$AREA/CrvFastSim/src/CrvReplayEventFromTable_module.cc" >> "$OUT/provenance.txt"
cd "$OUT"
mu2e -c "$OUT/replay.fcl" -T "$OUT/stock.root" > "$OUT/stock.log" 2>&1 || { tail -60 "$OUT/stock.log"; exit 1; }
tail -25 "$OUT/stock.log"
