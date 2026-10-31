#!/usr/bin/env bash
# Replay admitted Run1A tables through existing DQM clients without rerunning stock response.
set -eo pipefail
AREA=${1:?existing built DQM area}
CODE=${2:?CRV source tree}
DATA=${3:?named study root}
OUT=${4:?new output directory}
NEVENTS=${5:-1000}
[[ "${MUSE_WORK_DIR:-}" = "$AREA" ]] || { echo "wrong MUSE_WORK_DIR"; exit 2; }
mkdir "$OUT"
export FHICL_FILE_PATH="$CODE/art/CrvFastSim/fcl:$FHICL_FILE_PATH"
IN="${CRVFS_PNFS:?source site.env}/g4ref_run1a_v1/dts.${CRVFS_OWNER:?source site.env}.CosmicCRYrun1aRef.v1.001400_00000101.art"
for arm in stock a3 a4; do
  TABLE="$DATA/response_1k_v1/run1a/$arm.root"
  [[ "$arm" = stock ]] && TABLE="$DATA/stock_1k_v1/run1a/crvreco.root"
  {
    echo '#include "named_dqm_run1a.fcl"'
    for module in CrvDigi CrvRecoPulses CrvCoincidenceClusterFinder; do
      printf 'physics.producers.%s.tableFile : "%s"\n' "$module" "$TABLE"
    done
  } > "$OUT/$arm.fcl"
  mu2e -c "$OUT/$arm.fcl" -s "$IN" -n "$NEVENTS" -T "$OUT/$arm.root" > "$OUT/$arm.log" 2>&1 || { tail -60 "$OUT/$arm.log"; exit 1; }
  sha256sum "$TABLE" "$OUT/$arm.fcl" "$OUT/$arm.root" >> "$OUT/manifest.txt"
  grep 'TrigReport Events total' "$OUT/$arm.log"
done
