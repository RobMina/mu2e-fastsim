#!/bin/bash
# V2 path A on one g4ref_v1 shard: stock CRV chain on G4 CrvSteps (ref) and on fast-sim CrvSteps (fast). Needs site.env.
# usage: run_v2_arms.sh <shard 1..20> [stage C]
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${CRVFS_CODE:?source site.env}/art" && muse setup >/dev/null 2>&1 || true
set -u
K=${1:?shard}; ST=${2:-C}
S=$(printf "001400_%08d" "$K")
DTS=$CRVFS_PNFS/g4ref_v1/dts.$CRVFS_OWNER.CosmicCRYExtractedRef.v0.$S.art
O=$CRVFS_DATA/v2_fast${ST}_v1
mkdir -p "$O/logs" && cd "$O"
[ -s "ref.$S.root" ] || mu2e -c CrvFastSim/fcl/dump_crvreco_from_dts.fcl -s "$DTS" -T "ref.$S.root" > "logs/ref.$S.log" 2>&1
echo "ref rc=$?"
printf '#include "CrvFastSim/fcl/dump_crvreco_from_fasttable.fcl"\nphysics.producers.fastCrvSteps.tableFile : "%s"\n' \
  "$CRVFS_DATA/fast${ST}_v1/fast${ST}.$S.root" > "logs/fast.$S.fcl"
FHICL_FILE_PATH="$O/logs:$FHICL_FILE_PATH" mu2e -c "fast.$S.fcl" -s "$DTS" -T "fast.$S.root" > "logs/fast.$S.log" 2>&1
echo "fast rc=$?"
