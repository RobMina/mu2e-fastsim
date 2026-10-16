#!/bin/bash
# V2 path A on one g4ref_v1 shard: stock CRV chain on G4 CrvSteps (ref) and on a CrvStep table (candidate). Needs site.env.
# usage: run_v2_arms.sh <shard 1..20> [cand fast|g4mu] [table dir under CRVFS_DATA]
#   fast: fast-sim table <dir>/fastC.<shard>.root (default dir fastC_v2); g4mu: G4 primary-muon steps (g4muon) in g4mu_v1
# outputs in $CRVFS_DATA/v2_fastC_v1: ref.<shard>.root (made once) and <cand>.<shard>.root
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${CRVFS_CODE:?source site.env}/art" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$CRVFS_CODE/art" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
K=${1:?shard}; CAND=${2:-fast}
S=$(printf "001400_%08d" "$K")
case $CAND in
  fast) TAB=$CRVFS_DATA/${3:-fastC_v2}/fastC.$S.root ;;
  g4mu) TAB=$CRVFS_DATA/${3:-g4mu_v1}/g4mu.$S.root ;;
  *) echo "unknown candidate $CAND"; exit 2 ;;
esac
[ -s "$TAB" ] || { echo "missing $TAB"; exit 2; }
DTS=$CRVFS_PNFS/g4ref_v1/dts.$CRVFS_OWNER.CosmicCRYExtractedRef.v0.$S.art
O=$CRVFS_DATA/v2_fastC_v1
mkdir -p "$O/logs" && cd "$O"
[ -s "ref.$S.root" ] || mu2e -c CrvFastSim/fcl/dump_crvreco_from_dts.fcl -s "$DTS" -T "ref.$S.root" > "logs/ref.$S.log" 2>&1
echo "ref rc=$?"
printf '#include "CrvFastSim/fcl/dump_crvreco_from_fasttable.fcl"\nphysics.producers.fastCrvSteps.tableFile : "%s"\n' \
  "$TAB" > "logs/$CAND.$S.fcl"
FHICL_FILE_PATH="$O/logs:$FHICL_FILE_PATH" mu2e -c "$CAND.$S.fcl" -s "$DTS" -T "$CAND.$S.root" > "logs/$CAND.$S.log" 2>&1
echo "$CAND rc=$?"
