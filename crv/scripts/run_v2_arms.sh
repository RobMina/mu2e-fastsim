#!/bin/bash
# V2 path A on one g4ref_v1 shard: stock CRV chain on G4 CrvSteps (ref) and on a CrvStep table (candidate). Needs site.env.
# usage: run_v2_arms.sh <shard 1..20> [cand fast|g4mu|fast1b] [table dir under CRVFS_DATA] [output tag]
#   fast: fast-sim table <dir>/fastC.<shard>.root (default dir fastC_v2); g4mu: G4 primary-muon steps (g4muon) in g4mu_v1;
#   fast1b: tier 1a + 1b table <dir>/fast1b.<shard>.root (default dir tier1b_v1)
# [src dts|gen]: the candidate runs on the G4 dts file (accepted events; default) or on the gen-stream file (every
#   generated event: needed when the candidate has deposits in events G4 rejected, e.g. tier-1b showers)
# outputs in $CRVFS_DATA/v2_fastC_v1: ref.<shard>.root (made once) and <tag, default cand>.<shard>.root
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${CRVFS_CODE:?source site.env}/art" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$CRVFS_CODE/art" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
K=${1:?shard}; CAND=${2:-fast}; TAG=${4:-$CAND}; SRC=${5:-dts}
S=$(printf "001400_%08d" "$K")
case $CAND in
  fast) TAB=$CRVFS_DATA/${3:-fastC_v2}/fastC.$S.root ;;
  g4mu) TAB=$CRVFS_DATA/${3:-g4mu_v1}/g4mu.$S.root ;;
  fast1b) TAB=$CRVFS_DATA/${3:-tier1b_v1}/fast1b.$S.root ;;
  *) echo "unknown candidate $CAND"; exit 2 ;;
esac
[ -s "$TAB" ] || { echo "missing $TAB"; exit 2; }
DTS=$CRVFS_PNFS/g4ref_v1/dts.$CRVFS_OWNER.CosmicCRYExtractedRef.v0.$S.art
O=$CRVFS_DATA/v2_fastC_v1
mkdir -p "$O/logs" && cd "$O"
[ -s "ref.$S.root" ] || mu2e -c CrvFastSim/fcl/dump_crvreco_from_dts.fcl -s "$DTS" -T "ref.$S.root" > "logs/ref.$S.log" 2>&1
echo "ref rc=$?"
printf '#include "CrvFastSim/fcl/dump_crvreco_from_fasttable.fcl"\nphysics.producers.fastCrvSteps.tableFile : "%s"\n' \
  "$TAB" > "logs/$TAG.$S.fcl"
CIN=$DTS; [ "$SRC" = gen ] && CIN=$CRVFS_PNFS/g4ref_v1/gen.$CRVFS_OWNER.CosmicCRYExtractedRef.v0.$S.art
FHICL_FILE_PATH="$O/logs:$FHICL_FILE_PATH" mu2e -c "$TAG.$S.fcl" -s "$CIN" -T "$TAG.$S.root" > "logs/$TAG.$S.log" 2>&1
echo "$TAG rc=$?"
