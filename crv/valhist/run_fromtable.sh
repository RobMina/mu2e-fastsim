#!/bin/bash
# Arms that read CRV products from tables (valhist/fasttables.py output, or stock dumps for the round trip) on g4ref_v1
# shards, serially on one GPVM. usage: run_fromtable.sh FIRST LAST MODE TABLEDIR PREFIX TAG [SRC dts|gen]
#   MODE digis: table digis + stock reco (A2); all: digis, pulses, clusters from the table (A3, A4, E2E, round trip)
#   tables $CRVFS_DATA/TABLEDIR/PREFIX.<subrun>.root; outputs $CRVFS_DATA/valhist/g4ref_v1/TAG/valhist.<subrun>.root
#   SRC: art input, the G4 dts (default, G4-step arms) or the gen stream (fast-step arms)
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
AREA=${CRVVAL_AREA:-/exp/mu2e/app/users/ram2aq/crvval}
cd "$AREA" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$AREA" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
A=${1:?first}; B=${2:?last}; MODE=${3:?digis|all}; TD=${4:?table dir}; PFX=${5:?prefix}; TAG=${6:?tag}; SRC=${7:-dts}
case $MODE in digis) BASE=valhist_digisfromtable.fcl ;; all) BASE=valhist_fromtable.fcl ;; *) echo "bad MODE"; exit 2 ;; esac
O=${CRVFS_DATA:?source site.env}/valhist/g4ref_v1/$TAG
mkdir -p "$O/logs" && cd "$O"
echo "mode $MODE tables $CRVFS_DATA/$TD/$PFX.*; src $SRC; area backing $(readlink "$AREA/backing"); $(date -Is)" >> logs/provenance.txt
for k in $(seq "$A" "$B"); do
  S=$(printf "001400_%08d" "$k")
  [ -s "valhist.$S.root" ] && continue
  TAB=$CRVFS_DATA/$TD/$PFX.$S.root
  [ -s "$TAB" ] || { echo "$S missing $TAB"; continue; }
  { printf '#include "CrvFastSim/fcl/%s"\n' "$BASE"
    for m in CrvDigi CrvRecoPulses CrvCoincidenceClusterFinder; do
      [ "$MODE" = digis ] && [ "$m" != CrvDigi ] && continue
      printf 'physics.producers.%s.tableFile : "%s"\n' "$m" "$TAB"; done; } > "logs/$S.fcl"
  IN=$CRVFS_PNFS/g4ref_v1/dts.$CRVFS_OWNER.CosmicCRYExtractedRef.v0.$S.art
  [ "$SRC" = gen ] && IN=$CRVFS_PNFS/g4ref_v1/gen.$CRVFS_OWNER.CosmicCRYExtractedRef.v0.$S.art
  FHICL_FILE_PATH="$O/logs:$FHICL_FILE_PATH" mu2e -c "$S.fcl" -s "$IN" -T "valhist.$S.root" > "logs/$S.log" 2>&1
  echo "$S rc=$? $(grep -h 'Events total' logs/$S.log)"
done
