#!/bin/bash
# Arm A1 (fast steps -> stock digi + reco -> stock histograms) on g4ref_v1 shards, on the gen stream, serially on one
# GPVM. usage: run_faststeps_g4ref.sh FIRST LAST TABLEDIR [TAG] [PREFIX]
#   TABLEDIR: dir under $CRVFS_DATA holding <PREFIX>.<subrun>.root step tables (default PREFIX fast1b: tier 1a + 1b; its
#   library/sampler version is recorded in logs/provenance.txt); TAG names the output arm (default a1_<TABLEDIR>).
#   Reference on the same events: TABLEDIR g4ref_v1_tables, PREFIX crvsteps (G4's own steps injected, arm a0g).
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
AREA=${CRVVAL_AREA:?set CRVVAL_AREA to your built validation area}
cd "$AREA" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$AREA" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
A=${1:?first}; B=${2:?last}; TD=${3:?table dir}; TAG=${4:-a1_$TD}; PFX=${5:-fast1b}
O=${CRVFS_DATA:?source site.env}/valhist/g4ref_v1/$TAG
mkdir -p "$O/logs" && cd "$O"
echo "tables $CRVFS_DATA/$TD; area $AREA backing $(readlink "$AREA/backing"); $(date -Is)" >> logs/provenance.txt
ls "$CRVFS_DATA/$TD"/*.json "$CRVFS_DATA/$TD"/*.txt 2>/dev/null >> logs/provenance.txt
for k in $(seq "$A" "$B"); do
  S=$(printf "001400_%08d" "$k")
  [ -s "valhist.$S.root" ] && continue
  TAB=$CRVFS_DATA/$TD/$PFX.$S.root
  [ -s "$TAB" ] || { echo "$S missing $TAB"; continue; }
  printf '#include "CrvFastSim/fcl/valhist_faststeps.fcl"\nphysics.producers.fastCrvSteps.tableFile : "%s"\n' "$TAB" > "logs/$S.fcl"
  GEN=$CRVFS_PNFS/g4ref_v1/gen.$CRVFS_OWNER.CosmicCRYExtractedRef.v0.$S.art
  FHICL_FILE_PATH="$O/logs:$FHICL_FILE_PATH" mu2e -c "$S.fcl" -s "$GEN" -T "valhist.$S.root" > "logs/$S.log" 2>&1
  echo "$S rc=$? $(grep -h 'Events total' logs/$S.log)"
done
