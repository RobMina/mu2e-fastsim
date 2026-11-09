#!/bin/bash
# Arms A0 (seed 20261023) and A0' (floor, other seed) of the stock-equivalent validation on g4ref_v1 shards, serially on
# one GPVM. usage: run_stock_g4ref.sh FIRST LAST [ARM a0|a0p] [DUMP 0|1]   needs site.env; area $CRVVAL_AREA (default
# crvval). DUMP=1 also writes the stock products as tables (crvdump/ in the same file) for the round trip and fast reco.
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
AREA=${CRVVAL_AREA:?set CRVVAL_AREA to your built validation area}
cd "$AREA" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$AREA" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
A=${1:?first}; B=${2:?last}; ARM=${3:-a0}; DUMP=${4:-0}
case $ARM in a0) SEED=20261023 ;; a0p) SEED=20261123 ;; *) echo "unknown arm $ARM"; exit 2 ;; esac
O=${CRVFS_DATA:?source site.env}/valhist/g4ref_v1/$ARM; [ "$DUMP" = 1 ] && O=${O}_dump
mkdir -p "$O/logs" && cd "$O"
printf '#include "CrvFastSim/fcl/valhist_stock.fcl"\nservices.SeedService.baseSeed : %s\n' "$SEED" > "logs/job.fcl"
[ "$DUMP" = 1 ] && echo 'physics.ValPath : [ Validation, crvDigiDQM, crvRecoDQM, dqmCrv, crvdump ]' >> logs/job.fcl
for k in $(seq "$A" "$B"); do
  S=$(printf "001400_%08d" "$k")
  [ -s "valhist.$S.root" ] && continue
  DTS=$CRVFS_PNFS/g4ref_v1/dts.$CRVFS_OWNER.CosmicCRYExtractedRef.v0.$S.art
  FHICL_FILE_PATH="$O/logs:$FHICL_FILE_PATH" mu2e -c job.fcl -s "$DTS" -T "valhist.$S.root" > "logs/$S.log" 2>&1
  echo "$S rc=$? $(grep -h 'Events total' logs/$S.log)"
done
