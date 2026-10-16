#!/bin/bash
# Dump the g4ref_v1 dts/gen art files to flat tables with CrvStepTableDumper, serially on one GPVM (~5 s per file).
# usage: dump_g4ref.sh [first_shard] [last_shard]   (shards 1..20 = subruns); needs site.env (CRVFS_CODE, CRVFS_DATA,
# CRVFS_PNFS, CRVFS_OWNER)
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${CRVFS_CODE:?source site.env}/art" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$CRVFS_CODE/art" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
D=$CRVFS_PNFS/g4ref_v1
O=$CRVFS_DATA/g4ref_v1_tables
mkdir -p "$O/logs" && cd "$O"
for k in $(seq "${1:-1}" "${2:-20}"); do
  S=$(printf "001400_%08d" "$k")
  [ -s "crvsteps.$S.root" ] || mu2e -c CrvFastSim/fcl/dump_crvsteps.fcl -s "$D/dts.$CRVFS_OWNER.CosmicCRYExtractedRef.v0.$S.art" -T "crvsteps.$S.root" > "logs/dump.$S.log" 2>&1
  r1=$?
  [ -s "gen.$S.root" ] || mu2e -c CrvFastSim/fcl/dump_gen.fcl -s "$D/gen.$CRVFS_OWNER.CosmicCRYExtractedRef.v0.$S.art" -T "gen.$S.root" > "logs/dumpgen.$S.log" 2>&1
  echo "shard $k steps_rc=$r1 gen_rc=$? $(date +%T)"
done
echo "done $(date -Is)"
