#!/bin/bash
# Dump crygen_v1 gen art files (CRY-only primaries) to flat gen tables for chain_point:
# $CRVFS_PNFS/crygen_v1/gen.*.art -> $CRVFS_DATA/crygen_v1_tables/gen.001400_<subrun>.root (dump_gen.fcl).
# usage: dump_crygen.sh [first_subrun] [last_subrun]   (default 2001 2100); skips tables that exist. One job per GPVM.
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${CRVFS_CODE:?source site.env}/art" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$CRVFS_CODE/art" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
D=$CRVFS_PNFS/crygen_v1
O=$CRVFS_DATA/crygen_v1_tables
mkdir -p "$O/logs" && cd "$O"
n=0; bad=0
for k in $(seq "${1:-2001}" "${2:-2100}"); do
  S=$(printf "001400_%08d" "$k")
  [ -s "gen.$S.root" ] && continue
  mu2e -c CrvFastSim/fcl/dump_gen.fcl -s "$D/gen.$CRVFS_OWNER.CosmicCRYExtractedGen.v1.$S.art" -T "gen.$S.root" \
    > "logs/dumpgen.$S.log" 2>&1 && n=$((n+1)) || { bad=$((bad+1)); rm -f "gen.$S.root"; echo "FAILED $S"; }
done
echo "dumped $n, failed $bad, tables $(ls gen.*.root 2>/dev/null | wc -l) $(date -Is)"
