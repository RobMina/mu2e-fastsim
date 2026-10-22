#!/usr/bin/env bash
# submit_crygen_dump.sh [--test | --cluster] [-N n]: dump crygen_v1 gen art (pnfs) to gen tables on the grid, one file
# per job (~8 min each, 2 CPU-min). Packs the built CrvStepTableDumper library + CrvFastSim fcl of the art/ muse area
# (backing SimJob MDC2025ay) into a small tarball staged to pnfs. Tables land in $CRVFS_PNFS/crygen_v1_tables; copy to
# $CRVFS_DATA with ifdh afterwards. Prereq: setupmu2e-art.sh, muse setup ops, token, site.env; run from a data-area dir.
set -euo pipefail
MODE=test; NJOBS=100; SUBRUN0=2001
IN="${CRVFS_PNFS:?source site.env}/crygen_v1"; OUT="$CRVFS_PNFS/crygen_v1_tables"
while [[ $# -gt 0 ]]; do case "$1" in --test) MODE=test;; --cluster) MODE=cluster;; -N) NJOBS="$2"; shift;;
  *) echo "Unknown $1"; exit 1;; esac; shift; done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; A="$CRVFS_CODE/art"
T=$(mktemp -d); mkdir -p "$T/lib" "$T/fcl/CrvFastSim/fcl"
cp "$A"/build/*/CrvFastSim/lib/*.so "$T/lib/"; cp "$A"/CrvFastSim/fcl/*.fcl "$T/fcl/CrvFastSim/fcl/"
tar -czf dumper_crygen.tgz -C "$T" lib fcl; rm -rf "$T"
ifdh mkdir_p "$OUT" 2>/dev/null || true; ifdh rm "$OUT/dumper_crygen.tgz" 2>/dev/null || true
ifdh cp dumper_crygen.tgz "$OUT/dumper_crygen.tgz"
[[ $MODE == test ]] && NJOBS=1
TOKEN_DIR="$(echo "$OUT" | sed 's|^/pnfs/mu2e|/mu2e|')"
jobsub_submit -N "$NJOBS" -G mu2e --role=Analysis --expected-lifetime=1h \
  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --lines='+FERMIHTC_AutoRelease=True' \
  --memory=2000MB --disk=2000MB --need-storage-modify="$TOKEN_DIR" \
  -e INDIR="$IN" -e SUBRUN0="$SUBRUN0" -e DUMPER_PNFS="$OUT/dumper_crygen.tgz" -e OWNER="${CRVFS_OWNER:-$USER}" \
  "file://$HERE/grid_crygen_dump.sh" "$OUT"
