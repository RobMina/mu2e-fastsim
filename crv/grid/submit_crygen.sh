#!/usr/bin/env bash
# submit_crygen.sh [--test | --cluster] [-N njobs] [--nevts N] [--outdir DIR] [--config extracted|run1a|run2]
#                  [--subrun0 S] [--baseseed B]   (run1a: subruns 3001.., run2: 4001..; fcl/gen_only_<config>.fcl)
# CRY-only gen tables for mu2e-crv-fastsim chain_point (the 20 g4ref_v1 shards are too few for production points):
# NJOBS x NEVTS generated events, generator + genCounter + cosmicTimeOffset only (fcl/gen_only_extracted.fcl; g4run is
# initialised but tracks nothing). Shard k: subrun SUBRUN0+k (2001.., disjoint from g4ref_v1 1-20 and gen_as_v1 1001-1020), baseSeed
# BASESEED+k. Sizing (GPVM 2026-10-22): 2.1 ms CPU per generated event, VmHWM 1.9 GB -> 1 M events ~35-60 min.
# Prereq on the submit host: setupmu2e-art.sh, muse setup ops, valid token, site.env (CRVFS_PNFS). Run from a data-area dir.
set -euo pipefail
MODE="test"; NJOBS=100; NEVTS=1000000; BASESEED=20263000; SUBRUN0=2001; CFG=extracted
OUTDIR="${CRVFS_PNFS:?source site.env}/crygen_v1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test) MODE="test" ;; --cluster) MODE="cluster" ;;
    -N) NJOBS="${2:?}"; shift ;; --nevts) NEVTS="${2:?}"; shift ;; --outdir) OUTDIR="${2:?}"; shift ;;
    --config) CFG="${2:?}"; shift ;; --subrun0) SUBRUN0="${2:?}"; shift ;; --baseseed) BASESEED="${2:?}"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac; shift
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$HERE/grid_crygen.sh"; chmod +x "$WRAPPER"
FCL_PNFS="$OUTDIR/fcl"
OUTDIR_TOKEN="$(echo "$OUTDIR" | sed 's|^/pnfs/mu2e|/mu2e|')"

ifdh mkdir_p "$FCL_PNFS" 2>/dev/null || true
for f in gen_only_extracted.fcl gen_only_$CFG.fcl $( [ "$CFG" = run2 ] && echo geom_cosmic_run2_v10.txt ); do
  ifdh rm "$FCL_PNFS/$f" 2>/dev/null || true; ifdh cp "$HERE/fcl/$f" "$FCL_PNFS/$f"
done

[[ "$MODE" == "test" ]] && NJOBS=1
echo "Mode=$MODE njobs=$NJOBS nevts/job=$NEVTS baseSeed=$BASESEED subrun0=$SUBRUN0 out=$OUTDIR"
jobsub_submit -N "$NJOBS" -G mu2e --role=Analysis --expected-lifetime=3h \
  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC \
  --lines='+FERMIHTC_AutoRelease=True' --memory=2500MB --disk=2000MB \
  --need-storage-modify="$OUTDIR_TOKEN" \
  -e CFG="$CFG" -e NEVTS="$NEVTS" -e BASESEED="$BASESEED" -e SUBRUN0="$SUBRUN0" -e FCL_PNFS="$FCL_PNFS" -e MUSING=MDC2025ay \
  -e OWNER="${CRVFS_OWNER:-$USER}" \
  "file://$WRAPPER" "$OUTDIR"
echo "Monitor: jobsub_q -G mu2e --user $USER"
