#!/usr/bin/env bash
# submit_ref_sim.sh [--test | --cluster] [-N njobs] [--nevts N] [--outdir DIR]
# G4 reference sample for mu2e-crv-fastsim validation (plan section 5): NJOBS x NEVTS generated events of the stock
# ExtractedCRY chain, Musing SimJob MDC2025ay from CVMFS (no tarball). Shard k: subrun k+1, baseSeed BASESEED+k.
# Sizing from the 2026-10-15 GPVM profile: 23 ms CPU per generated event (GPVM) -> 50k events ~ 20-30 min per job.
# Prereq on the submit host: setupmu2e-art.sh, muse setup ops, valid token, site.env (CRVFS_PNFS). Run from a data-area dir.
set -euo pipefail
MODE="test"; NJOBS=20; NEVTS=50000; BASESEED=20261016
OUTDIR="${CRVFS_PNFS:?source site.env}/g4ref_v1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test) MODE="test" ;; --cluster) MODE="cluster" ;;
    -N) NJOBS="${2:?}"; shift ;; --nevts) NEVTS="${2:?}"; shift ;; --outdir) OUTDIR="${2:?}"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac; shift
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$HERE/grid_ref_sim.sh"; chmod +x "$WRAPPER"
FCL_PNFS="$OUTDIR/fcl"
OUTDIR_TOKEN="$(echo "$OUTDIR" | sed 's|^/pnfs/mu2e|/mu2e|')"

ifdh mkdir_p "$FCL_PNFS" 2>/dev/null || true
ifdh rm "$FCL_PNFS/ref_sim_extracted.fcl" 2>/dev/null || true
ifdh cp "$HERE/fcl/ref_sim_extracted.fcl" "$FCL_PNFS/ref_sim_extracted.fcl"

[[ "$MODE" == "test" ]] && NJOBS=1
echo "Mode=$MODE njobs=$NJOBS nevts/job=$NEVTS baseSeed=$BASESEED out=$OUTDIR"
jobsub_submit -N "$NJOBS" -G mu2e --role=Analysis --expected-lifetime=4h \
  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC \
  --lines='+FERMIHTC_AutoRelease=True' --memory=3000MB --disk=4000MB \
  --need-storage-modify="$OUTDIR_TOKEN" \
  -e NEVTS="$NEVTS" -e BASESEED="$BASESEED" -e FCL_PNFS="$FCL_PNFS" -e MUSING=MDC2025ay -e OWNER="${CRVFS_OWNER:-$USER}" \
  "file://$WRAPPER" "$OUTDIR"
echo "Monitor: jobsub_q -G mu2e --user $USER"
