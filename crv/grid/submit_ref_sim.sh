#!/usr/bin/env bash
# submit_ref_sim.sh [--test | --cluster] [-N njobs] [--nevts N] [--outdir DIR] [--first-subrun S] [--tag TAG] [--baseseed B]
#                   [--config extracted|run1a|run2]   (fcl/ref_sim_<config>.fcl; default extracted = g4ref_v1)
# G4 reference sample for mu2e-crv-fastsim validation (plan section 5): NJOBS x NEVTS generated events of the stock
# ExtractedCRY chain, Musing SimJob MDC2025ay from CVMFS (no tarball). Shard k: subrun S+k (default S = 1), baseSeed
# BASESEED+S-1+k, so a campaign with a later --first-subrun extends g4ref_v1 without repeating primaries; file tag TAG (v0).
# Sizing from the 2026-10-15 GPVM profile: 23 ms CPU per generated event (GPVM) -> 50k events ~ 20-30 min per job.
# Prereq on the submit host: setupmu2e-art.sh, muse setup ops, valid token, site.env (CRVFS_PNFS). Run from a data-area dir.
set -euo pipefail
MODE="test"; NJOBS=20; NEVTS=50000; BASESEED=20261016; FIRST_SUBRUN=1; TAG=v0; CFG=extracted
OUTDIR="${CRVFS_PNFS:?source site.env}/g4ref_v1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test) MODE="test" ;; --cluster) MODE="cluster" ;;
    -N) NJOBS="${2:?}"; shift ;; --nevts) NEVTS="${2:?}"; shift ;; --outdir) OUTDIR="${2:?}"; shift ;;
    --first-subrun) FIRST_SUBRUN="${2:?}"; shift ;; --tag) TAG="${2:?}"; shift ;; --baseseed) BASESEED="${2:?}"; shift ;;
    --config) CFG="${2:?}"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac; shift
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$HERE/grid_ref_sim.sh"; chmod +x "$WRAPPER"
FCL_PNFS="$OUTDIR/fcl"
DCACHE="${CRVFS_DCACHE:?set CRVFS_DCACHE to the dCache mount point (see site.env.example)}"
OUTDIR_TOKEN="${OUTDIR#"$DCACHE"}"

ifdh mkdir_p "$FCL_PNFS" 2>/dev/null || true
for f in ref_sim_extracted.fcl ref_sim_$CFG.fcl $( [ "$CFG" = run2 ] && echo geom_cosmic_run2_v10.txt ); do
  ifdh rm "$FCL_PNFS/$f" 2>/dev/null || true; ifdh cp "$HERE/fcl/$f" "$FCL_PNFS/$f"
done

[[ "$MODE" == "test" ]] && NJOBS=1
echo "Config=$CFG Mode=$MODE njobs=$NJOBS nevts/job=$NEVTS baseSeed=$BASESEED firstSubrun=$FIRST_SUBRUN tag=$TAG out=$OUTDIR"
jobsub_submit -N "$NJOBS" -G mu2e --role=Analysis --expected-lifetime=4h \
  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC \
  --lines='+FERMIHTC_AutoRelease=True' --memory=3000MB --disk=4000MB \
  --need-storage-modify="$OUTDIR_TOKEN" \
  -e NEVTS="$NEVTS" -e BASESEED="$BASESEED" -e FIRST_SUBRUN="$FIRST_SUBRUN" -e TAG="$TAG" -e CFG="$CFG" -e FCL_PNFS="$FCL_PNFS" -e MUSING=MDC2025ay -e OWNER="${CRVFS_OWNER:?source site.env}" \
  "file://$WRAPPER" "$OUTDIR"
echo "Monitor: jobsub_q -G mu2e --user ${CRVFS_OWNER}"
