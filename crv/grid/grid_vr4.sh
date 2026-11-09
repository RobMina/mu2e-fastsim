#!/bin/bash
# Grid worker: ONE g4ref_v1 shard of V-R4 (fast chain vs stock chain from the same G4 CrvSteps), JAX chain on CPU.
#   grid_vr4.sh OUTDIR INDIR [extra reco-vr4 args]
# $PROCESS selects the shard (shard = PROCESS + 1 + SHARD_OFFSET). Code + JAX from the CPU container image (SIF, pulled
# with ifdh from pnfs; nothing built on the worker), lookup tables from CVMFS, inputs (step table, stock ref, CRVCalib
# CSVs) ifdh-copied from INDIR (pnfs scratch), output: one vr4.<shard>.json (+ log) to OUTDIR. Submitted by
# submit_vr4.sh. No `set -e` (a failing step must name itself); `set -u` only after the mu2e setup.
OUTDIR="${1:?outdir}"; INDIR="${2:?indir}"; shift 2
P="${PROCESS:-0}"; K=$(printf "001400_%08d" $((P + 1 + ${SHARD_OFFSET:-0})))
die() { echo "FATAL[vr4 $K]: $*" >&2; exit 1; }
echo "=== vr4 shard $K host=$(hostname) PROCESS=$P SIF=${SIF_PNFS:-<unset>} ==="
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || echo "(setupmu2e-art rc=$?)"
command -v ifdh >/dev/null 2>&1 || setup ifdhc || die "no ifdh"
set -u
if ifdh ls "$OUTDIR/vr4.$K.json" >/dev/null 2>&1; then echo "SKIP: $OUTDIR/vr4.$K.json exists"; exit 0; fi
mkdir -p in || die "mkdir"
for f in "${SIF_PNFS:?}" "${CODE_PNFS:?}" "$INDIR/crvsteps.$K.root" "$INDIR/ref.$K.root" "$INDIR/crvcalib.csv" "$INDIR/crvphoton.csv"; do
  ifdh cp "$f" "in/$(basename "$f")" || die "stage-in failed: $f"
done
echo "staged: $(du -sh in | cut -f1)"
# code at the commit the submit host pinned (git archive of core + crv), on PYTHONPATH ahead of the image's copy
mkdir -p code && tar -xzf "in/$(basename "$CODE_PNFS")" -C code || die "code tarball"
cat code/VERSION 2>/dev/null
A=$(command -v apptainer || echo /usr/bin/apptainer)
# The job already runs inside the worker's apptainer container, without /dev/fuse: a SIF cannot be mounted there.
# Unpack it to a sandbox directory (no FUSE needed) and run that (nested, unprivileged).
SB=in/sandbox
$A build --sandbox "$SB" "in/$(basename "$SIF_PNFS")" > build_sandbox.txt 2>&1 || { tail -5 build_sandbox.txt; die "sandbox build failed"; }
rm -f "in/$(basename "$SIF_PNFS")"
echo "sandbox: $(du -sh $SB | cut -f1)"
X="$A exec --cleanenv -B $PWD --env PYTHONPATH=$PWD/code/core:$PWD/code/crv --env CRVFS_LOOKUP_DIR=/cvmfs/mu2e.opensciencegrid.org/DataFiles/CRVConditions/v6_0 --env MU2E_SEARCH_PATH=/opt/offline_data $SB"
$X cat /opt/mu2e-fastsim/VERSION || die "container did not start"
/usr/bin/time -v $X python -m crvfastsim reco-vr4 in/crvsteps.$K.root --refs in/ref.$K.root --calib in/crvcalib.csv \
  --deviation in/crvphoton.csv --jax --seed $((P + 1)) --emit vr4.$K.json --outdir . "$@" > vr4.$K.txt 2>&1
rc=$?; grep -v "^INFO" vr4.$K.txt | tail -20
[ $rc -eq 0 ] && [ -s vr4.$K.json ] || die "reco-vr4 rc=$rc, no output"
ifdh cp vr4.$K.json "$OUTDIR/vr4.$K.json" || die "stage-out json"
ifdh cp vr4.$K.txt "$OUTDIR/vr4.$K.txt" || echo "(log stage-out failed)"
rm -rf in
echo "=== done $K -> $OUTDIR ==="
