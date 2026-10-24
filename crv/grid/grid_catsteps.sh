#!/bin/bash
# Grid worker, stage 1 of F-fast from beam frames: dump the CrvSteps of a block of beam-frame *Cat dts files to flat
# tables (dump_crvsteps_cat.fcl), compacted to a parquet of the stepped frames (crvfastsim.veto.catsteps, frame count in
# the metadata), one output per input file, skipping existing outputs. Code: CODE_PNFS (git archive of core + crv).
#   grid_catsteps.sh OUTDIR       PROCESS selects block PROCESS of LIST_PNFS (lines "<dataset> <pnfs path>"), BLOCK lines
# Release SimJob MDC2025ay (CVMFS) + CrvFastSim dumper libs/fcl (DUMPER_PNFS); inputs streamed with xrootd.
OUTDIR="${1:?outdir}"; P="${PROCESS:-0}"; B="${BLOCK:?}"
die() { echo "FATAL[catsteps $P]: $*" >&2; exit 1; }
echo "=== catsteps PROCESS=$P BLOCK=$B host=$(hostname) $(date -u)"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || echo "(setupmu2e-art rc=$?)"
muse setup SimJob MDC2025ay || echo "(muse setup rc=$?)"
command -v ifdh >/dev/null 2>&1 || setup ifdhc || die "no ifdh"
set -u
ifdh cp "${LIST_PNFS:?}" list.txt || die "stage list"
ifdh cp "${DUMPER_PNFS:?}" dumper.tgz && mkdir -p dumper && tar -xzf dumper.tgz -C dumper || die "dumper tarball"
export CET_PLUGIN_PATH="$PWD/dumper/lib:$CET_PLUGIN_PATH" LD_LIBRARY_PATH="$PWD/dumper/lib:$LD_LIBRARY_PATH"
export FHICL_FILE_PATH="$PWD/dumper/fcl:$FHICL_FILE_PATH"
ifdh cp "${CODE_PNFS:?}" code.tgz && mkdir -p code && tar -xzf code.tgz -C code || die "code tarball"
cat code/VERSION
PY=/cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0/bin/python3
rc=0; n=0
while read -r DS PATHF; do
  OUT=steps.$(basename "$PATHF" .art).parquet
  if ifdh ls "$OUTDIR/$DS/$OUT" >/dev/null 2>&1; then echo "SKIP $OUT"; continue; fi
  URL=$(echo "$PATHF" | sed 's|^/pnfs/|root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/|')
  mu2e -c CrvFastSim/fcl/dump_crvsteps_cat.fcl -s "$URL" -T dump.root > dump.log 2>&1 || { tail -15 dump.log; echo "DUMP FAILED $PATHF"; rc=3; continue; }
  grep -E "TrigReport Events total|TimeReport CPU" dump.log | tr "\n" " "
  PYTHONPATH=$PWD/code/core:$PWD/code/crv $PY -m crvfastsim.veto.catsteps dump.root --dataset "$DS" --out "$OUT" || { echo "CONVERT FAILED $PATHF"; rc=4; rm -f dump.root; continue; }
  rm -f dump.root
  ifdh cp "$OUT" "$OUTDIR/$DS/$OUT" || { echo "COPY FAILED $OUT"; rc=5; }
  rm -f "$OUT"; n=$((n + 1))
done < <(sed -n "$((P * B + 1)),$(((P + 1) * B))p" list.txt)
echo "=== done PROCESS=$P files=$n rc=$rc"; exit $rc
