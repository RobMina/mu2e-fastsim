#!/bin/bash
# Submit stage 1 of F-fast from beam frames: CrvStep tables of the Run 1A beam-frame *Cat datasets (the MixBackgroundFrames
# inputs of CeMLeadingLogMix1BB.MDC2025au), BLOCK files per job (grid_catsteps.sh). Run on a GPVM from a data-area dir.
#   submit_catsteps.sh --prepare     # file list of the four datasets (on-disk files only) + dumper tarball -> pnfs
#   submit_catsteps.sh --test        # one job (block FIRST_BLOCK, default 0)   submit_catsteps.sh --cluster   # every block
# env: BLOCK (files per job), BASE, LIST (staged list name, default list.txt; e.g. list_rest.txt for the leftovers),
#      LIFETIME (default 3h). MuStopPileupCat files take ~10 min each: keep its blocks small.
BASE=${BASE:-/pnfs/mu2e/scratch/users/ram2aq/crvfs_catsteps_run1a}; IN=$BASE/in; OUT=$BASE/out
C=/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim; HERE=$(cd "$(dirname "$0")" && pwd); MODE="${1:-}"; B=${BLOCK:-40}
DATASETS="EleBeamFlashCat.MDC2025ac MuBeamFlashCat.MDC2025ac MuStopPileupCat.MDC2025ac NeutralsFlashCat.MDC2025ad"
[ -n "$MODE" ] || { sed -n 2,7p "$0"; exit 2; }
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
muse setup ops >/dev/null 2>&1 || true
command -v ifdh >/dev/null 2>&1 || setup ifdhc
set -u
if [ "$MODE" = --prepare ]; then
  : > list.txt
  for d in $DATASETS; do
    case $d in EleBeamFlashCat*) LOC=disk;; *) LOC=tape;; esac   # EleBeamFlashCat lives on persistent (the "disk" location)
    mdh print-url -l $LOC -s path dts.mu2e.$d.art > paths_$d.txt
    if [ $LOC = tape ]; then mdh query-dcache -o -v dts.mu2e.$d.art 2>/dev/null | awk '$1 ~ /ONLINE/ {print $2}' > online_$d.txt
    else xargs -n1 basename < paths_$d.txt > online_$d.txt; fi
    grep -F -f online_$d.txt paths_$d.txt | sort | sed "s|^|${d%%.*} |" >> list.txt
    echo "$d: $(wc -l < paths_$d.txt) files, $(wc -l < online_$d.txt) online"
  done
  wc -l < list.txt
  T=$(mktemp -d); mkdir -p "$T/lib" "$T/fcl/CrvFastSim/fcl"
  cp $C/crv/art/build/*/CrvFastSim/lib/*.so "$T/lib/"
  git -C $C/crv fetch -q origin; git -C $C/crv archive origin/main art/CrvFastSim/fcl | tar -x -C $T && mv $T/art/CrvFastSim/fcl/*.fcl "$T/fcl/CrvFastSim/fcl/" && rm -rf $T/art
  tar -czf dumper.tgz -C "$T" lib fcl; rm -rf "$T"
  T=$(mktemp -d); git -C $C/core fetch -q origin
  mkdir -p $T/core $T/crv && git -C $C/core archive origin/main | tar -x -C $T/core && git -C $C/crv archive origin/main | tar -x -C $T/crv
  printf "core %s\ncrv %s\n" "$(git -C $C/core rev-parse --short origin/main)" "$(git -C $C/crv rev-parse --short origin/main)" > $T/VERSION
  cat $T/VERSION; tar -czf code.tgz -C $T core crv VERSION; rm -rf $T
  for f in list.txt dumper.tgz code.tgz; do ifdh rm $IN/$f >/dev/null 2>&1; ifdh cp $f $IN/$f; done
  ifdh ls $IN; exit 0
fi
LIST=${LIST:-list.txt}
if [ -f "$LIST" ] && [ "$LIST" != list.txt ]; then ifdh rm $IN/$LIST >/dev/null 2>&1; ifdh cp $LIST $IN/$LIST; fi
T=$(mktemp -d); ifdh cp $IN/$LIST $T/list.txt >/dev/null 2>&1; NL=$(cat $T/list.txt 2>/dev/null | wc -l); rm -rf $T
[ "$NL" -gt 0 ] || { echo "ERROR: empty or missing $IN/$LIST" >&2; exit 1; }
N=$(( (NL + B - 1) / B )); [ "$MODE" = --test ] && N=1
OUT_TOKEN=$(echo "$OUT" | sed 's|^/pnfs/mu2e|/mu2e|')
CMD=(jobsub_submit -N "$N" -G mu2e --role=Analysis --expected-lifetime=${LIFETIME:-3h}
  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --lines='+FERMIHTC_AutoRelease=True'
  --memory=2000MB --disk=4000MB --need-storage-modify="$OUT_TOKEN"
  -e LIST_PNFS="$IN/$LIST" -e DUMPER_PNFS="$IN/dumper.tgz" -e CODE_PNFS="$IN/code.tgz" -e BLOCK="$B" -e BLOCK0="${FIRST_BLOCK:-0}" "file://$HERE/grid_catsteps.sh" "$OUT")
echo "${CMD[@]}"
"${CMD[@]}" 2>&1 | tee /tmp/$USER.submit_catsteps.txt
grep -qE "submitted to cluster" /tmp/$USER.submit_catsteps.txt || { echo "ERROR: no cluster id: NOTHING SUBMITTED" >&2; exit 1; }
