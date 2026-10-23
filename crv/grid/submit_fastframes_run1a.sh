#!/bin/bash
# Submit F-fast deadtime frames for Run 1A CE+Mix1BB subruns as grid jobs, one subrun per job (grid_fastframes_run1a.sh).
# Run on a Mu2e GPVM from a data-area dir (not an app dir).
#   submit_fastframes_run1a.sh --prepare N        # list the first N subruns (dig, mcs, nts paths), stage list + code +
#                                                 # dumper + CRVPhoton table to pnfs; prints the jobsub command
#   submit_fastframes_run1a.sh --test             # one job (line 1 of the list)
#   submit_fastframes_run1a.sh --cluster [--n M]  # M jobs (default: every line of the list)
# Sizing (GPVM, one subrun, ~8.3 k events): art dumps ~3 min each (three passes), fastframes ~95 s CPU and 4.4 GB RSS
# with 500-event chunks; ~10-15 min per job. Output: $BASE/out (pnfs scratch).
BASE=${BASE:-/pnfs/mu2e/scratch/users/ram2aq/crvfs_fastframes_run1a}; IN=$BASE/in; OUT=$BASE/out
D=/exp/mu2e/data/users/ram2aq/crv_fastsim; C=/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim
HERE=$(cd "$(dirname "$0")" && pwd); MODE=""; N=""; NPREP=""
while [ $# -gt 0 ]; do case "$1" in
  --prepare) MODE=prepare; NPREP=$2; shift 2 ;; --test) MODE=test; shift ;; --cluster) MODE=cluster; shift ;;
  --n) N=$2; shift 2 ;; *) echo "unknown $1"; exit 2 ;; esac; done
[ -n "$MODE" ] || { sed -n 2,9p "$0"; exit 2; }
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
muse setup ops >/dev/null 2>&1 || true
command -v ifdh >/dev/null 2>&1 || setup ifdhc
set -u
if [ "$MODE" = prepare ]; then
  ifdh mkdir_p $IN; ifdh mkdir_p $OUT
  loc() { echo "$(samweb locate-file "$1" | sed 's/^[a-z]*://; s/(.*//')/$1"; }
  : > list.txt
  for n in $(samweb list-files "defname: nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.root" | sort | head -"$NPREP"); do
    s=${n##*001430_}; s=${s%.root}
    echo "$s $(loc dig.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.001430_$s.art) $(loc mcs.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1.001430_$s.art) $(loc "$n")" >> list.txt
  done
  wc -l < list.txt
  ifdh rm $IN/list.txt >/dev/null 2>&1; ifdh cp list.txt $IN/list.txt
  T=$(mktemp -d)
  # code at origin/main of core and crv (the shared checkouts carry other sessions' uncommitted work)
  git -C $C/core fetch -q origin && git -C $C/crv fetch -q origin
  mkdir -p $T/core $T/crv && git -C $C/core archive origin/main | tar -x -C $T/core && git -C $C/crv archive origin/main | tar -x -C $T/crv
  printf "core %s\ncrv %s\n" "$(git -C $C/core rev-parse --short origin/main)" "$(git -C $C/crv rev-parse --short origin/main)" > $T/VERSION
  cat $T/VERSION
  tar -czf code.tgz -C $T core crv VERSION; rm -rf $T
  T=$(mktemp -d); mkdir -p "$T/lib" "$T/fcl/CrvFastSim/fcl"
  cp $C/crv/art/build/*/CrvFastSim/lib/*.so "$T/lib/"
  git -C $C/crv archive origin/main art/CrvFastSim/fcl | tar -x -C $T && mv $T/art/CrvFastSim/fcl/*.fcl "$T/fcl/CrvFastSim/fcl/" && rm -rf $T/art
  tar -czf dumper.tgz -C "$T" lib fcl; rm -rf "$T"
  for f in code.tgz dumper.tgz; do ifdh rm $IN/$f >/dev/null 2>&1; ifdh cp $f $IN/$f; done
  ifdh rm $IN/crvphoton.txt >/dev/null 2>&1; ifdh cp $D/deadtime_run1a/cond/CRVPhoton_cid196.txt $IN/crvphoton.txt
  ifdh ls $IN
  exit 0
fi
NL=$(ifdh cp $IN/list.txt /dev/stdout 2>/dev/null | wc -l)
[ "$MODE" = test ] && N=1
N=${N:-$NL}
OUT_TOKEN=$(echo "$OUT" | sed 's|^/pnfs/mu2e|/mu2e|')
CMD=(jobsub_submit -N "$N" -G mu2e --role=Analysis --expected-lifetime=3h
  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --lines='+FERMIHTC_AutoRelease=True'
  --memory=6000MB --disk=6000MB --need-storage-modify="$OUT_TOKEN"
  -e LIST_PNFS="$IN/list.txt" -e CODE_PNFS="$IN/code.tgz" -e DUMPER_PNFS="$IN/dumper.tgz"
  -e CONDPHOTON_PNFS="$IN/crvphoton.txt" "file://$HERE/grid_fastframes_run1a.sh" "$OUT")
echo "${CMD[@]}"
"${CMD[@]}" 2>&1 | tee /tmp/$USER.submit_fastframes.txt
grep -qE "submitted to cluster" /tmp/$USER.submit_fastframes.txt || { echo "ERROR: no cluster id: NOTHING SUBMITTED" >&2; exit 1; }
