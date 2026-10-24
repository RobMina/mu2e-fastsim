#!/bin/bash
# Submit stage 2 of F-fast from beam frames: N jobs x EVENTS mixed beam-only events each (grid_mixframes.sh), reading the
# stage-1 tables in $CAT/out. Run on a GPVM from a data-area dir.
#   submit_mixframes.sh --prepare           # stage code + conditions (Run 1A mcs dump) + CRVPhoton table to pnfs
#   submit_mixframes.sh --test              # one job          submit_mixframes.sh --cluster -N 20
# env: EVENTS (per job), SEED0, BASE, CAT (stage-1 base). Sizing from the one-job test.
BASE=${BASE:-/pnfs/mu2e/scratch/users/ram2aq/crvfs_mixframes_run1a}; IN=$BASE/in; OUT=$BASE/out
CAT=${CAT:-/pnfs/mu2e/scratch/users/ram2aq/crvfs_catsteps_run1a}
D=/exp/mu2e/data/users/ram2aq/crv_fastsim; C=/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim
HERE=$(cd "$(dirname "$0")" && pwd); MODE="${1:-}"; N=1; [ "${2:-}" = -N ] && N=$3
[ -n "$MODE" ] || { sed -n 2,7p "$0"; exit 2; }
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
muse setup ops >/dev/null 2>&1 || true
command -v ifdh >/dev/null 2>&1 || setup ifdhc
set -u
if [ "$MODE" = --prepare ]; then
  T=$(mktemp -d); git -C $C/core fetch -q origin; git -C $C/crv fetch -q origin
  mkdir -p $T/core $T/crv && git -C $C/core archive origin/main | tar -x -C $T/core && git -C $C/crv archive origin/main | tar -x -C $T/crv
  printf "core %s\ncrv %s\n" "$(git -C $C/core rev-parse --short origin/main)" "$(git -C $C/crv rev-parse --short origin/main)" > $T/VERSION
  cat $T/VERSION; tar -czf code.tgz -C $T core crv VERSION; rm -rf $T
  ifdh rm $IN/code.tgz >/dev/null 2>&1; ifdh cp code.tgz $IN/code.tgz
  ifdh rm $IN/crvreco_mcs.root >/dev/null 2>&1; ifdh cp /pnfs/mu2e/scratch/users/ram2aq/crvfs_fastframes_run1a/out/crvreco_mcs_00000000.root $IN/crvreco_mcs.root
  ifdh rm $IN/crvphoton.txt >/dev/null 2>&1; ifdh cp $D/deadtime_run1a/cond/CRVPhoton_cid196.txt $IN/crvphoton.txt
  ifdh ls $IN; exit 0
fi
[ "$MODE" = --test ] && N=1
OUT_TOKEN=$(echo "$OUT" | sed 's|^/pnfs/mu2e|/mu2e|')
CMD=(jobsub_submit -N "$N" -G mu2e --role=Analysis --expected-lifetime=3h
  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --lines='+FERMIHTC_AutoRelease=True'
  --memory="${MEM:-6000MB}" --disk="${DISK:-8000MB}" --need-storage-modify="$OUT_TOKEN"
  -e CODE_PNFS="$IN/code.tgz" -e COND_PNFS="$IN/crvreco_mcs.root" -e CONDPHOTON_PNFS="$IN/crvphoton.txt"
  -e TABLES_PNFS="$CAT/out" -e EVENTS="${EVENTS:-2000}" -e SEED0="${SEED0:-1000}" "file://$HERE/grid_mixframes.sh" "$OUT")
echo "${CMD[@]}"
"${CMD[@]}" 2>&1 | tee /tmp/$USER.submit_mixframes.txt
grep -qE "submitted to cluster" /tmp/$USER.submit_mixframes.txt || { echo "ERROR: no cluster id: NOTHING SUBMITTED" >&2; exit 1; }
