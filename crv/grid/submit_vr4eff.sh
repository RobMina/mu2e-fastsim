#!/bin/bash
# Submit the V-R4 EX layer-efficiency check (reco-vr4eff, JAX chain) as grid jobs, one per crvonly_v05 file. Run on a Mu2e GPVM from a data-area dir, after
# `--stage` once (copies the CPU SIF, step tables, stock refs and CRVCalib CSVs to pnfs scratch).
#   submit_vr4.sh --stage                  # stage inputs to $BASE/in (once)
#   submit_vr4.sh --test                   # one job (shard 1)
#   submit_vr4.sh [--first N] [--n M]      # M jobs from shard N (default 1..20)
#   submit_vr4.sh --dry-run ...            # print the jobsub command only
#   [--memory 12000MB] [--batch-events 100]  # for shards that went over the memory limit
# Sizing (measured on mu2egpvm06): ~4.5 min and ~8 GB RSS peak per shard (2,500 events, batches of 250).
BASE=/pnfs/mu2e/scratch/users/ram2aq/crvfs_vr4; IN=$BASE/in_eff; OUT=$BASE/out_eff_v05
D=/exp/mu2e/data/users/ram2aq/crv_fastsim; SIF=/exp/mu2e/data/users/ram2aq/container/mu2efastsim_cpu.sif
HERE=$(cd "$(dirname "$0")" && pwd); FIRST=1; N=20; DRY=false; STAGE=false
while [ $# -gt 0 ]; do case "$1" in
  --stage) STAGE=true; shift ;; --test) N=1; shift ;; --first) FIRST=$2; shift 2 ;; --n) N=$2; shift 2 ;;
  --out) OUT=$2; shift 2 ;; --dry-run) DRY=true; shift ;; --memory) MEM=$2; shift 2 ;;
  --batch-events) BEV=$2; shift 2 ;; *) echo "unknown $1"; exit 2 ;; esac; done
MEM=${MEM:-12000MB}; BEV=${BEV:-100}
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
muse setup ops >/dev/null 2>&1 || true
command -v ifdh >/dev/null 2>&1 || setup ifdhc
set -u
if $STAGE; then
  ifdh mkdir_p $IN
  ifdh cp $SIF $IN/mu2efastsim_cpu.sif
  ifdh cp $D/geometry/crvcalib_sim_best_v1_5_run1400.csv $IN/crvcalib.csv
  ifdh cp $D/geometry/crvphoton_sim_best_v1_5_run1400.csv $IN/crvphoton.csv
  for k in $(seq 0 19); do K=$(printf "001400_%08d" $((990 + 33 * k)))
    ifdh cp $D/prod_as_tables/crvsteps.$K.root $IN/crvsteps.$K.root; ifdh cp $D/vr4eff_v05/prep/prep.$K.npz $IN/prep.$K.npz; done
  ifdh ls $IN | wc -l; exit 0
fi
# code tarball: git archive of core and crv at the checked-out (committed) commits; staged next to the inputs
C=/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim; T=$(mktemp -d)
mkdir -p $T/core $T/crv && git -C $C/core archive HEAD | tar -x -C $T/core && git -C $C/crv archive HEAD | tar -x -C $T/crv
printf "core %s\ncrv %s\n" "$(git -C $C/core rev-parse --short HEAD)" "$(git -C $C/crv rev-parse --short HEAD)" > $T/VERSION
CODE_TGZ=code_$(git -C $C/crv rev-parse --short HEAD).tgz
tar -czf $T/$CODE_TGZ -C $T core crv VERSION && ($DRY || { ifdh rm $IN/$CODE_TGZ >/dev/null 2>&1; ifdh cp $T/$CODE_TGZ $IN/$CODE_TGZ; })
rm -rf $T
OUT_TOKEN=$(echo "$OUT" | sed 's|^/pnfs/mu2e|/mu2e|')
CMD=(jobsub_submit -N "$N" -G mu2e --role=Analysis --expected-lifetime=4h
  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --lines='+FERMIHTC_AutoRelease=True'
  --memory="$MEM" --disk=8000MB --need-storage-modify="$OUT_TOKEN"
  -e SIF_PNFS="$IN/mu2efastsim_cpu.sif" -e CODE_PNFS="$IN/$CODE_TGZ" -e SHARD_OFFSET=$((FIRST - 1))
  "file://$HERE/grid_vr4eff.sh" "$OUT" "$IN" ${BEV:+--batch-events $BEV})
echo "${CMD[@]}"
$DRY && exit 0
ifdh mkdir_p "$OUT"
"${CMD[@]}" 2>&1 | tee /tmp/$USER.submit_vr4.txt
grep -qE "submitted to cluster" /tmp/$USER.submit_vr4.txt || { echo "ERROR: no cluster id: NOTHING SUBMITTED" >&2; exit 1; }
