#!/bin/bash
# V-R4 batch: fast chain vs stock chain on the g4ref_v1 shards FIRST..LAST (CrvStep tables + stock refs of
# scripts/run_v2_arms.sh), one --emit JSON per shard, then the merged report. One GPVM, serially. Needs site.env.
# usage: run_vr4.sh FIRST LAST [OUTTAG] [extra reco-vr4 args, e.g. --jax]
source "${M2FS_CODE:?set M2FS_CODE to the mu2e-fastsim checkout}/site.env"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana >/dev/null 2>&1 || true
set -u
A=${1:?first}; B=${2:?last}; TAG=${3:-vr4_v1}; shift 3 2>/dev/null || shift $#
D=$CRVFS_DATA; O=$D/$TAG; mkdir -p "$O/batches"
cd "$CRVFS_CODE"
for k in $(seq "$A" "$B"); do
  S=$(printf "001400_%08d" "$k")
  [ -s "$O/batches/vr4.$S.json" ] && continue
  python3 -m crvfastsim reco-vr4 "$D/g4ref_v1_tables/crvsteps.$S.root" --refs "$D/v2_fastC_v1/ref.$S.root" \
    --calib "$D/geometry/crvcalib_sim_best_v1_5_run1400.csv" --deviation "$D/geometry/crvphoton_sim_best_v1_5_run1400.csv" \
    --seed "$k" --emit "$O/batches/vr4.$S.json" --outdir "$O" "$@" > "$O/batches/vr4.$S.txt" 2>&1
  echo "$S rc=$? $(date +%T)"
done
python3 -m crvfastsim reco-vr4 "$O/batches/vr4.*.json" --merge --outdir "$O" > "$O/vr4_merged.txt" 2>&1
tail -8 "$O/vr4_merged.txt"; echo RUN_VR4_DONE
