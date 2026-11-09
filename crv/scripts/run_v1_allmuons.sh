#!/bin/bash
# V1 (fast stage-C crossings vs G4) on the 20 g4ref_v1 shards for the transport default (all muons) and
# --first-muon-only, from this checkout. One GPVM, serially. Needs site.env. Outputs $CRVFS_DATA/fixvar/v1_{allmu,first}<SFX>/
# usage: [SEED_OFFSET=n] run_v1_allmuons.sh   (seed = shard + SEED_OFFSET; a nonzero offset adds _s<offset> to the dirs)
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana >/dev/null 2>&1 || true
set -u
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
D=${CRVFS_DATA:?source site.env}; T=$D/g4ref_v1_tables
export PYTHONPATH=$CODE:$M2FS_CODE/core:$M2FS_CODE/tracker
OFF=${SEED_OFFSET:-0}; SFX=""; [ "$OFF" != 0 ] && SFX=_s$OFF
for arm in allmu first; do
  O=$D/fixvar/v1_$arm$SFX; mkdir -p "$O/tables"
  opt=""; [ "$arm" = first ] && opt="--first-muon-only"
  for k in $(seq 1 20); do
    S=$(printf "001400_%08d" "$k")
    [ -s "$O/tables/fastC.$S.root" ] || python3 -m crvfastsim transport "$T/gen.$S.root" --outdir "$O/tables" --stage C \
      --seed "$((k + OFF))" $opt > /dev/null 2>&1 || echo "$arm $S transport failed"
  done
  python3 -m crvfastsim v1compare "$O/tables/fastC.*.root" --compare "$T/crvsteps.*.root" --outdir "$O" > "$O/v1.txt" 2>&1
  echo "$arm v1 rc=$? $(date +%T)"
done
