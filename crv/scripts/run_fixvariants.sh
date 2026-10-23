#!/bin/bash
# Fast-steps variants on g4ref_v1 shards FIRST..LAST: stage-C transport with options, then tier-1b apply with the
# current library (the tier1b_v8 recipe), for the DQM-validation deficit study. One GPVM, serially. Needs site.env.
# usage: [SEED_OFFSET=n] run_fixvariants.sh FIRST LAST TAG [transport options, e.g. --msc-tail --tail-k 2 --first-muon-only (to reproduce the old tables)]
#   outputs $CRVFS_DATA/fixvar/TAG/{fastC,fast1b}.<subrun>.root; seed = shard number + SEED_OFFSET (default 0)
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana >/dev/null 2>&1 || true
set -u
A=${1:?first}; B=${2:?last}; TAG=${3:?tag}; shift 3
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
D=${CRVFS_DATA:?source site.env}; O=$D/fixvar/$TAG; mkdir -p "$O/logs"
LIB=$D/tier1b_v7/lib_prod_as_20files.root
echo "code $CODE ($(git -C "$CODE" rev-parse --short HEAD 2>/dev/null)); options: $*; library $LIB; $(date -Is)" >> "$O/logs/provenance.txt"
export PYTHONPATH=$CODE:$M2FS_CODE/core
OFF=${SEED_OFFSET:-0}
for k in $(seq "$A" "$B"); do
  S=$(printf "001400_%08d" "$k"); SEED=$((k + OFF))
  [ -s "$O/fast1b.$S.root" ] && continue
  python3 -m crvfastsim transport "$D/g4ref_v1_tables/gen.$S.root" --outdir "$O" --stage C --seed "$SEED" "$@" \
    > "$O/logs/transport.$S.log" 2>&1 || { echo "$S transport rc=$?"; continue; }
  python3 -m crvfastsim tier1b-apply "$O/fastC.$S.root" --library "$LIB" --outdir "$O" --seed "$SEED" \
    > "$O/logs/tier1b.$S.log" 2>&1
  echo "$S rc=$? $(date +%T)"
done
