#!/bin/bash
# Tier-1b steps on the retained stage-C transport tables, one shard per call with seed = shard (the Thread B A1 recipe),
# then the per-layer step breakdown and the step y/z statistic against the G4 steps of the same generated events
# (valhist.steplayers). Extra arguments go to tier1b-apply (e.g. --crossing-context --contain-steps --point-placement).
# usage: run_step_layers.sh LIBRARY OUTDIR SHARDS(e.g. 1-20) [tier1b-apply options]
#        [env: FASTC_DIR, G4_DIR, PYTHONPATH with core and crv]
set -eo pipefail
LIB=$1; OUT=$2; SHARDS=${3:-1-20}; shift 3 || true
D=${CRVFS_DATA:?set CRVFS_DATA to the CRV data root (see site.env.example)}
FASTC_DIR=${FASTC_DIR:-$D/thread_a_ledger_20261025_c3895aae/extracted20/tables}
G4_DIR=${G4_DIR:-$D/valhist/g4ref_v1/g4all_tables}
mkdir -p "$OUT"
lo=${SHARDS%-*}; hi=${SHARDS#*-}
for k in $(seq "$lo" "$hi"); do
  s=$(printf "001400_%08d" "$k")
  python3 -m crvfastsim tier1b-apply "$FASTC_DIR/fastC.$s.root" --library "$LIB" --outdir "$OUT" --seed "$k" "$@"
done
python3 -m valhist.steplayers breakdown --ref "$G4_DIR/g4all.{s}.root" --cand "$OUT/fast1b.{s}.root" \
  --shards "$SHARDS" --out "$OUT/breakdown.json"
python3 -m valhist.steplayers posyz --arm "a0g=$G4_DIR/g4all.{s}.root" --arm "cand=$OUT/fast1b.{s}.root" \
  --shards "$SHARDS" --out "$OUT/posyz.json"
