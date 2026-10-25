#!/bin/bash
# Audit existing extracted candidate tables without regenerating transport; refuse an existing audit tag.
set -eo pipefail
D=${CRVFS_DATA:?set CRVFS_DATA}
OUT=${1:?candidate campaign directory}
TAG=${2:?new audit tag}
FIRST=${3:-1}
LAST=${4:-20}
mkdir "$OUT/$TAG"
for k in $(seq "$FIRST" "$LAST"); do
  S=$(printf '001400_%08d' "$k")
  python3 -m crvfastsim.analyses.transport_validation --gen "$D/g4ref_v1_tables/gen.$S.root" \
    --reference "$D/g4ref_v1_tables/crvsteps.$S.root" --baseline "$D/fixvar/v1_absorbsk/tables/fastC.$S.root" \
    --candidate "$OUT/tables/fastC.$S.root" --config extracted_v04 --expected-events 50000 \
    --gdml "$D/geometry/mu2e_extracted_v04.gdml" --output "$OUT/$TAG/shard_$k.json" > "$OUT/$TAG/shard_$k.log" 2>&1
  echo "audited shard $k"
done
python3 -m crvfastsim.analyses.transport_validation --merge "$OUT/$TAG/shard_*.json" --output "$OUT/${TAG}_merged.json"
