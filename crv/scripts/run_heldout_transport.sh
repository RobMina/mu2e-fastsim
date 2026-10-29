#!/bin/bash
# Stage-C fast transport of held-out generator shards (seed = shard, the extracted20 recipe), one shard at a time.
# usage: run_heldout_transport.sh GEN_DIR NEW_OUTDIR FIRST LAST   (PYTHONPATH with core and crv; env CRVFS_DATA)
set -eo pipefail
D=${CRVFS_DATA:-/exp/mu2e/data/users/ram2aq/crv_fastsim}
GEN=${1:?generator table directory (gen.SUBRUN.root)}
OUT=${2:?new output directory}
FIRST=${3:?first shard}
LAST=${4:?last shard}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$OUT/logs"
{ git -C "$CODE" rev-parse HEAD 2>/dev/null || echo "git revision unavailable"; echo "PYTHONPATH=$PYTHONPATH"; sha256sum "$CODE/crvfastsim/transport.py" "$D/geometry/mu2e_extracted_v04.gdml"; } \
  > "$OUT/manifest_${FIRST}_${LAST}.txt"
for k in $(seq "$FIRST" "$LAST"); do
  s=$(printf '001400_%08d' "$k")
  if [ -e "$OUT/fastC.$s.root" ]; then echo "exists $s"; continue; fi
  /usr/bin/time -f 'transport_wall_s=%e maxrss_kb=%M' python3 -m crvfastsim transport "$GEN/gen.$s.root" --stage C \
    --seed "$k" --run-config extracted_v04 --gdml "$D/geometry/mu2e_extracted_v04.gdml" --outdir "$OUT" \
    > "$OUT/logs/transport_$k.log" 2>&1
  echo "completed shard $k $(date -Iseconds)"
done
echo "transport complete $(date -Iseconds)"
