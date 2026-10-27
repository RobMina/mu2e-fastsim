#!/bin/bash
# Run explicit remaining shards sequentially through a frozen pilot checkout; existing outputs are refused.
set -e
TAG=${1:?unique output tag}; shift
CODE=${THREAD_B_CODE:?frozen pilot checkout}
D=${CRVFS_DATA:?data root}
TRANSPORT=${THREAD_B_TRANSPORT:?retained corrected transport directory}
LIB=${THREAD_B_LIBRARY:?frozen tier1b library}
GENROOT=${THREAD_B_GENROOT:?known generator input directory}
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[ "$#" -gt 0 ] || exit 2
for shard in "$@"; do
  [[ "$shard" =~ ^[0-9]+$ ]] && (( shard >= 1 && shard <= 20 && shard != 11 )) || exit 2
  printf -v id '%08d' "$shard"
  OUT="$D/valhist/${TAG}_s${id}"
  test ! -e "$OUT" || { echo "refusing existing output $OUT" >&2; exit 2; }
  test -f "$TRANSPORT/fastC.001400_${id}.root" || exit 2
done
for shard in "$@"; do
  printf -v id '%08d' "$shard"
  OUT="$D/valhist/${TAG}_s${id}"
  echo "BEGIN shard=$shard output=$OUT host=$(hostname) utc=$(date -u +%FT%TZ)"
  bash "$CODE/valhist/run_pilot.sh" "$OUT" "$TRANSPORT/fastC.001400_${id}.root" \
    "$LIB" "$GENROOT/gen.ram2aq.CosmicCRYExtractedRef.v0.001400_${id}.art" "$shard"
  bash "$CODE/valhist/audit_pilot.sh" "$OUT" "$D/valhist/g4ref_v1/a0g/valhist.001400_${id}.root"
  echo "END shard=$shard utc=$(date -u +%FT%TZ)"
done
