#!/bin/bash
# Assemble two donor-file half libraries as data, then replay the fixed shard-11 transport through both arms.
set -e
D=${CRVFS_DATA:?data root}
CODE=${THREAD_B_UNCERTAINTY_CODE:?frozen uncertainty checkout}
PHYSICS=${THREAD_B_CODE:?separate frozen pilot checkout}
export M2FS_CORE=${M2FS_CORE:?frozen core checkout} CRVVAL_AREA=${CRVVAL_AREA:?frozen stock area}
export PYTHONPATH="$CODE:$M2FS_CORE" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PLAN=${THREAD_B_LIBRARY_PLAN:?sealed plan json}
COUNTERS=${THREAD_B_COUNTERS:?versioned counters text}
TAG=${1:?unique output tag}
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
for variant in half_even half_odd; do
  test ! -e "$D/valhist/${TAG}_${variant}_library" || exit 2
  test ! -e "$D/valhist/${TAG}_${variant}_s11" || exit 2
done
for variant in half_even half_odd; do
  LIBOUT="$D/valhist/${TAG}_${variant}_library"
  OUT="$D/valhist/${TAG}_${variant}_s11"
  echo "BEGIN variant=$variant utc=$(date -u +%FT%TZ)"
  python3 -m valhist.library_variants build "$PLAN" --variant "$variant" --counters "$COUNTERS" --outdir "$LIBOUT"
  bash "$PHYSICS/valhist/run_pilot.sh" "$OUT" \
    "$D/thread_a_ledger_20261025_c3895aae/extracted20/tables/fastC.001400_00000011.root" \
    "$LIBOUT/library.root" \
    /pnfs/mu2e/scratch/users/ram2aq/crv_fastsim/g4ref_v1/gen.ram2aq.CosmicCRYExtractedRef.v0.001400_00000011.art 11
  bash "$PHYSICS/valhist/audit_pilot.sh" "$OUT" "$D/valhist/g4ref_v1/a0g/valhist.001400_00000011.root"
  echo "END variant=$variant utc=$(date -u +%FT%TZ)"
done
