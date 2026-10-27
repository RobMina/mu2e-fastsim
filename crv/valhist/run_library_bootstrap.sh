#!/bin/bash
# Evaluate explicit whole-donor-file bootstrap libraries with two conditional draws on retained shard 11.
set -e
D=${CRVFS_DATA:?data root}
A=/exp/mu2e/app/users/ram2aq/analysis
export M2FS_CORE="$A/core-thread-b-dqm-20261025" CRVVAL_AREA=/exp/mu2e/app/users/ram2aq/crvval
export PYTHONPATH="$A/crv-thread-b-uncertainty-20261025:$M2FS_CORE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PLAN="$D/valhist/thread_b_library_plan_20261025/plan.json"
COUNTERS=/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_39_00/Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt
[ "$#" -gt 0 ] || exit 2
for index in "$@"; do
  [[ "$index" =~ ^[1-8]$ ]] || exit 2
  printf -v variant 'bootstrap_%04d' "$index"
  LIBOUT="$D/valhist/thread_b_library_20261025_${variant}_library"
  test ! -e "$LIBOUT" || exit 2
  for draw in 1 2; do
    seed=$((200000 + 100 * index + draw))
    test ! -e "$D/valhist/thread_b_library_bootstrap_20261025_${variant}_seed${seed}_s11" || exit 2
  done
done
for index in "$@"; do
  printf -v variant 'bootstrap_%04d' "$index"
  LIBOUT="$D/valhist/thread_b_library_20261025_${variant}_library"
  echo "BEGIN variant=$variant utc=$(date -u +%FT%TZ)"
  python3 -m valhist.library_variants build "$PLAN" --variant "$variant" --counters "$COUNTERS" --outdir "$LIBOUT"
  for draw in 1 2; do
    seed=$((200000 + 100 * index + draw))
    OUT="$D/valhist/thread_b_library_bootstrap_20261025_${variant}_seed${seed}_s11"
    bash "$A/crv-thread-b-dqm-20261025/valhist/run_pilot.sh" "$OUT" \
      "$D/thread_a_ledger_20261025_c3895aae/extracted20/tables/fastC.001400_00000011.root" \
      "$LIBOUT/library.root" \
      /pnfs/mu2e/scratch/users/ram2aq/crv_fastsim/g4ref_v1/gen.ram2aq.CosmicCRYExtractedRef.v0.001400_00000011.art "$seed"
    echo "END replay=$variant seed=$seed utc=$(date -u +%FT%TZ)"
  done
done
