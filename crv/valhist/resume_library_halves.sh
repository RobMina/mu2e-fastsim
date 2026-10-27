#!/bin/bash
# Preserve the drift-rejected replay and reuse its assembled library through an untouched physics checkout.
set -e
D=${CRVFS_DATA:?data root}
A=/exp/mu2e/app/users/ram2aq/analysis
export M2FS_CORE="$A/core-thread-b-dqm-20261025" CRVVAL_AREA=/exp/mu2e/app/users/ram2aq/crvval
export PYTHONPATH="$A/crv-thread-b-uncertainty-20261025:$M2FS_CORE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PLAN="$D/valhist/thread_b_library_plan_20261025/plan.json"
COUNTERS=/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_39_00/Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt
for variant in half_even half_odd; do
  LIBOUT="$D/valhist/thread_b_library_20261025_${variant}_library"
  OUT="$D/valhist/thread_b_library_replay_20261025_${variant}_s11"
  test ! -e "$OUT" || exit 2
  if [ "$variant" = half_odd ]; then
    python3 -m valhist.library_variants build "$PLAN" --variant "$variant" --counters "$COUNTERS" --outdir "$LIBOUT"
  fi
  echo "BEGIN replay=$variant utc=$(date -u +%FT%TZ)"
  bash "$A/crv-thread-b-dqm-20261025/valhist/run_pilot.sh" "$OUT" \
    "$D/thread_a_ledger_20261025_c3895aae/extracted20/tables/fastC.001400_00000011.root" \
    "$LIBOUT/library.root" \
    /pnfs/mu2e/scratch/users/ram2aq/crv_fastsim/g4ref_v1/gen.ram2aq.CosmicCRYExtractedRef.v0.001400_00000011.art 11
  bash "$A/crv-thread-b-dqm-20261025/valhist/audit_pilot.sh" "$OUT" "$D/valhist/g4ref_v1/a0g/valhist.001400_00000011.root"
  echo "END replay=$variant utc=$(date -u +%FT%TZ)"
done
