#!/bin/bash
#SBATCH --partition=standard
#SBATCH --job-name=offline_two_art
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -eo pipefail
ROOT=${OFFLINE_ROOT:?set OFFLINE_ROOT}
INPUT=${OFFLINE_INPUT:?set OFFLINE_INPUT}
module load apptainer
OUT="$ROOT/parallel_${SLURM_JOB_ID}"
mkdir -p "$OUT"
pids=()
for slot in 0 1; do
  mkdir "$OUT/slot$slot"
  (
    cd "$OUT/slot$slot"
    printf '#include "CrvFastSim/fcl/valhist_stock.fcl"\nservices.SeedService.baseSeed : 20261023\nphysics.analyzers.crvdump : { module_type : CrvRecoTableDumper mc : false }\nphysics.ValPath : [ Validation, crvDigiDQM, crvRecoDQM, dqmCrv, crvdump ]\n' > a0.fcl
    export OFFLINE_OUTPUT_FILE="$PWD/a0.root" OFFLINE_LOG_PREFIX=a0 OFFLINE_BINDS="$INPUT:$INPUT"
    bash "$ROOT/code/container/offline/offline_uva.sh" \
      "${OFFLINE_AREA:?set OFFLINE_AREA to the trusted reference tarball}" "$PWD/a0.fcl" "$INPUT/dts.art" > a0.log 2>&1
  ) &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
exit "$rc"
