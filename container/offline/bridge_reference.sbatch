#!/bin/bash
#SBATCH --partition=standard
#SBATCH --job-name=offline_stock_bridge
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:35:00
set -eo pipefail
ROOT=${OFFLINE_ROOT:?set OFFLINE_ROOT}
INPUT=${OFFLINE_INPUT:?set OFFLINE_INPUT to the staged reference inputs}
module load apptainer
OUT="$ROOT/bridge_${SLURM_JOB_ID}"
mkdir -p "$OUT"
cd "$OUT"
if [[ ${OFFLINE_PREPARE_CLIENT:-0} == 1 ]]; then
  git clone https://github.com/cvmfs/cvmfsexec.git "$OUT/client" > client-clone.log 2>&1
  (cd "$OUT/client" && ./makedist osg) > client-makedist.log 2>&1
  export CVMFSEXEC_DIST="$OUT/client"
fi
export OFFLINE_BINDS="$INPUT:$INPUT"
export OFFLINE_LOG_PREFIX=stock
export OFFLINE_OUTPUT_FILE="$OUT/a0.root"
DUMP='physics.analyzers.crvdump : { module_type : CrvRecoTableDumper mc : false }
physics.ValPath : [ Validation, crvDigiDQM, crvRecoDQM, dqmCrv, crvdump ]'
printf '#include "CrvFastSim/fcl/valhist_stock.fcl"\nservices.SeedService.baseSeed : 20261023\n%s\n' "$DUMP" > a0.fcl
printf '#include "CrvFastSim/fcl/valhist_faststeps.fcl"\nphysics.producers.fastCrvSteps.tableFile : "%s"\n%s\n' \
  "${OFFLINE_G4_TABLE:?set OFFLINE_G4_TABLE to the staged G4 steps table}" "$DUMP" > a0g.fcl
TARBALL=${OFFLINE_AREA:?set OFFLINE_AREA to the reference area or trusted tarball}
for arm in ${OFFLINE_BRIDGE_ARMS:-a0 a0g}; do
  export OFFLINE_LOG_PREFIX=$arm OFFLINE_OUTPUT_FILE="$OUT/$arm.root"
  src="$INPUT/dts.art"; [[ $arm != a0g ]] || src="$INPUT/gen.art"
  bash "$ROOT/code/container/offline/offline_uva.sh" "$TARBALL" "$OUT/$arm.fcl" "$src" > "$arm.log" 2>&1
done
