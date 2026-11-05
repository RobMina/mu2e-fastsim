#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -J c4-primary
set -eo pipefail
module load apptainer
: "${C4_WORK:?new output area}"
: "${C4_GDML:?Run 1A GDML}"
: "${C4_CALIB:?calibration CSV}"
: "${C4_DEVIATION:?photon deviation CSV}"
: "${C4_LOOKUP:?photon lookup directory}"
: "${C4_SIF:?project image}"
: "${C4_PARQUET:?shared parquet site-packages}"
: "${C4_OFFLINE:?offline search path}"
O="$C4_WORK/${C4_LABEL:-primary}_${SLURM_JOB_ID:?}"
mkdir -p "$O"
C4_INPUTS=${C4_INPUTS:-$C4_WORK/inputs/primaries0}
C4_CRV=${C4_CRV:-$C4_WORK/code/crv}
extra=()
if [[ -n ${C4_LIBRARY:-} ]]; then extra+=(--library-dir "$C4_LIBRARY"); fi
extra+=(--split "${C4_SPLIT:-all}")
cd "$C4_CRV"
apptainer exec --cleanenv -B "$(dirname "$C4_WORK")" \
  --env "PYTHONPATH=$C4_CRV:$C4_WORK/code/core:$C4_PARQUET,PYTHONDONTWRITEBYTECODE=1,JAX_PLATFORMS=cpu,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,CRVFS_NO_FETCH=1,CRVFS_LOOKUP_DIR=$C4_LOOKUP,MU2E_SEARCH_PATH=$C4_OFFLINE" \
  "$C4_SIF" python3 -m crvfastsim.veto.primary_chain \
  --selection "$C4_INPUTS/selection.parquet" --primaries "$C4_INPUTS/primaries.parquet" \
  --gdml "$C4_GDML" --calib "$C4_CALIB" --deviation "$C4_DEVIATION" --outdir "$O" "${extra[@]}"
