#!/bin/bash
#SBATCH -J pvcreco_profile
#SBATCH -A ldmxuva
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 00:15:00
set -eo pipefail
module load apptainer
S=${PVC_RUN_DIR:?set PVC_RUN_DIR to a writable run directory}
F=/scratch/$USER/mu2efastsim
O=$S/profile_$SLURM_JOB_ID
mkdir "$O"
exec > "$O/run.log" 2>&1
for backend in cpu cuda; do
    extra=(); test "$backend" = cuda && extra=(--against "$O/cpu")
    apptainer exec --nv --cleanenv \
        --env "PYTHONPATH=$S/code/crv:$S/code/core,JAX_PLATFORMS=$backend,JAX_ENABLE_X64=1,PYTHONDONTWRITEBYTECODE=1,CRVFASTSIM_COMPACT_DISPATCH=1,OPENBLAS_NUM_THREADS=1,MU2E_SEARCH_PATH=$F/offline_data,XLA_PYTHON_CLIENT_PREALLOCATE=false" \
        -B "$S" -B "$F/offline_data" "$F/mu2efastsim.sif" \
        python3 -m crvfastsim.reco.pvc_profile --out "$O/$backend" --rows 4096 --events 1000 --groups 128 \
        "${extra[@]}" > "$O/$backend.log" 2>&1
done
