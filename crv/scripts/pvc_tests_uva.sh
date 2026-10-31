#!/bin/bash
#SBATCH -J pvcreco_tests
# Slurm account: set SBATCH_ACCOUNT (see site.env.example) or pass sbatch -A.
#SBATCH -p standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 00:10:00
set -eo pipefail
module load apptainer
S=${PVC_RUN_DIR:?set PVC_RUN_DIR to a writable run directory}
F=${FASTSIMKIT_SCRATCH:?set FASTSIMKIT_SCRATCH to the cluster working area}
O=$S/tests_$SLURM_JOB_ID
mkdir "$O"
exec > "$O/run.log" 2>&1
cd "$S/code/crv"
apptainer exec --cleanenv \
    --env "PYTHONPATH=$S/code/crv:$S/code/core,JAX_PLATFORMS=cpu,JAX_ENABLE_X64=1,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,MU2E_SEARCH_PATH=$F/offline_data" \
    -B "$S" -B "$F/offline_data" "$F/mu2efastsim.sif" \
    python3 -m pytest -q -p no:cacheprovider tests/test_pvc_profile.py
apptainer exec --cleanenv \
    --env "PYTHONPATH=$S/code/crv:$S/code/core,JAX_PLATFORMS=cpu,JAX_ENABLE_X64=1,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,MU2E_SEARCH_PATH=$F/offline_data" \
    -B "$S" -B "$F/offline_data" "$F/mu2efastsim.sif" \
    python3 -m pytest -q -p no:cacheprovider tests/test_reco_jax.py tests/test_reco_dispatch.py
