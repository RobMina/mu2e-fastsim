#!/bin/bash
#SBATCH -J r5_foil_adapters
#SBATCH -A ldmxuva
#SBATCH -p gpu-mig
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/foil_v11.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
S=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/code_foil_v11
C=$W/code_foil_v11
O=$W/out/foil_tests_v11
mkdir "$C"
cp -a "$S/." "$C/"
(cd "$C" && find beam -type f -print0 | sort -z | xargs -0 sha256sum) > "$W/code_foil_v11.sha256"
mkdir "$O"
cd "$O"
nvidia-smi -L > gpu.txt
E="PYTHONPATH=$W/code_boundary_v7/core:$C/beam,XLA_PYTHON_CLIENT_PREALLOCATE=false,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,JAX_ENABLE_X64=true,FLASH_FOIL_BASELINE_DIR=$W/code_geometry_v9/beam/beamfastsim"
apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -c 'import jax; assert jax.default_backend() == "gpu", jax.devices(); print(jax.devices())'
/usr/bin/time -v -o test.time apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -m pytest -q "$C/beam/tests" > tests.log 2>&1
cat tests.log
printf 'completed=%s\n' "$(date -Is)" > complete.txt
