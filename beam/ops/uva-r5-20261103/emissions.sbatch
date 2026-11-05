#!/bin/bash
#SBATCH -J r5_emission_tests
#SBATCH -A ldmxuva
#SBATCH -p gpu-mig
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/emissions.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=$W/code_v3
O=$W/out/emissions_v3
mkdir "$O"
cd "$O"
nvidia-smi -L > gpu.txt
E="PYTHONPATH=$C/core:$C/beam,XLA_PYTHON_CLIENT_PREALLOCATE=false,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,JAX_ENABLE_X64=true"
apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -c 'import jax; assert jax.default_backend() == "gpu", jax.devices(); print(jax.devices())'
/usr/bin/time -v -o test.time apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -m pytest -q "$C/core/tests/test_eslab_emissions_jax.py" "$C/beam/tests/test_flash_target_jax.py" --junitxml="$O/tests.xml" > tests.log 2>&1
printf 'completed=%s\n' "$(date -Is)" > complete.txt
cat tests.log
