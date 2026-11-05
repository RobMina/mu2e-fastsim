#!/bin/bash
#SBATCH -J r5_resume_tests
#SBATCH -A ldmxuva
#SBATCH -p gpu-mig
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/resume_tests.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=$W/code_v6
O=$W/out/resume_tests_v6
mkdir "$O"
cd "$O"
nvidia-smi -L > gpu.txt
E="PYTHONPATH=$C/core:$C/beam,XLA_PYTHON_CLIENT_PREALLOCATE=false,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,JAX_ENABLE_X64=true,FLASH_CE_BASELINE_SOURCE=$C/cechain_default_baseline.py,FLASH_TARGET_BASELINE_SOURCE=$C/flash_target_baseline.py"
apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -c 'import jax; assert jax.default_backend() == "gpu", jax.devices(); print(jax.devices())'
/usr/bin/time -v -o test.time apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -m pytest -q "$C/beam/tests/test_flash_resume_jax.py" "$C/beam/tests/test_flash_checkpoint.py" "$C/beam/tests/test_flash_target_run.py" "$C/beam/tests/test_flash_ce_defaults.py" "$C/beam/tests/test_flash_foil_boundary.py" > tests.log 2>&1
cat tests.log
printf 'completed=%s\n' "$(date -Is)" > complete.txt
