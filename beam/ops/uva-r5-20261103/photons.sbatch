#!/bin/bash
#SBATCH -J r5_photon_records
#SBATCH -A ldmxuva
#SBATCH -p gpu-mig
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/photons.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
C=$R/flash_r5_20261103/code_v2
O=$R/flash_r5_20261103/out/photon_records_v2
mkdir "$O"
cd "$O"
sha256sum "$C/core/mu2efastsim/eslab_jax.py" "$C/core/tests/test_eslab_photons_jax.py" > source.sha256
nvidia-smi -L > gpu.txt
E="PYTHONPATH=$C/core,XLA_PYTHON_CLIENT_PREALLOCATE=false,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1"
apptainer exec --nv --cleanenv --env "$E" -B "$R" -B "$C" "$R/mu2efastsim.sif" python3 -c 'import jax; assert jax.default_backend() == "gpu", jax.devices(); print(jax.devices())'
/usr/bin/time -v -o test.time apptainer exec --nv --cleanenv --env "$E" -B "$R" -B "$C" "$R/mu2efastsim.sif" python3 -m pytest -q "$C/core/tests/test_eslab_photons_jax.py" --junitxml="$O/tests.xml" > tests.log 2>&1
printf 'completed=%s\n' "$(date -Is)" > complete.txt
cat tests.log
