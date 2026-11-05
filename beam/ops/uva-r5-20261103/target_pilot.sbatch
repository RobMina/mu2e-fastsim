#!/bin/bash
#SBATCH -J r5_target_births
#SBATCH -A ldmxuva
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/target_pilot.%A.%a.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=$W/code_v3
G=$R/bo_sob_y5b_20261103/inputs
case $SLURM_ARRAY_TASK_ID in 0) POINT=bpzup21 ;; 1) POINT=bpzup12 ;; *) exit 2 ;; esac
O=$W/out/target_births_${POINT}_v3
mkdir "$O"
cd "$O"
nvidia-smi -L > gpu.txt
E="PYTHONPATH=$C/core:$C/beam,XLA_PYTHON_CLIENT_PREALLOCATE=false,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,JAX_ENABLE_X64=true"
apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -c 'import jax; assert jax.default_backend() == "gpu", jax.devices(); print(jax.devices())'
set +e
/usr/bin/time -v -o source.time apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_target_run --states "$W/out/input_generic_v1/source_states.npz" --certificate "$W/out/input_generic_v1/crosscheck.json" --geom "$G/overlays/$POINT.txt" --offline-root "$G/stage/offline" --maps "$G/stage/maps/DSMap.header" "$G/stage/maps/TSdMap.header" --capacity 256 --max-steps 100000 --h 1 --dtype float64 --out "$O/births.json" > source.log 2>&1
rc=$?
set -e
printf 'exit=%s time=%s\n' "$rc" "$(date -Is)" > outcome.txt
cat source.log
exit "$rc"
