#!/bin/bash
#SBATCH -J r5_resume_high
#SBATCH -A ldmxuva
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/resume_high.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=$W/code_v6
G=$R/bo_sob_y5b_20261103/inputs
O=$W/out/target_births_bpzup12_v6
mkdir "$O"
cd "$O"
nvidia-smi -L > gpu.txt
E="PYTHONPATH=$C/core:$C/beam,XLA_PYTHON_CLIENT_PREALLOCATE=false,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,JAX_ENABLE_X64=true"
apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -c 'import jax; assert jax.default_backend() == "gpu", jax.devices(); print(jax.devices())'
common=(--states "$W/out/input_generic_v1/source_states.npz" --certificate "$W/out/input_generic_v1/crosscheck.json" --geom "$G/overlays/bpzup12.txt" --offline-root "$G/stage/offline" --maps "$G/stage/maps/DSMap.header" "$G/stage/maps/TSdMap.header" --capacity 256 --h 1 --dtype float64)
set +e
/usr/bin/time -v -o initial.time apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_target_run "${common[@]}" --max-steps 100000 --checkpoint "$O/checkpoint.json" --out "$O/births.json" > initial.log 2>&1
rc=$?
set -e
printf 'exit=%s time=%s\n' "$rc" "$(date -Is)" > initial.outcome.txt
if [ "$rc" -ne 0 ] && [ "$rc" -ne 2 ]; then exit "$rc"; fi
apptainer exec --cleanenv --env "PYTHONPATH=$C/core:$C/beam,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" -B "$R" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_transport_report --births "$O/births.npz" --reference "$W/out/target_births_bpzup12_v3/births.npz" --geom "$G/overlays/bpzup12.txt" --offline-root "$G/stage/offline" --out "$O/transport_initial.json"
set +e
/usr/bin/time -v -o continued.time apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_target_run "${common[@]}" --max-steps 300000 --chunk-steps 100000 --resume "$O/checkpoint.json" --checkpoint "$O/continued_checkpoint.json" --out "$O/continued.json" > continued.log 2>&1
rc=$?
set -e
printf 'exit=%s time=%s\n' "$rc" "$(date -Is)" > continued.outcome.txt
if [ "$rc" -ne 0 ] && [ "$rc" -ne 2 ]; then exit "$rc"; fi
apptainer exec --cleanenv --env "PYTHONPATH=$C/core:$C/beam,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" -B "$R" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_birth_report --births "$O/continued.npz" --geom "$G/overlays/bpzup12.txt" --offline-root "$G/stage/offline" --out "$O/regions_continued.json"
apptainer exec --cleanenv --env "PYTHONPATH=$C/core:$C/beam,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" -B "$R" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_transport_report --births "$O/continued.npz" --geom "$G/overlays/bpzup12.txt" --offline-root "$G/stage/offline" --out "$O/transport_continued.json"
printf 'completed=%s\n' "$(date -Is)" > complete.txt
