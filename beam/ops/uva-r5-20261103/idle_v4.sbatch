#!/bin/bash
#SBATCH -J r5_idle_regression
#SBATCH -A ldmxuva
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/idle_v4.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=$W/code_v4
G=$R/bo_sob_y5b_20261103/inputs
O=$W/out/idle_v4
mkdir "$O"
cd "$O"
nvidia-smi -L > gpu.txt
E="PYTHONPATH=$C/core:$C/beam,XLA_PYTHON_CLIENT_PREALLOCATE=false,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,JAX_ENABLE_X64=true,FLASH_CE_BASELINE_SOURCE=$C/cechain_default_baseline.py"
apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -c 'import jax; assert jax.default_backend() == "gpu", jax.devices(); print(jax.devices())'
/usr/bin/time -v -o test.time apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -m pytest -q "$C/beam/tests/test_flash_idle_jax.py" "$C/beam/tests/test_flash_ce_defaults.py" "$C/beam/tests/test_flash_target_jax.py" > tests.log 2>&1
cat tests.log
P=$W/out/target_births_bpzup21_v4
mkdir "$P"
set +e
/usr/bin/time -v -o "$P/source.time" apptainer exec --nv --cleanenv --env "$E" -B "$R" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_target_run --states "$W/out/input_generic_v1/source_states.npz" --certificate "$W/out/input_generic_v1/crosscheck.json" --geom "$G/overlays/bpzup21.txt" --offline-root "$G/stage/offline" --maps "$G/stage/maps/DSMap.header" "$G/stage/maps/TSdMap.header" --capacity 256 --max-steps 100000 --h 1 --dtype float64 --out "$P/births.json" > "$P/source.log" 2>&1
rc=$?
set -e
printf 'exit=%s time=%s\n' "$rc" "$(date -Is)" > "$P/outcome.txt"
if [ "$rc" -ne 0 ] && [ "$rc" -ne 2 ]; then exit "$rc"; fi
apptainer exec --cleanenv --env "PYTHONPATH=$C/core:$C/beam,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" -B "$R" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_transport_report --births "$P/births.npz" --reference "$W/out/target_births_bpzup21_v3/births.npz" --geom "$G/overlays/bpzup21.txt" --offline-root "$G/stage/offline" --out "$P/transport.json"
printf 'completed=%s\n' "$(date -Is)" > complete.txt
