#!/bin/bash
#SBATCH -J r5_input_crosscheck
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/crosscheck.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/code_timing
O=$W/out/input_generic_v1
test -f "$O/complete.txt"
test ! -e "$O/crosscheck.json"
apptainer exec --cleanenv --env "PYTHONPATH=$C/beam,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" -B "$R" -B "$C" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_step_input --generic "$O/states.root" --legacy "$W/out/input_stock_v1/states.root" --log "$O/read.log" --without-ancestry --out "$O/crosscheck.json" --states "$O/source_states.npz"
cat "$O/crosscheck.json"
