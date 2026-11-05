#!/bin/bash
#SBATCH -J r5_s0_difference
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/difference.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/code_v1
cd "$W/out"
test ! -e difference.json
apptainer exec --cleanenv --env "PYTHONPATH=$C/beam,OPENBLAS_NUM_THREADS=1" -B "$W" -B "$C" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_report bpzup21_v1/ancestry_norm.json bpzup12_v1/ancestry_norm.json --out difference.json
