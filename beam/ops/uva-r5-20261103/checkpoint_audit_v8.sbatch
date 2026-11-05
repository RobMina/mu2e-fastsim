#!/bin/bash
#SBATCH -J r5_checkpoint_audit
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:05:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/checkpoint_audit.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/code_audit_v8
O=$W/out/target_births_bpzup21_v6
apptainer exec --cleanenv --env "PYTHONPATH=$C/beam,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" -B "$R" -B "$C" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_checkpoint_report "$O/checkpoint.json" "$O"/continued_checkpoint.chunk*.json --out "$O/checkpoint_chain_audit.json"
