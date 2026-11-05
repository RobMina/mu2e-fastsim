#!/bin/bash
#SBATCH -J r5_generic_schema
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/schema.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/code_timing
O=$W/out/input_generic_v1
test -f "$O/complete.txt"
test ! -e "$O/schema.json"
apptainer exec --cleanenv --env "PYTHONPATH=$C/beam,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" -B "$R" -B "$C" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_input "$O/states.root" --schema-only --out "$O/schema.json"
cat "$O/schema.json"
