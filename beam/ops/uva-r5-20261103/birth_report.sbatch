#!/bin/bash
#SBATCH -J r5_birth_report
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/birth_report.%A.%a.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/code_birth_report
G=$R/bo_sob_y5b_20261103/inputs
case $SLURM_ARRAY_TASK_ID in 0) POINT=bpzup21 ;; 1) POINT=bpzup12 ;; *) exit 2 ;; esac
O=$W/out/target_births_${POINT}_v3
if [ ! -f "$O/births.npz" ]; then echo 'No source output, no report'; exit 3; fi
test ! -e "$O/regions.json"
apptainer exec --cleanenv --env "PYTHONPATH=$W/code_v3/core:$C/beam,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" -B "$R" -B "$C" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_birth_report --births "$O/births.npz" --geom "$G/overlays/$POINT.txt" --offline-root "$G/stage/offline" --out "$O/regions.json"
cat "$O/regions.json"
