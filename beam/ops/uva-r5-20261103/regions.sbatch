#!/bin/bash
#SBATCH -J r5_birth_regions
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/regions.%A.%a.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/code_regions
G=$R/bo_sob_y5b_20261103/inputs
case $SLURM_ARRAY_TASK_ID in 0) CONFIG=bpzup21 ;; 1) CONFIG=bpzup12 ;; *) exit 2 ;; esac
O=$W/out/${CONFIG}_v1
cd "$O"
test -f complete.txt
test ! -e regions.json
E="PYTHONPATH=$W/code_v2/core:$C/beam,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1"
/usr/bin/time -v -o regions.time apptainer exec --cleanenv --env "$E" -B "$R" -B "$C" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_regions --input "$O/steps.root" --geom "$G/overlays/$CONFIG.txt" --offline-root "$G/stage/offline" --generated-events 11000000 --out "$O/regions.json"
