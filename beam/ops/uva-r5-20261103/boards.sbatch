#!/bin/bash
#SBATCH -J r5_flash_s0
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/s0.%A.%a.log
set -eo pipefail
R=/scratch/ram2aq/mu2efastsim/flash_r5_20261103
case $SLURM_ARRAY_TASK_ID in
  0) CONFIG=bpzup21 ;;
  1) CONFIG=bpzup12 ;;
  *) exit 2 ;;
esac
export RUN=${CONFIG}_v1
bash /home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/dump.sbatch "$R/inputs/boards/$CONFIG/"*.art
