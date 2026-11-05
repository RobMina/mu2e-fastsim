#!/bin/bash
#SBATCH -J r5_input_stock
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/read_input.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
O=$W/out/input_stock_v1
mkdir "$O"
cd "$O"
export CVMFSEXEC_DIST=$R/environments/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$R:$R" OFFLINE_LOG_PREFIX=input_stock OFFLINE_EVENTS=1000
export OFFLINE_OUTPUT_FILE="$O/states.root.tmp"
bash /home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/runtime/offline_uva.sh /cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_39_00 /home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/read_flash_input.fcl "$W/inputs/elebeamcat_ab_20000.art"
mv states.root.tmp states.root
printf 'completed=%s\n' "$(date -Is)" > complete.txt
