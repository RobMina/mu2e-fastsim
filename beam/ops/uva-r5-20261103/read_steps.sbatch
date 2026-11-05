#!/bin/bash
#SBATCH -J r5_input_generic
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/input_generic.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103
O=$W/out/input_generic_v1
mkdir "$O"
cd "$O"
cp "$C/read_flash_steps.fcl" input.fcl
cat > inspect.sh <<EOF
set -eo pipefail
config_dumper -P "$W/inputs/elebeamcat_ab_20000.art" > "$O/source.config.txt"
fhicl-dump -c "$O/input.fcl" > "$O/expanded.fcl"
mu2e -c "$O/input.fcl" -n 1 -s "$W/inputs/elebeamcat_ab_20000.art" -T "$O/pilot.root" > "$O/pilot.log" 2>&1
/usr/bin/time -v -o "$O/read.time" mu2e -c "$O/input.fcl" -n 1000 -s "$W/inputs/elebeamcat_ab_20000.art" -T "$O/states.root.tmp" > "$O/read.log" 2>&1
mv "$O/states.root.tmp" "$O/states.root"
printf 'completed=%s\n' "\$(date -Is)" > "$O/complete.txt"
EOF
export CVMFSEXEC_DIST=$R/environments/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$R:$R" OFFLINE_LOG_PREFIX=generic_input OFFLINE_CHAIN=$O/inspect.sh
bash "$R/trk_field_y4_20261101/runtime/offline_uva_chain.sh" /cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_39_00 "$O/input.fcl"
