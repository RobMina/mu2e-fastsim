#!/bin/bash
#SBATCH -J r5_exposure_census
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 00:20:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/census_exposure.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/code_census_v5
O=$W/out/census_exposure_v5
mkdir "$O"
cd "$O"
cat > config.fcl <<EOF
#include "Offline/fcl/minimalMessageService.fcl"
process_name: exposurecensus
source: { module_type: EmptyEvent maxEvents: 1 }
services: { message: @local::default_message }
physics: { analyzers: {} e1: [] end_paths: [e1] }
EOF
cat > inspect.sh <<EOF
set -eo pipefail
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
muse setup ops
export PYTHONPATH="$C/beam:\${PYTHONPATH:-}"
python3 -c 'from metacat.webapi import MetaCatClient; print("metacat read client available")'
python3 -m beamfastsim.flash_census --discover-namespace mu2e --pattern 'mcs.*EleBeam*' --pattern 'sim.*EleBeamFlash*' --pattern 'nts.*elebeamflash*' --pattern 'nts.*EleBeamFlash*' --examples 1 --out "$O/production.json"
python3 -m beamfastsim.flash_census --discover-namespace oksuzian --pattern '*EleBeam*' --pattern '*elebeam*' --examples 1 --out "$O/boards.json"
EOF
export CVMFSEXEC_DIST=$R/environments/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$R:$R,$C:$C" OFFLINE_LOG_PREFIX=census OFFLINE_CHAIN=$O/inspect.sh
bash "$R/trk_field_y4_20261101/runtime/offline_uva_chain.sh" /cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_39_00 "$O/config.fcl"
printf 'completed=%s\n' "$(date -Is)" > complete.txt
