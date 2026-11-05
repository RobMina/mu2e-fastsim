#!/bin/bash
#SBATCH -J r5_board_provenance
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/config.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
O=$W/out/config_v1
mkdir "$O"
cd "$O"
cat > config.fcl <<EOF
#include "Offline/fcl/minimalMessageService.fcl"
process_name: r5config
source: { module_type: EmptyEvent maxEvents: 1 }
services: { message: @local::default_message }
physics: { analyzers: {} e1: [] end_paths: [e1] }
EOF
cat > inspect.sh <<EOF
set -eo pipefail
config_dumper -P "$W/inputs/bpzup21_000.art" > "$O/bpzup21.config.txt"
config_dumper -P "$W/inputs/boards/bpzup12/dts.oksuzian.EarlyEleBeamFlash.MDC2025ax_bpzup12.001803_00000000.art" > "$O/bpzup12.config.txt"
EOF
export CVMFSEXEC_DIST=$R/environments/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$R:$R" OFFLINE_LOG_PREFIX=config OFFLINE_CHAIN=$O/inspect.sh
bash "$R/trk_field_y4_20261101/runtime/offline_uva_chain.sh" /cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_39_00 "$O/config.fcl"
printf 'completed=%s\n' "$(date -Is)" > complete.txt
