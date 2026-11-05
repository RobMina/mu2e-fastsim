#!/bin/bash
#SBATCH -J r5_input_states
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/input.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
I=$W/inputs/elebeamcat_ab_20000.art
test -f "$I" || { echo 'Source transfer incomplete; no inspection attempted'; exit 3; }
O=$W/out/input_states_v1
mkdir "$O"
cd "$O"
cat > input.fcl <<'EOF'
#include "TrkFastSim/fcl/dump_sgsteps.fcl"
physics.analyzers.dump.genParticles: "generate"
physics.analyzers.dump.simParticles: "BeamFilter"
physics.analyzers.dump.vdSteps: "BeamFilter:Beam"
physics.analyzers.dump.writeTrajectories: false
EOF
export CVMFSEXEC_DIST=$R/environments/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$R:$R" OFFLINE_LOG_PREFIX=input_states OFFLINE_EVENTS=1000
export OFFLINE_OUTPUT_FILE="$O/states.root.tmp"
A=$R/trk_ledger_s3_20261028/evidence/tracker_runtime_stage_v30/Code_tv2.tar.bz2
echo "dcb2bd3bb0581fcf422ec081aee5ec603aa54602996042e751187fc829c09a19  $A" | sha256sum -c -
bash /home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/runtime/offline_uva.sh "$A" "$O/input.fcl" "$I"
mv states.root.tmp states.root
printf 'completed=%s\n' "$(date -Is)" > complete.txt
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/code_v1
apptainer exec --cleanenv --env "PYTHONPATH=$C/beam,OPENBLAS_NUM_THREADS=1" -B "$R" -B "$C" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_input "$O/states.root" --out "$O/states.json"
