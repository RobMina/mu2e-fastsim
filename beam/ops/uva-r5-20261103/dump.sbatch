#!/bin/bash
#SBATCH -J r5_flash_dump
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/dump.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
A=$R/trk_ledger_s3_20261028/evidence/tracker_runtime_stage_v30/Code_tv2.tar.bz2
RUN=${RUN:-pilot_v2}
O=$W/out/$RUN
mkdir "$O"
cd "$O"
echo "dcb2bd3bb0581fcf422ec081aee5ec603aa54602996042e751187fc829c09a19  $A" | sha256sum -c -
cat > dump_flash.fcl <<'EOF'
#include "TrkFastSim/fcl/dump_sgsteps.fcl"
physics.analyzers.dump.genParticles: "compressDetStepMCs"
physics.analyzers.dump.writeTrajectories: false
physics.analyzers.genCount: { module_type: GenEventCountReader makeHistograms: false }
physics.analyzers.eventCount: { module_type: RunSubrunEvent printSam: false }
physics.e1: [ dump, genCount, eventCount ]
EOF
sha256sum dump_flash.fcl > fcl.sha256
export CVMFSEXEC_DIST=$R/environments/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$R:$R" OFFLINE_LOG_PREFIX=flash_dump
export OFFLINE_OUTPUT_FILE="$O/steps.root.tmp"
bash /home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/runtime/offline_uva.sh "$A" "$O/dump_flash.fcl" "$@"
mv steps.root.tmp steps.root
printf 'completed=%s\n' "$(date -Is)" > complete.txt
