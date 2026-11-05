#!/bin/bash
#SBATCH -J r5_ref_preflight
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/reference_preflight.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103
O=$W/out/reference_preflight_${SLURM_JOB_ID:?}
mkdir "$O"
cd "$O"
scontrol show job "${SLURM_JOB_ID:?}" -o > slurm.txt
cp "$C/reference_raw.fcl" raw.fcl
cp "$C/reference_prepare.py" prepare.py
cat > inspect.sh <<EOF
set -eo pipefail
A=/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_41_01
mu2e --print-description Mu2eG4 > "$O/Mu2eG4.description.txt"
mu2e --print-description RootOutput > "$O/RootOutput.description.txt"
for name in Mu2eG4Cuts.cc Mu2eG4TrackingAction.cc Mu2eG4PrimaryGeneratorAction.cc Mu2eG4SteppingAction.cc; do
  cp "\$A/Offline/Mu2eG4/src/\$name" "$O/\$name.txt"
done
cp "\$A/Offline/Mu2eG4/inc/Mu2eG4Config.hh" "$O/Mu2eG4Config.hh.txt"
cp "\$A/Offline/Mu2eG4/fcl/prolog.fcl" "$O/g4.prolog.fcl"
cp "\$A/Offline/Mu2eG4/src/physicsListDecider.cc" "$O/physicsListDecider.cc.txt"
sha256sum "\$(command -v mu2e)" > "$O/executable.sha256"
python3 "$O/prepare.py" --input "$W/inputs/elebeamcat_ab_20000.art" --geom "$R/bo_sob_y5b_20261103/inputs/overlays/bpzup21.txt" --offline-root "\$A" --field "\$A/Offline/Mu2eG4/geom/bfgeom_v01.txt" --template "$O/raw.fcl" --events 1000 --seed 51021 --physics-list FTFP_BERT --out "$O/low"
python3 "$O/prepare.py" --input "$W/inputs/elebeamcat_ab_20000.art" --geom "$R/bo_sob_y5b_20261103/inputs/overlays/bpzup12.txt" --offline-root "\$A" --field "\$A/Offline/Mu2eG4/geom/bfgeom_v01.txt" --template "$O/raw.fcl" --events 1000 --seed 51012 --physics-list FTFP_BERT --out "$O/high"
export FHICL_FILE_PATH="$O/low:\$FHICL_FILE_PATH"
fhicl-dump -c "$O/low/job.fcl" > "$O/low.expanded.fcl"
export FHICL_FILE_PATH="$O/high:\$FHICL_FILE_PATH"
fhicl-dump -c "$O/high/job.fcl" > "$O/high.expanded.fcl"
printf 'configuration_only=true\ntransport_events=0\ncompleted=%s\n' "\$(date -Is)" > "$O/complete.txt"
EOF
export CVMFSEXEC_DIST=$R/environments/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$R:$R" OFFLINE_LOG_PREFIX=reference_preflight OFFLINE_CHAIN=$O/inspect.sh
bash "$R/trk_field_y4_20261101/runtime/offline_uva_chain.sh" /cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_41_01 "$O/raw.fcl"
