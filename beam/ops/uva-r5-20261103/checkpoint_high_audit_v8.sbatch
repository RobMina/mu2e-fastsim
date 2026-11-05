#!/bin/bash
#SBATCH -J r5_high_chain_audit
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:05:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/checkpoint_high_audit.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/code_audit_v8
O=$W/out/target_births_bpzup12_v6
G=$R/bo_sob_y5b_20261103/inputs
test -f "$O/complete.txt"
apptainer exec --cleanenv --env "PYTHONPATH=$C/beam,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" -B "$R" -B "$C" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_checkpoint_report "$O/checkpoint.json" "$O"/continued_checkpoint.chunk*.json --out "$O/checkpoint_chain_audit.json"
A=$W/code_audit_v8_looper
mkdir "$A"
cp -a "$W/code_v6/beam" "$A/beam"
cp "$C/beam/beamfastsim/flash_transport_report.py" "$A/beam/beamfastsim/"
apptainer exec --cleanenv --env "PYTHONPATH=$W/code_v6/core:$A/beam,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" -B "$R" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_transport_report --births "$O/continued.npz" --geom "$G/overlays/bpzup12.txt" --offline-root "$G/stage/offline" --out "$O/transport_loopers.json"
