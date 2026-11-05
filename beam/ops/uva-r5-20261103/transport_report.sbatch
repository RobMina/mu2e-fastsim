#!/bin/bash
#SBATCH -J r5_transport_report
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/transport_report.%j.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/code_transport_report
G=$R/bo_sob_y5b_20261103/inputs
for POINT in bpzup21 bpzup12; do
    O=$W/out/target_births_${POINT}_v3
    test -f "$O/births.npz"
    test ! -e "$O/transport.json"
    apptainer exec --cleanenv --env "PYTHONPATH=$W/code_v3/core:$C/beam,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" -B "$R" -B "$C" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_transport_report --births "$O/births.npz" --geom "$G/overlays/$POINT.txt" --offline-root "$G/stage/offline" --out "$O/transport.json"
done
