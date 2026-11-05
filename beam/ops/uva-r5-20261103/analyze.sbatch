#!/bin/bash
#SBATCH -J r5_s0_analyze
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o /scratch/ram2aq/mu2efastsim/flash_r5_20261103/logs/analyze.%A.%a.log
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/code_v1
case $SLURM_ARRAY_TASK_ID in 0) CONFIG=bpzup21 ;; 1) CONFIG=bpzup12 ;; *) exit 2 ;; esac
O=$W/out/${CONFIG}_v1
test -f "$O/complete.txt"
grep -q 'GenEventCount total: 11000000 events in 100 SubRuns' "$W/logs/s0.22434755.$SLURM_ARRAY_TASK_ID.log"
cd "$O"
test ! -e ancestry_norm.json
sha256sum "$C/beam/beamfastsim/flash_ancestry.py" > analysis_source.sha256
E="PYTHONPATH=$C/beam,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1"
/usr/bin/time -v -o analysis.time apptainer exec --cleanenv --env "$E" -B "$R" -B "$C" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_ancestry "$O/steps.root" --out "$O/ancestry_norm.json" --artifact "$O/flash_dump.expanded.fcl" --artifact "$O/complete.txt" --certified-generated-events 11000000 --z-bins -30000 0 4000 5000 6000 7000 8000 9000 12000 30000 --r-bins 0 1000 3000 3500 4000 4500 5000 10000 50000
printf 'completed=%s\n' "$(date -Is)" > analysis_complete.txt
