#!/bin/bash
#SBATCH -J r5_exporter_checks
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
D=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/exporter_runtime_v1
I=$W/out/exporter_runtime_probe_22475084
O=$W/out/exporter_probe_checks_${SLURM_JOB_ID:?}
mkdir "$O"
exec > "$O/checks.log" 2>&1
ulimit -c 0
ulimit -f 32768
export APPTAINERENV_PYTHONPATH=$D/checks/beam APPTAINERENV_PYTHONDONTWRITEBYTECODE=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1
check() { apptainer exec --cleanenv -B "$R" -B "$D" "$R/mu2efastsim.sif" python3 -m "beamfastsim.$1" "${@:2}"; }
check flash_reference_fixture --input "$I/fixture/fixture.root" --out "$O/fixture.json"
check flash_reference_fixture --input "$I/reread/reread.root" --out "$O/reread.json"
check flash_reference_compare --left "$I/fixture/fixture.root" --right "$I/reread/reread.root" --out "$O/persistence.json"
check flash_reference_partition --full "$I/source/source.root" --first "$I/first10/source.root" --remainder "$I/remaining990/source.root" --out "$O/partition.json"
check flash_reference_source_check --input "$I/source/source.root" --states "$W/out/input_generic_v1/source_states.npz" --states-sha256 7a8e70c6b070c6ea89fee80715763671329d77c61451110a9114920da23e0a0b --out "$O/independent_source.json"
printf 'existing_outputs_only=true\ntransport_events=0\nchecks_passed=true\ncompleted=%s\n' "$(date -Is)" > "$O/complete.txt"
