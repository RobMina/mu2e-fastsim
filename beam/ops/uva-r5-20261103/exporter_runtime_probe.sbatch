#!/bin/bash
#SBATCH -J r5_exporter_probe
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -eo pipefail
module load apptainer
R=/scratch/ram2aq/mu2efastsim
W=$R/flash_r5_20261103
C=/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103
D=$C/exporter_runtime_v1
O=$W/out/exporter_runtime_probe_${SLURM_JOB_ID:?}
mkdir "$O"
exec > "$O/probe.log" 2>&1
ulimit -c 0
ulimit -f 131072
LOCAL=$(mktemp -d /tmp/r5-exporter-probe.XXXXXXXX)
trap 'rm -rf "$LOCAL"' EXIT
A=$LOCAL/area
mkdir -p "$A/build/al9-prof-e29-p109/FlashReference/lib"
cp -a "$D/FlashReference" "$A/FlashReference"
printf 'ENVSET p109\n' > "$A/.muse"
ln -s /cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_41_01 "$A/backing"
cp "$D/FlashReferenceExporter.plugin" "$A/build/al9-prof-e29-p109/FlashReference/lib/libflashreference_FlashReferenceExporter_module.so"
cp "$D/FlashReferenceFixture.plugin" "$A/build/al9-prof-e29-p109/FlashReference/lib/libflashreference_FlashReferenceFixture_module.so"
printf '%s  %s\n' bf61d4b5fafc6ac52119059f08171e943384da81a3aaba28094a73bc57284189 "$A/build/al9-prof-e29-p109/FlashReference/lib/libflashreference_FlashReferenceExporter_module.so" 1ec75a380b4714fe89827ccdea52596cb83d70827c3bee5def242fdde7df8c15 "$A/build/al9-prof-e29-p109/FlashReference/lib/libflashreference_FlashReferenceFixture_module.so" | sha256sum -c -
cp "$C/runtime_probe.sh" "$O/runtime_probe.sh"
cp "$(command -v bwrap)" "$LOCAL/bwrap"
cat > "$O/chain.sh" <<EOF
export R5_PROBE_ROOT=$O R5_PROBE_AREA=$A R5_PROBE_CODE=$D/code R5_HOST_BWRAP=$LOCAL/bwrap
bash $O/runtime_probe.sh
EOF
scontrol show job "$SLURM_JOB_ID" -o > "$O/slurm.txt"
uname -a > "$O/host.txt"
export CVMFSEXEC_DIST=$R/environments/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$R:$R,$LOCAL:$LOCAL,$D:$D" OFFLINE_LOG_PREFIX=runtime_probe OFFLINE_CHAIN=$O/chain.sh
cd "$O"
bash "$R/trk_field_y4_20261101/runtime/offline_uva_chain.sh" "$A" "$A/FlashReference/fcl/fixture.fcl"
export APPTAINERENV_PYTHONPATH=$D/code/beam APPTAINERENV_PYTHONDONTWRITEBYTECODE=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1
apptainer exec --cleanenv -B "$R" -B "$D" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_reference_fixture --input "$O/fixture/fixture.root" --out "$O/fixture/check.json"
apptainer exec --cleanenv -B "$R" -B "$D" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_reference_fixture --input "$O/reread/reread.root" --out "$O/reread/check.json"
apptainer exec --cleanenv -B "$R" -B "$D" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_reference_compare --left "$O/fixture/fixture.root" --right "$O/reread/reread.root" --out "$O/reread/comparison.json"
apptainer exec --cleanenv -B "$R" -B "$D" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_reference_partition --full "$O/source/source.root" --first "$O/first10/source.root" --remainder "$O/remaining990/source.root" --out "$O/partition.json"
apptainer exec --cleanenv -B "$R" -B "$D" "$R/mu2efastsim.sif" python3 -m beamfastsim.flash_reference_source_check --input "$O/source/source.root" --states "$W/out/input_generic_v1/source_states.npz" --states-sha256 7a8e70c6b070c6ea89fee80715763671329d77c61451110a9114920da23e0a0b --out "$O/source/independent_check.json"
printf 'analytic_and_persistence_checks=true\ntransport_events=0\ncompleted=%s\n' "$(date -Is)" > "$O/validated.txt"
