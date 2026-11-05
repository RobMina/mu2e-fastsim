#!/bin/bash
#SBATCH -J r5_sandbox_probe
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
O=$W/out/sandbox_runtime_probe_${SLURM_JOB_ID:?}
mkdir "$O"
exec > "$O/probe.log" 2>&1
ulimit -c 0
ulimit -f 131072
LOCAL=$(mktemp -d /tmp/r5-sandbox-probe.XXXXXXXX)
trap 'rm -rf "$LOCAL"' EXIT
A=$O/area
mkdir -p "$A/build/al9-prof-e29-p109/FlashReference/lib" "$LOCAL/tools"
cp -a "$D/FlashReference" "$A/FlashReference"
printf 'ENVSET p109\n' > "$A/.muse"
ln -s /cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_41_01 "$A/backing"
cp "$D/FlashReferenceExporter.plugin" "$A/build/al9-prof-e29-p109/FlashReference/lib/libflashreference_FlashReferenceExporter_module.so"
cp "$D/FlashReferenceFixture.plugin" "$A/build/al9-prof-e29-p109/FlashReference/lib/libflashreference_FlashReferenceFixture_module.so"
cp "$(command -v bwrap)" "$LOCAL/tools/bwrap"
cp -a "$D/checks" "$O/checks"
cp "$C/sandbox_runtime_probe.sh" "$O/inside.sh"
cat > "$O/chain.sh" <<EOF
export R5_SANDBOX_ROOT=$O R5_SANDBOX_CODE=$O/checks R5_HOST_BWRAP=$LOCAL/tools/bwrap
bash $O/inside.sh
EOF
scontrol show job "$SLURM_JOB_ID" -o > "$O/slurm.txt"
export CVMFSEXEC_DIST=$R/environments/cvmfsexec-el8
export OFFLINE_IMAGE=/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40
export OFFLINE_BINDS="$R:$R,$LOCAL:$LOCAL,$D:$D" OFFLINE_LOG_PREFIX=sandbox_probe OFFLINE_CHAIN=$O/chain.sh
cd "$O"
bash "$R/trk_field_y4_20261101/runtime/offline_uva_chain.sh" "$A" "$A/FlashReference/fcl/fixture.fcl"
