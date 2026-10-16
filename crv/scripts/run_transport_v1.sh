#!/bin/bash
# Stage-A transport over the 20 g4ref_v1 gen tables, then V1 against the G4 tables (one GPVM, ~2 min).
# usage: run_transport_v1.sh [stage] [outtag]   (needs site.env: CRVFS_CODE, CRVFS_DATA)
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
pyenv ana >/dev/null 2>&1 || true
set -u
STAGE=${1:-A}; TAG=${2:-fast${1:-A}_v1}
PKG=${CRVFS_CODE:?source site.env}
T=${CRVFS_DATA:?source site.env}/g4ref_v1_tables
F=$CRVFS_DATA/$TAG
P=$CRVFS_DATA/plots/v1_$TAG
cd "$PKG"
python3 -m crvfastsim transport "$T/gen.*.root" --outdir "$F" --stage "$STAGE" | grep -v "^wrote"
python3 -m crvfastsim v1compare "$F/fast$STAGE.*.root" --compare "$T/crvsteps.*.root" --outdir "$P"
