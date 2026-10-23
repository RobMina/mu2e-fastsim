#!/bin/bash
# mu2ebo_beam_point.sh MU2EBO_CHECKOUT CONFIG : graph.run of the Mu2eBO study fastsim_beam_point on EAF from a GPVM
# (Mu2eBO fork branch fastsimkit >= a0949a1; data root sandboxed under beam_fastsim/mu2ebo)
cd "$1" || exit 2
source /cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0/bin/activate
export AUTORESEARCH_VENV=/cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0
export AUTORESEARCH_DATA_ROOT=/exp/mu2e/data/users/ram2aq/beam_fastsim/mu2ebo
export AUTORESEARCH_STUDY_PATH=$PWD/tests/fixtures/engine_studies
export M2FS_CODE=/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim
export EAF_TOKEN_FILE=/exp/mu2e/app/users/ram2aq/goose/secrets/eaf_token
mkdir -p "$AUTORESEARCH_DATA_ROOT"
python -m graph.run --study fastsim_beam_point --config "$2" --campaign beampt
echo "graph.run exit=$?"
