#!/bin/bash
# Make the stock-validation muse area: Offline main from a CI backing build, Mu2e/DQM (PR #48 branch until it merges)
# and a copy of the CrvFastSim package. No muse command here: the user builds (printed at the end).
# usage: setup_crvval_area.sh [AREA] [CI_BUILD]   (defaults: /exp/mu2e/app/users/ram2aq/crvval, newest CI build of main)
set -e
AREA=${1:-/exp/mu2e/app/users/ram2aq/crvval}
CI=/cvmfs/mu2e-development.opensciencegrid.org/museCIBuild/main
BACK=${2:-$CI/$(ls -t "$CI" | head -1)}
SRC="${CRVFS_CODE:?source site.env}/art/CrvFastSim"
[ -f "$BACK/Offline/CRVDQM/data/CRVDQM_binning_v3_extracted.txt" ] || { echo "$BACK lacks CRVDQM binning v3"; exit 2; }
mkdir -p "$AREA" && cd "$AREA"
ln -sfn "$BACK" backing
[ -d DQM ] || git clone -q https://github.com/RobMina/DQM.git DQM
git -C DQM fetch -q origin && git -C DQM checkout -q crvdqm-configuration && git -C DQM merge -q --ff-only origin/crvdqm-configuration
# a real copy, not a symlink: muse skips symlinked packages; re-run this script after editing $SRC
mkdir -p CrvFastSim && cp -a "$SRC/." CrvFastSim/
echo "backing -> $(readlink backing)"
echo "Offline $(cat backing/Offline/.git/HEAD 2>/dev/null | cut -c1-12); DQM $(git -C DQM rev-parse --short HEAD) ($(git -C DQM branch --show-current))"
echo "next (user, on mu2ebuild02):"
echo "  cd $AREA && source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && setup muse && muse setup && muse build -j 32"
