#!/bin/bash
# Make art/ a muse work area for the CrvFastSim package, backed by the CVMFS Musing (no muse commands: the user builds).
# CrvFastSim must be a real directory here: muse's SConscript walk skips symlinked repos.
set -e
AREA="${CRVFS_CODE:?source site.env}/art"
ln -sfn /cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ay "$AREA/backing"
ls -la "$AREA"
echo "next (user, on mu2ebuild02):"
echo "  cd $AREA && source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && setup muse && muse setup && muse build -j 32"
