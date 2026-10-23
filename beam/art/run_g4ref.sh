#!/bin/bash
# run_g4ref.sh OUTDIR NEVENTS SEED [GEOM] : stock MuBeamResampler DS stage on MuBeamCat.MDC2025ac with the per-muon tree (beam/fcl/g4ref_mubeam_ntpart.fcl)
# GEOM: optional geometry file name under beam/geom (e.g. move_extras12.txt), found through MU2E_SEARCH_PATH
OUT=$1; NEV=${2:-20000}; SEED=${3:-1}; GEOM=${4:-}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT" && cd "$OUT" || exit 2
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
muse setup ops
samweb list-definition-files sim.mu2e.MuBeamCat.MDC2025ac.art | sort | while read f; do echo "$(samweb locate-file "$f" | head -1 | sed -E 's/^[a-z]+://; s/\(.*\)$//')/$f"; done > mubeamcat.txt
muse setup SimJob MDC2025ac || exit 3
{ echo "#include \"g4ref_mubeam_ntpart.fcl\""
  echo "physics.filters.beamResampler.fileNames : [ $(sed 's/.*/"&"/' mubeamcat.txt | paste -sd,) ]"
  echo "services.SeedService.baseSeed : $SEED"
  echo "source.maxEvents : $NEV"
  [ -n "$GEOM" ] && echo "services.GeometryService.inputFile : \"$GEOM\""; } > job.fcl
export FHICL_FILE_PATH=$HERE/fcl:$FHICL_FILE_PATH MU2E_SEARCH_PATH=$HERE/geom:$MU2E_SEARCH_PATH
mu2e -c job.fcl --debug-config job_config.txt >/dev/null 2>&1
grep -E "GeometryService|bFieldFile|inputFile|physicsListName" job_config.txt | head > job_geom.txt
/usr/bin/time -v mu2e -c job.fcl > job.log 2>&1
echo "exit $?" >> job.log
