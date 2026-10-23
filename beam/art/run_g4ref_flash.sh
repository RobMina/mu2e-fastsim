#!/bin/bash
# run_g4ref_flash.sh OUTDIR NEVENTS SEED : stock EleBeamResampler early-flash stage on EleBeamCat.Run1Baa with the
# StrawGasStep/SimParticle table dump (beam/fcl/g4ref_elebeam_dump.fcl), in the tracker art area (MDC2025ay backing,
# StrawGasStepTableDumper built there)
OUT=$1; NEV=${2:-2000}; SEED=${3:-1}
HERE=$(cd "$(dirname "$0")/.." && pwd)
AREA=$(cd "$HERE/../tracker/art" && pwd)
mkdir -p "$OUT" && cd "$OUT" || exit 2
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
muse setup ops
samweb list-definition-files sim.mu2e.EleBeamCat.Run1Baa.art | sort | while read f; do echo "$(samweb locate-file "$f" | head -1 | sed -E 's/^[a-z]+://; s/\(.*\)$//')/$f"; done > elebeamcat.txt
cd "$AREA" && muse setup || exit 3
cd "$OUT"
{ echo "#include \"g4ref_elebeam_dump.fcl\""
  echo "physics.filters.beamResampler.fileNames : [ $(sed 's/.*/"&"/' elebeamcat.txt | paste -sd,) ]"
  echo "services.SeedService.baseSeed : $SEED"
  echo "source.maxEvents : $NEV"; } > job.fcl
export FHICL_FILE_PATH=$HERE/fcl:$FHICL_FILE_PATH
mu2e -c job.fcl --debug-config job_config.txt >/dev/null 2>&1
grep -E "bFieldFile|inputFile|physicsListName" job_config.txt | head > job_geom.txt
/usr/bin/time -v mu2e -c job.fcl > job.log 2>&1
echo "exit $?" >> job.log
