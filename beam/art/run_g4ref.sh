#!/bin/bash
# run_g4ref.sh OUTDIR NEVENTS SEED [GEOM] : stock MuBeamResampler DS stage on MuBeamCat (default MDC2025ac) with the per-muon tree (beam/fcl/g4ref_mubeam_ntpart.fcl)
# GEOM: optional geometry file name under beam/geom (e.g. move_extras12.txt), found through MU2E_SEARCH_PATH
# Optional env: G4REF_DATASET (MuBeamCat dataset, default MDC2025ac), G4REF_MUSING (default "SimJob MDC2025ac"),
# G4REF_NFILES (first N files of the sorted list; 0 = all), G4REF_SEARCH_PATH (prepended to MU2E_SEARCH_PATH),
# G4REF_EXTRA_FCL (a file of FCL lines appended to job.fcl)
OUT=$1; NEV=${2:-20000}; SEED=${3:-1}; GEOM=${4:-}
DATASET=${G4REF_DATASET:-sim.mu2e.MuBeamCat.MDC2025ac.art}; MUSING=${G4REF_MUSING:-SimJob MDC2025ac}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT" && cd "$OUT" || exit 2
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
muse setup ops
samweb list-definition-files "$DATASET" | sort > files.txt
[ "${G4REF_NFILES:-0}" -gt 0 ] && { head -n "$G4REF_NFILES" files.txt > files.tmp && mv files.tmp files.txt; }
while read f; do echo "$(samweb locate-file "$f" | head -1 | sed -E 's/^[a-z]+://; s/\(.*\)$//')/$f"; done < files.txt > mubeamcat.txt
muse setup $MUSING || exit 3
{ echo "#include \"g4ref_mubeam_ntpart.fcl\""
  echo "physics.filters.beamResampler.fileNames : [ $(sed 's/.*/"&"/' mubeamcat.txt | paste -sd,) ]"
  echo "services.SeedService.baseSeed : $SEED"
  echo "source.maxEvents : $NEV"
  [ -n "$GEOM" ] && echo "services.GeometryService.inputFile : \"$GEOM\""
  [ -n "${G4REF_EXTRA_FCL:-}" ] && cat "$G4REF_EXTRA_FCL"; } > job.fcl
export FHICL_FILE_PATH=$HERE/fcl:$FHICL_FILE_PATH MU2E_SEARCH_PATH=${G4REF_SEARCH_PATH:+$G4REF_SEARCH_PATH:}$HERE/geom:$MU2E_SEARCH_PATH
mu2e -c job.fcl --debug-config job_config.txt >/dev/null 2>&1
grep -E "GeometryService|bFieldFile|inputFile|physicsListName" job_config.txt | head > job_geom.txt
/usr/bin/time -v mu2e -c job.fcl > job.log 2>&1
echo "exit $?" >> job.log
