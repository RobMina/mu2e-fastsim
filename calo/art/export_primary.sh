#!/bin/bash
# Read existing production art with a verified user-built exporter; never transport.
set -eo pipefail
: "${CALO_AREA:?verified built satellite}"
: "${CALO_INPUT:?existing development art input}"
: "${CALO_OUT:?fresh output directory}"
: "${CALO_SIM_OFFLINE:?original simulation search root containing Offline}"
: "${CALO_EVENTS:?explicit event limit, -1 for the complete file}"
test "$MUSE_WORK_DIR" = "$CALO_AREA"
test -f "$CALO_SIM_OFFLINE/Offline/Mu2eG4/geom/geom_run1_a.txt"
test -f "$CALO_SIM_OFFLINE/Offline/Mu2eG4/geom/bfgeom_no_tsu_ps_v01.txt"
PLUGIN="$CALO_AREA/build/$MUSE_STUB/CaloFastSim/lib/libcalofastsim_CaloLibraryDumper_module.so"
test -f "$PLUGIN"
test "$PLUGIN" -nt "$CALO_AREA/CaloFastSim/src/CaloLibraryDumper_module.cc"
case "${CALO_EXPORT_SOURCE_EVENTS:-0}" in
  0) EXPORT_FCL=dump_library_primary.fcl ;;
  1) EXPORT_FCL=dump_library_primary_sources.fcl ;;
  *) printf 'CALO_EXPORT_SOURCE_EVENTS must be 0 or 1\n' >&2; exit 2 ;;
esac
mkdir "$CALO_OUT"
cd "$CALO_OUT"
export MU2E_SEARCH_PATH="$CALO_SIM_OFFLINE:$MU2E_SEARCH_PATH"
export FHICL_FILE_PATH="$CALO_AREA:$FHICL_FILE_PATH"
printf 'input=%s\nevent_limit=%s\nsimulation_search_root=%s\narea=%s\nenvset=%s\n' \
  "$CALO_INPUT" "$CALO_EVENTS" "$CALO_SIM_OFFLINE" "$CALO_AREA" "$MUSE_ENVSET" > inputs.txt
mu2e --version > runtime.txt
fhicl-dump -c "$CALO_AREA/CaloFastSim/fcl/$EXPORT_FCL" > executed.fcl
/usr/bin/time -v -o export.time mu2e -c executed.fcl -s "$CALO_INPUT" -n "$CALO_EVENTS" \
  -T "$PWD/calo_library_tables.root" > export.log 2>&1
grep '^Geom: ' export.log > geometry.resolved.txt
printf 'Export completed.\n'
