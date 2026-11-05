#!/bin/bash
# Export an existing art file with a verified user-built satellite; never transport.
set -eo pipefail
: "${CALO_CODE:?source root}"
: "${CALO_AREA:?verified built satellite}"
: "${CALO_OUT:?fresh export directory}"
: "${CALO_INPUT:?existing art input}"
: "${CALO_INPUT_SHA256:?expected input checksum}"
: "${CALO_SIM_OFFLINE:?original simulation Offline search root}"
: "${CALO_BACKING:?pinned exporter backing}"
: "${CALO_ENVSET:?verified exporter envset}"
: "${CALO_EVENTS:?explicit export event count}"
test "$MUSE_WORK_DIR" = "$CALO_AREA"
test "$MUSE_ENVSET" = "$CALO_ENVSET"
test "$(readlink "$CALO_AREA/backing")" = "$CALO_BACKING"
test -f "$CALO_SIM_OFFLINE/Offline/Mu2eG4/geom/geom_run1_a.txt"
test -f "$CALO_SIM_OFFLINE/Offline/Mu2eG4/geom/bfgeom_no_tsu_ps_v01.txt"
echo "$CALO_INPUT_SHA256  $CALO_INPUT" | sha256sum -c -
mkdir "$CALO_OUT"
cd "$CALO_OUT"
export MU2E_SEARCH_PATH="$CALO_SIM_OFFLINE:$MU2E_SEARCH_PATH"
export FHICL_FILE_PATH="$CALO_CODE/calo/art:$FHICL_FILE_PATH"
cp "$CALO_CODE/calo/art/CaloFastSim/fcl/dump_library.fcl" executed.fcl
mu2e --version > runtime.version.txt
fhicl-dump -c executed.fcl > executed.expanded.fcl
printf 'host=%s\ninput_sha256=%s\nrequested_events=%s\nexporter_backing=%s\nexporter_envset=%s\nsimulation_search_root=%s\nstart_utc=%s\n' \
  "$(hostname)" "$CALO_INPUT_SHA256" "$CALO_EVENTS" "$CALO_BACKING" "$CALO_ENVSET" "$CALO_SIM_OFFLINE" "$(date -u +%FT%TZ)" > receipt.txt
/usr/bin/time -v -o export.art.time mu2e -c executed.expanded.fcl -s "$CALO_INPUT" -n "$CALO_EVENTS" \
  -T "$PWD/calo_library_tables.root" > export.log 2>&1
grep '^Geom: ' export.log > geometry.resolved.txt
sha256sum geometry.resolved.txt executed.expanded.fcl "$CALO_AREA"/build/"$MUSE_STUB"/CaloFastSim/lib/*CaloLibraryDumper_module.so > provenance.sha256
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >> receipt.txt
