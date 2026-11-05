#!/bin/bash
# Run deterministic stock response on saved shower steps with a verified user-built analyzer.
set -eo pipefail
: "${CALO_CODE:?source root}"
: "${CALO_AREA:?verified built satellite}"
: "${CALO_OUT:?fresh control directory}"
: "${CALO_INPUT:?existing art input}"
: "${CALO_INPUT_SHA256:?expected input checksum}"
: "${CALO_SIM_OFFLINE:?original simulation search root}"
: "${CALO_BACKING:?pinned response backing}"
: "${CALO_ENVSET:?verified envset}"
: "${CALO_PLUGIN_SHA256:?verified new analyzer checksum}"
: "${CALO_EVENTS:?explicit event count}"
test "$MUSE_WORK_DIR" = "$CALO_AREA"
test "$MUSE_ENVSET" = "$CALO_ENVSET"
test "$(readlink "$CALO_AREA/backing")" = "$CALO_BACKING"
case "$CALO_EVENTS" in ''|*[!0-9]*) exit 1;; esac
test "$CALO_EVENTS" -gt 0 && test "$CALO_EVENTS" -le 2000
plugin="$CALO_AREA/build/$MUSE_STUB/CaloFastSim/lib/libcalofastsim_CaloResponseDumper_module.so"
test "$plugin" -nt "$CALO_AREA/CaloFastSim/src/CaloResponseDumper_module.cc"
test "$plugin" -nt "$CALO_AREA/CaloFastSim/src/SConscript"
test "$plugin" -nt "$CALO_AREA/CaloFastSim/CMakeLists.txt"
echo "$CALO_PLUGIN_SHA256  $plugin" | sha256sum -c -
echo "$CALO_INPUT_SHA256  $CALO_INPUT" | sha256sum -c -
mkdir "$CALO_OUT"
cd "$CALO_OUT"
export MU2E_SEARCH_PATH="$CALO_SIM_OFFLINE:$MU2E_SEARCH_PATH"
export FHICL_FILE_PATH="$CALO_CODE/calo/art:$FHICL_FILE_PATH"
cp "$CALO_CODE/calo/art/CaloFastSim/fcl/response_control.fcl" executed.fcl
fhicl-dump -c executed.fcl > executed.expanded.fcl
mu2e --version > runtime.version.txt
printf 'host=%s\ninput_sha256=%s\nrequested_events=%s\nresponse_backing=%s\nresponse_envset=%s\nanalyzer_sha256=%s\nsimulation_search_root=%s\nstart_utc=%s\n' \
  "$(hostname)" "$CALO_INPUT_SHA256" "$CALO_EVENTS" "$CALO_BACKING" "$CALO_ENVSET" "$CALO_PLUGIN_SHA256" "$CALO_SIM_OFFLINE" "$(date -u +%FT%TZ)" > receipt.txt
resolve() {
  local entry
  local -a directories
  IFS=: read -r -a directories <<< "$MU2E_SEARCH_PATH"
  for entry in "${directories[@]}"; do
    if test -f "${entry:-.}/$1"; then printf '%s\n' "${entry:-.}/$1"; return 0; fi
  done
  echo "cannot resolve $1" >&2
  return 1
}
: > resolved_inputs.sha256
for name in Offline/CaloConditions/data/Sim_crystal.txt \
            Offline/GlobalConstantsService/data/globalConstants_01.txt \
            Offline/GlobalConstantsService/data/ParticleList.txt \
            Offline/Mu2eG4/geom/geom_run1_a.txt \
            Offline/Mu2eG4/geom/bfgeom_no_tsu_ps_v01.txt \
            OfflineData/ConditionsService/CsI-propag-2020-07-29.root; do
  path="$(resolve "$name")"
  sha256sum "$path" >> resolved_inputs.sha256
done
printf 'propagation_histogram=h_zposTime\n' >> receipt.txt
sha256sum executed.expanded.fcl "$plugin" "$CALO_CODE/calo/art/CaloFastSim/src/CaloResponseDumper_module.cc" \
  "$CALO_BACKING/Offline/CaloMC/src/CaloShowerROMaker_module.cc" \
  "$CALO_BACKING/Offline/CommonMC/src/EventWindowMarkerProducer_module.cc" \
  "$CALO_BACKING/Offline/CaloMC/src/CaloPhotonPropagation.cc" > source_build.sha256
status=0
/usr/bin/time -v -o response.art.time mu2e -c executed.expanded.fcl -s "$CALO_INPUT" -n "$CALO_EVENTS" \
  -T "$PWD/calo_response_tables.root" > response.log 2>&1 || status=$?
printf 'exit_code=%s\nfinished_utc=%s\n' "$status" "$(date -u +%FT%TZ)" >> receipt.txt
grep '^Geom: ' response.log > geometry.resolved.txt || true
grep '^GlobalConstants: ' response.log > global_constants.resolved.txt || true
if test "$status" -eq 0; then
  sha256sum calo_response_tables.root geometry.resolved.txt global_constants.resolved.txt > output.sha256
  printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >> receipt.txt
fi
exit "$status"
