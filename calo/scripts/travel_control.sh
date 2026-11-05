#!/bin/bash
# Run only travel-enabled stock RO on saved CSS with the existing verified analyzer.
set -eo pipefail
: "${CALO_CODE:?source root}"
: "${CALO_AREA:?verified built satellite}"
: "${CALO_OUT:?fresh control directory}"
: "${CALO_INPUT:?existing original art input}"
: "${CALO_INPUT_SHA256:?expected input checksum}"
: "${CALO_SIM_OFFLINE:?original simulation search root}"
: "${CALO_BACKING:?pinned backing}"
: "${CALO_ENVSET:?verified envset}"
: "${CALO_PLUGIN_SHA256:?verified response analyzer checksum}"
: "${CALO_BASELINE:?saved deterministic ROOT}"
: "${CALO_BASELINE_SHA256:?saved baseline checksum}"
: "${CALO_BASELINE_FCL:?saved expanded baseline configuration}"
: "${CALO_BASELINE_FCL_SHA256:?saved expanded baseline checksum}"
: "${CALO_EVENTS:?explicit count bounded by pilot}"
test "$MUSE_WORK_DIR" = "$CALO_AREA"
test "$MUSE_ENVSET" = "$CALO_ENVSET"
test "$(readlink "$CALO_AREA/backing")" = "$CALO_BACKING"
case "$CALO_EVENTS" in ''|*[!0-9]*) exit 1;; esac
test "$CALO_EVENTS" -gt 0 && test "$CALO_EVENTS" -le 2000
plugin="$CALO_AREA/build/$MUSE_STUB/CaloFastSim/lib/libcalofastsim_CaloResponseDumper_module.so"
for source in src/CaloResponseDumper_module.cc src/SConscript CMakeLists.txt; do
  test "$plugin" -nt "$CALO_AREA/CaloFastSim/$source"
done
for specification in "$CALO_PLUGIN_SHA256  $plugin" "$CALO_INPUT_SHA256  $CALO_INPUT" \
                     "$CALO_BASELINE_SHA256  $CALO_BASELINE" "$CALO_BASELINE_FCL_SHA256  $CALO_BASELINE_FCL"; do
  echo "$specification" | sha256sum -c -
done
mkdir "$CALO_OUT"
cd "$CALO_OUT"
export MU2E_SEARCH_PATH="$CALO_SIM_OFFLINE:$MU2E_SEARCH_PATH"
export FHICL_FILE_PATH="$CALO_CODE/calo/art:$FHICL_FILE_PATH"
cp "$CALO_CODE/calo/art/CaloFastSim/fcl/travel_control.fcl" executed.fcl
fhicl-dump -c executed.fcl > executed.expanded.fcl
sed -E 's/(addTravelTime: )true/\1false/' executed.expanded.fcl | sed '/^#/d' > baseline_config_prediction.fcl
sed '/^#/d' "$CALO_BASELINE_FCL" > baseline_config_observed.fcl
cmp baseline_config_prediction.fcl baseline_config_observed.fcl
ldd -r "$plugin" > linkage.txt 2>&1
if grep -E 'not found|undefined symbol' linkage.txt; then exit 1; fi
mu2e --version > runtime.version.txt
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
cmp resolved_inputs.sha256 "$(dirname "$CALO_BASELINE_FCL")/resolved_inputs.sha256"
sha256sum executed.expanded.fcl "$plugin" "$CALO_CODE/calo/scripts/check_travel_control.py" \
  "$CALO_CODE/calo/calofastsim/travel.py" "$CALO_CODE/calo/scripts/travel_control.sh" \
  "$CALO_CODE/calo/art/CaloFastSim/src/CaloResponseDumper_module.cc" \
  "$CALO_BACKING/Offline/CaloMC/src/CaloShowerROMaker_module.cc" \
  "$CALO_BACKING/Offline/CaloMC/src/CaloPhotonPropagation.cc" > source_build_policy.sha256
printf 'host=%s\ninput_sha256=%s\nbaseline_sha256=%s\nrequested_events=%s\narm=travel_only_PEStat_off\nbacking=%s\nenvset=%s\nanalyzer_sha256=%s\nstart_utc=%s\n' \
  "$(hostname)" "$CALO_INPUT_SHA256" "$CALO_BASELINE_SHA256" "$CALO_EVENTS" "$CALO_BACKING" "$CALO_ENVSET" \
  "$CALO_PLUGIN_SHA256" "$(date -u +%FT%TZ)" > receipt.txt
status=0
/usr/bin/time -v -o travel.art.time mu2e -c executed.expanded.fcl -s "$CALO_INPUT" -n "$CALO_EVENTS" \
  -T "$PWD/calo_response_tables.root" > travel.log 2>&1 || status=$?
printf 'exit_code=%s\nfinished_utc=%s\n' "$status" "$(date -u +%FT%TZ)" >> receipt.txt
grep '^Geom: ' travel.log > geometry.resolved.txt || true
if test "$status" -eq 0; then
  sha256sum calo_response_tables.root executed.expanded.fcl geometry.resolved.txt > output.sha256
  printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >> receipt.txt
fi
exit "$status"
