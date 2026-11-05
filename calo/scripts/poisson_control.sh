#!/bin/bash
# Run only PE-statistics-enabled stock RO on saved CSS with the existing verified analyzer.
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
: "${CALO_PRODUCER_SHA256:?verified stock producer checksum}"
: "${CALO_CLHEP_SHA256:?verified linked CLHEP checksum}"
: "${CALO_CLHEP_RANDOM_SHA256:?verified linked CLHEP Random checksum}"
: "${CALO_CLHEP_SOURCE:?reviewed CLHEP source root}"
: "${CALO_CLHEP_INCLUDE:?actual installed CLHEP include root}"
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
cp "$CALO_CODE/calo/art/CaloFastSim/fcl/poisson_control.fcl" executed.fcl
fhicl-dump -c executed.fcl > executed.expanded.fcl
sed -E 's/(PEStatCorrection: )true/\1false/' executed.expanded.fcl | sed '/^#/d' > baseline_config_prediction.fcl
sed '/^#/d' "$CALO_BASELINE_FCL" > baseline_config_observed.fcl
cmp baseline_config_prediction.fcl baseline_config_observed.fcl
ldd -r "$plugin" > linkage.txt 2>&1
if grep -E 'not found|undefined symbol' linkage.txt; then exit 1; fi
producer="$CALO_BACKING/build/$MUSE_STUB/Offline/lib/libmu2e_CaloMC_CaloShowerROMaker_module.so"
IFS=: read -r -a plugin_dirs <<< "$CET_PLUGIN_PATH"
resolved_producer=""
for directory in "${plugin_dirs[@]}"; do
  if test -f "$directory/$(basename "$producer")"; then
    resolved_producer="$(readlink -f "$directory/$(basename "$producer")")"
    break
  fi
done
test "$resolved_producer" = "$(readlink -f "$producer")"
ldd -r "$producer" > producer.linkage.txt 2>&1
if grep -E 'not found|undefined symbol' producer.linkage.txt; then exit 1; fi
clhep="$(awk '$1 == "libCLHEP-2.4.7.1.so" {print $3}' producer.linkage.txt)"
clhep_random="$(awk '$1 == "libCLHEP-Random-2.4.7.1.so" {print $3}' producer.linkage.txt)"
test -f "$clhep" && test -f "$clhep_random"
for specification in "$CALO_PRODUCER_SHA256  $producer" "$CALO_CLHEP_SHA256  $clhep" \
                     "$CALO_CLHEP_RANDOM_SHA256  $clhep_random"; do
  echo "$specification" | sha256sum -c -
done
readelf -d "$producer" > producer.dynamic.txt
sha256sum "$producer" "$clhep" "$clhep_random" > sampler_provenance.sha256
for name in RandPoissonQ RandGaussQ MixMaxRng; do
  sha256sum "$CALO_CLHEP_SOURCE/src/$name.cc" "$CALO_CLHEP_SOURCE/Random/$name.h" \
            "$CALO_CLHEP_INCLUDE/$name.h" >> sampler_provenance.sha256
done
for name in poissonTables.cdat gaussQTables.cdat; do
  sha256sum "$CALO_CLHEP_SOURCE/$name" >> sampler_provenance.sha256
done
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
sha256sum executed.expanded.fcl "$plugin" "$CALO_CODE/calo/scripts/check_poisson_control.py" \
  "$CALO_CODE/calo/calofastsim/poisson.py" "$CALO_CODE/calo/calofastsim/poisson_stats.py" \
  "$CALO_CODE/calo/POISSON_CONTROL.md" "$CALO_CODE/calo/scripts/poisson_control.sh" \
  "$CALO_CODE/calo/art/CaloFastSim/src/CaloResponseDumper_module.cc" \
  "$CALO_BACKING/Offline/CaloMC/src/CaloShowerROMaker_module.cc" \
  "$CALO_BACKING/Offline/CaloMC/src/CaloPhotonPropagation.cc" > source_build_policy.sha256
printf 'host=%s\ninput_sha256=%s\nbaseline_sha256=%s\nrequested_events=%s\narm=PEStat_only_travel_off\nbacking=%s\nenvset=%s\nanalyzer_sha256=%s\nstart_utc=%s\n' \
  "$(hostname)" "$CALO_INPUT_SHA256" "$CALO_BASELINE_SHA256" "$CALO_EVENTS" "$CALO_BACKING" "$CALO_ENVSET" \
  "$CALO_PLUGIN_SHA256" "$(date -u +%FT%TZ)" > receipt.txt
status=0
/usr/bin/time -v -o poisson.art.time mu2e -c executed.expanded.fcl -s "$CALO_INPUT" -n "$CALO_EVENTS" \
  -T "$PWD/calo_response_tables.root" > poisson.log 2>&1 || status=$?
printf 'exit_code=%s\nfinished_utc=%s\n' "$status" "$(date -u +%FT%TZ)" >> receipt.txt
grep '^Geom: ' poisson.log > geometry.resolved.txt || true
if test "$status" -eq 0; then
  sha256sum calo_response_tables.root executed.expanded.fcl geometry.resolved.txt > output.sha256
  printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >> receipt.txt
fi
exit "$status"
