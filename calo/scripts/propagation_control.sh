#!/bin/bash
# Run one synthetic event through public stock propagation, never transport or production RNG.
set -eo pipefail
: "${CALO_CODE:?source root}"
: "${CALO_AREA:?verified built satellite}"
: "${CALO_OUT:?fresh control directory}"
: "${CALO_SIM_OFFLINE:?original geometry search root}"
: "${CALO_BACKING:?pinned backing}"
: "${CALO_ENVSET:?verified envset}"
: "${CALO_PLUGIN_SHA256:?verified propagation analyzer checksum}"
: "${CALO_TABLE_SHA256:?expected actual propagation file checksum}"
: "${CALO_STOCK_SOURCE_SHA256:?expected stock implementation checksum}"
test "$MUSE_WORK_DIR" = "$CALO_AREA"
test "$MUSE_ENVSET" = "$CALO_ENVSET"
test "$(readlink "$CALO_AREA/backing")" = "$CALO_BACKING"
plugin="$CALO_AREA/build/$MUSE_STUB/CaloFastSim/lib/libcalofastsim_CaloPropagationDumper_module.so"
test "$plugin" -nt "$CALO_AREA/CaloFastSim/src/CaloPropagationDumper_module.cc"
test "$plugin" -nt "$CALO_AREA/CaloFastSim/src/SConscript"
test "$plugin" -nt "$CALO_AREA/CaloFastSim/CMakeLists.txt"
echo "$CALO_PLUGIN_SHA256  $plugin" | sha256sum -c -
stock_source="$CALO_BACKING/Offline/CaloMC/src/CaloPhotonPropagation.cc"
echo "$CALO_STOCK_SOURCE_SHA256  $stock_source" | sha256sum -c -
mkdir "$CALO_OUT"
cd "$CALO_OUT"
export MU2E_SEARCH_PATH="$CALO_SIM_OFFLINE:$MU2E_SEARCH_PATH"
export FHICL_FILE_PATH="$CALO_CODE/calo/art:$FHICL_FILE_PATH"
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
table="$(resolve OfflineData/ConditionsService/CsI-propag-2020-07-29.root)"
echo "$CALO_TABLE_SHA256  $table" | sha256sum -c -
ldd -r "$plugin" > linkage.txt 2>&1
if grep -E 'not found|undefined symbol' linkage.txt; then exit 1; fi
stock_library="$(awk '$1 == "libmu2e_CaloMC.so" && $2 == "=>" {print $3}' linkage.txt)"
test -n "$stock_library"
test "$(readlink -f "$stock_library")" = "$(readlink -f "$CALO_BACKING/build/$MUSE_STUB/Offline/lib/libmu2e_CaloMC.so")"
sha256sum "$table" "$stock_source" "$CALO_BACKING/Offline/CaloMC/inc/CaloPhotonPropagation.hh" \
  "$stock_library" "$plugin" "$CALO_CODE/calo/art/CaloFastSim/src/CaloPropagationDumper_module.cc" \
  "$CALO_CODE/calo/art/CaloFastSim/src/SConscript" "$CALO_CODE/calo/art/CaloFastSim/CMakeLists.txt" > source_build_inputs.sha256
cp "$CALO_CODE/calo/art/CaloFastSim/fcl/propagation_control.fcl" executed.fcl
fhicl-dump -c executed.fcl > executed.expanded.fcl
mu2e --version > runtime.version.txt
printf 'host=%s\nbacking=%s\nenvset=%s\nstock_library=%s\ntable=%s\nrequested_events=1\nengine=prescribed_NonRandomEngine\nstart_utc=%s\n' \
  "$(hostname)" "$CALO_BACKING" "$CALO_ENVSET" "$stock_library" "$table" "$(date -u +%FT%TZ)" > receipt.txt
status=0
/usr/bin/time -v -o propagation.art.time mu2e -c executed.expanded.fcl -n 1 \
  -T "$PWD/calo_propagation_tables.root" > propagation.log 2>&1 || status=$?
printf 'exit_code=%s\nfinished_utc=%s\n' "$status" "$(date -u +%FT%TZ)" >> receipt.txt
grep '^Geom: ' propagation.log > geometry.resolved.txt || true
if test "$status" -eq 0; then
  sha256sum calo_propagation_tables.root executed.expanded.fcl geometry.resolved.txt > output.sha256
  printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >> receipt.txt
fi
exit "$status"
