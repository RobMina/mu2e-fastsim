#!/bin/bash
# Provenance-only stock reads of explicitly frozen training/development inputs.
set -eo pipefail
: "${CALO_AREA:?verified built satellite}"
: "${CALO_AUDIT_INVENTORY:?validated source-audit inventory}"
: "${CALO_AUDIT_OUT:?fresh output directory}"
: "${CALO_AUDIT_PYTHON:?absolute ana python3 executable}"
: "${CALO_AUDIT_FILE0_LOG:?completed file0 stock-printer log}"
: "${CALO_AUDIT_FILE0_CONFIG:?saved file0 TargetStopResampler module config}"
test "$MUSE_WORK_DIR" = "$CALO_AREA"
test -x "$CALO_AUDIT_PYTHON"
export PYTHONPATH="$(dirname "$CALO_AREA")${PYTHONPATH:+:$PYTHONPATH}"
export FHICL_FILE_PATH="$CALO_AREA:$FHICL_FILE_PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir "$CALO_AUDIT_OUT"
"$CALO_AUDIT_PYTHON" -m calofastsim.showers.source_audit inventory --inventory "$CALO_AUDIT_INVENTORY" > "$CALO_AUDIT_OUT/inputs.tsv"
fhicl-dump -c "$CALO_AREA/CaloFastSim/fcl/print_primary_sources.fcl" > "$CALO_AUDIT_OUT/printer.fcl"
while IFS=$'\t' read -r role name input count sha; do
  out="$CALO_AUDIT_OUT/$name"
  mkdir "$out"
  printf 'role=%s\ninput=%s\ncount=%s\nsha256=%s\n' "$role" "$input" "$count" "$sha" > "$out/inputs.txt"
  "$CALO_AUDIT_PYTHON" -m calofastsim.showers.source_audit census --art "$input" --expected-count "$count" --sha256 "$sha" --out "$out/expected_events.npz" > "$out/census.json"
  if [ "$name" = dts.mu2e.CeEndpoint.MDC2025ba.001430_00000000.art ]; then
    log="$CALO_AUDIT_FILE0_LOG"
    config="$CALO_AUDIT_FILE0_CONFIG"
  else
    config="$out/source_config.txt"
    config_dumper -M -f TargetStopResampler "$input" > "$config"
    "$CALO_AUDIT_PYTHON" -m calofastsim.showers.source_audit config --config "$config" > "$out/config_check.json"
    log="$out/printer.log"
    /usr/bin/time -v -o "$out/printer.time" mu2e -c "$CALO_AUDIT_OUT/printer.fcl" -s "$input" -n -1 > "$log" 2>&1
  fi
  "$CALO_AUDIT_PYTHON" -m calofastsim.showers.source_audit bind --inventory "$CALO_AUDIT_INVENTORY" --item "$name" --expected-events "$out/expected_events.npz" --printer-log "$log" --config "$config" --census-receipt "$out/census.json" --out "$out/bound" > "$out/bind.txt"
  printf 'completed %s %s\n' "$role" "$name"
done < "$CALO_AUDIT_OUT/inputs.tsv"
printf 'All allocated training/development provenance reads completed.\n'
