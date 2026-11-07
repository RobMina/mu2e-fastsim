#!/bin/bash
# Export only the fixed training allocation, without building or sampling.
set -eo pipefail
: "${CALO_AREA:?verified built satellite}"
: "${CALO_COVERAGE_INVENTORY:?audited inventory}"
: "${CALO_COVERAGE_INVENTORY_SHA:?audited inventory digest}"
: "${CALO_COVERAGE_REFERENCE_CONFIG:?verified production config dump}"
: "${CALO_COVERAGE_OUT:?fresh coverage output base}"
: "${CALO_COVERAGE_PYTHON:?analysis python3 executable}"
: "${CALO_SIM_OFFLINE:?original simulation search root}"
test "$MUSE_WORK_DIR" = "$CALO_AREA"
test -x "$CALO_COVERAGE_PYTHON"
if pgrep -u "$USER" -x mu2e >/dev/null; then
  printf 'mu2e already active; refusing a second job\n' >&2
  exit 1
fi
export PYTHONPATH="$(dirname "$CALO_AREA")${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir "$CALO_COVERAGE_OUT"
"$CALO_COVERAGE_PYTHON" -m calofastsim.showers.training_audit inventory \
  --inventory "$CALO_COVERAGE_INVENTORY" --sha256 "$CALO_COVERAGE_INVENTORY_SHA" > "$CALO_COVERAGE_OUT/inputs.tsv"
sha256sum "$CALO_AREA/build/$MUSE_STUB/CaloFastSim/lib/libcalofastsim_CaloLibraryDumper_module.so" \
  "$CALO_AREA/CaloFastSim/src/CaloLibraryDumper_module.cc" > "$CALO_COVERAGE_OUT/runtime.sha256"
while IFS=$'\t' read -r name input count sha; do
  out="$CALO_COVERAGE_OUT/$name"
  mkdir "$out"
  sha256sum "$input" > "$out/input.before.sha256"
  test "$(cut -d ' ' -f 1 "$out/input.before.sha256")" = "$sha"
  config_dumper -P "$input" > "$out/source_config.txt"
  "$CALO_COVERAGE_PYTHON" -m calofastsim.showers.training_audit config \
    --reference "$CALO_COVERAGE_REFERENCE_CONFIG" --candidate "$out/source_config.txt" > "$out/geometry_config.json"
  CALO_INPUT="$input" CALO_EVENTS=-1 CALO_OUT="$out/export" CALO_EXPORT_SOURCE_EVENTS=1 \
    bash "$CALO_AREA/export_primary.sh"
  "$CALO_COVERAGE_PYTHON" -m calofastsim.showers.training_audit status \
    --tables "$out/export/calo_library_tables.root" --expected-count "$count" > "$out/status.json"
  "$CALO_COVERAGE_PYTHON" -m calofastsim.showers.production_check --art "$input" \
    --tables "$out/export/calo_library_tables.root" --check-geometry --out "$out/source_check.json" > "$out/source_check.txt"
  sha256sum "$input" > "$out/input.after.sha256"
  test "$(cut -d ' ' -f 1 "$out/input.after.sha256")" = "$sha"
  printf 'completed training %s expected_events=%s\n' "$name" "$count"
done < "$CALO_COVERAGE_OUT/inputs.tsv"
printf 'All ten training exports and raw CSS checks completed; saved-source binding and coverage report still required.\n'
