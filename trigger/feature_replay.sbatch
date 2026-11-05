#!/bin/bash
# Run a separately declared full-history feature experiment against a saved hybrid.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -eo pipefail
: "${TRIGGER_ROOT:?}" "${TRIGGER_CODE:?}" "${TRIGGER_CONFIG:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}" "${TRIGGER_ANALYSIS_SHA256:?}" "${TRIGGER_PYTHON:?}"
: "${TRIGGER_AREA:?}" "${OFFLINE_IMAGE:?}" "${CVMFSEXEC_DIST:?}"
: "${TRIGGER_FEATURE_BUNDLE:?}" "${TRIGGER_FEATURE_LIBRARY_SHA256:?}" "${TRIGGER_FEATURE_MANIFEST_SHA256:?}"
: "${TRIGGER_BASELINE_PREPARED:?}" "${TRIGGER_BASELINE_ART:?}" "${TRIGGER_BASELINE_COMPARISON:?}"
: "${TRIGGER_EXPECTED_EVENTS:?}" "${TRIGGER_PREPARED:?}" "${TRIGGER_OUTPUT:?}"
: "${TRIGGER_EXPORTER_SHA256:?}"
[[ $TRIGGER_EXPECTED_EVENTS == 28044 && ! -e $TRIGGER_PREPARED && ! -e $TRIGGER_OUTPUT ]] || exit 2
[[ ${OFFLINE_WARM_REPEAT:-0} == 0 && -z ${OFFLINE_OUTPUT_FILE:-} ]] || exit 2
module load apptainer
printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
binds="$TRIGGER_ROOT,$(readlink -f "$TRIGGER_ROOT")"
analysis=(apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/feature_runtime.py" verify-deployment --bundle "$TRIGGER_FEATURE_BUNDLE" --expected-library-sha256 "$TRIGGER_FEATURE_LIBRARY_SHA256" --expected-manifest-sha256 "$TRIGGER_FEATURE_MANIFEST_SHA256"
common=(--config "$TRIGGER_CONFIG" --exporter-source "$TRIGGER_CODE/trigger/art/R7Trigger/src/KalSeedSelectionFeatures_module.cc" --exporter-sha256 "$TRIGGER_EXPORTER_SHA256" --output "$TRIGGER_PREPARED" --expected-events "$TRIGGER_EXPECTED_EVENTS" --runtime-area "$TRIGGER_AREA" --runtime-image "$OFFLINE_IMAGE")
"${analysis[@]}" "$TRIGGER_CODE/trigger/feature_replay.py" prepare "${common[@]}" --baseline-prepared "$TRIGGER_BASELINE_PREPARED" --baseline-art "$TRIGGER_BASELINE_ART" --baseline-comparison "$TRIGGER_BASELINE_COMPARISON"
"${analysis[@]}" "$TRIGGER_CODE/trigger/feature_replay.py" preflight "${common[@]}"
mkdir -p "$TRIGGER_OUTPUT"
cd "$TRIGGER_OUTPUT"
export OFFLINE_EVENTS="$TRIGGER_EXPECTED_EVENTS" OFFLINE_LOG_PREFIX=replay OFFLINE_BINDS="$binds"
set +e
bash "$TRIGGER_FEATURE_BUNDLE/offline_features.sh" "$TRIGGER_AREA" "$TRIGGER_PREPARED/replay.fcl" > replay.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > art.exit
[[ $status == 0 ]] || { tail -80 replay.log; exit "$status"; }
"${analysis[@]}" "$TRIGGER_CODE/trigger/feature_validate.py" --config "$TRIGGER_CONFIG" --prepared "$TRIGGER_PREPARED" --art decisions.art --table r7-selection-features.root --log replay.log --bundle "$TRIGGER_FEATURE_BUNDLE" --expected-library-sha256 "$TRIGGER_FEATURE_LIBRARY_SHA256" --expected-manifest-sha256 "$TRIGGER_FEATURE_MANIFEST_SHA256" --output comparison.json
