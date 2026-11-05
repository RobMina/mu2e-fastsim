#!/bin/bash
# Saved-only campaign gate: bind reused parent, complete stock census and original production log.
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -eo pipefail
: "${TRIGGER_ROOT:?}" "${TRIGGER_CODE:?}" "${TRIGGER_CONFIG:?}" "${TRIGGER_CAMPAIGN:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}" "${TRIGGER_ANALYSIS_SHA256:?}" "${TRIGGER_PYTHON:?}"
module load apptainer
printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
binds="$TRIGGER_ROOT,$(readlink -f "$TRIGGER_ROOT")"
analysis=(apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/injection_campaign.py" check-completed --plan "$TRIGGER_CAMPAIGN" --ordinals 1 --output "$TRIGGER_ROOT/provenance/campaign-parent1-reuse-v2.json"
"${analysis[@]}" "$TRIGGER_CODE/trigger/hybrid_history.py" --config "$TRIGGER_CONFIG" --regeneration-manifest "$TRIGGER_ROOT/prepared/regenerate_v3/manifest.json" --directory "$TRIGGER_ROOT/out/ce_regen_full" --comparison "$TRIGGER_ROOT/out/ce_regen_full/production-comparison-v2.json" --expected-events 28044 --output "$TRIGGER_ROOT/provenance/hybrid-stock-history-v2.json"
"${analysis[@]}" "$TRIGGER_CODE/trigger/production_history_log.py" --config "$TRIGGER_CONFIG" --regeneration-manifest "$TRIGGER_ROOT/prepared/regenerate_v3/manifest.json" --production-log "$TRIGGER_CODE/ce_ba8.production.log" --output "$TRIGGER_ROOT/provenance/production-log-history-v1.json"
