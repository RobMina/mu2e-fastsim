#!/bin/bash
# Audit saved stock history and reused first-parent injection; no new art, transport or simulation.
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -eo pipefail
: "${TRIGGER_ROOT:?}" "${TRIGGER_CODE:?}" "${TRIGGER_CAMPAIGN:?}"
: "${TRIGGER_CONFIG:?}" "${TRIGGER_CAMPAIGN_RUNTIME:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}" "${TRIGGER_ANALYSIS_SHA256:?}" "${TRIGGER_PYTHON:?}"
module load apptainer
printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
binds="$TRIGGER_ROOT,$(readlink -f "$TRIGGER_ROOT")"
analysis=(apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
"${analysis[@]}" "$TRIGGER_CODE/trigger/injection_campaign.py" create --sources "$TRIGGER_ROOT/prepared/source_tables_full_v1/sources.txt" --counts "$TRIGGER_ROOT/prepared/source_tables_full_v1/event-counts.txt" --config "$TRIGGER_CONFIG" --runtime "$TRIGGER_CAMPAIGN_RUNTIME" --seed 20261104 --first-transport "$TRIGGER_ROOT/out/ce_transport_parent1_v1" --transport-root "$TRIGGER_ROOT/out/ce_transport_full_v2" --first-injection "$TRIGGER_ROOT/out/inject_ba55_parent1_v1" --injection-root "$TRIGGER_ROOT/out/inject_ba55_full_v1" --first-prepared "$TRIGGER_ROOT/prepared/inject_ba55_parent1_v1" --prepared-root "$TRIGGER_ROOT/prepared/inject_ba55_full_v1" --output "$TRIGGER_CAMPAIGN"
"${analysis[@]}" "$TRIGGER_CODE/trigger/injection_campaign.py" check-completed --plan "$TRIGGER_CAMPAIGN" --ordinals 1 --output "$TRIGGER_ROOT/provenance/campaign-parent1-reuse-v1.json"
"${analysis[@]}" "$TRIGGER_CODE/trigger/hybrid_history.py" --config "$TRIGGER_CONFIG" --regeneration-manifest "$TRIGGER_ROOT/prepared/regenerate_v3/manifest.json" --directory "$TRIGGER_ROOT/out/ce_regen_full" --comparison "$TRIGGER_ROOT/out/ce_regen_full/production-comparison-v2.json" --expected-events 28044 --output "$TRIGGER_ROOT/provenance/hybrid-stock-history-v1.json"
