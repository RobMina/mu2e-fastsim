#!/bin/bash
# Run the frozen owner CE transport on an audited selected parent; no Geant4 or reco.
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:30:00
set -eo pipefail
: "${TRIGGER_CODE:?}"
: "${TRIGGER_OUTPUT:?}"
: "${TRIGGER_SOURCE:?}"
: "${TRIGGER_TABLES:?}"
: "${TRIGGER_GEOMETRY_ROOT:?}"
: "${TRIGGER_GDML:?}"
: "${TRIGGER_DS_MAP:?}"
: "${TRIGGER_TSD_MAP:?}"
: "${TRIGGER_TRACKER:?}"
: "${TRIGGER_CORE:?}"
: "${TRIGGER_BEAM:?}"
: "${TRIGGER_EXPECTED_EVENTS:?}"
: "${TRIGGER_TRANSPORT_SEED:?}"
: "${TRIGGER_ANALYSIS_IMAGE:?}"
: "${TRIGGER_PYTHON:?}"
[[ $TRIGGER_EXPECTED_EVENTS =~ ^[1-9][0-9]{0,5}$ && $TRIGGER_TRANSPORT_SEED =~ ^[1-9][0-9]{0,8}$ ]] || exit 2
[[ ! -e $TRIGGER_OUTPUT ]] || exit 2
module load apptainer
if [[ -n ${TRIGGER_ANALYSIS_SHA256:-} ]]; then
  printf '%s  %s\n' "$TRIGGER_ANALYSIS_SHA256" "$TRIGGER_ANALYSIS_IMAGE" | sha256sum -c -
fi
mkdir -p "$TRIGGER_OUTPUT"
export APPTAINERENV_PYTHONPATH="$TRIGGER_CORE:$TRIGGER_TRACKER:$TRIGGER_BEAM"
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_JAX_PLATFORMS=cpu
binds="$TRIGGER_CODE,$TRIGGER_OUTPUT,$TRIGGER_TRACKER,$TRIGGER_CORE,$TRIGGER_BEAM,$TRIGGER_GEOMETRY_ROOT,$(dirname "$TRIGGER_SOURCE"),$(dirname "$TRIGGER_TABLES"),$(dirname "$TRIGGER_GDML"),$(dirname "$TRIGGER_DS_MAP"),$(dirname "$TRIGGER_TSD_MAP")"
[[ -z ${TRIGGER_EXTRA_TABLES:-} ]] || binds+=",$(dirname "$TRIGGER_EXTRA_TABLES")"
[[ -z ${TRIGGER_SOURCE_MANIFEST:-} ]] || binds+=",$(dirname "$TRIGGER_SOURCE_MANIFEST")"
analysis=(apptainer exec --cleanenv -B "$binds" "$TRIGGER_ANALYSIS_IMAGE" "$TRIGGER_PYTHON")
sha256sum "$TRIGGER_TRACKER/trkfastsim/validation/fieldce.py" "$TRIGGER_TRACKER/trkfastsim/sim/fieldsteps.py" "$TRIGGER_BEAM/beamfastsim/cechain.py" "$TRIGGER_GDML" "$TRIGGER_DS_MAP" "$TRIGGER_TSD_MAP" > "$TRIGGER_OUTPUT/source.sha256"
"${analysis[@]}" -m trkfastsim.validation.cegeometry --chain-root "$TRIGGER_GEOMETRY_ROOT" --stock-root "$TRIGGER_GEOMETRY_ROOT" --chain-geom Offline/Mu2eG4/geom/geom_run1_a.txt --stock-geom Offline/Mu2eG4/geom/geom_run1_a.txt --outdir "$TRIGGER_OUTPUT/geometry"
tables=("$TRIGGER_TABLES")
if [[ -n ${TRIGGER_EXTRA_TABLES:-} ]]; then tables+=("$TRIGGER_EXTRA_TABLES"); fi
"${analysis[@]}" -m trkfastsim.validation.fieldce "${tables[@]}" --mode chain --seed "$TRIGGER_TRANSPORT_SEED" --outdir "$TRIGGER_OUTPUT/transport" --reference-selection selected-dts --gdml "$TRIGGER_GDML" --maps "$TRIGGER_DS_MAP" "$TRIGGER_TSD_MAP" --offline-root "$TRIGGER_GEOMETRY_ROOT" --geom Offline/Mu2eG4/geom/geom_run1_a.txt
sources=("$TRIGGER_SOURCE")
if [[ -n ${TRIGGER_SOURCE_MANIFEST:-} ]]; then mapfile -t sources < "$TRIGGER_SOURCE_MANIFEST"; fi
"${analysis[@]}" "$TRIGGER_CODE/trigger/step_table_audit.py" --source "${sources[@]}" --states "$TRIGGER_OUTPUT/transport/states.root" --table "$TRIGGER_OUTPUT/transport/fasttrk_field.root" --handoff "$TRIGGER_OUTPUT/transport/handoff.json" --expected-events "$TRIGGER_EXPECTED_EVENTS" --output "$TRIGGER_OUTPUT/step-admission.json"
