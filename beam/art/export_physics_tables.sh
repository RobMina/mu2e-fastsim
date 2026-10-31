#!/bin/bash
# Export release physics tables from one saved beam-reference event without rebuilding or replacing the reference.
set -eo pipefail
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: export_physics_tables.sh SAVED_REFERENCE_DIR NEW_OUTPUT_DIR [BINS_PER_DECADE]" >&2
    exit 2
fi
SCRIPT=$(realpath "$0")
REF=$(realpath "$1")
OUT=$(realpath -m "$2")
HERE=$(cd "$(dirname "$SCRIPT")/.." && pwd)
if grep -E '^[[:space:]]*g4Macro[[:space:]]*:' "$REF/job_config.txt" | grep -Ev '^[[:space:]]*g4Macro:[[:space:]]*""[[:space:]]*$'; then
    echo "Refusing to replace a nonempty or unrecognized saved G4 macro" >&2
    exit 2
fi
BINS=${3:-}
if [ -n "$BINS" ] && ! [[ "$BINS" =~ ^[1-9][0-9]*$ ]]; then
    echo "BINS_PER_DECADE must be a positive integer" >&2
    exit 2
fi
: "${MUSE_WORK_DIR:?Set up the published SimJob MDC2025ac release first}"
if [ "$MUSE_WORK_DIR" != /cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ac ]; then
    echo "Refusing unexpected release: $MUSE_WORK_DIR" >&2
    exit 2
fi
DATA=$(realpath -m "${BEAMFS_DATA_ROOT:?set BEAMFS_DATA_ROOT to your beam_fastsim data area (site.env)}")
case "$OUT" in
    "$DATA"/*) ;;
    *) echo "Output must be in the beam_fastsim data area $DATA (BEAMFS_DATA_ROOT)" >&2; exit 2 ;;
esac
if [ -e "$OUT" ]; then
    echo "Output already exists: $OUT" >&2
    exit 2
fi
mkdir -p "$OUT/tables"
cd "$OUT"
export FHICL_FILE_PATH="$REF:$HERE/fcl:$FHICL_FILE_PATH"
export MU2E_SEARCH_PATH="$OUT:$HERE/fcl:$HERE/geom:$MU2E_SEARCH_PATH"
cat "$HERE/fcl/export_physics_tables.mac" > job.mac
if [ -n "$BINS" ]; then
    printf '/process/em/binsPerDecade %s\n' "$BINS" >> job.mac
fi
cat > export.fcl <<'FCL'
#include "job_config.txt"
source.maxEvents: 1
physics.producers.g4run.g4Macro: "job.mac"
physics.producers.g4run.debug.storePhysicsTablesDir: "tables"
services.TFileService.fileName: "table_export_ntpart.root"
FCL
{
    printf 'MUSE_WORK_DIR=%s\nMUSE_STUB=%s\n' "$MUSE_WORK_DIR" "${MUSE_STUB:-}"
    command -v mu2e
    sha256sum "$REF/job_config.txt" "$REF/job.log" "$HERE/fcl/export_physics_tables.mac" "$SCRIPT" job.mac export.fcl
} > provenance.txt
mu2e -c export.fcl --debug-config expanded.fcl
/usr/bin/time -v mu2e -c export.fcl > job.log 2>&1
printf 'exit=0\n' >> job.log
du -sh tables
