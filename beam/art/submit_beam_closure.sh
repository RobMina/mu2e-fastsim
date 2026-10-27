#!/usr/bin/env bash
# submit_beam_closure.sh STAGE_DIR TASKS [--test] [--submit] : stage code + task list to pnfs and submit one job per task.
#   STAGE_DIR  data-area directory (tarball and submit CWD; never the app dir or a mirrored tree)
#   TASKS      task list (one line per job, see grid_beam_closure.sh); chain GUNFILE paths must already be on pnfs
# Without --submit it prints the jobsub_submit command only. --test submits the first task alone.
# Run on a GPVM after: source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh; muse setup ops (valid grid token).
set -euo pipefail
STAGE="${1:?stage dir}"; TASKS="${2:?task list}"; shift 2
TEST=0; SUBMIT=0
for o in "$@"; do case "$o" in --test) TEST=1 ;; --submit) SUBMIT=1 ;; *) echo "unknown $o"; exit 1 ;; esac; done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"           # beam/
CORE="${BEAMFS_CORE:-$(cd "$HERE/../core" 2>/dev/null && pwd)}"   # core checkout holding mu2efastsim/
[ -d "$CORE/mu2efastsim" ] || { echo "ERROR: set BEAMFS_CORE to the core checkout"; exit 1; }
PNFS="${PNFS_OUT:-/pnfs/mu2e/scratch/users/ram2aq/beam_closure_20261026}"
OUTDIR="$PNFS/out"; TOKEN_DIR="$(echo "$PNFS" | sed 's|^/pnfs/mu2e|/mu2e|')"
case "$STAGE" in /exp/mu2e/data/users/ram2aq/*) ;; *) echo "ERROR: staging requires a data-area path"; exit 2 ;; esac
[ -f "$TASKS" ] || { echo "ERROR: no task list $TASKS"; exit 2; }
N=$(grep -cv '^[[:space:]]*$' "$TASKS")
[ "$N" -gt 0 ] || { echo "ERROR: empty task list"; exit 2; }
[ "$TEST" = 0 ] || N=1
TARBALL="beamcode_$(date -u +%Y%m%dT%H%M%S).tgz"
if [ "$SUBMIT" = 1 ]; then
  [ ! -e "$STAGE" ] || { echo "ERROR: use a fresh staging directory"; exit 2; }
  mkdir -p "$STAGE/code/beam" "$STAGE/code/core"
  cp -r "$HERE/beamfastsim" "$HERE/fcl" "$HERE/geom" "$STAGE/code/beam/"
  cp -r "$CORE/mu2efastsim" "$STAGE/code/core/"
  find "$STAGE/code" -name __pycache__ -prune -exec rm -rf {} +
  if [ "$TEST" = 1 ]; then
    awk 'NF {print; exit}' "$TASKS" > "$STAGE/code/tasks.txt"
  else
    awk 'NF' "$TASKS" > "$STAGE/code/tasks.txt"
  fi
  tar czf "$STAGE/$TARBALL" -C "$STAGE" code
  ifdh mkdir_p "$PNFS"
  ifdh cp "$STAGE/$TARBALL" "$PNFS/$TARBALL"
  cp "$HERE/art/grid_beam_closure.sh" "$STAGE/grid_beam_closure.sh"; chmod +x "$STAGE/grid_beam_closure.sh"
fi
CMD=(jobsub_submit -G mu2e --role=Analysis -N "$N" --expected-lifetime=4h
     --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --lines='+FERMIHTC_AutoRelease=True'
     --memory="${MEMORY:-4000MB}" --disk=10GB --need-storage-modify="$TOKEN_DIR"
     -e CODE_TARBALL="$PNFS/$TARBALL" "file://$STAGE/grid_beam_closure.sh" "$OUTDIR")
echo "tasks: $N  tarball: $PNFS/$TARBALL  out: $OUTDIR"
printf '%q ' "${CMD[@]}"; echo
if [ "$SUBMIT" = 1 ]; then cd "$STAGE" && "${CMD[@]}"; fi
