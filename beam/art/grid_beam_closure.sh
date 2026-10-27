#!/usr/bin/env bash
# grid_beam_closure.sh OUTDIR : one beam-closure Geant4 job on a grid worker (published SimJob MDC2025ac, no build).
# The job is line PROCESS+1 of tasks.txt in the code tarball (CODE_TARBALL, staged on pnfs by submit_beam_closure.sh):
#   foilscan SCAN PDG [EVENTS [SEEDOFF]] MODES...  beamfastsim.foilscan campaign (G4 slab), audited on the worker
#   chain LABEL GUNFILE SKIP N SEED GEOM  g4ref_gun_ntpart.fcl on N muons of GUNFILE after SKIP (GEOM "-" = nominal)
#   tables LABEL GUNFILE BINS GEOM       one-event g4ref_gun_ntpart.fcl exporting the ASCII physics tables (full geometry)
#   arms LABEL G4REF START N GEOM ARMS... beamfastsim.convergence transport arms (JAX from the LCG_108 view, CPU)
# Outputs (tgz of the job directory) are copied to OUTDIR; a job whose output exists is skipped.
set -o pipefail
RC=0
OUTDIR="${1:?need OUTDIR (pnfs scratch)}"
echo "=== grid_beam_closure: PROCESS=${PROCESS:-<unset>} HOST=$(hostname) $(date -u)"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
setup ifdhc 2>/dev/null || true
which ifdh &>/dev/null || { echo "ERROR: no ifdh"; exit 2; }
ifdh cp "${CODE_TARBALL:?}" ./code.tgz && tar xzf ./code.tgz && rm -f ./code.tgz || { echo "STAGE code FAILED"; exit 2; }
TASK=$(sed -n "$((PROCESS + 1))p" ./code/tasks.txt)
[ -n "$TASK" ] || { echo "ERROR: no task at line $((PROCESS + 1))"; exit 1; }
read -r KIND A1 A2 A3 A4 A5 A6 REST <<< "$TASK"
echo "task: $TASK"
CODE="$PWD/code"
if [ "$KIND" = arms ]; then
  # the LCG view only inside the Python subshell, so ifdh keeps the Mu2e environment
  # shellcheck disable=SC1091
  pyrun() { ( source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh || exit 3
              env PYTHONPATH="$CODE/beam:$CODE/core:$PYTHONPATH" JAX_PLATFORMS=cpu python3 "$@" ); }
else
  muse setup SimJob MDC2025ac || { echo "ERROR: muse setup"; exit 3; }
  which mu2e &>/dev/null || { echo "ERROR: mu2e not on PATH"; exit 3; }
  PY=/cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0/bin/python3
  pyrun() { env -u PYTHONPATH PYTHONPATH="$CODE/beam:$CODE/core" JAX_PLATFORMS=cpu "$PY" "$@"; }
fi
mkdir -p work
check_output() {
  if ifdh ls "$OUTDIR/$TAG.ok" >/dev/null 2>&1 && ifdh ls "$OUTDIR/$TAG.tgz" >/dev/null 2>&1; then
    echo "SKIP: verified completion marker and archive for $TAG"; exit 0
  fi
  if ifdh ls "$OUTDIR/$TAG.tgz" >/dev/null 2>&1; then
    echo "ERROR: unqualified existing archive $OUTDIR/$TAG.tgz; audit it or choose a new output namespace"; exit 6
  fi
}
case "$KIND" in
  foilscan)
    PDGNAME=$([ "$A2" = "13" ] && echo mum || echo mup)
    EVENTS=(); SEEDOFF=0; MODES="$A3 $A4 $A5 $A6 $REST"
    if [[ "$A3" =~ ^[0-9]+$ ]]; then EVENTS=(--events "$A3"); MODES="$A4 $A5 $A6 $REST"; fi
    if [[ "$A3" =~ ^[0-9]+$ && "$A4" =~ ^[0-9]+$ ]]; then SEEDOFF="$A4"; MODES="$A5 $A6 $REST"; fi
    MODETAG=$(echo $MODES | tr ' ' '-')
    TAG="foilscan_${A1}_${MODETAG}_${PDGNAME}_r${SEEDOFF}"
    check_output
    # shellcheck disable=SC2086
    BEAMFS_DATA_ROOT="$PWD/work" pyrun -m beamfastsim.foilscan campaign --out "work/$TAG" --scans "$A1" \
      --charges "$A2" "${EVENTS[@]}" --seed-offset "$SEEDOFF" --modes $MODES --budget-seconds 12000 --drop-root || { echo "FOILSCAN FAILED"; exit 4; }
    ;;
  chain)
    TAG="chain_${A1}_$(printf %07d "$A3")"
    check_output
    mkdir -p "work/$TAG" && cd "work/$TAG" || exit 4
    ifdh cp "$A2" ./particles.txt || { echo "STAGE particles FAILED"; exit 2; }
    { echo 'bool fromG4BLFile.do = true;'
      echo 'string fromG4BLFile.filename = "./particles.txt";'
      echo 'double fromG4BLFile.mean = -1;'
      echo 'vector<double> fromG4BLFile.g4beamlineOrigin = {0., 0., 0.};'
      echo 'bool fromG4BLFile.targetFrame = false;'
      echo "int fromG4BLFile.particlesToSkip = $A3;"; } > beamgun.txt
    { echo '#include "g4ref_gun_ntpart.fcl"'
      echo "services.SeedService.baseSeed : $A5"
      echo "source.maxEvents : $A4"
      [ "$A6" != "-" ] && echo "services.GeometryService.inputFile : \"$A6\""; } > job.fcl
    export FHICL_FILE_PATH="$CODE/beam/fcl:$FHICL_FILE_PATH" MU2E_SEARCH_PATH="$PWD:$CODE/beam/geom:$MU2E_SEARCH_PATH"
    mu2e -c job.fcl --debug-config job_config.txt > expand.log 2>&1
    RC=$?
    if [ "$RC" = 0 ]; then
      /usr/bin/time -v mu2e -c job.fcl > job.log 2>&1
      RC=$?
    else
      cp expand.log job.log
    fi
    echo "exit $RC" >> job.log
    if [ "$RC" = 0 ] && ! grep -q "Art has completed and will exit with status 0" job.log; then RC=4; fi
    rm -f particles.txt
    cd ../.. || exit 4
    [ "$RC" = 0 ] || { echo "MU2E FAILED rc=$RC"; tail -40 "work/$TAG/job.log"; }
    ;;
  tables)
    TAG="tables_${A1}_b${A3}"
    check_output
    mkdir -p "work/$TAG/tables" && cd "work/$TAG" || exit 4
    ifdh cp "$A2" ./particles.txt || { echo "STAGE particles FAILED"; exit 2; }
    { echo 'bool fromG4BLFile.do = true;'
      echo 'string fromG4BLFile.filename = "./particles.txt";'
      echo 'double fromG4BLFile.mean = -1;'
      echo 'vector<double> fromG4BLFile.g4beamlineOrigin = {0., 0., 0.};'
      echo 'bool fromG4BLFile.targetFrame = false;'
      echo 'int fromG4BLFile.particlesToSkip = 0;'; } > beamgun.txt
    cat "$CODE/beam/fcl/export_physics_tables.mac" > job.mac
    printf '/process/em/binsPerDecade %s\n' "$A3" >> job.mac
    { echo '#include "g4ref_gun_ntpart.fcl"'
      echo "source.maxEvents : 1"
      echo "services.SeedService.baseSeed : 9002"
      echo 'physics.producers.g4run.g4Macro : "job.mac"'
      echo 'physics.producers.g4run.debug.storePhysicsTablesDir : "tables"'
      [ "$A4" != "-" ] && echo "services.GeometryService.inputFile : \"$A4\""; } > job.fcl
    export FHICL_FILE_PATH="$CODE/beam/fcl:$FHICL_FILE_PATH" MU2E_SEARCH_PATH="$PWD:$CODE/beam/geom:$MU2E_SEARCH_PATH"
    mu2e -c job.fcl --debug-config job_config.txt > expand.log 2>&1
    RC=$?
    if [ "$RC" = 0 ]; then
      /usr/bin/time -v mu2e -c job.fcl > job.log 2>&1
      RC=$?
    else
      cp expand.log job.log
    fi
    echo "exit $RC" >> job.log
    if [ "$RC" = 0 ] && ! grep -q "Art has completed and will exit with status 0" job.log; then RC=4; fi
    if [ "$RC" = 0 ]; then
      for charge in mu- mu+; do
        for name in Ionisation DEDX DEDXnr Range CSDARange; do
          [ -s "tables/$name.muIoni.$charge.asc" ] || RC=4
        done
      done
      [ -s tables/LambdaMod0.msc.mu+.asc ] || RC=4
      sha256sum tables/* > tables.sha256 || RC=4
    fi
    rm -f particles.txt ./*.root
    cd ../.. || exit 4
    [ "$RC" = 0 ] || { echo "MU2E FAILED rc=$RC"; tail -40 "work/$TAG/job.log"; }
    ;;
  arms)
    TAG="arms_${A1}"
    check_output
    ifdh cp "$A2" ./g4ref.npz || { echo "STAGE g4ref FAILED"; exit 2; }
    GEOMARG=(); [ "$A5" != "-" ] && GEOMARG=(--geom "$CODE/beam/geom/$A5")
    # shellcheck disable=SC2086
    pyrun -m beamfastsim.convergence --g4 ./g4ref.npz --out "work/$TAG" --start "$A3" --n "$A4" --seeds 11 \
      "${GEOMARG[@]}" --arms $A6 $REST --budget-seconds 12000 > "arms.log" 2>&1
    RC=$?; tail -30 arms.log; mv arms.log "work/$TAG/" 2>/dev/null
    rm -f ./g4ref.npz
    [ "$RC" = 0 ] || { echo "ARMS FAILED rc=$RC"; }
    [ -d "work/$TAG" ] || exit 4
    ;;
  *) echo "ERROR: unknown task kind $KIND"; exit 1 ;;
esac
[ "$RC" = 0 ] || { echo "FAILED rc=$RC; no success archive or marker published"; exit "$RC"; }
{ printf "task=%s\nexit=0\n" "$TASK"; sha256sum "$0"; } > "work/$TAG/completion.txt"
tar czf "$TAG.tgz" -C work "$TAG" || exit 5
ifdh mkdir_p "$OUTDIR" || exit 5
ifdh cp "$TAG.tgz" "$OUTDIR/$TAG.tgz" || { echo "COPY FAILED"; exit 5; }
sha256sum "$TAG.tgz" > "$TAG.ok"
ifdh cp "$TAG.ok" "$OUTDIR/$TAG.ok" || { echo "MARKER COPY FAILED"; exit 5; }
echo "=== grid_beam_closure: done -> $OUTDIR/$TAG.tgz $(date -u)"
