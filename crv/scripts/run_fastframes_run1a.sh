#!/bin/bash
# F-fast deadtime frames for Run 1A CE+Mix1BB subruns (plan E4): per subrun, the mixed CrvSteps of the dig file
# (dump_crvsteps_mix.fcl) and the stock CRV conditions/pbtime of its mcs child (dump_crvreco_from_mcs_run1a.fcl), then
# crvfastsim.veto.fastframes; the step dump is deleted after use (1.4 GB per subrun). No build: the CrvFastSim art area
# must already be built. usage: run_fastframes_run1a.sh <subrun> [...]   (8-digit, e.g. 00000003)
# env: OUT (default $CRVFS_DATA/deadtime_run1a/ffast), CHUNK (500), KEEP_STEPS=1
D=${CRVFS_DATA:-/exp/mu2e/data/users/ram2aq/crv_fastsim}
C=${M2FS_CODE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim}
A=$C/crv/art
O=${OUT:-$D/deadtime_run1a/ffast}; mkdir -p "$O" || exit 2
V=$D/venv_eaf/bin/python
PHOTON=$D/deadtime_run1a/cond/CRVPhoton_cid196.txt    # dbTool print-content --cid 196 (Sim_best v1_3, runs 1430-1439)
L=/tmp/crvfs_lookup_ffast_$$; mkdir -p "$L" && chmod 700 "$L"
trap 'rm -rf "$L"' EXIT
for f in LookupTable_6000_0 photonMap.root; do cp /cvmfs/mu2e.opensciencegrid.org/DataFiles/CRVConditions/v6_0/$f "$L/" || exit 3; done
rc=0
for s in "$@"; do
  dig=dig.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_3.001430_$s.art
  mcs=mcs.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1.001430_$s.art
  nts=nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_$s.root
  paths=$( (source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1; muse setup ops >/dev/null 2>&1
            for f in $dig $mcs $nts; do echo "$(samweb locate-file $f | sed 's/^[a-z]*://; s/(.*//')/$f"; done) )
  read -r pdig pmcs pnts <<< "$(echo $paths)"
  [ -r "$pdig" ] && [ -r "$pmcs" ] && [ -r "$pnts" ] || { echo "subrun $s: missing input ($paths)"; rc=4; continue; }
  echo "== subrun $s $(date +%T)"
  ( source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1; cd "$A" && muse setup >/dev/null 2>&1; cd "$O"
    export FHICL_FILE_PATH=$A:$FHICL_FILE_PATH
    mu2e -c CrvFastSim/fcl/dump_crvsteps_mix.fcl -s "$pdig" -T "$O/mixsteps_$s.root" > "$O/dump_mix_$s.log" 2>&1 || exit 5
    mu2e -c CrvFastSim/fcl/dump_crvreco_from_mcs_run1a.fcl -s "$pmcs" -T "$O/crvreco_mcs_$s.root" > "$O/dump_mcs_$s.log" 2>&1 || exit 6
  ) || { echo "subrun $s: art dump failed (exit $?), see $O/dump_*_$s.log"; rc=5; rm -f "$O/mixsteps_$s.root"; continue; }
  ( cd "$C" && CRVFS_LOOKUP_DIR=$L PYTHONPATH=core:crv JAX_PLATFORMS=cpu \
      MU2E_SEARCH_PATH=/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_33_00 \
      "$V" -m crvfastsim.veto.fastframes "$O/mixsteps_$s.root" --conditions "$O/crvreco_mcs_$s.root" \
      --photon-dev "$PHOTON" --pbtime "$pnts" --chunk "${CHUNK:-500}" --out "$O/frames_fast_$s.parquet" ) \
      > "$O/frames_$s.log" 2>&1 || { echo "subrun $s: fastframes failed, see $O/frames_$s.log"; rc=7; }
  tail -1 "$O/frames_$s.log" | cut -c1-160
  [ -n "${KEEP_STEPS:-}" ] || rm -f "$O/mixsteps_$s.root"
  echo "$pnts" >> "$O/nts_used.txt"
done
echo "exit=$rc"; exit $rc
