#!/bin/bash
# run_reco_checks.sh <pulses|chain> -- the crvfastsim.reco checks on the CrvRecoTableDumper references, one host.
#   pulses : vr2stock (stock MakeCrvRecoPulses via PyROOT, NSTOCK events) then vr2 (all events), on MC and data
#   chain  : vr1 (noise off), direct, cost on the MC reference
# Needs site.env (CRVFS_CODE, CRVFS_DATA) and CRVFS_STOCK_AREA = a built muse area with libmu2e_CRVReco.so (for vr2stock).
# Outputs: $CRVFS_DATA/reco_ref/checks/<cmd>_<sample>/. No `set -u` before the setup (setupmu2e-art.sh).
WHAT=${1:?usage: run_reco_checks.sh <pulses|chain>}
NSTOCK=${NSTOCK:-5000}
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${CRVFS_STOCK_AREA:?set CRVFS_STOCK_AREA}" && muse setup >/dev/null 2>&1 || true
pyenv ana >/dev/null 2>&1 || true
cd "${CRVFS_CODE:?source site.env}" || exit 2
R=${CRVFS_DATA:?source site.env}/reco_ref
MC=$R/crvreco_mc.001400_00014516.root
DATA=$R/crvreco_data.124155_000001.root
O=$R/checks
mkdir -p "$O"
run() {  # name cmd input [args]
  local name=$1 cmd=$2 in=$3; shift 3
  local t0=$SECONDS
  python3 -m crvfastsim "reco-$cmd" "$in" --outdir "$O/$name" "$@" >| "$O/$name.txt" 2>&1
  echo "$name exit=$? wall=$((SECONDS-t0))s"
}
case "$WHAT" in
  pulses)
    run vr2stock_mc vr2stock "$MC" --max-events "$NSTOCK"
    run vr2stock_data vr2stock "$DATA" --max-events "$NSTOCK"
    run vr2_mc vr2 "$MC"
    run vr2_data vr2 "$DATA" ;;
  chain)
    run vr1_mc vr1 "$MC"
    run direct_mc direct "$MC"
    run cost_mc cost "$MC" --model "$O/direct_mc/direct_model.npz" ;;
  *) echo "unknown $WHAT"; exit 2 ;;
esac
echo RUN_RECO_CHECKS_DONE
