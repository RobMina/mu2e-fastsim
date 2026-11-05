#!/bin/bash
# Run existing compiled modules only; this script never builds software.
set -eo pipefail
: "${CALO_CODE:?snapshot root}"
: "${CALO_OUT:?output directory}"
: "${CALO_AREA:?already-built area or tarball}"
: "${CALO_STEPS:?existing stock dts file}"
: "${CALO_DIGIS:?existing stock digi file}"
: "${CVMFSEXEC_DIST:?prepared cvmfsexec}"
module load apptainer
mkdir -p "$CALO_OUT"/{products_steps,products_digis,reco}
export OFFLINE_BINDS="${CALO_BINDS:?same-path input binds}"
export OFFLINE_EVENTS=1 OFFLINE_LOG_PREFIX=products
cd "$CALO_OUT/products_steps"
bash "$CALO_CODE/container/offline/offline_uva.sh" "$CALO_AREA" "$CALO_CODE/calo/fcl/products.fcl" "$CALO_STEPS" > products.log 2>&1
cd "$CALO_OUT/products_digis"
bash "$CALO_CODE/container/offline/offline_uva.sh" "$CALO_AREA" "$CALO_CODE/calo/fcl/products.fcl" "$CALO_DIGIS" > products.log 2>&1
cd "$CALO_OUT/reco"
export OFFLINE_EVENTS="${CALO_EVENTS:-100}" OFFLINE_LOG_PREFIX=control OFFLINE_OUTPUT_FILE="$PWD/control.root"
bash "$CALO_CODE/container/offline/offline_uva.sh" "$CALO_AREA" "$CALO_CODE/calo/fcl/reco_control.fcl" "$CALO_DIGIS" > control.log 2>&1
