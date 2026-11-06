#!/bin/bash
# Warm throughput of the fast calorimeter chain in the fastsim image (GPU with --gres, or a CPU slot without).
#SBATCH --job-name=calo_time
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -eo pipefail
: "${CALO_CODE:?umbrella snapshot}"
: "${CALO_IN:?directory with the chain inputs}"
: "${CALO_SIF:?fastsim image}"
: "${CALO_OUT:?output JSON}"
module load apptainer
nv=(); dev=""; [[ -n ${CUDA_VISIBLE_DEVICES:-} ]] && { nv=(--nv); dev=0; }
apptainer exec "${nv[@]}" --cleanenv -B "$CALO_CODE" -B "$CALO_IN" -B "$(dirname "$CALO_OUT")" \
  --env PYTHONPATH="$CALO_CODE/calo:$CALO_CODE/core",CUDA_VISIBLE_DEVICES="$dev" "$CALO_SIF" \
  python3 "$CALO_CODE/calo/scripts/time_chain.py" --steps "$CALO_IN/calo_response_tables.root" \
  --geometry "$CALO_IN/calo_geometry_run1430.npz" --sim-crystal "$CALO_IN/Sim_crystal.txt" \
  --pulse "$CALO_IN/CsI-waveform-2020-02-12.root" --noise "$CALO_IN/ReadoutNoise.root" \
  --propagation "$CALO_IN/CsI-propag-2020-07-29.root" --repeat "${CALO_REPEAT:-5}" --out "$CALO_OUT" ${CALO_DEVICE_DIGITIZE:+--device-digitize}
