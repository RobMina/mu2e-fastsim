#!/bin/bash
# Warm throughput of the fast calorimeter chain in the fastsim image (GPU with --gres, or a CPU slot without).
#SBATCH --job-name=cal1_time
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -eo pipefail
: "${CAL1_CODE:?umbrella snapshot}"
: "${CAL1_IN:?directory with the chain inputs}"
: "${CAL1_SIF:?fastsim image}"
: "${CAL1_OUT:?output JSON}"
module load apptainer
nv=(); dev=""; [[ -n ${CUDA_VISIBLE_DEVICES:-} ]] && { nv=(--nv); dev=0; }
apptainer exec "${nv[@]}" --cleanenv -B "$CAL1_CODE" -B "$CAL1_IN" -B "$(dirname "$CAL1_OUT")" \
  --env PYTHONPATH="$CAL1_CODE/calo:$CAL1_CODE/core",CUDA_VISIBLE_DEVICES="$dev" "$CAL1_SIF" \
  python3 "$CAL1_CODE/calo/scripts/time_chain.py" --steps "$CAL1_IN/calo_response_tables.root" \
  --geometry "$CAL1_IN/calo_geometry_run1430.npz" --sim-crystal "$CAL1_IN/Sim_crystal.txt" \
  --pulse "$CAL1_IN/CsI-waveform-2020-02-12.root" --noise "$CAL1_IN/ReadoutNoise.root" \
  --propagation "$CAL1_IN/CsI-propag-2020-07-29.root" --repeat "${CAL1_REPEAT:-5}" --out "$CAL1_OUT"
