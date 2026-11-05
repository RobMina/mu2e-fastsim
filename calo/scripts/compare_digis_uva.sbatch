#!/bin/bash
# compare_digis.py in the fastsim image, reading the steps from the stock art file (CPU).
#SBATCH --job-name=calo_digis
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -eo pipefail
: "${CALO_CODE:?umbrella snapshot}"
: "${CALO_IN:?directory with Sim_crystal.txt, pulse and noise ROOT files}"
: "${CALO_ART:?stock art file with CaloShowerSteps}"
: "${CALO_NTUPLE:?stock_chain ntuple}"
: "${CALO_SIF:?fastsim image}"
: "${CALO_OUTJSON:?output JSON}"
module load apptainer
apptainer exec --cleanenv -B "$CALO_CODE" -B "$CALO_IN" -B "$(dirname "$CALO_ART")" -B "$(dirname "$CALO_NTUPLE")" \
  -B "$(dirname "$CALO_OUTJSON")" --env PYTHONPATH="$CALO_CODE/calo:$CALO_CODE/core" "$CALO_SIF" \
  python3 "$CALO_CODE/calo/scripts/compare_digis.py" --backend jax --steps "$CALO_ART" --ntuple "$CALO_NTUPLE" \
  ${CALO_BASELINE:+--baseline-ntuple "$CALO_BASELINE"} --sim-crystal "$CALO_IN/Sim_crystal.txt" \
  --pulse "$CALO_IN/CsI-waveform-2020-02-12.root" --noise "$CALO_IN/ReadoutNoise.root" --out "$CALO_OUTJSON"
