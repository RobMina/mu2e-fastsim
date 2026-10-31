#!/bin/bash
# Export a bounded paired-response pilot from one existing Cat table per species; no simulation or submission.
set -e
: "${CRVFS_DATA:?source site.env}"
: "${CRVFS_CODE:?source site.env}"
OUT=${1:?new output ROOT path}
INVENTORY=${2:?completed Cat basename inventory}
PNFS=${3:?Cat parquet output root}
EVENTS=${4:-300}
SEED=${5:-20261025}
[ ! -e "$OUT" ] && [ ! -e "${OUT%.root}.json" ] && [ ! -e "${OUT%.root}.sources.txt" ] || { echo "output already exists"; exit 2; }
command -v ifdh >/dev/null
WORK=$(mktemp -d -t crv-replay.XXXXXXXX)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$(dirname "$OUT")"
for species in EleBeamFlashCat MuBeamFlashCat MuStopPileupCat NeutralsFlashCat; do
  name=$(grep -m 1 "^dts[.]mu2e[.]${species}[.]" "$INVENTORY")
  [ -n "$name" ] || { echo "missing species $species"; exit 3; }
  mkdir -p "$WORK/$species"
  src="$PNFS/$species/steps.$name.parquet"
  echo "$src" | tee -a "${OUT%.root}.sources.txt"
  ifdh cp "$src" "$WORK/$species/steps.$name.parquet"
done
python3 -m crvfastsim.veto.replay --tables "$WORK" --events "$EVENTS" --seed "$SEED" --out "$OUT"
echo "Pilot uses one table/species: response wiring only, not a deadtime closure measurement."
