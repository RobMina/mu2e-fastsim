#!/bin/bash
# Stage a recorded Cat subset into per-job tmp, export raw mixture, remove only this job's staged tables.
set -e
LIST=${1:?species/path subset list}
OUT=${2:?new replay ROOT}
EVENTS=${3:?events}
SEED=${4:?mix seed}
[ ! -e "$OUT" ] && [ ! -e "${OUT%.root}.json" ] || { echo "output already exists"; exit 2; }
WORK=$(mktemp -d -t crv-replay-subset.XXXXXXXX)
trap 'rm -rf "$WORK"' EXIT
while read -r species src; do
  mkdir -p "$WORK/$species"
  ifdh cp "$src" "$WORK/$species/$(basename "$src")"
done < "$LIST"
python3 -m crvfastsim.veto.replay --tables "$WORK" --events "$EVENTS" --seed "$SEED" --out "$OUT"
