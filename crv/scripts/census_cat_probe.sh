#!/bin/bash
# Run bounded census probes using an explicitly built satellite area and known source list.
set -e
AREA=${1:?built art area}
LIST=${2:?species/source list}
OUT=${3:?new output directory}
SEQ=${4:?source filename sequencer}
EVENTS=${5:?event limit or -1 for full source}
MODE=${6:-steps}
[ "${MUSE_WORK_DIR:-}" = "$AREA" ] || { echo "wrong MUSE_WORK_DIR"; exit 2; }
[ ! -e "$OUT" ] || { echo "output exists"; exit 2; }
SRC=$(awk -v seq="$SEQ" '$1 == "NeutralsFlashCat" && index($2, "001430_" seq ".art") {print $2}' "$LIST")
[ "$(printf '%s\n' "$SRC" | wc -l)" -eq 1 ] && [ -n "$SRC" ] || { echo "ambiguous source"; exit 2; }
FCL=CrvFastSim/fcl/dump_crvsteps_cat.fcl
[ "$MODE" = steps ] || { [ "$MODE" = gen ] || exit 2; FCL=CrvFastSim/fcl/dump_cat_census_gen.fcl; }
mkdir "$OUT"
printf '%s\n' "$SRC" > "$OUT/source.txt"
printf '%s\n' "MUSE_WORK_DIR=$MUSE_WORK_DIR" "CET_PLUGIN_PATH=$CET_PLUGIN_PATH" > "$OUT/environment.txt"
sha256sum "$AREA/CrvFastSim/src/CrvStepTableDumper_module.cc" "$AREA/$FCL" > "$OUT/source_sha256.txt"
find "$AREA/build" -name libcrvfastsim_CrvStepTableDumper_module.so -exec sha256sum {} \; >> "$OUT/source_sha256.txt"
printf 'taskset -c 0 mu2e -c %s -s %s -n %s -T %s\n' "$FCL" "$SRC" "$EVENTS" "$OUT/dump.root" > "$OUT/command.txt"
taskset -c 0 mu2e -c "$FCL" -s "$SRC" -n "$EVENTS" -T "$OUT/dump.root" --debug-config "$OUT/effective.fcl" > "$OUT/config.log" 2>&1
taskset -c 0 mu2e -c "$FCL" -s "$SRC" -n "$EVENTS" -T "$OUT/dump.root" > "$OUT/job.log" 2>&1
