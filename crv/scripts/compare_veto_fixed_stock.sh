#!/bin/bash
# Replay a user-built fixed release or check its retained outputs, without replacing any reference.
set -eo pipefail
MODE=${1:?stock or check}
AREA=${2:?built satellite area}
BASE=${3:?retained replay data root}
OUT=${4:?candidate data directory with completed controls}
CONDITIONS=${5:?fast conditions dump}
SELECTION=${6:?saved selected-track parquet}
REFERENCE=${7:?saved deadtime stats}
CODE=$(cd "$(dirname "$0")/.." && pwd)
SCRIPT=$CODE/scripts/$(basename "$0")
BLOCKS=(block0 block1 block2_fullinventory)
BACKING=/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_41_00
for path in "$AREA" "$BASE" "$OUT" "$CONDITIONS" "$SELECTION" "$REFERENCE"; do
  case "$path" in /*) ;; *) echo "absolute path required: $path"; exit 2;; esac
done
case "$OUT" in "${M2FS_DATA:?set M2FS_DATA to the writable data area (see site.env.example)}"/*) ;; *) echo "output must be in the data area"; exit 2;; esac
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$MODE" = stock ]; then
  [ "${MUSE_WORK_DIR:-}" = "$AREA" ] || { echo "wrong MUSE_WORK_DIR"; exit 2; }
  [ "${MUSE_STUB:-}" = al9-prof-e29-p109 ] && [ "${MUSE_ENVSET:-}" = p109 ] || { echo "wrong muse build"; exit 2; }
  [ "$(readlink "$AREA/backing")" = "$BACKING" ] || { echo "wrong backing"; exit 2; }
  for block in "${BLOCKS[@]}"; do
    [ ! -e "$OUT/$block" ] || { echo "candidate block already exists: $block"; exit 2; }
    for file in mixed.root fast.parquet; do
      [ -f "$BASE/$block/$file" ] || { echo "missing input: $block/$file"; exit 2; }
    done
  done
  INPUT_SHA=$(sha256sum "$BASE/block0/mixed.root" | cut -d " " -f1)
  CONDITIONS_SHA=$(sha256sum "$CONDITIONS" | cut -d " " -f1)
  for n in 1 10; do
    STOCK=$OUT/control$n/stock.root
    STOCK_SHA=$(sha256sum "$STOCK" | cut -d " " -f1)
    jq -e --argjson n "$n" --arg input "$INPUT_SHA" --arg conditions "$CONDITIONS_SHA" --arg stock "$STOCK" --arg digest "$STOCK_SHA" \
      '.conditions_equal == true and .events_check.events == $n and .entries_highest_cycle.events == $n and .input_sha256 == $input and .conditions_sha256 == $conditions and .stock == $stock and .stock_sha256 == $digest' \
      "$OUT/control$n/check.json" >/dev/null
    for line in "area $AREA" "backing $BACKING" "table $BASE/block0/mixed.root" "events $n"; do
      grep -Fxq "$line" "$OUT/control$n/provenance.txt" || { echo "control provenance mismatch"; exit 2; }
    done
  done
  cd "$AREA"
  sha256sum -c adapter_source_sha256.txt > "$OUT/adapter_sources_verified.txt"
  {
    date -u
    hostname
    printf 'MUSE_WORK_DIR=%s\nMUSE_STUB=%s\nMUSE_ENVSET=%s\n' "$MUSE_WORK_DIR" "$MUSE_STUB" "$MUSE_ENVSET"
    sha256sum "$SCRIPT" "$AREA/run_veto_stock_replay.sh"
    for library in "$AREA"/build/al9-prof-e29-p109/CrvFastSim/lib/*.so "$AREA"/backing/build/al9-prof-e29-p109/Offline/lib/libmu2e_CRVResponse*.so "$AREA"/backing/build/al9-prof-e29-p109/Offline/lib/libmu2e_CRVReco*.so; do
      sha256sum "$library"
      ldd "$library"
    done
  } > "$OUT/runtime_libraries.txt"
  if grep -q "not found" "$OUT/runtime_libraries.txt"; then echo "missing library"; exit 2; fi
  for i in 0 1 2; do
    block=${BLOCKS[$i]}
    bash "$AREA/run_veto_stock_replay.sh" "$AREA" "$BASE/$block/mixed.root" "$OUT/$block" 3000 "$((9100 + i))"
    mu2e --debug-config "$OUT/$block/expanded.fcl" -c "$OUT/$block/replay.fcl" -T "$OUT/$block/stock.root" --annotate > "$OUT/$block/expand.txt" 2>&1
    sha256sum "$OUT/$block/replay.fcl" "$OUT/$block/expanded.fcl" "$OUT/$block/stock.root" > "$OUT/$block/completed_sha256.txt"
  done
elif [ "$MODE" = check ]; then
  [ -z "${MUSE_WORK_DIR:-}" ] || { echo "use an ana-only shell"; exit 2; }
  [ ! -e "$OUT/paired_intended_per_sector_v1.json" ] || { echo "report exists"; exit 2; }
  export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
  cd "$CODE"
  for block in "${BLOCKS[@]}"; do
    [ ! -e "$OUT/$block/check.json" ] || { echo "block check exists: $block"; exit 2; }
    for file in "$BASE/$block/mixed.root" "$BASE/$block/fast.parquet" "$OUT/$block/stock.root"; do
      [ -f "$file" ] || { echo "missing input: $file"; exit 2; }
    done
    jq -e '.response_contract == null or .response_contract.policy == "intended_per_sector_v1"' "$BASE/$block/fast.json" >/dev/null
  done
  CHECKS=(); FAST=()
  for block in "${BLOCKS[@]}"; do
    CHECK=$OUT/$block/check.json
    python3 -m crvfastsim.veto.replay_check --input "$BASE/$block/mixed.root" --stock "$OUT/$block/stock.root" \
      --events 3000 --conditions "$CONDITIONS" --fast "$BASE/$block/fast.parquet" \
      --legacy-yield-policy intended_per_sector_v1 --out "$CHECK" > "$OUT/$block/check.txt" 2>&1
    jq -e '.response_identity.response_contract.policy == "intended_per_sector_v1"' "$CHECK" >/dev/null
    CHECKS+=("$CHECK"); FAST+=("$BASE/$block/fast.parquet")
  done
  sha256sum "$CODE"/crvfastsim/veto/replay_{check,metrics,policy}.py "$SCRIPT" "$CONDITIONS" "$SELECTION" "$REFERENCE" > "$OUT/check_code_inputs_sha256.txt"
  python3 -m crvfastsim.veto.replay_metrics --checks "${CHECKS[@]}" --fast "${FAST[@]}" \
    --selection "$SELECTION" --reference "$REFERENCE" --out "$OUT/paired_intended_per_sector_v1.json"
else
  echo "mode must be stock or check"; exit 2
fi
