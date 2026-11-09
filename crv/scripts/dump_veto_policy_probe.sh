#!/usr/bin/env bash
# Run one user-built dumper regression, retaining effective config, loader trace and expected exit status.
set -euo pipefail
if [ "$#" -ne 6 ]; then
  echo "usage: $0 BUILT_AREA FCL SOURCE_ART EVENTS NEW_OUTPUT EXPECTED_EXIT" >&2
  exit 2
fi
area=$1
fcl=$2
source=$3
events=$4
out=$5
expected=$6
[ "${MUSE_WORK_DIR:-}" = "$area" ] || { echo "wrong MUSE_WORK_DIR" >&2; exit 2; }
mkdir "$out"
printf '%s\n' "$source" > "$out/source.txt"
printf '%s\n' "MUSE_WORK_DIR=$MUSE_WORK_DIR" "CET_PLUGIN_PATH=$CET_PLUGIN_PATH" > "$out/environment.txt"
sha256sum "$area/CrvFastSim/src/CrvStepTableDumper_module.cc" "$fcl" > "$out/source_sha256.txt"
find "$area/build" -name libcrvfastsim_CrvStepTableDumper_module.so -exec sha256sum {} \; >> "$out/source_sha256.txt"
cmd=(taskset -c 0 mu2e -c "$fcl" -s "$source" -n "$events" -T "$out/dump.root")
printf '%q ' "${cmd[@]}" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
"${cmd[@]}" --debug-config "$out/effective.fcl" > "$out/config.log" 2>&1
set +e
LD_DEBUG=libs LD_DEBUG_OUTPUT="$out/loader" "${cmd[@]}" > "$out/job.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$out/exit_status.txt"
[ "$rc" -eq "$expected" ] || { echo "unexpected exit $rc (expected $expected)" >&2; exit 1; }
tail -n 25 "$out/job.log"
