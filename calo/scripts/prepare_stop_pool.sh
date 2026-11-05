#!/bin/bash
# Copy upstream pool entries only; no generation, transport or detector cuts.
set -eo pipefail
FCL=${1:?pool-copy FCL}
OUT=${2:?output directory}
INPUT=${3:?upstream stop-pool art}
mkdir -p "$OUT"
cd "$OUT"
test ! -e stop_pool.art
fhicl-dump -c "$FCL" > copy.expanded.fcl
mu2e -c "$FCL" -n 2000 -s "$INPUT" > copy.log 2>&1
config_dumper -P stop_pool.art > pool.config.txt
sha256sum stop_pool.art > MANIFEST.sha256
printf '%s\n' "$INPUT" > SOURCE.txt
