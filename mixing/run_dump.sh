#!/bin/bash
# Extract complete donor pools in an already configured mu2e runtime.
set -eo pipefail
[[ $# -gt 0 ]] || { echo 'usage: run_dump.sh DONOR.art ...' >&2; exit 2; }
for input in "$@"; do
  name=$(basename "$input" .art)
  mu2e -c mixing/fcl/dump_frames.fcl -s "$input" -n -1 -T "$name.root"
done
