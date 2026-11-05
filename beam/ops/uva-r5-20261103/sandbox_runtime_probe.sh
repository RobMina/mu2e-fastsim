#!/bin/bash
# Probe the final candidate inside the pinned runtime; never run Geant4.
set -eo pipefail
: "${R5_SANDBOX_ROOT:?}" "${R5_SANDBOX_CODE:?}" "${R5_HOST_BWRAP:?}"
cd "$R5_SANDBOX_ROOT"
export PATH="$(dirname "$R5_HOST_BWRAP"):$PATH"
command -v python3 > python.txt
python3 --version >> python.txt
python3 -B "$R5_SANDBOX_CODE/beam/beamfastsim/flash_reference_sandbox.py" --node-tmp /tmp --file-size-limit 16777216 -- "$(command -v python3)" -B "$R5_SANDBOX_CODE/beam/beamfastsim/flash_reference_sandbox_probe.py" --persistent-target /scratch/ram2aq/mu2efastsim/flash_r5_20261103/sandbox_persistent_escape_1748 --persistent-target /home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/sandbox_persistent_escape_1748 > sandbox_probe.json
python3 -B "$R5_SANDBOX_CODE/beam/beamfastsim/flash_reference_sandbox.py" --node-tmp /tmp --file-size-limit 16777216 -- "$(command -v mu2e)" -c FlashReference/fcl/fixture.fcl -T fixture.root > fixture.log 2>&1
printf 'sandbox_commands_exit0=true\ntransport_events=0\ncompleted=%s\n' "$(date -Is)" > complete.txt
