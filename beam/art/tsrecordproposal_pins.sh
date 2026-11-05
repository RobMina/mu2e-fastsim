#!/bin/bash
# Bind reviewed upstream base files and local proposal bytes without executing native code.
set -euo pipefail
[[ $# -eq 3 ]] || { echo 'usage: tsrecordproposal_pins.sh BASE_ROOT PATCHED_ROOT PROPOSAL_ROOT' >&2; exit 2; }
BASE=$1
PATCHED=$2
PROPOSAL=$3
TEMP=$(mktemp -d)
trap 'rm -rf "$TEMP"' EXIT
hash() { shasum -a 256 "$1" | cut -d' ' -f1; }
{
  sed -n 's|^--- a/||p' "$PROPOSAL/target_recording.patch"
  printf '%s\n' Mu2eG4/src/SConscript Mu2eG4/src/SimParticleHelper.cc Mu2eG4/inc/SimParticleHelper.hh \
    Mu2eG4/inc/SimParticlePrimaryHelper.hh Mu2eG4/src/SimParticlePrimaryHelper.cc \
    Mu2eG4/src/Mu2eG4PrimaryGeneratorAction.cc Mu2eG4/src/Mu2eG4PerThreadStorage.cc \
    Mu2eG4/inc/Mu2eG4Inputs.hh Mu2eG4/src/Mu2eG4Inputs.cc \
    MCDataProducts/inc/GenParticle.hh MCDataProducts/inc/SimParticle.hh MCDataProducts/inc/GenId.hh MCDataProducts/inc/ProcessCode.hh
} | sort -u | while IFS= read -r F; do printf '%s\t%s\n' "$F" "$(hash "$BASE/$F")"; done \
  | jq -Rn '[inputs | split("\t") | {(.[0]): .[1]}] | add' > "$TEMP/base.json"
sed -n 's|^--- a/||p' "$PROPOSAL/target_recording.patch" \
  | while IFS= read -r F; do printf '%s\t%s\n' "$F" "$(hash "$PATCHED/$F")"; done \
  | jq -Rn '[inputs | split("\t") | {(.[0]): .[1]}] | add' > "$TEMP/patched.json"
for F in target_recording.patch TargetRecording.hh TargetRecording.cc README.md; do
  printf '%s\t%s\n' "$F" "$(hash "$PROPOSAL/$F")"
done | jq -Rn '[inputs | split("\t") | {(.[0]): .[1]}] | add' > "$TEMP/assets.json"
jq -n --slurpfile base "$TEMP/base.json" --slurpfile patched "$TEMP/patched.json" --slurpfile assets "$TEMP/assets.json" \
  '{schema:"target-recording-source-proposal-v1",offline_revision:"b24d4163e5e8f5ed2b5f72fccb25e4980fec64e2",library_eligible:false,source_sha256:$base[0],patched_sha256:$patched[0],proposal_sha256:$assets[0]}' \
  > "$PROPOSAL/source_pins.json"
