#!/usr/bin/env python3
"""Generate a source-only inventory review patch; never write the selected checkout."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = "Mu2eG4/src/Mu2eG4RunAction.cc"
HEADER = "Mu2eG4/inc/Mu2eG4PlacementInventory.hh"
REFERENCES = ["Mu2eG4/inc/Mu2eG4RunAction.hh", "Mu2eG4/inc/Mu2eG4Config.hh",
              "Mu2eG4/CMakeLists.txt", "Mu2eG4/src/SConscript"]
INCLUDE = '#include "Offline/Mu2eG4/inc/Mu2eG4RunAction.hh"'
ANCHOR = "    _steppingAction->finishConstruction();"
HOOK = '''
    // Optional all-world native placement inventory; separate output suffix.
    if (!debug_.boundaryRecorderPrefix().empty()) {
      Mu2eG4PlacementInventory::write(debug_, aRun->GetRunID());
    }'''
CONTRACT = {
    "boundaryRecorderPrefix": 'fhicl::Atom<std::string> boundaryRecorderPrefix {Name("boundaryRecorderPrefix"), ""};',
    "boundaryRecorderRoots": 'fhicl::Sequence<std::string> boundaryRecorderRoots {Name("boundaryRecorderRoots"), {}};',
    "boundaryRecorderMaxRows": 'fhicl::Atom<unsigned long long> boundaryRecorderMaxRows {Name("boundaryRecorderMaxRows"), 1000000ULL};',
    "boundaryRecorderMaxBytes": 'fhicl::Atom<unsigned long long> boundaryRecorderMaxBytes {Name("boundaryRecorderMaxBytes"), 1073741824ULL};',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(root):
    sources = {name: (root / name).read_bytes() for name in [RUN, *REFERENCES]}
    old = sources[RUN].decode()
    if (root / HEADER).exists():
        raise ValueError("new-header collision")
    if old.count(INCLUDE) != 1 or old.count(ANCHOR) != 1 or 'Mu2eG4PlacementInventory' in old:
        raise ValueError("not the reviewed unmodified run-action insertion sites")
    new = old.replace(INCLUDE, INCLUDE + '\n#include "Offline/' + HEADER + '"')
    new = new.replace(ANCHOR, ANCHOR + HOOK)
    changes = {RUN: new, HEADER: (HERE / 'payload' / Path(HEADER).name).read_text()}
    patch = ''.join(''.join(difflib.unified_diff(
        sources.get(name, b'').decode().splitlines(True), text.splitlines(True),
        fromfile='a/' + name if name in sources else '/dev/null', tofile='b/' + name))
        for name, text in sorted(changes.items()))
    manifest = {
        'format': 1,
        'status': 'audited-baseline compact-v2 source proposal; v2 compile/native validation pending',
        'sources': {name: sha(data) for name, data in sorted(sources.items())},
        'patched': {name: sha(text.encode()) for name, text in sorted(changes.items())},
        'new_files': [HEADER], 'patch_sha256': sha(patch.encode()),
        'requires_external_boundary_debug_contract': CONTRACT,
        'no_config_patch': True,
    }
    return patch, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', required=True, type=Path)
    args = parser.parse_args()
    patch, manifest = build(args.offline.resolve())
    (HERE / 'placement-inventory.patch').write_text(patch)
    (HERE / 'source-manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
