#!/usr/bin/env python3
"""Reproduce this review artifact from the exact paused snapshot; never write Offline."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(root):
    changes = {}
    sources = {}
    def change(name, before, after):
        if name not in changes:
            sources[name] = (root / name).read_bytes()
            changes[name] = sources[name].decode()
        assert changes[name].count(before) == 1, (name, before)
        changes[name] = changes[name].replace(before, after)
    change('Mu2eG4/inc/Mu2eG4Config.hh',
           '      fhicl::Atom<int> mtDebugOutput {Name("mtDebugOutput"), 0};',
           '''      fhicl::Atom<int> mtDebugOutput {Name("mtDebugOutput"), 0};
      // Empty prefix disables every boundary-recorder hook and all diagnostic I/O.
      fhicl::Atom<std::string> boundaryRecorderPrefix {Name("boundaryRecorderPrefix"), ""};
      fhicl::Sequence<std::string> boundaryRecorderRoots {Name("boundaryRecorderRoots"), {}};
      fhicl::Atom<unsigned long long> boundaryRecorderMaxRows {Name("boundaryRecorderMaxRows"), 1000000ULL};
      fhicl::Atom<unsigned long long> boundaryRecorderMaxBytes {Name("boundaryRecorderMaxBytes"), 1073741824ULL};''')
    change('Mu2eG4/inc/Mu2eG4SteppingAction.hh',
           '#include "Offline/Mu2eG4/inc/Mu2eG4Config.hh"',
           '#include "Offline/Mu2eG4/inc/Mu2eG4Config.hh"\n#include "Offline/Mu2eG4/inc/Mu2eG4BoundaryRecorder.hh"')
    change('Mu2eG4/inc/Mu2eG4SteppingAction.hh',
           '    void BeginOfTrack();',
           '    Mu2eG4BoundaryRecorder& boundaryRecorder() { return boundaryRecorder_; }\n\n    void BeginOfTrack();')
    change('Mu2eG4/inc/Mu2eG4SteppingAction.hh',
           '    double mcTrajectoryMinDistanceCut(const G4VPhysicalVolume* vol) const;',
           '    double mcTrajectoryMinDistanceCut(const G4VPhysicalVolume* vol) const;\n\n    Mu2eG4BoundaryRecorder boundaryRecorder_;')
    change('Mu2eG4/src/Mu2eG4SteppingAction.cc',
           '    _spHelper()\n', '    _spHelper(),\n    boundaryRecorder_(debug)\n')
    change('Mu2eG4/src/Mu2eG4SteppingAction.cc',
           '  void Mu2eG4SteppingAction::UserSteppingAction(const G4Step* step){\n',
           '  void Mu2eG4SteppingAction::UserSteppingAction(const G4Step* step){\n\n    boundaryRecorder_.step(*step);\n')
    change('Mu2eG4/src/Mu2eG4TrackingAction.cc',
           '    saveSimParticleStart(trk);',
           '''    saveSimParticleStart(trk);

    if (_steppingAction->boundaryRecorder().enabled()) {
      const auto key = perThreadObjects_->simParticleHelper->particleKeyFromG4TrackID(trk->GetTrackID());
      const auto saved = _transientMap.find(key);
      _steppingAction->boundaryRecorder().start(*trk, saved == _transientMap.end() ? nullptr : &saved->second);
    }''')
    change('Mu2eG4/src/Mu2eG4TrackingAction.cc',
           '    saveSimParticleEnd(trk);',
           '    saveSimParticleEnd(trk);\n\n    _steppingAction->boundaryRecorder().stop(*trk);')
    change('Mu2eG4/src/Mu2eG4EventAction.cc',
           '    SimParticleHelper *_spHelper = &perThreadObjects_->simParticleHelper.value();',
           '''    SimParticleHelper *_spHelper = &perThreadObjects_->simParticleHelper.value();

    _steppingAction->boundaryRecorder().begin(_artEvent->id(), *evt, *_spHelper);''')
    change('Mu2eG4/src/Mu2eG4EventAction.cc',
           '    if (event_passes) {',
           '''    _steppingAction->boundaryRecorder().end(*evt, event_passes,
        _trackingAction->nG4Tracks(), _trackingAction->overflowSimParticles(),
        _trackingAction->nKilledStepLimit(), _trackingAction->nKilledByFieldPropagator());

    if (event_passes) {''')
    helper = 'Mu2eG4/inc/Mu2eG4BoundaryRecorder.hh'
    assert not (root / helper).exists(), 'helper already exists; not the paused source snapshot'
    changes[helper] = (HERE / 'payload/Mu2eG4BoundaryRecorder.hh').read_text()
    patch = ''
    for name in sorted(changes):
        old = sources.get(name, b'').decode()
        patch += ''.join(difflib.unified_diff(old.splitlines(True), changes[name].splitlines(True),
                         fromfile='a/' + name if name in sources else '/dev/null', tofile='b/' + name))
    manifest = {'format': 1, 'status': 'paused source snapshot, not current-head/build/runtime verified',
                'sources': {}, 'patched': {}, 'new_files': [helper]}
    for name, text in changes.items():
        manifest['patched'][name] = hashlib.sha256(text.encode()).hexdigest()
    references = ['Mu2eG4/inc/Mu2eG4TrackingAction.hh', 'Mu2eG4/inc/Mu2eG4EventAction.hh',
                  'Mu2eG4/inc/SimParticleHelper.hh', 'Mu2eG4/src/SimParticleHelper.cc',
                  'Mu2eG4/src/Mu2eG4Config.cc', 'Mu2eG4/inc/Mu2eG4PerThreadStorage.hh',
                  'Mu2eG4/src/constructDiskCalorimeter.cc',
                  'MCDataProducts/inc/SimParticle.hh', 'Mu2eG4/CMakeLists.txt', 'Mu2eG4/src/SConscript']
    for name in references:
        sources[name] = (root / name).read_bytes()
    manifest['sources'] = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(sources.items())}
    manifest['patch_sha256'] = hashlib.sha256(patch.encode()).hexdigest()
    return patch, manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline', type=Path, required=True)
    args = p.parse_args()
    patch, manifest = build(args.offline.resolve())
    (HERE / 'boundary-recorder.patch').write_text(patch)
    (HERE / 'source-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
