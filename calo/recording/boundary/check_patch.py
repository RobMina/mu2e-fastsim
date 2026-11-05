#!/usr/bin/env python3
"""Hash applicability + structural tests; patch executable touches temporary copies only, never Offline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(test, message):
    if not test:
        raise RuntimeError(message)


def hashes(root, expected):
    for name, sha in expected.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'unsafe manifest path')
        require(digest((root / name).read_bytes()) == sha, 'snapshot hash mismatch: ' + name)


def structural(root):
    def text(name):
        return (root / ('Mu2eG4/' + name)).read_text()
    config = text('inc/Mu2eG4Config.hh')
    helper = text('inc/Mu2eG4BoundaryRecorder.hh')
    event = text('src/Mu2eG4EventAction.cc')
    tracking = text('src/Mu2eG4TrackingAction.cc')
    stepping = text('src/Mu2eG4SteppingAction.cc')
    require('boundaryRecorderPrefix {Name("boundaryRecorderPrefix"), ""}' in config, 'prefix must default off')
    require('boundaryRecorderRoots {Name("boundaryRecorderRoots"), {}}' in config, 'no implicit domain roots')
    require('if (prefix.empty()) return;' in helper, 'disabled constructor must have no I/O')
    require('IsMultithreadedApplication()' in helper, 'sequential-only guard required')
    require(event.index('boundaryRecorder().begin(') < event.index('_sensitiveDetectorHelper->createProducts'), 'begin before detector hooks')
    require('_artEvent->id(), *evt, *_spHelper' in event, 'independent art event identity')
    require(event.index('boundaryRecorder().end(') < event.index('    if (event_passes) {'), 'end must be outside acceptance gate')
    require(tracking.index('saveSimParticleStart(trk);') < tracking.index('boundaryRecorder().start(') < tracking.index('Mu2eG4UserHelpers::controlTrajectorySaving'), 'birth hook phase')
    require(tracking.index('saveSimParticleEnd(trk);') < tracking.index('boundaryRecorder().stop('), 'terminal hook phase')
    require(stepping.index('boundaryRecorder_.step(*step);') < stepping.index('++numTrackSteps_;'), 'step hook before stock changes')
    for token in ['pre.GetTouchable()', 'post.GetTouchable()', 'fWorldBoundary', 'GetHistoryDepth()',
                  'GetReplicaNumber(d)', 'GetCopyNo()', 'max_digits10', 'particlePtr(&t)',
                  'identity(saved->parent())', 'identity(saved->genParticle())', 'membership_unavailable',
                  'birth_membership_unresolved', 'born_inside', 'terminal_inside', 'tracking_resume',
                  'fStopAndKill', 'fKillTrackAndSecondaries', 'event_begin', 'event_end', 'e.IsAborted()',
                  'O_EXCL', '::mkdir', 'std::ios::app', 'out_.flush()', 'if (!out_)',
                  'rows_ >= maxRows_', 'maxBytes_ - bytes_', 'sequence_ = 0', '++sequence_', '++i.visit']:
        require(token in helper, 'missing structural contract: ' + token)
    for forbidden in ['findAndCount', 'LocateGlobalPoint', 'GetNavigator', 'G4UniformRand', 'G4Random',
                      'SetMomentum', 'SetTrackStatus', 'SetPosition', 'const_cast', 'TFile', 'TTree']:
        require(forbidden not in helper, 'forbidden helper behavior: ' + forbidden)
    require(helper.count('sequence_ = 0') == 2, 'sequence declaration and per-event reset only')
    require('CalorimeterMother' not in helper, 'root name must not be hardwired')
    require('/*' not in helper, 'helper uses one-line comments')
    for token in ['#include "Geant4/G4PhysicalVolumeStore.hh"', 'G4PhysicalVolumeStore::GetInstance()',
                  'for (const auto* volume : *store)', 'volume->GetName() == root', '++matches',
                  'if (matches == 0) fail(', 'if (matches != 1) fail(',
                  'selected->IsReplicated() || selected->IsParameterised()',
                  'p.id().value()', 'art_ProductID_value_uint32', 'mm_MeV_ns']:
        require(token in helper, 'missing root/identity contract: ' + token)
    begin = helper[helper.index('    void begin('):helper.index('    void start(')]
    require(begin.index('if (!enabled_) return;') < begin.index('validateRoots();')
            < begin.index('inEvent_ = true;') < begin.index('row("event_begin"'),
            'actual root validation must precede event activation and begin marker')
    require('G4_native_mm_MeV_ns' not in helper, 'units must match recording schema')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline', type=Path, required=True)
    args = p.parse_args()
    source = args.offline.resolve()
    manifest = json.loads((HERE / 'source-manifest.json').read_text())
    patch = HERE / 'boundary-recorder.patch'
    require(shutil.which('patch') is not None, 'patch executable required')
    allowed = {'Mu2eG4/inc/Mu2eG4Config.hh', 'Mu2eG4/inc/Mu2eG4SteppingAction.hh',
               'Mu2eG4/src/Mu2eG4SteppingAction.cc', 'Mu2eG4/src/Mu2eG4TrackingAction.cc',
               'Mu2eG4/src/Mu2eG4EventAction.cc', 'Mu2eG4/inc/Mu2eG4BoundaryRecorder.hh'}
    require(set(manifest['patched']) == allowed, 'unexpected patch footprint')
    require(digest(patch.read_bytes()) == manifest['patch_sha256'], 'patch artifact hash mismatch')
    hashes(source, manifest['sources'])
    for name in manifest['new_files']:
        require(not (source / name).exists(), 'new-file collision: ' + name)
    # Reproduction also checks the mirrored helper payload and exact insertion-only hook recipe.
    from make_patch import build
    reproduced, regenerated = build(source)
    for line in reproduced.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            require('findAndCount' not in line, 'new process counter mutation')
        if line.startswith('-') and not line.startswith('---'):
            require(line == '-    _spHelper()', 'unexpected removal of stock source logic')
    require(reproduced.encode() == patch.read_bytes(), 'patch differs from reproducible recipe/payload')
    require(regenerated['patched'] == manifest['patched'], 'patched hash manifest mismatch')
    with tempfile.TemporaryDirectory(prefix='mu2e-boundary-patch-') as tmp:
        target = Path(tmp) / 'Offline'
        target.mkdir()
        for name in manifest['sources']:
            dst = target / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((source / name).read_bytes())
        cmd = ['patch', '-p1', '--batch', '--fuzz=0', '-i', str(patch)]
        for extra in [['--dry-run'], []]:
            result = subprocess.run(cmd + extra, cwd=target, capture_output=True, text=True)
            require(result.returncode == 0, 'temporary patch failed:\n' + result.stdout + result.stderr)
            require('offset' not in result.stdout.lower() and 'fuzz' not in result.stdout.lower(), 'patch used offset/fuzz')
        hashes(target, manifest['patched'])
        structural(target)
        # These mutation checks exercise the source-only contract checker, not native G4 behavior.
        helper_path = target / 'Mu2eG4/inc/Mu2eG4BoundaryRecorder.hh'
        helper_text = helper_path.read_text()
        for guard in ['if (matches == 0) fail(', 'if (matches != 1) fail(',
                      'selected->IsReplicated() || selected->IsParameterised()', '      validateRoots();']:
            require(helper_text.count(guard) == 1, 'mutation guard must be unique')
            helper_path.write_text(helper_text.replace(guard, '// removed guard: ', 1))
            try:
                structural(target)
            except (RuntimeError, ValueError):
                pass
            else:
                raise RuntimeError('missing root guard was not rejected: ' + guard)
            finally:
                helper_path.write_text(helper_text)
        # Deliberate temporary drift must be rejected by the same preflight hash check.
        victim = target / 'Mu2eG4/inc/Mu2eG4Config.hh'
        original = victim.read_bytes()
        victim.write_bytes(original + b'\n// deliberate drift\n')
        try:
            hashes(target, manifest['patched'])
        except RuntimeError:
            pass
        else:
            raise RuntimeError('drift detection failed')
        victim.write_bytes(original)
        result = subprocess.run(cmd + ['-R'], cwd=target, capture_output=True, text=True)
        require(result.returncode == 0, 'temporary reverse patch failed: ' + result.stdout + result.stderr)
        hashes(target, manifest['sources'])
        for name in manifest['new_files']:
            require(not (target / name).exists() or (target / name).stat().st_size == 0, 'reverse left helper content')
    hashes(source, manifest['sources'])
    print('PASS: exact source hashes; reproduced patch; patch dry-run/apply/reverse in temporary copy;')
    print('PASS: patched hashes; structural contracts and root-guard mutation rejection; deliberate drift rejection; original sources unchanged.')
    print('NOT TESTED: C++ compile/link, FHiCL/G4 API compatibility, native runtime, geometry containment, physics equivalence.')


if __name__ == '__main__':
    main()
