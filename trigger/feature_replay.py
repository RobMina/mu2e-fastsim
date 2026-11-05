"""Separate full-history feature-export source preparation; never builds or launches art.

A successful preflight is NOT runtime admission: a user build and independently
verified source/library receipt remain required. Feature-table I/O is separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

import full_replay
import integrity
import quiet_replay as quiet
import replay


SOURCE_FILES = ('src/KalSeedSelectionFeatures_module.cc', 'src/SConscript', 'CMakeLists.txt', '.muse')
BACKING = '/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ba'
RELEASE = 'v13_41_01'
COMMIT = 'b132ab6567ac021851353e05b7a1feb45bfb7c14'
SOURCE_URL = f'https://github.com/Mu2e/Offline/blob/{COMMIT}/TrkFilters/src/KalSeedFilter_module.cc'
LABEL = 'r7SelectionFeatures'


def _sources(exporter_source, exporter_sha256):
    source = Path(exporter_source).absolute()
    owned = Path(__file__).resolve().parent / 'art/R7Trigger/src/KalSeedSelectionFeatures_module.cc'
    if source != owned or source.resolve() != owned:
        raise ValueError('explicit exporter source must be the R7-owned satellite module, not an owner checkout')
    root = source.parent.parent
    paths = [root / name for name in SOURCE_FILES]
    if any(path.resolve() != path for path in paths):
        raise ValueError('exporter source/build files must not redirect outside the R7 satellite')
    rows = integrity.files(paths)
    if (not isinstance(exporter_sha256, str) or not re.fullmatch('[0-9a-f]{64}', exporter_sha256)
            or rows[0]['sha256'] != exporter_sha256):
        raise ValueError('explicit exporter source SHA256 differs')
    return rows


def prepare(config_path, baseline_prepared, baseline_art, baseline_comparison, *,
            expected_events, runtime_area, runtime_image, exporter_source, exporter_sha256):
    """Admit the completed verbose FULLHYBRID; discard, never use, its quiet variant."""
    sources = _sources(exporter_source, exporter_sha256)
    _, admitted = quiet.prepare(config_path, baseline_prepared, baseline_art, baseline_comparison,
        expected_events=expected_events, runtime_area=runtime_area, runtime_image=runtime_image)
    evidence = admitted.pop('quiet_companion')
    if evidence['baseline_kind'] != 'fullhybrid':
        raise ValueError('feature export requires a completed verbose FULLHYBRID baseline')
    result = replay.parse_expanded((Path(baseline_prepared) / 'replay.fcl').read_text())
    output = replay.plain(result['outputs']['DecisionProducts'])
    if (output.get('fileName') != 'decisions.art' or output.get('SelectEvents') is not None
            or output.get('outputCommands') != ['drop *', 'keep art::TriggerResults_*_*_*']):
        raise ValueError('need unchanged unconditional decisions.art with both TriggerResults products')
    service = result['services'].get('TFileService')
    if service is not None and replay.plain(service) != {'fileName': '/dev/null', 'service_type': 'TFileService'}:
        raise ValueError('existing TFileService must be the exact production /dev/null placeholder')
    physics = result['physics']
    if any(LABEL in physics.get(group, {}) for group in ('analyzers', 'producers', 'filters')) or LABEL in physics:
        raise ValueError('exporter label already exists')
    physics['analyzers'][LABEL] = {'module_type': 'KalSeedSelectionFeatures',
        'kalSeeds': 'TrigKSFDe::TriggerReplay',
        'selectionInfo': ['TprTrkDe80m70pD0200KSFilter::TriggerReplay',
                          'TprTrkDe80m70pKSFilter::TriggerReplay']}
    physics['DecisionOutput'] = ['decisions', LABEL, 'DecisionProducts']
    result['services'].setdefault('TFileService', {})['fileName'] = 'r7-selection-features.root'
    admitted.update(mode='selection_feature_export', build_required=True,
        status='source_preparation_only', runtime_ready=False,
        replay_fcl_sha256=quiet._sha(full_replay.render_config(result)))
    admitted['feature_export'] = {key: evidence[key] for key in (
        'version', 'baseline_kind', 'runtime', 'config_path', 'baseline_prepared',
        'baseline_art', 'baseline_comparison', 'baseline_evidence', 'baseline_metadata')}
    admitted['feature_export'].update(exporter_source=str(Path(exporter_source).resolve()),
        exporter_sha256=exporter_sha256, source_files=sources, required_backing=BACKING,
        offline_release=RELEASE, offline_commit=COMMIT, source_binding_url=SOURCE_URL,
        transformation='one unconditional analyzer in existing DecisionOutput between decisions and DecisionProducts; TFileService only',
        physics_claim=False, measured_parity=False,
        required_build_receipt_fields=['library_path', 'library_sha256', 'source_files',
            'required_backing', 'muse_stub', 'production_profile', 'offline_release', 'resolved_library_paths'],
        runtime_gate='STOP: user build plus independently verified pinned library SHA256 and source/build/runtime receipt required; no boolean build proof')
    extra = admitted.get('reference_integrity', []) + admitted.get('injection_campaign_evidence', [])
    quiet.hybrid_history._same_files(evidence['baseline_evidence'],
        quiet._evidence(Path(config_path), Path(baseline_prepared), Path(baseline_art), Path(baseline_comparison)),
        'feature baseline evidence')
    quiet.hybrid_history._same_files(admitted['input_integrity'],
        integrity.files(result['source']['fileNames']), 'feature baseline sources')
    if extra:
        quiet.hybrid_history._same_files(extra, integrity.files([row['path'] for row in extra]),
            'feature reference or campaign evidence')
    if _sources(exporter_source, exporter_sha256) != sources:
        raise ValueError('exporter evidence changed during preparation')
    return result, admitted


def preflight(prepared, config_text, expected_events, runtime_area, runtime_image,
              exporter_source, exporter_sha256):
    """Re-admit all evidence and rederive the exact source preparation, not runtime readiness."""
    prepared = Path(prepared)
    saved = quiet._load(prepared / 'manifest.json')
    feature = saved['feature_export']
    if config_text != Path(feature['config_path']).read_text():
        raise ValueError('original configuration bytes differ')
    result, admitted = prepare(feature['config_path'], feature['baseline_prepared'], feature['baseline_art'],
        feature['baseline_comparison'], expected_events=expected_events, runtime_area=runtime_area,
        runtime_image=runtime_image, exporter_source=exporter_source, exporter_sha256=exporter_sha256)
    if saved != admitted or (prepared / 'replay.fcl').read_text() != full_replay.render_config(result):
        raise ValueError('feature preparation differs from deterministic admitted source variant')
    return saved, result


def compare_decisions(feature_art, baseline_art, manifest):
    """Saved-art packed-word equality only; not feature, execution or runtime validation."""
    feature = manifest['feature_export']
    runtime = feature['runtime']
    result, admitted = prepare(feature['config_path'], feature['baseline_prepared'], baseline_art,
        feature['baseline_comparison'], expected_events=manifest['expected_events'],
        runtime_area=runtime['TRIGGER_AREA'], runtime_image=runtime['OFFLINE_IMAGE'],
        exporter_source=feature['exporter_source'], exporter_sha256=feature['exporter_sha256'])
    if admitted != manifest:
        raise ValueError('manifest or baseline differs from admitted preparation')
    evidence = integrity.files([feature_art, baseline_art])
    config = Path(feature['config_path']).read_text()
    schedules = quiet._schedule(config, result, manifest)
    keys = quiet._source_keys(result, manifest['expected_events'])
    baseline, baseline_metadata = quiet._read_art(baseline_art, keys, manifest, schedules)
    words, metadata = quiet._read_art(feature_art, keys, manifest, schedules)
    if any(metadata[field] != baseline_metadata[field] for field in ('parameter_set_ids', 'schedule_registry')):
        raise ValueError('feature schedule IDs/registries differ from baseline')
    if any(not np.array_equal(words[process], matrix) for process, matrix in baseline.items()):
        raise ValueError('feature original/replay packed words differ from baseline')
    _, after = prepare(feature['config_path'], feature['baseline_prepared'], baseline_art,
        feature['baseline_comparison'], expected_events=manifest['expected_events'],
        runtime_area=runtime['TRIGGER_AREA'], runtime_image=runtime['OFFLINE_IMAGE'],
        exporter_source=feature['exporter_source'], exporter_sha256=feature['exporter_sha256'])
    quiet.hybrid_history._same_files(evidence, integrity.files([feature_art, baseline_art]), 'feature decision evidence')
    if after != admitted:
        raise ValueError('decision comparison evidence changed')
    return {'events': len(keys), **metadata, 'original_packed_words_equal': True,
        'replay_packed_words_equal': True, 'physics_claim': False,
        'scope': 'saved-art decision equality only; no measured-feature or runtime claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('prepare')
    for name in ('baseline-prepared', 'baseline-art', 'baseline-comparison'):
        create.add_argument('--' + name, type=Path, required=True)
    check = commands.add_parser('preflight')
    for command in (create, check):
        for name in ('config', 'exporter-source', 'output'):
            command.add_argument('--' + name, type=Path, required=True)
        command.add_argument('--exporter-sha256', required=True)
        command.add_argument('--expected-events', type=int, required=True)
        command.add_argument('--runtime-area', required=True)
        command.add_argument('--runtime-image', required=True)
    args = parser.parse_args()
    kwargs = dict(expected_events=args.expected_events, runtime_area=args.runtime_area,
        runtime_image=args.runtime_image, exporter_source=args.exporter_source, exporter_sha256=args.exporter_sha256)
    if args.command == 'prepare':
        if args.output.exists():
            raise ValueError('refuse feature preparation overwrite')
        result, manifest = prepare(args.config, args.baseline_prepared, args.baseline_art, args.baseline_comparison, **kwargs)
        args.output.mkdir(parents=True)
        (args.output / 'replay.fcl').write_text(full_replay.render_config(result))
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        preflight(args.output, args.config.read_text(), **kwargs)
    print('source preparation only; STOP before runtime: user build and independently verified pinned library receipt required')


if __name__ == '__main__':
    main()
