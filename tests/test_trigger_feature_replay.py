"""Synthetic feature-export admission/decision contracts; no build, art or feature I/O."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import feature_replay as feature
import full_replay
import integrity
import path_state
import quiet_replay as quiet
import replay
from test_trigger_quiet_replay import AREA, IMAGE, dump, hybrid, saved


SOURCE = Path(feature.__file__).resolve().parent / 'art/R7Trigger/src/KalSeedSelectionFeatures_module.cc'


@pytest.fixture
def fast(saved, monkeypatch):
    hybrid(saved, monkeypatch)
    return saved


def options():
    return dict(expected_events=6, runtime_area=AREA, runtime_image=IMAGE,
        exporter_source=SOURCE, exporter_sha256=integrity.digest(SOURCE)['sha256'])


def prepare(saved, **kwargs):
    return feature.prepare(saved.config, saved.baseline, saved.baseline_art, saved.comparison_path,
        **(options() | kwargs))


def install(saved):
    result, manifest = prepare(saved)
    prepared = saved.tmp / 'features'
    prepared.mkdir()
    (prepared / 'replay.fcl').write_text(full_replay.render_config(result))
    dump(prepared / 'manifest.json', manifest)
    art = saved.tmp / 'features.art'
    art.write_bytes(b'completed feature art')
    saved.trees[str(art)] = deepcopy(saved.trees[str(saved.baseline_art)])
    return prepared, art, manifest


def test_only_analyzer_and_new_tfileservice_change_verbose_baseline(fast):
    before = deepcopy(fast.result)
    evidence = integrity.files([fast.baseline / 'manifest.json', fast.baseline / 'replay.fcl', fast.baseline_art])
    result, manifest = prepare(fast)
    expected = deepcopy(before)
    expected['physics']['analyzers'][feature.LABEL] = {
        'module_type': 'KalSeedSelectionFeatures', 'kalSeeds': 'TrigKSFDe::TriggerReplay',
        'selectionInfo': ['TprTrkDe80m70pD0200KSFilter::TriggerReplay', 'TprTrkDe80m70pKSFilter::TriggerReplay']}
    expected['physics']['DecisionOutput'] = ['decisions', feature.LABEL, 'DecisionProducts']
    expected['services']['TFileService'] = {'fileName': 'r7-selection-features.root'}
    assert result == expected and fast.result == before
    assert result['outputs'] == before['outputs']
    assert result['source'] == before['source']
    assert len(replay.paths(result)) == 52
    assert result['physics']['end_paths'] == ['DecisionOutput']
    assert result['physics']['analyzers']['decisions'] == before['physics']['analyzers']['decisions']
    assert integrity.files([fast.baseline / 'manifest.json', fast.baseline / 'replay.fcl', fast.baseline_art]) == evidence
    assert manifest['status'] == 'source_preparation_only'
    assert manifest['build_required'] and not manifest['runtime_ready']
    assert manifest['feature_export']['source_files'] == integrity.files([
        SOURCE.parent.parent / name for name in feature.SOURCE_FILES])
    assert manifest['feature_export']['source_binding_url'] == feature.SOURCE_URL
    assert not manifest['feature_export']['measured_parity']
    assert 'quiet_companion' not in manifest
    assert not list(fast.tmp.glob('features*'))


def test_actual_production_devnull_tfileservice_changes_only_filename(fast, monkeypatch):
    original = replay.parse_expanded
    def with_service(text):
        result = original(text)
        if 'process_name' in result:
            result['services']['TFileService'] = {'fileName': '/dev/null', 'service_type': 'TFileService'}
        return result
    monkeypatch.setattr(replay, 'parse_expanded', with_service)
    result, manifest = prepare(fast)
    assert result['services']['TFileService'] == {'fileName': 'r7-selection-features.root', 'service_type': 'TFileService'}
    assert manifest['build_required'] and not manifest['runtime_ready']


def test_stock_and_incomplete_hybrid_are_not_admitted(saved, monkeypatch):
    with pytest.raises(ValueError, match='FULLHYBRID'):
        prepare(saved)
    hybrid(saved, monkeypatch)
    saved.comparison['all_scheduled_paths_measured'] = False
    dump(saved.comparison_path, saved.comparison)
    with pytest.raises(ValueError, match='hybrid evidence'):
        prepare(saved)


@pytest.mark.parametrize('what', ['service', 'selected-output', 'output-file', 'output-commands', 'label'])
def test_reject_baseline_changes_without_touching_its_files(fast, what):
    original = replay.parse_expanded
    def altered(text):
        result = original(text)
        if 'process_name' not in result:
            return result
        if what == 'service':
            result['services']['TFileService'] = {'fileName': 'existing.root', 'other': replay.Atom('12')}
        elif what == 'label':
            result['physics']['analyzers'][feature.LABEL] = {'module_type': 'Other'}
        else:
            key, value = {'selected-output': ('SelectEvents', ['p0']),
                'output-file': ('fileName', 'other.art'),
                'output-commands': ('outputCommands', ['drop *'])}[what]
            result['outputs']['DecisionProducts'][key] = value
        return result
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as patch:
        patch.setattr(replay, 'parse_expanded', altered)
        with pytest.raises(ValueError):
            prepare(fast)


@pytest.mark.parametrize('what', ['source', 'art', 'comparison', 'config', 'prepared-fcl', 'registry'])
def test_preflight_rechecks_baseline_evidence(fast, what):
    prepared, _, _ = install(fast)
    if what == 'registry':
        fast.registry['TriggerReplay']['registry_blob_sha256'] = 'changed'
    else:
        path = {'source': fast.sources[0], 'art': fast.baseline_art,
            'comparison': fast.comparison_path, 'config': fast.config,
            'prepared-fcl': fast.baseline / 'replay.fcl'}[what]
        path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        feature.preflight(prepared, fast.text, **options())


@pytest.mark.parametrize('what', ['threshold', 'manifest', 'runtime', 'source-sha', 'owner-source', 'expected'])
def test_preflight_determinism_and_explicit_pins(fast, what):
    prepared, _, _ = install(fast)
    kwargs = options()
    if what == 'threshold':
        path = prepared / 'replay.fcl'
        path.write_text(path.read_text().replace('1.0000000000000000001', '1'))
    elif what == 'manifest':
        path = prepared / 'manifest.json'
        manifest = quiet._load(path)
        manifest['runtime_ready'] = True
        dump(path, manifest)
    else:
        key, value = {'runtime': ('runtime_image', IMAGE + '.other'), 'source-sha': ('exporter_sha256', '0' * 64),
            'owner-source': ('exporter_source', fast.tmp / 'owner/src/KalSeedSelectionFeatures_module.cc'),
            'expected': ('expected_events', 5)}[what]
        kwargs[key] = value
    with pytest.raises(ValueError):
        feature.preflight(prepared, fast.text, **kwargs)


def test_saved_art_comparison_is_exact_and_not_runtime_validation(fast):
    prepared, art, manifest = install(fast)
    feature.preflight(prepared, fast.text, **options())
    report = feature.compare_decisions(art, fast.baseline_art, manifest)
    assert report['events'] == 6 and report['highest_cycle'] == 3
    assert report['original_packed_words_equal'] and report['replay_packed_words_equal']
    assert not report['physics_claim'] and 'runtime claim' in report['scope']
    assert manifest['build_required'] and not manifest['runtime_ready']


@pytest.mark.parametrize('what', ['original-word', 'replay-word', 'raw-index', 'ready', 'exception',
    'order', 'duplicate', 'highest-count', 'pset', 'registry'])
def test_saved_decisions_reject_every_identity_state_and_endpoint_change(fast, monkeypatch, what):
    _, art, manifest = install(fast)
    tree = fast.trees[str(art)]
    if what in ('original-word', 'replay-word', 'raw-index', 'ready', 'exception'):
        matrix = tree.stock if what == 'original-word' else tree.fast
        matrix[0, 1] = {'original-word': 10, 'replay-word': 6, 'raw-index': 9, 'ready': 0, 'exception': 3}[what]
    elif what == 'order':
        tree.keys = tree.keys[::-1]
    elif what == 'duplicate':
        tree.keys[0] = tree.keys[1]
    elif what == 'highest-count':
        tree.num_entries = 5
    elif what == 'pset':
        original = path_state.read_status
        monkeypatch.setattr(path_state, 'read_status', lambda t, process, n:
            (original(t, process, n)[0], 'd' * 40) if t is tree else original(t, process, n))
    else:
        original = quiet.schedule_registry.verify
        monkeypatch.setattr(quiet.schedule_registry, 'verify', lambda root, pset, paths:
            {**original(root, pset, paths), 'registry_blob_sha256': 'changed'}
            if root.tree is tree else original(root, pset, paths))
    with pytest.raises(ValueError):
        feature.compare_decisions(art, fast.baseline_art, manifest)


def test_cli_source_only_no_wrapper_no_overwrite(fast, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['feature_replay', 'prepare'])
    with pytest.raises(SystemExit):
        feature.main()
    output = fast.tmp / 'cli-features'
    common = ['--config', str(fast.config), '--output', str(output), '--expected-events', '6',
        '--runtime-area', AREA, '--runtime-image', IMAGE, '--exporter-source', str(SOURCE),
        '--exporter-sha256', options()['exporter_sha256']]
    monkeypatch.setattr(sys, 'argv', ['feature_replay', 'prepare', *common,
        '--baseline-prepared', str(fast.baseline), '--baseline-art', str(fast.baseline_art),
        '--baseline-comparison', str(fast.comparison_path)])
    feature.main()
    assert sorted(path.name for path in output.iterdir()) == ['manifest.json', 'replay.fcl']
    with pytest.raises(ValueError, match='overwrite'):
        feature.main()
    monkeypatch.setattr(sys, 'argv', ['feature_replay', 'preflight', *common])
    feature.main()
    assert 'STOP before runtime' in capsys.readouterr().out
    monkeypatch.setattr(sys, 'argv', ['feature_replay', 'validate', *common])
    with pytest.raises(SystemExit):
        feature.main()
