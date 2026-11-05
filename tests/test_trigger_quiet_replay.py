"""Synthetic I/O contracts for saved full-history quiet companions; no art invocation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import full_replay
from test_trigger_hybrid_history import case as history_case
import integrity
import path_state
import quiet_replay as quiet
import replay


AREA, IMAGE = '/released/Offline-v1', '/images/production.sif'


class Root:
    def __init__(self, tree):
        self.tree = tree

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __getitem__(self, name):
        assert name in ('Events', 'Events;3')
        return self.tree

    def keys(self, **kwargs):
        return ['Events;1', 'Events;3']


def dump(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def saved(tmp_path, monkeypatch):
    names = [f'p{i}' for i in range(52)]
    text = ('Digitize: { physics: { DigitizePath: [ Digi ] '
        + ' '.join(name + ': [ Digi, Cut ]' for name in names)
        + ' trigger_paths: [ DigitizePath, '
        + ', '.join(f'"{i + 10}:{name}"' for i, name in enumerate(names))
        + ' ] producers: { Digi: { module_type: TestDigi } } '
          'filters: { Cut: { module_type: TestFilter threshold: 1.0000000000000000001 } } } '
          'outputs: { Output: { SelectEvents: [ p0 ] } } '
          'services: { GeometryService: {} DbService: {} '
          'message: { destinations: { log: { categories: { ArtReport: { limit: 10 } Other: { limit: 20 } } } } } } }')
    config = tmp_path / 'config.txt'
    config.write_text(text)
    sources = [tmp_path / f'parent_{i}.art' for i in (1, 10, 2)]
    for source in sources:
        source.write_bytes(source.name.encode())
    result, manifest = replay.prepare(replay.production_config(text, 'Digitize'))
    result['source']['fileNames'] = list(map(str, sources))
    fcl = full_replay.render_config(result)
    manifest.update(config_sha256=hashlib.sha256(text.encode()).hexdigest(),
        replay_fcl_sha256=hashlib.sha256(fcl.encode()).hexdigest(), expected_events=6,
        parent_order_verified=True, input_integrity=integrity.files(sources))
    prepared = tmp_path / 'baseline'
    prepared.mkdir()
    dump(prepared / 'manifest.json', manifest)
    (prepared / 'replay.fcl').write_text(fcl)
    art = tmp_path / 'baseline.art'
    art.write_bytes(b'completed verbose art')
    keys = np.array([[1, 1, 2], [1, 1, 10], [1, 10, 1], [1, 10, 2], [1, 2, 1], [1, 2, 10]], dtype='u8')
    stock, fast = np.full((6, 53), 9, dtype='u2'), np.full((6, 52), 5, dtype='u2')
    stock[:, 0] = 5
    trees = {str(source): SimpleNamespace(keys=chunk[::-1].copy())
             for source, chunk in zip(sources, np.split(keys, 3))}
    trees[str(art)] = SimpleNamespace(keys=keys.copy(), stock=stock, fast=fast, classname='TTree', num_entries=6)
    monkeypatch.setattr(quiet.uproot, 'open', lambda path, **kw: Root(trees[str(path)]))
    monkeypatch.setattr(quiet.event_keys, 'read', lambda tree: tree.keys)
    monkeypatch.setattr(path_state, 'check_schema', lambda root: None)
    monkeypatch.setattr(path_state, 'read_status', lambda tree, process, n:
        (tree.stock, 'a' * 40) if process == 'Digitize' else (tree.fast, 'b' * 40))
    registry = {process: {'parameter_set_id': pset, 'registry_schedule_verified': True,
                          'paths': count, 'registry_blob_sha256': pset}
                for process, pset, count in [('Digitize', 'a' * 40, 53), ('TriggerReplay', 'b' * 40, 52)]}
    monkeypatch.setattr(quiet.schedule_registry, 'verify', lambda root, pset, paths:
        deepcopy(registry['Digitize' if pset == 'a' * 40 else 'TriggerReplay']))
    rows = path_state.compare_matrices(stock[:, 1:], fast, names,
        {name: ['Digi', 'Cut'] for name in names}, {name: ['Cut'] for name in names})
    comparison = dict(events=6, saved_replay_file=integrity.digest(art), decision_log_sha256='c' * 64,
        accepted_bits_crosschecked_with_log=True, root_event_ids_independently_decoded=True,
        trigger_schedule_registry_verified=True, complete_first_pass_words_preserved=True,
        complete_parent_order_verified=True, all_menu_accepted_equal=True,
        all_menu_states_and_endpoints_equal=True, config_sha256=manifest['config_sha256'],
        replay_fcl_sha256=manifest['replay_fcl_sha256'], highest_cycle=3,
        parameter_set_ids={'Digitize': 'a' * 40, 'TriggerReplay': 'b' * 40}, schedule_registry=registry,
        paths=rows, accepted_mismatches=0, state_mismatches=0, wasrun_mismatches=0, mapped_endpoint_mismatches=0)
    comparison_path = tmp_path / 'comparison.json'
    dump(comparison_path, comparison)
    return SimpleNamespace(config=config, text=text, baseline=prepared, baseline_art=art,
        comparison=comparison, comparison_path=comparison_path, manifest=manifest, result=result,
        sources=sources, keys=keys, trees=trees, tmp=tmp_path, registry=registry)


def prepare(saved, **kwargs):
    return quiet.prepare(saved.config, saved.baseline, saved.baseline_art, saved.comparison_path,
        **dict(expected_events=6, runtime_area=AREA, runtime_image=IMAGE, **kwargs))


def install(saved):
    result, manifest = prepare(saved)
    prepared = saved.tmp / 'quiet'
    prepared.mkdir()
    (prepared / 'replay.fcl').write_text(full_replay.render_config(result))
    dump(prepared / 'manifest.json', manifest)
    dump(prepared / 'replay_inventory.json', manifest['quiet_companion']['inventory'])
    art = saved.tmp / 'quiet.art'
    art.write_bytes(b'completed quiet art')
    saved.trees[str(art)] = deepcopy(saved.trees[str(saved.baseline_art)])
    (art.parent / 'replay.expanded.fcl').write_text(full_replay.render_config(result))
    return prepared, art


def log():
    return f'''image={IMAGE}
MUSE_WORK_DIR={AREA} release-pinned
TimeTracker printout (sec): Min Avg Max Median RMS nEvts
Full event 1 2 3 2 0 6
p0:Cut:TestFilter 0 1 2 1 0 6
source:RootInput(read) 0 0 0 0 0 6
DecisionOutput:DecisionProducts:RootOutput(write) 0 0 0 0 0 6
======
TrigReport Events total = 6 passed = 6 failed = 0
TimeReport CPU = 12 Real = 14
Art has completed and will exit with status 0.
'''


def test_prepare_changes_only_printer_and_existing_artreport(saved):
    before = deepcopy(saved.result)
    result, manifest = prepare(saved)
    expected = deepcopy(before)
    del expected['physics']['analyzers']['decisions']
    expected['physics']['DecisionOutput'] = ['DecisionProducts']
    expected['services']['message']['destinations']['log']['categories']['ArtReport']['limit'] = replay.Atom('0')
    assert result == expected
    assert saved.result == before
    assert result['source']['fileNames'] == list(map(str, saved.sources))
    assert result['outputs'] == before['outputs']
    assert len(manifest['quiet_companion']['inventory']['paths']) == 52
    assert manifest['quiet_companion']['inventory']['paths'][0]['sequence'] == ['Cut']
    assert manifest['quiet_companion']['baseline_evidence'][-1] == integrity.digest(saved.baseline_art)
    assert manifest['quiet_companion']['physics_claim'] is False
    assert list(saved.tmp.glob('quiet*')) == []


def test_no_message_service_is_introduced(saved):
    result = deepcopy(saved.result)
    del result['services']['message']
    assert quiet._variant(result)['services'] == result['services']


@pytest.mark.parametrize('field', ['complete_first_pass_words_preserved', 'complete_parent_order_verified',
    'all_menu_accepted_equal', 'all_menu_states_and_endpoints_equal', 'accepted_bits_crosschecked_with_log',
    'root_event_ids_independently_decoded', 'trigger_schedule_registry_verified'])
def test_stock_requires_completed_equal_full_history(saved, field):
    saved.comparison[field] = False
    dump(saved.comparison_path, saved.comparison)
    with pytest.raises(ValueError):
        prepare(saved)


@pytest.mark.parametrize('what', ['source', 'art', 'config', 'fcl', 'registry', 'path-evidence', 'count', 'raw-status'])
def test_baseline_admission_rejects_changed_evidence(saved, what):
    if what in ('source', 'art', 'config', 'fcl'):
        path = {'source': saved.sources[0], 'art': saved.baseline_art, 'config': saved.config,
                'fcl': saved.baseline / 'replay.fcl'}[what]
        path.write_bytes(path.read_bytes() + b' ')
    elif what == 'registry':
        saved.registry['Digitize']['registry_blob_sha256'] = 'changed'
    elif what == 'raw-status':
        saved.trees[str(saved.baseline_art)].fast[0, 0] = 6
    else:
        if what == 'path-evidence':
            saved.comparison['paths'].pop('p51')
        else:
            saved.comparison['events'] = 5
        dump(saved.comparison_path, saved.comparison)
    with pytest.raises(ValueError):
        prepare(saved)


def hybrid(saved, monkeypatch):
    saved.manifest['production_history_receipt'] = {'synthetic': 'receipt checked by mocked verifier'}
    saved.history_verify = Mock(return_value=saved.manifest['production_history_receipt'])
    monkeypatch.setattr(quiet.hybrid_history, 'verify', saved.history_verify)
    saved.manifest['full_production_history_certified'] = True
    dump(saved.baseline / 'manifest.json', saved.manifest)
    saved.comparison.update(full_production_history_certified=True, all_scheduled_paths_measured=True,
        validation_evidence=integrity.files([saved.baseline / 'manifest.json', saved.baseline / 'replay.fcl']))
    saved.trees[str(saved.baseline_art)].fast[0, 0] = 6
    saved.comparison['paths'] = path_state.compare_matrices(
        saved.trees[str(saved.baseline_art)].stock[:, 1:], saved.trees[str(saved.baseline_art)].fast,
        saved.manifest['paths'], {name: ['Digi', 'Cut'] for name in saved.manifest['paths']},
        {name: ['Cut'] for name in saved.manifest['paths']})
    dump(saved.comparison_path, saved.comparison)


def test_fullhybrid_allows_physics_migration_but_requires_full_history(saved, monkeypatch):
    hybrid(saved, monkeypatch)
    _, manifest = prepare(saved)
    assert manifest['quiet_companion']['baseline_kind'] == 'fullhybrid'
    saved.history_verify.assert_called_with(saved.manifest['production_history_receipt'], saved.text, [], 6)
    saved.comparison['all_scheduled_paths_measured'] = False
    dump(saved.comparison_path, saved.comparison)
    with pytest.raises(ValueError, match='hybrid evidence'):
        prepare(saved)


def test_preflight_rederives_variant_and_binds_runtime(saved):
    prepared, _ = install(saved)
    quiet.preflight(prepared, saved.text, 6, AREA, IMAGE)
    for args in [(5, AREA, IMAGE), (6, AREA + 'x', IMAGE), (6, AREA, IMAGE + 'x')]:
        with pytest.raises(ValueError):
            quiet.preflight(prepared, saved.text, *args)
    path = prepared / 'replay.fcl'
    path.write_text(path.read_text().replace('threshold: 1.0000000000000000001', 'threshold: 1'))
    manifest = json.loads((prepared / 'manifest.json').read_text())
    manifest['replay_fcl_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(prepared / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='deterministic'):
        quiet.preflight(prepared, saved.text, 6, AREA, IMAGE)


@pytest.mark.parametrize('target', ['manifest', 'comparison', 'inventory', 'source'])
def test_preflight_rechecks_all_bound_files(saved, target):
    prepared, _ = install(saved)
    path = {'manifest': saved.baseline / 'manifest.json', 'comparison': saved.comparison_path,
            'inventory': prepared / 'replay_inventory.json', 'source': saved.sources[0]}[target]
    if target == 'inventory':
        dump(path, {})
    else:
        path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        quiet.preflight(prepared, saved.text, 6, AREA, IMAGE)


@pytest.mark.parametrize('arm', ['stock', 'hybrid'])
def test_validate_without_any_printmodule_records(saved, monkeypatch, arm):
    if arm == 'hybrid':
        hybrid(saved, monkeypatch)
    prepared, art = install(saved)
    def forbidden(*args, **kwargs):
        raise AssertionError('quiet validation must never parse verbose decisions')
    monkeypatch.setattr(quiet.compare, 'read_decisions', forbidden)
    monkeypatch.setattr(quiet.compare, 'compare', forbidden)
    monkeypatch.setattr(path_state, 'compare_file', forbidden)
    report = quiet.validate(art, prepared, saved.text, log())
    assert report['events'] == 6 and report['highest_cycle'] == 3
    assert report['raw_packed_equal'] and report['original_packed_words_equal'] and report['replay_packed_words_equal']
    assert report['timing']['trigger']['total_s'] == 6
    assert report['saved_replay_file'] == integrity.digest(art)
    assert report['physics_claim'] is False
    assert 'JAX end-to-end' in ' '.join(report['caveats'])


@pytest.mark.parametrize('what', ['stock-word', 'replay-word', 'ready', 'exception', 'global-sort',
    'duplicate', 'highest-count', 'pset', 'registry'])
def test_quiet_metadata_must_match_exactly(saved, monkeypatch, what):
    prepared, art = install(saved)
    tree = saved.trees[str(art)]
    if what in ('stock-word', 'replay-word', 'ready', 'exception'):
        matrix = tree.stock if what == 'stock-word' else tree.fast
        matrix[0, 1] = {'stock-word': 10, 'replay-word': 6, 'ready': 0, 'exception': 3}[what]
    elif what == 'global-sort':
        tree.keys = tree.keys[np.lexsort((tree.keys[:, 2], tree.keys[:, 1], tree.keys[:, 0]))]
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
        quiet.validate(art, prepared, saved.text, log())


@pytest.mark.parametrize('change', ['exit', 'double-exit', 'count', 'area', 'image', 'suffix'])
def test_quiet_timing_completion_and_runtime_pins(saved, change):
    prepared, art = install(saved)
    text = log()
    text = {'exit': text.replace('status 0.', 'status 1.'),
        'double-exit': text + 'Art has completed and will exit with status 0.\n',
        'count': text.replace('0 6', '0 5').replace('total = 6 passed = 6', 'total = 5 passed = 5'),
        'area': text.replace(AREA, '/wrong'), 'image': text.replace(IMAGE, '/wrong'),
        'suffix': text.replace(IMAGE, IMAGE + '.wrong')}[change]
    with pytest.raises(ValueError):
        quiet.validate(art, prepared, saved.text, text)


@pytest.mark.parametrize('key', ['process_name', 'source', 'physics', 'outputs', 'services'])
def test_executed_expanded_fcl_exact_semantics(saved, key):
    prepared, art = install(saved)
    path = art.parent / 'replay.expanded.fcl'
    expanded = replay.parse_expanded(path.read_text())
    if key == 'process_name':
        expanded[key] = 'Wrong'
    elif key == 'physics':
        expanded[key]['filters']['Cut']['threshold'] = replay.Atom('1.0000000000000000002')
    else:
        expanded[key]['extra'] = replay.Atom('true')
    path.write_text(full_replay.render_config(expanded))
    with pytest.raises(ValueError, match='executed quiet'):
        quiet.validate(art, prepared, saved.text, log())


def test_executed_numeric_spellings_use_exact_decimal_not_float(saved):
    prepared, art = install(saved)
    path = art.parent / 'replay.expanded.fcl'
    path.write_text(path.read_text().replace('1.0000000000000000001', '10000000000000000001e-19'))
    assert quiet.validate(art, prepared, saved.text, log())['executed_configuration_verified']


@pytest.mark.parametrize('what', ['art', 'source', 'comparison', 'expanded'])
def test_validation_rechecks_bytes_after_timing(saved, monkeypatch, what):
    prepared, art = install(saved)
    original = quiet.timing.diagnose
    def mutate(*args):
        result = original(*args)
        path = {'art': art, 'source': saved.sources[0], 'comparison': saved.comparison_path,
                'expanded': art.parent / 'replay.expanded.fcl'}[what]
        path.write_bytes(path.read_bytes() + b' ')
        return result
    monkeypatch.setattr(quiet.timing, 'diagnose', mutate)
    with pytest.raises(ValueError):
        quiet.validate(art, prepared, saved.text, log())


@pytest.mark.parametrize('expected', [0, -1, True, 1.5])
def test_positive_expected_events(saved, expected):
    with pytest.raises(ValueError, match='positive'):
        quiet.prepare(saved.config, saved.baseline, saved.baseline_art, saved.comparison_path,
            expected_events=expected, runtime_area=AREA, runtime_image=IMAGE)


def test_cli_explicit_inputs_and_no_overwrite(saved, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['quiet_replay', 'prepare'])
    with pytest.raises(SystemExit):
        quiet.main()
    common = ['--config', str(saved.config), '--prepared', str(saved.tmp / 'cli-quiet'),
              '--expected-events', '6', '--runtime-area', AREA, '--runtime-image', IMAGE]
    create = ['quiet_replay', 'prepare', *common, '--baseline-prepared', str(saved.baseline),
              '--baseline-art', str(saved.baseline_art), '--baseline-comparison', str(saved.comparison_path)]
    monkeypatch.setattr(sys, 'argv', create)
    quiet.main()
    with pytest.raises(ValueError, match='overwrite'):
        quiet.main()
    monkeypatch.setattr(sys, 'argv', ['quiet_replay', 'preflight', *common])
    quiet.main()
    output = saved.tmp / 'existing.json'
    output.write_text('unchanged')
    monkeypatch.setattr(sys, 'argv', ['quiet_replay', 'validate', *common,
        '--art', 'absent.art', '--log', 'absent.log', '--output', str(output)])
    with pytest.raises(ValueError, match='overwrite'):
        quiet.main()
    assert output.read_text() == 'unchanged'


def test_quiet_raw_index_difference_is_not_waived(saved):
    prepared, art = install(saved)
    saved.trees[str(art)].fast[0, 0] = 9
    with pytest.raises(ValueError, match='packed words'):
        quiet.validate(art, prepared, saved.text, log())


def test_baseline_all52_schedule_and_verbose_printer_required(saved):
    result = deepcopy(saved.result)
    result['physics']['trigger_paths'].pop()
    with pytest.raises(ValueError, match='52'):
        quiet._schedule(saved.text, result, saved.manifest)
    result = deepcopy(saved.result)
    result['physics']['analyzers']['decisions']['module_type'] = 'OtherAnalyzer'
    with pytest.raises(ValueError, match='PrintModule'):
        quiet._variant(result)


def test_stock_aggregate_equality_does_not_hide_path_migration(saved, monkeypatch):
    hybrid(saved, monkeypatch)
    saved.manifest.pop('full_production_history_certified')
    dump(saved.baseline / 'manifest.json', saved.manifest)
    with pytest.raises(ValueError, match='decoded paths are not equal'):
        prepare(saved)


def test_reference_bytes_and_injection_runtime_pins_are_bound(saved, monkeypatch):
    hybrid(saved, monkeypatch)
    reference = saved.tmp / 'reference.art'
    reference.write_bytes(b'original stock')
    saved.manifest['reference_integrity'] = integrity.files([reference])
    saved.manifest['injection_campaign'] = {'runtime': {'TRIGGER_AREA': AREA, 'OFFLINE_IMAGE': IMAGE}}
    dump(saved.baseline / 'manifest.json', saved.manifest)
    saved.comparison['validation_evidence'] = integrity.files([
        saved.baseline / 'manifest.json', saved.baseline / 'replay.fcl'])
    dump(saved.comparison_path, saved.comparison)
    prepare(saved)
    saved.history_verify.assert_called_with(saved.manifest['production_history_receipt'], saved.text, [str(reference)], 6)
    reference.write_bytes(b'changed')
    with pytest.raises(ValueError, match='reference'):
        prepare(saved)

@pytest.mark.parametrize('copy', [False, True])
@pytest.mark.parametrize('binding', ['art', 'source', 'prepared', 'baseline', 'reference', 'campaign'])
def test_alias_identity_not_just_equal_bytes(saved, monkeypatch, binding, copy):
    if binding == 'prepared':
        hybrid(saved, monkeypatch)
    reference = saved.tmp / 'reference.art'
    reference.write_bytes(b'reference bytes')
    if binding in ('reference', 'campaign'):
        field = 'reference_integrity' if binding == 'reference' else 'injection_campaign_evidence'
        saved.manifest[field] = integrity.files([reference])
    target = {'art': saved.baseline_art, 'source': saved.sources[0],
              'prepared': saved.baseline / 'replay.fcl', 'baseline': saved.config,
              'reference': reference, 'campaign': reference}[binding]
    alias = saved.tmp / ('alias-' + target.name)
    if copy:
        alias.write_bytes(target.read_bytes())
    else:
        alias.hardlink_to(target)
    if binding == 'art':
        saved.comparison['saved_replay_file'] = integrity.digest(alias)
    elif binding == 'source':
        saved.manifest['input_integrity'][0] = integrity.digest(alias)
    elif binding == 'prepared':
        saved.comparison['validation_evidence'][1] = integrity.digest(alias)
    elif binding in ('reference', 'campaign'):
        digest = integrity.digest
        def aliased_digest(path):
            return digest(alias if Path(path) == target else path)
        monkeypatch.setattr(integrity, 'digest', aliased_digest)
    dump(saved.baseline / 'manifest.json', saved.manifest)
    if binding == 'prepared':
        saved.comparison['validation_evidence'][0] = integrity.digest(saved.baseline / 'manifest.json')
    dump(saved.comparison_path, saved.comparison)
    if binding == 'baseline':
        prepared, _ = install(saved)
        manifest = json.loads((prepared / 'manifest.json').read_text())
        manifest['quiet_companion']['baseline_evidence'][0] = integrity.digest(alias)
        dump(prepared / 'manifest.json', manifest)
    if copy:
        with pytest.raises(ValueError, match='bytes, paths or order'):
            if binding == 'baseline':
                quiet.preflight(prepared, saved.text, 6, AREA, IMAGE)
            else:
                prepare(saved)
    elif binding == 'baseline':
        quiet.preflight(prepared, saved.text, 6, AREA, IMAGE)
    else:
        prepared, art = install(saved)
        quiet.preflight(prepared, saved.text, 6, AREA, IMAGE)
        assert quiet.validate(art, prepared, saved.text, log())['saved_replay_file'] == integrity.digest(art)


def test_actual_history_is_reverified_at_prepare_and_preflight(saved, history_case, monkeypatch):
    history = quiet.hybrid_history
    case = history_case
    case.comparison_path = saved.tmp / 'history-comparison.json'
    digest = hashlib.sha256(saved.text.encode()).hexdigest()
    case.manifest.update(config_sha256=digest, original_processed_events=6)
    case.comparison.update(config_sha256=digest, regenerated_events=6)
    for parent, candidate, chunk in zip(case.parents, case.candidates, np.split(saved.keys, 2)):
        chunk = chunk[np.lexsort((chunk[:, 2], chunk[:, 1], chunk[:, 0]))]
        case.roots[parent.name] = chunk.tolist()
        case.roots[candidate.name] = chunk.tolist()
    dump(case.comparison_path, case.comparison)
    case.save()
    history_open = history.regeneration_census.uproot.open
    history_read = history.regeneration_census.event_keys.read
    monkeypatch.setattr(quiet.uproot, 'open', lambda path, **kw:
        Root(saved.trees[str(path)]) if str(path) in saved.trees else history_open(path))
    monkeypatch.setattr(quiet.event_keys, 'read', lambda tree:
        tree.keys if hasattr(tree, 'keys') else history_read(tree))
    receipt = history.certify(saved.text, case.manifest_path, case.directory, case.comparison_path, 6)
    actual_verify = history.verify
    hybrid(saved, monkeypatch)
    monkeypatch.setattr(history, 'verify', actual_verify)
    saved.manifest.update(production_history_receipt=receipt, reference_integrity=integrity.files(case.candidates))
    dump(saved.baseline / 'manifest.json', saved.manifest)
    saved.comparison['validation_evidence'] = integrity.files([
        saved.baseline / 'manifest.json', saved.baseline / 'replay.fcl'])
    dump(saved.comparison_path, saved.comparison)
    prepared, _ = install(saved)
    quiet.preflight(prepared, saved.text, 6, AREA, IMAGE)
    case.parents[0].write_bytes(b'altered underlying production history')
    with pytest.raises(ValueError, match='history evidence'):
        prepare(saved)
    with pytest.raises(ValueError, match='history evidence'):
        quiet.preflight(prepared, saved.text, 6, AREA, IMAGE)


def test_repeated_prepare_evidence_accepts_same_inode_alias(saved, monkeypatch):
    alias = saved.tmp / 'config-alias'
    alias.hardlink_to(saved.config)
    evidence = quiet._evidence
    calls = 0
    def alternating(*args):
        nonlocal calls
        calls += 1
        rows = evidence(*args)
        if calls % 2 == 0:
            rows[0] = integrity.digest(alias)
        return rows
    monkeypatch.setattr(quiet, '_evidence', alternating)
    prepared, _ = install(saved)
    quiet.preflight(prepared, saved.text, 6, AREA, IMAGE)


@pytest.mark.parametrize('binding', ['art', 'prepared'])
def test_repeated_validation_evidence_accepts_same_inode_alias(saved, monkeypatch, binding):
    prepared, art = install(saved)
    target = art if binding == 'art' else prepared / 'replay_inventory.json'
    alias = saved.tmp / 'validation-alias'
    alias.hardlink_to(target)
    digest = integrity.digest
    calls = 0
    def alternating(path):
        nonlocal calls
        if Path(path) == target:
            calls += 1
            if calls % 2 == 0:
                return digest(alias)
        return digest(path)
    monkeypatch.setattr(integrity, 'digest', alternating)
    report = quiet.validate(art, prepared, saved.text, log())
    assert calls >= 2
    assert report['saved_replay_file'] == digest(art)
