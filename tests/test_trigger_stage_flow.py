"""Synthetic saved-only admission and one-based executed-worker stage diagnostics."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import integrity
import path_state
import replay
import stage_flow as flow


SA = ['Digi', 'Seed', 'SharedFit', 'Cut', 'Audit']
SB = ['Seed', 'SharedFit', 'Cut', 'Audit']


def summary(stock, fast, names=('p',), sa=None, sb=None, filters=('Seed', 'Cut')):
    return flow.summarize(np.asarray(stock), np.asarray(fast), names,
                          sa or {'p': SA}, sb or {'p': SB}, filters)


def test_one_based_reach_returns_and_selected_pair_counts():
    # SharedFit and trailing Audit are producers. Pass includes a producer suffix.
    stock = [[21], [10], [18], [18], [10], [21], [10], [21]]
    fast = [[6], [17], [14], [17], [6], [14], [14], [17]]
    row = summary(stock, fast)['p']
    assert set(row['filters']) == {'Seed', 'Cut'}
    cut = row['filters']['Cut']
    assert (cut['stock_position'], cut['fast_position']) == (4, 3)
    assert cut['stock']['counts'] == {'reached': 5, 'returned_pass': 3, 'returned_fail': 2}
    assert cut['fast']['counts'] == {'reached': 6, 'returned_pass': 3, 'returned_fail': 3}
    assert cut['reach']['counts'] == {'neither': 1, 'both': 4, 'stock_only': 1, 'fast_only': 2}
    assert cut['both_reached']['counts'] == {'stock_pass': 2, 'fast_pass': 2,
        'both_pass': 1, 'both_fail': 1, 'lost': 1, 'gained': 1, 'mismatch': 2}
    assert cut['stock']['event_fractions']['returned_pass'] == 3 / 8
    assert cut['stock']['conditional_on_reached']['pass_fraction'] == 3 / 5
    assert cut['both_reached']['event_fractions']['lost'] == 1 / 8
    assert cut['both_reached']['conditional_on_both_reached']['fractions']['lost'] == 1 / 4
    seed = row['filters']['Seed']
    assert seed['stock']['counts']['reached'] == seed['fast']['counts']['reached'] == 8
    assert seed['both_reached']['counts']['lost'] == 1
    assert seed['both_reached']['counts']['gained'] == 2
    assert {k: row['final_path'][k] for k in flow.MIGRATIONS} == {
        'stock_pass': 3, 'replay_pass': 3, 'stock_only': 2, 'replay_only': 2, 'mismatches': 4}


def test_no_reach_zero_pass_and_empty_population_keep_null_denominators():
    cut = summary([[10], [10]], [[6], [6]])['p']['filters']['Cut']
    assert cut['stock']['counts'] == {'reached': 0, 'returned_pass': 0, 'returned_fail': 0}
    assert cut['reach']['counts']['neither'] == 2
    assert cut['stock']['event_fractions']['returned_pass'] == 0
    assert cut['stock']['conditional_on_reached']['pass_fraction'] is None
    assert all(value is None for value in cut['both_reached']['conditional_on_both_reached']['fractions'].values())
    empty = summary(np.empty((0, 1), dtype='u2'), np.empty((0, 1), dtype='u2'))['p']
    assert empty['filters']['Seed']['stock']['event_fractions']['reached'] is None
    assert empty['final_path']['paired_efficiency_difference'] is None
    json.dumps(empty, allow_nan=False)


def test_filtering_endpoint_and_unequal_unshared_sequences():
    sa = {'p': ['Removed', 'OnlyStock', 'Seed', 'SharedFit', 'Cut']}
    sb = {'p': ['Seed', 'Inserted', 'SharedFit', 'Cut']}
    row = summary([[22]], [[18]], sa=sa, sb=sb)['p']['filters']['Cut']
    assert row['stock']['counts'] == row['fast']['counts'] == {'reached': 1, 'returned_pass': 0, 'returned_fail': 1}
    assert row['both_reached']['counts']['both_fail'] == 1
    # A valid empty path can Pass at 0; no invented stages.
    assert summary([[1]], [[1]], sa={'p': []}, sb={'p': []})['p']['filters'] == {}


@pytest.mark.parametrize('word', [0, 3, 7, -1, 65536, 21.0, True, 1, 5, 17, 2, 26])
def test_invalid_dtype_state_and_executed_count(word):
    with pytest.raises(ValueError):
        summary([[word]], [[17]])


@pytest.mark.parametrize('stock,fast', [([21], [[17]]), ([[21]], [[17], [17]]),
    ([[21, 21]], [[17, 17]]), ([[complex(21)]], [[17]])])
def test_invalid_matrix_shapes(stock, fast):
    with pytest.raises(ValueError):
        summary(stock, fast)


@pytest.mark.parametrize('names', [[], ['p', 'p'], [''], [1]])
def test_invalid_path_names(names):
    with pytest.raises(ValueError):
        summary([[21]], [[17]], names=names)


@pytest.mark.parametrize('filters', ['Seed', ['Cut', 'Cut'], [1]])
def test_invalid_filter_labels(filters):
    with pytest.raises(ValueError):
        summary([[21]], [[17]], filters=filters)


def test_repeated_modules_missing_sequence_and_index_validate_even_without_filters():
    for sa in ({'p': ['Digi', 'Seed', 'Seed', 'Cut', 'Audit']}, {'wrong': SA}):
        with pytest.raises(ValueError):
            summary([[21]], [[17]], sa=sa)
    with pytest.raises(ValueError, match='index'):
        summary([[17]], [[17]], filters=[])


class Root:
    def __init__(self, tree):
        self.tree = tree
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def keys(self, recursive=False):
        return ['Events;1', 'Events;3', 'RootFileDB;1']
    def __getitem__(self, name):
        assert name == 'Events;3'  # Explicitly select highest cycle.
        return self.tree


@pytest.fixture
def saved(tmp_path, monkeypatch):
    names = [f'p{i}' for i in range(52)]
    paths_a = [('0', 'DigitizePath', ['Digi'])] + [(str(i + 10), name, SA) for i, name in enumerate(names)]
    paths_b = [(str(i + 10), name, SB) for i, name in enumerate(names)]
    filters = {'Seed': {'module_type': 'SeedFilter'}, 'Cut': {'module_type': 'CutFilter'}}
    producers = {label: {'module_type': 'NotAFilter'} for label in ('Digi', 'SharedFit', 'Audit')}
    config_a = {'physics': {'trigger_paths': [f'{bit}:{name}' for bit, name, _ in paths_a],
        **{name: seq for _, name, seq in paths_a}, 'filters': filters, 'producers': producers}}
    config_b = {'physics': {'trigger_paths': [f'{bit}:{name}' for bit, name, _ in paths_b],
        **{name: seq for _, name, seq in paths_b}, 'filters': filters, 'producers': producers},
        'source': {'fileNames': ['/absent/history/parent_1.art', '/absent/history/parent_10.art']}}
    text = 'Digitize: ' + replay.render(config_a) + '\n'
    fcl = '\n'.join(f'{key}: {replay.render(value)}' for key, value in config_b.items()) + '\n'
    config = tmp_path / 'config.txt'
    config.write_text(text)
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'replay.fcl').write_text(fcl)
    manifest = {'original_process': 'Digitize', 'replay_process': 'TriggerReplay',
        'paths': names, 'bits': [str(i + 10) for i in range(52)], 'expected_events': 3,
        'config_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'replay_fcl_sha256': hashlib.sha256(fcl.encode()).hexdigest(),
        'full_production_history_certified': True,
        'reference_integrity': [{'path': '/absent/stock/reference.art'}]}
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    art = tmp_path / 'decisions.art'
    art.write_bytes(b'completed art bytes, ROOT decoded via a mock seam')
    stock = np.full((3, 53), 21, dtype='u2')
    stock[:, 0] = 5
    fast = np.full((3, 52), 17, dtype='u2')
    stock[1, 1] = 10
    fast[0, 0] = 6
    # A zero-pass, zero-Cut-reach path must remain in all52 output.
    stock[:, 2] = 10
    fast[:, 1] = 6
    tree = SimpleNamespace(stock=stock, fast=fast, keys=np.array([[1, 10, 3], [1, 2, 1], [1, 2, 2]], dtype='u8'),
                           classname='TTree', num_entries=3)
    opened = []
    def open_saved(path, **kwargs):
        opened.append(Path(path))
        assert Path(path) == art  # Source history is never reopened.
        return Root(tree)
    monkeypatch.setattr(flow.uproot, 'open', open_saved)
    monkeypatch.setattr(flow.event_keys, 'read', lambda tree: tree.keys)
    monkeypatch.setattr(flow.path_state, 'check_schema', lambda root: None)
    monkeypatch.setattr(flow.path_state, 'read_status', lambda tree, process, n:
        (tree.stock, 'stock-id') if process == 'Digitize' else (tree.fast, 'fast-id'))
    registry = {process: {'parameter_set_id': pset, 'registry_schedule_verified': True}
        for process, pset in [('Digitize', 'stock-id'), ('TriggerReplay', 'fast-id')]}
    monkeypatch.setattr(flow.schedule_registry, 'verify', lambda root, pset, paths:
        next(value for value in registry.values() if value['parameter_set_id'] == pset))
    rows = flow.summarize(stock[:, 1:], fast, names, {n: SA for n in names}, {n: SB for n in names}, filters)
    comparison = {'events': 3, 'highest_cycle': 3,
        **dict.fromkeys(flow.COMPLETION_FLAGS, True),
        'saved_replay_file': integrity.digest(art),
        'validation_evidence': integrity.files([prepared / 'manifest.json', prepared / 'replay.fcl']),
        'parameter_set_ids': {'Digitize': 'stock-id', 'TriggerReplay': 'fast-id'}, 'schedule_registry': copy.deepcopy(registry),
        'paths': path_state.compare_matrices(stock[:, 1:], fast, names, {n: SA for n in names}, {n: SB for n in names}),
        'decision_migrations': {n: {field: row['final_path'][field]
            for field in (*flow.MIGRATIONS, 'paired_efficiency_difference')} for n, row in rows.items()}}
    comparison_path = tmp_path / 'comparison.json'
    comparison_path.write_text(json.dumps(comparison))
    return SimpleNamespace(art=art, config=config, prepared=prepared, comparison_path=comparison_path,
        comparison=comparison, tree=tree, manifest=manifest, opened=opened)


def analyze(saved):
    return flow.analyze(saved.art, saved.config, saved.prepared, saved.comparison_path)


def test_saved_all52_admission_never_recensuses_absent_sources(saved):
    report = analyze(saved)
    assert report['events'] == 3 and report['highest_cycle'] == 3
    assert len(report['paths']) == 52 and saved.opened == [saved.art]
    assert report['paths']['p1']['filters']['Cut']['stock']['conditional_on_reached']['pass_fraction'] is None
    assert set(report['paths']['p0']['filters']) == {'Seed', 'Cut'}
    assert report['completed_comparison_reconciled'] is True
    assert {row['path'] for row in report['inputs']} == {str(path) for path in [saved.art, saved.config,
        saved.prepared / 'manifest.json', saved.prepared / 'replay.fcl', saved.comparison_path]}
    assert 'selected diagnostic' in ' '.join(report['limitations'])
    assert 'SEM' in ' '.join(report['limitations'])


@pytest.mark.parametrize('target', ['art', 'config', 'manifest', 'fcl', 'evidence', 'pset', 'registry',
    'events', 'highest_cycle', *flow.COMPLETION_FLAGS])
def test_saved_admission_rejects_changed_bytes_receipts_counts_registry(saved, target):
    if target in ('art', 'config', 'manifest', 'fcl'):
        path = {'art': saved.art, 'config': saved.config, 'manifest': saved.prepared / 'manifest.json',
                'fcl': saved.prepared / 'replay.fcl'}[target]
        path.write_bytes(path.read_bytes() + b' ')
    else:
        if target == 'evidence':
            saved.comparison['validation_evidence'][0]['sha256'] = 'changed'
        elif target == 'pset':
            saved.comparison['parameter_set_ids']['Digitize'] = 'changed'
        elif target == 'registry':
            saved.comparison['schedule_registry']['Digitize']['registry_schedule_verified'] = False
        elif target in ('events', 'highest_cycle'):
            saved.comparison[target] += 1
        else:
            saved.comparison[target] = False
        saved.comparison_path.write_text(json.dumps(saved.comparison))
    with pytest.raises(ValueError):
        analyze(saved)


@pytest.mark.parametrize('field', [*flow.MIGRATIONS, 'paired_efficiency_difference',
    'accepted_mismatches', 'state_mismatches', 'wasrun_mismatches', 'raw_index_mismatches',
    'packed_status_mismatches', 'mapped_endpoint_mismatches', 'stock_states', 'replay_states'])
def test_every_final_path_field_is_reconciled(saved, field):
    section = 'decision_migrations' if field in (*flow.MIGRATIONS, 'paired_efficiency_difference') else 'paths'
    row = saved.comparison[section]['p0']
    if isinstance(row[field], dict):
        row[field]['Pass'] += 1
    else:
        row[field] += 1
    saved.comparison_path.write_text(json.dumps(saved.comparison))
    with pytest.raises(ValueError, match='final path'):
        analyze(saved)


@pytest.mark.parametrize('target', ['duplicate-id', 'zero-id', 'wrong-id-count', 'stock-count', 'fast-count',
    'producer-fail', 'short-pass', 'ready', 'exception', 'tree-type', 'tree-count'])
def test_root_structural_and_endpoint_failures(saved, target):
    tree = saved.tree
    if target == 'duplicate-id':
        tree.keys[0] = tree.keys[1]
    elif target == 'zero-id':
        tree.keys[0, 0] = 0
    elif target == 'wrong-id-count':
        tree.keys = tree.keys[:-1]
    elif target == 'stock-count':
        tree.stock = tree.stock[:-1]
    elif target == 'fast-count':
        tree.fast = tree.fast[:, :-1]
    elif target == 'tree-type':
        tree.classname = 'NotTTree'
    elif target == 'tree-count':
        tree.num_entries = 4
    else:
        tree.stock[0, 1] = {'producer-fail': 14, 'short-pass': 17, 'ready': 0, 'exception': 3}[target]
    with pytest.raises(ValueError):
        analyze(saved)


def test_cli_explicit_inputs_fresh_only_and_success(saved, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['stage_flow'])
    with pytest.raises(SystemExit):
        flow.main()
    output = tmp_path / 'report.json'
    args = ['stage_flow', '--art', str(saved.art), '--config', str(saved.config),
            '--prepared', str(saved.prepared), '--comparison', str(saved.comparison_path), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', args)
    flow.main()
    original = output.read_bytes()
    assert len(json.loads(original)['paths']) == 52
    with pytest.raises(ValueError, match='overwrite'):
        flow.main()
    assert output.read_bytes() == original
    output.unlink()
    output.symlink_to(tmp_path / 'absent-target')
    with pytest.raises(ValueError, match='overwrite'):
        flow.main()
    output.unlink()
    # Fail closed on a race occurring after the early existence check.
    monkeypatch.setattr(flow, 'analyze', lambda *args: (output.write_text('racer') or {}))
    with pytest.raises(FileExistsError):
        flow.main()
    assert output.read_text() == 'racer'


def test_only_labels_classified_filter_in_both_configs_are_stages(saved):
    fcl_path = saved.prepared / 'replay.fcl'
    config = replay.parse_expanded(fcl_path.read_text())
    config['physics']['producers']['Cut'] = config['physics']['filters'].pop('Cut')
    fcl = '\n'.join(f'{key}: {replay.render(value)}' for key, value in config.items()) + '\n'
    fcl_path.write_text(fcl)
    saved.manifest['replay_fcl_sha256'] = hashlib.sha256(fcl.encode()).hexdigest()
    (saved.prepared / 'manifest.json').write_text(json.dumps(saved.manifest))
    saved.comparison['validation_evidence'] = integrity.files([saved.prepared / 'manifest.json', fcl_path])
    saved.comparison_path.write_text(json.dumps(saved.comparison))
    report = analyze(saved)
    assert set(report['paths']['p0']['filters']) == {'Seed'}
    assert set(report['paths']['p51']['filters']) == {'Seed'}


@pytest.mark.parametrize('section', ['paths', 'decision_migrations'])
def test_completed_comparison_must_keep_all_zero_paths(saved, section):
    del saved.comparison[section]['p1']
    saved.comparison_path.write_text(json.dumps(saved.comparison))
    with pytest.raises(ValueError, match='all52'):
        analyze(saved)


def test_one_arm_never_reaches_and_last_filter_pass_followed_by_producer():
    row = summary([[10], [10]], [[17], [14]])['p']['filters']['Cut']
    assert row['reach']['counts'] == {'neither': 0, 'both': 0, 'stock_only': 0, 'fast_only': 2}
    assert row['stock']['conditional_on_reached']['pass_fraction'] is None
    assert row['fast']['conditional_on_reached']['pass_fraction'] == 0.5
    assert row['both_reached']['conditional_on_both_reached']['fractions']['fast_pass'] is None


def test_aggregate_completed_state_receipt_is_checked(saved):
    saved.comparison['accepted_mismatches'] = 999
    saved.comparison_path.write_text(json.dumps(saved.comparison))
    with pytest.raises(ValueError, match='aggregate'):
        analyze(saved)


def test_all_executed_endpoints_against_small_explicit_stage_oracle():
    # Every completed endpoint for these paths: only filters can reject; the
    # producer suffix is counted in the executed-worker Pass index.
    stock = np.array([[a] for a in (10, 18, 21) for _ in range(3)], dtype='u2')
    fast = np.array([[b] for _ in range(3) for b in (6, 14, 17)], dtype='u2')
    rows = summary(stock, fast)['p']['filters']
    for label, pa, pb in [('Seed', 2, 1), ('Cut', 4, 3)]:
        ra = np.array([int(word) >> 2 >= pa for word in stock[:, 0]])
        rb = np.array([int(word) >> 2 >= pb for word in fast[:, 0]])
        qa = np.array([(int(word) & 3) == 1 or int(word) >> 2 > pa for word in stock[:, 0]]) & ra
        qb = np.array([(int(word) & 3) == 1 or int(word) >> 2 > pb for word in fast[:, 0]]) & rb
        row = rows[label]
        assert row['stock']['counts']['reached'] == int(ra.sum())
        assert row['stock']['counts']['returned_pass'] == int(qa.sum())
        assert row['fast']['counts']['returned_pass'] == int(qb.sum())
        assert row['both_reached']['counts']['lost'] == int((ra & rb & qa & ~qb).sum())
        assert sum(row['reach']['counts'].values()) == 9
        for arm in ('stock', 'fast'):
            counts = row[arm]['counts']
            assert counts['returned_pass'] + counts['returned_fail'] == counts['reached']


def test_copied_evidence_requires_ordered_exact_bytes_not_site_path():
    actual = [{'path': 'local/a', 'bytes': 1, 'sha256': 'a'},
              {'path': 'local/b', 'bytes': 2, 'sha256': 'b'}]
    saved = [dict(row, path='original/' + str(i)) for i, row in enumerate(actual)]
    assert flow._same_bytes(saved, actual)
    assert not flow._same_bytes(saved[::-1], actual)
    assert not flow._same_bytes(saved[:1], actual)
    assert not flow._same_bytes(None, actual)
    assert not flow._same_bytes([None, saved[1]], actual)
    saved[0]['sha256'] = 'changed'
    assert not flow._same_bytes(saved, actual)
