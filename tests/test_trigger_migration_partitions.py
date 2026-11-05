"""Saved parent slices preserve traversal/counter history; never independent SEM replicas."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import hybrid_replay
import integrity
import migration_partitions as partitions
import path_state


def summarize(stock, fast, counts, names=('p',), stock_sequences=None, fast_sequences=None):
    return partitions.summarize(np.asarray(stock), np.asarray(fast), names,
        stock_sequences or {'p': ['Digi', 'Cut']}, fast_sequences or {'p': ['Cut']}, counts)


def test_contiguous_parent_traversal_1_10_2_and_unequal_counts():
    # These are source-parent boundaries, not sorted subruns or counter resets.
    stock = np.array([[9], [10], [10], [9], [9], [10]], dtype=np.uint16)
    fast = np.array([[6], [5], [6], [5], [6], [6]], dtype=np.uint16)
    report = summarize(stock, fast, [2, 1, 3])
    rows = report['partitions']
    assert [(r['start'], r['stop'], r['events']) for r in rows] == [(0, 2, 2), (2, 3, 1), (3, 6, 3)]
    assert [r['paths']['p']['stock_pass'] for r in rows] == [1, 0, 2]
    assert [r['paths']['p']['delta_fraction'] for r in rows] == [0, 0, -1 / 3]
    total = report['aggregate']['paths']['p']
    assert {field: total[field] for field in partitions.MIGRATIONS} == {
        'stock_pass': 3, 'replay_pass': 2, 'stock_only': 2, 'replay_only': 1, 'mismatches': 3}
    assert total['delta_fraction'] == -1 / 6
    assert total['accepted_mismatches'] == total['state_mismatches'] == 3
    assert total['raw_index_mismatches'] == total['packed_status_mismatches'] == 6
    assert total['mapped_endpoint_mismatches'] == 3
    assert total['stock_states'] == {'Ready': 0, 'Pass': 3, 'Fail': 3, 'Exception': 0}
    assert total['wasrun_mismatches'] == 0
    partitions._check_sums(report['aggregate'], rows)
    bad = copy.deepcopy(rows)
    bad[0]['paths']['p']['stock_states']['Pass'] += 1
    with pytest.raises(ValueError, match='partition counts'):
        partitions._check_sums(report['aggregate'], bad)
    bad = copy.deepcopy(rows)
    bad[0]['events'] += 1
    with pytest.raises(ValueError, match='partition event'):
        partitions._check_sums(report['aggregate'], bad)


def test_zero_pass_empty_collection_failure_and_empty_accepted_path():
    # A first-filter collection veto is a completed Fail, not Ready; [] can Pass at index 0.
    words = np.array([[6, 1], [6, 1]], dtype=np.uint16)
    sequences = {'empty_collection': ['Cut'], 'empty_path': []}
    report = summarize(words, words, [1, 1], list(sequences), sequences, sequences)
    row = report['aggregate']['paths']['empty_collection']
    assert row['stock_pass'] == row['replay_pass'] == row['mismatches'] == 0
    assert row['delta_fraction'] == 0
    assert row['stock_states']['Fail'] == 2
    assert report['aggregate']['paths']['empty_path']['stock_pass'] == 2


@pytest.mark.parametrize('counts', [[], [0, 2], [-1, 3], [1], [3], [1.0, 1], [True, 1], [np.bool_(True), 1]])
def test_invalid_parent_counts(counts):
    with pytest.raises(ValueError, match='positive integer'):
        summarize([[9], [9]], [[5], [5]], counts)


@pytest.mark.parametrize('names', [[], ['p', 'p'], [''], [1]])
def test_invalid_names(names):
    with pytest.raises(ValueError, match='nonempty unique'):
        summarize([[9]], [[5]], [1], names)


@pytest.mark.parametrize('word', [0, 3, 7, -1, 65536, 5.0, 5, 2, 14])
def test_ready_exception_malformed_words_and_invalid_execution_counters(word):
    # stock Pass must finish both workers; Fail must execute 1..2 workers.
    with pytest.raises(ValueError):
        summarize([[word]], [[5]], [1])


def test_shapes_and_empty_event_population_rejected():
    for stock, fast, counts in [([9], [[5]], [1]), ([[9]], [[5], [5]], [1]),
                                (np.empty((0, 1), dtype='u2'), np.empty((0, 1), dtype='u2'), [])]:
        with pytest.raises(ValueError):
            summarize(stock, fast, counts)


class Root:
    def __init__(self, tree):
        self.tree = tree
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __getitem__(self, name):
        assert name == 'Events'
        return self.tree


@pytest.fixture
def saved(tmp_path, monkeypatch):
    names = [f'p{i}' for i in range(52)]
    config = ('Digitize: { physics: { DigitizePath: [ Digi ] '
        + ' '.join(name + ': [ Digi, Cut ]' for name in names)
        + ' trigger_paths: [ DigitizePath, '
        + ', '.join(f'"{i + 10}:{name}"' for i, name in enumerate(names))
        + ' ] producers: { Digi: { module_type: TestDigi } } '
          'filters: { Cut: { module_type: TestFilter } } } '
          'outputs: { Output: { SelectEvents: [ p0 ] } } '
          'services: { GeometryService: {} DbService: {} } }')
    config_path = tmp_path / 'config.fcl'
    config_path.write_text(config)
    sources = [tmp_path / f'parent_{i}.art' for i in (1, 10, 2)]
    reference = tmp_path / 'reference.art'
    for path in [*sources, reference]:
        path.write_bytes(path.name.encode())
    admission = {'events': 6, 'same_event_handoff_metadata_passed': True,
        'reference_files': integrity.files([reference]), 'candidate_files': integrity.files(sources),
        'config_sha256': hashlib.sha256(config.encode()).hexdigest(),
        'straw_digi_tag': 'Fast::Injected', 'calo_digi_tag': None}
    fcl, manifest = hybrid_replay.prepare(config, [reference], sources, admission, 6)
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    (prepared / 'replay.fcl').write_text(fcl)
    art = tmp_path / 'saved.art'
    art.write_bytes(b'saved completed art')
    keys = np.array([[1430, 1, 2], [1430, 1, 10], [1430, 10, 1],
                     [1430, 2, 1], [1430, 2, 2], [1430, 2, 10]], dtype='u8')
    stock = np.full((6, 53), 9, dtype='u2')
    stock[:, 0] = 5  # DigitizePath is not a physics partition column.
    fast = np.full((6, 52), 5, dtype='u2')
    stock[1, 1] = 10
    fast[0, 0] = 6
    stock[:, 2] = 10
    fast[:, 1] = 6
    trees = {str(path): SimpleNamespace(keys=values[::-1].copy())
             for path, values in zip(sources, (keys[:2], keys[2:3], keys[3:]))}
    trees[str(art)] = SimpleNamespace(keys=keys.copy(), stock=stock, fast=fast,
                                    classname='TTree', num_entries=6)
    monkeypatch.setattr(partitions.uproot, 'open', lambda path, **kw: Root(trees[str(path)]))
    monkeypatch.setattr(partitions.event_keys, 'read', lambda tree: tree.keys)
    monkeypatch.setattr(path_state, 'check_schema', lambda root: None)
    monkeypatch.setattr(path_state, 'read_status', lambda tree, process, n:
        (tree.stock, 'stock-id') if process == 'Digitize' else (tree.fast, 'fast-id'))
    registry = {process: {'parameter_set_id': pset, 'registry_schedule_verified': True}
                for process, pset in [('Digitize', 'stock-id'), ('TriggerReplay', 'fast-id')]}
    monkeypatch.setattr(partitions.schedule_registry, 'verify', lambda root, pset, paths:
        next(row for row in registry.values() if row['parameter_set_id'] == pset))
    seq_a, seq_b = {name: ['Digi', 'Cut'] for name in names}, {name: ['Cut'] for name in names}
    aggregate = partitions.summarize(stock[:, 1:], fast, names, seq_a, seq_b, [2, 1, 3])['aggregate']
    comparison = {'events': 6, 'validation_evidence': integrity.files([prepared / 'manifest.json', prepared / 'replay.fcl']),
        'saved_replay_file': integrity.digest(art), 'full_production_history_certified': False,
        'all_scheduled_paths_measured': True, 'accepted_bits_crosschecked_with_log': True,
        'root_event_ids_independently_decoded': True, 'trigger_schedule_registry_verified': True,
        'parameter_set_ids': {'Digitize': 'stock-id', 'TriggerReplay': 'fast-id'}, 'schedule_registry': registry,
        'decision_migrations': {name: {**{field: row[field] for field in partitions.MIGRATIONS},
            'paired_efficiency_difference': row['delta_fraction']} for name, row in aggregate['paths'].items()},
        'paths': path_state.compare_matrices(stock[:, 1:], fast, names, seq_a, seq_b)}
    comparison_path = tmp_path / 'comparison.json'
    comparison_path.write_text(json.dumps(comparison))
    return SimpleNamespace(art=art, config=config_path, prepared=prepared, comparison_path=comparison_path,
        comparison=comparison, sources=sources, reference=reference, trees=trees)


def analyze(saved):
    return partitions.analyze(saved.art, saved.config, saved.prepared, saved.comparison_path)


def test_saved_analysis_preserves_source_order_and_digest_evidence(saved):
    report = analyze(saved)
    assert report['events'] == 6 and len(report['aggregate']['paths']) == 52
    assert [row['source'] for row in report['partitions']] == [str(path) for path in saved.sources]
    assert [row['events'] for row in report['partitions']] == [2, 1, 3]
    assert report['aggregate']['paths']['p0']['stock_only'] == 1
    evidence = {row['path'] for row in report['inputs']}
    assert evidence == {str(path) for path in [saved.art, saved.config, saved.comparison_path,
        saved.prepared / 'manifest.json', saved.prepared / 'replay.fcl', *saved.sources, saved.reference]}
    assert 'no independent error bars' in ' '.join(report['limitations'])
    assert 'no counter reset' in report['scope']


@pytest.mark.parametrize('section,field', [('decision_migrations', field) for field in
    (*partitions.MIGRATIONS, 'paired_efficiency_difference')] + [('paths', field) for field in
    ('accepted_mismatches', 'state_mismatches', 'wasrun_mismatches', 'raw_index_mismatches',
     'packed_status_mismatches', 'mapped_endpoint_mismatches', 'stock_states', 'replay_states')])
def test_every_path_aggregate_is_checked_not_only_totals(saved, section, field):
    rows = saved.comparison[section]
    # Transfer a count between paths without altering the all-path total.
    if isinstance(rows['p0'][field], dict):
        rows['p0'][field]['Pass'] += 1
        rows['p1'][field]['Pass'] -= 1
    else:
        rows['p0'][field] += 1
        rows['p1'][field] -= 1
    saved.comparison_path.write_text(json.dumps(saved.comparison))
    with pytest.raises(ValueError, match='aggregate.*p0'):
        analyze(saved)


@pytest.mark.parametrize('target', ['art', 'config', 'source', 'reference', 'manifest', 'fcl', 'evidence', 'completion', 'registry'])
def test_changed_bytes_or_incomplete_evidence_rejected(saved, target):
    if target in ('evidence', 'completion', 'registry'):
        if target == 'evidence':
            saved.comparison['validation_evidence'][0]['path'] = '/other/manifest.json'
        elif target == 'completion':
            saved.comparison['all_scheduled_paths_measured'] = False
        else:
            saved.comparison['parameter_set_ids']['Digitize'] = 'changed'
        saved.comparison_path.write_text(json.dumps(saved.comparison))
    else:
        path = {'art': saved.art, 'config': saved.config, 'source': saved.sources[0],
                'reference': saved.reference, 'manifest': saved.prepared / 'manifest.json',
                'fcl': saved.prepared / 'replay.fcl'}[target]
        path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        analyze(saved)


@pytest.mark.parametrize('change', ['global-sort', 'duplicate-source', 'duplicate-art', 'comparison-count', 'empty-parent'])
def test_identity_and_parent_boundary_failures(saved, change):
    if change == 'global-sort':
        tree = saved.trees[str(saved.art)]
        tree.keys = tree.keys[np.lexsort((tree.keys[:, 2], tree.keys[:, 1], tree.keys[:, 0]))]
    elif change == 'duplicate-source':
        saved.trees[str(saved.sources[1])].keys[0] = saved.trees[str(saved.sources[0])].keys[0]
    elif change == 'duplicate-art':
        saved.trees[str(saved.art)].keys[0] = saved.trees[str(saved.art)].keys[1]
    elif change == 'empty-parent':
        saved.trees[str(saved.sources[1])].keys = np.empty((0, 3), dtype='u8')
    else:
        saved.comparison['events'] = 5
        saved.comparison_path.write_text(json.dumps(saved.comparison))
    with pytest.raises(ValueError):
        analyze(saved)


def test_cli_requires_explicit_inputs_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['migration_partitions'])
    with pytest.raises(SystemExit):
        partitions.main()
    output = tmp_path / 'exists.json'
    output.write_text('unchanged')
    monkeypatch.setattr(sys, 'argv', ['migration_partitions', '--art', 'a', '--config', 'c',
        '--prepared', 'p', '--comparison', 'r', '--output', str(output)])
    with pytest.raises(ValueError, match='overwrite'):
        partitions.main()
    assert output.read_text() == 'unchanged'
