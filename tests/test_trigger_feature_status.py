"""Synthetic second-layer checks; not runtime or measured-parity evidence.

Routine tests mock only the digest of named synthetic source bytes. An opt-in
R7_FEATURE_STATUS_SOURCE_DIRECTORY test checks the actual release-source copies.
No site-local source path is committed or inferred.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import feature_status as status


def ragged(flat, counts, dtype):
    return ak.unflatten(np.asarray(flat, dtype=dtype), np.asarray(counts, dtype=np.int64))


def sample(raw=None, counts=(4, 0, 4, 0)):
    """The typed feature_table.read interface, with unreached actual tracks."""
    counts = np.asarray(counts, np.int64)
    n, total = len(counts), int(np.sum(counts))
    if raw is None:
        raw = np.array([0, 16, 1 << 21, (1 << 22) | 16,
                        (1 << 23) | 16, (1 << 21) | (1 << 22),
                        (1 << 21) | (1 << 22) | (1 << 23) | 16,
                        (1 << 31) | 16], np.uint32)
    raw = np.asarray(raw, np.uint32)
    assert raw.size == total
    tracks = np.arange(total, dtype=np.uint64) - np.repeat(np.cumsum(counts) - counts, counts).astype(np.uint64)
    keys = np.column_stack((np.ones(n, np.uint32), np.ones(n, np.uint32),
                            np.arange(1, n + 1, dtype=np.uint32)))
    features = {}
    for name in status.table.selection.FEATURES:
        dtype = status.table.VECTORS[name]
        flat = np.ones(total, dtype=dtype)
        if name == 'status_has_SeedOK':
            flat = ((raw & 16) != 0).astype(np.uint8)
        elif name == 'supported_fit':
            flat = ((raw & 0xE00000) != 0).astype(np.uint8)
        features[name] = ragged(flat, counts, dtype)
    # Fixtures represent the read API, not an independent feature extraction.
    return {
        'keys': keys, 'features': features, 'track_index': ragged(tracks, counts, 'uint64'),
        'product_present': np.ones(n, bool),
        'info_present': {path: np.zeros(n, bool) for path in status.PATHS},
        'accepted_keys': {path: ragged([], np.zeros(n, np.int64), 'uint64') for path in status.PATHS},
        'metadata': {
            'raw_status': ragged(raw, counts, 'uint32'),
            'raw_status_authenticated': False, 'supported_fit_authenticated': False,
            'runtime_admitted': False, 'measured_parity_established': False,
        },
    }


@pytest.fixture
def sources(tmp_path, monkeypatch):
    directory = tmp_path / 'release-sources'
    directory.mkdir()
    synthetic = {}
    for name, digest in status.SOURCE_SHA256.items():
        content = ('synthetic source fixture: ' + name).encode()
        (directory / name).write_bytes(content)
        synthetic[content] = digest
    monkeypatch.setattr(status, '_sha256', lambda content: synthetic.get(content, hashlib.sha256(content).hexdigest()))
    return directory


def test_all_tracks_and_multiple_fit_bits_report_without_promotion(sources, monkeypatch):
    data = sample()
    before = deepcopy(data['metadata'])
    monkeypatch.setattr(ak, 'to_list', lambda *args, **kwargs: pytest.fail('no Python event materialization'))
    report = status.validate(data, sources)
    assert report['source_bound_status_consistent'] is True
    assert report['counts'] == {
        'events': 4, 'tracks': 8, 'zero_track_events': 2,
        'status_has_SeedOK': 5, 'supported_fit': 5, 'multiple_fit_bits': 2,
        'fit_branch_precedence': {'KKLoopHelix': 3, 'KKCentralHelix': 1, 'KKLine': 1},
    }
    assert report['bit_positions'] == {'SeedOK': 4, 'KKLoopHelix': 21, 'KKCentralHelix': 22, 'KKLine': 23}
    assert report['masks'] == {'SeedOK': 16, 'KKLoopHelix': 2097152, 'KKCentralHelix': 4194304, 'KKLine': 8388608}
    assert report['supported_fit_mask'] == 0xE00000
    assert report['fit_precedence'] == ['KKLoopHelix', 'KKCentralHelix', 'KKLine']
    assert report['direction_signs'] == {'downstream': 1, 'upstream': -1}
    for path in status.PATHS:
        assert report['paths'][path]['tracks_checked_in_unreached_events'] == 8
    for name, receipt in report['source_files'].items():
        assert receipt['sha256'] == status.SOURCE_SHA256[name]
        assert receipt['size_bytes'] == (sources / name).stat().st_size
    assert 'loaded-plugin attestation' in report['scope']
    assert 'independent feature extraction' in report['scope']
    assert 'runtime_admitted' not in report
    assert 'measured_parity_established' not in report
    assert ak.almost_equal(data['metadata']['raw_status'], before.pop('raw_status'))
    assert {k: v for k, v in data['metadata'].items() if k != 'raw_status'} == before
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize('counts', [(), (0, 0, 0)])
def test_typed_zero_events_and_zero_tracks(sources, counts):
    data = sample(raw=[], counts=counts)
    report = status.validate(data, sources)
    assert report['counts']['events'] == len(counts)
    assert report['counts']['tracks'] == 0
    assert report['counts']['zero_track_events'] == len(counts)


def test_bit31_and_unrelated_bits_do_not_change_predicates(sources):
    raw = np.array([0x80000000, 0x80000010, 0xFFFFFFFF, 0x001FFFFF], np.uint32)
    report = status.validate(sample(raw, (4,)), sources)
    assert report['counts']['status_has_SeedOK'] == 3
    assert report['counts']['supported_fit'] == 1
    assert report['counts']['multiple_fit_bits'] == 1


@pytest.mark.parametrize('name', ['status_has_SeedOK', 'supported_fit'])
def test_contradiction_on_nonreached_track_fails(sources, name):
    data = sample()
    flat = ak.to_numpy(ak.flatten(data['features'][name])).copy()
    flat[-1] ^= 1
    data['features'][name] = ragged(flat, (4, 0, 4, 0), 'uint8')
    with pytest.raises(ValueError, match=name + '.*contradict'):
        status.validate(data, sources)


@pytest.mark.parametrize('name', ['status_has_SeedOK', 'supported_fit'])
@pytest.mark.parametrize('dtype,value', [('uint8', 2), ('bool', True), ('int8', 1), ('uint32', 1)])
def test_predicates_require_strict_byte_booleans(sources, name, dtype, value):
    data = sample()
    data['features'][name] = ragged(np.full(8, value), (4, 0, 4, 0), dtype)
    with pytest.raises(ValueError, match=name):
        status.validate(data, sources)


@pytest.mark.parametrize('name', ['raw_status', 'status_has_SeedOK', 'supported_fit'])
@pytest.mark.parametrize('kind', ['alignment', 'option_leaf', 'option_list', 'wrong_dtype'])
def test_status_alignment_and_nonoptional_leaves(sources, name, kind):
    data = sample()
    owner = data['metadata'] if name == 'raw_status' else data['features']
    value = owner[name]
    if kind == 'alignment':
        owner[name] = ak.unflatten(ak.flatten(value), np.array([3, 1, 4, 0]))
    elif kind == 'option_leaf':
        # All-valid option masks are still forbidden, not only actual None.
        owner[name] = ak.mask(value, ak.ones_like(value, dtype=bool))
    elif kind == 'option_list':
        owner[name] = ak.mask(value, np.ones(4, bool))
    else:
        owner[name] = ak.values_astype(value, np.int64)
    with pytest.raises(ValueError, match=name):
        status.validate(data, sources)


@pytest.mark.parametrize('kind', ['option', 'wrong_dtype', 'not_contiguous'])
def test_track_index_contract_rechecked(sources, kind):
    data = sample()
    if kind == 'option':
        data['track_index'] = ak.mask(data['track_index'], np.ones(4, bool))
    elif kind == 'wrong_dtype':
        data['track_index'] = ak.values_astype(data['track_index'], np.int64)
    else:
        data['track_index'] = ragged([0, 1, 2, 3, 0, 1, 3, 2], (4, 0, 4, 0), 'uint64')
    with pytest.raises(ValueError, match='track_index'):
        status.validate(data, sources)


@pytest.mark.parametrize('path', status.PATHS)
def test_accepted_keys_ascending_with_event_boundaries_and_empty_events(sources, path):
    data = sample()
    data['info_present'][path][:] = True
    # 3 -> 0 across events is legitimate; an empty event occurs between them.
    data['accepted_keys'][path] = ragged([0, 2, 3, 0, 3], (3, 0, 2, 0), 'uint64')
    report = status.validate(data, sources)
    assert report['paths'][path]['accepted_tracks'] == 5
    assert report['accepted_keys_strictly_ascending'] is True


@pytest.mark.parametrize('path', status.PATHS)
@pytest.mark.parametrize('values,match', [([3, 1], 'strictly ascending'), ([1, 1], 'strictly ascending'),
                                         ([0, 4], 'outside')])
def test_accepted_keys_wrong_traversal_order_duplicates_or_range(sources, path, values, match):
    data = sample()
    data['info_present'][path][:] = True
    data['accepted_keys'][path] = ragged(values, (2, 0, 0, 0), 'uint64')
    with pytest.raises(ValueError, match=match):
        status.validate(data, sources)


@pytest.mark.parametrize('kind', ['option', 'wrong_dtype', 'absent_info', 'missing_path'])
def test_accepted_key_contract_fail_closed(sources, kind):
    data = sample()
    path = status.PATHS[0]
    if kind == 'option':
        data['accepted_keys'][path] = ak.mask(data['accepted_keys'][path], np.ones(4, bool))
    elif kind == 'wrong_dtype':
        data['accepted_keys'][path] = ak.values_astype(data['accepted_keys'][path], np.int64)
    elif kind == 'absent_info':
        data['accepted_keys'][path] = ragged([0], (1, 0, 0, 0), 'uint64')
    else:
        del data['accepted_keys'][path]
    with pytest.raises(ValueError):
        status.validate(data, sources)


@pytest.mark.parametrize('name', tuple(status.SOURCE_SHA256))
@pytest.mark.parametrize('change', ['missing', 'mutated'])
def test_every_required_source_rechecked_each_call(sources, name, change):
    data = sample()
    status.validate(data, sources)
    path = sources / name
    if change == 'missing':
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='source|SHA256'):
        status.validate(data, sources)


def test_absent_directory_even_for_empty_table(tmp_path):
    with pytest.raises(ValueError, match='source'):
        status.validate(sample([], ()), tmp_path / 'absent')


def test_unmocked_wrong_source_bytes_fail(tmp_path):
    for name in status.SOURCE_SHA256:
        (tmp_path / name).write_bytes(b'not the released file')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        status.validate(sample(), tmp_path)


@pytest.mark.parametrize('field', ['metadata', 'features', 'track_index', 'keys', 'accepted_keys', 'info_present'])
def test_missing_required_input_field_is_value_error(sources, field):
    data = sample()
    del data[field]
    with pytest.raises(ValueError, match=field):
        status.validate(data, sources)


def test_structural_reader_output_composes_without_changing_loader_metadata(sources, tmp_path, monkeypatch):
    # Reuse the structural-loader tests' std::vector spelling facade: uproot
    # writes real ROOT buffers but cannot persist std::vector TTree branches.
    import test_trigger_feature_table as fixtures
    keys, events, schema = fixtures.sample()
    root_path = tmp_path / 'features.root'
    fixtures.write_root(root_path, events, schema)
    monkeypatch.setattr(status.table.uproot, 'open', lambda path: fixtures._Root(path, {}))
    data = status.table.read(root_path, expected_keys=keys)
    metadata = data['metadata']
    report = status.validate(data, sources)
    assert report['counts']['tracks'] == 3
    assert report['source_bound_status_consistent'] is True
    assert data['metadata'] is metadata
    assert metadata['raw_status_authenticated'] is False
    assert metadata['supported_fit_authenticated'] is False
    assert metadata['runtime_admitted'] is False
    assert metadata['measured_parity_established'] is False


def test_actual_released_sources_if_explicitly_supplied(tmp_path):
    source_directory = os.environ.get('R7_FEATURE_STATUS_SOURCE_DIRECTORY')
    if not source_directory:
        pytest.skip('set R7_FEATURE_STATUS_SOURCE_DIRECTORY to exact released source copies')
    report = status.validate(sample(), source_directory)
    assert set(report['source_files']) == set(status.SOURCE_SHA256)
    # Exercise the real digest, then prove even whitespace mutation fails.
    for name in status.SOURCE_SHA256:
        (tmp_path / name).write_bytes((Path(source_directory) / name).read_bytes())
    mutated = tmp_path / 'TrkFitFlag.hh'
    mutated.write_bytes(mutated.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='TrkFitFlag.hh.*SHA256'):
        status.validate(sample(), tmp_path)


def test_cli_refuses_existing_output(tmp_path, monkeypatch):
    import sys
    output = tmp_path / 'report.json'
    output.write_text('preserve')
    monkeypatch.setattr(sys, 'argv', ['feature_status', '--table', 'table', '--decision-art', 'art',
        '--source-directory', 'source', '--output', str(output)])
    with pytest.raises(ValueError, match='overwrite'):
        status.main()
    assert output.read_text() == 'preserve'


def test_cli_saved_table_art_gate_and_json_report(tmp_path, monkeypatch):
    import json
    import sys
    from unittest.mock import Mock
    import event_keys
    import integrity
    import uproot
    data = sample()
    class Tree:
        classname = 'TTree'
    class Root:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def keys(self, **kwargs): return ['Events;1', 'Events;3']
        def __getitem__(self, key):
            assert key == 'Events;3'
            return Tree()
    monkeypatch.setattr(uproot, 'open', lambda path: Root())
    monkeypatch.setattr(event_keys, 'read', lambda tree: data['keys'])
    reader = Mock(return_value=data)
    monkeypatch.setattr(status.table, 'read', reader)
    check = Mock(return_value={'counts': {'tracks': 4}, 'source_bound_status_consistent': True})
    monkeypatch.setattr(status, 'validate', check)
    monkeypatch.setattr(integrity, 'files', lambda paths: [{'sha256': 'bound'}])
    output = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['feature_status', '--table', 'table', '--decision-art', 'art',
        '--source-directory', 'source', '--output', str(output)])
    status.main()
    reader.assert_called_once()
    np.testing.assert_array_equal(reader.call_args.kwargs['expected_keys'], data['keys'])
    check.assert_called_once_with(data, Path('source'))
    report = json.loads(output.read_text())
    assert report['highest_decision_cycle'] == 3
    assert not report['measured_parity_established'] and not report['runtime_admitted']
