"""Packed trigger-status metadata decoding and endpoint comparisons."""
import importlib.util
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

TRIGGER = Path(__file__).resolve().parents[1] / 'trigger'
sys.path.insert(0, str(TRIGGER))
spec = importlib.util.spec_from_file_location('trigger_path_state', TRIGGER / 'path_state.py')
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


def vector_bytes(words, member_version=10, vector_version=10, extra=b''):
    body = struct.pack('>HHI', 0x4000 | vector_version, member_version, len(words))
    body += np.asarray(words, dtype='>u2').tobytes() + extra
    return struct.pack('>I', 0x40000000 | len(body)) + body


def read_vector(payload):
    chunk = uproot.source.chunk.Chunk.wrap(SimpleNamespace(file_path='synthetic'), payload)
    cursor = uproot.source.cursor.Cursor(0)
    obj = state.Model_art_3a3a_HLTGlobalStatus_v10.empty()
    obj.read_members(chunk, cursor, {}, SimpleNamespace(file_path='synthetic'))
    return obj, cursor


def test_memberwise_vector_framing_and_packed_decode():
    obj, cursor = read_vector(vector_bytes([0, 5, 10, 15, 65533]))
    assert cursor.index == 22
    words = obj.member('paths_')[None, :]
    states, indices, accepted = state.decode(words)
    np.testing.assert_array_equal(states, [[0, 1, 2, 3, 1]])
    np.testing.assert_array_equal(indices, [[0, 1, 2, 3, 16383]])
    np.testing.assert_array_equal(accepted, [[True, True, False, False, True]])


def test_empty_vector_is_valid():
    obj, cursor = read_vector(vector_bytes([]))
    assert len(obj.member('paths_')) == 0 and cursor.index == 12


@pytest.mark.parametrize('options', [dict(member_version=11), dict(vector_version=9), dict(extra=b'xx')])
def test_unrecognized_framing_rejected(options):
    with pytest.raises(ValueError):
        read_vector(vector_bytes([5], **options))


def test_truncated_payload_is_rejected():
    with pytest.raises(Exception):
        read_vector(vector_bytes([5, 10])[:-1])


@pytest.mark.parametrize('value', [np.array([-1]), np.array([[-1]]), np.array([[65536]]), np.array([[1.0]])])
def test_decode_rejects_unsupported_input(value):
    with pytest.raises(ValueError):
        state.decode(value)


def test_state_difference_not_hidden_by_same_accepted_boolean():
    a = np.array([[0], [5], [6]], dtype=np.uint16)
    b = np.array([[5], [5], [5]], dtype=np.uint16)
    report = state.compare_matrices(a, b, ['p'], {'p': ['Cut']}, {'p': ['Cut']})['p']
    assert report['accepted_mismatches'] == 1
    assert report['state_mismatches'] == 2 and report['wasrun_mismatches'] == 1
    assert report['mapped_endpoint_mismatches'] == 2


def test_removed_module_indices_reported_not_excused():
    a = np.array([[13], [10]], dtype=np.uint16)
    b = np.array([[9], [6]], dtype=np.uint16)
    report = state.compare_matrices(a, b, ['p'], {'p': ['Digitize', 'Cut', 'Info']}, {'p': ['Cut', 'Info']})['p']
    assert report['state_mismatches'] == 0
    assert report['packed_status_mismatches'] == report['raw_index_mismatches'] == 2
    assert report['mapped_endpoint_mismatches'] == 0


def test_ready_and_exception_are_explicit():
    values = state.endpoints(np.array([0, 3, 1, 2]), np.array([0, 0, 2, 1]), ['Cut', 'Info'])
    assert values.tolist() == ['<ready>', '<exception-unmapped>', '<end-of-path>', 'Cut']


@pytest.mark.parametrize('s,i', [(1, 1), (2, 0), (2, 3)])
def test_invalid_completed_path_indices_fail(s, i):
    with pytest.raises(ValueError):
        state.endpoints(np.array([s]), np.array([i]), ['Cut', 'Info'])


def test_empty_accepted_path_has_zero_executed_workers():
    assert state.endpoints(np.array([1]), np.array([0]), []).tolist() == ['<end-of-path>']
    with pytest.raises(ValueError):
        state.endpoints(np.array([2]), np.array([0]), [])


def test_repeated_module_sequence_is_not_guessed():
    with pytest.raises(ValueError, match='repeated'):
        state.endpoints(np.array([1]), np.array([2]), ['Cut', 'Cut'])


def test_schema_rejects_changed_version_or_members():
    root = SimpleNamespace(file=SimpleNamespace(streamers={}))
    with pytest.raises(ValueError, match='versions'):
        state.check_schema(root)
    root.file.streamers = {'art::HLTGlobalStatus': {10: SimpleNamespace(elements=[])}}
    with pytest.raises(ValueError, match='schema'):
        state.check_schema(root)


def test_matrix_shape_mismatch_rejected():
    with pytest.raises(ValueError, match='shapes'):
        state.compare_matrices(np.zeros((2, 1), dtype=np.uint16), np.zeros((1, 1), dtype=np.uint16),
                               ['p'], {'p': []}, {'p': []})


def test_schema_metadata_highest_cycle(tmp_path):
    spec = importlib.util.spec_from_file_location('trigger_art_schema', TRIGGER / 'art_schema.py')
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    path = tmp_path / 'test.art'
    with uproot.recreate(path) as root:
        root.mktree('Events', {'TriggerResultsTest': 'uint16'})
        root['Events'].extend({'TriggerResultsTest': np.array([1], dtype=np.uint16)})
    report = schema.describe(path)
    assert report['events'] == 1 and report['highest_cycle'] == 1
    assert report['branches'][0]['name'] == 'TriggerResultsTest'
    assert report['status_streamers'] == {}
