"""Serialized basket boundaries and fixed EventAuxiliary identity decoding."""
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import event_keys
import serialized


def events():
    raw = np.zeros((2, 78), dtype=np.uint8)
    for offset, expected in event_keys.HEADERS:
        raw[:, offset:offset + len(expected)] = np.frombuffer(expected, dtype=np.uint8)
    raw[:, 47:59] = np.array([[1430, 55, 1], [1430, 55, 3]], dtype='>u4').view(np.uint8).reshape(2, 12)
    return raw


def test_fixed_nested_event_identity_layout():
    assert event_keys.decode(ak.Array(events())).tolist() == [[1430, 55, 1], [1430, 55, 3]]


@pytest.mark.parametrize('offset', [0, 4, 6, 12, 29, 33, 35, 39, 41, 45, 59, 63])
def test_header_corruption_fails_closed(offset):
    raw = events()
    raw[0, offset] ^= 1
    with pytest.raises(ValueError, match='framing'):
        event_keys.decode(ak.Array(raw))


def test_unknown_auxiliary_size_fails():
    with pytest.raises(ValueError, match='size'):
        event_keys.decode(ak.Array(events()[:, :-1]))


def test_duplicate_ids_fail():
    raw = events()
    raw[1] = raw[0]
    with pytest.raises(ValueError, match='duplicate'):
        event_keys.decode(ak.Array(raw))


def test_basket_fixed_width_boundaries(tmp_path):
    path = tmp_path / 'test.root'
    with uproot.recreate(path) as root:
        root.mktree('Events', {'values': 'uint32'})
        root['Events'].extend({'values': np.array([1, 3, 7], dtype=np.uint32)})
    with uproot.open(path) as root:
        raw = serialized.entry_bytes(root['Events']['values'])
        assert ak.to_numpy(raw).tobytes() == np.array([1, 3, 7], dtype='>u4').tobytes()
        assert len(serialized.entry_bytes(root['Events']['values'], 2)) == 2


def test_malformed_basket_offsets_fail():
    class Branch:
        num_baskets = 1
        def basket_entry_start_stop(self, i):
            return 0, 1
        def basket(self, i):
            return SimpleNamespace(byte_offsets=np.array([1, 3]), num_entries=1, data=np.array([1, 2, 3], dtype=np.uint8))
    with pytest.raises(ValueError, match='boundaries'):
        serialized.entry_bytes(Branch())


def test_jagged_byte_comparison_preserves_order_lengths_and_exact_bits():
    a = ak.Array([[1, 2], [], [3], [4, 5], [255]])
    b = ak.Array([[1, 2], [], [3, 0], [5, 4], [254]])
    assert serialized.serialized_equal(a, b).tolist() == [True, True, False, False, False]
    with pytest.raises(ValueError, match='entry counts'):
        serialized.serialized_equal(a, b[:1])


def fake_root(keys, flip=False, absent=False):
    class Tree(dict):
        def items(self, recursive=False):
            return super().items()
    class Parent(dict):
        def __init__(self, name, values):
            self.name = name
            self.values = values
            self.typename = 'synthetic value-only product'
        def array(self, library):
            return np.full(len(keys), not absent, dtype=bool)
        def __getitem__(self, key):
            return self
    class Root(dict):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    tree = Tree()
    tree.keys_for_test = np.asarray(keys, dtype=np.uint64)
    for token in serialized.TOKENS:
        name = f'mu2e::{token}_Digitize.'
        values = [[int(key[2]), 1 if flip and key[2] == 3 else 0] for key in keys]
        tree[name] = Parent(name, ak.Array(values))
    return Root(Events=tree)


def test_full_product_comparison_joins_event_keys_and_catches_bytes(monkeypatch):
    a = fake_root([[1430, 55, 1], [1430, 55, 3]])
    b = fake_root([[1430, 55, 3], [1430, 55, 1]])
    roots = {'a': a, 'b': b}
    monkeypatch.setattr(serialized.uproot, 'open', lambda name: roots[name])
    monkeypatch.setattr(event_keys, 'read', lambda tree: tree.keys_for_test)
    monkeypatch.setattr(serialized, 'entry_bytes', lambda branch: branch.values)
    assert serialized.compare_products('a', 'b', 2)['equal']
    roots['b'] = fake_root([[1430, 55, 3], [1430, 55, 1]], flip=True)
    report = serialized.compare_products('a', 'b', 2)
    assert not report['equal'] and report['root_event_ids_verified']
    assert all(row['mismatches'] == 1 for row in report['products'].values())
    with pytest.raises(ValueError, match='overlap'):
        serialized.compare_products('a', 'b', 1)
    roots['b'] = fake_root([[1430, 55, 3]], absent=True)
    with pytest.raises(ValueError, match='absent'):
        serialized.compare_products('a', 'b', 1)


def test_auxiliary_count_must_match_events_tree(monkeypatch):
    class Branch:
        typename = 'art::EventAuxiliary'
    class Tree(dict):
        num_entries = 3
    tree = Tree(EventAuxiliary=Branch())
    monkeypatch.setattr(event_keys, 'entry_bytes', lambda branch: ak.Array(events()))
    with pytest.raises(ValueError, match='tree count'):
        event_keys.read(tree)


def test_serialized_missing_basket_entries_fail():
    class Branch:
        num_baskets = 1
        num_entries = 2
        def basket_entry_start_stop(self, i):
            return 0, 1
        def basket(self, i):
            return SimpleNamespace(byte_offsets=np.array([0, 1]), num_entries=1, data=np.array([1], dtype=np.uint8))
    with pytest.raises(ValueError, match='coverage'):
        serialized.entry_bytes(Branch())


def test_multi_file_serialized_comparison_requires_all_bytes():
    keys = np.array([[1, 1, 1], [1, 2, 1]], dtype=np.uint64)
    products = {token: ak.Array([[1, 2], [3]]) for token in serialized.TOKENS}
    types = {token: 'synthetic' for token in serialized.TOKENS}
    assert serialized.compare_product_sets((keys, products, types), (keys, products, types), 2)['equal']
    changed = dict(products)
    changed[serialized.TOKENS[0]] = ak.Array([[1], [3]])
    result = serialized.compare_product_sets((keys, products, types), (keys, changed, types), 2)
    assert not result['equal'] and result['products'][serialized.TOKENS[0]]['mismatches'] == 1
