"""Saved truth partitions preserve exact original parent keys, including explicit empty schemas."""
from pathlib import Path
import sys

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import split_source_tables


def test_split_saved_tables_by_exact_source_keys(tmp_path, monkeypatch):
    sources = [tmp_path / 'a.art', tmp_path / 'b.art']
    for path, event in zip(sources, [1, 2]):
        with uproot.recreate(path) as root:
            root.mktree('Events', {'event': 'uint32'})
            root['Events'].extend({'event': np.array([event], dtype=np.uint32)})
    monkeypatch.setattr(split_source_tables.event_keys, 'read', lambda tree:
        np.column_stack([np.ones(tree.num_entries, dtype=np.uint64)] * 2 + [tree['event'].array(library='np')]))
    tables = tmp_path / 'truth.root'
    with uproot.recreate(tables) as root:
        root.mktree('dump/particles', {'run': 'uint32', 'subrun': 'uint32', 'event': 'uint32'})
        root['dump/particles'].extend({'run': np.array([1, 1], dtype=np.uint32),
            'subrun': np.array([1, 1], dtype=np.uint32), 'event': np.array([2, 1], dtype=np.uint32)})
        root.mktree('dump/sgsteps', {'run': 'uint32', 'subrun': 'uint32', 'event': 'uint32'})
    result = split_source_tables.split(sources, [tables], tmp_path / 'split')
    assert result['events'] == 2
    with uproot.open(tmp_path / 'split/parent_1.root') as root:
        assert root['dump/particles']['event'].array(library='np').tolist() == [1]
        assert root['dump/sgsteps'].num_entries == 0


def test_subrun_metadata_uses_run_subrun_not_event(tmp_path, monkeypatch):
    sources = [tmp_path / 'a.art', tmp_path / 'b.art']
    for path, event in zip(sources, [1, 2]):
        with uproot.recreate(path) as root:
            root.mktree('Events', {'event': 'uint32'})
            root['Events'].extend({'event': np.array([event], dtype=np.uint32)})
    monkeypatch.setattr(split_source_tables.event_keys, 'read', lambda tree:
        np.column_stack([np.ones(tree.num_entries, dtype=np.uint64)] * 2 + [tree['event'].array(library='np')]))
    tables = tmp_path / 'truth.root'
    with uproot.recreate(tables) as root:
        root.mktree('dump/particles', {'run': 'uint32', 'subrun': 'uint32', 'event': 'uint32'})
        root['dump/particles'].extend({'run': np.array([1, 1], dtype=np.uint32),
            'subrun': np.array([1, 1], dtype=np.uint32), 'event': np.array([2, 1], dtype=np.uint32)})
        root.mktree('dump/subruns', {'run': 'uint32', 'subrun': 'uint32', 'primaries': 'uint64'})
        root['dump/subruns'].extend({'run': np.array([1], dtype=np.uint32),
            'subrun': np.array([1], dtype=np.uint32), 'primaries': np.array([100], dtype=np.uint64)})
    result = split_source_tables.split(sources, [tables], tmp_path / 'split')
    assert result['export_tree_rows']['subruns'] == 1
    assert 'not additive exposure' in result['subrun_metadata_policy']
    for ordinal in [1, 2]:
        with uproot.open(tmp_path / f'split/parent_{ordinal}.root') as root:
            assert root['dump/subruns']['primaries'].array(library='np').tolist() == [100]
            assert root['dump/particles']['event'].array(library='np').tolist() == [ordinal]
    assert (tmp_path / 'split/event-counts.txt').read_text() == '1\n1\n'


def test_unknown_tree_without_event_is_rejected(tmp_path, monkeypatch):
    import pytest
    source = tmp_path / 'a.art'
    with uproot.recreate(source) as root:
        root.mktree('Events', {'event': 'uint32'})
    monkeypatch.setattr(split_source_tables.event_keys, 'read', lambda tree: np.array([[1, 1, 1]], dtype=np.uint64))
    tables = tmp_path / 'truth.root'
    with uproot.recreate(tables) as root:
        root.mktree('dump/particles', {'run': 'uint32', 'subrun': 'uint32'})
    with pytest.raises(ValueError, match='identity columns'):
        split_source_tables.split([source], [tables], tmp_path / 'split')
    assert not (tmp_path / 'split').exists()


def test_foreign_truth_rows_are_rejected(tmp_path, monkeypatch):
    import pytest
    source = tmp_path / 'a.art'
    with uproot.recreate(source) as root:
        root.mktree('Events', {'event': 'uint32'})
    monkeypatch.setattr(split_source_tables.event_keys, 'read', lambda tree: np.array([[1, 1, 1]], dtype=np.uint64))
    tables = tmp_path / 'truth.root'
    with uproot.recreate(tables) as root:
        root.mktree('dump/particles', {'run': 'uint32', 'subrun': 'uint32', 'event': 'uint32'})
        root['dump/particles'].extend({key: np.array([value], dtype=np.uint32)
            for key, value in zip(['run', 'subrun', 'event'], [1, 1, 2])})
    with pytest.raises(ValueError, match='outside original parents'):
        split_source_tables.split([source], [tables], tmp_path / 'split')
    assert not (tmp_path / 'split').exists()
