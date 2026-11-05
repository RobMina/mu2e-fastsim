"""A parent filename does not imply that all contained events share its nominal subrun."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import regeneration_census as census


class Root:
    def __init__(self, keys):
        self.tree = SimpleNamespace(classname='TTree', num_entries=len(keys), values=np.asarray(keys, dtype=np.uint64))
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __getitem__(self, name):
        assert name == 'Events'
        return self.tree


def test_split_file_matches_real_parent_keys_not_filename_subrun(tmp_path, monkeypatch):
    for index in [1, 2]:
        (tmp_path / f'unfiltered_{index}.art').touch()
    values = {'parent_a': [[1430, 9000, 7], [1430, 5063, 3]],
              'parent_b': [[1430, 6000, 4]],
              'unfiltered_1.art': [[1430, 5063, 3], [1430, 9000, 7]],
              'unfiltered_2.art': [[1430, 6000, 4]]}
    monkeypatch.setattr(census.uproot, 'open', lambda path: Root(values[Path(path).name]))
    monkeypatch.setattr(census.event_keys, 'read', lambda tree: tree.values)
    manifest = {'split_input_files': True, 'parents': [{'did': 'mu2e:dts.test.001430_00005063.art'},
                  {'did': 'mu2e:dts.test.001430_00006000.art'}],
                'local_parent_paths': ['parent_a', 'parent_b'], 'original_processed_events': 3}
    result = census.census(manifest, tmp_path, 3)
    assert result['equal'] and result['complete_parent_event_ids_verified']
    values['unfiltered_1.art'][1][1] = 5063
    with pytest.raises(ValueError, match='parent event IDs'):
        census.census(manifest, tmp_path, 3)


def test_parent_prefix_does_not_certify_complete_history(tmp_path, monkeypatch):
    (tmp_path / 'unfiltered_1.art').touch()
    values = {'parent': [[1430, 5, 1], [1430, 7, 2]], 'unfiltered_1.art': [[1430, 5, 1]]}
    monkeypatch.setattr(census.uproot, 'open', lambda path: Root(values[Path(path).name]))
    monkeypatch.setattr(census.event_keys, 'read', lambda tree: tree.values)
    manifest = {'split_input_files': True, 'parents': [{'did': 'mu2e:dts.test.art'}],
                'local_parent_paths': ['parent'], 'original_processed_events': 2}
    result = census.census(manifest, tmp_path, 1)
    assert result['parent_prefix_event_ids_verified'] and not result['complete_parent_event_ids_verified']
