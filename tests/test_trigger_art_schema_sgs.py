"""SGS inventory observes concrete persisted metadata without guessing product labels."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import art_schema
import integrity


class Root:
    def __init__(self, trees):
        self.trees = trees
        self.file = SimpleNamespace(streamers={})
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def keys(self, **kwargs):
        return list(self.trees)
    def __getitem__(self, name):
        return self.trees[name]


class Tree:
    classname = 'TTree'
    num_entries = 3
    def items(self, **kwargs):
        return [('mu2e::StrawGasSteps_compressDigiMCs__Digitize.',
                 SimpleNamespace(typename='art::Wrapper<vector<mu2e::StrawGasStep> >', num_entries=3)),
                ('mu2e::GenParticles_compressDigiMCs__Digitize.',
                 SimpleNamespace(typename='art::Wrapper<vector<mu2e::GenParticle> >', num_entries=3))]


def test_highest_cycle_real_labels_and_generator_metadata(tmp_path, monkeypatch):
    path = tmp_path / 'source.art'
    path.write_bytes(b'art')
    import uproot
    monkeypatch.setattr(uproot, 'open', lambda p: Root({'Events;1': object(), 'Events;3': Tree()}))
    result = art_schema.describe_sgs(path)
    assert result['highest_cycle'] == 3 and result['events'] == 3
    assert result['source'] == integrity.digest(path)
    assert any('compressDigiMCs' in row['name'] for row in result['branches'])
    assert any('GenParticles' in row['name'] for row in result['branches'])
    assert 'no payload decoding' in result['scope']


def test_mutated_input_and_unsupported_highest_cycle_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'source.art'
    path.write_bytes(b'art')
    import uproot
    class Bad(Tree):
        classname = 'RNTuple'
    monkeypatch.setattr(uproot, 'open', lambda p: Root({'Events;1': Tree(), 'Events;2': Bad()}))
    with pytest.raises(ValueError, match='highest Events'):
        art_schema.describe_sgs(path)
    class Mutated(Tree):
        def items(self, **kwargs):
            path.write_bytes(b'changed')
            return super().items(**kwargs)
    monkeypatch.setattr(uproot, 'open', lambda p: Root({'Events;1': Mutated()}))
    with pytest.raises(ValueError, match='source changed'):
        art_schema.describe_sgs(path)
