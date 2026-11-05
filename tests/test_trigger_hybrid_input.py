"""Same-event handoff metadata rejects dropped/changed original products without claiming fast physics."""
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import hybrid_input


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


class Branch:
    def __init__(self, values, present=None, typename='SyntheticValue'):
        self.values = ak.Array(values)
        self.present = np.asarray([True] * len(values) if present is None else present)
        self.typename = typename
    def array(self, **kwargs):
        return self.present


class Parent:
    def __init__(self, branch):
        self.branch = branch
    def __getitem__(self, name):
        return self.branch


class Tree(dict):
    classname = 'TTree'
    def keys(self, recursive=False):
        return super().keys()


def tree(keys, inject=False):
    value = Tree()
    value.keys_for_test = np.array(keys, dtype=np.uint64)
    value.num_entries = len(keys)
    value.words = np.array([[5, 9] if key[2] == 1 else [5, 10] for key in keys], dtype=np.uint16)
    for friendly, label in hybrid_input.STOCK:
        name = f'{friendly}_{label}__Digitize'
        value[name + '.'] = Parent(Branch([[key[2], 42] for key in keys]))
    if inject:
        for friendly in ['mu2e::StrawDigis', 'mu2e::StrawDigiADCWaveforms']:
            value[f'{friendly}_Fast__Injected.'] = Parent(Branch([[1, 2] for key in keys]))
    return value


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    paths = [tmp_path / 'reference.art', tmp_path / 'candidate.art']
    for path in paths:
        path.write_bytes(b'file metadata fixture')
    a = tree([[1430, 55, 1], [1430, 55, 3]])
    b = tree([[1430, 55, 3], [1430, 55, 1]], inject=True)
    roots = dict(zip(paths, [a, b]))
    monkeypatch.setattr(hybrid_input.uproot, 'open', lambda path, **kwargs: Root(roots[Path(path)]))
    monkeypatch.setattr(hybrid_input.event_keys, 'read', lambda value: value.keys_for_test)
    monkeypatch.setattr(hybrid_input.serialized, 'entry_bytes', lambda branch: branch.values)
    monkeypatch.setattr(hybrid_input.path_state, 'check_schema', lambda root: None)
    monkeypatch.setattr(hybrid_input.path_state, 'read_status', lambda value, *args: (value.words, 'schedule-id'))
    monkeypatch.setattr(hybrid_input.schedule_registry, 'verify', lambda *args: {'registry_schedule_verified': True})
    config = 'Digitize:{physics:{DigitizePath:[Digi] p:[Digi,Cut] trigger_paths:[DigitizePath,"10:p"]}}'
    return SimpleNamespace(paths=paths, a=a, b=b, config=config)


def run(fixture, **kwargs):
    return hybrid_input.audit([fixture.paths[0]], [fixture.paths[1]], fixture.config, 2,
                              straw_tag=kwargs.pop('straw_tag', 'Fast::Injected'), **kwargs)


def test_full_unique_population_and_all_original_bytes_preserved(fixture):
    report = run(fixture)
    assert report['events'] == 2 and report['same_event_handoff_metadata_passed']
    assert len(report['files'][0]['stock_products']) == 6
    assert len(report['files'][0]['injected_products']) == 2
    assert report['files'][0]['original_statuses_preserved']
    assert 'no fast physics' in report['scope'] and report['prerequisites_not_certified']


@pytest.mark.parametrize('change', ['stock-byte', 'stock-presence', 'injected-presence', 'identity', 'status', 'type'])
def test_changed_or_missing_context_fails(fixture, change):
    stock = fixture.b['mu2e::CaloDigis_CaloDigiMaker__Digitize.'].branch
    if change == 'stock-byte':
        stock.values = ak.Array([[3, 42], [1, 43]])
    elif change == 'stock-presence':
        stock.present[0] = False
    elif change == 'injected-presence':
        fixture.b['mu2e::StrawDigis_Fast__Injected.'].branch.present[0] = False
    elif change == 'identity':
        fixture.b.keys_for_test[0, 2] = 4
    elif change == 'status':
        fixture.b.words[0, 1] += 1
    else:
        stock.typename = 'OtherType'
    with pytest.raises(ValueError):
        run(fixture)


@pytest.mark.parametrize('tag', ['Fast', 'makeSD::Digitize', 'Fast:bad:bad:tag'])
def test_implicit_or_original_substitution_tag_refused(fixture, tag):
    with pytest.raises(ValueError, match='tag'):
        run(fixture, straw_tag=tag)


def test_extra_same_label_product_cannot_shadow_preserved_stock_context(fixture):
    fixture.b['mu2e::ProtonBunchTime_EWMProducer__Injected.'] = Parent(Branch([[1], [1]]))
    with pytest.raises(ValueError, match='shadows'):
        run(fixture)


def test_injected_collection_type_must_match_stock_detector_type(fixture):
    fixture.b['mu2e::StrawDigis_Fast__Injected.'].branch.typename = 'WrongCollection'
    with pytest.raises(ValueError, match='injected product type'):
        run(fixture)


def test_empty_serialized_bytes_are_not_present_empty_collections(fixture):
    fixture.b['mu2e::StrawDigis_Fast__Injected.'].branch.values = ak.Array([[], [1]])
    with pytest.raises(ValueError, match='serialized payload'):
        run(fixture)
