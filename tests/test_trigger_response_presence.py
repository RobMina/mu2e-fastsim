"""Metadata-only SGS admission rejects absent, ambiguous or malformed persisted products."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import response_presence as presence


TAG = 'compressDetStepMCs::Compress'
STEM = 'mu2e::StrawGasSteps_compressDetStepMCs__Compress'


class Branch:
    def __init__(self, typename, values, count=2):
        self.typename, self.values, self.num_entries = typename, values, count

    def array(self, **kwargs):
        assert kwargs == {'library': 'np'}
        assert self.typename == 'bool', 'physics objects must not be deserialized'
        return self.values


class Parent(dict):
    typename = 'art::Wrapper<std::vector<mu2e::StrawGasStep> >'
    num_entries = 2

    def keys(self, recursive=False):
        return super().keys()


class Tree(dict):
    classname = 'TTree'
    num_entries = 2

    def keys(self, recursive=False):
        return super().keys()


def tree():
    result = Tree()
    result.identities = np.array([[1430, 55, 9], [1430, 55, 2]], dtype=np.uint64)
    result[STEM + '.'] = Parent({
        STEM + '.present': Branch('bool', np.array([True, True])),
        STEM + '.obj': Branch('std::vector<mu2e::StrawGasStep>', ak.Array([[0, 255, 41], [0, 0, 0]])),
    })
    return result


class Root:
    def __init__(self, trees):
        self.trees = trees
        self.file = SimpleNamespace(streamers={
            'mu2e::StrawGasStep': {3: SimpleNamespace(elements=[SimpleNamespace(name='_sp', typename='art::Ptr<mu2e::SimParticle>')])},
            'Unrelated': {1: SimpleNamespace(elements=[])},
        })

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def keys(self, recursive=False):
        return self.trees.keys()

    def __getitem__(self, key):
        return self.trees[key]


@pytest.fixture
def sample(tmp_path, monkeypatch):
    source = tmp_path / 'source.art'
    source.write_bytes(b'opaque persisted art fixture')
    latest = tree()
    stale = tree()
    stale[STEM + '.'][STEM + '.present'].values[:] = False
    roots = {source: Root({'Events;1': stale, 'Events;7': latest})}
    monkeypatch.setattr(presence.uproot, 'open', lambda path: roots[Path(path)])
    monkeypatch.setattr(presence.event_keys, 'read', lambda value: value.identities)
    monkeypatch.setattr(presence.serialized, 'entry_bytes', lambda branch: branch.values)
    return SimpleNamespace(source=source, tree=latest, roots=roots,
                           parent=latest[STEM + '.'], stale=stale)


def test_discovery_highest_cycle_and_explicit_verification(sample):
    report = presence.inspect(sample.source, 2)
    assert report['highest_cycle'] == 7
    assert report['tag'] == TAG
    assert report['product']['process'] == 'Compress'
    assert report['product']['branch'] == STEM + '.'
    assert report['all_present'] and report['product']['serialized_entries_nonempty']
    assert report['event_keys_physical'] == [[1430, 55, 9], [1430, 55, 2]]
    assert report['event_keys_numeric'] == [[1430, 55, 2], [1430, 55, 9]]
    assert report['source_before'] == report['source_after']
    assert len(report['source_before']['sha256']) == 64
    assert report['streamers']['mu2e::StrawGasStep']['3'][0]['typename'].startswith('art::Ptr')
    assert 'Unrelated' not in report['streamers']
    assert report['simparticles_presence_checked'] is False
    assert 'no SGS numerical values' in report['limits'][0]
    assert presence.verify(sample.source, report, 2, tag=TAG) == report
    with pytest.raises(TypeError):
        presence.verify(sample.source, report)
    with pytest.raises(ValueError, match='exact SGS tag'):
        presence.verify(sample.source, report, tag='compressDetStepMCs::Digitize')


def test_opaque_object_entries_allow_present_empty_framing_without_claiming_sizes(sample):
    sample.parent[STEM + '.obj'].values = ak.Array([[0], [0]])
    sample.tree['mu2e::SimParticles_compressDetStepMCs__Compress.'] = object()
    report = presence.inspect(sample.source, tag=TAG)
    assert report['all_present']
    assert report['simparticles_candidates'] == ['mu2e::SimParticles_compressDetStepMCs__Compress.']
    assert report['simparticles_presence_checked'] is False
    assert 'collection sizes' in report['limits'][0]
    assert 'SimParticles candidates are metadata only' in report['limits'][2]


@pytest.mark.parametrize('values', [np.array([True, False]), np.array([1, 1]),
    np.array([1.0, 1.0]), np.array([True, True], dtype=object), np.array([[True], [True]]),
    np.array([True]), np.ma.array([True, True], mask=[False, True])])
def test_exact_boolean_present_and_all_events_required(sample, values):
    sample.parent[STEM + '.present'].values = values
    with pytest.raises(ValueError, match='absent SGS|exact boolean'):
        presence.inspect(sample.source)


@pytest.mark.parametrize('change', ['parent-count', 'present-count', 'object-count', 'wrapper-type',
    'object-type', 'empty-bytes', 'byte-count', 'missing-object', 'missing-present', 'missing-parent'])
def test_malformed_or_missing_product_fails_closed(sample, change):
    if change == 'parent-count':
        sample.parent.num_entries = 1
    elif change == 'present-count':
        sample.parent[STEM + '.present'].num_entries = 1
    elif change == 'object-count':
        sample.parent[STEM + '.obj'].num_entries = 1
    elif change == 'wrapper-type':
        sample.parent.typename = 'art::Wrapper<std::vector<mu2e::StepPointMC> >'
    elif change == 'object-type':
        sample.parent[STEM + '.obj'].typename = 'std::vector<mu2e::StepPointMC>'
    elif change == 'empty-bytes':
        sample.parent[STEM + '.obj'].values = ak.Array([[], [42]])
    elif change == 'byte-count':
        sample.parent[STEM + '.obj'].values = ak.Array([[42]])
    elif change == 'missing-object':
        del sample.parent[STEM + '.obj']
    elif change == 'missing-present':
        del sample.parent[STEM + '.present']
    else:
        del sample.tree[STEM + '.']
    with pytest.raises(ValueError):
        presence.inspect(sample.source)


def test_discovery_ambiguity_requires_exact_branch_no_process_guess(sample):
    sample.tree['mu2e::StrawGasSteps_compressDetStepMCs__Simulate.'] = object()
    with pytest.raises(ValueError, match='ambiguous'):
        presence.inspect(sample.source)
    assert presence.inspect(sample.source, tag=TAG)['all_present']
    with pytest.raises(ValueError, match='exact SGS branch'):
        presence.inspect(sample.source, tag='compressDetStepMCs::Digitize')


@pytest.mark.parametrize('tag', ['compressDetStepMCs', 'compressDetStepMCs::', 'wrong::Compress',
    'compressDetStepMCs:a:b:c', 'compressDetStepMCs::Bad.Process'])
def test_explicit_tag_must_pin_correct_label_instance_process(sample, tag):
    with pytest.raises(ValueError, match='exact'):
        presence.inspect(sample.source, tag=tag)


@pytest.mark.parametrize('keys', [np.array([[1, 1, 2], [1, 1, 2]]),
    np.array([[1, 1, 0], [1, 1, 2]]), np.array([[0, 1, 1], [1, 1, 2]]),
    np.array([[1, 1, -1], [1, 1, 2]]), np.array([[1, 1, 2**32 - 1], [1, 1, 2]]),
    np.array([[1., 1., 1.], [1., 1., 2.]]), np.array([[1, 1, 2]])])
def test_unique_valid_event_keys_required(sample, keys):
    sample.tree.identities = keys
    with pytest.raises(ValueError, match='event keys'):
        presence.inspect(sample.source)


@pytest.mark.parametrize('count', [0, -1, True, 2.0, 3])
def test_expected_event_count_is_exact_positive_integer(sample, count):
    with pytest.raises(ValueError, match='expected_events'):
        presence.inspect(sample.source, count)


def test_changed_bytes_and_changed_receipt_rejected(sample, monkeypatch):
    report = presence.inspect(sample.source)
    edited = deepcopy(report)
    edited['event_keys_numeric'][0][2] = 123
    with pytest.raises(ValueError, match='receipt differs'):
        presence.verify(sample.source, edited, tag=TAG)
    sample.source.write_bytes(b'changed persisted bytes')
    with pytest.raises(ValueError, match='receipt differs'):
        presence.verify(sample.source, report, tag=TAG)
    original = presence.integrity.digest
    calls = []
    def digest(path):
        calls.append(path)
        result = original(path)
        return dict(result, sha256='changed') if len(calls) == 2 else result
    monkeypatch.setattr(presence.integrity, 'digest', digest)
    with pytest.raises(ValueError, match='changed during'):
        presence.inspect(sample.source)


def test_parent_sequence_preserved_with_numeric_only_within_parent(sample, tmp_path):
    other = tmp_path / 'other.art'
    other.write_bytes(b'second parent')
    second = tree()
    second.identities = np.array([[2, 4, 8], [2, 4, 3]], dtype=np.uint64)
    sample.roots[other] = Root({'Events;2': second})
    report = presence.admit([sample.source, other], [2, 2], TAG)
    assert report['parents'] == 2 and report['events'] == report['total_events'] == 4
    assert report['all_present']
    assert report['event_keys_numeric'] == [[1430, 55, 2], [1430, 55, 9], [2, 4, 3], [2, 4, 8]]
    second.identities = sample.tree.identities.copy()
    with pytest.raises(ValueError, match='across source parents'):
        presence.admit([sample.source, other])
    with pytest.raises(ValueError, match='one-to-one'):
        presence.admit([sample.source, other], [2])
    with pytest.raises(ValueError, match='unique sequential'):
        presence.admit([sample.source, sample.source])
    with pytest.raises(ValueError):
        presence.admit([])


def test_highest_non_tree_and_missing_events_are_not_hidden_by_stale_cycles(sample):
    latest = SimpleNamespace(classname='ROOT::RNTuple')
    sample.roots[sample.source].trees['Events;8'] = latest
    with pytest.raises(ValueError, match='highest Events cycle'):
        presence.inspect(sample.source)
    sample.roots[sample.source].trees = {'Other;1': sample.tree}
    with pytest.raises(ValueError, match='missing Events'):
        presence.inspect(sample.source)


def test_real_root_latest_cycle_and_absent_tree(tmp_path):
    path = tmp_path / 'cycles.root'
    with uproot.recreate(path) as output:
        output.mktree('Events', {'value': 'int32'}).extend({'value': np.array([1], dtype=np.int32)})
        output.mktree('Events', {'value': 'int32'}).extend({'value': np.array([2, 3], dtype=np.int32)})
    with uproot.open(path) as root:
        value, cycle = presence._latest(root)
        assert cycle == 2 and value.num_entries == 2
    absent = tmp_path / 'absent.root'
    with uproot.recreate(absent) as output:
        output.mktree('Other', {'value': 'int32'})
    with uproot.open(absent) as root, pytest.raises(ValueError, match='missing Events'):
        presence._latest(root)


def test_cli_optional_counts_and_no_overwrite(sample, tmp_path, monkeypatch, capsys):
    output = tmp_path / 'presence.json'
    monkeypatch.setattr(sys, 'argv', ['response_presence', '--source', str(sample.source)])
    presence.main()
    assert '"all_present": true' in capsys.readouterr().out
    monkeypatch.setattr(sys, 'argv', ['response_presence', '--source', str(sample.source),
        '--counts', '2', '--tag', TAG, '--output', str(output)])
    presence.main()
    original = output.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        presence.main()
    assert output.read_bytes() == original


def test_latest_absent_product_never_falls_back_to_older_present_cycle(sample):
    sample.stale[STEM + '.'][STEM + '.present'].values[:] = True
    sample.parent[STEM + '.present'].values[0] = False
    with pytest.raises(ValueError, match='absent SGS'):
        presence.inspect(sample.source)


def test_instance_is_accepted_only_with_exact_explicit_tag(sample):
    stem = 'mu2e::StrawGasSteps_compressDetStepMCs_named__instance_Simulate'
    tag = 'compressDetStepMCs:named__instance:Simulate'
    sample.tree.clear()
    parent = Parent({stem + '.present': Branch('bool', np.array([True, True])),
                     stem + '.obj': Branch('std::vector<mu2e::StrawGasStep>', ak.Array([[0], [0]]))})
    sample.tree[stem + '.'] = parent
    with pytest.raises(ValueError, match='instance needs an explicit'):
        presence.inspect(sample.source)
    assert presence.inspect(sample.source, tag=tag)['tag'] == tag


def test_parent_process_disagreement_fails_closed(sample, tmp_path):
    other = tmp_path / 'other.art'
    other.write_bytes(b'second parent')
    second = tree()
    stem = STEM.replace('Compress', 'Simulate')
    original = second.pop(STEM + '.')
    second[stem + '.'] = Parent({name.replace('Compress', 'Simulate'): branch
                              for name, branch in original.items()})
    second.identities = second.identities + 1
    sample.roots[other] = Root({'Events;1': second})
    with pytest.raises(ValueError, match='different exact SGS tags'):
        presence.admit([sample.source, other])


def test_zero_finished_events_not_vacuous_presence(sample):
    sample.tree.num_entries = 0
    with pytest.raises(ValueError, match='positive finished'):
        presence.inspect(sample.source)


def test_actual_digitize_compressed_label_requires_exact_explicit_tag(sample):
    old = sample.tree.pop(STEM + '.')
    replacement = STEM.replace('compressDetStepMCs', 'compressDigiMCs').replace('Compress', 'Digitize')
    sample.tree[replacement + '.'] = Parent({name.replace(STEM, replacement): branch for name, branch in old.items()})
    with pytest.raises(ValueError, match='unique SGS branch'):
        presence.inspect(sample.source)
    report = presence.inspect(sample.source, tag='compressDigiMCs::Digitize')
    assert report['tag'] == 'compressDigiMCs::Digitize'
    assert report['product']['branch'] == replacement + '.'
    assert report['all_present']
