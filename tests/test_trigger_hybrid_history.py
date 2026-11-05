"""Synthetic history/integrity tests only: no physics or trigger certification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import hybrid_history as history
import integrity


class Root:
    def __init__(self, keys):
        self.tree = SimpleNamespace(classname='TTree', num_entries=len(keys),
                                    keys_for_test=np.asarray(keys, dtype=np.uint64))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __getitem__(self, name):
        assert name == 'Events'
        return self.tree


@pytest.fixture
def case(tmp_path, monkeypatch):
    directory = tmp_path / 'stock'
    directory.mkdir()
    parents = [tmp_path / 'parent133.art', tmp_path / 'parent55.art']
    candidates = [directory / 'unfiltered_1.art', directory / 'unfiltered_2.art']
    reference = tmp_path / 'selected.art'
    for path in [*parents, *candidates, reference]:
        path.write_bytes(path.name.encode())
    # Numeric traversal sorts within parents, never globally across parents.
    roots = {parents[0].name: [[1430, 133, 3], [1430, 133, 1]],
             parents[1].name: [[1430, 55, 1]],
             candidates[0].name: [[1430, 133, 1], [1430, 133, 3]],
             candidates[1].name: [[1430, 55, 1]]}
    monkeypatch.setattr(history.regeneration_census.uproot, 'open', lambda path: Root(roots[Path(path).name]))
    monkeypatch.setattr(history.regeneration_census.event_keys, 'read', lambda tree: tree.keys_for_test)
    config = 'synthetic production configuration, not runnable physics'
    config_hash = hashlib.sha256(config.encode()).hexdigest()
    manifest = {'opened_parent_order_verified': True, 'split_input_files': True,
                'original_processed_events': 3, 'config_sha256': config_hash,
                'parents': [{'did': 'mu2e:parent133.art'}, {'did': 'mu2e:parent55.art'}],
                'local_parent_paths': [str(path) for path in parents]}
    comparison = {'equal': True, 'all_reference_events_present': True,
                  'root_event_ids_verified': True, 'original_output_selection_event_set_equal': True,
                  'regenerated_events': 3, 'config_sha256': config_hash,
                  'candidate_files': integrity.files(candidates), 'reference_files': integrity.files(reference),
                  'paths': {'synthetic': {'packed_status_mismatches': 0}},
                  'scope': 'synthetic original production overlap'}
    manifest_path, comparison_path = tmp_path / 'regeneration.json', tmp_path / 'comparison.json'
    data = SimpleNamespace(config=config, manifest=manifest, comparison=comparison,
                           manifest_path=manifest_path, comparison_path=comparison_path,
                           directory=directory, parents=parents, candidates=candidates,
                           reference=reference, roots=roots)

    def save():
        manifest_path.write_text(json.dumps(manifest))
        comparison_path.write_text(json.dumps(comparison))

    def certify():
        return history.certify(config, manifest_path, directory, comparison_path, 3)

    data.save, data.certify = save, certify
    save()
    return data


def test_complete_stored_history_is_bound_and_reverified(case):
    receipt = case.certify()
    for key, value in case.comparison.items():
        if key != 'scope':
            assert receipt[key] == value
    assert receipt['production_comparison_scope'] == case.comparison['scope']
    assert receipt['full_parent_history_verified'] is True
    assert receipt['parent_census']['original_source_events'] == 3
    assert receipt['parent_census']['complete_parent_event_ids_verified'] is True
    assert [row['events'] for row in receipt['parent_census']['original_parent_files']] == [2, 1]
    assert receipt['evidence_files'] == integrity.files([case.manifest_path, case.comparison_path, *case.parents])
    assert 'complete original stored' in receipt['scope']
    assert 'not generated exposure' in receipt['scope']
    assert 'not hybrid physics certification' in receipt['scope']
    assert history.verify(receipt, case.config, case.candidates, 3) == receipt
    assert history.verify(receipt, case.config, integrity.files(case.candidates), 3) == receipt
    assert history.verify(json.loads(json.dumps(receipt)), case.config, case.candidates, 3) == receipt


@pytest.mark.parametrize('flag', history._COMPARISON_FLAGS)
@pytest.mark.parametrize('bad', [None, False, 1])
def test_explicit_passing_overlap_flags_required(case, flag, bad):
    if bad is None:
        case.comparison.pop(flag)
    else:
        case.comparison[flag] = bad
    case.save()
    with pytest.raises(ValueError, match='production overlap'):
        case.certify()


@pytest.mark.parametrize('flag', ['opened_parent_order_verified', 'split_input_files'])
@pytest.mark.parametrize('bad', [None, False, 1])
def test_production_parent_order_and_split_required(case, flag, bad):
    case.manifest[flag] = bad
    case.save()
    with pytest.raises(ValueError, match='regeneration manifest'):
        case.certify()


@pytest.mark.parametrize('report,key,value', [
    ('manifest', 'original_processed_events', 4),
    ('manifest', 'original_processed_events', True),
    ('comparison', 'regenerated_events', 2),
    ('comparison', 'config_sha256', 'wrong'),
    ('manifest', 'config_sha256', 'wrong')])
def test_count_and_config_bound(case, report, key, value):
    getattr(case, report)[key] = value
    case.save()
    with pytest.raises(ValueError):
        case.certify()


def test_parent_prefix_cannot_be_promoted_by_manifest_count(case):
    # The existing census alone slices this fourth source event off and passes.
    case.roots[case.parents[1].name].append([1430, 55, 2])
    count = history.regeneration_census.census(case.manifest, case.directory, 3)
    assert count['complete_parent_event_ids_verified'] is True
    with pytest.raises(ValueError, match='all original source events.*prefix'):
        case.certify()


@pytest.mark.parametrize('change', ['parent_order', 'within_parent_order', 'missing_event'])
def test_identity_order_and_coverage_rechecked(case, change):
    if change == 'parent_order':
        case.manifest['local_parent_paths'].reverse()
        case.manifest['parents'].reverse()
    elif change == 'within_parent_order':
        case.roots[case.candidates[0].name].reverse()
    else:
        case.roots[case.candidates[0].name].pop()
    case.save()
    with pytest.raises(ValueError):
        case.certify()


@pytest.mark.parametrize('flag', history._CENSUS_FLAGS)
def test_census_flags_not_assumed(case, monkeypatch, flag):
    original = history.regeneration_census.census

    def incomplete(*args):
        result = original(*args)
        result[flag] = False
        return result

    monkeypatch.setattr(history.regeneration_census, 'census', incomplete)
    with pytest.raises(ValueError, match='complete parent census'):
        case.certify()


@pytest.mark.parametrize('change', ['boundary_count', 'missing_boundary', 'count'])
def test_census_totals_and_boundaries_not_assumed(case, monkeypatch, change):
    original = history.regeneration_census.census

    def incomplete(*args):
        result = original(*args)
        if change == 'boundary_count':
            result['files'][0]['events'] -= 1
        elif change == 'missing_boundary':
            result['files'].pop()
        else:
            result['events'] -= 1
        return result

    monkeypatch.setattr(history.regeneration_census, 'census', incomplete)
    with pytest.raises(ValueError, match='count|boundary'):
        case.certify()


@pytest.mark.parametrize('change', ['candidate_order', 'candidate_bytes', 'reference_bytes', 'reference_missing'])
def test_production_byte_bindings_required(case, change):
    if change == 'candidate_order':
        case.comparison['candidate_files'].reverse()
    elif change == 'candidate_bytes':
        case.candidates[0].write_bytes(b'changed stock')
    elif change == 'reference_bytes':
        case.reference.write_bytes(b'changed original')
    else:
        case.comparison['reference_files'] = []
    case.save()
    with pytest.raises(ValueError, match='reference|candidate'):
        case.certify()


@pytest.mark.parametrize('which', ['manifest_path', 'comparison_path', 'parent', 'reference', 'candidate'])
def test_verify_rejects_evidence_mutation(case, which):
    receipt = case.certify()
    path = {'manifest_path': case.manifest_path, 'comparison_path': case.comparison_path,
            'parent': case.parents[0], 'reference': case.reference, 'candidate': case.candidates[0]}[which]
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='bytes|changed'):
        history.verify(receipt, case.config, case.candidates, 3)


@pytest.mark.parametrize('change', ['plain_overlap', 'forged_flag', 'forged_census', 'missing_evidence',
                                    'reordered_evidence', 'forged_comparison', 'wrong_config',
                                    'wrong_count', 'wrong_order', 'truthy_census_flag'])
def test_verify_independently_checks_receipt(case, change):
    receipt = case.certify()
    config, expected, references = case.config, 3, case.candidates
    if change == 'plain_overlap':
        receipt = deepcopy(case.comparison)
    elif change == 'forged_flag':
        receipt['full_parent_history_verified'] = 1
    elif change == 'forged_census':
        receipt['parent_census']['events'] = 2
    elif change == 'truthy_census_flag':
        receipt['parent_census']['complete_parent_event_ids_verified'] = 1
    elif change == 'missing_evidence':
        receipt['evidence_files'] = receipt['evidence_files'][:-1]
    elif change == 'reordered_evidence':
        receipt['evidence_files'].reverse()
    elif change == 'forged_comparison':
        receipt['paths']['synthetic']['packed_status_mismatches'] = 1
    elif change == 'wrong_config':
        config += 'changed'
    elif change == 'wrong_count':
        expected = 2
    else:
        references = references[::-1]
    with pytest.raises(ValueError):
        history.verify(receipt, config, references, expected)


def test_alias_paths_are_canonicalized_on_both_sides(case, tmp_path):
    alias = tmp_path / 'alternate-mount'
    alias.symlink_to(tmp_path, target_is_directory=True)
    for name in ('candidate_files', 'reference_files'):
        for row in case.comparison[name]:
            row['path'] = str(alias / Path(row['path']).relative_to(tmp_path))
    case.manifest['local_parent_paths'] = [str(alias / path.name) for path in case.parents]
    case.save()
    receipt = history.certify(case.config, alias / case.manifest_path.name,
                              alias / case.directory.name, alias / case.comparison_path.name, 3)
    # Exercise old stored receipt rows, not just alias paths passed as arguments.
    for row in receipt['evidence_files']:
        row['path'] = str(alias / Path(row['path']).relative_to(tmp_path))
    assert history.verify(receipt, case.config, case.candidates, 3)['full_parent_history_verified']


def test_verify_does_not_trust_refreshed_hashes_after_manifest_change(case):
    receipt = case.certify()
    case.manifest['local_parent_paths'].reverse()
    case.save()
    # Updating hashes without restoring the ordered ROOT history is insufficient.
    receipt['evidence_files'] = integrity.files([case.manifest_path, case.comparison_path, *case.parents])
    with pytest.raises(ValueError, match='original parent'):
        history.verify(receipt, case.config, case.candidates, 3)


def test_bind_mount_alias_requires_same_inode_and_exact_hash(tmp_path):
    original, alias, unrelated = [tmp_path / name for name in ('original', 'alias', 'unrelated')]
    original.write_bytes(b'same bytes')
    alias.hardlink_to(original)
    unrelated.write_bytes(b'same bytes')
    history._same_files(integrity.files(original), integrity.files(alias), 'bind alias')
    with pytest.raises(ValueError, match='bytes, paths or order'):
        history._same_files(integrity.files(original), integrity.files(unrelated), 'unrelated copy')
