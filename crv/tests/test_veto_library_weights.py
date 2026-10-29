"""Provenance grouping is deterministic accounting, never an inferred library error."""
import copy
import hashlib
import itertools
import json

import numpy as np
import pytest

from crvfastsim.veto import library_weights as audit
from crvfastsim.veto.mixframes import MIXERS


SPECIES = 'NeutralsFlashCat'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture():
    sources = {}
    for i, group in enumerate(('a', 'a', 'b', 'c')):
        name = f'dts.mu2e.NeutralsFlashCat.TEST.{i}.art'
        count = 100 if group == 'a' else 200
        sources[name] = {'table_sha256': digest(group), 'table_frames': count,
                        'step_events': count // 10, 'sam': {
                            'file_name': name, 'source_sha256': digest(name), 'metadata_sha256': digest(name + 'sam'),
                            'parents_sha256': digest(group + 'parents'), 'parent_count': 50, 'event_count': count + 1}}
    names = list(sources)
    overlaps = [{'first': a, 'second': b, 'shared_parents': 50 if sources[a]['table_sha256'] == sources[b]['table_sha256'] else 0,
                 'identical_parent_sets': sources[a]['table_sha256'] == sources[b]['table_sha256']}
                for a, b in itertools.combinations(sorted(sources), 2)]
    manifests = {}
    for block, indices in enumerate(((0, 1, 2), (1, 3))):
        selected = []
        for i in indices:
            name, record = names[i], sources[names[i]]
            selected.append({'path': '/recorded/steps.' + name[:-4] + '.parquet', 'sha256': record['table_sha256'],
                             'metadata': {'dataset': SPECIES, 'frames': record['table_frames'],
                                          'stepped_frames': record['step_events']}})
        manifests[str(block)] = {'schema': 'crv-mixed-replay-v1', 'output_sha256': digest(str(block)),
                                'events': 10 * (block + 1), 'mean_intensity': 1e7, 'mixers': MIXERS,
                                'sources': {SPECIES: selected}}
    return manifests, {'sources': sources, 'parent_overlaps': overlaps}


def test_exact_accounting_and_no_mutation():
    manifests, provenance = fixture()
    before = copy.deepcopy((manifests, provenance))
    result = audit.audit(manifests, provenance)
    assert before == (manifests, provenance)
    assert result['finite_library_uncertainty'] is None
    first, second = result['blocks']
    assert first['occurrences'] == 3
    assert first['groups'] == 2
    assert first['table_frames_occurrences'] == 400
    assert first['catalog_minus_table_occurrences'] == 3
    assert first['unique_payload_table_frames'] == 300
    assert first['current']['inverse_concentration'] == pytest.approx(2)
    assert first['group_weight_total_variation_diagnostic'] == pytest.approx(1 / 6)
    assert first['stepped_probability_current'] == pytest.approx(.1)
    assert first['stepped_probability_one_copy_diagnostic'] == pytest.approx(.1)
    assert first['stepped_probability_catalog_denominator_only_diagnostic'] == pytest.approx(40 / 403)
    assert second['group_weight_total_variation_diagnostic'] == 0
    gram = np.asarray(result['block_group_weight_gram'])
    np.testing.assert_allclose(gram, [[.5, 1 / 6], [1 / 6, 5 / 9]])
    np.testing.assert_allclose(np.diag(result['block_group_weight_cosine']), 1)
    weights = np.average([r['current']['weights'] for r in result['blocks']], axis=0, weights=[1, 2])
    np.testing.assert_allclose(result['expected_source_draw_weighted_group_concentration']['weights'], weights)


def test_group_and_overlap_order_invariance():
    manifests, provenance = fixture()
    result = audit.audit(manifests, provenance)
    provenance['sources'] = dict(reversed(list(provenance['sources'].items())))
    provenance['parent_overlaps'].reverse()
    for manifest in manifests.values():
        manifest['sources'][SPECIES].reverse()
    assert audit.audit(manifests, provenance) == result


@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'unknown', 'partial', 'false_equality'])
def test_parent_census_fails_closed(mode):
    manifests, provenance = fixture()
    pairs = provenance['parent_overlaps']
    if mode == 'missing':
        pairs.pop()
    elif mode == 'duplicate':
        pairs.append(pairs[0])
    elif mode == 'unknown':
        pairs[0]['first'] = 'unknown'
    elif mode == 'partial':
        pairs[-1]['shared_parents'] = 1
    else:
        pairs[0]['identical_parent_sets'] = False
    with pytest.raises(ValueError):
        audit.audit(manifests, provenance)


@pytest.mark.parametrize('field,value', [('table_sha256', '0' * 64), ('table_frames', 99), ('step_events', 9)])
def test_parent_aliases_require_equal_payload_and_counts(field, value):
    manifests, provenance = fixture()
    next(iter(provenance['sources'].values()))[field] = value
    with pytest.raises(ValueError, match='unequal payload'):
        audit.audit(manifests, provenance)


@pytest.mark.parametrize('value', [None, True, 0, -1, 100.0])
def test_counts_strict(value):
    manifests, provenance = fixture()
    next(iter(provenance['sources'].values()))['table_frames'] = value
    with pytest.raises(ValueError):
        audit.audit(manifests, provenance)


@pytest.mark.parametrize('mode', ['schema', 'repeat_input', 'repeat_source', 'wrong_hash', 'wrong_count', 'unbound', 'missing_source'])
def test_manifest_binding(mode):
    manifests, provenance = fixture()
    first = manifests['0']
    item = first['sources'][SPECIES][0]
    if mode == 'schema':
        first['schema'] = 'crv-mixed-replay-v2'
    elif mode == 'repeat_input':
        manifests['1']['output_sha256'] = first['output_sha256']
    elif mode == 'repeat_source':
        first['sources'][SPECIES].append(item)
    elif mode == 'wrong_hash':
        item['sha256'] = '0' * 64
    elif mode == 'wrong_count':
        item['metadata']['frames'] += 1
    elif mode == 'unbound':
        item['path'] = '/recorded/steps.dts.mu2e.NeutralsFlashCat.TEST.999.parquet'
    else:
        manifests.pop('1')
    with pytest.raises(ValueError):
        audit.audit(manifests, provenance)


@pytest.mark.parametrize('weights', [[0, 0], [1, 1], [float('nan')], [-.1, 1.1], [], [[1]]])
def test_invalid_weight_vector(weights):
    with pytest.raises(ValueError):
        audit.weight_summary(weights)


def test_cli_bindings_and_no_overwrite(tmp_path):
    manifests, provenance = fixture()
    paths = []
    for name, manifest in manifests.items():
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(manifest))
        paths.append(str(path))
    prior, out = tmp_path / 'prior.json', tmp_path / 'result.json'
    prior.write_text(json.dumps(provenance))
    args = ['--manifests', *paths, '--provenance', str(prior), '--out', str(out)]
    audit.main(args)
    result = json.loads(out.read_text())
    assert len(result['inputs_sha256']) == 3
    assert result['code_sha256'] == audit.sha256(audit.__file__)
    with pytest.raises(ValueError, match='new report'):
        audit.main(args)
