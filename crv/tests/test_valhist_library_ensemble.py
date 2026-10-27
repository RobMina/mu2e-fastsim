"""Synthetic multi-shard certification and nested variance tests, without simulation/builds."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import uproot

from valhist import library_ensemble as E
from valhist.library_variants import _event_inventory, _variants
from valhist.pilot import digest, tree_inventory


def put_json(path, value, sealed=False):
    path.write_text(json.dumps(value))
    if sealed:
        path.with_suffix(path.suffix + '.sha256').write_text(digest(path) + '\n')


def tree(root, name, arrays):
    root.mktree(name, {k: np.asarray(v).dtype for k, v in arrays.items()})
    root[name].extend(arrays)


def key_arrays(keys):
    return {k: np.asarray(keys, dtype=np.int64)[:, j] for j, k in enumerate(('run', 'subrun', 'event'))}


def source(path, name, keys):
    with uproot.recreate(path) as root:
        tree(root, name, key_arrays(keys))
    _, inventory = _event_inventory(path, name)
    return dict(path=str(path), sha256=digest(path), events=inventory)


def replay(path, keys, count, offset=0):
    events = key_arrays(keys)
    events.update({k: np.full(len(keys), count, dtype=np.int64) for k in ('ncluster', 'ndigi', 'npulse')})
    clusters = key_arrays(np.repeat(keys, count, axis=0))
    clusters['sectortype'] = np.ones(len(keys) * count, dtype=np.int64)
    with uproot.recreate(path) as root:
        tree(root, 'crvdump/events', events)
        tree(root, 'crvdump/clusters', clusters)
        for name in E.HISTOGRAMS:
            root['crvRecoDQM/' + name] = (np.array([len(keys) * count - offset, offset], float), np.array([0., 1., 2.]))


@pytest.fixture
def ensemble(tmp_path):
    donors = [source(tmp_path / f'donor{i}.root', 'dump/particles', [[8, i, 2 ** 40 + 1]]) for i in range(4)]
    gen = [source(tmp_path / 'denominator.root', 'dump/gen', [[9, 1, 1], [9, 1, 2]])]
    plan = dict(schema=E.PLAN_SCHEMA, donors=donors, gen=gen, generated_per_file=10,
                nominal_generated=40, bootstrap=2, seed=173, build_code_sha256={'builder.py': 'frozen'},
                variants=_variants(4, 10, 2, 173))
    builds = {}
    for index in (1, 2):
        directory = tmp_path / f'library{index}'
        directory.mkdir()
        put_json(directory / 'plan.json', plan, sealed=True)
        (directory / 'library.root').write_bytes(f'library realization {index}'.encode())
        selection = plan['variants'][f'bootstrap_{index:04d}']
        info = dict(inputs=[donors[i]['path'] for i in selection['donor_indices']], gen=[r['path'] for r in gen],
                    generated_scaled_to=40, files=4,
                    provenance=dict(schema=E.PLAN_SCHEMA, variant=f'bootstrap_{index:04d}', selection=selection,
                                    plan_sha256=digest(directory / 'plan.json'),
                                    build_code_sha256=plan['build_code_sha256'],
                                    counters={'path': '/historical/counters', 'sha256': 'geometry'},
                                    library_sha256=digest(directory / 'library.root')))
        builds[index] = directory / 'library.json'
        put_json(builds[index], info, sealed=True)
    transports, populations = {}, {}
    for shard, n in [(11, 2), (12, 4)]:
        keys = np.column_stack([np.full(n, 1400), np.full(n, shard), np.arange(n) + 2 ** 40])
        path = tmp_path / f'fastC.{shard}.root'
        source(path, 'dump/gen', keys)
        transports[shard], populations[shard] = path, keys
    cells = []
    for index in (1, 2):
        for draw in (1, 2):
            for shard in (11, 12):
                directory = tmp_path / f'old_s{shard}_library{index}_draw{draw}'
                directory.mkdir()
                library, transport = builds[index].with_suffix('.root'), transports[shard]
                seed = 200000 + 1000 * index + 10 * draw + shard
                manifest = dict(crv={'path': '/frozen/crv'}, core={'path': '/frozen/core'},
                                area='/frozen/area', backing='/frozen/backing',
                                dqm={'path': '/frozen/area/DQM', 'head': 'frozen-dqm'},
                                fastc=str(transport), library=str(library), seed=seed,
                                generated=tree_inventory(transport, 'dump/gen'),
                                files={str(transport): digest(transport), str(library): digest(library),
                                       '/frozen/crv/crvfastsim/physics.py': 'physics',
                                       '/frozen/core/mu2efastsim/transport.py': 'transport-code'})
                put_json(directory / 'manifest.json', manifest)
                arms = {}
                for arm in E.ARMS:
                    (directory / arm).mkdir()
                    path = directory / arm / 'valhist.root'
                    count = index + draw + (arm == 'e2e')
                    replay(path, populations[shard], count, offset=index + (shard == 12))
                    arms[arm] = dict(path=str(path), completed=True, sha256=digest(path),
                                     trees={'events': tree_inventory(path, 'crvdump/events')})
                put_json(directory / 'completion.json', dict(completed=True, changed_inputs=[], arms=arms,
                                                            expected_generated_events=len(populations[shard])))
                cells.append(dict(index=index, draw=draw, shard=shard, directory=directory.name,
                                  library_metadata=str(builds[index].relative_to(tmp_path)), seed=seed,
                                  expected_generated_events=len(populations[shard])))
    document = dict(schema=E.SCHEMA, cells=cells)
    path = tmp_path / 'cells.json'
    put_json(path, document)
    return path, document


def test_complete_report_preserves_axes_inputs_and_shared_realization(ensemble, tmp_path):
    path, _ = ensemble
    before = {str(p): digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    result = E.run(path, tmp_path / 'report.json')
    assert result['axes'] == dict(index=[1, 2], draw=[1, 2], shard=[11, 12])
    assert result['generated_events_per_shard'] == [2, 4]
    assert result['pooled_generated_events'] == 6
    assert len(result['provenance']) == 8
    assert result['source_audit']['validation_donor_overlap'] == 0
    metric = result['metrics']['e2e_minus_a1']['cluster_rate']
    np.testing.assert_allclose(metric['values_pooled'], 1)
    assert metric['pooled']['components']['libraries'] == 2
    assert metric['pooled']['components']['draws'] == 2
    assert json.loads((tmp_path / 'report.json').read_text()) == result
    assert all(digest(p) == sha for p, sha in before.items())


def test_pool_counts_before_normalization_and_keep_flow():
    count = np.broadcast_to([10., 100.], (3, 2, 2)).copy()
    hist = np.broadcast_to([[8., 1., 0., 1.], [0., 0., 80., 20.]], (3, 2, 2, 4)).copy()
    raw = {a: dict(cluster_count=count, **{n: hist for n in E.HISTOGRAMS}) for a in E.ARMS}
    result = E.summarize(raw, [2, 20], [11, 12])
    metric = result['a1']['PEc_inclusive_fraction']
    np.testing.assert_allclose(metric['values_pooled'][0][0], np.array([8, 1, 80, 21]) / 110)
    assert metric['values_pooled'][0][0] != np.mean(metric['values_per_shard'][0][0], axis=0).tolist()
    assert result['a1']['cluster_rate']['values_pooled'][0][0] == 5


def test_covariance_keeps_shared_library_correlation_and_matches_pooled_variance():
    x = np.array([[[1., 2.], [3., 6.]], [[4., 8.], [6., 12.]], [[7., 14.], [9., 18.]]])
    weights = np.array([0.25, 0.75])
    got = E.covariance(x)
    cov = np.asarray(got['library_covariance_noise_subtracted'])
    pooled = E.components(x @ weights)
    assert cov[0, 1] > 0
    np.testing.assert_allclose(weights @ cov @ weights, pooled['library_variance_noise_subtracted'])
    np.testing.assert_allclose(np.diag(cov), E.components(x)['library_variance_noise_subtracted'])
    vector = E.covariance(np.stack([x, x * 3], axis=-1))
    np.testing.assert_allclose(vector['library_covariance_noise_subtracted'][1], cov * 9)


def test_negative_component_not_clipped():
    x = np.array([[[0., 0.], [2., 4.]], [[0., 0.], [2., 4.]]])
    got = E.components(x.sum(axis=2))
    assert got['library_variance_noise_subtracted'] == -9
    assert got['negative_component'] is True
    assert got['library_sd_nonnegative'] == 0
    assert E.covariance(x)['library_covariance_noise_subtracted'][0][0] == -1


def test_empty_shard_shape_is_explicit_but_pooled_shape_can_exist():
    count = np.broadcast_to([0., 2.], (2, 2, 2)).copy()
    hist = np.broadcast_to([[0., 0.], [1., 1.]], (2, 2, 2, 2)).copy()
    raw = {a: dict(cluster_count=count, **{n: hist for n in E.HISTOGRAMS}) for a in E.ARMS}
    result = E.summarize(raw, [2, 2], [11, 12])
    shape = result['a1']['PEc_inclusive_fraction']
    assert shape['per_shard']['11']['components'] is None
    assert shape['values_per_shard'][0][0][0] == [None, None]
    assert shape['pooled']['status'] == 'defined'
    assert shape['cross_shard']['status'].startswith('undefined')
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('bad', ['one_library', 'one_draw', 'one_shard', 'missing', 'duplicate', 'seed',
                                  'directory', 'metadata', 'exposure', 'unknown', 'bool_index'])
def test_bad_design_refused(ensemble, bad):
    path, document = ensemble
    doc = copy.deepcopy(document)
    if bad.startswith('one_'):
        key = {'one_library': 'index', 'one_draw': 'draw', 'one_shard': 'shard'}[bad]
        doc['cells'] = [c for c in doc['cells'] if c[key] == doc['cells'][0][key]]
    elif bad == 'missing':
        doc['cells'].pop()
    elif bad == 'duplicate':
        doc['cells'].append(doc['cells'][0])
    elif bad in ('seed', 'directory'):
        doc['cells'][1][bad] = doc['cells'][0][bad]
    elif bad == 'metadata':
        doc['cells'][0]['library_metadata'] = doc['cells'][-1]['library_metadata']
    elif bad == 'exposure':
        doc['cells'][0]['expected_generated_events'] += 1
    elif bad == 'unknown':
        doc['cells'][0]['ignored'] = True
    else:
        doc['cells'][0]['index'] = True
    with pytest.raises(ValueError):
        E.validate_manifest(doc, path.parent)


@pytest.mark.parametrize('bad', ['completed', 'drift', 'seed', 'physics', 'census', 'output_hash',
                                  'transport_hash', 'library_hash', 'seal', 'plan_seal', 'selection', 'wrong_shard'])
def test_bad_certification_has_no_output(ensemble, tmp_path, bad):
    path, document = ensemble
    directory = tmp_path / document['cells'][0]['directory']
    manifest_path, completion_path = directory / 'manifest.json', directory / 'completion.json'
    manifest, completion = json.loads(manifest_path.read_text()), json.loads(completion_path.read_text())
    if bad == 'completed':
        completion['completed'] = False
    elif bad == 'drift':
        completion['changed_inputs'] = ['changed']
    elif bad == 'seed':
        manifest['seed'] += 1
    elif bad == 'physics':
        manifest['files']['/frozen/crv/crvfastsim/physics.py'] = 'changed'
    elif bad == 'census':
        completion['arms']['a1']['trees']['events']['event_keys_sha256'] = 'different'
    elif bad == 'output_hash':
        completion['arms']['a1']['sha256'] = 'different'
    elif bad in ('transport_hash', 'library_hash'):
        key = 'fastc' if bad == 'transport_hash' else 'library'
        manifest['files'][manifest[key]] = 'different'
    elif bad in ('seal', 'plan_seal'):
        meta = tmp_path / document['cells'][0]['library_metadata']
        target = meta if bad == 'seal' else meta.parent / 'plan.json'
        target.write_text(target.read_text() + ' ')
    elif bad == 'selection':
        meta = tmp_path / document['cells'][0]['library_metadata']
        info = json.loads(meta.read_text())
        info['inputs'] = info['inputs'][:-1]
        put_json(meta, info, sealed=True)
    else:
        document['cells'][0]['shard'] = 13
        document['cells'] = [dict(c, shard=13 if c['shard'] == 11 else c['shard']) for c in document['cells']]
        put_json(path, document)
    put_json(manifest_path, manifest)
    put_json(completion_path, completion)
    with pytest.raises(ValueError):
        E.run(path, tmp_path / 'out.json')
    assert not (tmp_path / 'out.json').exists()


@pytest.mark.parametrize('field', ['generator_denominator_files', 'generator_denominator_metadata'])
def test_explicit_denominator_overlap_is_never_ignored(ensemble, tmp_path, field):
    path, doc = ensemble
    source(tmp_path / 'extra.root', 'dump/gen', [[1400, 11, 2 ** 40]])
    if field.endswith('_files'):
        doc[field] = ['extra.root']
    else:
        put_json(tmp_path / 'denominator.json', {'gen': ['extra.root']})
        doc[field] = 'denominator.json'
    put_json(path, doc)
    with pytest.raises(ValueError, match='validation/explicit denominator event-key overlap'):
        E.run(path, tmp_path / 'out.json')
    assert not (tmp_path / 'out.json').exists()


def test_disjoint_explicit_denominator_metadata_is_bound(ensemble, tmp_path):
    path, doc = ensemble
    source(tmp_path / 'extra.root', 'dump/gen', [[10, 9, 2 ** 40]])
    put_json(tmp_path / 'extra.json', {'gen': ['extra.root']})
    doc.update(generator_denominator_metadata='extra.json', generator_denominator_files=['extra.root'])
    put_json(path, doc)
    result = E.run(path, tmp_path / 'out.json')
    assert len(result['source_audit']['explicit_denominator_files']) == 1
    assert result['input_sha256'][str(tmp_path / 'extra.json')] == digest(tmp_path / 'extra.json')


@pytest.mark.parametrize('bank,tree_name', [('donors', 'dump/particles'), ('gen', 'dump/gen')])
def test_plan_source_overlap_even_with_updated_hashes_refused(ensemble, tmp_path, bank, tree_name):
    path, doc = ensemble
    new_source = source(tmp_path / 'overlap.root', tree_name, [[1400, 12, 2 ** 40]])
    for index in (1, 2):
        meta = tmp_path / f'library{index}/library.json'
        plan_path = meta.parent / 'plan.json'
        plan = json.loads(plan_path.read_text())
        plan[bank][0] = new_source
        put_json(plan_path, plan, sealed=True)
        info = json.loads(meta.read_text())
        info['provenance']['plan_sha256'] = digest(plan_path)
        info['inputs'] = [plan['donors'][i]['path'] for i in info['provenance']['selection']['donor_indices']]
        info['gen'] = [r['path'] for r in plan['gen']]
        put_json(meta, info, sealed=True)
    with pytest.raises(ValueError, match='validation/' + bank + ' event-key overlap'):
        E.run(path, tmp_path / 'out.json')


def test_no_overwrite_including_dangling_symlink(tmp_path):
    out = tmp_path / 'existing'
    out.symlink_to(tmp_path / 'missing')
    with pytest.raises(FileExistsError):
        E.run(tmp_path / 'does-not-exist.json', out)


def test_final_drift_check_prevents_publication(ensemble, tmp_path, monkeypatch):
    path, _ = ensemble
    original = E.summarize

    def change_during_analysis(*args):
        result = original(*args)
        path.write_text(path.read_text() + ' ')
        return result

    monkeypatch.setattr(E, 'summarize', change_during_analysis)
    with pytest.raises(ValueError, match='input changed during report'):
        E.run(path, tmp_path / 'out.json')
    assert not (tmp_path / 'out.json').exists()


@pytest.mark.parametrize('bad', ['keys', 'duplicates', 'fractional_ids', 'booking', 'histogram_total'])
def test_rehashed_output_still_requires_exact_events_and_bookings(ensemble, tmp_path, bad):
    path, doc = ensemble
    directory = tmp_path / doc['cells'][0]['directory']
    output = directory / 'a1/valhist.root'
    complete_path = directory / 'completion.json'
    complete = json.loads(complete_path.read_text())
    keys = np.array([[1400, 11, 2 ** 40], [1400, 11, 2 ** 40 + 1]])
    if bad == 'keys':
        keys[-1, -1] += 100
    elif bad == 'duplicates':
        keys[-1] = keys[0]
    if bad in ('keys', 'duplicates'):
        replay(output, keys, 2)
    else:
        with uproot.update(output) as root:
            if bad == 'fractional_ids':
                events = {k: v.astype(float) for k, v in key_arrays(keys).items()}
                events['event'] += .5
                events.update({k: np.full(2, 2, dtype=np.int64) for k in ('ncluster', 'ndigi', 'npulse')})
                tree(root, 'crvdump/events', events)
            else:
                edges = np.array([0., 1., 3. if bad == 'booking' else 2.])
                root['crvRecoDQM/PEc'] = (np.array([3., 2. if bad == 'histogram_total' else 1.]), edges)
    complete['arms']['a1']['sha256'] = digest(output)
    complete['arms']['a1']['trees']['events'] = tree_inventory(output, 'crvdump/events')
    put_json(complete_path, complete)
    with pytest.raises(ValueError):
        E.run(path, tmp_path / 'out.json')
    assert not (tmp_path / 'out.json').exists()


def test_donor_hash_drift_fails_before_replays(ensemble, tmp_path):
    path, _ = ensemble
    (tmp_path / 'donor0.root').write_bytes(b'drift')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        E.run(path, tmp_path / 'out.json')


def test_missing_generated_key_digest_is_not_inferred(ensemble, tmp_path):
    path, doc = ensemble
    manifest_path = tmp_path / doc['cells'][0]['directory'] / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    del manifest['generated']['event_keys_sha256']
    put_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='generated inventory'):
        E.run(path, tmp_path / 'out.json')
