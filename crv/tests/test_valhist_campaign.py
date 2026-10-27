"""Campaign assembly refuses partial or misidentified persisted shards."""
import json

import numpy as np
import pytest

from valhist import campaign


def test_paths_reuse_pilot_and_reject_bad_tag(tmp_path):
    rows = campaign.shard_paths(tmp_path, 'new_campaign')
    assert len(rows) == 20
    assert rows[10][1].name == 'thread_b_dqm_20261025_s11'
    assert rows[0][1].name == 'new_campaign_s00000001'
    for tag in ('', '../unsafe', 'with space'):
        with pytest.raises(ValueError):
            campaign.shard_paths(tmp_path, tag)


def test_matched_revision_pilot_never_falls_back_to_legacy(tmp_path):
    matched = tmp_path / 'valhist/new_s00000011'
    matched.mkdir(parents=True)
    assert campaign.shard_paths(tmp_path, 'new')[10][1] == matched
    rows = campaign.inventory(tmp_path, 'new')
    assert not rows[10]['completed']


def test_partial_refused_without_output(tmp_path):
    directory = tmp_path / 'valhist/thread_b_dqm_20261025_s11'
    directory.mkdir(parents=True)
    (directory / 'completion.json').write_text(json.dumps(dict(completed=True)))
    rows = campaign.inventory(tmp_path, 'new')
    assert [r['shard'] for r in rows if r['completed']] == [11]
    with pytest.raises(ValueError, match='incomplete shards'):
        campaign.prepare(tmp_path, 'new', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_existing_output_refused(tmp_path):
    with pytest.raises(FileExistsError):
        campaign.prepare(tmp_path, 'new', tmp_path)


@pytest.mark.parametrize('bad', ['hash', 'count', 'shard', 'transport', 'arms'])
def test_wrong_completed_input_fails(tmp_path, monkeypatch, bad):
    source = tmp_path / 'source'
    source.mkdir()
    manifest = dict(generated=dict(event_keys_sha256='keys', unique_events=50000))
    (source / 'manifest.json').write_text(json.dumps(manifest))
    completion = dict(completed=True, expected_generated_events=50000, changed_inputs=[],
                      arms={a: dict(completed=True, sha256='sha', path=str(source / a / 'valhist.root'),
                                    trees=dict(events=dict(rows=50000, unique_events=50000,
                                                           event_keys_sha256='keys')))
                            for a in ('a1', 'e2e')})
    (source / 'completion.json').write_text(json.dumps(completion))
    monkeypatch.setattr(campaign, 'inventory', lambda *args: [dict(shard=1, directory=str(source),
                                                                 completed=True, completion=completion)])
    monkeypatch.setattr(campaign, 'digest', lambda p: 'wrong' if bad == 'hash' else 'sha')
    n = 10 if bad == 'count' else 50000
    keys = np.column_stack([np.full(n, 1400), np.full(n, 2 if bad == 'shard' else 1), np.arange(n)])

    def read(path):
        k = keys.copy()
        if bad == 'arms' and path.parent.name == 'e2e':
            k[0, 2] += n
        return k, {'ncluster': np.zeros(n)}, dict(rows=n, unique_events=n,
                                                  event_keys_sha256='wrong' if bad == 'transport' else 'keys')

    monkeypatch.setattr(campaign, 'read', read)
    with pytest.raises(ValueError):
        campaign.prepare(tmp_path, 'new', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def physics_manifest(shard=1, crv='/frozen/crv', core='/frozen/core'):
    area = '/frozen/area'
    paths = [crv + '/crvfastsim/response/lookup.py', crv + '/crvfastsim/reco/chain.py',
             crv + '/valhist/pilot.py', crv + '/valhist/run_pilot.sh',
             core + '/mu2efastsim/transport.py', area + '/CrvFastSim/fcl/a.fcl',
             area + '/CrvFastSim/src/module.cc', area + '/build/linux/CrvFastSim/lib/art.so',
             area + '/build/linux/DQM/lib/CrvDQM.so',
             area + '/backing/Offline/CRVDQM/data/CRVDQM_binning_v3_extracted.txt',
             '/data/geometry/crvcalib_sim_best_v1_5_run1400.csv',
             '/data/geometry/crvphoton_sim_best_v1_5_run1400.csv',
             '/cvmfs/CRVConditions/v6/LookupTable_6000_0', '/cvmfs/CRVConditions/v6/photonMap.root',
             '/offline/CRVResponse/data/singlePEWaveform_v3.txt']
    fastc, library = f'/transport/fastC.{shard}.root', '/library/nominal.root'
    return dict(crv=dict(path=crv, head='crv-head', status=''),
                core=dict(path=core, head='core-head', status=''),
                area=area, backing='/frozen/backing', dqm=dict(path=area + '/DQM', head='dqm-head'),
                fastc=fastc, library=library, seed=shard,
                files={p: 'physics-' + p.rsplit('/', 1)[-1] for p in paths} | {fastc: 'transport', library: 'nominal'},
                asset_paths=paths[-5:], asset_paths_missing=[])


def physics_campaign():
    manifests = {s: physics_manifest(s) for s in range(1, 21)}
    legacy = manifests[11]
    for path in list(legacy['files']):
        if '/mu2efastsim/' in path or '/CRVConditions/' in path or 'singlePEWaveform' in path:
            del legacy['files'][path]
    legacy.pop('asset_paths')
    legacy['crv']['head'] = 'old-helper-only-commit'
    legacy['files']['/frozen/crv/valhist/pilot.py'] = 'old-helper'
    return manifests


def test_logical_physics_relocated_sources_and_helpers():
    original = physics_manifest()
    moved = physics_manifest(crv='/new/detector', core='/new/core')
    moved['crv']['head'] = 'helper-only-commit'
    moved['files']['/new/detector/valhist/pilot.py'] = 'new-helper'
    moved['files']['/new/detector/valhist/new_report.py'] = 'new-report'
    moved['library'] = '/library/variant.root'
    moved['files'][moved['library']] = moved['files'].pop('/library/nominal.root')
    moved['files'][moved['fastc']] = 'shard-specific-transport'
    assert campaign.logical_physics(original) == campaign.logical_physics(moved)
    moved['files']['/new/detector/crvfastsim/reco/chain.py'] = 'changed-physics'
    assert campaign.logical_physics(original) != campaign.logical_physics(moved)


def test_legacy_exception_is_recorded_and_not_fabricated():
    result = campaign._physics_campaign(physics_campaign())
    assert result['exceptions'][0]['shard'] == 11
    assert len(result['exceptions'][0]['missing_hashes']) == 4
    assert any(k.startswith('core/') for k in result['exceptions'][0]['missing_hashes'])


@pytest.mark.parametrize('bad', ['moved_changed', 'missing', 'extra', 'legacy_detector', 'legacy_condition',
                                  'core', 'asset', 'area', 'backing', 'dqm', 'library', 'seed'])
def test_physics_inventory_and_identities_are_strict(bad):
    manifests = physics_campaign()
    target = manifests[2]
    if bad == 'moved_changed':
        target = manifests[2] = physics_manifest(2, crv='/other/crv', core='/other/core')
        target['files']['/other/crv/crvfastsim/response/lookup.py'] = 'drift'
    elif bad == 'missing':
        del target['files']['/frozen/crv/crvfastsim/reco/chain.py']
    elif bad == 'extra':
        target['files']['/frozen/crv/crvfastsim/new_physics.py'] = 'extra'
    elif bad.startswith('legacy_'):
        path = ('/frozen/crv/crvfastsim/reco/chain.py' if bad == 'legacy_detector' else
                '/data/geometry/crvcalib_sim_best_v1_5_run1400.csv')
        del manifests[11]['files'][path]
    elif bad == 'core':
        del target['files']['/frozen/core/mu2efastsim/transport.py']
    elif bad == 'asset':
        del target['files']['/cvmfs/CRVConditions/v6/LookupTable_6000_0']
    elif bad in ('area', 'backing'):
        target[bad] = '/other/' + bad
    elif bad == 'dqm':
        target['dqm']['head'] = 'other'
    elif bad == 'library':
        target['files'][target['library']] = 'variant'
    elif bad == 'seed':
        target['seed'] = 42
    with pytest.raises(ValueError):
        campaign._physics_campaign(manifests)


def test_missing_same_required_inventory_everywhere_is_rejected():
    manifests = physics_campaign()
    for m in manifests.values():
        m['files'].pop('/frozen/core/mu2efastsim/transport.py', None)
    with pytest.raises(ValueError, match='missing required physics inventory'):
        campaign._physics_campaign(manifests)


@pytest.fixture
def prepared_campaign(tmp_path, monkeypatch):
    manifests = physics_campaign()
    for shard, directory in campaign.shard_paths(tmp_path, 'new'):
        directory.mkdir(parents=True)
        events = dict(rows=50000, unique_events=50000, event_keys_sha256=f'keys-{shard}')
        manifest = manifests[shard]
        manifest['generated'] = events
        completion = dict(completed=True, changed_inputs=[], expected_generated_events=50000, arms={})
        for arm in ('a1', 'e2e'):
            path = directory / arm / 'valhist.root'
            path.parent.mkdir()
            path.write_text(f'output-{arm}-{shard}')
            completion['arms'][arm] = dict(completed=True, path=str(path), sha256=campaign.digest(path),
                                            trees=dict(events=events))
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        (directory / 'completion.json').write_text(json.dumps(completion))
        for arm in ('a0g', 'a0', 'a0p'):
            path = tmp_path / 'valhist/g4ref_v1' / arm / f'valhist.001400_{shard:08d}.root'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'historical-{arm}-{shard}')

    def read(path):
        parent = path.parent.parent.name
        shard = 11 if parent.endswith('_s11') else int(parent.rsplit('_s', 1)[1])
        keys = np.column_stack([np.full(50000, 1400), np.full(50000, shard), np.arange(50000)])
        return keys, {'ncluster': np.ones(50000)}, dict(rows=50000, unique_events=50000,
                                                       event_keys_sha256=f'keys-{shard}')

    monkeypatch.setattr(campaign, 'read', read)
    out = tmp_path / 'report'
    record = campaign.prepare(tmp_path, 'new', out)
    return out, record


def test_validate_bound_campaign_is_read_only(prepared_campaign):
    out, original = prepared_campaign
    before = {p: p.read_bytes() for p in out.parent.rglob('*') if p.is_file()}
    record, pairs = campaign.validate_report_inputs(out)
    assert record == original
    assert set(pairs) == set(campaign.PAIR_NAMES)
    assert all(len(rows) == 20 for rows in pairs.values())
    from valhist.paired_shapes import _load_pairs
    for name in pairs:
        assert len(_load_pairs(out / f'pairs_{name}.json')) == 20
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize('bad', ['pairs', 'missing_pair', 'duplicate_pair', 'pair_path', 'candidate',
                                  'reference', 'floor_candidate', 'manifest', 'completion', 'provenance',
                                  'population', 'completion_population', 'missing_binding'])
def test_validate_rejects_tampering(prepared_campaign, monkeypatch, bad):
    out, record = prepared_campaign
    path = out / 'pairs_e2e_a1.json'
    rows = json.loads(path.read_text())
    if bad in ('pairs', 'missing_pair', 'duplicate_pair', 'pair_path'):
        if bad == 'missing_pair':
            rows.pop()
        elif bad == 'duplicate_pair':
            rows[0] = rows[1]
        else:
            rows[0]['cand'] = rows[1]['cand']
        path.write_text(json.dumps(rows))
        # Exercise semantic validation too, not just the outer JSON hash.
        if bad != 'pairs':
            record['pair_manifests']['e2e_a1']['sha256'] = campaign.digest(path)
            (out / 'campaign.json').write_text(json.dumps(record))
    elif bad in ('candidate', 'reference', 'floor_candidate'):
        name, role = {'candidate': ('e2e_a1', 'cand'), 'reference': ('a1_a0g', 'ref'),
                      'floor_candidate': ('floor', 'cand')}[bad]
        target = json.loads((out / f'pairs_{name}.json').read_text())[0][role]
        from pathlib import Path
        Path(target).write_text('changed-root')
    elif bad in ('manifest', 'completion', 'completion_population'):
        from pathlib import Path
        target = Path(record['provenance'][0]['manifest'])
        if bad != 'manifest':
            target = target.parent / 'completion.json'
        if bad == 'completion_population':
            content = json.loads(target.read_text())
            content['arms']['a1']['trees']['events']['unique_events'] = 49999
            target.write_text(json.dumps(content))
            record['provenance'][0]['completion_sha256'] = campaign.digest(target)
            (out / 'campaign.json').write_text(json.dumps(record))
        else:
            target.write_text(target.read_text() + ' ')
    elif bad == 'provenance':
        record['provenance'].pop()
        (out / 'campaign.json').write_text(json.dumps(record))
    elif bad == 'missing_binding':
        del record['input_sha256'][rows[0]['cand']]
        (out / 'campaign.json').write_text(json.dumps(record))
    elif bad == 'population':
        original = campaign.read
        def wrong_read(path):
            k, v, meta = original(path)
            k[:, 1] = 42
            return k, v, meta
        monkeypatch.setattr(campaign, 'read', wrong_read)
    with pytest.raises(ValueError):
        campaign.validate_report_inputs(out)


def test_analyze_validates_before_writing(prepared_campaign, monkeypatch):
    from valhist import paired_shapes
    out, _ = prepared_campaign
    (out / 'pairs_floor.json').write_text('[]')
    monkeypatch.setattr(paired_shapes, 'run', lambda *a, **k: pytest.fail('must validate before shapes'))
    with pytest.raises(ValueError, match='pair manifest changed'):
        campaign.analyze(out)
