import json

import numpy as np
import pytest

from beamfastsim import tsholdout as ts
from beamfastsim import tsrawreference
from test_tssmoke import particles, hits


def fixture(monkeypatch):
    monkeypatch.setattr(ts, 'INDICES', (100, 101))
    batches = []
    for i in (100, 101):
        p, vd, beam = particles(), hits([2], [-211], [10]), hits([3], [13], [50])
        for table in (p, vd, beam):
            table['run'][:] = 1430
            table['subrun'][:] = 920000 + i
        events = np.column_stack((np.full(5000, 1430), np.full(5000, 920000 + i), np.arange(1, 5001)))
        a, _ = tsrawreference.build(p, vd, beam, event_id=events)
        a = tsrawreference.root_lineages(a)
        if batches:
            for k in ('endpoint_source', 'crossing_source'):
                a[k] += 1
        batches.append(a)
    r = {k: np.concatenate([a[k] for a in batches]) for k in batches[0]}
    d = {'event_id': np.array([[1430, 920000, 1]]), 'id': np.array([[1430, 920000, 1, 2]])}
    c = dict(cohort='heldout', indices=[100, 101], checks_passed=True, finished_art_events=10000,
             ds_mu_pi_rows=2, coll1_mu_pi_forward_tracks=2)
    return r, d, c


def test_frozen_settings_no_scan():
    assert ts.INDICES == tuple(range(100, 200))
    assert ts.SEED == 20261105
    assert ts.SETTINGS == dict(step_mm=2.5, material_step_mm=.25, loss_fraction=.05,
                              max_path_mm=100000., method='rk4')


def test_complete_exposure_and_zero_source_events_retained(monkeypatch):
    r, d, c = fixture(monkeypatch)
    result = ts.validate(r, d, c)
    assert len(result['event_id']) == 10000
    assert len(result['id']) == 2
    np.testing.assert_array_equal(result['parent_shard_event_counts'], [5000, 5000])


@pytest.mark.parametrize('bad', ['partial', 'count', 'failed', 'mix', 'overlap', 'source', 'ancestry', 'unlinked', 'records'])
def test_invalid_or_mixed_reference_never_silently_drops(monkeypatch, bad):
    r, d, c = fixture(monkeypatch)
    if bad == 'partial':
        c['indices'] = [100]
    elif bad == 'count':
        r['event_id'] = r['event_id'][:-1]
    elif bad == 'failed':
        c['checks_passed'] = False
    elif bad == 'mix':
        r['parent_shard_ids'][0, 1] = 920000
    elif bad == 'overlap':
        d['event_id'] = r['event_id'][:1]
    elif bad == 'source':
        r['id'][0, 2] = 5001
    elif bad == 'ancestry':
        r['endpoint_ancestry_incomplete'][0] = True
    elif bad == 'unlinked':
        r['endpoint_source'][0] = -1
    else:
        c['ds_mu_pi_rows'] = 3
    with pytest.raises(ValueError):
        ts.validate(r, d, c)


@pytest.mark.parametrize('bad', [None, 'geometry', 'status', 'status_count'])
def test_prepare_geometry_and_stock_status_whole_cohort_review(tmp_path, monkeypatch, bad):
    from beamfastsim import tsgeometry
    r, d, c = fixture(monkeypatch)
    development = tmp_path / 'dev.npz'
    np.savez(development, **d)
    monkeypatch.setattr(ts.tscampaign, 'combine', lambda *args, **kwargs: c)
    report = dict(cohort='heldout', source_tracks=2, recorded_source_tracks=2, events=10000,
                  shards=2, stock_reached=2, nested_tracks_represented_by_ancestor=0,
                  upstream_ancestry_incomplete_sources=0)
    monkeypatch.setattr(ts.tsrawreference, 'combine', lambda *args, **kwargs: (r, report))
    class Geometry:
        def inside(self, x):
            return np.full(len(x), bad != 'geometry')
    monkeypatch.setattr(tsgeometry, 'Geometry', lambda *args: Geometry())
    status = dict(events=9999 if bad == 'status_count' else 10000,
                  nonzero_status_events=int(bad == 'status'), overflow_events=0, step_limit_events=0)
    monkeypatch.setattr(ts.tscampaign, 'status_histograms', lambda *args: status)
    out = tmp_path / 'prepared'
    if bad:
        with pytest.raises(ValueError):
            ts.prepare(tmp_path / 'extraction', development, 'fixed.gdml', 'status.root', out)
        assert not out.exists()
    else:
        ts.prepare(tmp_path / 'extraction', development, 'fixed.gdml', 'status.root', out)
        with np.load(out / 'reference.npz') as a:
            np.testing.assert_array_equal(a['event_id'], r['event_id'])
        assert json.loads((out / 'reference.json').read_text())['heldout_disjoint']


def test_once_only_driver_preserves_all_rows_and_metadata(tmp_path, monkeypatch):
    from beamfastsim import tsfield, tsgeometry, tstransport
    r, d, c = fixture(monkeypatch)
    reference, development = tmp_path / 'ref.npz', tmp_path / 'dev.npz'
    np.savez(reference, **r)
    np.savez(development, **d)
    gdml = tmp_path / 'fixed.gdml'
    reference.with_suffix('.json').write_text(json.dumps(dict(census=c, gdml=str(gdml))))
    calls = []
    def transport(source, field, geometry, **settings):
        calls.append(settings)
        for k in ('x', 'p', 't', 'pdg'):
            np.testing.assert_array_equal(source[k], r[k])
        return dict(x=r['x'].copy(), p=r['p'].copy(), t=r['t'] + 1,
                    pdg=np.full(2, 13), fate=np.array([1, 2]), path=np.ones(2), material_path=np.zeros(2),
                    steps=np.ones(2, dtype=int), pion_decay=np.ones(2, dtype=bool), seconds=np.array(1.))
    monkeypatch.setattr(tsfield, 'from_config', lambda *args: object())
    monkeypatch.setattr(tsgeometry, 'Geometry', lambda *args: object())
    monkeypatch.setattr(tstransport, 'transport', transport)
    monkeypatch.setattr(ts, 'plot', lambda *args, **kwargs: None)
    out = tmp_path / 'out'
    ts.run(reference, development, gdml, 'field', 'offline', 'maps', out)
    assert calls == [dict(seed=ts.SEED, **ts.SETTINGS)]
    with np.load(out / 'fast.npz') as f:
        np.testing.assert_array_equal(f['event_id'], r['event_id'])
        np.testing.assert_array_equal(f['id'], r['id'])
        assert len(f['fate']) == 2
    assert json.loads((out / 'numbers.json').read_text())['label'] == 'held-out'
    with pytest.raises(ValueError, match='fresh'):
        ts.run(reference, development, gdml, 'field', 'offline', 'maps', out)
    assert len(calls) == 1
