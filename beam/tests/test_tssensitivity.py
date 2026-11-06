import json

import numpy as np
import pytest

from beamfastsim import tssensitivity as ts
from beamfastsim import tsstudy
from test_tsstudy import tables


def fixture(monkeypatch):
    monkeypatch.setattr(ts, 'EXPECTED', 6)
    monkeypatch.setattr(ts, 'EXPECTED_EVENTS', 7)
    monkeypatch.setattr(tsstudy, 'EXPECTED', 3)
    monkeypatch.setattr(tsstudy, 'EXPECTED_SHARDS', 2)
    r = tables()
    b = dict(id=r['id'].copy(), shard=r['shard'].copy(), pdg=np.array([-13, -13, -13, 13, 13, 13]),
             x=r['x'] + 1, p=r['p'].copy(), t=r['t'] + 20, fate=np.array([1, 4, 1, 1, 2, 2]),
             path=np.ones(6), material_path=np.ones(6), seconds=np.asarray(1.))
    c = {k: v.copy() for k, v in b.items()}
    c['fate'] = np.array([4, 1, 1, 1, 1, 2])
    return r, b, c


def test_fixed_one_candidate_no_seed_scan():
    assert ts.EXPECTED == 25078
    assert ts.SEED == 20261105
    assert ts.BASELINE == (2.5, .25, .05)
    assert ts.TIGHTER == (1.25, .125, .025)


def test_same_cohort_shards_and_population_differences(monkeypatch):
    r, b, c = fixture(monkeypatch)
    result = ts.summarize(r, b, c)
    assert result['parent_events'] == 7
    assert result['independent_shards'] == [[1430, 920000], [1430, 920099]]
    minus = result['charges']['-1']['muons']['mu-']
    assert (minus['baseline'], minus['tighter'], minus['both'], minus['tighter_only']) == (1, 2, 1, 1)
    assert minus['difference'] == pytest.approx(1 / 3)
    assert minus['ratio'] == 2
    plus = result['charges']['1']['muons']['mu+']
    assert (plus['baseline'], plus['tighter'], plus['both'], plus['baseline_only'], plus['tighter_only']) == (2, 2, 1, 1, 1)
    assert result['charges']['1']['ninput_shard'] == [3, 0]
    assert np.sum(result['charges']['1']['fate_transition']) == 3
    for charge, name in (('-1', 'mu-'), ('1', 'mu+')):
        m = result['charges'][charge]['muons'][name]
        for h in m['histograms'].values():
            for arm in ('baseline', 'tighter'):
                assert sum(h[arm]) + sum(h[arm + '_flow']) == m[arm]
    json.dumps(result, allow_nan=False)


def test_exact_join_permutation_and_flow_bins(monkeypatch):
    r, b, c = fixture(monkeypatch)
    c['t'][4] = r['t'][4] + 1000
    c['p'][4] = [0, 0, 300]
    reference = ts.summarize(r, b, c)
    order = np.arange(6)[::-1]
    shuffled = {k: v[order] if v.ndim and len(v) == 6 else v for k, v in c.items()}
    assert ts.summarize(r, b, shuffled) == reference
    minus = reference['charges']['-1']['muons']['mu-']
    for h in minus['histograms'].values():
        assert h['tighter_flow'] == [0, 1]
        assert sum(h['tighter']) == 1


def test_identical_settings_output_has_zero_difference(monkeypatch):
    r, b, _ = fixture(monkeypatch)
    result = ts.summarize(r, b, b)
    for charge in ('-1', '1'):
        for m in result['charges'][charge]['muons'].values():
            assert m['difference'] == m['difference_se'] == 0
            for h in m['histograms'].values():
                assert h['chi2_diagonal'] == 0


def test_reject_missing_input_and_heldout(monkeypatch):
    r, b, c = fixture(monkeypatch)
    c['id'] = c['id'][:-1]
    with pytest.raises(ValueError):
        ts.summarize(r, b, c)
    r['event_id'][-1, 1] = 920100
    with pytest.raises(ValueError, match='development'):
        ts.summarize(r, b, b)


def test_cli_runs_only_full_candidate_and_preserves_baseline(tmp_path, monkeypatch):
    from beamfastsim import tsfield, tsgeometry, tstransport
    r, b, c = fixture(monkeypatch)
    reference, baseline = tmp_path / 'reference.npz', tmp_path / 'baseline.npz'
    np.savez(reference, **r)
    np.savez(baseline, **b)
    baseline.with_suffix('.log').write_text('steps=1\n' + json.dumps(dict(seed=ts.SEED, step_mm=ts.BASELINE[0],
        n=6, seconds=1., fates=dict(zip(ts.FATES, np.bincount(b['fate'], minlength=8).tolist())))) + '\nfinished')
    before = baseline.read_bytes()
    calls = []
    def transport(source, field, geometry, **kwargs):
        calls.append(kwargs)
        for key in ('x', 'p', 't', 'pdg'):
            np.testing.assert_array_equal(source[key], r[key])
        return {k: v.copy() for k, v in c.items() if k not in ('id', 'shard')}
    monkeypatch.setattr(tsfield, 'from_config', lambda *args: object())
    monkeypatch.setattr(tsgeometry, 'Geometry', lambda *args: object())
    monkeypatch.setattr(tstransport, 'transport', transport)
    monkeypatch.setattr(ts, 'plot', lambda *args: None)
    out = tmp_path / 'out'
    args = ['--reference', str(reference), '--baseline', str(baseline), '--out', str(out),
            '--gdml', 'geometry', '--field-config', 'field', '--offline-root', 'offline', '--maps', 'maps']
    ts.main(args)
    assert calls == [dict(seed=20261105, step_mm=1.25, material_step_mm=.125, loss_fraction=.025,
                          max_path_mm=100000., method='rk4')]
    with np.load(out / 'tighter.npz') as f:
        np.testing.assert_array_equal(f['id'], r['id'])
        np.testing.assert_array_equal(f['event_id'], r['event_id'])
    assert baseline.read_bytes() == before
    with pytest.raises(SystemExit):
        ts.main(args)
    assert len(calls) == 1
    ts.main(['--reference', str(reference), '--baseline', str(baseline), '--candidate', str(out / 'tighter.npz'),
             '--out', str(tmp_path / 'summary-only')])
    assert len(calls) == 1
    record = out / 'run.json'
    manifest = json.loads(record.read_text())
    manifest['tighter_settings'][0] = 2.5
    record.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='run record'):
        ts.saved_candidate(out / 'tighter.npz', reference, baseline)


def test_reference_event_total_and_fast_exposure_rejected(monkeypatch):
    r, b, c = fixture(monkeypatch)
    r['event_id'] = r['event_id'][:-1]
    r['parent_shard_event_counts'][-1] = 0
    with pytest.raises(ValueError):
        ts.summarize(r, b, c)
    r, b, c = fixture(monkeypatch)
    c['event_id'] = r['event_id'].copy()
    c['event_id'][-1, 1] = 920100
    with pytest.raises(ValueError, match='exposure differs'):
        ts.summarize(r, b, c)


def test_baseline_log_rejects_wrong_seed(tmp_path, monkeypatch):
    _, b, _ = fixture(monkeypatch)
    path = tmp_path / 'baseline.npz'
    path.with_suffix('.log').write_text(json.dumps(dict(seed=20261111, step_mm=ts.BASELINE[0],
        n=6, seconds=1., fates=dict(zip(ts.FATES, np.bincount(b['fate'], minlength=8).tolist())))))
    with pytest.raises(ValueError, match='baseline log'):
        ts.baseline_record(path, b)
