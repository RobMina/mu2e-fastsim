"""Event-normalization contracts for explicit multi-muon DIO source schedules."""
import json

import numpy as np
import pytest

from beamfastsim import target_source_events as E
from beamfastsim import target_sources as S


def parent():
    return dict(run=np.ones(4, np.int64), subrun=np.full(4, 2, np.int64),
        evt=np.array([10, 20, 20, 20]), trk=np.arange(4), pdg=np.full(4, 13),
        codestop=np.full(4, 32), at_rest=np.ones(4, bool), foil=np.arange(4),
        event_index=np.array([0, 1, 1, 1]), event_keys=np.array([[1, 2, 10], [1, 2, 20]]),
        event_multiplicity=np.array([1, 3]), xstop=np.arange(4.), ystop=np.zeros(4),
        zstop=np.ones(4), tstop=np.arange(4.) + 350.)


def test_all_muons_emitted_and_event_exposure_preserved_across_wraps():
    p = parent()
    schedule = np.array([1, 0, 1, 1])
    out = E.dio_events(np.random.default_rng(9), p, schedule, S.BinnedSpectrum([95, 100], [1]))
    assert np.array_equal(out['gencount'], [4])
    assert out['pdg'].size == 10
    assert np.array_equal(out['event_multiplicity'], [3, 1, 3, 3])
    assert np.array_equal(out['event_index'], [0, 0, 0, 1, 2, 2, 2, 3, 3, 3])
    assert np.array_equal(out['stop_idx'], [1, 2, 3, 0, 1, 2, 3, 1, 2, 3])
    assert np.array_equal(out['parent_evt'], p['evt'][out['stop_idx']])
    assert np.array_equal(out['scheduled_parent_event_keys'], p['event_keys'][schedule])
    assert np.array_equal(out['x'], p['xstop'][out['stop_idx']])
    assert np.all(out['time'] >= p['tstop'][out['stop_idx']])
    assert np.allclose(np.linalg.norm(out['u'], axis=1), 1)
    assert np.all((np.hypot(out['p'], S.ME) >= 95) & (np.hypot(out['p'], S.ME) < 100))
    assert np.allclose(out['px'], out['p'] * out['u'][:, 0])


def test_physical_event_order_need_not_be_row_order():
    p = parent()
    p['event_index'] = np.array([1, 0, 0, 0])
    p['event_keys'] = p['event_keys'][::-1]
    p['event_multiplicity'] = np.array([3, 1])
    out = E.dio_events(np.random.default_rng(1), p, np.array([0, 1]), S.BinnedSpectrum([95, 100], [1]))
    assert np.array_equal(out['stop_idx'], [1, 2, 3, 0])
    assert np.array_equal(out['parent_evt'], [20, 20, 20, 10])


def test_population_draw_is_event_not_flat_stop_sampling_and_retains_repeated_occurrences():
    a = E.draw_schedule(np.random.default_rng(1), np.array([0, 1]), 20000, 2)
    assert np.mean(a == 0) == pytest.approx(.5, abs=.02)
    b = E.draw_schedule(np.random.default_rng(1), np.array([0, 0, 1]), 20000, 2)
    assert np.mean(b == 0) == pytest.approx(2 / 3, abs=.02)
    assert np.array_equal(a, E.draw_schedule(np.random.default_rng(1), np.array([0, 1]), 20000, 2))


@pytest.mark.parametrize('schedule', [np.array([]), np.array([-1]), np.array([2]), np.array([0.5]),
                                     np.array([True]), np.array([[0]])])
def test_invalid_schedules_fail(schedule):
    with pytest.raises(ValueError):
        E.dio_events(np.random.default_rng(1), parent(), schedule, S.BinnedSpectrum([95, 100], [1]))


@pytest.mark.parametrize('key,value', [
    ('event_multiplicity', np.array([1, 2])), ('event_multiplicity', np.array([0, 4])),
    ('event_index', np.array([0, 1, 1, 2])), ('event_keys', np.array([[1, 2, 10], [1, 2, 10]])),
    ('evt', np.array([10, 20, 20, 21])), ('trk', np.zeros(4, int)),
    ('codestop', np.array([32, 32, 32, 1])), ('at_rest', np.ones(4)),
    ('foil', np.array([0, 1, 2, -1])), ('tstop', np.array([350, 351, 352, np.nan])),
    ('weight', np.ones(4))])
def test_parent_contracts_fail_closed(key, value):
    p = parent()
    p[key] = value
    with pytest.raises(ValueError):
        E.parent_layout(p)


def test_empty_or_missing_parent_and_invalid_lifetime():
    with pytest.raises(ValueError):
        E.parent_layout({})
    with pytest.raises(ValueError):
        E.dio_events(np.random.default_rng(1), parent(), np.array([0]), S.BinnedSpectrum([95, 100], [1]), tau=np.nan)
    with pytest.raises(ValueError):
        E.dio_events(np.random.default_rng(1), parent(), np.array([0]), S.BinnedSpectrum([0, .1], [1]))


@pytest.mark.parametrize('population,n,parents', [(np.array([2]), 5, 2), (np.array([]), 5, 2),
    (np.array([0.5]), 5, 2), (np.array([0]), True, 2), (np.array([0]), 0, 2), (np.array([0]), 2, 0)])
def test_invalid_population_draw(population, n, parents):
    with pytest.raises(ValueError):
        E.draw_schedule(np.random.default_rng(1), population, n, parents)


def test_cli_hashes_event_denominators_and_exclusive_output(tmp_path):
    p, pop, table, output = (tmp_path / k for k in ('parent.npz', 'population.npz', 'spectrum.tbl', 'source.npz'))
    np.savez(p, **parent())
    np.savez(pop, event_index=np.array([1, 1]))
    table.write_text('95 3\n96 2\n97 1\n')
    argv = ['--parent', str(p), '--population', str(pop), '--n-events', '10', '--schedule-seed', '71',
            '--source-seed', '83', '--spectrum-table', str(table), '--elow', '95', '--ehi', '98', '--out', str(output)]
    E.main(argv)
    with np.load(output, allow_pickle=False) as out:
        meta = json.loads(str(out['metadata']))
        assert meta['generated_events'] == 10 and meta['generated_electrons'] == 30
        assert meta['schedule_seed'] == 71 and meta['source_seed'] == 83
        assert all(len(meta[k]) == 64 for k in ('parent_sha256', 'population_sha256', 'spectrum_sha256'))
        assert not meta['certificate_established'] and not meta['source_validated']
        assert np.array_equal(out['event_multiplicity'], np.full(10, 3))
        assert np.array_equal(out['event_index'], np.repeat(np.arange(10), 3))
    with pytest.raises(FileExistsError):
        E.main(argv)


@pytest.mark.parametrize('schedule_seed,source_seed', [('1', '1'), ('-1', '2'), ('1', '-2')])
def test_cli_requires_distinct_nonnegative_rng_seeds(tmp_path, schedule_seed, source_seed):
    with pytest.raises(ValueError, match='distinct nonnegative'):
        E.main(['--parent', 'absent', '--population', 'absent', '--n-events', '1',
                '--schedule-seed', schedule_seed, '--source-seed', source_seed,
                '--spectrum-table', 'absent', '--elow', '95', '--ehi', '100', '--out', str(tmp_path / 'out.npz')])


def test_adapter_consumes_parent_auditor_output_with_floating_tree_identity(monkeypatch):
    from beamfastsim import target_parent_audit as A
    p = parent()
    columns = {k: np.asarray(p[k], float) for k in ('run', 'subrun', 'evt', 'trk', 'pdg', 'codestop',
                                                  'xstop', 'ystop', 'zstop', 'tstop')}
    columns['pstop'] = np.zeros(4)
    monkeypatch.setattr(A.G, 'foil_index', lambda geom, x, y, z: np.zeros(len(x), int))
    audited, report = A.reduce_parent(columns, None, expected_events=2)
    out = E.dio_events(np.random.default_rng(1), audited, np.array([1, 0]), S.BinnedSpectrum([95, 100], [1]))
    assert report['eligible_muons'] == 4
    assert np.array_equal(out['event_multiplicity'], [3, 1])
    assert np.array_equal(out['parent_evt'], [20, 20, 20, 10])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_integral_floating_auditor_identities_are_accepted_with_safe_conversion(dtype):
    p = parent()
    for k in ('run', 'subrun', 'evt', 'trk', 'event_keys'):
        p[k] = p[k].astype(dtype)
    counts, groups, keys = E.parent_layout(p)
    assert keys.dtype == np.int64
    out = E.dio_events(np.random.default_rng(1), p, np.array([1]), S.BinnedSpectrum([95, 100], [1]))
    assert out['pdg'].size == 3
    p['event_keys'][0, 0] = 2 ** (np.finfo(dtype).nmant + 1)
    with pytest.raises(ValueError, match='unsafe floating'):
        E.parent_layout(p)


def test_source_rng_is_reproducible_and_separate_from_schedule_rng():
    p, spec = parent(), S.BinnedSpectrum([95, 100], [1])
    a = E.dio_events(np.random.default_rng(1), p, np.array([1, 0]), spec)
    b = E.dio_events(np.random.default_rng(1), p, np.array([1, 0]), spec)
    c = E.dio_events(np.random.default_rng(2), p, np.array([1, 0]), spec)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert np.array_equal(a['stop_idx'], c['stop_idx'])
    assert not np.array_equal(a['p'], c['p'])
