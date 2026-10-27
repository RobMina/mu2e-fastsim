"""Explicit identity, pair-covariance and conditional full-event uncertainty oracles."""
import copy
import itertools

import awkward as ak
import numpy as np
import pytest

from crvfastsim.objective import joint_shapes as A, joint_shape_stats as S, joint_pairs as P, joint_shape_calib as C
from test_joint_pairs import hits


def join_input():
    keys = np.array([[7, 2, 2**53+9], [7, 3, 2**53+1], [8, 1, 1]], dtype=np.int64)
    c = {'event_slot': np.tile(np.arange(4), 3), 'event_source_index': np.zeros(12, int),
         'hit_source_index': np.zeros(2, int), 'hit_pulse_index': np.array([1, 0]),
         'hit_bar_id': np.array([256, 320]), 'hit_sipm_id': np.array([3, 1]),
         'hit_sequence_index': np.zeros(2, int), 'hit_chosen_cluster': np.array([2, 3]),
         'hit_time': np.array([101., 202.]), 'hit_pe_raw': np.array([30., -1.]), 'hit_pe': np.array([30., 0.])}
    for i, name in enumerate(('run', 'subrun', 'event')):
        c['event_'+name] = np.repeat(keys[:, i], 4)
        c['hit_'+name] = keys[[0, 1], i]
    pulses = {name: ak.Array(value) for name, value in dict(
        barId=[[256, 256], [320], []], SiPMId=[[1, 3], [1], []], sequenceIndex=[[0, 0], [0], []],
        crvHitIndex=[[2, 2], [3], []], time=[[100., 101.], [202.], []], PEs=[[25., 30.], [-1.], []],
        pulseBeta=[[12., 13.], [14.], []], pulseFitChi2=[[2., 3.], [-1.], []],
        pulseHeight=[[50., 60.], [70.], []], PEsPulseHeight=[[20., 21.], [22.], []]).items()}
    return c, keys, pulses


def test_exact_same_product_join_preserves_large_full_keys_and_empty_events():
    c, keys, p = join_input()
    before = {k: v.copy() for k, v in c.items()}
    joined, audit = A.join_fields(c, keys, p, source_index=0, entry_start=25)
    np.testing.assert_array_equal(joined['hit_beta'], [13., 14.])
    np.testing.assert_array_equal(joined['hit_entry_index'], [25, 26])
    assert audit == dict(events=3, hits=2, exact_identity_and_values=True, nonfinite_time=0, nonfinite_pe=0)
    assert all(np.array_equal(c[k], v) for k, v in before.items())


def test_event_reordering_uses_keys_not_row_position():
    c, keys, p = join_input()
    order = [2, 0, 1]
    joined, _ = A.join_fields(c, keys[order], {k: v[order] for k, v in p.items()}, source_index=0, entry_start=0)
    np.testing.assert_array_equal(joined['hit_beta'], [13., 14.])
    np.testing.assert_array_equal(joined['hit_entry_index'], [1, 2])


@pytest.mark.parametrize('field', ['hit_bar_id', 'hit_sipm_id', 'hit_sequence_index', 'hit_chosen_cluster',
                                  'hit_time', 'hit_pe_raw', 'hit_pe', 'hit_pulse_index', 'hit_source_index'])
def test_join_corruption_rejected(field):
    c, keys, p = join_input()
    c[field][0] += 1
    with pytest.raises(ValueError):
        A.join_fields(c, keys, p, source_index=0, entry_start=0)


def test_missing_or_duplicate_full_event_key_rejected():
    c, keys, p = join_input()
    keys[2] = keys[0]
    with pytest.raises(ValueError, match='duplicate ROOT'):
        A.join_fields(c, keys, p, source_index=0, entry_start=0)


def shape_hits():
    h = hits()
    h.update(beta=np.array([12., 13., 19., 11., 15., 12., 15., 14., 17., 11., 20., 16.]),
             chi2=np.array([2., 3., -1., 5., 2., 8., 9., 12., 3., 1., 4., 7.]),
             height_pe=np.arange(12, dtype=float)+20., pulse_height=np.arange(12, dtype=float)+100.)
    return h


def explicit_pairs(h):
    result = []
    for i, j in itertools.combinations(range(len(h['group'])), 2):
        if any(h[k][i] != h[k][j] for k in ('group', 'bar_id', 'side')) or h['sipm_id'][i] == h['sipm_id'][j]:
            continue
        if h['sipm_id'][i] > h['sipm_id'][j]:
            i, j = j, i
        bins, valid = [], True
        for field, edges, index in (('along', P.J.ALONG, i), ('path', P.J.PATH, i),
                                    ('pe', P.J.PE, i), ('pe', P.J.PE, j)):
            ib, ok = P.J._bins(h[field][[index]], edges)
            bins.append(int(ib[0]))
            valid &= ok[0]
        for field in ('beta', 'height_pe', 'pulse_height', 'chi2'):
            valid &= np.isfinite(h[field][[i, j]]).all()
        valid &= (h['beta'][[i, j]] > 0).all() and (h['height_pe'][[i, j]] > 0).all()
        valid &= (h['pulse_height'][[i, j]] >= 0).all()
        if not valid:
            continue
        dt, db = h['time_residual'][i]-h['time_residual'][j], h['beta'][i]-h['beta'][j]
        key = tuple([h['body'][i], h['side'][i], *bins, h['group_count'][i], h['layer'][i]])
        result.append(dict(key=key, group=h['group'][i], dt=dt, dbeta=db, dt2=dt*dt, dbeta2=db*db,
                           dt_dbeta=dt*db, le2=(dt-S.ALPHA*db)**2, beta_low=h['beta'][i], beta_high=h['beta'][j]))
    return result


def test_shape_sufficient_statistics_match_all_cross_sensor_pairs():
    h = shape_hits()
    rows, flow = S.pair_rows(h)
    expected = explicit_pairs(h)
    assert rows['count'].sum() == len(expected) == 7
    assert flow['shape_invalid_pulses'] == 0
    for i, key in enumerate(rows['keys']):
        values = [r for r in expected if r['key'] == tuple(key) and r['group'] == rows['group'][i]]
        assert len(values) == rows['count'][i]
        for field in ('dt', 'dbeta', 'dt2', 'dbeta2', 'dt_dbeta', 'le2', 'beta_low', 'beta_high'):
            assert rows[field][i] == pytest.approx(np.mean([v[field] for v in values]))


def test_nonfinite_shape_pulse_exclusion_is_explicit_and_does_not_change_N():
    h = shape_hits()
    h['beta'][1] = np.nan
    before = h['group_count'].copy()
    rows, flow = S.pair_rows(h)
    assert rows['count'].sum() == len(explicit_pairs(h)) == 6
    assert flow['shape_invalid_pulses'] == 1
    np.testing.assert_array_equal(before, h['group_count'])


def test_same_bin_repeated_sensors_and_large_time_offset_stable():
    h = shape_hits()
    h['pe'][:] = 30.
    a, _ = S.pair_rows(h)
    h['time_residual'] += 1e8
    b, _ = S.pair_rows(h)
    for field in S.MOMENTS:
        np.testing.assert_allclose(a[field], b[field], atol=1e-7, rtol=1e-8)


def test_both_sensors_repeated_same_bin_explicit_covariance_oracle():
    base = shape_hits()
    take = np.array([0, 0, 1, 2])
    h = {k: v[take].copy() for k, v in base.items()}
    h['pe'][:] = 30.
    h['time_residual'][:] = [1., 3., 2., 7.]
    h['beta'][:] = [10., 14., 11., 18.]
    rows, _ = S.pair_rows(h)
    expected = explicit_pairs(h)
    assert rows['count'].tolist() == [4]
    for field in ('dt', 'dbeta', 'dt2', 'dbeta2', 'dt_dbeta', 'le2', 'beta_low', 'beta_high'):
        assert rows[field][0] == pytest.approx(np.mean([v[field] for v in expected]))


def test_nearly_perfect_beta_time_correlation_has_nonnegative_le_moment():
    h = shape_hits()
    h['pe'][:] = 30.
    h['time_residual'] = S.ALPHA*h['beta'] + 1e8
    rows, _ = S.pair_rows(h)
    assert np.all(rows['le2'] >= 0)
    np.testing.assert_allclose(rows['le2'], 0, atol=1e-12)


def test_full_event_se_not_pair_or_slot_se():
    h = shape_hits()
    h['group'] = 4*h['group']
    rows, _ = S.pair_rows(h)
    keys = [list(k) for k in np.unique(S.mode_keys(rows, 'exact'), axis=0)]
    w = np.arange(1, len(keys)+1, dtype=float)
    w /= w.sum()
    result = S.fixed_support(rows, keys, w, 'exact', events=3)
    assert result['supported'] and result['pairs'] == 7
    assert result['le_minus_peak'] == pytest.approx(result['algebra']['beta_term']+result['algebra']['covariance_term'])
    assert result['algebra']['residual'] == pytest.approx(0., abs=1e-12)
    assert result['event_universe'] == 3
    # Identical rows duplicated into another slot of the same full event add no independent information.
    doubled = {k: np.concatenate((v, v.copy())) for k, v in rows.items()}
    doubled['group'][len(rows['group']):] += 1
    same = S.fixed_support(doubled, keys, w, 'exact', events=3)
    assert same['conditional_event_se'] == pytest.approx(result['conditional_event_se'])


def test_event_influence_matches_explicit_ratio_linearization():
    rows = dict(keys=np.array([[4, 1, 0, 0, 1, 1, 3, 0]]*3), group=np.array([0, 4, 8]),
                count=np.array([1, 2, 3]))
    values = np.array([1., 4., 9.])
    for field in S.MOMENTS:
        rows[field] = values if field in ('dt2', 'le2') else np.zeros(3)
    r = S.fixed_support(rows, [rows['keys'][0, :7].tolist()], [1.], 'exact', 3)
    mean = np.dot(values, rows['count'])/6
    psi = rows['count']*(values-mean)/6
    assert r['means']['dt2'] == pytest.approx(mean)
    assert r['conditional_event_se']['dt2'] == pytest.approx(np.sqrt(3/2*np.dot(psi, psi)))
    assert r['le_minus_peak_conditional_event_se'] == 0


def test_missing_support_fails_explicitly_and_weights_are_not_refitted():
    rows, _ = S.pair_rows(shape_hits())
    missing = [[4, 1, 4, 3, 4, 4, 100]]
    assert not S.fixed_support(rows, missing, [1.], 'exact', 3)['supported']
    for weights in ([.5], [1.000005]):
        with pytest.raises(ValueError, match='weights'):
            S.fixed_support(rows, missing, weights, 'exact', 3)


def test_vectorized_channel_quantiles_match_numpy():
    channel = np.array([3, 1, 3, 1, 2, 3])
    values = np.array([10., 8., 2., 4., 7., 1.])
    rows = C.grouped_quantiles(channel, values)
    for i, c in enumerate(rows['channel']):
        np.testing.assert_allclose(rows['quantiles'][i], np.quantile(values[channel == c], [0, .16, .5, .84, 1]))
    assert C.grouped_quantiles(np.empty(0, int), np.empty(0))['quantiles'].shape == (0, 5)


def test_implied_gains_recover_known_gumbel_convention_without_fitting():
    h = shape_hits()
    h['channel'] = 4*h['bar_id'] + h['sipm_id']
    h['height_pe'] = h['pulse_height']/15.
    h['pe'] = np.e*h['pulse_height']*h['beta']/350.
    channels = np.unique(h['channel'])
    table = np.zeros((len(channels), 9))
    table[:, 0], table[:, 2], table[:, 3] = channels, 400., 10.
    report = C.audit(h, table)
    assert not report['calibration_changed'] and not report['historical_constants_certified']
    for row in report['channels']:
        assert row['area_median_ratio_to_mc'] == pytest.approx(.875)
        assert row['height_median_ratio_to_mc'] == pytest.approx(1.5)
        assert row['area_full_range_relative'] < 1e-14
    h['pe'][0] = 0
    assert C.audit(h, table)['excluded_pulses'] == 1


def test_legacy_cell_reproduction_rejects_corruption():
    rows, _ = P.pair_rows(shape_hits())
    cells = P.cell_table(rows, 'exact')
    assert S.compare_cells(cells, cells)['full_census_and_moments_equal']
    bad = copy.deepcopy(cells)
    bad[0]['second_moment'] += 1
    with pytest.raises(ValueError, match='historical second_moment'):
        S.compare_cells(bad, cells)
