"""Tiny explicit pair oracles and support-composition controls for retained-array reports."""
import copy
import itertools
import json

import numpy as np
import pytest

from crvfastsim.objective import joint_pairs as P, joint_timing as T


def hits():
    h = dict(group=np.array([0]*5+[1]*4+[2]*3),
             bar_id=np.array([256]*3+[272]*2+[256]*2+[320]*2+[0]*3),
             sipm_id=np.array([1, 3, 3, 1, 3, 0, 2, 0, 2, 0, 2, 2]),
             time_residual=np.array([-2., 0., 2., -1., 1., -2., 2., -3., 3., -2., 0., 2.]),
             pe=np.array([20., 40., 100., 30., 60., 200., 10., 5., 50., 15., 20., 40.]),
             group_count=np.array([5]*5+[4]*4+[3]*3))
    h['slot'], h['body'], h['layer'] = T.G.bar_slot_body_layer(h['bar_id'])
    h['side'] = h['sipm_id'] % 2
    h['along'] = np.full(12, -.2)
    h['path'] = np.full(12, 1.25)
    h['distance'] = np.full(12, 500.)
    h['raw_centered'] = h['time_residual'].copy()
    h['propagation_centered'] = np.zeros(12)
    h['channel'] = 4*h['bar_id']+h['sipm_id']
    h['roc'] = np.ones(12, int)
    h['feb'] = h['bar_id']//32
    h['multiplicity_bin'] = np.searchsorted(T.MULT, h['group_count'], side='right')-1
    return h


def oracle(h):
    result = []
    for i, j in itertools.combinations(range(len(h['group'])), 2):
        if h['group'][i] != h['group'][j] or h['bar_id'][i] != h['bar_id'][j] or h['side'][i] != h['side'][j]:
            continue
        if h['sipm_id'][i] == h['sipm_id'][j]:
            continue
        if h['sipm_id'][i] > h['sipm_id'][j]:
            i, j = j, i
        coords, valid = [], True
        for key, edges, index in (('along', P.J.ALONG, i), ('path', P.J.PATH, i),
                                  ('pe', P.J.PE, i), ('pe', P.J.PE, j)):
            v, ok = P.J._bins(h[key][[index]], edges)
            coords.append(int(v[0]))
            valid &= ok[0]
        if valid:
            result.append((tuple([int(h['body'][i]), int(h['side'][i])] + coords +
                           [int(h['group_count'][i]), int(h['layer'][i])]),
                           int(h['group'][i]), float(h['time_residual'][i]-h['time_residual'][j]),
                           float(h['pe'][i]), float(h['pe'][j])))
    return result


@pytest.mark.parametrize('mode', P.MODES)
def test_vectorized_cells_match_pair_oracle(mode):
    h = hits()
    rows, flow = P.pair_rows(h)
    actual = {tuple(c['key']): c for c in P.cell_table(rows, mode)}
    expected = {}
    for key, group, dt, pe0, pe1 in oracle(h):
        k = list(key[:6])
        if mode != 'none':
            k.append(key[6] if mode == 'exact' else np.searchsorted(T.MULT, key[6], side='right')-1)
        if mode == 'layer_band':
            k.append(key[7])
        expected.setdefault(tuple(k), []).append((group, dt, pe0, pe1))
    assert set(actual) == set(expected)
    for key, values in expected.items():
        array = np.array(values)
        c = actual[key]
        assert c['count'] == len(values)
        assert c['groups'] == len(np.unique(array[:, 0]))
        assert c['mean'] == pytest.approx(array[:, 1].mean())
        assert c['second_moment'] == pytest.approx(np.mean(array[:, 1]**2))
        assert c['pe_low_mean'] == pytest.approx(array[:, 2].mean())
        assert c['pe_high_mean'] == pytest.approx(array[:, 3].mean())
    assert sum(c['count'] for c in actual.values()) == 7
    assert flow['T10']['all_pairs'] == 4
    assert flow['T10']['distinct_event_slots'] == 2
    assert flow['T11']['all_pairs'] == 1
    assert flow['EX0']['all_pairs'] == 2
    json.dumps(flow, allow_nan=False)


def test_same_channel_pulses_all_cross_only_and_sensor_oriented_not_pe_sorted():
    h = hits()
    rows, _ = P.pair_rows(h)
    assert rows['count'].sum() == 7
    selected = (rows['keys'][:, 0] == 4) & (rows['keys'][:, 1] == 0)
    assert rows['keys'][selected, 4:6].tolist() == [[4, 0]]
    assert rows['mean'][selected].tolist() == [-4.]


def test_previous_diagonal_report_is_reproduced_independently():
    h = hits()
    h['pe'][:] = 30.
    diagonal = P.legacy_diagonal(h)
    old = T.pair_report(h)['same_bar_same_end_distinct_sipm_equal_pe_bin']
    P.check_legacy(diagonal, old)
    assert sum(row['pairs'] for row in diagonal) == 7


@pytest.mark.parametrize('corrupt', ['pairs', 'groups', 'difference_second', 'body'])
def test_legacy_gate_rejects_mismatch(corrupt):
    h = hits()
    h['pe'][:] = 30.
    old = P.legacy_diagonal(h)
    actual = copy.deepcopy(old)
    item = next(c for c in actual if c['pairs'])
    item[corrupt] += 1
    with pytest.raises(ValueError, match='legacy pair'):
        P.check_legacy(actual, old)


def test_complete_empty_schema():
    h = {k: v[:0] for k, v in hits().items()}
    rows, flow = P.pair_rows(h)
    assert rows['keys'].shape == (0, 8)
    for mode in P.MODES:
        assert P.cell_table(rows, mode) == []
    assert flow['T10']['all_pairs'] == 0
    json.dumps(flow, allow_nan=False)


@pytest.mark.parametrize('field', ['along', 'path', 'group_count', 'distance', 'body', 'layer'])
def test_inconsistent_same_bar_coordinates_refused(field):
    h = hits()
    h[field][1] += 1
    with pytest.raises(ValueError, match='same-bar/end'):
        P.pair_rows(h)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -1., 200.01])
def test_pe_flows_do_not_fabricate_pairs(bad):
    h = hits()
    h['pe'][0] = bad
    rows, flow = P.pair_rows(h)
    assert rows['count'].sum() == 5
    assert flow['T10']['all_pairs'] == 4
    assert flow['T10']['pairs_with_both_pe_inrange'] == 2


def test_upper_edge_and_coordinate_exclusions_are_explicit():
    h = hits()
    h['pe'][:] = 200.
    h['along'][:3] = 1.
    rows, _ = P.pair_rows(h)
    assert rows['count'].sum() == 7
    h['along'][:3] = 1.001
    rows, flow = P.pair_rows(h)
    assert rows['count'].sum() == 5
    assert flow['T10']['coordinate_excluded_pairs'] == 2


def test_reordering_and_group_locality():
    h = hits()
    before = copy.deepcopy(h)
    rows, flow = P.pair_rows(h)
    reverse, f2 = P.pair_rows({k: v[::-1] for k, v in h.items()})
    assert flow == f2
    for mode in P.MODES:
        assert P.cell_table(rows, mode) == P.cell_table(reverse, mode)
    for k in h:
        np.testing.assert_array_equal(h[k], before[k])
    same = h['group'] == 1
    h['group'][same] = 2**53+5
    r2, _ = P.pair_rows(h)
    assert P.cell_table(rows, 'exact') == P.cell_table(r2, 'exact')


def test_every_body_and_both_sides_without_top_only_selection():
    bars = np.r_[np.arange(8)*64, 512+np.arange(8)*8]
    h = {k: np.repeat(v[:1], 64) for k, v in hits().items()}
    h.update(group=np.repeat(np.arange(16), 4), bar_id=np.repeat(bars, 4),
             sipm_id=np.tile([0, 1, 2, 3], 16), side=np.tile([0, 1, 0, 1], 16),
             time_residual=np.tile([-1., -2., 1., 2.], 16), group_count=np.full(64, 4))
    _, h['body'], h['layer'] = T.G.bar_slot_body_layer(h['bar_id'])
    rows, flow = P.pair_rows(h)
    assert set(rows['keys'][:, 0]) == set(range(16))
    assert set(rows['keys'][:, 1]) == {0, 1}
    assert all(flow[name]['all_pairs'] == 2 for name in T.G.BODY_NAMES)


def synthetic_tables():
    def row(n, count, second, groups=20):
        return dict(key=[4, 1, 2, 0, 1, 1, n], count=count, groups=groups,
                    mean=0., second_moment=second, variance=second, pe_low_mean=30., pe_high_mean=30.,
                    group_mean_second_moment=second)
    return {name: [row(4, 900 if name != 'data' else 100, 1.),
                   row(8, 100 if name != 'data' else 900, 9.)] for name in (*P.J.ARMS, 'data')}


def test_support_requirements_and_fixed_data_weights():
    samples = synthetic_tables()
    tables, common = P.support_tables(samples)
    result = P.standardize(tables, common, 4)
    assert result['common_cells'] == 2
    assert result['samples']['nominal']['marginal']['second_moment'] == pytest.approx(1.8)
    assert result['samples']['nominal']['data_weights']['second_moment'] == pytest.approx(8.2)
    samples['joint'][0]['groups'] = 9
    tables, common = P.support_tables(samples)
    result = P.standardize(tables, common, 4)
    assert result['common_cells'] == 1
    assert result['samples']['nominal']['coverage'] == .1
    samples['joint'][1]['count'] = 29
    tables, common = P.support_tables(samples)
    assert not common
    assert P.standardize(tables, common, 4)['samples']['data']['data_weights']['second_moment'] is None
    assert P.fixed_support_multiplicity(tables, common, 4)['common_exact_cells'] == 0


def test_fixed_support_changes_only_multiplicity_mixture():
    tables, common = P.support_tables(synthetic_tables())
    r = P.fixed_support_multiplicity(tables, common, 4)
    assert r['coarse_cells'] == 1
    s = r['samples']['nominal']
    assert s['pairs'] == 1000
    assert s['collapsed_N_own_mix_data_coarse_weights']['second_moment'] == pytest.approx(1.8)
    assert s['fine_data_weights']['second_moment'] == pytest.approx(8.2)
    assert s['second_moment_reweighting'] == pytest.approx(6.4)
    assert r['samples']['data']['second_moment_reweighting'] == pytest.approx(0.)


def test_group_equal_weights_and_group_cell_incidences_are_not_event_counts():
    rows = dict(keys=np.array([[4, 1, 2, 0, 1, 1, 5, 0]]*3), group=np.array([0, 0, 1]),
                count=np.array([2, 8, 1]), mean=np.zeros(3), second_moment=np.array([1., 1., 9.]),
                variance=np.array([1., 1., 9.]),
                pe_low_mean=np.full(3, 30.), pe_high_mean=np.full(3, 30.))
    c = P.cell_table(rows, 'band')[0]
    assert c['groups'] == 2
    assert c['group_mean_second_moment'] == 5.
    assert c['second_moment'] == pytest.approx(19/11)


def test_comparison_refuses_existing_output_and_wrong_prior(tmp_path):
    path = tmp_path/'out.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='already exists'):
        P.compare('/not-present', '/not-present', '/not-present', path)
    with pytest.raises(ValueError, match='invalid prior timing'):
        P.compare('/not-present', '/not-present', path, tmp_path/'new.json')


def test_signed_difference_has_stable_second_moment_with_large_common_offset():
    h = hits()
    before = P.cell_table(P.pair_rows(h)[0], 'band')
    h['time_residual'] += 1e8
    after = P.cell_table(P.pair_rows(h)[0], 'band')
    for a, b in zip(before, after):
        assert a['second_moment'] == pytest.approx(b['second_moment'])
        assert a['mean'] == pytest.approx(b['mean'])


@pytest.mark.parametrize('coordinate,edge', [('along', 1.), ('along', -.2), ('path', 1.25)])
def test_near_edge_coordinate_disagreement_rejected_regardless_of_order(coordinate, edge):
    h = hits()
    h[coordinate][:3] = edge
    h[coordinate][1] = edge+(5e-11 if edge == 1. else -5e-11)
    for order in (np.arange(12), np.arange(11, -1, -1)):
        with pytest.raises(ValueError, match='bin or validity mismatch'):
            P.pair_rows({k: v[order] for k, v in h.items()})


def test_literal_pe_bin_boundaries():
    bins, valid = P.J._bins(np.array([0., 19.999, 20., 40., 60., 100., 200., 200.001]), P.J.PE)
    assert bins.tolist() == [0, 0, 1, 2, 3, 4, 4, 5]
    assert valid.tolist() == [True]*7+[False]


@pytest.mark.parametrize('mixed', [False, True])
def test_two_by_three_sensor_multiplicity_both_nonzero_variances(mixed):
    h = {k: v[:5].copy() for k, v in hits().items()}
    h.update(bar_id=np.full(5, 256), sipm_id=np.array([1, 1, 3, 3, 3]),
             layer=np.zeros(5, int), side=np.ones(5, int), pe=np.full(5, 30.),
             time_residual=np.array([-3., 1., 0., 2., 6.]))
    if mixed:
        h['pe'] = np.array([10., 30., 15., 40., 60.])
    rows, flow = P.pair_rows(h)
    result = P.cell_table(rows, 'none')
    assert flow['T10']['all_pairs'] == 6
    explicit = oracle(h)
    total = sum(c['count'] for c in result)
    assert total == 6
    assert sum(c['count']*c['mean'] for c in result)/total == pytest.approx(np.mean([v[2] for v in explicit]))
    assert sum(c['count']*c['second_moment'] for c in result)/total == pytest.approx(np.mean([v[2]**2 for v in explicit]))
    if not mixed:
        assert len(result) == 1 and result[0]['groups'] == 1
        assert result[0]['variance'] == pytest.approx(np.var([v[2] for v in explicit]))


def test_stable_variance_for_large_signed_sensor_difference():
    h = {k: v[:3].copy() for k, v in hits().items()}
    h['pe'][:] = 30.
    h['time_residual'] = np.array([1e8, 0., 1.])
    cells = P.cell_table(P.pair_rows(h)[0], 'none')
    assert cells[0]['variance'] == pytest.approx(.25)
    assert P._moment(cells, [1.])['variance'] == pytest.approx(.25)


def test_fixed_support_multiple_coarse_cells_and_excluded_fine_cell():
    samples = synthetic_tables()
    for name, cells in samples.items():
        extra = copy.deepcopy(cells)
        for c in extra:
            c['key'][2] = 3
            c['second_moment'] *= 2
            c['count'] *= 3 if name == 'data' else 7
        cells.extend(extra)
        unsupported = copy.deepcopy(extra[0])
        unsupported['key'][6] = 10
        unsupported['count'] = 29 if name == 'joint' else 1000
        unsupported['second_moment'] = 1000.
        cells.append(unsupported)
    tables, common = P.support_tables(samples)
    result = P.fixed_support_multiplicity(tables, common, 4)
    assert result['common_exact_cells'] == 4 and result['coarse_cells'] == 2
    nominal = result['samples']['nominal']
    assert nominal['pairs'] == 8000
    assert nominal['fine_data_weights']['second_moment'] == pytest.approx(.25*8.2+.75*16.4)
    assert nominal['collapsed_N_own_mix_data_coarse_weights']['second_moment'] == pytest.approx(.25*1.8+.75*3.6)
    assert result['samples']['data']['second_moment_reweighting'] == pytest.approx(0.)


def test_prior_binding_rejects_missing_zero_hit_exposure_shard_and_exposure_change():
    prior = {'/a.npz': 'a', '/empty.npz': 'b', '/code.py': 'c'}
    current = {'/a.npz': {'sha256': 'a'}}
    with pytest.raises(ValueError, match='capture set'):
        P.bind_captures(prior, current, 100, 90)
    current['/empty.npz'] = {'sha256': 'b'}
    with pytest.raises(ValueError, match='data exposure'):
        P.bind_captures(prior, current, 100, 90)
    P.bind_captures(prior, current, 100, 100)


def test_summary_multiplicity_leverage_reports_single_N_limitation():
    from crvfastsim.objective import joint_pairs_report as R
    row = dict(keys=[[4, 1, 2, 0, 1, 1, 4], [4, 1, 2, 0, 1, 1, 8], [4, 1, 3, 0, 1, 1, 4]],
               data_pair_weights=[.1, .2, .7])
    r = R.multiplicity_leverage(row)
    assert r['coarse_cells'] == 2 and r['multi_N_coarse_cells'] == 1
    assert r['data_weight_in_multi_N_coarse_cells'] == pytest.approx(.3)
    assert R.multiplicity_leverage(dict(keys=[], data_pair_weights=[]))['coarse_cells'] == 0


def small_report():
    rows, flow = P.pair_rows(hits())
    sample = dict(legacy_pair_census_and_moments_equal=True, pair_flow=flow,
                  strata={mode: P.cell_table(rows, mode) for mode in P.MODES})
    return dict(comparison='joint-pairs', passed=True, samples={'data': sample},
                standardization={'exact': {}}, fixed_support_multiplicity={})


def test_saved_pair_summary_checks_data_equality_and_flow():
    from crvfastsim.objective import joint_pairs_report as R
    reports = dict(A=small_report(), B=small_report())
    assert R.summarize(reports)['identical_measured_data_diagnostics']
    reports['B']['samples']['data']['pair_flow']['T10']['all_pairs'] += 1
    with pytest.raises(ValueError, match='flow'):
        R.summarize(reports)


def test_saved_pair_summary_refuses_numerical_or_data_mismatch():
    from crvfastsim.objective import joint_pairs_report as R
    r = small_report()
    r['samples']['data']['strata']['none'][0]['variance'] += 1
    with pytest.raises(ValueError, match='variance'):
        R.validate(r)
    a, b = small_report(), small_report()
    b['samples']['data']['pair_flow']['T10']['all_second_moment'] += 1
    with pytest.raises(ValueError, match='reductions differ'):
        R.summarize(dict(A=a, B=b))
