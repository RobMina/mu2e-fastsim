"""Vectorized timing reductions checked against tiny explicit pair oracles."""
import copy
import itertools
import json

import numpy as np
import pytest

from crvfastsim.objective import joint_timing as T
from crvfastsim.objective import joint_compare as J
from test_joint_diagnosis import sample, accumulated
from crvfastsim.objective.joint import capture


def mapping():
    ch = np.arange(2304, dtype=np.int64)
    return np.column_stack((ch, ch//1024, ch//64))


def captured():
    c, p, keys = sample()
    _, o = accumulated(c, p)
    return capture(c, p, o, keys)


def hits():
    body = np.repeat([4, 4, 5], [4, 5, 3])
    group = np.repeat(np.arange(3), [4, 5, 3])
    r = np.array([-3., -1., 1., 3., -2., -1., 0., 1., 2., -3., 1., 2.])
    bar = np.array([256, 256, 272, 272, 256, 256, 256, 272, 272, 320, 320, 320])
    sipm = np.array([0, 2, 0, 2, 0, 0, 2, 1, 3, 0, 2, 1])
    prop = np.array([-1., -1., 1., 1., 2., 2., 2., -3., -3., -1., -1., 2.])
    return dict(body=body, group=group, time_residual=r, propagation_centered=prop, raw_centered=r+prop,
                bar_id=bar, sipm_id=sipm, side=sipm % 2, channel=4*bar+sipm, layer=(bar % 64)//16,
                roc=np.ones(12, int), feb=(bar % 64)//16, multiplicity_bin=np.repeat([0, 1, 0], [4, 5, 3]),
                group_count=np.repeat([4, 5, 3], [4, 5, 3]), pe=np.full(12, 30.),
                along=np.zeros(12), path=np.ones(12))


def test_prepare_all_bodies_exact_keys_and_no_mutation():
    cols = captured()
    before = copy.deepcopy(cols)
    h, flow = T.prepare(cols, mapping())
    assert flow['retained_members'] == 48
    assert flow['retained_event_slots'] == 4
    assert set(h['body']) == set(range(16))
    for name in cols:
        np.testing.assert_array_equal(cols[name], before[name])
    result = T.analyze(cols, mapping())
    json.dumps(result, allow_nan=False)
    assert set(result['event_multiplicity']) == set(T.G.BODY_NAMES)


@pytest.mark.parametrize('change,match', [
    ('count', 'incomplete'), ('layer', 'hardware'), ('side', 'hardware'),
    ('sipm', 'invalid channel'), ('residual', 'centered')])
def test_prepare_refuses_corrupt_capture(change, match):
    cols = captured()
    field = dict(count='group_count', layer='layer', side='side', sipm='sipm_id', residual='time_residual')[change]
    cols['hit_' + field][0] += 5
    with pytest.raises(ValueError, match=match):
        T.prepare(cols, mapping())


def test_nonfinite_member_excludes_whole_group_not_finite_subset():
    cols = captured()
    cols['hit_time'][0] = np.nan
    h, flow = T.prepare(cols, mapping())
    assert flow['nonfinite_members'] == 1
    assert flow['excluded_nonfinite_group_members'] == 12
    assert len(h['body']) == 36


def test_missing_map_channel_is_not_assigned_to_zero():
    with pytest.raises(ValueError, match='unmapped'):
        T.prepare(captured(), mapping()[1:])


@pytest.mark.parametrize('bad', ['duplicate', 'negative', 'fractional', 'hash'])
def test_map_fails_closed(tmp_path, bad):
    path = tmp_path/'map.csv'
    table = np.zeros((2, 9))
    table[:, 0] = [0, 1]
    if bad == 'duplicate':
        table[1, 0] = 0
    if bad == 'negative':
        table[1, 6] = -1
    if bad == 'fractional':
        table[1, 7] = 1.5
    np.savetxt(path, table, delimiter=',')
    with pytest.raises(ValueError):
        T.channel_map(path, {}, 'a'*64 if bad == 'hash' else T.V.digest(path))


@pytest.mark.parametrize('relation', ['all', 'same_layer', 'different_layer', 'same_sipm_index',
    'different_sipm_index', 'same_mapped_feb', 'different_mapped_feb', 'same_channel',
    'same_bar_same_end_distinct_sipm', 'same_bar_opposite_end'])
def test_pair_sufficient_statistics_against_oracle(relation):
    h = hits()
    rows = T.pair_report(h)[relation]
    def accepts(i, j):
        same_layer = h['layer'][i] == h['layer'][j]
        same_sipm = h['sipm_id'][i] == h['sipm_id'][j]
        same_feb = (h['roc'][i], h['feb'][i]) == (h['roc'][j], h['feb'][j])
        same_bar = h['bar_id'][i] == h['bar_id'][j]
        same_end = h['side'][i] == h['side'][j]
        return dict(all=True, same_layer=same_layer, different_layer=not same_layer,
            same_sipm_index=same_sipm, different_sipm_index=not same_sipm,
            same_mapped_feb=same_feb, different_mapped_feb=not same_feb,
            same_channel=same_bar and same_sipm,
            same_bar_same_end_distinct_sipm=same_bar and same_end and not same_sipm,
            same_bar_opposite_end=same_bar and not same_end)[relation]
    for body, band in ((4, 0), (4, 1), (5, 0)):
        pairs = [(i, j) for i, j in itertools.combinations(range(len(h['body'])), 2)
                 if h['group'][i] == h['group'][j] and h['body'][i] == h['body'][j] == body
                 and h['multiplicity_bin'][i] == band and accepts(i, j)]
        row = rows[body*6+band]
        assert row['pairs'] == len(pairs)
        assert row['groups'] == len({h['group'][i] for i, j in pairs})
        if not pairs:
            assert row['difference_second'] is None
            continue
        i, j = np.array(pairs).T
        r, a, b = (h[k] for k in ('time_residual', 'raw_centered', 'propagation_centered'))
        ends = np.r_[r[i], r[j]]
        assert row['endpoint_sum'] == pytest.approx(ends.mean())
        assert row['product'] == pytest.approx(np.mean(r[i]*r[j]))
        assert row['difference_second'] == pytest.approx(np.mean((r[i]-r[j])**2))
        assert row['raw_difference_second'] == pytest.approx(np.mean((a[i]-a[j])**2))
        assert row['propagation_difference_second'] == pytest.approx(np.mean((b[i]-b[j])**2))
        assert row['raw_propagation_difference'] == pytest.approx(np.mean((a[i]-a[j])*(b[i]-b[j])))
        assert row['difference_second'] == pytest.approx(row['raw_difference_second']+
               row['propagation_difference_second']-2*row['raw_propagation_difference'])


def test_propagation_cancels_same_bar_end_and_equal_pe_bins():
    h = hits()
    report = T.pair_report(h)
    for row in report['same_bar_same_end_distinct_sipm']:
        if row['pairs']:
            assert row['propagation_difference_second'] == pytest.approx(0, abs=1e-12)
            assert row['difference_second'] == pytest.approx(row['raw_difference_second'])
    rows = report['same_bar_same_end_distinct_sipm_equal_pe_bin']
    assert sum(r['pairs'] for r in rows) == 6
    h['pe'][1] = 200.01
    assert sum(r['pairs'] for r in T.pair_report(h)['same_bar_same_end_distinct_sipm_equal_pe_bin']) == 5


def test_full_group_centering_anticorrelation_identity():
    h = hits()
    for row in T.pair_report(h)['all']:
        if row['pairs']:
            n = { (4, 0): 4, (4, 1): 5, (5, 0): 3 }[(row['body'], row['bin'])]
            assert row['correlation'] == pytest.approx(-1/(n-1))


def test_permutation_invariance():
    h = hits()
    expected = T.pair_report(h)
    got = T.pair_report({k: v[::-1] for k, v in h.items()})
    assert got == expected


def test_cells_exact_multiplicity_and_decomposition():
    h = hits()
    got = T.cells(h, 'exact')
    assert {c['key'][-1] for c in got['cells']} == {3, 4, 5}
    for cell in got['cells']:
        assert cell['second_moment'] == pytest.approx(cell['raw_second_moment']+
                  cell['propagation_second_moment']-2*cell['raw_propagation'])
    h['pe'][0] = 201
    assert T.cells(h, 'band')['coordinate_excluded'] == 1


def test_support_requires_distinct_groups_and_preserves_coverage():
    cells = T.cells(hits(), 'exact')
    for c in cells['cells']:
        c.update(count=60, groups=10)
    samples = {name: copy.deepcopy(cells) for name in (*J.ARMS, 'data')}
    samples['data']['cells'][0]['count'] = 120
    samples['joint']['cells'][0]['groups'] = 9
    result = T.standardize(samples, 4)
    assert result['common_cells'] == 1
    assert result['samples']['data']['coverage'] == pytest.approx(1/3)
    assert result['data_weights'] == [1.]
    samples['joint']['cells'][1]['groups'] = 9
    result = T.standardize(samples, 4)
    assert result['common_cells'] == 0
    assert result['samples']['nominal']['data_weighted']['second_moment'] is None


def test_empty_population_json_and_overwrite_guard(tmp_path):
    h = {k: v[:0] for k, v in hits().items()}
    assert T.cells(h, 'exact')['cells'] == []
    json.dumps(T.pair_report(h), allow_nan=False)
    out = tmp_path/'exists.json'
    out.write_text('{}')
    with pytest.raises(ValueError, match='already exists'):
        T.compare('/unavailable', '/unavailable', out)


def test_saved_raw_time_relationship_is_independently_checked():
    cols = captured()
    cols['hit_time'][0] += 1
    with pytest.raises(ValueError, match='saved time'):
        T.prepare(cols, mapping())


def test_stable_pair_differences_with_large_subgroup_offset():
    h = {k: v[:3].copy() for k, v in hits().items()}
    h['time_residual'] = np.array([1e8, 1e8+1, 1e8+2])
    h['raw_centered'] = h['time_residual'].copy()
    h['propagation_centered'] = np.zeros(3)
    _, values = T._pair_sums(h, np.zeros((3, 1), int))
    assert values['difference_second'][0] == pytest.approx(6.)


def test_map_accepts_unused_paired_sentinels_but_not_selected_unmapped_channels(tmp_path):
    table = np.zeros((2304, 9))
    table[:, [0, 6, 7]] = mapping()
    table[2303, 6:8] = -1
    path = tmp_path/'map.csv'
    np.savetxt(path, table, delimiter=',')
    m = T.channel_map(path, {}, T.V.digest(path))
    T.prepare(captured(), m)
    m[0, 1:] = -1
    with pytest.raises(ValueError, match='unavailable ROC/FEB'):
        T.prepare(captured(), m)


def test_saved_report_summary_checks_cancellation_and_data_identity():
    from crvfastsim.objective import joint_timing_report as P
    analyzed = T.analyze(captured(), mapping())
    report = dict(comparison='joint-timing', passed=True, standardization={}, samples={'data': analyzed})
    reports = {'A': copy.deepcopy(report), 'B': copy.deepcopy(report)}
    result = P.summarize(reports)
    assert result['checks']['identical_measured_data_diagnostics']
    reports['B']['samples']['data']['flow']['retained_members'] += 1
    with pytest.raises(ValueError, match='data diagnostics differ'):
        P.summarize(reports)


def test_saved_report_summary_rejects_false_propagation_cancellation():
    from crvfastsim.objective import joint_timing_report as P
    analyzed = T.analyze(captured(), mapping())
    analyzed['pairs']['same_bar_same_end_distinct_sipm'][0].update(pairs=1,
        propagation_difference_second=1., difference_second=3., raw_difference_second=2.)
    report = dict(comparison='joint-timing', passed=True, standardization={}, samples={'data': analyzed})
    with pytest.raises(ValueError, match='decomposition'):
        P.summarize({'A': report, 'B': copy.deepcopy(report)})
