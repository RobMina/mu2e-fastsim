"""Synthetic complete ROOT event censuses; dense oracle for sparse influence algebra."""
import hashlib
import json

import awkward as ak
import numpy as np
import pytest
from scipy import sparse
import uproot

from valhist.event_shapes import main, run, shape, validate_edges


def dump(path, events=(1, 2, 3, 4), cluster_events=(2, 2, 3),
         pe=(0.2, 1.2, 0.8), *, subrun=1, update=False, idx=None, counts=None):
    events = np.asarray(events, dtype=np.int32)
    ce = np.asarray(cluster_events, dtype=np.int32)
    if counts is None:
        counts = np.searchsorted(np.sort(ce), events, side='right') - np.searchsorted(np.sort(ce), events)
    if idx is None:
        order = np.argsort(ce, kind='stable')
        first = np.searchsorted(ce[order], ce[order])
        idx = np.empty(len(ce), np.int32)
        idx[order] = np.arange(len(ce)) - first
    with (uproot.update(path) if update else uproot.recreate(path)) as root:
        ev = root.mktree('crvdump/events', {k: 'int32' for k in ('run', 'subrun', 'event', 'ncluster')})
        ev.extend(dict(run=np.full(len(events), 1400, np.int32),
                       subrun=np.full(len(events), subrun, np.int32), event=events,
                       ncluster=np.asarray(counts, np.int32)))
        cl = root.mktree('crvdump/clusters', dict(run='int32', subrun='int32', event='int32',
                                                idx='int32', pe='float64', start='float64',
                                                end='float64', x='float64', pulses='var * int32'))
        cl.extend(dict(run=np.full(len(ce), 1400, np.int32), subrun=np.full(len(ce), subrun, np.int32),
                       event=ce, idx=np.asarray(idx, np.int32), pe=np.asarray(pe, float),
                       start=np.asarray(pe, float), end=np.asarray(pe, float) + 0.25,
                       x=np.asarray(pe, float),
                       pulses=ak.unflatten(np.arange(len(ce) * 2, dtype=np.int32), np.full(len(ce), 2))))


def compare(tmp_path, **kwargs):
    return run([dict(ref=tmp_path / 'a.root', cand=tmp_path / 'b.root')],
               {'pe': [0, 1, 2]}, tmp_path / 'out.json', replicas=31, **kwargs)


def test_sparse_covariance_matches_dense_paired_influence_and_rng():
    a = np.array([[0, 0, 0], [2, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0.]])
    b = np.array([[0, 1, 0], [0, 1, 2], [0, 0, 0], [1, 0, 0], [0, 0, 0.]])
    n = len(a)
    p, q = a.sum(0) / a.sum(), b.sum(0) / b.sum()
    u = ((b - b.sum(1)[:, None] * q) / b.sum()
         - (a - a.sum(1)[:, None] * p) / a.sum())
    centered = u - u.mean(0)
    expected = n / (n - 1) * centered.T @ centered
    got = shape(sparse.csr_matrix(a), sparse.csr_matrix(b), replicas=37, seed=17)
    np.testing.assert_allclose(got['covariance'], expected, atol=1e-15)
    np.testing.assert_allclose(np.sum(got['covariance'], axis=1), 0, atol=1e-15)
    assert np.linalg.eigvalsh(got['covariance']).min() > -1e-14
    weights = np.random.default_rng(17).standard_normal((37, n))
    weights -= weights.mean(1, keepdims=True)
    fluctuations = np.sqrt(n / (n - 1)) * weights @ u
    expected_null = np.quantile(np.abs(fluctuations).sum(1) / 2, got['quantile_levels'])
    np.testing.assert_allclose(got['conditional_null_tv_quantiles'], expected_null, atol=1e-15)
    expected_uncertain = np.quantile(np.abs(fluctuations + q - p).sum(1) / 2, got['quantile_levels'])
    np.testing.assert_allclose(got['linearized_tv_uncertainty_quantiles'], expected_uncertain, atol=1e-15)
    # Shared-event covariance differs from treating two arms as independent.
    ua = (a - a.sum(1)[:, None] * p) / a.sum()
    ub = (b - b.sum(1)[:, None] * q) / b.sum()
    independent = n / (n - 1) * (ua.T @ ua + ub.T @ ub)
    assert not np.allclose(expected, independent)


def test_equal_aggregate_shapes_can_have_nonzero_paired_null():
    a = np.array([[1, 0], [0, 1], [0, 0.]])
    b = a[[1, 0, 2]]
    got = shape(a, b, replicas=21)
    assert got['tv'] == 0
    assert np.trace(got['covariance']) > 0
    assert got['conditional_null_tv_quantiles'][-1] > 0
    same = shape(a, a, replicas=21)
    np.testing.assert_allclose(same['covariance'], 0, atol=1e-15)
    np.testing.assert_allclose(same['conditional_null_tv_quantiles'], 0, atol=1e-15)
    scaled = shape(a, 2 * a, replicas=21)
    assert scaled['tv'] == 0
    np.testing.assert_allclose(scaled['covariance'], 0, atol=1e-15)


def test_pairing_zero_events_multi_cluster_flow_and_input_immutability(tmp_path):
    a, b = tmp_path / 'a.root', tmp_path / 'b.root'
    dump(a, pe=(-1, 0, 2))
    dump(b, events=(3, 1, 4, 2), cluster_events=(3, 2, 2), pe=(2, -1, 0))
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (a, b)]
    got = compare(tmp_path)
    assert got['events'] == 4
    obs = got['observables']['pe']
    assert obs['inclusive']['counts_ref'] == [1, 1, 0, 1]
    assert obs['visible']['counts_ref'] == [1, 0]
    assert obs['flow_rate']['total_ref'] == 2
    assert obs['inclusive']['tv'] == 0
    assert got['total_cluster_rate']['mean_ref'] == 0.75
    assert got['total_cluster_rate']['total_difference_se'] == 0
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (a, b)]
    assert json.loads((tmp_path / 'out.json').read_text())['events'] == 4
    with pytest.raises(FileExistsError):
        compare(tmp_path)


def test_rate_only_scaling_and_event_matched_shape_uncertainty(tmp_path):
    dump(tmp_path / 'a.root')
    dump(tmp_path / 'b.root', events=(4, 3, 2, 1), cluster_events=(2, 2, 3, 2, 2, 3),
         pe=(0.2, 1.2, 0.8, 0.2, 1.2, 0.8))
    got = compare(tmp_path)
    assert got['total_cluster_rate']['relative_difference'] == 1
    assert got['total_cluster_rate']['relative_se'] == 0
    assert got['observables']['pe']['inclusive']['tv'] == 0
    np.testing.assert_allclose(got['observables']['pe']['inclusive']['covariance'], 0, atol=1e-15)


def test_root_pairing_covariance_not_positional_or_aggregate(tmp_path):
    dump(tmp_path / 'a.root', cluster_events=(1, 2), pe=(0.2, 1.2))
    dump(tmp_path / 'b.root', events=(2, 4, 3, 1), cluster_events=(2, 1), pe=(0.2, 1.2))
    got = compare(tmp_path)['observables']['pe']['visible']
    expected = shape([[1, 0], [0, 1], [0, 0], [0, 0]],
                     [[0, 1], [1, 0], [0, 0], [0, 0]], replicas=31)
    assert got['tv'] == 0
    np.testing.assert_allclose(got['covariance'], expected['covariance'])
    assert np.trace(got['covariance']) > 0


def test_flow_shape_and_visible_shape_are_separate(tmp_path):
    dump(tmp_path / 'a.root', pe=(-1, 0.2, 2))
    dump(tmp_path / 'b.root', pe=(0.2, 0.2, 0.2))
    obs = compare(tmp_path)['observables']['pe']
    assert obs['inclusive']['tv'] > 0
    assert obs['visible']['tv'] == 0
    assert obs['visible_rate']['total_difference'] == 2
    assert obs['flow_rate']['total_difference'] == -2


def test_empty_regions_and_empty_arm_explicit(tmp_path):
    dump(tmp_path / 'a.root', pe=(-2, -1, 2))
    dump(tmp_path / 'b.root', cluster_events=(), pe=())
    got = compare(tmp_path)
    obs = got['observables']['pe']
    assert obs['inclusive']['status'] == 'undefined_empty_regional_total'
    assert obs['inclusive']['covariance'] is None
    assert obs['visible']['empty_ref'] and obs['visible']['empty_cand']
    assert got['total_cluster_rate']['total_cand'] == 0


def test_two_all_zero_arms_keep_every_generated_event(tmp_path):
    dump(tmp_path / 'a.root', cluster_events=(), pe=())
    dump(tmp_path / 'b.root', cluster_events=(), pe=(), events=(4, 2, 1, 3))
    got = compare(tmp_path)
    assert got['events'] == 4
    assert got['total_cluster_rate']['relative_difference'] is None
    assert got['total_cluster_rate']['total_difference_se'] == 0
    obs = got['observables']['pe']['inclusive']
    assert obs['empty_ref'] and obs['empty_cand']
    assert obs['tv'] is None and obs['covariance'] is None


@pytest.mark.parametrize('events,subrun', [((1, 2, 3), 1), ((1, 2, 3, 5), 1),
                                         ((1, 2, 3, 3), 1), ((1, 2, 3, 4), 2)])
def test_event_coverage_and_duplicates_fail(tmp_path, events, subrun):
    dump(tmp_path / 'a.root')
    dump(tmp_path / 'b.root', events=events, subrun=subrun)
    with pytest.raises(ValueError):
        compare(tmp_path)
    assert not (tmp_path / 'out.json').exists()


@pytest.mark.parametrize('extra', [dict(cluster_events=(2, 2, 9)), dict(counts=(0, 1, 1, 0)),
                                  dict(idx=(0, 0, 0)), dict(idx=(0, 2, 0))])
def test_cluster_census_or_indices_fail(tmp_path, extra):
    dump(tmp_path / 'a.root')
    dump(tmp_path / 'b.root', **extra)
    with pytest.raises(ValueError):
        compare(tmp_path)


def test_highest_cycles_are_used_for_both_trees(tmp_path):
    a, b = tmp_path / 'a.root', tmp_path / 'b.root'
    dump(a, events=(8, 9), cluster_events=(9,), pe=(100,))
    dump(a, update=True)
    dump(b)
    got = compare(tmp_path)
    assert got['pairs'][0]['ref']['events']['key'] == 'crvdump/events;2'
    assert got['pairs'][0]['ref']['clusters']['key'] == 'crvdump/clusters;2'
    assert got['observables']['pe']['inclusive']['tv'] == 0
    (tmp_path / 'out.json').unlink()
    with uproot.update(a) as f:
        f['crvdump/events'] = 'not a tree'
    with pytest.raises(ValueError, match='highest cycle'):
        compare(tmp_path)


def test_actual_latest_census_mismatch_not_older_matching_cycle(tmp_path):
    dump(tmp_path / 'a.root')
    dump(tmp_path / 'b.root')
    dump(tmp_path / 'b.root', events=(1, 2, 3), update=True)
    with pytest.raises(ValueError, match='full event keys'):
        compare(tmp_path)


def test_multishard_and_reused_events_fail(tmp_path):
    pairs = []
    for shard in (1, 2):
        a, b = tmp_path / f'a{shard}.root', tmp_path / f'b{shard}.root'
        dump(a, subrun=shard)
        dump(b, subrun=shard, events=(4, 3, 2, 1))
        pairs.append(dict(ref=str(a), cand=str(b)))
    got = run(pairs, {'pe': [0, 1, 2]}, tmp_path / 'ok.json', replicas=11)
    assert got['events'] == 8
    assert got['total_cluster_rate']['total_ref'] == 6
    for bad in (pairs + pairs[:1], [pairs[0], dict(ref=pairs[0]['cand'], cand=pairs[0]['ref'])]):
        with pytest.raises(ValueError, match='duplicate/reused'):
            run(bad, {'pe': [0, 1, 2]}, tmp_path / 'bad.json', replicas=11)
    # Identical event IDs in physically copied, distinct files are also refused.
    copy = tmp_path / 'copy.root'
    copy.write_bytes((tmp_path / 'a1.root').read_bytes())
    with pytest.raises(ValueError, match='duplicate/reused'):
        run([pairs[0], dict(ref=copy, cand=copy)], {'pe': [0, 1]}, tmp_path / 'bad.json')


@pytest.mark.parametrize('features', [{}, {'unknown': [0, 1]}, {'pe': [1, 1]}, {'pe': [1, 0]},
                                     {'pe': [0, float('nan')]}, {'pe': [[0, 1]]},
                                     {'pe': list(range(259))}])
def test_invalid_edges(features):
    with pytest.raises(ValueError):
        validate_edges(features)


def test_nonfinite_observable_and_limits(tmp_path):
    dump(tmp_path / 'a.root')
    dump(tmp_path / 'b.root', pe=(0, np.nan, 1))
    with pytest.raises(ValueError, match='nonfinite'):
        compare(tmp_path)
    with pytest.raises(ValueError, match='memory limits'):
        compare(tmp_path, max_events=3)
    with pytest.raises(ValueError, match='memory limits'):
        compare(tmp_path, max_cluster_rows=2)
    with pytest.raises(ValueError):
        shape(np.ones((1, 2)), np.ones((1, 2)))


def test_derived_observables_and_cli_pair_list(tmp_path, capsys):
    dump(tmp_path / 'a.root')
    dump(tmp_path / 'b.root')
    pairs = tmp_path / 'pairs.json'
    pairs.write_text(json.dumps([dict(ref=str(tmp_path / 'a.root'), cand=str(tmp_path / 'b.root'))]))
    edges = tmp_path / 'edges.json'
    edges.write_text(json.dumps({'npulses': [0, 1, 2, 3], 'duration': [0, 0.5], 'x': [0, 1, 2]}))
    out = tmp_path / 'out.json'
    main(['--pairs', str(pairs), '--edges', str(edges), '--out', str(out), '--replicas', '11'])
    got = json.loads(out.read_text())
    assert got['observables']['npulses']['inclusive']['counts_ref'] == [0, 0, 0, 3, 0]
    assert got['observables']['duration']['visible']['counts_ref'] == [3]
    assert 'conditional diagnostics only' in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / 'a.root'), '--pairs', str(pairs), '--edges', str(edges), '--out', str(out)])
    assert exc.value.code == 2
    main([str(tmp_path / 'a.root'), str(tmp_path / 'b.root'), '--edges', str(edges),
          '--out', str(tmp_path / 'two.json'), '--replicas', '11'])


def test_no_dense_event_by_bin_influence(monkeypatch):
    n, bins = 10000, 80
    a = sparse.csr_matrix((np.ones(n), (np.arange(n), np.arange(n) % bins)), shape=(n, bins))
    b = sparse.csr_matrix((np.ones(n), (np.arange(n), (np.arange(n) + 1) % bins)), shape=(n, bins))
    original = sparse.csc_matrix.toarray
    original_csr = sparse.csr_matrix.toarray

    def checked_csc(self, *args, **kwargs):
        assert self.shape[0] <= bins and self.shape[1] <= bins
        return original(self, *args, **kwargs)

    def checked_csr(self, *args, **kwargs):
        assert self.shape[0] <= bins and self.shape[1] <= bins
        return original_csr(self, *args, **kwargs)

    monkeypatch.setattr(sparse.csc_matrix, 'toarray', checked_csc)
    monkeypatch.setattr(sparse.csr_matrix, 'toarray', checked_csr)
    got = shape(a, b, replicas=7)
    assert len(got['covariance']) == bins
