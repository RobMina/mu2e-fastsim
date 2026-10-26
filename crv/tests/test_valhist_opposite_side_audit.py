"""Synthetic attachment/filter tests independent of remote files and production reco."""
import numpy as np
import pandas as pd
import pytest

from valhist import opposite_side_audit as A
from valhist import reco_detail as R


def pulses():
    return pd.DataFrame(dict(run=[1400] * 6, subrun=[11] * 6, event=[1] * 6, idx=np.arange(6),
        bar=[1] * 6, sipm=[0, 1, 1, 3, 1, 0], side=[0, 1, 1, 1, 1, 0],
        time=[100., 104., 102., 103., 110., 100.], seq=[0] * 6, sectortype=[3] * 6,
        attach_dt=[10.] * 6, layer=[0] * 6, counter=[1] * 6, pe=[10.] * 6,
        threshold=[8.] * 6, adjacent_dt=[10.] * 6))


def links():
    return pd.DataFrame(dict(run=[1400], subrun=[11], event=[1], idx=[0], target=[0]))


def test_nearest_candidate_per_sipm_and_occupied_side():
    p = pulses()
    got = A.attach(p, p, links(), None)
    assert sorted(got.target) == [2, 3]
    assert sorted(got.dt) == [2., 3.]


def test_strict_time_boundary():
    p = pulses().iloc[[0, 4]].copy()
    assert A.attach(p, p, links(), None).empty


def test_no_same_side_addition():
    p = pulses().iloc[[0, 5]].copy()
    assert A.attach(p, p, links(), None).empty


def test_missing_side_not_filled_if_already_occupied():
    p = pulses()
    l = pd.concat([links(), links().assign(target=1)])
    assert A.attach(p, p, l, None).empty


def test_no_reuse_from_other_cluster():
    p = pulses()
    l = pd.concat([links(), links().assign(idx=1, target=2)])
    got = A.attach(p, p, l, None)
    assert 2 not in got.target.to_numpy()


def test_collection_order_tie_break():
    p = pulses().iloc[[0, 1, 2]].copy()
    p.loc[p.idx == 2, 'time'] = 104.
    assert A.attach(p, p, links(), None).target.tolist() == [1]


def test_event_keys_do_not_cross():
    p = pulses().iloc[[0, 1]].copy()
    p.loc[p.idx == 1, 'run'] = 1401
    assert A.attach(p, p, links(), None).empty


def test_filter_inclusive_threshold_and_time():
    p = pulses().iloc[[0, 5]].copy()
    p['pe'] = 4.
    p.loc[p.idx == 5, 'time'] = 110.
    assert len(A.filter_hits(p)) == 2
    p.loc[p.idx == 5, 'time'] = np.nextafter(110., np.inf)
    assert A.filter_hits(p).empty


def test_adjacent_not_both_adjacent_sum():
    p = pulses().iloc[[0, 1, 2]].copy()
    p['side'] = 0
    p['time'] = 100.
    p['counter'] = [0, 1, 2]
    p['pe'] = [4., 1., 4.]
    assert A.filter_hits(p).empty


def test_exclusive_output(tmp_path):
    out = tmp_path / 'exists.json'
    out.write_text('old')
    with pytest.raises(FileExistsError):
        A.create('missing', out, 'missing', 'missing')
    assert out.read_text() == 'old'


def test_secondary_member_still_occupies_side():
    p = pulses()
    p.loc[p.idx == 1, 'seq'] = 1
    l = pd.concat([links(), links().assign(target=1)])
    assert A.attach(p, p, l, None).empty


def test_reordered_filtered_candidates_keep_first_tie():
    p = pulses().iloc[[0, 2, 1]].copy()
    p.loc[p.idx == 2, 'time'] = 104.
    assert A.attach(p, p, links(), None).target.tolist() == [2]


def test_competing_cluster_tie_prefers_first_cluster():
    p = pulses().iloc[[0, 1, 5]].copy()
    l = pd.concat([links(), links().assign(idx=1, target=5)])
    got = A.attach(p, p, l, None)
    assert got.target.tolist() == [1]
    assert got.idx.tolist() == [0]
