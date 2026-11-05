"""A finite zero-miss control gives an upper limit, never a logical guarantee."""
import numpy as np
import pytest

from beamfastsim.target_filter_probe import missed_proxy


def test_zero_miss_bound_and_counterexample():
    a = missed_proxy(np.ones(100, bool), np.ones(100, bool))
    assert a['missed_by_filter'] == 0
    assert a['miss_fraction_upper_95'] == pytest.approx(1 - .05 ** .01)
    b = missed_proxy(np.array([1, 1, 0]), np.array([1, 0, 0]))
    assert b['missed_by_filter'] == 1 and b['miss_fraction'] == .5
    assert missed_proxy(np.zeros(3, bool), np.zeros(3, bool))['miss_fraction_upper_95'] is None


def test_separate_passed_tree_matches_exact_primary_identity(tmp_path, monkeypatch):
    uproot = pytest.importorskip('uproot')
    from beamfastsim import target_filter_probe as P
    inclusive, passed = tmp_path / 'all.root', tmp_path / 'passed.root'
    a = dict(run=np.ones(3, np.int32), subrun=np.zeros(3, np.int32), evt=np.arange(1, 4, dtype=np.int32),
             trk=np.ones(3, np.int32), code=np.full(3, 167, np.int32), pdg=np.full(3, 11, np.int32))
    for k in ('isvd', 'xvd', 'yvd', 'zvd', 'pxvd', 'pyvd', 'pzvd'):
        a[k] = np.zeros((3, 13), np.float32)
    a['isvd'][:, 12] = 1
    a['pzvd'][:, 12] = 100
    a['xvd'][:, 12] = [460, 510, 560]
    with uproot.recreate(inclusive) as root:
        root['all/ntpart'] = a
    with uproot.recreate(passed) as root:
        root['passed/ntpart'] = {k: v[[0, 2]] for k, v in a.items()}
    monkeypatch.setattr(P.C, 'helix_rmax', lambda geom, x, *args: (x[:, 0], np.zeros(x.shape[0])))
    report = P.compare(inclusive, None, lambda x: np.ones_like(x), expected_events=3, passed_path=passed)
    assert report['cuts']['450.0']['missed_by_filter'] == 1
    assert report['cuts']['550.0']['missed_by_filter'] == 0
    assert report['cuts']['450.0']['missed_identities'] == [[1, 0, 2, 1]]
    assert not report['filter_implication_certified']
