"""Denominator counts use species/energy/event census, not generator time."""
import numpy as np
import pandas as pd

from crvfastsim import tier1b as T
from valhist.shower_uncertainty import bin_frame


def frame():
    return pd.DataFrame(dict(run=[1400]*4, subrun=[2001]*4, event=[1, 2, 3, 3],
                             idx=[0, 0, 0, 1], pdg=[22, 11, -13, 2112],
                             e=[100., 10., 1000., 1000.], t=[0., 1., 2., 2.]))


def test_native_reach_ignores_time_and_uses_full_census(monkeypatch):
    original = frame()
    changed = original.copy()
    changed['t'] = [1e12, -1e12, 77., 77.]
    current = [original]
    requested = []
    def read(paths, tree, columns):
        requested.append(columns)
        return current[0][columns].copy()
    monkeypatch.setattr(T.tables, 'read_many', read)
    census = pd.DataFrame(dict(run=[1400]*4, subrun=[2001]*4, event=[1, 2, 3, 4]))
    monkeypatch.setattr(T.genassoc, 'generator_event_keys', lambda path: census)
    showers = pd.DataFrame(dict(apdg=[22], kebin=T._ke_bin(np.array([100.]))))
    a, na = T._reach(showers, ['explicit'], 8)
    current[0] = changed
    b, nb = T._reach(showers, ['explicit'], 8)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert na == nb == 3 and a.ngen.tolist() == [2, 2, 2]
    assert all('t' not in columns for columns in requested)


def test_uncertainty_cell_counts_invariant_to_time_permutation():
    g = frame()
    a = bin_frame(g).groupby(['apdg', 'kebin']).size()
    g['t'] = g.t.to_numpy()[::-1] + 1e6
    b = bin_frame(g).groupby(['apdg', 'kebin']).size()
    pd.testing.assert_series_equal(a, b, check_exact=True)
