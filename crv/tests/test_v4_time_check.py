"""V4 time criterion: CDF shift with a cluster-robust paired error (v4photons.cdf_shift / time_check)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim.analyses.v4photons import cdf_shift, time_check  # noqa: E402


def _sample(rng, n_ev, offset=0.0):
    """Whole-ns delay staircase like the lookup tables; photons of one event share a per-event shift (correlated)."""
    per = rng.poisson(40, n_ev) + 1
    ev = np.repeat(np.arange(n_ev), per)
    t = np.floor(rng.gamma(3.0, 2.5, len(ev))) + rng.uniform(0, 0.07, len(ev)) + rng.normal(0, 0.3, n_ev)[ev] + offset
    return t, ev


def test_same_law_pulls_are_unit_and_pass():
    rng = np.random.default_rng(1)
    pulls, fails = [], 0
    for _ in range(60):
        tf, ef = _sample(rng, 400)
        ts, es = _sample(rng, 400)
        s, e = cdf_shift(tf, ts, ef, es)
        pulls += list(s / e)
        fails += not time_check(s, e)
    pulls = np.array(pulls)
    assert 0.75 < pulls.std() < 1.3 and abs(pulls.mean()) < 0.3
    assert fails <= 3                                   # same law: the 2-sigma allowance passes nearly always


def test_real_offset_is_caught():
    rng = np.random.default_rng(2)
    tf, ef = _sample(rng, 3000, offset=0.4)
    ts, es = _sample(rng, 3000)
    s, e = cdf_shift(tf, ts, ef, es)
    assert np.all(e < 0.05) and not time_check(s, e)    # 0.4 ns with small errors fails the 0.2 ns limit
    tf, ef = _sample(rng, 3000, offset=0.1)
    s, e = cdf_shift(tf, ts, ef, es)
    assert time_check(s, e)                             # 0.1 ns is within it


def test_clusters_widen_the_error():
    rng = np.random.default_rng(3)
    tf, ef = _sample(rng, 300)
    ts, es = _sample(rng, 300)
    _, e_cl = cdf_shift(tf, ts, ef, es)
    _, e_bin = cdf_shift(tf, ts)
    assert np.all(e_cl > e_bin)                         # correlated photons: binomial errors would be too small
