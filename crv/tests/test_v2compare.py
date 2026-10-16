"""Unit tests: the V2 summary passes identical arms, fails a 5 % PE shift and a 3 % cluster excess, merges additively."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim.analyses import v2compare as V                         # noqa: E402


def _arm(A, n_ev, pe_scale, clus_scale, seed):
    rng = np.random.default_rng(seed)
    A["events"] = n_ev
    A["bar_rate"] = rng.poisson(n_ev * 0.05, V.NBARS).tolist()
    for g in V.GROUPS:
        G = A["groups"][g]
        pe = rng.gamma(6.0, 7.0 * pe_scale, 200_000)
        G["npulse"] = len(pe)
        G["h_pe"] = np.histogram(pe, bins=V.PE_EDGES)[0].tolist()
        G["nclus"] = int(rng.poisson(0.8 * n_ev * clus_scale))
        G["eff_den"] = [50_000] * V.NLAYERS
        G["eff_num"] = rng.binomial(50_000, 0.9, V.NLAYERS).tolist()


def _state(pe_scale=1.0, clus_scale=1.0):
    S = V._new_state()
    _arm(S["ref"], 1_000_000, 1.0, 1.0, 1)
    _arm(S["fast"], 1_000_000, pe_scale, clus_scale, 2)
    return S


def test_identical_arms_pass():
    assert V.summarise(_state())["pass"]


def test_pe_shift_fails():
    r = V.summarise(_state(pe_scale=1.05))["groups"]["EX"]
    assert not r["checks"]["pe_median"]


def test_cluster_excess_fails():
    r = V.summarise(_state(clus_scale=1.03))["groups"]["T1"]
    assert not r["checks"]["clusters"] and r["checks"]["pe_median"]


def test_hist_median_matches_numpy():
    x = np.random.default_rng(3).gamma(6.0, 7.0, 100_000)
    m, _ = V.hist_median(np.histogram(x, bins=V.PE_EDGES)[0], V.PE_EDGES)
    assert abs(m - np.median(x)) < 0.05


def test_merge_is_additive():
    a = _state()
    tot = V._new_state()
    V._merge_into(tot, a)
    V._merge_into(tot, a)
    assert tot["ref"]["groups"]["EX"]["nclus"] == 2 * a["ref"]["groups"]["EX"]["nclus"]
    assert tot["fast"]["events"] == 2 * a["fast"]["events"]
