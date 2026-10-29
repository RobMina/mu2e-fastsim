"""ValCrvStep refill: ROOT bin lookup, clustered moments, paired variance and the yield statistic."""
import numpy as np
import pandas as pd
import pytest

from valhist import placement_closure as P
from valhist import stephist as H


def test_root_bin_matches_taxis_findbin():
    x = np.array([-1.0, 0.0, 0.0019999, 0.002, 0.1999999, 0.2, 5.0, np.nan])
    b = H.root_bin(x, 100, 0.0, 0.2)
    assert list(b[:7]) == [0, 1, 1, 2, 100, 101, 101]
    assert b[7] == 101                      # NaN is not < xmax: overflow, as ROOT


def test_clustered_second_moment_counts_events():
    ev = np.array([1, 1, 1, 2, 3])
    idx = np.array([2, 2, 3, 2, 3])
    n, w2 = H.clustered(ev, idx, 5)
    assert list(n) == [0, 0, 3, 2, 0]
    assert list(w2) == [0, 0, 4 + 1, 1 + 1, 0]


def test_paired_and_yield_errors():
    edges = {"y": np.linspace(4200, 4800, 7), "z": np.linspace(20000, 25000, 6)}
    ev = np.array([1, 1, 2])
    a = {"bar": (ev, np.array([1, 1, 2]), np.array(["mu"] * 3, object))}
    b = {"bar": (np.array([1, 2, 2]), np.array([1, 2, 2]), np.array(["mu"] * 3, object))}
    d2 = H.paired(a, b, {**edges})["bar"]
    assert d2[1] == 1.0 and d2[2] == 1.0    # event 1 loses one bin-1 entry, event 2 gains one bin-2 entry
    ref = {"all": (np.array([10.0]), np.array([30.0]))}
    cand = {"all": (np.array([12.0]), np.array([40.0]))}
    y = H.yield_stat(ref, cand, d2=np.array([4.0]))
    assert y["rel"] == pytest.approx(0.2)
    assert y["rel_se_independent"] == pytest.approx(np.sqrt(70.0) / 10)
    assert y["rel_se_paired"] == pytest.approx(0.2)


def test_deck_statistic_cuts_first_bin_and_flow():
    n1 = np.array([5.0, 100.0, 50.0, 50.0, 7.0])
    n2 = np.array([0.0, 0.0, 50.0, 60.0, 0.0])
    c, ndf = H.chi2(n1, n2, n1 + n2, cut=True)
    assert ndf == 2 and c == pytest.approx((0 + 100 / 110) / 2)


def test_parse_arm_rejects_unknown_option():
    name, opts = P.parse_arm("topo=context,contain,point,topology")
    assert name == "topo" and opts["topology"] and not opts["mirror"]
    assert not any(P.parse_arm("default=")[1].values())
    with pytest.raises(ValueError):
        P.parse_arm("x=nearest")
    name, opts = P.parse_arm("ccp2=context,contain,point@2")
    assert opts["seed_tag"] == 2 and opts["point"] and not opts["topology"]
