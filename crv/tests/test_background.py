"""reco.background: group finding on EventNtuple-shaped arrays, sampling rate and window, merge order."""
import awkward as ak
import numpy as np

from crvfastsim.reco import background as BG


def _arrays():
    # event 0: a cluster at 1000 ns; an in-time unclustered pulse (dropped), a 2-pulse group at 50 us on bar 3 and a
    # lone pulse on bar 300 at 50 us (other sector slot: its own group); event 1: no clusters, one pulse
    p = ak.Array({"crvpulses.time": [[1000.0, 1100.0, 50000.0, 50010.0, 50005.0], [70000.0]],
                  "crvpulses.PEs": [[40.0, 9.0, 15.0, 12.0, 20.0], [11.0]],
                  "crvpulses.barId": [[1, 2, 3, 3, 300], [400]],
                  "crvpulses.SiPMId": [[0, 0, 0, 1, 2], [3]],
                  "crvpulses.crvHitIndex": [[0, -1, -1, -1, -1], [-1]],
                  "crvpulses.sequenceIndex": [[0, 0, 0, 0, 0], [0]]})
    c = ak.Array({"crvcoincs.time": [[1000.0], []]})
    return p, c


def test_groups():
    p, c = _arrays()
    G = BG.oot_groups(p, c)
    assert G["groups_per_event"].tolist() == [2, 1]
    assert sorted(G["bar"].tolist()) == [3, 3, 300, 400]
    assert G["dt"].min() == 0.0 and abs(G["dt"].max() - 10.0) < 1e-9


def test_sample_rate_and_window():
    m = BG.BackgroundModel(rate_per_ns=2e-5, gamma_k=np.inf, start=np.array([0, 2]), count=np.array([2, 1]),
                           bar=np.array([3, 3, 300]), sipm=np.array([0, 1, 2]), pe=np.array([15.0, 12.0, 20.0]),
                           dt=np.array([0.0, 10.0, 0.0]), seq=np.zeros(3, np.int64), info={})
    rng = np.random.default_rng(1)
    ev = np.arange(20000, dtype=np.int64)
    B = m.sample(ev, (0.0, 1e5), rng)
    groups = (B["seq"] == 0) & np.isin(B["bar"], [3, 300]) & ((B["sipm"] == 0) | (B["sipm"] == 2))
    assert abs(groups.sum() / len(ev) - 2.0) < 0.05                  # 2e-5 /ns x 1e5 ns
    assert B["time"].min() >= 0.0 and B["time"].max() <= 1e5 + 10.0
    assert len(m.sample(ev, (0.0, 1e5), rng, rate_scale=0.0)["ev"]) == 0


def test_merge_sorted():
    B = {"ev": np.array([5, 1]), "bar": np.array([2, 0]), "sipm": np.array([0, 1]), "pe": np.array([10.0, 11.0]),
         "time": np.array([3.0, 4.0]), "seq": np.array([0, 0])}
    ev, idx, bar, sipm, tm, pe, seq, isbg = BG.merge_pulses(np.array([1, 1, 5]), np.array([0, 1, 0]), np.array([0, 1, 2]),
                                                           np.array([0, 0, 0]), np.array([9.0, 1.0, 2.0]),
                                                           np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0]), B)
    assert ev.tolist() == [1, 1, 1, 5, 5]
    assert idx.tolist() == [0, 1, 2, 0, 1]
    assert isbg.tolist() == [False, True, False, False, True]
