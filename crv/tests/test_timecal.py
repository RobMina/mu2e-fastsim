"""objective.timecal: the data - MC pair-median fit recovers injected module offsets; shift_times moves pulses and
clusters by their body's correction."""
import awkward as ak
import numpy as np

from crvfastsim.objective import crvresponse as R, timecal as TC


def _state(offsets, geom, rng, n=4000):
    S = R.new_state()
    dtp = np.zeros((4, 16, len(R.DTP_EDGES) - 1))
    for e in range(4):
        for m in (4, 5, 6, 7, 8):
            x = geom[e, m] + offsets[m] - offsets[e] + rng.normal(0, 1.5, n)
            dtp[e, m] = np.histogram(x, bins=R.DTP_EDGES)[0]
    S["dtp"] = dtp.tolist()
    return S


def test_fit_difference_recovers_offsets():
    rng = np.random.default_rng(3)
    geom = rng.uniform(-3, 10, (4, 16))                    # time of flight / light path per pair, same in both arms
    true = np.zeros(16)
    true[[1, 2, 3, 4, 5, 6, 7, 8]] = [1.0, 2.5, -1.5, 9.0, 8.5, 0.8, 1.8, 9.3]
    D = _state(true, geom, rng)
    M = _state(np.zeros(16), geom, rng)
    f = TC.fit_difference(D, M, group_mc_taggers=False)
    off = np.asarray(f["offset"])
    for b in (1, 2, 3, 4, 5, 6, 7, 8):
        assert abs(off[b] - true[b]) < 0.15, (b, off[b], true[b])


def test_shift_times():
    p = ak.Array({"crvpulses.barId": [[0, 0, 300], [512]], "crvpulses.time": [[10.0, 11.0, 20.0], [5.0]],
                  "crvpulses.crvHitIndex": [[0, 0, 1], [-1]], "crvpulses.PEs": [[30.0, 20.0, 40.0], [9.0]]})
    c = ak.Array({"crvcoincs.time": [[10.5, 20.0], []]})
    corr = np.zeros(16)
    corr[0], corr[4], corr[8] = 1.0, -2.0, 0.5
    c2, p2 = R.shift_times(c, p, corr)
    assert p2["crvpulses.time"].tolist() == [[11.0, 12.0, 18.0], [5.5]]
    assert c2["crvcoincs.time"].tolist() == [[11.5, 18.0], []]


def test_channel_table():
    mod = np.zeros(16)
    mod[6] = 2.0                                           # T20
    feb = np.zeros((16, 4))
    feb[6, 3] = 0.5                                        # T20, layers 2-3, side 1
    t = TC.channel_table(mod, feb)
    ch = lambda bar, sipm: 4 * bar + sipm
    assert t[ch(384, 0)] == 2.0                            # T20 bar 384: layer 0, SiPM 0 (side 0)
    assert t[ch(416, 1)] == 2.5                            # T20 bar 416: layer 2, SiPM 1 (side 1) -> + FEB group 3
    assert t[ch(416, 2)] == 2.0                            # SiPM 2 is side 0
    assert t[ch(0, 0)] == 0.0
