"""Unit tests for the chain glue: TDC0 per (event, FEB), digis -> reco waveforms grouping."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim import chain                                            # noqa: E402


def _cond(n=16):
    z = np.zeros(n)
    feb = np.repeat(np.arange(n // 4), 4)        # 4 channels per FEB
    return chain.Conditions(z + 100, z + 400, z + 11, z, np.zeros(n, np.int64), np.ones(n, np.int64), feb, z)


def test_tdc0_one_draw_per_event_and_feb():
    cond = _cond()
    lab = np.array([1 * chain.CHAN_MULT + 0, 1 * chain.CHAN_MULT + 3, 1 * chain.CHAN_MULT + 4, 2 * chain.CHAN_MULT + 0])
    t = chain.tdc0_per_feb(lab, cond, np.random.default_rng(1))
    assert t[0] == t[1] and t[0] != t[2] and t[0] != t[3]
    assert np.all((t >= 0) & (t < chain.PERIOD))


def test_reco_waveforms_concatenates_consecutive_digis():
    cond = _cond()
    lab = np.array([5 * chain.CHAN_MULT + 2] * 3)
    D = {"chan": lab, "starttdc": np.array([10, 22, 40]), "adc": np.arange(36).reshape(3, 12) + 100}
    W = chain.reco_waveforms(D, cond)
    assert W["n"].tolist() == [24, 12] and W["start"].tolist() == [10, 40] and W["ev"].tolist() == [5, 5]
    assert W["bar"].tolist() == [0, 0] and W["sipm"].tolist() == [2, 2] and np.all(W["ped"] == 100)
