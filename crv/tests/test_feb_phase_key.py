"""FEB clock phase per event: per (ROC, FEB) or, as the stock CrvWaveformsGenerator, per FEB number across ROCs."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crvfastsim import chain, runconfig  # noqa: E402


def _cond():
    ch = np.arange(4 * 64 * 3)                       # 3 ROCs x 4 FEBs x 64 channels
    return SimpleNamespace(roc=1 + ch // 256, feb=1 + (ch // 64) % 4)


def test_feb_key_shares_the_phase_across_rocs():
    cond = _cond()
    ch = np.arange(len(cond.roc))
    labels = np.concatenate([e * chain.CHAN_MULT + ch for e in (5, 9)])
    roc_feb = chain.tdc0_per_feb(labels, cond, np.random.default_rng(1))
    feb = chain.tdc0_per_feb(labels, cond, np.random.default_rng(1), key="feb")
    for t, groups in ((roc_feb, 12), (feb, 4)):
        for e in range(2):
            assert len(np.unique(t[e * len(ch):(e + 1) * len(ch)])) == groups
    # "feb": the same FEB number in every ROC has one phase; events draw their own
    first = feb[:len(ch)]
    for f in range(1, 5):
        assert len(np.unique(first[cond.feb == f])) == 1
    assert not np.array_equal(feb[:len(ch)], feb[len(ch):])
    assert np.all((feb >= 0) & (feb < chain.PERIOD))


def test_default_draw_unchanged_and_unknown_key_refused():
    cond = _cond()
    labels = 3 * chain.CHAN_MULT + np.arange(len(cond.roc))
    a = chain.tdc0_per_feb(labels, cond, np.random.default_rng(2))
    b = chain.tdc0_per_feb(labels, cond, np.random.default_rng(2), key="roc_feb")
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="phase key"):
        chain.tdc0_per_feb(labels, cond, np.random.default_rng(2), key="roc")


def test_run_configurations():
    assert runconfig.get("run1a_v01").chain_config().feb_phase_key == "feb"
    assert runconfig.get("run1b_v01").chain_config().feb_phase_key == "feb"
    assert runconfig.get("extracted_v04").chain_config().feb_phase_key == "roc_feb"
    assert runconfig.get("run2_v10").chain_config().feb_phase_key == "roc_feb"
    assert chain.ChainConfig().feb_phase_key == "roc_feb"
    assert runconfig.get("run1a_v01").chain_config(feb_phase_key="roc_feb").feb_phase_key == "roc_feb"
