"""device.response_batched: memory-sized batches of whole events (photon_budget, batch_bounds)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _dv():
    pytest.importorskip("jax")
    from crvfastsim.response import device
    return device


def test_bounds_whole_events_and_budget():
    dv = _dv()
    ev = np.repeat(np.arange(10), [3, 1, 4, 2, 2, 5, 1, 1, 3, 2])
    est = np.ones(len(ev)) * 10.0                 # 10 photons per step
    b = dv.batch_bounds(ev, est, budget=60, batch_steps=10000)
    assert b[0][0] == 0 and b[-1][1] == len(ev)
    assert all(b1 == a2 for (_, b1), (a2, _) in zip(b[:-1], b[1:]))           # contiguous, no gap
    starts = set(np.flatnonzero(np.r_[True, ev[1:] != ev[:-1]]).tolist()) | {len(ev)}
    assert all(a in starts and c in starts for a, c in b)                       # cuts only at event boundaries
    for a, c in b:
        assert est[a:c].sum() <= 60 or len(np.unique(ev[a:c])) == 1             # within budget, or one big event


def test_bounds_big_event_alone_and_step_fallback():
    dv = _dv()
    ev = np.repeat(np.arange(4), [2, 50, 2, 2])
    est = np.ones(len(ev))
    b = dv.batch_bounds(ev, est, budget=10, batch_steps=10000)
    assert (2, 52) in b                            # the 50-step event exceeds the budget: a batch of its own
    assert dv.batch_bounds(ev, est, budget=None, batch_steps=10000) == [(0, len(ev))]
    b = dv.batch_bounds(ev, est, budget=1e9, batch_steps=4)   # step limit binds when the budget does not
    assert b == [(0, 2), (2, 52), (52, 56)]


class _Dev:
    def __init__(self, limit, used=0):
        self.ms = {"bytes_limit": limit, "bytes_in_use": used}

    def memory_stats(self):
        return self.ms


def test_photon_budget_power_of_two_and_reserve():
    dv = _dv()
    b10 = dv.photon_budget(_Dev(7_650_410_496))     # EAF 1g.10gb: JAX bytes_limit
    b40 = dv.photon_budget(_Dev(39_000_000_000))
    p10 = round(b10 / dv.BUDGET_MARGIN)
    assert p10 & (p10 - 1) == 0                     # 85 % of a power of two: NB stays at that power of two
    assert p10 * dv.SIPM_BYTES_PER_SLOT <= 0.6 * 7_650_410_496 and b40 > b10
    assert dv.photon_budget(_Dev(7_650_410_496), reserve=4_000_000_000) < b10
    assert dv.photon_budget(_Dev(1)) == int(dv.BUDGET_MARGIN * dv.MIN_PHOTONS)
    assert dv.photon_budget(type("Cpu", (), {"memory_stats": lambda self: None})()) is None
