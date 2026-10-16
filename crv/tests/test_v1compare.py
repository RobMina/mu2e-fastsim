"""Unit tests: the V1 summary passes identical arms and fails a 5 % visible-energy shift (synthetic histograms)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim.analyses import v1compare as V                         # noqa: E402


def _xing(n, scale, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"run": 1400, "subrun": 1, "event": np.arange(n), "bar": rng.integers(0, 256, n),
                         "edep": rng.gumbel(3.3 * scale, 0.5, n), "len": rng.normal(21.0, 1.0, n),
                         "t0": rng.normal(0.0, 1.0, n), "nsteps": 1, "sector": 0})


def _state(scale_fast):
    S = V._new_state()
    V._fill_arm(S["ref"], _xing(200_000, 1.0, 1), 200_000)
    V._fill_arm(S["fast"], _xing(200_000, scale_fast, 2), 200_000)
    return S


def test_identical_arms_pass():
    assert V.summarise(_state(1.0))["groups"]["EX"]["pass"]


def test_edep_shift_fails():
    r = V.summarise(_state(1.05))["groups"]["EX"]
    assert not r["checks"]["edep_mean"] and not r["checks"]["edep_mpv"]


def test_merge_is_additive():
    a, b = _state(1.0), _state(1.0)
    tot = V._new_state()
    V._merge_into(tot, a)
    V._merge_into(tot, b)
    assert tot["ref"]["groups"]["EX"]["n"] == 2 * a["ref"]["groups"]["EX"]["n"]
