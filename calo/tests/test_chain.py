"""End-to-end fast chain on synthetic steps: a deposit in one crystal comes back as one cluster of that energy."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from calofastsim.chain import run_chain  # noqa: E402
from calofastsim.pulse import PulseCache  # noqa: E402
from calofastsim.readout import ReadoutConfig  # noqa: E402
from test_cluster import _geometry  # noqa: E402


def test_single_deposit_round_trip():
    x = (np.arange(6200) + 0.5) * 0.05
    pulse = PulseCache.from_histogram((x / 20.0) ** 2 * np.exp(-x / 10.0), 0.0, 310.0, 5.0)
    geom = _geometry()
    n_ev = 3
    timing = {"pbtime": np.full(n_ev, -200.0, np.float32), "mbtime": np.full(n_ev, 1695.0, np.float32),
              "event_length": np.full(n_ev, 1700.0), "onspill": np.ones(n_ev, bool),
              "marker": np.full(n_ev, 200.0, np.float32), "crystal_length": np.full(n_ev, 200.0, np.float32)}
    steps = {"event": np.array([0, 1, 1]), "crystal": np.array([20, 40, 41]),
             "energy_g4": np.array([50.0, 30.0, 20.0], np.float32), "energy_birks": np.array([50.0, 30.0, 20.0], np.float32),
             "time": np.array([700.0, 900.0, 900.5]), "z_local": np.full(3, 100.0, np.float32),
             "lru": np.zeros(3, np.float32), "yields": np.full((3, 2), 30.0, np.float32)}
    noise = np.full(10000, 12.0)  # constant baseline: the stock fit objective divides by it
    out = run_chain(steps, timing, lambda s: np.full(np.shape(s), np.float32(16) / np.float32(30)), pulse, noise,
                    geom, np.random.default_rng(1), ro_cfg=ReadoutConfig(pe_stat=False, travel=False))
    cl = out["clusters"]
    assert list(cl["event"]) == [0, 1]
    np.testing.assert_allclose(cl["energy"], [50.0, 50.0], rtol=0.03)
    assert list(cl["size"]) == [1, 2]
