"""CaloShowerStep -> photoelectron groups: deterministic counts, Poisson and travel-time modes."""
import numpy as np

from calofastsim import response
from calofastsim.readout import PropagationTable, ReadoutConfig, make_readouts


def _steps(n=50, seed=1):
    rng = np.random.default_rng(seed)
    timing = {"pbtime": np.array([-210.0, -200.0], np.float32), "mbtime": np.full(2, 1695.0, np.float32),
              "event_length": np.full(2, 1700.0), "onspill": np.ones(2, bool), "marker": np.full(2, 200.0, np.float32),
              "crystal_length": np.full(2, 200.0, np.float32)}
    steps = {"step_event": rng.integers(0, 2, n), "crystal": rng.integers(0, 674, n),
             "energy_g4": rng.uniform(0, 20, n).astype(np.float32), "energy_birks": rng.uniform(0, 20, n).astype(np.float32),
             "time": rng.uniform(0, 3000, n), "z_local": rng.uniform(0, 200, n).astype(np.float32),
             "lru": np.full(n, 0.05, np.float32), "yields": np.full((n, 2), 30.0, np.float32)}
    return steps, timing


def _call(steps, timing, cfg, **kw):
    return make_readouts(steps["step_event"], steps["crystal"], steps["energy_g4"], steps["energy_birks"],
                         steps["time"], steps["z_local"], steps["lru"], steps["yields"], timing, cfg, **kw)


def test_deterministic_counts_and_times():
    steps, timing = _steps()
    ev, sipm, t, count = _call(steps, timing, ReadoutConfig(pe_stat=False, travel=False))
    ev_s = steps["step_event"]
    hit, ok = response.fold_step_time(steps["time"], pbtime=timing["pbtime"][ev_s], mbtime=timing["mbtime"][ev_s],
                                      event_length=timing["event_length"][ev_s], onspill=True, digitization_start=450.0,
                                      digitization_end=1795.0, digitization_buffer=10.0,
                                      time_from_protons_to_dr_marker=200.0)
    e = response.corrected_energy(steps["energy_g4"], steps["energy_birks"], steps["z_local"], 200.0, 0.05)
    n = response.nonnegative_lround(response.photoelectron_mean(e, steps["yields"]))
    a = np.flatnonzero(ok)
    exp_step = np.repeat(a, 2)
    exp_sipm = 2 * steps["crystal"][exp_step] + np.tile([0, 1], a.size)
    exp_count = n[a].ravel().astype(np.int64)
    keep = exp_count > 0
    np.testing.assert_array_equal(ev, ev_s[exp_step][keep])
    np.testing.assert_array_equal(sipm, exp_sipm[keep])
    np.testing.assert_array_equal(count, exp_count[keep])
    np.testing.assert_array_equal(t, hit[exp_step][keep].astype(np.float32))


def test_poisson_and_travel_expand_to_single_pes():
    steps, timing = _steps(seed=2)
    contents = np.ones((20, 50), np.float32)
    table = PropagationTable.from_histogram(contents, np.linspace(0, 200, 21), np.linspace(0, 50, 51))
    rng = np.random.default_rng(4)
    ev, sipm, t, count = _call(steps, timing, ReadoutConfig(), table=table, rng=rng)
    assert np.all(count == 1)
    ev0, sipm0, t0, c0 = _call(steps, timing, ReadoutConfig(travel=False), rng=np.random.default_rng(4))
    assert c0.sum() == t.size
    assert t.min() >= t0.min() and t.max() <= t0.max() + 50.0
