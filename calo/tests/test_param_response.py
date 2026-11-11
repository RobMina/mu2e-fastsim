"""Parametric digi response (opt-in digi_response=parametric): table build, the per-cell draw and the chain wiring."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("scipy")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from calofastsim import device_chain as dc, param_response as PR  # noqa: E402
from calofastsim.readout import PropagationTable, ReadoutConfig  # noqa: E402

from test_device_chain import _geom, _grid, _pulse, _sim, _timing  # noqa: E402


def _table(tmp_path, threshold=1.0, ratio=0.9, dt=3.0):
    """Every cell above `threshold` MeV gets a hit with E * ratio at T + dt; below it none."""
    e = np.geomspace(0.02, 200, 4000)
    has = e > threshold
    v = np.full(e.size, -1000.0)
    v[::40] = np.linspace(-150, 30, v[::40].size)          # a few cells near the window end: hits up to v = 0
    has = has & (v < 0)
    tab = PR.build(e, has, np.full(e.size, ratio), np.full(e.size, dt), v)
    path = tmp_path / "t.npz"
    PR.save(path, tab)
    return path, tab


def test_build_pools_and_borrows_empty_bins():
    e = np.array([0.5, 0.6, 50.0, 40.0])
    tab = PR.build(e, np.array([False, True, True, False]), np.array([0, 1.0, 0.95, 0]), np.array([0, 1.0, 2.0, 0]),
                   np.array([-500.0, -500.0, -500.0, 10.0]))
    assert tab["n_inside"] == 3 and tab["survival"][-1] == 0.0      # past the window end: no hit
    filled = len(np.unique(np.searchsorted(PR.E_EDGES, e[:3], side="right") - 1))   # the inside cells
    assert filled == 3 and tab["empty_bins"].sum() == len(PR.E_EDGES) - 1 - filled
    assert np.all(tab["count"] > 0)                       # every bin draws from a filled pool
    b50 = np.searchsorted(tab["e_edges"], 50.0, side="right") - 1
    j = tab["start"][b50]
    assert tab["has_hit"][j] and tab["ratio"][j] == np.float32(0.95)
    assert not np.any(np.isnan(tab["ratio"])) and not np.any(np.isnan(tab["dt"]))


def _steps(rng, n_ev=6, per=12):
    pos, disk = _grid()
    ev = np.repeat(np.arange(n_ev), per)
    seedc = rng.integers(0, len(disk), n_ev)
    near = np.argsort(np.linalg.norm(pos[None, :, :] - pos[seedc][:, None, :], axis=2), axis=1)[:, :5]
    steps = {"event": ev, "crystal": near[ev, rng.integers(0, 5, ev.size)],
             "energy_g4": rng.uniform(0.05, 25, ev.size).astype(np.float32),
             "energy_birks": rng.uniform(0.05, 25, ev.size).astype(np.float32),
             "time": rng.uniform(600, 1400, n_ev)[ev] + rng.uniform(0, 1.5, ev.size),
             "z_local": rng.uniform(5, 195, ev.size).astype(np.float32)}
    return pos, disk, steps


def test_parametric_hits_are_the_cell_sums(tmp_path):
    path, _ = _table(tmp_path)
    pos, disk, steps = _steps(np.random.default_rng(3))
    geom = _geom(pos, disk)
    ro = ReadoutConfig(pe_stat=False, travel=False)
    table = PropagationTable(np.ones((1, 1), np.float32), np.zeros(1, np.float32), np.float32(1.0))
    resp = dc.pack_response(_sim(len(disk)), _pulse(), np.full(10000, 12.0), table, geom, ro_cfg=ro,
                            pbt_reco_sigma=0.0)
    caps = dc.Caps(events=8, steps=steps["event"].size, rows=128, digis=128, digi_length=64, hits=128,
                   event_hits=32, rounds=6)
    t = dc.entries_batch({k: np.zeros((0, 3) if k in ("position", "direction") else 0) for k in
                          ("energy", "cosine", "radius", "disk", "time", "position", "direction")},
                         _timing(6), np.zeros(0, int), caps.events)[1]
    tj = {k: jnp.asarray(v) for k, v in t.items()}
    n = steps["event"].size
    st = dict(row=jnp.arange(n), time=jnp.asarray(steps["time"]), energy_g4=jnp.asarray(steps["energy_g4"]),
              energy_birks=jnp.asarray(steps["energy_birks"]), valid=jnp.ones(n, bool))
    hits, over, _ = jax.tree_util.tree_map(np.asarray, dc.param_hits(
        resp, PR.to_device(PR.load(path)), jnp.asarray(steps["event"]), tj, st, jnp.asarray(steps["crystal"]),
        jnp.asarray(steps["z_local"]), jnp.ones(n, bool), jax.random.PRNGKey(0), tj["pbtime"].astype(jnp.float32),
        caps))
    assert not any(int(v) for v in over.values())
    # host cell sums (lru 0: the corrected energy is the Birks energy; times inside the window are not folded)
    key = steps["event"] * 10000 + steps["crystal"]
    cells, inv = np.unique(key, return_inverse=True)
    E = np.bincount(inv, steps["energy_birks"].astype(np.float64))
    T = np.bincount(inv, steps["energy_birks"] * steps["time"]) / E
    keep = E > 1.0
    k = int(hits["ok"].sum())
    assert k == keep.sum() and np.all(hits["ok"][:k]) and not hits["ok"][k:].any()
    np.testing.assert_array_equal(hits["event"][:k] * 10000 + hits["crystal"][:k], cells[keep])
    np.testing.assert_allclose(hits["e"][:k], 0.9 * E[keep], rtol=1e-6)
    np.testing.assert_allclose(hits["t"][:k], T[keep] + 3.0, atol=2e-4)
    # the chain: steps -> hits -> clusters, no digis
    chain = dc.make_steps_chain(resp, caps, digi_response=str(path))
    out = dc.run(chain, dc.steps_batch(steps, caps.steps), t, jax.random.PRNGKey(0))
    cl = dc.flat_clusters(out["clusters"])
    assert int(out["stats"]["digis"]) == 0 and int(out["stats"]["hits"]) == k
    np.testing.assert_allclose(cl["energy"].sum(), 0.9 * E[keep].sum(), rtol=1e-5)


def test_survival_curve_cuts_hits_past_the_window_end(tmp_path):
    _, tab = _table(tmp_path)
    tb = PR.to_device(tab)
    n = 4000
    E = jnp.full(n, 50.0)
    v = jnp.concatenate([jnp.full(n // 2, -500.0), jnp.full(n // 2, 30.0)])
    hit, e, t = (np.asarray(a) for a in PR.draw(tb, jax.random.PRNGKey(1), jnp.ones(n, bool), E, jnp.zeros(n), v))
    assert hit[: n // 2].all() and not hit[n // 2:].any()
    np.testing.assert_allclose(e[hit], 45.0, rtol=1e-6)


def test_waveform_is_the_default():
    import inspect
    for f in (dc.make_chain, dc.make_steps_chain):
        assert inspect.signature(f).parameters["digi_response"].default == "waveform"
    assert dc._digi_table("waveform") is None and dc._digi_table(None) is None
