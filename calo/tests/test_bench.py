"""Shared-timer entry points of the calorimeter device chain: inputs from export tables, fresh inputs per seed."""
import numpy as np
import pytest

uproot = pytest.importorskip("uproot")

from calofastsim import bench  # noqa: E402


def _tables(path, n_events=7, seed=0):
    rng = np.random.default_rng(seed)
    ev = np.arange(1, n_events + 1, dtype=np.uint32)
    per = rng.integers(0, 5, n_events)          # some events without steps
    sev = np.repeat(ev, per)
    m = len(sev)
    with uproot.recreate(path) as f:
        f["dump/events"] = {"run": np.full(n_events, 1430, np.uint32), "subrun": np.full(n_events, 7, np.uint32),
                            "event": ev}
        f["dump/showers"] = {"run": np.full(m, 1430, np.uint32), "subrun": np.full(m, 7, np.uint32),
                             "event": sev[::-1].copy(), "crystal": rng.integers(0, 20, m).astype(np.int32),
                             "time": rng.uniform(600, 1400, m), "energy_g4": rng.uniform(0.5, 20, m).astype(np.float32),
                             "energy_birks": rng.uniform(0.5, 20, m).astype(np.float32),
                             "position_z": rng.uniform(5, 195, m).astype(np.float32)}
    return per


def test_steps_per_event_and_fresh_offsets(tmp_path):
    per = _tables(tmp_path / "a.root")
    events = bench.load_steps([tmp_path / "a.root"])
    assert [len(e["crystal"]) for e in events] == list(per)
    a = bench.pack_batches(events, 10, batch=4, seed=3)
    b = bench.pack_batches(events, 10, batch=4, seed=4)
    assert len(a) == 3                                         # 10 events in batches of 4, the last one padded
    st, t = a[0]
    assert st["event"].shape == (bench.PER_EVENT["steps"] * 4,) and t["pbtime"].shape == (4,)
    assert set(np.unique(t["pbtime"])) <= set(np.float32(bench.PBTIMES))
    counts = lambda batches: [int((s["crystal"] >= 0).sum()) for s, _ in batches]  # noqa: E731
    assert sum(counts(a)) == sum(per[(np.random.default_rng(3).integers(7) + np.arange(10)) % 7])
    assert counts(a) != counts(b) or not np.array_equal(a[0][1]["pbtime"], b[0][1]["pbtime"])
    fixed = bench.pack_batches(events, 10, batch=4)
    assert sum(counts(fixed)) == sum(per[np.arange(10) % 7])


def test_caps_scale_with_batch():
    c = bench.caps(256, digi_length=64)
    assert c.events == 256 and c.steps == 256 * bench.PER_EVENT["steps"] and c.digi_length == 64
    assert c.rounds == bench.FIXED["rounds"]


def test_entry_point_runs(tmp_path, monkeypatch):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from calofastsim import device_chain as dc
    from calofastsim.cluster import Geometry
    from calofastsim.pulse import PulseCache
    from calofastsim.readout import PropagationTable
    _tables(tmp_path / "a.root", n_events=12)
    monkeypatch.setenv("CALO_BENCH_TABLES", str(tmp_path / "a.root"))
    pos = np.column_stack((np.arange(20) % 5 * 34.3, np.arange(20) // 5 * 34.3, np.zeros(20)))
    near = (np.abs(pos[:, None, :2] - pos[None, :, :2]) < 50).all(axis=2) & ~np.eye(20, dtype=bool)
    geom = Geometry(pos, np.zeros(20, int), near, np.zeros((1, 3)))
    x = (np.arange(6200) + 0.5) * 0.05
    pulse = PulseCache.from_histogram((x / 20.0) ** 2 * np.exp(-x / 10.0), 0.0, 310.0, 5.0)
    sim = {"lru": np.zeros(20, np.float32), "pe_per_mev": np.full((20, 2), 30.0, np.float32),
           "adc_per_mev": np.full((20, 2), 16.0, np.float32)}
    table = PropagationTable(np.array([[0.5, 1.0]], np.float32), np.array([0.5, 1.5], np.float32), np.float32(400.0))

    def conditions(dtype="float64", directory=None, fit_hessian="analytic"):
        return dc.pack_response(sim, pulse, np.full(10000, 12.0), table, geom, dtype=np.dtype(dtype),
                                fit_hessian=fit_hessian)
    monkeypatch.setattr(bench, "conditions", conditions)
    monkeypatch.setattr(bench, "_CHAINS", {})
    prepared = bench.bench_prepare(12, seed=5, batch=8)
    outs = bench.bench_steps(prepared, dtype="float32", fit_hessian="analytic", outliers=1 << 12)
    assert len(outs) == 2 and not any(bench.overflow(outs).values())
    assert np.asarray(outs[0]["clusters"]["valid"]).any()


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_fit_modes_keep_clusters(tmp_path, monkeypatch, dtype):
    """Two-pass refit and length buckets: the same clusters as one pass on all columns."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from calofastsim import device_chain as dc
    from calofastsim.cluster import Geometry
    from calofastsim.pulse import PulseCache
    from calofastsim.readout import PropagationTable
    _tables(tmp_path / "a.root", n_events=16, seed=4)
    monkeypatch.setenv("CALO_BENCH_TABLES", str(tmp_path / "a.root"))
    pos = np.column_stack((np.arange(20) % 5 * 34.3, np.arange(20) // 5 * 34.3, np.zeros(20)))
    near = (np.abs(pos[:, None, :2] - pos[None, :, :2]) < 50).all(axis=2) & ~np.eye(20, dtype=bool)
    geom = Geometry(pos, np.zeros(20, int), near, np.zeros((1, 3)))
    x = (np.arange(6200) + 0.5) * 0.05
    pulse = PulseCache.from_histogram((x / 20.0) ** 2 * np.exp(-x / 10.0), 0.0, 310.0, 5.0)
    sim = {"lru": np.zeros(20, np.float32), "pe_per_mev": np.full((20, 2), 30.0, np.float32),
           "adc_per_mev": np.full((20, 2), 16.0, np.float32)}
    table = PropagationTable(np.array([[0.5, 1.0]], np.float32), np.array([0.5, 1.5], np.float32), np.float32(400.0))

    def conditions(dtype="float64", directory=None, fit_hessian="analytic"):
        return dc.pack_response(sim, pulse, np.full(10000, 12.0), table, geom, dtype=np.dtype(dtype),
                                fit_hessian=fit_hessian)
    monkeypatch.setattr(bench, "conditions", conditions)
    monkeypatch.setattr(bench, "_CHAINS", {})
    prepared = bench.bench_prepare(16, seed=2, batch=16)
    ref = bench.bench_steps(prepared, dtype=dtype, outliers=1 << 12)
    alt = bench.bench_steps(prepared, dtype=dtype, outliers=1 << 12, refits=8, fit_short_length=48,
                            fit_short=16 * 16, fit_long=16 * 16)
    assert not any(bench.overflow(ref).values()) and not any(bench.overflow(alt).values())
    a, b = (jax.tree_util.tree_map(np.asarray, o[0]["clusters"]) for o in (ref, alt))
    np.testing.assert_array_equal(a["valid"], b["valid"])
    tol = 1e-9 if dtype == "float64" else 1e-3
    np.testing.assert_allclose(np.where(a["valid"], a["energy"], 0), np.where(b["valid"], b["energy"], 0), atol=tol)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_window_digis_equal_full(dtype):
    """Constant noise (deterministic): the windowed digitisation gives the full-waveform digis, wide rows included."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from calofastsim import device_chain as dc
    from calofastsim.cluster import Geometry
    from calofastsim.pulse import PulseCache
    from calofastsim.readout import PropagationTable
    rng = np.random.default_rng(5)
    n_ev, n_cr = 8, 20
    pos = np.column_stack((np.arange(n_cr) % 5 * 34.3, np.arange(n_cr) // 5 * 34.3, np.zeros(n_cr)))
    geom = Geometry(pos, np.zeros(n_cr, int), np.zeros((n_cr, n_cr), bool), np.zeros((1, 3)))
    x = (np.arange(6200) + 0.5) * 0.05
    pulse = PulseCache.from_histogram((x / 20.0) ** 2 * np.exp(-x / 10.0), 0.0, 310.0, 5.0)
    sim = {"lru": np.zeros(n_cr, np.float32), "pe_per_mev": np.full((n_cr, 2), 30.0, np.float32),
           "adc_per_mev": np.full((n_cr, 2), 16.0, np.float32)}
    table = PropagationTable(np.ones((1, 1), np.float32), np.zeros(1, np.float32), np.float32(1.0))
    resp = dc.pack_response(sim, pulse, np.full(10000, 12.0), table, geom, dtype=np.dtype(dtype))
    m = 400
    ev = rng.integers(0, n_ev, m)
    sipm = rng.integers(0, 2 * n_cr, m)
    pe_t = rng.uniform(300, 1200, n_ev)[ev] + rng.exponential(2.0, m)
    late = rng.random(m) < 0.03
    pe_t = np.where(late, pe_t + rng.uniform(150, 600, m), pe_t).astype(np.float32)   # late deposits: wide rows
    weight = rng.integers(1, 400, m).astype(np.float64)
    tm = {"pbtime": np.full(n_ev, -210.0, np.float32), "marker": np.full(n_ev, 200.0, np.float32)}
    tm = {k: jnp.asarray(v) for k, v in tm.items()}
    full = dc.Caps(events=n_ev, rows=256, digis=512, digi_length=128, outliers=4096)
    win = dc.Caps(events=n_ev, rows=256, digis=512, digi_length=128, outliers=4096, wave_window=96, wide_rows=64,
                  wide_digis=256)
    out = []
    for caps in (full, win):
        dg, over = jax.jit(lambda e, s_, p_, w_, c=caps: dc.digitise(resp, tm, e, s_, p_, w_, jnp.ones(m, bool),
                                                                     jax.random.PRNGKey(0), c))(
            jnp.asarray(ev), jnp.asarray(sipm), jnp.asarray(pe_t), jnp.asarray(weight))
        assert not any(int(v) for v in over.values())
        dg = jax.tree_util.tree_map(np.asarray, dg)
        ok = dg["ok"]
        rows = sorted((int(e), int(s_), int(t0), int(n), tuple(sm[:n])) for e, s_, t0, n, sm in
                      zip(dg["event"][ok], dg["sipm"][ok], dg["t0"][ok], dg["length"][ok], dg["samples"][ok]))
        out.append(rows)
    assert len(out[0]) > 10
    assert out[0] == out[1]
