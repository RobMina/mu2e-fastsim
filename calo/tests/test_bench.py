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
    """The default fit plan (two-pass refit, length buckets): the same clusters as one pass on all columns."""
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
    ref = bench.bench_steps(prepared, dtype=dtype, outliers=1 << 12, refits=0, fit_short_length=0)
    alt = bench.bench_steps(prepared, dtype=dtype, outliers=1 << 12)
    assert not any(bench.overflow(ref).values()) and not any(bench.overflow(alt).values())
    a, b = (jax.tree_util.tree_map(np.asarray, o[0]["clusters"]) for o in (ref, alt))
    np.testing.assert_array_equal(a["valid"], b["valid"])
    tol = 1e-9 if dtype == "float64" else 1e-3
    np.testing.assert_allclose(np.where(a["valid"], a["energy"], 0), np.where(b["valid"], b["energy"], 0), atol=tol)


def test_fit_plan_auto_and_explicit():
    from calofastsim import device_chain as dc
    assert dc.fit_plan(dc.Caps(digis=65536, digi_length=96)) == (256, 48, 40960, 16384)
    assert dc.fit_plan(dc.Caps(digis=100, digi_length=96)) == (32, 48, 63, 25)
    assert dc.fit_plan(dc.Caps(digis=512, digi_length=48))[1] == 0            # no shorter bucket than the digis
    assert dc.fit_plan(dc.Caps(refits=0, fit_short_length=0)) [:2] == (0, 0)  # one pass, one bucket
    assert dc.fit_plan(dc.Caps(refits=8, fit_short_length=32, fit_short=10, fit_long=5)) == (8, 32, 10, 5)
