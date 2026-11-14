"""Device TS transport: geometry and field parity with the NumPy route, and transport invariants."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from beamfastsim import tsdevice as dev, tsphysics as phys  # noqa: E402
from beamfastsim.tsfield import FieldBank  # noqa: E402
from beamfastsim.tsgeometry import Geometry  # noqa: E402
from mu2efastsim import ehelix, eloss_jax, materials  # noqa: E402

from test_tsfield import header  # noqa: E402
from test_tsgeometry import _fixture  # noqa: E402


@pytest.fixture(scope="module")
def geometry(tmp_path_factory):
    path = tmp_path_factory.mktemp("g") / "synthetic.gdml"
    ET.ElementTree(_fixture()).write(path)
    return Geometry(path)


def test_geometry_parity(geometry):
    g = dev.DeviceGeometry(geometry)
    rng = np.random.default_rng(1)
    lo = np.array([-500.0, -400.0, geometry.z_start - 100])
    hi = np.array([4500.0, 400.0, geometry.z_end + 100])
    a = rng.uniform(lo, hi, (4000, 3))
    b = a + rng.normal(0, 30, a.shape)
    np.testing.assert_array_equal(np.asarray(g.inside(jnp.asarray(a))), geometry.inside(a))
    cols = np.asarray(g.columns(jnp.asarray(a), jnp.asarray(b)))
    ref = geometry.columns(a, b)
    for m, name in enumerate(g.names):
        np.testing.assert_allclose(cols[:, m], ref[name], rtol=1e-12, atol=1e-9)
    assert any(ref[name].any() for name in g.names)


def test_field_parity(tmp_path):
    rng = np.random.default_rng(2)
    maps = [header(tmp_path, "a", grid=rng.normal(size=(4, 3, 5, 3)), reflect=True),
            header(tmp_path, "b", offset=(1.5, -2, 0), spacing=(0.5, 1, 1), grid=rng.normal(size=(5, 5, 6, 3)))]
    bank = FieldBank(maps, scale=1.5)
    fd = dev.DeviceField([(*ehelix.read_g4bl(str(h)), "flip" in h.read_text())
                          for h in maps], 1.5)
    x = rng.uniform([-1, -3, -0.5], [4, 3, 5.5], (3000, 3))
    b, ok = bank.evaluate(x)
    bd, okd = (np.asarray(v) for v in fd.evaluate(jnp.asarray(x)))
    np.testing.assert_array_equal(okd, ok)
    np.testing.assert_allclose(bd, b, rtol=1e-12, atol=1e-12)


class Tube:
    """Field-free straight vacuum tube z in [0, 100] with an optional aluminium slab at z in [40, 40 + slab]."""
    z_start, z_end = 0.0, 100.0

    def __init__(self, slab=0.0):
        self.names = ["Al"] if slab else []
        self.consts = [eloss_jax.consts(materials.ALUMINIUM)] if slab else []
        self.x0 = np.array([89.0])[:len(self.names)]
        self.slab = slab

    def inside(self, x):
        return (x[:, 2] >= -1e-8) & (x[:, 2] <= 100 + 1e-8) & (jnp.hypot(x[:, 0] + 100, x[:, 1]) < 500)

    def columns(self, a, b):
        if not self.slab:
            return jnp.zeros((a.shape[0], 0))
        d = b - a
        lo, hi = jnp.minimum(a[:, 2], b[:, 2]), jnp.maximum(a[:, 2], b[:, 2])
        overlap = jnp.clip(jnp.minimum(hi, 40 + self.slab) - jnp.maximum(lo, 40), 0, None)
        dz = jnp.where(hi > lo, hi - lo, 1.0)
        return (jnp.linalg.norm(d, axis=1) * jnp.where(hi > lo, overlap / dz, 0.0))[:, None]


class Zero:
    max_field_t = 0.0

    def evaluate(self, x):
        return jnp.zeros_like(x), jnp.ones(x.shape[0], bool)


def _run(geom, pdg, p, n=200, settings=dev.Settings(step_mm=7.0)):
    chunk = dev.make_ts_chunk(Zero(), geom, settings, chunk=64)
    x = np.tile([-100.0, 0, 0], (n, 1))
    return dev.transport_ts(chunk, geom, x, np.tile([0, 0, float(p)], (n, 1)), np.zeros(n), np.full(n, pdg),
                            jax.random.PRNGKey(0), min_bucket=64)


def test_straight_flight_reaches_plane():
    out = _run(Tube(), 13, 1000.0)
    ds = np.asarray(out.fate) == dev.DS
    assert ds.sum() >= 199
    np.testing.assert_allclose(np.asarray(out.x)[ds], np.tile([-100, 0, 100], (ds.sum(), 1)), atol=1e-9)
    np.testing.assert_allclose(np.asarray(out.t)[ds], 100 * np.hypot(1000, phys.MMU) / (1000 * phys.C_MM_NS))


def test_rangeout_in_slab_and_material_cap():
    out = _run(Tube(slab=20.0), 13, 20.0, settings=dev.Settings(step_mm=7.0, material_step_mm=0.1))
    fate = np.asarray(out.fate)
    assert np.all((fate == dev.RANGEOUT) | (fate == dev.MU_DECAY))   # a rare decay in flight is allowed
    rangeout = fate == dev.RANGEOUT
    assert rangeout.sum() >= 195
    assert np.all(np.asarray(out.mpath) < 3)
    assert np.all(np.asarray(out.steps)[rangeout] > 10)     # material-capped steps
    np.testing.assert_array_equal(np.asarray(out.p)[rangeout], 0)


def test_vacuum_step_is_opt_in_and_keeps_the_straight_flight():
    """ts_vacuum_step: long steps through the field-free vacuum reach the plane exactly, in fewer steps."""
    fixed = _run(Tube(), 13, 1000.0, n=50)
    long = _run(Tube(), 13, 1000.0, n=50, settings=dev.Settings(step_mm=7.0, vacuum_step_mm=30.0))
    for out in (fixed, long):
        ds = np.asarray(out.fate) == dev.DS
        assert ds.sum() >= 49
        np.testing.assert_allclose(np.asarray(out.x)[ds], np.tile([-100, 0, 100], (ds.sum(), 1)), atol=1e-9)
        np.testing.assert_allclose(np.asarray(out.t)[ds], 100 * np.hypot(1000, phys.MMU) / (1000 * phys.C_MM_NS))
    assert np.max(np.asarray(long.steps)) <= 5 < np.min(np.asarray(fixed.steps))
    assert dev.Settings().vacuum_step_mm == 0


def test_vacuum_step_keeps_the_material_steps():
    """A trial chord that crosses material is not taken: the slab is still crossed in capped steps."""
    settings = dev.Settings(step_mm=7.0, material_step_mm=0.1, vacuum_step_mm=30.0)
    out = _run(Tube(slab=20.0), 13, 20.0, settings=settings)
    fate = np.asarray(out.fate)
    rangeout = fate == dev.RANGEOUT
    assert rangeout.sum() >= 195
    assert np.all(np.asarray(out.mpath) < 3)
    assert np.all(np.asarray(out.steps)[rangeout] > 10)
    assert np.all(np.asarray(out.path)[rangeout] >= 40)       # every lane reached the slab


def test_full_gather_bucket_defers_lanes(monkeypatch):
    """With a small shrink bucket, over-cap lanes beyond it wait a step; every lane still ranges out."""
    monkeypatch.setattr(dev, "SHRINK_LEVELS", (0.05,))
    out = _run(Tube(slab=20.0), 13, 20.0, n=1000, settings=dev.Settings(step_mm=7.0, material_step_mm=0.1))
    fate = np.asarray(out.fate)
    assert np.all((fate == dev.RANGEOUT) | (fate == dev.MU_DECAY))
    assert np.sum(np.asarray(out.waits)) > 0
    assert not np.any(fate == dev.UNCONVERGED)


def test_material_bucket_runs_only_in_material_lanes(monkeypatch):
    """Material slots on a gathered bucket, with the every-lane branch when more lanes are in material: the slab
    ranges out as with every lane (fraction 0.9: the bucket once a tenth has stopped; 0.02: mostly every lane)."""
    monkeypatch.setattr(dev, "GATHER_FLOOR", 8)
    settings = dev.Settings(step_mm=7.0, material_step_mm=0.1)
    runs = {}
    for frac in (0.0, 0.9, 0.02):
        monkeypatch.setattr(dev, "MATERIAL_FRACTION", frac)
        runs[frac] = _run(Tube(slab=20.0), 13, 20.0, n=1000, settings=settings)
    ref = np.asarray(runs[0.0].path)[np.asarray(runs[0.0].fate) == dev.RANGEOUT]
    for frac, out in runs.items():
        fate = np.asarray(out.fate)
        assert np.all((fate == dev.RANGEOUT) | (fate == dev.MU_DECAY)), frac
        assert not np.any(fate == dev.UNCONVERGED)
        path = np.asarray(out.path)[fate == dev.RANGEOUT]
        assert abs(path.mean() - ref.mean()) < 5 * np.hypot(path.std() / np.sqrt(len(path)), ref.std() / np.sqrt(len(ref)))


def test_solid_bucket_is_exact(geometry, monkeypatch):
    """Generic-solid interval lengths on a candidate bucket equal the every-lane result to rounding, through the
    bucket and through the every-lane branch."""
    g = dev.DeviceGeometry(geometry)
    if not g._matter:
        pytest.skip("fixture has no generic solid")
    rng = np.random.default_rng(4)
    lo = np.array([-500.0, -400.0, geometry.z_start - 100])
    hi = np.array([4500.0, 400.0, geometry.z_end + 100])
    a = jnp.asarray(rng.uniform(lo, hi, (4000, 3)))
    d = jnp.asarray(rng.normal(0, 30, (4000, 3)))
    monkeypatch.setattr(dev, "GATHER_FLOOR", 4)
    out = {}
    for frac in (0.0, 0.5, 0.001):
        monkeypatch.setattr(dev, "SOLID_FRACTION", frac)
        out[frac] = np.asarray(jax.jit(g.columns_d)(a, d))
    assert out[0.0].any()
    np.testing.assert_allclose(out[0.5], out[0.0], rtol=1e-12, atol=1e-9)    # summation order may differ
    np.testing.assert_allclose(out[0.001], out[0.0], rtol=1e-12, atol=1e-9)


@pytest.mark.slow   # statistical, 4000 pions
def test_pion_decay_fraction():
    out = _run(Tube(), 211, 100.0, n=4000)
    prob = 1 - np.exp(-100 * phys.MPI / (100 * phys.C_MM_NS * phys.TAU_PI_NS))
    n = np.asarray(out.decayed).sum()
    assert abs(n - 4000 * prob) < 5 * np.sqrt(4000 * prob * (1 - prob))
    np.testing.assert_array_equal(np.asarray(out.pdg)[np.asarray(out.decayed)], -13)
    assert not np.any(np.asarray(out.fate) == dev.LIVE)


def test_resident_driver_compacts_and_keeps_rows():
    calls = []

    @jax.jit
    def chunk(key, st):
        return st._replace(fate=jnp.where(st.steps >= 0, jnp.int8(dev.DS), st.fate), steps=st.steps + 1)

    def run(key, st):
        calls.append(st.x.shape[0])
        return chunk(key, st)

    st = dev.ts_initial(np.arange(30.0).reshape(10, 3), np.ones((10, 3)), np.zeros(10), np.full(10, 13), 0,
                        jax.random.PRNGKey(1))
    out, _ = dev.run_resident(run, st, lambda s: s.fate == dev.LIVE, lambda s, pad: s, jax.random.PRNGKey(2))
    np.testing.assert_array_equal(np.asarray(out.fate), dev.DS)
    np.testing.assert_array_equal(np.asarray(out.x), np.arange(30.0).reshape(10, 3))
    assert calls == [10]
    calls.clear()
    out, _ = dev.run_resident(run, st, lambda s: s.fate == dev.LIVE, lambda s, pad: s, jax.random.PRNGKey(2), pool=4)
    np.testing.assert_array_equal(np.asarray(out.fate), dev.DS)          # every input ran once, rows kept
    np.testing.assert_array_equal(np.asarray(out.x), np.arange(30.0).reshape(10, 3))
    np.testing.assert_array_equal(np.asarray(out.steps), 1)
    assert calls == [4, 4, 4]


_COMPILES = []


def _count_compile(event, duration, **kw):
    if event == "/jax/core/compile/backend_compile_duration":
        _COMPILES.append(duration)


jax.monitoring.register_event_duration_secs_listener(_count_compile)


@jax.jit
def _die_at_t(key, st):
    """Toy chunk: a lane finishes once its step count reaches its t (so the live tail follows the data)."""
    steps = jnp.where(st.fate == dev.LIVE, st.steps + 1, st.steps)
    return st._replace(steps=steps, fate=jnp.where((st.fate == dev.LIVE) & (steps >= st.t), jnp.int8(dev.DS), st.fate))


def _toy_state(lifetimes):
    n = len(lifetimes)
    return dev.ts_initial(np.zeros((n, 3)), np.ones((n, 3)), np.asarray(lifetimes, float), np.full(n, 13), 0,
                          jax.random.PRNGKey(1))


def test_warm_ladder_is_exact_and_new_tails_do_not_compile():
    rng = np.random.default_rng(3)
    a, b = _toy_state(rng.integers(1, 6, 96)), _toy_state(np.r_[rng.integers(1, 3, 95), 40])
    assert dev.ladder(16, 4) == [16, 8, 4] and dev.ladder(20, 4) == [20, 16, 8, 4] and dev.ladder(3, 4) == [3]
    kw = dict(min_bucket=4, pool=16)
    for st in (a, b):
        cold, _ = dev.run_resident(_die_at_t, st, dev._ts_live, dev._ts_kill, jax.random.PRNGKey(2), **kw)
        warm, _ = dev.run_resident(_die_at_t, st, dev._ts_live, dev._ts_kill, jax.random.PRNGKey(2), warm_ladder=True,
                                   **kw)
        for x, y in zip(cold, warm):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    c = _toy_state(np.r_[rng.integers(1, 4, 90), [7, 9, 13, 21, 33, 50]])      # a tail through other bucket sizes
    before = len(_COMPILES)
    dev.run_resident(_die_at_t, c, dev._ts_live, dev._ts_kill, jax.random.PRNGKey(5), warm_ladder=True, **kw)
    assert len(_COMPILES) == before


def test_cache_every_compile_only_with_a_cache_dir(monkeypatch, tmp_path):
    keep = (jax.config.jax_compilation_cache_dir, jax.config.jax_persistent_cache_min_compile_time_secs,
            jax.config.jax_persistent_cache_min_entry_size_bytes)
    try:
        jax.config.update("jax_compilation_cache_dir", None)
        assert dev.cache_every_compile() is False
        jax.config.update("jax_compilation_cache_dir", str(tmp_path))
        monkeypatch.setattr(dev, "CACHE_ALL", False)
        assert dev.cache_every_compile() is False
        monkeypatch.setattr(dev, "CACHE_ALL", True)
        assert dev.cache_every_compile() is True
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
        assert jax.config.jax_persistent_cache_min_entry_size_bytes == -1
    finally:
        jax.config.update("jax_compilation_cache_dir", keep[0])
        jax.config.update("jax_persistent_cache_min_compile_time_secs", keep[1])
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", keep[2])


def test_float64_shortening_defaults(monkeypatch):
    """Float64 defaults: margin 1e-3 and the float32 cascade buckets; the earlier behaviour stays reachable."""
    import os
    if "TSDEV_SHRINK_MARGIN_F64" not in os.environ:
        assert dev.SHRINK_MARGIN_F64 == 1e-3
    assert dev.SHRINK_LEVELS_F64 == dev.SHRINK_LEVELS_F32 == (0.3, 0.015625, 0.00390625)
    monkeypatch.setattr(dev, "SHRINK_LEVELS", (0.3, 0.25, 0.00390625))
    assert dev.shrink_levels() == (0.3, 0.25, 0.00390625)


def test_load_references_and_stream_split(tmp_path, monkeypatch):
    paths = []
    for i, n in enumerate((3, 5)):
        path = tmp_path / f"ref{i}.npz"
        np.savez(path, x=np.full((n, 3), float(i)), p=np.ones((n, 3)), t=np.arange(n, dtype=float),
                 pdg=np.full(n, 13), id=np.arange(n), shard=np.full(n, i), event_id=np.zeros((7, 3)))
        paths.append(path)
    ref = dev.load_references(paths)
    np.testing.assert_array_equal(ref["batch"], [0, 0, 0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(ref["t"], [0, 1, 2, 0, 1, 2, 3, 4])
    assert "event_id" not in ref                      # exposure tables are not per input row

    def fake(chain, x, p, t, pdg, seed):
        st = dev.TSState(*(jnp.asarray(t) for _ in dev.TSState._fields))
        return st, st

    monkeypatch.setattr(dev, "run_chain", fake)
    batches = [{k: ref[k][ref["batch"] == b] for k in ("x", "p", "t", "pdg")} for b in (0, 1)]
    out = dev.stream_chain(None, batches)
    np.testing.assert_array_equal(np.asarray(out[0][0].t), [0, 1, 2])
    np.testing.assert_array_equal(np.asarray(out[1][1].t), [0, 1, 2, 3, 4])
