"""Synthetic CPU regressions for skipping idle, opt-in full-EM slab work."""
import importlib.util

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from mu2efastsim import eslab_jax as SJ
from beamfastsim import cechain as C
from beamfastsim import cechain_jax as CJ
from beamfastsim import geometry as G

pytestmark = pytest.mark.skipif(not hasattr(SJ, "lane_step_emissions"),
                                reason="full core emission interface required")


def _geom():
    b = G.BeamGeometry(0.0, np.array([5.0, 10.0]), np.zeros(2), np.full(2, 75.0), np.full(2, 0.3),
                       -50.0, 1.0, 100.0, 200.0, 300.0, 0.5, 100.0, 200.0, 900.0, 100.0, 200.0)
    return C.CEGeometry(b, 900.0, 900.0, 1.0, z_plane=30.0, z_up=-20.0, z_ds=-30.0,
                        coll5_z0=-29.0, coll5_z1=-28.0)


def _build(module, geom, dtype, positron, capacity=16):
    return module._build(geom, np.dtype(dtype), 1.0, 100.0, False, True, True, 15.0,
                         False, np.zeros(3), positron=positron, photon_capacity=capacity)


def _initial(init, dtype, vacuum=False):
    z = [12.0] * 4 if vacuum else [0.0, 3.0, 5.0, 12.0]
    return init((), jnp.asarray([[2.0, 0.0, zz] for zz in z], dtype),
                jnp.asarray([[0.0, 0.0, 1.0]] * 4, dtype), jnp.full(4, 100.0, dtype),
                jnp.zeros(4, dtype), jnp.asarray([-1, -1, -1 if vacuum else 0, -1], jnp.int32),
                jnp.asarray([0.0, 0.0, 0.3, 0.0], dtype), jax.random.PRNGKey(7))


def _assert_bytes(a, b):
    aa, ta = jax.tree_util.tree_flatten_with_path(a)
    bb, tb = jax.tree_util.tree_flatten_with_path(b)
    assert ta == tb
    for (path, av), (_, bv) in zip(aa, bb):
        av, bv = np.asarray(av), np.asarray(bv)
        assert av.dtype == bv.dtype and av.shape == bv.shape, str(path)
        assert av.tobytes() == bv.tobytes(), str(path)


class _UnconditionalEM:
    """Test-only original algorithm: run full EM even when no slab lane is live."""
    def __getattr__(self, name):
        return getattr(jax.lax, name)

    def cond(self, pred, yes, no, operand):
        if yes.__name__ == "full_em":
            return yes(operand)
        return jax.lax.cond(pred, yes, no, operand)


def _reference():
    spec = importlib.util.spec_from_file_location("beamfastsim._idle_unconditional_reference", CJ.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.lax = _UnconditionalEM()
    return module


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("positron", [False, True])
def test_finished_core_step_only_advances_iteration(dtype, positron):
    if dtype == np.float64 and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=true")
    geom = _geom()
    lt = SJ.lane_tables(CJ.layer_types(geom)[0], positron=positron, msc=False)
    st = SJ.lane_state(jnp.asarray([100.0, 0.0], dtype), jnp.asarray([[0, 0, 1]] * 2, dtype),
                       jnp.zeros(2, dtype), jnp.full(2, 0.6, dtype), positron=positron)
    st["exited"] = jnp.asarray([True, False])
    st["it"] = jnp.int32(19)
    step = jax.jit(lambda s: SJ.lane_step_emissions(jax.random.PRNGKey(3), lt, jnp.zeros(2, jnp.int32),
                                                  s, positron=positron, msc=False))
    for _ in range(3):
        new, emitted = step(st)
        _assert_bytes(new, dict(st, it=st["it"] + 1))
        assert not np.asarray(emitted.valid).any()
        st = new


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("positron", [False, True])
def test_full_state_bytes_match_unconditional_over_vacuum_and_slabs(dtype, positron):
    if dtype == np.float64 and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=true")
    geom = _geom()
    init, run = _build(CJ, geom, dtype, positron)
    old_init, old_run = _build(_reference(), geom, dtype, positron)
    st, old = _initial(init, dtype), _initial(old_init, dtype)
    # Retain nonempty records/counters through idle iterations, not just all-zero buffers.
    for state in (st, old):
        state["count"] = jnp.ones(4, jnp.int32)
        state["records"] = {name: value.at[:, 0].set(True if name == "valid" else 1)
                            for name, value in state["records"].items()}
        for name in ("nph", "eph", "nde", "ede", "nan", "ean", "cont"):
            state[name] = state[name] + 1
    _assert_bytes(st, old)
    slab_iterations = idle_iterations = 0
    # One-lane slab gather exercises both padding and contention; no Python loops in production.
    for i in range(40):
        active = np.asarray(st["inl"]) & (np.asarray(st["fate"]) == C.LIVE)
        slab_iterations += int(active.any())
        idle_iterations += int(not active.any())
        key = jax.random.fold_in(jax.random.PRNGKey(17), i // 8)
        ni, st = run((), key, st, jnp.int32(1), 2, 1)
        oi, old = old_run((), key, old, jnp.int32(1), 2, 1)
        _assert_bytes((ni, st), (oi, old))
    assert slab_iterations > 1 and idle_iterations > 1
    assert np.asarray(st["cnt"])[:, 0].max() == 2
    assert np.asarray(st["rec_ok"]).all()
    assert np.asarray(st["nslab"]).sum() > 2


@pytest.mark.parametrize("positron", [False, True])
def test_idle_does_not_execute_heavy_recorder(monkeypatch, positron):
    calls = []
    original = SJ.lane_step_emissions

    def recorder(*args, **kwargs):
        # A runtime effect, not a trace-time Python count: lax.cond traces both branches.
        jax.debug.callback(lambda it: calls.append(int(it)), args[3]["it"], ordered=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(SJ, "lane_step_emissions", recorder)
    # Fresh geometry keeps this instrumented executable separate from cached builds.
    init, run = _build(CJ, _geom(), np.float32, positron)
    st = _initial(init, np.float32, vacuum=True)
    before = {name: st[name] for name in ("records", "count", "nph", "eph", "nde", "ede", "nan", "ean", "cont")}
    _, idle = run((), jax.random.PRNGKey(9), st, jnp.int32(5), 2, 2)
    jax.block_until_ready(idle)
    assert calls == []
    assert int(idle["slab"]["it"]) == 5
    _assert_bytes({name: idle[name] for name in before}, before)
    _assert_bytes(idle["slab"], dict(st["slab"], it=st["slab"]["it"] + 5))
    active = _initial(init, np.float32)
    _, stepped = run((), jax.random.PRNGKey(9), active, jnp.int32(1), 2, 2)
    jax.block_until_ready(stepped)
    assert calls == [0], "the sentinel must execute when a slab lane is active"
