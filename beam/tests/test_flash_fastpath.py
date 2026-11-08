"""Raw-state A/B coverage for the opt-in empty-work navigator guards."""
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from beamfastsim import cechain as C
from beamfastsim import flash_navigator_jax as FN
from test_flash_target_jax import AXIS, KEY, _geom, _starts


def _build(geom, skip_empty=False, msc=False, s_max=1000., field=(0., 0., 0.)):
    return FN._build(geom, np.float64, 1., s_max, msc, True, True, False,
                     field, photon_capacity=32, skip_empty=skip_empty)


def _initial(init, x, u):
    n = len(x)
    return init((), jnp.asarray(x), jnp.asarray(u), jnp.full(n, 105.),
                jnp.zeros(n), jnp.full(n, -1, jnp.int32), jnp.zeros(n))


def _compare(a, b):
    leaves_a, tree_a = jax.tree_util.tree_flatten_with_path(a)
    leaves_b, tree_b = jax.tree_util.tree_flatten_with_path(b)
    assert tree_a == tree_b
    for (path, av), (_, bv) in zip(leaves_a, leaves_b):
        av, bv = np.asarray(av), np.asarray(bv)
        assert av.shape == bv.shape and av.dtype == bv.dtype, str(path)
        np.testing.assert_array_equal(av, bv, err_msg=str(path))
        assert av.tobytes() == bv.tobytes(), str(path)


def test_cache_key_and_default():
    geom = _geom()
    args = (geom, np.float64, 1., 1000., False, True, True, False, (0., 0., 0.))
    default = FN._build(*args, photon_capacity=32)
    assert default is _build(geom, False)
    assert _build(geom, True) is _build(geom, True)
    assert default is not _build(geom, True)


@pytest.mark.slow
def test_empty_candidates_preserve_frames_and_rng_iteration():
    geom = _geom(foil_z=(5000.,), ht=.0528, plane=5800.)
    x, u = _starts(6, z=5700.)
    runs = []
    for skip_empty in (False, True):
        init, run = _build(geom, skip_empty)
        initial = _initial(init, x, u)
        steps, state = run((), KEY, initial, jnp.int32(7), 3, 3)
        assert int(steps) == 7
        assert int(state["slab"]["it"]) == int(initial["slab"]["it"]) + 7
        for name in ("e1", "e2", "en", "xent", "Bl", "Tin", "lty", "records", "count"):
            _compare(state[name], initial[name])
        np.testing.assert_array_equal(state["nvac"], 7)
        assert not np.asarray(state["inl"]).any()
        runs.append(state)
    _compare(*runs)


@pytest.mark.slow
@pytest.mark.parametrize("msc, field", [(False, (0., 0., 0.)), (True, (0., 0., 0.)), (True, (0., 0., 1.))])
def test_foil_crossings_msc_finished_lanes_and_same_shape_chunks(msc, field):
    geom = _geom()
    x, u = _starts(12)
    x[8:10, 0] = AXIS + 10.
    runs = []
    for skip_empty in (False, True):
        init, run = _build(geom, skip_empty, msc=msc, field=field)
        initial = _initial(init, x, u)
        initial = dict(initial, fate=initial["fate"].at[-2:].set(C.PLANE))
        steps, state = run((), KEY, initial, jnp.int32(5000), 4, 4)
        first_steps, partial = run((), KEY, initial, jnp.int32(9), 4, 4)
        later_steps, chunked = run((), KEY, partial, jnp.int32(4991), 4, 4)
        assert int(steps) == int(first_steps + later_steps)
        _compare(state, chunked)
        assert not np.any(np.asarray(state["fate"]) == C.LIVE)
        assert np.asarray(state["cnt"])[:8, 0].sum() >= 8
        assert np.asarray(state["count"]).sum() > 0
        assert np.asarray(state["nstall"]).sum() > 0
        assert np.asarray(state["nslab"]).sum() > 0
        np.testing.assert_array_equal(state["records"]["valid"].sum(axis=1), state["count"])
        for name in ("x", "u", "T", "t", "s", "cnt", "nvac", "nslab", "count"):
            _compare(state[name][-2:], initial[name][-2:])
        runs.append((steps, state))
    _compare(*runs)


# per PR: the default build crosses every shell and ring; with --runslow also the skip_empty A/B
@pytest.mark.pr_params("default")
@pytest.mark.parametrize("modes", [(False,), (False, True)], ids=["default", "skip_empty"])
def test_shell_and_ring_crossings(modes):
    geom = replace(_geom(plane=8000.), ring_t=2., ring_hl=1.)
    x = np.array([[AXIS + 299.5, 0., 7000.], [AXIS + 899.5, 0., 7000.],
                  [AXIS + 297.5, 0., 6900.5], [AXIS + 40., 0., 7000.]])
    u = np.tile([1., 0., 0.], (4, 1))
    runs = []
    for skip_empty in modes:
        init, run = _build(geom, skip_empty, s_max=12.)
        steps, state = run((), KEY, _initial(init, x, u), jnp.int32(2000), 4, 4)
        assert not np.any(np.asarray(state["fate"]) == C.LIVE)
        assert np.all(np.asarray(state["cnt"]).sum(axis=0)[1:] > 0)
        runs.append((steps, state))
    if len(runs) > 1:
        _compare(*runs)
