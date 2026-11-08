"""Raw-state regressions for the opt-in vacuum-only step with explicit cross-JIT float budgets.

Run with JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu.  Public source output omits
slab state and RNG iteration, so these tests deliberately use the raw builder.
"""
from dataclasses import replace
import inspect

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from beamfastsim import cechain as C
from beamfastsim import flash_navigator_jax as FN
from test_flash_target_jax import AXIS, KEY, _geom, _starts


MODES = ({}, {"skip_empty": True}, {"skip_empty": True, "vacuum_fast": True})


def _build(geom, *, msc=False, positron=False, s_max=1000., field=(0., 0., 0.), **mode):
    return FN._build(geom, np.float64, 1., s_max, msc, True, True, False,
                     field, positron=positron, photon_capacity=32, **mode)


def _initial(init, x, u, T=105.):
    n = len(x)
    return init((), jnp.asarray(x, jnp.float64), jnp.asarray(u, jnp.float64),
                jnp.asarray(np.broadcast_to(T, (n,)), jnp.float64), jnp.zeros(n),
                jnp.full(n, -1, jnp.int32), jnp.zeros(n))


def _exact(expected, actual):
    """Include every leaf, dtype, shape, signed zero and scalar loop counter."""
    left, lt = jax.tree_util.tree_flatten_with_path(expected)
    right, rt = jax.tree_util.tree_flatten_with_path(actual)
    assert lt == rt
    differences = []
    for (path, a), (_, b) in zip(left, right):
        a, b = np.asarray(a), np.asarray(b)
        label = jax.tree_util.keystr(path)
        assert a.shape == b.shape and a.dtype == b.dtype, label
        if a.tobytes() != b.tobytes():
            detail = f"{np.count_nonzero(a != b)} numerically unequal elements"
            if a.dtype.kind == 'f' and np.isfinite(a).all() and np.isfinite(b).all():
                delta = np.abs(a - b)
                scale = np.abs(np.spacing(np.maximum(np.abs(a), np.abs(b))))
                ulps = np.divide(delta, scale, out=np.zeros_like(delta), where=scale > 0)
                detail += f", max_abs={np.max(delta):.17g}, max_spacing_units={np.max(ulps):.17g}"
            differences.append(f"raw bytes differ at {label}: {detail}")
    assert not differences, '\n'.join(differences)


def _float_ulps(a, b, limit, name):
    """Bound finite float64 rounding while preserving shape, dtype and zero bytes."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype == np.dtype('float64'), name
    assert np.isfinite(a).all() and np.isfinite(b).all(), name
    zero = (a == 0) | (b == 0)
    _exact({name: a[zero]}, {name: b[zero]})
    try:
        np.testing.assert_array_max_ulp(a, b, maxulp=limit)
    except AssertionError as exc:
        raise AssertionError(f'{name}: {exc}') from exc


def _bucket_roundoff(expected, actual):
    """Only direction rounding between compact and full compiled graphs may differ."""
    assert jax.tree_util.tree_structure(expected) == jax.tree_util.tree_structure(actual)
    _exact(expected[0], actual[0])
    left, tree = jax.tree_util.tree_flatten_with_path(expected[1])
    right, other = jax.tree_util.tree_flatten_with_path(actual[1])
    assert tree == other
    for (path, a), (_, b) in zip(left, right):
        name = '/'.join(str(k.key) for k in path)
        if name == 'u':
            _float_ulps(a, b, 2, name)
        else:
            _exact({name: a}, {name: b})


def _roundoff_only(expected, actual):
    """Tilted-field fusion changes permit bounded float roundoff, not discrete/RNG changes."""
    budgets = {"cont": 1e-15, "slab/cont": 1e-15, "slab/path": 1e-15, "slab_time": 1e-17,
               "records/direction": 1e-13, "records/position": 1e-13, "slab/d": 1e-13,
               "slab/x": 1e-13, "u": 1e-13, "x": 1e-13, "xent": 1e-13}
    assert jax.tree_util.tree_structure(expected) == jax.tree_util.tree_structure(actual)
    _exact(expected[0], actual[0])
    _exact(expected[1]['records']['valid'], actual[1]['records']['valid'])
    left, tree = jax.tree_util.tree_flatten_with_path(expected[1])
    right, other = jax.tree_util.tree_flatten_with_path(actual[1])
    assert tree == other
    for (path, a), (_, b) in zip(left, right):
        name = "/".join(str(k.key) for k in path)
        if name == 't':
            _float_ulps(a, b, 1, name)
            continue
        if name not in budgets:
            _exact({name: a}, {name: b})
            continue
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.isfinite(a).all() and np.isfinite(b).all()
        if name.startswith("records/"):
            invalid = ~np.asarray(expected[1]["records"]["valid"])
            _exact(a[invalid], b[invalid])
        np.testing.assert_allclose(a, b, rtol=0, atol=budgets[name], err_msg=name)


def _run_modes(geom, x, u, steps, *, m_cand=None, m_slab=None, prepare=None,
               modes=MODES, T=105., compare=_exact, **options):
    results = []
    for mode in modes:
        init, run = _build(geom, **options, **mode)
        initial = _initial(init, x, u, T)
        if prepare is not None:
            initial = prepare(initial)
        got = run((), KEY, initial, jnp.int32(steps),
                  len(x) if m_cand is None else m_cand,
                  len(x) if m_slab is None else m_slab)
        results.append((mode, got))
    differences = []
    for mode, got in results[1:]:
        try:
            compare(results[0][1], got)
        except AssertionError as exc:
            differences.append(f"mode={mode}: {exc}")
    assert not differences, "\n".join(differences)
    return results[0][1]


def test_vacuum_fast_is_opt_in_and_separate_cache_key():
    assert inspect.signature(FN._build).parameters["vacuum_fast"].default is False
    geom = _geom(ht=.05)
    default = _build(geom)
    assert default is _build(geom, vacuum_fast=False, skip_empty=False)
    for skip_empty in (False, True):
        fast = _build(geom, vacuum_fast=True, skip_empty=skip_empty)
        assert fast is _build(geom, vacuum_fast=True, skip_empty=skip_empty)
        assert fast is not _build(geom, vacuum_fast=False, skip_empty=skip_empty)
    assert _build(geom, vacuum_fast=True) is not _build(geom, vacuum_fast=True, skip_empty=True)


# per PR: the default against vacuum_fast alone; with --runslow also against the skip_empty modes
@pytest.mark.pr_params("vacuum_fast")
@pytest.mark.parametrize("modes", [({}, {"vacuum_fast": True}), MODES], ids=["vacuum_fast", "skip_empty"])
def test_pure_vacuum_nonzero_field_mixed_terminal_fates_and_lazy_flags(modes):
    geom = replace(_geom(foil_z=(5000.,), ht=.05, plane=5710.),
                   z_up=5690., z_ds=0., r_wall=1000.)
    x, u = _starts(6, z=5700.)
    x[0, 2] = x[5, 2] = 5709.
    x[1, 0] = AXIS + 999.75
    x[2, 2] = 5690.25
    u[1] = u[3] = [1., 0., 0.]
    u[2] = [0., 0., -1.]
    u[5] = [.6, 0., .8]

    def lazy(st):
        # Exercise deferred cleanup on live and terminal lanes with stale slab flags.
        return dict(st, slab=dict(st["slab"], exited=jnp.zeros(6, bool)))

    steps, state = _run_modes(
        geom, x, u, 20, s_max=2., field=(0., 0., 1.), prepare=lazy,
        T=[105., 105., 105., 105., .0001, 105.], modes=modes)
    np.testing.assert_array_equal(state["fate"],
                                  [C.PLANE, C.WALL, C.UPSTREAM, C.LOOPER, C.STOPPED, C.PLANE])
    assert int(steps) == int(state["slab"]["it"]) == 3
    assert np.asarray(state["slab"]["exited"]).all()
    assert not np.asarray(state["inl"]).any()
    assert not np.asarray(state["count"]).any()
    assert not np.asarray(state["nslab"]).any()
    for values in state["records"].values():
        assert not np.asarray(values).any()


@pytest.fixture(scope="module")
def thin_foil():
    # one geometry object for every cap: builds are cached per geometry object, and the cap is a traced argument
    return _geom(foil_z=(5800.,), ht=.05, plane=5805.)


@pytest.mark.slow
@pytest.mark.parametrize("cap", [2, 3, 4], ids=["before-entry", "at-entry", "after-entry"])
def test_caps_before_at_and_after_thin_foil_entry(thin_foil, cap):
    x, u = _starts(4, z=5796.95)
    steps, state = _run_modes(thin_foil, x, u, cap, msc=True)
    assert int(steps) == int(state["slab"]["it"]) == cap
    np.testing.assert_array_equal(state["fate"], C.LIVE)
    if cap < 3:
        assert not np.asarray(state["inl"]).any()
        assert not np.asarray(state["nslab"]).any()
    elif cap == 3:
        assert np.asarray(state["inl"]).all()
        assert not np.asarray(state["slab"]["exited"]).any()
        assert not np.asarray(state["nslab"]).any()
        assert not np.asarray(state["slab"]["path"]).any()
    else:
        np.testing.assert_array_equal(state["nslab"], 1)
        assert np.all(np.asarray(state["slab"]["path"]) > 0.)


@pytest.mark.slow
@pytest.mark.parametrize("positron", [False, True], ids=["electron", "positron"])
def test_msc_foil_ipa_opa_ring_crossings_and_photons(positron):
    geom = _geom(foil_z=(5800., 5803.), ht=.25, plane=8000.)
    geom = replace(geom, beam=replace(geom.beam, ipa_t=.05),
                   opa_t=.05, ring_t=.05, ring_hl=.5)
    x, u = _starts(32, z=5798.75)
    x[24:26] = [AXIS + 299.5, 0., 7000.]
    x[26:28] = [AXIS + 899.5, 0., 7000.]
    x[28:30] = [AXIS + 299.45, 0., 6900.5]
    x[30:] = [AXIS + 10., 0., 7000.]
    u[24:30] = [1., 0., 0.]
    steps, state = _run_modes(geom, x, u, 1000, msc=True, positron=positron,
                              s_max=8., field=(.05, .02, .5), compare=_roundoff_only)
    assert int(steps) == int(state["slab"]["it"])
    assert not np.any(np.asarray(state["fate"]) == C.LIVE)
    assert np.all(np.asarray(state["cnt"]).sum(axis=0) > 0)
    if positron:
        assert np.asarray(state["count"]).sum() > 0
    np.testing.assert_array_equal(state["records"]["valid"].sum(axis=1), state["count"])
    assert not np.asarray(state["inl"])[30:].any()


@pytest.mark.slow
def test_source_fallback_with_smaller_candidate_and_slab_batches():
    geom = _geom(foil_z=(5800.,), ht=.05, plane=5804.)
    x, u = _starts(8, z=5798.95)
    steps, state = _run_modes(
        geom, x, u, 500, m_cand=2, m_slab=1, s_max=8.,
        modes=MODES + ({"vacuum_fast": True},))
    assert int(steps) == int(state["slab"]["it"])
    assert not np.any(np.asarray(state["fate"]) == C.LIVE)
    assert np.asarray(state["nstall"]).sum() > 0
    assert np.all(np.asarray(state["nslab"]) > 0)
    np.testing.assert_array_equal(state["cnt"][:, 0], 1)


@pytest.mark.slow
def test_long_vacuum_500_to_2000_step_chunks_with_short_path_lane():
    geom = _geom(foil_z=(5000.,), ht=.05, plane=8500.)
    geom = replace(geom, beam=replace(geom.beam, ipa_z0=9000., ipa_z1=10000.))
    x, u = _starts(4, z=5700.)
    x[0, 2] = 8499.  # A short path terminates while other lanes keep running.
    u[3] = [.6, 0., .8]
    reference = None
    for mode in MODES:
        init, run = _build(geom, s_max=2000., field=(0., 0., 1.), **mode)
        initial = _initial(init, x, u)
        full = run((), KEY, initial, jnp.int32(2500), 4, 4)
        first_steps, first = run((), KEY, initial, jnp.int32(500), 4, 4)
        rest_steps, final = run((), KEY, first, jnp.int32(2000), 4, 4)
        assert int(first_steps) == 500
        assert int(rest_steps) == 1501
        _exact(full, (first_steps + rest_steps, final))
        assert int(full[0]) == int(final["slab"]["it"]) == 2001
        np.testing.assert_array_equal(final["fate"], [C.PLANE, C.LOOPER, C.LOOPER, C.LOOPER])
        np.testing.assert_array_equal(final["nvac"], [1, 2001, 2001, 2001])
        assert not np.asarray(final["records"]["valid"]).any()
        assert not np.asarray(final["count"]).any()
        if reference is None:
            reference = (full, first_steps, first, rest_steps, final)
        else:
            _exact(reference, (full, first_steps, first, rest_steps, final))
