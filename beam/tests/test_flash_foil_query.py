"""Synthetic actual-leg queries only; no source transport or detector data."""
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from beamfastsim import flash_foil_frames as F
from beamfastsim import flash_foil_query as Q
from beamfastsim import flash_foil_helix as H


def _table():
    return F.build_table(SimpleNamespace(axis_x=0., foil_z=np.array([0., 30.]), foil_ht=np.array([10., 10.]),
                                         foil_rin=np.array([1., 2.]), foil_rout=np.array([3., 4.])))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sg", [-1, 1])
def test_reordered_equal_thickness_query_keeps_foil_frame_and_depth(dtype, sg):
    table = _table()
    ids = np.array([1, 0, 1])
    x = np.array([[3., 0., 30.], [2., 0., 0.], [3., 0., 30.]], dtype)
    u = np.array([.6, 0., .8*sg], dtype)
    frame = F.entry_frame(table, ids, x, u, np.zeros(3, dtype))
    query = Q.make_query(table, ids, frame)
    distance, bad = jax.jit(query)(frame["local_x"], frame["local_u"], np.full(3, 1., dtype), -1., None,
                                   np.full(3, 2., dtype))
    np.testing.assert_array_equal(bad, False)
    np.testing.assert_allclose(distance, 1/.6, atol=2.e-6 if dtype == np.float32 else 2.e-12, rtol=0)
    np.testing.assert_array_equal(frame["depth"], 10.)


def test_exact_rim_entry_receipt_excludes_zero_without_face_projection():
    table = _table()
    u = np.array([-.6, 0., .8])
    frame = F.entry_frame(table, 0, np.array([3., 0., .5]), u, [0., 0., 0.])
    assert frame["depth"] == 10.5
    query = Q.make_query(table, 0, frame, entry_surface=4)
    distance, bad = jax.jit(query)(frame["local_x"], frame["local_u"], 1., -1., None, 5.)
    assert not bad
    np.testing.assert_allclose(distance, 2/.6, atol=1.e-12, rtol=0)
    distance, bad = jax.jit(Q.make_query(table, 0, frame))(frame["local_x"], frame["local_u"], 1., -1., None, 5.)
    assert bad and np.isinf(distance)


def test_query_only_consumes_token_at_retained_entry_not_later_legs():
    table = _table()
    frame = F.entry_frame(table, 0, [3., 0., 0.], [-.6, 0., .8], [0., 0., 0.])
    query = Q.make_query(table, 0, frame, entry_surface=4)
    # A later actual leg is inside; do not apply an old surface token there.
    xx = np.asarray(frame["local_x"]).copy()
    xx[0] = -.5
    distance, bad = jax.jit(query)(xx, frame["local_u"], 1., -1., None, 5.)
    assert not bad
    np.testing.assert_allclose(distance, 1.5/.6, atol=1.e-12, rtol=0)


def test_invalid_identity_and_frame_never_become_nohit():
    table = _table()
    frame = F.entry_frame(table, 0, [2., 0., 0.], [.6, 0., .8], [0., 0., 0.])
    for ids in (-1, 9, 1):
        query = Q.make_query(table, ids, frame)
        distance, bad = jax.jit(query)(frame["local_x"], frame["local_u"], 1., -1., None, .01)
        assert bad and np.isinf(distance)
    invalid = dict(frame, valid=np.array(False))
    distance, bad = jax.jit(Q.make_query(table, 0, invalid))(frame["local_x"], frame["local_u"], 1., -1., None, .01)
    assert bad and np.isinf(distance)


@pytest.mark.parametrize("sg", [-1, 1])
@pytest.mark.parametrize("q", [-1, 1])
def test_curved_local_field_and_charge_agree_with_global_query(sg, q):
    table = _table()
    x, u, B = np.array([2., 0., 0.]), np.array([.6, 0., .8*sg]), np.array([.1, .2, .7])
    frame = F.entry_frame(table, 0, x, u, B)
    query = Q.make_query(table, 0, frame)
    distance, bad = jax.jit(query)(frame["local_x"], frame["local_u"], 1., q, frame["local_B"], 2.)
    global_result = H.first_transition(x, u, 1., q, B, 1., 3., -10., 10., exiting=True, s_max=2.)
    np.testing.assert_array_equal(bad, global_result["unresolved"])
    np.testing.assert_allclose(distance, global_result["distance"], atol=1.e-12, rtol=0)


def test_dynamic_gather_frame_ids_remain_bound_under_jit():
    table = _table()
    ids = np.array([1, 0, 1])
    x = np.array([[3., 0., 30.], [2., 0., 0.], [3., 0., 30.]])
    frame = F.entry_frame(table, ids, x, [.6, 0., .8], [0., 0., 0.])
    order = np.array([2, 0, 1, 0])
    gathered = {name: np.asarray(value)[order] for name, value in frame.items()}
    fn = jax.jit(lambda indices, saved: Q.leg_query_jax(table, indices, saved, saved["local_x"],
        saved["local_u"], jnp.ones(4), -1., None, jnp.full(4, 2.)))
    distance, bad = fn(ids[order], gathered)
    np.testing.assert_array_equal(bad, False)
    np.testing.assert_allclose(distance, 1/.6, atol=1.e-12, rtol=0)
    mismatch = ids[order].copy()
    mismatch[0] = 0
    distance, bad = fn(mismatch, gathered)
    assert bad[0] and np.isinf(distance[0])
    assert not np.asarray(bad)[1:].any()


@pytest.mark.parametrize("kwargs", [dict(entry_surface=.5), dict(entry_surface=5), dict(entry_surface=True),
                                      dict(tolerance=-1), dict(max_intervals=0)])
def test_query_host_configuration_validation(kwargs):
    table = _table()
    frame = F.entry_frame(table, 0, [2., 0., 0.], [.6, 0., .8], [0., 0., 0.])
    with pytest.raises(ValueError):
        Q.make_query(table, 0, frame, **kwargs)
