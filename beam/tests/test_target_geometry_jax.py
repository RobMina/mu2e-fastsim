"""Shared target helpers: ownership, frame conventions, map precedence and gathering."""
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from beamfastsim import cechain as C
from beamfastsim import cechain_jax as CE
from beamfastsim import target_geometry_jax as TG
from beamfastsim import fieldmap as FM
from beamfastsim import geometry as G
from mu2efastsim import eslab_jax as SJ


@pytest.fixture
def geom():
    beam = G.BeamGeometry(-3904., np.array([5800., 5822., 5844.]), np.array([10., 20., 30.]),
                          np.array([50., 75., 100.]), np.array([.05, .10, .05]),
                          4204.6, 25.4, 235., 525., 300., .5, 6900., 7900., 900., 4280., 8530.)
    return replace(C.CEGeometry(beam, 900., 900., 1., z_plane=8000.), ring_t=2., ring_hl=1.)


def kernels(geom, dtype=np.float64, charge=-1., field=(.2, .3, 1.), layer_field=True, curvature=True):
    return TG.build(geom, dtype, TG.layer_types(geom)[1], charge=charge, const_field=field,
                    layer_field=layer_field, shell_curvature=curvature)


def test_ce_uses_shared_helpers():
    for public, shared in [('layer_types', 'layer_types'), ('device_maps', 'device_maps'),
                           ('_field', 'field'), ('_mom', 'momentum'), ('_beta', 'beta'),
                           ('_dot', 'dot'), ('gather', 'gather')]:
        assert getattr(CE, public) is getattr(TG, shared)
    assert CE.NCOUNT == TG.NCOUNT == 4


def test_foil_identity_is_not_a_material_type(geom):
    specs, lengths, ipa, opa, ring, counts = TG.layer_types(geom)
    k = kernels(geom)
    np.testing.assert_array_equal(lengths, [.1, .2])
    np.testing.assert_array_equal(k.ftype, [0, 1, 0])
    np.testing.assert_array_equal(k.frin, [10., 20., 30.])
    np.testing.assert_array_equal(k.frout, [50., 75., 100.])
    np.testing.assert_allclose(k.face_up, geom.beam.foil_z - geom.beam.foil_ht)
    assert (ipa, opa, ring) == (2, 3, 4)
    np.testing.assert_array_equal(counts, [0, 0, 1, 2, 3])
    assert [row[0].name for row in specs[:2]] == [G.STOPPING_TARGET_AL.name] * 2
    assert len(TG.layer_types(replace(geom, ring_t=0.))[0]) == 4
    assert TG.layer_types(replace(geom, ring_t=0.))[4] == -1


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('charge', [-1., 1.])
def test_frames_preserve_foil_origin_shell_charge_and_local_vectors(geom, dtype, charge):
    k = kernels(geom, dtype, charge)
    x = jnp.array([[-3864., 0., 5800.], [-3864., 0., 5844.], [-3604., 0., 7000.],
                   [-3603.5, 0., 7000.]], dtype)
    u = jnp.array([[0., 0., 1.], [0., 0., -1.], [.8, .6, 0.], [-.8, .6, 0.]], dtype)
    T = jnp.full(4, 100., dtype)
    e1, e2, en, entry, local, B = jax.jit(k.frames)((), x, u, T, jnp.array([0, 0, 2, 2]),
                                                 jnp.array([0, 2, -1, -1]))
    tol = 1e-6 if dtype == np.float32 else 1e-12
    basis = np.stack((e1, e2, en), axis=2)
    np.testing.assert_allclose(np.einsum('nji,njk->nik', basis, basis), np.broadcast_to(np.eye(3), (4, 3, 3)), atol=tol)
    np.testing.assert_allclose(np.linalg.det(basis), 1., atol=tol)
    np.testing.assert_allclose(np.einsum('nij,nj->ni', basis, local), u, atol=tol)
    np.testing.assert_allclose(entry[:2, 2], np.asarray(k.face_up)[[0, 2]] +
                                [0., 2 * geom.beam.foil_ht[2]], atol=.001 if dtype == np.float32 else 1e-12)
    np.testing.assert_array_equal(entry[2:], x[2:])
    correction = np.asarray(TG.momentum(T))[2:] * .6 / (SJ.K_FIELD * np.array([300., 300.5]))
    world_B = np.einsum('nij,nj->ni', basis, B)
    np.testing.assert_allclose(world_B[:2], np.tile([.2, .3, 1.], (2, 1)), atol=tol)
    np.testing.assert_allclose(world_B[2:, 2], 1. + charge * correction, rtol=tol, atol=tol)


def test_layer_field_and_shell_curvature_are_independent_static_options(geom):
    x, u = jnp.array([[-3604., 0., 7000.]]), jnp.array([[.8, .6, 0.]])
    args = ((), x, u, jnp.array([100.]), jnp.array([2]), jnp.array([-1]))
    no_field = kernels(geom, layer_field=False, curvature=False).frames(*args)[-1]
    np.testing.assert_array_equal(no_field, 0.)
    correction = kernels(geom, layer_field=False, curvature=True).frames(*args)[-1]
    assert np.linalg.norm(correction) > 0.


def test_first_map_precedence_and_outside_zero():
    def field_map(lo, value):
        return FM.FieldMap(jnp.array(lo), jnp.ones(3), jnp.array([3, 3, 3]),
                           jnp.broadcast_to(jnp.array(value), (3, 3, 3, 3)))
    maps = (field_map([0., 0., 0.], [1., 2., 3.]), field_map([1., 0., 0.], [4., 5., 6.]))
    x = jnp.array([[.5, .5, .5], [1.5, .5, .5], [2., .5, .5], [3., .5, .5]])
    got = jax.jit(TG.field)(maps, None, x)
    np.testing.assert_array_equal(got, [[1., 2., 3.], [1., 2., 3.], [4., 5., 6.], [0., 0., 0.]])
    np.testing.assert_array_equal(TG.field((), [1., 2., 3.], x), np.tile([1., 2., 3.], (4, 1)))


@pytest.mark.parametrize('mask,size,indices,valid,chosen', [
    ([False, True, False, True], 3, [1, 3, 4], [True, True, False], [False, True, False, True]),
    ([False, True, True, True], 2, [1, 2], [True, True], [False, True, True, False]),
    ([False, False], 1, [2], [False], [False, False]),
])
def test_gather_shape_and_slot_order(mask, size, indices, valid, chosen):
    idx, clipped, ok, selected = jax.jit(lambda mask: TG.gather(mask, size))(jnp.array(mask))
    np.testing.assert_array_equal(idx, indices)
    np.testing.assert_array_equal(clipped, np.minimum(indices, len(mask) - 1))
    np.testing.assert_array_equal(ok, valid)
    np.testing.assert_array_equal(selected, chosen)


def test_bracketed_shell_solver_and_opa_radius(geom):
    k = kernels(geom)
    x, u = jnp.array([[-3605., 0., 7000.]]), jnp.array([[1., 0., 0.]])
    solve = jax.jit(lambda x, u: k.bisect(x, u, jnp.array([100.]), jnp.zeros((1, 3)),
                                         jnp.array([2.]), lambda z: jnp.full_like(z, 300.)))
    np.testing.assert_allclose(solve(x, u), [1.], atol=2e-12)
    np.testing.assert_array_equal(k.opa_rin(jnp.array([4000., 5000., 9000.])), [900., 900., 900.])


def test_ce_capped_gather_keeps_chunked_state_and_rng_iteration(geom):
    n = 6
    init, run = CE._build(geom, np.float64, 1., 100., False, True, True, None, False, (0., 0., 0.))
    x = jnp.tile(jnp.array([-3864., 0., 5799.5]), (n, 1))
    u = jnp.tile(jnp.array([0., 0., 1.]), (n, 1))
    state = init((), x, u, jnp.full(n, 100.), jnp.zeros(n), jnp.full(n, -1, jnp.int32), jnp.zeros(n))
    key = jax.random.PRNGKey(33)
    total, full = run((), key, state, jnp.int32(8), 2, 2)
    first, mid = run((), key, state, jnp.int32(2), 2, 2)
    last, chunked = run((), key, mid, jnp.int32(6), 2, 2)
    assert int(total) == int(first + last) == 8
    assert int(full['slab']['it']) == 8
    assert np.sum(full['nstall']) > 0
    assert np.sum(full['nslab']) > 0
    for a, b in zip(jax.tree_util.tree_leaves(full), jax.tree_util.tree_leaves(chunked), strict=True):
        np.testing.assert_array_equal(a, b)
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_conical_opa_interpolation_and_clipping(geom, dtype):
    geom = replace(geom, opa_rin0=400., opa_rin1=700.)
    k = kernels(geom, dtype)
    za, zb = geom.beam.opa_z0, geom.beam.opa_z1
    z = jnp.asarray([za - 1., za, .5 * (za + zb), zb, zb + 1.], dtype)
    got = k.opa_rin(z)
    assert got.dtype == dtype
    np.testing.assert_allclose(got, [400., 400., 550., 700., 700.], rtol=1e-7)
