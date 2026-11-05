"""Frame prerequisites only, not navigation or finite-foil physics certification."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from beamfastsim.flash_foil_frames import (
    build_table, gather, gather_jax, entry_frame, entry_frame_jax,
    to_local, to_global, to_local_jax, to_global_jax, validate_basis,
)


def geometry(**kw):
    args = dict(axis_x=-3904., foil_z=np.array([10., 20., 30.]),
                foil_rin=np.array([1., 2., 0.]), foil_rout=np.array([4., 5., 6.]),
                foil_ht=np.array([.5, .5, 1.]))
    args.update(kw)
    return SimpleNamespace(**args)


def test_unique_identity_material_type_copy_and_strong_immutability():
    g = geometry()
    t = build_table(SimpleNamespace(beam=g))
    g.foil_z[:] = 999
    np.testing.assert_array_equal(t.foil_z, [10, 20, 30])
    np.testing.assert_array_equal(t.foil_id, [0, 1, 2])
    np.testing.assert_array_equal(t.material_type, [0, 0, 1])
    np.testing.assert_array_equal(t.material_thickness, [1, 2])
    for a in vars(t).values():
        assert not callable(a)
        if isinstance(a, np.ndarray):
            assert not a.flags.writeable
            with pytest.raises(ValueError):
                a.flat[0] = 0
            with pytest.raises(ValueError):
                a.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        t.axis_x = 0


@pytest.mark.parametrize('kw', [
    dict(axis_x=np.nan), dict(axis_x=[0]), dict(foil_z=[]),
    dict(foil_z=[1, np.inf, 3]), dict(foil_z=[[1, 2, 3]]),
    dict(foil_z=[1, 2]), dict(foil_rin=[-1, 0, 0]),
    dict(foil_rout=[1, 1, 1]), dict(foil_ht=[0, 1, 1]),
    dict(foil_z=[10, 10, 30]), dict(foil_z=[10, 11, 30]),
    dict(foil_z=[1.e20, 2.e20, 3.e20]), dict(foil_z=['a', 'b', 'c']),
])
def test_table_validation(kw):
    with pytest.raises(ValueError):
        build_table(geometry(**kw))


def test_unsorted_identity_and_reordered_repeated_gather_sentinel():
    t = build_table(geometry(foil_z=[30., 10., 20.]))
    r = gather(t, np.array([[2, 0, 2], [1, -1, 3]]))
    np.testing.assert_array_equal(r['foil_id'], [[2, 0, 2], [1, -1, -1]])
    np.testing.assert_array_equal(r['foil_z'][0], [20, 30, 20])
    np.testing.assert_array_equal(r['material_type'], [[1, 0, 1], [0, -1, -1]])
    assert np.all(np.isnan(r['foil_rout'][1, 1:]))
    assert np.all(r['unresolved'][1, 1:])
    assert not np.any(r['valid'][1, 1:])
    for ids in ([1.0], [True]):
        with pytest.raises(ValueError):
            gather(t, ids)
    huge = gather(t, np.array([2**64 - 1], dtype=np.uint64))
    assert not huge['valid'][0] and np.isnan(huge['foil_z'][0])
    assert huge['foil_id'][0] == -1


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('sg', [-1, 1])
def test_sided_frame_offset_actual_depth_and_exact_global_copy(dtype, sg):
    t = build_table(geometry())
    x = np.array([[-3902., -0., 9.5], [-3900., 0, 10.25],
                  [-3903., 0, 10.5], [-3902., 0, 10]], dtype=dtype)
    u = np.array([.6, 0., sg * .8], dtype=dtype)
    B = np.array([.3, -.7, 1.2], dtype=dtype)
    f = entry_frame(t, 0, x, u, B)
    assert np.all(f['valid']) and not np.any(f['unresolved'])
    assert f['global_x'].dtype == dtype
    assert f['global_x'].tobytes() == x.tobytes()  # includes negative zero
    assert f['local_u'].dtype == dtype
    np.testing.assert_array_equal(f['global_u'], np.broadcast_to(u, x.shape))
    expected = sg * (x[:, 2] - (10 - sg * .5))
    np.testing.assert_array_equal(f['depth'], expected)
    np.testing.assert_array_equal(f['local_x'][:, :2], 0)
    np.testing.assert_array_equal(f['local_x'][:, 2], expected)
    np.testing.assert_array_equal(f['anchor'][:, :2], x[:, :2])
    np.testing.assert_array_equal(f['anchor'][:, 2], 10 - sg * .5)
    np.testing.assert_array_equal(f['interior'], [False, False, False, True])
    np.testing.assert_array_equal(f['boundary'], [True, True, True, False])
    validate_basis(f['basis'])
    np.testing.assert_allclose(np.linalg.det(f['basis']), 1)
    # Rim/hole entries retain their actual z and nonzero depth; no ray projection.
    assert f['depth'][1] > 0 and f['depth'][2] >= 0
    np.testing.assert_allclose(to_local(f, x, point=True), f['local_x'])
    np.testing.assert_allclose(to_global(f, f['local_x'], point=True), x,
                               atol=10*np.finfo(dtype).eps, rtol=10*np.finfo(dtype).eps)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_refusal_without_tolerance_normalization_or_last_row_alias(dtype):
    t = build_table(geometry(axis_x=0))
    x = np.array([[2, 0, 10]] * 8, dtype=dtype)
    x[2, 0] = np.nextafter(dtype(4), dtype(np.inf))
    x[3, 2] = np.nextafter(dtype(10.5), dtype(np.inf))
    x[4, 0] = .9
    x[7, 0] = np.nan
    u = np.tile(np.array([0, 0, 1], dtype=dtype), (8, 1))
    u[5] = [1, 0, 0]
    u[6] = [0, 0, 2]
    f = entry_frame(t, [-1, 3, 0, 0, 0, 0, 0, 0], x, u, [0, 0, 1])
    assert np.all(f['unresolved']) and not np.any(f['valid'])
    for key in ('basis', 'anchor', 'depth', 'local_x', 'local_u', 'local_B'):
        assert np.all(np.isnan(f[key]))
    np.testing.assert_array_equal(f['global_x'], x)
    np.testing.assert_array_equal(f['global_u'], u)
    assert np.all(np.isnan(to_global(f, [0, 0, 1])))
    # No hole means the axis is interior, not an artificial boundary.
    assert entry_frame(t, 2, [0., 0, 30], [0., 0, 1], [0., 0, 0])['interior']


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('sg', [-1, 1])
@pytest.mark.parametrize('q', [-1, 0, 1])
def test_field_charge_covariance_and_saved_postscatter_frame(dtype, sg, q):
    t = build_table(geometry())
    x = np.array([-3902., 0, 10.125], dtype=dtype)
    u = np.array([.6, 0., sg * .8], dtype=dtype)
    B = np.array([.3, -.7, 1.2], dtype=dtype)
    f = entry_frame(t, 0, x, u, B)
    # Proper rotations transform the Lorentz cross product without changing q.
    np.testing.assert_allclose(to_local(f, q*np.cross(u, B)),
                               q*np.cross(f['local_u'], f['local_B']),
                               atol=8*np.finfo(dtype).eps, rtol=8*np.finfo(dtype).eps)
    scattered = np.array([0., 0., -1.], dtype=dtype)
    np.testing.assert_array_equal(to_global(f, scattered), [0, 0, -sg])
    displacement = np.array([.12345, -.23456, .34567], dtype=dtype)
    p = to_global(f, displacement, point=True)
    np.testing.assert_allclose(to_local(f, p, point=True), displacement,
                               rtol=0, atol=8*np.finfo(dtype).eps*3904)
    # Position subtraction at large offsets can lose low bits. No exact claim.
    if dtype == np.float32:
        assert not np.array_equal(to_local(f, p, point=True), displacement)


def test_basis_rejects_reflection_scaling_and_invalid_shape():
    for b in (np.diag([1., 1., -1.]), 2*np.eye(3), np.full((3, 3), np.nan), np.ones(3)):
        with pytest.raises(ValueError):
            validate_basis(b)
    t = build_table(geometry())
    f = entry_frame(t, 0, [-3902., 0, 10], [0., 0, 1], [0., 0, 0])
    f['basis'] = np.diag([1., 1., -1.])
    assert np.all(np.isnan(to_local(f, [1., 2., 3.])))


def test_import_does_not_load_geometry_files_or_jax():
    subprocess.run([sys.executable, '-c',
        'import sys; import beamfastsim.flash_foil_frames; '
        'assert "jax" not in sys.modules; '
        'assert "beamfastsim.geometry" not in sys.modules'], check=True)


@pytest.mark.parametrize('x64', [False, True])
def test_jax_jit_gather_frames_transforms_match_numpy(x64):
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        t = build_table(geometry())
        ids = np.array([1, 0, 1, -1, 3], dtype=np.int32)
        x = np.array([[-3901, 0, 20], [-3902, 0, 10], [-3900, 0, 20.25],
                      [0, 0, 0], [0, 0, 0]], dtype=dtype)
        u = np.array([[0, 0, -1], [0, 0, 1], [.6, 0, -.8],
                      [0, 0, 1], [0, 0, 1]], dtype=dtype)
        B = np.array([.3, -.2, 1.], dtype=dtype)
        a = entry_frame(t, ids, x, u, B)
        b = jax.jit(lambda i, p, v, field: entry_frame_jax(t, i, p, v, field))(
            jnp.asarray(ids), jnp.asarray(x), jnp.asarray(u), jnp.asarray(B))
        for key in a:
            np.testing.assert_allclose(np.asarray(b[key]), a[key], equal_nan=True,
                                       atol=8*np.finfo(dtype).eps, rtol=8*np.finfo(dtype).eps)
        r = jax.jit(lambda i: gather_jax(t, i))(jnp.asarray(ids))
        for key, v in gather(t, ids).items():
            np.testing.assert_allclose(r[key], v, equal_nan=True)
        local = jax.jit(lambda f, p: to_local_jax(f, p, point=True))(b, jnp.asarray(x))
        back = jax.jit(lambda f, p: to_global_jax(f, p, point=True))(b, local)
        np.testing.assert_allclose(back[:3], x[:3], atol=8*np.finfo(dtype).eps*3904)
        assert np.all(np.isnan(back[3:]))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_broadcast_lanes_and_nonfinite_field_refusal(dtype):
    t = build_table(geometry())
    ids = np.array([[0, 0], [1, 1]])
    x = np.array([[[-3901., 0, 10]], [[-3901., 0, 20]]], dtype=dtype)
    u = np.array([[0, 0, 1], [0, 0, -1]], dtype=dtype)
    B = np.array([0, 0, 1], dtype=dtype)
    f = entry_frame(t, ids, x, u, B)
    assert f['basis'].shape == (2, 2, 3, 3)
    assert np.all(f['valid'])
    np.testing.assert_array_equal(f['depth'], .5)
    np.testing.assert_allclose(to_global(f, f['local_B']),
                               np.broadcast_to(B, (2, 2, 3)))
    for bad_B in ([np.inf, 0, 0], [0, np.nan, 0]):
        bad = entry_frame(t, ids, x, u, np.array(bad_B, dtype=dtype))
        assert np.all(bad['unresolved']) and not np.any(bad['valid'])
    for bad_u in ([0, 0, 0], [0, 0, np.nan], [0, 0, np.inf]):
        assert entry_frame(t, 0, x[0, 0], np.array(bad_u, dtype=dtype), B)['unresolved']
    with pytest.raises(ValueError):
        entry_frame(t, 0, [0, 0], [0, 0, 1], B)
    with pytest.raises(ValueError):
        entry_frame(t, 0., x[0, 0], u[0], B)
