"""Synthetic constant-field geometry only: analytic checks plus diagnostic oracle."""

import numpy as np
import pytest

from beamfastsim.flash_foil_helix import (
    K_FIELD, NONE, ZLO, ZHI, RIN, ROUT, first_transition,
    first_transition_jax, validate_inputs,
)
from beamfastsim import flash_foil_boundary as ray


def trace(x, u, **kw):
    args = dict(p=K_FIELD, q=1, B=[0., 0., 1.], rin=1., rout=3.,
                zlo=-2., zhi=2., s_max=10.)
    args.update(kw)
    return first_transition(np.asarray(x, float), np.asarray(u, float), **args)


def hit(r, s, kind, atol=2.e-10):
    assert np.all(r['valid']) and not np.any(r['unresolved'])
    np.testing.assert_allclose(r['distance'], s, atol=atol, rtol=0)
    np.testing.assert_array_equal(r['kind'], np.broadcast_to(kind, r['kind'].shape))


def refused(r):
    assert np.all(r['unresolved']) and not np.any(r['valid'])
    assert np.all(np.isinf(r['distance'])) and np.all(r['kind'] == NONE)


def miss(r):
    assert not np.any(r['unresolved']) and not np.any(r['valid'])
    assert np.all(np.isinf(r['distance'])) and np.all(r['kind'] == NONE)


@pytest.mark.parametrize('q', [-1, 1])
def test_analytic_circle_and_same_endpoint_excursion(q):
    # r(s)^2 = 8 - 8*cos(s): a unit circle centred at (0,-q).
    expected = np.arccos(1-1/8)
    hit(trace([0, 0, 0], [1, 0, 0], p=2*K_FIELD, q=q,
              s_max=4*np.pi), 2*expected, RIN)
    # The endpoint is the origin again, but the annulus has been entered.
    hit(trace([0, 0, 0], [1, 0, 0], s_max=2*np.pi), np.pi/3, RIN)


@pytest.mark.parametrize('q', [-1, 1])
def test_analytic_circle_material_exit_reentry(q):
    # Centre (2,-q), R=1: radius first increases from 2 to 3.
    # r^2=6+4*sin(s)-2*cos(s).
    root = np.arcsin(3/np.sqrt(20))+np.arctan(.5)
    hit(trace([2, 0, 0], [1, 0, 0], q=q, exiting=True,
              s_max=2*np.pi), root, ROUT)


@pytest.mark.parametrize('q', [-1, 1])
def test_tilted_field_analytic_plane(q):
    # B along y, u along x: z=q*(1-cos(s)), x=2+sin(s).
    if q == 1:
        hit(trace([2, 0, 0], [1, 0, 0], B=[0, 1, 0], rin=0,
                  rout=20, zhi=.5, exiting=True), np.pi/3, ZHI)
    else:
        hit(trace([2, 0, 0], [1, 0, 0], B=[0, 1, 0], rin=0,
                  rout=20, zlo=-.5, q=q, exiting=True), np.pi/3, ZLO)


@pytest.mark.parametrize('x,u,ex,kind', [
    ([3, 0, 0], [-1, 0, 0], False, ROUT),
    ([3, 0, 0], [1, 0, 0], True, ROUT),
    ([1, 0, 0], [1, 0, 0], False, RIN),
    ([1, 0, 0], [-1, 0, 0], True, RIN),
    ([2, 0, -2], [0, 0, 1], False, ZLO),
    ([2, 0, -2], [0, 0, -1], True, ZLO),
    ([2, 0, 2], [0, 0, -1], False, ZHI),
    ([2, 0, 2], [0, 0, 1], True, ZHI),
])
def test_zero_distance_surface_ownership_without_skip(x, u, ex, kind):
    hit(trace(x, u, B=[.3, .2, .9], exiting=ex, s_max=0), 0, kind)
    refused(trace(x, u, B=[.3, .2, .9], exiting=not ex, s_max=0))


def test_cap_inclusive_and_nohit():
    # A field parallel to direction has no curvature, but remains a helix lane.
    hit(trace([2, 0, -4], [0, 0, 1], s_max=2), 2, ZLO)
    miss(trace([2, 0, -4], [0, 0, 1], s_max=1.9))
    refused(trace([2, 0, -4], [0, 0, 1], s_max=2-1.e-14))
    miss(trace([10, 0, 0], [1, 0, 0], s_max=2*np.pi))
    miss(trace([2, 0, 0], [0, 0, 1], exiting=True, s_max=0))


def test_refusal_tangent_corner_close_surface_and_budget():
    refused(trace([3, 0, 0], [0, 1, 0]))
    refused(trace([3, 0, -2], [-1, 0, 0]))
    refused(trace([3+1.e-14, 0, 0], [-1, 0, 0]))
    refused(trace([0, 0, 0], [1, 0, 0], max_intervals=1))
    refused(trace([0, 0, 0], [1, 0, 0], p=1.e-9, s_max=100,
                  max_intervals=3))
    # Circle tangent to r=2 at s=pi; endpoints both in the hole.
    refused(trace([0, 0, 0], [1, 0, 0], rin=2, s_max=2*np.pi))
    # Refuse a resolvable-in-principle thin material shell below requested scale.
    refused(trace([0, 0, 0], [1, 0, 0], rin=1, rout=1.0001, tolerance=.01))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_dtype_and_vectorized_target_broadcast(dtype):
    x = np.array([[0, 0, 0], [10, 0, 0]], dtype=dtype)[:, None, :]
    u = np.array([1, 0, 0], dtype=dtype)
    r = first_transition(x, u, K_FIELD, np.array([1, -1]), [0, 0, 1],
                         np.array([1., 1.5]), 3., -2., 2., s_max=2*np.pi)
    assert r['distance'].dtype == dtype
    hit({k: v[0] for k, v in r.items()}, [np.pi/3, 2*np.arcsin(.75)],
        RIN, atol=5.e-5 if dtype == np.float32 else 2.e-10)
    miss({k: v[1] for k, v in r.items()})


@pytest.mark.parametrize('B,q', [([0, 0, 0], 1), ([.5, 2, -.7], 0)])
def test_exact_ray_limit(B, q):
    x = np.array([[5., 0, 0], [2, 0, -4], [0, 0, 0], [-4, 3, 0]])
    u = np.array([[-1., 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0]])
    a = first_transition(x, u, 105., q, B, 1., 3., -2., 2., s_max=20.)
    b = ray.first_transition(x, u, 1., 3., -2., 2., s_max=20.)
    for key in b:
        np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize('kw', [dict(p=0), dict(p=-1), dict(p=np.inf),
    dict(q=.5), dict(B=[np.nan, 0, 0]), dict(B=[1, 2]), dict(s_max=-1),
    dict(max_intervals=0), dict(max_intervals=2.5), dict(max_intervals=True),
    dict(tolerance=-1), dict(rin=-1), dict(exiting=1)])
def test_validation(kw):
    with pytest.raises(ValueError):
        trace([0, 0, 0], [1, 0, 0], **kw)


@pytest.mark.parametrize('u', [[0, 0, 0], [2, 0, 0], [1, 1, 0], [np.nan, 0, 0]])
def test_unit_direction_validation(u):
    with pytest.raises(ValueError):
        trace([0, 0, 0], u)


def independent_orbit(x, u, p, q, B, s):
    """Independent Rodrigues rotation integrated using circle centre/radius."""
    bm = np.linalg.norm(B, axis=-1)
    bh = B/bm[:, None]
    par = np.sum(u*bh, axis=-1)[:, None]*bh
    perp = u-par
    k = K_FIELD*q*bm/p
    offset = np.cross(perp, bh)/k[:, None]
    centre = x+offset
    t = k[:, None]*s[None, :]
    return (centre[:, None, :]+par[:, None, :]*s[None, :, None]
            -offset[:, None, :]*np.cos(t)[..., None]
            +perp[:, None, :]*np.sin(t)[..., None]/k[:, None, None])


def test_seeded_tilted_fields_dense_diagnostic_oracle():
    rng = np.random.default_rng(7231)
    n = 96
    x = rng.uniform(-4, 4, (n, 3))
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=-1)[:, None]
    B = rng.normal(size=(n, 3)); B /= np.linalg.norm(B, axis=-1)[:, None]
    p = rng.uniform(.2, 1.5, n)
    q = rng.choice([-1, 1], n)
    radial = np.hypot(x[:, 0], x[:, 1])
    ex = (radial > 1) & (radial < 3) & (np.abs(x[:, 2]) < 2)
    r = first_transition(x, u, p, q, B, 1., 3., -2., 2., s_max=8., exiting=ex)
    s = np.linspace(0, 8, 16001)
    pos = independent_orbit(x, u, p, q, B, s)
    radius = np.hypot(pos[..., 0], pos[..., 1])
    inside = (radius > 1) & (radius < 3) & (np.abs(pos[..., 2]) < 2)
    changes = inside[:, 1:] != inside[:, :-1]
    found = np.any(changes, -1)
    first = np.argmax(changes, -1)
    accepted = ~r['unresolved']
    assert np.count_nonzero(accepted) >= 85
    np.testing.assert_array_equal(r['valid'][accepted], found[accepted])
    chosen = r['valid']
    assert np.all(r['distance'][chosen] >= s[first[chosen]]-2.e-10)
    assert np.all(r['distance'][chosen] <= s[first[chosen]+1]+2.e-10)
    # The dense grid is diagnostic only; it supplies no certification of misses.


def test_jax_optional_jit_parity_and_refusal():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    x = np.array([[0, 0, 0], [10, 0, 0], [3, 0, 0], [2, 0, -4]], np.float32)
    u = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    B = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [.3, .4, .5]], np.float32)
    kw = dict(p=K_FIELD, q=1, B=B, rin=1., rout=3., zlo=-2., zhi=2., s_max=7.)
    validate_inputs(x, u, **kw)
    ref = first_transition(x, u, **kw)
    fn = jax.jit(first_transition_jax, static_argnames=('max_intervals',))
    result = fn(x, u, **kw)
    for key in ('valid', 'unresolved', 'kind'):
        np.testing.assert_array_equal(result[key], ref[key])
    np.testing.assert_allclose(result['distance'], ref['distance'], atol=2.e-5, rtol=0)
    assert result['distance'].dtype == jnp.float32
    refused(fn(np.array([0, 0, 0], np.float32), np.array([2, 0, 0], np.float32), **kw))
    try:
        from jax import enable_x64
    except ImportError:
        from jax.experimental import enable_x64
    with enable_x64():
        hit(fn(jnp.array([0., 0, 0], dtype=jnp.float64),
               jnp.array([1., 0, 0], dtype=jnp.float64), K_FIELD, 1,
               jnp.array([0., 1., 0.]), 0., 20., -2., .5,
               s_max=3., exiting=True), np.pi/3, ZHI)


def test_curved_cap_and_interior_corner_are_not_skipped():
    # x=2+sin(s), z=1-cos(s); at pi/2 both the plane and outer rim meet.
    refused(trace([2, 0, 0], [1, 0, 0], B=[0, 1, 0],
                  rin=0, rout=3, zhi=1, exiting=True, s_max=3))
    # An exact curved endpoint is inclusive; a nearby ambiguous cap is refused.
    cap = np.pi/2
    hit(trace([2, 0, 0], [1, 0, 0], B=[0, 1, 0], rin=0,
              rout=20, zhi=1-np.cos(cap), exiting=True, s_max=cap), cap, ZHI)
    refused(trace([2, 0, 0], [1, 0, 0], B=[0, 1, 0], rin=0,
                  rout=20, zhi=1, exiting=True, s_max=cap-1.e-14))


def test_mixed_zero_force_and_full_vector_lanes():
    x = np.array([[0., 0, 0], [0, 0, 0], [0, 0, 0]])
    r = first_transition(x, [1., 0, 0], K_FIELD, [0, 1, -1],
                         [[.2, .3, .4], [0, 0, 0], [0, 0, 1]],
                         1., 3., -2., 2., s_max=3.)
    hit(r, [1., 1., np.pi/3], RIN)
