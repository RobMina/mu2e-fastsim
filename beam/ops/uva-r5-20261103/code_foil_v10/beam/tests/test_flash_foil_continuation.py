"""Exact synthetic receipts only; no transport/numerical-receipt integration.

Analytic waypoints below are supplied independently, never by projecting an
approximate returned endpoint. A general bisection endpoint is NOT a receipt.
"""
import numpy as np
import pytest

from beamfastsim import flash_foil_boundary as ray
from beamfastsim import flash_foil_helix as helix


def call(module, x, u, **kw):
    args = dict(rin=1., rout=3., zlo=-2., zhi=2., s_max=8.)
    if module is helix:
        args.update(p=helix.K_FIELD, q=1, B=[0., 0., 1.])
    args.update(kw)
    return module.first_transition(x, u, **args)


def hit(r, distance, kind, dtype=np.float64):
    assert np.all(r['valid']) and not np.any(r['unresolved'])
    np.testing.assert_array_equal(r['kind'], np.broadcast_to(kind, r['kind'].shape))
    np.testing.assert_allclose(r['distance'], distance, rtol=0,
                               atol=3.e-5 if dtype == np.float32 else 2.e-11)
    assert np.all(r['distance'] > 0)


def refused(r):
    assert np.all(r['unresolved']) and not np.any(r['valid'])
    assert np.all(np.isinf(r['distance'])) and np.all(r['kind'] == ray.NONE)


def miss(r):
    assert not np.any(r['unresolved']) and not np.any(r['valid'])
    assert np.all(np.isinf(r['distance'])) and np.all(r['kind'] == ray.NONE)


@pytest.mark.parametrize('module', [ray, helix])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_consecutive_exact_ray_rim_hole_entry_exit_reentry(module, dtype):
    # Outer entry -> hole exit -> far-side hole reentry -> outer exit.
    # Helix zero-force delegation must use precisely the same protocol.
    kw = dict(B=[0., 0., 0.]) if module is helix else {}
    u = np.array([1., 0., 0.], dtype)
    for x, token, ex, distance, kind in [
        (-3, ray.ROUT, True, 2, ray.RIN),
        (-1, ray.RIN, False, 2, ray.RIN),
        (1, ray.RIN, True, 2, ray.ROUT),
    ]:
        r = call(module, np.array([x, 0, 0], dtype), u,
                 after_surface=token, exiting=ex, s_max=distance, **kw)
        hit(r, distance, kind, dtype)
    miss(call(module, np.array([3., 0, 0], dtype), u,
              after_surface=ray.ROUT, exiting=False, **kw))


@pytest.mark.parametrize('module', [ray, helix])
@pytest.mark.parametrize('sign', [-1, 1])
def test_exact_face_entry_then_exit_and_vacuum(module, sign):
    lo, hi = (-2, 2) if sign > 0 else (2, -2)
    first, second = (ray.ZLO, ray.ZHI) if sign > 0 else (ray.ZHI, ray.ZLO)
    x = np.array([2., 0., lo]); u = np.array([0., 0., sign])
    hit(call(module, x, u, after_surface=first, exiting=True, s_max=4), 4, second)
    x[2] = hi
    miss(call(module, x, u, after_surface=second, exiting=False))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('q', [-1, 1])
def test_curved_exact_hole_chain_and_same_surface_return(dtype, q):
    # Circle centre (1,-q), R=1. r=1 crossings at 0,3pi/2,2pi.
    # The exact second waypoint is (0,-q), with velocity (0,q).
    # Avoid putting an approximate transcendental root on a dyadic endpoint.
    kw = dict(q=q, s_max=7.)
    hit(call(helix, np.array([1., 0, 0], dtype), np.array([1., 0, 0], dtype),
             after_surface=ray.RIN, exiting=True, **kw), 1.5*np.pi, ray.RIN, dtype)
    hit(call(helix, np.array([0., -q, 0], dtype), np.array([0., q, 0], dtype),
             after_surface=ray.RIN, exiting=False, **kw), .5*np.pi, ray.RIN, dtype)
    # Outer rim departure returns to the same rim later, with no hole crossing.
    hit(call(helix, np.array([3., 0, 0], dtype), np.array([-1., 0, 0], dtype),
             after_surface=ray.ROUT, exiting=True, **kw), 2*np.arctan(3), ray.ROUT, dtype)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('q', [-1, 1])
def test_tilted_B_exact_faces_exit_reentry_and_later_face(dtype, q):
    # B along y: z=sin(s), x=2-q*(1-cos(s)). Face receipts are
    # exact even though the transverse coordinates need not be rational.
    kw = dict(q=q, B=[0., 1., 0.], rin=0., rout=20., zlo=0., zhi=.5,
              s_max=7.)
    x = np.array([2., 0, 0], dtype); u = np.array([0., 0, 1], dtype)
    hit(call(helix, x, u, after_surface=ray.ZLO, exiting=True, **kw),
        np.pi/6, ray.ZHI, dtype)
    c = np.sqrt(.75)
    x = np.array([2-q*(1-c), 0, .5], dtype)
    u = np.array([-.5*q, 0, c], dtype)
    hit(call(helix, x, u, after_surface=ray.ZHI, exiting=False, **kw),
        2*np.pi/3, ray.ZHI, dtype)
    x = np.array([2-q*(1+c), 0, .5], dtype)
    u = np.array([-.5*q, 0, -c], dtype)
    hit(call(helix, x, u, after_surface=ray.ZHI, exiting=True, **kw),
        np.pi/6, ray.ZLO, dtype)


@pytest.mark.parametrize('module', [ray, helix])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_refuse_wrong_token_ownership_corner_tangent_and_near_receipts(module, dtype):
    u = np.array([-1., 0, 0], dtype)
    for x, v, token, ex in [
        ([3, 0, 0], u, ray.RIN, True),                # wrong kind
        ([3, 0, 0], u, ray.ROUT, False),              # backward ownership
        ([3, 0, -2], u, ray.ROUT, True),             # corner
        ([3, 0, 0], [0, 1, 0], ray.ROUT, False),     # tangent
        ([2, 0, 0], u, ray.ROUT, True),              # not a start root
        ([0, 0, -2], [0, 0, 1], ray.ZLO, True),     # outside other constraints
    ]:
        refused(call(module, np.array(x, dtype), np.asarray(v, dtype),
                     after_surface=token, exiting=ex))
    for side in [-np.inf, np.inf]:
        x = np.array([np.nextafter(dtype(3), dtype(side)), 0, 0], dtype)
        refused(call(module, x, u, after_surface=ray.ROUT, exiting=True))
    # Even a loose user tolerance must not permit an approximate receipt.
    refused(call(module, np.array([3.001, 0, 0], dtype), u,
                 after_surface=ray.ROUT, exiting=True, tolerance=.01))


@pytest.mark.parametrize('module', [ray, helix])
@pytest.mark.parametrize('token', [True, 1., 1.5, -1, 5, [0, 8], 'zlo'])
def test_token_host_validation_is_integer_kind_only(module, token):
    with pytest.raises(ValueError):
        call(module, [3., 0, 0], [-1., 0, 0], after_surface=token, exiting=True)


def test_curved_departure_budget_short_cap_and_short_material_are_not_nohit():
    x = np.array([3., 0, 0]); u = np.array([-1., 0, 0])
    for kw in [dict(max_intervals=1), dict(s_max=0), dict(s_max=1.e-15),
               dict(rin=2.9999, tolerance=.01)]:
        refused(call(helix, x, u, after_surface=ray.ROUT, exiting=True, **kw))
    miss(call(helix, x, u, after_surface=ray.ROUT, exiting=True, s_max=.01))
    # Rays can prove the cap=0 empty set after excluding their analytic root.
    miss(call(ray, x, u, after_surface=ray.ROUT, exiting=True, s_max=0))


def test_curved_inclusive_cap_after_certified_departure():
    # Nonzero B parallel to direction: genuine curved-search path, A=0.
    x = np.array([2., 0, -2]); u = np.array([0., 0, 1])
    hit(call(helix, x, u, after_surface=ray.ZLO, exiting=True, s_max=4), 4, ray.ZHI)
    refused(call(helix, x, u, after_surface=ray.ZLO, exiting=True, s_max=4-1.e-14))
    miss(call(helix, x, u, after_surface=ray.ZLO, exiting=True, s_max=3.9))


@pytest.mark.parametrize('module', [ray, helix])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_jit_mixed_none_continuation_zero_force_and_refusal(module, dtype):
    jax = pytest.importorskip('jax')
    x = np.array([[3, 0, 0], [3, 0, 0], [3, 0, 0], [3, 0, 0],
                  [3, 0, 0], [2, 0, -2], [3, 0, 0]], dtype)
    u = np.array([[-1, 0, 0]]*5+[[0, 0, 1], [0, 1, 0]], dtype)
    kw = dict(rin=1., rout=3., zlo=-2., zhi=2., s_max=7.,
              exiting=np.array([True, False, True, True, True, True, False]),
              after_surface=np.array([ray.ROUT, ray.NONE, ray.RIN, ray.ROUT,
                                      ray.ROUT, ray.ZLO, ray.ROUT], np.int32))
    if module is helix:
        kw.update(p=helix.K_FIELD, q=np.array([1, -1, 1, 0, 1, -1, 1]),
                  B=np.array([[0, 0, 1]]*4+[[0, 0, 0], [.2, .1, .9], [0, 0, 1]], dtype))
    module.validate_inputs(x, u, **kw)
    expected = module.first_transition(x, u, **kw)
    fn = jax.jit(module.first_transition_jax)
    actual = fn(x, u, **kw)
    for key in ('kind', 'valid', 'unresolved'):
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_allclose(actual['distance'], expected['distance'], rtol=0,
                               atol=3.e-5 if dtype == np.float32 else 2.e-11)
    assert actual['distance'].dtype == dtype
    assert actual['distance'][1] == 0  # NONE is still backward-owned.
    assert actual['unresolved'][2] and actual['unresolved'][-1]
    assert np.all(np.asarray(actual['valid'])[[0, 3, 4, 5]])
    assert np.all(np.asarray(actual['distance'])[[0, 3, 4, 5]] > 0)
    # Invalid traced numeric token kinds refuse rather than silently cast.
    refused(fn(x, u, **dict(kw, after_surface=np.ones(7, dtype))))


def test_numeric_receipt_integration_gate_nonzero_curved_residual_refused():
    # Independent propagation of an accepted approximate root is generally not
    # exactly on the named surface. Explicitly demonstrate the remaining gate.
    x = np.array([2., 0, 0]); u = np.array([1., 0, 0])
    r = call(helix, x, u, exiting=True, s_max=6.)
    assert r['valid'] and r['kind'] == ray.ROUT
    s = r['distance']
    endpoint = np.array([2+np.sin(s), np.cos(s)-1, 0])
    velocity = np.array([np.cos(s), -np.sin(s), 0])
    assert np.hypot(*endpoint[:2]) != 3
    refused(call(helix, endpoint, velocity, after_surface=ray.ROUT,
                 exiting=False, s_max=6.))


@pytest.mark.parametrize('module', [ray, helix])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_default_and_explicit_none_are_byte_identical(module, dtype):
    x = np.array([[3, 0, 0], [2, 0, 0], [0, 0, 0], [10, 0, 0],
                  [3, 0, -2], [3, 0, 0]], dtype)
    u = np.array([[-1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                  [-1, 0, 0], [0, 1, 0]], dtype)
    kw = dict(exiting=np.array([False, True, False, False, False, False]))
    default = call(module, x, u, **kw)
    explicit = call(module, x, u, after_surface=np.zeros(6, np.int32), **kw)
    for key in default:
        assert default[key].dtype == explicit[key].dtype
        assert default[key].tobytes() == explicit[key].tobytes()


@pytest.mark.parametrize('q', [-1, 1])
def test_departure_does_not_skip_other_faces_hole_or_rim(q):
    # Non-axial B: a consumed radial root must not hide a later plane crossing.
    x = np.array([3., 0, 0]); u = np.array([-1., 0, 0])
    r = call(helix, x, u, q=q, B=[0., 1., 0.], zlo=-.2, zhi=.2,
             after_surface=ray.ROUT, exiting=True, s_max=7.)
    hit(r, np.arccos(.8), ray.ZLO if q == 1 else ray.ZHI)
    # Consumed plane with nonzero B parallel to u, followed by hole then rim.
    u = np.array([.6, 0, .8])
    for x, kind, distance in [([-2., 0, -2], ray.RIN, 1/.6),
                              ([2., 0, -2], ray.ROUT, 1/.6)]:
        hit(call(helix, np.array(x), u, q=q, B=u, zhi=20.,
                 after_surface=ray.ZLO, exiting=True, s_max=7.), distance, kind)


@pytest.mark.parametrize('module', [ray, helix])
def test_after_surface_target_broadcast_offset_and_other_near_constraint(module):
    x = np.array([[12., 0, -2], [12., 0, -2]])[:, None, :]
    u = np.array([0., 0, 1])
    r = call(module, x, u, after_surface=np.array([ray.ZLO, ray.NONE]),
             exiting=np.array([True, False]), axis_x=10., s_max=4.)
    np.testing.assert_array_equal(r['valid'], np.ones((2, 2), bool))
    np.testing.assert_array_equal(r['distance'], [[4, 0], [4, 0]])
    # A second unresolved constraint cannot be ignored just because one token
    # happens to name an exact surface.
    refused(call(module, np.array([3-1.e-14, 0, -2]), u,
                 after_surface=ray.ZLO, exiting=True))
