"""Exact-rational arithmetic oracles for the restricted binary64 interval layer."""

from fractions import Fraction
import operator

import numpy as np
import pytest

from beamfastsim import tsinterval as iv, tsplane


def f(x):
    return Fraction.from_float(float(x))


@pytest.mark.parametrize('name,operation', [('add', operator.add), ('sub', operator.sub),
    ('mul', operator.mul), ('div', operator.truediv)])
def test_batched_corner_oracle(name, operation):
    rng = np.random.default_rng(12)
    a = np.sort(rng.uniform(-10, 10, (2, 128)), axis=0)
    b = np.sort(rng.uniform(1, 10, (2, 128)), axis=0)
    result = getattr(iv, name)(iv.Interval(*a), iv.Interval(*b))
    for j in range(128):
        oracle = [operation(f(x), f(y)) for x in a[:, j] for y in b[:, j]]
        assert f(result.lo[j]) <= min(oracle) <= max(oracle) <= f(result.hi[j])
    assert not result.lo.flags.writeable and not result.hi.flags.writeable
    with pytest.raises(ValueError):
        result.lo.setflags(write=True)


def test_cancellation_absorbed_addends_underflow_and_signed_zeros():
    tiny = np.nextafter(0., 1.)
    for a, b in ((2**54*1., 1.), (1., -1.), (tiny, -tiny), (0., -0.)):
        out = iv.add(iv.Interval(a), iv.Interval(b))
        assert f(out.lo) <= f(a)+f(b) <= f(out.hi)
    out = iv.mul(iv.Interval(tiny), iv.Interval(.5))
    assert f(out.lo) <= f(tiny)*f(.5) <= f(out.hi)
    out = iv.div(iv.Interval(tiny), iv.Interval(2.))
    assert f(out.lo) <= f(tiny)/2 <= f(out.hi)


def test_square_sqrt_and_scaled_norm_exact_squared_oracles():
    intervals = iv.Interval(np.array([-3., -2., 0., 1., 1e-300]), np.array([-1., 3., 0., 4., 2e-300]))
    sq = iv.square(intervals)
    for j, (a, b) in enumerate(zip(intervals.lo, intervals.hi)):
        low = 0 if a <= 0 <= b else min(f(a)**2, f(b)**2)
        high = max(f(a)**2, f(b)**2)
        assert f(sq.lo[j]) <= low <= high <= f(sq.hi[j])
    values = np.array([0., 2., 1e-300, 1e300])
    root = iv.sqrt(iv.Interval(values))
    for j, value in enumerate(values):
        assert f(root.lo[j])**2 <= f(value) <= f(root.hi[j])**2
    vectors = np.array([[3., 4., 0.], [1e300, 2e300, 0.], [1e-300, 2e-300, 0.], [0., 0., 0.]])
    norms = iv.norm(iv.Interval(vectors))
    for j, vector in enumerate(vectors):
        oracle = sum(f(x)**2 for x in vector)
        assert f(norms.lo[j])**2 <= oracle <= f(norms.hi[j])**2
    np.testing.assert_array_equal([norms.lo[-1], norms.hi[-1]], [0., 0.])


def test_normalization_and_plane_projection_bound_not_floating_unit_claim():
    vector = np.array([[3., 4., 0.], [0., 0., 7.]])
    unit = iv.normalize(iv.Interval(vector))
    expected = np.array([[.6, .8, 0.], [0., 0., 1.]])
    assert np.all(unit.lo <= expected) and np.all(unit.hi >= expected)
    points = iv.Interval(np.array([[2**54*1., 1., 2.], [0., 0., -1.]]))
    origin = iv.Interval(np.array([2**54*1., 0., 0.]))
    normal = iv.Interval(np.array([0., 0., 7.]))
    residual = tsplane.signed(points, origin, normal)
    assert np.all(residual.lo <= [2., -1.]) and np.all(residual.hi >= [2., -1.])
    projection = tsplane.projection(iv.Interval(vector), normal)
    assert np.all(projection.lo <= [0., 1.]) and np.all(projection.hi >= [0., 1.])


def test_interval_inflation_dot_and_broadcasting():
    a = iv.Interval(np.array([[1., 2., 3.], [4., 5., 6.]]))
    b = iv.Interval(np.array([1., -1., 2.]))
    result = iv.dot(a, b)
    assert np.all(result.lo <= [5., 11.]) and np.all(result.hi >= [5., 11.])
    inflated = iv.inflate(a, iv.Interval(np.array(.25)))
    assert np.all(inflated.lo <= a.lo-.25) and np.all(inflated.hi >= a.hi+.25)


@pytest.mark.parametrize('call', [lambda: iv.Interval([1, 2]), lambda: iv.Interval(np.array([1., np.nan])),
    lambda: iv.Interval(2., 1.), lambda: iv.div(iv.Interval(1.), iv.Interval(-1., 1.)),
    lambda: iv.sqrt(iv.Interval(-1., 1.)), lambda: iv.normalize(iv.Interval(np.zeros(3))),
    lambda: iv.add(iv.Interval(np.finfo(float).max), iv.Interval(np.finfo(float).max)),
    lambda: iv.mul(iv.Interval(1e300), iv.Interval(1e300)),
    lambda: iv.sum_last(iv.Interval(np.empty((3, 0)))), lambda: iv.norm(iv.Interval(np.array(1.))),
    lambda: iv.inflate(iv.Interval(1.), iv.Interval(-1.))])
def test_unsupported_input_and_arithmetic_fail_closed(call):
    with pytest.raises(ValueError):
        call()


def test_two_axis_broadcast_and_subnormal_normalization_refusal():
    a = iv.Interval(np.array([[1.], [-2.]]))
    b = iv.Interval(np.array([3., -4., 5.]))
    out = iv.mul(a, b)
    expected = np.array([[3., -4., 5.], [-6., 8., -10.]])
    assert np.all(out.lo <= expected) and np.all(out.hi >= expected)
    with pytest.raises(ValueError):
        iv.normalize(iv.Interval(np.array([np.nextafter(0., 1.), 0., 0.])))


def test_plane_cancellation_encloses_exact_rational_normalized_dot():
    from decimal import Decimal, localcontext
    points = np.array([[1e16, -1e16, 1.], [1., 2., 3.], [0., 0., 0.]])
    origin = np.array([1., -2., .5])
    normal = np.array([1., 1., 1.])
    out = tsplane.signed(iv.Interval(points), iv.Interval(origin), iv.Interval(normal))
    with localcontext() as context:
        context.prec = 90
        length = Decimal(3).sqrt()
        for i, point in enumerate(points):
            exact = sum(Decimal.from_float(float(x))-Decimal.from_float(float(y)) for x, y in zip(point, origin))/length
            assert Decimal.from_float(float(out.lo[i])) <= exact <= Decimal.from_float(float(out.hi[i]))


def test_plane_over_wide_normal_box_refuses_zero_normalization():
    normal = iv.Interval(np.array([-1., -1., -1.]), np.array([1., 1., 1.]))
    with pytest.raises(ValueError):
        tsplane.signed(iv.Interval(np.zeros((2, 3))), iv.Interval(np.zeros(3)), normal)
    for function in (tsplane.signed, tsplane.projection):
        with pytest.raises(ValueError):
            if function is tsplane.signed:
                function(iv.Interval(np.ones((1, 2))), iv.Interval(np.zeros(3)), iv.Interval(np.ones(3)))
            else:
                function(iv.Interval(np.ones((1, 2))), iv.Interval(np.ones(3)))
