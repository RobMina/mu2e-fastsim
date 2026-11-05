"""Independent high-precision and analytic checks of restricted helix bounds."""

from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import math

import numpy as np
import pytest

from beamfastsim import tshelix as h, tsstep


def decimal_state(x, raw, p, q, b, s):
    with localcontext() as ctx:
        ctx.prec = 100
        x, raw, b = ([D.from_float(float(v)) for v in a] for a in (x, raw, b))
        p, s, k = D.from_float(float(p)), D.from_float(float(s)), D.from_float(tsstep.K)
        norm = sum(v*v for v in raw).sqrt()
        u = [v/norm for v in raw]
        w = [D(q)*k*s*v/p for v in b]
        term, total, integral = u[:], u[:], u[:]
        for j in range(1, 100):
            term = [(term[a]*w[c]-term[c]*w[a])/j for a, c in ((1, 2), (2, 0), (0, 1))]
            total = [v+t for v, t in zip(total, term)]
            integral = [v+t/(j+1) for v, t in zip(integral, term)]
        return [v+s*t for v, t in zip(x, integral)], total


def contains(interval, row, vector):
    assert all(D.from_float(float(a)) <= v <= D.from_float(float(b)) for a, b, v in zip(interval.lo[row], interval.hi[row], vector))


@pytest.mark.parametrize('pdg,q', [(13, -1), (-13, 1), (211, 1), (-211, -1)])
def test_general_field_high_precision_and_center_balls(pdg, q):
    x = np.array([[3904., -.125, -5000.], [0., 0., 0.], [-.1, .2, -.3]])
    raw = np.array([[3., 4., -2.], [-2., .1, 7.], [1., 0., 0.]])
    b = np.array([[.3, -1., 3.], [1., 1., 1.], [0., 0., 1.]])
    p, s = np.array([100., 1., 10.]), np.array([50., 1., 20.])
    receipt = h.evaluate(x, raw, p, pdg, b, s)
    for row in range(len(x)):
        xx, uu = decimal_state(x[row], raw[row], p[row], q, b[row], s[row])
        contains(receipt.position, row, xx)
        contains(receipt.direction, row, uu)
        with localcontext() as ctx:
            ctx.prec = 80
            assert sum((a-D.from_float(float(v)))**2 for a, v in zip(xx, receipt.x_center_mm[row])) <= D.from_float(float(receipt.position_radius_mm[row]))**2
            center = [D.from_float(float(v)) for v in receipt.direction_vector[row]]
            norm = sum(v*v for v in center).sqrt()
            assert sum((a-v/norm)**2 for a, v in zip(uu, center)) <= D.from_float(float(receipt.direction_radius[row]))**2
    assert receipt.evaluation_arithmetic_outward and not receipt.transcendental_library_assumed
    assert not receipt.floating_point_certified and not receipt.restart_state
    for array in (receipt.x_center_mm, receipt.direction_vector, receipt.momentum, receipt.pdg):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_tail_formula_rational_upper_bounds():
    receipt = h.evaluate(np.zeros((1, 3)), np.array([[1., 0., 0.]]), 1., 13, np.array([[0., 0., 1.]]), 3.)
    theta = F.from_float(float(receipt.phase_upper[0]))
    assert F.from_float(float(receipt.direction_tail[0])) >= 3*theta**19/math.factorial(19)
    assert F.from_float(float(receipt.position_tail_mm[0])) >= 9*theta**19/math.factorial(20)
    assert receipt.direction_tail[0] > 0


def test_straight_zero_length_parallel_and_tiny_zero_field_momentum():
    x = np.array([[.5, -.25, 4.], [1., 2., 3.], [0., 0., 0.]])
    u = np.array([[3., 4., 0.], [1., 2., 3.], [0., 0., 2.]])
    b = np.array([[0., 0., 0.], [1., 2., 3.], [0., 0., 1.]])
    p, s = np.array([np.nextafter(0., 1.), 100., 100.]), np.array([.125, 0., 50.])
    receipt = h.evaluate(x, u, p, 13, b, s)
    np.testing.assert_array_equal(receipt.position.lo[1], x[1])
    np.testing.assert_array_equal(receipt.position.hi[1], x[1])
    assert np.all(receipt.direction_tail[:2] == 0)
    assert np.all(receipt.position_tail_mm[:2] == 0)
    for row in range(3):
        xx, uu = decimal_state(x[row], u[row], p[row], -1, b[row], s[row])
        contains(receipt.position, row, xx)
        contains(receipt.direction, row, uu)


def test_errors_against_existing_core_and_perturbed_supplied_evaluation():
    from mu2efastsim.ehelix import helix
    x, raw, b = np.array([[100., 0., -3000.]]), np.array([[3., 4., 1.]]), np.array([[.2, .4, 2.]])
    p, s = np.array([100.]), np.array([20.])
    receipt = h.evaluate(x, raw, p, 13, b, s)
    xx, uu = helix(x, raw/np.hypot.reduce(raw, axis=1)[:, None], p, np.array([-1.]), b, s)
    errors = h.difference(receipt, xx, uu)
    assert errors.position_mm[0] < 1e-7 and errors.direction[0] < 1e-8
    xx = xx+np.array([[.125, -.25, .5]])
    uu = uu+np.array([[.1, -.2, .3]])
    changed = h.difference(receipt, xx, uu)
    assert changed.position_mm[0] > .5 and changed.direction[0] > .01
    ideal_x, ideal_u = decimal_state(x[0], raw[0], p[0], -1, b[0], s[0])
    with localcontext() as ctx:
        ctx.prec = 80
        assert sum((v-D.from_float(float(a)))**2 for v, a in zip(ideal_x, xx[0])) <= D.from_float(float(changed.position_mm[0]))**2
        center = [D.from_float(float(v)) for v in uu[0]]
        norm = sum(v*v for v in center).sqrt()
        assert sum((v-a/norm)**2 for v, a in zip(ideal_u, center)) <= D.from_float(float(changed.direction[0]))**2


def test_empty_scaled_direction_and_unrepresentable_position_retains_uncertainty():
    empty = h.evaluate(np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0))
    assert empty.position_radius_mm.shape == (0,)
    receipt = h.evaluate(np.array([[2.**54, 0., 0.]]), np.array([[1e200, 1e200, 0.]]), 100., 13, np.zeros((1, 3)), 1.)
    assert receipt.position_radius_mm[0] >= .7
    assert receipt.position.hi[0, 0] > receipt.position.lo[0, 0]


@pytest.mark.parametrize('changes', [dict(momentum=0.), dict(length=-1.), dict(pdg=0), dict(pdg=13.),
    dict(direction=np.zeros((1, 3))), dict(field=np.array([[np.inf, 0., 0.]])),
    dict(length=1000.), dict(x=np.array([[1, 2, 3]])), dict(momentum=1)])
def test_invalid_inputs_fail(changes):
    args = dict(x=np.zeros((1, 3)), direction=np.array([[1., 0., 0.]]), momentum=100., pdg=13,
                field=np.array([[0., 0., 1.]]), length=1.)
    args.update(changes)
    with pytest.raises(ValueError):
        h.evaluate(**args)


@pytest.mark.parametrize('pdg,q', [(13, -1), (-13, 1), (211, 1), (-211, -1)])
def test_axial_closed_form_with_exact_rational_trigonometric_oracle(pdg, q):
    receipt = h.evaluate(np.zeros((1, 3)), np.array([[1., 0., 0.]]), 2., pdg, np.array([[0., 0., 1.]]), 1.)
    alpha = q*F.from_float(tsstep.K)/2
    sine = sum((-1)**j*alpha**(2*j+1)/math.factorial(2*j+1) for j in range(32))
    cosine = sum((-1)**j*alpha**(2*j)/math.factorial(2*j) for j in range(33))
    sin_error = 3*abs(alpha)**65/math.factorial(65)
    cos_error = 3*abs(alpha)**66/math.factorial(66)
    intervals = [((sine-sin_error)/alpha, (sine+sin_error)/alpha),
                 ((cosine-1-cos_error)/alpha, (cosine-1+cos_error)/alpha), (F(0), F(0))]
    directions = [(cosine-cos_error, cosine+cos_error), (-sine-sin_error, -sine+sin_error), (F(0), F(0))]
    for axis in range(3):
        lo, hi = sorted(intervals[axis])
        assert F.from_float(float(receipt.position.lo[0, axis])) <= lo <= hi <= F.from_float(float(receipt.position.hi[0, axis]))
        lo, hi = directions[axis]
        assert F.from_float(float(receipt.direction.lo[0, axis])) <= lo <= hi <= F.from_float(float(receipt.direction.hi[0, axis]))


def test_power_two_direction_scales_and_tiny_nonzero_phase():
    raw = np.array([[3., 4., 1.]])
    args = (np.zeros((1, 3)), raw, 100., 13, np.array([[0., 0., 1.]]), 5.)
    for exponent in (-500, 0, 500):
        receipt = h.evaluate(args[0], np.ldexp(raw, exponent), *args[2:])
        xx, uu = decimal_state(args[0][0], raw[0], 100., -1, args[4][0], 5.)
        contains(receipt.position, 0, xx)
        contains(receipt.direction, 0, uu)
    tiny = np.nextafter(0., 1.)
    receipt = h.evaluate(np.zeros((1, 3)), np.array([[1., 0., 0.]]), 100., 13, np.array([[0., 0., tiny]]), 1.)
    assert receipt.phase_upper[0] > 0 and receipt.direction_tail[0] > 0
    alpha = F.from_float(tsstep.K)*F.from_float(tiny)/100
    sin_lower, sin_upper = alpha-alpha**3/6, alpha
    assert F.from_float(float(receipt.direction.lo[0, 1])) <= sin_lower <= sin_upper <= F.from_float(float(receipt.direction.hi[0, 1]))
    y_lower, y_upper = alpha/2-alpha**3/24, alpha/2
    assert F.from_float(float(receipt.position.lo[0, 1])) <= y_lower <= y_upper <= F.from_float(float(receipt.position.hi[0, 1]))


@pytest.mark.parametrize('raw', [np.array([[np.nextafter(0., 1.), 0., 0.]]),
    np.array([[np.finfo(float).max, np.finfo(float).max, 0.]])])
def test_unsupported_normalization_extremes_fail_closed(raw):
    with pytest.raises(ValueError):
        h.evaluate(np.zeros((1, 3)), raw, 100., 13, np.zeros((1, 3)), 1.)


def test_invalid_difference_and_direction_zero_refused():
    receipt = h.evaluate(np.zeros((1, 3)), np.array([[1., 0., 0.]]), 100., 13, np.zeros((1, 3)), 0.)
    with pytest.raises(ValueError):
        h.difference(receipt, np.zeros((1, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        h.difference(receipt, np.zeros((2, 3)), np.ones((2, 3)))
