"""Exact-rational bound oracles, not full field or trajectory certification."""

from fractions import Fraction as F

import numpy as np
import pytest

from beamfastsim import tsbudget as b, tsfield, tsstep


def f(value):
    return F.from_float(float(value))


def item(nodes=None, spacing=(1., 2., 4.), *, reflect=False):
    if nodes is None:
        nodes = np.arange(2*3*4*3, dtype=np.float64).reshape(2, 3, 4, 3)/17
    head = tsfield.GridHeader('synthetic', [0., 0., 0.], [0., 0., 0.], nodes.shape[:3], spacing, reflect)
    return tsfield.FieldMap(head, nodes)


def test_all_nodes_and_edges_exact_squared_oracles_and_chunk_invariance():
    field = item()
    bound = b.scan(field, chunk_nodes=5)
    assert bound == b.scan(field, chunk_nodes=10000)
    assert bound.node_count == 24 and bound.edge_count == 46
    for node in field.B.reshape(-1, 3):
        assert f(bound.magnitude_tesla)**2 >= sum(f(v)**2 for v in node)
    for axis in range(3):
        for index in np.ndindex(*(field.n-np.eye(3, dtype=int)[axis])):
            other = list(index)
            other[axis] += 1
            slopes = [(f(x)-f(y))/f(field.d[axis]) for x, y in zip(field.B[tuple(other)], field.B[index])]
            assert f(bound.column_tesla_per_mm[axis])**2 >= sum(x*x for x in slopes)
    assert f(bound.lipschitz_tesla_per_mm)**2 >= sum(f(x)**2 for x in bound.column_tesla_per_mm)
    assert bound.bound_arithmetic_outward and not bound.floating_point_certified


def test_constant_zero_reflection_fingerprint_and_tiny_gradients():
    zero = b.scan(item(np.zeros((2, 2, 2, 3))))
    assert zero.magnitude_tesla == zero.lipschitz_tesla_per_mm == 0
    nodes = np.broadcast_to([3., 4., 0.], (2, 2, 2, 3)).copy()
    constant = b.scan(item(nodes))
    assert constant.magnitude_tesla >= 5. and constant.lipschitz_tesla_per_mm == 0
    assert not b.scan(item(nodes, reflect=True)).reflection_continuous
    nodes[..., 1] = 0
    assert b.scan(item(nodes, reflect=True)).reflection_continuous
    nodes = np.zeros((2, 2, 2, 3))
    nodes[1, :, :, 0] = np.nextafter(0., 1.)
    tiny = b.scan(item(nodes))
    assert tiny.lipschitz_tesla_per_mm > 0
    assert tiny.fingerprint != zero.fingerprint


def test_carried_polynomials_exact_rational_oracle_batched_and_caps():
    bound = b.scan(item())
    p = np.array([.5, 1., 100., 1e-8])
    s = np.array([0., .125, 2., 4.])
    ex, eu, eb = np.array([.1, 0., .25, .5]), np.array([.2, .1, .125, 1.]), np.array([0., .01, .02, .03])
    result = b.errors(bound, p, s, position_error_mm=ex, direction_error=eu, field_error_tesla=eb)
    for j in range(4):
        a, g, t = f(tsstep.K)/f(p[j]), f(bound.lipschitz_tesla_per_mm), f(s[j])
        rate = a*(g*f(ex[j])+f(eb[j]))
        dx = min(f(ex[j])+f(eu[j])*t+rate*t*t/2+a*g*t**3/6, f(ex[j])+2*t)
        du = min(F(2), f(eu[j])+rate*t+a*g*t*t/2)
        assert f(result.position_mm[j]) >= dx and f(result.direction[j]) >= du
        assert f(result.curvature_per_mm[j]) >= a*f(bound.magnitude_tesla)
    assert result.position_mm[0] == ex[0] and result.direction[0] == eu[0]
    with pytest.raises(ValueError):
        result.position_mm.setflags(write=True)


def test_zero_field_tiny_momentum_and_empty_rows():
    bound = b.scan(item(np.zeros((2, 2, 2, 3))))
    out = b.errors(bound, np.array([np.nextafter(0., 1.)]), np.array([1.]),
        position_error_mm=0., direction_error=0., field_error_tesla=0.)
    assert out.curvature_per_mm[0] == 0 and out.position_mm[0] >= 0
    out = b.errors(bound, np.empty(0), np.empty(0), position_error_mm=0., direction_error=0., field_error_tesla=0.)
    assert out.position_mm.shape == (0,)


@pytest.mark.parametrize('kwargs', [{'momentum': np.array([0.])}, {'length': np.array([-1.])},
    {'direction_error': 3.}, {'position_error_mm': -1.}, {'field_error_tesla': np.nan}, {'momentum': np.array([1], dtype=int)}])
def test_invalid_controls_fail(kwargs):
    inputs = dict(momentum=np.array([1.]), length=np.array([1.]), position_error_mm=0., direction_error=0., field_error_tesla=0.)
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        b.errors(b.scan(item()), **inputs)


@pytest.mark.parametrize('chunk', [0, True, 1.5, 1000001])
def test_invalid_chunk(chunk):
    with pytest.raises(ValueError):
        b.scan(item(), chunk_nodes=chunk)


def test_fortran_layout_same_scan_and_fingerprint():
    original = item()
    other = tsfield.FieldMap(original.header, np.asfortranarray(original.B))
    assert b.scan(original, chunk_nodes=3) == b.scan(other, chunk_nodes=7)


def test_nonzero_error_underflow_remains_positive_and_finite_overflow_refuses():
    bound = b.scan(item(np.zeros((2, 2, 2, 3))))
    tiny = np.nextafter(0., 1.)
    out = b.errors(bound, np.array([1.]), np.array([.5]), position_error_mm=0.,
        direction_error=0., field_error_tesla=tiny)
    assert f(out.position_mm[0]) >= f(tsstep.K)*f(tiny)/8
    assert f(out.direction[0]) >= f(tsstep.K)*f(tiny)/2
    with pytest.raises(ValueError):
        b.errors(b.scan(item()), np.array([tiny]), np.array([1.]), position_error_mm=0.,
                 direction_error=0., field_error_tesla=0.)
