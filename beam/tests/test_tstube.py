"""Whole-interval regression evidence; no native or material transport claim."""

from fractions import Fraction as F
import math

import numpy as np
import pytest

from beamfastsim import tstube as t, tsfield, tsinterval as iv, tslocate, tsnav, tssurface


def item(field=(0., 0., 0.), *, reflect=False, gap=0., varying=False):
    head = tsfield.GridHeader('synthetic', [-4., gap if reflect else -4., -4.], [0., 0., 0.], [3, 3, 3], [4., 4., 4.], reflect)
    nodes = np.broadcast_to(field, (3, 3, 3, 3)).copy()
    if varying:
        nodes[..., 2] = np.array([-.125, 0., .125])[None, None, :]
    return tsfield.FieldMap(head, nodes)


def enclose(ctx, **kwargs):
    args = dict(x=np.array([[0., 0., 0.]]), direction=np.array([[1., 0., 0.]]), momentum=100., pdg=13,
        lo=0., hi=.5, position_error_mm=0., direction_error=0.)
    args.update(kwargs)
    return t.enclose(ctx, **args)


def contains(interval, row, values):
    assert all(F.from_float(float(a)) <= v <= F.from_float(float(b)) for a, b, v in zip(interval.lo[row], interval.hi[row], values))


def test_whole_straight_interval_initial_uncertainty_and_endpoint():
    ctx = t.Context(item())
    result = enclose(ctx, x=np.array([[.25, -.5, .125]]), direction=np.array([[3., 4., 0.]]),
        lo=.125, hi=.5, position_error_mm=.001, direction_error=.002)
    for s in (F(1, 8), F(3, 16), F(1, 4), F(1, 2)):
        contains(result.position, 0, [F(1, 4)+s*F(3, 5), -F(1, 2)+s*F(4, 5), F(1, 8)])
    contains(result.endpoint_position, 0, [F(1, 4)+F(3, 10), -F(1, 2)+F(2, 5), F(1, 8)])
    assert result.model.position_mm[0] >= .002 and result.model.direction[0] >= .002
    assert result.whole_interval and not result.restart_state and not result.material_transport
    assert result.query.stored_grid_domain_checked
    for a in (result.initial_x_mm, result.hi_mm, result.pdg, ctx.item.B, ctx.item.header.n):
        with pytest.raises(ValueError):
            a.setflags(write=True)


@pytest.mark.parametrize('pdg,q', [(13, -1), (-13, 1), (211, 1), (-211, -1)])
def test_uniform_all_parameters_selected_exact_rational_axial_oracles(pdg, q):
    result = enclose(t.Context(item((0., 0., 1.))), momentum=1., pdg=pdg)
    k = F.from_float(float(result.reference.phase_upper[0]))
    assert k > 0
    for s in (F(0), F(1, 8), F(1, 4), F(3, 8), F(1, 2)):
        phase = q*F.from_float(.299792458)*s
        sine, cosine = trig_bounds(phase)
        for sin in sine:
            for cos in cosine:
                if s:
                    contains(result.position, 0, [s*sin/phase, s*(cos-1)/phase, F(0)])
                else:
                    contains(result.position, 0, [F(0)]*3)
                contains(result.direction, 0, [cos, -sin, F(0)])


def test_continuous_reflection_zero_by_allowed_discontinuous_or_gap_refused():
    good = t.Context(item((1., 0., 0.), reflect=True))
    result = enclose(good)
    assert result.context.bounds.reflection_continuous
    with pytest.raises(ValueError, match='continuous reflected seam'):
        enclose(t.Context(item((0., 1., 0.), reflect=True)))
    with pytest.raises(ValueError, match='domain cover'):
        enclose(t.Context(item(reflect=True, gap=.25)))
    result = enclose(t.Context(item((0., 1., 0.), reflect=True)), x=np.array([[0., 2., 0.]]))
    assert not result.context.bounds.reflection_continuous


def test_reach_cover_includes_initial_error_not_just_nominal_path():
    ctx = t.Context(item())
    with pytest.raises(ValueError, match='domain cover'):
        enclose(ctx, x=np.array([[3.5, 0., 0.]]), hi=.25, position_error_mm=.5)
    with pytest.raises(ValueError, match='work cap'):
        enclose(ctx, max_cells=1)
    with pytest.raises(ValueError, match='phase'):
        enclose(t.Context(item((0., 0., 100.))), momentum=1.)


def test_context_snapshots_flags_and_nodes():
    field = item()
    ctx = t.Context(field)
    field.B.setflags(write=True)
    field.B[:] = 100.
    field.header.grid_origin.setflags(write=True)
    field.header.grid_origin[:] = -100.
    result = enclose(ctx)
    assert np.all(result.query.value_tesla == 0.)
    assert np.all(ctx.item.header.grid_origin == -4.)


def test_varying_field_model_metadata_preserves_initial_errors():
    ctx = t.Context(item(varying=True))
    ex, eu = .001, .002
    result = enclose(ctx, position_error_mm=ex, direction_error=eu)
    assert result.model.position_mm[0] > ex+.5*eu
    assert result.model.direction[0] > eu
    assert result.context.bounds.lipschitz_tesla_per_mm > 0
    assert result.query.evaluation_error_tesla[0] > 0
    contains(result.position, 0, [F(0)]*3)
    for value in (result.endpoint_position, result.endpoint_direction):
        assert np.all(value.lo <= value.hi)


def test_direct_guard_over_whole_incoming_set_no_owner_or_advance():
    mother = tslocate.Volume(tsnav.Box([10., 10., 10.]), 'vac', None, 'root', True)
    leaf = tslocate.Volume(tsnav.Box([2., 2., .125]), 'metal', None, 'root', False)
    part = tslocate.Partition((tslocate.Cell(mother, (leaf,)),))
    face = next(f for f in tssurface.faces(part, 1) if f.key == 'z-')
    result = enclose(t.Context(item()), x=np.array([[0., 0., -.15625]]), direction=np.array([[0., 0., 1.]]), hi=.0625)
    bound = t.guard(result, face)
    assert bound.guard.clearance.lo[0] > 0 and bound.primitive_plane.lo[0] < 0 < bound.primitive_plane.hi[0]
    assert bound.face is face and not bound.material_transport
    long = enclose(t.Context(item()), x=np.array([[0., 0., -.15625]]), direction=np.array([[0., 0., 1.]]), hi=.5)
    assert t.guard(long, face).guard.clearance.lo[0] < 0
    np.testing.assert_array_equal(result.initial_x_mm, [[0., 0., -.15625]])


def test_zero_and_empty_intervals_and_no_error_reset():
    result = enclose(t.Context(item()), lo=0., hi=0., position_error_mm=.01, direction_error=.02)
    assert result.model.position_mm[0] == .01 and result.model.direction[0] == .02
    contains(result.endpoint_position, 0, [F(0)]*3)
    empty = t.enclose(t.Context(item()), np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0, dtype=int),
        np.empty(0), np.empty(0), position_error_mm=0., direction_error=0.)
    assert empty.position.lo.shape == (0, 3)


@pytest.mark.parametrize('kwargs', [dict(lo=.75), dict(hi=-1.), dict(momentum=0.), dict(direction_error=3.),
    dict(position_error_mm=-1.), dict(direction=np.zeros((1, 3))), dict(x=np.array([[0, 0, 0]]))])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        enclose(t.Context(item()), **kwargs)


def trig_bounds(phase):
    sine = sum((-1)**j*phase**(2*j+1)/math.factorial(2*j+1) for j in range(20))
    cosine = sum((-1)**j*phase**(2*j)/math.factorial(2*j) for j in range(21))
    se = 3*abs(phase)**41/math.factorial(41)
    ce = 3*abs(phase)**42/math.factorial(42)
    return (sine-se, sine+se), (cosine-ce, cosine+ce)


def test_initial_position_field_coupling_exact_series_and_degenerate_nonzero_interval():
    ctx = t.Context(item(varying=True))
    s, ex = 1./1024, .25
    result = enclose(ctx, lo=s, hi=s, momentum=1., position_error_mm=ex)
    sf, ef, k = F.from_float(s), F(1, 4), F.from_float(.299792458)
    phase = -k*F(1, 32)*ef*sf
    sin, cos = trig_bounds(phase)
    for y in (sin[0], sin[1]):
        for c in (cos[0], cos[1]):
            contains(result.endpoint_direction, 0, [c, -y, F(0)])
            contains(result.endpoint_position, 0, [sf*y/phase, sf*(c-1)/phase, ef])
    assert result.model.direction[0] >= float(k*F(1, 32)*ef*sf)
    assert result.lo_mm[0] == result.hi_mm[0] > 0


@pytest.mark.parametrize('pdg,q', [(13, -1), (-13, 1)])
def test_genuinely_varying_quadratic_phase_direction_oracle(pdg, q):
    ctx = t.Context(item(varying=True))
    result = enclose(ctx, direction=np.array([[3., 0., 4.]]), momentum=1., pdg=pdg,
        lo=.125, hi=.5, position_error_mm=.125, direction_error=.1)
    k, g, z0, uz = F.from_float(.299792458), F(1, 32), F(1, 8), F(4, 5)
    for s in (F(1, 8), F(1, 4), F(1, 2)):
        phase = q*k*g*(z0*s+uz*s*s/2)
        sine, cosine = trig_bounds(phase)
        for a in sine:
            for b in cosine:
                contains(result.direction, 0, [F(3, 5)*b, -F(3, 5)*a, uz])
                if s == F(1, 2):
                    contains(result.endpoint_direction, 0, [F(3, 5)*b, -F(3, 5)*a, uz])
        contains(iv.Interval(result.position.lo[:, 2:], result.position.hi[:, 2:]), 0, [z0+uz*s])


def test_independent_rational_direction_perturbation_straight_path():
    result = enclose(t.Context(item()), direction=np.array([[1., 0., 0.]]),
        position_error_mm=.125, direction_error=1.)
    for s in (F(0), F(1, 8), F(1, 2)):
        contains(result.position, 0, [s*F(3, 5), s*F(4, 5), F(1, 8)])
    contains(result.endpoint_position, 0, [F(3, 10), F(2, 5), F(1, 8)])
    contains(result.endpoint_direction, 0, [F(3, 5), F(4, 5), F(0)])
