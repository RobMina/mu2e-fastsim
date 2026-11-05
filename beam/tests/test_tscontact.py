"""First-contact set regressions; no postcontact vacuum continuation/restart."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from beamfastsim import tscontact, tscarry, tsinterval, tssurface
from test_tssurface import segment, bottom, geometry, call


def receipt(*, ex=2**-16, eu=2**-12, **kwargs):
    seg = segment(np.tile([0., 0., -.15625], (4, 1)), ex=ex, eu=eu, **kwargs)
    entry = call(seg, bottom(geometry()), path_atol=1e-12)
    return tscontact.contact(entry)


def test_nonzero_uncertainty_sets_contain_independent_contacts_and_incoming_directions():
    out = receipt()
    shift = np.array([-2**-16, -2**-17, 2**-17, 2**-16])
    ux = np.array([1., -1., 1., -1.])*2**-13
    uz = np.sqrt(1-ux*ux)
    roots = (.03125-shift)/uz
    true_x = np.column_stack((ux*roots, np.zeros(4), np.full(4, -.125)))
    assert np.all(np.linalg.norm(true_x-out.x_center_mm, axis=1) <= out.position_radius_mm)
    unit = out.direction_vector/np.linalg.norm(out.direction_vector, axis=1)[:, None]
    true_u = np.column_stack((ux, np.zeros(4), uz))
    assert np.all(np.linalg.norm(true_u-unit, axis=1) <= out.direction_radius)
    assert np.all(out.local_lo_mm <= roots) and np.all(out.local_hi_mm >= roots)
    assert out.face is out.entry.face
    assert isinstance(out.entry.segment, tscarry.Segment)
    assert not isinstance(out, tscarry.State)
    assert not out.restart_state and not out.material_transport and not out.floating_point_certified
    assert out.local_arithmetic_outward
    np.testing.assert_array_equal(out.momentum, 100.)
    assert np.all(out.x_center_mm[:, 2] < -.125)
    for value in (out.row_index, out.source_index, out.x_center_mm, out.position_radius_mm, out.direction_vector):
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_receipt_is_bound_to_original_segment_not_just_row_tags():
    first = call(segment([0., 0., -.15625]), bottom(geometry()))
    second = call(segment([0., 0., -.140625]), bottom(geometry()))
    a, b = tscontact.contact(first), tscontact.contact(second)
    assert a.entry is first and b.entry is second
    assert a.entry.segment is first.segment and a.entry.segment is not second.segment
    assert a.source_index[0] == b.source_index[0]
    assert a.local_lo_mm[0] != b.local_lo_mm[0]


def test_compact_unique_rows_and_capped_proof_retained():
    points = [[0., 0., -.15625], [2., 0., -.15625], [0., 0., -.5], [0., 0., -.140625]]
    entry = call(segment(points), bottom(geometry()), max_iterations=0, path_atol=1e-12)
    out = tscontact.contact(entry)
    np.testing.assert_array_equal(out.row_index, [0, 3])
    np.testing.assert_array_equal(out.entry_status, tssurface.ITERATION_LIMIT)
    np.testing.assert_array_equal(out.local_lo_mm, 0.)
    np.testing.assert_array_equal(out.local_hi_mm, .0625)
    assert len(out.entry.status) == 4
    empty = tscontact.contact(call(segment([[2., 0., -.15625]]), bottom(geometry())))
    assert empty.x_center_mm.shape == (0, 3) and len(empty.row_index) == 0


def test_exact_root_endpoint_is_not_projected_and_keeps_eval_errors():
    seg = segment([0., 0., -.15625], evalx=0., evalu=1e-8)
    out = tscontact.contact(call(seg, bottom(geometry()), path_atol=1e-12))
    assert out.local_lo_mm[0] <= .03125 <= out.local_hi_mm[0]
    assert out.local_hi_mm[0]-out.local_lo_mm[0] < 1e-12
    assert out.direction_radius[0] >= 1e-8
    assert out.position_radius_mm[0] >= 0
    assert out.center_plane_residual.lo[0] <= out.position_radius_mm[0]
    assert out.center_plane_residual.hi[0] >= -out.position_radius_mm[0]


def test_cumulative_path_outward_encloses_absorbed_tiny_increment():
    seg = segment([0., 0., -.15625])
    old = seg.initial
    initial = tscarry.State(old.context, old.x, old.direction, old.momentum, old.pdg,
        position_error_mm=old.position_error_mm, direction_error=old.direction_error, path_mm=2**54*1.)
    seg = tscarry.Segment(initial, .0625, field_error_tesla=0., position_error_mm=0., direction_error=0.)
    out = tscontact.contact(call(seg, bottom(geometry())))
    exact = Fraction(2**54)+Fraction(1, 32)
    assert Fraction.from_float(float(out.cumulative_path.lo[0])) <= exact <= Fraction.from_float(float(out.cumulative_path.hi[0]))
    assert out.cumulative_path.hi[0] > 2**54


def test_curved_contact_set_encloses_analytic_incoming_limit():
    from beamfastsim.tsstep import K
    curvature = K/100.
    root = np.arcsin(.03125*curvature)/curvature
    out = receipt(ex=0., eu=0., field=[1., 0., 0.], evalx=1e-12, evalu=1e-12)
    assert np.all(out.local_lo_mm <= root) and np.all(root <= out.local_hi_mm)
    charge = np.array([-1., 1., 1., -1.])
    y = charge*(1-np.cos(curvature*root))/curvature
    point = np.column_stack((np.zeros(4), y, np.full(4, -.125)))
    incoming = np.column_stack((np.zeros(4), charge*np.sin(curvature*root), np.full(4, np.cos(curvature*root))))
    nominal = out.direction_vector/np.linalg.norm(out.direction_vector, axis=1)[:, None]
    assert np.all(np.linalg.norm(point-out.x_center_mm, axis=1) <= out.position_radius_mm+1e-12)
    assert np.all(np.linalg.norm(incoming-nominal, axis=1) <= out.direction_radius+1e-12)


def test_cannot_silently_retag_or_supply_invalid_root_interval():
    entry = call(segment([0., 0., -.15625]), bottom(geometry()))
    for modified in (replace(entry, source_index=np.array([9])), replace(entry, lo_mm=np.array([-.1])),
                     replace(entry, hi_mm=np.array([1.])), replace(entry, guard_margin_mm=np.array([0.]))):
        with pytest.raises(ValueError):
            tscontact.contact(modified)
    with pytest.raises(ValueError):
        tscontact.contact(entry.segment)


def test_contact_zero_and_adjacent_width_do_not_erase_supplied_errors():
    entry = call(segment([0., 0., -.15625], ex=1e-8, eu=1e-8, evalx=2e-8, evalu=2e-8), bottom(geometry()))
    for hi in (.03125, np.nextafter(.03125, np.inf)):
        narrow = replace(entry, lo_mm=np.array([.03125]), hi_mm=np.array([hi]))
        out = tscontact.contact(narrow)
        exact_width = Fraction.from_float(hi)-Fraction(1, 32)
        exact_radius = Fraction.from_float(1e-8)+Fraction.from_float(2e-8)+exact_width
        assert Fraction.from_float(float(out.position_radius_mm[0])) >= exact_radius
        assert out.direction_radius[0] >= 1e-8+2e-8


def test_plane_representation_slab_and_original_error_declarations_preserved():
    seg = segment([0., 0., -.15625], ex=1e-6, eu=1e-8)
    face = bottom(geometry())
    entry = call(seg, face, plane_error_mm=1e-5, projection_error=1e-8, geometry_error_mm=1e-7,
                 path_atol=1e-4, max_iterations=12)
    out = tscontact.contact(entry)
    assert out.entry is entry and entry.segment is seg
    assert entry.path_atol_mm == 1e-4 and entry.max_iterations == 12
    np.testing.assert_array_equal(entry.plane_error_mm, [1e-5])
    np.testing.assert_array_equal(entry.projection_error, [1e-8])
    np.testing.assert_array_equal(entry.geometry_error_mm, [1e-7])
    np.testing.assert_array_equal(out.intended_plane_slab.lo, [-1e-5])
    np.testing.assert_array_equal(out.intended_plane_slab.hi, [1e-5])
    true_shift = 5e-6
    root = .03125+true_shift
    point = np.array([[0., 0., -.125+true_shift]])
    assert out.local_lo_mm[0] <= root <= out.local_hi_mm[0]
    assert np.linalg.norm(point-out.x_center_mm) <= out.position_radius_mm[0]
    assert out.intended_plane_slab.lo[0] <= true_shift <= out.intended_plane_slab.hi[0]


def test_capped_contact_direction_diameter_uses_mathematical_unit_center():
    seg = segment([0., 0., -.15625], direction=[0., 0., 7.], eu=.125,
                  field=[20., 0., 0.], evalu=.125)
    entry = call(seg, bottom(geometry()), max_iterations=0)
    if not entry.unique_entry[0]:
        pytest.fail('chosen capped contact fixture lost its first-entry proof')
    out = tscontact.contact(entry)
    assert np.linalg.norm(out.direction_vector[0]) > 0
    assert out.direction_radius[0] <= 2
    np.testing.assert_array_equal(out.entry_status, tssurface.ITERATION_LIMIT)
