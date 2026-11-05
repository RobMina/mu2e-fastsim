"""Conditional isolated-face entry tests, not native or material transport."""

import numpy as np
import pytest

from beamfastsim import tscarry, tsfield, tslocate, tsnav, tssolid, tssurface


def geometry(solid=None, *, rotation=None, extra=()):
    solid = tsnav.Box([2., 2., .125]) if solid is None else solid
    rotation = np.eye(3) if rotation is None else rotation
    mother = tslocate.Volume(tsnav.Box([100., 100., 100.]), 'vac', 'mother', 'cell', True)
    leaf = tslocate.Volume(tsnav.Placed(solid, [0., 0., 0.], rotation), 'Ti6Al4V', None,
        'cell', False, instance=2, logical='window', instance_path=(1, 2))
    return tslocate.Partition((tslocate.Cell(mother, (leaf, *extra)),))


def segment(x, direction=(0., 0., 1.), length=.0625, *, ex=0., eu=0., evalx=0., evalu=0., field=(0., 0., 0.)):
    x = np.atleast_2d(x)
    header = tsfield.GridHeader('synthetic', [-200., -200., -200.], [0., 0., 0.], [2, 2, 2], [400., 400., 400.])
    grid = tsfield.FieldMap(header, np.broadcast_to(field, (2, 2, 2, 3)))
    context = tscarry.MapContext(grid)
    state = tscarry.State(context, x, np.broadcast_to(direction, x.shape), 100.,
        np.resize([13, -13, 211, -211], len(x)), position_error_mm=ex, direction_error=eu)
    return tscarry.Segment(state, length, field_error_tesla=0., position_error_mm=evalx, direction_error=evalu)


def call(seg, face, **kwargs):
    return tssurface.entry(seg, face, plane_error_mm=kwargs.pop('plane_error_mm', 0.),
        projection_error=kwargs.pop('projection_error', 0.), geometry_error_mm=kwargs.pop('geometry_error_mm', 0.), **kwargs)


def bottom(partition):
    return next(f for f in tssurface.faces(partition, 1) if f.key == 'z-')


def test_exact_entry_batched_charge_identity_and_no_state_advance():
    part = geometry()
    face = bottom(part)
    seg = segment(np.tile([0., 0., -.15625], (4, 1)))
    original = seg.initial.x.copy()
    result = call(seg, face, path_atol=1e-10)
    assert np.all(result.status == tssurface.ISOLATED)
    assert np.all(result.unique_entry)
    assert np.all(result.lo_mm <= .03125) and np.all(result.hi_mm >= .03125)
    assert np.all(result.hi_mm-result.lo_mm < 1e-10)
    assert result.plane_root_arithmetic_outward
    np.testing.assert_array_equal(seg.initial.x, original)
    np.testing.assert_array_equal(result.source_index, seg.initial.source_index)
    assert result.face.volume is part.volumes[1]
    assert result.face.volume.instance_path == (1, 2)
    assert result.face.volume.physical is None
    assert not result.material_transport and not result.floating_point_certified and not result.native_query_certified
    for name in ('status', 'unique_entry', 'source_index', 'lo_mm', 'iterations'):
        with pytest.raises(ValueError):
            getattr(result, name).setflags(write=True)


def test_nonzero_errors_root_floor_does_not_false_complete():
    result = call(segment([0., 0., -.15625], ex=2**-16, eu=2**-12, evalx=2**-17, evalu=2**-15),
        bottom(geometry()), plane_error_mm=2**-18, projection_error=2**-16, geometry_error_mm=2**-19, path_atol=1e-9)
    assert result.status[0] == tssurface.UNCERTAINTY_LIMIT
    assert result.unique_entry[0]
    assert result.lo_mm[0] < .03125 < result.hi_mm[0]
    assert result.hi_mm[0]-result.lo_mm[0] > 1e-9
    wider = call(segment([0., 0., -.15625], ex=2**-16), bottom(geometry()), path_atol=.001)
    assert wider.status[0] == tssurface.ISOLATED


def test_oblique_incidence_exact_root_and_tangent_refusal():
    face = bottom(geometry())
    out = call(segment([0., 0., -.15625], direction=[.6, 0, .8], length=.078125), face)
    assert out.unique_entry[0]
    assert out.lo_mm[0] <= .0390625 <= out.hi_mm[0]
    tangent = call(segment([0., 0., -.15625], direction=[1., 0, 0]), face)
    assert tangent.status[0] == tssurface.TRANSVERSE_PENDING
    reverse = call(segment([0., 0., -.15625], direction=[0., 0, -1]), face)
    assert not reverse.unique_entry[0]
    uncertain = call(segment([0., 0., -.15625], eu=1.), face)
    assert uncertain.status[0] == tssurface.TRANSVERSE_PENDING


@pytest.mark.parametrize('x,reason', [([0., 0., -.125], tssurface.SIGN_PENDING),
    ([0., 0., -.0625], tssurface.SIGN_PENDING), ([0., 0., -.5], tssurface.SIGN_PENDING),
    ([2., 0., -.15625], tssurface.GUARD_PENDING)])
def test_boundary_initial_material_missing_signs_and_rim(x, reason):
    out = call(segment(x), bottom(geometry()))
    assert out.status[0] == reason
    assert not out.unique_entry[0]
    assert out.lo_mm[0] == out.hi_mm[0] == 0


def test_opposite_face_and_competing_material_guard():
    out = call(segment([0., 0., -.15625], length=.5), bottom(geometry()))
    assert out.status[0] == tssurface.GUARD_PENDING
    extra = tslocate.Volume(tsnav.Box([.01, 1., 1.]), 'metal', 'overlap', 'cell', False)
    out = call(segment([0., 0., -.15625]), bottom(geometry(extra=(extra,))))
    assert out.status[0] == tssurface.GUARD_PENDING
    part = geometry()
    other = tslocate.Cell(tslocate.Volume(tsnav.Box([1., 1., 1.]), 'vac', 'other', 'other', True))
    out = call(segment([0., 0., -.15625]), bottom(tslocate.Partition((*part.cells, other))))
    assert out.status[0] == tssurface.GUARD_PENDING


def test_tube_caps_full_sector_hole_and_rim():
    shape = tssolid.Sector(tsnav.Tube(2., .125, .5), 0., 2*np.pi)
    part = geometry(shape)
    assert len(tssurface.faces(part, 1)) == 2
    face = bottom(part)
    assert call(segment([1., 0., -.15625]), face).unique_entry[0]
    for x in (0., .5, 2.):
        assert call(segment([x, 0., -.15625]), face).status[0] == tssurface.GUARD_PENDING
    assert call(segment([1., 0., -.15625]), face, geometry_error_mm=1.).status[0] == tssurface.GUARD_PENDING


def test_rotated_face_and_high_curvature_refusal():
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    face = bottom(geometry(rotation=rotation))
    np.testing.assert_array_equal(face.inward_normal, [1., 0., 0.])
    assert call(segment([-.15625, 0., 0.], direction=[1., 0., 0.]), face).unique_entry[0]
    out = call(segment([0., 0., -.15625], field=[1e4, 0., 0.]), bottom(geometry()))
    assert not out.unique_entry[0]


def test_iteration_limit_keeps_original_unique_interval():
    result = call(segment([0., 0., -.15625]), bottom(geometry()), max_iterations=0, path_atol=1e-9)
    assert result.status[0] == tssurface.ITERATION_LIMIT
    assert result.unique_entry[0]
    assert result.lo_mm[0] == 0 and result.hi_mm[0] == .0625
    assert result.iterations[0] == 0


@pytest.mark.parametrize('solid', [tsnav.Sphere(1.), tsnav.Difference(tsnav.Box([1, 1, 1]), tsnav.Sphere(.5)),
    tssolid.Sector(tsnav.Tube(2., 1.), 0., np.pi)])
def test_unsupported_surfaces_fail_without_guessing(solid):
    with pytest.raises(ValueError, match='complete box'):
        tssurface.faces(geometry(solid), 1)


@pytest.mark.parametrize('kwargs', [{'path_atol': 0}, {'max_iterations': True}, {'max_iterations': -1},
    {'plane_error_mm': -1}, {'projection_error': np.nan}, {'geometry_error_mm': [0, 1]}])
def test_invalid_controls_fail(kwargs):
    with pytest.raises(ValueError):
        call(segment([0., 0., -.15625]), bottom(geometry()), **kwargs)


def test_invalid_leaf_and_zero_length():
    part = geometry()
    for idx in (0, -1, 8, True):
        with pytest.raises(ValueError):
            tssurface.faces(part, idx)
    with pytest.raises(ValueError, match='positive'):
        call(segment([0., 0., -.15625], length=0.), bottom(part))


def test_interleaved_active_subsets_match_independent_rows():
    points = np.array([[0., 0., -.15625], [2., 0., -.15625], [0., 0., -.5],
                       [0., 0., -.1484375], [0., 0., -.15625], [0., 0., -.140625]])
    directions = np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.],
                           [0., 0., 1.], [1., 0., 0.], [0., 0., 1.]])
    errors = np.array([0., 0., 0., 0., 0., 2**-16])
    face = bottom(geometry())
    batch = call(segment(points, directions, ex=errors), face, path_atol=1e-9)
    for i in range(len(points)):
        single = call(segment(points[i], directions[i], ex=errors[i]), face, path_atol=1e-9)
        for key in ('status', 'unique_entry', 'lo_mm', 'hi_mm', 'iterations'):
            np.testing.assert_array_equal(getattr(batch, key)[i], getattr(single, key)[0])


def test_all_box_faces_and_nested_placement_second_cell_identity():
    part = geometry()
    for face in tssurface.faces(part, 1):
        start = face.origin-face.inward_normal*.03125
        out = call(segment(start, face.inward_normal), face)
        assert out.unique_entry[0]
        assert out.lo_mm[0] <= .03125 <= out.hi_mm[0]
    rz = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    leaf = tslocate.Volume(tsnav.Placed(tsnav.Placed(tsnav.Box([1., 1., .125]), [1., 0., 0.], rx),
        [20., 0., 0.], rz), 'metal', 'nested', 'second', False)
    mother = tslocate.Volume(tsnav.Placed(tsnav.Box([5., 5., 5.]), [20., 0., 0.], np.eye(3)),
        'vac', 'second', 'second', True)
    first = tslocate.Cell(tslocate.Volume(tsnav.Box([5., 5., 5.]), 'vac', 'first', 'first', True))
    partition = tslocate.Partition((first, tslocate.Cell(mother, (leaf,))))
    face = next(f for f in tssurface.faces(partition, 2) if f.key == 'z-')
    assert face.mother_index == 1 and face.volume_index == 2
    expected_normal = (rz @ rx)[:, 2]
    np.testing.assert_array_equal(face.inward_normal, expected_normal)
    np.testing.assert_array_equal(face.origin, np.array([20., 1., 0.])-.125*expected_normal)
    out = call(segment(face.origin-face.inward_normal*.03125, face.inward_normal), face)
    assert out.unique_entry[0]


def test_independent_error_declarations_can_refuse_entry():
    face = bottom(geometry())
    for seg, kwargs, status in (
        (segment([0., 0., -.15625]), {'plane_error_mm': .1}, tssurface.SIGN_PENDING),
        (segment([0., 0., -.15625], evalx=.04), {}, tssurface.SIGN_PENDING),
        (segment([0., 0., -.15625]), {'projection_error': 1.}, tssurface.TRANSVERSE_PENDING),
        (segment([0., 0., -.15625], evalu=1.), {}, tssurface.TRANSVERSE_PENDING)):
        out = call(seg, face, **kwargs)
        assert out.status[0] == status and not out.unique_entry[0]


def test_uniform_curved_entry_encloses_independent_analytic_root():
    from beamfastsim.tsstep import K
    field, momentum = 1., 100.
    curvature = K*field/momentum
    root = np.arcsin(.03125*curvature)/curvature
    for direction_scale in (1., 7.):
        seg = segment(np.tile([0., 0., -.15625], (4, 1)), [0., 0., direction_scale],
                      field=[field, 0., 0.], evalx=1e-12, evalu=1e-12)
        out = call(seg, bottom(geometry()), plane_error_mm=1e-12, path_atol=1e-8)
        assert np.all(out.status == tssurface.ISOLATED)
        assert np.all(out.lo_mm <= root) and np.all(out.hi_mm >= root)


def test_faces_from_compiled_gdml_keep_export_identity(tmp_path):
    from beamfastsim import tsgeom
    from test_tsgeom import fixture
    path = fixture(tmp_path, root_position='<position x="0" y="0" z="0"/>',
        daughter_solid='<tube name="insert" rmin="0" rmax="2" z="0.25" startphi="0" deltaphi="360" aunit="deg"/>')
    scene = tsgeom.load(path, ['VacuumPlacement'], vacuum_materials=['vac'])
    face = bottom(scene.partition)
    assert face.volume.material == 'metal' and face.volume.physical == 'insertPlacement'
    assert face.volume.instance_path[-1] == face.volume.instance
    assert len(face.volume.instance_path) == 4
    np.testing.assert_array_equal(face.origin, [100., 0., -.125])
    result = call(segment([100., 0., -.15625]), face)
    assert result.unique_entry[0]


def test_nonzero_initial_uncertainty_contains_independent_straight_contacts():
    ex, eu = 2**-16, 2**-12
    seg = segment(np.tile([0., 0., -.15625], (4, 1)), ex=ex, eu=eu)
    out = call(seg, bottom(geometry()), path_atol=1e-12)
    shifts = np.array([-ex, -ex/2, ex/2, ex])
    ux = eu/2
    uz = np.sqrt(1-ux*ux)
    actual_roots = (.03125-shifts)/uz
    assert np.all(actual_roots >= out.lo_mm) and np.all(actual_roots <= out.hi_mm)
    assert np.all(out.status == tssurface.UNCERTAINTY_LIMIT)


def test_large_origin_stored_plane_sign_cancellation_never_false_exact_root():
    from beamfastsim import tsplane, tsinterval as iv
    center = np.array([1e16, -1e16, 0.])
    point = np.array([[1e16, -1e16, -.125]])
    normal = np.array([1., 1., 1.])
    result = tsplane.signed(iv.Interval(point), iv.Interval(center), iv.Interval(normal))
    assert result.lo[0] < 0 and result.hi[0] < 0
    assert result.lo[0] <= -.125/np.sqrt(3.) <= result.hi[0]
