"""Restricted export compilation checks; not Geant4 native query validation."""

import json

import numpy as np
import pytest

from beamfastsim import tsgeom


def fixture(tmp_path, *, root_rotation='', daughter_rotation='', daughter_material='metal',
            root_position='<position x="10" y="20" z="30" unit="mm"/>',
            daughter_position='<position x="0" y="0" z="0" unit="mm"/>',
            daughter_solid='<tube name="insert" rmin="1" rmax="2" z="4" startphi="0" deltaphi="360" aunit="deg" lunit="mm"/>',
            daughter_children='', extra_solids='', extra_defines='', nested='', parent_rotation=''):
    path = tmp_path / 'scene.gdml'
    path.write_text(f'''<gdml><define>{extra_defines}</define><materials>
      <material name="vac0xabc"/><material name="metal"/></materials><solids>
      <box name="worldShape" x="1000" y="1000" z="1000"/>
      <tube name="motherShape" rmin="0" rmax="5" z="20" startphi="0" deltaphi="360" aunit="deg" lunit="mm"/>
      {daughter_solid}{extra_solids}</solids><structure>
      <volume name="World"><materialref ref="vac0xabc"/><solidref ref="worldShape"/>
        <physvol name="hallPlacement"><volumeref ref="Hall"/>{parent_rotation}<position x="100" unit="mm"/></physvol></volume>
      <volume name="Hall"><materialref ref="vac0xabc"/><solidref ref="worldShape"/>
        <physvol name="VacuumPlacement"><volumeref ref="Mother"/>{root_position}{root_rotation}</physvol></volume>
      <volume name="Mother"><materialref ref="vac0xabc"/><solidref ref="motherShape"/>
        <physvol name="insertPlacement"><volumeref ref="Daughter"/>{daughter_position}{daughter_rotation}</physvol></volume>
      <volume name="Daughter"><materialref ref="{daughter_material}"/><solidref ref="insert"/>{daughter_children}</volume>
      {nested}</structure><setup><world ref="World"/></setup></gdml>''')
    return path


def load(path):
    return tsgeom.load(path, ['VacuumPlacement'], vacuum_materials=['vac'])


def test_ancestry_translation_all_material_daughters_and_no_fitted_shift(tmp_path):
    scene = load(fixture(tmp_path))
    center = np.array([110, 20, 30])
    q = np.array([[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [0, 0, 11]])
    np.testing.assert_allclose(scene.clearance(q + center), [1, -0.5, 1, -1])
    receipt = scene.receipt
    assert len(receipt['placed_volumes']) == 2
    np.testing.assert_array_equal(receipt['placed_volumes'][0]['center_world_mm'], center)
    assert receipt['placed_volumes'][1]['material'] == 'metal'
    assert not receipt['material_transport']
    assert not receipt['native_query_certified']
    assert 'world frame' in receipt['coordinates']
    json.dumps(receipt, allow_nan=False)


def test_vacuum_daughter_coalesces_without_false_axis_or_disk_seam(tmp_path):
    scene = load(fixture(tmp_path, daughter_material='vac0xabc'))
    np.testing.assert_allclose(scene.clearance([[110, 20, 30]]), [5])
    assert all(v['vacuum'] for v in scene.receipt['placed_volumes'])


def test_physical_rotation_is_inverse_frame_rotation_not_boolean_rotation(tmp_path):
    angle = '<rotation y="90" unit="deg"/>'
    scene = load(fixture(tmp_path, root_rotation=angle,
                         daughter_position='<position x="0" y="0" z="3" unit="mm"/>'))
    records = scene.receipt['placed_volumes']
    np.testing.assert_allclose(records[0]['rotation_local_to_world'], [[0, 0, -1], [0, 1, 0], [1, 0, 0]], atol=1e-15)
    np.testing.assert_allclose(records[1]['center_world_mm'], [107, 20, 30], atol=1e-14)
    np.testing.assert_allclose(scene.clearance([[116, 20, 30], [110, 20, 36]]), [4, -1], atol=1e-14)


def test_multiaxis_ancestor_composition_has_stated_euler_order(tmp_path):
    scene = load(fixture(tmp_path, parent_rotation='<rotation x="90" y="90" z="0" unit="deg"/>'))
    expected_r = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    expected_t = [70, 10, -20]
    np.testing.assert_allclose(scene.receipt['placed_volumes'][0]['rotation_local_to_world'], expected_r, atol=1e-15)
    np.testing.assert_allclose(scene.receipt['placed_volumes'][0]['center_world_mm'], expected_t, atol=1e-14)


def test_boolean_rotation_is_direct_and_first_operand_transform_retained(tmp_path):
    operands = '''<box name="a" x="2" y="8" z="4"/><box name="b" x="2" y="4" z="2"/>'''
    subtraction = '''<subtraction name="insert"><first ref="a"/><second ref="b"/>
      <position x="2"/><rotation z="90" unit="deg"/>
      <firstposition x="1"/></subtraction>'''
    scene = load(fixture(tmp_path, daughter_solid=subtraction, extra_solids=operands))
    q = np.array([[1., 3, 0], [1, 0, 0], [3, 0, 0]])
    native = scene.region.solids[0].second.solids[0].solid
    np.testing.assert_allclose(native.clearance(q), [1, -1, -1], atol=1e-14)


def test_shifted_circle_strip_aperture_is_intersection_not_circle(tmp_path):
    pieces = '''<tube name="outer" rmin="0" rmax="5" z="10" startphi="0" deltaphi="360" aunit="deg"/>
      <tube name="circle" rmin="0" rmax="3" z="12" startphi="0" deltaphi="360" aunit="deg"/>
      <box name="strip" x="8" y="2" z="12"/>
      <intersection name="hole"><first ref="circle"/><second ref="strip"/></intersection>'''
    subtraction = '''<subtraction name="insert"><first ref="outer"/><second ref="hole"/><position y="1"/></subtraction>'''
    scene = load(fixture(tmp_path, daughter_solid=subtraction, extra_solids=pieces))
    q = np.array([[0, 1, 0], [0, 2.5, 0], [2, 1, 0]]) + [110, 20, 30]
    assert scene.clearance(q)[0] > 0
    assert scene.clearance(q)[1] < 0
    assert scene.clearance(q)[2] > 0


def test_torus_sector_and_nonconvex_extrusion_compile(tmp_path):
    torus = '<torus name="insert" rtor="3" rmin="0" rmax="1" startphi="0" deltaphi="90" aunit="deg"/>'
    scene = load(fixture(tmp_path, daughter_solid=torus))
    assert scene.clearance([[112, 22, 30]])[0] < 0
    assert scene.clearance([[108, 18, 30]])[0] > 0
    extrusion = '''<xtru name="insert" lunit="mm"><twoDimVertex x="0" y="0"/>
      <twoDimVertex x="4" y="0"/><twoDimVertex x="4" y="1"/><twoDimVertex x="2" y="1"/>
      <twoDimVertex x="2" y="3"/><twoDimVertex x="0" y="3"/>
      <section zOrder="0" zPosition="-2" xOffset="0" yOffset="0" scalingFactor="1"/>
      <section zOrder="1" zPosition="2" xOffset="0" yOffset="0" scalingFactor="1"/></xtru>'''
    scene = load(fixture(tmp_path, daughter_solid=extrusion))
    values = scene.clearance(np.array([[1, 1, 0], [3, 2, 0]]) + [110, 20, 30])
    np.testing.assert_allclose(values, [-1, 1])


@pytest.mark.parametrize('kwargs,message', [
    ({'root_position': '<position x="4" unit="cm"/>'}, 'mm'),
    ({'root_position': '<position x="a+1"/>'}, 'numeric'),
    ({'root_rotation': '<scale x="2"/>'}, 'scale'),
    ({'root_rotation': '<rotation y="1" unit="mrad"/>'}, 'deg or rad'),
    ({'daughter_solid': '<futureShape name="insert"/>'}, 'unsupported'),
    ({'daughter_solid': '<tube name="insert" rmax="1" z="2" deltaphi="360" aunit="deg" lunit="cm"/>'}, 'mm'),
    ({'daughter_children': '<physvol><volumeref ref="Nested"/></physvol>',
      'nested': '<volume name="Nested"><materialref ref="vac0xabc"/><solidref ref="worldShape"/></volume>'}, 'leaf'),
])
def test_fail_closed_unsupported(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        load(fixture(tmp_path, **kwargs))


def test_explicit_roots_material_allowlist_and_cli_aliases(tmp_path):
    path = fixture(tmp_path)
    for roots, materials in (([], ['vac']), (['Mother'], []), (['Mother'], 'vac'), (['Mother'], ['metal'])):
        with pytest.raises(ValueError):
            tsgeom.load(path, roots, vacuum_materials=materials)
    out = tmp_path / 'receipt.json'
    tsgeom.main([str(path), '--root', 'Mother', '--vacuum-material', 'vac', '--out', str(out)])
    assert json.loads(out.read_text())['material_transport'] is False
    tsgeom.main([str(path), '--root', 'Mother', '--vacuum-material', 'vac', '--out', str(out),
                 '--point', '110', '20', '30', '--point', '111.5', '20', '30'])
    queries = json.loads(out.read_text())['numerical_clearance_queries']
    np.testing.assert_allclose(queries['clearance_mm'], [1, -0.5])
    assert not queries['native_validation']
    alias = tmp_path / 'alias.json'
    alias.symlink_to(path)
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        tsgeom.main([str(path), '--root', 'Mother', '--vacuum-material', 'vac', '--out', str(alias)])
    assert path.read_bytes() == before


def test_exact_material_reference_allowlist_and_ambiguous_alias(tmp_path):
    path = fixture(tmp_path, daughter_material='vac0xabc')
    scene = tsgeom.load(path, ['Mother'], vacuum_materials=['vac0xabc'])
    assert all(v['vacuum'] for v in scene.receipt['placed_volumes'])
    path.write_text(path.read_text().replace('<material name="metal"/>', '<material name="metal"/><material name="vac0xdef"/>'))
    with pytest.raises(ValueError, match='ambiguous'):
        tsgeom.load(path, ['Mother'], vacuum_materials=['vac'])
    scene = tsgeom.load(path, ['Mother'], vacuum_materials=['vac0xabc'])
    assert all(v['vacuum'] for v in scene.receipt['placed_volumes'])


def test_boolean_unsupported_attributes_are_not_ignored(tmp_path):
    boolean = '<subtraction name="insert" offset="symbol"><first ref="a"/><second ref="b"/></subtraction>'
    path = fixture(tmp_path, daughter_solid=boolean,
                   extra_solids='<box name="a" x="2" y="2" z="2"/><box name="b" x="1" y="1" z="1"/>')
    with pytest.raises(ValueError, match='Boolean solid attributes'):
        load(path)
