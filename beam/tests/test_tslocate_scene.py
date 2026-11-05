"""GDML point-owner integration, not native crossing validation."""

import json

import numpy as np

from beamfastsim import tsgeom, tslocate
from test_tsgeom import fixture


def test_export_scene_preserves_exact_material_placement_and_receipt_indices(tmp_path):
    scene = tsgeom.load(fixture(tmp_path), ['Mother'], vacuum_materials=['vac'])
    points = np.array([[110, 20, 30], [111.5, 20, 30], [112, 20, 30], [120, 20, 30]])
    out = scene.locate(points)
    np.testing.assert_array_equal(out.status, [tslocate.VACUUM, tslocate.MATERIAL,
                                             tslocate.BOUNDARY, tslocate.OUTSIDE_SCOPE])
    assert scene.partition.volumes[1].physical == 'insertPlacement'
    assert scene.partition.volumes[1].material == scene.receipt['placed_volumes'][1]['material']
    np.testing.assert_allclose(out.clearance, [1, .5, 0, 0])


def test_cli_locations_are_explicitly_numerical_not_native_or_material_transport(tmp_path):
    path = fixture(tmp_path)
    out = tmp_path / 'receipt.json'
    tsgeom.main([str(path), '--root', 'Mother', '--vacuum-material', 'vac', '--out', str(out),
                 '--point', '110', '20', '30', '--point', '111.5', '20', '30',
                 '--point', '112', '20', '30'])
    receipt = json.loads(out.read_text())
    loc = receipt['numerical_locations']
    assert loc['status'] == ['vacuum', 'material', 'boundary_unresolved']
    assert loc['material'] == ['vac0xabc', 'metal', None]
    assert loc['volume_index'] == [0, 1, -1]
    assert loc['boundary_index'] == [-1, -1, 1]
    assert not loc['native_validation']
    assert not receipt['material_transport']
    assert not receipt['native_query_certified']


def test_duplicate_scope_aliases_rejected_instead_of_manufacturing_overlap(tmp_path):
    import pytest
    with pytest.raises(ValueError, match='duplicate selected physical instance'):
        tsgeom.load(fixture(tmp_path), ['Mother', 'VacuumPlacement'], vacuum_materials=['vac'])


def test_unnamed_and_repeated_placements_keep_instance_path_and_copy_identity(tmp_path):
    path = fixture(tmp_path)
    text = path.read_text().replace('name="insertPlacement"', 'copynumber="3"')
    needle = '<physvol copynumber="3"><volumeref ref="Daughter"/><position x="0" y="0" z="0" unit="mm"/></physvol>'
    text = text.replace(needle, needle + '<physvol copynumber="4"><volumeref ref="Daughter"/><position z="6"/></physvol>')
    path.write_text(text)
    scene = tsgeom.load(path, ['Mother'], vacuum_materials=['vac'])
    leaves = scene.partition.volumes[1:]
    assert len(leaves) == 2
    assert all(v.physical is None and v.logical == 'Daughter' for v in leaves)
    assert leaves[0].instance != leaves[1].instance
    records = scene.receipt['placed_volumes'][1:]
    assert [v['copy_number']['attribute'] for v in records] == ['3', '4']
    for volume, record in zip(leaves, records):
        assert volume.instance == record['instance']
        assert volume.instance_path == tuple(a['instance'] for a in record['path'])
        assert record['parent_instance'] == scene.partition.volumes[0].instance
