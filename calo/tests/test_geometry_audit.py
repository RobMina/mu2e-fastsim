"""Geometry consistency does not imply historical equality or surface completeness."""
import numpy as np
import pytest

from test_controls import load_script


def vectors(table, prefix, values):
    table.update({f"{prefix}_{axis}": np.asarray(values)[:, index] for index, axis in enumerate("xyz")})


def fixture():
    crystals = {"run": np.array([1, 1]), "crystal": np.array([0, 1]), "disk": np.array([0, 0])}
    disks = {"run": np.array([1]), "disk": np.array([0])}
    vd = {"run": np.array([1]), "id": np.array([73]), "half_length": np.array([.01])}
    vectors(crystals, "position", [[3., 2., 4.], [4., 3., 4.]])
    vectors(crystals, "local_position", [[1., 0., 0.], [2., 1., 0.]])
    vectors(crystals, "size", [[1., 1., 1.], [1., 1., 1.]])
    vectors(disks, "origin", [[2., 2., 4.]])
    vectors(disks, "front_face_origin", [[2., 2., 3.]])
    vectors(disks, "size", [[10., 10., 10.]])
    vectors(vd, "position", [[2., 2., 3.]])
    for index, axis in enumerate("xyz"):
        value = np.eye(3)[index:index + 1]
        vectors(disks, f"axis_{axis}", value)
        vectors(vd, f"axis_{axis}", value)
    return crystals, disks, vd


def test_geometry_consistency_not_simulation_or_material_certificate():
    report = load_script("audit_geometry").audit(*fixture(), expected_crystals=2, expected_disks=1)
    assert report["frame_dimension_consistency"] is True
    assert report["max_crystal_transform_difference_mm"] == 0
    assert report["simulation_geometry_equality_certified"] is False
    assert report["material_boundary_coverage_certified"] is False


@pytest.mark.parametrize("change", ["count", "duplicate", "wrong_frame", "wrong_crystal", "unknown_disk", "negative_size", "bad_vd_thickness"])
def test_broken_geometry_refused(change):
    crystals, disks, vd = fixture()
    if change == "count":
        crystals = {name: value[:1] for name, value in crystals.items()}
    elif change == "duplicate":
        crystals["crystal"][1] = 0
    elif change == "wrong_frame":
        disks["axis_x_x"][0] = 2
    elif change == "wrong_crystal":
        crystals["position_y"][0] += 1
    elif change == "unknown_disk":
        crystals["disk"][0] = 1
    elif change == "negative_size":
        crystals["size_x"][0] = -1
    else:
        vd["half_length"][0] = 0
    with pytest.raises(ValueError):
        load_script("audit_geometry").audit(crystals, disks, vd, expected_crystals=2, expected_disks=1)
