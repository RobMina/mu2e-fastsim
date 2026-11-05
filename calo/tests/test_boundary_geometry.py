"""A positive passive-material coverage counterexample is not inferred from response."""
import numpy as np
import pytest

from test_controls import load_script


def fixture():
    script = load_script("audit_boundary_geometry")
    config = {name: 0. for name in script.REQUIRED}
    config.update({"calorimeter.hasCrates": True, "ds.hasCableRunCal": True,
                   "calorimeter.caloDiskRadiusOut": 719.,
                   "calorimeter.vdThickness": .02, "calorimeter.crateZLength": 350.,
                   "calorimeter.crateFShieldDeltaZ": 102., "calorimeter.crateFShieldThickness": 8.,
                   "calorimeter.FEBToDiskZOffset": 35., "ds.CableRunCal.UpRin": 721.,
                   "ds.CableRunCal.UpRout": 821., "ds.CableRunCal.dPhi": 13.})
    disks = dict(disk=np.array([0, 1]), origin_z=np.array([12000., 12700.]), size_z=np.array([384.6, 384.6]))
    for axis in "xyz":
        for component in "xyz":
            disks[f"axis_{axis}_{component}"] = np.full(2, float(axis == component))
    vd = dict(id=np.array([73, 74, 75, 76, 77, 78, 79, 80, 102, 103, 104, 105, 106, 107, 108, 109]), position_z=np.zeros(16))
    return config, disks, vd


def test_uncovered_cable_ray_is_explicit_topology_not_data_selection():
    report = load_script("audit_boundary_geometry").describe(*fixture())
    witness = report["uncovered_cable_entry_witness"]
    assert witness["local_phi_degrees"] == 90
    assert witness["local_radius_inside_cable_mm"] == 771
    assert witness["gap_interval_outside_all_disk_and_feb_shells_mm"][1] > witness["gap_interval_outside_all_disk_and_feb_shells_mm"][0]
    assert report["existing_disk_feb_vds_cover_all_calo_passive_material"] is False


@pytest.mark.parametrize("change", ["disabled", "rotated", "no_gap", "missing_vd"])
def test_source_specific_witness_refuses_unsupported_geometry(change):
    config, disks, vd = fixture()
    if change == "disabled":
        config["ds.hasCableRunCal"] = False
    elif change == "rotated":
        disks["axis_x_x"][0] = 0
    elif change == "no_gap":
        disks["origin_z"][1] = 12100
    else:
        vd = {name: value[:-1] for name, value in vd.items()}
    with pytest.raises(ValueError):
        load_script("audit_boundary_geometry").describe(config, disks, vd)


def test_resolved_config_must_be_complete_and_unambiguous(tmp_path):
    script = load_script("audit_boundary_geometry")
    path = tmp_path / "geometry.txt"
    path.write_text("Geom: double calorimeter.vdThickness = 0.02;\n")
    with pytest.raises(ValueError, match="resolved"):
        script.read_config(path)
    lines = [f"Geom: double {name} = 1;\n" for name in script.REQUIRED]
    path.write_text("".join(lines))
    assert script.read_config(path)["calorimeter.vdThickness"] == 1
    path.write_text("".join(lines) + lines[0])
    with pytest.raises(ValueError, match="resolved"):
        script.read_config(path)


@pytest.mark.parametrize("change", ["nan", "overlap", "reversed"])
def test_invalid_cable_geometry_refused(change):
    config, disks, vd = fixture()
    if change == "nan":
        config["ds.CableRunCal.UpRout"] = np.nan
    elif change == "overlap":
        config["ds.CableRunCal.UpRin"] = 600
    else:
        config["ds.CableRunCal.UpRout"] = 700
    with pytest.raises(ValueError):
        load_script("audit_boundary_geometry").describe(config, disks, vd)
