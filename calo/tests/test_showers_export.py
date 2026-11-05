"""Minimal numeric-export fixture exercises persistent identity and frame mapping."""
import numpy as np
import pytest
import uproot

from calofastsim.showers.export import read


def tables():
    event = dict(run=np.ones(3, dtype=np.int32), subrun=np.zeros(3, dtype=np.int32), event=np.arange(1, 4, dtype=np.int32))
    rows = dict(events={**event, "g4_status": np.zeros(3, dtype=np.int32), "n_showers": np.array([2, 0, 1])},
                generated={**event, "pdg": np.full(3, 11), "parent_key": np.full(3, 10), "parent_valid": np.ones(3, bool)},
                products={**event, "kind": np.full(3, -1), "product_id": np.full(3, 100),
                          "branch_name": np.array(["mu2eSimParticles_g4run__CaloLibraryCE."] * 3)},
                particles={**event, "sim_product_id": np.full(3, 100), "sim_key": np.full(3, 20), "pdg": np.full(3, 11),
                           "parent_product_id": np.full(3, 100), "parent_key": np.full(3, 10), "parent_valid": np.ones(3, bool)})
    rows["steps"] = dict(run=np.ones(5, int), subrun=np.zeros(5, int), event=np.array([1, 1, 1, 2, 3]),
                         sim_product_id=np.full(5, 100), sim_key=np.array([20, 20, 99, 20, 99]),
                         kind=np.zeros(5, int), volume=np.array([73, 75, 73, 73, 73]), time=np.array([10., 12., 9., 20., 30.]),
                         position_x=np.full(5, 5.), position_y=np.zeros(5), position_z=np.array([100., 200., 100., 100., 100.]),
                         momentum_x=np.zeros(5), momentum_y=np.zeros(5), momentum_z=np.full(5, 100.))
    disks = dict(run=np.ones(2, int), disk=np.array([0, 1]))
    for axis in "xyz":
        disks["origin_" + axis] = np.array([100., 200.]) if axis == "z" else np.zeros(2)
        disks["size_" + axis] = np.full(2, 20.)
        for coord in "xyz":
            disks["axis_" + axis + "_" + coord] = np.full(2, float(axis == coord))
    rows["disks"] = disks
    crystals = dict(run=np.ones(2, int), disk=np.array([0, 1]), crystal=np.array([10, 11]))
    for axis in "xyz":
        crystals["position_" + axis] = np.array([100., 200.]) if axis == "z" else np.zeros(2)
        crystals["size_" + axis] = np.full(2, 20.)
    rows["crystals"] = crystals
    rows["virtualdetectors"] = dict(run=np.ones(2, int), id=np.array([73, 75]), position_x=np.zeros(2),
                                    position_y=np.zeros(2), position_z=np.array([100., 200.]))
    rows["showers"] = dict(run=np.ones(3, int), subrun=np.zeros(3, int), event=np.array([1, 1, 3]), crystal=np.array([10, 11, 10]),
                           energy_g4=np.array([40., 30., 20.]), energy_birks=np.array([39., 29., 19.]), time=np.array([11., 13., 31.]),
                           position_x=np.full(3, 5.), position_y=np.zeros(3), position_z=np.full(3, 10.))
    return rows


def write(path, data):
    with uproot.recreate(path) as output:
        for name, columns in data.items():
            output.mktree("dump/" + name, columns)


def test_numeric_export(tmp_path):
    path = tmp_path / "export.root"
    write(path, tables())
    entries, css, geometry, census, total = read(path)
    np.testing.assert_array_equal(entries["event_row"], [0, 1])
    np.testing.assert_array_equal(entries["time"], [10, 20])
    np.testing.assert_array_equal(css["entry"], [0, 0])
    np.testing.assert_array_equal(css["disk"], [0, 1])
    np.testing.assert_allclose(css["position"], [[5, 0, 110], [5, 0, 210]])
    np.testing.assert_array_equal(total, [68, 0, 19])
    assert census == dict(events=3, front_entry_events=2, empty_css_events=1, non_front_entry_events=1,
                          non_front_entry_positive_events=1, css_rows=3, retained_css_rows=2)
    np.testing.assert_array_equal(geometry["vd_id"], [73, 75])
    np.testing.assert_array_equal(geometry["disk_size_x"], [20, 20])


def test_reverse_crossing_not_an_entry(tmp_path):
    data = tables()
    data["steps"]["momentum_z"][0] = -100
    path = tmp_path / "reverse.root"
    write(path, data)
    entries, _, _, _, _ = read(path)
    np.testing.assert_array_equal(entries["disk"], [1, 0])
    np.testing.assert_array_equal(entries["time"], [12, 20])


def test_wrong_product_same_key_not_direct_root(tmp_path):
    data = tables()
    data["steps"]["sim_product_id"][:2] = 999
    path = tmp_path / "other.root"
    write(path, data)
    entries, _, _, census, _ = read(path)
    np.testing.assert_array_equal(entries["event_row"], [1])
    assert census["non_front_entry_positive_events"] == 2


def test_failed_transport_is_not_a_zero_response(tmp_path):
    data = tables()
    data["events"]["g4_status"][1] = 1
    path = tmp_path / "failed.root"
    write(path, data)
    with pytest.raises(ValueError, match="failed transport"):
        read(path)
