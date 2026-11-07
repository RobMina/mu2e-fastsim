"""Synthetic saved-table comparisons: no production data or reconstruction backend."""
import json
from unittest.mock import patch

import numpy as np
import pytest

from calofastsim.artsteps import onspill_timing
from calofastsim.showers import production_compare as pc
from calofastsim.showers.export import EVENT, keys, read
from calofastsim.showers.library import build
from test_showers_export import tables, write
from test_showers_production import production_tables


@pytest.fixture
def case(tmp_path):
    data, donor = production_tables(), tables()
    # Collapse the two geometry runs, preserving three distinct event identities.
    for name, table in data.items():
        if "event" in table:
            old_run, old_subrun = table["run"].copy(), table["subrun"].copy()
            table["event"] = np.where(old_run == 2, 19, np.where(old_subrun == 1, 14, 11))
            table["run"] = np.ones_like(old_run)
        elif name in ("disks", "crystals", "virtualdetectors"):
            data[name] = {k: v[table["run"] == 1] for k, v in table.items()}
    for source in (data, donor):
        source["disks"]["size_x"][:] = 0.
        source["disks"]["size_y"][:] = 1000.
        source["steps"]["volume"][3] = 75  # empty-CSS query on disk 1
        source["steps"]["position_z"][3] = 200.
    data["showers"]["index"] = np.array([0, 1, 0])
    path, donor_path = tmp_path / "primary.root", tmp_path / "numeric.root"
    write(path, data)
    write(donor_path, donor)
    entry, css, geometry, _, _ = read(donor_path)
    library = build(entry, css, development=np.ones(len(entry["disk"]), bool))
    library.update({"geometry_" + k: v for k, v in geometry.items()})
    # Reused synthetic patterns test file provenance, NOT independent physics samples.
    metadata = {"input": str(donor_path)}
    sim = dict(lru=np.arange(12, dtype=float), pe_per_mev=np.arange(12, dtype=float) + 100.)
    return dict(path=path, data=data, library=library, geometry=geometry, metadata=metadata, sim=sim)


def prepare(case, **kw):
    return pc.prepare(case["path"], case["library"], case["sim"], metadata=case["metadata"],
                      stock_matched_phase0=True, **kw)


def test_geometry_sorted_ids_ignores_only_run(case):
    geom = {k: v[::-1].copy() for k, v in case["geometry"].items()}
    for k in geom:
        if k.endswith("_run"):
            geom[k][:] = 99
    checked = pc.check_geometry(case["library"], geom)
    assert "crystal_position_x" in checked and "vd_position_z" in checked
    assert not any(k.endswith("_run") for k in checked)


@pytest.mark.parametrize("fault", ["coordinate", "missing", "extra", "duplicate", "identity"])
def test_geometry_rejects_mismatch(case, fault):
    geom = {k: v.copy() for k, v in case["geometry"].items()}
    if fault == "coordinate":
        geom["crystal_position_x"][0] += 1.
    elif fault == "missing":
        del geom["vd_position_z"]
    elif fault == "extra":
        geom["disk_extra"] = np.zeros(2)
    else:
        geom["crystal_crystal"][0] = geom["crystal_crystal"][1] if fault == "duplicate" else 999
    with pytest.raises(ValueError, match="geometry"):
        pc.check_geometry(case["library"], geom)


def test_independence_provenance_is_not_integer_disjointness(case):
    ids = keys(case["data"]["events"], EVENT)
    assert "No integer-event disjointness claim" in pc.check_independence(
        case["path"], case["library"], case["metadata"], ids)
    for metadata, message in [({}, "provenance"), ({"input": str(case["path"])}, "comparison file")]:
        with pytest.raises(ValueError, match=message):
            pc.check_independence(case["path"], case["library"], metadata, ids)


def test_independence_checks_exact_full_ids(case):
    ids = keys(case["data"]["events"], EVENT)
    lib = dict(case["library"], source_full_ids=np.array([[2, 0, 11]]))
    assert "no overlap" in pc.check_independence(case["path"], lib, {}, ids)
    lib["source_full_ids"] = np.array([[1, 0, 11]])
    with pytest.raises(ValueError, match="identity overlap"):
        pc.check_independence(case["path"], lib, case["metadata"], ids)


@pytest.mark.parametrize("loss", [False, True])
def test_prepare_full_population_and_one_shared_draw(case, loss):
    if loss:
        case["library"]["step_relative_position"][0, 1] = 40.
    with patch.object(pc, "sample", wraps=pc.sample) as sample, \
         patch.object(pc, "place", wraps=pc.place) as geometric, \
         patch.object(pc, "place_crystals", wraps=pc.place_crystals) as crystal:
        result = prepare(case)
    assert sample.call_count == geometric.call_count == crystal.call_count == 1
    assert sample.call_args.kwargs["orientation"] == "disk"
    assert geometric.call_args.args[1] is crystal.call_args.args[2]
    np.testing.assert_array_equal(result["full_ids"], keys(case["data"]["events"], EVENT))
    np.testing.assert_array_equal(result["entries"]["event_row"], [0, 1])
    np.testing.assert_array_equal(result["sampling"]["supported"], [True, True])
    assert result["audit"]["sampling"]["empty_donors"] == 1
    census = result["audit"]["census"]
    assert census["events"] == 3 and census["non_front_entry_positive_events"] == 1
    control = result["steps_by_arm"]["control"]
    np.testing.assert_array_equal(control["energy_g4"], [40., 30., 20.])
    np.testing.assert_array_equal(control["time"], [11., 13., 31.])
    np.testing.assert_array_equal(control["z_local"], [10., 10., 10.])
    for arm, steps in result["steps_by_arm"].items():
        energy = np.bincount(steps["event"], weights=steps["energy_birks"], minlength=3)
        np.testing.assert_allclose(energy, [68, 0, 19] if arm == "control" else [29 if loss else 68, 0, 0])
        np.testing.assert_array_equal(steps["lru"], case["sim"]["lru"][steps["crystal"]])
        np.testing.assert_array_equal(steps["yields"], case["sim"]["pe_per_mev"][steps["crystal"]])
    for percent in result["audit"]["sampling"]["unmapped_energy_percent"].values():
        assert percent == pytest.approx(100 * 39 / 68 if loss else 0)
    expected = onspill_timing(case["data"]["events"]["event"], **pc.TIMING)
    for k in expected:
        np.testing.assert_array_equal(result["timing"][k], expected[k])
    assert not np.array_equal(result["timing"]["pbtime"], onspill_timing(np.arange(3))["pbtime"])


@pytest.mark.parametrize("fault,message", [("phase", "phase-zero"), ("calibration", "calibration"),
                                           ("order", "collection order")])
def test_prepare_guards(case, fault, message):
    if fault == "calibration":
        case["sim"]["lru"] = np.ones(11)
    elif fault == "order":
        case["data"]["showers"]["index"][:] = 0
        write(case["path"], case["data"])
    with pytest.raises(ValueError, match=message):
        if fault == "phase":
            pc.prepare(case["path"], case["library"], case["sim"])
        else:
            prepare(case)


def test_diagnostics_same_query_rows_both_disks_radius_and_nulls(case, tmp_path):
    prepared = prepare(case)
    prepared["entries"]["radius"][:] = [5., 500.]
    prepared["entries"]["position"][1, 0] = 500.
    clusters = {name: dict(energy=np.array([20., 0., 1000.]), time=np.zeros(3),
                           x=np.zeros(3), y=np.zeros(3), z=np.zeros(3)) for name in (*pc.ARMS, "stockref")}
    result = pc.diagnostics(prepared, clusters)
    for arm in pc.ARMS:
        report = result[arm]
        assert report["paired"]["events"] == 2 and report["paired"]["energy_mean"]["fast"] == 10.
        assert report["common_selected"] == 1
        for name in ("disk0", "disk1", "radius_0_450", "radius_450_550"):
            assert report["strata"][name]["events"] == 1
        empty = report["strata"]["radius_550_1000"]
        assert empty["events"] == 0 and empty["energy_mean"]["fast"] is None
        assert empty["acceptance_ratio"]["ratio"] is None
        assert report["paired"]["energy_mean"]["error"] is None
        energy_hist = report["observables"]["Leading-cluster energy [MeV]"]
        assert sum(energy_hist["fast"]) == sum(energy_hist["stock"]) == 2
    path = tmp_path / "numbers.json"
    pc._json(path, result)
    assert json.loads(path.read_text()) == result
    assert "NaN" not in path.read_text()


def test_unsupported_queries_are_not_empty_donors(case):
    case["library"]["entry_disk"][:] = 2
    result = prepare(case)
    np.testing.assert_array_equal(result["sampling"]["supported"], [False, False])
    np.testing.assert_array_equal(result["sampling"]["donor"], [-1, -1])
    audit = result["audit"]["sampling"]
    assert audit["unsupported"] == 2 and audit["empty_donors"] == 0
    assert audit["unmapped_energy_percent"] == dict(geometric=None, crystal=None)
    assert len(result["full_ids"]) == 3 and len(result["steps_by_arm"]["control"]["event"]) == 3
