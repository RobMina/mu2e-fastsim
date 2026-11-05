"""Library completeness contracts on synthetic independent incident events."""
import json

import numpy as np
import pytest

from calofastsim.library import (BOUNDARY_POLICY, ENTRY_FIELDS, ENTRY_INTEGER_FIELDS,
                                 MANIFEST_FIELDS, STEP_FIELDS, STEP_INTEGER_FIELDS, pack)


def fixture():
    entries = {name: np.zeros(3) for name in ENTRY_FIELDS}
    entries.update({name: np.zeros(3, dtype=np.int64) for name in ENTRY_INTEGER_FIELDS})
    entries.update(event=np.arange(3), source_shard=np.array([1, 2, 1]),
                   source_sha256=np.array(["a" * 64, "b" * 64, "a" * 64]),
                   pz=np.ones(3), pdg=np.array([11, -11, 13]), product_id=np.ones(3, dtype=np.int64))
    steps = {name: np.zeros(2) for name in STEP_FIELDS}
    steps.update({name: np.zeros(2, dtype=np.int64) for name in STEP_INTEGER_FIELDS})
    steps.update(root_index=np.array([2, 0]), energy_g4=np.array([3., 1.]), n_compress=np.ones(2, dtype=np.int64),
                 sim_product_id=np.ones(2, dtype=np.int64))
    provenance = dict(unfiltered_retention=True, complete_incident_census=True,
                      zero_response_retained=True, completed_transport=True,
                      geometry_sha256="c" * 64, expanded_fcl_sha256="d" * 64,
                      incident_manifest_sha256="e" * 64, physics_release="synthetic",
                      boundary_definition="synthetic", boundary_policy=BOUNDARY_POLICY)
    manifest = {name: entries[name].copy() for name in MANIFEST_FIELDS}
    return [entries, steps, provenance, manifest]


def run_pack(data):
    entries, steps, provenance, manifest = data
    return pack(entries, steps, provenance=provenance, incident_manifest=manifest)


def test_zero_response_and_whole_root_retained(tmp_path):
    data = fixture()
    result = run_pack(data)
    np.testing.assert_array_equal(result["step_offsets"], [0, 1, 1, 2])
    np.testing.assert_array_equal(result["step_energy_g4"], [1, 3])
    np.testing.assert_array_equal(result["development"], [True, False, True])
    assert len(result["entry_event"]) == 3
    assert json.loads(str(result["provenance_json"])) == data[2]
    np.savez(tmp_path / "library.npz", **result)
    with np.load(tmp_path / "library.npz", allow_pickle=False) as saved:
        assert saved["schema_version"] == 2
        assert json.loads(str(saved["provenance_json"])) == data[2]
    data[0]["event"][0] = 99
    assert result["entry_event"][0] == 0


@pytest.mark.parametrize("gate", ["unfiltered_retention", "complete_incident_census",
                                 "zero_response_retained", "completed_transport"])
def test_selected_pruned_or_incomplete_input_refused(gate):
    data = fixture()
    data[2][gate] = False
    with pytest.raises(ValueError, match="required"):
        run_pack(data)


def test_multi_root_event_requires_real_assignment():
    data = fixture()
    data[0]["event"] = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="one complete"):
        run_pack(data)


@pytest.mark.parametrize("column", ["x", "disk"])
def test_nonscalar_columns_refused(column):
    data = fixture()
    data[0][column] = np.zeros((3, 2))
    with pytest.raises(ValueError, match="one-dimensional"):
        run_pack(data)


@pytest.mark.parametrize("column,value", [("event", [0., .5, 2.]),
                                          ("event", [0., np.nan, 2.]),
                                          ("source_shard", [1, -2, 1]),
                                          ("source_shard", [1., 2.5, 1.]),
                                          ("px", [1j, 1j, 1j]),
                                          ("pz", [0., 0., 0.]),
                                          ("product_id", [1, 0, 1]),
                                          ("disk", [0, 2, 0]),
                                          ("g4_status", [0, 1, 0]),
                                          ("g4_status", [0, 10, 0])])
def test_invalid_incident_state_refused(column, value):
    data = fixture()
    data[0][column] = np.asarray(value)
    with pytest.raises(ValueError):
        run_pack(data)


@pytest.mark.parametrize("column,value", [("root_index", [-1, 0]),
                                          ("root_index", [3, 0]),
                                          ("root_index", [1.5, 0]),
                                          ("crystal", [-1, 0]),
                                          ("sim_product_id", [0, 1]),
                                          ("n_compress", [0, 1]),
                                          ("momentum_in", [-1., 0.]),
                                          ("x", [np.inf, 0]),
                                          ("energy_g4", [-1., 0.]),
                                          ("energy_birks", [0., np.nan])])
def test_invalid_step_refused(column, value):
    data = fixture()
    data[1][column] = np.asarray(value)
    with pytest.raises(ValueError):
        run_pack(data)


@pytest.mark.parametrize("which", ["missing", "extra", "wrong_identity"])
def test_manifest_mismatch_refused(which):
    data = fixture()
    if which == "missing":
        data[3] = {name: array[:-1] for name, array in data[3].items()}
    elif which == "extra":
        data[3] = {name: np.r_[array, array[:1]] for name, array in data[3].items()}
        data[3]["event"][-1] = 10
    else:
        data[3]["particle_key"][0] = 5
    with pytest.raises(ValueError, match="manifest"):
        run_pack(data)


def test_manifest_required():
    entries, steps, provenance, _ = fixture()
    with pytest.raises(ValueError, match="manifest"):
        pack(entries, steps, provenance=provenance)


@pytest.mark.parametrize("key,value", [("geometry_sha256", "synthetic"),
                                       ("expanded_fcl_sha256", "C" * 64),
                                       ("physics_release", " "),
                                       ("boundary_policy", "second_crossing"),
                                       ("unexpected_nonfinite", np.nan)])
def test_bad_provenance_refused(key, value):
    data = fixture()
    data[2][key] = value
    with pytest.raises(ValueError):
        run_pack(data)


def test_same_source_cannot_cross_split_or_duplicate_events():
    data = fixture()
    data[0]["source_sha256"][:] = "a" * 64
    data[3]["source_sha256"][:] = "a" * 64
    with pytest.raises(ValueError, match="source artifact"):
        run_pack(data)


def test_distinct_sources_may_have_same_event_number():
    data = fixture()
    data[0]["event"][1] = 0
    data[3]["event"][1] = 0
    result = run_pack(data)
    assert len(result["entry_event"]) == 3


def test_stable_sort_and_all_empty_step_tables():
    data = fixture()
    data[1] = {name: np.repeat(array, 2) for name, array in data[1].items()}
    data[1]["time"] = np.arange(4.)
    result = run_pack(data)
    np.testing.assert_array_equal(result["step_time"], [2, 3, 0, 1])
    np.testing.assert_array_equal(np.diff(result["step_offsets"]), [2, 0, 2])
    data[1] = {name: array[:0] for name, array in data[1].items()}
    result = run_pack(data)
    np.testing.assert_array_equal(result["step_offsets"], [0, 0, 0, 0])
    data[0] = {name: array[:0] for name, array in data[0].items()}
    data[3] = {name: array[:0] for name, array in data[3].items()}
    result = run_pack(data)
    np.testing.assert_array_equal(result["step_offsets"], [0])


def test_all_compressed_shower_fields_and_identity_survive():
    data = fixture()
    data[1]["n_compress"] = np.array([5, 7])
    data[1]["momentum_in"] = np.array([80., 100.], dtype=np.float32)
    data[1]["sim_product_id"] = np.array([19, 19])
    data[1]["sim_particle_key"] = np.array([300, 100])
    result = run_pack(data)
    np.testing.assert_array_equal(result["step_n_compress"], [7, 5])
    np.testing.assert_array_equal(result["step_momentum_in"], [100., 80.])
    np.testing.assert_array_equal(result["step_sim_particle_key"], [100, 300])
    assert result["step_momentum_in"].dtype == np.float32


def test_whole_event_may_deposit_across_both_disks():
    data = fixture()
    data[1]["root_index"] = np.array([0, 0])
    data[1]["crystal"] = np.array([0, 900])
    result = run_pack(data)
    np.testing.assert_array_equal(result["step_crystal"], [0, 900])
    np.testing.assert_array_equal(result["step_offsets"], [0, 2, 2, 2])
