"""Synthetic entrance-set contracts; no native recording or library certificate."""
import json

import numpy as np
import pytest

from calofastsim.recording import (BORN_INSIDE, EVENT_FIELDS, EVENT_INTS, INCIDENT_FIELDS,
                                    INCIDENT_INTS, INITIAL_EXTERIOR, MANIFEST_FIELDS, RETURN,
                                    STEP_FIELDS, STEP_INTS, UNCLASSIFIED, pack)


def fixture():
    events = {field: np.zeros(4, dtype=np.int64) for field in EVENT_INTS}
    events.update(source_sha256=np.array(["a" * 64] * 4), source_shard=np.ones(4, dtype=np.int64),
                  event=np.arange(4), root_product_id=np.ones(4, dtype=np.int64),
                  transport_status=np.array([0, 0, 10, 0]), recording_complete=np.array([1, 1, 1, 0]),
                  topology_supported=np.array([1, 1, 0, 0]))
    incidents = {field: np.zeros(5, dtype=np.float64) for field in INCIDENT_FIELDS}
    incidents.update({field: np.zeros(5, dtype=np.int64) for field in INCIDENT_INTS})
    incidents.update(event_index=np.array([1, 0, 0, 0, 2]), sequence=np.array([0, 2, 0, 1, 0]),
                     product_id=np.ones(5, dtype=np.int64), particle_key=np.array([9, 1, 1, 2, 8]),
                     visit=np.array([1, 2, 1, 1, 1]), role=np.array([BORN_INSIDE, RETURN, INITIAL_EXTERIOR,
                                                                INITIAL_EXTERIOR, UNCLASSIFIED]),
                     pdg=np.array([11, 11, 11, 22, -11]), pz=np.ones(5))
    steps = {field: np.zeros(2, dtype=np.float32) for field in STEP_FIELDS}
    steps.update({field: np.zeros(2, dtype=np.int64) for field in STEP_INTS})
    steps.update(event_index=np.array([0, 0]), product_id=np.ones(2, dtype=np.int64), index=np.arange(2),
                 sim_product_id=np.ones(2, dtype=np.int64), n_compress=np.ones(2, dtype=np.int64),
                 crystal=np.array([0, 1347]), time=np.array([2**54, 2**54 + 4.], dtype=np.float64))
    manifest = {field: events[field].copy() for field in MANIFEST_FIELDS}
    provenance = dict(purpose="diagnostic_entrance_set_recording", domain_sha256="b" * 64,
                      geometry_sha256="c" * 64, expanded_fcl_sha256="d" * 64,
                      independent_manifest_sha256="e" * 64, coordinate_frame="G4_world", units="mm_MeV_ns",
                      product_id_encoding="art_ProductID_value_uint32")
    return [events, incidents, steps, manifest, provenance]


def run(data):
    return pack(*data[:3], independent_manifest=data[3], provenance=data[4])


def test_multiple_incidents_both_disks_and_returns_remain_one_event(tmp_path):
    data = fixture()
    saved = run(data)
    np.testing.assert_array_equal(saved["incident_offsets"], [0, 3, 4, 5, 5])
    np.testing.assert_array_equal(saved["step_offsets"], [0, 2, 2, 2, 2])
    np.testing.assert_array_equal(saved["incident_role"], [INITIAL_EXTERIOR, INITIAL_EXTERIOR, RETURN,
                                                          BORN_INSIDE, UNCLASSIFIED])
    np.testing.assert_array_equal(saved["step_crystal"], [0, 1347])
    assert not np.any(saved["library_eligible"])
    np.testing.assert_array_equal(saved["failure_or_incomplete"], [False, False, True, True])
    np.testing.assert_array_equal(saved["step_time"].view(np.uint64), data[2]["time"].view(np.uint64))
    assert saved["step_time"].dtype == np.float64
    np.savez(tmp_path / "recording.npz", **saved)
    with np.load(tmp_path / "recording.npz", allow_pickle=False) as loaded:
        assert loaded["artifact_kind"] == "entrance_set_diagnostic_recording"
        assert json.loads(str(loaded["provenance_json"])) == data[4]
    data[0]["event"][0] = 42
    assert saved["event_event"][0] == 0


def test_manifest_order_independent_but_content_complete():
    data = fixture()
    data[3] = {field: value[::-1] for field, value in data[3].items()}
    assert len(run(data)["event_event"]) == 4
    data[3]["event"][0] = 99
    with pytest.raises(ValueError, match="independent"):
        run(data)


def test_empty_source_is_not_lost_or_promoted():
    data = fixture()
    for i in range(4):
        data[i] = {field: value[:0] for field, value in data[i].items()}
    result = run(data)
    np.testing.assert_array_equal(result["incident_offsets"], [0])
    np.testing.assert_array_equal(result["step_offsets"], [0])
    assert len(result["library_eligible"]) == 0


@pytest.mark.parametrize("table,field,value", [
    (0, "recording_complete", [1, 2, 1, 0]), (0, "topology_supported", [1, 1, 2, 0]),
    (0, "root_product_id", [1, 0, 1, 1]), (1, "event_index", [1, 0, 0, 0, 4]),
    (1, "role", [0, 1, 0, 0, 4]), (1, "visit", [1, 0, 1, 1, 1]),
    (1, "pdg", [11, 11, 11, 22, 0]), (1, "pz", [1., 1., 0., 1., 1.]),
    (1, "kinetic_energy", [0., -1., 0., 0., 0.]), (1, "time", [0., np.inf, 0., 0., 0.]),
    (1, "product_id", [1, 0, 1, 1, 1]), (2, "index", [0, 0]), (2, "n_compress", [1, 0]),
    (2, "sim_product_id", [0, 1]), (2, "energy_g4", [-1., 0.]),
    (1, "sequence", [0, 0, 0, 1, 0]), (1, "visit", [1, 1, 1, 1, 1]),
])
def test_malformed_recording_fails_without_dropping_rows(table, field, value):
    data = fixture()
    data[table][field] = np.asarray(value)
    with pytest.raises(ValueError):
        run(data)


def test_source_shard_cannot_split_one_artifact():
    data = fixture()
    for table in (data[0], data[3]):
        table["source_shard"] = np.array([1, 1, 2, 1])
    with pytest.raises(ValueError, match="one-to-one"):
        run(data)


@pytest.mark.parametrize("field", ["domain_sha256", "geometry_sha256", "expanded_fcl_sha256",
                                   "independent_manifest_sha256", "purpose"])
def test_unidentified_recording_refused(field):
    data = fixture()
    del data[4][field]
    with pytest.raises(ValueError):
        run(data)


def test_no_manifest_invented_from_survivors():
    data = fixture()
    data[0] = {field: value[:2] for field, value in data[0].items()}
    with pytest.raises(ValueError, match="independent"):
        run(data)


def test_recorded_float_storage_required():
    data = fixture()
    data[1]["time"] = np.zeros(5, dtype=np.int64)
    with pytest.raises(ValueError, match="floating storage"):
        run(data)


def test_born_inside_at_rest_is_retained():
    data = fixture()
    data[1]["pz"][0] = 0.
    assert run(data)["incident_pz"][3] == 0.


@pytest.mark.parametrize("field", ["coordinate_frame", "units", "product_id_encoding"])
def test_explicit_native_state_convention_required(field):
    data = fixture()
    data[4][field] = "ambiguous"
    with pytest.raises(ValueError, match=field):
        run(data)


def test_uint32_product_value_range_not_particle_key_range():
    data = fixture()
    data[1]["particle_key"][0] = 2**50
    assert 2**50 in run(data)["incident_particle_key"]
    data[1]["product_id"][0] = 2**32
    with pytest.raises(ValueError, match="uint32"):
        run(data)


@pytest.mark.parametrize("budget", [0, True, 1.5, 14])
def test_batch_budget_prevents_large_structured_temporaries(budget):
    data = fixture()
    with pytest.raises(ValueError, match="budget|max_rows"):
        pack(*data[:3], independent_manifest=data[3], provenance=data[4], max_rows=budget)
