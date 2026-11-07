"""Exact sampling, full-truth digests, and reduced ROOT regression contracts."""
from copy import deepcopy
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from mixing.direct_frames import to_detector_inputs
from mixing.direct_regression import compare_outputs, compare_sampling, input_digests
from mixing.fast_detectors import save_sampling
from mixing.frames import FrameBatch, IDS
from mixing.sampling import STREAMS
from test_mixing_direct_frames import KEYS, batch


@pytest.fixture
def sampling_reference(tmp_path):
    fingerprint = {"source_sha256": "a" * 64, "census_sha256": "b" * 64,
                   "event_log_sha256": "c" * 64, "product_id_checks": {"tracker": "checked"}}
    provenance = {
        "campaign": {"mode": "sequential", "extended_mean": 10., "beam_primary": True},
        "seed": 421, "stream_order": STREAMS,
        "rates": np.arange(1., len(STREAMS) + 1), "intensity": np.array([1., 2.]),
        "multiplicities": np.array([[1] * len(STREAMS), [0] * len(STREAMS)], dtype=np.int64),
        "sources": {"signal": {"path": "signal.root"},
                    "pools": {s: {"path": "pool.root", "event_log": "pool.log"} for s in STREAMS}},
        "source_fingerprints": {"signal": deepcopy(fingerprint),
                                "pools": {s: deepcopy(fingerprint) for s in STREAMS}},
        "signal_events_full": 5, "signal_event_ids": KEYS[:2].copy(),
        "pool_event_counts": dict.fromkeys(STREAMS, 5),
        "estimated_selected_rows": {"tracker": 6, "crv": 12, "particles": 12, "gen": 6},
        "actual_rows": {"tracker": 6, "crv": 12, "particles": 12, "gen": 6},
        "selections": {s: {"counts": np.array([1, 0], dtype=np.int64),
                           "donors": np.array([3], dtype=np.int64),
                           "source_event_ids": KEYS[[3]].copy(),
                           "event_index": np.array([0], dtype=np.int64),
                           "initial_cursor": 3, "next_cursor": 4} for s in STREAMS},
        "timing_s": {"read_validate": 1.},
    }
    save_sampling(tmp_path, provenance)
    return provenance, tmp_path


def test_saved_sampling_roundtrip_uses_json_npz_and_ignores_runtime(sampling_reference):
    provenance, directory = sampling_reference
    metadata = json.loads((directory / "sampling.json").read_text())
    receipt = metadata["selections"][STREAMS[0]]["source_event_ids"]
    with np.load(directory / "sampling.npz", allow_pickle=False) as arrays:
        assert receipt["dtype"] == "uint64" and receipt["shape"] == [1, 3]
        np.testing.assert_array_equal(arrays[receipt["npz_key"]], KEYS[[3]])
    provenance["timing_s"]["read_validate"] = 999.
    assert compare_sampling(provenance, directory) is True


@pytest.mark.parametrize("field", ["rates", "intensity", "multiplicities", "signal_event_ids",
                                    "counts", "donors", "source_event_ids", "event_index"])
@pytest.mark.parametrize("change", ["value", "dtype", "shape"])
def test_saved_sampling_rejects_each_array_change(sampling_reference, field, change):
    provenance, directory = sampling_reference
    actual = deepcopy(provenance)
    owner = actual if field in actual else actual["selections"][STREAMS[0]]
    values = owner[field]
    if change == "value":
        values.flat[0] += 1
    elif change == "dtype":
        owner[field] = values.astype(np.float32 if values.dtype.kind == "f" else np.float64)
    else:
        owner[field] = values.reshape((1, *values.shape))
    with pytest.raises(ValueError, match="saved sampling array changed"):
        compare_sampling(actual, directory)


@pytest.mark.parametrize("keys,value", [
    (("campaign", "mode"), "randomReplace"),
    (("seed",), 422),
    (("stream_order",), STREAMS[::-1]),
    (("sources", "signal", "path"), "different.root"),
    (("sources", "pools", STREAMS[0], "event_log"), "different.log"),
    (("signal_events_full",), 4),
    (("pool_event_counts", STREAMS[0]), 4),
    (("estimated_selected_rows", "tracker"), 7),
    (("actual_rows", "crv"), 13),
    (("selections", STREAMS[0], "initial_cursor"), 2),
    (("selections", STREAMS[0], "next_cursor"), 0),
    (("source_fingerprints", "signal", "source_sha256"), "d" * 64),
    (("source_fingerprints", "signal", "census_sha256"), "d" * 64),
    (("source_fingerprints", "pools", STREAMS[0], "source_sha256"), "d" * 64),
    (("source_fingerprints", "pools", STREAMS[0], "event_log_sha256"), "d" * 64),
    (("source_fingerprints", "pools", STREAMS[0], "product_id_checks", "tracker"), "unavailable"),
])
def test_saved_sampling_rejects_metadata_and_source_checks(sampling_reference, keys, value):
    provenance, directory = sampling_reference
    actual = deepcopy(provenance)
    owner = actual
    for key in keys[:-1]:
        owner = owner[key]
    owner[keys[-1]] = value
    with pytest.raises(ValueError, match="saved sampling .*changed"):
        compare_sampling(actual, directory)


def test_saved_sampling_rejects_nested_key_changes(sampling_reference):
    provenance, directory = sampling_reference
    actual = deepcopy(provenance)
    actual["selections"][STREAMS[0]]["unexpected"] = 1
    with pytest.raises(ValueError, match="saved sampling keys changed"):
        compare_sampling(actual, directory)


def test_saved_sampling_checks_actual_npz_values_not_only_json_receipts(sampling_reference):
    provenance, directory = sampling_reference
    with np.load(directory / "sampling.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    arrays["sampling/selections/" + STREAMS[0] + "/donors"][0] += 1
    np.savez(directory / "sampling.npz", **arrays)
    with pytest.raises(ValueError, match="saved sampling array changed"):
        compare_sampling(provenance, directory)


def test_input_digests_cover_every_truth_and_detector_field_and_stable_nans():
    mixed = batch()
    tracker, crv, pdg = to_detector_inputs(mixed)
    assert np.isnan(crv.t0).any() and np.isnan(crv.t1).any()
    expected_keys = {"event_ids", "pdg"}
    for name, rows in mixed.collections.items():
        expected_keys.update([name + "/counts", *(name + "/" + field for field in ak.fields(rows))])
    for name, table in (("tracker_input", tracker), ("crv_input", crv)):
        expected_keys.update(name + "/" + field for field in table.columns)
    first = input_digests(mixed, tracker, crv, pdg)
    assert set(first) == expected_keys
    repeated = batch()
    assert input_digests(repeated, *to_detector_inputs(repeated)) == first


@pytest.mark.parametrize("collection,field", [("particles", "sx"), ("particles", "pdg"),
                                               ("gen", "e"), ("tracker", "width"), ("crv", "t0")])
def test_input_digests_detect_same_count_truth_value_changes(collection, field):
    original = batch()
    tracker, crv, pdg = to_detector_inputs(original)
    expected = input_digests(original, tracker, crv, pdg)
    changed = deepcopy(original)
    rows = changed.collections[collection]
    rows[field] = rows[field] + 1
    actual = input_digests(changed, tracker, crv, pdg)
    assert actual[collection + "/counts"] == expected[collection + "/counts"]
    assert actual[collection + "/" + field] != expected[collection + "/" + field]
    assert {key for key in actual if actual[key] != expected[key]} == {collection + "/" + field}


def test_input_digests_detect_same_counts_truth_row_order_changes():
    original = batch()
    tracker, crv, pdg = to_detector_inputs(original)
    changed = FrameBatch(original.event_ids, dict(original.collections,
                         particles=original.collections["particles"][:, ::-1]))
    before = input_digests(original, tracker, crv, pdg)
    after = input_digests(changed, tracker, crv, pdg)
    assert before["particles/counts"] == after["particles/counts"]
    assert before["particles/pdg"] != after["particles/pdg"]
    assert before["particles/key"] != after["particles/key"]
    assert before["crv_input/sp"] == after["crv_input/sp"]


@pytest.mark.parametrize("change,key", [("ids", "event_ids"), ("pdg", "pdg"),
                                          ("tracker", "tracker_input/t"), ("crv", "crv_input/t0"),
                                          ("counts", "particles/counts"), ("dtype", "pdg")])
def test_input_digests_include_ids_counts_inputs_pdg_and_dtype(change, key):
    mixed = batch()
    tracker, crv, pdg = to_detector_inputs(mixed)
    before = input_digests(mixed, tracker, crv, pdg)
    if change == "ids":
        mixed.event_ids[0, 2] += 1
    elif change == "pdg":
        pdg[0] += 1
    elif change == "dtype":
        pdg = pdg.astype(np.int32)
    elif change == "tracker":
        tracker.loc[0, "t"] += 1
    elif change == "crv":
        crv.loc[0, "t0"] += 1
    else:
        rows = mixed.collections["particles"]
        mixed.collections["particles"] = ak.unflatten(ak.flatten(rows), [1, 3, 2, 2, 0])
    after = input_digests(mixed, tracker, crv, pdg)
    assert before[key] != after[key]
    if change == "counts":
        assert before["particles/pdg"] == after["particles/pdg"]


def output_tables(tracker=True):
    ids = {name: KEYS[:3, i].copy() for i, name in enumerate(IDS)}
    tables = {
        "crvfast/events": {**ids, "ndigis": np.array([1, 1, 0], dtype=np.int64)},
        "crvfast/digis": {**{k: v[:2].copy() for k, v in ids.items()},
                          "adc": ak.values_astype(ak.Array([[10, 11], [12]]), np.int16)},
        "crvfast/pulses": {"event": KEYS[:2, 2].copy(), "time": np.array([np.nan, 20.])},
        "crvfast/clusters": {"event": KEYS[:1, 2].copy(), "pe": np.array([3.])},
    }
    if tracker:
        tables["trkdigi/events"] = {**{k: v.copy() for k, v in ids.items()},
                                    "adc": ak.values_astype(ak.Array([[10, 11], [12], []]), np.int16)}
    return tables


def write_outputs(path, tables, *, stale=None, rntuple=False):
    with uproot.recreate(path) as root:
        for name, columns in (stale or {}).items():
            root.mktree(name, columns)
        for name, columns in tables.items():
            if rntuple:
                root.mkrntuple(name, ak.Array(columns))
            else:
                root.mktree(name, columns)


@pytest.mark.parametrize("tracker", [False, True])
def test_compare_outputs_equal_nans_and_full_empty_inclusive_census(tmp_path, tracker):
    left, right = tmp_path / "left.root", tmp_path / "right.root"
    tables = output_tables(tracker)
    write_outputs(left, tables)
    write_outputs(right, deepcopy(tables))
    report = compare_outputs(left, right)
    assert report["equal"] is True
    objects = {row["name"]: row for row in report["objects"]}
    assert set(objects) == set(tables)
    assert objects["crvfast/events"]["entries"] == 3
    if tracker:
        assert objects["trkdigi/events"]["entries"] == 3


def test_compare_outputs_uses_highest_cycle_not_first(tmp_path):
    left, right = tmp_path / "left.root", tmp_path / "right.root"
    current, stale = output_tables(), output_tables()
    stale["crvfast/events"]["event"][0] += 50
    write_outputs(left, current, stale=stale)
    write_outputs(right, current)
    with uproot.open(left) as root:
        assert "crvfast/events;2" in root
    assert compare_outputs(left, right)["equal"]
    write_outputs(right, stale, stale=current)
    with pytest.raises(ValueError, match="output column changed: crvfast/events/event"):
        compare_outputs(left, right)


@pytest.mark.parametrize("change", ["census", "id", "order", "counts", "adc", "tracker_adc",
                                     "dtype", "nan", "tiny", "schema", "missing", "tracker_missing"])
def test_compare_outputs_rejects_strict_census_schema_values_and_dtypes(tmp_path, change):
    left, right = tmp_path / "left.root", tmp_path / "right.root"
    original, changed = output_tables(), output_tables()
    if change == "census":
        changed["crvfast/events"] = {k: v[:2] for k, v in changed["crvfast/events"].items()}
    elif change == "id":
        changed["crvfast/events"]["event"][-1] += 1
    elif change == "order":
        changed["crvfast/events"] = {k: v[::-1] for k, v in changed["crvfast/events"].items()}
    elif change == "counts":
        changed["crvfast/events"]["ndigis"][0] += 1
    elif change in ("adc", "tracker_adc"):
        name = "trkdigi/events" if change == "tracker_adc" else "crvfast/digis"
        changed[name]["adc"] = changed[name]["adc"] + 1
    elif change == "dtype":
        changed["crvfast/clusters"]["pe"] = changed["crvfast/clusters"]["pe"].astype(np.float32)
    elif change == "nan":
        changed["crvfast/pulses"]["time"][0] = 0.
    elif change == "tiny":
        changed["crvfast/pulses"]["time"][1] = np.nextafter(20., np.inf)
    elif change == "schema":
        changed["crvfast/clusters"]["extra"] = np.array([0])
    elif change == "missing":
        del changed["crvfast/clusters"]
    else:
        del changed["trkdigi/events"]
    write_outputs(left, original)
    write_outputs(right, changed)
    with pytest.raises(ValueError, match="output|missing"):
        compare_outputs(left, right)


@pytest.mark.parametrize("table", ["crvfast/digis", "trkdigi/events"])
def test_compare_outputs_rejects_different_jagged_offsets_with_identical_flat_values(tmp_path, table):
    left, right = tmp_path / "left.root", tmp_path / "right.root"
    original, changed = output_tables(), output_tables()
    values = [[10], [11, 12], []] if table == "trkdigi/events" else [[10], [11, 12]]
    changed[table]["adc"] = ak.values_astype(ak.Array(values), np.int16)
    np.testing.assert_array_equal(ak.to_numpy(ak.flatten(original[table]["adc"])),
                                  ak.to_numpy(ak.flatten(changed[table]["adc"])))
    write_outputs(left, original, rntuple=True)
    write_outputs(right, changed, rntuple=True)
    with pytest.raises(ValueError, match="output row shape changed"):
        compare_outputs(left, right)


def test_compare_outputs_rejects_array_rank_change_with_same_flat_values(tmp_path):
    left, right = tmp_path / "left.root", tmp_path / "right.root"
    original, changed = output_tables(), output_tables()
    original["crvfast/digis"]["adc"] = ak.Array([[10], [11]])
    changed["crvfast/digis"]["adc"] = np.array([10, 11])
    write_outputs(left, original, rntuple=True)
    write_outputs(right, changed, rntuple=True)
    with pytest.raises(ValueError, match="output branch schema changed|output array rank changed"):
        compare_outputs(left, right)


def test_compare_outputs_requires_crv_tables_even_when_absent_on_both_sides(tmp_path):
    left, right = tmp_path / "left.root", tmp_path / "right.root"
    tables = output_tables()
    del tables["crvfast/pulses"]
    write_outputs(left, tables)
    write_outputs(right, tables)
    with pytest.raises(ValueError, match="missing output table: crvfast/pulses"):
        compare_outputs(left, right)


def test_empty_fixed_width_adc_schema_mismatch_rejected(tmp_path):
    left, right = tmp_path / "left.root", tmp_path / "right.root"
    original, changed = output_tables(), output_tables()
    original["crvfast/digis"] = {"adc": np.empty((0, 12), np.int64)}
    changed["crvfast/digis"] = {"adc": np.empty((0, 13), np.int64)}
    write_outputs(left, original)
    write_outputs(right, changed)
    with pytest.raises(ValueError, match="branch schema changed"):
        compare_outputs(left, right)


def test_nullable_positions_cannot_disappear_when_flattening(tmp_path):
    left, right = tmp_path / "left.root", tmp_path / "right.root"
    original, changed = output_tables(), output_tables()
    original["crvfast/pulses"] = {"time": ak.Array([None, 10., 20.])}
    changed["crvfast/pulses"] = {"time": ak.Array([10., None, 20.])}
    write_outputs(left, original, rntuple=True)
    write_outputs(right, changed, rntuple=True)
    with pytest.raises(ValueError, match="missing-value positions changed"):
        compare_outputs(left, right)
