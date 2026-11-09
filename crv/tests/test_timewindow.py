"""Synthetic named-window and runchain merge contract tests; no simulation or input files."""
import copy
import json

import pytest

from crvfastsim import runconfig
from crvfastsim.timewindow import resolve_runchain_window, validate_runchain_merge


def _record(name="run2_v10", window=None):
    rc, contract = resolve_runchain_window(name, window)
    return {"run_config": rc.name, "window": list(rc.window), "time_window": contract,
            "type_names": {1: "R1", 2: "R2"}, "generated_events": 10}


@pytest.mark.parametrize("name", [None, *runconfig.names()])
def test_implicit_cosmic_offspill_preserves_current_behavior(name):
    original = runconfig.get(name)
    rc, contract = resolve_runchain_window(name)
    old = runconfig.get(name, "offspill" if name else None)
    assert rc == old
    assert rc.window == runconfig.OFFSPILL_WINDOW
    assert contract["configured_window_ns"] == list(original.window)
    assert contract["effective_window_ns"] == list(rc.window)
    assert contract["selections"] == [{"source": "implicit_cosmic_offspill", "requested_window_ns": None}]
    assert runconfig.get(name) is original
    assert json.loads(json.dumps(contract)) == contract


@pytest.mark.parametrize("name", runconfig.names())
@pytest.mark.parametrize("window,expected", [("offspill", (0.0, 1e5)), ("onspill", (200.0, 1500.0)),
                                               ((-10, 900), (-10.0, 900.0)), ([0, 40], (0.0, 40.0))])
def test_explicit_selection(name, window, expected):
    rc, contract = resolve_runchain_window(name, window)
    assert rc.window == expected
    assert contract["configured_window_ns"] == list(runconfig.get(name).window)
    assert contract["selections"] == [{"source": "explicit_override", "requested_window_ns": list(expected)}]
    assert contract["time_reference"] == "input_absolute_time_ns"
    assert contract["time_rebase_ns"] == 0
    assert contract["generator_reweighting"] == "none"
    assert contract["denominator"] == "generated_events_unchanged"
    assert any("time acceptance changes" in text for text in contract["warnings"]) == (expected != (0, 1e5))


def test_generator_provenance_is_not_arrival_support():
    _, contract = resolve_runchain_window("run2_v10", "onspill")
    generator = contract["generator_convention"]
    assert generator["producer"] == "cosmicTimeOffset"
    assert generator["nominal_offset_window_ns"] == [0, 100005]
    assert generator["step_arrival_support"] == "not_inferred_from_offset_window"
    assert generator["evidence"] == "workflow_convention_not_input_validation"
    assert "arrival times need not lie inside it" in contract["warnings"][0]
    assert any("no time rebasing" in warning for warning in contract["warnings"])
    contract["generator_convention"]["nominal_offset_window_ns"][1] = 1
    assert resolve_runchain_window()[1]["generator_convention"]["nominal_offset_window_ns"][1] == 100005


@pytest.mark.parametrize("window", ["", "beam", "200,1500", 1, True, [], [0], [0, 1, 2], [0, 0], [1, 0],
                                    [False, 2], ["0", 2], [0, float("inf")], [float("nan"), 1]])
def test_bad_windows(window):
    with pytest.raises(ValueError):
        resolve_runchain_window(window=window)


@pytest.mark.parametrize("name", ["unknown", "", 42, []])
def test_bad_run_name(name):
    with pytest.raises(ValueError):
        resolve_runchain_window(name)


def test_merge_json_roundtrip_and_compatible_selection_provenance():
    implicit = _record()
    explicit = json.loads(json.dumps(_record(window="offspill")))
    before = copy.deepcopy([implicit, explicit])
    merged = validate_runchain_merge([implicit, explicit])
    assert [implicit, explicit] == before
    assert merged["time_window_status"] == "declared_compatible"
    assert merged["window"] == [0, 1e5]
    assert merged["type_names"] == {"1": "R1", "2": "R2"}
    assert {s["source"] for s in merged["time_window"]["selections"]} == {"implicit_cosmic_offspill", "explicit_override"}
    assert validate_runchain_merge([merged, implicit]) == merged
    assert validate_runchain_merge([explicit, implicit]) == merged
    assert json.loads(json.dumps(merged)) == merged


@pytest.mark.parametrize("second", [_record("run1a_v01"), _record(window="onspill"), _record(window=(0, 99999))])
def test_different_run_or_effective_window_rejected(second):
    with pytest.raises(ValueError, match="incompatible"):
        validate_runchain_merge([_record(), second])


def test_configured_window_mismatch_rejected_without_current_default_inference():
    first, second = _record(), _record()
    second["time_window"]["configured_window_ns"] = [200, 1400]
    with pytest.raises(ValueError, match="incompatible declared"):
        validate_runchain_merge([first, second])
    assert validate_runchain_merge([second, second])["time_window_status"] == "declared_compatible"


@pytest.mark.parametrize("field,value", [("schema", "unknown-v2"), ("generator_reweighting", "width_ratio"),
                                        ("time_rebase_ns", 200), ("time_rebase_ns", False),
                                        ("generator_convention", {}), ("warnings", "text"), ("selections", []),
                                        ("effective_window_ns", [0, float("nan")]),
                                        ("configured_window_ns", [100, 0]), ("run_config", "run1a_v01")])
def test_malformed_or_conflicting_contract(field, value):
    record = _record()
    record["time_window"][field] = value
    with pytest.raises(ValueError):
        validate_runchain_merge([record])


@pytest.mark.parametrize("contract", [None, {}, [], "legacy"])
def test_present_invalid_contract_is_not_legacy(contract):
    record = _record()
    record["time_window"] = contract
    with pytest.raises(ValueError):
        validate_runchain_merge([record])


@pytest.mark.parametrize("change", ["run_config", "window", "request", "source", "implicit"])
def test_inconsistent_declarations_rejected(change):
    record = _record(window="onspill")
    if change in ("run_config", "window"):
        del record[change]
    elif change == "request":
        record["time_window"]["selections"][0]["requested_window_ns"] = [0, 100000]
    elif change == "source":
        record["time_window"]["selections"][0]["source"] = "named_default"
    else:
        record["time_window"]["selections"] = [{"source": "implicit_cosmic_offspill", "requested_window_ns": None}]
    with pytest.raises(ValueError):
        validate_runchain_merge([record])


def test_legacy_homogeneous_is_unverified_and_not_upgraded():
    legacy = {"run_config": "run2_v10", "type_names": {1: "R1"}}
    metadata = validate_runchain_merge([legacy, copy.deepcopy(legacy)])
    assert metadata["time_window_status"] == "unverified_legacy"
    assert metadata["time_window_warnings"]
    assert "time_window" not in metadata and "window" not in metadata
    assert validate_runchain_merge([metadata, legacy]) == metadata
    assert validate_runchain_merge([{}, {}])["time_window_status"] == "unverified_legacy"
    declared_window = {**legacy, "window": [0, 100000]}
    assert validate_runchain_merge([declared_window, declared_window])["window"] == [0, 100000]


@pytest.mark.parametrize("records", [[{}, _record()], [_record(), {}], [{"window": [0, 100000]}, {}],
                                     [{"run_config": "run2_v10"}, {}], [{"type_names": {}}, {}],
                                     [{"run_config": "run2_v10"}, {"run_config": "run1a_v01"}],
                                     [{"window": [0, 100000]}, {"window": [200, 1500]}],
                                     [{"time_window_status": "declared_compatible"}], [], [None]])
def test_mixed_legacy_or_partially_declared_or_empty_rejected(records):
    with pytest.raises(ValueError):
        validate_runchain_merge(records)


@pytest.mark.parametrize("names", [{1: "T1"}, {1: "R1", "1": "R1"}, {True: "R1"}, {"a": "R1"}, {1: None}, []])
def test_sector_names_cannot_be_overwritten_or_malformed(names):
    first, second = _record(), _record()
    second["type_names"] = names
    with pytest.raises(ValueError):
        validate_runchain_merge([first, second])


def test_legacy_sector_conflict_and_new_missing_sector_map_rejected():
    with pytest.raises(ValueError, match="type_names"):
        validate_runchain_merge([{"type_names": {1: "R1"}}, {"type_names": {"1": "T1"}}])
    first, second = _record(), _record()
    del second["type_names"]
    with pytest.raises(ValueError, match="type_names"):
        validate_runchain_merge([first, second])
