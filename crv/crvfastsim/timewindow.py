"""Explicit cosmic runchain window declarations and fail-closed merge compatibility."""
from __future__ import annotations

import copy
import dataclasses
import json
import math
from collections.abc import Mapping
from numbers import Real

from . import runconfig

SCHEMA = "crv-runchain-time-window-v1"
_GENERATOR = {
    "producer": "cosmicTimeOffset",
    "nominal_offset_window_ns": [0.0, 100005.0],
    "nominal_offset_distribution": "uniform",
    "evidence": "workflow_convention_not_input_validation",
    "step_arrival_support": "not_inferred_from_offset_window",
}
_POLICY = {
    "time_reference": "input_absolute_time_ns",
    "time_rebase_ns": 0.0,
    "generator_reweighting": "none",
    "denominator": "generated_events_unchanged",
    "stock_onspill_alignment": "not_implemented_event_pbtime_required",
}
_SUPPORT_WARNING = (
    "cosmicTimeOffset's nominal 100 us offset window is a workflow convention, not verified input support; "
    "absolute generator times and transported step, secondary, photon or pulse arrival times need not lie inside it."
)
_ACCEPTANCE_WARNING = (
    "The effective window differs from the nominal cosmic offspill window: time acceptance changes, "
    "with no time rebasing, generator reweighting or generated-event denominator adjustment; "
    "a window-width ratio is not a measured retained-event fraction."
)
_LEGACY_WARNING = (
    "Legacy runchain shards lack an explicit time-window contract; compatible declared fields do not "
    "verify their effective window, generator convention or physics compatibility."
)


def _bounds(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label}: expected two numeric bounds in ns")
    if any(isinstance(x, bool) or not isinstance(x, Real) for x in value):
        raise ValueError(f"{label}: bounds must be real numbers, not booleans or strings")
    bounds = [float(x) for x in value]
    if not all(math.isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
        raise ValueError(f"{label}: bounds must be finite and strictly increasing")
    return bounds


def resolve_runchain_window(run_config=None, window=None):
    """Return (resolved RunConfig, JSON-safe provenance), preserving cosmic offspill defaults."""
    if run_config is not None and (not isinstance(run_config, str) or not run_config):
        raise ValueError("run_config must be a nonempty named configuration or None")
    try:
        configured = runconfig.get(run_config)
    except KeyError as exc:
        raise ValueError(f"unknown run configuration: {run_config!r}") from exc
    if window is None:
        effective = list(runconfig.OFFSPILL_WINDOW)
        selection = {"source": "implicit_cosmic_offspill", "requested_window_ns": None}
    else:
        if isinstance(window, str):
            aliases = {"offspill": runconfig.OFFSPILL_WINDOW, "onspill": runconfig.ONSPILL_WINDOW}
            if window not in aliases:
                raise ValueError(f"unknown window: {window!r}; use offspill, onspill or numeric bounds")
            window = aliases[window]
        effective = _bounds(window, "window")
        selection = {"source": "explicit_override", "requested_window_ns": list(effective)}
    configured_window = _bounds(configured.window, "configured window")
    warnings = [_SUPPORT_WARNING]
    if effective != configured_window:
        warnings.append("The effective cosmic window differs from the named run's configured digitisation window.")
    if effective != list(runconfig.OFFSPILL_WINDOW):
        warnings.append(_ACCEPTANCE_WARNING)
        warnings.append("Stock onspill uses eventWindowStart=-pbtimeMC for response windows and FEB TDC0; "
                        "this fixed absolute-time crop does not implement that event-dependent alignment.")
    contract = {
        "schema": SCHEMA,
        "run_config": configured.name,
        "configured_window_ns": configured_window,
        "effective_window_ns": effective,
        "selections": [selection],
        "generator_convention": copy.deepcopy(_GENERATOR),
        **_POLICY,
        "warnings": warnings,
    }
    return dataclasses.replace(configured, window=tuple(effective)), contract


def _strings(value, label):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"{label}: expected a list of nonempty strings")
    return list(value)


def _contract(value):
    required = {"schema", "run_config", "configured_window_ns", "effective_window_ns", "selections",
                "generator_convention", "warnings", *_POLICY}
    if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != SCHEMA:
        raise ValueError("time_window: malformed or unsupported contract schema")
    result = copy.deepcopy(dict(value))
    if not isinstance(result["run_config"], str) or not result["run_config"]:
        raise ValueError("time_window: missing run configuration")
    for key in ("configured_window_ns", "effective_window_ns"):
        result[key] = _bounds(result[key], key)
    if result["generator_convention"] != _GENERATOR:
        raise ValueError("time_window: incompatible generator convention")
    for key, expected in _POLICY.items():
        if result[key] != expected or isinstance(result[key], bool):
            raise ValueError(f"time_window: unsupported {key}")
    selections = result["selections"]
    if not isinstance(selections, list) or not selections:
        raise ValueError("time_window: missing selection provenance")
    for selection in selections:
        if not isinstance(selection, Mapping) or set(selection) != {"source", "requested_window_ns"}:
            raise ValueError("time_window: malformed selection provenance")
        if selection["source"] == "implicit_cosmic_offspill":
            if selection["requested_window_ns"] is not None or result["effective_window_ns"] != list(runconfig.OFFSPILL_WINDOW):
                raise ValueError("time_window: inconsistent implicit offspill selection")
        elif selection["source"] == "explicit_override":
            if _bounds(selection["requested_window_ns"], "requested window") != result["effective_window_ns"]:
                raise ValueError("time_window: explicit request disagrees with effective window")
        else:
            raise ValueError("time_window: unknown selection source")
    result["warnings"] = _strings(result["warnings"], "time_window warnings")
    return result


def _type_names(value):
    if not isinstance(value, Mapping):
        raise ValueError("type_names must be a mapping")
    names = {}
    for key, name in value.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise ValueError("type_names keys must be integer sector types")
        try:
            canonical = str(int(key))
        except ValueError as exc:
            raise ValueError("type_names keys must be integer sector types") from exc
        if not isinstance(name, str) or not name or canonical in names:
            raise ValueError("type_names contains invalid names or duplicate normalised keys")
        names[canonical] = name
    return names


def _common_field(records, key, normalise):
    present = [key in record for record in records]
    if any(present) and not all(present):
        raise ValueError(f"incompatible {key}: partially declared across shards")
    if not any(present):
        return {}
    values = [normalise(record[key]) for record in records]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"incompatible {key} across shards")
    return {key: values[0]}


def _run_name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("run_config must be a nonempty string when declared")
    return value


def validate_runchain_merge(records):
    """Return merge metadata or raise ValueError; legacy compatibility is explicitly unverified."""
    records = list(records)
    if not records or any(not isinstance(record, Mapping) for record in records):
        raise ValueError("runchain merge requires nonempty mapping records")
    declared = ["time_window" in record for record in records]
    if any(declared) and not all(declared):
        raise ValueError("cannot mix legacy and declared time-window contracts")
    metadata = {}
    metadata.update(_common_field(records, "run_config", _run_name))
    metadata.update(_common_field(records, "window", lambda value: _bounds(value, "window")))
    metadata.update(_common_field(records, "type_names", _type_names))
    if not any(declared):
        if any(record.get("time_window_status", "unverified_legacy") != "unverified_legacy" for record in records):
            raise ValueError("legacy records cannot claim verified or declared window status")
        metadata.update(time_window_status="unverified_legacy", time_window_warnings=[_LEGACY_WARNING])
        return metadata
    if "run_config" not in metadata or "window" not in metadata:
        raise ValueError("declared contracts require top-level run_config and effective window")
    contracts = [_contract(record["time_window"]) for record in records]
    for record, contract in zip(records, contracts):
        if record.get("time_window_status", "declared_compatible") != "declared_compatible":
            raise ValueError("declared records have incompatible time_window_status")
        if contract["run_config"] != metadata["run_config"] or contract["effective_window_ns"] != metadata["window"]:
            raise ValueError("top-level run/window disagrees with time_window contract")
    semantic = lambda contract: {k: v for k, v in contract.items() if k not in ("selections", "warnings")}
    if any(semantic(contract) != semantic(contracts[0]) for contract in contracts[1:]):
        raise ValueError("incompatible declared run/window contracts")
    merged = copy.deepcopy(contracts[0])
    selections = {json.dumps(selection, sort_keys=True) for contract in contracts for selection in contract["selections"]}
    merged["selections"] = [json.loads(selection) for selection in sorted(selections)]
    merged["warnings"] = sorted({warning for contract in contracts for warning in contract["warnings"]})
    metadata.update(time_window=merged, time_window_status="declared_compatible")
    return metadata
