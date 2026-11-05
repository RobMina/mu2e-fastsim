"""Pack audited single-root whole-event draws, including valid empty responses."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping

import numpy as np


SCHEMA_VERSION = 2
ENTRY_INTEGER_FIELDS = ("source_shard", "run", "subrun", "event", "product_id", "particle_key",
                        "disk", "crossing_index", "pdg", "g4_status")
ENTRY_REAL_FIELDS = ("x", "y", "z", "px", "py", "pz", "time")
ENTRY_FIELDS = ("source_sha256",) + ENTRY_INTEGER_FIELDS + ENTRY_REAL_FIELDS
STEP_INTEGER_FIELDS = ("root_index", "crystal", "sim_product_id", "sim_particle_key", "n_compress")
STEP_REAL_FIELDS = ("x", "y", "z", "time", "energy_g4", "energy_birks", "momentum_in")
STEP_FIELDS = STEP_INTEGER_FIELDS + STEP_REAL_FIELDS
MANIFEST_FIELDS = ("source_sha256", "run", "subrun", "event", "product_id", "particle_key",
                   "disk", "crossing_index", "g4_status")
BOUNDARY_POLICY = "first_incoming_outside_passive_material_whole_event"


def _columns(table, fields, name):
    if not isinstance(table, Mapping) or not all(field in table for field in fields):
        raise ValueError(f"missing required {name} columns")
    arrays = {field: np.asarray(table[field]) for field in fields}
    if any(array.ndim != 1 for array in arrays.values()):
        raise ValueError(f"{name} columns must be one-dimensional")
    if len({array.size for array in arrays.values()}) != 1:
        raise ValueError(f"{name} column lengths disagree")
    return arrays


def _integers(table, fields):
    for name in fields:
        array = table[name]
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must have integer dtype")
        if name != "pdg" and np.any(array < 0):
            raise ValueError(f"negative {name}")
        if np.any(array > np.iinfo(np.int64).max):
            raise ValueError(f"{name} exceeds supported int64 range")
        table[name] = array.astype(np.int64)


def _reals(table, fields):
    for name in fields:
        array = table[name]
        if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite real numbers")


def _hashes(values, name):
    if values.dtype.kind != "U" or not np.all(np.char.str_len(values) == 64):
        raise ValueError(f"{name} must be lowercase SHA256 text")
    if bool(np.any(np.char.strip(values, "0123456789abcdef") != "")):
        raise ValueError(f"{name} must be lowercase SHA256 text")


def _keys(table, fields):
    return np.rec.fromarrays([table[field] for field in fields], names=fields)


def _provenance(provenance):
    gates = ("unfiltered_retention", "complete_incident_census", "zero_response_retained",
             "completed_transport")
    if not isinstance(provenance, Mapping) or not all(provenance.get(name) is True for name in gates):
        raise ValueError("audited unfiltered retention, incident census, zero-response and completed transport required")
    for name in ("geometry_sha256", "expanded_fcl_sha256", "incident_manifest_sha256"):
        if not isinstance(provenance.get(name), str) or re.fullmatch("[0-9a-f]{64}", provenance[name]) is None:
            raise ValueError(f"invalid provenance: {name}")
    for name in ("physics_release", "boundary_definition"):
        if not isinstance(provenance.get(name), str) or not provenance[name].strip():
            raise ValueError(f"missing provenance: {name}")
    if provenance.get("boundary_policy") != BOUNDARY_POLICY:
        raise ValueError("unsupported boundary policy")
    try:
        return json.dumps(dict(provenance), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance must be finite JSON data") from exc


def pack(entries, steps, *, provenance, incident_manifest=None):
    """Retain one designated incident state and all causally assigned event steps.

    The independent manifest must enumerate every designated incident root,
    including those with no crystal response. This contract cannot audit its
    producer: its provenance gates and checksum still require external evidence.
    Only status-zero transport enters this packer; retained failures need a
    separate failure ledger, never empty-root promotion or silent dropping.
    There is one first incoming crossing outside passive disk material per
    independent source event. Re-entries and both-disk deposits remain part of
    that entire event draw, not additional independent conditioning rows.
    """
    provenance_json = _provenance(provenance)
    entries = _columns(entries, ENTRY_FIELDS, "entry")
    steps = _columns(steps, STEP_FIELDS, "step")
    manifest = _columns(incident_manifest, MANIFEST_FIELDS, "incident manifest")
    _integers(entries, ENTRY_INTEGER_FIELDS)
    _integers(steps, STEP_INTEGER_FIELDS)
    _integers(manifest, MANIFEST_FIELDS[1:])
    _reals(entries, ENTRY_REAL_FIELDS)
    _reals(steps, STEP_REAL_FIELDS)
    for table in (entries, manifest):
        _hashes(table["source_sha256"], "source_sha256")
        if np.any(table["g4_status"] != 0):
            raise ValueError("incomplete or failed transport cannot be a valid response")
        if np.any((table["disk"] < 0) | (table["disk"] > 1)):
            raise ValueError("disk outside supported two-disk geometry")
    if np.any(entries["product_id"] == 0) or np.any(manifest["product_id"] == 0) or np.any(steps["sim_product_id"] == 0):
        raise ValueError("invalid zero persistent product identity")
    if np.any(entries["pdg"] == 0):
        raise ValueError("incident pdg cannot be zero")
    if np.any((entries["px"] == 0) & (entries["py"] == 0) & (entries["pz"] == 0)):
        raise ValueError("incident momentum must be nonzero")
    n = len(entries["event"])
    event_fields = ("source_sha256", "run", "subrun", "event")
    for table in (entries, manifest):
        if len(np.unique(_keys(table, event_fields))) != len(table["event"]):
            raise ValueError("one complete incident root per source event required")
    entry_keys, manifest_keys = _keys(entries, MANIFEST_FIELDS), _keys(manifest, MANIFEST_FIELDS)
    if not np.array_equal(np.sort(entry_keys), np.sort(manifest_keys)):
        raise ValueError("entries do not match complete independent incident manifest")
    source_shard = np.unique(_keys(entries, ("source_sha256", "source_shard")))
    if len(np.unique(source_shard["source_sha256"])) != len(source_shard):
        raise ValueError("one source artifact cannot cross development/held-out shards")
    if len(np.unique(source_shard["source_shard"])) != len(source_shard):
        raise ValueError("each source shard must name a unique source artifact")
    idx = steps["root_index"]
    if np.any(idx >= n):
        raise ValueError("invalid root_index")
    if np.any(steps["energy_g4"] < 0) or np.any(steps["energy_birks"] < 0):
        raise ValueError("negative step energy")
    if np.any(steps["momentum_in"] < 0) or np.any(steps["n_compress"] == 0):
        raise ValueError("invalid compressed-step momentum or constituent count")
    order = np.argsort(idx, kind="stable")
    counts = np.bincount(idx, minlength=n)
    packed = {"entry_" + name: entries[name].copy() for name in ENTRY_FIELDS}
    packed.update({"step_" + name: steps[name][order].copy() for name in STEP_FIELDS})
    packed["step_offsets"] = np.r_[0, np.cumsum(counts)].astype(np.int64)
    packed["development"] = entries["source_shard"] % 2 == 1
    packed["schema_version"] = np.asarray(SCHEMA_VERSION, dtype=np.int64)
    packed["provenance_json"] = np.asarray(provenance_json)
    return packed
