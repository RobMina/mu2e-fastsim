"""Pack correlated entrance-set diagnostic recordings; never certify a library."""
from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from .library import _columns, _hashes, _integers, _keys, _reals


SCHEMA_VERSION = 1
FRAME = "G4_world"
UNITS = "mm_MeV_ns"
PRODUCT_ID_ENCODING = "art_ProductID_value_uint32"
DEFAULT_MAX_ROWS = 2_000_000
EVENT_INTS = ("source_shard", "run", "subrun", "event", "root_product_id", "root_key",
              "transport_status", "recording_complete", "topology_supported")
EVENT_FIELDS = ("source_sha256",) + EVENT_INTS
MANIFEST_FIELDS = ("source_sha256", "source_shard", "run", "subrun", "event", "root_product_id", "root_key")
INCIDENT_INTS = ("event_index", "sequence", "product_id", "particle_key", "visit", "role", "pdg")
INCIDENT_REALS = ("x", "y", "z", "px", "py", "pz", "kinetic_energy", "time", "proper_time")
INCIDENT_FIELDS = INCIDENT_INTS + INCIDENT_REALS
STEP_INTS = ("event_index", "product_id", "index", "crystal", "sim_product_id", "sim_particle_key", "n_compress")
STEP_REALS = ("x", "y", "z", "time", "energy_g4", "energy_birks", "momentum_in")
STEP_FIELDS = STEP_INTS + STEP_REALS
INITIAL_EXTERIOR, RETURN, BORN_INSIDE, UNCLASSIFIED = range(4)


def _unique(table, fields, label):
    if len(np.unique(_keys(table, fields))) != len(table[fields[0]]):
        raise ValueError(f"duplicate {label}")


def _offsets(indices, n):
    return np.r_[0, np.cumsum(np.bincount(indices, minlength=n))].astype(np.int64)


def pack(events, incidents, steps, *, independent_manifest, provenance, max_rows=DEFAULT_MAX_ROWS):
    """Retain complete input events and observed incident sets without training promotion.

    Roles and root assignments are supplied recorder/audit classifications, not
    inferred by matching position, time or response. The independent manifest is
    pre-response evidence, not a copy made from surviving detector rows. This
    structural validator cannot certify the producer or feedback closure.
    """
    events = _columns(events, EVENT_FIELDS, "event")
    manifest = _columns(independent_manifest, MANIFEST_FIELDS, "independent manifest")
    incidents = _columns(incidents, INCIDENT_FIELDS, "incident")
    steps = _columns(steps, STEP_FIELDS, "step")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("positive integer max_rows required")
    if sum(len(table[field]) for table, field in ((events, "event"), (manifest, "event"),
                                                 (incidents, "event_index"), (steps, "event_index"))) > max_rows:
        raise ValueError("recording exceeds bounded batch row budget")
    for table, fields in ((events, EVENT_INTS), (manifest, MANIFEST_FIELDS[1:]),
                          (incidents, INCIDENT_INTS), (steps, STEP_INTS)):
        _integers(table, fields)
    for table, fields in ((incidents, INCIDENT_REALS), (steps, STEP_REALS)):
        _reals(table, fields)
        if any(table[field].dtype.kind != "f" for field in fields):
            raise ValueError("recorded real fields must retain floating storage")
    for table in (events, manifest):
        _hashes(table["source_sha256"], "source_sha256")
        _unique(table, ("source_sha256", "run", "subrun", "event"), "source event")
        if np.any(table["root_product_id"] == 0):
            raise ValueError("designated root requires a persistent product")
    if not np.array_equal(np.sort(_keys(events, MANIFEST_FIELDS)), np.sort(_keys(manifest, MANIFEST_FIELDS))):
        raise ValueError("events must equal the independent pre-response manifest")
    source_shards = np.unique(_keys(events, ("source_sha256", "source_shard")))
    if (len(np.unique(source_shards["source_sha256"])) != len(source_shards)
            or len(np.unique(source_shards["source_shard"])) != len(source_shards)):
        raise ValueError("each source artifact and shard must correspond one-to-one")
    for field in ("recording_complete", "topology_supported"):
        if np.any(events[field] > 1):
            raise ValueError(f"{field} must be a zero/one flag")
    n = len(events["event"])
    for table in (incidents, steps):
        if np.any(table["event_index"] >= n):
            raise ValueError("row refers to an unknown event")
        if np.any(table["product_id"] == 0):
            raise ValueError("row requires persistent product identity")
    _unique(incidents, ("event_index", "sequence"), "incident sequence")
    _unique(incidents, ("event_index", "product_id", "particle_key", "visit"), "incident visit")
    _unique(steps, ("event_index", "product_id", "index"), "shower step")
    if np.any(incidents["role"] > UNCLASSIFIED):
        raise ValueError("unknown incident role")
    if np.any(incidents["pdg"] == 0) or np.any(incidents["visit"] == 0):
        raise ValueError("incident requires species and positive visit")
    crossing = (incidents["role"] == INITIAL_EXTERIOR) | (incidents["role"] == RETURN)
    if np.any(crossing & (incidents["px"] == 0) & (incidents["py"] == 0) & (incidents["pz"] == 0)):
        raise ValueError("crossing momentum must be nonzero")
    if np.any(incidents["kinetic_energy"] < 0) or np.any(incidents["proper_time"] < 0):
        raise ValueError("negative kinetic energy or proper time")
    if (np.any(steps["sim_product_id"] == 0) or np.any(steps["n_compress"] == 0)
            or any(np.any(steps[field] < 0) for field in ("energy_g4", "energy_birks", "momentum_in"))):
        raise ValueError("invalid shower-step state")
    if not isinstance(provenance, Mapping) or provenance.get("purpose") != "diagnostic_entrance_set_recording":
        raise ValueError("recording-purpose provenance required")
    for field in ("domain_sha256", "geometry_sha256", "expanded_fcl_sha256", "independent_manifest_sha256"):
        _hashes(np.asarray([provenance.get(field)]), field)
    for field, expected in (("coordinate_frame", FRAME), ("units", UNITS),
                             ("product_id_encoding", PRODUCT_ID_ENCODING)):
        if provenance.get(field) != expected:
            raise ValueError(f"unsupported or missing {field}")
    for table, fields in ((events, ("root_product_id",)), (manifest, ("root_product_id",)),
                          (incidents, ("product_id",)), (steps, ("product_id", "sim_product_id"))):
        if any(np.any(table[field] > np.iinfo(np.uint32).max) for field in fields):
            raise ValueError("ProductID exceeds art uint32 value encoding")
    policy = json.dumps(dict(provenance), sort_keys=True, allow_nan=False)
    incident_order = np.lexsort((incidents["sequence"], incidents["event_index"]))
    step_order = np.argsort(steps["event_index"], kind="stable")
    packed = {"event_" + field: events[field].copy() for field in EVENT_FIELDS}
    packed.update({"incident_" + field: incidents[field][incident_order].copy() for field in INCIDENT_FIELDS})
    packed.update({"step_" + field: steps[field][step_order].copy() for field in STEP_FIELDS})
    packed.update(incident_offsets=_offsets(incidents["event_index"], n),
                  step_offsets=_offsets(steps["event_index"], n),
                  development=events["source_shard"] % 2 == 1,
                  failure_or_incomplete=(events["transport_status"] != 0) | (events["recording_complete"] == 0),
                  initial_exterior_candidate=incidents["role"][incident_order] == INITIAL_EXTERIOR,
                  library_eligible=np.zeros(n, dtype=bool),
                  schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
                  artifact_kind=np.asarray("entrance_set_diagnostic_recording"),
                  provenance_json=np.asarray(policy))
    return packed
