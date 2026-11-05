"""Typed, vectorized audits of raw reference exports, not extraction certificates.

Run with ``python -m beamfastsim.flash_reference_export --help``. A successful
internal audit does not prove a closed art job, source extraction fidelity,
source-to-primary correspondence, or a particle-to-volume-product mapping.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import awkward as ak
import numpy as np
import uproot


class SchemaError(ValueError):
    """A required TTree, scalar branch, or exact stored dtype is missing."""


def _fields(dtype, names):
    return dict.fromkeys(names.split(), dtype)


_EVENT = {**_fields("uint32", "run subrun event"), "ordinal": "uint64"}
_COLLECTION = {**_EVENT, **_fields("uint64", "product index")}


def _pointer(name):
    return {name + "_state": "uint32", name + "_product": "uint64", name + "_key": "uint64"}


REQUIRED = {
    "schema": _fields("uint32", "version product_id_bytes ptr_key_bytes"),
    "completion": {"complete": "uint32", **_fields("uint64", "events ordinal_base subrun_products")},
    "requirements": {"kind": "uint32", "tag": "string", **_fields("uint64", "expected_events ordinal_base")},
    "storage": {**_fields("string", "group persistent export"), **_fields("uint32", "persistent_bytes export_bytes")},
    "events": {**_EVENT, **_fields("uint64", "nproducts nparticles ngenerators nsteps ndaughters nstatuses ptr_failures")},
    "products": {**_COLLECTION, **_fields("uint32", "kind scope"), "size": "uint64", **_fields("string", "role branch friendly label instance process type produced_type")},
    "steps": {**_COLLECTION, **_pointer("particle"), "volume": "uint64", **_fields("float64", "x y z post_x post_y post_z px py pz post_px post_py post_pz time proper_time"), **_fields("float32", "total_edep nonionizing_edep visible_edep length"), "end_process": "int32"},
    "particles": {**_COLLECTION, **_fields("uint64", "key id ndaughters"), **_fields("uint32", "stage start_volume end_volume start_g4status end_g4status end_defined"), **_fields("int32", "pdg creation stopping nsteps ion_float_level"), **_pointer("parent"), **_pointer("generator"), **_fields("float64", "x y z end_x end_y end_z"), **_fields("float32", "px py pz energy end_px end_py end_pz end_energy time proper_time end_time end_proper_time pre_last_ke end_ke track_length ion_excitation")},
    "daughters": {**_COLLECTION, "owner_key": "uint64", **_pointer("daughter")},
    "generators": {**_COLLECTION, **_fields("int32", "pdg gen_id"), **_fields("float64", "x y z px py pz energy time proper_time")},
    "statuses": {**_COLLECTION, **_fields("int32", "status ntracks killed_step killed_field"), "overflow": "uint32", **_fields("float32", "cpu real")},
    "stages": {**_COLLECTION, "stage": "uint32", "size": "uint64"},
    "volumes": {**_COLLECTION, **_fields("uint32", "stage copy"), **_fields("string", "name material")},
}
_EVENT_ID = ("run", "subrun", "event")
_PRODUCT_ID = (*_EVENT_ID, "product")
_SUBRUN_PRODUCT = ("run", "subrun", "product")
_TYPES = {1: "mu2e::SimParticleCollection", 2: "mu2e::GenParticleCollection", 3: "mu2e::StepPointMCCollection", 4: "mu2e::StatusG4", 5: "mu2e::PhysicalVolumeInfoMultiCollection"}
# Exact group names emitted by the exporter; key widths remain platform metadata.
_STORAGE = {
    "positions;step_vectors;step_times;generator_states": ("double", 8, "Float64", 8),
    "particle_momenta_times_energies_length_excitation;step_deposits_length;status_times": ("float", 4, "Float32", 4),
    "particle_ion_float_level": ("short", 2, "Int32", 4),
    "status_overflow;particle_end_defined": ("bool", 1, "UInt32", 4),
    "product_id": ("art::ProductID::value_type", None, "UInt64", 8),
    "ptr_key": ("art::Ptr::key_type", None, "UInt64", 8),
    "map_key;particle_id": ("cet::map_vector_key", None, "UInt64", 8),
    "step_volume": ("StepPointMC::VolumeId_type", None, "UInt64", 8),
}
_TABLE_KINDS = {"particles": 1, "generators": 2, "steps": 3, "statuses": 4}


@dataclass(frozen=True)
class ReferenceExport:
    tables: Mapping[str, Mapping[str, np.ndarray]]
    directory: str = "export"

    def __getitem__(self, name):
        return self.tables[name]


def read_export(path, directory="export"):
    """Read highest-cycle TTrees, retaining all required strings and scalar types."""
    tables = {}
    with uproot.open(path) as root:
        try:
            group = root[directory] if directory else root
        except KeyError as exc:
            raise SchemaError(f"missing export directory {directory!r}") from exc
        for name, contract in REQUIRED.items():
            try:
                tree = group[name]
            except KeyError as exc:
                raise SchemaError(f"missing TTree {name}") from exc
            if tree.classname != "TTree":
                raise SchemaError(f"{name}: expected TTree, got {tree.classname}")
            missing = set(contract) - set(tree.keys())
            if missing:
                raise SchemaError(f"{name}: missing branches {sorted(missing)}")
            columns = {}
            for branch, dtype in contract.items():
                leaf = tree[branch]
                if dtype == "string":
                    if leaf.typename not in ("std::string", "string", "char*"):
                        raise SchemaError(f"{name}.{branch}: expected string, got {leaf.typename}")
                    values = ak.to_numpy(leaf.array(library="ak"))
                    if values.ndim != 1 or values.dtype.kind not in "UOS":
                        raise SchemaError(f"{name}.{branch}: expected scalar string")
                else:
                    try:
                        stored = np.dtype(leaf.interpretation.numpy_dtype)
                    except (AttributeError, TypeError) as exc:
                        raise SchemaError(f"{name}.{branch}: expected scalar {dtype}") from exc
                    if stored != np.dtype(dtype):
                        raise SchemaError(f"{name}.{branch}: expected {dtype}, got {stored}")
                    values = leaf.array(library="np")
                    if values.ndim != 1 or values.dtype != np.dtype(dtype):
                        raise SchemaError(f"{name}.{branch}: expected scalar {dtype}")
                columns[branch] = values
            tables[name] = columns
    return ReferenceExport(tables, directory)


load_export = read_export


def _keys(table, names):
    arrays = [table[name] for name in names]
    result = np.empty(len(arrays[0]), dtype=[(f"f{i}", a.dtype) for i, a in enumerate(arrays)])
    for i, a in enumerate(arrays):
        result[f"f{i}"] = a
    return result


def _lookup(target, query):
    if not len(target):
        return np.full(len(query), -1, dtype=np.int64)
    order = np.argsort(target, kind="stable")
    pos = np.searchsorted(target[order], query)
    safe = np.minimum(pos, len(target) - 1)
    return np.where((pos < len(target)) & (target[order[safe]] == query), order[safe], -1)


def _unique(keys):
    return len(np.unique(keys)) == len(keys)


def _ordered(keys, index):
    if not len(keys):
        return True
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    expected = np.arange(len(keys), dtype=np.uint64) - np.repeat(starts, counts).astype(np.uint64)
    return bool(np.array_equal(index[order], expected))


def _counts(indices, length):
    return np.bincount(indices[indices >= 0], minlength=length).astype(np.uint64)


def _as_export(value, directory):
    return value if isinstance(value, ReferenceExport) else read_export(value, directory)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def audit_export(value, *, directory="export", expected_events=None, photon_role=None,
                 photon_tag=None, photon_cap=None, particle_role=None, particle_tag=None,
                 particle_cap=None, require_status=True):
    """Audit internal consistency without dropping or renormalizing failed events.

    Caps apply only to explicitly selected product roles/tags, never inferred
    particle stages, names, PDG codes, or the largest particle key. A tag is the
    exact ``label:instance:process`` string for cap selection. Configured
    requirement tags separately support omitted instance/process components;
    processless tags check presence, not independent art history resolution.
    Size equal to a cap fails.
    """
    for name, parameter in (("expected_events", expected_events), ("photon_cap", photon_cap), ("particle_cap", particle_cap)):
        if parameter is not None:
            _positive_integer(parameter, name)
    export = _as_export(value, directory)
    t = export.tables
    ev, products = t["events"], t["products"]
    ne = len(ev["run"])
    failures = []

    def check(ok, code, category="structure"):
        if not bool(ok):
            failures.append({"category": category, "code": code})

    event_keys = _keys(ev, _EVENT_ID)
    check(_unique(event_keys), "duplicate_event_id")
    check(_unique(_keys(ev, ("ordinal",))), "duplicate_source_ordinal")
    if expected_events is not None:
        check(ne == expected_events, "expected_event_count")
    schema, completion = t["schema"], t["completion"]
    check(len(schema["version"]) == 1, "schema_row_count")
    check(np.all(schema["version"] == 1), "schema_version")
    check(np.all(np.isin(schema["product_id_bytes"], (1, 2, 4, 8))), "product_id_width")
    check(np.all(np.isin(schema["ptr_key_bytes"], (4, 8))), "ptr_key_width")
    check(len(completion["complete"]) == 1, "completion_row_count")
    check(np.all(completion["complete"] == 1), "completion_marker")
    check(np.all(completion["events"] == ne), "completion_event_count")
    if len(completion["ordinal_base"]) == 1:
        base = int(completion["ordinal_base"][0])
        check(base + ne <= 2**64, "source_ordinal_overflow")
        if base + ne <= 2**64:
            check(np.array_equal(ev["ordinal"], np.arange(ne, dtype=np.uint64) + np.uint64(base)), "source_ordinal_order")

    requirements = t["requirements"]
    check(len(requirements["kind"]) > 0, "requirements_missing")
    check(np.all(np.isin(requirements["kind"], tuple(_TYPES))), "requirement_kind")
    check(all(np.any(requirements["kind"] == kind) for kind in (1, 2, 3, 5)), "required_requirement_kinds")
    check(_unique(_keys(requirements, ("kind", "tag"))), "duplicate_requirement")
    check(np.all(requirements["expected_events"] > 0), "requirement_expected_events_positive")
    check(np.all(requirements["expected_events"] == ne), "requirement_event_count")
    for field, code in (("expected_events", "requirement_expected_events_consistent"), ("ordinal_base", "requirement_ordinal_base_consistent")):
        check(len(np.unique(requirements[field])) == 1, code)
    if len(completion["ordinal_base"]) == 1:
        check(np.all(requirements["ordinal_base"] == completion["ordinal_base"][0]), "requirement_ordinal_base")

    storage = t["storage"]
    check(_unique(_keys(storage, ("group",))), "duplicate_storage_group")
    check(set(storage["group"]) == set(_STORAGE), "storage_groups")
    for group, (persistent, width, destination, out_width) in _STORAGE.items():
        selected = storage["group"] == group
        check(np.count_nonzero(selected) == 1, "storage_group_" + group)
        check(np.all(storage["persistent"][selected] == persistent), "storage_persistent_" + group)
        check(np.all(storage["export"][selected] == destination), "storage_export_" + group)
        check(np.all(storage["export_bytes"][selected] == out_width), "storage_export_bytes_" + group)
        widths = storage["persistent_bytes"][selected]
        check(np.all(widths == width) if width is not None else np.all(np.isin(widths, (1, 2, 4, 8))), "storage_persistent_bytes_" + group)
    for group, field in (("product_id", "product_id_bytes"), ("ptr_key", "ptr_key_bytes")):
        if len(schema[field]) == 1:
            check(np.all(storage["persistent_bytes"][storage["group"] == group] == schema[field][0]), "storage_schema_" + group)

    key_columns = {
        "product_id": [rows["product"] for rows in t.values() if "product" in rows],
        "ptr_key": [],
        "map_key;particle_id": [t["particles"]["key"], t["particles"]["id"], t["daughters"]["owner_key"], t["volumes"]["index"]],
        "step_volume": [t["steps"]["volume"]],
    }
    for name, prefix in (("steps", "particle"), ("particles", "parent"), ("particles", "generator"), ("daughters", "daughter")):
        key_columns["product_id"].append(t[name][prefix + "_product"])
        key_columns["ptr_key"].append(t[name][prefix + "_key"])
    for group, columns in key_columns.items():
        widths = storage["persistent_bytes"][storage["group"] == group]
        if len(widths) == 1 and int(widths[0]) in (1, 2, 4, 8):
            maximum = np.uint64(2 ** (8 * int(widths[0])) - 1)
            check(all(np.all(column <= maximum) for column in columns), "storage_value_range_" + group)

    scope = products["scope"]
    ep, sp = scope == 0, scope == 1
    check(np.all(ep | sp), "product_scope")
    check(np.all(products["index"] == 0), "product_index")
    check(np.all(products["product"] != 0), "null_product_id")
    check(np.all(np.isin(products["kind"], tuple(_TYPES))), "product_kind")
    check(np.all((products["kind"] == 5) == sp), "product_kind_scope")
    check(_unique(_keys(products, ("scope", *_PRODUCT_ID))), "duplicate_product_row")
    check(np.all(products["event"][sp] == 0) and np.all(products["ordinal"][sp] == 0), "subrun_product_identity")
    check(np.all(completion["subrun_products"] == np.count_nonzero(sp)), "completion_subrun_products")
    for kind, typename in _TYPES.items():
        check(np.all(products["type"][products["kind"] == kind] == typename), f"product_type_{kind}")
    for name in ("role", "branch", "friendly", "label", "process", "type", "produced_type"):
        check(np.all(products[name] != ""), f"empty_provenance_{name}")
    pi = _lookup(event_keys, _keys(products, _EVENT_ID))
    check(np.all(pi[ep] >= 0), "product_outside_event_ledger")
    valid = ep & (pi >= 0)
    check(np.all(products["ordinal"][valid] == ev["ordinal"][pi[valid]]), "product_ordinal")
    check(np.array_equal(_counts(pi[ep], ne), ev["nproducts"]), "event_product_count")
    for kind in (3, 4):
        configured = ep & (products["kind"] == kind)
        role_rows = {name: values[configured] for name, values in products.items()}
        check(_unique(_keys(role_rows, (*_EVENT_ID, "role"))), f"duplicate_configured_role_{kind}")
        _, role_counts = np.unique(products["role"][configured], return_counts=True)
        check(np.all(role_counts == ne), f"required_product_role_coverage_{kind}")
    sr = _keys(ev, ("run", "subrun"))
    check(np.all(_lookup(sr, _keys(products, ("run", "subrun")))[sp] >= 0), "product_outside_subrun_ledger")
    # Requirement coverage cannot certify independent configuration or process-history selection.
    subruns = np.unique(sr)
    for kind, tag in zip(requirements["kind"], requirements["tag"]):
        parts = str(tag).split(":")
        valid_tag = 1 <= len(parts) <= 3 and bool(parts[0])
        check(valid_tag, "requirement_tag")
        if not valid_tag:
            continue
        label, instance, process = (parts + [""] * (3 - len(parts)))
        selected = (products["kind"] == kind) & (products["label"] == label) & (products["instance"] == instance)
        if process:
            selected &= products["process"] == process
        if kind == 5:
            selected &= sp
            indices = _lookup(subruns, _keys(products, ("run", "subrun")))
            counts = _counts(indices[selected], len(subruns))
        else:
            selected &= ep
            counts = _counts(pi[selected], ne)
        check(np.all(counts == 1) if process else np.all(counts >= 1), f"required_product_tag_coverage_{kind}")
    product_keys = _keys(products, _PRODUCT_ID)
    row_product_indices = {}
    event_indices = {}
    for name in ("particles", "generators", "steps", "statuses", "daughters"):
        rows = t[name]
        ei = _lookup(event_keys, _keys(rows, _EVENT_ID))
        event_indices[name] = ei
        check(np.all(ei >= 0), f"{name}_outside_event_ledger")
        valid = ei >= 0
        check(np.all(rows["ordinal"][valid] == ev["ordinal"][ei[valid]]), f"{name}_ordinal")
        check(np.array_equal(_counts(ei, ne), ev["n" + name]), f"event_{name}_count")
        ri = _lookup(product_keys, _keys(rows, _PRODUCT_ID))
        row_product_indices[name] = ri
        check(np.all(ri >= 0), f"{name}_undeclared_product")
        valid = ri >= 0
        kind = 1 if name == "daughters" else _TABLE_KINDS[name]
        check(np.all(products["kind"][ri[valid]] == kind) and np.all(scope[ri[valid]] == 0), f"{name}_product_kind")
        if name != "daughters":
            check(_ordered(_keys(rows, _PRODUCT_ID), rows["index"]), f"{name}_collection_index_order")
            counts = _counts(ri, len(scope))
            declared = (products["kind"] == kind) & ep
            check(np.array_equal(counts[declared], products["size"][declared]), f"{name}_declared_size")
    status_products = (products["kind"] == 4) & ep
    check(np.all(products["size"][status_products] == 1), "status_product_size")
    particles, generators, daughters = t["particles"], t["generators"], t["daughters"]
    particle_keys = _keys(particles, (*_PRODUCT_ID, "key"))
    check(_unique(particle_keys), "duplicate_particle_key")
    check(_unique(_keys(particles, (*_PRODUCT_ID, "id"))), "duplicate_particle_id")
    owner = _lookup(particle_keys, _keys(daughters, (*_PRODUCT_ID, "owner_key")))
    check(np.all(owner >= 0), "daughter_owner_unresolved")
    check(np.array_equal(_counts(owner, len(particle_keys)), particles["ndaughters"]), "particle_daughter_count")
    check(_ordered(_keys(daughters, (*_PRODUCT_ID, "owner_key")), daughters["index"]), "daughter_index_order")
    generator_keys = _keys(generators, (*_PRODUCT_ID, "index"))
    ptr_failure_counts = np.zeros(ne, dtype=np.uint64)
    for name, prefix, target in (("steps", "particle", particle_keys), ("particles", "parent", particle_keys), ("particles", "generator", generator_keys), ("daughters", "daughter", particle_keys)):
        rows = t[name]
        state, product, key = (rows[prefix + suffix] for suffix in ("_state", "_product", "_key"))
        check(np.all(state <= 3), f"{name}_{prefix}_state")
        # State is authoritative; raw null keys retain platform sentinels without inferred nullness.
        if name in ("steps", "daughters"):
            check(not np.any(state == 0), f"{name}_{prefix}_null", "physics")
        check(np.all(product[state != 0] != 0), f"{name}_{prefix}_nonnull_product")
        ptr = _keys(rows, (*_EVENT_ID, prefix + "_product", prefix + "_key"))
        resolved = _lookup(target, ptr)
        check(np.all(resolved[state == 1] >= 0), f"{name}_{prefix}_unresolved")
        check(not np.any(state >= 2), f"{name}_{prefix}_unavailable", "physics")
        ptr_failure_counts += _counts(event_indices[name][state >= 2], ne)
    check(np.array_equal(ptr_failure_counts, ev["ptr_failures"]), "event_ptr_failure_count")

    subrun_product_keys = _keys(products, _SUBRUN_PRODUCT)
    stages, volumes = t["stages"], t["volumes"]
    for name in ("stages", "volumes"):
        rows = t[name]
        check(np.all(rows["event"] == 0) and np.all(rows["ordinal"] == 0), f"{name}_subrun_identity")
        ri = _lookup(subrun_product_keys[sp], _keys(rows, _SUBRUN_PRODUCT))
        check(np.all(ri >= 0), f"{name}_undeclared_volume_product")
    check(_unique(_keys(stages, (*_SUBRUN_PRODUCT, "stage"))), "duplicate_volume_stage")
    check(np.array_equal(stages["index"], stages["stage"]), "volume_stage_index")
    check(_ordered(_keys(stages, _SUBRUN_PRODUCT), stages["index"]), "volume_stage_order")
    si = _lookup(subrun_product_keys[sp], _keys(stages, _SUBRUN_PRODUCT))
    check(np.array_equal(_counts(si, np.count_nonzero(sp)), products["size"][sp]), "volume_product_stage_count")
    stage_keys = _keys(stages, (*_SUBRUN_PRODUCT, "stage"))
    vi = _lookup(stage_keys, _keys(volumes, (*_SUBRUN_PRODUCT, "stage")))
    check(np.all(vi >= 0), "volume_stage_unresolved")
    check(np.array_equal(_counts(vi, len(stage_keys)), stages["size"]), "volume_stage_size")
    check(_unique(_keys(volumes, (*_SUBRUN_PRODUCT, "stage", "index"))), "duplicate_volume_key")

    statuses = t["statuses"]
    if require_status:
        check(np.all(ev["nstatuses"] > 0) and ne > 0, "required_status_missing", "physics")
    for field in ("status", "overflow", "killed_step", "killed_field"):
        check(np.all(statuses[field] == 0), "status_" + field, "physics")
    check(np.all(statuses["ntracks"] >= 0), "status_negative_tracks", "physics")
    for field in ("cpu", "real"):
        check(np.all(np.isfinite(statuses[field]) & (statuses[field] >= 0)), "status_" + field, "physics")
    cap_checks = {}
    for label, kind, role, tag, cap in (("photon", 3, photon_role, photon_tag, photon_cap), ("particle", 1, particle_role, particle_tag, particle_cap)):
        if cap is not None and (cap <= 0 or (role is None and tag is None)):
            raise ValueError(f"{label}_cap requires a positive cap and an explicit role or tag")
        if role is None and tag is None:
            cap_checks[label] = "not_requested"
            continue
        selected = ep & (products["kind"] == kind)
        if role is not None:
            selected &= products["role"] == role
        if tag is not None:
            tags = np.char.add(np.char.add(np.char.add(np.char.add(products["label"].astype(str), ":"), products["instance"].astype(str)), ":"), products["process"].astype(str))
            selected &= tags == tag
        check(np.all(_counts(pi[selected], ne) == 1) and ne > 0, label + "_product_selection", "physics")
        if cap is not None:
            check(np.all(products["size"][selected] < cap), label + "_cap_headroom", "physics")
        cap_checks[label] = "checked" if cap is not None else "selection_only"
    structural = not any(f["category"] == "structure" for f in failures)
    physics = not any(f["category"] == "physics" for f in failures)
    return {
        "table_structure_valid": structural,
        "physics_checks_passed": physics,
        "physics_completeness": False,
        "audit_passed": structural and physics,
        "events": ne,
        "rows": {name: len(columns[next(iter(columns))]) for name, columns in t.items()},
        "failures": failures,
        "cap_checks": cap_checks,
        "denominator_policy": "all ledger events retained; no survivor renormalization",
        "scope": "Internal typed-table, declared-requirement, ledger, Ptr, status and requested cap checks only; no runtime or extraction certificate.",
        "volume_scope": "Structural product/stage/table-key validation only; copy numbers are not keys and no particle-to-volume-product mapping is inferred.",
        "requirements_scope": "Coverage of exported declarations only; incomplete requirements versus independent job configuration are not certified.",
        "external_checks_required": ["independent job configuration and required-product completeness", "closed input/output files and successful external job audit", "independent persistent-product extraction comparison", "exact saved EventID and ordered source assignment agreement", "source-to-primary correspondence", "explicit particle-to-volume-product mapping", "configured recorder and new-particle cap headroom"],
    }


def lookup_volumes(value, *, run, subrun, product, stage, index, directory="export"):
    """Resolve explicit volume table identities, never a step copy number."""
    export = _as_export(value, directory)
    validated = []
    for name, value, dtype in zip(("run", "subrun", "product", "stage", "index"),
                                  (run, subrun, product, stage, index),
                                  ("uint32", "uint32", "uint64", "uint32", "uint64")):
        a = np.asarray(value)
        if a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a > np.iinfo(dtype).max):
            raise ValueError(f"{name} must contain integers in the {dtype} range")
        validated.append(a.astype(dtype, copy=False))
    arrays = np.broadcast_arrays(*validated)
    query = {name: a.reshape(-1) for name, a in zip(("run", "subrun", "product", "stage", "index"), arrays)}
    names = (*_SUBRUN_PRODUCT, "stage", "index")
    keys = _keys(export["volumes"], names)
    if not _unique(keys):
        raise ValueError("duplicate volume identities")
    return _lookup(keys, _keys(query, names)).reshape(arrays[0].shape)


def compare_source_exports(left, right, *, source_role, directory="export", source_tag=None,
                           source_kind=3):
    """Compare exact ordered source rows and provenance, paired by EventID/role.

    Select kind 1 (particles, including ordered daughters), 2 (generators), or 3
    (steps). Compare each configured source collection separately. ProductIDs
    are exact provenance: differing process histories are not silently aliased.
    Float equality is byte equality, including signed zeros. This compares
    exported source tables, not independent persistent products.
    """
    if source_kind not in (1, 2, 3):
        raise ValueError("source_kind must be 1 (particles), 2 (generators), or 3 (steps)")
    source_table = {1: "particles", 2: "generators", 3: "steps"}[source_kind]
    exports = [_as_export(value, directory) for value in (left, right)]
    reports = [audit_export(value, require_status=False) for value in exports]
    if not all(report["table_structure_valid"] for report in reports):
        return {"equal": False, "failures": ["invalid_table_structure"], "scope": "exported source tables only"}
    if any(f["code"].endswith(("_unavailable", "_null")) for report in reports for f in report["failures"]):
        return {"equal": False, "failures": ["invalid_reference_state"], "scope": "exported source tables only"}
    selected_tables = []
    failures = []
    for export in exports:
        products = export["products"]
        selected = (products["scope"] == 0) & (products["kind"] == source_kind) & (products["role"] == source_role)
        if source_tag is not None:
            tags = np.char.add(np.char.add(np.char.add(np.char.add(products["label"].astype(str), ":"), products["instance"].astype(str)), ":"), products["process"].astype(str))
            selected &= tags == source_tag
        pi = _lookup(_keys(export["events"], _EVENT_ID), _keys(products, _EVENT_ID))
        if not np.all(_counts(pi[selected], len(export["events"]["run"])) == 1):
            failures.append("source_product_selection")
        source_products = {name: a[selected] for name, a in products.items()}
        source_keys = _keys(source_products, _PRODUCT_ID)
        rows = export[source_table]
        keep = _lookup(source_keys, _keys(rows, _PRODUCT_ID)) >= 0
        source_rows = {name: a[keep] for name, a in rows.items()}
        rows = export["daughters"]
        keep = _lookup(source_keys, _keys(rows, _PRODUCT_ID)) >= 0
        source_daughters = {name: a[keep] for name, a in rows.items()}
        selected_tables.append((source_products, source_rows, source_daughters))
    comparisons = [("provenance", 0, _EVENT_ID), ("rows", 1, (*_PRODUCT_ID, "index"))]
    if source_kind == 1:
        comparisons.append(("daughters", 2, (*_PRODUCT_ID, "owner_key", "index")))
    for label, offset, order_names in comparisons:
        a, b = (tables[offset] for tables in selected_tables)
        oa, ob = (np.argsort(_keys(table, order_names), kind="stable") for table in (a, b))
        for name in a:
            av, bv = a[name][oa], b[name][ob]
            equal = av.shape == bv.shape
            if equal:
                if av.dtype.kind in "UOS" and bv.dtype.kind in "UOS":
                    equal = np.array_equal(av, bv)
                else:
                    equal = av.dtype == bv.dtype and av.tobytes() == bv.tobytes()
            if not equal:
                failures.append(f"source_{label}_{name}")
    a, b = (export["events"] for export in exports)
    for field in (*_EVENT_ID, "ordinal"):
        if not np.array_equal(a[field], b[field]):
            failures.append("source_event_assignment_" + field)
    return {"equal": not failures, "failures": failures, "scope": "Exact exported source products/rows only; no runtime or independent extraction certificate."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--directory", default="export")
    parser.add_argument("--out", required=True, help="New JSON file; existing files are never overwritten")
    parser.add_argument("--expected-events", type=int)
    parser.add_argument("--photon-role")
    parser.add_argument("--photon-tag")
    parser.add_argument("--photon-cap", type=int)
    parser.add_argument("--particle-role")
    parser.add_argument("--particle-tag")
    parser.add_argument("--particle-cap", type=int)
    parser.add_argument("--source-only", action="store_true", help="Do not require StatusG4 for source-only reads")
    args = parser.parse_args(argv)
    options = vars(args).copy()
    path, out = options.pop("input"), options.pop("out")
    options["require_status"] = not options.pop("source_only")
    try:
        report = audit_export(path, **options)
    except (SchemaError, ValueError) as exc:
        report = {"table_structure_valid": False, "physics_completeness": False, "audit_passed": False, "failures": [{"category": "schema_or_configuration", "code": str(exc)}], "scope": "No runtime or extraction certificate."}
    with Path(out).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if report["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
