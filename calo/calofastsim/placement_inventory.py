"""Bounded structural audit of compact v2 inventories, not native geometry validation.

Binary JSONL input is streamed. A footer alone cannot prove fsync/close or job
success: acceptance also needs an independently obtained, hash-bound receipt.
No result is a containment/navigation certificate or a library eligibility claim.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re

SCHEMA = "mu2e-placement-inventory-v2"
SCOPE = "all_world_physical_placements"
UNITS = "native_G4_mm_rad"
FRAME = "local_object_to_parent; world row has no parent"
NAME_ENCODING = "JSON U+00xx escapes preserve native bytes, not UTF-8 decoding"
IDENTITY = "zero-based inventory-local first-encounter dictionaries and expanded preorder placements"
UNSUPPORTED_REASON = "not_allowlisted; boolean operands/transforms and other solid topology not resolved"
_U64 = 2**64 - 1
_I32 = (-(2**31), 2**31 - 1)
_BEGIN = set("type schema g4_run scope units frame name_encoding identity certificate max_depth max_placements max_bytes max_rows helper_allocator_max_bytes stream_info_max_bytes polycone_max_planes candidate_roots".split())
_END = set("type complete_physical_tree resolved_all_solids placements unique_paths materials solids logical_volumes candidate_placements unsupported_solid_placements deepest rows_including_footer bytes_including_footer_decimal root_placements".split())
_PLACEMENT = set("type id parent_id depth pv copy logical_volume_id selected candidate_root_id local_translation local_object_rotation".split())


class InventoryError(ValueError):
    """Malformed, incomplete, inconsistent, or over-budget inventory."""


def _need(ok, message):
    if not ok:
        raise InventoryError(message)


def _integer(value, name, low=0, high=_U64):
    _need(type(value) is int and low <= value <= high, f"invalid integer {name}")
    return value


def _boolean(value, name):
    _need(type(value) is bool, f"invalid boolean {name}")
    return value


def _text(value, name):
    _need(type(value) is str and all(ord(c) <= 255 for c in value), f"invalid byte-preserving string {name}")
    return value


def _keys(value, keys, name):
    _need(type(value) is dict and value.keys() == keys, f"unexpected fields in {name}")


def _real(value):
    _need(type(value) in (int, float) and math.isfinite(value), "invalid finite native number")


def _vector(value, n):
    _need(type(value) is list and len(value) == n, "invalid native vector")
    for x in value:
        _real(x)


def _pairs(pairs):
    result = {}
    for k, v in pairs:
        _need(k not in result, f"duplicate JSON key: {k}")
        result[k] = v
    return result


def _constant(value):
    raise InventoryError(f"nonfinite JSON constant: {value}")


def _float(value):
    result = float(value)
    _need(math.isfinite(result), "nonfinite JSON number")
    return result


def _json_int(value):
    _need(len(value.lstrip("-")) <= 20, "JSON integer token bound")
    return int(value)


@dataclass(frozen=True)
class Limits:
    """Independent reader budgets, never enlarged by an untrusted header."""

    max_bytes: int = 256 * 1024 * 1024
    max_line_bytes: int = 1024 * 1024
    max_rows: int = 2_000_000
    max_placements: int = 1_000_000

    def __post_init__(self):
        for name in ("max_bytes", "max_line_bytes", "max_rows", "max_placements"):
            _integer(getattr(self, name), name, 1)
        _need(self.max_placements <= 1_000_000, "reader placement ceiling is 1000000")


@dataclass(frozen=True)
class CompletionReceipt:
    """Caller-supplied independent evidence; not inferred from inventory bytes.

    The caller is responsible for authenticity and provenance of this receipt.
    Both flags must describe this exact SHA-256 artifact, not a planned job.
    """

    sha256: str
    writer_closed: bool
    job_success: bool

    def __post_init__(self):
        _need(type(self.sha256) is str and re.fullmatch(r"[0-9a-f]{64}", self.sha256) is not None, "invalid receipt sha256")
        _boolean(self.writer_closed, "writer_closed")
        _boolean(self.job_success, "job_success")


@dataclass(frozen=True)
class PlacementColumns:
    """Primitive columns, not one dictionary or full path per placement."""

    parent_id: array
    depth: array
    pv: tuple[str, ...]
    copy: array
    logical_volume_id: array
    selected: bytes
    candidate_root_id: array
    local_transform: array

    def __len__(self):
        return len(self.parent_id)


@dataclass(frozen=True)
class AuditReport:
    counts: dict
    sha256: str
    header: dict
    materials: tuple[dict, ...]
    solids: tuple[dict, ...]
    logical_volumes: tuple[dict, ...]
    placements: PlacementColumns
    root_placements: tuple[int, ...]
    external_completion_verified: bool
    accepted: bool
    protocol_complete: bool = True
    diagnostic_only: bool = True
    certificate: bool = False
    library_eligible: bool = False
    native_validation: bool = False
    root_store_verified: bool = False

    def path(self, placement_id: int) -> tuple[tuple[str, int], ...]:
        """Reconstruct an unambiguous world-to-leaf (byte-name, copy) path."""
        _integer(placement_id, "placement_id", 0, len(self.placements) - 1)
        result = []
        while placement_id != -1:
            result.append((self.placements.pv[placement_id], self.placements.copy[placement_id]))
            placement_id = self.placements.parent_id[placement_id]
        return tuple(reversed(result))


def _begin(row):
    _keys(row, _BEGIN, "inventory_begin")
    for key, expected in (("schema", SCHEMA), ("scope", SCOPE), ("units", UNITS), ("frame", FRAME),
                          ("name_encoding", NAME_ENCODING), ("identity", IDENTITY)):
        _need(row[key] == expected, f"invalid v2 {key}")
    _need(_boolean(row["certificate"], "certificate") is False, "inventory is not a certificate")
    _integer(row["g4_run"], "g4_run", *_I32)
    for key, expected in (("max_depth", 128), ("max_placements", 1_000_000),
                          ("stream_info_max_bytes", 65536), ("polycone_max_planes", 16384)):
        _need(_integer(row[key], key) == expected, f"invalid v2 {key}")
    _integer(row["max_bytes"], "max_bytes", 1, 1073741824)
    _integer(row["max_rows"], "max_rows", 3)
    _need(_integer(row["helper_allocator_max_bytes"], "helper_allocator_max_bytes") == min(row["max_bytes"], 268435456), "invalid allocator cap")
    roots = row["candidate_roots"]
    _need(type(roots) is list and roots, "nonempty candidate_roots required")
    for name in roots:
        _need(_text(name, "candidate root") != "", "empty candidate root")
    _need(len(roots) == len(set(roots)), "duplicate configured root")


def _solid(row, header):
    fields = set("type id name entity_type parameters parameter_status native_stream_info".split())
    status = row.get("parameter_status")
    supported = status == "native_parameters"
    _need(supported or status == "unsupported_metadata_only", "invalid solid parameter_status")
    _keys(row, fields if supported else fields | {"reason"}, "solid")
    _text(row["name"], "solid name")
    entity = _text(row["entity_type"], "entity_type")
    if not supported:
        _need(row["parameters"] is None and row["native_stream_info"] is None and row["reason"] == UNSUPPORTED_REASON, "unsupported solid must be metadata only")
        return False
    info = _text(row["native_stream_info"], "native_stream_info")
    _need(len(info) <= min(header["stream_info_max_bytes"], header["max_bytes"]), "StreamInfo byte cap")
    p = row["parameters"]
    if entity == "G4Box":
        _keys(p, {"half_x", "half_y", "half_z"}, "box parameters")
        for value in p.values():
            _real(value)
    elif entity == "G4Tubs":
        _keys(p, {"rmin", "rmax", "half_z", "start_phi", "delta_phi"}, "tubs parameters")
        for value in p.values():
            _real(value)
    elif entity == "G4Polycone":
        _keys(p, {"representation", "start_phi", "delta_phi", "planes"}, "polycone parameters")
        _need(p["representation"] == "native_original_z_planes", "invalid polycone representation")
        _real(p["start_phi"])
        _real(p["delta_phi"])
        _need(type(p["planes"]) is list and 2 <= len(p["planes"]) <= header["polycone_max_planes"], "polycone plane cap")
        for plane in p["planes"]:
            _vector(plane, 3)
    else:
        raise InventoryError("unallowlisted native solid")
    return True


def read(stream, *, limits: Limits | None = None, receipt: CompletionReceipt | None = None) -> AuditReport:
    """Audit a binary stream; malformed/over-budget input raises InventoryError.

    A structurally complete file without a matching successful external receipt
    returns accepted=False. Neither acceptance nor native parameters prove native
    authenticity, store uniqueness, shape validity, rotation conventions, or containment.
    """
    limits = Limits() if limits is None else limits
    _need(isinstance(limits, Limits), "Limits required")
    _need(receipt is None or isinstance(receipt, CompletionReceipt), "CompletionReceipt required")
    header = None
    ended = False
    phase = 0
    rows = total = candidates = unsupported = deepest = 0
    digest = hashlib.sha256()
    materials, solids, logicals, supported = [], [], [], bytearray()
    parents, depths, copies, lv_ids, selected, root_ids, transforms = array("i"), array("B"), array("i"), array("I"), bytearray(), array("i"), array("d")
    names, name_pool, active, siblings = [], {}, [], []
    root_lookup, root_occurrences = {}, []
    used_logicals = used_materials = used_solids = 0
    material_seen, solid_seen = bytearray(), bytearray()
    # Exact direct-child references only: no copied subtrees or hash collisions.
    first_children, next_children = array("i"), array("i")
    last_children, child_references, transform_keys = [], [], []

    def close_node(node):
        lv = lv_ids[node]
        child = first_children[node]
        reference = child_references[lv]
        if reference == -2:
            child_references[lv] = child
            return
        while child != -1 and reference != -1:
            _need((names[child], copies[child], lv_ids[child], transform_keys[child]) ==
                  (names[reference], copies[reference], lv_ids[reference], transform_keys[reference]),
                  "inconsistent logical-volume direct daughters")
            child, reference = next_children[child], next_children[reference]
        _need(child == reference, "inconsistent logical-volume direct daughter count")
    order = {"inventory_begin": 0, "material": 1, "solid": 2, "logical_volume": 3, "placement": 4, "inventory_end": 5}
    while True:
        byte_cap = min(limits.max_bytes, header["max_bytes"]) if header else limits.max_bytes
        allowance = min(limits.max_line_bytes, max(0, byte_cap - total))
        raw = stream.readline(allowance + 1)
        _need(type(raw) is bytes, "binary input stream required")
        if not raw:
            break
        _need(not ended, "data after inventory_end")
        _need(len(raw) <= limits.max_line_bytes, "line byte cap")
        total += len(raw)
        rows += 1
        _need(total <= byte_cap, "file byte cap")
        _need(rows <= min(limits.max_rows, header["max_rows"] if header else limits.max_rows), "row cap")
        _need(raw.endswith(b"\n"), "JSONL row requires newline")
        _need(all(c < 128 for c in raw), "native byte names require ASCII JSON U+00xx escapes")
        try:
            row = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float, parse_int=_json_int)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise InventoryError(f"row {rows}: invalid JSON: {exc}") from exc
        _need(type(row) is dict and type(row.get("type")) is str, "JSON object with type required")
        typ = row["type"]
        _need(typ in order and order[typ] >= phase, "invalid row sequence")
        phase = order[typ]
        digest.update(raw)
        if header is None:
            _need(typ == "inventory_begin", "inventory_begin required first")
            _begin(row)
            header = row
            _need(total <= header["max_bytes"], "declared file byte cap")
            root_lookup = {name: i for i, name in enumerate(row["candidate_roots"])}
            root_occurrences = [-1] * len(root_lookup)
            continue
        _need(typ != "inventory_begin", "duplicate inventory_begin")
        if typ in ("material", "solid", "logical_volume"):
            table = {"material": materials, "solid": solids, "logical_volume": logicals}[typ]
            _need(_integer(row.get("id"), "id") == len(table), "noncontiguous dictionary id")
            if typ == "material":
                _keys(row, {"type", "id", "name", "native_index"}, typ)
                _text(row["name"], "material name")
                _integer(row["native_index"], "native_index")
                material_seen.append(0)
            elif typ == "solid":
                supported.append(_solid(row, header))
                solid_seen.append(0)
            else:
                _keys(row, {"type", "id", "name", "material_id", "solid_id"}, typ)
                _text(row["name"], "logical volume name")
                _integer(row["material_id"], "material_id", 0, len(materials) - 1)
                _integer(row["solid_id"], "solid_id", 0, len(solids) - 1)
                child_references.append(-2)
            table.append(row)
        elif typ == "placement":
            _keys(row, _PLACEMENT, typ)
            i = len(parents)
            _need(i < min(limits.max_placements, header["max_placements"], header["max_rows"] - 2), "placement cap")
            _need(_integer(row["id"], "id") == i, "noncontiguous placement id")
            d = _integer(row["depth"], "depth", 0, header["max_depth"])
            parent = row["parent_id"]
            if i == 0:
                _need(parent is None and d == 0, "world must have null parent and depth zero")
            else:
                _integer(parent, "parent_id", 0, i - 1)
                _need(d == depths[parent] + 1 and d <= len(active), "parent/depth mismatch")
                _need(active[d - 1] == parent, "not expanded preorder")
            for node in reversed(active[d:]):
                close_node(node)
            del active[d:]
            del siblings[d:]
            del last_children[d:]
            name = _text(row["pv"], "pv")
            copy = _integer(row["copy"], "copy", *_I32)
            if i:
                key = (name, copy)
                _need(key not in siblings[d - 1], "ambiguous sibling name/copy path")
                siblings[d - 1].add(key)
            active.append(i)
            siblings.append(set())
            lv = _integer(row["logical_volume_id"], "logical_volume_id", 0, len(logicals) - 1)
            _need(lv <= used_logicals, "logical-volume first-encounter id order")
            if lv == used_logicals:
                used_logicals += 1
                material, solid = logicals[lv]["material_id"], logicals[lv]["solid_id"]
                if not material_seen[material]:
                    _need(material == used_materials, "material first-encounter id order")
                    material_seen[material] = 1
                    used_materials += 1
                if not solid_seen[solid]:
                    _need(solid == used_solids, "solid first-encounter id order")
                    solid_seen[solid] = 1
                    used_solids += 1
            root = root_lookup.get(name, -1)
            expected_root = None if root == -1 else root
            if row["candidate_root_id"] is not None:
                _integer(row["candidate_root_id"], "candidate_root_id", 0, len(root_lookup) - 1)
            _need(row["candidate_root_id"] == expected_root, "candidate_root_id/name mismatch")
            if root != -1:
                _need(root_occurrences[root] == -1, "configured root occurs on multiple paths")
                root_occurrences[root] = i
            chosen = _boolean(row["selected"], "selected")
            _need(chosen == (root != -1 or (parent is not None and bool(selected[parent]))), "invalid inherited selection")
            _vector(row["local_translation"], 3)
            rotation = row["local_object_rotation"]
            _need(type(rotation) is list and len(rotation) == 3, "invalid rotation matrix")
            for axis in rotation:
                _vector(axis, 3)
            first_children.append(-1)
            next_children.append(-1)
            if parent is not None:
                previous = last_children[d - 1]
                if previous == -1:
                    first_children[parent] = i
                else:
                    next_children[previous] = i
                last_children[d - 1] = i
            last_children.append(-1)
            # Compare parsed numbers before narrowing to the public double columns.
            transform_keys.append(tuple(row["local_translation"]) + tuple(x for axis in rotation for x in axis))
            parents.append(-1 if parent is None else parent)
            depths.append(d)
            names.append(name_pool.setdefault(name, name))
            copies.append(copy)
            lv_ids.append(lv)
            selected.append(chosen)
            root_ids.append(root)
            transforms.extend(row["local_translation"])
            for axis in rotation:
                transforms.extend(axis)
            candidates += chosen
            unsupported += not supported[logicals[lv]["solid_id"]]
            deepest = max(deepest, d)
        else:
            _keys(row, _END, "inventory_end")
            _need(len(parents) > 0, "missing world placement")
            for node in reversed(active):
                close_node(node)
            _need(used_logicals == len(logicals), "unreachable logical-volume dictionary entry")
            _need(used_materials == len(materials), "unreachable material dictionary entry")
            _need(used_solids == len(solids), "unreachable solid dictionary entry")
            _need(_boolean(row["complete_physical_tree"], "complete_physical_tree"), "incomplete physical tree")
            _need(_boolean(row["resolved_all_solids"], "resolved_all_solids") == (unsupported == 0), "resolved solid count mismatch")
            expected = dict(placements=len(parents), unique_paths=len(parents), materials=len(materials), solids=len(solids),
                            logical_volumes=len(logicals), candidate_placements=candidates,
                            unsupported_solid_placements=unsupported, deepest=deepest, rows_including_footer=rows)
            for key, value in expected.items():
                _need(_integer(row[key], key) == value, f"footer {key} mismatch")
            size = row["bytes_including_footer_decimal"]
            _need(type(size) is str and re.fullmatch(r"[0-9]{20}", size) is not None and int(size) == total, "footer byte total mismatch (fixed 20 digits required)")
            roots = row["root_placements"]
            _need(type(roots) is list and len(roots) == len(root_lookup), "footer root count mismatch")
            for root_id, item in enumerate(roots):
                _keys(item, {"root_id", "name", "count", "placement_id"}, "root reference")
                _need(_integer(item["root_id"], "root_id") == root_id, "footer root id mismatch")
                _need(_text(item["name"], "root name") == header["candidate_roots"][root_id], "footer root name mismatch")
                _need(_integer(item["count"], "root count") == 1 and root_occurrences[root_id] != -1, "missing/duplicate root")
                _need(_integer(item["placement_id"], "root placement_id") == root_occurrences[root_id], "footer root placement mismatch")
            ended = True
    _need(header is not None and ended, "missing completion footer")
    sha = digest.hexdigest()
    verified = receipt is not None and receipt.sha256 == sha and receipt.writer_closed and receipt.job_success
    counts = dict(placements=len(parents), unique_paths=len(parents), materials=len(materials), solids=len(solids),
                  logical_volumes=len(logicals), candidate_placements=candidates, unsupported_solid_placements=unsupported,
                  deepest=deepest, rows=rows, bytes=total)
    columns = PlacementColumns(parents, depths, tuple(names), copies, lv_ids, bytes(selected), root_ids, transforms)
    return AuditReport(counts, sha, header, tuple(materials), tuple(solids), tuple(logicals), columns,
                       tuple(root_occurrences), bool(verified), bool(verified))


def audit_file(path: str | os.PathLike, *, limits: Limits | None = None,
               receipt: CompletionReceipt | None = None) -> AuditReport:
    """Open a local file in binary mode and audit it without site-specific paths."""
    with open(path, "rb") as stream:
        return read(stream, limits=limits, receipt=receipt)


audit = read
