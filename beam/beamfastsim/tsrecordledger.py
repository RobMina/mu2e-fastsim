"""Bounded source-only structural audit; never admits a source/physics library.

The proposal writes geometry_file='records.jsonl', geometry_hash=null (not two
nulls). Cuts expose step ordinals, not external endpoints: external-step and
stock-product coverage remain unverified. No physics quantities are analyzed.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .tsrecordsource import SCHEMA_V2, SOURCE_HOOKS, SOURCE_KINDS, V2_ONLY_FIELDS, SourceAudit

SCHEMA = "target-recording-proposal-v1"
NATIVE_KINDS = set("event_begin volume inventory_end stacking track_begin target_step secondary_birth birth_track_association cuts track_end event_end completion".split())
KINDS = NATIVE_KINDS | SOURCE_KINDS
U64 = (1 << 64) - 1


class LedgerError(ValueError):
    """Incomplete, unsupported or inconsistent diagnostic stream."""


def _need(ok, why):
    if not ok:
        raise LedgerError(why)


def _uint(value):
    _need(type(value) is int and 0 <= value <= U64, "invalid U64 integer")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _finite(text):
    value = float(text)
    _need(math.isfinite(value), "nonfinite JSON number")
    return value


def _bad_constant(text):
    raise LedgerError("nonfinite JSON constant: " + text)


def _identifiers(value, birth_state=False):
    if type(value) is dict:
        for key, item in value.items():
            if birth_state and key in {"track_id", "parent_id"}:
                _need(type(item) is int and -(1 << 31) <= item < (1 << 31), "invalid raw birth integer")
                continue
            if key in {"instance_id", "track_id", "parent_id", "child_token", "run", "subrun", "event", "row", "step_number", "current_step_number", "lifecycle_visit", "product_id", "key", "sim_stage", "daughter_index"} and item is not None:
                _uint(item)
            if key in {"copy_no", "pdg", "native_status", "step_status_native", "track_status_native"}:
                _need(type(item) is int, "invalid signed native integer")
            _identifiers(item, birth_state=key == "native_child_at_birth")
    elif type(value) is list:
        for item in value:
            _identifiers(item)


def _rows(path, limits, digest, sizes):
    _need(path.is_file() and not path.is_symlink(), "missing or linked ledger file")
    _need(path.stat().st_size <= limits[1], "file byte limit")
    with path.open("rb") as stream:
        size = 0
        while True:
            line = stream.readline(limits[0] + 1)
            if not line:
                break
            size += len(line)
            _need(len(line) <= limits[0] and size <= limits[1], "byte limit")
            _need(line.endswith(b"\n"), "unterminated native row")
            digest.update(line)
            row = json.loads(line, object_pairs_hook=_pairs, parse_float=_finite, parse_constant=_bad_constant)
            _need(type(row) is dict, "row must be object")
            _identifiers(row)
            yield row
        sizes[path.name] = size


def _audit(directory, limits):
    counts = Counter()
    hooks = Counter(beginEvent=1, endEvent=1, stacking=0, beginTrack=0, endTrack=0, step=0, cuts=0)
    volumes, births, associations, tracks = {}, {}, {}, {}
    mapped_tokens = set()
    sizes, hashes = {}, {}
    phase, active, pending, selected, total = "start", None, None, 0, 0
    identity = caps = end = completion = awaiting = None
    schema = source = None

    def eqint(obj, key, value):
        _need(_uint(obj[key]) == value, "counter mismatch: " + key)

    def geometry(g, mode="track", status=None):
        vid = g["instance_id"]
        if vid is None:
            if mode == "track":
                _need(g["availability"] == "unavailable_native_track_touchable" and g["target_member"] is None
                      and g.get("copy_history_root_to_leaf") is None, "unavailable track geometry")
            else:
                _need(mode == "post" and status == 0 and g["availability"] == "explicit_null_world_post" and g["target_member"] is False and g["copy_history_root_to_leaf"] is None, "invalid world endpoint")
            return False
        _uint(vid)
        _need(vid in volumes and type(g["target_member"]) is bool, "unknown volume or nonboolean membership")
        v = volumes[vid]
        expected = "native_track_touchable_not_vertex_membership" if mode == "track" else "native_touchable"
        _need(g["availability"] == expected and g["target_member"] is v["target_member"] and g["copy_history_root_to_leaf"] == v["copy_history_root_to_leaf"], "geometry identity mismatch")
        _need(mode != "post" or status != 0, "world post must be null")
        return g["target_member"]

    def causal(c, tid):
        t = tracks[tid]
        eqint(c, "parent_id", t["parent"])
        token = associations.get(tid)
        _need(c["child_token"] is None if token is None else _uint(c["child_token"]) == token, "causal token mismatch")
        _need(c["birth_association"] == ("explicit_primary_root" if not t["parent"] else "observed_secondary_birth_incidence"), "unresolved causal ancestry")

    def parent_step(p):
        nonlocal pending
        tid, step = _uint(p["track_id"]), _uint(p["step_number"])
        _need(tid == active and step == tracks[tid]["steps"] + 1, "parent step sequence")
        eqint(p, "lifecycle_visit", 1)
        pre = geometry(p["pre"]["geometry"], "pre", p["pre"]["step_status_native"])
        post = geometry(p["post"]["geometry"], "post", p["post"]["step_status_native"])
        _need(type(p["target_step_row_present"]) is bool and p["target_step_row_present"] is (pre or post), "target subset mismatch")
        boundary = p["post"]["step_status_native"] in (0, 1) or p["pre"]["geometry"]["instance_id"] != p["post"]["geometry"]["instance_id"]
        _need(type(p["boundary_observed"]) is bool and p["boundary_observed"] is boundary, "boundary observation mismatch")
        _need(p["transition_observation"] == ("internal" if pre and post else "leaving" if pre else "entering" if post else "outside"), "transition mismatch")
        _need(pending is None or pending == p, "conflicting parent step")
        pending = p

    for name in ("records.jsonl", "completion.json"):
        digest, file_rows = hashlib.sha256(), 0
        for r in _rows(directory / name, limits, digest, sizes):
            total += 1
            file_rows += 1
            _need(total <= limits[2], "row limit")
            k, d = r["kind"], r["data"]
            _need(k in KINDS and type(d) is dict, "unknown kind or payload")
            eqint(r, "row", total)
            ids = tuple(_uint(r[key]) for key in ("run", "subrun", "event"))
            identity = ids if identity is None else identity
            if schema is None:
                schema = r["schema"]
                _need(schema in (SCHEMA, SCHEMA_V2), "unsupported schema")
                if schema == SCHEMA_V2:
                    source = SourceAudit(_need, _uint, geometry)
            _need(ids == identity and r["schema"] == schema and r["library_eligible"] is False and r["proposal_only"] is True, "envelope mismatch")
            _need(r["geometry_file"] == "records.jsonl" and r["geometry_hash"] is None and r["hash_binding"] == "requires_external_postread_file_hash", "geometry binding mismatch")
            _need((name == "completion.json") == (k == "completion"), "completion location")
            if source is None:
                _need(k in NATIVE_KINDS and not (set(d) & V2_ONLY_FIELDS), "v2 fields/rows in v1 stream")
                if k == "completion" and type(d.get("observed_hooks")) is dict:
                    _need(not (set(d["observed_hooks"]) & SOURCE_HOOKS), "v2 hooks in v1 stream")
            counts[k] += 1
            _need(awaiting is None or (k == "stacking" and d.get("track_id") == awaiting), "association not immediately stacked")
            consumed = source.observe(k, d, phase, active, tracks) if source is not None else False
            if source is not None and source.caps is not None:
                caps = source.caps
            if caps is not None:
                _need(total <= caps[0] and len(volumes) <= caps[1], "native caps exceeded")
            if consumed:
                continue
            if k == "event_begin":
                _need(phase == "start" and d["outcome"] == "begun_not_complete" and d["proposal_only"] is True, "event begin order")
                caps = tuple(_uint(d[key]) for key in ("max_rows", "max_volumes", "max_bytes"))
                _need(all(caps), "zero native cap")
                selectors, material = d["target_physical_names"], d["target_material"]
                _need(type(selectors) is list and selectors and all(type(x) is str and x for x in selectors) and len(set(selectors)) == len(selectors) and type(material) is str and material, "invalid target selectors")
                phase = "inventory"
            elif k == "volume":
                _need(phase == "inventory", "volume outside inventory")
                vid, parent = _uint(d["instance_id"]), d["parent_id"]
                _need(vid == len(volumes) + 1 and type(d["target_member"]) is bool, "volume IDs/membership")
                _need(parent is None if vid == 1 else _uint(parent) in volumes, "volume parent")
                h = d["copy_history_root_to_leaf"]
                prefix = [] if parent is None else volumes[parent]["copy_history_root_to_leaf"]
                _need(type(h) is list and len(h) == len(prefix) + 1 and h[:-1] == prefix, "volume history")
                leaf = h[-1]
                eqint(leaf, "instance_id", vid)
                _need(type(leaf["copy_no"]) is int and leaf["physical_name"] == d["physical_name"] and leaf["logical_name"] == d["logical_name"], "history leaf")
                _need(leaf["daughter_index"] is None if parent is None else type(leaf["daughter_index"]) is int and _uint(leaf["daughter_index"]) >= 0, "daughter index")
                _need(d["target_member"] is (d["physical_name"] in selectors) and (not d["target_member"] or d["material"] == material), "target selector mismatch")
                volumes[vid] = d
            elif k == "inventory_end":
                _need(phase == "inventory" and volumes, "inventory order")
                eqint(d, "volume_count", len(volumes))
                _need({v["physical_name"] for v in volumes.values() if v["target_member"]} == set(selectors), "missing target selector")
                phase = "callbacks"
            elif k == "completion":
                _need(phase == "ended" and file_rows == 1, "completion order/count")
                completion, phase = d, "complete"
            else:
                _need(phase == "callbacks", "callback order")
                if k == "birth_track_association":
                    token, tid = _uint(d["child_token"]), _uint(d["track_id"])
                    _need(active is None and tid > 0 and token in births and tid not in associations and token not in mapped_tokens, "duplicate/unknown birth association")
                    b = births[token]
                    _need(d["observed_at"] == "stacking" and _uint(d["parent_id"]) == b["parent_step"]["track_id"] and _uint(d["parent_step"]) == b["parent_step"]["step_number"] and (b["raw_child_track_id"] <= 0 or b["raw_child_track_id"] == tid), "birth mapping mismatch")
                    associations[tid], awaiting = token, tid
                    mapped_tokens.add(token)
                elif k == "stacking":
                    tid, parent = _uint(d["track_id"]), _uint(d["parent_id"])
                    _need(active is None and tid > 0 and tid not in tracks and type(d["pdg"]) is int, "duplicate stacking/identity")
                    token = associations.get(tid)
                    _need((parent == 0 and token is None) or (parent in tracks and token is not None and births[token]["parent_step"]["track_id"] == parent and births[token]["pdg"] == d["pdg"]), "secondary ancestry mismatch")
                    classification = d["classification_native"]
                    _need(type(classification) is int and classification in (0, -9), "unsupported stacking classification")
                    tracks[tid] = dict(parent=parent, pdg=d["pdg"], life="killed" if classification == -9 else "stacked", steps=0)
                    eqint(d, "observation_number", 1)
                    awaiting = None
                    hooks["stacking"] += 1
                    causal(d["causal"], tid)
                elif k in ("track_begin", "track_end"):
                    n = d["native"]
                    tid = _uint(n["track_id"])
                    _need(tid in tracks, "missing stacking")
                    t = tracks[tid]
                    eqint(n, "parent_id", t["parent"])
                    _need(type(n["pdg"]) is int and n["pdg"] == t["pdg"], "native identity mismatch")
                    eqint(d, "lifecycle_visit", 1)
                    causal(d["causal"], tid)
                    geometry(n["track_geometry"])
                    if k == "track_begin":
                        _need(active is None and t["life"] == "stacked", "invalid track begin")
                        eqint(n, "current_step_number", 0)
                        eqint(d, "stacking_observations", 1)
                        active, t["life"] = tid, "active"
                        hooks["beginTrack"] += 1
                    else:
                        _need(active == tid and pending is None and _uint(n["native_status"]) in (2, 3), "nonterminal or incomplete track end")
                        eqint(n, "current_step_number", t["steps"])
                        _need(n["native_status"] != 3 or not any(
                            b["parent_step"]["track_id"] == tid and token not in mapped_tokens
                            for token, b in births.items()), "unsupported killed-secondary disposition")
                        active, t["life"] = None, "ended"
                        hooks["endTrack"] += 1
                elif k in ("target_step", "secondary_birth"):
                    p = d["parent_step"]
                    parent_step(p)
                    if k == "target_step":
                        _need(p["target_step_row_present"] and not tracks[active].get("target_pending") and not tracks[active].get("birth_pending"), "duplicate/non-target step")
                        _need(_uint(d["track_status_native"]) not in (4, 5, 6), "unsupported step continuation")
                        selected += 1
                        eqint(d, "target_step_ordinal", selected)
                        tracks[active]["target_pending"] = True
                        causal(d["causal"], active)
                    else:
                        token = _uint(d["child_token"])
                        _need(token == len(births) + 1 and type(d["pdg"]) is int, "birth token sequence")
                        for key in ("raw_child_track_id", "raw_child_parent_id"):
                            _need(type(d[key]) is int and -(1 << 31) <= d[key] < (1 << 31), "invalid raw birth integer")
                        n = d["native_child_at_birth"]
                        _need(type(n["pdg"]) is int and n["pdg"] == d["pdg"], "birth native PDG")
                        _need(n["track_id"] == d["raw_child_track_id"] and n["parent_id"] == d["raw_child_parent_id"],
                              "raw/native birth identity mismatch")
                        geometry(n["track_geometry"])
                        causal(d["parent_causal"], active)
                        tracks[active]["birth_pending"] = True
                        births[token] = d
                elif k == "cuts":
                    tid = _uint(d["track_id"])
                    _need(tid == active, "cuts without active track")
                    t = tracks[tid]
                    eqint(d, "parent_id", t["parent"])
                    eqint(d, "step_number", t["steps"] + 1)
                    eqint(d, "lifecycle_visit", 1)
                    _need(pending is None or pending["target_step_row_present"] is bool(t.get("target_pending")), "missing target subset row")
                    _need(_uint(d["native_status"]) not in (4, 5, 6), "unsupported cut continuation")
                    t["steps"] += 1
                    t["target_pending"], t["birth_pending"], pending = False, False, None
                    causal(d["causal"], tid)
                    hooks["step"] += 1
                    hooks["cuts"] += 1
                elif k == "event_end":
                    _need(active is None and all(t["life"] in ("killed", "ended") for t in tracks.values()) and len(associations) == len(births) and set(associations) <= set(tracks), "incomplete event lifecycle")
                    end, phase = d, "ended"
                    killed = sum(t["life"] == "killed" for t in tracks.values())
                    eqint(d, "tracks_without_begin", killed)
                    eqint(d, "stacking_killed_native_enum", killed)
                    _need(d["zero_target_steps"] is (selected == 0) and d["zero_exit_or_exposure_claim"] is False, "zero-event claims")
                if k in ("stacking", "cuts"):
                    cut = d["stacking_cut" if k == "stacking" else "stepping_cut"]
                    _need(type(cut) is bool and d["common_cut_evaluated"] is (not cut) and (d["common_cut"] is None if cut else type(d["common_cut"]) is bool), "cut short circuit")
                    _need(k != "stacking" or not (cut or d["common_cut"]) or classification == -9, "stacking cut classification")
                    _need(k != "cuts" or not (cut or d["common_cut"]) or d["native_status"] == 2,
                          "stepping cut must observe stop-and-kill")
        hashes[name] = digest.hexdigest()
    _need(awaiting is None and phase == "complete" and completion["outcome"] == "endEvent_and_ledger_flush_complete" and completion["library_eligible"] is False and completion["completion_not_physics_or_exposure_certification"] is True, "missing/invalid completion")
    _need(completion["manifest_hash"] is None and completion["byte_cap_scope"] == "records.jsonl_plus_completion.json; filesystem_metadata_excluded", "completion binding/scope mismatch")
    for obj in (end, completion):
        for key, value in dict(secondary_births=len(births), associated_births=len(associations), unconsumed_birth_tokens=0, unresolved_nonprimary_tracks=0, stacked_not_begun_at_end=0, suspended_or_postponed_ended_visits=0).items():
            eqint(obj, key, value)
        _need(obj["event_aborted_native"] is False, "missing/aborted native event")
        _need(obj["sim_overflow"] is False, "simulation overflow")
        for key in ("status_native", "step_kills", "field_kills"):
            _need(type(obj[key]) is int, "invalid native counter/status")
            _need(obj[key] == end[key], "end/completion mismatch")
        _uint(obj["step_kills"])
        _uint(obj["field_kills"])
    for key in ("recording_row_overflow", "recording_byte_overflow", "recording_volume_overflow"):
        _need(completion[key] is False, "recording overflow")
    for key, value in dict(row_count=total, attempted_rows=total, volume_count=len(volumes), target_step_count=selected, ledger_bytes=sizes["records.jsonl"]).items():
        eqint(completion, key, value)
    if source is not None:
        hooks.update(source.hooks)
    for key, actual in (("per_kind_counts", counts), ("observed_hooks", hooks)):
        supplied = completion[key]
        _need(type(supplied) is dict and all(type(n) is int and _uint(n) >= 0 for n in supplied.values()), "invalid count map")
        allowed = (KINDS if source is not None else NATIVE_KINDS) if key == "per_kind_counts" else set(hooks)
        _need(set(supplied) <= allowed and Counter(supplied) == actual, "count map mismatch")
        if source is not None and key == "observed_hooks":
            _need(SOURCE_HOOKS <= set(supplied), "missing source hooks")
    _need(sum(completion["per_kind_counts"].values()) == total and total <= caps[0] and len(volumes) <= caps[1] and sum(sizes.values()) <= caps[2], "native caps/accounting mismatch")
    return dict(library_eligible=False, source_valid=False, native_valid=False, physics_certified=False,
                native_certified=False, source_native_valid=False, complete_source_only_diagnostic=True,
                external_step_coverage="unverified", stock_product_coverage="unverified",
                physics_payload_schema="unverified", geometry_certified=False, exposure_certified=False,
                file_sha256=hashes, file_bytes=sizes, row_count=total,
                per_kind_counts=dict(counts), observed_hooks=dict(hooks),
                **(source.result() if source is not None else dict(
                    source_reconciliation="unverified", persistent_output_verified=False)))


def audit(event_directory, *, max_line_bytes=1 << 20, max_file_bytes=64 << 20, max_rows=100_000):
    """Audit one event directory, bounded by caller limits; raise LedgerError.

    Only structural callback fields are checked; uninterpreted physics payload is
    not schema-qualified. Hashes bind both files separately, not source/runtime or
    geometry certification. A valid empty/zero-step event is not certified POT.
    """
    limits = (max_line_bytes, max_file_bytes, max_rows)
    _need(all(type(x) is int and 0 < x <= U64 for x in limits), "invalid audit limits")
    try:
        return _audit(Path(event_directory), limits)
    except (OSError, ValueError, KeyError, TypeError, RecursionError, OverflowError) as exc:
        if isinstance(exc, LedgerError):
            raise
        raise LedgerError("malformed/incomplete ledger: " + str(exc)) from exc
