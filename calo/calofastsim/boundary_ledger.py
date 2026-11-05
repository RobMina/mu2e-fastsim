"""Bounded audit of native boundary JSONL; no ancestry inference or training output.

See recording/LEDGER_CONTRACT.md. Input must be a binary stream (or a path).
Protocol errors raise LedgerError with a compact, reconciled prefix ``report``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
import re

import numpy as np

SCHEMA = "mu2e-boundary-diagnostic-v1"
TRACK_STATUS_SCHEMA = "geant4-11.3.2-p110-G4TrackStatus-v1"
# Fixed installed p110 enum: Alive, StopButAlive, StopAndKill,
# KillTrackAndSecondaries, Suspend, SuspendAndWait, PostponeToNextEvent.
FRAME = "G4_world"
UNITS = "mm_MeV_ns"
ENCODING = "art_ProductID_value_uint32"
MANIFEST_FIELDS = ("source_sha256", "run", "subrun", "event", "root_product_id", "root_key")
DEFAULT_MAX_ROWS = 2_000_000
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_LINE_BYTES = 1024 * 1024
_U32, _U64 = 2**32 - 1, 2**64 - 1
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMON = {"schema", "type", "run", "subrun", "event", "sequence"}
_TRACK = {"track", "parent_track", "particle", "sim_saved_at_start", "step", "visit", "track_status"}
_BEGIN = {"g4_event", "sim_stage", "candidate_only", "coordinate_frame", "units", "product_id_encoding", "roots"}
_START = {"membership", "birth_phase", "pdg", "creator_process", "persistent_parent", "gen_particle", "state"}
_END = {"g4_event", "g4_aborted", "stock_filter_pass", "status_g4_equivalent", "n_g4_tracks", "overflow_sim_particles", "killed_step_limit", "killed_field", "unclosed_visits", "unknown_birth_memberships", "missing_saved_particles", "unknown_tracking_ends", "rows_before_end"}
_TRANSITION = {"pre_membership", "post_membership", "classification", "endpoint", "hook_phase", "step_length", "process", "pre", "post"}
_STOP = {"terminal", "membership", "process", "endpoint_phase", "state"}


class LedgerError(ValueError):
    """Rejected input; report retains observed events, never a successful-zero claim."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class AuditReport:
    """Compact per-event audit columns plus independent reconciliation indices."""

    events: dict
    counts: dict
    missing_manifest_indices: np.ndarray
    foreign_event_indices: np.ndarray
    manifest_indices: np.ndarray
    reconciled: bool
    protocol_complete: bool
    ledger_sha256: str
    source_sha256: str
    independent_manifest_sha256: str
    manifest_checksum_verified: bool = False
    source_checksum_verified: bool = False
    diagnostic_only: bool = True
    native_execution_certified: bool = False


def _need(ok, message):
    if not ok:
        raise LedgerError(message)


def _int(x, name, low=0, high=_U64):
    _need(type(x) is int and low <= x <= high, f"invalid integer {name}")
    return x


def _bool(x, name):
    _need(type(x) is bool, f"invalid boolean {name}")


def _text(x, name, nullable=False):
    _need((nullable and x is None) or isinstance(x, str), f"invalid string {name}")


def _real(x, name, nonnegative=False):
    _need(type(x) in (int, float) and math.isfinite(x) and (not nonnegative or x >= 0), f"invalid real {name}")


def _keys(obj, fields, name):
    _need(isinstance(obj, dict) and obj.keys() == fields, f"unexpected fields in {name}")


def _identity(x):
    if x is not None:
        _keys(x, {"product", "key"}, "persistent identity")
        _int(x["product"], "product", 1, _U32)
        _int(x["key"], "particle key")
    return None if x is None else (x["product"], x["key"])


def _state(s, roots, kind=None):
    common = {"position", "momentum", "kinetic_energy", "total_energy", "global_time", "proper_time", "local_time", "material", "touchable"}
    step = isinstance(s, dict) and "step_status" in s
    _keys(s, common | ({"step_status"} if step else {"material_source"}), "state")
    _need(kind is None or step == (kind == "step"), "wrong state kind")
    if step:
        _int(s["step_status"], "step_status", 0, 7)
    else:
        _need(s["material_source"] == "touchable_logical_not_step_material", "invalid material source")
    for k in ("position", "momentum"):
        _need(isinstance(s[k], list) and len(s[k]) == 3, "invalid state vector")
        for v in s[k]:
            _real(v, k)
    for k in ("kinetic_energy", "total_energy", "global_time", "proper_time", "local_time"):
        _real(s[k], k, k in ("kinetic_energy", "total_energy", "proper_time", "local_time"))
    m = s["material"]
    if m is not None:
        _keys(m, {"name", "index"}, "material")
        _text(m["name"], "material name")
        _int(m["index"], "material index")
    p = s["touchable"]
    _keys(p, {"availability", "nodes"}, "touchable")
    _need(p["availability"] in ("available", "unavailable", "world_null") and isinstance(p["nodes"], list), "invalid touchable")
    _need(bool(p["nodes"]) == (p["availability"] == "available"), "invalid touchable nodes")
    inside = False
    for node in p["nodes"]:
        _keys(node, {"pv", "lv", "copy", "replica"}, "touchable node")
        _text(node["pv"], "pv")
        _text(node["lv"], "lv")
        for k in ("copy", "replica"):
            _int(node[k], k, -(2**31), 2**31 - 1)
        inside |= node["pv"] in roots
    if p["availability"] == "world_null":
        _need(step and s["step_status"] == 0, "invalid world-null state")
        return 0
    return int(inside) if p["availability"] == "available" else -1


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(x):
    raise LedgerError(f"nonfinite JSON constant: {x}")


def _float(x):
    value = float(x)
    _need(math.isfinite(value), "nonfinite JSON number")
    return value


def _json_int(x):
    # No unbounded big-integer construction, even for an unused field.
    _need(len(x.lstrip("-")) <= 20, "JSON integer exceeds 64-bit token bound")
    return int(x)


def _manifest(table, cap, sha):
    _need(isinstance(table, Mapping) and set(table) == set(MANIFEST_FIELDS), "independent columnar manifest required")
    # Check lengths before coercion, structured allocation or sorting.
    try:
        lengths = [len(table[k]) for k in MANIFEST_FIELDS]
    except TypeError as exc:
        raise LedgerError("manifest columns must be sized") from exc
    _need(len(set(lengths)) == 1 and lengths[0] <= cap, "manifest length/row bound")
    columns = {}
    for k in MANIFEST_FIELDS:
        raw = table[k]
        # Reject nested/object inputs before coercion can allocate an unbounded
        # rectangular array or a huge inferred Unicode dtype.
        if isinstance(raw, np.ndarray):
            _need(raw.ndim == 1, "manifest columns must be one dimensional")
            a = raw
        else:
            _need(isinstance(raw, (list, tuple)), "manifest columns must be arrays/lists/tuples")
            if k == "source_sha256":
                _need(all(type(v) is str and v == sha for v in raw), "manifest source SHA does not match provenance binding")
                a = np.full(len(raw), sha, dtype="U64")
            else:
                high = _U32 if k in ("run", "subrun", "root_product_id") else _U64
                _need(all(type(v) is int and 0 <= v <= high for v in raw), f"invalid manifest integer {k}")
                a = np.asarray(raw, dtype=np.uint64)
        if k == "source_sha256":
            _need(a.dtype.kind == "U" or len(a) == 0, "manifest SHA must be Unicode strings")
            _need(np.all(a == sha), "manifest source SHA does not match provenance binding")
            columns[k] = np.full(len(a), sha, dtype="U64")
        else:
            _need(a.dtype.kind in "iu" or len(a) == 0, f"manifest {k} must retain integer storage")
            _need(np.all(a >= (1 if k == "root_product_id" else 0)), "negative/zero manifest identity")
            high = _U32 if k in ("run", "subrun", "root_product_id") else _U64
            _need(np.all(a <= high), "manifest identity out of range")
            columns[k] = a.astype(np.uint64)
    keys = _event_keys(columns)
    _need(len(np.unique(keys)) == len(keys), "duplicate independent manifest event")
    return columns, keys


def _event_keys(columns):
    n = len(columns["event"])
    keys = np.empty(n, dtype=[("run", "u8"), ("subrun", "u8"), ("event", "u8")])
    for k in keys.dtype.names:
        keys[k] = columns[k]
    return keys


class _Machine:
    def __init__(self):
        self.events = []
        self.current = None
        self.seen = set()
        self.tracks = {}
        self.rows = 0
        self.bytes = 0
        self.blocked = False

    def consume(self, r):
        _need(isinstance(r, dict), "row must be an object")
        _need(_COMMON <= r.keys() and r["schema"] == SCHEMA, "invalid row header/schema")
        for k in ("run", "subrun", "event", "sequence"):
            _int(r[k], k, high=_U32 if k in ("run", "subrun") else _U64)
        typ = r["type"]
        _text(typ, "row type")
        key = (r["run"], r["subrun"], r["event"])
        _need(not self.blocked, "row after fatal membership_unavailable")
        if typ == "event_begin":
            _keys(r, _COMMON | _BEGIN, typ)
            _need(self.current is None and key not in self.seen and r["sequence"] == 0, "duplicate/nested/out-of-sequence event_begin")
            _int(r["g4_event"], "g4_event", -(2**31), 2**31 - 1)
            _int(r["sim_stage"], "sim_stage")
            _need(r["candidate_only"] is True and r["coordinate_frame"] == FRAME and r["units"] == UNITS and r["product_id_encoding"] == ENCODING, "unsupported event conventions")
            roots = r["roots"]
            _need(isinstance(roots, list) and roots and all(isinstance(x, str) and x for x in roots), "invalid candidate roots")
            _need(len(set(roots)) == len(roots), "duplicate roots")
            e = dict(run=key[0], subrun=key[1], event=key[2], ended=False, transport_failed=False,
                     stock_filter_pass=False, entries=0, starts=0, born_inside=0,
                     domain_occupancy_observed=False, unknown_births=0, missing_saved=0,
                     unknown_ends=0, unclosed_visits=0, open_tracking=0, pending_tracks=0, postponed=False,
                     unsupported=r["sim_stage"] != 0, status=-1, next_sequence=1,
                     g4_event=r["g4_event"], roots=set(roots))
            self.events.append(e)
            self.current = e
            self.tracks = {}
            self.seen.add(key)
            return
        e = self.current
        _need(e is not None and key == (e["run"], e["subrun"], e["event"]), "row outside its event")
        _need(r["sequence"] == e["next_sequence"], "non-contiguous sequence")
        e["next_sequence"] += 1
        if typ == "event_end":
            self.end(r)
            return
        extra = {"track_start": _START, "born_inside": _START, "tracking_resume": {"state"},
                 "birth_membership_unresolved": {"first_observed_pre"},
                 "membership_unavailable": {"pre", "post"}, "entry": _TRANSITION,
                 "exit": _TRANSITION, "tracking_end": _STOP, "terminal_inside": _STOP}.get(typ)
        _need(extra is not None, "unknown row type")
        _keys(r, _COMMON | _TRACK | extra, typ)
        for k in ("track", "parent_track", "step", "visit", "track_status"):
            _int(r[k], k, 1 if k == "track" else 0, 6 if k == "track_status" else (2**31 - 1 if k in ("track", "parent_track", "step") else _U64))
        _bool(r["sim_saved_at_start"], "sim_saved_at_start")
        particle = _identity(r["particle"])
        e["postponed"] |= r["track_status"] == 6
        e["unsupported"] |= r["track_status"] == 1  # at-rest lifecycle not supported
        if typ in ("track_start", "born_inside"):
            _need(r["track"] not in self.tracks, "duplicate track start")
            _need(r["step"] == 0, "initial birth must be step zero")
            _need(r["track_status"] in (0, 1), "invalid initial birth status")
            _int(r["membership"], "membership", -1, 1)
            _int(r["pdg"], "pdg", -(2**31), 2**31 - 1)
            _need(r["birth_phase"] == "PreUserTracking_after_saveSimParticleStart", "wrong birth phase")
            _text(r["creator_process"], "creator", True)
            _identity(r["persistent_parent"])
            _identity(r["gen_particle"])
            m = _state(r["state"], e["roots"], "track")
            _need(m == r["membership"] and (typ == "born_inside") == (m == 1), "birth membership/type mismatch")
            _need(r["visit"] == int(m == 1), "invalid initial visit")
            t = dict(parent=r["parent_track"], particle=particle, saved=r["sim_saved_at_start"],
                     visit=r["visit"], active=m == 1, unknown=m == -1, running=True,
                     terminal=False, step=r["step"], transition_step=-1)
            self.tracks[r["track"]] = t
            e["starts"] += 1
            e["born_inside"] += m == 1
            e["unsupported"] |= m == 1  # current kernel is exterior-only
            e["domain_occupancy_observed"] |= m == 1
            e["open_tracking"] += 1
            e["pending_tracks"] += 1
            e["unknown_births"] += m == -1
            e["missing_saved"] += not t["saved"]
            e["unclosed_visits"] += t["active"]
            return
        _need(r["track"] in self.tracks, "track without start")
        t = self.tracks[r["track"]]
        _need((r["parent_track"], particle, r["sim_saved_at_start"]) == (t["parent"], t["particle"], t["saved"]), "track identity changed")
        _need(r["step"] >= t["step"], "track step went backwards")
        previous_step = t["step"]
        t["step"] = r["step"]
        if typ == "tracking_resume":
            _need(not t["running"] and not t["terminal"], "resume without nonterminal end")
            _need(t["end_status"] in (4, 5) and r["track_status"] == 0, "unsupported resume status")
            _need(r["step"] == previous_step, "resume changed step")
            m = _state(r["state"], e["roots"], "track")
            _need(m == t["end_membership"] and (t["unknown"] or m == -1 or m == int(t["active"])), "resume membership discontinuity")
            e["domain_occupancy_observed"] |= m == 1
            t["running"] = True
            e["open_tracking"] += 1
        else:
            _need(t["running"], "track row after end without resume")
            if typ == "birth_membership_unresolved":
                _need(t["unknown"], "unexpected unresolved-birth row")
                _need(r["step"] == 1, "unknown birth first observation must be step one")
                m = _state(r["first_observed_pre"], e["roots"], "step")
                e["domain_occupancy_observed"] |= m == 1
                t["unknown"] = False
                if m == 1:
                    t["visit"] += 1
                    t["active"] = True
                    e["unclosed_visits"] += 1
            elif typ in ("entry", "exit", "membership_unavailable"):
                _need(r["step"] > 0, "transition must follow birth step")
                _need(not t["unknown"], "missing unresolved-birth row")
                a = _state(r["pre"], e["roots"], "step")
                b = _state(r["post"], e["roots"], "step")
                e["domain_occupancy_observed"] |= a == 1 or b == 1
                if typ == "membership_unavailable":
                    _need(a == -1 or b == -1, "spurious unavailable membership")
                    self.blocked = True
                    e["unsupported"] = True
                else:
                    _need(a >= 0 and b >= 0 and a != b and a == int(t["active"]), "impossible visit transition")
                    _need(r["pre_membership"] == a and type(r["pre_membership"]) is int and r["post_membership"] == b and type(r["post_membership"]) is int, "transition membership mismatch")
                    _need((typ == "entry") == (b == 1) and r["step"] > t["transition_step"], "transition type/order mismatch")
                    t["transition_step"] = r["step"]
                    if b == 1:
                        t["visit"] += 1
                        e["entries"] += 1
                    e["unclosed_visits"] += b - int(t["active"])
                    t["active"] = b == 1
                    expected = "exit" if b == 0 else ("first_observed_entry" if t["visit"] == 1 else "reentry")
                    _need(r["classification"] == expected and r["endpoint"] == "post_step" and r["hook_phase"] == "UserSteppingAction_before_stock_cuts", "invalid transition annotation")
                    _real(r["step_length"], "step_length", True)
                    _text(r["process"], "process", True)
            elif typ in ("tracking_end", "terminal_inside"):
                _bool(r["terminal"], "terminal")
                _need(r["track_status"] in (1, 2, 3, 4, 5, 6), "invalid tracking end status")
                terminal = r["track_status"] in (2, 3)
                _need(r["terminal"] == terminal, "terminal/status mismatch")
                _int(r["membership"], "membership", -1, 1)
                _need(r["endpoint_phase"] in ("last_post_step", "track_no_step"), "invalid end phase")
                _need((r["endpoint_phase"] == "track_no_step") == (r["step"] == 0), "end phase/step mismatch")
                m = _state(r["state"], e["roots"], "track" if r["step"] == 0 else "step")
                _need(m == r["membership"] and (typ == "terminal_inside") == (terminal and m == 1), "end membership/type mismatch")
                _need(t["unknown"] or m == -1 or m == int(t["active"]), "end without matching transition")
                _text(r["process"], "process", True)
                e["domain_occupancy_observed"] |= m == 1
                t["end_membership"] = m
                t["end_status"] = r["track_status"]
                e["unknown_ends"] += m == -1
                if terminal:
                    e["unclosed_visits"] -= int(t["active"])
                    t["active"] = False
                    e["pending_tracks"] -= 1
                t["running"] = False
                t["terminal"] = terminal
                e["open_tracking"] -= 1
        _need(r["visit"] == t["visit"], "visit counter mismatch")

    def end(self, r):
        _keys(r, _COMMON | _END, "event_end")
        e = self.current
        _int(r["g4_event"], "g4_event", -(2**31), 2**31 - 1)
        _need(r["g4_event"] == e["g4_event"] and r["rows_before_end"] == r["sequence"], "event_end identity/count mismatch")
        for k in _END - {"g4_event", "g4_aborted", "stock_filter_pass", "overflow_sim_particles"}:
            _int(r[k], k)
        for k in ("g4_aborted", "stock_filter_pass", "overflow_sim_particles"):
            _bool(r[k], k)
        expected_status = 10 if r["overflow_sim_particles"] else int(bool(r["killed_step_limit"] or r["killed_field"]))
        _need(r["status_g4_equivalent"] == expected_status, "inconsistent transport status")
        for native, local in (("unclosed_visits", "unclosed_visits"), ("unknown_birth_memberships", "unknown_births"), ("missing_saved_particles", "missing_saved"), ("unknown_tracking_ends", "unknown_ends")):
            _need(r[native] == e[local], f"inconsistent {native}")
        _need(r["n_g4_tracks"] >= e["starts"], "n_g4_tracks below observed starts")
        e.update(ended=True, status=expected_status, transport_failed=expected_status != 0 or r["g4_aborted"], stock_filter_pass=r["stock_filter_pass"])
        self.current = None
        self.tracks = {}

    def report(self, manifest, manifest_keys, sha, manifest_sha, digest, rejected=False):
        # Only column extraction is Python-level; all event analysis/reconciliation is vectorized.
        names = ("run", "subrun", "event", "ended", "transport_failed", "stock_filter_pass", "entries", "starts", "born_inside", "domain_occupancy_observed", "unknown_births", "missing_saved", "unknown_ends", "unclosed_visits", "open_tracking", "pending_tracks", "postponed", "unsupported", "status")
        flags = {"domain_occupancy_observed", "ended", "transport_failed", "stock_filter_pass", "postponed", "unsupported"}
        columns = {k: np.fromiter((e[k] for e in self.events), dtype=bool if k in flags else (np.int64 if k == "status" else np.uint64), count=len(self.events)) for k in names}
        keys = _event_keys(columns)
        order = np.argsort(manifest_keys)
        pos = np.searchsorted(manifest_keys[order], keys)
        matched = np.zeros(len(keys), dtype=bool)
        candidate = pos < len(order)
        matched[candidate] = manifest_keys[order[pos[candidate]]] == keys[candidate]
        indices = np.full(len(keys), -1, dtype=np.int64)
        indices[matched] = order[pos[matched]]
        seen = np.zeros(len(order), dtype=bool)
        seen[indices[matched]] = True
        missing = np.flatnonzero(~seen)
        foreign = np.flatnonzero(~matched)
        columns["manifest_matched"] = matched.copy()
        columns["root_product_id"] = np.zeros(len(keys), dtype=np.uint64)
        columns["root_key"] = np.zeros(len(keys), dtype=np.uint64)
        for k in ("root_product_id", "root_key"):
            columns[k][matched] = manifest[k][indices[matched]]
        columns["unsupported"] |= columns["postponed"] | (columns["unknown_births"] > 0) | (columns["missing_saved"] > 0) | (columns["unknown_ends"] > 0)
        columns["incomplete"] = (~columns["ended"] | (columns["unclosed_visits"] > 0) | (columns["open_tracking"] > 0) | (columns["pending_tracks"] > 0) | columns["unsupported"])
        columns["no_entry_observed"] = columns["entries"] == 0
        columns["no_domain_occupancy_observed"] = ~columns["domain_occupancy_observed"]
        columns["library_eligible"] = np.zeros(len(keys), dtype=bool)
        counts = {k: int(np.count_nonzero(columns[k])) for k in ("ended", "transport_failed", "incomplete", "unsupported", "postponed", "no_entry_observed", "no_domain_occupancy_observed")}
        counts["born_inside"] = int(np.sum(columns["born_inside"]))
        counts.update(events=len(keys), manifest_events=len(order), missing_events=len(missing), foreign_events=len(foreign), rows=self.rows, bytes=self.bytes)
        return AuditReport(columns, counts, missing, foreign, indices, not (len(missing) or len(foreign)), not rejected and self.current is None and not self.blocked, digest, sha, manifest_sha)


def audit(source, *, independent_manifest, provenance, max_rows=DEFAULT_MAX_ROWS,
          max_bytes=DEFAULT_MAX_BYTES, max_line_bytes=DEFAULT_MAX_LINE_BYTES,
          max_manifest_rows=DEFAULT_MAX_ROWS):
    """Audit a binary JSONL stream/path against a separately supplied columnar manifest.

    EOF within an event returns an incomplete report. Invalid JSON/protocol or a
    resource cap raises LedgerError carrying that same compact prefix report.
    Missing/foreign events make ``reconciled`` false, not an exception; callers
    must inspect both reconciled and protocol_complete. No eligibility is granted.
    """
    for name, limit in (("max_rows", max_rows), ("max_bytes", max_bytes), ("max_line_bytes", max_line_bytes), ("max_manifest_rows", max_manifest_rows)):
        _need(type(limit) is int and limit > 0, f"positive integer {name} required")
    _need(isinstance(provenance, Mapping) and provenance.get("purpose") == "diagnostic_boundary_ledger", "diagnostic ledger provenance required")
    for k in ("source_sha256", "independent_manifest_sha256"):
        _need(isinstance(provenance.get(k), str) and _HASH.fullmatch(provenance[k]) is not None, f"invalid provenance {k}")
    _need(provenance.get("track_status_schema") == TRACK_STATUS_SCHEMA, "explicit p110 track_status_schema required")
    sha, manifest_sha = provenance["source_sha256"], provenance["independent_manifest_sha256"]
    manifest, manifest_keys = _manifest(independent_manifest, max_manifest_rows, sha)
    machine = _Machine()
    digest = hashlib.sha256()
    owned = isinstance(source, (str, bytes, os.PathLike))
    stream = open(source, "rb") if owned else source
    try:
        while True:
            # One-byte lookahead only; never call unbounded readline/read or decode
            # a line exceeding either remaining byte budget or line cap.
            allowance = min(max_line_bytes, max_bytes - machine.bytes)
            line = stream.readline(1 if machine.rows >= max_rows else allowance + 1)
            _need(isinstance(line, bytes), "binary input stream required")
            if not line:
                break
            _need(machine.rows < max_rows, "ledger row bound exceeded")
            _need(len(line) <= allowance, "ledger byte/line bound exceeded")
            machine.bytes += len(line)
            digest.update(line)
            _need(line.endswith(b"\n"), "unterminated/truncated JSONL row")
            machine.rows += 1
            row = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float, parse_int=_json_int)
            machine.consume(row)
    except (LedgerError, ValueError, UnicodeError, RecursionError, OverflowError, OSError) as exc:
        report = machine.report(manifest, manifest_keys, sha, manifest_sha, digest.hexdigest(), rejected=True)
        if isinstance(exc, LedgerError):
            exc.report = report
            raise
        raise LedgerError(f"invalid/incomplete boundary ledger: {exc}", report) from exc
    finally:
        if owned:
            stream.close()
    return machine.report(manifest, manifest_keys, sha, manifest_sha, digest.hexdigest())


def audit_file(source, *, independent_manifest_path, source_artifact, provenance,
               max_manifest_bytes=DEFAULT_MAX_BYTES, max_source_bytes=DEFAULT_MAX_BYTES,
               **kwargs):
    """Strict artifact path: verify canonical manifest bytes and actual source SHA.

    Canonical manifest is ASCII JSON of exactly MANIFEST_FIELDS, all columns lists,
    sort_keys=True, separators=(",", ":"), ensure_ascii=True, no final newline.
    Source mapping is checked by _manifest against the verified source SHA. This
    does not establish manifest independence or native ledger/source association.
    """
    for name, limit in (("max_manifest_bytes", max_manifest_bytes),
                        ("max_source_bytes", max_source_bytes)):
        _need(type(limit) is int and limit > 0, f"positive integer {name} required")
    _need(isinstance(provenance, Mapping), "provenance mapping required")
    try:
        with open(independent_manifest_path, "rb") as stream:
            data = stream.read(max_manifest_bytes + 1)
        _need(len(data) <= max_manifest_bytes, "manifest byte bound exceeded")
        _need(hashlib.sha256(data).hexdigest() == provenance.get("independent_manifest_sha256"),
              "manifest artifact checksum mismatch")
        table = json.loads(data.decode("ascii"), object_pairs_hook=_pairs,
                           parse_constant=_constant, parse_float=_float, parse_int=_json_int)
        _need(isinstance(table, dict) and set(table) == set(MANIFEST_FIELDS)
              and all(type(v) is list for v in table.values()), "canonical manifest columns required")
        _need(json.dumps(table, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("ascii") == data, "noncanonical manifest artifact")
        digest = hashlib.sha256()
        total = 0
        with open(source_artifact, "rb") as stream:
            while True:
                block = stream.read(min(1024 * 1024, max_source_bytes - total + 1))
                if not block:
                    break
                total += len(block)
                _need(total <= max_source_bytes, "source artifact byte bound exceeded")
                digest.update(block)
        _need(digest.hexdigest() == provenance.get("source_sha256"), "source artifact checksum mismatch")
    except (ValueError, UnicodeError, RecursionError, OverflowError, OSError) as exc:
        if isinstance(exc, LedgerError):
            raise
        raise LedgerError(f"invalid verified artifact: {exc}") from exc
    try:
        report = audit(source, independent_manifest=table, provenance=provenance, **kwargs)
    except LedgerError as exc:
        if exc.report is not None:
            exc.report = replace(exc.report, manifest_checksum_verified=True, source_checksum_verified=True)
        raise
    return replace(report, manifest_checksum_verified=True, source_checksum_verified=True)
