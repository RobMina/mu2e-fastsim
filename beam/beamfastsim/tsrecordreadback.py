"""Bounded original-product-shaped readback join; never certifies persistence or POT."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .tsrecordledger import LedgerError, audit as audit_native

SCHEMA = "target-written-products-proposal-v1"
U64 = (1 << 64) - 1


class ReadbackError(ValueError):
    """Incomplete, inconsistent or unsupported readback-shaped evidence."""


def _need(ok, message):
    if not ok:
        raise ReadbackError(message)


def _fields(obj, names):
    _need(type(obj) is dict and set(obj) == set(names.split()), "readback payload fields")


def _uint(value, positive=False):
    _need(type(value) is int and int(positive) <= value <= U64, "invalid readback integer")
    return value


def _text(value, empty=False):
    _need(type(value) is str and (empty or bool(value)), "invalid readback string")
    return value


def _number(value):
    _need(type(value) in (int, float), "invalid readback number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    _need(finite, "nonfinite readback number")


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        _need(key not in obj, "duplicate JSON key")
        obj[key] = value
    return obj


def _constant(value):
    raise ReadbackError("nonfinite JSON constant: " + value)


def _rows(path, line_cap, file_cap, row_cap, digest):
    _need(path.is_file() and not path.is_symlink(), "missing or linked readback file")
    _need(path.stat().st_size <= file_cap, "readback file byte cap")
    size = 0
    with path.open("rb") as stream:
        for ordinal in range(row_cap + 1):
            line = stream.readline(line_cap + 1)
            if not line:
                return
            size += len(line)
            _need(ordinal < row_cap and len(line) <= line_cap and size <= file_cap, "readback row/byte cap")
            _need(line.endswith(b"\n"), "unterminated readback row")
            digest.update(line)
            value = json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
            yield value, size


def _event_id(value):
    _need(type(value) in (list, tuple) and len(value) == 3, "invalid readback EventID")
    return tuple(_uint(x, positive=True) for x in value)


def _tag(value):
    parts = _text(value).split(":")
    _need(len(parts) == 3 and parts[0] and parts[2], "explicit-process readback tag required")
    _need(not any(x in parts[2] for x in "@*?:"), "literal readback process required")
    return parts


def _provenance(value, tag):
    _fields(value, "branch_name module_label product_instance_name process_name friendly_class_name parameter_set_ids")
    for key in ("branch_name", "module_label", "process_name", "friendly_class_name"):
        _text(value[key])
    _text(value["product_instance_name"], empty=True)
    _need([value[k] for k in ("module_label", "product_instance_name", "process_name")] == _tag(tag),
          "resolved readback provenance/tag mismatch")
    ids = value["parameter_set_ids"]
    _need(type(ids) is list and ids and len(set(_text(x) for x in ids)) == len(ids),
          "missing or duplicate readback parameter set IDs")


def _pointer(value):
    if value is None:
        return None
    _fields(value, "product_id key")
    return _uint(value["product_id"], positive=True), _uint(value["key"])


def _generated(values, product, cap):
    _need(type(values) is list and len(values) <= cap, "readback generator cap/type")
    protons = 0
    for key, value in enumerate(values):
        _fields(value, "generator_product_id generator_key pdg generator_id position_mu2e_mm four_momentum_MeV global_time_ns proper_time_ns generator_weight")
        _need(_uint(value["generator_product_id"], True) == product and _uint(value["generator_key"]) == key,
              "readback generator product/key")
        _need(type(value["pdg"]) is int and type(value["generator_id"]) is int, "readback generator identity type")
        for field, size in (("position_mu2e_mm", 3), ("four_momentum_MeV", 4)):
            vector = value[field]
            _need(type(vector) is list and len(vector) == size, "readback generator vector")
            for item in vector:
                _number(item)
        for field in ("global_time_ns", "proper_time_ns"):
            _number(value[field])
        _need(value["generator_weight"] is None, "invented generator weight")
        protons += value["pdg"] == 2212
    return protons


def _particles(values, product, generated_product, generator_size, cap):
    _need(type(values) is list and len(values) <= cap, "readback SimParticle cap/type")
    particles, children = {}, {}
    for value in values:
        _fields(value, "product_id key object_key sim_stage pdg creation_code parent generator daughters")
        key = _uint(value["key"])
        _need(_uint(value["product_id"], True) == product and _uint(value["object_key"]) == key and key not in particles,
              "readback SimParticle collection/object identity")
        _uint(value["sim_stage"])
        _need(type(value["pdg"]) is int and type(value["creation_code"]) is int, "readback SimParticle identity type")
        parent, generator = _pointer(value["parent"]), _pointer(value["generator"])
        if parent is None:
            _need(generator is not None and generator[0] == generated_product and generator[1] < generator_size,
                  "readback root generator identity")
        else:
            _need(parent[0] == product and parent[1] != key and generator is None, "readback parent product/identity")
            children.setdefault(parent[1], set()).add(key)
        daughters = value["daughters"]
        _need(type(daughters) is list and len(daughters) <= cap, "readback daughter cap/type")
        refs = [_pointer(x) for x in daughters]
        _need(all(x is not None and x[0] == product and x[1] != key for x in refs) and len(set(refs)) == len(refs),
              "readback invalid/duplicate daughter")
        particles[key] = value
    _need(set(children) <= set(particles), "readback dangling parent")
    for key, value in particles.items():
        _need({x["key"] for x in value["daughters"]} == children.get(key, set()), "readback nonreciprocal daughters")
    return particles


def _native(directory, line_cap, file_cap, row_cap):
    path = Path(directory)
    _need(path.is_dir() and not path.is_symlink(), "missing or linked native directory")
    receipt = audit_native(path, max_line_bytes=line_cap, max_file_bytes=file_cap, max_rows=row_cap)
    _need(receipt["source_reconciliation"] == "structurally_consistent_in_memory", "v2 source ledger required")
    header = end = event_id = None
    generated, finals = [], {}
    for filename in ("records.jsonl", "completion.json"):
        digest = hashlib.sha256()
        for row, _ in _rows(path / filename, line_cap, file_cap, row_cap, digest):
            identity = tuple(row[x] for x in ("run", "subrun", "event"))
            _need(event_id is None or identity == event_id, "native EventID changed during join")
            event_id = identity
            kind, data = row["kind"], row["data"]
            if kind == "source_begin":
                header = data
            elif kind == "generated_primary":
                generated.append(data)
            elif kind == "sim_particle_final":
                finals[data["key"]] = data
            elif kind == "event_end":
                end = data
        _need(digest.hexdigest() == receipt["file_sha256"][filename], "native bytes changed during join")
    return event_id, (header, generated, finals, end, receipt)


def _audit(directory, native_directories, requested_event_ids, limits):
    line_cap, file_cap, row_cap, event_cap, particle_cap, native_byte_cap = limits
    _need(type(requested_event_ids) in (list, tuple) and len(requested_event_ids) <= event_cap,
          "bounded requested EventID list required")
    requested = [_event_id(x) for x in requested_event_ids]
    _need(len(set(requested)) == len(requested), "duplicate requested EventID")
    _need(type(native_directories) in (list, tuple) and len(native_directories) <= event_cap,
          "bounded native directory list required")
    native, native_bytes, audited_native_bytes = {}, 0, 0
    for path in native_directories:
        for name in ("records.jsonl", "completion.json"):
            file = Path(path) / name
            _need(file.is_file() and not file.is_symlink(), "missing or linked native file")
            native_bytes += file.stat().st_size
        _need(native_bytes <= native_byte_cap, "aggregate native byte cap")
        identity, info = _native(path, line_cap, file_cap, row_cap)
        audited_native_bytes += sum(info[4]["file_bytes"].values())
        _need(audited_native_bytes <= native_byte_cap, "aggregate audited native byte cap")
        _need(identity not in native, "duplicate native EventID")
        native[identity] = info
    _need(set(native) == set(requested), "requested/native EventID mismatch")
    directory = Path(directory)
    _need(directory.is_dir() and not directory.is_symlink(), "missing or linked readback directory")
    digest, rows, file_bytes = hashlib.sha256(), 0, 0
    header = terminal = current_file = None
    files, closed, seen, subruns = set(), set(), set(), {}
    event_counts = Counter()
    subrun_files, histories = {}, {}
    generated_protons = generated_entries = persisted_particles = 0
    for row, file_bytes in _rows(directory / "readback.jsonl", line_cap, file_cap, row_cap, digest):
        rows += 1
        _need(terminal is None, "readback row after terminal")
        _fields(row, "schema kind data proposal_only library_eligible")
        _need(row["schema"] == SCHEMA and row["proposal_only"] is True and row["library_eligible"] is False,
              "readback envelope/eligibility")
        kind, data = row["kind"], row["data"]
        _need(type(data) is dict and type(kind) is str, "readback row kind/payload")
        if header is None:
            _need(kind == "reader_begin", "missing readback begin")
            _fields(data, "generator_tag sim_tag status_tag count_tag max_rows max_bytes max_events max_particles_per_event")
            for key in ("generator_tag", "sim_tag", "status_tag", "count_tag"):
                _tag(data[key])
            for key, cap in (("max_rows", row_cap), ("max_bytes", file_cap), ("max_events", event_cap),
                             ("max_particles_per_event", particle_cap)):
                _need(_uint(data[key], True) <= cap, "reader cap exceeds caller cap")
            header = data
        elif kind in ("file_open", "file_close"):
            _fields(data, "input_file_name")
            name = _text(data["input_file_name"])
            if kind == "file_open":
                _need(current_file is None and name not in files, "nested or repeated input file")
                files.add(name)
                current_file = name
            else:
                _need(current_file == name, "readback file close mismatch")
                _need(all(identity in subruns for identity, owner in subrun_files.items() if owner == name),
                      "file close before SubRun closure")
                closed.add(name)
                current_file = None
        elif kind == "event":
            _fields(data, "input_file_name event_id process_history_id generator_product_id generator_provenance generated_primaries sim_product_id sim_provenance sim_particles status_product_id status_provenance status links_resolved_in_readback persistent_output_verified")
            _need(current_file is not None and data["input_file_name"] == current_file, "event outside input file")
            identity = _event_id(data["event_id"])
            _need(identity not in seen and identity in native and len(seen) < header["max_events"], "duplicate, unexpected or capped readback event")
            _need(identity[:2] not in subruns, "event after SubRun closure")
            seen.add(identity)
            _need(subrun_files.setdefault(identity[:2], current_file) == current_file,
                  "SubRun split across input files is unsupported")
            event_counts[identity[:2]] += 1
            source, generated, finals, end, _ = native[identity]
            histories[":".join(map(str, identity))] = dict(source=source["process_history_id"],
                                                        readback=_text(data["process_history_id"]))
            gp, sp, status_pid = (_uint(data[x], True) for x in ("generator_product_id", "sim_product_id", "status_product_id"))
            _need(len({gp, sp, status_pid}) == 3, "aliased readback product IDs")
            _need(gp == source["generator_product_id"] and sp == source["sim_product_id"], "source/readback product mismatch")
            for prefix, tag in (("generator", "generator_tag"), ("sim", "sim_tag"), ("status", "status_tag")):
                _provenance(data[prefix + "_provenance"], header[tag])
            _need(data["generator_provenance"] == source["generator_provenance"], "source/readback generator provenance mismatch")
            generated_protons += _generated(data["generated_primaries"], gp, header["max_particles_per_event"])
            _need(data["generated_primaries"] == generated, "source/readback generator payload mismatch")
            generated_entries += len(generated)
            particles = _particles(data["sim_particles"], sp, gp, len(generated), header["max_particles_per_event"])
            _need(set(particles) == set(finals), "in-memory/readback SimParticle population mismatch")
            common = "product_id key sim_stage pdg creation_code parent generator daughters".split()
            for key, particle in particles.items():
                _need(all(particle[x] == finals[key][x] for x in common), "in-memory/readback SimParticle identity mismatch")
            persisted_particles += len(particles)
            status = data["status"]
            _fields(status, "status_native sim_overflow step_kills field_kills n_g4_tracks")
            _need(type(status["status_native"]) is int and type(status["sim_overflow"]) is bool, "readback status type")
            for key in ("step_kills", "field_kills", "n_g4_tracks"):
                _uint(status[key])
            _need(all(status[x] == end[x] for x in ("status_native", "sim_overflow", "step_kills", "field_kills")),
                  "native/readback status mismatch")
            _need(status["status_native"] == 0 and status["sim_overflow"] is False and status["step_kills"] == 0 and status["field_kills"] == 0,
                  "abnormal/truncated intended shard")
            _need(data["links_resolved_in_readback"] is True and data["persistent_output_verified"] is False,
                  "readback link observation or certification claim")
        elif kind == "subrun":
            _fields(data, "input_file_name run subrun count_product_id count_provenance generated_event_callbacks")
            _need(current_file is not None and data["input_file_name"] == current_file, "SubRun outside input file")
            identity = (_uint(data["run"], True), _uint(data["subrun"], True))
            _need(identity not in subruns and identity in event_counts and subrun_files[identity] == current_file,
                  "duplicate, eventless or wrong-file SubRun")
            _uint(data["count_product_id"], True)
            _provenance(data["count_provenance"], header["count_tag"])
            count = _uint(data["generated_event_callbacks"])
            _need(count == event_counts[identity], "SubRun callbacks/written EventID count mismatch")
            subruns[identity] = count
        elif kind == "reader_end":
            _fields(data, "event_count subrun_count file_count readback_callback_complete framework_job_success_verified persistent_output_verified exposure_certified")
            _need(current_file is None and files == closed and files, "unclosed/empty readback input set")
            _need(seen == set(requested) and set(subruns) == set(event_counts), "incomplete written/requested/SubRun sets")
            for key, count in (("event_count", len(seen)), ("subrun_count", len(subruns)), ("file_count", len(files))):
                _need(_uint(data[key]) == count, "readback terminal count mismatch")
            _need(data["readback_callback_complete"] is True and all(data[x] is False for x in
                  ("framework_job_success_verified", "persistent_output_verified", "exposure_certified")), "readback terminal claim")
            terminal = data
        else:
            raise ReadbackError("unexpected readback kind")
        _need(rows <= header["max_rows"] and file_bytes <= header["max_bytes"], "reader declared row/byte cap")
    _need(terminal is not None, "incomplete readback-shaped stream")
    return dict(schema=SCHEMA, library_eligible=False, persistent_output_verified=False, exposure_certified=False,
                source_valid=False, native_valid=False, geometry_certified=False, physics_certified=False,
                readback_reconciliation="structurally_consistent_original_product_shaped_join",
                original_file_content_verified=False, registry_authenticated=False, framework_job_success_verified=False,
                output_selection_verified=False, weighting_policy_verified=False, shard_disjointness_verified=False,
                process_histories_observed=histories, process_history_relation_authenticated=False,
                pot_normalization=None, event_count=len(seen), subrun_count=len(subruns), file_count=len(files),
                generated_entries_observed=generated_entries, generated_protons_observed=generated_protons,
                completed_protons_observed=None, sim_particles_observed=persisted_particles,
                readback_sha256=digest.hexdigest(), readback_bytes=file_bytes, row_count=rows,
                native_file_sha256={":".join(map(str, identity)): info[4]["file_sha256"] for identity, info in native.items()})


def audit(readback_directory, native_directories, *, requested_event_ids,
          max_line_bytes=1 << 20, max_file_bytes=64 << 20, max_rows=100_000,
          max_events=10_000, max_particles_per_event=100_000, max_native_bytes=64 << 20):
    """Join bounded diagnostic streams; caller requests are not authenticated exposure."""
    limits = (max_line_bytes, max_file_bytes, max_rows, max_events, max_particles_per_event, max_native_bytes)
    _need(all(type(x) is int and 0 < x <= U64 for x in limits), "invalid readback limits")
    try:
        return _audit(readback_directory, native_directories, requested_event_ids, limits)
    except (OSError, ValueError, KeyError, TypeError, RecursionError, OverflowError) as exc:
        if isinstance(exc, ReadbackError):
            raise
        raise ReadbackError("malformed/incomplete readback: " + str(exc)) from exc
