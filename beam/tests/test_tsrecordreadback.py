"""Synthetic source-only joins, NOT actual art/native/persistence certification.

Reuse the independently authored complete v2 fixture, then project raw native-
shaped records into the readback proposal schema. Neither these JSONL fixtures
nor their successful audit authenticate original ROOT bytes,
Ptr dereferencing in art, product registries, job success, transport, or POT.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from beamfastsim.tsrecordreadback import ReadbackError, audit
from test_tsrecordsourceledger import (  # noqa: F401 -- imported pytest fixture
    data, duplicate, move_before, remove, set_path, source_ledger,
)


SCHEMA = "target-written-products-proposal-v1"
EVENT_ID = (1, 2, 3)
INPUT_NAME = "synthetic-not-an-original-art-file.root"
FALSE_FLAGS = (
    "library_eligible", "source_valid", "native_valid", "geometry_certified",
    "physics_certified", "persistent_output_verified", "exposure_certified",
    "original_file_content_verified", "registry_authenticated",
    "framework_job_success_verified", "output_selection_verified",
    "weighting_policy_verified", "shard_disjointness_verified",
)


def encode(rows):
    return b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode()
                    for row in rows)


def write_rows(directory, rows):
    raw = encode(rows)
    (directory / "readback.jsonl").write_bytes(raw)
    return raw


def provenance(module, instance, friendly):
    # Deliberately independent of auditor builders/constants.
    return dict(branch_name=f"{friendly}_{module}_{instance}_SIM.",
                module_label=module, product_instance_name=instance,
                process_name="SIM", friendly_class_name=friendly,
                parameter_set_ids=["fedcba9876543210"])


@pytest.fixture
def readback(source_ledger):
    def make(mode="proton", mutate=None, native_mutate=None):
        native = source_ledger(mode, mutate=native_mutate)
        source_rows = [json.loads(line) for line in
                       (native / "records.jsonl").read_bytes().splitlines()]
        begin = data(source_rows, "source_begin")
        end = data(source_rows, "event_end")
        identity = [source_rows[0][k] for k in ("run", "subrun", "event")]
        generated = [deepcopy(row["data"]) for row in source_rows
                     if row["kind"] == "generated_primary"]
        particles = []
        for row in source_rows:
            if row["kind"] != "sim_particle_final":
                continue
            final = row["data"]
            particle = {key: deepcopy(final[key]) for key in
                        "product_id key sim_stage pdg creation_code parent generator daughters".split()}
            particle["object_key"] = final["key"]
            particles.append(particle)
        rows = []

        def add(kind, **payload):
            rows.append(dict(schema=SCHEMA, kind=kind, data=payload,
                             proposal_only=True, library_eligible=False))

        add("reader_begin", generator_tag="generate:beam:SIM", sim_tag="g4run::SIM",
            status_tag="g4run:status:SIM", count_tag="generate:count:SIM",
            max_rows=1000, max_bytes=1 << 20, max_events=100,
            max_particles_per_event=100)
        add("file_open", input_file_name=INPUT_NAME)
        add("event", input_file_name=INPUT_NAME, event_id=identity,
            process_history_id=begin["process_history_id"],
            generator_product_id=begin["generator_product_id"],
            generator_provenance=deepcopy(begin["generator_provenance"]),
            generated_primaries=generated, sim_product_id=begin["sim_product_id"],
            sim_provenance=provenance("g4run", "", "SimParticles"),
            sim_particles=particles, status_product_id=109,
            status_provenance=provenance("g4run", "status", "StatusG4"),
            status={key: deepcopy(end[key]) for key in
                    ("status_native", "sim_overflow", "step_kills", "field_kills")},
            links_resolved_in_readback=True, persistent_output_verified=False)
        data(rows, "event")["status"]["n_g4_tracks"] = sum(
            row["kind"] == "track_begin" for row in source_rows)
        add("subrun", input_file_name=INPUT_NAME, run=identity[0], subrun=identity[1],
            count_product_id=113,
            count_provenance=provenance("generate", "count", "GenEventCount"),
            generated_event_callbacks=1)
        add("file_close", input_file_name=INPUT_NAME)
        add("reader_end", event_count=1, subrun_count=1, file_count=1,
            readback_callback_complete=True, framework_job_success_verified=False,
            persistent_output_verified=False, exposure_certified=False)
        if mutate:
            mutate(rows)
        directory = native / "readback-shaped"
        directory.mkdir()
        write_rows(directory, rows)
        return directory, native, rows
    return make


def check(directory, native, **kwargs):
    return audit(directory, [native], requested_event_ids=[EVENT_ID], **kwargs)


@pytest.mark.parametrize("mode,generated,protons,particles", [
    ("empty", 0, 0, 0), ("proton", 1, 1, 1), ("killed", 1, 1, 0),
    ("secondary", 1, 1, 2), ("mixed", 2, 1, 2),
])
def test_complete_synthetic_join_never_certifies(readback, mode, generated, protons, particles):
    directory, native, rows = readback(mode)
    result = check(directory, native)
    assert result["schema"] == SCHEMA
    assert result["readback_reconciliation"] == "structurally_consistent_original_product_shaped_join"
    for key in FALSE_FLAGS:
        assert result[key] is False
    assert result["pot_normalization"] is None
    assert (result["event_count"], result["subrun_count"], result["file_count"]) == (1, 1, 1)
    assert result["generated_entries_observed"] == generated
    assert result["generated_protons_observed"] == protons
    # Generated protons include stacking kills and cannot imply completed transport.
    assert result["completed_protons_observed"] is None
    assert result["sim_particles_observed"] == particles
    raw = (directory / "readback.jsonl").read_bytes()
    assert result["readback_bytes"] == len(raw)
    assert result["readback_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["row_count"] == len(rows)
    assert result["native_file_sha256"] == {"1:2:3": {
        name: hashlib.sha256((native / name).read_bytes()).hexdigest()
        for name in ("records.jsonl", "completion.json")}}
    assert data(rows, "subrun")["generated_event_callbacks"] == 1
    if mode == "empty":
        assert data(rows, "event")["generated_primaries"] == []
        assert data(rows, "event")["sim_particles"] == []
    if mode == "killed":
        assert data(rows, "event")["sim_particles"] == []
        assert data(rows, "event")["generated_primaries"][0]["pdg"] == 2212


def test_nondense_secondary_identity_is_not_native_track_arithmetic(readback):
    directory, native, rows = readback("secondary")
    particles = data(rows, "event")["sim_particles"]
    assert [p["key"] for p in particles] == [902, 11]
    assert [p["object_key"] for p in particles] == [902, 11]
    assert particles[0]["daughters"] == [{"product_id": 101, "key": 11}]
    assert particles[1]["parent"] == {"product_id": 101, "key": 902}
    assert particles[0]["generator"] == {"product_id": 83, "key": 0}
    assert particles[1]["generator"] is None
    # A map-vector's serialization order is not its key identity.
    particles.reverse()
    write_rows(directory, rows)
    assert check(directory, native)["sim_particles_observed"] == 2


# Each control changes one independent payload in otherwise complete byte streams.
@pytest.mark.parametrize("kind,path,value", [
    ("reader_begin", ["generator_tag"], "generate:beam"),
    ("reader_begin", ["sim_tag"], "g4run::"),
    ("reader_begin", ["status_tag"], "g4run"),
    ("reader_begin", ["count_tag"], "generate:count:"),
    ("reader_begin", ["max_rows"], 0),
    ("reader_begin", ["max_bytes"], True),
    ("reader_begin", ["max_events"], -1),
    ("reader_begin", ["max_particles_per_event"], 1),
    ("file_open", ["input_file_name"], ""),
    ("file_close", ["input_file_name"], "other.root"),
    ("event", ["input_file_name"], "other.root"),
    ("event", ["event_id"], [1, 2]),
    ("event", ["event_id"], [True, 2, 3]),
    ("event", ["event_id"], [1, 2, 0]),
    ("event", ["event_id"], [1, 2, 4]),
    ("event", ["process_history_id"], ""),
    ("event", ["generator_product_id"], 84),
    ("event", ["sim_product_id"], 102),
    ("event", ["status_product_id"], 101),
    ("event", ["status_product_id"], False),
    ("event", ["generated_primaries"], []),
    ("event", ["generated_primaries", 0, "generator_product_id"], 84),
    ("event", ["generated_primaries", 0, "generator_key"], 1),
    ("event", ["generated_primaries", 0, "pdg"], 13),
    ("event", ["generated_primaries", 0, "generator_id"], True),
    ("event", ["generated_primaries", 0, "position_mu2e_mm"], [1, 2]),
    ("event", ["generated_primaries", 0, "four_momentum_MeV"], [0, 0, 100, 944]),
    ("event", ["generated_primaries", 0, "global_time_ns"], 6.0),
    ("event", ["generated_primaries", 0, "proper_time_ns"], "0"),
    ("event", ["generated_primaries", 0, "generator_weight"], 1.0),
    ("event", ["sim_particles"], []),
    ("event", ["sim_particles", 0, "product_id"], 102),
    ("event", ["sim_particles", 0, "object_key"], 17),
    ("event", ["sim_particles", 0, "key"], 17),
    ("event", ["sim_particles", 0, "sim_stage"], 4),
    ("event", ["sim_particles", 0, "pdg"], 13),
    ("event", ["sim_particles", 0, "creation_code"], 2),
    ("event", ["sim_particles", 0, "parent"], {"product_id": 101, "key": 902}),
    ("event", ["sim_particles", 0, "generator"], None),
    ("event", ["sim_particles", 0, "generator"], {"product_id": 101, "key": 0}),
    ("event", ["sim_particles", 0, "generator"], {"product_id": 83, "key": 1}),
    ("event", ["sim_particles", 0, "daughters"], []),
    ("event", ["sim_particles", 0, "daughters"], [None]),
    ("event", ["sim_particles", 0, "daughters"], [{"product_id": 101, "key": 902}]),
    ("event", ["sim_particles", 1, "parent"], {"product_id": 101, "key": 999}),
    ("event", ["sim_particles", 1, "parent"], {"product_id": 83, "key": 902}),
    ("event", ["sim_particles", 1, "generator"], {"product_id": 83, "key": 0}),
    ("event", ["status", "status_native"], 1),
    ("event", ["status", "status_native"], False),
    ("event", ["status", "sim_overflow"], True),
    ("event", ["status", "sim_overflow"], 0),
    ("event", ["status", "step_kills"], 1),
    ("event", ["status", "field_kills"], 1),
    ("event", ["status", "n_g4_tracks"], -1),
    ("event", ["links_resolved_in_readback"], False),
    ("event", ["links_resolved_in_readback"], 1),
    ("event", ["persistent_output_verified"], True),
    ("subrun", ["input_file_name"], "other.root"),
    ("subrun", ["run"], 2),
    ("subrun", ["subrun"], False),
    ("subrun", ["count_product_id"], 0),
    ("subrun", ["generated_event_callbacks"], 0),
    ("subrun", ["generated_event_callbacks"], 2),
    ("subrun", ["generated_event_callbacks"], True),
    ("reader_end", ["event_count"], 2),
    ("reader_end", ["subrun_count"], 0),
    ("reader_end", ["file_count"], 2),
    ("reader_end", ["readback_callback_complete"], False),
    ("reader_end", ["framework_job_success_verified"], True),
    ("reader_end", ["persistent_output_verified"], True),
    ("reader_end", ["exposure_certified"], True),
])
def test_independent_readback_payload_controls(readback, kind, path, value):
    directory, native, _ = readback("secondary", mutate=lambda rows:
                                    set_path(data(rows, kind), path, value))
    with pytest.raises(ReadbackError):
        check(directory, native)


@pytest.mark.parametrize("kind,prefix", [
    ("event", "generator"), ("event", "sim"), ("event", "status"),
    ("subrun", "count"),
])
@pytest.mark.parametrize("field,value", [
    ("module_label", "wrong"), ("product_instance_name", "wrong"),
    ("process_name", "OTHER"), ("branch_name", ""),
    ("friendly_class_name", ""), ("parameter_set_ids", []),
    ("parameter_set_ids", ["same", "same"]),
])
def test_provenance_must_resolve_explicit_tag(readback, kind, prefix, field, value):
    def change(rows):
        data(rows, kind)[prefix + "_provenance"][field] = value
    directory, native, _ = readback(mutate=change)
    with pytest.raises(ReadbackError):
        check(directory, native)


def test_plausible_but_changed_generator_provenance_does_not_join(readback):
    def change(rows):
        data(rows, "event")["generator_provenance"]["parameter_set_ids"] = ["other-valid-id"]
    directory, native, _ = readback(mutate=change)
    with pytest.raises(ReadbackError, match="generator provenance mismatch"):
        check(directory, native)


@pytest.mark.parametrize("kind", ["reader_begin", "file_open", "event", "subrun", "file_close", "reader_end"])
@pytest.mark.parametrize("operation", ["missing", "duplicate", "extra_payload_field"])
def test_row_presence_uniqueness_and_exact_payloads(readback, kind, operation):
    def change(rows):
        if operation == "missing":
            remove(rows, kind)
        elif operation == "duplicate":
            duplicate(rows, kind)
        else:
            data(rows, kind)["unsupported_claim"] = False
    directory, native, _ = readback(mutate=change)
    with pytest.raises(ReadbackError):
        check(directory, native)


@pytest.mark.parametrize("kind,before", [
    ("event", "file_open"), ("subrun", "event"), ("file_close", "subrun"),
    ("reader_end", "file_close"), ("subrun", "reader_begin"),
])
def test_out_of_order_rows(readback, kind, before):
    directory, native, _ = readback(mutate=lambda rows: move_before(rows, kind, before))
    with pytest.raises(ReadbackError):
        check(directory, native)


@pytest.mark.parametrize("field,value", [
    ("schema", "target-recording-proposal-v2"), ("proposal_only", False),
    ("proposal_only", 1), ("library_eligible", True), ("library_eligible", 0),
    ("kind", "unknown"), ("data", []), ("extra", None),
])
def test_envelope_controls(readback, field, value):
    def change(rows):
        rows[2][field] = value
    directory, native, _ = readback(mutate=change)
    with pytest.raises(ReadbackError):
        check(directory, native)


@pytest.mark.parametrize("case", ["duplicate_particle", "duplicate_daughter", "row_after_end", "reopen", "event_after_subrun"])
def test_duplicate_and_closed_identity_controls(readback, case):
    directory, native, rows = readback("secondary")
    event = data(rows, "event")
    if case == "duplicate_particle":
        event["sim_particles"].append(deepcopy(event["sim_particles"][0]))
    elif case == "duplicate_daughter":
        event["sim_particles"][0]["daughters"] *= 2
    elif case == "row_after_end":
        rows.append(deepcopy(rows[1]))
    elif case == "reopen":
        rows[-1:-1] = [deepcopy(rows[1]), deepcopy(rows[-2])]
    else:
        # Close SubRun before its first written event, not a valid callback order.
        move_before(rows, "subrun", "event")
    write_rows(directory, rows)
    with pytest.raises(ReadbackError):
        check(directory, native)


@pytest.mark.parametrize("requested,native_mode", [
    ([], "normal"), ([(1, 2, 4)], "normal"),
    ([EVENT_ID, EVENT_ID], "normal"), (None, "normal"),
    ([(1, 2, True)], "normal"), ([(1, 2, 1 << 64)], "normal"),
    ([EVENT_ID], "missing"), ([EVENT_ID], "duplicate"),
])
def test_requested_native_written_event_sets(readback, requested, native_mode):
    directory, native, _ = readback()
    natives = [] if native_mode == "missing" else [native, native] if native_mode == "duplicate" else [native]
    with pytest.raises(ReadbackError):
        audit(directory, natives, requested_event_ids=requested)


@pytest.mark.parametrize("name", ["max_line_bytes", "max_file_bytes", "max_rows", "max_events", "max_particles_per_event"])
@pytest.mark.parametrize("value", [0, -1, True, 1.0, 1 << 64])
def test_invalid_caller_limits(readback, name, value):
    directory, native, _ = readback()
    with pytest.raises(ReadbackError):
        check(directory, native, **{name: value})


@pytest.mark.parametrize("name,value", [
    ("max_line_bytes", 32), ("max_file_bytes", 64), ("max_rows", 2),
    ("max_events", 1), ("max_particles_per_event", 1),
])
def test_caller_caps_cannot_be_overridden_by_reader(readback, name, value):
    directory, native, _ = readback("secondary")
    with pytest.raises(ReadbackError):
        check(directory, native, **{name: value})


@pytest.mark.parametrize("field,value", [("max_rows", 5), ("max_bytes", 100), ("max_events", 10001), ("max_particles_per_event", 100001)])
def test_reader_declared_caps_enforced(readback, field, value):
    directory, native, _ = readback(mutate=lambda rows: data(rows, "reader_begin").update({field: value}))
    with pytest.raises(ReadbackError):
        check(directory, native)


def test_exact_reader_row_and_byte_boundaries(readback):
    directory, native, rows = readback("secondary")
    header = data(rows, "reader_begin")
    header.update(max_rows=len(rows), max_events=1, max_particles_per_event=2)
    # Decimal length of max_bytes is itself part of the serialized byte count.
    for _ in range(8):
        size = len(encode(rows))
        if header["max_bytes"] == size:
            break
        header["max_bytes"] = size
    raw = write_rows(directory, rows)
    assert header["max_bytes"] == len(raw)
    assert check(directory, native, max_events=1, max_particles_per_event=2)["readback_bytes"] == len(raw)
    header["max_bytes"] -= 1
    write_rows(directory, rows)
    with pytest.raises(ReadbackError):
        check(directory, native)


@pytest.mark.parametrize("case", ["duplicate_key", "duplicate_nested_key", "nan", "infinity", "negative_infinity", "overflow_float", "missing_lf", "blank_line", "invalid_utf8", "truncated", "array_row"])
def test_serialized_byte_controls(readback, case):
    directory, native, _ = readback()
    path = directory / "readback.jsonl"
    raw = path.read_bytes()
    if case == "duplicate_key":
        old, new = b'"proposal_only":true', b'"proposal_only":true,"proposal_only":true'
    elif case == "duplicate_nested_key":
        old, new = b'"generator_key":0', b'"generator_key":0,"generator_key":0'
    elif case in ("nan", "infinity", "negative_infinity", "overflow_float"):
        token = {"nan": b"NaN", "infinity": b"Infinity", "negative_infinity": b"-Infinity", "overflow_float": b"1e999"}[case]
        old, new = b'"global_time_ns":5.0', b'"global_time_ns":' + token
    else:
        old = new = None
    if old is not None:
        assert old in raw
        raw = raw.replace(old, new, 1)
    elif case == "missing_lf":
        raw = raw[:-1]
    elif case == "blank_line":
        raw += b"\n"
    elif case == "invalid_utf8":
        raw = b"\xff\n" + raw
    elif case == "truncated":
        raw = raw[:len(raw) // 2]
    else:
        raw = b"[]\n" + raw
    path.write_bytes(raw)
    with pytest.raises(ReadbackError):
        check(directory, native)


@pytest.mark.parametrize("target", ["readback_directory", "readback_file", "native_directory", "native_records", "native_completion"])
def test_symlink_evidence_rejected(readback, tmp_path, target):
    directory, native, _ = readback()
    if target.endswith("directory"):
        link = tmp_path / "directory-link"
        link.symlink_to(directory if target == "readback_directory" else native, target_is_directory=True)
        if target == "readback_directory":
            directory = link
        else:
            native = link
    else:
        path = (directory / "readback.jsonl" if target == "readback_file" else
                native / ("records.jsonl" if target == "native_records" else "completion.json"))
        saved = path.with_name(path.name + ".real")
        path.rename(saved)
        path.symlink_to(saved)
    with pytest.raises(ReadbackError):
        check(directory, native)


@pytest.mark.parametrize("name", ["readback.jsonl", "records.jsonl", "completion.json"])
def test_missing_evidence_is_not_an_empty_success(readback, name):
    directory, native, _ = readback()
    ((directory if name == "readback.jsonl" else native) / name).unlink()
    with pytest.raises(ReadbackError):
        check(directory, native)


@pytest.mark.parametrize("kind,field", [
    ("reader_begin", "count_tag"), ("file_open", "input_file_name"),
    ("event", "generated_primaries"), ("event", "sim_particles"),
    ("event", "status_provenance"), ("subrun", "generated_event_callbacks"),
    ("file_close", "input_file_name"), ("reader_end", "readback_callback_complete"),
])
def test_required_payload_fields_cannot_be_omitted(readback, kind, field):
    directory, native, _ = readback(mutate=lambda rows: data(rows, kind).pop(field))
    with pytest.raises(ReadbackError):
        check(directory, native)


@pytest.mark.parametrize("path,value", [
    (["generated_primaries", 0, "pdg"], True),
    (["generated_primaries", 0, "position_mu2e_mm", 0], True),
    (["generated_primaries", 0, "four_momentum_MeV"], "not-a-vector"),
    (["sim_particles", 0, "sim_stage"], True),
    (["sim_particles", 0, "creation_code"], "1"),
    (["sim_particles", 0, "generator"], {"product_id": 83, "key": True}),
    (["sim_particles", 0, "generator"], {"product_id": 83, "key": 0, "available": True}),
    (["status", "n_g4_tracks"], 1 << 64),
])
def test_nested_types_and_exact_pointer_shape(readback, path, value):
    directory, native, _ = readback(mutate=lambda rows: set_path(data(rows, "event"), path, value))
    with pytest.raises(ReadbackError):
        check(directory, native)


@pytest.mark.parametrize("case", ["complete", "reverse_native_order", "missing_written", "late_event", "copied_callback_count"])
def test_two_requested_events_reconcile_as_sets_not_proton_or_counter_sums(readback, case):
    directory, native, rows = readback("mixed")
    # Clone synthetic bytes and recompute the completion ledger byte count.
    other_native = native / "second-synthetic-native"
    other_native.mkdir()
    native_rows = [json.loads(line) for line in (native / "records.jsonl").read_bytes().splitlines()]
    for row in native_rows:
        row["event"] = 4
    native_raw = encode(native_rows)
    (other_native / "records.jsonl").write_bytes(native_raw)
    completion_rows = [json.loads(line) for line in (native / "completion.json").read_bytes().splitlines()]
    for row in completion_rows:
        row["event"] = 4
        row["data"]["ledger_bytes"] = len(native_raw)
    (other_native / "completion.json").write_bytes(encode(completion_rows))
    second = deepcopy(next(row for row in rows if row["kind"] == "event"))
    second["data"]["event_id"] = [1, 2, 4]
    rows.insert(3, second)
    data(rows, "subrun")["generated_event_callbacks"] = 2
    data(rows, "reader_end")["event_count"] = 2
    if case == "missing_written":
        rows.remove(second)
    elif case == "late_event":
        rows.remove(second)
        rows.insert(next(i for i, row in enumerate(rows) if row["kind"] == "file_close"), second)
    elif case == "copied_callback_count":
        # GenEventCount is copied once per SubRun, not one count per proton.
        data(rows, "subrun")["generated_event_callbacks"] = 4
    write_rows(directory, rows)
    natives = [native, other_native]
    if case == "reverse_native_order":
        natives.reverse()
    if case in ("complete", "reverse_native_order"):
        result = audit(directory, natives, requested_event_ids=[(1, 2, 4), EVENT_ID])
        assert result["event_count"] == 2
        assert result["generated_entries_observed"] == 4
        assert result["generated_protons_observed"] == 2
        assert result["sim_particles_observed"] == 4
        assert set(result["native_file_sha256"]) == {"1:2:3", "1:2:4"}
        for flag in FALSE_FLAGS:
            assert result[flag] is False
    else:
        with pytest.raises(ReadbackError):
            audit(directory, natives, requested_event_ids=[EVENT_ID, (1, 2, 4)])


@pytest.mark.parametrize("field,value", [
    ("status_native", 1), ("step_kills", 1), ("field_kills", 1),
    ("sim_overflow", True),
])
def test_matching_abnormal_native_and_readback_status_is_not_an_intended_shard(readback, field, value):
    # Matching abnormal source and readback statuses still refuse the intended shard.
    directory, native, rows = readback(native_mutate=lambda source_rows:
                                     data(source_rows, "event_end").update({field: value}))
    assert data(rows, "event")["status"][field] == value
    with pytest.raises(ReadbackError):
        check(directory, native)


def test_zero_byte_stream_is_not_empty_generated_collection(readback):
    directory, native, _ = readback("empty")
    (directory / "readback.jsonl").write_bytes(b"")
    with pytest.raises(ReadbackError):
        check(directory, native)


def test_native_completion_must_still_bind_its_serialized_record_bytes(readback):
    directory, native, _ = readback()
    # Semantically identical JSON must still satisfy serialized native byte accounting.
    path = native / "records.jsonl"
    raw = path.read_bytes()
    assert b'"kind":' in raw
    path.write_bytes(raw.replace(b'"kind":', b'"kind": ', 1))
    with pytest.raises(ReadbackError):
        check(directory, native)


def test_process_history_observations_are_distinct_and_unauthenticated(readback):
    directory, native, rows = readback(mutate=lambda rows: data(rows, "event").update(
        process_history_id="different-reader-history"))
    result = check(directory, native)
    assert result["process_histories_observed"]["1:2:3"] == dict(
        source="0123456789abcdef", readback="different-reader-history")
    assert result["process_history_relation_authenticated"] is False
    assert result["persistent_output_verified"] is False


@pytest.mark.parametrize("case", ["close_before_subrun", "subrun_in_other_file"])
def test_file_cannot_disown_its_pending_subrun(readback, case):
    directory, native, rows = readback()
    move_before(rows, "file_close", "subrun")
    if case == "subrun_in_other_file":
        opened = deepcopy(rows[1])
        opened["data"]["input_file_name"] = "second.root"
        index = next(i for i, r in enumerate(rows) if r["kind"] == "subrun")
        rows.insert(index, opened)
        data(rows, "subrun")["input_file_name"] = "second.root"
        closed = deepcopy(next(r for r in rows if r["kind"] == "file_close"))
        closed["data"]["input_file_name"] = "second.root"
        rows.insert(-1, closed)
        data(rows, "reader_end")["file_count"] = 2
    write_rows(directory, rows)
    with pytest.raises(ReadbackError):
        check(directory, native)


def test_aggregate_native_byte_budget_has_exact_boundary(readback):
    directory, native, _ = readback()
    size = sum((native / name).stat().st_size for name in ("records.jsonl", "completion.json"))
    assert check(directory, native, max_native_bytes=size)["event_count"] == 1
    with pytest.raises(ReadbackError, match="aggregate native byte cap"):
        check(directory, native, max_native_bytes=size - 1)


@pytest.mark.parametrize("value", [0, True, -1, 1.0, 1 << 64])
def test_invalid_aggregate_native_byte_budget(readback, value):
    directory, native, _ = readback()
    with pytest.raises(ReadbackError):
        check(directory, native, max_native_bytes=value)


@pytest.mark.parametrize("process", ["@current_process", "@skipCurrentProcess", "*", "PRO?ESS"])
def test_process_alias_is_not_an_explicit_original_product(readback, process):
    directory, native, rows = readback()
    data(rows, "reader_begin")["sim_tag"] = "g4run::" + process
    data(rows, "event")["sim_provenance"]["process_name"] = process
    write_rows(directory, rows)
    with pytest.raises(ReadbackError):
        check(directory, native)


def test_reader_source_is_an_uninstalled_no_g4_proposal():
    from pathlib import Path
    source = (Path(__file__).parents[1] / "native" / "TargetReadback_module.cc").read_text()
    assert '#include "art/Framework/Core/FileBlock.h"' in source
    assert "getValidHandle<GenParticleCollection>" in source
    assert "getValidHandle<SimParticleCollection>" in source
    assert "getValidHandle<StatusG4>" in source
    assert "getValidHandle<GenEventCount>" in source
    assert "ptr.isAvailable()" in source and "ptr.get() != &found->second" in source
    assert "O_EXCL" in source and "O_NOFOLLOW" in source
    assert "~Output()" in source and 'row("reader_end"' in source
    for forbidden in ("G4RunManager", "BeamOn", "gallery::", "TTree", "new G4Navigator"):
        assert forbidden not in source


def test_readback_proposal_manifest_binds_separate_assets_and_apis():
    from pathlib import Path
    root = Path(__file__).parents[1]
    pins = json.loads((root / "native" / "readback_pins.json").read_text())
    assert pins["schema"] == "target-written-products-source-proposal-v1"
    assert pins["offline_revision"] == "b24d4163e5e8f5ed2b5f72fccb25e4980fec64e2"
    for flag in ("library_eligible", "build_performed", "native_execution_performed", "original_art_read_performed"):
        assert pins[flag] is False
    for name, digest in pins["proposal_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert "MCDataProducts/inc/StatusG4.hh" in pins["offline_source_sha256"]
    assert "MCDataProducts/inc/GenEventCount.hh" in pins["offline_source_sha256"]
    assert "art/Framework/Core/FileBlock.h" in pins["installed_header_sha256"]
