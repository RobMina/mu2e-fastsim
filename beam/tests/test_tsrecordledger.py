"""Synthetic native-shaped diagnostics: structural success is never certification."""
from collections import Counter
from copy import deepcopy
import hashlib
import json

import pytest

from beamfastsim.tsrecordledger import LedgerError, SCHEMA, audit


@pytest.fixture
def ledger(tmp_path):
    def make(mode="empty", mutate=None, completion_mutate=None):
        rows = []
        history = [dict(instance_id=1, copy_no=0, physical_name="Target",
                        logical_name="TargetLV", daughter_index=None)]

        def geometry(track=False):
            return dict(instance_id=1, target_member=True,
                        availability=("native_track_touchable_not_vertex_membership"
                                      if track else "native_touchable"),
                        copy_history_root_to_leaf=deepcopy(history))

        def causal(parent=0, token=None):
            return dict(parent_id=parent, child_token=token,
                        birth_association=("explicit_primary_root" if not parent
                                           else "observed_secondary_birth_incidence"))

        def add(kind, **data):
            rows.append(dict(schema=SCHEMA, kind=kind, data=deepcopy(data),
                             row=len(rows) + 1, run=1, subrun=2, event=3,
                             library_eligible=False, proposal_only=True,
                             geometry_file="records.jsonl", geometry_hash=None,
                             hash_binding="requires_external_postread_file_hash"))

        def stack(tid=1, parent=0, token=None, killed=False):
            add("stacking", track_id=tid, parent_id=parent, pdg=11,
                classification_native=-9 if killed else 0, observation_number=1,
                stacking_cut=killed, common_cut_evaluated=not killed,
                common_cut=None if killed else False, causal=causal(parent, token))

        add("event_begin", outcome="begun_not_complete", proposal_only=True,
            target_physical_names=["Target"], target_material="Tungsten",
            max_rows=1000, max_volumes=10, max_bytes=1 << 20)
        add("volume", instance_id=1, parent_id=None, target_member=True,
            physical_name="Target", logical_name="TargetLV", material="Tungsten",
            copy_history_root_to_leaf=history)
        add("inventory_end", volume_count=1)
        stepped, born = mode in ("target", "birth"), mode == "birth"
        if mode != "empty":
            stack()
            native = dict(track_id=1, parent_id=0, pdg=11, current_step_number=0,
                          native_status=0, track_geometry=geometry(True))
            add("track_begin", native=native, causal=causal(), lifecycle_visit=1,
                stacking_observations=1)
            if stepped:
                endpoint = dict(geometry=geometry(), step_status_native=4)
                parent_step = dict(track_id=1, step_number=1, lifecycle_visit=1,
                                   target_step_row_present=True, boundary_observed=False,
                                   transition_observation="internal", pre=endpoint,
                                   post=deepcopy(endpoint))
                add("target_step", parent_step=parent_step, target_step_ordinal=1,
                    track_status_native=0, causal=causal())
                if born:
                    child = dict(track_id=0, parent_id=1, pdg=11,
                                 track_geometry=geometry(True))
                    add("secondary_birth", child_token=1, parent_step=parent_step,
                        parent_causal=causal(), raw_child_track_id=0,
                        raw_child_parent_id=1, pdg=11, native_child_at_birth=child)
                add("cuts", track_id=1, parent_id=0, step_number=1, lifecycle_visit=1,
                    stepping_cut=False, common_cut_evaluated=True, common_cut=False,
                    native_status=2, causal=causal())
            native.update(current_step_number=int(stepped), native_status=2)
            add("track_end", native=native, lifecycle_visit=1, causal=causal())
            if born:
                add("birth_track_association", child_token=1, track_id=2,
                    parent_id=1, parent_step=1, observed_at="stacking")
                stack(2, 1, 1, killed=True)
        accounting = dict(secondary_births=int(born), associated_births=int(born),
                          unconsumed_birth_tokens=0, unresolved_nonprimary_tracks=0,
                          stacked_not_begun_at_end=0, suspended_or_postponed_ended_visits=0,
                          status_native=0, event_aborted_native=False,
                          sim_overflow=False, step_kills=0, field_kills=0)
        add("event_end", **accounting, tracks_without_begin=int(born),
            stacking_killed_native_enum=int(born), zero_target_steps=not stepped,
            zero_exit_or_exposure_claim=False)
        if mutate:
            mutate(rows)
        # Rebuild bookkeeping after a semantic mutation, so byte/count mismatches
        # do not accidentally provide the only reason a negative control fails.
        for number, row in enumerate(rows, 1):
            row["row"] = number
        kinds = Counter(row["kind"] for row in rows)
        hooks = dict(beginEvent=1, endEvent=1, stacking=kinds["stacking"],
                     beginTrack=kinds["track_begin"], endTrack=kinds["track_end"],
                     step=kinds["cuts"], cuts=kinds["cuts"])
        encode = lambda row: (json.dumps(row, separators=(",", ":")) + "\n").encode()
        records = b"".join(map(encode, rows))
        kinds["completion"] += 1
        completion = dict(accounting, outcome="endEvent_and_ledger_flush_complete",
                          library_eligible=False, row_count=len(rows) + 1,
                          attempted_rows=len(rows) + 1, per_kind_counts=dict(kinds),
                          observed_hooks=hooks, volume_count=1,
                          target_step_count=int(stepped), ledger_bytes=len(records),
                          manifest_hash=None, recording_row_overflow=False,
                          recording_byte_overflow=False, recording_volume_overflow=False,
                          byte_cap_scope="records.jsonl_plus_completion.json; filesystem_metadata_excluded",
                          completion_not_physics_or_exposure_certification=True)
        add("completion", **completion)
        if completion_mutate:
            completion_mutate(rows[-1])
        (tmp_path / "records.jsonl").write_bytes(records)
        (tmp_path / "completion.json").write_bytes(encode(rows[-1]))
        return tmp_path
    return make


@pytest.mark.parametrize("mode", ["empty", "primary", "target", "birth"])
def test_complete_diagnostic_never_certifies(ledger, mode):
    directory = ledger(mode)
    result = audit(directory)
    assert result["complete_source_only_diagnostic"] is True
    for key in ("library_eligible", "source_valid", "native_valid", "physics_certified",
                "native_certified", "source_native_valid", "exposure_certified"):
        assert result[key] is False
    assert result["external_step_coverage"] == result["stock_product_coverage"] == "unverified"
    assert result["per_kind_counts"]["completion"] == 1
    assert result["row_count"] == sum(result["per_kind_counts"].values())
    assert result["observed_hooks"]["step"] == int(mode in ("target", "birth"))
    if mode == "birth":
        assert result["per_kind_counts"]["stacking"] == 2
        assert result["per_kind_counts"]["track_begin"] == 1
    for name in ("records.jsonl", "completion.json"):
        raw = (directory / name).read_bytes()
        assert result["file_bytes"][name] == len(raw)
        assert result["file_sha256"][name] == hashlib.sha256(raw).hexdigest()


def data(rows, kind):
    return next(row["data"] for row in rows if row["kind"] == kind)


@pytest.mark.parametrize("change", [
    lambda r: r[1].update(event=99),
    lambda r: data(r, "track_end")["native"].update(track_id=99),
    lambda r: data(r, "track_end")["native"].update(native_status=1),
    lambda r: data(r, "track_end")["native"].update(native_status=4),
    lambda r: r.remove(next(x for x in r if x["kind"] == "stacking")),
    lambda r: data(r, "birth_track_association").update(child_token=2),
    lambda r: data(r, "birth_track_association").update(parent_step=2),
    lambda r: data(r, "secondary_birth").update(raw_child_track_id=7),
    lambda r: r.remove(next(x for x in r if x["kind"] == "birth_track_association")),
    lambda r: data(r, "cuts").update(step_number=2),
    lambda r: data(r, "event_end").update(sim_overflow=True),
    lambda r: data(r, "event_begin").update(max_rows=2),
], ids=["event-id", "track-id", "nonterminal", "suspended", "missing-stack",
        "birth-token", "birth-parent-step", "birth-raw-id", "missing-map",
        "step-sequence", "simulation-overflow", "native-row-cap"])
def test_invalid_lifecycle(ledger, change):
    with pytest.raises(LedgerError):
        audit(ledger("birth", mutate=change))


@pytest.mark.parametrize("change", [
    lambda r: r.update(row=99),
    lambda r: r["data"].update(row_count=99),
    lambda r: r["data"].update(attempted_rows=99),
    lambda r: r["data"].update(ledger_bytes=0),
    lambda r: r["data"]["per_kind_counts"].update(volume=2),
    lambda r: r["data"]["observed_hooks"].update(beginTrack=1),
    lambda r: r["data"].update(recording_row_overflow=True),
    lambda r: r["data"].update(recording_byte_overflow=True),
    lambda r: r["data"].update(recording_volume_overflow=True),
    lambda r: r["data"].update(completion_not_physics_or_exposure_certification=False),
    lambda r: r["data"].update(library_eligible=True),
], ids=["row-number", "rows", "attempts", "bytes", "kind-count", "hook-count",
        "row-overflow", "byte-overflow", "volume-overflow", "certification", "eligibility"])
def test_invalid_completion(ledger, change):
    with pytest.raises(LedgerError):
        audit(ledger(completion_mutate=change))


@pytest.mark.parametrize("insertion", [b'"event":3,', b'"unchecked":NaN,',
                                       b'"unchecked":1e999,'])
def test_strict_json(ledger, insertion):
    directory = ledger()
    path = directory / "records.jsonl"
    path.write_bytes(b"{" + insertion + path.read_bytes()[1:])
    with pytest.raises(LedgerError, match="duplicate JSON key|nonfinite JSON"):
        audit(directory)


@pytest.mark.parametrize("limits", [dict(max_rows=1), dict(max_line_bytes=8),
                                    dict(max_file_bytes=8)])
def test_reader_bounds(ledger, limits):
    with pytest.raises(LedgerError):
        audit(ledger(), **limits)


def test_signed_unassigned_birth_id(ledger):
    def change(rows):
        birth = data(rows, "secondary_birth")
        birth["raw_child_track_id"] = -1
        birth["native_child_at_birth"]["track_id"] = -1
    assert audit(ledger("birth", mutate=change))["library_eligible"] is False


@pytest.mark.parametrize("value", [True, 0.5, -(1 << 40)])
def test_invalid_raw_birth_ids(ledger, value):
    def change(rows):
        birth = data(rows, "secondary_birth")
        birth["raw_child_track_id"] = value
        birth["native_child_at_birth"]["track_id"] = value
    with pytest.raises(LedgerError):
        audit(ledger("birth", mutate=change))


def test_assigned_track_id_must_be_nonnegative(ledger):
    def change(rows):
        data(rows, "stacking")["track_id"] = -1
    with pytest.raises(LedgerError):
        audit(ledger("primary", mutate=change))


@pytest.mark.parametrize("where", ["event_end", "completion"])
@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_native_event_abort_must_be_explicit_false(ledger, where, value):
    def change(rows):
        data(rows, "event_end")["event_aborted_native"] = value

    def completion_change(row):
        row["data"]["event_aborted_native"] = value

    with pytest.raises(LedgerError):
        audit(ledger(mutate=change if where == "event_end" else None,
                     completion_mutate=completion_change if where == "completion" else None))


@pytest.mark.parametrize("where", ["event_end", "completion"])
def test_missing_native_event_abort_is_not_legacy_success(ledger, where):
    def change(rows):
        data(rows, "event_end").pop("event_aborted_native")

    def completion_change(row):
        row["data"].pop("event_aborted_native")

    with pytest.raises(LedgerError):
        audit(ledger(mutate=change if where == "event_end" else None,
                     completion_mutate=completion_change if where == "completion" else None))


def test_kill_with_secondaries_cannot_be_followed_by_fabricated_stacking(ledger):
    def change(rows):
        data(rows, "track_end")["native"]["native_status"] = 3
    with pytest.raises(LedgerError, match="killed-secondary disposition"):
        audit(ledger("birth", mutate=change))


def test_kill_without_observed_offspring_remains_structural_only(ledger):
    def change(rows):
        data(rows, "track_end")["native"]["native_status"] = 3
    assert audit(ledger("primary", mutate=change))["library_eligible"] is False


@pytest.mark.parametrize("stepping", [True, False], ids=["stepping", "common"])
@pytest.mark.parametrize("status", [0, 1, 3, 4, 5, 6])
def test_evaluated_cut_requires_observed_stop_and_kill(ledger, stepping, status):
    def change(rows):
        data(rows, "cuts").update(stepping_cut=stepping,
                                   common_cut_evaluated=not stepping,
                                   common_cut=None if stepping else True,
                                   native_status=status)
    with pytest.raises(LedgerError):
        audit(ledger("target", mutate=change))


@pytest.mark.parametrize("stepping", [True, False], ids=["stepping", "common"])
def test_evaluated_cut_with_stop_and_kill_is_structural_only(ledger, stepping):
    def change(rows):
        data(rows, "cuts").update(stepping_cut=stepping,
                                   common_cut_evaluated=not stepping,
                                   common_cut=None if stepping else True,
                                   native_status=2)
    assert audit(ledger("target", mutate=change))["library_eligible"] is False
