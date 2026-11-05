"""Synthetic native emit-format contracts only; not a native execution certificate."""
from copy import deepcopy
import hashlib
import io
import json

import numpy as np
import pytest

from calofastsim.boundary_ledger import LedgerError, MANIFEST_FIELDS, SCHEMA, TRACK_STATUS_SCHEMA, audit, audit_file

SHA = "a" * 64
PROVENANCE = dict(purpose="diagnostic_boundary_ledger", source_sha256=SHA,
                  independent_manifest_sha256="b" * 64, track_status_schema=TRACK_STATUS_SCHEMA)
BIG = 2**63 + 17


def manifest(events=(BIG,)):
    n = len(events)
    return dict(source_sha256=np.full(n, SHA), run=np.ones(n, dtype=np.uint32),
                subrun=np.full(n, 2, dtype=np.uint32), event=np.array(events, dtype=np.uint64),
                root_product_id=np.full(n, 2**32 - 1, dtype=np.uint32),
                root_key=np.full(n, 2**64 - 1, dtype=np.uint64))


def state(m=0, step=False):
    nodes = [] if m < 0 else [dict(pv="World", lv="world", copy=0, replica=0)]
    if m == 1:
        nodes.append(dict(pv="Calo", lv="disk", copy=1, replica=1))
    s = dict(position=[1, 2, 3], momentum=[0, 0, 10], kinetic_energy=10,
             total_energy=11, global_time=1, proper_time=0, local_time=0,
             material=dict(name="Vacuum", index=0),
             touchable=dict(availability="available" if m >= 0 else "unavailable", nodes=nodes))
    s.update(step_status=1) if step else s.update(material_source="touchable_logical_not_step_material")
    return s


def fixture(event=BIG, born=False):
    rows = []

    def row(typ, **kw):
        rows.append(dict(schema=SCHEMA, type=typ, run=1, subrun=2, event=event,
                         sequence=len(rows), **kw))

    row("event_begin", g4_event=0, sim_stage=0, candidate_only=True,
        coordinate_frame="G4_world", units="mm_MeV_ns",
        product_id_encoding="art_ProductID_value_uint32", roots=["Calo"])
    track = dict(track=1, parent_track=0, particle=dict(product=2**32 - 1, key=BIG),
                 sim_saved_at_start=True, step=0, visit=int(born), track_status=0)
    row("born_inside" if born else "track_start", **track, membership=int(born),
        birth_phase="PreUserTracking_after_saveSimParticleStart", pdg=11,
        creator_process=None, persistent_parent=None, gen_particle=dict(product=1, key=0),
        state=state(int(born)))
    if not born:
        track.update(step=1, visit=1)
        row("entry", **track, pre_membership=0, post_membership=1,
            classification="first_observed_entry", endpoint="post_step",
            hook_phase="UserSteppingAction_before_stock_cuts", step_length=1,
            process="Transportation", pre=state(0, True), post=state(1, True))
    track.update(step=2, track_status=2)
    row("terminal_inside", **track, terminal=True, membership=1, process=None,
        endpoint_phase="last_post_step", state=state(1, True))
    row("event_end", g4_event=0, g4_aborted=False, stock_filter_pass=True,
        status_g4_equivalent=0, n_g4_tracks=1, overflow_sim_particles=False,
        killed_step_limit=0, killed_field=0, unclosed_visits=0,
        unknown_birth_memberships=0, missing_saved_particles=0,
        unknown_tracking_ends=0, rows_before_end=len(rows))
    return rows


def encoded(rows):
    return b"".join((json.dumps(r, separators=(",", ":")) + "\n").encode() for r in rows)


def run(rows, *, independent_manifest=None, provenance=None, **kw):
    data = rows if isinstance(rows, bytes) else encoded(rows)
    return audit(io.BytesIO(data), independent_manifest=manifest() if independent_manifest is None else independent_manifest,
                 provenance=PROVENANCE if provenance is None else provenance, **kw)


def reseq(rows):
    for i, r in enumerate(rows):
        r["sequence"] = i
        if r["type"] == "event_end":
            r["rows_before_end"] = i
    return rows


def test_native_emit_format_exact_ids_and_compact_report(tmp_path):
    rows = fixture()
    report = run(rows)
    assert report.reconciled and report.protocol_complete
    assert report.events["event"].tolist() == [BIG]
    assert report.events["root_key"].tolist() == [2**64 - 1]
    assert not report.events["incomplete"][0]
    assert report.counts["rows"] == 5
    assert not report.events["library_eligible"].any()
    assert not report.native_execution_certified
    assert "particle" not in report.events and "state" not in report.events
    path = tmp_path / "boundary.jsonl"
    path.write_bytes(encoded(rows))
    other = audit(path, independent_manifest=manifest(), provenance=PROVENANCE)
    assert other.ledger_sha256 == report.ledger_sha256


def test_missing_foreign_unordered_and_empty_independent_manifest():
    rows = fixture(BIG + 1) + fixture(BIG)
    report = run(rows, independent_manifest=manifest((BIG, BIG + 1, BIG + 2)))
    assert report.manifest_indices.tolist() == [1, 0]
    assert report.missing_manifest_indices.tolist() == [2]
    assert not report.reconciled
    report = run(fixture(99))
    assert report.foreign_event_indices.tolist() == [0]
    assert report.missing_manifest_indices.tolist() == [0]
    assert report.events["root_product_id"].tolist() == [0]  # unresolved, never inferred
    assert not run(b"").reconciled
    report = run(b"", independent_manifest=manifest(()))
    assert report.reconciled and report.protocol_complete and report.counts["events"] == 0


def test_failed_empty_and_born_inside_are_not_successful_zero():
    rows = fixture()
    rows = reseq([rows[0], rows[-1]])
    rows[-1].update(n_g4_tracks=0, g4_aborted=True, overflow_sim_particles=True,
                    status_g4_equivalent=10, stock_filter_pass=False)
    report = run(rows)
    assert report.events["transport_failed"][0]
    assert report.events["no_entry_observed"][0]
    assert report.events["ended"][0] and not report.events["library_eligible"][0]
    report = run(fixture(born=True))
    assert report.events["no_entry_observed"][0]  # born-inside is not exterior entry
    assert report.events["starts"][0] == 1
    assert report.events["born_inside"][0] == report.counts["born_inside"] == 1
    assert report.events["unsupported"][0]
    assert not report.events["no_domain_occupancy_observed"][0]


@pytest.mark.parametrize("cut", [1, 2, 3, 4])
def test_eof_retains_incomplete_event(cut):
    report = run(fixture()[:cut])
    assert report.reconciled and not report.protocol_complete
    assert report.events["incomplete"].tolist() == [True]
    assert not report.events["ended"][0]


def test_byte_truncation_and_invalid_row_keep_prefix_report():
    with pytest.raises(LedgerError) as exc:
        run(encoded(fixture())[:-3])
    assert exc.value.report.events["incomplete"][0]
    assert not exc.value.report.protocol_complete
    with pytest.raises(LedgerError) as exc:
        run(encoded(fixture()[:1]) + b"garbage\n")
    assert exc.value.report.counts["events"] == 1


@pytest.mark.parametrize("index,field,value", [
    (0, "event", float(BIG)), (0, "sequence", True), (0, "run", -1),
    (0, "candidate_only", False), (0, "roots", ["Calo", "Calo"]),
    (1, "visit", 1), (1, "track_status", 7), (1, "parent_track", 1.0),
    (2, "sequence", 9), (2, "visit", 2), (2, "classification", "reentry"),
    (2, "post_membership", True), (2, "step_length", -1),
    (3, "terminal", False), (3, "track_status", 4), (3, "visit", 2),
    (4, "g4_event", 1), (4, "status_g4_equivalent", 1),
    (4, "unclosed_visits", 1), (4, "unknown_birth_memberships", 1),
    (4, "rows_before_end", 1), (4, "n_g4_tracks", 0),
])
def test_protocol_mutations_rejected(index, field, value):
    rows = fixture()
    rows[index][field] = value
    with pytest.raises(LedgerError):
        run(rows)


def test_missing_duplicate_foreign_reordered_lifecycle():
    base = fixture()
    for rows in ([base[0], *base[2:]], [*base[:2], base[1], *base[2:]],
                 [base[0], base[2], base[1], *base[3:]], base + base,
                 [*base[:3], base[-1]], [base[-1]]):
        with pytest.raises(LedgerError):
            run(reseq(deepcopy(rows)))
    rows = reseq(deepcopy([*base[:3], base[-1]]))
    rows[-1]["unclosed_visits"] = 1
    assert run(rows).events["incomplete"][0]
    rows = fixture()
    rows[2]["event"] += 1
    with pytest.raises(LedgerError, match="outside its event"):
        run(rows)


@pytest.mark.parametrize("bad", [b'NaN', b'Infinity', b'-Infinity', b'1e999', b'123456789012345678901'])
def test_nonfinite_and_oversized_tokens_rejected_everywhere(bad):
    data = encoded(fixture()).replace(b'"kinetic_energy":10', b'"kinetic_energy":' + bad, 1)
    with pytest.raises(LedgerError):
        run(data)


def test_duplicate_keys_including_nested_and_unknown_fields():
    data = encoded(fixture())
    for changed in (data.replace(b'"run":1', b'"run":1,"run":1', 1),
                    data.replace(b'"product":4294967295', b'"product":1,"product":4294967295', 1)):
        with pytest.raises(LedgerError, match="duplicate JSON key"):
            run(changed)
    rows = fixture()
    rows[1]["unexpected"] = 0
    with pytest.raises(LedgerError, match="unexpected fields"):
        run(rows)


def test_exit_reentry_and_suspend_resume_lifecycle():
    rows = fixture()
    transition = deepcopy(rows[2])
    exit_row = deepcopy(transition)
    exit_row.update(type="exit", step=2, pre_membership=1, post_membership=0,
                    classification="exit", pre=state(1, True), post=state(0, True))
    stop = deepcopy(rows[3])
    stop.update(type="tracking_end", step=2, track_status=4, terminal=False,
                membership=0, state=state(0, True))
    resume = {k: deepcopy(stop[k]) for k in ("schema", "run", "subrun", "event", "sequence", "track", "parent_track", "particle", "sim_saved_at_start", "step", "visit")}
    resume.update(type="tracking_resume", track_status=0, state=state(0))
    transition.update(step=3, visit=2, classification="reentry")
    rows[3].update(step=4, visit=2)
    rows = reseq(rows[:3] + [exit_row, stop, resume, transition] + rows[3:])
    report = run(rows)
    assert report.events["entries"].tolist() == [2]
    assert not report.events["incomplete"][0]
    rows[5]["type"] = "track_start"
    with pytest.raises(LedgerError):
        run(rows)


def test_unknown_birth_missing_saved_and_postponed_explicit():
    rows = fixture()
    rows[1].update(membership=-1, state=state(-1), sim_saved_at_start=False, particle=None)
    unresolved = {k: deepcopy(rows[2][k]) for k in ("schema", "run", "subrun", "event", "sequence", "track", "parent_track", "particle", "sim_saved_at_start", "step", "visit", "track_status")}
    unresolved.update(type="birth_membership_unresolved", visit=0, first_observed_pre=state(0, True))
    for r in rows[2:4] + [unresolved]:
        r.update(sim_saved_at_start=False, particle=None)
    rows[-1].update(unknown_birth_memberships=1, missing_saved_particles=1)
    report = run(reseq(rows[:2] + [unresolved] + rows[2:]))
    assert report.events["unsupported"][0] and report.events["incomplete"][0]
    rows = fixture()
    rows[3].update(type="tracking_end", terminal=False, track_status=6)
    rows[-1]["unclosed_visits"] = 1
    report = run(rows)
    assert report.events["postponed"][0] and report.events["pending_tracks"][0] == 1


def test_membership_unavailable_is_fatal_prefix_not_empty_success():
    rows = fixture()
    row = {k: deepcopy(rows[2][k]) for k in ("schema", "type", "run", "subrun", "event", "sequence", "track", "parent_track", "particle", "sim_saved_at_start", "step", "visit", "track_status")}
    row.update(type="membership_unavailable", visit=0, pre=state(0, True), post=state(-1, True))
    report = run(rows[:2] + [row])
    assert not report.protocol_complete and report.events["incomplete"][0]
    with pytest.raises(LedgerError):
        run(reseq(rows[:2] + [row, rows[-1]]))


def test_manifest_provenance_and_preallocation_caps():
    for field, value in (("event", [float(BIG)]), ("event", [True]),
                         ("event", [[1] * 100]), ("root_product_id", [0]),
                         ("root_product_id", [2**32]), ("source_sha256", ["c" * 64])):
        m = manifest()
        m[field] = value
        with pytest.raises(LedgerError):
            run(fixture(), independent_manifest=m)
    with pytest.raises(LedgerError, match="duplicate independent"):
        run(fixture(), independent_manifest=manifest((BIG, BIG)))
    for field in ("purpose", "source_sha256", "independent_manifest_sha256", "track_status_schema"):
        p = dict(PROVENANCE)
        del p[field]
        with pytest.raises(LedgerError):
            run(fixture(), provenance=p)
    class NoRead:
        def readline(self, _):
            raise AssertionError("manifest cap must precede stream reading")
    with pytest.raises(LedgerError, match="row bound"):
        audit(NoRead(), independent_manifest=manifest((1, 2)), provenance=PROVENANCE, max_manifest_rows=1)
    assert set(manifest()) == set(MANIFEST_FIELDS)


def test_row_byte_line_caps_before_json_allocation():
    data = encoded(fixture())
    assert run(data, max_rows=5, max_bytes=len(data), max_line_bytes=len(data)).protocol_complete
    for kwargs in (dict(max_rows=4), dict(max_bytes=len(data)-1), dict(max_line_bytes=10)):
        with pytest.raises(LedgerError) as exc:
            run(data, **kwargs)
        assert not exc.value.report.protocol_complete
    class BoundedStream:
        def readline(self, limit):
            assert limit <= 17
            return b"x" * limit
    with pytest.raises(LedgerError, match="byte/line"):
        audit(BoundedStream(), independent_manifest=manifest(), provenance=PROVENANCE,
              max_line_bytes=16, max_bytes=100)
    for kwargs in (dict(max_rows=True), dict(max_bytes=0), dict(max_manifest_rows=-1)):
        with pytest.raises(LedgerError):
            run(data, **kwargs)


def test_world_boundary_unknown_end_and_at_rest_birth():
    rows = fixture()
    outside = state(0, True)
    outside["step_status"] = 0
    outside["touchable"] = dict(availability="world_null", nodes=[])
    exit_row = deepcopy(rows[2])
    exit_row.update(type="exit", step=2, pre_membership=1, post_membership=0,
                    classification="exit", pre=state(1, True), post=outside)
    rows[3].update(type="tracking_end", step=2, membership=0, state=outside)
    assert not run(reseq(rows[:3] + [exit_row] + rows[3:])).events["incomplete"][0]
    rows = fixture(born=True)
    rows[1]["state"]["momentum"] = [0, 0, 0]
    assert run(rows).events["unsupported"][0]  # exterior kernel excludes born inside
    rows[2].update(type="tracking_end", membership=-1, state=state(-1, True))
    rows[-1]["unknown_tracking_ends"] = 1
    report = run(rows)
    assert report.events["unknown_ends"][0] == 1
    assert report.events["unsupported"][0]


def test_resume_after_terminal_and_missing_resume_rejected():
    rows = fixture()
    resume = {k: deepcopy(rows[3][k]) for k in ("schema", "run", "subrun", "event", "sequence", "track", "parent_track", "particle", "sim_saved_at_start", "step", "visit", "track_status")}
    resume.update(type="tracking_resume", state=state(1))
    with pytest.raises(LedgerError, match="resume without"):
        run(reseq(rows[:-1] + [resume, rows[-1]]))
    rows = fixture()
    stop = deepcopy(rows[3])
    stop.update(type="tracking_end", terminal=False, track_status=4)
    with pytest.raises(LedgerError, match="without resume"):
        run(reseq(rows[:-2] + [stop] + rows[-2:]))


def test_complete_no_entry_status_kills_and_unfinished_outside():
    rows = fixture()
    empty = reseq([rows[0], rows[-1]])
    empty[-1]["n_g4_tracks"] = 0
    report = run(empty)
    assert report.events["no_entry_observed"][0]
    assert not report.events["library_eligible"][0]
    empty[-1].update(killed_field=1, status_g4_equivalent=1)
    assert run(empty).events["transport_failed"][0]
    rows = fixture()
    stop = rows[3]
    stop.update(type="tracking_end", step=0, visit=0, terminal=False,
                track_status=4, membership=0, state=state(0), endpoint_phase="track_no_step")
    report = run(reseq([rows[0], rows[1], stop, rows[-1]]))
    assert report.events["unclosed_visits"][0] == 0
    assert report.events["pending_tracks"][0] == 1
    assert report.events["incomplete"][0]


def test_deep_json_and_bool_manifest_rejected():
    with pytest.raises(LedgerError):
        run(b"[" * 2000 + b"0" + b"]" * 2000 + b"\n")
    m = manifest()
    m["root_key"] = np.array([True])
    with pytest.raises(LedgerError):
        run(fixture(), independent_manifest=m)
    rows = fixture()
    rows[1]["particle"]["key"] = 1.0
    with pytest.raises(LedgerError, match="particle key"):
        run(rows)


@pytest.mark.parametrize("index,step", [(1, 1), (2, 0)])
def test_birth_and_transition_step_contract(index, step):
    rows = fixture()
    rows[index]["step"] = step
    with pytest.raises(LedgerError):
        run(rows)


@pytest.mark.parametrize("step", [0, 2])
def test_unknown_birth_first_observation_is_exactly_step_one(step):
    rows = fixture()
    rows[1].update(membership=-1, state=state(-1))
    unresolved = {k: deepcopy(rows[2][k]) for k in ("schema", "run", "subrun", "event", "sequence", "track", "parent_track", "particle", "sim_saved_at_start", "step", "visit", "track_status")}
    unresolved.update(type="birth_membership_unresolved", step=step, visit=0,
                      first_observed_pre=state(0, True))
    with pytest.raises(LedgerError, match="step one"):
        run(reseq(rows[:2] + [unresolved] + rows[2:]))


@pytest.mark.parametrize("status", [0, 1, 4, 5, 6])
def test_actual_p110_nonterminal_end_semantics(status):
    rows = fixture()
    rows[3].update(type="tracking_end", terminal=False, track_status=status)
    rows[-1]["unclosed_visits"] = 1
    if status == 0:
        with pytest.raises(LedgerError, match="tracking end status"):
            run(rows)
    else:
        report = run(rows)
        assert bool(report.events["postponed"][0]) == (status == 6)
        assert bool(report.events["unsupported"][0]) == (status in (1, 6))
        assert report.events["pending_tracks"][0] == 1


def suspended_rows(status=4):
    rows = fixture()
    stop = deepcopy(rows[3])
    stop.update(type="tracking_end", terminal=False, track_status=status)
    resume = {k: deepcopy(stop[k]) for k in ("schema", "run", "subrun", "event", "sequence", "track", "parent_track", "particle", "sim_saved_at_start", "step", "visit")}
    resume.update(type="tracking_resume", track_status=0, state=state(1))
    rows[3]["step"] = 3
    return reseq(rows[:3] + [stop, resume] + rows[3:])


@pytest.mark.parametrize("status", [4, 5])
def test_supported_suspend_resume_statuses_and_continuity(status):
    rows = suspended_rows(status)
    report = run(rows)
    assert not report.events["postponed"][0]
    assert not report.events["incomplete"][0]
    for field, value in (("track_status", 4), ("step", 3), ("state", state(0)),
                         ("state", state(-1))):
        changed = deepcopy(rows)
        changed[4][field] = value
        with pytest.raises(LedgerError):
            run(changed)


@pytest.mark.parametrize("status", [1, 6])
def test_atrest_and_postponed_resume_unsupported(status):
    with pytest.raises(LedgerError, match="unsupported resume status"):
        run(suspended_rows(status))


def test_no_domain_occupancy_is_distinct_from_no_exterior_entry():
    rows = fixture()
    stop = rows[3]
    stop.update(type="tracking_end", step=0, visit=0, membership=0,
                state=state(0), endpoint_phase="track_no_step")
    report = run(reseq([rows[0], rows[1], stop, rows[-1]]))
    assert report.events["no_domain_occupancy_observed"][0]
    assert report.events["no_entry_observed"][0]
    assert not report.events["born_inside"][0]
    assert not run(fixture()).events["no_domain_occupancy_observed"][0]


def test_verified_canonical_artifact_loader(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"independent pre-response source")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    table = {k: v.tolist() for k, v in manifest().items()}
    table["source_sha256"] = [sha]
    data = json.dumps(table, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    artifact = tmp_path / "manifest.json"
    artifact.write_bytes(data)
    provenance = dict(PROVENANCE, source_sha256=sha,
                      independent_manifest_sha256=hashlib.sha256(data).hexdigest())
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_bytes(encoded(fixture()))

    def verified(**kw):
        return audit_file(ledger, independent_manifest_path=artifact, source_artifact=source,
                          provenance=provenance, **kw)

    report = verified(max_manifest_bytes=len(data), max_source_bytes=source.stat().st_size)
    assert report.manifest_checksum_verified and report.source_checksum_verified
    assert not report.native_execution_certified and not report.events["library_eligible"].any()
    external = audit(ledger, independent_manifest=table, provenance=provenance)
    assert not external.manifest_checksum_verified and not external.source_checksum_verified
    with pytest.raises(LedgerError, match="manifest byte bound"):
        verified(max_manifest_bytes=len(data)-1)
    with pytest.raises(LedgerError, match="source artifact byte bound"):
        verified(max_source_bytes=1)
    source.write_bytes(b"changed source")
    with pytest.raises(LedgerError, match="source artifact checksum"):
        verified()
    source.write_bytes(b"independent pre-response source")
    artifact.write_bytes(data + b" ")
    with pytest.raises(LedgerError, match="manifest artifact checksum"):
        verified()
    provenance["independent_manifest_sha256"] = hashlib.sha256(data + b" ").hexdigest()
    with pytest.raises(LedgerError, match="noncanonical"):
        verified()
    table["source_sha256"] = [SHA]
    changed = json.dumps(table, sort_keys=True, separators=(",", ":")).encode()
    artifact.write_bytes(changed)
    provenance["independent_manifest_sha256"] = hashlib.sha256(changed).hexdigest()
    with pytest.raises(LedgerError, match="manifest source SHA"):
        verified()


def test_status_enum_schema_must_be_explicit_and_versioned():
    with pytest.raises(LedgerError, match="p110 track_status_schema"):
        run(fixture(), provenance=dict(PROVENANCE, track_status_schema="legacy-5-postpone"))


def test_unknown_birth_first_observed_inside_is_occupancy_not_entry():
    rows = fixture(born=True)
    rows[1].update(type="track_start", membership=-1, visit=0, state=state(-1))
    unresolved = {k: deepcopy(rows[2][k]) for k in ("schema", "run", "subrun", "event", "sequence", "track", "parent_track", "particle", "sim_saved_at_start", "step", "visit")}
    unresolved.update(type="birth_membership_unresolved", track_status=0, step=1,
                      first_observed_pre=state(1, True))
    rows[-1]["unknown_birth_memberships"] = 1
    report = run(reseq(rows[:2] + [unresolved] + rows[2:]))
    assert report.events["no_entry_observed"][0]
    assert not report.events["no_domain_occupancy_observed"][0]
    assert report.events["born_inside"][0] == 0
    assert report.events["unsupported"][0]


def test_verified_wrapper_prefix_retains_verification_flags(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"source")
    sha = hashlib.sha256(b"source").hexdigest()
    table = {k: v.tolist() for k, v in manifest().items()}
    table["source_sha256"] = [sha]
    data = json.dumps(table, sort_keys=True, separators=(",", ":")).encode()
    artifact = tmp_path / "manifest"
    artifact.write_bytes(data)
    provenance = dict(PROVENANCE, source_sha256=sha,
                      independent_manifest_sha256=hashlib.sha256(data).hexdigest())
    with pytest.raises(LedgerError) as exc:
        audit_file(io.BytesIO(encoded(fixture()[:1]) + b"bad\n"),
                   independent_manifest_path=artifact, source_artifact=source,
                   provenance=provenance)
    assert exc.value.report.manifest_checksum_verified
    assert exc.value.report.source_checksum_verified
    assert not exc.value.report.protocol_complete
