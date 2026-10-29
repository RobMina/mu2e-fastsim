"""Saved-only retained-cache verification against real tiny generation and resealed mutations."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import retained_cache_report as R, retained_cache_audit as C
from test_veto_retained_cache_audit import mock_admission, fixture as cache_fixture

PLAN_SHA = "a" * 64


def reseal(directory):
    """Rebind edited payloads so semantic failures cannot be attributed to stale receipts."""
    receipt = R.Q._json(directory / C.A.COMPLETE)
    for name in receipt["artifacts"]:
        path = directory / name
        receipt["artifacts"][name] = {"bytes": path.stat().st_size, "sha256": C.sha256(path)}
    receipt["artifact_bytes"] = sum(a["bytes"] for a in receipt["artifacts"].values())
    receipt["incomplete_sha256"] = C.sha256(directory / C.A.INCOMPLETE)
    data = C.A._canonical(receipt)
    (directory / C.A.COMPLETE).write_bytes(data)
    (directory / C.A.PENDING).write_bytes(data)


def edit_json(root, name, change):
    path = root / name
    value = R.Q._json(path)
    change(value)
    path.write_text(json.dumps(value, allow_nan=False))


def edit_table(root, name, change):
    path = root / name
    table = pd.read_parquet(path)
    result = change(table)
    (table if result is None else result).to_parquet(path, index=False)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    record, registry, frames, batches = mock_admission(monkeypatch, tmp_path)
    root = tmp_path / "saved"
    C.run("unused", root, expected_plan_sha256=PLAN_SHA)
    def forbidden(*a, **k):
        pytest.fail("saved-only verifier invoked raw loading or generation")
    monkeypatch.setattr(C.I, "load_input", forbidden)
    monkeypatch.setattr(C.I.uproot, "open", forbidden)
    monkeypatch.setattr(C, "audit_input", forbidden)
    monkeypatch.setattr(C.M.K, "make_photons", forbidden)
    monkeypatch.setattr(R.Q.R, "run_arm", forbidden)
    monkeypatch.setattr(R.Q.R.W, "synthesize", forbidden)
    monkeypatch.setattr(R.Q.R.W.WF, "single_pe_template", forbidden)
    return root, tmp_path / "verified.json", record


def test_real_tiny_run_saved_only_exact_census_and_compact(evidence, monkeypatch):
    root, out, record = evidence
    calls = []
    admitted = C.admission
    def admission(plan, digest):
        calls.append((plan, digest))
        return admitted(plan, digest)
    monkeypatch.setattr(C, "admission", admission)
    report = R.verify(root, out, expected_plan_sha256=PLAN_SHA)
    assert calls == [("unused", PLAN_SHA)]
    assert json.loads(out.read_text()) == report
    assert {k: report[k] for k in ("frames", "batches", "steps", "finite_steps", "particles")} == {
        "frames": 4, "batches": 2, "steps": 5, "finite_steps": 4, "particles": 4}
    assert report["saved_artifact_protocol_exact"] and report["saved_frame_batch_census_exact"]
    assert report["source_admission_exact"] and report["particle_totals_manifest_exact"]
    assert report["coverage_flags_generation_bound"] and report["selection_hashes_generation_bound"]
    assert not report["raw_joins_independently_rederived"] and not report["source_coverage_independently_rederived"]
    assert not report["raw_payload_reloaded"] and not report["regenerated_optics"] and not report["regenerated_response"]
    assert not report["execution_admitted"] and report["F"] is None
    assert "not independently" in report["scope"] and "cumulative" in report["scope"]
    saved_batches = pd.read_parquet(root / "batches.parquet")
    assert report["max_selection_bytes"] == int(saved_batches.typed_payload_bytes.max())
    assert report["persisted_path_bytes"] == sum(p.stat().st_size for p in root.iterdir())
    small = R.compact(report)
    assert small["max_selection_bytes"] == report["max_selection_bytes"]
    assert small["inputs"][0]["coverage_bitmap_bytes"] == 13
    assert small["inputs"][0]["counts"] == {"events": 4, "steps": 5, "particles": 4}
    assert small["inputs"][0]["memory_released"]["peak_rss_bytes_cumulative"] >= small["inputs"][0]["memory_before"]["peak_rss_bytes_cumulative"]
    before = out.read_bytes()
    with pytest.raises(ValueError, match="new"):
        R.verify(root, out, expected_plan_sha256=PLAN_SHA)
    assert out.read_bytes() == before


SUMMARY_MUTATIONS = {
    "schema": lambda s: s.update(schema="unknown"),
    "extra_schema": lambda s: s.update(extra=0),
    "missing_schema": lambda s: s.pop("scope"),
    "scope": lambda s: s.update(scope="raw joins certified"),
    "frames": lambda s: s.update(frames=5),
    "float_count": lambda s: s.update(frames=4.),
    "bool_count": lambda s: s.update(batches=True),
    "steps": lambda s: s.update(steps=6),
    "particles": lambda s: s.update(particles=5),
    "coverage": lambda s: s.update(all_source_rows_once=False),
    "bool_coverage": lambda s: s.update(whole_frames_exact=1),
    "execution": lambda s: s.update(execution_admitted=True),
    "optics": lambda s: s.update(optics_executed=True),
    "response": lambda s: s.update(response_executed=True),
    "F": lambda s: s.update(F=0),
    "projection": lambda s: s.update(runtime_projection=1.),
    "uncertainty": lambda s: s.update(finite_library_uncertainty=0),
    "plan": lambda s: s.update(plan_sha256="b" * 64),
    "registry_hash": lambda s: s.update(registry_sha256="b" * 64),
    "limits": lambda s: s["limits"].update(chunk_rows=1),
    "cap": lambda s: s.update(output_cap=C.OUTPUT_CAP + 1),
    "record_count": lambda s: s["records"][0]["counts"].update(steps=6),
    "record_count_bool": lambda s: s["records"][0]["counts"].update(events=True),
    "record_digest": lambda s: s["records"][0].update(input_sha256="b" * 64),
    "record_manifest": lambda s: s["records"][0].update(manifest_sha256="b" * 64),
    "record_ledger": lambda s: s["records"][0].update(ledger_sha256="b" * 64),
    "record_domain": lambda s: s["records"].append(s["records"][0].copy()),
    "record_missing": lambda s: s.update(records=[]),
    "record_schema": lambda s: s["records"][0].update(extra=1),
    "record_batches": lambda s: s["records"][0].update(batches=1),
    "record_bitmap": lambda s: s["records"][0].update(coverage_bitmap_bytes=12),
    "record_bytes": lambda s: s["records"][0].update(resident_array_bytes=1),
    "record_coverage": lambda s: s["records"][0].update(all_source_rows_once=1),
    "record_negative_time": lambda s: s["records"][0].update(load_seconds=-1.),
    "record_bool_time": lambda s: s["records"][0].update(load_seconds=True),
    "record_elapsed": lambda s: s["records"][0].update(elapsed_seconds=0.),
    "record_audit_time": lambda s: s["records"][0].update(selection_audit_seconds=0.),
    "negative_time": lambda s: s.update(elapsed_seconds_before_commit=-1.),
    "short_time": lambda s: s.update(elapsed_seconds_before_commit=0.),
    "memory_schema": lambda s: s["memory_initial"].update(peak=0),
    "memory_negative": lambda s: s["memory_final"].update(current_rss_bytes=-1),
    "memory_float": lambda s: s["memory_initial"].update(peak_rss_bytes_cumulative=1.),
    "memory_decreased": lambda s: s["records"][0]["memory_loaded"].update(peak_rss_bytes_cumulative=0),
    "memory_final": lambda s: s["memory_final"].update(peak_rss_bytes_cumulative=0),
    "environment": lambda s: s["environment"].update(python=3),
    "missing_code": lambda s: s["code_sha256"].pop(str(C.code_paths()[0])),
    "bad_code": lambda s: s["code_sha256"].update({str(C.code_paths()[0]): "0" * 64}),
    "upper_hash": lambda s: s["code_sha256"].update({str(C.code_paths()[0]): "A" * 64}),
    "extra_input": lambda s: s["inputs"].update({str(C.code_paths()[0]): C.sha256(C.code_paths()[0])}),
}


@pytest.mark.parametrize("case", SUMMARY_MUTATIONS)
def test_resealed_summary_mutation_fails(evidence, case):
    root, out, _ = evidence
    edit_json(root, "summary.json", SUMMARY_MUTATIONS[case])
    reseal(root)
    with pytest.raises((ValueError, AssertionError)):
        R.verify(root, out, expected_plan_sha256=PLAN_SHA)
    assert not out.exists()


@pytest.mark.parametrize("filename,column,kind", [
    ("frames", "frame_id", "value"), ("frames", "input_sha256", "digest"),
    ("frames", "event", "value"), ("frames", "run", "value"), ("frames", "subrun", "value"),
    ("frames", "pbtime", "value"), ("frames", "intensity", "value"),
    ("frames", "event_source_row", "value"), ("frames", "steps", "value"),
    ("frames", "finite_steps", "value"), ("frames", "particles", "value"),
    ("frames", "batch", "value"), ("frames", "event_source_row", "float"),
    ("frames", "particles", "bool"), ("frames", "steps", "int32"),
    ("frames", "steps", "negative"), ("frames", "particles", "huge"), ("frames", "frame_id", "missing"),
    ("batches", "batch", "value"), ("batches", "frames", "value"),
    ("batches", "steps", "value"), ("batches", "particles", "value"),
    ("batches", "input_sha256", "digest"), ("batches", "typed_payload_bytes", "value"),
    ("batches", "frames", "float"), ("batches", "batch", "bool"),
    ("batches", "select_seconds", "negative"), ("batches", "validate_seconds", "nan"),
    ("batches", "validate_seconds", "inf"), ("batches", "select_seconds", "bool"),
    ("batches", "events_sha256", "uppercase"), ("batches", "steps_sha256", "short"),
    ("batches", "particles_sha256", "number"), ("batches", "typed_payload_bytes", "missing"),
    ("registry", "intensity", "value"), ("registry", "frame_id", "value"),
    ("registry", "event", "float"),
])
def test_resealed_table_mutation_fails(evidence, filename, column, kind):
    root, out, _ = evidence
    def change(t):
        if kind == "value":
            t.loc[0, column] += 1
        elif kind == "digest":
            t.loc[0, column] = "f" * 64
        elif kind in ("float", "bool", "int32"):
            t[column] = t[column].astype({"float": float, "bool": bool, "int32": np.int32}[kind])
        elif kind == "missing":
            return t.drop(columns=column)
        elif kind == "number":
            t[column] = 5
        else:
            t.loc[0, column] = {"negative": -1, "nan": np.nan, "inf": np.inf,
                                "uppercase": "A" * 64, "short": "a", "huge": 2**62}[kind]
    edit_table(root, filename + ".parquet", change)
    reseal(root)
    with pytest.raises((ValueError, AssertionError)):
        R.verify(root, out, expected_plan_sha256=PLAN_SHA)
    assert not out.exists()


@pytest.mark.parametrize("filename,operation", [(n, op) for n in ("registry", "frames", "batches")
                                               for op in ("drop", "duplicate", "reverse", "extra")])
def test_resealed_table_domains(evidence, filename, operation):
    root, out, _ = evidence
    def change(t):
        if operation == "drop":
            return t.iloc[1:]
        if operation == "duplicate":
            return pd.concat([t, t.iloc[:1]], ignore_index=True)
        if operation == "reverse":
            return t.iloc[::-1]
        return t.assign(extra=0)
    edit_table(root, filename + ".parquet", change)
    reseal(root)
    with pytest.raises((ValueError, AssertionError)):
        R.verify(root, out, expected_plan_sha256=PLAN_SHA)


@pytest.mark.parametrize("case", ["snapshot", "admission", "receipt", "pending", "incomplete", "missing_receipt",
                                  "extra_file", "failed", "payload", "symlink", "raw", "manifest", "ledger",
                                  "duplicate_json", "approved_loader", "duplicate_admitted_domain"])
def test_binding_protocol_and_approved_code_fail_closed(evidence, monkeypatch, case):
    root, out, record = evidence
    if case == "snapshot":
        (root / "code0.txt").write_bytes(b"changed snapshot")
        reseal(root)
    elif case == "admission":
        edit_json(root, "admission.json", lambda r: r.update(registry_batches_exact=False))
        reseal(root)
    elif case == "receipt":
        edit_json(root, C.A.COMPLETE, lambda r: r.update(execution_admitted=True))
    elif case == "pending":
        (root / C.A.PENDING).unlink()
        (root / C.A.PENDING).write_bytes(b"{}")
    elif case == "incomplete":
        edit_json(root, C.A.INCOMPLETE, lambda r: r.update(max_bytes=C.OUTPUT_CAP + 1))
        reseal(root)
    elif case == "missing_receipt":
        (root / C.A.COMPLETE).unlink()
    elif case in ("extra_file", "failed"):
        (root / ("extra.txt" if case == "extra_file" else "failed.json")).write_bytes(b"{}")
    elif case == "payload":
        (root / "code0.txt").write_bytes(b"unsealed")
    elif case == "symlink":
        target = root.parent / "external"
        (root / "code0.txt").rename(target)
        (root / "code0.txt").symlink_to(target)
    elif case in ("raw", "manifest", "ledger"):
        path = Path(record["path"] if case != "ledger" else record["ledger"])
        if case == "manifest":
            path = path.with_suffix(".json")
        path.write_bytes(b"changed source")
    elif case == "duplicate_json":
        path = root / "summary.json"
        text = path.read_text()
        path.write_text('{"frames":4,' + text[1:])
        reseal(root)
    elif case == "approved_loader":
        monkeypatch.setattr(C, "LOADER_SHA", "0" * 64)
    elif case == "duplicate_admitted_domain":
        original = C.admission
        def bad(*a):
            r, f, b, sources, bindings, verified = original(*a)
            return r, f, b, sources * 2, bindings, verified
        monkeypatch.setattr(C, "admission", bad)
        edit_json(root, "summary.json", lambda s: s["records"].append(s["records"][0].copy()))
        reseal(root)
    with pytest.raises((ValueError, AssertionError)):
        R.verify(root, out, expected_plan_sha256=PLAN_SHA)
    assert not out.exists()


@pytest.mark.parametrize("digest", ["b" * 64, "A" * 64, "short", None, 5])
def test_external_plan_required_and_bound(evidence, digest):
    root, out, _ = evidence
    with pytest.raises(ValueError):
        R.verify(root, out, expected_plan_sha256=digest)
    with pytest.raises(TypeError):
        R.verify(root, out)
    assert not out.exists()


def test_cli_requires_external_digest_and_emits_compact_json(evidence, capsys):
    root, out, _ = evidence
    capsys.readouterr()
    with pytest.raises(SystemExit):
        R.main([str(root), "--out", str(out)])
    assert not out.exists()
    R.main([str(root), "--out", str(out), "--expected-plan-sha256", PLAN_SHA, "--compact"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["frames"] == 4 and printed["plan_sha256"] == PLAN_SHA
    assert not printed["execution_admitted"]


def test_cannot_write_inside_transaction(evidence):
    root, _, _ = evidence
    with pytest.raises(ValueError, match="outside"):
        R.verify(root, root / "report.json", expected_plan_sha256=PLAN_SHA)
    assert not (root / "report.json").exists()


def test_scope_does_not_claim_unpersisted_selection_reconstruction(evidence):
    root, out, _ = evidence
    # These internally consistent alterations cannot be distinguished from the original
    # measurements without persisted raw selections/bitmaps; a receipt is not a signature.
    def frames(t):
        t.loc[0, "particles"] -= 1
        t.loc[1, "particles"] += 1
    edit_table(root, "frames.parquet", frames)
    edit_table(root, "batches.parquet", lambda t: t.assign(events_sha256="0" * 64))
    reseal(root)
    report = R.verify(root, out, expected_plan_sha256=PLAN_SHA)
    assert report["selection_hashes_generation_bound"]
    assert not report["raw_joins_independently_rederived"]
    assert not report["source_coverage_independently_rederived"]
    assert "particle allocations remain generation-bound" in report["scope"]


def test_parquet_metadata_admitted_before_saved_payload(evidence, monkeypatch):
    root, out, _ = evidence
    called = []
    original = C.parquet_admit
    def admit(path, rows=None):
        called.append((Path(path).name, rows))
        return original(path, rows)
    monkeypatch.setattr(C, "parquet_admit", admit)
    R.verify(root, out, expected_plan_sha256=PLAN_SHA)
    assert called == [("registry.parquet", 4), ("frames.parquet", 4), ("batches.parquet", 2)]


def test_two_inputs_have_independent_event_rows_and_cumulative_memory(tmp_path, monkeypatch):
    sources, assignments = [], []
    for name in ("one", "two"):
        path = tmp_path / name
        path.mkdir()
        record, _, frames, _, _, _, _ = cache_fixture(path)
        sources.append(record)
        assignments.append(frames)
    joined = pd.concat(assignments, ignore_index=True)
    registry = C.F.freeze(joined)
    index = pd.MultiIndex.from_frame(registry[list(C.F.FRAME_KEYS)])
    joined["frame_id"] = index.get_indexer(pd.MultiIndex.from_frame(joined[list(C.F.FRAME_KEYS)]))
    frames, batches = C.PR.P.batch_plan(joined.drop(columns="batch"), {**C.PR.P.DEFAULT_LIMITS, "frames": 2})
    sources.sort(key=lambda s: s["input_sha256"], reverse=True)
    monkeypatch.setattr(C, "admission", lambda *a: (registry, frames, batches, sources, {}, {"registry_batches_exact": True}))
    root = tmp_path / "saved"
    C.run("unused", root, expected_plan_sha256=PLAN_SHA)
    def forbidden(*a, **k):
        pytest.fail("raw or generation entrypoint invoked")
    for module, name in ((C.I, "load_input"), (C.I.uproot, "open"), (C.M.K, "make_photons"),
                         (R.Q.R.W, "synthesize"), (R.Q.R, "run_arm")):
        monkeypatch.setattr(module, name, forbidden)
    report = R.verify(root, tmp_path / "verified.json", expected_plan_sha256=PLAN_SHA)
    assert report["frames"] == 8 and report["batches"] == 4 and report["particles"] == 8
    assert len(report["records"]) == 2
    a, b = report["records"]
    assert a["memory_released"]["peak_rss_bytes_cumulative"] <= b["memory_before"]["peak_rss_bytes_cumulative"]
    assert all(r["coverage_bitmap_bytes"] == 13 for r in report["records"])


def test_plan_preserves_replay_uint32_identities(evidence, monkeypatch):
    root, out, _ = evidence
    original = C.admission
    def admitted(*args):
        registry, frames, batches, sources, bindings, verified = original(*args)
        frames = frames.astype({name: "uint32" for name in C.IDS})
        return registry, frames, batches, sources, bindings, verified
    monkeypatch.setattr(C, "admission", admitted)
    assert R.verify(root, out, expected_plan_sha256=PLAN_SHA)["saved_frame_batch_census_exact"]
