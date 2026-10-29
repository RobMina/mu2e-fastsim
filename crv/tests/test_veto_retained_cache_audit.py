"""Whole-plan cache coverage on small real ROOT inputs and adversarial selections."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import retained_cache_audit as C, indexed_replay as I, retained_plan as P
from test_veto_indexed_replay import artifact


def fixture(tmp_path):
    path, registry, args, tables = artifact(tmp_path)
    rows = registry[registry.input_sha256 == args["expected_input_sha256"]].copy()
    index = pd.MultiIndex.from_frame(rows[list(C.IDS)])
    sr = index.get_indexer(pd.MultiIndex.from_frame(tables["steps"][list(C.IDS)]))
    count = np.bincount(sr, minlength=len(rows))
    finite = np.isfinite(tables["steps"].t0) & np.isfinite(tables["steps"].t1)
    rows["steps"], rows["finite_steps"] = count, np.bincount(sr[finite], minlength=len(rows))
    rows["simulated_substeps"], rows["live_pairs"] = rows.finite_steps * 2, 4
    first, last = np.full(len(rows), len(sr), np.int64), np.full(len(rows), -1, np.int64)
    np.minimum.at(first, sr, np.arange(len(sr)))
    np.maximum.at(last, sr, np.arange(len(sr)))
    rows["source_first"], rows["source_stop"] = np.where(count, first, -1), np.where(count, last + 1, -1)
    frames, batches = P.batch_plan(rows, {**P.DEFAULT_LIMITS, "frames": 2})
    ledger = tables["steps"][C.LEDGER_COLUMNS[:-1]].assign(step_row=np.arange(len(sr)))
    lp = tmp_path / "ledger.parquet"
    ledger.to_parquet(lp, index=False)
    record = {"path": str(path), "input_sha256": args["expected_input_sha256"],
              "manifest_sha256": args["expected_manifest_sha256"], "ledger": str(lp), "ledger_sha256": C.sha256(lp)}
    return record, registry, frames, batches, ledger, tables, args


def selected(registry, frames, batch):
    return registry.iloc[frames.loc[frames.batch == batch["batch"], "frame_id"]]


def tracker(record, registry, frames, ledger, tables):
    return C.Coverage({n: len(t) for n, t in tables.items()}, registry, frames, ledger, record["input_sha256"])


def test_real_cache_walk_exact_all_rows_empty_frame_and_no_generation(tmp_path, monkeypatch):
    record, registry, frames, batches, ledger, tables, args = fixture(tmp_path)
    monkeypatch.setattr(C.M.K, "make_photons", lambda *a, **k: pytest.fail("optical redraw"))
    actual_f, actual_b, result = C.audit_input(record, registry, frames, batches)
    assert result["all_source_rows_once"] and result["counts"] == {n: len(t) for n, t in tables.items()}
    assert result["resident_array_bytes"] == I.resident_bytes(result["counts"])
    assert result["coverage_bitmap_bytes"] == 13
    assert result["batches"] == 2
    assert actual_f.event_source_row.tolist() == [0, 1, 2, 3]
    assert actual_f.steps.tolist() == [2, 2, 1, 0]
    assert actual_f.particles.tolist() == [2, 1, 1, 0]
    assert actual_f.finite_steps.tolist() == [1, 2, 1, 0]
    assert actual_b.steps.tolist() == [4, 1]
    assert actual_b.particles.tolist() == [3, 1]
    assert result["memory_loaded"]["peak_rss_bytes_cumulative"] > 0
    assert result["load_seconds"] > 0
    assert result["elapsed_seconds"] >= result["load_seconds"]


@pytest.mark.parametrize("case", ["drop_event", "drop_step", "drop_particle", "duplicate_step", "duplicate_particle",
                                 "reverse_steps", "bad_source", "fractional_source", "bad_frame", "wrong_input",
                                 "changed_phase", "changed_energy", "changed_pdg", "changed_particle_key",
                                 "bad_typed_bytes", "wrong_digest", "wrong_input_binding", "event_row_swap",
                                 "missing_column", "wrong_dtype"])
def test_corrupt_selection_fails_before_coverage_credit(tmp_path, case):
    record, registry, frames, batches, ledger, tables, args = fixture(tmp_path)
    c = tracker(record, registry, frames, ledger, tables)
    b = batches.iloc[0].to_dict()
    sel = selected(registry, frames, b)
    with I.load_input(record["path"], registry, **args) as cache:
        value = cache.select(sel, expected_registry_sha256=args["expected_registry_sha256"])
    if case.startswith("drop_"):
        name = {"event": "events", "step": "steps", "particle": "particles"}[case.removeprefix("drop_")]
        value = replace(value, **{name: getattr(value, name).iloc[1:]})
    elif case.startswith("duplicate_"):
        name = "steps" if case.endswith("step") else "particles"
        t = getattr(value, name)
        value = replace(value, **{name: pd.concat([t, t.iloc[:1]], ignore_index=True)})
    elif case == "reverse_steps":
        value = replace(value, steps=value.steps.iloc[::-1])
    elif case == "bad_source":
        value.steps.loc[0, "source_row"] = 999
    elif case == "fractional_source":
        value.steps["source_row"] = value.steps.source_row.astype(float) + .5
    elif case == "bad_frame":
        value.steps.loc[0, "frame_id"] = 0
    elif case == "wrong_input":
        value.steps.loc[0, "input_sha256"] = "0" * 64
    elif case == "changed_phase":
        value.events.loc[0, "pbtime"] += 1
    elif case == "changed_energy":
        value.steps.loc[0, "edep"] += 1
    elif case == "changed_pdg":
        value.steps.loc[0, "pdg"] += 1
    elif case == "changed_particle_key":
        value.particles.loc[0, "key"] = 999
    elif case == "bad_typed_bytes":
        value = replace(value, typed_payload_bytes=1)
    elif case == "wrong_digest":
        value = replace(value, registry_sha256="0" * 64)
    elif case == "wrong_input_binding":
        value = replace(value, input_sha256="0" * 64)
    elif case == "event_row_swap":
        value.events["source_row"] = value.events.source_row.to_numpy()[::-1]
    elif case == "missing_column":
        value = replace(value, particles=value.particles.drop(columns="parent"))
    elif case == "wrong_dtype":
        value.steps["bar"] = value.steps.bar.astype(float)
    with pytest.raises((ValueError, AssertionError)):
        c.consume(value, sel, b)
    assert not any(a.any() for a in c.seen.values())


@pytest.mark.parametrize("case", ["early_finish", "duplicate_batch", "missing_empty_frame", "wrong_plan_count", "wrong_finite", "ledger_reorder"])
def test_coverage_and_census_fail_closed(tmp_path, case):
    record, registry, frames, batches, ledger, tables, args = fixture(tmp_path)
    if case == "ledger_reorder":
        with pytest.raises(ValueError, match="ordering"):
            tracker(record, registry, frames, ledger.iloc[::-1], tables)
        return
    if case == "wrong_finite":
        frames.loc[0, "finite_steps"] += 1
    c = tracker(record, registry, frames, ledger, tables)
    with I.load_input(record["path"], registry, **args) as cache:
        b = batches.iloc[0].to_dict()
        sel = selected(registry, frames, b)
        value = cache.select(sel, expected_registry_sha256=args["expected_registry_sha256"])
        if case == "early_finish":
            with pytest.raises(ValueError, match="incomplete"):
                c.finish()
        elif case == "duplicate_batch":
            c.consume(value, sel, b)
            with pytest.raises(ValueError, match="duplicate"):
                c.consume(value, sel, b)
        elif case == "missing_empty_frame":
            c.consume(value, sel, b)
            with pytest.raises(ValueError, match="incomplete"):
                c.finish()
        else:
            if case == "wrong_plan_count":
                b["steps"] += 1
            with pytest.raises(ValueError, match="census"):
                c.consume(value, sel, b)


def test_hash_mutation_and_owned_output_tables(tmp_path):
    record, registry, frames, batches, ledger, tables, args = fixture(tmp_path)
    with I.load_input(record["path"], registry, **args) as cache:
        value = cache.select(registry[registry.input_sha256 == record["input_sha256"]],
                             expected_registry_sha256=args["expected_registry_sha256"])
    digest = C.fingerprint(value.steps, "steps")
    value.steps.loc[0, "x0"] += 1
    assert digest != C.fingerprint(value.steps, "steps")
    value.steps.loc[0, "x0"] -= 1
    assert digest == C.fingerprint(value.steps, "steps")
    value.steps.loc[0, "input_sha256"] = "f" * 64
    assert digest != C.fingerprint(value.steps, "steps")


def test_changed_ledger_rejected(tmp_path):
    record, registry, frames, batches, ledger, tables, args = fixture(tmp_path)
    Path(record["ledger"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        C.audit_input(record, registry, frames, batches)


def mock_admission(monkeypatch, tmp_path):
    record, full, frames, batches, ledger, tables, args = fixture(tmp_path)
    registry = C.F.freeze(full[full.input_sha256 == record["input_sha256"]])
    frames["frame_id"] -= 4
    batches["first_frame_id"] -= 4
    batches["last_frame_id"] -= 4
    monkeypatch.setattr(C, "admission", lambda *a: (registry, frames, batches, [record], {}, {"registry_batches_exact": True}))
    return record, registry, frames, batches


def test_transactional_run_snapshots_and_no_overwrite(tmp_path, monkeypatch):
    mock_admission(monkeypatch, tmp_path)
    out = tmp_path / "output"
    result = C.run("unused", out, expected_plan_sha256="a" * 64)
    assert result["frames"] == 4 and result["steps"] == 5 and result["particles"] == 4
    assert result["execution_admitted"] is False and result["F"] is None
    assert result["optics_executed"] is False and result["response_executed"] is False
    receipt = json.loads((out / "complete.json").read_text())
    assert len(receipt["artifacts"]) == 5 + len(C.code_paths())
    assert sum(p.stat().st_size for p in out.iterdir()) <= C.OUTPUT_CAP
    with pytest.raises(ValueError, match="new"):
        C.run("unused", out, expected_plan_sha256="a" * 64)


@pytest.mark.parametrize("case", ["audit", "write_cap", "changed_code", "final_binding"])
def test_failed_run_never_complete(tmp_path, monkeypatch, case):
    mock_admission(monkeypatch, tmp_path)
    out = tmp_path / "failed"
    def fail(*a, **k):
        raise ValueError("injected failure")
    if case == "audit":
        monkeypatch.setattr(C, "audit_input", fail)
    elif case == "write_cap":
        monkeypatch.setattr(C, "OUTPUT_CAP", 2000)
    elif case == "changed_code":
        monkeypatch.setattr(C, "LOADER_SHA", "0" * 64)
    else:
        monkeypatch.setattr(C.M, "check_bindings", fail)
    with pytest.raises((ValueError, OSError, C.A.ByteLimitExceeded)):
        C.run("unused", out, expected_plan_sha256="a" * 64)
    assert not (out / "complete.json").exists()
    if out.exists():
        assert (out / "failed.json").exists()


@pytest.mark.parametrize("name,column", [("steps", "x0"), ("particles", "pdg"), ("particles", "source_row"), ("events", "intensity")])
def test_snapshot_check_covers_non_census_fields_and_particle_source_slots(tmp_path, name, column):
    record, registry, frames, batches, ledger, tables, args = fixture(tmp_path)
    with I.load_input(record["path"], registry, **args) as cache:
        value = cache.select(registry[registry.input_sha256 == record["input_sha256"]],
                             expected_registry_sha256=args["expected_registry_sha256"])
        C.compare_snapshot(cache, value)
        table = getattr(value, name)
        if column == "source_row":
            table[column] = table[column].to_numpy()[::-1]
        else:
            table.loc[0, column] += 1
        with pytest.raises(ValueError, match="snapshot"):
            C.compare_snapshot(cache, value)


def test_one_cache_no_root_reopen_during_all_selections(tmp_path, monkeypatch):
    record, registry, frames, batches, ledger, tables, args = fixture(tmp_path)
    load = I.load_input
    calls = []
    def once(*a, **k):
        calls.append(a[0])
        cache = load(*a, **k)
        monkeypatch.setattr(I.uproot, "open", lambda *a, **k: pytest.fail("ROOT reopened after cache load"))
        return cache
    monkeypatch.setattr(I, "load_input", once)
    C.audit_input(record, registry, frames, batches)
    assert len(calls) == 1


def admission_fixture(tmp_path, monkeypatch):
    record, full, frames, batches, ledger, tables, args = fixture(tmp_path)
    registry = C.F.freeze(full[full.input_sha256 == record["input_sha256"]])
    frames["frame_id"] -= 4
    batches["first_frame_id"] -= 4
    batches["last_frame_id"] -= 4
    plan, smoke, census = (tmp_path / n for n in ("plan", "smoke", "census"))
    for p in (plan, smoke, census):
        p.mkdir()
    ledger_path = census / "steps.parquet"
    ledger.to_parquet(ledger_path, index=False)
    source = {"path": record["path"], "sha256": record["input_sha256"], "manifest_sha256": record["manifest_sha256"],
              "artifacts": {"steps": {"path": "steps.parquet", "sha256": C.sha256(ledger_path)}}}
    cp, sp = census / "summary.json", smoke / "summary.json"
    cp.write_text(json.dumps({"inputs": [source]}))
    sp.write_text(json.dumps({"paths": {"audit": str(census)}}))
    artifacts = {}
    for name, table in (("registry", registry), ("frames", frames), ("batches", batches)):
        p = plan / (name + ".parquet")
        table.to_parquet(p, index=False)
        artifacts[name] = {"path": p.name, "sha256": C.sha256(p), "rows": len(table)}
    report = {"artifacts": artifacts, "smoke": str(smoke), "code_sha256": {},
              "inputs": {str(sp): C.sha256(sp), str(ledger_path): C.sha256(ledger_path)},
              "admission_bindings": {"assets": {}, "inputs": {str(cp): C.sha256(cp)}, "code": {}, "blocks": []}}
    path = plan / "summary.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(C.PR, "verify", lambda *a: {"registry_sha256": C.F.digest(registry)})
    monkeypatch.setattr(I, "load_input", lambda *a, **k: pytest.fail("cache load during saved admission"))
    return path, report, cp, source


def test_unmocked_admission_binds_original_census(tmp_path, monkeypatch):
    path, report, cp, source = admission_fixture(tmp_path, monkeypatch)
    registry, frames, batches, records, bindings, verified = C.admission(path.parent, C.sha256(path))
    assert len(registry) == 4 and len(batches) == 2
    assert len(records) == 1 and records[0]["input_sha256"] == source["sha256"]
    assert bindings[str(path)] == C.sha256(path)
    assert bindings[str(cp)] == C.sha256(cp)


@pytest.mark.parametrize("case", ["plan_hash", "ledger_binding", "duplicate_input", "missing_input", "unsafe_table", "table_count", "conflicting_binding"])
def test_admission_rejects_corrupt_bindings_without_cache(tmp_path, monkeypatch, case):
    path, report, cp, source = admission_fixture(tmp_path, monkeypatch)
    if case == "ledger_binding":
        report["inputs"].pop(str(cp.parent / "steps.parquet"))
    elif case in ("duplicate_input", "missing_input"):
        cp.write_text(json.dumps({"inputs": [source, source] if case == "duplicate_input" else []}))
        report["admission_bindings"]["inputs"][str(cp)] = C.sha256(cp)
    elif case == "unsafe_table":
        report["artifacts"]["registry"]["path"] = "../registry.parquet"
    elif case == "table_count":
        report["artifacts"]["registry"]["rows"] += 1
    elif case == "conflicting_binding":
        report["code_sha256"] = {str(cp): "0" * 64}
    path.write_text(json.dumps(report))
    digest = "0" * 64 if case == "plan_hash" else C.sha256(path)
    with pytest.raises(ValueError):
        C.admission(path.parent, digest)


def test_registry_digest_not_recomputed_per_batch(tmp_path, monkeypatch):
    record, registry, frames, batches, ledger, tables, args = fixture(tmp_path)
    digest = C.F.digest
    calls = []
    def observed(value):
        calls.append(1)
        return digest(value)
    monkeypatch.setattr(C.F, "digest", observed)
    C.audit_input(record, registry, frames, batches)
    assert len(calls) == 3


def test_binding_merge_has_no_io_and_rejects_conflicts(monkeypatch):
    monkeypatch.setattr(C, "sha256", lambda *a: pytest.fail("hash during merge"))
    assert C.merge_bindings({"p": "a" * 64}, {"p": "a" * 64, "q": "b" * 64}) == {"p": "a" * 64, "q": "b" * 64}
    with pytest.raises(ValueError, match="conflicting"):
        C.merge_bindings({"p": "a" * 64}, {"p": "b" * 64})


def test_wrong_ledger_size_rejected_before_payload_or_cache(tmp_path, monkeypatch):
    record, registry, frames, batches, ledger, tables, args = fixture(tmp_path)
    pd.concat([ledger, ledger]).to_parquet(record["ledger"], index=False)
    record["ledger_sha256"] = C.sha256(record["ledger"])
    monkeypatch.setattr(C.pd, "read_parquet", lambda *a, **k: pytest.fail("Parquet materialized"))
    monkeypatch.setattr(I, "load_input", lambda *a, **k: pytest.fail("cache loaded"))
    with pytest.raises(ValueError, match="Parquet"):
        C.audit_input(record, registry, frames, batches)


@pytest.mark.parametrize("case", ["file_bytes", "rows", "columns", "groups", "group_bytes", "total_bytes"])
def test_saved_parquet_metadata_caps(tmp_path, monkeypatch, case):
    from types import SimpleNamespace
    path = tmp_path / "fake.parquet"
    path.touch()
    if case == "file_bytes":
        with path.open("wb") as stream:
            stream.truncate(128 * 1024**2 + 1)
    count = 3 if case == "total_bytes" else 1
    sizes = 200 * 1024**2 if case == "total_bytes" else (256 * 1024**2 + 1 if case == "group_bytes" else 1)
    meta = SimpleNamespace(num_rows=2_000_001 if case == "rows" else 1,
                           num_columns=65 if case == "columns" else 1,
                           num_row_groups=4097 if case == "groups" else count,
                           row_group=lambda i: SimpleNamespace(total_byte_size=sizes))
    class Fake:
        metadata = meta
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            pass
    monkeypatch.setattr(C.pq, "ParquetFile", lambda *a: Fake())
    with pytest.raises(ValueError, match="Parquet"):
        C.parquet_admit(path)


def test_plan_artifact_preflight_precedes_frozen_verifier(tmp_path, monkeypatch):
    path, report, cp, source = admission_fixture(tmp_path, monkeypatch)
    report["artifacts"]["frames"]["rows"] += 1
    path.write_text(json.dumps(report))
    monkeypatch.setattr(C.PR, "verify", lambda *a: pytest.fail("verifier before metadata admission"))
    with pytest.raises(ValueError, match="Parquet"):
        C.admission(path.parent, C.sha256(path))
