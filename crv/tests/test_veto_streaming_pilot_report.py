"""Real tiny saved pilots, no regeneration, and resealed adversarial evidence.

Admission is mocked at S.admission, not claimed to be ROOT-read-free: the real
frozen source verifier chain can still read raw/stock ROOT evidence.
"""
from copy import deepcopy
from dataclasses import replace
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import streaming_pilot_report as T
from test_veto_streaming_pilot import pilot_fixture
from test_veto_saved_qualified_report import dump, reseal, seal_child, edit_json

S, R, A = T.S, T.R, T.A


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    base = tmp_path_factory.mktemp("streaming-report")
    with pytest.MonkeyPatch.context() as patch:
        directory, admitted, resources, hashes, _ = pilot_fixture(base, patch, invoke=False)
        # Bind a synthetic template pathname while the fixture supplies its numerical samples.
        template = base / "template.txt"
        template.write_text("synthetic template; numerical samples supplied by fixture\n")
        template_sha = S.C.sha256(template)
        admitted["bindings"][str(template)] = template_sha
        resources["waveform_cfg"] = replace(resources["waveform_cfg"], single_pe_file=str(template))
        summary = S.run("plan", "cache", "response", directory, **hashes)
    return directory, admitted, hashes, summary, template_sha


@pytest.fixture
def saved(archive, tmp_path, monkeypatch):
    source, admitted, hashes, summary, template_sha = archive
    directory = tmp_path / "saved"
    # A concurrent source edit must fail setup, not make tamper tests pass.
    T.M.check_bindings(summary["inputs"], summary["code_sha256"])
    shutil.copytree(source, directory)
    d, calls = deepcopy(admitted), []

    def admission(*args, **kwargs):
        calls.append((args, kwargs))
        return d

    monkeypatch.setattr(S, "admission", admission)
    monkeypatch.setattr(R.D, "TEMPLATE_SHA", template_sha)
    original = R.Q.conditions
    monkeypatch.setattr(R.Q, "conditions", lambda r, s, c, h, nbars=1: original(r, s, c, h, nbars=1))

    def forbidden(*args, **kwargs):
        pytest.fail("saved report attempted cache loading or physics generation")

    for module, names in (
        (S, ("run", "resources")), (T.I, ("load_input",)), (T.O, ("make_batch",)),
        (T.K, ("make_photons",)), (T.M, ("generate",)), (T.V, ("run",)), (R, ("run_arm",)),
        (R.S, ("optical_candidates", "thermal_candidates", "simulate_candidates")),
        (R.S.S, ("optical_candidates", "thermal_candidates", "simulate_candidates")),
        (R.W, ("admit", "synthesize")), (R.W.W, ("synthesize", "reconstruct")),
        (R.P, ("reconstruct", "reconstruct_matrix", "fit_buckets")), (R.P.LM, ("fit_gumbel",)),
        (R.P.P, ("reconstruct", "reconstruct_full", "fit_gumbel")), (T.V.SI, ("simulate",)),
        (T.V.WF, ("single_pe_template", "synthesize")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    yield directory, tmp_path / "report.json", d, hashes.copy(), calls, summary
    for path, digest in summary["code_sha256"].items():
        assert S.C.sha256(path) == digest, "live code changed during adversarial test"


def verify(saved):
    directory, out, _, hashes, _, _ = saved
    return T.verify(directory, out, **hashes)


def rejected(saved, match=None):
    with pytest.raises((ValueError, AssertionError, FileNotFoundError), match=match):
        verify(saved)
    assert not saved[1].exists()


def child(saved):
    tag = saved[5]["records"][0]["path"]
    return saved[0] / tag


def change_table(path, transform):
    table = pd.read_parquet(path)
    changed = transform(table)
    (table if changed is None else changed).to_parquet(path, index=False)


def test_real_products_saved_only_report_and_compact(saved):
    result = verify(saved)
    directory, out, d, hashes, calls, summary = saved
    assert json.loads(out.read_text()) == result
    assert calls == [(("plan", "cache", "response"), hashes)]
    assert result["frames"] == 2 and result["batches"] == 1
    assert result["registry_frames"] == len(d["registry"])
    assert len(result["records"]) == 2 and len(result["frame_counts"]) == 8
    assert result["root_union"]["roots"] > 0
    assert result["persisted_path_bytes"] == sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    for key in ("saved_artifact_protocol_exact", "source_admission_exact", "typed_source_fingerprints_exact",
                "photon_timing_fields_exact", "authoritative_all_live_conditions_exact", "whole_selected_frames_exact",
                "all_declared_arm_seeds_complete"):
        assert result[key] is True
    for key in ("regenerated_optics", "regenerated_response", "execution_admitted", "full_retained_campaign_admitted", "timing_closure"):
        assert result[key] is False
    assert result["F"] is result["finite_library_uncertainty"] is None
    for record, generated in zip(result["records"], summary["records"]):
        assert record["counts"] == generated["counts"]
        assert record["source_counts"] == {"events": 2, "steps": 1, "particles": 1}
        for arm in S.ARMS:
            assert record["counts"][arm]["live_channels"] == 8
            assert record["stage_rows"][arm]["frames"] == 2
    assert any(t["pulses"] > 0 for t in result["stage_totals"].values())
    for rows in result["frame_counts"].values():
        quiet = next(row for row in rows if row["event"] == 10)
        assert quiet["conditions"] == 4  # all-live thermal support survives no optical steps
    for record in result["records"]:
        q = pd.read_parquet(directory / record["path"] / "photons.parquet")
        assert not q.event.eq(10).any()
    compact = T.compact(result)
    assert len(compact["counts"]) == 8 and compact["F"] is None
    assert compact["stage_totals"] == result["stage_totals"]
    assert "not an isolated peak or hard cap" in result["generation_measurements"]["interpretation"]


def test_cli_compact_still_writes_complete_report(saved, capsys):
    directory, out, _, hashes, _, _ = saved
    argv = [str(directory), "--out", str(out), "--compact"]
    for name, value in hashes.items():
        argv.extend(["--" + name.replace("_", "-"), value])
    T.main(argv)
    full = json.loads(out.read_text())
    assert json.loads(capsys.readouterr().out) == T.compact(full)
    assert "frame_counts" in full


@pytest.mark.parametrize("case", ["existing", "inside", "symlink"])
def test_report_output_is_exclusive_and_external(saved, case):
    directory, out, _, hashes, calls, _ = saved
    if case == "existing":
        out.write_text("keep")
    elif case == "inside":
        out = directory / "new.json"
    else:
        out.symlink_to(directory / "absent")
    with pytest.raises(ValueError):
        T.verify(directory, out, **hashes)
    assert not calls
    if case == "existing":
        assert out.read_text() == "keep"


@pytest.mark.parametrize("name", ["plan", "cache", "response"])
def test_external_hashes_required(saved, name):
    saved[3]["expected_" + name + "_sha256"] = "0" * 64
    rejected(saved, "external admission")
    assert not saved[4]


@pytest.mark.parametrize("case", ["failed", "missing_stage", "missing_complete", "missing_pending", "extra", "symlink", "hash", "pending", "context", "cap", "bytes", "execution"])
def test_child_protocol_tampering(saved, case):
    c = child(saved)
    if case == "failed":
        (c / A.FAILED).write_text("failed")
    elif case.startswith("missing"):
        (c / {"missing_stage": "historical_attempts.parquet", "missing_complete": A.COMPLETE, "missing_pending": A.PENDING}[case]).unlink()
    elif case == "extra":
        (c / "undeclared.txt").write_text("extra")
    elif case == "symlink":
        p = c / "events.parquet"
        target = saved[1].parent / "events.parquet"
        p.rename(target)
        p.symlink_to(target)
    elif case == "hash":
        with (c / "photons.parquet").open("ab") as stream:
            stream.write(b"changed")
    elif case == "pending":
        (c / A.PENDING).unlink()  # clone has independent receipt names
        (c / A.PENDING).write_text("{}")
    elif case in ("context", "cap"):
        def mutate(value):
            if case == "context":
                value["context"]["seeds"] = [7]
            else:
                value["max_bytes"] += 1
        edit_json(c / A.INCOMPLETE, mutate)
        seal_child(saved[0], c.name)
    else:
        def mutate(value):
            if case == "bytes":
                value["artifacts"]["events.parquet"]["bytes"] += 1
            else:
                value["execution_admitted"] = True
        edit_json(c / A.COMPLETE, mutate)
        dump(c / A.PENDING, T.SR._json(c / A.COMPLETE))
    rejected(saved)


@pytest.mark.parametrize("case", ["events_value", "steps_value", "particles_value", "source_row", "foreign_source", "bool_identity", "float_identity", "joined_pdg", "missing_quiet_frame", "duplicate_source"])
def test_resealed_typed_source_and_fingerprints(saved, case):
    c = child(saved)
    name = "events" if case in ("events_value", "missing_quiet_frame") else "particles" if case == "particles_value" else "steps"
    def mutate(t):
        if case == "events_value":
            t.loc[0, "intensity"] += 1
        elif case == "steps_value":
            t.loc[0, "t0"] += 1
        elif case == "particles_value":
            t.loc[0, "pdg"] = 13
        elif case == "source_row":
            t.loc[0, "source_row"] += 100
        elif case == "foreign_source":
            t.loc[0, "input_sha256"] = "f" * 64
        elif case in ("bool_identity", "float_identity"):
            t["frame_id"] = t.frame_id.astype(bool if case == "bool_identity" else float)
        elif case == "joined_pdg":
            t.loc[0, "pdg"] = 13
        elif case == "missing_quiet_frame":
            return t.iloc[:1]
        else:
            return pd.concat([t, t], ignore_index=True)
    change_table(c / (name + ".parquet"), mutate)
    seal_child(saved[0], c.name)
    rejected(saved)


@pytest.mark.parametrize("case", ["events", "steps", "particles", "typed_payload_bytes", "event_source_row"])
def test_completed_cache_evidence_is_independent_authority(saved, case):
    d = saved[2]
    if case == "event_source_row":
        d["cache_frames"].loc[d["cache_frames"].batch == 1, case] += 1
    elif case == "typed_payload_bytes":
        d["cache_batches"].loc[d["cache_batches"].batch == 1, case] += 1
    else:
        d["cache_batches"].loc[d["cache_batches"].batch == 1, case + "_sha256"] = "0" * 64
    rejected(saved)


@pytest.mark.parametrize("case", ["origin", "time_raw_wrapped_relative", "raw_folded_residual_ns", "historical_kept", "photon_id", "event_row", "source_row", "draw_seed", "photon_type", "bool_key", "float_key", "integer_mask", "nonfinite", "duplicate"])
def test_resealed_photon_timing_roots_and_strict_types(saved, case):
    c = child(saved)
    def mutate(q):
        assert len(q)
        if case == "historical_kept":
            q.loc[0, case] = not q.loc[0, case]
        elif case in ("bool_key", "float_key"):
            q["ordinal"] = q.ordinal.astype(bool if case == "bool_key" else float)
        elif case == "integer_mask":
            q["live_channel"] = q.live_channel.astype(np.int64)
        elif case == "nonfinite":
            q.loc[0, "time_raw_absolute"] = np.inf
        elif case == "duplicate":
            return pd.concat([q, q.iloc[:1]], ignore_index=True)
        elif case == "photon_type":
            q.loc[0, case] = 3
        else:
            q.loc[0, case] += 1
    change_table(c / "photons.parquet", mutate)
    seal_child(saved[0], c.name)
    rejected(saved)


@pytest.mark.parametrize("case", ["drop", "duplicate", "batch", "seed", "arm", "bool_count", "calibration", "selected", "registry"])
def test_resealed_top_coverage_and_authoritative_tables(saved, case):
    top = saved[0] / "run"
    name = case if case in ("calibration", "selected", "registry") else "coverage"
    def mutate(t):
        if case == "drop":
            return t.iloc[:-1]
        if case == "duplicate":
            return pd.concat([t, t.iloc[:1]], ignore_index=True)
        if case == "batch":
            t.loc[0, "batch"] += 1
        elif case == "seed":
            t.loc[0, "response_seed"] += np.uint64(1)
        elif case == "arm":
            t.loc[0, "arm"] = "unapproved"
        elif case == "bool_count":
            t["pulses"] = t.pulses.astype(bool)
        elif case == "calibration":
            t.loc[0, "pedestal"] += 1
        elif case in ("selected", "registry"):
            t.loc[0, "intensity"] += 1
    change_table(top / (name + ".parquet"), mutate)
    reseal(top)
    rejected(saved)


@pytest.mark.parametrize("stage,column,kind", [("conditions", "pedestal", "value"), ("conditions", "status", "bool"),
    ("phases", "phase", "value"), ("candidates", "time", "value"), ("attempts", "primary", "integer"),
    ("charges", "frame_id", "bool"), ("digis", "adc", "float"), ("frames", "pulses", "bool")])
def test_resealed_response_structure_and_conditions(saved, stage, column, kind):
    c = child(saved)
    def mutate(t):
        assert len(t)
        if kind == "value":
            t.loc[0, column] += 1
        else:
            t[column] = t[column].astype({"bool": bool, "integer": np.int64, "float": float}[kind])
    change_table(c / ("historical_" + stage + ".parquet"), mutate)
    seal_child(saved[0], c.name)
    rejected(saved)


@pytest.mark.parametrize("case", ["validation_receipt", "validation_counts", "validation_comparison", "copied_receipt", "admission", "record_counts", "record_order", "snapshot", "snapshot_order", "missing_code", "source_binding", "root_union", "memory_bool", "memory_decreased", "elapsed_bool"])
def test_resealed_receipts_bindings_snapshots_and_measurements(saved, case):
    c, top = child(saved), saved[0] / "run"
    if case.startswith("validation"):
        def mutate(value):
            if case == "validation_receipt":
                value["receipts"]["historical"]["attempts"] += 1
            elif case == "validation_counts":
                value["source_counts"]["steps"] += 1
            else:
                value["comparisons"] = {}
        edit_json(c / "validation.json", mutate)
        seal_child(saved[0], c.name)
    elif case in ("snapshot", "snapshot_order"):
        first, second = top / "code0.txt", top / "code1.txt"
        original = first.read_bytes()
        first.write_bytes(second.read_bytes() if case == "snapshot_order" else b"changed")
        if case == "snapshot_order":
            second.write_bytes(original)
    elif case == "copied_receipt":
        edit_json(top / (c.name + ".json"), lambda v: v.update(artifact_bytes=0))
    elif case == "admission":
        edit_json(top / "admission.json", lambda v: v.update(unapproved=True))
    else:
        def mutate(value):
            if case == "record_counts":
                value["records"][0]["counts"]["historical"]["attempts"] += 1
            elif case == "record_order":
                value["records"].reverse()
            elif case == "missing_code":
                value["code_sha256"].pop(str(S.code_paths()[0]))
            elif case == "source_binding":
                value["inputs"].pop(next(iter(value["inputs"])))
            elif case == "root_union":
                value["root_union"]["roots"] += 1
            elif case == "memory_bool":
                value["memory_before_execution"]["peak_rss_bytes_cumulative"] = True
            elif case == "memory_decreased":
                value["memory_after_execution"]["peak_rss_bytes_cumulative"] = 0
            else:
                value["records"][0]["elapsed_seconds"] = True
        edit_json(top / "summary.json", mutate)
    reseal(top)
    rejected(saved)


def test_logical_byte_cap_checked_before_reading_child_parquets(saved, monkeypatch):
    read_parquet = pd.read_parquet
    def read(path, *args, **kwargs):
        assert path.parent.name == "run", "child Parquet read before byte-cap admission"
        return read_parquet(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", read)
    original = T.SR._batch
    def limited(directory, context, names, cap):
        return original(directory, context, names, 1 if directory.name != "run" else cap)
    monkeypatch.setattr(T.SR, "_batch", limited)
    rejected(saved, "persisted byte cap exceeded")


def test_source_bindings_rechecked_after_saved_processing(saved, monkeypatch):
    original, calls = T.M.check_bindings, []
    def check(*args):
        calls.append(args)
        if len(calls) == 2:
            raise ValueError("source changed during saved verification")
        return original(*args)
    monkeypatch.setattr(T.M, "check_bindings", check)
    rejected(saved, "source changed during")
    assert len(calls) == 2


@pytest.mark.parametrize("case", ["changed_bytes", "hash_rebound", "admission_failure"])
def test_source_evidence_cannot_be_substituted_or_regenerated(saved, monkeypatch, case):
    directory, out, d, _, calls, _ = saved
    if case == "admission_failure":
        def fail(*args, **kwargs):
            raise ValueError("source admission failed")
        monkeypatch.setattr(S, "admission", fail)
        rejected(saved, "source admission failed")
        return
    # Use a per-test external copy, never mutate the shared real ROOT fixture.
    source = out.parent / "external.txt"
    source.write_text("bound external source")
    digest = S.C.sha256(source)
    top = directory / "run"
    if case == "changed_bytes":
        d["bindings"][str(source)] = digest
        edit_json(top / "summary.json", lambda v: v["inputs"].update({str(source): digest}))
        source.write_text("altered external source")
    else:
        # A newly self-consistent hash binding is not the independently admitted one.
        edit_json(top / "summary.json", lambda v: v["inputs"].update({str(source): digest}))
    reseal(top)
    rejected(saved, "changed bound input" if case == "changed_bytes" else "admitted source bindings")
    assert len(calls) == (0 if case == "changed_bytes" else 1)


@pytest.mark.parametrize("case", ["extra_summary", "missing_environment", "numeric_environment", "missing_version", "extra_record", "missing_record"])
def test_summary_environment_and_record_schema_are_mandatory(saved, case):
    def mutate(value):
        if case == "extra_summary":
            value["full_campaign_certified"] = True
        elif case == "missing_environment":
            value.pop("environment")
        elif case == "numeric_environment":
            value["environment"]["python"] = 3
        elif case == "missing_version":
            value["environment"].pop("scipy")
        elif case == "extra_record":
            value["records"][0]["independent_exposure"] = True
        else:
            value["records"][0].pop("elapsed_seconds")
    edit_json(saved[0] / "run/summary.json", mutate)
    reseal(saved[0] / "run")
    rejected(saved)


@pytest.mark.parametrize("target", ["child", "top"])
def test_late_saved_artifact_mutation_fails_final_transaction_recheck(saved, monkeypatch, target):
    original, calls = T.M.check_bindings, []
    def changed(*args):
        calls.append(1)
        original(*args)
        if len(calls) == 2:
            path = child(saved) / "historical_pulses.parquet" if target == "child" else saved[0] / "run/coverage.parquet"
            with path.open("ab") as stream:
                stream.write(b"changed after processing")
    monkeypatch.setattr(T.M, "check_bindings", changed)
    rejected(saved)
    assert len(calls) == 2


def test_all_child_parquet_metadata_precedes_materialization(saved, monkeypatch):
    original = T.S.C.parquet_admit
    def admit(path, *args):
        if path.parent.name != "run":
            raise ValueError("injected saved Parquet metadata cap")
        return original(path, *args)
    original_read = pd.read_parquet
    def read(path, *args, **kwargs):
        assert path.parent.name == "run", "child decoded before metadata checks"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(T.S.C, "parquet_admit", admit)
    monkeypatch.setattr(pd, "read_parquet", read)
    rejected(saved, "metadata cap")
