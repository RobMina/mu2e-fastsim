"""Synthetic certificate, provenance, and orchestration gates; no real transport/data."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from crvfastsim.analyses import named_reference_campaign as N


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def inventory(events):
    return {"cycles": [1, 2], "selected_cycle": 2, "rows": events + 1, "unique_events": events}


@pytest.fixture
def certificates(tmp_path):
    tmp_path = tmp_path.resolve()
    paths = []
    for first, last in ((101, 101), (102, 104), (105, 120)):
        report = {"schema_version": 1, "status": "PASS", "run": 1400, "configs": list(N.CONFIGS),
                  "expected_events": N.EVENTS, "first": first, "last": last, "shards": []}
        for cfg in N.CONFIGS:
            for subrun in range(first, last + 1):
                accepted = 8000 + subrun
                artifacts = {}
                for name in N.ARTIFACTS:
                    suffix = f"001400_{subrun:08d}"
                    filename = f"{name}.{suffix}.root" if name in ("gen", "crvsteps") else f"{name}.{suffix}.log"
                    path = tmp_path / "inputs" / cfg / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"synthetic {cfg} {subrun} {name}\n")
                    artifacts[name] = N.C.fingerprint(path)
                artifacts["gen"]["trees"] = {"gen": inventory(N.EVENTS)}
                artifacts["crvsteps"]["trees"] = {k: inventory(accepted) for k in ("gen", "particles", "steps")}
                for name, total, passed in (("worker_log", N.EVENTS, accepted),
                                           ("dump_gen_log", N.EVENTS, N.EVENTS),
                                           ("dump_crvsteps_log", accepted, accepted)):
                    artifacts[name].update(exit_status=0, summary={"total": total, "passed": passed,
                                                                 "failed": total - passed})
                report["shards"].append({"config": cfg, "run": 1400, "subrun": subrun,
                    "counts": {"generated_events": N.EVENTS, "accepted_events": accepted,
                               "particle_events": accepted, "step_events": accepted,
                               "worker_passed": accepted, "worker_failed": N.EVENTS - accepted},
                    "artifacts": artifacts})
        paths.append(dump(tmp_path / f"certificate_{first}_{last}.json", report))
    return paths


def mutate(path, change):
    data = N.read_json(path)
    change(data)
    dump(path, data)


def test_exact_40_certificate_manifest(certificates):
    shards, evidence = N.validate_certificates(certificates)
    assert set(shards) == {(cfg, s) for cfg in N.CONFIGS for s in N.SUBRUNS}
    assert len(evidence) == 40 * 5 + 3
    for cfg in N.CONFIGS:
        assert sum(shards[cfg, s]["counts"]["generated_events"] for s in N.SUBRUNS) == 1000000


@pytest.mark.parametrize("change,message", [
    (lambda r: r.update(status="FAIL"), "require PASS"),
    (lambda r: r.update(schema_version=2), "require PASS"),
    (lambda r: r.update(expected_events=49999), "incompatible certificate"),
    (lambda r: r.update(last=121), "outside 101..120"),
    (lambda r: r["shards"].pop(), "incomplete certificate interval"),
    (lambda r: r["shards"].append(r["shards"][0]), "duplicate certified"),
    (lambda r: r["shards"][0].update(config="run1b"), "unexpected config"),
    (lambda r: r["shards"][0].update(run=1401), "unexpected config"),
    (lambda r: r["shards"][0]["counts"].update(generated_events=49999), "count disagreement"),
    (lambda r: r["shards"][0]["counts"].update(worker_passed=1), "count disagreement"),
    (lambda r: r["shards"][0]["counts"].update(particle_events=1), "count disagreement"),
    (lambda r: r["shards"][0]["counts"].update(step_events=0), "count disagreement"),
    (lambda r: r["shards"][0]["counts"].update(accepted_events=True), "noninteger"),
    (lambda r: r["shards"][0]["artifacts"].pop("dump_gen_log"), "incomplete certificate artifacts"),
    (lambda r: r["shards"][0]["artifacts"]["worker_log"].update(exit_status=1), "unsuccessful"),
    (lambda r: r["shards"][0]["artifacts"]["dump_crvsteps_log"]["summary"].update(total=1), "inconsistent"),
    (lambda r: r["shards"][0]["artifacts"]["gen"]["trees"]["gen"].update(selected_cycle=1), "highest-cycle"),
    (lambda r: r["shards"][0]["artifacts"]["crvsteps"]["trees"]["particles"].update(unique_events=1), "inventory event"),
])
def test_certificate_guards(certificates, change, message):
    mutate(certificates[0], change)
    with pytest.raises(ValueError, match=message):
        N.validate_certificates(certificates)


def test_missing_and_duplicate_certificates(certificates):
    with pytest.raises(ValueError, match="exactly subruns"):
        N.validate_certificates(certificates[:-1])
    with pytest.raises(ValueError, match="duplicate certified"):
        N.validate_certificates([*certificates, certificates[0]])


def test_changed_artifact_and_duplicate_source(certificates):
    data = N.read_json(certificates[1])
    artifact = data["shards"][0]["artifacts"]["gen"]
    Path(artifact["path"]).write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        N.validate_certificates(certificates)
    # Give a different subrun identical generator bytes and a matching digest: source gate still rejects it.
    other = data["shards"][1]["artifacts"]["gen"]
    Path(other["path"]).write_text("changed")
    for record in (artifact, other):
        record.update(N.C.fingerprint(record["path"]))
    dump(certificates[1], data)
    with pytest.raises(ValueError, match="duplicate generator sources"):
        N.validate_certificates(certificates)


def test_core_environment_requires_loaded_validation_core(tmp_path):
    good = tmp_path / "core-thread-a-validation"
    (good / "mu2efastsim").mkdir(parents=True)
    bad = tmp_path / "core"
    (bad / "mu2efastsim").mkdir(parents=True)
    N.check_core_environment({"root": str(good)}, str(good))
    with pytest.raises(ValueError, match="loaded core"):
        N.check_core_environment({"root": str(bad)}, str(good))
    with pytest.raises(ValueError, match="explicitly include"):
        N.check_core_environment({"root": str(good)}, "")
    with pytest.raises(ValueError, match="umbrella core"):
        N.check_core_environment({"root": str(good)}, f"{good}:{bad}")


def fake_snapshot():
    return {"code": {k: {"head": k + "-head", "source_sha256": {"transport.py": k},
                         "root": "/unused", "status": ""} for k in ("crv", "core")},
            "diagnostic_sha256": "diagnostic", "path_sources": {"path": "source"},
            "configs": {cfg: {"gdml": f"/geometry/{cfg}.gdml", "geometry": [
                {"path": "/counters", "bytes": 10, "sha256": "c" * 64},
                {"path": "/world", "bytes": 20, "sha256": "d" * 64}], "sectors": ["T1", "T2"],
                "groups": ["T1-T2"]} for cfg in N.CONFIGS}}


def fake_reports(shard, candidate, frozen):
    inputs = {}
    for arm, key in (("gen", "gen"), ("reference", "crvsteps")):
        inputs[arm] = deepcopy(shard["artifacts"][key])
        for inv in inputs[arm]["trees"].values():
            inv["unique_ids"] = inv.pop("unique_events")
    inputs["candidate"] = N.C.fingerprint(candidate)
    inputs["candidate"]["trees"] = {"gen": {"unique_ids": N.EVENTS}}
    common = {"run_config": N.CONFIGS[shard["config"]], "persisted_generated_events": N.EVENTS,
              "event_ranges": [{"run": 1400, "subrun": shard["subrun"], "min": 1, "max": N.EVENTS, "size": N.EVENTS}],
              "inputs": inputs, "geometry": frozen["configs"][shard["config"]]["geometry"], "crossing_key": N.P.KEY}
    audit = {**deepcopy(common), "expected_events": N.EVENTS,
             "expected_accepted": shard["counts"]["accepted_events"], "accepted_count_gate": True,
             "persisted_reference_accepted_events": shard["counts"]["accepted_events"],
             "code": frozen["code"], "diagnostic_sha256": frozen["diagnostic_sha256"],
             "p_bin_edges_MeV": [0, 300, 600, 1000, 2000, 5000, None], "artifacts": [],
             "sectors": {k: {"candidate_vs_reference": {"relative": .02, "se": .001}}
                         for k in ("T1", "T2", "group:T1-T2")}}
    paths = {**deepcopy(common), "schema": N.P.SCHEMA, "source_sha256": frozen["path_sources"],
             "settings": {"face_tolerance_mm": N.P.TOL_MM, "stop_p1_MeV": 1., "incidence_edges": N.P.INCIDENCE,
                          "reference_p_edges_MeV": audit["p_bin_edges_MeV"]},
             "source_ids": [inputs["gen"]["sha256"]],
             "sectors": {k: {} for k in ("T1", "T2", "group:T1-T2", "all_sectors")}}
    return deepcopy(audit), deepcopy(paths)


@pytest.fixture
def retained(certificates, tmp_path):
    shards, _ = N.validate_certificates(certificates)
    roots = [tmp_path.resolve() / k for k in ("candidates", "audits", "paths")]
    frozen = fake_snapshot()
    for cfg in N.CONFIGS:
        candidate = roots[0] / cfg / "tables" / "fastC.001400_00000101.root"
        candidate.parent.mkdir(parents=True)
        candidate.write_text(f"retained candidate {cfg}")
        log = roots[0] / cfg / "transport.log"
        log.write_text(f"retained transport log {cfg}")
        audit, paths = fake_reports(shards[cfg, 101], candidate, frozen)
        audit["artifacts"] = [N.C.fingerprint(log)]
        dump(roots[1] / cfg / "audit50k.json", audit)
        dump(roots[2] / cfg / "paths50k.json", paths)
    return shards, frozen, roots


def test_retained_hash_checks_and_no_rewrite(retained):
    shards, frozen, roots = retained
    files, evidence = N.retained_inputs("run1a", shards["run1a", 101], frozen, *roots)
    assert len(evidence) == 10
    Path(files["candidate"]).write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        N.retained_inputs("run1a", shards["run1a", 101], frozen, *roots)


@pytest.mark.parametrize("which,change,message", [
    ("audit", lambda r: r["code"]["core"].update(head="other"), "audit code"),
    ("audit", lambda r: r["code"]["crv"]["source_sha256"].update(x="new"), "audit code"),
    ("audit", lambda r: r.update(diagnostic_sha256="other"), "diagnostic changed"),
    ("audit", lambda r: r["geometry"][0].update(sha256="0" * 64), "geometry changed"),
    ("audit", lambda r: r.update(expected_accepted=1), "accepted-event"),
    ("audit", lambda r: r["inputs"]["gen"].update(sha256="0" * 64), "certificate hash"),
    ("audit", lambda r: r["inputs"]["reference"]["trees"]["particles"].update(unique_ids=1), "inventory mismatch"),
    ("audit", lambda r: r["event_ranges"][0].update(subrun=9111), "certified generated"),
    ("audit", lambda r: r["sectors"].pop("T2"), "all-sector"),
    ("paths", lambda r: r["source_sha256"].update(path="other"), "diagnostic/source"),
    ("paths", lambda r: r["settings"].update(stop_p1_MeV=2.), "settings changed"),
    ("paths", lambda r: r["inputs"]["candidate"].update(sha256="0" * 64), "candidate mismatch"),
])
def test_retained_provenance_guards(retained, which, change, message):
    shards, frozen, roots = retained
    path = roots[1 if which == "audit" else 2] / "run1a" / ("audit50k.json" if which == "audit" else "paths50k.json")
    mutate(path, change)
    with pytest.raises(ValueError, match=message):
        N.retained_inputs("run1a", shards["run1a", 101], frozen, *roots)


def test_freeze_signature_ignores_worktree_location_status_not_source():
    before, after = fake_snapshot(), fake_snapshot()
    after["code"]["core"].update(root="/different", status="?? new driver")
    N.assert_frozen(before, after)
    after["code"]["core"]["source_sha256"]["transport.py"] = "changed"
    with pytest.raises(ValueError, match="frozen code"):
        N.assert_frozen(before, after)


def test_merge_coverage_guards(retained):
    shards, frozen, roots = retained
    candidate = roots[0] / "run1a" / "tables" / "fastC.001400_00000101.root"
    reports = [fake_reports(shards["run1a", s], candidate, frozen)[0] for s in N.SUBRUNS]
    N.validate_shard_set(reports, "run1a")
    with pytest.raises(ValueError, match="exactly 20"):
        N.validate_shard_set(reports[:-1], "run1a")
    with pytest.raises(ValueError, match="disjoint"):
        N.validate_shard_set([reports[0], *reports[:-1]], "run1a")
    reports[1]["inputs"]["gen"]["sha256"] = reports[0]["inputs"]["gen"]["sha256"]
    with pytest.raises(ValueError, match="duplicate generator"):
        N.validate_shard_set(reports, "run1a")


@pytest.fixture
def mocked_driver(retained, certificates, monkeypatch, tmp_path):
    from crvfastsim.analyses import v1compare as B
    shards, frozen, roots = retained
    monkeypatch.setattr(N, "current_snapshot", lambda *args: deepcopy(frozen))
    calls = []

    def fake_transport(inputs, outdir, **kw):
        assert len(inputs) == 1
        subrun = int(Path(inputs[0]).stem.split("_")[-1])
        assert subrun != 101
        assert kw["seed"] == subrun - 100 and kw["stage"] == "C"
        calls.append((kw["run_config"], subrun, kw["seed"]))
        candidate = Path(outdir) / Path(inputs[0]).name.replace("gen.", "fastC.")
        candidate.write_text("candidate " + inputs[0])

    def reports(gen, candidate):
        cfg = Path(gen).parent.name
        subrun = int(Path(gen).stem.split("_")[-1])
        return fake_reports(shards[cfg, subrun], candidate, frozen)

    def audit_run(gen, ref, baseline, candidate, config, events, output, gdml, artifacts, expected_accepted):
        assert baseline is None and events == N.EVENTS
        audit, _ = reports(gen, candidate)
        assert audit["expected_accepted"] == expected_accepted
        audit["artifacts"] = [N.C.fingerprint(p) for p in artifacts]
        dump(Path(output), audit)
        return audit

    def path_run(gen, ref, candidate, config, events, output, gdml):
        _, paths = reports(gen, candidate)
        dump(Path(output), paths)
        return paths

    def merge_audit(inputs, output):
        rows = [N.read_json(p) for p in inputs]
        value = {"shards": len(rows), "persisted_generated_events": sum(r["persisted_generated_events"] for r in rows),
                 "expected_accepted": sum(r["expected_accepted"] for r in rows), "accepted_count_gate": True,
                 "persisted_reference_accepted_events": sum(r["expected_accepted"] for r in rows),
                 "sectors": rows[0]["sectors"]}
        dump(Path(output), value)
        return value

    def merge_paths(inputs, output):
        value = {"persisted_generated_events": sum(N.read_json(p)["persisted_generated_events"] for p in inputs)}
        dump(Path(output), value)
        return value

    def v1_run(inputs, outdir, compare=None, emit=None, merge=False, run_config=None):
        if emit:
            assert len(inputs) == len(compare) == 1
            return dump(Path(emit), {"source": inputs, "reference": compare})
        assert merge and len(inputs) == 20
        # Run1A deliberately fails: the second config must still complete.
        value = {"pass": run_config == "run2_v10", "events_union": 999000,
                 "groups": {"T1-T2": {"pass": run_config == "run2_v10", "len_mean_rel": .02}}}
        dump(Path(outdir) / "v1compare_stats.json", value)
        return value

    monkeypatch.setattr(N.transport, "run", fake_transport)
    monkeypatch.setattr(N.V, "run", audit_run)
    monkeypatch.setattr(N.P, "run", path_run)
    monkeypatch.setattr(N.V, "merge", merge_audit)
    monkeypatch.setattr(N.P, "merge", merge_paths)
    monkeypatch.setattr(B, "run", v1_run)
    args = {"certificates": certificates, "retained_candidates": roots[0], "retained_audits": roots[1],
            "retained_paths": roots[2], "geometry_dir": tmp_path / "geometry", "outdir": tmp_path.resolve() / "output"}
    return args, calls, frozen


def test_serial_complete_physics_fail_is_not_execution_failure(mocked_driver):
    args, calls, _ = mocked_driver
    summary = N.run_campaign(**args)
    assert summary["execution_complete"] is True and summary["v1_pass"] is False
    assert calls == [(rc, s, s - 100) for rc in N.CONFIGS.values() for s in range(102, 121)]
    for cfg, data in summary["configs"].items():
        assert data["generated_events"] == 1000000 and data["shards"] == 20
        assert data["accepted_events"] == sum(8000 + s for s in N.SUBRUNS)
        assert data["v1_status"] == ("FAIL" if cfg == "run1a" else "PASS")
        assert data["v1"]["events_union"] != data["generated_events"]
        reuse = N.read_json(args["outdir"] / cfg / "101" / "shard.json")
        assert reuse["transport_reused"] and not (args["outdir"] / cfg / "101" / "tables").exists()
    records = [json.loads(line) for line in (args["outdir"] / "progress.jsonl").read_text().splitlines()]
    assert any(r["stage"] == "transport" and r["state"] == "complete" and r["seconds"] >= 0 for r in records)
    saved = (args["outdir"] / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="existing output"):
        N.run_campaign(**args)
    assert (args["outdir"] / "summary.json").read_bytes() == saved


def test_single_config_still_requires_all_certificates(mocked_driver):
    args, calls, _ = mocked_driver
    N.run_campaign(**args, config="run2")
    assert calls == [("run2_v10", s, s - 100) for s in range(102, 121)]
    assert not (args["outdir"] / "run1a").exists()


def test_operational_failure_keeps_partial_output_and_is_nonzero(mocked_driver, monkeypatch):
    args, calls, _ = mocked_driver

    def fail(inputs, outdir, **kw):
        (Path(outdir) / "partial.root").write_text("partial")
        raise RuntimeError("synthetic transport failure")

    monkeypatch.setattr(N.transport, "run", fail)
    argv = ["--certificates", *map(str, args["certificates"])]
    for key, value in args.items():
        if key != "certificates":
            argv += ["--" + key.replace("_", "-"), str(value)]
    assert N.main(argv) == 1
    summary = N.read_json(args["outdir"] / "summary.json")
    assert summary["execution_complete"] is False and summary["v1_pass"] is None
    assert (args["outdir"] / "run1a" / "102" / "tables" / "partial.root").exists()
    assert not (args["outdir"] / "run2").exists()


def test_source_change_during_generation_stops(mocked_driver, monkeypatch):
    args, calls, frozen = mocked_driver
    original = N.transport.run
    current = deepcopy(frozen)
    monkeypatch.setattr(N, "current_snapshot", lambda *args: deepcopy(current))

    def changed(*args, **kwargs):
        original(*args, **kwargs)
        current["code"]["core"]["source_sha256"]["transport.py"] = "modified"

    monkeypatch.setattr(N.transport, "run", changed)
    with pytest.raises(ValueError, match="frozen code changed"):
        N.run_campaign(**args)
    assert len(calls) == 1
    assert N.read_json(args["outdir"] / "summary.json")["execution_complete"] is False


def test_preflight_failure_creates_no_output(mocked_driver):
    args, calls, _ = mocked_driver
    mutate(args["retained_audits"] / "run2" / "audit50k.json", lambda r: r.update(diagnostic_sha256="wrong"))
    with pytest.raises(ValueError, match="diagnostic changed"):
        N.run_campaign(**args)
    assert not args["outdir"].exists() and not calls


def test_missing_v1_result_is_operational_failure(mocked_driver, monkeypatch):
    from crvfastsim.analyses import v1compare as B
    args, calls, _ = mocked_driver
    original = B.run

    def missing(*pos, **kw):
        if kw.get("merge"):
            return {"pass": False}  # an exception/missing result is not a completed physics FAIL
        return original(*pos, **kw)

    monkeypatch.setattr(B, "run", missing)
    with pytest.raises(FileNotFoundError):
        N.run_campaign(**args)
    assert len(calls) == 19
    summary = N.read_json(args["outdir"] / "summary.json")
    assert not summary["execution_complete"] and summary["v1_pass"] is None


def test_final_input_mutation_is_caught(mocked_driver, monkeypatch):
    from crvfastsim.analyses import v1compare as B
    args, calls, _ = mocked_driver
    original = B.run

    def corrupt_after_merge(*pos, **kw):
        result = original(*pos, **kw)
        if kw.get("merge"):
            retained = args["retained_candidates"] / "run2" / "tables" / "fastC.001400_00000101.root"
            retained.write_text("modified after last V1 analysis")
        return result

    monkeypatch.setattr(B, "run", corrupt_after_merge)
    with pytest.raises(ValueError, match="artifact changed"):
        N.run_campaign(**args, config="run2")
    assert len(calls) == 19
    summary = N.read_json(args["outdir"] / "summary.json")
    assert not summary["execution_complete"] and summary["v1_pass"] is None


def test_conflicting_fingerprints_fail_without_coalescing(tmp_path):
    path = tmp_path.resolve() / "artifact"
    path.write_text("original")
    original = N.C.fingerprint(path)
    path.write_text("modified")
    with pytest.raises(ValueError, match="conflicting recorded"):
        N.verify_evidence([original, N.C.fingerprint(path)])


def test_no_overwrite_helper(tmp_path):
    path = tmp_path / "existing.json"
    N.write_json(path, {"first": True})
    with pytest.raises(FileExistsError):
        N.write_json(path, {"first": False})
    assert N.read_json(path) == {"first": True}
