"""Bounded, serial 1M/config comparison of certified named-reference subruns 101..120.

Use ``python -m crvfastsim.analyses.named_reference_campaign --help``. No submission,
resume, physics tuning, or transport of retained subrun 101 is supported.
"""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from crvfastsim import geometry, runconfig, tables, transport as physics_transport
from crvfastsim.analyses import reference_certification as C
from crvfastsim.analyses import transport, transport_paths as P, transport_validation as V
from mu2efastsim import eloss

CONFIGS = {"run1a": "run1a_v01", "run2": "run2_v10"}
SUBRUNS = tuple(range(101, 121))
EVENTS = 50000
TOTAL = EVENTS * len(SUBRUNS)
ARTIFACTS = {"gen", "crvsteps", "worker_log", "dump_gen_log", "dump_crvsteps_log"}
SCOPE = ("Frozen-physics named-reference comparison only. V1 group results are not per-sector precision "
         "or response/reconstruction certification. Paired/path results are truth diagnostics, not physics gates.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    # Never replace an existing result, including one from an interrupted attempt.
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload)


def verify_fingerprint(record, path=None):
    """Verify bytes and digest; allow an explicitly supplied, byte-identical relocated input."""
    require(type(record.get("bytes")) is int and record["bytes"] > 0, "invalid artifact byte count")
    digest = record.get("sha256", "")
    require(isinstance(digest, str) and len(digest) == 64 and
            all(c in "0123456789abcdef" for c in digest), "invalid artifact SHA256")
    current = C.fingerprint(path if path is not None else record["path"])
    require(all(current[k] == record[k] for k in ("bytes", "sha256")),
            f"artifact changed: {current['path']}")
    return current


def check_inventory(tree, count):
    cycles = tree["cycles"]
    require(bool(cycles) and all(type(c) is int and c > 0 for c in cycles) and
            cycles == sorted(set(cycles)) and tree["selected_cycle"] == max(cycles),
            "invalid highest-cycle ROOT inventory")
    require(type(tree["rows"]) is int and tree["rows"] >= count > 0 and
            tree["unique_events"] == count, "ROOT inventory event count mismatch")


def validate_certificates(paths):
    """Require all 40 completed shards, even for a single-config execution.

    The existing certifier establishes ROOT identities. This workflow validates its
    schema/count evidence and rehashes every certified byte, without recertifying ROOT.
    Returns an indexed manifest plus immutable file evidence for subsequent checks.
    """
    shards, evidence = {}, []
    for path in paths:
        cert_fp = C.fingerprint(path)
        report = read_json(cert_fp["path"])
        require(report.get("schema_version") == 1 and report.get("status") == "PASS",
                "require PASS reference_certification schema 1")
        require(report["run"] == C.RUN and report["expected_events"] == EVENTS and
                report["configs"] == list(CONFIGS), "incompatible certificate campaign")
        first, last = report["first"], report["last"]
        require(type(first) is int and type(last) is int and 101 <= first <= last <= 120,
                "certificate interval outside 101..120")
        expected = {(cfg, s) for cfg in CONFIGS for s in range(first, last + 1)}
        found = set()
        for shard in report["shards"]:
            cfg, subrun = shard["config"], shard["subrun"]
            key = (cfg, subrun)
            require(type(subrun) is int and key in expected and shard["run"] == C.RUN,
                    "unexpected config/run/subrun")
            require(key not in shards and key not in found, f"duplicate certified shard: {key}")
            counts = shard["counts"]
            require(all(type(counts[k]) is int for k in ("generated_events", "accepted_events",
                    "particle_events", "step_events", "worker_passed", "worker_failed")), "noninteger counts")
            accepted = counts["accepted_events"]
            require(counts["generated_events"] == EVENTS and 0 < accepted <= EVENTS and
                    counts["particle_events"] == counts["worker_passed"] == accepted and
                    counts["worker_failed"] == EVENTS - accepted and
                    0 < counts["step_events"] <= accepted, "generated/accepted/worker count disagreement")
            artifacts = shard["artifacts"]
            require(set(artifacts) == ARTIFACTS, "incomplete certificate artifacts")
            for name, total, passed in (("worker_log", EVENTS, accepted),
                                        ("dump_gen_log", EVENTS, EVENTS),
                                        ("dump_crvsteps_log", accepted, accepted)):
                record = artifacts[name]
                require(record["exit_status"] == 0 and record["summary"] ==
                        {"total": total, "passed": passed, "failed": total - passed},
                        f"unsuccessful or inconsistent {name}")
            check_inventory(artifacts["gen"]["trees"]["gen"], EVENTS)
            for tree in ("gen", "particles", "steps"):
                check_inventory(artifacts["crvsteps"]["trees"][tree],
                                counts["step_events"] if tree == "steps" else accepted)
            suffix = f"{C.RUN:06d}_{subrun:08d}.root"
            for name in ("gen", "crvsteps"):
                require(Path(artifacts[name]["path"]).name == f"{name}.{suffix}",
                        "unexpected certified ROOT filename")
            for record in artifacts.values():
                evidence.append(verify_fingerprint(record))
            shards[key] = {**shard, "certificate": cert_fp}
            found.add(key)
        require(found == expected, "incomplete certificate interval")
        evidence.append(verify_fingerprint(cert_fp))
    require(set(shards) == {(cfg, s) for cfg in CONFIGS for s in SUBRUNS},
            "require exactly subruns 101..120 once per configuration (40 shards)")
    for cfg in CONFIGS:
        sources = [shards[cfg, s]["artifacts"]["gen"]["sha256"] for s in SUBRUNS]
        require(len(set(sources)) == len(SUBRUNS), f"duplicate generator sources for {cfg}")
    return shards, evidence


def check_core_environment(core, pythonpath=None):
    root = Path(core["root"]).resolve()
    require(root.name == "core-thread-a-validation", "loaded core must be core-thread-a-validation, not umbrella core")
    entries = [Path(p).resolve() for p in (os.environ.get("PYTHONPATH", "") if pythonpath is None
                                          else pythonpath).split(os.pathsep) if p]
    require(root in entries, "PYTHONPATH must explicitly include core-thread-a-validation")
    require(not any((p / "mu2efastsim").is_dir() and p != root for p in entries),
            "PYTHONPATH contains another/umbrella core")


def current_snapshot(configs, geometry_dir):
    # repo_state uses read-only git commands; disable git's optional index refresh writes.
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    os.environ["CRVFS_NO_FETCH"] = "1"
    code = {"crv": V.repo_state(runconfig), "core": V.repo_state(eloss)}
    check_core_environment(code["core"])
    modules = [V, geometry, runconfig, tables, physics_transport, physics_transport.crossings,
               eloss, physics_transport.core_tr, physics_transport.world_mod,
               physics_transport.materials, tables._core, P]
    for module in modules:
        if module.__name__.startswith("mu2efastsim."):
            require(Path(module.__file__).resolve().parent == Path(code["core"]["root"]) / "mu2efastsim",
                    "mixed imported core sources")
    from crvfastsim.analyses import v1compare
    out = {"code": code, "diagnostic_sha256": V.fingerprint(V.__file__)["sha256"],
           "path_sources": {m.__name__: V.fingerprint(m.__file__)["sha256"] for m in modules},
           "workflow_sha256": V.fingerprint(__file__)["sha256"],
           "v1_sha256": V.fingerprint(v1compare.__file__)["sha256"],
           "certifier_sha256": V.fingerprint(C.__file__)["sha256"], "configs": {}}
    for cfg in configs:
        rc = runconfig.get(CONFIGS[cfg])
        gdml = C._local_path(Path(geometry_dir) / Path(rc.gdml).name)
        counters = C._local_path(geometry.find_counters(rc.counters))
        geom = rc.geometry()
        groups = list(runconfig.sector_type_names(geom, rc.counters_cfg()).values())
        out["configs"][cfg] = {"gdml": str(gdml), "geometry": [C.fingerprint(counters), C.fingerprint(gdml)],
                                "sectors": list(geom.sector_names), "groups": groups}
    return out


def code_signature(code):
    # Worktree root/status can differ between hosts and because this driver is new.
    return {name: {k: code[name][k] for k in ("head", "source_sha256")} for name in ("crv", "core")}


def geometry_signature(records):
    return [(r["bytes"], r["sha256"]) for r in records]


def assert_frozen(before, after):
    require(code_signature(before["code"]) == code_signature(after["code"]), "frozen code changed")
    require({k: v for k, v in before.items() if k != "code"} ==
            {k: v for k, v in after.items() if k != "code"}, "frozen diagnostic/geometry/workflow changed")


def validate_shard_report(report, shard):
    require(report["run_config"] == CONFIGS[shard["config"]] and
            report["persisted_generated_events"] == EVENTS, "shard config/generated count mismatch")
    require(report["event_ranges"] == [{"run": C.RUN, "subrun": shard["subrun"],
                                        "min": 1, "max": EVENTS, "size": EVENTS}],
            "shard requires exactly the certified generated-event universe")
    require(set(report["inputs"]) == {"gen", "reference", "candidate"},
            "require candidate/reference-only comparison")
    for arm, artifact in (("gen", "gen"), ("reference", "crvsteps")):
        saved, certified = report["inputs"][arm], shard["artifacts"][artifact]
        require(geometry_signature([saved]) == geometry_signature([certified]),
                f"{arm}: audit/certificate hash mismatch")
        for tree, inv in certified["trees"].items():
            got = saved["trees"][tree]
            require(all(got[k] == inv[k] for k in ("cycles", "selected_cycle", "rows")) and
                    got["unique_ids"] == inv["unique_events"], f"{arm}: audit/certificate inventory mismatch")
    require(report["inputs"]["candidate"]["trees"]["gen"]["unique_ids"] == EVENTS,
            "candidate missing generated events")


def validate_provenance(audit, paths, shard, snapshot):
    cfg = shard["config"]
    expected = snapshot["configs"][cfg]
    for report in (audit, paths):
        validate_shard_report(report, shard)
        require(geometry_signature(report["geometry"]) == geometry_signature(expected["geometry"]),
                "retained geometry changed")
    require(audit["expected_events"] == EVENTS and audit["expected_accepted"] == shard["counts"]["accepted_events"]
            and audit["persisted_reference_accepted_events"] == shard["counts"]["accepted_events"]
            and audit["accepted_count_gate"] is True, "audit accepted-event evidence mismatch")
    require(code_signature(audit["code"]) == code_signature(snapshot["code"]), "retained audit code mismatch")
    require(audit["diagnostic_sha256"] == snapshot["diagnostic_sha256"], "retained audit diagnostic changed")
    require(paths["schema"] == P.SCHEMA and paths["source_sha256"] == snapshot["path_sources"],
            "retained path diagnostic/source changed")
    require(paths["settings"] == {"face_tolerance_mm": P.TOL_MM, "stop_p1_MeV": 1.,
            "incidence_edges": P.INCIDENCE, "reference_p_edges_MeV": [0, 300, 600, 1000, 2000, 5000, None]},
            "retained path settings changed")
    require(paths["source_ids"] == [shard["artifacts"]["gen"]["sha256"]], "path generator source mismatch")
    require(audit["p_bin_edges_MeV"] == paths["settings"]["reference_p_edges_MeV"] and
            audit["crossing_key"] == paths["crossing_key"] == P.KEY, "incompatible diagnostic keys/bins")
    sectors = set(expected["sectors"]) | {"group:" + g for g in expected["groups"]}
    require(set(audit["sectors"]) == sectors and set(paths["sectors"]) == sectors | {"all_sectors"},
            "incomplete all-sector/group diagnostics")
    require(all(set(value) == {"candidate_vs_reference"} for value in audit["sectors"].values()),
            "incompatible comparison sets")
    require(geometry_signature([audit["inputs"]["candidate"]]) ==
            geometry_signature([paths["inputs"]["candidate"]]), "audit/path candidate mismatch")


def retained_inputs(cfg, shard, snapshot, candidates, audits, paths):
    candidate = Path(candidates) / cfg / "tables" / "fastC.001400_00000101.root"
    audit_path, path_path = Path(audits) / cfg / "audit50k.json", Path(paths) / cfg / "paths50k.json"
    evidence = [C.fingerprint(audit_path), C.fingerprint(path_path)]
    audit, diagnostic = read_json(audit_path), read_json(path_path)
    validate_provenance(audit, diagnostic, shard, snapshot)
    for report in (audit, diagnostic):
        for arm, local in (("gen", shard["artifacts"]["gen"]["path"]),
                           ("reference", shard["artifacts"]["crvsteps"]["path"]), ("candidate", candidate)):
            evidence.append(verify_fingerprint(report["inputs"][arm], local))
    # Retain all original production-log evidence. Embedded artifact paths must be readable.
    for record in audit["artifacts"]:
        evidence.append(verify_fingerprint(record))
    transport_log = Path(candidates) / cfg / "transport.log"
    log_fp = C.fingerprint(transport_log)
    require(any(geometry_signature([r]) == geometry_signature([log_fp]) for r in audit["artifacts"]),
            "retained transport log absent from audit provenance")
    evidence.append(log_fp)
    for record in evidence:
        verify_fingerprint(record)
    return {"candidate": str(candidate), "audit": str(audit_path), "paths": str(path_path)}, evidence


def verify_evidence(records):
    """Check each distinct path once; conflicting recorded hashes fail rather than coalesce."""
    unique = {}
    for record in records:
        path = record["path"]
        if path in unique:
            require(geometry_signature([record]) == geometry_signature([unique[path]]),
                    f"conflicting recorded fingerprints: {path}")
        unique[path] = record
    for record in unique.values():
        verify_fingerprint(record)


def validate_shard_set(reports, cfg):
    require(len(reports) == len(SUBRUNS), "merge requires exactly 20 shards")
    identities = [tuple((r["run"], r["subrun"]) for r in report["event_ranges"]) for report in reports]
    require(set(identities) == {((C.RUN, s),) for s in SUBRUNS} and len(set(identities)) == len(SUBRUNS),
            "merge requires exact disjoint subruns 101..120")
    require(all(r["run_config"] == CONFIGS[cfg] and r["persisted_generated_events"] == EVENTS for r in reports)
            and sum(r["persisted_generated_events"] for r in reports) == TOTAL, "merge requires 1M/config")
    require(len({r["inputs"]["gen"]["sha256"] for r in reports}) == len(SUBRUNS), "duplicate generator sources")


class Progress:
    def __init__(self, stream):
        self.stream = stream

    def record(self, **fields):
        row = {"utc": datetime.now(timezone.utc).isoformat(), **fields}
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        print(json.dumps(row), flush=True)

    @contextmanager
    def stage(self, cfg, subrun, name, log):
        started = time.monotonic()
        self.record(config=cfg, subrun=subrun, stage=name, state="started")
        try:
            with Path(log).open("x") as stream, redirect_stdout(stream), redirect_stderr(stream):
                yield
        except BaseException as error:
            self.record(config=cfg, subrun=subrun, stage=name, state="failed",
                        seconds=time.monotonic() - started, error=repr(error))
            raise
        self.record(config=cfg, subrun=subrun, stage=name, state="complete", seconds=time.monotonic() - started)


def run_campaign(certificates, retained_candidates, retained_audits, retained_paths, geometry_dir, outdir, config="all"):
    require(config in (*CONFIGS, "all"), "unknown configuration")
    configs = list(CONFIGS) if config == "all" else [config]
    outdir = C._local_path(outdir)
    require(not outdir.exists(), f"refusing existing output directory: {outdir}")
    started = time.monotonic()
    # Establish imported-source identity before the potentially lengthy certificate hashing.
    frozen = current_snapshot(configs, geometry_dir)
    shards, evidence = validate_certificates(certificates)
    retained = {}
    for cfg in configs:
        retained[cfg], records = retained_inputs(cfg, shards[cfg, 101], frozen,
                                                retained_candidates, retained_audits, retained_paths)
        evidence.extend(records)
    assert_frozen(frozen, current_snapshot(configs, geometry_dir))
    # No output is created until all certification and retained-code gates have passed.
    outdir.mkdir()
    summary = {"schema": "named-reference-campaign-v1", "scope": SCOPE, "execution_complete": False,
               "v1_pass": None, "requested_configs": configs, "configs": {}}
    from crvfastsim.analyses import v1compare as B
    with (outdir / "progress.jsonl").open("x") as stream:
        progress = Progress(stream)
        try:
            write_json(outdir / "manifest.json", {"schema": summary["schema"], "scope": SCOPE,
                       "configs": configs, "subruns": list(SUBRUNS), "generated_events_per_shard": EVENTS,
                       "generated_events_per_config": TOTAL, "snapshot": frozen, "inputs": evidence,
                       "shards": [shards[cfg, s] for cfg in CONFIGS for s in SUBRUNS], "retained": retained,
                       "pythonpath": os.environ.get("PYTHONPATH"), "python": sys.version,
                       "seed_rule": "subrun - 100; one input per transport.run (file index zero)",
                       "preflight_seconds": time.monotonic() - started})
            for cfg in configs:
                base = outdir / cfg
                base.mkdir()
                audit_files, path_files, emits = [], [], []
                gdml = frozen["configs"][cfg]["gdml"]
                for subrun in SUBRUNS:
                    shard = shards[cfg, subrun]
                    dest = base / str(subrun)
                    dest.mkdir()
                    gen, ref = (shard["artifacts"][k]["path"] for k in ("gen", "crvsteps"))
                    with progress.stage(cfg, subrun, "input_check", dest / "input_check.log"):
                        verify_fingerprint(shard["certificate"])
                        for record in shard["artifacts"].values():
                            verify_fingerprint(record)
                    if subrun == 101:
                        candidate, audit_file, path_file = (retained[cfg][k] for k in ("candidate", "audit", "paths"))
                        progress.record(config=cfg, subrun=subrun, stage="reuse", state="complete",
                                        candidate=candidate, audit=audit_file, paths=path_file, transport_run=False)
                    else:
                        tabledir = dest / "tables"
                        tabledir.mkdir()
                        candidate = str(tabledir / f"fastC.001400_{subrun:08d}.root")
                        audit_file, path_file = str(dest / "audit50k.json"), str(dest / "paths50k.json")
                        log = dest / "transport.log"
                        with progress.stage(cfg, subrun, "transport", log):
                            before = current_snapshot(configs, geometry_dir)
                            assert_frozen(frozen, before)
                            write_json(dest / "source_before.json", before)
                            require(not Path(candidate).exists(), "candidate output already exists")
                            print(f"config={CONFIGS[cfg]} subrun={subrun} seed={subrun - 100} stage=C "
                                  f"gen={gen} gdml={gdml}", flush=True)
                            transport.run([gen], str(tabledir), seed=subrun - 100, stage="C",
                                          run_config=CONFIGS[cfg], gdml=gdml)
                            after = current_snapshot(configs, geometry_dir)
                            write_json(dest / "source_after.json", after)
                            assert_frozen(frozen, after)
                            for record in shard["artifacts"].values():
                                verify_fingerprint(record)
                            evidence.append(C.fingerprint(candidate))
                        with progress.stage(cfg, subrun, "audit", dest / "audit.log"):
                            V.run(gen, ref, None, candidate, CONFIGS[cfg], EVENTS, audit_file, gdml,
                                  artifacts=[shard["certificate"]["path"],
                                             *[shard["artifacts"][k]["path"] for k in
                                               ("worker_log", "dump_gen_log", "dump_crvsteps_log")], str(log)],
                                  expected_accepted=shard["counts"]["accepted_events"])
                        with progress.stage(cfg, subrun, "paths", dest / "paths.log"):
                            P.run(gen, ref, candidate, CONFIGS[cfg], EVENTS, path_file, gdml)
                    with progress.stage(cfg, subrun, "diagnostic_check", dest / "diagnostic_check.log"):
                        audit, diagnostic = read_json(audit_file), read_json(path_file)
                        validate_provenance(audit, diagnostic, shard, frozen)
                        for record in audit["artifacts"]:
                            evidence.append(verify_fingerprint(record))
                        for report in (audit, diagnostic):
                            for arm, local in (("gen", gen), ("reference", ref), ("candidate", candidate)):
                                evidence.append(verify_fingerprint(report["inputs"][arm], local))
                    emit = dest / "v1_emit.json"
                    with progress.stage(cfg, subrun, "v1_emit", dest / "v1_emit.log"):
                        require(not emit.exists(), "V1 emit already exists")
                        B.run([candidate], str(dest), compare=[ref], emit=str(emit), run_config=CONFIGS[cfg])
                        require(emit.is_file(), "V1 emit missing")
                    files = {k: C.fingerprint(p) for k, p in
                             (("candidate", candidate), ("audit", audit_file), ("paths", path_file), ("v1_emit", emit))}
                    evidence.extend(files.values())
                    write_json(dest / "shard.json", {"config": cfg, "subrun": subrun,
                               "transport_reused": subrun == 101, "seed": subrun - 100,
                               "generated_events": EVENTS, "accepted_events": shard["counts"]["accepted_events"],
                               "certificate": shard["certificate"], "files": files})
                    audit_files.append(audit_file)
                    path_files.append(path_file)
                    emits.append(str(emit))
                with progress.stage(cfg, None, "merge", base / "merge.log"):
                    assert_frozen(frozen, current_snapshot(configs, geometry_dir))
                    for records in (audit_files, path_files):
                        validate_shard_set([read_json(p) for p in records], cfg)
                    # Include emit/source evidence before merging; merges themselves enforce provenance equality.
                    verify_evidence(evidence)
                    merged = V.merge(audit_files, base / "audit1M.json")
                    path_merged = P.merge(path_files, base / "paths1M.json")
                    accepted = sum(shards[cfg, s]["counts"]["accepted_events"] for s in SUBRUNS)
                    require(merged["shards"] == 20 and merged["persisted_generated_events"] == TOTAL and
                            path_merged["persisted_generated_events"] == TOTAL and
                            merged["expected_accepted"] == merged["persisted_reference_accepted_events"] == accepted
                            and merged["accepted_count_gate"] is True, "merged counts/completeness mismatch")
                    v1dir = base / "v1"
                    v1dir.mkdir()
                    result = B.run(emits, str(v1dir), merge=True, run_config=CONFIGS[cfg])
                    stats = read_json(v1dir / "v1compare_stats.json")  # missing output is operational failure
                    require(type(stats.get("pass")) is bool and stats["pass"] == result["pass"] and
                            set(stats["groups"]) == set(frozen["configs"][cfg]["groups"]), "incomplete V1 result")
                    summary["configs"][cfg] = {"execution_complete": True, "shards": 20,
                        "generated_events": TOTAL, "accepted_events": accepted, "v1_pass": stats["pass"],
                        "v1_status": "PASS" if stats["pass"] else "FAIL", "v1": stats,
                        "all_sector_paired_diagnostics": merged["sectors"],
                        "audit": C.fingerprint(base / "audit1M.json"),
                        "paths": C.fingerprint(base / "paths1M.json"),
                        "v1_stats": C.fingerprint(v1dir / "v1compare_stats.json")}
                    evidence.extend(summary["configs"][cfg][k] for k in ("audit", "paths", "v1_stats"))
                    write_json(base / "summary.json", summary["configs"][cfg])
                progress.record(config=cfg, subrun=None, stage="config", state="complete", v1_pass=stats["pass"])
            with progress.stage(None, None, "final_check", outdir / "final_check.log"):
                assert_frozen(frozen, current_snapshot(configs, geometry_dir))
                verify_evidence(evidence)
                final = current_snapshot(configs, geometry_dir)
                write_json(outdir / "source_final.json", final)
                assert_frozen(frozen, final)
            summary["execution_complete"] = True
            summary["v1_pass"] = all(r["v1_pass"] for r in summary["configs"].values())
            summary["seconds"] = time.monotonic() - started
            write_json(outdir / "summary.json", summary)
        except BaseException as error:
            summary.update(execution_complete=False, v1_pass=None, error=repr(error), seconds=time.monotonic() - started)
            # Keep all partial artifacts; no cleanup, retry, overwrite, or resume.
            if not (outdir / "summary.json").exists():
                write_json(outdir / "summary.json", summary)
            raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", type=Path, nargs="+", required=True)
    for flag in ("retained-candidates", "retained-audits", "retained-paths", "geometry-dir", "outdir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--config", choices=[*CONFIGS, "all"], default="all")
    args = parser.parse_args(argv)
    try:
        result = run_campaign(**vars(args))
    except (Exception, SystemExit) as error:
        print(f"Named-reference campaign stopped: {error}", file=sys.stderr)
        return 1
    print(f"Execution complete; V1 {'PASS' if result['v1_pass'] else 'FAIL'}; {args.outdir / 'summary.json'}")
    # A complete physics FAIL remains a result, not an execution failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
