"""Campaign contracts use mocked subprocesses, never detector assets or accelerators."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "context_benchmark.py"
SPEC = importlib.util.spec_from_file_location("context_campaign", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    data, inputs, summary = (tmp_path / name for name in ("data", "inputs", "summary"))
    for path in (data, inputs, summary):
        path.mkdir()
    for name in ("lib_prod_as_20files.root", "crvcalib_sim_best_v1_5_run1400.csv",
                 "crvphoton_sim_best_v1_5_run1400.csv", "mu2e_extracted_v04.gdml"):
        (data / name).write_bytes(name.encode())
    for shard in (11, 12, 13):
        (inputs / f"gen.001400_{shard:08d}.root").write_bytes(str(shard).encode())
    args = M.parser().parse_args(["--data", str(data), "--inputs-dir", str(inputs),
                                 "--outdir", str(tmp_path / "details"), "--summary-dir", str(summary)])
    sources = {"/snapshot/crvfastsim/chain.py": "physics", "/snapshot/mu2efastsim/tables.py": "core"}
    monkeypatch.setattr(M, "source_snapshot", lambda: dict(sources))
    calls = []

    def execute(command, *, env, stdout, stderr, check):
        assert command[:3] == [sys.executable, "-m", "crvfastsim.analyses.e2e_bench"]
        assert stderr == subprocess.STDOUT and check is True
        calls.append((command[:], env.copy(), stdout.name))
        stdout.write("completed mock benchmark\n")
        positional = command[3:command.index("--library")]
        options = {}
        for key in ("library", "calib", "deviation", "gdml", "run_config", "window", "seed", "max_events", "repeat"):
            value = command[command.index("--" + key.replace("_", "-")) + 1]
            options[key] = int(value) if key in ("seed", "max_events", "repeat") else value
        options.update(inputs=positional, jax=True, prefetch="--prefetch" in command)
        prepared = "--reuse-context" in command
        files = [{"input": p, "input_index": i, "fingerprint": f"exact-{p}",
                  "counts": {"generated_events": 17 + i, "input_generated_events": 107 + i}}
                 for i, p in enumerate(positional)]
        report = {"status": "complete", "repeat_fingerprints_identical": True, "arguments": options,
                  "boundary": "cached chain", "device": {"response": "cuda:0"},
                  "context_policy": "prepared_once" if prepared else "reload_each_pass",
                  "context_audits_passed": prepared, "preparation_elapsed_s": 1.,
                  "invocation_elapsed_before_report_s": 20.,
                  "hardware_software": {"versions": {"jax": "test"}, "environment": env.copy()},
                  "provenance": {"source_sha256": dict(sources),
                                 "input_sha256": {p: M.digest(p) for p in positional},
                                 "assets_sha256": {options[k]: M.digest(options[k])
                                                   for k in ("library", "calib", "deviation", "gdml")},
                                 "effective_run_config": {"name": "extracted_v04"},
                                 "effective_chain_config": {"jax": True}},
                  "passes": [{"elapsed_s": 2. if prepared else 3., "files": copy.deepcopy(files),
                              "generated_events": sum(f["counts"]["generated_events"] for f in files)}
                             for _ in range(options["repeat"])]}
        out = Path(command[command.index("--outdir") + 1])
        out.mkdir()
        (out / "benchmark.json").write_text(json.dumps(report))
        (out / "private.npz").write_bytes(b"private detailed arrays")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(M.subprocess, "run", execute)
    return args, calls, execute, sources


def test_deterministic_campaign_exact_comparison_and_measured_counts(campaign, monkeypatch):
    args, calls, _, _ = campaign
    monkeypatch.setenv("CRVFASTSIM_COMPACT_DISPATCH", "unexpected inherited value")
    monkeypatch.setenv("PYTHONPATH", "ownedCRV:ownedCore")
    result = M.run(args)
    assert result["status"] == "complete" and len(calls) == 4
    for (name, shards, compact, seed, repeat, prepared, prefetch), (command, env, log) in zip(M.RUNS, calls):
        assert command[3:command.index("--library")] == [str(args.inputs_dir / f"gen.001400_{n:08d}.root") for n in shards]
        for flag, value in (("--seed", seed), ("--repeat", repeat), ("--max-events", 50000),
                            ("--run-config", "extracted_v04"), ("--window", "offspill")):
            assert command[command.index(flag) + 1] == str(value)
        assert ("--reuse-context" in command) is prepared
        assert ("--prefetch" in command) is prefetch
        assert "--jax" in command
        assert not any("tolerance" in arg or "rounding" in arg for arg in command)
        assert env["CRVFASTSIM_COMPACT_DISPATCH"] == str(compact)
        assert env["PYTHONPATH"] == "ownedCRV:ownedCore"
        assert env["CRVFS_DATA"] == env["CRVFS_LOOKUP_DIR"] == str(args.data)
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert log == str(args.summary_dir / (name + ".log"))
        assert (args.summary_dir / (name + ".json")).read_bytes() == (args.outdir / name / "benchmark.json").read_bytes()
    assert not list(args.summary_dir.glob("*.npz"))
    assert len(list(args.outdir.glob("*/private.npz"))) == 4
    heldout = result["runs"]["context_heldout_candidate"]["dataset"]
    assert heldout["generated_events_per_pass"] == 35
    assert heldout["input_generated_events"] == 215
    assert heldout["generated_event_executions"] == 140
    assert [f["generated_events"] for f in heldout["files"]] == [17, 18]
    for compact in (0, 1):
        assert result["comparisons"][f"n50000_compact{compact}"]["status"] == "deferred"
    comparison = json.loads((args.summary_dir / "context_heldout_comparison.json").read_text())
    assert comparison["exact_output_identity"] is True
    assert comparison["source_changes"] == []
    assert comparison["reference_context_policy"] == "reload_each_pass"
    assert comparison["candidate_context_policy"] == "prepared_once"
    assert comparison["reference_warm_elapsed_s"] == [3.]
    assert comparison["candidate_warm_elapsed_s"] == [2., 2., 2.]
    assert json.loads((args.summary_dir / "campaign.json").read_text()) == result


@pytest.mark.parametrize("fail_at", [1, 3, 4])
def test_subprocess_failure_stops_without_completed_campaign(campaign, monkeypatch, fail_at):
    args, calls, execute, _ = campaign

    def failing(command, **kwargs):
        if len(calls) == fail_at - 1:
            kwargs["stdout"].write("failure diagnostics\n")
            raise subprocess.CalledProcessError(9, command)
        return execute(command, **kwargs)

    monkeypatch.setattr(M.subprocess, "run", failing)
    with pytest.raises(subprocess.CalledProcessError):
        M.run(args)
    assert len(calls) == fail_at - 1
    assert not (args.summary_dir / "campaign.json").exists()
    assert not (args.summary_dir / (M.RUNS[fail_at - 1][0] + ".json")).exists()
    assert "failure" in (args.summary_dir / (M.RUNS[fail_at - 1][0] + ".log")).read_text()


@pytest.mark.parametrize("mutation", ["fingerprint", "source", "ordering", "seed", "incomplete", "counts", "audit"])
def test_invalid_reports_and_exact_mismatch_never_complete(campaign, monkeypatch, mutation):
    args, calls, execute, _ = campaign

    def altered(command, **kwargs):
        result = execute(command, **kwargs)
        if len(calls) == 4:
            path = args.outdir / "context_heldout_candidate" / "benchmark.json"
            report = json.loads(path.read_text())
            if mutation == "fingerprint":
                for item in report["passes"]:
                    item["files"][0]["fingerprint"] = "one ulp different"
            elif mutation == "source":
                report["provenance"]["source_sha256"]["/snapshot/crvfastsim/chain.py"] = "changed"
            elif mutation == "ordering":
                report["arguments"]["inputs"].reverse()
            elif mutation == "seed":
                report["arguments"]["seed"] += 1
            elif mutation == "incomplete":
                report["status"] = "partial"
            elif mutation == "counts":
                report["passes"][0]["generated_events"] += 1
            else:
                report["context_audits_passed"] = False
            path.write_text(json.dumps(report))
        return result

    monkeypatch.setattr(M.subprocess, "run", altered)
    with pytest.raises(ValueError):
        M.run(args)
    assert len(calls) == 4
    assert not (args.summary_dir / "context_heldout_comparison.json").exists()
    assert not (args.summary_dir / "campaign.json").exists()


@pytest.mark.parametrize("target", ["outdir", "dangling", "log", "report", "comparison", "campaign"])
def test_no_clobber_admission(campaign, target):
    args, calls, _, _ = campaign
    if target == "outdir":
        args.outdir.mkdir()
        path = args.outdir / "old-reference.json"
        path.write_text("preserve")
    elif target == "dangling":
        args.outdir.symlink_to(args.outdir.parent / "missing")
        path = args.outdir
    else:
        name = {"log": "n50000_compact0.log", "report": "context_heldout_reference.json",
                "comparison": "context_heldout_comparison.json", "campaign": "campaign.json"}[target]
        path = args.summary_dir / name
        path.write_text("preserve")
    with pytest.raises(FileExistsError):
        M.run(args)
    assert calls == []
    assert path.is_symlink() if target == "dangling" else path.read_text() == "preserve"


def test_existing_summary_required(campaign):
    args, calls, _, _ = campaign
    args.summary_dir = args.summary_dir / "missing"
    with pytest.raises(ValueError, match="existing directory"):
        M.run(args)
    assert calls == [] and not args.outdir.exists()


def test_library_fallback_and_explicit_assets(campaign):
    args, calls, _, _ = campaign
    nested = args.data / "tier1b_v7"
    nested.mkdir()
    (args.data / "lib_prod_as_20files.root").rename(nested / "lib_prod_as_20files.root")
    override = args.data / "custom.csv"
    override.write_text("override")
    args.calib = args.deviation = override
    args.gdml = args.data / "custom.gdml"
    args.gdml.write_text("geometry")
    M.run(args)
    for command, _, _ in calls:
        assert command[command.index("--library") + 1] == str(nested / "lib_prod_as_20files.root")
        assert command[command.index("--calib") + 1] == str(override)
        assert command[command.index("--deviation") + 1] == str(override)
        assert command[command.index("--gdml") + 1] == str(args.gdml)


def test_explicit_lookup_directory_preserved(campaign):
    args, calls, _, _ = campaign
    args.lookup_dir = args.data / "lookup_v6_0"
    args.lookup_dir.mkdir()
    M.run(args)
    assert all(env["CRVFS_LOOKUP_DIR"] == str(args.lookup_dir) for _, env, _ in calls)


def test_source_drift_stops_before_next_subprocess(campaign, monkeypatch):
    args, calls, execute, sources = campaign

    def drifting(command, **kwargs):
        result = execute(command, **kwargs)
        sources["/snapshot/crvfastsim/chain.py"] = "changed"
        return result

    monkeypatch.setattr(M.subprocess, "run", drifting)
    with pytest.raises(ValueError, match="source snapshot changed"):
        M.run(args)
    assert len(calls) == 1
    assert not (args.summary_dir / "n50000_compact0.json").exists()
    assert not (args.summary_dir / "campaign.json").exists()


@pytest.mark.parametrize("mismatch", [False, True])
def test_saved_reference_compared_without_shard11_reload(campaign, monkeypatch, mismatch):
    args, calls, execute, _ = campaign
    args.reference_dir = args.data / "references"
    args.reference_dir.mkdir()

    def with_reference(command, **kwargs):
        result = execute(command, **kwargs)
        if len(calls) == 1:
            report = json.loads((args.outdir / "n50000_compact0" / "benchmark.json").read_text())
            report["context_policy"] = "reload_each_pass"
            report["arguments"]["repeat"] = 2
            report["passes"] = report["passes"][:2]
            if mismatch:
                report["passes"][0]["files"][0]["fingerprint"] = "different"
            (args.reference_dir / "n50000_compact0.json").write_text(json.dumps(report))
        return result

    monkeypatch.setattr(M.subprocess, "run", with_reference)
    if mismatch:
        with pytest.raises(ValueError, match="no tolerance"):
            M.run(args)
        assert len(calls) == 1
        assert not (args.summary_dir / "campaign.json").exists()
    else:
        result = M.run(args)
        assert len(calls) == 4
        assert result["comparisons"]["n50000_compact0"]["status"] == "complete"
        assert result["comparisons"]["n50000_compact1"]["status"] == "deferred"
    assert all("--reuse-context" in c[0] for c in calls[:2])
