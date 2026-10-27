"""Synthetic persisted-table and log gates for named-reference certification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.analyses import reference_certification as C


N = 5
FIRST = 102


def _gen(subrun):
    events = np.array([1, 1, 2, 3, 4, 5], dtype=np.uint32)
    frame = pd.DataFrame({"run": np.full(6, 1400, dtype=np.uint32),
                          "subrun": np.full(6, subrun, dtype=np.uint32), "event": events,
                          "idx": np.array([0, 1, 0, 0, 0, 0], dtype=np.int32),
                          "pdg": np.array([13, -13, 13, 13, 13, 13], dtype=np.int32),
                          "genid": np.full(6, 42, dtype=np.int32)})
    for column in C.VALUES[1:]:
        frame[column] = events.astype(np.float64) + frame.idx.to_numpy() / 10
    return frame


def _write_tree(path, name, frame, append=False):
    opener = uproot.update if append else uproot.recreate
    with opener(path) as root:
        tree = root.mktree("dump/" + name, {column: frame[column].dtype for column in frame})
        tree.extend({column: frame[column].to_numpy() for column in frame})


def _summary(total, passed, failed):
    return f"TrigReport Events total = {total} passed = {passed} failed = {failed}\n"


def _path(base, kind, config="run1a", subrun=FIRST):
    extension = "root" if kind in ("gen", "crvsteps") else "log"
    return base / config / f"{kind}.001400_{subrun:08d}.{extension}"


@pytest.fixture
def campaign(tmp_path):
    base = tmp_path / "dumps"
    for config in C.CONFIGS:
        (base / config).mkdir(parents=True)
        for subrun in (FIRST, FIRST + 1):
            gen = _gen(subrun)
            accepted = gen.loc[gen.event.isin([1, 3])].iloc[::-1]
            particles = accepted[C.KEY].iloc[[0, 1, 1, 2]].copy()
            particles["key"] = np.arange(len(particles), dtype=np.int32)
            steps = accepted.loc[accepted.event == 1, C.KEY].copy()
            steps["bar"] = np.zeros(len(steps), dtype=np.int32)
            _write_tree(_path(base, "gen", config, subrun), "gen", gen)
            root = _path(base, "crvsteps", config, subrun)
            _write_tree(root, "gen", accepted)
            _write_tree(root, "particles", particles, append=True)
            _write_tree(root, "steps", steps, append=True)
            _path(base, "worker", config, subrun).write_text(
                "worker starts\n" + _summary(N, 2, N - 2) + "timing\nExit status: 0\ncopy-out complete\n")
            _path(base, "dump_gen", config, subrun).write_text(
                "dumper starts\n" + _summary(N, N, 0) + "timing\ndump_exit=0\n")
            _path(base, "dump_crvsteps", config, subrun).write_text(
                _summary(2, 2, 0) + "dump_exit=0\n")
    return base


def _args(base, output, first=FIRST, last=FIRST + 1, expected=N):
    return [str(base), "--first", str(first), "--last", str(last),
            "--expected-events", str(expected), "--output", str(output)]


def _fails(base, output, capsys, message):
    assert C.main(_args(base, output)) == 1
    captured = capsys.readouterr()
    assert message in captured.err
    assert "PASS" not in captured.out
    assert not output.exists()


def _frame(path, name):
    with uproot.open(path) as root:
        return root["dump/" + name].arrays(library="pd")


def test_success_counts_hashes_cycles_and_values(campaign, tmp_path, capsys):
    output = tmp_path / "report.json"
    assert C.main(_args(campaign, output)) == 0
    assert "PASS" in capsys.readouterr().out
    report = json.loads(output.read_text())
    assert report["status"] == "PASS"
    assert report["configs"] == ["run1a", "run2"]
    assert report["first"] == FIRST and report["last"] == FIRST + 1
    assert len(report["shards"]) == 4
    for shard in report["shards"]:
        assert shard["run"] == 1400
        assert shard["counts"] == {"generated_events": N, "accepted_events": 2, "particle_events": 2,
                                   "step_events": 1, "worker_passed": 2, "worker_failed": N - 2}
        artifacts = shard["artifacts"]
        assert len(artifacts) == 5
        for artifact in artifacts.values():
            content = Path(artifact["path"]).read_bytes()
            assert artifact["sha256"] == hashlib.sha256(content).hexdigest()
            assert artifact["bytes"] == len(content)
        assert artifacts["gen"]["trees"]["gen"] == {
            "cycles": [1], "selected_cycle": 1, "rows": 6, "unique_events": N}
        assert artifacts["crvsteps"]["trees"]["gen"]["rows"] == 3
        assert artifacts["crvsteps"]["trees"]["particles"]["rows"] == 4
        assert artifacts["crvsteps"]["trees"]["steps"]["rows"] == 2


def test_module_cli_works_without_transport_imports(campaign, tmp_path):
    output = tmp_path / "module.json"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-m", "crvfastsim.analyses.reference_certification",
                             *_args(campaign, output)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout and output.is_file()
    probe = subprocess.run([sys.executable, "-B", "-c",
                            "import sys; from crvfastsim.analyses import reference_certification; "
                            "assert 'mu2efastsim' not in sys.modules; "
                            "assert 'crvfastsim.analyses.transport_validation' not in sys.modules"],
                           capture_output=True, text=True, env=env)
    assert probe.returncode == 0, probe.stderr


@pytest.mark.parametrize("kind,name", [("gen", "gen"), ("crvsteps", "gen"),
                                       ("crvsteps", "particles"), ("crvsteps", "steps")])
def test_higher_partial_cycle_never_falls_back(campaign, tmp_path, capsys, kind, name):
    path = _path(campaign, kind)
    frame = _frame(path, name)
    partial = frame.iloc[:0] if name == "steps" else frame.loc[frame.event != 1]
    _write_tree(path, name, partial, append=True)
    _fails(campaign, tmp_path / "report.json", capsys, "Reference certification failed")


def test_complete_highest_cycle_selected(campaign, tmp_path):
    path = _path(campaign, "gen")
    good = _frame(path, "gen")
    _write_tree(path, "gen", good.iloc[:1])
    _write_tree(path, "gen", good, append=True)
    out = C.certify(campaign, FIRST, FIRST, N, tmp_path / "report.json")
    inventory = out["shards"][0]["artifacts"]["gen"]["trees"]["gen"]
    assert inventory == {"cycles": [1, 2], "selected_cycle": 2, "rows": 6, "unique_events": N}


@pytest.mark.parametrize("kind", ["worker", "dump_gen", "dump_crvsteps", "gen", "crvsteps"])
def test_missing_artifact_fails(campaign, tmp_path, capsys, kind):
    _path(campaign, kind, "run2", FIRST + 1).unlink()
    _fails(campaign, tmp_path / "report.json", capsys, "No such file")


@pytest.mark.parametrize("kind", ["worker", "dump_gen", "dump_crvsteps"])
@pytest.mark.parametrize("change", ["nonzero", "missing_exit", "duplicate_exit", "duplicate_summary",
                                    "missing_summary", "wrong_total", "bad_sum", "zero_passed",
                                    "malformed", "exit_first"])
def test_bad_log_evidence_fails(campaign, tmp_path, capsys, kind, change):
    path = _path(campaign, kind)
    total, passed = (2, 2) if kind == "dump_crvsteps" else (N, 2 if kind == "worker" else N)
    summary = _summary(total, passed, total - passed)
    marker = "Exit status: 0\n" if kind == "worker" else "dump_exit=0\n"
    text = summary + marker
    if change == "nonzero":
        text = summary + marker.replace("0", "9")
    elif change == "missing_exit":
        text = summary
    elif change == "duplicate_exit":
        text += marker
    elif change == "duplicate_summary":
        text = summary + text
    elif change == "missing_summary":
        text = marker
    elif change == "wrong_total":
        text = _summary(total + 1, passed, total + 1 - passed) + marker
    elif change == "bad_sum":
        text = _summary(total, passed, total - passed + 1) + marker
    elif change == "zero_passed":
        text = _summary(total, 0, total) + marker
    elif change == "malformed":
        text = summary.rstrip() + " extra text\n" + marker
    elif change == "exit_first":
        text = marker + summary
    path.write_text(text)
    _fails(campaign, tmp_path / "report.json", capsys, "Reference certification failed")


@pytest.mark.parametrize("kind", ["dump_gen", "dump_crvsteps"])
def test_dump_exit_must_be_final_line(campaign, tmp_path, capsys, kind):
    path = _path(campaign, kind)
    path.write_text(path.read_text() + "still writing\n")
    _fails(campaign, tmp_path / "report.json", capsys, "must end the dump log")


def test_worker_accepted_count_matches_persisted(campaign, tmp_path, capsys):
    _path(campaign, "worker").write_text(_summary(N, 3, N - 3) + "Exit status: 0\n")
    _path(campaign, "dump_crvsteps").write_text(_summary(3, 3, 0) + "dump_exit=0\n")
    _fails(campaign, tmp_path / "report.json", capsys, "accepted event count")


@pytest.mark.parametrize("kind,name", [("gen", "gen"), ("crvsteps", "gen"),
                                       ("crvsteps", "particles"), ("crvsteps", "steps")])
@pytest.mark.parametrize("column,value", [("run", 1401), ("subrun", FIRST + 1)])
def test_wrong_run_or_subrun(campaign, tmp_path, capsys, kind, name, column, value):
    path = _path(campaign, kind)
    frame = _frame(path, name)
    frame.loc[frame.index[0], column] = value
    _write_tree(path, name, frame, append=True)
    _fails(campaign, tmp_path / "report.json", capsys, "wrong run/subrun")


@pytest.mark.parametrize("kind", ["gen", "crvsteps"])
def test_duplicate_generator_identity(campaign, tmp_path, capsys, kind):
    path = _path(campaign, kind)
    frame = _frame(path, "gen")
    _write_tree(path, "gen", pd.concat([frame, frame.iloc[:1]], ignore_index=True), append=True)
    _fails(campaign, tmp_path / "report.json", capsys, "duplicate generator identity")


def test_shifted_generator_event_universe(campaign, tmp_path, capsys):
    path = _path(campaign, "gen")
    frame = _frame(path, "gen")
    frame["event"] += 1
    _write_tree(path, "gen", frame, append=True)
    _fails(campaign, tmp_path / "report.json", capsys, "event IDs must be exactly")


def test_particles_same_count_different_ids(campaign, tmp_path, capsys):
    path = _path(campaign, "crvsteps")
    frame = _frame(path, "particles")
    frame.loc[frame.event == 3, "event"] = 4
    _write_tree(path, "particles", frame, append=True)
    _fails(campaign, tmp_path / "report.json", capsys, "gen/particles event IDs differ")


@pytest.mark.parametrize("change", ["missing_primary", "unknown_idx", "negative_idx"])
def test_accepted_generator_identity_mismatch(campaign, tmp_path, capsys, change):
    path = _path(campaign, "crvsteps")
    frame = _frame(path, "gen")
    if change == "missing_primary":
        frame = frame.loc[frame.idx != 1]
    else:
        frame.loc[frame.idx == 1, "idx"] = -1 if change == "negative_idx" else 7
    _write_tree(path, "gen", frame, append=True)
    _fails(campaign, tmp_path / "report.json", capsys,
           "negative generator idx" if change == "negative_idx" else "generator identities differ")


@pytest.mark.parametrize("column", C.VALUES + ["genid"])
def test_each_generator_value_is_checked(campaign, tmp_path, capsys, column):
    path = _path(campaign, "crvsteps")
    frame = _frame(path, "gen")
    frame.loc[frame.index[0], column] += 1
    _write_tree(path, "gen", frame, append=True)
    _fails(campaign, tmp_path / "report.json", capsys, "generator value mismatch: " + column)


def test_steps_must_belong_to_accepted_events(campaign, tmp_path, capsys):
    path = _path(campaign, "crvsteps")
    frame = _frame(path, "steps")
    frame["event"] = np.full(len(frame), 2, dtype=np.uint32)
    _write_tree(path, "steps", frame, append=True)
    _fails(campaign, tmp_path / "report.json", capsys, "steps event IDs outside accepted")


@pytest.mark.parametrize("column", ["idx", "pdg", "e"])
def test_missing_generator_column_fails(campaign, tmp_path, capsys, column):
    path = _path(campaign, "gen")
    frame = _frame(path, "gen").drop(columns=column)
    _write_tree(path, "gen", frame, append=True)
    _fails(campaign, tmp_path / "report.json", capsys, "missing required columns")


@pytest.mark.parametrize("change", ["nan_value", "float_id", "missing_tree", "not_root"])
def test_malformed_root_evidence(campaign, tmp_path, capsys, change):
    path = _path(campaign, "gen")
    frame = _frame(path, "gen")
    message = "Reference certification failed"
    if change == "nan_value":
        frame.loc[0, "px"] = np.nan
        _write_tree(path, "gen", frame, append=True)
        message = "finite scalar numeric"
    elif change == "float_id":
        frame["event"] = frame.event.astype(float)
        _write_tree(path, "gen", frame, append=True)
        message = "must be integer"
    elif change == "missing_tree":
        _write_tree(path, "particles", frame[C.KEY])
        message = "missing dump/gen"
    else:
        path.write_bytes(b"not a ROOT file")
    _fails(campaign, tmp_path / "report.json", capsys, message)


@pytest.mark.parametrize("first,last,expected", [(0, FIRST, N), (FIRST + 1, FIRST, N),
                                                (FIRST, FIRST, 0), (FIRST, FIRST, -1)])
def test_invalid_cli_range_and_counts(tmp_path, capsys, first, last, expected):
    output = tmp_path / "report.json"
    assert C.main(_args(tmp_path / "absent", output, first, last, expected)) == 1
    assert "PASS" not in capsys.readouterr().out
    assert not output.exists()


def test_existing_output_never_overwritten(campaign, tmp_path, capsys):
    output = tmp_path / "report.json"
    output.write_text("retain this evidence\n")
    assert C.main(_args(campaign, output)) == 1
    assert output.read_text() == "retain this evidence\n"
    captured = capsys.readouterr()
    assert "refusing to overwrite" in captured.err and "PASS" not in captured.out


def test_cli_failure_nonzero_and_no_report(campaign, tmp_path):
    output = tmp_path / "report.json"
    _path(campaign, "worker").unlink()
    result = subprocess.run([sys.executable, "-B", "-m", "crvfastsim.analyses.reference_certification",
                             *_args(campaign, output)], capture_output=True, text=True,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert result.returncode != 0 and "PASS" not in result.stdout
    assert not output.exists()


@pytest.mark.parametrize("base", ["/pnfs/never/read", "//pnfs/never/read"])
def test_pnfs_rejected_before_metadata_lookup(tmp_path, monkeypatch, base):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not inspect pnfs metadata")
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(Path, "lstat", forbidden)
    with pytest.raises(ValueError, match="pnfs paths are forbidden"):
        C.certify(base, FIRST, FIRST, N, tmp_path / "report.json")


@pytest.mark.parametrize("kind", ["worker", "gen"])
def test_artifact_change_during_campaign_fails(campaign, tmp_path, capsys, monkeypatch, kind):
    original = C._certify_shard
    changed = _path(campaign, kind)

    def audit_then_mutate(*args):
        shard = original(*args)
        if shard["config"] == "run2" and shard["subrun"] == FIRST + 1:
            with changed.open("ab") as stream:
                stream.write(b"changed after inspection\n")
        return shard

    monkeypatch.setattr(C, "_certify_shard", audit_then_mutate)
    _fails(campaign, tmp_path / "report.json", capsys, "artifact changed during certification")


def test_output_created_concurrently_is_preserved(campaign, tmp_path, capsys, monkeypatch):
    output = tmp_path / "report.json"
    original = C._certify_shard

    def audit_then_create(*args):
        shard = original(*args)
        if shard["config"] == "run2" and shard["subrun"] == FIRST + 1:
            output.write_text("another certifier owns this path\n")
        return shard

    monkeypatch.setattr(C, "_certify_shard", audit_then_create)
    assert C.main(_args(campaign, output)) == 1
    captured = capsys.readouterr()
    assert "PASS" not in captured.out
    assert output.read_text() == "another certifier owns this path\n"


def test_all_accepted_ids_must_be_in_original_universe(campaign, tmp_path, capsys):
    path = _path(campaign, "crvsteps")
    for name in ("gen", "particles"):
        frame = _frame(path, name)
        frame.loc[frame.event == 3, "event"] = 6
        _write_tree(path, name, frame, append=True)
    _fails(campaign, tmp_path / "report.json", capsys, "accepted event outside generator universe")


def test_dumper_must_pass_every_input_event(campaign, tmp_path, capsys):
    _path(campaign, "dump_gen").write_text(_summary(N, N - 1, 1) + "dump_exit=0\n")
    _fails(campaign, tmp_path / "report.json", capsys, "dumper did not pass every input event")


def test_symlink_artifact_rejected_without_following(campaign, tmp_path, capsys):
    path = _path(campaign, "worker")
    path.unlink()
    path.symlink_to("/pnfs/never/read")
    _fails(campaign, tmp_path / "report.json", capsys, "symlink paths are not allowed")
