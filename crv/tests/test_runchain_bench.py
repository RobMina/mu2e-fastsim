"""analyses.runchain_bench: inputs read once, setup built in the first timed pass, every pass streams all files."""
import json

import pandas as pd
import pytest

pytest.importorskip("mu2efastsim.bench")

from crvfastsim import tables
from crvfastsim.analyses import runchain, runchain_bench


def test_bench_streams_in_memory_tables_with_one_context(tmp_path, monkeypatch):
    reads, prepared, runs = [], [], []
    gen = pd.DataFrame(dict(subrun=[1, 1, 1], event=[1, 2, 2], run=1))
    monkeypatch.setattr(tables, "expand", lambda inputs: list(inputs))
    monkeypatch.setattr(tables, "read", lambda f, what: reads.append((f, what)) or gen)
    monkeypatch.setattr(runchain, "prepare", lambda **kw: prepared.append(kw) or "context")

    def fake_run(files, outdir, **kw):
        runs.append(kw)
        assert kw["prepared"] == "context" and set(kw["gen_tables"]) == {"a.root", "b.root"}
        return dict(generated_events=4, clusters_per_generated={"T1": 0.5}, timers={}, run_config="run1a_v01")
    monkeypatch.setattr(runchain, "run", fake_run)
    out = tmp_path / "bench.json"
    runchain_bench.main(["a.root", "b.root", "--calib", "c.csv", "--run-config", "run1a_v01", "--out", str(out)])
    report = json.loads(out.read_text())
    assert reads == [("a.root", "gen"), ("b.root", "gen")]
    assert len(prepared) == 1 and len(runs) == 3
    assert report["events"] == 4 and report["passes"] == 3 and report["workflow"] == "W5"
    assert report["runchain"]["clusters_per_generated"] == {"T1": 0.5}


def fresh_setup(monkeypatch, sizes):
    """Fake tables and runchain: file name -> number of generated events; returns the recorded runs."""
    from mu2efastsim import bench
    if not hasattr(bench, "fresh_inputs"):
        pytest.skip("core without mu2efastsim.bench.fresh_inputs")
    runs, prepared = [], []
    monkeypatch.setattr(tables, "expand", lambda inputs: list(inputs))
    monkeypatch.setattr(tables, "read", lambda f, what: pd.DataFrame(dict(subrun=1, event=range(sizes[f]), run=1)))
    monkeypatch.setattr(runchain, "prepare", lambda **kw: prepared.append(kw) or "context")

    def fake_run(files, outdir, **kw):
        runs.append((list(files), sorted(kw["gen_tables"]), kw["seed"]))
        assert kw["prepared"] == "context"
        return dict(generated_events=0, clusters_per_generated={}, timers={})
    monkeypatch.setattr(runchain, "run", fake_run)
    return runs, prepared


def test_fresh_inputs_give_each_pass_its_files_and_seed(tmp_path, monkeypatch):
    sizes = {"a": 3, "b": 4, "c": 5, "d": 6, "e": 7, "f": 8}
    runs, prepared = fresh_setup(monkeypatch, sizes)
    out = tmp_path / "bench.json"
    runchain_bench.main([*sizes, "--calib", "c.csv", "--seed", "7", "--fresh-inputs", "--files-per-pass", "2",
                         "--out", str(out)])
    assert runs == [(["a", "b"], ["a", "b"], 7), (["c", "d"], ["c", "d"], 8), (["e", "f"], ["e", "f"], 9)]
    assert len(prepared) == 1
    report = json.loads(out.read_text())
    assert report["inputs"] == "fresh" and report["seeds"] == [7, 8, 9] and report["file_reuse"] is False
    assert report["events_per_pass"] == [7, 11, 15] and report["events"] == 11
    assert report["files_per_pass"] == [["a", "b"], ["c", "d"], ["e", "f"]]
    assert report["warm_rate"] == pytest.approx(26 / sum(report["pass_s"][1:]))


def test_fresh_inputs_refuse_too_few_files_unless_reuse_is_allowed(tmp_path, monkeypatch):
    sizes = {"a": 3, "b": 4, "c": 5, "d": 6}
    runs, _ = fresh_setup(monkeypatch, sizes)
    out = tmp_path / "bench.json"
    with pytest.raises(SystemExit, match="distinct files"):
        runchain_bench.main([*sizes, "--calib", "c.csv", "--fresh-inputs", "--files-per-pass", "2", "--out", str(out)])
    assert runs == [] and not out.exists()
    runchain_bench.main([*sizes, "--calib", "c.csv", "--fresh-inputs", "--files-per-pass", "2", "--allow-file-reuse",
                         "--out", str(out)])
    assert [r[0] for r in runs] == [["a", "b"], ["c", "d"], ["a", "b"]] and [r[2] for r in runs] == [1, 2, 3]
    assert json.loads(out.read_text())["file_reuse"] is True


def test_fresh_input_options_are_checked(tmp_path):
    out = str(tmp_path / "bench.json")
    with pytest.raises(SystemExit):
        runchain_bench.main(["a", "--calib", "c.csv", "--fresh-inputs", "--out", out])
    with pytest.raises(SystemExit):
        runchain_bench.main(["a", "--calib", "c.csv", "--files-per-pass", "2", "--out", out])
