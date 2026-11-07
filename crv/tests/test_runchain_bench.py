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
