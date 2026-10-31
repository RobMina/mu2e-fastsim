"""Contracts for the unprivileged Offline runner and exact ROOT bridge comparison."""
import importlib.util
import os
import re
from pathlib import Path
import subprocess

import numpy as np
import pytest
import uproot

HERE = Path(__file__).resolve().parents[1] / "offline"
SCRIPTS = sorted(p for p in HERE.iterdir() if p.suffix in (".sh", ".sbatch") and not p.name.startswith("."))
spec = importlib.util.spec_from_file_location("offline_compare", HERE / "compare_root.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def make_root(path, value=2.0, entries=3):
    with uproot.recreate(path) as root:
        root["counts"] = np.histogram(np.array([value, 2, 3]), bins=np.arange(6))
        root.mktree("events", {"event": "int64", "value": "float64"})
        root["events"].extend({"event": np.arange(entries), "value": np.ones(entries)})


def test_exact_content(tmp_path):
    a, b = tmp_path / "a.root", tmp_path / "b.root"
    make_root(a)
    make_root(b)
    result = compare.compare(a, b)
    assert result["equal"]
    assert next(x for x in result["objects"] if x["key"] == "events")["candidate_entries"] == 3


def test_changed_histogram(tmp_path):
    a, b = tmp_path / "a.root", tmp_path / "b.root"
    make_root(a)
    make_root(b, value=4.0)
    result = compare.compare(a, b)
    assert not result["equal"]
    assert [x["key"] for x in result["objects"] if not x["equal"]] == ["counts"]


def test_named_metadata(tmp_path):
    from uproot.models.TNamed import Model_TNamed
    from uproot.models.TObject import Model_TObject
    a, b = tmp_path / "a.root", tmp_path / "b.root"
    for path, title in ((a, "v3"), (b, "v4")):
        named = Model_TNamed.empty()
        named._bases.append(Model_TObject.empty())
        named._members.update(fName="version", fTitle=title)
        with uproot.recreate(path) as root:
            root["version"] = named
    assert compare.compare(a, a)["equal"]
    assert not compare.compare(a, b)["equal"]


def test_changed_entry_count(tmp_path):
    a, b = tmp_path / "a.root", tmp_path / "b.root"
    make_root(a)
    make_root(b, entries=2)
    assert not compare.compare(a, b)["equal"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_runner_requires_allocation():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM", "OFFLINE"))}
    result = subprocess.run(["bash", str(HERE / "offline_uva.sh"), "area", "job.fcl"],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "compute-node Slurm allocation" in result.stderr


def test_node_cache_and_no_builds():
    runner = (HERE / "offline_uva.sh").read_text()
    assert "CVMFS cache must be on node-local" in runner
    assert "trap 'rm -rf" in runner
    assert "singularity.opensciencegrid.org --" in runner
    assert "apptainer exec --userns" in runner
    for script in SCRIPTS:
        assert not re.search(r"^\s*muse\s+(build|tarball)\b", script.read_text(), re.MULTILINE)
