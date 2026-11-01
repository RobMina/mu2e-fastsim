"""scripts/chain_identity.py on synthetic records: column flattening, compare (two runs, in-process repeats), diff."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("chain_identity", SCRIPTS / "chain_identity.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


def test_columns_flatten_frames_dicts_and_lists():
    cols = ci.columns((pd.DataFrame({"x": [1.0, 2.0]}), {"b": np.arange(3), "a": None}, [np.ones(2)]))
    assert sorted(cols) == ["0.x", "1.a", "1.b", "2.0"]
    assert ci.digest(cols["1.b"]) != ci.digest(cols["1.b"].astype(np.int32))      # dtype is part of the hash


def _record(path, passes):
    path.write_text(json.dumps({"passes": passes}))


def test_compare_names_stage_and_columns(tmp_path, capsys):
    same = {"stages": {"digis": [{"adc": "h1", "volt": "v1"}]}}
    other = {"stages": {"digis": [{"adc": "h1", "volt": "v2"}]}}
    _record(tmp_path / "a.json", {"22#0": same})
    _record(tmp_path / "b.json", {"22#0": other})
    assert ci.compare(argparse.Namespace(a=str(tmp_path / "a.json"), b=str(tmp_path / "b.json"))) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["first_difference"] == {"22#0": "digis"} and out["changed_columns"]["22#0"]["digis"] == ["volt"]
    _record(tmp_path / "r.json", {"22#0": same, "22#1": same})                    # in-process repeats
    assert ci.compare(argparse.Namespace(a=str(tmp_path / "r.json"), b=None)) == 0


def test_diff_reports_changed_elements(tmp_path, capsys):
    for d, v in (("a", 1.0), ("b", 1.0 + 2 ** -52)):
        (tmp_path / d).mkdir()
        np.savez(tmp_path / d / "digis.npz", **{"0|volt": np.array([1.0, v]), "0|adc": np.array([3, 4])})
    assert ci.diff(argparse.Namespace(a=str(tmp_path / "a"), b=str(tmp_path / "b"))) == 1
    row = json.loads(capsys.readouterr().out)["stages"]["digis"]
    assert list(row) == ["0|volt"] and row["0|volt"]["changed"] == 1
    assert row["0|volt"]["max_abs"] == pytest.approx(2 ** -52)
