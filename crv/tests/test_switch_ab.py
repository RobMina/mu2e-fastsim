"""The A/B harness's schedule, pairing and difference report on synthetic records."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow   # campaign and validation tooling, not the simulation chain: runs with --runslow

pytest.importorskip("uproot")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("switch_ab", SCRIPTS / "switch_ab.py")
switch_ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(switch_ab)


def test_parse_arm():
    assert switch_ab.parse_arm("ref") == ("ref", {})
    assert switch_ab.parse_arm("b:X=1,Y=a=b") == ("b", {"X": "1", "Y": "a=b"})
    with pytest.raises(ValueError):
        switch_ab.parse_arm("b:X")


def test_schedule_alternates_after_warmup():
    assert switch_ab.schedule(["a", "b"], 1, 3) == [("a", -1), ("b", -1), ("a", 0), ("b", 0), ("b", 1), ("a", 1),
                                                     ("a", 2), ("b", 2)]


def test_set_arm_clears_the_other_arm(monkeypatch):
    arms = {"a": {}, "b": {"CRVFS_TEST_SWITCH": "1"}}
    monkeypatch.delenv("CRVFS_TEST_SWITCH", raising=False)
    switch_ab.set_arm(arms, "b")
    assert switch_ab.os.environ["CRVFS_TEST_SWITCH"] == "1"
    switch_ab.set_arm(arms, "a")
    assert "CRVFS_TEST_SWITCH" not in switch_ab.os.environ


def test_summary_pairs_rounds_and_skips_warmup():
    records = [dict(arm="a", round=-1, elapsed_s=100.0, timers={}),
               dict(arm="a", round=0, elapsed_s=10.0, timers={"reco_s": 4.0}),
               dict(arm="b", round=0, elapsed_s=5.0, timers={"reco_s": 2.0}),
               dict(arm="b", round=1, elapsed_s=20.0, timers={"reco_s": 2.0}),
               dict(arm="a", round=1, elapsed_s=20.0, timers={"reco_s": 4.0})]
    rows = switch_ab.summarize(records, "a")
    assert rows["a"]["median_s"] == 15.0 and rows["a"]["rounds"] == 2
    assert rows["b"]["paired_speedup_median"] == 1.5 and rows["b"]["paired_speedup_range"] == [1.0, 2.0]
    assert rows["b"]["stage_median_s"] == {"reco_s": 2.0}


def test_diff_npz_reports_changed_arrays(tmp_path):
    np.savez(tmp_path / "a.npz", same=np.arange(3), moved=np.array([1.0, 2.0, np.nan]), kind=np.zeros(2, np.int32))
    np.savez(tmp_path / "b.npz", same=np.arange(3), moved=np.array([1.0, 2.5, np.nan]), kind=np.zeros(2, np.int64))
    result = switch_ab.diff_npz(tmp_path / "a.npz", tmp_path / "b.npz")
    assert set(result) == {"moved", "kind"}
    assert result["moved"]["changed"] == 1 and result["moved"]["max_abs"] == 0.5


def test_difference_sums_over_inputs_and_chunks(tmp_path):
    for arm, offset in (("a", 0.0), ("b", 0.25)):
        folder = tmp_path / arm / "benchmark/pass_000"
        folder.mkdir(parents=True)
        for index in range(2):
            np.savez(folder / f"input_{index:04d}.npz", chunk_0000__cl_pe=np.array([1.0, 2.0 + offset]),
                     chunk_0001__cl_pe=np.array([3.0, 4.0]), cl_count=np.arange(2))
    assert switch_ab.difference(tmp_path / "a", tmp_path / "b") == {"cl_pe": {"size": 4, "changed": 2, "max_abs": 0.25}}
