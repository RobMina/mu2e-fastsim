"""Saved-input CLI contracts for empty responses and strict JSON summaries."""
import json
import sys

import numpy as np
import pytest
import uproot

from calofastsim.showers import compare
from calofastsim.showers.export import read
from calofastsim.showers.library import build, save
from test_showers_export import tables, write


BACKEND = """import numpy as np

def corrected_energy(g4, birks, z, length, lru):
    return birks

def fold_step_time(time, *, digitization_start, digitization_end, **conditions):
    return time, (time >= digitization_start) & (time <= digitization_end)
"""


def strict_json(text):
    def reject(value):
        raise ValueError("nonstandard JSON token " + value)
    return json.loads(text, parse_constant=reject)


def inputs(tmp_path, mode):
    rows = tables()
    rows["steps"]["time"] += 1000.
    rows["showers"]["time"] += 1000.
    split = 2 if mode == "empty_arm" else 1
    if mode in ("both_positive", "rejected_time"):
        heldout = {k: v[:1].copy() for k, v in rows["showers"].items()}
        heldout["event"][:] = 2
        heldout["time"] += 10.
        rows["showers"] = {k: np.concatenate([v, heldout[k]]) for k, v in rows["showers"].items()}
        rows["events"]["n_showers"][1] = 1
    if mode == "empty_donor":
        keep = rows["showers"]["event"] != 1
        rows["showers"] = {k: v[keep] for k, v in rows["showers"].items()}
        rows["events"]["n_showers"][0] = 0
    if mode == "empty_response":
        rows["showers"] = {k: v[:0] for k, v in rows["showers"].items()}
        rows["events"]["n_showers"][:] = 0
    if mode == "zero_energy":
        rows["showers"]["energy_g4"][:] = 0.
        rows["showers"]["energy_birks"][:] = 0.
    if mode == "rejected_time":
        rows["steps"]["time"] -= 1000.
        rows["showers"]["time"] -= 1000.
    path = tmp_path / "input.root"
    write(path, rows)
    entries, steps, geometry, _, _ = read(path)
    library = build(entries, steps, development=entries["event_row"] < split)
    library.update({"geometry_" + k: v for k, v in geometry.items()})
    save(tmp_path / "library.npz", library, dict(split_row=split, split="source event row"))
    events = {k: rows["events"][k][::-1] for k in ("run", "subrun", "event")}
    events.update(pbtime=np.zeros(3), mbtime=np.full(3, 2000.), event_length=np.full(3, 2200.),
                  marker=np.zeros(3), spill=np.ones(3, int), crystal_length=np.full(3, 20.))
    n = 0 if mode in ("empty_response", "missing_calibration") else 2
    with uproot.recreate(tmp_path / "response.root") as out:
        out.mktree("response/events", events)
        out.mktree("response/steps", dict(crystal=np.array([10, 11])[:n], lru=np.full(n, .2)))
    (tmp_path / "backend.py").write_text(BACKEND)
    return ["compare", "--library", str(tmp_path / "library.npz"), "--input", str(path),
            "--response-input", str(tmp_path / "response.root"), "--response-source", str(tmp_path / "backend.py"),
            "--out", str(tmp_path / "out"), "--arm", "heldout", "--digitization-start", "450",
            "--digitization-end", "1795", "--digitization-buffer", "10"]


@pytest.mark.parametrize("mode", ["both_positive", "no_joint_response", "empty_arm", "empty_donor",
                                  "rejected_time", "empty_response", "zero_energy"])
def test_compare_cli_empty_populations(tmp_path, monkeypatch, capsys, mode):
    monkeypatch.setattr(sys, "argv", inputs(tmp_path, mode))
    compare.main()
    out = tmp_path / "out"
    result = strict_json((out / "numbers.json").read_text())
    stdout = strict_json(capsys.readouterr().out)
    assert stdout == {k: v for k, v in result.items() if k != "histograms"}
    assert result["source_events"] == 3
    assert (out / "calo_ce.png").is_file()
    with np.load(out / "clusters.npz", allow_pickle=False) as clusters:
        n = 0 if mode == "empty_arm" else 1
        assert result["entries"] == n == len(clusters["event_row"])
        np.testing.assert_array_equal(clusters["event_row"], [] if n == 0 else [1])
        if mode == "both_positive":
            np.testing.assert_array_equal(clusters["fast_energy"], [68.])
            np.testing.assert_array_equal(clusters["stock_energy"], [39.])
            assert result["acceptance_ratio"] == dict(ratio=1., error=None)
            assert result["energy_mean"] == dict(fast=68., stock=39., difference=29., error=None)
            assert np.isfinite(result["mean_time_difference_ns"])
            assert np.isfinite(result["mean_position_distance_mm"])
        else:
            assert result["mean_time_difference_ns"] is None
            assert result["mean_position_distance_mm"] is None
            assert result["acceptance_ratio"] == dict(ratio=None, error=None)
            for arm in ("fast", "stock"):
                empty = clusters[arm + "_energy"] == 0
                assert np.isnan(clusters[arm + "_time"][empty]).all()
                assert np.isnan(clusters[arm + "_position"][empty]).all()
        if mode == "empty_arm":
            for name in ("energy_mean", "acceptance_above_10_mev", "truth_ep_mean"):
                assert result[name] == dict(fast=None, stock=None, difference=None, error=None)
        elif mode in ("empty_donor", "empty_response", "zero_energy", "rejected_time"):
            assert result["energy_mean"] == dict(fast=0., stock=0., difference=0., error=None)
            assert result["acceptance_above_10_mev"] == dict(fast=0., stock=0., difference=0., error=None)
        assert result["unmapped_visible_energy_fraction"] == (
            None if mode in ("empty_arm", "empty_donor", "empty_response", "zero_energy") else 0.)


def test_empty_response_conditions_preserve_events_and_no_calibration(tmp_path):
    inputs(tmp_path, "empty_response")
    events, lru = compare.response_inputs(tmp_path / "response.root")
    np.testing.assert_array_equal(events["event"], [3, 2, 1])
    assert lru.shape == (0,) and lru.dtype.kind == "f"


def test_nonempty_deposit_without_calibration_still_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", inputs(tmp_path, "missing_calibration"))
    with pytest.raises(ValueError, match="sampled crystal has no saved response conditions"):
        compare.main()


def test_paired_mean_empty_is_unavailable():
    assert compare.paired_mean([], [], []) == dict(fast=None, stock=None, difference=None, error=None)
