"""Native occupancy joins complete IDs and retains empty sectors/events."""
import awkward as ak
import numpy as np
import pytest
import uproot

from mixing.counts import align, json_values, read_counts
from mixing.plot_counts import plot


def make_file(path, bad_pulse=False, wrong_count=False):
    ids = {"run": np.array([1, 1, 1]), "subrun": np.array([2, 3, 2]), "event": np.array([5, 5, 6])}
    pulses = {"run": np.array([1, 1]), "subrun": np.array([3, 2]),
              "event": np.array([5, 99 if bad_pulse else 5]), "bar": np.array([2, 0])}
    with uproot.recreate(path) as root:
        root.mktree("trkdigi/events", ids | {"d_sid": ak.Array([[1, 1025], [2048], []])})
        root.mktree("crvdump/events", ids | {"npulse": np.array([1, 1, 1 if wrong_count else 0])})
        root.mktree("crvdump/pulses", pulses)


def test_full_ids_and_empty_events(tmp_path):
    path = tmp_path / "counts.root"
    make_file(path)
    counts = read_counts(path, np.array([0, 0, 1]), 3)
    np.testing.assert_array_equal(counts["planes"][:, :3], [[1, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert counts["planes"].shape == (3, 36)
    np.testing.assert_array_equal(counts["sectors"], [[1, 0, 0], [0, 1, 0], [0, 0, 0]])


def test_reject_missing_census_and_wrong_counts(tmp_path):
    path = tmp_path / "bad.root"
    make_file(path, bad_pulse=True)
    with pytest.raises(ValueError, match="without matching"):
        read_counts(path, np.array([0, 0, 1]), 2)
    make_file(path, wrong_count=True)
    with pytest.raises(ValueError, match="disagrees"):
        read_counts(path, np.array([0, 0, 1]), 2)


def test_pair_alignment_is_not_row_alignment(tmp_path):
    path = tmp_path / "counts.root"
    make_file(path)
    stock = read_counts(path, np.array([0, 0, 1]), 2)
    candidate = {key: values[[2, 0]] for key, values in stock.items()}
    result = align(stock, candidate)
    assert result["matched_events"] == 2
    assert result["stock_only_events"] == 1
    assert result["candidate_only_events"] == 0
    np.testing.assert_array_equal(result["planes"]["difference"], np.zeros(36))
    np.testing.assert_array_equal(result["sectors"]["difference"], np.zeros(2))
    np.testing.assert_array_equal(result["planes_total"]["difference"], [0])
    np.testing.assert_array_equal(result["sectors_total"]["difference"], [0])
    assert result["planes_changed_events"] == result["sectors_changed_events"] == 0


def test_event_changes_cannot_cancel_in_totals(tmp_path):
    path = tmp_path / "counts.root"
    make_file(path)
    stock = read_counts(path, np.array([0, 0, 1]), 2)
    candidate = {key: values.copy() for key, values in stock.items()}
    candidate["planes"][[0, 1]] = candidate["planes"][[1, 0]]
    result = align(stock, candidate)
    assert result["planes_changed_events"] == 2
    assert result["sectors_changed_events"] == 0
    np.testing.assert_array_equal(result["planes"]["difference"], np.zeros(36))


def test_plot_has_measured_paired_input(tmp_path):
    path = tmp_path / "counts.root"
    make_file(path)
    stock = read_counts(path, np.array([0, 0, 1]), 2)
    report = json_values({"paired": align(stock, stock), "sector_names": ["T1", "T2"]})
    output = tmp_path / "occupancy.png"
    plot(report, output, "Synthetic plotting test")
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
