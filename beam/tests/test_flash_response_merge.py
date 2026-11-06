"""Only independent, completed response event ranges can be concatenated."""
import json

import numpy as np
import pytest

from beamfastsim.flash_response_merge import merge, main
from test_flash_response_compare import sample


def write(path, offset=0, **report):
    d = sample()
    for k in ("event_event", "event_ordinal", "event"):
        d[k] += np.uint64(offset)
    d["response_report_json"] = np.asarray(json.dumps(dict(input_events=4, expected_events=4,
        failed_recording_events=0, incomplete_tracker_events=0, **report)))
    np.savez_compressed(path, **d)
    return path


def test_reverse_input_order_sorted_ledger_and_photon_mapping(tmp_path):
    a, b = write(tmp_path/"a.npz"), write(tmp_path/"b.npz", 4)
    out = merge([b, a], expected_events=8)
    np.testing.assert_array_equal(out["event_ordinal"], np.arange(8))
    assert out["input_event_count"] == 8
    assert out["photon_assigned_ionizing_mev"].sum() == 8.
    assert "tracker_owner_photon_row" not in out


def test_overlapping_events_and_missing_shards_rejected(tmp_path):
    a = write(tmp_path/"a.npz")
    with pytest.raises(ValueError, match="Duplicate"):
        merge([a, a], expected_events=8)
    with pytest.raises(ValueError, match="count"):
        merge([a], expected_events=8)


def test_same_event_different_ordinal_rejected(tmp_path):
    a = write(tmp_path/"a.npz")
    with np.load(a) as z:
        d = dict(z)
    d["event_ordinal"] += np.uint64(10)
    b = tmp_path/"b.npz"
    np.savez(b, **d)
    with pytest.raises(ValueError, match="Duplicate"):
        merge([a, b], expected_events=8)


@pytest.mark.parametrize("field", ["failed_recording_events", "incomplete_tracker_events", "expected_events"])
def test_failed_shard_report_rejected(tmp_path, field):
    a = write(tmp_path/"a.npz")
    with np.load(a) as z:
        d = dict(z)
    report = json.loads(str(d["response_report_json"]))
    report[field] = 1
    d["response_report_json"] = np.asarray(json.dumps(report))
    np.savez(a, **d)
    with pytest.raises(ValueError):
        merge([a], expected_events=4)


def test_cli_no_overwrite(tmp_path):
    a, out = write(tmp_path/"a.npz"), tmp_path/"merged.npz"
    argv = ["--input", str(a), "--events", "4", "--out", str(out)]
    assert main(argv) == 0
    with pytest.raises(FileExistsError):
        main(argv)
