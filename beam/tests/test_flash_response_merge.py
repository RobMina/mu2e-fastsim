"""Only independent, completed response event ranges can be concatenated."""
import json

import numpy as np
import pytest

from beamfastsim.flash_response_merge import EVENT_COLUMNS, PHOTON_COLUMNS, merge, main
from beamfastsim.flash_response_compare import EVENT_FIELDS, _arm
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


IDENTITY_COLUMNS = (*EVENT_FIELDS, *PHOTON_COLUMNS[:5])


def shard_data(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


@pytest.mark.parametrize("dtypes", [(np.uint64, np.int64), (np.int64, np.uint64),
                                    (np.uint64, np.uint64), (np.int64, np.int64)])
def test_integer_identity_dtypes_preserve_exact_large_values(tmp_path, dtypes):
    paths = [write(tmp_path / "a.npz"), write(tmp_path / "b.npz", 4)]
    pieces = []
    for path, dtype in zip(paths, dtypes):
        part = shard_data(path)
        for name in IDENTITY_COLUMNS:
            part[name] = (part[name] + np.uint64(2**53 + 17)).astype(dtype)
        if dtype == np.uint64:
            for name in ("event_run", "run", "product", "key"):
                part[name] += np.uint64(2**63)
        _arm(part)
        pieces.append(part)
        np.savez_compressed(path, **part)
    merged = merge(paths[::-1], expected_events=8)
    for name in IDENTITY_COLUMNS:
        order = pieces if name in EVENT_FIELDS else pieces[::-1]
        expected = np.concatenate([part[name].astype(np.uint64) for part in order])
        assert merged[name].dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(merged[name], expected)
    assert merged["input_event_count"] == 8
    assert merged["photon_assigned_ionizing_mev"].sum() == 8.
    assert merged["frontier_status"].dtype == pieces[0]["frontier_status"].dtype


@pytest.mark.parametrize("reverse", [False, True])
def test_signed_empty_photons_keep_large_unsigned_identities_and_zero_events(tmp_path, reverse):
    a, b = write(tmp_path / "a.npz"), write(tmp_path / "zero.npz", 4)
    populated, empty = shard_data(a), shard_data(b)
    for name in ("product", "key"):
        populated[name] += np.uint64(2**63 + 11)
    for name in PHOTON_COLUMNS:
        empty[name] = empty[name][:0]
    for name in PHOTON_COLUMNS[:5]:
        empty[name] = empty[name].astype(np.int64)
    for name in EVENT_COLUMNS[len(EVENT_FIELDS):]:
        empty[name][:] = 0
    for path, part in ((a, populated), (b, empty)):
        _arm(part)
        np.savez_compressed(path, **part)
    merged = merge([b, a] if reverse else [a, b], expected_events=8)
    for name in PHOTON_COLUMNS[:5]:
        assert merged[name].dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(merged[name], populated[name])
    np.testing.assert_array_equal(merged["event_ordinal"], np.arange(8, dtype=np.uint64))
    np.testing.assert_array_equal(merged["event_tracker_ionizing_mev"][4:], np.zeros(4))
    assert merged["input_event_count"] == 8


@pytest.mark.parametrize("field", IDENTITY_COLUMNS)
@pytest.mark.parametrize("invalid", ["negative", "float"])
def test_identity_validation_precedes_unsigned_conversion(tmp_path, field, invalid):
    path = write(tmp_path / "bad.npz")
    part = shard_data(path)
    part[field] = part[field].astype(np.int64 if invalid == "negative" else float)
    if invalid == "negative":
        part[field][0] = -1
    np.savez_compressed(path, **part)
    with pytest.raises(ValueError, match=field):
        merge([path], expected_events=4)
