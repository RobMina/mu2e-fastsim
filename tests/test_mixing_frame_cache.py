"""Content-checked source caching without mutable aliases or validation bypasses."""
from copy import deepcopy
from unittest.mock import Mock

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from mixing import direct_frames
from mixing.direct_frames import load_manifest
from mixing.frame_cache import FrameSourceCache
from mixing.frames import FrameBatch, overlay
from test_mixing_direct_frames import (KEYS, batch, manifest, native_products,
                                       subset, write_dump)


def assert_batch_equal(actual, expected):
    assert actual.event_ids.dtype == expected.event_ids.dtype
    np.testing.assert_array_equal(actual.event_ids, expected.event_ids)
    assert set(actual.collections) == set(expected.collections)
    for name, rows in expected.collections.items():
        other = actual.collections[name]
        assert ak.fields(other) == ak.fields(rows)
        np.testing.assert_array_equal(ak.to_numpy(ak.num(other)), ak.to_numpy(ak.num(rows)))
        for field in ak.fields(rows):
            left, right = (ak.to_numpy(ak.flatten(values[field])) for values in (other, rows))
            assert left.dtype == right.dtype
            np.testing.assert_array_equal(left, right)


def assert_nested_equal(actual, expected):
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    else:
        assert actual == expected


def write_log(path, ids):
    path.write_text("\n".join(f"Event: {i} run: {r} subRun: {s} event: {e}"
                              for i, (r, s, e) in enumerate(ids)) + "\n")


@pytest.mark.parametrize("with_log", [False, True])
def test_hit_hashes_every_file_but_skips_parse_and_native_validation(tmp_path, monkeypatch, with_log):
    path = tmp_path / "source.root"
    write_dump(path, batch(), products=native_products(batch()))
    source = {"path": str(path)}
    if with_log:
        log = tmp_path / "events.log"
        write_log(log, KEYS)
        source["event_log"] = str(log)
    spies = {}
    for name in ("_sha256", "read_frames", "_validate_native_products"):
        spies[name] = Mock(wraps=getattr(direct_frames, name))
        monkeypatch.setattr(direct_frames, name, spies[name])
    cache = FrameSourceCache()
    first, fingerprint = cache.read(source)
    for _ in range(2):
        other, other_fingerprint = cache.read(dict(reversed(list(source.items()))))
        assert_batch_equal(other, first)
        assert other_fingerprint == fingerprint
        assert other is not first
        assert all(other.collections[name] is not first.collections[name] for name in first.collections)
    assert cache.hits == 2 and cache.misses == 1
    assert spies["read_frames"].call_count == spies["_validate_native_products"].call_count == 1
    assert [call.args[0] for call in spies["_sha256"].call_args_list] == list(source.values()) * 4
    assert fingerprint["product_id_checks"]["tracker"] == "checked"


def test_source_replacement_at_same_path_invalidates(tmp_path, monkeypatch):
    path = tmp_path / "source.root"
    source = {"path": str(path)}
    original = batch()
    write_dump(path, original)
    cache = FrameSourceCache()
    _, before = cache.read(source)
    changed = deepcopy(original)
    changed.collections["tracker"]["t"] = changed.collections["tracker"].t + 17
    write_dump(path, changed)
    reader = Mock(wraps=direct_frames.read_frames)
    monkeypatch.setattr(direct_frames, "read_frames", reader)
    actual, after = cache.read(source)
    assert reader.call_count == 1
    assert cache.misses == 2 and cache.hits == 0
    assert before["source_sha256"] != after["source_sha256"]
    assert before["census_sha256"] == after["census_sha256"]
    assert_batch_equal(actual, changed)
    cache.read(source)
    assert reader.call_count == 1 and cache.hits == 1


def test_log_replacement_at_same_path_invalidates_ordered_census(tmp_path, monkeypatch):
    path, log = tmp_path / "source.root", tmp_path / "events.log"
    write_dump(path, batch(), census=False)
    write_log(log, KEYS)
    source = {"path": str(path), "event_log": str(log)}
    cache = FrameSourceCache()
    _, before = cache.read(source)
    write_log(log, KEYS[::-1])
    reader = Mock(wraps=direct_frames.read_frames)
    monkeypatch.setattr(direct_frames, "read_frames", reader)
    actual, after = cache.read(source)
    assert reader.call_count == 1 and cache.misses == 2
    assert before["source_sha256"] == after["source_sha256"]
    assert before["event_log_sha256"] != after["event_log_sha256"]
    assert before["census_sha256"] != after["census_sha256"]
    assert_batch_equal(actual, subset(batch(), slice(None, None, -1)))


@pytest.mark.parametrize("damage", ["native", "reference", "log"])
def test_malformed_replacement_is_rejected_not_served_from_cache(tmp_path, damage):
    path, log = tmp_path / "source.root", tmp_path / "events.log"
    original = batch()
    write_dump(path, original, products=native_products(original))
    source = {"path": str(path)}
    if damage == "log":
        write_log(log, KEYS)
        source["event_log"] = str(log)
    cache = FrameSourceCache()
    cache.read(source)
    if damage == "native":
        products = native_products(original)
        products["tracker"]["sp_product_id"][-1] += 1
        write_dump(path, original, products=products)
        message = "native product ID/key mismatch"
    elif damage == "reference":
        original.collections["tracker"]["sp"] = original.collections["tracker"].sp + 100
        write_dump(path, original)
        message = "dangling tracker"
    else:
        write_log(log, np.concatenate([KEYS, KEYS[:1]]))
        message = "duplicate event IDs"
    for _ in range(2):
        with pytest.raises(ValueError, match=message):
            cache.read(source)
    assert cache.hits == 0


def test_returned_wrappers_fingerprints_and_buffers_cannot_corrupt_snapshot(tmp_path):
    path = tmp_path / "source.root"
    write_dump(path, batch())
    source, cache = {"path": str(path)}, FrameSourceCache()
    returned, fingerprint = cache.read(source)
    expected, expected_fingerprint = cache.read(source)
    buffers = [returned.event_ids]
    for rows in returned.collections.values():
        buffers.extend(ak.to_buffers(rows)[2].values())
    for values in buffers:
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)
        if values.size:
            with pytest.raises(ValueError):
                values.flat[0] = 0
    returned.event_ids.shape = (returned.event_ids.size,)
    returned.collections["tracker"]["t"] = returned.collections["tracker"].t + 123
    returned.collections["particles"]["new_field"] = returned.collections["particles"].key
    del returned.collections["gen"]
    returned.collections["crv"] = ak.Array([[]])
    fingerprint["source_sha256"] = "corrupted"
    fingerprint["product_id_checks"]["tracker"] = "corrupted"
    actual, actual_fingerprint = cache.read(source)
    assert_batch_equal(actual, expected)
    assert actual_fingerprint == expected_fingerprint
    assert actual.collections["tracker"] is not expected.collections["tracker"]


def test_snapshot_owns_bytes_not_original_reader_buffers(tmp_path, monkeypatch):
    path = tmp_path / "source.root"
    write_dump(path, batch())
    original_read, originals = direct_frames._read, []
    def capture(*args, **kwargs):
        result = original_read(*args, **kwargs)
        originals.append(result)
        return result
    monkeypatch.setattr(direct_frames, "_read", capture)
    cache, source = FrameSourceCache(), {"path": str(path)}
    expected, fingerprint = cache.read(source)
    original, _ = originals[0]
    expected = deepcopy(expected)
    original.event_ids[:] = 123
    for rows in original.collections.values():
        for values in ak.to_buffers(rows)[2].values():
            if values.dtype.kind in "iuf" and values.size:
                values[:] = 0
    actual, actual_fingerprint = cache.read(source)
    assert len(originals) == 1
    assert_batch_equal(actual, expected)
    assert actual_fingerprint == fingerprint
    np.testing.assert_array_equal(actual.event_ids, KEYS)


def test_clear_forces_reload_without_mutating_existing_results(tmp_path, monkeypatch):
    path = tmp_path / "source.root"
    write_dump(path, batch())
    cache, source = FrameSourceCache(), {"path": str(path)}
    old, fingerprint = cache.read(source)
    reader = Mock(wraps=direct_frames.read_frames)
    monkeypatch.setattr(direct_frames, "read_frames", reader)
    cache.clear()
    actual, actual_fingerprint = cache.read(source)
    assert reader.call_count == 1
    assert_batch_equal(actual, old)
    assert actual_fingerprint == fingerprint


def test_zero_event_source_can_be_cached(tmp_path):
    path = tmp_path / "empty.root"
    empty = subset(batch(), slice(0, 0))
    write_dump(path, empty, products=native_products(empty))
    cache = FrameSourceCache()
    for _ in range(2):
        actual, fingerprint = cache.read({"path": str(path)})
        assert_batch_equal(actual, empty)
        assert actual.event_ids.shape == (0, 3)
        assert "census_sha256" in fingerprint
    assert cache.hits == cache.misses == 1


@pytest.mark.parametrize("limit", [None, 0, 2, 99])
def test_cached_manifest_exactly_matches_all_uncached_inputs_truth_and_donors(manifest, limit):
    path, _ = manifest
    expected = load_manifest(path, max_events=limit)
    cache = FrameSourceCache()
    for _ in range(2):
        actual = load_manifest(path, max_events=limit, source_cache=cache)
        assert_batch_equal(actual[0], expected[0])
        pd.testing.assert_frame_equal(actual[1], expected[1], check_exact=True)
        pd.testing.assert_frame_equal(actual[2], expected[2], check_exact=True)
        np.testing.assert_array_equal(actual[3], expected[3])
        assert actual[3].dtype == expected[3].dtype
        for key in expected[4].keys() - {"timing_s"}:
            assert_nested_equal(actual[4][key], expected[4][key])
    assert cache.misses == 2 and cache.hits == 8


def test_warm_cache_still_validates_overlay_and_adapter_every_call(manifest, monkeypatch):
    path, _ = manifest
    cache = FrameSourceCache()
    load_manifest(path, max_events=2, source_cache=cache)
    def forbidden(*args, **kwargs):
        pytest.fail("warm source was parsed or natively validated again")
    monkeypatch.setattr(direct_frames, "read_frames", forbidden)
    monkeypatch.setattr(direct_frames, "_validate_native_products", forbidden)
    original, seen = FrameBatch.validate_references, []
    def validate(self):
        seen.append(self)
        return original(self)
    monkeypatch.setattr(FrameBatch, "validate_references", validate)
    for _ in range(2):
        seen.clear()
        mixed, *_ = load_manifest(path, max_events=2, source_cache=cache)
        assert any(item is mixed for item in seen)
        assert sum(len(item.event_ids) == len(KEYS) for item in seen) >= 4
        assert len(seen) >= 6
    returned, _ = cache.read({"path": str(path.parent / "data/signal.root")})
    returned.collections["tracker"]["sp"] = returned.collections["tracker"].sp + 100
    with pytest.raises(ValueError, match="dangling tracker"):
        overlay(returned, [])


def test_hash_read_race_does_not_publish_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "source.root"
    write_dump(path, batch())
    original = direct_frames.read_frames
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        write_dump(path, batch(), products={"events": {"extra": np.arange(len(KEYS))}})
        return result
    monkeypatch.setattr(direct_frames, "read_frames", changed)
    cache = FrameSourceCache()
    with pytest.raises(ValueError, match="source changed"):
        cache.read({"path": str(path)})
    assert cache.hits == cache.misses == 0
    monkeypatch.setattr(direct_frames, "read_frames", original)
    cache.read({"path": str(path)})
    assert cache.misses == 1
