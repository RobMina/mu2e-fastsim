"""Resident source streaming contracts, using tiny synthetic pools only."""
import gc
import json
import weakref
from unittest.mock import Mock

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from mixing import direct_frames, stream_source
from mixing.counts import json_values
from mixing.frames import FrameBatch, IDS
from mixing.sampling import BunchLaw, CHAINS, STREAMS, draw_multiplicities, select_frames, stock_rates
from mixing.stream_source import StreamBatch, StreamSource
from test_mixing_direct_frames import KEYS, batch, manifest as root_manifest, rows, subset


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    config = {
        "version": 1, "signal": {"path": "signal.root"},
        "pools": {stream: {"path": stream + ".root"} for stream in STREAMS}, "seed": 421,
        "campaign": {"extended_mean": 10., "sdf": .8, "cut_max": 60., "beam_primary": True,
                     "efficiencies": {tag: .6 for chain in CHAINS.values() for tag in chain},
                     "stop_prescale": 3., "max_skip": dict.fromkeys(STREAMS, 19),
                     "mode": "sequential", "postmix_selection": "universal",
                     "additional_time_offset_ns": 0},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(config))
    reader = Mock(side_effect=lambda source: (batch(), {"source_sha256": source["path"],
                                                       "census_sha256": "synthetic"}))
    monkeypatch.setattr(direct_frames, "_read", reader)
    return path, config, reader


def zero_counts(monkeypatch):
    monkeypatch.setattr(stream_source, "draw_multiplicities",
                        lambda rng, intensity, rates: np.zeros((len(intensity), 4), dtype=np.int64))


def test_wrap_original_ids_empty_events_and_global_ordinals(inputs, monkeypatch):
    path, _, reader = inputs
    zero_counts(monkeypatch)
    source = StreamSource(path, max_selected_rows=100)
    assert source.population == 5
    first, second, third = (source.next_batch(size) for size in (3, 3, 5))
    assert isinstance(first, StreamBatch)
    for result, indices, ordinal, index in (
            (first, [0, 1, 2], 0, 0), (second, [3, 4, 0], 3, 1), (third, [1, 2, 3, 4, 0], 6, 2)):
        np.testing.assert_array_equal(result.keys, KEYS[indices])
        np.testing.assert_array_equal(result.provenance["signal_indices"], indices)
        np.testing.assert_array_equal(result.provenance["signal_event_ids"], result.keys)
        assert result.keys.dtype == np.uint64
        assert len(np.unique(result.keys, axis=0)) == len(indices)
        assert (result.ordinal_start, result.batch_index) == (ordinal, index)
        assert not hasattr(result, "mixed") and not hasattr(result, "collections")
    np.testing.assert_array_equal(second.tracker[list(IDS)], KEYS[[3, 0]])
    np.testing.assert_array_equal(second.pdg, [2212, -211, 11, -11])
    assert reader.call_count == 5


def test_exact_batch_rng_domain_shared_intensity_and_cursor_continuation(inputs):
    path, config, reader = inputs
    seed = 272
    source = StreamSource(path, seed=seed, max_selected_rows=100000)
    campaign = config["campaign"]
    law = BunchLaw.from_sdf(campaign["extended_mean"], campaign["sdf"], campaign["cut_max"])
    rates = stock_rates(campaign["efficiencies"], stop_prescale=campaign["stop_prescale"])
    cursors = dict.fromkeys(STREAMS)
    for index, size in enumerate((2, 4, 1)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        intensity = law.draw(rng, size, beam_primary=True)
        counts = draw_multiplicities(rng, intensity, rates)
        result = source.next_batch(size)
        receipt = result.provenance
        np.testing.assert_array_equal(receipt["intensity"], intensity)
        np.testing.assert_array_equal(receipt["multiplicities"], counts)
        assert receipt["actual_rows"] == receipt["estimated_selected_rows"]
        for column, stream in enumerate(STREAMS):
            selection, cursor = select_frames(rng, counts[:, column], 5, cursor=cursors[stream],
                                               max_events_to_skip=campaign["max_skip"][stream])
            saved = receipt["selections"][stream]
            np.testing.assert_array_equal(saved["donors"], selection.donors)
            np.testing.assert_array_equal(saved["counts"], selection.counts)
            np.testing.assert_array_equal(saved["event_index"], selection.event_index)
            np.testing.assert_array_equal(saved["source_event_ids"], KEYS[selection.donors])
            assert saved["next_cursor"] == cursor
            if index:
                assert saved["initial_cursor"] == cursors[stream]
            cursors[stream] = cursor
    assert reader.call_count == 5


def test_repeatable_sequences_and_batch_size_changes_draws(inputs):
    path, _, _ = inputs
    left = StreamSource(path, seed=19, max_selected_rows=100000)
    right = StreamSource(path, seed=19, max_selected_rows=100000)
    for size in (2, 5, 3):
        a, b = left.next_batch(size), right.next_batch(size)
        np.testing.assert_array_equal(a.keys, b.keys)
        pd.testing.assert_frame_equal(a.tracker, b.tracker)
        pd.testing.assert_frame_equal(a.crv, b.crv)
        np.testing.assert_array_equal(a.pdg, b.pdg)
        np.testing.assert_array_equal(a.provenance["multiplicities"], b.provenance["multiplicities"])
        for stream in STREAMS:
            np.testing.assert_array_equal(a.provenance["selections"][stream]["donors"],
                                          b.provenance["selections"][stream]["donors"])
    short = StreamSource(path, seed=19, max_selected_rows=100000).next_batch(2)
    long = StreamSource(path, seed=19, max_selected_rows=100000).next_batch(5)
    assert not np.array_equal(short.provenance["multiplicities"], long.provenance["multiplicities"][:2])


def test_row_guard_counts_repeated_truth_and_optional_calo_before_overlay(inputs, monkeypatch):
    path, config, reader = inputs
    config["campaign"]["max_skip"] = dict.fromkeys(STREAMS, 0)
    path.write_text(json.dumps(config))
    original = batch()
    calo = rows(("sp", "t", "e"), [1, 1, 1, 1, 0], sp=np.full(4, 9))
    reader.side_effect = lambda source: (FrameBatch(original.event_ids, dict(original.collections, calo=calo)), {})
    monkeypatch.setattr(stream_source, "draw_multiplicities",
                        lambda rng, intensity, rates: np.full((len(intensity), 4), 6, dtype=np.int64))
    spy = Mock(wraps=stream_source.overlay)
    monkeypatch.setattr(stream_source, "overlay", spy)
    rejected = StreamSource(path, seed=4, max_selected_rows=146)
    with pytest.raises(ValueError, match="selected rows 147 exceed max_selected_rows=146"):
        rejected.next_batch(1)
    assert spy.call_count == 0
    assert rejected._ordinal == 0 and rejected._batch_index == 0
    assert rejected._cursors == dict.fromkeys(STREAMS)
    accepted = StreamSource(path, seed=4, max_selected_rows=147).next_batch(1)
    assert spy.call_count == 1
    assert accepted.provenance["selected_rows_total"] == 147
    assert accepted.provenance["actual_rows"] == {"particles": 42, "gen": 21, "tracker": 21,
                                                  "crv": 42, "calo": 21}


def test_real_validated_snapshots_not_reread_and_sampling_saves(root_manifest, monkeypatch, tmp_path):
    from mixing.fast_detectors import save_sampling

    path, _ = root_manifest
    reader = Mock(wraps=direct_frames._read)
    monkeypatch.setattr(direct_frames, "_read", reader)
    source = StreamSource(path, seed=17, max_selected_rows=100000)
    assert reader.call_count == 5
    json.dumps(json_values(source.metadata), allow_nan=False)
    assert source.metadata["manifest_seed"] == 421 and source.metadata["seed"] == 17
    assert set(source.metadata["source_fingerprints"]["pools"]) == set(STREAMS)
    (path.parent / "data/signal.root").unlink()
    (path.parent / "data/pool.root").unlink()
    monkeypatch.setattr(direct_frames, "_sha256", Mock(side_effect=AssertionError("disk hash in batch")))
    result = source.next_batch(3)
    source.next_batch(4)
    assert reader.call_count == 5
    save_sampling(tmp_path, result.provenance)
    saved = json.loads((tmp_path / "sampling.json").read_text())
    assert saved["source_metadata_ref"] == "StreamSource.metadata"
    with np.load(tmp_path / "sampling.npz") as arrays:
        np.testing.assert_array_equal(arrays["sampling/signal_event_ids"], result.keys)
        np.testing.assert_array_equal(arrays["sampling/signal_indices"], [0, 1, 2])


def test_mixed_truth_and_returned_batches_not_retained_and_checks_are_fresh(inputs, monkeypatch):
    path, _, _ = inputs
    source = StreamSource(path, max_selected_rows=100000)
    truth_refs = []
    original_overlay = stream_source.overlay
    def record_truth(*args, **kwargs):
        mixed = original_overlay(*args, **kwargs)
        truth_refs.append(weakref.ref(mixed))
        return mixed
    monkeypatch.setattr(stream_source, "overlay", record_truth)
    result = source.next_batch(2)
    result_ref = weakref.ref(result)
    saved_tracker = result.tracker.copy(deep=True)
    source.next_batch(2)
    pd.testing.assert_frame_equal(result.tracker, saved_tracker)
    del result
    gc.collect()
    assert result_ref() is None and all(reference() is None for reference in truth_refs)
    pool = source._pools[STREAMS[0]]
    pool.collections["tracker"] = ak.with_field(pool.collections["tracker"], pool.collections["tracker"].sp + 1, "sp")
    with pytest.raises(ValueError, match="dangling tracker"):
        source.next_batch(1)
    assert source._batch_index == 2 and source._ordinal == 4


@pytest.mark.parametrize("value", [0, -1, 6, True, 1.5, "2", None, np.bool_(True)])
def test_invalid_batch_counts_do_not_advance(inputs, value):
    source = StreamSource(inputs[0], max_selected_rows=1000)
    with pytest.raises(ValueError, match="count"):
        source.next_batch(value)
    assert source.next_batch(np.int64(1)).batch_index == 0


@pytest.mark.parametrize("option,value", [("seed", -1), ("seed", True), ("seed", 1.5), ("seed", None),
                                         ("max_selected_rows", 0), ("max_selected_rows", -1),
                                         ("max_selected_rows", True), ("max_selected_rows", 4.5)])
def test_invalid_constructor_arguments_before_read(inputs, option, value):
    path, _, reader = inputs
    options = {"seed": 0, "max_selected_rows": 1000, option: value}
    with pytest.raises(ValueError, match=option):
        StreamSource(path, **options)
    reader.assert_not_called()


@pytest.mark.parametrize("section,key,value", [(None, "seed", -1), (None, "version", 2),
                                              ("campaign", "mode", "randomReplace"),
                                              ("campaign", "beam_primary", False),
                                              ("campaign", "additional_time_offset_ns", 1),
                                              ("campaign", "postmix_selection", "filtered")])
def test_shared_manifest_validation_before_read(inputs, section, key, value):
    path, config, reader = inputs
    (config if section is None else config[section])[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        StreamSource(path, max_selected_rows=1000)
    reader.assert_not_called()


@pytest.mark.parametrize("empty_source", ["signal", STREAMS[0]])
def test_empty_census_rejected_after_all_five_sources_read(inputs, empty_source):
    path, _, reader = inputs
    reader.side_effect = lambda spec: (subset(batch(), slice(0, 0)) if
                                      spec["path"].endswith(empty_source + ".root") else batch(), {})
    with pytest.raises(ValueError, match="nonempty event census"):
        StreamSource(path, max_selected_rows=1000)
    assert reader.call_count == 5


def test_all_empty_events_are_valid_and_skip_is_not_redrawn(inputs, monkeypatch):
    path, _, reader = inputs
    reader.side_effect = lambda spec: (subset(batch(), slice(4, 5)), {})
    zero_counts(monkeypatch)
    source = StreamSource(path, max_selected_rows=1)
    selector = Mock(wraps=stream_source.select_frames)
    monkeypatch.setattr(stream_source, "select_frames", selector)
    for index in range(3):
        result = source.next_batch(1)
        np.testing.assert_array_equal(result.keys, KEYS[4:])
        assert result.tracker.empty and result.crv.empty and result.pdg.size == 0
        assert result.provenance["selected_rows_total"] == 0
        assert result.batch_index == index
    assert [call.kwargs["cursor"] for call in selector.call_args_list] == [None] * 4 + [0] * 8
