"""Repeated preparation checks full truth without retaining duplicate expanded batches."""
import json

import awkward as ak
import numpy as np
import pytest

from mixing import direct_frames
from mixing.fast_detectors import prepare_frames, save_sampling
from test_mixing_direct_frames import manifest


def test_repeated_preparation_same_choices_and_reference(manifest, tmp_path):
    path, _ = manifest
    baseline = direct_frames.load_manifest(path, max_events=2)
    out = tmp_path / "baseline"
    out.mkdir()
    save_sampling(out, baseline[-1])
    result, report = prepare_frames(path, 2, 3, out)
    assert report["repeats_identical"] and report["sampling_reference_identical"]
    assert len(report["passes"]) == 3 and report["cache_hits"] > 0
    np.testing.assert_array_equal(result[3], baseline[3])
    assert report["exact_check_s"] > 0


def test_repeated_preparation_catches_truth_only_mutation(manifest, monkeypatch):
    path, _ = manifest
    original = direct_frames.load_manifest
    calls = []
    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        if calls:
            parts = result[0].collections["particles"]
            result[0].collections["particles"] = ak.with_field(parts, parts.st + 1, "st")
        calls.append(True)
        return result
    monkeypatch.setattr(direct_frames, "load_manifest", load)
    with pytest.raises(ValueError, match="full flat truth"):
        prepare_frames(path, 2, 2, None)


def test_reference_failure_before_response(manifest, tmp_path):
    path, _ = manifest
    baseline = direct_frames.load_manifest(path, max_events=2)
    out = tmp_path / "baseline"
    out.mkdir()
    save_sampling(out, baseline[-1])
    metadata = json.loads((out / "sampling.json").read_text())
    metadata["seed"] += 1
    (out / "sampling.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="sampling value changed: seed"):
        prepare_frames(path, 2, 1, out)
