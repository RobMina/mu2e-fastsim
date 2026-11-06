"""Finished native-digi summary counts actual entries and aligns both detectors."""
import awkward as ak
import numpy as np
import pytest
import uproot

from mixing.inspect import summarize_native


def fixture(path, mismatch=False, mc=True):
    ids = {"run": np.array([1, 1]), "subrun": np.array([2, 2]), "event": np.array([3, 4])}
    with uproot.recreate(path) as root:
        root.mktree("trkdigi/events", ids | {"d_sid": ak.Array([[1, 2, 2], []])})
        root.mktree("crvdump/events", ids | ({"event": np.array([3, 5])} if mismatch else {}))
        for name, n in (("charges", 8), ("digimc", 2), ("digis", 2), ("pulses", 0), ("clusters", 0)):
            if mc or name not in ("charges", "digimc"):
                root.mktree("crvdump/" + name, {"bar": np.zeros(n, dtype=np.int32)})


def test_native_summary(tmp_path):
    path = tmp_path / "native.root"
    fixture(path)
    result = summarize_native(path)
    assert result == {"events": 2, "tracker_digis": 3, "tracker_empty_events": 1,
                      "crv_charges": 8, "crv_digimc": 2, "crv_digis": 2,
                      "crv_pulses": 0, "crv_clusters": 0}


def test_native_saved_digis_without_unsaved_mc(tmp_path):
    path = tmp_path / "saved_digis.root"
    fixture(path, mc=False)
    result = summarize_native(path)
    assert result["crv_charges"] is None and result["crv_digimc"] is None
    assert result["crv_digis"] == 2


def test_native_rejects_unpaired_detector_events(tmp_path):
    path = tmp_path / "mismatch.root"
    fixture(path, mismatch=True)
    with pytest.raises(ValueError, match="identities differ"):
        summarize_native(path)
