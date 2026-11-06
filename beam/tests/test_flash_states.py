"""Read flat stock source states, preserving every event in the sample index."""
import numpy as np
import pytest
import uproot

from beamfastsim.flash_states import event_index, extract

TEXT = """Hits for BeamFilter:Beam in 1430:20000:1: (2)
 trackId: (1:2, 100) volumeId: 0 pdgId: 11
 trackId: (1:2, 101) volumeId: 0 pdgId: 22
Hits for BeamFilter:Beam in run: 1430 subRun: 20000 event: 2: (0)
Hits for BeamFilter:Beam in 1430:20001:1: (1)
 trackId: (1:2, 100) volumeId: 0 pdgId: -11
"""


def test_integer_event_index_counts_empty_events():
    index, events = event_index(TEXT)
    np.testing.assert_array_equal(events, [[1430, 20000, 1], [1430, 20000, 2], [1430, 20001, 1]])
    np.testing.assert_array_equal(index["particle_key"], [100, 101, 100])
    np.testing.assert_array_equal(index["subrun"], [20000, 20000, 20001])
    assert all(v.dtype.kind in "iu" for v in index.values())


@pytest.mark.parametrize("text", [TEXT + TEXT, TEXT.replace("(2)", "(3)"), "", TEXT.replace("pdgId", "bad")])
def test_malformed_counts_are_not_silently_accepted(text):
    with pytest.raises(ValueError):
        event_index(text)


def test_extract_float_kinematics_double_times_and_zero_events(tmp_path):
    path, log, out = tmp_path / "hits.root", tmp_path / "read.log", tmp_path / "source.npz"
    log.write_text(TEXT)
    cols = {"hits/" + k: np.array([0., 1., 2.], dtype="f4") for k in ("x", "y", "z", "px", "py", "pz")}
    times = np.array([1e16 + 2, 10., 500.], dtype="f8")
    cols.update({"hits/particleId": np.array([100, 101, 100], dtype="u4"),
                 "hits/pdgId": np.array([11, 22, -11], dtype="i4"),
                 "hits/volumeCopy": np.zeros(3, dtype="u4"),
                 "hits/time": times.astype("f4"), "particle/time_": times})
    with uproot.recreate(path) as root:
        root.mktree("input/nt", cols)
    extract(path, log, out)
    with np.load(out) as saved:
        np.testing.assert_array_equal(saved["time_ns"], times[[0, 2]])
        np.testing.assert_array_equal(saved["pdg"], [11, -11])
        np.testing.assert_array_equal(saved["subrun"], [20000, 20001])
    with np.load(tmp_path / "source_events.npz") as saved:
        assert saved["event"].shape == (3, 3)
