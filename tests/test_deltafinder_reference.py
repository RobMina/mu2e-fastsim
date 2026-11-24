"""The literal DeltaFinder transcription (tests/deltafinder_reference.py) without JAX: DeltaCandidate merging."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import deltafinder_reference as ref  # noqa: E402

GEO = {"stationZ": np.linspace(-1500.0, 1500.0, ref.NST).astype(np.float32)}


class _Seed:
    """The DeltaSeed members DeltaCandidate reads."""
    def __init__(self, station, t, nhits=2, nsh=2):
        self.station, self.nhits, self.nsh = station, nhits, nsh
        self.sumedep = np.float32(0.002 * nsh)
        self.snx2, self.snxy, self.sny2 = np.float64(1.0), np.float64(0.25), np.float64(0.5)
        self.snxr, self.snyr = np.float64(200.0), np.float64(150.0)
        self.t = np.float32(t)
        self.delta = -1

    def tmean(self):
        return self.t


def _pair():
    a = ref.Delta(0, _Seed(5, 800.0), GEO)
    a.add_seed(_Seed(6, 801.0))
    b = ref.Delta(1, _Seed(6, 802.0, nhits=3, nsh=4), GEO)
    b.add_seed(_Seed(7, 803.0))
    return a, b


def test_merge_replaces_the_seed_of_a_shared_station():
    """MergeDeltaCandidate with printErrors 0 (the trigger menu): the merged candidate's seed replaces the existing one
    in a station both hold, and it is counted again."""
    a, b = _pair()
    old6, new6 = a.seeds[6], b.seeds[6]
    a.merge(b, 0)
    assert a.seeds[6] is new6 and new6.delta == 0
    assert old6.delta == -1                    # the replaced seed is left as it was
    assert a.nseeds == 4 and a.nhits == 2 + 2 + 3 + 2 and a.nsh == 2 + 2 + 4 + 2
    assert (a.first, a.last) == (5, 7)


def test_merge_keeps_the_seed_when_errors_are_printed():
    a, b = _pair()
    old6 = a.seeds[6]
    a.merge(b, 1)
    assert a.seeds[6] is old6
    assert a.nseeds == 3 and a.nhits == 2 + 2 + 2
