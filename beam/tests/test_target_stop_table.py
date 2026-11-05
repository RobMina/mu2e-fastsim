"""Synthetic identity-preserving physical pion stop extraction."""
import numpy as np
import pytest

from beamfastsim import target_stop_table as T
from beamfastsim import geometry as G


def _geom():
    return G.BeamGeometry(0., np.array([7.]), np.array([0.]), np.array([10.]), np.array([0.1]),
                          0., 1., 1., 2., 20., 1., 0., 10., 30., 0., 10.)


def test_extracts_only_physical_pion_stops_and_retains_event_identity(tmp_path):
    uproot = pytest.importorskip("uproot")
    f = tmp_path / "stops.root"
    a = dict(run=np.ones(5, np.int32), subrun=np.ones(5, np.int32), evt=np.arange(5, dtype=np.int32),
             trk=np.arange(5, dtype=np.int32), pdg=np.array([-211, -211, 13, -211, -211], np.int32),
             xstop=np.ones(5), ystop=np.ones(5), zstop=np.full(5, 7.),
             tstop=np.array([350., 349., 500., 400., 450.]), pstop=np.array([0., 0., 0., 1., 0.]),
             codestop=np.array([78, 78, 78, 78, 79], np.int32))
    with uproot.recreate(f) as root:
        root.mktree("cetree/ntpart", {k: v.dtype for k, v in a.items()}).extend(a)
    out, events = T.read([f], _geom())
    assert events == [5]
    assert out["pdg"].tolist() == [-211]
    assert out["evt"].tolist() == [0] and out["trk"].tolist() == [0]
    assert out["foil"].tolist() == [0] and out["at_rest"].tolist() == [True]
    with pytest.raises(ValueError, match="duplicate"):
        T.read([f, f], _geom())
