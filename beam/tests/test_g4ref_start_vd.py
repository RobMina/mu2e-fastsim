"""g4table --start-vd: a muon starts at its first crossing of a virtual detector (synthetic rows)."""
import numpy as np

from beamfastsim import g4ref


def _obj(rows):
    out = np.empty(len(rows), dtype=object)
    out[:] = [np.asarray(r) for r in rows]
    return out


def test_start_at_vd_replaces_start_and_drops_non_crossers():
    a = {"evt": np.array([1, 2, 3]), "x": np.zeros(3), "pstop": np.array([0.0, 1.0, 0.0]),
         "isvd": _obj([[False, True], [False, False], [True, True]])}
    for k, b in g4ref.VD_BRANCHES.items():
        a[b] = _obj([[0.0, 10.0], [0.0, 20.0], [5.0, 30.0]])
    out = g4ref.start_at_vd(a, 2)
    assert out["evt"].tolist() == [1, 3]
    assert out["z"].tolist() == [10.0, 30.0] and out["time"].tolist() == [10.0, 30.0]
    assert out["pstop"].tolist() == [0.0, 0.0]
    assert "isvd" not in out and "xvd" not in out
