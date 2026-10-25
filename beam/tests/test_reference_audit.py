"""Reference audit uses highest persisted cycles and compares raw columns in row order."""
import numpy as np
import uproot

from beamfastsim import g4ref, reference_audit


def _rows(n):
    a = {b: np.zeros(n, dtype=np.float64) for b in g4ref.BRANCHES}
    a.update(run=np.ones(n), subrun=np.zeros(n), evt=np.arange(1, n + 1, dtype=float),
             trk=np.ones(n), code=np.full(n, 56.0), pdg=np.full(n, 13.0))
    return a


def test_highest_cycle_and_table_identity(tmp_path):
    path = tmp_path / "reference.root"
    a = _rows(3)
    with uproot.recreate(path) as root:
        root.mktree("mutree/ntpart", {b: "float64" for b in a}).extend(_rows(1))
    with uproot.update(path) as root:
        root.mktree("mutree/ntpart", {b: "float64" for b in a}).extend(a)
    table = tmp_path / "g4.npz"
    g4ref.save(a, table)
    report = reference_audit.audit([path], table=table)
    assert report["files"][0]["selected_key"] == "mutree/ntpart;2"
    assert report["files"][0]["cycles"] == {"1": 1, "2": 3}
    assert report["represented_events"] == report["primary_muons"] == 3
    assert report["duplicate_primary_ids"] == 0
    assert report["table"]["all_raw_columns_match"]
    a["px"][0] = 1.0
    g4ref.save(a, table)
    report = reference_audit.audit([path], table=table)
    assert not report["table"]["all_raw_columns_match"]
    assert not report["table"]["raw_columns_match"]["px"]


def test_duplicates_and_secondary_rows(tmp_path):
    path = tmp_path / "reference.root"
    a = _rows(3)
    a["evt"][:] = [1, 1, 2]
    a["code"][-1] = 99
    with uproot.recreate(path) as root:
        root.mktree("mutree/ntpart", {b: "float64" for b in a}).extend(a)
    report = reference_audit.audit([path])
    assert report["represented_events"] == 2
    assert report["primary_events"] == 1
    assert report["primary_muons"] == 2
    assert report["duplicate_primary_ids"] == 1
