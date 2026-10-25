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


def test_derived_labels_and_include_provenance(tmp_path):
    from beamfastsim.__main__ import main
    import pytest

    cfg = tmp_path / "base.txt"
    cfg.write_text('''double mu2e.solenoidOffset = 3904;
vector<double> stoppingTarget.radii = {75};
vector<double> stoppingTarget.halfThicknesses = {0.05};
double stoppingTarget.holeRadius = 20;
double stoppingTarget.deltaZ = 20;
double stoppingTarget.z0InMu2e = 6000;
double protonabsorber.distFromTargetEnd = -500;
double protonabsorber.OutRadius0 = 300;
double protonabsorber.thickness = 0.5;
double protonabsorber.halfLength = 1000;
double protonabsorber.outerPAInnerRadius0 = 400;
double protonabsorber.outerPAZCenter = 6000;
double protonabsorber.outerPAHalfLength = 1000;
double tsda.z0 = 4500;
double tsda.halfLength4 = 25;
double tsda.rin = 300;
double tsda.r4 = 500;
''')
    overlay = tmp_path / "overlay.txt"
    overlay.write_text('#include "base.txt"\n')
    path = tmp_path / "reference.root"
    a = _rows(3)
    a["xstop"][:] = [-3874, -3904, -3874]
    a["zstop"][:] = 6000
    a["pstop"][-1] = 10
    a["pdg"][-1] = -13
    with uproot.recreate(path) as root:
        root.mktree("mutree/ntpart", {b: "float64" for b in a}).extend(a)
    table = tmp_path / "g4.npz"
    saved = dict(a, q=np.array([-1., -1., 1.]), at_rest=np.array([True, True, False]),
                 foil=np.array([0, -1, -1]))
    g4ref.save(saved, table)
    report = reference_audit.audit([path], table=table, geom_file=str(overlay), offline_root=str(tmp_path),
                                   baseline_geom=str(cfg))
    assert report["table"]["all_derived_columns_match"]
    assert report["geometry"]["allowed_overlay_changes"] == []
    assert report["geometry"]["foil_profile"] == [1]
    assert report["geometry"]["at_rest"] == 2
    sources = report["geometry"]["sources"]
    assert [s["path"] for s in sources] == [str(overlay.resolve()), str(cfg.resolve())]
    assert all(len(s["sha256"]) == 64 for s in sources)
    saved["foil"][0] = -1
    g4ref.save(saved, table)
    with pytest.raises(SystemExit) as exc:
        main(["reference-audit", "--nts", str(path), "--g4", str(table), "--geom", str(overlay),
              "--offline-root", str(tmp_path), "--out", str(tmp_path / "audit.json")])
    assert exc.value.code == 1
