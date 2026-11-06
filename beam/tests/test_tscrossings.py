"""Synthetic bounded stock extraction CLI regressions; no private ROOT inputs."""
import json

import numpy as np
import pytest
import uproot

from beamfastsim import tscrossings as ts


TARGET = np.array(ts.TARGETS[0])


def rows(times, sid=None):
    n = len(times)
    a = {k: np.full(n, v, dtype="f4") for k, v in zip(ts.IDS, TARGET)}
    a.update({k: np.full(n, v, dtype="f4") for k, v in dict(pdg=-13, code=14,
               x=3900, y=1, z=-4004, px=3, py=4, pz=50, ke=10).items()})
    a["time"] = np.asarray(times, dtype="f4")
    a["sid"] = np.asarray([1] * n if sid is None else sid, dtype="f4")
    return a


def fixture(tmp_path):
    vd = rows([20, 10, 15, 15, 9], [2, 1, 91, 91, 1])
    vd["pz"][3] = -50
    vd["evt"][4] += 1
    ds = rows([30], [0])
    root = tmp_path / "shards" / "075" / "stock.root"
    root.parent.mkdir(parents=True)
    with uproot.recreate(root) as f:
        f.mktree("inspect/ntvd", rows([999]))
        f.mktree("inspect/ntvd", vd)
        f.mktree("inspect/nttvd", ds)
        f.mktree("inspect/ntpart", {"must_not_read": np.array([42])})
    census = tmp_path / "census-75.json"
    c = dict(index=75, cohort="development", diagnostic_mode="compact", checks_passed=True,
             ntuple=str(root), vd_rows=len(vd["time"]), ds_beam_all_species_rows=1)
    census.write_text(json.dumps(c))
    ref = tmp_path / "reference.npz"
    a = dict(id=TARGET[None], stock_id=TARGET[None], endpoint_id=TARGET[None],
             pdg=np.array([-13]), stock_pdg=np.array([-13]), stock_reached=np.array([True]),
             source_parent_ancestry_incomplete=np.array([True]), endpoint_ancestry_incomplete=np.array([False]))
    for prefix, table, index in (("", vd, 1), ("stock_", ds, 0)):
        a[prefix + "x"] = np.array([[table[k][index] for k in ("x", "y", "z")]])
        a[prefix + "p"] = np.array([[table[k][index] for k in ("px", "py", "pz")]])
        a[prefix + "t"] = table["time"][[index]]
    np.savez(ref, **a)
    return census, ref, a


def cli(census, ref, out):
    return ["--census", str(census), "--reference", str(ref), "--id", *map(str, TARGET), "--out", str(out)]


def test_cli_exact_ids_highest_cycle_all_crossings_and_provenance(tmp_path, monkeypatch):
    census, ref, _ = fixture(tmp_path)
    out = tmp_path / "fresh" / "crossings.npz"
    original = ts.read_table
    calls = []

    def read(path, tree):
        assert tree in ("ntvd", "nttvd")
        calls.append(tree)
        return original(path, tree)

    monkeypatch.setattr(ts, "read_table", read)
    ts.main(cli(census, ref, out))
    assert calls == ["ntvd", "nttvd"]
    with np.load(out, allow_pickle=False) as a:
        np.testing.assert_array_equal(a["vd_time"], [10, 15, 15, 20])
        np.testing.assert_array_equal(a["vd_entry"], [1, 2, 3, 0])
        np.testing.assert_array_equal(a["vd_sid"], [1, 91, 91, 2])
        np.testing.assert_array_equal(a["timeline_time"], [10, 15, 15, 20, 30])
        np.testing.assert_array_equal(a["timeline_kind"], [0, 0, 0, 0, 1])
        assert a["vd_pz"][2] < 0
        assert a["vd_momentum_mev_c"].dtype == np.dtype("f8")
        assert a["reference_source_parent_ancestry_incomplete"][0]
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["counts"] == {"vd": [4], "ds": [1]}
    assert report["inputs"][0]["trees"]["vd"]["tree"] == "inspect/ntvd;2"
    assert report["ntpart_rows_read"] == 0
    before = out.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        ts.main(cli(census, ref, out))
    assert out.read_bytes() == before


def test_repeated_ds_kept_while_validating_exact_reference_sample(tmp_path):
    census, ref, _ = fixture(tmp_path)
    c = json.loads(census.read_text())
    with uproot.update(c['ntuple']) as f:
        f.mktree('inspect/nttvd', rows([31, 30], [0, 0]))
    c['ds_beam_all_species_rows'] = 2
    census.write_text(json.dumps(c))
    out = tmp_path / 'repeat.npz'
    ts.main(cli(census, ref, out))
    with np.load(out) as a:
        np.testing.assert_array_equal(a['ds_time'], [30, 31])
        assert np.sum(a['timeline_kind'] == 1) == 2


def test_reference_only_no_root_reads(tmp_path, monkeypatch, capsys):
    _, ref, _ = fixture(tmp_path)
    monkeypatch.setattr(ts, "read_small", lambda *args: pytest.fail("unexpected ROOT read"))
    ts.main(["--reference", str(ref), "--id", *map(str, TARGET), "--check-reference"])
    assert json.loads(capsys.readouterr().out)["reference"]["same_source_endpoint_id"]


@pytest.mark.parametrize("bad", [((1430, 920100, 1, 1),), ((1430, 920075, 4321, 1156),),
                                  (ts.TARGETS[0], ts.TARGETS[0]), ()])
def test_reject_unauthorized_duplicate_or_empty_ids(bad):
    with pytest.raises(ValueError):
        ts.targets_checked(bad)


@pytest.mark.parametrize("field,value", [("index", 100), ("cohort", "heldout"),
                                         ("diagnostic_mode", "full"), ("checks_passed", False)])
def test_census_guard_before_root_access(tmp_path, monkeypatch, field, value):
    census, ref, _ = fixture(tmp_path)
    c = json.loads(census.read_text())
    c[field] = value
    census.write_text(json.dumps(c))
    monkeypatch.setattr(ts, "read_small", lambda *args: pytest.fail("unexpected ROOT read"))
    with pytest.raises(ValueError, match="development census"):
        ts.main(cli(census, ref, tmp_path / "out.npz"))


def test_float_id_precision_and_species_guard():
    a = rows([1])
    a["evt"][0] = 2**24
    with pytest.raises(ValueError, match="precision"):
        ts.select_exact(a, np.array(ts.TARGETS))
    a = rows([1])
    a["pdg"][0] = 13
    with pytest.raises(ValueError, match="PDG"):
        ts.select_exact(a, np.array(ts.TARGETS))


def test_no_spatial_fallback_and_missing_ds(tmp_path):
    census, ref, _ = fixture(tmp_path)
    c = json.loads(census.read_text())
    ds = rows([30])
    ds["trk"] += 1
    with uproot.update(c["ntuple"]) as f:
        f.mktree("inspect/nttvd", ds)
    with pytest.raises(ValueError, match="missing recorded"):
        ts.main(cli(census, ref, tmp_path / "out.npz"))
    assert not (tmp_path / "out.npz").exists()


def test_row_budget_checked_before_arrays(tmp_path, monkeypatch):
    census, ref, _ = fixture(tmp_path)
    monkeypatch.setattr(ts, "MAX_TABLE_ROWS", 4)
    monkeypatch.setattr(ts, "read_table", lambda *args: pytest.fail("unexpected array read"))
    with pytest.raises(ValueError, match="row budget"):
        ts.main(cli(census, ref, tmp_path / "out.npz"))


def test_reference_mismatch_rejected(tmp_path):
    census, ref, a = fixture(tmp_path)
    a["stock_id"] = a["stock_id"].copy()
    a["stock_id"][0, 3] += 1
    np.savez(ref, **a)
    with pytest.raises(ValueError, match="same exact"):
        ts.main(cli(census, ref, tmp_path / "out.npz"))


def test_cli_requires_output_and_census(tmp_path):
    with pytest.raises(SystemExit):
        ts.main(["--reference", str(tmp_path / "absent.npz")])
