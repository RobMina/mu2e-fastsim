"""Saved-output diagnosis regressions with synthetic arrays, no transport/raw data."""
import json

import numpy as np
import pytest

from beamfastsim import tsdiagnose as ts


def tables():
    n = 6
    ids = np.column_stack((np.full(n, 1430), np.full(n, 920000),
                           np.arange(n), np.arange(n, dtype=np.int64) + 2**53 + 1))
    x = np.arange(n * 3, dtype=float).reshape(n, 3)
    p = np.tile([3., 4., 12.], (n, 1))
    reached = np.array([1, 1, 0, 1, 1, 0], dtype=bool)
    r = dict(id=ids, shard=ids[:, 1].copy(), pdg=np.array([-13, 211, -13, 13, -211, 13]),
             x=x, p=p, t=np.arange(n, dtype=float), stock_reached=reached,
             stock_pdg=np.array([-13, -13, 0, 13, 13, 0]),
             stock_x=np.where(reached[:, None], x + 100., np.nan),
             stock_p=np.where(reached[:, None], p, np.nan),
             stock_t=np.where(reached, np.arange(n) + 10., np.nan),
             source_parent_ancestry_incomplete=np.array([0, 1, 1, 0, 1, 0], dtype=bool),
             event_id=ids[:, :3].copy(), parent_shard_ids=np.array([[1430, 920000]]),
             parent_shard_event_counts=np.array([n]))
    f = dict(id=ids.copy(), shard=ids[:, 1].copy(), pdg=np.array([-13, 211, -13, 13, 13, 13]),
             x=x + 200., p=p.copy(), t=np.arange(n) + 20., fate=np.array([1, 4, 1, 4, 1, 5]),
             path=np.arange(n) + 300., material_path=np.arange(n) + .5,
             pion_decay=np.array([0, 0, 0, 0, 1, 0], dtype=bool), steps=np.arange(n) + 10)
    return r, f


def test_contingency_muplus_overlap_and_source_pdg():
    out, rows = ts.diagnose(*tables())
    c = out["source_groups"]["all"]
    assert [c[k] for k in ("ninput", "stock", "fast", "both", "stock_only", "fast_only", "neither")] == [6, 4, 3, 2, 2, 1, 1]
    assert c["stock_ds_by_fast_fate"] == [[0, 1, 0, 0, 0, 1, 0, 0], [0, 2, 0, 0, 2, 0, 0, 0]]
    m = out["muons"]["mu+"]
    assert [m[k] for k in ("stock", "fast", "both", "stock_only", "fast_only")] == [2, 2, 1, 1, 1]
    assert m["stock_only_fast_fates"] == [0, 0, 0, 0, 1, 0, 0, 0]
    assert m["source_groups"]["pdg_211"]["stock_only"] == 1
    assert out["source_groups"]["charge_1"]["ninput"] == 3
    assert np.all(rows["source_pmag"] == 13.)
    assert np.all(rows["fast_elapsed"] == 20.)
    assert rows["fast_material_path"][1] == 1.5
    json.dumps(out, allow_nan=False)


def test_exact_join_reorders_without_using_positions():
    r, f = tables()
    expected, expected_rows = ts.diagnose(r, f)
    permutation = np.array([5, 1, 4, 0, 3, 2])
    f = {k: v[permutation] for k, v in f.items()}
    actual, rows = ts.diagnose(r, f)
    assert actual == expected
    for key in rows:
        np.testing.assert_equal(rows[key], expected_rows[key])


@pytest.mark.parametrize("column", [0, 1, 2, 3])
def test_each_exact_id_component_required(column):
    r, f = tables()
    f["id"][0, column] += 100
    with pytest.raises(ValueError, match="IDs differ"):
        ts.diagnose(r, f)


@pytest.mark.parametrize("arm", [0, 1])
def test_duplicate_ids_rejected(arm):
    pair = tables()
    pair[arm]["id"][1] = pair[arm]["id"][0]
    with pytest.raises(ValueError, match="duplicate IDs"):
        ts.diagnose(*pair)


def test_missing_rows_and_shard_mismatch_rejected():
    r, f = tables()
    with pytest.raises(ValueError, match="complete ID cohort"):
        ts.diagnose(r, {k: v[:-1] for k, v in f.items()})
    f["shard"][0] += 1
    with pytest.raises(ValueError, match="shard mismatch"):
        ts.diagnose(r, f)


def test_gap_association_and_missing_or_constant_flags():
    r, f = tables()
    out, _ = ts.diagnose(r, f)
    c = out["ancestry"]["source_groups"]["all"]["fast_ds"]
    assert c["gap_by_survival"] == [[2, 1], [1, 2]]
    assert c["difference_gap_minus_no_gap"] == pytest.approx(1 / 3)
    assert c["phi"] == pytest.approx(1 / 3)
    r["source_parent_ancestry_incomplete"][:] = False
    out, _ = ts.diagnose(r, f)
    c = out["ancestry"]["source_groups"]["all"]["fast_ds"]
    assert c["phi"] is c["survival_fraction_gap"] is c["difference_gap_minus_no_gap"] is None
    del r["source_parent_ancestry_incomplete"]
    out, rows = ts.diagnose(r, f)
    assert out["ancestry"] is None and np.all(rows["ancestry_gap"] == -1)


def test_different_ds_species_is_not_terminal_loss():
    r, f = tables()
    f["fate"][1] = 1
    out, rows = ts.diagnose(r, f)
    assert out["muons"]["mu+"]["stock_only"] == 1
    assert out["muons"]["mu+"]["stock_only_fast_fates"][1] == 1
    assert not (rows["stock_reached"] & (rows["fast_fate"] != 1))[1]


@pytest.mark.parametrize("key,value,match", [("fate", 20, "unknown fast fate"),
    ("path", -1., "invalid fast path"), ("material_path", np.nan, "invalid fast material_path"),
    ("p", np.nan, "nonfinite fast terminal p")])
def test_invalid_terminal_diagnostics(key, value, match):
    r, f = tables()
    f[key][1] = value
    with pytest.raises(ValueError, match=match):
        ts.diagnose(r, f)


def test_unresolved_ds_ancestry_not_counted_as_loss():
    r, f = tables()
    r["endpoint_ancestry_incomplete"] = np.array([True])
    with pytest.raises(ValueError, match="unresolved DS ancestry"):
        ts.diagnose(r, f)


def test_saved_event_inventory_checked():
    r, f = tables()
    r["parent_shard_event_counts"][0] += 1
    with pytest.raises(ValueError, match="event IDs and parent counts disagree"):
        ts.diagnose(r, f)


def test_cli_outputs_fresh_directory_and_loss_rows(tmp_path):
    r, f = tables()
    reference, fast, output = tmp_path / "development.npz", tmp_path / "fast.npz", tmp_path / "diagnosis"
    np.savez(reference, **r)
    np.savez(fast, **f)
    args = ["--reference", str(reference), "--fast", str(fast), "--out", str(output)]
    ts.main(args)
    report = json.loads((output / "report.json").read_text())
    assert report["saved_event_count"] == 6
    with np.load(output / "stock-only.npz") as saved:
        np.testing.assert_array_equal(saved["track"], r["id"][[1, 3], 3])
        np.testing.assert_array_equal(saved["fast_xx"], f["x"][[1, 3], 0])
    text = (output / "stock-only.csv").read_text()
    assert str(2**53 + 2) in text and str(2**53 + 4) in text
    with np.load(output / "muplus-union.npz") as saved:
        assert len(saved["track"]) == 3
    assert "not the cause" in (output / "report.md").read_text()
    with pytest.raises(SystemExit):
        ts.main(args)


def test_heldout_name_rejected_before_open(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("held-out path must never be opened")
    monkeypatch.setattr(ts, "_load", forbidden)
    with pytest.raises(SystemExit):
        ts.main(["--reference", "held-out.npz", "--fast", "fast.npz", "--out", str(tmp_path / "new")])


def test_cli_rejects_non_development_subrun(tmp_path):
    r, f = tables()
    r["parent_shard_ids"][0, 1] = 920100
    reference, fast = tmp_path / "reference.npz", tmp_path / "fast.npz"
    np.savez(reference, **r)
    np.savez(fast, **f)
    with pytest.raises(SystemExit):
        ts.main(["--reference", str(reference), "--fast", str(fast), "--out", str(tmp_path / "new")])


def test_terminal_position_materials_supplied_geometry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from beamfastsim import tsgeometry

    r, f = tables()
    # Local x equals global y after translation and rotation; includes boundary.
    R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    t = f["x"][1].copy()
    calls = []

    def contains(local):
        calls.append(local.copy())
        return abs(local[:, 0]) <= 3.

    solid = SimpleNamespace(contains=contains)
    geometry = SimpleNamespace(materials={"B": None, "A": None},
        _matter=[(solid, R, t, "A"), (solid, R, t, "A"), (solid, R, t, "B")])
    # Reorder fast rows: classification must follow the exact-ID join.
    permutation = np.array([5, 1, 4, 0, 3, 2])
    out, rows = ts.diagnose(r, {k: v[permutation] for k, v in f.items()}, geometry)
    np.testing.assert_array_equal(rows["fast_terminal_material_bitmask"], [3, 3, 3, 0, 0, 0])
    assert len(calls) == 3
    np.testing.assert_array_equal(calls[0], (f["x"] - t) @ R)
    m = out["terminal_position_materials"]
    assert m["bit_to_material"] == {"1": "A", "2": "B"}
    assert m["stock_only_muplus_material_bitmasks"] == [3]
    assert m["stock_only_muplus_fast_fate_by_material_bitmask"] == [[0], [0], [0], [0], [1], [0], [0], [0]]
    assert "TERMINAL POSITION" in ts.markdown(out) and "boundaries" in ts.markdown(out)
    json.dumps(out, allow_nan=False)
    r["stock_pdg"][1] = 13  # No missed mu+; empty table stays valid.
    empty, _ = ts.diagnose(r, f, geometry)
    assert empty["terminal_position_materials"]["stock_only_muplus_fast_fate_by_material_bitmask"] == [[]] * len(ts.FATES)
    r["stock_pdg"][1] = -13
    reference, fast, output = tmp_path / "reference.npz", tmp_path / "fast.npz", tmp_path / "out"
    np.savez(reference, **r)
    np.savez(fast, **f)
    monkeypatch.setattr(tsgeometry, "Geometry", lambda path: geometry)
    ts.main(["--reference", str(reference), "--fast", str(fast), "--out", str(output), "--gdml", "synthetic.gdml"])
    with np.load(output / "stock-only.npz") as saved:
        np.testing.assert_array_equal(saved["fast_terminal_material_bitmask"], [3, 0])
    assert json.loads((output / "report.json").read_text())["inputs"]["gdml"].endswith("synthetic.gdml")
    # Preserve full-width material bits and source-track IDs in numeric CSV.
    rows["fast_terminal_material_bitmask"][:] = np.uint64(2**63 + 1)
    ts._csv(tmp_path / "bits.csv", rows, np.ones(6, bool))
    text = (tmp_path / "bits.csv").read_text()
    assert str(2**63 + 1) in text and str(2**53 + 1) in text


def test_exact_join_does_not_permute_same_length_event_metadata():
    r, f = tables()
    permutation = np.array([5, 1, 4, 0, 3, 2])
    f = {k: v[permutation] for k, v in f.items()}
    f['event_id'] = r['event_id'].copy()
    f['parent_shard_ids'] = r['id'][:, :2].copy()
    f['parent_shard_event_counts'] = np.arange(6)
    joined = ts._join(r, f)
    np.testing.assert_array_equal(joined['id'], r['id'])
    for key in ('event_id', 'parent_shard_ids', 'parent_shard_event_counts'):
        np.testing.assert_array_equal(joined[key], f[key])
