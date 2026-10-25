"""Counting-error and truth identity contracts for transport physics diagnostics."""
import numpy as np
import pandas as pd
import uproot

from crvfastsim.analyses import transport_validation as V


def test_paired_error_includes_empty_events():
    universe = pd.DataFrame(dict(run=1, subrun=1, event=np.arange(4)))
    a = universe.iloc[[0, 0, 1]].copy()
    b = universe.iloc[[0, 1]].copy()
    out = V.paired_count(a, b, universe)
    assert out["relative"] == .5
    np.testing.assert_allclose(out["se"], np.sqrt(4 * .5 / 3) / 2)
    assert V.paired_count(a, a, universe)["se"] == 0.
    assert V.paired_count(a, b.iloc[:0], universe)["relative"] is None


def test_count_moments_merge_equals_full_sample():
    universe = pd.DataFrame(dict(run=1, subrun=1, event=np.arange(6)))
    a = universe.iloc[[0, 0, 1, 4]].copy()
    b = universe.iloc[[0, 1, 4, 5]].copy()
    full = V.paired_count(a, b, universe)
    first = V.paired_count(a[a.event < 3], b[b.event < 3], universe.iloc[:3])
    last = V.paired_count(a[a.event >= 3], b[b.event >= 3], universe.iloc[3:])
    assert V.merge_counts([first, last]) == full


def test_merge_rejects_different_diagnostics_or_bins(tmp_path):
    import json
    import pytest
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    row = {"run_config": "synthetic", "diagnostic_sha256": "one", "p_bin_edges_MeV": [0, 300, None]}
    a.write_text(json.dumps(row))
    b.write_text(json.dumps(dict(row, diagnostic_sha256="two")))
    with pytest.raises(ValueError, match="incompatible diagnostic"):
        V.merge([str(a), str(b)], tmp_path / "out.json")
    b.write_text(json.dumps(dict(row, p_bin_edges_MeV=[0, 600, None])))
    with pytest.raises(ValueError, match="momentum bins"):
        V.merge([str(a), str(b)], tmp_path / "out.json")


def test_cli_routes_transport_validation(monkeypatch):
    from crvfastsim import cli
    seen = []
    monkeypatch.setattr(V, "main", lambda argv: seen.extend(argv))
    cli.main(["transport-validation", "--summary", "example.json"])
    assert seen == ["--summary", "example.json"]


def test_highest_cycle_inventory(tmp_path):
    p = tmp_path / "tables.root"
    with uproot.recreate(p) as f:
        f.mktree("dump/gen", {k: "int32" for k in V.tables.KEY})
        f["dump/gen"].extend(dict(run=[1], subrun=[2], event=[3]))
    with uproot.update(p) as f:
        f.mktree("dump/gen", {k: "int32" for k in V.tables.KEY})
        f["dump/gen"].extend(dict(run=[1, 1], subrun=[2, 2], event=[3, 4]))
    r = V.audit(p)["trees"]["gen"]
    assert r == dict(cycles=[1, 2], selected_cycle=2, rows=2, unique_ids=2)


def test_multiple_primary_muons_remain_distinct(monkeypatch):
    gen = pd.DataFrame(dict(run=1, subrun=1, event=1, idx=[0, 2], pdg=[13, -13]))
    parts = pd.DataFrame(dict(run=1, subrun=1, event=1, key=[1, 2], isprim=1, pdg=[13, -13]))
    steps = pd.DataFrame(dict(run=1, subrun=1, event=1, sp=[1, 2], bar=0, t0=[0., 1.],
                              px0=0., py0=[-100., -200.], pz0=0., edep=2., len=10.))
    monkeypatch.setattr(V.tables, "read", lambda path, name: steps if name == "steps" else parts)
    from types import SimpleNamespace
    x = V.primary_crossings("synthetic", gen, SimpleNamespace(sector=np.array([0])), fast_format=True)
    assert len(x) == 2 and x.genidx.tolist() == [0, 2]
    np.testing.assert_array_equal(x.p, [100., 200.])
    import pytest
    with pytest.raises(ValueError, match="require genidx"):
        V.primary_crossings("synthetic", gen, SimpleNamespace(sector=np.array([0])))
    parts["key"] = [7, 99]
    parts["genidx"] = [2, 0]
    steps["sp"] = [99, 7]
    x = V.primary_crossings("synthetic", gen, SimpleNamespace(sector=np.array([0])))
    assert x.genidx.tolist() == [0, 2]
    np.testing.assert_array_equal(x.p, [100., 200.])
