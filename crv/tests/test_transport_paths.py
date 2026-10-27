"""Saved-table path geometry, truth identity and additive decomposition contracts."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.analyses import transport_paths as P


def box():
    return SimpleNamespace(sector=np.array([0]), sector_names=["S"], center=np.zeros((1, 3)),
                           halflen=np.array([[5., 10., 50.]]), local_to_world=np.array([[0, 1, 2]]))


def test_remaining_box_chord_parallel_and_invalid():
    entry = np.array([[-5, 0, 0], [0, 0, 0], [-5, 0, 0], [-5.02, 0, 0], [-5, 0, 0], [-5.005, 0, 0]])
    direction = np.array([[100, 0, 0], [0, 2, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]])
    chord = P.ray_chord(entry, direction, box().center, box().halflen)
    np.testing.assert_allclose(chord[[0, 1, 5]], [10, 10, 10])
    assert np.isnan(chord[[2, 3, 4]]).all()
    diagonal = P.ray_chord(np.array([[-5, 0, 0]]), np.array([[1, 2, 0]]), box().center, box().halflen)
    assert diagonal[0] == pytest.approx(5 * np.sqrt(5))


def test_face_classification_through_edge_corner_and_interior():
    entry = np.array([[-5., 0, 0], [-5, 0, 0], [-5, 10, 0], [-5, 0, 0], [-5.02, 0, 0]])
    exit = np.array([[5., 0, 0], [0, 10, 0], [5, 0, 0], [0, 0, 0], [5, 0, 0]])
    assert P.classify_faces(entry, exit, box().center, box().halflen, np.zeros(5, int)).tolist() == [
        "thickness_through", "edge_clipped", "ambiguous", "ambiguous", "ambiguous"]


def tables(subrun=1):
    gen = pd.DataFrame(dict(run=1, subrun=subrun, event=[1, 1, 2, 3], idx=[0, 2, 0, 0],
                            pdg=[13, -13, 13, 13], px=1000., py=0., pz=0., x=-20., y=0., z=0., t=0.))
    parts = pd.DataFrame(dict(run=1, subrun=subrun, event=1, key=[11, 99], genidx=[2, 0], isprim=1, pdg=[-13, 13]))
    steps = pd.DataFrame(dict(run=1, subrun=subrun, event=1, sp=[99, 11], bar=0, t0=[0., 1.], t1=[1., 2.],
                              px0=[1000., 2000.], py0=0., pz0=0., x0=-5., y0=0., z0=0., x1=5., y1=0., z1=0.,
                              p1=900., edep=[2., 4.], len=[10., 10.]))
    return gen, parts, steps


def mock_crossings(monkeypatch, parts, steps, gen=None, fast=False):
    if gen is None:
        gen = tables()[0]
    monkeypatch.setattr(P.tables, "read", lambda path, name: {"steps": steps, "particles": parts}[name])
    return P.path_crossings("synthetic", gen, box(), fast)


def test_multiple_primary_identities_and_stopping(monkeypatch):
    gen, parts, steps = tables()
    steps.loc[0, ["p1", "len"]] = [0., 3.]
    x = mock_crossings(monkeypatch, parts, steps)
    assert x.genidx.tolist() == [0, 2]
    assert x.p.tolist() == [1000., 2000.]
    assert x.ray_chord.tolist() == [10., 10.]
    assert x.face_class.tolist() == ["thickness_through"] * 2
    assert x.stopped.tolist() == [True, False]
    assert x.shortened.tolist() == [True, False]
    assert x.clean.tolist() == [False, True]
    assert x.record_excess.tolist() == [-7., 0.]
    parts = parts.drop(columns="genidx")
    parts["key"] = [2, 1]
    steps["sp"] = [1, 2]
    y = mock_crossings(monkeypatch, parts, steps, gen, True)
    assert y.genidx.tolist() == [0, 2]
    with pytest.raises(ValueError, match="require genidx"):
        P.path_crossings("synthetic", gen, box())


@pytest.mark.parametrize("end_time", [float("nan"), float("inf"), -1.])
def test_invalid_end_times_retained_but_not_clean(monkeypatch, end_time):
    gen, parts, steps = tables()
    steps.loc[0, "t1"] = end_time
    x = mock_crossings(monkeypatch, parts, steps)
    assert x.invalid_time.tolist() == [True, False]
    assert x.clean.tolist() == [False, True]
    out = P.decomposition(x, x, gen[P.tables.KEY].drop_duplicates())
    assert out["counts"]["reference"]["invalid_time"] == 1
    json.dumps(out, allow_nan=False)


def test_cli_routes_transport_paths(monkeypatch):
    from crvfastsim import cli
    seen = []
    monkeypatch.setattr(P, "main", lambda argv: seen.extend(argv))
    cli.main(["transport-paths", "--help"])
    assert seen == ["--help"]


def test_multi_record_first_last_and_excess(monkeypatch):
    gen, parts, steps = tables()
    extra = steps.iloc[[0]].copy()
    extra["t0"], extra["t1"], extra["x0"], extra["x1"], extra["len"], extra["p1"] = 3., 4., 0., 5., 6., 123.
    steps.loc[0, ["x1", "len"]] = [0., 5.5]
    x = mock_crossings(monkeypatch, parts, pd.concat([extra, steps], ignore_index=True))
    first = x.iloc[0]
    assert first.nrecords == 2 and first.x0 == -5 and first.x1 == 5 and first.p1 == 123
    assert first.record_chord == 10 and first.record_excess == 1.5 and first.length == 11.5
    assert not first.clean
    assert x.iloc[1].clean


def test_decomposition_ratios_and_empty_finite_json(monkeypatch):
    gen, parts, steps = tables()
    b = mock_crossings(monkeypatch, parts, steps)
    a = b.copy()
    a["length"], a["ray_chord"], a["edep"] = b.length * 1.1, b.ray_chord * 1.05, b.edep * 1.21
    out = P.decomposition(a, b, gen[P.tables.KEY].drop_duplicates())
    assert out["transport"]["length_mean_relative"] == pytest.approx(.1)
    assert out["transport"]["edep_mean_relative"] == pytest.approx(.21)
    assert out["transport"]["energy_per_path_relative"] == pytest.approx(.1)
    assert out["entry_ray_candidate_vs_reference"]["length_mean_relative"] == pytest.approx(.05)
    assert out["candidate_true_vs_entry_ray"]["length_mean_relative"] == pytest.approx(1.1 / 1.05 - 1)
    assert out["reference_true_vs_entry_ray"]["length_mean_relative"] == 0
    assert out["reference_true_vs_record_chords"]["length_mean_relative"] == 0
    assert out["transport"]["moments"]["events"] == 3
    json.dumps(out, allow_nan=False)
    empty = P.decomposition(a.iloc[:0], b, gen[P.tables.KEY].drop_duplicates())
    assert empty["transport"]["length_mean_relative"] is None
    json.dumps(empty, allow_nan=False)


def test_truth_strata_include_incidence_one_and_exclude_stops(monkeypatch):
    gen, parts, steps = tables()
    a = mock_crossings(monkeypatch, parts, steps)
    b = a.copy()
    a.loc[0, "stopped"], a.loc[0, "clean"] = True, False
    a.loc[1, "p"] = 5.  # Stratum is reference momentum, not candidate momentum.
    out = P.populations(a, b, gen[P.tables.KEY].drop_duplicates())
    assert out["all_matched"]["transport"]["a"] == 2
    assert out["both_nonstopping_clean_single_record"]["transport"]["a"] == 1
    assert out["clean_reference_incidence_3"]["transport"]["a"] == 1
    assert out["clean_reference_p_0"]["transport"]["a"] == 0
    assert out["clean_reference_p_4"]["transport"]["a"] == 1


@pytest.fixture
def persisted(tmp_path, monkeypatch):
    counters = tmp_path / "counters.txt"
    counters.write_text("synthetic box")
    rc = SimpleNamespace(geometry=box, counters=str(counters), counters_cfg=lambda: {"crs.sectorTypeS": 0})
    monkeypatch.setattr(P.runconfig, "get", lambda _: rc)
    monkeypatch.setattr(P.runconfig, "sector_type_names", lambda *args: {0: "side"})

    def make(subrun):
        gen, parts, steps = tables(subrun)
        paths = {arm: tmp_path / f"{arm}_{subrun}.root" for arm in ("gen", "reference", "candidate")}
        P.tables.write(str(paths["gen"]), None, None, gen=gen, as_ttree=True)
        for arm in ("reference", "candidate"):
            P.tables.write(str(paths[arm]), steps, parts, gen=gen, as_ttree=True)
        return paths
    return make


def test_run_merge_and_new_paths(persisted, tmp_path):
    paths, audits, rows = [], [], []
    for subrun in (1, 2):
        p = persisted(subrun)
        paths.append(p)
        dest = tmp_path / f"audit_{subrun}.json"
        rows.append(P.run(p["gen"], p["reference"], p["candidate"], "synthetic", 3, dest))
        audits.append(str(dest))
    assert set(rows[0]["sectors"]) == {"S", "all_sectors", "group:side"}
    assert json.loads(Path(audits[0]).read_text()) == rows[0]
    with pytest.raises(FileExistsError):
        P.run(paths[0]["gen"], paths[0]["reference"], paths[0]["candidate"], "synthetic", 3, audits[0])
    out = P.merge(audits, tmp_path / "merged.json")
    stats = out["sectors"]["S"]["all_matched"]["transport"]
    assert out["persisted_generated_events"] == 6 and stats["a"] == 4
    assert out["input_path_census"]["reference"]["crossings"] == 4
    assert out["input_path_census"]["reference"]["invalid_time"] == 0
    assert stats["length_mean_se"] == 0 and stats["energy_per_path_se"] == 0
    assert stats == P.V.merge_counts([r["sectors"]["S"]["all_matched"]["transport"] for r in rows])
    with pytest.raises(ValueError, match="disjoint"):
        P.merge([audits[0], audits[0]], tmp_path / "duplicate.json")
    changed = rows[1].copy()
    changed["settings"] = dict(changed["settings"], face_tolerance_mm=.1)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="incompatible"):
        P.merge([audits[0], str(bad)], tmp_path / "bad_merge.json")
    with pytest.raises(ValueError, match="event count"):
        P.run(paths[0]["gen"], paths[0]["reference"], paths[0]["candidate"], "synthetic", 4, tmp_path / "wrong.json")



def test_paired_decomposition_merge_equals_pooled(monkeypatch):
    gen, parts, steps = tables()
    b = mock_crossings(monkeypatch, parts, steps)
    b2 = b.copy()
    b2["subrun"], b2["length"], b2["edep"] = 2, [9., 8.], [3., 6.]
    a, a2 = b.copy(), b2.copy()
    a["length"], a["edep"] = [11., 12.], [2., 3.]
    a2["length"], a2["edep"] = [13., 9.], [4., 7.]
    u = gen[P.tables.KEY].drop_duplicates()
    u2 = u.assign(subrun=2)
    first, second = P.decomposition(a, b, u), P.decomposition(a2, b2, u2)
    full = P.decomposition(pd.concat([a, a2]), pd.concat([b, b2]), pd.concat([u, u2]))
    for name in ("transport", "entry_ray_candidate_vs_reference", "candidate_true_vs_entry_ray", "reference_true_vs_entry_ray",
                 "reference_true_vs_record_chords"):
        merged = P.V.merge_counts([first[name], second[name]])
        assert merged == full[name]


def test_nonfinite_raw_and_invalid_ray_tagged(monkeypatch):
    gen, parts, steps = tables()
    steps.loc[0, "x0"] = -6.
    x = mock_crossings(monkeypatch, parts, steps)
    assert not x.loc[0, "clean"] and np.isnan(x.loc[0, "ray_chord"])
    out = P.decomposition(x, x, gen[P.tables.KEY].drop_duplicates())
    assert out["valid_ray_pairs"] == 1 and out["transport"]["a"] == 2
    json.dumps(out, allow_nan=False)
    steps.loc[0, "p1"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        mock_crossings(monkeypatch, parts, steps)
