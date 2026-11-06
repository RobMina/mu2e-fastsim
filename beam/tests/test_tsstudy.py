"""Bounded driver regressions: synthetic full-reference arrays and mocked transport only."""
import json
import sys
from types import ModuleType

import numpy as np
import pytest

from beamfastsim import tsstudy as ts


def tables():
    n = 6
    ids = np.column_stack((np.full(n, 1430), np.full(n, 920000), np.arange(n), np.arange(n) + 2**53 + 1))
    reached = np.array([True, True, False, True, True, False])
    p = np.tile([3., 4., 12.], (n, 1))
    r = dict(id=ids, shard=ids[:, 1].copy(), x=np.zeros((n, 3)), p=p, t=np.arange(n, dtype=float),
             pdg=np.array([-13, 211, -13, 13, -211, 13]), stock_reached=reached,
             stock_x=np.where(reached[:, None], 1., np.nan) * np.ones((n, 3)),
             stock_p=np.where(reached[:, None], p, np.nan), stock_t=np.where(reached, np.arange(n) + 10., np.nan),
             stock_pdg=np.array([-13, -13, 0, 13, 13, 0]),
             event_id=np.vstack((ids[:, :3], [1430, 920099, 100])),
             parent_shard_ids=np.array([[1430, 920000], [1430, 920099]]), parent_shard_event_counts=np.array([n, 1]))
    return r


def fake_output(source):
    return dict(x=source["x"] + 1, p=source["p"].copy(), t=source["t"] + 20., pdg=np.array([-13, 211, -13]),
                fate=np.array([1, 4, 1]), path=np.array([100., 200., 300.]), material_path=np.array([1., 2., 3.]),
                steps=np.array([10, 20, 30]), pion_decay=np.zeros(3, bool), seconds=np.asarray(1.))


@pytest.fixture
def mock_study(tmp_path, monkeypatch):
    import beamfastsim

    monkeypatch.setattr(ts, "EXPECTED", 3)
    monkeypatch.setattr(ts, "EXPECTED_SHARDS", 2)
    reference = tmp_path / "development.npz"
    np.savez(reference, **tables())
    calls = {"field": [], "geometry": [], "transport": []}
    field, geometry = object(), object()
    modules = {name: ModuleType("beamfastsim." + name) for name in ("tsfield", "tsgeometry", "tstransport")}
    modules["tsfield"].from_config = lambda *args: calls["field"].append(args) or field
    modules["tsgeometry"].Geometry = lambda path: calls["geometry"].append(path) or geometry

    def transport(source, supplied_field, supplied_geometry, **settings):
        assert supplied_field is field and supplied_geometry is geometry
        calls["transport"].append((source, settings))
        result = fake_output(source)
        source["p"][:] = -999.
        return result

    modules["tstransport"].transport = transport
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, "beamfastsim." + name, module)
        monkeypatch.setattr(beamfastsim, name, module, raising=False)
    args = ["--reference", str(reference), "--gdml", "synthetic.gdml", "--field-config", "synthetic.txt",
            "--offline-root", "offline", "--maps", "maps"]
    return args, calls, reference


def test_fixed_preregistered_settings():
    assert ts.SEEDS == (20261111, 20261112, 20261113, 20261114)
    assert ts.EXPECTED == 12832
    assert ts.VARIANTS == {"baseline": (2.5, .25, .05), "half_material": (2.5, .125, .025), "half_step": (1.25, .25, .05)}


def test_all_positive_including_lost_exact_ids_and_full_event_exposure(monkeypatch):
    monkeypatch.setattr(ts, "EXPECTED", 3)
    monkeypatch.setattr(ts, "EXPECTED_SHARDS", 2)
    r = tables()
    selected, stock = ts.cohort(r)
    np.testing.assert_array_equal(selected["id"], r["id"][:3])
    assert selected["id"][0, 3] == 2**53 + 1
    np.testing.assert_array_equal(selected["pdg"], [-13, 211, -13])
    assert not selected["stock_reached"][2]
    for key in ("event_id", "parent_shard_ids", "parent_shard_event_counts"):
        np.testing.assert_array_equal(selected[key], r[key])
    assert len(selected["event_id"]) == 7 and len(selected["id"]) == 3
    assert stock["1"]["-13"] == stock["-1"]["13"] == 2
    r["stock_reached"][:] = False
    second, _ = ts.cohort(r)
    np.testing.assert_array_equal(second["id"], selected["id"])


@pytest.mark.parametrize("key", ["id", "event_id", "parent_shard_ids"])
@pytest.mark.parametrize("subrun", [919999, 920100])
def test_no_heldout_or_other_subruns(monkeypatch, key, subrun):
    monkeypatch.setattr(ts, "EXPECTED", 3)
    monkeypatch.setattr(ts, "EXPECTED_SHARDS", 2)
    r = tables()
    r[key][-1, 1] = subrun
    with pytest.raises(ValueError, match="920000..920099"):
        ts.cohort(r)


@pytest.mark.parametrize("change,match", [
    (lambda r: r.update(p=r["p"][:, :2]), "shape"),
    (lambda r: r["shard"].__setitem__(0, 920001), "shard/event"),
    (lambda r: r["parent_shard_event_counts"].__setitem__(0, 8), "exposure"),
    (lambda r: r["event_id"].__setitem__(1, r["event_id"][0]), "duplicate"),
    (lambda r: r.update(endpoint_ancestry_incomplete=np.array([True])), "unresolved"),
    (lambda r: r["pdg"].__setitem__(2, 13), "ALL 3"),
])
def test_bad_references_rejected(monkeypatch, change, match):
    monkeypatch.setattr(ts, "EXPECTED", 3)
    monkeypatch.setattr(ts, "EXPECTED_SHARDS", 2)
    r = tables()
    change(r)
    with pytest.raises(ValueError, match=match):
        ts.cohort(r)


def test_counts_overlap_histograms_paths_and_empty_shards(monkeypatch):
    monkeypatch.setattr(ts, "EXPECTED", 3)
    monkeypatch.setattr(ts, "EXPECTED_SHARDS", 2)
    r, _ = ts.cohort(tables())
    f = fake_output(r)
    f.update(id=r["id"], shard=r["shard"])
    out = ts.report(r, f)
    assert out["endpoints"]["mu+"]["overlap"] == dict(stock=2, fast=2, both=1, stock_only=1, fast_only=1, neither=0)
    assert out["metrics"]["mu+.stock_only_mean_path_mm"] == 200.
    assert out["metrics"]["mu+.stock_only_mean_material_path_mm"] == 2.
    assert out["metrics"]["pi+.stock_only_mean_path_mm"] is None
    assert out["fates"]["ds"] == 2 and out["fates"]["wall"] == 1
    mu = out["endpoints"]["mu+"]
    assert mu["paired_shards"]["ratio"] == 1.
    assert mu["histograms"]["momentum"]["fast"][1] == 2
    assert mu["histograms"]["elapsed_time"]["stock"][0] == 2
    assert mu["histograms"]["elapsed_time"]["fast"][1] == 2
    f["pdg"][2], f["p"][2] = 211, [400., 0., 0.]
    out = ts.report(r, f)
    assert out["metrics"]["pi+.fast"] == 1
    assert out["endpoints"]["pi+"]["histograms"]["momentum"]["fast_flow"] == [0, 1]
    json.dumps(out, allow_nan=False)


def test_cli_default_seeds_independent_batches_load_once_and_no_overwrite(mock_study, tmp_path):
    args, calls, reference = mock_study
    output = tmp_path / "run"
    command = args + ["--variant", "baseline", "--seeds", "--out", str(output)]
    ts.main(command)
    assert len(calls["field"]) == len(calls["geometry"]) == 1
    assert len(calls["transport"]) == 4
    for seed, (_, settings) in zip(ts.SEEDS, calls["transport"]):
        assert settings == dict(seed=seed, step_mm=2.5, material_step_mm=.25, loss_fraction=.05)
        with np.load(output / f"baseline-{seed}.npz") as f:
            np.testing.assert_array_equal(f["id"], tables()["id"][:3])
            np.testing.assert_array_equal(f["p"], tables()["p"][:3])
            np.testing.assert_array_equal(f["event_id"], tables()["event_id"])
        result = json.loads((output / f"baseline-{seed}.json").read_text())
        assert result["ninput"] == 3 and result["events"] == 7
        assert result["stock_per_source_charge"]["-1"]["13"] == 2
        assert result['inputs']['reference'] == str(reference.resolve())
    with pytest.raises(SystemExit):
        ts.main(command)
    assert len(calls["transport"]) == 4
    with np.load(reference) as r:
        np.testing.assert_array_equal(r["p"], tables()["p"])


@pytest.mark.parametrize("variant", ts.VARIANTS)
def test_explicit_subsets_and_reproducible_settings(mock_study, tmp_path, variant):
    args, calls, _ = mock_study
    for name in ("first", "second"):
        ts.main(args + ["--variant", variant, "--seeds", "20261112", "--out", str(tmp_path / name)])
    assert calls["transport"][0][1] == calls["transport"][1][1]
    assert tuple(calls["transport"][0][1][k] for k in ("step_mm", "material_step_mm", "loss_fraction")) == ts.VARIANTS[variant]
    for extension in ("npz", "json"):
        assert (tmp_path / "first" / f"{variant}-20261112.{extension}").read_bytes() == (tmp_path / "second" / f"{variant}-20261112.{extension}").read_bytes()


@pytest.mark.parametrize("flags", [[], ["--variant", "baseline"], ["--seeds", "20261111"],
    ["--variant", "baseline", "--seeds", "20261110"], ["--variant", "baseline", "--seeds", "20261111", "20261111"]])
def test_explicit_run_guards_before_any_load(mock_study, tmp_path, flags):
    args, calls, _ = mock_study
    with pytest.raises(SystemExit):
        ts.main(args + flags + ["--out", str(tmp_path / "not-created")])
    assert calls == dict(field=[], geometry=[], transport=[])


def test_heldout_filename_never_opened(monkeypatch, tmp_path):
    monkeypatch.setattr(ts, "cohort", lambda *a: pytest.fail("must reject before opening"))
    with pytest.raises(SystemExit):
        ts.main(["--reference", "held-out.npz", "--gdml", "g", "--maps", "m", "--offline-root", "o", "--field-config", "f",
                 "--out", str(tmp_path / "out"), "--variant", "baseline", "--seeds"])


def test_summary_seed_statistics_paired_differences_no_transport(tmp_path, monkeypatch):
    files = []
    for variant, seed, value in (("baseline", 20261111, 2), ("baseline", 20261112, 4), ("half_step", 20261111, 5), ("half_step", 20261113, 8)):
        path = tmp_path / f"{variant}-{seed}.json"
        path.write_text(json.dumps(dict(variant=variant, seed=seed, settings=list(ts.VARIANTS[variant]), inputs={"reference": "development.npz"}, metrics={"mu+.fast": value, "empty": None})))
        files.append(path)
    monkeypatch.setattr(ts, "cohort", lambda *a: pytest.fail("summarizing cannot load references or transport"))
    summary = ts.summarize(files)
    stats = summary["seed_statistics"]["baseline"]["metrics"]["mu+.fast"]
    assert stats == dict(n=2, mean=3., sd=np.sqrt(2), min=2., max=4.)
    paired = summary["paired_setting_differences"]["half_step"]
    assert paired["setting_minus_baseline_by_seed"] == {"20261111": {"mu+.fast": 3, "empty": None}}
    assert paired["metrics"]["mu+.fast"]["sd"] is None
    assert any("NOT common-random" in text for text in summary["warnings"])
    output = tmp_path / "summary.json"
    ts.main(["--summarize", *map(str, files), "--out", str(output)])
    assert json.loads(output.read_text()) == summary
    with pytest.raises(FileExistsError):
        ts.main(["--summarize", *map(str, files), "--out", str(output)])
    with pytest.raises(ValueError, match="unique"):
        ts.summarize(files + files[:1])
    value = json.loads(files[-1].read_text())
    value["inputs"]["reference"] = "different.npz"
    files[-1].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="different"):
        ts.summarize(files)


def test_full_inventory_required_and_expectation_not_cli_configurable(monkeypatch):
    monkeypatch.setattr(ts, "EXPECTED", 3)
    with pytest.raises(ValueError, match="full 100-shard"):
        ts.cohort(tables())
    monkeypatch.setattr(ts, "EXPECTED_SHARDS", 2)
    r = tables()
    r["parent_shard_ids"] = r["parent_shard_ids"][::-1]
    r["parent_shard_event_counts"] = r["parent_shard_event_counts"][::-1]
    selected, _ = ts.cohort(r)
    f = fake_output(selected)
    f.update(id=selected["id"], shard=selected["shard"])
    assert ts.report(selected, f)["endpoints"]["mu+"]["paired_shards"]["ratio"] == 1.


@pytest.mark.parametrize("key,value,match", [("path", np.ones(2), "invalid fast path"),
    ("material_path", np.array([1., -1., 0.]), "invalid fast material_path"),
    ("fate", np.array([1, 9, 1]), "unknown fast fate"), ("p", np.ones((3, 2)), "shape")])
def test_bad_mock_transport_arrays_rejected(monkeypatch, key, value, match):
    monkeypatch.setattr(ts, "EXPECTED", 3)
    monkeypatch.setattr(ts, "EXPECTED_SHARDS", 2)
    r, _ = ts.cohort(tables())
    f = fake_output(r)
    f.update(id=r["id"], shard=r["shard"])
    f[key] = value
    with pytest.raises(ValueError, match=match):
        ts.report(r, f)


@pytest.fixture
def saved_study(mock_study, tmp_path):
    _, calls, reference = mock_study
    r, _ = ts.cohort(reference)
    paths = []
    for vi, variant in enumerate(ts.VARIANTS):
        for si, seed in enumerate(ts.SEEDS):
            f = fake_output(r)
            f.update(id=r["id"], shard=r["shard"])
            f["fate"][:] = [2, 1 if (vi + si) % 2 else 4, 1]
            f["pdg"][:] = -13
            f["material_path"][:] = [vi + 1., si + 10., 30.]
            f["x"][:, 0] = [1., 2., 3.]
            path = tmp_path / f"{variant}-{seed}.json"
            path.write_text(json.dumps(dict(ts.report(r, f), variant=variant, seed=seed,
                settings=list(ts.VARIANTS[variant]), ninput=3,
                inputs={"reference": str(reference.resolve()), "gdml": str((tmp_path / "geometry.gdml").resolve())})))
            np.savez(path.with_suffix(".npz"), **{k: v[[2, 0, 1]] if v.ndim else v for k, v in f.items()})
            paths.append(path)
    return paths, reference, calls


def test_saved_lineages_exact_subset_all_variant_seeds_and_fates(saved_study, tmp_path):
    paths, reference, calls = saved_study
    summary = ts.summarize(paths, reference)
    out = summary["stock_muplus_lineages"]
    assert out["nstock_muplus"] == 2 and out["rate_ninput"] == 3
    assert "not a rate estimator" in out["label"] and "not stock histories" in out["limitations"]
    assert out["id"] == tables()["id"][:2].tolist()
    assert out["source_pdg"] == [-13, 211] and out["source_pmag"] == out["stock_ds_pmag"] == [13., 13.]
    assert out["never_survived_ids"] == tables()["id"][:1].tolist()
    for vi, variant in enumerate(ts.VARIANTS):
        counts = out["by_variant"][variant]
        assert counts["survived_by_seed"] == [(vi + si) % 2 for si in range(4)]
        assert counts["survived_runs_per_lineage"] == [0, 2]
        assert counts["survival_count_histogram"] == [1, 0, 1, 0, 0]
        assert counts["never_survived_ids"] == out["never_survived_ids"]
        assert summary["seed_statistics"][variant]["metrics"]["mu+.fast_only"]["mean"] == 1.
        for si, row in enumerate(out["runs"][vi * 4:(vi + 1) * 4]):
            assert row["seed"] == ts.SEEDS[si] and row["variant"] == variant
            assert row["material_path"] == [vi + 1., si + 10.]
            assert row["fast_fate"] == [2, 1 if (vi + si) % 2 else 4]
            assert row["fate_counts"] == np.bincount(row["fast_fate"], minlength=8).tolist()
            assert row["fast_only"] == 1 and row["terminal_material_bitmask"] is None
    output = tmp_path / "lineages.json"
    ts.main(["--summarize", *map(str, paths), "--reference", str(reference), "--out", str(output)])
    assert json.loads(output.read_text()) == summary
    assert calls == dict(field=[], geometry=[], transport=[])


def test_saved_terminal_materials_vectorized_and_parameter_guards(saved_study, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from beamfastsim import tsgeometry
    paths, reference, calls = saved_study
    gdml = tmp_path / "geometry.gdml"
    solid = SimpleNamespace(contains=lambda x: x[:, 0] > 1.5)
    geometry = SimpleNamespace(materials={"al": None}, _matter=[(solid, np.eye(3), np.zeros(3), "al")])
    monkeypatch.setattr(tsgeometry, "Geometry", lambda path: geometry)
    out = ts.summarize(paths, reference, gdml)["stock_muplus_lineages"]
    assert out["terminal_materials"]["bit_to_material"] == {"1": "al"}
    assert all(row["terminal_material_bitmask"] == [0, 1] for row in out["runs"])
    with pytest.raises(ValueError, match="requires --reference"):
        ts.summarize(paths, gdml=gdml)
    for ref, geom, message in ((tmp_path / "other.npz", gdml, "reference differs"), (reference, tmp_path / "other.gdml", "gdml differs")):
        with pytest.raises(ValueError, match=message):
            ts.summarize(paths, ref, geom)
    for flags in (["--variant", "baseline"], ["--seeds"], ["--field-config", "f"], ["--offline-root", "o"], ["--maps", "m"]):
        with pytest.raises(SystemExit):
            ts.main(["--summarize", *map(str, paths), *flags])
    assert calls == dict(field=[], geometry=[], transport=[])


@pytest.mark.parametrize("change,match", [
    ("missing", None), ("subset", "complete ID cohort"), ("identity", "IDs differ"),
    ("counts", "counts disagree"), ("reference", "full 2-shard"), ("settings", "non-prespecified")])
def test_saved_lineages_reject_missing_mismatched_or_incomplete(saved_study, change, match):
    paths, reference, _ = saved_study
    path = paths[0].with_suffix(".npz")
    if change == "missing":
        path.unlink()
    elif change == "reference":
        r = ts._load(reference)
        r["parent_shard_ids"] = r["parent_shard_ids"][:1]
        np.savez(reference, **r)
    elif change == "settings":
        run = json.loads(paths[0].read_text())
        run["settings"] = list(ts.VARIANTS["half_step"])
        paths[0].write_text(json.dumps(run))
    else:
        f = ts._load(path)
        if change == "subset":
            f = {k: v[:2] if v.ndim else v for k, v in f.items()}
        elif change == "identity":
            f["id"][0, 3] += 100
        else:
            f["pdg"][0] = 211  # DS of another species is not mu+ survival.
        np.savez(path, **f)
    with pytest.raises(FileNotFoundError if change == "missing" else ValueError, match=match):
        ts.summarize(paths, reference)


def test_saved_partial_runs_other_ds_species_not_muplus_survival(saved_study):
    paths, reference, _ = saved_study
    path = paths[0].with_suffix(".npz")
    f = ts._load(path)
    f["fate"][1], f["pdg"][1] = 1, 211  # Shuffled row is the first stock mu+ lineage.
    np.savez(path, **f)
    out = ts.summarize(paths[:1], reference)["stock_muplus_lineages"]
    assert out["runs"][0]["fast_fate"] == [1, 4]
    assert out["runs"][0]["survived_muplus"] == [False, False]
    assert out["runs"][0]["fate_counts"] == [0, 1, 0, 0, 1, 0, 0, 0]
    assert out["never_survived_ids"] == out["id"]
    assert list(out["by_variant"]) == ["baseline"]
    assert out["by_variant"]["baseline"]["survival_count_histogram"] == [2, 0]


def test_summary_keeps_capped_nonstock_inputs_explicit(saved_study):
    paths, reference, _ = saved_study
    path = paths[0]
    with np.load(path.with_suffix('.npz')) as data:
        f = {k: data[k] for k in data.files}
    capped = f['id'][:, 2] == 2
    f['fate'][capped] = 6
    f['path'][capped] = 100000.
    np.savez(path.with_suffix('.npz'), **f)
    r, _ = ts.cohort(reference)
    metadata = json.loads(path.read_text())
    metadata.update(ts.report(r, f))
    path.write_text(json.dumps(metadata))
    row = ts.summarize(paths, reference)['stock_muplus_lineages']['runs'][0]
    assert row['capped_ids'] == r['id'][[2]].tolist()
    assert row['capped_stock_reached'] == [False]
    assert row['capped_path_mm'] == [100000.]
