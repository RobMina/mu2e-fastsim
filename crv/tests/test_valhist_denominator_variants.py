"""Synthetic ROOT data only; no transport, geometry download or donor physics build."""
import ast
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crvfastsim import tables, tier1b
from mu2efastsim import tables as core_tables
from valhist import denominator_variants as D
from valhist import library_variants as V


def events(path, run, count, index=0):
    # Two species in every file, variable primary multiplicities; muon-only events count in exposure.
    key = pd.DataFrame({"run": np.full(count, run), "subrun": np.zeros(count, dtype=np.int64),
                        "event": np.arange(count, dtype=np.int64) + 2 ** 40})
    mu = key.assign(pdg=13, e=200.0)
    gamma = key.iloc[np.arange(count - 1 + index) % count].assign(pdg=22, e=10.0)
    electron = key.iloc[np.arange(index + 1) % count].assign(pdg=-11, e=10.511)
    frame = pd.concat([mu, gamma, electron], ignore_index=True)
    core_tables.write(str(path), {"gen": frame}, as_ttree=True)
    return frame


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "_code_identity", lambda: {"synthetic_builder": "frozen"})
    donors, gen = [], []
    for index in range(4):
        path = tmp_path / ("donor%d.root" % index)
        frame = pd.DataFrame({"run": [1], "subrun": [index], "event": [2 ** 40]})
        core_tables.write(str(path), {"particles": frame}, as_ttree=True)
        donors.append(str(path))
    for index, count in enumerate([2, 3, 4]):
        path = tmp_path / ("gen%d.root" % index)
        events(path, 10 + index, count, index)
        gen.append(str(path))
    nominal = tmp_path / "nominal.json"
    nominal.write_bytes(V._json_bytes({"inputs": donors, "gen": gen, "generated_scaled_to": 400}))
    nominal.with_suffix(".root").write_bytes(b"synthetic nominal")
    counters = tmp_path / "counters.txt"
    counters.write_text("synthetic geometry")
    V.plan(nominal, tmp_path / "donor_plan", 100, bootstrap=1)

    def fake_build(donors, gen, counters, generated, output):
        showers = pd.DataFrame({"shid": [0, 1], "apdg": [11, 22], "kebin": [3, 3], "ke": [10., 10.]})
        reach, nonmu = tier1b._reach(showers, gen, generated)
        lib = {name: pd.DataFrame({"payload": np.array([3, 5], dtype=np.int64)})
               for name in tier1b.LIB_TREES}
        lib.update({"libshower": showers, "libgen": reach})
        tier1b.write_library(str(output), lib)
        return {"generated_scaled_to": generated, "generated_nonmu": nonmu, "files": len(donors)}

    monkeypatch.setattr(V, "_build_library", fake_build)
    V.build(tmp_path / "donor_plan/plan.json", "bootstrap_0001", counters, tmp_path / "base")
    events(tmp_path / "held.root", 99, 3)
    return tmp_path / "base/library.json"


def make_plan(base, tmp_path, name="denominator", **kwargs):
    args = {"held_out": [tmp_path / "held.root"], "generator_events": [2, 3, 4],
            "held_out_events": [3], "replicas": 5, "seed": 6}
    args.update(kwargs)
    manifest = D.plan(base, tmp_path / name, **args)
    return manifest, tmp_path / name / "plan.json"


def test_seed_prefix_and_multiplicity():
    first = D._variants([2, 3, 4], 3, 7)
    assert first == D._variants([2, 3, 4], 3, 7)
    assert first != D._variants([2, 3, 4], 3, 8)
    assert all(D._variants([2, 3, 4], 4, 7)[k] == v for k, v in first.items())
    for value in first.values():
        indices = value["generator_indices"]
        assert len(indices) == sum(value["file_multiplicity"]) == 3
        assert value["selected_generator_events"] == sum([2, 3, 4][i] for i in indices)
    assert any(max(x["file_multiplicity"]) > 1 for x in first.values())


def test_plan_audit_hashes_immutable_and_deterministic(base, tmp_path):
    manifest, path = make_plan(base, tmp_path)
    again, second = make_plan(base, tmp_path, name="second")
    assert manifest == again and path.read_bytes() == second.read_bytes()
    assert V._read_sealed(path) == manifest
    assert manifest["mode"] == "denominator_only"
    assert manifest["audit"]["held_out_generator"] == manifest["audit"]["held_out_donor"] == 0
    assert manifest["audit"]["held_out_unique_events"] == 3
    assert manifest["exposure"]["donor_generated"] == 400
    assert len(manifest["source_sha256"]) == len(manifest["exposure_sha256"]) == 64
    assert all(not p.stat().st_mode & 0o222 for p in path.parent.iterdir())
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        make_plan(base, tmp_path)
    assert path.read_bytes() == before


def test_build_direct_formula_latest_cycle_and_unchanged_payload(base, tmp_path):
    manifest, plan_path = make_plan(base, tmp_path)
    original = tier1b.read_library(str(base.with_suffix(".root")))
    for name, choice in manifest["variants"].items():
        out = tmp_path / name
        result = D.build(plan_path, name, out)
        library = tier1b.read_library(str(out / "library.root"))
        for tree in tier1b.LIB_TREES:
            if tree != "libgen":
                pd.testing.assert_frame_equal(library[tree], original[tree])
        # Independent direct weighted-total formula: no event ID deduplication across occurrences.
        counts = np.zeros(2, dtype=np.int64)  # electron, gamma
        exposure = 0
        for i in choice["generator_indices"]:
            frame = tables.read(manifest["generators"][i]["path"], "gen")
            counts += [int((frame.pdg.abs() == 11).sum()), int((frame.pdg.abs() == 22).sum())]
            exposure += frame[list(V.KEYS)].drop_duplicates().shape[0]
        assert library["libgen"].ngen.tolist() == np.rint(counts * 400.0 / exposure).astype(int).tolist()
        assert library["libgen"].nshower.tolist() == [1, 1]
        assert result["generated_nonmu"] == counts.sum()
        assert result["provenance"]["selected_generator_events"] == exposure
        assert result["provenance"]["selection"] == choice
        assert result["provenance"]["library_sha256"] == V.sha256(out / "library.root")
        assert V._read_sealed(out / "library.json") == result
        assert V._read_sealed(out / "plan.json") == manifest
        assert all(not p.stat().st_mode & 0o222 for p in out.iterdir())
        with uproot.open(out / "library.root") as root:
            assert "dump/libgen;1" in root and "dump/libgen;2" in root
        with pytest.raises(FileExistsError):
            D.build(plan_path, name, out)


def test_repeat_exposure_matches_occurrence_rekeyed_existing_reach(base, tmp_path):
    manifest, _ = make_plan(base, tmp_path)
    weights = [2, 0, 1]
    showers = tables.read(base.with_suffix(".root"), "libshower")
    reach, nonmu, exposure = D._reach(showers, manifest["generators"], weights, 400)
    paths, repeated = [], []
    for occurrence, index in enumerate([0, 0, 2]):
        source = manifest["generators"][index]["path"]
        frame = tables.read(source, "gen")
        frame["run"] = occurrence + 1000  # label occurrences, not individual events or primaries
        path = tmp_path / ("occurrence%d.root" % occurrence)
        core_tables.write(str(path), {"gen": frame}, as_ttree=True)
        paths.append(str(path))
        repeated.append(source)
    direct, count = tier1b._reach(showers, paths, 400)
    pd.testing.assert_frame_equal(reach, direct)
    assert nonmu == count and exposure == 8
    one, _, _ = D._reach(showers, manifest["generators"], [1, 0, 0], 400)
    repeated_one, _, _ = D._reach(showers, manifest["generators"], [3, 0, 0], 400)
    pd.testing.assert_frame_equal(one, repeated_one)
    first_file = tables.read(manifest["generators"][0]["path"], "gen")
    assert first_file[list(V.KEYS)].drop_duplicates().shape[0] == 2
    assert first_file[first_file.pdg.abs() != 13][list(V.KEYS)].drop_duplicates().shape[0] == 1
    naive, _ = tier1b._reach(showers, repeated, 400)
    assert not naive.equals(reach)  # original helper deduplicates repeated keys, so cannot bootstrap it directly


@pytest.mark.parametrize("bank", ["generator", "donor"])
def test_held_out_overlap_full_width_keys_rejected(base, tmp_path, bank):
    if bank == "generator":
        events(tmp_path / "held.root", 10, 2)
    else:
        events(tmp_path / "held.root", 1, 2)
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match="event ID overlap"):
        make_plan(base, tmp_path)
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("kwargs", [
    {"generator_events": [2, 3, 5]}, {"generator_events": [2, 3]},
    {"generator_events": [True, 3, 4]}, {"held_out_events": [4]},
    {"held_out": []}, {"replicas": 0}, {"seed": -1},
])
def test_invalid_exposure_or_selection_leaves_no_output(base, tmp_path, kwargs):
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError):
        make_plan(base, tmp_path, **kwargs)
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("mutation", ["selection", "exposure", "counts", "audit", "source_hash", "unsealed",
                                      "source", "base_root", "base_json", "code", "unknown"])
def test_drift_and_resealed_plan_mutations_rejected(base, tmp_path, monkeypatch, mutation):
    manifest, path = make_plan(base, tmp_path)
    if mutation in ["selection", "exposure", "counts", "audit", "source_hash", "unsealed"]:
        path.chmod(0o644)
        path.with_suffix(".json.sha256").chmod(0o644)
        if mutation == "selection":
            manifest["variants"]["denominator_0001"]["generator_indices"] = [0, 0, 0]
        elif mutation == "exposure":
            manifest["exposure"]["donor_generated"] += 1
        elif mutation == "counts":
            manifest["generators"][0]["bin_counts"][0][2] += 1
        elif mutation == "audit":
            manifest["audit"]["held_out_generator"] = 100
        else:
            manifest["source_sha256"] = "changed"
        if mutation == "unsealed":
            path.write_bytes(V._json_bytes(manifest))
        else:
            V._sealed_json(path, manifest)
    elif mutation in ["source", "base_root", "base_json"]:
        target = {"source": tmp_path / "held.root", "base_root": base.with_suffix(".root"),
                  "base_json": base}[mutation]
        target.chmod(0o644)
        with target.open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "code":
        monkeypatch.setattr(D, "_code_identity", lambda: {"changed": "code"})
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError):
        D.build(path, "typo" if mutation == "unknown" else "denominator_0001", tmp_path / "bad")
    assert set(tmp_path.iterdir()) == before


def test_base_formula_mismatch_even_when_resealed(base, tmp_path):
    root = base.with_suffix(".root")
    original = tables.read(root, "libgen")
    original["ngen"] += 1
    root.chmod(0o644)
    with uproot.update(root) as f:
        f.mktree("dump/libgen", {c: x.dtype for c, x in original.items()})
        f["dump/libgen"].extend({c: x.to_numpy() for c, x in original.items()})
    metadata = V._read_sealed(base)
    metadata["provenance"]["library_sha256"] = V.sha256(root)
    base.chmod(0o644)
    base.with_suffix(".json.sha256").chmod(0o644)
    V._sealed_json(base, metadata)
    with pytest.raises(ValueError, match="source-reviewed denominator formula"):
        make_plan(base, tmp_path)
    assert not (tmp_path / "denominator").exists()


@pytest.mark.parametrize("failure", ["write", "source_drift", "plan_drift"])
def test_staging_cleanup_and_postflight_checks(base, tmp_path, monkeypatch, failure):
    manifest, path = make_plan(base, tmp_path)
    real = D._write_variant
    def broken(source, output, reach):
        real(source, output, reach)
        if failure == "write":
            raise RuntimeError("synthetic failure")
        if failure == "source_drift":
            with Path(manifest["generators"][0]["path"]).open("ab") as f:
                f.write(b"drift")
        if failure == "plan_drift":
            path.chmod(0o644)
            path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(D, "_write_variant", broken)
    before = set(tmp_path.iterdir())
    with pytest.raises((RuntimeError, ValueError)):
        D.build(path, "denominator_0001", tmp_path / "bad")
    assert set(tmp_path.iterdir()) == before
    assert not list(tmp_path.glob(".denominator-*"))


def test_probability_support_rounding_and_zero_nonmu():
    generators = [{"events": {"unique_events": 4}, "bin_counts": [[22, 3, 2], [11, 3, 1]]}]
    showers = pd.DataFrame({"apdg": [22], "kebin": [3]})
    reach, nonmu, exposure = D._reach(showers, generators, [1], 5)
    assert reach.ngen.tolist() == [1, 2]  # np.rint(1.25), np.rint(2.5), ties-to-even
    assert nonmu == 3 and exposure == 4
    for bad in [pd.DataFrame({"apdg": [211], "kebin": [3]}),
                pd.DataFrame({"apdg": [22, 22, 22], "kebin": [3, 3, 3]})]:
        with pytest.raises(ValueError, match="unsupported shower bin or reach probability above one"):
            D._reach(bad, generators, [1], 5)
    with pytest.raises(ValueError, match="unsupported shower bin"):
        D._reach(showers, generators, [1], 1)  # rounds denominator to zero
    with pytest.raises(ValueError, match="no nonmuon"):
        D._reach(showers, [{"events": {"unique_events": 3}, "bin_counts": []}], [1], 5)
    for weights in [[0], [-1], [1.0], [1, 2]]:
        with pytest.raises(ValueError, match="multiplicity"):
            D._reach(showers, generators, weights, 5)


def test_held_out_cross_file_duplicates_rejected(base, tmp_path):
    events(tmp_path / "held_copy.root", 99, 3)
    with pytest.raises(ValueError, match="cross-file duplicate"):
        make_plan(base, tmp_path, held_out=[tmp_path / "held.root", tmp_path / "held_copy.root"],
                  held_out_events=[3, 3])
    assert not (tmp_path / "denominator").exists()


def test_generator_highest_cycle_all_primary_rows_and_finite_energy(tmp_path):
    path = tmp_path / "cycles.root"
    first = events(path, 20, 2)
    second = pd.concat([first, first.iloc[[2]]], ignore_index=True)
    second["run"] = 21
    with uproot.update(path) as root:
        root.mktree("dump/gen", {c: x.dtype for c, x in second.items()})
        root["dump/gen"].extend({c: x.to_numpy() for c, x in second.items()})
    keys, inventory = V._event_inventory(path, "dump/gen")
    assert inventory["cycles"] == [1, 2] and inventory["selected_cycle"] == 2
    assert inventory["unique_events"] == 2 and np.all(keys[:, 0] == 21)
    assert D._counts({"path": str(path)}) == [[11, 3, 1], [22, 3, 2]]
    second.loc[0, "e"] = np.nan
    core_tables.write(str(path), {"gen": second}, as_ttree=True)
    with pytest.raises(ValueError, match="invalid generator pdg/energy"):
        D._counts({"path": str(path)})


def test_cli_and_output_guards(base, tmp_path):
    out = tmp_path / "cli"
    D.main(["plan", str(base), "--outdir", str(out), "--held-out", str(tmp_path / "held.root"),
            "--generator-events", "2", "3", "4", "--held-out-events", "3"])
    D.main(["build", str(out / "plan.json"), "--variant", "denominator_0001",
            "--outdir", str(tmp_path / "built")])
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        D.build(out / "plan.json", "denominator_0001", link)
    with pytest.raises(ValueError, match="outside the code tree"):
        D.build(out / "plan.json", "denominator_0001", V.CODE_ROOT / "never-create-denominator-output")
    ast.parse(Path(D.__file__).read_text(), feature_version=(3, 9))
