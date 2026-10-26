"""Synthetic ROOT audits only; production data and the heavy tier1b build are not needed."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from valhist import library_variants as V  # noqa: E402


def write_events(path, tree, keys):
    keys = np.asarray(keys, dtype=np.int64).reshape(-1, 3)
    with uproot.recreate(path) as root:
        root.mktree(tree, {k: "int64" for k in V.KEYS})
        root[tree].extend({k: keys[:, i] for i, k in enumerate(V.KEYS)})


@pytest.fixture
def nominal(tmp_path, monkeypatch):
    # Fingerprinting the full package is orthogonal to synthetic I/O tests.
    monkeypatch.setattr(V, "_code_identity", lambda: {"synthetic_builder.py": "frozen-code"})
    donors, gen = [], []
    for index in range(4):
        path = tmp_path / ("donor_%02d.root" % index)
        # Large full-width keys would collide under older bit-packing conventions.
        write_events(path, "dump/particles", [(7, index, 2 ** 40 + 1), (7, index, 2 ** 40 + 1),
                                               (7, index, 2 ** 40 + 2)])
        donors.append(str(path))
    for index in range(2):
        path = tmp_path / ("gen_%02d.root" % index)
        write_events(path, "dump/gen", [(8, index, 1), (8, index, 1), (8, index, 2)])
        gen.append(str(path))
    path = tmp_path / "nominal.json"
    path.write_text(json.dumps({"inputs": donors, "gen": gen, "generated_scaled_to": 400}))
    path.with_suffix(".root").write_bytes(b"nominal library: no donor keys")
    (tmp_path / "counters.txt").write_text("synthetic local geometry")
    return path


def make_plan(nominal, tmp_path, **kwargs):
    result = V.plan(nominal, tmp_path / "plan", 100, **kwargs)
    return result, tmp_path / "plan/plan.json"


def assert_no_new_files(tmp_path, before):
    assert set(tmp_path.iterdir()) == before
    assert not list(tmp_path.glob(".library-*"))


def fake_builder(calls):
    def run(donors, gen, counters, generated, output):
        calls.append((donors, gen, counters, generated))
        # A real small ROOT product tests publication/hash, not tier1b physics.
        write_events(output, "dump/libxing", [(1, 1, 1)])
        return {"generated_scaled_to": generated, "files": len(donors), "crossings": 1}
    return run


def test_alternating_split_exposure_inventory_and_hashes(nominal, tmp_path):
    result, path = make_plan(nominal, tmp_path)
    assert result["variants"]["half_even"]["donor_indices"] == [0, 2]
    assert result["variants"]["half_odd"]["donor_indices"] == [1, 3]
    assert all(v["generated"] == 200 for v in result["variants"].values())
    assert result["certification"] == {"donor_cross_file_duplicates": 0, "gen_cross_file_duplicates": 0,
                                       "donor_gen_overlap": 0, "donor_unique_events": 8, "gen_unique_events": 4}
    for record in result["donors"] + result["gen"]:
        assert record["events"]["rows"] == 3
        assert record["events"]["unique_events"] == 2
        assert record["events"]["selected_cycle"] == 1
        assert record["sha256"] == V.sha256(record["path"])
    assert V._read_sealed(path) == result
    assert result["nominal_build_code_identity"]["status"] == "unavailable"
    assert not (path.stat().st_mode & 0o222)
    assert "NOT a one-sigma" in result["limitations"][0]


def test_bootstrap_repeats_full_exposure_seed_and_prefix_stability():
    first = V._variants(20, 500000, 3, 123)
    assert first == V._variants(20, 500000, 3, 123)
    longer = V._variants(20, 500000, 4, 123)
    assert all(longer[k] == value for k, value in first.items())
    assert first != V._variants(20, 500000, 3, 124)
    replica = first["bootstrap_0001"]
    assert len(replica["donor_indices"]) == 20
    assert len(set(replica["donor_indices"])) < 20
    assert replica["generated"] == 20 * 500000
    assert sum(replica["file_multiplicity"]) == 20
    assert max(replica["file_multiplicity"]) > 1


@pytest.mark.parametrize("exposure", [99, 0, -1, 100.0, True])
def test_wrong_exposure_has_no_output(nominal, tmp_path, exposure):
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError):
        V.plan(nominal, tmp_path / "bad", exposure)
    assert_no_new_files(tmp_path, before)


@pytest.mark.parametrize("bank", ["donor", "gen", "cross_bank"])
def test_cross_file_overlap_rejected_without_output(nominal, tmp_path, bank):
    if bank == "donor":
        write_events(tmp_path / "donor_01.root", "dump/particles", [(7, 0, 2 ** 40 + 1)])
    elif bank == "gen":
        write_events(tmp_path / "gen_01.root", "dump/gen", [(8, 0, 1)])
    else:
        write_events(tmp_path / "gen_01.root", "dump/gen", [(7, 0, 2 ** 40 + 1)])
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match="cross-file duplicate"):
        V.plan(nominal, tmp_path / "bad", 100)
    assert_no_new_files(tmp_path, before)


def test_highest_cycle_inventory(tmp_path):
    source = tmp_path / "cycles.root"
    write_events(source, "dump/particles", [(1, 1, 1)])
    with uproot.update(source) as root:
        root.mktree("dump/particles", {k: "int64" for k in V.KEYS})
        root["dump/particles"].extend({k: np.array([9, 9]) for k in V.KEYS})
    keys, inv = V._event_inventory(source, "dump/particles")
    assert keys.tolist() == [[9, 9, 9]]
    assert inv["cycles"] == [1, 2]
    assert inv["selected_cycle"] == 2
    assert inv["rows"] == 2 and inv["unique_events"] == 1


@pytest.mark.parametrize("variant", ["half_even", "half_odd", "bootstrap_0001"])
def test_build_passes_repeated_inputs_fixed_gen_and_correct_exposure(nominal, tmp_path, monkeypatch, variant):
    manifest, path = make_plan(nominal, tmp_path, bootstrap=2, seed=3)
    calls = []
    monkeypatch.setattr(V, "_build_library", fake_builder(calls))
    out = tmp_path / "built"
    result = V.build(path, variant, tmp_path / "counters.txt", out)
    indices = manifest["variants"][variant]["donor_indices"]
    expected = [manifest["donors"][i]["path"] for i in indices]
    assert calls[0][0] == result["inputs"] == expected
    assert calls[0][1] == result["gen"] == [r["path"] for r in manifest["gen"]]
    assert calls[0][3] == len(indices) * 100
    if variant.startswith("bootstrap"):
        assert len(set(expected)) < len(expected)
        assert calls[0][3] == manifest["nominal_generated"]
    assert result["provenance"]["library_sha256"] == V.sha256(out / "library.root")
    assert result["provenance"]["plan_sha256"] == V.sha256(path)
    assert V._read_sealed(out / "library.json") == result
    assert V._read_sealed(out / "plan.json") == manifest
    before = {p.name: V.sha256(p) for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        V.build(path, variant, tmp_path / "counters.txt", out)
    assert before == {p.name: V.sha256(p) for p in out.iterdir()}
    assert len(calls) == 1


def test_existing_plan_and_dangling_symlink_never_overwritten(nominal, tmp_path):
    _, path = make_plan(nominal, tmp_path)
    digest = V.sha256(path)
    with pytest.raises(FileExistsError):
        V.plan(nominal, path.parent, 100)
    assert V.sha256(path) == digest
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        V.plan(nominal, link, 100)


@pytest.mark.parametrize("mutation", ["unsealed", "exposure", "selection", "source", "nominal_root", "code", "unknown"])
def test_invalid_build_no_outputs_and_no_builder_call(nominal, tmp_path, monkeypatch, mutation):
    manifest, path = make_plan(nominal, tmp_path)
    if mutation in ("unsealed", "exposure", "selection"):
        path.chmod(0o644)
        path.with_suffix(".json.sha256").chmod(0o644)
        if mutation == "selection":
            manifest["variants"]["half_even"]["donor_indices"] = [0, 0]
        else:
            manifest["variants"]["half_even"]["generated"] = 999
        path.write_bytes(V._json_bytes(manifest))
        if mutation != "unsealed":
            V._sealed_json(path, manifest)
    if mutation == "source":
        with open(manifest["donors"][0]["path"], "ab") as stream:
            stream.write(b"changed")
    if mutation == "nominal_root":
        nominal.with_suffix(".root").write_bytes(b"changed")
    if mutation == "code":
        monkeypatch.setattr(V, "_code_identity", lambda: {"changed": "code"})
    calls = []
    monkeypatch.setattr(V, "_build_library", fake_builder(calls))
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError):
        V.build(path, "typo" if mutation == "unknown" else "half_even", tmp_path / "counters.txt", tmp_path / "bad")
    assert not calls
    assert_no_new_files(tmp_path, before)


@pytest.mark.parametrize("failure", ["builder", "exposure", "drift"])
def test_failed_heavy_build_cleans_staging_and_never_publishes(nominal, tmp_path, monkeypatch, failure):
    _, path = make_plan(nominal, tmp_path)
    def broken(donors, gen, counters, generated, output):
        output.write_bytes(b"partial")
        if failure == "builder":
            raise RuntimeError("synthetic failure")
        if failure == "drift":
            Path(counters).write_text("changed geometry")
        return {"files": len(donors), "generated_scaled_to": -1 if failure == "exposure" else generated}
    monkeypatch.setattr(V, "_build_library", broken)
    before = set(tmp_path.iterdir())
    with pytest.raises((RuntimeError, ValueError)):
        V.build(path, "half_even", tmp_path / "counters.txt", tmp_path / "bad")
    assert_no_new_files(tmp_path, before)


@pytest.mark.parametrize("bad", ["duplicate_path", "odd_count", "missing_tree", "empty", "missing_root"])
def test_invalid_nominal_no_output(nominal, tmp_path, bad):
    metadata = json.loads(nominal.read_text())
    if bad == "duplicate_path":
        metadata["inputs"][1] = metadata["inputs"][0]
    if bad == "odd_count":
        metadata["inputs"].pop()
        metadata["generated_scaled_to"] = 300
    if bad == "missing_tree":
        write_events(metadata["inputs"][0], "dump/gen", [(1, 2, 3)])
    if bad == "empty":
        write_events(metadata["inputs"][0], "dump/particles", [])
    if bad == "missing_root":
        nominal.with_suffix(".root").unlink()
    nominal.write_text(json.dumps(metadata))
    before = set(tmp_path.iterdir())
    with pytest.raises((ValueError, FileNotFoundError)):
        V.plan(nominal, tmp_path / "bad", 100)
    assert_no_new_files(tmp_path, before)


def test_cli_requires_explicit_generated_per_file(nominal, tmp_path):
    before = set(tmp_path.iterdir())
    with pytest.raises(SystemExit) as error:
        V.main(["plan", str(nominal), "--outdir", str(tmp_path / "bad")])
    assert error.value.code == 2
    assert_no_new_files(tmp_path, before)


def test_python39_syntax_and_streaming_hash(tmp_path):
    source = Path(V.__file__).read_text()
    ast.parse(source, feature_version=(3, 9))
    path = tmp_path / "bytes"
    path.write_bytes(b"payload" * 300000)
    assert V.sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_no_outputs_in_code_tree(nominal):
    with pytest.raises(ValueError, match="outside the code tree"):
        V.plan(nominal, V.CODE_ROOT / "never-write-library-test-data", 100)


def test_default_builder_delegates_without_deduplicating(tmp_path, monkeypatch):
    from crvfastsim import geometry, tier1b
    geom, library = object(), object()
    calls = []
    def fake_geometry(path):
        assert path == str(tmp_path / "counters.txt")
        return geom
    def fake_build(donors, gen, received_geom, generated=None):
        calls.append((donors, gen, received_geom, generated))
        return library, {"files": len(donors), "generated_scaled_to": generated}
    def fake_write(path, received_library):
        assert received_library is library
        Path(path).write_bytes(b"built")
    monkeypatch.setattr(geometry, "from_offline_config", fake_geometry)
    monkeypatch.setattr(tier1b, "build", fake_build)
    monkeypatch.setattr(tier1b, "write_library", fake_write)
    info = V._build_library(["a.root", "a.root", "b.root"], ["g.root"], tmp_path / "counters.txt",
                            300, tmp_path / "output.root")
    assert calls == [(["a.root", "a.root", "b.root"], ["g.root"], geom, 300)]
    assert info == {"files": 3, "generated_scaled_to": 300}
    assert (tmp_path / "output.root").read_bytes() == b"built"


def test_manifest_bytes_deterministic_and_cli_end_to_end(nominal, tmp_path, monkeypatch):
    first, path = make_plan(nominal, tmp_path, bootstrap=2, seed=7)
    V.main(["plan", str(nominal), "--outdir", str(tmp_path / "second"),
            "--generated-per-file", "100", "--bootstrap", "2", "--seed", "7"])
    assert (tmp_path / "second/plan.json").read_bytes() == path.read_bytes()
    calls = []
    monkeypatch.setattr(V, "_build_library", fake_builder(calls))
    V.main(["build", str(path), "--variant", "half_odd", "--outdir", str(tmp_path / "built"),
            "--counters", str(tmp_path / "counters.txt")])
    assert len(calls) == 1
    assert V._read_sealed(tmp_path / "built/plan.json") == first


def test_publication_refuses_existing_empty_directory(tmp_path):
    stage, out = tmp_path / "stage", tmp_path / "existing"
    stage.mkdir()
    out.mkdir()
    (stage / "plan.json").write_text("new")
    with pytest.raises(FileExistsError):
        V._publish(stage, out)
    assert not list(out.iterdir())
    assert (stage / "plan.json").read_text() == "new"
