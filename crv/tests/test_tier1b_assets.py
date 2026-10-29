"""Tier-1b asset sets: resolution per run configuration, loading order and hash checks, and the callers' options."""
import json
from pathlib import Path

import pandas as pd
import pytest

from crvfastsim import tier1b_assets as T
from crvfastsim.analyses import runchain as R
from crvfastsim.analyses import tier1b as A
from test_e2e_bench import pipeline  # noqa: F401  (fixture)

STRICT = dict(context=True, contain=True, point=True, topology=True, unvisited=True, nearmiss=True)
DEFAULT = dict(STRICT, chord=True)                       # strict_v3: strict_v2 + v3 templates + chord placement


def test_default_per_run_configuration():
    s = T.resolve(data="/D")
    assert s.name == "strict_v3" and s.options == DEFAULT
    assert s.library == "/D/tier1b_strict_v2/library.root"
    assert s.paths() == ["/D/tier1b_strict_v2/library.root", "/D/tier1b_strict_v2/topology_sidecar_strict_v2_v2.npz",
                         "/D/tier1b_strict_v2/topology_sidecar_strict_v2_v2.npz.json",
                         "/D/tier1b_strict_v2/templates_v3.npz", "/D/tier1b_strict_v2/templates_v3.npz.json"]
    old = T.resolve("strict_v2", data="/D")
    assert old.options == STRICT and old.templates == "/D/tier1b_strict_v2/templates_v2.npz"
    assert old.library == s.library and old.library_sha256 == s.library_sha256 and old.topology == s.topology
    for rc in ("run1a_v01", "run1b_v01", "run2_v10"):        # no library of their own: legacy only
        assert T.resolve(run_config=rc, data="/D").name == "legacy"
        with pytest.raises(ValueError, match="no library for run configuration"):
            T.resolve("strict_v2", run_config=rc, data="/D")
    legacy = T.resolve("legacy", data="/D")
    assert legacy.library == "/D/tier1b_v7/lib_prod_as_20files.root" and legacy.options == {} and legacy.paths() == [legacy.library]
    assert T.resolve("strict_v2", directory="/shm/x").templates == "/shm/x/templates_v2.npz"


def test_custom_library_and_refusals(monkeypatch):
    c = T.resolve(library="lib.root")
    assert c.name is None and c.options == {} and c.paths() == ["lib.root"]
    with pytest.raises(ValueError, match="not both"):
        T.resolve("legacy", library="lib.root")
    with pytest.raises(ValueError, match="unknown tier-1b set"):
        T.resolve("strict_v4", data="/D")
    monkeypatch.delenv("CRVFS_DATA", raising=False)
    with pytest.raises(ValueError, match="CRVFS_DATA"):
        T.resolve()


def test_sets_match_the_endpoint_registry():
    cp = pytest.importorskip("mu2efastsim.endpoints.chainpoint")
    for name, s in T.SETS.items():
        directory, files, _ = cp.TIER1B_SETS[name]
        assert directory == s.directory and list(files) == s.files()
    assert cp.LIBRARY == "/".join((T.SETS["legacy"].directory, T.SETS["legacy"].library))


def _fake_set(tmp_path, monkeypatch, library=b"library bytes"):
    spec = T.resolve(directory=tmp_path)
    Path(spec.library).write_bytes(library)
    calls = []
    monkeypatch.setattr("crvfastsim.tier1b.read_library", lambda p: calls.append(("read", p)) or {"lib": p})
    monkeypatch.setattr("crvfastsim.tier1b.attach_topology", lambda lib, c, v, g: calls.append(("topology", c, v)) or {**lib, "t": 1})
    monkeypatch.setattr("crvfastsim.tier1b.attach_templates", lambda lib, side, g: calls.append(("templates", side)) or {**lib, "u": 1})
    monkeypatch.setattr(A, "read_topology_sidecar", lambda p, library=None, library_sha256=None: (p, library_sha256))
    monkeypatch.setattr(A, "read_template_sidecar", lambda p, library=None, library_sha256=None: (p, library_sha256))
    return spec, calls


def test_load_order_and_library_pin(tmp_path, monkeypatch):
    spec, calls = _fake_set(tmp_path, monkeypatch)
    pinned = T.sha256(spec.library)
    spec = T.Tier1bSpec(spec.name, spec.library, spec.topology, spec.templates, pinned, spec.options)
    lib = T.load(spec, geom=None)
    assert lib == {"lib": spec.library, "t": 1, "u": 1}
    assert [c[0] for c in calls] == ["read", "topology", "templates"]          # tier1b-apply's order
    assert calls[1][1:] == (spec.topology, pinned) and calls[2][1] == (spec.templates, pinned)
    bad = T.Tier1bSpec(spec.name, spec.library, spec.topology, spec.templates, "0" * 64, spec.options)
    with pytest.raises(ValueError, match="not the pinned library"):
        T.load(bad, geom=None)
    calls.clear()
    assert T.load(T.Tier1bSpec("legacy", spec.library), geom=None) == {"lib": spec.library}
    assert calls == [("read", spec.library)]                                   # legacy: read only, as before


def test_sidecar_reader_checks_library_by_hash(tmp_path):
    side = tmp_path / "s.npz"
    side.write_bytes(b"sidecar")
    lib = tmp_path / "lib.root"
    lib.write_bytes(b"lib")
    meta = {"sidecar_sha256": A._sha256(side), "library_sha256": A._sha256(lib)}
    Path(str(side) + ".json").write_text(json.dumps(meta))
    for reader in (A.read_topology_sidecar, A.read_template_sidecar):
        with pytest.raises(ValueError, match="another library"):
            reader(side, library_sha256="f" * 64)
        with pytest.raises(ValueError, match="another library"):
            reader(side, library=side)
        with pytest.raises(Exception) as e:          # right library: gets past the checks to np.load of fake bytes
            reader(side, library_sha256=meta["library_sha256"])
        assert "another library" not in str(e.value) and "changed" not in str(e.value)


def test_runchain_passes_the_set_options(pipeline, monkeypatch, tmp_path):  # noqa: F811
    seen, loaded = [], []
    monkeypatch.setattr(R.tier1b, "sample", lambda st, pa, *a, **kw: seen.append(kw) or (st, pa, None))
    monkeypatch.setattr(R.tier1b_assets, "load", lambda spec, geom: loaded.append(spec) or None)
    monkeypatch.setenv("CRVFS_DATA", str(tmp_path))
    base = dict(inputs=[str(tmp_path / "gen.root")], calib="nominal", gdml="world")
    res = R.run(outdir=str(tmp_path / "d"), **base)
    assert seen[-1] == {"showers": True, **DEFAULT} and loaded[-1].name == "strict_v3"
    assert res["tier1b"]["set"] == "strict_v3" and res["tier1b"]["options"] == DEFAULT
    res = R.run(outdir=str(tmp_path / "v2"), tier1b_set="strict_v2", **base)
    assert seen[-1] == {"showers": True, **STRICT} and res["tier1b"]["set"] == "strict_v2"
    res = R.run(outdir=str(tmp_path / "l"), tier1b_set="legacy", **base)
    assert seen[-1] == {"showers": True} and res["tier1b"]["set"] == "legacy"
    R.run(outdir=str(tmp_path / "c"), library="custom.root", **base)
    assert seen[-1] == {"showers": True} and loaded[-1].library == "custom.root"
    with pytest.raises(SystemExit, match="no library for run configuration"):
        R.run(outdir=str(tmp_path / "n"), tier1b_set="strict_v2", run_config="run2_v10", **base)


def test_prepared_signature_binds_the_set(tmp_path):
    rc, cfg = R.runconfig.get(), R.chain.ChainConfig()
    sig = lambda spec: R._signature(rc, cfg, "world", spec, "nominal", None, False, False, False)
    strict, legacy = T.resolve(data=str(tmp_path)), T.resolve("legacy", data=str(tmp_path))
    assert sig(strict) != sig(legacy) and sig(strict)["tier1b"]["options"] == DEFAULT
    assert sig(strict) != sig(T.resolve("strict_v2", data=str(tmp_path)))
    assert sig(legacy)["paths"] == [str(Path("world").resolve()), str(Path(legacy.library).resolve()), "nominal", None]


def test_merge_refuses_mixed_sets(tmp_path):
    shard = {"generated_events": 1, "timers": {}, "wall_s": 1.0, "cl_type_counts": {}, "run_config": "extracted_v04",
             "window": [0.0, 1e5], "conditions": None}
    strict = {**shard, "tier1b": T.resolve(data="/A").describe()}
    other = {**shard, "tier1b": T.resolve(directory="/B").describe()}       # same set elsewhere: compatible
    legacy = {**shard, "tier1b": T.resolve("legacy", data="/A").describe()}
    paths = []
    for i, j in enumerate((strict, other, legacy)):
        paths.append(tmp_path / f"s{i}.json")
        paths[-1].write_text(json.dumps(j))
    try:
        R.merge([str(p) for p in paths[:2]], str(tmp_path / "ok"))
    except ValueError as e:
        assert "tier-1b" not in str(e)
    with pytest.raises(ValueError, match="tier-1b"):
        R.merge([str(paths[0]), str(paths[2])], str(tmp_path / "bad"))


def test_tier1b_apply_default_legacy_and_custom(tmp_path, monkeypatch):
    seen, loaded = [], []
    empty = pd.DataFrame({"run": [], "subrun": [], "event": []})
    monkeypatch.setattr(A.tables, "read", lambda f, tree: empty)
    monkeypatch.setattr(A.tables, "write", lambda *a, **kw: None)
    monkeypatch.setattr(A, "_geom", lambda counters: None)
    monkeypatch.setattr(A.tier1b, "sample", lambda *a, **kw: seen.append(kw) or (empty, empty, {}))
    monkeypatch.setattr(A.tier1b, "read_library", lambda p: {"custom": p})
    monkeypatch.setattr(A.tier1b_assets, "load", lambda spec, geom: loaded.append(spec) or {})
    monkeypatch.setenv("CRVFS_DATA", str(tmp_path))
    A.apply(["fastC.x.root"], str(tmp_path))
    assert seen[-1] == DEFAULT and loaded[-1].name == "strict_v3"
    A.apply(["fastC.x.root"], str(tmp_path), tier1b_set="legacy")
    assert seen[-1] == {} and loaded[-1].name == "legacy"
    A.apply(["fastC.x.root"], str(tmp_path), library="lib.root")
    assert seen[-1] == {k: False for k in ("context", "contain", "point", "mirror", "translate", "topology")}
    with pytest.raises(SystemExit, match="custom --library"):
        A.apply(["fastC.x.root"], str(tmp_path), crossing_context=True)
    with pytest.raises(SystemExit, match="not both"):
        A.apply(["fastC.x.root"], str(tmp_path), library="lib.root", tier1b_set="legacy")


def test_cli_exposes_the_sets():
    from crvfastsim import cli
    p = cli.build_parser()
    a = p.parse_args(["tier1b-apply", "x.root", "--tier1b", "legacy", "--tier1b-dir", "/d"])
    assert a.tier1b_set == "legacy" and a.tier1b_dir == "/d" and a.library is None
    a = p.parse_args(["runchain", "x.root", "--calib", "c.csv"])
    assert a.tier1b_set is None and a.library is None
    with pytest.raises(SystemExit):
        p.parse_args(["runchain", "x.root", "--tier1b", "other"])
