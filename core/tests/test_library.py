"""Shared library core (mu2efastsim.library): hashing, axes, domain accounting, manifest, store, registry, dtypes,
fidelity vocabulary. NumPy only, except the JAX twins (skipped without JAX)."""
import json

import numpy as np
import pytest

from mu2efastsim.library import (Axis, DomainCounter, Grid, OutOfDomain, Registry, axes, domain, dtypes, fidelity,
                                 hashing, manifest, registry, store)
from mu2efastsim.library.__main__ import main as cli


def _fields(**kw):
    f = dict(name="core.test.toy", subsystem="core", version=1, kind="binned", description="toy",
             axes=[Axis("e", "MeV", [0.0, 1.0, 2.0, 4.0], array="e_edges").describe()],
             outputs=[dict(name="y", unit="1", dtype="float")],
             source=dict(samples=[dict(dataset="synthetic", selection="all", events=3)]),
             generator=dict(name="numpy", release="test"), release=dict(repository="test", commit="0" * 7),
             seeds=dict(build=1), geometry=dict(tag="none", hash=None), config_hash=None,
             split=dict(training="all", held_out="none", rule="synthetic"),
             dtype=dict(storage={}, compute=["float32", "float64"]), rng=dict(host="none", device="none"),
             domain=dict(coverage="0-4 MeV"), validation=dict(status="not validated", record=None),
             unrecorded=dict(config_hash="no configuration"))
    f.update(kw)
    return f


def _arrays():
    return {"e_edges": np.array([0.0, 1.0, 2.0, 4.0]), "y": np.array([0.5, 0.25, 0.125], np.float32)}


# ------------------------------------------------------------------------------------------------------ hashing
def test_hashes_are_canonical_and_content_based(tmp_path):
    assert hashing.canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
    assert hashing.digest({"b": 1, "a": 2}) == hashing.digest({"a": 2, "b": 1})
    assert len(hashing.digest({"a": 1})) == hashing.HASH_LEN
    with pytest.raises(ValueError):
        hashing.canonical_json({"x": float("nan")})
    a = _arrays()
    h = hashing.content_hash(a)
    assert hashing.content_hash(dict(reversed(list(a.items())))) == h            # order of names does not matter
    assert hashing.content_hash({**a, "y": a["y"].astype(np.float64)}) != h       # dtype does
    assert hashing.content_hash({**a, "y": a["y"].reshape(3, 1)}) != h           # shape does
    b = {k: v.copy() for k, v in a.items()}
    b["y"][0] = np.nextafter(b["y"][0], np.float32(1))
    assert hashing.content_hash(b) != h                                         # one ulp does
    np.savez(tmp_path / "a.npz", **a)
    np.savez_compressed(tmp_path / "b.npz", **a)
    assert hashing.file_sha256(tmp_path / "a.npz") != hashing.file_sha256(tmp_path / "b.npz")
    assert hashing.content_hash(store.load_arrays(tmp_path / "a.npz")) == hashing.content_hash(
        store.load_arrays(tmp_path / "b.npz")) == h
    with pytest.raises(TypeError):
        hashing.content_hash({"o": np.array([object()])})


# --------------------------------------------------------------------------------------------------------- axes
def test_axis_binning_and_domain_codes():
    ax = Axis("e", "MeV", [0.0, 1.0, 2.0, 4.0])
    assert ax.nbins == 3 and ax.array == "e_edges" and not ax.edges.flags.writeable
    v = np.array([-1.0, 0.0, 0.999, 1.0, 3.9, 4.0, 9.0, np.nan])
    assert axes.locate(ax.edges, v[:-1]).tolist() == [0, 0, 0, 1, 2, 2, 2]
    assert axes.domain_codes(ax.edges, v).tolist() == [1, 0, 0, 0, 0, 2, 2, 3]
    for bad in ([1.0], [0.0, 0.0], [0.0, np.inf], [[0.0, 1.0]]):
        with pytest.raises(ValueError):
            Axis("e", "MeV", bad)
    with pytest.raises(ValueError):
        Axis("e", "", [0, 1])
    with pytest.raises(ValueError):
        Axis("e", "MeV", [0, 1], policy="extrapolate")
    d = ax.describe()
    assert Axis.from_description(d, ax.edges) == Axis.from_description(json.loads(json.dumps(d)), ax.edges)
    with pytest.raises(ValueError):
        Axis.from_description(d, np.array([0.0, 1.0, 2.0, 5.0]))


def test_grid_is_row_major():
    g = Grid((Axis("a", "1", [0, 1, 2]), Axis("b", "1", [0, 1, 2, 3]), Axis("c", "1", [0, 1, 2, 3, 4])))
    assert g.shape == (2, 3, 4) and g.size == 24
    i = [np.array([0, 1, 1]), np.array([2, 0, 2]), np.array([3, 1, 3])]
    assert g.ravel(i).tolist() == np.ravel_multi_index(i, g.shape).tolist()
    cells = g.cells({"a": np.array([1.5]), "b": np.array([9.0]), "c": np.array([-1.0])})
    assert cells.tolist() == [np.ravel_multi_index((1, 2, 0), g.shape)]
    assert [x.tolist() for x in g.unravel(cells)] == [[1], [2], [0]]
    with pytest.raises(ValueError):
        Grid((Axis("a", "1", [0, 1]), Axis("a", "1", [0, 1])))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jax_twins_give_the_numpy_indices(dtype):
    pytest.importorskip("jax")
    import jax.numpy as jnp
    edges = np.round(np.arange(0.30, 1.501, 0.05), 6)
    v = np.random.default_rng(1).uniform(0.0, 2.0, 5000).astype(dtype)
    v[:3] = [np.nan, edges[0], edges[-1]]
    e = edges.astype(dtype)
    fin = np.isfinite(v)               # the bin of a NaN is backend-dependent: it is counted as nonfinite instead
    assert np.array_equal(np.asarray(axes.locate_jax(jnp.asarray(e), jnp.asarray(v[fin]))), axes.locate(e, v[fin]))
    assert np.array_equal(np.asarray(axes.domain_codes_jax(jnp.asarray(e), jnp.asarray(v))), axes.domain_codes(e, v))


# ------------------------------------------------------------------------------------------------------- domain
def test_domain_counter_and_refusal():
    edges = np.array([0.0, 1.0, 2.0])
    c = DomainCounter("core.test.toy")
    v = np.array([-1.0, 0.5, 1.5, 3.0, np.nan, 0.1])
    c.add_lookups(len(v))
    c.add_values("e", edges, v)
    c.add_levels("cell_level", np.array([0, 0, 1, 2, 0, 0]))
    rec = c.record()
    assert rec["axes"]["e"] == dict(below=1, above=1, nonfinite=1, fraction=0.5)
    assert rec["coverage"]["e"] == 0.5 and rec["out_of_domain_fraction"] == 0.5
    assert rec["fallbacks"]["cell_level"]["levels"] == {"0": 4, "1": 1, "2": 1}
    assert c.fallback("cell_level") == 2
    c2 = DomainCounter("core.test.toy")
    c2.add_lookups(2)
    c2.add_values("e", edges, np.array([5.0, 0.5]))
    c.merge(c2)
    assert c.lookups == 8 and c.axes["e"]["above"] == 2
    with pytest.raises(ValueError):
        c.merge(DomainCounter("other"))
    with pytest.raises(OutOfDomain) as err:
        domain.check("core.test.toy", "e", axes.domain_codes(edges, v))
    assert err.value.rows.tolist() == [0, 3, 4]
    domain.check("core.test.toy", "e", axes.domain_codes(edges, np.array([0.5])))


# ----------------------------------------------------------------------------------------------------- manifest
def test_manifest_schema_and_unrecorded_fields():
    a = _arrays()
    m = store.finalize(a, _fields())
    assert m["content_hash"] == hashing.content_hash(a) and m["arrays"]["y"] == {"dtype": "<f4", "shape": [3]}
    assert not manifest.complete(m)
    assert manifest.complete(store.finalize(a, _fields(config_hash="abcdef0123456789", unrecorded={})))
    for bad, msg in ((dict(subsystem="optics"), "subsystem"), (dict(name="crv.toy"), "must start with"),
                     (dict(kind="neural"), "kind"), (dict(version=0), "positive"),
                     (dict(seeds=None), "no reason"), (dict(unrecorded=dict(seeds="x")), "has a value"),
                     (dict(validation=dict(status="validated")), "validation.status"),
                     (dict(dtype=dict(compute=["float16"])), "dtype.compute"), (dict(extra=1), "unknown"),
                     (dict(interpolation=dict(method="spline")), "interpolation"),
                     (dict(assumptions=[dict(name="azimuthal symmetry")]), "assumption")):
        with pytest.raises(manifest.ManifestError, match=msg):
            store.finalize(a, _fields(**bad))
    with pytest.raises(manifest.ManifestError, match="edges array"):
        store.finalize({"y": a["y"]}, _fields())
    assert manifest.manifest_hash(m) == hashing.digest(m, None)
    flash = store.finalize(a, _fields(interpolation=dict(method="linear", axes=["e"]),
                                      assumptions=[dict(name="azimuthal symmetry", description="about the DS axis",
                                                        tested=False)]))
    assert flash["interpolation"]["method"] == "linear"


# -------------------------------------------------------------------------------------------------------- store
def test_store_round_trip_and_refusals(tmp_path):
    a = _arrays()
    path = tmp_path / "toy.npz"
    m = store.save(path, a, _fields())
    lib = store.load(path)
    assert lib.manifest == m and lib.name == "core.test.toy" and lib.version == 1
    assert all(lib.arrays[k].tobytes() == a[k].tobytes() and not lib.arrays[k].flags.writeable for k in a)
    assert lib.axis("e").nbins == 3 and lib.grid().shape == (3,)
    assert lib.provenance()["content_hash"] == m["content_hash"] and lib.file_sha256 == hashing.file_sha256(path)
    assert store.load(path, expect={"version": 1}).version == 1
    with pytest.raises(store.LibraryError, match="version is 1, expected 2"):
        store.load(path, expect={"version": 2})
    # tampered array, extra array, missing manifest, sidecar disagreement
    np.savez(tmp_path / "t.npz", e_edges=a["e_edges"], y=a["y"] * 2, __manifest__=np.str_(manifest.dumps(m)))
    with pytest.raises(store.LibraryError, match="content hash"):
        store.load(tmp_path / "t.npz")
    np.savez(tmp_path / "x.npz", z=np.zeros(1), **a, __manifest__=np.str_(manifest.dumps(m)))
    with pytest.raises(store.LibraryError, match="undeclared arrays"):
        store.load(tmp_path / "x.npz")
    np.savez(tmp_path / "n.npz", **a)
    with pytest.raises(store.LibraryError, match="no manifest"):
        store.load(tmp_path / "n.npz")
    side = store.write_sidecar(tmp_path / "n.npz", _fields())
    assert store.load(tmp_path / "n.npz").manifest == side == m
    store.save(tmp_path / "s.npz", a, _fields(), sidecar=True)
    manifest.write(store.sidecar_path(tmp_path / "s.npz"), store.finalize(a, _fields(version=2)))
    with pytest.raises(store.LibraryError, match="differ"):
        store.load(tmp_path / "s.npz")
    np.savez(tmp_path / "p.npz", y=np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        store.load(tmp_path / "p.npz")
    assert store.verify(m, a) == m                                   # another container's arrays, same checks
    with pytest.raises(store.LibraryError, match="content hash"):
        store.verify(m, {**a, "y": a["y"] + 1})
    with pytest.raises(ValueError, match="reserved"):
        store.finalize({**a, "__manifest__": np.zeros(1)}, _fields())


# ----------------------------------------------------------------------------------------------------- registry
def test_registry_fetch_find_and_checks(tmp_path):
    src = tmp_path / "release"
    src.mkdir()
    store.save(src / "toy_v1.npz", _arrays(), _fields())
    store.save(src / "toy_v2.npz", _arrays(), _fields(version=2))
    reg = Registry(cache=tmp_path / "cache", named={"default": src / "toy_v1.npz"})
    for f in ("toy_v1.npz", "toy_v2.npz"):
        e = reg.entry(src / f, url=(src / f).as_uri())
        reg.add(f"{e['name']}@{e['version']}", e)
    assert reg.entries["core.test.toy@1"]["content_hash"] == hashing.content_hash(_arrays())
    assert reg.find("core.test.toy") == "core.test.toy@2"
    assert reg.find("core.test.toy@1") == "core.test.toy@1"
    p = reg.resolve("core.test.toy@2")
    assert p.parent == tmp_path / "cache" and store.load(p).version == 2
    assert reg.resolve("default") == src / "toy_v1.npz" and reg.resolve(str(p)) == p
    with pytest.raises(registry.RegistryError, match="2 registry entries"):
        reg.find(hashing.content_hash(_arrays())[:12])              # both versions hold the same arrays
    with pytest.raises(FileNotFoundError):
        reg.resolve("core.test.none")
    p.write_bytes(p.read_bytes() + b"x")                            # a corrupted cache file is refused
    with pytest.raises(registry.RegistryError, match="sha256"):
        reg.fetch("core.test.toy@2")
    out = reg.write(tmp_path / "registry.json")
    again = Registry.read(out, cache=tmp_path / "cache2")
    assert again.entries == reg.entries
    assert again.resolve("core.test.toy@1").exists()                # downloaded into the new cache
    with pytest.raises(registry.RegistryError):
        reg.add("core.test.toy@1", {"file": "other", "sha256": "0" * 64})
    assert registry.release_url("tracker.nrow.run1a", 2, "f.npz").endswith("/lib-tracker.nrow.run1a-v2/f.npz")


def test_registry_reads_the_tracker_response_layout(tmp_path):
    """A bare {id: {file, sha256, url}} mapping (trkfastsim.recoresp data/registry.json) reads unchanged."""
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"3ef0dabd9a013ff1": {"file": "f.npz", "sha256": "0" * 64, "url": "https://x/f.npz"}}))
    reg = Registry.read(path, cache=tmp_path)
    assert reg.find("3ef0dabd") == "3ef0dabd9a013ff1"
    with pytest.raises(registry.RegistryError, match="not in the cache"):
        reg.fetch("3ef0dabd9a013ff1", download=False)
    path.write_text(json.dumps({"k": {"file": "f.npz"}}))
    with pytest.raises(registry.RegistryError, match="needs file and sha256"):
        Registry.read(path)


# ------------------------------------------------------------------------------------------------------- dtypes
def test_dtype_policy(monkeypatch):
    assert dtypes.compute_dtype("float32") == np.float32 and dtypes.compute_dtype(np.float64) == np.float64
    monkeypatch.setenv("MU2EFS_DEVICE_FLOAT", "float64")
    assert dtypes.compute_dtype() == np.float64
    with pytest.raises(ValueError):
        dtypes.compute_dtype("float16")
    a = dict(e=np.array([0.1, 0.2]), i=np.array([1, 2], np.int64), b=np.array([True]), s=np.str_("ref"))
    for dt in (np.float32, np.float64):
        d = dtypes.device_arrays(a, dt, xp=np)
        assert d["e"].dtype == dt and d["i"].dtype == np.int32 and d["b"].dtype == bool and d["s"] == "ref"
    with pytest.raises(OverflowError):
        dtypes.device_arrays(dict(i=np.array([2 ** 40])), np.float32, xp=np)


# ----------------------------------------------------------------------------------------------------- fidelity
def test_paired_ratio_and_shapes():
    rng = np.random.default_rng(3)
    ref = rng.poisson(5.0, 2000).astype(float)
    cand = ref + rng.normal(0, 0.5, 2000)
    r = fidelity.paired_ratio(ref, cand)
    assert r["status"] == "ok" and abs(r["ratio"] - cand.sum() / ref.sum()) < 1e-15
    np.testing.assert_allclose(r["err"], np.sqrt(2000 * np.var(cand - r["ratio"] * ref, ddof=1)) / ref.sum())
    indep = fidelity.paired_ratio(ref, rng.poisson(5.0, 2000).astype(float))
    assert indep["err"] > 3 * r["err"]                               # pairing removes the shared fluctuation
    assert fidelity.paired_ratio(np.zeros(3), np.ones(3))["status"] == "undefined_zero_reference"
    with pytest.raises(ValueError):
        fidelity.paired_ratio([1.0], [1.0])
    assert fidelity.shape_chi2([10, 20, 30], [10, 20, 30]) == (0.0, 2)
    assert fidelity.shape_chi2([0, 0], [1, 2]) == (0.0, 0)
    a, b = rng.poisson(200, 30), rng.poisson(200, 30)
    t = fidelity.shape_test(a, b, rng.poisson(200, 30), rng.poisson(200, 30))
    assert t["ndf"] == 29 and 0.3 < t["chi2_ndf"] < 3 and 0.1 < t["excess"] < 10
    F = rng.poisson(50, (20, 8)).astype(float)
    s = fidelity.paired_shape(F, F + rng.normal(0, 1, F.shape))
    assert s["shards"] == 20 and s["ndf"] == 8 and s["chi2_ndf_null"] == pytest.approx(19 / 17)


def test_spread_rank_and_coverage():
    s = fidelity.realization_spread([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["std"] == 1.0 and s["n"] == 3
    with pytest.raises(ValueError):
        fidelity.realization_spread([1.0])
    x = np.array([1.0, 2.0, 2.0, 3.0, 5.0, 4.0])
    rho = fidelity.rank_correlation(x, x ** 3)["spearman"]
    assert rho == pytest.approx(1.0)
    assert fidelity.rank_correlation(x, -x)["spearman"] == pytest.approx(-1.0)
    scipy_stats = pytest.importorskip("scipy.stats")
    y = np.random.default_rng(4).normal(size=200)
    z = y + np.random.default_rng(5).normal(size=200)
    assert fidelity.rank_correlation(y, np.round(z, 1))["spearman"] == pytest.approx(
        scipy_stats.spearmanr(y, np.round(z, 1)).statistic, abs=1e-12)
    c = DomainCounter("core.test.toy")
    c.add_lookups(4)
    c.add_values("e", np.array([0.0, 1.0]), np.array([0.5, 0.5, 0.5, 2.0]))
    cov = fidelity.coverage(c.record())
    assert cov["coverage"] == {"e": 0.75} and cov["out_of_domain_fraction"] == 0.25


def test_validation_record_status_levels():
    lib = dict(name="core.test.toy", version=1, content_hash="0" * 64)
    rec = fidelity.ValidationRecord(lib)
    assert rec.status() == "not validated"
    crit = fidelity.Criterion("yield", "events selected", "paired_ratio", "ratio", "within_sigma", target=1.0, k=2)
    with pytest.raises(ValueError):
        rec.read_held_out("seeds 1-3")                               # nothing declared yet
    rec.declare(crit, when="2026-11-11T10:00:00Z")
    assert len(rec.criteria[0]["criterion_hash"]) == hashing.HASH_LEN
    with pytest.raises(ValueError):
        rec.add("float64", "yield", dict(ratio=1.0, err=0.01))       # held-out read not marked
    rec.read_held_out("seeds 1-3", when="2026-11-11T11:00:00Z")
    with pytest.raises(ValueError, match="before the held-out read"):
        rec.declare(fidelity.Criterion("late", "x", "m", "q", "le", tol=1.0))
    rec.add("float64", "yield", dict(ratio=1.01, err=0.01))
    assert rec.status("float64") == "statistically equivalent"
    assert rec.status() == "not validated"                           # float32 missing caps the library
    rec.add("float32", "yield", dict(ratio=1.05, err=0.01))
    assert rec.status("float32") == "tested, not closed" and rec.status() == "tested, not closed"
    rec.reference("float32", bit_identical=True)
    rec.reference("float64", bit_identical=True)
    assert rec.status() == "bit-identical"
    out = rec.as_dict()
    assert out["overall"] == "bit-identical" and json.dumps(out)
    with pytest.raises(ValueError, match="billing"):
        rec.add("float64", "yield", dict(ratio=1.0, err=0.01, su_per_1e6=3.0))
    with pytest.raises(KeyError):
        rec.add("float64", "nope", dict(ratio=1.0))
    assert fidelity.bit_identical({"a": np.arange(3)}, {"a": np.arange(3)})
    assert not fidelity.bit_identical(np.arange(3), np.arange(3).astype(np.int32))


def test_performance_entry_labels_the_boundary():
    bench = dict(workflow="W2", boundary="stops -> CE track", device="cpu", dtype="float32", inputs="fresh",
                 warm_rate=4638.0, setup_s=46.0, n10=1.9e6)
    same = fidelity.performance(bench, stock_rate=121.0, match="yes")
    assert same["multiplier"] == 38.0 and same["label"] == "38x"
    stage = fidelity.performance(bench, stock_rate=121.0, match="stage")
    assert stage["multiplier"] is None and stage["label"] == "stage (38x)"
    with pytest.raises(ValueError):
        fidelity.performance(bench, stock_rate=121.0, match="close")
    with pytest.raises(ValueError):
        fidelity.check_public({"rows": [{"Su_h": 1}]})
    fidelity.check_public({"status": 1, "summary": 2, "issue": 3})


# ---------------------------------------------------------------------------------------------------------- cli
def test_cli(tmp_path, capsys):
    path = tmp_path / "toy.npz"
    m = store.save(path, _arrays(), _fields())
    assert cli(["check", str(path), "--name", "core.test.toy"]) == 0
    assert json.loads(capsys.readouterr().out)["content_hash"] == m["content_hash"]
    assert cli(["check", str(path), "--version", "3"]) == 1
    assert cli(["hash", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["content_hash"] == m["content_hash"]
    assert cli(["describe", str(path)]) == 0
