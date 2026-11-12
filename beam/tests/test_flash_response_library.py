"""Flash photon response library on the shared core: tabulation, manifest, domain and lookups (synthetic rows)."""
import json

import numpy as np
import pytest

from beamfastsim import flash_response_library as F
from mu2efastsim.library import OutOfDomain, manifest

EDGES = dict(energy=[0.05, 0.5, 5.0, 100.05], z=[5300.0, 5875.0, 6450.0], r=[0.0, 125.0, 250.0001],
             cos_theta=[-1.0, 0.0, 0.8, 1.0], u_r=[-1.0, 0.0, 1.0])
CLASSES = {"forward": {"cos_theta": [0.8, 1.0]}, "rest": {"cos_theta": [-1.0, 0.8]}}


def _rows(rng, n, cos_range):
    e = np.exp(rng.uniform(np.log(0.05), np.log(100.0), n))
    z, r = rng.uniform(5300, 6450, n), 250 * np.sqrt(rng.uniform(0, 1, n))
    ct = rng.uniform(*cos_range, n)
    ur = rng.uniform(-1, 1, n)
    edep = np.where(rng.uniform(0, 1, n) < 0.02 * (1 + ct), rng.exponential(0.02, n) * e ** 0.3, 0.0)
    return dict(energy=e, z=z, r=r, cos_theta=ct, u_r=ur, edep=edep)


def _cfg(tmp_path, rng):
    files, data = [], {}
    for cls, n0 in (("forward", 3), ("rest", 4)):
        for job in range(n0):
            path = str(tmp_path / f"{cls}_{job}.root")
            data[path] = _rows(rng, 5000, CLASSES[cls]["cos_theta"])
            files.append(dict(path=path, **{"class": cls}, job=job, seed=1000 * (cls == "rest") + job, events=5000))
    cfg = dict(name="flash.response.synthetic", version=1, edges=EDGES, classes=CLASSES, files=files,
               source_description="synthetic", generator=dict(name="numpy", release="test", physics_list=None,
                                                              configuration="synthetic"),
               release=dict(repository="test", commit="0" * 7, entry_point="tests"),
               geometry=dict(tag="synthetic volume", hash="0" * 64), config_hashes=dict(cfg="synthetic"),
               coverage="synthetic")
    return cfg, data


@pytest.fixture
def built(tmp_path):
    rng = np.random.default_rng(7)
    cfg, data = _cfg(tmp_path, rng)
    out = tmp_path / "lib.npz"
    m = F.build(cfg, out, reader=lambda p: data[p])
    return F.load(out), m, cfg, data


def test_cell_statistics_match_a_direct_computation(built):
    lib, _, cfg, data = built
    rows = {k: np.concatenate([d[k] for d in data.values()]) for k in next(iter(data.values()))}
    axes = F.make_axes(EDGES)
    idx = [np.clip(np.searchsorted(a.edges, np.clip(rows[a.name], -1, np.nextafter(1, 0))
                                   if a.name in ("cos_theta", "u_r") else rows[a.name], side="right") - 1,
                   0, a.nbins - 1) for a in axes[1:]]
    flat = np.ravel_multi_index(idx, [a.nbins for a in axes[1:]])
    n = np.bincount(flat, minlength=np.prod([a.nbins for a in axes[1:]]))
    s = np.bincount(flat, weights=rows["edep"], minlength=n.size)
    nz = np.bincount(flat, weights=rows["edep"] > 0, minlength=n.size)
    assert np.array_equal(np.asarray(lib.arrays["n"]).ravel(), n)
    assert np.array_equal(np.asarray(lib.arrays["n_nonzero"]).ravel(), nz.astype(np.int64))
    ok = n > 0
    np.testing.assert_allclose(np.asarray(lib.arrays["mean"]).ravel()[ok], s[ok] / n[ok], rtol=1e-12)
    for c in np.flatnonzero(n > 1)[:20]:
        np.testing.assert_allclose(np.asarray(lib.arrays["variance"]).ravel()[c], np.var(rows["edep"][flat == c], ddof=1),
                                   rtol=1e-9, atol=1e-18)
    assert int(n.sum()) == 35000


def test_manifest_is_complete_and_round_trips(built, tmp_path):
    lib, m, cfg, _ = built
    assert manifest.complete(m) and lib.manifest == m
    assert lib.manifest["kind"] == "binned" and lib.manifest["subsystem"] == "flash"
    assert [a["name"] for a in m["axes"]] == list(F.AXES)
    assert m["seeds"]["rest_0003"] == 1003 and len(m["source"]["samples"]) == 7
    assert m["split"]["held_out"] == "odd job indices" and m["interpolation"]["method"] == "nearest"
    with pytest.raises(Exception):
        F.load(tmp_path / "lib.npz", expect={"name": "flash.response.other"})


def test_declared_events_must_match(tmp_path):
    rng = np.random.default_rng(1)
    cfg, data = _cfg(tmp_path, rng)
    cfg["files"][0]["events"] = 4999
    with pytest.raises(ValueError, match="declared"):
        F.build(cfg, tmp_path / "x.npz", reader=lambda p: data[p])


def test_rows_outside_their_direction_class_are_rejected(tmp_path):
    rng = np.random.default_rng(2)
    cfg, data = _cfg(tmp_path, rng)
    cfg["files"][0]["class"] = "rest"
    with pytest.raises(ValueError, match="direction class"):
        F.build(cfg, tmp_path / "x.npz", reader=lambda p: data[p])


def test_host_lookup_domain_rules(built):
    lib = built[0]
    e = np.array([1.0, 0.01, 1.0]); z = np.full(3, 6000.0); r = np.full(3, 100.0)
    ct = np.array([1.0, 0.9, -1.0]); ur = np.array([0.5, 1.0, -1.0])
    c = F.domain_counts(lib, e, z, r, ct, ur)
    out = F.lookup_numpy(lib, e, z, r, ct, ur)
    assert out[1] == 0.0 and c.fallback(F.FALLBACK) == 1 and c.refused == 0
    assert np.isfinite(out[0]) and np.isfinite(out[2])
    with pytest.raises(OutOfDomain):
        F.lookup_numpy(lib, np.array([200.0]), z[:1], r[:1], ct[:1], ur[:1])
    with pytest.raises(OutOfDomain):
        F.lookup_numpy(lib, e[:1], np.array([7000.0]), r[:1], ct[:1], ur[:1])
    with pytest.raises(OutOfDomain):
        F.lookup_numpy(lib, e[:1], z[:1], r[:1], ct[:1], ur[:1], species=np.array([11]))
    assert F.domain_counts(lib, np.array([200.0]), z[:1], r[:1], ct[:1], ur[:1]).refused == 1


def _x64(on):
    """JAX float64 scoped to a block (no global flag left behind for other tests)."""
    import jax
    ctx = getattr(jax, "enable_x64", None)
    if ctx is None:
        from jax.experimental import enable_x64 as ctx
    return ctx(on)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_device_lookup_matches_host(built, dtype):
    jax = pytest.importorskip("jax")
    lib = built[0]
    rng = np.random.default_rng(3)
    n = 2000
    k = dict(energy=np.exp(rng.uniform(np.log(0.06), np.log(99.0), n)), z=rng.uniform(5310, 6440, n),
             r=rng.uniform(1, 249, n), cos_theta=rng.uniform(-1, 1, n), u_r=rng.uniform(-1, 1, n))
    host = F.lookup_numpy(lib, **k)
    with _x64(dtype is np.float64):
        dl = F.to_device(lib, dtype)
        args = [np.asarray(k[x], dtype=dtype) for x in ("energy", "z", "r", "cos_theta", "u_r")]
        dev = np.asarray(jax.jit(lambda *a: F.lookup(dl, *a))(*args))
        special = np.asarray(F.lookup(dl, np.array([0.01, 200.0, 1.0], dtype), np.array([6000.0, 6000.0, 7000.0], dtype),
                                      np.full(3, 100.0, dtype), np.full(3, 0.9, dtype), np.full(3, 0.1, dtype)))
    assert dev.dtype == np.dtype(dtype)
    # float32 keys can land in a neighbouring cell within one ulp of an edge: compare where both bin alike
    host32 = F.lookup_numpy(lib, *(a.astype(np.float64) for a in args))
    same = np.isfinite(host32)
    np.testing.assert_allclose(dev[same], host32[same], rtol=1e-6 if dtype is np.float32 else 0)
    if dtype is np.float64:
        np.testing.assert_array_equal(dev, host)
    assert special[0] == 0 and np.isnan(special[1]) and np.isnan(special[2])


def test_keys_from_momentum():
    k = F.keys_from_momentum(np.array([-3904.0 + 100]), np.array([0.0]), np.array([6000.0]),
                             np.array([1.0]), np.array([0.0]), np.array([1.0]))
    assert k["r"][0] == pytest.approx(100.0) and k["u_r"][0] == pytest.approx(2 ** -0.5)
    assert k["cos_theta"][0] == pytest.approx(2 ** -0.5)


def test_cli_build_prints_summary(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(4)
    cfg, data = _cfg(tmp_path, rng)
    monkeypatch.setattr(F, "iter_edepana", lambda p: iter([data[p]]))
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    F.main(["build", "--config", str(tmp_path / "c.json"), "--out", str(tmp_path / "cli.npz")])
    out = json.loads(capsys.readouterr().out)
    assert out["complete"] and out["photons"] == 35000


def _write_edepana(path, rows, pdg=22):
    uproot = pytest.importorskip("uproot")
    ak = pytest.importorskip("awkward")
    n = len(rows["energy"])
    one = np.ones(n, dtype=np.int64)
    p = rows["energy"]
    st = np.sqrt(np.clip(1 - rows["cos_theta"] ** 2, 0, None))
    phi = np.zeros(n)
    x, y = F.DS_X + rows["r"], np.zeros(n)
    px, py, pz = p * st * rows["u_r"], p * st * np.sqrt(np.clip(1 - rows["u_r"] ** 2, 0, None)), p * rows["cos_theta"]
    jag = lambda v, dt=np.float32: ak.unflatten(np.asarray(v, dtype=dt), one)
    with uproot.recreate(path) as f:
        f.mkdir("edep")
        f["edep/tree"] = {"nprimaries": np.ones(n, np.int32), "primary_start_x": jag(x), "primary_start_y": jag(y),
                          "primary_start_z": jag(rows["z"]), "primary_start_px": jag(px), "primary_start_py": jag(py),
                          "primary_start_pz": jag(pz), "primary_start_e": jag(p),
                          "primary_start_pdg": jag(np.full(n, pdg), np.int32), "event_trk_edep": rows["edep"]}
    return dict(x=x, y=y, z=rows["z"], px=px, py=py, pz=pz)


def test_chunked_edepana_reader_round_trip(tmp_path):
    rng = np.random.default_rng(5)
    rows = _rows(rng, 1000, (-1.0, 0.8))
    path = str(tmp_path / "e.root")
    raw = _write_edepana(path, rows)
    whole = F.read_edepana(path)
    chunks = list(F.iter_edepana(path, step_size=137))
    assert len(chunks) > 5
    for k, v in whole.items():
        np.testing.assert_array_equal(np.concatenate([c[k] for c in chunks]), v)
    want = F.keys_from_momentum(*(np.asarray(raw[k], np.float32).astype(np.float64)
                                  for k in ("x", "y", "z", "px", "py", "pz")))
    for k in ("z", "r", "cos_theta", "u_r"):
        np.testing.assert_allclose(whole[k], want[k], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(whole["edep"], rows["edep"])


def test_chunked_reader_rejects_non_photons(tmp_path):
    rng = np.random.default_rng(6)
    path = str(tmp_path / "e.root")
    _write_edepana(path, _rows(rng, 50, (-1.0, 0.8)), pdg=11)
    with pytest.raises(ValueError, match="non-photon"):
        F.read_edepana(path)


def test_build_streams_real_files(tmp_path):
    rng = np.random.default_rng(8)
    cfg, data = _cfg(tmp_path, rng)
    for p, rows in data.items():
        _write_edepana(p, rows)
    m = F.build(cfg, tmp_path / "s.npz")
    lib = F.load(tmp_path / "s.npz")
    assert manifest.complete(m) and int(np.asarray(lib.arrays["n"]).sum()) == 35000
