"""The reco response tables on the shared library core (mu2efastsim.library): bit-identical to the pre-migration loader.

``_legacy_load`` is a frozen copy of trkfastsim.recoresp.table.load before the migration.
"""
import json
import shutil

import numpy as np
import pytest

from mu2efastsim.library import hashing, store as LS
from trkfastsim.recoresp import table as RT

SHIPPED = [RT.CEONLY, RT.FULLDTS]


def _legacy_load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def _entries(n, seed, axis):
    rng = np.random.default_rng(seed)
    pm = rng.uniform(90, 106, n)
    td = rng.uniform(0.3, 1.4, n)
    pt = pm / np.sqrt(1 + td ** 2)
    phi = rng.uniform(0, 2 * np.pi, n)
    p = np.column_stack([pt * np.cos(phi), pt * np.sin(phi), pt * td])
    r = rng.uniform(380, 450, n)
    a = rng.uniform(0, 2 * np.pi, n)
    x = np.column_stack([axis[0] + r * np.cos(a), axis[1] + r * np.sin(a), np.full(n, -1500.0)])
    return rng.uniform(size=n) < 0.9, rng.integers(0, 140, n), x, p, rng.uniform(100, 1500, n)


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_shipped_tables_load_bit_identical_with_their_manifests(path):
    new, old = RT.load(path), _legacy_load(path)
    assert set(new) == set(old)
    for k in old:
        assert new[k].dtype == old[k].dtype and new[k].tobytes() == old[k].tobytes(), k
    m = LS.load(path).manifest
    assert m["content_hash"] == hashing.content_hash(old) and m["name"].startswith("tracker.response.ce_loophelix.")
    assert set(m["unrecorded"]) == {"release", "seeds", "config_hash"}
    assert [a["name"] for a in m["axes"]] == ["nrow", "tandip", "t_entry"]


def test_shipped_tables_have_distinct_names_and_reproducible_sidecars(tmp_path, monkeypatch):
    names = {LS.load(p).manifest["name"] for p in SHIPPED}
    assert len(names) == 3
    data = tmp_path / "data"
    data.mkdir()
    for p in SHIPPED:
        shutil.copyfile(p, data / p.name)
    monkeypatch.setattr(RT, "__file__", str(tmp_path / "table.py"))
    RT.write_shipped_manifests()
    for p in SHIPPED:
        assert json.loads(LS.sidecar_path(data / p.name).read_text()) == json.loads(LS.sidecar_path(p).read_text())


@pytest.mark.parametrize("path", [RT.CEONLY, RT.FULLDTS], ids=lambda p: p.stem)
def test_host_response_is_bit_identical(path):
    new, old = RT.load(path), _legacy_load(path)
    has, nrow, x, p, t = _entries(20000, 31, old["axis_xy"])
    a = RT.apply_numpy(new, np.random.default_rng(7), has, nrow, x, p, t=t)
    b = RT.apply_numpy(old, np.random.default_rng(7), has, nrow, x, p, t=t)
    assert set(a) == set(b)
    for k in a:
        assert np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes(), k


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("path", [RT.CEONLY, RT.FULLDTS], ids=lambda p: p.stem)
def test_device_response_is_bit_identical(path, dtype):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from trkfastsim.recoresp import device as RD
    new, old = RT.load(path), _legacy_load(path)
    has, nrow, x, p, t = _entries(20000, 32, old["axis_xy"])
    args = (jnp.asarray(has), jnp.asarray(nrow), jnp.asarray(x, dtype), jnp.asarray(p, dtype), jnp.asarray(t, dtype))
    key = jax.random.fold_in(jax.random.PRNGKey(9), 2)
    a = RD.apply(RD.to_device(new, dtype), key, *args)
    b = RD.apply(RD.to_device(old, dtype), key, *args)
    for k in a:
        assert np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes(), k


def test_a_registered_table_without_a_manifest_loads_by_its_registry_pin(tmp_path):
    raw = _legacy_load(RT.CEONLY)
    legacy = tmp_path / "ce_loophelix_0123456789abcdef.npz"
    np.savez_compressed(legacy, **raw)                       # a release asset published before the manifests
    with pytest.raises(LS.LibraryError, match="no manifest"):
        RT.load(legacy, reg={})
    reg = {"0123456789abcdef": dict(file=legacy.name, sha256=RT._sha256(legacy), url="")}
    got = RT.load(legacy, reg=reg)
    assert all(got[k].tobytes() == raw[k].tobytes() and not got[k].flags.writeable for k in raw)
    m = RT.write_manifest(legacy)                           # a sidecar makes it a checked library
    assert RT.load(legacy, reg={})["eff_fit"].tobytes() == raw["eff_fit"].tobytes()
    assert "source" in m["unrecorded"]


def test_save_embeds_a_manifest_and_tampering_is_refused(tmp_path):
    raw = {k: np.array(v) for k, v in _legacy_load(RT.FULLDTS).items()}
    m = RT.save(tmp_path / "t.npz", raw)
    assert m["kind"] == "pool" and m["content_hash"] == hashing.content_hash(raw)
    raw["eff_fit"][0, 0] += 1e-9
    np.savez_compressed(tmp_path / "bad.npz", **raw, __manifest__=np.str_(json.dumps(m)))
    with pytest.raises(LS.LibraryError, match="content hash"):
        RT.load(tmp_path / "bad.npz")


def test_domain_counts_report_widened_bins():
    tab = RT.load(RT.FULLDTS)
    has, nrow, x, p, t = _entries(10000, 33, tab["axis_xy"])
    rec = RT.domain_counts(tab, has, nrow, x, p, t=t).record()
    assert rec["lookups"] == int(has.sum()) and set(rec["axes"]) == {"nrow", "tandip", "t_entry"}
    assert sum(rec["fallbacks"]["level"]["levels"].values()) == rec["lookups"]
