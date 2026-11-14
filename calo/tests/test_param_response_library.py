"""The parametric digi-response tables on the shared library core (mu2efastsim.library): the frozen v1 table loads
bit-identical to the pre-migration loader (``_legacy_load`` is a frozen copy of calofastsim.param_response.load before
the migration); the shipped v2 table carries its manifest, provenance and cell definition."""
import json
import shutil

import numpy as np
import pytest

from calofastsim import param_response as PR
from mu2efastsim.library import hashing, store as LS


def _legacy_load(path=PR.DEFAULT_V1):
    with np.load(path, allow_pickle=False) as f:
        tab = {k: f[k] for k in f.files}
    if int(tab["schema"]) != PR.SCHEMA:
        raise ValueError("schema")
    return tab


def _cells(n, seed):
    rng = np.random.default_rng(seed)
    E = np.exp(rng.uniform(np.log(1e-3), np.log(500.0), n))
    T = rng.uniform(300, 1600, n)
    v = rng.uniform(-260, 60, n)
    return rng.uniform(size=n) < 0.95, E, T, v


def test_v1_table_loads_bit_identical_with_its_manifest():
    new, old = PR.load("v1"), _legacy_load()
    assert set(new) == set(old)
    for k in old:
        assert new[k].dtype == old[k].dtype and new[k].tobytes() == old[k].tobytes(), k
    m = LS.load(PR.DEFAULT_V1).manifest
    assert m["name"] == PR.LIBRARY and m["content_hash"] == hashing.content_hash(old)
    assert set(m["unrecorded"]) == {"config_hash"} and m["legacy"]["schema"] == 2
    assert [a["name"] for a in m["axes"]] == ["E", "v"]


def test_v1_sidecar_is_reproducible(tmp_path):
    copy = tmp_path / PR.DEFAULT_V1.name
    shutil.copyfile(PR.DEFAULT_V1, copy)
    PR.write_shipped_manifest(copy)
    assert json.loads(LS.sidecar_path(copy).read_text()) == json.loads(LS.sidecar_path(PR.DEFAULT_V1).read_text())


def test_shipped_v2_table_and_manifest():
    tab = PR.load("default")
    assert PR.resolve("default") == PR.resolve("v2") == PR.DEFAULT
    assert PR.core_window(tab) == PR.CORE_WINDOW and PR.to_device(tab).core_window == PR.CORE_WINDOW
    m = LS.load(PR.DEFAULT).manifest
    side = json.loads(LS.sidecar_path(PR.DEFAULT).read_text())
    assert m == side and m["name"] == PR.LIBRARY and m["version"] == 2
    assert m["content_hash"] == hashing.content_hash({k: v for k, v in tab.items()})
    assert set(m["unrecorded"]) == {"config_hash"} and [a["name"] for a in m["axes"]] == ["E", "v"]
    assert int(tab["n_training"]) == 312247 and np.all(tab["count"] > 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_device_draw_is_bit_identical(dtype):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    ok, E, T, v = _cells(30000, 41)
    args = (jnp.asarray(ok), jnp.asarray(E, dtype), jnp.asarray(T, dtype), jnp.asarray(v, dtype))
    key = jax.random.fold_in(jax.random.PRNGKey(2), 3)
    a = PR.draw(PR.to_device(PR.load("v1")), key, *args)
    b = PR.draw(PR.to_device(_legacy_load()), key, *args)
    for x, y in zip(a, b):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype and x.tobytes() == y.tobytes()
    assert np.asarray(a[0]).sum() > 0


def test_domain_counts_declare_clips_and_borrowed_bins():
    tab = PR.load("default")
    ok, E, T, v = _cells(20000, 42)
    rec = PR.domain_counts(tab, E, v, ok).record()
    assert rec["lookups"] == int(ok.sum())
    assert rec["axes"]["E"]["below"] > 0 and rec["axes"]["E"]["above"] > 0 and rec["axes"]["v"]["above"] > 0
    assert rec["axes"]["v"]["below"] == 0                  # below the first v edge is inside the window
    assert sum(rec["fallbacks"]["empty_bin"]["levels"].values()) == rec["lookups"]


def test_save_embeds_a_manifest_and_tampering_is_refused(tmp_path):
    raw = {k: np.array(v) for k, v in _legacy_load().items()}
    m = PR.save(tmp_path / "t.npz", raw, seeds={"build": 1})
    assert "source" in m["unrecorded"] and "seeds" not in m["unrecorded"]
    assert PR.load(tmp_path / "t.npz")["ratio"].tobytes() == raw["ratio"].tobytes()
    raw["ratio"][0] += np.float32(1e-3)
    np.savez_compressed(tmp_path / PR.DEFAULT_V1.name, **raw)
    shutil.copyfile(LS.sidecar_path(PR.DEFAULT_V1), LS.sidecar_path(tmp_path / PR.DEFAULT_V1.name))
    with pytest.raises(LS.LibraryError, match="content hash"):
        PR.load(tmp_path / PR.DEFAULT_V1.name)
