"""The albedo library on the shared library core (mu2efastsim.library): bit-identical to the pre-migration code.

``_legacy_load`` is a frozen copy of trkfastsim.sim.albedo.Library.load before the migration; the migrated loader
must give the same arrays bit for bit, so every draw of the device sampler is the same in float32 and float64.
"""
import json
import shutil

import numpy as np
import pytest

from mu2efastsim.library import hashing, store as LS
from trkfastsim.sim import albedo as AL


def _legacy_load(path=AL.DEFAULT):
    with np.load(path, allow_pickle=False) as z:
        if not (np.array_equal(z["r_edges"], AL.R_EDGES) and np.array_equal(z["cos_edges"], AL.COS_EDGES)):
            raise ValueError(f"{path}: library built with other bins")
        return AL.Library(**{k: z[k] for k in AL.Library.FIELDS}, species=str(z["species"]), source=str(z["source"]))


def _exits(n, seed):
    """Fast tracks near the plane with directions over the whole exit domain and beyond the radius bins."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.0, 900.0, n)
    a = rng.uniform(0, 2 * np.pi, n)
    x = np.column_stack([AL.AXIS_XY[0] + r * np.cos(a), AL.AXIS_XY[1] + r * np.sin(a), np.full(n, AL.VD_Z - 5.0)])
    cz = rng.uniform(0.2, 1.0, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - cz ** 2)
    u = np.column_stack([st * np.cos(ph), st * np.sin(ph), cz])
    return x, u, rng.uniform(80, 105, n), rng.uniform(500, 1500, n), rng.uniform(size=n) < 0.8


def test_shipped_library_loads_bit_identical_with_its_manifest():
    new, old = AL.Library.load(), _legacy_load()
    for k in AL.Library.FIELDS:
        a, b = getattr(new, k), getattr(old, k)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), k
    assert (new.species, new.source) == (old.species, old.source)
    m = LS.load(AL.DEFAULT).manifest
    assert m["name"] == AL.LIBRARY and m["kind"] == "pool"
    assert m["content_hash"] == hashing.content_hash(LS.load_arrays(AL.DEFAULT))
    assert [a["name"] for a in m["axes"]] == ["r", "cos"]
    assert set(m["unrecorded"]) == {"release", "seeds", "config_hash"}


def test_shipped_sidecar_is_reproducible(tmp_path):
    copy = tmp_path / AL.DEFAULT.name
    shutil.copyfile(AL.DEFAULT, copy)
    AL.write_shipped_manifest(copy)
    assert json.loads(LS.sidecar_path(copy).read_text()) == json.loads(LS.sidecar_path(AL.DEFAULT).read_text())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("species", ["all", "ce"])
def test_device_draws_are_bit_identical_to_the_legacy_loader(dtype, species):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    x, u, p, t, m = _exits(20000, 21)
    args = tuple(jnp.asarray(a, dtype) for a in (x, u, p, t))
    key = jax.random.fold_in(jax.random.PRNGKey(5), 4)
    out = []
    for lib in (AL.Library.load(), _legacy_load()):
        dl = AL.DeviceLibrary(lib.select(species), dtype)
        g, phi, te = dl.pick(key, *args, jnp.asarray(m), 1.0)
        ret = dl.returns(g, phi, te, jnp.arange(len(p)), 4096)
        out.append({k: np.asarray(v) for k, v in dict(g=g, phi=phi, te=te, **ret).items()})
    for k in out[0]:
        assert out[0][k].dtype == out[1][k].dtype and out[0][k].tobytes() == out[1][k].tobytes(), k
    assert (out[0]["g"] >= 0).sum() > 0


def test_domain_counts_declare_empty_bins():
    lib = AL.Library.load()
    rng = np.random.default_rng(3)
    r, cos, ur = rng.uniform(0, 1200, 5000), rng.uniform(-1.2, 1.2, 5000), rng.normal(size=5000)
    rec = AL.domain_counts(lib, r, cos, ur).record()
    assert rec["lookups"] == 5000 and rec["axes"]["cos"]["below"] > 0 and rec["axes"]["cos"]["above"] > 0
    assert rec["axes"]["r"]["above"] == 0
    assert sum(rec["fallbacks"][AL.EMPTY_BIN]["levels"].values()) == 5000


def test_a_tampered_library_is_refused(tmp_path):
    arrays = {k: np.array(v) for k, v in LS.load_arrays(AL.DEFAULT).items()}
    arrays["dt"][0] += 1.0
    bad = tmp_path / AL.DEFAULT.name
    np.savez_compressed(bad, **arrays)
    with pytest.raises(LS.LibraryError, match="no manifest"):
        AL.Library.load(bad)
    shutil.copyfile(LS.sidecar_path(AL.DEFAULT), LS.sidecar_path(bad))
    with pytest.raises(LS.LibraryError, match="content hash"):
        AL.Library.load(bad)


def test_save_embeds_a_manifest(tmp_path):
    lib = AL.Library.load()
    m = lib.save(tmp_path / "a.npz", seeds={"production": "unknown"})
    assert m["seeds"] == {"production": "unknown"} and "source" in m["unrecorded"]
    back = AL.Library.load(tmp_path / "a.npz")
    assert all(getattr(back, k).tobytes() == getattr(lib, k).tobytes() for k in AL.Library.FIELDS)
