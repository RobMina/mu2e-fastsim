"""The sampled-row library on the shared library core (mu2efastsim.library): bit-identical to the pre-migration code.

The legacy functions below are frozen copies of trkfastsim.recoresp.nrow as of PR #169 (before the migration); every
output of the migrated module must equal theirs bit for bit, on the shipped library, in float32 and float64.
"""
import json
import shutil

import numpy as np
import pytest

from mu2efastsim.library import hashing, manifest as LM, store as LS
from trkfastsim.recoresp import nrow as NR
from trkfastsim.recoresp import table as RT


# ------------------------------------------------------------------------------- frozen pre-migration code (PR #169)
def _legacy_bins(edges, v):
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)


def _legacy_load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def _legacy_sample_numpy(tab, rng, has_entry, x, p):
    has = np.asarray(has_entry, bool)
    pm, td, _, rmax = RT.entry_truth(np.asarray(x, float), np.asarray(p, float), tab["axis_xy"], float(tab["bz"]))
    ntb, nrb = len(tab["tandip_edges"]) - 1, len(tab["rmax_edges"]) - 1
    c = (_legacy_bins(tab["p_edges"], pm) * ntb + _legacy_bins(tab["tandip_edges"], td)) * nrb \
        + _legacy_bins(tab["rmax_edges"], rmax)
    cnt = tab["cell_count"][c]
    j = tab["cell_start"][c] + np.minimum((rng.uniform(size=len(c)) * cnt).astype(np.int64), cnt - 1)
    return np.where(has, tab["pool"][j], 0).astype(np.int64)


def _legacy_sample(dn, key, has_entry, x, p):
    import jax
    import jax.numpy as jnp
    from trkfastsim.recoresp import device as RD
    dt = x.dtype
    pm, td, _, rmax = RD.entry_truth(x, p, dn.axis_xy.astype(dt), dn.bz)
    ntb, nrb = dn.tandip_edges.shape[0] - 1, dn.rmax_edges.shape[0] - 1
    c = (RD._bin(dn.p_edges.astype(dt), pm) * ntb + RD._bin(dn.tandip_edges.astype(dt), td)) * nrb \
        + RD._bin(dn.rmax_edges.astype(dt), rmax)
    cnt = dn.cell_count[c]
    u = jax.random.uniform(key, (x.shape[0],), dt)
    j = dn.cell_start[c] + jnp.minimum((u * cnt).astype(jnp.int32), cnt - 1)
    return jnp.where(has_entry, dn.pool[j], 0)


# ------------------------------------------------------------------------------------------------------ inputs
def _entries(n, seed, axis):
    """Entry states across and beyond the cell domain (|p| 0-260 MeV/c, tan dip 0.1-2, radii 300-800 mm), with a
    tenth of the rows without a tracker entry."""
    rng = np.random.default_rng(seed)
    pm = np.concatenate([rng.uniform(90, 106, n - n // 4), rng.uniform(0, 260, n // 4)])
    td = rng.uniform(0.1, 2.0, n)
    pt = pm / np.sqrt(1 + td ** 2)
    phi = rng.uniform(0, 2 * np.pi, n)
    p = np.column_stack([pt * np.cos(phi), pt * np.sin(phi), pt * td])
    r = rng.uniform(300, 800, n)
    a = rng.uniform(0, 2 * np.pi, n)
    x = np.column_stack([axis[0] + r * np.cos(a), axis[1] + r * np.sin(a), np.full(n, -1500.0)])
    return rng.uniform(size=n) < 0.9, x, p


@pytest.fixture(scope="module")
def shipped():
    return NR.load("default")


# ------------------------------------------------------------------------------------------------------- tests
def test_shipped_library_loads_bit_identical_with_its_manifest(shipped):
    raw = _legacy_load(NR.DEFAULT)
    assert set(shipped) == set(raw)
    for k in raw:
        assert raw[k].dtype == shipped[k].dtype and raw[k].tobytes() == shipped[k].tobytes(), k
        assert not shipped[k].flags.writeable
    lib = NR.load_library("default")
    m = lib.manifest
    assert m["name"] == NR.LIBRARY and m["version"] == 1 and m["kind"] == "pool"
    assert m["content_hash"] == hashing.content_hash(raw)
    assert [a["name"] for a in m["axes"]] == ["p", "tandip", "rmax"]
    assert set(m["unrecorded"]) == {"release", "config_hash"}       # the gaps are declared, not invented
    assert m["legacy"]["schema"] == NR.SCHEMA
    assert lib.grid().size == len(raw["cell_count"])


def test_shipped_sidecar_is_reproducible(tmp_path):
    copy = tmp_path / NR.DEFAULT.name
    shutil.copyfile(NR.DEFAULT, copy)
    NR.write_shipped_manifest(copy)
    assert json.loads(LS.sidecar_path(copy).read_text()) == json.loads(LS.sidecar_path(NR.DEFAULT).read_text())


def test_host_sample_is_bit_identical_to_the_legacy_code(shipped):
    raw = _legacy_load(NR.DEFAULT)
    has, x, p = _entries(60000, 11, raw["axis_xy"])
    new = NR.sample_numpy(shipped, np.random.default_rng(12), has, x, p)
    old = _legacy_sample_numpy(raw, np.random.default_rng(12), has, x, p)
    assert new.dtype == old.dtype and new.tobytes() == old.tobytes()
    assert np.all(new[~has] == 0) and np.isin(new[has], raw["pool"]).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_device_sample_is_bit_identical_to_the_legacy_code(shipped, dtype):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    raw = _legacy_load(NR.DEFAULT)
    has, x, p = _entries(40000, 13, raw["axis_xy"])
    dn_new, dn_old = NR.to_device(shipped, dtype), NR.to_device(raw, dtype)
    args = (jnp.asarray(has), jnp.asarray(x, dtype), jnp.asarray(p, dtype))
    key = jax.random.fold_in(jax.random.PRNGKey(3), 6)
    new = np.asarray(jax.jit(NR.sample)(dn_new, key, *args))
    old = np.asarray(jax.jit(_legacy_sample)(dn_old, key, *args))
    assert new.dtype == old.dtype and new.tobytes() == old.tobytes()


def test_domain_counts_declare_clips_and_fallbacks(shipped):
    has, x, p = _entries(20000, 14, shipped["axis_xy"])
    rec = NR.domain_counts(shipped, has, x, p).record()
    assert rec["lookups"] == int(has.sum())
    assert set(rec["axes"]) == {"p", "tandip", "rmax"}
    assert rec["axes"]["p"]["above"] > 0 and rec["axes"]["tandip"]["below"] > 0     # the inputs reach beyond
    assert 0 < rec["out_of_domain_fraction"] < 1
    assert NR.FALLBACK in rec["fallbacks"]
    lv = rec["fallbacks"][NR.FALLBACK]["levels"]
    assert sum(lv.values()) == rec["lookups"]


def test_a_tampered_or_unmanifested_library_is_refused(tmp_path, shipped):
    tab = {k: np.array(v) for k, v in shipped.items()}
    tab["pool"][0] += 1
    bad = tmp_path / NR.DEFAULT.name
    np.savez_compressed(bad, **tab)
    with pytest.raises(LS.LibraryError, match="no manifest"):
        NR.load(bad)
    shutil.copyfile(LS.sidecar_path(NR.DEFAULT), LS.sidecar_path(bad))
    with pytest.raises(LS.LibraryError, match="content hash"):
        NR.load(bad)


def test_save_embeds_a_manifest_with_declared_gaps(tmp_path):
    rng = np.random.default_rng(1)
    n = 5000
    has, x, p = _entries(n, 15, np.array([-3904.0, 0.0]))
    tab = NR.build(has, rng.poisson(40, n), x, p, np.array([-3904.0, 0.0]), 1.0)
    path = tmp_path / "lib.npz"
    m = NR.save(path, tab, seeds={"training": 15})
    assert m["seeds"] == {"training": 15} and "seeds" not in m["unrecorded"] and "source" in m["unrecorded"]
    assert not LM.complete(m)
    got = NR.load(path)
    assert all(got[k].tobytes() == np.asarray(tab[k]).tobytes() for k in tab)
