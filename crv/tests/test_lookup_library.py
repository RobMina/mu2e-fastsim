"""CRV v6 lookup tables and photon map on the shared library core: pinned by the run configurations, recorded as
unpinned otherwise, decoded bit-identically to the pre-migration reader (frozen copy ``_legacy_load``)."""
import os
import struct

import numpy as np
import pytest

from crvfastsim import runconfig
from crvfastsim.response import lookup, photons as P
from mu2efastsim.library import hashing, provenance as LP
from mu2efastsim.library.store import LibraryError

REAL = all(os.path.exists(os.path.join(lookup.lookup_dir(), f)) for f in
           ("LookupTable_6000_0", "LookupTable_2370_1", "LookupTable_3200_0", "photonMap.root"))
real = pytest.mark.skipif(not REAL, reason="CRV lookup tables not available (Mu2e-internal)")


# ----------------------------------------------------------------------------------- frozen pre-migration reader
def _legacy_load(name, directory=None):
    path = os.path.join(directory or lookup.lookup_dir(), f"LookupTable_{name}")
    with open(path, "rb") as fh:
        buf = fh.read()
    r = lookup._Reader(buf)
    v1, v2, refl = (int(x) for x in r.take("<i4", 3))
    r.take("<u1", 4)
    c = r.take("<f8", 10)
    if v1 != 6:
        raise ValueError("version")

    def cmap():
        n = r.size_t()
        return r.take("<f8", 2 * n).reshape(n, 2).copy()

    ck_s, ck_f = cmap(), cmap()
    edges = {k: r.vec("<f8").copy() for k in ("x", "y", "z", "beta", "theta", "phi", "r")}
    nb = {k: len(v) - 1 for k, v in edges.items()}
    n_scint = nb["x"] * nb["y"] * nb["z"]
    tables = {"scint": lookup._read_bins(r, n_scint), "ckscint": lookup._read_bins(r, n_scint * nb["beta"]),
              "ckfiber": lookup._read_bins(r, nb["beta"] * nb["theta"] * nb["phi"] * nb["r"] * nb["z"])}
    assert r.o == len(buf)
    return lookup.LookupTable(os.path.basename(path), (v1, v2), refl, *(float(x) for x in c), ck_s, ck_f, edges, tables)


# -------------------------------------------------------------------------------------- a synthetic v6 table file
def _write_table(path, rng):
    """A small, well-formed v6 binary table (a test double; the production name is allowed)."""
    edges = {"x": np.linspace(0, 10, 5), "y": np.linspace(-26, 26, 9), "z": np.linspace(-1000, 1000, 11),
             "beta": np.linspace(0.4, 1, 4), "theta": np.linspace(0, np.pi, 5), "phi": np.linspace(0, np.pi, 4),
             "r": np.linspace(0, 0.7, 4)}
    b = bytearray(struct.pack("<3i4x", 6, 0, 0))
    b += np.array([9.64, 25.42, 1000.0, 26.0, 2.0, 1.0, 0.7, 1.75, 0.126, 0.0], "<f8").tobytes()
    for cm in (np.array([[0.5, 0.0], [1.0, 200.0]]), np.array([[0.5, 0.0], [1.0, 150.0]])):
        b += struct.pack("<Q", len(cm)) + cm.astype("<f8").tobytes()
    for k in ("x", "y", "z", "beta", "theta", "phi", "r"):
        b += struct.pack("<Q", len(edges[k])) + edges[k].astype("<f8").tobytes()
    nb = {k: len(v) - 1 for k, v in edges.items()}
    ns = nb["x"] * nb["y"] * nb["z"]
    for n in (ns, ns * nb["beta"], nb["beta"] * nb["theta"] * nb["phi"] * nb["r"] * nb["z"]):
        for i in range(n):
            d = rng.integers(0, 40, rng.integers(1, 20)).astype(np.uint8)
            e = rng.integers(0, 40, rng.integers(1, 4)).astype(np.uint8)
            b += struct.pack("<If", i, float(rng.uniform(0, 0.01)))
            b += struct.pack("<Q", len(d)) + d.tobytes() + struct.pack("<Q", len(e)) + e.tobytes()
    path.write_bytes(bytes(b))


@pytest.fixture
def fake_dir(tmp_path):
    _write_table(tmp_path / "LookupTable_6000_0", np.random.default_rng(1))
    return tmp_path


def _same(a, b):
    A, B = lookup.arrays(a), lookup.arrays(b)
    return set(A) == set(B) and all(A[k].dtype == B[k].dtype and A[k].tobytes() == B[k].tobytes() for k in A)


def test_ad_hoc_load_is_allowed_and_recorded_unpinned(fake_dir):
    LP.clear()
    L = lookup.load("6000_0", str(fake_dir))                  # a test double with a production name: no refusal
    assert _same(L, _legacy_load("6000_0", str(fake_dir)))
    p = L.provenance
    assert p["status"] == "unpinned" and p["pin_set"] is None and p["name"] == "crv.lookup.6000_0"
    assert p["content_hash"] == hashing.content_hash(lookup.arrays(L))
    assert LP.loaded()[-1]["content_hash"] == p["content_hash"] and not LP.all_pinned()


def test_a_pin_is_checked_and_a_mismatch_refused(fake_dir):
    h = lookup.load("6000_0", str(fake_dir)).provenance["content_hash"]
    L = lookup.load("6000_0", str(fake_dir), pin={"content_hash": h, "set": "test"})
    assert L.provenance["status"] == "pinned" and L.provenance["pin_set"] == "test"
    with pytest.raises(LibraryError, match="differs from the pinned"):
        lookup.load("6000_0", str(fake_dir), pin={"content_hash": "0" * 64})


def test_pinned_load_tables_refuse_a_table_outside_the_set(fake_dir):
    cfg = runconfig.get("run1a_v01").response()               # T1, T2 -> 6000_0
    h = lookup.load("6000_0", str(fake_dir)).provenance["content_hash"]
    with lookup.pinned({"LookupTable_6000_0": {"content_hash": h, "set": "test"}}):
        tabs = P.load_tables(cfg, str(fake_dir))
    assert tabs["6000_0"].provenance["status"] == "pinned"
    with lookup.pinned({"LookupTable_2370_1": {"content_hash": h}}), pytest.raises(LibraryError, match="not in the"):
        P.load_tables(cfg, str(fake_dir))
    with lookup.pinned({"LookupTable_6000_0": {"content_hash": "0" * 64}}), pytest.raises(LibraryError):
        P.load_tables(cfg, str(fake_dir))
    assert P.load_tables(cfg, str(fake_dir))["6000_0"].provenance["status"] == "unpinned"   # outside the block


def test_substep_photons_are_bit_identical_with_the_legacy_reader(fake_dir):
    from test_photons import _rows
    new, old = lookup.load("6000_0", str(fake_dir)), _legacy_load("6000_0", str(fake_dir))
    rows = _rows(np.random.default_rng(2))
    a = P._substep_photons(new, rows, np.random.default_rng(3))
    b = P._substep_photons(old, rows, np.random.default_rng(3))
    assert all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(a, b)) and len(a[0]) > 0


def test_device_substep_photons_are_bit_identical(fake_dir):
    jax = pytest.importorskip("jax")
    from crvfastsim.response import photons_jax as PJ
    from test_photons import _rows
    rows = _rows(np.random.default_rng(4))
    a = PJ.substep_photons(PJ.DeviceTable(lookup.load("6000_0", str(fake_dir))), rows, jax.random.PRNGKey(5))
    b = PJ.substep_photons(PJ.DeviceTable(_legacy_load("6000_0", str(fake_dir))), rows, jax.random.PRNGKey(5))
    assert all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(a, b))


def test_domain_counts_report_the_stock_minus_one_bins(fake_dir):
    from mu2efastsim.library.domain import DomainCounter
    from test_photons import _rows
    L = lookup.load("6000_0", str(fake_dir))
    rows = _rows(np.random.default_rng(6))
    rows["beta"][:50] = 0.2                                    # below the beta edges: no Cerenkov photons
    rec = P.domain_counts_rows(L, rows, DomainCounter("crv.lookup.v6")).record()
    assert rec["lookups"] > 0 and rec["axes"]["beta"]["below"] > 0
    assert set(rec["axes"]) == {"x", "y", "z", "beta", "theta", "phi", "r"}


@real
def test_run_configurations_pin_the_shipped_optical_assets():
    pins = lookup.pins()
    assert set(pins) == {"LookupTable_6000_0", "LookupTable_2370_1", "LookupTable_3200_0", "photonMap.root"}
    LP.clear()
    for name in ("run1a_v01", "run1b_v01", "extracted_v04"):
        rc = runconfig.get(name)
        tabs = rc.lookup_tables()
        assert all(t.provenance["status"] == "pinned" for t in tabs.values())
        pmap = rc.photon_map()
        assert pmap.shape == (40, 40)
    assert LP.all_pinned()
    assert runconfig.get("run2_v10").asset_pin_map() is None   # the Run 2 tables are not in the set yet
    for t in ("6000_0", "2370_1", "3200_0"):
        new, old = lookup.load(t), _legacy_load(t)
        assert _same(new, old)
        assert hashing.file_sha256(os.path.join(lookup.lookup_dir(), f"LookupTable_{t}")) == \
            pins[f"LookupTable_{t}"]["file_sha256"]
