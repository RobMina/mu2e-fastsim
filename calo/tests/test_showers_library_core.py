"""The calorimeter shower libraries on the shared library core (mu2efastsim.library): manifests for new libraries, the
registered pre-manifest CAL2 library loading as before, and every draw bit-identical to the pre-migration loader
(frozen copy ``_legacy_load``). The CAL2 checks run where the file is (CALOFS_SHOWER_LIBRARY)."""
import json
import os
import shutil

import numpy as np
import pytest

from calofastsim.showers import build, library as SL, load, sample, save
from mu2efastsim.library import hashing, store as LS

CAL2 = os.environ.get("CALOFS_SHOWER_LIBRARY", "")
cal2 = pytest.mark.skipif(not (CAL2 and os.path.exists(CAL2)), reason="CAL2 shower library not available")


def _legacy_load(path):
    with np.load(path, allow_pickle=False) as source:
        library = {k: source[k] for k in source.files if k != "metadata"}
        metadata = json.loads(str(source["metadata"]))
    return library, metadata


def _queries(library, n, seed):
    """Query entries: library entries with their energies and radii jittered."""
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(library["entry_energy"]), n)
    q = {k[6:]: np.asarray(v)[rows] for k, v in library.items() if k.startswith("entry_") and k[6:] in
         ("energy", "cosine", "radius", "time", "disk", "position", "direction")}
    q["energy"] = q["energy"] * rng.uniform(0.9, 1.1, n)
    q["radius"] = q["radius"] + rng.normal(0, 5, n).clip(-q["radius"] + 1e-3)
    return q


def _same(a, b):
    return set(a) == set(b) and all(np.asarray(a[k]).dtype == np.asarray(b[k]).dtype and
                                    np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes() for k in a)


def _fixture():
    from test_showers import fixture
    e, s = fixture()
    return build(e, s, development=np.array([True, True, True, False]))


def test_save_embeds_a_manifest_and_load_returns_library_and_metadata(tmp_path):
    lib = _fixture()
    m = save(tmp_path / "library.npz", lib, {"candidate": "test double", "events": 4}, seeds={"split": 1})
    assert m["name"] == "calo.showers.test_double" and m["kind"] == "pool"
    assert [a["name"] for a in m["axes"]] == ["energy", "cosine", "radius"]
    got, meta = load(tmp_path / "library.npz")
    assert meta == {"candidate": "test double", "events": 4} and _same(got, lib)


def test_an_unregistered_library_without_a_manifest_is_refused(tmp_path):
    lib = _fixture()
    np.savez_compressed(tmp_path / "old.npz", **lib, metadata=np.array(json.dumps({"events": 4})))
    with pytest.raises(LS.LibraryError, match="no manifest"):
        load(tmp_path / "old.npz")
    m = SL.write_manifest(tmp_path / "old.npz")                     # a sidecar makes it a checked library
    assert _same(load(tmp_path / "old.npz")[0], lib) and m["content_hash"] == hashing.content_hash(
        dict(lib, metadata=np.array(json.dumps({"events": 4}))))


def test_domain_counts_report_clipped_entries():
    lib = _fixture()
    rec = SL.domain_counts(lib, np.array([10.0, 200.0, 60.0]), np.array([0.5, 0.5, 0.5]),
                           np.array([500.0, 500.0, 2000.0])).record()
    assert rec["lookups"] == 3 and rec["axes"]["energy"]["above"] == 1 and rec["axes"]["radius"]["above"] == 1


@cal2
def test_registered_cal2_library_loads_as_before():
    assert hashing.file_sha256(CAL2) in SL.registered()
    (new, meta), (old, ometa) = load(CAL2), _legacy_load(CAL2)
    assert meta == ometa and _same(new, old)


@cal2
@pytest.mark.parametrize("neighbors", [0, 8])
def test_cal2_draws_are_bit_identical(neighbors):
    new, _ = load(CAL2)
    old, _ = _legacy_load(CAL2)
    q = _queries(old, 5000, 7)
    kw = dict(orientation="disk", radius_neighbors=neighbors)
    a = sample(new, q, np.random.default_rng(8), **kw)
    b = sample(old, q, np.random.default_rng(8), **kw)
    assert _same(a, b) and (np.asarray(a["donor"]) >= 0).any()


@cal2
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cal2_disk_geometry_is_bit_identical(dtype):
    from calofastsim.showers import handoff_device as HD
    a = HD.disk_geometry(load(CAL2)[0], dtype)
    b = HD.disk_geometry(_legacy_load(CAL2)[0], dtype)
    flat = lambda g: g if isinstance(g, dict) else dict(enumerate(g)) if isinstance(g, (tuple, list)) else {0: g}
    assert _same(flat(a), flat(b))


@cal2
def test_a_changed_copy_of_cal2_is_refused_and_a_sidecar_restores_it(tmp_path):
    copy = tmp_path / "shower_library_cal2.npz"
    lib, meta = _legacy_load(CAL2)
    meta = dict(meta, note="re-saved copy")                       # same library arrays, other bytes
    np.savez_compressed(copy, **lib, metadata=np.array(json.dumps(meta, sort_keys=True)))
    assert hashing.file_sha256(copy) not in SL.registered()
    with pytest.raises(LS.LibraryError, match="no manifest"):
        load(copy)
    SL.write_manifest(copy, source=dict(description="CAL2 production candidate", samples=[]))
    assert _same(load(copy)[0], lib)
    shutil.copyfile(CAL2, tmp_path / "pinned.npz")
    assert _same(load(tmp_path / "pinned.npz")[0], lib)           # the registered bytes load from any path
