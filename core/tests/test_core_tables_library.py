"""The bundled Seltzer-Berger and Mott tables on the shared library core: bit-identical to the pre-migration loaders.

The legacy loaders below are frozen copies of ebrem.tables and emsc.mott_coefficients before the migration.
"""
import json
import shutil

import numpy as np
import pytest

from mu2efastsim import ebrem as B
from mu2efastsim import emsc as E
from mu2efastsim.library import hashing, manifest as LM, store as LS


def _legacy_tables(path=B.DATA):
    d = np.load(path)
    return {int(z): i for i, z in enumerate(d["z"])}, d["kappa"], d["lnT"], d["chi"]


def _legacy_mott(path=E.MOTT_DATA):
    return np.load(path)["coef"]


def _legacy_chi(tab, z, kappa, lnT):
    row, xs, ys, c = tab
    t = c[row[int(z)]]
    x = np.clip(np.asarray(kappa, float), xs[0], xs[-1])
    y = np.clip(np.asarray(lnT, float), ys[0], ys[-1])
    i, j = B._bin(xs, x), B._bin(ys, y)
    x1, x2, y1, y2 = xs[i], xs[i + 1], ys[j], ys[j + 1]
    return ((y2 - y) * (t[j, i] * (x2 - x) + t[j, i + 1] * (x - x1))
            + (y - y1) * (t[j + 1, i] * (x2 - x) + t[j + 1, i + 1] * (x - x1))) / ((x2 - x1) * (y2 - y1))


def test_bundled_tables_load_bit_identical_with_their_manifests():
    new, old = B.tables(), _legacy_tables()
    assert new[0] == old[0]
    for a, b in zip(new[1:], old[1:]):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert E.mott_coefficients().tobytes() == _legacy_mott().tobytes()
    for path, name in ((B.DATA, B.LIBRARY), (E.MOTT_DATA, E.MOTT_LIBRARY)):
        m = LS.load(path).manifest
        assert m["name"] == name and m["content_hash"] == hashing.content_hash(LS.load_arrays(path))
        assert set(m["unrecorded"]) == {"config_hash"}
    files = LS.load(B.DATA).manifest["source"]["files"]
    assert [f["file"] for f in files] == [f"br{z}" for z in sorted(new[0])]


def test_interpolated_values_are_bit_identical():
    rng = np.random.default_rng(1)
    kappa = np.concatenate([rng.uniform(0, 1.2, 2000), [0.0, 1e-12, 1.0]])
    lnT = rng.uniform(np.log(5e-4), np.log(2e4), kappa.size)
    old = _legacy_tables()
    for z in old[0]:
        assert B.chi(z, kappa, lnT).tobytes() == _legacy_chi(old, z, kappa, lnT).tobytes()


@pytest.mark.parametrize("writer,path", [(B.write_shipped_manifest, B.DATA), (E.write_shipped_mott_manifest,
                                                                              E.MOTT_DATA)])
def test_bundled_sidecars_are_reproducible(tmp_path, writer, path):
    copy = tmp_path / path.name
    shutil.copyfile(path, copy)
    writer(copy)
    assert json.loads(LS.sidecar_path(copy).read_text()) == json.loads(LS.sidecar_path(path).read_text())


def test_a_tampered_table_is_refused(tmp_path):
    arrays = {k: np.array(v) for k, v in LS.load_arrays(E.MOTT_DATA).items()}
    arrays["coef"][1, 0, 0] *= 1.0 + 1e-12
    bad = tmp_path / E.MOTT_DATA.name
    np.savez_compressed(bad, **arrays)
    shutil.copyfile(LS.sidecar_path(E.MOTT_DATA), LS.sidecar_path(bad))
    with pytest.raises(LS.LibraryError, match="content hash"):
        E.mott_coefficients.__wrapped__(bad)


def test_convert_embeds_the_manifest(tmp_path):
    src = tmp_path / "brem_SB"
    src.mkdir()
    kappa, lnT = np.linspace(0, 1, 4), np.linspace(-6, 9, 3)
    for z in (1, 6):
        vals = " ".join(f"{v:.6g}" for v in np.r_[kappa, lnT, np.arange(12.0) * z])
        (src / f"br{z}").write_text(f"0 4 3 {vals}\n")
    m = B.convert(src, tmp_path / "t.npz", zs=[6, 1])
    assert m["name"] == B.LIBRARY and [f["file"] for f in m["source"]["files"]] == ["br1", "br6"]
    lib = LS.load(tmp_path / "t.npz")
    assert LM.complete(lib.manifest) is False and lib.arrays["chi"].shape == (2, 3, 4)
