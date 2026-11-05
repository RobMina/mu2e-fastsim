"""Synthetic geometry staging: include closure, hash-based reuse and no clobber."""
import pytest

from beamfastsim import target_geometry as T


def test_reuses_only_identical_geometry_and_never_overwrites_inputs(tmp_path, monkeypatch):
    files = {"Offline/geom.txt": b'#include "Offline/foil.txt"\n#include "Offline/field.txt"\n',
             "Offline/foil.txt": b"double foil.r = 1;\n", "Offline/field.txt": b"double field = 2;\n"}
    class Repo:
        def __init__(self, name):
            assert name == "mu2e.opensciencegrid.org"
        def read(self, path):
            return files[path.removeprefix("/release/")]
    monkeypatch.setattr(T.cvmfsget, "Repo", Repo)
    reuse, out = tmp_path / "old", tmp_path / "new"
    (reuse / "Offline").mkdir(parents=True)
    (reuse / "Offline/foil.txt").write_bytes(files["Offline/foil.txt"])
    (reuse / "Offline/field.txt").write_text("double field = 9;\n")
    r = T.stage(out, "/release", "Offline/geom.txt", reuse)
    assert len(r["files"]) == 3 and sum(x["reused"] for x in r["files"]) == 1
    assert (out / "Offline/foil.txt").is_symlink()
    assert not (out / "Offline/field.txt").is_symlink()
    assert (reuse / "Offline/field.txt").read_text() == "double field = 9;\n"
    with pytest.raises(FileExistsError):
        T.stage(out, "/release", "Offline/geom.txt", reuse)
