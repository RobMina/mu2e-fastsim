"""The beam reader (beamfastsim.fieldmap.load) under the opt-in field-map scales of mu2efastsim.bfield_scale."""
import numpy as np

from beamfastsim import fieldmap
from mu2efastsim import bfield_scale


def write_map(tmp_path, name="DSMap", n=(2, 3, 4)):
    """A tiny G4BL map: .header with the grid line and .bin with marker + doubles."""
    hdr = tmp_path / f"{name}.header"
    hdr.write_text(f"param\ngrid X0=-10 Y0=0 Z0=100 nX={n[0]} nY={n[1]} nZ={n[2]} dX=5 dY=5 dZ=10\n")
    B = np.random.default_rng(3).normal(size=(n[0] * n[1] * n[2], 3))
    with open(tmp_path / f"{name}.bin", "wb") as f:
        np.array([0xDEADBEEF], np.uint32).tofile(f)
        B.astype(np.float64).tofile(f)
    return str(hdr), B.reshape(n[0], n[1], n[2], 3)


def test_unset_load_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv(bfield_scale.ENV, raising=False)
    hdr, B = write_map(tmp_path)
    assert np.array_equal(fieldmap.load(hdr, dtype=np.float32).B, B.astype(np.float32))


def test_load_scales_the_named_map_only(tmp_path, monkeypatch):
    hdr, B = write_map(tmp_path, "DSMap")
    other, B2 = write_map(tmp_path, "TSdMap")
    monkeypatch.setenv(bfield_scale.ENV, "DSMap=0.9")
    monkeypatch.setattr(bfield_scale, "_reported", set())
    assert np.array_equal(fieldmap.load(hdr, dtype=np.float32).B, (B * 0.9).astype(np.float32))  # scaled in f64
    assert np.array_equal(fieldmap.load(other, dtype=np.float64).B, B2)
