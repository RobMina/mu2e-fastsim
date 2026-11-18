"""Opt-in per-map field scales (mu2efastsim.bfield_scale): parsing, the ehelix reader, unset = unchanged.

The beam reader (beamfastsim.fieldmap) is checked in beam/tests/test_fieldmap_scale.py (core imports no detector).
"""
import numpy as np
import pytest

from mu2efastsim import bfield_scale, ehelix


def write_map(tmp_path, name="DSMap", n=(2, 3, 4)):
    """A tiny G4BL map: .header with the grid line and .bin with marker + doubles."""
    hdr = tmp_path / f"{name}.header"
    hdr.write_text(f"param\ngrid X0=-10 Y0=0 Z0=100 nX={n[0]} nY={n[1]} nZ={n[2]} dX=5 dY=5 dZ=10\n")
    rng = np.random.default_rng(3)
    B = rng.normal(size=(n[0] * n[1] * n[2], 3))
    with open(tmp_path / f"{name}.bin", "wb") as f:
        np.array([0xDEADBEEF], np.uint32).tofile(f)
        B.astype(np.float64).tofile(f)
    return str(hdr), B.reshape(n[0], n[1], n[2], 3)


def test_parse_and_limits():
    assert bfield_scale.parse("") == {}
    assert bfield_scale.parse("DSMap=0.9:DSExtension=0.9, TSdMap=1.05") == {
        "DSMap": 0.9, "DSExtension": 0.9, "TSdMap": 1.05}
    for bad in ("DSMap", "DSMap=x", "DSMapp=0.9", "DSMap=0.4", "DSMap=1.6", "DSMap=0.9:DSMap=1.1"):
        with pytest.raises(ValueError):
            bfield_scale.parse(bad)


def test_task_setting_drops_unit_factors():
    assert bfield_scale.setting({}) == ""
    assert bfield_scale.setting({"ds_bscale": 1.0, "tsd_bscale": 1.0}) == ""
    assert bfield_scale.setting({"ds_bscale": 0.9}) == "DSMap=0.9:DSExtension=0.9"
    assert bfield_scale.setting({"ds_bscale": 1.0, "tsd_bscale": 1.1}) == "TSdMap=1.1"


def test_unset_reads_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv(bfield_scale.ENV, raising=False)
    hdr, B = write_map(tmp_path)
    lo, d, n, grid = ehelix.read_g4bl(hdr)
    assert np.array_equal(grid, B) and grid.dtype == np.float64


def test_both_readers_scale_the_named_map_only(tmp_path, monkeypatch, capsys):
    hdr, B = write_map(tmp_path, "DSMap")
    other, B2 = write_map(tmp_path, "TSdMap")
    monkeypatch.setenv(bfield_scale.ENV, "DSMap=0.9")
    monkeypatch.setattr(bfield_scale, "_reported", set())
    assert np.allclose(ehelix.read_g4bl(hdr)[3], 0.9 * B, rtol=0, atol=0)
    assert np.array_equal(ehelix.read_g4bl(other)[3], B2)
    assert capsys.readouterr().err.count("bfield_scale: map DSMap scaled by 0.9") == 1
    assert bfield_scale.record() == {"DSMap": 0.9}


def test_bad_setting_fails_at_read(tmp_path, monkeypatch):
    hdr, _ = write_map(tmp_path)
    monkeypatch.setenv(bfield_scale.ENV, "DSmap=0.9")
    with pytest.raises(ValueError, match="unknown field map"):
        ehelix.read_g4bl(hdr)
