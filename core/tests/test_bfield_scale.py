"""Opt-in field-map scale factors (mu2efastsim.bfield_scale) and the read_g4bl hook."""
import subprocess

import numpy as np
import pytest

from mu2efastsim import bfield_scale as S
from mu2efastsim import ehelix


def _map(tmp_path, name, rng):
    """A 2x3x4 G4BL map (header + marker + Hep3Vector doubles) with random values."""
    n = (2, 3, 4)
    (tmp_path / f"{name}.header").write_text(
        "param maxline=1000\ngrid X0=-10 Y0=-20 Z0=3000 nX=2 nY=3 nZ=4 dX=25 dY=25 dZ=50\n")
    B = rng.normal(size=n + (3,))
    with open(tmp_path / f"{name}.bin", "wb") as f:
        np.array([0xDEADBEEF], np.uint32).tofile(f)
        B.astype(np.float64).tofile(f)
    return str(tmp_path / f"{name}.header"), B


@pytest.fixture(autouse=True)
def _unset(monkeypatch):
    monkeypatch.delenv(S.ENV, raising=False)
    S._reported.clear()


def test_parse():
    assert S.parse("") == {} and S.parse(None) == {} and S.parse(" , ") == {}
    assert S.parse("DSMap=0.9, DSExtension=0.9,TSdMap=1.0") == {"DSMap": 0.9, "DSExtension": 0.9, "TSdMap": 1.0}
    for bad in ("DSMap", "DSMap=", "DSMap=x", "Foo=0.9", "dsmap=0.9", "DSMap=0.4", "DSMap=1.6", "DSMap=nan",
                "DSMap=inf", "DSMap=0.9,DSMap=0.8"):
        with pytest.raises(ValueError, match=S.ENV):
            S.parse(bad)


def test_unset_leaves_maps_identical(tmp_path):
    header, B = _map(tmp_path, "DSMap", np.random.default_rng(1))
    lo, d, n, got = ehelix.read_g4bl(header)
    assert got.dtype == np.float64 and np.array_equal(got, B)
    assert list(n) == [2, 3, 4] and list(lo) == [-10, -20, 3000] and list(d) == [25, 25, 50]
    assert S.scale_for(header) == 1.0 and S.provenance() == {}


def test_named_map_scaled_others_untouched(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(2)
    ds, Bds = _map(tmp_path, "DSMap", rng)
    tsd, Btsd = _map(tmp_path, "TSdMap", rng)
    monkeypatch.setenv(S.ENV, "DSMap=0.9,DSExtension=0.9")
    assert np.array_equal(ehelix.read_g4bl(ds)[3], Bds * 0.9)
    assert np.array_equal(ehelix.read_g4bl(tsd)[3], Btsd)
    ehelix.read_g4bl(ds)
    assert capsys.readouterr().err.count("map DSMap scaled by 0.9") == 1
    assert S.provenance() == {"bfield_mapscale": {"DSMap": 0.9, "DSExtension": 0.9}}


def test_unknown_map_fails_at_read(tmp_path, monkeypatch):
    header, _ = _map(tmp_path, "DSMap", np.random.default_rng(3))
    monkeypatch.setenv(S.ENV, "DSmap=0.9")
    with pytest.raises(ValueError, match="DSmap"):
        ehelix.read_g4bl(header)


def test_kit_exports():
    assert S.kit_exports({}) == "" and S.kit_exports({"ds_bscale": 1.0, "tsd_bscale": 1}) == ""
    assert S.kit_value({"ds_bscale": 0.9}) == "DSMap=0.9,DSExtension=0.9"
    assert S.kit_value({"ds_bscale": 1.1, "tsd_bscale": 0.95}) == "DSMap=1.1,DSExtension=1.1,TSdMap=0.95"
    with pytest.raises(ValueError):
        S.kit_value({"tsd_bscale": 2.0})
    line = S.kit_exports({"ds_bscale": 0.9})
    echo = 'echo "$' + S.ENV + '|$APPTAINERENV_' + S.ENV + '"'
    out = subprocess.run(["bash", "-c", line + echo], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "DSMap=0.9,DSExtension=0.9|DSMap=0.9,DSExtension=0.9"
