"""Reject arrays outside the parallel-annular-plane stepping contract before JIT."""
from dataclasses import replace

import numpy as np
import pytest

from beamfastsim import geometry, transport
from test_transport import _geom, _uniform_map


@pytest.mark.parametrize("changes", [
    {"foil_z": np.array([])},
    {"foil_z": np.array([6000.0, 6000.0])},
    {"foil_z": np.array([6001.0, 6000.0])},
    {"foil_z": np.array([6000.0, np.nan])},
    {"foil_rin": np.array([0.0])},
    {"foil_rin": np.array([-1.0, 0.0])},
    {"foil_rout": np.array([0.0, 75.0])},
    {"foil_ht": np.array([0.0, 0.05])},
    {"foil_ht": np.array([[0.05, 0.05]])},
    {"foil_ht": np.array([11.0, 11.0])},
])
def test_invalid_foil_arrays(changes):
    g = replace(_geom(nfoil=2), **changes)
    with pytest.raises(ValueError, match="foil"):
        geometry.validate_foils(g, 5.0)


@pytest.mark.parametrize("ds", [0.0, -1.0, np.nan, np.inf, 20.01])
def test_invalid_step_or_spacing(ds):
    with pytest.raises(ValueError):
        transport.make_stepper(_uniform_map(), _geom(nfoil=2), transport.Params(ds=ds))


def test_spacing_equal_step_is_safe_for_half_open_crossing():
    geometry.validate_foils(_geom(nfoil=2), 20.0)


def test_positive_short_step():
    geometry.validate_foils(_geom(nfoil=2), 1.0)


@pytest.mark.parametrize("key", ["stoppingTarget.material", "stoppingTarget.rotation", "stoppingTarget.raidi",
                                "protonabsorber.thickness", "mu2e.solenoidOffset"])
def test_unsupported_overlay_changes(key):
    baseline = {"stoppingTarget.radii": [75.0], key: 1.0}
    with pytest.raises(ValueError, match=key):
        geometry.check_overlay(dict(baseline, **{key: 2.0}), baseline)


def test_overlay_allowed_keys_and_shadowing():
    base = {"stoppingTarget.radii": [75.0], "stoppingTarget.holeRadius": 20.0, "fixed": "value"}
    candidate = dict(base, **{"stoppingTarget.radii": [80.0], "stoppingTarget.holeRadii": [21.0]})
    assert geometry.check_overlay(candidate, base) == ["stoppingTarget.holeRadii", "stoppingTarget.radii"]
    candidate["stoppingTarget.holeRadius"] = 22.0
    with pytest.raises(ValueError, match="shadowed"):
        geometry.check_overlay(candidate, base)
    with pytest.raises(ValueError, match="fixed"):
        geometry.check_overlay({"stoppingTarget.radii": [80.0]}, base)


def test_cli_geometry_contract_follows_nested_includes(tmp_path, monkeypatch):
    from beamfastsim import __main__ as cli

    folder = tmp_path / "Offline" / "Mu2eG4" / "geom"
    folder.mkdir(parents=True)
    (folder / "geom_common.txt").write_text('double stoppingTarget.rotation = 0;\n')
    nested = tmp_path / "nested.txt"
    nested.write_text('#include "Offline/Mu2eG4/geom/geom_common.txt"\ndouble stoppingTarget.rotation = 1;\n')
    top = tmp_path / "candidate.txt"
    top.write_text('#include "nested.txt"\n')
    monkeypatch.setattr(cli, "OFFLINE_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="stoppingTarget.rotation"):
        cli._geom(str(top), target_contract=True)
