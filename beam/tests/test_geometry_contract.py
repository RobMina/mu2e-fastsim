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
