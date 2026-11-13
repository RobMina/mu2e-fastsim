"""Umbrella integration: the CE device chain's default digitiser options against the tracker digitiser."""
import pytest

pytest.importorskip("trkfastsim")
pytest.importorskip("jax")
from beamfastsim import cetracker as H                                    # noqa: E402
from trkfastsim.digi import digi_resident as DR                           # noqa: E402


def test_default_digi_electronics_are_valid_levers():
    """The default electronics levers of the resident digitiser validate (digi_resident.electronics_options) and keep
    the flat entries at most the pair capacity."""
    opts = DR.electronics_options(H.DIGI_ELECTRONICS)
    assert 1 <= opts["entries_per_straw"] <= 4 and opts["saturation_pool"] and opts["saturation_tiles"] == "bounded"
