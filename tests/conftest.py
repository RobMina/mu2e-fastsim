"""Shared fastsim test hooks (mu2efastsim.testing): JAX caches released per module; @pytest.mark.slow opt-in."""
from mu2efastsim.testing import (_release_jax_caches, pytest_collection_modifyitems,  # noqa: F401
                                 pytest_configure)
from mu2efastsim.testing import pytest_addoption as _shared_addoption


def pytest_addoption(parser):
    # One pytest run can also load calo/tests/conftest.py (the shower job): --runslow is registered once.
    try:
        _shared_addoption(parser)
    except ValueError:
        pass
