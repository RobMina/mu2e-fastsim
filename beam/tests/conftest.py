"""Shared fastsim test hooks (mu2efastsim.testing): JAX caches released per module; @pytest.mark.slow opt-in."""
import sys

import pytest

try:
    from mu2efastsim.testing import (_release_jax_caches, pytest_addoption,  # noqa: F401
                                     pytest_collection_modifyitems, pytest_configure)
except ImportError:   # an older core without the shared hooks
    pass


@pytest.fixture(autouse=True, scope="module")
def _release_navigator_builds():
    """Drop the navigator builds a module made: each keeps its geometry and device lane tables alive otherwise."""
    yield
    for name in ("beamfastsim.flash_navigator_jax", "beamfastsim.cechain_jax"):
        module = sys.modules.get(name)
        if module is not None:
            module._BUILT.clear()
