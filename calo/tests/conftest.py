"""Shared fastsim test hooks (mu2efastsim.testing): JAX caches released per module; @pytest.mark.slow opt-in."""
try:
    from mu2efastsim.testing import (_release_jax_caches, pytest_collection_modifyitems,  # noqa: F401
                                     pytest_configure)
    from mu2efastsim.testing import pytest_addoption as _addoption
except ImportError:   # an older core without the shared hooks
    pass
else:
    def pytest_addoption(parser):
        # The shower workflow runs these tests and the umbrella tests/ in one session: the first conftest adds --runslow
        try:
            _addoption(parser)
        except ValueError:
            pass
