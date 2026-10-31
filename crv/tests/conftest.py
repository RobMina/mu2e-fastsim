"""Test-process defaults that bound memory (see mu2efastsim.testing for the shared hooks).

The coincidence kernels pad every launch to a fixed group batch sized for a GPU (16,384 groups at H=16), so a
small CPU test sample allocates the same multi-GB tensors as a production chunk. Tests default to the compact
dispatch, which pads the last batch to a bounded power of two and gives identical results; tests of the dispatch
modes set or clear CRVFASTSIM_COMPACT_DISPATCH themselves.
"""
import os

os.environ.setdefault("CRVFASTSIM_COMPACT_DISPATCH", "1")

try:
    from mu2efastsim.testing import (_release_jax_caches, pytest_addoption,  # noqa: F401
                                     pytest_collection_modifyitems, pytest_configure)
except ImportError:   # an older core without the shared hooks
    pass


import pytest  # noqa: E402

# Synthetic site configuration for tests of tools that name dCache or grid-sample paths (see site.env.example).
TEST_DCACHE, TEST_XROOTD, TEST_PNFS, TEST_OWNER = "/dcache-test", "root://door.test//ns", "/dcache-test/mu2e/scratch/users/tester/crv_fastsim", "tester"


@pytest.fixture
def dcache_site(monkeypatch):
    monkeypatch.setenv("CRVFS_DCACHE", TEST_DCACHE)
    monkeypatch.setenv("CRVFS_XROOTD", TEST_XROOTD)
    monkeypatch.setenv("CRVFS_PNFS", TEST_PNFS)
    monkeypatch.setenv("CRVFS_OWNER", TEST_OWNER)
