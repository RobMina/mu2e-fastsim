"""Tests of the fastsim MCP server: the tool package with a stand-in kit, and the generic wrapper.

Paths: this directory's parent (the two packages) and the umbrella's core and beam (the kit's geometry rules), so
the suite also runs without PYTHONPATH from inside fastsim-mcp/. The shared hooks of mu2efastsim.testing give it the CI
tier options (--runslow, --runweekly).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, ROOT / "core", ROOT / "beam", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# The shared fastsim test hooks (mu2efastsim.testing), as in the other suites: --runslow / --runweekly and the
# slow and weekly markers (the nightly run passes --runslow to every suite), and JAX caches released per module.
try:
    from mu2efastsim.testing import (_release_jax_caches, pytest_addoption,  # noqa: E402,F401
                                     pytest_collection_modifyitems, pytest_configure)
except ImportError:   # core not importable: still accept the CI tier options
    def pytest_addoption(parser):
        parser.addoption("--runslow", action="store_true", default=False)
        parser.addoption("--runweekly", action="store_true", default=False)
