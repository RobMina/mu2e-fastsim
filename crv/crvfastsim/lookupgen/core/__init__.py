"""Experiment-neutral core of the strip optical-photon table generator.

The implementation lives in the public package scint-fibre-fastsim (import name
``scintfibrefastsim``, pinned in ``crv/pyproject.toml`` and ``core/ci/requirements.txt``).
Importing this package makes ``crvfastsim.lookupgen.core.<module>`` the same module
object as ``scintfibrefastsim.core.<module>``, so every existing import path stays valid.

Modules: counter, binning, optics, materials, transport, run, limits.
"""
import importlib
import sys

MODULES = ("binning", "counter", "limits", "materials", "optics", "run", "transport")
for _name in MODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"scintfibrefastsim.core.{_name}")

from scintfibrefastsim.core import (  # noqa: E402,F401
    binning,
    counter,
    limits,
    materials,
    optics,
    run,
    transport,
)
