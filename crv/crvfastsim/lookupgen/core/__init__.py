"""Experiment-neutral core of the strip optical-photon table generator.

The implementation lives in the public package scint-fibre-fastsim (import name
``scintfibrefastsim``, pinned in ``crv/pyproject.toml`` and ``core/ci/requirements.txt``).
Importing this package makes ``crvfastsim.lookupgen.core.<module>`` the same module
object as ``scintfibrefastsim.core.<module>``, so every existing import path stays valid.

Eager modules are installed here. ``run`` and ``transport`` need JAX: thin shim files
in this directory re-export them on first import (so no-JAX CI can still import
``optics`` / ``materials``). ``__getattr__`` only triggers that same submodule import —
one lazy mechanism, not a second loader.
"""
import importlib
import sys

__all__ = ["binning", "counter", "limits", "materials", "optics", "run", "transport"]

_EAGER = ("binning", "counter", "limits", "materials", "optics")
_LAZY = ("run", "transport")

for _name in _EAGER:
    _mod = importlib.import_module(f"scintfibrefastsim.core.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _mod
    globals()[_name] = _mod


def __getattr__(name: str):
    if name in _LAZY:
        # loads the shim file in this directory (single mechanism)
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
