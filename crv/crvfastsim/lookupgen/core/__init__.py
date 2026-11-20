"""Experiment-neutral core of the strip optical-photon table generator.

The implementation lives in the public package scint-fibre-fastsim (import name
``scintfibrefastsim``, pinned in ``crv/pyproject.toml`` and ``core/ci/requirements.txt``).
Importing this package makes ``crvfastsim.lookupgen.core.<module>`` the same module
object as ``scintfibrefastsim.core.<module>``, so every existing import path stays valid.

``run`` and ``transport`` need JAX and are loaded lazily so no-JAX CI can still import
``optics`` / ``materials`` / geometry helpers.
"""
import importlib
import sys

__all__ = ["binning", "counter", "limits", "materials", "optics", "run", "transport"]

_EAGER = ("binning", "counter", "limits", "materials", "optics")
_LAZY = ("run", "transport")


def _install(name: str):
    mod = importlib.import_module(f"scintfibrefastsim.core.{name}")
    sys.modules[f"{__name__}.{name}"] = mod
    globals()[name] = mod
    return mod


for _name in _EAGER:
    _install(_name)


def __getattr__(name: str):
    if name in _LAZY:
        return _install(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
