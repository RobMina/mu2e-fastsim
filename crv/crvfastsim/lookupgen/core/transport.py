"""Lazy re-export of scintfibrefastsim.core.transport (needs JAX)."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("scintfibrefastsim.core.transport")
