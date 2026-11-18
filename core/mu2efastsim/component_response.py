"""Component interaction banks: now hepsimlib.component (hep-simlib, pinned in pyproject.toml); docs/component-response.md.

This module is replaced in sys.modules by hepsimlib.component itself, after the Mu2e profile is installed (import of
mu2efastsim.library), so ``mu2efastsim.component_response`` keeps every name, private helper and behaviour, and bank
names keep the ``secondaries.component.`` prefix.
"""
import sys

from . import library as _library  # noqa: F401  (installs the Mu2e profile)
import hepsimlib.component as _component

sys.modules[__name__] = _component
