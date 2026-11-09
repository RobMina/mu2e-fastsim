"""Retained source snapshots of the mu2e-fastsim repository: hependpoints.snapshots with the Mu2e repository.

The URL and module rules are Mu2e configuration (mu2e_config: one repository, no submodules). The URL is read from
this module at call time (tests point it at a local fixture).
"""
from __future__ import annotations

from hependpoints import snapshots as _snapshots
from hependpoints.snapshots import SourceRepo  # noqa: F401

from . import mu2e_config as cfg

UMBRELLA_URL = cfg.UMBRELLA_URL


def _repo():
    return cfg.source_repo(UMBRELLA_URL)


def prepare_code(endpoint, base, ref, modules=None, branch="") -> dict:
    """Create a unique, retained snapshot at the exact commit of ref (hependpoints.snapshots); modules=None selects
    the repository's default modules (none for Mu2e)."""
    return _snapshots.prepare_code(endpoint, base, ref, _repo(), modules=modules, branch=branch)


def code_guard(snapshot) -> str:
    """Non-mutating Bash prefix verifying the snapshot before the runtime (hependpoints.snapshots.code_guard)."""
    return _snapshots.code_guard(snapshot, _repo())
