"""Load pinned CRY 1.7 distributions; host setup only, no shower draws.

The loader lives in the experiment-neutral package cryjax (cryjax.tables, extracted from this module on 2026-11-18).
This module keeps the import path and the Mu2e default: the sea-level table under $CRYDATAPATH or /opt/cry/data
(the fastsim image). Notices: docs/third-party/cry.txt.
"""
import os
from pathlib import Path

from cryjax import tables as _tables
from cryjax.tables import PDG, SEA_LEVEL_SHA256, SPECIES, Pdf, _spectrum, parse  # noqa: F401


def load(path=None, *, latitude=41.8, when="2021-06-21"):
    """Verified sea-level 300 m/all-species setup matching the extracted CPU reference."""
    path = Path(path or Path(os.environ.get("CRYDATAPATH", "/opt/cry/data")) / "cosmics_0.data")
    return _tables.load(path, altitude_m=0, latitude=latitude, when=when)


def main(argv=None):
    _tables.main(argv)


if __name__ == "__main__":
    main()
