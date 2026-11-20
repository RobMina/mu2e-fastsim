"""Per-step and per-event energy accounting of linked Mu2eG4 component captures; diagnostic only, never bank admission.

The accounting itself (its balance per step and per event, and the declared Geant4 terms) is hep-simlib's
(hepsimlib.capture.accounting), ported from this module; this module runs it on Mu2eG4 captures, read with
capture.read and audited with the Mu2eG4 checks (capture.AUDIT_OPTIONS).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hepsimlib.capture import accounting as _generic
from hepsimlib.capture.accounting import (BOOST_ULPS, DECAY, DECLARED, DESTROYING, ELECTRON, EM, KILLED,  # noqa: F401
                                          MASS_ROUNDING, MUONS, NEUTRINOS, PHOTON, POSITRON, PROTON, TRANSPORT,
                                          electron_mass_bound)

from . import capture
from .saved_g4 import require


def report(tables, links, expected_events, *, rtol=1e-9, atol=1e-9):
    """Audit with links and the Mu2eG4 checks, verify the manifest, then account every inside step and every event."""
    return _generic.report(capture.generic_view(tables), links, expected_events, rtol=rtol, atol=atol,
                           **capture.AUDIT_OPTIONS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--directory", default="dump")
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-closure", action="store_true", help="exit nonzero, writing no report, on any open balance")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "output already exists; refusing to clobber")
    tables = capture.read(args.input, args.directory)
    links = capture.read_links(args.input, args.directory)
    result = report(tables, links, args.expected_events)
    result["input"] = str(args.input)
    if args.require_closure:
        require(result["energy_accounting_passed"], "energy accounting does not close")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
