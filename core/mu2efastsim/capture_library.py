"""Audited Mu2eG4 component captures to a component bank (docs/component-training.md, "Captures to a bank").

The converter is hep-simlib's (hepsimlib.capture.bank), ported from this module; this module feeds it Mu2eG4 captures:
read with capture.read and capture.read_links, audited with the Mu2eG4 checks (capture.AUDIT_OPTIONS), and with the
generator kept unrecorded (the Mu2eG4 manifest does not record cuts and materials). Every capture must pass the strict
linked audit before it is converted; nothing is repaired, clamped or dropped. The bank is written by
component_response.save_library (hepsimlib.component with the Mu2e profile: secondaries.component.*);
bank_admissible stays false (docs/libraries.md, section 8).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hepsimlib.capture import bank as _bank
from hepsimlib.capture.bank import (EXIT_REASONS, PLACEMENT_ATOL, SUPPLIED_FIELDS, VOLUME_LABELS,  # noqa: F401
                                    _concat, _dataset, _inventory, _same_inventory, _volumes, inventory_hash)

from . import capture, component_response as cr  # noqa: F401  (component_response installs the Mu2e profile)
from .saved_g4 import require

EVENT = capture.EVENT


def read(path):
    """(tables, links, runinfo) of one Mu2eG4 capture file, as the converter reads them (no runinfo in Mu2eG4)."""
    return capture.generic_view(capture.read(path)), capture.read_links(path), None


def trials(tables, links, expected_events):
    """One audited capture as bank pieces: (component, inventory, entrances, particles, steps, exits, counts, info)."""
    return _bank.trials(capture.generic_view(tables), links, expected_events, **capture.AUDIT_OPTIONS)


def library_from_captures(paths, expected_events, *, relative_to=None, sample_info=None):
    """The Library of one component type from audited Mu2eG4 capture files, and the source/geometry provenance."""
    return _bank.library_from_captures(paths, expected_events, relative_to=relative_to, sample_info=sample_info,
                                       reader=read, record_generator=False, **capture.AUDIT_OPTIONS)


def convert(paths, output, expected_events, *, relative_to=None, sample_info=None, configuration=None, **provenance):
    """Write the bank of the given Mu2eG4 captures; returns its manifest (hepsimlib.capture.bank.convert)."""
    return _bank.convert(paths, output, expected_events, relative_to=relative_to, sample_info=sample_info,
                         configuration=configuration, reader=read, record_generator=False,
                         audit_options=capture.AUDIT_OPTIONS, **provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--expected-events", type=int, required=True, help="events in every capture file")
    parser.add_argument("--output", type=Path, required=True, help="bank .npz")
    parser.add_argument("--relative-to", type=Path, help="name datasets relative to this directory (default: cwd)")
    parser.add_argument("--provenance", type=Path, help="JSON object of further manifest fields (seeds, release, ...)")
    parser.add_argument("--sample-info", type=Path, help="JSON list, one object per capture, added to its sample")
    parser.add_argument("--configuration", help="bank name suffix: <bank_name>.<configuration>")
    args = parser.parse_args(argv)
    require(args.output.suffix == ".npz" and all(args.output.resolve() != p.resolve() for p in args.captures),
            "output must be a new .npz, not a capture")
    provenance = json.loads(args.provenance.read_text()) if args.provenance else {}
    info = json.loads(args.sample_info.read_text()) if args.sample_info else None
    manifest = convert(args.captures, args.output, args.expected_events, relative_to=args.relative_to,
                       sample_info=info, configuration=args.configuration, **provenance)
    print(json.dumps({k: manifest[k] for k in ("name", "kind", "content_hash", "unrecorded", "validation")}, indent=2))


if __name__ == "__main__":
    main()
