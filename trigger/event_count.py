"""Verify finished art Events entry count from ROOT metadata only, highest cycle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(path, expected):
    import uproot

    if type(expected) is not int or expected <= 0:
        raise ValueError('expected event count must be positive')
    with uproot.open(path) as root:
        if 'Events' not in root:
            raise ValueError('missing art Events tree')
        tree = root['Events']
        if tree.classname != 'TTree':
            raise ValueError('Events object is not an art TTree')
        entries = tree.num_entries
        if entries != expected:
            raise ValueError(f'finished art Events entries {entries} != expected {expected}')
        cycles = [int(key.rsplit(';', 1)[1]) for key in root.keys(recursive=False) if key.startswith('Events;')]
        return {'events': entries, 'expected_events': expected, 'highest_cycle': max(cycles),
                'equal': True, 'scope': 'ROOT entry-count metadata only, not physics products or path states'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('art', type=Path)
    parser.add_argument('--expected-events', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.art, args.expected_events)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Events entries={report['events']} highest_cycle={report['highest_cycle']}")


if __name__ == '__main__':
    main()
