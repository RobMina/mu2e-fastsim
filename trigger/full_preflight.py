"""Check complete-history replay launch invariants before art starts."""
import argparse
from pathlib import Path

import full_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--expected-events', type=int, required=True)
    args = parser.parse_args()
    full_replay.preflight(args.prepared, args.directory, args.config.read_text(), args.expected_events)
    print('full replay preflight: count/configuration/ordered input bytes verified inside analysis binds')


if __name__ == '__main__':
    main()
