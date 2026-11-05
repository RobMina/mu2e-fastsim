#!/usr/bin/env python3
"""Run structural pytest against an explicit read-only audited checkout; no build."""
import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline', type=Path, required=True)
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    raise SystemExit(subprocess.call([
        sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider', '-q',
        str(here / 'tests'), '--offline', str(args.offline.resolve())]))
