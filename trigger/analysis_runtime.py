"""Record the selected analysis interpreter and imported transport modules without transporting events."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

import integrity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modules', nargs='+', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse runtime receipt overwrite')
    modules = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in args.modules}
    report = {'python': sys.version, 'executable': sys.executable, 'modules': modules,
        'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'uproot', 'awkward')},
        'module_files': integrity.files(modules.values()), 'scope': 'import/runtime preflight, not physics validation'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
