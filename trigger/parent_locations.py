"""Resolve exact catalogue parent DIDs to read-only staging paths without changing production history."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess


def resolve(report, storage='tape', runner=subprocess.run):
    if storage not in ('tape', 'disk', 'scratch'):
        raise ValueError('unsupported storage class')
    parents = report['parents']
    dids = [parent['did'] for parent in parents]
    names = {did.split(':', 1)[1]: did for did in dids}
    if not dids or len(set(dids)) != len(dids) or len(names) != len(dids):
        raise ValueError('empty or duplicate parent identities')
    result = runner(['mdh', 'print-url', '-s', 'path', '-l', storage, '-'],
                    input='\n'.join(dids) + '\n', text=True, capture_output=True, check=True, timeout=600)
    paths = result.stdout.splitlines()
    if len(paths) != len(dids):
        raise ValueError('location query returned missing or extra parent paths')
    resolved = {}
    for value in paths:
        path = Path(value)
        if not path.is_absolute() or '..' in path.parts or any(c.isspace() for c in value):
            raise ValueError('location query did not return a safe absolute path')
        did = names.get(path.name)
        if did is None or did in resolved:
            raise ValueError('location query returned unknown or duplicate parent identity')
        resolved[did] = value
    output = deepcopy(report)
    output['resolved_source_paths'] = [{'did': did, 'path': resolved[did]} for did in dids]
    output['location_query'] = {'tool': 'mdh print-url', 'format': 'path', 'storage': storage,
        'stdout_sha256': hashlib.sha256(result.stdout.encode()).hexdigest(),
        'scope': 'catalogue-approved storage paths; no existence, locality, checksum or payload read'}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--storage', choices=['tape', 'disk', 'scratch'], default='tape')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse location report overwrite')
    output = resolve(json.loads(args.report.read_text()), args.storage)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(f"resolved {len(output['resolved_source_paths'])} exact parent paths without reading payloads")


if __name__ == '__main__':
    main()
