"""Check a staged parent prefix against catalogue bytes without inventing a source SHA receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import stage_parents


def verify(report, directory, count):
    rows = stage_parents.records(report)
    if type(count) is not int or not 1 <= count <= len(rows):
        raise ValueError('prefix count must be a positive bounded parent count')
    result = []
    for name, _, size, adler in rows[:count]:
        with (directory / name).open('rb') as source:
            reader = stage_parents.CheckedReader(source)
            while reader.read(8 * 1024 * 1024):
                pass
        result.append({'name': name, **stage_parents.checksum(reader, size, adler)})
    return {'files': result, 'total_bytes': sum(row['size'] for row in result),
            'catalog_byte_checks_passed': True, 'source_sha256_receipt_verified': False,
            'verified_parent_prefix': count, 'total_original_parents': len(rows),
            'scope': 'catalogue size/Adler-32 and measured staged SHA256 only; no source SHA256 or full-history certification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--parent-count', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse staged checksum report overwrite')
    report = verify(json.loads(args.report.read_text()), args.directory, args.parent_count)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"verified catalogue size/Adler-32 for {report['verified_parent_prefix']} staged parents; source SHA receipt NOT verified")


if __name__ == '__main__':
    main()
