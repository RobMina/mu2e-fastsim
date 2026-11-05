"""Stream catalogued parent bytes as a tar archive, or verify staged parents; never use inside a synced tree."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from urllib.parse import urlsplit
import zlib


class CheckedReader:
    def __init__(self, source):
        self.source = source
        self.size = 0
        self.adler = 1
        self.sha = hashlib.sha256()

    def read(self, size=-1):
        data = self.source.read(size)
        self.size += len(data)
        self.adler = zlib.adler32(data, self.adler)
        self.sha.update(data)
        return data


def records(report):
    parents = report['parents']
    catalog = report['catalog']['files']
    if len(parents) != len(catalog) or not parents:
        raise ValueError('parent/catalog counts differ')
    resolved = report.get('resolved_source_paths')
    if resolved is not None and ([row['did'] for row in resolved] != [parent['did'] for parent in parents]
            or len(resolved) != len(parents)):
        raise ValueError('resolved source paths differ from original parent identities/order')
    names = []
    for ordinal, (parent, record) in enumerate(zip(parents, catalog)):
        name = parent['did'].split(':', 1)[1]
        if Path(name).name != name or name != record['name'] or record.get('retired'):
            raise ValueError('unsafe or inconsistent parent name')
        path = resolved[ordinal]['path'] if resolved is not None else urlsplit(parent['url']).path
        if not path.startswith('/') or '..' in Path(path).parts or Path(path).name != name:
            raise ValueError('unexpected source namespace')
        size = record['size']
        checksum = record['checksums']['adler32'].lower()
        if type(size) is not int or size <= 0 or len(checksum) != 8 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError('invalid expected size/checksum')
        names.append((name, path, size, checksum))
    if len(set(name for name, _, _, _ in names)) != len(names):
        raise ValueError('duplicate parent name')
    return names


def checksum(reader, expected_size, adler):
    if reader.size != expected_size or f'{reader.adler & 0xffffffff:08x}' != adler:
        raise ValueError('parent byte count or catalog adler32 mismatch')
    return {'size': reader.size, 'adler32': adler, 'sha256': reader.sha.hexdigest()}


def stream(report, destination):
    receipt = []
    with tarfile.open(fileobj=destination, mode='w|') as archive:
        for name, path, size, adler in records(report):
            with open(path, 'rb') as source:
                reader = CheckedReader(source)
                header = tarfile.TarInfo(name)
                header.size = size
                archive.addfile(header, reader)
                if source.read(1):
                    raise ValueError('source larger than catalog size')
                receipt.append({'name': name, **checksum(reader, size, adler)})
            print(f'verified streamed parent {name}', file=sys.stderr, flush=True)
        payload = json.dumps(receipt, indent=2).encode() + b'\n'
        header = tarfile.TarInfo('stream-receipt.json')
        header.size = len(payload)
        archive.addfile(header, io.BytesIO(payload))


def verify(report, directory):
    result = []
    for name, _, size, adler in records(report):
        with (directory / name).open('rb') as source:
            reader = CheckedReader(source)
            while reader.read(8 * 1024 * 1024):
                pass
        result.append({'name': name, **checksum(reader, size, adler)})
    if json.loads((directory / 'stream-receipt.json').read_text()) != result:
        raise ValueError('received parent checksums differ from source stream receipt')
    return {'files': result, 'total_bytes': sum(row['size'] for row in result), 'equal': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stream', action='store_true')
    group.add_argument('--verify', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if args.stream:
        stream(report, sys.stdout.buffer)
    else:
        if args.output is None:
            raise ValueError('verification needs --output')
        result = verify(report, args.verify)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(f"verified {len(result['files'])} parents, bytes={result['total_bytes']}")


if __name__ == '__main__':
    main()
