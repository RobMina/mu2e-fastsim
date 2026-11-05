"""Verify a staged existing-reference payload before admitting it for an Offline audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zlib


def verify(path, size, adler32):
    digest, adler, total = hashlib.sha256(), 1, 0
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            adler = zlib.adler32(block, adler)
            total += len(block)
    result = dict(bytes=total, adler32=f"{adler & 0xffffffff:08x}", sha256=digest.hexdigest())
    if total != size or result["adler32"] != adler32.lower():
        raise ValueError(f"staged input integrity mismatch: {result}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--adler32", required=True)
    ap.add_argument("--admit-as", help="rename verified partial input; fail if destination exists")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.admit_as and Path(a.admit_as).exists():
        raise ValueError("admitted input already exists")
    before = os.stat(a.path)
    report = verify(a.path, a.size, a.adler32)
    after = os.stat(a.path)
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("input changed during integrity check")
    if a.admit_as:
        os.rename(a.path, a.admit_as)
    report.update(source_path=a.path, admitted_path=a.admit_as)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
