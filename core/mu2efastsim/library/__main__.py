"""Library files from the command line (docs/libraries.md).

    python3 -m mu2efastsim.library check FILE [--name NAME] [--version N] [--content-hash HEX]
    python3 -m mu2efastsim.library describe FILE        # the manifest, indented
    python3 -m mu2efastsim.library hash FILE            # file sha256 and content hash of the arrays

``check`` exits 0 and prints the provenance line when every load check passes, 1 with the problems otherwise.
"""
import argparse
import json
import sys

from . import hashing, store


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m mu2efastsim.library", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("file")
    c.add_argument("--name")
    c.add_argument("--version", type=int)
    c.add_argument("--content-hash")
    d = sub.add_parser("describe")
    d.add_argument("file")
    h = sub.add_parser("hash")
    h.add_argument("file")
    a = ap.parse_args(argv)
    if a.cmd == "hash":
        print(json.dumps(dict(file_sha256=hashing.file_sha256(a.file),
                              content_hash=hashing.content_hash(store.load_arrays(a.file)))))
        return 0
    expect = {k: v for k, v in (("name", getattr(a, "name", None)), ("version", getattr(a, "version", None)),
                                ("content_hash", getattr(a, "content_hash", None))) if v is not None}
    try:
        lib = store.load(a.file, expect=expect)
    except store.LibraryError as exc:
        print("\n".join(exc.problems), file=sys.stderr)
        return 1
    print(json.dumps(lib.manifest if a.cmd == "describe" else lib.provenance(), indent=1 if a.cmd == "describe"
                     else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
