"""Library files from the command line (hepsimlib.library.__main__ with the Mu2e profile; docs/libraries.md).

    python3 -m mu2efastsim.library check FILE [--name NAME] [--version N] [--content-hash HEX]
    python3 -m mu2efastsim.library describe FILE        # the manifest, indented
    python3 -m mu2efastsim.library hash FILE            # file sha256 and content hash of the arrays

``check`` exits 0 and prints the provenance line when every load check passes, 1 with the problems otherwise.
"""
import sys

from hepsimlib.library.__main__ import main as _main


def main(argv=None):
    return _main(argv, prog="python3 -m mu2efastsim.library")


if __name__ == "__main__":
    sys.exit(main())
