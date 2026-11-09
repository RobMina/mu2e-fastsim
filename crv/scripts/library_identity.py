"""Content identity of tier-1b libraries: per tree and column, a hash of dtype, shape and bytes.

    python3 scripts/library_identity.py A.root B.root     prints differing trees/columns; exit 1 if any differ
    python3 scripts/library_identity.py build --run-config NAME --gen G.root [...] --out L.root STEPS.root [...]
        the historical (event, bar) build (tier1b.build without per_primary, roots on) on a named geometry: the
        default build path on donors with several muons per event, for an identity check between code versions
"""
import argparse
import hashlib
import json
import sys

import numpy as np

from crvfastsim import tier1b
from mu2efastsim import tables as core_tables


def digests(path):
    out = {}
    for tree in tier1b.LIB_TREES:
        frame = core_tables.read(path, tree)
        for col in frame.columns:
            a = np.ascontiguousarray(frame[col].to_numpy())
            out[f"{tree}.{col}"] = hashlib.sha256(f"{a.dtype}{a.shape}".encode() + a.tobytes()).hexdigest()
        out[f"{tree}#rows"] = str(len(frame))
    return out


def build(argv):
    from crvfastsim import runconfig
    p = argparse.ArgumentParser()
    p.add_argument("steps", nargs="+")
    p.add_argument("--gen", nargs="+", required=True)
    p.add_argument("--run-config", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    lib, info = tier1b.build(a.steps, a.gen, runconfig.get(a.run_config).geometry(), roots=True)
    tier1b.write_library(a.out, lib)
    print(json.dumps({k: info[k] for k in ("crossings", "desc_steps", "desc_steps_unattached", "showers")}))


def main():
    if sys.argv[1] == "build":
        return build(sys.argv[2:])
    a, b = digests(sys.argv[1]), digests(sys.argv[2])
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    print(json.dumps({"columns": len(a), "differ": diff, "identical": not diff}))
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
