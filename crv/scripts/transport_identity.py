"""Default identity of the fast transport: two sets of fast tables (same inputs, seeds, file names) hold the same rows.

Compares the steps and particles trees of every file present in both directories exactly (no tolerance); exit 1 on any
difference or on a file missing from either side.

    python3 scripts/transport_identity.py DIR_A DIR_B [--glob 'fastC.*.root']
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from crvfastsim import tables


def same(a, b, tree):
    x, y = tables.read(str(a), tree), tables.read(str(b), tree)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return False, f"{tree}: shape/columns {x.shape} vs {y.shape}"
    bad = [c for c in x.columns
           if not np.array_equal(x[c].to_numpy(), y[c].to_numpy(), equal_nan=x[c].dtype.kind == "f")]
    return not bad, f"{tree}: {len(x)} rows" + (f", differ in {bad}" if bad else " identical")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.add_argument("--glob", default="fastC.*.root")
    a = p.parse_args(argv)
    fa = {f.name: f for f in Path(a.dir_a).rglob(a.glob)}
    fb = {f.name: f for f in Path(a.dir_b).rglob(a.glob)}
    ok = bool(fa) and fa.keys() == fb.keys()
    if fa.keys() != fb.keys():
        print(f"file sets differ: only A {sorted(fa.keys() - fb.keys())}, only B {sorted(fb.keys() - fa.keys())}")
    for name in sorted(fa.keys() & fb.keys()):
        for tree in ("steps", "particles"):
            r, msg = same(fa[name], fb[name], tree)
            ok &= r
            print(f"{name} {msg}")
    print("IDENTICAL" if ok else "DIFFERENT", f"({len(fa.keys() & fb.keys())} files)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
