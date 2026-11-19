"""Opt-in raw CRY showers from the standalone C ABI; no Mu2e projection or shower selection here.

The binding lives in the experiment-neutral package cryjax (cryjax.native, extracted from this module on
2026-11-18). This module keeps the import path and the defaults of the fastsim CRY image: tables under $CRYDATAPATH
or /opt/cry/data, shim $MU2E_CRY_LIBRARY or /opt/cry/lib/libmu2ecry.so.
"""
import os

from cryjax import native as _native
from cryjax.native import ABI_VERSION, PARTICLE_DTYPE, CryBatch, _Particle  # noqa: F401

DATA = "/opt/cry/data"
LIBRARY = "/opt/cry/lib/libmu2ecry.so"


class CrySource(_native.CrySource):
    """cryjax.native.CrySource with the Mu2e image defaults for the table directory and the shim."""

    def __init__(self, *, data=None, library=None, **kwargs):
        super().__init__(data=data or os.environ.get("CRYDATAPATH", DATA),
                         library=library or os.environ.get("MU2E_CRY_LIBRARY", LIBRARY), **kwargs)


def main(argv=None):
    import argparse
    import json
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    with CrySource(seed=args.seed) as source:
        batch = source.next_showers(args.showers)
    ids, counts = np.unique(batch.particles["pdg"], return_counts=True)
    print(json.dumps({"raw_showers": args.showers, "particles": len(batch.particles),
                      "live_seconds": batch.live_seconds,
                      "pdg_counts": dict(zip(map(str, ids), map(int, counts))),
                      "mu2e_projection": False, "stock_validation": False}, indent=2))


if __name__ == "__main__":
    main()
