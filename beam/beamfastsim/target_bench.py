"""Benchmark explicit target electrons in the resident device CE loop; no validity claim."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source")
    src.add_argument("--ref")
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_common.txt")
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--lanes", type=int, default=4096)
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    ap.add_argument("--min-bucket", type=int, default=1024)
    ap.add_argument("--shrink", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--unroll", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.repeat < 1 or a.lanes < 1 or a.min_bucket < 1 or a.shrink < 2:
        ap.error("repeat/lanes/min-bucket must be positive and shrink >= 2")
    t0 = time.perf_counter()
    import jax
    from . import cechain as C
    from . import cechain_jax as CJ
    geom = C.from_offline(a.geom, a.offline_root)
    a.chunk, a.nchunks = 0, 1
    x, u, p, t, idx = C._starts(a, np.random.default_rng(a.seed))
    maps = CJ.device_maps(a.maps, np.dtype(a.dtype))
    setup = time.perf_counter() - t0
    timings = []
    first = None
    for rep in range(a.repeat + 1):
        t1 = time.perf_counter()
        result = CJ.transport(jax.random.PRNGKey(a.seed), geom, maps, x, u, p, t,
                              dtype=np.dtype(a.dtype), lanes=a.lanes, unroll=a.unroll,
                              min_bucket=a.min_bucket, shrink=a.shrink)
        timing = time.perf_counter() - t1
        timings.append(timing)
        if first is None:
            first = result
        elif not all(np.array_equal(v, w) for v, w in zip(first, result) if v is not None):
            raise ValueError("repeat with identical starts and key changed the transport output")
    fields = {k: v for k, v in first._asdict().items() if v is not None}
    np.savez_compressed(a.out + ".npz", idx=idx, x0=x, u0=u, p0=p, t0=t, wall=timings[0], **fields)
    warm = np.array(timings[1:])
    report = dict(n=int(p.size), device=str(jax.devices()[0]), device_kind=jax.devices()[0].device_kind,
                  dtype=a.dtype, lanes=a.lanes, unroll=a.unroll,
                  min_bucket=a.min_bucket, shrink=a.shrink,
                  setup_s=setup, first_transport_s=timings[0], first_with_setup_s=setup + timings[0],
                  first_with_setup_e_per_s=p.size / (setup + timings[0]), warm_transport_s=warm.tolist(),
                  warm_e_per_s=(p.size / warm).tolist(), repeat_identical=True,
                  physics_validated=False, scope="primary electrons to VD; maps/inputs setup separate; no reco or photon conversion",
                  fates={name: int((first.fate == i).sum()) for i, name in enumerate(C.FATES)})
    Path(a.out + ".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
