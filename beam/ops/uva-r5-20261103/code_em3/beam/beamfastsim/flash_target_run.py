"""Run an opt-in charged-primary birth diagnostic on identity-checked source states.

No photons are propagated and no source-normalized objective is inferred. This
entry point preserves the original absolute time separately from relative device
times and retains overflow/capped outputs even when the completeness check fails.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    import jax
    from . import cechain as C
    from . import cechain_jax as CJ
    from . import flash_target_jax as F
    from . import simpleconfig
    certificate = json.loads(args.certificate.read_text())
    if not certificate.get("certified", False) or certificate.get("schema") != "flash-step-input-v1":
        raise ValueError("An independently matched input certificate is required")
    if certificate.get("artifacts", {}).get("states", {}).get("sha256") != _sha256(args.states):
        raise ValueError("Source state artifact does not match its cross-check certificate")
    with np.load(args.states, allow_pickle=False) as source:
        source = {k: source[k] for k in source.files}
    required = ("run", "subrun", "event", "particle_key", "pdg", "x", "p", "time_ns")
    if any(k not in source for k in required):
        raise ValueError("Source state schema is incomplete")
    n = len(source["pdg"])
    if n != certificate["rows"] or any(len(source[k]) != n for k in required):
        raise ValueError("Source row counts do not match the input certificate")
    if args.parents is not None:
        if args.parents < 1 or args.parents > n:
            raise ValueError("parents must be within the retained input size")
        n = args.parents
    source = {k: v[:n] for k, v in source.items()}
    geom = C.from_offline(str(args.geom), str(args.offline_root))
    maps = CJ.device_maps([str(p) for p in args.maps], dtype=np.dtype(args.dtype))
    p = np.linalg.norm(source["p"], axis=1)
    if np.any(p <= 0):
        raise ValueError("Source momentum must be positive")
    started = time.perf_counter()
    result = F.target_source(jax.random.PRNGKey(args.seed), geom, maps, source["x"], source["p"] / p[:, None],
                             p, source["time_ns"], source["pdg"], capacity=args.capacity,
                             max_steps=args.max_steps, h=args.h, s_max=args.s_max, dtype=np.dtype(args.dtype))
    result = jax.tree_util.tree_map(lambda a: np.asarray(a), result)
    wall = time.perf_counter() - started
    complete = True
    try:
        F.validate_complete(result)
    except ValueError:
        complete = False
    payload = {"source_" + k: v for k, v in source.items()}
    payload.update({"state_" + k: v for k, v in result["state"].items()})
    payload.update({"photon_" + k: v for k, v in result["records"].items()})
    payload.update({k: result[k] for k in ("count", "overflow", "capped", "loopers", "parent_index", "pdg", "t0")})
    np.savez_compressed(args.out.with_suffix(".npz"), **payload)
    valid = result["records"]["valid"]
    code = result["records"]["mechanism"]
    names = ("bremsstrahlung", "annihilation_in_flight", "annihilation_at_rest")
    includes = []
    simpleconfig.read(str(args.geom), str(args.offline_root), source_files=includes)
    report = dict(schema="flash-target-birth-pilot-v1", complete=complete, parents=n,
                  source_hashes=dict(states=_sha256(args.states), certificate=_sha256(args.certificate)),
                  geometry_hashes=[dict(path=str(p), sha256=_sha256(p)) for p in sorted(set(includes))],
                  maps=[dict(header=str(p), sha256=_sha256(p), binary_sha256=_sha256(Path(p).with_suffix(".bin"))) for p in args.maps],
                  seed=args.seed, dtype=args.dtype, backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                  capacity=args.capacity, max_steps=args.max_steps, h_mm=args.h, path_cap_mm=args.s_max,
                  fate_counts={name: int((result["state"]["fate"] == k).sum()) for k, name in enumerate(C.FATES)},
                  recorded_photons=int(valid.sum()), emitted_photons=int(result["count"].sum()),
                  mechanism_counts={name: int((valid & (code == k)).sum()) for k, name in enumerate(names, 1)},
                  recorded_energy_mev=float(result["records"]["energy"].sum()),
                  dropped_births=int(result["overflow"].sum()), capped_parents=int(result["capped"].sum()),
                  loopers=int(result["loopers"].sum()), delta_rays_not_transported=int(result["ndelta"].sum()),
                  cold_seconds=wall, scope="Primary-only photon births; no attenuation, photon transport, delta cascades, exit flux or generated normalization")
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if complete else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--geom", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--maps", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parents", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--s-max", type=float, default=30000.0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.with_suffix(".npz").exists():
        parser.error("refusing to overwrite a birth diagnostic")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
