"""Run an opt-in charged-primary birth diagnostic on identity-checked source states.

No photons are propagated and no source-normalized objective is inferred. This
entry point preserves the original absolute time separately from relative device
times and retains overflow/capped outputs even when the completeness check fails.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(args, source_hashes, geometry_hashes, maps, n):
    import jax
    import jaxlib
    import scipy
    from mu2efastsim import eslab_jax as SJ
    from . import flash_checkpoint as CP
    root = Path(__file__).parent
    core = Path(SJ.__file__).parent
    beam_names = ("cechain", "cechain_jax", "geometry", "simpleconfig", "fieldmap", "flash_target_jax",
                  "flash_target_run", "flash_checkpoint")
    code = {"beam/" + name + ".py": CP.sha256(root / (name + ".py")) for name in beam_names}
    code.update({"core/" + path.relative_to(core).as_posix(): CP.sha256(path)
                 for path in sorted(core.rglob("*.py")) if "__pycache__" not in path.parts})
    assets = {path.name: CP.sha256(path) for path in sorted((core / "data").glob("*.npz"))}
    return dict(schema="flash-target-binding-v1", source_hashes=source_hashes,
                geometry=[dict(name=Path(item["path"]).name, sha256=item["sha256"]) for item in geometry_hashes],
                maps=[dict(sha256=item["sha256"], binary_sha256=item["binary_sha256"]) for item in maps],
                settings=dict(seed=args.seed, dtype=args.dtype, capacity=args.capacity, h_mm=args.h,
                              path_cap_mm=args.s_max, parents=n, msc=True, layer_field=True, shell_curvature=True),
                code=code, assets=assets,
                runtime=dict(python=sys.version, machine=platform.machine(), numpy=np.__version__,
                             scipy=scipy.__version__, jax=jax.__version__, jaxlib=jaxlib.__version__,
                             x64_enabled=jax.config.x64_enabled, prng_impl=str(jax.config.jax_default_prng_impl),
                             backend=jax.default_backend(), devices=[device.device_kind for device in jax.devices()]))


def run(args):
    import jax
    from . import cechain as C
    from . import cechain_jax as CJ
    from . import flash_target_jax as F
    from . import flash_checkpoint as CP
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
    includes = []
    simpleconfig.read(str(args.geom), str(args.offline_root), source_files=includes)
    geometry_hashes = [dict(path=str(path), sha256=_sha256(path)) for path in sorted(set(includes))]
    map_hashes = [dict(header=str(path), sha256=_sha256(path),
                       binary_sha256=_sha256(Path(path).with_suffix(".bin"))) for path in args.maps]
    source_hashes = dict(states=_sha256(args.states), certificate=_sha256(args.certificate))
    checkpoint = getattr(args, "checkpoint", None)
    resume = getattr(args, "resume", None)
    chunk_steps = getattr(args, "chunk_steps", None)
    if chunk_steps is not None and (chunk_steps < 1 or checkpoint is None):
        raise ValueError("chunk_steps must be positive and requires a checkpoint output")
    if resume is not None and checkpoint is None:
        raise ValueError("resuming requires a new checkpoint output")
    if args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    binding = _binding(args, source_hashes, geometry_hashes, map_hashes, n) if checkpoint else None
    started = time.perf_counter()
    session = F.initialize_source(jax.random.PRNGKey(args.seed), geom, maps, source["x"], source["p"] / p[:, None],
                                  p, source["time_ns"], source["pdg"], capacity=args.capacity,
                                  h=args.h, s_max=args.s_max, dtype=np.dtype(args.dtype))
    parent_hash = None
    if resume is not None:
        arrays, receipt = CP.load(resume, binding)
        session = F.import_state(session, arrays)
        parent_hash = receipt["sha256"]
    remaining = args.max_steps
    chunk_number = 0
    while remaining > 0:
        amount = min(remaining, args.max_steps if chunk_steps is None else chunk_steps)
        session = F.advance_source(session, amount)
        result = jax.tree_util.tree_map(lambda a: np.asarray(a), F.source_result(session))
        remaining -= amount
        chunk_number += 1
        if checkpoint is not None:
            path = checkpoint if chunk_steps is None else checkpoint.with_name(
                checkpoint.stem + f".chunk{chunk_number:04d}.json")
            receipt = CP.save(path, F.export_state(session), binding, parent_sha256=parent_hash)
            parent_hash = receipt["sha256"]
            print(f"checkpoint={path} capped={int(result['capped'].sum())} "
                  f"loopers={int(result['loopers'].sum())}", flush=True)
        if not np.any(result["capped"]) or np.any(result["overflow"]):
            break
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
    report = dict(schema="flash-target-birth-pilot-v1", complete=complete, parents=n,
                  source_hashes=source_hashes, geometry_hashes=geometry_hashes, maps=map_hashes,
                  seed=args.seed, dtype=args.dtype, backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                  capacity=args.capacity, max_steps=args.max_steps, h_mm=args.h, path_cap_mm=args.s_max,
                  fate_counts={name: int((result["state"]["fate"] == k).sum()) for k, name in enumerate(C.FATES)},
                  recorded_photons=int(valid.sum()), emitted_photons=int(result["count"].sum()),
                  mechanism_counts={name: int((valid & (code == k)).sum()) for k, name in enumerate(names, 1)},
                  recorded_energy_mev=float(result["records"]["energy"].sum()),
                  dropped_births=int(result["overflow"].sum()), capped_parents=int(result["capped"].sum()),
                  loopers=int(result["loopers"].sum()), delta_rays_not_transported=int(result["ndelta"].sum()),
                  cold_seconds=wall, scope="Primary-only photon births; no attenuation, photon transport, delta cascades, exit flux or generated normalization")
    report["transport_complete"] = not bool(np.any(result["capped"]) or np.any(result["overflow"]))
    report["completion_scope"] = "No live/capped or dropped births; path-loopers remain excluded by validate_complete"
    if checkpoint is not None:
        report["checkpoint"] = dict(manifest=str(path), sha256=parent_hash,
                                    resumed_from=None if resume is None else str(resume),
                                    chunk_steps=chunk_steps, chunks=chunk_number,
                                    total_species_iterations=np.asarray(session["total_iterations"]).tolist(),
                                    scope="Full raw resumable state; capped/looper fates are not certified complete")
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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--chunk-steps", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--s-max", type=float, default=30000.0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.with_suffix(".npz").exists():
        parser.error("refusing to overwrite a birth diagnostic")
    if args.resume is not None and args.checkpoint is None:
        parser.error("--resume requires a new --checkpoint output")
    if args.chunk_steps is not None and (args.chunk_steps < 1 or args.checkpoint is None):
        parser.error("positive --chunk-steps requires --checkpoint")
    if args.checkpoint is not None:
        if args.checkpoint.suffix != ".json":
            parser.error("--checkpoint must name a JSON manifest")
        paths = [args.checkpoint] if args.chunk_steps is None else [args.checkpoint.with_name(
            args.checkpoint.stem + f".chunk{i:04d}.json")
            for i in range(1, (args.max_steps + args.chunk_steps - 1) // args.chunk_steps + 1)]
        if any(p.exists() or p.with_suffix(".npz").exists() or p == args.out or
               p.with_suffix(".npz") == args.out.with_suffix(".npz") for p in paths):
            parser.error("checkpoint collides with an existing artifact or diagnostic output")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
