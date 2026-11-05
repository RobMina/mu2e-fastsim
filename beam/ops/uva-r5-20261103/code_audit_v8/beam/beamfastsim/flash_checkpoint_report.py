"""Audit saved checkpoint lineage and retained births without advancing transport."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import flash_checkpoint as C


FATES = ("live", "plane", "stopped", "upstream", "wall", "absorbed", "looper")
IDENTITY = ("schema_version", "capacity", "parent_index", "pdg", "t0", "species_codes")
FROZEN = ("x", "u", "T", "t", "s", "cnt", "nph", "nan", "eph", "ean", "nde", "ede", "cont")
COUNTERS = ("nvac", "nslab", "nstall", "nadapt", "cnt", "nph", "nan", "nde", "count")


def _equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def summarize(paths):
    """Check a complete initial-to-child chain; manifests supply, not certify, binding."""
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("At least one checkpoint is required")
    binding = json.loads(paths[0].read_text()).get("binding")
    if not isinstance(binding, dict):
        raise ValueError("Missing checkpoint binding")
    previous, previous_manifest = None, None
    receipts = []
    initial_fates = None
    for path in paths:
        data, manifest = C.load(path, binding)
        parent = None if previous_manifest is None else previous_manifest["sha256"]
        if manifest.get("parent_sha256") != parent:
            raise ValueError("Checkpoint parent lineage mismatch")
        if previous is not None and (set(data) != set(previous) or any(
                data[name].dtype != old.dtype or data[name].shape != old.shape for name, old in previous.items())):
            raise ValueError("Checkpoint chain schema mismatch")
        if any(name not in data for name in IDENTITY + ("iterations", "total_iterations")):
            raise ValueError("Incomplete source checkpoint schema")
        if previous is not None and any(not _equal(data[name], previous[name]) for name in IDENTITY):
            raise ValueError("Checkpoint source identity changed")
        n = len(data["parent_index"])
        fates = np.full(n, -1, np.int32)
        counts = np.zeros(n, np.int64)
        recorded = 0
        energy = 0.0
        for slot, code in enumerate((11, -11)):
            prefix = f"species/{code}/"
            if code not in data["species_codes"]:
                if data["total_iterations"][slot] != 0 or data["iterations"][slot] != 0:
                    raise ValueError("Absent species has nonzero iterations")
                continue
            ids, key = data[prefix + "ids"], data[prefix + "key"]
            root = prefix + "state/"
            get = lambda name: data[root + name]
            fate, count, valid = get("fate"), get("count"), get("records/valid")
            total, last = data["total_iterations"][slot], data["iterations"][slot]
            if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= n) or
                    len(np.unique(ids)) != len(ids) or np.any(fates[ids] != -1) or
                    fate.shape != ids.shape or np.any(fate < 0) or np.any(fate >= len(FATES)) or
                    np.any(data["pdg"][ids] != code)):
                raise ValueError("Invalid species row/fate identity")
            if total < 0 or last < 0 or last > total or int(get("slab/it")) != total:
                raise ValueError("Invalid species iteration accounting")
            if (count.shape != ids.shape or valid.shape != (len(ids), int(data["capacity"])) or
                    np.any(count < 0) or not np.array_equal(valid, np.arange(valid.shape[1])[None, :] < count[:, None]) or
                    np.any(count != get("nph").astype(np.int64) + get("nan"))):
                raise ValueError("Invalid retained birth accounting")
            if previous is not None:
                old = lambda name: previous[root + name]
                if not _equal(ids, previous[prefix + "ids"]) or not _equal(key, previous[prefix + "key"]):
                    raise ValueError("Species rows or RNG key changed")
                if total != previous["total_iterations"][slot] + last:
                    raise ValueError("Cumulative iteration increment mismatch")
                terminal = old("fate") != 0
                if np.any(fate[terminal] != old("fate")[terminal]) or any(
                        not _equal(get(name)[terminal], old(name)[terminal]) for name in FROZEN):
                    raise ValueError("Terminal parent changed during continuation")
                if any(np.any(get(name) < old(name)) for name in COUNTERS):
                    raise ValueError("Cumulative parent counter decreased")
                old_valid = old("records/valid")
                if np.any(old_valid & ~valid) or any(not _equal(get("records/" + name)[old_valid],
                        old("records/" + name)[old_valid]) for name in
                        ("energy", "position", "direction", "path", "time", "mechanism")):
                    raise ValueError("Previously retained photon changed")
            fates[ids] = fate
            counts[ids] = count
            recorded += int(valid.sum())
            energy += float(get("records/energy")[valid].sum())
        if np.any(fates < 0):
            raise ValueError("Missing parent rows")
        if initial_fates is None:
            initial_fates = fates.copy()
        receipts.append(dict(manifest=path.name, sha256=manifest["sha256"], parent_sha256=parent,
                             total_species_iterations=data["total_iterations"].tolist(),
                             fate_counts={name: int((fates == k).sum()) for k, name in enumerate(FATES)},
                             emitted_photons=int(counts.sum()), retained_photons=recorded,
                             overflow_births=int(counts.sum()) - recorded, retained_energy_mev=energy))
        previous, previous_manifest = data, manifest
    formerly_live = initial_fates == 0
    return dict(schema="flash-checkpoint-chain-report-v1", parents=n, checkpoints=receipts,
                initial_live_parents=int(formerly_live.sum()),
                initially_live_final_fates={name: int((formerly_live & (fates == k)).sum())
                                           for k, name in enumerate(FATES)},
                scope="Internal saved-chain integrity only; binding is manifest-supplied, not independently certified; no transport, exit flux or physics validation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.checkpoints)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
