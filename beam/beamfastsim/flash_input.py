"""Inspect saved beam-boundary states, with no charged-primary or shower transport."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot

from .flash_ancestry import _ids, _join, _read_tree, _take, EVENT, PARTICLE

VD = EVENT + ("vdid", "sp", "x", "y", "z", "px", "py", "pz", "t")


def summarize(states, particles):
    """Join only within the same event; a beam-tag state is not a TS3/DS boundary certificate."""
    found = _join(_ids(particles, EVENT + ("key",)), _ids(dict(states, key=states["sp"]), EVENT + ("key",)))
    pdg = _take(particles["pdg"], found, 0)
    group, inverse = np.unique(np.column_stack((states["vdid"], pdg, found >= 0)), axis=0, return_inverse=True)
    columns = ("x", "y", "z", "px", "py", "pz", "t")
    if any(np.any(~np.isfinite(states[c])) for c in columns):
        raise ValueError("Nonfinite boundary state")
    momentum = np.sqrt(sum(np.asarray(states[c], float)**2 for c in ("px", "py", "pz")))
    return dict(state_rows=len(found), represented_events=len(np.unique(_ids(states, EVENT))),
                groups=dict(vdid=group[:, 0].tolist(), pdg=group[:, 1].tolist(), particle_found=group[:, 2].astype(bool).tolist(),
                            count=np.bincount(inverse, minlength=len(group)).tolist()),
                coordinate_ranges={c: [float(np.min(states[c])), float(np.max(states[c]))] if len(found) else None for c in columns},
                momentum_quantiles_mev=np.quantile(momentum, [0, .05, .5, .95, 1]).tolist() if len(found) else None,
                electron_or_positron_rows=int(np.isin(pdg, [11, -11]).sum()),
                photon_rows=int((pdg == 22).sum()), unresolved_particle_rows=int((found < 0).sum()),
                scope="Saved retained states only; neither generated-event normalization nor a particular boundary is certified")


def stock_summary(path):
    """Read the stock ReadVirtualDetector ntvd table, retaining its float-ID limitation."""
    with uproot.open(path) as root:
        tree = root["input/ntvd"]
        a = tree.arrays(["run", "subrun", "evt", "trk", "sid", "pdg", "x", "y", "z", "px", "py", "pz", "time"], library="np")
        states = {name: a[source] for name, source in zip(VD, ("run", "subrun", "evt", "sid", "trk", "x", "y", "z", "px", "py", "pz", "time"))}
        particles = {name: a[name] for name in ("run", "subrun", "pdg")}
        particles.update(event=a["evt"], key=a["trk"])
        identity = _ids(particles, EVENT + ("key",))
        _, first = np.unique(identity, return_index=True)
        particles = {name: value[first] for name, value in particles.items()}
        if any(np.any(np.abs(a[name]) >= 2**24) for name in ("run", "subrun", "evt", "trk")):
            raise ValueError("Float ntvd identity exceeds exact integer range")
        return dict(tree="input/ntvd", entries=int(tree.num_entries), summary=summarize(states, particles),
                    identity_limit="Stock ntvd stores float IDs; exact only below 2**24")


def schema_summary(path):
    """List persisted tree branches without reading or interpreting any source-state values."""
    with uproot.open(path) as root:
        return {name: dict(entries=int(obj.num_entries), branches=obj.typenames())
                for name, obj in root.items(recursive=True) if hasattr(obj, "num_entries")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stock", action="store_true", help="Stock ReadVirtualDetector input/ntvd diagnostic schema")
    mode.add_argument("--schema-only", action="store_true", help="Persisted branch schema only, no state interpretation")
    args = parser.parse_args(argv)
    if args.schema_only:
        report = dict(schema="flash-root-branch-inventory-v1", trees=schema_summary(args.file))
    elif args.stock:
        report = dict(schema="flash-stock-input-states-v1", **stock_summary(args.file))
    else:
        with uproot.open(args.file) as root:
            states, smeta = _read_tree(root, "vdsteps", VD)
            particles, pmeta = _read_tree(root, "particles", PARTICLE)
        report = dict(schema="flash-input-states-v1", trees=dict(states=smeta, particles=pmeta), summary=summarize(states, particles))
    digest = hashlib.sha256()
    with args.file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    report["sha256"] = digest.hexdigest()
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
