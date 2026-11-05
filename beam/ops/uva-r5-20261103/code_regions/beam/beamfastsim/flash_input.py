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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    with uproot.open(args.file) as root:
        states, smeta = _read_tree(root, "vdsteps", VD)
        particles, pmeta = _read_tree(root, "particles", PARTICLE)
    digest = hashlib.sha256()
    with args.file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    report = dict(schema="flash-input-states-v1", sha256=digest.hexdigest(),
                  trees=dict(states=smeta, particles=pmeta), summary=summarize(states, particles))
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
