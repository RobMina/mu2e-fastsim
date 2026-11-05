"""Summarize saved primary-only birth diagnostics without calling them photon exit flux."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import cechain as C
from . import flash_regions as R


def summarize(data, geom):
    valid = np.asarray(data["photon_valid"], bool)
    energy = np.asarray(data["photon_energy"])
    xyz = np.asarray(data["photon_position"])
    direction = np.asarray(data["photon_direction"])
    mechanism = np.asarray(data["photon_mechanism"])
    n, capacity = valid.shape
    if (energy.shape != valid.shape or mechanism.shape != valid.shape or xyz.shape != (n, capacity, 3)
            or direction.shape != xyz.shape or len(data["source_pdg"]) != n):
        raise ValueError("Malformed birth-record arrays")
    if (np.any(~np.isfinite(energy[valid])) or np.any(energy[valid] <= 0)
            or np.any(~np.isfinite(xyz[valid])) or np.any(~np.isfinite(direction[valid]))):
        raise ValueError("Nonfinite or nonpositive valid photon birth")
    if np.any(~np.isin(mechanism[valid], (1, 2, 3))):
        raise ValueError("Unknown photon mechanism")
    if np.any(np.abs(np.linalg.norm(direction[valid], axis=1) - 1) > 1e-4):
        raise ValueError("Nonunit valid photon direction")
    count = np.asarray(data["count"])
    overflow = np.asarray(data["overflow"])
    if np.any(count != valid.sum(axis=1) + overflow):
        raise ValueError("Birth count and retained record accounting disagree")
    if np.any(count != data["state_nph"] + data["state_nan"]):
        raise ValueError("Primary photon counters do not account for emitted births")
    total_energy = np.asarray(data["state_eph"]) + data["state_ean"]
    closed = np.isclose(energy.sum(axis=1), total_energy, rtol=5e-5, atol=1e-8)
    if np.any(~closed & (overflow == 0)):
        raise ValueError("Complete parent birth energy does not match primary counters")
    region = R.classify(geom, xyz[valid], tolerance_mm=1e-3)
    pdg = np.broadcast_to(np.asarray(data["source_pdg"])[:, None], valid.shape)[valid]
    groups, inverse = np.unique(np.rec.fromarrays([region, pdg, mechanism[valid]], names="region,pdg,mechanism"), return_inverse=True)
    unique_regions = np.unique(region)
    return dict(parents=n, retained_photons=int(valid.sum()), emitted_photons=int(count.sum()),
                overflow_births=int(overflow.sum()), capped_parents=int(np.asarray(data["capped"]).sum()),
                loopers=int(np.asarray(data["loopers"]).sum()),
                retained_energy_mev=float(energy.sum()),
                primary_photon_energy_mev=float(total_energy.sum()),
                region_counts={str(k): int((region == k).sum()) for k in unique_regions},
                region_energy_mev={str(k): float(energy[valid][region == k].sum()) for k in unique_regions},
                grouped=dict(region=groups["region"].tolist(), pdg=groups["pdg"].astype(int).tolist(),
                             mechanism=groups["mechanism"].astype(int).tolist(),
                             count=np.bincount(inverse, minlength=len(groups)).tolist(),
                             energy_mev=np.bincount(inverse, weights=energy[valid], minlength=len(groups)).tolist()),
                birth_energy_quantiles_mev=np.quantile(energy[valid], [0, .05, .5, .95, 1]).tolist() if valid.any() else None,
                geometry_warning="Other/ambiguous births expose represented-volume escape/overlap issues, not a material truth certificate",
                scope="Primary-only births, not exit flux; no generated-event normalization or photon-response validity")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--births", type=Path, required=True)
    parser.add_argument("--geom", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Refusing to overwrite a birth summary")
    with np.load(args.births, allow_pickle=False) as f:
        result = summarize({k: f[k] for k in f.files}, C.from_offline(str(args.geom), str(args.offline_root)))
    digest = hashlib.sha256()
    with args.births.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    args.out.write_text(json.dumps(dict(schema="flash-birth-summary-v1", sha256=digest.hexdigest(), **result), indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
