"""Inspect saved birth-source transport caps without rerunning transport or changing fates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import cechain as C
from . import flash_regions as R


def _columns(data, rows):
    names = ("source_run", "source_subrun", "source_event", "source_particle_key", "source_pdg",
             "state_x", "state_u", "state_T", "state_s", "state_t", "state_cnt", "state_nvac",
             "state_nslab", "state_nstall", "state_nadapt", "state_nph", "state_nan", "count")
    return {name: np.asarray(data[name])[rows].tolist() for name in names}


def summarize(data, geom):
    """Return columnar capped-parent and nonfoil-birth diagnostics with source identities."""
    n = len(data["source_pdg"])
    required = ("source_run", "source_subrun", "source_event", "source_particle_key", "source_pdg",
                "state_x", "state_u", "state_T", "state_s", "state_t", "state_cnt", "state_nvac",
                "state_nslab", "state_nstall", "state_nadapt", "state_nph", "state_nan", "count",
                "capped", "loopers", "overflow", "photon_valid", "photon_position", "photon_energy",
                "photon_mechanism")
    if any(name not in data or len(data[name]) != n for name in required):
        raise ValueError("Incomplete or misaligned saved transport schema")
    x = np.asarray(data["state_x"])
    u = np.asarray(data["state_u"])
    if x.shape != (n, 3) or u.shape != (n, 3):
        raise ValueError("Malformed final position/direction")
    valid = np.asarray(data["photon_valid"], bool)
    xyz = np.asarray(data["photon_position"])
    if valid.ndim != 2 or xyz.shape != valid.shape + (3,):
        raise ValueError("Malformed saved birth positions")
    if any(not np.all(np.isfinite(np.asarray(data[name]))) for name in required):
        raise ValueError("Nonfinite saved transport values")
    capped = np.asarray(data["capped"], bool)
    rows = np.flatnonzero(capped)
    b = geom.beam
    distances = np.minimum(np.abs(x[rows, 2, None] - (b.foil_z - b.foil_ht)),
                           np.abs(x[rows, 2, None] - (b.foil_z + b.foil_ht)))
    nearest = np.argmin(distances, axis=1)
    radial = C.radial(geom, x[rows])
    capped_columns = dict(parent_index=rows.tolist(), **_columns(data, rows),
                          represented_region=R.classify(geom, x[rows]).tolist(),
                          nearest_foil_index=nearest.tolist(),
                          nearest_foil_face_distance_mm=distances[np.arange(len(rows)), nearest].tolist(),
                          radius_mm=radial.tolist(),
                          nearest_foil_rin_mm=b.foil_rin[nearest].tolist(),
                          nearest_foil_rout_mm=b.foil_rout[nearest].tolist())
    parents, slots = np.nonzero(valid)
    regions = R.classify(geom, xyz[valid])
    other = regions != "foil"
    pr, sl = parents[other], slots[other]
    quantiles = (0, .05, .5, .95, 1)
    fields = ("state_T", "state_s", "state_t", "state_nvac", "state_nslab", "state_nstall", "state_nadapt")
    return dict(parents=n, capped_parents=int(capped.sum()),
                loopers=int(np.asarray(data["loopers"]).sum()),
                overflow_births=int(np.asarray(data["overflow"]).sum()),
                quantile_probabilities=list(quantiles),
                all_parent_quantiles={name: np.quantile(data[name], quantiles).tolist() for name in fields},
                capped_parent_quantiles={name: np.quantile(np.asarray(data[name])[rows], quantiles).tolist()
                                         if rows.size else None for name in fields},
                capped_columns=capped_columns,
                nonfoil_births=dict(parent_index=pr.tolist(), slot=sl.tolist(), region=regions[other].tolist(),
                                    position_mm=xyz[pr, sl].tolist(),
                                    energy_mev=np.asarray(data["photon_energy"])[pr, sl].tolist(),
                                    mechanism=np.asarray(data["photon_mechanism"])[pr, sl].tolist(),
                                    **_columns(data, pr)),
                scope="Saved numerical-state diagnostics only; no cap repair, new transport, exit flux or physics validation")


def compare(data, reference):
    """Compare every saved column byte-for-byte, including identities, counters and births."""
    common = sorted(set(data) & set(reference))
    missing = sorted(set(reference) - set(data))
    extra = sorted(set(data) - set(reference))
    different = []
    for name in common:
        current, old = np.asarray(data[name]), np.asarray(reference[name])
        if current.dtype != old.dtype or current.shape != old.shape or current.tobytes() != old.tobytes():
            different.append(name)
    return dict(exact=not (missing or extra or different), compared_columns=len(common),
                missing_columns=missing, extra_columns=extra, different_columns=different,
                scope="Exact saved-array regression, not physics validation or generated-event pairing")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--births", type=Path, required=True)
    parser.add_argument("--geom", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Refusing to overwrite a transport summary")
    with np.load(args.births, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    report = summarize(data, C.from_offline(str(args.geom), str(args.offline_root)))
    if args.reference is not None:
        with np.load(args.reference, allow_pickle=False) as source:
            report["reference_comparison"] = compare(data, {name: source[name] for name in source.files})
        report["reference_comparison"]["sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    digest = hashlib.sha256(args.births.read_bytes()).hexdigest()
    args.out.write_text(json.dumps(dict(schema="flash-transport-summary-v1", sha256=digest, **report),
                                   indent=2, allow_nan=False) + "\n")
    return 0 if report.get("reference_comparison", {}).get("exact", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
