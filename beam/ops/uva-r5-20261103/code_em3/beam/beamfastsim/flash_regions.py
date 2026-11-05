"""Geometric diagnostic of nearest retained photon births feeding tracker steps.

This is truth-level geometric attribution, NOT a Geant4 volume/material tag or a
transported-boundary-species measurement. Strict retained photon ancestry can be
incomplete after compression. No upstream material is inferred from coordinates.
Only the CLI loads CEGeometry (and its transitive physics/JAX dependencies).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REGIONS = ("foil", "ipa", "ring", "opa", "other", "ambiguous", "missing_birth", "no_photon")
MODEL = {
    "coordinates": "Dumped Mu2e frame, mm; r=hypot(sx-beam.axis_x,sy); axis_y=0",
    "membership": "Closed surfaces expanded by tolerance_mm; multiple volume matches are ambiguous, with no precedence",
    "foil": "Exact union of represented annular slabs, each with its own z, half thickness and radii; no nearest-plane shortcut",
    "ipa": "Uniform cylindrical shell from ipa_r to ipa_r+ipa_t over [ipa_z0,ipa_z1]",
    "ring": "Uniform inward shells [ipa_r-ring_t,ipa_r], at [end,end+2*ring_hl] for BOTH IPA ends, as CEGeometry",
    "opa": "Linear conical inner radius, uniform radial (not surface-normal) thickness",
    "limitations": "Ideal coaxial volumes only: no foil tilts, wires, OPA slots/supports or corrugation detail; not an Offline solid/material tag. Other includes upstream and unmodelled volumes; no upstream assignment is guessed.",
    "ancestry": "Nearest strict retained photon ancestor, not necessarily the original photon; root genpdg only when retained root joins dump/gen in the same run/subrun/event; -1 denotes unjoined",
    "energy": "Tracker-step ionizing deposit eion in MeV, not photon energy; counts are step rows, not independent photons/events",
}


def classify(geom, xyz, tolerance_mm=1e-3):
    """Return exclusive string labels for an (N,3) array; NumPy only.

    Nonfinite coordinates are missing_birth, distinct from finite other. The
    caller separately labels absent retained photons no_photon. Overlapping
    foils also count as ambiguous. Zero-thickness shells/rings are absent.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if not np.isscalar(tolerance_mm) or not np.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError("tolerance_mm must be finite and nonnegative")
    b, tol = geom.beam, float(tolerance_mm)
    foils = [np.asarray(getattr(b, name), dtype=float) for name in ("foil_z", "foil_rin", "foil_rout", "foil_ht")]
    zf, rin, rout, ht = foils
    if any(a.ndim != 1 or a.shape != zf.shape or not np.all(np.isfinite(a)) for a in foils):
        raise ValueError("foil arrays must be finite, 1D and equally sized")
    if np.any(rin < 0) or np.any(rout <= rin) or np.any(ht <= 0):
        raise ValueError("foils require 0 <= rin < rout and positive half thickness")
    scalars = np.asarray([b.axis_x, b.ipa_r, b.ipa_t, b.ipa_z0, b.ipa_z1,
                          b.opa_z0, b.opa_z1, geom.opa_rin0, geom.opa_rin1,
                          geom.opa_t, geom.ring_t, geom.ring_hl], dtype=float)
    if not np.all(np.isfinite(scalars)) or min(b.ipa_t, geom.opa_t, geom.ring_t, geom.ring_hl) < 0:
        raise ValueError("geometry scalars must be finite and thicknesses nonnegative")
    if b.ipa_z1 < b.ipa_z0 or b.opa_z1 <= b.opa_z0 or min(b.ipa_r, geom.opa_rin0, geom.opa_rin1) < 0:
        raise ValueError("invalid shell radii or z ranges")
    r, z = np.hypot(xyz[:, 0] - b.axis_x, xyz[:, 1]), xyz[:, 2]
    finite = np.all(np.isfinite(xyz), axis=1)
    count = np.zeros(len(xyz), dtype=np.int32)
    labels = np.full(len(xyz), "other", dtype="U16")

    def add(mask, label):
        mask = mask & finite
        count[mask] += 1
        labels[mask] = label

    for k in range(len(zf)):
        add((np.abs(z - zf[k]) <= ht[k] + tol) & (r >= rin[k] - tol) & (r <= rout[k] + tol), "foil")
    if b.ipa_t > 0:
        add((r >= b.ipa_r - tol) & (r <= b.ipa_r + b.ipa_t + tol)
            & (z >= b.ipa_z0 - tol) & (z <= b.ipa_z1 + tol), "ipa")
    if geom.ring_t > 0 and geom.ring_hl > 0:
        for end in (b.ipa_z0, b.ipa_z1):
            add((r >= b.ipa_r - geom.ring_t - tol) & (r <= b.ipa_r + tol)
                & (z >= end - tol) & (z <= end + 2 * geom.ring_hl + tol), "ring")
    if geom.opa_t > 0:
        fraction = np.clip((z - b.opa_z0) / (b.opa_z1 - b.opa_z0), 0, 1)
        inner = geom.opa_rin0 + fraction * (geom.opa_rin1 - geom.opa_rin0)
        add((z >= b.opa_z0 - tol) & (z <= b.opa_z1 + tol)
            & (r >= inner - tol) & (r <= inner + geom.opa_t + tol), "opa")
    labels[count > 1] = "ambiguous"
    labels[~finite] = "missing_birth"
    return labels


def _denominator(value):
    if value is not None and (isinstance(value, (bool, np.bool_))
                              or not isinstance(value, (int, np.integer)) or value <= 0):
        raise ValueError("certified_generated_events must be an explicitly certified positive integer")
    return None if value is None else int(value)


def summarize(steps, attributed, geom, *, tolerance_mm=1e-3, certified_generated_events=None):
    """Conserve every step in raw region and joint run/subrun/region/root-genpdg totals."""
    from .flash_ancestry import _ids

    denominator = _denominator(certified_generated_events)
    energy = np.asarray(steps["eion"], dtype=float)
    if energy.ndim != 1 or np.any(~np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError("eion must be flat, finite and nonnegative")
    n = len(energy)
    fields = ("photon_index", "photon_sx", "photon_sy", "photon_sz", "root_gen_pdg", "photon_status")
    if any(np.asarray(attributed[name]).shape != (n,) for name in fields):
        raise ValueError("attribution must be step-row aligned")
    if any(np.asarray(steps[name]).shape != (n,) for name in ("run", "subrun")):
        raise ValueError("run/subrun must be step-row aligned")
    shard = _ids(steps, ("run", "subrun"))
    xyz = np.column_stack([attributed[f"photon_s{axis}"] for axis in "xyz"])
    labels = classify(geom, xyz, tolerance_mm)
    labels[np.asarray(attributed["photon_index"]) < 0] = "no_photon"
    code = np.zeros(n, dtype=np.int64)
    for index, label in enumerate(REGIONS):
        code[labels == label] = index
    keys = np.empty(n, dtype=[("run", "i8"), ("subrun", "i8"), ("region", "i8"), ("root_gen_pdg", "i8")])
    keys["run"], keys["subrun"] = shard["run"], shard["subrun"]
    keys["region"], keys["root_gen_pdg"] = code, attributed["root_gen_pdg"]
    unique, inverse = np.unique(keys, return_inverse=True)
    by_region = {"region": list(REGIONS), "step_count": np.bincount(code, minlength=len(REGIONS)).tolist(),
                 "eion_sum_mev": np.bincount(code, weights=energy, minlength=len(REGIONS)).tolist()}
    joint = {name: unique[name].tolist() for name in ("run", "subrun", "root_gen_pdg")}
    joint.update(region=np.asarray(REGIONS)[unique["region"]].tolist(),
                 step_count=np.bincount(inverse, minlength=len(unique)).tolist(),
                 eion_sum_mev=np.bincount(inverse, weights=energy, minlength=len(unique)).tolist())
    statuses, status_inverse = np.unique(attributed["photon_status"], return_inverse=True)
    report = {"step_count": n, "eion_sum_mev": float(energy.sum()), "by_region": by_region,
              "by_subrun_region_root_genpdg": joint,
              "photon_status": {"status": statuses.tolist(),
                                "step_count": np.bincount(status_inverse, minlength=len(statuses)).tolist(),
                                "eion_sum_mev": np.bincount(status_inverse, weights=energy, minlength=len(statuses)).tolist()},
              "normalization": {"certified_generated_events": denominator,
                                "source": "Explicit caller certification" if denominator is not None else "Not certified",
                                "represented_events_are_generated_total": False,
                                "denominator_scope": "Whole input certified generated total, including zero-step events; not a per-subrun denominator"}}
    if denominator is not None:
        report["normalized_totals"] = {"step_count_per_generated_event": n / denominator,
                                       "eion_mev_per_generated_event": float(energy.sum()) / denominator}
        for table in (by_region, joint):
            table["step_count_per_generated_event"] = (np.asarray(table["step_count"]) / denominator).tolist()
            table["eion_mev_per_generated_event"] = (np.asarray(table["eion_sum_mev"]) / denominator).tolist()
    return report


def analyze(path, geom, *, tolerance_mm=1e-3, certified_generated_events=None, geometry_provenance=None):
    """Read highest ROOT cycles and trace each step with existing ancestry joins."""
    import uproot
    from .flash_ancestry import EVENT, GEN, PARTICLE, STEP, _artifact, _ids, _read_tree, trace_steps

    denominator = _denominator(certified_generated_events)
    with uproot.open(path) as root:
        steps, smeta = _read_tree(root, "sgsteps", STEP)
        particles, pmeta = _read_tree(root, "particles", PARTICLE)
        gen, gmeta = _read_tree(root, "gen", GEN)
    represented = len(np.unique(np.concatenate([_ids(table, EVENT) for table in (steps, particles, gen)])))
    if denominator is not None and denominator < represented:
        raise ValueError("Certified generated count is smaller than represented event identities")
    attributed = trace_steps(steps, particles, gen)
    report = summarize(steps, attributed, geom, tolerance_mm=tolerance_mm, certified_generated_events=denominator)
    report.update(schema="flash-regions-v1", tolerance_mm=float(tolerance_mm), model=MODEL.copy(),
                  input=dict(_artifact(path), trees={"sgsteps": smeta, "particles": pmeta, "gen": gmeta}),
                  represented_any_tree_events=represented,
                  represented_step_events=len(np.unique(_ids(steps, EVENT))), max_ancestry_depth=64,
                  geometry=geometry_provenance if geometry_provenance is not None else {"source": "Caller supplied; not certified against production geometry"})
    return report


def _load_geometry(overlay, offline_root):
    from dataclasses import asdict
    from . import cechain, simpleconfig
    from .flash_ancestry import _artifact

    geom = cechain.from_offline(overlay, offline_root)
    path = Path(overlay) if Path(overlay).is_absolute() else Path(offline_root) / overlay
    sources = []
    simpleconfig.read(path, offline_root, source_files=sources)
    parameters = asdict(geom)
    parameters["beam"] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                          for key, value in parameters["beam"].items()}
    return geom, {"reader": "cechain.from_offline -> geometry.from_offline + simpleconfig.read",
                  "overlay": str(path.resolve()), "offline_root": str(Path(offline_root).resolve()),
                  "source_files": [_artifact(source) for source in sources], "parameters": parameters,
                  "certification": "Hashes identify geometry inputs, not proof they match ROOT production"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Flat dump ROOT file")
    parser.add_argument("--geom", required=True, help="Offline SimpleConfig overlay")
    parser.add_argument("--offline-root", required=True)
    parser.add_argument("--generated-events", type=int, help="Explicitly certify total generated events, including zero-step events")
    parser.add_argument("--tolerance-mm", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    _denominator(args.generated_events)
    geom, provenance = _load_geometry(args.geom, args.offline_root)
    report = analyze(args.input, geom, tolerance_mm=args.tolerance_mm,
                     certified_generated_events=args.generated_events, geometry_provenance=provenance)
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
