"""Vectorized ancestry accounting for flat dump/{sgsteps,particles,gen} tables.

Run with ``python -m beamfastsim.flash_ancestry FILE ... --out report.json
--z-bins Z0 Z1 ... --r-bins R0 R1 ...``. Coordinates are the particle birth
coordinates as dumped (Mu2e frame, mm); r = hypot(sx, sy), not a material tag.
Photon ancestry means the nearest *retained*, strict photon ancestor. Compression
can remove intermediate ancestors. A retained root is not certified as the original
primary; a generator link is reported only when that root's genidx actually joins.
Counts of represented events never certify generated events, including zero-step
events. An optional generated-event denominator must be explicitly certified by
the caller. Hashes identify input artifacts, not their production configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot

EVENT = ("run", "subrun", "event")
PARTICLE = EVENT + ("key", "parent", "pdg", "ccode", "sx", "sy", "sz", "spx", "spy", "spz", "se", "genidx")
STEP = EVENT + ("sp", "eion", "t")
GEN = EVENT + ("idx", "pdg")
STATUSES = ("root_reached", "missing_parent", "cycle", "depth_limit")


def _table(table, fields):
    result = {name: np.asarray(table[name]) for name in fields}
    lengths = {len(value) for value in result.values()}
    if len(lengths) != 1 or any(value.ndim != 1 for value in result.values()):
        raise ValueError("Flat columns must have identical lengths")
    return result


def _ids(table, fields):
    """Structured identities avoid packed-key overflow and cross-event joins."""
    result = np.empty(len(table[fields[0]]), dtype=[(name, "i8") for name in fields])
    for name in fields:
        value = table[name]
        if value.dtype.kind not in "iuf" or np.any(~np.isfinite(value)):
            raise ValueError(f"Invalid identity column: {name}")
        if np.any(value < -(2**63)) or np.any(value >= 2**63) or np.any(value != np.floor(value)):
            raise ValueError(f"Nonintegral or overflowing identity column: {name}")
        if name in EVENT and np.any(value < 0):
            raise ValueError(f"Missing event identity column: {name}")
        result[name] = value
    return result


def _join(source, query):
    """Return -1 for absent identities, including an entirely empty source."""
    out = np.full(len(query), -1, dtype=np.int64)
    if not len(source):
        return out
    order = np.argsort(source)
    sorted_ids = source[order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError("Duplicate particle/generator keys within run/subrun/event")
    pos = np.searchsorted(sorted_ids, query)
    safe = np.minimum(pos, len(source) - 1)
    found = (pos < len(source)) & (sorted_ids[safe] == query)
    out[found] = order[safe[found]]
    return out


def _take(values, indices, missing):
    """Never let a missing -1 index silently select the last particle."""
    out = np.full(len(indices), missing, dtype=np.asarray(values).dtype)
    valid = indices >= 0
    out[valid] = values[indices[valid]]
    return out


def trace_steps(steps, particles, gen=None, *, max_depth=64):
    """Return row-aligned step attribution; no Python loop over data rows.

    max_depth is the maximum number of parent edges. Photon status may be found
    while root status remains unresolved: these are deliberately separate claims.
    Floyd's vectorized walker identifies cycles; longer unresolved chains remain
    explicitly depth_limit. Negative sp/parent/genidx are absent-ID sentinels.
    """
    if not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    s, p = _table(steps, STEP), _table(particles, PARTICLE)
    n = len(p["key"])
    identity = _ids(p, EVENT + ("key",))
    parent_query = dict(p, key=p["parent"])
    parent = _join(identity, _ids(parent_query, EVENT + ("key",)))
    parent[p["parent"] < 0] = -1
    step_query = dict(s, key=s["sp"])
    depositing = _join(identity, _ids(step_query, EVENT + ("key",)))
    depositing[s["sp"] < 0] = -1
    origin = np.arange(n)
    current = origin.copy()
    slow, fast = origin.copy(), origin.copy()
    cycle = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)
    photon = np.full(n, -1, dtype=np.int64)
    root = np.full(n, -1, dtype=np.int64)
    status = np.full(n, "depth_limit", dtype="U24")
    for depth in range(max_depth + 1):
        pdg = _take(p["pdg"], current, 0)
        found = active & (depth > 0) & (current != origin) & (pdg == 22) & (photon < 0)
        photon[found] = current[found]
        raw_parent = _take(p["parent"], current, -1)
        next_index = _take(parent, current, -1)
        at_root = active & (raw_parent < 0)
        missing = active & ~at_root & (next_index < 0)
        root[at_root] = current[at_root]
        status[at_root] = "root_reached"
        status[missing] = "missing_parent"
        active &= ~(at_root | missing)
        if depth == max_depth or not np.any(active):
            break
        current = np.where(active, next_index, -1)
        slow = _take(parent, slow, -1)
        fast = _take(parent, _take(parent, fast, -1), -1)
        cycle |= (slow >= 0) & (slow == fast)
    status[active & cycle] = "cycle"
    photon_status = np.where(photon >= 0, "found", np.where(status == "root_reached", "no_retained_photon", status))
    out = {"particle_index": depositing,
           "depositing_pdg": _take(p["pdg"], depositing, 0),
           "depositing_ccode": _take(p["ccode"], depositing, 0),
           "photon_index": _take(photon, depositing, -1),
           "photon_status": _take(photon_status, depositing, "missing_particle"),
           "root_index": _take(root, depositing, -1),
           "root_status": _take(status, depositing, "missing_particle")}
    for prefix in ("photon", "root"):
        index = out[f"{prefix}_index"]
        for name in ("key", "pdg", "ccode", "genidx"):
            out[f"{prefix}_{name}"] = _take(p[name], index, -1)
        for name in ("sx", "sy", "sz", "spx", "spy", "spz", "se"):
            out[f"{prefix}_{name}"] = _take(np.asarray(p[name], dtype=float), index, np.nan)
    mother = _take(parent, depositing, -1)
    out["direct_electron_gamma_mother"] = ((out["depositing_pdg"] == 11) & (_take(p["pdg"], mother, 0) == 22))
    out["root_gen_index"] = np.full(len(depositing), -1, dtype=np.int64)
    out["root_gen_pdg"] = np.full(len(depositing), -1, dtype=np.int64)
    if gen is not None:
        g = _table(gen, GEN)
        root_gen = _join(_ids(g, EVENT + ("idx",)), _ids(dict(s, idx=out["root_genidx"]), EVENT + ("idx",)))
        root_gen[(out["root_index"] < 0) | (out["root_genidx"] < 0)] = -1
        out["root_gen_index"] = root_gen
        out["root_gen_pdg"] = _take(g["pdg"], root_gen, -1)
    return out


def _edges(values, name):
    edges = np.asarray(values, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} bins must be finite, strictly increasing edges")
    return edges


def summarize(steps, attributed, z_bins, r_bins):
    """Persist counts and ionizing energy sums (MeV), including unresolved energy."""
    z, r = _edges(z_bins, "z"), _edges(r_bins, "r")
    energy = np.asarray(steps["eion"], dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError("eion must be finite and nonnegative")
    matched = attributed["particle_index"] >= 0
    keys = np.column_stack((matched.astype(int), attributed["depositing_pdg"], attributed["depositing_ccode"]))
    groups, inv = np.unique(keys, axis=0, return_inverse=True)
    known = attributed["photon_index"] >= 0
    birth_z = attributed["photon_sz"]
    birth_r = np.hypot(attributed["photon_sx"], attributed["photon_sy"])
    finite = known & np.isfinite(birth_z) & np.isfinite(birth_r)
    inside = finite & (birth_z >= z[0]) & (birth_z <= z[-1]) & (birth_r >= r[0]) & (birth_r <= r[-1])
    hist = np.histogram2d(birth_z[finite], birth_r[finite], bins=(z, r))[0]
    hist_energy = np.histogram2d(birth_z[finite], birth_r[finite], bins=(z, r), weights=energy[finite])[0]
    report = {"step_count": len(energy), "eion_sum_mev": float(energy.sum()),
              "by_depositor": {"particle_found": groups[:, 0].astype(bool).tolist(),
                               "pdg": groups[:, 1].tolist(), "creation_code": groups[:, 2].tolist(),
                               "step_count": np.bincount(inv, minlength=len(groups)).tolist(),
                               "eion_sum_mev": np.bincount(inv, weights=energy, minlength=len(groups)).tolist()},
              "photon_birth": {"z_edges_mm": z.tolist(), "r_edges_mm": r.tolist(),
                               "step_count": hist.astype(np.int64).tolist(), "eion_sum_mev": hist_energy.tolist(),
                               "coordinate_convention": "Dumped Mu2e birth frame; r=hypot(sx,sy); no material inference",
                               "bin_convention": "Left closed, right open; final right edge included"}}
    shard_ids, shard_inv = np.unique(_ids(steps, ("run", "subrun")), return_inverse=True)
    report["by_subrun"] = dict(run=shard_ids["run"].tolist(), subrun=shard_ids["subrun"].tolist(),
                                step_count=np.bincount(shard_inv, minlength=len(shard_ids)).tolist(),
                                eion_sum_mev=np.bincount(shard_inv, weights=energy, minlength=len(shard_ids)).tolist(),
                                scope="Step-bearing subruns only; no generated-event denominator or paired-source certification")
    reached = attributed["root_index"] >= 0
    linked_genidx = np.where(attributed["root_gen_index"] >= 0, attributed["root_genidx"], -1)
    root_columns = ("run", "subrun", "event", "key", "pdg", "creation_code", "linked_gen_idx", "linked_gen_pdg")
    root_values = np.column_stack((steps["run"], steps["subrun"], steps["event"], attributed["root_key"],
                                   attributed["root_pdg"], attributed["root_ccode"], linked_genidx,
                                   attributed["root_gen_pdg"]))[reached].astype(np.int64)
    roots, root_inv = np.unique(root_values, axis=0, return_inverse=True)
    report["retained_roots"] = {name: roots[:, column].tolist() for column, name in enumerate(root_columns)}
    report["retained_roots"].update(
        step_count=np.bincount(root_inv, minlength=len(roots)).tolist(),
        eion_sum_mev=np.bincount(root_inv, weights=energy[reached], minlength=len(roots)).tolist(),
        scope="Reached retained roots only; not certified original primaries. Unjoined generator IDs are -1.")
    for field in ("photon_status", "root_status"):
        labels = ("found", "no_retained_photon", "missing_particle") + STATUSES
        report[field] = {label: {"step_count": int(np.count_nonzero(attributed[field] == label)),
                                 "eion_sum_mev": float(energy[attributed[field] == label].sum())} for label in labels}
    for label, mask in (("direct_electron_gamma_mother", attributed["direct_electron_gamma_mother"]),
                        ("generator_linked_retained_root", attributed["root_gen_index"] >= 0)):
        report[label] = {"step_count": int(mask.sum()), "eion_sum_mev": float(energy[mask].sum())}
    iz = np.searchsorted(z, birth_z, side="right") - 1
    ir = np.searchsorted(r, birth_r, side="right") - 1
    iz = np.where(birth_z == z[-1], len(z) - 2, iz)
    ir = np.where(birth_r == r[-1], len(r) - 2, ir)
    source_keys = np.column_stack((steps["run"], steps["subrun"], attributed["root_gen_pdg"],
                                  np.where(inside, iz, -1), np.where(inside, ir, -1)))
    source_groups, source_inverse = np.unique(source_keys, axis=0, return_inverse=True)
    source_columns = ("run", "subrun", "linked_root_generator_pdg", "photon_z_bin", "photon_r_bin")
    report["source_by_subrun"] = {name: source_groups[:, column].tolist()
                                   for column, name in enumerate(source_columns)}
    report["source_by_subrun"].update(
        step_count=np.bincount(source_inverse, minlength=len(source_groups)).tolist(),
        eion_sum_mev=np.bincount(source_inverse, weights=energy, minlength=len(source_groups)).tolist(),
        scope="Joint retained-photon birth bins and linked root generator species; bin -1 is unresolved or out of range")
    for label, mask in (("no_photon", ~known), ("nonfinite_birth", known & ~finite),
                        ("outside_bins", finite & ~inside)):
        report["photon_birth"][label] = {"step_count": int(mask.sum()), "eion_sum_mev": float(energy[mask].sum())}
    return report


def _artifact(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _read_tree(root, name, fields):
    directory = root["dump"]
    cycles = sorted(int(key.rsplit(";", 1)[1]) for key in directory.keys(cycle=True, recursive=False)
                    if key.rsplit(";", 1)[0] == name)
    if not cycles:
        raise ValueError(f"Missing dump/{name} tree")
    key = f"dump/{name};{cycles[-1]}"
    tree = root[key]
    arrays = tree.arrays(list(fields), library="np")
    return {field: arrays[field] for field in fields}, {"selected_key": key, "entries": int(tree.num_entries),
            "cycles": {str(c): int(root[f"dump/{name};{c}"].num_entries) for c in cycles}}


def analyze(paths, *, z_bins, r_bins, max_depth=64, certified_generated_events=None, artifacts=()):
    """Analyze each file independently; ambiguous duplicate keys across files fail.

    Identical event IDs in different input files are rejected rather than silently
    concatenated or used as separate generated events. All three trees are required,
    though any may have zero rows. No gen-row count is a generated-event certificate.
    """
    z, r = _edges(z_bins, "z"), _edges(r_bins, "r")
    if certified_generated_events is not None and (not isinstance(certified_generated_events, (int, np.integer))
                                                  or certified_generated_events <= 0):
        raise ValueError("Certified generated-event count must be a positive integer")
    files, event_ids, step_ids, gen_ids = [], [], [], []
    total_steps, total_particles, total_gen = 0, 0, 0
    for path in paths:
        with uproot.open(path) as root:
            steps, smeta = _read_tree(root, "sgsteps", STEP)
            particles, pmeta = _read_tree(root, "particles", PARTICLE)
            gen, gmeta = _read_tree(root, "gen", GEN)
        attr = trace_steps(steps, particles, gen, max_depth=max_depth)
        summary = summarize(steps, attr, z, r)
        ids = np.unique(np.concatenate([_ids(table, EVENT) for table in (steps, particles, gen)]))
        event_ids.append(ids)
        step_ids.append(np.unique(_ids(steps, EVENT)))
        gen_ids.append(np.unique(_ids(gen, EVENT)))
        total_steps += len(steps["sp"])
        total_particles += len(particles["key"])
        total_gen += len(gen["idx"])
        files.append(dict(_artifact(path), trees={"sgsteps": smeta, "particles": pmeta, "gen": gmeta},
                          represented_step_events=len(step_ids[-1]), summary=summary))
    if not files:
        raise ValueError("At least one input file is required")
    all_events = np.concatenate(event_ids)
    if len(np.unique(all_events)) != len(all_events):
        raise ValueError("Overlapping run/subrun/event identities across input files")
    represented = len(np.unique(np.concatenate(step_ids)))
    if certified_generated_events is not None and certified_generated_events < len(all_events):
        raise ValueError("Certified count is smaller than represented event identities")
    total_energy = sum(file["summary"]["eion_sum_mev"] for file in files)
    return {"schema": "flash-ancestry-v1", "files": files, "artifacts": [_artifact(path) for path in artifacts],
            "counts": {"step_rows": total_steps, "particle_rows": total_particles, "gen_rows": total_gen,
                       "represented_step_events": represented, "represented_gen_events": len(np.unique(np.concatenate(gen_ids))),
                       "represented_any_tree_events": len(all_events)},
            "max_depth": max_depth, "eion_sum_mev": total_energy,
            "normalization": {"certified_generated_events": certified_generated_events,
                              "source": "Explicit caller certification" if certified_generated_events is not None else "Not certified",
                              "eion_mev_per_generated_event": None if certified_generated_events is None else total_energy / certified_generated_events,
                              "represented_step_events_are_generated_total": False},
            "interpretation": "Nearest retained photon only; compressed ancestry may be incomplete. A retained root is not the original primary."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--z-bins", nargs="+", type=float, required=True)
    parser.add_argument("--r-bins", nargs="+", type=float, required=True)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--certified-generated-events", type=int)
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args(argv)
    report = analyze(args.files, z_bins=args.z_bins, r_bins=args.r_bins, max_depth=args.max_depth,
                     certified_generated_events=args.certified_generated_events, artifacts=args.artifact)
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
