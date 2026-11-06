"""Same-stage Run 1A VD1-to-DS reference; no restarted-particle assumptions."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import tssmoke
from .tsreference import ids, load, lookup, read_table, save, state, subset


def build(particles, vd, beam, *, compact=True, event_id=None):
    """Preserve every entrance track, crossing and endpoint with exact ancestry keys.

    The aligned stock view is emitted only for fully resolved, single-endpoint
    lineages without nested entrance tracks. Flat endpoint/crossing tables retain
    multiplicities otherwise; unresolved endpoints are never classified as losses.
    """
    report = tssmoke.census(particles, vd, beam, compact=compact)
    pk = ids(particles)
    parent_keys = pk.copy()
    parent_keys[:, 3] = ids(particles, ("parent_id",))[:, 0]
    parent = lookup(pk, parent_keys)
    missing = (parent_keys[:, 3] > 0) & (parent < 0)
    crossing = subset(vd, (vd["sid"] == 1) & np.isin(np.abs(vd["pdg"]), [13, 211]) & (vd["pz"] > 0))
    _, inverse = np.unique(ids(crossing), axis=0, return_inverse=True)
    order = np.lexsort((crossing["time"], inverse))
    first = order[np.r_[True, inverse[order][1:] != inverse[order][:-1]]] if len(order) else order
    source = subset(crossing, first)
    source_rows = lookup(pk, ids(source))
    source_at = np.full(len(pk), -1, dtype=np.int64)
    source_at[source_rows] = np.arange(len(source_rows))
    ends = subset(beam, np.isin(np.abs(beam["pdg"]), [13, 211]))

    def trace(rows):
        current = rows.copy()
        root = np.full(len(rows), -1, dtype=np.int64)
        incomplete = np.zeros(len(rows), dtype=bool)
        depth = 0
        while np.any(current >= 0):
            if depth > len(pk):
                raise ValueError("cyclic particle ancestry")
            active = np.flatnonzero(current >= 0)
            found = source_at[current[active]]
            matched = active[found >= 0]
            root[matched] = found[found >= 0]
            current[matched] = -1
            unresolved = active[found < 0]
            incomplete[unresolved] |= missing[current[unresolved]]
            current[unresolved] = parent[current[unresolved]]
            depth += 1
        return root, incomplete

    roots, incomplete = trace(lookup(pk, ids(ends)))
    source_parent, source_parent_incomplete = trace(parent[source_rows])
    source_parent_incomplete |= missing[source_rows]
    nested = source_parent >= 0
    n = len(source_rows)
    multiplicity = np.bincount(roots[roots >= 0], minlength=n)
    observed = np.unique(pk[:, :3], axis=0)
    if event_id is None:
        event_id = observed
    else:
        event_id = np.asarray(event_id)
        if event_id.ndim != 2 or event_id.shape[1] != 3 or event_id.dtype.kind not in "iu":
            raise ValueError("event_id must be an integer N by 3 table")
        lookup(event_id, event_id)
        if np.any(lookup(event_id, observed) < 0):
            raise ValueError("event_id does not cover observed particles")
    shard_ids, shard_counts = np.unique(event_id[:, :2], axis=0, return_counts=True)
    arrays = {"id": ids(source), **state(source), "shard": ids(source, ("subrun",))[:, 0],
              "event_id": event_id, "parent_shard_ids": shard_ids, "parent_shard_event_counts": shard_counts,
              "crossing_id": ids(crossing), **state(crossing, "crossing_"), "crossing_source": inverse,
              "endpoint_id": ids(ends), **state(ends, "endpoint_"), "endpoint_source": roots,
              "endpoint_ancestry_incomplete": incomplete, "stock_multiplicity": multiplicity,
              "source_parent_source": source_parent, "source_parent_ancestry_incomplete": source_parent_incomplete}
    for prefix in ("", "crossing_", "endpoint_"):
        for k in ("x", "p", "t"):
            if not np.all(np.isfinite(arrays[prefix + k])):
                raise ValueError(f"nonfinite {prefix + k}")
    usable = not (np.any(incomplete) or np.any(multiplicity > 1) or np.any(nested)
                  or report["linked_ds_rows_preceding_source"])
    if usable:
        matched = roots >= 0
        aligned_roots = roots[matched]
        arrays["stock_reached"] = multiplicity > 0
        arrays["stock_id"] = np.full((n, 4), -1, dtype=np.int64)
        arrays["stock_id"][aligned_roots] = ids(ends)[matched]
        for k, v in state(ends, "stock_").items():
            arrays[k] = np.full((n,) + v.shape[1:], 0 if k == "stock_pdg" else np.nan, dtype=v.dtype)
            arrays[k][aligned_roots] = v[matched]
    report.update(aligned_reference_available=usable, nested_source_tracks=int(nested.sum()),
                  source_parent_ancestry_incomplete=int(source_parent_incomplete.sum()),
                  state_kind="recorded VD1 pre-step entrance and DS Beam pre-step endpoint",
                  source_selection="VD sid1, mu/pi, pz>0; earliest crossing per exact particle ID",
                  endpoint_selection="all recorded DS Beam mu/pi rows; nearest exact source ancestor",
                  geometry="Run1A geom_run1_a.txt; field bfgeom_no_ds_v01.txt; Offline v10_47_00",
                  limits=report["limits"] + [
                      "flat crossing/endpoint tables retain repeat crossings and multiple descendants",
                      "aligned transport comparison unavailable for unresolved DS ancestry, branching or nested sources",
                      "source ancestry gaps upstream of VD1 are reported, not asserted to be complete",
                      "unrelated DS particles are retained with endpoint_source=-1, not assigned by proximity"])
    return arrays, report


def read(ntuple):
    particles, _ = read_table(ntuple, "ntpart")
    vd, _ = read_table(ntuple, "ntvd")
    beam, _ = read_table(ntuple, "nttvd")
    return build(particles, vd, beam)


def root_lineages(a):
    """Project known nested entrances to their ancestor entrance, preserving all event IDs.

    Original per-track references remain the backing; nested states and root IDs
    are also retained in this projection. Missing ancestry above an entrance is
    still a reported limitation, not a claim of a complete primary-root census.
    """
    n = len(a["id"])
    parent = a["source_parent_source"]
    if parent.shape != (n,) or parent.dtype.kind not in "iu" or np.any((parent < -1) | (parent >= n)):
        raise ValueError("invalid source parent indices")
    nested = np.flatnonzero(parent >= 0)
    if (np.any(a["id"][nested, :3] != a["id"][parent[nested], :3])
            or np.any(a["t"][nested] < a["t"][parent[nested]])):
        raise ValueError("nested source must follow its ancestor in the same event")
    roots = np.arange(n)
    for _ in range(n + 1):
        active = np.flatnonzero(parent[roots] >= 0)
        if not len(active):
            break
        roots[active] = parent[roots[active]]
    else:
        raise ValueError("cyclic source ancestry")
    keep = np.flatnonzero(parent < 0)
    renumber = np.full(n, -1, dtype=np.int64)
    renumber[keep] = np.arange(len(keep))
    mapping = renumber[roots]
    out = {k: v.copy() for k, v in a.items() if not k.startswith("stock_")}
    for k in ("id", "x", "p", "t", "pdg", "shard", "source_parent_source", "source_parent_ancestry_incomplete"):
        out[k] = a[k][keep].copy()
    for k in ("id", "x", "p", "t", "pdg"):
        out["nested_" + k] = a[k][nested].copy()
    out["nested_root_id"] = a["id"][roots[nested]].copy()
    for k in ("crossing_source", "endpoint_source"):
        index = a[k]
        if index.dtype.kind not in "iu" or np.any((index < -1) | (index >= n)):
            raise ValueError(f"invalid {k} indices")
        out[k] = np.full(index.shape, -1, dtype=np.int64)
        present = index >= 0
        out[k][present] = mapping[index[present]]
    if np.any(a["endpoint_ancestry_incomplete"]):
        raise ValueError("unresolved DS ancestry cannot define source losses")
    reached = out["endpoint_source"] >= 0
    eroot = out["endpoint_source"][reached]
    multiplicity = np.bincount(eroot, minlength=len(keep))
    if np.any(multiplicity > 1):
        raise ValueError("multiple DS endpoints for a root lineage")
    if np.any(a["endpoint_t"][reached] < out["t"][eroot]):
        raise ValueError("DS endpoint precedes its root entrance")
    out["stock_multiplicity"] = multiplicity
    out["stock_reached"] = multiplicity > 0
    out["stock_id"] = np.full((len(keep), 4), -1, dtype=np.int64)
    out["stock_id"][eroot] = a["endpoint_id"][reached]
    for k in ("x", "p", "t", "pdg"):
        v = a["endpoint_" + k]
        out["stock_" + k] = np.full((len(keep),) + v.shape[1:], 0 if k == "pdg" else np.nan, dtype=v.dtype)
        out["stock_" + k][eroot] = v[reached]
    return out


def combine(paths, *, root_cohort=False, cohort="development"):
    """Merge explicitly selected references, optionally projecting known root lineages."""
    if cohort not in ("development", "heldout"):
        raise ValueError("cohort must be development or explicitly heldout")
    tables = [load(p) for p in paths]
    reports = [json.loads(Path(p).with_suffix(".json").read_text()) for p in paths]
    recorded_tracks = sum(len(a["id"]) for a in tables)
    if not tables:
        raise ValueError("nonempty reference list required")
    if root_cohort:
        tables = [root_lineages(a) for a in tables]
    elif not all(r["aligned_reference_available"] for r in reports):
        raise ValueError("all references must have resolved single-endpoint lineages")
    offset = 0
    for a in tables:
        for k in ("endpoint_source", "crossing_source", "source_parent_source"):
            a[k] = np.where(a[k] >= 0, a[k] + offset, -1)
        offset += len(a["id"])
    out = {k: np.concatenate([a[k] for a in tables]) for k in tables[0]}
    lookup(out["id"], out["id"])
    lookup(out["event_id"], out["event_id"])
    lookup(out["parent_shard_ids"], out["parent_shard_ids"])
    first = 920000 if cohort == "development" else 920100
    if np.any((out["parent_shard_ids"][:, 1] < first) | (out["parent_shard_ids"][:, 1] >= first + 100)):
        raise ValueError(f"{cohort} Run1A subruns only")
    report = {"cohort": cohort, "files": list(map(str, paths)), "source_tracks": len(out["id"]),
              "events": len(out["event_id"]), "shards": len(out["parent_shard_ids"]),
              "stock_reached": int(out["stock_reached"].sum()),
              "recorded_source_tracks": recorded_tracks,
              "nested_tracks_represented_by_ancestor": recorded_tracks - len(out["id"]),
              "cohort_definition": "known root entrance lineages" if root_cohort else "all recorded entrance tracks",
              "upstream_ancestry_incomplete_sources": int(out["source_parent_ancestry_incomplete"].sum()),
              "limits": reports[0]["limits"] + ([
                  "known nested entrances represented once by their ancestor, with all descendants remapped",
                  "compact upstream ancestry can hide other nesting; this is not certified complete root ancestry"]
                  if root_cohort else []), "aligned_reference_available": True}
    return out, report


def diagnose(path):
    """Describe recorded nested source states without reopening bulk diagnostic files."""
    a = load(path)
    nested = np.flatnonzero(a["source_parent_source"] >= 0)
    parent = a["source_parent_source"][nested]
    keys = ("id", "pdg", "x", "p", "t")
    return {"nested_rows": nested.tolist(), "parent_rows": parent.tolist(),
            "child": {k: a[k][nested].tolist() for k in keys},
            "parent": {k: a[k][parent].tolist() for k in keys},
            "endpoint_source": a["endpoint_source"].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose")
    parser.add_argument("--ntuple")
    parser.add_argument("--combine", nargs="+")
    parser.add_argument("--root-lineages", action="store_true", help="represent known nested entrances once by their ancestor")
    parser.add_argument("--cohort", choices=("development", "heldout"), default="development",
                        help="explicit cohort for combining references; default never accepts held-out subruns")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gdml", help="check source and endpoint coverage in the matching saved geometry")
    args = parser.parse_args()
    if args.diagnose:
        report = diagnose(args.diagnose)
        Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    if args.combine:
        arrays, report = combine(args.combine, root_cohort=args.root_lineages, cohort=args.cohort)
    elif args.ntuple:
        arrays, report = read(args.ntuple)
    else:
        parser.error("use --ntuple or --combine")
    if args.gdml:
        from .tsgeometry import Geometry
        geometry = Geometry(args.gdml)
        report["geometry_check"] = {"path": args.gdml, "z_start": geometry.z_start, "z_end": geometry.z_end,
                                    "source_outside": int(np.sum(~geometry.inside(arrays["x"]))),
                                    "endpoint_outside": int(np.sum(~geometry.inside(arrays["endpoint_x"]))),
                                    "materials": sorted(geometry.materials)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save(arrays, args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
