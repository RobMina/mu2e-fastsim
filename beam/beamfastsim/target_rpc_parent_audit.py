"""Exact matching-parent RPC exposure for the audited one-eligible-pion-per-event case."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import geometry as G
from .target_parent_audit import checksum, physical_event_order, production_exposure


COLUMNS = ("run", "subrun", "evt", "trk", "pdg", "codestop", "pstop", "xstop", "ystop", "zstop", "tstop")


def reduce_parent(columns, geom, expected_events):
    """Fail on multiplicity or empty events rather than flattening RPCGun's event-then-pion law."""
    a = {k: np.asarray(columns[k]) for k in COLUMNS}
    n = a["pdg"].size
    if not n or any(v.shape != (n,) or not np.isfinite(v).all() for v in a.values()):
        raise ValueError("finite equal-length parent columns required")
    identity = np.column_stack([a[k] for k in ("run", "subrun", "evt", "trk")])
    if np.unique(identity, axis=0).shape[0] != n:
        raise ValueError("duplicate parent event/track identity")
    event, first, inverse = np.unique(identity[:, :3], axis=0, return_index=True, return_inverse=True)
    if event.shape[0] != expected_events:
        raise ValueError("parent tree does not contain every expected event")
    order = np.argsort(first)
    rank = np.empty(order.size, int)
    rank[order] = np.arange(order.size)
    index = rank[inverse]
    if np.any(np.diff(index) < 0):
        raise ValueError("parent event rows are not contiguous")
    sel = (a["pdg"] == -211) & (a["codestop"] == 78) & (a["tstop"] >= 350.)
    counts = np.bincount(index[sel], minlength=expected_events)
    if np.any(counts != 1):
        raise ValueError("limited RPC adapter requires exactly one eligible pion per parent event")
    out = {k: v[sel] for k, v in a.items()}
    if np.any(out["pstop"] != 0):
        raise ValueError("eligible pions have nonzero stop momentum")
    out.update(at_rest=np.ones(int(sel.sum()), bool), event_index=index[sel],
               event_keys=event[order], event_multiplicity=counts)
    out["foil"] = G.foil_index(geom, out["xstop"], out["ystop"], out["zstop"])
    if np.any(out["foil"] < 0):
        raise ValueError("eligible pion outside target foils")
    return out


def validate_reference(reference, files):
    if "creation_code" not in reference or int(reference["creation_code"]) != 179:
        raise ValueError("RPC internal primary-electron reference required")
    ng = np.asarray(reference["gencount"])
    if ng.shape != (files,) or np.any(ng <= 0) or np.any(ng != np.floor(ng)):
        raise ValueError("one positive generated-event exposure per parent required")
    group = np.asarray(reference["ifile"])
    if not np.array_equal(np.unique(group), np.arange(files)):
        raise ValueError("reference must identify all parent files")
    identity = np.column_stack([reference[k] for k in ("ifile", "subrun", "evt")])
    if np.unique(identity, axis=0).shape[0] != group.size:
        raise ValueError("RPC adapter requires one primary electron per retained event")
    return ng.astype(int)


def exact_window(parent, reference, generated_events):
    """Resolve repeated physical stops only through a unanimous mixer phase from unique anchors."""
    events = np.asarray(reference["evt"])
    if (not np.isfinite(events).all() or np.any(events != np.floor(events))
            or np.any(events < 1) or np.any(events > generated_events)):
        raise ValueError("RPC reference event outside integer generated window")
    xyz = np.column_stack([parent[k] for k in ("xstop", "ystop", "zstop")]).astype('<f4')
    query = np.column_stack([reference[k] for k in ("x", "y", "z")]).astype('<f4')
    dtype = np.dtype([(k, '<f4') for k in ('x', 'y', 'z')])
    keys = np.ascontiguousarray(xyz).view(dtype).ravel()
    q = np.ascontiguousarray(query).view(dtype).ravel()
    ordered = np.sort(keys)
    multiplicity = np.searchsorted(ordered, q, 'right') - np.searchsorted(ordered, q, 'left')
    if np.any(multiplicity == 0) or not np.any(multiplicity == 1):
        raise ValueError("RPC births need exact parent matches and unique phase anchors")
    anchors = {k: np.asarray(v)[multiplicity == 1] for k, v in reference.items()}
    record = production_exposure(parent, anchors, generated_events)["files"][0]
    offset = record["inferred_first_parent_event_index"]
    index = (offset + events.astype(int) - 1) % parent["event_multiplicity"].size
    rows = np.argsort(parent["event_index"])[index]
    if not np.array_equal(xyz[rows], query):
        raise ValueError("not every RPC birth agrees with the anchored exact mixer phase")
    if not np.array_equal(np.asarray(parent["tstop"])[rows].astype('<f4'), np.asarray(reference["time"]).astype('<f4')):
        raise ValueError("RPC birth times do not exactly match phase-resolved pion stop times")
    record.update(matched_electrons=int(query.shape[0]), matched_generated_events=int(query.shape[0]),
                  unique_phase_anchor_rows=int((multiplicity == 1).sum()),
                  repeated_stop_rows_resolved_by_exact_phase=int((multiplicity > 1).sum()))
    return record


def audit(roots, arts, expected, reference_path, geom, byteswap=False):
    import uproot
    if not len(roots) == len(arts) == len(expected):
        raise ValueError("parallel parent-root/art/expected-events lists required")
    with np.load(reference_path) as ref:
        ng = validate_reference(ref, len(roots))
        columns, reports = [], []
        for i, (root_path, art_path, nevent) in enumerate(zip(roots, arts, expected)):
            with uproot.open(root_path) as root:
                parent = reduce_parent(root["cetree/ntpart"].arrays(list(COLUMNS), library="np"), geom, nevent)
                cycles = {k: int(root[k].num_entries) for k in root.keys() if k.split(";")[0] == "cetree/ntpart"}
            parent, ordering = physical_event_order(parent, art_path, byteswap)
            chosen = np.asarray(ref["ifile"]) == i
            subset = {k: ref[k][chosen] for k in ("x", "y", "z", "time", "evt", "ifile")}
            record = exact_window(parent, subset, int(ng[i]))
            offset = record["inferred_first_parent_event_index"]
            physical_events = (offset + np.arange(int(ng[i]))) % nevent
            row_for_event = np.argsort(parent["event_index"])
            rows = row_for_event[physical_events]
            if rows.size != record["generated_electrons"]:
                raise ValueError("window row count disagrees with exact exposure")
            out = {k: parent[k][rows] for k in COLUMNS + ("at_rest", "foil", "event_index")}
            out.update(parent_stop_index=rows, production_ifile=np.full(rows.size, i, np.int32),
                       rpc_event_occurrence=np.arange(int(ng[i])) + sum(ng[:i]))
            columns.append(out)
            record["denominator_method"] = "exact retained-birth join and unanimous physical mixer offset; one eligible pion and one primary electron per generated capture"
            reports.append(dict(**record, parent_events=nevent, tree_cycles=cycles, physical_event_order=ordering,
                                parent_art_sha256=checksum(art_path), parent_tree_sha256=checksum(root_path)))
    out = {k: np.concatenate([part[k] for part in columns]) for k in columns[0]}
    report = dict(mode="exact-RPC-one-pion-parent-window-population", files=reports,
                  reference_sha256=checksum(reference_path), exact_production_electron_exposure_known=True,
                  total_generated_electrons=int(sum(ng)), filter_implication_certified=False,
                  generated_yield_validated=False,
                  assumptions="released RPCGun emits one e-/e+ pair per capture; all audited parent events have exactly one eligible negative pion; no empty StopSelector events",
                  caveat="exact source population and exposure only, not detector-filter completeness, pair transport or yield certification")
    return out, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent-root", action="append", required=True)
    ap.add_argument("--parent-art", action="append", required=True)
    ap.add_argument("--expected-events", action="append", type=int, required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_run1_a.txt")
    ap.add_argument("--fileindex-byteswap", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out, report = audit(a.parent_root, a.parent_art, a.expected_events, a.reference,
                        G.from_offline(a.geom, a.offline_root), a.fileindex_byteswap)
    np.savez_compressed(a.out, **out, metadata=json.dumps(report, allow_nan=False))
    Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
