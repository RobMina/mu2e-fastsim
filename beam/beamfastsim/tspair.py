"""Identity-only diagnostic of native TS1 input and TS1ToTS5 stage-2 ntpart.

Not transport, a proximity matcher, an efficiency or an exposure certificate.
A stage-2 mu2ePrimary (56) has a NEW track ID and names its previous-stage
particle as parent. Other retained particles require actual same-event ancestry.
Only retained mu/pi rows and first STORED VD hits are available to this audit.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import tsreference as ref

PRIMARY = 56
MAX_DEPTH = 128
LABELS = np.array(["pending", "resolved_direct_same_particle",
                   "resolved_decay_descendant", "resolved_other_descendant",
                   "copied_stage1", "missing_link", "missing_code", "unknown_code",
                   "pdg_conflict", "identity_conflict", "cycle", "depth_limit"])
DIRECT, DECAY, OTHER, COPIED, MISSING, NO_CODE, UNKNOWN, PDG, ID, CYCLE, DEPTH = range(1, 12)


def _keys(columns, names=ref.IDENTITY, track=None):
    """Structured int64 keys: never column-stack mixed floats or stringify IDs."""
    result = np.empty(len(columns[names[0]]), dtype=[(k, "<i8") for k in names])
    for k in names:
        result[k] = track if k == "trk" and track is not None else columns[k]
    return result


def _join(keys, queries):
    """Unique-key left join; -1 denotes absence, including an empty pool."""
    if not len(keys):
        return np.full(len(queries), -1, dtype=np.int64)
    order = np.argsort(keys)
    pos = np.searchsorted(keys[order], queries)
    safe = np.minimum(pos, len(keys) - 1)
    return np.where((pos < len(keys)) & (keys[order[safe]] == queries), order[safe], -1)


def _cohort(paths, tree, vds):
    columns, receipt = ref.read(paths, tree=tree, vds=vds)
    files = receipt["files"]
    if len({f["path"] for f in files}) != len(files) or len({f["sha256"] for f in files}) != len(files):
        raise ValueError("duplicate/byte-identical files within a cohort")
    # All species must have disjoint native event shards, not just disjoint particle IDs.
    import uproot
    parts = []
    for f in files:
        with uproot.open(f["path"]) as root:
            a = root[f["selected_key"]].arrays(list(ref.IDENTITY[:3]), library="np", how=dict)
        event = np.unique(_keys(a, ref.IDENTITY[:3]))
        part = np.empty(len(event), dtype=event.dtype.descr + [("ifile", "<i8")])
        for k in ref.IDENTITY[:3]:
            part[k] = event[k]
        part["ifile"] = f["ifile"]
        parts.append(part)
    events = np.concatenate(parts)
    event_keys = _keys(events, ref.IDENTITY[:3])
    if len(np.unique(event_keys)) != len(event_keys):
        raise ValueError("overlapping native event identities across files within a cohort")
    if len(np.unique(_keys(columns))) != len(columns["trk"]):
        raise ValueError("duplicate native particle identity within a cohort")
    return columns, receipt


def _resolve(out, roots):
    """Vectorized graph walk; each iteration advances an entire ancestry layer.

    Floyd cursors detect cycles without an N x depth history allocation. Root
    terminals deliberately cut the graph: upstream ancestry is not the question.
    """
    n = len(out["trk"])
    own = _join(_keys(roots), _keys(out))
    parent_root = _join(_keys(roots), _keys(out, track=out["parent_id"]))
    parent = _join(_keys(out), _keys(out, track=out["parent_id"]))
    primary = out["has_code"] & (out["code"] == PRIMARY)
    # Code 56 needs parent linkage; reused native IDs are unsafe without stage/ProductID.
    copied = (own >= 0) & out["has_code"] & ~primary
    reason = np.zeros(n, dtype=np.int8)
    terminal = np.full(n, -1, dtype=np.int64)
    reason[parent < 0] = MISSING
    # Unknown codes are not evidence for a physical production/decay edge.
    unknown = ~((out["code"] > 0) & (out["code"] <= 194)) | np.isin(out["code"], [76, 183, 184, 185])
    reason[unknown] = UNKNOWN
    reason[~out["has_code"]] = NO_CODE
    terminal[copied] = own[copied]
    reason[copied] = 0
    immediate = primary & (parent_root >= 0)
    terminal[immediate] = parent_root[immediate]
    reason[immediate] = 0

    # Exact event-scoped parent links and code-56 continuations must preserve species.
    linked = np.flatnonzero((parent >= 0) & ~copied)
    bad = np.zeros(n, dtype=bool)
    bad[linked] = out["parent_pdg"][linked] != out["pdg"][parent[linked]]
    bad[linked] |= primary[linked] & (out["pdg"][linked] != out["pdg"][parent[linked]])
    matched = np.flatnonzero(parent_root >= 0)
    bad[matched] |= out["parent_pdg"][matched] != roots["pdg"][parent_root[matched]]
    bad[matched] |= primary[matched] & (out["pdg"][matched] != roots["pdg"][parent_root[matched]])
    matched = np.flatnonzero(own >= 0)
    bad[matched] |= out["pdg"][matched] != roots["pdg"][own[matched]]
    reason[bad] = PDG
    conflict = (own >= 0) & primary  # a NEW continuation cannot reuse a root ID
    for k in ("parent_id", "parent_pdg", "code"):
        compare = copied[matched]
        if k == "code":
            compare &= roots["has_code"][own[matched]]
        conflict[matched] |= compare & (out[k][matched] != roots[k][own[matched]])
    reason[conflict] = ID
    terminal[reason != 0] = -1

    # A known non-primary process is a physical edge, not a continuation.
    physical = ~primary & ~copied & out["has_code"] & ~unknown
    decay = physical & np.isin(out["code"], [14, 111])
    # Charged-pion decay to a muon preserves electric charge, reversing the PDG sign.
    check = np.flatnonzero(decay & (parent >= 0))
    valid_decay = ((np.abs(out["pdg"][check]) == 13)
                   & (np.abs(out["pdg"][parent[check]]) == 211)
                   & (np.sign(out["pdg"][check]) == -np.sign(out["pdg"][parent[check]])))
    reason[check[~valid_decay]] = PDG
    terminal[reason != 0] = -1

    following = parent.copy()
    following[(terminal >= 0) | (reason != 0)] = -1
    result = np.full(n, -1, dtype=np.int64)
    status = np.zeros(n, dtype=np.int8)
    cursor = np.arange(n, dtype=np.int64)
    slow, fast = cursor.copy(), cursor.copy()
    has_process, has_decay = np.zeros(n, bool), np.zeros(n, bool)

    def advance(index):
        value = np.full(n, -1, dtype=np.int64)
        valid = index >= 0
        value[valid] = following[index[valid]]
        return value

    for depth in range(MAX_DEPTH + 1):
        active = np.flatnonzero(status == 0)
        if not len(active):
            break
        here = cursor[active]
        has_process[active] |= physical[here]
        has_decay[active] |= decay[here]
        failed = reason[here] != 0
        status[active[failed]] = reason[here[failed]]
        # A code-56 terminal consumes its final parent edge; copied roots consume none.
        over = (depth == MAX_DEPTH) & ~copied[here] & ~failed
        status[active[over]] = DEPTH
        done = (terminal[here] >= 0) & ~failed & ~over
        target = active[done]
        result[target] = terminal[here[done]]
        status[target] = np.where(has_process[target], np.where(has_decay[target], DECAY, OTHER), DIRECT)
        status[target[copied[target]]] = COPIED
        slow, fast = advance(slow), advance(advance(fast))
        loop = (status == 0) & (slow >= 0) & (slow == fast)
        status[loop] = CYCLE
        pending = status == 0
        cursor[pending] = following[cursor[pending]]
    status[status == 0] = DEPTH
    return result, status, primary, parent_root


def _counts(pdg, mask):
    return {str(p): int(np.count_nonzero(mask & (pdg == p))) for p in ref.PDGS}


def audit(input_paths, output_paths, tree="mutree/ntpart", entrance_vd=1, exit_vd=8):
    """Return a JSON-safe report. Unresolved ancestry fails the identity gate.

    Malformed/ambiguous identity and overlapping cohort event shards raise
    ValueError. Missing links, cycles, code/PDG conflicts remain explicit report
    outcomes, never guessed matches. Input and output may share paths/hashes;
    overlap is forbidden only WITHIN each native cohort.
    """
    vd = ref._integers([entrance_vd, exit_vd], "VD IDs")
    if not np.all(np.isin(vd, range(1, 9))):
        raise ValueError("VD IDs must be in [1, 8]")
    vds = np.unique(vd).tolist()
    incoming, input_receipt = _cohort(list(input_paths), tree, vds)
    out, output_receipt = _cohort(list(output_paths), tree, vds)
    selected = incoming[f"vd{int(vd[0])}_hit"]
    roots = {k: a[selected] for k, a in incoming.items()}
    root_index, status, primary, parent_root = _resolve(out, roots)
    exits = out[f"vd{int(vd[1])}_hit"]
    resolved = np.isin(status, [DIRECT, DECAY, OTHER])
    matched_exit = exits & resolved
    multiplicity = np.bincount(root_index[matched_exit], minlength=len(roots["trk"]))
    continued = np.bincount(root_index[primary & resolved], minlength=len(roots["trk"]))
    values, counts = np.unique(multiplicity, return_counts=True)
    relevant = primary | exits
    failures = (status >= MISSING) & relevant
    integrity = np.isin(status, [PDG, ID, CYCLE, DEPTH])
    by_status = {str(label): int(np.count_nonzero(exits & (status == i)))
                 for i, label in enumerate(LABELS) if i}
    result = dict(
        schema="ts-native-pair-audit-v1", gate=bool(not np.any(failures | integrity)),
        purpose="Diagnostic native TS1-to-TS1ToTS5 identity audit, not fast transport",
        tree=tree, entrance_vd=int(vd[0]), exit_vd=int(vd[1]), max_ancestry_edges=MAX_DEPTH,
        input=input_receipt, output=output_receipt,
        identity=list(ref.IDENTITY), parent_scope="Exact same run/subrun/evt, parent_id -> trk",
        roots={k: roots[k].tolist() for k in ref.IDENTITY + ("pdg",)},
        root_exit_multiplicity=multiplicity.tolist(), root_continuation_multiplicity=continued.tolist(),
        exit_multiplicity_histogram={str(v): int(c) for v, c in zip(values, counts)},
        counts=dict(
            roots=_counts(roots["pdg"], np.ones(len(roots["trk"]), bool)),
            exits=_counts(out["pdg"], exits),
            native_continued_tracks=_counts(out["pdg"], primary & resolved),
            code56_parent_root_exact=_counts(out["pdg"], primary & (parent_root >= 0) & resolved),
            resolved_exit_root_ids=_counts(roots["pdg"], multiplicity > 0),
            resolved_exit_root_ids_total=int(np.count_nonzero(multiplicity)),
            resolved_exit_tracks=int(np.count_nonzero(matched_exit)),
            copied_stage1_exit_hits=int(np.count_nonzero(exits & (status == COPIED))),
            exit_status=by_status,
            unresolved_relevant_rows=int(np.count_nonzero(failures)),
            pool_issues={str(LABELS[i]): int(np.count_nonzero(status == i))
                         for i in range(MISSING, DEPTH + 1)}),
        exit_mapping={**{k: out[k][exits].tolist() for k in ref.IDENTITY + ("pdg", "parent_id", "parent_pdg", "code")},
                      "root_index": root_index[exits].tolist(), "status": LABELS[status[exits]].tolist()},
        limits=["ReadVirtualDetector first STORED hit is not certified earliest/forward crossing",
                "collection ProductID, simStage and the full charged population are lost",
                "Retained mu/pi rows only; missing parents may be outside this population",
                "Code 56 alone is not proof of a continuation; parent identity and PDG must agree",
                "Copied stage-1 hits are not new stage-2 continuation endpoints",
                "No time/distance matching, efficiencies, transmission or per-POT rates",
                "Cannot certify inclusive Run1A/Run1B provenance or exposure",
                "Decay-descendant means ancestry contains a pion-to-muon decay edge; additional nondecay production may also occur",
                "Known code mapping is a bounded source-level classification (0..194, reserved rejected), not historical binary certification",
                "Known non-decay-only process chains are reported separately, not called decay",
                "gate certifies only represented identity links, not population completeness or physics"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, dest="inputs")
    parser.add_argument("--output", nargs="+", required=True, dest="outputs", help="native stage-2 ROOT files")
    parser.add_argument("--tree", default="mutree/ntpart")
    parser.add_argument("--entrance-vd", type=int, default=1)
    parser.add_argument("--exit-vd", type=int, default=8)
    parser.add_argument("--out", required=True, help="audit JSON destination")
    args = parser.parse_args(argv)
    destination = Path(args.out)
    sources = [Path(p) for p in args.inputs + args.outputs]
    if destination.suffix.lower() != ".json":
        parser.error("--out must end in .json")
    if any(destination.resolve() == p.resolve() or
           (destination.exists() and p.exists() and destination.samefile(p)) for p in sources):
        parser.error("--out must not overwrite/alias an input ROOT file")
    try:
        report = audit(args.inputs, args.outputs, args.tree, args.entrance_vd, args.exit_vd)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0 if report["gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
