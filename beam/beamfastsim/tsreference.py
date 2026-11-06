"""Run1B TS1Beam -> TS1ToTS5s100 recorded Beam StepPoint reference.

Default input: ReadVirtualDetector nttvd from BeamFilter:Beam / TS5Filter:Beam.
These are pre-step positions/momenta/times (mm, MeV/c, ns), NOT SimParticle
creation or terminal states. All recorded mu/pi rows are used without position,
time, direction or momentum cuts. ntpart supplies exact identity/parent links.
Terminal-only selection is available explicitly for diagnostics, not as fallback.
"""
import argparse
import json
from pathlib import Path
import re
import time
import awkward as ak
import numpy as np
import uproot

IDS = ("run", "subrun", "evt", "trk")
SOURCE_Z = -4003.98
TOL = 0.01


def _tree(f, name):
    keys = [k for k in f.keys(recursive=True) if k.split(";")[0].split("/")[-1] == name]
    if not keys or len({k.rsplit(";", 1)[0] for k in keys}) != 1:
        raise ValueError(f"missing/ambiguous {name}: {keys}")
    key = max(keys, key=lambda k: int(k.rsplit(";", 1)[1]))
    return f[key], {"tree": key, "cycles": keys}


def read_table(path, tree="ntpart"):
    """Scalar arrays and schema from the highest cycle of a unique stock tree."""
    with uproot.open(path) as f:
        t, meta = _tree(f, tree)
        names = [k for k in t.keys() if not any(s in t[k].typename for s in ("[", "vector<"))]
        a = t.arrays(names, library="ak")
        return {k: ak.to_numpy(a[k]) for k in names}, {**meta,
                "schema": {k: t[k].typename for k in t.keys()}}


def ids(a, names=IDS):
    """Exact IDs; floating TNtuple identifiers must be integral and |ID| < 2**24."""
    columns = []
    for k in names:
        v = np.asarray(a[k])
        if not np.all(np.isfinite(v)) or np.any(v != np.floor(v)):
            raise ValueError(f"nonintegral/nonfinite IDs: {k}")
        if np.issubdtype(v.dtype, np.floating) and np.any(np.abs(v) >= 2**24):
            raise ValueError(f"float ID outside exact TNtuple precision range (<2**24): {k}")
        if np.any(v < -(2**63)) or np.any(v >= 2**63):
            raise ValueError(f"ID outside int64 range: {k}")
        columns.append(v.astype(np.int64))
    return np.column_stack(columns)


def _keys(x):
    x = np.ascontiguousarray(x, dtype=np.int64)
    return x.view(np.dtype([(f"k{i}", "<i8") for i in range(x.shape[1])])).ravel()


def lookup(index, query):
    """Exact structured-key join, -1 for absent keys; duplicate index is an error."""
    k, q = _keys(index), _keys(query)
    order = np.argsort(k)
    k = k[order]
    if np.any(k[1:] == k[:-1]):
        raise ValueError("duplicate particle/event keys")
    if not len(k):
        return np.full(len(q), -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(k, q), len(k) - 1)
    return np.where(k[pos] == q, order[pos], -1)


def subset(a, take):
    return {k: v[take] for k, v in a.items()}


def state(a, prefix=""):
    return {prefix + "x": np.column_stack([a[k] for k in ("x", "y", "z")]).astype(float),
            prefix + "p": np.column_stack([a[k] for k in ("px", "py", "pz")]).astype(float),
            prefix + "t": a["time"].astype(float), prefix + "pdg": ids(a, ("pdg",))[:, 0]}


def source_mask(a):
    """Legacy terminal diagnostic only."""
    return ((a["codestop"] == 54) & (np.abs(a["zstop"] - SOURCE_Z) <= TOL) & (a["pzstop"] > 0))


def endpoint_mask(a):
    """Legacy terminal diagnostic only; not a recorded Beam selection."""
    return (a["codestop"] == 54) & (a["zstop"] > 2929) & (a["xstop"] < 0)


def terminal_beam(a, mask):
    b = subset(a, mask)
    return {**{k: b[k] for k in IDS + ("pdg",)}, "time": b["tstop"],
            **{k: b[k + "stop"] for k in ("x", "y", "z", "px", "py", "pz")}}


def match_beam(beam, particles):
    key = ids(beam)
    lookup(key, key)
    keep = np.isin(np.abs(beam["pdg"]), [13, 211])
    beam = subset(beam, keep)
    ids(beam, ("pdg",))
    rows = lookup(ids(particles), key[keep])
    if np.any(rows < 0):
        raise ValueError("recorded Beam identity unmatched in ntpart")
    if np.any(beam["pdg"] != particles["pdg"][rows]):
        raise ValueError("recorded Beam/ntpart PDG mismatch")
    return beam, rows


def summary(a):
    fields = ("xstop", "ystop", "zstop", "pstop", "pzstop") if "xstop" in a else ("x", "y", "z", "px", "py", "pz", "time")
    return {"rows": len(a["trk"]), "pdg": np.column_stack(np.unique(a["pdg"], return_counts=True)).tolist(),
            "overall_ranges": {k: [float(np.min(a[k])), float(np.max(a[k]))] if len(a[k]) else [] for k in fields}}


def build(source, downstream, event_id=None, *, source_beam=None, downstream_beam=None, terminal_only=False):
    """Align independent recorded source rows with exact-ancestry DS endpoints.

    Missing endpoints are stock zeros (false reached, NaN state), including when
    FilterG4Out retains only survivors/ancestors. Nonendpoint ancestry is diagnostic;
    endpoint ancestry must resolve uniquely through a restart to the source copy.
    """
    for a in (source, downstream):
        lookup(ids(a), ids(a))
        ids(a, ("parent_id", "pdg"))
    if terminal_only:
        source_beam = terminal_beam(source, source_mask(source))
        downstream_beam = terminal_beam(downstream, endpoint_mask(downstream))
    elif source_beam is None or downstream_beam is None:
        raise ValueError("recorded Beam nttvd required; use terminal_only=True only for diagnostics")
    s, si = match_beam(source_beam, source)
    e, end = match_beam(downstream_beam, downstream)
    sp, d = subset(source, si), downstream
    n, dk = len(si), ids(d)
    out = {"id": ids(s), **state(s), "shard": ids(s, ("subrun",))[:, 0]}
    observed = np.concatenate((ids(source, IDS[:3]), ids(source_beam, IDS[:3])))
    if event_id is not None:
        observed = np.concatenate((observed, ids(dict(zip(IDS[:3], np.asarray(event_id).T)), IDS[:3])))
    out["event_id"] = np.unique(observed, axis=0)
    copied = lookup(ids(source, IDS + ("parent_id",)), ids(d, IDS + ("parent_id",))) >= 0
    restart = np.flatnonzero(d["code"] == 56)
    target = np.unique(np.concatenate((end, restart, np.flatnonzero(~copied))))
    required = np.isin(target, end)
    parent_key = dk.copy(); parent_key[:, 3] = ids(d, ("parent_id",))[:, 0]
    parent = lookup(dk, parent_key)
    copy_to_source = lookup(ids(sp, IDS + ("parent_id",)), ids(d, IDS + ("parent_id",)))
    cur, root, depth = target.copy(), np.full(len(target), -1, dtype=np.int64), 0

    def reject(active, bad, message):
        if np.any(required[active[bad]]):
            raise ValueError(message)
        root[active[bad]] = -2
        return active[~bad]

    while np.any(root == -1):
        active = np.flatnonzero(root == -1)
        if depth > len(d):
            reject(active, np.ones(len(active), dtype=bool), "cyclic ancestry")
            break
        active = reject(active, ~np.isin(d["code"][cur[active]], [14, 56]), "unsupported ancestry creation code")
        active = reject(active, parent[cur[active]] < 0, "unmatched ancestry parent keys")
        row, up = cur[active], parent[cur[active]]
        is_restart = d["code"][row] == 56
        same = d["pdg"][row] == d["pdg"][up]
        decay = ((np.abs(d["pdg"][row]) == 13) & (np.abs(d["pdg"][up]) == 211) & (d["pdg"][row] * d["pdg"][up] < 0))
        active = reject(active, ~np.where(is_restart, same, decay), "unsupported ancestry: expected same-species restart or charged pi->mu decay")
        up = parent[cur[active]]
        is_restart = d["code"][cur[active]] == 56
        ra, copied_rows = active[is_restart], up[is_restart]
        found = copy_to_source[copied_rows]
        valid = found >= 0
        reject(ra, ~valid, "missing source root: restart parent is not an exact selected source copy")
        ra, copied_rows, found = ra[valid], copied_rows[valid], found[valid]
        mismatch = (d["pdg"][copied_rows] != sp["pdg"][found]) | (d["code"][copied_rows] != sp["code"][found])
        reject(ra, mismatch, "source copy PDG/creation-code mismatch")
        root[ra[~mismatch]] = found[~mismatch]
        cur[active[~is_restart]] = up[~is_restart]
        depth += 1
    eroot = root[np.searchsorted(target, end)]
    u, cnt = np.unique(eroot, return_counts=True)
    if np.any(cnt > 1):
        raise ValueError(f"multiple DS endpoints for source IDs: {out['id'][u[cnt > 1]][:10].tolist()}")
    restart_roots = root[np.searchsorted(target, restart)]
    out["stock_restarted"] = np.zeros(n, dtype=bool)
    out["stock_restarted"][restart_roots[restart_roots >= 0]] = True
    out["stock_reached"] = np.zeros(n, dtype=bool)
    out["stock_reached"][eroot] = True
    for k, v in state(e, "stock_").items():
        if not np.all(np.isfinite(v)):
            raise ValueError(f"nonfinite endpoint {k}")
        out[k] = np.full((n,) + v.shape[1:], 0 if k == "stock_pdg" else np.nan, dtype=v.dtype)
        out[k][eroot] = v
    for k in ("x", "p", "t"):
        if not np.all(np.isfinite(out[k])):
            raise ValueError(f"nonfinite source {k}")
    out["stock_id"] = np.full((n, 4), -1, dtype=np.int64)
    out["stock_id"][eroot] = dk[end]
    occupied = np.unique(out["id"][:, :3], axis=0)
    out["parent_shard_ids"], out["parent_shard_event_counts"] = np.unique(out["event_id"][:, :2], axis=0, return_counts=True)
    report = {"source_rows": len(source["trk"]), "source_beam_rows": len(source_beam["trk"]),
              "downstream_beam_rows": len(downstream_beam["trk"]), "cohort": n, "stock_reached": len(end),
              "stock_not_reached": n - len(end), "parent_events_observed": len(out["event_id"]),
              "parent_events_total": None, "events_without_source": len(out["event_id"]) - len(occupied),
              "restarts": len(restart), "restarted_sources": int(out["stock_restarted"].sum()),
              "unmapped_restarts": int(np.sum(restart_roots < 0)), "max_ancestry_depth": depth,
              "unrelated_new_descendants": int(np.sum((root < 0) & ~copied[target])),
              "unmatched_endpoints": 0, "event_inventory_complete": False, "shard_event_inventory_complete": False,
              "source_selection": "terminal diagnostic: code54, Coll1_In z, pz>0" if terminal_only else "all recorded source Beam nttvd rows with |pdg|=13 or 211; no other cuts",
              "endpoint_selection": "terminal diagnostic: code54, z>2929, x<0" if terminal_only else "all recorded downstream Beam nttvd rows with |pdg|=13 or 211; exact ancestry",
              "state_kind": "terminal diagnostic" if terminal_only else "recorded pre-step Beam",
              "source": summary(s), "endpoints": summary(e),
              "limits": ["nttvd must be produced from the stated Beam StepPoint collections",
                         "missing endpoints are stock zeros under the established complete input schedule",
                         "event_id contains observed IDs only; art Events counts supply shard totals, not missing event IDs"]}
    return out, report


def read_events(path, a):
    """Union all available parent-table event IDs, including noncohort particles."""
    arrays = [ids(a, IDS[:3])]
    with uproot.open(path) as f:
        trees = {k.split(";")[0].split("/")[-1] for k in f.keys(recursive=True)}
    for tree in ("ntvd", "nttvd"):
        if tree in trees:
            v, _ = read_table(path, tree)
            arrays.append(ids(v, IDS[:3]))
    return np.unique(np.concatenate(arrays), axis=0)


def art_inventory(paths):
    """Count finished Events trees; filename .RUN_SUBRUN.art supplies shard IDs.

    No decoding or guessing of unsplit EventAuxiliary, and no fabricated event IDs.
    Each supplied file must represent a distinct source run/subrun shard.
    """
    shards, totals, files = [], [], []
    for path in paths:
        match = re.search(r"\.(\d+)_(\d+)\.art$", Path(path).name)
        if match is None:
            raise ValueError(f"source art filename lacks .RUN_SUBRUN.art: {path}")
        shards.append([int(v) for v in match.groups()])
        with uproot.open(path) as f:
            t, meta = _tree(f, "Events")
            totals.append(int(t.num_entries))
            files.append({"path": str(path), **meta, "events": totals[-1]})
    if not shards:
        raise ValueError("source art inventory requires at least one file")
    shard = np.asarray(shards, dtype=np.int64).reshape(-1, 2)
    lookup(shard, shard)
    order = np.lexsort((shard[:, 1], shard[:, 0]))
    return shard[order], np.asarray(totals, dtype=np.int64)[order], files


def read(source, downstream, source_art_paths=None, *, terminal_only=False):
    """Local ROOT API; recorded nttvd required unless terminal_only is explicit."""
    start = time.perf_counter()
    s, sm = read_table(source)
    d, dm = read_table(downstream)
    sb = db = None
    if not terminal_only:
        sb, sm["beam"] = read_table(source, "nttvd")
        db, dm["beam"] = read_table(downstream, "nttvd")
    arrays, report = build(s, d, read_events(source, s), source_beam=sb, downstream_beam=db, terminal_only=terminal_only)
    if source_art_paths is not None:
        shard, totals, files = art_inventory(source_art_paths)
        match = lookup(shard, arrays["parent_shard_ids"])
        if np.any(match < 0) or np.any(arrays["parent_shard_event_counts"] > totals[match]):
            raise ValueError("source art inventory does not cover observed source events")
        arrays.update(parent_shard_ids=shard, parent_shard_event_counts=totals)
        report.update(parent_events_total=int(totals.sum()), shard_event_inventory_complete=True,
                      events_without_source=int(totals.sum()) - len(np.unique(arrays["id"][:, :3], axis=0)), source_art_files=files)
    report.update(source_file=sm, downstream_file=dm, seconds=time.perf_counter() - start)
    return arrays, report


def save(arrays, path):
    np.savez_compressed(path, **arrays)


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspect", nargs="+")
    p.add_argument("--source"); p.add_argument("--downstream"); p.add_argument("--out")
    p.add_argument("--source-art", nargs="+", help="finished source art files, one run/subrun shard each")
    p.add_argument("--terminal-only", action="store_true", help="explicit terminal-state diagnostic instead of recorded Beam")
    args = p.parse_args()
    if args.inspect:
        report = {}
        for path in args.inspect:
            a, meta = read_table(path)
            report[path] = {**meta, **summary(a), "events_observed": len(read_events(path, a))}
    else:
        if not all((args.source, args.downstream, args.out)):
            p.error("require --source, --downstream, --out (or --inspect)")
        try:
            arrays, report = read(args.source, args.downstream, args.source_art, terminal_only=args.terminal_only)
            save(arrays, args.out)
        except ValueError as exc:
            print(json.dumps({"error": str(exc)}))
            raise SystemExit(1) from exc
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
