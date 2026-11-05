"""Certify full collection-generic input rows without time folding or selection."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import uproot


EVENT = ("run", "subrun", "event")
IDENTITY = EVENT + ("particle_key",)
KINEMATICS = ("x", "y", "z", "time", "px", "py", "pz")
HIT_FIELDS = KINEMATICS + ("pdgId", "particleId", "volumeCopy")
SUCCESS = re.compile(r"(?m)^Art has completed and will exit with status 0\.?\s*$")
TOKEN = re.compile(r"(?m)^(?:Hits for[^\r\n]*|[ \t]*trackId\b[^\r\n]*)")
HEADER = re.compile(
    r"^Hits for BeamFilter:Beam in (?:run: ([0-9]+) subRun: ([0-9]+) event: ([0-9]+)"
    r"|([0-9]+):([0-9]+):([0-9]+)): \(([0-9]+)\)$", re.M)
TRACK = re.compile(
    r"^[ \t]*trackId: +\( *([0-9]+(?::[0-9]+)?), *([0-9]+)\)"
    r" +volumeId: +([0-9]+) +pdgId: +(-?[0-9]+)(?: +[^\r\n]*)?$", re.M)


def _ids(columns, names=IDENTITY):
    """Build exact structured identities for vectorized event-qualified joins."""
    out = np.empty(len(columns[names[0]]), dtype=[(k, "<i8") for k in names])
    for k in names:
        out[k] = _integers(columns[k], k)
    return out


def _integers(value, name, *, float_ok=False):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in "iuf":
        raise ValueError(f"{name}: expected integer vector")
    if a.dtype.kind == "f":
        if not float_ok or np.any(~np.isfinite(a)) or np.any(a != np.floor(a)) or np.any(np.abs(a) >= 2**24):
            raise ValueError(f"{name}: invalid or ambiguous Float_t integer identity")
    if np.any(a >= 2**63) or (a.dtype.kind != "u" and np.any(a < -(2**63))):
        raise ValueError(f"{name}: integer outside int64 range")
    return a.astype(np.int64)


def parse_event_index(logtext):
    """Parse exact BeamFilter:Beam headers and track keys, rejecting unknown formats."""
    tokens = np.asarray(TOKEN.findall(logtext), dtype=str)
    is_header = np.char.startswith(tokens, "Hits for")
    headers = np.asarray(HEADER.findall("\n".join(tokens[is_header])), dtype=str).reshape(-1, 7)
    if not len(headers) or len(headers) != int(is_header.sum()):
        raise ValueError("Unknown EventID format or collection label; expected BeamFilter:Beam")
    triples = np.where(headers[:, :3] != "", headers[:, :3], headers[:, 3:6]).astype(np.int64)
    counts = headers[:, 6].astype(np.int64)
    events = {k: triples[:, i] for i, k in enumerate(EVENT)}
    event_ids = _ids(events, EVENT)
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("Duplicate event header")
    tracks = np.asarray(TRACK.findall("\n".join(tokens[~is_header])), dtype=str).reshape(-1, 4)
    if len(tracks) != int((~is_header).sum()):
        raise ValueError("Malformed trackId line or missing exact volumeId/pdgId labels")
    owner = (np.cumsum(is_header) - 1)[~is_header]
    if np.any(owner < 0) or not np.array_equal(np.bincount(owner, minlength=len(counts)), counts):
        raise ValueError("Per-event track line count does not match declared hits")
    if len(tracks) != int(counts.sum()):
        raise ValueError("Total track line count does not match declared hits")
    out = {k: np.repeat(triples[:, i], counts) for i, k in enumerate(EVENT)}
    out.update(particle_key=tracks[:, 1].astype(np.int64), volume_copy=tracks[:, 2].astype(np.int64),
               pdg=tracks[:, 3].astype(np.int64), product_id=tracks[:, 0],
               event_headers=event_ids, event_hit_counts=counts)
    out["event_particle_id"] = _ids(out)
    return out


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree(root, name):
    tree = root[name]
    return tree, {"name": name, "object_path": tree.object_path, "rows": int(tree.num_entries)}


def _columns(tree, names):
    return {k: tree[k].array(library="np") for k in names}


def _hits(tree):
    keys = set(tree.keys(recursive=True))
    if "hits" in keys:
        try:
            a = tree["hits"].array(library="np")
        except (ValueError, TypeError, NotImplementedError):
            a = None
        if a is not None and a.dtype.names and set(HIT_FIELDS).issubset(a.dtype.names):
            return {k: a[k] for k in HIT_FIELDS}, "hits leaf-list"
    out, labels = {}, {}
    for k in HIT_FIELDS:
        candidates = [name for name in (f"hits.{k}", f"hits/{k}") if name in keys]
        if len(candidates) != 1:
            raise ValueError(f"Missing or ambiguous hits field {k}; available branches: {sorted(keys)}")
        labels[k] = candidates[0]
        out[k] = tree[candidates[0]].array(library="np")
    return out, labels


def _particle_time(tree, hits):
    keys = list(tree.keys(recursive=True))
    names = [k for k in keys if k in ("particle.time_", "particle/time_", "particle/particle.time_")]
    info = {"available_branches": [k for k in keys if k.startswith("particle")],
            "state_vector_mapping": "Not used; no unverified state component mapping"}
    if not names:
        info.update(available=False, reason="No recognized split ParticleState time branch")
        return hits["time"].copy(), info
    if len(names) != 1:
        raise ValueError("Ambiguous ParticleState time branches")
    t = tree[names[0]].array(library="np")
    if t.shape != hits["time"].shape or t.dtype.kind != "f" or t.dtype.itemsize != 8 or np.any(~np.isfinite(t)):
        raise ValueError("ParticleState time must be a finite aligned Double_t vector")
    if not np.array_equal(t.astype(np.float32), hits["time"]):
        raise ValueError("ParticleState double time does not round to hits Float_t time")
    info.update(available=True, branch=names[0], dtype=t.dtype.name)
    return t, info


def _join(source, target):
    order = np.argsort(source, kind="stable")
    sorted_source = source[order]
    if len(np.unique(sorted_source)) != len(source):
        raise ValueError("Duplicate ancestry event-particle identity")
    indices = np.searchsorted(sorted_source, target)
    if not len(source) and len(target):
        raise ValueError("Missing ancestry event-particle identity")
    if np.any(indices >= len(source)) or (len(target) and np.any(sorted_source[indices] != target)):
        raise ValueError("Missing ancestry event-particle identity")
    return order[indices]


def _ancestry(root, index):
    subruns = np.unique(index["event_headers"]["subrun"])
    if len(subruns) != 1:
        raise ValueError("ancestry/ntpssp lacks subrun; exactly one printer-certified subrun is required")
    tree, info = _tree(root, "ancestry/ntpssp")
    columns = _columns(tree, ("run", "evt", "trk", "pk", "pdg", "sgt", "scc", "rcc", "egt"))
    a = {k: _integers(columns[k], k, float_ok=True) for k in ("run", "evt", "trk", "pk", "pdg", "scc", "rcc")}
    source = dict(run=a["run"], subrun=np.full(len(a["run"]), subruns[0]), event=a["evt"], particle_key=a["trk"])
    found = _join(_ids(source), index["event_particle_id"])
    if not np.array_equal(a["pdg"][found], index["pdg"]):
        raise ValueError("Ancestry pdg mismatch")
    for name in ("sgt", "egt"):
        if np.any(~np.isfinite(columns[name])):
            raise ValueError(f"Nonfinite ancestry {name}")
    process, inverse = np.unique(np.column_stack((a["scc"][found], a["rcc"][found])), axis=0, return_inverse=True)
    info.update(joined_rows=len(found), certified_subrun=int(subruns[0]),
                process_codes={"scc": process[:, 0].tolist(), "rcc": process[:, 1].tolist(),
                               "count": np.bincount(inverse, minlength=len(process)).tolist()})
    payload = {f"ancestry_{k}": columns[k][found] for k in ("pk", "sgt", "scc", "rcc", "egt")}
    payload["ancestry_row"] = found
    return payload, info


def crosscheck(generic, legacy, log, states=None, *, include_ancestry=True):
    """Reject any row mismatch before exporting unchanged, event-qualified states."""
    paths = {"generic": Path(generic), "legacy": Path(legacy), "log": Path(log)}
    hashes = {k: _sha256(v) for k, v in paths.items()}
    text = paths["log"].read_text()
    if not SUCCESS.search(text):
        raise ValueError("Missing successful completed job marker in log")
    index = parse_event_index(text)
    with uproot.open(paths["generic"]) as root:
        tree, generic_meta = _tree(root, "input/nt")
        hits, labels = _hits(tree)
        generic_meta["hits_layout"] = labels
        n = len(index["particle_key"])
        if int(tree.num_entries) != n:
            raise ValueError("Generic row count differs from printer hit count")
        for k in KINEMATICS:
            a = hits[k]
            if a.shape != (n,) or a.dtype.kind != "f" or a.dtype.itemsize != 4 or np.any(~np.isfinite(a)):
                raise ValueError(f"Generic {k} must be finite Float_t values")
        integers = {k: _integers(hits[k], k) for k in ("pdgId", "particleId", "volumeCopy")}
        for k, name in (("particleId", "particle_key"), ("volumeCopy", "volume_copy"), ("pdgId", "pdg")):
            if not np.array_equal(integers[k], index[name]):
                raise ValueError(f"Generic/printer {k} mismatch")
        time_ns, particle_meta = _particle_time(tree, hits)
        if include_ancestry:
            ancestry, ancestry_meta = _ancestry(root, index)
            ancestry_meta["artifact"] = "generic"
        else:
            ancestry, ancestry_meta = {}, dict(certified=False, reason="Explicitly omitted: ntpssp lacks subrun identity")
    with uproot.open(paths["legacy"]) as root:
        tree, legacy_meta = _tree(root, "input/ntvd")
        old = _columns(tree, ("run", "subrun", "evt", "trk", "sid", "pdg") + KINEMATICS)
        if int(tree.num_entries) != n:
            raise ValueError("Legacy row count differs from printer hit count")
        for k, name in (("run", "run"), ("subrun", "subrun"), ("evt", "event"),
                        ("trk", "particle_key"), ("sid", "volume_copy"), ("pdg", "pdg")):
            if not np.array_equal(_integers(old[k], k, float_ok=True), index[name]):
                raise ValueError(f"Legacy/printer {k} mismatch")
        for k in KINEMATICS:
            a = old[k]
            if a.dtype.kind != "f" or a.dtype.itemsize != 4 or not np.array_equal(a, hits[k]):
                raise ValueError(f"Exact Float_t kinematic/time mismatch: {k}")
    if any(_sha256(v) != hashes[k] for k, v in paths.items()):
        raise ValueError("Input artifact changed during cross-check")
    float_spacing = np.abs(np.spacing(hits["time"]))
    spacing = np.abs(np.spacing(time_ns))
    payload = {k: index[k] for k in IDENTITY + ("volume_copy", "pdg", "product_id", "event_particle_id")}
    payload.update(x=np.column_stack([hits[k] for k in ("x", "y", "z")]),
                   p=np.column_stack([hits[k] for k in ("px", "py", "pz")]), time_ns=time_ns,
                   time_float32_ns=hits["time"], time_storage_spacing_ns=spacing,
                   time_float32_spacing_ns=float_spacing, input_row=np.arange(n, dtype=np.int64),
                   event_headers=index["event_headers"], event_hit_counts=index["event_hit_counts"], **ancestry)
    report = dict(schema="flash-step-input-v1", certified=True, rows=n, events=len(index["event_headers"]),
                  completed_job_marker="Art has completed and will exit with status 0",
                  artifacts={k: {"sha256": hashes[k], "bytes": v.stat().st_size} for k, v in paths.items()},
                  generic_tree=generic_meta, legacy_tree=legacy_meta, ancestry_tree=ancestry_meta,
                  particle_state=particle_meta, time_storage_dtype=time_ns.dtype.name,
                  precision_warning="Float_t diagnostic time may not resolve pulse phase; no folding or precision recovery from Float_t",
                  max_time_storage_spacing_ns=float(spacing.max()) if n else None,
                  max_time_float32_spacing_ns=float(float_spacing.max()) if n else None,
                  scope="Full input row identity and exact stored Float_t equality; no normalization, cuts, time folding or geometry certification")
    if states is not None:
        target = Path(states)
        if target.resolve() in [v.resolve() for v in paths.values()]:
            raise ValueError("States output must not overwrite an input artifact")
        with target.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        report["artifacts"]["states"] = {"sha256": _sha256(target), "bytes": target.stat().st_size, "rows": n}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generic", "legacy", "log", "out", "states"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--without-ancestry", action="store_true", help="Explicitly omit ntpssp ancestry, which lacks subrun identity")
    args = parser.parse_args(argv)
    outputs = (args.out.resolve(), args.states.resolve())
    if len(set(outputs)) != 2 or any(p.resolve() in outputs for p in (args.generic, args.legacy, args.log)):
        parser.error("Output artifacts must be distinct from each other and all inputs")
    report = crosscheck(args.generic, args.legacy, args.log, args.states, include_ancestry=not args.without_ancestry)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
