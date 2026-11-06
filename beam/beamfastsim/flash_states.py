"""Extract DS e-/e+ states from stock StepPointMCDumper output and event headers.

No Geant4 is run and no time cuts, pulse offsets or momentum selections are made.
StepPointsPrinter supplies integer event/particle identities; ParticleState supplies
Double_t time. All event headers, including zero-hit events, are saved separately.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import uproot

HEADER = re.compile(r"^Hits for BeamFilter:Beam in (?:run: (\d+) subRun: (\d+) event: (\d+)|"
                    r"(\d+):(\d+):(\d+)): \((\d+)\)$", re.M)
TOKEN = re.compile(r"(?m)^(?:Hits for[^\r\n]*|[ \t]*trackId\b[^\r\n]*)")
TRACK = re.compile(r"^[ \t]*trackId: +\( *([0-9]+(?::[0-9]+)?), *([0-9]+)\)"
                   r" +volumeId: +([0-9]+) +pdgId: +(-?[0-9]+)(?: +[^\r\n]*)?$", re.M)
HIT_FIELDS = ("x", "y", "z", "time", "px", "py", "pz", "pdgId", "particleId", "volumeCopy")


def event_index(text):
    """Integer identities aligned to each printed step and all persisted input events."""
    tokens = np.asarray(TOKEN.findall(text), dtype=str)
    is_header = np.char.startswith(tokens, "Hits for")
    headers = np.asarray(HEADER.findall("\n".join(tokens[is_header])), dtype=str).reshape(-1, 7)
    if not len(headers) or len(headers) != is_header.sum():
        raise ValueError("Unrecognized BeamFilter:Beam event headers")
    triples = np.where(headers[:, :3] != "", headers[:, :3], headers[:, 3:6]).astype(np.int64)
    counts = headers[:, 6].astype(np.int64)
    if len(np.unique(triples, axis=0)) != len(triples):
        raise ValueError("Duplicate source event identity")
    tracks = np.asarray(TRACK.findall("\n".join(tokens[~is_header])), dtype=str).reshape(-1, 4)
    owner = (np.cumsum(is_header) - 1)[~is_header]
    if (len(tracks) != (~is_header).sum() or np.any(owner < 0)
            or not np.array_equal(np.bincount(owner, minlength=len(counts)), counts)):
        raise ValueError("Step identities do not align with event headers")
    out = dict(run=np.repeat(triples[:, 0], counts), subrun=np.repeat(triples[:, 1], counts),
               event=np.repeat(triples[:, 2], counts), particle_key=tracks[:, 1].astype(np.int64),
               pdg=tracks[:, 3].astype(np.int64), volume_copy=tracks[:, 2].astype(np.int64))
    return out, triples


def read_hits(tree):
    """Stock leaf-list layouts; no guessed branch or particle-state coordinate mapping."""
    keys = set(tree.keys(recursive=True))
    if "hits" in keys:
        data = tree["hits"].array(library="np")
        if data.dtype.names and set(HIT_FIELDS).issubset(data.dtype.names):
            return {k: data[k] for k in HIT_FIELDS}
    out = {}
    for k in HIT_FIELDS:
        candidates = [s for s in (f"hits.{k}", f"hits/{k}") if s in keys]
        if len(candidates) != 1:
            raise ValueError(f"Unknown stock hit field {k}")
        out[k] = tree[candidates[0]].array(library="np")
    return out


def extract(root_path, log_path, out):
    """Write state and full event-index NPZs plus sample counts; highest ROOT tree cycle."""
    index, events = event_index(Path(log_path).read_text())
    with uproot.open(root_path) as root:
        tree = root["input/nt"]
        hits = read_hits(tree)
        for k, field in (("particle_key", "particleId"), ("pdg", "pdgId"), ("volume_copy", "volumeCopy")):
            if not np.array_equal(index[k], hits[field]):
                raise ValueError(f"StepPointMCDumper/printer mismatch: {k}")
        keys = set(tree.keys(recursive=True))
        branches = [k for k in ("particle.time_", "particle/time_", "particle/particle.time_") if k in keys]
        if len(branches) != 1:
            raise ValueError("A unique Double_t particle time branch is required")
        time_ns = tree[branches[0]].array(library="np")
        if (time_ns.dtype.itemsize != 8 or not np.isfinite(time_ns).all()
                or not np.array_equal(time_ns.astype(hits["time"].dtype), hits["time"])):
            raise ValueError("Particle time does not agree with stock hit time")
    select = np.isin(index["pdg"], (11, -11))
    source = {k: v[select] for k, v in index.items() if k != "volume_copy"}
    source.update(x=np.column_stack([hits[k] for k in ("x", "y", "z")])[select].astype(float),
                  p=np.column_stack([hits[k] for k in ("px", "py", "pz")])[select].astype(float),
                  time_ns=time_ns[select])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **source)
    np.savez_compressed(out.with_name(out.stem + "_events.npz"), event=events)
    report = dict(input=str(root_path), source_events=len(events), boundary_rows=len(select),
                  charged_rows=int(select.sum()), other_species_rows=int((~select).sum()),
                  species={str(code): int(np.count_nonzero(index["pdg"] == code)) for code in np.unique(index["pdg"])},
                  time_max_ns=float(time_ns.max()) if len(time_ns) else None,
                  source_time_cut=False, scope="Stored float kinematics and double time, exact integer event identity")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    extract(a.root, a.log, a.out)


if __name__ == "__main__":
    main()
