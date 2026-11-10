"""Neighbour-partition check: one recorded component beside an unrecorded touching neighbour; diagnostic only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import capture, capture_accounting
from .saved_g4 import frame, match, require, unique

FACES = {f"{sign}{axis}": (index, 1. if sign == "+" else -1.) for index, axis in enumerate("xyz") for sign in "+-"}


def _component(placements, volume_ids):
    """True where a placement path passes through one of the given per-row volume IDs."""
    return np.array([f"/{vid}:" in str(path) for path, vid in zip(placements, volume_ids)], dtype=bool)


def report(tables, links, expected_events, *, neighbour="NeighbourEnvelope", face="+x", atol=1e-7):
    """Accounting plus partition: nothing recorded in the neighbour, and every crossing into it on the shared face."""
    require(face in FACES, f"face must be one of {sorted(FACES)}")
    accounting = capture_accounting.report(tables, links, expected_events)
    e, s, v = (tables[name] for name in ("events", "steps", "volumes"))
    events = unique(e, capture.EVENT, "event").assign(_event=np.arange(len(e["event"])))
    ve = match(frame(v, capture.EVENT), events, capture.EVENT, "volume")["_event"].to_numpy()
    nb = v["name"] == neighbour
    require(np.array_equal(np.bincount(ve[nb], minlength=len(events)), np.ones(len(events), int)),
            "expected one neighbour envelope per event")
    env = v["name"] == e["envelope"][ve]
    env_id, nb_id = np.zeros(len(events), np.int64), np.zeros(len(events), np.int64)
    env_id[ve[env]], nb_id[ve[nb]] = v["id"][env], v["id"][nb]
    se = match(frame(s, capture.EVENT), events, capture.EVENT, "step")["_event"].to_numpy()
    pre_in_neighbour = _component(s["pre_placement"], nb_id[se])
    post_in_neighbour = _component(s["post_placement"], nb_id[se])
    pre_in_component = _component(s["pre_placement"], env_id[se])
    inside = s["preInside"].astype(bool)
    require(not pre_in_neighbour.any(), "recorded step starts in the neighbour")
    require(np.array_equal(pre_in_component, inside), "inside flag disagrees with the component subtree")
    into = post_in_neighbour
    require(np.all(~into | s["firstExit"].astype(bool)), "crossing into the neighbour is not a first exit")
    axis, sign = FACES[face]
    half = e["halfSize"][se]
    local = capture.vectors(s, "post_l", "xyz")
    on_face = np.abs(local[:, axis] - sign * half[:, axis]) <= atol
    require(np.all(on_face[into]), "crossing into the neighbour is off the shared face")
    exits = s["firstExit"].astype(bool)
    return {"diagnostic_only": True, "bank_admissible": False, "partition_passed": True,
            "neighbour": neighbour, "face": face, "accounting": accounting,
            "first_exits": int(exits.sum()), "first_exits_into_neighbour": int(into.sum()),
            "first_exits_elsewhere": int((exits & ~into).sum()),
            "steps_starting_in_neighbour": int(pre_in_neighbour.sum()),
            "component_deposit_MeV": float(np.sum(s["deposit"][inside]))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--directory", default="dump")
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--neighbour", default="NeighbourEnvelope")
    parser.add_argument("--face", default="+x")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "output already exists; refusing to clobber")
    tables = capture.read(args.input, args.directory)
    links = capture.read_links(args.input, args.directory)
    result = report(tables, links, args.expected_events, neighbour=args.neighbour, face=args.face)
    result["input"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
