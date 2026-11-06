"""Write an explicit inclusive-G4 shard manifest; no transport is launched here.

Designs are labels plus frozen geometry paths, never board-ID lookups. Shards
partition one source event pool without wrapping. All designs see the same event
ranges, with distinct positive seeds. Geometric/physics admission is separate.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def manifest(designs, *, events, shards, skip, pool_events, seed_base, source):
    """Return tab-separated rows: point, geometry, source, skip, events, seed."""
    for name, value, minimum in (("events", events, 1), ("shards", shards, 1),
                                 ("skip", skip, 0), ("pool_events", pool_events, 1), ("seed_base", seed_base, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if source not in ("ts3", "ds") or not designs:
        raise ValueError("source must be ts3/ds and at least one design is required")
    if skip + events * shards > pool_events:
        raise ValueError("shard ranges exceed source pool; wrapping is forbidden")
    if seed_base + len(designs) * shards - 1 >= 900_000_000:
        raise ValueError("seed range exceeds conservative positive CLHEP range")
    rows, labels = [], set()
    for point, geometry in designs:
        if not LABEL.fullmatch(point) or point in labels:
            raise ValueError("design labels must be unique safe filename identifiers")
        labels.add(point)
        if not str(geometry).startswith("/") or any(c.isspace() for c in str(geometry)):
            raise ValueError("geometry must be an absolute remote path without whitespace")
        for shard in range(shards):
            rows.append((point, str(geometry), source, skip + shard * events, events, seed_base + len(rows)))
    return "".join("\t".join(map(str, row)) + "\n" for row in rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", action="append", required=True, help="label=/absolute/remote/frozen_geometry.txt")
    p.add_argument("--events", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--skip", type=int, required=True)
    p.add_argument("--pool-events", type=int, required=True)
    p.add_argument("--seed-base", type=int, required=True)
    p.add_argument("--source", choices=("ds", "ts3"), required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    designs = [x.split("=", 1) for x in a.design]
    if any(len(x) != 2 for x in designs):
        p.error("each design must be label=path")
    text = manifest(designs, events=a.events, shards=a.shards, skip=a.skip, pool_events=a.pool_events,
                    seed_base=a.seed_base, source=a.source)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        f.write(text)
    note = dict(source=a.source, designs=designs, events_per_shard=a.events, shards_per_design=a.shards,
                jobs=len(designs)*a.shards, range_start_inclusive=a.skip,
                range_end_exclusive=a.skip+a.events*a.shards, pool_events=a.pool_events,
                unused_pool_tail=a.pool_events-a.skip-a.events*a.shards, seed_base=a.seed_base,
                scope="Manifest only; no submission or geometry/physics validation")
    a.out.with_suffix(".json").write_text(json.dumps(note, indent=2) + "\n")
    print(json.dumps(note, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
