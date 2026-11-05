"""Build a CE library from the development half of saved numeric stock tables."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .export import read
from .library import Binning, build, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split-row", required=True, type=int)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    begin = time.perf_counter()
    entries, css, geometry, census, _ = read(args.input)
    if not 0 < args.split_row < census["events"]:
        parser.error("split-row must lie inside the source event range")
    development = entries["event_row"] < args.split_row
    library = build(entries, css, development=development, binning=Binning())
    library.update({"geometry_" + k: v for k, v in geometry.items()})
    metadata = dict(input=str(args.input), source_commit=args.source_commit, code_commit=args.code_commit,
                    split_row=args.split_row, split="source event row, not independent production shards",
                    census=census, development_entries=int(development.sum()),
                    heldout_entries=int((~development).sum()), development_css=int(library["offsets"][-1]),
                    empty_development_showers=int(np.sum(np.diff(library["offsets"]) == 0)),
                    conditioning="simulated incident electron state; not a reco correction",
                    elapsed_seconds=time.perf_counter() - begin)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, library, metadata)
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
