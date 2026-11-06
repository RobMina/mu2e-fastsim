"""Summarize extracted joint frames, including empty events and stored time ranges."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np

from .frames import read_frames, read_event_log


def summarize(batch):
    result = {"events": len(batch.event_ids), "collections": {}}
    for name, rows in batch.collections.items():
        counts = ak.to_numpy(ak.num(rows))
        summary = {"rows": int(counts.sum()), "empty_events": int(np.sum(counts == 0))}
        for field in ("t", "t0", "t1"):
            if field in ak.fields(rows):
                values = ak.to_numpy(ak.flatten(rows[field]))
                finite = values[np.isfinite(values)]
                summary[field] = {"min": float(finite.min()) if finite.size else None,
                                  "max": float(finite.max()) if finite.size else None,
                                  "nonfinite": int(np.sum(~np.isfinite(values)))}
        result["collections"][name] = summary
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    ids = read_event_log(args.event_log) if args.event_log else None
    result = summarize(read_frames(args.input, event_ids=ids))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
