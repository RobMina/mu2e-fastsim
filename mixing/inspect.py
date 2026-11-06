"""Summarize extracted joint frames, including empty events and stored time ranges."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

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


def summarize_native(path):
    """Count finished native-digi entries and confirm both detector event sequences."""
    with uproot.open(path) as root:
        tracker = root["trkdigi/events"].arrays(library="ak")
        crv = root["crvdump/events"].arrays(library="ak")
        for key in ("run", "subrun", "event"):
            if not np.array_equal(ak.to_numpy(tracker[key]), ak.to_numpy(crv[key])):
                raise ValueError("tracker/CRV event identities differ")
        result = {"events": len(tracker), "tracker_digis": int(ak.sum(ak.num(tracker.d_sid))),
                  "tracker_empty_events": int(ak.sum(ak.num(tracker.d_sid) == 0))}
        for name in ("charges", "digimc", "digis", "pulses", "clusters"):
            key = f"crvdump/{name}"
            if name in ("charges", "digimc") and key not in root:
                result["crv_" + name] = None
            else:
                result["crv_" + name] = root[key].num_entries
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--native", action="store_true", help="summarize native tracker/CRV digi dumps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.native:
        result = summarize_native(args.input)
    else:
        ids = read_event_log(args.event_log) if args.event_log else None
        result = summarize(read_frames(args.input, event_ids=ids))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
