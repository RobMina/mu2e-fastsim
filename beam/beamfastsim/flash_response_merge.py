"""Merge only comparison fields from finished response shards, never raw step rows.

Full event identity and ordinal uniqueness are checked across shards. Photon
identity is event/product/key qualified. No missing shard/event is imputed, no
unknown response is relabelled zero, and causal row-index arrays are not merged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .flash_inclusive_check import keys, join
from .flash_response_compare import _arm, EVENT_FIELDS, TOTAL_FIELDS

EVENT_COLUMNS = (*EVENT_FIELDS, *("event_" + k + "_ionizing_mev" for k in TOTAL_FIELDS.values()))
PHOTON_COLUMNS = ("run", "subrun", "event", "product", "key", "pre_energy_mev", "photon_response_known",
                  "frontier_status", "photon_assigned_ionizing_mev")


def merge(paths, *, expected_events):
    """Concatenate event/photon columns after validating each input independently."""
    if type(expected_events) is not int or expected_events <= 0 or not paths:
        raise ValueError("positive expected_events and nonempty input paths required")
    pieces = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            part = {k: z[k] for k in (*EVENT_COLUMNS, *PHOTON_COLUMNS)}
            _arm(part)
            report = json.loads(str(z["response_report_json"]))
            if report["input_events"] != len(part["event_run"]) or report["expected_events"] != report["input_events"]:
                raise ValueError("shard completion/count mismatch")
            if report["failed_recording_events"] or report["incomplete_tracker_events"]:
                raise ValueError("failed recording shard cannot enter an ordinary merged comparison")
            for k in (*EVENT_FIELDS, *PHOTON_COLUMNS[:5]):
                part[k] = part[k].astype(np.uint64, copy=False)
            pieces.append(part)
    result = {k: np.concatenate([p[k] for p in pieces]) for k in (*EVENT_COLUMNS, *PHOTON_COLUMNS)}
    if len(result["event_run"]) != expected_events:
        raise ValueError("merged input event count differs from expected_events")
    join(keys(result, EVENT_FIELDS[:3]), keys(result, EVENT_FIELDS[:3]))
    join(keys(result, ("event_ordinal",)), keys(result, ("event_ordinal",)))
    order = np.argsort(result["event_ordinal"], kind="stable")
    for k in EVENT_COLUMNS:
        result[k] = result[k][order]
    result["input_event_count"] = np.asarray(expected_events, np.uint64)
    _arm(result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, nargs="+", type=Path)
    p.add_argument("--events", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    result = merge(args.input, expected_events=args.events)
    result["merge_input_files"] = np.asarray(list(map(str, args.input)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, **result)
    print(f"{len(args.input)} shards, {args.events} events, {len(result['run'])} photon rows; comparison-only {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
