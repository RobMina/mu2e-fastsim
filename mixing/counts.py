"""Read event-aligned plane/sector counts from native detector dump tables."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from .compare import occupancy, paired_counts


def event_keys(ids):
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.shape[1] != 3 or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("event IDs must be a nonnegative integer N by 3 array")
    return np.ascontiguousarray(ids, dtype=np.uint64).view(
        np.dtype([("run", "u8"), ("subrun", "u8"), ("event", "u8")])).reshape(-1)


def event_ids(rows):
    return np.column_stack([ak.to_numpy(rows[key]) for key in ("run", "subrun", "event")])


def read_counts(path, bar_to_sector, n_sectors):
    """Count all stored pulses, including zero-pulse events, using an explicit geometry map."""
    with uproot.open(path) as root:
        tracker = root["trkdigi/events"].arrays(["run", "subrun", "event", "d_sid"], library="ak")
        crv = root["crvdump/events"].arrays(["run", "subrun", "event", "npulse"], library="ak")
        pulses = root["crvdump/pulses"].arrays(["run", "subrun", "event", "bar"], library="ak")
    ids = event_ids(tracker)
    if not np.array_equal(ids, event_ids(crv)):
        raise ValueError("tracker/CRV event identities differ")
    keys = event_keys(ids)
    if np.unique(keys).size != keys.size:
        raise ValueError("duplicate primary event IDs")
    planes = occupancy(tracker.d_sid, (np.arange(65536) & 0xFC00) >> 10, 36)
    mapping = np.asarray(bar_to_sector)
    if mapping.ndim != 1 or mapping.dtype.kind not in "iu" or n_sectors <= 0:
        raise ValueError("invalid sector geometry map")
    if np.any(mapping < 0) or np.any(mapping >= n_sectors):
        raise ValueError("sector map contains out-of-range bins")
    bars = ak.to_numpy(pulses.bar)
    if np.any(bars < 0) or np.any(bars >= mapping.size):
        raise ValueError("CRV bar outside geometry map")
    order = np.argsort(keys)
    pulse_keys = event_keys(event_ids(pulses))
    where = np.searchsorted(keys[order], pulse_keys)
    if np.any(where == len(keys)) or not np.array_equal(keys[order[where]], pulse_keys):
        raise ValueError("pulse without matching event census entry")
    sector_counts = np.bincount(order[where] * n_sectors + mapping[bars],
                               minlength=len(keys) * n_sectors).reshape(len(keys), n_sectors)
    if not np.array_equal(sector_counts.sum(axis=1), ak.to_numpy(crv.npulse)):
        raise ValueError("pulse table count disagrees with event census")
    return {"event_ids": ids, "planes": planes, "sectors": sector_counts}


def align(stock, candidate):
    """Pair the intersection explicitly and report unmatched counts, never truncate by row."""
    skey, ckey = event_keys(stock["event_ids"]), event_keys(candidate["event_ids"])
    if np.unique(skey).size != skey.size or np.unique(ckey).size != ckey.size:
        raise ValueError("duplicate primary event IDs")
    _, si, ci = np.intersect1d(skey, ckey, assume_unique=True, return_indices=True)
    result = {"matched_events": len(si), "stock_only_events": len(skey) - len(si),
              "candidate_only_events": len(ckey) - len(ci)}
    for name in ("planes", "sectors"):
        result[name] = paired_counts(stock[name][si], candidate[name][ci])
        result[name + "_changed_events"] = int(np.count_nonzero(
            np.any(stock[name][si] != candidate[name][ci], axis=1)))
        result[name + "_total"] = paired_counts(stock[name][si].sum(axis=1, keepdims=True),
                                               candidate[name][ci].sum(axis=1, keepdims=True))
    return result


def json_values(value):
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_values(value.tolist())
    if isinstance(value, list):
        return [json_values(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--geometry", type=Path, required=True,
                        help="JSON with bar_to_sector, sector_names, geometry and coverage")
    parser.add_argument("--selection", required=True, help="explicit event selection description")
    parser.add_argument("--out", type=Path, required=True, help="NPZ output; JSON summary written beside it")
    parser.add_argument("--stock", type=Path, help="optional saved NPZ for an event-intersection comparison")
    args = parser.parse_args(argv)
    geometry = json.loads(args.geometry.read_text())
    names = geometry["sector_names"]
    mapping = (np.asarray(geometry["bar_to_sector"]) if "bar_to_sector" in geometry else
               np.repeat(np.arange(len(names)), geometry["bars_per_sector"]))
    data = read_counts(args.input, mapping, len(names))
    np.savez_compressed(args.out, **data)
    report = {"events": len(data["event_ids"]), "geometry": geometry["geometry"],
              "coverage": geometry["coverage"], "selection": args.selection, "sector_names": names,
              "straw_digis_per_plane": data["planes"].sum(axis=0),
              "crv_pulses_per_sector": data["sectors"].sum(axis=0),
              "tracker_empty_events": int(np.sum(data["planes"].sum(axis=1) == 0)),
              "crv_empty_events": int(np.sum(data["sectors"].sum(axis=1) == 0))}
    if args.stock:
        with np.load(args.stock, allow_pickle=False) as stock:
            report["paired"] = align(stock, data)
        report["comparison_note"] = "Event intersection only; selection and detector coverage are not corrected."
    text = json.dumps(json_values(report), indent=2, allow_nan=False) + "\n"
    args.out.with_suffix(".json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
