"""Inspect art event-order metadata and exact DIO parent joins; no art physics decoding."""
import argparse
import json

import numpy as np


def inspect(art, parent, reference):
    import uproot
    report = {}
    with uproot.open(art) as root:
        report["art_keys"] = root.keys()
        for name in ("Events", "MetaData"):
            tree = root[name]
            report[name + "_keys"] = tree.keys()
            report[name + "_types"] = tree.typenames()
        tree = root["FileIndex"]
        report["FileIndex_keys"] = tree.keys()
        report["FileIndex_types"] = tree.typenames()
        for name in tree.keys():
            try:
                a = tree[name].array(entry_stop=3, library="ak")
                report["FileIndex_" + name] = dict(type=str(a.type), fields=a.fields, preview=str(a))
            except Exception as e:
                report["FileIndex_" + name] = dict(error=str(e))
        a = tree["Element"].array(entry_stop=12, library="ak")
        report["FileIndex_first12"] = dict(run=np.asarray(a.eventID.subRun_.run_.run_).tolist(),
            subrun=np.asarray(a.eventID.subRun_.subRun_).tolist(), evt=np.asarray(a.eventID.event_).tolist(),
            entry=np.asarray(a.entry).tolist())
        tree = root["Events"]
        for name in tree.keys():
            if "EventAuxiliary" in name:
                try:
                    a = tree[name].array(entry_stop=2, library="ak")
                    report[name] = dict(type=str(a.type), fields=a.fields, preview=str(a))
                except Exception as e:
                    report[name] = dict(error=str(e))
    with np.load(parent) as p, np.load(reference) as r:
        xyz = np.column_stack([p[k] for k in ("xstop", "ystop", "zstop")]).astype("<f4")
        query = np.column_stack([r[k] for k in ("x", "y", "z")]).astype("<f4")
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        keys, q = np.ascontiguousarray(xyz).view(dtype).ravel(), np.ascontiguousarray(query).view(dtype).ravel()
        order = np.argsort(keys)
        left, right = np.searchsorted(keys[order], q, "left"), np.searchsorted(keys[order], q, "right")
        if np.any(right - left != 1):
            raise ValueError("nonunique parent matches")
        idx = order[left]
        event_index = p["event_index"][idx]
        event_keys = np.column_stack([p[k][idx] for k in ("run", "subrun", "evt")])
        rows = np.flatnonzero(r["ifile"] == 0)
        offsets = (event_index[rows] - (r["evt"][rows] - 1)) % len(p["event_multiplicity"])
        change = np.r_[True, np.diff(offsets) != 0]
        selected = rows[change]
        report["first_file_offset_boundaries"] = dict(generated_event=r["evt"][selected].tolist(),
            parent_event_index=event_index[selected].tolist(), parent_keys=event_keys[selected].tolist(),
            offsets=offsets[change].tolist())
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--art", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    report = inspect(a.art, a.parent, a.reference)
    with open(a.out, "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
