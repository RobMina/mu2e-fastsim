"""Exact direct-input regression checks against saved choices and detector tables."""
import argparse
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


def compare_sampling(provenance, reference):
    """Require every persisted selection/ID plus source identity and campaign to match."""
    reference = Path(reference)
    metadata = json.loads((reference / "sampling.json").read_text())
    with np.load(reference / "sampling.npz", allow_pickle=False) as arrays:
        def check(actual, saved, key):
            if isinstance(saved, dict) and "npz_key" in saved:
                values = arrays[saved["npz_key"]]
                if not np.array_equal(actual, values) or np.asarray(actual).dtype != values.dtype:
                    raise ValueError(f"saved sampling array changed: {key}")
            elif isinstance(saved, dict):
                if set(actual) != set(saved):
                    raise ValueError(f"saved sampling keys changed: {key}")
                for name, value in saved.items():
                    check(actual[name], value, key + "/" + name)
            elif isinstance(saved, list):
                if list(actual) != saved:
                    raise ValueError(f"saved sampling list changed: {key}")
            elif actual != saved:
                raise ValueError(f"saved sampling value changed: {key}")
        for key in ("campaign", "seed", "stream_order", "rates", "intensity", "multiplicities", "sources",
                    "source_fingerprints", "signal_events_full", "signal_event_ids", "pool_event_counts",
                    "estimated_selected_rows", "actual_rows", "selections"):
            check(provenance[key], metadata[key], key)
    return True


def input_digests(mixed, tracker, crv, pdg):
    """Ordered buffers for full flat truth, detector columns and PDGs, including NaN bits."""
    def digest(values):
        values = np.ascontiguousarray(values)
        h = hashlib.sha256()
        h.update(str(values.dtype).encode())
        h.update(str(values.shape).encode())
        h.update(memoryview(values.reshape(-1)).cast("B"))
        return h.hexdigest()
    result = {"event_ids": digest(mixed.event_ids), "pdg": digest(pdg)}
    for name, rows in mixed.collections.items():
        result[name + "/counts"] = digest(ak.to_numpy(ak.num(rows)))
        for field in ak.fields(rows):
            result[name + "/" + field] = digest(ak.to_numpy(ak.flatten(rows[field])))
    for name, table in (("tracker_input", tracker), ("crv_input", crv)):
        for field in table.columns:
            result[name + "/" + field] = digest(table[field].to_numpy())
    return result


def compare_outputs(reference, candidate):
    """Compare highest-cycle reduced detector tables, equal NaNs but never relaxed tolerances."""
    report = {"equal": True, "objects": []}
    with uproot.open(reference) as left, uproot.open(candidate) as right:
        names = ("crvfast/events", "crvfast/digis", "crvfast/pulses", "crvfast/clusters", "trkdigi/events")
        for name in names:
            if name not in left or name not in right:
                if (name in left) != (name in right) or name != "trkdigi/events":
                    raise ValueError(f"missing output table: {name}")
                continue
            a, b = left[name], right[name]
            if a.num_entries != b.num_entries or set(a.keys()) != set(b.keys()):
                raise ValueError(f"output census or schema changed: {name}")
            for field in a.keys():
                x, y = a[field].array(), b[field].array()
                if a[field].typename != b[field].typename:
                    raise ValueError(f"output branch schema changed: {name}/{field}")
                if x.ndim != y.ndim:
                    raise ValueError(f"output array rank changed: {name}/{field}")
                for axis in range(1, x.ndim):
                    if not ak.almost_equal(ak.num(x, axis=axis), ak.num(y, axis=axis), rtol=0, atol=0):
                        raise ValueError(f"output row shape changed: {name}/{field}")
                for axis in range(x.ndim):
                    if not ak.almost_equal(ak.is_none(x, axis=axis), ak.is_none(y, axis=axis), rtol=0, atol=0):
                        raise ValueError(f"output missing-value positions changed: {name}/{field}")
                xf, yf = ak.to_numpy(ak.flatten(x, axis=None)), ak.to_numpy(ak.flatten(y, axis=None))
                if xf.dtype != yf.dtype or not np.array_equal(xf, yf, equal_nan=True):
                    raise ValueError(f"output column changed: {name}/{field}")
            report["objects"].append({"name": name, "entries": a.num_entries, "columns": len(a.keys())})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_outputs(args.reference, args.candidate)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
