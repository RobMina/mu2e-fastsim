"""Summarize saved complete-route profiles without new transport, draws or physics validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KEYS = ("energy", "time", "x", "y", "z")


def array_differences(a, b):
    if not np.array_equal(a["event_row"], b["event_row"]):
        raise ValueError("profile event rows differ")
    differences = {}
    for key in KEYS:
        x, y = a["fast_" + key], b["fast_" + key]
        finite = np.isfinite(x) & np.isfinite(y)
        delta = np.abs(x[finite] - y[finite])
        differences[key] = dict(exact=bool(np.array_equal(x, y, equal_nan=True)),
            nonfinite_mask_equal=bool(np.array_equal(np.isfinite(x), np.isfinite(y))),
            changed_finite_values=int(np.count_nonzero(delta)), max_abs=float(delta.max(initial=0)))
    return dict(selection_equal=bool(np.array_equal(a["fast_energy"] > 10, b["fast_energy"] > 10)), fields=differences)


def summarize(paths):
    results, references, scope = {}, {}, None
    for device, path in paths.items():
        results[device] = {}
        for placement in ("geometric", "crystal"):
            source = json.loads((path / placement / "numbers.json").read_text())
            if not source["timing"]["method"].startswith("single-call"):
                raise ValueError("only synchronized single-call complete routes are comparable")
            signature = {k: source["provenance"][k] for k in ("seed", "reco_commit", "reco_inputs", "stock_ntuple", "inputs", "library")}
            if scope is not None and signature != scope:
                raise ValueError("profile settings or input references differ")
            scope = signature
            with np.load(path / placement / "clusters.npz", allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            rows = arrays["event_row"]
            if (not np.array_equal(rows, np.arange(1000)) or len(rows) != source["events"]):
                raise ValueError("this profile summary requires the development source rows")
            runs = source["repeats"]
            if len(runs) < 2:
                raise ValueError("profile needs cold and warm calls")
            median = lambda name: float(np.median([r[name] for r in runs[1:]]))
            warm = median("total_seconds")
            stages = {key: float(np.median([r["stage_seconds"][key] for r in runs[1:]]))
                      for key in runs[0]["stage_seconds"]}
            row = dict(events=source["events"], first_seconds=runs[0]["total_seconds"], warm_seconds=warm,
                       generated_per_second=source["events"] / warm,
                       transport_seconds=median("transport_seconds"), handoff_seconds=median("handoff_seconds"),
                       reco_seconds=median("reco_seconds"), reco_stage_seconds=stages,
                       transport_fraction=median("transport_seconds") / warm,
                       selected_for_timing_consistency=source["fast_selected"],
                       nonfinite_reco=runs[0]["n_nonfinite_reco"], refit_skipped=runs[0]["n_refit_skipped"],
                       device=source["device"])
            if placement not in references:
                references[placement] = arrays
            row["vs_first_device"] = array_differences(references[placement], arrays)
            results[device][placement] = row
    return dict(devices=results, caveat="Development profiling, including potential self-donors; not an unbiased physics comparison. "
                "Same 1000 primaries and native reconstruction, cold plus two warm same-seed calls; finite-batch rates, not saturation. "
                "No matched stock full-physics speedup is established.", provenance=scope)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, metavar="DEVICE=DIRECTORY")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    paths = dict(item.split("=", 1) for item in args.run)
    result = summarize({key: Path(value) for key, value in paths.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
