"""Inventory existing generator-bank support; never replace a library or certify provenance.

Legacy inferred event counts are checked against the retained census; subrun primaries
are recorded without assuming their semantics. Strict stream role remains a separate gate.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import uproot

from crvfastsim import tables, tier1b as T
from .corrected_denominator import support_report
from .descendant_audit import digest


def count_file(path, cells, expected_events):
    start = time.monotonic()
    with uproot.open(path) as root:
        cycles = sorted(int(k.rsplit(";", 1)[1]) for k in root["dump"].keys() if k.startswith("gen;"))
        tree = root[f"dump/gen;{cycles[-1]}"]
        g = tree.arrays(tables.KEY + ["idx", "pdg", "e", "t"], library="pd")
        sub = root["dump/subruns"].arrays(library="pd")
        strict_census = "dump/events" in root
    if g.duplicated(tables.KEY + ["idx"]).any() or not np.isfinite(g[["e", "t"]]).all().all():
        raise ValueError("duplicate/nonfinite generator payload")
    events = g[tables.KEY].drop_duplicates()
    sr = events[tables.KEY[:2]].drop_duplicates()
    if len(sr) != 1 or len(events) != expected_events:
        raise ValueError(f"full-event census/subrun exposure mismatch: subruns={len(sr)}, inferred={len(events)}, "
                         f"expected={expected_events}, subrun_primaries={sub.primaries.sum()}")
    from .shower_uncertainty import bin_frame
    nonmu = bin_frame(g)
    counts = nonmu.groupby(["apdg", "kebin"]).size().reindex(pd.MultiIndex.from_frame(cells), fill_value=0).to_numpy()
    return counts, {"path": str(path), "sha256": digest(path), "cycles": cycles, "selected_cycle": cycles[-1],
                    "rows": len(g), "events_inferred": len(events), "primaries_subruns": int(sub.primaries.sum()),
                    "run": int(sr.run.iloc[0]), "subrun": int(sr.subrun.iloc[0]), "strict_census_present": strict_census,
                    "seconds": time.monotonic() - start}


def run(census, directory, audit, out, first=1, last=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    source = json.loads(Path(census).read_text())["crygen_v1"]
    old = json.loads(Path(audit).read_text())
    if source["problems"]:
        raise ValueError("retained generator census has problems")
    records = source["files"][first - 1:last]
    if len(records) < 2:
        raise ValueError("at least two independent files required")
    cells = pd.DataFrame(old["bins"])
    counts, rows, seen = [], [], set()
    start = time.monotonic()
    for record in records:
        path = Path(directory) / record["file"]
        c, row = count_file(path, cells, record["events"])
        key = row["run"], row["subrun"]
        if key in seen:
            raise ValueError("overlapping generator files")
        seen.add(key)
        counts.append(c)
        rows.append(row)
        print(f"{record['file']}: {row['events_inferred']} events; {row['seconds']:.2f} s", flush=True)
    events = np.array([r["events_inferred"] for r in rows], dtype=np.int64)
    report = support_report(np.array(counts, dtype=np.int64), events,
                            np.asarray(old["corrected"]["nominal_showers"], dtype=np.int64),
                            old["generated_exposure"], 1000, 20261027)
    result = {"schema": 1, "candidate_only": True, "downstream_authorized": False,
              "census_sha256": digest(census), "audit_sha256": digest(audit), "files": rows,
              "bins": old["bins"], "per_file_counts": np.asarray(counts).tolist(), "events": int(events.sum()),
              "support": report, "source_sha256": digest(__file__), "seconds": time.monotonic() - start,
              "limits": ["Original library and libgen remain unchanged; candidate-bank calculation only.",
                         "Legacy stream role/empty-event completeness need strict census and producer-compatibility proof.",
                         "No independent G4 reference exists for this generator-only bank.",
                         "No donor variance propagation or raw-zero repair is authorized."]}
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"events": result["events"], "invalid_draws": report["any_invalid_replicas"],
                      "missing_cells": sum(report["nominal_missing_support"]),
                      "saturated_cells": sum(report["nominal_saturated"]), "seconds": result["seconds"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("census", "directory", "audit", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--first", type=int, default=1)
    p.add_argument("--last", type=int)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
