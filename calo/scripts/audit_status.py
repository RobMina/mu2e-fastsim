"""Reconcile exact integer PrintModule G4 status against finished art entries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import uproot


HEADER = re.compile(r"PrintModule Run/Subrun/Event\s+(\d+)\s+(\d+)\s+(\d+)")
STATUS = re.compile(r"status:\s+(\d+)\s+nG4Tracks:\s+(\d+)\s+overflowSimP:\s+(\d+)\s+nStepLimit:\s+(\d+)")
ROW = re.compile(HEADER.pattern + r"(?:(?!PrintModule Run/Subrun/Event).)*?" + STATUS.pattern, re.DOTALL)


def read_status_rows(status_log, entries):
    """Read one unique integer status record per independently printed event."""
    text = Path(status_log).read_text()
    if len(HEADER.findall(text)) != entries or len(STATUS.findall(text)) != entries:
        raise ValueError("one exact event header and G4 status required per saved entry")
    rows = np.asarray(ROW.findall(text), dtype=np.int64).reshape(-1, 7)
    if len(rows) != entries:
        raise ValueError("status rows do not align with event headers")
    ids, values = rows[:, :3], rows[:, 3:]
    if len(np.unique(ids, axis=0)) != entries:
        raise ValueError("duplicate status event identity")
    return rows


def audit(art_file, status_log, expected_events):
    with uproot.open(art_file) as source:
        entries = source["Events"].num_entries
    if entries != expected_events:
        raise ValueError(f"finished art has {entries} entries, expected {expected_events}")
    rows = read_status_rows(status_log, entries)
    ids, values = rows[:, :3], rows[:, 3:]
    codes, counts = np.unique(values[:, 0], return_counts=True)
    return {"events": entries, "status_counts": {str(int(code)): int(count) for code, count in zip(codes, counts)},
            "status_zero_events": int(np.count_nonzero(values[:, 0] == 0)),
            "nonzero_status_events": int(np.count_nonzero(values[:, 0] != 0)),
            "overflow_events": int(np.count_nonzero(values[:, 2])),
            "killed_step_limit_tracks": int(values[:, 3].sum()),
            "event_id_min": ids.min(axis=0).tolist() if entries else None,
            "event_id_max": ids.max(axis=0).tolist() if entries else None,
            "scope": "retained event/status census; not incident/shower completeness",
            "complete_incident_census": False, "zero_response_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("art_file", type=Path)
    parser.add_argument("status_log", type=Path)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.art_file, args.status_log, args.expected_events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
