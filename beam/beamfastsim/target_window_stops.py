"""Build the exact DIO production-window stop population, not a full-parent-pool surrogate."""
import argparse
import json
from pathlib import Path

import numpy as np

from .target_parent_audit import checksum


STOP_COLUMNS = ("run", "subrun", "evt", "trk", "pdg", "codestop", "pstop", "xstop", "ystop", "zstop", "tstop", "at_rest", "foil", "event_index")


def select_windows(parent, receipt):
    """Repeat parent rows by audited window exposure while retaining their original identity."""
    exposure = receipt.get("production_exposure", {})
    if not exposure.get("exact_production_electron_exposure_known", False):
        raise ValueError("exact production-window exposure receipt required")
    records = exposure.get("files", [])
    if not records or sorted(r["ifile"] for r in records) != list(range(len(records))):
        raise ValueError("receipt must enumerate each reference file exactly once")
    if not exposure.get("reference_sha256"):
        raise ValueError("receipt must bind the reference hash")
    counts = np.asarray(parent["event_multiplicity"])
    index = np.asarray(parent["event_index"])
    if (counts.ndim != 1 or not counts.size or np.any(counts < 1)
            or np.any(counts != np.floor(counts)) or index.ndim != 1
            or np.any(index < 0) or np.any(index >= counts.size)
            or np.any(index != np.floor(index))):
        raise ValueError("invalid parent event multiplicities or indices")
    if any(np.shape(parent[k]) != index.shape for k in STOP_COLUMNS):
        raise ValueError("parent stop columns must have equal length")
    if not np.array_equal(np.bincount(index.astype(int), minlength=counts.size), counts):
        raise ValueError("parent stop rows disagree with audited event multiplicities")
    if not np.all((parent["pdg"] == 13) & (parent["codestop"] == 32) & parent["at_rest"] & (parent["foil"] >= 0)):
        raise ValueError("all parent rows must be eligible target muon stops")
    selected, groups, reports = [], [], []
    for record in sorted(records, key=lambda r: r["ifile"]):
        offset, n = record["inferred_first_parent_event_index"], record["generated_events"]
        if (not isinstance(offset, int) or not 0 <= offset < counts.size
                or not isinstance(n, int) or n < 1):
            raise ValueError("invalid production window offset or length")
        cycles, rest = divmod(n, counts.size)
        repetitions = cycles + ((index - offset) % counts.size < rest).astype(int)
        rows = np.repeat(np.arange(index.size), repetitions)
        if rows.size != record["generated_electrons"]:
            raise ValueError("window population disagrees with exact electron exposure")
        selected.append(rows)
        groups.append(np.full(rows.size, record["ifile"], np.int32))
        reports.append(dict(ifile=record["ifile"], generated_events=n, generated_electrons=int(rows.size),
                            inferred_first_parent_event_index=offset))
    rows = np.concatenate(selected)
    if rows.size != exposure["total_generated_electrons"]:
        raise ValueError("window total disagrees with exposure receipt")
    out = {k: np.asarray(parent[k])[rows] for k in STOP_COLUMNS}
    out.update(parent_stop_index=rows, production_ifile=np.concatenate(groups))
    report = dict(mode="exact-DIO-production-window-stop-population", files=reports,
                  eligible_muons=int(rows.size), reference_sha256=exposure["reference_sha256"],
                  filter_implication_certified=False, generated_yield_validated=False,
                  caveat="source-population alignment only; independent electron draws do not reproduce mixer random history or certify the downstream filter")
    return out, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--exposure", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    receipt = json.loads(Path(a.exposure).read_text())
    with np.load(a.parent) as parent:
        if "metadata" not in parent or json.loads(str(parent["metadata"])) != receipt:
            raise ValueError("parent metadata does not match exposure receipt")
        if receipt["production_exposure"]["reference_sha256"] != checksum(a.reference):
            raise ValueError("exposure reference hash mismatch")
        out, report = select_windows(parent, receipt)
    report.update(parent_npz_sha256=checksum(a.parent), exposure_sha256=checksum(a.exposure))
    np.savez_compressed(a.out, **out, metadata=json.dumps(report, allow_nan=False))
    Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
