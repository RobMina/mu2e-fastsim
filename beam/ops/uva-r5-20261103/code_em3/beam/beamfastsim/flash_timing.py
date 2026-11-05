"""Audit stored beam times without folding, clipping, smearing or selecting particles.

Float ntuples lose clock phase at sufficiently large absolute times. A finite time
alone does not certify a microbunch phase. Reports preserve the largest-time input
identities and creation codes for independent collection-generic inspection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot


THRESHOLDS_NS = (1695.0, 1e6, 1e9, 1e12, 1e15)


def timing_summary(columns, *, resolution_ns=1.0, thresholds_ns=THRESHOLDS_NS):
    """Describe original storage precision; never convert a long absolute time to phase."""
    t = np.asarray(columns["time"])
    if t.ndim != 1 or t.dtype.kind != "f" or np.any(~np.isfinite(t)):
        raise ValueError("Stored times must be a finite floating-point vector")
    if not np.isfinite(resolution_ns) or resolution_ns <= 0:
        raise ValueError("resolution_ns must be finite and positive")
    thresholds = np.asarray(thresholds_ns, float)
    if thresholds.ndim != 1 or np.any(~np.isfinite(thresholds)) or np.any(thresholds <= 0) or np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds_ns must be finite, positive and increasing")
    names = ("run", "subrun", "evt", "trk", "pdg", "code")
    if any(np.asarray(columns[k]).shape != t.shape for k in names):
        raise ValueError("Time identity and process columns must align")
    if any(np.any(~np.isfinite(columns[k])) or np.any(columns[k] != np.floor(columns[k])) for k in names):
        raise ValueError("Invalid identity, species or process code")
    if any(np.any(np.abs(columns[k]) >= 2**24) for k in names[:4]):
        raise ValueError("Float ntuple identity exceeds exact integer range")
    spacing = np.abs(np.spacing(t)).astype(float)
    species, inv = np.unique(np.column_stack((columns["pdg"], columns["code"])), axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=len(species))
    largest = np.argsort(t, kind="stable")[-min(10, len(t)):][::-1]
    quantiles = (0, .05, .5, .95, .99, 1)
    return dict(rows=len(t), storage_dtype=t.dtype.name, requested_resolution_ns=float(resolution_ns),
                quantile_probabilities=list(quantiles),
                time_quantiles_ns=np.quantile(t, quantiles).tolist() if len(t) else None,
                spacing_quantiles_ns=np.quantile(spacing, quantiles).tolist() if len(t) else None,
                phase_unresolved_rows=int((spacing > resolution_ns).sum()), negative_time_rows=int((t < 0).sum()),
                thresholds_ns=thresholds.tolist(), rows_above_threshold=[int((t > v).sum()) for v in thresholds],
                species_creation_code=dict(pdg=species[:, 0].astype(int).tolist(),
                                           code=species[:, 1].astype(int).tolist(), count=counts.tolist()),
                largest_rows=dict(row=largest.tolist(), **{k: np.asarray(columns[k])[largest].astype(int).tolist() for k in names},
                                  time_ns=t[largest].astype(float).tolist(), spacing_ns=spacing[largest].tolist()),
                convention="Stored global time, unchanged; creation codes are numeric; no pulse or phase certification")


def audit(path, *, resolution_ns=1.0):
    """Audit the stock ntvd values as diagnostics, not as a cut-collection boundary certificate."""
    with uproot.open(path) as root:
        tree = root["input/ntvd"]
        a = tree.arrays(["run", "subrun", "evt", "trk", "pdg", "code", "time", "gtime", "g4bl_time"], library="np")
        report = timing_summary(a, resolution_ns=resolution_ns)
        report["proper_time_quantiles_ns"] = np.quantile(a["gtime"], [0, .5, 1]).tolist() if len(a["gtime"]) else None
        report["g4beamline_time_nonzero_rows"] = int((a["g4bl_time"] != 0).sum())
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(schema="flash-time-audit-v1", sha256=digest.hexdigest(), **report,
                reader_qualification="ReadVirtualDetector assumes VD geometry; cut-collection values require independent cross-check")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--resolution-ns", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.write_text(json.dumps(audit(args.file, resolution_ns=args.resolution_ns), indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
