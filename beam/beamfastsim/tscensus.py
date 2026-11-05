"""Read-only MetaCat inventory for upstream beam references; no event-content inference."""
from __future__ import annotations

import argparse
import fnmatch
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATTERNS = ("*.mu2e.POT*.MDC2025*.*", "sim.mu2e.Beam.MDC2025*.art",
                    "sim.mu2e.BeamCat.MDC2025*.art", "sim.mu2e.MuBeamCat.MDC2025*.art",
                    "cnf.mu2e.BeamSplitter.MDC2025*.tar", "log.mu2e.BeamSplitter.MDC2025*.log")


def census(client, patterns=DEFAULT_PATTERNS, *, samples=2, namespace="mu2e", include_retired=False):
    """Return catalogue facts only, retaining errors rather than interpreting absence as zero."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if not isinstance(namespace, str) or not namespace.strip() or namespace != namespace.strip():
        raise ValueError("namespace must be a nonempty exact namespace or wildcard pattern")
    if type(include_retired) is not bool:
        raise ValueError("include_retired must be a bool")
    records, errors = [], []
    discovered = []
    try:
        for dataset in client.list_datasets(namespace_pattern=namespace, with_counts=False):
            discovered.append(dataset)
    except Exception as exc:
        errors.append({"stage": "discovery", "error": f"{type(exc).__name__}: {exc}"})
    selected = sorted((d for d in discovered if any(fnmatch.fnmatchcase(d["name"], p) for p in patterns)),
                      key=lambda d: (d["namespace"], d["name"]))
    for dataset in selected:
        did = f"{dataset['namespace']}:{dataset['name']}"
        row = {"did": did, "contents_verified": False, "files": []}
        try:
            row["dataset"] = client.get_dataset(did=did, exact_file_count=False)
            query = f"files from {did} limit {samples}"
            row["sample_query"] = query
            options = {"include_retired_files": True} if include_retired else {}
            files = list(client.query(query, with_metadata=True, with_provenance=True, **options))
            row["sample_count"] = len(files)
            if not files and row["dataset"].get("file_count", 0):
                scope = "including-retired" if include_retired else "non-retired"
                row["sample_warning"] = f"Discovery count is nonzero but current {scope} sample query returned no files"
            for f in files:
                file_did = f"{f['namespace']}:{f['name']}"
                detail = client.get_file(did=file_did, with_metadata=True)
                if detail is None:
                    raise ValueError(f"sample file disappeared: {file_did}")
                row["files"].append(detail)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append({"did": did, "error": row["error"]})
        records.append(row)
    return {"schema": "ts-census-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "patterns": list(patterns), "namespace_pattern": namespace, "datasets": records, "errors": errors,
            "include_retired_files": include_retired,
            "complete": not errors,
            "claims": {"vd_contents_verified": False, "unbiased_ts_entrance": False,
                       "pot_normalisation_verified": False},
            "warning": "Counts are catalogue counts; sampled metadata are not event sums or VD-content evidence."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", action="append", help="dataset-name wildcard; repeat to extend discovery")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--namespace", default="mu2e", help="namespace pattern; default mu2e, explicit * includes personal studies")
    parser.add_argument("--include-retired", action="store_true",
                        help="include retired metadata records; this does not establish usable replicas")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    from metacat.webapi import MetaCatClient
    result = census(MetaCatClient(), args.pattern or DEFAULT_PATTERNS, samples=args.samples,
                    namespace=args.namespace, include_retired=args.include_retired)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"datasets": len(result["datasets"]), "errors": len(result["errors"]),
                      "complete": result["complete"]}))
    return 0 if result["complete"] and result["datasets"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
