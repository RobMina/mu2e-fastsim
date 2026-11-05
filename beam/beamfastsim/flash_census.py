"""Read-only, explicit-dataset inventory for beam-flash and mixing references."""
import argparse
import fnmatch
import json
from datetime import datetime, timezone
from pathlib import Path


def census(client, datasets, examples=2):
    """Keep missing datasets and query failures distinct from empty datasets."""
    if examples < 1:
        raise ValueError("examples must be positive")
    rows = []
    for did in datasets:
        if ":" not in did or any(c in did for c in "*?;\n\r\t "):
            raise ValueError("Supply explicit namespace:dataset identifiers, not patterns")
        row = {"did": did}
        try:
            info = client.get_dataset(did=did, exact_file_count=False)
            if info is None:
                row.update(status="missing", dataset=None, files=[])
            else:
                files = list(client.query(f"files from {did} limit {examples}"))
                detailed = []
                for file in files:
                    file_did = f"{file['namespace']}:{file['name']}"
                    detailed.append(client.get_file(did=file_did, with_metadata=True))
                row.update(status="ok", dataset=info, files=detailed)
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__, error=str(exc))
        rows.append(row)
    return dict(schema="flash-input-census-v1", timestamp_utc=datetime.now(timezone.utc).isoformat(),
                scope="Read-only catalog metadata; neither file locality nor payload completeness is certified",
                datasets=rows)


def discover(client, namespace, patterns):
    """List one explicit namespace without counts; filter names locally before detailed reads."""
    if not namespace or any(c in namespace for c in ":*?;\n\r\t "):
        raise ValueError("Discovery requires one explicit namespace")
    if not patterns or any(not p or any(c in p for c in ":;\n\r\t ") for p in patterns):
        raise ValueError("Supply nonempty dataset-name patterns")
    try:
        datasets = list(client.list_datasets(namespace_pattern=namespace, with_counts=False))
        selected = sorted({f"{d['namespace']}:{d['name']}" for d in datasets
                           if d["namespace"] == namespace and any(fnmatch.fnmatchcase(d["name"], p) for p in patterns)})
        return dict(namespace=namespace, patterns=patterns, status="ok", scanned_datasets=len(datasets),
                    selected_datasets=selected)
    except Exception as exc:
        return dict(namespace=namespace, patterns=patterns, status="error",
                    error_type=type(exc).__name__, error=str(exc), selected_datasets=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*")
    parser.add_argument("--discover-namespace")
    parser.add_argument("--pattern", action="append")
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.discover_namespace and (args.datasets or not args.pattern):
        parser.error("Namespace discovery requires --pattern and no explicit datasets")
    if not args.discover_namespace and (not args.datasets or args.pattern):
        parser.error("Supply explicit datasets, or --discover-namespace with --pattern")
    from metacat.webapi import MetaCatClient
    client = MetaCatClient()
    discovery = discover(client, args.discover_namespace, args.pattern) if args.discover_namespace else None
    datasets = discovery["selected_datasets"] if discovery is not None else args.datasets
    result = census(client, datasets, args.examples)
    if discovery is not None:
        result["discovery"] = discovery
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    for row in result["datasets"]:
        print(row["did"], row["status"], (row.get("dataset") or {}).get("file_count"))
    return int(any(row["status"] == "error" for row in result["datasets"]) or
               (discovery is not None and discovery["status"] == "error"))


if __name__ == "__main__":
    raise SystemExit(main())
