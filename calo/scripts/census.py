"""Read-only dataset census; names alone never certify detector products."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def census(client, datasets):
    rows = []
    for did in datasets:
        info = client.get_dataset(did=did, exact_file_count=True)
        if info is None:
            raise ValueError(f"dataset not found: {did}")
        files = list(client.query(f"files from {did} limit 1"))
        example = None
        if files:
            file_did = f"{files[0]['namespace']}:{files[0]['name']}"
            example = client.get_file(did=file_did, with_metadata=True)
            if example is None:
                raise ValueError(f"file not found: {file_did}")
        rows.append({"did": did, "dataset": info, "example": example,
                     "product_content_verified": False})
    return {"schema_version": 1,
            "queried_utc": datetime.now(timezone.utc).isoformat(),
            "datasets": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()
    from metacat.webapi import MetaCatClient
    result = census(MetaCatClient(), args.datasets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for row in result["datasets"]:
        sample = row["example"]
        print(row["did"], row["dataset"].get("file_count"),
              sample["name"] if sample else "EMPTY", flush=True)


if __name__ == "__main__":
    main()
