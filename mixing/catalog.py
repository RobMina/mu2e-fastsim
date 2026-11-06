"""Read-only, bounded file discovery for frame pools and mixed references."""
import argparse
import itertools
import json
from pathlib import Path


def census(client, datasets, limit=3):
    """Report bounded query and membership examples alongside cached dataset counts."""
    if limit < 1:
        raise ValueError("limit must be positive")
    result = []
    for dataset in datasets:
        info = client.get_dataset(did=dataset, exact_file_count=False)
        active = list(client.query(f"files from {dataset} limit {limit}"))
        members = list(itertools.islice(
            client.get_dataset_files(did=dataset, with_metadata=True), limit))
        examples = []
        for row in members:
            did = f"{row['namespace']}:{row['name']}"
            detail = client.get_file(did=did, with_metadata=True)
            if detail is None:
                raise ValueError(f"catalog member missing: {did}")
            examples.append({key: detail.get(key) for key in
                             ("namespace", "name", "size", "retired", "retired_by", "retired_timestamp", "metadata")})
        result.append({"dataset": dataset, "estimated_files": info.get("file_count"),
                       "query_examples": [f"{r['namespace']}:{r['name']}" for r in active],
                       "membership_examples": examples})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from metacat.webapi import MetaCatClient
    result = census(MetaCatClient(), args.datasets, args.limit)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for dataset in result:
        print(dataset["dataset"], "query examples:", len(dataset["query_examples"]),
              "membership examples:", len(dataset["membership_examples"]))


if __name__ == "__main__":
    main()
