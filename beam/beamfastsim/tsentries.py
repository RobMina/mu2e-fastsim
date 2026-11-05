"""Count persisted art event-tree entries without interpreting logs/GenEventCount as saved events."""
import argparse
import json
from pathlib import Path

from .tsg4bl import _local_path
from .tsreference import _source, _tree


def audit(paths):
    """Return highest Events cycles and source hashes for local inputs; no generated-exposure claim."""
    import uproot

    paths = list(paths)
    if not paths:
        raise ValueError("At least one art input is required")
    files = []
    seen = set()
    for path in paths:
        path = _local_path(path)
        source = _source(path)
        if source["sha256"] in seen:
            raise ValueError("Duplicate or byte-identical art input")
        seen.add(source["sha256"])
        with uproot.open(path) as root:
            events, selected, cycles = _tree(root, "Events")
            files.append(dict(source, selected_key=selected, cycles=cycles,
                              saved_events=int(events.num_entries)))
    return dict(schema="ts-art-entries-v1", files=files,
                saved_events=sum(item["saved_events"] for item in files),
                normalization_unverified=True,
                scope="Persisted highest-cycle Events entries only; not generated POT or an inclusive population")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inputs = [_local_path(path) for path in args.inputs]
        if any(args.out.resolve() == path or (args.out.exists() and args.out.samefile(path)) for path in inputs):
            raise ValueError("--out must not overwrite inputs")
        report = audit(inputs)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved event entries: {report['saved_events']}; receipt: {args.out}")


if __name__ == "__main__":
    main()
