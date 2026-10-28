"""Recheck completed pilot population and historical external assets without rerunning art."""
import argparse
import json
from pathlib import Path

from .descendant_audit import digest
from .descendant_inputs import bind_pilot, external_assets
from .tier1b_replay import frozen_files


def run(pilots, out):
    dest = Path(out)
    if dest.exists():
        raise FileExistsError(dest)
    rows = []
    for directory in pilots:
        root = Path(directory)
        m = json.loads((root / "manifest.json").read_text())
        c = json.loads((root / "completion.json").read_text())
        oldroot = Path(m["historical"])
        old = json.loads((oldroot / "manifest.json").read_text())
        if not c["completed"] or c["changed_inputs"]:
            raise ValueError("pilot incomplete")
        assets = external_assets(old)
        frozen_files(old, m["area"])
        step = next(p for p in old["files"] if Path(p).name.startswith("fastC."))
        bind_pilot(m, step)
        for arm in ("a1", "e2e"):
            if digest(root / arm / "valhist.root") != c["arms"][arm]["sha256"]:
                raise ValueError("pilot histogram drift")
        for path, sha in assets.items():
            if m["files"].get(path) != sha:
                raise ValueError("historical vs pilot assets differ")
        rows.append({"pilot": str(root), "manifest_sha256": digest(root / "manifest.json"),
                     "completion_sha256": digest(root / "completion.json"), "events": c["expected_generated_events"],
                     "external_assets": assets, "population_matches_historical": True})
    with dest.open("x") as stream:
        json.dump({"pilots": rows, "source_sha256": digest(__file__)}, stream, indent=2)
    print(f"certified {len(rows)} completed pilots", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pilots", nargs="+")
    p.add_argument("--out", required=True)
    run(**vars(p.parse_args()))
