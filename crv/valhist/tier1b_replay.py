"""Prepare a new A1/E2E replay of retained steps in a hash-verified frozen stock area.

The historical pilot's successful completion and exact area hashes authorize reuse;
this never compiles or rewrites an area, baseline, step table or library.
"""
import argparse
import json
from pathlib import Path

from . import pilot
from .descendant_audit import digest


def frozen_files(manifest, area):
    """Bind all recorded files under the stock area, rejecting missing/stale assets."""
    prefix = str(Path(area).resolve()) + "/"
    hashes = {p: sha for p, sha in manifest["files"].items() if p.startswith(prefix)}
    if not hashes or not any(p.endswith(".so") for p in hashes):
        raise ValueError("frozen area has no recorded binaries")
    drift = [p for p, sha in hashes.items() if not Path(p).is_file() or digest(p) != sha]
    if drift:
        raise ValueError(f"frozen stock area changed: {drift}")
    return hashes


def prepare(steps, out, historical, core, calib, deviation, seed, label="retained strict CCP"):
    out, historical = Path(out).resolve(), Path(historical).resolve()
    if out.exists():
        raise FileExistsError(out)
    old = json.loads((historical / "manifest.json").read_text())
    completion = json.loads((historical / "completion.json").read_text())
    if not completion.get("completed") or completion.get("changed_inputs"):
        raise ValueError("historical stock pilot not certified complete")
    area = Path(old["area"])
    from .descendant_inputs import external_assets
    hashes = {**frozen_files(old, area), **external_assets(old)}
    steps = Path(steps).resolve()
    inv = pilot.tree_inventory(steps, "dump/gen")
    expected = old["generated"]
    if inv["unique_events"] != expected["unique_events"] or inv["event_keys_sha256"] != expected["event_keys_sha256"]:
        raise ValueError("retained steps do not match historical full generated population")
    code = Path(__file__).resolve().parents[1]
    files = [steps, Path(calib), Path(deviation), historical / "manifest.json", historical / "completion.json"]
    files += [Path(p) for p in old["asset_paths"]]
    for package in (code / "crvfastsim", code / "valhist", Path(core) / "mu2efastsim"):
        files += sorted(package.rglob("*.py"))
    if not (Path(core) / "mu2efastsim/__init__.py").is_file():
        raise FileNotFoundError(core)
    hashes.update({str(p): digest(p) for p in files})
    driver = code / "scripts/run_tier1b_replay.sh"
    hashes[str(driver)] = digest(driver)
    manifest = {"area": str(area), "backing": str((area / "backing").resolve()), "files": hashes,
                "steps": str(steps), "seed": seed, "calib": str(calib), "deviation": str(deviation),
                "generated": inv, "gen_art": old["gen_art"], "historical": str(historical),
                "scope": f"{label} steps, current Python response, unchanged frozen stock response"}
    out.mkdir(parents=True)
    for name in ("logs", "tables", "a1", "e2e"):
        (out / name).mkdir()
    for arm, base in (("a1", "valhist_faststeps.fcl"), ("e2e", "valhist_fromtable.fcl")):
        text = f'#include "CrvFastSim/fcl/{base}"\n'
        if arm == "a1":
            text += f'physics.producers.fastCrvSteps.tableFile : "{steps}"\n'
        else:
            for name in ("CrvDigi", "CrvRecoPulses", "CrvCoincidenceClusterFinder"):
                text += f'physics.producers.{name}.tableFile : "{out / "tables/chain.root"}"\n'
        text += 'physics.analyzers.crvdump : { module_type : CrvRecoTableDumper mc : false }\n'
        text += 'physics.ValPath : [ Validation, crvDigiDQM, crvRecoDQM, dqmCrv, crvdump ]\n'
        fcl = out / f"{arm}.fcl"
        fcl.write_text(text, encoding="ascii")
        hashes[str(fcl)] = digest(fcl)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared {out}: {inv['unique_events']} events; frozen-area hashes unchanged", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("steps", "out", "historical", "core", "calib", "deviation"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--label", default="retained strict CCP")
    check = sub.add_parser("verify")
    check.add_argument("out")
    a = vars(p.parse_args())
    command = a.pop("command")
    (prepare if command == "prepare" else pilot.verify)(**a)


if __name__ == "__main__":
    main()
