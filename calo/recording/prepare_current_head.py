"""Prepare reviewed recorder drafts in an explicitly selected clean current-head clone; never build."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
HEAD = "b24d4163e5e8f5ed2b5f72fccb25e4980fec64e2"
REVIEWED_DRIFT = {
    "Mu2eG4/inc/Mu2eG4Config.hh": "962a32bb7cd7f3ed7258122730bb8b70dd1ebc68f81b34e30a93119211043bbe",
    "Mu2eG4/CMakeLists.txt": "b698e1d65b30ddb10b1ad933804a2625586c1024d0bd35a80e9c4da8dab13151",
    "Mu2eG4/src/SConscript": "f9b358138b63b64dca99552cc8a000f03d5bc9ff8b7fd307ace4378ba2f224c2",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, *, cwd, text=None):
    return subprocess.run(command, cwd=cwd, input=text, text=True, capture_output=True, check=True).stdout


def require_sources(root, expected):
    for name, digest in expected.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or sha(root / name) != digest:
            raise ValueError("unreviewed source: " + name)


def prepare(root, out):
    if out.exists():
        raise ValueError("artifact output must be new")
    if run(["git", "rev-parse", "HEAD"], cwd=root).strip() != HEAD:
        raise ValueError("explicitly reviewed current-head revision required")
    if run(["git", "status", "--porcelain"], cwd=root).strip():
        raise ValueError("isolated source clone must be clean")
    boundary_manifest = json.loads((HERE / "boundary/source-manifest.json").read_text())
    compression_manifest = json.loads((HERE / "compression/source_manifest.json").read_text())
    expected = dict(boundary_manifest["sources"])
    expected.update(compression_manifest["files"])
    expected.update(REVIEWED_DRIFT)
    expected[".muse"] = "99dddc0ed075a76effd17ac67f2f1c4b395f4657746900b14226683f2b220325"
    require_sources(root, expected)
    compression = HERE / "compression/membership.patch"
    if sha(compression) != compression_manifest["proposal_artifacts"]["membership.patch"]:
        raise ValueError("membership artifact drift")
    spec = importlib.util.spec_from_file_location("boundary_recipe", HERE / "boundary/make_patch.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    boundary_patch, generated = recipe.build(root)
    for line in boundary_patch.splitlines():
        if line.startswith("-") and not line.startswith("---") and line != "-    _spHelper()":
            raise ValueError("unexpected stock removal")
    patches = {"boundary-current-head.patch": boundary_patch, "membership-current-head.patch": compression.read_text()}
    for text in patches.values():
        output = run(["patch", "--dry-run", "--batch", "--fuzz=0", "-p1"], cwd=root, text=text)
        if "offset" in output.lower() or "fuzz" in output.lower():
            raise ValueError("non-exact patch")
    out.mkdir(parents=True)
    for name, text in patches.items():
        (out / name).write_text(text)
        output = run(["patch", "--batch", "--fuzz=0", "-p1"], cwd=root, text=text)
        if "offset" in output.lower() or "fuzz" in output.lower():
            raise ValueError("non-exact applied patch; inspect partial preparation")
    final = dict(generated["patched"])
    final["CaloMC/src/CaloShowerStepMaker_module.cc"] = compression_manifest["patched_producer_sha256"]
    require_sources(root, final)
    run(["git", "diff", "--check"], cwd=root)
    manifest = {"base_revision": HEAD, "environment_recommendation": "p110", "base_sources": expected,
                "patched_sources": final, "patches": {name: sha(out / name) for name in patches},
                "status": "prepared source only; not compiled or runtime validated", "build_performed": False}
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--artifact-out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.offline.resolve(), args.artifact_out.resolve())


if __name__ == "__main__":
    main()
