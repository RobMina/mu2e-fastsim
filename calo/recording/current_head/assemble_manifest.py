"""Install reviewed input-manifest sources into the isolated Offline clone; never build or run."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
BASE_ROOT_CMAKE = "069bda362f01658c9abf07c6aaab21d9d9c1573c44361e4ee8afabd962a47b92"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(root):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest = json.loads((HERE / "source-manifest.json").read_text())
    if head != manifest["base_revision"]:
        raise ValueError("unexpected base revision")
    for name, sha in manifest["patched_sources"].items():
        if digest(root / name) != sha:
            raise ValueError("prepared source drift: " + name)
    cmake = root / "CMakeLists.txt"
    if digest(cmake) != BASE_ROOT_CMAKE:
        raise ValueError("top-level CMake drift")
    package = root / "CaloRecording"
    if package.exists():
        raise ValueError("CaloRecording package already exists")
    review = HERE.parent / "manifest"
    pins = json.loads((review / "source_manifest.json").read_text())
    for name, sha in pins["proposal_files"].items():
        if digest(review / name) != sha:
            raise ValueError("manifest review artifact drift: " + name)
    names = ["CMakeLists.txt", "src/SConscript", "src/ManifestMetadata.hh",
             "src/CaloInputEventManifest_module.cc", "src/CaloIncidentManifest_module.cc",
             "fcl/manifest_ce_review.fcl"]
    for name in names:
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(review / name, path)
    text = cmake.read_text()
    anchor = "add_subdirectory(CaloMC)"
    if text.count(anchor) != 1:
        raise ValueError("CMake integration anchor not unique")
    cmake.write_text(text.replace(anchor, anchor + "\nadd_subdirectory(CaloRecording)"))
    manifest["patched_sources"]["CMakeLists.txt"] = digest(cmake)
    manifest["base_sources"]["CMakeLists.txt"] = BASE_ROOT_CMAKE
    for name in names:
        manifest["patched_sources"]["CaloRecording/" + name] = digest(package / name)
    inventory = json.loads((HERE.parent / "inventory/source-manifest.json").read_text())
    for name, sha in inventory["patched"].items():
        if digest(root / name) != sha:
            raise ValueError("inventory integration drift: " + name)
        manifest["patched_sources"][name] = sha
    manifest["inventory_patch_sha256"] = inventory["patch_sha256"]
    manifest["manifest_review_pins_sha256"] = digest(review / "source_manifest.json")
    manifest["status"] = "current-head sources assembled; user build and native validation pending"
    (HERE / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    subprocess.run(["git", "diff", "--check"], cwd=root, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", required=True, type=Path)
    assemble(parser.parse_args().offline.resolve())
