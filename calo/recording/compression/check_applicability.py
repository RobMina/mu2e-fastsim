"""Check exact source hashes and patch applicability without modifying the source tree."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "source_manifest.json").read_text())


def check_hashes(root):
    failures = []
    for relative, expected in MANIFEST["files"].items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if actual != expected:
            failures.append(f"{relative}: expected {expected}, found {actual}")
    if failures:
        raise ValueError("Source snapshot mismatch:\n" + "\n".join(failures))


def apply_copy(root, destination):
    check_hashes(root)
    for relative in MANIFEST["files"]:
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, output)
    patch = (HERE / "membership.patch").read_text()
    for mode in (["--dry-run"], []):
        result = subprocess.run(
            ["patch", "--batch", "--fuzz=0", "-p1", *mode],
            cwd=destination, input=patch, text=True, capture_output=True, check=True,
        )
        if "offset" in result.stdout.lower() or "fuzz" in result.stdout.lower():
            raise ValueError("Non-exact patch application: " + result.stdout)
    return destination / "CaloMC/src/CaloShowerStepMaker_module.cc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Read-only Offline source root to check")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix=".applicability-", dir=HERE) as temporary:
        apply_copy(args.source, Path(temporary))
    print("Exact snapshot hashes and temporary-copy patch application passed; not a build check.")


if __name__ == "__main__":
    main()
