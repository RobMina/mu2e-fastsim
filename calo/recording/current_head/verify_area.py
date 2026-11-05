"""Verify hash-pinned prepared or exactly checkpointed sources without setup/build/transport."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def verify(root):
    pins = json.loads((HERE / "source-manifest.json").read_text())
    base = pins["base_revision"]
    actual = git(root, "rev-parse", "HEAD")
    if actual != base:
        subprocess.run(["git", "merge-base", "--is-ancestor", base, actual], cwd=root, check=True)
        changes = git(root, "diff", "--no-renames", "--name-only", base, actual).splitlines()
        if set(changes) != set(pins["patched_sources"]):
            raise ValueError("checkpoint tree differs from the exact prepared changeset")
        for name, expected in pins["patched_sources"].items():
            entry = git(root, "ls-tree", actual, "--", name).split()
            if len(entry) < 4 or entry[:2] != ["100644", "blob"]:
                raise ValueError("checkpoint file type/mode changed: " + name)
            blob = subprocess.check_output(["git", "show", f"{actual}:{name}"], cwd=root)
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError("checkpoint source changed: " + name)
        if git(root, "diff", "--name-only", "HEAD"):
            raise ValueError("committed checkpoint has working-tree changes")
        if git(root, "ls-files", "--others", "--exclude-standard"):
            raise ValueError("committed checkpoint has unexpected untracked files")
    for name, expected in pins["patched_sources"].items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("prepared source changed: " + name)
    if git(root, "diff", "--cached", "--name-only"):
        raise ValueError("source verification expects no staged changes")
    if git(root, "diff", "--name-only", "--diff-filter=D"):
        raise ValueError("unintended source deletion")
    subprocess.run(["git", "diff", "--check"], cwd=root, check=True)
    print("Prepared-source hashes, exact base/checkpoint tree, index/deletion and whitespace checks passed.")
    print("No compile/link, native inventory, manifest or transport validation implied.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", required=True, type=Path)
    verify(parser.parse_args().offline.resolve())
