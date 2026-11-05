"""Stage a release-specific geometry include tree, reusing byte-identical existing files."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from mu2efastsim import cvmfsget


def stage(out, release, geom, reuse=None):
    """Write only into out; links to reused files are read-only inputs, never overwritten."""
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    repo = cvmfsget.Repo("mu2e.opensciencegrid.org")
    includes = re.compile(r'^\s*#include\s+"([^"]+)"', re.M)
    todo, seen, manifest = [geom], set(), []
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        if not path.startswith("Offline/") or ".." in Path(path).parts:
            raise ValueError(f"unsupported geometry include: {path}")
        seen.add(path)
        data = repo.read(f"{release.rstrip('/')}/{path}")
        sha = hashlib.sha256(data).hexdigest()
        dest = root / path
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(f"refusing to overwrite geometry input {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        old = Path(reuse) / path if reuse else None
        reused = old is not None and old.is_file() and hashlib.sha256(old.read_bytes()).hexdigest() == sha
        if reused:
            dest.symlink_to(old.resolve())
        else:
            dest.write_bytes(data)
        manifest.append(dict(file=path, sha256=sha, reused=bool(reused)))
        text = re.sub(r"//[^\n]*", "", data.decode())
        todo.extend(includes.findall(text))
    report = dict(release=release, geom=geom, files=manifest)
    (root / "geometry_manifest.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", required=True, help="CVMFS release root, e.g. /Musings/Offline/RELEASE")
    ap.add_argument("--geom", required=True)
    ap.add_argument("--reuse", help="existing staged Offline root (never modified)")
    ap.add_argument("--out", required=True, help="new output root (no existing files overwritten)")
    a = ap.parse_args(argv)
    r = stage(a.out, a.release, a.geom, a.reuse)
    print(json.dumps(dict(files=len(r["files"]), reused=sum(x["reused"] for x in r["files"]),
                          release=a.release, geom=a.geom)))


if __name__ == "__main__":
    main()
