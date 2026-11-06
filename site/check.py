#!/usr/bin/env python3
"""Print the site variables a preset sets, whether each is set and whether its path is usable (standard library only).

    source site/gpvm.env && python3 site/check.py gpvm        # or eaf, slurm; no argument: every variable
Exit 0 when every variable the preset needs is set and every local path is readable (and writable where outputs go);
1 otherwise. Paths on the Slurm cluster are reported as remote and not checked from here.
"""
import json
import os
import sys

# name: (presets that need it, kind) with kind read (local, readable), write (local, writable), value, remote
VARIABLES = {
    "M2FS_CODE": ("eaf gpvm slurm", "read"),
    "M2FS_INPUTS": ("eaf gpvm", "read"),
    "MANIFEST": ("eaf gpvm", "read"),
    "M2FS_DATA": ("eaf gpvm slurm", "write"),
    "PYTHONPATH": ("eaf gpvm slurm", "value"),
    "CRVFS_DATA": ("eaf gpvm", "read"),
    "CRVFS_LOOKUP_DIR": ("eaf gpvm", "read"),
    "MU2E_SEARCH_PATH": ("eaf gpvm", "read"),
    "M2FS_VENV": ("", "read"),
    "FASTSIMKIT_ENDPOINT": ("slurm", "value"),
    "FASTSIMKIT_STATE_DIR": ("", "value"),
    "FASTSIMKIT_EAF_SHARED_PREFIX": ("", "read"),
    "EAF_TOKEN_FILE": ("", "secret"),
    "FASTSIMKIT_FNAL_HOST": ("slurm", "value"),
    "FASTSIMKIT_SLURM_HOST": ("slurm", "value"),
    "FASTSIMKIT_SLURM_ACCOUNT": ("slurm", "value"),
    "FASTSIMKIT_SCRATCH": ("slurm", "remote"),
    "FASTSIMKIT_SIF": ("slurm", "remote"),
    "FASTSIMKIT_WORKDIR": ("slurm", "remote"),
}
PRESETS = ("eaf", "gpvm", "slurm")


def state(name, kind):
    """(ok, text) for one variable; ok is None when nothing can be checked here."""
    value = os.environ.get(name)
    if not value:
        return False, "unset"
    if kind in ("value", "remote"):
        return None, value + ("  (remote path, not checked here)" if kind == "remote" else "")
    if kind == "secret":
        ok = os.path.isfile(value) and os.access(value, os.R_OK) and not os.stat(value).st_mode & 0o077
        return ok, value + ("  (readable, mode 600)" if ok else "  (missing, unreadable or readable by others)")
    if not os.path.exists(value):
        return False, value + "  (missing)"
    mode = os.R_OK | (os.W_OK if kind == "write" else 0) | (os.X_OK if os.path.isdir(value) else 0)
    ok = os.access(value, mode)
    return ok, value + ("" if ok else "  (not %s)" % ("writable" if kind == "write" else "readable"))


def release(path):
    """Release name and file count of a manifest, or None."""
    try:
        with open(path) as f:
            m = json.load(f)
        return f"{m.get('release')} ({len(m.get('files', []))} files)"
    except (OSError, ValueError):
        return None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    preset = argv[0] if argv else None
    if preset not in (None,) + PRESETS:
        print(f"usage: check.py [{'|'.join(PRESETS)}]", file=sys.stderr)
        return 2
    bad = 0
    for name, (needed_by, kind) in VARIABLES.items():
        needed = preset is not None and preset in needed_by.split()
        ok, text = state(name, kind)
        mark = {True: "ok", None: "set", False: "MISSING" if needed else "-"}[ok]
        if ok is False and not needed and text == "unset":
            text = "unset (optional here)"
        print(f"{mark:8s} {name:30s} {text}")
        bad += ok is False and needed
    name = release(os.environ["MANIFEST"]) if os.environ.get("MANIFEST") else None
    if name:
        print(f"release  {name}")
    if bad:
        print(f"{bad} variable(s) the {preset} preset needs are unset or unusable", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
