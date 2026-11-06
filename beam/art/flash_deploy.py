"""Copy a selected source snapshot to an existing local mirror directory.

This prepares files for a live mirror; it performs no network transfer or Git
operation. Public binary tables excluded by the mirror are reused on the worker.
"""
import argparse
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core", type=Path, required=True)
    p.add_argument("--beam", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("Use a fresh snapshot directory")
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "*.npz")
    shutil.copytree(a.core / "mu2efastsim", a.out / "core/mu2efastsim", ignore=ignored)
    shutil.copytree(a.beam / "beamfastsim", a.out / "beam/beamfastsim", ignore=ignored)
    shutil.copytree(a.beam / "tests", a.out / "beam/tests", ignore=ignored)
    for name in ("conftest.py", "test_eslab_photons_jax.py"):
        dest = a.out / "core/tests" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.core / "tests" / name, dest)


if __name__ == "__main__":
    main()
