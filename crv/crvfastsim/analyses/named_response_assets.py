"""Read-only asset identity guard for serial named-response diagnostics.

Run before and after a replay and compare stdout byte-for-byte::

    python -m crvfastsim.analyses.named_response_assets \
        --run-config run2_v10 --offline /explicit/Offline

The explicit root is an assertion, not a resolver override: both counters and
waveform must resolve there using geometry.find_counters, including its cwd
precedence. A differently named root is supported only if the actual resolver
can reach it (for example through an Offline symlink). No fallback is invented.
All configured optical tables are included, regardless of event occupancy.

This is diagnostic asset identity only, not a provenance certificate, proof of
execution, or an atomic filesystem snapshot. Two reads detect observed changes,
not changes restored between reads. Use in a serial process with stable assets;
CRVFS_NO_FETCH is temporarily set to forbid resolver downloads/cache writes.
Optical payloads, including photonMap.root, are hashed as opaque bytes, not parsed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat

from .. import geometry, runconfig
from ..reco.waveform import WaveformConfig
from ..response import lookup


RUN_CONFIGS = ("run1a_v01", "run2_v10")


@contextmanager
def _no_fetch():
    previous = os.environ.get("CRVFS_NO_FETCH")
    os.environ["CRVFS_NO_FETCH"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CRVFS_NO_FETCH", None)
        else:
            os.environ["CRVFS_NO_FETCH"] = previous


def _regular_path(path):
    resolved = Path(path).resolve(strict=True)
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError(f"asset is not a regular file: {resolved}")
    return resolved


def _offline_asset(name, configured, root):
    if not configured.startswith("Offline/"):
        raise ValueError(f"{name}: expected an Offline-relative configured asset: {configured}")
    expected = _regular_path(root / configured[len("Offline/"):])
    actual = _regular_path(geometry.find_counters(configured))
    if actual != expected:
        raise ValueError(f"{name}: resolver selected {actual}; expected explicit Offline asset {expected}")
    return actual


def _paths(rc, offline):
    root = Path(offline).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"explicit Offline root is not a directory: {root}")
    paths = {
        "counters": _offline_asset("counters", rc.counters, root),
        "waveform": _offline_asset("waveform", WaveformConfig().single_pe_file, root),
    }
    directory = Path(lookup.lookup_dir())
    # Match lookup.load's filename rule, not a glob or a hand-maintained list.
    for table in sorted({sector.table for sector in rc.response().sectors.values()}):
        filename = table if table.startswith("LookupTable") else f"LookupTable_{table}"
        paths[filename] = _regular_path(directory / filename)
    # Match sipm.photon_map's default; never call its ROOT reader here.
    paths["photonMap.root"] = _regular_path(directory / "photonMap.root")
    return paths


def _stat_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _record(path):
    if _regular_path(path) != path:
        raise ValueError(f"asset path changed while taking inventory: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"asset is not a regular file: {path}")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (size != before.st_size or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(path.stat())):
        raise ValueError(f"asset changed while taking inventory: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "size": size}


def _snapshot(rc, offline):
    return {name: _record(path) for name, path in sorted(_paths(rc, offline).items())}


def inventory(run_config, offline):
    """Return JSON-safe identities of actual resolved assets; never write assets.

    Refuse missing/nonregular files, a counters/waveform resolver mismatch, or
    changes observed during the final re-resolution and rehash. Does not alter
    MU2E_SEARCH_PATH, OFFLINE_ROOTS, cwd, waveform defaults, or optical selection.
    """
    if run_config not in RUN_CONFIGS:
        raise ValueError(f"unsupported named response run config: {run_config}")
    rc = runconfig.get(run_config)
    with _no_fetch():
        assets = _snapshot(rc, offline)
        if _snapshot(rc, offline) != assets:
            raise ValueError("assets changed while taking inventory")
    return {"schema_version": 1, "run_config": rc.name,
            "scope": "diagnostic_asset_identity_only", "assets": assets}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, choices=RUN_CONFIGS)
    parser.add_argument("--offline", required=True, help="explicit Offline payload root (not a search parent)")
    args = parser.parse_args(argv)
    try:
        result = inventory(args.run_config, args.offline)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
