"""Fail-closed population and provenance guards shared by descendant diagnostics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import tables
from . import steplayers as S
from .descendant_audit import digest
from .pilot import tree_inventory


def unique_shards(spec):
    selected = S.shards(spec)
    if not selected or len(selected) != len(set(selected)) or min(selected) < 1:
        raise ValueError("shards must be positive and nonrepeated")
    return selected


def population(gen, steps, refparts, donor_ids, seen):
    """Require disjoint full generator identities and subset membership for both physics arms."""
    ids = pd.MultiIndex.from_frame(gen[tables.KEY].drop_duplicates())
    if not len(ids) or not ids.is_unique:
        raise ValueError("empty or duplicate generated population")
    sr = set(gen[tables.KEY[:2]].drop_duplicates().itertuples(index=False, name=None))
    if sr.intersection(seen):
        raise ValueError("overlapping evaluation run/subrun")
    if ids.isin(donor_ids).any():
        raise ValueError("evaluation overlaps training donors")
    for name, frame in (("transport", steps), ("reference", refparts)):
        if not pd.MultiIndex.from_frame(frame[tables.KEY]).isin(ids).all():
            raise ValueError(name + " outside generated population")
    seen.update(sr)
    return ids


def bind_pilot(manifest, frozen_steps):
    actual = tree_inventory(frozen_steps, "dump/gen")
    expected = manifest["generated"]
    if any(actual[k] != expected[k] for k in ("unique_events", "event_keys_sha256")):
        raise ValueError("requested shard differs from pilot population")


def external_assets(old):
    result = {}
    for path in old["asset_paths"]:
        sha = digest(path)
        if old["files"].get(path) != sha:
            raise ValueError("historical external asset changed: " + path)
        result[path] = sha
    return result


def bind_reports(expected, replicas, placement):
    root = Path(expected["cache"])
    if str(root) != replicas["cache"] or expected["cache_sha256"] != replicas["cache_sha256"]:
        raise ValueError("different expectation/replica cache")
    cache = json.loads((root / "cache.json").read_text())
    library = cache["library"]
    if placement["hashes"].get(library) != cache["hashes"][library]:
        raise ValueError("different placement library")
    es = [r["shard"] for r in expected["shards"]]
    if es != [r["shard"] for r in replicas["shards"]] or es != [r["shard"] for r in placement["shards"]]:
        raise ValueError("different report shards")
    if len(es) != len(set(es)):
        raise ValueError("duplicate report shards")
    for path, sha in expected["input_sha256"].items():
        if replicas["hashes"].get(path) != sha:
            raise ValueError("expectation/replica input mismatch")
    for row in placement["shards"]:
        path = row["transport"]
        if expected["input_sha256"].get(path) != placement["hashes"].get(path):
            raise ValueError("placement transport mismatch")
    if replicas["retained_seed_rule"] != "[shard,1000]" or placement["seed_rule"] != "[shard,1000]":
        raise ValueError("retained seed rules differ")
