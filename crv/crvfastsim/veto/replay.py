"""Export raw mixed full-step events for paired stock/fast response, without running either arm.

python3 -m crvfastsim.veto.replay --tables DIR --events 300 --seed 20261025 --out NEW.root

The ROOT file contains exact CrvStepsFromTable branch types and a dump/events TTree,
including stepless events and sampled intensity/pbtime. The JSON manifest hashes each
source parquet and the output and records source-code hashes. Raw times are not folded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from .catsteps import F32, read_meta
from .mixframes import MIXERS, PBI, PBTIMES, Pool, mix

IDS = {k: np.uint32 for k in ("run", "subrun", "event")}
STEP_TYPES = IDS | {k: np.int64 for k in ("bar", "sp")} | {k: np.float64 for k in (*F32, "t0", "t1")}
PART_TYPES = IDS | {k: np.int64 for k in ("key", "parent", "isprim")} | {"pdg": np.int32}
EVENT_TYPES = IDS | {"pbtime": np.float64, "intensity": np.float64}


def sha256(path):
    """Streaming hash without loading a whole Cat table in memory."""
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def replay_tables(steps, events):
    """Type and validate raw steps, remapping particle IDs densely within each event."""
    ev = events[list(EVENT_TYPES)].copy()
    if not len(ev) or not np.isfinite(ev.to_numpy()).all():
        raise ValueError("events must be nonempty and finite")
    if ev.duplicated(list(IDS)).any() or np.any(ev.intensity < 0):
        raise ValueError("duplicate event or negative intensity")
    for k in IDS:
        if np.any(ev[k] < 0) or np.any(ev[k] > np.iinfo(np.uint32).max) or np.any(ev[k] != np.floor(ev[k])):
            raise ValueError(f"invalid event ID: {k}")
    if np.any(ev.event >= 1 << 20) or ev.run.nunique() != 1:
        raise ValueError("fast replay requires one run and event IDs below 2**20")
    ev = ev.astype(EVENT_TYPES)
    if not len(steps):
        return {"steps": pd.DataFrame({k: pd.Series(dtype=v) for k, v in STEP_TYPES.items()}),
                "particles": pd.DataFrame({k: pd.Series(dtype=v) for k, v in PART_TYPES.items()}), "events": ev}
    st = steps.copy()
    invalid = {k: int((~np.isfinite(st[k].to_numpy())).sum()) for k in [*STEP_TYPES, "pdg"] if k not in ("t0", "t1")}
    invalid.update({k: int(np.isinf(st[k].to_numpy()).sum()) for k in ("t0", "t1")})
    invalid = {k: n for k, n in invalid.items() if n}
    if invalid:
        raise ValueError(f"nonfinite step or particle value: {invalid}")
    if np.any(st.sp < 0) or np.any(st.pdg == 0):
        raise ValueError("replay requires known SimParticle identities and PDG codes")
    for k in (*IDS, "bar", "sp", "pdg"):
        if np.any(st[k] != np.floor(st[k])):
            raise ValueError(f"nonintegral step identity: {k}")
    ev_index = pd.MultiIndex.from_frame(ev[list(IDS)])
    if not pd.MultiIndex.from_frame(st[list(IDS)]).isin(ev_index).all():
        raise ValueError("step event is absent from all-event table")
    keys = [*IDS, "sp"]
    particles = st[keys + ["pdg"]].drop_duplicates()
    if particles.duplicated(keys).any():
        raise ValueError("particle identity has inconsistent PDG codes")
    particles["key"] = particles.groupby(list(IDS), sort=False).cumcount() + 1
    st = st.merge(particles[keys + ["key"]], on=keys, how="left", sort=False, validate="many_to_one")
    st["sp"] = st.pop("key")
    particles["parent"], particles["isprim"] = -1, 1
    return {"steps": st[list(STEP_TYPES)].astype(STEP_TYPES),
            "particles": particles[list(PART_TYPES)].astype(PART_TYPES), "events": ev}


def write_replay(path, trees):
    """Create TTrees even for empty steps, never overwriting an existing sample."""
    with uproot.create(path) as f:
        for name, df in trees.items():
            f.mktree(f"dump/{name}", {k: df[k].dtype for k in df})
            if len(df):
                f[f"dump/{name}"].extend({k: df[k].to_numpy() for k in df})
    with uproot.open(path) as f:
        return {name: f[f"dump/{name}"].num_entries for name in trees}


def export(tables, events, seed, out):
    """Mix a recorded set of source tables once and save raw steps plus event conditions."""
    if events <= 0 or events >= 1 << 20:
        raise ValueError("events must be in [1, 2**20)")
    output = Path(out)
    sidecar = output.with_suffix(".json")
    if output.suffix != ".root" or output.exists() or sidecar.exists():
        raise ValueError("choose a new .root path with no existing ROOT or JSON output")
    sources, pools = {}, {}
    for name in MIXERS:
        files = sorted((Path(tables) / name).glob("*.parquet"))
        if not files:
            raise ValueError(f"missing Cat pool: {name}")
        sources[name] = [{"path": str(p.resolve()), "bytes": p.stat().st_size,
                          "sha256": sha256(p), "metadata": read_meta(p)} for p in files]
        if any(s["metadata"]["dataset"] != name for s in sources[name]):
            raise ValueError(f"wrong dataset in pool: {name}")
        pool = Pool(files)
        if pool.frames <= 0 or pool.stepped > pool.frames:
            raise ValueError(f"invalid frame denominator: {name}")
        if pool.stepped != sum(s["metadata"]["stepped_frames"] for s in sources[name]):
            raise ValueError(f"stepped frame count mismatch: {name}")
        if len(pool.steps) != sum(s["metadata"]["steps"] for s in sources[name]):
            raise ValueError(f"step count mismatch: {name}")
        if len(pool.steps) and (np.any(pool.steps.sp < 0) or np.any(pool.steps.sp >= 100000)):
            raise ValueError(f"particle keys exceed mixer's frame stride: {name}")
        pools[name] = pool
    st, _, _, info, ev = mix(pools, events, np.random.default_rng(seed), return_events=True)
    trees = replay_tables(st, ev)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = write_replay(output, trees)
    source_dir = Path(__file__).parent
    manifest = {"schema": "crv-mixed-replay-v1", "seed": seed, "events": events, "entries": counts,
                "run": 1430, "subrun": 0, "first_event": 1, "raw_times": True,
                "pbi": PBI, "pbtimes": PBTIMES.tolist(), "mixers": MIXERS, **info,
                "sources": sources, "output_sha256": sha256(output),
                "nan_time_steps": {k: int(trees["steps"][k].isna().sum()) for k in ("t0", "t1")},
                "time_policy": "NaN step times are preserved and counted; stock CrvPhotonGenerator and fast photon rows skip them. Infinite times are rejected.",
                "code_sha256": {n: sha256(source_dir / n) for n in ("replay.py", "mixframes.py", "catsteps.py")},
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "uproot": uproot.__version__},
                "scope": "Response diagnostic only; particle ancestry is synthetic, all table particles have parent=-1/isprim=1. Cat-library and mixer fidelity are not certified. Intensity is the floating draw used for Poisson mixing."}
    with sidecar.open("x") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--events", required=True, type=int)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    result = export(a.tables, a.events, a.seed, a.out)
    print(json.dumps({k: result[k] for k in ("schema", "seed", "entries", "output_sha256")}, indent=2))


if __name__ == "__main__":
    main()
